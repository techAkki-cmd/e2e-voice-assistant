package ai.e2e.orchestrator.websocket;

import static ai.e2e.orchestrator.config.RabbitMQConfig.AUDIO_INCOMING_RAW_QUEUE;
import static ai.e2e.orchestrator.config.RabbitMQConfig.CONTROL_SIGNALS_EXCHANGE;

import com.rabbitmq.client.AMQP;
import io.opentelemetry.api.trace.SpanContext;
import io.opentelemetry.api.trace.TraceFlags;
import io.opentelemetry.api.trace.TraceState;
import java.nio.charset.StandardCharsets;
import java.net.URLDecoder;
import java.security.SecureRandom;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.socket.WebSocketHandler;
import org.springframework.web.reactive.socket.WebSocketMessage;
import org.springframework.web.reactive.socket.WebSocketSession;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Sinks;
import reactor.rabbitmq.OutboundMessage;
import reactor.rabbitmq.QueueSpecification;
import reactor.rabbitmq.Sender;

@Component
public class AudioStreamHandler implements WebSocketHandler {

    private static final Logger LOGGER = LoggerFactory.getLogger(AudioStreamHandler.class);
    private static final long KILL_DEBOUNCE_MILLIS = 750;
    private static final int MAX_USER_ID_LENGTH = 128;
    private static final SecureRandom SECURE_RANDOM = new SecureRandom();
    private static final ConcurrentHashMap<String, WebSocketSession> sessionRegistry = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<String, Sinks.Many<byte[]>> outboundAudioSinks = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<String, Long> lastKillSignalAtMillis = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<String, String> sessionTraceparents = new ConcurrentHashMap<>();

    private final Sender sender;

    public AudioStreamHandler(Sender sender) {
        this.sender = sender;
    }

    @Override
    public Mono<Void> handle(WebSocketSession session) {
        String userId = resolveUserId(session);
        Sinks.Many<byte[]> outboundAudioSink = Sinks.many().unicast().onBackpressureBuffer();
        Sinks.Many<byte[]> previousSink = outboundAudioSinks.put(userId, outboundAudioSink);
        sessionRegistry.put(userId, session);
        sessionTraceparents.remove(userId);

        if (previousSink != null) {
            previousSink.tryEmitComplete();
            LOGGER.info("Replacing active Audio WebSocket route for user_id={}", userId);
        }

        Flux<OutboundMessage> audioMessages = session.receive()
                .flatMap(message -> {
                    if (message.getType() == WebSocketMessage.Type.BINARY) {
                        return Mono.just(toOutboundMessage(userId, message));
                    }

                    if (message.getType() == WebSocketMessage.Type.TEXT) {
                        return handleControlMessage(userId, message).then(Mono.empty());
                    }

                    return Mono.empty();
                });

        Mono<Void> inboundAudio = sender.declareQueue(QueueSpecification.queue(AUDIO_INCOMING_RAW_QUEUE).durable(true))
                .then(sender.send(audioMessages))
                .onErrorResume(throwable -> {
                    LOGGER.debug(
                            "Audio WebSocket session {} closed while streaming audio: {}",
                            userId,
                            throwable.getMessage()
                    );
                    return Mono.empty();
                })
                .doFinally(signalType -> outboundAudioSink.tryEmitComplete());

        Mono<Void> outboundAudio = session.send(
                outboundAudioSink.asFlux()
                        .map(audio -> session.binaryMessage(bufferFactory -> bufferFactory.wrap(audio)))
        );

        return Mono.when(inboundAudio, outboundAudio)
                .doFinally(signalType -> {
                    cleanupSession(userId, outboundAudioSink);
                    LOGGER.debug(
                            "Audio WebSocket route for user_id={} finished with signal {}",
                            userId,
                            signalType
                    );
                });
    }

    public static boolean sendAudioToSession(String correlationId, byte[] audio) {
        if (correlationId == null || correlationId.isBlank()) {
            LOGGER.debug("Dropping outbound audio without correlation ID");
            return false;
        }

        WebSocketSession session = sessionRegistry.get(correlationId);
        Sinks.Many<byte[]> outboundAudioSink = outboundAudioSinks.get(correlationId);

        if (session == null || outboundAudioSink == null || !session.isOpen()) {
            cleanupSession(correlationId);
            LOGGER.debug("Dropping outbound audio for missing or closed WebSocket session {}", correlationId);
            return false;
        }

        Sinks.EmitResult emitResult = outboundAudioSink.tryEmitNext(audio.clone());
        if (emitResult.isFailure()) {
            LOGGER.debug("Failed to route outbound audio to session {}: {}", correlationId, emitResult);
            return false;
        }

        return true;
    }

    private static void cleanupSession(String sessionId) {
        sessionRegistry.remove(sessionId);
        outboundAudioSinks.remove(sessionId);
        lastKillSignalAtMillis.remove(sessionId);
        sessionTraceparents.remove(sessionId);
    }

    private static void cleanupSession(String userId, Sinks.Many<byte[]> expectedSink) {
        if (outboundAudioSinks.remove(userId, expectedSink)) {
            sessionRegistry.remove(userId);
            lastKillSignalAtMillis.remove(userId);
            sessionTraceparents.remove(userId);
        }
    }

    private Mono<Void> handleControlMessage(String userId, WebSocketMessage message) {
        String payload = message.getPayloadAsText();
        if (!payload.contains("barge_in")) {
            LOGGER.debug("Ignoring unsupported WebSocket control message for user_id={}: {}", userId, payload);
            return Mono.empty();
        }

        return publishBargeInSignal(userId);
    }

    private Mono<Void> publishBargeInSignal(String userId) {
        long now = System.currentTimeMillis();
        Long lastKillAt = lastKillSignalAtMillis.get(userId);
        if (lastKillAt != null && now - lastKillAt < KILL_DEBOUNCE_MILLIS) {
            return Mono.empty();
        }

        lastKillSignalAtMillis.put(userId, now);
        String traceparent = traceparentFor(userId);
        String controlJson = "{\"action\":\"kill\",\"correlation_id\":\"" + userId + "\"}";
        Map<String, Object> headers = traceHeaders(userId, traceparent);
        AMQP.BasicProperties properties = new AMQP.BasicProperties.Builder()
                .contentType("application/json")
                .correlationId(userId)
                .deliveryMode(2)
                .headers(headers)
                .build();

        LOGGER.info("Publishing barge-in kill signal for user_id={} traceparent={}", userId, traceparent);
        return sender.send(Mono.just(new OutboundMessage(
                        CONTROL_SIGNALS_EXCHANGE,
                        "",
                        properties,
                        controlJson.getBytes(StandardCharsets.UTF_8)
                )))
                .then();
    }

    private OutboundMessage toOutboundMessage(String userId, WebSocketMessage message) {
        DataBuffer payload = message.getPayload();
        byte[] audioChunk = new byte[payload.readableByteCount()];
        payload.read(audioChunk);
        String traceparent = traceparentFor(userId);

        AMQP.BasicProperties properties = new AMQP.BasicProperties.Builder()
                .contentType("application/octet-stream")
                .correlationId(userId)
                .deliveryMode(2)
                .headers(traceHeaders(userId, traceparent))
                .build();

        return new OutboundMessage("", AUDIO_INCOMING_RAW_QUEUE, properties, audioChunk);
    }

    private static String resolveUserId(WebSocketSession session) {
        String userId = parseQueryParams(session.getHandshakeInfo().getUri().getRawQuery()).get("user_id");
        if (userId != null) {
            userId = userId.trim();
            if (!userId.isBlank()) {
                return userId.length() > MAX_USER_ID_LENGTH ? userId.substring(0, MAX_USER_ID_LENGTH) : userId;
            }
        }

        return session.getId();
    }

    private static Map<String, String> parseQueryParams(String rawQuery) {
        Map<String, String> params = new HashMap<>();
        if (rawQuery == null || rawQuery.isBlank()) {
            return params;
        }

        String[] pairs = rawQuery.split("&");
        for (String pair : pairs) {
            int equalsIndex = pair.indexOf('=');
            String key = equalsIndex >= 0 ? pair.substring(0, equalsIndex) : pair;
            String value = equalsIndex >= 0 ? pair.substring(equalsIndex + 1) : "";
            params.put(urlDecode(key), urlDecode(value));
        }

        return params;
    }

    private static String urlDecode(String value) {
        try {
            return URLDecoder.decode(value, StandardCharsets.UTF_8);
        } catch (IllegalArgumentException exception) {
            return "";
        }
    }

    private static String traceparentFor(String userId) {
        return sessionTraceparents.computeIfAbsent(userId, ignored -> newTraceparent());
    }

    private static String newTraceparent() {
        String traceId = randomHex(16);
        String spanId = randomHex(8);
        SpanContext.createFromRemoteParent(
                traceId,
                spanId,
                TraceFlags.getSampled(),
                TraceState.getDefault()
        );
        return "00-" + traceId + "-" + spanId + "-01";
    }

    private static String randomHex(int byteCount) {
        byte[] bytes = new byte[byteCount];
        StringBuilder hex = new StringBuilder(byteCount * 2);
        do {
            SECURE_RANDOM.nextBytes(bytes);
            hex.setLength(0);
            for (byte value : bytes) {
                hex.append(String.format("%02x", value));
            }
        } while (hex.toString().chars().allMatch(character -> character == '0'));
        return hex.toString();
    }

    private static Map<String, Object> traceHeaders(String userId, String traceparent) {
        Map<String, Object> headers = new HashMap<>();
        headers.put("traceparent", traceparent);
        headers.put("user_id", userId);
        return headers;
    }
}
