package ai.e2e.orchestrator.websocket;

import static ai.e2e.orchestrator.config.RabbitMQConfig.AUDIO_INCOMING_RAW_QUEUE;
import static ai.e2e.orchestrator.config.RabbitMQConfig.CONTROL_SIGNALS_EXCHANGE;

import com.rabbitmq.client.AMQP;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.core.io.buffer.DataBufferUtils;
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
    private static final int OUTPUT_SAMPLE_RATE = 44_100;
    private static final int PCM16_BYTES_PER_SAMPLE = 2;
    private static final long PLAYBACK_GRACE_MILLIS = 250;
    private static final long KILL_DEBOUNCE_MILLIS = 750;
    private static final ConcurrentHashMap<String, WebSocketSession> sessionRegistry = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<String, Sinks.Many<byte[]>> outboundAudioSinks = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<String, Long> playbackActiveUntilMillis = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<String, Long> lastKillSignalAtMillis = new ConcurrentHashMap<>();

    private final Sender sender;

    public AudioStreamHandler(Sender sender) {
        this.sender = sender;
    }

    @Override
    public Mono<Void> handle(WebSocketSession session) {
        String sessionId = session.getId();
        Sinks.Many<byte[]> outboundAudioSink = Sinks.many().unicast().onBackpressureBuffer();

        sessionRegistry.put(sessionId, session);
        outboundAudioSinks.put(sessionId, outboundAudioSink);

        Flux<OutboundMessage> audioMessages = session.receive()
                .filter(message -> message.getType() == WebSocketMessage.Type.BINARY)
                .flatMap(message -> maybePublishBargeInSignal(sessionId)
                        .thenReturn(toOutboundMessage(sessionId, message)));

        Mono<Void> inboundAudio = sender.declareQueue(QueueSpecification.queue(AUDIO_INCOMING_RAW_QUEUE).durable(true))
                .then(sender.send(audioMessages))
                .onErrorResume(throwable -> {
                    LOGGER.debug(
                            "Audio WebSocket session {} closed while streaming audio: {}",
                            sessionId,
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
                    cleanupSession(sessionId);
                    LOGGER.debug(
                            "Audio WebSocket session {} finished with signal {}",
                            sessionId,
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

        markPlaybackActive(correlationId, audio.length);
        return true;
    }

    private static void cleanupSession(String sessionId) {
        sessionRegistry.remove(sessionId);
        outboundAudioSinks.remove(sessionId);
        playbackActiveUntilMillis.remove(sessionId);
        lastKillSignalAtMillis.remove(sessionId);
    }

    private static void markPlaybackActive(String correlationId, int audioByteLength) {
        long now = System.currentTimeMillis();
        long durationMillis = Math.max(
                20,
                Math.round((audioByteLength / (double) PCM16_BYTES_PER_SAMPLE) * 1000.0 / OUTPUT_SAMPLE_RATE)
        );

        playbackActiveUntilMillis.compute(correlationId, (key, existingUntil) -> {
            long base = existingUntil != null && existingUntil > now ? existingUntil : now;
            return base + durationMillis + PLAYBACK_GRACE_MILLIS;
        });
    }

    private Mono<Void> maybePublishBargeInSignal(String sessionId) {
        long now = System.currentTimeMillis();
        Long activeUntil = playbackActiveUntilMillis.get(sessionId);
        if (activeUntil == null || activeUntil < now) {
            return Mono.empty();
        }

        Long lastKillAt = lastKillSignalAtMillis.get(sessionId);
        if (lastKillAt != null && now - lastKillAt < KILL_DEBOUNCE_MILLIS) {
            return Mono.empty();
        }

        lastKillSignalAtMillis.put(sessionId, now);
        String controlJson = "{\"action\":\"kill\",\"correlation_id\":\"" + sessionId + "\"}";
        AMQP.BasicProperties properties = new AMQP.BasicProperties.Builder()
                .contentType("application/json")
                .correlationId(sessionId)
                .deliveryMode(2)
                .build();

        LOGGER.info("Publishing barge-in kill signal for session {}", sessionId);
        return sender.send(Mono.just(new OutboundMessage(
                        CONTROL_SIGNALS_EXCHANGE,
                        "",
                        properties,
                        controlJson.getBytes(StandardCharsets.UTF_8)
                )))
                .then();
    }

    private OutboundMessage toOutboundMessage(String sessionId, WebSocketMessage message) {
        DataBuffer payload = message.getPayload();
        byte[] audioChunk = new byte[payload.readableByteCount()];
        try {
            payload.read(audioChunk);
        } finally {
            DataBufferUtils.release(payload);
        }

        AMQP.BasicProperties properties = new AMQP.BasicProperties.Builder()
                .contentType("application/octet-stream")
                .correlationId(sessionId)
                .deliveryMode(2)
                .build();

        return new OutboundMessage("", AUDIO_INCOMING_RAW_QUEUE, properties, audioChunk);
    }
}
