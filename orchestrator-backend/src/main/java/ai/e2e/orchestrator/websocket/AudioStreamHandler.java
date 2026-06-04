package ai.e2e.orchestrator.websocket;

import static ai.e2e.orchestrator.config.RabbitMQConfig.AUDIO_INCOMING_RAW_QUEUE;

import java.util.function.Function;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.socket.WebSocketHandler;
import org.springframework.web.reactive.socket.WebSocketMessage;
import org.springframework.web.reactive.socket.WebSocketSession;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.rabbitmq.OutboundMessage;
import reactor.rabbitmq.QueueSpecification;
import reactor.rabbitmq.Sender;

@Component
public class AudioStreamHandler implements WebSocketHandler {

    private static final Logger LOGGER = LoggerFactory.getLogger(AudioStreamHandler.class);

    private final Sender sender;

    public AudioStreamHandler(Sender sender) {
        this.sender = sender;
    }

    @Override
    public Mono<Void> handle(WebSocketSession session) {
        Flux<OutboundMessage> audioMessages = session.receive()
                .handle((message, sink) -> {
                    if (message.getType() == WebSocketMessage.Type.BINARY) {
                        sink.next(toOutboundMessage().apply(message));
                    }
                });

        return sender.declareQueue(QueueSpecification.queue(AUDIO_INCOMING_RAW_QUEUE).durable(true))
                .then(sender.send(audioMessages))
                .doFinally(signalType -> LOGGER.debug(
                        "Audio WebSocket session {} finished with signal {}",
                        session.getId(),
                        signalType
                ))
                .onErrorResume(throwable -> {
                    LOGGER.debug(
                            "Audio WebSocket session {} closed while streaming audio: {}",
                            session.getId(),
                            throwable.getMessage()
                    );
                    return Mono.empty();
                });
    }

    private Function<WebSocketMessage, OutboundMessage> toOutboundMessage() {
        return message -> {
            DataBuffer payload = message.getPayload();
            byte[] audioChunk = new byte[payload.readableByteCount()];
            payload.read(audioChunk);
            return new OutboundMessage("", AUDIO_INCOMING_RAW_QUEUE, audioChunk);
        };
    }
}
