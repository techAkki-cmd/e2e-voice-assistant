package ai.e2e.orchestrator.amqp;

import static ai.e2e.orchestrator.config.RabbitMQConfig.AUDIO_OUTGOING_STREAM_QUEUE;

import ai.e2e.orchestrator.websocket.AudioStreamHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.amqp.core.Message;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.stereotype.Component;

@Component
public class AudioResponseListener {

    private static final Logger LOGGER = LoggerFactory.getLogger(AudioResponseListener.class);

    @RabbitListener(queues = AUDIO_OUTGOING_STREAM_QUEUE)
    public void onAudioResponse(Message message) {
        String correlationId = message.getMessageProperties().getCorrelationId();

        boolean routed = AudioStreamHandler.sendAudioToSession(correlationId, message.getBody());
        if (!routed) {
            LOGGER.debug("Dropped outbound audio response with correlation ID {}", correlationId);
        }
    }
}
