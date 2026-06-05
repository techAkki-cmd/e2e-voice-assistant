package ai.e2e.orchestrator.amqp;

import static ai.e2e.orchestrator.config.RabbitMQConfig.TEXT_ASR_LIVE_EXCHANGE;

import ai.e2e.orchestrator.websocket.AudioStreamHandler;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.amqp.core.ExchangeTypes;
import org.springframework.amqp.core.Message;
import org.springframework.amqp.rabbit.annotation.Exchange;
import org.springframework.amqp.rabbit.annotation.Queue;
import org.springframework.amqp.rabbit.annotation.QueueBinding;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.stereotype.Component;

@Component
public class AsrLiveTranscriptListener {

    private static final Logger LOGGER = LoggerFactory.getLogger(AsrLiveTranscriptListener.class);
    private final ObjectMapper objectMapper;

    public AsrLiveTranscriptListener(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    @RabbitListener(bindings = @QueueBinding(
            value = @Queue(name = "", durable = "false", exclusive = "true", autoDelete = "true"),
            exchange = @Exchange(value = TEXT_ASR_LIVE_EXCHANGE, type = ExchangeTypes.FANOUT, durable = "true")
    ))
    public void onAsrLiveTranscript(Message message) {
        String correlationId = message.getMessageProperties().getCorrelationId();
        Object traceparent = message.getMessageProperties().getHeaders().get("traceparent");

        try {
            JsonNode payload = objectMapper.readTree(message.getBody());
            if (!"partial".equals(payload.path("type").asText())) {
                return;
            }

            String userId = payload.path("user_id").asText(correlationId);
            String text = payload.path("text").asText("");
            if (userId == null || userId.isBlank() || text.isBlank()) {
                return;
            }

            String websocketEvent = objectMapper.createObjectNode()
                    .put("event", "transcript_partial")
                    .put("text", text)
                    .toString();
            boolean routed = AudioStreamHandler.sendTextToSession(userId, websocketEvent);
            if (!routed) {
                LOGGER.debug("Dropped ASR partial for missing WebSocket user_id={}", userId);
            } else {
                LOGGER.debug(
                        "Routed ASR partial user_id={} traceparent={} text={}",
                        userId,
                        traceparent,
                        text
                );
            }
        } catch (Exception exception) {
            LOGGER.debug("Ignoring malformed ASR live transcript message: {}", exception.getMessage());
        }
    }
}
