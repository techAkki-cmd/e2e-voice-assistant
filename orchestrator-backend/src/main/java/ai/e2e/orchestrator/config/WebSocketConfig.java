package ai.e2e.orchestrator.config;

import ai.e2e.orchestrator.websocket.AudioStreamHandler;
import java.util.Map;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.HandlerMapping;
import org.springframework.web.reactive.handler.SimpleUrlHandlerMapping;
import org.springframework.web.reactive.socket.server.support.WebSocketHandlerAdapter;

@Configuration
public class WebSocketConfig {

    public static final String AUDIO_STREAM_PATH = "/api/v1/audio/stream";

    @Bean
    public HandlerMapping webSocketHandlerMapping(AudioStreamHandler audioStreamHandler) {
        SimpleUrlHandlerMapping handlerMapping = new SimpleUrlHandlerMapping();
        handlerMapping.setUrlMap(Map.of(AUDIO_STREAM_PATH, audioStreamHandler));
        handlerMapping.setOrder(-1);
        return handlerMapping;
    }

    @Bean
    public WebSocketHandlerAdapter webSocketHandlerAdapter() {
        return new WebSocketHandlerAdapter();
    }
}
