package ai.e2e.orchestrator.config;

import org.springframework.amqp.core.Queue;
import org.springframework.amqp.core.FanoutExchange;
import org.springframework.amqp.rabbit.connection.CachingConnectionFactory;
import org.springframework.amqp.rabbit.core.RabbitAdmin;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import reactor.rabbitmq.RabbitFlux;
import reactor.rabbitmq.Receiver;
import reactor.rabbitmq.ReceiverOptions;
import reactor.rabbitmq.Sender;
import reactor.rabbitmq.SenderOptions;

@Configuration
public class RabbitMQConfig {

    public static final String AUDIO_INCOMING_RAW_QUEUE = "audio.incoming.raw";
    public static final String AUDIO_OUTGOING_STREAM_QUEUE = "audio.outgoing.stream";
    public static final String CONTROL_SIGNALS_EXCHANGE = "control.signals";

    @Bean
    public Queue audioIncomingRawQueue() {
        return new Queue(AUDIO_INCOMING_RAW_QUEUE, true);
    }

    @Bean
    public Queue audioOutgoingStreamQueue() {
        return new Queue(AUDIO_OUTGOING_STREAM_QUEUE, true);
    }

    @Bean
    public FanoutExchange controlSignalsExchange() {
        return new FanoutExchange(CONTROL_SIGNALS_EXCHANGE, true, false);
    }

    @Bean
    public CachingConnectionFactory cachingConnectionFactory(
            @Value("${spring.rabbitmq.host:localhost}") String host,
            @Value("${spring.rabbitmq.port:5672}") int port,
            @Value("${spring.rabbitmq.username:guest}") String username,
            @Value("${spring.rabbitmq.password:guest}") String password
    ) {
        CachingConnectionFactory connectionFactory = new CachingConnectionFactory(host, port);
        connectionFactory.setUsername(username);
        connectionFactory.setPassword(password);
        return connectionFactory;
    }

    @Bean
    public RabbitAdmin rabbitAdmin(CachingConnectionFactory connectionFactory) {
        return new RabbitAdmin(connectionFactory);
    }

    @Bean(destroyMethod = "close")
    public Sender rabbitSender(CachingConnectionFactory connectionFactory) {
        return RabbitFlux.createSender(
                new SenderOptions().connectionFactory(connectionFactory.getRabbitConnectionFactory())
        );
    }

    @Bean(destroyMethod = "close")
    public Receiver rabbitReceiver(CachingConnectionFactory connectionFactory) {
        return RabbitFlux.createReceiver(
                new ReceiverOptions().connectionFactory(connectionFactory.getRabbitConnectionFactory())
        );
    }
}
