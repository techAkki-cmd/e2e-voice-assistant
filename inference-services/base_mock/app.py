import pika
import time
import json

def connect_broker():
    while True:
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(host='rabbitmq', port=5672)
            )
            return connection
        except pika.exceptions.AMQPConnectionError:
            print("Waiting for RabbitMQ to start...")
            time.sleep(2)

def callback(ch, method, properties, body):
    start_time = time.time()

    print(f"[Inference] Received chunk of size: {len(body)} bytes")

    time.sleep(0.005)

    ch.basic_ack(delivery_tag=method.delivery_tag)

def main():
    connection = connect_broker()
    channel = connection.channel()

    channel.queue_declare(queue='audio.incoming.raw', durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='audio.incoming.raw', on_message_callback=callback)

    print("[Inference Node] Listening to audio.incoming.raw...")
    channel.start_consuming()

if __name__ == '__main__':
    main()