import asyncio
import os

import aio_pika


RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "guest")

TEXT_TTS_QUEUE = "text.tts.processing"
AUDIO_OUTGOING_QUEUE = "audio.outgoing.stream"


def rabbitmq_url() -> str:
    return f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASSWORD}@{RABBITMQ_HOST}:{RABBITMQ_PORT}/"


async def connect_broker() -> aio_pika.RobustConnection:
    while True:
        try:
            return await aio_pika.connect_robust(rabbitmq_url())
        except aio_pika.exceptions.AMQPConnectionError:
            print("[TTS] Waiting for RabbitMQ...")
            await asyncio.sleep(2)


def synthesize_text_to_audio(text_token: str):
    """Placeholder for streaming TTS synthesis."""
    yield b"PCM_FRAME:" + text_token.encode("utf-8")


async def main() -> None:
    connection = await connect_broker()

    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)

        incoming_queue = await channel.declare_queue(TEXT_TTS_QUEUE, durable=True)
        await channel.declare_queue(AUDIO_OUTGOING_QUEUE, durable=True)

        print(f"[TTS] Listening to {TEXT_TTS_QUEUE}...")

        async with incoming_queue.iterator() as queue_iter:
            async for message in queue_iter:
                try:
                    text_token = message.body.decode("utf-8")
                    print(f"[TTS] Received token: {text_token!r}")

                    for audio_frame in synthesize_text_to_audio(text_token):
                        await channel.default_exchange.publish(
                            aio_pika.Message(
                                audio_frame,
                                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                            ),
                            routing_key=AUDIO_OUTGOING_QUEUE,
                        )
                        print(f"[TTS] Published {len(audio_frame)} audio bytes")

                    await message.ack()
                except Exception as exc:
                    print(f"[TTS] Failed to synthesize token: {exc}")
                    await message.nack(requeue=True)


if __name__ == "__main__":
    asyncio.run(main())
