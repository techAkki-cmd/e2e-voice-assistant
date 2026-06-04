import asyncio
import os

import aio_pika


RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "guest")

AUDIO_INCOMING_QUEUE = "audio.incoming.raw"
TEXT_LLM_QUEUE = "text.llm.processing"
UTTERANCE_CHUNK_TARGET = int(os.getenv("ASR_UTTERANCE_CHUNKS", "1"))


def rabbitmq_url() -> str:
    return f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASSWORD}@{RABBITMQ_HOST}:{RABBITMQ_PORT}/"


async def connect_broker() -> aio_pika.RobustConnection:
    while True:
        try:
            return await aio_pika.connect_robust(rabbitmq_url())
        except aio_pika.exceptions.AMQPConnectionError:
            print("[ASR] Waiting for RabbitMQ...")
            await asyncio.sleep(2)


def process_speech_to_text(audio_bytes: bytes) -> str:
    """Placeholder for streaming ASR + VAD."""
    return f"customer said {len(audio_bytes)} bytes of audio"


async def main() -> None:
    connection = await connect_broker()

    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)

        incoming_queue = await channel.declare_queue(AUDIO_INCOMING_QUEUE, durable=True)
        await channel.declare_queue(TEXT_LLM_QUEUE, durable=True)

        audio_buffer = bytearray()
        buffered_chunks = 0

        print(f"[ASR] Listening to {AUDIO_INCOMING_QUEUE}...")

        async with incoming_queue.iterator() as queue_iter:
            async for message in queue_iter:
                try:
                    correlation_id = message.correlation_id
                    if not correlation_id:
                        print("[ASR] Received audio chunk without correlation_id")

                    audio_buffer.extend(message.body)
                    buffered_chunks += 1
                    print(
                        f"[ASR] Buffered {len(message.body)} bytes; "
                        f"chunks={buffered_chunks}; correlation_id={correlation_id}"
                    )

                    if buffered_chunks >= UTTERANCE_CHUNK_TARGET:
                        transcript = process_speech_to_text(bytes(audio_buffer))
                        await channel.default_exchange.publish(
                            aio_pika.Message(
                                transcript.encode("utf-8"),
                                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                                correlation_id=correlation_id,
                            ),
                            routing_key=TEXT_LLM_QUEUE,
                        )
                        print(
                            f"[ASR] Published transcript: {transcript}; "
                            f"correlation_id={correlation_id}"
                        )
                        audio_buffer.clear()
                        buffered_chunks = 0

                    await message.ack()
                except Exception as exc:
                    print(f"[ASR] Failed to process audio chunk: {exc}")
                    await message.nack(requeue=True)


if __name__ == "__main__":
    asyncio.run(main())
