import asyncio
import os

import aio_pika


RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "guest")

TEXT_LLM_QUEUE = "text.llm.processing"
TEXT_TTS_QUEUE = "text.tts.processing"

SYSTEM_PROMPT = (
    "You are JarvisLabs customer helpdesk assistant. "
    "Answer clearly, stay concise, and ground responses in known support context."
)


def rabbitmq_url() -> str:
    return f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASSWORD}@{RABBITMQ_HOST}:{RABBITMQ_PORT}/"


async def connect_broker() -> aio_pika.RobustConnection:
    while True:
        try:
            return await aio_pika.connect_robust(rabbitmq_url())
        except aio_pika.exceptions.AMQPConnectionError:
            print("[LLM] Waiting for RabbitMQ...")
            await asyncio.sleep(2)


async def stream_response_tokens(user_text: str):
    """Placeholder for streaming Qwen/Gemma token generation."""
    grounded_prompt = f"{SYSTEM_PROMPT}\nCustomer: {user_text}\nAssistant:"
    print(f"[LLM] Grounded prompt: {grounded_prompt}")

    simulated_response = "I can help with that. Please share the issue details."
    for token in simulated_response.split(" "):
        await asyncio.sleep(0.02)
        yield token + " "


async def main() -> None:
    connection = await connect_broker()

    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)

        incoming_queue = await channel.declare_queue(TEXT_LLM_QUEUE, durable=True)
        await channel.declare_queue(TEXT_TTS_QUEUE, durable=True)

        print(f"[LLM] Listening to {TEXT_LLM_QUEUE}...")

        async with incoming_queue.iterator() as queue_iter:
            async for message in queue_iter:
                try:
                    correlation_id = message.correlation_id
                    if not correlation_id:
                        print("[LLM] Received transcript without correlation_id")

                    user_text = message.body.decode("utf-8")
                    print(f"[LLM] Received transcript: {user_text}; correlation_id={correlation_id}")

                    async for token in stream_response_tokens(user_text):
                        await channel.default_exchange.publish(
                            aio_pika.Message(
                                token.encode("utf-8"),
                                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                                correlation_id=correlation_id,
                            ),
                            routing_key=TEXT_TTS_QUEUE,
                        )
                        print(f"[LLM] Published token: {token!r}; correlation_id={correlation_id}")

                    await message.ack()
                except Exception as exc:
                    print(f"[LLM] Failed to process transcript: {exc}")
                    await message.nack(requeue=True)


if __name__ == "__main__":
    asyncio.run(main())
