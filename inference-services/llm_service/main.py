import asyncio
import json
import os
import threading
import time
from queue import Empty
from typing import AsyncIterator, Dict, List

import aio_pika
import redis.asyncio as redis
import torch
from redis.exceptions import RedisError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)


RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "guest")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))

TEXT_LLM_QUEUE = "text.llm.processing"
TEXT_TTS_QUEUE = "text.tts.processing"
CONTROL_SIGNALS_EXCHANGE = "control.signals"

MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "56"))
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
REPETITION_PENALTY = float(os.getenv("LLM_REPETITION_PENALTY", "1.05"))
STREAM_FLUSH_CHARS = int(os.getenv("LLM_STREAM_FLUSH_CHARS", "16"))
HISTORY_TURNS = int(os.getenv("LLM_HISTORY_TURNS", "6"))
HISTORY_TTL_SECONDS = int(os.getenv("LLM_HISTORY_TTL_SECONDS", "86400"))
INTERRUPT_TTL_SECONDS = float(os.getenv("LLM_INTERRUPT_TTL_SECONDS", "15"))
LLM_WARMUP_ENABLED = os.getenv("LLM_WARMUP_ENABLED", "true").lower() == "true"

SYSTEM_PROMPT = os.getenv(
    "LLM_SYSTEM_PROMPT",
    (
        "You are JarvisLabs customer support assistant for a GPU cloud and AI deployment platform. "
        "Ground every answer in this support scope: GPU instances, CUDA/PyTorch setup, model deployment, "
        "billing/account help, SSH access, storage, containers, and troubleshooting inference workloads. "
        "Known safe facts: JarvisLabs provides cloud GPU compute for AI/ML workloads; GPU availability, "
        "pricing, and exact models can change, so do not invent a live catalog. If asked for exact current "
        "inventory or pricing, say you can explain the types of GPUs and guide the user to check the live "
        "JarvisLabs dashboard. If the transcript looks garbled or off-topic, ask one concise clarification. "
        "Use the conversation history. Reply in one short sentence when possible."
    ),
)


class KillSignalStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.stop_event.is_set()


def rabbitmq_url() -> str:
    return f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASSWORD}@{RABBITMQ_HOST}:{RABBITMQ_PORT}/"


def redis_url() -> str:
    auth = f":{REDIS_PASSWORD}@" if REDIS_PASSWORD else ""
    return f"redis://{auth}{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"


async def connect_broker() -> aio_pika.RobustConnection:
    while True:
        try:
            return await aio_pika.connect_robust(rabbitmq_url())
        except aio_pika.exceptions.AMQPConnectionError:
            print("[LLM] Waiting for RabbitMQ...", flush=True)
            await asyncio.sleep(2)


async def connect_redis() -> redis.Redis:
    pool = redis.ConnectionPool.from_url(
        redis_url(),
        max_connections=REDIS_MAX_CONNECTIONS,
        decode_responses=True,
    )
    client = redis.Redis(connection_pool=pool)
    while True:
        try:
            await client.ping()
            print(
                f"[LLM] Connected to Redis host={REDIS_HOST} port={REDIS_PORT} "
                f"db={REDIS_DB} max_connections={REDIS_MAX_CONNECTIONS}",
                flush=True,
            )
            return client
        except RedisError as exc:
            print(f"[LLM] Waiting for Redis: {exc}", flush=True)
            await asyncio.sleep(2)


def load_model():
    print(f"[LLM] Loading {MODEL_NAME} with bitsandbytes 8-bit quantization", flush=True)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    if LLM_WARMUP_ENABLED:
        print("[LLM] Running startup warmup generation...", flush=True)
        inputs = build_inputs(tokenizer, model, [], "Say ready.")
        with torch.inference_mode():
            model.generate(
                **inputs,
                max_new_tokens=2,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        print("[LLM] Startup warmup complete", flush=True)
    return tokenizer, model


def build_inputs(tokenizer, model, history: List[dict], user_text: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_text.strip()},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {key: value.to(model.device) for key, value in inputs.items()}


def next_streamer_text(iterator):
    try:
        return True, next(iterator)
    except StopIteration:
        return False, ""


async def stream_response_chunks(
    tokenizer,
    model,
    history: List[dict],
    user_text: str,
    stop_event: threading.Event,
) -> AsyncIterator[str]:
    inputs = build_inputs(tokenizer, model, history, user_text)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=1.0)

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "do_sample": True,
        "repetition_penalty": REPETITION_PENALTY,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "stopping_criteria": StoppingCriteriaList([KillSignalStoppingCriteria(stop_event)]),
    }

    generation_thread = threading.Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
    generation_thread.start()

    buffer = ""
    loop = asyncio.get_running_loop()
    iterator = iter(streamer)

    while not stop_event.is_set():
        try:
            has_value, token_text = await loop.run_in_executor(None, next_streamer_text, iterator)
        except Empty:
            if generation_thread.is_alive():
                continue
            break

        if not has_value:
            break

        if not token_text:
            continue

        buffer += token_text
        should_flush = (
            any(mark in buffer for mark in [" ", "\n", ",", ".", "!", "?"])
            or len(buffer) >= STREAM_FLUSH_CHARS
        )
        if should_flush:
            yield buffer
            buffer = ""

    if buffer and not stop_event.is_set():
        yield buffer

    generation_thread.join(timeout=1.0)


def trim_history(history: List[dict]) -> List[dict]:
    return history[-HISTORY_TURNS * 2 :]


def history_key(user_id: str) -> str:
    return f"voice:history:{user_id}"


async def load_history(redis_client: redis.Redis, user_id: str) -> List[dict]:
    raw_history = await redis_client.get(history_key(user_id))
    if not raw_history:
        return []

    try:
        history = json.loads(raw_history)
    except json.JSONDecodeError:
        print(f"[LLM] Ignoring corrupt Redis history user_id={user_id}", flush=True)
        return []

    if not isinstance(history, list):
        return []

    cleaned_history = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            cleaned_history.append({"role": role, "content": content})

    return trim_history(cleaned_history)


async def save_history(redis_client: redis.Redis, user_id: str, history: List[dict]) -> None:
    trimmed_history = trim_history(history)
    await redis_client.set(
        history_key(user_id),
        json.dumps(trimmed_history, separators=(",", ":")),
        ex=HISTORY_TTL_SECONDS,
    )


def cleanup_interrupted_state(interrupted_at: Dict[str, float]) -> None:
    now = time.monotonic()
    for key in [key for key, seen_at in interrupted_at.items() if now - seen_at > INTERRUPT_TTL_SECONDS]:
        interrupted_at.pop(key, None)


def header_as_text(headers: dict | None, name: str) -> str | None:
    if not headers:
        return None
    value = headers.get(name)
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def trace_headers(correlation_id: str | None, traceparent: str | None) -> dict:
    headers = {}
    if traceparent:
        headers["traceparent"] = traceparent
    if correlation_id:
        headers["user_id"] = correlation_id
    return headers


async def publish_text_chunk(
    channel: aio_pika.Channel,
    chunk: str,
    correlation_id: str | None,
    traceparent: str | None,
) -> None:
    await channel.default_exchange.publish(
        aio_pika.Message(
            chunk.encode("utf-8"),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            correlation_id=correlation_id,
            content_type="text/plain",
            headers=trace_headers(correlation_id, traceparent),
        ),
        routing_key=TEXT_TTS_QUEUE,
    )


async def consume_control_signals(
    channel: aio_pika.Channel,
    active_generations: Dict[str, threading.Event],
    interrupted_at: Dict[str, float],
) -> None:
    exchange = await channel.declare_exchange(CONTROL_SIGNALS_EXCHANGE, aio_pika.ExchangeType.FANOUT, durable=True)
    queue = await channel.declare_queue(exclusive=True, auto_delete=True)
    await queue.bind(exchange)
    print(f"[LLM] Listening to fanout exchange {CONTROL_SIGNALS_EXCHANGE}...", flush=True)

    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            async with message.process(ignore_processed=True):
                try:
                    payload = json.loads(message.body.decode("utf-8"))
                except json.JSONDecodeError:
                    print("[LLM] Ignored malformed control signal", flush=True)
                    continue

                if payload.get("action") != "kill":
                    continue

                correlation_id = payload.get("correlation_id")
                if not correlation_id:
                    continue

                interrupted_at[correlation_id] = time.monotonic()
                stop_event = active_generations.get(correlation_id)
                if stop_event:
                    stop_event.set()
                traceparent = header_as_text(message.headers, "traceparent")
                print(
                    f"[LLM] Kill signal received correlation_id={correlation_id}; traceparent={traceparent}",
                    flush=True,
                )


async def main() -> None:
    tokenizer, model = load_model()
    active_generations: Dict[str, threading.Event] = {}
    interrupted_at: Dict[str, float] = {}
    redis_client = await connect_redis()
    connection = await connect_broker()

    try:
        async with connection:
            channel = await connection.channel()
            await channel.set_qos(prefetch_count=1)

            control_channel = await connection.channel()
            control_task = asyncio.create_task(consume_control_signals(control_channel, active_generations, interrupted_at))

            incoming_queue = await channel.declare_queue(TEXT_LLM_QUEUE, durable=True)
            await channel.declare_queue(TEXT_TTS_QUEUE, durable=True)

            print(f"[LLM] Listening to {TEXT_LLM_QUEUE}...", flush=True)

            try:
                async with incoming_queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        correlation_id = None
                        try:
                            correlation_id = message.correlation_id
                            traceparent = header_as_text(message.headers, "traceparent")
                            if not correlation_id:
                                print("[LLM] Received transcript without correlation_id", flush=True)

                            user_text = message.body.decode("utf-8", errors="replace").strip()
                            print(
                                f"[LLM] Received transcript: {user_text!r}; "
                                f"correlation_id={correlation_id}; traceparent={traceparent}",
                                flush=True,
                            )

                            user_id = correlation_id or "__manual_smoke_test__"
                            history = await load_history(redis_client, user_id)
                            stop_event = threading.Event()

                            if correlation_id:
                                active_generations[correlation_id] = stop_event

                            assistant_chunks = []
                            async for chunk in stream_response_chunks(tokenizer, model, history, user_text, stop_event):
                                if stop_event.is_set():
                                    break
                                assistant_chunks.append(chunk)
                                await publish_text_chunk(channel, chunk, correlation_id, traceparent)
                                print(
                                    f"[LLM] Published chunk: {chunk!r}; "
                                    f"correlation_id={correlation_id}; traceparent={traceparent}",
                                    flush=True,
                                )

                            if stop_event.is_set():
                                print(
                                    "[LLM] Generation interrupted; dropping partial assistant response "
                                    f"correlation_id={correlation_id}; traceparent={traceparent}",
                                    flush=True,
                                )
                            else:
                                assistant_text = "".join(assistant_chunks).strip()
                                if assistant_text:
                                    await save_history(
                                        redis_client,
                                        user_id,
                                        [
                                            *history,
                                            {"role": "user", "content": user_text},
                                            {"role": "assistant", "content": assistant_text},
                                        ],
                                    )

                            if correlation_id:
                                active_generations.pop(correlation_id, None)

                            cleanup_interrupted_state(interrupted_at)
                            await message.ack()
                        except Exception as exc:
                            if correlation_id:
                                active_generations.pop(correlation_id, None)
                            print(f"[LLM] Failed to process transcript: {exc}", flush=True)
                            await message.nack(requeue=True)
            finally:
                control_task.cancel()
                await asyncio.gather(control_task, return_exceptions=True)
    finally:
        await redis_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
