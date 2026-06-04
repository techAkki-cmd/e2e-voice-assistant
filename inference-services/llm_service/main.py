import asyncio
import json
import os
import threading
import time
from queue import Empty
from typing import AsyncIterator, Dict, List

import aio_pika
import torch
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

TEXT_LLM_QUEUE = "text.llm.processing"
TEXT_TTS_QUEUE = "text.tts.processing"
CONTROL_SIGNALS_EXCHANGE = "control.signals"

MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "56"))
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
REPETITION_PENALTY = float(os.getenv("LLM_REPETITION_PENALTY", "1.05"))
STREAM_FLUSH_CHARS = int(os.getenv("LLM_STREAM_FLUSH_CHARS", "16"))
HISTORY_TURNS = int(os.getenv("LLM_HISTORY_TURNS", "5"))
SESSION_TTL_SECONDS = int(os.getenv("LLM_SESSION_TTL_SECONDS", "900"))
INTERRUPT_TTL_SECONDS = float(os.getenv("LLM_INTERRUPT_TTL_SECONDS", "15"))

SYSTEM_PROMPT = os.getenv(
    "LLM_SYSTEM_PROMPT",
    (
        "You are JarvisLabs customer support assistant. "
        "Use the conversation history. Reply in one short sentence when possible. "
        "Ask at most one focused follow-up question. Stay helpful and concise."
    ),
)


class KillSignalStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.stop_event.is_set()


def rabbitmq_url() -> str:
    return f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASSWORD}@{RABBITMQ_HOST}:{RABBITMQ_PORT}/"


async def connect_broker() -> aio_pika.RobustConnection:
    while True:
        try:
            return await aio_pika.connect_robust(rabbitmq_url())
        except aio_pika.exceptions.AMQPConnectionError:
            print("[LLM] Waiting for RabbitMQ...", flush=True)
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


def cleanup_expired_state(
    last_seen: Dict[str, float],
    histories: Dict[str, List[dict]],
    interrupted_at: Dict[str, float],
) -> None:
    now = time.monotonic()
    for key in [key for key, seen_at in last_seen.items() if now - seen_at > SESSION_TTL_SECONDS]:
        last_seen.pop(key, None)
        histories.pop(key, None)
        print(f"[LLM] Expired conversation history correlation_id={key}", flush=True)

    for key in [key for key, seen_at in interrupted_at.items() if now - seen_at > INTERRUPT_TTL_SECONDS]:
        interrupted_at.pop(key, None)


async def publish_text_chunk(channel: aio_pika.Channel, chunk: str, correlation_id: str | None) -> None:
    await channel.default_exchange.publish(
        aio_pika.Message(
            chunk.encode("utf-8"),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            correlation_id=correlation_id,
            content_type="text/plain",
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
                print(f"[LLM] Kill signal received correlation_id={correlation_id}", flush=True)


async def main() -> None:
    tokenizer, model = load_model()
    histories: Dict[str, List[dict]] = {}
    last_seen: Dict[str, float] = {}
    active_generations: Dict[str, threading.Event] = {}
    interrupted_at: Dict[str, float] = {}
    connection = await connect_broker()

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
                    try:
                        correlation_id = message.correlation_id
                        if not correlation_id:
                            print("[LLM] Received transcript without correlation_id", flush=True)

                        user_text = message.body.decode("utf-8", errors="replace").strip()
                        print(f"[LLM] Received transcript: {user_text!r}; correlation_id={correlation_id}", flush=True)

                        session_key = correlation_id or "__manual_smoke_test__"
                        history = histories.get(session_key, [])
                        stop_event = threading.Event()

                        if correlation_id:
                            active_generations[correlation_id] = stop_event

                        assistant_chunks = []
                        async for chunk in stream_response_chunks(tokenizer, model, history, user_text, stop_event):
                            if stop_event.is_set():
                                break
                            assistant_chunks.append(chunk)
                            await publish_text_chunk(channel, chunk, correlation_id)
                            print(f"[LLM] Published chunk: {chunk!r}; correlation_id={correlation_id}", flush=True)

                        if stop_event.is_set():
                            print(f"[LLM] Generation interrupted; dropping partial assistant response correlation_id={correlation_id}", flush=True)
                        else:
                            assistant_text = "".join(assistant_chunks).strip()
                            if assistant_text:
                                histories[session_key] = trim_history(
                                    [
                                        *history,
                                        {"role": "user", "content": user_text},
                                        {"role": "assistant", "content": assistant_text},
                                    ]
                                )
                                last_seen[session_key] = time.monotonic()

                        if correlation_id:
                            active_generations.pop(correlation_id, None)

                        cleanup_expired_state(last_seen, histories, interrupted_at)
                        await message.ack()
                    except Exception as exc:
                        print(f"[LLM] Failed to process transcript: {exc}", flush=True)
                        await message.nack(requeue=True)
        finally:
            control_task.cancel()
            await asyncio.gather(control_task, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
