import asyncio
import json
import os
import re
import threading
import time
import uuid
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
MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "64"))
REPETITION_PENALTY = float(os.getenv("LLM_REPETITION_PENALTY", "1.05"))
STREAM_FLUSH_CHARS = int(os.getenv("LLM_STREAM_FLUSH_CHARS", "16"))
HISTORY_TURNS = int(os.getenv("LLM_HISTORY_TURNS", "6"))
HISTORY_TTL_SECONDS = int(os.getenv("LLM_HISTORY_TTL_SECONDS", "86400"))
INTERRUPT_TTL_SECONDS = float(os.getenv("LLM_INTERRUPT_TTL_SECONDS", "15"))
LLM_WARMUP_ENABLED = os.getenv("LLM_WARMUP_ENABLED", "true").lower() == "true"

SYSTEM_PROMPT = os.getenv(
    "LLM_SYSTEM_PROMPT",
    (
        "You are Jarvis, a highly capable, conversational AI assistant. You are speaking to the user "
        "over a live voice interface, so keep answers concise, natural, and easy to listen to. "
        "Sound warm, intelligent, and direct. Do not use markdown, bullet points, numbered lists, or emojis. "
        "For greetings and casual conversation, respond briefly like a real voice assistant. "
        "For coding, technical explanations, general knowledge, and reasoning questions, use your broad "
        "knowledge to be genuinely helpful. If relevant company context is provided below, seamlessly "
        "incorporate it into your answer without announcing that you are using context. If the context is "
        "empty or irrelevant, answer general questions normally. For specific company facts such as pricing, "
        "policies, GPU availability, account details, refunds, or support commitments, do not invent details. "
        "If you are unsure, give a helpful general answer and suggest checking the live dashboard or official "
        "support. Never say 'Based on the context' or 'The document does not mention this.' Just answer fluidly."
    ),
)
GARBLED_TRANSCRIPT_FALLBACK = "I didn't catch that clearly. Please repeat."
ASSISTANT_IDENTITY_ANSWER = "I'm Jarvis, your voice assistant."
COURTESY_ANSWER = "You're welcome."
UNKNOWN_NAME_ANSWER = "I don't know your name yet."

DIRECT_NAME_QUESTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bwhat(?:'s| is)\s+my\s+name\b",
        r"\btell\s+me\s+my\s+name\b",
        r"\b(?:do\s+you\s+)?remember\s+my\s+name\b",
        r"\b(?:can\s+you|could\s+you|please)\s+tell\s+me\s+my\s+name\b",
    ]
]
DIRECT_ASSISTANT_IDENTITY_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bwhat(?:'s| is)\s+your\s+name\b",
        r"\bwho\s+are\s+you\b",
        r"\bare\s+you\s+jarvis\b",
        r"\b(?:can\s+you|could\s+you|please)\s+tell\s+me\s+your\s+name\b",
    ]
]
DIRECT_COURTESY_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"^\s*(?:okay[,\s]*)?(?:thank\s+you|thanks)(?:\s+jarvis)?\s*[.!?]*\s*$",
        r"^\s*(?:thanks\s+a\s+lot|thank\s+you\s+very\s+much)\s*[.!?]*\s*$",
    ]
]
DIRECT_GREETING_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"^\s*(?:hi|hello|hey)\s+(?:jarvis|jarvislabs)\s*[.!?]*\s*$",
        r"^\s*(?:hi|hello|hey)\s*[.!?]*\s*$",
    ]
]
NAME_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\b(?:no[,\s]*)?my name is\s+([A-Za-z][A-Za-z' -]{0,60}?)(?=$|[.!?,;:]|\s+(?:and|but|so|because|i\s+need|i\s+want|please|can|could)\b)",
        r"\bcall me\s+([A-Za-z][A-Za-z' -]{0,60}?)(?=$|[.!?,;:]|\s+(?:and|but|so|because|i\s+need|i\s+want|please|can|could)\b)",
    ]
]
COMPANY_SUPPORT_TOPIC_PATTERN = re.compile(
    r"\b(?:jarvislabs?|gpu|gpus|llm|dashboard|notebook|terminal|instance|deployment|deploy|"
    r"pricing|billing|account|ssh|storage|container|cuda|pytorch|inference|refund|support|"
    r"company|companies|platform|product|service|services|feature|features|policy|policies|"
    r"document|documents|documentation|docs|terms)\b",
    re.IGNORECASE,
)
GARBLED_TOKEN_PATTERN = re.compile(r"\b[A-Z]{4,}\b")
REJECTED_NAME_VALUES = {
    "asking",
    "checking",
    "choosing",
    "doing",
    "fine",
    "good",
    "help",
    "here",
    "interested",
    "looking",
    "trying",
}


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
        inputs = build_inputs(tokenizer, model, [], "Say ready.", "", False)
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


def build_system_prompt(retrieved_context: str, is_company_question: bool) -> str:
    context = retrieved_context.strip()
    if not context:
        return SYSTEM_PROMPT
    return f"{SYSTEM_PROMPT}\n\nRelevant company context:\n{context}"


def parse_rag_payload(body: bytes) -> tuple[str, str]:
    decoded = body.decode("utf-8", errors="replace").strip()
    if not decoded:
        return "", ""

    try:
        payload = json.loads(decoded)
    except json.JSONDecodeError:
        return decoded, ""

    if not isinstance(payload, dict):
        return decoded, ""

    user_transcript = payload.get("user_transcript")
    retrieved_context = payload.get("retrieved_context")
    if not isinstance(user_transcript, str):
        return decoded, ""
    if not isinstance(retrieved_context, str):
        retrieved_context = ""

    return user_transcript.strip(), retrieved_context.strip()


def build_inputs(tokenizer, model, history: List[dict], user_text: str, retrieved_context: str, is_company_question: bool):
    messages = [
        {"role": "system", "content": build_system_prompt(retrieved_context, is_company_question)},
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
    retrieved_context: str,
    is_company_question: bool,
    stop_event: threading.Event,
) -> AsyncIterator[str]:
    inputs = build_inputs(tokenizer, model, history, user_text, retrieved_context, is_company_question)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=1.0)

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
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
    try:
        raw_history = await redis_client.get(history_key(user_id))
    except RedisError as exc:
        print(f"[LLM] Redis history load failed user_id={user_id}: {exc}", flush=True)
        return []

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
    try:
        await redis_client.set(
            history_key(user_id),
            json.dumps(trimmed_history, separators=(",", ":")),
            ex=HISTORY_TTL_SECONDS,
        )
    except RedisError as exc:
        print(f"[LLM] Redis history save failed user_id={user_id}: {exc}", flush=True)


def normalize_name(raw_name: str) -> str | None:
    name = " ".join(raw_name.strip(" .!?,-;:").split())
    if not name:
        return None

    words = name.split()
    if len(words) > 4:
        return None
    if any(not re.fullmatch(r"[A-Za-z][A-Za-z'-]*", word) for word in words):
        return None
    if name.lower() in REJECTED_NAME_VALUES:
        return None

    return " ".join(word[:1].upper() + word[1:] for word in words)


def is_direct_name_question(user_text: str) -> bool:
    return any(pattern.search(user_text) for pattern in DIRECT_NAME_QUESTION_PATTERNS)


def is_direct_assistant_identity_question(user_text: str) -> bool:
    return any(pattern.search(user_text) for pattern in DIRECT_ASSISTANT_IDENTITY_PATTERNS)


def is_direct_courtesy(user_text: str) -> bool:
    return any(pattern.match(user_text) for pattern in DIRECT_COURTESY_PATTERNS)


def is_direct_greeting(user_text: str) -> bool:
    return any(pattern.match(user_text) for pattern in DIRECT_GREETING_PATTERNS)


def is_company_support_question(user_text: str) -> bool:
    return bool(COMPANY_SUPPORT_TOPIC_PATTERN.search(user_text))


def looks_garbled(user_text: str) -> bool:
    text = user_text.strip()
    if not text:
        return True
    if text.endswith("..."):
        return True
    words = text.split()
    if len(words) >= 4 and sum(1 for word in words if GARBLED_TOKEN_PATTERN.fullmatch(word.strip(".,!?"))) >= 3:
        return True
    alpha_chars = sum(1 for character in text if character.isalpha())
    return alpha_chars < 3


def extract_latest_user_name(history: List[dict], current_transcript: str) -> str | None:
    latest_name = None
    user_texts = [item["content"] for item in history if item.get("role") == "user"]
    user_texts.append(current_transcript)

    for text in user_texts:
        for pattern in NAME_PATTERNS:
            for match in pattern.finditer(text):
                name = normalize_name(match.group(1))
                if name:
                    latest_name = name

    return latest_name


def extract_name_from_current_transcript(current_transcript: str) -> str | None:
    for pattern in NAME_PATTERNS:
        for match in pattern.finditer(current_transcript):
            name = normalize_name(match.group(1))
            if name:
                return name
    return None


def cleanup_interrupted_state(interrupted_at: Dict[str, float]) -> None:
    now = time.monotonic()
    for key in [key for key, seen_at in interrupted_at.items() if now - seen_at > INTERRUPT_TTL_SECONDS]:
        interrupted_at.pop(key, None)


def cleanup_recent_timestamps(recent_events: Dict[str, float], ttl_seconds: float) -> None:
    now = time.monotonic()
    for key in [key for key, seen_at in recent_events.items() if now - seen_at > ttl_seconds]:
        recent_events.pop(key, None)


def header_as_text(headers: dict | None, name: str) -> str | None:
    if not headers:
        return None
    value = headers.get(name)
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def trace_headers(correlation_id: str | None, traceparent: str | None, response_id: str | None = None) -> dict:
    headers = {}
    if traceparent:
        headers["traceparent"] = traceparent
    if correlation_id:
        headers["user_id"] = correlation_id
    if response_id:
        headers["response_id"] = response_id
    return headers


async def publish_text_chunk(
    channel: aio_pika.Channel,
    chunk: str,
    correlation_id: str | None,
    traceparent: str | None,
    response_id: str,
) -> None:
    await channel.default_exchange.publish(
        aio_pika.Message(
            chunk.encode("utf-8"),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            correlation_id=correlation_id,
            content_type="text/plain",
            headers=trace_headers(correlation_id, traceparent, response_id),
        ),
        routing_key=TEXT_TTS_QUEUE,
    )


async def publish_direct_answer(
    channel: aio_pika.Channel,
    redis_client: redis.Redis,
    user_id: str,
    history: List[dict],
    user_text: str,
    assistant_text: str,
    correlation_id: str | None,
    traceparent: str | None,
    response_id: str,
    log_label: str,
) -> None:
    await publish_text_chunk(channel, assistant_text, correlation_id, traceparent, response_id)
    await save_history(
        redis_client,
        user_id,
        [
            *history,
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
    )
    print(
        f"[LLM] Published {log_label}: {assistant_text!r}; "
        f"correlation_id={correlation_id}; response_id={response_id}; traceparent={traceparent}",
        flush=True,
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

                            user_text, retrieved_context = parse_rag_payload(message.body)
                            print(
                                f"[LLM] Received transcript: {user_text!r}; "
                                f"retrieved_context_chars={len(retrieved_context)}; "
                                f"correlation_id={correlation_id}; traceparent={traceparent}",
                                flush=True,
                            )

                            user_id = correlation_id or "__manual_smoke_test__"
                            history = await load_history(redis_client, user_id)

                            response_id = uuid.uuid4().hex
                            remembered_name = extract_latest_user_name(history, user_text)
                            current_name = extract_name_from_current_transcript(user_text)
                            is_company_question = is_company_support_question(user_text)
                            if is_direct_assistant_identity_question(user_text):
                                await publish_direct_answer(
                                    channel,
                                    redis_client,
                                    user_id,
                                    history,
                                    user_text,
                                    ASSISTANT_IDENTITY_ANSWER,
                                    correlation_id,
                                    traceparent,
                                    response_id,
                                    "direct assistant identity",
                                )
                                cleanup_interrupted_state(interrupted_at)
                                await message.ack()
                                continue

                            if is_direct_courtesy(user_text):
                                await publish_direct_answer(
                                    channel,
                                    redis_client,
                                    user_id,
                                    history,
                                    user_text,
                                    COURTESY_ANSWER,
                                    correlation_id,
                                    traceparent,
                                    response_id,
                                    "direct courtesy answer",
                                )
                                cleanup_interrupted_state(interrupted_at)
                                await message.ack()
                                continue

                            if is_direct_greeting(user_text):
                                await publish_direct_answer(
                                    channel,
                                    redis_client,
                                    user_id,
                                    history,
                                    user_text,
                                    "Hello.",
                                    correlation_id,
                                    traceparent,
                                    response_id,
                                    "direct greeting",
                                )
                                cleanup_interrupted_state(interrupted_at)
                                await message.ack()
                                continue

                            if is_direct_name_question(user_text):
                                assistant_text = f"Your name is {remembered_name}." if remembered_name else UNKNOWN_NAME_ANSWER
                                await publish_direct_answer(
                                    channel,
                                    redis_client,
                                    user_id,
                                    history,
                                    user_text,
                                    assistant_text,
                                    correlation_id,
                                    traceparent,
                                    response_id,
                                    "direct memory answer",
                                )
                                cleanup_interrupted_state(interrupted_at)
                                await message.ack()
                                continue

                            if current_name:
                                await publish_direct_answer(
                                    channel,
                                    redis_client,
                                    user_id,
                                    history,
                                    user_text,
                                    "I'll remember that.",
                                    correlation_id,
                                    traceparent,
                                    response_id,
                                    "direct name capture",
                                )
                                cleanup_interrupted_state(interrupted_at)
                                await message.ack()
                                continue

                            if looks_garbled(user_text):
                                assistant_text = GARBLED_TRANSCRIPT_FALLBACK
                                await publish_text_chunk(
                                    channel,
                                    assistant_text,
                                    correlation_id,
                                    traceparent,
                                    response_id,
                                )
                                await save_history(
                                    redis_client,
                                    user_id,
                                    [
                                        *history,
                                        {"role": "user", "content": user_text},
                                        {"role": "assistant", "content": assistant_text},
                                    ],
                                )
                                print(
                                    f"[LLM] Published guard answer: {assistant_text!r}; "
                                    f"correlation_id={correlation_id}; response_id={response_id}; traceparent={traceparent}",
                                    flush=True,
                                )
                                cleanup_interrupted_state(interrupted_at)
                                await message.ack()
                                continue

                            stop_event = threading.Event()

                            if correlation_id:
                                active_generations[correlation_id] = stop_event

                            assistant_chunks = []
                            async for chunk in stream_response_chunks(
                                tokenizer,
                                model,
                                history,
                                user_text,
                                retrieved_context,
                                is_company_question,
                                stop_event,
                            ):
                                if stop_event.is_set():
                                    break
                                assistant_chunks.append(chunk)
                                await publish_text_chunk(channel, chunk, correlation_id, traceparent, response_id)
                                print(
                                    f"[LLM] Published chunk: {chunk!r}; "
                                    f"correlation_id={correlation_id}; response_id={response_id}; traceparent={traceparent}",
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
