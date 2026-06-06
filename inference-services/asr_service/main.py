import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import aio_pika
import numpy as np
import sherpa_onnx


RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "guest")

AUDIO_INCOMING_QUEUE = "audio.incoming.raw"
TEXT_RAG_QUEUE = "text.rag.processing"
TEXT_ASR_LIVE_EXCHANGE = "text.asr.live"

SAMPLE_RATE = int(os.getenv("ASR_SAMPLE_RATE", "16000"))
SESSION_TTL_SECONDS = int(os.getenv("ASR_SESSION_TTL_SECONDS", "120"))
PARTIAL_THROTTLE_SECONDS = float(os.getenv("ASR_PARTIAL_THROTTLE_SECONDS", "0.15"))

SHERPA_MODEL_DIR = Path(
    os.getenv(
        "ASR_SHERPA_MODEL_DIR",
        "/models/sherpa-onnx/sherpa-onnx-streaming-zipformer-en-2023-06-26",
    )
)
SHERPA_ENCODER = os.getenv(
    "ASR_SHERPA_ENCODER",
    "encoder-epoch-99-avg-1-chunk-16-left-128.onnx",
)
SHERPA_DECODER = os.getenv(
    "ASR_SHERPA_DECODER",
    "decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
)
SHERPA_JOINER = os.getenv(
    "ASR_SHERPA_JOINER",
    "joiner-epoch-99-avg-1-chunk-16-left-128.onnx",
)
SHERPA_TOKENS = os.getenv("ASR_SHERPA_TOKENS", "tokens.txt")
SHERPA_PROVIDER = os.getenv("ASR_SHERPA_PROVIDER", "cpu")
SHERPA_NUM_THREADS = int(os.getenv("ASR_SHERPA_NUM_THREADS", "2"))
SHERPA_DECODING_METHOD = os.getenv("ASR_SHERPA_DECODING_METHOD", "greedy_search")
SHERPA_MAX_ACTIVE_PATHS = int(os.getenv("ASR_SHERPA_MAX_ACTIVE_PATHS", "4"))
SHERPA_RULE1_MIN_TRAILING_SILENCE = float(os.getenv("ASR_SHERPA_RULE1_MIN_TRAILING_SILENCE", "2.4"))
SHERPA_RULE2_MIN_TRAILING_SILENCE = float(os.getenv("ASR_SHERPA_RULE2_MIN_TRAILING_SILENCE", "1.2"))
SHERPA_RULE3_MIN_UTTERANCE_LENGTH = float(os.getenv("ASR_SHERPA_RULE3_MIN_UTTERANCE_LENGTH", "20"))

ASR_TRANSCRIPT_REPLACEMENTS = os.getenv(
    "ASR_TRANSCRIPT_REPLACEMENTS",
    (
        "Erycheet=Arijit;Arycheet=Arijit;Arigit=Arijit;Ari Jeet=Arijit;"
        "L&M=LLM;L and M=LLM;L.N.=LLM;"
        "GUE=GUI;gooey=GUI;G U I=GUI;G P U=GPU;4GPU=What GPU;"
        "geo needs=GPU needs;geo instance=GPU instance"
    ),
)


@dataclass
class StreamingSession:
    stream: object
    last_partial_text: str = ""
    last_published_partial_text: str = ""
    latest_traceparent: str | None = None
    last_partial_published_at: float = 0.0
    last_seen_monotonic: float = field(default_factory=time.monotonic)


def rabbitmq_url() -> str:
    return f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASSWORD}@{RABBITMQ_HOST}:{RABBITMQ_PORT}/"


async def connect_broker() -> aio_pika.RobustConnection:
    while True:
        try:
            return await aio_pika.connect_robust(rabbitmq_url())
        except aio_pika.exceptions.AMQPConnectionError:
            print("[ASR] Waiting for RabbitMQ...", flush=True)
            await asyncio.sleep(2)


def load_transcript_replacements() -> list[tuple[re.Pattern, str]]:
    replacements = []
    for item in ASR_TRANSCRIPT_REPLACEMENTS.split(";"):
        if "=" not in item:
            continue
        source, target = item.split("=", 1)
        source = source.strip()
        target = target.strip()
        if not source or not target:
            continue
        replacements.append((re.compile(rf"\b{re.escape(source)}\b", re.IGNORECASE), target))
    return replacements


TRANSCRIPT_REPLACEMENTS = load_transcript_replacements()


def apply_transcript_replacements(transcript: str) -> str:
    corrected = transcript
    for pattern, replacement in TRANSCRIPT_REPLACEMENTS:
        corrected = pattern.sub(replacement, corrected)
    return " ".join(corrected.split())


def model_path(filename: str) -> str:
    path = SHERPA_MODEL_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing sherpa-onnx ASR model file: {path}")
    return str(path)


def load_recognizer():
    print(
        "[ASR] Loading sherpa-onnx streaming Zipformer "
        f"model_dir={SHERPA_MODEL_DIR} provider={SHERPA_PROVIDER} threads={SHERPA_NUM_THREADS}",
        flush=True,
    )
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=model_path(SHERPA_TOKENS),
        encoder=model_path(SHERPA_ENCODER),
        decoder=model_path(SHERPA_DECODER),
        joiner=model_path(SHERPA_JOINER),
        num_threads=SHERPA_NUM_THREADS,
        provider=SHERPA_PROVIDER,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        decoding_method=SHERPA_DECODING_METHOD,
        max_active_paths=SHERPA_MAX_ACTIVE_PATHS,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=SHERPA_RULE1_MIN_TRAILING_SILENCE,
        rule2_min_trailing_silence=SHERPA_RULE2_MIN_TRAILING_SILENCE,
        rule3_min_utterance_length=SHERPA_RULE3_MIN_UTTERANCE_LENGTH,
    )
    print("[ASR] sherpa-onnx recognizer ready", flush=True)
    return recognizer


def pcm16_to_float32(body: bytes) -> np.ndarray:
    usable_length = len(body) - (len(body) % 2)
    if usable_length <= 0:
        return np.empty(0, dtype=np.float32)
    samples = np.frombuffer(body[:usable_length], dtype="<i2")
    return samples.astype(np.float32) / 32768.0


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


def get_result_text(recognizer, stream) -> str:
    result = recognizer.get_result(stream)
    text = getattr(result, "text", result)
    return apply_transcript_replacements(str(text).strip())


def decode_ready(recognizer, stream) -> None:
    while recognizer.is_ready(stream):
        if hasattr(recognizer, "decode_stream"):
            recognizer.decode_stream(stream)
        else:
            recognizer.decode_streams([stream])


def session_for(sessions: Dict[str, StreamingSession], recognizer, user_id: str) -> StreamingSession:
    session = sessions.get(user_id)
    if session:
        return session

    session = StreamingSession(stream=recognizer.create_stream())
    sessions[user_id] = session
    print(f"[ASR] Created streaming session user_id={user_id}", flush=True)
    return session


def reset_stream(sessions: Dict[str, StreamingSession], recognizer, user_id: str, traceparent: str | None) -> None:
    session = sessions[user_id]
    if hasattr(recognizer, "reset"):
        recognizer.reset(session.stream)
        stream = session.stream
    else:
        stream = recognizer.create_stream()

    sessions[user_id] = StreamingSession(
        stream=stream,
        latest_traceparent=traceparent,
        last_seen_monotonic=time.monotonic(),
    )


async def publish_partial(
    exchange: aio_pika.Exchange,
    transcript: str,
    correlation_id: str,
    traceparent: str | None,
) -> None:
    payload = {
        "type": "partial",
        "text": transcript,
        "user_id": correlation_id,
    }
    await exchange.publish(
        aio_pika.Message(
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            delivery_mode=aio_pika.DeliveryMode.NOT_PERSISTENT,
            correlation_id=correlation_id,
            content_type="application/json",
            headers=trace_headers(correlation_id, traceparent),
        ),
        routing_key="",
    )
    print(
        f"[ASR] Published partial transcript: {transcript!r}; "
        f"correlation_id={correlation_id}; traceparent={traceparent}",
        flush=True,
    )


async def publish_final(
    channel: aio_pika.Channel,
    transcript: str,
    correlation_id: str,
    traceparent: str | None,
) -> None:
    await channel.default_exchange.publish(
        aio_pika.Message(
            transcript.encode("utf-8"),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            correlation_id=correlation_id,
            content_type="text/plain",
            headers=trace_headers(correlation_id, traceparent),
        ),
        routing_key=TEXT_RAG_QUEUE,
    )
    print(
        f"[ASR] Published final transcript: {transcript!r}; "
        f"correlation_id={correlation_id}; traceparent={traceparent}",
        flush=True,
    )


async def maybe_publish_partial(
    exchange: aio_pika.Exchange,
    session: StreamingSession,
    transcript: str,
    correlation_id: str,
) -> None:
    if not transcript or transcript == session.last_published_partial_text:
        return

    now = time.monotonic()
    if now - session.last_partial_published_at < PARTIAL_THROTTLE_SECONDS:
        return

    session.last_published_partial_text = transcript
    session.last_partial_published_at = now
    await publish_partial(exchange, transcript, correlation_id, session.latest_traceparent)


def cleanup_expired_sessions(sessions: Dict[str, StreamingSession]) -> None:
    now = time.monotonic()
    expired = [user_id for user_id, session in sessions.items() if now - session.last_seen_monotonic > SESSION_TTL_SECONDS]
    for user_id in expired:
        del sessions[user_id]
        print(f"[ASR] Expired streaming session user_id={user_id}", flush=True)


async def main() -> None:
    recognizer = load_recognizer()
    sessions: Dict[str, StreamingSession] = {}

    connection = await connect_broker()
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=128)

        incoming_queue = await channel.declare_queue(AUDIO_INCOMING_QUEUE, durable=True)
        await channel.declare_queue(TEXT_RAG_QUEUE, durable=True)
        live_exchange = await channel.declare_exchange(
            TEXT_ASR_LIVE_EXCHANGE,
            aio_pika.ExchangeType.FANOUT,
            durable=True,
        )

        print(
            f"[ASR] Streaming {AUDIO_INCOMING_QUEUE} as {SAMPLE_RATE} Hz PCM16 into sherpa-onnx...",
            flush=True,
        )

        async with incoming_queue.iterator() as queue_iter:
            async for message in queue_iter:
                correlation_id = message.correlation_id or header_as_text(message.headers, "user_id")
                traceparent = header_as_text(message.headers, "traceparent")
                if not correlation_id:
                    correlation_id = "__manual_smoke_test__"
                    print("[ASR] Received audio without correlation_id", flush=True)

                try:
                    session = session_for(sessions, recognizer, correlation_id)
                    session.latest_traceparent = traceparent or session.latest_traceparent
                    session.last_seen_monotonic = time.monotonic()

                    samples = pcm16_to_float32(message.body)
                    if samples.size == 0:
                        await message.ack()
                        continue

                    session.stream.accept_waveform(SAMPLE_RATE, samples)
                    decode_ready(recognizer, session.stream)

                    transcript = get_result_text(recognizer, session.stream)
                    session.last_partial_text = transcript
                    await maybe_publish_partial(live_exchange, session, transcript, correlation_id)

                    if recognizer.is_endpoint(session.stream):
                        final_transcript = transcript or session.last_published_partial_text
                        print(
                            f"[ASR] Endpoint detected; final={final_transcript!r}; "
                            f"correlation_id={correlation_id}; traceparent={session.latest_traceparent}",
                            flush=True,
                        )
                        if final_transcript:
                            await publish_final(channel, final_transcript, correlation_id, session.latest_traceparent)
                        reset_stream(sessions, recognizer, correlation_id, session.latest_traceparent)

                    cleanup_expired_sessions(sessions)
                    await message.ack()
                except Exception as exc:
                    print(f"[ASR] Failed to process streaming audio chunk: {exc}", flush=True)
                    await message.nack(requeue=True)


if __name__ == "__main__":
    asyncio.run(main())
