import asyncio
import json
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict

import aio_pika
import numpy as np
import sherpa_onnx
import webrtcvad
from faster_whisper import WhisperModel


RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "guest")

AUDIO_INCOMING_QUEUE = "audio.incoming.raw"
TEXT_RAG_QUEUE = "text.rag.processing"
TEXT_ASR_LIVE_EXCHANGE = "text.asr.live"

SAMPLE_RATE = int(os.getenv("ASR_SAMPLE_RATE", "16000"))
SESSION_TTL_SECONDS = int(os.getenv("ASR_SESSION_TTL_SECONDS", "120"))
SPEECH_START_FRAMES = int(os.getenv("ASR_SPEECH_START_FRAMES", "4"))
PRE_ROLL_MS = int(os.getenv("ASR_PRE_ROLL_MS", "250"))
TRAILING_SILENCE_MS = int(os.getenv("ASR_TRAILING_SILENCE_MS", "1350"))
MIN_SPEECH_MS = int(os.getenv("ASR_MIN_SPEECH_MS", "550"))
MAX_UTTERANCE_SECONDS = float(os.getenv("ASR_MAX_UTTERANCE_SECONDS", "12"))
MIN_FINAL_CHARS = int(os.getenv("ASR_MIN_FINAL_CHARS", "3"))
MIN_FINAL_WORDS = int(os.getenv("ASR_MIN_FINAL_WORDS", "1"))
VAD_GAIN = float(os.getenv("ASR_VAD_GAIN", "3.0"))
VAD_AGGRESSIVENESS = int(os.getenv("ASR_VAD_AGGRESSIVENESS", "1"))
MIN_VAD_RMS = float(os.getenv("ASR_MIN_VAD_RMS", "0.0025"))

WHISPER_MODEL_NAME = os.getenv("ASR_WHISPER_MODEL", "small.en")
WHISPER_DEVICE = os.getenv("ASR_WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("ASR_WHISPER_COMPUTE_TYPE", "float16")
WHISPER_BEAM_SIZE = int(os.getenv("ASR_WHISPER_BEAM_SIZE", "5"))
WHISPER_NO_SPEECH_THRESHOLD = float(os.getenv("ASR_WHISPER_NO_SPEECH_THRESHOLD", "0.72"))
WHISPER_LOGPROB_THRESHOLD = float(os.getenv("ASR_WHISPER_LOGPROB_THRESHOLD", "-1.15"))

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
    "L and M=LLM;L.N.=LLM;GUE=GUI;gooey=GUI;G U I=GUI;G P U=GPU",
)
ASR_REJECT_FINAL_PATTERNS = os.getenv(
    "ASR_REJECT_FINAL_PATTERNS",
    (
        r"^\s*(?:thanks for watching|thank you for watching|please subscribe|subscribe)\b;"
        r"\bcommon terms\b;\bjupyter notebook\.?\s+common terms\b;"
        r"\bciao\s+out\s+of\s+base\b;\bout\s+of\s+base\b;"
        r"^(.{1,24})(?:\s+\1){2,}$"
    ),
)


@dataclass
class StreamingSession:
    stream: object
    pre_roll_frames: Deque[bytes] = field(default_factory=deque)
    utterance_buffer: bytearray = field(default_factory=bytearray)
    in_speech: bool = False
    consecutive_voiced_frames: int = 0
    pending_voiced_samples: int = 0
    speech_samples: int = 0
    trailing_silence_samples: int = 0
    utterance_samples: int = 0
    latest_traceparent: str | None = None
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
REJECT_FINAL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in ASR_REJECT_FINAL_PATTERNS.split(";")
    if pattern.strip()
]


def apply_transcript_replacements(transcript: str) -> str:
    corrected = transcript
    for pattern, replacement in TRANSCRIPT_REPLACEMENTS:
        corrected = pattern.sub(replacement, corrected)
    return " ".join(corrected.split())


def should_publish_final(transcript: str, segments: list, speech_ms: float) -> bool:
    normalized = " ".join(transcript.strip().split())
    if len(normalized) < MIN_FINAL_CHARS:
        return False
    if len(normalized.split()) < MIN_FINAL_WORDS:
        return False
    if speech_ms < MIN_SPEECH_MS:
        return False
    if any(pattern.search(normalized) for pattern in REJECT_FINAL_PATTERNS):
        return False
    if normalized.endswith("..."):
        return False

    if not segments:
        return False

    worst_no_speech = max(float(getattr(segment, "no_speech_prob", 0.0)) for segment in segments)
    avg_logprob = sum(float(getattr(segment, "avg_logprob", 0.0)) for segment in segments) / len(segments)
    return not (worst_no_speech >= WHISPER_NO_SPEECH_THRESHOLD and avg_logprob <= WHISPER_LOGPROB_THRESHOLD)


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


def load_whisper_model() -> WhisperModel:
    print(
        "[ASR] Loading authoritative Whisper final ASR "
        f"model={WHISPER_MODEL_NAME} device={WHISPER_DEVICE} compute_type={WHISPER_COMPUTE_TYPE}",
        flush=True,
    )
    model = WhisperModel(
        WHISPER_MODEL_NAME,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
        download_root=os.getenv("HF_HOME", "/models/huggingface"),
    )
    print("[ASR] Whisper final ASR ready", flush=True)
    return model


def pcm16_to_float32(body: bytes) -> np.ndarray:
    usable_length = len(body) - (len(body) % 2)
    if usable_length <= 0:
        return np.empty(0, dtype=np.float32)
    audio_int16 = np.frombuffer(body[:usable_length], dtype="<i2")
    return audio_int16.astype(np.float32) / 32768.0


def boost_pcm16_for_vad(raw_audio_bytes: bytes) -> bytes:
    usable_length = len(raw_audio_bytes) - (len(raw_audio_bytes) % 2)
    if usable_length <= 0:
        return b""
    audio_int16 = np.frombuffer(raw_audio_bytes[:usable_length], dtype="<i2")
    boosted_audio = np.clip(
        audio_int16.astype(np.int32) * VAD_GAIN,
        -32768,
        32767,
    ).astype(np.int16)
    return boosted_audio.tobytes()


def vad_is_speech(vad: webrtcvad.Vad, body: bytes) -> bool:
    boosted_body = boost_pcm16_for_vad(body)
    if not boosted_body:
        return False
    try:
        return vad.is_speech(boosted_body, SAMPLE_RATE)
    except webrtcvad.Error as exc:
        print(
            f"[ASR] Invalid VAD frame ignored; bytes={len(boosted_body)}; "
            f"sample_rate={SAMPLE_RATE}; error={exc}",
            flush=True,
        )
        return False


def frame_rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    audio = samples.astype(np.float32, copy=False)
    return float(np.sqrt(np.mean(audio * audio)))


def is_voiced_frame(vad: webrtcvad.Vad, body: bytes, samples: np.ndarray) -> tuple[bool, float]:
    rms = frame_rms(samples)
    return vad_is_speech(vad, body) and rms >= MIN_VAD_RMS, rms


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


def reset_sherpa_stream(session: StreamingSession, recognizer) -> None:
    if hasattr(recognizer, "reset"):
        recognizer.reset(session.stream)
    else:
        session.stream = recognizer.create_stream()


def reset_speech_gate(session: StreamingSession) -> None:
    session.pre_roll_frames.clear()
    session.utterance_buffer = bytearray()
    session.in_speech = False
    session.consecutive_voiced_frames = 0
    session.pending_voiced_samples = 0
    session.speech_samples = 0
    session.trailing_silence_samples = 0
    session.utterance_samples = 0


def pre_roll_limit() -> int:
    frame_ms = 20
    return max(1, int(round(PRE_ROLL_MS / frame_ms)))


def update_speech_gate(
    session: StreamingSession,
    vad: webrtcvad.Vad,
    body: bytes,
    samples: np.ndarray,
    correlation_id: str,
) -> bool:
    voiced, rms = is_voiced_frame(vad, body, samples)
    started_this_frame = False
    session.pre_roll_frames.append(bytes(body))
    while len(session.pre_roll_frames) > pre_roll_limit():
        session.pre_roll_frames.popleft()

    if voiced:
        session.consecutive_voiced_frames += 1
        session.pending_voiced_samples += int(samples.size)
    else:
        session.consecutive_voiced_frames = 0
        session.pending_voiced_samples = 0

    if not session.in_speech and session.consecutive_voiced_frames >= SPEECH_START_FRAMES:
        session.in_speech = True
        session.utterance_buffer = bytearray().join(session.pre_roll_frames)
        session.utterance_samples = len(session.utterance_buffer) // 2
        session.speech_samples = session.pending_voiced_samples
        session.trailing_silence_samples = 0
        started_this_frame = True
        print(
            f"[ASR] Speech gate started; vad_mode={VAD_AGGRESSIVENESS}; rms={rms:.4f}; "
            f"min_vad_rms={MIN_VAD_RMS:.4f}; "
            f"correlation_id={correlation_id}; pre_roll_frames={len(session.pre_roll_frames)}",
            flush=True,
        )
        return False

    if not session.in_speech:
        return False

    if not started_this_frame:
        session.utterance_buffer.extend(body)
        session.utterance_samples += int(samples.size)

    if not started_this_frame:
        if voiced:
            session.speech_samples += int(samples.size)
            session.trailing_silence_samples = 0
        else:
            session.trailing_silence_samples += int(samples.size)

    trailing_ms = session.trailing_silence_samples * 1000 / SAMPLE_RATE
    utterance_seconds = session.utterance_samples / SAMPLE_RATE
    return trailing_ms >= TRAILING_SILENCE_MS or utterance_seconds >= MAX_UTTERANCE_SECONDS


def transcribe_whisper(model: WhisperModel, audio_bytes: bytes) -> tuple[str, list]:
    audio = pcm16_to_float32(audio_bytes)
    if audio.size == 0:
        return "", []

    segments_iter, _info = model.transcribe(
        audio,
        language="en",
        task="transcribe",
        beam_size=WHISPER_BEAM_SIZE,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    segments = list(segments_iter)
    transcript = apply_transcript_replacements(" ".join(segment.text.strip() for segment in segments).strip())
    return transcript, segments


async def publish_live_transcript(
    exchange: aio_pika.Exchange,
    transcript_type: str,
    transcript: str,
    correlation_id: str,
    traceparent: str | None,
) -> None:
    payload = {
        "type": transcript_type,
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
        f"[ASR] Published live {transcript_type} transcript: {transcript!r}; "
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
        f"[ASR] Published RAG final transcript: {transcript!r}; "
        f"correlation_id={correlation_id}; traceparent={traceparent}",
        flush=True,
    )


async def finalize_utterance(
    channel: aio_pika.Channel,
    live_exchange: aio_pika.Exchange,
    whisper_model: WhisperModel,
    session: StreamingSession,
    correlation_id: str,
) -> None:
    audio_bytes = bytes(session.utterance_buffer)
    speech_ms = session.speech_samples * 1000 / SAMPLE_RATE
    utterance_seconds = session.utterance_samples / SAMPLE_RATE
    print(
        f"[ASR] Speech gate finalized; speech_ms={speech_ms:.1f}; "
        f"audio_seconds={utterance_seconds:.2f}; correlation_id={correlation_id}; "
        f"traceparent={session.latest_traceparent}",
        flush=True,
    )

    if speech_ms < MIN_SPEECH_MS:
        print(
            f"[ASR] Dropped non-speech utterance; speech_ms={speech_ms:.1f}; "
            f"correlation_id={correlation_id}; traceparent={session.latest_traceparent}",
            flush=True,
        )
        reset_speech_gate(session)
        return

    transcript, segments = await asyncio.to_thread(transcribe_whisper, whisper_model, audio_bytes)
    print(
        f"[ASR] Whisper final transcript: {transcript!r}; segments={len(segments)}; "
        f"correlation_id={correlation_id}; traceparent={session.latest_traceparent}",
        flush=True,
    )

    if should_publish_final(transcript, segments, speech_ms):
        await publish_live_transcript(live_exchange, "final", transcript, correlation_id, session.latest_traceparent)
        await publish_final(channel, transcript, correlation_id, session.latest_traceparent)
    else:
        print(
            f"[ASR] Dropped non-speech utterance; transcript={transcript!r}; "
            f"speech_ms={speech_ms:.1f}; correlation_id={correlation_id}; "
            f"traceparent={session.latest_traceparent}",
            flush=True,
        )

    reset_speech_gate(session)


def cleanup_expired_sessions(sessions: Dict[str, StreamingSession]) -> None:
    now = time.monotonic()
    expired = [user_id for user_id, session in sessions.items() if now - session.last_seen_monotonic > SESSION_TTL_SECONDS]
    for user_id in expired:
        del sessions[user_id]
        print(f"[ASR] Expired streaming session user_id={user_id}", flush=True)


async def main() -> None:
    recognizer = load_recognizer()
    whisper_model = load_whisper_model()
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
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
            f"[ASR] Streaming {AUDIO_INCOMING_QUEUE} as {SAMPLE_RATE} Hz PCM16; "
            "sherpa endpointing enabled, Whisper finals authoritative.",
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

                    if recognizer.is_endpoint(session.stream):
                        reset_sherpa_stream(session, recognizer)

                    if update_speech_gate(session, vad, message.body, samples, correlation_id):
                        await finalize_utterance(channel, live_exchange, whisper_model, session, correlation_id)
                        reset_sherpa_stream(session, recognizer)

                    cleanup_expired_sessions(sessions)
                    await message.ack()
                except Exception as exc:
                    print(f"[ASR] Failed to process streaming audio chunk: {exc}", flush=True)
                    await message.nack(requeue=True)


if __name__ == "__main__":
    asyncio.run(main())
