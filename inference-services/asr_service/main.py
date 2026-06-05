import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List

import aio_pika
import numpy as np
import webrtcvad
from faster_whisper import WhisperModel


RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "guest")

AUDIO_INCOMING_QUEUE = "audio.incoming.raw"
TEXT_LLM_QUEUE = "text.llm.processing"

ASR_MODEL_NAME = os.getenv("ASR_MODEL_NAME", "base.en")
ASR_DEVICE = os.getenv("ASR_DEVICE", "cuda")
ASR_COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "float16")
SAMPLE_RATE = int(os.getenv("ASR_SAMPLE_RATE", "16000"))
VAD_FRAME_MS = int(os.getenv("ASR_VAD_FRAME_MS", "20"))
VAD_AGGRESSIVENESS = int(os.getenv("ASR_VAD_AGGRESSIVENESS", "2"))
SILENCE_FLUSH_MS = int(os.getenv("ASR_SILENCE_FLUSH_MS", "350"))
MIN_SPEECH_MS = int(os.getenv("ASR_MIN_SPEECH_MS", "240"))
MAX_UTTERANCE_MS = int(os.getenv("ASR_MAX_UTTERANCE_MS", "12000"))
INACTIVITY_FLUSH_SECONDS = float(os.getenv("ASR_INACTIVITY_FLUSH_SECONDS", "0.6"))
ENERGY_SILENCE_THRESHOLD = float(os.getenv("ASR_ENERGY_SILENCE_THRESHOLD", "220"))
SESSION_TTL_SECONDS = int(os.getenv("ASR_SESSION_TTL_SECONDS", "120"))
ASR_INITIAL_PROMPT = os.getenv(
    "ASR_INITIAL_PROMPT",
    (
        "JarvisLabs, E2E Networks, GPU, GPUs, NVIDIA L4, A100, H100, RTX, CUDA, "
        "VRAM, cloud instance, inference, deployment, pricing, account, support."
    ),
)


@dataclass
class SessionAudioBuffer:
    pcm_chunks: List[np.ndarray] = field(default_factory=list)
    pending_pcm16: bytes = b""
    speech_started: bool = False
    speech_ms: int = 0
    trailing_silence_ms: int = 0
    total_audio_ms: int = 0
    received_frames: int = 0
    last_seen_monotonic: float = field(default_factory=time.monotonic)
    flush_task: asyncio.Task | None = None
    flush_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def rabbitmq_url() -> str:
    return f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASSWORD}@{RABBITMQ_HOST}:{RABBITMQ_PORT}/"


async def connect_broker() -> aio_pika.RobustConnection:
    while True:
        try:
            return await aio_pika.connect_robust(rabbitmq_url())
        except aio_pika.exceptions.AMQPConnectionError:
            print("[ASR] Waiting for RabbitMQ...", flush=True)
            await asyncio.sleep(2)


def load_asr_model() -> WhisperModel:
    print(
        f"[ASR] Loading faster-whisper model={ASR_MODEL_NAME} "
        f"device={ASR_DEVICE} compute_type={ASR_COMPUTE_TYPE}",
        flush=True,
    )
    return WhisperModel(ASR_MODEL_NAME, device=ASR_DEVICE, compute_type=ASR_COMPUTE_TYPE)


def pcm16_from_message(body: bytes) -> np.ndarray:
    usable_length = len(body) - (len(body) % 2)
    if usable_length <= 0:
        return np.empty(0, dtype=np.int16)
    return np.frombuffer(body[:usable_length], dtype="<i2").copy()


def frame_has_enough_energy(frame: bytes) -> bool:
    samples = np.frombuffer(frame, dtype="<i2").astype(np.float32)
    if samples.size == 0:
        return False
    rms = float(np.sqrt(np.mean(samples * samples)))
    return rms >= ENERGY_SILENCE_THRESHOLD


def update_vad_state(session: SessionAudioBuffer, vad: webrtcvad.Vad, pcm16: np.ndarray) -> None:
    if pcm16.size == 0:
        return

    session.pcm_chunks.append(pcm16)
    session.received_frames += 1
    session.total_audio_ms += int(pcm16.size / SAMPLE_RATE * 1000)
    combined = session.pending_pcm16 + pcm16.tobytes()

    frame_samples = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)
    frame_bytes = frame_samples * 2
    consumed_until = 0

    for offset in range(0, len(combined) - frame_bytes + 1, frame_bytes):
        frame = combined[offset : offset + frame_bytes]
        consumed_until = offset + frame_bytes
        is_speech = frame_has_enough_energy(frame) and vad.is_speech(frame, SAMPLE_RATE)

        if is_speech:
            session.speech_started = True
            session.speech_ms += VAD_FRAME_MS
            session.trailing_silence_ms = 0
        elif session.speech_started:
            session.trailing_silence_ms += VAD_FRAME_MS

    session.pending_pcm16 = combined[consumed_until:]
    session.last_seen_monotonic = time.monotonic()


def should_flush(session: SessionAudioBuffer) -> bool:
    if not session.speech_started:
        return session.total_audio_ms >= MAX_UTTERANCE_MS

    has_min_speech = session.speech_ms >= MIN_SPEECH_MS
    has_pause = session.trailing_silence_ms >= SILENCE_FLUSH_MS
    hit_max = session.total_audio_ms >= MAX_UTTERANCE_MS
    return (has_min_speech and has_pause) or hit_max


def reset_utterance(session: SessionAudioBuffer) -> None:
    session.pcm_chunks.clear()
    session.pending_pcm16 = b""
    session.speech_started = False
    session.speech_ms = 0
    session.trailing_silence_ms = 0
    session.total_audio_ms = 0
    session.received_frames = 0
    session.last_seen_monotonic = time.monotonic()


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


def transcribe_buffer(model: WhisperModel, session: SessionAudioBuffer) -> str:
    if not session.pcm_chunks:
        return ""

    pcm16 = np.concatenate(session.pcm_chunks)
    if pcm16.size == 0:
        return ""

    audio_float32 = pcm16.astype(np.float32) / 32768.0
    segments, _ = model.transcribe(
        audio_float32,
        language="en",
        beam_size=1,
        best_of=1,
        vad_filter=False,
        condition_on_previous_text=False,
        initial_prompt=ASR_INITIAL_PROMPT,
        without_timestamps=True,
        temperature=0.0,
    )
    transcript = " ".join(segment.text.strip() for segment in segments).strip()
    return " ".join(transcript.split())


async def publish_transcript(
    channel: aio_pika.Channel,
    transcript: str,
    correlation_id: str | None,
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
        routing_key=TEXT_LLM_QUEUE,
    )


async def flush_session(
    channel: aio_pika.Channel,
    model: WhisperModel,
    session: SessionAudioBuffer,
    correlation_id: str | None,
    traceparent: str | None,
    reason: str,
) -> None:
    async with session.flush_lock:
        if not session.pcm_chunks or session.speech_ms < MIN_SPEECH_MS:
            if session.total_audio_ms >= MAX_UTTERANCE_MS:
                reset_utterance(session)
            return

        buffered_ms = session.total_audio_ms
        transcript = await asyncio.to_thread(transcribe_buffer, model, session)
        reset_utterance(session)

        if transcript:
            await publish_transcript(channel, transcript, correlation_id, traceparent)
            print(
                f"[ASR] Published transcript via {reason}: {transcript!r}; "
                f"buffered_ms={buffered_ms}; correlation_id={correlation_id}; traceparent={traceparent}",
                flush=True,
            )
        else:
            print(
                f"[ASR] Empty transcript after {reason} flush; "
                f"correlation_id={correlation_id}; traceparent={traceparent}",
                flush=True,
            )


async def inactivity_flush_later(
    channel: aio_pika.Channel,
    model: WhisperModel,
    session: SessionAudioBuffer,
    correlation_id: str | None,
    traceparent: str | None,
) -> None:
    await asyncio.sleep(INACTIVITY_FLUSH_SECONDS)
    if time.monotonic() - session.last_seen_monotonic >= INACTIVITY_FLUSH_SECONDS:
        await flush_session(channel, model, session, correlation_id, traceparent, "inactivity")


def cancel_flush_task(session: SessionAudioBuffer) -> None:
    if session.flush_task and not session.flush_task.done():
        session.flush_task.cancel()
    session.flush_task = None


def cleanup_expired_sessions(sessions: Dict[str, SessionAudioBuffer]) -> None:
    now = time.monotonic()
    expired = [key for key, value in sessions.items() if now - value.last_seen_monotonic > SESSION_TTL_SECONDS]
    for key in expired:
        print(f"[ASR] Expiring inactive session buffer correlation_id={key}", flush=True)
        cancel_flush_task(sessions[key])
        del sessions[key]


async def main() -> None:
    model = load_asr_model()
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    sessions: Dict[str, SessionAudioBuffer] = {}

    connection = await connect_broker()
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=32)

        incoming_queue = await channel.declare_queue(AUDIO_INCOMING_QUEUE, durable=True)
        await channel.declare_queue(TEXT_LLM_QUEUE, durable=True)

        print(
            f"[ASR] Listening to {AUDIO_INCOMING_QUEUE} as {SAMPLE_RATE} Hz mono PCM16 "
            f"({VAD_FRAME_MS}ms VAD frames)...",
            flush=True,
        )

        async with incoming_queue.iterator() as queue_iter:
            async for message in queue_iter:
                try:
                    correlation_id = message.correlation_id
                    traceparent = header_as_text(message.headers, "traceparent")
                    if not correlation_id:
                        print("[ASR] Received audio without correlation_id", flush=True)

                    session_key = correlation_id or "__manual_smoke_test__"
                    session = sessions.setdefault(session_key, SessionAudioBuffer())
                    cancel_flush_task(session)

                    pcm16 = pcm16_from_message(message.body)
                    update_vad_state(session, vad, pcm16)

                    if session.received_frames <= 3:
                        print(
                            f"[ASR] Received PCM samples={pcm16.size}; bytes={len(message.body)}; "
                            f"correlation_id={correlation_id}; traceparent={traceparent}",
                            flush=True,
                        )

                    if should_flush(session):
                        await flush_session(channel, model, session, correlation_id, traceparent, "vad")
                    elif session.speech_started:
                        session.flush_task = asyncio.create_task(
                            inactivity_flush_later(channel, model, session, correlation_id, traceparent)
                        )

                    cleanup_expired_sessions(sessions)
                    await message.ack()
                except Exception as exc:
                    print(f"[ASR] Failed to process audio chunk: {exc}", flush=True)
                    await message.nack(requeue=True)


if __name__ == "__main__":
    asyncio.run(main())
