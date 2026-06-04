import asyncio
import io
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict

import aio_pika
import numpy as np
import torch
from melo.api import TTS
from scipy.io import wavfile


RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "guest")

TEXT_TTS_QUEUE = "text.tts.processing"
AUDIO_OUTGOING_QUEUE = "audio.outgoing.stream"

TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "EN")
TTS_SPEAKER = os.getenv("TTS_SPEAKER", "EN-US")
TTS_DEVICE = os.getenv("TTS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.05"))
TTS_INACTIVITY_FLUSH_SECONDS = float(os.getenv("TTS_INACTIVITY_FLUSH_SECONDS", "1.2"))
TTS_SESSION_TTL_SECONDS = float(os.getenv("TTS_SESSION_TTL_SECONDS", "120"))
CLAUSE_PATTERN = re.compile(r"^(.+?[,.!?])(?:\s+|$)", re.DOTALL)


@dataclass
class TextBuffer:
    text: str = ""
    last_seen_monotonic: float = field(default_factory=time.monotonic)
    flush_task: asyncio.Task | None = None


def rabbitmq_url() -> str:
    return f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASSWORD}@{RABBITMQ_HOST}:{RABBITMQ_PORT}/"


async def connect_broker() -> aio_pika.RobustConnection:
    while True:
        try:
            return await aio_pika.connect_robust(rabbitmq_url())
        except aio_pika.exceptions.AMQPConnectionError:
            print("[TTS] Waiting for RabbitMQ...", flush=True)
            await asyncio.sleep(2)


def load_tts_model():
    print(f"[TTS] Loading MeloTTS language={TTS_LANGUAGE} device={TTS_DEVICE}", flush=True)
    model = TTS(language=TTS_LANGUAGE, device=TTS_DEVICE)
    speaker_ids = model.hps.data.spk2id
    if not isinstance(speaker_ids, dict):
        speaker_ids = vars(speaker_ids)
    speaker_id = speaker_ids.get(TTS_SPEAKER) or speaker_ids.get("EN-Default") or next(iter(speaker_ids.values()))
    print(f"[TTS] Using speaker={TTS_SPEAKER} speaker_id={speaker_id}", flush=True)
    return model, speaker_id


def normalize_audio_array(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return np.zeros(1, dtype=np.int16)
    audio = np.nan_to_num(audio)
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak
    return np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)


def synthesize_clause_to_wav(model: TTS, speaker_id: int, clause: str) -> bytes:
    with torch.inference_mode():
        audio = model.tts_to_file(clause, speaker_id, output_path=None, speed=TTS_SPEED, quiet=True)
    pcm16 = normalize_audio_array(audio)
    sample_rate = int(model.hps.data.sampling_rate)
    output = io.BytesIO()
    wavfile.write(output, sample_rate, pcm16)
    return output.getvalue()


async def publish_wav(channel: aio_pika.Channel, wav_bytes: bytes, correlation_id: str | None) -> None:
    await channel.default_exchange.publish(
        aio_pika.Message(
            wav_bytes,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            correlation_id=correlation_id,
            content_type="audio/wav",
        ),
        routing_key=AUDIO_OUTGOING_QUEUE,
    )


async def synthesize_and_publish(
    channel: aio_pika.Channel,
    model: TTS,
    speaker_id: int,
    clause: str,
    correlation_id: str | None,
) -> None:
    clause = " ".join(clause.split())
    if not clause:
        return
    wav_bytes = await asyncio.to_thread(synthesize_clause_to_wav, model, speaker_id, clause)
    await publish_wav(channel, wav_bytes, correlation_id)
    print(
        f"[TTS] Published WAV bytes={len(wav_bytes)} for clause={clause!r}; correlation_id={correlation_id}",
        flush=True,
    )


async def flush_buffer(
    channel: aio_pika.Channel,
    model: TTS,
    speaker_id: int,
    buffers: Dict[str, TextBuffer],
    session_key: str,
    correlation_id: str | None,
) -> None:
    state = buffers.get(session_key)
    if not state or not state.text.strip():
        return

    clause = state.text.strip()
    state.text = ""
    await synthesize_and_publish(channel, model, speaker_id, clause, correlation_id)


async def inactivity_flush_later(
    channel: aio_pika.Channel,
    model: TTS,
    speaker_id: int,
    buffers: Dict[str, TextBuffer],
    session_key: str,
    correlation_id: str | None,
) -> None:
    await asyncio.sleep(TTS_INACTIVITY_FLUSH_SECONDS)
    state = buffers.get(session_key)
    if state and time.monotonic() - state.last_seen_monotonic >= TTS_INACTIVITY_FLUSH_SECONDS:
        await flush_buffer(channel, model, speaker_id, buffers, session_key, correlation_id)


def cancel_flush_task(state: TextBuffer) -> None:
    if state.flush_task and not state.flush_task.done():
        state.flush_task.cancel()


def cleanup_expired_buffers(buffers: Dict[str, TextBuffer]) -> None:
    now = time.monotonic()
    for key in [key for key, state in buffers.items() if now - state.last_seen_monotonic > TTS_SESSION_TTL_SECONDS]:
        cancel_flush_task(buffers[key])
        del buffers[key]


async def main() -> None:
    model, speaker_id = load_tts_model()
    buffers: Dict[str, TextBuffer] = {}
    connection = await connect_broker()

    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)

        incoming_queue = await channel.declare_queue(TEXT_TTS_QUEUE, durable=True)
        await channel.declare_queue(AUDIO_OUTGOING_QUEUE, durable=True)

        print(f"[TTS] Listening to {TEXT_TTS_QUEUE}...", flush=True)

        async with incoming_queue.iterator() as queue_iter:
            async for message in queue_iter:
                try:
                    correlation_id = message.correlation_id
                    if not correlation_id:
                        print("[TTS] Received text chunk without correlation_id", flush=True)

                    session_key = correlation_id or "__manual_smoke_test__"
                    state = buffers.setdefault(session_key, TextBuffer())
                    cancel_flush_task(state)

                    chunk = message.body.decode("utf-8", errors="replace")
                    state.text += chunk
                    state.last_seen_monotonic = time.monotonic()
                    print(f"[TTS] Buffered text chunk: {chunk!r}; correlation_id={correlation_id}", flush=True)

                    while True:
                        match = CLAUSE_PATTERN.match(state.text)
                        if not match:
                            break
                        clause = match.group(1).strip()
                        state.text = state.text[match.end() :]
                        await synthesize_and_publish(channel, model, speaker_id, clause, correlation_id)

                    if state.text.strip():
                        state.flush_task = asyncio.create_task(
                            inactivity_flush_later(channel, model, speaker_id, buffers, session_key, correlation_id)
                        )

                    cleanup_expired_buffers(buffers)
                    await message.ack()
                except Exception as exc:
                    print(f"[TTS] Failed to synthesize text chunk: {exc}", flush=True)
                    await message.nack(requeue=True)


if __name__ == "__main__":
    asyncio.run(main())
