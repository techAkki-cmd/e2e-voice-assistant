import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict

import aio_pika
import numpy as np
import torch
from melo.api import TTS


RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "guest")

TEXT_TTS_QUEUE = "text.tts.processing"
AUDIO_OUTGOING_QUEUE = "audio.outgoing.stream"
CONTROL_SIGNALS_EXCHANGE = "control.signals"

TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "EN")
TTS_SPEAKER = os.getenv("TTS_SPEAKER", "EN-US")
TTS_DEVICE = os.getenv("TTS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.05"))
TTS_OUTPUT_SAMPLE_RATE = int(os.getenv("TTS_OUTPUT_SAMPLE_RATE", "44100"))
TTS_INACTIVITY_FLUSH_SECONDS = float(os.getenv("TTS_INACTIVITY_FLUSH_SECONDS", "1.2"))
TTS_SESSION_TTL_SECONDS = float(os.getenv("TTS_SESSION_TTL_SECONDS", "120"))
INTERRUPT_DRAIN_SECONDS = float(os.getenv("TTS_INTERRUPT_DRAIN_SECONDS", "0.8"))
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
    sample_rate = int(model.hps.data.sampling_rate)
    print(
        f"[TTS] Using speaker={TTS_SPEAKER} speaker_id={speaker_id} "
        f"native_sample_rate={sample_rate} output_sample_rate={TTS_OUTPUT_SAMPLE_RATE}",
        flush=True,
    )
    return model, speaker_id, sample_rate


def resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or audio.size <= 1:
        return audio.astype(np.float32, copy=False)

    target_length = max(1, int(round(audio.size * target_rate / source_rate)))
    source_positions = np.arange(audio.size, dtype=np.float32)
    target_positions = np.linspace(0, audio.size - 1, target_length, dtype=np.float32)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def normalize_audio_array(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return np.zeros(1, dtype=np.int16)
    audio = np.nan_to_num(audio)
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak
    return np.clip(audio * 32767.0, -32768, 32767).astype("<i2")


def synthesize_clause_to_pcm16(model: TTS, speaker_id: int, sample_rate: int, clause: str) -> bytes:
    with torch.inference_mode():
        audio = model.tts_to_file(clause, speaker_id, output_path=None, speed=TTS_SPEED, quiet=True)
    audio = resample_linear(np.asarray(audio, dtype=np.float32), sample_rate, TTS_OUTPUT_SAMPLE_RATE)
    pcm16 = normalize_audio_array(audio)
    return pcm16.tobytes()


def cancel_flush_task(state: TextBuffer) -> None:
    if state.flush_task and not state.flush_task.done():
        state.flush_task.cancel()
    state.flush_task = None


def is_interrupted(session_key: str, interrupted_at: Dict[str, float]) -> bool:
    interrupted_time = interrupted_at.get(session_key)
    return interrupted_time is not None and time.monotonic() - interrupted_time <= INTERRUPT_DRAIN_SECONDS


async def publish_pcm(channel: aio_pika.Channel, pcm_bytes: bytes, correlation_id: str | None) -> None:
    await channel.default_exchange.publish(
        aio_pika.Message(
            pcm_bytes,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            correlation_id=correlation_id,
            content_type=f"audio/pcm;rate={TTS_OUTPUT_SAMPLE_RATE}",
        ),
        routing_key=AUDIO_OUTGOING_QUEUE,
    )


async def synthesize_and_publish(
    channel: aio_pika.Channel,
    model: TTS,
    speaker_id: int,
    sample_rate: int,
    clause: str,
    session_key: str,
    correlation_id: str | None,
    interrupted_at: Dict[str, float],
    interrupt_versions: Dict[str, int],
) -> None:
    clause = " ".join(clause.split())
    if not clause:
        return
    if is_interrupted(session_key, interrupted_at):
        print(f"[TTS] Dropping clause before synthesis due to kill correlation_id={correlation_id}", flush=True)
        return

    start_version = interrupt_versions.get(session_key, 0)
    pcm_bytes = await asyncio.to_thread(synthesize_clause_to_pcm16, model, speaker_id, sample_rate, clause)

    if interrupt_versions.get(session_key, 0) != start_version or is_interrupted(session_key, interrupted_at):
        print(f"[TTS] Dropping synthesized PCM due to kill correlation_id={correlation_id}", flush=True)
        return

    await publish_pcm(channel, pcm_bytes, correlation_id)
    print(
        f"[TTS] Published PCM bytes={len(pcm_bytes)} for clause={clause!r}; correlation_id={correlation_id}",
        flush=True,
    )


async def flush_buffer(
    channel: aio_pika.Channel,
    model: TTS,
    speaker_id: int,
    sample_rate: int,
    buffers: Dict[str, TextBuffer],
    session_key: str,
    correlation_id: str | None,
    interrupted_at: Dict[str, float],
    interrupt_versions: Dict[str, int],
) -> None:
    state = buffers.get(session_key)
    if not state or not state.text.strip():
        return
    if is_interrupted(session_key, interrupted_at):
        state.text = ""
        return

    clause = state.text.strip()
    state.text = ""
    await synthesize_and_publish(
        channel,
        model,
        speaker_id,
        sample_rate,
        clause,
        session_key,
        correlation_id,
        interrupted_at,
        interrupt_versions,
    )


async def inactivity_flush_later(
    channel: aio_pika.Channel,
    model: TTS,
    speaker_id: int,
    sample_rate: int,
    buffers: Dict[str, TextBuffer],
    session_key: str,
    correlation_id: str | None,
    interrupted_at: Dict[str, float],
    interrupt_versions: Dict[str, int],
) -> None:
    await asyncio.sleep(TTS_INACTIVITY_FLUSH_SECONDS)
    state = buffers.get(session_key)
    if state and time.monotonic() - state.last_seen_monotonic >= TTS_INACTIVITY_FLUSH_SECONDS:
        await flush_buffer(
            channel,
            model,
            speaker_id,
            sample_rate,
            buffers,
            session_key,
            correlation_id,
            interrupted_at,
            interrupt_versions,
        )


def cleanup_expired_state(buffers: Dict[str, TextBuffer], interrupted_at: Dict[str, float]) -> None:
    now = time.monotonic()
    for key in [key for key, state in buffers.items() if now - state.last_seen_monotonic > TTS_SESSION_TTL_SECONDS]:
        cancel_flush_task(buffers[key])
        del buffers[key]

    for key in [key for key, seen_at in interrupted_at.items() if now - seen_at > INTERRUPT_DRAIN_SECONDS]:
        interrupted_at.pop(key, None)


async def consume_control_signals(
    channel: aio_pika.Channel,
    buffers: Dict[str, TextBuffer],
    interrupted_at: Dict[str, float],
    interrupt_versions: Dict[str, int],
) -> None:
    exchange = await channel.declare_exchange(CONTROL_SIGNALS_EXCHANGE, aio_pika.ExchangeType.FANOUT, durable=True)
    queue = await channel.declare_queue(exclusive=True, auto_delete=True)
    await queue.bind(exchange)
    print(f"[TTS] Listening to fanout exchange {CONTROL_SIGNALS_EXCHANGE}...", flush=True)

    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            async with message.process(ignore_processed=True):
                try:
                    payload = json.loads(message.body.decode("utf-8"))
                except json.JSONDecodeError:
                    print("[TTS] Ignored malformed control signal", flush=True)
                    continue

                if payload.get("action") != "kill":
                    continue

                correlation_id = payload.get("correlation_id")
                if not correlation_id:
                    continue

                interrupted_at[correlation_id] = time.monotonic()
                interrupt_versions[correlation_id] = interrupt_versions.get(correlation_id, 0) + 1
                state = buffers.get(correlation_id)
                if state:
                    cancel_flush_task(state)
                    state.text = ""
                print(f"[TTS] Kill signal received; cleared buffer correlation_id={correlation_id}", flush=True)


async def main() -> None:
    model, speaker_id, sample_rate = load_tts_model()
    buffers: Dict[str, TextBuffer] = {}
    interrupted_at: Dict[str, float] = {}
    interrupt_versions: Dict[str, int] = {}
    connection = await connect_broker()

    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)

        control_channel = await connection.channel()
        control_task = asyncio.create_task(
            consume_control_signals(control_channel, buffers, interrupted_at, interrupt_versions)
        )

        incoming_queue = await channel.declare_queue(TEXT_TTS_QUEUE, durable=True)
        await channel.declare_queue(AUDIO_OUTGOING_QUEUE, durable=True)

        print(f"[TTS] Listening to {TEXT_TTS_QUEUE}...", flush=True)

        try:
            async with incoming_queue.iterator() as queue_iter:
                async for message in queue_iter:
                    try:
                        correlation_id = message.correlation_id
                        if not correlation_id:
                            print("[TTS] Received text chunk without correlation_id", flush=True)

                        session_key = correlation_id or "__manual_smoke_test__"
                        state = buffers.setdefault(session_key, TextBuffer())
                        cancel_flush_task(state)

                        if is_interrupted(session_key, interrupted_at):
                            state.text = ""
                            await message.ack()
                            continue

                        chunk = message.body.decode("utf-8", errors="replace")
                        state.text += chunk
                        state.last_seen_monotonic = time.monotonic()
                        print(f"[TTS] Buffered text chunk: {chunk!r}; correlation_id={correlation_id}", flush=True)

                        while not is_interrupted(session_key, interrupted_at):
                            match = CLAUSE_PATTERN.match(state.text)
                            if not match:
                                break
                            clause = match.group(1).strip()
                            state.text = state.text[match.end() :]
                            await synthesize_and_publish(
                                channel,
                                model,
                                speaker_id,
                                sample_rate,
                                clause,
                                session_key,
                                correlation_id,
                                interrupted_at,
                                interrupt_versions,
                            )

                        if is_interrupted(session_key, interrupted_at):
                            state.text = ""
                        elif state.text.strip():
                            state.flush_task = asyncio.create_task(
                                inactivity_flush_later(
                                    channel,
                                    model,
                                    speaker_id,
                                    sample_rate,
                                    buffers,
                                    session_key,
                                    correlation_id,
                                    interrupted_at,
                                    interrupt_versions,
                                )
                            )

                        cleanup_expired_state(buffers, interrupted_at)
                        await message.ack()
                    except Exception as exc:
                        print(f"[TTS] Failed to synthesize text chunk: {exc}", flush=True)
                        await message.nack(requeue=False)
        finally:
            control_task.cancel()
            await asyncio.gather(control_task, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
