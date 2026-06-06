import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable, List

import aio_pika
import asyncpg
import torch
from sentence_transformers import SentenceTransformer


RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "guest")

PGVECTOR_HOST = os.getenv("PGVECTOR_HOST", "pgvector")
PGVECTOR_PORT = int(os.getenv("PGVECTOR_PORT", "5432"))
PGVECTOR_DB = os.getenv("PGVECTOR_DB", "voice_rag")
PGVECTOR_USER = os.getenv("PGVECTOR_USER", "voice")
PGVECTOR_PASSWORD = os.getenv("PGVECTOR_PASSWORD", "voice")

TEXT_RAG_QUEUE = "text.rag.processing"
TEXT_LLM_QUEUE = "text.llm.processing"

EMBEDDING_MODEL_NAME = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_MIN_SIMILARITY = float(os.getenv("RAG_MIN_SIMILARITY", "0.30"))
SEED_CHUNKS_PATH = Path(os.getenv("RAG_SEED_CHUNKS_PATH", "/app/seed_chunks.json"))

KEYWORD_FALLBACK_TRIGGER = re.compile(
    r"\b(?:company|companies|document|documents|documented|docs|jarvislabs?|platform|"
    r"service|services|dashboard|gpu|gpus|llm|features?|polic(?:y|ies))\b",
    re.IGNORECASE,
)
POLICY_QUERY_PATTERN = re.compile(r"\bpolic(?:y|ies)\b", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
SOURCE_PRIORITIES = {
    "jarvislabs-platform-overview": 90,
    "company-docs": 80,
    "jarvislabs-dashboard-and-access": 70,
    "gpu-guidance-small-llm": 60,
    "deployment-support-scope": 50,
    "live-inventory-caveat": 40,
}
STOP_WORDS = {
    "a",
    "about",
    "and",
    "are",
    "can",
    "could",
    "do",
    "for",
    "have",
    "i",
    "is",
    "me",
    "of",
    "please",
    "tell",
    "the",
    "to",
    "what",
    "which",
    "you",
}
NON_CONTEXT_VALUES = {
    "miss",
    "none",
    "null",
    "no context",
    "no relevant context",
    "error",
}


def rabbitmq_url() -> str:
    return f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASSWORD}@{RABBITMQ_HOST}:{RABBITMQ_PORT}/"


async def connect_broker() -> aio_pika.RobustConnection:
    while True:
        try:
            return await aio_pika.connect_robust(rabbitmq_url())
        except aio_pika.exceptions.AMQPConnectionError:
            print("[RAG] Waiting for RabbitMQ...", flush=True)
            await asyncio.sleep(2)


async def connect_pg_pool() -> asyncpg.Pool:
    while True:
        try:
            pool = await asyncpg.create_pool(
                host=PGVECTOR_HOST,
                port=PGVECTOR_PORT,
                database=PGVECTOR_DB,
                user=PGVECTOR_USER,
                password=PGVECTOR_PASSWORD,
                min_size=1,
                max_size=4,
            )
            async with pool.acquire() as connection:
                await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            print(
                f"[RAG] Connected to pgvector host={PGVECTOR_HOST} port={PGVECTOR_PORT} db={PGVECTOR_DB}",
                flush=True,
            )
            return pool
        except (OSError, asyncpg.PostgresError) as exc:
            print(f"[RAG] Waiting for pgvector: {exc}", flush=True)
            await asyncio.sleep(2)


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as connection:
        await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_chunks (
                id bigserial PRIMARY KEY,
                source text,
                content text NOT NULL,
                source_hash text,
                embedding vector(384),
                created_at timestamptz DEFAULT now()
            )
            """
        )
        await connection.execute("ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS source_hash text")
        await connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS rag_chunks_source_hash_idx
            ON rag_chunks (source_hash)
            WHERE source_hash IS NOT NULL
            """
        )
        await connection.execute(
            """
            CREATE INDEX IF NOT EXISTS rag_chunks_embedding_ivfflat_idx
            ON rag_chunks
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
            """
        )


def load_embedding_model() -> SentenceTransformer:
    preferred_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[RAG] Loading embedding model={EMBEDDING_MODEL_NAME} device={preferred_device}", flush=True)
    try:
        return SentenceTransformer(EMBEDDING_MODEL_NAME, device=preferred_device)
    except RuntimeError as exc:
        if preferred_device != "cuda":
            raise
        print(f"[RAG] CUDA embedding load failed; falling back to CPU: {exc}", flush=True)
        return SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")


def embed_texts(model: SentenceTransformer, texts: Iterable[str]) -> List[List[float]]:
    embeddings = model.encode(
        list(texts),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.astype("float32").tolist()


def vector_literal(values: List[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def chunk_hash(source: str, content: str) -> str:
    return hashlib.sha256(f"{source}\n{content}".encode("utf-8")).hexdigest()


def tokenize(text: str) -> set[str]:
    return {token for token in TOKEN_PATTERN.findall(text.lower()) if token not in STOP_WORDS}


def source_priority(source: str) -> int:
    for prefix, priority in SOURCE_PRIORITIES.items():
        if source.startswith(prefix):
            return priority
    return 0


def keyword_query_terms(transcript: str) -> set[str]:
    terms = tokenize(transcript)
    lowered = transcript.lower()
    if re.search(r"\b(?:company|companies|document|documents|documented|docs)\b", lowered):
        terms.update({"jarvislabs", "cloud", "gpu", "compute", "platform"})
    if re.search(r"\b(?:service|services|platform)\b", lowered):
        terms.update({"jarvislabs", "support", "instance", "deployment", "dashboard"})
    if re.search(r"\b(?:feature|features|dashboard)\b", lowered):
        terms.update({"dashboard", "instance", "notebook", "terminal", "deployment"})
    if re.search(r"\b(?:gpu|gpus|llm)\b", lowered):
        terms.update({"gpu", "llm", "l4", "a100", "h100"})
    return terms


def keyword_score(transcript: str, source: str, content: str) -> int:
    query_terms = keyword_query_terms(transcript)
    if not query_terms:
        return 0

    content_terms = tokenize(f"{source} {content}")
    overlap = len(query_terms & content_terms)
    if POLICY_QUERY_PATTERN.search(transcript) and not POLICY_QUERY_PATTERN.search(content):
        return 0
    if overlap == 0:
        return 0
    return overlap * 100 + source_priority(source)


def sanitize_retrieved_context(retrieved_context: str) -> str:
    if not isinstance(retrieved_context, str):
        return ""

    context = retrieved_context.strip()
    if not context:
        return ""

    normalized = context.lower()
    if normalized in NON_CONTEXT_VALUES:
        return ""
    if normalized.startswith("error ") or normalized.startswith("error:"):
        return ""
    if "error connecting to db" in normalized or "error connecting to database" in normalized:
        return ""

    return context


async def seed_if_empty(pool: asyncpg.Pool, model: SentenceTransformer) -> None:
    async with pool.acquire() as connection:
        total_count = await connection.fetchval("SELECT count(*) FROM rag_chunks")
        embedded_count = await connection.fetchval("SELECT count(*) FROM rag_chunks WHERE embedding IS NOT NULL")
    if embedded_count:
        print(
            f"[RAG] Seed skipped; rag_chunks has rows={total_count}, embedded_rows={embedded_count}",
            flush=True,
        )
        return

    if total_count:
        print(
            f"[RAG] Found rag_chunks rows={total_count} but embedded_rows=0; inserting embedded seed chunks",
            flush=True,
        )

    seed_chunks = json.loads(SEED_CHUNKS_PATH.read_text(encoding="utf-8"))
    contents = [chunk["content"] for chunk in seed_chunks]
    embeddings = await asyncio.to_thread(embed_texts, model, contents)

    async with pool.acquire() as connection:
        async with connection.transaction():
            for chunk, embedding in zip(seed_chunks, embeddings):
                await connection.execute(
                    """
                    INSERT INTO rag_chunks (source, content, source_hash, embedding)
                    VALUES ($1, $2, $3, $4::vector)
                    ON CONFLICT (source_hash) WHERE source_hash IS NOT NULL DO NOTHING
                    """,
                    chunk.get("source", "seed"),
                    chunk["content"],
                    chunk_hash(chunk.get("source", "seed"), chunk["content"]),
                    vector_literal(embedding),
                )

    print(f"[RAG] Seeded rag_chunks rows={len(seed_chunks)}", flush=True)


def header_as_text(headers: dict | None, name: str) -> str | None:
    if not headers:
        return None
    value = headers.get(name)
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def forward_headers(correlation_id: str | None, traceparent: str | None, user_id: str | None) -> dict:
    headers = {}
    if traceparent:
        headers["traceparent"] = traceparent
    if user_id or correlation_id:
        headers["user_id"] = user_id or correlation_id
    return headers


async def retrieve_keyword_context(pool: asyncpg.Pool, transcript: str) -> str:
    if not KEYWORD_FALLBACK_TRIGGER.search(transcript):
        return ""

    async with pool.acquire() as connection:
        rows = await connection.fetch(
            """
            SELECT source, content
            FROM rag_chunks
            WHERE content IS NOT NULL
            """
        )

    ranked_rows = sorted(
        (
            (keyword_score(transcript, row["source"], row["content"]), row)
            for row in rows
        ),
        key=lambda item: (-item[0], -source_priority(item[1]["source"]), item[1]["source"]),
    )
    useful_rows = [row for score, row in ranked_rows if score > 0][:RAG_TOP_K]
    scores = ", ".join(f"{row['source']}={keyword_score(transcript, row['source'], row['content'])}" for row in useful_rows)
    print(f"[RAG] Keyword fallback scores: {scores or 'none'}", flush=True)
    return "\n---\n".join(f"[{row['source']}]\n{row['content']}" for row in useful_rows)


async def retrieve_context(pool: asyncpg.Pool, embedding: List[float], transcript: str) -> tuple[str, str]:
    async with pool.acquire() as connection:
        async with connection.transaction():
            # Exact scan avoids IVFFlat returning no candidates on tiny demo datasets.
            await connection.execute("SET LOCAL enable_indexscan = off")
            await connection.execute("SET LOCAL enable_bitmapscan = off")
            rows = await connection.fetch(
                """
                SELECT source, content, 1 - (embedding <=> $1::vector) AS similarity
                FROM rag_chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                vector_literal(embedding),
                RAG_TOP_K,
            )

    top_scores = ", ".join(f"{row['source']}={float(row['similarity']):.3f}" for row in rows)
    print(f"[RAG] Retrieval scores: {top_scores or 'none'}", flush=True)
    useful_rows = [row for row in rows if float(row["similarity"]) >= RAG_MIN_SIMILARITY]
    if useful_rows:
        return "\n---\n".join(f"[{row['source']}]\n{row['content']}" for row in useful_rows), "hit-vector"

    keyword_context = await retrieve_keyword_context(pool, transcript)
    if keyword_context:
        return keyword_context, "hit-keyword"
    return "", "miss"


async def publish_llm_payload(
    channel: aio_pika.Channel,
    user_transcript: str,
    retrieved_context: str,
    correlation_id: str | None,
    traceparent: str | None,
    user_id: str | None,
) -> None:
    payload = {
        "user_transcript": user_transcript,
        "retrieved_context": sanitize_retrieved_context(retrieved_context),
    }
    await channel.default_exchange.publish(
        aio_pika.Message(
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            correlation_id=correlation_id,
            content_type="application/json",
            headers=forward_headers(correlation_id, traceparent, user_id),
        ),
        routing_key=TEXT_LLM_QUEUE,
    )


async def main() -> None:
    model = load_embedding_model()
    pg_pool = await connect_pg_pool()
    await ensure_schema(pg_pool)
    await seed_if_empty(pg_pool, model)

    connection = await connect_broker()
    try:
        async with connection:
            channel = await connection.channel()
            await channel.set_qos(prefetch_count=8)

            incoming_queue = await channel.declare_queue(TEXT_RAG_QUEUE, durable=True)
            await channel.declare_queue(TEXT_LLM_QUEUE, durable=True)
            print(f"[RAG] Listening to {TEXT_RAG_QUEUE}...", flush=True)

            async with incoming_queue.iterator() as queue_iter:
                async for message in queue_iter:
                    try:
                        correlation_id = message.correlation_id
                        traceparent = header_as_text(message.headers, "traceparent")
                        user_id = header_as_text(message.headers, "user_id")
                        user_transcript = message.body.decode("utf-8", errors="replace").strip()

                        if not user_transcript:
                            await message.ack()
                            continue

                        embedding = (await asyncio.to_thread(embed_texts, model, [user_transcript]))[0]
                        retrieved_context, context_status = await retrieve_context(pg_pool, embedding, user_transcript)
                        await publish_llm_payload(
                            channel,
                            user_transcript,
                            retrieved_context,
                            correlation_id,
                            traceparent,
                            user_id,
                        )

                        print(
                            f"[RAG] Published RAG payload context={context_status}; "
                            f"transcript={user_transcript!r}; context_chars={len(retrieved_context)}; "
                            f"correlation_id={correlation_id}; traceparent={traceparent}",
                            flush=True,
                        )
                        await message.ack()
                    except Exception as exc:
                        print(f"[RAG] Failed to process transcript: {exc}", flush=True)
                        await message.nack(requeue=True)
    finally:
        await pg_pool.close()


if __name__ == "__main__":
    asyncio.run(main())
