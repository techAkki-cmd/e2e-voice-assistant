import argparse
import asyncio
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, List

import asyncpg
from pypdf import PdfReader

from main import (
    connect_pg_pool,
    embed_texts,
    ensure_schema,
    load_embedding_model,
    vector_literal,
)


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".json", ".csv", ".pdf"}


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strings_from_json(value) -> Iterable[str]:
    if isinstance(value, str):
        if value.strip():
            yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings_from_json(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings_from_json(item)


def read_text_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return "\n\n".join(strings_from_json(payload))
    if suffix == ".csv":
        rows = []
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append("; ".join(f"{key}: {value}" for key, value in row.items() if value))
        return "\n\n".join(rows)
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return ""


def chunk_text(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", normalize_text(text)) if paragraph.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            start = 0
            while start < len(paragraph):
                end = start + max_chars
                chunks.append(paragraph[start:end].strip())
                if end >= len(paragraph):
                    break
                start = max(0, end - overlap_chars)
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = paragraph

    if current:
        chunks.append(current.strip())

    if overlap_chars <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for previous, chunk in zip(chunks, chunks[1:]):
        prefix = previous[-overlap_chars:].strip()
        overlapped.append(f"{prefix}\n\n{chunk}".strip() if prefix else chunk)
    return overlapped


def chunk_hash(source: str, content: str) -> str:
    return hashlib.sha256(f"{source}\n{content}".encode("utf-8")).hexdigest()


def collect_chunks(paths: list[Path], source_prefix: str, max_chars: int, overlap_chars: int) -> list[dict]:
    chunks = []
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(
                file_path
                for file_path in sorted(path.rglob("*"))
                if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        elif path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)

    for file_path in files:
        text = normalize_text(read_text_file(file_path))
        if not text:
            continue
        for index, content in enumerate(chunk_text(text, max_chars, overlap_chars), start=1):
            source = f"{source_prefix}:{file_path.name}:chunk-{index:04d}"
            chunks.append(
                {
                    "source": source,
                    "content": content,
                    "source_hash": chunk_hash(source, content),
                }
            )

    return chunks


async def delete_source(pool: asyncpg.Pool, source_prefix: str) -> int:
    async with pool.acquire() as connection:
        return await connection.fetchval(
            "WITH deleted AS (DELETE FROM rag_chunks WHERE source LIKE $1 RETURNING 1) SELECT count(*) FROM deleted",
            f"{source_prefix}:%",
        )


async def insert_chunks(pool: asyncpg.Pool, model, chunks: list[dict], batch_size: int) -> int:
    inserted = 0
    async with pool.acquire() as connection:
        async with connection.transaction():
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                embeddings = await asyncio.to_thread(embed_texts, model, [chunk["content"] for chunk in batch])
                for chunk, embedding in zip(batch, embeddings):
                    result = await connection.execute(
                        """
                        INSERT INTO rag_chunks (source, content, source_hash, embedding)
                        VALUES ($1, $2, $3, $4::vector)
                        ON CONFLICT (source_hash) WHERE source_hash IS NOT NULL DO NOTHING
                        """,
                        chunk["source"],
                        chunk["content"],
                        chunk["source_hash"],
                        vector_literal(embedding),
                    )
                    if result.endswith(" 1"):
                        inserted += 1
    return inserted


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest company documents into pgvector rag_chunks.")
    parser.add_argument("paths", nargs="+", help="Files or directories to ingest")
    parser.add_argument("--source", default="company-docs", help="Source prefix used in rag_chunks.source")
    parser.add_argument("--reset-source", action="store_true", help="Delete existing chunks for this source prefix first")
    parser.add_argument("--max-chars", type=int, default=900, help="Maximum characters per chunk")
    parser.add_argument("--overlap-chars", type=int, default=120, help="Context overlap between adjacent chunks")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding insert batch size")
    args = parser.parse_args()

    paths = [Path(path) for path in args.paths]
    chunks = collect_chunks(paths, args.source, args.max_chars, args.overlap_chars)
    if not chunks:
        print("[INGEST] No supported document text found.", flush=True)
        return

    model = load_embedding_model()
    pool = await connect_pg_pool()
    try:
        await ensure_schema(pool)
        if args.reset_source:
            deleted = await delete_source(pool, args.source)
            print(f"[INGEST] Deleted existing chunks for source={args.source}: {deleted}", flush=True)
        inserted = await insert_chunks(pool, model, chunks, args.batch_size)
        skipped = len(chunks) - inserted
        print(
            f"[INGEST] Completed source={args.source}; chunks_seen={len(chunks)}; "
            f"inserted={inserted}; skipped_duplicates={skipped}",
            flush=True,
        )
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
