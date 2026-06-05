CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_chunks (
    id bigserial PRIMARY KEY,
    source text,
    content text NOT NULL,
    source_hash text,
    embedding vector(384),
    created_at timestamptz DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS rag_chunks_source_hash_idx
    ON rag_chunks (source_hash)
    WHERE source_hash IS NOT NULL;

CREATE INDEX IF NOT EXISTS rag_chunks_embedding_ivfflat_idx
    ON rag_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
