-- Long-term memory schema (Neon PostgreSQL + pgvector)
-- Run once: psql $DATABASE_URL -f schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
  id SERIAL PRIMARY KEY,
  user_id TEXT NOT NULL,
  content TEXT NOT NULL,
  memory_type TEXT NOT NULL DEFAULT 'general',
  importance FLOAT NOT NULL DEFAULT 0.5,
  embedding vector(768),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories (user_id);
-- Vector similarity index (cosine). Use HNSW on Neon / pgvector >= 0.5.
-- CREATE INDEX IF NOT EXISTS idx_memories_embedding
--   ON memories USING hnsw (embedding vector_cosine_ops);
