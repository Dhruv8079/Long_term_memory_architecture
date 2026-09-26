# long_term_memory_architecture

Long-term + short-term memory architecture for conversational AI using Gemini, LangChain, Neon PostgreSQL + pgvector.

## What's inside

- `buffer.py` — Short-term memory chatbot with `ConversationTokenBufferMemory` (4000 tokens max)
- `long_term_memory.py` — Full chatbot with:
  - Short-term memory (4000 token buffer)
  - Long-term memory in Neon PostgreSQL + pgvector
  - Semantic search (`embedding <=> query`)
  - Automatic memory extraction (name, skills, goals, projects, preferences)
- `test_neon.py` — Minimal Neon connection test
- `buffer_memory.ipynb` — Notebook version of buffer memory

## Architecture

```
User Input
   |
   v
[ Semantic Search in Neon pgvector ] -> Relevant long-term memories
   |
   v
[ Prompt + Long-term context + Short-term buffer (4000 tokens) ]
   |
   v
[ Gemini 3.6 Flash via LangChain ]
   |
   v
[ Auto-extract + Save new long-term memory ]
```

Short-term: `ConversationTokenBufferMemory(llm, max_token_limit=4000)`
Long-term table `memories`:
`user_id, content, memory_type, importance, embedding vector(768)`

Embeddings: `models/gemini-embedding-001` with `output_dimensionality=768`

## Setup

1. Create Neon PostgreSQL with pgvector:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE memories (
  id SERIAL PRIMARY KEY,
  user_id TEXT NOT NULL,
  content TEXT NOT NULL,
  memory_type TEXT DEFAULT 'general',
  importance FLOAT DEFAULT 0.5,
  embedding vector(768),
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

2. Create `.env` from `.env.example`:

```bash
cp .env.example .env
```

Fill in:
```
GOOGLE_API_KEY=your_gemini_key
DATABASE_URL=postgresql://user:pass@host/db?sslmode=require
```

3. Install:

```bash
pip install -r requirements.txt
```

4. Run:

```bash
python test_neon.py        # test DB
python buffer.py           # short-term only
python long_term_memory.py # full long-term memory chatbot
```

## Files copied from original `gen ai/memory_architecture/`
Only memory-related code — no secrets (`.env` is gitignored).
