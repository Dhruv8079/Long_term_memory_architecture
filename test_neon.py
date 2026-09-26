"""Neon PostgreSQL + pgvector connection check.

Checks:
  1. DATABASE_URL is set
  2. Server version / connectivity
  3. pgvector extension availability
  4. memories table existence
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(Path.cwd() / ".env")


def main() -> int:
    try:
        import psycopg
    except ImportError:
        print("psycopg is not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("DATABASE_URL is missing. Copy .env.example to .env and fill it in.", file=sys.stderr)
        return 1

    try:
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
                print("Connected to Neon successfully!")
                print(version)

                cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
                row = cur.fetchone()
                if row:
                    print("pgvector extension: enabled")
                else:
                    print("WARNING: pgvector extension not enabled. Run schema.sql.")

                cur.execute(
                    "SELECT to_regclass('public.memories');"
                )
                table = cur.fetchone()[0]
                if table:
                    print("memories table: found")
                    cur.execute("SELECT count(*) FROM memories;")
                    print(f"memories rows: {cur.fetchone()[0]}")
                else:
                    print("WARNING: memories table not found. Run schema.sql.")
        return 0
    except Exception as e:
        print("Neon connection failed:")
        print(e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
