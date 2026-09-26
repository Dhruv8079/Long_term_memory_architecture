"""PostgreSQL helpers (Neon + pgvector)."""

from contextlib import contextmanager
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = BASE_DIR / "schema.sql"


@contextmanager
def get_connection(database_url: str):
    import psycopg

    conn = psycopg.connect(database_url)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(database_url: str) -> None:
    """Create extension + table from schema.sql."""
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with get_connection(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
