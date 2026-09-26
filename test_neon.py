import os
from dotenv import load_dotenv
import psycopg

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing from .env")

try:
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]

    print("✅ Connected to Neon successfully!")
    print(version)

except Exception as e:
    print("❌ Neon connection failed:")
    print(e)
