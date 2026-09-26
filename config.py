"""Central configuration for long_term_memory_architecture."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(Path.cwd() / ".env")


@dataclass(frozen=True)
class Settings:
    google_api_key: str
    database_url: str
    user_id: str = "dhruv"
    chat_model: str = "gemini-2.0-flash"
    embedding_model: str = "models/gemini-embedding-001"
    embedding_dims: int = 768
    max_tokens: int = 4000
    top_k: int = 5
    similarity_threshold: float = 0.35


def load_settings(user_id: str | None = None) -> Settings:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Copy .env.example to .env.")
    if not db_url:
        raise ValueError("DATABASE_URL not found. Copy .env.example to .env.")
    return Settings(
        google_api_key=api_key,
        database_url=db_url,
        user_id=user_id or os.getenv("USER_ID", "dhruv"),
        chat_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001"),
        embedding_dims=int(os.getenv("EMBEDDING_DIMS", "768")),
        max_tokens=int(os.getenv("MAX_TOKENS", "4000")),
        top_k=int(os.getenv("TOP_K", "5")),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.35")),
    )
