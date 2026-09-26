"""Gemini chatbot with short-term buffer + Neon pgvector long-term memory.

Short-term: ConversationTokenBufferMemory (default 4000 tokens).
Long-term:  Neon PostgreSQL + pgvector, semantic recall + auto-extraction.

Run:
    python test_neon.py
    python long_term_memory.py --user dhruv
"""

import argparse
import logging
import sys

from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationTokenBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from config import load_settings
from database import get_connection

log = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are a long-term memory extraction system.

Analyze the user's message. Only remember useful information that can help
in future conversations.

Useful: name, skills, learning goals, projects, preferences, long-term plans, important facts.
Do NOT remember: greetings, temporary questions, casual chat, one-time calculations.

User message:
{user_input}

If there is useful information, return EXACTLY:
MEMORY: <memory>
TYPE: <type>
IMPORTANCE: <number between 0 and 1>

If there is nothing useful to remember, return:
NO_MEMORY
"""


def get_text_from_response(response) -> str:
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts).strip()
    return str(content).strip()


class LongTermMemory:
    def __init__(self, settings, embeddings: GoogleGenerativeAIEmbeddings):
        self.settings = settings
        self.embeddings = embeddings

    def save(self, content: str, memory_type: str = "general", importance: float = 0.5) -> None:
        try:
            vector = self.embeddings.embed_query(content)
            with get_connection(self.settings.database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO memories (user_id, content, memory_type, importance, embedding)
                        VALUES (%s, %s, %s, %s, %s::vector)
                        """,
                        (self.settings.user_id, content, memory_type, importance, str(vector)),
                    )
        except Exception as e:
            log.warning("Memory save error: %s", e)

    def search(self, query: str, limit: int = 5):
        try:
            query_vector = self.embeddings.embed_query(query)
            with get_connection(self.settings.database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT content, memory_type, importance,
                               1 - (embedding <=> %s::vector) AS similarity
                        FROM memories
                        WHERE user_id = %s AND embedding IS NOT NULL
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (str(query_vector), self.settings.user_id, str(query_vector), limit),
                    )
                    return cur.fetchall()
        except Exception as e:
            log.warning("Memory search error: %s", e)
            return []

    def context(self, query: str, limit: int = 5) -> str:
        results = self.search(query, limit)
        if not results:
            return "No relevant long-term memories."
        lines = [f"- {content}" for content, _t, _i, sim in results if sim >= self.settings.similarity_threshold]
        return "\n".join(lines) if lines else "No relevant long-term memories."


def extract_and_save(llm, store: LongTermMemory, user_input: str) -> None:
    try:
        text = get_text_from_response(llm.invoke(EXTRACTION_PROMPT.format(user_input=user_input)))
        if not text or "NO_MEMORY" in text.upper():
            return
        content, mtype, importance = None, "general", 0.5
        for line in text.splitlines():
            line = line.strip()
            upper = line.upper()
            if upper.startswith("MEMORY:"):
                content = line.split(":", 1)[1].strip()
            elif upper.startswith("TYPE:"):
                mtype = line.split(":", 1)[1].strip() or "general"
            elif upper.startswith("IMPORTANCE:"):
                try:
                    importance = max(0.0, min(1.0, float(line.split(":", 1)[1].strip())))
                except ValueError:
                    importance = 0.5
        if content:
            store.save(content, mtype, importance)
    except Exception as e:
        log.warning("Memory extraction error: %s", e)


def run_chat(user_id: str | None = None) -> None:
    settings = load_settings(user_id)

    llm = ChatGoogleGenerativeAI(model=settings.chat_model, google_api_key=settings.google_api_key)
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
        output_dimensionality=settings.embedding_dims,
    )
    memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=settings.max_tokens)
    conversation = ConversationChain(llm=llm, memory=memory, verbose=False)
    store = LongTermMemory(settings, embeddings)

    print("\nGemini Long-Term Memory Chatbot")
    print(f"Short-term: {settings.max_tokens} tokens | Long-term: Neon pgvector | User: {settings.user_id}")
    print("Type 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nChat ended.")
            break
        if user_input.lower() == "exit":
            print("Chat ended.")
            break
        if not user_input:
            continue
        try:
            context = store.context(user_input, limit=settings.top_k)
            prompt = (
                "You are a helpful AI assistant.\n\n"
                f"Relevant long-term information about the user:\n{context}\n\n"
                "Use it only when relevant. Do not invent memories. "
                "Do not mention the memory system.\n\n"
                f"User message:\n{user_input}"
            )
            print(f"Bot: {conversation.predict(input=prompt)}")
            extract_and_save(llm, store, user_input)
        except Exception as e:
            print(f"Error: {e}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Gemini long-term memory chatbot")
    parser.add_argument("--user", default=None, help="User id for scoped memories (default: $USER_ID or 'dhruv')")
    args = parser.parse_args()
    try:
        run_chat(user_id=args.user)
    except ValueError as e:
        print(e, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
