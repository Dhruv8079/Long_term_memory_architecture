import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_classic.memory import ConversationTokenBufferMemory
from langchain_classic.chains import ConversationChain


# ============================================================
# 1. LOAD ENVIRONMENT
# ============================================================

env_path = Path.cwd().parent / ".env"
load_dotenv(env_path, override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env")


# ============================================================
# 2. USER ID
# ============================================================

USER_ID = "dhruv"


# ============================================================
# 3. GEMINI CHAT MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY
)


# ============================================================
# 4. GEMINI EMBEDDINGS
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
    output_dimensionality=768
)


# ============================================================
# 5. SHORT-TERM MEMORY
# ============================================================

memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=4000
)


# ============================================================
# 6. CONVERSATION CHAIN
# ============================================================

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)


# ============================================================
# 7. DATABASE CONNECTION
# ============================================================

def get_db_connection():
    return psycopg.connect(DATABASE_URL)


# ============================================================
# 8. NORMALIZE GEMINI RESPONSE
# ============================================================

def get_text_from_response(response):

    content = response.content

    # Normal string response
    if isinstance(content, str):
        return content.strip()

    # Gemini may return a list of content blocks
    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, str):
                text_parts.append(item)

            elif isinstance(item, dict):

                text_value = item.get("text")

                if text_value:
                    text_parts.append(
                        str(text_value)
                    )

        return "\n".join(text_parts).strip()

    return str(content).strip()


# ============================================================
# 9. SAVE LONG-TERM MEMORY
# ============================================================

def save_memory(
    user_id,
    content,
    memory_type="general",
    importance=0.5
):

    try:

        vector = embeddings.embed_query(content)

        conn = get_db_connection()

        with conn.cursor() as cur:

            cur.execute(
                """
                INSERT INTO memories
                (
                    user_id,
                    content,
                    memory_type,
                    importance,
                    embedding
                )
                VALUES
                (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s::vector
                )
                """,
                (
                    user_id,
                    content,
                    memory_type,
                    importance,
                    str(vector)
                )
            )

        conn.commit()
        conn.close()

    except Exception as e:

        print(
            f"❌ Memory save error: {e}"
        )


# ============================================================
# 10. SEARCH LONG-TERM MEMORY
# ============================================================

def search_memory(
    user_id,
    query,
    limit=5
):

    try:

        query_vector = embeddings.embed_query(
            query
        )

        conn = get_db_connection()

        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT
                    content,
                    memory_type,
                    importance,
                    1 - (
                        embedding <=> %s::vector
                    ) AS similarity

                FROM memories

                WHERE user_id = %s

                AND embedding IS NOT NULL

                ORDER BY
                    embedding <=> %s::vector

                LIMIT %s
                """,
                (
                    str(query_vector),
                    user_id,
                    str(query_vector),
                    limit
                )
            )

            results = cur.fetchall()

        conn.close()

        return results

    except Exception as e:

        print(
            f"❌ Memory search error: {e}"
        )

        return []


# ============================================================
# 11. GET RELEVANT MEMORY CONTEXT
# ============================================================

def get_memory_context(
    user_id,
    query,
    limit=5
):

    results = search_memory(
        user_id,
        query,
        limit
    )

    if not results:
        return "No relevant long-term memories."

    memory_lines = []

    for (
        content,
        memory_type,
        importance,
        similarity
    ) in results:

        if similarity >= 0.35:

            memory_lines.append(
                f"- {content}"
            )

    if not memory_lines:
        return "No relevant long-term memories."

    return "\n".join(
        memory_lines
    )


# ============================================================
# 12. AUTOMATIC MEMORY EXTRACTION
# ============================================================

def extract_and_save_memory(
    user_id,
    user_input
):

    extraction_prompt = f"""
You are a long-term memory extraction system.

Analyze the user's message.

Only remember useful information that can help
in future conversations.

Useful information includes:

- Name
- Skills
- Learning goals
- Projects
- Preferences
- Long-term plans
- Important facts

Do NOT remember:

- Greetings
- Temporary questions
- Casual conversation
- One-time calculations
- Random information

User message:

{user_input}

If there is useful information, return EXACTLY:

MEMORY: <memory>
TYPE: <type>
IMPORTANCE: <number between 0 and 1>

If there is nothing useful to remember, return:

NO_MEMORY
"""

    try:

        result = llm.invoke(
            extraction_prompt
        )

        text = get_text_from_response(
            result
        )

        if not text:
            return

        if "NO_MEMORY" in text.upper():
            return

        memory_content = None
        memory_type = "general"
        importance = 0.5

        for line in text.splitlines():

            line = line.strip()

            if line.upper().startswith(
                "MEMORY:"
            ):

                memory_content = (
                    line.split(
                        ":",
                        1
                    )[1].strip()
                )

            elif line.upper().startswith(
                "TYPE:"
            ):

                memory_type = (
                    line.split(
                        ":",
                        1
                    )[1].strip()
                )

            elif line.upper().startswith(
                "IMPORTANCE:"
            ):

                try:

                    importance = float(
                        line.split(
                            ":",
                            1
                        )[1].strip()
                    )

                    importance = max(
                        0.0,
                        min(
                            1.0,
                            importance
                        )
                    )

                except ValueError:

                    importance = 0.5

        if memory_content:

            save_memory(
                user_id,
                memory_content,
                memory_type,
                importance
            )

    except Exception as e:

        print(
            f"❌ Memory extraction error: {e}"
        )


# ============================================================
# 13. START CHATBOT
# ============================================================

print("\n🤖 Gemini Long-Term Memory Chatbot")
print("🧠 Short-term memory: 4000 tokens")
print("💾 Long-term memory: Neon PostgreSQL")
print("🔎 Semantic search: pgvector")
print("Type 'exit' to stop.\n")


# ============================================================
# 14. CHAT LOOP
# ============================================================

while True:

    user_input = input("You: ").strip()

    if user_input.lower() == "exit":

        print("Chat ended.")

        break

    if not user_input:
        continue

    try:

        # ----------------------------------------------------
        # Retrieve relevant long-term memory
        # ----------------------------------------------------

        long_term_context = get_memory_context(
            USER_ID,
            user_input,
            limit=5
        )

        # ----------------------------------------------------
        # Create prompt
        # ----------------------------------------------------

        prompt = f"""
You are a helpful AI assistant.

Relevant long-term information about the user:

{long_term_context}

Use the information above only when relevant.

Do not invent memories.
Do not mention the memory system.

User message:

{user_input}
"""

        # ----------------------------------------------------
        # Generate response
        # ----------------------------------------------------

        response = conversation.predict(
            input=prompt
        )

        print(
            f"Bot: {response}"
        )

        # ----------------------------------------------------
        # Extract and save long-term memory
        # ----------------------------------------------------

        extract_and_save_memory(
            USER_ID,
            user_input
        )

    except Exception as e:

        print(
            f"❌ Error: {e}"
        )