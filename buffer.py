"""Short-term memory chatbot using Gemini + LangChain token buffer.

Keeps the last ~MAX_TOKENS tokens of conversation history.
No database required.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationTokenBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env from this file's directory first, then cwd (repo root support)
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(Path.cwd() / ".env")

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))


def build_chain(api_key: str, model: str, max_tokens: int) -> tuple[ConversationChain, ConversationTokenBufferMemory, ChatGoogleGenerativeAI]:
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=0.7,
        google_api_key=api_key,
    )
    memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=max_tokens)
    conversation = ConversationChain(llm=llm, memory=memory, verbose=False)
    return conversation, memory, llm


def run_chat(model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not found. Copy .env.example to .env and fill it in.", file=sys.stderr)
        raise SystemExit(1)

    conversation, memory, llm = build_chain(api_key, model, max_tokens)

    print("\nGemini Token Buffer Chatbot Started!")
    print(f"Model: {model} | Max memory: {max_tokens} tokens")
    print("Type 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print("\nChat ended.")
            break

        if user_input.strip().lower() == "exit":
            print("Chat ended.")
            break
        if not user_input.strip():
            continue

        try:
            response = conversation.predict(input=user_input)
            print(f"Bot: {response}")

            try:
                history = memory.load_memory_variables({})["history"]
                token_count = llm.get_num_tokens(history)
                print(f"[memory: {token_count}/{max_tokens} tokens]\n")
            except Exception:
                print()
        except Exception as e:
            print(f"\nError: {e}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Short-term token-buffer chatbot")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Buffer size in tokens")
    args = parser.parse_args()
    run_chat(model=args.model, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
