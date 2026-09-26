import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.memory import ConversationTokenBufferMemory
from langchain_classic.chains import ConversationChain


# =====================================================
# 1. Load .env file
# =====================================================

# Notebook is inside:
# gen ai/memory_architecture/
# .env is inside:
# gen ai/

env_path = Path.cwd().parent / ".env"

load_dotenv(env_path, override=True)


# =====================================================
# 2. Get Gemini API Key
# =====================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        f"GOOGLE_API_KEY not found.\n"
        f"Check your .env file at: {env_path}"
    )

print("✅ Gemini API key loaded successfully")


# =====================================================
# 3. Gemini Model
# =====================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY
)


# =====================================================
# 4. Token Buffer Memory
#    Maximum = 4000 tokens
# =====================================================

memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=4000
)


# =====================================================
# 5. Conversation Chain
# =====================================================

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)


# =====================================================
# 6. Start Chatbot
# =====================================================

print("\n🤖 Gemini Token Buffer Chatbot Started!")
print("🧠 Maximum memory: 4000 tokens")
print("Type 'exit' to stop.\n")


while True:

    user_input = input("You: ")

    if user_input.lower().strip() == "exit":
        print("Chat ended.")
        break

    try:

        response = conversation.predict(input=user_input)

        print("Bot:", response)

        # Display approximate current memory usage
        history = memory.load_memory_variables({})["history"]

        try:
            token_count = llm.get_num_tokens(history)
            print(f"🧠 Memory: {token_count}/4000 tokens\n")
        except Exception:
            print()

    except Exception as e:
        print(f"\n❌ Error: {e}\n")