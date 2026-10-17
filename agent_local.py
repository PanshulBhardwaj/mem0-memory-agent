"""
Self-learning AI agent on mem0 - 100% local and free (Ollama).

No API keys. Chat model, memory extraction, embeddings and the vector DB
all run on your own PC. Memory persists in ./qdrant_data between runs.

Commands inside the chat:
  /memories        list everything stored
  /search <text>   search memory directly
  /forget <id>     delete one memory by id
  /reset           wipe all memory for this user
  /quit            exit
"""

import os
import sys

os.environ.setdefault("MEM0_TELEMETRY", "False")

import ollama
from mem0 import Memory

USER_ID = os.getenv("MEM0_USER_ID", "panshul")
# 3B runs on almost any PC. If you have 16 GB+ RAM, try: llama3.1:8b
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

MEM0_CONFIG = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": OLLAMA_MODEL,
            "temperature": 0.1,
            "max_tokens": 2000,
            "ollama_base_url": OLLAMA_URL,
        },
    },
    "embedder": {
        "provider": "huggingface",
        "config": {"model": "multi-qa-MiniLM-L6-cos-v1"},  # 384 dims
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "agent_memory",
            "path": "./qdrant_data",
            "embedding_model_dims": 384,
            "on_disk": True,
        },
    },
    "history_db_path": "./mem0_history.db",
}

SYSTEM_TEMPLATE = """You are a helpful assistant with long-term memory.

Below are memories retrieved from past conversations with this user. Use them
when relevant. If the user asks about something and the answer is in these
memories, answer from them directly. If it is not there, say you don't have
that stored yet - never invent personal details.

Relevant memories:
{memories}
"""


def _results(resp):
    if isinstance(resp, dict):
        return resp.get("results", [])
    return resp or []


class MemoryAgent:
    def __init__(self, user_id: str = USER_ID):
        self.user_id = user_id
        self.memory = Memory.from_config(MEM0_CONFIG)
        self.llm = ollama.Client(host=OLLAMA_URL)
        self.history = []

    def recall(self, query: str, limit: int = 5):
        return _results(
            self.memory.search(query, filters={"user_id": self.user_id}, limit=limit)
        )

    def learn(self, user_msg: str, assistant_msg: str):
        try:
            self.memory.add(
                [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ],
                user_id=self.user_id,
            )
        except Exception as e:  # small local models sometimes return bad JSON
            print(f"  (memory not saved this turn: {e})")

    def chat(self, user_msg: str) -> str:
        found = self.recall(user_msg)
        memories_text = (
            "\n".join(f"- {m['memory']}" for m in found) if found else "(nothing stored yet)"
        )

        self.history.append({"role": "user", "content": user_msg})
        messages = [
            {"role": "system", "content": SYSTEM_TEMPLATE.format(memories=memories_text)}
        ] + self.history[-20:]

        response = self.llm.chat(model=OLLAMA_MODEL, messages=messages)
        answer = response["message"]["content"]
        self.history.append({"role": "assistant", "content": answer})

        self.learn(user_msg, answer)
        return answer

    def list_all(self):
        return _results(self.memory.get_all(filters={"user_id": self.user_id}))

    def forget(self, memory_id: str):
        self.memory.delete(memory_id)

    def reset(self):
        self.memory.delete_all(user_id=self.user_id)
        self.history.clear()


def main():
    try:
        ollama.Client(host=OLLAMA_URL).list()
    except Exception:
        sys.exit(
            "Can't reach Ollama. Install it from ollama.com, make sure it is "
            f"running, then run:  ollama pull {OLLAMA_MODEL}"
        )

    agent = MemoryAgent()
    print(f"Local memory agent ready (user: {agent.user_id}, model: {OLLAMA_MODEL}).")
    print("Type /quit to exit.\n")

    while True:
        try:
            text = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue

        if text == "/quit":
            break
        elif text == "/memories":
            items = agent.list_all()
            if not items:
                print("(no memories yet)")
            for m in items:
                print(f"[{m['id']}] {m['memory']}")
        elif text.startswith("/search "):
            for m in agent.recall(text[len("/search "):], limit=10):
                print(f"{m.get('score', 0):.2f}  [{m['id']}] {m['memory']}")
        elif text.startswith("/forget "):
            agent.forget(text[len("/forget "):].strip())
            print("deleted.")
        elif text == "/reset":
            agent.reset()
            print("all memory wiped.")
        else:
            print(f"\nai  > {agent.chat(text)}\n")


if __name__ == "__main__":
    main()
