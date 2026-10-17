"""
Self-learning AI agent on mem0.

- Every conversation turn is passed to mem0, which uses an LLM to extract
  durable facts and store them as embeddings (this is the "learning").
- Before every reply, the agent searches memory for what is relevant to the
  user's message and injects it into the prompt (this is the "recall").
- Everything is stored locally (Qdrant on disk + local embedding model),
  so memory persists between runs.

Commands inside the chat:
  /memories        list everything stored
  /search <text>   search memory directly
  /forget <id>     delete one memory by id
  /reset           wipe all memory for this user
  /quit            exit
"""

import os
import sys

import anthropic
from mem0 import Memory

USER_ID = os.getenv("MEM0_USER_ID", "panshul")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

# ---------------------------------------------------------------------------
# mem0 configuration (fully local except for the LLM calls)
# ---------------------------------------------------------------------------
MEM0_CONFIG = {
    # LLM mem0 uses to extract / update / deduplicate memories
    "llm": {
        "provider": "anthropic",
        "config": {
            "model": CLAUDE_MODEL,
            "temperature": 0.1,
            "max_tokens": 2000,
        },
    },
    # Local embeddings: free, no extra API key (Anthropic has no embeddings API)
    "embedder": {
        "provider": "huggingface",
        "config": {"model": "multi-qa-MiniLM-L6-cos-v1"},  # 384 dims
    },
    # Local on-disk vector DB
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
    """mem0 returns {'results': [...]} in newer versions and a list in older ones."""
    if isinstance(resp, dict):
        return resp.get("results", [])
    return resp or []


class MemoryAgent:
    def __init__(self, user_id: str = USER_ID):
        self.user_id = user_id
        self.memory = Memory.from_config(MEM0_CONFIG)
        self.llm = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.history = []  # short-term context for this session only

    # ---- recall -----------------------------------------------------------
    def recall(self, query: str, limit: int = 5):
        return _results(self.memory.get_all(filters={"user_id": self.user_id}))

    # ---- learn ------------------------------------------------------------
    def learn(self, user_msg: str, assistant_msg: str):
        self.memory.add(
            [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ],
            user_id=self.user_id,
        )

    # ---- chat -------------------------------------------------------------
    def chat(self, user_msg: str) -> str:
        found = self.recall(user_msg)
        memories_text = (
            "\n".join(f"- {m['memory']}" for m in found) if found else "(nothing stored yet)"
        )

        self.history.append({"role": "user", "content": user_msg})
        response = self.llm.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1000,
            system=SYSTEM_TEMPLATE.format(memories=memories_text),
            messages=self.history[-20:],  # keep short-term window small
        )
        answer = response.content[0].text
        self.history.append({"role": "assistant", "content": answer})

        self.learn(user_msg, answer)
        return answer

    # ---- management -------------------------------------------------------
    def list_all(self):
        return _results(self.memory.get_all(user_id=self.user_id))

    def forget(self, memory_id: str):
        self.memory.delete(memory_id)

    def reset(self):
        self.memory.delete_all(user_id=self.user_id)
        self.history.clear()


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first (see README in the reply).")

    agent = MemoryAgent()
    print(f"Memory agent ready (user: {agent.user_id}). Type /quit to exit.\n")

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
