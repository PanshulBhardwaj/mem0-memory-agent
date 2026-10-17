# mem0 Memory Agent

A self-learning chat agent built on [mem0](https://github.com/mem0ai/mem0). It extracts facts from
every conversation, stores them as embeddings, and searches them before each reply, so it remembers
you across restarts.

Two versions:

| File | LLM | Cost |
|------|-----|------|
| `agent_local.py` | Ollama (runs on your PC) | Free, no API key |
| `agent.py` | Anthropic Claude | Needs an Anthropic API key with credits |

Both store memory locally in `./qdrant_data` using a local HuggingFace embedding model.

## Setup

```bash
python -m venv venv
# Windows: venv\Scripts\activate  |  macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### Local (free)
1. Install [Ollama](https://ollama.com) and run `ollama pull llama3.2:3b`
2. `python agent_local.py`

Use a bigger model for better memory extraction: set `OLLAMA_MODEL=llama3.1:8b`

### Claude
```bash
export ANTHROPIC_API_KEY="your-key"   # PowerShell: $env:ANTHROPIC_API_KEY = "your-key"
python agent.py
```

## Chat commands

`/memories` list stored memories | `/search <text>` search | `/forget <id>` delete one | `/reset` wipe all | `/quit` exit

## Notes
- `qdrant_data/` holds personal memories. It is git-ignored; don't commit it.
- Never commit API keys.
