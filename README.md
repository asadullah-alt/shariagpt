# ShariaGPT — Islamic Finance AI Assistant

A production-ready RAG-powered AI assistant for Mal customers to query Islamic finance rules
and account context. Built with FastAPI, Qdrant Cloud, and OpenRouter (GPT-4o-mini).

## Features
- **RAG pipeline**: 8 Sharia finance documents (Murabaha, Sukuk, Ijara, Mudaraba, Musharaka, Takaful, Riba, Zakat)
- **PII redaction**: Emirates ID, account numbers, IBAN, email, phone, and names masked before LLM
- **Stateful conversations**: Session history persisted in Redis (in-memory fallback)
- **Structured observability**: JSONL trace logs with latency, token usage, chunk IDs, relevance scores
- **Automated eval tests**: Grounding, PII redaction, off-topic refusal

## API Endpoints

### `POST /chat`
```json
{
  "session_id": "user-123",
  "message": "How does Murabaha financing work?",
  "customer_context": {"account_type": "Murabaha", "balance": "AED 50,000"}
}
```
Response:
```json
{
  "session_id": "user-123",
  "request_id": "uuid",
  "response": "Murabaha is a cost-plus-profit sale...",
  "retrieved_chunk_ids": ["1000", "1001"],
  "relevance_scores": [0.91, 0.87],
  "avg_relevance_score": 0.89,
  "pii_detected": []
}
```

### `GET /health`
```json
{
  "status": "ok",
  "version": "1.0.0",
  "model": "openai/gpt-4o-mini",
  "vector_store_docs": 42,
  "active_sessions": 3,
  "uptime_seconds": 120.5
}
```

## Setup

### 1. Clone and install
```bash
git clone <repo-url>
cd shariagpt
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env with your real keys:
#   OPENROUTER_API_KEY  → https://openrouter.ai/
#   QDRANT_URL          → https://cloud.qdrant.io/
#   QDRANT_API_KEY      → Qdrant Cloud API key
#   REDIS_URL           → https://upstash.com/ (free tier)
```

### 3. Ingest documents (optional — runs automatically on first startup)
```bash
python data/ingest.py
```

### 4. Run locally
```bash
python app/main.py
# API docs: http://localhost:10000/docs
```

### 5. Run tests
```bash
pytest tests/ -v
```

## Deployment (Render)

1. Push this repo to GitHub.
2. Go to [render.com](https://render.com) → **New Web Service** → Connect GitHub repo.
3. Render will detect `render.yaml` automatically.
4. Set the secret environment variables in the Render dashboard:
   - `OPENROUTER_API_KEY`
   - `QDRANT_URL`
   - `QDRANT_API_KEY`
   - `REDIS_URL`
5. Deploy — the build step pre-downloads the embedding model.

## Architecture & Technical Reasoning

We have documented our comprehensive architectural decisions, trade-offs, security implementations (PII redaction), and scalability strategy in our dedicated architecture document. 

👉 **[Read the Architecture Document here (ARCHITECTURE.md)](ARCHITECTURE.md)**

### High-level Flow:
```
POST /chat

  → PII Redactor (regex: Emirates ID, account #, IBAN, email, phone, name)
  → Sentence-Transformers Embedder (all-MiniLM-L6-v2, local)
  → Qdrant Cloud Vector Search (top-5 chunks)
  → Prompt Builder (system prompt + context + history + query)
  → OpenRouter → GPT-4o-mini
  → Session Store (Redis / in-memory)
  → JSONL Trace Logger
  → Response
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | ✅ | OpenRouter API key |
| `QDRANT_URL` | ✅ | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | ✅ | Qdrant Cloud API key |
| `REDIS_URL` | Optional | Redis URL for persistent sessions |
| `OPENROUTER_MODEL` | Optional | Default: `openai/gpt-4o-mini` |
| `TOP_K_CHUNKS` | Optional | Default: `5` |
| `SESSION_TTL_SECONDS` | Optional | Default: `86400` (24h) |
