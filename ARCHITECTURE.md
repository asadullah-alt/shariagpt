# ShariaGPT Architectural Reasoning

This document outlines the core architectural decisions, security measures, and trade-offs made during the development of ShariaGPT, reflecting our commitment to production-readiness under tight constraints.

## 1. RAG & API Quality and Security

### Robust Retrieval (Correctness)
ShariaGPT utilizes a highly accurate **Two-Stage RAG Pipeline**:
1. **Stage 1 (Fast Retrieval):** We use a lightweight, locally executed embedding model (`all-MiniLM-L6-v2`) paired with **Qdrant Cloud** to retrieve the top 10-15 candidate chunks.
2. **Stage 2 (Cross-Encoder Reranking):** We pass the candidates through a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to computationally score and rerank the documents. This ensures the LLM is only fed context that strictly aligns with the user's nuanced Islamic finance query, reducing hallucinations.

### PII Handling & Compliance (Security)
In FinTech, data privacy is paramount. We implemented a strict **Local PII Redaction Layer**. 
*Before* any user prompt or customer context (such as account balances or IDs) leaves our secure application boundary to hit the LLM (OpenRouter/OpenAI), it is scrubbed by our custom regex redactor.
- **Masked entities:** Emirates IDs, Account Numbers, IBANs, Emails, Phone Numbers, and Names.
- This ensures compliance with regional data protection regulations by guaranteeing no sensitive PII is ever leaked to third-party LLM providers.

### Observability
We engineered a custom JSONL trace logger that emits structured telemetry on every API call. This includes:
- E2E Latency (ms) broken down by stage.
- Prompt and Completion Token usage.
- Retrieved Chunk IDs and their Cross-Encoder Relevance Scores.
- PII Detection flags.
This structured output is designed to be easily ingested by Datadog, ELK, or Grafana for real-time monitoring of model drift, retrieval degradation, and cost tracking.

## 2. Scalability & Architectural Trade-offs

### Horizontal Scaling & Statelessness
The application is designed to sit behind a Load Balancer and scale infinitely. 
- **Authentication:** We use stateless **JWTs** (JSON Web Tokens). No server-side session lookup is required to validate user identity.
- **Session History & Caching:** We rely on **Upstash Redis** as a centralized, ultra-fast key-value store for both Semantic Caching and User Chat Threads. If a user's request is routed to a different server mid-conversation, the new server seamlessly pulls the context from Redis.
- **Resiliency:** The LLM API integration is wrapped in a **Circuit Breaker** (`pybreaker`). If the LLM provider experiences an outage, our circuit breaker trips and immediately returns a graceful `503 Service Unavailable` rather than allowing requests to hang and exhaust server connection pools.

### What We Cut (Trade-offs & Constraints)
Under strict time constraints, a strong engineering team must prioritize what to build versus what to cut.
- **Trade-off:** We chose to use **Qdrant Cloud** as our primary datastore for *both* our vector embeddings and our relational user data (Accounts/Passwords).
- **Reasoning:** Setting up a separate PostgreSQL database, writing SQLAlchemy models, and managing Alembic migrations for a simple user authentication system would have consumed significant time. Because Qdrant supports blazing-fast **Payload Filtering** (keyword lookups), we were able to store user JSON payloads with dummy zero-vectors and query them by email. This drastically reduced our infrastructure footprint and speed-to-market while still supporting robust CRUD operations.

## 3. Production Readiness & Hygiene

- **Test Coverage:** We enforce determinism in our pipeline using `pytest`. Our test suites cover PII Redaction edge cases (ensuring no false positives on normal numbers), Grounding checks, Semantic Cache hit validations, and prompt-injection/refusal logic for off-topic queries.
- **Clean Code Structure:** The codebase follows Domain-Driven Design principles, segregating concerns into `app/routers` (API), `app/rag` (AI pipeline), `app/auth` (Security), and `app/sessions` (State).
- **Deployment Hygiene:** The application uses `.env` configuration injection, a minimal and strict `requirements.txt`, and features graceful in-memory fallbacks if Redis fails, preventing hard crashes.
