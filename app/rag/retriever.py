"""
Async Parallel Retriever
────────────────────────
Runs two retrieval strategies concurrently:

  Task A (Raw Search):   embed(query) → hybrid_search           (~300ms)
  Task B (HyDE Search):  LLM → embed(hyde) → hybrid_search      (~2000ms)

Both tasks start simultaneously. Results are merged via application-level
Reciprocal Rank Fusion (RRF). If HyDE fails or times out, Task A's results
are returned as-is — the user never notices the degradation.
"""
import asyncio
from app.rag.query_transformer import generate_hypothetical_document
from app.rag.embedder import embed_text, sparse_encode_text
from app.rag.vector_store import hybrid_search
from app.config import get_settings
from typing import Optional


def _merge_rrf(raw_results: list[dict], hyde_results: list[dict], k: int, rrf_k: int = 60) -> list[dict]:
    """
    Merge two ranked result lists using Reciprocal Rank Fusion.
    Documents appearing in both lists get a score boost.
    """
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for rank, r in enumerate(raw_results):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
        payloads[cid] = r

    for rank, r in enumerate(hyde_results):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
        if cid not in payloads:
            payloads[cid] = r

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for cid, score in ranked[:k]:
        entry = payloads[cid].copy()
        entry["score"] = round(score, 6)
        results.append(entry)
    return results


async def retrieve(query: str, k: Optional[int] = None) -> list[dict]:
    """
    Async parallel retrieval pipeline.

    1. Compute raw query embeddings (dense + sparse) — fast, ~100ms
    2. Fire two tasks in parallel:
       - Task A: hybrid_search with raw vectors
       - Task B: HyDE LLM call → blend embeddings → hybrid_search
    3. Merge results via application-level RRF
    """
    k = k or get_settings().top_k_chunks

    # Step 1: Compute raw embeddings (shared by both tasks)
    raw_dense = await asyncio.to_thread(embed_text, query)
    raw_sparse = await asyncio.to_thread(sparse_encode_text, query)

    # Step 2: Define parallel tasks
    async def raw_search() -> list[dict]:
        """Task A: Search with raw query vectors."""
        return await asyncio.to_thread(hybrid_search, raw_dense, raw_sparse, k)

    async def hyde_search() -> list[dict]:
        """Task B: Generate HyDE passage, blend vectors, search."""
        hyde_passage = await generate_hypothetical_document(query)
        if not hyde_passage:
            return []

        hyde_dense = await asyncio.to_thread(embed_text, hyde_passage)

        # Blend raw + HyDE dense vectors (average)
        blended_dense = [
            (r + h) / 2.0 for r, h in zip(raw_dense, hyde_dense)
        ]

        # Sparse: encode the concatenation to capture both vocabularies
        blended_sparse = await asyncio.to_thread(
            sparse_encode_text, query + " " + hyde_passage
        )

        return await asyncio.to_thread(hybrid_search, blended_dense, blended_sparse, k)

    # Step 3: Run both tasks concurrently
    raw_results, hyde_results = await asyncio.gather(
        raw_search(),
        hyde_search(),
        return_exceptions=True,  # Don't crash if HyDE fails
    )

    # Handle exceptions gracefully
    if isinstance(raw_results, Exception):
        raw_results = []
    if isinstance(hyde_results, Exception):
        hyde_results = []

    # Step 4: Merge via RRF (or return raw-only if HyDE produced nothing)
    if not hyde_results:
        return raw_results
    if not raw_results:
        return hyde_results

    return _merge_rrf(raw_results, hyde_results, k)
