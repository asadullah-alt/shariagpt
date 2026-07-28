"""
Optimized Retriever
───────────────────
Stage 0: Query Transformation (HyDE) — generate hypothetical answer, blend embeddings
Stage 1: Dense search via Qdrant → top k results
"""
from app.rag.query_transformer import transform_query
from app.rag.vector_store import dense_search
from app.config import get_settings
from typing import Optional


def retrieve(query: str, k: Optional[int] = None) -> list[dict]:
    """
    Optimized retrieval pipeline (Dense-only to conserve memory).

    Stage 0: HyDE query transformation
      - LLM generates a hypothetical passage answering the query
      - Dense embedding = average(raw_query_emb, hyde_passage_emb)

    Stage 1: Dense retrieval → top k results
    """
    k = k or get_settings().top_k_chunks

    # Stage 0: Query transformation (HyDE)
    transformed = transform_query(query)
    dense_vec = transformed["dense_vector"]

    # Stage 1: Dense retrieval
    candidates = dense_search(
        query_embedding=dense_vec,
        k=k,
    )

    return candidates
