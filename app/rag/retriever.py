"""
Optimized Retriever
───────────────────
Stage 0: Query Transformation (HyDE) — generate hypothetical answer, blend embeddings
Stage 1: Dense search via Qdrant → top k results
"""
from app.rag.query_transformer import transform_query
from app.rag.vector_store import hybrid_search
from app.config import get_settings
from typing import Optional


def retrieve(query: str, k: Optional[int] = None) -> list[dict]:
    """
    Optimized retrieval pipeline (Hybrid: Dense + Sparse + RRF).

    Stage 0: HyDE query transformation
      - LLM generates a hypothetical passage answering the query
      - Dense embedding = average(raw_query_emb, hyde_passage_emb)
      - Sparse embedding = fastembed on query + hyde_passage

    Stage 1: Hybrid retrieval → top k results
    """
    k = k or get_settings().top_k_chunks

    # Stage 0: Query transformation (HyDE + Encode)
    transformed = transform_query(query)
    dense_vec = transformed["dense_vector"]
    sparse_vec = transformed["sparse_vector"]

    # Stage 1: Hybrid retrieval
    candidates = hybrid_search(
        query_dense=dense_vec,
        query_sparse=sparse_vec,
        k=k,
    )

    return candidates
