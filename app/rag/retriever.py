"""
Three-Stage Retriever
─────────────────────
Stage 0: Query Transformation (HyDE) — generate hypothetical answer, blend embeddings
Stage 1: Hybrid search (Dense + SPLADE) via Qdrant → top 50 candidates
Stage 2: Cross-encoder reranker (BGE) → top k results
"""
from app.rag.query_transformer import transform_query
from app.rag.vector_store import hybrid_search
from app.rag.reranker import rerank
from app.config import get_settings
from typing import Optional


def retrieve(query: str, k: Optional[int] = None) -> list[dict]:
    """
    Full retrieval pipeline.

    Stage 0: HyDE query transformation
      - LLM generates a hypothetical passage answering the query
      - Dense embedding = average(raw_query_emb, hyde_passage_emb)
      - Sparse embedding = SPLADE(raw_query) (unchanged, lexical is kept pure)

    Stage 1: Hybrid retrieval (Dense + Sparse with RRF) → 50 candidates

    Stage 2: Cross-encoder reranking (BGE-reranker-base) → top k results
    """
    k = k or get_settings().top_k_chunks

    # Stage 0: Query transformation (HyDE)
    transformed = transform_query(query)
    dense_vec = transformed["dense_vector"]
    sparse_vec = transformed["sparse_vector"]

    # Stage 1: Hybrid retrieval — fetch broad candidate set
    candidates = hybrid_search(
        dense_vector=dense_vec,
        sparse_vector=sparse_vec,
        k_stage1=50,
    )

    if not candidates:
        return []

    # Stage 2: Cross-encoder reranking — precision pass
    results = rerank(query, candidates, top_k=k)

    return results
