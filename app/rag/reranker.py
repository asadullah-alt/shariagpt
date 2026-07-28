"""
Stage 2 Reranker — Cross-Encoder (BGE-reranker-base)
────────────────────────────────────────────────────────
Takes Stage 1 candidate chunks and re-scores them using a cross-encoder
that jointly attends to the query and each passage. This produces much
more accurate relevance scores than bi-encoder similarity.
"""
from functools import lru_cache
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "BAAI/bge-reranker-base"


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL, max_length=512)


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Re-rank candidate chunks using the cross-encoder.

    Parameters
    ----------
    query      : the user's search query
    candidates : list of dicts from Stage 1, each must have a "text" key
    top_k      : number of results to return after reranking

    Returns
    -------
    Top-k candidates sorted by cross-encoder relevance score.
    """
    if not candidates:
        return []

    # Build query-passage pairs for the cross-encoder
    pairs = [(query, c["text"]) for c in candidates]

    # Score all pairs in a single batch
    scores = _get_reranker().predict(pairs, show_progress_bar=False)

    # Attach scores and sort
    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = round(float(score), 6)
        # Keep original RRF score as stage1_score for observability
        candidate["stage1_score"] = candidate.get("score", 0.0)
        candidate["score"] = candidate["rerank_score"]

    # Sort descending by rerank score
    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_k]
