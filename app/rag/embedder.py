from functools import lru_cache
from sentence_transformers import SentenceTransformer
import torch

# ── Dense Model ──────────────────────────────────────────────────────────────
DENSE_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIM = 384


@lru_cache(maxsize=1)
def _get_dense_model() -> SentenceTransformer:
    return SentenceTransformer(DENSE_MODEL_NAME)


def embed_text(text: str) -> list[float]:
    res = _get_dense_model().encode(text, normalize_embeddings=True)
    if hasattr(res, "tolist"):
        return res.tolist()
    return list(res)


def embed_texts(texts: list[str]) -> list[list[float]]:
    res = _get_dense_model().encode(texts, normalize_embeddings=True)
    if hasattr(res, "tolist"):
        return res.tolist()
    return [list(r) for r in res]



