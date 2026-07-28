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


# ── SPLADE Sparse Model ─────────────────────────────────────────────────────
SPLADE_MODEL_NAME = "naver/splade-cocondenser-ensembledistil"

_splade_model = None
_splade_tokenizer = None


def _get_splade():
    global _splade_model, _splade_tokenizer
    if _splade_model is None:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        _splade_tokenizer = AutoTokenizer.from_pretrained(SPLADE_MODEL_NAME)
        _splade_model = AutoModelForMaskedLM.from_pretrained(SPLADE_MODEL_NAME)
        _splade_model.eval()
    return _splade_model, _splade_tokenizer


def sparse_encode_text(text: str) -> dict[int, float]:
    """Encode text into a SPLADE sparse vector: {token_id: weight}."""
    model, tokenizer = _get_splade()
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        output = model(**tokens)
    # SPLADE log-saturated attention weights
    logits = output.logits
    weights = torch.max(torch.log1p(torch.relu(logits)), dim=1).values.squeeze()
    # Extract non-zero entries
    non_zero = weights.nonzero(as_tuple=True)[0]
    sparse_vec = {}
    for idx in non_zero:
        token_id = idx.item()
        weight = weights[token_id].item()
        if weight > 0:
            sparse_vec[token_id] = round(weight, 4)
    return sparse_vec


def sparse_encode_texts(texts: list[str]) -> list[dict[int, float]]:
    """Batch encode texts into SPLADE sparse vectors."""
    return [sparse_encode_text(t) for t in texts]
