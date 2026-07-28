import requests
from app.config import get_settings

# ── Dense Model ──────────────────────────────────────────────────────────────
DENSE_MODEL_NAME = "openai/text-embedding-3-small"
VECTOR_DIM = 384


def _get_headers():
    s = get_settings()
    return {
        "Authorization": f"Bearer {s.openrouter_api_key}",
        "Content-Type": "application/json"
    }


def embed_text(text: str) -> list[float]:
    s = get_settings()
    url = f"{s.openrouter_base_url.rstrip('/')}/embeddings"
    payload = {
        "model": DENSE_MODEL_NAME,
        "input": text,
        "dimensions": VECTOR_DIM
    }
    r = requests.post(url, headers=_get_headers(), json=payload)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    s = get_settings()
    url = f"{s.openrouter_base_url.rstrip('/')}/embeddings"
    payload = {
        "model": DENSE_MODEL_NAME,
        "input": texts,
        "dimensions": VECTOR_DIM
    }
    r = requests.post(url, headers=_get_headers(), json=payload)
    r.raise_for_status()
    data = r.json()["data"]
    # Ensure they are in the same order
    return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]
