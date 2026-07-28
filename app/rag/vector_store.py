from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, NamedVector
)
from app.config import get_settings
from app.rag.embedder import VECTOR_DIM
from typing import Optional

_client: Optional[QdrantClient] = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        s = get_settings()
        _client = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=20)
    return _client


def ensure_collection() -> None:
    """Create the knowledge collection with dense named vectors."""
    s = get_settings()
    client = get_client()
    names = [c.name for c in client.get_collections().collections]
    if s.qdrant_collection not in names:
        client.create_collection(
            collection_name=s.qdrant_collection,
            vectors_config={
                "dense": VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            },
        )
        print(f"[VectorStore] Created collection '{s.qdrant_collection}'")


def upsert_chunks(chunks: list[dict]) -> None:
    """
    Upsert chunks with dense vectors.
    chunks: list of {id, text, embedding, source, page_number, pdf_url}
    """
    s = get_settings()
    client = get_client()

    final_points = []
    for c in chunks:
        vectors_dict = {
            "dense": c["embedding"],
        }
        p = PointStruct(
            id=c["id"],
            vector=vectors_dict,
            payload={
                "text": c["text"],
                "source": c["source"],
                "chunk_id": str(c["id"]),
                "page_number": c.get("page_number", 1),
                "pdf_url": c.get("pdf_url"),
            },
        )
        final_points.append(p)

    client.upsert(collection_name=s.qdrant_collection, points=final_points, wait=True)


def dense_search(query_embedding: list[float], k: int = 5) -> list[dict]:
    """Dense-only search (used by semantic cache and RAG)."""
    s = get_settings()
    results = get_client().search(
        collection_name=s.qdrant_collection,
        query_vector=NamedVector(name="dense", vector=query_embedding),
        limit=k,
        with_payload=True,
    )
    return [
        {
            "chunk_id": str(r.id),
            "text": r.payload["text"],
            "source": r.payload.get("source", ""),
            "page_number": r.payload.get("page_number", 1),
            "pdf_url": r.payload.get("pdf_url"),
            "score": round(float(r.score), 4),
        }
        for r in results
    ]


def collection_count() -> int:
    try:
        s = get_settings()
        info = get_client().get_collection(s.qdrant_collection)
        return info.points_count or 0
    except Exception:
        return 0
