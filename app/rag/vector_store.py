from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, NamedVector,
    Filter, FieldCondition, MatchValue,
    SparseVectorParams, Modifier, SparseVector, Prefetch, FusionQuery, Fusion, NamedSparseVector
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
            sparse_vectors_config={
                "sparse": SparseVectorParams(modifier=Modifier.IDF),
            }
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
        if "sparse_embedding" in c:
            vectors_dict["sparse"] = SparseVector(
                indices=c["sparse_embedding"]["indices"],
                values=c["sparse_embedding"]["values"]
            )
            
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


def hybrid_search(query_dense: list[float], query_sparse: dict, k: int = 5) -> list[dict]:
    """Hybrid search using Reciprocal Rank Fusion (RRF)."""
    s = get_settings()
    results = get_client().query_points(
        collection_name=s.qdrant_collection,
        prefetch=[
            Prefetch(
                query=NamedVector(name="dense", vector=query_dense),
                limit=k * 2,
            ),
            Prefetch(
                query=NamedSparseVector(
                    name="sparse",
                    vector=SparseVector(
                        indices=query_sparse["indices"],
                        values=query_sparse["values"]
                    )
                ),
                limit=k * 2,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=k,
        with_payload=True,
    ).points
    
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


def delete_chunks_by_source(source_name: str) -> None:
    """Delete all chunks in Qdrant originating from a specific source."""
    s = get_settings()
    client = get_client()
    try:
        client.delete(
            collection_name=s.qdrant_collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=source_name)
                    )
                ]
            ),
        )
        print(f"[VectorStore] Deleted all chunks for source '{source_name}'")
    except Exception as e:
        print(f"[VectorStore] Failed to delete chunks for '{source_name}': {e}")
