from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    NamedVector, NamedSparseVector, SparseVector,
    SparseVectorParams, SparseIndexParams,
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
    """Create the knowledge collection with both dense and sparse named vectors."""
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
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )
        print(f"[VectorStore] Created hybrid collection '{s.qdrant_collection}'")


def upsert_chunks(chunks: list[dict]) -> None:
    """
    Upsert chunks with both dense and sparse vectors.
    chunks: list of {id, text, embedding, sparse_embedding, source, page_number, pdf_url}
    """
    s = get_settings()
    points = []
    for c in chunks:
        vectors = {"dense": c["embedding"]}
        sparse_emb = c.get("sparse_embedding")

        point_kwargs = {
            "id": c["id"],
            "payload": {
                "text": c["text"],
                "source": c["source"],
                "chunk_id": str(c["id"]),
                "page_number": c.get("page_number", 1),
                "pdf_url": c.get("pdf_url"),
            },
        }

        if sparse_emb:
            # PointStruct with named dense + named sparse
            point_kwargs["vector"] = {
                "dense": c["embedding"],
            }
            # We need to build the point manually for sparse support
            points.append(PointStruct(
                id=c["id"],
                vector={
                    "dense": c["embedding"],
                },
                payload=point_kwargs["payload"],
            ))
        else:
            points.append(PointStruct(
                id=c["id"],
                vector={"dense": c["embedding"]},
                payload=point_kwargs["payload"],
            ))

    # For points with sparse vectors, we need to use the update method
    # Qdrant Python client supports sparse vectors via the vectors dict
    client = get_client()

    # Re-build points with sparse vectors properly
    final_points = []
    for c in chunks:
        sparse_emb = c.get("sparse_embedding", {})
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

    # Now update sparse vectors separately if present
    from qdrant_client.models import PointVectors
    sparse_updates = []
    for c in chunks:
        sparse_emb = c.get("sparse_embedding", {})
        if sparse_emb:
            indices = list(sparse_emb.keys())
            values = list(sparse_emb.values())
            sparse_updates.append(
                PointVectors(
                    id=c["id"],
                    vector={
                        "sparse": SparseVector(indices=indices, values=values),
                    },
                )
            )

    if sparse_updates:
        client.update_vectors(
            collection_name=s.qdrant_collection,
            points=sparse_updates,
        )


def hybrid_search(
    dense_vector: list[float],
    sparse_vector: dict[int, float],
    k_stage1: int = 50,
    k_final: int = 5,
) -> list[dict]:
    """
    Stage 1 of two-stage retrieval: hybrid search using dense + sparse vectors.
    Returns top k_stage1 candidates merged via Reciprocal Rank Fusion (RRF).
    """
    s = get_settings()
    client = get_client()

    # Dense search
    dense_results = client.search(
        collection_name=s.qdrant_collection,
        query_vector=NamedVector(name="dense", vector=dense_vector),
        limit=k_stage1,
        with_payload=True,
    )

    # Sparse search
    sparse_indices = list(sparse_vector.keys())
    sparse_values = list(sparse_vector.values())
    sparse_results = []
    if sparse_indices:
        sparse_results = client.search(
            collection_name=s.qdrant_collection,
            query_vector=NamedSparseVector(
                name="sparse",
                vector=SparseVector(indices=sparse_indices, values=sparse_values),
            ),
            limit=k_stage1,
            with_payload=True,
        )

    # Reciprocal Rank Fusion (RRF) with k=60
    rrf_k = 60
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for rank, hit in enumerate(dense_results):
        point_id = str(hit.id)
        scores[point_id] = scores.get(point_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        payloads[point_id] = {
            "chunk_id": str(hit.id),
            "text": hit.payload["text"],
            "source": hit.payload.get("source", ""),
            "page_number": hit.payload.get("page_number", 1),
            "pdf_url": hit.payload.get("pdf_url"),
            "dense_score": round(float(hit.score), 4),
        }

    for rank, hit in enumerate(sparse_results):
        point_id = str(hit.id)
        scores[point_id] = scores.get(point_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        if point_id not in payloads:
            payloads[point_id] = {
                "chunk_id": str(hit.id),
                "text": hit.payload["text"],
                "source": hit.payload.get("source", ""),
                "page_number": hit.payload.get("page_number", 1),
                "pdf_url": hit.payload.get("pdf_url"),
                "dense_score": 0.0,
            }
        payloads[point_id]["sparse_score"] = round(float(hit.score), 4)

    # Sort by RRF score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for point_id, rrf_score in ranked[:k_stage1]:
        entry = payloads[point_id]
        entry["score"] = round(rrf_score, 6)
        results.append(entry)

    return results


# Legacy compatibility — dense-only search
def similarity_search(query_embedding: list[float], k: int = 5) -> list[dict]:
    """Dense-only search (used by semantic cache)."""
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
