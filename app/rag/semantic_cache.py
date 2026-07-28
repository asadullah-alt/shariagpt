import hashlib
import uuid
import time
import json
from typing import Optional

from app.config import get_settings
from app.rag.embedder import embed_text
from app.rag.vector_store import get_client


CACHE_COLLECTION = "sharia_cache_queries"
_redis_client = None
_redis_initialized = False


def _get_redis_client():
    global _redis_client, _redis_initialized
    if not _redis_initialized:
        _redis_initialized = True
        s = get_settings()
        if s.upstash_redis_rest_url and s.upstash_redis_rest_token:
            from upstash_redis import Redis
            _redis_client = Redis(url=s.upstash_redis_rest_url, token=s.upstash_redis_rest_token)
    return _redis_client


def _ensure_cache_collection():
    client = get_client()
    try:
        names = [c.name for c in client.get_collections().collections]
        if CACHE_COLLECTION not in names:
            from qdrant_client.models import VectorParams, Distance
            from app.rag.embedder import VECTOR_DIM

            client.create_collection(
                collection_name=CACHE_COLLECTION,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
    except Exception:
        pass


def get_knowledge_version() -> int:
    """Get the global knowledge version from Redis."""
    redis_client = _get_redis_client()
    if not redis_client:
        return 0
    val = redis_client.get("shariagpt:knowledge_version")
    return int(val) if val else 0


def increment_knowledge_version() -> int:
    """Increment the global knowledge version in Redis (used when new PDFs are uploaded)."""
    redis_client = _get_redis_client()
    if not redis_client:
        return 0
    return redis_client.incr("shariagpt:knowledge_version")


def get_cached_response(query: str, user_id: str, account_type: str, threshold: float = 0.85) -> Optional[dict]:
    """
    Search Qdrant for semantically similar questions. 
    If a match > threshold is found, fetch the corresponding response from Upstash Redis.
    Validates user_id, account_type, TTL, and knowledge_version.
    """
    redis_client = _get_redis_client()
    if not redis_client:
        return None

    _ensure_cache_collection()
    q_client = get_client()

    query_emb = embed_text(query)
    try:
        results = q_client.search(
            collection_name=CACHE_COLLECTION,
            query_vector=query_emb,
            limit=1,
            with_payload=True,
        )
    except Exception:
        return None

    if results and results[0].score >= threshold:
        payload = results[0].payload or {}
        cache_id = payload.get("cache_id")
        if cache_id:
            try:
                cached_val = redis_client.get(cache_id)
                if cached_val:
                    data = json.loads(cached_val)
                    
                    # 1. TTL Check (1 day)
                    if time.time() - data.get("created_at", 0) > 86400:
                        return None
                        
                    # 2. User Scope Check
                    if data.get("user_id") != user_id:
                        return None
                        
                    # 3. Account Type Check
                    if data.get("account_type") != account_type:
                        return None
                        
                    # 4. Knowledge Version Check
                    if data.get("knowledge_version", 0) < get_knowledge_version():
                        return None
                        
                    return data
            except Exception:
                return None
    return None


def set_cached_response(
    query: str, 
    response: str, 
    user_id: str, 
    account_type: str,
    citations: list = None, 
    ttl_seconds: int = 86400
) -> None:
    """
    Save the LLM response in Upstash Redis and index the query's embedding in Qdrant.
    Scoped to user and current knowledge version.
    """
    redis_client = _get_redis_client()
    if not redis_client:
        return

    _ensure_cache_collection()
    q_client = get_client()

    # Create a unique cache_id via hash of user + query
    cache_id = hashlib.sha256(f"{user_id}:{query}".encode("utf-8")).hexdigest()

    try:
        # Store response in Redis with metadata
        cache_data = json.dumps({
            "response": response,
            "citations": citations or [],
            "user_id": user_id,
            "account_type": account_type,
            "knowledge_version": get_knowledge_version(),
            "created_at": time.time(),
        })
        redis_client.set(cache_id, cache_data, ex=ttl_seconds)

        # Store query embedding in Qdrant
        query_emb = embed_text(query)
        from qdrant_client.models import PointStruct

        q_client.upsert(
            collection_name=CACHE_COLLECTION,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=query_emb,
                    payload={"cache_id": cache_id, "query": query},
                )
            ],
        )
    except Exception as e:
        print(f"[SemanticCache] Error caching response: {e}")
