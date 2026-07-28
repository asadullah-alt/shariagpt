import time
from fastapi import APIRouter
from app.config import get_settings
from app.rag.vector_store import collection_count
from app.sessions.store import get_session_store

router = APIRouter()
_started_at = time.monotonic()


@router.get("/health")
async def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "version": s.app_version,
        "model": s.openrouter_model,
        "vector_store_docs": collection_count(),
        "active_sessions": get_session_store().active_sessions(),
        "uptime_seconds": round(time.monotonic() - _started_at, 1),
    }
