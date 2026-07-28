"""
ShariaGPT — Islamic Finance AI Assistant
FastAPI application entry point.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import get_settings
from app.rag.vector_store import ensure_collection, collection_count
from app.routers import chat, health, admin, auth


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    s = get_settings()
    print(f"[ShariaGPT] Starting v{s.app_version} | model={s.openrouter_model}")

    # Ensure Qdrant collection exists
    ensure_collection()

    # Auto-ingest if collection is empty
    doc_count = collection_count()
    if doc_count == 0:
        print("[ShariaGPT] Collection empty — running document ingestion…")
        from data.ingest import run_ingestion
        run_ingestion()
        print(f"[ShariaGPT] Ingestion complete. Docs: {collection_count()}")
    else:
        print(f"[ShariaGPT] Vector store ready. Docs: {doc_count}")

    yield
    # ── Shutdown ─────────────────────────────────────────────────────────────
    print("[ShariaGPT] Shutting down.")


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(
        title="ShariaGPT",
        description="Islamic Finance AI Assistant with RAG, PII Redaction & Observability",
        version=s.app_version,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat.router)
    app.include_router(health.router)
    app.include_router(admin.router)
    app.include_router(auth.router)

    # Serve static frontend
    import os
    if os.path.isdir("frontend"):
        app.mount("/static", StaticFiles(directory="frontend"), name="static")

        @app.get("/")
        async def root():
            return FileResponse("frontend/index.html")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=10000)
