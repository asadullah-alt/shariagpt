"""
Admin Router — POST /admin/upload
-----------------------------------
Accepts a PDF (or Markdown) file upload, converts it to Markdown using
LlamaCloud's agentic parser, saves the .md to data/sharia_docs/, then
chunks with the heading-aware paragraph chunker and upserts to Qdrant.

Security: requires X-Admin-Key header matching ADMIN_API_KEY env var.

Usage:
    curl -X POST https://your-app.onrender.com/admin/upload \\
         -H "X-Admin-Key: your-secret" \\
         -F "file=@murabaha_2024.pdf"
"""
import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile, status
from pydantic import BaseModel
import cloudinary
import cloudinary.uploader
from app.rag.registry import register_pdf, load_registry, unregister_pdf

from app.config import get_settings
from app.rag.chunker import chunk_markdown
from app.rag.embedder import embed_texts
from app.rag.vector_store import collection_count, ensure_collection, upsert_chunks, delete_chunks_by_source
from app.rag.semantic_cache import increment_knowledge_version

router = APIRouter(prefix="/admin", tags=["Admin"])

# Configure Cloudinary if URL is available
settings = get_settings()
if settings.cloudinary_url:
    cloudinary.config(cloudinary_url=settings.cloudinary_url)

DOCS_DIR = Path("data/sharia_docs")
ALLOWED_TYPES = {"application/pdf", "text/markdown", "text/plain"}


# ── Auth dependency ───────────────────────────────────────────────────────────

def require_admin(x_admin_key: str = Header(..., alias="X-Admin-Key")) -> None:
    if x_admin_key != get_settings().admin_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin key")


# ── Response model ────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    filename: str
    source_name: str
    markdown_saved_to: str
    chunks_ingested: int
    total_docs_in_store: int


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_stem(filename: str) -> str:
    """Convert filename to a safe snake_case identifier."""
    stem = Path(filename).stem
    stem = re.sub(r"[^\w\s-]", "", stem)
    stem = re.sub(r"[\s-]+", "_", stem).lower().strip("_")
    return stem or "document"


async def _pdf_to_markdown(file_bytes: bytes, filename: str) -> str:
    """Use LlamaCloud agentic parser to convert PDF → Markdown."""
    from llama_cloud import AsyncLlamaCloud  # imported lazily — not installed in test env

    s = get_settings()
    if not s.llama_cloud_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLAMA_CLOUD_API_KEY is not configured on this server.",
        )

    client = AsyncLlamaCloud(api_key=s.llama_cloud_api_key)

    # Write bytes to a named temp file so LlamaCloud SDK can open it
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        file_obj = await client.files.create(file=tmp_path, purpose="parse")
        result = await client.parsing.parse(
            file_id=file_obj.id,
            tier="agentic",
            version="latest",
            expand=["markdown_full", "items"],
        )
        
        pages_md = []
        if result.items and hasattr(result.items, "pages") and result.items.pages:
            for p in result.items.pages:
                page_num = getattr(p, "page_number", 1)
                page_text = ""
                items = getattr(p, "items", []) or []
                for item in items:
                    val = ""
                    if hasattr(item, "md"):
                        val = item.md
                    elif isinstance(item, dict):
                        val = item.get("md")
                    if not val:
                        if hasattr(item, "value"):
                            val = item.value
                        elif isinstance(item, dict):
                            val = item.get("value")
                    if val:
                        page_text += str(val) + "\n"
                pages_md.append(f"<!-- PAGE_BREAK page=\"{page_num}\" -->\n{page_text}")
            return "\n\n".join(pages_md)

        return result.markdown_full or ""
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _ingest_markdown(markdown: str, source: str, id_offset: int) -> int:
    """Chunk, embed (dense), upsert. Returns number of chunks."""
    chunks = chunk_markdown(markdown, source=source, start_id=id_offset, max_words=400)
    if not chunks:
        return 0
    texts = [c.text for c in chunks]
    dense_embeddings = embed_texts(texts)

    from app.rag.registry import get_pdf_url
    pdf_url = get_pdf_url(source)

    records = [
        {
            "id": c.id,
            "text": c.text,
            "embedding": dense_emb,
            "source": c.source,
            "page_number": c.page_number,
            "pdf_url": pdf_url,
        }
        for c, dense_emb in zip(chunks, dense_embeddings)
    ]
    upsert_chunks(records)
    return len(records)


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse, dependencies=[Depends(require_admin)])
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload a PDF or Markdown file. PDFs are parsed to Markdown via LlamaCloud.
    The resulting Markdown is saved to data/sharia_docs/ and ingested into Qdrant.
    """
    content_type = file.content_type or ""
    filename = file.filename or "upload.pdf"

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    source_name = _safe_stem(filename)

    # ── Convert PDF → Markdown if needed ─────────────────────────────────────
    if "pdf" in content_type or filename.lower().endswith(".pdf"):
        try:
            upload_result = cloudinary.uploader.upload(
                file_bytes,
                resource_type="raw",
                public_id=f"shariagpt/{source_name}",
                filename=filename
            )
            cloudinary_url = upload_result.get("secure_url")
            if cloudinary_url:
                register_pdf(source_name, filename, cloudinary_url)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Cloudinary upload failed: {str(e)}"
            )

        markdown_text = await _pdf_to_markdown(file_bytes, filename)
        if not markdown_text.strip():
            raise HTTPException(status_code=422, detail="LlamaCloud returned empty Markdown.")
    else:
        # Already Markdown / plain text
        markdown_text = file_bytes.decode("utf-8", errors="replace")

    # ── Persist Markdown file ─────────────────────────────────────────────────
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = DOCS_DIR / f"{source_name}.md"
    md_path.write_text(markdown_text, encoding="utf-8")

    # ── Ingest into Qdrant ────────────────────────────────────────────────────
    ensure_collection()

    # Use a high ID offset based on current collection size to avoid collisions
    current_count = collection_count()
    id_offset = (current_count + 1) * 10_000

    num_chunks = _ingest_markdown(markdown_text, source=source_name, id_offset=id_offset)

    # Invalidate cache globally since new knowledge was added
    increment_knowledge_version()

    return UploadResponse(
        filename=filename,
        source_name=source_name,
        markdown_saved_to=str(md_path),
        chunks_ingested=num_chunks,
        total_docs_in_store=collection_count(),
    )


@router.get("/docs", dependencies=[Depends(require_admin)])
async def list_docs() -> dict:
    """List all Markdown documents currently in the knowledge base folder."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    files = [
        {"name": f.name, "size_kb": round(f.stat().st_size / 1024, 1)}
        for f in sorted(DOCS_DIR.glob("*.md"))
    ]
    return {
        "docs_dir": str(DOCS_DIR),
        "documents": files,
        "total_chunks_in_store": collection_count(),
    }


@router.get("/pdfs")
async def list_pdfs() -> dict:
    """List all registered PDFs and their Cloudinary URLs."""
    return {"pdfs": load_registry()}


@router.delete("/pdfs/{source_name}", dependencies=[Depends(require_admin)])
async def delete_pdf(source_name: str) -> dict:
    """Delete a PDF from Cloudinary, Qdrant, and local cache."""
    registry = load_registry()
    if source_name not in registry:
        raise HTTPException(status_code=404, detail="Document not found")
        
    try:
        # 1. Delete from Cloudinary
        if settings.cloudinary_url:
            cloudinary.uploader.destroy(f"shariagpt/{source_name}", resource_type="raw")
            
        # 2. Delete chunks from Qdrant
        delete_chunks_by_source(source_name)
        
        # 3. Delete local markdown file
        md_path = DOCS_DIR / f"{source_name}.md"
        if md_path.exists():
            md_path.unlink()
            
        # 4. Remove from registry
        unregister_pdf(source_name)
        
        # 5. Invalidate cache
        increment_knowledge_version()
        
        return {"message": f"Successfully deleted {source_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {e}")
