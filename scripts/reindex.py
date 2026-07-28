"""
Re-index Script
───────────────
Drops and recreates the Qdrant collection with hybrid vector config,
then re-ingests all Markdown documents from data/sharia_docs/.

Usage:
    python -m scripts.reindex
"""
import sys
from pathlib import Path

# Ensure the project root is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.rag.chunker import chunk_markdown
from app.rag.embedder import embed_texts, sparse_encode_texts
from app.rag.vector_store import get_client, ensure_collection, upsert_chunks
from app.rag.registry import get_pdf_url

DOCS_DIR = Path("data/sharia_docs")


def main():
    s = get_settings()
    client = get_client()

    # ── Step 1: Drop existing collection ─────────────────────────────────────
    try:
        names = [c.name for c in client.get_collections().collections]
        if s.qdrant_collection in names:
            print(f"[Reindex] Dropping collection '{s.qdrant_collection}'...")
            client.delete_collection(s.qdrant_collection)
    except Exception as e:
        print(f"[Reindex] Warning during drop: {e}")

    # ── Step 2: Recreate with hybrid config ──────────────────────────────────
    print("[Reindex] Creating collection with dense and sparse vectors...")
    ensure_collection()

    # ── Step 3: Find all Markdown docs ───────────────────────────────────────
    md_files = sorted(DOCS_DIR.glob("*.md"))
    if not md_files:
        print(f"[Reindex] No .md files found in {DOCS_DIR}. Nothing to index.")
        return

    print(f"[Reindex] Found {len(md_files)} documents to re-index.")

    total_chunks = 0
    id_offset = 0

    for md_file in md_files:
        source = md_file.stem
        text = md_file.read_text(encoding="utf-8")
        print(f"  - Indexing '{source}'...")

        chunks = chunk_markdown(text, source=source, start_id=id_offset, max_words=400)
        if not chunks:
            print(f"    (no chunks produced, skipping)")
            continue

        texts = [c.text for c in chunks]

        print(f"    Computing dense embeddings for {len(texts)} chunks...")
        dense_embeddings = embed_texts(texts)
        
        print(f"    Computing sparse embeddings for {len(texts)} chunks...")
        sparse_embeddings = sparse_encode_texts(texts)

        pdf_url = get_pdf_url(source)

        records = [
            {
                "id": c.id,
                "text": c.text,
                "embedding": dense_emb,
                "sparse_embedding": sparse_emb,
                "source": c.source,
                "page_number": c.page_number,
                "pdf_url": pdf_url,
            }
            for c, dense_emb, sparse_emb in zip(chunks, dense_embeddings, sparse_embeddings)
        ]

        upsert_chunks(records)
        total_chunks += len(records)
        id_offset += len(records) * 10_000

        print(f"    > Upserted {len(records)} chunks")

    print(f"\n[Reindex] Done! Total chunks indexed: {total_chunks}")


if __name__ == "__main__":
    main()
