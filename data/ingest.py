"""
Sharia Knowledge Base Ingestion Script
---------------------------------------
Loads Markdown documents from data/sharia_docs/, chunks them with the
heading-aware paragraph chunker, embeds with sentence-transformers, and
upserts into Qdrant Cloud.

Run manually:   python data/ingest.py
Auto-runs:      on startup if the Qdrant collection is empty.
"""
import sys
from pathlib import Path

# Allow direct execution: python data/ingest.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.chunker import chunk_markdown
from app.rag.embedder import embed_texts
from app.rag.vector_store import ensure_collection, upsert_chunks

DOCS_DIR = Path(__file__).parent / "sharia_docs"


def ingest_markdown(text: str, source: str, id_offset: int) -> int:
    """
    Chunk, embed, and upsert a single Markdown document.
    Returns number of chunks inserted.
    """
    chunks = chunk_markdown(text, source=source, start_id=id_offset, max_words=400)
    if not chunks:
        return 0

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    records = [
        {
            "id": c.id,
            "text": c.text,
            "embedding": emb,
            "source": c.source,
        }
        for c, emb in zip(chunks, embeddings)
    ]
    upsert_chunks(records)
    return len(records)


def run_ingestion() -> int:
    """
    Ingest all *.md files in data/sharia_docs/.
    Returns total chunks upserted.
    """
    ensure_collection()
    total = 0
    id_offset = 1000  # base ID; each doc gets its own namespace

    md_files = sorted(DOCS_DIR.glob("*.md"))
    if not md_files:
        print("[Ingest] No .md files found in", DOCS_DIR)
        return 0

    for idx, md_file in enumerate(md_files):
        text = md_file.read_text(encoding="utf-8")
        source = md_file.stem
        # Each document gets a block of 1000 IDs so they never collide
        doc_offset = id_offset + idx * 1000
        count = ingest_markdown(text, source=source, id_offset=doc_offset)
        total += count
        print(f"  '{md_file.name}' → {count} chunk(s)  (IDs {doc_offset}–{doc_offset+count-1})")

    print(f"[Ingest] ✓ Total {total} chunks upserted into Qdrant.")
    return total


if __name__ == "__main__":
    n = run_ingestion()
    print(f"Done. Chunks: {n}")
