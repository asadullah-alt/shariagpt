"""
Query Transformer — HyDE (Hypothetical Document Embeddings)
────────────────────────────────────────────────────────────
Pre-processes the raw user query before Stage 1 retrieval:

1. HyDE: Uses the LLM to generate a hypothetical document passage that
   would answer the query. The passage contains domain-specific terminology
   (e.g. "Murabaha", "profit margin", "cost-plus") that pulls the embedding
   closer to actual stored documents, dramatically improving recall.

2. The ORIGINAL query embedding is ALSO used alongside the HyDE embedding,
   so we never lose the literal user intent.

Flow:
    [Raw Query] → LLM → [Hypothetical Passage]
                  ↓
    embed(query) + embed(hypothesis) → averaged dense vector
    sparse(query)                    → sparse vector (unchanged)
"""
from typing import Optional
from openai import OpenAI
from app.config import get_settings

HYDE_PROMPT = """You are an Islamic finance expert. Given the question below, 
write a short, factual paragraph (3-5 sentences) that would appear in an 
Islamic finance textbook or Sharia compliance document answering this question.
Use precise Islamic finance terminology (e.g., Murabaha, Musharaka, Sukuk, 
Riba, Ijara, Takaful, Mudaraba, Zakat, Gharar). 
Do NOT say "I think" or "In my opinion". Write as if this is a reference document.

Question: {query}

Passage:"""


def generate_hypothetical_document(query: str) -> Optional[str]:
    """
    Use the LLM to generate a hypothetical passage that answers the query.
    Returns None if the LLM call fails (retrieval falls back to raw query).
    """
    s = get_settings()
    if not s.openrouter_api_key:
        return None

    try:
        client = OpenAI(api_key=s.openrouter_api_key, base_url=s.openrouter_base_url)
        completion = client.chat.completions.create(
            model=s.openrouter_model,
            messages=[
                {"role": "user", "content": HYDE_PROMPT.format(query=query)},
            ],
            max_tokens=200,
            temperature=0.0,
            extra_headers={
                "HTTP-Referer": "https://shariagpt.onrender.com",
                "X-Title": "ShariaGPT-HyDE",
            },
        )
        return completion.choices[0].message.content or None
    except Exception as e:
        print(f"[HyDE] LLM call failed, falling back to raw query: {e}")
        return None


def transform_query(query: str) -> dict:
    """
    Transform a raw user query for improved retrieval.

    Returns
    -------
    dict with keys:
        - "dense_vectors": list of dense embeddings to average for search
        - "sparse_vector": sparse vector from the original query
        - "hyde_passage":  the generated hypothetical passage (or None)
    """
    from app.rag.embedder import embed_text, sparse_encode_text

    # Always compute the raw query vectors
    raw_dense = embed_text(query)
    sparse_vec = sparse_encode_text(query)

    # Generate HyDE passage
    hyde_passage = generate_hypothetical_document(query)

    if hyde_passage:
        hyde_dense = embed_text(hyde_passage)
        # Average the raw query and HyDE embeddings for a blended dense vector
        blended_dense = [
            (r + h) / 2.0 for r, h in zip(raw_dense, hyde_dense)
        ]
        return {
            "dense_vector": blended_dense,
            "sparse_vector": sparse_vec,
            "hyde_passage": hyde_passage,
        }

    # Fallback: no HyDE, just raw query
    return {
        "dense_vector": raw_dense,
        "sparse_vector": sparse_vec,
        "hyde_passage": None,
    }
