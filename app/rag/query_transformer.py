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
"""
from typing import Optional
from openai import AsyncOpenAI
from app.config import get_settings

HYDE_PROMPT = """You are an Islamic finance expert. Given the question below, 
write a short, factual paragraph (3-5 sentences) that would appear in an 
Islamic finance textbook or Sharia compliance document answering this question.
Use precise Islamic finance terminology (e.g., Murabaha, Musharaka, Sukuk, 
Riba, Ijara, Takaful, Mudaraba, Zakat, Gharar). 
Do NOT say "I think" or "In my opinion". Write as if this is a reference document.

Question: {query}

Passage:"""


async def generate_hypothetical_document(query: str) -> Optional[str]:
    """
    Use the LLM to generate a hypothetical passage that answers the query.
    Returns None if the LLM call fails (retrieval falls back to raw query).
    Now async to avoid blocking the event loop during the ~2s LLM call.
    """
    s = get_settings()
    if not s.openrouter_api_key:
        return None

    try:
        client = AsyncOpenAI(
            api_key=s.openrouter_api_key, 
            base_url=s.openrouter_base_url,
            timeout=3.0  # Strict 3-second timeout for HyDE
        )
        completion = await client.chat.completions.create(
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
