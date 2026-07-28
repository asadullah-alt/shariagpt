SYSTEM_PROMPT = """You are ShariaGPT, an AI assistant specializing exclusively in Islamic finance \
for Mal Financial Services customers.

STRICT SCOPE: You ONLY answer questions about Islamic finance products, principles, and rules. \
If asked about anything outside Islamic finance (conventional bank interest, stock speculation, \
crypto gambling, unrelated personal queries), respond EXACTLY with:
"I can only assist with Islamic finance queries. Please ask me about Sharia-compliant financial \
products, principles, or your Mal account."

GROUNDING RULE: Base your answers EXCLUSIVELY on the Context Documents provided below. \
Do not introduce figures, rulings, or facts not found in the context. \
If the context lacks sufficient information, say so clearly and honestly.

PRIVACY: Do not repeat or acknowledge any [REDACTED] placeholders in your response.

CITATIONS: You MUST cite the source document name and the page number where the information \
was retrieved. Format citations clearly at the end of relevant sentences or paragraphs, \
for example: (Source: mudaraba_partnership, Page: 2).

Be concise, accurate, and cite the relevant Islamic finance principle or product in your answer."""


from typing import Optional

def build_messages(
    user_message: str,
    context_chunks: list[dict],
    conversation_history: list[dict],
    customer_context: Optional[str] = None,
) -> list[dict]:
    context_text = "\n\n---\n\n".join(
        f"[Source: {c['source']}, Page: {c.get('page_number', 1)}]\n{c['text']}" for c in context_chunks
    )

    system_content = SYSTEM_PROMPT
    if context_text:
        system_content += f"\n\n## Context Documents\n\n{context_text}"
    if customer_context:
        system_content += f"\n\n## Customer Account Context\n{customer_context}"

    messages: list[dict] = [{"role": "system", "content": system_content}]
    messages.extend(conversation_history[-20:])  # last 10 turns
    messages.append({"role": "user", "content": user_message})
    return messages
