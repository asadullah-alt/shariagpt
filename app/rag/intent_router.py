import json
import logging
from openai import OpenAI
from app.config import get_settings

logger = logging.getLogger(__name__)

INTENT_PROMPT = """You are an intelligent router for an Islamic Finance AI Assistant.
Analyze the user's query and classify it into one of three intents:

1. "SHARIA_FINANCE": The user is asking about Islamic finance concepts, rules, Sharia compliance, or general knowledge (e.g., "What is Murabaha?", "How does Sukuk work?").
2. "ACCOUNT_INFO": The user is asking about their personal bank account details (e.g., "What is my balance?", "Show me my recent transactions", "What is my account status?", "How much money do I have?").
3. "OUT_OF_SCOPE": The user is asking a question that is completely unrelated to Islamic finance or their bank account (e.g., "What's the weather like?", "Who won the game?").

If the intent is "ACCOUNT_INFO", also specify the "action" from the following options: "CHECK_BALANCE", "RECENT_TRANSACTIONS", "ACCOUNT_STATUS", or "UNKNOWN".
If the intent is not "ACCOUNT_INFO", the action should be null.

Respond ONLY with a valid JSON object in this format:
{"intent": "...", "action": "..."}

User Query: {query}
"""

def route_intent(query: str) -> dict:
    """Route a sanitized query to the appropriate intent category."""
    s = get_settings()
    try:
        from app.routers.chat import llm_breaker
        client = OpenAI(
            api_key=s.openrouter_api_key, 
            base_url=s.openrouter_base_url,
            timeout=5.0 
        )
        completion = llm_breaker.call(
            client.chat.completions.create,
            model=s.openrouter_model,
            messages=[
                {"role": "system", "content": INTENT_PROMPT.format(query=query)}
            ],
            max_tokens=100,
            temperature=0.0,
            response_format={"type": "json_object"},
            extra_headers={
                "HTTP-Referer": "https://shariagpt.onrender.com",
                "X-Title": "ShariaGPT-Intent",
            },
        )
        response_text = completion.choices[0].message.content
        return json.loads(response_text)
    except Exception as e:
        logger.warning(f"[IntentRouter] Routing failed, defaulting to SHARIA_FINANCE. Error: {e}")
        # Default to SHARIA_FINANCE to maintain fallback RAG behavior
        return {"intent": "SHARIA_FINANCE", "action": None}
