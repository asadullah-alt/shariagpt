import time
import logging
import uuid
import pybreaker
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from openai import OpenAI
from typing import Optional

from app.config import get_settings
from app.pii.redactor import redact, restore
from app.rag.retriever import retrieve
from app.rag.prompt_builder import build_messages
from app.rag.semantic_cache import get_cached_response, set_cached_response
from app.rag.intent_router import route_intent
from app.observability.tracer import TraceRecord, emit_trace
from app.sessions.store import get_session_store
from app.auth.jwt_handler import require_auth
from app.auth.user_store import find_user_by_email

router = APIRouter()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Circuit breaker for LLM calls: opens after 5 consecutive failures, resets after 60s
llm_breaker = pybreaker.CircuitBreaker(fail_max=5, reset_timeout=60)


class ChatRequest(BaseModel):
    session_id: str
    message: str
    customer_context: Optional[dict] = None


class Citation(BaseModel):
    source: str
    page_number: int
    pdf_url: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    request_id: str
    response: str
    retrieved_chunk_ids: list[str]
    relevance_scores: list[float]
    avg_relevance_score: float
    pii_detected: list[str]
    cache_hit: bool = False
    citations: list[Citation] = []


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    req: ChatRequest,
    token_payload: dict = Depends(require_auth)
) -> ChatResponse:
    s = get_settings()
    request_id = str(uuid.uuid4())
    t0 = time.monotonic()

    # ── 1. PII Redaction ────────────────────────────────────────────────────
    redaction = redact(req.message)
    clean_message = redaction.redacted_text
    pii_types = list(redaction.detected_types)
    pii_mapping = redaction.mapping.copy()
    
    t_pii = time.monotonic()
    logger.info(f"[Chat] PII redaction completed in {round((t_pii - t0) * 1000, 2)}ms")

    clean_context: Optional[str] = None
    if req.customer_context:
        ctx_str = str(req.customer_context)
        ctx_red = redact(ctx_str)
        clean_context = ctx_red.redacted_text
        pii_types += ctx_red.detected_types
        pii_mapping.update(ctx_red.mapping)

    user_id = token_payload.get("sub")
    account_type = "guest"
    
    user = find_user_by_email(user_id)
    if user:
        account_type = user.get("account_type", "guest")
        # Override request context with authoritative DB context
        clean_context = (
            f"Name: {user['name']}\n"
            f"Emirates ID: {user['emirates_id']}\n"
            f"Account Number: {user['account_number']}\n"
            f"Account Type: {user['account_type']}\n"
            f"Balance: {user['balance']}"
        )
        # Redact the DB context before sending to LLM
        ctx_red = redact(clean_context)
        clean_context = ctx_red.redacted_text
        pii_types += ctx_red.detected_types
        pii_mapping.update(ctx_red.mapping)

    # ── 1.5 Intent Routing ──────────────────────────────────────────────────
    intent_result = route_intent(clean_message)
    intent = intent_result.get("intent", "SHARIA_FINANCE")
    action = intent_result.get("action")

    t_intent = time.monotonic()
    logger.info(f"[Chat] Intent routing completed in {round((t_intent - t_pii) * 1000, 2)}ms: {intent}")

    chunks = []
    chunk_ids = []
    scores = []
    avg_score = 0.0
    citations = []
    usage = None
    cache_hit = False

    if intent == "OUT_OF_SCOPE":
        answer = "I am not equipped to answer that."
    elif intent == "ACCOUNT_INFO" and user:
        # ── Bank API Flow ───────────────────────────────────────────────────
        try:
            import json
            from app.services import bank_api
            api_result = {}
            if action == "CHECK_BALANCE":
                api_result = bank_api.get_account_balance(user['account_number'])
            elif action == "RECENT_TRANSACTIONS":
                api_result = bank_api.get_recent_transactions(user['account_number'])
            elif action == "ACCOUNT_STATUS":
                api_result = bank_api.get_account_status(user['account_number'])
            
            client = OpenAI(
                api_key=s.openrouter_api_key, 
                base_url=s.openrouter_base_url,
                timeout=10.0 
            )
            prompt = f"User asked: {clean_message}\nBank API returned: {json.dumps(api_result)}\nAnswer the user based on the API result. Keep it brief. If the API returned empty, say you could not find the info."
            completion = llm_breaker.call(
                client.chat.completions.create,
                model=s.openrouter_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.0
            )
            answer = completion.choices[0].message.content or ""
            usage = completion.usage
        except pybreaker.CircuitBreakerError:
            answer = "I am unable to retrieve your account information at this time because the banking service is unavailable."
        except Exception as e:
            answer = f"I am unable to retrieve your account information at this time. ({str(e)})"
            
    else:
        # ── SHARIA_FINANCE Flow (RAG + Cache) ───────────────────────────────
        cached_data = get_cached_response(clean_message, user_id=user_id, account_type=account_type)
        
        t_cache = time.monotonic()
        logger.info(f"[Chat] Semantic cache check completed in {round((t_cache - t_intent) * 1000, 2)}ms")
        
        if cached_data:
            if isinstance(cached_data, dict):
                answer = cached_data.get("response", "")
                citations = cached_data.get("citations", [])
            else:
                answer = cached_data
            cache_hit = True
        else:
            chunks = retrieve(clean_message, k=s.top_k_chunks)
            t_retrieve = time.monotonic()
            
            chunk_ids = [c["chunk_id"] for c in chunks]
            scores = [c["score"] for c in chunks]
            avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0

            from app.rag.registry import get_pdf_url
            seen_citations = set()
            for c in chunks:
                cit_key = (c["source"], c.get("page_number", 1))
                if cit_key not in seen_citations:
                    seen_citations.add(cit_key)
                    source_name = c["source"]
                    pdf_url = c.get("pdf_url") or get_pdf_url(source_name)
                    if pdf_url:
                        citations.append({
                            "source": source_name,
                            "page_number": c.get("page_number", 1),
                            "pdf_url": pdf_url
                        })

            store = get_session_store()
            history = store.get_history(req.session_id)

            messages = build_messages(
                user_message=clean_message,
                context_chunks=chunks,
                conversation_history=history,
                customer_context=clean_context,
            )
            
            client = OpenAI(
                api_key=s.openrouter_api_key, 
                base_url=s.openrouter_base_url,
                timeout=30.0 
            )
            try:
                completion = llm_breaker.call(
                    client.chat.completions.create,
                    model=s.openrouter_model,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.1,
                    extra_headers={"HTTP-Referer": "https://shariagpt.onrender.com", "X-Title": "ShariaGPT"},
                )
                answer = completion.choices[0].message.content or ""
                usage = completion.usage
            except pybreaker.CircuitBreakerError:
                raise HTTPException(status_code=503, detail="LLM service is temporarily unavailable. Please try again later.")
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"LLM request failed: {str(e)}")

    t_llm = time.monotonic()
    logger.info(f"[Chat] Generation completed in {round((t_llm - t_intent) * 1000, 2)}ms")

    # ── 6. Persist turn & Cache ──────────────────────────────────────────────
    store = get_session_store()
    store.append_turn(req.session_id, clean_message, answer, user_id)
    if not cache_hit and intent == "SHARIA_FINANCE":
        set_cached_response(
            query=clean_message, 
            response=answer, 
            user_id=user_id, 
            account_type=account_type, 
            citations=citations
        )

    # ── 7. Anonymization Reverter ────────────────────────────────────────────
    final_answer = restore(answer, pii_mapping)

    # ── 8. Emit trace ────────────────────────────────────────────────────────
    latency_ms = round((time.monotonic() - t0) * 1000, 2)
    emit_trace(
        TraceRecord(
            request_id=request_id,
            session_id=req.session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            latency_ms=latency_ms,
            model="cache" if cache_hit else s.openrouter_model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            chunk_ids=chunk_ids,
            relevance_scores=scores,
            avg_relevance_score=avg_score,
            pii_detected=list(set(pii_types)),
            query_length=len(req.message),
            response_length=len(final_answer),
            cache_hit=cache_hit,
        ),
        log_dir=s.log_dir,
    )

    return ChatResponse(
        session_id=req.session_id,
        request_id=request_id,
        response=final_answer,
        retrieved_chunk_ids=chunk_ids,
        relevance_scores=scores,
        avg_relevance_score=avg_score,
        pii_detected=list(set(pii_types)),
        cache_hit=cache_hit,
        citations=citations,
    )

@router.get("/chat/sessions")
async def get_user_chats(token_payload: dict = Depends(require_auth)):
    user_id = token_payload.get("sub")
    store = get_session_store()
    chats = store.get_user_chats(user_id)
    return {"chats": chats}

@router.get("/chat/sessions/{session_id}")
async def get_chat_history(session_id: str, token_payload: dict = Depends(require_auth)):
    user_id = token_payload.get("sub")
    store = get_session_store()
    
    # We must ensure the session belongs to the user
    # A simple check: get user's chats and see if session_id is in them
    chats = store.get_user_chats(user_id)
    if not any(c.get("session_id") == session_id for c in chats):
        raise HTTPException(status_code=404, detail="Chat not found or access denied")
        
    history = store.get_history(session_id)
    return {"history": history}
