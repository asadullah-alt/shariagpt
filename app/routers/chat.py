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
from app.pii.redactor import redact
from app.rag.retriever import retrieve
from app.rag.prompt_builder import build_messages
from app.rag.semantic_cache import get_cached_response, set_cached_response
from app.observability.tracer import TraceRecord, emit_trace
from app.sessions.store import get_session_store
from app.auth.jwt_handler import optional_auth
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
    token_payload: Optional[dict] = Depends(optional_auth)
) -> ChatResponse:
    s = get_settings()
    request_id = str(uuid.uuid4())
    t0 = time.monotonic()

    # ── 1. PII Redaction ────────────────────────────────────────────────────
    redaction = redact(req.message)
    clean_message = redaction.redacted_text
    pii_types = list(redaction.detected_types)
    
    t_pii = time.monotonic()
    logger.info(f"[Chat] PII redaction completed in {round((t_pii - t0) * 1000, 2)}ms")

    clean_context: Optional[str] = None
    if req.customer_context:
        ctx_str = str(req.customer_context)
        ctx_red = redact(ctx_str)
        clean_context = ctx_red.redacted_text
        pii_types += ctx_red.detected_types

    user_id = "guest"
    account_type = "guest"
    
    if token_payload:
        user_id = token_payload.get("sub")
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

    # ── 2. Check Semantic Cache ─────────────────────────────────────────────
    cached_data = get_cached_response(clean_message, user_id=user_id, account_type=account_type)
    
    t_cache = time.monotonic()
    logger.info(f"[Chat] Semantic cache check completed in {round((t_cache - t_pii) * 1000, 2)}ms")
    
    if cached_data:
        if isinstance(cached_data, dict):
            cached_answer = cached_data.get("response", "")
            cached_citations = cached_data.get("citations", [])
        else:
            cached_answer = cached_data
            cached_citations = []
            
        # ── 2a. Cache Hit Flow ──────────────────────────────────────────────
        latency_ms = round((time.monotonic() - t0) * 1000, 2)
        
        # Persist turn
        store = get_session_store()
        store.append_turn(req.session_id, clean_message, cached_answer, user_id)

        emit_trace(
            TraceRecord(
                request_id=request_id,
                session_id=req.session_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                latency_ms=latency_ms,
                model="cache",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                chunk_ids=[],
                relevance_scores=[],
                avg_relevance_score=0.0,
                pii_detected=list(set(pii_types)),
                query_length=len(req.message),
                response_length=len(cached_answer),
                cache_hit=True,
            ),
            log_dir=s.log_dir,
        )

        return ChatResponse(
            session_id=req.session_id,
            request_id=request_id,
            response=cached_answer,
            retrieved_chunk_ids=[],
            relevance_scores=[],
            avg_relevance_score=0.0,
            pii_detected=list(set(pii_types)),
            cache_hit=True,
            citations=cached_citations,
        )

    # ── 3. Retrieve chunks (Cache Miss Flow) ────────────────────────────────
    chunks = retrieve(clean_message, k=s.top_k_chunks)
    
    t_retrieve = time.monotonic()
    logger.info(f"[Chat] Vector retrieval completed in {round((t_retrieve - t_cache) * 1000, 2)}ms")
    
    chunk_ids = [c["chunk_id"] for c in chunks]
    scores = [c["score"] for c in chunks]
    avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0

    # Extract unique citations
    from app.rag.registry import get_pdf_url
    seen_citations = set()
    citations = []
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

    # ── 3. Session history ──────────────────────────────────────────────────
    store = get_session_store()
    history = store.get_history(req.session_id)

    # ── 4. Build prompt ─────────────────────────────────────────────────────
    messages = build_messages(
        user_message=clean_message,
        context_chunks=chunks,
        conversation_history=history,
        customer_context=clean_context,
    )
    
    t_prompt = time.monotonic()
    logger.info(f"[Chat] Prompt building completed in {round((t_prompt - t_retrieve) * 1000, 2)}ms")

    # ── 5. LLM via OpenRouter ───────────────────────────────────────────────
    client = OpenAI(
        api_key=s.openrouter_api_key, 
        base_url=s.openrouter_base_url,
        timeout=30.0  # 30-second timeout to prevent Render 502s
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
    except pybreaker.CircuitBreakerError:
        raise HTTPException(status_code=503, detail="LLM service is temporarily unavailable. Please try again later.")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {str(e)}")

    answer = completion.choices[0].message.content or ""
    usage = completion.usage

    t_llm = time.monotonic()
    logger.info(f"[Chat] LLM generation completed in {round((t_llm - t_prompt) * 1000, 2)}ms")

    # ── 6. Persist turn & Cache ──────────────────────────────────────────────
    store.append_turn(req.session_id, clean_message, answer, user_id)
    set_cached_response(
        query=clean_message, 
        response=answer, 
        user_id=user_id, 
        account_type=account_type, 
        citations=citations
    )

    # ── 7. Emit trace ────────────────────────────────────────────────────────
    latency_ms = round((time.monotonic() - t0) * 1000, 2)
    emit_trace(
        TraceRecord(
            request_id=request_id,
            session_id=req.session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            latency_ms=latency_ms,
            model=s.openrouter_model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            chunk_ids=chunk_ids,
            relevance_scores=scores,
            avg_relevance_score=avg_score,
            pii_detected=list(set(pii_types)),
            query_length=len(req.message),
            response_length=len(answer),
            cache_hit=False,
        ),
        log_dir=s.log_dir,
    )

    return ChatResponse(
        session_id=req.session_id,
        request_id=request_id,
        response=answer,
        retrieved_chunk_ids=chunk_ids,
        relevance_scores=scores,
        avg_relevance_score=avg_score,
        pii_detected=list(set(pii_types)),
        cache_hit=False,
        citations=citations,
    )

@router.get("/chat/sessions")
async def get_user_chats(token_payload: dict = Depends(optional_auth)):
    if not token_payload:
        raise HTTPException(status_code=401, detail="Authentication required")
    user_id = token_payload.get("sub")
    store = get_session_store()
    chats = store.get_user_chats(user_id)
    return {"chats": chats}

@router.get("/chat/sessions/{session_id}")
async def get_chat_history(session_id: str, token_payload: dict = Depends(optional_auth)):
    if not token_payload:
        raise HTTPException(status_code=401, detail="Authentication required")
    user_id = token_payload.get("sub")
    store = get_session_store()
    
    # We must ensure the session belongs to the user
    # A simple check: get user's chats and see if session_id is in them
    chats = store.get_user_chats(user_id)
    if not any(c.get("session_id") == session_id for c in chats):
        raise HTTPException(status_code=404, detail="Chat not found or access denied")
        
    history = store.get_history(session_id)
    return {"history": history}
