import time
import logging
import uuid
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
from typing import Optional

from app.config import get_settings
from app.pii.redactor import redact, restore
from app.pii.stream_reverter import stream_restore
from app.rag.retriever import retrieve
from app.rag.prompt_builder import build_messages
from app.rag.semantic_cache import get_cached_response, set_cached_response
from app.rag.intent_router import route_intent
from app.observability.tracer import TraceRecord, emit_trace
from app.sessions.store import get_session_store
from app.auth.jwt_handler import require_auth
from app.auth.user_store import find_user_by_email
from app.security.guardrails import check_prompt_injection
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

router = APIRouter()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ChatRequest(BaseModel):
    session_id: str
    message: str
    customer_context: Optional[dict] = None


class FeedbackRequest(BaseModel):
    run_id: str
    score: int
    comment: Optional[str] = None


@router.post("/chat/feedback")
async def chat_feedback(req: FeedbackRequest, token_payload: dict = Depends(require_auth)):
    try:
        from langsmith import Client
        ls_client = Client()
        ls_client.create_feedback(
            run_id=req.run_id,
            key="user_score",
            score=req.score,
            comment=req.comment
        )
        
        # Save locally for admin
        feedback_file = Path("data/feedback.jsonl")
        with open(feedback_file, "a") as f:
            f.write(json.dumps({
                "run_id": req.run_id,
                "score": req.score,
                "comment": req.comment,
                "user_id": token_payload.get("sub"),
                "timestamp": time.time()
            }) + "\n")
            
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")


@router.post("/chat")
@traceable(run_type="chain", name="chat_pipeline")
async def chat_endpoint(
    req: ChatRequest,
    token_payload: dict = Depends(require_auth)
):
    s = get_settings()
    request_id = str(uuid.uuid4())
    t0 = time.monotonic()

    # ── 0. Prompt Injection Guardrail ───────────────────────────────────────
    if check_prompt_injection(req.message):
        raise HTTPException(status_code=400, detail="Security policy violation: Prompt injection detected.")

    # ── 1. PII Redaction ────────────────────────────────────────────────────
    redaction = redact(req.message)
    clean_message = redaction.redacted_text
    pii_types = list(redaction.detected_types)
    pii_mapping = redaction.mapping.copy()
    
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
        clean_context = (
            f"Name: {user['name']}\n"
            f"Emirates ID: {user['emirates_id']}\n"
            f"Account Number: {user['account_number']}\n"
            f"Account Type: {user['account_type']}\n"
            f"Balance: {user['balance']}"
        )
        ctx_red = redact(clean_context)
        clean_context = ctx_red.redacted_text
        pii_types += ctx_red.detected_types
        pii_mapping.update(ctx_red.mapping)

    # ── 1.5 Intent Routing ──────────────────────────────────────────────────
    intent_result = route_intent(clean_message)
    intent = intent_result.get("intent", "SHARIA_FINANCE")
    action = intent_result.get("action")

    # We will generate a stream of Server-Sent Events (SSE)
    async def event_generator():
        run_tree = get_current_run_tree()
        run_id = str(run_tree.id) if run_tree else None
        
        chunk_ids = []
        scores = []
        avg_score = 0.0
        citations = []
        cache_hit = False
        final_answer = ""
        
        aclient = AsyncOpenAI(
            api_key=s.openrouter_api_key, 
            base_url=s.openrouter_base_url,
            timeout=30.0 
        )

        if intent == "OUT_OF_SCOPE":
            final_answer = "I am not equipped to answer that."
            yield f"data: {json.dumps({'content': final_answer})}\n\n"
        
        elif intent == "ACCOUNT_INFO" and user:
            # ── Bank API Flow ───────────────────────────────────────────────────
            from app.services import bank_api
            api_result = {}
            if action == "CHECK_BALANCE":
                api_result = bank_api.get_account_balance(user['account_number'])
            elif action == "RECENT_TRANSACTIONS":
                api_result = bank_api.get_recent_transactions(user['account_number'])
            elif action == "ACCOUNT_STATUS":
                api_result = bank_api.get_account_status(user['account_number'])
            
            prompt = f"User asked: {clean_message}\nBank API returned: {json.dumps(api_result)}\nAnswer the user based on the API result. Keep it brief. If the API returned empty, say you could not find the info."
            
            try:
                stream_res = await aclient.chat.completions.create(
                    model=s.openrouter_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.0,
                    stream=True
                )
                
                async def llm_token_gen():
                    async for chunk in stream_res:
                        content = chunk.choices[0].delta.content
                        if content:
                            yield content

                async for safe_token in stream_restore(llm_token_gen(), pii_mapping):
                    final_answer += safe_token
                    yield f"data: {json.dumps({'content': safe_token})}\n\n"
                    
            except Exception as e:
                err_msg = "I am unable to retrieve your account information at this time."
                final_answer = err_msg
                yield f"data: {json.dumps({'content': err_msg})}\n\n"
                
        else:
            # ── SHARIA_FINANCE Flow (RAG + Cache) ───────────────────────────────
            cached_data = get_cached_response(clean_message, user_id=user_id, account_type=account_type)
            
            if cached_data:
                if isinstance(cached_data, dict):
                    final_answer = cached_data.get("response", "")
                    citations = cached_data.get("citations", [])
                else:
                    final_answer = cached_data
                cache_hit = True
                
                final_answer = restore(final_answer, pii_mapping)
                yield f"data: {json.dumps({'content': final_answer})}\n\n"
            else:
                chunks = await retrieve(clean_message, k=s.top_k_chunks)
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
                
                try:
                    stream_res = await aclient.chat.completions.create(
                        model=s.openrouter_model,
                        messages=messages,
                        max_tokens=1024,
                        temperature=0.1,
                        stream=True,
                        extra_headers={"HTTP-Referer": "https://shariagpt.onrender.com", "X-Title": "ShariaGPT"},
                    )
                    
                    async def rag_token_gen():
                        async for chunk in stream_res:
                            content = chunk.choices[0].delta.content
                            if content:
                                yield content

                    async for safe_token in stream_restore(rag_token_gen(), pii_mapping):
                        final_answer += safe_token
                        yield f"data: {json.dumps({'content': safe_token})}\n\n"
                        
                except Exception as e:
                    err_msg = "LLM service is temporarily unavailable. Please try again later."
                    final_answer = err_msg
                    yield f"data: {json.dumps({'content': err_msg})}\n\n"

        # ── Persist turn & Cache ──────────────────────────────────────────────
        store = get_session_store()
        store.append_turn(req.session_id, clean_message, final_answer, user_id)
        
        if not cache_hit and intent == "SHARIA_FINANCE" and final_answer:
            # We must cache the *unrestored* PII answer for consistency, so we map it back.
            # But the user has seen the restored one. Actually semantic cache caches the restored one?
            # Previously, the restored answer was returned, but we cached the non-restored answer.
            # Let's re-mask it to cache securely.
            masked_answer = final_answer
            for placeholder, original in pii_mapping.items():
                masked_answer = masked_answer.replace(original, placeholder)
                
            set_cached_response(
                query=clean_message, 
                response=masked_answer, 
                user_id=user_id, 
                account_type=account_type, 
                citations=citations
            )

        # Emit final metadata event
        meta = {
            "done": True,
            "citations": citations,
            "run_id": run_id,
            "pii_detected": list(set(pii_types)),
            "cache_hit": cache_hit
        }
        yield f"data: {json.dumps(meta)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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
    chats = store.get_user_chats(user_id)
    if not any(c.get("session_id") == session_id for c in chats):
        raise HTTPException(status_code=404, detail="Chat not found or access denied")
        
    history = store.get_history(session_id)
    return {"history": history}
