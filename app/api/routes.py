"""
app/api/routes.py
-----------------
All HTTP endpoints live here. This is the file your frontend teammate
builds against. The three routes are:

  GET  /health          — liveness check, always returns 200
  POST /chat            — main chat endpoint
  POST /chat/reset      — clears a session's conversation history

API CONTRACT (share this with your frontend teammate):

  POST /chat
    Request body:  { "session_id": str, "message": str }
    Response body: { "session_id": str, "reply": str, "sources": list[str] }
    Error 422:     invalid request body (missing fields, wrong types)
    Error 503:     LLM or retrieval pipeline is unavailable
    Error 500:     unexpected internal error

  POST /chat/reset
    Request body:  { "session_id": str }
    Response body: { "session_id": str, "status": "reset" }

  GET /health
    Response body: { "status": "ok", "stub_mode": bool }
"""

import logging
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Import our own modules
import config
from app.session_store import get_history, add_turn, reset_session
from app.rag_stub import rag_query, USE_STUB

# ── Logger ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Router ───────────────────────────────────────────────────────────────────
# We use a separate APIRouter (not the app directly) so main.py stays clean
# and we could add versioning (e.g. /v1) later if needed.
router = APIRouter()


# ===========================================================================
# Pydantic models — define and validate the request/response shapes
# ===========================================================================

class ChatRequest(BaseModel):
    """What the frontend sends us."""
    session_id: str = Field(..., min_length=1, description="Unique session identifier")
    message: str = Field(..., min_length=1, description="The user's message")


class ChatResponse(BaseModel):
    """What we send back to the frontend."""
    session_id: str
    reply: str
    sources: list[str]


class ResetRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


class ResetResponse(BaseModel):
    session_id: str
    status: str  # always "reset" on success


# ===========================================================================
# Routes
# ===========================================================================

@router.get("/health", tags=["Meta"])
async def health_check():
    """
    Liveness endpoint. The frontend / HF Spaces health checker hits this.
    Returns whether we're running in stub mode so the frontend can display
    a warning badge if needed.
    """
    return {"status": "ok", "stub_mode": USE_STUB}


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Main chat endpoint.

    Flow:
      1. Load existing conversation history for this session.
      2. Call rag_query(message, history) — either stub or real pipeline.
      3. Append this turn to the session history.
      4. Return reply + sources to the frontend.

    Error handling:
      - TimeoutError   → 503, LLM took too long
      - NotImplementedError → 503, real pipeline not wired in yet
      - Any other Exception → 500, generic internal error
    """
    logger.info(
        "Chat request | session=%s | message_len=%d",
        request.session_id,
        len(request.message),
    )

    # Step 1 — fetch history (empty list for a new session, no error)
    history = get_history(request.session_id)

    # Step 2 — call the RAG pipeline (wrapped in asyncio.to_thread because
    #           rag_query may do blocking I/O — chroma queries, HTTP calls)
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(rag_query, request.message, history),
            timeout=config.LLM_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("RAG pipeline timed out | session=%s", request.session_id)
        raise HTTPException(
            status_code=503,
            detail=(
                f"The AI took longer than {config.LLM_TIMEOUT_SECONDS}s to respond. "
                "Please try again with a shorter question."
            ),
        )
    except NotImplementedError as e:
        # Thrown when USE_STUB=False but real pipeline isn't implemented yet
        logger.error("RAG pipeline not implemented: %s", e)
        raise HTTPException(
            status_code=503,
            detail="The RAG pipeline is not yet available. Running in stub mode.",
        )
    except Exception as e:
        # Catch-all: LLM API error, Chroma connection error, etc.
        logger.exception("Unexpected error in RAG pipeline | session=%s", request.session_id)
        raise HTTPException(
            status_code=500,
            detail=(
                "Something went wrong while generating your answer. "
                "Please try again in a moment."
            ),
        )

    # Step 3 — validate that the RAG result has the expected shape
    reply = result.get("answer", "").strip()
    sources = result.get("sources", [])

    # If the pipeline returned an empty answer, give a graceful fallback
    if not reply:
        reply = (
            "I couldn't find relevant information to answer that question. "
            "Try rephrasing, or ask about a different finance topic."
        )

    # Step 4 — persist this turn to session history
    add_turn(request.session_id, request.message, reply)

    logger.info(
        "Chat response | session=%s | sources=%d", request.session_id, len(sources)
    )

    return ChatResponse(
        session_id=request.session_id,
        reply=reply,
        sources=sources,
    )


@router.post("/chat/reset", response_model=ResetResponse, tags=["Chat"])
async def reset_chat(request: ResetRequest):
    """
    Clears the conversation history for a given session.
    Safe to call even if the session doesn't exist — it's a no-op in that case.
    """
    reset_session(request.session_id)
    logger.info("Session reset | session=%s", request.session_id)
    return ResetResponse(session_id=request.session_id, status="reset")
