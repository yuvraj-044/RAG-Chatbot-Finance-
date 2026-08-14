"""
app/main.py
-----------
FastAPI application entrypoint.

This file:
  1. Creates the FastAPI app instance
  2. Validates config at startup (catches missing secrets immediately)
  3. Attaches CORS middleware so the frontend can call us from a browser
  4. Mounts the router from routes.py

To run locally:
  uvicorn app.main:app --reload --port 8000

Then test with:
  curl http://localhost:8000/health
  curl -X POST http://localhost:8000/chat \
       -H "Content-Type: application/json" \
       -d '{"session_id": "test-1", "message": "What is EBITDA?"}'
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from app.api.routes import router

# ── Logging setup ────────────────────────────────────────────────────────────
# Simple stdout logging — good enough for a hackathon demo.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan handler ─────────────────────────────────────────────────────────
# FastAPI's modern way to run startup/shutdown logic (replaces @app.on_event).
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    logger.info("Starting Finance RAG API...")
    try:
        config.validate_config()   # Crash early if secrets are missing
        logger.info("Config validated ✓")
    except ValueError as e:
        # Log the error but don't crash the server — makes debugging on HF Spaces easier
        logger.error("Config validation failed: %s", e)
        logger.warning("Server will start but /chat requests may fail until config is fixed.")

    logger.info(
        "Stub mode: %s | Model: %s | Chroma dir: %s",
        config.validate_config.__module__,  # just to reference config
        config.GROQ_MODEL,
        config.CHROMA_PERSIST_DIR,
    )
    yield
    # --- SHUTDOWN ---
    logger.info("Finance RAG API shutting down.")


# ── App instance ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Finance RAG Chatbot API",
    description=(
        "Backend API for a finance-domain Retrieval-Augmented Generation chatbot. "
        "Exposes a chat endpoint that wraps a RAG pipeline (ChromaDB + Groq)."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",    # Swagger UI — your frontend teammate can explore here
    redoc_url="/redoc",  # ReDoc alternative
)


# ── CORS middleware ───────────────────────────────────────────────────────────
# Allows the browser-based frontend to call this API even if it's on a different
# origin (e.g., frontend on HF Spaces, API on a different Space or port).
# For production you'd restrict allow_origins to your actual frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ← tighten this to your frontend URL after the demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Mount routes ─────────────────────────────────────────────────────────────
# All routes defined in routes.py are now available at the root.
app.include_router(router)


# ── Dev entrypoint ───────────────────────────────────────────────────────────
# Allows `python app/main.py` for quick local testing (not recommended for prod).
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
