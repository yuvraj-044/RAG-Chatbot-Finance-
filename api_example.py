"""
api_example.py  ──  Reference implementation for Member 2 (Backend Lead)
═══════════════════════════════════════════════════════════════════════════════
This file is NOT part of Member 1's deliverable.
It is a ready-to-run FastAPI skeleton showing exactly how Member 2 can wire
up the four Member-1 modules into REST endpoints.

Run with:
    uvicorn api_example:app --reload --port 8000

Endpoints:
    POST /index          — trigger a full (re)index of data_dir
    POST /query          — submit a RAG query
    GET  /health         — liveness check
    GET  /store/stats    — inspect the vector store
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ─── Member 1 imports ─────────────────────────────────────────────────────────
from ingest_and_clean import FinanceDataPipeline
from chunking          import ChunkingPipeline, chunk_stats
from embeddings        import EmbeddingStore, get_store
from rag_chain         import RAGChain, get_chain, RAGResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Config (override via env vars) ───────────────────────────────────────────
DATA_DIR      = os.environ.get("DATA_DIR",      "./data")
PERSIST_PATH  = os.environ.get("PERSIST_PATH",  "./vector_store/embeddings")
CHAT_BACKEND  = os.environ.get("CHAT_BACKEND",  "gemini")
EMB_BACKEND   = os.environ.get("EMBEDDING_BACKEND", "sentence_transformers")


# ═══════════════════════════════════════════════════════════════════════════════
# APP LIFECYCLE — warm up on startup
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the vector store from disk (if it exists) on startup."""
    logger.info("🚀  FinSight API starting up …")
    store = get_store(persist_path=PERSIST_PATH, backend=EMB_BACKEND)   # type: ignore
    get_chain(chat_backend=CHAT_BACKEND, persist_path=PERSIST_PATH)     # type: ignore
    logger.info("✅  Store loaded. Vectors indexed: %d", store.size)
    yield
    logger.info("👋  FinSight API shutting down.")


app = FastAPI(
    title="FinSight RAG API",
    description="High-Performance Finance RAG Chatbot — Hackathon Project",
    version="1.0.0",
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class IndexRequest(BaseModel):
    data_dir: str = Field(DATA_DIR, description="Path to the dataset directory")

class IndexResponse(BaseModel):
    status       : str
    total_chunks : int
    vector_dim   : int

class QueryRequest(BaseModel):
    query           : str  = Field(...,  description="Natural language question")
    top_k           : int  = Field(3,    ge=1, le=10)
    filter_category : str | None = Field(None, description="e.g. quarterly_financials")
    filter_ticker   : str | None = Field(None, description="e.g. AAPL")

class QueryResponse(BaseModel):
    query            : str
    answer           : str
    is_grounded      : bool
    citations        : list[dict]
    relevance_scores : list[float]
    retrieved_chunks : list[dict]


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """Liveness probe."""
    store = get_store(persist_path=PERSIST_PATH, backend=EMB_BACKEND)  # type: ignore
    return {
        "status"         : "ok",
        "vectors_indexed": store.size,
    }


@app.post("/index", response_model=IndexResponse)
async def index_data(req: IndexRequest):
    """
    (Re)index all datasets in the specified data directory.
    This is a long-running operation — consider running it as a background task.
    """
    try:
        docs   = FinanceDataPipeline(data_dir=req.data_dir).run()
        chunks = ChunkingPipeline().run(docs)

        store = EmbeddingStore(backend=EMB_BACKEND, persist_path=PERSIST_PATH)  # type: ignore
        store.build(chunks)

        # Update the global singleton
        from embeddings import get_store as _gs
        import embeddings as _emb_module
        _emb_module._store = store

        return IndexResponse(
            status="indexed",
            total_chunks=store.size,
            vector_dim=store.dim,
        )
    except Exception as exc:
        logger.exception("Indexing failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Submit a natural language query to the RAG chain.
    Returns a grounded answer with inline citations.
    """
    try:
        response: RAGResponse = get_chain().ask(
            query=req.query,
            k=req.top_k,
            filter_category=req.filter_category,
            filter_ticker=req.filter_ticker,
        )
        return QueryResponse(**response.to_dict())
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/store/stats")
async def store_stats():
    """Inspect the current state of the vector store."""
    store = get_store(persist_path=PERSIST_PATH, backend=EMB_BACKEND)  # type: ignore
    return {
        "vectors_indexed": store.size,
        "embedding_dim"  : store.dim,
        "persist_path"   : PERSIST_PATH,
    }
