"""
app/rag_stub.py
---------------
Interface contract between the FastAPI routes and the RAG pipeline.
Supports both stub mode for frontend development and real RAG execution
via rag_chain.py.

CONTRACT:
  Input:
    query   : str        — the user's latest message
    history : list[dict] — previous turns: [{"role": "user"|"assistant", "content": "..."}]

  Output: dict with keys:
    "answer"  : str        — the LLM-generated response
    "sources" : list[str]  — list of source document identifiers (filenames, tickers, dates)
"""

import os
import logging

logger = logging.getLogger(__name__)

# ── Stub mode flag ──────────────────────────────────────────────────────────
# Set USE_STUB = False to use the real RAG pipeline (can also be toggled via env var)
USE_STUB = os.environ.get("USE_STUB", "false").lower() in ("true", "1", "yes")


# ===========================================================================
# STUB IMPLEMENTATION (active when USE_STUB = True)
# ===========================================================================
def _stub_rag_query(query: str, history: list[dict]) -> dict:
    """
    Returns a fake but plausible response so frontend dev can start immediately.
    Echoes back the query and shows what a real sources list looks like.
    """
    return {
        "answer": (
            f"[STUB RESPONSE] You asked: '{query}'. "
            "The real RAG pipeline will return an LLM-generated answer here, "
            "grounded in retrieved finance documents."
        ),
        "sources": [
            "LQ_AAPL_Q3_2024.csv | Ticker: AAPL | Date: 2024-09-30",
            "Hist_BS_Fin_Stmt.csv | Ticker: MSFT | Date: 2024-06-30",
            "nse_indexes.csv | Ticker: NIFTY50 | Date: 2024-08-01",
        ],
    }


# ===========================================================================
# REAL IMPLEMENTATION (connected to rag_chain.py)
# ===========================================================================
def _real_rag_query(query: str, history: list[dict]) -> dict:
    """
    Executes the real RAG pipeline via rag_chain.get_chain().
    Grounded with inline citations and guardrails.
    """
    try:
        from rag_chain import get_chain
        chain = get_chain()
        response = chain.ask(query=query)

        sources = []
        if response.citations:
            for c in response.citations:
                source = {
                    "doc_title": c.get("source_file", "unknown"),
                    "source_file": c.get("source_file", "unknown"),
                    "chunk_text": c.get("text", ""),
                    "score": c.get("score", 0.0),
                    "ticker": c.get("ticker", ""),
                    "date": c.get("date", ""),
                }
                if source not in sources:
                    sources.append(source)

        return {
            "answer": response.answer,
            "sources": sources,
            "is_grounded": response.is_grounded,
        }
    except RuntimeError as e:
        if "EmbeddingStore is empty" in str(e):
            logger.warning("Vector store is not indexed yet. Please run run_pipeline.py first.")
            return {
                "answer": (
                    "The vector database has not been built yet. "
                    "Please run `python ingest_large.py <path_to_data>` to index the financial datasets."
                ),
                "sources": [],
            }
        logger.exception("Error in real RAG query execution: %s", e)
        raise e
    except Exception as e:
        logger.exception("Error in real RAG query execution: %s", e)
        raise e


# ===========================================================================
# PUBLIC INTERFACE — routes.py calls only this function
# ===========================================================================
def rag_query(query: str, history: list[dict]) -> dict:
    """
    Main entrypoint called by the API layer.
    Routes to stub or real implementation based on the USE_STUB flag.
    """
    if USE_STUB:
        return _stub_rag_query(query, history)
    return _real_rag_query(query, history)
