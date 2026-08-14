"""
app/rag_stub.py
---------------
THIS IS THE ONLY FILE YOUR TEAMMATE NEEDS TO TOUCH.

It defines the interface contract between your API and the RAG pipeline.
Right now it returns a hardcoded mock so the frontend team can build
against a real API immediately.

When your Data & AI teammate is ready, they replace the body of `rag_query`
with their actual implementation. The function signature MUST stay the same.

CONTRACT (do not change this):
  Input:
    query   : str        — the user's latest message
    history : list[dict] — previous turns: [{"role": "user"|"assistant", "content": "..."}]

  Output: dict with keys:
    "answer"  : str        — the LLM-generated response
    "sources" : list[str]  — list of source document identifiers (filenames, URLs, chunk IDs, etc.)
"""

# ── Stub mode flag ──────────────────────────────────────────────────────────
# Set USE_STUB = False once your teammate's real pipeline is wired in.
USE_STUB = True


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
            "The real RAG pipeline will return a Groq-generated answer here, "
            "grounded in retrieved finance documents."
        ),
        "sources": [
            "annual_report_2023.pdf — page 12",
            "earnings_call_Q3.txt — paragraph 4",
            "10K_filing_2024.pdf — section 3.2",
        ],
    }


# ===========================================================================
# REAL IMPLEMENTATION PLACEHOLDER (your teammate fills this in)
# ===========================================================================
def _real_rag_query(query: str, history: list[dict]) -> dict:
    """
    YOUR TEAMMATE REPLACES THIS BODY with the actual RAG pipeline call.

    Example of what this might look like after handoff:

        from pipeline.retriever import retrieve_chunks
        from pipeline.generator import generate_answer

        chunks = retrieve_chunks(query, top_k=5)
        answer = generate_answer(query, chunks, history)
        sources = [chunk["source"] for chunk in chunks]
        return {"answer": answer, "sources": sources}

    As long as this returns {"answer": str, "sources": list[str]}, the API
    will work without any other changes.
    """
    raise NotImplementedError(
        "Real RAG pipeline not wired in yet. Set USE_STUB = True to use the mock."
    )


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
