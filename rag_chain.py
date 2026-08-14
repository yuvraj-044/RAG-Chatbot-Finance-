"""
rag_chain.py
═══════════════════════════════════════════════════════════════════════════════
Finance RAG Chatbot — Member 1: Data & AI Lead
Purpose : Compose retrieved chunks into a hallucination-proof, citation-aware
          RAG prompt and call an LLM to generate a grounded answer.

Guardrails:
  • If retrieved context is missing/irrelevant the model MUST output the
    standard refusal string — enforced both in the system prompt and by a
    post-generation regex check.
  • Every factual claim must be annotated with [SOURCE: ...] inline.

Supported LLM backends (CHAT_BACKEND env var):
    • "gemini"    — Google Gemini 1.5 Flash / Pro  (default)
    • "openai"    — OpenAI GPT-4o / GPT-4o-mini
    • "mock"      — deterministic stub for unit testing

Usage:
    from rag_chain import RAGChain
    chain = RAGChain()
    answer = chain.ask("What was AAPL's revenue in Q3 2024?")
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import re
import json
import logging
from dataclasses import dataclass
from typing import Literal

from embeddings import similarity_search

logger = logging.getLogger(__name__)

# ─── Runtime config ───────────────────────────────────────────────────────────
ChatBackend = Literal["gemini", "openai", "groq", "mock"]
_CHAT_BACKEND: ChatBackend = os.environ.get(
    "CHAT_BACKEND",
    "groq" if os.environ.get("GROQ_API_KEY") else ("gemini" if os.environ.get("GEMINI_API_KEY") else "mock")
)  # type: ignore

_GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")
_GROQ_MODEL       = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

_GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "")
_GEMINI_CHAT_MODEL= os.environ.get("GEMINI_CHAT_MODEL", "gemini-1.5-flash")

_OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "")
_OPENAI_CHAT_MODEL= os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# Token budget reserved for system prompt + answer (leave room for context)
_MAX_CONTEXT_TOKENS = 3000


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SYSTEM PROMPT  — the "soul" of the RAG guardrails
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are FinSight, a senior financial analyst assistant backed by a verified \
dataset of corporate filings, market indices, and transaction records.

━━━━━━━━━━━━  STRICT OPERATING RULES  ━━━━━━━━━━━━

RULE 1 — GROUNDING ONLY
  • Every factual statement you make MUST be directly traceable to the \
CONTEXT PASSAGES provided below.
  • Do NOT use any knowledge from your pre-training. Do NOT extrapolate, \
estimate, or infer figures that are not explicitly stated in the context.

RULE 2 — MANDATORY CITATION FORMAT
  Whenever you state a fact, immediately follow it with an inline citation:
      [SOURCE: <source_file> | Ticker: <ticker> | Date: <date>]
  Example:
      "Net income was $2.1B in Q3 2024. \
[SOURCE: LQ_AAPL_Q3_2024.csv | Ticker: AAPL | Date: 2024-09-30]"

RULE 3 — REFUSAL ON INSUFFICIENT CONTEXT
  If the CONTEXT PASSAGES do not contain information sufficient to answer the \
question, you MUST respond with this exact sentence and nothing else:
      "I cannot find verified records for this inquiry in the dataset."
  Do NOT attempt a partial answer. Do NOT apologise or elaborate further.

RULE 4 — STRUCTURED RESPONSE FORMAT
  Always format your answer as follows:
  ┌─────────────────────────────────────────────┐
  │ ANSWER                                      │
  │  <your grounded answer with inline citations>│
  │                                             │
  │ SOURCES CITED                               │
  │  • <source_file> | <ticker> | <date>        │
  │  (one bullet per unique source used)        │
  └─────────────────────────────────────────────┘

RULE 5 — TONE
  Be concise, precise, and professional. Avoid filler phrases.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# Refusal string — used for guardrail enforcement
_REFUSAL_STRING = "I cannot find verified records for this inquiry in the dataset."

# Minimum cosine similarity to consider a chunk "relevant"
_RELEVANCE_THRESHOLD: float = float(os.environ.get("RELEVANCE_THRESHOLD", "0.35"))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CONTEXT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievedContext:
    """Holds the assembled context block and its source citations."""
    context_block : str          # formatted string injected into the prompt
    citations     : list[dict]   # list of raw chunk metadata dicts
    is_relevant   : bool         # True if at least one chunk meets threshold


def build_context_block(
    retrieved_chunks: list[dict],
    relevance_threshold: float = _RELEVANCE_THRESHOLD,
) -> RetrievedContext:
    """
    Format retrieved chunks into an LLM-injectable context block.

    Each chunk appears as:
        ─── PASSAGE 1 ───────────────────────────────
        Source  : LQ_AAPL_Q3.csv
        Ticker  : AAPL
        Date    : 2024-09-30
        Category: quarterly_financials
        Score   : 0.8421
        ─────────────────────────────────────────────
        <chunk text>

    Parameters
        retrieved_chunks : output of embeddings.similarity_search()
        relevance_threshold : minimum score to include a chunk

    Returns
        RetrievedContext with formatted block and citation metadata
    """
    if not retrieved_chunks:
        return RetrievedContext(
            context_block="No relevant context found.",
            citations=[],
            is_relevant=False,
        )

    relevant  = [c for c in retrieved_chunks if c["score"] >= relevance_threshold]
    if not relevant:
        return RetrievedContext(
            context_block="No passages met the relevance threshold.",
            citations=[],
            is_relevant=False,
        )

    lines = []
    for i, chunk in enumerate(relevant, start=1):
        sep = "─" * 50
        lines.append(f"\n{sep}\nPASSAGE {i}")
        lines.append(f"Source  : {chunk.get('source_file', 'N/A')}")
        lines.append(f"Ticker  : {chunk.get('ticker', 'N/A')}")
        lines.append(f"Date    : {chunk.get('date', 'N/A')}")
        lines.append(f"Category: {chunk.get('data_category', 'N/A')}")
        lines.append(f"Score   : {chunk.get('score', 0.0):.4f}")
        lines.append(sep)
        lines.append(chunk.get("text", ""))

    return RetrievedContext(
        context_block="\n".join(lines),
        citations=relevant,
        is_relevant=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PROMPT ASSEMBLER
# ═══════════════════════════════════════════════════════════════════════════════

def assemble_prompt(query: str, context: RetrievedContext) -> str:
    """
    Build the full user-turn message sent to the LLM.

    Structure:
        CONTEXT PASSAGES
        <formatted retrieved passages>

        USER QUESTION
        <query>
    """
    return (
        "CONTEXT PASSAGES\n"
        "═" * 60 + "\n"
        + context.context_block
        + "\n\n" + "═" * 60 + "\n"
        "USER QUESTION\n"
        f"{query}\n\n"
        "Answer strictly following the rules in the system prompt."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LLM BACKENDS
# ═══════════════════════════════════════════════════════════════════════════════

class GeminiChatBackend:
    """Thin wrapper around Google Gemini generative models."""

    def __init__(self, model: str = _GEMINI_CHAT_MODEL, api_key: str = _GEMINI_API_KEY):
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set.")
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model_name=model,
            system_instruction=SYSTEM_PROMPT,
        )
        logger.info("Gemini chat backend ready: %s", model)

    def complete(self, user_message: str) -> str:
        response = self._model.generate_content(user_message)
        return response.text.strip()


class OpenAIChatBackend:
    """Thin wrapper around OpenAI chat completions."""

    def __init__(self, model: str = _OPENAI_CHAT_MODEL, api_key: str = _OPENAI_API_KEY):
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        from openai import OpenAI  # type: ignore
        self._client = OpenAI(api_key=api_key)
        self._model  = model
        logger.info("OpenAI chat backend ready: %s", model)

    def complete(self, user_message: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0,      # zero temperature → deterministic, factual
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()


class GroqChatBackend:
    """Thin wrapper around Groq API for ultra-fast inference."""

    def __init__(self, model: str = _GROQ_MODEL, api_key: str = _GROQ_API_KEY):
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is not set.")
        from groq import Groq  # type: ignore
        self._client = Groq(api_key=api_key)
        self._model  = model
        logger.info("Groq chat backend ready: %s", model)

    def complete(self, user_message: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0,      # zero temperature → deterministic, factual
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()


class MockChatBackend:
    """
    Deterministic stub for unit tests — always returns the refusal string
    unless the context contains at least one realistic-looking passage.
    """

    def complete(self, user_message: str) -> str:
        if "PASSAGE" in user_message:
            return (
                "ANSWER\n"
                "  Net income for the queried period was $X.XX B "
                "[SOURCE: mock_file.csv | Ticker: MOCK | Date: 2024-01-01]\n\n"
                "SOURCES CITED\n"
                "  • mock_file.csv | MOCK | 2024-01-01"
            )
        return _REFUSAL_STRING


def get_chat_backend(backend: ChatBackend = _CHAT_BACKEND):
    """Factory: return the requested LLM backend instance."""
    if backend == "groq":
        return GroqChatBackend()
    if backend == "openai":
        return OpenAIChatBackend()
    if backend == "mock":
        return MockChatBackend()
    if backend == "gemini":
        return GeminiChatBackend()
    
    # Auto fallback
    if os.environ.get("GROQ_API_KEY"):
        return GroqChatBackend()
    if os.environ.get("GEMINI_API_KEY"):
        return GeminiChatBackend()
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIChatBackend()
    return MockChatBackend()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GUARDRAIL CHECKER
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_guardrails(
    answer: str,
    context: RetrievedContext,
) -> str:
    """
    Post-generation safety layer:
      1. If context was not relevant, override with refusal string.
      2. If the model forgot to cite anything despite being given context,
         append a warning (soft enforcement).
    """
    # Hard guard: no relevant context → force refusal
    if not context.is_relevant:
        logger.info("Guardrail triggered: no relevant context — overriding with refusal.")
        return _REFUSAL_STRING

    # Soft guard: if answer looks like a hallucination (no [SOURCE:] tags)
    if "[SOURCE:" not in answer and answer != _REFUSAL_STRING:
        logger.warning(
            "Model response lacks [SOURCE:] citations — appending warning."
        )
        answer += (
            "\n\n⚠️  Note: This response may lack proper source citations. "
            "Please verify against the original dataset."
        )

    return answer


# ═══════════════════════════════════════════════════════════════════════════════
# 6. RESPONSE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RAGResponse:
    """
    Structured response object returned by RAGChain.ask().
    Member 2 serialises this directly for API responses.
    """
    query          : str
    answer         : str                 # grounded LLM response
    citations      : list[dict]          # list of chunk metadata dicts used
    retrieved_chunks: list[dict]         # raw Top-K chunks (for debug / UI)
    is_grounded    : bool                # False if refusal was triggered
    relevance_scores: list[float]        # cosine scores of retrieved chunks

    def to_dict(self) -> dict:
        return {
            "query"           : self.query,
            "answer"          : self.answer,
            "citations"       : self.citations,
            "retrieved_chunks": self.retrieved_chunks,
            "is_grounded"     : self.is_grounded,
            "relevance_scores": self.relevance_scores,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MAIN RAG CHAIN
# ═══════════════════════════════════════════════════════════════════════════════

class RAGChain:
    """
    End-to-end RAG pipeline:
        query → retrieve → build context → prompt → LLM → guardrail → response

    ─── Usage by Member 2 (FastAPI) ──────────────────────────────────────────
        from rag_chain import RAGChain
        chain = RAGChain()               # initialised once at app startup

        @app.post("/query")
        async def query_endpoint(request: QueryRequest):
            response = chain.ask(request.query, k=request.top_k)
            return response.to_dict()
    ──────────────────────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        chat_backend: ChatBackend = _CHAT_BACKEND,
        top_k: int = 3,
        relevance_threshold: float = _RELEVANCE_THRESHOLD,
        persist_path: str = "./vector_store/embeddings",
    ):
        self.llm                 = get_chat_backend(chat_backend)
        self.top_k               = top_k
        self.relevance_threshold = relevance_threshold
        self.persist_path        = persist_path

    def ask(
        self,
        query: str,
        k: int | None = None,
        filter_category: str | None = None,
        filter_ticker: str | None = None,
    ) -> RAGResponse:
        """
        Run the full RAG pipeline for a user query.

        Parameters
            query           : natural language question
            k               : override Top-K (default: self.top_k)
            filter_category : restrict retrieval to a data category
            filter_ticker   : restrict retrieval to a specific ticker

        Returns
            RAGResponse with answer, citations, and raw chunks
        """
        top_k = k or self.top_k
        logger.info("RAG query: '%s'  (k=%d)", query, top_k)

        # ── Step 1: Retrieve ──────────────────────────────────────────────────
        retrieved = similarity_search(
            query=query,
            k=top_k,
            filter_category=filter_category,
            filter_ticker=filter_ticker,
            persist_path=self.persist_path,
        )
        relevance_scores = [r.get("score", 0.0) for r in retrieved]
        logger.debug("Retrieved %d chunks. Scores: %s", len(retrieved), relevance_scores)

        # ── Step 2: Build context ─────────────────────────────────────────────
        context = build_context_block(retrieved, self.relevance_threshold)

        # ── Step 3: Assemble prompt ───────────────────────────────────────────
        user_message = assemble_prompt(query, context)

        # ── Step 4: Generate ──────────────────────────────────────────────────
        if not context.is_relevant:
            # Skip LLM call entirely — just return refusal
            raw_answer = _REFUSAL_STRING
        else:
            raw_answer = self.llm.complete(user_message)

        # ── Step 5: Apply guardrails ──────────────────────────────────────────
        final_answer = _apply_guardrails(raw_answer, context)
        is_grounded  = final_answer != _REFUSAL_STRING

        return RAGResponse(
            query            = query,
            answer           = final_answer,
            citations        = context.citations,
            retrieved_chunks = retrieved,
            is_grounded      = is_grounded,
            relevance_scores = relevance_scores,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CONVENIENCE FUNCTION — for Member 2's API handlers
# ═══════════════════════════════════════════════════════════════════════════════

# Global singleton
_chain: RAGChain | None = None


def get_chain(**kwargs) -> RAGChain:
    """
    Lazily initialise and return the global RAGChain singleton.
    Pass kwargs to override defaults (chat_backend, top_k, etc.).

    ─── Member 2 Integration ─────────────────────────────────────────────────
    from rag_chain import get_chain
    chain = get_chain()

    # In your FastAPI startup event:
    @app.on_event("startup")
    async def startup():
        get_chain()    # warm up

    # In your endpoint:
    response = get_chain().ask(query)
    ──────────────────────────────────────────────────────────────────────────
    """
    global _chain
    if _chain is None:
        _chain = RAGChain(**kwargs)
    return _chain


# ═══════════════════════════════════════════════════════════════════════════════
# 9. ENTRY POINT (interactive REPL for testing)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    backend = sys.argv[1] if len(sys.argv) > 1 else "mock"
    chain   = RAGChain(chat_backend=backend)  # type: ignore

    print("\n🤖  FinSight RAG Chatbot — type 'quit' to exit\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        response = chain.ask(query)
        print(f"\nFinSight:\n{response.answer}\n")
        if response.citations:
            print("📎  Citations:")
            for c in response.citations:
                print(
                    f"   • {c.get('source_file')} | "
                    f"{c.get('ticker')} | {c.get('date')}"
                )
        print()
