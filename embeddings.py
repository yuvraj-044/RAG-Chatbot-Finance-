"""
embeddings.py
═══════════════════════════════════════════════════════════════════════════════
Finance RAG Chatbot — Member 1: Data & AI Lead
Purpose : Generate vector embeddings for all chunks and expose a similarity
          search interface that returns Top-K chunks with full metadata.

Supported backends (switchable via EMBEDDING_BACKEND env var):
    • "sentence_transformers"  — local, no API key needed
          Models: BAAI/bge-small-en-v1.5 (default), all-MiniLM-L6-v2
    • "gemini"                 — Google Gemini text-embedding-004 via API

Storage:
    Embeddings are persisted as a .npy matrix + a JSON sidecar so the
    server does not re-embed on every restart.

Usage:
    from embeddings import EmbeddingStore
    store = EmbeddingStore()
    store.build(chunks)                         # one-time indexing
    results = store.search("net profit Q3", k=3)
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Literal

import numpy as np

from chunking import Chunk

logger = logging.getLogger(__name__)

# ─── Backend selector ─────────────────────────────────────────────────────────
EmbeddingBackend = Literal["sentence_transformers", "gemini"]

_DEFAULT_BACKEND: EmbeddingBackend = os.environ.get(          # type: ignore[assignment]
    "EMBEDDING_BACKEND", "sentence_transformers"
)
_ST_MODEL   = os.environ.get("ST_MODEL",     "BAAI/bge-small-en-v1.5")
_GEMINI_MODEL = os.environ.get("GEMINI_EMB_MODEL", "models/text-embedding-004")
_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BACKEND ADAPTERS
# ═══════════════════════════════════════════════════════════════════════════════

class SentenceTransformerBackend:
    """
    Local embedding via sentence-transformers.
    Install: pip install sentence-transformers
    """

    def __init__(self, model_name: str = _ST_MODEL):
        logger.info("Loading sentence-transformers model: %s", model_name)
        from sentence_transformers import SentenceTransformer  # type: ignore
        self.model = SentenceTransformer(model_name)
        self.dim   = (
            self.model.get_embedding_dimension()
            if hasattr(self.model, "get_embedding_dimension")
            else self.model.get_sentence_embedding_dimension()
        )
        logger.info("Model loaded. Embedding dim: %d", self.dim)

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Embed a list of texts in batches.
        Returns shape (N, dim) float32 numpy array.
        """
        all_vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vecs  = self.model.encode(
                batch,
                normalize_embeddings=True,   # cosine similarity via dot product
                show_progress_bar=False,
            )
            all_vectors.append(vecs)
            logger.debug("Embedded batch %d–%d", i, i + len(batch))
        return np.vstack(all_vectors).astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single query string. Returns shape (dim,)."""
        return self.model.encode(text, normalize_embeddings=True).astype(np.float32)


class GeminiEmbeddingBackend:
    """
    Cloud embedding via Google Gemini text-embedding-004.
    Install: pip install google-generativeai
    Requires: GEMINI_API_KEY environment variable.
    """

    # API rate limits (free tier: 1500 req/min)
    _REQUESTS_PER_BATCH = 100
    _BATCH_DELAY_SECS   = 2

    def __init__(self, model: str = _GEMINI_MODEL, api_key: str = _GEMINI_API_KEY):
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. "
                "Export it or switch to EMBEDDING_BACKEND=sentence_transformers."
            )
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=api_key)
        self.genai = genai
        self.model = model
        # Probe dimension with a test call
        test = self._call_api(["test"])
        self.dim = len(test[0])
        logger.info("Gemini backend ready. Model: %s  Dim: %d", model, self.dim)

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Call the Gemini embedding API for a batch of texts."""
        result = self.genai.embed_content(
            model=self.model,
            content=texts,
            task_type="RETRIEVAL_DOCUMENT",
        )
        return [emb["values"] for emb in result["embedding"]]

    def embed_batch(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        """Embed in batches with rate-limit courtesy delay."""
        all_vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vecs  = self._call_api(batch)
            all_vectors.extend(vecs)
            logger.debug("Gemini embedded batch %d–%d", i, i + len(batch))
            if i + batch_size < len(texts):
                time.sleep(self._BATCH_DELAY_SECS)
        return np.array(all_vectors, dtype=np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single query string."""
        result = self.genai.embed_content(
            model=self.model,
            content=text,
            task_type="RETRIEVAL_QUERY",  # separate task type for queries
        )
        return np.array(result["embedding"]["values"], dtype=np.float32)


def get_backend(backend: EmbeddingBackend = _DEFAULT_BACKEND):
    """Factory: return the appropriate embedding backend instance."""
    if backend == "gemini":
        return GeminiEmbeddingBackend()
    return SentenceTransformerBackend()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. VECTOR STORE (in-memory with optional disk persistence)
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingStore:
    """
    Stores chunk embeddings in a numpy matrix and chunk metadata in a list.
    Supports cosine similarity search (dot product on L2-normalised vectors).

    Persistence
        save(path)  → writes  <path>.npy  and  <path>_meta.json
        load(path)  ← reads   <path>.npy  and  <path>_meta.json
    """

    def __init__(
        self,
        backend: EmbeddingBackend = _DEFAULT_BACKEND,
        persist_path: str | None = "./vector_store/embeddings",
    ):
        self.backend      = get_backend(backend)
        self.persist_path = persist_path
        self._matrix: np.ndarray | None = None     # shape (N, dim)
        self._meta:   list[dict]         = []       # parallel to matrix rows

    # ── Build index ───────────────────────────────────────────────────────────

    def build(self, chunks: list[Chunk]) -> None:
        """
        Embed all chunks and build the in-memory vector index.
        Automatically persists to disk if persist_path is set.

        Parameters
            chunks : output of chunking.ChunkingPipeline.run()
        """
        if not chunks:
            logger.warning("No chunks provided — nothing to embed.")
            return

        logger.info("Building vector index for %d chunks ...", len(chunks))
        texts = [c.text for c in chunks]

        self._matrix = self.backend.embed_batch(texts)
        self._meta   = [c.to_dict() for c in chunks]   # serialisable metadata

        logger.info(
            "Index built. Matrix shape: %s  (dtype: %s)",
            self._matrix.shape, self._matrix.dtype
        )

        if self.persist_path:
            self.save(self.persist_path)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the embedding matrix and metadata to disk."""
        dir_ = Path(path).parent
        dir_.mkdir(parents=True, exist_ok=True)

        np.save(f"{path}.npy", self._matrix)

        with open(f"{path}_meta.json", "w", encoding="utf-8") as f:
            json.dump(self._meta, f, ensure_ascii=False, indent=2)

        logger.info("Vector store saved → %s.npy + %s_meta.json", path, path)

    def load(self, path: str) -> bool:
        """
        Load a previously saved index from disk.
        Returns True on success, False if files are not found.
        """
        npy_path  = Path(f"{path}.npy")
        meta_path = Path(f"{path}_meta.json")

        if not npy_path.exists() or not meta_path.exists():
            logger.info("No persisted store found at %s — will build from scratch.", path)
            return False

        self._matrix = np.load(str(npy_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            self._meta = json.load(f)

        logger.info(
            "Loaded vector store from disk. Vectors: %d  Dim: %d",
            self._matrix.shape[0], self._matrix.shape[1]
        )
        return True

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 3,
        filter_category: str | None = None,
        filter_ticker: str | None = None,
    ) -> list[dict]:
        """
        Embed a user query and return the Top-K most relevant chunks.

        Parameters
            query           : natural language search string
            k               : number of results to return (default 3)
            filter_category : optionally restrict to a data_category
            filter_ticker   : optionally restrict to a ticker symbol

        Returns
            list[dict] sorted by descending similarity, each element:
            {
                "rank"          : int,
                "score"         : float (cosine similarity, 0–1),
                "chunk_id"      : str,
                "text"          : str,
                "source_file"   : str,
                "ticker"        : str,
                "date"          : str,
                "data_category" : str,
                ...extra_meta
            }
        """
        if self._matrix is None or len(self._meta) == 0:
            raise RuntimeError(
                "EmbeddingStore is empty. Call build() or load() first."
            )

        # Build candidate mask for optional filters
        mask = np.ones(len(self._meta), dtype=bool)
        if filter_category:
            mask &= np.array(
                [m["data_category"] == filter_category for m in self._meta]
            )
        if filter_ticker:
            mask &= np.array(
                [m["ticker"].upper() == filter_ticker.upper() for m in self._meta]
            )

        candidate_indices = np.where(mask)[0]
        if len(candidate_indices) == 0:
            logger.warning("No chunks matched the filters — returning empty list.")
            return []

        # Embed query
        q_vec = self.backend.embed_single(query)              # shape (dim,)
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-10)     # L2-normalise

        # Compute cosine similarity (dot product on normalised vectors)
        candidate_matrix = self._matrix[candidate_indices]   # shape (C, dim)
        scores = candidate_matrix @ q_vec                     # shape (C,)

        # Select Top-K
        top_k_local = min(k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k_local]

        results = []
        for rank, local_idx in enumerate(top_indices, start=1):
            global_idx = int(candidate_indices[local_idx])
            meta       = dict(self._meta[global_idx])
            results.append({
                "rank"   : rank,
                "score"  : float(round(scores[local_idx], 6)),
                **meta,
            })

        return results

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of indexed vectors."""
        return 0 if self._matrix is None else self._matrix.shape[0]

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        if self._matrix is None:
            return self.backend.dim
        return self._matrix.shape[1]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CONVENIENCE FUNCTION — for Member 2 (Backend) to call from API handlers
# ═══════════════════════════════════════════════════════════════════════════════

# Global singleton — lazily loaded on first import
_store: EmbeddingStore | None = None


def get_store(
    persist_path: str = "./vector_store/embeddings",
    backend: EmbeddingBackend = _DEFAULT_BACKEND,
) -> EmbeddingStore:
    """
    Return (or lazily initialise) the global EmbeddingStore singleton.
    Attempts to load from disk first; falls back to an empty (unbuilt) store.

    ─── Called by Member 2's API handlers ────────────────────────────────────
    from embeddings import get_store
    store = get_store()
    results = store.search(query, k=3)
    """
    global _store
    if _store is None:
        _store = EmbeddingStore(backend=backend, persist_path=persist_path)
        _store.load(persist_path)     # no-op if files don't exist
    return _store


def similarity_search(
    query: str,
    k: int = 3,
    filter_category: str | None = None,
    filter_ticker: str | None = None,
    persist_path: str = "./vector_store/embeddings",
) -> list[dict]:
    """
    Stateless convenience wrapper for Member 2's FastAPI endpoint.

    Parameters
        query    : user's natural language question
        k        : number of chunks to return (default 3)
        filter_* : optional pre-filters

    Returns
        list[dict] — Top-K chunks with score + all metadata
    """
    store = get_store(persist_path=persist_path)
    return store.search(query, k=k,
                        filter_category=filter_category,
                        filter_ticker=filter_ticker)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ENTRY POINT — build the index from scratch
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from ingest_and_clean import FinanceDataPipeline
    from chunking import ChunkingPipeline

    data_dir     = sys.argv[1] if len(sys.argv) > 1 else "./data"
    persist_path = sys.argv[2] if len(sys.argv) > 2 else "./vector_store/embeddings"

    # Full pipeline
    docs   = FinanceDataPipeline(data_dir=data_dir).run()
    chunks = ChunkingPipeline().run(docs)
    store  = EmbeddingStore(persist_path=persist_path)
    store.build(chunks)

    # Quick smoke test
    test_query = "What was the net profit in Q3?"
    results = store.search(test_query, k=3)
    print(f"\n🔍 Query: '{test_query}'")
    for r in results:
        print(f"\n  Rank {r['rank']}  Score={r['score']:.4f}")
        print(f"  Source: {r['source_file']} | {r['ticker']} | {r['date']}")
        print(f"  Text  : {r['text'][:200]}...")
