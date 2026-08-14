"""
chunking.py
═══════════════════════════════════════════════════════════════════════════════
Finance RAG Chatbot — Member 1: Data & AI Lead
Purpose : Split cleaned document texts into overlapping token-aware chunks
          and attach full metadata to each chunk for downstream citation.

Strategy: Recursive Character Text Splitter
  • Primary split on paragraph boundaries → sentences → words
  • Target  : 600 tokens   (~2400 characters at ~4 chars/token)
  • Overlap : 100 tokens   (~400 characters)

Usage:
    from chunking import ChunkingPipeline
    pipeline = ChunkingPipeline()
    chunks = pipeline.run(docs)   # docs = output of ingest_and_clean.py
═══════════════════════════════════════════════════════════════════════════════
"""

import re
import uuid
import logging
from dataclasses import dataclass, field, asdict
from typing import Generator

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    """
    A single text chunk with its full provenance metadata.

    Fields used by embeddings.py for retrieval and rag_chain.py for citation:
        chunk_id      — unique identifier (UUID4)
        text          — the actual chunk content fed to the embedder
        source_file   — original filename (e.g., 'LQ_AAPL_2024.csv')
        ticker        — company ticker or 'N/A'
        date          — filing / transaction date in YYYY-MM-DD
        data_category — e.g., 'quarterly_financials', 'transaction_record'
        chunk_index   — position within the parent document
        char_start    — character offset in the original document text
    """
    chunk_id      : str
    text          : str
    source_file   : str
    ticker        : str
    date          : str
    data_category : str
    chunk_index   : int
    char_start    : int
    extra_meta    : dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TOKEN ESTIMATOR  (no dependency on a tokenizer library)
# ═══════════════════════════════════════════════════════════════════════════════

# Empirical approximation: 1 token ≈ 4 characters for English financial text.
# Swap this with a real tokenizer (tiktoken / transformers) if precision matters.
_CHARS_PER_TOKEN: float = 4.0


def estimate_tokens(text: str) -> int:
    """Approximate token count without loading a full tokenizer."""
    return max(1, round(len(text) / _CHARS_PER_TOKEN))


def tokens_to_chars(tokens: int) -> int:
    return round(tokens * _CHARS_PER_TOKEN)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. RECURSIVE CHARACTER SPLITTER
# ═══════════════════════════════════════════════════════════════════════════════

class RecursiveCharacterSplitter:
    """
    Mimics LangChain's RecursiveCharacterTextSplitter without the dependency.

    Splitting priority (highest → lowest granularity):
        1. Double newline  (\n\n)  — paragraph boundary
        2. Single newline  (\n)    — line boundary
        3. Period+space    (. )    — sentence boundary
        4. Comma+space     (, )    — clause boundary
        5. Space           ( )     — word boundary
        6. Character-level         — last resort (never splits mid-byte)

    Parameters
        target_tokens : target chunk size in tokens  (default 600)
        overlap_tokens: overlap between consecutive chunks (default 100)
    """

    SEPARATORS = ["\n\n", "\n", ". ", ", ", " ", ""]

    def __init__(self, target_tokens: int = 600, overlap_tokens: int = 100):
        self.target_chars  = tokens_to_chars(target_tokens)
        self.overlap_chars = tokens_to_chars(overlap_tokens)

    # ── Core splitting logic ──────────────────────────────────────────────────

    def _split_by_separator(self, text: str, separator: str) -> list[str]:
        """Split text by separator, keeping the separator at the end of each piece."""
        if separator == "":
            return list(text)               # character-level fallback
        parts = text.split(separator)
        # Re-attach separator to restore context (except last piece)
        return [p + separator for p in parts[:-1]] + [parts[-1]] if parts else []

    def _merge_splits(self, splits: list[str]) -> list[str]:
        """
        Greedily merge small splits into chunks up to target_chars,
        then slide forward by (target_chars - overlap_chars).
        """
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for piece in splits:
            piece_len = len(piece)

            # If adding this piece would overflow the target, flush current chunk
            if current_len + piece_len > self.target_chars and current:
                chunk_text = "".join(current).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                # Slide window: keep overlap worth of characters
                while current and current_len > self.overlap_chars:
                    removed = current.pop(0)
                    current_len -= len(removed)

            current.append(piece)
            current_len += piece_len

        # Flush remainder
        if current:
            chunk_text = "".join(current).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks

    def split_text(self, text: str) -> list[tuple[str, int]]:
        """
        Recursively split text into chunks.

        Returns
            list of (chunk_text, char_start_offset) tuples.
        """
        return list(self._recursive_split(text, 0, self.SEPARATORS))

    def _recursive_split(
        self, text: str, offset: int, separators: list[str]
    ) -> Generator[tuple[str, int], None, None]:
        """DFS recursive splitting — move to finer separator when chunk too large."""
        if len(text) <= self.target_chars:
            # Small enough to emit directly
            yield (text.strip(), offset)
            return

        separator = separators[0] if separators else ""
        remaining_seps = separators[1:]

        splits = self._split_by_separator(text, separator)
        merged = self._merge_splits(splits)

        running_offset = offset
        for chunk in merged:
            if len(chunk) <= self.target_chars:
                yield (chunk, running_offset)
            else:
                # Still too large: recurse with finer separator
                yield from self._recursive_split(chunk, running_offset, remaining_seps)
            # Advance offset (approximate — exact tracking not critical here)
            running_offset += len(chunk) - self.overlap_chars


# ═══════════════════════════════════════════════════════════════════════════════
# 4. METADATA NORMALISER
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_metadata(doc_meta: dict) -> dict:
    """
    Normalise the raw metadata dict produced by ingest_and_clean.py
    into the canonical fields required by a Chunk.
    """
    return {
        "source_file"   : str(doc_meta.get("source_file", "unknown")),
        "ticker"        : str(doc_meta.get("ticker", "N/A")),
        "date"          : str(doc_meta.get("date", "N/A")),
        "data_category" : str(doc_meta.get("data_category", "unknown")),
    }


def _passthrough_extra(doc_meta: dict) -> dict:
    """Preserve any additional metadata fields not in the canonical set."""
    canonical = {"source_file", "ticker", "date", "data_category"}
    return {k: v for k, v in doc_meta.items() if k not in canonical}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN CHUNKING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class ChunkingPipeline:
    """
    Accepts the cleaned document list from FinanceDataPipeline.run() and
    produces a flat list of Chunk objects ready for embedding.

    Example
    ───────
        from ingest_and_clean import FinanceDataPipeline
        from chunking import ChunkingPipeline

        docs   = FinanceDataPipeline(data_dir="./data").run()
        chunks = ChunkingPipeline().run(docs)
    """

    def __init__(self, target_tokens: int = 600, overlap_tokens: int = 100):
        self.splitter = RecursiveCharacterSplitter(
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        )

    def run(self, docs: list[dict]) -> list[Chunk]:
        """
        Process all documents and return a flat list of Chunk objects.

        Parameters
            docs : list[dict] with keys {"text": str, "metadata": dict}
                   (output of ingest_and_clean.FinanceDataPipeline.run())

        Returns
            list[Chunk] — every chunk has a unique chunk_id and full metadata
        """
        all_chunks: list[Chunk] = []
        total_docs = len(docs)

        for doc_idx, doc in enumerate(docs):
            raw_text  = doc.get("text", "").strip()
            doc_meta  = doc.get("metadata", {})

            if not raw_text:
                logger.debug("Skipping empty document at index %d", doc_idx)
                continue

            canonical = _extract_metadata(doc_meta)
            extra     = _passthrough_extra(doc_meta)

            splits = self.splitter.split_text(raw_text)

            for chunk_idx, (chunk_text, char_start) in enumerate(splits):
                if not chunk_text.strip():
                    continue

                chunk = Chunk(
                    chunk_id      = str(uuid.uuid4()),
                    text          = chunk_text,
                    source_file   = canonical["source_file"],
                    ticker        = canonical["ticker"],
                    date          = canonical["date"],
                    data_category = canonical["data_category"],
                    chunk_index   = chunk_idx,
                    char_start    = char_start,
                    extra_meta    = extra,
                )
                all_chunks.append(chunk)

            if (doc_idx + 1) % 500 == 0:
                logger.info(
                    "Chunked %d / %d documents → %d chunks so far",
                    doc_idx + 1, total_docs, len(all_chunks)
                )

        logger.info(
            "Chunking complete. Documents: %d → Chunks: %d  "
            "(avg %.1f chunks/doc)",
            total_docs, len(all_chunks),
            len(all_chunks) / max(total_docs, 1)
        )
        return all_chunks

    def run_as_dicts(self, docs: list[dict]) -> list[dict]:
        """
        Convenience wrapper that returns chunks as plain dicts instead of
        dataclasses — useful when serialising to JSON for API responses.
        """
        return [c.to_dict() for c in self.run(docs)]


# ═══════════════════════════════════════════════════════════════════════════════
# 6. UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def filter_chunks_by_category(
    chunks: list[Chunk], category: str
) -> list[Chunk]:
    """Filter chunks by data_category (e.g., 'quarterly_financials')."""
    return [c for c in chunks if c.data_category == category]


def filter_chunks_by_ticker(
    chunks: list[Chunk], ticker: str
) -> list[Chunk]:
    """Filter chunks to those belonging to a specific ticker symbol."""
    return [c for c in chunks if c.ticker.upper() == ticker.upper()]


def chunk_stats(chunks: list[Chunk]) -> dict:
    """Return simple descriptive statistics about the chunk collection."""
    if not chunks:
        return {}
    lengths = [len(c.text) for c in chunks]
    tokens  = [estimate_tokens(c.text) for c in chunks]
    cats    = {}
    for c in chunks:
        cats[c.data_category] = cats.get(c.data_category, 0) + 1
    return {
        "total_chunks"        : len(chunks),
        "avg_chars"           : round(sum(lengths) / len(lengths), 1),
        "avg_tokens_estimated": round(sum(tokens) / len(tokens), 1),
        "min_chars"           : min(lengths),
        "max_chars"           : max(lengths),
        "by_category"         : cats,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ENTRY POINT (for local testing)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, json as _json
    from ingest_and_clean import FinanceDataPipeline

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    docs   = FinanceDataPipeline(data_dir=data_dir).run()
    chunks = ChunkingPipeline().run(docs)

    stats = chunk_stats(chunks)
    print("\n✅  Chunking Stats:")
    print(_json.dumps(stats, indent=2))

    if chunks:
        print("\n── Sample Chunk ─────────────────────────────────────────────")
        c = chunks[0]
        print(f"ID       : {c.chunk_id}")
        print(f"Source   : {c.source_file}")
        print(f"Ticker   : {c.ticker}")
        print(f"Date     : {c.date}")
        print(f"Category : {c.data_category}")
        print(f"Tokens~  : {estimate_tokens(c.text)}")
        print(f"Text     :\n{c.text[:300]}...")
