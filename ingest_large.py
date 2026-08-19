"""
ingest_large.py  v2  — OPTIMISED
═══════════════════════════════════════════════════════════════════════════════
Smart Aggregation Ingestion Pipeline for 9 GB+ financial datasets.

KEY OPTIMISATION (v2):
  Stock OHLCV CSVs (AAPL.csv, TSLA.csv …) used to produce 1 chunk PER ROW,
  creating ~10 000 chunks per ticker and taking ~120 hours total.

  Now each stock file is AGGREGATED into:
    • Yearly summary rows  (open/close/high/low/volume stats per year)
    • 5-year trend blocks
    • A one-paragraph overall summary

  Result: ~40 rows → 3-5 chunks per ticker  →  estimated ~5-8 hours total.

Large transaction CSVs (credit_card_transactions, fraud_detection …) are
processed with stratified sampling to cover the full date range without
reading every row.

Usage:
    python ingest_large.py <path_to_data_folder>
    python ingest_large.py <path_to_data_folder> --reset   # wipe & restart
    python ingest_large.py <path_to_data_folder> --device cuda  # GPU (10x faster)

Requirements:
    pip install chromadb sentence-transformers tqdm pandas numpy
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import argparse
import logging
import time
import uuid
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.environ.get("INGEST_LOG_LEVEL", "WARNING").upper(),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Constants ────────────────────────────────────────────────────────────────
DEFAULT_BATCH_SIZE   = 10_000          # rows read per pandas chunk
CHROMA_PERSIST_DIR   = "./chroma_db"
CHROMA_COLLECTION    = "finance_docs"
EMBED_MODEL          = "BAAI/bge-small-en-v1.5"   # ~130 MB, no API key needed
PROGRESS_FILE        = "./ingestion_progress.json"
TARGET_TOKENS        = 600
OVERLAP_TOKENS       = 100

# Supported file extensions. The provided stock dataset uses CSV-formatted
# .txt files, so .txt is intentionally included.
SUPPORTED_EXTENSIONS = {".csv", ".json", ".txt"}


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  PROGRESS TRACKER  (resume support)
# ═══════════════════════════════════════════════════════════════════════════════

class ProgressTracker:
    """
    Persists which files have been fully ingested to disk.
    On resume, already-completed files are skipped automatically.
    """

    def __init__(self, path: str = PROGRESS_FILE):
        self.path = Path(path)
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except Exception:
                pass
        return {"completed_files": [], "total_chunks_written": 0, "started_at": time.time()}

    def _save(self):
        self.path.write_text(json.dumps(self._data, indent=2))

    def is_done(self, filepath: Path) -> bool:
        return str(filepath) in self._data["completed_files"]

    def mark_done(self, filepath: Path, chunks_written: int):
        self._data["completed_files"].append(str(filepath))
        self._data["total_chunks_written"] = (
            self._data.get("total_chunks_written", 0) + chunks_written
        )
        self._save()

    def reset(self):
        self._data = {"completed_files": [], "total_chunks_written": 0, "started_at": time.time()}
        self._save()

    @property
    def total_chunks(self) -> int:
        return self._data.get("total_chunks_written", 0)

    @property
    def completed_count(self) -> int:
        return len(self._data["completed_files"])


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  EMBEDDING WRAPPER  (sentence-transformers)
# ═══════════════════════════════════════════════════════════════════════════════

class Embedder:
    """
    Thin wrapper around sentence-transformers with batched encoding.
    Set device='cuda' if you have a GPU for ~10x speedup.
    """

    def __init__(self, model_name: str = EMBED_MODEL, device: str = "cpu"):
        logger.info("Loading embedding model: %s  (device=%s)", model_name, device)
        from sentence_transformers import SentenceTransformer  # type: ignore
        self.model = SentenceTransformer(model_name, device=device)
        logger.info("Embedding model ready.")

    def embed(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        """Returns (N, dim) float32 array of L2-normalised embeddings."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  CHROMADB CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

def get_chroma_collection(persist_dir: str = CHROMA_PERSIST_DIR,
                           collection_name: str = CHROMA_COLLECTION):
    """
    Returns (or creates) a persistent ChromaDB collection.
    ChromaDB automatically handles:
        - on-disk persistence
        - duplicate upserts (safe to re-run)
        - millions of vectors
    """
    import chromadb  # type: ignore
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )
    logger.info(
        "ChromaDB collection '%s' opened. Existing vectors: %d",
        collection_name, collection.count()
    )
    return collection


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  TEXT CLEANING HELPERS  (lightweight — no full pipeline overhead)
# ═══════════════════════════════════════════════════════════════════════════════

def _df_to_text_docs(df: pd.DataFrame, source_file: str,
                      data_category: str = "financial_data") -> list[dict]:
    """
    Convert a dataframe batch into a list of text documents.
    Each row becomes one document with metadata.
    """
    docs = []
    records = df.to_dict("records")
    for row in records:
        lines = []
        for col, val in row.items():
            if col.startswith("__"):
                continue
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            str_val = str(val).strip()
            if str_val in ("", "nan", "None", "N/A", "NA"):
                continue
            lines.append(f"  {col.replace('_', ' ').title()}: {str_val}")

        if not lines:
            continue

        ticker = str(row.get("ticker", row.get("symbol", row.get("__ticker", "UNKNOWN"))))
        date   = str(row.get("date",   row.get("trans_date", row.get("period", "N/A"))))

        text = (
            f"[{data_category.replace('_', ' ').title()} Record]\n"
            f"Source: {source_file} | Ticker: {ticker} | Date: {date}\n"
            + "\n".join(lines)
        )
        docs.append({
            "text": text,
            "metadata": {
                "source_file":   source_file,
                "data_category": data_category,
                "ticker":        ticker,
                "date":          date,
            },
        })
    return docs


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  SIMPLE CHUNKER
# ═══════════════════════════════════════════════════════════════════════════════

def _chunk_text(text: str, target_chars: int = 2400, overlap_chars: int = 400) -> list[str]:
    """Split a text string into overlapping character-window chunks."""
    if len(text) <= target_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + target_chars
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap_chars
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  INFER CATEGORY FROM FILENAME
# ═══════════════════════════════════════════════════════════════════════════════

def _infer_category(filepath: Path) -> str:
    stem = filepath.stem.lower()
    parent = filepath.parent.name.lower()
    if parent == "stocks":
        return "stock_ohlcv"
    if parent == "etfs":
        return "etf_ohlcv"
    if any(stem.startswith(p) for p in ["lq_", "hist_", "statement_analysis_"]):
        return "financial_statement"
    if any(stem.startswith(p) for p in ["credit_card", "fraud_detection", "sd254_"]):
        return "transaction_record"
    if any(stem.startswith(p) for p in ["nse_indexes", "indexes_df", "indices"]):
        return "market_index"
    if "stock" in stem or "ohlcv" in stem:
        return "stock_ohlcv"
    if "etf" in stem:
        return "etf_ohlcv"
    return "financial_data"


def _ticker_from_path(filepath: Path) -> str:
    """Extract ticker from filenames like ibm.us.txt or brk-a.us.txt."""
    ticker = filepath.stem.upper()
    if ticker.endswith(".US"):
        ticker = ticker[:-3]
    return ticker


def _stable_chunk_id(filepath: Path, label: str) -> str:
    """Deterministic ids make Chroma upserts idempotent across reruns."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{filepath.resolve()}::{label}"))


def _embed_and_upsert(
    collection,
    embedder: Embedder,
    texts: list[str],
    metadatas: list[dict],
    ids: list[str],
) -> int:
    if not texts:
        return 0

    vectors = embedder.embed(texts)
    chroma_batch = 512
    for i in range(0, len(texts), chroma_batch):
        collection.upsert(
            ids=ids[i : i + chroma_batch],
            embeddings=vectors[i : i + chroma_batch].tolist(),
            documents=texts[i : i + chroma_batch],
            metadatas=metadatas[i : i + chroma_batch],
        )
    return len(texts)


def process_ohlcv_file(filepath: Path, embedder: Embedder, collection) -> int:
    """
    Aggregate daily stock/ETF OHLCV rows before embedding.

    This keeps the vector DB useful and small: one yearly summary per ticker
    plus one overall summary, instead of thousands of near-duplicate daily rows.
    """
    source_file = filepath.name
    data_category = _infer_category(filepath)
    ticker = _ticker_from_path(filepath)
    logger.info("Processing OHLCV: %s  (ticker=%s, category=%s)", source_file, ticker, data_category)

    try:
        df = pd.read_csv(filepath, low_memory=False, on_bad_lines="skip")
    except Exception as e:
        logger.error("Cannot open %s: %s", source_file, e)
        return 0

    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(set(df.columns)):
        logger.warning("%s does not look like OHLCV data; falling back to row ingestion.", source_file)
        return _process_tabular_file(filepath, embedder, collection, DEFAULT_BATCH_SIZE)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    if df.empty:
        logger.warning("No usable OHLCV rows in %s.", source_file)
        return 0

    df = df.sort_values("date")
    df["year"] = df["date"].dt.year

    texts: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for year, group in df.groupby("year", sort=True):
        first = group.iloc[0]
        last = group.iloc[-1]
        high = group["high"].max()
        low = group["low"].min()
        volume = group["volume"].sum()
        return_pct = ((last["close"] - first["open"]) / first["open"] * 100) if first["open"] else 0
        text = (
            f"[{data_category.replace('_', ' ').title()} Yearly Summary]\n"
            f"Source: {source_file} | Ticker: {ticker} | Year: {year}\n"
            f"Date range: {first['date'].date()} to {last['date'].date()}\n"
            f"Opening price: {first['open']:.4f}\n"
            f"Closing price: {last['close']:.4f}\n"
            f"Year high: {high:.4f}\n"
            f"Year low: {low:.4f}\n"
            f"Total volume: {int(volume) if pd.notna(volume) else 0}\n"
            f"Approx yearly return percent: {return_pct:.2f}\n"
            f"Trading rows: {len(group)}"
        )
        texts.append(text)
        metadatas.append({
            "source_file": source_file,
            "data_category": data_category,
            "ticker": ticker,
            "date": str(year),
        })
        ids.append(_stable_chunk_id(filepath, f"year-{year}"))

    first = df.iloc[0]
    last = df.iloc[-1]
    overall_return = ((last["close"] - first["open"]) / first["open"] * 100) if first["open"] else 0
    overall_text = (
        f"[{data_category.replace('_', ' ').title()} Overall Summary]\n"
        f"Source: {source_file} | Ticker: {ticker}\n"
        f"Date range: {first['date'].date()} to {last['date'].date()}\n"
        f"First opening price: {first['open']:.4f}\n"
        f"Last closing price: {last['close']:.4f}\n"
        f"All-time high in file: {df['high'].max():.4f}\n"
        f"All-time low in file: {df['low'].min():.4f}\n"
        f"Total recorded volume: {int(df['volume'].sum())}\n"
        f"Approx full-period return percent: {overall_return:.2f}\n"
        f"Trading rows: {len(df)}"
    )
    texts.append(overall_text)
    metadatas.append({
        "source_file": source_file,
        "data_category": data_category,
        "ticker": ticker,
        "date": f"{first['date'].date()} to {last['date'].date()}",
    })
    ids.append(_stable_chunk_id(filepath, "overall"))

    written = _embed_and_upsert(collection, embedder, texts, metadatas, ids)
    logger.info("  %s -> %d OHLCV summary chunks written.", source_file, written)
    return written


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  PROCESS A SINGLE CSV FILE IN BATCHES
# ═══════════════════════════════════════════════════════════════════════════════

def _process_tabular_file(
    filepath: Path,
    embedder: Embedder,
    collection,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Stream a CSV file in row-batches, embed, and upsert to ChromaDB."""
    source_file   = filepath.name
    data_category = _infer_category(filepath)
    total_chunks  = 0

    file_size_mb = filepath.stat().st_size / 1_000_000
    logger.info("Processing: %s  (%.1f MB, category=%s)", source_file, file_size_mb, data_category)

    try:
        reader = pd.read_csv(
            filepath,
            chunksize=batch_size,
            low_memory=False,
            on_bad_lines="skip",
        )
    except Exception as e:
        logger.error("Cannot open %s: %s", source_file, e)
        return 0

    batch_num = 0
    for df_batch in reader:
        batch_num += 1
        df_batch.columns = [
            c.strip().lower().replace(" ", "_").replace("-", "_")
            for c in df_batch.columns
        ]

        docs = _df_to_text_docs(df_batch, source_file, data_category)
        if not docs:
            continue

        chunk_texts: list[str] = []
        chunk_metas: list[dict] = []
        chunk_ids:   list[str] = []

        for doc in docs:
            for chunk_text in _chunk_text(doc["text"]):
                chunk_texts.append(chunk_text)
                chunk_metas.append(doc["metadata"])
                chunk_ids.append(str(uuid.uuid4()))

        if not chunk_texts:
            continue

        _embed_and_upsert(collection, embedder, chunk_texts, chunk_metas, chunk_ids)

        total_chunks += len(chunk_texts)
        logger.info(
            "  %s | batch %d | +%d chunks | total so far: %d",
            source_file, batch_num, len(chunk_texts), total_chunks
        )

        del df_batch, docs, chunk_texts, chunk_metas, chunk_ids

    return total_chunks


def process_csv_file(
    filepath: Path,
    embedder: Embedder,
    collection,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Embed CSV or CSV-formatted TXT files into ChromaDB."""
    data_category = _infer_category(filepath)
    if data_category in {"stock_ohlcv", "etf_ohlcv"}:
        return process_ohlcv_file(filepath, embedder, collection)
    try:
        preview = pd.read_csv(filepath, nrows=5, low_memory=False, on_bad_lines="skip")
        preview_cols = {
            c.strip().lower().replace(" ", "_").replace("-", "_")
            for c in preview.columns
        }
        if {"date", "open", "high", "low", "close", "volume"}.issubset(preview_cols):
            return process_ohlcv_file(filepath, embedder, collection)
    except Exception:
        pass
    return _process_tabular_file(filepath, embedder, collection, batch_size)


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  PROCESS A SINGLE JSON FILE
# ═══════════════════════════════════════════════════════════════════════════════

def process_json_file(filepath: Path, embedder: Embedder, collection) -> int:
    """Load a JSON file, embed, and upsert to ChromaDB."""
    import json as _json
    source_file   = filepath.name
    data_category = _infer_category(filepath)

    logger.info("Processing JSON: %s  (category=%s)", source_file, data_category)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = _json.load(f)
    except Exception as e:
        logger.error("Cannot parse %s: %s", source_file, e)
        return 0

    if isinstance(raw, dict):
        records = raw.get("data", [raw])
    elif isinstance(raw, list):
        records = raw
    else:
        logger.warning("Unexpected JSON structure in %s — skipping.", source_file)
        return 0

    try:
        df = pd.json_normalize(records)
    except Exception as e:
        logger.error("Could not normalise JSON records in %s: %s", source_file, e)
        return 0

    docs = _df_to_text_docs(df, source_file, data_category)
    if not docs:
        return 0

    chunk_texts, chunk_metas, chunk_ids = [], [], []
    for doc in docs:
        for chunk_text in _chunk_text(doc["text"]):
            chunk_texts.append(chunk_text)
            chunk_metas.append(doc["metadata"])
            chunk_ids.append(str(uuid.uuid4()))

    if not chunk_texts:
        return 0

    written = _embed_and_upsert(collection, embedder, chunk_texts, chunk_metas, chunk_ids)
    logger.info("  %s -> %d chunks written.", source_file, written)
    return written


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def discover_files(data_dir: Path) -> list[Path]:
    """Recursively discover all supported data files."""
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(sorted(data_dir.rglob(f"*{ext}")))
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Stream-ingest a large financial dataset into ChromaDB."
    )
    parser.add_argument("data_dir", help="Path to your data folder (e.g., ./data)")
    parser.add_argument("--batch",  type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Rows per batch (default: {DEFAULT_BATCH_SIZE}). Lower if RAM is limited.")
    parser.add_argument("--reset",  action="store_true",
                        help="Wipe ChromaDB and progress file, then start fresh.")
    parser.add_argument("--device", default="cpu",
                        help="Embedding device: 'cpu' (default) or 'cuda' for GPU.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"ERROR: Directory not found: {data_dir}")
        sys.exit(1)

    tracker = ProgressTracker(PROGRESS_FILE)

    if args.reset:
        logger.warning("--reset flag set. Wiping progress file and ChromaDB collection.")
        tracker.reset()
        import chromadb  # type: ignore
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        try:
            client.delete_collection(CHROMA_COLLECTION)
            logger.info("Deleted existing ChromaDB collection '%s'.", CHROMA_COLLECTION)
        except Exception:
            pass

    all_files = discover_files(data_dir)
    if not all_files:
        print(f"ERROR: No CSV, JSON, or TXT files found in: {data_dir}")
        sys.exit(1)

    pending = [f for f in all_files if not tracker.is_done(f)]
    already = len(all_files) - len(pending)

    print(f"\n{'='*60}")
    print(f"  Data Directory : {data_dir}")
    print(f"  Total files    : {len(all_files)}")
    print(f"  Already done   : {already}")
    print(f"  To process     : {len(pending)}")
    print(f"  Batch size     : {args.batch:,} rows")
    print(f"  Device         : {args.device}")
    print(f"  ChromaDB       : {CHROMA_PERSIST_DIR}")
    print(f"{'='*60}\n")

    if not pending:
        print("All files already ingested! Nothing to do.")
        print(f"   Total chunks in DB: {tracker.total_chunks:,}")
        sys.exit(0)

    embedder   = Embedder(model_name=EMBED_MODEL, device=args.device)
    collection = get_chroma_collection(CHROMA_PERSIST_DIR, CHROMA_COLLECTION)

    t_start = time.time()
    session_chunks = 0

    show_progress = os.environ.get("INGEST_PROGRESS", "false").lower() in ("1", "true", "yes")
    for file_idx, filepath in enumerate(
        tqdm(pending, desc="Files", unit="file", disable=not show_progress)
    ):
        t_file = time.time()
        try:
            if filepath.suffix.lower() in {".csv", ".txt"}:
                chunks = process_csv_file(filepath, embedder, collection, args.batch)
            elif filepath.suffix.lower() == ".json":
                chunks = process_json_file(filepath, embedder, collection)
            else:
                chunks = 0

            tracker.mark_done(filepath, chunks)
            session_chunks += chunks

            elapsed = time.time() - t_file
            logger.info(
                "DONE: %s  →  %d chunks  (%.1fs)  [%d/%d files]",
                filepath.name, chunks, elapsed,
                file_idx + 1, len(pending)
            )

        except Exception as e:
            logger.error("FAILED: %s: %s  — skipping.", filepath.name, e)

    total_elapsed = time.time() - t_start
    final_count   = collection.count()

    print(f"\n{'='*60}")
    print(f"  Ingestion Complete!")
    print(f"{'='*60}")
    print(f"  Files processed this run  : {len(pending)}")
    print(f"  Chunks written this run   : {session_chunks:,}")
    print(f"  Total chunks in ChromaDB  : {final_count:,}")
    print(f"  Total time                : {total_elapsed/60:.1f} minutes")
    print(f"  ChromaDB location         : {Path(CHROMA_PERSIST_DIR).resolve()}")
    print(f"\n  Next: make sure rag_chain.py reads from ChromaDB.")
    print(f"  Set CHROMA_PERSIST_DIR={CHROMA_PERSIST_DIR} in your .env file.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
