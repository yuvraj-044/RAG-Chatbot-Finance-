"""
run_pipeline.py  --  One-shot full pipeline runner
══════════════════════════════════════════════════════════════════════════════
Run this ONCE after you have your dataset in place.
It will:
  1. Scan your data directory and show what files were found
  2. Clean & normalize all records
  3. Chunk into 600-token windows
  4. Embed every chunk with BAAI/bge-small-en-v1.5 (local, no API key needed)
  5. Save the vector store to ./vector_store/

Usage:
    python run_pipeline.py <path_to_your_data_folder>

Example:
    python run_pipeline.py "C:/Users/admin/Downloads/hackathon_data"
══════════════════════════════════════════════════════════════════════════════
"""

import sys
import json
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── pipeline imports ──────────────────────────────────────────────────────────
from ingest_and_clean import FinanceDataPipeline
from chunking          import ChunkingPipeline, chunk_stats
from embeddings        import EmbeddingStore

PERSIST_PATH = "./vector_store/embeddings"


def print_banner(text: str):
    bar = "=" * 60
    print(f"\n{bar}\n  {text}\n{bar}")


def scan_and_report(data_dir: str):
    """Show the user exactly which files will be processed and in which bucket."""
    from ingest_and_clean import FinanceDataPipeline as FDP
    pipeline = FDP(data_dir=data_dir)
    buckets  = pipeline._discover_files()

    print_banner("File Discovery Report")
    total = 0
    for bucket, files in buckets.items():
        print(f"\n  [{bucket.upper()}]  --  {len(files)} file(s)")
        for fp in files:
            size_kb = fp.stat().st_size / 1024
            print(f"    OK  {fp.name:<50}  ({size_kb:.1f} KB)")
        total += len(files)

    if total == 0:
        print("\n  WARNING: No matching files found!")
        print("  Check that your filenames start with the expected prefixes:")
        print("    Financial   : LQ_, Hist_, Statement_Analysis_")
        print("    Transaction : credit_card_transactions, fraud_detection_, sd254_")
        print("    Market CSV  : nse_indexes, indexes_df")
        print("    Market JSON : indices")
        sys.exit(1)

    print(f"\n  Total files to process: {total}")
    return pipeline


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py <data_directory>")
        print("Example: python run_pipeline.py ./data")
        sys.exit(1)

    data_dir = sys.argv[1]

    if not Path(data_dir).is_dir():
        print(f"❌  Directory not found: {data_dir}")
        sys.exit(1)

    # ── Step 1: Scan ──────────────────────────────────────────────────────────
    pipeline = scan_and_report(data_dir)

    auto_yes = "-y" in sys.argv or "--yes" in sys.argv or not sys.stdin.isatty()
    if not auto_yes:
        try:
            input("\n  Press ENTER to start the pipeline ...")
        except (EOFError, KeyboardInterrupt):
            pass

    t0 = time.time()

    # ── Step 2: Ingest & Clean ────────────────────────────────────────────────
    print_banner("Step 1/3 -- Ingesting & Cleaning")
    docs = pipeline.run()
    print(f"\n  Documents produced: {len(docs):,}")
    if docs:
        print(f"  Sample metadata: {json.dumps(docs[0]['metadata'], indent=4)}")
        print(f"  Sample text preview:\n  {docs[0]['text'][:300]!r}")

    # ── Step 3: Chunk ─────────────────────────────────────────────────────────
    print_banner("Step 2/3 -- Chunking (600 tokens, 100 overlap)")
    chunks = ChunkingPipeline(target_tokens=600, overlap_tokens=100).run(docs)
    stats  = chunk_stats(chunks)
    print("\n  Chunk statistics:")
    print(json.dumps(stats, indent=4))

    # ── Step 4: Embed & Save ──────────────────────────────────────────────────
    print_banner("Step 3/3 -- Embedding (BAAI/bge-small-en-v1.5)")
    print("  Note: First run will download the model (~130 MB). Subsequent runs use cache.\n")

    store = EmbeddingStore(backend="sentence_transformers", persist_path=PERSIST_PATH)
    store.build(chunks)

    elapsed = time.time() - t0

    # ── Done ──────────────────────────────────────────────────────────────────
    print_banner("Pipeline Complete!")
    print(f"\n  Vectors indexed : {store.size:,}")
    print(f"  Embedding dim   : {store.dim}")
    print(f"  Saved to        : {PERSIST_PATH}.npy + {PERSIST_PATH}_meta.json")
    print(f"  Total time      : {elapsed:.1f}s")

    # ── Quick smoke test ──────────────────────────────────────────────────────
    print_banner("Quick Search Test")
    test_queries = [
        "net profit quarterly earnings",
        "fraudulent transaction credit card",
        "market index performance",
    ]
    for q in test_queries:
        results = store.search(q, k=2)
        print(f"\n  Query: '{q}'")
        for r in results:
            print(f"    Rank {r['rank']}  Score={r['score']:.4f}  "
                  f"{r['source_file']} | {r['ticker']} | {r['date']}")

    print("\n\n  DONE. Hand the following to Member 2 (Backend Lead):")
    print(f"     PERSIST_PATH = '{PERSIST_PATH}'")
    print( "     They just need:  uvicorn api_example:app --reload --port 8000\n")


if __name__ == "__main__":
    main()
