# ChromaDB Embedding Steps

This project embeds the provided finance dataset into a local ChromaDB vector
database. The app then retrieves the most similar chunks from Chroma before
sending context to the LLM.

## What The Pipeline Does

1. Scans the dataset folder recursively for `.csv`, `.json`, and CSV-formatted
   `.txt` files.
2. Converts each record into readable text with source metadata.
3. For stock/ETF OHLCV files under `Stocks/` or `ETFs/`, creates yearly summary
   chunks plus one overall summary per file.
4. Embeds each chunk using `BAAI/bge-small-en-v1.5` through
   `sentence-transformers`.
5. Stores vectors, chunk text, and metadata in ChromaDB at `./chroma_db`, inside
   the `finance_docs` collection.
6. At chat time, embeds the user query with the same model and asks Chroma for
   the closest vectors.

## Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Embed your provided dataset folder:

```bash
python ingest_large.py "/Users/yuvii/Downloads/drive-download-20260814T171128Z-1-001/archive (1)/Data"
```

Start over from a clean Chroma collection:

```bash
python ingest_large.py "/Users/yuvii/Downloads/drive-download-20260814T171128Z-1-001/archive (1)/Data" --reset
```

Run the API after embeddings are built:

```bash
export CHROMA_PERSIST_DIR="./chroma_db"
export CHROMA_COLLECTION_NAME="finance_docs"
export VECTOR_STORE_BACKEND="chroma"
uvicorn app.main:app --reload --port 8000
```

## Quick Chroma Check

```bash
python - <<'PY'
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("finance_docs")
print("vectors:", collection.count())
PY
```
