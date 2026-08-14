"""
ingest_and_clean.py
═══════════════════════════════════════════════════════════════════════════════
Finance RAG Chatbot — Member 1: Data & AI Lead
Purpose : Parse, clean, normalize, and convert all financial and transaction
          datasets into standardized, chunking-ready text representations.

Supported dataset categories
  • Financial Statements  : LQ_*, Hist_*, Statement_Analysis_*
  • Credit Card / Fraud   : credit_card_transactions*, fraud_detection_*, sd254_*
  • Stock & Market Data   : nse_indexes.csv, indexes_df.csv, indices.json
  • Corporate Metadata    : quote data, news, peer comparisons

Usage (standalone):
    from ingest_and_clean import FinanceDataPipeline
    pipeline = FinanceDataPipeline(data_dir="./data")
    docs = pipeline.run()          # returns list[dict]  ← ready for chunking.py
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import re
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FIELD-LEVEL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def clean_currency(value: Any) -> float | None:
    """
    Strip currency symbols and formatting, returning a clean float.

    Examples
        '$1,234,567.89'  →  1234567.89
        '(500.00)'       →  -500.0   (parentheses = negative)
        None / NaN       →  None
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text in ("-", "N/A", "NA", "nan"):
        return None

    # Parentheses notation → negative
    negative = text.startswith("(") and text.endswith(")")
    # Remove all non-numeric characters except period and minus
    text = re.sub(r"[$,£€\(\)\s]", "", text)
    try:
        result = float(text)
        return -result if negative else result
    except ValueError:
        return None


def standardize_date(value: Any) -> str | None:
    """
    Parse a wide variety of date strings into ISO-8601 'YYYY-MM-DD'.

    Handles: 'Jan 2024', '01/15/2024', '2024-Q1', timestamps, Unix ints, etc.
    Returns None when the value cannot be parsed.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text in ("N/A", "NA", "nan"):
        return None

    # Quarter notation: 2024-Q1 → 2024-03-31
    qtr_match = re.match(r"(\d{4})-?Q([1-4])", text, re.IGNORECASE)
    if qtr_match:
        year, q = int(qtr_match.group(1)), int(qtr_match.group(2))
        month_end = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
        return f"{year}-{month_end[q]}"

    # Unix timestamp (integer seconds)
    if re.fullmatch(r"\d{9,13}", text):
        ts = int(text)
        if ts > 1e10:           # milliseconds
            ts //= 1000
        return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")

    FORMATS = [
        "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y",
        "%m/%d/%Y", "%m-%d-%Y", "%b %d, %Y", "%B %d, %Y",
        "%d %b %Y", "%d %B %Y", "%b %Y", "%B %Y",
        "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in FORMATS:
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    logger.debug("Could not parse date: %r", value)
    return None


def flatten_json(nested: dict, sep: str = "__", prefix: str = "") -> dict:
    """
    Recursively flatten a nested dictionary.

    Example
        {"a": {"b": 1, "c": [2, 3]}}
        → {"a__b": 1, "a__c": "[2, 3]"}
    """
    flat: dict = {}
    for key, val in nested.items():
        full_key = f"{prefix}{sep}{key}" if prefix else key
        if isinstance(val, dict):
            flat.update(flatten_json(val, sep=sep, prefix=full_key))
        elif isinstance(val, list):
            flat[full_key] = json.dumps(val)   # keep arrays as JSON strings
        else:
            flat[full_key] = val
    return flat


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PER-CATEGORY LOADERS & CLEANERS
# ═══════════════════════════════════════════════════════════════════════════════

class FinancialStatementCleaner:
    """
    Handles: LQ_*.csv, Hist_*.csv, Statement_Analysis_*.csv
    Covers  : Balance sheets, income statements, cash flow statements.
    """

    CURRENCY_COLS_PATTERNS = [
        "revenue", "income", "profit", "loss", "assets", "liabilities",
        "equity", "cash", "debt", "ebitda", "eps", "dividend",
        "operating", "gross", "net", "total", "value", "amount",
    ]
    DATE_COLS_PATTERNS = ["date", "period", "quarter", "year", "filing"]

    def load_and_clean(self, filepath: Path) -> pd.DataFrame:
        logger.info("Loading financial statement: %s", filepath.name)
        df = pd.read_csv(filepath, low_memory=False)
        df = self._rename_columns(df)
        df = self._clean_currency_columns(df)
        df = self._standardize_date_columns(df)
        df = self._fill_missing_values(df)
        df["__source_file"] = filepath.name
        df["__data_category"] = self._infer_category(filepath.name)
        df["__ticker"] = self._extract_ticker(filepath.name)
        return df

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [
            re.sub(r"[\s\-/]+", "_", c.strip()).lower() for c in df.columns
        ]
        return df

    def _clean_currency_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            col_lower = col.lower()
            if any(p in col_lower for p in self.CURRENCY_COLS_PATTERNS):
                df[col] = df[col].apply(clean_currency)
        return df

    def _standardize_date_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            col_lower = col.lower()
            if any(p in col_lower for p in self.DATE_COLS_PATTERNS):
                df[col] = df[col].apply(standardize_date)
        return df

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Balance sheet line items default to 0 when absent
        df[numeric_cols] = df[numeric_cols].fillna(0)
        # Non-numeric columns: forward-fill then back-fill (time series)
        non_numeric = df.select_dtypes(exclude=[np.number]).columns
        df[non_numeric] = df[non_numeric].ffill().bfill()
        return df

    def _infer_category(self, filename: str) -> str:
        name = filename.lower()
        if "lq_" in name:            return "quarterly_financials"
        if "hist_" in name:          return "historical_financials"
        if "statement_analysis" in name: return "statement_analysis"
        if "cash" in name:           return "cash_flow"
        if "balance" in name:        return "balance_sheet"
        return "financial_statement"

    def _extract_ticker(self, filename: str) -> str:
        """Best-effort ticker extraction from filename, e.g. 'LQ_AAPL_2024.csv' → 'AAPL'."""
        parts = re.split(r"[_\-\.]", filename.upper())
        # Filter short alpha tokens likely to be tickers
        candidates = [p for p in parts if 2 <= len(p) <= 5 and p.isalpha()
                      and p not in {"LQ", "HIST", "CSV", "STMT", "AN"}]
        return candidates[0] if candidates else "UNKNOWN"

    def to_text_chunks(self, df: pd.DataFrame) -> list[dict]:
        """
        Convert each row of a financial statement into a structured text block
        ready for chunking, with metadata attached.
        """
        docs = []
        records = df.to_dict("records")
        for row in records:
            meta = {
                "source_file": row.get("__source_file", "unknown"),
                "data_category": row.get("__data_category", "financial_statement"),
                "ticker": row.get("__ticker", "UNKNOWN"),
            }
            # Identify the primary date field for citation
            date_val = None
            for col in row:
                if any(p in col for p in ["date", "period", "quarter"]):
                    date_val = row[col]
                    break
            meta["date"] = str(date_val) if date_val is not None else "N/A"

            # Build human-readable text block from all data columns
            data_cols = [c for c in row if not c.startswith("__")]
            lines = [f"  {col.replace('_',' ').title()}: {row[col]}"
                     for col in data_cols if pd.notna(row[col]) and row[col] not in ("", "N/A", "nan")]
            text = (
                f"[Financial Record]\n"
                f"Source : {meta['source_file']} | Ticker: {meta['ticker']}\n"
                f"Period : {meta['date']} | Category: {meta['data_category']}\n"
                + "\n".join(lines)
            )
            docs.append({"text": text, "metadata": meta})
        return docs


# ─────────────────────────────────────────────────────────────────────────────

class TransactionDataCleaner:
    """
    Handles: credit_card_transactions*.csv, fraud_detection_*.csv, sd254_*.csv
    Covers : Transaction logs, merchant info, fraud flags, demographics.
    """

    # Threshold: files larger than this will be sampled
    _LARGE_FILE_THRESHOLD = 50 * 1024 * 1024   # 50 MB
    _LARGE_FILE_SAMPLE    = 5000               # max rows

    def load_and_clean(self, filepath: Path) -> pd.DataFrame:
        logger.info("Loading transaction dataset: %s", filepath.name)
        size = filepath.stat().st_size
        if size > self._LARGE_FILE_THRESHOLD:
            logger.warning(
                "Large transaction file (%.1f MB) — sampling %d rows.",
                size / 1e6, self._LARGE_FILE_SAMPLE,
            )
            df = pd.read_csv(filepath, nrows=self._LARGE_FILE_SAMPLE, low_memory=False)
        else:
            df = pd.read_csv(filepath, low_memory=False)
        df.columns = [re.sub(r"[\s\-/]+", "_", c.strip()).lower() for c in df.columns]
        df = self._clean_amounts(df)
        df = self._standardize_dates(df)
        df = self._encode_fraud_flag(df)
        df = self._handle_missing(df)
        df["__source_file"] = filepath.name
        df["__data_category"] = "transaction_record"
        return df

    def _clean_amounts(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if any(k in col for k in ["amount", "balance", "limit", "spend", "value"]):
                df[col] = df[col].apply(clean_currency)
        return df

    def _standardize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if any(k in col for k in ["date", "time", "dob", "birth", "acct_open"]):
                df[col] = df[col].apply(standardize_date)
        return df

    def _encode_fraud_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if "fraud" in col or "is_fraud" in col:
                df[col] = df[col].map(
                    {1: "Yes", 0: "No", "1": "Yes", "0": "No",
                     True: "Yes", False: "No", "yes": "Yes", "no": "No"}
                ).fillna("Unknown")
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        # Fill numeric NaN with 0; string NaN with "N/A"
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna("N/A")
        return df

    def to_text_chunks(self, df: pd.DataFrame) -> list[dict]:
        """Convert transaction rows into readable text documents with metadata."""
        docs = []
        records = df.to_dict("records")
        for row in records:
            meta = {
                "source_file": row.get("__source_file", "unknown"),
                "data_category": row.get("__data_category", "transaction_record"),
                "ticker": "N/A",
                "date": str(row.get("trans_date", row.get("date", "N/A"))),
            }
            data_cols = [c for c in row if not c.startswith("__")]
            lines = [f"  {col.replace('_',' ').title()}: {row[col]}"
                     for col in data_cols if pd.notna(row[col]) and str(row[col]) not in ("N/A", "", "nan", "None")]
            text = (
                f"[Transaction Record]\n"
                f"Source : {meta['source_file']}\n"
                f"Date   : {meta['date']} | Category: {meta['data_category']}\n"
                + "\n".join(lines)
            )
            docs.append({"text": text, "metadata": meta})
        return docs


# ─────────────────────────────────────────────────────────────────────────────

class MarketDataCleaner:
    """
    Handles: nse_indexes.csv, indexes_df.csv, indices.json
    Covers : Market index prices, volumes, historical OHLCV data.
    """

    _LARGE_FILE_THRESHOLD = 20 * 1024 * 1024   # 20 MB
    _LARGE_FILE_SAMPLE    = 2000               # max rows

    def load_csv(self, filepath: Path) -> pd.DataFrame:
        logger.info("Loading market CSV: %s", filepath.name)
        size = filepath.stat().st_size
        if size > self._LARGE_FILE_THRESHOLD:
            logger.warning(
                "Large market file (%.1f MB) — sampling %d rows.",
                size / 1e6, self._LARGE_FILE_SAMPLE,
            )
            df = pd.read_csv(filepath, nrows=self._LARGE_FILE_SAMPLE, low_memory=False)
        else:
            df = pd.read_csv(filepath, low_memory=False)

        df.columns = [re.sub(r"[\s\-/]+", "_", c.strip()).lower() for c in df.columns]
        df = self._clean_numeric_cols(df)
        df = self._standardize_date_cols(df)
        df = df.ffill().bfill()        # forward-fill time-series gaps
        df["__source_file"] = filepath.name
        df["__data_category"] = "market_index"
        return df

    def load_json(self, filepath: Path) -> list[dict]:
        """Load and flatten indices.json into a list of flat records."""
        logger.info("Loading market JSON: %s", filepath.name)
        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)

        records = []
        # Support both top-level list and dict-of-lists
        items = raw if isinstance(raw, list) else raw.get("data", [raw])
        for item in items:
            flat = flatten_json(item)
            flat["__source_file"] = filepath.name
            flat["__data_category"] = "market_index_json"
            records.append(flat)
        return records

    def _clean_numeric_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if any(k in col for k in ["open", "high", "low", "close",
                                       "price", "volume", "change", "value",
                                       "index", "points", "return"]):
                df[col] = df[col].apply(clean_currency)
        return df

    def _standardize_date_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if any(k in col for k in ["date", "time", "period"]):
                df[col] = df[col].apply(standardize_date)
        return df

    def df_to_text_chunks(self, df: pd.DataFrame) -> list[dict]:
        docs = []
        records = df.to_dict("records")
        for row in records:
            meta = {
                "source_file": row.get("__source_file", "unknown"),
                "data_category": row.get("__data_category", "market_index"),
                "ticker": str(row.get("symbol", row.get("index_name", "MARKET"))),
                "date": str(row.get("date", row.get("trade_date", "N/A"))),
            }
            data_cols = [c for c in row if not c.startswith("__")]
            lines = [f"  {col.replace('_',' ').title()}: {row[col]}"
                     for col in data_cols if pd.notna(row[col]) and row[col] not in ("", "N/A", "nan")]
            text = (
                f"[Market Index Record]\n"
                f"Source : {meta['source_file']} | Symbol: {meta['ticker']}\n"
                f"Date   : {meta['date']}\n"
                + "\n".join(lines)
            )
            docs.append({"text": text, "metadata": meta})
        return docs

    def json_records_to_text_chunks(self, records: list[dict]) -> list[dict]:
        docs = []
        for rec in records:
            meta = {
                "source_file": rec.get("__source_file", "unknown"),
                "data_category": rec.get("__data_category", "market_index_json"),
                "ticker": str(rec.get("symbol", rec.get("indexName", "MARKET"))),
                "date": standardize_date(rec.get("date", rec.get("lastUpdated"))) or "N/A",
            }
            data_lines = [
                f"  {k.replace('__','').replace('_',' ').title()}: {v}"
                for k, v in rec.items()
                if not k.startswith("__") and v not in (None, "", "null")
            ]
            text = (
                f"[Market Index JSON Record]\n"
                f"Source : {meta['source_file']} | Symbol: {meta['ticker']}\n"
                f"Date   : {meta['date']}\n"
                + "\n".join(data_lines)
            )
            docs.append({"text": text, "metadata": meta})
        return docs


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ORCHESTRATOR — Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class FinanceDataPipeline:
    """
    Discovers all supported dataset files in a directory tree and runs the
    full cleaning + text-representation pipeline.

    Returns a unified list[dict] where each element is:
        {
            "text"     : str,   ← human-readable text block
            "metadata" : dict   ← source_file, data_category, ticker, date
        }
    This output is the direct input to chunking.py.
    """

    FINANCIAL_PREFIXES = ("lq_", "hist_", "statement_analysis_")
    TRANSACTION_STEMS  = (
        "credit_card_transactions", "fraud_detection_",
        "sd254_", "user0_credit_card",
    )
    MARKET_CSV_STEMS   = (
        "nse_indexes", "indexes_df", "stocks_df", "transactions_df",
        "symbols_valid_meta", "active_companies_list",
        "period_trend_data", "quote_data",
        "peers_comparisons_data", "corporate_news_data",
        "customer_profiles_table", "terminal_profiles_table",
    )
    MARKET_JSON_STEMS  = ("indices", "stk")

    # Large files to sample instead of loading fully (size threshold in bytes)
    _LARGE_FILE_THRESHOLD = 50 * 1024 * 1024   # 50 MB
    _LARGE_FILE_SAMPLE    = 5000               # max rows to sample

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.fs_cleaner  = FinancialStatementCleaner()
        self.tx_cleaner  = TransactionDataCleaner()
        self.mkt_cleaner = MarketDataCleaner()

    # ── File discovery ────────────────────────────────────────────────────────

    def _discover_files(self) -> dict[str, list[Path]]:
        buckets: dict[str, list[Path]] = {
            "financial": [], "transaction": [],
            "market_csv": [], "market_json": [],
            "stock_csv":  [], "etf_csv":    [],
        }
        for path in sorted(self.data_dir.rglob("*")):
            if not path.is_file():
                continue
            stem   = path.stem.lower()
            suffix = path.suffix.lower()

            # ── Identify parent folder context ────────────────────────────
            parts_upper = [p.name.upper() for p in path.parents]
            in_stocks = "STOCKS" in parts_upper
            in_etfs   = "ETFS"   in parts_upper

            if suffix == ".csv":
                if in_stocks:
                    buckets["stock_csv"].append(path)
                elif in_etfs:
                    buckets["etf_csv"].append(path)
                elif any(stem.startswith(p) for p in self.FINANCIAL_PREFIXES):
                    buckets["financial"].append(path)
                elif any(stem.startswith(p) for p in self.TRANSACTION_STEMS):
                    buckets["transaction"].append(path)
                elif any(stem.startswith(p) for p in self.MARKET_CSV_STEMS):
                    buckets["market_csv"].append(path)

            elif suffix == ".json":
                if any(stem.startswith(p) for p in self.MARKET_JSON_STEMS):
                    buckets["market_json"].append(path)

        return buckets

    # ── Run ───────────────────────────────────────────────────────────────────

    def _safe_load_csv(self, fp: Path, nrows: int | None = None) -> pd.DataFrame:
        """
        Load a CSV, automatically sampling large files to avoid OOM errors.
        Files > _LARGE_FILE_THRESHOLD are sampled to _LARGE_FILE_SAMPLE rows.
        """
        size = fp.stat().st_size
        if size > self._LARGE_FILE_THRESHOLD:
            logger.warning(
                "Large file detected (%s, %.1f MB) — sampling %d rows.",
                fp.name, size / 1e6, self._LARGE_FILE_SAMPLE
            )
            # Read in chunks and take a representative head
            return pd.read_csv(fp, nrows=self._LARGE_FILE_SAMPLE, low_memory=False)
        return pd.read_csv(fp, nrows=nrows, low_memory=False)

    def run(self) -> list[dict]:
        """
        Execute the full pipeline and return all cleaned text documents.
        """
        buckets = self._discover_files()
        all_docs: list[dict] = []

        logger.info(
            "Discovered — Financial: %d | Transaction: %d | "
            "Market CSV: %d | Market JSON: %d | Stocks: %d | ETFs: %d",
            len(buckets["financial"]), len(buckets["transaction"]),
            len(buckets["market_csv"]), len(buckets["market_json"]),
            len(buckets["stock_csv"]),  len(buckets["etf_csv"]),
        )

        # Financial Statements
        for fp in buckets["financial"]:
            try:
                df = self.fs_cleaner.load_and_clean(fp)
                all_docs.extend(self.fs_cleaner.to_text_chunks(df))
            except Exception as e:
                logger.error("Failed to process %s: %s", fp.name, e)

        # Transaction Records (large files are auto-sampled)
        for fp in buckets["transaction"]:
            try:
                df = self.tx_cleaner.load_and_clean(fp)
                all_docs.extend(self.tx_cleaner.to_text_chunks(df))
            except Exception as e:
                logger.error("Failed to process %s: %s", fp.name, e)

        # Market CSVs
        for fp in buckets["market_csv"]:
            try:
                df = self.mkt_cleaner.load_csv(fp)
                all_docs.extend(self.mkt_cleaner.df_to_text_chunks(df))
            except Exception as e:
                logger.error("Failed to process %s: %s", fp.name, e)

        # Market JSONs
        for fp in buckets["market_json"]:
            try:
                records = self.mkt_cleaner.load_json(fp)
                all_docs.extend(self.mkt_cleaner.json_records_to_text_chunks(records))
            except Exception as e:
                logger.error("Failed to process %s: %s", fp.name, e)

        # Individual Stock CSVs (Stocks/ folder) — OHLCV per ticker
        stock_files = buckets["stock_csv"]
        if len(stock_files) > 50:
            logger.info("Discovered %d stock CSVs — sampling top 50 for indexing ...", len(stock_files))
            stock_files = stock_files[:50]
        else:
            logger.info("Processing %d stock CSVs ...", len(stock_files))

        for fp in stock_files:
            try:
                df = self._safe_load_csv(fp)
                df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                # Clean numeric columns
                for col in df.columns:
                    if any(k in col for k in ["open","high","low","close","volume","adj"]):
                        df[col] = df[col].apply(clean_currency)
                # Standardize date column
                for col in df.columns:
                    if "date" in col:
                        df[col] = df[col].apply(standardize_date)
                df = df.ffill().bfill()
                # Take recent snapshot to keep chunks relevant & fast
                if len(df) > 10:
                    df = df.tail(10)
                df["__source_file"]   = fp.name
                df["__data_category"] = "stock_ohlcv"
                df["__ticker"]        = fp.stem.upper()  # filename IS the ticker
                all_docs.extend(self.mkt_cleaner.df_to_text_chunks(df))
            except Exception as e:
                logger.error("Failed to process stock %s: %s", fp.name, e)

        # Individual ETF CSVs (ETFs/ folder)
        etf_files = buckets["etf_csv"]
        if len(etf_files) > 30:
            logger.info("Discovered %d ETF CSVs — sampling top 30 for indexing ...", len(etf_files))
            etf_files = etf_files[:30]
        else:
            logger.info("Processing %d ETF CSVs ...", len(etf_files))

        for fp in etf_files:
            try:
                df = self._safe_load_csv(fp)
                df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                for col in df.columns:
                    if any(k in col for k in ["open","high","low","close","volume","adj"]):
                        df[col] = df[col].apply(clean_currency)
                for col in df.columns:
                    if "date" in col:
                        df[col] = df[col].apply(standardize_date)
                df = df.ffill().bfill()
                if len(df) > 10:
                    df = df.tail(10)
                df["__source_file"]   = fp.name
                df["__data_category"] = "etf_ohlcv"
                df["__ticker"]        = fp.stem.upper()
                all_docs.extend(self.mkt_cleaner.df_to_text_chunks(df))
            except Exception as e:
                logger.error("Failed to process ETF %s: %s", fp.name, e)

        logger.info("Pipeline complete. Total documents produced: %d", len(all_docs))
        return all_docs


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ENTRY POINT (for local testing)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    pipeline = FinanceDataPipeline(data_dir=data_dir)
    docs = pipeline.run()
    print(f"\n✅  Produced {len(docs)} documents.")
    if docs:
        print("\n── Sample Document ──────────────────────────────────────────")
        print("TEXT:\n", docs[0]["text"][:400])
        print("METADATA:\n", docs[0]["metadata"])
