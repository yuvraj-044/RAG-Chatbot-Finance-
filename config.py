"""
config.py
---------
Single source of truth for all configuration values.
Everything is read from environment variables so we never hardcode secrets.
python-dotenv loads the .env file automatically in local dev.
On Hugging Face Spaces, you set these as Space Secrets (Settings → Variables and Secrets).
"""

import os
from dotenv import load_dotenv

# Load .env file when running locally. On HF Spaces this is a no-op
# because secrets are injected as real environment variables.
load_dotenv()


# ---------------------------------------------------------------------------
# Groq LLM settings
# ---------------------------------------------------------------------------
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
# Which Groq model to use. llama-3.1-8b-instant is fast and free-tier friendly.
# Swap to "llama-3.1-70b-versatile" for better quality if your quota allows.
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# ---------------------------------------------------------------------------
# ChromaDB settings
# ---------------------------------------------------------------------------
# Path to the ChromaDB persistence directory (your teammate created this during ingestion).
# On HF Spaces, this should be a relative path inside your repo, e.g. "./chroma_db"
CHROMA_PERSIST_DIR: str = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
# The name of the collection your teammate created during embedding/ingestion.
CHROMA_COLLECTION_NAME: str = os.environ.get("CHROMA_COLLECTION_NAME", "finance_docs")

# ---------------------------------------------------------------------------
# API behaviour
# ---------------------------------------------------------------------------
# How many retrieved chunks to pass to the LLM as context.
TOP_K_RESULTS: int = int(os.environ.get("TOP_K_RESULTS", "5"))
# Request timeout in seconds for the Groq API call.
LLM_TIMEOUT_SECONDS: int = int(os.environ.get("LLM_TIMEOUT_SECONDS", "30"))
# Maximum conversation turns to keep in memory per session (older turns are dropped).
MAX_HISTORY_TURNS: int = int(os.environ.get("MAX_HISTORY_TURNS", "10"))


def validate_config() -> None:
    """
    Call this at startup to catch missing secrets early.
    Raises ValueError if any required variable is missing.
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Add it to your .env file locally, or as a Space Secret on Hugging Face."
        )
