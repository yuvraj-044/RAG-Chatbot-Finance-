# FinSight — Architecture Overview

> **Finance RAG Chatbot** · Hackathon Project · 3-person team

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FINSIGHT ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐         ┌──────────────────────────────────────┐  │
│  │   Frontend    │  HTTPS  │          Backend API                 │  │
│  │  (React+Vite) │◄──────►│        (FastAPI + Python)            │  │
│  │   Vercel      │         │       Hugging Face Spaces            │  │
│  └──────────────┘         └──────────┬───────────────────────────┘  │
│                                      │                              │
│                            ┌─────────▼─────────┐                   │
│                            │  RAG Pipeline      │                   │
│                            │  (rag_chain.py)    │                   │
│                            └──┬──────────┬──────┘                   │
│                               │          │                          │
│                    ┌──────────▼──┐  ┌────▼──────────────┐          │
│                    │  ChromaDB   │  │   Groq LLM API    │          │
│                    │ Vector Store│  │ (llama-3.1-8b)    │          │
│                    └─────────────┘  └───────────────────┘          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow (per query)

1. **User types a question** in the React frontend (e.g., "What was Q3 revenue?")
2. **Frontend sends** `POST /chat` with `{session_id, message}` to the FastAPI backend
3. **Backend calls** `rag_query()` which:
   a. Embeds the query using Sentence Transformers
   b. Searches ChromaDB for the top-K most similar document chunks
   c. Constructs a grounded prompt with retrieved context
   d. Calls Groq's LLM API (llama-3.1-8b-instant) for generation
4. **Backend responds** with:
   - `reply` — the LLM-generated answer with inline citations
   - `sources` — structured array of retrieved chunks with similarity scores
   - `latency_ms` — end-to-end retrieval + generation time
   - `is_grounded` — whether the answer is backed by retrieved documents
5. **Frontend renders** the answer with typewriter animation, source inspection drawer,
   confidence badges, and latency indicators

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | React 19 + Vite | Chat UI, source citations, trust indicators |
| Styling | Vanilla CSS + CSS Custom Properties | Dark/light themes, glassmorphism, animations |
| Backend | FastAPI (Python 3.11) | REST API, session management, error handling |
| Embeddings | Sentence Transformers | Document chunking → vector embeddings |
| Vector DB | ChromaDB | Similarity search over embedded document chunks |
| LLM | Groq API (llama-3.1-8b-instant) | Grounded answer generation from retrieved context |
| Deployment | Vercel (frontend) + HF Spaces (backend) | Free-tier, zero-config hosting |

## Team Responsibilities

| Member | Owns | Key Files |
|--------|------|-----------|
| Member 1 | Data ingestion, embeddings, prompting | `ingest_and_clean.py`, `chunking.py`, `embeddings.py`, `rag_chain.py` |
| Member 2 | Vector DB, backend API, deployment | `app/main.py`, `app/api/routes.py`, `config.py`, `Dockerfile` |
| Member 3 | Frontend UI, source citations, demo polish | `frontend/src/**`, `docs/architecture.md`, `docs/demo_outline.md` |

## Key Design Decisions

- **Client-side typewriter effect**: Simulates streaming without requiring SSE on the backend.
  The full response arrives at once, but is revealed character-by-character for a polished feel.
- **Backward-compatible source parsing**: Frontend handles both structured `{doc_title, score}`
  objects and legacy flat strings `"file.csv | Ticker: X"`. This means the frontend works
  regardless of which version of the backend is deployed.
- **Mock API layer**: A built-in mock API (`src/api/mock.js`) returns realistic finance
  responses so the frontend can be developed and demoed without the real backend.
  Switch to real backend by setting `VITE_API_URL` environment variable — zero code changes.
- **No external component libraries**: Pure React + vanilla CSS keeps the bundle at ~67KB
  gzipped and eliminates dependency risk during a 6-hour build window.
