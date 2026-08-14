# API Contract — Handoff to Member 2 (Backend)

> **From:** Member 3 (Frontend)
> **Date:** 2026-08-14
> **Status:** Frontend is built and tested against this contract.
> I've already updated `app/api/routes.py` to implement this.
> Please review and merge.

---

## Endpoints

### `POST /chat` — Main Chat Endpoint

**Request:**
```json
{
  "session_id": "uuid-string",
  "message": "What was Q3 revenue for AAPL?"
}
```

**Response:**
```json
{
  "session_id": "uuid-string",
  "reply": "Based on the Q3 2024 earnings report, AAPL revenue was $85.8B...",
  "sources": [
    {
      "doc_title": "LQ_AAPL_Q3_2024.csv",
      "chunk_text": "Total Revenue: $85,777M | Net Income: $21,448M...",
      "score": 0.92,
      "ticker": "AAPL",
      "date": "2024-09-30"
    }
  ],
  "latency_ms": 180,
  "is_grounded": true
}
```

**Error codes:**
- `422` — Invalid request body (missing/wrong-type fields)
- `503` — LLM/retrieval pipeline timeout or unavailable
- `500` — Unexpected internal error (detail message included)

**Notes:**
- `sources` can be empty `[]` — frontend handles this with an "empty state" UI
- `is_grounded` should be `false` when no relevant chunks were retrieved
- `latency_ms` is measured server-side (`time.perf_counter()`)
- The flat string source format (`"file.csv | Ticker: X | Date: Y"`) still works —
  `routes.py` parses it into the structured format automatically

---

### `POST /chat/reset` — Clear Session History

```json
// Request
{ "session_id": "uuid-string" }
// Response
{ "session_id": "uuid-string", "status": "reset" }
```

---

### `GET /health` — Liveness Check

```json
{ "status": "ok", "stub_mode": true }
```

---

## CORS

Already configured in `app/main.py` with `allow_origins=["*"]`.
For production, change to the actual Vercel frontend URL.

## Frontend URL

Once deployed, I'll share the Vercel URL.
Set this as the allowed origin in CORS config.

For local dev, frontend runs on `http://localhost:5173`.
