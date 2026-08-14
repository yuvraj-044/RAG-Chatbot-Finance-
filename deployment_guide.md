# Finance RAG API — Deployment Guide
## Hugging Face Spaces (FastAPI + Docker)

---

### Prerequisites
- A [Hugging Face account](https://huggingface.co/join) (free)
- Your repo pushed to GitHub (or you can push directly to HF)
- A Groq API key from [console.groq.com](https://console.groq.com/keys)

---

## Step 1 — Create a new Hugging Face Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in:
   - **Space name**: `finance-rag-api` (or anything you like)
   - **License**: MIT
   - **SDK**: select **Docker**
   - **Visibility**: Public (or Private if you have HF Pro)
3. Click **Create Space**

---

## Step 2 — Set your secrets (NOT in code)

> ⚠️ Never upload a `.env` file. Use HF Space Secrets instead.

1. In your Space, go to **Settings → Variables and Secrets**
2. Click **New Secret** for each of the following:

| Secret Name | Value |
|---|---|
| `GROQ_API_KEY` | Your Groq API key (`gsk_...`) |
| `GROQ_MODEL` | `llama-3.1-8b-instant` |
| `CHROMA_COLLECTION_NAME` | `finance_docs` (must match your teammate's collection name) |
| `CHROMA_PERSIST_DIR` | `./chroma_db` |

> The other variables (`TOP_K_RESULTS`, `LLM_TIMEOUT_SECONDS`, `MAX_HISTORY_TURNS`) have sensible defaults — only add them as secrets if you want to override.

---

## Step 3 — Push your code to the Space

HF Spaces is a Git repo. Push directly:

```bash
# Clone your new Space's git repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/finance-rag-api
cd finance-rag-api

# Copy your project files in
cp -r /path/to/your/Backend/* .

# Make sure .gitignore exists (see below) and .env is NOT staged
git status   # verify .env is not listed

git add .
git commit -m "Initial deploy"
git push
```

HF Spaces automatically detects the `Dockerfile` and starts a build. Watch build logs in the Space's **Logs** tab.

---

## Step 4 — Include the ChromaDB index

If your teammate has a pre-built ChromaDB index:

```bash
# Copy the chroma_db folder into your repo before pushing
cp -r /path/to/chroma_db ./chroma_db
git add chroma_db/
git commit -m "Add pre-built ChromaDB index"
git push
```

> If the index is large, consider using [HF Datasets](https://huggingface.co/docs/datasets) to host it separately and download it at startup. For a hackathon, committing it directly is fastest.

---

## Step 5 — Verify the deployment

Once the build succeeds (green indicator on the Space):

```bash
# Your API base URL will be:
# https://YOUR_USERNAME-finance-rag-api.hf.space

# Test the health endpoint
curl https://YOUR_USERNAME-finance-rag-api.hf.space/health

# Expected: {"status":"ok","stub_mode":true}

# Test a chat message
curl -X POST https://YOUR_USERNAME-finance-rag-api.hf.space/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-1", "message": "What is EBITDA?"}'

# Browse the auto-generated API docs:
# https://YOUR_USERNAME-finance-rag-api.hf.space/docs
```

---

## Give your frontend teammate this URL

Share the base URL and the API contract:

```
Base URL:  https://YOUR_USERNAME-finance-rag-api.hf.space

POST /chat
  Body:     { "session_id": "abc123", "message": "What is P/E ratio?" }
  Response: { "session_id": "abc123", "reply": "...", "sources": ["doc.pdf — page 3"] }

POST /chat/reset
  Body:     { "session_id": "abc123" }
  Response: { "session_id": "abc123", "status": "reset" }

GET /health
  Response: { "status": "ok", "stub_mode": true }

Full docs: https://YOUR_USERNAME-finance-rag-api.hf.space/docs
```

---

## Common Issues

| Problem | Fix |
|---|---|
| Build fails with `chromadb` error | The `build-essential` apt package in the Dockerfile handles this — check the Logs tab for the exact error |
| `GROQ_API_KEY is not set` in logs | You forgot to add it as a Space Secret in Step 2 |
| 503 on `/chat` | Check that `CHROMA_COLLECTION_NAME` matches what your teammate used |
| `/health` returns `stub_mode: true` | This is expected until your teammate wires in the real pipeline and sets `USE_STUB = False` in `app/rag_stub.py` |
| Space keeps restarting | Free-tier Spaces sleep after inactivity. Hit `/health` to wake it up |

---

## Local Development

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and fill in your secrets
cp .env.example .env
# Edit .env with your actual GROQ_API_KEY

# 4. Run the server
uvicorn app.main:app --reload --port 8000

# 5. Open the docs
open http://localhost:8000/docs
```
