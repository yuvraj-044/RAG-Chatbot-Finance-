# FinSight — Demo Deck Outline (5 Slides, 3 Minutes)

> **Live judge walkthrough**: Problem 30s → Architecture 60s → Live Demo 60s → Q&A 30s

---

## Slide 1: The Problem (30 seconds)

**Title:** "Financial Data is Scattered, Inaccessible, and Untrustworthy"

**Talking points:**
- Financial analysts spend 40%+ of their time *finding* information across spreadsheets,
  SEC filings, market data feeds, and internal reports
- Existing search tools return documents, not answers — users must still read and synthesize
- When AI chatbots do give answers, there's no way to verify the source or check accuracy
- **Our question:** Can we build a chatbot that gives *grounded, citable, trustworthy*
  answers from real financial data?

---

## Slide 2: Architecture (60 seconds)

**Title:** "FinSight: RAG-Powered Finance Assistant"

**Visual:** System architecture diagram (from `docs/architecture.md`)

**Talking points:**
- **Retrieval-Augmented Generation (RAG)** — every answer is grounded in real documents
- **3-stage pipeline:**
  1. Member 1: Data ingestion → chunking → vector embeddings (Sentence Transformers)
  2. Member 2: ChromaDB vector search → FastAPI backend → Groq LLM generation
  3. Member 3: React UI with source citations, confidence scoring, and streaming effect
- **Trust by design:** Every answer shows which documents it came from, with similarity
  scores so users can verify claims
- **Performance:** Sub-200ms retrieval, instant UI feedback via typewriter streaming

---

## Slide 3: Live Demo — Normal Query (30 seconds)

**Title:** "Let's Ask FinSight a Question"

**Demo script:**
1. Open the app → show the clean landing page with starter chips
2. Click **"What was Q3 revenue for AAPL?"** starter chip
3. Watch the typewriter animation render the answer
4. Point out: **latency badge** (e.g., "142ms"), **confidence score** (e.g., "92%"),
   **grounded badge** ("✓ Grounded")
5. Click **"View Sources (2)"** → show the source drawer with:
   - Document title, raw chunk text, similarity score bar
   - Explain: "This is the exact text from our indexed dataset that the AI used"

---

## Slide 4: Live Demo — Edge Cases (30 seconds)

**Title:** "What Happens When Things Go Wrong?"

**Demo script:**
1. Type an out-of-scope question: **"What's the weather today?"**
   → Show the graceful "No relevant documents found" empty state
   → Point out: the AI doesn't hallucinate — it tells you it has no data
2. Toggle the theme: Dark → Light mode (1 second)
3. Resize to mobile (if possible): "Works on judges' phones too"
4. *Optional:* Show the "New Chat" button to reset session

**Key message:** "We handle every edge case — the user never sees a blank screen or a crash"

---

## Slide 5: Q&A / Wrap-up (30 seconds)

**Title:** "Key Takeaways"

**Summary bullets:**
- ✅ RAG ensures answers are **grounded in real financial data**, not hallucinated
- ✅ Source citations let users **verify every claim** with one click
- ✅ Confidence scoring provides **transparency** about answer quality
- ✅ Built in 6 hours with zero external UI libraries — **minimal, stable, deployable**

**Pre-prepared Q&A answers:**

| Expected Question | Answer |
|---|---|
| "How do you handle hallucination?" | "RAG constrains the LLM to only use retrieved document chunks. The `is_grounded` flag detects when no relevant documents were found, and we show a clear 'no data' state instead of guessing." |
| "What data sources does it support?" | "CSVs of financial statements, market indices, SEC/SEBI filings. The ingestion pipeline handles multiple formats and normalizes them for embedding." |
| "How does confidence scoring work?" | "Each retrieved chunk has a cosine similarity score from ChromaDB. We average across sources and display as a percentage — green (>80%), yellow (>50%), red (<50%)." |
| "Can this scale beyond a hackathon?" | "Yes — ChromaDB can be swapped for Pinecone/Weaviate, the LLM can be upgraded to GPT-4/Claude, and the frontend is production-ready React deployed on Vercel." |
