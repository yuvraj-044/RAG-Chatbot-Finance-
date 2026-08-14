// ── Real API Client ─────────────────────────────────────────────────────
// Talks to the FastAPI backend. Set VITE_API_URL in .env to point at your
// deployed backend (e.g., https://your-space.hf.space).

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const TIMEOUT_MS = 30000;

function parseSource(source) {
  // Handle structured source objects (enhanced API)
  if (typeof source === "object" && source !== null) {
    return {
      doc_title: source.doc_title || source.source_file || "Unknown Document",
      chunk_text: source.chunk_text || source.text || "",
      score: source.score ?? source.relevance_score ?? 0,
      ticker: source.ticker || "",
      date: source.date || "",
    };
  }

  // Handle flat string format: "file.csv | Ticker: X | Date: Y"
  if (typeof source === "string") {
    const parts = source.split(" | ");
    const doc_title = parts[0] || "Unknown Document";
    let ticker = "";
    let date = "";
    for (const part of parts.slice(1)) {
      if (part.startsWith("Ticker: ")) ticker = part.replace("Ticker: ", "");
      if (part.startsWith("Date: ")) date = part.replace("Date: ", "");
    }
    return {
      doc_title,
      chunk_text: source,
      score: 0.8, // Default when not provided
      ticker,
      date,
    };
  }

  return {
    doc_title: "Unknown Document",
    chunk_text: String(source),
    score: 0,
    ticker: "",
    date: "",
  };
}

export async function apiChat(sessionId, message) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);
  const start = performance.now();

  try {
    const res = await fetch(`${BASE_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);
    const elapsed = Math.round(performance.now() - start);

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(
        errorData.detail ||
          `Server error (${res.status}). Please try again.`
      );
    }

    const data = await res.json();

    // Normalize response — handle both legacy and enhanced formats
    const sources = (data.sources || []).map(parseSource);

    return {
      session_id: data.session_id || sessionId,
      reply: data.reply || data.answer || "",
      sources,
      latency_ms: data.latency_ms || elapsed,
      is_grounded: data.is_grounded ?? sources.length > 0,
    };
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === "AbortError") {
      throw new Error(
        "Request timed out. The AI is taking too long — please try a shorter question."
      );
    }
    throw err;
  }
}

export async function apiHealth() {
  try {
    const res = await fetch(`${BASE_URL}/health`, { method: "GET" });
    if (!res.ok) throw new Error("Health check failed");
    return await res.json();
  } catch {
    return { status: "error", stub_mode: false };
  }
}

export async function apiReset(sessionId) {
  const res = await fetch(`${BASE_URL}/chat/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) throw new Error("Failed to reset session");
  return await res.json();
}
