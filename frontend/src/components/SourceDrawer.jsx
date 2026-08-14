import { useState, useEffect, useCallback } from "react";

export default function SourceDrawer({ sources, onClose }) {
  const [isClosing, setIsClosing] = useState(false);

  const handleClose = useCallback(() => {
    setIsClosing(true);
    setTimeout(onClose, 250); // Match animation duration
  }, [onClose]);

  // Close on Escape key
  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === "Escape") handleClose();
    };
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [handleClose]);

  if (!sources || sources.length === 0) return null;

  function scoreLevel(score) {
    if (score >= 0.8) return "high";
    if (score >= 0.5) return "medium";
    return "low";
  }

  return (
    <>
      <div className="drawer-overlay" onClick={handleClose} />
      <aside className={`source-drawer ${isClosing ? "closing" : ""}`}>
        <div className="drawer-header">
          <h3>
            📄 Retrieved Sources
            <span
              style={{
                fontSize: "var(--font-size-xs)",
                color: "var(--text-tertiary)",
                fontWeight: 400,
              }}
            >
              ({sources.length})
            </span>
          </h3>
          <button
            className="drawer-close"
            onClick={handleClose}
            aria-label="Close drawer"
          >
            ✕
          </button>
        </div>

        <div className="drawer-body">
          {sources.map((source, index) => {
            const level = scoreLevel(source.score);
            const pct = Math.round(source.score * 100);

            return (
              <div
                className="source-card"
                key={index}
                style={{ animationDelay: `${index * 80}ms` }}
              >
                <div className="source-card-header">
                  <span className="source-card-title">
                    📄 {source.doc_title}
                  </span>
                  {source.date && (
                    <span className="source-card-date">{source.date}</span>
                  )}
                </div>

                {source.ticker && (
                  <span
                    style={{
                      fontSize: "var(--font-size-xs)",
                      color: "var(--accent-secondary)",
                      fontWeight: 500,
                    }}
                  >
                    Ticker: {source.ticker}
                  </span>
                )}

                <div className="source-card-chunk">{source.chunk_text}</div>

                <div className="source-card-score">
                  <span className="score-label">Similarity</span>
                  <div className="score-bar-track">
                    <div
                      className={`score-bar-fill ${level}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className={`score-value ${level}`}>{pct}%</span>
                </div>
              </div>
            );
          })}
        </div>
      </aside>
    </>
  );
}
