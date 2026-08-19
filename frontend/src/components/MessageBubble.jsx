import { useTypewriter } from "../hooks/useTypewriter.js";
import SourceChip from "./SourceChip.jsx";
import LatencyBadge from "./LatencyBadge.jsx";
import ConfidenceBar from "./ConfidenceBar.jsx";
import EmptyState from "./EmptyState.jsx";

export default function MessageBubble({ message, isLatest, onViewSources }) {
  const isUser = message.role === "user";
  const shouldAnimate = isLatest && !isUser;

  const { displayedText, isTyping } = useTypewriter(
    message.content,
    shouldAnimate
  );

  // Compute average confidence from sources
  const avgScore =
    message.sources && message.sources.length > 0
      ? message.sources.reduce((sum, s) => sum + (s.score || 0), 0) /
        message.sources.length
      : null;

  return (
    <div className={`message-row ${isUser ? "user" : "bot"}`}>
      <div className={`message-avatar ${isUser ? "user" : "bot"}`}>
        {isUser ? "You" : "AI"}
      </div>

      <div className="message-content">
        <div className={`message-bubble ${isUser ? "user" : "bot"}`}>
          {renderFormattedText(isUser ? message.content : displayedText)}
          {isTyping && <span className="typing-cursor" />}
        </div>

        {/* Meta info — only on bot messages after typing completes */}
        {!isUser && !isTyping && (
          <>
            <div className="message-meta">
              {message.sources && message.sources.length > 0 && (
                <SourceChip
                  count={message.sources.length}
                  onClick={() => onViewSources(message.sources)}
                />
              )}
              <LatencyBadge latencyMs={message.latency_ms} />
              {avgScore !== null && <ConfidenceBar score={avgScore} />}
              {message.is_grounded !== undefined && (
                <span
                  className={`grounded-badge ${message.is_grounded ? "grounded" : "ungrounded"}`}
                >
                  {message.is_grounded ? "Grounded" : "Unverified"}
                </span>
              )}
            </div>

            {/* Show empty state when bot has no sources */}
            {message.is_grounded === false && (
              <EmptyState hasSources={(message.sources?.length || 0) > 0} />
            )}
          </>
        )}
      </div>
    </div>
  );
}

/**
 * Simple markdown-like renderer for bold (**text**), bullet points, and tables.
 * Avoids pulling in a full markdown library to keep bundle size minimal.
 */
function renderFormattedText(text) {
  if (!text) return null;

  const lines = text.split("\n");
  const elements = [];
  let inTable = false;
  let tableRows = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Table detection
    if (line.includes("|") && line.trim().startsWith("|")) {
      if (!inTable) {
        inTable = true;
        tableRows = [];
      }
      // Skip separator rows (|---|---|)
      if (!/^\|[\s\-|:]+\|$/.test(line.trim())) {
        tableRows.push(line);
      }
      continue;
    } else if (inTable) {
      // End of table
      elements.push(renderTable(tableRows, `table-${i}`));
      inTable = false;
      tableRows = [];
    }

    // Headers (### h3, ## h2)
    if (line.startsWith("### ")) {
      elements.push(
        <h4 key={i} style={{ margin: "8px 0 4px", fontSize: "0.9em", fontWeight: 600 }}>
          {formatInline(line.slice(4))}
        </h4>
      );
      continue;
    }

    if (line.startsWith("## ")) {
      elements.push(
        <h3 key={i} style={{ margin: "8px 0 4px", fontSize: "1em", fontWeight: 600 }}>
          {formatInline(line.slice(3))}
        </h3>
      );
      continue;
    }

    // Bullet points
    if (line.match(/^[•\-\*] /)) {
      elements.push(
        <div key={i} style={{ paddingLeft: "12px", display: "flex", gap: "6px" }}>
          <span style={{ color: "var(--accent-primary)" }}>•</span>
          <span>{formatInline(line.slice(2))}</span>
        </div>
      );
      continue;
    }

    // Numbered lists
    if (line.match(/^\d+\. /)) {
      const match = line.match(/^(\d+)\. (.*)/);
      elements.push(
        <div key={i} style={{ paddingLeft: "12px", display: "flex", gap: "6px" }}>
          <span style={{ color: "var(--accent-primary)", fontWeight: 600, minWidth: "18px" }}>
            {match[1]}.
          </span>
          <span>{formatInline(match[2])}</span>
        </div>
      );
      continue;
    }

    // Empty lines
    if (line.trim() === "") {
      elements.push(<br key={i} />);
      continue;
    }

    // Normal paragraph
    elements.push(
      <span key={i} style={{ display: "block" }}>
        {formatInline(line)}
      </span>
    );
  }

  // Flush remaining table
  if (inTable && tableRows.length > 0) {
    elements.push(renderTable(tableRows, "table-end"));
  }

  return <>{elements}</>;
}

function formatInline(text) {
  // Bold: **text**
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={i} style={{ fontWeight: 600, color: "var(--text-primary)" }}>
          {part.slice(2, -2)}
        </strong>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

function renderTable(rows, key) {
  if (rows.length === 0) return null;

  const parseRow = (row) =>
    row
      .split("|")
      .filter((cell) => cell.trim() !== "")
      .map((cell) => cell.trim());

  const headerCells = parseRow(rows[0]);
  const bodyRows = rows.slice(1).map(parseRow);

  return (
    <div
      key={key}
      style={{
        overflowX: "auto",
        margin: "8px 0",
        borderRadius: "var(--radius-sm)",
        border: "1px solid var(--border-primary)",
      }}
    >
      <table
        style={{
          width: "100%",
          borderCollapse: "collapse",
          fontSize: "var(--font-size-xs)",
          fontFamily: "var(--font-mono)",
        }}
      >
        <thead>
          <tr>
            {headerCells.map((cell, j) => (
              <th
                key={j}
                style={{
                  padding: "6px 10px",
                  textAlign: "left",
                  background: "var(--bg-tertiary)",
                  borderBottom: "1px solid var(--border-primary)",
                  fontWeight: 600,
                  whiteSpace: "nowrap",
                }}
              >
                {formatInline(cell)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {bodyRows.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => (
                <td
                  key={j}
                  style={{
                    padding: "6px 10px",
                    borderBottom: "1px solid var(--border-secondary)",
                    whiteSpace: "nowrap",
                  }}
                >
                  {formatInline(cell)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
