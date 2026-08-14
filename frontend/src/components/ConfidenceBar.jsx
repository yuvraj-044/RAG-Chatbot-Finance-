export default function ConfidenceBar({ score }) {
  if (score == null || score === 0) return null;

  const pct = Math.round(score * 100);
  const level = score >= 0.8 ? "high" : score >= 0.5 ? "medium" : "low";
  const label =
    level === "high"
      ? "High confidence"
      : level === "medium"
        ? "Medium confidence"
        : "Low confidence";

  return (
    <span
      className={`confidence-inline ${level}`}
      title={`${label}: ${pct}% match`}
    >
      {level === "high" ? "✓" : level === "medium" ? "~" : "!"} {pct}%
    </span>
  );
}
