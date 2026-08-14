export default function LatencyBadge({ latencyMs }) {
  if (!latencyMs) return null;

  return (
    <span className="latency-badge" title="Backend retrieval + generation time">
      <span className="latency-icon">⚡</span>
      {latencyMs}ms
    </span>
  );
}
