export default function EmptyState() {
  return (
    <div className="empty-state">
      <span className="empty-state-icon">🔍</span>
      <h4>No relevant documents found</h4>
      <p>
        The query didn't match any indexed financial documents. Try asking
        about specific tickers, dates, or financial metrics.
      </p>
    </div>
  );
}
