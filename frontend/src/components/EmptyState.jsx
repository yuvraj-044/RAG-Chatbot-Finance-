export default function EmptyState({ hasSources = false }) {
  return (
    <div className="empty-state">
      <span className="empty-state-icon">Search</span>
      <h4>{hasSources ? "Closest records need review" : "No relevant documents found"}</h4>
      <p>
        {hasSources
          ? "FinSight found nearby indexed records, but not enough verified evidence for a grounded answer. Open sources to inspect what matched."
          : "The query did not match indexed financial documents. Try a specific ticker, year, metric, or dataset category."}
      </p>
    </div>
  );
}
