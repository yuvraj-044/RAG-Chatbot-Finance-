export default function LoadingSkeleton() {
  return (
    <div className="skeleton-message">
      <div className="skeleton-avatar" />
      <div className="skeleton-content">
        <div className="skeleton-line" />
        <div className="skeleton-line" />
        <div className="skeleton-line" />
      </div>
    </div>
  );
}
