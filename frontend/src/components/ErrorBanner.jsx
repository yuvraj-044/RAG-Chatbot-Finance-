import { useEffect, useRef } from "react";

export default function ErrorBanner({ message, onClose, onRetry }) {
  const timerRef = useRef(null);

  useEffect(() => {
    // Auto-dismiss after 8 seconds
    timerRef.current = setTimeout(() => {
      onClose();
    }, 8000);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [message, onClose]);

  if (!message) return null;

  return (
    <div className="error-banner animate-slide-down" role="alert">
      <span className="error-banner-icon">⚠️</span>
      <span className="error-banner-text">{message}</span>
      {onRetry && (
        <button className="error-banner-retry" onClick={onRetry}>
          Retry
        </button>
      )}
      <button
        className="error-banner-close"
        onClick={onClose}
        aria-label="Dismiss error"
      >
        ✕
      </button>
    </div>
  );
}
