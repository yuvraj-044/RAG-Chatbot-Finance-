import { useState, useRef, useEffect } from "react";

export default function InputBar({ onSend, disabled }) {
  const [value, setValue] = useState("");
  const inputRef = useRef(null);

  // Auto-focus on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Re-focus after sending
  useEffect(() => {
    if (!disabled) {
      inputRef.current?.focus();
    }
  }, [disabled]);

  function handleSubmit(e) {
    e.preventDefault();
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      handleSubmit(e);
    }
  }

  return (
    <div className="input-bar-container">
      <form className="input-bar" onSubmit={handleSubmit}>
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about financial data, reports, or compliance..."
          disabled={disabled}
          autoComplete="off"
          aria-label="Chat input"
        />
        <button
          type="submit"
          className="send-button"
          disabled={!value.trim() || disabled}
          aria-label="Send message"
        >
          ↑
        </button>
      </form>
      <div className="input-hint">
        Press Enter to send · FinSight may produce inaccurate information
      </div>
    </div>
  );
}
