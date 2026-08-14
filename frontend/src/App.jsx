import { useState, useCallback, useEffect } from "react";
import { useChat } from "./hooks/useChat.js";
import { health, USE_REAL_API } from "./api/index.js";
import ThemeToggle from "./components/ThemeToggle.jsx";
import ChatWindow from "./components/ChatWindow.jsx";
import InputBar from "./components/InputBar.jsx";
import StarterChips from "./components/StarterChips.jsx";
import SourceDrawer from "./components/SourceDrawer.jsx";
import ErrorBanner from "./components/ErrorBanner.jsx";

export default function App() {
  // ── Theme ────────────────────────────────────────────────────────────
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem("finsight-theme") || "dark";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("finsight-theme", theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((t) => (t === "dark" ? "light" : "dark"));
  }, []);

  // ── Chat state ───────────────────────────────────────────────────────
  const {
    messages,
    isLoading,
    error,
    sendMessage,
    clearError,
    resetSession,
    retryLast,
  } = useChat();

  // ── Source drawer ────────────────────────────────────────────────────
  const [drawerSources, setDrawerSources] = useState(null);

  const openDrawer = useCallback((sources) => {
    setDrawerSources(sources);
  }, []);

  const closeDrawer = useCallback(() => {
    setDrawerSources(null);
  }, []);

  // ── Backend health ───────────────────────────────────────────────────
  const [backendStatus, setBackendStatus] = useState("checking");

  useEffect(() => {
    health()
      .then((data) => {
        setBackendStatus(data.status === "ok" ? "online" : "offline");
      })
      .catch(() => {
        setBackendStatus("offline");
      });
  }, []);

  // ── Determine if we should show starter chips (empty state) ──────────
  const showStarter = messages.length === 0;

  return (
    <div className="app-container">
      {/* Error Banner */}
      <ErrorBanner message={error} onClose={clearError} onRetry={retryLast} />

      {/* Header */}
      <header className="app-header">
        <div className="app-logo">
          <div className="app-logo-icon">🏦</div>
          <span className="app-logo-text">FinSight</span>
          <span className="app-logo-subtitle">AI Research Assistant</span>
        </div>

        <div className="header-actions">
          <div className="header-status">
            <span
              className={`status-dot ${backendStatus === "online" ? "" : "offline"}`}
            />
            {backendStatus === "checking"
              ? "Connecting..."
              : backendStatus === "online"
                ? USE_REAL_API
                  ? "API Connected"
                  : "Demo Mode"
                : "Offline"}
          </div>

          {messages.length > 0 && (
            <button className="new-session-btn" onClick={resetSession}>
              🗑️ New Chat
            </button>
          )}

          <ThemeToggle theme={theme} onToggle={toggleTheme} />
        </div>
      </header>

      {/* Main Content */}
      {showStarter ? (
        <StarterChips onSelect={sendMessage} />
      ) : (
        <ChatWindow
          messages={messages}
          isLoading={isLoading}
          onViewSources={openDrawer}
        />
      )}

      {/* Input Bar */}
      <InputBar onSend={sendMessage} disabled={isLoading} />

      {/* Source Drawer */}
      {drawerSources && (
        <SourceDrawer sources={drawerSources} onClose={closeDrawer} />
      )}
    </div>
  );
}
