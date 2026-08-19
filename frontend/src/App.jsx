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
  const sourceCount = messages.reduce(
    (count, message) => count + (message.sources?.length || 0),
    0
  );
  const answerCount = messages.filter((message) => message.role === "bot").length;

  return (
    <div className="app-container">
      {/* Error Banner */}
      <ErrorBanner message={error} onClose={clearError} onRetry={retryLast} />

      {/* Header */}
      <header className="app-header">
        <div className="app-logo">
          <div className="app-logo-icon" aria-hidden="true">F</div>
          <div>
            <span className="app-logo-text">FinSight</span>
            <span className="app-logo-subtitle">Finance RAG Copilot</span>
          </div>
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
              <span aria-hidden="true">+</span>
              New Chat
            </button>
          )}

          <ThemeToggle theme={theme} onToggle={toggleTheme} />
        </div>
      </header>

      {/* Main Content */}
      <main className="app-main">
        <aside className="context-panel" aria-label="Session context">
          <div className="context-panel-header">
            <span>Research Desk</span>
            <strong>{USE_REAL_API ? "Live API" : "Demo"}</strong>
          </div>
          <div className="metric-grid">
            <div className="metric-card">
              <span>Answers</span>
              <strong>{answerCount}</strong>
            </div>
            <div className="metric-card">
              <span>Sources</span>
              <strong>{sourceCount}</strong>
            </div>
          </div>
          <div className="desk-note">
            Ask about filings, ratios, market context, fraud signals, or balance
            sheet comparisons. Answers can open retrieved evidence when sources
            are available.
          </div>
        </aside>

        <section className="conversation-shell" aria-label="Chat conversation">
          {showStarter ? (
            <StarterChips onSelect={sendMessage} />
          ) : (
            <ChatWindow
              messages={messages}
              isLoading={isLoading}
              onViewSources={openDrawer}
            />
          )}
        </section>
      </main>

      {/* Input Bar */}
      <InputBar onSend={sendMessage} disabled={isLoading} />

      {/* Source Drawer */}
      {drawerSources && (
        <SourceDrawer sources={drawerSources} onClose={closeDrawer} />
      )}
    </div>
  );
}
