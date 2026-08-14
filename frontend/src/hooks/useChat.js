import { useState, useCallback, useRef } from "react";
import { chat, reset } from "../api/index.js";

function generateSessionId() {
  return "session-" + Math.random().toString(36).substring(2, 10);
}

export function useChat() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const sessionIdRef = useRef(generateSessionId());

  const sendMessage = useCallback(async (text) => {
    const userMsg = {
      id: Date.now().toString(),
      role: "user",
      content: text,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);
    setError(null);

    try {
      const response = await chat(sessionIdRef.current, text);

      const botMsg = {
        id: (Date.now() + 1).toString(),
        role: "bot",
        content: response.reply,
        sources: response.sources || [],
        latency_ms: response.latency_ms || 0,
        is_grounded: response.is_grounded ?? true,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      setError(err.message || "An unexpected error occurred. Please try again.");
    } finally {
      setIsLoading(false);
    }
  }, []);

  const clearError = useCallback(() => setError(null), []);

  const resetSession = useCallback(async () => {
    try {
      await reset(sessionIdRef.current);
    } catch {
      // Swallow reset errors — not critical
    }
    sessionIdRef.current = generateSessionId();
    setMessages([]);
    setError(null);
    setIsLoading(false);
  }, []);

  const retryLast = useCallback(() => {
    setError(null);
    const lastUserMsg = [...messages].reverse().find((m) => m.role === "user");
    if (lastUserMsg) {
      // Remove the last user message and re-send
      setMessages((prev) => prev.slice(0, -1));
      sendMessage(lastUserMsg.content);
    }
  }, [messages, sendMessage]);

  return {
    messages,
    isLoading,
    error,
    sendMessage,
    clearError,
    resetSession,
    retryLast,
    sessionId: sessionIdRef.current,
  };
}
