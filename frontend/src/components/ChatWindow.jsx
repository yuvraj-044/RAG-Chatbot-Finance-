import { useEffect, useRef } from "react";
import MessageBubble from "./MessageBubble.jsx";
import LoadingSkeleton from "./LoadingSkeleton.jsx";

export default function ChatWindow({
  messages,
  isLoading,
  onViewSources,
}) {
  const bottomRef = useRef(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  return (
    <div className="chat-window">
      {messages.map((msg, index) => (
        <MessageBubble
          key={msg.id}
          message={msg}
          isLatest={index === messages.length - 1}
          onViewSources={onViewSources}
        />
      ))}

      {isLoading && <LoadingSkeleton />}

      <div ref={bottomRef} />
    </div>
  );
}
