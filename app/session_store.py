"""
app/session_store.py
--------------------
Simple in-memory store for per-session conversation history.

Why in-memory? For a hackathon, a dict is perfectly fine. There's no persistence
across server restarts, but that's acceptable for a live demo.

Structure: { session_id: [{"role": "user"|"assistant", "content": "..."}] }
"""

from collections import defaultdict
import config

# The main store — a dict of session_id → list of message dicts.
# defaultdict(list) means accessing a new session_id auto-creates an empty list.
_store: dict[str, list[dict]] = defaultdict(list)


def get_history(session_id: str) -> list[dict]:
    """
    Returns the conversation history for a session as a list of message dicts.
    Returns an empty list if the session doesn't exist yet.
    """
    return list(_store[session_id])  # return a copy so callers can't mutate the store


def add_turn(session_id: str, user_message: str, assistant_reply: str) -> None:
    """
    Appends a user→assistant exchange to the session history.
    Also trims history to MAX_HISTORY_TURNS to avoid sending huge contexts to the LLM.
    Each "turn" = 1 user message + 1 assistant message = 2 entries.
    """
    _store[session_id].append({"role": "user", "content": user_message})
    _store[session_id].append({"role": "assistant", "content": assistant_reply})

    # Keep only the last N turns (N*2 messages) to cap memory and token usage.
    max_messages = config.MAX_HISTORY_TURNS * 2
    if len(_store[session_id]) > max_messages:
        _store[session_id] = _store[session_id][-max_messages:]


def reset_session(session_id: str) -> None:
    """
    Clears all history for a given session.
    Called by POST /chat/reset.
    """
    _store[session_id] = []


def session_exists(session_id: str) -> bool:
    """Returns True if a session has any history at all."""
    return session_id in _store and len(_store[session_id]) > 0


def get_all_sessions() -> list[str]:
    """Returns list of all active session IDs. Useful for debugging."""
    return list(_store.keys())
