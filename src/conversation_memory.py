"""SQLite-backed conversation history with bounded recent turns and summaries."""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List

try:
    import tiktoken
except ImportError:  # SQLite memory remains usable before optional deps are installed.
    tiktoken = None

try:
    from langchain_core.chat_history import BaseChatMessageHistory
except ImportError:  # Lets database-only tests run without installing the app requirements.
    class BaseChatMessageHistory:  # type: ignore[no-redef]
        pass


class ConversationMemory:
    """Persist all messages; return only bounded history to the LLM."""

    def __init__(self, db_path: str = "data/chat_history.db", recent_turns: int = 5,
                 summary_trigger_turns: int = 5, summary_token_limit: int = 250):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.recent_turns = max(1, int(recent_turns))
        self.summary_trigger_turns = max(2, int(summary_trigger_turns))
        self.summary_token_limit = max(50, int(summary_token_limit))
        self.encoding = tiktoken.get_encoding("cl100k_base") if tiktoken else None
        self._initialize()

    @contextmanager
    def _connect(self):
        """Yield a transactional SQLite connection and always close it."""
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _initialize(self):
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS conversations (
                session_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL DEFAULT '',
                summarized_through_id INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES conversations(session_id)
            )""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id, id)")

    def ensure_session(self, session_id: str):
        if not session_id or not session_id.strip():
            raise ValueError("session_id must not be empty")
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO conversations(session_id) VALUES (?)", (session_id,))

    def add_message(self, session_id: str, role: str, content: str):
        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")
        self.ensure_session(session_id)
        with self._connect() as conn:
            conn.execute("INSERT INTO messages(session_id, role, content) VALUES (?, ?, ?)",
                         (session_id, role, content))
            conn.execute("UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE session_id=?",
                         (session_id,))

    def get_full_messages(self, session_id: str) -> List[Dict]:
        self.ensure_session(session_id)
        with self._connect() as conn:
            rows = conn.execute("SELECT id, role, content, created_at FROM messages WHERE session_id=? ORDER BY id",
                                (session_id,)).fetchall()
        return [dict(row) for row in rows]

    def get_context(self, session_id: str) -> Dict:
        """Return summary plus the latest complete user/assistant turns."""
        self.ensure_session(session_id)
        with self._connect() as conn:
            conv = conn.execute("SELECT summary FROM conversations WHERE session_id=?", (session_id,)).fetchone()
            rows = conn.execute("SELECT id, role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                                (session_id, self.recent_turns * 2)).fetchall()
        rows = list(reversed(rows))
        # Convert only complete user->assistant pairs to the format expected by rag_pipeline.
        history, pending_user = [], None
        for row in rows:
            if row["role"] == "user":
                pending_user = row["content"]
            elif row["role"] == "assistant" and pending_user is not None:
                history.append({"question": pending_user, "answer": row["content"]})
                pending_user = None
        summary = self._clip_tokens(conv["summary"] if conv else "", self.summary_token_limit)
        return {"summary": summary, "history": history}

    def maybe_update_summary(self, session_id: str, summarize_callable):
        """Summarize older messages incrementally; recent turns stay verbatim.

        summarize_callable receives (old_summary, old_messages_text) and returns a summary string.
        """
        self.ensure_session(session_id)
        with self._connect() as conn:
            conv = conn.execute("SELECT summary, summarized_through_id FROM conversations WHERE session_id=?",
                                (session_id,)).fetchone()
            cursor_id = int(conv["summarized_through_id"])
            rows = conn.execute("SELECT id, role, content FROM messages WHERE session_id=? AND id>? ORDER BY id",
                                (session_id, cursor_id)).fetchall()
        # Do not summarize the newest recent_turns complete pairs.
        keep_messages = self.recent_turns * 2
        cutoff = len(rows) - keep_messages
        if cutoff < self.summary_trigger_turns * 2:
            return
        old_rows = rows[:cutoff]
        if not old_rows:
            return
        old_text = "\n".join(f"{r['role'].upper()}: {r['content']}" for r in old_rows)
        updated = summarize_callable(conv["summary"], old_text)
        updated = self._clip_tokens((updated or "").strip(), self.summary_token_limit)
        if not updated:
            return
        with self._connect() as conn:
            conn.execute("UPDATE conversations SET summary=?, summarized_through_id=?, updated_at=CURRENT_TIMESTAMP WHERE session_id=?",
                         (updated, int(old_rows[-1]["id"]), session_id))

    def clear_session(self, session_id: str) -> None:
        """Delete a conversation's messages and rolling summary."""
        self.ensure_session(session_id)
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            conn.execute(
                "UPDATE conversations SET summary='', summarized_through_id=0, updated_at=CURRENT_TIMESTAMP WHERE session_id=?",
                (session_id,),
            )

    def _clip_tokens(self, text: str, limit: int) -> str:
        text = text or ""
        if self.encoding is None:
            # Conservative fallback (~4 chars/token); use tiktoken in the app environment.
            return text[: limit * 4] if len(text) > limit * 4 else text
        tokens = self.encoding.encode(text)
        if len(tokens) <= limit:
            return text
        return self.encoding.decode(tokens[:limit])


class SQLiteChatMessageHistory(BaseChatMessageHistory):
    """LangChain BaseChatMessageHistory adapter backed by ConversationMemory's SQLite DB.

    A tiny adapter keeps LangChain's RunnableWithMessageHistory responsible for
    loading/saving turns while the app retains its rolling-summary table.
    """

    def __init__(self, memory: ConversationMemory, session_id: str):
        # LangChain reads `messages` and calls `add_messages`; this adapter stores
        # only supported user/assistant message types in the existing database.
        self._memory = memory
        self._session_id = session_id
        self._memory.ensure_session(session_id)

    @property
    def messages(self):
        from langchain_core.messages import AIMessage, HumanMessage

        converted = []
        for row in self._memory.get_full_messages(self._session_id):
            if row["role"] == "user":
                converted.append(HumanMessage(content=row["content"]))
            elif row["role"] == "assistant":
                converted.append(AIMessage(content=row["content"]))
        return converted

    def add_messages(self, messages) -> None:
        from langchain_core.messages import AIMessage, HumanMessage

        # RunnableWithMessageHistory generally appends the user and assistant
        # messages together. Preserve their order in the existing SQLite schema.
        for message in messages:
            if isinstance(message, HumanMessage):
                self._memory.add_message(self._session_id, "user", str(message.content))
            elif isinstance(message, AIMessage):
                self._memory.add_message(self._session_id, "assistant", str(message.content))

    def add_message(self, message) -> None:
        self.add_messages([message])

    def clear(self) -> None:
        self._memory.clear_session(self._session_id)
