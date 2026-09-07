from __future__ import annotations

import uuid

from app.core.database import db_conn
from app.core.time_utils import now_iso


def _new_id() -> str:
    return uuid.uuid4().hex


def create_chat_db(title: str | None) -> dict:
    """Create a new chat row and return its summary."""
    chat_id = _new_id()
    now = now_iso()
    # Default title "New chat" if not provided, will be replaced by first user message;
    # cap the title at 200 chars for display, matching the update path.

    effective_title = title.strip()[:200] if title and title.strip() else "New chat"
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO chats (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (chat_id, effective_title, now, now),
        )
    return {"id": chat_id, "title": effective_title, "created_at": now, "updated_at": now}


def list_chats_db() -> list[dict]:
    """List chats, newest first, with message counts."""
    with db_conn() as conn:
        cur = conn.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.id) as message_count
            FROM chats c
            ORDER BY c.updated_at DESC, c.created_at DESC
            """
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]


def get_chat_db(chat_id: str) -> dict | None:
    """Return a chat with its messages, or None if it does not exist."""
    with db_conn() as conn:
        cur = conn.execute("SELECT id, title, created_at, updated_at FROM chats WHERE id = ?", (chat_id,))
        row = cur.fetchone()
        if not row:
            return None
        chat = dict(row)
        cur = conn.execute(
            "SELECT id, chat_id, role, content, model, created_at, response_time_ms FROM messages WHERE chat_id = ? ORDER BY created_at ASC, rowid ASC",
            (chat_id,),
        )
        msgs = [dict(r) for r in cur.fetchall()]
        chat["messages"] = msgs
        return chat


def chat_exists_db(chat_id: str) -> bool:
    """Return True if a chat with the given id exists."""
    with db_conn() as conn:
        cur = conn.execute("SELECT 1 FROM chats WHERE id = ?", (chat_id,))
        return cur.fetchone() is not None


def update_title_db(chat_id: str, title: str) -> dict | None:
    """Update a chat's title, capping it at 200 chars, and return its summary, or None if missing."""
    now = now_iso()
    normalized_title = title.strip()[:200]
    with db_conn() as conn:
        cur = conn.execute("UPDATE chats SET title = ?, updated_at = ? WHERE id = ?", (normalized_title, now, chat_id))
        if cur.rowcount == 0:
            return None
        cur = conn.execute("SELECT id, title, created_at, updated_at FROM chats WHERE id = ?", (chat_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def delete_chat_db(chat_id: str) -> bool:
    """Delete a chat, returning True if it existed."""
    with db_conn() as conn:
        cur = conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        return cur.rowcount > 0


def add_messages_db(chat_id: str, user_content: str, assistant_content: str, model: str | None, response_time_ms: int, user_created_at: str, assistant_created_at: str) -> tuple[dict, dict]:
    """Insert a user+assistant message pair, and bump the chat title/updated_at."""
    user_id = _new_id()
    assistant_id = _new_id()
    with db_conn() as conn:
        # Ensure chat exists
        cur = conn.execute("SELECT title FROM chats WHERE id = ?", (chat_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError("chat not found")
        # Update title if default and first message
        current_title = row["title"]
        # Count existing messages
        cur = conn.execute("SELECT COUNT(*) as cnt FROM messages WHERE chat_id = ?", (chat_id,))
        cnt = cur.fetchone()["cnt"]
        new_title = current_title
        if cnt == 0 and current_title == "New chat":
            # Default to first 60 chars of user message
            new_title = user_content.strip()[:60] or current_title
            if not new_title.strip():
                new_title = current_title
        # Insert user message
        conn.execute(
            "INSERT INTO messages (id, chat_id, role, content, model, created_at, response_time_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, chat_id, "user", user_content, None, user_created_at, None),
        )
        # Insert assistant message
        conn.execute(
            "INSERT INTO messages (id, chat_id, role, content, model, created_at, response_time_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (assistant_id, chat_id, "assistant", assistant_content, model, assistant_created_at, response_time_ms),
        )
        # Update chat updated_at and title if changed
        if new_title != current_title:
            conn.execute("UPDATE chats SET title = ?, updated_at = ? WHERE id = ?", (new_title, assistant_created_at, chat_id))
        else:
            conn.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (assistant_created_at, chat_id))

    user_msg = {"id": user_id, "chat_id": chat_id, "role": "user", "content": user_content, "model": None, "created_at": user_created_at, "response_time_ms": None}
    assistant_msg = {"id": assistant_id, "chat_id": chat_id, "role": "assistant", "content": assistant_content, "model": model, "created_at": assistant_created_at, "response_time_ms": response_time_ms}
    return user_msg, assistant_msg
