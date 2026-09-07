from __future__ import annotations

import sqlite3
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path.home() / "Documents" / "metis-server" / "chats.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    # Enable foreign keys and WAL for better concurrency
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def init_db() -> None:
    conn = get_connection()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                content TEXT NOT NULL,
                model TEXT,
                created_at TEXT NOT NULL,
                response_time_ms INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_chats_updated_at ON chats(updated_at);
            """
        )
        conn.commit()
    finally:
        conn.close()


@contextmanager
def db_conn():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
