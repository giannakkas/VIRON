"""
VIRON Hybrid Gateway — SQLite Persistence
Stores sessions, messages, and routing decisions.
Reuses /data/viron.db (bind-mounted).
"""
import sqlite3
import json
import time
import os
from contextlib import contextmanager
from typing import Optional

import config as cfg

_db_path = cfg.DB_PATH

def init_db():
    """Create tables if they don't exist."""
    os.makedirs(os.path.dirname(_db_path), exist_ok=True)
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT,
                age INTEGER DEFAULT 10,
                language TEXT DEFAULT 'en',
                created_at REAL,
                last_seen REAL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                started_at REAL,
                ended_at REAL,
                FOREIGN KEY (student_id) REFERENCES students(student_id)
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                student_id TEXT,
                role TEXT,
                content TEXT,
                mode TEXT,
                cloud_provider TEXT,
                router_json TEXT,
                latency_ms REAL,
                created_at REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                session_id INTEGER,
                summary TEXT,
                created_at REAL
            );
        """)

@contextmanager
def _connect():
    conn = sqlite3.connect(_db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def ensure_student(student_id: str, age: int = 10, language: str = "en"):
    """Create student if not exists, update last_seen."""
    now = time.time()
    with _connect() as conn:
        row = conn.execute("SELECT 1 FROM students WHERE student_id=?", (student_id,)).fetchone()
        if row:
            conn.execute("UPDATE students SET last_seen=?, age=?, language=? WHERE student_id=?",
                         (now, age, language, student_id))
        else:
            conn.execute("INSERT INTO students (student_id, name, age, language, created_at, last_seen) VALUES (?,?,?,?,?,?)",
                         (student_id, student_id, age, language, now, now))

def log_message(student_id: str, role: str, content: str,
                mode: str = "", cloud_provider: str = "",
                router_json: Optional[dict] = None, latency_ms: float = 0):
    """Log a message (user or assistant) to the messages table."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (student_id, role, content, mode, cloud_provider, router_json, latency_ms, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (student_id, role, content, mode, cloud_provider,
             json.dumps(router_json) if router_json else None,
             latency_ms, time.time()))

def get_recent_messages(student_id: str, limit: int = 20) -> list:
    """Get recent messages for context window."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE student_id=? ORDER BY created_at DESC LIMIT ?",
            (student_id, limit)).fetchall()
    # Return in chronological order
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
