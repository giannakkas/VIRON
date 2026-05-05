"""
Face database: persistent storage of family member face encodings.

Schema (SQLite):

    faces
      id INTEGER PRIMARY KEY
      name TEXT NOT NULL                 -- "Andreas", "Mom", etc.
      role TEXT                          -- "student" | "mother" | "father" | "sister" | "brother" | "grandmother" | "grandfather" | "aunt" | "uncle" | "cousin" | "friend" | "teacher" | "other" (legacy: "parent", "sibling")
      encoding BLOB NOT NULL             -- numpy float64 (128,) tobytes()
      thumbnail BLOB                     -- small JPG (~256x256) of the cropped face
      created_at REAL                    -- unix timestamp
      last_seen REAL                     -- unix timestamp of most recent recognition
      recognized_count INTEGER DEFAULT 0
      drift_updates INTEGER DEFAULT 0    -- count of silent encoding updates

Encodings are 128-dim float64 vectors produced by dlib's face_recognition_model_v1.
They are NOT reversible to a face image — even with full database access, an
attacker cannot reconstruct what the family looks like.
"""
import os
import sqlite3
import threading
import time
import logging
from typing import Optional

import numpy as np

log = logging.getLogger("VIRON")

ENCODING_DIM = 128
ENCODING_DTYPE = np.float64
ENCODING_BYTES = ENCODING_DIM * 8   # float64 = 8 bytes


class FaceDB:
    """Thread-safe SQLite store for face encodings + thumbnails."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _conn(self):
        # New connection per thread — sqlite3 connections aren't thread-safe by default
        c = sqlite3.connect(self.db_path, timeout=5.0)
        c.row_factory = sqlite3.Row
        return c

    def _init_schema(self):
        with self._lock, self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS faces (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    name             TEXT NOT NULL,
                    role             TEXT,
                    encoding         BLOB NOT NULL,
                    thumbnail        BLOB,
                    created_at       REAL NOT NULL,
                    last_seen        REAL,
                    recognized_count INTEGER DEFAULT 0,
                    drift_updates    INTEGER DEFAULT 0
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_faces_name ON faces(name)")

    # ─── enrollment ──────────────────────────────────────────

    def add_face(self, name: str, role: str, encoding: np.ndarray,
                 thumbnail_jpeg: Optional[bytes] = None) -> int:
        """Insert a new face. Returns the new row's id."""
        if encoding.shape != (ENCODING_DIM,):
            raise ValueError(f"encoding must be ({ENCODING_DIM},), got {encoding.shape}")
        enc_bytes = encoding.astype(ENCODING_DTYPE).tobytes()
        now = time.time()
        with self._lock, self._conn() as c:
            cur = c.execute("""
                INSERT INTO faces (name, role, encoding, thumbnail, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (name.strip(), role or "other", enc_bytes, thumbnail_jpeg, now))
            return cur.lastrowid

    def update_encoding(self, face_id: int, new_encoding: np.ndarray,
                        averaging_weight: float = 0.1) -> bool:
        """Drift-correct a stored encoding. The new encoding is blended into
        the old one (default 10% new, 90% old) so we adapt slowly to lighting,
        haircuts, growth — without making one bad frame override good history."""
        if new_encoding.shape != (ENCODING_DIM,):
            return False
        with self._lock, self._conn() as c:
            row = c.execute("SELECT encoding, drift_updates FROM faces WHERE id=?",
                            (face_id,)).fetchone()
            if not row:
                return False
            old = np.frombuffer(row["encoding"], dtype=ENCODING_DTYPE).copy()
            blended = (1 - averaging_weight) * old + averaging_weight * new_encoding
            c.execute("""
                UPDATE faces SET encoding=?, drift_updates=drift_updates+1 WHERE id=?
            """, (blended.astype(ENCODING_DTYPE).tobytes(), face_id))
            return True

    def replace_encoding(self, face_id: int, new_encoding: np.ndarray,
                         new_thumbnail_jpeg: Optional[bytes] = None) -> bool:
        """Hard replace the encoding (and optionally thumbnail). Used when a
        parent explicitly re-registers a face from the settings panel."""
        if new_encoding.shape != (ENCODING_DIM,):
            return False
        enc_bytes = new_encoding.astype(ENCODING_DTYPE).tobytes()
        with self._lock, self._conn() as c:
            if new_thumbnail_jpeg is not None:
                c.execute("UPDATE faces SET encoding=?, thumbnail=? WHERE id=?",
                          (enc_bytes, new_thumbnail_jpeg, face_id))
            else:
                c.execute("UPDATE faces SET encoding=? WHERE id=?",
                          (enc_bytes, face_id))
            return c.total_changes > 0

    # ─── retrieval ───────────────────────────────────────────

    def list_faces(self) -> list[dict]:
        with self._lock, self._conn() as c:
            rows = c.execute("""
                SELECT id, name, role, created_at, last_seen,
                       recognized_count, drift_updates,
                       (thumbnail IS NOT NULL) AS has_thumbnail
                FROM faces ORDER BY name COLLATE NOCASE
            """).fetchall()
            return [dict(r) for r in rows]

    def get_all_encodings(self) -> list[tuple[int, str, str, np.ndarray]]:
        """Returns [(id, name, role, encoding)] for in-memory matching."""
        with self._lock, self._conn() as c:
            rows = c.execute("SELECT id, name, role, encoding FROM faces").fetchall()
            return [
                (r["id"], r["name"], r["role"] or "other",
                 np.frombuffer(r["encoding"], dtype=ENCODING_DTYPE).copy())
                for r in rows
            ]

    def get_thumbnail(self, face_id: int) -> Optional[bytes]:
        with self._lock, self._conn() as c:
            row = c.execute("SELECT thumbnail FROM faces WHERE id=?",
                            (face_id,)).fetchone()
            return bytes(row["thumbnail"]) if row and row["thumbnail"] else None

    # ─── housekeeping ────────────────────────────────────────

    def touch_recognition(self, face_id: int):
        """Bump last_seen + recognized_count when a face is identified."""
        with self._lock, self._conn() as c:
            c.execute("""
                UPDATE faces SET last_seen=?, recognized_count=recognized_count+1
                WHERE id=?
            """, (time.time(), face_id))

    def delete_face(self, face_id: int) -> bool:
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM faces WHERE id=?", (face_id,))
            return c.total_changes > 0

    def count(self) -> int:
        with self._lock, self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
