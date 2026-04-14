"""SQLite database helpers for prompt storage."""

import sqlite3

from prompt_similarity.config import DB_PATH


def get_db() -> sqlite3.Connection:
    """Return a new SQLite connection with Row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the prompts table if it does not already exist."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                prompt_id          TEXT PRIMARY KEY,
                category           TEXT,
                layer              TEXT,
                name               TEXT,
                content            TEXT,
                content_normalized TEXT,
                embedding          BLOB
            )
        """)
