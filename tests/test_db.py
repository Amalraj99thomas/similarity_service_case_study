"""Tests for prompt_similarity.db — SQLite helpers."""

import sqlite3

import pytest

from prompt_similarity import db, config


@pytest.fixture(autouse=True)
def _use_temp_db(monkeypatch, tmp_path):
    """Point the DB at a temporary file for each test."""
    test_db = str(tmp_path / "test.db")
    monkeypatch.setattr(config, "DB_PATH", test_db)
    monkeypatch.setattr(db, "DB_PATH", test_db)


class TestDatabase:
    """Tests for database initialisation and connection."""

    def test_init_db_creates_table(self):
        db.init_db()
        conn = db.get_db()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='prompts'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_init_db_is_idempotent(self):
        db.init_db()
        db.init_db()  # should not raise

    def test_get_db_returns_row_factory(self):
        db.init_db()
        conn = db.get_db()
        assert conn.row_factory == sqlite3.Row
        conn.close()

    def test_insert_and_read_prompt(self):
        db.init_db()
        with db.get_db() as conn:
            conn.execute(
                "INSERT INTO prompts (prompt_id, category, layer, content) VALUES (?,?,?,?)",
                ("test_1", "greeting", "engine", "Hello world"),
            )
        with db.get_db() as conn:
            row = conn.execute(
                "SELECT * FROM prompts WHERE prompt_id = ?", ("test_1",)
            ).fetchone()
        assert row["prompt_id"] == "test_1"
        assert row["content"] == "Hello world"
