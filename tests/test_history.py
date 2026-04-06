"""HistoryDB のユニットテスト"""

import sqlite3
from pathlib import Path

import pytest

from history import HistoryDB


@pytest.fixture()
def db(tmp_path: Path) -> HistoryDB:
    """テストごとに新しいDBを作成"""
    return HistoryDB(db_path=tmp_path / "test.db")


def _add_sample(db: HistoryDB, **overrides) -> int:
    defaults = dict(
        source_type="file",
        source_name="test.wav",
        model="base",
        language="ja",
        detected_language="ja",
        duration_sec=3.5,
        text="テスト文字起こし結果",
    )
    defaults.update(overrides)
    return db.add(**defaults)


# ── テーブル初期化 ──────────────────────────────────────


class TestInitTables:
    def test_table_exists(self, db: HistoryDB):
        tables = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='history'"
        ).fetchall()
        assert len(tables) == 1

    def test_idempotent(self, db: HistoryDB):
        db._init_tables()
        tables = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='history'"
        ).fetchall()
        assert len(tables) == 1


# ── 追加 ─────────────────────────────────────────────


class TestAdd:
    def test_returns_row_id(self, db: HistoryDB):
        row_id = _add_sample(db)
        assert row_id == 1

    def test_increments_id(self, db: HistoryDB):
        id1 = _add_sample(db)
        id2 = _add_sample(db, text="二番目")
        assert id2 == id1 + 1

    def test_stores_all_fields(self, db: HistoryDB):
        _add_sample(
            db,
            source_type="recording",
            source_name=None,
            model="small",
            language="auto",
            detected_language="en",
            duration_sec=12.3,
            text="hello world",
        )
        row = db.get_by_id(1)
        assert row["source_type"] == "recording"
        assert row["source_name"] is None
        assert row["model"] == "small"
        assert row["language"] == "auto"
        assert row["detected_language"] == "en"
        assert row["duration_sec"] == pytest.approx(12.3)
        assert row["text"] == "hello world"
        assert row["created_at"]  # ISO形式の日時文字列

    def test_nullable_duration(self, db: HistoryDB):
        _add_sample(db, duration_sec=None)
        row = db.get_by_id(1)
        assert row["duration_sec"] is None


# ── 取得 ─────────────────────────────────────────────


class TestGetAll:
    def test_empty(self, db: HistoryDB):
        assert db.get_all() == []

    def test_returns_dicts(self, db: HistoryDB):
        _add_sample(db)
        rows = db.get_all()
        assert isinstance(rows[0], dict)

    def test_order_desc(self, db: HistoryDB):
        _add_sample(db, text="first")
        _add_sample(db, text="second")
        rows = db.get_all()
        assert rows[0]["text"] == "second"
        assert rows[1]["text"] == "first"

    def test_limit(self, db: HistoryDB):
        for i in range(5):
            _add_sample(db, text=f"item-{i}")
        rows = db.get_all(limit=3)
        assert len(rows) == 3


class TestGetById:
    def test_found(self, db: HistoryDB):
        row_id = _add_sample(db)
        row = db.get_by_id(row_id)
        assert row is not None
        assert row["id"] == row_id

    def test_not_found(self, db: HistoryDB):
        assert db.get_by_id(999) is None


# ── 削除 ─────────────────────────────────────────────


class TestDelete:
    def test_delete_existing(self, db: HistoryDB):
        row_id = _add_sample(db)
        assert db.delete(row_id) is True
        assert db.get_by_id(row_id) is None

    def test_delete_nonexistent(self, db: HistoryDB):
        assert db.delete(999) is False

    def test_delete_does_not_affect_others(self, db: HistoryDB):
        id1 = _add_sample(db, text="keep")
        id2 = _add_sample(db, text="remove")
        db.delete(id2)
        assert db.get_by_id(id1) is not None
        assert len(db.get_all()) == 1


# ── close ────────────────────────────────────────────


class TestClose:
    def test_close(self, db: HistoryDB):
        db.close()
        with pytest.raises(sqlite3.ProgrammingError):
            db.get_all()
