"""文字起こし履歴をSQLiteで管理するモジュール"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / ".whisper-desk" / "history.db"

# ── SQL ───────────────────────────────────────────────

_SQL_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS history (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        TEXT    NOT NULL,
    source_type       TEXT    NOT NULL,  -- 'file' or 'recording'
    source_name       TEXT,              -- 元ファイル名 (録音の場合はNULL)
    model             TEXT    NOT NULL,
    language          TEXT    NOT NULL,  -- 'auto' or language code
    detected_language TEXT,              -- Whisperが検出した言語
    duration_sec      REAL,             -- 音声の長さ(秒)
    text              TEXT    NOT NULL
)
"""

_SQL_INSERT = """
INSERT INTO history
    (created_at, source_type, source_name, model, language,
     detected_language, duration_sec, text)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

_SQL_SELECT_ALL = "SELECT * FROM history ORDER BY id DESC LIMIT ?"

_SQL_SELECT_BY_ID = "SELECT * FROM history WHERE id = ?"

_SQL_DELETE = "DELETE FROM history WHERE id = ?"


# ── HistoryDB ─────────────────────────────────────────


class HistoryDB:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()
        logger.info("履歴DB接続: %s", self.db_path)

    def _init_tables(self):
        self.conn.execute(_SQL_CREATE_TABLE)
        self.conn.commit()

    def add(
        self,
        *,
        source_type: str,
        source_name: str | None,
        model: str,
        language: str,
        detected_language: str | None,
        duration_sec: float | None,
        text: str,
    ) -> int:
        cur = self.conn.execute(
            _SQL_INSERT,
            (
                datetime.now().isoformat(),
                source_type,
                source_name,
                model,
                language,
                detected_language,
                duration_sec,
                text,
            ),
        )
        self.conn.commit()
        row_id = cur.lastrowid
        logger.info(
            "履歴追加: id=%d, model=%s, source=%s",
            row_id,
            model,
            source_name or "(録音)",
        )
        return row_id

    def get_all(self, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(_SQL_SELECT_ALL, (limit,)).fetchall()
        return [dict(row) for row in rows]

    def get_by_id(self, row_id: int) -> dict | None:
        row = self.conn.execute(_SQL_SELECT_BY_ID, (row_id,)).fetchone()
        return dict(row) if row else None

    def delete(self, row_id: int) -> bool:
        cur = self.conn.execute(_SQL_DELETE, (row_id,))
        self.conn.commit()
        deleted = cur.rowcount > 0
        if deleted:
            logger.info("履歴削除: id=%d", row_id)
        return deleted

    def close(self):
        self.conn.close()
        logger.info("履歴DB切断")
