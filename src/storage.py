from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


DB_PATH = Path("data/project_runs.sqlite3")


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                requirement TEXT NOT NULL,
                result_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL
            )
            """
        )


def save_run(requirement: str, result: dict[str, Any], metrics: dict[str, float]) -> int:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "INSERT INTO runs (created_at, requirement, result_json, metrics_json) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), requirement, json.dumps(result), json.dumps(metrics)),
        )
        return int(cursor.lastrowid)


def load_recent_runs(limit: int = 5) -> list[dict[str, Any]]:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, created_at, requirement, metrics_json FROM runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {
            "id": row[0],
            "created_at": row[1],
            "requirement": row[2],
            "metrics": json.loads(row[3]),
        }
        for row in rows
    ]
