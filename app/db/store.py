from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.utils import json_dumps


class EvalStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                config_json TEXT NOT NULL,
                summary_json TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS case_results (
                run_id TEXT NOT NULL,
                case_id TEXT NOT NULL,
                status TEXT NOT NULL,
                round1_json TEXT,
                round1_metrics_json TEXT,
                t2_json TEXT,
                t2_metrics_json TEXT,
                cot_json TEXT,
                error TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (run_id, case_id)
            )
            """
        )
        self.conn.commit()

    def create_or_resume_run(self, run_id: str, config: Dict[str, Any], case_ids: List[str]) -> None:
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """
            INSERT OR IGNORE INTO runs(run_id, status, created_at, updated_at, config_json)
            VALUES (?, 'running', ?, ?, ?)
            """,
            (run_id, now, now, json_dumps(config)),
        )
        self.conn.execute("UPDATE runs SET status='running', updated_at=? WHERE run_id=?", (now, run_id))
        for cid in case_ids:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO case_results(run_id, case_id, status, updated_at)
                VALUES (?, ?, 'pending', ?)
                """,
                (run_id, cid, now),
            )
        self.conn.commit()

    def update_case(
        self,
        run_id: str,
        case_id: str,
        status: Optional[str] = None,
        round1_json: Optional[Dict[str, Any]] = None,
        round1_metrics: Optional[Dict[str, Any]] = None,
        t2_json: Optional[Dict[str, Any]] = None,
        t2_metrics: Optional[Dict[str, Any]] = None,
        cot_json: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        now = datetime.utcnow().isoformat()
        fields = ["updated_at=?"]
        values: List[Any] = [now]
        if status is not None:
            fields.append("status=?")
            values.append(status)
        if round1_json is not None:
            fields.append("round1_json=?")
            values.append(json_dumps(round1_json))
        if round1_metrics is not None:
            fields.append("round1_metrics_json=?")
            values.append(json_dumps(round1_metrics))
        if t2_json is not None:
            fields.append("t2_json=?")
            values.append(json_dumps(t2_json))
        if t2_metrics is not None:
            fields.append("t2_metrics_json=?")
            values.append(json_dumps(t2_metrics))
        if cot_json is not None:
            fields.append("cot_json=?")
            values.append(json_dumps(cot_json))
        if error is not None:
            fields.append("error=?")
            values.append(error)
        values.extend([run_id, case_id])
        sql = f"UPDATE case_results SET {', '.join(fields)} WHERE run_id=? AND case_id=?"
        self.conn.execute(sql, values)
        self.conn.commit()

    def get_pending_case_ids(self, run_id: str) -> List[str]:
        rows = self.conn.execute(
            "SELECT case_id FROM case_results WHERE run_id=? AND status!='success' ORDER BY case_id",
            (run_id,),
        ).fetchall()
        return [r["case_id"] for r in rows]

    def get_run_cases(self, run_id: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM case_results WHERE run_id=? ORDER BY case_id",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def set_run_status(self, run_id: str, status: str, summary: Optional[Dict[str, Any]] = None) -> None:
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            "UPDATE runs SET status=?, updated_at=?, summary_json=? WHERE run_id=?",
            (status, now, json_dumps(summary or {}), run_id),
        )
        self.conn.commit()

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

