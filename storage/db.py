# storage/db.py
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Optional


DEFAULT_DB_PATH = os.path.join("data", "agent_runs.db")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)



def connect(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def init_db(con: sqlite3.Connection) -> None:
    cur = con.cursor()

    # One row per execution
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            started_at_utc TEXT NOT NULL,
            input_csv TEXT NOT NULL,
            max_rows INTEGER NOT NULL,
            save_memory INTEGER NOT NULL,
            model TEXT,
            used_llm INTEGER,
            notes TEXT
        )
        """
    )

    # One row per campaign within a run
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS campaign_outputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            campaign_id TEXT NOT NULL,

            stance TEXT,
            severity TEXT,

            cpa REAL,
            roas REAL,
            target_cpa REAL,
            cpa_trend_7d REAL,
            roas_trend_7d REAL,
            days_active INTEGER,

            state_json TEXT NOT NULL,
            decision_json TEXT NOT NULL,
            analysis_json TEXT NOT NULL,
            advisor_json TEXT NOT NULL,
            scenarios_json TEXT NOT NULL,

            FOREIGN KEY(run_id) REFERENCES runs(run_id)
        )
        """
    )

    cur.execute("CREATE INDEX IF NOT EXISTS idx_campaign_outputs_run ON campaign_outputs(run_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_campaign_outputs_campaign ON campaign_outputs(campaign_id)")

    con.commit()


def start_run(
    con: sqlite3.Connection,
    *,
    input_csv: str,
    max_rows: int,
    save_memory: int,
    model: Optional[str] = None,
    used_llm: Optional[bool] = None,
    notes: str = "",
) -> str:
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO runs (run_id, started_at_utc, input_csv, max_rows, save_memory, model, used_llm, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            _utc_now_iso(),
            input_csv,
            int(max_rows),
            int(save_memory),
            model,
            None if used_llm is None else int(bool(used_llm)),
            notes,
        ),
    )
    con.commit()
    return run_id


def save_campaign_output(
    con: sqlite3.Connection,
    *,
    run_id: str,
    campaign_id: str,
    state: Dict[str, Any],
    decision: Dict[str, Any],
    analysis: Dict[str, Any],
    advisor: Dict[str, Any],
    scenarios: Any,
) -> None:
    # Pull a few fields for easy filtering in the dashboard
    stance = decision.get("stance")
    severity = decision.get("severity")

    def _f(x):
        try:
            return None if x is None else float(x)
        except Exception:
            return None

    def _i(x):
        try:
            return None if x is None else int(x)
        except Exception:
            return None

    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO campaign_outputs (
            run_id, campaign_id,
            stance, severity,
            cpa, roas, target_cpa, cpa_trend_7d, roas_trend_7d, days_active,
            state_json, decision_json, analysis_json, advisor_json, scenarios_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            campaign_id,
            stance,
            severity,
            _f(state.get("CPA")),
            _f(state.get("ROAS")),
            _f(state.get("target_CPA")),
            _f(state.get("CPA_trend_7d")),
            _f(state.get("ROAS_trend_7d")),
            _i(state.get("days_active")),
            _json_dumps(state),
            _json_dumps(decision),
            _json_dumps(analysis),
            _json_dumps(advisor),
            _json_dumps(scenarios),
        ),
    )
    con.commit()
