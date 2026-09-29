"""
db.py
-----
Tiny SQLite persistence layer. Every module analysis (disease scan, yield
prediction, soil test, irrigation check) is logged so the dashboard and
history views have real data to show, and so the demo still looks alive
after a page refresh.
"""

import sqlite3
import json
import os
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "agrivision.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            module TEXT NOT NULL,
            summary TEXT NOT NULL,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_analysis(module: str, summary: str, payload: dict):
    conn = get_conn()
    conn.execute(
        "INSERT INTO analyses (module, summary, payload, created_at) VALUES (?, ?, ?, ?)",
        (module, summary, json.dumps(payload), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def get_history(limit=50, module=None):
    conn = get_conn()
    if module:
        rows = conn.execute(
            "SELECT * FROM analyses WHERE module = ? ORDER BY id DESC LIMIT ?",
            (module, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [
        {
            "id": r["id"],
            "module": r["module"],
            "summary": r["summary"],
            "payload": json.loads(r["payload"]),
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def get_dashboard_stats():
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) c FROM analyses").fetchone()["c"]
    by_module = conn.execute(
        "SELECT module, COUNT(*) c FROM analyses GROUP BY module"
    ).fetchall()
    disease_alerts = conn.execute(
        "SELECT COUNT(*) c FROM analyses WHERE module = 'disease' AND summary != 'healthy'"
    ).fetchone()["c"]
    conn.close()
    return {
        "total_analyses": total,
        "by_module": {r["module"]: r["c"] for r in by_module},
        "disease_alerts": disease_alerts,
    }
