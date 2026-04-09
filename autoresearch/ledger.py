"""Experiment ledger — SQLite-backed record of every autoresearch run.

The ledger is the agent's memory of the search. It stores what was tried,
what happened, and what was learned. Project-agnostic: metrics are stored
as JSON with a denormalized primary_metric for fast ranking.

Usage:
    ledger = Ledger("/path/to/ledger.db")
    ledger.insert_run(run_id="...", project="hydra", ...)
    ledger.complete_run(run_id, result_dict)
    rows = ledger.query(project="hydra", phase="quick", status="keep")
    print(ledger.tsv_export("hydra"))
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    project         TEXT NOT NULL,
    parent_run_id   TEXT,
    commit_sha      TEXT NOT NULL,
    image_hash      TEXT DEFAULT '',
    track           TEXT DEFAULT 'architecture',
    phase           TEXT NOT NULL,
    config_json     TEXT NOT NULL DEFAULT '{}',
    metrics_json    TEXT,
    primary_metric  REAL,
    hypothesis      TEXT DEFAULT '',
    observation     TEXT DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'running',
    cost_gpu_min    REAL,
    started_at      REAL,
    finished_at     REAL,
    log_path        TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS lessons (
    lesson_id       TEXT PRIMARY KEY,
    text            TEXT NOT NULL,
    evidence        TEXT DEFAULT '',
    created_at      REAL,
    superseded_by   TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


@dataclass
class Run:
    run_id: str
    project: str
    commit_sha: str
    phase: str
    status: str = "running"
    parent_run_id: str = ""
    image_hash: str = ""
    track: str = "architecture"
    config_json: str = "{}"
    metrics_json: str | None = None
    primary_metric: float | None = None
    hypothesis: str = ""
    observation: str = ""
    cost_gpu_min: float | None = None
    started_at: float | None = None
    finished_at: float | None = None
    log_path: str = ""


@dataclass
class Lesson:
    lesson_id: str
    text: str
    evidence: str = ""
    created_at: float | None = None
    superseded_by: str = ""


class Ledger:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        # Store schema version for future migrations.
        self._conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def insert_run(
        self,
        *,
        project: str,
        commit_sha: str,
        phase: str,
        config_overrides: dict | None = None,
        run_id: str | None = None,
        parent_run_id: str = "",
        track: str = "architecture",
        hypothesis: str = "",
        image_hash: str = "",
    ) -> str:
        """Insert a new run with status='running'. Returns the run_id."""
        run_id = run_id or _new_id()
        self._conn.execute(
            """INSERT INTO runs
               (run_id, project, commit_sha, phase, config_json,
                parent_run_id, track, hypothesis, image_hash,
                status, started_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)""",
            (
                run_id, project, commit_sha, phase,
                json.dumps(config_overrides or {}),
                parent_run_id, track, hypothesis, image_hash,
                time.time(),
            ),
        )
        self._conn.commit()
        return run_id

    def complete_run(
        self,
        run_id: str,
        result: dict,
        *,
        primary_metric_key: str = "",
    ) -> None:
        """Update a run with the result dict returned by run_phase.

        result shape matches the CLI contract:
            {status, metrics, cost, image_hash, actual_commit_sha, notes, ...}
        """
        metrics = result.get("metrics", {})
        # Strip nested per_dataset dict for the flat metrics_json;
        # keep it in a separate key if needed later.
        metrics_flat = {k: v for k, v in metrics.items() if k != "per_dataset"}
        primary = None
        if primary_metric_key and primary_metric_key in metrics_flat:
            primary = float(metrics_flat[primary_metric_key])

        cost_s = result.get("cost", {}).get("gpu_seconds")
        cost_min = round(cost_s / 60, 2) if cost_s else None

        self._conn.execute(
            """UPDATE runs SET
                 status = ?,
                 metrics_json = ?,
                 primary_metric = ?,
                 cost_gpu_min = ?,
                 image_hash = COALESCE(NULLIF(?, ''), image_hash),
                 finished_at = ?,
                 observation = ?
               WHERE run_id = ?""",
            (
                result.get("status", "failed"),
                json.dumps(metrics),
                primary,
                cost_min,
                result.get("image_hash", ""),
                time.time(),
                result.get("notes", ""),
                run_id,
            ),
        )
        self._conn.commit()

    def update_run(self, run_id: str, **fields: Any) -> None:
        """Generic field update. Only declared columns are accepted."""
        allowed = {
            "status", "hypothesis", "observation", "track",
            "parent_run_id", "image_hash", "log_path",
        }
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"Cannot update fields: {bad}")
        if not fields:
            return
        sets = ", ".join(f"{k} = ?" for k in fields)
        vals = list(fields.values()) + [run_id]
        self._conn.execute(f"UPDATE runs SET {sets} WHERE run_id = ?", vals)
        self._conn.commit()

    def set_status(self, run_id: str, status: str, note: str = "") -> None:
        """Set status and optionally append to observation."""
        if note:
            self._conn.execute(
                """UPDATE runs SET status = ?,
                     observation = CASE WHEN observation = '' THEN ?
                                       ELSE observation || '\n' || ? END
                   WHERE run_id = ?""",
                (status, note, note, run_id),
            )
        else:
            self._conn.execute(
                "UPDATE runs SET status = ? WHERE run_id = ?",
                (status, run_id),
            )
        self._conn.commit()

    def get_run(self, run_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None

    def query(
        self,
        *,
        project: str | None = None,
        phase: str | None = None,
        status: str | None = None,
        track: str | None = None,
        order_by: str = "started_at DESC",
        limit: int = 100,
    ) -> list[dict]:
        """Query runs with optional filters. Returns list of dicts."""
        clauses: list[str] = []
        params: list[Any] = []
        if project:
            clauses.append("project = ?")
            params.append(project)
        if phase:
            clauses.append("phase = ?")
            params.append(phase)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if track:
            clauses.append("track = ?")
            params.append(track)
        where = " AND ".join(clauses) if clauses else "1"
        sql = f"SELECT * FROM runs WHERE {where} ORDER BY {order_by} LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self._conn.execute(sql, params)]

    def best_runs(
        self,
        project: str,
        phase: str | None = None,
        n: int = 5,
        higher_is_better: bool = True,
    ) -> list[dict]:
        """Top-N runs by primary_metric."""
        direction = "DESC" if higher_is_better else "ASC"
        clauses = ["project = ?", "primary_metric IS NOT NULL"]
        params: list[Any] = [project]
        if phase:
            clauses.append("phase = ?")
            params.append(phase)
        where = " AND ".join(clauses)
        sql = (
            f"SELECT * FROM runs WHERE {where} "
            f"ORDER BY primary_metric {direction} LIMIT ?"
        )
        params.append(n)
        return [dict(r) for r in self._conn.execute(sql, params)]

    # ------------------------------------------------------------------
    # Lessons
    # ------------------------------------------------------------------

    def add_lesson(self, text: str, evidence: str = "") -> str:
        lid = _new_id()
        self._conn.execute(
            "INSERT INTO lessons (lesson_id, text, evidence, created_at) "
            "VALUES (?, ?, ?, ?)",
            (lid, text, evidence, time.time()),
        )
        self._conn.commit()
        return lid

    def query_lessons(self, include_superseded: bool = False) -> list[dict]:
        if include_superseded:
            sql = "SELECT * FROM lessons ORDER BY created_at"
        else:
            sql = "SELECT * FROM lessons WHERE superseded_by = '' ORDER BY created_at"
        return [dict(r) for r in self._conn.execute(sql)]

    def supersede_lesson(self, old_id: str, new_id: str) -> None:
        self._conn.execute(
            "UPDATE lessons SET superseded_by = ? WHERE lesson_id = ?",
            (new_id, old_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Import / Export
    # ------------------------------------------------------------------

    def import_tsv(
        self,
        tsv_path: str | Path,
        project: str,
        primary_metric_key: str = "avg_ndcg10",
    ) -> int:
        """Bulk-import a hydra-style results.tsv into the ledger.

        TSV columns: commit, track, mode, scifact_ndcg10, fiqa_ndcg10,
                     avg_ndcg10, task_vs_generic, status, description

        Returns the number of rows imported.
        """
        tsv_path = Path(tsv_path)
        lines = tsv_path.read_text().strip().splitlines()
        header = lines[0].split("\t")
        count = 0
        for i, line in enumerate(lines[1:], 1):
            fields = line.split("\t")
            if len(fields) != len(header):
                continue
            row = dict(zip(header, fields))

            # Build metrics dict from the numeric columns.
            metrics: dict[str, Any] = {}
            for col in ["scifact_ndcg10", "fiqa_ndcg10", "avg_ndcg10", "task_vs_generic"]:
                val = row.get(col, "")
                if val and val != "n/a":
                    try:
                        metrics[col] = float(val)
                    except ValueError:
                        pass

            primary = metrics.get(primary_metric_key)
            run_id = f"tsv-{i:04d}"

            self._conn.execute(
                """INSERT OR REPLACE INTO runs
                   (run_id, project, commit_sha, phase, track, config_json,
                    metrics_json, primary_metric, status, observation)
                   VALUES (?, ?, ?, ?, ?, '{}', ?, ?, ?, ?)""",
                (
                    run_id, project,
                    row.get("commit", ""),
                    row.get("mode", ""),
                    row.get("track", ""),
                    json.dumps(metrics),
                    primary,
                    row.get("status", ""),
                    row.get("description", ""),
                ),
            )
            count += 1
        self._conn.commit()
        return count

    def tsv_export(
        self,
        project: str,
        columns: list[str] | None = None,
    ) -> str:
        """Export runs as a TSV string matching hydra's results.tsv format.

        Default columns mirror the original: commit, track, mode,
        scifact_ndcg10, fiqa_ndcg10, avg_ndcg10, task_vs_generic, status,
        description.
        """
        if columns is None:
            columns = [
                "commit", "track", "mode", "scifact_ndcg10", "fiqa_ndcg10",
                "avg_ndcg10", "task_vs_generic", "status", "description",
            ]

        rows = self._conn.execute(
            "SELECT * FROM runs WHERE project = ? ORDER BY rowid",
            (project,),
        ).fetchall()

        lines = ["\t".join(columns)]
        for row in rows:
            d = dict(row)
            metrics = json.loads(d.get("metrics_json") or "{}")
            vals: list[str] = []
            for col in columns:
                if col == "commit":
                    vals.append(d.get("commit_sha", ""))
                elif col == "mode":
                    vals.append(d.get("phase", ""))
                elif col == "description":
                    vals.append(d.get("observation", ""))
                elif col == "status":
                    vals.append(d.get("status", ""))
                elif col == "track":
                    vals.append(d.get("track", ""))
                elif col in metrics:
                    v = metrics[col]
                    if v is None:
                        vals.append("n/a")
                    elif isinstance(v, float):
                        # Format to match TSV precision (4 decimal places,
                        # with +/- prefix for task_vs_generic).
                        if col == "task_vs_generic":
                            vals.append(f"{v:+.4f}")
                        else:
                            vals.append(f"{v:.4f}")
                    else:
                        vals.append(str(v))
                else:
                    vals.append("n/a")
            lines.append("\t".join(vals))

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self, project: str) -> dict:
        """Quick summary stats for a project."""
        row = self._conn.execute(
            """SELECT
                 COUNT(*) as total,
                 SUM(CASE WHEN status = 'keep' THEN 1 ELSE 0 END) as kept,
                 SUM(CASE WHEN status = 'discard' THEN 1 ELSE 0 END) as discarded,
                 SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running,
                 SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                 SUM(COALESCE(cost_gpu_min, 0)) as total_gpu_min,
                 MAX(primary_metric) as best_primary
               FROM runs WHERE project = ?""",
            (project,),
        ).fetchone()
        return dict(row) if row else {}


def _new_id() -> str:
    """Time-sortable UUID (v7-style)."""
    return str(uuid.uuid4())
