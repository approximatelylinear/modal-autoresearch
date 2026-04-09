"""M5 validation: test session + tools integration.

Exercises the full stack: Session, Tools, context building, stop conditions,
system prompt generation, and tool dispatch. All mocked (no real Modal calls).

Usage:
    uv run python test_session.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

os.environ["AUTORESEARCH_MANIFEST"] = str(
    Path(__file__).parent.parent / "hydra" / "autoresearch.toml"
)

from autoresearch.gate import SessionBudget
from autoresearch.session import Session
from autoresearch.tools import TOOL_SCHEMAS, Tools


HYDRA_TSV = Path(__file__).parent.parent / "hydra" / "results.tsv"


def main() -> int:
    errors = 0

    def check(label: str, ok: bool):
        nonlocal errors
        print(f"  {label}: {'OK' if ok else 'FAIL'}")
        if not ok:
            errors += 1

    with tempfile.TemporaryDirectory() as td:
        budget = SessionBudget(max_runs=200, max_gpu_min=1000)
        session = Session.from_manifest(
            os.environ["AUTORESEARCH_MANIFEST"],
            ledger_path=Path(td) / "test.db",
            budget=budget,
        )
        tools = Tools(session)

        # Mock the run_phase function.
        session.launcher._run_phase_fn = MagicMock()
        mock_fc = MagicMock()
        mock_fc.get.return_value = {
            "status": "ok",
            "metrics": {"avg_ndcg10": 0.54, "scifact_ndcg10": 0.65},
            "cost": {"gpu_seconds": 600},
            "image_hash": "test123",
        }
        session.launcher._run_phase_fn.spawn.return_value = mock_fc

        # ---- 1. Import history ----
        print("1. Import history")
        n = session.import_history(HYDRA_TSV)
        check(f"imported {n} rows", n == 99)

        # ---- 2. Context building ----
        print("\n2. Context building")
        ctx = tools.context()
        check("context is string", isinstance(ctx, str))
        check("contains project name", "hydra" in ctx)
        check("contains budget info", "Runs:" in ctx)
        check("contains best runs", "Top" in ctx)
        check("contains lessons header", "Lessons" in ctx or "Recent" in ctx)

        # ---- 3. System prompt ----
        print("\n3. System prompt")
        sp = session.system_prompt()
        check("system prompt is string", isinstance(sp, str))
        check("mentions primary metric", "avg_ndcg10" in sp)
        check("mentions phases", "quick" in sp and "extensive" in sp)
        check("mentions tools", "launch" in sp and "poll" in sp)

        # ---- 4. Tool schemas ----
        print("\n4. Tool schemas")
        check(f"have {len(TOOL_SCHEMAS)} schemas", len(TOOL_SCHEMAS) >= 10)
        schema_names = {s["name"] for s in TOOL_SCHEMAS}
        check("launch in schemas", "launch" in schema_names)
        check("poll in schemas", "poll" in schema_names)
        check("set_status in schemas", "set_status" in schema_names)

        # ---- 5. Tool dispatch ----
        print("\n5. Tool dispatch")
        # stats
        result = tools.dispatch("stats", {})
        check("stats returns dict", isinstance(result, dict))
        check("stats.total=99", result.get("total") == 99)

        # best_runs
        result = tools.dispatch("best_runs", {"n": 3})
        check("best_runs returns list", isinstance(result, list))
        check("best_runs has 3", len(result) == 3)

        # query
        result = tools.dispatch("query", {"phase": "quick", "limit": 5})
        check("query returns list", isinstance(result, list))

        # launch
        result = tools.dispatch("launch", {
            "commit_sha": "abc123",
            "phase": "quick",
            "hypothesis": "test hypothesis",
        })
        check("launch returns run_id", "run_id" in result)
        run_id = result["run_id"]

        # poll (mock returns immediately)
        result = tools.dispatch("poll", {"run_id": run_id})
        check("poll returns dict", isinstance(result, dict))
        check("poll shows ok status", result.get("status") == "ok")

        # set_status
        result = tools.dispatch("set_status", {
            "run_id": run_id, "status": "keep", "note": "good result",
        })
        check("set_status returns status", result.get("status") == "keep")

        # add_lesson
        result = tools.dispatch("add_lesson", {
            "text": "Test lesson", "evidence": run_id,
        })
        check("add_lesson returns id", "lesson_id" in result)

        # unknown tool
        result = tools.dispatch("nonexistent", {})
        check("unknown tool returns error", "error" in result)

        # ---- 6. Stop conditions ----
        print("\n6. Stop conditions")

        # Not stopped yet.
        sc = session.check_stop()
        check("not stopped initially", not sc.triggered)

        # Simulate budget exhaustion.
        tight = Session.from_manifest(
            os.environ["AUTORESEARCH_MANIFEST"],
            ledger_path=Path(td) / "tight.db",
            budget=SessionBudget(max_runs=5),
        )
        for i in range(5):
            tight.launcher.ledger.insert_run(
                project="hydra", commit_sha=f"sha{i}", phase="quick",
            )
        sc = tight.check_stop()
        check("stops at run limit", sc.triggered)
        check("reason mentions limit", "limit" in sc.reason.lower())
        tight.close()

        # Consecutive rejections.
        session._consecutive_rejections = 5
        sc = session.check_stop()
        check("stops on consecutive rejections", sc.triggered)
        check("reason mentions stuck", "stuck" in sc.reason.lower())
        session._consecutive_rejections = 0  # reset

        # ---- 7. Context after actions ----
        print("\n7. Context after actions")
        ctx2 = tools.context()
        check("context shows new run", run_id[:8] in ctx2 or "100" in ctx2)
        check("context shows lesson", "Test lesson" in ctx2)

        session.close()

    print(f"\n{'PASS' if errors == 0 else 'FAIL'}: {errors} failures")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
