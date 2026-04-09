"""M3 validation: test the Launcher's local code paths.

Does NOT spawn real Modal functions — tests the ledger integration,
spec construction, query/status/lesson flows, and inflight bookkeeping
with a mock function call.

Usage:
    uv run python test_launcher.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Set manifest before any autoresearch imports.
os.environ["AUTORESEARCH_MANIFEST"] = str(
    Path(__file__).parent.parent / "hydra" / "autoresearch.toml"
)

from autoresearch.launcher import ExperimentSpec, Launcher
from autoresearch.manifest import load_manifest


def main() -> int:
    manifest = load_manifest(os.environ["AUTORESEARCH_MANIFEST"])
    errors = 0

    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "test.db"
        launcher = Launcher(db, manifest=manifest)

        # ---- Test 1: launch + inflight tracking ----
        # Mock the run_phase function so we don't hit Modal.
        mock_fc = MagicMock()
        mock_fc.get.side_effect = TimeoutError  # still running
        launcher._run_phase_fn = MagicMock()
        launcher._run_phase_fn.spawn.return_value = mock_fc

        spec = ExperimentSpec(
            commit_sha="7268a7c",
            phase="quick",
            hypothesis="FiLM conditioning helps",
            config_overrides={"lr": 1e-4},
        )
        run_id = launcher.launch(spec)
        print(f"1. launch: run_id={run_id}")

        assert launcher.inflight_count == 1, "expected 1 inflight"
        row = launcher.get_run(run_id)
        assert row is not None, "run should exist in ledger"
        assert row["status"] == "running"
        assert row["phase"] == "quick"
        print(f"   ledger row: status={row['status']} phase={row['phase']} OK")

        # ---- Test 2: poll while still running ----
        polled = launcher.poll(run_id)
        assert polled["status"] == "running"
        assert launcher.inflight_count == 1
        print("2. poll (still running): OK")

        # ---- Test 3: poll after completion ----
        mock_result = {
            "status": "ok",
            "metrics": {
                "scifact_ndcg10": 0.6480,
                "fiqa_ndcg10": 0.3740,
                "avg_ndcg10": 0.5110,
                "task_vs_generic": 0.0026,
            },
            "cost": {"gpu_seconds": 612},
            "image_hash": "abc123",
            "notes": "converged at epoch 12",
        }
        mock_fc.get.side_effect = None
        mock_fc.get.return_value = mock_result

        polled = launcher.poll(run_id)
        assert polled["status"] == "ok", f"expected ok, got {polled['status']}"
        assert launcher.inflight_count == 0
        assert polled["primary_metric"] == 0.5110
        print(f"3. poll (completed): status={polled['status']} primary={polled['primary_metric']} OK")

        # ---- Test 4: launch a second run, cancel it ----
        mock_fc2 = MagicMock()
        mock_fc2.get.side_effect = TimeoutError
        launcher._run_phase_fn.spawn.return_value = mock_fc2

        spec2 = ExperimentSpec(
            commit_sha="df7be85",
            phase="quick",
            parent_run_id=run_id,
            hypothesis="Rank=128 shared residual",
        )
        run_id2 = launcher.launch(spec2)
        launcher.cancel(run_id2)
        row2 = launcher.get_run(run_id2)
        assert row2["status"] == "killed"
        assert launcher.inflight_count == 0
        print(f"4. cancel: status={row2['status']} OK")

        # ---- Test 5: query + best_runs ----
        runs = launcher.query(phase="quick")
        assert len(runs) == 2
        print(f"5. query(phase=quick): {len(runs)} runs OK")

        best = launcher.best_runs(phase="quick", n=1)
        assert len(best) == 1
        assert best[0]["run_id"] == run_id  # only completed run with primary_metric
        print(f"6. best_runs: {best[0]['run_id']} primary={best[0]['primary_metric']} OK")

        # ---- Test 6: set_status ----
        launcher.set_status(run_id, "keep", "confirmed improvement")
        row = launcher.get_run(run_id)
        assert row["status"] == "keep"
        print(f"7. set_status: {row['status']} OK")

        # ---- Test 7: lessons ----
        lid = launcher.add_lesson(
            "FiLM with trainable A/B collapses at extensive scale",
            evidence=run_id,
        )
        lessons = launcher.query_lessons()
        assert len(lessons) == 1
        print(f"8. lessons: {len(lessons)} lesson(s) OK")

        # ---- Test 8: stats ----
        stats = launcher.stats()
        assert stats["total"] == 2
        assert stats["kept"] == 1
        print(f"9. stats: total={stats['total']} kept={stats['kept']} OK")

        # ---- Test 9: poll_all drains multiple ----
        mock_fc3 = MagicMock()
        mock_fc3.get.return_value = {
            "status": "ok", "metrics": {"avg_ndcg10": 0.52},
            "cost": {"gpu_seconds": 500}, "image_hash": "def456",
        }
        mock_fc4 = MagicMock()
        mock_fc4.get.side_effect = TimeoutError  # still running

        launcher._run_phase_fn.spawn.side_effect = [mock_fc3, mock_fc4]
        r3 = launcher.launch(ExperimentSpec("aaa", "quick"))
        r4 = launcher.launch(ExperimentSpec("bbb", "quick"))
        assert launcher.inflight_count == 2

        completed = launcher.poll_all()
        assert len(completed) == 1  # only r3 completed
        assert launcher.inflight_count == 1  # r4 still running
        print(f"10. poll_all: {len(completed)} drained, {launcher.inflight_count} inflight OK")

        launcher.close()

    print(f"\nAll tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
