"""M4 validation: test promotion gate rules.

Exercises:
  Rule 1: Phase gating (gates_from)
  Rule 2: Budget caps (runs, GPU minutes, high-trust phases)
  Rule 4: Baseline freshness interval

Rule 3 (auto-kill on collapse) is deferred — it runs inside the training
function via progress.jsonl monitoring.

Usage:
    uv run python test_gate.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

os.environ["AUTORESEARCH_MANIFEST"] = str(
    Path(__file__).parent.parent / "hydra" / "autoresearch.toml"
)

from autoresearch.gate import GateResult, PromotionGate, SessionBudget
from autoresearch.launcher import ExperimentSpec, GateRejection, Launcher
from autoresearch.ledger import Ledger
from autoresearch.manifest import load_manifest


def main() -> int:
    manifest = load_manifest(os.environ["AUTORESEARCH_MANIFEST"])
    errors = 0

    def check(label: str, ok: bool):
        nonlocal errors
        status = "OK" if ok else "FAIL"
        print(f"  {label}: {status}")
        if not ok:
            errors += 1

    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "test.db"
        ledger = Ledger(db)

        # -----------------------------------------------------------
        # Rule 1: Phase gating
        # -----------------------------------------------------------
        print("Rule 1: Phase gating (gates_from)")

        budget = SessionBudget()
        gate = PromotionGate(manifest, ledger, budget)

        # 1a. Quick phase has no gates_from → always allowed.
        spec_quick = ExperimentSpec("abc123", "quick")
        r = gate.check(spec_quick)
        check("1a. quick (no gates): allowed", r.allowed)

        # 1b. Extensive requires a quick parent → rejected without parent.
        spec_ext = ExperimentSpec("abc123", "extensive")
        r = gate.check(spec_ext)
        check("1b. extensive without parent: rejected", not r.allowed)
        check("1b. reason mentions gates_from", "gates_from" in r.reason)

        # 1c. Extensive with a quick parent that has status=keep → allowed.
        parent_id = ledger.insert_run(
            project="hydra", commit_sha="abc123", phase="quick",
        )
        # Simulate completion with good metrics.
        ledger.complete_run(parent_id, {
            "status": "ok",
            "metrics": {"avg_ndcg10": 0.52},
            "cost": {"gpu_seconds": 600},
        }, primary_metric_key="avg_ndcg10")
        ledger.set_status(parent_id, "keep")

        spec_ext_ok = ExperimentSpec(
            "abc123", "extensive", parent_run_id=parent_id,
        )
        r = gate.check(spec_ext_ok)
        check("1c. extensive with keep parent: allowed", r.allowed)

        # 1d. Extensive with a quick parent that has status=discard → rejected.
        parent_id2 = ledger.insert_run(
            project="hydra", commit_sha="def456", phase="quick",
        )
        ledger.complete_run(parent_id2, {
            "status": "ok",
            "metrics": {"avg_ndcg10": 0.50},
            "cost": {"gpu_seconds": 500},
        }, primary_metric_key="avg_ndcg10")
        ledger.set_status(parent_id2, "discard")

        spec_ext_bad = ExperimentSpec(
            "def456", "extensive", parent_run_id=parent_id2,
        )
        r = gate.check(spec_ext_bad)
        check("1d. extensive with discarded parent: rejected", not r.allowed)
        check("1d. reason mentions status", "discard" in r.reason)

        # 1e. Extensive with a parent that's an extensive run → rejected
        # (gates_from is ["quick"], not ["extensive"]).
        parent_id3 = ledger.insert_run(
            project="hydra", commit_sha="ghi789", phase="extensive",
        )
        ledger.complete_run(parent_id3, {
            "status": "ok", "metrics": {"avg_ndcg10": 0.49},
            "cost": {"gpu_seconds": 7200},
        }, primary_metric_key="avg_ndcg10")
        ledger.set_status(parent_id3, "keep")

        spec_ext_wrong = ExperimentSpec(
            "ghi789", "extensive", parent_run_id=parent_id3,
        )
        r = gate.check(spec_ext_wrong)
        check("1e. extensive with extensive parent: rejected", not r.allowed)

        # 1f. Unknown phase → rejected.
        spec_bad = ExperimentSpec("abc123", "nonexistent")
        r = gate.check(spec_bad)
        check("1f. unknown phase: rejected", not r.allowed)

        # -----------------------------------------------------------
        # Rule 2: Budget caps
        # -----------------------------------------------------------
        print("\nRule 2: Budget caps")

        # 2a. Max runs.
        tight_budget = SessionBudget(max_runs=4)  # we already have 3 runs
        gate2 = PromotionGate(manifest, ledger, tight_budget)
        r = gate2.check(spec_quick)
        check("2a. under max_runs (3/4): allowed", r.allowed)

        # Add one more run to hit the cap.
        ledger.insert_run(
            project="hydra", commit_sha="jkl000", phase="quick",
        )
        r = gate2.check(spec_quick)
        check("2a. at max_runs (4/4): rejected", not r.allowed)
        check("2a. reason mentions limit", "limit" in r.reason.lower())

        # 2b. Max GPU minutes.
        gpu_budget = SessionBudget(max_gpu_min=20)  # we have ~138 min total
        gate3 = PromotionGate(manifest, ledger, gpu_budget)
        r = gate3.check(spec_quick)
        check("2b. GPU budget exceeded: rejected", not r.allowed)

        # 2c. Max high-trust runs.
        ht_budget = SessionBudget(max_high_trust_runs=1)
        gate4 = PromotionGate(manifest, ledger, ht_budget)
        # We have 1 extensive run (parent_id3), which is high-trust.
        spec_ext_ht = ExperimentSpec(
            "abc123", "extensive", parent_run_id=parent_id,
        )
        r = gate4.check(spec_ext_ht)
        check("2c. high-trust limit (1/1): rejected", not r.allowed)
        check("2c. reason mentions high-trust", "high-trust" in r.reason.lower())

        # Quick is low trust → allowed under this budget.
        r = gate4.check(spec_quick)
        check("2c. quick (low trust) still allowed", r.allowed)

        # -----------------------------------------------------------
        # Rule 4: Baseline freshness
        # -----------------------------------------------------------
        print("\nRule 4: Baseline freshness")

        # 4a. With interval=3 and 4 runs but no baseline → rejected.
        bf_budget = SessionBudget(baseline_rerun_interval=3)
        gate5 = PromotionGate(manifest, ledger, bf_budget)
        r = gate5.check(spec_quick)
        check("4a. no baseline in last 3 runs: rejected", not r.allowed)
        check("4a. action=baseline_rerun_required",
              r.action == "baseline_rerun_required")

        # 4b. Add a baseline run → allowed.
        bl_id = ledger.insert_run(
            project="hydra", commit_sha="0597b29", phase="quick",
            track="baseline",
        )
        r = gate5.check(spec_quick)
        check("4b. baseline present in recent runs: allowed", r.allowed)

        # -----------------------------------------------------------
        # Integration: launcher.launch() enforces the gate
        # -----------------------------------------------------------
        print("\nIntegration: launcher enforces gate")

        launcher = Launcher(
            Path(td) / "launcher.db",
            manifest=manifest,
            budget=SessionBudget(max_runs=2),
        )
        launcher._run_phase_fn = MagicMock()
        launcher._run_phase_fn.spawn.return_value = MagicMock()

        # First two launches succeed.
        r1 = launcher.launch(spec_quick)
        r2 = launcher.launch(spec_quick)
        check("int. first 2 launches: OK", True)

        # Third launch should raise GateRejection.
        try:
            launcher.launch(spec_quick)
            check("int. third launch rejected: FAIL (no exception)", False)
        except GateRejection as e:
            check("int. third launch rejected", True)
            check("int. reason mentions limit", "limit" in e.result.reason.lower())

        # skip_gate=True bypasses.
        r3 = launcher.launch(spec_quick, skip_gate=True)
        check("int. skip_gate bypasses", r3 is not None)

        launcher.close()
        ledger.close()

    print(f"\n{'PASS' if errors == 0 else 'FAIL'}: {errors} failures")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
