"""Promotion gate — the most important guardrail in the autoresearch system.

Enforced inside the launcher, not trusted to the agent. All rules read
project-specific values (phase definitions, metrics, baseline) from the
manifest. The gate is a pure checker with no side effects: it returns
an allow/reject decision with a human-readable reason.

Rules:
  1. Phase gating: phases with `gates_from` require a passing parent run.
  2. Budget caps: per-session limits on GPU-minutes, total runs, and
     high-trust (expensive) phase runs.
  3. Auto-kill on collapse: implemented in the training function itself
     via progress.jsonl monitoring, not here. (Deferred.)
  4. Baseline freshness: after every N experiments, require a baseline
     rerun before continuing.

Usage:
    gate = PromotionGate(manifest, ledger, budget)
    result = gate.check(spec)
    if not result.allowed:
        print(f"Rejected: {result.reason}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .ledger import Ledger
from .manifest import Manifest


@dataclass
class SessionBudget:
    """Per-session resource limits. Set by the user/agent at session start.

    Run counts are relative to the session start (historical imports don't
    count). The gate receives `runs_at_session_start` to compute this.
    """
    max_gpu_min: float = float("inf")
    max_runs: int = 999
    max_high_trust_runs: int = 20     # phases with trust="high"
    baseline_rerun_interval: int = 20  # require baseline rerun every N runs
    primary_metric_epsilon: float = 0.005  # min acceptable primary metric delta vs baseline


@dataclass
class GateResult:
    allowed: bool
    reason: str = ""
    action: str = ""  # "baseline_rerun_required" | "" — tells caller what to do

    @staticmethod
    def allow() -> GateResult:
        return GateResult(allowed=True)

    @staticmethod
    def reject(reason: str, action: str = "") -> GateResult:
        return GateResult(allowed=False, reason=reason, action=action)


class PromotionGate:
    def __init__(
        self,
        manifest: Manifest,
        ledger: Ledger,
        budget: SessionBudget | None = None,
        runs_at_session_start: int = 0,
    ):
        self.manifest = manifest
        self.ledger = ledger
        self.budget = budget or SessionBudget()
        self.runs_at_session_start = runs_at_session_start

    def check(self, spec: Any) -> GateResult:
        """Check whether a proposed experiment is allowed to launch.

        Args:
            spec: an ExperimentSpec (from launcher.py) with commit_sha,
                  phase, parent_run_id, etc.

        Returns:
            GateResult with allowed=True or allowed=False + reason.
        """
        # ---- Rule 1: Phase gating ----
        result = self._check_phase_gate(spec)
        if not result.allowed:
            return result

        # ---- Rule 2: Budget caps ----
        result = self._check_budget(spec)
        if not result.allowed:
            return result

        # ---- Rule 4: Baseline freshness ----
        result = self._check_baseline_freshness()
        if not result.allowed:
            return result

        return GateResult.allow()

    # ------------------------------------------------------------------
    # Rule 1: Phase gating
    # ------------------------------------------------------------------

    def _check_phase_gate(self, spec: Any) -> GateResult:
        phase_name = spec.phase
        phase_def = self.manifest.phases.get(phase_name)
        if phase_def is None:
            return GateResult.reject(
                f"Unknown phase {phase_name!r}. "
                f"Available: {sorted(self.manifest.phases)}"
            )

        gates_from = phase_def.gates_from
        if not gates_from:
            # No gating required — this is a low-cost phase anyone can run.
            return GateResult.allow()

        # A parent_run_id is required and must satisfy conditions.
        parent_id = spec.parent_run_id
        if not parent_id:
            return GateResult.reject(
                f"Phase {phase_name!r} requires a parent run from "
                f"phases {gates_from} (gates_from). "
                f"Set parent_run_id on the spec."
            )

        parent = self.ledger.get_run(parent_id)
        if parent is None:
            return GateResult.reject(
                f"Parent run {parent_id!r} not found in ledger."
            )

        # Parent must be from an allowed gate phase.
        parent_phase = parent.get("phase", "")
        if parent_phase not in gates_from:
            return GateResult.reject(
                f"Parent run {parent_id!r} is phase {parent_phase!r}, "
                f"but {phase_name!r} requires a parent from {gates_from}."
            )

        # Parent must have an acceptable status.
        parent_status = parent.get("status", "")
        if parent_status not in ("keep", "review"):
            return GateResult.reject(
                f"Parent run {parent_id!r} has status {parent_status!r}. "
                f"Phase {phase_name!r} requires parent status keep or review."
            )

        # Parent's primary metric must be above baseline - epsilon.
        parent_metric = parent.get("primary_metric")
        if parent_metric is not None:
            baseline = self._get_baseline_metric()
            if baseline is not None:
                threshold = baseline - self.budget.primary_metric_epsilon
                if parent_metric < threshold:
                    return GateResult.reject(
                        f"Parent run {parent_id!r} primary_metric "
                        f"({parent_metric:.4f}) is below baseline "
                        f"({baseline:.4f}) - epsilon ({self.budget.primary_metric_epsilon}). "
                        f"Not promoting to {phase_name!r}."
                    )

        return GateResult.allow()

    # ------------------------------------------------------------------
    # Rule 2: Budget caps
    # ------------------------------------------------------------------

    def _check_budget(self, spec: Any) -> GateResult:
        stats = self.ledger.stats(self.manifest.name)
        total_runs = stats.get("total", 0)
        session_runs = total_runs - self.runs_at_session_start
        total_gpu = stats.get("total_gpu_min", 0) or 0

        # Session runs cap (excludes historical imports).
        if session_runs >= self.budget.max_runs:
            return GateResult.reject(
                f"Run limit reached: {session_runs} new runs "
                f"(limit {self.budget.max_runs})."
            )

        # Total GPU minutes cap.
        if total_gpu >= self.budget.max_gpu_min:
            return GateResult.reject(
                f"GPU budget exhausted: {total_gpu:.1f} / "
                f"{self.budget.max_gpu_min:.1f} minutes."
            )

        # High-trust phase cap.
        phase_def = self.manifest.phases.get(spec.phase)
        if phase_def and phase_def.trust == "high":
            high_trust_count = self._count_high_trust_runs()
            if high_trust_count >= self.budget.max_high_trust_runs:
                return GateResult.reject(
                    f"High-trust phase limit reached: {high_trust_count} / "
                    f"{self.budget.max_high_trust_runs}."
                )

        return GateResult.allow()

    def _count_high_trust_runs(self) -> int:
        high_trust_phases = [
            name for name, ph in self.manifest.phases.items()
            if ph.trust == "high"
        ]
        if not high_trust_phases:
            return 0
        count = 0
        for ph in high_trust_phases:
            runs = self.ledger.query(
                project=self.manifest.name, phase=ph, limit=9999
            )
            count += len(runs)
        return count

    # ------------------------------------------------------------------
    # Rule 4: Baseline freshness
    # ------------------------------------------------------------------

    def _check_baseline_freshness(self) -> GateResult:
        interval = self.budget.baseline_rerun_interval
        if interval <= 0:
            return GateResult.allow()

        stats = self.ledger.stats(self.manifest.name)
        total_runs = stats.get("total", 0)

        if total_runs == 0:
            return GateResult.allow()

        # Check if there's a baseline run in the last `interval` runs.
        recent = self.ledger.query(
            project=self.manifest.name,
            order_by="started_at DESC",
            limit=interval,
        )
        baseline_sha = (
            self.manifest.baseline.commit_sha if self.manifest.baseline else None
        )
        if not baseline_sha:
            return GateResult.allow()

        has_recent_baseline = any(
            r.get("commit_sha", "").startswith(baseline_sha)
            and r.get("track") == "baseline"
            for r in recent
        )

        if not has_recent_baseline and total_runs >= interval:
            return GateResult.reject(
                f"No baseline rerun in the last {interval} experiments. "
                f"Run the baseline ({baseline_sha}) before continuing.",
                action="baseline_rerun_required",
            )

        return GateResult.allow()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_baseline_metric(self) -> float | None:
        """Get the baseline's expected primary metric from the manifest."""
        if not self.manifest.baseline:
            return None
        expected = self.manifest.baseline.expected
        primary_key = self.manifest.metrics.primary
        val = expected.get(primary_key)
        return float(val) if val is not None else None
