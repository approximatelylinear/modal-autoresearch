"""Session lifecycle for an autoresearch run.

A session owns a Launcher and manages:
  - Context building (what the agent sees before proposing)
  - Stop condition detection
  - Budget tracking
  - Transition between HITL and autonomous mode

Usage:
    session = Session.from_manifest("path/to/autoresearch.toml", budget=...)
    session.import_history("path/to/results.tsv")
    context = session.build_context()
    # ... agent proposes spec, session.launch(spec), etc. ...
"""

from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .gate import SessionBudget
from .launcher import ExperimentSpec, GateRejection, Launcher
from .manifest import Manifest, load_manifest


@dataclass
class StopCondition:
    triggered: bool
    reason: str

    @staticmethod
    def ok() -> StopCondition:
        return StopCondition(False, "")

    @staticmethod
    def stop(reason: str) -> StopCondition:
        return StopCondition(True, reason)


class Session:
    """Manages one autoresearch session (interactive or overnight)."""

    def __init__(
        self,
        manifest: Manifest,
        launcher: Launcher,
        *,
        no_improvement_limit: int = 15,
        max_consecutive_rejections: int = 5,
    ):
        self.manifest = manifest
        self.launcher = launcher
        self.no_improvement_limit = no_improvement_limit
        self.max_consecutive_rejections = max_consecutive_rejections
        self._consecutive_rejections = 0
        self._describe_cache: str | None = None
        # Snapshot of run count at session start, so budget caps count only
        # new runs launched this session (not historical imports).
        self._initial_run_count: int | None = None

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        ledger_path: str | Path = "session_ledger.db",
        budget: SessionBudget | None = None,
        **kwargs: Any,
    ) -> Session:
        manifest = load_manifest(manifest_path)
        launcher = Launcher(ledger_path, manifest=manifest, budget=budget)
        return cls(manifest, launcher, **kwargs)

    # ------------------------------------------------------------------
    # Context — what the agent sees before each proposal
    # ------------------------------------------------------------------

    def build_context(self, recent_n: int = 20, best_n: int = 5) -> str:
        """Build a text summary of the current state for the agent."""
        stats = self.launcher.stats()
        recent = self.launcher.query(limit=recent_n)
        best = self.launcher.best_runs(n=best_n)
        lessons = self.launcher.query_lessons()
        budget = self.launcher.budget

        lines: list[str] = []
        lines.append(f"## Project: {self.manifest.name}")
        lines.append(f"{self.manifest.description}")
        lines.append("")

        # Phases
        lines.append("## Phases")
        for name, ph in self.manifest.phases.items():
            gates = f" (requires passing {ph.gates_from} parent)" if ph.gates_from else ""
            lines.append(f"- **{name}**: {ph.description}{gates}")
        lines.append("")

        # Budget
        total = stats.get("total", 0)
        session_runs = total - self.initial_run_count
        lines.append("## Budget")
        lines.append(
            f"- New runs this session: {session_runs} / {budget.max_runs} "
            f"({total} total incl. history)"
        )
        gpu_used = stats.get("total_gpu_min", 0) or 0
        lines.append(f"- GPU: {gpu_used:.1f} / {budget.max_gpu_min:.1f} min")
        lines.append(f"- In-flight: {self.launcher.inflight_count}")
        lines.append("")

        # Stats
        lines.append("## Stats")
        lines.append(
            f"- Total: {stats.get('total', 0)} | "
            f"Kept: {stats.get('kept', 0)} | "
            f"Discarded: {stats.get('discarded', 0)} | "
            f"Failed: {stats.get('failed', 0)}"
        )
        best_metric = stats.get("best_primary")
        if best_metric is not None:
            lines.append(
                f"- Best {self.manifest.metrics.primary}: {best_metric:.4f}"
            )
        lines.append("")

        # Best runs
        if best:
            lines.append(f"## Top {len(best)} runs ({self.manifest.metrics.primary})")
            lines.append("(run_id | commit_sha | phase | metric | status | notes)")
            for r in best:
                pm = r.get("primary_metric")
                pm_str = f"{pm:.4f}" if pm is not None else "n/a"
                lines.append(
                    f"- run_id=`{r['run_id']}` "
                    f"commit_sha=`{r['commit_sha']}` "
                    f"phase={r['phase']} "
                    f"{self.manifest.metrics.primary}={pm_str} "
                    f"[{r['status']}] {r.get('observation', '')[:60]}"
                )
            lines.append("")

        # Recent runs
        if recent:
            lines.append(f"## Recent {len(recent)} runs")
            lines.append("(run_id | commit_sha | phase | metric | status)")
            for r in recent:
                pm = r.get("primary_metric")
                pm_str = f"{pm:.4f}" if pm is not None else "---"
                lines.append(
                    f"- run_id=`{r['run_id']}` "
                    f"commit_sha=`{r['commit_sha']}` "
                    f"{r['phase']} "
                    f"{self.manifest.metrics.primary}={pm_str} "
                    f"[{r['status']}]"
                )
            lines.append("")

        # Lessons
        if lessons:
            lines.append(f"## Lessons ({len(lessons)})")
            for l in lessons:
                lines.append(f"- {l['text']}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stop conditions
    # ------------------------------------------------------------------

    def check_stop(self) -> StopCondition:
        """Check whether the session should stop."""
        stats = self.launcher.stats()

        # Budget exhausted (runs) — count only new runs this session.
        total = stats.get("total", 0)
        session_runs = total - self.initial_run_count
        if session_runs >= self.launcher.budget.max_runs:
            return StopCondition.stop(
                f"Run limit reached: {session_runs} new runs "
                f"(limit {self.launcher.budget.max_runs})"
            )

        # Budget exhausted (GPU)
        gpu = stats.get("total_gpu_min", 0) or 0
        if gpu >= self.launcher.budget.max_gpu_min:
            return StopCondition.stop(
                f"GPU budget exhausted: {gpu:.1f} / {self.launcher.budget.max_gpu_min:.1f} min"
            )

        # Consecutive rejections
        if self._consecutive_rejections >= self.max_consecutive_rejections:
            return StopCondition.stop(
                f"Agent stuck: {self._consecutive_rejections} consecutive "
                f"gate rejections"
            )

        # No improvement in last K *session* experiments (skip if no new runs yet).
        if self.no_improvement_limit > 0 and session_runs >= self.no_improvement_limit:
            recent = self.launcher.query(limit=self.no_improvement_limit)
            best_overall = self.launcher.best_runs(n=1)
            if best_overall and recent:
                best_id = best_overall[0]["run_id"]
                recent_ids = {r["run_id"] for r in recent}
                if best_id not in recent_ids:
                    return StopCondition.stop(
                        f"No improvement in last {self.no_improvement_limit} "
                        f"experiments (best run {best_id} is older)"
                    )

        return StopCondition.ok()

    # ------------------------------------------------------------------
    # Convenience wrappers (track rejection streak)
    # ------------------------------------------------------------------

    def launch(self, spec: ExperimentSpec, **kwargs: Any) -> str:
        """Launch with rejection tracking for stop conditions."""
        try:
            run_id = self.launcher.launch(spec, **kwargs)
            self._consecutive_rejections = 0
            return run_id
        except GateRejection:
            self._consecutive_rejections += 1
            raise

    def import_history(self, tsv_path: str | Path) -> int:
        """Import historical runs from a TSV file."""
        n = self.launcher.ledger.import_tsv(
            tsv_path, project=self.manifest.name,
            primary_metric_key=self.manifest.metrics.primary,
        )
        # Update the baseline so budget caps only count new runs.
        self._snapshot_initial_count()
        return n

    def _snapshot_initial_count(self) -> None:
        """Record how many runs exist right now, so budget checks can
        distinguish historical imports from new launches."""
        stats = self.launcher.stats()
        self._initial_run_count = stats.get("total", 0)
        self.launcher.set_session_baseline(self._initial_run_count)

    @property
    def initial_run_count(self) -> int:
        if self._initial_run_count is None:
            self._snapshot_initial_count()
        return self._initial_run_count

    # ------------------------------------------------------------------
    # System prompt for LLM agents
    # ------------------------------------------------------------------

    def system_prompt(self) -> str:
        """Generate a system prompt for an LLM agent driving this session."""
        phases_desc = "\n".join(
            f"  - {name}: {ph.description}"
            + (f" (gates_from: {ph.gates_from})" if ph.gates_from else "")
            for name, ph in self.manifest.phases.items()
        )
        metrics_desc = "\n".join(
            f"  - {c.name}: {c.description}" for c in self.manifest.metrics.columns
        )

        return textwrap.dedent(f"""\
            You are an ML research agent running experiments on the
            "{self.manifest.name}" project: {self.manifest.description}

            ## Your goal
            Improve the primary metric ({self.manifest.metrics.primary},
            {"higher" if self.manifest.metrics.higher_is_better else "lower"} is better)
            by proposing experiments, analyzing results, and iterating.

            ## Phases
            {phases_desc}

            ## Metrics
            {metrics_desc}

            ## Available tools
            - launch(commit_sha, phase, config_overrides, hypothesis, parent_run_id, track)
              Launch an experiment. Returns run_id. Fire-and-forget.
            - poll(run_id) — Check if a run is done. Returns status + metrics if complete.
            - poll_all() — Drain all completed runs.
            - wait(run_id) — Block until a run completes.
            - query(phase, status, limit) — Query runs from the ledger.
            - best_runs(phase, n) — Top N runs by primary metric.
            - set_status(run_id, status, note) — Mark a run as keep/discard/review.
            - add_lesson(text, evidence) — Record a lesson learned.
            - cancel(run_id) — Cancel an in-flight run.
            - context() — Get the current session context summary.
            - describe() — Get the project's config schema (what overrides are valid).
            - stats() — Get session statistics.

            ## Rules
            - commit_sha must be a REAL git commit hash (e.g. "967130b",
              "7268a7c") — NOT a run_id like "tsv-0092". Look at the
              commit_sha field of existing runs in context/query output.
            - config_overrides must use keys from the project's config schema
              (call describe() to see valid keys and defaults).
            - Always start with cheap phases (low trust) before expensive ones.
            - Phases with gates_from require a parent_run_id from a passing run.
            - Record lessons when you notice patterns across runs.
            - Use set_status to mark runs as keep/discard/review with reasoning.
            - If a gate rejects your launch, read the reason and adjust.

            ## Strategy
            1. Review context to understand what's been tried.
            2. Form a hypothesis about what might improve the metric.
            3. Design a minimal experiment to test it (quick phase first).
            4. Launch, then IMMEDIATELY call wait(run_id) to block until
               the run completes. Do NOT poll in a loop — use wait().
            5. Analyze the result. Set status to keep/discard/review.
            6. If promising (keep), consider promoting to extensive phase.
            7. If not (discard), record why and try a different direction.
            8. Record lessons when you notice patterns across runs.
        """)

    def close(self) -> None:
        self.launcher.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
