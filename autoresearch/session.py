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
import time
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

        # Agentic loop state — plan, scratchpad, conclusion. The agent
        # owns these via tools (set_plan, note, conclude). All in-memory;
        # the conclusion is also written to lessons via add_lesson.
        self.plan: dict | None = None
        self.scratchpad: list[dict] = []
        self.concluded: bool = False
        self.conclusion: dict | None = None

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        ledger_path: str | Path = "session_ledger.db",
        budget: SessionBudget | None = None,
        local: bool = False,
        **kwargs: Any,
    ) -> Session:
        manifest = load_manifest(manifest_path)
        launcher = Launcher(ledger_path, manifest=manifest, budget=budget, local=local)
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

        # Plan — investigation goal/hypotheses, set by the agent.
        if self.plan:
            lines.append("## Current plan")
            lines.append(f"- Goal: {self.plan['goal']}")
            if self.plan.get("hypotheses"):
                lines.append("- Working hypotheses:")
                for h in self.plan["hypotheses"]:
                    lines.append(f"  - {h}")
            if self.plan.get("success_criteria"):
                lines.append(f"- Success criteria: {self.plan['success_criteria']}")
            n_rev = len(self.plan.get("revisions", []))
            if n_rev:
                last = self.plan["revisions"][-1]
                lines.append(f"- Plan revised {n_rev}x; latest reason: {last['reason']}")
            lines.append("")
        else:
            lines.append("## Current plan")
            lines.append("_No plan set. Call set_plan(goal, hypotheses, success_criteria)._")
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

        # In-session scratchpad — agent's working memory for this session.
        if self.scratchpad:
            recent_notes = self.scratchpad[-10:]
            lines.append(
                f"## Session notes ({len(self.scratchpad)} total"
                + (f", showing last {len(recent_notes)}" if len(self.scratchpad) > len(recent_notes) else "")
                + ")"
            )
            for n in recent_notes:
                lines.append(f"- {n['text']}")
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
    # Agentic loop state — plan / scratchpad / conclusion
    # ------------------------------------------------------------------

    def set_plan(
        self,
        goal: str,
        hypotheses: list[str] | None = None,
        success_criteria: str = "",
    ) -> dict:
        self.plan = {
            "goal": goal,
            "hypotheses": list(hypotheses or []),
            "success_criteria": success_criteria,
            "set_at": time.time(),
            "revisions": [],
        }
        return dict(self.plan)

    def update_plan(
        self,
        reason: str,
        goal: str | None = None,
        hypotheses: list[str] | None = None,
        success_criteria: str | None = None,
    ) -> dict:
        if self.plan is None:
            raise RuntimeError("No plan set; call set_plan first")
        self.plan["revisions"].append({
            "at": time.time(),
            "reason": reason,
            "previous_goal": self.plan["goal"],
            "previous_hypotheses": list(self.plan["hypotheses"]),
            "previous_success_criteria": self.plan["success_criteria"],
        })
        if goal is not None:
            self.plan["goal"] = goal
        if hypotheses is not None:
            self.plan["hypotheses"] = list(hypotheses)
        if success_criteria is not None:
            self.plan["success_criteria"] = success_criteria
        return dict(self.plan)

    def add_note(self, text: str) -> int:
        self.scratchpad.append({"at": time.time(), "text": text})
        return len(self.scratchpad)

    def conclude(self, summary: str, lessons: list[str] | None = None) -> dict:
        lesson_ids: list[str] = []
        for lesson_text in (lessons or []):
            lid = self.launcher.add_lesson(lesson_text)
            lesson_ids.append(lid)
        self.conclusion = {
            "summary": summary,
            "lessons": list(lessons or []),
            "lesson_ids": lesson_ids,
            "at": time.time(),
        }
        self.concluded = True
        return self.conclusion

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
            by proposing experiments, analyzing results, and iterating —
            until you reach a defensible conclusion or run out of useful
            things to try in this session.

            ## How this loop works
            You drive the session. Each turn you may:
              - Call tools to act (launch, wait, query, set_status, …)
              - Emit reasoning text to think through what to do next
              - Call `conclude(summary, lessons)` to END the session

            The loop keeps running until you call `conclude` or an
            external safety stop trips (budget exhausted, agent stuck,
            etc.). There is NO fixed turn budget driving you forward —
            pace yourself. Reasoning between actions is encouraged;
            there is no penalty for thinking before acting.

            Don't drag the session out, but don't bail prematurely
            either. When the evidence is clear or you've hit a wall,
            conclude with a summary that captures what was learned.

            ## Plan and notes (your working memory)
            - `set_plan(goal, hypotheses, success_criteria)` — declare
              what you're investigating this session. Call this near
              the start so future-you (and the human) can see the
              intent. Surfaced in context() output every turn.
            - `update_plan(reason, ...)` — revise when evidence shifts
              your strategy. The `reason` field is mandatory — articulate
              what changed your mind.
            - `note(text)` — append to the in-session scratchpad. Use
              for observations, intermediate conclusions, things to try
              next. Notes appear in context() so you stay coherent
              across many turns.
            - `add_lesson(text, evidence)` — for DURABLE insights that
              should survive past this session (recorded in the lessons
              table that future sessions will read).

            ## Phases
            {phases_desc}

            ## Metrics
            {metrics_desc}

            ## Available tools

            Plan & lifecycle:
              - set_plan(goal, hypotheses, success_criteria)
              - update_plan(reason, goal?, hypotheses?, success_criteria?)
              - note(text)
              - conclude(summary, lessons)

            Experiments:
              - launch(commit_sha, phase, config_overrides, hypothesis,
                       parent_run_id, track) — fire-and-forget; returns run_id
              - wait(run_id) — block until completion (preferred)
              - poll(run_id) — non-blocking status check
              - poll_all() — drain all completed in-flight runs
              - cancel(run_id) — kill an in-flight run

            Inspection:
              - context() — full session state (plan, notes, recent runs, lessons)
              - query(phase, status, track, limit)
              - best_runs(phase, n)
              - describe() — project's config schema
              - stats()

            Curation:
              - set_status(run_id, status, note) — keep / discard / review
              - add_lesson(text, evidence) — durable cross-session insight

            ## Rules
            - commit_sha must be a REAL git commit hash (e.g. "967130b",
              "7268a7c") — NOT a run_id like "tsv-0092". Look at the
              commit_sha field of existing runs in context/query output.
            - config_overrides must use keys from the project's config
              schema (call describe() to see valid keys and defaults).
            - Always start with cheap phases (low trust) before
              expensive ones.
            - Phases with gates_from require a parent_run_id from a
              passing run.
            - After launch(), call wait(run_id). Don't poll in a loop.
            - If a gate rejects your launch, read the reason and adjust.

            ## When to conclude
            Good moments:
              - You've answered the question your plan posed.
              - You've seen enough trend that more runs won't change
                your mind.
              - You've hit a wall and a different angle (or different
                session) is needed.
              - External stop fired — wrap up with a clean record.

            A good `conclude(summary, lessons)` includes:
              - What you set out to do (echo your plan).
              - What you actually found (cite run_ids).
              - Why you stopped now.
              - Lessons that should outlive this session.
        """)

    def close(self) -> None:
        self.launcher.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
