"""Agent tool definitions for autoresearch.

Each function wraps a Launcher/Session method with:
  - A clear docstring the LLM can read
  - Typed parameters
  - A dict return value the LLM can parse

These are framework-agnostic: they work as plain function calls, as MCP
tool handlers, or as OpenAI-style function definitions. The TOOL_SCHEMAS
list at the bottom provides machine-readable schemas for LLM tool-calling.

Usage:
    tools = Tools(session)
    result = tools.launch(commit_sha="abc", phase="quick", hypothesis="...")
    result = tools.poll("run-id")
    result = tools.context()
"""

from __future__ import annotations

import json
import time
from typing import Any

from .launcher import ExperimentSpec, GateRejection
from .session import Session


class Tools:
    """Agent-facing tool surface wrapping a Session."""

    def __init__(self, session: Session):
        self.session = session

    # ------------------------------------------------------------------
    # Plan & lifecycle — agent's working memory and self-termination
    # ------------------------------------------------------------------

    def set_plan(
        self,
        goal: str,
        hypotheses: list[str] | None = None,
        success_criteria: str = "",
    ) -> dict:
        """Declare the investigation plan for this session.

        Call near the start so the goal is visible in context() every
        turn. Use update_plan() to revise as evidence comes in.

        Args:
            goal: What you're trying to learn or improve, in one sentence.
            hypotheses: Working theories you'll test (list of strings).
            success_criteria: When you'd consider this investigation done.
        """
        return self.session.set_plan(goal, hypotheses, success_criteria)

    def update_plan(
        self,
        reason: str,
        goal: str | None = None,
        hypotheses: list[str] | None = None,
        success_criteria: str | None = None,
    ) -> dict:
        """Revise the plan in light of new evidence.

        Args:
            reason: Why you're updating — what shifted your thinking.
                    This is mandatory; it forces explicit articulation.
            goal, hypotheses, success_criteria: Omit any field to keep
                its current value.
        """
        try:
            return self.session.update_plan(
                reason, goal=goal, hypotheses=hypotheses,
                success_criteria=success_criteria,
            )
        except RuntimeError as e:
            return {"error": str(e)}

    def note(self, text: str) -> dict:
        """Append to the in-session scratchpad (working memory).

        Use for observations, intermediate conclusions, things to try
        next. Notes appear in context() output so you stay coherent
        across many turns. Distinct from add_lesson(), which is for
        DURABLE cross-session insights.
        """
        n = self.session.add_note(text)
        return {"note_count": n, "text": text}

    def conclude(self, summary: str, lessons: list[str] | None = None) -> dict:
        """End this session with a summary.

        Call when you've reached a defensible conclusion, hit a wall
        not worth pushing further, or run out of useful things to try.
        The agent loop exits cleanly after this call.

        Args:
            summary: What you set out to do, what you found (with
                     run_ids), and why you stopped now.
            lessons: List of durable lessons to record. Each becomes
                     a row in the cross-session lessons table that
                     future sessions will see.
        """
        return self.session.conclude(summary, lessons)

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------

    def launch(
        self,
        commit_sha: str,
        phase: str,
        hypothesis: str = "",
        config_overrides: dict | None = None,
        parent_run_id: str = "",
        track: str = "architecture",
    ) -> dict:
        """Launch an experiment on Modal.

        Args:
            commit_sha: Git commit to test.
            phase: Validation phase (e.g. "quick", "extensive").
            hypothesis: What you expect to happen and why.
            config_overrides: Training config overrides (see describe()).
            parent_run_id: Required for phases with gates_from.
            track: Category label (default "architecture").

        Returns:
            {"run_id": str, "status": "launched"} on success, or
            {"error": str, "reason": str} if the gate rejects.
        """
        spec = ExperimentSpec(
            commit_sha=commit_sha,
            phase=phase,
            hypothesis=hypothesis,
            config_overrides=config_overrides or {},
            parent_run_id=parent_run_id,
            track=track,
        )
        try:
            print(f"  [modal] Spawning run on Modal (phase={phase}, sha={commit_sha[:7]})...",
                  flush=True)
            run_id = self.session.launch(spec)
            print(f"  [modal] Spawned: {run_id}", flush=True)
            return {"run_id": run_id, "status": "launched"}
        except GateRejection as e:
            return {"error": "gate_rejected", "reason": e.result.reason,
                    "action": e.result.action}

    def poll(self, run_id: str) -> dict:
        """Check if a run is done. Non-blocking.

        Returns the ledger row with status, metrics (if complete), cost, etc.
        """
        return self.session.launcher.poll(run_id)

    def poll_all(self) -> dict:
        """Drain all completed in-flight runs.

        Returns {"completed": [...], "still_running": int}.
        """
        completed = self.session.launcher.poll_all()
        return {
            "completed": completed,
            "still_running": self.session.launcher.inflight_count,
        }

    def wait(self, run_id: str) -> dict:
        """Block until a run completes. Returns the final result."""
        print(f"  [modal] Waiting for {run_id} to complete "
              f"(Ctrl+C to stop waiting)...", flush=True)
        t0 = time.time()
        try:
            result = self.session.launcher.wait(run_id)
        except KeyboardInterrupt:
            elapsed = time.time() - t0
            print(f"\n  [modal] Wait interrupted after {elapsed:.0f}s. "
                  f"Run {run_id} may still be running on Modal.", flush=True)
            return self.session.launcher.poll(run_id)
        elapsed = time.time() - t0
        status = result.get("status", "?")
        pm = result.get("primary_metric")
        pm_str = f" primary={pm:.4f}" if pm is not None else ""
        print(f"  [modal] {run_id} completed: status={status}{pm_str} ({elapsed:.0f}s)",
              flush=True)
        return result

    def query(
        self,
        phase: str | None = None,
        status: str | None = None,
        track: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Query runs from the ledger with optional filters.

        Returns list of run dicts, newest first.
        """
        return self.session.launcher.query(
            phase=phase, status=status, track=track, limit=limit,
        )

    def best_runs(self, phase: str | None = None, n: int = 5) -> list[dict]:
        """Get the top N runs by primary metric.

        Useful for understanding the current frontier before proposing.
        """
        return self.session.launcher.best_runs(phase=phase, n=n)

    def set_status(self, run_id: str, status: str, note: str = "") -> dict:
        """Mark a run as keep, discard, or review.

        Args:
            run_id: The run to update.
            status: One of "keep", "discard", "review".
            note: Why (important for the lessons system).

        Returns:
            {"run_id": str, "status": str}.
        """
        self.session.launcher.set_status(run_id, status, note)
        return {"run_id": run_id, "status": status}

    def add_lesson(self, text: str, evidence: str = "") -> dict:
        """Record a lesson learned across experiments.

        Good lessons are specific, actionable, and reference evidence:
        "FiLM with trainable A/B collapses at extensive scale (evidence: run-xyz)"

        Args:
            text: The lesson.
            evidence: Run IDs or other evidence supporting it.

        Returns:
            {"lesson_id": str}.
        """
        lid = self.session.launcher.add_lesson(text, evidence)
        return {"lesson_id": lid}

    def cancel(self, run_id: str) -> dict:
        """Cancel an in-flight run.

        Returns {"run_id": str, "status": "killed"}.
        """
        self.session.launcher.cancel(run_id)
        return {"run_id": run_id, "status": "killed"}

    def context(self) -> str:
        """Get a text summary of the current session state.

        Includes: project info, budget, stats, top runs, recent runs, lessons.
        Call this before proposing to stay oriented.
        """
        return self.session.build_context()

    def describe(self) -> str:
        """Get the project's config schema — what config_overrides are valid.

        Returns the output of the project CLI's `describe` command.
        """
        # TODO: actually call the project CLI's describe subcommand via Modal.
        # For now, return the phase defaults from the manifest.
        from .run_phase import manifest
        phases_info = {}
        for name, ph in manifest.phases.items():
            phases_info[name] = {
                "description": ph.description,
                "default_gpu": ph.default_gpu,
                "trust": ph.trust,
                "gates_from": ph.gates_from,
            }
        return json.dumps({"phases": phases_info}, indent=2)

    def stats(self) -> dict:
        """Get session statistics: total runs, kept, discarded, GPU-min, etc."""
        return self.session.launcher.stats()

    def check_stop(self) -> dict:
        """Check whether the session should stop.

        Returns {"should_stop": bool, "reason": str}.
        """
        sc = self.session.check_stop()
        return {"should_stop": sc.triggered, "reason": sc.reason}

    # ------------------------------------------------------------------
    # Dispatch — call a tool by name (for REPL / agent integration)
    # ------------------------------------------------------------------

    def dispatch(self, tool_name: str, kwargs: dict) -> Any:
        """Call a tool by name with keyword arguments."""
        fn = getattr(self, tool_name, None)
        if fn is None or tool_name.startswith("_"):
            return {"error": f"unknown tool: {tool_name}"}
        try:
            return fn(**kwargs)
        except TypeError as e:
            return {"error": f"bad arguments for {tool_name}: {e}"}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def tool_names(self) -> list[str]:
        """List available tool names."""
        return [
            "set_plan", "update_plan", "note", "conclude",
            "launch", "poll", "poll_all", "wait", "query", "best_runs",
            "set_status", "add_lesson", "cancel", "context", "describe",
            "stats", "check_stop",
        ]


# ------------------------------------------------------------------
# Machine-readable tool schemas (OpenAI function-calling style)
# ------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "set_plan",
        "description": (
            "Declare the investigation plan for this session. Call near "
            "the start so the goal is visible in context() every turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "What you're trying to learn or improve, in one sentence.",
                },
                "hypotheses": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Working theories you'll test.",
                },
                "success_criteria": {
                    "type": "string",
                    "description": "When you'd consider this investigation done.",
                },
            },
            "required": ["goal"],
        },
    },
    {
        "name": "update_plan",
        "description": (
            "Revise the plan in light of new evidence. Omit any field to "
            "keep its current value; `reason` is mandatory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "What shifted your thinking.",
                },
                "goal": {"type": "string"},
                "hypotheses": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "success_criteria": {"type": "string"},
            },
            "required": ["reason"],
        },
    },
    {
        "name": "note",
        "description": (
            "Append to the in-session scratchpad. Use for observations, "
            "intermediate conclusions, things to try next. Notes appear "
            "in context() output. Distinct from add_lesson()."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "conclude",
        "description": (
            "End this session with a summary. Call when you've reached a "
            "defensible conclusion, hit a wall, or run out of useful "
            "things to try. The loop exits cleanly after this call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": (
                        "What you set out to do, what you found (cite "
                        "run_ids), and why you stopped now."
                    ),
                },
                "lessons": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Durable lessons to record in the cross-session "
                        "lessons table."
                    ),
                },
            },
            "required": ["summary"],
        },
    },
    {
        "name": "launch",
        "description": "Launch an experiment on Modal. Returns run_id or gate rejection.",
        "parameters": {
            "type": "object",
            "properties": {
                "commit_sha": {"type": "string", "description": "Git commit to test"},
                "phase": {"type": "string", "description": "Validation phase (quick, extensive, etc.)"},
                "hypothesis": {"type": "string", "description": "What you expect and why"},
                "config_overrides": {"type": "object", "description": "Training config overrides"},
                "parent_run_id": {"type": "string", "description": "Required for gated phases"},
                "track": {"type": "string", "description": "Category (default: architecture)"},
            },
            "required": ["commit_sha", "phase"],
        },
    },
    {
        "name": "poll",
        "description": "Check if a run is done (non-blocking). Returns status + metrics.",
        "parameters": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"],
        },
    },
    {
        "name": "poll_all",
        "description": "Drain all completed in-flight runs.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "wait",
        "description": "Block until a run completes. Returns final result.",
        "parameters": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"],
        },
    },
    {
        "name": "query",
        "description": "Query runs with optional filters (phase, status, track, limit).",
        "parameters": {
            "type": "object",
            "properties": {
                "phase": {"type": "string"},
                "status": {"type": "string"},
                "track": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
            },
        },
    },
    {
        "name": "best_runs",
        "description": "Top N runs by primary metric.",
        "parameters": {
            "type": "object",
            "properties": {
                "phase": {"type": "string"},
                "n": {"type": "integer", "default": 5},
            },
        },
    },
    {
        "name": "set_status",
        "description": "Mark a run as keep/discard/review with reasoning.",
        "parameters": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "status": {"type": "string", "enum": ["keep", "discard", "review"]},
                "note": {"type": "string"},
            },
            "required": ["run_id", "status"],
        },
    },
    {
        "name": "add_lesson",
        "description": "Record a lesson learned across experiments.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "evidence": {"type": "string"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "cancel",
        "description": "Cancel an in-flight run.",
        "parameters": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"],
        },
    },
    {
        "name": "context",
        "description": "Get current session state: budget, stats, top runs, recent runs, lessons.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "describe",
        "description": "Get the project's config schema (what overrides are valid).",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "stats",
        "description": "Session statistics: total runs, kept, discarded, GPU minutes.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "check_stop",
        "description": "Check if the session should stop (budget, no improvement, etc.).",
        "parameters": {"type": "object", "properties": {}},
    },
]
