"""Launcher: wraps run_phase (M1) + ledger (M2) into the tool surface pi calls.

The launcher runs **locally** (on the user's machine or inside a pi agent).
It spawns Modal functions for GPU work and owns a local SQLite ledger for
bookkeeping. The Modal-side run_phase also writes to a Volume ledger as a
backup, but the launcher's local copy is the primary.

Key design: `launch()` is fire-and-forget. It inserts a `running` row in
the local ledger, calls `run_phase.spawn(...)`, stores the function-call
handle in memory, and returns immediately. `poll()` / `poll_all()` drain
completed calls and update the ledger. This enables parallel sweeps and
async agent loops.

Usage:
    import os
    os.environ["AUTORESEARCH_MANIFEST"] = "path/to/autoresearch.toml"

    from autoresearch.launcher import Launcher

    launcher = Launcher("./ledger.db")
    run_id = launcher.launch(
        commit_sha="7268a7c",
        phase="quick",
        hypothesis="FiLM conditioning helps",
    )
    # ... do other work ...
    completed = launcher.poll_all()
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .gate import GateResult, PromotionGate, SessionBudget
from .ledger import Ledger
from .manifest import Manifest


class GateRejection(Exception):
    """Raised when the promotion gate rejects a launch."""
    def __init__(self, result: GateResult):
        self.result = result
        super().__init__(result.reason)


@dataclass
class ExperimentSpec:
    """What the agent passes to launch(). Maps directly to run_phase's spec dict."""
    commit_sha: str
    phase: str
    config_overrides: dict = field(default_factory=dict)
    parent_run_id: str = ""
    hypothesis: str = ""
    track: str = "architecture"

    def to_spec_dict(self, run_id: str) -> dict:
        return {
            "run_id": run_id,
            "phase": self.phase,
            "commit_sha": self.commit_sha,
            "config_overrides": self.config_overrides,
            "parent_run_id": self.parent_run_id,
            "hypothesis": self.hypothesis,
            "track": self.track,
        }


class Launcher:
    """Local orchestration layer over Modal run_phase + SQLite ledger."""

    def __init__(
        self,
        ledger_path: str | Path,
        manifest: Manifest | None = None,
        budget: SessionBudget | None = None,
    ):
        self.ledger = Ledger(ledger_path)
        self.budget = budget or SessionBudget()
        # Lazy-import manifest and run_phase to avoid triggering Modal image
        # construction until actually needed. Callers can also pass manifest.
        self._manifest = manifest
        self._run_phase_fn = None
        self._gate: PromotionGate | None = None
        # In-flight function calls keyed by run_id. These are ephemeral —
        # if the launcher process dies, they're lost (but the Volume ledger
        # inside Modal still has the records).
        self._inflight: dict[str, Any] = {}  # run_id -> modal.functions.FunctionCall

    @property
    def manifest(self) -> Manifest:
        if self._manifest is None:
            from .run_phase import manifest as _m
            self._manifest = _m
        return self._manifest

    @property
    def _run_phase(self):
        if self._run_phase_fn is None:
            from .run_phase import run_phase as _rp
            self._run_phase_fn = _rp
        return self._run_phase_fn

    @property
    def gate(self) -> PromotionGate:
        if self._gate is None:
            self._gate = PromotionGate(self.manifest, self.ledger, self.budget)
        return self._gate

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def check_gate(self, spec: ExperimentSpec) -> GateResult:
        """Check whether a spec would pass the promotion gate without launching."""
        return self.gate.check(spec)

    def launch(self, spec: ExperimentSpec, *, skip_gate: bool = False) -> str:
        """Fire-and-forget: check gate, insert running row, spawn Modal function.

        Raises GateRejection if the promotion gate rejects the spec.
        Pass skip_gate=True to bypass (e.g. for baseline reruns).
        """
        if not skip_gate:
            result = self.gate.check(spec)
            if not result.allowed:
                raise GateRejection(result)

        run_id = _new_run_id()

        self.ledger.insert_run(
            run_id=run_id,
            project=self.manifest.name,
            commit_sha=spec.commit_sha,
            phase=spec.phase,
            config_overrides=spec.config_overrides,
            parent_run_id=spec.parent_run_id,
            track=spec.track,
            hypothesis=spec.hypothesis,
        )

        fc = self._run_phase.spawn(spec.to_spec_dict(run_id))
        self._inflight[run_id] = fc
        return run_id

    # ------------------------------------------------------------------
    # Poll
    # ------------------------------------------------------------------

    def poll(self, run_id: str) -> dict:
        """Check a single run. If the Modal call completed, drain it and
        update the ledger. Returns the ledger row (dict)."""
        fc = self._inflight.get(run_id)
        if fc is not None:
            self._try_drain(run_id, fc)
        return self.ledger.get_run(run_id) or {"run_id": run_id, "status": "unknown"}

    def poll_all(self) -> list[dict]:
        """Drain all completed in-flight calls. Returns list of newly
        completed run dicts."""
        completed = []
        for run_id in list(self._inflight):
            fc = self._inflight[run_id]
            result = self._try_drain(run_id, fc)
            if result is not None:
                completed.append(self.ledger.get_run(run_id))
        return completed

    def _try_drain(self, run_id: str, fc: Any) -> dict | None:
        """Non-blocking attempt to read a function call result.
        Returns the result dict if done, None if still running."""
        try:
            result = fc.get(timeout=0)
        except TimeoutError:
            return None
        except Exception as e:
            # Function crashed / was cancelled.
            result = {
                "status": "failed",
                "metrics": {},
                "cost": {},
                "notes": f"Modal function error: {e}",
            }

        self.ledger.complete_run(
            run_id, result,
            primary_metric_key=self.manifest.metrics.primary,
        )
        del self._inflight[run_id]
        return result

    def wait(self, run_id: str) -> dict:
        """Block until a run completes. Returns the result dict."""
        fc = self._inflight.get(run_id)
        if fc is None:
            return self.ledger.get_run(run_id) or {}

        try:
            result = fc.get()  # blocks
        except Exception as e:
            result = {
                "status": "failed",
                "metrics": {},
                "cost": {},
                "notes": f"Modal function error: {e}",
            }

        self.ledger.complete_run(
            run_id, result,
            primary_metric_key=self.manifest.metrics.primary,
        )
        del self._inflight[run_id]
        return result

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def cancel(self, run_id: str) -> None:
        """Cancel an in-flight run."""
        fc = self._inflight.pop(run_id, None)
        if fc is not None:
            try:
                fc.cancel()
            except Exception:
                pass
        self.ledger.set_status(run_id, "killed", "cancelled by launcher")

    # ------------------------------------------------------------------
    # Query (delegate to ledger)
    # ------------------------------------------------------------------

    def query(self, **filters: Any) -> list[dict]:
        return self.ledger.query(project=self.manifest.name, **filters)

    def best_runs(
        self, phase: str | None = None, n: int = 5
    ) -> list[dict]:
        return self.ledger.best_runs(
            self.manifest.name,
            phase=phase,
            n=n,
            higher_is_better=self.manifest.metrics.higher_is_better,
        )

    def get_run(self, run_id: str) -> dict | None:
        return self.ledger.get_run(run_id)

    def set_status(self, run_id: str, status: str, note: str = "") -> None:
        self.ledger.set_status(run_id, status, note)

    # ------------------------------------------------------------------
    # Lessons
    # ------------------------------------------------------------------

    def add_lesson(self, text: str, evidence: str = "") -> str:
        return self.ledger.add_lesson(text, evidence)

    def query_lessons(self) -> list[dict]:
        return self.ledger.query_lessons()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return self.ledger.stats(self.manifest.name)

    @property
    def inflight_count(self) -> int:
        return len(self._inflight)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.ledger.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _new_run_id() -> str:
    return str(uuid.uuid4())[:12]
