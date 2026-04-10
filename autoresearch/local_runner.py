"""Local GPU runner — runs experiments via subprocess instead of Modal.

Same CLI contract (spec.json → result.json), no image building, no
worktrees, no Modal overhead. Runs against the live project working tree.

Implements a FunctionCall-compatible interface so the launcher's poll/wait/
cancel all work identically in local and Modal modes.

Usage:
    runner = LocalRunner(manifest)
    fc = runner.spawn(spec_dict)
    result = fc.get()           # blocks
    result = fc.get(timeout=0)  # non-blocking, raises TimeoutError if not done
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from .manifest import Manifest


class LocalFunctionCall:
    """Mimics Modal's FunctionCall interface for local subprocess runs."""

    def __init__(self, proc: subprocess.Popen, result_path: Path, run_id: str):
        self._proc = proc
        self._result_path = result_path
        self._run_id = run_id
        self._result: dict | None = None
        self._error: Exception | None = None
        self._done = threading.Event()

        # Monitor the process in a background thread.
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def _monitor(self) -> None:
        """Wait for the subprocess to finish, then read the result."""
        try:
            self._proc.wait()
            if self._result_path.exists():
                self._result = json.loads(self._result_path.read_text())
            else:
                self._result = {
                    "run_id": self._run_id,
                    "status": "failed",
                    "metrics": {},
                    "cost": {},
                    "notes": f"CLI exited {self._proc.returncode} without writing result",
                }
        except Exception as e:
            self._error = e
            self._result = {
                "run_id": self._run_id,
                "status": "failed",
                "metrics": {},
                "cost": {},
                "notes": f"Local runner error: {e}",
            }
        finally:
            self._done.set()

    def get(self, timeout: float | None = None) -> dict:
        """Block until done. timeout=0 for non-blocking (raises TimeoutError)."""
        if timeout == 0:
            if not self._done.is_set():
                raise TimeoutError("Run still in progress")
        else:
            if not self._done.wait(timeout=timeout):
                raise TimeoutError(f"Timed out after {timeout}s")

        if self._error:
            raise self._error
        return self._result

    def cancel(self) -> None:
        """Kill the subprocess."""
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()


class LocalRunner:
    """Spawns experiments as local subprocesses instead of Modal functions."""

    def __init__(self, manifest: Manifest):
        self.manifest = manifest
        self._project_root = manifest.project_root

    def spawn(self, spec: dict) -> LocalFunctionCall:
        """Launch an experiment locally. Returns a FunctionCall-compatible handle."""
        run_id = spec["run_id"]

        # Create temp dirs for this run.
        run_dir = Path(tempfile.mkdtemp(prefix=f"autoresearch-{run_id}-"))
        log_dir = run_dir / "logs"
        ckpt_dir = run_dir / "checkpoints"
        log_dir.mkdir()
        ckpt_dir.mkdir()

        # Write spec.
        spec_full = {
            **spec,
            "checkpoint_dir": str(ckpt_dir),
            "log_dir": str(log_dir),
        }
        spec_path = run_dir / "spec.json"
        result_path = run_dir / "result.json"
        spec_path.write_text(json.dumps(spec_full))

        # Build command.
        cmd = list(self.manifest.entrypoint.command) + [
            "run",
            "--spec", str(spec_path),
            "--output", str(result_path),
        ]

        env = {
            **os.environ,
            "PYTHONPATH": f"{self._project_root}:" + os.environ.get("PYTHONPATH", ""),
        }

        print(f"  [local] $ {shlex.join(cmd)}", flush=True)
        print(f"  [local] cwd={self._project_root}  run_dir={run_dir}", flush=True)

        proc = subprocess.Popen(
            cmd,
            cwd=str(self._project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Stream stdout in a background thread.
        def _stream_output():
            for line in proc.stdout:
                sys.stdout.buffer.write(b"  [local] " + line)
                sys.stdout.buffer.flush()

        threading.Thread(target=_stream_output, daemon=True).start()

        return LocalFunctionCall(proc, result_path, run_id)
