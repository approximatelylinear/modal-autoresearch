"""Generic Modal Function that runs one experiment phase for any project.

Module-scope app pattern: the manifest path is read from the
`AUTORESEARCH_MANIFEST` env var at import time, and the app/image/function
are constructed at module scope so Modal can import them by qualified name.

Driver scripts (replay_baseline.py, etc.) set the env var *before* importing
this module:

    os.environ["AUTORESEARCH_MANIFEST"] = "../hydra/autoresearch.toml"
    from autoresearch.run_phase import app, run_phase

Architecture:
  - One persistent Volume holds bare clones at /repos/<project>/git
  - On every call, the launcher mounts the *local* project repo at /seed
  - Inside the function:
      1. Ensure the bare clone exists (clone /seed -> /repos/<project>/git)
      2. Fetch new commits from /seed into the bare clone
      3. `git worktree add /work/wt-<run_id> <commit_sha>` for isolation
      4. Write the spec JSON, invoke the project's CLI per the manifest,
         read the result JSON
      5. Capture the resolved-dep image_hash, return everything
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

from .image import image_from_manifest, project_source_dir
from .manifest import load_manifest

# ---------------------------------------------------------------------------
# Module-scope construction (driven by AUTORESEARCH_MANIFEST env var)
# ---------------------------------------------------------------------------

# Path resolution: locally, the driver script sets AUTORESEARCH_MANIFEST to
# a host path. Inside the Modal container, the image bakes the manifest at
# /work/autoresearch.toml and sets the env var via .env() — so the same code
# loads the manifest from the right place in both environments.
_CONTAINER_MANIFEST_PATH = "/work/autoresearch.toml"
_MANIFEST_PATH = os.environ.get("AUTORESEARCH_MANIFEST")
if not _MANIFEST_PATH:
    raise RuntimeError(
        "autoresearch.run_phase: set AUTORESEARCH_MANIFEST=path/to/autoresearch.toml "
        "before importing this module."
    )

manifest = load_manifest(_MANIFEST_PATH)

if modal.is_local():
    # Local: build the deps image, bake the manifest into it, mount the live
    # project source (incl. .git) at /seed, and propagate the manifest path
    # to the container via the image's env.
    image = image_from_manifest(manifest)
    image = image.add_local_file(
        str(manifest.source_path), _CONTAINER_MANIFEST_PATH, copy=True
    )
    image = image.env({"AUTORESEARCH_MANIFEST": _CONTAINER_MANIFEST_PATH})
    image = image.add_local_dir(str(project_source_dir(manifest)), "/seed")
else:
    # In-container re-import: the image already exists, we just need a
    # placeholder for the @app.function decorator binding. Modal won't
    # rebuild from this; it uses the actual built image.
    image = modal.Image.debian_slim()

REPOS_VOLUME_NAME = "autoresearch-repos"
CACHE_VOLUME_NAME = "autoresearch-cache"

app = modal.App(f"autoresearch-{manifest.name}", image=image)
repos_vol = modal.Volume.from_name(REPOS_VOLUME_NAME, create_if_missing=True)
cache_vol = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

# Hoist out of the function so the closure stays minimal.
PROJECT_NAME = manifest.name
CLI_COMMAND = list(manifest.entrypoint.command)


@app.function(
    gpu=manifest.entrypoint.default_gpu,
    timeout=60 * 60 * 6,
    volumes={"/repos": repos_vol, "/cache": cache_vol},
)
def run_phase(spec: dict) -> dict:
    # ---- 0. Resolve scratch dirs and capture image_hash ----
    run_id = spec["run_id"]
    commit_sha = spec["commit_sha"]
    log_dir = f"/cache/logs/{PROJECT_NAME}/{run_id}"
    ckpt_dir = f"/cache/checkpoints/{PROJECT_NAME}/{run_id}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True
    )
    image_hash = hashlib.sha256(freeze.encode()).hexdigest()[:12]

    # ---- 1. Bare clone bootstrap / fetch ----
    bare_dir = Path(f"/repos/{PROJECT_NAME}/git")
    bare_dir.parent.mkdir(parents=True, exist_ok=True)
    if not (bare_dir / "HEAD").exists():
        print(f"[run_phase] seeding bare clone -> {bare_dir}", flush=True)
        _git("clone", "--bare", "/seed", str(bare_dir))
    else:
        print(f"[run_phase] fetching new commits into {bare_dir}", flush=True)
        _git(
            "--git-dir", str(bare_dir),
            "fetch", "/seed", "+refs/heads/*:refs/heads/*",
        )
    repos_vol.commit()

    # ---- 2. Worktree for this commit ----
    wt = Path(f"/work/wt-{run_id}")
    if wt.exists():
        shutil.rmtree(wt)
    _git("--git-dir", str(bare_dir), "worktree", "prune")
    _git(
        "--git-dir", str(bare_dir),
        "worktree", "add", "--detach", str(wt), commit_sha,
    )
    # Read HEAD from the worktree itself (not the bare clone — its HEAD is
    # the default branch tip, not what we just checked out).
    actual_sha = _git_capture("-C", str(wt), "rev-parse", "HEAD").strip()
    print(f"[run_phase] worktree at {wt} HEAD={actual_sha}", flush=True)

    # ---- 2b. Inject the autoresearch CLI from the live tree ----
    # The CLI ("hydra/autoresearch.py" for hydra) is infrastructure, not
    # experiment code: it must work regardless of which historical commit
    # we just checked out. Overlay it from /seed onto the worktree after
    # the commit-pinned checkout.
    overlay_root = Path("/seed")
    # The CLI's module path is derived from the manifest's entrypoint
    # command, e.g. ["python", "-m", "hydra.autoresearch"] -> hydra/autoresearch.py
    module_dotted = None
    for i, tok in enumerate(CLI_COMMAND):
        if tok == "-m" and i + 1 < len(CLI_COMMAND):
            module_dotted = CLI_COMMAND[i + 1]
            break
    if module_dotted:
        rel = Path(*module_dotted.split(".")).with_suffix(".py")
        src = overlay_root / rel
        dst = wt / rel
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"[run_phase] overlay {rel} from /seed", flush=True)
        else:
            print(
                f"[run_phase] WARN: expected CLI at {src} not found in /seed",
                flush=True,
            )

    # ---- 3. Write spec JSON, invoke the project's CLI ----
    spec_full = {
        **spec,
        "checkpoint_dir": ckpt_dir,
        "log_dir": log_dir,
    }
    spec_path = wt / ".autoresearch_spec.json"
    result_path = wt / ".autoresearch_result.json"
    spec_path.write_text(json.dumps(spec_full))

    env = {
        **os.environ,
        "PYTHONPATH": f"{wt}:" + os.environ.get("PYTHONPATH", ""),
        "HF_HOME": "/cache/hf",
        "TRANSFORMERS_CACHE": "/cache/hf",
        "BEIR_DATA_DIR": "/cache/beir",
    }
    cmd = list(CLI_COMMAND) + [
        "run",
        "--spec", str(spec_path),
        "--output", str(result_path),
    ]
    print(f"[run_phase] $ {shlex.join(cmd)}  (cwd={wt})", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(wt), env=env)
    wall_s = round(time.time() - t0, 2)
    print(f"[run_phase] CLI exit={proc.returncode} wall_s={wall_s}", flush=True)

    # ---- 4. Parse result ----
    if not result_path.exists():
        result = {
            "run_id": run_id,
            "status": "failed",
            "metrics": {},
            "cost": {"gpu_seconds": wall_s},
            "notes": f"CLI exited {proc.returncode} without writing a result file",
        }
    else:
        result = json.loads(result_path.read_text())
        result.setdefault("cost", {}).setdefault("gpu_seconds", wall_s)

    # ---- 5. Annotate with substrate provenance ----
    result["image_hash"] = image_hash
    result["actual_commit_sha"] = actual_sha

    # ---- 5b. Record in ledger ----
    try:
        from .ledger import Ledger

        ledger_path = f"/cache/ledger/{PROJECT_NAME}/ledger.db"
        ledger = Ledger(ledger_path)

        # Insert if not already present (idempotent for retries).
        existing = ledger.get_run(run_id)
        if not existing:
            ledger.insert_run(
                run_id=run_id,
                project=PROJECT_NAME,
                commit_sha=actual_sha,
                phase=spec.get("phase", ""),
                config_overrides=spec.get("config_overrides"),
                parent_run_id=spec.get("parent_run_id", ""),
                track=spec.get("track", "architecture"),
                hypothesis=spec.get("hypothesis", ""),
                image_hash=image_hash,
            )

        # Read primary metric key from manifest (if available in-container).
        primary_key = ""
        try:
            primary_key = manifest.metrics.primary
        except Exception:
            pass

        ledger.complete_run(run_id, result, primary_metric_key=primary_key)
        ledger.close()
        print(f"[run_phase] recorded in ledger at {ledger_path}", flush=True)
    except Exception as e:
        # Ledger failures must not break the run itself.
        print(f"[run_phase] WARN: ledger write failed: {e}", flush=True)

    # ---- 6. Cleanup worktree ----
    try:
        _git(
            "--git-dir", str(bare_dir),
            "worktree", "remove", "--force", str(wt),
        )
    except subprocess.CalledProcessError as e:
        print(f"[run_phase] worktree cleanup warning: {e}", flush=True)

    cache_vol.commit()
    return result


# ---------------------------------------------------------------------------
# Helpers (run inside the Modal container)
# ---------------------------------------------------------------------------

def _git(*args: str) -> None:
    subprocess.run(["git", *args], check=True)


def _git_capture(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True)
