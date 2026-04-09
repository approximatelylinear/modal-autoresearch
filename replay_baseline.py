"""M1 definition-of-done: replay a project's declared baseline run on Modal
and compare against the manifest's expected metrics.

Usage:
    uv run modal run replay_baseline.py
"""

from __future__ import annotations

import os
from pathlib import Path

MANIFEST_PATH = (Path(__file__).parent.parent / "hydra" / "autoresearch.toml").resolve()
os.environ["AUTORESEARCH_MANIFEST"] = str(MANIFEST_PATH)

# Module-scope app + run_phase are constructed at import time using the
# manifest path above.
from autoresearch.run_phase import app, manifest, run_phase  # noqa: E402


@app.local_entrypoint()
def main():
    if not manifest.baseline:
        raise SystemExit("manifest has no [baseline] section")

    spec = {
        "run_id": "replay-baseline-0001",
        "phase": manifest.baseline.phase,
        "commit_sha": manifest.baseline.commit_sha,
        "config_overrides": manifest.baseline.config_overrides,
    }
    print(f"=== Replaying {manifest.name} baseline ===")
    print(f"  commit: {manifest.baseline.commit_sha}")
    print(f"  phase:  {manifest.baseline.phase}")
    print(f"  expected: {manifest.baseline.expected}")
    print()

    result = run_phase.remote(spec)

    print("\n=== result ===")
    print(f"  status:      {result.get('status')}")
    print(f"  image_hash:  {result.get('image_hash')}")
    print(f"  actual_sha:  {result.get('actual_commit_sha')}")
    print(f"  cost:        {result.get('cost')}")
    if result.get("notes"):
        print(f"  notes:       {result.get('notes')}")

    metrics = result.get("metrics", {})
    print("\n  metrics:")
    for k, v in metrics.items():
        if k == "per_dataset":
            continue
        print(f"    {k}: {v}")

    expected = manifest.baseline.expected
    tol = float(expected.get("tolerance", 0.005))
    print(f"\n  comparison (tolerance ±{tol}):")
    ok = True
    for key, exp in expected.items():
        if key == "tolerance":
            continue
        got = metrics.get(key)
        if got is None:
            print(f"    {key}: MISSING (expected {exp})")
            ok = False
            continue
        delta = got - exp
        mark = "OK" if abs(delta) <= tol else "FAIL"
        print(f"    {key}: got={got:.4f} expected={exp:.4f} delta={delta:+.4f}  [{mark}]")
        if abs(delta) > tol:
            ok = False

    print()
    if ok:
        print("REPLAY PASSED")
    else:
        print("REPLAY FAILED")
        raise SystemExit(1)
