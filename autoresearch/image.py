"""Build a Modal image from a project's autoresearch manifest.

Mirrors the smoke_baseline.py pattern that we already validated:
  1. apt install + uv
  2. copy the project's pyproject.toml + lockfile
  3. uv pip install --system from the frozen lock
  4. (the project source itself is mounted per-call by run_phase, not here)
"""

from __future__ import annotations

from pathlib import Path

import modal

from .manifest import Manifest


def image_from_manifest(manifest: Manifest) -> modal.Image:
    env = manifest.environment
    project_root = manifest.project_root
    pyproject = project_root / env.pyproject
    lockfile = project_root / env.lockfile

    if not pyproject.exists():
        raise FileNotFoundError(
            f"manifest references {env.pyproject!r} but {pyproject} not found"
        )
    if not lockfile.exists():
        raise FileNotFoundError(
            f"manifest references {env.lockfile!r} but {lockfile} not found"
        )

    img = modal.Image.debian_slim(python_version=env.python)
    if env.apt_packages:
        img = img.apt_install(*env.apt_packages)
    img = (
        img.pip_install("uv")
        .add_local_file(str(pyproject), "/work/pyproject.toml", copy=True)
        .add_local_file(str(lockfile), "/work/uv.lock", copy=True)
        .run_commands(
            # uv sync needs the package itself to exist; create a stub then
            # export the locked deps and install them into the system Python
            # so we don't have to fight Modal's interpreter / PATH.
            f"mkdir -p /work/{manifest.name} && touch /work/{manifest.name}/__init__.py",
            "cd /work && uv export --frozen --no-hashes --no-emit-project -o /tmp/reqs.txt",
            "uv pip install --system -r /tmp/reqs.txt",
        )
    )
    return img


def project_source_dir(manifest: Manifest) -> Path:
    """The directory whose contents (including .git) get mounted per-call as
    the live project source. For local projects this is just the repo root."""
    return manifest.project_root
