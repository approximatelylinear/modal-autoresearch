"""Loader for autoresearch.toml project manifests.

Validates the contract described in DESIGN.md and exposes typed accessors.
The launcher / image builder / run_phase function all consume Manifest
objects rather than raw dicts so a malformed manifest fails loudly at load
time rather than mid-experiment.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from pathlib import Path
from typing import Any


@dataclass
class Phase:
    name: str
    description: str = ""
    default_gpu: str = "T4"
    typical_runtime_min: int = 30
    trust: str = "low"            # "low" | "high"
    gates_from: list[str] = field(default_factory=list)


@dataclass
class MetricColumn:
    name: str
    type: str = "float"
    description: str = ""


@dataclass
class Metrics:
    primary: str
    higher_is_better: bool
    columns: list[MetricColumn]


@dataclass
class Baseline:
    commit_sha: str
    phase: str
    config_overrides: dict[str, Any]
    expected: dict[str, Any]      # includes a `tolerance` key


@dataclass
class Environment:
    python: str
    lockfile: str
    pyproject: str
    apt_packages: list[str]


@dataclass
class Entrypoint:
    command: list[str]
    default_gpu: str


@dataclass
class Manifest:
    name: str
    description: str
    repo_url: str                  # local path or git URL
    environment: Environment
    entrypoint: Entrypoint
    phases: dict[str, Phase]
    metrics: Metrics
    baseline: Baseline | None
    source_path: Path              # path to the autoresearch.toml file

    @property
    def project_root(self) -> Path:
        """Local filesystem root of the project repo, derived from repo_url
        if it's a local path, else from the manifest's location."""
        p = Path(self.repo_url)
        if p.exists():
            return p.resolve()
        return self.source_path.parent.resolve()

    def phase(self, name: str) -> Phase:
        if name not in self.phases:
            raise KeyError(
                f"phase {name!r} not in manifest (have: {sorted(self.phases)})"
            )
        return self.phases[name]


def load_manifest(path: str | Path) -> Manifest:
    path = Path(path)
    raw = tomllib.loads(path.read_text())

    proj = _require(raw, "project", path)
    env = _require(raw, "environment", path)
    ep = _require(raw, "entrypoint", path)
    metrics_raw = _require(raw, "metrics", path)
    phases_raw = raw.get("phases", [])
    if not phases_raw:
        raise ValueError(f"{path}: at least one [[phases]] entry required")

    phases = {
        ph["name"]: Phase(
            name=ph["name"],
            description=ph.get("description", ""),
            default_gpu=ph.get("default_gpu", "T4"),
            typical_runtime_min=ph.get("typical_runtime_min", 30),
            trust=ph.get("trust", "low"),
            gates_from=list(ph.get("gates_from", [])),
        )
        for ph in phases_raw
    }

    # Validate gates_from references
    for ph in phases.values():
        for g in ph.gates_from:
            if g not in phases:
                raise ValueError(
                    f"{path}: phase {ph.name!r} gates_from unknown phase {g!r}"
                )

    metrics = Metrics(
        primary=_require(metrics_raw, "primary", path),
        higher_is_better=metrics_raw.get("higher_is_better", True),
        columns=[
            MetricColumn(
                name=c["name"],
                type=c.get("type", "float"),
                description=c.get("description", ""),
            )
            for c in metrics_raw.get("columns", [])
        ],
    )
    if metrics.primary not in {c.name for c in metrics.columns}:
        raise ValueError(
            f"{path}: metrics.primary {metrics.primary!r} not in columns"
        )

    baseline_raw = raw.get("baseline")
    baseline = None
    if baseline_raw:
        baseline = Baseline(
            commit_sha=_require(baseline_raw, "commit_sha", path),
            phase=_require(baseline_raw, "phase", path),
            config_overrides=baseline_raw.get("config_overrides", {}),
            expected=baseline_raw.get("expected", {}),
        )
        if baseline.phase not in phases:
            raise ValueError(
                f"{path}: baseline.phase {baseline.phase!r} not declared in [[phases]]"
            )

    return Manifest(
        name=_require(proj, "name", path),
        description=proj.get("description", ""),
        repo_url=_require(proj, "repo_url", path),
        environment=Environment(
            python=env.get("python", "3.10"),
            lockfile=env.get("lockfile", "uv.lock"),
            pyproject=env.get("pyproject", "pyproject.toml"),
            apt_packages=list(env.get("apt_packages", [])),
        ),
        entrypoint=Entrypoint(
            command=list(_require(ep, "command", path)),
            default_gpu=ep.get("default_gpu", "T4"),
        ),
        phases=phases,
        metrics=metrics,
        baseline=baseline,
        source_path=path.resolve(),
    )


def _require(d: dict, key: str, path: Path) -> Any:
    if key not in d:
        raise ValueError(f"{path}: missing required key {key!r}")
    return d[key]


if __name__ == "__main__":
    # Smoke test: `python -m autoresearch.manifest path/to/autoresearch.toml`
    m = load_manifest(sys.argv[1])
    print(f"Loaded manifest: {m.name}")
    print(f"  project_root: {m.project_root}")
    print(f"  phases: {list(m.phases)}")
    print(f"  primary metric: {m.metrics.primary}")
    if m.baseline:
        print(f"  baseline: {m.baseline.commit_sha} @ {m.baseline.phase}")
