# ML Autoresearch on Modal — Design

Build an autonomous ML research loop on top of Modal, targeted initially at
hydra (hypernet-conditioned retrieval). An orchestrator agent (pi) proposes
experiments, runs them on Modal, records results in a structured ledger, and
iterates — capable of running unattended overnight.

## Status

- ✅ **Substrate validated.** `smoke_baseline.py` reproduces the `0597b29`
  baseline row (scifact NDCG@10 = 0.6450816, matches TSV `0.6451`) on a
  Modal T4. Image built reproducibly from hydra's `uv.lock`. Resolved
  dep set is content-hashed and persisted to `image_freezes/`.
- ⏭ Everything below.

## Guiding principles

- **Replay before autonomy.** Every new layer (training, ledger, launcher)
  earns trust by re-running existing TSV rows and matching their numbers.
  We don't give pi any tool we haven't validated against ground truth.
- **Quick mode lies.** Hydra's TSV shows quick→extensive collapse is the
  norm, not the exception. The promotion gate is the most important guardrail.
- **Lockfile is truth.** Every dep change goes through `uv.lock` → image
  rebuild → new `image_hash`. No ad-hoc pip installs in functions.
- **Autonomous-first, gated.** Build the loop assuming overnight operation;
  human-in-the-loop is just the gate policy set to "ask before each action."
- **One stack to start.** HF Trainer + hydra training entrypoint. No
  framework-picking by the agent until much later.

## Project interface contract

The autoresearch system is **project-agnostic**. Hydra is the first concrete
project that plugs into it, but nothing in the launcher, ledger, or agent
should know about hypernets, BEIR, or NDCG. A project becomes
"autoresearchable" by satisfying a small contract.

### Core idea

Each project ships an **autoresearch manifest** plus a **CLI entrypoint**.
The manifest declares what the project is, what phases of validation it
supports, what metrics it produces, and how to build its environment.
The CLI is what the launcher actually invokes inside Modal.

### Manifest (`autoresearch.toml` at the project root)

```toml
[project]
name = "hydra"
description = "Hypernet-conditioned retrieval"
repo_url = "git@github.com:.../hydra.git"   # for the worktree clone

[environment]
# How to build the Modal image. Lockfile-based is the default.
python = "3.10"
lockfile = "uv.lock"
pyproject = "pyproject.toml"
apt_packages = ["git", "build-essential"]

[entrypoint]
# The CLI the launcher invokes inside the container. Receives a JSON
# spec on stdin (or via --spec-json), writes a JSON result to a path
# given by --output. See "CLI contract" below.
command = ["python", "-m", "hydra.autoresearch"]
default_gpu = "A10G"

# Phases of validation. Each phase is an independent way to evaluate an
# experiment, with its own cost profile and trust level. The agent
# composes these via the promotion gate.
[[phases]]
name = "quick"
description = "Cheap signal: 5 datasets, few epochs, small eval set"
default_gpu = "T4"
typical_runtime_min = 10
trust = "low"        # results from this phase do not count as evidence alone

[[phases]]
name = "extensive"
description = "Full validation: all datasets, full epochs"
default_gpu = "A10G"
typical_runtime_min = 120
trust = "high"
gates_from = ["quick"]   # may only run if a passing `quick` parent exists

[[phases]]
name = "colbert"
description = "ColBERT-style late-interaction variant"
default_gpu = "A10G"
typical_runtime_min = 90
trust = "high"
gates_from = ["quick"]

# The metric schema. Determines ledger columns and what the agent can
# query / optimize for.
[metrics]
primary = "avg_ndcg10"          # used by stop conditions and ranking
higher_is_better = true
columns = [
    { name = "scifact_ndcg10",  type = "float" },
    { name = "fiqa_ndcg10",     type = "float" },
    { name = "avg_ndcg10",      type = "float" },
    { name = "task_vs_generic", type = "float" },
]

# A reference run the launcher can re-execute to detect substrate drift.
[baseline]
commit_sha = "0597b29"
phase = "quick"
config_overrides = {}
expected = { avg_ndcg10 = 0.5069, scifact_ndcg10 = 0.6451, tolerance = 0.002 }
```

Nothing in this file is hydra-specific in *structure*. Another project
might declare phases `[smoke, full]` with metrics `[loss, perplexity]`.
The launcher reads the manifest and treats phases / metrics as opaque
strings with declared semantics.

### CLI contract

The launcher invokes the entrypoint inside the Modal container with a
single JSON spec and expects a single JSON result file:

```bash
$PROJECT_CMD --spec /tmp/spec.json --output /tmp/result.json
```

**Spec (input):**
```json
{
  "run_id": "01HXYZ...",
  "phase": "quick",
  "config_overrides": {"lr": 1e-4, "epochs": 5},
  "commit_sha": "7268a7c",
  "checkpoint_dir": "/cache/checkpoints/01HXYZ...",
  "log_dir": "/cache/logs/01HXYZ..."
}
```

**Result (output):**
```json
{
  "run_id": "01HXYZ...",
  "status": "ok",
  "metrics": {
    "scifact_ndcg10": 0.6480,
    "fiqa_ndcg10": 0.3740,
    "avg_ndcg10": 0.5110,
    "task_vs_generic": 0.0026
  },
  "artifacts": {"checkpoint": "/cache/checkpoints/01HXYZ.../best.pt"},
  "cost": {"gpu_seconds": 612, "peak_vram_gb": 8.1},
  "notes": "training converged at epoch 12"
}
```

The project is also expected to **stream intermediate metrics** to a
known path (e.g. `$log_dir/progress.jsonl`) so the launcher's auto-kill
rule (M4 rule 3) can detect collapse mid-run without needing project
internals. One JSON object per line:
```json
{"step": 500, "wall_s": 240, "metrics": {"val_avg_ndcg10": 0.48}}
```

Status values the launcher understands: `ok`, `failed`, `oom`,
`collapsed`, `killed`.

### What this buys us

- **The launcher / ledger / agent never import project code.** They shell
  out via the CLI contract. A new project = new manifest + new CLI
  implementation, no changes to autoresearch internals.
- **Phases generalize cleanly.** Hydra's `quick → extensive → colbert`
  is one instantiation. A vision project might do `tiny → mid → full`.
  An RL project might do `unit → smoke → benchmark`. The promotion gate
  rule is "phase B can only run if a passing parent exists in phases
  listed in `gates_from`," which works for any of these.
- **Replay tests are uniform.** "Re-run this commit at this phase, assert
  metrics within tolerance" works for every project.
- **The agent's tools become project-agnostic.** It calls
  `launch(phase=..., overrides=...)`, not `train_hypernet(...)`.

### Open questions on the contract

- **Config schema discoverability.** How does the agent know what
  `config_overrides` are valid for a given project/phase? Options: (a)
  free-form dict, agent learns from failures; (b) JSON schema in the
  manifest; (c) a `describe` subcommand on the CLI that returns the
  schema. **Lean (c)** — keeps the manifest small and lets the project
  generate the schema from its config dataclasses.
- **How much does the manifest leak into the agent's context?** The
  agent should see phase descriptions, metric names, and gate rules,
  but probably not raw image config. Filter at the launcher.
- **Multi-project sessions.** Out of scope for now — assume one project
  per autoresearch session.

## Milestones

### M1 — Project interface + hydra as first implementer

Two coupled deliverables:

**(a) The project-agnostic side (lives in `modal-out/`):**
- A loader for `autoresearch.toml`.
- An `image_from_manifest(manifest)` function that builds the Modal
  image per the `[environment]` block (using the same lockfile pattern
  the smoke test validated).
- A `run_phase(manifest, spec) -> result` Modal Function: generic. It
  checks out `commit_sha` into a per-run `git worktree` on a Volume,
  invokes the project CLI per the contract above, parses the result
  JSON, captures the freeze hash, and returns.

**(b) Hydra as the first project (lives in `hydra/`):**
- Write `hydra/autoresearch.toml` with phases `quick`, `extensive`,
  `colbert` and the metric schema matching `results.tsv`.
- Write `python -m hydra.autoresearch` — a thin CLI that maps a spec
  JSON onto the existing `train_hypernet` + eval code, and emits the
  result JSON. This is also where the canonical phase configs live
  (epoch counts, dataset lists, eval depth) — the manifest just names
  them, the CLI implements them.
- Stream `progress.jsonl` from inside the training loop.

**Worktree mechanics:**
- A long-lived `modal.Volume` holds a single bare clone of the project
  repo (`/repos/<project>.git`).
- Each `run_phase` call does `git worktree add /work/<run_id> <commit_sha>`,
  runs the CLI from there, then `git worktree remove`.
- This gives commit-level isolation across concurrent runs without
  re-cloning, and the bare clone is the only shared mutable state.

**Replay test (definition of done):**
Run `7268a7c` at phase `quick` → assert `avg_ndcg10 ≈ 0.5110 ± 0.001`.
Run `7268a7c` at phase `extensive` → assert it collapses
(`avg_ndcg10 ≈ 0.3430`). Both replays passing proves: (i) the worktree
mechanic works, (ii) the CLI contract round-trips metrics correctly,
(iii) hydra's training reproduces *including its failure modes* on Modal.

### M2 — Ledger

SQLite on a Modal Volume. Single source of truth for everything pi has tried.

**Schema:**
```sql
CREATE TABLE runs (
    run_id          TEXT PRIMARY KEY,    -- ulid or uuid7
    project         TEXT NOT NULL,       -- from manifest [project].name
    parent_run_id   TEXT,                -- search-tree edge
    commit_sha      TEXT NOT NULL,
    image_hash      TEXT NOT NULL,
    track           TEXT NOT NULL,       -- baseline|architecture|...
    phase           TEXT NOT NULL,       -- project-defined phase name
    config_json     TEXT NOT NULL,       -- overrides applied
    -- metrics: stored as JSON; columns are project-defined per the manifest.
    -- A view per project can flatten these into typed columns for tsv_export.
    metrics_json    TEXT,
    primary_metric  REAL,                -- denormalized for fast ranking
    -- agent annotations
    hypothesis      TEXT,                -- what we expected
    observation     TEXT,                -- what happened
    status          TEXT,                -- keep|discard|review|running|failed
    -- cost & timing
    cost_gpu_min    REAL,
    started_at      INTEGER,
    finished_at     INTEGER,
    -- raw
    log_path        TEXT                 -- volume path to full logs
);

CREATE TABLE lessons (
    lesson_id   TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    evidence    TEXT,                   -- run_ids that support it
    created_at  INTEGER,
    superseded_by TEXT
);
```

**Operations:**
- `insert_run(spec) -> run_id` (status=running)
- `update_run(run_id, **fields)`
- `query(filter) -> list[Row]` (used by agent for "what's my best so far")
- `set_status(run_id, status, note)`
- `tsv_export() -> str` mirroring `results.tsv` columns for human diffing
- `add_lesson(text, evidence)` / `query_lessons()`

**Replay test:**
Bulk-import `hydra/results.tsv` into the ledger, then `tsv_export()` and
diff against the original. Must round-trip cleanly.

### M3 — Launcher abstraction

A normal Python library (no agent involvement yet) that wraps M1 + M2
into the surface pi will eventually call.

```python
class Launcher:
    def launch(self, spec: ExperimentSpec) -> str: ...    # returns run_id
    def poll(self, run_id: str) -> RunState: ...
    def cancel(self, run_id: str) -> None: ...
    def query(self, **filters) -> list[Run]: ...
    def set_status(self, run_id, status, note) -> None: ...
    def tail_logs(self, run_id, n=100) -> str: ...
    def add_lesson(self, text, evidence) -> None: ...
```

`launch` is **fire-and-forget**: it inserts a `running` row, calls
`run_training.spawn(...)`, and returns immediately. A small reaper
(Modal cron or local process) drains completed `FunctionCall`s and writes
results back. This is what enables parallel sweeps and async agent loops.

**Spec:**
```python
@dataclass
class ExperimentSpec:
    commit_sha: str
    mode: Literal["quick", "extensive", "colbert"]
    config_overrides: dict
    parent_run_id: str | None
    hypothesis: str
    track: str = "architecture"
```

**Validated by:** writing a small driver that launches 3 quick runs in
parallel and confirms the ledger reflects all three transitioning
running → done with correct metrics.

### M4 — Promotion gate

The single most important guardrail. Enforced **inside the launcher**,
not trusted to the agent. All rules read project-specific values
(phase names, primary metric, baseline) from the manifest — no
hydra-specific logic.

**Rules (v1):**
1. **Phase gating from manifest**: a launch in phase B requires a
   `parent_run_id` whose `phase ∈ B.gates_from`, with
   `status ∈ {keep, review}` and primary metric ≥ baseline − epsilon.
   For hydra this enforces "no extensive without a passing quick";
   for any other project it enforces whatever `gates_from` they declared.
2. **Per-session budget caps**: `max_extensive_runs`, `max_gpu_min_total`,
   `max_dollars_total`. Launcher tracks against running totals and refuses
   launches that would exceed.
3. **Auto-kill on collapse**: training jobs report intermediate
   `val_avg_ndcg10` every N steps; if it drops >X% below baseline at >Y%
   of training, the job self-terminates. (Encoded in `run_training`, not
   the launcher, since it needs the live process.)
4. **Mandatory baseline rerun**: every K experiments (or once per session),
   rerun the frozen baseline and refuse to continue if it drifts beyond
   noise. Detects infra/dep drift mid-session.

Rules 1 and 2 are hard rejects with a clear error message the agent can read.
Rule 3 is a kill signal. Rule 4 is a stop-the-world.

**Validated by:** unit tests on the launcher with fake function calls
verifying each rule rejects what it should and admits what it should.

### M5 — Wire pi

Once M1–M4 work, the agent layer is small. Pi gets a tool surface that
maps directly onto `Launcher` methods:

- `propose_experiment(...)` — agent's job, not a tool
- `launch(commit_sha, mode, overrides, hypothesis, parent_run_id)`
- `poll(run_id)` / `tail_logs(run_id)`
- `query_ledger(filters)` — for "what are my top-5 by avg_ndcg10 in mode=quick"
- `set_status(run_id, keep|discard|review, note)`
- `record_lesson(text, evidence_run_ids)`
- `cancel(run_id)`

**The agent loop:**
```
while session_budget_remaining and not stop_condition:
    context = query_ledger(recent=20) + query_lessons()
    hypothesis = propose(context)              # ← interrupt point (HITL gate)
    spec = design(hypothesis)                  # ← interrupt point
    run_id = launch(spec)                      # async
    while not done(run_id):
        do_other_reasoning_or_wait()
    obs = analyze(run_id)
    update_run(run_id, observation=obs, status=...)
    if pattern_detected: record_lesson(...)
```

**Gate policy** is the only thing that distinguishes interactive from
overnight: in HITL mode, interrupt points pause for approval; in overnight
mode, they auto-approve unless a guardrail trips. Same loop.

**Stop conditions for overnight:**
- Budget exhausted
- No `extensive`-mode improvement over best in last K experiments
- Baseline rerun (rule 4) failed
- N consecutive launches rejected by promotion gate (agent stuck)

**Open question:** where does pi run? Locally on the laptop is fine for
HITL. For overnight, either (a) leave the laptop awake, (b) run pi itself
inside a long-lived `modal.Sandbox` or scheduled function. (b) is the
real answer for unattended runs but adds a hosting layer — defer until
M1–M4 are solid.

## Things explicitly out of scope (for now)

- Architecture/recipe search from scratch
- Multiple training stacks (HF Trainer only)
- Distributed/multi-GPU training
- A web UI (the TSV export + Modal dashboard are enough)
- Dataset curation / synthesis
- The agent editing its own training code (config-only first; code-edit
  privileges come after promotion-gate-style guardrails for code changes
  are designed)

## Known fragilities to address eventually

- **BEIR zips fetched from `public.ukp.informatik.tu-darmstadt.de`** — single
  external SPOF. Mirror to a Modal Volume before relying on overnight runs.
- **No per-run worktree isolation yet** — concurrent training runs against
  the shared hydra clone will collide. Resolve in M1 (`git worktree add`).
- **HF model downloads** — same SPOF risk as BEIR. Cached on Volume after
  first run, but a cold cache + HF outage = stuck.
- **Modal image rebuilds on dep changes** — fine, but `image_hash` must be
  recorded for every run so we can diff failures across substrate versions.

## Sequencing

M1 → M2 → M3 → M4 → M5, but M2 (ledger) can start in parallel with M1
since its only dependency is the schema, which is already drafted above.
Don't start M5 until M1–M4 each have their replay/validation tests
passing — the whole point of the replay-first discipline is that the
agent inherits trusted infrastructure, not infrastructure it has to
debug alongside the research problem.
