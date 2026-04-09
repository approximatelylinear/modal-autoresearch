# Architecture

Project-agnostic ML autoresearch on Modal. An orchestrator agent proposes
experiments, runs them on Modal GPUs, records results in a structured
ledger, and iterates — capable of running unattended overnight.

## System diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Agent (pi) — M5                       │
│  propose → launch → poll → analyze → set_status → loop  │
└────────────────────────┬────────────────────────────────┘
                         │ calls
┌────────────────────────▼────────────────────────────────┐
│               Launcher (local Python) — M3               │
│  launch() ──► gate.check() ──► run_phase.spawn()         │
│  poll() / poll_all() ──► drain completed ──► ledger      │
│  query() / best_runs() / set_status() / add_lesson()     │
├──────────────┬──────────────────────┬───────────────────┤
│  Gate — M4   │    Ledger — M2       │   Modal — M1      │
│  phase gates │    SQLite (local)    │   run_phase()     │
│  budget caps │    runs + lessons    │   on GPU          │
│  baseline    │    import/export TSV │                    │
└──────────────┴──────────────────────┴───────────────────┘
                                            │
                              ┌──────────────▼──────────────┐
                              │   Modal Container (GPU)     │
                              │                             │
                              │  1. git worktree add <sha>  │
                              │  2. overlay CLI from /seed  │
                              │  3. run project CLI:        │
                              │     spec.json → result.json │
                              │  4. record in Volume ledger │
                              │  5. cleanup worktree        │
                              ├─────────────────────────────┤
                              │  Volumes:                   │
                              │   /repos — bare git clones  │
                              │   /cache — BEIR, HF, logs,  │
                              │            checkpoints,     │
                              │            ledger backup     │
                              └──────────────────────────────┘
```

## Layers

### M1 — Run Phase

`autoresearch/run_phase.py` + `autoresearch/image.py`

The execution substrate. A single generic Modal Function that runs any
project's experiment. Given a spec `{run_id, phase, commit_sha,
config_overrides}`, it:

- Builds and caches the Modal image from the project's `uv.lock`.
- Maintains a bare git clone on a Modal Volume, fetches new commits
  each call.
- Creates an isolated `git worktree` for the target commit so
  concurrent runs at different commits are safe.
- Overlays the autoresearch CLI from `/seed` (live working tree) so
  the CLI works against any historical commit.
- Shells out to the project's CLI per the contract (`spec.json` in,
  `result.json` out).
- Captures `image_hash` (content-addressed dep fingerprint from
  `pip freeze | sha256`) and `actual_commit_sha` for provenance.

### M2 — Ledger

`autoresearch/ledger.py`

SQLite database — the agent's memory of the search. Two tables:

**`runs`**: `run_id`, `project`, `commit_sha`, `phase`, `track`,
`config_json`, `metrics_json` (project-specific, stored as JSON),
`primary_metric` (denormalized for fast ranking), `hypothesis`,
`observation`, `status` (running / ok / failed / keep / discard /
review / killed), `cost_gpu_min`, timestamps, `image_hash`.

**`lessons`**: agent meta-observations with evidence links and
supersession chains. ("FiLM with trainable A/B collapses at
extensive scale" — the kind of insight that prevents the agent from
re-deriving the same failure three nights in a row.)

Key operations: `insert_run`, `complete_run`, `query` (filtered),
`best_runs` (top-N by primary metric), `set_status`, `import_tsv` /
`tsv_export`.

### M3 — Launcher

`autoresearch/launcher.py`

Local orchestration layer that wraps M1 + M2. Runs on the user's
machine or inside an agent process.

| Method | Behavior |
|--------|----------|
| `launch(spec)` | Gate check, ledger insert, `run_phase.spawn()`, return `run_id` |
| `poll(run_id)` | Non-blocking: drain if done, return ledger row |
| `poll_all()` | Drain all completed in-flight calls |
| `wait(run_id)` | Blocking until completion |
| `cancel(run_id)` | Kill in-flight, mark `killed` |
| `query(**filters)` | Filter by phase / status / track |
| `best_runs(phase, n)` | Top-N by primary metric |
| `set_status(run_id, s, note)` | Agent's keep / discard / review decision |
| `add_lesson(text, evidence)` | Record a meta-observation |
| `stats()` | Totals: runs, kept, discarded, GPU-min, best metric |

`launch()` is fire-and-forget: it returns immediately after spawning.
Multiple runs can be in-flight for parallel sweeps. `poll_all()` drains
completed calls in a batch.

### M4 — Promotion Gate

`autoresearch/gate.py`

Enforced inside the launcher, not trusted to the agent.

| Rule | What it checks | Effect |
|------|---------------|--------|
| Phase gating | `gates_from` in manifest: parent must exist, be from the right phase, have `keep` or `review` status, primary metric above baseline - epsilon | Hard reject |
| Budget caps | `max_runs`, `max_gpu_min`, `max_high_trust_runs` per session | Hard reject |
| Baseline freshness | Every N experiments, require a baseline rerun | Reject with `action="baseline_rerun_required"` |
| Auto-kill on collapse | Mid-run metric drops below threshold | Designed, deferred (needs `progress.jsonl` monitoring) |

The gate returns a `GateResult` with `allowed`, `reason` (human-readable
for the agent), and `action` (machine-readable for the caller).
`skip_gate=True` on `launch()` bypasses for baseline reruns.

### M5 — Agent loop (not yet built)

Pi gets the launcher's tool surface directly. The loop:

```
while budget_remaining and not stop_condition:
    context = query(recent=20) + query_lessons()
    hypothesis = propose(context)        # ← HITL interrupt point
    spec = design(hypothesis)            # ← HITL interrupt point
    run_id = launch(spec)                # async
    ...poll / wait...
    observe + set_status + maybe record_lesson
```

Human-in-the-loop vs overnight: same loop, different gate policy.
HITL pauses at interrupt points for approval. Overnight auto-approves
unless a guardrail trips.

Stop conditions for overnight:
- Budget exhausted
- No improvement in last K experiments
- Baseline rerun failed (substrate drift)
- N consecutive gate rejections (agent stuck)

## Project interface contract

The system is project-agnostic. Any ML project becomes "autoresearchable"
by providing two artifacts:

### `autoresearch.toml` (manifest)

Declares the project's identity, environment, phases, metrics, and
baseline. The launcher reads this; the agent sees phase descriptions
and metric names.

```toml
[project]
name = "hydra"
description = "Hypernet-conditioned retrieval"

[environment]
python = "3.10"
lockfile = "uv.lock"

[entrypoint]
command = ["python", "-m", "hydra.autoresearch"]
default_gpu = "A10G"

[[phases]]
name = "quick"
trust = "low"
typical_runtime_min = 15

[[phases]]
name = "extensive"
trust = "high"
gates_from = ["quick"]
typical_runtime_min = 120

[metrics]
primary = "avg_ndcg10"
higher_is_better = true
columns = [
    { name = "scifact_ndcg10", type = "float" },
    { name = "fiqa_ndcg10",    type = "float" },
    { name = "avg_ndcg10",     type = "float" },
    { name = "task_vs_generic", type = "float" },
]

[baseline]
commit_sha = "0597b29"
phase = "quick"
```

### CLI entrypoint

The launcher shells out to this inside the Modal container. Never
imported as a library.

**`describe`** — emits the config schema (what `config_overrides` keys
are valid, their types and defaults) as JSON. The agent reads this to
understand what knobs it can turn.

**`run --spec X --output Y`** — reads a spec JSON, runs the experiment,
writes a result JSON:

```
spec.json                          result.json
{                                  {
  "run_id": "...",                   "run_id": "...",
  "phase": "quick",                  "status": "ok",
  "commit_sha": "7268a7c",           "metrics": {
  "config_overrides": {...},            "scifact_ndcg10": 0.648,
  "checkpoint_dir": "...",              "avg_ndcg10": 0.511,
  "log_dir": "..."                      ...
}                                     },
                                      "cost": {"gpu_seconds": 612}
                                   }
```

**`progress.jsonl`** — streamed to `log_dir` during training for the
auto-kill rule. One JSON object per line:
```json
{"step": 500, "wall_s": 240, "metrics": {"val_avg_ndcg10": 0.48}}
```

A new project = new manifest + new CLI. Zero changes to autoresearch
internals.

## Reproducibility model

Every run is pinned on three axes:

| Axis | How it's captured |
|------|-------------------|
| **Code** | `commit_sha` via git worktree checkout from bare clone |
| **Deps** | `image_hash` (SHA-256 of `pip freeze` output, first 12 chars) |
| **Config** | `config_json` in the ledger (the exact overrides applied) |

The autoresearch CLI is an *overlay* — it's copied from the live
working tree onto the worktree after checkout. This means experiment
code is commit-pinned but the CLI infrastructure always tracks the
latest version.

## Validation status

| Test | What it proves |
|------|---------------|
| `smoke_baseline.py` | Image builds from lockfile, eval reproduces exactly (NDCG@10 = 0.6451) |
| `replay_baseline.py` | Full stack: manifest → image → worktree → CLI → training → eval → result JSON |
| `test_ledger.py` | 99-row TSV round-trip with zero diffs |
| `test_launcher.py` | 10 cases: launch, poll, cancel, query, best_runs, set_status, stats, poll_all |
| `test_gate.py` | 21 cases: phase gating, budget caps, baseline freshness, launcher integration |

## Files

```
modal-out/
├── DESIGN.md                       Design doc (milestones, open questions)
├── ARCHITECTURE.md                 This file
├── smoke_baseline.py               Substrate validation (eval-only)
├── replay_baseline.py              M1 end-to-end validation (training)
├── test_ledger.py                  M2 validation
├── test_launcher.py                M3 validation
├── test_gate.py                    M4 validation
├── image_freezes/
│   └── 5262b5408d94.txt            Known-good resolved dep set
└── autoresearch/
    ├── __init__.py
    ├── manifest.py                 Load + validate autoresearch.toml
    ├── image.py                    Build Modal image from manifest
    ├── run_phase.py                Generic Modal Function + worktree mechanics
    ├── ledger.py                   SQLite runs + lessons
    ├── launcher.py                 Local orchestration layer
    └── gate.py                     Promotion gate

hydra/                              First project to implement the contract
├── autoresearch.toml               Manifest
└── hydra/
    └── autoresearch.py             CLI (quick phase implemented, others stubbed)
```
