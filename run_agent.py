"""LLM-driven autoresearch agent using OpenAI function calling.

The agent reads the session context, proposes experiments, calls tools,
analyzes results, and iterates. Same loop as the interactive REPL, but
the LLM makes the decisions.

Usage:
    # Create .env with OPENAI_API_KEY=sk-...
    uv run python run_agent.py

    # With options:
    uv run python run_agent.py --model gpt-4o --max-turns 20 --import-tsv ../hydra/results.tsv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Set manifest before autoresearch imports.
MANIFEST_PATH = (Path(__file__).parent.parent / "hydra" / "autoresearch.toml").resolve()
os.environ["AUTORESEARCH_MANIFEST"] = str(MANIFEST_PATH)

from openai import OpenAI

from autoresearch.gate import SessionBudget
from autoresearch.session import Session
from autoresearch.tools import TOOL_SCHEMAS, Tools


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM-driven autoresearch agent")
    p.add_argument("--manifest", default=str(MANIFEST_PATH))
    p.add_argument("--ledger", default="session_ledger.db")
    p.add_argument("--import-tsv", default="")
    p.add_argument("--max-runs", type=int, default=50)
    p.add_argument("--max-gpu-min", type=float, default=600)
    p.add_argument("--max-high-trust", type=int, default=10)
    p.add_argument("--model", default="gpt-4o",
                    help="OpenAI model (gpt-4o, gpt-4o-mini, etc.)")
    p.add_argument("--max-turns", type=int, default=100,
                    help="Safety cap on turns. The agent normally exits via "
                         "conclude(); this is a backstop if it gets stuck.")
    p.add_argument("--hitl", action="store_true",
                    help="Human-in-the-loop: pause for approval before each launch")
    p.add_argument("--local", action="store_true",
                    help="Run experiments on local GPU instead of Modal")
    p.add_argument("--dry-run", action="store_true",
                    help="Print system prompt + initial context and exit")
    return p.parse_args()


def build_openai_tools() -> list[dict]:
    """Convert our TOOL_SCHEMAS to OpenAI function-calling format."""
    return [
        {"type": "function", "function": schema}
        for schema in TOOL_SCHEMAS
    ]


def run_agent(
    client: OpenAI,
    model: str,
    session: Session,
    tools: Tools,
    *,
    max_turns: int = 100,
    hitl: bool = False,
) -> None:
    """Run the agentic loop.

    The agent owns pacing and termination: it decides when to think,
    when to act, and when to stop (via the `conclude` tool). External
    stops (budget exhausted, agent stuck) prompt the agent to conclude
    with a clean record but don't terminate mid-thought. `max_turns` is
    a backstop for runaway loops only.
    """
    system = session.system_prompt()
    openai_tools = build_openai_tools()
    messages: list[dict] = [{"role": "system", "content": system}]

    initial_context = tools.context()
    messages.append({
        "role": "user",
        "content": (
            "Session started. Here is the current state:\n\n"
            f"{initial_context}\n\n"
            "Begin by reviewing what's been tried, then call set_plan() "
            "to declare what you're investigating this session. From "
            "there, drive the loop yourself — think, act, reflect, and "
            "call conclude() when you've reached your conclusion."
        ),
    })

    external_stop_announced = False

    for turn in range(max_turns):
        print(f"\n{'='*60}")
        print(f"Turn {turn + 1}/{max_turns}  (max_turns is a safety cap; "
              f"agent exits via conclude)")
        print(f"{'='*60}")

        # Agent self-termination — primary exit path.
        if session.concluded:
            print("\n*** Agent called conclude() — session ending cleanly ***")
            break

        # External safety stop — surface to the agent once so it can
        # wrap up via conclude(). If it ignores the message and keeps
        # acting, we'll end up here next turn too.
        sc = session.check_stop()
        if sc.triggered and not external_stop_announced:
            print(f"\n*** External stop signal: {sc.reason} ***")
            messages.append({
                "role": "user",
                "content": (
                    f"External stop signal: {sc.reason}. "
                    f"Wrap up by calling conclude(summary, lessons) with a "
                    f"record of what was learned this session. Do not "
                    f"launch more experiments."
                ),
            })
            external_stop_announced = True

        # LLM call.
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump())

        # Print reasoning text alongside actions (the model may emit both).
        if msg.content:
            print(f"\nAgent: {msg.content}")

        if msg.tool_calls:
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)

                print(f"\n  Tool: {fn_name}({_compact_args(fn_args)})")

                # HITL gate: pause before launch.
                if hitl and fn_name == "launch":
                    print(f"\n  Hypothesis: {fn_args.get('hypothesis', 'n/a')}")
                    print(f"  Phase: {fn_args.get('phase', '?')}")
                    print(f"  Commit: {fn_args.get('commit_sha', '?')}")
                    if fn_args.get('config_overrides'):
                        print(f"  Overrides: {fn_args['config_overrides']}")
                    approval = input("  Approve? [y/N/stop] ").strip().lower()
                    if approval in ("stop", "quit", "q"):
                        print("  Ending session by user request.")
                        return
                    elif approval != "y":
                        result = {"error": "human_rejected", "reason": "User declined this launch"}
                        print(f"  -> Rejected by user")
                    else:
                        result = tools.dispatch(fn_name, fn_args)
                        print(f"  -> {_compact_result(result)}")
                else:
                    result = tools.dispatch(fn_name, fn_args)
                    print(f"  -> {_compact_result(result)}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str),
                })

        elif not session.concluded:
            # Text-only response and no conclusion. Send a minimal nudge
            # that doesn't push toward a specific action — just tells
            # the agent it has the floor.
            messages.append({
                "role": "user",
                "content": (
                    "Your turn. Continue the investigation, or call "
                    "conclude(summary, lessons) when you're done."
                ),
            })

        if response.usage:
            u = response.usage
            print(f"  [tokens: {u.prompt_tokens}p + {u.completion_tokens}c "
                  f"= {u.total_tokens}t]")
    else:
        # Loop exited via max_turns without conclude — the safety cap fired.
        print(f"\n*** Safety cap: agent ran {max_turns} turns without "
              f"calling conclude(). Forcing exit. ***")

    # Surface the conclusion (or lack thereof).
    print("\n" + "=" * 60)
    if session.conclusion:
        print("CONCLUSION")
        print("=" * 60)
        print(session.conclusion["summary"])
        if session.conclusion.get("lessons"):
            print(f"\nLessons recorded ({len(session.conclusion['lessons'])}):")
            for l in session.conclusion["lessons"]:
                print(f"  - {l}")
    else:
        print("Session ended without an explicit conclusion.")
    print("=" * 60)
    stats = tools.stats()
    print(f"Final stats: {json.dumps(stats, indent=2)}")


def _compact_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 40:
            v = v[:37] + "..."
        parts.append(f"{k}={v!r}")
    return ", ".join(parts)


def _compact_result(result: any) -> str:
    if isinstance(result, str):
        return result[:100]
    if isinstance(result, dict):
        if "error" in result:
            return f"ERROR: {result['error']}: {result.get('reason', '')}"
        if "run_id" in result:
            return f"run_id={result['run_id']} status={result.get('status', '?')}"
        return json.dumps(result, default=str)[:120]
    if isinstance(result, list):
        return f"[{len(result)} items]"
    return str(result)[:100]


def main() -> int:
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY in .env or environment")
        return 1

    client = OpenAI(api_key=api_key)

    budget = SessionBudget(
        max_runs=args.max_runs,
        max_gpu_min=args.max_gpu_min,
        max_high_trust_runs=args.max_high_trust,
    )

    session = Session.from_manifest(
        args.manifest,
        ledger_path=args.ledger,
        budget=budget,
        local=args.local,
    )
    tools = Tools(session)

    if args.import_tsv:
        n = session.import_history(args.import_tsv)
        print(f"Imported {n} historical runs")

    if args.dry_run:
        print("=== SYSTEM PROMPT ===")
        print(session.system_prompt())
        print("\n=== INITIAL CONTEXT ===")
        print(tools.context())
        print("\n=== TOOL SCHEMAS ===")
        for s in TOOL_SCHEMAS:
            print(f"  {s['name']}: {s['description']}")
        session.close()
        return 0

    def _run():
        run_agent(
            client, args.model, session, tools,
            max_turns=args.max_turns,
            hitl=args.hitl,
        )

    try:
        if args.local:
            print(f"Running in local mode (project: {session.manifest.project_root})\n")
            _run()
        else:
            # Start the Modal app so run_phase.spawn() works.
            print("Starting Modal app...")
            from autoresearch.run_phase import app as modal_app

            with modal_app.run() as app_handle:
                print(f"Modal app running (app_id={modal_app.app_id})")
                print(f"View at: https://modal.com/apps/{modal_app.app_id}\n")
                _run()
                print("\nStopping Modal app...")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        _graceful_shutdown(session)

    return 0


def _graceful_shutdown(session: Session) -> None:
    """Clean up on exit: mark in-flight runs, print summary."""
    launcher = session.launcher

    # Mark in-flight runs so they don't stay as "running" forever.
    inflight = list(launcher._inflight.keys())
    if inflight:
        print(f"\nCleaning up {len(inflight)} in-flight run(s)...")
        for run_id in inflight:
            launcher.ledger.set_status(
                run_id, "interrupted",
                "Session ended while run was in-flight"
            )
            print(f"  {run_id}: marked interrupted")
        launcher._inflight.clear()

    # Print final summary.
    stats = launcher.stats()
    print(f"\nSession summary:")
    print(f"  Total runs:  {stats.get('total', 0)}")
    print(f"  Kept:        {stats.get('kept', 0)}")
    print(f"  Discarded:   {stats.get('discarded', 0)}")
    print(f"  Failed:      {stats.get('failed', 0)}")
    gpu = stats.get('total_gpu_min', 0) or 0
    print(f"  GPU minutes: {gpu:.1f}")
    best = stats.get('best_primary')
    if best is not None:
        print(f"  Best {session.manifest.metrics.primary}: {best:.4f}")

    lessons = launcher.query_lessons()
    if lessons:
        print(f"\n  Lessons ({len(lessons)}):")
        for l in lessons:
            print(f"    - {l['text']}")

    print(f"\n  Ledger saved to: {launcher.ledger.db_path}")
    session.close()


if __name__ == "__main__":
    sys.exit(main())
