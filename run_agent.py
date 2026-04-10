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
    p.add_argument("--max-turns", type=int, default=30,
                    help="Max agent turns before stopping")
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
    max_turns: int = 30,
    hitl: bool = False,
) -> None:
    """Run the agent loop: context → LLM proposes → call tools → repeat."""

    system = session.system_prompt()
    openai_tools = build_openai_tools()
    messages: list[dict] = [{"role": "system", "content": system}]

    # Seed with initial context.
    initial_context = tools.context()
    messages.append({
        "role": "user",
        "content": (
            "Session started. Here is the current state:\n\n"
            f"{initial_context}\n\n"
            "Review the context and begin the autoresearch loop. "
            "Start by checking the best runs and lessons, then propose "
            "your first experiment."
        ),
    })

    for turn in range(max_turns):
        print(f"\n{'='*60}")
        print(f"Turn {turn + 1}/{max_turns}")
        print(f"{'='*60}")

        # Check stop conditions.
        sc = session.check_stop()
        if sc.triggered:
            print(f"\n*** STOP: {sc.reason} ***")
            # Tell the agent to wrap up.
            messages.append({
                "role": "user",
                "content": f"Session stop condition triggered: {sc.reason}. "
                           f"Summarize what was accomplished and any lessons learned.",
            })
            response = client.chat.completions.create(
                model=model, messages=messages,
            )
            print(f"\nAgent: {response.choices[0].message.content}")
            break

        # Call the LLM.
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump())

        # If the LLM wants to call tools.
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

        # If the LLM just sends text (reasoning / analysis).
        elif msg.content:
            print(f"\nAgent: {msg.content}")

            # After reasoning, nudge to continue the loop.
            if turn < max_turns - 1:
                messages.append({
                    "role": "user",
                    "content": "Continue the autoresearch loop. "
                               "Propose the next experiment or call tools as needed.",
                })

        # Usage tracking.
        if response.usage:
            u = response.usage
            print(f"  [tokens: {u.prompt_tokens}p + {u.completion_tokens}c = {u.total_tokens}t]")

    print("\n" + "=" * 60)
    print("Session complete.")
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
