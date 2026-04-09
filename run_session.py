"""Interactive autoresearch session.

Starts a session, optionally imports historical runs, and enters a REPL
where you (or an LLM agent) can call tools to run experiments.

Usage:
    # Interactive (human at the keyboard):
    uv run python run_session.py

    # With custom budget:
    uv run python run_session.py --max-runs 50 --max-gpu-min 120

    # Import historical runs first:
    uv run python run_session.py --import-tsv ../hydra/results.tsv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Set manifest before any autoresearch imports.
MANIFEST_PATH = (Path(__file__).parent.parent / "hydra" / "autoresearch.toml").resolve()
os.environ["AUTORESEARCH_MANIFEST"] = str(MANIFEST_PATH)

from autoresearch.gate import SessionBudget
from autoresearch.session import Session
from autoresearch.tools import Tools


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive autoresearch session")
    p.add_argument("--manifest", default=str(MANIFEST_PATH),
                    help="Path to autoresearch.toml")
    p.add_argument("--ledger", default="session_ledger.db",
                    help="Path to session ledger SQLite file")
    p.add_argument("--import-tsv", default="",
                    help="Import historical runs from a TSV file on startup")
    p.add_argument("--max-runs", type=int, default=50)
    p.add_argument("--max-gpu-min", type=float, default=600)
    p.add_argument("--max-high-trust", type=int, default=10)
    p.add_argument("--print-system-prompt", action="store_true",
                    help="Print the LLM system prompt and exit")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    budget = SessionBudget(
        max_runs=args.max_runs,
        max_gpu_min=args.max_gpu_min,
        max_high_trust_runs=args.max_high_trust,
    )

    session = Session.from_manifest(
        args.manifest,
        ledger_path=args.ledger,
        budget=budget,
    )
    tools = Tools(session)

    if args.print_system_prompt:
        print(session.system_prompt())
        return 0

    # Import history if requested.
    if args.import_tsv:
        n = session.import_history(args.import_tsv)
        print(f"Imported {n} historical runs from {args.import_tsv}")

    # Print initial context.
    print("=" * 60)
    print("AUTORESEARCH SESSION")
    print("=" * 60)
    print(tools.context())
    print(f"Tools: {', '.join(tools.tool_names())}")
    print(f"Type a tool call as: tool_name key=value key=value ...")
    print(f"Type 'help' for usage, 'quit' to exit.")
    print()

    # REPL.
    while True:
        try:
            line = input("autoresearch> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not line:
            continue
        if line in ("quit", "exit", "q"):
            break
        if line == "help":
            _print_help(tools)
            continue
        if line == "system_prompt":
            print(session.system_prompt())
            continue

        # Parse: tool_name key=value key=value ...
        parts = line.split()
        tool_name = parts[0]
        kwargs: dict = {}
        positional_keys = _positional_keys(tool_name)

        pos_idx = 0
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                kwargs[k] = _parse_value(v)
            elif pos_idx < len(positional_keys):
                kwargs[positional_keys[pos_idx]] = _parse_value(part)
                pos_idx += 1
            else:
                kwargs[f"_arg{pos_idx}"] = part
                pos_idx += 1

        result = tools.dispatch(tool_name, kwargs)
        _print_result(result)

        # Check stop conditions after each action.
        sc = session.check_stop()
        if sc.triggered:
            print(f"\n*** STOP CONDITION: {sc.reason} ***\n")

    session.close()
    return 0


def _positional_keys(tool_name: str) -> list[str]:
    """Positional argument keys for common tools, so you can type
    `poll abc123` instead of `poll run_id=abc123`."""
    return {
        "poll": ["run_id"],
        "wait": ["run_id"],
        "cancel": ["run_id"],
        "set_status": ["run_id", "status", "note"],
        "launch": ["commit_sha", "phase", "hypothesis"],
        "best_runs": ["n"],
        "query": ["phase"],
    }.get(tool_name, [])


def _parse_value(s: str) -> any:
    """Try to parse as JSON (for dicts/lists/numbers), fall back to string."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return s


def _print_result(result: any) -> None:
    if isinstance(result, str):
        print(result)
    elif isinstance(result, dict):
        print(json.dumps(result, indent=2, default=str))
    elif isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                # Compact one-liner for list items.
                pm = item.get("primary_metric")
                pm_str = f"{pm:.4f}" if pm is not None else "---"
                print(
                    f"  {item.get('run_id', '?'):15s} "
                    f"{item.get('phase', ''):10s} "
                    f"pm={pm_str} "
                    f"[{item.get('status', '?')}] "
                    f"{item.get('observation', '')[:50]}"
                )
            else:
                print(f"  {item}")
    else:
        print(result)


def _print_help(tools: Tools) -> None:
    print("Available tools:")
    print()
    for name in tools.tool_names():
        fn = getattr(tools, name)
        doc = (fn.__doc__ or "").strip().split("\n")[0]
        print(f"  {name:15s}  {doc}")
    print()
    print("Usage: tool_name key=value key=value ...")
    print("Positional args work for common tools:")
    print("  launch abc123 quick \"my hypothesis\"")
    print("  poll run-id")
    print("  set_status run-id keep \"good result\"")
    print("  best_runs 10")
    print()


if __name__ == "__main__":
    sys.exit(main())
