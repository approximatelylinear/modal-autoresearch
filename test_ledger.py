"""M2 validation: import hydra/results.tsv into the ledger, export, and diff.

Usage:
    uv run python test_ledger.py

The round-trip must be clean: import → export should reproduce the original
TSV columns (commit, track, mode, metrics, status, description). Run IDs
and internal fields are allowed to differ (they don't exist in the original).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from autoresearch.ledger import Ledger

HYDRA_TSV = Path(__file__).parent.parent / "hydra" / "results.tsv"


def main() -> int:
    if not HYDRA_TSV.exists():
        print(f"ERROR: {HYDRA_TSV} not found")
        return 1

    original = HYDRA_TSV.read_text().strip()
    original_lines = original.splitlines()
    print(f"Original: {len(original_lines) - 1} data rows")

    # Import into a temp ledger.
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        ledger = Ledger(db_path)

        n = ledger.import_tsv(HYDRA_TSV, project="hydra")
        print(f"Imported: {n} rows")

        # Export back.
        exported = ledger.tsv_export("hydra").strip()
        exported_lines = exported.splitlines()
        print(f"Exported: {len(exported_lines) - 1} data rows")

        # Compare line by line.
        errors = 0
        max_lines = max(len(original_lines), len(exported_lines))
        for i in range(max_lines):
            orig = original_lines[i] if i < len(original_lines) else "<MISSING>"
            exp = exported_lines[i] if i < len(exported_lines) else "<MISSING>"
            if orig != exp:
                errors += 1
                if errors <= 10:
                    print(f"\n  line {i + 1} DIFF:")
                    print(f"    orig: {orig}")
                    print(f"    got:  {exp}")

        if errors > 10:
            print(f"\n  ... and {errors - 10} more diffs")

        # Also test query + best_runs while we're at it.
        stats = ledger.stats("hydra")
        print(f"\nStats: {stats}")

        best = ledger.best_runs("hydra", phase="quick", n=3)
        print(f"\nTop 3 quick runs by primary_metric (avg_ndcg10):")
        for r in best:
            print(f"  {r['run_id']}: {r['primary_metric']:.4f} [{r['status']}]")

        # Test lessons.
        lid = ledger.add_lesson(
            "FiLM with trainable A/B collapses at extensive scale",
            evidence="tsv-0008",
        )
        lessons = ledger.query_lessons()
        print(f"\nLessons: {len(lessons)}")
        for l in lessons:
            print(f"  {l['lesson_id'][:8]}... {l['text']}")

        ledger.close()

    print(f"\n{'PASS' if errors == 0 else 'FAIL'}: {errors} diffs")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
