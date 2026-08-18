"""Command-line entry point for safe Strategy Lab analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .dataset import AnalysisError, open_dataset
from .evaluate import evaluate_scope
from .output import write_analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tools.strategy_lab.analysis.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="evaluate a frozen Strategy Lab scope")
    analyze.add_argument("--dataset", type=Path, required=True)
    analyze.add_argument("--scope", default="development")
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--allow-partial-scope", action="store_true")
    analyze.add_argument("--allow-incomplete-dataset", action="store_true")
    analyze.add_argument("--unlock-scope", action="store_true")
    analyze.add_argument("--policy", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        dataset = open_dataset(
            args.dataset, allow_incomplete=args.allow_incomplete_dataset
        )
        scope = dataset.resolve_scope(
            args.scope,
            allow_partial=args.allow_partial_scope,
            unlock=args.unlock_scope,
            policy_path=args.policy,
        )
        result = evaluate_scope(dataset, scope)
        publication = write_analysis(result, args.output, dataset_root=dataset.root)
    except AnalysisError as exc:
        print(f"Strategy Lab analysis error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(publication, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
