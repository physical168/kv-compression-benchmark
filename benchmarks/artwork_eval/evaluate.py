"""CLI: evaluate artwork ``results.csv`` trees against bundled ground truth.

Same layout as CompressionExperiments ``evaluate.py``:
``python evaluate.py --results-dir <RUN_DIR>/results``
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from evaluation.evaluator import EvaluationManager  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate compression experiment results (artwork) against ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(_HERE / "results"),
        help="Root of results/{dataset}/{model_tag}/{press}/{ratio}/results.csv",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_HERE / "evaluation" / "evaluation_config.yaml"),
        help="Path to evaluation_config.yaml",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_HERE / "evaluation_results.csv"),
        help="Output CSV for aggregated metrics.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary table instead of saving CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manager = EvaluationManager(
        config_path=args.config,
        results_dir=args.results_dir,
    )

    logger.info("Evaluating results under: %s", args.results_dir)
    df = manager.evaluate_all()

    if df.empty:
        print("No results found to evaluate.")
        return

    group_cols = ["dataset", "model_tag", "press", "ratio"]
    metric_cols = ["precision", "recall", "f1"]

    def _agg(subset):
        return subset.groupby(group_cols)[metric_cols].mean().round(4)

    filter_df = df[df["query_type"] == "filter"]
    extract_df = df[df["query_type"] == "extract"]

    if args.summary:
        for label, subset in [("Filter", filter_df), ("Extract", extract_df), ("All", df)]:
            print(f"\n=== {label} queries ===")
            print(_agg(subset).to_string())
    else:
        parts = []
        for label, subset in [("filter", filter_df), ("extract", extract_df), ("all", df)]:
            agg = _agg(subset).reset_index()
            agg.insert(len(group_cols), "query_type", label)
            parts.append(agg)
        summary = pd.concat(parts, ignore_index=True).sort_values(
            group_cols + ["query_type"]
        )
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_path, index=False)
        logger.info("Evaluation results saved to %s", out_path)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
