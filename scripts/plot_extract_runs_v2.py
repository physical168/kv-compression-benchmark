"""Plot curves from aggregated extract_runs_v2.csv (repo root).

Run from repo root:  python scripts/plot_extract_runs_v2.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "extract_runs_v2.csv"
OUT_DIR = ROOT / "figures"
OUT_MEAN_F1 = OUT_DIR / "extract_runs_v2_mean_f1_vs_ratio.png"
OUT_MEAN_ACC = OUT_DIR / "extract_runs_v2_mean_acc_vs_ratio.png"
OUT_BY_QUERY = OUT_DIR / "extract_runs_v2_f1_by_query.png"


def main() -> None:
    if not RUNS.is_file():
        raise SystemExit(f"Missing {RUNS}")

    df = pd.read_csv(RUNS)
    for col in ("query_id", "ratio", "method", "accuracy", "f1_macro", "n"):
        if col not in df.columns:
            raise SystemExit(f"Expected column {col!r} in {RUNS}")

    df["ratio"] = df["ratio"].astype(float)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Mean across queries per (ratio, method) — same spirit as eval_extract notebook aggregation
    avg = (
        df.groupby(["ratio", "method"])[["accuracy", "f1_macro"]]
        .mean()
        .reset_index()
    )

    def line_plot(y_col: str, y_label: str, out_path: Path, title: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        for mname, label in [("ea", "ExpectedAttention"), ("kvzip", "KVzip")]:
            sub = avg[avg["method"] == mname].sort_values("ratio")
            if len(sub) == 0:
                continue
            ax.plot(sub["ratio"], sub[y_col], marker="o", label=label)
        ax.set_xlabel("compression ratio")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print("Saved", out_path.resolve())

    line_plot(
        "f1_macro",
        "mean f1_macro (per-query macro, then average)",
        OUT_MEAN_F1,
        "Extract v2: mean F1 vs compression",
    )
    line_plot(
        "accuracy",
        "mean accuracy (per-query, then average)",
        OUT_MEAN_ACC,
        "Extract v2: mean accuracy vs compression",
    )

    # Per-query F1 panels
    qids = sorted(df["query_id"].unique())
    nq = len(qids)
    fig, axes = plt.subplots(1, nq, figsize=(4 * nq, 4), squeeze=False)
    for ax, qid in zip(axes[0], qids):
        part = df[df["query_id"] == qid]
        for mname, label in [("ea", "EA"), ("kvzip", "KVzip")]:
            sub = part[part["method"] == mname].sort_values("ratio")
            if len(sub) == 0:
                continue
            ax.plot(sub["ratio"], sub["f1_macro"], marker="o", label=label)
        ax.set_title(f"query_{int(qid):03d}")
        ax.set_xlabel("compression ratio")
        ax.set_ylabel("f1_macro")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Extract v2: F1 by query", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_BY_QUERY, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Saved", OUT_BY_QUERY.resolve())


if __name__ == "__main__":
    main()
