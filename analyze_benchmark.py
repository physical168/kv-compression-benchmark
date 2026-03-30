"""
Plot latency and speedup from benchmark_runs.csv (output of main_eval.ipynb Step 6).

- Notebook with explanations: analyze_benchmark.ipynb
- CLI: python analyze_benchmark.py [path/to/benchmark_runs.csv]

Figures are written to figures/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Style: readable without custom fonts (avoids CJK font issues in matplotlib)
plt.rcParams.update(
    {
        "figure.figsize": (9, 5),
        "figure.dpi": 120,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    }
)


def main() -> None:
    root = Path(__file__).resolve().parent
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "benchmark_runs.csv"
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    for col in ("ea_s", "kvzip_s", "ratio"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ea_s", "kvzip_s", "ratio"])

    df["kvzip_over_ea"] = df["kvzip_s"] / df["ea_s"]

    by_ratio = df.groupby("ratio", sort=True).agg(
        ea_mean=("ea_s", "mean"),
        ea_std=("ea_s", "std"),
        kv_mean=("kvzip_s", "mean"),
        kv_std=("kvzip_s", "std"),
        n=("ea_s", "count"),
    ).reset_index()

    ratios = by_ratio["ratio"].values
    x = np.arange(len(ratios))
    w = 0.36

    # --- Figure 1: grouped bar (mean latency) ---
    fig, ax = plt.subplots()
    ax.bar(x - w / 2, by_ratio["ea_mean"], w, label="Expected Attention", color="#2E86AB", yerr=by_ratio["ea_std"], capsize=3)
    ax.bar(x + w / 2, by_ratio["kv_mean"], w, label="KVzip", color="#E94F37", yerr=by_ratio["kv_std"], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ratios])
    ax.set_xlabel("Compression ratio")
    ax.set_ylabel("Wall time per run (s)")
    ax.set_title("Mean generation latency vs compression ratio (±1 std)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_by_ratio_bars.png")
    plt.close(fig)

    # --- Figure 2: line plot ---
    fig, ax = plt.subplots()
    ax.plot(ratios, by_ratio["ea_mean"], "o-", color="#2E86AB", label="Expected Attention", linewidth=2, markersize=8)
    ax.fill_between(
        ratios,
        by_ratio["ea_mean"] - by_ratio["ea_std"],
        by_ratio["ea_mean"] + by_ratio["ea_std"],
        color="#2E86AB",
        alpha=0.2,
    )
    ax.plot(ratios, by_ratio["kv_mean"], "s-", color="#E94F37", label="KVzip", linewidth=2, markersize=8)
    ax.fill_between(
        ratios,
        by_ratio["kv_mean"] - by_ratio["kv_std"],
        by_ratio["kv_mean"] + by_ratio["kv_std"],
        color="#E94F37",
        alpha=0.2,
    )
    ax.set_xlabel("Compression ratio")
    ax.set_ylabel("Wall time per run (s)")
    ax.set_title("Latency vs compression ratio (shaded: ±1 std across tasks)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_by_ratio_lines.png")
    plt.close(fig)

    # --- Figure 3: KVzip / EA speedup ratio (distribution by compression) ---
    fig, ax = plt.subplots()
    parts = ax.violinplot(
        [df.loc[df["ratio"] == r, "kvzip_over_ea"].values for r in sorted(df["ratio"].unique())],
        positions=range(len(ratios)),
        showmeans=True,
        showmedians=True,
    )
    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels([str(r) for r in sorted(df["ratio"].unique())])
    ax.set_xlabel("Compression ratio")
    ax.set_ylabel("KVzip time / Expected Attention time")
    ax.set_title("Relative cost of KVzip vs Expected Attention (per row)")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Equal cost")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "kvzip_speedup_violin.png")
    plt.close(fig)

    # --- Figure 4: scatter — all runs ---
    fig, ax = plt.subplots()
    for r, c, m in zip(sorted(df["ratio"].unique()), plt.cm.viridis(np.linspace(0.2, 0.9, len(ratios))), ["o", "s", "^", "D", "v"]):
        sub = df[df["ratio"] == r]
        ax.scatter(sub["ea_s"], sub["kvzip_s"], label=f"ratio {r}", alpha=0.7, s=36, marker=m, edgecolors="white", linewidths=0.5)
    lim = max(df["ea_s"].max(), df["kvzip_s"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.4, label="y = x")
    ax.set_xlabel("Expected Attention (s)")
    ax.set_ylabel("KVzip (s)")
    ax.set_title("Per-run latency: EA vs KVzip (color = compression ratio)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_ea_vs_kvzip.png", bbox_inches="tight")
    plt.close(fig)

    # Console summary
    print(f"Read {len(df)} rows from {csv_path}")
    print(f"Wrote figures to {out_dir}/")
    print("\nSummary by ratio:")
    print(by_ratio.to_string(index=False))
    print(f"\nOverall: mean EA {df['ea_s'].mean():.3f}s, mean KVzip {df['kvzip_s'].mean():.3f}s, mean KVzip/EA {df['kvzip_over_ea'].mean():.3f}x")


if __name__ == "__main__":
    main()
