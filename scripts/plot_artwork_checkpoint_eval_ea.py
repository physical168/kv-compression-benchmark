#!/usr/bin/env python3
"""
Artwork plots:
  1) Checkpoint success + EA artwork + optional EA reviews (ea_kvzip_summary).
  2) Four methods on artwork, same metric: mean F1 per query vs compression ratio.
     - EA: from evaluation_results.csv (ExpectedAttentionPress).
     - KVzip, Finch_Full, Finch_CPT: from checkpoint CSV via EvaluationManager + ground_truth.

Outputs (under benchmarks/artwork_eval/figures by default):
  - integrated_metrics.csv
  - artwork_checkpoint_eval_ea.png
  - artwork_four_methods_mean_f1.png
  - four_methods_mean_f1.csv
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Repo-local import (script lives in scripts/)
_REPO = Path(__file__).resolve().parents[1]
_ART_EVAL = _REPO / "benchmarks" / "artwork_eval"
if str(_ART_EVAL) not in sys.path:
    sys.path.insert(0, str(_ART_EVAL))
from evaluation.evaluator import EvaluationManager  # noqa: E402


def _repo_root() -> Path:
    return _REPO


def load_checkpoint_latest_per_key(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    err = df.get("error", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
    df = df.assign(_ok=err.isin(["", "nan", "None"]))
    key_cols = ["method", "ratio", "record_id", "query"]
    for c in key_cols:
        if c not in df.columns:
            raise ValueError(f"checkpoint missing column {c}: {path}")
    df["ratio"] = df["ratio"].astype(float)
    df["record_id"] = df["record_id"].astype(int)
    df["query"] = df["query"].astype(str)
    df["method"] = df["method"].astype(str)
    df = df.sort_index().groupby(key_cols, as_index=False).tail(1)
    return df


def checkpoint_success_by_method_ratio(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["method", "ratio"], as_index=False)
        .agg(success_rate=("_ok", "mean"), n_rows=("method", "count"))
        .sort_values(["method", "ratio"])
    )
    g["source"] = "checkpoint"
    g["benchmark"] = "artwork"
    g["metric"] = "success_rate"
    return g.rename(columns={"method": "method_label"})


def eval_mean_f1_by_ratio(path: Path) -> pd.DataFrame:
    ev = pd.read_csv(path)
    if "f1" not in ev.columns or "ratio" not in ev.columns:
        raise ValueError(f"unexpected evaluation_results schema: {path}")
    ev["ratio"] = ev["ratio"].astype(float)
    press = ev.get("press", pd.Series([""] * len(ev))).astype(str)
    ev = ev.assign(_press=press)
    g = (
        ev.groupby(["_press", "ratio"], as_index=False)
        .agg(mean_f1=("f1", "mean"), n_queries=("query", "count"))
        .sort_values(["_press", "ratio"])
    )
    g["source"] = "evaluation_results"
    g["benchmark"] = "artwork"
    g["metric"] = "mean_f1_per_query"
    return g.rename(columns={"_press": "method_label"})


def ea_summary_macro(path: Path) -> pd.DataFrame:
    s = pd.read_csv(path)
    s = s[s["method"].astype(str).str.lower() == "ea"].copy()
    if s.empty:
        return pd.DataFrame(
            columns=[
                "method_label",
                "ratio",
                "value",
                "source",
                "benchmark",
                "metric",
                "use_cpt",
                "config",
            ]
        )
    s["ratio"] = s["compression_ratio"].astype(float)
    out = s.rename(columns={"f1_macro": "value", "method": "method_label"})
    out["source"] = "ea_kvzip_summary"
    out["benchmark"] = "reviews1000 (ea rows only)"
    out["metric"] = "f1_macro"
    return out[
        [
            "method_label",
            "ratio",
            "value",
            "source",
            "benchmark",
            "metric",
            "use_cpt",
            "config",
        ]
    ].sort_values(["use_cpt", "ratio"])


def evaluate_checkpoint_methods(
    ck: pd.DataFrame,
    config_path: Path,
    tmp_root: Path,
    model_tag: str = "llama3-llava-next-8b-hf",
    dataset: str = "artwork",
) -> pd.DataFrame:
    """Run EvaluationManager on successful checkpoint rows (KVzip / Finch_*)."""
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    ok = ck[ck["_ok"]].copy()
    methods = [m for m in ("KVzip", "Finch_Full", "Finch_CPT") if m in set(ok["method"])]
    for m in methods:
        for ratio in sorted(ok["ratio"].unique()):
            sub = ok[(ok["method"] == m) & (ok["ratio"] == float(ratio))]
            if sub.empty:
                continue
            rdir = tmp_root / dataset / model_tag / m / f"{float(ratio):.2f}"
            rdir.mkdir(parents=True, exist_ok=True)
            out = sub[["record_id", "query", "answer"]].copy()
            out.to_csv(rdir / "results.csv", index=False, encoding="utf-8")

    mgr = EvaluationManager(config_path=config_path, results_dir=tmp_root)
    ev_df = mgr.evaluate_all()
    if ev_df.empty:
        return pd.DataFrame(columns=["press", "ratio", "mean_f1", "plot_label"])
    g = (
        ev_df.groupby(["press", "ratio"], as_index=False)
        .agg(mean_f1=("f1", "mean"), n_queries=("query", "count"))
        .sort_values(["press", "ratio"])
    )
    label_map = {
        "KVzip": "KVzip",
        "Finch_Full": "Finch (no CPT)",
        "Finch_CPT": "Finch + CPT",
    }
    g["plot_label"] = g["press"].map(lambda x: label_map.get(str(x), str(x)))
    return g


def build_integrated_table(
    ck_g: pd.DataFrame,
    ev_g: pd.DataFrame,
    ea_g: pd.DataFrame,
    four_f1: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    for _, r in ck_g.iterrows():
        rows.append(
            {
                "source": r["source"],
                "benchmark": r["benchmark"],
                "method_label": r["method_label"],
                "ratio": float(r["ratio"]),
                "metric": r["metric"],
                "value": float(r["success_rate"]),
                "n": int(r["n_rows"]),
            }
        )
    for _, r in ev_g.iterrows():
        rows.append(
            {
                "source": r["source"],
                "benchmark": r["benchmark"],
                "method_label": r["method_label"],
                "ratio": float(r["ratio"]),
                "metric": r["metric"],
                "value": float(r["mean_f1"]),
                "n": int(r["n_queries"]),
            }
        )
    for _, r in ea_g.iterrows():
        rows.append(
            {
                "source": r["source"],
                "benchmark": r["benchmark"],
                "method_label": f"ea (use_cpt={bool(r['use_cpt'])})",
                "ratio": float(r["ratio"]),
                "metric": r["metric"],
                "value": float(r["value"]),
                "n": np.nan,
            }
        )
    if four_f1 is not None and not four_f1.empty:
        for _, r in four_f1.iterrows():
            rows.append(
                {
                    "source": "checkpoint+evaluator",
                    "benchmark": "artwork",
                    "method_label": str(r["plot_label"]),
                    "ratio": float(r["ratio"]),
                    "metric": "mean_f1_per_query",
                    "value": float(r["mean_f1"]),
                    "n": int(r["n_queries"]),
                }
            )
    return pd.DataFrame(rows)


def plot_figure(ck_g: pd.DataFrame, ev_g: pd.DataFrame, ea_g: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    ax = axes[0]
    methods = sorted(ck_g["method_label"].unique())
    ratios = sorted(ck_g["ratio"].unique())
    x = np.arange(len(ratios))
    w = 0.8 / max(1, len(methods))
    for i, m in enumerate(methods):
        sub = ck_g[ck_g["method_label"] == m].set_index("ratio").reindex(ratios)
        vals = sub["success_rate"].to_numpy(dtype=float)
        ax.bar(x + (i - (len(methods) - 1) / 2) * w, vals, width=w, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:g}" for r in ratios])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("compression ratio (artwork checkpoint)")
    ax.set_ylabel("success rate (no error row)")
    ax.set_title("Checkpoint completion (artwork)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ev_ea = ev_g[ev_g["method_label"].str.contains("ExpectedAttention", case=False, na=False)]
    if not ev_ea.empty:
        ax.plot(
            ev_ea["ratio"],
            ev_ea["mean_f1"],
            "o-",
            linewidth=2,
            markersize=7,
            label="EA (artwork, mean F1 / query)",
        )

    if not ea_g.empty:
        for use_cpt, sty in [(False, "--"), (True, ":")]:
            sub = ea_g[ea_g["use_cpt"] == use_cpt].sort_values("ratio")
            if sub.empty:
                continue
            ax.plot(
                sub["ratio"],
                sub["value"],
                sty,
                linewidth=2,
                markersize=5,
                label=f"ea_kvzip_summary: EA f1_macro (reviews, use_cpt={use_cpt})",
            )

    ax.set_xlabel("compression ratio")
    ax.set_ylabel("F1 (macro or mean per query — see legend)")
    ax.set_title("Evaluation + EA summary (mixed benchmarks)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "Artwork: checkpoint vs evaluation_results; EA reviews curve from ea_kvzip_summary",
        fontsize=11,
    )
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_four_methods_mean_f1(
    ev_g: pd.DataFrame,
    ck_eval: pd.DataFrame,
    out_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    styles = {
        "EA (ExpectedAttention)": dict(marker="o", linestyle="-", linewidth=2.2),
        "KVzip": dict(marker="s", linestyle="-", linewidth=2.0),
        "Finch (no CPT)": dict(marker="^", linestyle="-", linewidth=2.0),
        "Finch + CPT": dict(marker="D", linestyle="-", linewidth=2.0),
    }

    ev_ea = ev_g[ev_g["method_label"].str.contains("ExpectedAttention", case=False, na=False)]
    if not ev_ea.empty:
        ax.plot(
            ev_ea["ratio"],
            ev_ea["mean_f1"],
            label="EA (ExpectedAttention)",
            **styles["EA (ExpectedAttention)"],
        )

    if not ck_eval.empty:
        for lbl in ["KVzip", "Finch (no CPT)", "Finch + CPT"]:
            sub = ck_eval[ck_eval["plot_label"] == lbl].sort_values("ratio")
            if sub.empty:
                continue
            ax.plot(sub["ratio"], sub["mean_f1"], label=lbl, **styles[lbl])

    ax.set_xlabel("compression ratio (artwork)")
    ax.set_ylabel("mean F1 per query (20 queries)")
    ax.set_title("Artwork: four methods (same evaluator as benchmark)")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=_repo_root() / "checkpoint_all (11).csv",
    )
    ap.add_argument(
        "--evaluation",
        type=Path,
        default=_repo_root() / "evaluation_results.csv",
    )
    ap.add_argument(
        "--ea-summary",
        type=Path,
        default=_repo_root() / "ea_kvzip_summary.csv",
    )
    ap.add_argument(
        "--eval-config",
        type=Path,
        default=_repo_root() / "benchmarks" / "artwork_eval" / "evaluation" / "evaluation_config.yaml",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_repo_root() / "benchmarks" / "artwork_eval" / "figures",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "integrated_metrics.csv"
    out_png = args.out_dir / "artwork_checkpoint_eval_ea.png"
    out_four_png = args.out_dir / "artwork_four_methods_mean_f1.png"
    out_four_csv = args.out_dir / "four_methods_mean_f1.csv"

    ck = load_checkpoint_latest_per_key(args.checkpoint)
    ck_g = checkpoint_success_by_method_ratio(ck)
    ev_g = eval_mean_f1_by_ratio(args.evaluation)
    ea_g = ea_summary_macro(args.ea_summary)

    tmp_eval = args.out_dir / "_tmp_checkpoint_eval"
    ck_eval = evaluate_checkpoint_methods(ck, args.eval_config, tmp_eval)
    if tmp_eval.exists():
        shutil.rmtree(tmp_eval, ignore_errors=True)

    # Long table for four-method F1 (EA from file + checkpoint-derived)
    ea_f1 = ev_g[ev_g["method_label"].str.contains("ExpectedAttention", case=False, na=False)].copy()
    ea_f1 = ea_f1.assign(plot_label="EA (ExpectedAttention)", press="ExpectedAttentionPress")
    four_tbl = pd.concat(
        [
            ea_f1[["press", "ratio", "mean_f1", "n_queries", "plot_label"]],
            ck_eval[["press", "ratio", "mean_f1", "n_queries", "plot_label"]],
        ],
        ignore_index=True,
    )
    four_tbl.to_csv(out_four_csv, index=False, encoding="utf-8")

    integrated = build_integrated_table(ck_g, ev_g, ea_g, ck_eval)
    integrated.to_csv(out_csv, index=False, encoding="utf-8")

    plot_figure(ck_g, ev_g, ea_g, out_png)
    plot_four_methods_mean_f1(ev_g, ck_eval, out_four_png)

    print("Wrote:", out_csv)
    print("Wrote:", out_png)
    print("Wrote:", out_four_csv)
    print("Wrote:", out_four_png)


if __name__ == "__main__":
    main()
