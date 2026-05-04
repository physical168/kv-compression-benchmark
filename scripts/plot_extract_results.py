"""Plot extract-eval curves from extract_predictions_checkpoint.csv (repo root).

Run from repo root:  python scripts/plot_extract_results.py

Handles gold labels True/False for yes-no tasks (q14, q17, q18) like yes/no.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "extract_predictions_checkpoint.csv"
COMPRESSION_RATIOS = [0.2, 0.4, 0.6, 0.8, 0.9]
OUT_DIR = ROOT / "figures"
OUT_F1 = OUT_DIR / "extract_f1_vs_ratio.png"
OUT_ACC = OUT_DIR / "extract_acc_vs_ratio.png"
OUT_BY_QUERY = OUT_DIR / "by_query"
OUT_GRID = OUT_DIR / "extract_all_queries_f1_acc_grid.png"
OUT_RUNS = ROOT / "extract_runs.csv"


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def parse_yes_no(text: str) -> str | None:
    t = norm_ws(text)
    if re.search(r"\byes\b", t) and not re.search(r"\bno\b", t[:20]):
        return "yes"
    if re.search(r"\bno\b", t) and not re.search(r"\byes\b", t[:20]):
        return "no"
    if t.startswith("y") and not t.startswith("n"):
        return "yes"
    if t.startswith("n"):
        return "no"
    return None


def gold_to_yes_no(raw: str) -> str | None:
    """Map CSV gold to yes/no for binary extract tasks."""
    t = norm_ws(str(raw).strip())
    if t in ("yes", "no"):
        return t
    if t in ("true", "1"):
        return "yes"
    if t in ("false", "0"):
        return "no"
    return parse_yes_no(str(raw))


def parse_sentiment(text: str) -> str | None:
    t = norm_ws(text)
    for w in ("mixed", "neutral", "positive", "negative"):
        if re.search(rf"\b{re.escape(w)}\b", t):
            return w
    if "positive" in t and "negative" not in t[:40]:
        return "positive"
    if "negative" in t and "positive" not in t[:40]:
        return "negative"
    return None


def parse_aspect(text: str):
    t = norm_ws(text)
    if "special effect" in t:
        return "special effects"
    for o in ("cinematography", "soundtrack", "dialogue", "acting", "plot"):
        if o in t:
            return o
    if re.search(r"\bnone\b", t):
        return "none"
    return None


def gold_aspect_norm(g: str) -> str:
    return norm_ws(str(g))


def em_soft(pred: str, gold: str) -> bool:
    p = norm_ws(pred).strip(" \t")
    g = norm_ws(gold).strip(" \t")
    for stripchars in (".", ",", "!", "?", ":", ";", "'", '"'):
        p = p.strip(stripchars)
        g = g.strip(stripchars)
    if g == "none" and p.startswith("n"):
        return True
    if p == g:
        return True
    if len(g) > 2 and g in p:
        return True
    if len(p) > 2 and p in g:
        return True
    return False


def score_pair(query_id: int, pred: str, gold: str) -> tuple[float, str]:
    gold_s = str(gold).strip()
    if query_id in (14, 17, 18):
        pg = parse_yes_no(pred)
        gg = gold_to_yes_no(gold_s)
        ok = (pg == gg) if (pg and gg) else em_soft(pred, gold_s)
        return (1.0 if ok else 0.0), "f1_binary"
    if query_id == 10:
        pg, gg = parse_sentiment(pred), norm_ws(gold_s)
        ok = pg == gg if pg else em_soft(pred, gold_s)
        return (1.0 if ok else 0.0), "f1_multiclass"
    if query_id == 13:
        pg = parse_aspect(pred)
        gg = gold_aspect_norm(gold_s)
        ok = (pg == gg) if pg else em_soft(pred, gold_s)
        return (1.0 if ok else 0.0), "f1_multiclass"
    if query_id in (11, 12, 16, 15, 19):
        return (1.0 if em_soft(pred, gold_s) else 0.0), "em"
    return (1.0 if em_soft(pred, gold_s) else 0.0), "unknown"


def aggregate(ck_ok: pd.DataFrame) -> pd.DataFrame:
    rows_summary = []
    for qid in range(10, 20):
        part = ck_ok[ck_ok["query_id"].astype(int) == qid]
        for ratio in COMPRESSION_RATIOS:
            for mname in ("ea", "kvzip"):
                sub = part[
                    (part["ratio"].astype(float) == float(ratio)) & (part["method"] == mname)
                ]
                if len(sub) == 0:
                    continue
                y_true, y_pred, correct = [], [], []
                for _, r in sub.iterrows():
                    s, _tag = score_pair(qid, str(r["pred"]), str(r["gold"]))
                    correct.append(s)
                    if qid in (10, 13):
                        pg = parse_sentiment(str(r["pred"])) if qid == 10 else parse_aspect(str(r["pred"]))
                        gg = norm_ws(str(r["gold"]))
                        if qid == 13:
                            gg = str(gold_aspect_norm(str(r["gold"])))
                        if pg:
                            y_pred.append(pg)
                            y_true.append(gg)
                    elif qid in (14, 17, 18):
                        pg = parse_yes_no(str(r["pred"]))
                        gg = gold_to_yes_no(str(r["gold"]))
                        if pg and gg:
                            y_pred.append(pg)
                            y_true.append(gg)
                acc = float(sum(correct) / len(correct)) if correct else float("nan")
                if len(y_true) > 0 and len(y_pred) == len(y_true):
                    try:
                        f1m = f1_score(y_true, y_pred, average="macro")
                    except Exception:
                        f1m = acc
                else:
                    f1m = acc
                rows_summary.append(
                    {
                        "query_id": qid,
                        "ratio": ratio,
                        "method": mname,
                        "accuracy": acc,
                        "f1_macro": f1m,
                        "n": len(sub),
                    }
                )
    return pd.DataFrame(rows_summary)


def main() -> None:
    ck_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CHECKPOINT
    if not ck_path.is_file():
        print("Missing checkpoint:", ck_path, file=sys.stderr)
        sys.exit(1)

    ck = pd.read_csv(ck_path, dtype=str, low_memory=False)
    _err = ck["error"].fillna("").astype(str).str.strip()
    ck_ok = ck[(_err == "") | (_err.str.lower() == "nan")].copy()
    ck_ok = ck_ok[ck_ok["pred"].notna()]

    sum_df = aggregate(ck_ok)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sum_df.to_csv(OUT_RUNS, index=False)
    print("Wrote", OUT_RUNS.resolve())

    avg = (
        sum_df.groupby(["ratio", "method"])[["accuracy", "f1_macro"]].mean().reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for mname, label in [("ea", "ExpectedAttention"), ("kvzip", "KVzip")]:
        sub = avg[avg["method"] == mname]
        ax.plot(sub["ratio"], sub["f1_macro"], marker="o", label=label, linewidth=2, markersize=7)
    ax.set_xlabel("compression ratio", fontsize=11)
    ax.set_ylabel("mean F1 (macro per query, then average)", fontsize=11)
    ax.set_title("Extract tasks (query_010–019): F1 vs compression", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(COMPRESSION_RATIOS)
    fig.tight_layout()
    fig.savefig(OUT_F1, dpi=150)
    plt.close(fig)
    print("Saved", OUT_F1.resolve())

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for mname, label in [("ea", "ExpectedAttention"), ("kvzip", "KVzip")]:
        sub = avg[avg["method"] == mname]
        ax2.plot(sub["ratio"], sub["accuracy"], marker="s", label=label, linewidth=2, markersize=7)
    ax2.set_xlabel("compression ratio", fontsize=11)
    ax2.set_ylabel("mean accuracy (per query, then average)", fontsize=11)
    ax2.set_title("Extract tasks (query_010–019): Accuracy vs compression", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(COMPRESSION_RATIOS)
    fig2.tight_layout()
    fig2.savefig(OUT_ACC, dpi=150)
    plt.close(fig2)
    print("Saved", OUT_ACC.resolve())

    # One figure: 10 rows × 2 cols = F1 | Accuracy per query (query_010 … query_019)
    nq = 10
    fig_grid, axes = plt.subplots(
        nq,
        2,
        figsize=(10, 22),
        sharex=True,
        constrained_layout=True,
    )
    legend_done_f1 = False
    legend_done_acc = False
    for row, qid in enumerate(range(10, 20)):
        part = sum_df[sum_df["query_id"].astype(int) == qid]
        ax_f1 = axes[row, 0]
        ax_acc = axes[row, 1]
        if part.empty:
            ax_f1.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax_f1.transAxes)
            ax_acc.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax_acc.transAxes)
        else:
            for mname, label in [("ea", "EA"), ("kvzip", "KVzip")]:
                sub = part[part["method"] == mname].sort_values("ratio")
                if sub.empty:
                    continue
                ax_f1.plot(
                    sub["ratio"],
                    sub["f1_macro"],
                    marker="o",
                    label=label if not legend_done_f1 else None,
                    linewidth=1.4,
                    markersize=4,
                )
                ax_acc.plot(
                    sub["ratio"],
                    sub["accuracy"],
                    marker="s",
                    label=label if not legend_done_acc else None,
                    linewidth=1.4,
                    markersize=4,
                )
            legend_done_f1 = True
            legend_done_acc = True
        ax_f1.set_ylim(0, 1.05)
        ax_acc.set_ylim(0, 1.05)
        ax_f1.set_ylabel("F1", fontsize=8)
        ax_acc.set_ylabel("Acc", fontsize=8)
        ax_f1.set_title(f"query_{qid:03d}", fontsize=9, fontweight="bold")
        ax_acc.set_title(f"query_{qid:03d}", fontsize=9, fontweight="bold")
        ax_f1.grid(True, alpha=0.25)
        ax_acc.grid(True, alpha=0.25)
        ax_f1.set_xticks(COMPRESSION_RATIOS)
        ax_acc.set_xticks(COMPRESSION_RATIOS)
    axes[-1, 0].set_xlabel("compression ratio", fontsize=9)
    axes[-1, 1].set_xlabel("compression ratio", fontsize=9)
    axes[0, 0].legend(loc="lower left", fontsize=7)
    axes[0, 1].legend(loc="lower left", fontsize=7)
    fig_grid.suptitle("Extract eval per query: F1 (left) vs Accuracy (right)", fontsize=12, y=1.002)
    fig_grid.savefig(OUT_GRID, dpi=150, bbox_inches="tight")
    plt.close(fig_grid)
    print("Saved", OUT_GRID.resolve())

    # Per-query figures (same y-scale across queries for F1/acc can mislead; keep auto per figure)
    OUT_BY_QUERY.mkdir(parents=True, exist_ok=True)
    for qid in range(10, 20):
        part = sum_df[sum_df["query_id"].astype(int) == qid]
        if part.empty:
            continue
        fig_q, (ax_f1, ax_acc) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        for mname, label in [("ea", "ExpectedAttention"), ("kvzip", "KVzip")]:
            sub = part[part["method"] == mname].sort_values("ratio")
            if sub.empty:
                continue
            ax_f1.plot(
                sub["ratio"],
                sub["f1_macro"],
                marker="o",
                label=label,
                linewidth=2,
                markersize=7,
            )
            ax_acc.plot(
                sub["ratio"],
                sub["accuracy"],
                marker="s",
                label=label,
                linewidth=2,
                markersize=7,
            )
        ax_f1.set_ylabel("F1 (macro)", fontsize=11)
        ax_f1.set_title(f"query_{qid:03d} — F1 vs compression", fontsize=12)
        ax_f1.legend(loc="best")
        ax_f1.grid(True, alpha=0.3)
        ax_f1.set_xticks(COMPRESSION_RATIOS)
        ax_acc.set_xlabel("compression ratio", fontsize=11)
        ax_acc.set_ylabel("accuracy", fontsize=11)
        ax_acc.set_title(f"query_{qid:03d} — Accuracy vs compression", fontsize=12)
        ax_acc.legend(loc="best")
        ax_acc.grid(True, alpha=0.3)
        ax_acc.set_xticks(COMPRESSION_RATIOS)
        fig_q.tight_layout()
        fpq = OUT_BY_QUERY / f"extract_q{qid:03d}_f1_acc.png"
        fig_q.savefig(fpq, dpi=150)
        plt.close(fig_q)
        print("Saved", fpq.resolve())


if __name__ == "__main__":
    main()
