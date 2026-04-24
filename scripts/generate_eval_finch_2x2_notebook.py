"""Generate eval_finch_2x2_qwen05b.ipynb for Colab runs.

Run from repo root:
    python scripts/generate_eval_finch_2x2_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval_finch_2x2_qwen05b.ipynb"


def main() -> None:
    cells = [
        md(
            """# Finch 2x2 evaluation (Qwen2.5-0.5B, sentiment on reviews_1000)

This Colab-first notebook runs a 2x2 grid:

- `FINCH_ENABLED` in `{False, True}`
- `USE_CPT` in `{False, True}`

Data source is `reviews_1000.csv` with:

- input text: `reviewtext` (or `cpt` when `USE_CPT=True`)
- label: `scoresentiment` (`POSITIVE`/`NEGATIVE`)

Outputs are saved to Google Drive under `RUN_DIR`:

- `finch_2x2_runs.csv` (row-level predictions + latency)
- `finch_2x2_summary.csv` (per-config metrics)
- figures for quality/latency comparisons
"""
        ),
        md("### Step 1: Install dependencies"),
        code(
            """!pip install -q transformers accelerate pandas scikit-learn matplotlib tqdm kvpress"""
        ),
        md(
            """### Step 2: Mount Google Drive and set output/input paths

If your dataset is in a different Drive folder, edit `DATASET_PATH`.
"""
        ),
        code(
            """from pathlib import Path
import os

RUN_ON_COLAB = os.path.isdir("/content")
USE_GOOGLE_DRIVE = True
DRIVE_SUBDIR = "kv-compression-benchmark/finch_2x2_qwen05b"

if RUN_ON_COLAB and USE_GOOGLE_DRIVE:
    try:
        from google.colab import drive

        drive.mount("/content/drive", force_remount=False)
    except ImportError:
        USE_GOOGLE_DRIVE = False
        print("google.colab not found; fallback to local paths.")

if RUN_ON_COLAB and USE_GOOGLE_DRIVE and Path("/content/drive/MyDrive").is_dir():
    RUN_DIR = Path("/content/drive/MyDrive") / DRIVE_SUBDIR
elif RUN_ON_COLAB:
    RUN_DIR = Path("/content/finch_2x2_workspace")
else:
    RUN_DIR = Path("finch_2x2_output")

RUN_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = RUN_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = Path("/content/drive/MyDrive/kv-compression-benchmark/reviews_1000.csv")
if not DATASET_PATH.is_file() and RUN_ON_COLAB:
    DATASET_PATH = Path("/content/reviews_1000.csv")
if not DATASET_PATH.is_file():
    DATASET_PATH = Path("reviews_1000.csv")

print("RUN_DIR:", RUN_DIR.resolve())
print("DATASET_PATH:", DATASET_PATH.resolve())
"""
        ),
        md("### Step 3: Load model (Qwen2.5-0.5B-Instruct)"),
        code(
            """import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
if getattr(tokenizer, "pad_token_id", None) is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map="auto",
)

print("Model loaded:", model_id)
if torch.cuda.is_available():
    print("CUDA:", torch.cuda.get_device_name(0))
"""
        ),
        md("### Step 4: Configuration"),
        code(
            """from __future__ import annotations

import re
import time
import csv
import gc
import pandas as pd

RANDOM_SEED = 42
MAX_ROWS = 1000  # full run; checkpoint resume skips completed rows
MAX_NEW_TOKENS = 8
RESUME_FROM_CHECKPOINT = True
FINCH_COMPRESSION_RATIO = 0.5

# 2x2 grid: Finch on/off × CPT with/without
CONFIGS = [
    {"finch_enabled": False, "use_cpt": False},
    {"finch_enabled": False, "use_cpt": True},
    {"finch_enabled": True, "use_cpt": False},
    {"finch_enabled": True, "use_cpt": True},
]

RUNS_PATH = RUN_DIR / "finch_2x2_runs.csv"
SUMMARY_PATH = RUN_DIR / "finch_2x2_summary.csv"


def normalize_label(s: str) -> str | None:
    t = str(s).strip().lower()
    if t in ("positive", "pos", "1", "true"):
        return "positive"
    if t in ("negative", "neg", "0", "false"):
        return "negative"
    return None


def parse_sentiment(pred_text: str) -> str | None:
    t = str(pred_text).strip().lower()
    if re.search(r"\\bpositive\\b", t):
        return "positive"
    if re.search(r"\\bnegative\\b", t):
        return "negative"
    if t.startswith("pos"):
        return "positive"
    if t.startswith("neg"):
        return "negative"
    return None


def cfg_name(cfg: dict) -> str:
    return f"finch_{int(cfg['finch_enabled'])}_cpt_{int(cfg['use_cpt'])}"


CK_FIELDS = [
    "config",
    "finch_enabled",
    "use_cpt",
    "row_id",
    "reviewid",
    "gold",
    "pred_raw",
    "pred_label",
    "latency_ms",
    "error",
]


def load_done(path):
    if not path.is_file() or not RESUME_FROM_CHECKPOINT:
        return set()
    ck = pd.read_csv(path)
    done = set()
    for _, r in ck.iterrows():
        err = str(r.get("error", "")).strip().lower()
        if err in ("", "nan"):
            done.add((str(r["config"]), str(r["row_id"])))
    return done


def append_row(path, row):
    newfile = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CK_FIELDS, extrasaction="ignore")
        if newfile:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CK_FIELDS})
        f.flush()
        os.fsync(f.fileno())
"""
        ),
        md("### Step 5: Finch adapter (NVIDIA/kvpress FinchPress)"),
        code(
            """from kvpress import FinchPress


def run_generate_with_optional_finch(prompt: str, finch_enabled: bool) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "pad_token_id": tokenizer.pad_token_id,
    }
    with torch.no_grad():
        if finch_enabled:
            # Apply NVIDIA/kvpress Finch compression during prefill.
            with FinchPress(compression_ratio=FINCH_COMPRESSION_RATIO)(model):
                out = model.generate(**inputs, **gen_kwargs)
        else:
            out = model.generate(**inputs, **gen_kwargs)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text
"""
        ),
        md("### Step 6: Run 2x2 inference with checkpoint"),
        code(
            """import random
from tqdm.auto import tqdm

random.seed(RANDOM_SEED)
df = pd.read_csv(DATASET_PATH, dtype=str)
need_cols = {"id", "reviewid", "reviewtext", "scoresentiment", "cpt"}
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["gold"] = df["scoresentiment"].apply(normalize_label)
df = df[df["gold"].notna()].copy()
if MAX_ROWS and MAX_ROWS > 0:
    df = df.head(min(MAX_ROWS, len(df))).reset_index(drop=True)

print("Rows to run:", len(df))
done = load_done(RUNS_PATH)
print("Resume: already completed keys:", len(done))

for cfg in CONFIGS:
    name = cfg_name(cfg)
    finch_enabled = bool(cfg["finch_enabled"])
    use_cpt = bool(cfg["use_cpt"])
    print("\\nRunning config:", name)

    for i, row in tqdm(df.iterrows(), total=len(df), desc=name, leave=False):
        row_id = str(i)
        key = (name, row_id)
        if key in done:
            continue

        text_source = str(row["cpt"]) if use_cpt and pd.notna(row["cpt"]) and str(row["cpt"]).strip() else str(row["reviewtext"])
        prompt = (
            "Classify the movie review sentiment. "
            "Answer with one word only: positive or negative.\\n\\n"
            f"Review: {text_source}\\n"
            "Answer:"
        )

        err = ""
        pred_raw = ""
        pred_label = ""
        t0 = time.perf_counter()
        try:
            pred_raw = run_generate_with_optional_finch(prompt, finch_enabled=finch_enabled)
            # Only parse generated tail after "Answer:"
            tail = pred_raw.split("Answer:")[-1].strip()
            pred_label = parse_sentiment(tail) or parse_sentiment(pred_raw) or ""
        except Exception as e:
            err = str(e)[:500]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        append_row(
            RUNS_PATH,
            {
                "config": name,
                "finch_enabled": finch_enabled,
                "use_cpt": use_cpt,
                "row_id": row_id,
                "reviewid": str(row["reviewid"]),
                "gold": str(row["gold"]),
                "pred_raw": pred_raw,
                "pred_label": pred_label,
                "latency_ms": latency_ms,
                "error": err,
            },
        )
        if not err:
            done.add(key)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print("Inference done. Runs file:", RUNS_PATH)
"""
        ),
        md("### Step 7: Aggregate metrics and export figures"),
        code(
            """import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

runs = pd.read_csv(RUNS_PATH)
runs["error"] = runs["error"].fillna("").astype(str).str.strip()
ok = runs[(runs["error"] == "") | (runs["error"].str.lower() == "nan")].copy()
ok = ok[ok["pred_label"].notna() & (ok["pred_label"].astype(str).str.len() > 0)]

rows = []
for cfg in sorted(ok["config"].unique()):
    part = ok[ok["config"] == cfg].copy()
    if len(part) == 0:
        continue
    y_true = part["gold"].astype(str).tolist()
    y_pred = part["pred_label"].astype(str).tolist()
    try:
        acc = float(accuracy_score(y_true, y_pred))
    except Exception:
        acc = float("nan")
    try:
        f1m = float(f1_score(y_true, y_pred, average="macro"))
    except Exception:
        f1m = float("nan")
    rows.append(
        {
            "config": cfg,
            "finch_enabled": bool(part["finch_enabled"].iloc[0]),
            "use_cpt": bool(part["use_cpt"].iloc[0]),
            "accuracy": acc,
            "f1_macro": f1m,
            "latency_ms_mean": float(part["latency_ms"].astype(float).mean()),
            "n": int(len(part)),
        }
    )

summary = pd.DataFrame(rows).sort_values(["finch_enabled", "use_cpt"]).reset_index(drop=True)
summary.to_csv(SUMMARY_PATH, index=False)
print("Wrote", SUMMARY_PATH)
display(summary)

if len(summary) > 0:
    # quality chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(summary))
    ax.bar([i - 0.18 for i in x], summary["accuracy"], width=0.36, label="accuracy")
    ax.bar([i + 0.18 for i in x], summary["f1_macro"], width=0.36, label="f1_macro")
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary["config"], rotation=20, ha="right")
    ax.set_ylabel("score")
    ax.set_title("Finch 2x2 quality")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fp = FIG_DIR / "finch_2x2_quality.png"
    fig.savefig(fp, dpi=120)
    plt.show()
    print("Saved", fp)

    # latency chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(summary["config"], summary["latency_ms_mean"])
    ax.set_ylabel("mean latency (ms/sample)")
    ax.set_title("Finch 2x2 latency")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fp = FIG_DIR / "finch_2x2_latency.png"
    fig.savefig(fp, dpi=120)
    plt.show()
    print("Saved", fp)

    # optional trade-off scatter
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(summary["latency_ms_mean"], summary["f1_macro"])
    for _, r in summary.iterrows():
        ax.annotate(r["config"], (r["latency_ms_mean"], r["f1_macro"]), fontsize=8)
    ax.set_xlabel("mean latency (ms/sample)")
    ax.set_ylabel("f1_macro")
    ax.set_title("Finch 2x2 trade-off")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fp = FIG_DIR / "finch_2x2_tradeoff.png"
    fig.savefig(fp, dpi=120)
    plt.show()
    print("Saved", fp)
"""
        ),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3", "language": "python"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }

    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

