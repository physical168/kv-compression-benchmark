"""Generate eval_artwork_llava.ipynb for Colab runs.

Uses paintings.csv as the dataset (title, inception, movement, genre, image_url).
Images are matched to local files in artworks_files/artworks_files/ via URL filename.

Run from repo root:
    python scripts/generate_eval_artwork_notebook.py
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
OUT = ROOT / "eval_artwork_llava.ipynb"


def main() -> None:
    cells = [
        md(
            """# Artwork Evaluation: ExpectedAttention · KVzip · Finch · FinchWithCPT

Vision-Language Model benchmark on the paintings dataset.

**Dataset**: `paintings.csv` (66 artworks — Renaissance & Neoclassicism)
**Model**: `llava-hf/llama3-llava-next-8b-hf`  (fallback: `Qwen/Qwen2-VL-2B-Instruct` for small GPU)
**Presses**: ExpectedAttention, KVzip, Finch, FinchWithCPT
**Ratios**: `[0.2, 0.4, 0.6, 0.8, 0.9, 0.95]`
**Queries**: `filter` (yes/no movement check) · `extract` (genre extraction) · `both` (combined)

## Ground truths
| Query type | Question | Gold label |
|---|---|---|
| filter | Is this a `{movement}` painting? | `movement` column |
| extract | What is the genre of this painting? | `genre` column |
| both | Is this a `{movement}` painting? If yes, what is its genre? | `movement` + `genre` |

## Checkpointing
Each finished `(config, ratio, query_type, row_id)` is appended with disk-flush to
`artwork_runs.csv` under `RUN_DIR`. Re-run after reconnect — already-done keys are skipped.
"""
        ),

        # ── Step 1 ──────────────────────────────────────────────────────────
        md("### Step 1 — Install dependencies"),
        code("!pip install -q transformers accelerate bitsandbytes pandas scikit-learn matplotlib tqdm kvpress pillow"),

        # ── Step 2 ──────────────────────────────────────────────────────────
        md("### Step 2 — Mount Google Drive & set paths"),
        code(
            """\
from pathlib import Path
import os

RUN_ON_COLAB = os.path.isdir("/content")
USE_GOOGLE_DRIVE = True
DRIVE_SUBDIR = "kv-compression-benchmark/artwork_eval_runs"

if RUN_ON_COLAB and USE_GOOGLE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    except ImportError:
        USE_GOOGLE_DRIVE = False
        print("google.colab not found; falling back to local paths.")

if RUN_ON_COLAB and USE_GOOGLE_DRIVE and Path("/content/drive/MyDrive").is_dir():
    RUN_DIR  = Path("/content/drive/MyDrive") / DRIVE_SUBDIR
    REPO_DIR = Path("/content/drive/MyDrive/kv-compression-benchmark")
elif RUN_ON_COLAB:
    RUN_DIR  = Path("/content/artwork_eval_workspace")
    REPO_DIR = Path("/content")
else:
    RUN_DIR  = Path("artwork_eval_output")
    REPO_DIR = Path(".")

RUN_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = RUN_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# paintings.csv lives at the repo root
DATASET_PATH = REPO_DIR / "paintings.csv"
# Local images (URL filename-decoded) live here
IMAGES_DIR   = REPO_DIR / "artworks_files" / "artworks_files"

print("RUN_DIR     :", RUN_DIR.resolve())
print("DATASET_PATH:", DATASET_PATH.resolve())
print("IMAGES_DIR  :", IMAGES_DIR.resolve())
"""
        ),

        # ── Step 3 ──────────────────────────────────────────────────────────
        md("### Step 3 — Load model"),
        code(
            """\
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

# Primary: 8B model (fits on A100 or T4 with 8-bit quant)
MODEL_ID = "llava-hf/llama3-llava-next-8b-hf"
LOAD_IN_8BIT = True   # set False if running on A100 / plenty of VRAM

dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)

quantization_config = BitsAndBytesConfig(load_in_8bit=True) if LOAD_IN_8BIT else None

print(f"Loading {MODEL_ID} (quantization_config={quantization_config}) ...")
try:
    processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        quantization_config=quantization_config,
    )
except Exception as e:
    print(f"LlavaNext load failed ({e}); trying AutoModelForVision2Seq ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        quantization_config=quantization_config,
    )

print("Model ready:", MODEL_ID)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
"""
        ),

        # ── Step 4 ──────────────────────────────────────────────────────────
        md("### Step 4 — Configuration"),
        code(
            """\
from __future__ import annotations
import csv, gc, os, re, time
from urllib.parse import unquote
import pandas as pd
from PIL import Image

# ── Hyper-parameters ──────────────────────────────────────────────────────
MAX_ROWS             = 0       # 0 = all rows in paintings.csv
MAX_NEW_TOKENS       = 40
RESUME_FROM_CHECKPOINT = True

# Text-style ratios + aggressive image ratios (0.9 / 0.95)
COMPRESSION_RATIOS   = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]

# 4 presses
CONFIGS = [
    {"method": "ea",    "use_cpt": False},   # ExpectedAttention
    {"method": "kvzip", "use_cpt": False},   # KVzip
    {"method": "finch", "use_cpt": False},   # Finch
    {"method": "finch", "use_cpt": True},    # FinchWithCPT
]

# 3 query types
QUERY_TYPES = ["filter", "extract", "both"]

# ── Paths ─────────────────────────────────────────────────────────────────
RUNS_PATH    = RUN_DIR / "artwork_runs.csv"
SUMMARY_PATH = RUN_DIR / "artwork_summary.csv"

# ── Checkpoint fields ─────────────────────────────────────────────────────
CK_FIELDS = [
    "config", "method", "use_cpt", "compression_ratio", "query_type",
    "row_id", "image_file",
    "gold", "pred_raw", "pred_label",
    "latency_ms", "error",
]

# ── Helpers ───────────────────────────────────────────────────────────────
def cfg_name(cfg: dict) -> str:
    return f"{cfg['method']}_cpt{int(cfg['use_cpt'])}"

def url_to_local_filename(url: str) -> str:
    # Extract percent-encoded filename from Wikimedia URL and decode it.
    # URL looks like: .../Special:FilePath/Some%20File.jpg
    fname = url.split("/")[-1]
    return unquote(fname)

def load_done(path: Path) -> set:
    if not path.is_file() or not RESUME_FROM_CHECKPOINT:
        return set()
    df = pd.read_csv(path)
    done = set()
    for _, r in df.iterrows():
        err = str(r.get("error", "")).strip().lower()
        if err in ("", "nan"):
            done.add((
                str(r["config"]),
                str(r["compression_ratio"]),
                str(r["query_type"]),
                str(r["row_id"]),
            ))
    return done

def append_row(path: Path, row: dict) -> None:
    newfile = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CK_FIELDS, extrasaction="ignore")
        if newfile:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CK_FIELDS})
        f.flush()
        os.fsync(f.fileno())

print("Config ready.")
print(f"  Presses           : {[cfg_name(c) for c in CONFIGS]}")
print(f"  Compression ratios: {COMPRESSION_RATIOS}")
print(f"  Query types       : {QUERY_TYPES}")
print(f"  Runs path         : {RUNS_PATH}")
"""
        ),

        # ── Step 5 ──────────────────────────────────────────────────────────
        md("### Step 5 — Load dataset & build queries"),
        code(
            """\
df = pd.read_csv(DATASET_PATH)
# Drop rows without image_url
df = df[df["image_url"].notna()].copy().reset_index(drop=True)

if MAX_ROWS and MAX_ROWS > 0:
    df = df.head(min(MAX_ROWS, len(df))).reset_index(drop=True)

# Resolve local image path from URL
df["image_file"] = df["image_url"].apply(url_to_local_filename)
df["image_path"]  = df["image_file"].apply(lambda f: str(IMAGES_DIR / f))

# CPT: use title + movement + genre as context text for FinchWithCPT
df["cpt"] = df.apply(
    lambda r: f"Context: This is a painting titled '{r['title']}', "
              f"created in the {r['movement']} movement.",
    axis=1,
)

# Gold labels
df["gold_movement"] = df["movement"].str.strip().str.lower()
df["gold_genre"]    = df["genre"].str.strip().str.lower()

# Build per-query-type prompts
def build_prompt(row: pd.Series, query_type: str, use_cpt: bool) -> str:
    cpt_prefix = (row["cpt"] + "\\n") if use_cpt else ""
    mv = row["movement"].strip()
    if query_type == "filter":
        return (
            f"{cpt_prefix}"
            f"Is this painting from the {mv} movement? "
            "Answer with one word only: yes or no."
        )
    elif query_type == "extract":
        return (
            f"{cpt_prefix}"
            "What is the genre of this painting? "
            "Choose from: portrait, religious art, history painting, "
            "mythological painting, nude, genre art. "
            "Answer with the genre only."
        )
    else:  # both
        return (
            f"{cpt_prefix}"
            f"Is this painting from the {mv} movement? "
            "If yes, also state its genre (one of: portrait, religious art, "
            "history painting, mythological painting, nude, genre art). "
            "Answer concisely."
        )

def build_gold(row: pd.Series, query_type: str) -> str:
    if query_type == "filter":
        return "yes"   # all rows in CSV already match their stated movement
    elif query_type == "extract":
        return row["gold_genre"]
    else:
        return f"yes, {row['gold_genre']}"

print(f"Loaded {len(df)} rows from {DATASET_PATH.name}")
print(df[["title", "movement", "genre", "image_file"]].head(5).to_string())

# Verify images exist locally
missing = df[~df["image_path"].apply(os.path.isfile)]
if len(missing):
    print(f"\\nWARNING: {len(missing)} image(s) not found locally:")
    for _, r in missing.iterrows():
        print("  ", r["image_file"])
else:
    print("\\nAll images found locally.")
"""
        ),

        # ── Step 6 ──────────────────────────────────────────────────────────
        md("### Step 6 — Inference adapters"),
        code(
            """\
import torch
from kvpress import ExpectedAttentionPress, KVzipPress, FinchPress

_PRESS_MAP = {
    "ea":    ExpectedAttentionPress,
    "kvzip": KVzipPress,
    "finch": FinchPress,
}

def run_generate_vision(
    image_path: str,
    prompt: str,
    method: str,
    compression_ratio: float,
) -> str:
    image = Image.open(image_path).convert("RGB")

    # Build chat template (LlavaNext style)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    formatted_prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True
    )
    inputs = processor(
        images=image, text=formatted_prompt, return_tensors="pt"
    ).to(model.device)

    gen_kwargs = {"max_new_tokens": MAX_NEW_TOKENS}
    tok_id = getattr(processor, "tokenizer", processor)
    if getattr(tok_id, "pad_token_id", None) is not None:
        gen_kwargs["pad_token_id"] = tok_id.pad_token_id

    PressCls = _PRESS_MAP.get(method)
    with torch.no_grad():
        if PressCls is not None:
            with PressCls(compression_ratio=compression_ratio)(model):
                out = model.generate(**inputs, **gen_kwargs)
        else:
            out = model.generate(**inputs, **gen_kwargs)

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    generated = out[0][input_len:]
    return tok_id.decode(generated, skip_special_tokens=True).strip()

print("Adapters ready.")
"""
        ),

        # ── Step 7 ──────────────────────────────────────────────────────────
        md(
            """\
### Step 7 — Inference loop (checkpointed)

For each `(config × ratio × query_type × row)`:
1. Loads image from local `artworks_files/artworks_files/`.
2. Runs generation through the press.
3. Appends result to `artwork_runs.csv` with an `os.fsync` flush.

**Resume**: set `RESUME_FROM_CHECKPOINT = True` (default), re-run from Step 4 onward.
"""
        ),
        code(
            """\
from tqdm.auto import tqdm

done = load_done(RUNS_PATH)
print(f"Checkpoint: {len(done)} already-completed (config, ratio, qtype, row) keys.")

total = len(CONFIGS) * len(COMPRESSION_RATIOS) * len(QUERY_TYPES) * len(df)
print(f"Total cells to run: {total}  (skipping done)")

for cfg in CONFIGS:
    name    = cfg_name(cfg)
    method  = cfg["method"]
    use_cpt = bool(cfg["use_cpt"])

    for ratio in COMPRESSION_RATIOS:
        for qtype in QUERY_TYPES:
            run_label = f"{name}_r{ratio}_{qtype}"

            for i, row in tqdm(df.iterrows(), total=len(df), desc=run_label, leave=False):
                key = (name, str(ratio), qtype, str(i))
                if key in done:
                    continue

                img_path = row["image_path"]
                if not os.path.isfile(img_path):
                    append_row(RUNS_PATH, {
                        "config": name, "method": method, "use_cpt": use_cpt,
                        "compression_ratio": ratio, "query_type": qtype,
                        "row_id": str(i), "image_file": row["image_file"],
                        "gold": build_gold(row, qtype),
                        "pred_raw": "", "pred_label": "",
                        "latency_ms": 0, "error": "image_not_found",
                    })
                    continue

                prompt = build_prompt(row, qtype, use_cpt)
                gold   = build_gold(row, qtype)

                err = ""
                pred_raw = ""
                t0 = time.perf_counter()
                try:
                    pred_raw = run_generate_vision(img_path, prompt, method, float(ratio))
                except Exception as e:
                    err = str(e)[:500]
                latency_ms = (time.perf_counter() - t0) * 1000.0

                # Simple label extraction (lower-cased first word or phrase)
                pred_label = pred_raw.strip().lower().split("\\n")[0][:80]

                append_row(RUNS_PATH, {
                    "config": name, "method": method, "use_cpt": use_cpt,
                    "compression_ratio": ratio, "query_type": qtype,
                    "row_id": str(i), "image_file": row["image_file"],
                    "gold": gold,
                    "pred_raw": pred_raw, "pred_label": pred_label,
                    "latency_ms": latency_ms, "error": err,
                })
                if not err:
                    done.add(key)

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

print(f"\\nInference complete. Results at: {RUNS_PATH}")
"""
        ),

        # ── Step 8 ──────────────────────────────────────────────────────────
        md("### Step 8 — Aggregate metrics & plot"),
        code(
            """\
import matplotlib.pyplot as plt
import numpy as np

runs = pd.read_csv(RUNS_PATH)
runs["error"] = runs["error"].fillna("").astype(str).str.strip()
ok = runs[(runs["error"] == "") | (runs["error"].str.lower() == "nan")].copy()

# ── Accuracy: exact-match on lowercase ────────────────────────────────────
def soft_match(pred: str, gold: str) -> bool:
    p = str(pred).strip().lower()
    g = str(gold).strip().lower()
    return g in p or p in g or p == g

rows = []
for cfg in ok["config"].unique():
    for r in ok["compression_ratio"].unique():
        for qt in ok["query_type"].unique():
            part = ok[
                (ok["config"] == cfg) &
                (ok["compression_ratio"] == r) &
                (ok["query_type"] == qt)
            ].copy()
            if len(part) == 0:
                continue
            acc = float(np.mean([
                soft_match(p, g)
                for p, g in zip(part["pred_label"], part["gold"])
            ]))
            rows.append({
                "config": cfg,
                "method": part["method"].iloc[0],
                "use_cpt": part["use_cpt"].iloc[0],
                "compression_ratio": float(r),
                "query_type": qt,
                "accuracy": acc,
                "latency_ms_mean": float(part["latency_ms"].astype(float).mean()),
                "n": len(part),
            })

summary = (
    pd.DataFrame(rows)
    .sort_values(["query_type", "method", "use_cpt", "compression_ratio"])
    .reset_index(drop=True)
)
summary.to_csv(SUMMARY_PATH, index=False)
print("Wrote", SUMMARY_PATH)
display(summary)

# ── Plot: accuracy vs ratio, one subplot per query type ───────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, qt in zip(axes, ["filter", "extract", "both"]):
    qt_data = summary[summary["query_type"] == qt]
    for cfg in qt_data["config"].unique():
        d = qt_data[qt_data["config"] == cfg].sort_values("compression_ratio")
        ax.plot(d["compression_ratio"], d["accuracy"], marker="o", label=cfg)
    ax.set_title(f"Query: {qt}")
    ax.set_xlabel("Compression Ratio")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
fig.suptitle("Artwork VLM Accuracy vs Compression Ratio", fontsize=13)
plt.tight_layout()
fp = FIG_DIR / "artwork_accuracy.png"
fig.savefig(fp, dpi=120)
plt.show()
print("Saved", fp)

# ── Plot: latency vs ratio, one subplot per query type ────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, qt in zip(axes, ["filter", "extract", "both"]):
    qt_data = summary[summary["query_type"] == qt]
    for cfg in qt_data["config"].unique():
        d = qt_data[qt_data["config"] == cfg].sort_values("compression_ratio")
        ax.plot(d["compression_ratio"], d["latency_ms_mean"], marker="o", label=cfg)
    ax.set_title(f"Query: {qt}")
    ax.set_xlabel("Compression Ratio")
    ax.set_ylabel("Mean Latency (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
fig.suptitle("Artwork VLM Latency vs Compression Ratio", fontsize=13)
plt.tight_layout()
fp = FIG_DIR / "artwork_latency.png"
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
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3",
                "language": "python",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }

    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
