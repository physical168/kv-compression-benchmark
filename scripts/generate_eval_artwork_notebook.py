"""Generate eval_artwork_llava.ipynb for Colab runs.

Uses paintings.csv as the dataset.
Hardcodes queries from image_queries.yaml.

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

**Dataset**: `datasets/artwork/paintings.csv`
**Images**: `datasets/artwork/images/`
**Presses**: ExpectedAttention, KVzip, Finch, FinchWithCPT
**Ratios**: `[0.2, 0.4, 0.6, 0.8, 0.9, 0.95]`
**Queries**: 10 Filters + 10 Extracts from `image_queries.yaml`
"""
        ),

        # ── Step 1 ──────────────────────────────────────────────────────────
        md("### Step 1 — Install dependencies"),
        code("!pip install -q -U git+https://github.com/huggingface/transformers.git accelerate bitsandbytes pandas scikit-learn matplotlib tqdm kvpress pillow"),

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

# Dataset and images paths
DATASET_PATH = REPO_DIR / "datasets" / "artwork" / "paintings.csv"
IMAGES_DIR   = REPO_DIR / "datasets" / "artwork" / "images"

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
from transformers import (
    LlavaNextProcessor, 
    LlavaNextForConditionalGeneration,
    AutoProcessor, 
    AutoModel, 
    BitsAndBytesConfig
)

MODEL_ID = "llava-hf/llama3-llava-next-8b-hf"
LOAD_IN_8BIT = True

dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)

quantization_config = BitsAndBytesConfig(load_in_8bit=True) if LOAD_IN_8BIT else None

print(f"Loading {MODEL_ID} ...")
try:
    processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        quantization_config=quantization_config,
    )
except Exception as e:
    print(f"LlavaNext load failed ({e}); trying generic AutoModel ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True
    )

print("Model ready:", MODEL_ID)
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

MAX_ROWS             = 2
MAX_NEW_TOKENS       = 40
RESUME_FROM_CHECKPOINT = True

COMPRESSION_RATIOS   = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
CONFIGS = [
    {"method": "ea",    "use_cpt": False},
    {"method": "kvzip", "use_cpt": False},
    {"method": "finch", "use_cpt": False},
    {"method": "finch", "use_cpt": True},
]
QUERY_TYPES = ["filter", "extract"]

RUNS_PATH    = RUN_DIR / "artwork_runs.csv"
SUMMARY_PATH = RUN_DIR / "artwork_summary.csv"

CK_FIELDS = [
    "config", "method", "use_cpt", "compression_ratio", "query_type",
    "row_id", "image_file", "gold", "pred_raw", "pred_label",
    "latency_ms", "error",
]

def cfg_name(cfg: dict) -> str:
    return f"{cfg['method']}_cpt{int(cfg['use_cpt'])}"

def url_to_local_filename(url: str) -> str:
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
            # Use (config, ratio, query_type, row_id) as unique key
            done.add((str(r["config"]), str(r["compression_ratio"]), str(r["query_type"]), str(r["row_id"])))
    return done

def append_row(path: Path, row: dict) -> None:
    newfile = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CK_FIELDS, extrasaction="ignore")
        if newfile: w.writeheader()
        w.writerow({k: row.get(k, "") for k in CK_FIELDS})
        f.flush()
        os.fsync(f.fileno())
"""
        ),

        # ── Step 5 ──────────────────────────────────────────────────────────
        md("### Step 5 — Load dataset & queries"),
        code(
            """\
FILTER_QUERIES = [
    "Does this painting depict Madonna and Child?",
    "Does this painting depict more than two people?",
    "Does this painting depict more than three people?",
    "Does this painting depict saints identifiable by their halos?",
    "Does this painting depict a scene in which death is a dominant theme?",
    "Does this painting depict a religious scene?",
    "Does this painting show a still life?",
    "Does this painting depict a scene of war?",
    "Does this painting depict an angel with wings?",
    "Does this painting depict a crucifixion scene?"
]

EXTRACT_QUERIES = [
    "Extract the number of people depicted in this painting.",
    "Extract the primary background color of this painting. Choose from Red, Blue, Yellow, Green, Orange, Purple, Black, White.",
    "Extract the number of saints with halos from this painting.",
    "Extract the number of animals from this painting.",
    "Extract the gender of the main character from this painting. (male / female / undefined)",
    "Extract the type of setting of this painting. (interior / exterior / undefined)",
    "Extract the dominant material depicted in this painting. (stone / wood / metal / fabric / undefined)",
    "Extract the approximate lighting type in this painting. (natural / candle / undefined)",
    "Extract the level of movement in this painting. (static / moderate / dynamic)",
    "Extract the landscape type depicted in this painting. (mountain / forest / sea / plain / undefined)"
]

df = pd.read_csv(DATASET_PATH)
df = df[df["image_url"].notna()].copy().reset_index(drop=True)
if MAX_ROWS > 0: df = df.head(MAX_ROWS)

df["image_file"] = df["image_url"].apply(url_to_local_filename)
df["image_path"]  = df["image_file"].apply(lambda f: str(IMAGES_DIR / f))
df["cpt"] = df.apply(lambda r: f"Context: This is a painting titled '{r['title']}', created in the {r['movement']} movement.", axis=1)

print(f"Loaded {len(df)} rows.")
"""
        ),

        # ── Step 6 ──────────────────────────────────────────────────────────
        md("### Step 6 — Inference adapters"),
        code(
            """\
from kvpress import ExpectedAttentionPress, KVzipPress, FinchPress

_PRESS_MAP = {"ea": ExpectedAttentionAttentionPress if 'ExpectedAttentionAttentionPress' in globals() else ExpectedAttentionPress, "kvzip": KVzipPress, "finch": FinchPress}

def run_generate_vision(image_path: str, prompt: str, method: str, compression_ratio: float) -> str:
    image = Image.open(image_path).convert("RGB")
    conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    formatted_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=formatted_prompt, return_tensors="pt").to(model.device)
    
    gen_kwargs = {"max_new_tokens": MAX_NEW_TOKENS}
    tok = getattr(processor, "tokenizer", processor)
    if hasattr(tok, "pad_token_id") and tok.pad_token_id is not None:
        gen_kwargs["pad_token_id"] = tok.pad_token_id

    PressCls = _PRESS_MAP.get(method)
    with torch.no_grad():
        if PressCls:
            with PressCls(compression_ratio=compression_ratio)(model):
                out = model.generate(**inputs, **gen_kwargs)
        else:
            out = model.generate(**inputs, **gen_kwargs)
    
    input_len = inputs["input_ids"].shape[1]
    return tok.decode(out[0][input_len:], skip_special_tokens=True).strip()
"""
        ),

        # ── Step 7 ──────────────────────────────────────────────────────────
        md("### Step 7 — Inference loop"),
        code(
            """\
from tqdm.auto import tqdm

done = load_done(RUNS_PATH)

for cfg in CONFIGS:
    name, method, use_cpt = cfg_name(cfg), cfg["method"], bool(cfg["use_cpt"])
    for ratio in COMPRESSION_RATIOS:
        for qtype in QUERY_TYPES:
            queries = FILTER_QUERIES if qtype == "filter" else EXTRACT_QUERIES
            for q_idx, base_query in enumerate(queries):
                for i, row in tqdm(df.iterrows(), total=len(df), desc=f"{name}_r{ratio}_{qtype}_q{q_idx}", leave=False):
                    row_key = f"{i}_q{q_idx}"
                    key = (name, str(ratio), qtype, row_key)
                    if key in done: continue
                    
                    if not os.path.isfile(row["image_path"]): continue
                    
                    prompt = f"{row['cpt'] + '\\n' if use_cpt else ''}Question: {base_query}\\nAnswer:"
                    err, pred_raw = "", ""
                    t0 = time.perf_counter()
                    try:
                        pred_raw = run_generate_vision(row["image_path"], prompt, method, float(ratio))
                    except Exception as e:
                        err = str(e)[:500]
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    
                    append_row(RUNS_PATH, {
                        "config": name, "method": method, "use_cpt": use_cpt,
                        "compression_ratio": ratio, "query_type": qtype,
                        "row_id": row_key, "image_file": row["image_file"],
                        "gold": "N/A", "pred_raw": pred_raw, "pred_label": pred_raw.strip().lower()[:80],
                        "latency_ms": latency_ms, "error": err
                    })
                    if not err: done.add(key)
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
"""
        ),

        # ── Step 8 ──────────────────────────────────────────────────────────
        md("### Step 8 — Summary"),
        code(
            """\
import matplotlib.pyplot as plt
runs = pd.read_csv(RUNS_PATH)
summary = runs.groupby(["query_type", "config", "compression_ratio"])["latency_ms"].mean().reset_index()
display(summary)
"""
        )
    ]

    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}},
        "cells": cells,
    }
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
