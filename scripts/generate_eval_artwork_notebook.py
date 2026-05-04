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
        md(
            """### Step 1 — Install dependencies（本格一般只跑一次）

Colab 里旧的 Pillow C 扩展 `_imaging.so` 会留在**当前 Python 进程**里，光 `pip` 重装不够，需要**重启内核**后才会加载新编译的扩展。

- 本格末尾会**请求重启内核**（不是机器坏了）。Colab 有时用「会话崩溃 / 不明原因」提示，**可以忽略**；重启完成后请从 **Step 2** 继续。
- **不要反复运行 Step 1**：否则会一次次重启。若已装过，本格会自动跳过。
- 若要强制重装：在 Colab 里删掉文件 `/content/.eval_artwork_llava_step1_done` 后再运行本格。
"""
        ),
        code(
            """# One-time install + optional kernel restart (see Step 1 markdown above).
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_MARK = Path("/content/.eval_artwork_llava_step1_done") if os.path.isdir("/content") else Path(
    ".eval_artwork_llava_step1_done"
)


def _pil_import_ok() -> bool:
    try:
        from PIL import Image  # noqa: F401

        return True
    except Exception:
        return False


if _MARK.is_file() and _pil_import_ok():
    print("Step 1 已做过：标记存在且 Pillow 可导入。请直接从 Step 2 继续。")
    print("若要强制重装依赖，请先删除:", _MARK.resolve())
else:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-U",
            "transformers>=5.0",
            "accelerate",
            "bitsandbytes",
            "pandas",
            "scikit-learn",
            "matplotlib",
            "tqdm",
            "kvpress",
        ]
    )
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "Pillow", "pillow"])
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-binary",
            "Pillow",
            "Pillow==10.4.0",
        ]
    )
    _MARK.write_text("ok", encoding="utf-8")
    print("依赖与 Pillow(源码编译)已装好。正在请求**重启内核**…")
    print("（Colab 若弹出「会话崩溃」多为误报；重启后请从 Step 2 继续，勿重复跑本格。）")
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None and getattr(ip.kernel, "do_shutdown", None) is not None:
            ip.kernel.do_shutdown(restart=True)
        else:
            os._exit(0)
    except Exception:
        os.kill(os.getpid(), 9)"""
        ),

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
        md(
            """### Step 3 — Load model

Straight `LlavaNext*` load only (no `AutoModel` fallback). 默认已开 `LOAD_IN_8BIT` 以降低 Colab OOM 概率。

> **Prerequisite**: Step 1 has run once (kernel restarted or Step 1 printed “已做过”). Then run **Step 2** before this cell.
> **If the runtime restarts after this cell** (OOM / Colab disconnect): next time run **Step 2 → Step 3 → Step 4** in order — `RUN_DIR` and `model` are gone after restart.

若本格跑完后**会话重启**：下次必须从 **Step 2** 开始（再 Step 3、Step 4），否则会出现 `RUN_DIR` / `model` 未定义。

> **已经打印 `Model ready` 却仍立刻崩溃？** 多半是 **显存/内存峰值（OOM）** 或 Colab 断连。下次从 Step 2 重跑。若 8bit 仍 OOM，可把 **`LOAD_IN_8BIT = False`**（更吃显存）或换更大 GPU 运行时。
"""
        ),
        code(
            """\
import torch
import transformers
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig

print("transformers:", transformers.__version__)
print("imported from:", transformers.__file__)

MODEL_ID = "llava-hf/llama3-llava-next-8b-hf"
# Colab: True 省显存；若 device_map 把部分层放 CPU，必须开 llm_int8_enable_fp32_cpu_offload
LOAD_IN_8BIT = True

dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)

quantization_config = (
    BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    if LOAD_IN_8BIT
    else None
)

print(f"Loading {MODEL_ID} ...")
processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map="auto",
    quantization_config=quantization_config,
)
print("Model ready:", MODEL_ID)
"""
        ),

        # ── Step 4 ──────────────────────────────────────────────────────────
        md(
            """### Step 4 — Configuration

**Any kernel restart** clears all variables. `RUN_DIR` / `DATASET_PATH` / `IMAGES_DIR` are created in **Step 2**; the model lives in **Step 3**. After a crash or manual restart, run **Step 2 → Step 3 → Step 4** in order (do not start from Step 4 alone).

内核一旦重启，内存里的变量会清空。`RUN_DIR` 等在 **Step 2** 定义，模型在 **Step 3** 加载。会话崩溃或手动重启后，请按 **Step 2 → Step 3 → Step 4** 顺序执行，不要单独从 Step 4 开始。
"""
        ),
        code(
            """\
from __future__ import annotations
import csv, gc, os, re, time
from urllib.parse import unquote
import pandas as pd
from PIL import Image

if "RUN_DIR" not in globals():
    raise RuntimeError(
        "RUN_DIR 未定义：当前内核里还没跑过 Step 2，或刚发生过重启导致变量清空。"
        "请先运行 Step 2，再运行 Step 3（如需模型），最后运行本格。"
    )

MAX_ROWS             = 2
MAX_NEW_TOKENS       = 40
RESUME_FROM_CHECKPOINT = True

COMPRESSION_RATIOS   = [0.4, 0.8, 0.95]
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
        md(
            """### Step 5 — Load dataset & build dynamic queries

若图片在 Drive 上**文件名仍带 `%20`**（上传 zip 或未解码重命名），而 `image_url` 里是编码形式，仅用 `unquote` 后的路径会找不到文件。下面 `image_path` 会**先试解码名、再试 URL 尾段的原始字面名**。
"""
        ),
        code(
            """\
df = pd.read_csv(DATASET_PATH)
df = df[df["image_url"].notna()].copy().reset_index(drop=True)
if MAX_ROWS > 0: df = df.head(MAX_ROWS)


def resolve_artwork_image_path(url: str) -> str:
    \"\"\"Prefer human-readable filename; fall back to %-encoded literal on disk (common on Drive).\"\"\"
    tail = url.split("/")[-1].split("?")[0]
    p_decoded = IMAGES_DIR / unquote(tail)
    p_raw = IMAGES_DIR / tail
    if p_decoded.is_file():
        return str(p_decoded)
    if p_raw.is_file():
        return str(p_raw)
    return str(p_decoded)


df["image_file"] = df["image_url"].apply(url_to_local_filename)
df["image_path"] = df["image_url"].apply(resolve_artwork_image_path)
df["cpt"] = df.apply(lambda r: f"Context: This is a painting titled '{r['title']}'.", axis=1)

# Dynamic Query Templates
def get_prompts_for_row(row: pd.Series, qtype: str, use_cpt: bool) -> list[tuple[str, str]]:
    \"\"\"Returns a list of (prompt, gold_answer) tuples for a row.\"\"\"
    cpt_prefix = (row["cpt"] + "\\n") if use_cpt else ""
    mv = row["movement"].strip()
    gn = row["genre"].strip().lower()
    
    if qtype == "filter":
        # We ask if it IS the correct movement. Gold is always 'yes'.
        return [
            (f"{cpt_prefix}Is this a {mv} painting? Answer yes or no.", "yes"),
            (f"{cpt_prefix}Does this artwork belong to the {mv} movement? Answer yes or no.", "yes"),
        ]
    else: # extract
        # We ask for the genre. Gold is the genre value.
        choices = "Choices: portrait, religious art, history painting, mythological painting, nude, genre art."
        return [
            (f"{cpt_prefix}What is the genre of this painting? {choices} Answer with the genre name only.", gn),
            (f"{cpt_prefix}Identify the genre of this artwork. {choices}", gn),
        ]

print(f"Loaded {len(df)} rows. Accuracy will be measured against 'movement' and 'genre' columns.")
"""
        ),

        # ── Step 6 ──────────────────────────────────────────────────────────
        md("### Step 6 — Inference adapters"),
        code(
            """\
from kvpress import ExpectedAttentionPress, KVzipPress, FinchPress

_PRESS_MAP = {"ea": ExpectedAttentionPress, "kvzip": KVzipPress, "finch": FinchPress}

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
        md(
            """### Step 7 — Inference loop

结果写入 **Step 4** 的 `RUNS_PATH`（例如 Drive 下 `artwork_eval_runs/artwork_runs.csv`）。若本格跑完仍没有 CSV，多半是 **`image_path` 在 Colab 上不存在**：请把图片放在 `REPO_DIR/datasets/artwork/images/`（与 `paintings.csv` 里 `image_url` 的文件名一致），或把 `REPO_DIR` 指到含该目录的仓库根。

本格结束会打印统计：`written` / `skip_missing_image` / `skip_done`。
"""
        ),
        code(
            """\
from tqdm.auto import tqdm

done = load_done(RUNS_PATH)
n_written = 0
n_skip_done = 0
n_skip_missing = 0

n_rows = len(df)
n_images_ok = sum(1 for _, r in df.iterrows() if os.path.isfile(r["image_path"])) if n_rows else 0
print("Step 7 — df rows:", n_rows, "| image_path exists:", n_images_ok)
print("IMAGES_DIR:", IMAGES_DIR)
if n_rows:
    _p0 = df.iloc[0]["image_path"]
    print("First image_path:", _p0, "| exists:", os.path.isfile(_p0))

for cfg in CONFIGS:
    name, method, use_cpt = cfg_name(cfg), cfg["method"], bool(cfg["use_cpt"])
    for ratio in COMPRESSION_RATIOS:
        for qtype in QUERY_TYPES:
            for i, row in tqdm(df.iterrows(), total=len(df), desc=f"{name}_r{ratio}_{qtype}", leave=False):
                queries = get_prompts_for_row(row, qtype, use_cpt)

                for q_idx, (prompt, gold) in enumerate(queries):
                    row_key = f"{i}_q{q_idx}"
                    key = (name, str(ratio), qtype, row_key)
                    if key in done:
                        n_skip_done += 1
                        continue

                    if not os.path.isfile(row["image_path"]):
                        n_skip_missing += 1
                        continue

                    err, pred_raw = "", ""
                    t0 = time.perf_counter()
                    try:
                        pred_raw = run_generate_vision(row["image_path"], prompt, method, float(ratio))
                    except Exception as e:
                        err = str(e)[:500]
                    latency_ms = (time.perf_counter() - t0) * 1000.0

                    append_row(
                        RUNS_PATH,
                        {
                            "config": name,
                            "method": method,
                            "use_cpt": use_cpt,
                            "compression_ratio": ratio,
                            "query_type": qtype,
                            "row_id": row_key,
                            "image_file": row["image_file"],
                            "gold": gold,
                            "pred_raw": pred_raw,
                            "pred_label": pred_raw.strip().lower()[:80],
                            "latency_ms": latency_ms,
                            "error": err,
                        },
                    )
                    n_written += 1
                    if not err:
                        done.add(key)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

print(
    "Step 7 summary — written:", n_written,
    "| skip_done:", n_skip_done,
    "| skip_missing_image:", n_skip_missing,
)
if n_written == 0:
    print("未写入任何行：请确认图片已同步到 IMAGES_DIR，且路径与 paintings.csv 中文件名一致。")
    print("RUNS_PATH:", RUNS_PATH)
"""
        ),

        # ── Step 8 ──────────────────────────────────────────────────────────
        md(
            """### Step 8 — Summary

需要先有 **Step 7** 写出的 `artwork_runs.csv`（路径即 Step 4 里的 `RUNS_PATH`）。若文件不存在，本格会提示而不是报错。

If `artwork_runs.csv` is missing, run **Step 7** first.
"""
        ),
        code(
            """\
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

if "RUNS_PATH" not in globals():
    raise RuntimeError("RUNS_PATH 未定义：请先运行 Step 2 与 Step 4。")

if not Path(RUNS_PATH).is_file():
    print("未找到结果文件:", RUNS_PATH)
    print("请先运行 Step 7（推理循环）生成 CSV，再运行本格。")
else:
    runs = pd.read_csv(RUNS_PATH)
    runs["error"] = runs["error"].fillna("").astype(str)
    ok = runs[runs["error"] == ""].copy()

    def is_correct(pred, gold):
        p, g = str(pred).strip().lower(), str(gold).strip().lower()
        return g in p or p in g

    if ok.empty:
        print("没有 error 为空的行；请检查 Step 7 输出或 runs 中的 error 列。")
        display(runs.head(10))
    else:
        ok["correct"] = ok.apply(lambda r: is_correct(r["pred_label"], r["gold"]), axis=1)
        summary = ok.groupby(["query_type", "config", "compression_ratio"]).agg(
            accuracy=("correct", "mean"),
            latency_ms=("latency_ms", "mean"),
        ).reset_index()
        display(summary)

        for qt in summary["query_type"].unique():
            subset = summary[summary["query_type"] == qt]
            plt.figure(figsize=(10, 5))
            for cfg in subset["config"].unique():
                d = subset[subset["config"] == cfg].sort_values("compression_ratio")
                plt.plot(d["compression_ratio"], d["accuracy"], marker="o", label=cfg)
            plt.title(f"Accuracy vs Ratio ({qt})")
            plt.xlabel("Compression Ratio")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)
            plt.show()
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
