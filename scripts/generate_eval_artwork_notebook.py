"""Generate eval_artwork_llava.ipynb aligned with CompressionExperiments artwork flow.

- Queries: benchmarks/artwork_eval/configs/image_queries.yaml (same strings as CE).
- Prompts: same answer_prefix pattern as experiment_manager/src/engine.py (image filter 1/0,
  image extract + suffix).
- Outputs: results/{dataset}/{model_tag}/{press}/{ratio}/results.csv with columns
  record_id, query, press, ratio, answer.
- Step 8: EvaluationManager + P/R/F1 (evaluation_config.yaml + ground_truth/).

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
            """# Artwork Evaluation (Llava) — aligned with CompressionExperiments

Vision-language benchmark on **paintings** using the same **queries** as
`CompressionExperiments/experiment_manager/configs/image_queries.yaml`, the same **user prompt**
shape as `engine.py` (`run_single`, image modality), and the same **results layout** as
`src/utils.py` / `evaluate.py`:

- **Results**: `RUN_DIR/results/artwork/{model_tag}/{PressClass}/{ratio:.2f}/results.csv`
- **Columns**: `record_id`, `query`, `press`, `ratio`, `answer`
- **Metrics (Step 8)**: `benchmarks/artwork_eval/evaluation/EvaluationManager` → P/R/F1 vs
  `benchmarks/artwork_eval/ground_truth/query_*.csv`

**Dataset**: `datasets/artwork/paintings.csv` · **Images**: `datasets/artwork/images/`  
**Config copy**: `benchmarks/artwork_eval/configs/image_queries.yaml`
"""
        ),
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
            "pyyaml",
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

ARTWORK_EVAL_ROOT = REPO_DIR / "benchmarks" / "artwork_eval"
IMAGE_QUERIES_YAML = ARTWORK_EVAL_ROOT / "configs" / "image_queries.yaml"
EVAL_CONFIG_YAML = ARTWORK_EVAL_ROOT / "evaluation" / "evaluation_config.yaml"

DATASET_PATH = REPO_DIR / "datasets" / "artwork" / "paintings.csv"
IMAGES_DIR   = REPO_DIR / "datasets" / "artwork" / "images"

print("RUN_DIR            :", RUN_DIR.resolve())
print("REPO_DIR           :", REPO_DIR.resolve())
print("IMAGE_QUERIES_YAML :", IMAGE_QUERIES_YAML.resolve())
print("DATASET_PATH       :", DATASET_PATH.resolve())
print("IMAGES_DIR         :", IMAGES_DIR.resolve())
"""
        ),
        md(
            """### Step 3 — Load model + kvpress / Llava patches

与 **CompressionExperiments** 的 `engine.py` 一致：在加载 **LlavaNext** 后应用
`DynamicCache` / `BasePress.forward_hook` / `language_model` 等补丁，便于 **ExpectedAttention**、
**KVzip**、**Finch** 在 transformers 5.x 下工作。

> **Prerequisite**: Step 1 完成（必要时重启内核）→ **Step 2** → 本格。  
> 若 OOM 或断连重启：下次 **Step 2 → Step 3 → Step 4**。
"""
        ),
        code(
            """\
import sys
import torch
import transformers
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig

if "REPO_DIR" not in globals():
    raise RuntimeError("请先运行 Step 2 定义 REPO_DIR / RUN_DIR。")

print("transformers:", transformers.__version__)

MODEL_ID = "llava-hf/llama3-llava-next-8b-hf"
MODEL_TAG = MODEL_ID.split("/")[-1]
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

_art_eval = REPO_DIR / "benchmarks" / "artwork_eval"
sys.path.insert(0, str(_art_eval))
from llava_kvpress_patch import apply_kvpress_compatibility_patches

apply_kvpress_compatibility_patches(model)
print("Model ready + kvpress patches:", MODEL_ID, "| MODEL_TAG:", MODEL_TAG)
"""
        ),
        md(
            """### Step 4 — 实验配置（与 CE 一致的目录与 press 名）

- **RESULTS_ROOT** = `RUN_DIR / "results"`，其下 **`artwork / MODEL_TAG / PressClass / 0.xx / results.csv`**。
- **PRESS_NAMES**：类名字符串列表（如 `ExpectedAttentionPress`），与 `evaluation_config` / GT 评测一致。
- 可选打开 **KVzipPress** / **FinchPress**（需本机环境支持；Finch 会在每次推理前 `update_model_and_tokenizer`）。

内核重启后请 **Step 2 → Step 3 → Step 4**。
"""
        ),
        code(
            """\
from __future__ import annotations
import gc
import os
import time
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import yaml
from PIL import Image

if "RUN_DIR" not in globals() or "MODEL_TAG" not in globals():
    raise RuntimeError("请先顺序运行 Step 2、Step 3。")

DATASET_NAME = "artwork"
RESULTS_ROOT = RUN_DIR / "results"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

MAX_ROWS = 0  # 0 = 全表；否则 head(MAX_ROWS)
MAX_NEW_TOKENS = 50
COMPRESSION_RATIOS = [0.4, 0.8, 0.95]

# 与 CE PRESS_REGISTRY 键一致（写入 results.csv 的 press 列）
PRESS_NAMES = ["ExpectedAttentionPress"]
ENABLE_KVZIP_ON_LLAVA = False
ENABLE_FINCH_ON_LLAVA = False

if ENABLE_KVZIP_ON_LLAVA and "KVzipPress" not in PRESS_NAMES:
    PRESS_NAMES = PRESS_NAMES + ["KVzipPress"]
if ENABLE_FINCH_ON_LLAVA and "FinchPress" not in PRESS_NAMES:
    PRESS_NAMES = PRESS_NAMES + ["FinchPress"]

# ---- 与 engine.run_single 一致的 image extract 后缀 ----
_IMAGE_EXTRACT_SUFFIX = (
    " (be concise, no explanation, no introductory text, just the answer,"
    " output datatype: STRING, do not repeat the datatype in the answer) ?"
)


def build_answer_prefix(question: str, is_boolean: bool) -> str:
    context_ref = "image"
    if is_boolean:
        instruction = (
            f"Answer the following question based on the {context_ref}"
            " with '1' or '0'. Do not add any other comments."
        )
        formatted_question = question
    else:
        instruction = (
            f"Answer the following question based on the {context_ref}."
            " Do not add any other comments."
        )
        formatted_question = question + _IMAGE_EXTRACT_SUFFIX
    return f"{instruction} {formatted_question}\nAnswer: "


def save_results_ce_style(df: pd.DataFrame, base_dir: Path, dataset: str, model_tag: str) -> None:
    # Same layout as CE utils.save_results: one results.csv per (press, ratio).
    if df.empty:
        print("save_results_ce_style: empty DataFrame, skip.")
        return
    base_dir = Path(base_dir)
    for (press, ratio), group in df.groupby(["press", "ratio"]):
        ratio_tag = f"{float(ratio):.2f}"
        out_dir = base_dir / dataset / model_tag / str(press) / ratio_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "results.csv"
        group.to_csv(out_path, index=False)
        print("Saved", out_path)
"""
        ),
        md(
            """### Step 5 — 读 image_queries.yaml、构建记录与问句

`record_id` 与 **CompressionExperiments** 的 `load_dataset` 一致：当前 `paintings.csv` 行序下的
**从 0 开始的行号**（与 `ground_truth/query_*.csv` 的 `_index_artworks` 对齐）。

图片路径：先试 URL 解码文件名，再试原始 `%20` 字面名（Drive 上常见）。
"""
        ),
        code(
            """\
from kvpress import ExpectedAttentionPress, FinchPress, KVzipPress

_PRESS_REGISTRY = {
    "ExpectedAttentionPress": ExpectedAttentionPress,
    "KVzipPress": KVzipPress,
    "FinchPress": FinchPress,
}

with open(IMAGE_QUERIES_YAML, encoding="utf-8") as f:
    _y = yaml.safe_load(f)
art_cfg = _y["artwork"]
filter_queries = list(art_cfg["filter_queries"])
extract_queries = list(art_cfg["extract_queries"])
image_url_column = art_cfg.get("image_url_column", "image_url")

df = pd.read_csv(DATASET_PATH)
df = df[df[image_url_column].notna()].copy().reset_index(drop=True)
if MAX_ROWS and MAX_ROWS > 0:
    df = df.head(int(MAX_ROWS))
df.insert(0, "record_id", range(len(df)))


def resolve_artwork_image_path(url: str) -> str:
    tail = url.split("/")[-1].split("?")[0]
    p_decoded = IMAGES_DIR / unquote(tail)
    p_raw = IMAGES_DIR / tail
    if p_decoded.is_file():
        return str(p_decoded)
    if p_raw.is_file():
        return str(p_raw)
    return str(p_decoded)


df["image_path"] = df[image_url_column].apply(resolve_artwork_image_path)

queries_plan: list[tuple[str, bool]] = (
    [(q, True) for q in filter_queries] + [(q, False) for q in extract_queries]
)

print("Rows:", len(df), "| filter_q:", len(filter_queries), "| extract_q:", len(extract_queries))
print("Total (record × query) pairs:", len(df) * len(queries_plan))
"""
        ),
        md(
            """### Step 6 — 单次推理（chat template + kvpress context）

用户消息文本 = **Step 4** 的 `build_answer_prefix(...)`（与 CE `CompressionEngine.run_single` 中
`answer_prefix` 一致）。图像经 processor 与 CE 的 image 管线一致地放入 user turn。
"""
        ),
        code(
            """\
import torch

_tok = getattr(processor, "tokenizer", processor)


def run_generate_vision(
    image_path: str,
    answer_prefix: str,
    press_name: str,
    compression_ratio: float,
) -> str:
    sz = os.path.getsize(image_path)
    if sz < 512:
        raise OSError(f"Image too small ({sz} B), likely truncated: {image_path}")

    PressCls = _PRESS_REGISTRY[press_name]
    press = PressCls(compression_ratio=float(compression_ratio))
    if hasattr(press, "update_model_and_tokenizer"):
        press.update_model_and_tokenizer(model, _tok)

    image = Image.open(image_path).convert("RGB")
    conversation = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": answer_prefix}]}
    ]
    formatted_prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True
    )
    inputs = processor(images=image, text=formatted_prompt, return_tensors="pt").to(model.device)

    gen_kwargs = {"max_new_tokens": MAX_NEW_TOKENS}
    if getattr(_tok, "pad_token_id", None) is not None:
        gen_kwargs["pad_token_id"] = _tok.pad_token_id

    with torch.no_grad():
        with press(model):
            out = model.generate(**inputs, **gen_kwargs)

    inp_len = inputs["input_ids"].shape[1]
    return _tok.decode(out[0][inp_len:], skip_special_tokens=True).strip()
"""
        ),
        md(
            """### Step 7 — 全量推理并写入 CE 风格 results 树

对每个 **press × ratio**：跑完所有 `(record_id, query)` 后写一份 **`results.csv`**
（列：`record_id`, `query`, `press`, `ratio`, `answer`）。

若某行图片不存在则跳过该 record 的全部 query（不写入占位符，与「无结果」区分）。
"""
        ),
        code(
            """\
rows_all: list[dict] = []
n_skip_missing = 0

for _, row in df.iterrows():
    ip = row["image_path"]
    if not os.path.isfile(ip):
        n_skip_missing += 1
        continue
    rid = int(row["record_id"])
    for qtext, is_bool in queries_plan:
        ap = build_answer_prefix(qtext, is_bool)
        for pname in PRESS_NAMES:
            for ratio in COMPRESSION_RATIOS:
                err = ""
                ans = ""
                t0 = time.perf_counter()
                try:
                    ans = run_generate_vision(ip, ap, pname, float(ratio))
                except Exception as e:
                    err = str(e)[:800]
                _ = time.perf_counter() - t0
                if err:
                    print(f"[err] rid={rid} press={pname} r={ratio} q={qtext[:40]}… -> {err[:120]}")
                else:
                    rows_all.append(
                        {
                            "record_id": rid,
                            "query": qtext,
                            "press": pname,
                            "ratio": float(ratio),
                            "answer": ans,
                        }
                    )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

out_df = pd.DataFrame(rows_all)
print("rows written:", len(out_df), "| skip_missing_image_records:", n_skip_missing)
if len(out_df):
    save_results_ce_style(out_df, RESULTS_ROOT, DATASET_NAME, MODEL_TAG)
else:
    print("无有效行：请检查 IMAGES_DIR 与 paintings 中 URL 对应文件名。")
    print("RESULTS_ROOT:", RESULTS_ROOT)
"""
        ),
        md(
            """### Step 8 — 与 CE ``evaluate.py`` 相同的 P/R/F1 汇总

使用 **`benchmarks/artwork_eval/evaluation/evaluator.py`** 的 `EvaluationManager`，对
**`RESULTS_ROOT`** 下所有 **`results.csv`** 扫一遍，对齐 **`evaluation_config.yaml`** 中的
query 文本与 **`ground_truth/query_*.csv`**。

也可在仓库根执行：
`python benchmarks/artwork_eval/evaluate.py --results-dir <你的 RUN_DIR>/results`
"""
        ),
        code(
            """\
import sys

if "RESULTS_ROOT" not in globals():
    raise RuntimeError("请先运行 Step 4 与 Step 7。")

_ae = REPO_DIR / "benchmarks" / "artwork_eval"
sys.path.insert(0, str(_ae))
from evaluation.evaluator import EvaluationManager

mgr = EvaluationManager(
    config_path=str(EVAL_CONFIG_YAML),
    results_dir=str(RESULTS_ROOT),
)
ev_df = mgr.evaluate_all()
if ev_df.empty:
    print("未得到任何评测行：确认 Step 7 已生成 results/artwork/.../results.csv，且 query 字符串与 yaml 完全一致。")
else:
    display(ev_df.head(20))
    gcols = ["dataset", "model_tag", "press", "ratio"]
    mcols = ["precision", "recall", "f1"]
    for label, subset in [
        ("filter", ev_df[ev_df["query_type"] == "filter"]),
        ("extract", ev_df[ev_df["query_type"] == "extract"]),
        ("all", ev_df),
    ]:
        print("\\n===", label, "===")
        display(subset.groupby(gcols)[mcols].mean().round(4))

    summ_path = RUN_DIR / "evaluation_results.csv"
    parts = []
    for label, subset in [
        ("filter", ev_df[ev_df["query_type"] == "filter"]),
        ("extract", ev_df[ev_df["query_type"] == "extract"]),
        ("all", ev_df),
    ]:
        agg = subset.groupby(gcols)[mcols].mean().round(4).reset_index()
        agg.insert(len(gcols), "query_type", label)
        parts.append(agg)
    import pandas as pd

    summary = pd.concat(parts, ignore_index=True).sort_values(gcols + ["query_type"])
    summary.to_csv(summ_path, index=False)
    print("Saved summary:", summ_path.resolve())
"""
        ),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}},
        "cells": cells,
    }
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
