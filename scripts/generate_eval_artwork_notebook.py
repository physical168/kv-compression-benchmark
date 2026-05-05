"""Generate eval_artwork_llava.ipynb for the artwork benchmark.

Notebook goals:
- Fully English content (markdown + user-facing messages)
- Query strings aligned with evaluation_config.yaml
- CE-style output tree: results/{dataset}/{model_tag}/{press}/{ratio}/results.csv
- CE-style evaluator (P/R/F1)
- kvpress installed from CompressionExperiments repository
"""

from __future__ import annotations

import json
from pathlib import Path


def _cell_source(text: str) -> list[str]:
    """Store each cell source as one string to avoid split-line issues."""
    if not text.endswith("\n"):
        text += "\n"
    return [text]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _cell_source(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _cell_source(text),
    }


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval_artwork_llava.ipynb"


def main() -> None:
    cells = [
        md(
            """# Artwork Evaluation (Llava) — CE-aligned

This notebook runs the artwork benchmark with:
- query strings aligned to `benchmarks/artwork_eval/evaluation/evaluation_config.yaml`
- CE-compatible result layout
- CE evaluator (precision / recall / F1)
- `kvpress` installed from
  `https://github.com/GabrieleSanmartino/CompressionExperiments.git`

Dataset: `datasets/artwork/paintings.csv`  
Images: `datasets/artwork/images/`
"""
        ),
        md(
            """### Step 1 — Install dependencies (run once)

This cell installs Python dependencies and recompiles Pillow from source
(recommended for stable image handling on Colab).

After installation, the cell requests a kernel restart.  
When the runtime is back, continue with **Step 1b**.
"""
        ),
        code(
            """# One-time install + optional kernel restart
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
    print("Step 1 already completed. Continue with Step 1b.")
    print("To force reinstall, delete:", _MARK.resolve())
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
    print("Dependencies installed. Requesting kernel restart...")
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
        md(
            """### Step 1b — Clone repositories on Colab

This cell does two things:
1) Clones this benchmark repo to `/content/kv-compression-benchmark` (if needed)
2) Clones `CompressionExperiments` and installs `kvpress` from that repo:
   `/content/CompressionExperiments/kvpress`
"""
        ),
        code(
            """\
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BENCH_REPO_URL = "https://github.com/physical168/kv-compression-benchmark.git"
CE_REPO_URL = "https://github.com/GabrieleSanmartino/CompressionExperiments.git"
BENCH_DIR = Path("/content/kv-compression-benchmark")
CE_DIR = Path("/content/CompressionExperiments")
PATCH_FILE = BENCH_DIR / "benchmarks" / "artwork_eval" / "llava_kvpress_patch.py"

if not Path("/content").is_dir():
    print("Non-Colab environment: skipping clone/install.")
else:
    if not BENCH_DIR.is_dir():
        print("Cloning benchmark repo...")
        subprocess.check_call(["git", "clone", "--depth", "1", BENCH_REPO_URL, str(BENCH_DIR)])
    else:
        print("Benchmark repo exists:", BENCH_DIR)

    if not CE_DIR.is_dir():
        print("Cloning CompressionExperiments repo...")
        subprocess.check_call(["git", "clone", "--depth", "1", CE_REPO_URL, str(CE_DIR)])
    else:
        print("CompressionExperiments repo exists:", CE_DIR)

    # Ensure submodules (kvpress is a submodule in CE)
    subprocess.check_call(["git", "-C", str(CE_DIR), "submodule", "update", "--init", "--recursive"])
    kvpress_path = CE_DIR / "kvpress"
    if not kvpress_path.is_dir():
        raise FileNotFoundError(f"kvpress submodule not found: {kvpress_path}")

    print("Installing kvpress from CE repo:", kvpress_path)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(kvpress_path)])

    if not PATCH_FILE.is_file():
        raise FileNotFoundError(f"Missing patch file: {PATCH_FILE}")

    print("Step 1b complete.")
"""
        ),
        md(
            """### Step 2 — Mount Drive and configure paths

Path priority on Colab:
1) `/content/drive/MyDrive/kv-compression-benchmark`
2) `/content/kv-compression-benchmark` (from Step 1b)
"""
        ),
        code(
            """\
from pathlib import Path
import os

RUN_ON_COLAB = os.path.isdir("/content")
USE_GOOGLE_DRIVE = True
DRIVE_SUBDIR = "kv-compression-benchmark/artwork_eval_runs"

PATCH_REL = Path("benchmarks/artwork_eval/llava_kvpress_patch.py")

if RUN_ON_COLAB and USE_GOOGLE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    except ImportError:
        USE_GOOGLE_DRIVE = False
        print("google.colab not found; falling back to local paths.")

_drive_repo = Path("/content/drive/MyDrive/kv-compression-benchmark")
_clone_repo = Path("/content/kv-compression-benchmark")

if RUN_ON_COLAB and USE_GOOGLE_DRIVE and Path("/content/drive/MyDrive").is_dir():
    RUN_DIR = Path("/content/drive/MyDrive") / DRIVE_SUBDIR
elif RUN_ON_COLAB:
    RUN_DIR = Path("/content/artwork_eval_workspace")
else:
    RUN_DIR = Path("artwork_eval_output")

REPO_DIR: Path
if RUN_ON_COLAB:
    if (_drive_repo / PATCH_REL).is_file():
        REPO_DIR = _drive_repo
    elif (_clone_repo / PATCH_REL).is_file():
        REPO_DIR = _clone_repo
    else:
        REPO_DIR = _drive_repo if _drive_repo.is_dir() else _clone_repo if _clone_repo.is_dir() else Path("/content")
        print(
            "WARN: could not find", PATCH_REL, "- run Step 1b or copy full repo to",
            _drive_repo,
        )
else:
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
            """### Step 3 — Load model and apply Llava/kvpress patches

This applies compatibility patches from `benchmarks/artwork_eval/llava_kvpress_patch.py`
so kvpress works with Llava + transformers 5.x.
"""
        ),
        code(
            """\
import sys
import torch
import transformers
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig

if "REPO_DIR" not in globals():
    raise RuntimeError("Run Step 2 first (REPO_DIR / RUN_DIR are missing).")

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

import importlib.util

_art_eval = REPO_DIR / "benchmarks" / "artwork_eval"
_patch_py = _art_eval / "llava_kvpress_patch.py"
if not _patch_py.is_file():
    _nl = chr(10)
    raise FileNotFoundError(
        "Missing patch file: "
        + str(_patch_py.resolve())
        + _nl
        + "Expected: benchmarks/artwork_eval/llava_kvpress_patch.py"
        + _nl
        + "Run Step 1b or place full repo in Drive."
    )
_spec = importlib.util.spec_from_file_location("llava_kvpress_patch", _patch_py)
_llava_patch = importlib.util.module_from_spec(_spec)
sys.modules["llava_kvpress_patch"] = _llava_patch
assert _spec.loader is not None
_spec.loader.exec_module(_llava_patch)
_llava_patch.apply_kvpress_compatibility_patches(model)
print("Model ready + kvpress patches:", MODEL_ID, "| MODEL_TAG:", MODEL_TAG)
"""
        ),
        md(
            """### Step 4 — Experiment configuration

- **RESULTS_ROOT**: `RUN_DIR / "results"` with CE layout
- **PRESS_NAMES**: class names written to `press` column
- Toggle KVzip / Finch if needed

After runtime restart, rerun Step 2 -> Step 3 -> Step 4.
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
    raise RuntimeError("Run Step 2 and Step 3 first.")

DATASET_NAME = "artwork"
RESULTS_ROOT = RUN_DIR / "results"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

MAX_ROWS = 0  # 0 means full dataset; otherwise head(MAX_ROWS)
MAX_NEW_TOKENS = 50
COMPRESSION_RATIOS = [0.4, 0.8, 0.95]

# Keep names aligned with CE PRESS_REGISTRY keys.
PRESS_NAMES = ["ExpectedAttentionPress"]
ENABLE_KVZIP_ON_LLAVA = False
ENABLE_FINCH_ON_LLAVA = False

if ENABLE_KVZIP_ON_LLAVA and "KVzipPress" not in PRESS_NAMES:
    PRESS_NAMES = PRESS_NAMES + ["KVzipPress"]
if ENABLE_FINCH_ON_LLAVA and "FinchPress" not in PRESS_NAMES:
    PRESS_NAMES = PRESS_NAMES + ["FinchPress"]

# Same image extract suffix as CE engine.run_single
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
    return f"{instruction} {formatted_question}" + chr(10) + "Answer: "


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
            """### Step 5 — Build records and query plan

Filter/extract query strings are loaded from `evaluation_config.yaml`
(single source of truth for evaluation matching).
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

with open(EVAL_CONFIG_YAML, encoding="utf-8") as f:
    _ev = yaml.safe_load(f)["artwork"]
filter_queries = list(_ev["filter_query_mapping"].keys())
extract_queries = list(_ev["extract_query_mapping"].keys())

with open(IMAGE_QUERIES_YAML, encoding="utf-8") as f:
    _y = yaml.safe_load(f)["artwork"]
image_url_column = _y.get("image_url_column", "image_url")

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
            """### Step 6 — Single inference helper

Prompts follow CE `CompressionEngine.run_single` formatting.
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
            """### Step 7 — Run all queries and save CE-style results

For each `(press, ratio)`, the notebook writes:
`results/artwork/{model_tag}/{press}/{ratio}/results.csv`
with columns:
`record_id, query, press, ratio, answer`.
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
                            "query": str(qtext).strip(),
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
    print("No valid rows written. Check image paths under IMAGES_DIR.")
    print("RESULTS_ROOT:", RESULTS_ROOT)
"""
        ),
        md(
            """### Step 8 — CE-style evaluation (P/R/F1)

This cell evaluates all `results.csv` files under `RESULTS_ROOT`
using `benchmarks/artwork_eval/evaluation/EvaluationManager`.
"""
        ),
        code(
            """\
import os
import sys
from pathlib import Path

import pandas as pd

if "RESULTS_ROOT" not in globals():
    raise RuntimeError("Run Step 4 and Step 7 first.")

# Diagnostics for empty evaluation output
print("RESULTS_ROOT =", str(RESULTS_ROOT), "| exists:", os.path.isdir(RESULTS_ROOT))
_rroot = Path(RESULTS_ROOT)
_csvs = sorted(_rroot.rglob("results.csv")) if _rroot.is_dir() else []
print("results.csv count:", len(_csvs))
for _p in _csvs[:5]:
    print(" ", _p)
    print("   last 5 path parts:", _p.parts[-5:])
if _csvs:
    _s0 = pd.read_csv(_csvs[0])
    print("sample CSV columns:", list(_s0.columns), "rows:", len(_s0))
    if "query" in _s0.columns:
        print("first 2 queries:", _s0["query"].astype(str).head(2).tolist())
else:
    print("No results.csv found. Run Step 7 first.")

_ae = REPO_DIR / "benchmarks" / "artwork_eval"
sys.path.insert(0, str(_ae))
from evaluation.evaluator import EvaluationManager

mgr = EvaluationManager(
    config_path=str(EVAL_CONFIG_YAML),
    results_dir=str(RESULTS_ROOT),
)
ev_df = mgr.evaluate_all()
if ev_df.empty:
    print(
        "No evaluation rows produced. Common causes: "
        "(1) Step 7 did not write CSV files, "
        "(2) RESULTS_ROOT is wrong, "
        "(3) path depth is wrong (expected .../results/artwork/<model>/<press>/<ratio>/results.csv), "
        "(4) query strings in old runs do not match evaluation_config."
    )
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
