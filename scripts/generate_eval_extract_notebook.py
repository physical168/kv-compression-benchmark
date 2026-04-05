"""Generator for eval_extract_v2.ipynb — run from repo root: python scripts/generate_eval_extract_notebook.py"""
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
OUT = ROOT / "eval_extract_v2.ipynb"

cells = [
    md(
        """# Extract evaluation v2 (query_010–012): EA vs KVzip

Self-contained for **Google Colab** (open this notebook from GitHub: *File → Open notebook → GitHub*).

## Data (scheme B)

1. This run uses **`query_010.csv` … `query_012.csv`** only (3 extract tasks). Your zip or folder may contain more CSVs; only these three must be present.
2. **Typical Colab layout**: put CSVs directly under **`/content/movie_result/`** (see left file browser). The notebook defaults to that path.
3. If your folder name differs (e.g. nested `movie-results/movie-results/`), set **`MOVIE_RESULTS_DIR`** in the config cell.

## Robustness

- **Why KVzip feels slow**: `KVzipPress` does **multiple forward passes** per token (see library warnings). On **T4**, **~60–120 s per `(row, ratio)`** is common; **Expected Attention** is usually much faster. Tip: set **`ENABLE_KVZIP = False`** in the config cell to draw an **EA-only** curve first, or shrink **`COMPRESSION_RATIOS`** to 2–3 values, then turn KVzip back on for the final run (use Drive checkpoint + resume).

- **Rows per query (`MAX_ROWS_PER_QUERY`)**: Default **80** rows sampled uniformly from each `query_010.csv` (aligned index reused for 011–012 when row counts match). Set **`MAX_ROWS_PER_QUERY = 0`** to use **`SAMPLE_FRAC`** only (no hard cap).
- **Smoke test (`SMOKE_MAX_ROWS`)**: **`SMOKE_MAX_ROWS = 3`** further caps to the first 3 rows after the above sampling. Use for a quick pipeline check.
- **Checkpoints**: Each finished `(row, ratio, method)` is appended with flush to **`extract_predictions_checkpoint.csv`** under **`RUN_DIR`**. **Step 2b** mounts Drive (if enabled) before loading the model; default **`DRIVE_SUBDIR`** is **`kv-compression-benchmark/extract_eval_v2_q10_12`** so this notebook does not resume from the older 010–019 checkpoint path—change it if you intentionally want to share a folder.
- Set **`RESUME_FROM_CHECKPOINT = True`**, then after reconnect run **Step 1 → … → Step 2b → model load → inference** again; the loop skips keys already in the checkpoint.
- After inference completes, **re-run only the metrics/plots cells** if you change parsing (no full re-generate).

## Order

Run cells **top to bottom** once; then you can re-run from the metrics cell onward. **Configuration** assumes **Step 2b** has already set **`RUN_DIR`** / **`CHECKPOINT_PATH`** / **`OUT_DIR`**.
"""
    ),
    md(
        """### Step 1: Clone kvpress and install (Colab: `/content`)

Local Windows/Linux: if you do not have `/content`, the first cell skips clone when `kvpress` is already importable.
"""
    ),
    code(
        """import os
import shutil
import sys

CONTENT = "/content"
def on_colab() -> bool:
    return os.path.isdir(CONTENT)

if on_colab():
    os.chdir(CONTENT)
    if os.path.isdir("kvpress"):
        shutil.rmtree("kvpress")
    !git clone https://github.com/NVIDIA/kvpress.git
    %cd /content/kvpress
    !pip install -q -e .
    !pip install -q transformers accelerate bitsandbytes huggingface_hub scikit-learn pandas matplotlib tqdm
else:
    # Local: pip deps; put repo-root ./kvpress on path if you cloned it next to this project
    !pip install -q scikit-learn pandas matplotlib tqdm transformers accelerate bitsandbytes huggingface_hub
    _kr = os.path.abspath(os.path.join(os.getcwd(), "kvpress"))
    if os.path.isdir(_kr):
        sys.path.insert(0, _kr)

print("Step 1 done. CWD:", os.getcwd())
"""
    ),
    md("### Step 1b: QuantizedLayer cache patch (same as main_eval)"),
    code(
        """import torch
from transformers.cache_utils import QuantizedLayer

import kvpress.utils as _kv_utils
import kvpress.presses.base_press as _kv_base


def _extract_keys_and_values_fixed(cache, layer_idx: int):
    layer = cache.layers[layer_idx]
    if isinstance(layer, QuantizedLayer):
        return _kv_utils.dequantize_layer(layer)
    return layer.keys, layer.values


def _forward_hook_fixed(self, module, input, kwargs, output):
    hidden_states = kwargs["hidden_states"]
    cache = kwargs["past_key_values"]
    cache_layer = cache.layers[module.layer_idx]
    q_len = hidden_states.shape[1]
    if kwargs["cache_position"][-1] > q_len:
        return output
    keys, values = _extract_keys_and_values_fixed(cache, module.layer_idx)
    keys, values = self.compress(module, hidden_states, keys, values, output[1], kwargs)
    if isinstance(cache_layer, QuantizedLayer):
        cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)
        cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
        cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)
        cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)
        cache_layer.cumulative_length = keys.shape[2]
    else:
        cache_layer.keys = keys
        cache_layer.values = values
    return output


_kv_utils.extract_keys_and_values = _extract_keys_and_values_fixed
_kv_base.BasePress.forward_hook = _forward_hook_fixed

import kvpress.presses.kvzip_press as _kv_kz


def _kvzip_forward_hook_fixed(self, module, input, kwargs, output):
    hidden_states = kwargs["hidden_states"]
    cache = kwargs.get("past_key_values", None) or kwargs.get("past_key_value", None)
    cache_layer = cache.layers[module.layer_idx]
    keys, values = _extract_keys_and_values_fixed(cache, module.layer_idx)
    keys, values = self.score_kvzip(module, hidden_states, keys, values, output[1], kwargs)
    if isinstance(cache_layer, QuantizedLayer):
        cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)
        cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
        cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)
        cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)
        cache_layer.cumulative_length = keys.shape[2]
    else:
        cache_layer.keys = keys
        cache_layer.values = values
    return output


_kv_kz.KVzipPress.forward_hook = _kvzip_forward_hook_fixed
print("kvpress Step 1b patch applied.")
"""
    ),
    md("### Step 2: Hugging Face login (Llama gated model)"),
    code(
        """from huggingface_hub import notebook_login

notebook_login()
"""
    ),
    md(
        """### Step 2b: Google Drive — mount and set `RUN_DIR`

Runs **before** loading the model so checkpoints and figures are on Drive as soon as training starts. Edit **`USE_GOOGLE_DRIVE`** / **`DRIVE_SUBDIR`** here (local: always uses **`extract_eval_output`** under the current working directory).
"""
    ),
    code(
        """from pathlib import Path
import os

RUN_ON_COLAB = os.path.isdir("/content")
USE_GOOGLE_DRIVE = True
# Distinct folder so v2 (010–012) does not resume from older 010–019 checkpoints.
DRIVE_SUBDIR = "kv-compression-benchmark/extract_eval_v2_q10_12"

if RUN_ON_COLAB and USE_GOOGLE_DRIVE:
    try:
        from google.colab import drive

        drive.mount("/content/drive", force_remount=False)
    except ImportError:
        USE_GOOGLE_DRIVE = False
        print("google.colab not found; using local /content workspace only.")

if RUN_ON_COLAB and USE_GOOGLE_DRIVE and Path("/content/drive/MyDrive").is_dir():
    RUN_DIR = Path("/content/drive/MyDrive") / DRIVE_SUBDIR
elif RUN_ON_COLAB and USE_GOOGLE_DRIVE:
    print("WARN: Google Drive not mounted; using /content/extract_eval_workspace (session may reset).")
    RUN_DIR = Path("/content/extract_eval_workspace")
elif RUN_ON_COLAB:
    RUN_DIR = Path("/content/extract_eval_workspace")
else:
    RUN_DIR = Path("extract_eval_output")

RUN_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = RUN_DIR / "extract_predictions_checkpoint.csv"
OUT_DIR = RUN_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("RUN_DIR:", RUN_DIR.resolve())
print("CHECKPOINT_PATH:", CHECKPOINT_PATH.resolve())
print("OUT_DIR:", OUT_DIR.resolve())
"""
    ),
    md("### Step 3: bitsandbytes (Colab)"),
    code(
        """!pip install -q -U bitsandbytes>=0.46.1
"""
    ),
    md("### Step 4: Load Llama 3.1 8B 4-bit"),
    code(
        """import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "meta-llama/Llama-3.1-8B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    dtype=torch.float16,
)
print("Model loaded.")
"""
    ),
    md(
        """### Configuration: data path, sampling, query list

**Requires Step 2b** so **`RUN_DIR`**, **`CHECKPOINT_PATH`**, and **`OUT_DIR`** already exist.

- Default **`MOVIE_RESULTS_DIR`** on Colab: **`/content/movie_result`** (`query_010.csv` … `query_012.csv` for this notebook).
- This notebook runs **`QUERY_IDS_TO_RUN = [10, 11, 12]`** only; change that list if you want other extract tasks (must match available CSVs and task prompts).
- Edit **`SAMPLE_FRAC`**, **`SMOKE_MAX_ROWS`**, **`RESUME_FROM_CHECKPOINT`** below.
"""
    ),
    code(
        """from __future__ import annotations

import csv
import gc
import os
import random
import re
from pathlib import Path

import pandas as pd
import torch

RUN_ON_COLAB = os.path.isdir("/content")
if RUN_ON_COLAB:
    MOVIE_RESULTS_DIR = Path("/content/movie_result")
else:
    MOVIE_RESULTS_DIR = Path("movie-results/movie-results")

# --- Run knobs ---
# If MAX_ROWS_PER_QUERY > 0: that many rows per query (uniform random). If 0: use SAMPLE_FRAC only.
MAX_ROWS_PER_QUERY = 80
SAMPLE_FRAC = 0.08
# Full grid (slow on T4 with KVzip). Faster preset e.g. [0.2, 0.5, 0.9] or [0.4, 0.8].
COMPRESSION_RATIOS = [0.2, 0.4, 0.6, 0.8, 0.9]
MAX_NEW_TOKENS = 96
RANDOM_SEED = 42

# KVzip is much slower per call than EA; set False to benchmark EA only (or debug pipeline).
ENABLE_EXPECTED_ATTENTION = True
ENABLE_KVZIP = True

# Smoke: e.g. 3 = only first 3 rows per query (quick test). 0 = full sampled run.
SMOKE_MAX_ROWS = 0
RESUME_FROM_CHECKPOINT = True

QUERY_IDS_TO_RUN = [10, 11, 12]

_EXTRACT_QUERY_TEXTS = [
    "Extract the sentiment of the review (positive or negative):",
    "Extract the movie title mentioned in the review:",
    "Extract one actor that is praised particularly in the review (or 'none' if no actor is praised):",
    "Extract one aspect of the movie that is criticized in the review (choose from: plot, acting, cinematography, soundtrack, special effects, dialogue, or 'none'):",
    "Extract whether the reviewer would recommend the movie based on the review (yes/no):",
    "Extract the main emotion expressed by the reviewer (e.g., joy, disappointment, anger, excitement, confusion):",
    "Extract the director's name mentioned in the review (or 'none' if not mentioned):",
    "Extract whether the review contains spoilers (yes/no):",
    "Extract whether the reviewer compares the movie to another film (yes/no):",
    "Extract the target audience implied or stated (e.g., families, children, adults, fans of action, or 'none'):",
]
TASK_BY_QUERY_ID = {10 + i: t for i, t in enumerate(_EXTRACT_QUERY_TEXTS)}


def build_prompt(review_text: str, task: str) -> str:
    return f"Review: {review_text}\\nTask: {task}\\nAnswer:"


def decode_answer(outputs, tok) -> str:
    full = tok.decode(outputs[0], skip_special_tokens=True)
    return full.split("Answer:")[-1].strip()


def stable_row_key(row: pd.Series) -> str:
    rid = str(row.get("reviewid", ""))
    ir = str(row.get("_index_reviews", ""))
    return f"{rid}|{ir}"


def preflight(movie_dir: Path, qids: list[int]) -> list[int]:
    missing = []
    for q in qids:
        p = movie_dir / f"query_{q:03d}.csv"
        if not p.is_file():
            missing.append(q)
    if missing:
        raise FileNotFoundError(
            f"Missing CSV files for queries {missing} under {movie_dir}. "
            "Upload CSVs to Colab (e.g. /content/movie_result/query_010.csv … query_012.csv)."
        )
    if torch.cuda.is_available():
        print("CUDA:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: no CUDA — inference will be very slow.")
    return list(qids)


QUERY_IDS = preflight(MOVIE_RESULTS_DIR, QUERY_IDS_TO_RUN)
print("QUERY_IDS:", QUERY_IDS)
print("MOVIE_RESULTS_DIR:", MOVIE_RESULTS_DIR.resolve())
print("CHECKPOINT_PATH:", CHECKPOINT_PATH.resolve())
"""
    ),
    md(
        """### Helpers: generate, checkpoint IO, normalize for metrics

Run this cell before the inference loop.
"""
    ),
    code(
        """from __future__ import annotations

from kvpress import ExpectedAttentionPress, KVzipPress
from tqdm.auto import tqdm


def run_with_press(tok, mdl, review_text: str, task: str, press_cls, compression_ratio: float) -> str:
    prompt = build_prompt(review_text, task)
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    gen_kw = {"max_new_tokens": MAX_NEW_TOKENS}
    if getattr(tok, "pad_token_id", None) is not None:
        gen_kw["pad_token_id"] = tok.pad_token_id
    with press_cls(compression_ratio=compression_ratio)(mdl):
        outputs = mdl.generate(**inputs, **gen_kw)
    return decode_answer(outputs, tok)


CK_FIELDS = [
    "query_id",
    "row_key",
    "ratio",
    "method",
    "gold",
    "pred",
    "error",
]


def load_done_keys(path: Path) -> set[tuple]:
    if not path.is_file() or not RESUME_FROM_CHECKPOINT:
        return set()
    df_ck = pd.read_csv(path)
    keys = set()
    for _, r in df_ck.iterrows():
        if pd.isna(r.get("error")) or str(r.get("error")) in ("", "nan"):
            err = ""
        else:
            err = str(r["error"])
        if err and err.strip().lower() not in ("", "nan"):
            continue
        keys.add((int(r["query_id"]), str(r["row_key"]), float(r["ratio"]), str(r["method"])))
    return keys


def append_checkpoint(row: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    newfile = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CK_FIELDS, extrasaction="ignore")
        if newfile:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CK_FIELDS})
        f.flush()
        os.fsync(f.fileno())


def norm_ws(s: str) -> str:
    return re.sub(r"\\s+", " ", str(s).strip().lower())


def parse_yes_no(text: str) -> str | None:
    t = norm_ws(text)
    if re.search(r"\\byes\\b", t) and not re.search(r"\\bno\\b", t[:20]):
        return "yes"
    if re.search(r"\\bno\\b", t) and not re.search(r"\\byes\\b", t[:20]):
        return "no"
    if t.startswith("y") and not t.startswith("n"):
        return "yes"
    if t.startswith("n"):
        return "no"
    return None


def gold_to_yes_no(raw: str) -> str | None:
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
        if re.search(rf"\\b{re.escape(w)}\\b", t):
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
    if re.search(r"\\bnone\\b", t):
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
    \"\"\"Return (score 0/1 for acc, metric tag).\"\"\"
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
    if query_id in (11, 12, 16):
        return (1.0 if em_soft(pred, gold_s) else 0.0), "em"
    if query_id == 15:
        return (1.0 if em_soft(pred, gold_s) else 0.0), "em"
    if query_id == 19:
        return (1.0 if em_soft(pred, gold_s) else 0.0), "em"
    return (1.0 if em_soft(pred, gold_s) else 0.0), "unknown"


print("Helpers ready.")
"""
    ),
    md(
        """### Inference loop (checkpointed)

**Checkpointed:** each row commits to disk (Drive if mounted in Step 2b). Safe if Colab disconnects — reconnect, run setup through Step 2b + model load + this cell with `RESUME_FROM_CHECKPOINT = True`. Quick test: `SMOKE_MAX_ROWS = 3`.
"""
    ),
    code(
        """random.seed(RANDOM_SEED)

# Load base index from query_010
base_path = MOVIE_RESULTS_DIR / "query_010.csv"
df0 = pd.read_csv(base_path, dtype=str)
df0_idx = df0.index.tolist()
if MAX_ROWS_PER_QUERY and MAX_ROWS_PER_QUERY > 0:
    n = min(MAX_ROWS_PER_QUERY, len(df0_idx))
else:
    n = max(1, int(len(df0_idx) * SAMPLE_FRAC))
    n = min(n, len(df0_idx))
idx_sample = sorted(random.sample(df0_idx, n))
print("Rows per query:", len(idx_sample), "(MAX_ROWS_PER_QUERY=", MAX_ROWS_PER_QUERY, ")", flush=True)
if SMOKE_MAX_ROWS and SMOKE_MAX_ROWS > 0:
    idx_sample = idx_sample[: SMOKE_MAX_ROWS]
    print("SMOKE MODE: using", len(idx_sample), "rows per query")

done = load_done_keys(CHECKPOINT_PATH)
print("Resume: skipping", len(done), "completed keys from checkpoint")

methods = []
if ENABLE_EXPECTED_ATTENTION:
    methods.append(("ea", ExpectedAttentionPress))
if ENABLE_KVZIP:
    methods.append(("kvzip", KVzipPress))
if not methods:
    raise ValueError("Enable at least one of ENABLE_EXPECTED_ATTENTION / ENABLE_KVZIP")

print("Methods:", [m[0] for m in methods], "| ratios:", COMPRESSION_RATIOS)

for qid in tqdm(QUERY_IDS, desc="queries"):
    path = MOVIE_RESULTS_DIR / f"query_{qid:03d}.csv"
    df = pd.read_csv(path, dtype=str)
    if len(df) != len(df0):
        print(f"WARN query_{qid}: row count {len(df)} != query_010 {len(df0)}; sampling independently.")
        alt_idx = sorted(random.sample(df.index.tolist(), min(len(idx_sample), len(df))))
        sub = df.loc[alt_idx]
    else:
        sub = df.loc[idx_sample]
    task = TASK_BY_QUERY_ID[qid]

    for _, row in tqdm(sub.iterrows(), total=len(sub), desc=f"q{qid} rows", leave=False):
        rkey = stable_row_key(row)
        review = str(row["reviewtext"])
        gold = str(row["answer"])

        for ratio in COMPRESSION_RATIOS:
            for mname, Press in methods:
                key = (qid, rkey, float(ratio), mname)
                if key in done:
                    continue
                err = ""
                pred = ""
                try:
                    pred = run_with_press(
                        tokenizer, model, review, task, Press, float(ratio)
                    )
                except Exception as e:
                    err = str(e)[:500]
                append_checkpoint(
                    {
                        "query_id": qid,
                        "row_key": rkey,
                        "ratio": ratio,
                        "method": mname,
                        "gold": gold,
                        "pred": pred,
                        "error": err,
                    },
                    CHECKPOINT_PATH,
                )
                if not err:
                    done.add(key)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

print("Inference finished. Checkpoint:", CHECKPOINT_PATH)
"""
    ),
    md(
        """### Metrics aggregation + F1-style summary + plots

Re-runnable **without** re-inference if checkpoint exists.
"""
    ),
    code(
        """import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

ck = pd.read_csv(CHECKPOINT_PATH)
_err = ck["error"].fillna("").astype(str).str.strip()
ck_ok = ck[(_err == "") | (_err.str.lower() == "nan")].copy()
ck_ok = ck_ok[ck_ok["pred"].notna()]

rows_summary = []

for qid in QUERY_IDS:
    part = ck_ok[ck_ok["query_id"] == qid]
    for ratio in COMPRESSION_RATIOS:
        for mname in ("ea", "kvzip"):
            sub = part[
                (part["ratio"].astype(float) == float(ratio)) & (part["method"] == mname)
            ]
            if len(sub) == 0:
                continue
            y_true = []
            y_pred = []
            correct = []
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
            f1m = float("nan")
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

sum_df = pd.DataFrame(rows_summary)
sum_path = CHECKPOINT_PATH.parent / "extract_runs.csv"
sum_df.to_csv(sum_path, index=False)
print("Wrote", sum_path)

# Average across extract queries per (ratio, method)
avg = (
    sum_df.groupby(["ratio", "method"])[["accuracy", "f1_macro"]]
    .mean()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(8, 5))
for mname, label in [("ea", "ExpectedAttention"), ("kvzip", "KVzip")]:
    sub = avg[avg["method"] == mname]
    ax.plot(sub["ratio"], sub["f1_macro"], marker="o", label=label)
ax.set_xlabel("compression ratio")
ax.set_ylabel("mean f1_macro (per-query macro, then average)")
ax.set_title("Extract tasks (010–012): F1 vs compression")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fp = OUT_DIR / "extract_f1_vs_ratio.png"
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
print("Wrote", OUT)
