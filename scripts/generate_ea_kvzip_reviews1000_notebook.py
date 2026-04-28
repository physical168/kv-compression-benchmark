"""Generate eval_ea_kvzip_reviews1000_qwen05b.ipynb for Colab runs."""
from __future__ import annotations
import json
from pathlib import Path

def md(text: str) -> dict: return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}
def code(text: str) -> dict: return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval_ea_kvzip_reviews1000_qwen05b.ipynb"

def main() -> None:
    cells = [
        md("# EA & KVzip 4x4 evaluation (Qwen2.5-0.5B, sentiment on reviews_1000)\n\nThis Colab-first notebook runs a 4x4 grid:\n\n- `METHOD` in `{'ea', 'kvzip'}`\n- `USE_CPT` in `{False, True}`\n- `COMPRESSION_RATIOS = [0.2, 0.4, 0.6, 0.8]`\n\nData source is `reviews_1000.csv` (first 100 rows)."),
        md("### Step 1: Install dependencies & patch"),
        code("!pip install -q transformers accelerate pandas scikit-learn matplotlib tqdm kvpress\n\n"
             "import torch\n"
             "from transformers.cache_utils import QuantizedLayer\n"
             "import kvpress.utils as _kv_utils\n"
             "import kvpress.presses.base_press as _kv_base\n"
             "import kvpress.presses.kvzip_press as _kv_kz\n\n"
             "def _extract_keys_and_values_fixed(cache, layer_idx: int):\n"
             "    layer = cache.layers[layer_idx]\n"
             "    if isinstance(layer, QuantizedLayer):\n"
             "        return _kv_utils.dequantize_layer(layer)\n"
             "    return layer.keys, layer.values\n\n"
             "def _forward_hook_fixed(self, module, input, kwargs, output):\n"
             "    hidden_states = kwargs[\"hidden_states\"]\n"
             "    cache = kwargs[\"past_key_values\"]\n"
             "    cache_layer = cache.layers[module.layer_idx]\n"
             "    q_len = hidden_states.shape[1]\n"
             "    if kwargs[\"cache_position\"][-1] > q_len:\n"
             "        return output\n"
             "    keys, values = _extract_keys_and_values_fixed(cache, module.layer_idx)\n"
             "    keys, values = self.compress(module, hidden_states, keys, values, output[1], kwargs)\n"
             "    if hasattr(self, 'window_size') and self.window_size:\n"
             "        pass # Custom window_size handling removed for EA/KVzip\n"
             "    if isinstance(cache_layer, QuantizedLayer):\n"
             "        cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)\n"
             "        cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)\n"
             "        cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)\n"
             "        cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)\n"
             "        cache_layer.cumulative_length = keys.shape[2]\n"
             "    else:\n"
             "        cache_layer.keys = keys\n"
             "        cache_layer.values = values\n"
             "    return output\n\n"
             "def _kvzip_forward_hook_fixed(self, module, input, kwargs, output):\n"
             "    hidden_states = kwargs[\"hidden_states\"]\n"
             "    cache = kwargs.get(\"past_key_values\", None) or kwargs.get(\"past_key_value\", None)\n"
             "    cache_layer = cache.layers[module.layer_idx]\n"
             "    keys, values = _extract_keys_and_values_fixed(cache, module.layer_idx)\n"
             "    keys, values = self.score_kvzip(module, hidden_states, keys, values, output[1], kwargs)\n"
             "    if isinstance(cache_layer, QuantizedLayer):\n"
             "        cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)\n"
             "        cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)\n"
             "        cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)\n"
             "        cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)\n"
             "        cache_layer.cumulative_length = keys.shape[2]\n"
             "    else:\n"
             "        cache_layer.keys = keys\n"
             "        cache_layer.values = values\n"
             "    return output\n\n"
             "_kv_utils.extract_keys_and_values = _extract_keys_and_values_fixed\n"
             "_kv_base.BasePress.forward_hook = _forward_hook_fixed\n"
             "_kv_kz.KVzipPress.forward_hook = _kvzip_forward_hook_fixed\n"
             "print(\"kvpress Step 1b patch applied.\")"
        ),
        md("### Step 2: Mount Google Drive"),
        code(
"""from pathlib import Path\nimport os\n
RUN_ON_COLAB = os.path.isdir("/content")\nUSE_GOOGLE_DRIVE = True\nDRIVE_SUBDIR = "kv-compression-benchmark/ea_kvzip_reviews1000_qwen05b"
if RUN_ON_COLAB and USE_GOOGLE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    except ImportError:
        USE_GOOGLE_DRIVE = False
if RUN_ON_COLAB and USE_GOOGLE_DRIVE and Path("/content/drive/MyDrive").is_dir():
    RUN_DIR = Path("/content/drive/MyDrive") / DRIVE_SUBDIR
elif RUN_ON_COLAB:
    RUN_DIR = Path("/content/ea_kvzip_workspace")
else:
    RUN_DIR = Path("ea_kvzip_output")
RUN_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = RUN_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATASET_PATH = Path("/content/drive/MyDrive/kv-compression-benchmark/reviews_1000.csv")
if not DATASET_PATH.is_file() and RUN_ON_COLAB:
    DATASET_PATH = Path("/content/reviews_1000.csv")
if not DATASET_PATH.is_file():
    DATASET_PATH = Path("reviews_1000.csv")
print("RUN_DIR:", RUN_DIR.resolve())
print("DATASET_PATH:", DATASET_PATH.resolve())"""
        ),
        md("### Step 3: Load Model"),
        code(
            """import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
tokenizer = AutoTokenizer.from_pretrained(model_id)
if getattr(tokenizer, "pad_token_id", None) is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")
"""
        ),
        md("### Step 4: Configuration"),
        code(
"""from __future__ import annotations
import re, time, csv, gc
import pandas as pd
MAX_ROWS = 100
MAX_NEW_TOKENS = 8
RESUME_FROM_CHECKPOINT = True
COMPRESSION_RATIOS = [0.2, 0.4, 0.6, 0.8]

CONFIGS = [
    {"method": "ea", "use_cpt": False},
    {"method": "ea", "use_cpt": True},
    {"method": "kvzip", "use_cpt": False},
    {"method": "kvzip", "use_cpt": True},
]

RUNS_PATH = RUN_DIR / "ea_kvzip_runs.csv"
SUMMARY_PATH = RUN_DIR / "ea_kvzip_summary.csv"

def normalize_label(s: str) -> str | None:
    t = str(s).strip().lower()
    if t in ("positive", "pos", "1", "true"): return "positive"
    if t in ("negative", "neg", "0", "false"): return "negative"
    return None

def parse_sentiment(pred_text: str) -> str | None:
    t = str(pred_text).strip().lower()
    if re.search(r"\\bpositive\\b", t): return "positive"
    if re.search(r"\\bnegative\\b", t): return "negative"
    if t.startswith("pos"): return "positive"
    if t.startswith("neg"): return "negative"
    return None

def cfg_name(cfg: dict) -> str:
    return f"{cfg['method']}_cpt_{int(cfg['use_cpt'])}"

CK_FIELDS = ["config", "method", "use_cpt", "compression_ratio", "row_id", "reviewid", "gold", "pred_raw", "pred_label", "latency_ms", "error"]

def load_done(path):
    if not path.is_file() or not RESUME_FROM_CHECKPOINT: return set()
    ck = pd.read_csv(path)
    done = set()
    for _, r in ck.iterrows():
        err = str(r.get("error", "")).strip().lower()
        if err in ("", "nan"): done.add((str(r["config"]), str(r["compression_ratio"]), str(r["row_id"])))
    return done

def append_row(path, row):
    newfile = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CK_FIELDS, extrasaction="ignore")
        if newfile: w.writeheader()
        w.writerow({k: row.get(k, "") for k in CK_FIELDS})
        f.flush(); os.fsync(f.fileno())"""
        ),
        md("### Step 5: Adapter"),
        code(
            """from kvpress import ExpectedAttentionPress, KVzipPress
def run_generate(prompt: str, method: str, compression_ratio: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = {"max_new_tokens": MAX_NEW_TOKENS, "pad_token_id": tokenizer.pad_token_id}
    PressCls = ExpectedAttentionPress if method == "ea" else KVzipPress
    with torch.no_grad(), PressCls(compression_ratio=compression_ratio)(model):
        out = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(out[0], skip_special_tokens=True)
"""
        ),
        md("### Step 6: Inference"),
        code(
"""import random
from tqdm.auto import tqdm

random.seed(42)
df = pd.read_csv(DATASET_PATH, dtype=str)
df["gold"] = df["scoresentiment"].apply(normalize_label)
df = df[df["gold"].notna()].copy()
if MAX_ROWS and MAX_ROWS > 0:
    df = df.head(min(MAX_ROWS, len(df))).reset_index(drop=True)

done = load_done(RUNS_PATH)

for cfg in CONFIGS:
    name = cfg_name(cfg)
    method = cfg["method"]
    use_cpt = bool(cfg["use_cpt"])
    for ratio in COMPRESSION_RATIOS:
        run_name = f"{name}_r{ratio}"
        print("\\nRunning:", run_name)
        for i, row in tqdm(df.iterrows(), total=len(df), desc=run_name, leave=False):
            key = (name, str(ratio), str(i))
            if key in done: continue
            text = str(row["cpt"]) if use_cpt and pd.notna(row["cpt"]) and str(row["cpt"]).strip() else str(row["reviewtext"])
            prompt = f"Classify the movie review sentiment. Answer with one word only: positive or negative.\\n\\nReview: {text}\\nAnswer:"
            err, pred_raw, pred_label = "", "", ""
            t0 = time.perf_counter()
            try:
                pred_raw = run_generate(prompt, method=method, compression_ratio=float(ratio))
                tail = pred_raw.split("Answer:")[-1].strip()
                pred_label = parse_sentiment(tail) or parse_sentiment(pred_raw) or ""
            except Exception as e:
                err = str(e)[:500]
            latency_ms = (time.perf_counter() - t0) * 1000.0
            append_row(RUNS_PATH, {
                "config": name, "method": method, "use_cpt": use_cpt, "compression_ratio": ratio,
                "row_id": str(i), "reviewid": str(row["reviewid"]), "gold": str(row["gold"]),
                "pred_raw": pred_raw, "pred_label": pred_label, "latency_ms": latency_ms, "error": err
            })
            if not err: done.add(key)
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()"""
        ),
        md("### Step 7: Aggregation"),
        code(
"""import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

runs = pd.read_csv(RUNS_PATH)
runs["error"] = runs["error"].fillna("").astype(str).str.strip()
ok = runs[(runs["error"] == "") | (runs["error"].str.lower() == "nan")].copy()
ok = ok[ok["pred_label"].notna() & (ok["pred_label"].astype(str).str.len() > 0)]

rows = []
for cfg in ok["config"].unique():
    for r in ok["compression_ratio"].unique():
        part = ok[(ok["config"] == cfg) & (ok["compression_ratio"] == r)].copy()
        if len(part) == 0: continue
        y_true = part["gold"].astype(str).tolist()
        y_pred = part["pred_label"].astype(str).tolist()
        try: acc = float(accuracy_score(y_true, y_pred))
        except: acc = float("nan")
        try: f1m = float(f1_score(y_true, y_pred, average="macro"))
        except: f1m = float("nan")
        rows.append({
            "config": cfg, "method": part["method"].iloc[0], "use_cpt": part["use_cpt"].iloc[0],
            "compression_ratio": r, "accuracy": acc, "f1_macro": f1m,
            "latency_ms_mean": float(part["latency_ms"].astype(float).mean()), "n": int(len(part))
        })

summary = pd.DataFrame(rows).sort_values(["method", "use_cpt", "compression_ratio"]).reset_index(drop=True)
summary.to_csv(SUMMARY_PATH, index=False)
print("Wrote", SUMMARY_PATH)
display(summary)"""
        )
    ]
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": cells}
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Wrote", OUT)

if __name__ == "__main__": main()
