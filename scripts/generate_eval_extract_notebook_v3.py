"""Generate eval_extract_v3.ipynb from eval_extract_v2.ipynb template.

Run from repo root:
    python scripts/generate_eval_extract_notebook_v3.py
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
IN_NB = ROOT / "eval_extract_v2.ipynb"
OUT_NB = ROOT / "eval_extract_v3.ipynb"


def set_cell_source(nb: dict, idx: int, text: str) -> None:
    nb["cells"][idx]["source"] = text.splitlines(keepends=True)


def main() -> None:
    if not IN_NB.is_file():
        raise SystemExit(f"Missing template notebook: {IN_NB}")

    nb = json.loads(IN_NB.read_text(encoding="utf-8"))

    # 0) Intro markdown
    set_cell_source(
        nb,
        0,
        """# Extract evaluation v3 (query_010-012): EA vs KVzip with Qwen2.5 0.5B + Prefill KV cache

Self-contained for **Google Colab** (open this notebook from GitHub: *File -> Open notebook -> GitHub*).

Model: **[Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)** (Apache-2.0; use a recent `transformers` as in the model card).

## Data (scheme B)

1. This run uses **`query_010.csv` ... `query_012.csv`** only (3 extract tasks). Your zip or folder may contain more CSVs; only these three must be present.
2. **Colab data**: by default the notebook uses **`MyDrive/kv-compression-benchmark/movie_results/`** after Step 2b (if that folder exists); otherwise **`/content/movie_result/`**. Upload `query_010.csv` … `query_012.csv` there.
3. If your folder name differs (e.g. nested `movie-results/movie-results/`), set **`MOVIE_RESULTS_DIR`** in the config cell.

## Robustness

- **Why KVzip feels slow**: `KVzipPress` does **multiple forward passes** per token (see library warnings). On **T4**, **~60-120 s per `(row, ratio)`** is common; **Expected Attention** is usually much faster. Tip: set **`ENABLE_KVZIP = False`** in the config cell to draw an **EA-only** curve first, or shrink **`COMPRESSION_RATIOS`** to 2-3 values, then turn KVzip back on for the final run (use Drive checkpoint + resume).
- **Model + decode path**: v3 uses **Qwen2.5 0.5B Instruct** and a **prefill KV cache** flow. For each `(row, ratio, method)`, review prefix is prefetched once, then all query tasks decode from that cache clone.
- **Rows per query (`MAX_ROWS_PER_QUERY`)**: Default **120** rows sampled uniformly from each `query_010.csv` (aligned index reused for 011-012 when row counts match). Set **`MAX_ROWS_PER_QUERY = 0`** to use **`SAMPLE_FRAC`** only (no hard cap).
- **Compression ratios**: Default **3** values (`COMPRESSION_RATIOS` in config)-fewer ratio points than a dense grid to save wall time; edit the list as needed.
- **Smoke test (`SMOKE_MAX_ROWS`)**: **`SMOKE_MAX_ROWS = 3`** further caps to the first 3 rows after the above sampling. Use for a quick pipeline check.
- **Checkpoints**: Each finished `(row, ratio, method)` is appended with flush to **`extract_predictions_checkpoint.csv`** under **`RUN_DIR`**. **Step 2b** mounts Drive (if enabled) before loading the model; default **`DRIVE_SUBDIR`** is **`kv-compression-benchmark/extract_eval_v3_qwen05_q10_12`**.
- Set **`RESUME_FROM_CHECKPOINT = True`**, then after reconnect run **Step 1 -> ... -> Step 2b -> model load -> inference** again; the loop skips keys already in the checkpoint.
- After inference completes, **re-run only the metrics/plots cells** if you change parsing (no full re-generate).

## Order

Run cells **top to bottom** once; then you can re-run from the metrics cell onward. **Configuration** assumes **Step 2b** has already set **`RUN_DIR`** / **`CHECKPOINT_PATH`** / **`OUT_DIR`**.
""",
    )

    # 0b) Step 2 title: Qwen weights are not Llama-gated
    c5 = "".join(nb["cells"][5]["source"])
    c5 = c5.replace(
        "### Step 2: Hugging Face login (Llama gated model)",
        "### Step 2: Hugging Face login (optional; Qwen is Apache-2.0)",
    )
    nb["cells"][5]["source"] = c5.splitlines(keepends=True)

    # 1) Drive subdir in Step 2b (cell 8)
    c8 = "".join(nb["cells"][8]["source"])
    c8 = c8.replace(
        'DRIVE_SUBDIR = "kv-compression-benchmark/extract_eval_v2_q10_12"',
        'DRIVE_SUBDIR = "kv-compression-benchmark/extract_eval_v3_qwen05_q10_12"',
    )
    c8 = c8.replace(
        "# Distinct folder so v2 (010–012) does not resume from older 010–019 checkpoints.",
        "# v3: Qwen 0.5B + prefill; checkpoint dir separate from v2.",
    )
    nb["cells"][8]["source"] = c8.splitlines(keepends=True)

    # 2) Model markdown + code (cells 11, 12)
    set_cell_source(nb, 11, "### Step 4: Load Qwen2.5 0.5B Instruct")
    set_cell_source(
        nb,
        12,
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
""",
    )

    # 3) Config markdown hint (cell 13)
    c13 = "".join(nb["cells"][13]["source"])
    c13 = c13.replace(
        "### Configuration: data path, sampling, query list",
        "### Configuration: data path, sampling, query list (v3 prefill)",
    )
    c13 = c13.replace(
        "- Default **`MOVIE_RESULTS_DIR`** on Colab: **`/content/movie_result`**",
        "- Default **Colab data path**: after Step 2b, **`/content/drive/MyDrive/kv-compression-benchmark/movie_results`** if that folder exists; otherwise **`/content/movie_result`**",
    )
    nb["cells"][13]["source"] = c13.splitlines(keepends=True)

    # 3d) Prefer CSVs on Google Drive (cell 14)
    c14 = "".join(nb["cells"][14]["source"])
    _old_movie = """if RUN_ON_COLAB:
    MOVIE_RESULTS_DIR = Path("/content/movie_result")
else:
    MOVIE_RESULTS_DIR = Path("movie-results/movie-results")"""
    _new_movie = """if RUN_ON_COLAB:
    _drive_movie = Path("/content/drive/MyDrive/kv-compression-benchmark/movie_results")
    MOVIE_RESULTS_DIR = _drive_movie if _drive_movie.is_dir() else Path("/content/movie_result")
else:
    MOVIE_RESULTS_DIR = Path("movie-results/movie-results")"""
    if _old_movie not in c14:
        raise SystemExit("v3 generator: expected MOVIE_RESULTS_DIR block missing in template cell 14")
    c14 = c14.replace(_old_movie, _new_movie)
    nb["cells"][14]["source"] = c14.splitlines(keepends=True)

    # 4) Helpers markdown + code (cells 15, 16)
    set_cell_source(
        nb,
        15,
        """### Helpers: prefill+decode, checkpoint IO, normalize for metrics

Run this cell before the inference loop.
""",
    )

    set_cell_source(
        nb,
        16,
        """from __future__ import annotations

import torch
from kvpress import ExpectedAttentionPress, KVzipPress
from tqdm.auto import tqdm


def build_review_prefix(review_text: str) -> str:
    return f"Review: {review_text}\\n"


def build_task_suffix(task: str) -> str:
    return f"Task: {task}\\nAnswer:"


def _clone_past_key_values(pkv):
    if pkv is None:
        return None
    # Try cloning legacy cache tuples first; if unsupported, keep object as-is.
    try:
        if hasattr(pkv, "to_legacy_cache"):
            pkv = pkv.to_legacy_cache()
    except Exception:
        pass
    try:
        return tuple(tuple(t.detach().clone() for t in layer) for layer in pkv)
    except Exception:
        return pkv


def prefill_review_with_press(tok, mdl, review_text: str, press_cls, compression_ratio: float):
    review_prefix = build_review_prefix(review_text)
    inputs = tok(review_prefix, return_tensors="pt").to(mdl.device)
    with press_cls(compression_ratio=compression_ratio)(mdl):
        with torch.no_grad():
            out = mdl(**inputs, use_cache=True)
    return _clone_past_key_values(out.past_key_values)


def decode_task_from_prefilled(tok, mdl, prefilled_pkv, task: str) -> str:
    task_suffix = build_task_suffix(task)
    inputs = tok(task_suffix, return_tensors="pt", add_special_tokens=False).to(mdl.device)
    gen_kw = {"max_new_tokens": MAX_NEW_TOKENS}
    if getattr(tok, "pad_token_id", None) is not None:
        gen_kw["pad_token_id"] = tok.pad_token_id
    pkv = _clone_past_key_values(prefilled_pkv)
    outputs = mdl.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask", None),
        past_key_values=pkv,
        **gen_kw,
    )
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
    p = norm_ws(pred).strip(" \\t")
    g = norm_ws(gold).strip(" \\t")
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
""",
    )

    # 5) Inference markdown + code (cells 17, 18)
    set_cell_source(
        nb,
        17,
        """### Inference loop (checkpointed, prefill-enabled)

**Checkpointed:** each row commits to disk (Drive if mounted in Step 2b). Safe if Colab disconnects - reconnect, run setup through Step 2b + model load + this cell with `RESUME_FROM_CHECKPOINT = True`. Quick test: `SMOKE_MAX_ROWS = 3`.

For each `(row, ratio, method)`, v3 prefills review KV once, then decodes all `QUERY_IDS` tasks from cloned cache.
""",
    )

    set_cell_source(
        nb,
        18,
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

# Build sampled subsets for all queries; shared index when row counts match query_010
sub_by_qid = {}
for qid in QUERY_IDS:
    path = MOVIE_RESULTS_DIR / f"query_{qid:03d}.csv"
    df = pd.read_csv(path, dtype=str)
    if len(df) != len(df0):
        print(f"WARN query_{qid}: row count {len(df)} != query_010 {len(df0)}; sampling independently.")
        alt_idx = sorted(random.sample(df.index.tolist(), min(len(idx_sample), len(df))))
        sub = df.loc[alt_idx].reset_index(drop=True)
    else:
        sub = df.loc[idx_sample].reset_index(drop=True)
    sub_by_qid[qid] = sub

base_qid = QUERY_IDS[0]
base_sub = sub_by_qid[base_qid]

for i in tqdm(range(len(base_sub)), desc="rows"):
    base_row = base_sub.iloc[i]
    review = str(base_row["reviewtext"])

    for ratio in COMPRESSION_RATIOS:
        for mname, Press in methods:
            # Prefill once per (row, ratio, method), then decode all tasks from cloned cache.
            prefill_err = ""
            prefilled_pkv = None
            try:
                prefilled_pkv = prefill_review_with_press(
                    tokenizer, model, review, Press, float(ratio)
                )
            except Exception as e:
                prefill_err = str(e)[:500]

            for qid in QUERY_IDS:
                sub = sub_by_qid[qid]
                if i >= len(sub):
                    continue
                row = sub.iloc[i]
                rkey = stable_row_key(row)
                gold = str(row["answer"])
                task = TASK_BY_QUERY_ID[qid]

                key = (qid, rkey, float(ratio), mname)
                if key in done:
                    continue

                err = prefill_err
                pred = ""
                if not err:
                    try:
                        pred = decode_task_from_prefilled(
                            tokenizer, model, prefilled_pkv, task
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
""",
    )

    OUT_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT_NB}")


if __name__ == "__main__":
    main()

