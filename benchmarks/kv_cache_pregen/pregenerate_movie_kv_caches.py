"""CLI: pre-generate text KV caches for movie review CSVs (kvpress ``kv-press-text-generation``).

**Legacy mode** (no ``--run-tags``): ExpectedAttention only; optional ``--use-cpt`` /
``--run-both-cpt-modes``; layout ``{cache_dir}/{model}/comp{ratio}/``.

**Run-tag mode** (``--run-tags``): one or more of ``ea``, ``kvzip``, ``finch_no_cpt``,
``finch_with_cpt``. Layout::

    {cache_dir}/{model_safe}/{run_tag}/comp{ratio}/cache_entry_{sha256}.pt

- ``ea`` / ``kvzip``: body is review + fixed sentiment task string (**no** CPT column).
- ``finch_no_cpt``: compressible = ``Review: …`` + review; window = task string.
- ``finch_with_cpt``: compressible = ``Review: …`` + review + task; window = ``CPT:`` block.
  Missing / empty ``cpt`` falls back to ``finch_no_cpt`` body.

Checkpoint CSV (``--checkpoint-csv``): resume skips batches with ``status=ok`` per
``(run_tag, compression_ratio)`` in run-tag mode; legacy uses ``(compression_ratio, use_cpt)``.

Example (one method per GPU session; reuse ``--cache-dir`` and ``--checkpoint-csv`` across sessions)::

    python pregenerate_movie_kv_caches.py --csv reviews_1000.csv --cache-dir ./caches \\
        --model Qwen/Qwen2.5-0.5B-Instruct --tail 500 \\
        --compression-ratio 0.2 0.4 0.6 0.8 --run-tags ea --checkpoint-csv ck.csv --bf16 --no-flash-attn

    For ``kvzip``, add ``--kvzip-patch`` when Qwen + ``QuantizedLayer`` raises in hooks.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm
from transformers import DynamicCache, pipeline  # type: ignore

import kvpress  # noqa: F401
from kvpress import ExpectedAttentionPress, FinchPress, KeyRerotationPress  # type: ignore

try:
    from kvpress import KVzipPress  # type: ignore
except ImportError:
    KVzipPress = None  # type: ignore

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from text_kvpress_patch import apply_kvpress_patches_text_only  # noqa: E402

from cache_io import (  # noqa: E402
    compression_tag,
    dynamic_cache_to_cpu_inplace,
    hash_path,
    save_cache,
    write_errors_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RUN_TAGS = ("ea", "kvzip", "finch_no_cpt", "finch_with_cpt")

# Finch clustering window (must not exceed prompt length; kvpress may require this to be set).
FINCH_WINDOW_DEFAULT = 16

PRESS_LEGACY = {
    "expected_attention": lambda cr: ExpectedAttentionPress(compression_ratio=cr),
}

REVIEW_PREFIX = "Review: "
REVIEW_SUFFIX = "\n"

MOVIE_SENTIMENT_TASK = (
    "Classify the movie review sentiment. "
    "Answer with one word only: positive or negative."
)


def _sanitize_model_dir(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


def apply_kvzip_quantized_layer_patch() -> None:
    """Match eval_ea_kvzip_reviews1000_qwen05b: fix BasePress/KVzip hooks with QuantizedLayer caches."""
    try:
        from transformers.cache_utils import QuantizedLayer  # type: ignore
    except ImportError:
        logger.warning("QuantizedLayer not available; KVzip patch skipped.")
        return

    import kvpress.utils as _kv_utils  # type: ignore
    import kvpress.presses.base_press as _kv_base  # type: ignore
    import kvpress.presses.kvzip_press as _kv_kz  # type: ignore

    if getattr(_kv_base.BasePress, "_kvzip_movie_pregen_patch", False):
        return

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

    _kv_utils.extract_keys_and_values = _extract_keys_and_values_fixed
    _kv_base.BasePress.forward_hook = _forward_hook_fixed
    _kv_kz.KVzipPress.forward_hook = _kvzip_forward_hook_fixed
    _kv_base.BasePress._kvzip_movie_pregen_patch = True
    logger.info("Applied KVzip / BasePress QuantizedLayer patch for movie pregen.")


def apply_finch_press_hook_fix() -> None:
    """Guard FinchPress.__call__ when hook registration fails (Kaggle / some builds)."""
    if getattr(FinchPress, "_hook_fix_applied", False):
        return

    @contextmanager
    def __finch_call(self, model):
        if self.delimiter_token_id is None:
            raise ValueError(
                "No delimiter token ID provided. Use update_model_and_tokenizer before calling the press."
            )
        self._is_multimodal = hasattr(model, "language_model")
        embed_tokens = (
            model.language_model.embed_tokens
            if self._is_multimodal
            else model.model.embed_tokens
        )
        with super(FinchPress, self).__call__(model):
            hook = None
            try:
                hook = embed_tokens.register_forward_hook(self.embed_token_forward_hook)
                yield
            finally:
                if hook is not None:
                    hook.remove()

    FinchPress.__call__ = __finch_call
    FinchPress._hook_fix_applied = True
    logger.info("Applied FinchPress.__call__ hook fix.")


def build_context_for_cache(review_text: str, max_chars: int) -> str:
    text = str(review_text).strip()
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars]
    return f"{REVIEW_PREFIX}{text}{REVIEW_SUFFIX}"


def build_ea_kvzip_body(review: str) -> str:
    review = str(review).strip()
    nl = "\n"
    return f"{review}{nl}{nl}{MOVIE_SENTIMENT_TASK}"


def build_legacy_body(row: pd.Series, text_col: str, use_cpt: bool, cpt_col: str) -> str:
    review = str(row[text_col]).strip()
    nl = "\n"
    core = f"{review}{nl}{nl}{MOVIE_SENTIMENT_TASK}"
    if use_cpt and cpt_col in row.index and pd.notna(row.get(cpt_col)) and str(row[cpt_col]).strip():
        cpt = str(row[cpt_col]).strip()
        raw = f"{core}{nl}{nl}CPT:{nl}{cpt}"
    else:
        raw = core
    return raw


def build_finch_prefill_string(press: FinchPress, review: str, with_cpt: bool, cpt: str | None) -> str:
    review = str(review).strip()
    if with_cpt and cpt and str(cpt).strip():
        cpt_t = str(cpt).strip()
        compress = f"{REVIEW_PREFIX}{review}{REVIEW_SUFFIX}{MOVIE_SENTIMENT_TASK}\n"
        window = f"CPT:\n{cpt_t}"
    else:
        compress = f"{REVIEW_PREFIX}{review}{REVIEW_SUFFIX}"
        window = MOVIE_SENTIMENT_TASK
    return compress + press.delimiter_token + window


def _trim_finch_window_from_cache(cache: DynamicCache, window_size: int) -> None:
    if window_size <= 0:
        return
    if hasattr(cache, "layers") and cache.layers:
        for layer in cache.layers:
            layer.keys = layer.keys[:, :, :-window_size, :].contiguous()
            layer.values = layer.values[:, :, :-window_size, :].contiguous()
    elif hasattr(cache, "key_cache"):
        cache.key_cache = [k[:, :, :-window_size, :].contiguous() for k in cache.key_cache]
        cache.value_cache = [v[:, :, :-window_size, :].contiguous() for v in cache.value_cache]


def build_pipeline(
    model_name: str,
    device_map: str | None,
    device: str | None,
    torch_dtype: torch.dtype,
    use_flash_attn: bool,
):
    model_kw: dict[str, Any] = {}
    if use_flash_attn:
        model_kw["attn_implementation"] = "flash_attention_2"

    attempts: list[dict] = []
    if model_kw:
        attempts.append(model_kw)
    attempts.append({})

    last_err: Exception | None = None
    for mkw in attempts:
        try:
            pipe = pipeline(
                "kv-press-text-generation",
                model=model_name,
                device_map=device_map if device_map else None,
                device=device if device_map is None and device else None,
                torch_dtype=torch_dtype,
                model_kwargs=mkw or None,
            )
            return pipe
        except Exception as e:
            last_err = e
            logger.warning("Pipeline init failed with model_kwargs=%s: %s", mkw, e)
    raise RuntimeError(f"Failed to create kv-press-text-generation pipeline: {last_err}")


@contextmanager
def press_context(pipe, run_tag: str, compression_ratio: float, finch_press: FinchPress | None):
    if run_tag == "ea":
        press = KeyRerotationPress(ExpectedAttentionPress(compression_ratio=compression_ratio))
        with torch.inference_mode():
            with press(pipe.model):
                yield press
        return
    if run_tag == "kvzip":
        if KVzipPress is None:
            raise RuntimeError("KVzipPress is not available in this kvpress install.")
        press = KVzipPress(compression_ratio=compression_ratio)
        with torch.inference_mode():
            with press(pipe.model):
                yield press
        return
    if run_tag in ("finch_no_cpt", "finch_with_cpt"):
        assert finch_press is not None
        apply_finch_press_hook_fix()
        with torch.inference_mode():
            with finch_press(pipe.model):
                yield finch_press
        return
    raise ValueError(f"Unknown run_tag={run_tag!r}")


def generate_row_cache_run_tag(
    pipe,
    *,
    run_tag: str,
    context_for_preprocess: str,
    compression_ratio: float,
    finch_press: FinchPress | None,
) -> DynamicCache:
    inputs = pipe.preprocess(
        context=context_for_preprocess,
        questions=[""],
        answer_prefix="Answer: ",
        max_context_length=128_000,
    )
    if run_tag in ("finch_no_cpt", "finch_with_cpt") and finch_press is not None:
        input_ids = inputs["input_ids"]
        seq_len = int(input_ids.shape[1])
        finch_press.window_size = max(1, min(int(FINCH_WINDOW_DEFAULT), seq_len))
    cache = DynamicCache()
    with press_context(pipe, run_tag, compression_ratio, finch_press) as press:
        _ = pipe._forward(inputs, press=press, cache=cache)
    if run_tag in ("finch_no_cpt", "finch_with_cpt") and finch_press is not None:
        ws = getattr(finch_press, "window_size", None)
        if ws is not None and int(ws) > 0:
            _trim_finch_window_from_cache(cache, int(ws))
    dynamic_cache_to_cpu_inplace(cache)
    return cache


def generate_text_row_cache_legacy(
    pipe,
    context: str,
    compression_ratio: float,
    press_name: str,
) -> DynamicCache:
    if press_name not in PRESS_LEGACY:
        raise ValueError(f"Unknown press_name={press_name!r}; choose from {list(PRESS_LEGACY.keys())}")
    press = KeyRerotationPress(PRESS_LEGACY[press_name](compression_ratio=compression_ratio))
    inputs = pipe.preprocess(
        context=context,
        questions=[""],
        answer_prefix="Answer: ",
        max_context_length=128_000,
    )
    cache = DynamicCache()
    with torch.inference_mode():
        with press(pipe.model):
            _ = pipe._forward(inputs, press=press, cache=cache)
    dynamic_cache_to_cpu_inplace(cache)
    return cache


def _default_csv_path() -> Path:
    repo = Path(__file__).resolve().parents[2]
    for p in (Path.cwd() / "reviews_1000.csv", repo / "reviews_1000.csv"):
        if p.is_file():
            return p
    return repo / "reviews_1000.csv"


def prepare_movie_caches_run_tag(
    pipe,
    df: pd.DataFrame,
    cache_root: Path,
    model_name: str,
    compression_ratio: float,
    run_tag: str,
    text_col: str,
    cpt_col: str,
    id_col: str | None,
    tok,
) -> dict[str, int]:
    model_safe = _sanitize_model_dir(model_name)
    tag = compression_tag(compression_ratio)
    save_dir = cache_root / model_safe / run_tag / f"comp{tag}"
    save_dir.mkdir(parents=True, exist_ok=True)
    errors: dict[str, str] = {}
    n_written = 0
    n_skipped = 0
    n_failed = 0

    finch_press: FinchPress | None = None
    if run_tag in ("finch_no_cpt", "finch_with_cpt"):
        finch_press = FinchPress(compression_ratio=float(compression_ratio))
        finch_press.update_model_and_tokenizer(pipe.model, tok)

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"{run_tag} CR={compression_ratio}"):
        review = str(row[text_col]).strip()
        rid = str(row[id_col]) if id_col and id_col in row.index and pd.notna(row.get(id_col)) else str(i)

        if run_tag in ("ea", "kvzip"):
            body = build_ea_kvzip_body(review)
            ctx = build_context_for_cache(body, max_chars=0)
        elif run_tag == "finch_no_cpt":
            assert finch_press is not None
            ctx = build_finch_prefill_string(finch_press, review, with_cpt=False, cpt=None)
        else:
            assert finch_press is not None
            cpt_cell = row.get(cpt_col) if cpt_col in row.index else None
            cpt_str = str(cpt_cell).strip() if cpt_cell is not None and pd.notna(cpt_cell) else ""
            ctx = build_finch_prefill_string(finch_press, review, with_cpt=True, cpt=cpt_str or None)

        stable = f"{run_tag}|{compression_ratio}|{rid}|{i}|{ctx}"
        h = hash_path(stable)
        out_file = save_dir / f"cache_entry_{h}.pt"
        if out_file.is_file():
            n_skipped += 1
            continue
        try:
            cache = generate_row_cache_run_tag(
                pipe,
                run_tag=run_tag,
                context_for_preprocess=ctx,
                compression_ratio=float(compression_ratio),
                finch_press=finch_press,
            )
            save_cache(cache, str(out_file))
            del cache
            torch.cuda.empty_cache()
            n_written += 1
        except Exception as e:
            logger.warning("Row %s (id=%s): %s", i, rid, e)
            errors[str(out_file)] = repr(e)
            n_failed += 1

    write_errors_json(str(save_dir), errors)
    return {"written": n_written, "skipped": n_skipped, "failed": n_failed, "errors": len(errors)}


def prepare_movie_caches_legacy(
    pipe,
    df: pd.DataFrame,
    cache_root: Path,
    model_name: str,
    compression_ratio: float,
    press_name: str,
    text_col: str,
    use_cpt: bool,
    cpt_col: str,
    id_col: str | None,
) -> dict[str, int]:
    tag = compression_tag(compression_ratio)
    save_dir = cache_root / model_name / f"comp{tag}"
    save_dir.mkdir(parents=True, exist_ok=True)
    errors: dict[str, str] = {}
    n_written = 0
    n_skipped = 0
    n_failed = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"movie KV pregen CR={compression_ratio}"):
        raw = build_legacy_body(row, text_col, use_cpt, cpt_col)
        ctx = build_context_for_cache(raw, max_chars=0)
        rid = str(row[id_col]) if id_col and id_col in row.index and pd.notna(row.get(id_col)) else str(i)
        stable = f"{rid}|{i}|{ctx}"
        h = hash_path(stable)
        out_file = save_dir / f"cache_entry_{h}.pt"
        if out_file.is_file():
            n_skipped += 1
            continue
        try:
            cache = generate_text_row_cache_legacy(pipe, ctx, compression_ratio, press_name)
            save_cache(cache, str(out_file))
            del cache
            torch.cuda.empty_cache()
            n_written += 1
        except Exception as e:
            logger.warning("Row %s (id=%s): %s", i, rid, e)
            errors[str(out_file)] = repr(e)
            n_failed += 1

    write_errors_json(str(save_dir), errors)
    return {"written": n_written, "skipped": n_skipped, "failed": n_failed, "errors": len(errors)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=str, default=None, help="Path to reviews CSV.")
    p.add_argument("--cache-dir", type=str, required=True, help="Root directory for saved caches.")
    p.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Hugging Face model id for kv-press-text-generation.",
    )
    p.add_argument(
        "--compression-ratio",
        type=float,
        nargs="+",
        default=(0.0, 0.5, 0.8),
        help="One or more compression ratios.",
    )
    p.add_argument(
        "--run-tags",
        type=str,
        nargs="+",
        default=None,
        choices=list(RUN_TAGS),
        metavar="TAG",
        help=(
            f"Run-tag mode: one or more of {list(RUN_TAGS)}. "
            "Recommended: pass a single tag per GPU session. "
            "Omit to use legacy --press / --use-cpt mode."
        ),
    )
    p.add_argument("--press", type=str, default="expected_attention", choices=sorted(PRESS_LEGACY))
    p.add_argument("--text-column", type=str, default="reviewtext")
    p.add_argument("--id-column", type=str, default="reviewid")
    p.add_argument("--use-cpt", action="store_true")
    p.add_argument(
        "--run-both-cpt-modes",
        action="store_true",
        help="Legacy: run each ratio with and without CPT.",
    )
    p.add_argument("--checkpoint-csv", type=str, default=None)
    p.add_argument("--cpt-column", type=str, default="cpt")
    p.add_argument("--max-rows", type=int, default=None, help="Keep only the first N rows after --tail.")
    p.add_argument("--tail", type=int, default=None, help="Keep only the last N rows of the CSV.")
    p.add_argument("--device-map", type=str, default="auto")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--no-flash-attn", action="store_true")
    p.add_argument(
        "--kvzip-patch",
        action="store_true",
        help="Apply QuantizedLayer/BasePress/KVzip hook patch (recommended for Qwen + transformers 5.x).",
    )
    return p.parse_args()


def main() -> None:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = tok

    args = parse_args()
    csv_path = Path(args.csv) if args.csv else _default_csv_path()
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    device_map = args.device_map.strip() or None

    df = pd.read_csv(csv_path, dtype=str)
    if args.text_column not in df.columns:
        raise SystemExit(f"Column {args.text_column!r} missing; columns={list(df.columns)}")

    if args.tail is not None and int(args.tail) > 0:
        df = df.tail(int(args.tail)).copy()
    if args.max_rows is not None:
        df = df.head(int(args.max_rows)).copy()

    id_col = args.id_column if args.id_column in df.columns else None
    if id_col is None:
        logger.info("Column %r not found; cache keys use dataframe index only.", args.id_column)

    run_tag_mode = args.run_tags is not None and len(args.run_tags) > 0

    if run_tag_mode:
        if args.use_cpt or args.run_both_cpt_modes:
            logger.warning("Ignoring --use-cpt / --run-both-cpt-modes in --run-tags mode.")
        if "finch_with_cpt" in args.run_tags and args.cpt_column not in df.columns:
            raise SystemExit(f"finch_with_cpt requires column {args.cpt_column!r}")
        if "kvzip" in args.run_tags and KVzipPress is None:
            raise SystemExit("KVzipPress not importable from kvpress.")
    else:
        if args.use_cpt and not args.run_both_cpt_modes and args.cpt_column not in df.columns:
            raise SystemExit(f"--use-cpt requires column {args.cpt_column!r}")

    logger.info("Loading model %s (%d rows from %s)…", args.model, len(df), csv_path)
    pipe = build_pipeline(
        args.model,
        device_map=device_map,
        device=args.device,
        torch_dtype=dtype,
        use_flash_attn=not args.no_flash_attn,
    )
    pipe.model.eval()
    apply_kvpress_patches_text_only()
    # BasePress hook patch is used by EA (KeyRerotation+EA) and KVzip on Qwen + transformers 5.x.
    if run_tag_mode and args.kvzip_patch and args.run_tags:
        if any(t in ("ea", "kvzip") for t in args.run_tags):
            apply_kvzip_quantized_layer_patch()
    if run_tag_mode and any(t in args.run_tags for t in ("finch_no_cpt", "finch_with_cpt")):
        apply_finch_press_hook_fix()

    _tok = getattr(pipe, "tokenizer", None)
    if _tok is None:
        raise SystemExit("Pipeline has no tokenizer")

    ck_path = Path(args.checkpoint_csv).resolve() if args.checkpoint_csv else None
    done_run: set[tuple[str, str]] = set()
    done_legacy: set[tuple[str, str]] = set()
    if ck_path and ck_path.is_file():
        with ck_path.open("r", encoding="utf-8", newline="") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                if str(row.get("status", "")).strip().lower() != "ok":
                    continue
                cr_raw = str(row.get("compression_ratio", "")).strip()
                try:
                    cr_norm = str(float(cr_raw))
                except ValueError:
                    continue
                rt = str(row.get("run_tag", "")).strip()
                if rt:
                    done_run.add((rt, cr_norm))
                else:
                    done_legacy.add((cr_norm, str(row.get("use_cpt", "")).strip().lower()))

    def append_checkpoint(
        *,
        compression_ratio: float,
        stats: dict[str, int],
        status: str,
        err_msg: str = "",
        run_tag: str | None = None,
        use_cpt: bool | None = None,
        press_legacy: str | None = None,
    ) -> None:
        if ck_path is None:
            return
        ck_path.parent.mkdir(parents=True, exist_ok=True)
        newfile = not ck_path.is_file()
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "run_tag": run_tag or "",
            "press": press_legacy or args.press,
            "compression_ratio": str(compression_ratio),
            "use_cpt": str(bool(use_cpt)).lower() if use_cpt is not None else "",
            "n_rows": str(len(df)),
            "written": str(stats.get("written", 0)),
            "skipped": str(stats.get("skipped", 0)),
            "failed": str(stats.get("failed", 0)),
            "status": status,
            "error": err_msg[:2000],
            "cache_dir": str(Path(args.cache_dir).resolve()),
        }
        fieldnames = list(row.keys())
        with ck_path.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if newfile:
                w.writeheader()
            w.writerow(row)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass

    if run_tag_mode:
        for run_tag in args.run_tags:
            for cr in args.compression_ratio:
                key = (run_tag, str(float(cr)))
                if key in done_run:
                    logger.info("Skip finished (checkpoint): run_tag=%s ratio=%s", run_tag, cr)
                    continue
                logger.info("run_tag=%s compression_ratio=%s → comp%s", run_tag, cr, compression_tag(float(cr)))
                stats: dict[str, int] = {}
                err_msg = ""
                status = "ok"
                try:
                    stats = prepare_movie_caches_run_tag(
                        pipe,
                        df,
                        Path(args.cache_dir),
                        args.model,
                        float(cr),
                        run_tag,
                        args.text_column,
                        args.cpt_column,
                        id_col,
                        _tok,
                    )
                except Exception as e:
                    status = "error"
                    err_msg = repr(e)
                    logger.exception("prepare_movie_caches_run_tag failed: %s", e)
                    stats = {"written": 0, "skipped": 0, "failed": 0, "errors": 0}
                elif stats.get("failed", 0) > 0:
                    status = "partial"
                    err_msg = f"{stats['failed']} row(s) failed; see ERRORS.json in comp dir"
                    logger.warning(
                        "run_tag=%s ratio=%s finished with %s failures",
                        run_tag,
                        cr,
                        stats["failed"],
                    )
                append_checkpoint(
                    compression_ratio=float(cr),
                    stats=stats,
                    status=status,
                    err_msg=err_msg,
                    run_tag=run_tag,
                )
                if status == "error":
                    raise SystemExit(f"KV pregen failed run_tag={run_tag} ratio={cr}: {err_msg}")
    else:

        def _cpt_modes() -> list[bool]:
            if args.run_both_cpt_modes:
                return [False, True]
            return [bool(args.use_cpt)]

        for use_cpt in _cpt_modes():
            if use_cpt and args.cpt_column not in df.columns:
                logger.warning("Skipping use_cpt=True: column %r missing", args.cpt_column)
                continue
            for cr in args.compression_ratio:
                key = (str(float(cr)), str(use_cpt).lower())
                if key in done_legacy:
                    logger.info("Skip finished (checkpoint): ratio=%s use_cpt=%s", cr, use_cpt)
                    continue
                logger.info("Compression ratio %s use_cpt=%s", cr, use_cpt)
                stats = {}
                err_msg = ""
                status = "ok"
                try:
                    stats = prepare_movie_caches_legacy(
                        pipe,
                        df,
                        Path(args.cache_dir),
                        args.model,
                        float(cr),
                        args.press,
                        args.text_column,
                        use_cpt,
                        args.cpt_column,
                        id_col,
                    )
                except Exception as e:
                    status = "error"
                    err_msg = repr(e)
                    logger.exception("prepare_movie_caches_legacy failed: %s", e)
                    stats = {"written": 0, "skipped": 0, "failed": 0, "errors": 0}
                elif stats.get("failed", 0) > 0:
                    status = "partial"
                    err_msg = f"{stats['failed']} row(s) failed; see ERRORS.json in comp dir"
                append_checkpoint(
                    compression_ratio=float(cr),
                    stats=stats,
                    status=status,
                    err_msg=err_msg,
                    use_cpt=use_cpt,
                    press_legacy=args.press,
                )
                if status == "error":
                    raise SystemExit(f"KV pregen failed ratio={cr} use_cpt={use_cpt}: {err_msg}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
