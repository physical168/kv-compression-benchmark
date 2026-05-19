#!/usr/bin/env python3
"""Batch movie-review KV cache pregen via ``KvTextQaModelWrapper.prepare_caches``.

Maps session aliases to ``press_name`` in ``kv_cache_text_qa_server_new``:
  ea -> expected_attention
  kvzip -> kvzip
  finch_no_cpt -> finch (movie preformatted context)
  finch_with_cpt -> finch-cachenotes
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

_BUNDLE = Path(__file__).resolve().parent
sys.path.insert(0, str(_BUNDLE))
os.chdir(_BUNDLE)

from kv_cache_text_qa_server_new import CPT_PATH, KvTextQaModelWrapper  # noqa: E402
from text_kvpress_patch import apply_kvpress_patches_text_only  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PRESS_ALIASES = {
    "ea": "expected_attention",
    "kvzip": "kvzip",
    "finch_no_cpt": "finch",
    "finch_with_cpt": "finch-cachenotes",
}

MOVIE_SENTIMENT_TASK = (
    "Classify the movie review sentiment. "
    "Answer with one word only: positive or negative."
)
REVIEW_PREFIX = "Review: "
REVIEW_SUFFIX = "\n"


def build_ea_kvzip_body(review: str) -> str:
    review = str(review).strip()
    return f"{review}\n\n{MOVIE_SENTIMENT_TASK}"


def build_finch_no_cpt_body(review: str, delimiter: str) -> str:
    review = str(review).strip()
    compress = f"{REVIEW_PREFIX}{review}{REVIEW_SUFFIX}"
    window = MOVIE_SENTIMENT_TASK
    return compress + delimiter + window


def build_finch_with_cpt_body(review: str, delimiter: str, cpt: str) -> str:
    review = str(review).strip()
    cpt_t = str(cpt).strip()
    compress = f"{REVIEW_PREFIX}{review}{REVIEW_SUFFIX}{MOVIE_SENTIMENT_TASK}\n"
    window = f"CPT:\n{cpt_t}"
    return compress + delimiter + window


def to_compression_tag(compression_ratio: float) -> str:
    return str(compression_ratio).replace(".", "_") if compression_ratio != 0.0 else "0"


def resolve_press_name(alias: str) -> str:
    key = alias.strip().lower()
    if key in PRESS_ALIASES:
        return PRESS_ALIASES[key]
    if key in PRESS_ALIASES.values():
        return key
    raise ValueError(f"Unknown press {alias!r}; use one of {list(PRESS_ALIASES)}")


def build_texts_for_press(df: pd.DataFrame, text_col: str, press_name: str, wrapper) -> list[str]:
    if press_name in ("expected_attention", "kvzip"):
        return [build_ea_kvzip_body(row[text_col]) for _, row in df.iterrows()]

    if press_name == "finch-cachenotes":
        return [str(row[text_col]).strip() for _, row in df.iterrows()]

    if press_name == "finch":
        sample_press = next(p for cr, p in wrapper.presses.items() if p is not None)
        delim = sample_press.delimiter_token
        return [build_finch_no_cpt_body(row[text_col], delim) for _, row in df.iterrows()]

    raise ValueError(f"Unsupported press_name={press_name!r}")


def configure_finch_env(press_name: str) -> None:
    if press_name == "finch":
        os.environ["MOVIE_FINCH_PREFORMATTED"] = "1"
        os.environ["MOVIE_FINCH_WINDOW_TEXT"] = MOVIE_SENTIMENT_TASK
    else:
        os.environ.pop("MOVIE_FINCH_PREFORMATTED", None)
        os.environ.pop("MOVIE_FINCH_WINDOW_TEXT", None)


def apply_kvzip_quantized_layer_patch() -> None:
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
    logger.info("Applied KVzip / BasePress QuantizedLayer patch.")


def apply_finch_press_hook_fix() -> None:
    from contextlib import contextmanager

    from kvpress import FinchPress  # type: ignore

    if getattr(FinchPress, "_hook_fix_applied", False):
        return

    @contextmanager
    def __finch_call(self, model):
        if self.delimiter_token_id is None:
            raise ValueError("Finch delimiter_token_id is unset.")
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


def count_batch_stats(save_dir: Path, n_rows: int) -> dict[str, int]:
    pts = list(save_dir.glob("cache_entry_*.pt"))
    err_path = save_dir / "ERRORS.json"
    failed = 0
    if err_path.is_file():
        with err_path.open(encoding="utf-8") as f:
            errors = json.load(f)
        failed = len(errors) if isinstance(errors, dict) else 0
    n_pt = len(pts)
    skipped = max(0, n_rows - n_pt)
    written = max(0, n_pt - (n_rows - skipped - failed))
    return {"written": written, "skipped": skipped, "failed": failed, "errors": failed}


def load_done_checkpoint(ck_path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not ck_path.is_file():
        return done
    with ck_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("status", "")).strip().lower() != "ok":
                continue
            press = str(row.get("press_name", "")).strip()
            cr_raw = str(row.get("compression_ratio", "")).strip()
            try:
                cr_norm = str(float(cr_raw))
            except ValueError:
                continue
            if press:
                done.add((press, cr_norm))
    return done


def append_checkpoint(
    ck_path: Path,
    *,
    model: str,
    press_name: str,
    compression_ratio: float,
    n_rows: int,
    stats: dict[str, int],
    status: str,
    err_msg: str,
    cache_dir: str,
) -> None:
    ck_path.parent.mkdir(parents=True, exist_ok=True)
    newfile = not ck_path.is_file()
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "press_name": press_name,
        "compression_ratio": str(compression_ratio),
        "n_rows": str(n_rows),
        "written": str(stats.get("written", 0)),
        "skipped": str(stats.get("skipped", 0)),
        "failed": str(stats.get("failed", 0)),
        "status": status,
        "error": err_msg[:2000],
        "cache_dir": cache_dir,
    }
    with ck_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if newfile:
            w.writeheader()
        w.writerow(row)
        f.flush()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument(
        "--press-name",
        type=str,
        required=True,
        help=f"Press alias or name: {list(PRESS_ALIASES)}",
    )
    p.add_argument("--compression-ratio", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    p.add_argument("--tail", type=int, default=500)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--text-column", type=str, default="reviewtext")
    p.add_argument("--cpt-csv", type=str, default=None, help="Override CPT_PATH[movie] for finch-cachenotes")
    p.add_argument("--checkpoint-csv", type=str, default=None)
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--kvzip-patch", action="store_true")
    p.add_argument("--column-name", type=str, default="reviewtext")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    press_name = resolve_press_name(args.press_name)

    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = tok

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    if args.cpt_csv:
        os.environ["MOVIE_CPT_CSV"] = str(Path(args.cpt_csv).resolve())
        CPT_PATH["movie"] = os.environ["MOVIE_CPT_CSV"]

    df = pd.read_csv(csv_path, dtype=str)
    if args.text_column not in df.columns:
        raise SystemExit(f"Missing column {args.text_column!r}; have {list(df.columns)}")
    if press_name == "finch-cachenotes" and "cpt" not in df.columns:
        raise SystemExit("finch-cachenotes requires a 'cpt' column in the CSV.")

    if args.tail and int(args.tail) > 0:
        df = df.tail(int(args.tail)).copy()
    if args.max_rows is not None:
        df = df.head(int(args.max_rows)).copy()

    ratios = [float(r) for r in args.compression_ratio]
    ck_path = Path(args.checkpoint_csv).resolve() if args.checkpoint_csv else None
    done = load_done_checkpoint(ck_path) if ck_path else set()

    configure_finch_env(press_name)
    apply_kvpress_patches_text_only()
    if args.kvzip_patch and press_name in ("expected_attention", "kvzip"):
        apply_kvzip_quantized_layer_patch()
    if press_name in ("finch", "finch-cachenotes"):
        apply_finch_press_hook_fix()

    logger.info("Loading model %s press=%s (%d rows)", args.model, press_name, len(df))
    wrapper = KvTextQaModelWrapper(
        args.model,
        args.device_id,
        compression_ratios=ratios,
        press_name=press_name,
    )
    texts = build_texts_for_press(df, args.text_column, press_name, wrapper)
    cache_dir = str(Path(args.cache_dir).resolve())

    for cr in ratios:
        key = (press_name, str(float(cr)))
        if key in done:
            logger.info("Skip checkpoint-complete batch: press=%s ratio=%s", press_name, cr)
            continue

        save_dir = Path(cache_dir) / args.model / press_name / f"comp{to_compression_tag(cr)}"
        before = len(list(save_dir.glob("cache_entry_*.pt"))) if save_dir.is_dir() else 0

        stats: dict[str, int] = {}
        status = "ok"
        err_msg = ""
        try:
            asyncio.run(
                wrapper.prepare_caches(
                    column_name=args.column_name,
                    texts=texts,
                    cache_dir=cache_dir,
                    compression_ratio=cr,
                )
            )
            stats = count_batch_stats(save_dir, len(df))
            after = len(list(save_dir.glob("cache_entry_*.pt")))
            stats["written"] = max(0, after - before)
            if stats.get("failed", 0) > 0:
                status = "partial"
                err_msg = f"{stats['failed']} row(s) failed; see {save_dir}/ERRORS.json"
        except Exception as e:
            status = "error"
            err_msg = repr(e)
            logger.exception("prepare_caches failed press=%s ratio=%s", press_name, cr)
            stats = {"written": 0, "skipped": 0, "failed": len(df), "errors": len(df)}

        if ck_path:
            append_checkpoint(
                ck_path,
                model=args.model,
                press_name=press_name,
                compression_ratio=cr,
                n_rows=len(df),
                stats=stats,
                status=status,
                err_msg=err_msg,
                cache_dir=cache_dir,
            )
        if status == "error":
            raise SystemExit(f"Batch failed press={press_name} ratio={cr}: {err_msg}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
