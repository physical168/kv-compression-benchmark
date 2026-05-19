"""CLI: pre-generate multimodal KV caches (LLaVA-NeXT + kvpress) for image files.

Depends on: ``transformers``, ``torch``, ``kvpress``, ``Pillow``, and (optional)
``flash-attn`` for ``attn_implementation="flash_attention_2"`` — same stack as
CompressionExperiments / ReasonDB vision service snippets you referenced.

Run from repo root (example)::

    python benchmarks/kv_cache_pregen/pregenerate_kv_caches.py \\
        --image-dir artworks_files/artworks_files \\
        --cache-dir ./cache_pregen \\
        --model llava-hf/llama3-llava-next-8b-hf \\
        --compression-ratio 0.5 \\
        --max-images 10
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import sys
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import DynamicCache, pipeline  # type: ignore

# kvpress registers the ``kv-press-text-generation`` pipeline task.
import kvpress  # noqa: F401
from kvpress import ExpectedAttentionPress, KeyRerotationPress  # type: ignore

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

_ARTWORK_EVAL = _HERE.parent / "artwork_eval"
sys.path.insert(0, str(_ARTWORK_EVAL))
from llava_kvpress_patch import apply_kvpress_compatibility_patches  # noqa: E402

from cache_io import (  # noqa: E402
    compression_tag,
    dynamic_cache_to_cpu_inplace,
    hash_path,
    save_cache,
    write_errors_json,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 250_000_000

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
IMAGE_MAX_PIXELS_DEFAULT = 1400 * 1400

PRESS = {
    "expected_attention": lambda cr: ExpectedAttentionPress(compression_ratio=cr),
}


def _list_images(image_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(image_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            out.append(p)
    return out


def deserialize_image(
    path: str | Path,
    max_pixels: int = IMAGE_MAX_PIXELS_DEFAULT,
) -> Image.Image:
    image = Image.open(path)
    image.load()
    image = image.convert("RGB")
    w, h = image.size
    if w * h > max_pixels:
        ratio = math.sqrt(max_pixels / (w * h))
        image.thumbnail((int(w * ratio), int(h * ratio)))
    return image


def build_pipeline(
    model_name: str,
    device_map: str | None,
    device: str | None,
    torch_dtype: torch.dtype,
    use_flash_attn: bool,
):
    model_kw = {}
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


def generate_cache_for_image(
    pipe,
    image: Image.Image,
    compression_ratio: float,
    press_name: str,
) -> DynamicCache:
    if press_name not in PRESS:
        raise ValueError(f"Unknown press_name={press_name!r}; choose from {list(PRESS)}")

    context = " "
    answer_prefix = "Answer: "
    press = KeyRerotationPress(PRESS[press_name](compression_ratio=compression_ratio))

    inputs = pipe.preprocess(
        context=context,
        questions=[""],
        answer_prefix=answer_prefix,
        max_context_length=128_000,
        image=image,
    )
    cache = DynamicCache()
    with torch.inference_mode():
        with press(pipe.model) if press is not None else contextlib.nullcontext():
            _ = pipe._forward(inputs, press=press, cache=cache)
    dynamic_cache_to_cpu_inplace(cache)
    return cache


def prepare_caches_for_paths(
    pipe,
    image_paths: Sequence[str | Path],
    cache_root: Path,
    model_name: str,
    compression_ratio: float,
    press_name: str,
) -> None:
    tag = compression_tag(compression_ratio)
    save_dir = cache_root / model_name / f"comp{tag}"
    save_dir.mkdir(parents=True, exist_ok=True)
    errors: dict[str, str] = {}

    for i, image_path in enumerate(
        tqdm(list(image_paths), desc=f"KV pregen CR={compression_ratio}")
    ):
        p = Path(image_path)
        key = str(p.resolve()) if p.exists() else str(p)
        h = hash_path(key)
        out_file = save_dir / f"cache_entry_{h}.pt"
        if out_file.is_file():
            continue
        try:
            img = deserialize_image(p)
            cache = generate_cache_for_image(pipe, img, compression_ratio, press_name)
            save_cache(cache, str(out_file))
            del cache
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning("Failed [%s] %s: %s", i, p, e)
            errors[str(out_file)] = repr(e)

    write_errors_json(str(save_dir), errors)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image-dir", type=str, default=None, help="Directory of images to index.")
    p.add_argument(
        "--image-list",
        type=str,
        default=None,
        help="Text file with one image path per line (overrides --image-dir).",
    )
    p.add_argument("--cache-dir", type=str, required=True, help="Root directory for saved caches.")
    p.add_argument(
        "--model",
        type=str,
        default="llava-hf/llama3-llava-next-8b-hf",
        help="Hugging Face model id (LLaVA-NeXT family).",
    )
    p.add_argument(
        "--compression-ratio",
        type=float,
        nargs="+",
        default=(0.0, 0.5),
        help="One or more compression ratios (kvpress ExpectedAttention).",
    )
    p.add_argument("--press", type=str, default="expected_attention", choices=sorted(PRESS))
    p.add_argument("--max-images", type=int, default=None, help="Cap number of images processed.")
    p.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help='Transformers device_map (e.g. "auto"). Use "" to disable and set --device.',
    )
    p.add_argument("--device", type=str, default=None, help='Single device, e.g. "cuda:0" (if device_map unset).')
    p.add_argument("--bf16", action="store_true", help="Load weights in bfloat16 (recommended on Ampere+).")
    p.add_argument(
        "--no-flash-attn",
        action="store_true",
        help="Do not pass attn_implementation=flash_attention_2 to the model.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    device_map = args.device_map.strip() or None

    if args.image_list:
        paths = [line.strip() for line in Path(args.image_list).read_text(encoding="utf-8").splitlines() if line.strip()]
    elif args.image_dir:
        paths = [str(x) for x in _list_images(Path(args.image_dir))]
    else:
        raise SystemExit("Provide --image-dir or --image-list")

    if args.max_images is not None:
        paths = paths[: int(args.max_images)]

    if not paths:
        raise SystemExit("No images found to process.")

    logger.info("Loading model %s …", args.model)
    pipe = build_pipeline(
        args.model,
        device_map=device_map,
        device=args.device,
        torch_dtype=dtype,
        use_flash_attn=not args.no_flash_attn,
    )
    pipe.model.eval()
    apply_kvpress_compatibility_patches(pipe.model)

    for cr in args.compression_ratio:
        logger.info("Compression ratio %s → under comp%s", cr, compression_tag(cr))
        prepare_caches_for_paths(
            pipe,
            paths,
            Path(args.cache_dir),
            args.model,
            float(cr),
            args.press,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
