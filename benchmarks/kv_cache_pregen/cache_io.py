"""Save/load helpers for ``DynamicCache`` across transformers 4.x / 5.x layouts."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Iterator, Tuple

import torch
from transformers import DynamicCache  # type: ignore

logger = logging.getLogger(__name__)


def hash_path(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()


def compression_tag(compression_ratio: float) -> str:
    return str(compression_ratio).replace(".", "_") if compression_ratio != 0.0 else "0"


def iter_cache_layers(cache: DynamicCache) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(cache, "layers") and cache.layers:
        for layer in cache.layers:
            yield layer.keys, layer.values
    elif hasattr(cache, "_cache") and cache._cache and hasattr(cache._cache[0], "key_states"):
        for item in cache._cache:
            yield item.key_states, item.value_states
    else:
        yield from zip(cache.__dict__["key_cache"], cache.__dict__["value_cache"])


def dynamic_cache_to_cpu_inplace(cache: DynamicCache) -> None:
    if hasattr(cache, "layers") and cache.layers:
        for layer in cache.layers:
            layer.keys = layer.keys.detach().cpu()
            layer.values = layer.values.detach().cpu()
    elif hasattr(cache, "_cache") and cache._cache:
        for item in cache._cache:
            item.key_states = item.key_states.detach().cpu()
            item.value_states = item.value_states.detach().cpu()
    else:
        cache.key_cache = [k.detach().cpu() for k in cache.key_cache]
        cache.value_cache = [v.detach().cpu() for v in cache.value_cache]


def save_cache(cache: DynamicCache, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(cache, path)
    if not os.path.exists(path):
        raise RuntimeError(f"Cache file was not created: {path}")


def write_errors_json(save_dir: str, errors: dict[str, str]) -> None:
    os.makedirs(save_dir, exist_ok=True)
    err_path = os.path.join(save_dir, "ERRORS.json")
    with open(err_path, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2)
    logger.info("Wrote %s (%d entries)", err_path, len(errors))
