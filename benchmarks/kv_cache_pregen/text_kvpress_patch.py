"""Text-only kvpress compatibility patches (Qwen, Llama, …).

Kept under ``kv_cache_pregen`` so movie pregen does not depend on
``artwork_eval/llava_kvpress_patch.py`` (avoids stale Kaggle clones).
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
from pathlib import Path

import torch
from transformers import DynamicCache

logger = logging.getLogger(__name__)


def _resolve_base_press():
    try:
        from kvpress import BasePress as BP  # type: ignore[attr-defined]

        return BP
    except ImportError:
        pass
    try:
        from kvpress.presses.base_press import BasePress as BP  # type: ignore[import-not-found]

        return BP
    except ImportError:
        pass

    try:
        from kvpress import KVzipPress
    except ImportError as e:
        raise ImportError(
            "kvpress is present but BasePress cannot be resolved and KVzipPress is not importable."
        ) from e

    kvzip_file = Path(inspect.getfile(KVzipPress)).resolve()
    for candidate in (
        kvzip_file.parent / "base_press.py",
        kvzip_file.parent.parent / "presses" / "base_press.py",
    ):
        if not candidate.is_file():
            continue
        spec = importlib.util.spec_from_file_location("_kvpress_base_press_dyn", candidate)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "BasePress"):
            return mod.BasePress

    for cls in KVzipPress.__mro__:
        if cls.__name__ == "BasePress":
            return cls

    raise ImportError("Could not locate BasePress in this kvpress install.")


class _CacheListProxy:
    def __init__(self, cache: DynamicCache, attr: str) -> None:
        self._cache = cache
        self._attr = attr

    def __getitem__(self, idx: int):
        return getattr(self._cache.layers[idx], self._attr)

    def __setitem__(self, idx: int, value) -> None:
        setattr(self._cache.layers[idx], self._attr, value)

    def __len__(self) -> int:
        return len(self._cache.layers)

    def __iter__(self):
        return (getattr(layer, self._attr) for layer in self._cache.layers)


def _patch_dynamic_cache_for_kvpress() -> None:
    if hasattr(DynamicCache, "_kvpress_patched"):
        return

    def _key_cache_getter(self):
        return _CacheListProxy(self, "keys")

    def _key_cache_setter(self, value):
        for i, t in enumerate(value):
            self.layers[i].keys = t

    def _value_cache_getter(self):
        return _CacheListProxy(self, "values")

    def _value_cache_setter(self, value):
        for i, t in enumerate(value):
            self.layers[i].values = t

    DynamicCache.key_cache = property(_key_cache_getter, _key_cache_setter)
    DynamicCache.value_cache = property(_value_cache_getter, _value_cache_setter)
    DynamicCache._kvpress_patched = True


def _patch_base_press_for_transformers5() -> None:
    BasePress = _resolve_base_press()
    _orig = BasePress.forward_hook

    def _forward_hook(self, module, input, kwargs, output):
        if "cache_position" not in kwargs:
            hidden_states = kwargs["hidden_states"]
            q_len = hidden_states.shape[1]
            cache = kwargs.get("past_key_value") or kwargs.get("past_key_values")
            if cache is not None and cache.get_seq_length() > q_len:
                return output
            kwargs = dict(kwargs)
            kwargs["cache_position"] = torch.tensor([q_len - 1])
        return _orig(self, module, input, kwargs, output)

    BasePress.forward_hook = _forward_hook


def apply_kvpress_patches_text_only() -> None:
    """DynamicCache + BasePress patches for text-only ``kv-press-text-generation``."""
    _patch_dynamic_cache_for_kvpress()
    _patch_base_press_for_transformers5()
