"""Compatibility patches so kvpress works with LlavaNext + transformers 5.x.

Ported from CompressionExperiments ``experiment_manager/src/engine.py``:
``DynamicCache`` proxy, ``BasePress.forward_hook`` cache_position fallback,
and ``_patch_llava_for_kvpress`` (language_model alias, config fields, input_ids pre-hook).
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
    """Resolve BasePress across kvpress layouts (PyPI, NVIDIA repo, CE submodule quirks)."""
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

    # Slim / partial installs: load base_press.py next to KVzipPress, or take BasePress from MRO.
    try:
        from kvpress import KVzipPress
    except ImportError as e:
        raise ImportError(
            "kvpress is present but BasePress cannot be resolved and KVzipPress is not importable. "
            "Reinstall kvpress from NVIDIA/kvpress (or CE submodule with full `presses/`)."
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
            logger.debug("Loaded BasePress from %s", candidate)
            return mod.BasePress

    for cls in KVzipPress.__mro__:
        if cls.__name__ == "BasePress":
            logger.debug("Resolved BasePress from KVzipPress.__mro__")
            return cls

    try:
        import kvpress as _kp

        _root = Path(inspect.getfile(_kp))
        if _root.name == "__init__.py":
            _root = _root.parent
        for p in sorted(_root.rglob("base_press.py")):
            spec = importlib.util.spec_from_file_location("_kvpress_base_press_rglob", p)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "BasePress"):
                logger.debug("Loaded BasePress from %s (package scan)", p)
                return mod.BasePress
    except Exception:
        pass

    raise ImportError(
        "Could not locate BasePress (tried kvpress import, kvpress.presses.base_press, "
        "base_press.py beside KVzipPress, KVzipPress MRO, and package tree scan)."
    )


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


def _patch_llava_for_kvpress(model) -> None:
    try:
        from transformers import LlavaNextForConditionalGeneration
    except ImportError:
        return

    if not isinstance(model, LlavaNextForConditionalGeneration):
        return
    if hasattr(model, "language_model"):
        return

    inner = model.model.language_model
    object.__setattr__(model, "language_model", inner)
    logger.debug("Patched LlavaNext.language_model → model.model.language_model")

    tc = model.config.text_config
    if not hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = tc.num_hidden_layers
    if not hasattr(model.config, "num_key_value_heads"):
        model.config.num_key_value_heads = tc.num_key_value_heads

    object.__setattr__(model.model, "layers", inner.layers)
    object.__setattr__(model.model, "rotary_emb", inner.rotary_emb)

    def _normalize_input_ids(module, args, kwargs):
        if args and "input_ids" not in kwargs and kwargs.get("inputs_embeds") is None:
            kwargs = dict(kwargs)
            kwargs["input_ids"] = args[0]
            args = args[1:]
        return args, kwargs

    model.model.register_forward_pre_hook(_normalize_input_ids, with_kwargs=True)


def apply_kvpress_compatibility_patches(model) -> None:
    """Apply global transformers/kvpress patches and Llava-specific hooks on ``model``."""
    _patch_dynamic_cache_for_kvpress()
    _patch_base_press_for_transformers5()
    # CE disables cuDNN for some CUDA setups where Conv2d (vision tower) fails to init cudnn.
    torch.backends.cudnn.enabled = False
    _patch_llava_for_kvpress(model)
