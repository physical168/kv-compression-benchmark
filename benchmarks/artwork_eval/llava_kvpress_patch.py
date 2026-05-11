"""Compatibility patches so kvpress works with LlavaNext + transformers 5.x.

Ported from CompressionExperiments ``experiment_manager/src/engine.py``:
``DynamicCache`` proxy, ``BasePress.forward_hook`` cache_position fallback,
and ``_patch_llava_for_kvpress`` (language_model alias, config fields, input_ids pre-hook).
"""

from __future__ import annotations

import logging

import torch
from transformers import DynamicCache

try:
    from kvpress import BasePress
except ImportError:
    from kvpress.presses.base_press import BasePress

logger = logging.getLogger(__name__)


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
