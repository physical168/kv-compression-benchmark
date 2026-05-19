import argparse
import json
import os
import yaml
import asyncio
import logging
from tqdm import tqdm
from typing import List

# Movie batch pregen (run_movie_kv_pregen.py) does not need the HTTP server.
_PREGEN_ONLY = os.environ.get("MOVIE_KV_PREGEN") == "1"
if _PREGEN_ONLY:

    class Resource:  # noqa: D101 — stub base for unused API classes
        pass

    Flask = Api = request = None  # type: ignore[misc, assignment]
    app = api = None
else:
    from flask import Flask, request
    from flask_restful import Api, Resource

    app = Flask(__name__)
    api = Api(app)


from reasondb.backends.kv_cache_base import KVCachingBackendBase
from reasondb.backends.text_qa import PORT_KV_TEXT_QA
import torch
from kvpress import ExpectedAttentionPress, KeyRerotationPress, KVzipPress, FinchPress

from transformers import DynamicCache, pipeline  # type: ignore
import contextlib

from reasondb.memory_footprint.memory_report import compute_memory_footprints, update_compressed_cache_footprint


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"

PRESS = {
    "expected_attention": lambda compression_ratio: ExpectedAttentionPress(
        compression_ratio=compression_ratio
    ),
    "kvzip": lambda compression_ratio: KVzipPress(compression_ratio=compression_ratio),
    "finch": lambda compression_ratio: FinchPress(compression_ratio=compression_ratio),
    "finch-cachenotes": lambda compression_ratio: FinchPress(compression_ratio=compression_ratio),
}

CPT_PATH = {
    "movie": os.environ.get(
        "MOVIE_CPT_CSV",
        "reasondb/evaluation/benchmarks/files/reviews_1000.csv",
    ),
    "rotowire": "reasondb/evaluation/benchmarks/files/reports.csv",
    "email": "reasondb/evaluation/benchmarks/files/emails_with_cpt.csv",
}

# Maps the text column name (used as column_name at runtime) to (task key, CSV text column)
# column_name at runtime is the full dotted name like "reviews.reviewtext"
CPT_COLUMN_MAP = {
    "reviews.reviewtext":  ("movie", "reviewtext"),
    "reviewtext":          ("movie", "reviewtext"),
    "reports.report":      ("rotowire", "report"),
    "report":              ("rotowire", "report"),
    "emails.text":         ("email", "text"),
    "text":                ("email", "text"),
}

class KvTextQaModelWrapper(KVCachingBackendBase):
    def __init__(
        self,
        model_name,
        device_id: int,
        compression_ratios=(0.0, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9),
        # compression_ratios=(0.5, 0.6, 0.8, 0.9),
        batch_sizes=None,
        press_name="expected_attention",
        gold_vanilla=False,
    ):
        self.device_id = device_id
        self.compression_ratios = compression_ratios
        batch_sizes = batch_sizes or (None,) * len(compression_ratios)
        self.compression_ratio_to_batch_size = {
            cr: bs for cr, bs in zip(compression_ratios, batch_sizes)
        }
        self.model_name = model_name
        self.press_name = press_name
        self.gold_vanilla = gold_vanilla
        self.init()

    def compute_text_qa_response(
        self,
        column_name: str,
        texts: List[str],
        questions: List[str],
        compression_ratio: float,
        boolean_question: bool,
        cache_dir: str,
    ):
        assert (
            compression_ratio in self.compression_ratios
        ), f"Compression ratio {compression_ratio} not in supported ratios: {self.compression_ratios}"
        return asyncio.run(
            self._run_kv_cache_text(
                column_name=column_name,
                texts=texts,
                all_questions=questions,
                compression_ratio=compression_ratio,
                boolean_question=boolean_question,
                cache_dir=cache_dir,
            )
        )

    def hash_text(self, text: str) -> str:
        """Generate a sha256hash for a given path."""
        import hashlib

        return hashlib.sha256(text.encode()).hexdigest()

    async def prepare_caches(
        self,
        column_name: str,
        texts: List[str],
        cache_dir: str,
        compression_ratio: float,
    ):
        # Skip cache preparation if gold_vanilla is enabled for CR=0 and 70B model
        if self.gold_vanilla and compression_ratio == 0.0 and "70B" in self.model_name:
            logger.info(
                "Skipping cache preparation for gold-vanilla mode (CR=0, 70B model)"
            )
            return

        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.pipe.tokenizer is not None, "Model tokenizer is not initialized."
        assert (
            compression_ratio in self.compression_ratios
        ), f"Compression ratio {compression_ratio} not in supported ratios: {self.compression_ratios}"

        # Check if caches exist and generate if not (current method: checked one by one)
        # Store mapping of row indices to cache files
        save_dir = f"{cache_dir}/{self.model_name}/{self.press_name}/comp{self.to_compression_tag(compression_ratio)}"
        os.makedirs(save_dir, exist_ok=True)

        errors = {}
        cache_filenames = []

        for i, text in tqdm(
            enumerate(texts),
            total=len(texts),
            desc=f"Preparing caches {compression_ratio}",
        ):
            hash_name = self.hash_text(text)
            cache_filename = f"{save_dir}/cache_entry_{hash_name}.pt"
            cache_filenames.append(cache_filename.split("/")[-1])

            if os.path.exists(cache_filename):
                continue

            try:
                await self._generate_cache_for_text(
                    text, cache_filename, compression_ratio, column_name
                )
            except Exception as e:
                logger.warning(
                    f"Error processing text {i} with hash {hash_name}: {str(e)}",
                )
                errors[cache_filename] = str(e)

        compute_memory_footprints(cache_dir, column_name, cache_filenames, model_name=self.model_name)
        update_compressed_cache_footprint(
            cache_path=cache_dir,
            compression_ratio=compression_ratio,
            cache_filenames=cache_filenames,
            model_name=self.model_name,
            column_name=column_name,
            press_name=self.press_name,
        )
        with open(
            f"{save_dir}/ERRORS.json",
            "w",
        ) as f:
            json.dump(errors, f, indent=4)

    async def prepare_caches_multi_cr(
        self,
        column_name: str,
        texts: List[str],
        cr_to_cache_dir: dict,
    ):
        """
        Like prepare_caches, but generates all compression ratios in a SINGLE prefill
        pass per text. Scores are computed once and top-k selection is applied per CR.

        Parameters
        ----------
        cr_to_cache_dir : dict[float, str]
            Mapping from compression ratio to its base cache directory.
            Each CR may have its own directory (e.g. one per method name).
        """
        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.pipe.tokenizer is not None, "Model tokenizer is not initialized."
        compression_ratios = list(cr_to_cache_dir.keys())
        for cr in compression_ratios:
            assert cr in self.compression_ratios, (
                f"Compression ratio {cr} not in supported ratios: {self.compression_ratios}"
            )

        # Set up per-CR save directories (one per CR, each under its own cache_dir)
        save_dirs: dict[float, str] = {}
        for cr, cache_dir in cr_to_cache_dir.items():
            save_dir = (
                f"{cache_dir}/{self.model_name}/{self.press_name}"
                f"/comp{self.to_compression_tag(cr)}"
            )
            os.makedirs(save_dir, exist_ok=True)
            save_dirs[cr] = save_dir

        errors: dict[str, str] = {}
        all_cache_filenames: dict[float, list[str]] = {cr: [] for cr in compression_ratios}

        for i, text in tqdm(
            enumerate(texts),
            total=len(texts),
            desc="Preparing multi-CR caches",
        ):
            hash_name = self.hash_text(text)

            # Determine which CRs are missing for this text
            cache_filenames_by_cr: dict[float, str] = {}
            missing_crs: list[float] = []
            for cr in compression_ratios:
                cache_filename = f"{save_dirs[cr]}/cache_entry_{hash_name}.pt"
                all_cache_filenames[cr].append(cache_filename.split("/")[-1])
                cache_filenames_by_cr[cr] = cache_filename
                if not os.path.exists(cache_filename):
                    missing_crs.append(cr)

            if not missing_crs:
                continue

            try:
                await self._generate_caches_for_text_multi_cr(
                    text_content=text,
                    cache_filenames_by_cr={cr: cache_filenames_by_cr[cr] for cr in missing_crs},
                    compression_ratios=missing_crs,
                    column_name=column_name,
                )
            except Exception as e:
                logger.warning(
                    f"Error processing text {i} with hash {hash_name}: {str(e)}",
                )
                errors[hash_name] = str(e)

        # Write error logs and memory footprints for each CR
        for cr in compression_ratios:
            save_dir = save_dirs[cr]
            cache_dir = cr_to_cache_dir[cr]
            with open(f"{save_dir}/ERRORS.json", "w") as f:
                json.dump(errors, f, indent=4)
            compute_memory_footprints(
                cache_dir, column_name, all_cache_filenames[cr], model_name=self.model_name
            )
            update_compressed_cache_footprint(
                cache_path=cache_dir,
                compression_ratio=cr,
                cache_filenames=all_cache_filenames[cr],
                model_name=self.model_name,
                column_name=column_name,
                press_name=self.press_name,
            )

    async def _generate_caches_for_text_multi_cr(
        self,
        text_content: str,
        cache_filenames_by_cr: dict,
        compression_ratios: list,
        column_name: str = None,
    ):
        """
        Run ONE prefill pass and save compressed caches for multiple CRs.

        Strategy:
          1. Prepare context (same logic as _generate_cache_for_text).
          2. Run prefill with a score-capturing hook — no compression applied,
             full uncompressed KV cache is kept in memory.
          3. For each CR: apply topk(n_kept) on the captured scores + key
             rerotation (when applicable), then save the compressed cache.

        This avoids N separate prefill passes for N compression ratios.
        """
        assert self.pipe is not None, "Model pipeline is not initialized."

        is_finch = self.press_name in ("finch", "finch-cachenotes")
        uses_rerotation = self.press_name == "expected_attention"  # kvzip: no rerotation

        try:
            answer_prefix = "Answer: "
            window_text = None  # set only for Finch

            # ---- 1. Prepare context (mirrors _generate_cache_for_text) ----
            if self.press_name == "finch":
                yaml_path = "queries_workloads/_workload.yaml"
                with open(yaml_path, "r") as f:
                    data = yaml.safe_load(f)
                query_list = data["queries"]
                queries_workload = (
                    "Pay attention to these examples of questions:\n"
                    + "\n".join(f"- {q}" for q in query_list)
                )
                sample_press = next(
                    p for cr, p in self.presses.items() if p is not None
                )
                context = text_content[: min(128000, len(text_content))]
                context_aware = context + sample_press.delimiter_token + queries_workload
                window_text = queries_workload

            elif self.press_name == "finch-cachenotes":
                import re
                import hashlib as _hashlib
                import pandas as _pd

                def _normalize(t):
                    t = t.strip()
                    t = re.sub(r"[ \t]+", " ", t)
                    t = re.sub(r"\n\s*\n", "\n\n", t)
                    t = t.replace("\r\n", "\n").replace("\r", "\n")
                    t = t.replace("\u200b", "")
                    return t

                assert column_name is not None
                assert column_name in CPT_COLUMN_MAP, (
                    f"Unknown column_name '{column_name}' for finch-cachenotes."
                )
                task_key, text_col = CPT_COLUMN_MAP[column_name]
                cpt_path = CPT_PATH[task_key]
                df = _pd.read_csv(cpt_path)
                norm_content = _normalize(text_content)
                text_hash = _hashlib.sha256(text_content.encode()).hexdigest()

                matching_row = None
                for _, row in df.iterrows():
                    row_text = str(row.get(text_col, ""))
                    if (
                        row_text == text_content
                        or _normalize(row_text) == norm_content
                        or _hashlib.sha256(row_text.encode()).hexdigest() == text_hash
                    ):
                        matching_row = row
                        break

                sample_press = next(
                    p for cr, p in self.presses.items() if p is not None
                )
                if matching_row is not None:
                    cpt = str(matching_row["cpt"])
                else:
                    logger.warning(
                        f"No matching CPT found for column '{column_name}' — using fallback."
                    )
                    cpt = "Your task is to answer questions based on the context."
                context = text_content[: min(128000, len(text_content))]
                context_aware = context + sample_press.delimiter_token + cpt
                window_text = cpt

            else:
                context_aware = text_content[: min(128000, len(text_content))]

            # ---- 2. Tokenize ----
            inputs = self.pipe.preprocess(
                context=context_aware,
                questions=[""],
                answer_prefix=answer_prefix,
                max_context_length=128000,
            )
            context_ids = inputs["context_ids"]
            actual_context_ids_length = context_ids.shape[1]

            # For Finch: compute token counts needed for per-CR n_kept calculation
            if is_finch and window_text is not None:
                window_tokens_count = len(
                    self.tokenizer.encode(window_text, add_special_tokens=False)
                )
                context_tokens_count = actual_context_ids_length - window_tokens_count - 1
            else:
                window_tokens_count = 0
                context_tokens_count = actual_context_ids_length

            # ---- 3. Build a score-capturing hook (only needed when compressing) ----
            has_nonzero_crs = any(cr > 0.0 for cr in compression_ratios)
            scoring_press = None
            if has_nonzero_crs:
                if is_finch:
                    scoring_press = FinchPress(compression_ratio=0.0)
                    scoring_press.update_model_and_tokenizer(self.pipe.model, self.pipe.tokenizer)
                elif uses_rerotation:
                    any_nonzero_cr = next(cr for cr in self.compression_ratios if cr > 0.0)
                    scoring_press = self.presses[any_nonzero_cr].press
                else:
                    any_nonzero_cr = next(cr for cr in self.compression_ratios if cr > 0.0)
                    scoring_press = self.presses[any_nonzero_cr]

            layer_scores: dict[int, torch.Tensor] = {}
            layer_modules: dict[int, object] = {}

            def capturing_forward_hook(module, input, kwargs, output):
                if scoring_press is None:
                    return output
                hidden_states = kwargs["hidden_states"]
                cache = kwargs.get("past_key_value") or kwargs.get("past_key_values")
                q_len = hidden_states.shape[1]
                # Only during prefill
                if kwargs["cache_position"][-1] > q_len:
                    return output
                keys = cache.key_cache[module.layer_idx]
                values = cache.value_cache[module.layer_idx]
                # output[1] is attention weights (None when output_attentions=False)
                scores = scoring_press.score(
                    module, hidden_states, keys, values, output[1], kwargs
                )
                layer_scores[module.layer_idx] = scores.detach().cpu()
                layer_modules[module.layer_idx] = module
                # Return output unchanged — keep full uncompressed cache
                return output

            # ---- 4. Run a single prefill ----
            first_device = next(self.pipe.model.parameters()).device
            context_ids = context_ids.to(first_device)
            full_cache = DynamicCache()

            # Set rotary embeddings (required by scoring methods)
            for layer in self.pipe.model.model.layers:
                layer.self_attn.rotary_emb = self.pipe.model.model.rotary_emb

            hooks = [
                layer.self_attn.register_forward_hook(
                    capturing_forward_hook, with_kwargs=True
                )
                for layer in self.pipe.model.model.layers
            ]
            embed_hook = None
            if is_finch and scoring_press is not None:
                embed_hook = self.pipe.model.model.embed_tokens.register_forward_hook(
                    scoring_press.embed_token_forward_hook
                )

            try:
                logger.info(
                    f"Single prefill for {actual_context_ids_length} tokens "
                    f"→ generating {len(compression_ratios)} cache(s)"
                )
                with torch.inference_mode():
                    self.pipe.model.model(
                        input_ids=context_ids,
                        past_key_values=full_cache,
                        use_cache=True,
                        output_attentions=False,
                    )
            finally:
                for hook in hooks:
                    hook.remove()
                if embed_hook is not None:
                    embed_hook.remove()

            # Capture Finch window_size (set by embed_token_hook during prefill)
            window_size = 0
            if is_finch:
                window_size = scoring_press.window_size
                assert window_size is not None and window_size > 0, (
                    "Finch window_size was not detected during prefill."
                )
                logger.info(f"Finch window_size = {window_size}")

            # ---- 5. Apply per-CR compression and save ----
            num_layers = len(full_cache.key_cache)

            for cr in compression_ratios:
                compressed_cache = DynamicCache()

                for layer_idx in range(num_layers):
                    full_keys = full_cache.key_cache[layer_idx].to(first_device)
                    full_values = full_cache.value_cache[layer_idx].to(first_device)

                    if cr == 0.0:
                        # No compression: just trim window tokens for Finch
                        k = full_keys[:, :, :-window_size, :] if window_size > 0 else full_keys.clone()
                        v = full_values[:, :, :-window_size, :] if window_size > 0 else full_values.clone()
                    else:
                        module = layer_modules[layer_idx]
                        head_dim = module.head_dim
                        scores = layer_scores[layer_idx].to(first_device)
                        total_len = full_keys.shape[2]

                        if is_finch:
                            # Keep n_kept_context context tokens + all window tokens
                            n_kept_context = int(context_tokens_count * (1 - cr))
                            n_kept_total = min(n_kept_context + window_size, total_len)
                        else:
                            n_kept_total = int(total_len * (1 - cr))

                        indices = scores.topk(n_kept_total, dim=-1).indices

                        if uses_rerotation or is_finch:
                            # Sort required for key rerotation
                            indices = torch.sort(indices, dim=2).values
                            k = KeyRerotationPress.rerotate_keys(module, indices, full_keys)
                            idx_exp = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                            v = full_values.gather(2, idx_exp).contiguous()
                        else:
                            # kvzip-style: no rerotation
                            idx_exp = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                            k = full_keys.gather(2, idx_exp).contiguous()
                            v = full_values.gather(2, idx_exp).contiguous()

                        # Finch: trim window tokens appended at the end after sorting
                        if is_finch and window_size > 0:
                            k = k[:, :, :-window_size, :]
                            v = v[:, :, :-window_size, :]

                    compressed_cache.update(k.detach().cpu(), v.detach().cpu(), layer_idx)

                cache_filename = cache_filenames_by_cr[cr]
                os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
                torch.save(compressed_cache, cache_filename)
                logger.info(f"Saved cache CR={cr} → {cache_filename}")
                del compressed_cache

            del full_cache
            torch.cuda.empty_cache()

        except StopIteration as e:
            logger.error("StopIteration escaped coroutine", exc_info=True)
            raise RuntimeError("StopIteration escaped coroutine") from e
        except Exception as e:
            logger.error(f"Error in multi-CR cache generation: {str(e)}", exc_info=True)
            raise

    def to_compression_tag(self, compression_ratio: float) -> str:
        """Convert compression ratio to a string tag for directory naming."""
        return (
            str(compression_ratio).replace(".", "_")
            if compression_ratio != 0.0
            else "0"
        )

    def _run_kvpress_prefill(self, inputs, press, cache: DynamicCache) -> None:
        """Prefill via kv-press pipeline (supports Qwen without ``output_attentions``)."""
        with torch.inference_mode():
            with (
                press(self.pipe.model)
                if press is not None
                else contextlib.nullcontext()
            ):
                if hasattr(self.pipe, "_forward"):
                    self.pipe._forward(inputs, press=press, cache=cache)
                else:
                    context_ids = inputs["context_ids"].to(
                        next(self.pipe.model.parameters()).device
                    )
                    out_attn = (
                        self.pipe.output_attentions(press)
                        if hasattr(self.pipe, "output_attentions")
                        else False
                    )
                    self.pipe.model.model(
                        input_ids=context_ids,
                        past_key_values=cache,
                        use_cache=True,
                        output_attentions=out_attn,
                    )

    def _trim_finch_window_cache(self, cache: DynamicCache, window_size: int) -> None:
        if window_size <= 0:
            return
        if hasattr(cache, "layers") and cache.layers:
            for layer in cache.layers:
                layer.keys = layer.keys[:, :, :-window_size, :].contiguous()
                layer.values = layer.values[:, :, :-window_size, :].contiguous()
        elif hasattr(cache, "key_cache") and cache.key_cache:
            cache.key_cache = [
                k[:, :, :-window_size, :].contiguous() for k in cache.key_cache
            ]
            cache.value_cache = [
                v[:, :, :-window_size, :].contiguous() for v in cache.value_cache
            ]

    def _save_dynamic_cache(self, cache: DynamicCache, cache_filename: str) -> None:
        if hasattr(cache, "layers") and cache.layers:
            for layer in cache.layers:
                layer.keys = layer.keys.detach().cpu()
                layer.values = layer.values.detach().cpu()
        elif hasattr(cache, "key_cache") and cache.key_cache:
            cache.key_cache = [k.detach().cpu() for k in cache.key_cache]
            cache.value_cache = [v.detach().cpu() for v in cache.value_cache]
        os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
        torch.save(cache, cache_filename)

    def init(self):
        """Initialize the text model pipeline and compression settings."""
        logger.info("Setting up KV Cache Text Filter...")

        # Set up device (single GPU — avoid device_map="auto" splitting across cuda:0/1 on Kaggle)
        self.device = f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            device_map: dict | str | None = {"": int(self.device_id)}
        else:
            device_map = None

        args = [
            {"attn_implementation": "flash_attention_2"},
            {"attn_implementation": "sdpa"},
            {},
        ]
        self.pipe = None
        for x in args:
            try:
                pipe_kw: dict = {
                    "model": self.model_name,
                    "torch_dtype": torch.bfloat16,
                    "model_kwargs": x or None,
                }
                if device_map is not None:
                    pipe_kw["device_map"] = device_map
                else:
                    pipe_kw["device"] = self.device
                self.pipe = pipeline("kv-press-text-generation", **pipe_kw)  # type: ignore
                break
            except Exception as e:
                logger.warning(
                    f"Error initializing model with args {x}: {str(type(e))}({str(e)})",
                    exc_info=True,
                )

        assert self.pipe is not None, "Failed to initialize the model pipeline."
        self.pipe.model.eval()
        self.tokenizer = self.pipe.tokenizer
        self.tokenizer.pad_token = self.pipe.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Set up compression press
        if self.press_name not in PRESS:
            raise ValueError(
                f"Unknown press_name '{self.press_name}'. Available options: {list(PRESS.keys())}"
            )
        if self.press_name in ("finch", "kvzip", "finch-cachenotes"):
            self.presses = {
                cr: PRESS[self.press_name](compression_ratio=cr)
                if cr > 0.0
                else None
                for cr in self.compression_ratios
            }
        else:
            self.presses = {
                cr: KeyRerotationPress(PRESS[self.press_name](compression_ratio=cr))
                if cr > 0.0
                else None
                for cr in self.compression_ratios
            }

        # Initialize Finch presses with model and tokenizer if using Finch
        if self.press_name in ("finch", "finch-cachenotes"):
            for cr, press in self.presses.items():
                if press is not None and hasattr(press, 'update_model_and_tokenizer'):
                    press.update_model_and_tokenizer(self.pipe.model, self.pipe.tokenizer)
                    logger.info(f"Initialized Finch press for CR={cr} with delimiter token")

        logger.info(
            f"Using press {self.press_name} with compression_ratios {self.compression_ratios}",
        )
        logger.info(
            f"Model {self.model_name} loaded on {next(self.pipe.model.parameters()).device}",
        )

    async def _run_vanilla_inference(
        self,
        texts: List[str],
        all_questions: List[str],
        boolean_question: bool,
    ):
        """Run vanilla inference without KV caching."""
        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.tokenizer is not None, "Tokenizer is not initialized."

        answer_prefix = "Answer: "
        max_new_tokens = 4 if boolean_question else 64

        answers = []

        # Process texts one by one (could be batched for efficiency)
        for text, question in tqdm(
            zip(texts, all_questions),
            total=len(texts),
            desc="Processing texts (vanilla)",
        ):
            try:
                # Determine question suffix based on chat template
                if self.tokenizer.chat_template is None:
                    question_suffix = "\n"
                else:
                    template_context = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": "Example of context\n###"}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    _, question_suffix = template_context.split("\n###")

                # Create full prompt
                full_prompt = text + "\n" + question + question_suffix + answer_prefix

                # Tokenize input
                inputs = self.tokenizer(full_prompt, return_tensors="pt")

                # Move inputs to device
                first_device = next(self.pipe.model.parameters()).device
                inputs = {k: v.to(first_device) for k, v in inputs.items()}

                # Generate response
                with torch.no_grad():
                    generated = self.pipe.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                # Decode the generated tokens
                decoded = self.tokenizer.decode(
                    generated[0][inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )

                answers.append(decoded)

                # Cleanup
                del inputs, generated
                torch.cuda.empty_cache()

            except Exception as e:
                logger.warning(
                    f"Error processing text with question '{question}': {str(e)}"
                )
                answers.append("Not sure")

        return {"answers": answers}

    async def _run_kv_cache_text(
        self,
        column_name: str,
        texts: List[str],
        all_questions: List[str],
        compression_ratio: float,
        cache_dir: str,
        boolean_question: bool,
    ):
        """Run KV cache-based text inference on the data."""

        # Use vanilla inference if gold_vanilla is enabled for CR=0 and 70B model
        if self.gold_vanilla and compression_ratio == 0.0 and "70B" in self.model_name:
            logger.info("Using gold-vanilla mode for inference (CR=0, 70B model)")
            return await self._run_vanilla_inference(
                texts=texts,
                all_questions=all_questions,
                boolean_question=boolean_question,
            )

        assert self.pipe is not None, "Model pipeline is not initialized."
        assert self.tokenizer is not None, "Tokenizer is not initialized."

        # Check if caches exist and generate if not
        save_dir = f"{cache_dir}/{self.model_name}/{self.press_name}/comp{self.to_compression_tag(compression_ratio)}"
        os.makedirs(save_dir, exist_ok=True)

        cache_files = []
        hashed_texts = []

        for i, text in tqdm(
            enumerate(texts),
            total=len(texts),
            desc=f"Preparing caches for cr {compression_ratio}",
        ):
            hashed_text = self.hash_text(text)
            cache_filename = f"{save_dir}/cache_entry_{hashed_text}.pt"

            assert os.path.exists(
                cache_filename
            ), f"Cache file does not exist for text at index {i}: {cache_filename}"
            cache_files.append(cache_filename)
            hashed_texts.append(hashed_text)

        if not cache_files:
            logger.warning("No valid caches or texts found")
            return []

        # Set up text processing parameters
        if boolean_question:
            context = "Answer the following question based on the context with '1' or '0'. Do not add any other comments."
        else:
            context = "Answer the following question based on the context. Do not add any other comments."
        all_questions = [context + " " + q for q in all_questions]
        answer_prefix = "Answer: "
        batch_size = self.compression_ratio_to_batch_size[compression_ratio]
        layer_devices = list({p.device for p in self.pipe.model.parameters()})
        batch_size = self._get_max_batch_size(
            column_name=column_name,
            batch_size=batch_size,
            compression_ratio=compression_ratio,
            layer_devices=layer_devices,
            cache_dir=cache_dir,
            file_paths=cache_files,
        )
        max_new_tokens = 4 if boolean_question else 64

        answers = []
        log_odds = []

        # Process texts in batches
        for batch_start in tqdm(
            range(0, len(cache_files), batch_size),
            desc=f"Processing batches for CR {compression_ratio}",
        ):
            batch_input_ids = []
            batch_attention_masks = []
            caches = []
            context_lengths = []
            questions = []

            batch_cache_files = cache_files[
                batch_start : min(batch_start + batch_size, len(cache_files))
            ]
            batch_questions = all_questions[
                batch_start : min(batch_start + batch_size, len(cache_files))
            ]
            batch_size_actual = len(batch_cache_files)

            # Generate or load caches for each text in the batch
            for i in range(batch_size_actual):
                # Load the pre-generated cache to CPU first to avoid GPU memory accumulation
                cache = torch.load(
                    batch_cache_files[i], map_location="cpu", weights_only=False
                )
                caches.append(cache)
                context_lengths.append(cache.get_seq_length())
                questions.append(batch_questions[i])

            if not caches:
                continue

            max_context_len = max(context_lengths)

            question_ids_list = []
            context_ids_list = []
            padded_context_ids_mask_list = []

            for i, (ctx_len, q) in enumerate(zip(context_lengths, questions)):
                # Pad context to align with KV cache
                padded_context_ids = torch.full(
                    (1, ctx_len),
                    self.tokenizer.pad_token_id + 1,
                    device=self.device,
                )
                pad_len = max_context_len - ctx_len
                padding_ids = torch.full(
                    (1, pad_len), self.tokenizer.pad_token_id, device=self.device
                )
                padded_context = torch.cat([padding_ids, padded_context_ids], dim=1)
                padding_mask = torch.zeros_like(padding_ids)
                padded_context_mask = torch.ones_like(padded_context_ids)
                padded_context_ids_mask = torch.cat(
                    [padding_mask, padded_context_mask], dim=1
                )

                # Determine question suffix
                if self.tokenizer.chat_template is None:
                    question_suffix = "\n"
                else:
                    separator = "\n" + "#" * ctx_len if ctx_len > 0 else "\n#"
                    template_context = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": "Example of context" + separator}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    _, question_suffix = template_context.split(separator)

                # Tokenize question (variable-length)
                complete_question = q + question_suffix + answer_prefix
                question_ids = self.tokenizer.encode(
                    complete_question,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(self.device)

                # Store for later padding
                context_ids_list.append(padded_context)
                question_ids_list.append(question_ids)
                padded_context_ids_mask_list.append(padded_context_ids_mask)

            # Compute max question length across batch
            max_question_len = max(q.shape[1] for q in question_ids_list)

            # Pad questions and build batch tensors
            batch_input_ids = []
            batch_attention_masks = []

            for padded_context, question_ids, padded_context_ids_mask in zip(
                context_ids_list, question_ids_list, padded_context_ids_mask_list
            ):
                q_len = question_ids.shape[1]
                q_pad_len = max_question_len - q_len

                # Pad question to match longest in batch
                if q_pad_len > 0:
                    q_padding = torch.full(
                        (1, q_pad_len),
                        self.tokenizer.pad_token_id,
                        device=self.device,
                    )
                    padded_question = torch.cat([q_padding, question_ids], dim=1)
                    padded_question_mask = torch.cat(
                        [torch.zeros_like(q_padding), torch.ones_like(question_ids)],
                        dim=1,
                    )
                else:
                    padded_question = question_ids
                    padded_question_mask = torch.ones_like(question_ids)

                # Concatenate full input
                input_ids = torch.cat([padded_context, padded_question], dim=1)

                # Build attention mask: padding + context + padding + question
                attention_mask = torch.cat(
                    [padded_context_ids_mask, padded_question_mask], dim=1
                )

                batch_input_ids.append(input_ids)
                batch_attention_masks.append(attention_mask)

            # Stack batched tensors
            batched_inputs = torch.cat(batch_input_ids, dim=0)
            batched_attention_mask = torch.cat(batch_attention_masks, dim=0)

            # Batch the caches
            batched_cache = []
            for layers in zip(*caches):
                max_seq_len = max(k.shape[2] for k, _ in layers)
                keys_padded = []
                values_padded = []

                for k, v in layers:
                    seq_len = k.shape[2]
                    pad_len = max_seq_len - seq_len
                    k_padded = (
                        torch.nn.functional.pad(k, (0, 0, pad_len, 0))
                        if pad_len > 0
                        else k
                    )
                    v_padded = (
                        torch.nn.functional.pad(v, (0, 0, pad_len, 0))
                        if pad_len > 0
                        else v
                    )
                    k_padded = k_padded.contiguous()
                    v_padded = v_padded.contiguous()
                    keys_padded.append(k_padded)
                    values_padded.append(v_padded)

                keys_cat = torch.cat(keys_padded, dim=0)
                values_cat = torch.cat(values_padded, dim=0)
                batched_cache.append((keys_cat, values_cat))

            padded_cache = DynamicCache()
            for layer_idx, (keys, values) in enumerate(batched_cache):
                padded_cache.update(keys, values, layer_idx)

            # Move inputs to the device of the embedding layer (first layer)
            first_device = next(self.pipe.model.parameters()).device
            batched_inputs = batched_inputs.to(first_device)
            batched_attention_mask = batched_attention_mask.to(first_device)

            # Move each cache layer to the same device as the corresponding model layer
            for layer_idx in range(len(padded_cache.key_cache)):
                layer_device = self.pipe.model.model.layers[
                    layer_idx
                ].self_attn.q_proj.weight.device
                padded_cache.key_cache[layer_idx] = padded_cache.key_cache[
                    layer_idx
                ].to(layer_device)
                padded_cache.value_cache[layer_idx] = padded_cache.value_cache[
                    layer_idx
                ].to(layer_device)

            # Generate responses
            with torch.no_grad():
                generated = self.pipe.model.generate(
                    input_ids=batched_inputs,
                    attention_mask=batched_attention_mask,
                    past_key_values=padded_cache,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            logits = generated.scores[0]  # type: ignore
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            id0 = self.tokenizer.convert_tokens_to_ids("0")
            id1 = self.tokenizer.convert_tokens_to_ids("1")
            log_probs_0 = log_probs[:, id0]
            log_probs_1 = log_probs[:, id1]
            log_odds_1_vs_0 = log_probs_1 - log_probs_0

            # Decode the generated tokens
            decoded = self.tokenizer.batch_decode(
                generated.sequences[:, batched_inputs.shape[1] :],  # type: ignore
                skip_special_tokens=True,
            )

            answers.extend(decoded)
            log_odds.extend(log_odds_1_vs_0.cpu().tolist())

            # Cleanup batch resources
            del (
                caches,
                batched_inputs,
                batched_attention_mask,
                padded_cache,
                generated,
                logits,
                log_probs,
            )
            del (
                batch_input_ids,
                batch_attention_masks,
                context_ids_list,
                question_ids_list,
                padded_context_ids_mask_list,
                batched_cache,
            )
            torch.cuda.empty_cache()

        # Process answers and create mask
        # mask = []
        # for data_id, answer in zip(data_ids, answers):
        #     predicted_answer = keep_answer.strip().lower() in answer.lower()
        #     # logger.info(f"Row {data_id} answer: {answer} -> {predicted_answer}")
        #     mask.append((data_id, predicted_answer))

        return {"answers": answers, "log_odds": log_odds}


    async def _generate_cache_for_text(
        self, text_content, cache_filename, compression_ratio, column_name=None
    ):
        """Generate and save text KV cache for a single text."""
        assert (
            self.pipe is not None
        ), "Model pipeline for cache generation is not initialized."

        try:
            # Set the press
            press = self.presses[compression_ratio]

            # For Finch, prepare context with delimiter token and query workload
            if self.press_name == "finch" and press is not None:
                if os.environ.get("MOVIE_FINCH_PREFORMATTED", "").lower() in (
                    "1",
                    "true",
                    "yes",
                ):
                    context_aware = text_content[: min(128000, len(text_content))]
                    window_text = os.environ.get("MOVIE_FINCH_WINDOW_TEXT", "")
                    queries_workload = window_text
                    cpt = window_text
                else:
                    yaml_path = "queries_workloads/_workload.yaml"
                    with open(yaml_path, "r") as f:
                        data = yaml.safe_load(f)

                    query_list = data["queries"]
                    queries_workload = "Pay attention to these examples of questions:\n" + "\n".join(
                        f"- {q}" for q in query_list
                    )

                    context = text_content[: min(128000, len(text_content))]
                    context_aware = context + press.delimiter_token + queries_workload

            elif self.press_name == "finch-cachenotes" and press is not None:
                import pandas as pd
                import hashlib
                import re
                
                def normalize_text(text):
                    """Normalize text by removing extra whitespace and standardizing characters."""
                    text = text.strip()
                    # Normalize whitespace but preserve newlines
                    text = re.sub(r'[ \t]+', ' ', text)  # Only collapse spaces and tabs
                    text = re.sub(r'\n\s*\n', '\n\n', text)  # Collapse multiple blank lines
                    text = text.replace('\r\n', '\n').replace('\r', '\n')  # Normalize line endings
                    text = text.replace('\u200b', '')  # Remove zero-width spaces
                    return text
                
                # Resolve task and text column from column_name
                assert column_name is not None, "column_name is required for finch-cachenotes"
                assert column_name in CPT_COLUMN_MAP, (
                    f"Unknown column_name '{column_name}' for finch-cachenotes. "
                    f"Supported: {list(CPT_COLUMN_MAP.keys())}"
                )
                task_key, text_col = CPT_COLUMN_MAP[column_name]
                cpt_path = CPT_PATH[task_key]

                # Load CSV file
                df = pd.read_csv(cpt_path)
                
                # Normalize text_content for comparison
                normalized_text_content = normalize_text(text_content)
                text_hash = hashlib.sha256(text_content.encode()).hexdigest()
                
                # Try to find matching row by comparing text content with the correct column
                matching_row = None
                for _, row in df.iterrows():
                    row_text = str(row.get(text_col, ''))
                    normalized_row_text = normalize_text(row_text)
                    
                    # Try exact match first, then normalized match, then hash match
                    if (row_text == text_content or 
                        normalized_row_text == normalized_text_content or
                        hashlib.sha256(row_text.encode()).hexdigest() == text_hash):
                        matching_row = row
                        break
                
                # Get the CPT from the matching row
                if matching_row is not None:
                    cpt = str(matching_row['cpt'])
                    context = text_content[: min(128000, len(text_content))]
                    context_aware = context + press.delimiter_token + cpt
                else:
                    logger.warning(
                        f"No matching CPT found for column '{column_name}' "
                        f"(task={task_key}, text_col={text_col})"
                    )
                    #raise ValueError(f"No matching CPT found for text in {task_key} dataset")
                    cpt = "Your task is to answer questions based on the context."
                    context = text_content[: min(128000, len(text_content))]
                    context_aware = context + press.delimiter_token + cpt
            else:
                context_aware = text_content[: min(128000, len(text_content))]  # Limit context length

            answer_prefix = "Answer: "

            # Generate cache using preprocess method
            inputs = self.pipe.preprocess(
                context=context_aware,
                questions=[""],
                answer_prefix=answer_prefix,
                max_context_length=128000,
            )

            context_ids = inputs["context_ids"]
            
            # For Finch: Adjust compression ratio to ensure fair comparison
            # Goal: After removing window tokens, retain same % of context as other methods
            if self.press_name in ("finch", "finch-cachenotes") and press is not None and compression_ratio > 0:
                # Use the actual tokenized length from context_ids instead of re-encoding
                actual_context_ids_length = context_ids.shape[1]
                
                # Tokenize to get exact counts of components
                if self.press_name == "finch":
                    window_tokens_count = len(self.tokenizer.encode(queries_workload, add_special_tokens=False))
                else:
                    window_tokens_count = len(self.tokenizer.encode(cpt, add_special_tokens=False))

                context_tokens_count = actual_context_ids_length - window_tokens_count - 1
                
                # Calculate adjusted ratio
                desired_context_kept = int(context_tokens_count * (1 - compression_ratio))
                # Total to keep before window removal: desired_context + window
                total_to_keep = desired_context_kept + window_tokens_count
                # Effective compression ratio
                adjusted_ratio = 1.0 - (total_to_keep / actual_context_ids_length)
                adjusted_ratio = max(0.0, min(1.0, adjusted_ratio)) 
                
                # Create a new press instance with adjusted ratio
                press = FinchPress(compression_ratio=adjusted_ratio)
                press.update_model_and_tokenizer(self.pipe.model, self.pipe.tokenizer)
            
            cache = DynamicCache()
            logger.info(
                f"Generating cache for text with initial length {context_ids.shape[1]} tokens"
            )
            self._run_kvpress_prefill(inputs, press, cache)

            if self.press_name in ("finch", "finch-cachenotes") and press is not None:
                ws = getattr(press, "window_size", None)
                if ws is not None and int(ws) > 0:
                    self._trim_finch_window_cache(cache, int(ws))

            logger.info(f"Saving cache to: {cache_filename}")
            self._save_dynamic_cache(cache, cache_filename)

            if not os.path.exists(cache_filename):
                logger.error(f"Cache file was not actually created: {cache_filename}")

            # Cleanup
            del cache
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error generating cache for text {cache_filename}: {str(e)}")
            raise


model_wrapper = None


class Status(Resource):
    def get(self):
        assert model_wrapper is not None
        return {
            "status": "alive",
            "model_name": model_wrapper.model_name,
            "compression_ratios": list(model_wrapper.compression_ratios),
        }, 200


class PrepareCaches(Resource):
    def post(self):
        """Expects JSON of the form:
        {
            "texts": [
                "text1",
                "text2",
            ],
            "cache_dir": "/path/to/cache/dir"
            "compression_ratio": 0.5
        }
        """
        data = request.get_json(force=True)
        column_name = data["column_name"]
        texts = data["texts"]
        cache_dir = data["cache_dir"]
        compresion_ratio = data["compression_ratio"]
        assert model_wrapper is not None
        asyncio.run(
            model_wrapper.prepare_caches(
                column_name=column_name,
                texts=texts,
                cache_dir=cache_dir,
                compression_ratio=compresion_ratio,
            )
        )
        return {"status": "cache_ready"}, 200


class TextQA(Resource):
    def post(self):
        """Expects JSON of the form:
        {
            "texts": [
                "text 1",
                "text 2",
            ],
            "questions": [
                "How many cats are there?"
                "How many dogs are there?"
            ],
            "compression_ratio": 0.5,
            "boolean": true,
            "cache_dir": "/path/to/cache/dir"
        }
        """
        data = request.get_json(force=True)
        column_name = data["column_name"]
        texts = data["texts"]
        questions = data["questions"]
        compression_ratio = data["compression_ratio"]
        boolean_question = data["boolean"]
        cache_dir = data["cache_dir"]
        assert model_wrapper is not None
        assert (
            len(texts) == len(questions)
        ), f"Number of texts {len(texts)} must match number of questions {len(questions)}"
        responses = model_wrapper.compute_text_qa_response(
            column_name=column_name,
            texts=texts,
            questions=questions,
            compression_ratio=compression_ratio,
            boolean_question=boolean_question,
            cache_dir=cache_dir,
        )
        return responses, 200


if not _PREGEN_ONLY:
    api.add_resource(Status, "/status")
    api.add_resource(TextQA, "/text_qa")
    api.add_resource(PrepareCaches, "/prepare_caches")

if __name__ == "__main__":
    if _PREGEN_ONLY:
        raise SystemExit("Set MOVIE_KV_PREGEN=0 to run the HTTP server, or use run_movie_kv_pregen.py")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Device ID for GPU to use",
    )
    parser.add_argument(
        "--gold-vanilla",
        action="store_true",
        help="Use vanilla inference (no KV caching) for CR=0 with 70B models",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help="Hugging Face model id (e.g. Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--press-name",
        type=str,
        choices=sorted(PRESS.keys()),
        default="expected_attention",
        help="Name of the compression press to use",
    )
    args = parser.parse_args()
    device_id = args.device_id
    gold_vanilla = args.gold_vanilla
    model_name = args.model_name
    press_name = args.press_name
    model_wrapper = KvTextQaModelWrapper(
        model_name,
        device_id,
        batch_sizes=[None, None, None, None, None, None, None],
        compression_ratios=[
            0.0,
            0.5,
            0.8,
            0.9,
            0.6,
            0.4,
            0.3,
        ], 
        gold_vanilla=gold_vanilla,
        press_name=press_name,
    )
    app.run(host="127.0.0.1", port=PORT_KV_TEXT_QA.get(model_name), debug=False)