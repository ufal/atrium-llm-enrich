"""
llm_utils.py — Reusable components for the LLM Semantic Enrichment Pipeline.

Contains:
  • Compatibility shims for bitsandbytes and transformers 5.x.
  • MODEL_REGISTRY — extended with large-scale models for multi-GPU runs.
  • Configuration loader.
  • Line-quality filter (_should_process_line) — bug-fixed empty-categ handling.
  • Model/tokenizer loader for the transformers backend (BnB 4-bit, AWQ, GGUF).
  • vLLM engine loader (load_vllm_engine) for high-throughput multi-GPU runs.
  • Context-window builder (get_context_window).
  • Single-document processor for the transformers backend (process_document).
  • Batched single-document processor for the vLLM backend (process_document_vllm).

Backend dispatch:
  • BACKEND=transformers  — uses HuggingFace Transformers + lmformatenforcer for
                            constrained JSON decoding. Good for single-GPU / ≤31B.
  • BACKEND=vllm          — uses vLLM + xgrammar for native guided JSON decoding,
                            Automatic Prefix Caching (APC), and tensor parallelism.
                            Required for models ≥70B or any multi-GPU node.

Import order note:
  Import this module before any other CUDA-touching library. The
  PYTORCH_CUDA_ALLOC_CONF guard at the top fires at module-load time and must
  precede any library that initialises the CUDA allocator (e.g. bitsandbytes).
"""

# PYTORCH_CUDA_ALLOC_CONF must be set before ANY import that can touch the CUDA
# context — bitsandbytes initialises it via a C extension on import.
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import csv  # noqa: E402
import fnmatch  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys as _sys_tc  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional, Tuple, Union  # noqa: E402

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

_api_util_path = str(Path(__file__).parent / "api_util")
if _api_util_path not in _sys_tc.path:
    _sys_tc.path.insert(0, _api_util_path)
import transformers.tokenization_utils as _tu  # noqa: E402
import transformers.tokenization_utils_base as _tub  # noqa: E402

from api_util import teitok_read  # noqa: E402
from api_util.teitok_read import doc_id_from_path  # noqa: E402


def validate_llm_output(
    result_json: str, EnrichmentModel: type, file_id: str, page_num: int, line_num: int
) -> dict:
    """
    Pure helper to validate and sanitize LLM JSON output against a Pydantic model.
    Extracted for unit testing independent of GPU inference.
    """
    from pydantic import ValidationError  # noqa: E402

    try:
        semantic_data = EnrichmentModel.model_validate_json(result_json)
    except ValidationError:
        try:
            raw_dict = json.loads(result_json, strict=False)
            if "confidence_score" in raw_dict:
                try:
                    val = float(raw_dict["confidence_score"])
                    raw_dict["confidence_score"] = min(1.0, max(0.0, val))
                except (ValueError, TypeError):
                    pass
            semantic_data = EnrichmentModel.model_validate(raw_dict)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(
                f"[{file_id}] Persistent validation error P{page_num} L{line_num}: {exc}"
            ) from exc

    dump_data = semantic_data.model_dump()

    # Graceful handling for teater_category based on Pydantic methods or dictionary keys
    if hasattr(semantic_data, "category_name"):
        dump_data["teater_category"] = semantic_data.category_name()
    else:
        dump_data["teater_category"] = dump_data.get("teater_category", "")

    if dump_data.get("teater_category") == "Nerelevantní (meta-text)":
        dump_data["extracted_keywords_cs"] = []
        dump_data["extracted_keywords_en"] = []

    return dump_data


def _patch_tokenizer_compat() -> bool:
    """
    Inject PreTrainedTokenizerBase into the live transformers.tokenization_utils
    module if it is missing.

    Must be called TWICE:
      1. At module load time (Pass 1, below) — patches the lazy stub.
      2. After ``from transformers import AutoModelForCausalLM, AutoTokenizer``
         (Pass 2, below) — patches the real module that replaces the lazy stub.

    Safe to call any number of times; subsequent calls are no-ops once the
    attribute is present.
    """
    # Always operate on the *live* sys.modules entry, not the stub captured
    # at import time — they may be different objects after lazy resolution.
    live = _sys_tc.modules.get("transformers.tokenization_utils", _tu)
    if not hasattr(live, "PreTrainedTokenizerBase"):
        if hasattr(_tub, "PreTrainedTokenizerBase"):
            live.PreTrainedTokenizerBase = _tub.PreTrainedTokenizerBase
            return True
    return False


_patch_tokenizer_compat()  # Pass 1 — lazy stub


# ---------------------------------------------------------------------------
# Compatibility shim: transformers 5.x removed all_special_tokens_extended
# ---------------------------------------------------------------------------
#
# Root-cause (observed with vLLM 0.7.3 + transformers 5.x):
#
#   vLLM's get_cached_tokenizer() calls tokenizer.all_special_tokens_extended
#   to pre-populate a special-token cache.  This property existed on
#   PreTrainedTokenizerBase in transformers 4.x but was removed in 5.x.
#   The result is:
#       AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended
#   This hits every model that goes through vLLM's tokenizer init path —
#   hence all MoE models (vLLM backend) fail while transformers-backend models
#   (which never call get_cached_tokenizer) work fine.
#
# Fix: inject a compatible property onto PreTrainedTokenizerBase once, before
#   vLLM touches any tokenizer.  The reconstructed property mirrors the old
#   4.x behaviour: returns a list of AddedToken objects (or plain strings as
#   fallback) for every entry in all_special_tokens.


def _patch_all_special_tokens_extended() -> bool:
    """
    Inject ``all_special_tokens_extended`` onto ``PreTrainedTokenizerBase`` if
    the property is missing (transformers 5.x).

    The reconstructed property returns a list of ``AddedToken`` / ``str``
    objects for every special token — sufficient for vLLM 0.7.3's
    ``get_cached_tokenizer()`` call.  Safe to call multiple times; subsequent
    calls are no-ops once the attribute is present.
    """
    try:
        import transformers.tokenization_utils_base as _tub2  # noqa: E402

        _PTTB = getattr(_tub2, "PreTrainedTokenizerBase", None)
        if _PTTB is None:
            return False
        if hasattr(_PTTB, "all_special_tokens_extended"):
            return False  # already present — nothing to do

        @property  # type: ignore[misc]
        def _all_special_tokens_extended(self):  # type: ignore[return]
            """Reconstructed shim for vLLM compatibility (transformers 5.x)."""
            result = []
            seen: set = set()
            for tok_str in self.all_special_tokens:
                if tok_str in seen:
                    continue
                seen.add(tok_str)
                tok_id = self.added_tokens_encoder.get(tok_str)
                if tok_id is not None:
                    added_tok = self.added_tokens_decoder.get(tok_id)
                    result.append(added_tok if added_tok is not None else tok_str)
                else:
                    result.append(tok_str)
            return result

        _PTTB.all_special_tokens_extended = _all_special_tokens_extended
        print(
            "[COMPAT] Patched PreTrainedTokenizerBase.all_special_tokens_extended "
            "(removed in transformers 5.x; required by vLLM 0.7.3). "
            "Permanent fix: pip install -U vllm"
        )
        return True
    except Exception as exc:
        print(f"[WARN] Could not apply all_special_tokens_extended shim: {exc}")
        return False


_patch_all_special_tokens_extended()


# ---------------------------------------------------------------------------
# Compatibility shim: bitsandbytes < 0.44 vs newer transformers / accelerate
# ---------------------------------------------------------------------------


def _patch_params4bit_compat() -> bool:
    """
    Patch two known bitsandbytes breakage points against newer accelerate:

    1. ``Params4bit.__new__`` / ``__init__`` chokes on the ``_is_hf_initialized``
       kwarg injected by accelerate.
    2. ``QuantState.as_dict`` raises when ``offset`` is a meta-device tensor.

    Permanent fix: ``pip install -U bitsandbytes`` inside your virtual env.
    These patches are no-ops if the installed version is already fixed.
    """
    try:
        import inspect  # noqa: E402

        import bitsandbytes.functional as _bnb_func  # noqa: E402
        import bitsandbytes.nn as _bnb_nn  # noqa: E402

        patched = False

        # 1. Params4bit stray-kwarg fix
        new_sig = str(inspect.signature(_bnb_nn.Params4bit.__new__))
        if "_is_hf_initialized" not in new_sig and "**" not in new_sig:
            _orig_new = _bnb_nn.Params4bit.__new__

            def _p4b_new(cls, *args, **kwargs):
                kwargs.pop("_is_hf_initialized", None)
                return _orig_new(cls, *args, **kwargs)

            _bnb_nn.Params4bit.__new__ = _p4b_new

            if "__init__" in _bnb_nn.Params4bit.__dict__:
                _orig_init = _bnb_nn.Params4bit.__init__

                def _p4b_init(self, *args, **kwargs):
                    kwargs.pop("_is_hf_initialized", None)
                    return _orig_init(self, *args, **kwargs)

                _bnb_nn.Params4bit.__init__ = _p4b_init

            patched = True

        # 2. QuantState meta-tensor fix
        if hasattr(_bnb_func, "QuantState"):
            _orig_as_dict = _bnb_func.QuantState.as_dict

            def _patched_as_dict(self, packed: bool = False):
                orig_offset = getattr(self, "offset", None)
                is_meta = (
                    isinstance(orig_offset, torch.Tensor) and orig_offset.device.type == "meta"
                )
                if is_meta:
                    self.offset = torch.tensor(0.0)
                try:
                    return _orig_as_dict(self, packed=packed)
                finally:
                    if is_meta:
                        self.offset = orig_offset

            if _bnb_func.QuantState.as_dict.__name__ != "_patched_as_dict":
                _bnb_func.QuantState.as_dict = _patched_as_dict
                patched = True

        if patched:
            print(
                "[COMPAT] Patched bitsandbytes (Params4bit & QuantState) for "
                "accelerate/meta-device compatibility. "
                "Permanent fix: pip install -U bitsandbytes"
            )
        return patched

    except Exception as exc:
        print(f"[WARN] Could not apply bitsandbytes compat patch: {exc}")
        return False


_patch_params4bit_compat()


# ---------------------------------------------------------------------------
# GPU diagnostics helpers
# ---------------------------------------------------------------------------


def log_gpu_info() -> None:
    """Print CUDA device names and total VRAM at startup; no-op if CUDA unavailable."""
    if not torch.cuda.is_available():
        print("[GPU] No CUDA devices — running on CPU.")
        return
    n = torch.cuda.device_count()
    print(f"[GPU] {n} device(s) detected:")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / 1024**3
        free_gb = (props.total_memory - torch.cuda.memory_reserved(i)) / 1024**3
        print(f"  [{i}] {props.name}  total={total_gb:.1f} GB  free={free_gb:.1f} GB")


def log_gpu_memory(label: str = "") -> None:
    """Print allocated/reserved VRAM per device; no-op if CUDA unavailable."""
    if not torch.cuda.is_available():
        return
    tag = f" ({label})" if label else ""
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        reserv = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        pct = reserv / total * 100 if total else 0
        print(
            f"[GPU:{i}]{tag} allocated={alloc:.2f} GB  "
            f"reserved={reserv:.2f} GB  ({pct:.1f}% of {total:.1f} GB)"
        )


# ---------------------------------------------------------------------------
# vLLM compatibility shim: rope_scaling 'type' → 'rope_type' rename
# ---------------------------------------------------------------------------
#
# Root-cause (vLLM 0.7.3 + transformers 5.x model configs):
#
#   Newer HuggingFace model configs (Gemma 4, some Qwen variants) store their
#   rope_scaling dict with a 'type' key, e.g. {"type": "linear", "factor": 8}.
#   vLLM 0.7.3's patch_rope_scaling_dict was tightened to require the newer
#   'rope_type' key and raises unconditionally if it is absent:
#       ValueError: rope_scaling should have a 'rope_type' key
#
# Fix: monkey-patch patch_rope_scaling_dict to silently rename 'type' →
#   'rope_type' before the strictness check fires.  Applied lazily inside
#   load_vllm_engine() (vLLM may not be installed, so we can't apply at
#   module load time).


def _patch_vllm_rope_scaling_compat() -> bool:
    """
    Patch ``vllm.transformers_utils.config.patch_rope_scaling_dict`` to
    accept rope_scaling dicts that use the legacy ``'type'`` key instead of
    the newer ``'rope_type'`` key (vLLM 0.7.3 + Gemma-4 / newer model configs).

    Safe to call multiple times; subsequent calls are no-ops once patched.
    Applied inside ``load_vllm_engine()`` before ``LLM(**engine_kwargs)``.

    Prints ``[COMPAT] vLLM rope_scaling shim installed`` at patch time so the
    log unambiguously shows the shim is active — separate from the per-call
    ``[COMPAT] rope_scaling: renamed`` message that only prints when a config
    actually needs the fix.
    """
    try:
        import vllm.transformers_utils.config as _vllm_cfg  # noqa: E402

        _orig = getattr(_vllm_cfg, "patch_rope_scaling_dict", None)
        if _orig is None:
            return False  # vLLM version without this function — nothing to do
        if getattr(_orig, "_atrium_patched", False):
            return False  # already patched in a previous call

        def _patched_rope_scaling_dict(rope_scaling: dict) -> None:
            if isinstance(rope_scaling, dict) and "rope_type" not in rope_scaling:
                if "type" in rope_scaling:
                    # In-place rename so all downstream vLLM code sees rope_type.
                    # The dict is owned by the HF config object — mutating it is safe.
                    rope_scaling["rope_type"] = rope_scaling.pop("type")
                    print(
                        "[COMPAT] rope_scaling: renamed legacy 'type' → 'rope_type' "
                        "(vLLM 0.7.x + newer model config). "
                        "Permanent fix: pip install -U vllm"
                    )
                # If neither key is present let the original function raise naturally —
                # that case requires a genuine vLLM upgrade, not just a rename.
            return _orig(rope_scaling)

        _patched_rope_scaling_dict._atrium_patched = True
        _vllm_cfg.patch_rope_scaling_dict = _patched_rope_scaling_dict
        print(
            "[COMPAT] vLLM rope_scaling shim installed "
            "('type'→'rope_type' auto-rename for older model configs). "
            "Permanent fix: pip install -U vllm"
        )
        return True

    except Exception as exc:
        print(f"[WARN] Could not apply vLLM rope_scaling compat patch: {exc}")
        return False


# Transformers imports — deferred inside functions where only needed by one backend
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# Pass 2 — real module (lazy stubs resolved by the import above)
_patch_tokenizer_compat()


# ---------------------------------------------------------------------------
# 1. Model Registry
# ---------------------------------------------------------------------------
#
# Static fields (model identity / hardware requirements):
#   hf_id               — HuggingFace model ID (or local path for GGUF).
#   context_window      — Maximum input tokens (prompt + output combined).
#   trust_remote_code   — Pass to from_pretrained (needed for some models).
#   torch_dtype         — Compute dtype for transformers backend.
#   hf_token_required   — Whether a HF_TOKEN is needed to download the model.
#   load_in_4bit        — Use BitsAndBytes 4-bit NF4 quantization (transformers).
#   is_awq              — Model is pre-quantized AWQ; use autoawq loader.
#   is_gguf             — GGUF file; use llama.cpp loader.
#   is_moe              — Mixture-of-Experts architecture.
#   bnb_experts_broken  — BnB 4-bit fails on fused expert blocks; use vLLM.
#   vllm_only           — Cannot be used with the transformers backend.
#   recommended_tp      — Suggested tensor_parallel_size for vLLM.
#   max_quant_ratio     — Override for _verify_quantization_effective threshold.
#   weight_footprint_gb — Approx. serving-weight size in GB as stored on HF
#                         (pre-quantized checkpoints: their real file size).
#                         Last-resort source for CPU_OFFLOAD_GB=auto sizing —
#                         the local HF cache / Hub API take precedence.
#   notes               — Human-readable deployment notes.
#
# inference_defaults:
#   Per-model recommended values for all runtime parameters that are otherwise
#   set in llm_config.txt.  These become the effective values unless the user
#   explicitly overrides them in the config file.  See get_inference_defaults().
#
#   backend               — "transformers" | "vllm"
#   tensor_parallel_size  — Number of GPUs (vLLM only; ignored for transformers)
#   gpu_memory_utilization— Fraction of each GPU's VRAM for vLLM (0.0–0.95)
#   max_model_len         — Override for model's native context window (int|None)
#   vllm_batch_size       — Lines per vLLM generate() call
#   cpu_offload_gb        — GB of weights kept in CPU RAM (vLLM UVA zero-copy),
#                           or "auto" to size from footprint vs. detected VRAM
#                           at engine load (issue ufal/atrium-project#26)

MODEL_REGISTRY: Dict[str, Dict] = {
    "qwen-3.6-35b-moe": {
        "hf_id": "Qwen/Qwen3.6-35B-A3B",
        "context_window": 262144,
        "trust_remote_code": False,
        "torch_dtype": torch.bfloat16,
        "hf_token_required": False,
        "weight_footprint_gb": 70,
        "is_moe": True,
        "load_in_4bit": True,  # Updated via correction
        "recommended_tp": 1,
        "min_vllm_version": "0.8.0",
        "inference_defaults": {
            "backend": "vllm",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.90,
            "max_model_len": None,
            "vllm_batch_size": 16,
        },
    },
    "gemma-4-26b-moe": {
        "hf_id": "google/gemma-4-26B-A4B-it",
        "context_window": 256000,
        "trust_remote_code": False,
        "torch_dtype": torch.bfloat16,
        "hf_token_required": True,
        "weight_footprint_gb": 52,
        "is_moe": True,
        "load_in_4bit": True,  # Updated via correction
        "recommended_tp": 2,
        "inference_defaults": {
            "backend": "vllm",
            "tensor_parallel_size": 2,
            "gpu_memory_utilization": 0.90,
            "max_model_len": None,
            "vllm_batch_size": 16,
        },
    },
    "qwen3-235b-a22b": {
        "hf_id": "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "context_window": 131072,
        "trust_remote_code": False,
        "torch_dtype": torch.bfloat16,
        "hf_token_required": False,
        "is_moe": True,
        "load_in_4bit": True,  # Updated via correction
        "recommended_tp": 8,
        "min_vllm_version": "0.8.0",
        "weight_footprint_gb": 470,  # BF16, ~2 bytes/param
        "inference_defaults": {
            "backend": "vllm",
            "tensor_parallel_size": 8,
            "gpu_memory_utilization": 0.88,
            "max_model_len": 16384,
            "vllm_batch_size": 8,
            "cpu_offload_gb": "auto",  # sized at load; 0 where weights fit
        },
    },
    "llama4-maverick": {
        "hf_id": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        "context_window": 1048576,
        "trust_remote_code": False,
        "torch_dtype": torch.bfloat16,
        "hf_token_required": True,
        "is_moe": True,
        "load_in_4bit": True,  # Updated via correction
        "recommended_tp": 8,
        "min_vllm_version": "0.8.0",
        "weight_footprint_gb": 800,  # BF16, ~400 B total params
        "inference_defaults": {
            "backend": "vllm",
            "tensor_parallel_size": 8,
            "gpu_memory_utilization": 0.88,
            "max_model_len": 16384,
            "vllm_batch_size": 4,
            "cpu_offload_gb": "auto",  # sized at load; 0 where weights fit
        },
    },
    # ------------------------------------------------------------------
    # Small / mid models — single GPU, transformers backend
    # ------------------------------------------------------------------
    "qwen3-8b": {
        "hf_id": "Qwen/Qwen3-8B",
        "context_window": 131072,
        "trust_remote_code": False,
        "torch_dtype": torch.bfloat16,
        "hf_token_required": False,
        "weight_footprint_gb": 16,
        "inference_defaults": {
            "backend": "transformers",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.90,
            "max_model_len": None,
            "vllm_batch_size": 16,
        },
    },
    "qwen-3.5-9b-it": {
        "hf_id": "Qwen/Qwen3.5-9B",
        "context_window": 262144,
        "trust_remote_code": False,
        "torch_dtype": torch.bfloat16,
        "hf_token_required": False,
        "weight_footprint_gb": 18,
        "inference_defaults": {
            "backend": "transformers",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.90,
            "max_model_len": None,
            "vllm_batch_size": 16,
        },
    },
    "qwen2.5-14b-awq": {
        "hf_id": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "context_window": 131072,
        "trust_remote_code": False,
        "torch_dtype": torch.float16,
        "hf_token_required": False,
        "weight_footprint_gb": 10,  # AWQ 4-bit
        "is_awq": True,
        "inference_defaults": {
            "backend": "transformers",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.90,
            "max_model_len": None,
            "vllm_batch_size": 16,
        },
    },
    "qwen3-14b": {
        "hf_id": "OpenPipe/Qwen3-14B-Instruct",
        "context_window": 131072,
        "trust_remote_code": False,
        "torch_dtype": torch.bfloat16,
        "hf_token_required": False,
        "weight_footprint_gb": 30,
        "load_in_4bit": True,
        "inference_defaults": {
            "backend": "transformers",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.90,
            "max_model_len": None,
            "vllm_batch_size": 16,
        },
    },
    "gemma-3-12b-it": {
        "hf_id": "google/gemma-3-12b-it",
        "context_window": 131072,
        "trust_remote_code": False,
        "torch_dtype": torch.bfloat16,
        "hf_token_required": True,
        "weight_footprint_gb": 24,
        "inference_defaults": {
            "backend": "transformers",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.90,
            "max_model_len": None,
            "vllm_batch_size": 16,
        },
    },
    "qwen2.5-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "context_window": 32768,
        "trust_remote_code": False,
        "torch_dtype": torch.bfloat16,
        "hf_token_required": False,
        "weight_footprint_gb": 15,
        "inference_defaults": {
            "backend": "transformers",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.90,
            "max_model_len": None,
            "vllm_batch_size": 16,
        },
    },
    # ------------------------------------------------------------------
    # Mid / large — single 80 GB GPU with BnB 4-bit, or vLLM
    # ------------------------------------------------------------------
    "qwen-3.6-27b-it": {
        "hf_id": "Qwen/Qwen3.6-27B",
        "context_window": 262144,
        "trust_remote_code": False,
        "torch_dtype": torch.bfloat16,
        "hf_token_required": False,
        "weight_footprint_gb": 54,
        "load_in_4bit": True,
        "notes": "Best accuracy/VRAM ratio for single-GPU runs.",
        "inference_defaults": {
            "backend": "transformers",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.90,
            "max_model_len": None,
            "vllm_batch_size": 16,
        },
    },
    "gemma-4-31b-it": {
        "hf_id": "google/gemma-4-31B-it",
        "context_window": 256000,
        "trust_remote_code": False,
        "torch_dtype": torch.bfloat16,
        "hf_token_required": True,
        "weight_footprint_gb": 62,
        "load_in_4bit": True,
        "notes": "Highest accuracy on single GPU (4-bit). Gated model.",
        "inference_defaults": {
            "backend": "transformers",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.90,
            "max_model_len": None,
            "vllm_batch_size": 16,
        },
    },
    "llama3.1-70b": {
        "hf_id": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "context_window": 131072,
        "trust_remote_code": False,
        "torch_dtype": torch.bfloat16,
        "hf_token_required": True,
        "weight_footprint_gb": 141,  # BF16; BnB 4-bit (transformers) is ~35 GB
        "load_in_4bit": True,
        "recommended_tp": 2,
        "notes": (
            "Default: transformers backend + BnB 4-bit (~35 GB, single A40/A100). "
            "For higher throughput on 2×A100 80 GB nodes override in config: "
            "BACKEND=vllm + TENSOR_PARALLEL_SIZE=2."
        ),
        "inference_defaults": {
            "backend": "transformers",  # 4-bit BnB fits on any 48 GB+ GPU (single)
            "tensor_parallel_size": 1,  # vLLM override needs 2×80 GB (TP=2) or 3×48 GB (TP=3)
            "gpu_memory_utilization": 0.90,
            "max_model_len": None,
            "vllm_batch_size": 8,
        },
    },
    # ------------------------------------------------------------------
    # MoE models — GGUF fallback (llama.cpp, any single GPU)
    # ------------------------------------------------------------------
    "gemma-4-26b-moe-gguf": {
        "hf_id": "bartowski/google_gemma-4-26B-A4B-it-GGUF",
        "filename": "*Q4_K_M.gguf",
        "context_window": 8192,
        "is_gguf": True,
        "hf_token_required": False,
        "weight_footprint_gb": 16,  # Q4_K_M
        "notes": "MoE via llama.cpp. BnB 4-bit unsupported (fused experts).",
        "inference_defaults": {
            "backend": "transformers",  # llama.cpp is reached via the transformers path
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.90,
            "max_model_len": None,
            "vllm_batch_size": 16,
        },
    },
    # ------------------------------------------------------------------
    # MoE models — vLLM only (single or multi-GPU)
    # ------------------------------------------------------------------
    "gemma-4-26b-moe-awq": {
        "hf_id": "google/gemma-4-26B-A4B-it",
        "context_window": 256000,
        "trust_remote_code": False,
        "torch_dtype": torch.bfloat16,
        "hf_token_required": True,
        "weight_footprint_gb": 15,  # AWQ 4-bit
        "is_moe": True,
        "bnb_experts_broken": True,
        "is_awq": True,
        "notes": "AWQ variant of gemma-4-26b-moe.",
        "inference_defaults": {
            "backend": "vllm",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.90,
            "max_model_len": None,
            "vllm_batch_size": 16,
        },
    },
    # ------------------------------------------------------------------
    # Large models — vLLM only, multi-GPU
    # ------------------------------------------------------------------
    "qwen3-235b-a22b-fp8": {
        "hf_id": "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
        "context_window": 131072,
        "trust_remote_code": False,
        "torch_dtype": torch.bfloat16,  # compute dtype; FP8 storage handled by vLLM
        "hf_token_required": False,
        "is_moe": True,
        "bnb_experts_broken": True,
        "vllm_only": True,
        "recommended_tp": 8,
        "min_vllm_version": "0.8.0",
        "weight_footprint_gb": 235,  # native FP8, ~1 byte/param
        "notes": (
            "FP8-quantised Qwen3 235B MoE. Weight footprint ≈ 235 GB — fits in "
            "8× A100 40 GB = 320 GB total (85 GB headroom for KV cache). "
            "Requires vLLM ≥ 0.8.x — Qwen3 MoE released April 2025, after "
            "vLLM 0.7.3 (Feb 2025). "
            "FP8 COMPUTE WARNING: hardware FP8 tensor-core acceleration requires "
            "Compute Capability ≥ 8.9 (Ada Lovelace / Hopper). "
            "The A100 on tdll-8gpu is CC 8.0 — weights are stored as FP8 "
            "(memory benefit preserved) but all matmuls execute in BF16 "
            "(same throughput as a hypothetical BF16 run, but that would not fit). "
            "Full FP8 speed benefit requires H100 / L40S class hardware."
        ),
        "inference_defaults": {
            "backend": "vllm",
            "tensor_parallel_size": 8,
            "gpu_memory_utilization": 0.88,
            "max_model_len": 16384,
            "vllm_batch_size": 8,
            "cpu_offload_gb": "auto",  # sized at load; 0 where weights fit
        },
    },
    "deepseek-v3": {
        "hf_id": "deepseek-ai/DeepSeek-V3",
        "context_window": 131072,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "hf_token_required": False,
        "is_moe": True,
        "bnb_experts_broken": True,
        "vllm_only": True,
        "recommended_tp": 4,
        "weight_footprint_gb": 690,  # official FP8 checkpoint file size
        "notes": (
            "671B MoE. Official FP8 checkpoint: deepseek-ai/DeepSeek-V3-0324. "
            "Requires 8× A100 40 GB minimum in FP8 (exceeds available hardware). "
            "Best used with vLLM + tensor_parallel_size=8 + "
            "gpu_memory_utilization=0.92."
        ),
        "inference_defaults": {
            "backend": "vllm",
            "tensor_parallel_size": 8,  # minimum viable on 8× A100 40 GB
            "gpu_memory_utilization": 0.92,
            "max_model_len": 16384,
            "vllm_batch_size": 4,
            "cpu_offload_gb": "auto",  # sized at load; 0 where weights fit
        },
    },
    # ------------------------------------------------------------------
    # Archived / Unsuccessful models (kept for reference)
    # ------------------------------------------------------------------
    "bielik-11b-v3.0": {
        "hf_id": "speakleash/Bielik-11B-v3.0-Instruct",
        "context_window": 131072,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "hf_token_required": False,
        "load_in_4bit": True,
        "inference_defaults": {
            "backend": "transformers",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.90,
            "max_model_len": None,
            "vllm_batch_size": 16,
        },
    },
    "ministral-3-14b": {
        "hf_id": "Aratako/Ministral-3-14B-Instruct-2512-BF16-TextOnly",
        "context_window": 131072,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "hf_token_required": False,
        "load_in_4bit": True,
        "inference_defaults": {
            "backend": "transformers",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.90,
            "max_model_len": None,
            "vllm_batch_size": 16,
        },
    },
    "mistral-nemo-12b": {
        "hf_id": "mistralai/Mistral-Nemo-Instruct-2407",
        "context_window": 128000,
        "trust_remote_code": False,
        "torch_dtype": torch.bfloat16,
        "hf_token_required": False,
        "inference_defaults": {
            "backend": "transformers",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.90,
            "max_model_len": None,
            "vllm_batch_size": 16,
        },
    },
    "aya-expanse-8b": {
        "hf_id": "CohereForAI/aya-expanse-8b",
        "context_window": 8192,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "hf_token_required": False,
        "inference_defaults": {
            "backend": "transformers",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.90,
            "max_model_len": None,
            "vllm_batch_size": 16,
        },
    },
    "bielik-11b": {
        "hf_id": "speakleash/Bielik-11B-v2.3-Instruct",
        "context_window": 8192,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "hf_token_required": False,
        "inference_defaults": {
            "backend": "transformers",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.90,
            "max_model_len": None,
            "vllm_batch_size": 16,
        },
    },
    "llama3.1-8b": {
        "hf_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "context_window": 128000,
        "trust_remote_code": False,
        "torch_dtype": torch.bfloat16,
        "hf_token_required": True,
        "inference_defaults": {
            "backend": "transformers",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.90,
            "max_model_len": None,
            "vllm_batch_size": 16,
        },
    },
}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_NEW_TOKENS = 2048
CONTEXT_RESERVED = MAX_NEW_TOKENS + 512  # tokens reserved for output + formatting
_ALWAYS_SKIP_CATEG = {"Empty", "Trash"}


# ---------------------------------------------------------------------------
# 2. Configuration Loader
# ---------------------------------------------------------------------------


def _unquote(value: str) -> str:
    """Strip one matched pair of surrounding quotes from a config value.

    Config files here are shell-flavoured KEY=VALUE, and both quoted and bare
    values occur in the wild — this repo's own llm_config.txt writes them bare,
    while generated configs (the cross-repo e2e workflow, deployment templates)
    quote paths out of shell habit. Without this, a quoted value keeps its quote
    characters and every path built from it is wrong by two bytes.

    That is not hypothetical: `VOCAB_PATH="/workspace/work/…json"` parsed to a
    path that could not exist, VocabularyManager fell through to auto-sync, and
    the run produced a single-term enum that rejected every correct answer the
    model gave. atrium-nlp-enrich's sibling parser (api_util/summarize_nt_udp.py)
    has always stripped quotes; the two repos consume configs written by the same
    hand, so differing on this is a trap rather than a design choice.

    Only a MATCHED outer pair is removed, so a Windows path or a value with an
    apostrophe inside survives untouched.
    """
    for quote in ('"', "'"):
        if len(value) >= 2 and value.startswith(quote) and value.endswith(quote):
            return value[1:-1]
    return value


def load_config(config_path: str = "llm_config.txt") -> Dict[str, str]:
    """Parse a KEY=VALUE config file, ignoring blank lines and # comments."""
    config: Dict[str, str] = {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}") from None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                config[key.strip()] = _unquote(value.strip())
    return config


# ---------------------------------------------------------------------------
# 3. Inference-parameter resolver
# ---------------------------------------------------------------------------

# Lowest-priority fallbacks — used only when neither llm_config.txt nor the
# model's inference_defaults specify a value.
_GLOBAL_INFERENCE_FALLBACKS: Dict[str, Any] = {
    "backend": "transformers",
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.90,
    "guided_decoding_backend": "xgrammar",
    "enable_prefix_caching": True,
    "vllm_batch_size": 16,
    "max_model_len": None,  # None → use model's native context window
    "cpu_offload_gb": 0,
}

# Map CONFIG_FILE_KEY → inference_defaults key (lower_snake_case)
_PARAM_KEYS: Dict[str, str] = {
    "BACKEND": "backend",
    "TENSOR_PARALLEL_SIZE": "tensor_parallel_size",
    "GPU_MEMORY_UTILIZATION": "gpu_memory_utilization",
    "GUIDED_DECODING_BACKEND": "guided_decoding_backend",
    "ENABLE_PREFIX_CACHING": "enable_prefix_caching",
    "VLLM_BATCH_SIZE": "vllm_batch_size",
    "MAX_MODEL_LEN": "max_model_len",
    "CPU_OFFLOAD_GB": "cpu_offload_gb",
}


def get_inference_defaults(
    model_key: str,
    user_config: Dict[str, str],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Resolve all inference parameters with three-tier priority:

      1. llm_config.txt (``user_config``)                  — highest priority
      2. ``MODEL_REGISTRY[model_key]["inference_defaults"]`` — model-specific
      3. ``_GLOBAL_INFERENCE_FALLBACKS``                    — lowest priority

    Additionally enforces:
      • If the model is ``vllm_only=True`` and the resolved backend is
        "transformers", the backend is silently upgraded to "vllm" (the
        transformers loader would fail anyway, so this gives a clear message
        at startup rather than a cryptic error later).

    Returns:
        resolved — fully typed dict ready for use in main():
                   str, int, float, bool, or None per parameter.
        sources  — dict mapping each CONFIG_KEY to its source string,
                   one of "config" | "model" | "global" | "forced".
                   Used by the startup summary to show where every value
                   came from so users know what to override.
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown MODEL_KEY '{model_key}'. Available: {', '.join(MODEL_REGISTRY.keys())}"
        ) from None

    spec = MODEL_REGISTRY[model_key]
    model_defaults = spec.get("inference_defaults", {})
    resolved: Dict[str, Any] = {}
    sources: Dict[str, str] = {}

    for cfg_key, def_key in _PARAM_KEYS.items():
        if cfg_key in user_config:
            raw = user_config[cfg_key]
            source = "config"
        elif def_key in model_defaults and model_defaults[def_key] is not None:
            raw = str(model_defaults[def_key])
            source = "model"
        elif def_key in model_defaults and model_defaults[def_key] is None:
            # Explicit None in model_defaults (max_model_len = use native)
            raw = ""
            source = "model"
        else:
            fb = _GLOBAL_INFERENCE_FALLBACKS.get(def_key)
            raw = str(fb) if fb is not None else ""
            source = "global"

        # Type coercion
        if def_key == "backend":
            resolved[cfg_key] = raw.lower()
        elif def_key == "cpu_offload_gb":
            # "auto" (issue #26) is sized at engine-load time from the model
            # footprint vs. detected VRAM; otherwise an explicit integer GB.
            if raw.strip().lower() == "auto":
                resolved[cfg_key] = "auto"
            else:
                try:
                    resolved[cfg_key] = int(raw) if raw else 0
                except ValueError:
                    raise ValueError(
                        f"CPU_OFFLOAD_GB must be an integer or 'auto', got {raw!r}"
                    ) from None
        elif def_key in ("tensor_parallel_size", "vllm_batch_size"):
            resolved[cfg_key] = int(raw) if raw else 0
        elif def_key == "gpu_memory_utilization":
            resolved[cfg_key] = float(raw) if raw else 0.90
        elif def_key == "enable_prefix_caching":
            resolved[cfg_key] = raw.lower() == "true" if raw else True
        elif def_key == "max_model_len":
            resolved[cfg_key] = int(raw) if raw else None
        else:
            resolved[cfg_key] = raw

        sources[cfg_key] = source

    # Enforce vllm_only constraint — upgrade silently rather than failing later
    if spec.get("vllm_only") and resolved["BACKEND"] != "vllm":
        print(
            f"[INFO] {model_key} is vllm_only — "
            f"upgrading BACKEND={resolved['BACKEND']} → vllm automatically."
        )
        resolved["BACKEND"] = "vllm"
        sources["BACKEND"] = "forced"

    # CPU_OFFLOAD_GB=auto is a vLLM mechanism (UVA zero-copy weight spill);
    # the transformers backend has no equivalent, so fall back to 0 there.
    # For models larger than GPU VRAM the supported answer is BACKEND=vllm.
    if resolved["CPU_OFFLOAD_GB"] == "auto" and resolved["BACKEND"] != "vllm":
        print(
            "[INFO] CPU_OFFLOAD_GB=auto applies to the vLLM backend only — using 0. "
            "For models larger than GPU VRAM set BACKEND=vllm."
        )
        resolved["CPU_OFFLOAD_GB"] = 0
        sources["CPU_OFFLOAD_GB"] = "forced"

    return resolved, sources


# ---------------------------------------------------------------------------
# 3b. CPU-offload auto-sizing — issue ufal/atrium-project#26
# ---------------------------------------------------------------------------
#
# Runs models whose weight footprint exceeds GPU VRAM by spilling the surplus
# into CPU RAM via vLLM's cpu_offload_gb (UVA zero-copy: the GPU stays the
# sole compute engine and reads offloaded weights through a unified address
# space, so no CPU cores are consumed — they stay free for the rest of the
# pipeline). CPU_OFFLOAD_GB=auto sizes the spill at engine load:
#
#   offload = footprint − (VRAM × gpu_memory_utilization − KV reserve) × TP
#
# The footprint is measured from the local HF cache when possible (exact for
# the quantization actually downloaded), the HF Hub API next, and the
# registry's weight_footprint_gb estimate last. Offloaded weights cross the
# PCIe bus on every forward pass — expect a 3–6× throughput penalty.
# Quantization (FP8/AWQ/GGUF) remains the primary strategy; offload is what
# lets a job run on a smaller-VRAM node at all.

_KV_RESERVE_GB_PER_GPU = 4.0  # VRAM kept for KV cache / activations, per GPU
_OFFLOAD_MARGIN_GB = 2.0  # safety margin added to the computed deficit
_RAM_HEADROOM_GB = 24.0  # CPU RAM kept free for the pipeline + OS

_WEIGHT_FILE_SUFFIXES = (".safetensors", ".bin", ".gguf")


def _available_ram_gb() -> Optional[float]:
    """MemAvailable from /proc/meminfo in GB; None where unreadable (non-Linux)."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024**2  # kB → GB
    except (OSError, ValueError, IndexError):
        pass
    return None


def _snapshot_weight_bytes(model_dir: Path, filename_pattern: Optional[str]) -> int:
    """Largest per-snapshot sum of weight-file sizes under one HF cache entry."""
    best = 0
    snapshots = model_dir / "snapshots"
    if not snapshots.is_dir():
        return 0
    for snap in snapshots.iterdir():
        if not snap.is_dir():
            continue
        total = 0
        for f in snap.rglob("*"):
            if f.suffix.lower() not in _WEIGHT_FILE_SUFFIXES:
                continue
            if filename_pattern and not fnmatch.fnmatch(f.name, filename_pattern):
                continue
            try:
                total += f.stat().st_size  # follows the blob symlink
            except OSError:
                continue
        best = max(best, total)
    return best


def _hub_weight_bytes(hf_id: str, filename_pattern: Optional[str]) -> int:
    """Total weight-file bytes reported by the HF Hub API; 0 when unreachable."""
    try:
        from huggingface_hub import HfApi  # noqa: E402

        siblings = HfApi().model_info(hf_id, files_metadata=True).siblings or []
    except Exception:
        return 0
    total = 0
    for s in siblings:
        if not s.size:
            continue
        name = Path(s.rfilename).name
        if not name.lower().endswith(_WEIGHT_FILE_SUFFIXES):
            continue
        if filename_pattern and not fnmatch.fnmatch(name, filename_pattern):
            continue
        total += s.size
    return total


def estimate_weight_footprint_gb(model_key: str) -> Optional[float]:
    """
    Best-effort weight footprint of *model_key* in GB, trying in order:

      1. Local HF cache (HF_HUB_CACHE / HF_HOME, read at call time) — exact
         for the files actually downloaded, quantized variants included.
      2. HF Hub API file metadata — needs network access; skipped offline.
      3. The registry's ``weight_footprint_gb`` estimate.

    Returns None (with a warning) when no source can answer — callers must
    degrade gracefully.
    """
    spec = MODEL_REGISTRY.get(model_key)
    if spec is None:
        return None
    hf_id = spec.get("hf_id", "")
    pattern = spec.get("filename")  # GGUF repos hold several quantizations

    # 1) Local HF cache. Env vars are read here (not at import) so SLURM jobs
    #    and tests that retarget HF_HOME are honoured.
    cache_dirs: List[Path] = []
    for env_var, sub in (("HF_HUB_CACHE", ""), ("HF_HOME", "hub"), ("HF_HOME", "")):
        root = os.environ.get(env_var)
        if root:
            cache_dirs.append(Path(root) / sub if sub else Path(root))
    try:
        from huggingface_hub.constants import HF_HUB_CACHE  # noqa: E402

        cache_dirs.append(Path(HF_HUB_CACHE))
    except Exception:
        pass

    entry_name = "models--" + hf_id.replace("/", "--")
    seen = set()
    for cache_dir in cache_dirs:
        try:
            resolved = cache_dir.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        nbytes = _snapshot_weight_bytes(cache_dir / entry_name, pattern)
        if nbytes > 0:
            gb = nbytes / 1024**3
            print(f"[OFFLOAD] {model_key} weight footprint: {gb:.1f} GB (local HF cache)")
            return gb

    # 2) HF Hub API
    nbytes = _hub_weight_bytes(hf_id, pattern)
    if nbytes > 0:
        gb = nbytes / 1024**3
        print(f"[OFFLOAD] {model_key} weight footprint: {gb:.1f} GB (HF Hub API)")
        return gb

    # 3) Registry estimate
    gb = spec.get("weight_footprint_gb")
    if gb:
        print(f"[OFFLOAD] {model_key} weight footprint: ~{float(gb):.0f} GB (registry estimate)")
        return float(gb)

    print(
        f"[WARN] Cannot estimate the weight footprint of {model_key} "
        f"(not in the local HF cache, Hub unreachable, no registry estimate)."
    )
    return None


def resolve_auto_cpu_offload(
    footprint_gb: float,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    per_gpu_vram_gb: float,
    ram_available_gb: Optional[float],
    kv_reserve_gb_per_gpu: float = _KV_RESERVE_GB_PER_GPU,
    margin_gb: float = _OFFLOAD_MARGIN_GB,
) -> int:
    """
    Pure budget math behind CPU_OFFLOAD_GB=auto — how many GB of weights must
    spill to CPU RAM for the model to fit the GPU weight budget.

    Returns 0 when the weights fit. Raises ValueError when even CPU RAM
    cannot absorb the deficit (message includes the SLURM --mem fix).
    """
    vram_budget_gb = (
        per_gpu_vram_gb * gpu_memory_utilization - kv_reserve_gb_per_gpu
    ) * tensor_parallel_size
    deficit_gb = footprint_gb - vram_budget_gb
    if deficit_gb <= 0:
        return 0
    offload_gb = math.ceil(deficit_gb + margin_gb)
    if ram_available_gb is not None and offload_gb > ram_available_gb - _RAM_HEADROOM_GB:
        raise ValueError(
            f"Model does not fit even with CPU offload: weights ≈ {footprint_gb:.0f} GB, "
            f"GPU weight budget ≈ {vram_budget_gb:.0f} GB "
            f"({tensor_parallel_size}× {per_gpu_vram_gb:.0f} GB × {gpu_memory_utilization:.2f} "
            f"− {kv_reserve_gb_per_gpu:.0f} GB/GPU KV reserve) → needs {offload_gb} GB in "
            f"CPU RAM, but only {ram_available_gb:.0f} GB is available "
            f"({_RAM_HEADROOM_GB:.0f} GB reserved for the pipeline/OS). "
            f"Request more memory in the SLURM job (--mem ≥ {offload_gb + 40}G), add GPUs "
            f"(TENSOR_PARALLEL_SIZE), or pick a smaller / more-quantized MODEL_KEY."
        )
    return offload_gb


def _size_cpu_offload(
    model_key: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    explicit_offload_gb: Optional[int] = None,
) -> int:
    """
    Resolve CPU_OFFLOAD_GB=auto (``explicit_offload_gb=None``) or sanity-check
    an explicit value. Returns the offload GB to pass to vLLM.

    Auto mode raises ValueError when the model cannot fit at all; with an
    explicit value the same condition only warns (the operator decided).
    """
    auto_mode = explicit_offload_gb is None

    if not torch.cuda.is_available():
        if auto_mode:
            print("[OFFLOAD] CPU_OFFLOAD_GB=auto: no CUDA devices visible — using 0.")
            return 0
        return explicit_offload_gb

    footprint_gb = estimate_weight_footprint_gb(model_key)
    if footprint_gb is None:
        if auto_mode:
            print(
                "[WARN] CPU_OFFLOAD_GB=auto could not determine the model footprint — "
                "using 0. Set CPU_OFFLOAD_GB=<int> manually if the model exceeds VRAM."
            )
            return 0
        return explicit_offload_gb

    per_gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    ram_available_gb = _available_ram_gb()

    try:
        needed_gb = resolve_auto_cpu_offload(
            footprint_gb=footprint_gb,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            per_gpu_vram_gb=per_gpu_vram_gb,
            ram_available_gb=ram_available_gb,
        )
    except ValueError:
        if auto_mode:
            raise
        print(
            f"[WARN] {model_key} (~{footprint_gb:.0f} GB of weights) is unlikely to fit: "
            f"even CPU RAM cannot absorb the VRAM deficit on this node. Expect OOM — "
            f"raise the job's --mem, add GPUs, or pick a smaller MODEL_KEY."
        )
        return explicit_offload_gb

    if auto_mode:
        budget_gb = (
            per_gpu_vram_gb * gpu_memory_utilization - _KV_RESERVE_GB_PER_GPU
        ) * tensor_parallel_size
        ram_str = (
            f"{ram_available_gb:.0f} GB RAM available"
            if ram_available_gb is not None
            else "RAM unknown"
        )
        print(
            f"[OFFLOAD] CPU_OFFLOAD_GB=auto → {needed_gb} GB  "
            f"(weights ≈ {footprint_gb:.0f} GB, GPU weight budget ≈ {budget_gb:.0f} GB on "
            f"{tensor_parallel_size}× {per_gpu_vram_gb:.0f} GB, {ram_str})"
        )
        if needed_gb > 0:
            print(
                f"[OFFLOAD] Ensure the job requests enough CPU RAM (SLURM: "
                f"--mem ≥ {needed_gb + 40}G). Offloaded weights cross PCIe every forward "
                f"pass — expect a 3–6× throughput reduction; fine for overnight batch runs."
            )
        return needed_gb

    if explicit_offload_gb < needed_gb:
        print(
            f"[WARN] {model_key} weights ≈ {footprint_gb:.0f} GB likely exceed the GPU "
            f"weight budget: CPU_OFFLOAD_GB={explicit_offload_gb} looks insufficient — "
            f"expect OOM. Suggested: CPU_OFFLOAD_GB={needed_gb} (or CPU_OFFLOAD_GB=auto)."
        )
    return explicit_offload_gb


# ---------------------------------------------------------------------------
# 4. Backend dependency preflight check
# ---------------------------------------------------------------------------


def _check_backend_deps(backend: str, model_key: str) -> None:
    """
    Verify that the required runtime libraries for *backend* are importable.

    Called in ``main()`` **before** the ``with logger:`` context so that a
    missing dependency is reported immediately — not swallowed by the logger's
    ``__exit__`` and silently printed only after "[paradata] Log written".

    Raises ``ImportError`` with an actionable fix message if a required library
    is absent.  Also warns (without raising) if the selected backend looks
    suboptimal for the model (e.g. vLLM requested but ``load_in_4bit=True``
    makes the transformers path cheaper).
    """
    spec = MODEL_REGISTRY.get(model_key, {})

    if backend == "vllm":
        try:
            import vllm  # noqa: F401
        except ModuleNotFoundError:
            raise ImportError(
                "\n"
                "  BACKEND=vllm is set but vLLM is not installed.\n"
                "\n"
                "  Install it (keep existing PyTorch):\n"
                "    pip install vllm --no-build-isolation\n"
                "\n"
                "  If pip tries to downgrade torch, pin the version:\n"
                "    pip install 'vllm>=0.8.0' --no-build-isolation\n"
                "\n"
                "  Or switch to the 4-bit transformers path instead:\n"
                "    Add  BACKEND=transformers  to llm_config.txt\n"
            ) from None

        # Pre-flight: check minimum vLLM version declared in MODEL_REGISTRY.
        # Catches known unsupported architectures *before* the engine load
        # attempt so the error is printed immediately rather than buried in
        # a 60-second startup sequence.
        min_ver = spec.get("min_vllm_version")
        if min_ver:
            try:
                import vllm as _vllm_mod  # noqa: E402

                _installed = getattr(_vllm_mod, "__version__", "0.0.0")

                # Simple tuple comparison — handles N.N.N and N.N.N.postM forms
                def _ver_tuple(v: str):
                    return tuple(int(x) for x in v.split(".")[:3] if x.split("post")[0].isdigit())

                if _ver_tuple(_installed) < _ver_tuple(min_ver):
                    raise RuntimeError(
                        f"\n"
                        f"  '{model_key}' requires vLLM >= {min_ver}, "
                        f"but {_installed!r} is installed.\n"
                        f"\n"
                        f"  The model's architecture is not registered in this vLLM version.\n"
                        f"\n"
                        f"  Fix:\n"
                        f"    pip install -U vllm --no-build-isolation\n"
                        f"\n"
                        f"  Supported architectures per version:\n"
                        f"    https://docs.vllm.ai/en/latest/models/supported_models.html\n"
                    ) from None
            except RuntimeError:
                raise
            except Exception as _ver_exc:
                print(f"[WARN] Could not verify vLLM version for {model_key}: {_ver_exc}")

    elif backend == "transformers":
        # Warn if the model would benefit from vLLM but won't get it
        if spec.get("vllm_only"):
            # Should have been caught by get_inference_defaults forced upgrade,
            # but guard here too just in case.
            raise ValueError(
                f"{model_key} is vllm_only but BACKEND=transformers was forced. "
                "Remove the BACKEND override from llm_config.txt or install vLLM."
            ) from None
        # Soft warning: BnB availability
        try:
            import bitsandbytes  # noqa: F401
        except ModuleNotFoundError:
            if spec.get("load_in_4bit"):
                print(
                    f"[WARN] bitsandbytes not found — {model_key} uses load_in_4bit=True. "
                    "Loading in full BF16 instead (requires more VRAM).\n"
                    "  Fix: pip install bitsandbytes"
                )


# ---------------------------------------------------------------------------
# 5. Transformers backend — model loader helpers
# ---------------------------------------------------------------------------


def _verify_quantization_effective(model: Any, model_key: str, spec: dict) -> None:
    """
    Sanity-check that BnB 4-bit quantization actually reduced memory footprint.

    Raises ``RuntimeError`` if the footprint ratio vs. BF16 baseline exceeds
    ``spec.get("max_quant_ratio", 0.65)``.

    Background: correctly-quantised 27–31 B models land around 0.54–0.60 because
    the embedding table and layernorm/lm-head weights are kept in BF16. The 0.65
    threshold gives comfortable headroom while catching genuine full-precision
    fallbacks (ratio ≈ 1.0). MoE models that cannot be quantised should be flagged
    with ``bnb_experts_broken: True`` in the registry rather than relying on this
    check.
    """
    if not spec.get("load_in_4bit"):
        return

    footprint_bytes = model.get_memory_footprint()
    footprint_gb = footprint_bytes / 1024**3
    total_params = sum(p.numel() for p in model.parameters())
    bf16_est_gb = total_params * 2 / 1024**3
    ratio = footprint_gb / bf16_est_gb if bf16_est_gb > 0 else 0.0

    print(
        f"[INFO] Model footprint: {footprint_gb:.1f} GB "
        f"(BF16 estimate: {bf16_est_gb:.1f} GB, ratio: {ratio:.2f})"
    )

    threshold = spec.get("max_quant_ratio", 0.65)
    if ratio > threshold:
        raise RuntimeError(
            f"Quantization ineffective for {model_key}: ratio {ratio:.2f} "
            f"exceeds threshold {threshold:.2f}. "
            f"Expected ≲0.35 for clean 4-bit, ≲0.65 for large dense models. "
            f"If this is a MoE model, set bnb_experts_broken=True and use vLLM."
        ) from None


def count_tokens(text: str, tokenizer: Any) -> int:
    """Count tokens for both HuggingFace tokenizers and llama.cpp models."""
    if hasattr(tokenizer, "tokenize") and not isinstance(tokenizer, _tu.PreTrainedTokenizerBase):
        # llama_cpp.Llama acts as both model and tokenizer
        return len(tokenizer.tokenize(text.encode("utf-8")))
    return len(tokenizer.encode(text))


def load_model_and_tokenizer(
    model_key: str,
    hf_token: Optional[str] = None,
) -> Tuple[Any, Any, dict]:
    """
    Load model and tokenizer via HuggingFace Transformers (or llama.cpp for GGUF).

    Returns:
        (model, tokenizer, spec)
        For GGUF models the same ``llama_cpp.Llama`` object fills both slots.

    Raises:
        ValueError  — unknown model key or vllm_only model requested with
                      the transformers backend.
        RuntimeError — quantization check failed.
        ImportError  — missing optional dependency (llama-cpp-python, autoawq).
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown MODEL_KEY '{model_key}'. Available: {', '.join(MODEL_REGISTRY.keys())}"
        ) from None

    spec = MODEL_REGISTRY[model_key]

    if spec.get("vllm_only"):
        raise ValueError(
            f"Model '{model_key}' is flagged vllm_only — it cannot be loaded via "
            f"the transformers backend. Set BACKEND=vllm in llm_config.txt.\n"
            f"Note: {spec.get('notes', '')}"
        ) from None

    hf_id = spec["hf_id"]

    # --- GGUF path (llama.cpp) ---
    if spec.get("is_gguf"):
        try:
            from llama_cpp import Llama  # noqa: E402
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is required for GGUF models:\n  pip install llama-cpp-python"
            ) from exc
        print(f"=== Loading GGUF via llama.cpp: {hf_id} ===")
        model = Llama.from_pretrained(
            repo_id=hf_id,
            filename=spec.get("filename", "*.gguf"),
            n_ctx=spec["context_window"],
            n_gpu_layers=-1,
            flash_attn=True,
            verbose=False,
        )
        return model, model, spec  # llama.cpp object serves as both

    # --- Guard: BnB 4-bit on broken MoE experts ---
    if spec.get("bnb_experts_broken") and spec.get("load_in_4bit"):
        raise RuntimeError(
            f"BnB 4-bit quantization is unsupported for '{model_key}' "
            f"(fused MoE expert blocks). Use BACKEND=vllm or a GGUF variant."
        ) from None

    print(f"=== Loading (transformers): {hf_id} ===")

    from transformers import BitsAndBytesConfig  # noqa: E402

    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        trust_remote_code=spec.get("trust_remote_code", False),
        token=hf_token or None,
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", None)

    is_awq = spec.get("is_awq", False)

    # --- AWQ path ---
    if is_awq:
        try:
            from awq import AutoAWQForCausalLM  # noqa: E402
        except ImportError as exc:
            raise ImportError(
                f"Model '{model_key}' requires autoawq:\n  pip install autoawq"
            ) from exc
        model = AutoAWQForCausalLM.from_quantized(
            hf_id,
            fuse_layers=False,
            device_map="auto",
            token=hf_token or None,
        )
        if not hasattr(model, "device"):
            model.device = next(model.parameters()).device
        import warnings  # noqa: E402

        warnings.filterwarnings("ignore", category=DeprecationWarning, module="awq")
        model.eval()
        return model, tokenizer, spec

    # --- Standard transformers path (BnB / plain fp16/bf16) ---
    bnb_config = None
    if spec.get("load_in_4bit"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=spec["torch_dtype"],
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True,
        )

    load_kwargs: Dict[str, Any] = dict(
        device_map="auto",
        dtype=spec["torch_dtype"],
        trust_remote_code=spec.get("trust_remote_code", False),
        token=hf_token or None,
        attn_implementation="sdpa",
    )
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config

    # Enforce explicit memory capping for large MoE clusters (to resolve meta tensors and memory offloading)
    if spec.get("is_moe") and spec.get("load_in_4bit"):
        print("[INFO] Applying explicit memory capping for large MoE.")
        num_gpus = torch.cuda.device_count() or 1
        max_memory = {i: "40GiB" for i in range(num_gpus)}
        max_memory["cpu"] = "100GiB"
        load_kwargs["max_memory"] = max_memory

    model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
    _verify_quantization_effective(model, model_key, spec)
    model.eval()
    return model, tokenizer, spec


# ---------------------------------------------------------------------------
# 6. vLLM backend — engine loader
# ---------------------------------------------------------------------------


def load_vllm_engine(
    model_key: str,
    hf_token: Optional[str] = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    guided_decoding_backend: str = "xgrammar",
    enable_prefix_caching: bool = True,
    max_model_len: Optional[int] = None,
    cpu_offload_gb: Union[int, str] = 0,
) -> Tuple[Any, Any, dict]:
    """
    Load a model via vLLM for high-throughput, multi-GPU inference.

    Key features enabled:
      • Tensor parallelism (``tensor_parallel_size``) — shard the model across
        multiple GPUs without manual pipeline cuts.
      • Automatic Prefix Caching (``enable_prefix_caching=True``) — the system
        prompt (which embeds the full TEATER vocabulary) is computed once per
        run; its KV-cache is reused for every line in the document. This is the
        primary throughput multiplier for this pipeline.
      • Native guided JSON decoding via xgrammar — no lmformatenforcer needed.
      • CPU weight offloading (``cpu_offload_gb > 0``) — keeps
        ``cpu_offload_gb`` GB of model weights in CPU RAM and transfers them
        to GPU on demand. Use on nodes whose GPU VRAM is insufficient for the
        model weights but whose CPU RAM is large enough (e.g. dll-4gpu3 or
        dll-8gpu both have 515 GB RAM). Reduces throughput by 3–6×; acceptable
        for overnight batch runs. Pass ``"auto"`` to size the offload here at
        load time from the measured weight footprint vs. detected VRAM and
        available CPU RAM (issue ufal/atrium-project#26); explicit integers
        are sanity-checked and a warning is printed when they look too small.

    Returns:
        (llm_engine, tokenizer, spec)
        ``tokenizer`` is the HuggingFace tokenizer obtained from
        ``llm_engine.get_tokenizer()``.

    Raises:
        ImportError  — vLLM is not installed.
        ValueError   — unknown model key or GGUF model requested.
        RuntimeWarning — tensor_parallel_size below model's recommended_tp.
    """
    try:
        from vllm import LLM  # noqa: E402
    except ImportError as exc:
        raise ImportError(
            "vLLM is not installed. Install it with:\n"
            "  pip install vllm\n"
            "Or switch to BACKEND=transformers in llm_config.txt."
        ) from exc

    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown MODEL_KEY '{model_key}'. Available: {', '.join(MODEL_REGISTRY.keys())}"
        ) from None

    spec = MODEL_REGISTRY[model_key]

    if spec.get("is_gguf"):
        raise ValueError(
            f"Model '{model_key}' is a GGUF file. "
            "vLLM requires a HuggingFace model ID or a local directory "
            "(not a GGUF filename). Use BACKEND=transformers for llama.cpp."
        ) from None

    hf_id = spec["hf_id"]
    recommended_tp = spec.get("recommended_tp", 1)
    if tensor_parallel_size < recommended_tp:
        print(
            f"[WARN] {model_key} recommends tensor_parallel_size={recommended_tp} "
            f"but got {tensor_parallel_size}. "
            f"You may encounter out-of-memory errors."
        )

    # Resolve CPU offload — "auto" is sized here from the model footprint vs.
    # detected VRAM (issue ufal/atrium-project#26); explicit integers get a
    # fit sanity check that warns instead of failing.
    if isinstance(cpu_offload_gb, str):
        if cpu_offload_gb.strip().lower() != "auto":
            raise ValueError(
                f"CPU_OFFLOAD_GB must be an integer or 'auto', got {cpu_offload_gb!r}"
            ) from None
        cpu_offload_gb = _size_cpu_offload(model_key, tensor_parallel_size, gpu_memory_utilization)
    else:
        cpu_offload_gb = _size_cpu_offload(
            model_key,
            tensor_parallel_size,
            gpu_memory_utilization,
            explicit_offload_gb=cpu_offload_gb,
        )

    # Resolve dtype string for vLLM
    dtype_map = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }
    dtype_str = dtype_map.get(spec.get("torch_dtype", torch.bfloat16), "bfloat16")
    if "fp8" in model_key.lower():
        dtype_str = "auto"  # Let vLLM detect native FP8

        # Warn if the current GPU lacks hardware FP8 tensor-core support.
        # FP8 matmuls are hardware-accelerated only on CC ≥ 8.9 (Ada / Hopper).
        # On older GPUs (A100 = CC 8.0, A40 = CC 8.6) vLLM falls back to BF16
        # computation, so the MEMORY benefit of FP8 weights is preserved but
        # there is NO throughput improvement over a native BF16 run.
        if torch.cuda.is_available():
            _cc_maj, _cc_min = torch.cuda.get_device_capability(0)
            _cc = _cc_maj * 10 + _cc_min  # e.g. 80 for A100, 86 for A40, 89 for L40
            if _cc < 89:
                _gpu_name = torch.cuda.get_device_properties(0).name
                print(
                    f"[WARN] FP8 hardware acceleration requires Compute Capability "
                    f"≥ 8.9 (Ada / Hopper). Current device: {_gpu_name} "
                    f"(CC {_cc_maj}.{_cc_min}). "
                    f"Weights will be stored as FP8 (memory budget preserved), "
                    f"but all matmuls execute in BF16 — no throughput gain over "
                    f"a BF16 model of the same size. "
                    f"Full FP8 speed requires H100, L40S, or RTX 4090-class hardware."
                )

    print(
        f"=== Loading via vLLM: {hf_id} ===\n"
        f"    dtype={dtype_str}, "
        f"tensor_parallel_size={tensor_parallel_size}, "
        f"gpu_memory_utilization={gpu_memory_utilization:.2f}, "
        f"enable_prefix_caching={enable_prefix_caching}, "
        f"guided_decoding_backend={guided_decoding_backend}"
    )
    if max_model_len:
        print(f"    max_model_len={max_model_len} (overriding model default)")
    if cpu_offload_gb > 0:
        print(
            f"    cpu_offload_gb={cpu_offload_gb} "
            f"(offloading {cpu_offload_gb} GB of weights to CPU RAM; "
            f"throughput will be reduced)"
        )

    engine_kwargs: Dict[str, Any] = dict(
        model=hf_id,
        tokenizer=hf_id,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=enable_prefix_caching,
        guided_decoding_backend=guided_decoding_backend,
        dtype=dtype_str,
        trust_remote_code=spec.get("trust_remote_code", False),
        # Separate tokenizer threads speed up batched prompt formatting
        tokenizer_pool_size=max(1, tensor_parallel_size),
    )

    if max_model_len:
        engine_kwargs["max_model_len"] = max_model_len

    if cpu_offload_gb > 0:
        engine_kwargs["cpu_offload_gb"] = cpu_offload_gb

    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)

    # Apply compat shims that must fire before LLM() touches the model config.
    _patch_vllm_rope_scaling_compat()

    try:
        engine_kwargs.pop("tokenizer_pool_size", None)
        llm_engine = LLM(**engine_kwargs)
    except ValueError as _exc:
        _msg = str(_exc)
        if "has no vLLM implementation" in _msg:
            # Parse architecture name from "XxxForYyy has no vLLM implementation…"
            _arch = _msg.split(" has no vLLM implementation")[0].strip()
            raise ValueError(
                f"\n"
                f"  Model architecture '{_arch}' is not supported by the installed\n"
                f"  vLLM version (v0.7.3 or earlier).\n"
                f"\n"
                f"  Fix — upgrade vLLM inside the virtual environment:\n"
                f"    pip install -U vllm --no-build-isolation\n"
                f"\n"
                f"  Supported architectures per version:\n"
                f"    https://docs.vllm.ai/en/latest/models/supported_models.html\n"
                f"\n"
                f"  Interim workaround for '{model_key}':\n"
                f"    Use the GGUF variant (if available in MODEL_REGISTRY) or\n"
                f"    switch to BACKEND=transformers with load_in_4bit=True.\n"
            ) from _exc
        if "rope_scaling should have a 'rope_type' key" in _msg:
            raise ValueError(
                f"\n"
                f"  vLLM rope_scaling compat patch did not apply in time for '{model_key}'.\n"
                f"  This is a known vLLM 0.7.x issue with newer model configs.\n"
                f"\n"
                f"  Fix:\n"
                f"    pip install -U vllm --no-build-isolation\n"
            ) from _exc
        raise

    # Obtain a HuggingFace-compatible tokenizer for chat-template formatting
    # and token counting (used in build_system_prompt).
    tokenizer = llm_engine.get_tokenizer()

    return llm_engine, tokenizer, spec


# ---------------------------------------------------------------------------
# 7. Line-quality filter
# ---------------------------------------------------------------------------


def _should_process_line(
    text: str,
    categ: str,
    quality_score: float,  # Added via correction
    include_non_text: bool,
    min_char_count: int,
    min_char_non_text: int,
    min_alpha_ratio_non_text: float,
) -> Tuple[bool, str]:

    # Strict range-based decision matrix for categorization
    if quality_score < 0.40:
        categ = "Trash"
    elif quality_score < 0.70 and categ != "Trash":
        categ = "Noisy"

    if not text:
        return False, "empty text"

    if categ in _ALWAYS_SKIP_CATEG:
        return False, f"categ={categ!r} (quality={quality_score})"

    if categ == "Non-text":
        if not include_non_text:
            return False, "Non-text excluded by config"
        char_count = len(text)
        if char_count < min_char_non_text:
            return False, f"Non-text too short ({char_count} < {min_char_non_text} chars)"
        alpha_count = sum(c.isalpha() for c in text)
        alpha_ratio = alpha_count / char_count if char_count else 0.0
        if alpha_ratio < min_alpha_ratio_non_text:
            return False, f"Non-text alpha ratio too low ({alpha_ratio:.2f})"
        return True, ""

    if not categ:
        if len(text) < min_char_count:
            return False, f"text too short ({len(text)} < {min_char_count} chars) [unknown categ]"
        return True, ""

    if len(text) < min_char_count:
        return False, f"text too short ({len(text)} < {min_char_count} chars)"

    return True, ""


def read_input_rows(input_path: Path) -> list[dict]:
    """Reads rows from a CSV or synthesizes lines from a TEITOK XML document."""
    if input_path.name.lower().endswith(".teitok.xml"):
        return [
            {
                "text": r["text"],
                "page_num": str(r.get("page_num", "")),
                "line_num": str(r.get("line_num", "")),
                "categ": "",  # Falls back to plain text handling
                "quality_score": 0.0,
            }
            for r in teitok_read.read_teitok_rows(str(input_path))
        ]
    with open(input_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# 8. Context-window builder
# ---------------------------------------------------------------------------


def get_context_window(rows: List[dict], center_idx: int, window: int = 2) -> str:
    """
    Build a text snippet around ``rows[center_idx]`` for the LLM user prompt.

    The target line is wrapped in ``<target_line>`` tags. Surrounding lines on
    the same page (within ±``window``) are included as plain context. For
    non-leading rows the first two non-noise document lines are prepended as a
    global header so the model can anchor the archaeological context.
    """
    _NOISE_CATEG = {"Empty", "Trash", "Non-text"}

    center_row = rows[center_idx]
    center_page = center_row.get("page_num", center_row.get("page", None))
    start = max(0, center_idx - window)
    end = min(len(rows), center_idx + window + 1)

    parts: List[str] = []

    # Global document header (prepended for non-leading lines)
    if center_idx > window + 2:
        parts.append("--- GLOBAL DOCUMENT HEADER ---")
        added = 0
        for row in rows:
            if row.get("categ", "").strip() not in _NOISE_CATEG:
                pg = row.get("page_num", row.get("page", 0))
                ln = row.get("line_num", row.get("line", 0))
                parts.append(f"    [P{pg} L{ln}] {row.get('text', '').strip()}")
                added += 1
                if added >= 2:
                    break

    # Most recent section heading before the target line
    current_section = "Unknown Section"
    for i in range(center_idx - 1, -1, -1):
        if rows[i].get("categ", "").strip() in {"Header", "Heading"}:
            current_section = rows[i].get("text", "").strip()
            break

    parts.append(f"--- CURRENT SECTION: {current_section} ---")
    parts.append("--- LOCAL CONTEXT WINDOW ---")

    for i in range(start, end):
        row = rows[i]
        row_page = row.get("page_num", row.get("page", None))
        categ = row.get("categ", "").strip()

        if row_page != center_page and i != center_idx:
            continue
        if i != center_idx and categ in _NOISE_CATEG:
            continue

        text = row.get("text", "").strip()
        pg = row_page
        ln = row.get("line_num", row.get("line", 0))

        if i == center_idx:
            parts.append(f"<target_line> >>> [P{pg} L{ln}] {text} </target_line>")
        else:
            parts.append(f"    [P{pg} L{ln}] {text}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 9. Chat-message formatting helper (shared by both backends)
# ---------------------------------------------------------------------------


def _format_chat_prompt(
    messages: List[dict],
    tokenizer: Any,
    model_key: str,
) -> str:
    """
    Apply the model's chat template to a list of messages and return the
    formatted prompt string.

    For Qwen3 / Qwen3.5 / Qwen3.6 models, thinking mode is suppressed via
    ``/no_think`` in the user message and ``enable_thinking=False`` in the
    template call (falls back gracefully if the tokenizer does not support
    the kwarg).
    """
    is_qwen3 = any(k in model_key.lower() for k in ("qwen3", "qwen-3.5", "qwen-3.6", "qwen3-235b"))

    if is_qwen3:
        # Work on copies to avoid mutating the caller's list
        messages = [m.copy() for m in messages]
        if messages and messages[-1].get("role") == "user":
            messages[-1]["content"] += "\n/no_think"

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **({"enable_thinking": False} if is_qwen3 else {}),
        )
    except TypeError:
        # Older tokenizer — does not accept enable_thinking
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


# ---------------------------------------------------------------------------
# 10. Transformers backend — document processor
# ---------------------------------------------------------------------------


def process_document(
    input_path: Path,
    csv_path: Path,
    model: Any,
    tokenizer: Any,
    parser: Any,  # lmformatenforcer.JsonSchemaParser
    prefix_function: Any,  # build_transformers_prefix_allowed_tokens_fn result
    system_prompt: str,
    EnrichmentModel: type,
    max_input_tokens: int,
    is_gguf: bool,
    model_key: str,
    include_non_text: bool = True,
    min_char_count: int = 3,
    min_char_non_text: int = 8,
    min_alpha_ratio_non_text: float = 0.40,
) -> Tuple[List[dict], Dict[str, int]]:
    """
    Run LLM inference over every qualifying line in a single CSV document
    using the HuggingFace Transformers backend.

    Returns:
        (enriched_lines, stats)

    ``stats`` keys:
        processed       — lines successfully enriched.
        skipped_filter  — lines dropped by _should_process_line.
        skipped_error   — lines that raised inference or validation errors.
        aborted         — True if the document was aborted after 10 consecutive
                          inference errors (the abort marker is written by the
                          caller, not here).
    """
    from pydantic import ValidationError  # noqa: E402

    file_id = doc_id_from_path(input_path)
    enriched_lines: List[dict] = []
    stats: Dict[str, int] = {
        "processed": 0,
        "skipped_filter": 0,
        "skipped_error": 0,
        "aborted": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_inference_seconds": 0.0,
    }
    consecutive_errors = 0
    page_num = line_num = 0  # ensure defined for error messages

    # with open(csv_path, "r", encoding="utf-8") as f:
    #     rows = list(csv.DictReader(f))
    rows = read_input_rows(input_path)

    if is_gguf:
        from llama_cpp import LogitsProcessorList  # noqa: E402
        from lmformatenforcer.integrations.llamacpp import (  # noqa: E402
            build_llamacpp_logits_processor,
        )

    pbar = tqdm(
        enumerate(rows),
        total=len(rows),
        desc=f"[{file_id}]",
        unit="row",
        leave=False,
        dynamic_ncols=True,
    )
    for i, row in pbar:
        inputs = output = None
        try:
            try:
                page_num = int(row.get("page_num", row.get("page", 0)))
                line_num = int(row.get("line_num", row.get("line", 0)))
            except (ValueError, TypeError):
                stats["skipped_filter"] += 1
                continue

            text_chunk = row.get("text", "").strip()
            categ = row.get("categ", "").strip()

            should_process, _ = _should_process_line(
                text_chunk,
                categ,
                include_non_text,
                min_char_count,
                min_char_non_text,
                min_alpha_ratio_non_text,
            )
            if not should_process:
                stats["skipped_filter"] += 1
                continue

            context_chunk = get_context_window(rows, i, window=2)
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"DOCUMENT CONTEXT:\n{context_chunk}\n\n"
                        "Task: Extract keywords and determine the TEATER category "
                        "ONLY for the line marked inside <target_line>."
                    ),
                },
            ]

            # --- Inference ---
            if is_gguf:
                lp = LogitsProcessorList([build_llamacpp_logits_processor(model, parser)])
                output = model.create_chat_completion(
                    messages=messages,
                    max_tokens=MAX_NEW_TOKENS,
                    temperature=0.0,
                    logits_processor=lp,
                )
                result_json = output["choices"][0]["message"]["content"]
                # Input prompt is opaque in llama.cpp; count only output tokens
                stats["total_output_tokens"] += len(model.tokenize(result_json.encode("utf-8")))
                # total_inference_seconds left at 0 for GGUF — llama.cpp has its own timing

            else:
                prompt = _format_chat_prompt(messages, tokenizer, model_key)
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_input_tokens,
                ).to(model.device)

                _t0 = time.monotonic()
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        prefix_allowed_tokens_fn=prefix_function,
                    )
                _t1 = time.monotonic()

                generated_tokens = output[0][inputs["input_ids"].shape[1] :]
                result_json = tokenizer.decode(generated_tokens, skip_special_tokens=True)

                input_tok = inputs["input_ids"].shape[1]
                output_tok = len(generated_tokens)
                stats["total_input_tokens"] += input_tok
                stats["total_output_tokens"] += output_tok
                stats["total_inference_seconds"] += _t1 - _t0

                del inputs, output
                torch.cuda.empty_cache()
                inputs = output = None

            # --- Parse & validate ---
            try:
                semantic_data = EnrichmentModel.model_validate_json(result_json)
            except ValidationError:
                try:
                    raw_dict = json.loads(result_json, strict=False)
                    if "confidence_score" in raw_dict:
                        try:
                            val = float(raw_dict["confidence_score"])
                            raw_dict["confidence_score"] = min(1.0, max(0.0, val))
                        except (ValueError, TypeError):
                            pass
                    semantic_data = EnrichmentModel.model_validate(raw_dict)
                except (json.JSONDecodeError, ValidationError) as exc:
                    print(
                        f"  [{file_id}] Persistent validation error P{page_num} L{line_num}: {exc}"
                    )
                    stats["skipped_error"] += 1
                    consecutive_errors += 1
                    if consecutive_errors >= 10:
                        stats["aborted"] = 1
                        print(
                            f"  [{file_id}] Aborting after {consecutive_errors} consecutive errors."
                        )
                        break
                    continue

            dump_data = semantic_data.model_dump()
            dump_data["teater_category"] = semantic_data.category_name()

            if dump_data.get("teater_category") == "Nerelevantní (meta-text)":
                dump_data["extracted_keywords_cs"] = []
                dump_data["extracted_keywords_en"] = []

            enriched_lines.append(
                {
                    "file_id": file_id,
                    "page": page_num,
                    "line": line_num,
                    "categ": categ,
                    "quality_score": float(row.get("quality_score") or 0.0),
                    "original_text": text_chunk,
                    "enrichment": dump_data,
                }
            )
            stats["processed"] += 1
            consecutive_errors = 0
            pbar.set_postfix(proc=stats["processed"], err=stats["skipped_error"], refresh=False)

        except Exception as exc:
            print(f"  [{file_id}] Inference error P{page_num} L{line_num}: {exc}")
            stats["skipped_error"] += 1
            consecutive_errors += 1

            if not is_gguf:
                if inputs is not None:
                    del inputs
                if output is not None:
                    del output
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if consecutive_errors >= 10:
                stats["aborted"] = 1
                print(f"  [{file_id}] Aborting after {consecutive_errors} consecutive errors.")
                break

    pbar.close()
    if stats["total_inference_seconds"] > 0 and stats["processed"] > 0:
        tps_in = stats["total_input_tokens"] / stats["total_inference_seconds"]
        tps_out = stats["total_output_tokens"] / stats["total_inference_seconds"]
        print(
            f"  [{file_id}] Speed: {tps_in:.0f} in tok/s, {tps_out:.0f} out tok/s "
            f"({stats['total_input_tokens']:,} in / {stats['total_output_tokens']:,} out tokens)"
        )
    return enriched_lines, stats


# ---------------------------------------------------------------------------
# 11. vLLM backend — batched document processor
# ---------------------------------------------------------------------------


def process_document_vllm(
    input_path: Path,
    csv_path: Path,
    llm_engine: Any,  # vllm.LLM
    tokenizer: Any,
    system_prompt: str,
    EnrichmentModel: type,  # Pydantic model; JSON schema derived internally
    max_input_tokens: int,
    model_key: str,
    batch_size: int = 16,
    include_non_text: bool = True,
    min_char_count: int = 3,
    min_char_non_text: int = 8,
    min_alpha_ratio_non_text: float = 0.40,
) -> Tuple[List[dict], Dict[str, int]]:
    """
    Run batched LLM inference over every qualifying line in a single CSV document
    using the vLLM backend.

    Differences from ``process_document`` (transformers backend):
      • Lines are submitted in mini-batches (``batch_size``) for high throughput.
      • Guided JSON decoding is handled natively by vLLM + xgrammar; no
        lmformatenforcer is needed.
      • With Automatic Prefix Caching (APC) enabled on the engine, the system
        prompt's KV-cache is shared across all lines in the document — this is
        the primary throughput multiplier.
      • Consecutive-error detection is evaluated after each mini-batch rather
        than after each line. If ≥10 errors accumulate across a batch, the
        document is aborted and stats["aborted"] is set.

    Returns:
        (enriched_lines, stats)  — same schema as ``process_document``.
    """
    try:
        from vllm import SamplingParams  # noqa: E402
        from vllm.sampling_params import GuidedDecodingParams  # noqa: E402
    except ImportError as exc:
        raise ImportError(
            "vLLM is required for process_document_vllm. Install with: pip install vllm"
        ) from exc

    from pydantic import ValidationError  # noqa: E402

    file_id = doc_id_from_path(input_path)
    enriched_lines: List[dict] = []
    stats: Dict[str, int] = {
        "processed": 0,
        "skipped_filter": 0,
        "skipped_error": 0,
        "aborted": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_inference_seconds": 0.0,
    }
    consecutive_errors = 0

    rows = read_input_rows(input_path)

    # Build the guided-decoding sampling params once and reuse for all batches.
    schema_dict = EnrichmentModel.model_json_schema()
    guided = GuidedDecodingParams(json_schema=schema_dict)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
        guided_decoding=guided,
    )

    # First pass: collect all qualifying lines with their row metadata.
    # This keeps the batch-submission logic clean and makes it easy to
    # correlate outputs back to rows.
    qualifying: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        try:
            page_num = int(row.get("page_num", row.get("page", 0)))
            line_num = int(row.get("line_num", row.get("line", 0)))
        except (ValueError, TypeError):
            stats["skipped_filter"] += 1
            continue

        text_chunk = row.get("text", "").strip()
        categ = row.get("categ", "").strip()

        should_process, _ = _should_process_line(
            text_chunk,
            categ,
            include_non_text,
            min_char_count,
            min_char_non_text,
            min_alpha_ratio_non_text,
        )
        if not should_process:
            stats["skipped_filter"] += 1
            continue

        context_chunk = get_context_window(rows, i, window=2)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"DOCUMENT CONTEXT:\n{context_chunk}\n\n"
                    "Task: Extract keywords and determine the TEATER category "
                    "ONLY for the line marked inside <target_line>."
                ),
            },
        ]

        qualifying.append(
            {
                "row_index": i,
                "page_num": page_num,
                "line_num": line_num,
                "text_chunk": text_chunk,
                "categ": categ,
                "quality_score": float(row.get("quality_score") or 0.0),
                "messages": messages,
            }
        )

    total_qualifying = len(qualifying)
    print(
        f"  [{file_id}] {total_qualifying} lines pass filter "
        f"({stats['skipped_filter']} skipped). "
        f"Submitting in batches of {batch_size}…"
    )

    # Second pass: submit mini-batches to vLLM.
    aborted = False
    total_batches = (total_qualifying + batch_size - 1) // batch_size
    batch_iter = tqdm(
        range(0, total_qualifying, batch_size),
        desc=f"[{file_id}]",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )
    for batch_start in batch_iter:
        if aborted:
            break

        batch = qualifying[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1

        # Format prompts using the model's chat template
        prompts: List[str] = []
        for item in batch:
            try:
                prompts.append(_format_chat_prompt(item["messages"], tokenizer, model_key))
            except Exception as fmt_err:
                tqdm.write(
                    f"  [{file_id}] Prompt-format error "
                    f"P{item['page_num']} L{item['line_num']}: {fmt_err}"
                )
                prompts.append("")  # placeholder; will fail gracefully below

        # Submit entire batch to vLLM in one call
        _t0 = time.monotonic()
        try:
            outputs = llm_engine.generate(prompts, sampling_params)
        except Exception as batch_err:
            # Batch-level failure (e.g. OOM, NCCL error) — count all lines as errors
            tqdm.write(f"  [{file_id}] Batch {batch_num}/{total_batches} failed: {batch_err}")
            stats["skipped_error"] += len(batch)
            consecutive_errors += len(batch)
            if consecutive_errors >= 10:
                stats["aborted"] = 1
                aborted = True
                tqdm.write(
                    f"  [{file_id}] Aborting after batch failure "
                    f"({consecutive_errors} consecutive errors)."
                )
            continue
        _t1 = time.monotonic()

        batch_time = _t1 - _t0
        batch_in = sum(len(o.prompt_token_ids) for o in outputs)
        batch_out = sum(len(o.outputs[0].token_ids) for o in outputs if o.outputs)
        stats["total_input_tokens"] += batch_in
        stats["total_output_tokens"] += batch_out
        stats["total_inference_seconds"] += batch_time
        tps_in = batch_in / batch_time if batch_time > 0 else 0
        tps_out = batch_out / batch_time if batch_time > 0 else 0
        tqdm.write(
            f"  [{file_id}] Batch {batch_num}/{total_batches} "
            f"({len(batch)} lines, {batch_in:,}→{batch_out:,} tok) "
            f"{tps_in:.0f} in tok/s  {tps_out:.0f} out tok/s"
        )

        # Parse each output
        for item, vllm_output in zip(batch, outputs, strict=True):
            page_num = item["page_num"]
            line_num = item["line_num"]

            try:
                result_json = vllm_output.outputs[0].text
            except (IndexError, AttributeError) as exc:
                tqdm.write(f"  [{file_id}] Empty output P{page_num} L{line_num}: {exc}")
                stats["skipped_error"] += 1
                consecutive_errors += 1
                if consecutive_errors >= 10:
                    stats["aborted"] = 1
                    aborted = True
                    tqdm.write(
                        f"  [{file_id}] Aborting after {consecutive_errors} consecutive errors."
                    )
                    break
                continue

            # Validate against Pydantic model
            try:
                semantic_data = EnrichmentModel.model_validate_json(result_json)
            except ValidationError:
                try:
                    raw_dict = json.loads(result_json, strict=False)
                    if "confidence_score" in raw_dict:
                        try:
                            val = float(raw_dict["confidence_score"])
                            raw_dict["confidence_score"] = min(1.0, max(0.0, val))
                        except (ValueError, TypeError):
                            pass
                    semantic_data = EnrichmentModel.model_validate(raw_dict)
                except (json.JSONDecodeError, ValidationError) as exc:
                    tqdm.write(
                        f"  [{file_id}] Persistent validation error P{page_num} L{line_num}: {exc}"
                    )
                    stats["skipped_error"] += 1
                    consecutive_errors += 1
                    if consecutive_errors >= 10:
                        stats["aborted"] = 1
                        aborted = True
                        tqdm.write(
                            f"  [{file_id}] Aborting after {consecutive_errors} consecutive errors."
                        )
                        break
                    continue

            dump_data = semantic_data.model_dump()
            dump_data["teater_category"] = semantic_data.category_name()

            if dump_data.get("teater_category") == "Nerelevantní (meta-text)":
                dump_data["extracted_keywords_cs"] = []
                dump_data["extracted_keywords_en"] = []

            enriched_lines.append(
                {
                    "file_id": file_id,
                    "page": page_num,
                    "line": line_num,
                    "categ": item["categ"],
                    "quality_score": item["quality_score"],
                    "original_text": item["text_chunk"],
                    "enrichment": dump_data,
                }
            )
            stats["processed"] += 1
            consecutive_errors = 0

    batch_iter.close()
    if stats["total_inference_seconds"] > 0 and stats["processed"] > 0:
        tps_in = stats["total_input_tokens"] / stats["total_inference_seconds"]
        tps_out = stats["total_output_tokens"] / stats["total_inference_seconds"]
        print(
            f"  [{file_id}] Speed: {tps_in:.0f} in tok/s, {tps_out:.0f} out tok/s "
            f"({stats['total_input_tokens']:,} in / {stats['total_output_tokens']:,} out tokens)"
        )
    return enriched_lines, stats
