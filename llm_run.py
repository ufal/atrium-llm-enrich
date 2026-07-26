"""
llm_run.py — Entry point for the LLM Semantic Enrichment Pipeline.

Reads llm_config.txt, initialises the model and vocabulary, then iterates
over every CSV file in INPUT_DIR and writes per-document JSON enrichment
files to OUTPUT_DIR.
"""

__import__("pysqlite3")
import sys  # noqa: E402

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import datetime  # noqa: E402
import enum  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Tuple  # noqa: E402

import torch  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from tqdm import tqdm  # noqa: E402

import llm_utils  # noqa: E402  (side-effect: env-var guard + compat patches)
from atrium_paradata import ParadataLogger  # noqa: E402
from llm_client_shared import write_document_record  # noqa: E402
from llm_utils import (  # noqa: E402
    CONTEXT_RESERVED,
    _check_backend_deps,
    count_tokens,
    get_inference_defaults,
    load_config,
    load_model_and_tokenizer,
    load_vllm_engine,
    log_gpu_info,
    log_gpu_memory,
    process_document,
    process_document_vllm,
)
from vocab_manager import VocabularyManager  # noqa: E402

_EXAMPLES_FOOTER = (
    "\nEXAMPLES:\n\n"
    'Input line: "Výzkum odhalil základy gotického kostela ze 14. '
    'století."\n'
    "Correct output:\n"
    "{\n"
    '  "extracted_keywords_cs": ["základy", "gotický kostel"],\n'
    '  "extracted_keywords_en": ["foundations", "Gothic church"],\n'
    '  "teater_category": "kostel",\n'
    '  "confidence_score": 0.92\n'
    "}\n\n"
    'Input line: "Praha, dne 6. října 1956, Dr. Solle"\n'
    "Correct output:\n"
    "{\n"
    '  "extracted_keywords_cs": [],\n'
    '  "extracted_keywords_en": [],\n'
    '  "teater_category": "Nerelevantní (meta-text)",\n'
    '  "confidence_score": 1.0\n'
    "}\n"
)

DOCUMENT_JSON_DIR = os.environ.get("DOCUMENT_JSON_DIR", "document_records")


def build_schema(term_names: List[str]) -> type:
    if not term_names:
        raise ValueError("term_names is empty — vocabulary failed to load or was fully truncated.")

    TermEnum = enum.Enum("TermEnum", {f"term_{i}": name for i, name in enumerate(term_names)})

    class ConstrainedEnrichment(BaseModel):
        extracted_keywords_cs: List[str] = Field(
            ...,
            description=(
                "Key Czech archaeological terms, methods, or objects found ONLY in "
                "the text marked with (>>>). "
                "DO NOT copy terms from the THEMATIC VOCABULARY list. "
                "If no relevant archaeological terms appear in the target line, "
                "return []. "
                "If teater_category is 'Nerelevantní (meta-text)', MUST be []. "
                "Do not extract names of researchers or administrative words. "
                "Prefer normalised multi-word phrases over isolated single words."
            ),
        )
        extracted_keywords_en: List[str] = Field(
            ...,
            description=(
                "Accurate English translations of extracted_keywords_cs. "
                "Do not copy Czech words unchanged."
            ),
        )
        teater_category: TermEnum = Field(
            ...,
            description="The single most relevant category from the thematic vocabulary.",
        )
        confidence_score: float = Field(
            ...,
            ge=0.0,
            le=1.0,
            description=(
                "Confidence that the selected teater_category is correct. "
                "1.0 — unambiguous match, no interpretation required. "
                "0.7–0.9 — reasonable but non-obvious match. "
                "0.5–0.7 — multiple categories could apply. "
                "< 0.5 — forced guess. "
                "Do NOT output 1.0 uniformly — this field is used for filtering."
            ),
        )

        def category_name(self) -> str:
            return self.teater_category.value

    return ConstrainedEnrichment


def build_system_prompt(
    vocab_data: dict,
    tokenizer: Any,
    max_tokens: int,
    skip_truncation: bool = False,
) -> Tuple[str, List[str]]:
    header = (
        "You are an expert archaeological data extractor. "
        "Analyze the MARKED LINE enclosed in <target_line> ... </target_line> "
        "within its surrounding document context.\n"
        "1. Extract ONLY archaeological entities, features, periods, or materials "
        "from the marked line. "
        "Do NOT extract names of researchers, dates, conjunctions, or "
        "administrative words.\n"
        "2. Select the SINGLE most relevant category from the thematic vocabulary "
        "list below.\n"
        "CRITICAL: If the marked line is purely administrative, a table of contents, "
        "a generic heading (e.g. page numbers, titles, author names, 'Práce:', "
        "'Obsah:', literature references) or lacks direct archaeological context, "
        "you MUST select 'Nerelevantní (meta-text)'.\n"
        "NEVER select a country name, language name, or geographic region name "
        "as the teater_category for any line — including administrative lines. "
        "For any line that lacks direct archaeological significance, "
        "you MUST use 'Nerelevantní (meta-text)'.\n"
        "When extracting keywords, normalize obvious OCR artifacts and typos to "
        "their correct Czech forms. "
        "Do NOT include garbled tokens or split words as keywords. "
        "Prefer the normalized phrase over the raw OCR text.\n"
        "You MUST use the exact Czech term as written in the vocabulary.\n"
        "You MUST respond ONLY with a valid JSON object matching the requested "
        "schema.\n\n"
        "THEMATIC VOCABULARY:\n"
    )

    raw_terms: List[dict] = []
    raw_terms.append(
        {
            "theme": "Administrative / Meta",
            "cs": "Nerelevantní (meta-text)",
            "en": "Irrelevant / Meta-text",
        }
    )

    for theme, data in vocab_data.items():
        if theme.lower() == "other":
            continue
        if isinstance(data, dict):
            if "keywords" in data and isinstance(data["keywords"], dict):
                cs_list = data["keywords"].get("cs", [])
                en_list = data["keywords"].get("en", [])
                for i, cs_key in enumerate(cs_list):
                    en = en_list[i] if i < len(en_list) else cs_key
                    raw_terms.append({"theme": theme, "cs": cs_key, "en": en})
            else:
                for cs_key, pair in data.items():
                    en = pair.get("en", cs_key) if isinstance(pair, dict) else cs_key
                    raw_terms.append({"theme": theme, "cs": cs_key, "en": en})

    prioritised = raw_terms

    def _build_candidate_prompt(term_list: List[dict], other_cap: int = 15) -> str:
        themes: Dict[str, List[str]] = {}
        other_terms: List[dict] = []

        for t in term_list:
            if t["theme"] == "Other":
                other_terms.append(t)
            else:
                themes.setdefault(t["theme"], []).append(f"{t['cs']} ({t['en']})")

        prompt = header
        for theme_name, lines in themes.items():
            prompt += f"\n--- {theme_name} ---\n"
            prompt += "\n".join(f"- {line}" for line in lines) + "\n"

        if other_terms:
            prompt += "\n--- Other (Misc) ---\n"
            prompt += "\n".join(f"- {t['cs']} ({t['en']})" for t in other_terms[:other_cap]) + "\n"

        prompt += _EXAMPLES_FOOTER
        return prompt

    full_prompt = _build_candidate_prompt(prioritised)
    token_count = count_tokens(full_prompt, tokenizer)

    _header_tok = count_tokens(header, tokenizer)
    _examples_tok = count_tokens(_EXAMPLES_FOOTER, tokenizer)
    _vocab_tok = token_count - _header_tok - _examples_tok
    print(
        f"[vocab] {len(prioritised)} terms, {token_count} tokens total "
        f"(header: {_header_tok}, vocabulary: {_vocab_tok}, examples: {_examples_tok})"
    )

    if skip_truncation:
        print(
            "[vocab] Prefix caching enabled — skipping truncation, "
            f"injecting full vocabulary ({token_count} tokens)."
        )
        return full_prompt, [t["cs"] for t in prioritised]

    if token_count <= max_tokens:
        print("[vocab] Full vocabulary fits within token budget.")
        return full_prompt, [t["cs"] for t in prioritised]

    print(
        f"[WARN] Vocabulary ({token_count} tokens) exceeds budget "
        f"({max_tokens}). Binary-searching for largest fitting prefix…"
    )

    lo, hi = 0, len(prioritised)
    while lo < hi - 1:
        mid = (lo + hi) // 2
        candidate = _build_candidate_prompt(prioritised[:mid])
        if count_tokens(candidate, tokenizer) <= max_tokens:
            lo = mid
        else:
            hi = mid

    surviving_terms = prioritised[:lo]
    surviving_prompt = _build_candidate_prompt(surviving_terms)
    surviving_cs = [t["cs"] for t in surviving_terms]

    print(
        f"[vocab] Truncated to {len(surviving_cs)} terms "
        f"({count_tokens(surviving_prompt, tokenizer)} tokens)."
    )
    return surviving_prompt, surviving_cs


def _write_abort_marker(
    out_file: Path,
    stats: Dict[str, int],
    reason: str = "10 consecutive inference errors",
) -> None:
    abort_file = out_file.with_suffix("").with_suffix(".abort.json")
    payload = {
        "aborted": True,
        "abort_reason": reason,
        "processed_before_abort": stats.get("processed", 0),
        "errors_before_abort": stats.get("skipped_error", 0),
        "timestamp_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
    }
    with open(abort_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  -> Abort marker written: {abort_file.name}")


def main(config_path: str = "llm_config.txt") -> None:
    config = load_config(config_path)

    MODEL_KEY = config.get("MODEL_KEY", "qwen-3.6-27b-it")
    HF_TOKEN = config.get("HF_TOKEN", os.environ.get("HF_TOKEN", None))
    INPUT_DIR = Path(config.get("INPUT_DIR", "data_samples/DOC_LINE_LANG_CLASS"))
    VOCAB_PATH = config.get("VOCAB_PATH", "data_samples/teater_nested_vocab.json")
    PARADATA_DIR = config.get("PARADATA_DIR", "paradata")
    _document_json_dir_raw = config.get("DOCUMENT_JSON_DIR", "").strip()
    DOCUMENT_JSON_DIR = Path(_document_json_dir_raw) if _document_json_dir_raw else None

    _base_out = Path(config.get("OUTPUT_DIR", "data_samples/KW_PER_DOC_LLM"))
    _model_suffix = MODEL_KEY.replace(".", "").replace("-", "_")
    OUTPUT_DIR = _base_out.parent / f"{_base_out.name}_{_model_suffix}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    INCLUDE_NON_TEXT = config.get("INCLUDE_NON_TEXT", "true").lower() == "true"
    MIN_CHAR_COUNT = int(config.get("MIN_CHAR_COUNT", "3"))
    MIN_CHAR_NON_TEXT = int(config.get("MIN_CHAR_NON_TEXT", "8"))
    MIN_ALPHA_RATIO_NON_TEXT = float(config.get("MIN_ALPHA_RATIO_NON_TEXT", "0.40"))

    infer, sources = get_inference_defaults(MODEL_KEY, config)

    BACKEND = infer["BACKEND"]
    TENSOR_PARALLEL_SIZE = infer["TENSOR_PARALLEL_SIZE"]
    GPU_MEMORY_UTILIZATION = infer["GPU_MEMORY_UTILIZATION"]
    GUIDED_DECODING_BACKEND = infer["GUIDED_DECODING_BACKEND"]
    ENABLE_PREFIX_CACHING = infer["ENABLE_PREFIX_CACHING"]
    VLLM_BATCH_SIZE = infer["VLLM_BATCH_SIZE"]
    MAX_MODEL_LEN = infer["MAX_MODEL_LEN"]
    CPU_OFFLOAD_GB = infer["CPU_OFFLOAD_GB"]

    if BACKEND not in {"transformers", "vllm"}:
        raise ValueError(f"Unknown BACKEND='{BACKEND}'. Must be 'transformers' or 'vllm'.")

    _SRC_LABEL = {
        "config": "← llm_config.txt",
        "model": "  (model default)",
        "global": "  (global default)",
        "forced": "  (enforced — model is vllm_only)",
    }

    print(
        f"\n=== LLM Semantic Enrichment Pipeline ===\n"
        f"    model:   {MODEL_KEY}\n"
        f"    output:  {OUTPUT_DIR}\n"
        + (
            f"    document pairing: {DOCUMENT_JSON_DIR}  "
            "(atrium_document.py — reads/writes <doc_id>.document.json)\n"
            if DOCUMENT_JSON_DIR is not None
            else ""
        )
    )
    print("  Inference parameters:")
    print(f"    {'BACKEND':<26} = {BACKEND:<12}  {_SRC_LABEL[sources['BACKEND']]}")
    if BACKEND == "vllm":
        for key, val in [
            ("TENSOR_PARALLEL_SIZE", TENSOR_PARALLEL_SIZE),
            ("GPU_MEMORY_UTILIZATION", f"{GPU_MEMORY_UTILIZATION:.2f}"),
            ("VLLM_BATCH_SIZE", VLLM_BATCH_SIZE),
            ("MAX_MODEL_LEN", MAX_MODEL_LEN if MAX_MODEL_LEN else "(model native)"),
            ("CPU_OFFLOAD_GB", CPU_OFFLOAD_GB),
            ("ENABLE_PREFIX_CACHING", ENABLE_PREFIX_CACHING),
            ("GUIDED_DECODING_BACKEND", GUIDED_DECODING_BACKEND),
        ]:
            print(f"    {key:<26} = {str(val):<12}  {_SRC_LABEL[sources.get(key, 'global')]}")
    print(
        "\n  To override any value add it to llm_config.txt. Values marked '← llm_config.txt' are already user-set.\n"
    )

    _check_backend_deps(BACKEND, MODEL_KEY)
    log_gpu_info()

    logger = ParadataLogger(
        program="llm-enrich",
        config={
            **config,
            "output_dir_resolved": str(OUTPUT_DIR),
            "backend": BACKEND,
            "include_non_text": INCLUDE_NON_TEXT,
            "min_char_count": MIN_CHAR_COUNT,
            "min_char_non_text": MIN_CHAR_NON_TEXT,
            "min_alpha_ratio_non_text": MIN_ALPHA_RATIO_NON_TEXT,
            "document_json_dir": str(DOCUMENT_JSON_DIR) if DOCUMENT_JSON_DIR else None,
        },
        paradata_dir=PARADATA_DIR,
        output_types=["json"],
    )

    with logger:
        vocab_mgr = VocabularyManager(vocab_path=VOCAB_PATH)
        vocab_data = vocab_mgr.load()
        total_terms = sum(
            len(v.get("keywords", {}).get("cs", []))
            if isinstance(v, dict) and "keywords" in v
            else len(v)
            for v in vocab_data.values()
            if isinstance(v, dict)
        )
        if total_terms == 0:
            raise RuntimeError(
                "Vocabulary is empty. Run vocab_manager.py on a node with internet access first."
            )
        print(f"=== Vocabulary: {total_terms} terms in {len(vocab_data)} broad categories ===")

        try:
            if BACKEND == "vllm":
                llm_engine, tokenizer, spec = load_vllm_engine(
                    model_key=MODEL_KEY,
                    hf_token=HF_TOKEN,
                    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
                    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                    guided_decoding_backend=GUIDED_DECODING_BACKEND,
                    enable_prefix_caching=ENABLE_PREFIX_CACHING,
                    max_model_len=MAX_MODEL_LEN,
                    cpu_offload_gb=CPU_OFFLOAD_GB,
                )
                model = None
                is_gguf = False
                prefix_function = None
                parser = None
            else:
                model, tokenizer, spec = load_model_and_tokenizer(MODEL_KEY, HF_TOKEN)
                llm_engine = None
                is_gguf = spec.get("is_gguf", False)
        except Exception as exc:
            print(f"\n[ERROR] Model loading failed: {type(exc).__name__}: {exc}\n", flush=True)
            raise

        log_gpu_memory(label="after model load")
        max_input_tokens = spec["context_window"] - CONTEXT_RESERVED

        skip_trunc = BACKEND == "vllm" and ENABLE_PREFIX_CACHING
        system_prompt, surviving_terms = build_system_prompt(
            vocab_data,
            tokenizer,
            max_tokens=max_input_tokens,
            skip_truncation=skip_trunc,
        )
        EnrichmentModel = build_schema(surviving_terms)

        if BACKEND == "vllm":
            print("=== vLLM guided decoding: JSON schema registered ===")
        else:
            print("=== Compiling JSON Schema State Machine (lmformatenforcer) ===")
            llm_utils._patch_tokenizer_compat()
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.integrations.transformers import (
                build_transformers_prefix_allowed_tokens_fn,
            )

            parser = JsonSchemaParser(EnrichmentModel.model_json_schema())
            prefix_function = (
                None if is_gguf else build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
            )

        print("=== Pipeline ready ===")

        input_files = sorted(
            p
            for p in INPUT_DIR.iterdir()
            if p.suffix.lower() == ".csv" or p.name.lower().endswith(".teitok.xml")
        )
        total_processed = 0
        total_errors = 0
        total_aborted = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_inference_seconds = 0.0

        for input_file in tqdm(input_files, desc="Documents", unit="doc", dynamic_ncols=True):
            doc_id = llm_utils.teitok_read.doc_id_from_path(input_file)
            out_file = OUTPUT_DIR / f"{doc_id}_enriched.json"

            if out_file.exists():
                tqdm.write(f"[skip] {input_file.name} — output already exists.")
                logger.log_skip(input_file.name, "already_exists")
                continue

            tqdm.write(f"\nProcessing: {input_file.name} …")

            try:
                if BACKEND == "vllm":
                    enriched_results, doc_stats = process_document_vllm(
                        input_path=input_file,  # WAS: csv_path=csv_file
                        llm_engine=llm_engine,
                        tokenizer=tokenizer,
                        system_prompt=system_prompt,
                        EnrichmentModel=EnrichmentModel,
                        max_input_tokens=max_input_tokens,
                        model_key=MODEL_KEY,
                        batch_size=VLLM_BATCH_SIZE,
                        include_non_text=INCLUDE_NON_TEXT,
                        min_char_count=MIN_CHAR_COUNT,
                        min_char_non_text=MIN_CHAR_NON_TEXT,
                        min_alpha_ratio_non_text=MIN_ALPHA_RATIO_NON_TEXT,
                    )
                else:
                    enriched_results, doc_stats = process_document(
                        csv_path=input_file,
                        model=model,
                        tokenizer=tokenizer,
                        parser=parser,
                        prefix_function=prefix_function,
                        system_prompt=system_prompt,
                        EnrichmentModel=EnrichmentModel,
                        max_input_tokens=max_input_tokens,
                        is_gguf=is_gguf,
                        model_key=MODEL_KEY,
                        include_non_text=INCLUDE_NON_TEXT,
                        min_char_count=MIN_CHAR_COUNT,
                        min_char_non_text=MIN_CHAR_NON_TEXT,
                        min_alpha_ratio_non_text=MIN_ALPHA_RATIO_NON_TEXT,
                    )

                was_aborted = bool(doc_stats.get("aborted"))
                total_processed += doc_stats["processed"]
                total_errors += doc_stats["skipped_error"]
                if was_aborted:
                    total_aborted += 1
                total_input_tokens += doc_stats.get("total_input_tokens", 0)
                total_output_tokens += doc_stats.get("total_output_tokens", 0)
                total_inference_seconds += doc_stats.get("total_inference_seconds", 0.0)

                print(
                    f"  processed={doc_stats['processed']}, "
                    f"skipped_filter={doc_stats['skipped_filter']}, "
                    f"errors={doc_stats['skipped_error']}" + (" [ABORTED]" if was_aborted else "")
                )

                if enriched_results:
                    with open(out_file, "w", encoding="utf-8") as out_f:
                        json.dump(enriched_results, out_f, indent=4, ensure_ascii=False)
                    print(f"  -> {len(enriched_results)} records → {out_file.name}")
                    logger.log_success("json", count=1)
                    logger.log_document_success()

                    if DOCUMENT_JSON_DIR is not None:
                        write_document_record(
                            doc_id,
                            enriched_results,
                            DOCUMENT_JSON_DIR,
                            run_id=logger._run_id,
                            enriched_path=out_file,
                            license_detail=logger.get_license_block(),
                        )
                else:
                    logger.log_skip(
                        input_file.name,
                        "No lines passed quality filter or all inference calls failed.",
                    )

                if was_aborted:
                    _write_abort_marker(
                        out_file=out_file, stats=doc_stats, reason="10 consecutive inference errors"
                    )

            except Exception as exc:
                print(f"  Critical error on {input_file.name}: {exc}")
                logger.log_skip(input_file.name, str(exc))

            finally:
                if BACKEND == "transformers" and not is_gguf:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        already_done = sum(1 for s in logger._skipped if s.get("reason") == "already_exists")
        true_failures = sum(1 for s in logger._skipped if s.get("reason") != "already_exists")
        _avg_in = total_input_tokens / total_inference_seconds if total_inference_seconds > 0 else 0
        _avg_out = (
            total_output_tokens / total_inference_seconds if total_inference_seconds > 0 else 0
        )
        print(
            f"\n=== Run complete ===\n"
            f"    lines enriched:          {total_processed}\n"
            f"    inference errors:        {total_errors}\n"
            f"    aborted documents:       {total_aborted}\n"
            f"    files processed:         {len(input_files)}\n"  # CHANGED
            f"    skipped (already done):  {already_done}\n"
            f"    skipped (errors):        {true_failures}\n"
            f"    total input tokens:      {total_input_tokens:,}\n"
            f"    total output tokens:     {total_output_tokens:,}\n"
            f"    avg speed:               {_avg_in:.0f} in tok/s, {_avg_out:.0f} out tok/s"
        )
        logger.finalize(input_total=len(input_files))  # CHANGED


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "llm_config.txt"
    main(config_path)
