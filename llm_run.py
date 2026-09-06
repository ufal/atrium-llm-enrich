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
from typing import Any, Dict, List, Optional, Set, Tuple  # noqa: E402

import torch  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from tqdm import tqdm  # noqa: E402

import llm_utils  # noqa: E402  (side-effect: env-var guard + compat patches)
import prompt_template  # noqa: E402
from atrium_paradata import ParadataLogger  # noqa: E402
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
from vocab_manager import VocabularyManager, vocabulary_provenance  # noqa: E402


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


IdList = List[Dict[str, str]]


def _id_lookup_and_strip_map(terms: List[dict]) -> Tuple[Dict[str, IdList], Dict[str, str]]:
    """From the final surviving term list, build the two small maps ``main()`` needs
    after inference: which record ids back a given enum label (B2/M7), and which
    qualified labels (B3) need the bracket stripped back off for the emitted keyword.

    Keyed by the exact label the model was offered (post-truncation), so a term
    dropped by truncation simply has no entry — ``main()`` falls back to an empty id
    list rather than inventing one for a category the model was never shown.
    """
    id_lookup = {t["cs"]: t["ids"] for t in terms if t.get("ids")}
    strip_map = {t["cs"]: t["bare_cs"] for t in terms if t.get("bare_cs")}
    return id_lookup, strip_map


def build_system_prompt(
    vocab_data: dict,
    tokenizer: Any,
    max_tokens: int,
    skip_truncation: bool = False,
    excluded_themes: Optional[Set[str]] = None,
    prompt_config: Optional[Dict[str, str]] = None,
) -> Tuple[str, List[str], Dict[str, IdList], Dict[str, str]]:
    """Render the thematic vocabulary into a system prompt and return the term list.

    ``excluded_themes`` names the themes to withhold from the model, lower-cased.
    It defaults to ``{"other"}`` — today's hard-coded behaviour — but main() derives it
    from each theme's ``in_prompt`` flag in taxonomy_config.json, so which terms the
    model can reach is a reviewable configuration decision rather than a literal in the
    prompt builder. This matters: a term absent from the prompt is unreachable by
    construction, so withholding one turns "the model was wrong" and "the label was
    withheld" into the same score.

    Returns ``(prompt, surviving_terms, id_lookup, strip_map)``. ``id_lookup`` maps
    each surviving enum label to the full set of ``{source, id}`` records it stands
    for — its own plus every one B1's dedup discarded onto it (issue #6, M7) — for
    ``main()`` to attach as ``teater_category_ids``. ``strip_map`` maps a qualified
    label (B3, e.g. ``"zámek (sídlo elity)"``) back to its bare form, so the emitted
    ``teater_category`` reads ``"zámek"`` while the id set still disambiguates which
    sense was meant. Neither map is serialised into the prompt itself.
    """
    # The instruction text is prompts/system_prompt.txt, not a literal here: which
    # rules reach the model is a config decision (issue #6, C1), and the geographic
    # guardrail in particular has to stay in step with taxonomy_config.json — which it
    # cannot do while it lives in Python that vocab_build.py would have to grep.
    header, examples_footer = prompt_template.render(prompt_config or {})

    # Flattening and grouping live in prompt_template so that `prompt_template.py --full`
    # renders the same prompt this function sends, without importing torch to do it.
    prioritised = prompt_template.vocabulary_terms(vocab_data, excluded_themes)
    grouping = prompt_template.resolve_grouping(prompt_config or {})

    def _build_candidate_prompt(term_list: List[dict]) -> str:
        return header + prompt_template.vocabulary_block(term_list, grouping) + examples_footer

    full_prompt = _build_candidate_prompt(prioritised)
    token_count = count_tokens(full_prompt, tokenizer)

    _header_tok = count_tokens(header, tokenizer)
    _examples_tok = count_tokens(examples_footer, tokenizer)
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
        id_lookup, strip_map = _id_lookup_and_strip_map(prioritised)
        return full_prompt, [t["cs"] for t in prioritised], id_lookup, strip_map

    if token_count <= max_tokens:
        print("[vocab] Full vocabulary fits within token budget.")
        id_lookup, strip_map = _id_lookup_and_strip_map(prioritised)
        return full_prompt, [t["cs"] for t in prioritised], id_lookup, strip_map

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
    id_lookup, strip_map = _id_lookup_and_strip_map(surviving_terms)
    return surviving_prompt, surviving_cs, id_lookup, strip_map


def _attach_category_ids(
    enriched_results: List[dict],
    id_lookup: Dict[str, List[Dict[str, str]]],
    strip_map: Dict[str, str],
    emit_ids: bool,
) -> None:
    """Post-inference id passthrough (B2/B3/C4) — mutates each record's ``enrichment``
    dict in place. Deliberately done here rather than inside process_document*: the
    inference loop should not need to know about vocabulary internals to run a batch,
    and M7 asked for this to stay easy to disable ("drop it if it will create some
    issues") without touching either backend.

    ``teater_category`` is looked up *before* stripping, since ``id_lookup`` is keyed
    by the label the model actually selected (which may carry a B3 qualifier); the
    output field is then rewritten to the bare label so a qualifier never leaks past
    this pipeline into a downstream index.
    """
    for record in enriched_results:
        enrichment = record.get("enrichment")
        if not isinstance(enrichment, dict):
            continue
        category = enrichment.get("teater_category", "")
        if emit_ids:
            enrichment["teater_category_ids"] = id_lookup.get(category, [])
        bare = strip_map.get(category)
        if bare is not None:
            enrichment["teater_category"] = bare


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
    INPUT_DIR = Path(config.get("INPUT_DIR", "data_samples/DOC_LINE_CATEG"))
    VOCAB_PATH = config.get("VOCAB_PATH", "data_samples/vocab/union_nested.json")
    PARADATA_DIR = config.get("PARADATA_DIR", "paradata")

    _base_out = Path(config.get("OUTPUT_DIR", "data_samples/KW_PER_DOC_LLM"))
    _model_suffix = MODEL_KEY.replace(".", "").replace("-", "_")
    OUTPUT_DIR = _base_out.parent / f"{_base_out.name}_{_model_suffix}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    INCLUDE_NON_TEXT = config.get("INCLUDE_NON_TEXT", "true").lower() == "true"
    MIN_CHAR_COUNT = int(config.get("MIN_CHAR_COUNT", "3"))
    MIN_CHAR_NON_TEXT = int(config.get("MIN_CHAR_NON_TEXT", "8"))
    MIN_ALPHA_RATIO_NON_TEXT = float(config.get("MIN_ALPHA_RATIO_NON_TEXT", "0.40"))

    # M7: "list them now and drop it if it will create some issues" — kept behind a
    # config switch rather than hard-wired, so the id passthrough can be turned off
    # without a code change if it does.
    EMIT_CATEGORY_IDS = config.get("EMIT_CATEGORY_IDS", "true").lower() == "true"

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

    # D3 (issue #6): every run injects the AMCR + TEATER vocabulary into its prompt,
    # and until now the paradata recorded neither which build of it was used nor that
    # it was used at all. vocab_build.py has always written the identity beside the
    # artifact — taxonomy sha256s, tool version, TEATER's pinned commit — and nothing
    # read it. Empty for a legacy or auto-synced vocabulary with no sidecar; a run
    # must not fail over missing provenance.
    provenance = vocabulary_provenance(VOCAB_PATH)

    logger = ParadataLogger(
        program="nlp-enrich",
        config={
            **config,
            "output_dir_resolved": str(OUTPUT_DIR),
            "backend": BACKEND,
            "include_non_text": INCLUDE_NON_TEXT,
            "min_char_count": MIN_CHAR_COUNT,
            "min_char_non_text": MIN_CHAR_NON_TEXT,
            "min_alpha_ratio_non_text": MIN_ALPHA_RATIO_NON_TEXT,
            **({"vocabulary": provenance} if provenance else {}),
        },
        paradata_dir=PARADATA_DIR,
        output_types=["json"],
    )

    # Both vocabulary sources are CC BY-NC 4.0 and declared *conditional* in
    # para_config.txt, so they constrain a run's effective licence only when named
    # here. Logged per source actually present in the build, not hard-wired to both:
    # an AMCR-only artifact must not claim it used TEATER data.
    print(prompt_template.describe(config))
    for component in provenance.get("components", []):
        logger.log_component(component)
    if provenance:
        src = provenance["sources"]
        print(
            f"  vocabulary: {provenance['vocab']['terms']} terms, built by "
            f"{provenance['vocab']['tool_version']}, sources "
            + ", ".join(f"{n} ({v['records']} records)" for n, v in sorted(src.items()))
        )

    with logger:
        vocab_mgr = VocabularyManager(vocab_path=VOCAB_PATH)
        vocab_data = vocab_mgr.load()

        # Which themes reach the model is a taxonomy_config decision, not a literal in
        # the prompt builder. Absent an explicit in_prompt flag the default is today's
        # behaviour: everything except "Other".
        excluded_themes = {
            theme.lower()
            for theme, cfg in vocab_mgr.themes().items()
            if not cfg.get("in_prompt", theme.lower() != "other")
        }
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
        system_prompt, surviving_terms, id_lookup, strip_map = build_system_prompt(
            vocab_data,
            tokenizer,
            max_tokens=max_input_tokens,
            skip_truncation=skip_trunc,
            excluded_themes=excluded_themes,
            prompt_config=config,
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

                _attach_category_ids(enriched_results, id_lookup, strip_map, EMIT_CATEGORY_IDS)

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
