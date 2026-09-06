"""
tests/test_prompt_template.py — the system prompt as data (issue #6, C1).

These run against the SHIPPED `prompts/system_prompt.txt` and `prompts/output_template.json`
rather than fixtures wherever the assertion is about what the pipeline actually sends.
A fixture would pass while the real prompt said something else, which is the failure
this module exists to prevent: the guardrail wording and `taxonomy_config.json` are one
decision held in two files, and the build gate can only enforce that if the text it
checks is the text the model gets.
"""

import json
from pathlib import Path

import pytest

import prompt_template as pt

REPO_ROOT = Path(__file__).resolve().parent.parent


def _blocks():
    return pt.load_blocks()


# ── template parsing ────────────────────────────────────────────────────────────


def test_shipped_template_parses_and_declares_every_required_block():
    blocks = _blocks()
    for name in pt.REQUIRED_BLOCKS:
        assert name in blocks, f"required block [[{name}]] missing from the shipped template"
    assert blocks["vocabulary.header"].endswith("THEMATIC VOCABULARY:\n")


def test_every_shipped_block_is_reachable():
    """A block with no flag and no required status would sit in the file rendering
    nothing — the exact dead config `validate_settings` refuses elsewhere."""
    known = (
        set(pt.REQUIRED_BLOCKS)
        | set(pt.BLOCK_FLAGS)
        | {b for b in pt.GEO_GUARDRAIL_BLOCKS.values() if b}
    )
    assert set(_blocks()) <= known, f"unreachable blocks: {set(_blocks()) - known}"


def test_a_body_keeps_its_leading_blank_line_and_normalises_the_trailing_one(tmp_path):
    """Leading blank lines are load-bearing (that is how `vocabulary.header` asks for a
    blank line ahead of it); trailing ones are not, so an editor stripping whitespace
    cannot change the prompt."""
    f = tmp_path / "t.txt"
    f.write_text("[[a]]\n\nbody\n\n\n[[b]]\nx\n", encoding="utf-8")
    blocks = pt.load_blocks(f)
    assert blocks["a"] == "\nbody\n"
    assert blocks["b"] == "x\n"


def test_a_duplicate_block_is_an_error(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("[[a]]\nfirst\n[[a]]\nsecond\n", encoding="utf-8")
    with pytest.raises(pt.TemplateError, match="duplicate block"):
        pt.load_blocks(f)


def test_text_before_the_first_block_is_an_error(tmp_path):
    """Almost always a typo'd header; silently discarding it would drop prompt text."""
    f = tmp_path / "t.txt"
    f.write_text("[a]\noops\n[[role]]\nx\n", encoding="utf-8")
    with pytest.raises(pt.TemplateError, match="text before the first"):
        pt.load_blocks(f)


def test_comments_before_the_first_block_are_allowed(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("# a file comment\n\n[[role]]\nx\n", encoding="utf-8")
    assert pt.load_blocks(f) == {"role": "x\n"}


# ── block selection ─────────────────────────────────────────────────────────────


def test_default_config_renders_every_optional_block():
    """An absent flag means "on": a config that says nothing gets the prompt the
    pipeline has always sent, not a silently reduced one."""
    chosen = set(pt.selected_blocks(_blocks(), {}))
    assert set(pt.BLOCK_FLAGS) <= chosen
    assert "guardrail.geographic.strict" in chosen
    assert "guardrail.geographic.preference" not in chosen


@pytest.mark.parametrize("block,flag", sorted(pt.BLOCK_FLAGS.items(), key=lambda kv: kv[1]))
def test_each_flag_actually_removes_its_block(block, flag):
    blocks = _blocks()
    assert block in pt.selected_blocks(blocks, {})
    assert block not in pt.selected_blocks(blocks, {flag: "false"})


def test_required_blocks_ignore_every_flag():
    off = {v: "false" for v in pt.BLOCK_FLAGS.values()}
    chosen = pt.selected_blocks(_blocks(), off)
    assert set(pt.REQUIRED_BLOCKS) <= set(chosen)


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("strict", "guardrail.geographic.strict"),
        ("preference", "guardrail.geographic.preference"),
        ("off", None),
    ],
)
def test_the_guardrail_switch_picks_exactly_one_wording(mode, expected):
    chosen = set(pt.selected_blocks(_blocks(), {"PROMPT_GEO_GUARDRAIL": mode}))
    geo = {b for b in pt.GEO_GUARDRAIL_BLOCKS.values() if b} & chosen
    assert geo == ({expected} if expected else set())


def test_an_unknown_guardrail_mode_is_an_error():
    with pytest.raises(pt.TemplateError, match="PROMPT_GEO_GUARDRAIL"):
        pt.selected_blocks(_blocks(), {"PROMPT_GEO_GUARDRAIL": "loose"})


def test_a_template_missing_the_selected_guardrail_is_an_error(tmp_path):
    """Flag and template drifting apart must fail loudly: the prompt would otherwise
    silently carry no geographic rule while the config says it does."""
    f = tmp_path / "t.txt"
    f.write_text(
        "[[role]]\nr\n[[rule.json_only]]\nj\n[[vocabulary.header]]\nV:\n", encoding="utf-8"
    )
    with pytest.raises(pt.TemplateError, match="which the template does not define"):
        pt.selected_blocks(pt.load_blocks(f), {"PROMPT_GEO_GUARDRAIL": "strict"})


# ── rendering ───────────────────────────────────────────────────────────────────


def test_preamble_ends_at_the_vocabulary_anchor_and_footer_follows_it():
    preamble, footer = pt.render({})
    assert preamble.endswith("THEMATIC VOCABULARY:\n")
    assert footer.startswith("\nEXAMPLES:")
    assert "EXAMPLES" not in preamble


def test_the_default_render_is_the_prompt_the_pipeline_shipped_with():
    """Byte-for-byte against the wording that was hard-coded in llm_run.py before the
    template existed. Moving text into a file must not change what the model reads —
    otherwise every score before and after this refactor is incomparable."""
    preamble, footer = pt.render({})
    assert preamble == (
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
    assert footer == (
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


def test_turning_a_block_off_removes_exactly_that_text():
    full, _ = pt.render({})
    without, _ = pt.render({"PROMPT_OCR_NORMALISATION": "false"})
    assert "normalize obvious OCR artifacts" in full
    assert "normalize obvious OCR artifacts" not in without
    # and nothing else moved
    assert len(full) > len(without)
    assert without == full.replace(_blocks()["rule.ocr_normalisation"], "")


def test_the_template_path_is_configurable(tmp_path):
    f = tmp_path / "variant.txt"
    f.write_text(
        "[[role]]\nvariant role\n[[rule.json_only]]\nj\n[[vocabulary.header]]\nV:\n",
        encoding="utf-8",
    )
    preamble, _ = pt.render({"PROMPT_TEMPLATE": str(f), "PROMPT_GEO_GUARDRAIL": "off"})
    assert preamble.startswith("variant role")


# ── the guardrail's two halves ──────────────────────────────────────────────────


def test_guardrail_text_returns_the_wording_actually_in_force():
    strict = pt.guardrail_text({"PROMPT_GEO_GUARDRAIL": "strict"})
    pref = pt.guardrail_text({"PROMPT_GEO_GUARDRAIL": "preference"})
    assert "NEVER select a country name" in strict
    assert "genuinely about it" in pref
    assert pt.guardrail_text({"PROMPT_GEO_GUARDRAIL": "off"}) == ""


def test_the_shipped_config_and_the_shipped_taxonomy_agree():
    """The invariant vocab_build.py enforces, asserted here too so a bad pairing fails
    the fast test lane and not only the build."""
    from vocab_manager import VocabularyManager

    config = pt.load_run_config(REPO_ROOT / "llm_config.txt")
    manager = VocabularyManager(
        config_path=str(REPO_ROOT / "data_samples" / "taxonomy_config.json")
    )
    assert manager.geo_guardrail_problems(pt.guardrail_text(config)) == []


def test_the_strict_wording_carries_every_marker_the_config_looks_for():
    """`geo_guardrail.prompt_markers` is how the build recognises the clause. If the
    wording is edited without the markers, the gate silently stops detecting it."""
    from vocab_manager import VocabularyManager

    manager = VocabularyManager(
        config_path=str(REPO_ROOT / "data_samples" / "taxonomy_config.json")
    )
    strict = pt.guardrail_text({"PROMPT_GEO_GUARDRAIL": "strict"}).lower()
    for marker in manager.geo_guardrail()["prompt_markers"]:
        assert marker.lower() in strict, f"marker {marker!r} not in the strict wording"


# ── the CLI: reading the prompt without a GPU run ───────────────────────────────


def test_blocks_lists_every_block_with_its_state(capsys):
    assert pt.main(["--blocks"]) == 0
    out = capsys.readouterr().out
    for name in pt.load_blocks():
        assert name in out
    assert "[on ]" in out and "[off]" in out


def test_blocks_reports_a_token_cost_per_rule(capsys):
    """The instruction preamble competes with the vocabulary for one budget, so "is this
    rule worth its tokens" is a real question — and at a tight window a rule kept is
    terms dropped. The examples block is the expensive one."""
    assert pt.main(["--blocks"]) == 0
    out = capsys.readouterr().out
    assert "tokens of instructions" in out
    assert "tok" in out
    blocks = pt.load_blocks()
    assert pt.block_cost(blocks["examples"]) > pt.block_cost(blocks["rule.exact_term"])
    assert pt.block_cost(blocks["examples"]) > 100


def test_preview_prints_the_prompt_and_marks_where_the_vocabulary_goes(capsys):
    """The term list is 4 700 lines and lives in the built artifact; a preview that
    dumped it would be unreadable and would duplicate union_nested.json."""
    assert pt.main(["--preview", "--set", "PROMPT_GEO_GUARDRAIL=strict"]) == 0
    out = capsys.readouterr().out
    assert "You are an expert archaeological data extractor" in out
    assert "NEVER select a country name" in out
    assert "the vocabulary term list is injected here" in out
    assert out.index("THEMATIC VOCABULARY") < out.index("EXAMPLES")


def test_diff_shows_exactly_the_guardrail_swap(capsys):
    """The M11/M12 decision, in one command: what changes when the wording is relaxed.
    A reviewer ruling on prompt text should be able to read the change, not infer it."""
    assert (
        pt.main(["--diff", "PROMPT_GEO_GUARDRAIL=strict", "PROMPT_GEO_GUARDRAIL=preference"]) == 0
    )
    out = capsys.readouterr().out
    assert "-NEVER select a country name" in out
    assert "+Select a geographic, ethnic or dynastic term only when" in out
    # nothing else moved
    changed = [
        ln for ln in out.splitlines() if ln[:1] in "+-" and not ln.startswith(("+++", "---"))
    ]
    assert len(changed) == 2


def test_diff_of_two_identical_settings_says_so(capsys):
    assert pt.main(["--diff", "PROMPT_EXAMPLES=true", "PROMPT_EXAMPLES=true"]) == 0
    assert "no difference" in capsys.readouterr().out


def test_a_malformed_set_is_rejected():
    with pytest.raises(SystemExit, match="KEY=VALUE"):
        pt.main(["--preview", "--set", "PROMPT_EXAMPLES"])


def test_no_action_prints_help_rather_than_doing_something(capsys):
    assert pt.main([]) == 2


# ── the committed output template ───────────────────────────────────────────────


def test_output_template_is_valid_and_documents_the_record_shape():
    tpl = pt.output_template()
    assert tpl["shape"] == "array of records"
    assert tpl["example_records"], "the template must carry at least one example record"
    for record in tpl["example_records"]:
        assert set(record) >= {
            "file_id",
            "page",
            "line",
            "categ",
            "quality_score",
            "original_text",
            "enrichment",
        }


def test_output_template_fields_match_the_documented_field_list():
    """The prose `fields` block and the machine-readable `example_records` are two
    statements of the same contract; a reader who trusts one must not be misled."""
    tpl = pt.output_template()
    documented = set(tpl["fields"]["enrichment"])
    for record in tpl["example_records"]:
        assert set(record["enrichment"]) == documented


def test_output_template_matches_the_schema_the_model_is_given():
    """The four fields `build_schema` declares must be exactly the four the template
    says the model returns — `teater_category_ids` is the one addition, attached after
    inference rather than generated."""
    src = (REPO_ROOT / "llm_run.py").read_text(encoding="utf-8")
    model_fields = {
        "extracted_keywords_cs",
        "extracted_keywords_en",
        "teater_category",
        "confidence_score",
    }
    for field in model_fields:
        assert f"{field}: " in src, f"{field} is not declared in build_schema"

    documented = set(pt.output_record_fields())
    assert documented == model_fields | {"teater_category_ids"}


def test_the_prompt_examples_agree_with_the_output_template():
    """The examples in the prompt teach the model the output shape; the template
    documents it for consumers. They are the same contract stated twice, so they get
    checked against each other rather than trusted separately."""
    _, footer = pt.render({})
    example_blocks = [b for b in footer.split("Correct output:\n")[1:]]
    assert len(example_blocks) >= 2

    model_fields = set(pt.output_record_fields()) - {"teater_category_ids"}
    for block in example_blocks:
        obj = json.loads(block[block.index("{") : block.index("}") + 1])
        assert set(obj) == model_fields, f"prompt example fields {set(obj)} != {model_fields}"


def test_the_metatext_example_shows_empty_keyword_arrays():
    """A rule the pipeline also enforces in code (llm_utils blanks them). The example
    must not contradict it, or the model is being shown one thing and corrected after."""
    _, footer = pt.render({})
    block = footer.split("Correct output:\n")[2]
    obj = json.loads(block[block.index("{") : block.index("}") + 1])
    assert obj["teater_category"] == "Nerelevantní (meta-text)"
    assert obj["extracted_keywords_cs"] == []
    assert obj["extracted_keywords_en"] == []


# ── the vocabulary half (shared with llm_run.build_system_prompt) ────────────────
#
# `vocabulary_terms` and `vocabulary_block` were lifted verbatim out of
# `llm_run.build_system_prompt` so that `--full` renders the prompt the pipeline
# actually sends. The risk that creates is a silent transcription error: the extraction
# would still run, still look plausible, and quietly render a different prompt than the
# one under test elsewhere. `_reference_terms` / `_reference_block` below are the
# pre-extraction code, kept as the independent second opinion, and the first test asserts
# byte equality against the real 4 718-term artifact.

VOCAB_FILE = REPO_ROOT / "data_samples" / "vocab" / "union_nested.json"


def _reference_terms(vocab_data, excluded_themes=None):
    skip = {"other"} if excluded_themes is None else {t.lower() for t in excluded_themes}
    raw_terms = [
        {
            "theme": "Administrative / Meta",
            "sub": "",
            "cs": "Nerelevantní (meta-text)",
            "en": "Irrelevant / Meta-text",
        }
    ]
    for theme, data in vocab_data.items():
        if theme.startswith("_") or theme.lower() in skip:
            continue
        if isinstance(data, dict):
            if "keywords" in data and isinstance(data["keywords"], dict):
                cs_list = data["keywords"].get("cs", [])
                en_list = data["keywords"].get("en", [])
                for i, cs_key in enumerate(cs_list):
                    en = en_list[i] if i < len(en_list) else cs_key
                    raw_terms.append({"theme": theme, "sub": "", "cs": cs_key, "en": en})
            else:
                for cs_key, pair in data.items():
                    en = pair.get("en", cs_key) if isinstance(pair, dict) else cs_key
                    sub = pair.get("sub", "") if isinstance(pair, dict) else ""
                    term = {"theme": theme, "sub": sub, "cs": cs_key, "en": en}
                    if isinstance(pair, dict) and pair.get("source") and pair.get("source_id"):
                        term["ids"] = [{"source": pair["source"], "id": pair["source_id"]}] + [
                            {"source": d["source"], "id": d["id"]}
                            for d in (pair.get("discarded_ids") or [])
                        ]
                        if pair.get("bare_cs"):
                            term["bare_cs"] = pair["bare_cs"]
                    raw_terms.append(term)
    return raw_terms


def _reference_block(term_list):
    groups = {}
    for t in term_list:
        key = (t["theme"], t.get("sub") or "")
        groups.setdefault(key, []).append(f"{t['cs']} ({t['en']})")
    prompt = ""
    for (theme_name, sub_name), lines in groups.items():
        title = f"{theme_name} / {sub_name}" if sub_name else theme_name
        prompt += f"\n--- {title} ---\n"
        prompt += "\n".join(f"- {line}" for line in lines) + "\n"
    return prompt


def _shipped_vocab():
    if not VOCAB_FILE.exists():
        pytest.skip("union_nested.json not built")
    return json.loads(VOCAB_FILE.read_text(encoding="utf-8"))


def test_llm_run_renders_the_vocabulary_through_the_shared_functions():
    """The extraction is only worth anything while `llm_run` is the *caller*.

    It was published with `prompt_template` holding the functions and `llm_run` still
    holding a copy of the loop — at which point `--full` and `prompts/prompt_full.txt`
    claim to be the prompt the pipeline sends while nothing makes that true. That state
    passes every other test in this file, so it gets its own: read the source, since
    `llm_run` cannot be imported here (it pulls in torch, transformers and pysqlite3).
    """
    src = (REPO_ROOT / "llm_run.py").read_text(encoding="utf-8")

    assert "prompt_template.vocabulary_terms(" in src
    assert "prompt_template.vocabulary_block(" in src

    # the two lines that only exist in a second copy of the renderer
    for reimplementation in ("groups.setdefault(key, []).append(", "raw_terms: List[dict] = []"):
        assert reimplementation not in src, (
            f"llm_run.py re-implements the vocabulary renderer ({reimplementation!r}) — "
            "prompt_template must stay the single definition"
        )


def test_the_extracted_renderer_reproduces_the_reference_implementation_byte_for_byte():
    vocab = _shipped_vocab()
    withheld = pt.themes_withheld()

    assert pt.vocabulary_terms(vocab, withheld) == _reference_terms(vocab, withheld)
    assert pt.vocabulary_block(pt.vocabulary_terms(vocab, withheld)) == _reference_block(
        _reference_terms(vocab, withheld)
    )


def test_vocabulary_terms_defaults_to_withholding_only_other():
    vocab = {
        "Artefact": {"nůž": {"en": "knife", "sub": "tools"}},
        "Other": {"cosi": {"en": "something"}},
    }
    labels = {t["cs"] for t in pt.vocabulary_terms(vocab)}
    assert "nůž" in labels
    assert "cosi" not in labels, "the default must match build_system_prompt's own"


def test_the_meta_text_sentinel_is_always_first_and_carries_no_ids():
    """It is the answer for a line that is not archaeology, so it cannot depend on which
    facets are in the vocabulary — and it must not look like a term with a source id,
    or `_id_lookup_and_strip_map` would claim records back it."""
    terms = pt.vocabulary_terms({})
    assert terms[0]["cs"] == "Nerelevantní (meta-text)"
    assert "ids" not in terms[0]


def test_vocabulary_terms_carries_ids_and_bare_cs_through():
    vocab = {
        "Feature": {
            "zámek (sídlo elity)": {
                "en": "chateau",
                "source": "teater",
                "source_id": "1439",
                "bare_cs": "zámek",
                "discarded_ids": [{"source": "amcr", "id": "HES-000817"}],
            }
        }
    }
    term = pt.vocabulary_terms(vocab, [])[1]
    assert term["ids"] == [
        {"source": "teater", "id": "1439"},
        {"source": "amcr", "id": "HES-000817"},
    ]
    assert term["bare_cs"] == "zámek"
    assert "bare_cs" not in pt.vocabulary_block([term]), "internals must not reach the prompt"


def test_themes_withheld_reads_in_prompt_from_the_shipped_taxonomy_config(tmp_path):
    assert pt.themes_withheld() == {"other"}, "the shipped config withholds only Other"

    config = tmp_path / "taxonomy_config.json"
    config.write_text(
        json.dumps(
            {
                "_settings": {"tie_break": []},
                "Artefact": {"priority": 5},
                "Draft": {"priority": 0, "in_prompt": False},
                "Other": {"priority": 0, "in_prompt": True},
            }
        ),
        encoding="utf-8",
    )
    assert pt.themes_withheld(config) == {"draft"}


def test_render_full_is_the_preamble_then_the_terms_then_the_footer():
    _shipped_vocab()
    preamble, footer = pt.render({})
    full = pt.render_full({})

    assert full.startswith(preamble)
    assert full.endswith(footer)
    assert "… the vocabulary term list is injected here …" not in full
    assert "\n--- Administrative / Meta ---\n" in full


def test_render_full_renders_every_term_the_prompt_offers():
    vocab = _shipped_vocab()
    terms = pt.vocabulary_terms(vocab, pt.themes_withheld())
    full = pt.render_full({})
    # one bullet per term, plus the sentinel, and nothing else claiming to be one
    assert full.count("\n- ") == len(terms)


def test_render_full_respects_the_prompt_flags():
    """The instruction half of --full is the configured one — otherwise a reviewer would
    read a prompt nobody sends."""
    _shipped_vocab()
    strict = pt.render_full({"PROMPT_GEO_GUARDRAIL": "strict"})
    off = pt.render_full({"PROMPT_GEO_GUARDRAIL": "off"})
    assert "NEVER select a country name" in strict
    assert "NEVER select a country name" not in off


def test_render_full_says_how_to_fix_a_missing_vocabulary(tmp_path):
    with pytest.raises(pt.TemplateError) as excinfo:
        pt.render_full({}, vocab_path=tmp_path / "absent.json")
    assert "vocab_build.py --from-flat" in str(excinfo.value)


def test_full_cli_prints_the_whole_prompt(capsys):
    _shipped_vocab()
    assert pt.main(["--full"]) == 0
    out = capsys.readouterr().out
    assert "THEMATIC VOCABULARY:" in out
    assert "--- Administrative / Meta ---" in out
    assert out.rstrip().endswith("}"), "the examples footer must be the last thing printed"


# ── vocabulary grouping (M2's open half: does the layout matter?) ────────────────


def test_the_shipped_grouping_is_the_prompt_that_was_always_sent():
    """`facet_sub` must be byte-identical to the layout the pipeline used before the flag
    existed, or every earlier run becomes incomparable with every later one."""
    vocab = _shipped_vocab()
    terms = pt.vocabulary_terms(vocab, pt.themes_withheld())
    assert pt.vocabulary_block(terms, "facet_sub") == _reference_block(terms)
    assert pt.vocabulary_block(terms) == _reference_block(terms), "the default must be facet_sub"


def _bullets(terms, grouping):
    return [
        line[2:]
        for line in pt.vocabulary_block(terms, grouping).splitlines()
        if line.startswith("- ")
    ]


@pytest.mark.parametrize("grouping", pt.GROUPING_MODES)
def test_every_grouping_offers_exactly_the_same_terms(grouping):
    """The point of the ablation: the vocabulary on offer must not move with the layout,
    or a score difference would be measuring which terms were available instead."""
    vocab = _shipped_vocab()
    terms = pt.vocabulary_terms(vocab, pt.themes_withheld())
    want = [f"{t['cs']} ({t['en']})" for t in terms]
    assert sorted(_bullets(terms, grouping)) == sorted(want)


def test_only_the_sub_branch_layout_reorders_terms_and_only_inside_a_facet():
    """The ablation's design, asserted rather than assumed. `facet` and `flat` keep the
    term order, so comparing them isolates the headers; `facet_sub` makes a facet's
    sub-groups contiguous, so comparing it against `facet` measures the source's second
    level. Neither ever moves a term across a facet boundary."""
    vocab = _shipped_vocab()
    terms = pt.vocabulary_terms(vocab, pt.themes_withheld())
    want = [f"{t['cs']} ({t['en']})" for t in terms]

    assert _bullets(terms, "flat") == want
    assert _bullets(terms, "facet") == want

    grouped = _bullets(terms, "facet_sub")
    assert grouped != want, "the shipped layout does reorder — the contrast is real"

    by_facet: dict = {}
    for t in terms:
        by_facet.setdefault(t["theme"], []).append(f"{t['cs']} ({t['en']})")
    cut = 0
    for facet, members in by_facet.items():
        segment = grouped[cut : cut + len(members)]
        cut += len(members)
        assert sorted(segment) == sorted(members), f"{facet} leaked across a boundary"


def test_each_grouping_mode_emits_the_headers_it_promises():
    vocab = _shipped_vocab()
    terms = pt.vocabulary_terms(vocab, pt.themes_withheld())
    rendered = {g: pt.vocabulary_block(terms, g) for g in pt.GROUPING_MODES}

    assert "--- Chronology / geologická doba ---" in rendered["facet_sub"]
    assert "--- Chronology ---" in rendered["facet"]
    headers = [
        line.strip("- ").strip()
        for line in rendered["facet"].splitlines()
        if line.startswith("---")
    ]
    assert "Chronology / geologická doba" not in headers, "facet mode must drop the sub level"
    assert headers == list(dict.fromkeys(t["theme"] for t in terms))
    assert "---" not in rendered["flat"]

    # each step down removes structure and nothing else
    assert len(rendered["facet_sub"]) > len(rendered["facet"]) > len(rendered["flat"])


def test_group_titles_reports_what_each_layout_would_emit():
    terms = [
        {"theme": "Feature", "sub": "objekt", "cs": "a", "en": "a"},
        {"theme": "Feature", "sub": "areál", "cs": "b", "en": "b"},
        {"theme": "Artefact", "sub": "", "cs": "c", "en": "c"},
    ]
    assert pt.group_titles(terms, "facet_sub") == [
        "Feature / objekt",
        "Feature / areál",
        "Artefact",
    ]
    assert pt.group_titles(terms, "facet") == ["Feature", "Artefact"]
    assert pt.group_titles(terms, "flat") == []


def test_an_unknown_grouping_is_refused_rather_than_defaulted():
    """A typo must not quietly render the shipped layout — an ablation would then report
    a difference nobody applied."""
    with pytest.raises(pt.TemplateError, match="PROMPT_VOCAB_GROUPING"):
        pt.resolve_grouping({"PROMPT_VOCAB_GROUPING": "facets"})
    with pytest.raises(pt.TemplateError, match="unknown grouping"):
        pt.vocabulary_block([], "facets")


def test_the_shipped_config_selects_the_shipped_grouping():
    config = pt.load_run_config(REPO_ROOT / "llm_config.txt")
    assert pt.GROUPING_FLAG in config, "llm_config.txt must state the grouping explicitly"
    assert pt.resolve_grouping(config) == pt.DEFAULT_GROUPING


def test_render_full_and_the_banner_honour_the_grouping():
    _shipped_vocab()
    assert "--- Chronology / geologická doba ---" in pt.render_full({})
    assert (
        "---"
        not in pt.render_full({"PROMPT_VOCAB_GROUPING": "flat"}).split("THEMATIC VOCABULARY:")[1]
    )
    assert "vocabulary grouping: flat" in pt.describe({"PROMPT_VOCAB_GROUPING": "flat"})


def test_llm_run_renders_with_the_configured_grouping():
    """Same reasoning as the shared-renderer pin: the flag is only real if the pipeline
    reads it, and llm_run cannot be imported here."""
    src = (REPO_ROOT / "llm_run.py").read_text(encoding="utf-8")
    assert "prompt_template.resolve_grouping(" in src
    assert "prompt_template.vocabulary_block(term_list, grouping)" in src


# ── the committed prompt sheets ─────────────────────────────────────────────────


def test_the_committed_sheets_match_a_fresh_render():
    """`prompts/prompt_*.txt` are generated files the reviewers read. They move whenever
    the vocabulary, a `PROMPT_*` flag or the template changes — and none of those edits
    touches the sheets, which is how `union_nested.json` went stale twice."""
    _shipped_vocab()
    config = pt.load_run_config(REPO_ROOT / "llm_config.txt")
    assert pt.stale_sheets(config) == [], "regenerate with `python3 prompt_template.py --write`"


def test_a_tampered_sheet_is_reported_stale(tmp_path):
    config = pt.load_run_config(REPO_ROOT / "llm_config.txt")
    pt.write_sheets(config, tmp_path)
    assert pt.stale_sheets(config, tmp_path) == []

    (tmp_path / "prompt_blocks.txt").write_text("nonsense\n", encoding="utf-8")
    assert pt.stale_sheets(config, tmp_path) == ["prompt_blocks.txt"]


def test_a_missing_sheet_is_reported_stale(tmp_path):
    config = pt.load_run_config(REPO_ROOT / "llm_config.txt")
    pt.write_sheets(config, tmp_path)
    (tmp_path / "prompt_full.txt").unlink()
    assert "prompt_full.txt" in pt.stale_sheets(config, tmp_path)


def test_write_is_idempotent_and_reports_only_what_it_changed(tmp_path):
    config = pt.load_run_config(REPO_ROOT / "llm_config.txt")
    assert sorted(pt.write_sheets(config, tmp_path)) == sorted(pt.sheet_contents(config))
    assert pt.write_sheets(config, tmp_path) == []


def test_the_sheets_follow_the_flags_they_document(tmp_path):
    """A sheet generated under different flags must differ — otherwise `--check` would
    pass against a configuration the sheets do not describe."""
    strict = pt.sheet_contents({"PROMPT_GEO_GUARDRAIL": "strict"})
    flat = pt.sheet_contents({"PROMPT_VOCAB_GROUPING": "flat"})
    shipped = pt.sheet_contents(pt.load_run_config(REPO_ROOT / "llm_config.txt"))

    assert strict["prompt_preview.txt"] != shipped["prompt_preview.txt"]
    assert flat["prompt_full.txt"] != shipped["prompt_full.txt"]
    assert "vocabulary grouping: flat" in flat["prompt_blocks.txt"]


def test_the_sheets_are_exactly_what_the_cli_prints(capsys):
    """`--write` and a shell redirect must produce the same bytes, or the RUNBOOK's two
    documented ways of getting a sheet would disagree."""
    _shipped_vocab()
    config = pt.load_run_config(REPO_ROOT / "llm_config.txt")
    sheets = pt.sheet_contents(config)

    for flag, name in (("--blocks", "prompt_blocks.txt"), ("--preview", "prompt_preview.txt")):
        capsys.readouterr()
        assert pt.main([flag]) == 0
        assert capsys.readouterr().out == sheets[name]


def test_check_exits_1_on_drift_and_0_when_current(tmp_path, monkeypatch, capsys):
    config_file = REPO_ROOT / "llm_config.txt"
    monkeypatch.setattr(pt, "SHEET_DIR", tmp_path)
    assert pt.main(["--check", "--config", str(config_file)]) == 1
    capsys.readouterr()
    assert pt.main(["--write", "--config", str(config_file)]) == 0
    assert pt.main(["--check", "--config", str(config_file)]) == 0
    assert "match the configured prompt" in capsys.readouterr().out
