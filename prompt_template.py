"""
prompt_template.py — the system prompt's instruction text, as data rather than code.

The prompt used to be two string literals inside ``llm_run.py``. That made every
wording change a Python edit, and it made one particular change — relaxing the
geographic guardrail (issue #6, C1/M11/M12) — a code change that had to be kept in step
with a config file by hand. Both halves now live in a plain text file that a domain
reviewer can edit, and which of its blocks reach the model is a set of flags in
``llm_config.txt``.

**Import-light on purpose.** ``llm_run.py`` pulls in torch, transformers and pysqlite3;
``vocab_build.py`` must not. Both need to know what the prompt says, so this module
depends on nothing outside the standard library and can be imported from either side —
which is what lets the build gate check the *rendered* prompt rather than grepping
Python source for a sentence.

Template format
---------------
A block header is ``[[name]]`` alone on a line; everything up to the next header is that
block's body. Order in the file is render order: to move a rule, move it in the file.
A body's trailing newlines are normalised to exactly one, so an editor that strips
trailing whitespace cannot silently change the prompt; leading blank lines are kept,
because that is how a block asks for a blank line before itself.

    [[role]]
    You are an expert archaeological data extractor. …

    [[guardrail.geographic.strict]]
    NEVER select a country name, …

Selection
---------
:data:`REQUIRED_BLOCKS` always render. Every other block is gated by a flag in
:data:`BLOCK_FLAGS` — a plain ``true``/``false`` in the config. The geographic guardrail
is the one three-way switch (``strict`` / ``preference`` / ``off``), because that is the
shape of the actual decision: the wording is not merely on or off, it has two forms and
the vocabulary must agree with whichever is in force (see
``vocab_manager.geo_guardrail_problems``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = REPO_ROOT / "prompts" / "system_prompt.txt"
DEFAULT_OUTPUT_TEMPLATE = REPO_ROOT / "prompts" / "output_template.json"
DEFAULT_VOCAB = REPO_ROOT / "data_samples" / "vocab" / "union_nested.json"
DEFAULT_TAXONOMY_CONFIG = REPO_ROOT / "data_samples" / "taxonomy_config.json"

_HEADER_RE = re.compile(r"^\[\[([A-Za-z0-9_.]+)\]\]\s*$")

# Rendered whatever the config says. Removing any of these would produce a prompt that
# does not describe the task, the output contract, or where the vocabulary begins.
REQUIRED_BLOCKS = ("role", "rule.json_only", "vocabulary.header")

# block name -> the llm_config.txt flag that includes it. Every one defaults to true:
# the shipped configuration renders the prompt the pipeline has always sent.
BLOCK_FLAGS: Dict[str, str] = {
    "task.extract": "PROMPT_TASK_EXTRACT",
    "task.select": "PROMPT_TASK_SELECT",
    "rule.metatext": "PROMPT_METATEXT_RULE",
    "rule.ocr_normalisation": "PROMPT_OCR_NORMALISATION",
    "rule.exact_term": "PROMPT_EXACT_TERM",
    "examples": "PROMPT_EXAMPLES",
}

# The three-way switch. Values map to the block that carries that wording; "off" renders
# no guardrail at all.
GEO_GUARDRAIL_FLAG = "PROMPT_GEO_GUARDRAIL"
GEO_GUARDRAIL_BLOCKS: Dict[str, Optional[str]] = {
    "strict": "guardrail.geographic.strict",
    "preference": "guardrail.geographic.preference",
    "off": None,
}
DEFAULT_GEO_GUARDRAIL = "strict"

# How the term list is laid out under the vocabulary header. @motyc asked to "distinguish
# clearly between which terms are available to the model and how those terms are grouped
# for inference", and to "check empirically how much facet placement affects the results
# at all" once the whole vocabulary fits the context (issue #6, M2 — the half of that
# ruling that is an experiment rather than a policy). It fits now, so the layout is a
# flag: the same terms, in the same order, under three different amounts of structure.
#
#   facet_sub — "--- Facet / Subgroup ---", the shipped prompt: both curated levels
#   facet     — "--- Facet ---" only: the facets, without the source's own second level
#   flat      — no headers at all: the terms, and nothing claiming they belong together
#
# All three offer the same terms and truncate identically. What else moves is *order*:
# grouping by sub-branch makes a facet's sub-groups contiguous, which `vocabulary_terms`
# does not, so `facet_sub` is a permutation within each facet while `facet` and `flat`
# keep the term order exactly. That gives two clean contrasts rather than one muddled
# scale — `facet` vs `flat` isolates the headers alone, `facet_sub` vs `facet` measures
# the source's second level (headers and adjacency together, as a real grouping is).
GROUPING_FLAG = "PROMPT_VOCAB_GROUPING"
GROUPING_MODES = ("facet_sub", "facet", "flat")
DEFAULT_GROUPING = "facet_sub"

# Blocks the vocabulary is rendered between: everything before `vocabulary.header` is
# the instruction preamble, everything after it is appended once the terms are in.
VOCABULARY_ANCHOR = "vocabulary.header"


# Where the template lives, overridable per run so a variant prompt is a config edit
# rather than a file swap.
TEMPLATE_PATH_FLAG = "PROMPT_TEMPLATE"


class TemplateError(ValueError):
    """A template or flag problem that would produce a prompt nobody intended."""


def template_path(config: Optional[Dict[str, str]] = None, override: Optional[Path] = None) -> Path:
    """Resolve the template file: explicit argument, then ``PROMPT_TEMPLATE``, then the
    shipped default. A relative configured path is taken against the repo root, so the
    same config works whatever directory the run started in."""
    if override is not None:
        return Path(override)
    configured = (config or {}).get(TEMPLATE_PATH_FLAG)
    if not configured:
        return DEFAULT_TEMPLATE
    path = Path(configured)
    return path if path.is_absolute() else REPO_ROOT / path


def load_run_config(path: Path) -> Dict[str, str]:
    """Read a ``KEY=VALUE`` config file. A duplicate of ``llm_utils.load_config`` on
    purpose: that module imports torch, and both ``vocab_build.py`` and this module's
    own tests need the prompt flags without paying for a GPU stack."""
    config: Dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            config[key.strip()] = value.strip()
    return config


def load_blocks(path: Optional[Path] = None) -> Dict[str, str]:
    """Parse a template file into ``{block name: body}``, preserving file order.

    Raises :class:`TemplateError` on a duplicate block name — a silently-shadowed block
    is exactly the kind of edit this file exists to make reviewable — and on text before
    the first header, which is almost always a typo'd header.
    """
    path = Path(path) if path else DEFAULT_TEMPLATE
    if not path.exists():
        raise TemplateError(f"prompt template not found: {path}")

    blocks: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = _HEADER_RE.match(raw)
        if match:
            name = match.group(1)
            if name in blocks:
                raise TemplateError(f"{path}:{lineno}: duplicate block [[{name}]]")
            blocks[name] = []
            current = name
            continue
        if current is None:
            if raw.strip() and not raw.lstrip().startswith("#"):
                raise TemplateError(f"{path}:{lineno}: text before the first [[block]]")
            continue
        blocks[current].append(raw)

    # Normalise each body to end in exactly one newline; keep leading blank lines, which
    # are how a block asks for a blank line ahead of itself.
    return {name: "\n".join(lines).rstrip("\n") + "\n" for name, lines in blocks.items()}


def _is_true(value: object, default: bool = True) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def resolve_geo_guardrail(config: Optional[Dict[str, str]] = None) -> str:
    """The configured guardrail mode, validated. See :data:`GEO_GUARDRAIL_BLOCKS`."""
    raw = (config or {}).get(GEO_GUARDRAIL_FLAG, DEFAULT_GEO_GUARDRAIL)
    mode = str(raw).strip().lower()
    if mode not in GEO_GUARDRAIL_BLOCKS:
        raise TemplateError(
            f"{GEO_GUARDRAIL_FLAG}={raw!r} is not one of {sorted(GEO_GUARDRAIL_BLOCKS)}"
        )
    return mode


def resolve_grouping(config: Optional[Dict[str, str]] = None) -> str:
    """Which vocabulary layout is in force. An unknown value is an error, not a fallback:
    a typo that silently rendered the default would make an ablation report a difference
    that was never applied."""
    mode = (config or {}).get(GROUPING_FLAG, DEFAULT_GROUPING).strip().lower()
    if mode not in GROUPING_MODES:
        raise TemplateError(f"{GROUPING_FLAG}={mode!r} is not one of {', '.join(GROUPING_MODES)}")
    return mode


def selected_blocks(blocks: Dict[str, str], config: Optional[Dict[str, str]] = None) -> List[str]:
    """Block names to render, in template order, for this configuration.

    A flag naming a block the template does not define is an error rather than a no-op:
    it means the flag and the template have drifted apart, and the prompt is then not
    the one the config describes.
    """
    config = config or {}
    mode = resolve_geo_guardrail(config)
    guardrail_block = GEO_GUARDRAIL_BLOCKS[mode]

    missing = [b for b in REQUIRED_BLOCKS if b not in blocks]
    if missing:
        raise TemplateError(f"prompt template is missing required block(s): {missing}")
    if guardrail_block and guardrail_block not in blocks:
        raise TemplateError(
            f"{GEO_GUARDRAIL_FLAG}={mode!r} needs block [[{guardrail_block}]], "
            "which the template does not define"
        )

    chosen: List[str] = []
    for name in blocks:  # dict preserves the template's own order
        if name in REQUIRED_BLOCKS:
            chosen.append(name)
        elif name in BLOCK_FLAGS:
            if _is_true(config.get(BLOCK_FLAGS[name])):
                chosen.append(name)
        elif name in GEO_GUARDRAIL_BLOCKS.values():
            if name == guardrail_block:
                chosen.append(name)
        else:
            raise TemplateError(
                f"block [[{name}]] has no flag in BLOCK_FLAGS and is not required — "
                "add a flag for it, or remove it from the template"
            )
    return chosen


def render(
    config: Optional[Dict[str, str]] = None,
    path: Optional[Path] = None,
    blocks: Optional[Dict[str, str]] = None,
) -> tuple[str, str]:
    """Return ``(preamble, footer)`` — the prompt with the vocabulary left out.

    ``preamble`` runs from ``role`` through ``vocabulary.header`` inclusive and is what
    the term list is appended to; ``footer`` is everything after it (the examples). The
    vocabulary itself is rendered by ``llm_run.build_system_prompt``, which is the only
    place that knows about truncation.
    """
    blocks = blocks if blocks is not None else load_blocks(template_path(config, path))
    names = selected_blocks(blocks, config)
    if VOCABULARY_ANCHOR not in names:
        raise TemplateError(f"[[{VOCABULARY_ANCHOR}]] must render — it is where terms go")
    split = names.index(VOCABULARY_ANCHOR)
    preamble = "".join(blocks[n] for n in names[: split + 1])
    footer = "".join(blocks[n] for n in names[split + 1 :])
    return preamble, footer


# ── The vocabulary half of the prompt ────────────────────────────────────────────
#
# These two functions used to live inside ``llm_run.build_system_prompt``. They are here
# because two callers need the *same* answer and only one of them can import torch:
# ``llm_run`` renders the prompt it sends, and the CLI below renders the prompt a
# reviewer reads. A second copy of this loop would let the two drift, and a prompt
# preview that is not the prompt is worse than no preview at all.
#
# Truncation deliberately stays in ``llm_run``: it needs a real tokenizer, and it is the
# one part of prompt assembly that depends on the model rather than on the vocabulary.

META_TEXT_TERM = {
    "theme": "Administrative / Meta",
    "sub": "",
    "cs": "Nerelevantní (meta-text)",
    "en": "Irrelevant / Meta-text",
}


def vocabulary_terms(
    vocab_data: dict, excluded_themes: Optional[Sequence[str]] = None
) -> List[dict]:
    """Flatten a nested vocabulary into the term dicts the prompt is rendered from.

    Each term carries ``theme``, ``sub``, ``cs`` and ``en``; a term that came from a
    source record also carries ``ids`` (its own ``{source, id}`` plus everything B1's
    dedup discarded onto it, M7) and, when B3 qualified it, ``bare_cs``. Neither extra
    reaches the prompt text — ``llm_run`` uses them to build ``teater_category_ids`` and
    to strip the qualifier back off the emitted label.

    ``excluded_themes`` is lower-cased theme names to withhold, defaulting to
    ``{"other"}``. The meta-text sentinel is always first: it is the answer for a line
    that is not archaeology, so it cannot be a term the vocabulary happens to contain.
    """
    skip = {"other"} if excluded_themes is None else {t.lower() for t in excluded_themes}

    terms: List[dict] = [dict(META_TEXT_TERM)]
    for theme, data in vocab_data.items():
        if theme.startswith("_") or theme.lower() in skip:
            continue
        if not isinstance(data, dict):
            continue
        if "keywords" in data and isinstance(data["keywords"], dict):
            cs_list = data["keywords"].get("cs", [])
            en_list = data["keywords"].get("en", [])
            for i, cs_key in enumerate(cs_list):
                en = en_list[i] if i < len(en_list) else cs_key
                terms.append({"theme": theme, "sub": "", "cs": cs_key, "en": en})
            continue
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
            terms.append(term)
    return terms


def group_titles(term_list: Sequence[dict], grouping: str = DEFAULT_GROUPING) -> List[str]:
    """The header lines this layout would emit, in order — ``[]`` for ``flat``.

    Split out because the headers are the whole of what the grouping modes differ by, and
    because the context-budget report has to charge them against the window.
    """
    if grouping == "flat":
        return []
    titles: List[str] = []
    for t in term_list:
        theme, sub = t["theme"], (t.get("sub") or "")
        title = f"{theme} / {sub}" if (sub and grouping == "facet_sub") else theme
        if title not in titles:
            titles.append(title)
    return titles


def vocabulary_block(term_list: Sequence[dict], grouping: Optional[str] = None) -> str:
    """Render the term list under the configured amount of structure.

    Both AMCR and TEATER curate a second level — 50 heslars, and TEATER's depth-2 groups
    — and the shipped ``facet_sub`` layout keeps both, because flattening a 700-term
    facet into one undifferentiated list throws away structure a domain expert already
    built. ``facet`` and ``flat`` are the comparison arms for M2's open question, not
    proposals: they cost ~120 and ~130 header lines less, and whether that buys or costs
    accuracy is exactly what nobody has measured.

    Every mode emits the same terms. ``facet`` and ``flat`` emit them in ``term_list``
    order; ``facet_sub`` reorders within each facet so its sub-groups are contiguous,
    which is what grouping by sub-branch means. So ``facet`` vs ``flat`` varies the
    headers alone, and ``facet_sub`` vs ``facet`` varies the second level.

    ``grouping`` is a mode name; ``None`` means the shipped default. Pass what
    ``resolve_grouping(config)`` returns to honour ``llm_config.txt``.
    """
    grouping = grouping or DEFAULT_GROUPING
    if grouping not in GROUPING_MODES:
        raise TemplateError(
            f"unknown grouping {grouping!r}; expected one of {', '.join(GROUPING_MODES)}"
        )

    if grouping == "flat":
        lines = [f"- {t['cs']} ({t['en']})" for t in term_list]
        return "\n" + "\n".join(lines) + "\n" if lines else ""

    groups: Dict[str, List[str]] = {}
    for t in term_list:
        theme, sub = t["theme"], (t.get("sub") or "")
        title = f"{theme} / {sub}" if (sub and grouping == "facet_sub") else theme
        groups.setdefault(title, []).append(f"{t['cs']} ({t['en']})")

    out: List[str] = []
    for title, lines in groups.items():
        out.append(f"\n--- {title} ---\n")
        out.append("\n".join(f"- {line}" for line in lines) + "\n")
    return "".join(out)


def themes_withheld(path: Optional[Path] = None) -> set:
    """Theme names withheld from the prompt, read from taxonomy_config.json.

    The same derivation ``llm_run.main`` makes: a theme reaches the model unless its
    ``in_prompt`` is false, and absent the flag the default is everything except
    ``Other``. Read here so the CLI's rendered prompt matches the shipped one without
    importing the pipeline.
    """
    import json

    path = Path(path) if path else DEFAULT_TAXONOMY_CONFIG
    if not path.exists():
        raise TemplateError(f"taxonomy config not found: {path}")
    taxonomy = json.loads(path.read_text(encoding="utf-8"))
    return {
        name.lower()
        for name, cfg in taxonomy.items()
        if not name.startswith("_")
        and isinstance(cfg, dict)
        and not cfg.get("in_prompt", name.lower() != "other")
    }


def render_full(
    config: Optional[Dict[str, str]] = None,
    vocab_path: Optional[Path] = None,
    taxonomy_path: Optional[Path] = None,
    path: Optional[Path] = None,
) -> str:
    """The whole prompt — instructions *and* every term — without importing the pipeline.

    This is the untruncated prompt: what the model sees when the context window holds
    the full vocabulary (``SKIP_TRUNCATION`` or a budget that fits, which at 128k it
    does — see ``vocab_review.py --budget``). At a tighter window ``llm_run`` drops a
    tail of terms; the instruction half is identical either way.
    """
    import json

    vocab_file = Path(vocab_path) if vocab_path else DEFAULT_VOCAB
    if not vocab_file.exists():
        raise TemplateError(
            f"vocabulary not found: {vocab_file} — build it with "
            "`python3 vocab_build.py --from-flat`"
        )
    vocab_data = json.loads(vocab_file.read_text(encoding="utf-8"))
    preamble, footer = render(config, path)
    terms = vocabulary_terms(vocab_data, themes_withheld(taxonomy_path))
    return preamble + vocabulary_block(terms, resolve_grouping(config)) + footer


# Chars per token, matching the ratio the vocabulary sheets use. Estimated rather than
# tokenised: no tokenizer is available where this runs (vocab_build, the CLI, the tests).
CHARS_PER_TOKEN = 3.35


def block_cost(body: str) -> int:
    """Rough token cost of one block, for "is this rule worth its tokens"."""
    return round(len(body) / CHARS_PER_TOKEN)


def describe(config: Optional[Dict[str, str]] = None, path: Optional[Path] = None) -> str:
    """One line per block, for the run banner: what the model is being told, and what
    each rule costs. The cost matters because the instruction preamble competes with the
    vocabulary for the same budget — at a tight context window a rule kept is terms
    dropped (see ``vocab_review.py --budget``)."""
    blocks = load_blocks(template_path(config, path))
    active = set(selected_blocks(blocks, config))
    mode = resolve_geo_guardrail(config)
    total = sum(block_cost(body) for name, body in blocks.items() if name in active)
    lines = [
        f"  prompt blocks (geo guardrail: {mode}, vocabulary grouping: "
        f"{resolve_grouping(config)}) — ~{total} tokens of instructions"
    ]
    for name, body in blocks.items():
        mark = "on " if name in active else "off"
        lines.append(f"    [{mark}] {name:32s} ~{block_cost(body):4d} tok")
    return "\n".join(lines)


def guardrail_text(config: Optional[Dict[str, str]] = None, path: Optional[Path] = None) -> str:
    """Just the guardrail wording currently in force ("" when the mode is ``off``).

    ``vocab_build.py`` checks this against ``taxonomy_config.json``'s declared
    ``geo_guardrail``, so the vocabulary and the prompt cannot disagree about whether
    geographic terms are selectable.
    """
    mode = resolve_geo_guardrail(config)
    block = GEO_GUARDRAIL_BLOCKS[mode]
    if block is None:
        return ""
    return load_blocks(template_path(config, path)).get(block, "")


def output_template(path: Optional[Path] = None) -> dict:
    """The committed per-document output shape (``prompts/output_template.json``).

    Read it to know what ``<doc_id>_enriched.json`` looks like without running the
    pipeline; a test asserts it stays in step with the fields the schema actually emits.
    """
    import json

    path = Path(path) if path else DEFAULT_OUTPUT_TEMPLATE
    if not path.exists():
        raise TemplateError(f"output template not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def output_record_fields(path: Optional[Path] = None) -> Sequence[str]:
    """The enrichment field names the output template documents, in order."""
    tpl = output_template(path)
    example = (tpl.get("example_records") or [{}])[0]
    return list((example.get("enrichment") or {}).keys())


# ── The committed prompt sheets ──────────────────────────────────────────────────
#
# `prompts/prompt_*.txt` are the prompt as a reviewer reads it — @david-spacil and
# @motyc rule on wording and on vocabulary content without running the pipeline, and a
# file in the repo is what they can open. That makes them exactly the kind of generated
# artifact this repository has twice let go stale (union_nested.json, a5e3c8a and
# d4c46b2): committed, useful, and derived from files that keep moving.
#
# So they are generated, never hand-edited. `--write` regenerates all four and `--check`
# fails if any is out of date, in the same shape as `vocab_build.py --from-flat --check`.

SHEET_DIR = REPO_ROOT / "prompts"
GUARDRAIL_DIFF_PAIR = ("PROMPT_GEO_GUARDRAIL=strict", "PROMPT_GEO_GUARDRAIL=preference")


def sheet_contents(config: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """The four committed sheets, by filename, as the CLI would print them.

    Kept in one function so the writer and the drift check cannot disagree about what
    "up to date" means.
    """
    config = config or {}
    return {
        "prompt_blocks.txt": describe(config) + "\n",
        "prompt_preview.txt": "\n" + _preview_text(config),
        "prompt_full.txt": "\n" + render_full(config),
        "prompt_guardrail_diff.txt": _diff_text(config, *GUARDRAIL_DIFF_PAIR),
    }


def stale_sheets(
    config: Optional[Dict[str, str]] = None, directory: Optional[Path] = None
) -> List[str]:
    """Names of the committed sheets that no longer match a fresh render."""
    directory = Path(directory) if directory else SHEET_DIR
    stale = []
    for name, body in sheet_contents(config).items():
        path = directory / name
        if not path.exists() or path.read_text(encoding="utf-8") != body:
            stale.append(name)
    return stale


def write_sheets(
    config: Optional[Dict[str, str]] = None, directory: Optional[Path] = None
) -> List[str]:
    """Regenerate the committed sheets; returns the names actually rewritten."""
    directory = Path(directory) if directory else SHEET_DIR
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, body in sheet_contents(config).items():
        path = directory / name
        if not path.exists() or path.read_text(encoding="utf-8") != body:
            path.write_text(body, encoding="utf-8")
            written.append(name)
    return written


# ── CLI: read the prompt without running the pipeline ────────────────────────────
#
# A reviewer deciding whether a rule is worth its tokens should not have to start a GPU
# job to find out what the model is currently told. `--preview` prints the rendered
# instruction text with the term list elided; `--full` prints the real thing, all 4 719
# terms of it, which is the version to redirect into a file and read or diff between
# builds; `--diff` shows what changes between two flag settings, which is the form a
# wording decision actually takes.


def _cli_config(pairs: Sequence[str], config_file: Optional[Path]) -> Dict[str, str]:
    config = load_run_config(config_file) if config_file else {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects KEY=VALUE, got {pair!r}")
        key, _, value = pair.partition("=")
        config[key.strip()] = value.strip()
    return config


def _preview_text(config: Optional[Dict[str, str]] = None) -> str:
    preamble, footer = render(config)
    return preamble + "    … the vocabulary term list is injected here …\n" + footer


def _diff_text(config: Optional[Dict[str, str]], left_pair: str, right_pair: str) -> str:
    import difflib

    left = dict(config or {})
    right = dict(config or {})
    for target, pair in ((left, left_pair), (right, right_pair)):
        key, _, value = pair.partition("=")
        target[key.strip()] = value.strip()
    lp, lf = render(left)
    rp, rf = render(right)
    delta = list(
        difflib.unified_diff(
            (lp + lf).splitlines(keepends=True),
            (rp + rf).splitlines(keepends=True),
            fromfile=left_pair,
            tofile=right_pair,
        )
    )
    return "".join(delta) if delta else f"no difference between {left_pair} and {right_pair}\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="prompt_template.py",
        description="Render, inspect and diff the system prompt's instruction text.",
    )
    parser.add_argument("--preview", action="store_true", help="print the rendered prompt")
    parser.add_argument("--blocks", action="store_true", help="list every block, on or off")
    parser.add_argument(
        "--full",
        action="store_true",
        help="print the complete prompt, vocabulary included (redirect this to a file)",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"nested vocabulary to render (default: {DEFAULT_VOCAB.name})",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"regenerate the committed sheets in {SHEET_DIR.name}/",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if a committed sheet no longer matches the configured prompt",
    )
    parser.add_argument(
        "--diff",
        nargs=2,
        metavar=("KEY=VALUE", "KEY=VALUE"),
        help="unified diff of the prompt under two settings, e.g. "
        "--diff PROMPT_GEO_GUARDRAIL=strict PROMPT_GEO_GUARDRAIL=preference",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a flag for this run (repeatable)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "llm_config.txt",
        help="config file to read flags from (default: llm_config.txt)",
    )
    args = parser.parse_args(argv)

    config_file = args.config if args.config and Path(args.config).exists() else None
    config = _cli_config(args.set, config_file)

    if not (args.preview or args.blocks or args.diff or args.full or args.write or args.check):
        parser.print_help()
        return 2

    if args.blocks:
        print(describe(config))

    if args.preview:
        print("\n" + _preview_text(config), end="")

    if args.full:
        print("\n" + render_full(config, vocab_path=args.vocab), end="")

    if args.diff:
        print(_diff_text(config, args.diff[0], args.diff[1]), end="")

    if args.write:
        written = write_sheets(config)
        print(
            f"sheets up to date ({SHEET_DIR})"
            if not written
            else "regenerated: " + ", ".join(written)
        )

    if args.check:
        stale = stale_sheets(config)
        if stale:
            print(
                "stale prompt sheets: "
                + ", ".join(stale)
                + "\nregenerate with `python3 prompt_template.py --write`"
            )
            return 1
        print("prompt sheets match the configured prompt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
