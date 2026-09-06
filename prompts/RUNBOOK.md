# Prompt & output runbook

What the model is told, how to change it, and what comes back. Companion to
[`data_samples/vocab/RUNBOOK.md`](../data_samples/vocab/RUNBOOK.md), which covers the term
list this prompt injects. Issue
[#6](https://github.com/ufal/atrium-nlp-enrich/issues/6).

**Nothing here needs a GPU, a model, or the Python ML stack.** `prompt_template.py`
imports only the standard library, so every command below runs in a bare checkout — which
is the point: a reviewer ruling on wording should be able to read the wording.

### What is in this directory

| File                                                                                      | Kind      | Written by                            |
|-------------------------------------------------------------------------------------------|-----------|---------------------------------------|
| `system_prompt.txt`                                                                       | **input** | hand-edited — the instruction text    |
| `output_template.json`                                                                    | **input** | hand-edited — the output contract, §4 |
| `prompt_blocks.txt`, `prompt_preview.txt`, `prompt_full.txt`, `prompt_guardrail_diff.txt` | generated | `python3 prompt_template.py --write`  |
| `RUNBOOK.md`                                                                              | prose     | hand-written                          |

The four `prompt_*.txt` sheets are the prompt as a file you can open in a diff. They are
**generated, never hand-edited** — see §3.

---

## 1. `system_prompt.txt` — the instruction text

Eleven blocks. A header is `[[name]]` alone on a line; everything to the next header is
that block's body, and **order in the file is render order**. To move a rule, move it in
the file. Lines starting with `#` *before the first block* are file comments; inside a
block everything is prompt text.

Three blocks always render (`role`, `rule.json_only`, `vocabulary.header`) — without them
the prompt would not state the task, the output contract, or where the terms begin.
Everything else is gated by a flag.

| Block                             | Flag                       | ~tok | Says                                                        |
|-----------------------------------|----------------------------|-----:|-------------------------------------------------------------|
| `role`                            | *always*                   |   47 | who the model is, and that `<target_line>` is the subject   |
| `task.extract`                    | `PROMPT_TASK_EXTRACT`      |   53 | pull archaeological entities, not researchers or dates      |
| `task.select`                     | `PROMPT_TASK_SELECT`       |   25 | choose exactly one category                                 |
| `rule.metatext`                   | `PROMPT_METATEXT_RULE`     |   79 | when to answer `Nerelevantní (meta-text)`                   |
| `guardrail.geographic.strict`     | `PROMPT_GEO_GUARDRAIL`     |   72 | **never** a country/language/region name                    |
| `guardrail.geographic.preference` | `PROMPT_GEO_GUARDRAIL`     |   41 | geographic terms only when the line is genuinely about them |
| `rule.ocr_normalisation`          | `PROMPT_OCR_NORMALISATION` |   62 | normalise OCR damage before emitting a keyword              |
| `rule.exact_term`                 | `PROMPT_EXACT_TERM`        |   19 | use the vocabulary's exact Czech string                     |
| `rule.json_only`                  | *always*                   |   23 | reply with JSON matching the schema                         |
| `vocabulary.header`               | *always*                   |    7 | `THEMATIC VOCABULARY:` — the terms are injected here        |
| `examples`                        | `PROMPT_EXAMPLES`          |  146 | two worked records, one of them the meta-text case          |

Regenerate the table with `python3 prompt_template.py --blocks`; the same line prints in
the run banner, so a log always says what the model was told. Token figures are estimated
at 3.35 chars/token, not tokenised — read them as relative weights.

**Why the costs are on the same page as the wording.** The instructions and the term list
compete for one budget. At 8 192 tokens the whole prompt admits 419 of 4 718 terms, so
`examples` at ~146 tokens is worth roughly 40 terms there. At 128k it is free. The trade is
real only at the small end — see [`context_budget.csv`](../data_samples/vocab/context_budget.csv).

### The nine flags

All live in `llm_config.txt`, all read by `prompt_template.py`.

| Flag                                                                                                                                         | Values                          | Shipped                     | Code default |
|----------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------|-----------------------------|--------------|
| `PROMPT_TEMPLATE`                                                                                                                            | a path                          | `prompts/system_prompt.txt` | same         |
| `PROMPT_TASK_EXTRACT` · `PROMPT_TASK_SELECT` · `PROMPT_METATEXT_RULE` · `PROMPT_OCR_NORMALISATION` · `PROMPT_EXACT_TERM` · `PROMPT_EXAMPLES` | `true` / `false`                | all `true`                  | `true`       |
| `PROMPT_GEO_GUARDRAIL`                                                                                                                       | `strict` / `preference` / `off` | **`preference`** (M11/M12)  | `strict`     |
| `PROMPT_VOCAB_GROUPING`                                                                                                                      | `facet_sub` / `facet` / `flat`  | `facet_sub`                 | `facet_sub`  |

The two columns differ in one place, deliberately. The code default is `strict` — the
wording the pipeline used for its whole history, so a config that says nothing behaves as
it always did. `llm_config.txt` states `preference` explicitly, because M11 changed it and
a decision that size should be visible in the config rather than implied by a default.
`llm_run.py` reads the config, so the shipped column is what actually runs.

A flag naming a block the template does not define is an **error**, not a no-op, and so is
an unrecognised `PROMPT_GEO_GUARDRAIL` or `PROMPT_VOCAB_GROUPING` value. A typo that
silently rendered the default would make a comparison report a change nobody applied.

Adding a block means adding its flag to `BLOCK_FLAGS` in `prompt_template.py` — a block
with no flag and no required status would sit in the file rendering nothing, and
`tests/test_prompt_template.py::test_every_shipped_block_is_reachable` refuses it.

---

## 2. The two pairings that are unsafe to change alone

Everything else in this file is a one-line edit with local effect. These two are not.

### 2.1 The geographic guardrail ↔ the vocabulary

`PROMPT_GEO_GUARDRAIL` and `taxonomy_config.json`'s `geo_guardrail.active` are **one
decision written in two files**. The prompt says whether a country name may be selected;
the taxonomy says whether country names are in the term list at all. Set them
inconsistently and a score measures the contradiction rather than the model — the model is
offered `Malta` and told never to pick it.

`vocab_build.py` renders the configured prompt and refuses to build a vocabulary that
disagrees with it, so the pairing cannot drift:

| `PROMPT_GEO_GUARDRAIL` | `geo_guardrail.active` | Result                                  |
|------------------------|------------------------|-----------------------------------------|
| `strict`               | `true`                 | builds — terms excluded, prompt forbids |
| `preference` / `off`   | `false`                | builds — terms offered, prompt prefers  |
| any other combination  |                        | **build refused**, naming both halves   |

`prompt_markers` in `taxonomy_config.json` is how the build recognises the strict wording
(`"country name"`, `"language name"`, `"geographic region name"`). Rewording the strict
block without those phrases silently disables the detection, so
`test_the_strict_wording_carries_every_marker_the_config_looks_for` asserts they survive.

The shipped state is `preference` + `active: false`: M11 reinstated the geographic,
ethnic and dynastic branches and M12 accepted the relaxed wording, in one change.

### 2.2 The vocabulary ↔ the committed sheets

`prompt_full.txt` contains every term. Any taxonomy change makes it stale, and no
vocabulary command rewrites it. Run `--write` (§3) in the same commit.

---

## 3. `prompt_template.py` — read, compare, regenerate

```bash
python3 prompt_template.py --blocks       # which rules are on, and what each costs
python3 prompt_template.py --preview      # the instruction text, term list elided
python3 prompt_template.py --full         # the whole prompt: instructions + all 4 719 labels
python3 prompt_template.py --diff PROMPT_GEO_GUARDRAIL=strict PROMPT_GEO_GUARDRAIL=preference
python3 prompt_template.py --write        # regenerate the four committed sheets
python3 prompt_template.py --check        # exit 1 if a sheet is out of date
```

Modifiers: `--set KEY=VALUE` (repeatable) overrides a flag for one command without editing
`llm_config.txt`; `--config PATH` reads flags from a different file; `--vocab PATH` renders
a different build.

`--full` prints **4 719** bullets — the 4 718 vocabulary terms plus
`Nerelevantní (meta-text)`, which is injected at index 0 and is not part of the vocabulary.
It is the *untruncated* prompt: what a model with room for everything sees. At a tighter
window `llm_run.py` drops a tail of terms; the instruction half is identical either way.

**`--diff` is the form a wording decision actually takes.** The whole M11/M12 guardrail
change is two lines:

```
-NEVER select a country name, language name, or geographic region name as the teater_category …
+Select a geographic, ethnic or dynastic term only when the line is genuinely about it; …
```

### The committed sheets, and why they are gated

`prompt_blocks.txt`, `prompt_preview.txt`, `prompt_full.txt` and
`prompt_guardrail_diff.txt` are the output of the first four commands. They exist so a
domain reviewer can read the prompt in a pull-request diff without running Python.

That makes them generated files derived from things that keep moving — the exact shape
this repository has already let go stale twice with `union_nested.json` (`a5e3c8a`,
`d4c46b2`). So:

* `--write` regenerates them and reports only what changed;
* `--check` exits 1 and names the stale files;
* `.github/workflows/vocab-drift.yml` runs `--check` on every PR touching the prompt, the
  config, the vocabulary or the sheets, **and** a negative test proving the gate still
  fails on a tampered sheet;
* `tests/test_prompt_template.py` asserts the same thing offline.

Hand-editing a sheet is discarded on the next `--write`. Edit `system_prompt.txt` or a
flag instead.

### `PROMPT_VOCAB_GROUPING` — the layout experiment

@motyc, [26 August](https://github.com/ufal/atrium-nlp-enrich/issues/6#issuecomment-5424905539):
*"distinguish clearly between which terms are available to the model and how those terms
are grouped for inference … if the full vocabulary fits into the model context, it may
also be worth checking empirically how much facet placement affects the results at all."*
It fits now, so the layout is a flag.

| Mode        | Renders                              | Headers | Term order                 |
|-------------|--------------------------------------|--------:|----------------------------|
| `facet_sub` | `--- Facet / Subgroup ---` (shipped) |     125 | sub-groups made contiguous |
| `facet`     | `--- Facet ---`                      |      11 | unchanged                  |
| `flat`      | no headers                           |       0 | unchanged                  |

All three offer the **same terms** and truncate identically. Because `facet` and `flat`
preserve term order while `facet_sub` permutes within each facet (never across one), the
three modes give **two clean contrasts** rather than one scale:

* `facet` vs `flat` — the header lines alone;
* `facet_sub` vs `facet` — the source's own second level, headers and adjacency together.

The cost side is already measured: the headers buy their place at 8 192 tokens (419 terms
grouped vs 445 flat) and at 32 768 (3 106 vs 3 232), and cost nothing at 128k where
everything fits. Whether they *help* is unmeasured — it needs the corpus and the D1 rubric,
not code. Tracked as **A2-grouping**.

```bash
# what the model would see under each layout
for g in facet_sub facet flat; do
  python3 prompt_template.py --full --set PROMPT_VOCAB_GROUPING=$g > /tmp/prompt.$g.txt
done
```

---

## 4. The output contract

`output_template.json` is the committed shape of one `<doc_id>_enriched.json`, so a
consumer can read the contract without running the pipeline. The file the pipeline writes
is a JSON **array** of records; the template documents that array.

| Field                                                                | From                                                                 |
|----------------------------------------------------------------------|----------------------------------------------------------------------|
| `file_id`, `page`, `line`, `categ`, `quality_score`, `original_text` | the input CSV, carried through                                       |
| `enrichment.extracted_keywords_cs`                                   | the model — Czech terms found *in the line*                          |
| `enrichment.extracted_keywords_en`                                   | the model — same length and order as the Czech list                  |
| `enrichment.teater_category`                                         | the model — one label from the vocabulary, or the meta-text sentinel |
| `enrichment.confidence_score`                                        | the model — float in [0, 1]                                          |
| `enrichment.teater_category_ids`                                     | **attached after inference**, not generated (M7/M9)                  |

The model returns exactly the first four `enrichment` fields, because `build_schema` turns
the surviving term list into a constrained enum and those four into a schema it must match.
Three tests hold the schema, this template and the prompt's own examples in step, so the
three statements of the contract cannot drift apart.

### `teater_category_ids` — the id passthrough

@motyc asked to keep the id connection through dedup
([`5426147439`](https://github.com/ufal/atrium-nlp-enrich/issues/6#issuecomment-5426147439));
@david-spacil's point was that a correct `kostel` could not be linked back to anything,
because it exists three times across the two sources. So every emitted label carries the
full set of `{source, id}` records it stands for — its own, plus every record dedup
absorbed onto it.

Set `EMIT_CATEGORY_IDS=false` in `llm_config.txt` to drop the field; that is M7's *"drop it
if it will create some issues"*, and it is one config key rather than a code change. The
prompt is identical either way — ids never enter it.

### Bracketed qualifiers are stripped back off

A homonym split (M8/M13) makes the *prompt* offer `zámek (sídlo elity)`, so the model can
tell the château from the lock. The emitted `teater_category` is rewritten back to `zámek`
— the qualifier is a disambiguation device for the enum, not a keyword — while
`teater_category_ids` still says which sense was meant. A qualifier never leaks downstream.

### When a document fails

Ten consecutive inference errors abandon a document and write
`<doc_id>_enriched.abort.json` beside it: `aborted`, `abort_reason`,
`processed_before_abort`, `errors_before_abort`, `timestamp_utc`. A document with an abort
marker is incomplete by construction — do not score it as a low result.

### Provenance

Each run's paradata records which vocabulary it used: the artifact's `tool_version` and
term count, the sha256 of both taxonomy files, and per-source record counts including
TEATER's pinned commit — read from the `*.meta.json` sidecar beside `VOCAB_PATH` (D3).
Both sources are CC BY-NC 4.0 and declared *conditional* in `para_config.txt`, so the run
logs a component per source actually present in the build; an AMCR-only artifact does not
claim TEATER data. A vocabulary with no sidecar logs nothing and does not fail the run.

---

## 5. Order of operations after a prompt change

```bash
$EDITOR prompts/system_prompt.txt          # or a PROMPT_* flag in llm_config.txt
python3 prompt_template.py --preview       # 1. read what you just wrote
python3 prompt_template.py --diff PROMPT_GEO_GUARDRAIL=strict PROMPT_GEO_GUARDRAIL=preference
                                           #    (or any two settings you are weighing)
python3 vocab_build.py --from-flat --check # 2. only if you touched the guardrail — §2.1
python3 prompt_template.py --write         # 3. refresh the committed sheets
python3 -m pytest -q                       # 4. all gates
git add prompts llm_config.txt && git commit
```

Step 2 is the one that catches a half-made guardrail change. If it exits 1, the prompt and
the vocabulary now disagree about geographic terms and the fix is to move the other half in
the same commit — not to skip the check.

Coming the other way, **after a vocabulary change**, step 3 is the one that matters:
`prompt_full.txt` holds every term, and no vocabulary command rewrites it. The vocabulary
runbook's own order of operations includes it for that reason.

## Where a prompt or output decision gets recorded

| Decision                                  | Goes in                                                                                              |
|-------------------------------------------|------------------------------------------------------------------------------------------------------|
| The wording of a rule                     | `prompts/system_prompt.txt`, inside its `[[block]]`                                                  |
| Whether a rule reaches the model at all   | the block's `PROMPT_*` flag in `llm_config.txt`                                                      |
| Where a rule sits in the prompt           | its position in `system_prompt.txt` — file order is render order                                     |
| A new rule                                | a new `[[block]]` **plus** its flag in `BLOCK_FLAGS` (`prompt_template.py`), or the build refuses it |
| Whether geographic terms may be chosen    | `PROMPT_GEO_GUARDRAIL` **and** `taxonomy_config.json` → `geo_guardrail.active`, together — §2.1      |
| How the term list is laid out             | `PROMPT_VOCAB_GROUPING`                                                                              |
| Which vocabulary build is used            | `VOCAB_PATH` in `llm_config.txt`                                                                     |
| Whether a facet reaches the prompt at all | `taxonomy_config.json` → that facet's `in_prompt`                                                    |
| Whether ids are emitted                   | `EMIT_CATEGORY_IDS`                                                                                  |
| What one output record looks like         | `prompts/output_template.json` — and the schema and prompt examples move with it                     |
| Nothing                                   | `prompts/prompt_*.txt` — generated; edit the template or a flag, then `--write`                      |
