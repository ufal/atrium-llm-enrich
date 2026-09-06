# Vocabulary management runbook

Every script that reads or writes the controlled vocabulary, what it does, when to run
it, and what must **not** be run blindly. Companion to
[`prompts/RUNBOOK.md`](../../prompts/RUNBOOK.md), which covers the prompt this term list
is injected into and the records the run writes back. Issue
[#6](https://github.com/ufal/atrium-nlp-enrich/issues/6).

Two decision packages live beside the sheets they are built from. Each frames an open
question with numbers from this directory and decides nothing:

| Package                        | Question                                             | Sheets it reads                                                                    |
|--------------------------------|------------------------------------------------------|------------------------------------------------------------------------------------|
| `6.O3O4.decision-package.md`   | which excluded terms come back? (O3/O4)              | `exclusion_impact.csv`, `teater_subbranch_impact.csv`, `reinstatement_preview.csv` |
| `6.D-eval.decision-package.md` | what counts as a correct answer? (O2, the D1 rubric) | `specificity_pairs.csv`, `composite_pairs.csv`, `collision_review.csv`             |

Three tools, in dependency order:

| Tool               | Needs network      | Needs the document corpus | Writes                                        |
|--------------------|--------------------|---------------------------|-----------------------------------------------|
| `vocab_build.py`   | only for a harvest | no                        | the 14 vocabulary artifacts in this directory |
| `vocab_review.py`  | no                 | no                        | 8 review sheets (corpus-independent)          |
| `corpus_review.py` | no                 | **yes**                   | 3 evidence sheets + `corpus_review.meta.json` |

### What is in this directory

29 files, three kinds. Only the first kind is input; everything else is generated and is
overwritten without warning by the command in its row.

| Files                                                | Kind                                                                                                       | Written by                               |
|------------------------------------------------------|------------------------------------------------------------------------------------------------------------|------------------------------------------|
| `{amcr,teater}_flat.{json,csv}`                      | **harvest** — every field the source gave us, 16 per record                                                | `vocab_build.py --source both` (network) |
| `{amcr,teater,union}_nested.json`                    | the vocabulary itself, grouped into facets                                                                 | `vocab_build.py --from-flat`             |
| `{amcr,teater,union}_nested.meta.json`               | provenance sidecar: source endpoints, record counts, `tool_version`, and the sha256 of both taxonomy files | `vocab_build.py --from-flat`             |
| `{amcr,teater,union}_placement_audit.csv`            | one row per term: which rule placed it, and where                                                          | `vocab_build.py --from-flat`             |
| `vocabulary.csv`                                     | the union as one flat spreadsheet, for reading outside the repo                                            | `vocab_build.py --from-flat`             |
| the 8 review sheets (§2)                             | evidence for an open question                                                                              | `vocab_review.py --all`                  |
| the 3 corpus sheets + `corpus_review.meta.json` (§3) | evidence from real report text                                                                             | `corpus_review.py --all`                 |
| `6.*.decision-package.md`, `RUNBOOK.md`              | prose                                                                                                      | hand-written                             |

**`union_nested.json` is the one the pipeline reads** (`VOCAB_PATH` in `llm_config.txt`).
The per-source `amcr_nested.json` / `teater_nested.json` are the same nesting applied to
one source alone — useful for seeing what each contributes, read by nothing at runtime.

---

## 1. `vocab_build.py` — build the vocabulary itself

Two stages. Stage 1 (harvest) is the only thing that touches the network; stage 2
(nesting) is pure and offline.

```bash
# Everyday case: re-nest from the committed flat harvests after editing the taxonomy.
python3 vocab_build.py --from-flat            # rewrites data_samples/vocab/*
python3 vocab_build.py --from-flat --check    # exit 0 = artifacts match the config
python3 vocab_build.py --from-flat --stats    # per-facet counts + which rule placed what

# Refresh the source data (needs api.aiscr.cz + raw.githubusercontent.com).
python3 vocab_build.py --source both --teater-mode snapshot
```

**The rule that matters:** any edit to `taxonomy_config.json` or
`taxonomy_overrides.json` must be followed by `--from-flat`, in the same commit.
Config and artifacts moving separately is exactly what commit `a5e3c8a` did — the
config dissolved the `Documentation` facet while the shipped vocabulary still offered
all 107 of its terms to the model, and nothing failed. `--from-flat --check` is the
gate; it runs in CI (`.github/workflows/vocab-drift.yml`) and as
`tests/test_vocab_build.py::test_committed_artifacts_match_a_fresh_build`.

`--update-legacy` additionally writes `data_samples/teater_nested_vocab.json` (the
pre-union path). The refresh workflow does not pass it, so that file keeps diverging;
do not rely on it.

**`vocab_build.py` is the only thing that writes to this directory.** `python3
vocab_manager.py` is a read-only inspector — it prints a prompt preview and the per-facet
counts and says so on exit. (It was not always: it used to run the legacy AMCR-only sync
against `union_nested.json` and save the result, which sent every term to `Other` and left
the pipeline injecting an empty vocabulary. Looking at the vocabulary destroyed it.)

## 2. `vocab_review.py` — the eight reviewer sheets

Offline, pure, deterministic. Reads the committed `*_flat.json` plus the taxonomy
config; writes only the CSVs below. Nothing here decides anything — each sheet ranks
and surfaces candidates for a human.

```bash
python3 vocab_review.py --all            # all eight
python3 vocab_review.py --collisions     # collision_review.csv        (M8/M13, @david-spacil)
python3 vocab_review.py --composites     # composite_pairs.csv         (O1/F)
python3 vocab_review.py --exclusions     # exclusion_impact.csv        (O3/O4, @motyc)
python3 vocab_review.py --subbranches    # teater_subbranch_impact.csv (O3/O4, finer grain)
python3 vocab_review.py --reinstate      # reinstatement_preview.csv   (O3/O4 go/no-go)
python3 vocab_review.py --specificity    # specificity_pairs.csv       (D1/O2, @motyc)
python3 vocab_review.py --budget         # context_budget.csv          (which models still fit)
python3 vocab_review.py --census         # facet_census.csv            (A1-facets, @david-spacil)
```

Two of the eight are **empty or near-empty on purpose** right now, and that is not a
broken report: `teater_subbranch_impact.csv` decomposes *excluded* TEATER branches and
M11 reinstated all of them, and `exclusion_impact.csv` is down to the 23 settled AMCR
lists. Both come back the moment something is excluded again.

Re-run after **any** taxonomy change, for the same reason as the artifacts:
`tests/test_vocab_review.py::test_committed_review_sheets_match_a_fresh_generation`
fails if a committed sheet no longer matches the config it claims to describe.

Reading `collision_review.csv`: 308 rows / 135 groups on the current build, sorted by
`dissimilarity` ascending. Check `aat_verdict` before spending time on a row —
`agreeing` means both records align to the same Getty AAT concept and can be
bulk-confirmed as one; `conflicting` is positive homonym evidence.

**The ranking is the weaker signal, and M13 proved it.** Of @david-spacil's 7 splits
only 1 was in the dissimilarity top 30, while all 3 `conflicting` groups split. Read
`aat_verdict` first and treat `dissimilarity` as a tie-breaker.

**`holds_bare_label` is the column to read before writing a qualifier.** Exactly one member
of each group owns the plain Czech word — the lowest-sorting record with no `qualifier_cs`,
which is how `to_term_pairs` picks it. M13's convention ("qualify only the record that
*leaves* the group") is precisely: **never put a qualifier on the row marked `yes`**, unless
you mean to hand the bare word to the next record in the group. The column is computed by
the same rule the build uses, and three tests hold it to what the built vocabulary actually
says.

> ⚠️ **`malta` — a live homonym that no column flags, and the one worth looking at
> first.** M11 reinstated the 250 country names, which brought `amcr:HES-001366`
> (*Malta*, the country) into a label group already held by three mortar records.
> `record_sort_key` sorts AMCR before TEATER and low ident first, so the bare label
> `malta` goes to `HES-000910` — **mortar, in `Material`** — and the country survives
> only as a `discarded_ids` entry on it. The country `Malta` is currently **not
> selectable by the model at all**, and `teater_category_ids` would report a line about
> Malta as a Material term.
>
> Both signals point the wrong way here: all four records read `aat_verdict = agreeing`
> (the class this runbook tells you to deprioritise), and the glosses *mortar* / *Malta*
> are far enough apart that dissimilarity does not rank them together either. The fix,
> if @david-spacil rules it a split, is one `qualifier_cs` on `HES-001366`.
>
> **Which member carries the qualifier is now visible, but still not enforced.** The
> group's four rows read `holds_bare_label = yes` on `HES-000910` and blank on the other
> three, so the sheet says plainly that the plain word is a mortar record. Qualifying the
> country is correct and leaves `malta` where it is; qualifying `HES-000910` instead hands
> `malta` to `HES-000992` — a different mortar record — and that build is green, validates,
> and separates the wrong concept. No test can rule which concept deserves the bare word;
> the column exists so the choice is made with the answer in front of you.

Reading `facet_census.csv`: one row per facet, in render order. `cumulative_tokens` is
the column that explains truncation — a facet is cut when everything *ahead* of it has
spent the budget, not when its own size crosses it — and `top_rules` names the exact
`heslar_map` / `teater_branch_map` values a re-layout would flip. This is the sheet for
**A1-facets**.

## 3. `corpus_review.py` — evidence from the documents ⚠️

**This is the one with a footgun.** It reports what the vocabulary actually matches in
real report text — the evidence standard @motyc set for excluding a branch (M3).

```bash
python3 corpus_review.py --all               # all three sheets
python3 corpus_review.py --term-evidence     # corpus_term_evidence.csv
python3 corpus_review.py --branch-evidence   # corpus_branch_evidence.csv  (O3/O4)
python3 corpus_review.py --gold-workbook     # gold_workbook.csv           (D1/D2)
```

### The corpus is not in git

`data_samples/DOC_LINE_CATEG` and `data_samples/UDP` track only three **synthetic**
demo documents (`CTX000000001-3`, 91 tokens). The real 16 reports (`CTX192100040`,
`CTX195603828`, …) arrive as the zip attachment on issue
[#19](https://github.com/ufal/atrium-nlp-enrich/issues/19) and are untracked. So:

- On a machine with the real corpus, these sheets are **evidence** (19 documents,
  2172 lines).
- On a clean checkout, they are a **smoke test** (3 documents, 4 hits).

Every run prints its corpus size before writing anything — read that line first:

```
  [corpus] 19 document(s), … tokens, … distinct lemmas, 101/1017 single-word terms hit
```

`corpus_review.meta.json` records the corpus each committed sheet was built from, and
the tool **refuses to overwrite sheets built from a larger corpus** (exit 1) rather
than silently replacing real evidence with placeholder numbers. Restore the corpus, or
pass `--force` if shrinking really is intended.

> ⚠️ **The committed corpus sheets are stale, and the guard is why.**
> `corpus_review.meta.json` records `vocabulary_single_word_terms: 1017` — computed
> against the **2 074**-term vocabulary, before M11 reinstated 2 638 terms. Against
> today's 4 718 those denominators are wrong. They cannot be refreshed on a machine
> without the real corpus, because the guard correctly refuses to replace 19-document
> evidence with 3-document placeholders. **Re-run `python3 corpus_review.py --all` on
> the machine that has the issue #19 documents** and commit the result; it is a
> ten-minute job and nothing else in the repo can do it.

### Reading `corpus_branch_evidence.csv` — `unique_*`, not the totals

Read `unique_occurrences` / `unique_hit_term_count`. They count only labels that no
kept list already offers. The plain `occurrences` total includes labels that collide
with an offered term, where the hit is explained by the term already in the
vocabulary, not by the excluded branch.

The reason this matters: when it was generated, `heslar:zeme` (the 250 country names —
O3/O4 Q1, the branch the geographic guardrail existed for) scored **7 occurrences, all
of them the word `malta`**. In those lines `malta` is *mortar*
("spojovaných vápennou maltou"), an offered Material term; the record being credited was
the country `Malta`, which casefolding makes indistinguishable. Quoted as a total, that
row argues country names are attested in archaeological reports. Its
`unique_occurrences` is **0**.

`heslar:letiste` and `heslar:nalez_typ` are the **opposite** shape, and the same rule
says so: `letiste` scores `unique_occurrences` 3 / `shared` 0 and `nalez_typ` 21 / 0.
Their hits are not explained away by an offered term — read as evidence, they count.

Matching is lemma-based (`data_samples/UDP/*.conllu`, UDPipe `czech-pdt-ud-2.15`), and
covers **single-word terms only** — `zlomek keramiky` is never matched.

---

## Making the A1-facets call — the one decision still open

@david-spacil owns where the 2 638 terms M11 reinstated actually live. Today the tooling
proposes two facets at `priority: 0`, both last in render order:

| Facet                           | Terms | Holds                                                                |
|---------------------------------|------:|----------------------------------------------------------------------|
| `Cultural & Geographic Context` |   972 | Q1 — countries, ethnic groups, historical regions, dynasties         |
| `Related Disciplines & Society` | 1 666 | Q2 — cross-disciplinary, professions, society, theory, battles, wars |

That is a **proposal, not a ruling**. `facet_census.csv` is the sheet to rule from:
`top_rules` names exactly the map values a different layout would flip, and
`cumulative_tokens` shows what each facet costs where it sits.

### The move, and what catches a mistake

```bash
$EDITOR data_samples/taxonomy_config.json    # 1. change teater_branch_map values;
                                             #    declare any NEW facet with priority + in_prompt,
                                             #    and list it in tie_break
python3 vocab_build.py --from-flat           # 2. rebuild
python3 vocab_review.py --all                # 3. refresh all eight sheets
python3 prompt_template.py --write           # 4. prompt_full.txt contains every term
python3 -m pytest -q                         # 5. all gates
```

Four things are checked for you, and each reports rather than guesses:

| Mistake                                                                    | Caught by                                                                                       |
|----------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| Facet name typo in a map value                                             | `validate_settings` — *undeclared facet*                                                        |
| A renamed facet still named in `tie_break`, `heslar_labels` or an override | `validate_settings` — every stale reference, all at once                                        |
| A new facet sharing a priority but missing from `tie_break`                | `validate_settings` — render order would be decided by fallback, not by you                     |
| Config edited, artifacts or sheets not regenerated                         | `vocab_build.py --from-flat --check`, the review-sheet drift test, `prompt_template.py --check` |

**Priority is the load-bearing part, not the facet name.** Render order *is* truncation
order: the prompt keeps the largest fitting *prefix*, so a facet's position decides
whether a 32k model sees it at all. Q2 sitting last is M11's *evaluate later if problems
arise* made concrete — it is the first thing a tight budget drops, and retiring it is one
map value. Move it up the order and that property is gone; that is a decision worth
making deliberately, which is why the tie_break guard exists.

**What is *not* checked:** whether the grouping is *right*. No test knows that `bitvy`
belongs with professions rather than with chronology. That is the ruling, and it is why
the sheet exists.

## Order of operations after a taxonomy change

```bash
$EDITOR data_samples/taxonomy_config.json      # or taxonomy_overrides.json
python3 vocab_build.py --from-flat             # 1. rebuild the vocabulary
python3 vocab_build.py --from-flat --check     # 2. must exit 0
python3 vocab_review.py --all                  # 3. refresh the reviewer sheets
python3 prompt_template.py --write             # 4. refresh the committed prompt sheets
python3 corpus_review.py --all                 # 5. ONLY with the real corpus present
python3 -m pytest -q                           # 6. all gates
git add data_samples/vocab data_samples/taxonomy_*.json prompts && git commit
```

Step 4 is not optional when the vocabulary moved: `prompts/prompt_full.txt` contains every
term, so any taxonomy change makes it stale, and `prompt_template.py --check` fails in CI.
The same applies to a `PROMPT_*` flag or a `prompts/system_prompt.txt` edit, where steps
1–3 change nothing but step 4 does.

Skip step 5 if you do not have the real documents — the guard will refuse anyway, and
that refusal is correct, not a failure to work around.

## Where a decision gets recorded

Every decision below is a config edit. None of them needs a code change, and the build
refuses rather than guesses when an edit would not do what it says (`validate_settings`
reports every problem at once, not one per rebuild).

| Decision                                       | Goes in                                                                                                                                                                                                                                                                                        |
|------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| A whole AMCR list / TEATER branch in or out    | `taxonomy_config.json` → `heslar_map` / `teater_branch_map`, **plus** a `_exclusions` entry stating `status` and `reason` when the value is `__exclude__`                                                                                                                                      |
| Whether an exclusion is still an open question | `_exclusions[rule].status` — `settled`, `open_geo_ethnic` (Q1), `open_other` (Q2). Drives the `status` column of `exclusion_impact.csv`                                                                                                                                                        |
| One term's facet                               | `taxonomy_overrides.json` → `facet`, keyed on `(source, id)` — never on the bare label                                                                                                                                                                                                         |
| One term's sub-header                          | `taxonomy_overrides.json` → `sub`. Give the rendered Czech label (`druh objektu`, not `objekt_druh`), and prefer one the target facet already uses                                                                                                                                             |
| One term out, inside a list worth keeping      | `taxonomy_overrides.json` → `facet: "__exclude__"`. Applied before dedup, so a term another kept record also offers survives under that record                                                                                                                                                 |
| A bracketed homonym qualifier                  | `taxonomy_overrides.json` → `qualifier_cs`. Only after a human has confirmed the split — most same-label collisions are one concept (`collision_review.csv`)                                                                                                                                   |
| Two entries mean the same thing / do not       | `taxonomy_overrides.json` → `same_as` / `same_as_suppress`. Shows up as `link_status` in `composite_pairs.csv`; neither changes what the prompt offers                                                                                                                                         |
| Reinstating a geographic branch                | three edits in **one** change: `PROMPT_GEO_GUARDRAIL=preference`, `geo_guardrail.active` false, flip the branch. Any one alone fails the build. Done for Q1+Q2 under M11                                                                                                                       |
| What splits a composite `X/Y` label            | `taxonomy_config.json` → `composite_separators`                                                                                                                                                                                                                                                |
| Which keys reach the prompt payload            | `taxonomy_config.json` → `nested_keep`. `discarded_ids` and `bare_cs` are read back by the enrichment output, so dropping either changes behaviour                                                                                                                                             |
| What counts as boilerplate for truncation      | `taxonomy_config.json` → `admin_stop_words`. Sorts matching terms to the back of their facet, which decides what a small-context model still sees                                                                                                                                              |
| Facet order and priority                       | `taxonomy_config.json` → per-facet `priority`, `tie_break`. Load-bearing: the prompt truncates a *prefix* of the flattened term list, so position decides what a small-context model sees. Every facet sharing a priority with another **must** be listed in `tie_break`, or the build refuses |
| Which instruction reaches the model at all     | `prompts/system_prompt.txt` holds the text as `[[blocks]]` in render order; `llm_config.txt`'s `PROMPT_*` flags choose which render. The run banner prints the on/off list                                                                                                                     |
| The geographic guardrail's wording             | `llm_config.txt` → `PROMPT_GEO_GUARDRAIL` = `strict` / `preference` / `off`, paired with `taxonomy_config.json` → `geo_guardrail.active`. `vocab_build.py` renders the selected block and refuses a build where the two disagree                                                               |
| What one enrichment record looks like          | `prompts/output_template.json` — the committed shape of `<doc_id>_enriched.json`, held in step with the schema and the prompt's own examples by test                                                                                                                                           |
| **Reading the prompt without a GPU run**       | `python3 prompt_template.py --blocks` (what is on) · `--preview` (instructions, terms elided) · `--full` (the whole prompt, all 4 718 terms) · `--diff KEY=A KEY=B` (what a flag change does to the wording). All four are committed under `prompts/`                                          |
| **A committed sheet after any prompt change**  | `python3 prompt_template.py --write`, then commit `prompts/prompt_*.txt`. `--check` is the gate, and `vocab-drift.yml` runs it on every PR, so a stale sheet fails CI rather than misleading a reviewer                                                                                        |
| **Whether the facet grouping matters**         | `llm_config.txt` → `PROMPT_VOCAB_GROUPING` = `facet_sub` / `facet` / `flat`. Same terms, same truncation; only the headers move. `facet` vs `flat` isolates the headers, `facet_sub` vs `facet` the source's second level. Costs 126 terms at 32k, nothing at 128k                             |
| **Whether a model can still hold it**          | `context_budget.csv` — per context window and facet, how much survives truncation. At 32k only the probation facet is cut; at 8k 419 of 4 718 terms reach the model                                                                                                                            |
| **Which facet a reinstated branch lands in**   | `taxonomy_config.json` → the `teater_branch_map` value, **plus** the facet's own `priority` / `in_prompt` block if it is a new one. `facet_census.csv` is the evidence: contents, feeding rules, cost and cut point, one row per facet. This is **A1-facets**                                  |
| Nothing                                        | a generated CSV. Every sheet in this directory is rebuilt from the two files above; hand-editing one is discarded on the next run.                                                                                                                                                             |
