# Vocabulary management runbook

Every script that reads or writes the controlled vocabulary, what it does, when to run
it, and what must **not** be run blindly. Issue
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
| `vocab_build.py`   | only for a harvest | no                        | the 10 vocabulary artifacts in this directory |
| `vocab_review.py`  | no                 | no                        | 6 review sheets (corpus-independent)          |
| `corpus_review.py` | no                 | **yes**                   | 3 evidence sheets + `corpus_review.meta.json` |

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

## 2. `vocab_review.py` — the six reviewer sheets

Offline, pure, deterministic. Reads the committed `*_flat.json` plus the taxonomy
config; writes only the CSVs below. Nothing here decides anything — each sheet ranks
and surfaces candidates for a human.

```bash
python3 vocab_review.py --all            # all six
python3 vocab_review.py --collisions     # collision_review.csv        (M8, @david-spacil)
python3 vocab_review.py --composites     # composite_pairs.csv         (O1/F)
python3 vocab_review.py --exclusions     # exclusion_impact.csv        (O3/O4, @motyc)
python3 vocab_review.py --subbranches    # teater_subbranch_impact.csv (O3/O4, finer grain)
python3 vocab_review.py --reinstate      # reinstatement_preview.csv   (O3/O4 go/no-go)
python3 vocab_review.py --specificity    # specificity_pairs.csv       (D1/O2, @motyc)
```

Re-run after **any** taxonomy change, for the same reason as the artifacts:
`tests/test_vocab_review.py::test_committed_review_sheets_match_a_fresh_generation`
fails if a committed sheet no longer matches the config it claims to describe.

Reading `collision_review.csv`: sorted by `dissimilarity` ascending, so the likeliest
homonyms come first (`zámek`, the one confirmed case, ranks 10th of 127). Check
`aat_verdict` before spending time on a row — `agreeing` means both records align to
the same Getty AAT concept and can be bulk-confirmed as one concept; `conflicting`
is positive homonym evidence.

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
demo documents (`CTX000000001-3`, 69 words). The real 16 reports (`CTX192100040`,
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

### Reading `corpus_branch_evidence.csv` — `unique_*`, not the totals

Read `unique_occurrences` / `unique_hit_term_count`. They count only labels that no
kept list already offers. The plain `occurrences` total includes labels that collide
with an offered term, where the hit is explained by the term already in the
vocabulary, not by the excluded branch.

The reason this matters: on the real corpus `heslar:zeme` (the 249 country names —
O3/O4 Q1, the branch the geographic guardrail exists for) scores **7 occurrences, all
of them the word `malta`**. In those lines `malta` is *mortar*
("spojovaných vápennou maltou"), an offered Material term; the excluded record being
credited is the country `Malta`, which casefolding makes indistinguishable. Quoted as
a total, that row argues country names are attested in archaeological reports. Its
`unique_occurrences` is **0**. Same shape, milder, in `heslar:letiste`
(`kladno`/`kolín` are towns) and `heslar:nalez_typ` (`objekt`/`předmět`).

Matching is lemma-based (`data_samples/UDP/*.conllu`, UDPipe `czech-pdt-ud-2.15`), and
covers **single-word terms only** — `zlomek keramiky` is never matched.

---

## Order of operations after a taxonomy change

```bash
$EDITOR data_samples/taxonomy_config.json      # or taxonomy_overrides.json
python3 vocab_build.py --from-flat             # 1. rebuild the vocabulary
python3 vocab_build.py --from-flat --check     # 2. must exit 0
python3 vocab_review.py --all                  # 3. refresh the reviewer sheets
python3 corpus_review.py --all                 # 4. ONLY with the real corpus present
python3 -m pytest -q                           # 5. all four gates
git add data_samples/vocab data_samples/taxonomy_*.json && git commit
```

Skip step 4 if you do not have the real documents — the guard will refuse anyway, and
that refusal is correct, not a failure to work around.

## Where a decision gets recorded

Every decision below is a config edit. None of them needs a code change, and the build
refuses rather than guesses when an edit would not do what it says (`validate_settings`
reports every problem at once, not one per rebuild).

| Decision                                       | Goes in                                                                                                                                                      |
|------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| A whole AMCR list / TEATER branch in or out    | `taxonomy_config.json` → `heslar_map` / `teater_branch_map`, **plus** a `_exclusions` entry stating `status` and `reason` when the value is `__exclude__`    |
| Whether an exclusion is still an open question | `_exclusions[rule].status` — `settled`, `open_geo_ethnic` (Q1), `open_other` (Q2). Drives the `status` column of `exclusion_impact.csv`                      |
| One term's facet                               | `taxonomy_overrides.json` → `facet`, keyed on `(source, id)` — never on the bare label                                                                       |
| One term's sub-header                          | `taxonomy_overrides.json` → `sub`. Give the rendered Czech label (`druh objektu`, not `objekt_druh`), and prefer one the target facet already uses           |
| One term out, inside a list worth keeping      | `taxonomy_overrides.json` → `facet: "__exclude__"`. Applied before dedup, so a term another kept record also offers survives under that record               |
| A bracketed homonym qualifier                  | `taxonomy_overrides.json` → `qualifier_cs`. Only after a human has confirmed the split — most same-label collisions are one concept (`collision_review.csv`) |
| Two entries mean the same thing / do not       | `taxonomy_overrides.json` → `same_as` / `same_as_suppress`. Shows up as `link_status` in `composite_pairs.csv`; neither changes what the prompt offers       |
| Reinstating a geographic branch                | three edits in **one** change: relax the wording in `llm_run.py`, set `geo_guardrail.active` false, flip the branch. Any one alone fails the build           |
| What splits a composite `X/Y` label            | `taxonomy_config.json` → `composite_separators`                                                                                                              |
| Which keys reach the prompt payload            | `taxonomy_config.json` → `nested_keep`. `discarded_ids` and `bare_cs` are read back by the enrichment output, so dropping either changes behaviour           |
| What counts as boilerplate for truncation      | `taxonomy_config.json` → `admin_stop_words`. Sorts matching terms to the back of their facet, which decides what a small-context model still sees            |
| Facet order and priority                       | `taxonomy_config.json` → per-facet `priority`, `tie_break`. Load-bearing: the prompt truncates a *prefix* of the flattened term list                         |
| Nothing                                        | a generated CSV. Every sheet in this directory is rebuilt from the two files above; hand-editing one is discarded on the next run.                           |
