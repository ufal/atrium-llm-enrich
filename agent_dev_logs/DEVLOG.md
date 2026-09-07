# 📓 atrium-llm-enrich — agent_dev_logs/DEVLOG.md (timeline index)
> _LLM-driven enrichment of archaeological documents (local multi-GPU + remote-as-a-service). 5 open
> issues (#10, #11, #13, #18, #24); #8 closed. `test`==`main` HEAD `ee16913` (2026-09-06) · **v0.6.2**._
> _Per-issue detail: `digests/{id}.digest.md` · `plans/{id}.plan.md` · `issues/` exports
> (source of truth). Cross-repo/hub history (DU benchmark hub#22, this repo's spin-out from
> hub#24) lives in `ufal/atrium-project/agent_dev_logs/DEVLOG.md` (deduplicated out of this file).
> Note: this repo's own **#24 "Application of olmOCR"** (opened 09-03) is a different,
> coincidentally-numbered issue — not the hub spin-out issue of the same number._

## 2026-07-12
- **#8 LLM applications to data — initialization of repository** — Opened by K4TEL as the repo-side
  continuation of hub [ufal/atrium-project#24](https://github.com/ufal/atrium-project/issues/24):
  spin the LLM-only subtasks out of the NameTag3/UDPipe-entangled `atrium-nlp-enrich` into a focused
  sibling repo. Engine copied byte-identical; the actual new work is `openrouter_client.py`,
  `ollama_client.py`, `api_util/xml_to_md.py`, and the torch-free `llm_client_shared.py`.
- **#8 (repo)** — **Document-Understanding scripts** landed (`f2ec956`): `eval_metrics.py` (CER/WER,
  normalized edit distance, entity F1, optional TEDS) and `sample_stratify.py` (quality-stratified
  80/10/10 page sampling) — the hub **#22** benchmark primitives. Licenses test `tests/test_para_licenses.py`
  (`83d7480`), GHA version bumps, dependabot merges (transformers ≥4.57.6, pydantic 2.13.4). Issue #8
  logs + digest/plan added and renamed from the `24.*` hub pair (`1b94264`, `c259453`). Suite at **83
  `def test_`** across 5 files; ruff clean.

## 2026-07-13 → 2026-07-15
- **#8 (repo)** — Test infrastructure hardened: test reqs + formatting (`3779460`); fixtures and tests
  imported from `nlp-enrich` with the `pandas` dependency (`2a09efe`, `43259a4`) — the suite grows well
  past the 83 baseline; `pytest` requirement bumped `>=8.0 → >=9.1.1` (PR #9, `fde9334`); version bump
  (`2540ad9`).

## 2026-07-16
- **#8 / hub #22 (repo)** — DU next steps and test-coverage/reqs updates (`9267382`, `dd90b13`); issue
  logs refreshed (`19e6a03`).

## 2026-07-17
- **#10 PDF and DOCX inputs handling for DU** — Opened by K4TEL: process input PDFs of three kinds
  (curves-instead-of-letters / scanned images / digital-born-with-text), add DOCX as an alternative
  input, and consider DOCX/HTML as intermediate formats on the PDF→LLM path.
- **#11 Decide on inputs of LLMs based on benchmarks existing for DU** — Opened by K4TEL (`question`,
  `development`): a format × detail-level matrix (PDF / InDesign XML / HTML+CSS / DOCX / Page-ALTO XML
  / MD / TXT); find the format used in DU benchmarks toward a FAIR standard; consider
  `ufal/atrium-page-classification` and **Grobid** for PDF processing.
- **#8 (repo)** — Large-model-on-CPU run notes updated (`90a8762`).

## 2026-07-19 → 2026-07-20
- **#10** — **Research pass** (survey + routing, no code): three PDF classes collapse to two paths via
  a cheap `pdffonts`/PyMuPDF font+char+image census (digital-born → extract; curves+scans → render+OCR),
  with a decode-sanity guard for the garbled-diacritics (no-`/ToUnicode`) case. Permissive-first tool
  survey; recommendation to reuse the repo's **Markdown** (doc-level) or **TEITOK** (line-level) targets,
  not a new format. Digest+plan added (`5ee844f`). **Blocker** noted: flexiconv's license is undeclared
  upstream (`para_config.txt:32-33`).
- **#11** — Deep-research report (Gemini) on agentic, roadmap-driven FAIR document navigation added as
  `digests/11.digest.md` (`59f0ddb`, `f88b216`); issue logs refreshed (`752ac64`).

## 2026-07-21
- **#10** — K4TEL added two comments steering the issue to implementation: use PDF-to-MD / DOCX-to-MD
  tools and record **page borders + as many visual layout cues as possible as HTML comments inside the
  Markdown**, and made "the list of all possible visual layout pieces" — an **exhaustive taxonomy** of
  cues with their exact MD/HTML encodings (`<!-- PAGE_BREAK -->`, `<!-- BBOX -->`, `<!-- FONT -->`,
  `<!-- HEADER_START -->`, `~~strike~~`, footnotes, tables, `<!-- WATERMARK -->`, …).

## 2026-07-22

* **#10** — **First implementation landed** (`3e7a909`): a visually-rich Markdown converter for **DOCX + digital-born PDF**.
* New `api_util/` modules introduced: `layout_md.py` (dependency-free cue vocabulary + `CUE_SCHEMA`), `docx_to_md.py`, 
`pdf_to_md.py`, and `doc_to_visual_md.py`.
* Pipeline creates a `.md` in `INPUT_DIR`, which is consumed unchanged by `run_document_level()` — HTML-comment cues 
pass through as inert text.
* The scanned/curve-only **OCR path stays deferred** (pages flagged `NEEDS_OCR`; tool choice benchmark-gated under hub #22).

## 2026-07-23

* **#10** — Implementation updated on the `test` branch: unified schema via `api_util/xml_to_md.py --format layout` 
now emits identical cues for TEITOK/ALTO based on coordinate sets.
* The opt-in OCR path is enabled via `--ocr`, resolving pages with `pypdfium2` and transcribing with Tesseract `ces`, 
tagged explicitly with `<!-- OCR: engine=tesseract, lang=ces -->`.
* Pipeline integration auto-converts `.pdf`/`.docx` files dropped into `INPUT_DIR` via openrouter/ollama clients, 
fetching citations natively via the `page` field.
* **#11** — Decision drafted: the ingestion diet is annotated Markdown, utilizing HTML comments as low-token positional 
hints, bypassing HTML and keeping TEITOK as the spatial truth.
* Slated a bake-off through the #22 harness with Docling/Marker for robust table processing and token cost analyses.
* **#13 The intermediate steps - data format to use** — Opened by K4TEL to solidify intermediate candidates: MD with 
HTML for forms/tables as LLM input, TEITOK.XML for correct layout, and JSON for search and metadata storage.
* Drafted a systemic approach allocating separate authority for each plane: Reading (Annotated Markdown),
Layout/preservation (TEITOK.XML), and Search/knowledge (`AtriumDocument` JSON).

## 2026-07-24

* **#13** — Opus 5 ultracode refinements proposed preventing the pipeline from being forced into full monolithic 
runs by relying entirely on a deterministic merger.
* Transient images and thumbnail paths were entirely dropped from references to restrict linkages solely to persistent 
items like original inputs or output artifacts.
* Proposed a dedicated stateless pure function service called `atrium-aggregate` to assemble the `AtriumDocument` via `POST /aggregate`.

## 2026-07-25

* **#10** — Concluded that MD equipped with visual info inside comments supplies LLM input, whereas JSON defines 
the overarching document schema per Issue #13, and TEITOK.XML acts as the visually accurate record alongside NLP enrichment capabilities.
* **#13** — Adopted the paradata-pair model where every tool receives a document JSON, modifies its owned blocks,
and emits an updated JSON byte-identical to untouched parameters.
* `atrium_document.py` and `atrium_document.schema.json` defined as hub-canonical shared files within the ecosystem.
* `alto-postprocess` refined internally to remove redundant langID values and coordinate closely with the `atrium_document` 
draft added initially to `atrium-project` and `llm-enrich` (commit `4175b06`).

## 2026-07-26

* **#13** — `AtriumDocument` ecosystem integration expanded: the JSON schema draft landed on `atrium-nlp-enrich`, 
`atrium-page-classification`, and `atrium-translator`.
* `para-drift` GHA check added to the `atrium-project` hub.
* Component beta releases shipped: `atrium-alto-postprocess` (v1.3.0-beta), `atrium-page-classification` 
(v1.7.0-beta), `atrium-nlp-enrich` (v0.18.0), and `atrium-translator` (v0.10.0).
* `atrium-llm-enrich` integrated the schema at the API and `llm_run` levels (Commit `c565e1a`), pending decisions 
approval for a formal release.

## 2026-07-31

* **#13** — Alignment pass completed by Sonnet uncovering and resolving critical pipeline bugs.
* `atrium-nlp-enrich` `run_document_hook()` rewritten with real ALTO+CoNLL-U integration testing after failing to 
produce entities natively in production due to key errors.
* `atrium-alto-postprocess` switched from `set_blocks` to `merge_blocks` for field-split outputs, preventing 
downstream overwrites.
* `atrium-translator` logic cleaned up by stripping dead entity translation code and enforcing correct schema boundaries.
* `atrium-llm-enrich` introduced `api_util/json_to_md.py` to regenerate the annotated-Markdown diet efficiently 
straight from the JSON record, rather than re-requesting a TEITOK or PDF file.

## 2026-08-01

* **#13** — E2E-related pipeline smoke achieved CLI convergence: `--document-json`/`--document-json-out` file-pair 
flags successfully added to `atrium-page-classification`, `atrium-translator`, `atrium-alto-postprocess`, 
`atrium-nlp-enrich`, and `atrium-llm-enrich`.
* `openrouter_client.py` within `atrium-llm-enrich` fully supports the converged flag structure for single-file scoping.
* A silent baseline dropping bug triggered by a mismatch in `doc_id` derivation between the translator file output and
`nlp-enrich` expectations was permanently fixed.

## 2026-08-02

* **#13** — Identified the necessity to build a PDF and DOCX to JSON converter explicitly to service digital-born 
documents appropriately, mapping required actions back to the criteria in Issue #10.
* **#8** — Closed: the repo-initialization work this issue tracked (engine copy, `openrouter_client.py`/
`ollama_client.py`, torch-free `llm_client_shared.py`) has been done and superseded by #10/#11/#13 since July.
* **#18 Build explicit PDF/DOCX-2-JSON converter — digital-born documents** — Opened by K4TEL: supersedes/extends
#10. The Markdown-with-HTML-comments intermediate format is now deprecated in favor of `atrium_document` JSON as
the first-class canonical IR; a new `api_util/digital_to_json.py` must ingest digital-born PDF/DOCX directly into
the schema, mapping the #10 layout-cue taxonomy onto JSON nodes instead of Markdown comments.

## 2026-08-03 – 2026-08-04

* **#18** — Architecture defined and largely built same week: block-ownership decided (`BLOCK_OWNERS` gains tuple
values; `source.origin` selects the originator per document) — three defects this surfaced are fixed. A 08-04
review pass found the §1a "one originator per block" contract itself **escapable in five ways**; all five fixed
and tested. Digital-born converter scripts land (`4c28020`) alongside a large GHA infrastructure copy (dependabot,
CodeQL, api-contract, docker workflows — the standard five-repo template); doc-related dependencies added
(`054288d`); `atrium_document.py` re-aligned to the hub template repeatedly (`070908e`, `0f231ca`, `abbac79`,
`a02eee6`, `30c1c99`, `9ef2ae6`, `bd8ea39`, `8381b13`) as the shared contract kept moving underneath. Suite reaches
382 passed / 6 skipped / 0 failed by the end of this window.

## 2026-08-06

* **v0.6.0 — first release of the `digital-convert` originator.** `api_util/digital_to_json.py` (807 lines) plus
`tests/test_digital_to_json.py` (404 lines) land (`796c795`), completing #18's core task: a four-layer born-digital
PDF/DOCX converter with a **decode-sanity gate** that catches a text layer extracting *successfully and wrongly*
(CP1250 bytes misread as CP1252), reporting `needs_ocr` rather than silently rewriting text that `source.sha256`
still describes as unchanged. **Real defect fixed**: DOCX tables previously lost their text entirely —
`extract_docx()` now walks paragraphs and tables in document order (reusing `docx_to_md.py`'s existing in-order
walker), so cell text reaches both `lines[]` and `cells[].group_id`; before this, every table's text existed only
in `tables[].cells[].text`, invisible to `json_to_md`, nlp-enrich and the translator, with the documented join
resolving to nothing. Degenerate grids are now omitted rather than raising a Layer D violation on the converter's
own output. Also: a further "LLM review+fix round by Opus" (`f5418f3`), `ruff.toml` hardening (`f58e24c`), one more
`atrium_document.py` fix pass (`ce95280`), the version bump (`6c71425`), GHA test-coverage req fixes (`d475a78`),
and a Docker GHA update (`57d8073`).

## 2026-08-18 – 2026-08-20

* **The AMCR + TEATER vocabulary work ported over from `atrium-nlp-enrich`'s issue #6** (no dedicated issue here —
tracked purely through commits): `45d2668` aligns the vocabulary with nlp-enrich's harvest, `05aec39` fixes the
union, `8331b2e` fixes it again for the E2E GHA run. **v0.6.1** ships (08-19): vocabulary expanded and aligned,
explicitly flagged "work-in-progress in terms of the vocabulary definition" — llm-enrich is following nlp-enrich's
governance rulings (see that repo's DEVLOG, Phases 4-8) rather than making its own. Further GHA hardening
(`d52e0d7`, `3699c42` — another Opus-reviewed round touching GHA/template/converters) and a `vllm`-related
requirements fix (`549afa6`).

## 2026-09-03

* **#24 Application of olmOCR** — Opened by K4TEL: flags the LLM-based OCR system described in a blog post
(structure + coordinates, no comments yet) as worth reading about — a candidate for the still-deferred
scanned/curve-only OCR path from #10.
* Routine action bump (`d1b6bdd`); issue logs refreshed.

## 2026-09-04

* Vocabulary work continues in lockstep with nlp-enrich's #6 Phase 9-10 (see that repo's DEVLOG): `806efad` adds
the `vocab_sources.py` util here, `a92b6e6` brings over the decision-package docs (`6.D-eval.decision-package.md`,
`6.O3O4.decision-package.md`) and the new `vocab-drift.yml`/`vocab-refresh.yml` workflows, `5468e29` rebuilds the
vocab files from the flat harvest. Version bumped (`e5a674c`).

## 2026-09-06

* **v0.6.2** ships. Vocabulary and prompts brought fully in step with nlp-enrich's shipped state (`609c247`,
`084dda9`); `bef7fae` fixes the Docker build for the GHA digital-born e2e run; `ee16913` fixes GHA-related code in
`llm_client_shared.py`/`ollama_client.py`/`openrouter_client.py` with a new `tests/test_llm_client_shared.py`
(78 lines). Issue logs refreshed same day.

## 2026-09-07

* **State**: 5 open issues (#10, #11, #13, #18, #24); #8 closed. `test` and `main` both at `ee16913`, **v0.6.2**.
Confirmed live: the hub reusable-workflow reference is pinned to `@v1`. The document-format architecture (#10/#11/
#13) is implemented and stable; #18's core converter shipped in v0.6.0 with the DOCX-table defect it was built to
avoid already caught and fixed. The active thread is the vocabulary/prompt alignment with `nlp-enrich`'s #6 — this
repo has no issue of its own for that work, so its state should be read alongside nlp-enrich's DEVLOG rather than
this repo's issue tracker. #24 (olmOCR) is a fresh, unstarted lead on the still-deferred OCR path from #10.

---
_Timeline index refreshed 2026-09-07 against live `test`/`main` HEAD, the `CONTRIBUTING.md` changelog table, and
open-issue state via the GitHub API. Nothing removed from the issues themselves (per hub #29); this file is a
derived reading aid in `agent_dev_logs/`._
