# 🤝 Contributing to the ATRIUM LLM Enricher

Thank you for your interest in contributing! This repository is the LLM-only sibling of
[`atrium-nlp-enrich`](https://github.com/ufal/atrium-nlp-enrich) (spun out per
[ATRIUM issue #24](https://github.com/ufal/atrium-project/issues/24)) — it maps archival text
onto the TEATER/AMCR archaeological vocabulary using an LLM, run either locally
(`transformers`/`vLLM`, multi-GPU) or as a service (`OpenRouter`, lightweight local `Ollama`).

This document describes the development workflow, code conventions, and rules for
contributors. ATRIUM-wide conventions (branching, commit types, the test/lint standard) are
identical across all repositories; anything repo-specific is called out explicitly.

## 📦 Release History

| Version    | Highlights                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Status      |
|:-----------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------|
| **v0.6.1** | Vocabulry expanded and aligned with the `nlp-enrich` repo. Supplementary materials and scripts included. Work-in-progress in terms of the vocabulary definition.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Pre-release |
| **v0.6.0** | **First release of the `digital-convert` originator** — `api_util/digital_to_json.py`, the four-layer born-digital PDF/DOCX converter whose decode-sanity gate catches a text layer that extracts *successfully and wrongly* (CP1250 bytes read as CP1252), reporting `needs_ocr` rather than rewriting text that `source.sha256` still describes. **DOCX tables no longer lose their text:** `extract_docx()` walks paragraphs and tables in document order using the in-order walker `docx_to_md.py` already had, so cell text reaches `lines[]` and `cells[].group_id` joins back to it — previously every table's text existed only in `tables[].cells[].text`, invisible to `json_to_md`, nlp-enrich and the translator, with the documented join resolving to nothing; degenerate grids are now omitted rather than raising Layer D on the converter's own output. Also the AMCR/TEATER vocabulary subsystem aligned with nlp-enrich, a nightly that runs the full suite instead of collecting nothing, and the re-vendored `atrium_document.py`. | Pre-release |
| **v0.5.2** | Re-vendored `atrium_document.py`: `DocumentRecord` now **inherits `doc_id` from the baseline** instead of overwriting it with the caller's derivation. Fixing the `Path.stem` derivation stopped *this* tool forking a record, but `DocumentRecord.__init__` still stamped whatever id its caller passed over the key the baseline arrived with, so no tool could opt out of the class by being careful at one call site. Also in this window: `canonical_doc_id()` replaced `Path.stem` in `openrouter_client.py` / `ollama_client.py` and retired `teitok_read.doc_id_from_path()`, `/enrich` reached parity on the `document_json` part, digital-born probe scripts were added under `digital_born/`, and `tests/test_document_originators.py` -> canonical shared set.                                                                                                                                                                                                                                                                              | Pre-release |
| **v0.5.1** | End-to-end GHA pipeline for JSON input-output by `atrium_document` standard is refined, and tested for the draft JSON schema design. Template updated.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Pre-release |
| **v0.5.0** | Major GHA workflows update with references to `@v1` on hub repo. Updated template scripts. Added atrium_document dtandard for input-output of JSONs.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Pre-release |
| **v0.4.0** | OpenAPI standards applied - draft. `atrium_document` draft added for cross-repo JSON expansion. Added PDF and DOCS to MD convertors draft. Edited GHA release workflow.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Pre-release |
| **v0.3.0** | Updated dependencies. Added DU code. Added <PDF/DOCX>-2-MD transformers draft. Updated tests coverage. Refreshed issue logs (digests+plans)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Pre-release |
| **v0.2.0** | Added new files according to plan. Added new GHA files. Added Document Understanding draft scripts. Added new tests. Fixed according to Fable review.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Pre-release |
| **v0.1.0** | Initial repo: LLM engine copied from `atrium-nlp-enrich`, NameTag/UDPipe dropped, `openrouter_client.py` + `ollama_client.py` + `api_util/xml_to_md.py` added.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Pre-release |

**Versioning rules (enforced by CI):** the `[tool] version` in `para_config.txt`, `version:` in
`CITATION.cff`, and the git tag MUST agree (prefix-tolerant: `v0.1.0` in `para_config.txt` ==
`0.1.0` in `CITATION.cff`) — `security.yml` fails the build otherwise. Update `date-released:`
in `CITATION.cff` to the actual release date on every version bump.

---

## 🏗️ Project Contributions & Capabilities

See [README.md](README.md) for full usage; in brief, four backends share one output contract
(`llm_utils.py`/`llm_client_shared.py`):

1. **`transformers`** (`llm_run.py`) — single-GPU, BnB 4-bit/AWQ/GGUF, models ≤ 31 B.
2. **`vllm`** (`llm_run.py`) — multi-GPU, native guided JSON decoding, models ≥ 70 B.
3. **`openrouter`** (`openrouter_client.py`) — remote LLM-as-a-service, provider-routed
   data-sovereignty controls, optional file attachment for document-level input.
4. **`ollama`** (`ollama_client.py`) — lightweight local server, native `format` JSON schema.

`api_util/xml_to_md.py` renders whole TEITOK/ALTO documents to Markdown/plain-text for the
document-level input path used by the remote/lightweight-local backends.

---

## 🌿 Branches & Environments

| Branch | Environment          | Rule                                                                          |
|--------|----------------------|-------------------------------------------------------------------------------|
| `test` | Staging              | Base for all development. Always branch from `test`.                          |
| `main` | Stable / Integration | Merged exclusively by a human reviewer. Do not open PRs directly into `main`. |

```text
test  ←  feature-<name>
test  ←  bugfix-<name>
main  ←  (humans only, after test stabilises)
```

### 🏷️ Branch Naming

| Type           | Pattern          | Example                    |
|----------------|------------------|----------------------------|
| New feature    | `feature-<name>` | `feature-ollama-streaming` |
| Bug fix        | `bugfix-<name>`  | `bugfix-vocab-truncation`  |
| Hotfix on main | `hotfix-<name>`  | `hotfix-openrouter-retry`  |

---

## 🔁 Contributor Workflow

1. **Create an issue** (or find an existing one) describing the problem or feature.
2. **Branch from `test`:**
```bash
   git checkout test && git pull origin test
   git checkout -b feature-<name>
```
3. **Implement** following the code conventions below.
4. **Run the fast checks** (see Testing) before every commit.
5. **Open a Pull Request** targeting `test`. Use a **Draft PR** while work is in progress.

---

## 📋 Pull Request Format

Every PR must include:

* **Issue link:** `Closes #<number>` or `Refs #<number>`
* **Motivation:** why the change is needed
* **Description of change:** what changed and how
* **Testing:** what was run, what passed, what could not be executed (and why)

**Do not open PRs into `main`** — merging into `main` is the maintainers' responsibility.

---

## ✏️ Commit Messages

Format: `[type] concise description of what changed`

| Type       | When to use                           |
|------------|---------------------------------------|
| `add`      | Added content (general)               |
| `edit`     | Edited existing content (general)     |
| `remove`   | Removed existing content (general)    |
| `fix`      | Bug fix                               |
| `refactor` | Refactoring without behaviour change  |
| `test`     | Adding or updating tests              |
| `docs`     | Documentation only                    |
| `chore`    | Build, dependencies, CI configuration |
| `style`    | Formatting, no logic change           |
| `perf`     | Performance optimisation              |

---

## 🧪 Code Conventions & Testing

### Code conventions
* **Comments:** short and informative; add one when the function name doesn't fully explain intent.
* **Argument types:** give every function argument a default type (`int`, `list`, …).
* **Console flags:** every new CLI flag ships with a `help=` message.
* **Config files:** when the set of `llm_config.txt` variables changes, reflect it in `README.md`.
* **`llm_client_shared.py` parity:** if you change the quality filter, context-window builder, or
  archaeological system prompt in `llm_utils.py`/`llm_run.py`, mirror the change in
  `llm_client_shared.py` by hand (see that module's docstring) — and update
  `tests/test_llm_client_shared.py` to cover it.
* **Engine files stay untouched:** `llm_utils.py`, `vocab_manager.py`, `atrium_paradata.py`,
  `para_licenses.py`, and the `api_util/{teitok_read,teitok_alto,flexiconv_convert,bbox_scale}.py`
  files are copied verbatim from `atrium-nlp-enrich` — do not fork their logic locally; a repo
  drift-check (`para-drift.yml`) enforces this for the paradata pair.

### Minimum checks before every commit
```bash
python -m compileall -q .                 # 1. compiles
pre-commit run --all-files                # 2. ruff (shared ruff.toml)
pytest -m "not slow" --tb=short           # 3. fast lane — no models, no GPU, no network
```

### Running the test suite
The fast lane requires **no ML models, GPU, or network**:
```bash
pip install -r requirements.txt -r requirements_remote.txt pytest
pytest -m "not slow" --tb=short                          # before every commit
pytest -m "not slow" --cov=. --cov-report=term-missing   # with coverage
```
Tests that load model weights, hit the network (OpenRouter, Ollama, HuggingFace), or need a GPU
must be marked `@pytest.mark.slow` and are excluded by default — see
[`.github/workflows/scheduled-smoke.yml`](.github/workflows/scheduled-smoke.yml) and
[`.github/workflows/gpu-inference.yml`](.github/workflows/gpu-inference.yml) for where they run.

### Linting
Ruff is the ATRIUM standard. Run `ruff check --config ruff.toml .` before opening a PR — this
repo's `ruff.toml` (line-length 100, `E`/`F`/`W`/`I`/`B`) matches `atrium-nlp-enrich`'s, not the
hub's 120-column default, since the LLM engine files are shared verbatim between the two repos.

---

## 🔗 Shared ("drop-in") code
`atrium_paradata.py` and `para_licenses.py` are **canonical** in
`ufal/atrium-project/docs/templates/shared/` and copied verbatim into each tool repository.

* **Do not fork their logic locally:** edit the canonical copy in the hub, then re-sync to the tools.
* **CI drift-check:** [`para-drift.yml`](.github/workflows/para-drift.yml) fails the build if this
  repo's copy diverges from the canonical source.
* **Configuration:** `para_config.txt` is the only per-repo dependency. Its `[components]` section
  currently has an open TODO — see the comment block at the top of that file — for resolving
  per-`MODEL_KEY`/provider licensing rather than a single fixed license row.

---

## 📁 Repository Documentation Management

| File              | Audience        | Responsibility                               |
|-------------------|-----------------|----------------------------------------------|
| `README.md`       | GitHub visitors | Project overview, backend usage, quick start |
| `CONTRIBUTING.md` | Developers      | Code conventions, branches, PRs, testing     |

Do not duplicate rules across files — cross-reference the canonical source.

---

## 📞 Contacts & Acknowledgements
Maintainer: **lutsai.k@gmail.com** [^1] · Developed by UFAL [^2] · Funded by ATRIUM [^3]

**©️ 2026 UFAL & ATRIUM**

[^1]: https://github.com/ufal/atrium-llm-enrich
[^2]: https://ufal.mff.cuni.cz/
[^3]: https://atrium-research.eu/