<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" title="Python Version"></a>
  <a href="https://openrouter.ai/"><img src="https://img.shields.io/badge/remote-OpenRouter-6E56CF.svg" title="OpenRouter"></a>
  <a href="https://ollama.com/"><img src="https://img.shields.io/badge/local--light-Ollama-000000.svg" title="Ollama"></a>
  <a href="https://opensource.org/license/mit/"><img src="https://img.shields.io/github/license/ufal/atrium-llm-enrich" title="MIT License"></a>
  <a href="https://atrium-research.eu/"><img src="https://img.shields.io/badge/funded%20by-ATRIUM-8A2BE2.svg" title="ATRIUM Project"></a>
</p>

---

# ATRIUM LLM Enrichment - Agent Skill 🤖🗝️

### Goal: let coding agents extract vocabulary-guided keywords via a server-client skill

This branch (`agent-skill`) packages the **ATRIUM LLM Enrichment API service**
together with a **Skill for coding agents** (Claude Code, Codex,
Gemini/Antigravity). The design follows a strict server-client split:

- **Server** 🖥️ - the FastAPI service in [`service/`](service/) drives the LLM
  backends (OpenRouter remote / local Ollama) with TEATER/AMCR
  vocabulary-constrained prompting (Docker Compose `api` profile or torch-free
  local venv).
- **Client** 🪶 - [`scripts/atrium_keywords.py`](scripts/atrium_keywords.py), a
  **zero-dependency** stdlib-only script that agents call directly.
- **Skill contract** 📜 - [`SKILL.md`](SKILL.md) tells the agent when and how to
  use it: backend selection, confidence discipline, budget limits, error
  playbooks.

For the batch CLI (including the local multi-GPU `transformers`/`vLLM` path),
benchmarking, and full project documentation, see the
[`test`](https://github.com/ufal/atrium-llm-enrich/tree/test) branch - this
branch intentionally carries only what the skill needs.

### Table of contents 📑

  * [Quick start 🚀](#quick-start-)
  * [Skill installation 🔧](#skill-installation-)
  * [Server setup 🖥️](#server-setup-)
  * [Client usage 🪶](#client-usage-)
  * [Remote server / LINDAT 🌐](#remote-server--lindat-)
  * [Maintenance notes 🔍](#maintenance-notes-)
  * [Contacts 📧](#contacts-)

----

## Quick start 🚀

```bash
git clone -b agent-skill https://github.com/ufal/atrium-llm-enrich.git
cd atrium-llm-enrich

export OPENROUTER_API_KEY=sk-or-...                              # or run Ollama locally
bash scripts/server.sh                                           # start the server
python3 scripts/atrium_keywords.py small_data_samples/lines_sample.txt
```

> [!NOTE]
> The first server start auto-syncs the TEATER/AMCR vocabulary from the AMCR
> API (minutes, network-bound), and every extraction calls an LLM - this is
> the slowest ATRIUM service. ⏳

## Skill installation 🔧

### Claude Code

```bash
git clone -b agent-skill https://github.com/ufal/atrium-llm-enrich.git \
    ~/.claude/skills/atrium-llm-enrich
```

Restart Claude Code - the skill is available as `/atrium-llm-enrich` and is
selected automatically for semantic-enrichment requests. For a project-local
install, clone into `.claude/skills/atrium-llm-enrich` inside the target
repository.

### Codex

```bash
git clone -b agent-skill https://github.com/ufal/atrium-llm-enrich.git \
    ~/.codex/skills/atrium-llm-enrich
```

The skill is detected automatically in the next Codex session.

### Google Antigravity

Clone the branch into your project and point `AGENTS.md` at it:

```
Use the ATRIUM LLM enrichment skill from `atrium-llm-enrich/SKILL.md` for
vocabulary-guided keyword extraction from archival text lines.
Start the server with `bash atrium-llm-enrich/scripts/server.sh`, then run
`python3 atrium-llm-enrich/scripts/atrium_keywords.py [FILES...]`.
```

Update any install with `git pull` inside the cloned skill directory.

## Server setup 🖥️

The server exposes five endpoints (see [`service/README.md`](service/README.md)
for details): `GET /info`, `GET /health`, `GET /ready`, `POST /extract_keywords`,
`POST /extract_keywords_text`. A minimal demo frontend — which documents the same
calls as a `curl` recipe — is mounted at `/frontend`.

```bash
bash scripts/server.sh          # auto: Docker Compose api profile, else local uvicorn
bash scripts/server.sh --local  # force local uvicorn (torch-free venv)
```

The script is idempotent and health-waits on `/info`. Port defaults to `8000`
(`ATRIUM_LE_PORT` to change). Backend configuration: `OPENROUTER_API_KEY` +
`OPENROUTER_MODEL`, or `OLLAMA_HOST` + `OLLAMA_MODEL` (defaults in
`llm_config.txt`); `BACKEND` selects the server default.

## Client usage 🪶

```bash
python3 scripts/atrium_keywords.py lines.txt                        # plain lines
python3 scripts/atrium_keywords.py lines.csv --backend ollama       # explicit backend
python3 scripts/atrium_keywords.py page.teitok.xml --format json    # TEITOK, full envelope
python3 scripts/atrium_keywords.py - --doc-id CTX1 < lines.txt      # stdin lines
python3 scripts/atrium_keywords.py --info                           # capabilities
```

Output rows: `DOC, PAGE, LINE, CATEGORY, CONF, KEYWORDS_CS[, KEYWORDS_EN]`
(`--format table|csv|json`). Semantics are documented in
[`SKILL.md`](SKILL.md#backends--vocabulary-).

## Remote server / LINDAT 🌐

The client is location-agnostic: point it at any deployment with `--base-url` or

```bash
export ATRIUM_LE_URL="https://<hosted-instance>/atrium-le"
```

A hosted instance is planned; once available, the environment variable is the
only change needed - the skill contract and client stay identical.

## Maintenance notes 🔍

Review checklist for every change / sync-merge into this branch (the ATRIUM skill
anti-pattern checklist):

- [ ] no doc references a script name that differs from the committed file;
- [ ] no provenance/paradata claim unless the service imports it on this branch;
- [ ] no reference to directories/files absent from this branch;
- [ ] documented response fields match what `service/api.py` actually returns;
- [ ] client smoke test re-run on `small_data_samples/` against a locally started server.

## Contacts 📧

**For support write to:** lutsai.k@gmail.com responsible for the
[GitHub repository](https://github.com/ufal/atrium-llm-enrich)

### Acknowledgements 🙏

- **Developed by** UFAL, Charles University 👥
- **Funded by** [ATRIUM](https://atrium-research.eu/) 💰
- **Vocabulary** harvested from the [AMCR](https://amcr-info.aiscr.cz/) API (TEATER taxonomy) 🔗
