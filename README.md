# `gh-pages` &mdash; published stub for `atrium-llm-enrich`

**This branch is not source.** It is an orphan branch holding one static landing
card, published by GitHub Pages at <https://ufal.github.io/atrium-llm-enrich/>.

* The **code** lives on [`main`](https://github.com/ufal/atrium-llm-enrich/tree/main).
* The **documentation** lives in the ATRIUM hub site at
  <https://ufal.github.io/atrium-project/tools/llm-enrich/>, assembled from this repository's own
  `README.md`, `CONTRIBUTING.md` and `agent_dev_logs/DEVLOG.md`. Nothing is copied
  here &mdash; see [`atrium-project` issue #57](https://github.com/ufal/atrium-project/issues/57).

## What is in here

| Path               | Purpose                                                         |
|--------------------|-----------------------------------------------------------------|
| `index.html`       | the landing card                                                |
| `404.html`         | the same card, framed as a not-found page                       |
| `assets/style.css` | self-contained styles; no webfont, no CDN, no script            |
| `.nojekyll`        | tell Pages to serve these files as-is instead of running Jekyll |
| `README.md`        | this file                                                       |

## Maintaining it

A static card does not need regenerating &mdash; that is the point of putting it on a
branch rather than building it in CI. Edit it only when the repository's name,
one-line description or place in the pipeline changes.

The card deliberately renders the README's badges as **local chips** rather than
`img.shields.io` images, so the page has zero external requests and cannot be left
half-drawn by a slow third-party image host.

## Enabling Pages for this repository

One-time, needs repository admin: **Settings &rarr; Pages &rarr; Build and deployment
&rarr; Source: Deploy from a branch &rarr; Branch: `gh-pages` / `/ (root)`**.
See `PAGES_SETUP.md` in the hub repository for the full note.
