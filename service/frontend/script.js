/**
 * Minimal client logic for the ATRIUM LLM Enrichment demo frontend.
 *
 * Sends pasted text to POST /extract_keywords_text, or an uploaded file to
 * POST /extract_keywords, and renders the returned records.
 *
 * CONTRACT NOTE (2026-09-09). This file was rewritten against the current API. The
 * previous version had been left behind by the service redesign and was broken in
 * three separate ways at once:
 *
 *   * it POSTed `{lines: [...], top_k, backend}` to /extract_keywords_text, which now
 *     takes `{text, document_json}` — every submission on that path answered 422;
 *   * it rendered `data.lines`, which the response no longer carries, so even the
 *     upload path (where the extra form fields were merely ignored) fell through to
 *     "Unexpected response format";
 *   * it offered a backend selector and a top_k box. Both moved server-side: the
 *     backend is chosen once at server start from LLM_BACKEND, and neither field is
 *     read from a request any more. Sending them looked like control and was not.
 *
 * The response envelope this file is written against is:
 *
 *     {service, doc_id, backend, model, mode, results[], stats,
 *      document_json?, document_json_schema_error?}
 *
 * `mode` is "line" or "document"; the per-record projection below deliberately
 * mirrors `result_rows()` in scripts/atrium_keywords.py so the page and the client
 * agree on what a record means.
 */

/** Escape text before it reaches innerHTML.
 *
 * Everything rendered here is document content that came back from the server, i.e.
 * ultimately from a file the user supplied. The previous version interpolated it raw.
 */
function esc(value) {
    if (value === null || value === undefined) {
        return '';
    }
    return String(value).replace(/[&<>"']/g, (ch) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    })[ch]);
}

/** page/line locator for a record — line-level records carry `line`, document-level
 *  records carry `locator` instead (no fixed line grid). Same rule as the client. */
function locatorOf(item) {
    if (Object.prototype.hasOwnProperty.call(item, 'line')) {
        return `p${item.page ?? '?'}/l${item.line ?? '?'}`;
    }
    return `p${item.page ?? '?'}:${item.locator ?? ''}`;
}

function renderResults(data) {
    const container = document.getElementById('results');

    if (!data || !Array.isArray(data.results)) {
        container.innerHTML =
            '<div class="error">Unexpected response format from the API — expected a '
            + '<code>results</code> array. Check <a href="/docs">/docs</a> for the current schema.</div>';
        return;
    }

    const stats = data.stats || {};
    let html = `<h3>Results for ${esc(data.doc_id)}</h3>`;
    html += `<p class="meta">mode: <strong>${esc(data.mode)}</strong> · backend: `
          + `${esc(data.backend)} · model: ${esc(data.model || 'n/a')}</p>`;
    html += `<p class="meta">processed: ${esc(stats.processed ?? '?')} · `
          + `filtered out: ${esc(stats.skipped_filter ?? '?')} · `
          + `errors: ${esc(stats.skipped_error ?? '?')}</p>`;

    if (!data.results.length) {
        html += '<p>No records were produced. Every input line was filtered out before '
              + 'reaching the model, or the document held no qualifying text.</p>';
        container.innerHTML = html;
        return;
    }

    html += '<table><thead><tr><th>Locator</th><th>Category</th><th>Conf.</th>'
          + '<th>Keywords (cs)</th><th>Keywords (en)</th></tr></thead><tbody>';
    data.results.forEach((item) => {
        const enrichment = item.enrichment || {};
        const confidence = enrichment.confidence_score;
        html += `<tr>
            <td>${esc(locatorOf(item))}</td>
            <td>${esc(enrichment.teater_category || '')}</td>
            <td>${confidence != null ? esc(Number(confidence).toFixed(2)) : ''}</td>
            <td>${esc((enrichment.extracted_keywords_cs || []).join('; '))}</td>
            <td>${esc((enrichment.extracted_keywords_en || []).join('; '))}</td>
        </tr>`;
    });
    html += '</tbody></table>';

    // Surfaced rather than swallowed: the service accepts a baseline that fails its own
    // schema (Layer D, rule 6) and says so in this field instead of erroring.
    if (data.document_json_schema_error) {
        html += `<div class="error"><strong>document_json_schema_error:</strong> `
              + `${esc(data.document_json_schema_error)}</div>`;
    }
    if (data.document_json) {
        html += '<p class="meta">An accreted ATRIUM Document record came back with this '
              + 'response (<code>document_json</code>) — see the browser console.</p>';
        console.log('document_json:', data.document_json);
    }

    container.innerHTML = html;
}

/** Show which backend/model the running server actually chose (GET /info, §4.1).
 *  Read-only: the client cannot influence it, so the page reports rather than offers. */
async function showServerInfo(baseUrl) {
    const target = document.getElementById('serverInfo');
    try {
        const response = await fetch(`${baseUrl}/info`);
        if (!response.ok) {
            throw new Error(`/info returned ${response.status}`);
        }
        const info = await response.json();
        const ready = info.ready === false ? ' · <strong>still warming up</strong>' : '';
        target.innerHTML = `Server: <strong>${esc(info.service)}</strong> `
            + `v${esc(info.version)} · backend ${esc(info.backend || 'n/a')} · `
            + `model ${esc(info.model || 'n/a')} · max upload `
            + `${esc((info.limits || {}).max_upload_mb)} MB${ready}`;
    } catch (err) {
        target.textContent = `Could not read /info (${err.message}) — is the server running?`;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('extractForm');
    const loader = document.getElementById('loader');
    const resultDiv = document.getElementById('results');

    const baseUrl = window.location.origin.includes('localhost') ? 'http://localhost:8000' : '';
    showServerInfo(baseUrl);

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        resultDiv.innerHTML = '';
        loader.style.display = 'block';

        const text = document.getElementById('textInput').value.trim();
        const file = document.getElementById('fileInput').files[0];

        try {
            let response;
            if (file) {
                const formData = new FormData();
                formData.append('file', file);
                response = await fetch(`${baseUrl}/extract_keywords`, {
                    method: 'POST',
                    body: formData,
                });
            } else if (text) {
                // Two Body(...) parameters on the endpoint means FastAPI expects them
                // embedded by name. `document_json` is optional and omitted here — the
                // page has no baseline record to accrete into.
                response = await fetch(`${baseUrl}/extract_keywords_text`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text }),
                });
            } else {
                throw new Error('Paste some text or choose a file first.');
            }

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.detail ? JSON.stringify(data.detail) : `Server error: ${response.status}`);
            }
            renderResults(data);
        } catch (err) {
            console.error(err);
            resultDiv.innerHTML = `<div class="error"><strong>Error:</strong> ${esc(err.message)}</div>`;
        } finally {
            loader.style.display = 'none';
        }
    });
});
