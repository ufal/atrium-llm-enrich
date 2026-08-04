#!/usr/bin/env bash
#
# scripts/revendor_shared.sh — copy the hub-canonical shared files into a tool repo
# and verify byte-identity, i.e. do locally exactly what para-drift.reusable.yml does
# in CI, before CI has to tell you.
#
# Why this exists (Issue #18 §0b): the `group_id` / `tables` / `forms` edits landed in
# docs/templates/shared/ on the hub and NOT in the five vendored copies. Since
# `para-drift.reusable.yml` diffs `atrium_document.py` and `atrium_document.schema.json`
# against every tool repo, that divergence is a CI failure waiting on a `v1` tag move.
# Editing a vendored copy is always the wrong fix — edit the hub copy and re-vendor.
#
# Usage:
#   scripts/revendor_shared.sh --hub ../atrium-project --check          # report only
#   scripts/revendor_shared.sh --hub ../atrium-project                  # copy + verify
#   scripts/revendor_shared.sh --hub ../atrium-project --all ../repos    # every sibling repo
#
# Exit 0 = every file identical. Exit 1 = drift (in --check) or copy failure.

set -euo pipefail

HUB=""
MODE="apply"
ALL_ROOT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hub)   HUB="$2"; shift 2 ;;
        --check) MODE="check"; shift ;;
        --all)   ALL_ROOT="$2"; shift 2 ;;
        -h|--help) sed -n '3,18p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$HUB" ]]; then
    echo "error: --hub <path-to-atrium-project-checkout> is required" >&2
    exit 2
fi
if [[ ! -d "$HUB/docs/templates/shared" ]]; then
    echo "error: $HUB does not look like an atrium-project checkout" >&2
    exit 2
fi

# Exactly the list para-drift.reusable.yml checks, in the same order. Keep in step:
# a file added to that workflow and not to this list is a file that drifts silently.
#   <destination-in-tool-repo>:<source-in-hub>
FILES=(
    "atrium_paradata.py:atrium_paradata.py"
    "para_licenses.py:para_licenses.py"
    "tests/test_para_licenses.py:test_para_licenses.py"
    "service/atrium_service.py:atrium_service.py"
    "atrium_document.py:atrium_document.py"
    "atrium_document.schema.json:atrium_document.schema.json"
    "check_version.py:check_version.py"
)

revendor_one_repo() {
    local repo="$1"
    local drifted=0 copied=0 skipped=0

    echo "── $repo"
    for pair in "${FILES[@]}"; do
        local dest="${pair%%:*}"
        local src="$HUB/docs/templates/shared/${pair##*:}"

        if [[ ! -f "$src" ]]; then
            echo "   ?? missing in hub: ${pair##*:}"
            drifted=1
            continue
        fi
        # A repo legitimately may not have every file (e.g. no service/ layer).
        if [[ ! -f "$repo/$dest" ]]; then
            echo "   -- absent in repo, not creating: $dest"
            skipped=$((skipped + 1))
            continue
        fi

        if cmp -s "$src" "$repo/$dest"; then
            echo "   ok $dest"
        elif [[ "$MODE" == "check" ]]; then
            echo "   DRIFT $dest"
            diff -u "$repo/$dest" "$src" | head -40 || true
            drifted=1
        else
            cp "$src" "$repo/$dest"
            cmp -s "$src" "$repo/$dest" || { echo "   FAILED to copy $dest" >&2; drifted=1; }
            echo "   -> re-vendored $dest"
            copied=$((copied + 1))
        fi
    done

    if [[ "$MODE" == "check" ]]; then
        [[ $drifted -eq 0 ]] && echo "   clean" || echo "   ^ drift above"
    else
        echo "   $copied re-vendored, $skipped absent"
    fi
    return $drifted
}

status=0
if [[ -n "$ALL_ROOT" ]]; then
    # nullglob, or a non-matching glob leaves the literal string "…/atrium-*" as the loop
    # variable, `[[ -d $repo/.git ]]` fails, every iteration is skipped and the script exits
    # 0 having checked NOTHING — reported as success by the one tool whose job is to prove
    # the vendored copies are in step.
    shopt -s nullglob
    repos=("$ALL_ROOT"/atrium-*)
    shopt -u nullglob
    checked=0
    for repo in "${repos[@]}"; do
        [[ -d "$repo/.git" ]] || continue
        [[ "$(basename "$repo")" == "atrium-project" ]] && continue
        checked=$((checked + 1))
        revendor_one_repo "$repo" || status=1
    done
    if [[ $checked -eq 0 ]]; then
        echo "error: --all $ALL_ROOT matched no atrium-* git checkouts — nothing verified" >&2
        exit 2
    fi
else
    revendor_one_repo "." || status=1
fi

cat <<'EOF'

Reminder — para-drift reads the HUB side at its `hub-ref` input, default `v1`, NOT at the
branch you just merged. Re-vendoring alone does not make CI agree with you. Per
docs/docker_gha.md: land the hub change on main, validate via a caller temporarily pinned
`@test` (passing `hub-ref: test`), then move the tag:

    git tag -f v1 <sha> && git push -f origin v1

Until v1 moves, tool repos carrying the NEW files will fail against the OLD templates —
so re-vendor and move the tag in the same window, not days apart.
EOF

exit $status
