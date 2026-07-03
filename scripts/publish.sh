#!/usr/bin/env bash
# Publish a clean snapshot of the current branch to origin/main.
#
# The repo keeps full development history on the local `main` branch (which is
# never pushed). The public GitHub repo shows only squashed snapshots. This
# script snapshots HEAD's tree — MINUS the private Tier-2 research lab-notebook
# (see PRIVATE_PATHS below / the TIER-2 note in .gitignore) — onto the `public`
# branch and pushes it to origin/main.
#
# Two modes:
#   default : a fresh PARENT-LESS commit, force-pushed (the historical mode —
#             the public repo shows exactly one commit).
#   --ff    : a snapshot whose PARENT is the current origin/<branch> tip,
#             pushed WITHOUT force (a fast-forward). The public repo then
#             accumulates one commit per publish — a visible public timeline —
#             while the full private history still never leaves local `main`.
#             (Adopted 2026-07-03; also lets the Pages workflow's `paths:`
#             filter evaluate naturally against the parent.)
#
# Usage:
#   scripts/publish.sh                  # parent-less snapshot (asks to confirm)
#   scripts/publish.sh --ff             # fast-forward snapshot on top of origin/main
#   scripts/publish.sh -y               # skip the confirmation prompt
#   scripts/publish.sh -m "My message"  # custom commit message
#   scripts/publish.sh -h               # this help
#
# Env overrides:
#   PUBLISH_REMOTE  (default: origin)
#   PUBLISH_BRANCH  (default: main)
#
# Safety:
#   - Never pushes your local `main` (which carries the full private history).
#   - Pushes ONLY the filtered snapshot to origin/<branch> (--force only in
#     the parent-less mode; --ff mode is a plain fast-forward push).
#   - Refuses to run with uncommitted tracked changes (the snapshot uses the
#     committed HEAD tree; uncommitted work would silently be excluded).
#   - Verifies the snapshot tree matches HEAD exactly (minus Tier-2) before
#     pushing; in --ff mode also prints what the publish deletes relative to
#     the current public tip, for eyes-on review.

set -euo pipefail

MSG="Standard Model from First Principles"
ASSUME_YES=0
FF_MODE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes) ASSUME_YES=1; shift ;;
    -m|--message) MSG="${2:?-m requires a message}"; shift 2 ;;
    --ff) FF_MODE=1; shift ;;
    -h|--help) sed -n 's/^# \{0,1\}//p' "$0" | sed '/^!/d'; exit 0 ;;
    *) echo "unknown argument: $1 (use -h for help)" >&2; exit 2 ;;
  esac
done

cd "$(git rev-parse --show-toplevel)"

REMOTE="${PUBLISH_REMOTE:-origin}"
TARGET_BRANCH="${PUBLISH_BRANCH:-main}"

# Block if there are uncommitted tracked changes (staged or unstaged).
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "⚠  Uncommitted tracked changes detected." >&2
  echo "   The published snapshot uses the committed HEAD tree, so those" >&2
  echo "   changes would NOT be included. Commit or stash them first." >&2
  exit 1
fi

# Warn (don't block) if untracked files exist — they're not in HEAD, so they
# won't be published either, but the user may not expect that.
if [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
  echo "note: untracked files exist; they are not committed and will NOT be published."
fi

HEAD_SHA="$(git rev-parse --short HEAD)"
PARENT=""
if [[ "$FF_MODE" -eq 1 ]]; then
  git fetch --quiet "$REMOTE" "$TARGET_BRANCH"
  PARENT="$(git rev-parse "refs/remotes/${REMOTE}/${TARGET_BRANCH}")"
  echo "Publish a FAST-FORWARD snapshot of HEAD ($HEAD_SHA) → ${REMOTE}/${TARGET_BRANCH}"
  echo "  parent:      ${PARENT:0:9} (current ${REMOTE}/${TARGET_BRANCH} tip)"
  echo "  message:     \"$MSG\""
  echo "  mechanism:   plain (non-force) push of a filtered snapshot child commit"
else
  echo "Publish a single-commit snapshot of HEAD ($HEAD_SHA) → ${REMOTE}/${TARGET_BRANCH}"
  echo "  message:     \"$MSG\""
  echo "  mechanism:   force-push a squashed commit (full history on local main stays private)"
fi

if [[ "$ASSUME_YES" -ne 1 ]]; then
  read -r -p "Proceed? [y/N] " reply
  case "$reply" in
    [Yy]*) ;;
    *) echo "aborted."; exit 0 ;;
  esac
fi

# ── TIER-2 FILTER ────────────────────────────────────────────────────────────
# The public snapshot = HEAD's tree MINUS the Tier-2 research lab-notebook, which
# is tracked privately on `main` (backed up + visible) but NEVER published.
# See the TIER-2 note in .gitignore. Keep this list in sync with it.
PRIVATE_PATHS=(
  docs/_scratch proofs/_scratch _scratch
  docs/scoping docs/open_problems papers
)
TMP_INDEX="$(mktemp)"
GIT_INDEX_FILE="$TMP_INDEX" git read-tree HEAD
GIT_INDEX_FILE="$TMP_INDEX" git rm -r --cached --quiet --ignore-unmatch -- "${PRIVATE_PATHS[@]}"
PUBLIC_TREE="$(GIT_INDEX_FILE="$TMP_INDEX" git write-tree)"
rm -f "$TMP_INDEX"

# Create the snapshot commit capturing the FILTERED (public) tree; point `public` at it.
# In --ff mode the snapshot's parent is the current public tip (fast-forward chain);
# otherwise it is parent-less (the historical single-commit mode).
if [[ "$FF_MODE" -eq 1 ]]; then
  SNAP="$(git commit-tree "$PUBLIC_TREE" -p "$PARENT" -m "$MSG")"
else
  SNAP="$(git commit-tree "$PUBLIC_TREE" -m "$MSG")"
fi
git branch -f public "$SNAP"

# Sanity 1: the public branch's tree must equal the filtered snapshot exactly.
if [[ "$(git rev-parse 'public^{tree}')" != "$PUBLIC_TREE" ]]; then
  echo "✗ public tree does not match the filtered snapshot — aborting before push." >&2
  exit 1
fi
# Sanity 2: the ONLY files that may differ from HEAD are Tier-2 paths. If anything
# else was dropped, abort (guards against silently losing public content).
PRIV_RE="^($(IFS='|'; echo "${PRIVATE_PATHS[*]}"))/"
LEAKED="$(git diff --name-only public HEAD | grep -vE "$PRIV_RE" || true)"
if [[ -n "$LEAKED" ]]; then
  echo "✗ public snapshot would drop non-Tier-2 files — aborting:" >&2
  printf '   %s\n' "$LEAKED" >&2
  exit 1
fi

if [[ "$FF_MODE" -eq 1 ]]; then
  # Sanity 3 (ff only): confirm the push is a genuine fast-forward, and show
  # what this publish DELETES relative to the current public tip (legitimate
  # dev deletions are expected; a surprise here deserves eyes before pushing).
  if ! git merge-base --is-ancestor "$PARENT" "$SNAP"; then
    echo "✗ snapshot is not a descendant of ${REMOTE}/${TARGET_BRANCH} — aborting." >&2
    exit 1
  fi
  DELETED="$(git diff --name-only --diff-filter=D "$PARENT" "$SNAP" || true)"
  if [[ -n "$DELETED" ]]; then
    echo "note: this publish deletes the following files from the public repo:"
    printf '   %s\n' "$DELETED"
  fi
  echo "  diffstat vs public tip:"
  git diff --shortstat "$PARENT" "$SNAP" | sed 's/^/   /'
  git push "$REMOTE" "public:${TARGET_BRANCH}"
else
  git push "$REMOTE" "public:${TARGET_BRANCH}" --force
fi

echo "✓ published ${SNAP:0:9} → ${REMOTE}/${TARGET_BRANCH}"

# The Pages workflow's `paths:` filter cannot evaluate against a parent-less
# force-pushed commit, so that mode does NOT trigger it (found 2026-07-02 — the
# site had silently served stale content since the previous manual deploy).
# In --ff mode the filter evaluates against the parent and normally fires on
# its own; dispatching explicitly anyway is a harmless belt-and-braces.
# Fall back to a reminder if `gh` is unavailable.
if command -v gh >/dev/null 2>&1 && gh workflow run deploy-explainer.yml >/dev/null 2>&1; then
  echo "  GitHub Pages deploy dispatched; watch with:"
else
  echo "  ⚠ Could not dispatch the Pages deploy (gh missing/unauthenticated)."
  echo "    Trigger it manually: gh workflow run deploy-explainer.yml   (or the Actions tab)"
  echo "  Then watch with:"
fi
echo "    gh run watch \$(gh run list --workflow=deploy-explainer.yml --limit 1 --json databaseId -q '.[0].databaseId')"
