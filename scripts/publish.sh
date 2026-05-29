#!/usr/bin/env bash
# Publish a clean single-commit snapshot of the current branch to origin/main.
#
# The repo keeps full development history on the local `main` branch (which is
# never pushed). The public GitHub repo shows only ONE commit. This script
# re-squashes the current HEAD tree into a fresh parent-less commit on the
# `public` branch and force-pushes it to origin/main, where the GitHub Pages
# workflow redeploys automatically.
#
# Usage:
#   scripts/publish.sh                  # publish with default message (asks to confirm)
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
#   - Force-pushes ONLY the squashed snapshot to origin/<branch>.
#   - Refuses to run with uncommitted tracked changes (the snapshot uses the
#     committed HEAD tree; uncommitted work would silently be excluded).
#   - Verifies the snapshot tree matches HEAD exactly before pushing.

set -euo pipefail

MSG="Standard Model from First Principles"
ASSUME_YES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes) ASSUME_YES=1; shift ;;
    -m|--message) MSG="${2:?-m requires a message}"; shift 2 ;;
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
echo "Publish a single-commit snapshot of HEAD ($HEAD_SHA) → ${REMOTE}/${TARGET_BRANCH}"
echo "  message:     \"$MSG\""
echo "  mechanism:   force-push a squashed commit (full history on local main stays private)"

if [[ "$ASSUME_YES" -ne 1 ]]; then
  read -r -p "Proceed? [y/N] " reply
  case "$reply" in
    [Yy]*) ;;
    *) echo "aborted."; exit 0 ;;
  esac
fi

# Create a parent-less commit capturing HEAD's exact tree; point `public` at it.
SNAP="$(git commit-tree 'HEAD^{tree}' -m "$MSG")"
git branch -f public "$SNAP"

# Sanity check: snapshot tree must match HEAD's tree exactly.
if ! git diff --quiet public HEAD; then
  echo "✗ snapshot tree does not match HEAD — aborting before push." >&2
  exit 1
fi

git push "$REMOTE" "public:${TARGET_BRANCH}" --force

echo "✓ published ${SNAP:0:9} → ${REMOTE}/${TARGET_BRANCH}"
echo "  GitHub Pages will redeploy automatically; watch with:"
echo "    gh run watch \$(gh run list --workflow=deploy-explainer.yml --limit 1 --json databaseId -q '.[0].databaseId')"
