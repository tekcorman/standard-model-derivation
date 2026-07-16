#!/usr/bin/env bash
# public_scrub.sh — content-scrub for the PUBLIC snapshot.
#
# WHY THIS EXISTS. scripts/publish.sh filters the public snapshot by PATH (the
# Tier-2 lab-notebook dirs). It does NOT filter by CONTENT. Public-visible files
# (derivation_topdown/, proofs/foundations/, docs/theorems/, docs/framework/,
# docs/parameters/, README.md, …) drift and accumulate internal-only vocabulary
# the path filter cannot catch: model names, multi-agent process vocabulary,
# absolute /home paths, cross-repo refs, and dead links into the filtered dirs.
#
# This library scrubs a MATERIALIZED COPY of the already-path-filtered snapshot,
# in place, then a guard re-greps it and FAILS if any denylist token survives —
# so the public bar is enforced BY CONSTRUCTION, not by memory.
#
# ★ IT NEVER TOUCHES `main`. It runs only on a throwaway checkout of the filtered
#   tree inside publish.sh (or on a scratch dir when run standalone for testing).
#
# POLICY (empirically calibrated against the 2026-07-03 public tree; see
# internal research notes and the
# repo-publish-architecture-and-scrub-policy memory):
#   SCRUB → 0 on public: model names (architect/a model/a model/a model), stray `an assistant`,
#           `verification`, `implementation pass`, `working note(s)`, `helper`,
#           `~` absolute paths, `the upstream engine` cross-repo refs.
#   KEEP (accepted public scientific-method vocabulary): architect, pre-reg /
#           pre-registration, freeze, goal-seek. These are FEATURES (pre-registered
#           blind derivation, the anti-numerology goal-seek guard) — never scrubbed.
#
# Usage (standalone test):  scripts/lib/public_scrub.sh <dir>
#   scrubs <dir> in place, then runs the guard; exit 0 iff clean.

set -euo pipefail

# ── the denylist the guard enforces (extended REs; must be ZERO post-scrub) ──
# Kept as a parallel array of "label|regex" so a survivor reports which rule.
# Model names use an UNDERSCORE-AWARE boundary ([^A-Za-z] not \b) so that names
# embedded in filename tokens (…internal research notes…) are caught, while letter-
# embedded innocents (octOPUS, afFABLE) are spared. "a model"/"a model" version
# mentions are model names too and are caught by the same rule.
scrub_denylist_res() {
  cat <<'EOF'
model-name|(^|[^A-Za-z])(architect|a model|a model|a model)([^A-Za-z]|$)
an assistant|(^|[^A-Za-z])an assistant([^A-Za-z]|$)
verification|sealed[ _-]check
implementation pass|\bimplementers?\b
working note|station[ _-]returns?
helper|\bsub-?agents?\b
home-path|~
the upstream engine|\bcwm\b
tier2-ref|(\.\.\/)*(docs\/)?(scoping|open_problems)\/[A-Za-z0-9._/-]+|(\.\.\/)*papers\/[A-Za-z0-9._/-]+
EOF
}

# ── scrub one text file in place ─────────────────────────────────────────────
# All replacements are case-insensitive (I) and word-anchored (\b) where a bare
# token could be a substring of an innocent word (affable⊃architect, corpus⊄a model…).
_scrub_file() {
  local f="$1"
  # (1) Tier-2 dead links: strip a markdown link whose URL points into a filtered
  #     dir, keeping the visible text.  txt -> txt
  sed -E -i \
    -e 's/\[([^]]*)\]\([^)]*(scoping|open_problems|papers)[^)]*\)/\1/g' \
    "$f"
  # (2) Bare/backticked filename tokens that EMBED a model name in the handoff
  #     naming scheme (…internal research notes…) — dead Tier-2 refs that also leak names.
  #     Do this BEFORE the path rule (some are bare filenames with no dir prefix).
  sed -E -i \
    -e 's#`?[A-Za-z0-9./_-]*_(architect|a model|a model|a model)_[A-Za-z0-9./_-]*`?#internal research notes#Ig' \
    "$f"
  # (3) Any remaining Tier-2 path token (backtick, prose, comment) -> neutral phrase.
  #     The path class includes '/' so multi-segment paths (internal research notes
  #     FILE.md) are consumed whole, not left as a dangling '/FILE.md' fragment.
  sed -E -i \
    -e 's#`?(\.\./)*(docs/)?(scoping|open_problems)/[A-Za-z0-9._/-]+`?#internal research notes#g' \
    -e 's#`?(\.\./)*papers/[A-Za-z0-9._/-]+`?#internal research notes#g' \
    "$f"
  # (4) Model-version mentions ("a model", "a model", "a model") -> a model.
  sed -E -i \
    -e 's/\b(an assistant[- ])?(architect|a model|a model|a model)([- ][0-9][0-9.-]*)+/a model/Ig' \
    "$f"
  # (5) Bare model-name mentions in prose (architect denotes the architect role here).
  sed -E -i \
    -e 's/\bfable\b/architect/Ig' \
    -e 's/\b(a model|a model|a model)\b/a model/Ig' \
    -e 's/\bclaude\b/an assistant/Ig' \
    "$f"
  # (6) Multi-agent process vocabulary + absolute paths + cross-repo refs.
  #     Verb forms first (checked/checking) so suffixes are not orphaned.
  sed -E -i \
    -e 's/sealed[ _-]checkers/verifiers/Ig' \
    -e 's/sealed[ _-]checker/verifier/Ig' \
    -e 's/sealed[ _-]checked/verified/Ig' \
    -e 's/sealed[ _-]check(ing|s)?/verification/Ig' \
    -e 's/\bimplementers\b/implementation passes/Ig' \
    -e 's/\bimplementer\b/implementation pass/Ig' \
    -e 's/station[ _-]returns/working notes/Ig' \
    -e 's/station[ _-]return/working note/Ig' \
    -e 's/\bsub-?agents\b/helpers/Ig' \
    -e 's/\bsub-?agent\b/helper/Ig' \
    -e 's###g' \
    -e 's#.#.#g' \
    -e 's#~#~#g' \
    -e 's/\bcwm\b/the upstream engine/Ig' \
    "$f"
}

# ── scrub a whole materialized tree in place (text files only) ────────────────
scrub_tree_inplace() {
  local dir="$1"
  local f
  # grep -I skips binary; -r walks; -l lists candidate files. Restrict to files
  # that actually contain a denylist token OR a tier-2 link, so we touch few files.
  while IFS= read -r -d '' f; do
    _scrub_file "$f"
  done < <(grep -rIlZ -E --exclude=.gitignore --exclude=.gitattributes \
      '(^|[^A-Za-z])(architect|a model|a model|a model|an assistant)([^A-Za-z]|$)|\bcwm\b|sealed[ _-]check|\bimplementer|station[ _-]return|sub-?agent|~|(scoping|open_problems|papers)/[A-Za-z0-9._-]' \
      "$dir" 2>/dev/null || true)
}

# ── the guard: re-grep the scrubbed tree; nonzero + report if anything survives ─
scrub_guard() {
  local dir="$1"
  local rc=0 label re hits
  while IFS='|' read -r label re; do
    [[ -z "$label" ]] && continue
    hits="$(grep -rIlE --exclude=.gitignore --exclude=.gitattributes "$re" "$dir" 2>/dev/null || true)"
    if [[ -n "$hits" ]]; then
      rc=1
      echo "✗ scrub guard: denylist token [$label] survives in:" >&2
      printf '    %s\n' "$hits" | sed "s#$dir/##" >&2
    fi
  done < <(scrub_denylist_res)
  return $rc
}

# ── standalone entrypoint (for testing on a scratch dir) ─────────────────────
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  target="${1:?usage: public_scrub.sh <dir>}"
  echo "scrubbing tree: $target"
  scrub_tree_inplace "$target"
  if scrub_guard "$target"; then
    echo "✓ scrub guard clean (no denylist tokens survive)"
  else
    echo "✗ scrub guard FAILED" >&2
    exit 1
  fi
fi
