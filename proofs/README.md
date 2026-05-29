# proofs/

Standalone verification scripts ("probes") for the framework's derivations —
≈810 of them, the largest directory in the repo. This README is the map.

## What this directory is — and is not

`proofs/` is **not an imported library.** Each script is self-contained: it runs
one calculation or structural check and prints a value or PASS/FAIL. The
framework's executable *product* is `predictions/` (the parameter DAG) plus
`verify.py` (the backbone check), a ~115-file import core.

A probe exists to *establish* a result that a `theorems/` or
`predictions/*_derivation.md` doc then cites; once cited, the **doc carries the
claim and the probe is the evidence behind it.** A probe's relevance is
therefore measured by *which docs cite it*.

## Relevance tiers (snapshot 2026-05-21, post-triage)

Each probe is classified by where its filename is cited across `docs/`:

| Tier | Count | Meaning |
|------|------:|---------|
| **current** | 449 | Cited by at least one non-archived doc (a live `theorems/`, `predictions/*_derivation.md`, `framework/`, or an internal working note file). Backs a claim the framework currently makes. |
| **archive-only** | 247 | Cited *only* by archived docs (an internal working note, an internal working note, an internal working note, an internal working note, `predictions/retracted/`). Backed work since archived — likely superseded; treat as historical unless re-verified. |
| **orphan** | 114 | Cited by no doc. Of these, **42 superseded/dead-end probes have been retired to `_archive/`** (see below); the remaining ~72 are either uncited-but-load-bearing or pending the citation-graph repair. |

**Caveat — doc-citation can undercount.** A probe uncited by any doc may still
be load-bearing as a *code* utility imported by live probes — e.g.
`srs_cycle_enumerator.py` is imported by 14 other scripts. Before treating an
"orphan" as disposable, also check `git grep "import <name>"`.

## Per-subdirectory

| Subdirectory | Topic | Scripts | current | archive-only | orphan |
|--------------|-------|--------:|--------:|-------------:|-------:|
| `foundations/` | Substrate / srs structure, Cl(6), Hashimoto walker, observer | 521 | 296 | 192 | 33 |
| `cosmology/` | H_0, Λ_CC, cascade, N_hub, bias function | 106 | 52 | 39 | 15 |
| `flavor/` | CKM / PMNS, Yukawas, Koide, Majorana phases | 62 | 36 | 6 | 20 |
| `_archive/` | Retired superseded / dead-end probes (see `_archive/README.md`) | 42 | 0 | 0 | 42 |
| `masses/` | Absolute mass scales, m_W / m_Z, M_unif | 40 | 30 | 6 | 4 |
| `gauge/` | β-coefficients, gauge couplings, unification | 15 | 15 | 0 | 0 |
| `wave_engine/` | The 219-operation derivation engine | 15 | 13 | 2 | 0 |
| `lorentz/` | Lorentz signature, causal sector | 9 | 7 | 2 | 0 |
| **Total** | | **810** | **449** | **247** | **114** |

(`common.py` and `__init__.py` are package infrastructure, not probes.)

## proofs/_archive/

42 probes retired by the 2026-05-21 orphan triage — cited by no doc and, on
being read and cross-checked, found superseded or dead-end. Kept as the record
of abandoned routes; nothing there is live. See `_archive/README.md`.

## Checking one file's tier yourself

    git grep -l '<script-basename>' -- 'docs/**/*.md'   # doc citation
    git grep -l 'import <script-basename>'              # code use

Any doc hit outside an internal working note, an internal working note, an internal working note,
an internal working note — **current**. Hits only inside those — **archive-only**. No doc hits
and no importers — **orphan**.
