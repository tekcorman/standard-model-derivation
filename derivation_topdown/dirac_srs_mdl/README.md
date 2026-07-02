# dirac_srs_mdl — a walled-off clean room

**THE WALL.** This directory imports **nothing** from the rest of the repository — no
`proofs/`, no `predictions/`, no `docs/`, no `common.py`. It is entirely self-contained.
Nothing here may reference the Standard Model, particles, couplings, masses, PDG values,
or any fitted/imported number. If a future file here imports from outside this directory,
it has broken the wall and does not belong.

**THE THREE INGREDIENTS — and nothing else:**
1. **srs** — the substrate = the maximal abelian (ℤ³) cover of the complete graph K₄
   (Sunada's *K₄ crystal*, a.k.a. the (10,3)-a / Laves / srs net). Defined here from scratch.
2. **the Dirac operator** on srs.
3. **MDL** — minimum description length, as the selection principle.

**THE ONLY QUESTION:** *what mathematical structure flows from {D, srs, MDL}?*

**DISCIPLINE.** Compute, do not assert. Report mathematical structure (spectra, zeta
functions, symmetry groups, algebra), never a physical interpretation. This is pure math.

## Files
- `srs.py` — the substrate and its operators (adjacency, Hodge–Dirac, non-backtracking, Ihara ζ).
- `explore_01_what_flows.py` — first pass: spectrum, symmetry, Ihara–Bass, Ramanujan structure.
