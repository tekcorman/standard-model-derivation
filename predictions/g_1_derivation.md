# Derivation of g_1(M_Z) — U(1)_Y gauge coupling, GUT-normalized

**Date:** 2026-05-04 EOD+1. **Status:** THEOREM-GRADE-CONDITIONAL.

## Abstract

g_1(M_Z) GUT-normalized from RG running α_GUT → α_1(M_Z) via one-loop MSSM, with g_1 = √(4πα_1).

## Framework axioms invoked

Inherited via M_unif (Row P62) + M_Z (Row TBD).

## Derivation

**Step 1**: α_GUT = 1/24, M_unif from `predictions/M_unif.py`, M_Z from `predictions/M_Z.py` [Type 4].

**Step 2**: 1/α_1(M_Z) = 1/α_GUT - (b_1/2π) ln(M_Z/M_unif), b_1 = 33/5 (MSSM) [Type 3].

**Step 3**: g_1 = √(4πα_1) [Type 2].

## Result

$$g_1(M_Z) = \sqrt{4\pi \alpha_1(M_Z)} \approx 0.4628 \text{ (GUT-normalized)}$$

Equivalent SM hypercharge coupling: g' = g_1 × √(3/5) ≈ 0.358.

## Comparison with experiment

| Source | Value | Deviation |
|---|---|---|
| Derived from PDG α_EM, sin²θ_W | g_1 ≈ 0.4626 | reference |
| Framework | 0.4628 | +0.04% |

Evaluated against σ_PDG only — the +0.04% deviation corresponds to roughly +25σ_PDG (FAIL Clause 8).

## Open questions

Two-loop + threshold corrections would tighten match. Per parameter linter §2c, RG-running parameter; bridge convention does NOT apply.

## References

`predictions/M_unif.py`, `predictions/M_Z.py`, `predictions/alpha_GUT.py`, `predictions/alpha_EM.py`.
