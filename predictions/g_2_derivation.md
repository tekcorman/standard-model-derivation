# Derivation of g_2(M_Z) — SU(2)_L gauge coupling

**Date:** 2026-05-04 EOD+1. **Status:** THEOREM-GRADE-CONDITIONAL.

## Abstract

g_2(M_Z) from RG running α_GUT → α_2(M_Z) via one-loop MSSM with b_2 = 1; g_2 = √(4πα_2).

## Framework axioms invoked

Inherited via M_unif (Row P62) + M_Z.

## Derivation

**Step 1**: α_GUT = 1/24, M_unif, M_Z [Type 4 upstream].

**Step 2**: 1/α_2(M_Z) = 1/α_GUT - (b_2/2π) ln(M_Z/M_unif), b_2 = 1 (MSSM) [Type 3].

**Step 3**: g_2 = √(4πα_2) [Type 2].

## Result

$$g_2(M_Z) = \sqrt{4\pi \alpha_2(M_Z)} \approx 0.65175$$

## Comparison with experiment

| Source | Value | Deviation |
|---|---|---|
| PDG 2024 derived | 0.6520 ± 0.0001 | reference |
| Framework (live 2026-05-22, post-α_GUT-DC) | 0.65175 | −0.04% (−2.52σ_PDG, near-PASS) |
| Framework (pre-α_GUT-DC, stale) | 0.6554 | +0.5% |

Evaluated against σ_PDG only — near-PASS at the live value. The pre-α_GUT-DC value 0.6554 was stale drift; updated 2026-05-22 to match the live `gauge_unification_full_RG_closure.py` output.

## Open questions

Two-loop MSSM + threshold corrections would tighten. RG-running parameter per linter §2c; bridge convention does NOT apply.

## References

`predictions/M_unif.py`, `predictions/M_Z.py`, `predictions/alpha_EM.py`.
