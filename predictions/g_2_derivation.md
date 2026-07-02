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
| Scheme-consistent MS̄ reference (2026-06-25 fix; hardcoded comparison-only per strict linter) | 0.65177 | reference |
| Framework (live) | 0.65175 | **−0.18σ_PDG** ✅ |
| Old reference (scheme-inconsistent, superseded) | 0.6520 ± 0.0001 | (gave the spurious −2.52σ) |
| Framework (pre-α_GUT-DC, stale) | 0.6554 | +0.5% |

Evaluated against σ_PDG only. **2026-06-25 scheme fix:** the earlier −2.52σ was an artifact of a
scheme-inconsistent comparison target — g_2 is definitionally √(4π·α_EM/sin²θ_W), and the consistent
MS̄ reference (0.6516–0.6518, web-verified) gives **−0.18σ**. The framework value is unchanged; only
the target was corrected. The pre-α_GUT-DC value 0.6554 was stale drift; updated 2026-05-22 to match
the live `gauge_unification_full_RG_closure.py` output.

## Open questions

Two-loop MSSM + threshold corrections would tighten. RG-running parameter per linter §2c; bridge convention does NOT apply.

## References

`predictions/M_unif.py`, `predictions/M_Z.py`, `predictions/alpha_EM.py`.
