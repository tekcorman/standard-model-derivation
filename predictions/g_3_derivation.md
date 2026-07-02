# Derivation of g_3(M_Z) — SU(3)_c gauge coupling

**Date:** 2026-05-26 EOD+1 (sector-specific c_color = 1/4 update; supersedes 2026-05-17 OUT-OF-SCOPE re-grade).
**Status:** ✅ **THEOREM-GRADE-NUMERICAL** for the SU(3)_c sector under sector-specific dark correction c_color = 1/4 per `../theorems/theorem_alpha_GUT_sector_specific_c_BST_J_2026-05-26.md`. Live-node value 1.2171 (−0.18σ_PDG). Authority: `../parameters/target_parameters.md` row g_3.

## Abstract

g_3(M_Z) = √(4π α_3(M_Z)) from one-loop MSSM RG running of α_3 from M_unif to M_Z, with α_3^observed at M_unif computed via sector-specific dark correction c_color = 1/4 derived from BS-T × J=±1 algebraic decomposition of K_4 Hashimoto marginal modes (Wilson-loop H¹ content only). Consistency requirement: α_s = g_3²/(4π) must agree with `predictions/alpha_s.py`; both files share the same c_color path.

## Framework axioms invoked

Inherited via α_s (this file is a thin √(4π·) wrapper around the α_s derivation):
- **A1**, **A2-T**, **A4** (per `alpha_s_derivation.md`)

## Derivation

**Step 1** — α_3^observed at M_unif via sector-specific c_color = 1/4 (this file's substantive content, inherited from `theorem_alpha_GUT_sector_specific_c_BST_J_2026-05-26.md`):

$$\alpha_3^{\rm observed} = \alpha_{\rm GUT}^{\rm bare} \cdot \left(1 - \frac{1}{4} \cdot \frac{256}{6305}\right) = \frac{6241}{151320}, \quad 1/\alpha_3^{\rm observed} = 24.2461.$$

The c_color = 1/4 derivation: V_cycle (= J=-1 sub-sector of B's u=±1 eigenspace on K_4) has dim β_1 = 3, equal to dim H¹(K_4; ℝ). Standard SU(N) lattice gauge theory (Wilson 1974) restricts SU(3)_c gauge-boson self-energy to Wilson-loop H¹ content, so the dark Q-projector samples V_cycle only. Therefore c_color = dim V_cycle / (2|E|) = 3/12 = 1/4.

**Step 2** — One-loop MSSM RG running with b_3 = -3 (asymptotic freedom; Martin 1997 §6.4 Eq. 6.30):

$$\frac{1}{\alpha_3(M_Z)} = \frac{1}{\alpha_3^{\rm observed}} - \frac{b_3}{2\pi}\ln\frac{M_Z}{M_{\rm unif}} = 24.2461 + \frac{3}{2\pi}(−33.014) = 8.4843.$$

**Step 3** — g_3 from α_3 (Type 2 algebra):

$$g_3(M_Z) = \sqrt{4\pi \alpha_3(M_Z)} = \sqrt{4\pi / 8.4843} = 1.2171.$$

## Result

$$\boxed{g_3(M_Z) = 1.2171}$$

## Comparison with experiment

| Source | Value | Deviation |
|---|---|---|
| PDG 2024 derived from α_s | 1.218 ± 0.005 | reference |
| Framework (this work, c_color=1/4) | **1.2171** | **−0.18σ_PDG** (−0.07%) |
| Framework (prior, uniform c=1/3, OUT-OF-SCOPE) | 1.211 | −1.36σ_PDG (−0.57%) |
| Framework (pre-α_GUT-DC, retired) | 1.235 | +1.4% |

The sector-specific c_color = 1/4 closes the residual within σ_PDG. The prior "OUT-OF-SCOPE-BY-CONSTRUCTION" attribution to hadronic-VP / threshold matching is SUPERSEDED.

**Internal consistency:** α_s = g_3²/(4π) = 1.2171²/(4π) = 0.1179 = `predictions/alpha_s.py`'s value. Verified in `predictions/g_3.py` `__main__` block (assertion).

## Open questions

Same as `alpha_s_derivation.md`:
- Two-loop precision: framework's M_unif is one-loop-tuned; two-loop running breaks cluster precision (W23 finding). NOT a defect of this derivation specifically.

RG-running parameter per linter §2c; bridge convention does NOT directly apply to g_3(M_Z) (the boundary at M_unif uses sector-specific bridge per the theorem doc).

**Supersedes:** `proofs/gauge/g_3_derivation.py` (retracted, used sin²θ_W = 3/13).

## Linter clause verdict

Inherits from `alpha_s_derivation.md`: ALL 9 CLAUSES PASS → THEOREM-GRADE-NUMERICAL. See an internal working note for the full audit.

## References

- `alpha_s_derivation.md` (parent derivation; g_3 = √(4π α_s))
- `../theorems/theorem_alpha_GUT_sector_specific_c_BST_J_2026-05-26.md`
- `predictions/M_unif.py`, `predictions/M_Z.py`, `predictions/alpha_s.py`
- Martin, S.P. (1997). hep-ph/9709356 §6.4
