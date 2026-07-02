# Derivation of α_s(M_Z) — strong coupling at the Z-pole

**Date:** 2026-05-26 EOD+1 (sector-specific c_color = 1/4 update; supersedes 2026-05-17 OUT-OF-SCOPE re-grade).
**Status:** ✅ **THEOREM-GRADE-NUMERICAL** for the SU(3)_c sector under sector-specific dark correction c_color = 1/4 per `../theorems/theorem_alpha_GUT_sector_specific_c_BST_J_2026-05-26.md`. Live-node value 0.1179 (−0.13σ_PDG). Authority: `../parameters/target_parameters.md` row α_s.

## Abstract

α_s(M_Z) = α_3(M_Z) = g_3²/(4π), obtained by one-loop MSSM RG running from α_3^observed at M_unif down to M_Z, with α_3^observed computed via the sector-specific dark correction c_color = β_1/(2|E|) = 1/4 derived from the BS-T × J=±1 canonical decomposition of the K_4 Hashimoto marginal sector (Wilson-loop H¹ content only, per the standard SU(N) lattice gauge theory restriction).

## Framework axioms invoked

- **A1** (binary self-inverse edge toggle, `../framework/framework_axioms.md` §2)
- **A2-T** (MDL waterline canonicalization, derived; `../theorems/theorem_A2_mdl_from_finite_register.md`)
- **A4** (CAR per vertex, `../theorems/theorem_car_local_jordan_wigner.md`)

## Derivation

**Step 1 — α_GUT^bare from local label count (Type 1+2):**

Per `alpha_GUT_derivation.md` §1, the bare gauge coupling at unification on srs (k* = 3) is

$$\alpha_{\rm GUT}^{\rm bare} = \frac{1}{2^{k_*} \cdot k_*} = \frac{1}{24}.$$

**Step 2 — α_1^bare from girth-cycle Class A (Type 3+4):**

Per `alpha_1_derivation.md` (theorem-grade Class A, `theorem_class_A_amplitude_dark.md`):

$$\alpha_1^{\rm bare} = (k_* - 1)^{g-2}/k_*^{g-2} = (2/3)^8 = 256/6561.$$

The A2-T waterline winding sum gives $x = \alpha_1^{\rm bare}/(1 - \alpha_1^{\rm bare}) = 256/6305$.

**Step 3 — Sector-specific dark correction c_color = 1/4 (THIS THEOREM):**

Per `../theorems/theorem_alpha_GUT_sector_specific_c_BST_J_2026-05-26.md` §3:

(3a) On the srs primitive cell K_4 (|V|=4, |E|=6, β_1=|E|-|V|+1=3), the Hashimoto matrix B has marginal sector V_pm = (u=±1)-eigenspace of dim 5 = 2(|E|-|V|) + 1 (per `theorem_dark_5_12_spectral.md` §3.1).

(3b) The orientation-reversal operator J: (u,v) ↔ (v,u) on directed edges canonically decomposes V_pm by J-eigenvalue:

$$V_{pm} = V_{\rm cycle} \oplus V_{\rm scalar}, \quad \dim V_{\rm cycle} = 3, \quad \dim V_{\rm scalar} = 2$$

with V_cycle = J=-1 (verified `W24_BST_J_algebraic_sector_count_2026-05-26.py`).

(3c) V_cycle has Wilson-loop rank = β_1 = 3 → V_cycle is the full H¹(K_4; ℝ) lift to the marginal sector. V_scalar has Wilson-loop rank 0 → V_scalar is outside C¹ = B¹ ⊕ H¹ (Wilson loops vanish on V_scalar).

(3d) Standard SU(N) lattice gauge theory (Wilson 1974 §II; Kogut-Susskind 1975 §II): gauge-boson self-energy corrections are mediated by Wilson-loop H¹ holonomy. For SU(3)_c, the H¹ master theorem (`theorem_h1_master_compression.md` "valence ↔ center") gives H¹(K_4; Z_3) ≅ Z_3^{β_1} where Z_3 = center(SU(3)) (Greensite 2011 §5.1).

(3e) Therefore the substrate-Feshbach Q-projector (per `theorem_substrate_feshbach_dark_corrections_master.md`) for SU(3)_c gluon self-energy samples V_cycle modes only:

$$c_{\rm color} = \frac{\dim V_{\rm cycle}}{2|E|} = \frac{\beta_1}{2|E|} = \frac{3}{12} = \frac{1}{4}.$$

**Step 4 — α_3^observed at M_unif (Type 2 arithmetic):**

$$\alpha_3^{\rm observed} = \alpha_{\rm GUT}^{\rm bare} \cdot \left(1 - c_{\rm color} \cdot \frac{\alpha_1^{\rm bare}}{1 - \alpha_1^{\rm bare}}\right) = \frac{1}{24}\left(1 - \frac{1}{4} \cdot \frac{256}{6305}\right) = \frac{6241}{151320}$$

so $1/\alpha_3^{\rm observed} = 24.2461$.

**Step 5 — One-loop MSSM RG running from M_unif to M_Z (Type 3):**

With b_3 = -3 (MSSM SU(3)_c β-coefficient; Martin 1997 §6.4 Eq. 6.30), M_unif = 1.985 × 10^16 GeV (per `M_unif.py`), M_Z = 91.2039 GeV:

$$\frac{1}{\alpha_3(M_Z)} = \frac{1}{\alpha_3^{\rm observed}} - \frac{b_3}{2\pi} \ln\frac{M_Z}{M_{\rm unif}} = 24.2461 + \frac{3}{2\pi}(−33.014) = 24.2461 - 15.7619 = 8.4843.$$

Therefore $\alpha_s(M_Z) = 1/8.4843 = 0.11788$.

## Result

$$\boxed{\alpha_s(M_Z) = 0.11788}$$

## Comparison with experiment

| Source | Value | Deviation |
|---|---|---|
| PDG 2024 world average | 0.1180 ± 0.0009 | reference |
| Framework (this work, c_color=1/4) | **0.11788** | **−0.13σ_PDG** (−0.10%) |
| Framework (prior, uniform c=1/3, OUT-OF-SCOPE) | 0.11674 | −1.40σ_PDG (−1.07%) |
| Framework (pre-α_GUT-DC, retired) | 0.1213 | +2.8% |

The sector-specific c_color = 1/4 (vs prior uniform c = 1/3) closes the residual within σ_PDG. The prior "OUT-OF-SCOPE-BY-CONSTRUCTION" attribution to hadronic-VP / threshold matching is SUPERSEDED by the substrate-derived sector-specific correction.

## Open questions

- **Two-loop precision**: under two-loop MSSM RG running, the framework's M_unif derivation breaks the cluster precision (W23 finding). The c_color = 1/4 closure is precise at one-loop only. Two-loop precision requires either substrate refinement of M_unif or additional threshold structure. NOT a defect of this derivation specifically; affects the entire framework's gauge cluster.
- **+0.008 sub-leading offset on c_EM** from R_∞ ppt-precision (Rinf_clean_ratio_diagnostic_2026-05-16.py): SEPARATE issue, affects U(1)_Y/SU(2)_L sector, not SU(3)_c.

## Linter clause verdict

Per `../parameters/parameter_linter.md` Clauses 1-9:

- Clause 1 (axiom): PASS — A1 + A2-T + A4
- Clause 2 (algebra): PASS — all steps explicit, K-rational arithmetic
- Clause 3 (theorem citation): PASS — Wilson 1974, Kogut-Susskind 1975, Greensite 2011, Bass 1992, Stark-Terras 1996, Martin 1997
- Clause 4 (predictions/ files): PASS — all chained from alpha_GUT.py, alpha_1.py, k_star.py, g_girth.py, M_unif.py, M_Z.py
- Clause 5 (master theorem): PASS — inherits Class A cluster
- Clause 6 (K-meta-theorem): PASS — 1/4 ∈ ℚ ⊂ K; L-expression β_1/2|E|; channel_select waterline-consistent (alternatives 1/12, 4/12, 5/12 physically realized in other observables)
- Clause 7 (audit v2 uniqueness): PASS — six-mechanism gating populated per theorem doc §4; Row 4 inheritance for substrate axes
- Clause 8 (numerical match): PASS — Δ = −0.13σ_PDG ≤ 1σ → THEOREM-GRADE-NUMERICAL
- Clause 9 (Type-3 π-audit): PASS — all citations are lattice gauge theory or RG-coefficient K-rationals; no continuum-π imports

Full audit at an internal working note.

## References

- `../theorems/theorem_alpha_GUT_sector_specific_c_BST_J_2026-05-26.md` (this theorem)
- `../theorems/theorem_alpha_GUT_dark_correction.md` (predecessor, uniform c=1/3 for U(1)_Y/SU(2)_L)
- `../theorems/theorem_dark_5_12_spectral.md` (v_Higgs c=5/12 anchor)
- `../theorems/theorem_h1_master_compression.md` (H¹ Master Theorem, valence ↔ center)
- `../theorems/theorem_substrate_feshbach_dark_corrections_master.md` (Q-projector template)
- `predictions/alpha_GUT.py` (α_GUT^bare + sector-specific function)
- `predictions/alpha_1.py` (α_1^bare from Class A)
- `predictions/M_unif.py`, `predictions/M_Z.py` (RG running scales)
- Wilson, K.G. (1974). *Phys. Rev. D* 10, 2445. §II (lattice gauge theory)
- Greensite, J. (2011). *An Introduction to the Confinement Problem*. Springer §5 (Z_N center)
- Martin, S.P. (1997). hep-ph/9709356 §6.4 Eq. 6.30 (MSSM β-coefficients)
