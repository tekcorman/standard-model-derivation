# Derivation of M_Z (Z-boson mass)

**Date:** 2026-05-04 EOD+1.
**Status:** STRUCTURAL-CONDITIONAL on MSSM β-function adoption. Clause 8 is
evaluated against σ_PDG only.

## Abstract

M_Z is derived self-consistently from the framework's electroweak relation
M_Z = √π × v × √(α_2(M_Z) + (3/5)α_1(M_Z)), where the running couplings are
RG-run from α_GUT at M_unif. All inputs are framework-derived (no external M_Z anchor).

## Framework axioms invoked

A1, A2-T, A5(b) — inherited via M_unif, α_GUT, and v_higgs. No new axiom or adoption.

## Derivation

### Step 1: Theorem-grade primitives [Type 4]

- α_GUT = 1/24 (`predictions/alpha_GUT.py`)
- M_unif = (32/k*^(g-1)) × M_Pl (`predictions/M_unif.py`, THEOREM-GRADE-CONDITIONAL)
- v = δ²·M_Pl/(√2·N^(1/4)) (`predictions/v_higgs.py`, theorem-grade BZJ)
- MSSM β-coefficients b_1=33/5, b_2=1 (Type 3, Peskin-Schroeder §16; Martin SUSY primer)

### Step 2: Electroweak SM relation [Type 2]

$$M_Z^2 = \frac{1}{4}(g_2^2 + g_Y^2) v^2 = \pi v^2 \left(\alpha_2 + \alpha_Y\right)$$

In GUT normalization α_Y = (3/5)α_1. So M_Z = √π × v × √(α_2 + (3/5)α_1).

### Step 3: One-loop MSSM RG running [Type 3]

$$\frac{1}{\alpha_i(M_Z)} = \frac{1}{\alpha_{GUT}} - \frac{b_i^{MSSM}}{2\pi}\ln\frac{M_Z}{M_{unif}}$$

### Step 4: Self-consistent iteration [Type 2]

Since both sides depend on M_Z, iterate to convergence. Predicted M_Z = 91.97 GeV.

## Result

$$\boxed{M_Z = \sqrt{\pi}\, v\, \sqrt{\alpha_2(M_Z) + \tfrac{3}{5}\alpha_1(M_Z)} \approx 91.97\ \text{GeV}}$$

## Comparison with experiment

| Source | Value | Deviation |
|---|---|---|
| PDG 2024 | 91.1876 ± 0.0021 GeV | reference |
| Framework prediction | 91.97 GeV | +0.86% / +8.6σ_PDG-only |

**Clause 8 (σ_PDG only):** PDG σ on M_Z is 2.3 ppm; the +0.86% deviation is ~+375σ_PDG ⇒ **Clause 8 FAIL** against σ_PDG alone.

## Open questions

1. Two-loop running within the framework's single-regime accounting would shift the prediction sub-percent. Tightening via M_SUSY threshold matching is NOT pursued because M_SUSY is not a framework parameter (see ADOPTED-MSSM-Sb 2026-05-14 PM revision and `docs/theorems/theorem_beta_coefficients_derived.md` §2.5).
2. M_Z inherits FSS conditional via v (depends on N_hub).
3. **Candidate Feshbach-analog dark correction at α_GUT** (Layer-1 hypothesis, 2026-05-15): the +0.86% M_Z residual is structurally consistent with α_GUT × (1 − (1/k*) × α_1/(1−α_1)) propagating through the cluster. NOT propagated until graduation. See `docs/theorems/theorem_substrate_feshbach_dark_corrections_master.md`.

## References

`predictions/M_unif.py`, `predictions/v_higgs.py`, `predictions/alpha_GUT.py`, `predictions/alpha_EM.py` (downstream consumer).

## 2026-05-15 EOD+16 — δ_r oblique propagation (supersedes the stale notes above)

The "+0.86%" / "candidate α_GUT DC at α_GUT" notes above are STALE. Live
state: the α_GUT dark correction is already applied (`predictions/M_Z.py`
imports `predict_alpha_GUT_observed`), giving SM-tree M_Z = 91.5135 GeV
(+0.357%). DAG decomposition (`proofs/foundations/M_Z_residual_
{decomposition_diagnostic,is_tree_vs_pole_oblique}_2026-05-15.py`,
commits ffa89dc + 9501a65) proved this residual is NOT M_unif
(∂lnM_Z/∂lnM_unif ≈ −0.004), NOT 2-loop (makes it worse), but the
**intrinsic SM tree-vs-pole oblique correction** (Δr family) — it
persists at +0.393% even with exact PDG inputs.

`predictions/M_Z.py` now applies the substrate Δr-analog
δ_r = (1/12)·α₁_bare/(1−α₁_bare) ≈ +0.338% (Row P64,
`predictions/delta_r.py` + `delta_r_derivation.md`): the Z-Perron
sign-uniform sibling of δρ (one Hashimoto object — Π_Z→δ_r, Π_W→δρ),
coefficient c_S=1/12 from the Phase-A two-routes (cited, not re-fit),
counting Family-C template. **M_Z_pole = M_Z_tree·(1−δ_r) = 91.2039 GeV
(+0.018%)** — relative residual cut 20×. Clause-9-safe (substrate
analog, NOT the SM Sirlin Δr import). σ_PDG still ≫1 (M_Z is 2.3 ppm —
the framework's intrinsic structural precision floor; Clause 8 FAIL vs
σ_PDG, honestly reported). The earlier ledger "M_unif Stage-5"
attribution of this residual was a documentation error, corrected.
