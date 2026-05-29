# Derivation of m_d (down quark mass)

**Status:** THEOREM-GRADE-STRUCTURAL-CONDITIONAL.
**File:** `predictions/m_d.py`

## Abstract

The down quark mass is the lightest generation of the down sector via
the Koide cosine ratio anchored on m_b with ε²_down = 2 + 6·α₁_full and
δ_down = 1/9 (W3). Result: 4.60 MeV, -1.40% vs PDG 4.67 MeV.

## Framework axioms invoked

A1, A2-T, A3-T, A5(b), B3, R3.

## Derivation

Same Koide cosine machinery as m_s; m_d uses f_min (smallest factor):

$$m_d = m_b \cdot \left(\frac{f_\text{min}(\varepsilon_\text{d}, \delta_\text{d})}{f_\text{max}(\varepsilon_\text{d}, \delta_\text{d})}\right)^2$$

With ε_down ≈ 1.546, δ_down = 1/9:
- f_max ≈ 2.536, f_min ≈ 0.083.
- m_d = 4.270 · (0.083/2.536)² = 4.270 · 0.00108 = **4.605 MeV.**

## Result

$$\boxed{\;m_d = 4.605 \text{ MeV}\;}$$

## Comparison with experiment

| Quantity | Predicted | PDG 2024 | Deviation |
|---|---|---|---|
| m_d | 4.605 MeV | 4.67 ± 0.48 MeV (MS-bar 2 GeV) | **−1.40%, −0.14σ_PDG** |

Sub-σ match. The negative sign indicates the framework slightly under-
predicts m_d, while it slightly over-predicts m_u — these residuals are
correlated through the same ε_down, δ_down values via the Koide
relation, and both stay within sector systematic.

## Inputs

m_b, alpha_1_full, k_star, g_girth (framework chain).

## Cross-references

Same as `m_s_derivation.md`.
