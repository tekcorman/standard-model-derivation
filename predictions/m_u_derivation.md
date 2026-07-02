# Derivation of m_u (up quark mass)

**Status:** THEOREM-GRADE-STRUCTURAL-CONDITIONAL.
**File:** `predictions/m_u.py`

## Abstract

The up quark mass is the lightest generation of the up sector via the
Koide cosine ratio anchored on m_t with framework-derived ε²_up = 2 +
6·α₁_full·14/5 and δ_up = 2/27 (W3 closure). Result: 2.50 MeV, +15.5% vs
PDG 2.16 MeV — within PDG 1σ (m_u has the largest experimental
uncertainty in the quark sector).

## Framework axioms invoked

A1, A2-T, A3-T, A5(b), B3, R3.

## Derivation

The up sector (n = 2) Koide cosine yields three factors (f_min, f_mid,
f_max) per `_koide_quark.py`; m_u is the lightest:

$$m_u = m_t \cdot \left(\frac{f_\text{min}}{f_\text{max}}\right)^2.$$

With ε_up = √(2 + 28·α₁_full) ≈ 1.7585 and δ_up = 2/27 (W3 theorem):
- f_max ≈ 2.7537
- f_min ≈ 0.0104

m_u = 174.10 · (0.0104/2.7537)² = 174.10 · 1.434×10⁻⁵ = **2.495 MeV.**

## Result

$$\boxed{\;m_u = m_t \cdot \left(\frac{f_\text{min}(\varepsilon_\text{up}, \delta_\text{up})}{f_\text{max}(\varepsilon_\text{up}, \delta_\text{up})}\right)^2 = 2.495 \text{ MeV}\;}$$

## Comparison with experiment

| Quantity | Predicted | PDG 2024 | Deviation |
|---|---|---|---|
| m_u | 2.495 MeV | 2.16 ± 0.49 MeV (MS-bar 2 GeV) | **+15.5%, +0.68σ_PDG** |

m_u has the largest framework-vs-PDG relative deviation in the quark
sector. The +0.68σ_PDG match is **within PDG experimental uncertainty**
(±0.49 MeV is ±23%). The amplified sensitivity is structural: f_min ≈
0.0104 is small (near a cancellation in the Koide cosine), so any small
perturbation in ε_up or δ_up is amplified by ~(f_max/f_min)² in m_u.

This is the analog of m_e in the lepton sector (which the framework
predicts to 0.002% via the same Koide formula; the lepton case is
saturated by precision m_τ measurement). For m_u the framework's
sensitivity is the framework's ε² formula precision (~0.04% per
`koide_quark_ratio.py` ratio check) — small uncertainties amplify near
the cancellation.

## Inputs

| Symbol | Value | Status | predictions/ file | Meaning |
|---|---|---|---|---|
| m_t | 174.10 GeV | [derived] | m_t.py | up gen-3 anchor |
| alpha_1_full | 1280/19683 | [derived] | alpha_1_full.py | |
| k_star, g_girth | 3, 10 | [derived] | k_star.py, g_girth.py | |

## Open questions

1. **Cancellation amplification.** The 15.5% match reflects amplified
   sensitivity at f_min ≈ 0. Sub-leading ε² or δ corrections that the
   framework's selection rule doesn't capture get amplified into m_u.
   Whether the framework can produce more precise ε² and δ via Family-D
   analogs is open research (Priority 4.4 step 2.2 territory).
2. **PDG m_u itself has 23% uncertainty.** The match is within 1σ_PDG.

## Cross-references

Same as `m_c_derivation.md`.
