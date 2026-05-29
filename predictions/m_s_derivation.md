# Derivation of m_s (strange quark mass)

**Status:** THEOREM-GRADE-STRUCTURAL-CONDITIONAL.
**File:** `predictions/m_s.py`

## Abstract

The strange quark mass is the middle generation of the down sector via
the Koide cosine ratio anchored on m_b with ε²_down = 2 + 6·α₁_full
(Type IV n=1 with f(1)=1) and δ_down = 1/9 (W3 PS sector connectivity
closure). Result: 95.94 MeV, +2.72% vs PDG 93.4 MeV.

## Framework axioms invoked

A1, A2-T, A3-T, A5(b), B3, R3.

## Derivation

Down sector (n = 1) Koide cosine:
- ε²_down = 2 + 6·α₁_full·1·f(1) = 2 + 6·α₁_full (since f(1) = 1).
- α₁_full = (5/3)(2/3)^8 → ε²_down ≈ 2.390, ε_down ≈ 1.546.
- δ_down = 2/(9·2) = 1/9 ≈ 0.1111 rad = 6.366° (W3 theorem-grade-structural).

Sorted Koide factors:
- f_max ≈ 2.536 (→ m_b, anchor)
- f_mid ≈ 0.380 (→ m_s)
- f_min ≈ 0.083 (→ m_d)

$$m_s = m_b \cdot \left(\frac{f_\text{mid}}{f_\text{max}}\right)^2 = 4.270 \cdot (0.380/2.536)^2 = 95.94 \text{ MeV}.$$

## Result

$$\boxed{\;m_s = m_b \cdot \left(\frac{f_\text{mid}(\varepsilon_\text{d}, \delta_\text{d})}{f_\text{max}(\varepsilon_\text{d}, \delta_\text{d})}\right)^2 = 95.94 \text{ MeV}\;}$$

## Comparison with experiment

| Quantity | Predicted | PDG 2024 | Deviation |
|---|---|---|---|
| m_s | 95.94 MeV | 93.4 ± 8.6 MeV (MS-bar 2 GeV) | **+2.72%, +0.30σ_PDG** |

Sub-1σ match. Within framework systematic.

## Inputs

m_b, alpha_1_full, k_star, g_girth (all framework chain).

## Cross-references

- `docs/theorems/theorem_W3_PS_sector_connectivity_2026-05-26.md` (δ_d = 1/9, theorem-grade-structural)
- `docs/theorems/theorem_quark_koide_eps_n_2026-05-26.md` (ε²_d = 2 + 6·α₁_full, theorem-grade-structural via PS leptoquark count N_LQ = 6 + many-body expansion)
- `predictions/m_b.py`, `predictions/_koide_quark.py`
