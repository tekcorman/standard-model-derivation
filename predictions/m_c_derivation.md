# Derivation of m_c (charm quark mass)

**Status:** THEOREM-GRADE-STRUCTURAL-CONDITIONAL.
**File:** `predictions/m_c.py`

## Abstract

The charm quark mass is derived as the middle generation of the up
sector via the Koide cosine ratio anchored on the framework's m_t. The
within-sector Koide parameters are fully theorem-grade:
ε²_up = 2 + 6·α₁_full·14/5 (color-generation entanglement on the K_3
Laplacian) and δ_up = 2/27 (W3 PS sector connectivity closure, this
session). Result: 1.277 GeV, +0.56% vs PDG 1.27 GeV.

## Framework axioms invoked

A1, A2-T, A3-T (Born rule), A5(b), B3, R3.

## Derivation

### Step 1 — Within-species Koide cosine spectrum

Per the persistence theorem framing and the cyclic-Toeplitz Hermitian
form of M_gen on C³_obs, the three masses in a sector satisfy:

$$\sqrt{m_j} = \sqrt{M_0} \, (1 + \varepsilon \cos(2\pi j/k_* + \delta)), \quad j = 0, 1, 2.$$

The three values for j = 0, 1, 2 sorted ascending give (f_min, f_mid,
f_max). Anchoring the heaviest mass to m_t:

$$m_c = m_t \cdot \left(\frac{f_\text{mid}}{f_\text{max}}\right)^2.$$

### Step 2 — ε²_up from PS leptoquark coset + many-body expansion (theorem-grade)

Per `../docs/theorems/theorem_quark_koide_eps_n_2026-05-26.md` (W4, theorem-grade-structural;
verification probe `proofs/foundations/W27_eps_n_theorem_closure_2026-05-26.py`, 7/7 PASS):

$$\varepsilon^2_n = 2 + N_{\rm LQ} \, \alpha_{1,\text{full}} \, n \, f(n), \quad f(n) = 1 + (n-1)(g-2)/(2g)$$

where $N_{\rm LQ} = \dim \mathrm{SU}(4)/(\mathrm{SU}(3)\times\mathrm{U}(1)) = 15 - 8 - 1 = 6$
is the Pati-Salam leptoquark coset dimension (the broken SU(4)$_{\rm PS}$ generators
mediating inter-sector Koide-deviation), and the many-body factor $n\,f(n)$ decomposes as
$n$ one-body contributions + $\binom{n}{2}$ pair correlations with pair-correlation
ratio $\alpha_{12}/\alpha_1 = (g-2)/g$ (Type 4 from `koide_quark_ratio_derivation.md` Step 3).

For up sector (n = 2, g = 10): f(2) = 1 + 8/20 = 14/10.
$$\varepsilon^2_\text{up} = 2 + 6 \cdot \alpha_{1,\text{full}} \cdot 2 \cdot (14/10) = 2 + (84/10)\alpha_{1,\text{full}}.$$

With α₁_full = (5/3)(2/3)^8 = 1280/19683:
$$\varepsilon^2_\text{up} \approx 3.0925, \quad \varepsilon_\text{up} \approx 1.7585.$$

### Step 3 — δ_up via W3 PS sector connectivity (theorem-grade-structural)

Per `docs/theorems/theorem_W3_PS_sector_connectivity_2026-05-26.md`
(promoted this session): δ(n) = 2/(9(n+1)). For n = 2:

$$\delta_\text{up} = \frac{2}{27} \approx 0.0741 \text{ rad} = 4.244°.$$

### Step 4 — Evaluate Koide factors

With ε ≈ 1.7585 and δ = 2/27:
- f(j=0) = 1 + ε·cos(δ) = 2.754 (heaviest, f_max)
- f(j=1) = 1 + ε·cos(2π/3 + δ) = 0.236 (f_mid for the up sector — assignment)
- f(j=2) = 1 + ε·cos(4π/3 + δ) = 0.0104 (f_min)

Actually for the assignment in the up sector with this δ value, sorting
ascending gives:
- f_min ≈ 0.0104 (→ m_u)
- f_mid ≈ 0.236 (→ m_c)
- f_max ≈ 2.754 (→ m_t, anchor)

### Step 5 — Direct evaluation

$$m_c = m_t \cdot \left(\frac{0.2358}{2.7537}\right)^2 = 174.10 \cdot (0.0856)^2 = 1.277 \text{ GeV}.$$

## Result

$$\boxed{\;m_c = m_t \cdot \left(\frac{1 + \varepsilon_\text{up} \cos(2\pi/3 + \delta_\text{up})}{1 + \varepsilon_\text{up} \cos(\delta_\text{up})}\right)^2 = 1.277 \text{ GeV}\;}$$

## Comparison with experiment

| Quantity | Predicted | PDG 2024 | Deviation |
|---|---|---|---|
| m_c | 1.277 GeV | 1.27 ± 0.02 GeV (MS-bar at m_c) | **+0.56%, +0.35σ_PDG** |

## Inputs

| Symbol | Value | Status | predictions/ file | Meaning |
|---|---|---|---|---|
| m_t | 174.10 GeV | [derived] | m_t.py | up-sector gen-3 anchor (Type II) |
| alpha_1_full | 1280/19683 | [derived] | alpha_1_full.py | chirality coupling |
| k_star | 3 | [derived] | k_star.py | |
| g_girth | 10 | [derived] | g_girth.py | |

## Open questions

1. **The "middle" assignment.** Within the up-sector Koide cosine,
   physical m_c is the middle mass (between m_t and m_u). The framework
   assigns this to f_mid by ordering — verifying this against an
   independent labeling theorem (e.g., via flavor structure) is open.
2. **Sub-leading corrections.** The 0.56% residual is at the framework's
   structural floor.

## Cross-references

- `docs/theorems/theorem_W3_PS_sector_connectivity_2026-05-26.md` (δ_up = 2/27)
- `predictions/koide_quark_ratio.py` (ε² up/down ratio 14/5)
- `predictions/_koide_quark.py` (Koide factor helper)
- `predictions/m_t.py`, `predictions/M_persistence.py`
