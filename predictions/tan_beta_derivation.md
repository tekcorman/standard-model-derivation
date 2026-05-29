# Derivation of tan(β) (MSSM Higgs VEV ratio)

**Status:** THEOREM-GRADE-STRUCTURAL-CONDITIONAL.
**File:** `predictions/tan_beta.py`

## Abstract

tan(β) is determined by Georgi-Jarlskog bottom-tau unification at M_GUT:
y_b(M_GUT) / y_τ(M_GUT) = k* = 3. The framework's low-scale SM-equivalent
Yukawas (y_τ = α₁_full/k*², y_b = (2/3)^g, both theorem-grade) plus
framework gauge couplings + MSSM 1-loop Yukawa RGE pick out tan(β) ≈
44.73 via the cos(β) bridge factor that maps SM-eff to MSSM Yukawa
conventions. No PDG mass anchors enter the derivation.

## Framework axioms invoked

A1, A2-T, A5(b), B3 (Pati-Salam → MSSM matter content).

## Derivation

### Step 1 — Framework low-scale Yukawas (theorem-grade)

From the selection rule:
- y_τ_SM(M_Z) = α₁_full / k*² ≈ 0.00722 (Type III, predictions/y_tau.py).
- y_b_SM(M_Z) = ((k*-1)/k*)^g = (2/3)^10 ≈ 0.01734 (Type IV, predictions/m_b.py).

Tree-level ratio:
$$\frac{y_b^{SM}}{y_\tau^{SM}} = \frac{(2/3)^{10}}{(5/3)(2/3)^8 / 9} = \frac{(2/3)^2 \cdot 9}{5/3} = \frac{4/9 \cdot 9 \cdot 3}{5} = \frac{12}{5} = 2.4.$$

This is the low-scale SM-effective ratio. Framework prediction at M_GUT
(after MSSM RGE running up from M_Z) is y_b/y_τ = 3 (Georgi-Jarlskog).

### Step 2 — MSSM-SM bridge (Type 3 standard QFT)

In MSSM convention:
$$y_\tau^{MSSM}(M_Z) = \frac{y_\tau^{SM}(M_Z)}{\cos\beta}, \quad y_b^{MSSM}(M_Z) = \frac{y_b^{SM}(M_Z)}{\cos\beta}.$$

cos(β) cancels in the ratio at M_Z, so y_b/y_τ at M_Z is convention-
independent (= 12/5).

### Step 3 — MSSM RGE running from M_Z to M_GUT (Type 3 standard QFT)

The MSSM 1-loop Yukawa β-functions are different for y_b vs y_τ (y_b
couples to g_3² which is much larger than g_1², g_2²; y_τ has weaker
gauge feedback). Bottom-up RGE with tan(β)-dependent ABSOLUTE values
(via cos β) gives different evolution rates, so y_b/y_τ EVOLVES with μ.

### Step 4 — Self-consistency solve

Find tan(β) such that:
$$\boxed{\;\frac{y_b^{MSSM}(M_{GUT}; \tan\beta)}{y_\tau^{MSSM}(M_{GUT}; \tan\beta)} = k_* = 3 \quad \text{(Georgi-Jarlskog unification)}\;}$$

with framework gauge couplings (α_GUT, M_unif, M_Z, MSSM β) supplying
the RGE. Numerical solve via brentq: **tan(β) ≈ 44.73.**

## Result

$$\boxed{\;\tan\beta \approx 44.73\;}$$

Matches `proofs/masses/srs_tan_beta.py` to <1%.

## Comparison with experiment

tan(β) is not directly observed. The MSSM large-tan(β) regime
(~40-50) is required by Georgi-Jarlskog at k*=3 + bottom-tau
unification. The framework's 44.73 lies cleanly within this regime.

## Inputs

| Symbol | Value | Status | predictions/ file |
|---|---|---|---|
| k_star | 3 | [derived] | k_star.py |
| g_girth | 10 | [derived] | g_girth.py |
| alpha_GUT | ≈ 0.04110 | [derived] | alpha_GUT.py |
| alpha_1_full | (5/3)(2/3)^8 | [derived] | alpha_1_full.py |
| M_unif | ≈ 1.985e16 GeV | [derived] | M_unif.py |
| M_Z | ≈ 91.97 GeV | [derived] | M_Z.py |
| GJ = k* | 3 | [derived] | georgi_jarlskog.py |

MSSM 1-loop β-functions: Type 3 standard QFT (Martin SUSY Primer 1997
§5; not derived from substrate).

## Notes on usage

tan(β) is NOT load-bearing for the framework's m_t and m_b prediction
chain — those use the framework's low-scale convention (`m = v·y` for
L>0 walkers, `m = (v/√2)·y` for L=0 walkers per scale assignment).
tan(β) emerges from GJ unification at M_GUT as a consistent MSSM
parameter; it directly controls the Higgs sector (m_h prediction) but
is not propagated to fermion masses.

## Open questions

1. **Independent verification.** The framework's tan(β) = 44.73 matches
   `srs_tan_beta.py` (which uses observed gauge couplings) — these
   small differences (~0.5%) reflect framework α_s ≈ 0.117 vs observed
   ≈ 0.118.
2. **MSSM matter content as structural choice.** The Type 3 standard
   QFT MSSM β-functions are inherited; the framework's commitment to
   MSSM matter content (vs e.g. SM or other GUT) is a theorem-grade-
   structural decision but not derived from A1.

## Cross-references

- `proofs/masses/srs_tan_beta.py` (proof script, observed-gauge version)
- `predictions/georgi_jarlskog.py` (GJ = k* = 3)
- `predictions/y_tau.py`, `predictions/m_b.py` (Yukawa boundaries)
