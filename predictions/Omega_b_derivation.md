# Omega_b — absolute baryon fraction (ΛCDM-fit frame)

**Parameter:** Omega_b · **File:** `predictions/Omega_b.py` · **Row:** P23 (companion)
**Status:** MATHEMATICALLY-COMPLETE-CONDITIONAL-ON-ADOPTED-z_eff (Type-4 composition).

## Abstract

Ω_b = Ω_m_LCDM(z_eff) × (1 − Ω_DM/Ω_m) = Ω_m_LCDM × visible fraction (Row P22 Poisson head P(k≤k*|Poisson(2k*))).

## Framework axioms invoked

Same as `predictions/Omega_DM.py`: bias-function matter fraction @ adopted z_eff × theorem-grade Row P22 partition (visible complement).

## Derivation

Ω_b = Ω_m_LCDM(z_eff) · (1 − (1 − 61·e⁻⁶)) = Ω_m_LCDM(z_eff) · 61·e⁻⁶.

## Result

At adopted z_eff ≈ 1.852: **Ω_b ≈ 0.0487**.

## Comparison with experiment

Planck 2018: 0.04930 ± 0.00046 → **−1.5σ_obs** at adopted z_eff. Matter-fraction closure Ω_DM + Ω_b = Ω_m_LCDM exact (Type-4, asserted in validation).

## Open questions

Inherits z_eff's; plus the same ~0.7% Row-P22-partition-vs-Planck-empirical residual as Ω_DM (downstream of the total, not movable by z_eff; separate non-load-bearing item). The −1.5σ here is dominated by that partition residual, not by z_eff.

## References

`predictions/Omega_m_LCDM.py`; `predictions/Omega_DM_over_Omega_m.py`; `predictions/z_eff.py`; `docs/parameters/parameter_uniqueness_ledger.md` Row P23.
