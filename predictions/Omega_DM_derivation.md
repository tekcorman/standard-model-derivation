# Omega_DM — absolute dark-matter fraction (ΛCDM-fit frame)

**Parameter:** Omega_DM · **File:** `predictions/Omega_DM.py` · **Row:** P23
**Status:** MATHEMATICALLY-COMPLETE-CONDITIONAL-ON-ADOPTED-z_eff (Type-4 composition). Supersedes `predictions/retracted/Omega_DM.py` (external-Ω_b smuggle removed).

## Abstract

Ω_DM = Ω_m_LCDM(z_eff) × (Ω_DM/Ω_m), composing the bias-function matter fraction at the adopted z_eff with the theorem-grade Row P22 visible/dark partition.

## Framework axioms invoked

- `predictions/Omega_m_LCDM.py` (theorem-grade bias form @ adopted z_eff).
- Row P22 `predictions/Omega_DM_over_Omega_m.py` (UNIQUE-THEOREM-GRADE: 1 − P(k≤k*|Poisson(2k*)) from Cl(2k*) Fock + A2-T waterline; k*=3 Row 4).

## Derivation

Ω_DM = Ω_m_LCDM(z_eff) · (1 − 61·e⁻⁶), the second factor theorem-grade and z-independent (Cl(2k*) Fock is epoch-independent in coasting).

## Result

At adopted z_eff ≈ 1.852, with Ω_DM/Ω_m = 0.8488: **Ω_DM ≈ 0.273**.

## Comparison with experiment

Planck 2018: 0.2645 ± 0.0050 → **+1.7σ_obs** at adopted z_eff (definitional band wider). The matter-fraction closure Ω_DM + Ω_b = Ω_m_LCDM holds exactly (Type-4).

## Open questions

Inherits z_eff's. Plus: the Row P22 ratio (0.8488) vs Planck-empirical Ω_DM/Ω_m (≈0.843) is a ~0.7% partition residual NOT movable by z_eff (it's downstream of the total); a separate non-load-bearing item.

## References

`predictions/Omega_m_LCDM.py`; `predictions/Omega_DM_over_Omega_m.py`; `predictions/z_eff.py`; `docs/parameters/parameter_uniqueness_ledger.md` Row P23.
