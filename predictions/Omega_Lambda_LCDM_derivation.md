# Omega_Lambda_LCDM — ΛCDM-fit dark-energy fraction

**Parameter:** Omega_Lambda_LCDM · **File:** `predictions/Omega_Lambda_LCDM.py` · **Row:** P24
**Status:** MATHEMATICALLY-COMPLETE-CONDITIONAL-ON-ADOPTED-z_eff (Type-4 inheritance from Omega_m_LCDM).

## Abstract

Flat-ΛCDM normalization: Ω_Λ_LCDM = 1 − Ω_m_LCDM = u²/(u²+u+1), u=1+z_eff. The Λ_CC factor-of-2 of Row P24 is structurally Ω_Λ_LCDM(z_eff)/Ω_Λ_substrate = Ω_Λ_LCDM/(1/3); exactly 2 at the K-rational anchor z=√3.

## Framework axioms invoked

Inherits `predictions/Omega_m_LCDM.py` (theorem-grade bias form) + flat-ΛCDM Ω_total=1.

## Derivation

Ω_Λ_LCDM(z) = 1 − (u+1)/(u²+u+1) = u²/(u²+u+1). z=0 → 1/3 (substrate Ω_Λ, exact); z=√3 → 2/3 (exact); z→∞ → 1.

## Result

At adopted z_eff ≈ 1.852: **Ω_Λ_LCDM ≈ 0.679**; Λ_CC ratio Ω_Λ_LCDM/(1/3) ≈ **2.036** (exactly 2 at z=√3 — the K-rational origin of the Row P24 factor-of-2).

## Comparison with experiment

Planck 2018: 0.6847 ± 0.0073 → **−0.8σ_obs** at adopted z_eff. Λ_LCDM/Λ_substrate observed ≈ 2.05; predicted ≈ 2.04 at z_eff (≈0.2σ on the ratio). Load-bearing validation = predicted-curve fit quality (see `predictions/z_eff_derivation.md`).

## Open questions

Inherits z_eff's (definitional band, CMB/Item-5 wall, substrate-derivability).

## References

`predictions/Omega_m_LCDM.py`; `predictions/z_eff.py`; `docs/parameters/parameter_uniqueness_ledger.md` Row P24.
