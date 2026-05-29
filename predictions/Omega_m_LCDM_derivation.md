# Omega_m_LCDM — ΛCDM-fit matter fraction (downstream of adopted z_eff)

**Parameter:** Omega_m_LCDM · **File:** `predictions/Omega_m_LCDM.py` · **Row:** P24
**Status:** MATHEMATICALLY-COMPLETE-CONDITIONAL-ON-ADOPTED-z_eff (bias-function FORM theorem-grade; one adopted input z_eff, N_hub-class).

## Abstract

The ΛCDM-fit total matter fraction is the theorem-grade local-Friedmann bias function evaluated at the ADOPTED cosmology effective redshift z_eff. One adopted number replaces ΛCDM's free Ω_m.

## Framework axioms invoked

- H_coast(z)=H₀(1+z) (Row P19, theorem-grade) and ΛCDM two-component Friedmann; the bias function is their forced equality.
- z_eff adopted (predictions/z_eff.py, N_hub-class).

## Derivation

Setting H_coast²(z)=H_LCDM²(z): (1+z)²−1 = Ω_m[(1+z)³−1] ⇒
Ω_m_LCDM(z) = (u²−1)/(u³−1) = **(u+1)/(u²+u+1)**, u=1+z. K-rational, parameter-free, no fitting. Properties: z=0 → 2/3 (substrate-frame, exact); z=√3 → 1/3 (K-rational exact-halving anchor); z→∞ → 0.

## Result

At adopted z_eff ≈ 1.852: **Ω_m_LCDM ≈ 0.322**.

## Comparison with experiment

Observed (Planck 2018): 0.3153 ± 0.0073. Predicted at adopted z_eff: 0.322 → **+0.8σ_obs**. Definitional band (bias-inverted z_eff) → ≈ +3σ_obs (dominant systematic, not collapsed to favorable). Load-bearing validation is the predicted-curve fit quality (χ²/dof≈1.37, zero fitted shape params; see `predictions/z_eff_derivation.md`).

## Open questions

Inherits z_eff's: definitional band, CMB/Item-5 wall (clean resolution), substrate-derivability of z_eff. The form itself is theorem-grade and carries no open question.

## References

`predictions/z_eff.py`; `proofs/cosmology/Lambda_CC_parametric_translation_bias.py`; `docs/parameters/parameter_uniqueness_ledger.md` Row P24.
