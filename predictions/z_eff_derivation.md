# z_eff — cosmology effective redshift (ADOPTED, N_hub-class)

**Parameter:** z_eff · **File:** `predictions/z_eff.py` · **Rows:** P23/P24 (upstream)
**Status:** ADOPTED cosmology parameter (N_hub-pattern). Value computed from the SN+BAO survey Fisher geometry; validated by the downstream predicted-curve fit quality.

## Abstract

z_eff is the framework's one adopted late-time cosmology background parameter, in exact analogy with N_hub (the one adopted dimensional input, value pinned by G_F-consistency). It is the Fisher-information-weighted mean redshift of the SN+BAO survey combination — a property of the survey *design*, not fitted to distances and not substrate-derived. Adopting it lets the theorem-grade bias function fix the entire late-time energy budget from ONE number.

## Framework axioms invoked

- Coasting H_substrate(z)=H₀(1+z) (Row P19, theorem-grade) — underlies the bias function the adopted z_eff feeds.
- Observer-MDL posture (`theorem_observer_persistence_closure_IC_amplitude.md`): cosmological observables are functionals of the observer's compressed model; the observer fits ΛCDM and recovers parameters the bias function maps from z_eff.

## Derivation

z_eff = ∫ z F(z) dz / ∫ F(z) dz, F(z) the per-redshift Fisher information for Ω_m extraction: F_SN ∝ (∂μ/∂Ω_m / σ_μ)²·n_SN(z); F_BAO ∝ (∂D/∂Ω_m / σ_rel)² per anchor. F is fixed by the survey design (z-distribution + error model), passed as [external] inputs (BOSS DR12 + eBOSS DR16 anchors; Pantheon+-like SN model) — the same epistemic status as G_F for N_hub.

## Result

z_eff (SN+BAO Fisher first-moment, ADOPTED) = **1.852**. Definitional alternative (bias-inverted) = 1.663 — the definitional band is the dominant systematic, reported, not collapsed to the favorable value.

## Comparison with experiment

Per the 2026-05-15 amendment the **load-bearing validation is the predicted-curve fit quality**, not a derived-z_eff comparison: the framework's predicted expansion curve (ΛCDM-shaped, Ω_m FIXED = bias(z_eff), zero fitted shape parameters; only the distance scale marginalized) fits the measured BOSS DR12 + eBOSS DR16 BAO consensus at **χ²/dof ≈ 1.37** (first-moment z_eff) vs ΛCDM-best **1.21** (which spends a *free* Ω_m). See `proofs/cosmology/z_eff_predicted_curve_vs_observations_2026-05-15.py` + figure. Non-load-bearing cross-check: observation-implied z_eff (invert bias at Planck Ω_m) = 1.916 ± 0.079; adopted 1.852 → −0.8σ.

## Open questions

- Definitional band (first-moment vs bias-inverted) is the dominant systematic; clean resolution needs CMB-weighted Fisher = Item 5 = the L6 wall (Sprints A/B, doubly-confirmed dead). Favorable definition not selected.
- r_d (sound horizon) is NOT separately predicted — the distance scale is marginalized. That is the honest L6 limitation; the CMB acoustic sector (r_s, θ_*, σ_8, n_s) is out of scope for this late-time model.
- Whether z_eff is substrate-derivable (removing the adoption, like closing Gap-G1 for N_hub) is open.

## References

- `proofs/cosmology/z_eff_predicted_curve_vs_observations_2026-05-15.py` (curve test + figure)
- `proofs/cosmology/z_eff_adopted_reduced_parameter_model_2026-05-15.py`
- `predictions/N_hub.py` (the adopted-input pattern this mirrors)
