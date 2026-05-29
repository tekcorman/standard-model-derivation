"""
Generic simulation library for the framework's cosmological scenarios.

This package is *machinery*, not predictions. It is referenced by probes in
proofs/cosmology/ but is NOT importable from predictions/*.py (that path
uses inlined derivations, per the predictions/ DAG contract).

PROJECT-NATIVE vs EXTRACTION-LAYER (read first)
-----------------------------------------------
Per `feedback_no_side_loaded_physics_no_adoptions.md` (2026-05-09):

- The framework's project-native cosmography is `cosmography.coasting`
  (cascade theorem D1+D2+D3, theorem-grade).
- `bias_functions.py` is pure algebraic identity — given two parametric
  classes f and g, find the parameter value in g that matches f at z.
  No physics claim about substrate; just translation between fit classes.
- `cosmography.flat_LCDM` and `flat_wCDM` are EXTERNAL PARAMETRIC CLASSES
  that humans use to fit data. They are NOT framework physics; only
  comparison objects for the bias function machinery.
- `distances.py` and `forward_models.py` use FRW-metric formulae
  (D_C = integral c/H dz, etc.) and the standard candle/ruler
  interpretation of cosmological observables. They are EXTRACTION
  LAYERS — what an external FRW observer extracts from a given H(z) —
  not framework substrate physics. They are useful for comparing
  framework's H(z) to FRW-fit data, but not for making framework
  substrate claims.

Earlier `pressure.py` and `sound_horizon.py` modules — built 2026-05-08,
imported standard-cosmology continuum-fluid hydrodynamics as if it were
framework physics — were DELETED 2026-05-09 per the no-side-loading
correction. The architecture going forward (LCDM-fit emulator, NOT
substrate fluid simulator) is documented in
an internal working note.

Design rules (Hard Quality Gate, parameter_linter.md):

  1. Every public function is a *pure function* of its arguments. The only
     literals permitted inside function bodies are mathematical constants
     (pi, e). All physical inputs — H_0, Omega_m, alpha_EM, c, etc. — are
     explicit named parameters. No hidden defaults that hide a fitted value.

  2. Every quantity carries a Frame tag (Frame.SUBSTRATE / Frame.OBSERVER /
     Frame.LCDM_EXTRACTED). Composition operations propagate the tag. Mixing
     frames without an explicit translation step is a structural error.

  3. The library performs no fitting against observation data. It provides
     forward models that *generate* observable predictions from framework
     parameters; multi-dataset fitting is a separate compositional step that
     a caller must request explicitly.

  4. The library is silent about what frame is "physically real". The
     framework's posture (observer-MDL primary, post-2026-05-07 IC closure)
     is encoded in higher-level probes that select frames, not in the
     library itself. The library exposes all three frames symmetrically.

  5. NO SIDE-LOADED PHYSICS. Modules must not import textbook formulae
     that assume substrate behavior the framework hasn't derived. The
     deletion of pressure.py + sound_horizon.py 2026-05-09 is the
     reference precedent.

Module map (current + planned):

  Layer 1 (project-native):
    ontology.py            — Frame enum, frame-translation primitives.
    cosmography.py         — coasting H(z) factory (cascade theorem); flat_LCDM/wCDM as external comparison classes.
    substrate_densities.py — [Phase B, NOT YET BUILT] rho_m(z), rho_gamma(z), rho_DM, rho_b from substrate.

  Layer 2 (algebra):
    bias_functions.py      — local Friedmann decomposition; closed-form algebra.

  Layer 3 (FRW extraction layer):
    distances.py           — D_C, D_A, D_L via FRW geometry.
    forward_models.py      — SN1a mu(z), CMB theta_* (with r_s as input), BAO D_V.

  Layer 4 (LCDM-fit emulator):
    fisher.py              — per-dataset, per-parameter Fisher info; finite-difference partials of forward-model observables, frame-tagged Fisher matrix container.
    lcdm_fitter.py         — chi^2 minimization on mock observable data; recovers LCDM-class best-fit parameters; covariance via fisher.py at best fit.
    multi_dataset.py       — multi-dataset orchestrator; sums per-dataset chi^2, computes combined Fisher (additivity check), reports z_eff_bias_inversion + z_eff_fisher (per-dataset and combined). z_eff graduated from empirical anchor.

See an internal working note
for architecture, phase plan, and per-target-row closure roadmap.
"""
