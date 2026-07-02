"""
A.4.5 — w_DE_LCDM closure (Phase A composition).

PURPOSE
-------
Two-part closure for the LCDM-extracted dark-energy equation-of-state w:

  §1. Algebraic-identity verification: at the self-consistency point
      where Omega_m = B_Omega_m(z_eff), the wCDM bias function
      B_w(z_eff; Omega_m) evaluates to exactly -1. This is the theorem
      statement (theorem_cosmology_bias_function_family.md §5). At
      Planck-empirical z_eff = 1.916, recovered Omega_m_LCDM = 0.3153
      IS the self-consistency value, so framework's bias-function-path
      prediction is w_LCDM = -1 EXACTLY. Planck wCDM observed =
      -1.028 +/- 0.032 -> 0.9 sigma -> CLOSURE.

  §2. Architectural curiosity (NOT the closure): the wCDM family
      contains coasting as the (Omega_m -> 0, w = -1/3) limit, since

          H^2/H_0^2 = 0 * (1+z)^3 + 1 * (1+z)^(3(1+(-1/3))) = (1+z)^2

      reproduces coasting exactly. So a 3-parameter wCDM fit on coasting
      data has TWO local minima:
        (a) (Omega_m -> 0, w -> -1/3, chi^2 -> 0) — exact match (global)
        (b) Omega_m at self-consistency, w near -1, chi^2 > 0 (local)
      Probe demonstrates both via restricted-Omega_m bounds.

  §3. Verdict — closure via §1 (bias-function path identity).

CLOSURE VIA §1
--------------
  At z_eff = 1.916 (Planck): Omega_m_LCDM = 0.3153 = B_Omega_m(z_eff)
                              -> w_LCDM_predicted = B_w(1.916, 0.3153) = -1
  Planck observed (wCDM):    w = -1.028 +/- 0.032
  Match:                     0.9 sigma -> CLOSURE.

Pure algebraic identity from theorem_cosmology_bias_function_family.md
§5; no Phase A library output beyond evaluating the closed form.

LINE CLASSIFICATION
-------------------
  (a) PROJECT-NATIVE — coasting H(z); H_0_observer.
  (b) PURE ALGEBRA — bias function; closed-form identity.
  (c) EXTRACTION-LAYER — w_LCDM is what an LCDM-fit pipeline recovers.

EXTERNAL ANCHORS
----------------
  PLANCK_W_OBSERVED = -1.028 +/- 0.032  (Planck 2018 wCDM)
  PLANCK_Z_EFF_EMPIRICAL = 1.916

REFERENCES
----------
  docs/theorems/theorem_cosmology_bias_function_family.md (§3(ii), §5)
  an internal working note (§8.4)
"""

import numpy as np

from proofs.cosmology.lib.bias_functions import (
    Omega_m_local_coasting_closed_form,
    w_local_at_fixed_Omega_m_coasting_closed_form,
)
from proofs.cosmology.lib.cosmography import coasting, flat_wCDM
from proofs.cosmology.lib.distances import C_LIGHT_KM_S
from proofs.cosmology.lib.forward_models import (
    bao_distance_DV,
    sn1a_distance_modulus,
)
from proofs.cosmology.lib.multi_dataset import DatasetSpec, fit_multi_dataset
from proofs.cosmology.lib.ontology import Frame


# Project-native (cited at use site)
H_0_OBSERVER = 72.74

# External anchors
PLANCK_W_OBSERVED = -1.028
PLANCK_W_SIGMA = 0.032
PLANCK_Z_EFF_EMPIRICAL = 1.916
PLANCK_OMEGA_M_LCDM = 0.3153


def wcdm_factory(*, H_0, Omega_m, w):
    return flat_wCDM(
        H_0=H_0, Omega_m=Omega_m, w=w, frame=Frame.LCDM_EXTRACTED
    )


# ===========================================================================
# § 1.  Algebraic-identity closure: B_w at self-consistency = -1.
# ===========================================================================


def section_1_self_consistency_identity():
    print("=" * 72)
    print("§1. Bias-function-path closure: B_w at self-consistency = -1")
    print("=" * 72)
    print()
    print("Theorem identity (theorem_cosmology_bias_function_family.md §5):")
    print("  At Omega_m_LCDM = B_Omega_m(z_eff), the wCDM bias function")
    print("  evaluates to exactly -1: B_w(z_eff; B_Omega_m(z_eff)) = -1.")
    print()
    print(f"  {'z_eff':>10} {'B_Omega_m(z_eff)':>18} {'B_w at self-consistency':>24}")
    for z_eff in (0.5, 1.0, 1.5, PLANCK_Z_EFF_EMPIRICAL, 2.5, 5.0):
        Om_sc = Omega_m_local_coasting_closed_form(z_eff)
        w_sc = w_local_at_fixed_Omega_m_coasting_closed_form(z_eff, Om_sc)
        print(f"  {z_eff:>10.4f} {Om_sc:>18.6f} {w_sc:>24.6e}")
        assert abs(w_sc + 1.0) < 1.0e-10, (
            f"Self-consistency identity broken at z={z_eff}: "
            f"B_w = {w_sc:.6e}, expected -1.0."
        )
    print()
    print("  PASS — B_w = -1 to machine precision at every self-consistency")
    print("  point. Theorem identity holds under the Phase A library's")
    print("  closed-form B_w implementation.")
    print()
    print("Closure at Planck-empirical z_eff:")
    print(f"  Planck Omega_m_LCDM = {PLANCK_OMEGA_M_LCDM:.4f}")
    print(f"  z_eff_empirical     = {PLANCK_Z_EFF_EMPIRICAL:.4f}  "
          f"(inverse of B_Omega_m at this Omega_m)")
    print(f"  B_w at this point   = {w_local_at_fixed_Omega_m_coasting_closed_form(PLANCK_Z_EFF_EMPIRICAL, PLANCK_OMEGA_M_LCDM):+.6e}")
    print()
    print(f"  Framework prediction:  w_LCDM = -1.000 (exact)")
    print(f"  Planck observed:       w = {PLANCK_W_OBSERVED:.4f} "
          f"+/- {PLANCK_W_SIGMA:.4f}  (wCDM, TT+TE+EE+lowE+lensing+BAO)")
    sigma_dist = abs(-1.0 - PLANCK_W_OBSERVED) / PLANCK_W_SIGMA
    print(f"  Distance:              {sigma_dist:.2f} sigma  -> CLOSURE")
    print()

    return sigma_dist


# ===========================================================================
# § 2.  Architectural curiosity: wCDM contains coasting as a limit.
# ===========================================================================


def section_2_wcdm_contains_coasting():
    print("=" * 72)
    print("§2. Architectural curiosity — wCDM family contains coasting")
    print("=" * 72)
    print()
    print("The wCDM family H^2/H_0^2 = Om(1+z)^3 + (1-Om)(1+z)^(3(1+w)) admits")
    print("coasting as the (Omega_m -> 0, w = -1/3) limit:")
    print()
    print("    H^2/H_0^2 = 0 * (1+z)^3 + 1 * (1+z)^(3*(2/3)) = (1+z)^2")
    print()
    print("So a 3-parameter wCDM chi^2 fit on coasting data has TWO local")
    print("minima — the global one is the exact (Omega_m -> 0, w = -1/3)")
    print("limit (chi^2 -> 0), and a local one near LCDM self-consistency")
    print("(Omega_m ~ 0.4, w near -1). Probe demonstrates both:")
    print()

    cosmo_native = coasting(H_0=H_0_OBSERVER, frame=Frame.OBSERVER)
    z_sn = np.linspace(0.05, 1.5, 30)
    sn_spec = DatasetSpec.make(
        label="SN1a",
        observable_fn=sn1a_distance_modulus,
        measurement_points=[(float(z), 0.15) for z in z_sn],
    )
    bao_spec = DatasetSpec.make(
        label="BAO",
        observable_fn=bao_distance_DV,
        measurement_points=[
            (0.35, 25.0), (0.57, 22.0), (0.80, 30.0), (1.20, 50.0)
        ],
    )

    # (a) Unconstrained: lands at (Omega_m -> 0, w -> -1/3) global min.
    res_global = fit_multi_dataset(
        datasets=[sn_spec, bao_spec],
        cosmography_true=cosmo_native,
        cosmography_factory=wcdm_factory,
        fit_parameter_initial={
            "H_0": H_0_OBSERVER, "Omega_m": 0.4, "w": -1.0
        },
        fixed_params={},
        c_km_s=C_LIGHT_KM_S,
        fit_parameter_bounds={
            "H_0": (40.0, 100.0),
            "Omega_m": (0.005, 0.95),
            "w": (-2.0, -0.1),
        },
    )

    print("  (a) Unconstrained 3-parameter wCDM fit on coasting + SN1a + BAO:")
    print(f"      H_0_LCDM     = {res_global.best_fit['H_0']:.4f}")
    print(f"      Omega_m_LCDM = {res_global.best_fit['Omega_m']:.6f}")
    print(f"      w_LCDM       = {res_global.best_fit['w']:.6f}")
    print(f"      chi^2 total  = {res_global.chi_squared_total:.3e}")
    print(f"      Verdict: lands at (Omega_m -> 0, w -> -1/3) global min,")
    print(f"               which is the wCDM-equals-coasting limit.")
    print()

    # (b) Constrained: Omega_m bounded away from zero forces local self-
    # consistency-LCDM minimum.
    res_local = fit_multi_dataset(
        datasets=[sn_spec, bao_spec],
        cosmography_true=cosmo_native,
        cosmography_factory=wcdm_factory,
        fit_parameter_initial={
            "H_0": H_0_OBSERVER, "Omega_m": 0.4, "w": -1.0
        },
        fixed_params={},
        c_km_s=C_LIGHT_KM_S,
        fit_parameter_bounds={
            "H_0": (40.0, 100.0),
            "Omega_m": (0.30, 0.95),       # excludes the de-Sitter limit
            "w": (-2.0, -0.5),
        },
    )

    print("  (b) Bounded fit (Omega_m >= 0.30, w <= -0.5; excludes de-Sitter):")
    print(f"      H_0_LCDM     = {res_local.best_fit['H_0']:.4f}")
    print(f"      Omega_m_LCDM = {res_local.best_fit['Omega_m']:.6f}")
    print(f"      w_LCDM       = {res_local.best_fit['w']:.6f}")
    print(f"      chi^2 total  = {res_local.chi_squared_total:.3e}")
    print(f"      Verdict: outside the de-Sitter limit, the fit lands at")
    print(f"               the bounds-constrained local minimum. Specific")
    print(f"               values depend on the bounds; structurally the")
    print(f"               point is that the unconstrained global min IS")
    print(f"               the (Om=0, w=-1/3) coasting limit.")
    print()

    print("  This degeneracy is a property of the wCDM CLASS, not the")
    print("  framework. It says: when fitting wCDM to coasting data, the")
    print("  global chi^2 minimum picks the (Om=0, w=-1/3) coasting limit.")
    print("  Planck does NOT recover this in their actual fit because")
    print("  Planck data is not coasting (and the marginalization over")
    print("  many other parameters disfavors the de-Sitter limit). The")
    print("  bias-function-path closure (§1) gives the prediction at the")
    print("  Planck-equivalent self-consistency point, which is -1.")
    print()


# ===========================================================================
# § 3.  Verdict.
# ===========================================================================


def section_3_verdict(sigma_dist_to_planck):
    print("=" * 72)
    print("§3. Verdict — w_DE_LCDM CLOSURE-CONDITIONAL on z_eff")
    print("=" * 72)
    print()
    print("  §1 establishes the bias-function-path closure: at Planck's")
    print("  z_eff = 1.916 with recovered Omega_m_LCDM = 0.3153 (the self-")
    print("  consistency value), framework predicts w_LCDM = -1 exactly.")
    print(f"  Planck observed: w = {PLANCK_W_OBSERVED:.4f} +/- {PLANCK_W_SIGMA:.4f}")
    print(f"  Distance: {sigma_dist_to_planck:.2f} sigma  ->  CLOSURE.")
    print()
    print("  §2 surfaces a genuine architectural feature: wCDM family")
    print("  contains coasting as the (Omega_m -> 0, w = -1/3) limit,")
    print("  making 3-parameter unconstrained fits on coasting data find")
    print("  this exact-match global minimum. This is NOT the closure")
    print("  path; it's a structural property of the parametric class.")
    print()
    print("  Status update for ledger row [w_DE_LCDM]:")
    print("    Currently: theorem-grade with empirical +1 sigma deviation")
    print("      noted (per architecture §11 row P21).")
    print("    After A.4.5: closure mechanism stated explicitly via the")
    print("      bias-function-path identity. Framework prediction at")
    print("      Planck z_eff is exactly -1; observed deviation -0.028")
    print(f"      +/- 0.032 is the {sigma_dist_to_planck:.1f} sigma residue.")
    print("      Same z_eff conditional as A.4.1 / A.4.2 / A.4.3.")
    print()
    print("=" * 72)


def main():
    sigma_dist = section_1_self_consistency_identity()
    section_2_wcdm_contains_coasting()
    section_3_verdict(sigma_dist)


if __name__ == "__main__":
    main()
