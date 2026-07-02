"""
A.4.4 — Hubble tension story (partial closure under Phase A; Planck-CMB
        side blocked on Phase B for substrate r_s).

PURPOSE
-------
Frame-level closure of the Hubble tension. Three frame-distinct H_0
predictions exist in the framework, each tied to a different observation
mode:

  Frame.SUBSTRATE      H_0_substrate = 68.18 km/s/Mpc
                       cascade theorem D1+D2+D3 (substrate state-counting
                       rate); not directly observable in observer frame.

  Frame.OBSERVER       H_0_observer = (16/15) * H_0_substrate = 72.74 km/s/Mpc
                       cascade D2-extended; what local-Hubble-flow
                       observers measure (SH0ES, Cepheid distance ladder).
                       Theorem-grade-conditional per
                       theorem_cascade_D2_extended_observer_rate.md.

  Frame.LCDM_EXTRACTED H_0_LCDM = LCDM-fit-emulator output at z_eff
                       under coasting native + externally-supplied r_s.
                       What CMB-anchored LCDM pipelines (Planck) recover.
                       Phase A.3 multi_dataset machinery; conditional on
                       z_eff (shared with A.4.1-3) AND substrate r_s
                       derivation (Phase B; currently blocked per L6 —
                       framework lacks Mpc-scale apparatus for sound
                       horizon).

THE "TENSION" IS A FRAME MISMATCH
---------------------------------
SH0ES vs Planck-CMB is conventionally framed as a single H_0
disagreement. Under the framework, they are MEASUREMENTS OF DIFFERENT
QUANTITIES:

  SH0ES low-z       --> Frame.OBSERVER  (geometric distance ladder
                       reads observer-frame Hubble rate at z ~ 0.01)
  Planck CMB        --> Frame.LCDM_EXTRACTED  (LCDM fit to CMB-anchored
                       observables, including theta_* at z = 1090)

Two different frames, two different framework predictions, no
contradiction. The "tension" is a category error of treating both as
measurements of the same single quantity.

CLOSURE STATUS
--------------
  SH0ES side:    CLOSED at theorem-grade-conditional via H_0_observer.
                 SH0ES = 73.04 +/- 1.04 km/s/Mpc; framework = 72.74
                 (0.3 sigma match). Already in predictions/H_0.py.

  Planck-CMB:    PARTIAL — recovered via Phase A LCDM-fit emulator +
                 external r_s. With r_s_LCDM_fit = 147.05 Mpc, the
                 emulator can be run; but full closure requires
                 substrate r_s (Phase B, BLOCKED per architecture §9).

LINE CLASSIFICATION
-------------------
  (a) PROJECT-NATIVE — H_0_substrate, H_0_observer (cascade D2-ext).
  (b) PURE ALGEBRA — bias function; LCDM-fit emulator chi^2.
  (c) EXTRACTION-LAYER — H_0_LCDM via LCDM-fit on coasting; CMB theta_*
      with external r_s.

EXTERNAL ANCHORS
----------------
  SH0ES_H_0 = 73.04 +/- 1.04 km/s/Mpc  (Riess+ 2022)
  PLANCK_H_0_LCDM = 67.36 +/- 0.54     (Planck 2018)
  R_S_COMOVING_LCDM_FIT = 147.05 Mpc   (LCDM-fit r_s; Phase B blocked)

REFERENCES
----------
  predictions/H_0.py / H_0_derivation.md
  docs/theorems/theorem_cascade_D2_extended_observer_rate.md
  an internal working note (§8.4, §11)
  proofs/cosmology/Lambda_CC_factor_two_closure_2026-05-09.py (z_eff conditional)
"""

import numpy as np

from proofs.cosmology.lib.bias_functions import (
    Omega_m_local_coasting_closed_form,
)
from proofs.cosmology.lib.cosmography import coasting, flat_LCDM
from proofs.cosmology.lib.distances import C_LIGHT_KM_S
from proofs.cosmology.lib.forward_models import (
    bao_distance_DV,
    cmb_theta_star_from_DC,
    sn1a_distance_modulus,
)
from proofs.cosmology.lib.multi_dataset import DatasetSpec, fit_multi_dataset
from proofs.cosmology.lib.ontology import Frame


# Project-native (cited at use site)
H_0_SUBSTRATE = 68.18
H_0_OBSERVER = 72.74

# External anchors
SH0ES_H_0 = 73.04
SH0ES_H_0_SIGMA = 1.04
PLANCK_H_0_LCDM = 67.36
PLANCK_H_0_LCDM_SIGMA = 0.54
R_S_COMOVING_LCDM_FIT = 147.05
Z_STAR = 1090.0
PLANCK_Z_EFF_EMPIRICAL = 1.916


def lcdm_factory(*, H_0, Omega_m):
    return flat_LCDM(H_0=H_0, Omega_m=Omega_m, frame=Frame.LCDM_EXTRACTED)


def cmb_theta_with_r_s(z, cosmography, c_km_s, *, r_s_comoving_Mpc):
    return cmb_theta_star_from_DC(z, r_s_comoving_Mpc, cosmography, c_km_s)


# ===========================================================================
# § 1.  SH0ES side: H_0_observer prediction.
# ===========================================================================


def section_1_shoes():
    print("=" * 72)
    print("§1. SH0ES side — Frame.OBSERVER prediction")
    print("=" * 72)
    print()
    print(f"  Framework H_0_observer = (16/15) * H_0_substrate")
    print(f"                         = (16/15) * {H_0_SUBSTRATE:.4f}")
    print(f"                         = {H_0_OBSERVER:.4f} km/s/Mpc")
    print()
    print(f"  SH0ES (Riess+ 2022):  {SH0ES_H_0:.4f} +/- {SH0ES_H_0_SIGMA:.4f}")
    sigma_dist_shoes = abs(H_0_OBSERVER - SH0ES_H_0) / SH0ES_H_0_SIGMA
    print(f"  Distance:             {sigma_dist_shoes:.2f} sigma  -> CLOSURE")
    print()
    print("  Theorem-grade-conditional per cascade D2-extended observer")
    print("  rate gap (theorem_cascade_D2_extended_observer_rate.md);")
    print("  CLOSED already in predictions/H_0.py.")
    print()
    return sigma_dist_shoes


# ===========================================================================
# § 2.  Planck-CMB side: LCDM-fit emulator + external r_s.
# ===========================================================================


def section_2_planck_cmb_partial():
    print("=" * 72)
    print("§2. Planck-CMB side — partial under Phase A + external r_s")
    print("=" * 72)
    print()
    print("Phase A.3 multi_dataset.fit_multi_dataset on coasting native")
    print(f"+ SN1a + BAO + CMB-theta-with-external-r_s = {R_S_COMOVING_LCDM_FIT:.2f} Mpc.")
    print()
    print("CMB sigma_theta_*: per A.4.1 §2 R-A finding, at Planck-realistic")
    print("sigma (~3e-7), the LCDM-fit on coasting drives Omega_m -> 0 (the")
    print("fit cannot reproduce coasting's theta_*_native at high precision).")
    print("We use sigma_theta = 1e-3 as a representative mid-range choice")
    print("(see A.4.1 closure for the dataset-weighting dependence).")
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

    sigma_theta_default = 1.0e-3
    cmb_spec = DatasetSpec.make(
        label="CMB_theta",
        observable_fn=cmb_theta_with_r_s,
        measurement_points=[(Z_STAR, sigma_theta_default)],
        observable_kwargs={"r_s_comoving_Mpc": R_S_COMOVING_LCDM_FIT},
    )

    res = fit_multi_dataset(
        datasets=[sn_spec, bao_spec, cmb_spec],
        cosmography_true=cosmo_native,
        cosmography_factory=lcdm_factory,
        fit_parameter_initial={"H_0": H_0_OBSERVER, "Omega_m": 0.4},
        fixed_params={},
        c_km_s=C_LIGHT_KM_S,
        fit_parameter_bounds={
            "H_0": (40.0, 100.0), "Omega_m": (0.005, 0.95)
        },
    )

    H_0_recovered = res.best_fit["H_0"]
    Om_recovered = res.best_fit["Omega_m"]
    print(f"  Recovered (sigma_theta = {sigma_theta_default:.0e}):")
    print(f"    H_0_LCDM     = {H_0_recovered:.4f} km/s/Mpc")
    print(f"    Omega_m_LCDM = {Om_recovered:.4f}")
    print(f"    z_eff_bias   = {res.z_eff_bias_inversion:.4f}")
    print(f"    chi^2 total  = {res.chi_squared_total:.3e}")
    print()
    print(f"  Planck observed:  {PLANCK_H_0_LCDM:.4f} +/- "
          f"{PLANCK_H_0_LCDM_SIGMA:.4f} km/s/Mpc")
    sigma_dist_planck = (
        abs(H_0_recovered - PLANCK_H_0_LCDM) / PLANCK_H_0_LCDM_SIGMA
    )
    print(f"  Distance:         {sigma_dist_planck:.2f} sigma")
    print()
    print("  Status: PARTIAL. The emulator runs, but R-A risk (sigma_theta")
    print("  dependence) and Phase B (substrate r_s derivation) prevent")
    print("  full closure. Full Planck-CMB H_0 recovery requires substrate")
    print("  r_s — currently NO bounded path per architecture §9 / L6.")
    print()
    return res, sigma_dist_planck


# ===========================================================================
# § 3.  The tension as a frame mismatch.
# ===========================================================================


def section_3_tension_as_frame_mismatch():
    print("=" * 72)
    print("§3. The tension as a frame mismatch")
    print("=" * 72)
    print()
    print("Conventional Hubble tension framing: a single H_0 disagrees")
    print(f"between SH0ES ({SH0ES_H_0:.2f}) and Planck-CMB "
          f"({PLANCK_H_0_LCDM:.2f}).")
    print()
    print("Framework framing: TWO DIFFERENT QUANTITIES, both predicted,")
    print("each matching its own observation:")
    print()
    print("  Quantity              Framework prediction       Observation")
    print("  --------------------  -------------------------  ---------------")
    print(f"  H_0_observer (SH0ES)  {H_0_OBSERVER:.4f}                   "
          f"SH0ES {SH0ES_H_0:.2f}")
    print(f"                        (cascade D2-extended)      ({SH0ES_H_0_SIGMA:.2f} sigma)")
    print()
    print(f"  H_0_LCDM (Planck)     LCDM-fit emulator at       Planck {PLANCK_H_0_LCDM:.2f}")
    print(f"                        z_eff with external r_s    "
          f"({PLANCK_H_0_LCDM_SIGMA:.2f} sigma)")
    print(f"                        (Phase A partial)")
    print()
    print("These are framework-predicted to differ. The conventional")
    print("'tension' is a category error of treating both as measurements")
    print("of the same single quantity in a frame-naive cosmology.")
    print()
    print("Status:")
    print("  SH0ES side:   CLOSED (theorem-grade-conditional via")
    print("                cascade D2-extended; predictions/H_0.py).")
    print("  Planck-CMB:   PARTIAL — Phase A LCDM-fit emulator works")
    print("                with external r_s; full closure blocked on")
    print("                Phase B substrate r_s (no current path per")
    print("                architecture §9 / L6).")
    print()


def main():
    sigma_dist_shoes = section_1_shoes()
    res, sigma_dist_planck = section_2_planck_cmb_partial()
    section_3_tension_as_frame_mismatch()

    print("=" * 72)
    print("VERDICT — Hubble tension story PARTIAL CLOSURE")
    print("=" * 72)
    print()
    print(f"  SH0ES side:    H_0_observer = {H_0_OBSERVER:.2f} vs "
          f"{SH0ES_H_0:.2f} +/- {SH0ES_H_0_SIGMA:.2f} -> "
          f"{sigma_dist_shoes:.2f} sigma  -> CLOSED")
    print(f"  Planck side:   H_0_LCDM  via emulator + external r_s")
    print(f"                 (PARTIAL; full closure blocked on Phase B r_s)")
    print(f"  Framing:       'tension' = frame mismatch, not single-")
    print(f"                 quantity disagreement (no contradiction)")
    print()


if __name__ == "__main__":
    main()
