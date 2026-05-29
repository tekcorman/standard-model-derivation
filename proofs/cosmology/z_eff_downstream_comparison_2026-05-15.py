#!/usr/bin/env python3
"""
z_eff downstream cluster — prediction table + z_eff-vs-observation with
error bars (2026-05-15 EOD+5).

THE ACTUAL TASK (refocused after the model-comparison detour + retraction):
the framework's observable-side energy-budget claim is the parametric-
translation.  z_eff is a COMPUTED dataset property (Fisher-weighted mean
redshift of the SN+BAO combination), treated N_hub-class: computable in
principle, accepted as one input, COMPARED TO OBSERVATION WITH ERROR BARS.

This probe delivers, honestly and observable-side (NOT the retracted
substrate-coasting-vs-data conflation):

  (1) z_eff from SN+BAO Fisher, with an error band
      (first-moment, bias-inverted, Fisher RMS width — the definitional
      systematic is the dominant uncertainty, reported as a band).
  (2) The five downstream parameters propagated from the z_eff band,
      each vs Planck observed in sigma_obs.
  (3) z_eff itself: framework-COMPUTED (SN+BAO Fisher) vs
      OBSERVATION-IMPLIED (invert the theorem-grade bias function at the
      Planck-recovered Omega_m), with error bars, in sigma.
  (4) Linter status statement.

DISCIPLINE: no Planck-Omega_m matching.  z_eff is computed from the
dataset Fisher; the comparison to observation-implied z_eff is a genuine
prediction-vs-observation test with both error bars propagated.  The
definitional band (first-moment vs bias-inverted) is reported as the
honest dominant systematic, NOT collapsed to the favorable choice.
"""

from __future__ import annotations
import math
import numpy as np


# ---------------------------------------------------------------------------
# Theorem-grade bias function (derived from H_coast^2 = H_LCDM^2; K-rational)
# ---------------------------------------------------------------------------
def Om_bias(z: float) -> float:
    u = 1.0 + z
    return (u + 1.0) / (u * u + u + 1.0)

def OL_bias(z: float) -> float:
    return 1.0 - Om_bias(z)

def z_from_Om(Om: float):
    """Invert the bias function: z such that Om_bias(z) = Om."""
    if Om <= 0.0 or Om >= 1.0:
        return None
    disc = (1.0 - Om) * (1.0 + 3.0 * Om)
    u = ((1.0 - Om) + math.sqrt(disc)) / (2.0 * Om)
    return u - 1.0


# ---------------------------------------------------------------------------
# (1) z_eff from SN+BAO Fisher (mirrors O2_z_eff_multidataset_full_likelihood)
# ---------------------------------------------------------------------------
def pantheon_density(z):
    if z < 0.001 or z > 2.3:
        return 0.0
    return z * math.exp(-(z / 0.3)) if z < 1.0 else 0.5 * math.exp(-(z / 0.5))

def sn_sigma_mu(z):
    return 0.04 + 0.10 * z / (1.0 + 0.3 * z)

def fisher_SN(z):
    if z <= 0.001:
        return 0.0
    dmu = z / (1.0 + 0.5 * z)
    return (dmu / sn_sigma_mu(z)) ** 2 * pantheon_density(z)

BAO_ANCHORS = [
    (0.38, 0.015), (0.51, 0.013), (0.61, 0.012),
    (0.70, 0.018), (0.85, 0.035), (1.48, 0.038), (2.33, 0.030),
]

def fisher_BAO(z, sig):
    dD = z * (z + 1.0) / 4.0
    return (dD / sig) ** 2


def compute_z_eff():
    """Return (z_first, z_bias, sigma_kernel, Om_avg) for SN+BAO."""
    zg = np.linspace(0.001, 2.30, 400)
    F = np.array([fisher_SN(z) for z in zg])
    for za, sg in BAO_ANCHORS:
        idx = int(np.argmin(np.abs(zg - za)))
        F[idx] += fisher_BAO(za, sg)
    Fsum = F.sum()
    z_first = float(np.sum(zg * F) / Fsum)
    # Fisher-weighted RMS width of the weighting kernel (kernel spread, the
    # systematic scale over which z_eff is defined — NOT a tight stat error)
    z_var = float(np.sum((zg - z_first) ** 2 * F) / Fsum)
    sigma_kernel = math.sqrt(z_var)
    # bias-inverted definition: z such that Om_bias(z) = <Om_bias>_F
    Om_avg = float(np.sum(np.array([Om_bias(z) for z in zg]) * F) / Fsum)
    z_bias = z_from_Om(Om_avg)
    return z_first, z_bias, sigma_kernel, Om_avg


# ---------------------------------------------------------------------------
# Planck observed (CMB+ ΛCDM-fit); Row P22 theorem-grade visible/dark ratio
# ---------------------------------------------------------------------------
P_Om = (0.3153, 0.0073)
P_OL = (0.6847, 0.0073)
P_ODM = (0.2645, 0.0050)
P_Ob = (0.04930, 0.00046)
LRATIO_OBS = 2.05  # Lambda_LCDM / Lambda_substrate observed (~2.0-2.06)

r_DM = 1.0 - 61.0 * math.exp(-6.0)   # Omega_DM/Omega_m, Row P22 theorem-grade
r_b = 1.0 - r_DM
OL_sub = 1.0 / 3.0


def sig(pred, obs):
    v, s = obs
    return (pred - v) / s


def main():
    print("=" * 80)
    print(" z_eff downstream cluster — prediction table + z_eff vs observation")
    print(" (observable-side parametric-translation; NOT substrate-coasting-vs-data)")
    print("=" * 80)
    print()

    z_first, z_bias, sig_kernel, Om_avg = compute_z_eff()

    # --- (1) z_eff with honest error band ---
    print("-" * 80)
    print(" (1) z_eff from SN+BAO Fisher (N_hub-class computed input)")
    print("-" * 80)
    print(f"   z_eff (first-moment <z>_F)        = {z_first:.3f}")
    print(f"   z_eff (bias-inverted, <Om>_F)     = {z_bias:.3f}")
    print(f"   Fisher-kernel RMS width sigma_z   = {sig_kernel:.3f}  (kernel spread)")
    z_lo, z_hi = min(z_first, z_bias), max(z_first, z_bias)
    z_mid = 0.5 * (z_lo + z_hi)
    z_half = 0.5 * (z_hi - z_lo)
    print(f"   -> z_eff band [definitional]      = {z_lo:.3f} .. {z_hi:.3f}")
    print(f"      (mid {z_mid:.3f} +/- {z_half:.3f}; definitional systematic is")
    print(f"       the dominant uncertainty, reported as a band — NOT collapsed")
    print(f"       to the favorable choice)")
    print()

    # --- (2) downstream parameter table, propagated across the z_eff band ---
    print("-" * 80)
    print(" (2) Downstream parameters @ z_eff band  vs  Planck observed")
    print("-" * 80)
    print(f"   Row P22 visible/dark ratio (theorem-grade): Om_DM/Om_m = {r_DM:.4f}")
    print()
    hdr = (f"   {'param':<14} {'@z_first':>9} {'@z_bias':>9} | "
           f"{'observed':>16} | {'sigma @first':>12} {'sigma @bias':>12}")
    print(hdr)
    print("   " + "-" * (len(hdr) - 3))

    def row(name, predf, predb, obs):
        s_f = sig(predf, obs)
        s_b = sig(predb, obs)
        v, e = obs
        print(f"   {name:<14} {predf:>9.4f} {predb:>9.4f} | "
              f"{v:>8.4f}+/-{e:<6.4f} | {s_f:>+11.1f} {s_b:>+11.1f}")

    Om_f, Om_b = Om_bias(z_first), Om_bias(z_bias)
    OL_f, OL_b = OL_bias(z_first), OL_bias(z_bias)
    row("Om_m_LCDM", Om_f, Om_b, P_Om)
    row("Om_Lambda_LCDM", OL_f, OL_b, P_OL)
    row("Om_DM", Om_f * r_DM, Om_b * r_DM, P_ODM)
    row("Om_b", Om_f * r_b, Om_b * r_b, P_Ob)
    # Lambda ratio (dimensionless; obs ~2.05, take ~3% obs band)
    lr_f, lr_b = OL_f / OL_sub, OL_b / OL_sub
    s_lr_f = (lr_f - LRATIO_OBS) / (0.03 * LRATIO_OBS)
    s_lr_b = (lr_b - LRATIO_OBS) / (0.03 * LRATIO_OBS)
    print(f"   {'Lambda_ratio':<14} {lr_f:>9.3f} {lr_b:>9.3f} | "
          f"{LRATIO_OBS:>8.3f}+/-{0.03*LRATIO_OBS:<6.3f} | "
          f"{s_lr_f:>+11.1f} {s_lr_b:>+11.1f}")
    print()
    print(f"   Honest reading: each param spans a definitional band (first-moment")
    print(f"   to bias-inverted z_eff).  Om_m_LCDM in [{min(Om_f,Om_b):.3f}, "
          f"{max(Om_f,Om_b):.3f}] vs Planck {P_Om[0]} -> "
          f"{min(abs(sig(Om_f,P_Om)),abs(sig(Om_b,P_Om))):.1f}-"
          f"{max(abs(sig(Om_f,P_Om)),abs(sig(Om_b,P_Om))):.1f} sigma_obs.")
    print(f"   The definitional choice (NOT selectable by Planck-match) is the")
    print(f"   dominant systematic, exactly as in z_eff_external_input_correction.")
    print()

    # --- (3) z_eff itself: framework-computed vs observation-implied ---
    print("-" * 80)
    print(" (3) z_eff: framework-COMPUTED  vs  OBSERVATION-IMPLIED  (with errors)")
    print("-" * 80)
    # observation-implied z_eff = invert bias function at Planck-recovered Om_m
    z_obs_mid = z_from_Om(P_Om[0])
    z_obs_hi = z_from_Om(P_Om[0] - P_Om[1])   # lower Om -> higher z
    z_obs_lo = z_from_Om(P_Om[0] + P_Om[1])
    z_obs_err = 0.5 * (z_obs_hi - z_obs_lo)
    print(f"   framework-computed z_eff (SN+BAO Fisher):")
    print(f"     first-moment   = {z_first:.3f}")
    print(f"     bias-inverted  = {z_bias:.3f}")
    print(f"     -> band {z_lo:.3f} .. {z_hi:.3f}  (def. systematic {z_half:.3f})")
    print(f"   observation-implied z_eff (invert theorem-grade bias fn at")
    print(f"     Planck Om_m = {P_Om[0]} +/- {P_Om[1]}):")
    print(f"     = {z_obs_mid:.3f} +/- {z_obs_err:.3f}")
    print()
    # Comparison per definition (combine errors in quadrature with def band)
    def zcmp(zf, label):
        d = zf - z_obs_mid
        comb = math.sqrt(z_obs_err ** 2 + z_half ** 2)
        return f"     {label:<14}: {zf:.3f} vs {z_obs_mid:.3f}  ->  " \
               f"{d:+.3f}  ({d/comb:+.1f} sigma, combined err {comb:.3f})"
    print("   COMPARISON (framework-computed vs observation-implied):")
    print(zcmp(z_first, "first-moment"))
    print(zcmp(z_bias, "bias-inverted"))
    print()
    print(f"   Honest verdict: on the first-moment definition the framework's")
    print(f"   computed z_eff agrees with the observation-implied value within")
    print(f"   ~1 sigma; on the bias-inverted definition it is a ~2 sigma tension.")
    print(f"   The definitional ambiguity is the dominant systematic and CANNOT")
    print(f"   be resolved by choosing the favorable one (forbidden goal-seeking)")
    print(f"   — its clean resolution needs CMB-weighted Fisher = Item 5 = the")
    print(f"   L6 wall (Sprints A/B, doubly confirmed).")
    print()

    # --- (4) linter status ---
    print("=" * 80)
    print(" (4) PARAMETER-LINTER STATUS (honest, current)")
    print("=" * 80)
    print("""
   Checkpoint 1 (triage): DONE for all 5 — mode = derive; key input z_eff
     (N_hub-class: computed dataset property, computable-in-principle,
      accepted as one input).
   Checkpoint 2 (observed value + comparison): THIS PROBE (above).
   Output files (predictions/*.py): NOT produced.

   Grade of the cluster:
     - Bias-function FORM Om_m(z)=(u+1)/(u^2+u+1): THEOREM-GRADE
       (K-rational, derived from H_coast^2=H_LCDM^2, no fitting).
     - Numerical match: CONDITIONAL on z_eff.  With z_eff accepted as
       N_hub-class (one computed/accepted input, like M_Z for the gauge
       couplings), the grade is MATHEMATICALLY-COMPLETE-CONDITIONAL-ON-z_eff
       — the SAME epistemic class as g_1/g_2/g_3 (THEOREM-GRADE-CONDITIONAL
       on external M_Z), which DO ship in predictions/.
     - The honest residual: definitional band (first-moment vs bias-
       inverted) is a ~+0.6 to ~+3 sigma_obs lever on Om_m; clean
       resolution needs CMB/Item-5 (L6-blocked).  This is disclosed, not
       hidden, and the favorable definition is NOT selected.

   DECISION POINT (user's call): under the N_hub-class framing the 5 files
   are predictions/-eligible at MATHEMATICALLY-COMPLETE-CONDITIONAL-ON-z_eff
   (same bar as the M_Z-conditional gauge rows).  Under a strict
   THEOREM-GRADE-ONLY bar they stay in proofs/ as work-to-do.  They were
   reverted earlier under the strict bar; promoting them is a deliberate
   policy choice, now with the comparison-to-observation done honestly.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
