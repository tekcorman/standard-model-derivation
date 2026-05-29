#!/usr/bin/env python3
"""
proofs/cosmology/Lambda_CC_path_A_session1_coasting_lcdm_fit.py

Λ_CC PATH A — SESSION 1: ΛCDM-pipeline mis-extraction test on mock
coasting SN1a data.

Setup
-----
The Λ_CC factor-of-2 decomposition (`Lambda_CC_factor_two_decomposition_
2026-05-05.md`) hypothesizes that ΛCDM observational fitting of TRUE-
COASTING cosmology data mis-extracts (Ω_m, Ω_Λ) by factor of 2:

  framework substrate (true):  (Ω_m, Ω_Λ) = (2/3, 1/3)
  ΛCDM fit (predicted):        (Ω_m, Ω_Λ) = (1/3, 2/3)

Empirical Planck:                (0.315, 0.685)  matches predicted (1/3, 2/3)
                                 at percent precision (1.4% / 2.8%).

The ⇐⇒ test of this hypothesis (Session 1 of Path A): generate mock
SN1a Hubble-diagram data under TRUE-COASTING with framework's H_0
(observer-side H_0 = 72.74 = (16/15)·68.19, since SN1a Hubble flow is
observer-side per cascade D2-extended), then fit it with the standard
ΛCDM pipeline (free H_0, Ω_m, flat universe).

If recovered (Ω_m, Ω_Λ) ≈ (1/3, 2/3) → factor-of-2 confirmed as
ΛCDM-pipeline mis-extraction. P24 closes; Row P23 closes by inheritance.

If recovered values differ from (1/3, 2/3) → factor-of-2 hypothesis
needs reframing or fails this test.

What this session does NOT do
-----------------------------
- Use real Pantheon+ likelihoods (does use realistic Pantheon+ z-distribution
  + per-bin σ_μ ≈ 0.04-0.08 mag floor)
- Include CMB acoustic scale or BAO (sessions 2-3)
- MCMC; uses scipy.optimize.curve_fit for best-fit + covariance
- Test substrate-frame H_0 (sessions 2+); we use observer-frame here
  because SH0ES distance-ladder is observer-side per cascade D2-extended

Three sub-tests this session
----------------------------
1. ΛCDM (free H_0, Ω_m; flat) — does it recover Ω_m ≈ 1/3?
2. wCDM (free H_0, Ω_m, w; flat) — what w does it prefer? (DESI 2024
   reports w_0 ≈ -0.83; consistent with framework's Λ ∝ 1/t² being
   misfit as time-varying w?)
3. Cosmographic q_0 expansion (free H_0, q_0; series to z²) — does it
   recover q_0 ≈ -0.55 (ΛCDM value) from coasting data?

Selection grammar discipline: every fit is named explicitly as a
channel_select step (different fitter parameterization picks a different
observational channel; coasting vs ΛCDM vs wCDM are all live channels
above the A2-T waterline, and observation distinguishes the K-candidate
that best matches the data).
"""

import sys
import os
import math

import numpy as np
from scipy import integrate, optimize

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# =============================================================================
# §0. Framework predictions
# =============================================================================
H0_SUBSTRATE = 68.19            # km/s/Mpc, framework cascade H = 1/(N t_P)
H0_OBSERVER = (16.0 / 15.0) * H0_SUBSTRATE  # = 72.74, D2-extended
OMEGA_M_SUBSTRATE = 2.0 / 3.0   # G1a anisotropic eigenspace
OMEGA_LAMBDA_SUBSTRATE = 1.0 / 3.0  # G1a isotropic eigenspace

# Factor-of-2 prediction for ΛCDM-fit recovery
OMEGA_M_FACTOR_TWO_PREDICTED = 1.0 / 3.0     # = (1/2) × Ω_m_substrate
OMEGA_LAMBDA_FACTOR_TWO_PREDICTED = 2.0 / 3.0

# Planck 2018 (TT,TE,EE+lowE+lensing baseline)
PLANCK_OMEGA_M = 0.315
PLANCK_OMEGA_M_SIGMA = 0.007
PLANCK_OMEGA_LAMBDA = 0.685
PLANCK_OMEGA_LAMBDA_SIGMA = 0.007
PLANCK_H0 = 67.4
PLANCK_H0_SIGMA = 0.5
SHOES_H0 = 73.04
SHOES_H0_SIGMA = 1.04

c_km_s = 2.99792458e5

print("=" * 78)
print("Λ_CC PATH A — SESSION 1: ΛCDM-fit mis-extraction test on coasting mock")
print("=" * 78)
print(f"  Framework H_0 (substrate, cascade): {H0_SUBSTRATE:.4f} km/s/Mpc")
print(f"  Framework H_0 (observer, ×16/15):   {H0_OBSERVER:.4f} km/s/Mpc")
print(f"  Substrate (Ω_m, Ω_Λ) = (2/3, 1/3)")
print()
print(f"  Factor-of-2 predicted ΛCDM-fit: (Ω_m, Ω_Λ) ≈ (1/3, 2/3)")
print(f"  Planck observed:                (Ω_m, Ω_Λ) = ({PLANCK_OMEGA_M:.3f}, {PLANCK_OMEGA_LAMBDA:.3f})")
print()


# =============================================================================
# §1. Distance modulus formulae
# =============================================================================
def mu_coasting(z, H0):
    """μ(z) = 5 log10[(c/H0)(1+z) ln(1+z)] + 25  for coasting a ∝ t."""
    if np.isscalar(z):
        z_arr = np.array([z], dtype=float)
    else:
        z_arr = np.asarray(z, dtype=float)
    dL = (c_km_s / H0) * (1.0 + z_arr) * np.log1p(z_arr)
    mu = 5.0 * np.log10(dL) + 25.0
    return mu[0] if np.isscalar(z) else mu


def mu_lcdm(z, H0, Om):
    """μ(z) for flat ΛCDM with Ω_m = Om, Ω_Λ = 1 - Om."""
    OL = 1.0 - Om
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    out = np.zeros_like(z_arr)
    for i, zi in enumerate(z_arr):
        if zi <= 1e-9:
            out[i] = 0.0
            continue
        E = lambda zp: 1.0 / math.sqrt(Om * (1 + zp) ** 3 + OL)
        chi, _ = integrate.quad(E, 0.0, zi, limit=200)
        dL = (c_km_s / H0) * (1.0 + zi) * chi
        out[i] = 5.0 * math.log10(dL) + 25.0
    return out[0] if np.isscalar(z) else out


def mu_wcdm(z, H0, Om, w):
    """μ(z) for flat wCDM with constant w, Ω_DE = 1 - Om."""
    ODE = 1.0 - Om
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    out = np.zeros_like(z_arr)
    for i, zi in enumerate(z_arr):
        if zi <= 1e-9:
            out[i] = 0.0
            continue
        E = lambda zp: 1.0 / math.sqrt(Om * (1 + zp) ** 3 + ODE * (1 + zp) ** (3 * (1 + w)))
        chi, _ = integrate.quad(E, 0.0, zi, limit=200)
        dL = (c_km_s / H0) * (1.0 + zi) * chi
        out[i] = 5.0 * math.log10(dL) + 25.0
    return out[0] if np.isscalar(z) else out


def mu_cosmographic_q0(z, H0, q0):
    """μ(z) from cosmographic Hubble expansion to second order in z:
       d_L ≈ (c/H_0) [z + (1/2)(1 − q_0) z² + ...]
       Use this for the q_0 sub-test only (z < 0.3 regime)."""
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    out = np.zeros_like(z_arr)
    for i, zi in enumerate(z_arr):
        if zi <= 1e-9:
            out[i] = 0.0
            continue
        # Series: d_L = (c/H_0)·z·[1 + (1/2)(1-q_0)·z + O(z²)]
        dL = (c_km_s / H0) * zi * (1.0 + 0.5 * (1.0 - q0) * zi)
        out[i] = 5.0 * math.log10(dL) + 25.0
    return out[0] if np.isscalar(z) else out


# =============================================================================
# §2. Generate Pantheon+-like mock SN1a data under coasting truth
# =============================================================================
print("§1. Generate Pantheon+-like mock under TRUE-COASTING")
print("-" * 78)

# Pantheon+ z-distribution: ~1701 SNe across z = 0.001-2.26.
# Approximate density: peaks around z = 0.1-0.3, falls off at z > 0.6.
# Mock: 50 z-values logarithmically spaced 0.01-2.0 (representative bins).
np.random.seed(2026_05_05)
N_BINS = 50
z_min, z_max = 0.01, 2.0
z_data = np.exp(np.linspace(math.log(z_min), math.log(z_max), N_BINS))

# Realistic per-bin σ_μ: 0.04 mag at low z (statistical) up to 0.15 mag at z>1
# (low SN density); roughly σ_μ(z) ≈ 0.04 + 0.06·z is a reasonable approximation
sigma_mu = 0.04 + 0.06 * z_data

# True signal under coasting at observer-frame H_0
mu_true_obs = mu_coasting(z_data, H0_OBSERVER)
# Add Gaussian noise
mu_obs = mu_true_obs + np.random.normal(0, sigma_mu)

print(f"  z range: [{z_min}, {z_max}]   N_bins = {N_BINS}")
print(f"  σ_μ(z=0.01) = {sigma_mu[0]:.3f} mag")
print(f"  σ_μ(z=2.0)  = {sigma_mu[-1]:.3f} mag")
print(f"  TRUTH cosmology:  coasting at H_0_observer = {H0_OBSERVER:.4f} km/s/Mpc")
print(f"                    (SN1a is observer-side per cascade D2-extended)")
print()


# =============================================================================
# §3. Sub-test 1: ΛCDM fit (free H_0, Ω_m, flat)
# =============================================================================
print("§3. Sub-test 1: flat ΛCDM fit (free H_0, Ω_m)")
print("-" * 78)

def lcdm_residuals(params, z, mu_obs, sigma_mu):
    H0, Om = params
    if Om < 0 or Om > 1 or H0 < 30 or H0 > 120:
        return np.full_like(mu_obs, 1e10)
    mu_pred = mu_lcdm(z, H0, Om)
    return (mu_obs - mu_pred) / sigma_mu

p0 = [70.0, 0.3]
result_lcdm = optimize.least_squares(
    lcdm_residuals, p0, args=(z_data, mu_obs, sigma_mu),
    bounds=([30, 1e-4], [120, 0.9999]),
)
H0_lcdm, Om_lcdm = result_lcdm.x
chi2_lcdm = np.sum(result_lcdm.fun ** 2)
dof_lcdm = N_BINS - 2

# Estimate uncertainties from Jacobian
J = result_lcdm.jac
cov = np.linalg.inv(J.T @ J)
H0_lcdm_sigma = math.sqrt(cov[0, 0])
Om_lcdm_sigma = math.sqrt(cov[1, 1])

print(f"  Best-fit H_0     = {H0_lcdm:.4f} ± {H0_lcdm_sigma:.4f} km/s/Mpc")
print(f"  Best-fit Ω_m     = {Om_lcdm:.4f} ± {Om_lcdm_sigma:.4f}")
print(f"  Best-fit Ω_Λ     = {1-Om_lcdm:.4f} ± {Om_lcdm_sigma:.4f}")
print(f"  χ²/dof           = {chi2_lcdm:.3f}/{dof_lcdm} = {chi2_lcdm/dof_lcdm:.3f}")
print()
print(f"  Factor-of-2 prediction: Ω_m → 1/3 = {OMEGA_M_FACTOR_TWO_PREDICTED:.4f}")
print(f"  Δ(Ω_m) recovered − predicted = {Om_lcdm - OMEGA_M_FACTOR_TWO_PREDICTED:+.4f}  ({(Om_lcdm - OMEGA_M_FACTOR_TWO_PREDICTED)/Om_lcdm_sigma:+.2f}σ_fit)")
print(f"  vs Planck observed Ω_m = {PLANCK_OMEGA_M:.4f}")
print(f"  Δ(Ω_m) recovered − Planck    = {Om_lcdm - PLANCK_OMEGA_M:+.4f}  ({(Om_lcdm - PLANCK_OMEGA_M)/PLANCK_OMEGA_M_SIGMA:+.2f}σ_obs)")
print()


# =============================================================================
# §4. Sub-test 2: wCDM fit (free H_0, Ω_m, w; flat)
# =============================================================================
print("§4. Sub-test 2: flat wCDM fit (free H_0, Ω_m, w)")
print("-" * 78)

def wcdm_residuals(params, z, mu_obs, sigma_mu):
    H0, Om, w = params
    if Om < 0 or Om > 1 or H0 < 30 or H0 > 120 or w < -3 or w > 0:
        return np.full_like(mu_obs, 1e10)
    mu_pred = mu_wcdm(z, H0, Om, w)
    return (mu_obs - mu_pred) / sigma_mu

p0 = [70.0, 0.3, -1.0]
result_wcdm = optimize.least_squares(
    wcdm_residuals, p0, args=(z_data, mu_obs, sigma_mu),
    bounds=([30, 1e-4, -3], [120, 0.9999, 0]),
)
H0_w, Om_w, w_w = result_wcdm.x
chi2_w = np.sum(result_wcdm.fun ** 2)
dof_w = N_BINS - 3

J_w = result_wcdm.jac
cov_w = np.linalg.inv(J_w.T @ J_w)
H0_w_sigma = math.sqrt(cov_w[0, 0])
Om_w_sigma = math.sqrt(cov_w[1, 1])
w_w_sigma = math.sqrt(cov_w[2, 2])

print(f"  Best-fit H_0     = {H0_w:.4f} ± {H0_w_sigma:.4f} km/s/Mpc")
print(f"  Best-fit Ω_m     = {Om_w:.4f} ± {Om_w_sigma:.4f}")
print(f"  Best-fit w       = {w_w:.4f} ± {w_w_sigma:.4f}")
print(f"  χ²/dof           = {chi2_w:.3f}/{dof_w} = {chi2_w/dof_w:.3f}")
print()
print(f"  Coasting (TRUE) has w_eff = -1/3 (since a ∝ t).")
print(f"  ΛCDM assumes w = -1.")
print(f"  DESI 2024 reports w_0 = -0.83 ± 0.06 (consistent with time-varying Λ).")
print(f"  Δw recovered − ΛCDM = {w_w + 1:+.4f}  ({(w_w + 1)/w_w_sigma:+.2f}σ_fit)")
print(f"  Δw recovered − coasting (-1/3) = {w_w + 1.0/3:+.4f}")
print()


# =============================================================================
# §5. Sub-test 3: cosmographic q_0 fit (low-z only, z < 0.3)
# =============================================================================
print("§5. Sub-test 3: cosmographic q_0 fit (low-z, z<0.3)")
print("-" * 78)

mask_low = z_data < 0.3
z_low = z_data[mask_low]
mu_low = mu_obs[mask_low]
sigma_low = sigma_mu[mask_low]
print(f"  Low-z subset: {len(z_low)} bins, z = {z_low[0]:.3f} … {z_low[-1]:.3f}")

def q0_residuals(params, z, mu_obs, sigma_mu):
    H0, q0 = params
    if H0 < 30 or H0 > 120 or q0 < -2 or q0 > 2:
        return np.full_like(mu_obs, 1e10)
    mu_pred = mu_cosmographic_q0(z, H0, q0)
    return (mu_obs - mu_pred) / sigma_mu

p0 = [70.0, -0.5]
result_q0 = optimize.least_squares(
    q0_residuals, p0, args=(z_low, mu_low, sigma_low),
    bounds=([30, -2], [120, 2]),
)
H0_q, q0_q = result_q0.x
chi2_q = np.sum(result_q0.fun ** 2)
dof_q = len(z_low) - 2

J_q = result_q0.jac
cov_q = np.linalg.inv(J_q.T @ J_q)
H0_q_sigma = math.sqrt(cov_q[0, 0])
q0_q_sigma = math.sqrt(cov_q[1, 1])

# Coasting (a ∝ t) has q_0 = 0 by definition (ä = 0).
# ΛCDM (Ω_m=0.315) has q_0 = (Ω_m/2) - Ω_Λ = 0.158 - 0.685 = -0.528.
# Factor-of-2 prediction: ΛCDM-fit-of-coasting recovers q_0 ≈ -0.5 (the
# fictitious "acceleration" produced by the mis-fit).
q0_lcdm_value = 0.5 * PLANCK_OMEGA_M - PLANCK_OMEGA_LAMBDA
print(f"  Best-fit H_0      = {H0_q:.4f} ± {H0_q_sigma:.4f} km/s/Mpc")
print(f"  Best-fit q_0      = {q0_q:.4f} ± {q0_q_sigma:.4f}")
print(f"  χ²/dof            = {chi2_q:.3f}/{dof_q} = {chi2_q/dof_q:.3f}")
print()
print(f"  Coasting TRUE q_0 = 0 (ä = 0).")
print(f"  ΛCDM (Planck Ω_m=0.315): q_0 = Ω_m/2 - Ω_Λ = {q0_lcdm_value:.3f}.")
print(f"  Δq_0 recovered − coasting truth = {q0_q:+.4f}  ({q0_q/q0_q_sigma:+.2f}σ_fit)")
print(f"  Δq_0 recovered − ΛCDM Planck    = {q0_q - q0_lcdm_value:+.4f}")
print()


# =============================================================================
# §6. Verdict
# =============================================================================
print("§6. Verdict — does ΛCDM-fitting of coasting mock recover (1/3, 2/3)?")
print("=" * 78)

# The acid test: how far is recovered Ω_m from the factor-of-2 prediction
# and from Planck observation?
delta_factor_two = abs(Om_lcdm - OMEGA_M_FACTOR_TWO_PREDICTED) / Om_lcdm_sigma
delta_planck = abs(Om_lcdm - PLANCK_OMEGA_M) / PLANCK_OMEGA_M_SIGMA

# Determine verdict based on whether recovered Ω_m matches predicted 1/3
within_pred = abs(Om_lcdm - OMEGA_M_FACTOR_TWO_PREDICTED) < 2 * Om_lcdm_sigma
within_planck = abs(Om_lcdm - PLANCK_OMEGA_M) < 2 * PLANCK_OMEGA_M_SIGMA

print(f"  Test 1 (ΛCDM fit):      Ω_m recovered = {Om_lcdm:.4f} ± {Om_lcdm_sigma:.4f}")
print(f"                          Predicted 1/3 = {OMEGA_M_FACTOR_TWO_PREDICTED:.4f}    "
      f"Δ/σ_fit = {(Om_lcdm - OMEGA_M_FACTOR_TWO_PREDICTED)/Om_lcdm_sigma:+.2f}σ")
print(f"                          Planck obs    = {PLANCK_OMEGA_M:.4f}    "
      f"Δ/σ_obs = {(Om_lcdm - PLANCK_OMEGA_M)/PLANCK_OMEGA_M_SIGMA:+.2f}σ")
print()
print(f"  Test 2 (wCDM fit):      w recovered = {w_w:.4f} ± {w_w_sigma:.4f}")
print(f"                          ΛCDM w = -1; coasting truth w_eff = -1/3.")
print(f"                          Recovered w deviation from -1: {(w_w + 1)/w_w_sigma:+.2f}σ_fit")
print()
print(f"  Test 3 (q_0 cosmography): q_0 recovered = {q0_q:.4f} ± {q0_q_sigma:.4f}")
print(f"                          Coasting truth q_0 = 0.")
print(f"                          ΛCDM Planck q_0 = {q0_lcdm_value:.3f}.")
print(f"                          Recovered q_0 deviation from 0: {q0_q/q0_q_sigma:+.2f}σ_fit")
print()

if within_pred and within_planck:
    verdict = "FACTOR-OF-2 CONFIRMED at <2σ"
    closure_status = "Path A SUCCEEDS Session 1; proceed to CMB+BAO sessions."
elif delta_factor_two < delta_planck:
    verdict = "PARTIAL — recovered Ω_m closer to predicted 1/3 than to Planck"
    closure_status = "Mixed; needs CMB+BAO to discriminate."
else:
    verdict = "FACTOR-OF-2 NOT confirmed by this test"
    closure_status = ("Recovered Ω_m differs from 1/3 prediction; coasting mock "
                      "doesn't fit ΛCDM at the (Ω_m → 1/3) value the factor-of-2 "
                      "decomposition predicts.")

print(f"  VERDICT: {verdict}")
print(f"  CLOSURE: {closure_status}")
print()


# =============================================================================
# §7. Diagnostic: residuals
# =============================================================================
print("§7. Diagnostic — fit residuals at sample z-values")
print("-" * 78)
mu_lcdm_pred = mu_lcdm(z_data, H0_lcdm, Om_lcdm)
residuals_lcdm = mu_obs - mu_lcdm_pred
print(f"  ΛCDM-fit residuals (mu_obs - mu_pred):")
print(f"    {'z':>8} {'mu_obs':>10} {'mu_pred':>10} {'Δμ':>10} {'Δμ/σ':>10}")
sample_idxs = [0, 5, 15, 25, 35, 45, N_BINS-1]
for i in sample_idxs:
    print(f"    {z_data[i]:>8.4f} {mu_obs[i]:>10.4f} {mu_lcdm_pred[i]:>10.4f} "
          f"{residuals_lcdm[i]:>10.4f} {residuals_lcdm[i]/sigma_mu[i]:>10.2f}")
print()
print(f"  RMS of |Δμ| = {np.sqrt(np.mean(residuals_lcdm**2)):.4f} mag")
print(f"  RMS of |Δμ|/σ = {np.sqrt(np.mean((residuals_lcdm/sigma_mu)**2)):.4f}")
print()


# =============================================================================
# §8. Selection grammar disclosure
# =============================================================================
print("§8. Selection grammar disclosure")
print("-" * 78)
print("""
  Each fit (§3, §4, §5) is a channel_select step: different fitter
  parameterizations correspond to different observational channels
  (ΛCDM, wCDM, q_0-cosmographic). The mock data is a single observed
  signal; the K-candidate channels are above the A2-T waterline (each
  is a physically realizable cosmography), and observation distinguishes
  them by χ²/dof + best-fit-parameter recovery.

  Per `theorem_lattice_coupling_general.md` §2 grammar, this is NOT
  canonical_encoding (the channels do NOT evaluate to the same numerical
  prediction; they fit different (Ω_m, Ω_Λ, w, q_0) values). It IS
  channel_select.

  Path A's structural argument is: the framework's TRUE cosmology lives
  in one channel (substrate-coasting at H_0 = 68.19); ΛCDM-fitting is a
  DIFFERENT channel that mis-extracts (Ω_m, Ω_Λ) by the factor-of-2.
  The two channels coexist above the waterline; observation
  (this Session 1's mock-then-fit experiment) discriminates whether
  the ΛCDM channel mis-extracts at the predicted magnitude.
""")


# =============================================================================
# §9. Session-1 deliverable + next-session entry
# =============================================================================
print("§9. Session-1 deliverable")
print("=" * 78)
print(f"""
  Generated mock SN1a Hubble-diagram data under TRUE-COASTING with
  framework H_0_observer = {H0_OBSERVER:.4f} km/s/Mpc and realistic
  Pantheon+-like z-distribution + σ_μ floor 0.04-0.16 mag.

  Three independent fits to the same mock:
    1. flat ΛCDM:   Ω_m = {Om_lcdm:.4f} ± {Om_lcdm_sigma:.4f}, H_0 = {H0_lcdm:.2f} ± {H0_lcdm_sigma:.2f}
    2. flat wCDM:   Ω_m = {Om_w:.4f}, w = {w_w:.4f} ± {w_w_sigma:.4f}
    3. q_0 cosmographic (z<0.3):  q_0 = {q0_q:.4f} ± {q0_q_sigma:.4f}

  VERDICT: {verdict}

  Next-session entries (Session 2 candidates):
    - Add CMB acoustic scale θ_* (= r_s/D_A): test if coasting recovers
      θ_* = 1.04 × 10⁻² rad consistent with Planck.
    - Add BAO from BOSS/DESI: test if coasting recovers correct sound
      horizon scale.
    - Run actual Pantheon+ likelihood (need real data file) instead of
      mock; check Ω_m recovered from real distance moduli under ΛCDM-fit.

  Item-2 status of cosmology roadmap: depends on this session's verdict.
""")
print("=" * 78)
print("DONE: Session 1 of Λ_CC Path A.")
print("=" * 78)
