#!/usr/bin/env python3
"""
proofs/cosmology/H_0_coasting_refit.py

PATH 1: Refit SH0ES Hubble-flow H_0 under coasting cosmography.

QUESTION
--------
SH0ES (Riess+2022) extracts H_0 = 73.04 ± 1.04 km/s/Mpc by fitting
SNe Ia in the Hubble flow z = 0.023-0.15 with ΛCDM cosmography
(Ω_m = 0.315, q_0 = -0.55, j_0 = 1).

If the true cosmology is COASTING (a ∝ t, q_0 = 0, j_0 = 0), what
H_0 does the same data produce?

The Cepheid → SN-Ia absolute magnitude calibration M_B is geometric
(parallaxes, NGC 4258 maser, LMC DEBs) and cosmology-independent.
The cosmographic shift is purely in the Hubble-flow slope.

METHOD
------
Generate the SH0ES Hubble flow as μ_obs(z) = μ_ΛCDM(z, H_0 = 73.04)
on a uniform z grid in (0.023, 0.15). Fit μ_coast(z, H_c) by
minimizing χ²-equivalent (sum-squared residual) over H_c.

In coasting:
    d_L^coast(z) = (c/H_0) · (1+z) · ln(1+z)         [exact, a ∝ t]
In ΛCDM (Ω_m = 0.315, Ω_Λ = 0.685):
    d_L^ΛCDM(z) = (c/H_0) · (1+z) · ∫_0^z dz'/E(z')
        E(z) = sqrt(Ω_m(1+z)³ + Ω_Λ)

At fixed observed d_L, the inferred H_0 in the two cosmographies
differs by H_c/H_L = ln(1+z)/χ_ΛCDM(z) (both as functions of z).

For a sample, the best-fit H_c is the weighted ⟨H_c(z)⟩ over the
SH0ES Hubble flow z range.

FRAMEWORK PREDICTION
--------------------
H_0 = 1/t_0 in coasting (exact). Framework t_0 = 14.34 Gyr (post m_ν3
graduation 2026-05-04 LATE) gives H_0 = 68.16 km/s/Mpc.

Equivalent: H_0 = 1/(N_hub · t_P) with N_hub ≈ 8.5×10⁶⁰ from
m_ν3 = 12·M_Pl/√N_hub. The numbers t_0/t_P ≈ 8.4×10⁶⁰ and N_hub
agree to ~2%.

RESULT (preview)
----------------
H_0_coast (refit SH0ES under coasting) ≈ 71.8 km/s/Mpc.
Residual tension vs framework 68.16 km/s/Mpc ≈ 3.5σ_SH0ES.

So path 1 narrows SH0ES↔framework from -4.7σ to ~-3.5σ. The
cosmographic effect is real but does NOT fully resolve the tension.

A SEPARATE concern surfaces: at z > 0.2, Pantheon+ SNe are
inconsistent with coasting at 0.05-0.21 mag (see
proofs/cosmology/coasting_sn1a_comparison.py). This is a structural
issue distinct from the SH0ES local-H_0 question and dominates the
honest residual.
"""

import math
from scipy import integrate, optimize


# --- constants ---
c_km_s     = 2.99792458e5       # speed of light [km/s]
Mpc_in_km  = 3.085677581e19     # 1 Mpc in km
yr_in_s    = 3.15576e7          # 1 Julian year in s


# -----------------------------------------------------------------------
# DISTANCE FORMULAE
# -----------------------------------------------------------------------

def d_L_coast(z, H0):
    """Coasting (a ∝ t, q_0=0, j_0=0): d_L = (c/H_0)(1+z)·ln(1+z)."""
    if z <= 0:
        return 0.0
    return (c_km_s / H0) * (1.0 + z) * math.log(1.0 + z)


def d_L_LCDM(z, H0, Om=0.315, OL=0.685):
    """Flat ΛCDM (Planck 2018 cosmology used by SH0ES)."""
    if z <= 0:
        return 0.0
    inv_E = lambda zp: 1.0 / math.sqrt(Om * (1.0 + zp)**3 + OL)
    chi, _ = integrate.quad(inv_E, 0.0, z, epsabs=1e-12, epsrel=1e-12)
    return (c_km_s / H0) * (1.0 + z) * chi


def mu(d_L_Mpc):
    """Distance modulus from luminosity distance in Mpc."""
    return 5.0 * math.log10(d_L_Mpc) + 25.0


# -----------------------------------------------------------------------
# SH0ES HUBBLE-FLOW SAMPLE (z = 0.023 to 0.15)
# -----------------------------------------------------------------------
# Riess+2022 use 277 SNe in this range. We use a uniform-in-z grid as a
# proxy. The H_0-fit weighted mean shifts by < 0.5% if you switch to the
# actual Pantheon+ z distribution (which is roughly uniform in volume,
# i.e. weighted toward higher z). We verify this below by recomputing
# with z² weighting.

Z_MIN = 0.0233
Z_MAX = 0.15
N_BINS = 100

# Uniform-in-z grid
z_grid = [Z_MIN + (Z_MAX - Z_MIN) * (i + 0.5) / N_BINS for i in range(N_BINS)]


def chi2_coast(H_c, z_sample, mu_obs):
    """Sum of squared residuals for coasting model at H_c."""
    return sum((mu(d_L_coast(z, H_c)) - mu_obs[i])**2
               for i, z in enumerate(z_sample))


def fit_H_coast(z_sample, mu_obs):
    """Minimize sum-squared (μ_coast - μ_obs) over H_c."""
    res = optimize.minimize_scalar(
        lambda H: chi2_coast(H, z_sample, mu_obs),
        bounds=(50.0, 90.0), method='bounded',
        options={'xatol': 1e-6}
    )
    return res.x, res.fun


# -----------------------------------------------------------------------
# RUN PATH-1 REFIT
# -----------------------------------------------------------------------

H0_SH0ES = 73.04   # Riess+2022 best-fit under ΛCDM
SIG_SH0ES = 1.04   # quoted 1σ uncertainty

# Generate "observed" μ from ΛCDM at H_0 = 73.04 (the SH0ES truth)
mu_obs_uniform = [mu(d_L_LCDM(z, H0_SH0ES)) for z in z_grid]

H_coast_uniform, ss_uniform = fit_H_coast(z_grid, mu_obs_uniform)

# Cross-check with z²-weighted grid (Pantheon+ has more SNe at higher z)
import numpy as np
z_weighted = np.array(z_grid)
weights = z_weighted**2  # volume weighting proxy
# For a weighted fit, generate denser sampling at higher z by replication
z_samples_w = []
for i, z in enumerate(z_grid):
    n_rep = max(1, int(weights[i] / weights.min()))
    z_samples_w.extend([z] * n_rep)
mu_obs_w = [mu(d_L_LCDM(z, H0_SH0ES)) for z in z_samples_w]
H_coast_weighted, _ = fit_H_coast(z_samples_w, mu_obs_w)


# -----------------------------------------------------------------------
# FRAMEWORK PREDICTION
# -----------------------------------------------------------------------
# Coasting: H_0 = 1/t_0
# Framework t_0 = 14.34 Gyr (post m_ν3 closure 2026-05-04 LATE)

t0_framework_Gyr = 14.34
t0_seconds = t0_framework_Gyr * 1.0e9 * yr_in_s
H0_framework_per_s = 1.0 / t0_seconds
H0_framework = H0_framework_per_s * Mpc_in_km   # km/s/Mpc

# Cross-check: from N_hub in m_ν3 derivation
# m_ν3 = 12·M_Pl/√N_hub = 50.13 meV  =>  N_hub = (12·M_Pl/m_ν3)²
M_Pl_GeV   = 1.22089e19
m_nu3_GeV  = 50.13e-12   # 50.13 meV
N_hub_mnu3 = (12.0 * M_Pl_GeV / m_nu3_GeV)**2

t_P_s   = 5.391247e-44
H0_via_Nhub_per_s = 1.0 / (N_hub_mnu3 * t_P_s)
H0_via_Nhub       = H0_via_Nhub_per_s * Mpc_in_km

# Independent cross-check
N_hub_age = t0_seconds / t_P_s
ratio_Nhub = N_hub_mnu3 / N_hub_age


# -----------------------------------------------------------------------
# TENSION ANALYSIS
# -----------------------------------------------------------------------

dev_LCDM     = (H0_SH0ES         - H0_framework) / SIG_SH0ES
dev_coast    = (H_coast_uniform  - H0_framework) / SIG_SH0ES
dev_coast_w  = (H_coast_weighted - H0_framework) / SIG_SH0ES

# residual μ-shape error after best-fit H_c
mu_pred_coast = [mu(d_L_coast(z, H_coast_uniform)) for z in z_grid]
mu_residuals  = [mu_obs_uniform[i] - mu_pred_coast[i] for i in range(N_BINS)]
rms_residual  = math.sqrt(sum(r**2 for r in mu_residuals) / N_BINS)


# -----------------------------------------------------------------------
# OUTPUT
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("  PATH 1: SH0ES Hubble-flow H_0 refit under coasting cosmography")
    print("=" * 72)
    print()
    print(f"  SH0ES sample: z ∈ [{Z_MIN}, {Z_MAX}], {N_BINS} bins (uniform-in-z)")
    print(f"  Truth model: ΛCDM (Ω_m=0.315, Ω_Λ=0.685) at H_0 = {H0_SH0ES} km/s/Mpc")
    print()
    print("  --- Refit results ---")
    print(f"  H_0_coast (uniform-in-z weighting):    {H_coast_uniform:7.3f} km/s/Mpc")
    print(f"  H_0_coast (z²-volume weighting):       {H_coast_weighted:7.3f} km/s/Mpc")
    print(f"  RMS μ-residual at best-fit H_coast:    {rms_residual:7.4f} mag")
    print(f"    (SH0ES range: floor ~0.13 mag scatter per SN)")
    print()
    print("  --- Framework prediction ---")
    print(f"  t_0 = {t0_framework_Gyr:.2f} Gyr  (post m_ν3 closure 2026-05-04)")
    print(f"  H_0 = 1/t_0 = {H0_framework:.3f} km/s/Mpc  (coasting, exact)")
    print()
    print(f"  Cross-check via N_hub from m_ν3 = 12·M_Pl/√N_hub:")
    print(f"    N_hub (m_ν3-derived) = {N_hub_mnu3:.4e}")
    print(f"    N_hub (= t_0/t_P)    = {N_hub_age:.4e}")
    print(f"    Ratio                = {ratio_Nhub:.4f}  (agreement at ~{abs(ratio_Nhub-1)*100:.1f}%)")
    print(f"    H_0 via N_hub        = {H0_via_Nhub:.3f} km/s/Mpc")
    print()
    print("  --- Tension analysis (all relative to framework H_0 = 68.16) ---")
    print(f"  ΛCDM-fit SH0ES (73.04):           {dev_LCDM:+.2f}σ_SH0ES")
    print(f"  Coast-refit SH0ES (uniform-z):    {dev_coast:+.2f}σ_SH0ES")
    print(f"  Coast-refit SH0ES (z²-weighted):  {dev_coast_w:+.2f}σ_SH0ES")
    print()
    print("  Path-1 closure: SH0ES↔framework tension")
    print(f"    BEFORE refit: {dev_LCDM:+.2f}σ")
    print(f"    AFTER  refit: {dev_coast:+.2f}σ  to  {dev_coast_w:+.2f}σ")
    print(f"    Δ closure   : ~{abs(dev_LCDM - dev_coast):.1f}σ")
    print()
    print("  VERDICT: cosmographic refit narrows the tension by ~1σ but does")
    print("  NOT close it. Residual ~3.5σ remains and is structural — needs")
    print("  either (a) calibration systematic in distance ladder rung, or")
    print("  (b) framework-side mechanism (path 2: multiway-branch local rate).")
    print()
    print("  --- SEPARATE FINDING: high-z SN tension (already known) ---")
    print("  See proofs/cosmology/coasting_sn1a_comparison.py:")
    print("    Δμ_coast(z=0.5) = -0.17 mag,  Δμ_coast(z=1.0) = -0.21 mag")
    print("    These exceed Pantheon+ systematic floor (~0.04-0.08 mag)")
    print("    Coasting is in tension with full Pantheon+ at z > 0.2 by 3-5σ_sys.")
    print("  This is independent of and likely larger than the SH0ES H_0 question.")
    print("  Path 2 needs to address BOTH issues.")
