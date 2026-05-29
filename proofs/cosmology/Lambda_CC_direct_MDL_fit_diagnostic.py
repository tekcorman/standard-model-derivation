"""
Λ_CC factor-of-2 — direct MDL-fit diagnostic (mechanism 3/4 test).

PURPOSE
-------
Test whether observer-MDL primary, without inserting a substrate-frame
Ω partition as intermediate, predicts the observed ΛCDM Ω parameters
when humans fit ΛCDM to framework cosmology.

The setup (entirely framework-internal, no external math references):

  - Framework's substrate is COASTING: H_substrate(z) = H_0_substrate × (1+z).
    This is the cascade theorem D1+D2+D3 prediction.
  - Framework's observer at z=0 has rate-gap: H_observer(z=0) = (16/15) ×
    H_0_substrate. This is theorem-grade as of yesterday's IC closure.
  - For the diagnostic, extend the observer's H(z) to general z by the
    simplest framework-internal extension: assume the rate-gap factor
    is z-independent (constant 16/15 at all z). This is a strong
    simplifying assumption — the OS-2 scoping doc flags z-dependence
    as the genuine open question. Reporting under this assumption
    is the diagnostic for mechanism (3)/(4).

The data: simulate luminosity distances d_L(z) for SN1a-like redshifts
z ∈ [0.01, 2] under framework's observer H(z). Add Gaussian noise at
typical SN1a precision (σ_d_L ~ 5% per supernova).

The fit: ΛCDM model H(z) = H_0 × √(Ω_m (1+z)³ + Ω_Λ + Ω_r (1+z)⁴).
Flat universe constraint: Ω_m + Ω_Λ + Ω_r = 1. Ω_r ≈ 0 at low-z.
Free parameters: (H_0, Ω_m) with Ω_Λ = 1 - Ω_m.

The MDL-optimal fit minimizes χ² = Σ_i (d_L_obs - d_L_model)² / σ_i².
Under Gaussian likelihood, MDL ≡ χ² minimization (cite: Grünwald 2007
§5.3 — same MDL-Bayes duality used in A2-T waterline derivation).

VERDICT CRITERION
-----------------
H1 (mechanism 3/4 holds): recovered Ω_m_LCDM ≈ 0.315 within ~1σ Planck.
   Direct MDL-optimal compression gives the empirical Ω; no (1/2)
   reorganization needed.

H0 (mechanism 3/4 fails): recovered Ω_m_LCDM ≠ 0.315; the (1/2)
   reorganization in Row P23 is empirically necessary. The factor-of-2
   has a structural origin not captured by direct MDL optimization
   on coasting + (16/15).

Either outcome is informative. H1 closes the factor-of-2; H0 sharpens
the gap to a specific structural target.
"""

import math
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize


# ============================================================
# FRAMEWORK CONSTANTS
# ============================================================

H0_SUBSTRATE = 68.18      # km/s/Mpc, Row P19 substrate-side, theorem-grade
RATE_GAP = 16.0 / 15.0    # cascade D2-extended, theorem-grade post-2026-05-07
H0_OBSERVER = RATE_GAP * H0_SUBSTRATE  # = 72.74 km/s/Mpc

C_KM_S = 299792.458       # speed of light, km/s


# ============================================================
# PLANCK 2018 OBSERVATIONS (for comparison)
# ============================================================

OMEGA_M_PLANCK = 0.3153
SIGMA_OMEGA_M_PLANCK = 0.0073
H0_PLANCK = 67.36         # CMB-anchored ΛCDM-fit
SIGMA_H0_PLANCK = 0.54


# ============================================================
# §1. FRAMEWORK FORWARD MODEL — coasting + rate-gap
# ============================================================

def H_framework_observer(z):
    """
    Framework's observer-side Hubble rate at redshift z.

    Substrate is coasting (cascade theorem D1+D2+D3): H_sub = H_0 (1+z).
    Observer rate-gap (cascade D2-extended) at z=0: H_obs = (16/15) H_sub.

    For this diagnostic, extend rate-gap to all z (constant 16/15
    factor). This is the simplest framework-internal extension. OS-2
    flags z-dependence as the genuine open question.
    """
    return H0_OBSERVER * (1.0 + z)


def luminosity_distance_framework(z):
    """
    d_L(z) = (1+z) × c × ∫_0^z dz'/H(z') under framework's observer H(z).

    For coasting H = H_0 (1+z), the comoving distance is
       D_C(z) = (c/H_0) × ln(1+z)
    so d_L(z) = (1+z) × (c/H_0) × ln(1+z).
    """
    if z <= 0:
        return 0.0
    return (1.0 + z) * (C_KM_S / H0_OBSERVER) * math.log(1.0 + z)


# ============================================================
# §2. ΛCDM MODEL (the parametric class humans fit)
# ============================================================

def H_LCDM(z, H0, Omega_m):
    """
    ΛCDM Hubble rate. Flat universe, Ω_Λ = 1 − Ω_m, Ω_r ≈ 0 at low z.
    """
    return H0 * math.sqrt(Omega_m * (1.0 + z) ** 3 + (1.0 - Omega_m))


def luminosity_distance_LCDM(z, H0, Omega_m):
    """d_L(z) under ΛCDM via numerical integration of dz'/H(z')."""
    if z <= 0:
        return 0.0
    integrand = lambda zp: 1.0 / H_LCDM(zp, H0, Omega_m)
    DC_integral, _ = quad(integrand, 0, z, limit=100)
    return (1.0 + z) * C_KM_S * DC_integral


# ============================================================
# §3. SYNTHETIC SN1a DATA from framework
# ============================================================

def generate_synthetic_data(n_sne=200, z_min=0.01, z_max=2.0, seed=42):
    """
    Generate synthetic SN1a-like d_L measurements under framework H(z).

    Redshift distribution: uniform in z (good enough for diagnostic;
    real SN1a samples are denser at low z but the χ² fit is dominated
    by the high-z lever arm anyway).

    Uncertainty: 5% relative on d_L per supernova (typical Pantheon+
    precision after standardization).
    """
    rng = np.random.default_rng(seed)
    z_array = np.linspace(z_min, z_max, n_sne)
    d_L_truth = np.array([luminosity_distance_framework(z) for z in z_array])
    sigma_rel = 0.05  # 5% per SN
    sigma_array = sigma_rel * d_L_truth
    noise = rng.normal(loc=0.0, scale=sigma_array)
    d_L_obs = d_L_truth + noise
    return z_array, d_L_obs, sigma_array


# ============================================================
# §4. ΛCDM χ²-fit to framework data
# ============================================================

def chi_squared_LCDM(params, z_array, d_L_obs, sigma_array):
    """χ² = Σ_i (d_L_obs - d_L_LCDM)² / σ_i²."""
    H0, Omega_m = params
    if Omega_m < 0 or Omega_m > 1.5:
        return 1e10
    if H0 < 30 or H0 > 120:
        return 1e10
    chi2 = 0.0
    for z, d_obs, sigma in zip(z_array, d_L_obs, sigma_array):
        d_model = luminosity_distance_LCDM(z, H0, Omega_m)
        chi2 += ((d_obs - d_model) / sigma) ** 2
    return chi2


def fit_LCDM(z_array, d_L_obs, sigma_array):
    """Find (H_0, Ω_m) minimizing χ² to framework synthetic data."""
    initial_guess = [70.0, 0.3]
    result = minimize(
        chi_squared_LCDM,
        initial_guess,
        args=(z_array, d_L_obs, sigma_array),
        method='Nelder-Mead',
        options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 10000},
    )
    H0_fit, Omega_m_fit = result.x
    chi2_min = result.fun
    dof = len(z_array) - 2
    return {
        'H0_fit': H0_fit,
        'Omega_m_fit': Omega_m_fit,
        'Omega_Lambda_fit': 1.0 - Omega_m_fit,
        'chi2_min': chi2_min,
        'chi2_per_dof': chi2_min / dof,
        'dof': dof,
        'success': result.success,
    }


# ============================================================
# §5. DIAGNOSTIC — does direct MDL-fit recover Planck Ω?
# ============================================================

def diagnostic():
    print("=" * 72)
    print("Λ_CC factor-of-2 — direct MDL-fit diagnostic")
    print("=" * 72)
    print()
    print("Setup:")
    print(f"  Framework substrate H_0:  {H0_SUBSTRATE} km/s/Mpc")
    print(f"  Framework observer H_0:   {H0_OBSERVER:.2f} km/s/Mpc (= 16/15 × substrate)")
    print(f"  Framework H(z):           coasting (H ∝ (1+z) at all z)")
    print(f"  Rate-gap z-dependence:    assumed constant for diagnostic")
    print()
    print(f"  Planck ΛCDM-fit Ω_m:      {OMEGA_M_PLANCK} ± {SIGMA_OMEGA_M_PLANCK}")
    print(f"  Planck ΛCDM-fit H_0:      {H0_PLANCK} ± {SIGMA_H0_PLANCK} km/s/Mpc")
    print()

    # Generate synthetic data
    print("-" * 72)
    print("§1. Synthetic data from framework H(z)")
    print("-" * 72)
    z_array, d_L_obs, sigma_array = generate_synthetic_data()
    print(f"  Generated {len(z_array)} synthetic SN1a-like measurements")
    print(f"  Redshift range:  [{z_array[0]:.2f}, {z_array[-1]:.2f}]")
    print(f"  Per-SN σ:        5% relative on d_L")
    print()
    print(f"  Sample d_L predictions (framework, no noise):")
    for z in [0.01, 0.1, 0.5, 1.0, 1.5, 2.0]:
        d = luminosity_distance_framework(z)
        print(f"    z = {z:.2f}  →  d_L = {d:>10.1f} Mpc")
    print()

    # Run ΛCDM fit
    print("-" * 72)
    print("§2. ΛCDM χ²-fit to framework data")
    print("-" * 72)
    fit = fit_LCDM(z_array, d_L_obs, sigma_array)
    print(f"  Recovered H_0:           {fit['H0_fit']:.2f} km/s/Mpc")
    print(f"  Recovered Ω_m:           {fit['Omega_m_fit']:.4f}")
    print(f"  Recovered Ω_Λ:           {fit['Omega_Lambda_fit']:.4f}")
    print(f"  χ² minimum:              {fit['chi2_min']:.2f}")
    print(f"  χ²/dof:                  {fit['chi2_per_dof']:.2f}  (dof = {fit['dof']})")
    print()

    # Compare to Planck
    print("-" * 72)
    print("§3. Comparison to Planck observation")
    print("-" * 72)
    om_dev = fit['Omega_m_fit'] - OMEGA_M_PLANCK
    om_sigma = abs(om_dev) / SIGMA_OMEGA_M_PLANCK
    h_dev = fit['H0_fit'] - H0_PLANCK
    h_sigma = abs(h_dev) / SIGMA_H0_PLANCK
    print(f"  Ω_m:  recovered = {fit['Omega_m_fit']:.4f}  vs Planck = {OMEGA_M_PLANCK}")
    print(f"        deviation = {om_dev:+.4f}  ({om_sigma:.1f}σ)")
    print()
    print(f"  H_0:  recovered = {fit['H0_fit']:.2f}  vs Planck = {H0_PLANCK}")
    print(f"        deviation = {h_dev:+.2f}  ({h_sigma:.1f}σ)")
    print()

    # Compare with framework's Row P23 prediction
    print("-" * 72)
    print("§4. Comparison to Row P23 prediction (with (1/2) reorganization)")
    print("-" * 72)
    OMEGA_M_SUBSTRATE = 2.0 / 3.0
    OMEGA_M_LCDM_ROW_P23 = 0.5 * OMEGA_M_SUBSTRATE  # = 1/3
    print(f"  Row P23 prediction Ω_m_LCDM:     {OMEGA_M_LCDM_ROW_P23:.4f}")
    print(f"  Direct MDL-fit Ω_m:               {fit['Omega_m_fit']:.4f}")
    print(f"  Without (1/2): naive Ω_m would be substrate's {OMEGA_M_SUBSTRATE:.4f}")
    print()

    # Verdict
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    print()
    if om_sigma < 1.0:
        verdict = "H1 PASS"
        msg = ("Direct MDL-fit recovers Planck Ω_m within 1σ. The factor-of-2 "
               "is an artifact of the substrate-frame Ω intermediate; mechanism "
               "(3)/(4) closes the gap directly. (1/2) reorganization NOT needed.")
    elif om_sigma < 3.0:
        verdict = "AMBIGUOUS"
        msg = ("Direct MDL-fit gives Ω_m within 1-3σ of Planck. Suggestive of "
               "mechanism (3)/(4) but residual remains. Multi-session work to "
               "tighten the diagnostic (multi-dataset, z-dependent rate-gap).")
    else:
        verdict = "H0 — H1 fails"
        msg = ("Direct MDL-fit gives Ω_m well outside Planck precision. "
               "Mechanism (3)/(4) does not directly produce observed Ω. The "
               "(1/2) reorganization in Row P23 is empirically necessary; "
               "it has a structural origin not captured by direct ΛCDM-fit "
               "of coasting + constant rate-gap. Needs deeper investigation.")
    print(f"  {verdict}")
    print()
    print(f"  Ω_m deviation: {om_sigma:.1f}σ from Planck")
    print(f"  H_0 deviation: {h_sigma:.1f}σ from Planck")
    print()
    print(f"  {msg}")
    print()
    print("=" * 72)


if __name__ == "__main__":
    diagnostic()
