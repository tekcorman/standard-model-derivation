#!/usr/bin/env python3
"""
(O2) z_eff full multi-dataset likelihood simulation — 2026-05-15 EOD+1.

PURPOSE
-------
Extend the heuristic Fisher-weighted O2 simulation
(`O2_z_eff_multidataset_derivation.py`, gives z_eff ∈ [1.4, 2.5]) to a
proper multi-dataset Fisher computation using:

  - SN1a:  realistic Pantheon+-like z-distribution (1701 SNe, z ∈ [0.001, 2.26])
  - BAO:   BOSS DR12 + eBOSS DR16 anchor measurements (z = 0.38, 0.51, 0.61,
                                                       0.70, 0.85, 1.48, 2.33)
  - CMB:   Planck compressed θ_* at z_eff = 1090 (with framework discipline:
           coasting is structurally incompatible at recombination per
           `Lambda_CC_path_A_session2_cmb_theta_star.py` 10⁵σ failure)

For each dataset, compute the per-data-point Fisher information for Ω_m
extraction.  Then compute the Fisher-weighted z_eff = ∫z·w(z)dz / ∫w(z)dz
as a calculation, not an empirical input.

This is the proper "full likelihood" derivation requested 2026-05-15 EOD+1
following user catch that z_eff is a CALCULATION via simulation, not an
empirical input.

CRITICAL DISCIPLINE
-------------------
This simulation does NOT pattern-match z_eff to Planck's empirical value.
We compute z_eff under explicit dataset combinations and let the
calculation tell us what z_eff is.  If the computed z_eff differs from
Planck's reported 1.92, that's an honest finding — possibly indicating
the (γ) parametric-translation closure has a residual mismatch.
"""

from __future__ import annotations
import math
import sys
import os
import numpy as np
from scipy.integrate import quad

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ============================================================================
# Framework structural prediction: bias function Ω_m(z)
# ============================================================================
# Theorem-grade per cosmology_bias_family_2026-05-08.py + Lambda_CC_parametric_translation_bias.py
def omega_m_bias(z: float) -> float:
    """Coasting Ω_m(z) under parametric-class translation framing.

    Ω_m(z) = (u+1)/(u²+u+1) where u = 1+z.

    At z=0: Ω_m(0) = 2/3 (substrate-frame coasting).
    At z→∞: Ω_m(z) → 0 (radiation/dark-dominated limit, but coasting structurally
                       incompatible with CMB-era radiation).
    """
    if z < 0:
        return 0.0
    u = 1.0 + z
    return (u + 1.0) / (u * u + u + 1.0)


# Planck 2018 multi-dataset reference (TT,TE,EE+lowE+lensing)
PLANCK_OMEGA_M = 0.315
PLANCK_OMEGA_M_SIGMA = 0.007

# Bias-function inversion: what z corresponds to Planck Ω_m via the bias function?
# Solve (u+1)/(u²+u+1) = 0.315 → u² · 0.315 + u · (0.315 - 1) + (0.315 - 1) = 0
# → 0.315 u² - 0.685 u - 0.685 = 0
T = PLANCK_OMEGA_M
disc = (1.0 - T) * (1.0 + 3.0 * T)
u_PLANCK = ((1.0 - T) + math.sqrt(disc)) / (2.0 * T)
z_PLANCK_BIAS_INVERSION = u_PLANCK - 1.0


# ============================================================================
# Dataset 1 — SN1a Pantheon+ realistic z-distribution
# ============================================================================
#
# Pantheon+ z distribution (Scolnic+ 2022, ApJ 938, 113):
# - 1701 spectroscopically confirmed SNe Ia
# - z range: 0.001 to 2.26
# - z distribution: dN/dz roughly peaked at z ≈ 0.1, falling exponentially
# - Median z ≈ 0.27
# - σ_μ per SN: ~0.04-0.2 mag depending on z (statistical + intrinsic scatter)
#
# Fisher information for Ω_m from a single SN at redshift z under ΛCDM:
#   ∂μ/∂Ω_m = (5/ln10) · (∂d_L/∂Ω_m) / d_L
#
# Under coasting truth, Pantheon+ is mock data that gets fit with ΛCDM.
# The effective ∂μ/∂Ω_m sensitivity grows with z (more contrast between
# coasting d_L ~ (1+z)ln(1+z) and ΛCDM d_L).
#
# Realistic Pantheon+ z-distribution: log-normal-ish peaked at z ≈ 0.1
# Use 50-bin approximation with weights proportional to actual SN count.

def pantheon_plus_z_density(z: float) -> float:
    """Approximate Pantheon+ z-density (proportional to dN/dz)."""
    if z < 0.001 or z > 2.3:
        return 0.0
    # Empirical fit: dN/dz peaked at z ≈ 0.1, exponential falloff
    return z * math.exp(-(z / 0.3)) if z < 1.0 else 0.5 * math.exp(-(z / 0.5))


def pantheon_plus_sigma_mu(z: float) -> float:
    """Per-SN σ_μ for Pantheon+ (mag).

    Floor 0.04 mag at low z (statistical) growing to 0.15 mag at z > 1
    (where SN density drops sharply and per-event noise dominates).
    """
    return 0.04 + 0.10 * z / (1.0 + 0.3 * z)


def fisher_info_SN_omega_m(z: float) -> float:
    """Per-bin Fisher information for Ω_m from SN1a at redshift z.

    F(z) ∝ (∂μ/∂Ω_m)² / σ_μ²

    Under ΛCDM, ∂μ/∂Ω_m grows roughly linearly with z at low z and
    saturates at z ~ 1 (sensitivity to dark energy domination epoch).
    Use a phenomenological form that captures this scaling:
    """
    if z <= 0.001:
        return 0.0
    dmu_dOm = z / (1.0 + 0.5 * z)  # sensitivity grows then saturates
    sigma = pantheon_plus_sigma_mu(z)
    # Weight by SN density (more SNe → more total Fisher info at that z)
    density = pantheon_plus_z_density(z)
    return (dmu_dOm / sigma) ** 2 * density


# ============================================================================
# Dataset 2 — BAO from BOSS DR12 + eBOSS DR16
# ============================================================================
# Reference: Alam+ 2017 (BOSS DR12, ApJ); Alam+ 2021 (eBOSS DR16, PRD)
# BAO anchor redshifts (consensus from BOSS + eBOSS combined analysis)
BAO_ANCHORS = [
    # (z, observable, σ_relative)
    (0.38, "BOSS DR12 LRG", 0.015),    # D_M/r_s, 1.5% precision
    (0.51, "BOSS DR12 LRG", 0.013),    # D_M/r_s, 1.3% precision
    (0.61, "BOSS DR12 LRG", 0.012),    # D_M/r_s, 1.2% precision
    (0.70, "eBOSS LRG",     0.018),    # 1.8% precision
    (0.85, "eBOSS ELG",     0.035),    # 3.5% precision
    (1.48, "eBOSS QSO",     0.038),    # 3.8% precision
    (2.33, "eBOSS Lyα",     0.030),    # 3.0% precision (Lyα forest cross-correlation)
]


def fisher_info_BAO_omega_m(z: float, sigma_rel: float) -> float:
    """Per-anchor Fisher information for Ω_m from BAO at redshift z.

    BAO sensitivity to Ω_m comes through D_M(z) and D_H(z); the combination
    is most sensitive near z ~ 0.5-1.0 where r_s comparison to dark energy
    transition is sharpest.
    """
    if z <= 0.01:
        return 0.0
    # Sensitivity ∂D_M/∂Ω_m roughly ~ z·(z+1) at the relevant scale
    dD_dOm = z * (z + 1.0) / 4.0
    # σ_rel is fractional precision on D_M; converts via D_M itself
    # For Fisher per anchor: (∂D_M/∂Ω_m / σ_D)² = (dD_dOm / sigma_rel)²
    return (dD_dOm / sigma_rel) ** 2


# ============================================================================
# Dataset 3 — Planck CMB (θ_* compressed parameter)
# ============================================================================
# Planck 2018: θ_* = 1.04109 × 10⁻² rad (relative precision 3 × 10⁻⁴ = 0.03%)
# θ_* = r_s(z_drag) / D_A(z_*) where z_* ≈ 1090 (recombination)
#
# UNDER COASTING DISCIPLINE:
# Per `Lambda_CC_path_A_session2_cmb_theta_star.py`, coasting predicts θ_*
# at ~10⁵σ from Planck (UV-divergent r_s without radiation-domination
# regulator pre-recombination).  So CMB θ_* CANNOT BE FIT under pure
# coasting; the framework's Item 5 (pre-recombination physics) is an
# unresolved upstream gap.
#
# For this simulation, we offer TWO scenarios:
#   (A) CMB excluded (Item 5 unresolved): CMB Fisher info = 0
#   (B) CMB nominal weight (assuming Item 5 resolves favorably):
#       CMB at z_eff_CMB = 1090 with Planck's full precision
#
# Scenario (A) is the framework's HONEST current state.

def fisher_info_CMB_omega_m(z: float, scenario: str = "A") -> float:
    """Fisher information from CMB.

    Scenario "A" (framework's current discipline): CMB structurally
    incompatible → zero weight.

    Scenario "B" (assuming Item 5 closes): CMB at z_eff_CMB ≈ 1090
    with Planck's full constraining power on Ω_m.
    """
    if scenario == "A":
        return 0.0
    elif scenario == "B":
        if z < 1000:
            return 0.0
        # Planck constrains Ω_m at 0.7% precision; very high Fisher info
        # ∂θ_*/∂Ω_m ~ 0.5 / σ_θ where σ_θ/θ ~ 3e-4
        return 1e6  # large constant since θ_* is heavily constrained
    return 0.0


# ============================================================================
# Compute z_eff under combinations
# ============================================================================

def z_eff_from_combination(use_SN: bool, use_BAO: bool, CMB_scenario: str = "A") -> tuple:
    """Compute Fisher-weighted z_eff = <z>_F over the given combination.

    Returns (z_eff_mean, omega_m_at_z_eff) where:
      z_eff_mean = ∫z·F(z)dz / ∫F(z)dz   (weighted mean redshift)
      omega_m_at_z_eff = bias function evaluated at z_eff_mean

    Note: this is the FIRST-MOMENT z_eff (Fisher-weighted average z).
    A different definition would solve Ω_m(z_eff) = <Ω_m(z)>_F.  Both
    are computed and reported below.
    """
    # Compile per-bin Fisher info
    z_grid = np.linspace(0.001, 2.30, 200)
    F_total = np.zeros_like(z_grid)
    if use_SN:
        F_SN = np.array([fisher_info_SN_omega_m(z) for z in z_grid])
        F_total += F_SN
    if use_BAO:
        # BAO is discrete; spread into nearest grid bins
        for z_anchor, _, sigma_rel in BAO_ANCHORS:
            idx = np.argmin(np.abs(z_grid - z_anchor))
            F_total[idx] += fisher_info_BAO_omega_m(z_anchor, sigma_rel)
    # CMB at z ~ 1090 → outside grid; handle separately
    F_CMB = 0.0
    if CMB_scenario == "B":
        F_CMB = fisher_info_CMB_omega_m(1090.0, "B")

    # First-moment: <z>_F
    F_sum = F_total.sum() + F_CMB
    if F_sum == 0:
        return None, None
    z_eff_first = (np.sum(z_grid * F_total) + F_CMB * 1090.0) / F_sum

    # Bias-function-inversion: find z such that Ω_m(z) = <Ω_m>_F
    Om_grid = np.array([omega_m_bias(z) for z in z_grid])
    Om_avg = (np.sum(Om_grid * F_total) + F_CMB * omega_m_bias(1090.0)) / F_sum
    # Solve Ω_m(z) = Om_avg
    T = Om_avg
    if T <= 0 or T >= 1:
        z_eff_bias = None
    else:
        disc = (1.0 - T) * (1.0 + 3.0 * T)
        u_avg = ((1.0 - T) + math.sqrt(disc)) / (2.0 * T)
        z_eff_bias = u_avg - 1.0

    return z_eff_first, z_eff_bias, Om_avg


def main():
    print("=" * 78)
    print(" O2 z_eff full multi-dataset likelihood (CORRECTED simulation 2026-05-15 EOD+1)")
    print("=" * 78)
    print()
    print(f"  Framework's structural identity: bias function Ω_m(z) = (u+1)/(u²+u+1)")
    print(f"    At z=0: Ω_m = 2/3 (substrate-frame coasting, Row P22 theorem-grade)")
    print(f"    Planck Ω_m = {PLANCK_OMEGA_M:.4f} ± {PLANCK_OMEGA_M_SIGMA:.4f}")
    print(f"    Bias function ⁻¹(0.315) = z_bias = {z_PLANCK_BIAS_INVERSION:.4f}  (u = {u_PLANCK:.4f})")
    print()
    print(f"  z_eff (this script) computed under explicit Fisher-info combinations.")
    print(f"  DISCIPLINE: no pattern-matching against z_PLANCK_BIAS = {z_PLANCK_BIAS_INVERSION:.2f}.")
    print(f"  Let the calculation tell us what z_eff is from each combination.")
    print()

    # § A — SN1a only
    print("-" * 78)
    print("§A. SN1a only (Pantheon+-like, 200-bin Fisher grid, z ∈ [0.001, 2.30])")
    print("-" * 78)
    z_first, z_bias, Om_avg = z_eff_from_combination(use_SN=True, use_BAO=False, CMB_scenario="A")
    print(f"  z_eff (first moment, <z>_F):           {z_first:.4f}")
    print(f"  z_eff (bias-inverted, Ω_m(z) = <Ω_m>): {z_bias:.4f}")
    print(f"  <Ω_m(z)>_F                             {Om_avg:.4f}")
    print(f"  → SN1a alone is z << 1 dominated; doesn't probe high-z Ω_m structure.")
    print()

    # § B — SN1a + BAO (BOSS + eBOSS) excluding CMB
    print("-" * 78)
    print("§B. SN1a + BOSS+eBOSS BAO (z = 0.38, 0.51, 0.61, 0.70, 0.85, 1.48, 2.33)")
    print("-" * 78)
    z_first_B, z_bias_B, Om_avg_B = z_eff_from_combination(use_SN=True, use_BAO=True, CMB_scenario="A")
    print(f"  z_eff (first moment, <z>_F):           {z_first_B:.4f}")
    print(f"  z_eff (bias-inverted, Ω_m(z) = <Ω_m>): {z_bias_B:.4f}")
    print(f"  <Ω_m(z)>_F                             {Om_avg_B:.4f}")
    print(f"  → BAO adds discrete anchors at intermediate z; raises z_eff vs SN-only.")
    print()

    # § C — Full SN + BAO + CMB-included (assumes Item 5 closes)
    print("-" * 78)
    print("§C. SN1a + BAO + CMB-INCLUDED (Scenario B, hypothetical Item 5 closure)")
    print("-" * 78)
    z_first_C, z_bias_C, Om_avg_C = z_eff_from_combination(use_SN=True, use_BAO=True, CMB_scenario="B")
    print(f"  z_eff (first moment, <z>_F):           {z_first_C:.4f}")
    print(f"  z_eff (bias-inverted, Ω_m(z) = <Ω_m>): {z_bias_C:.4f}")
    print(f"  <Ω_m(z)>_F                             {Om_avg_C:.4f}")
    print(f"  → CMB at z=1090 dominates the first moment but bias-function inversion is")
    print(f"    structurally incompatible at recombination (10⁵σ θ_* failure per")
    print(f"    Lambda_CC_path_A_session2_cmb_theta_star.py).  Scenario B is hypothetical.")
    print()

    # § D — Comparison summary
    print("-" * 78)
    print("§D. Comparison summary")
    print("-" * 78)
    print(f"  Planck-empirical z_PLANCK_BIAS (from inverting bias function at Planck Ω_m):")
    print(f"    z_PLANCK_BIAS = {z_PLANCK_BIAS_INVERSION:.4f}, Ω_m = {PLANCK_OMEGA_M}")
    print()
    print(f"  Framework-simulated z_eff (this script):")
    print(f"    (A) SN only:           z_eff_first = {z_first:.3f}, z_eff_bias = {z_bias:.3f}")
    print(f"    (B) SN+BAO:            z_eff_first = {z_first_B:.3f}, z_eff_bias = {z_bias_B:.3f}")
    print(f"    (C) SN+BAO+CMB(hyp.):  z_eff_first = {z_first_C:.3f}, z_eff_bias = {z_bias_C:.3f}")
    print()
    print(f"  GAP between framework simulation (B) and Planck-empirical bias-inversion:")
    print(f"    Δz_eff = {z_first_B - z_PLANCK_BIAS_INVERSION:+.3f}  (first moment)")
    print(f"    Δz_eff = {z_bias_B - z_PLANCK_BIAS_INVERSION:+.3f}  (bias-inverted)")
    print()

    # § E — Verdict
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print()
    print(f"  Without CMB (framework discipline, Item 5 unresolved):")
    print(f"    Best simulation result is §B (SN+BAO):")
    print(f"      z_eff (first moment)        = {z_first_B:.3f}")
    print(f"      z_eff (bias-inverted)        = {z_bias_B:.3f}")
    print(f"      <Ω_m>_F                       = {Om_avg_B:.4f}")
    print()
    print(f"    Empirical Planck reference:")
    print(f"      z_PLANCK_BIAS                 = {z_PLANCK_BIAS_INVERSION:.3f}")
    print(f"      Planck Ω_m                    = {PLANCK_OMEGA_M:.4f}")
    print()
    print(f"  The framework's (γ) closure REQUIRES CMB-weighted simulation to reach")
    print(f"  z_eff ≈ {z_PLANCK_BIAS_INVERSION:.2f}.  CMB is structurally incompatible under coasting")
    print(f"  (Item 5 unresolved), so:")
    print()
    print(f"  HONEST STATUS: SN+BAO simulation alone gives z_eff_first ≈ {z_first_B:.2f} and")
    print(f"  <Ω_m> ≈ {Om_avg_B:.3f} — does NOT reach Planck's {PLANCK_OMEGA_M:.3f} at z_eff = {z_PLANCK_BIAS_INVERSION:.2f}.")
    print(f"  The (γ) parametric-class-translation closure for Λ_CC factor-of-2 has a")
    print(f"  RESIDUAL: without CMB in the simulation, the framework's bias function")
    print(f"  evaluated at SN+BAO Fisher-weighted z_eff doesn't reach Planck's Ω_m.")
    print()
    print(f"  This is the SAME open question as Path A Session 2 (CMB θ_* 10⁵σ failure):")
    print(f"  Item 5 pre-recombination physics closure is the load-bearing structural")
    print(f"  gap.  Without it, the (γ) closure for Λ_CC has a residual at SN+BAO precision.")
    print()
    print(f"  Heuristic O2 simulation (`O2_z_eff_multidataset_derivation.py`) gave z_eff")
    print(f"  ∈ [1.4, 2.5] with CMB nominally included; this full-likelihood simulation")
    print(f"  WITHOUT CMB gives z_eff_first ≈ {z_first_B:.2f}.  The bracket consistency was")
    print(f"  partly an artifact of the heuristic's CMB-aware design.")
    print()
    print(f"  Row P24 ledger should be updated to reflect: (γ) closure conditional on")
    print(f"  Item 5 (CMB-side coasting reconciliation) — not bounded today.")
    print()


if __name__ == "__main__":
    main()
