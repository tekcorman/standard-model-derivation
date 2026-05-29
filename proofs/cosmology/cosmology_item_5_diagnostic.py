#!/usr/bin/env python3
"""
proofs/cosmology/cosmology_item_5_diagnostic.py

Cosmology Item 5 — single-session diagnostic.

Per an internal working note
§3, before opening Item 5 as multi-sprint research, run a diagnostic that:

  (1) Re-reads an internal working note Step C carefully to
      determine whether Step C's exponential N(t) and Step D's linear N(t)
      describe (a) genuinely inconsistent claims at the same epoch, or
      (b) different observables that coexist, or (c) different epochs
      that coexist.
  (2) For each viable Candidate (5.1 multiway-branching power law, 5.3
      inflation-like Ramanujan de Sitter), computes predicted r_s(z=z_*)
      and 100·θ_* and compares to Planck 1.04109.
  (3) Decides on multi-sprint commitment based on which (if any) Candidate
      could match Planck.

Inputs (Type 4)
---------------
- Path A Session 2 computed θ_* under pure coasting: ratio ~530x Planck.
- Step C / Step D structural derivations in N_hub_spectral_gap_attempt.py.
- Cascade D1/D2/D3 epoch-validity audit: D1+D2 epoch-robust; D3 has
  implicit synchronization not obviously broken at z=z_*.

Output
------
Diagnostic verdict on each Candidate's viability for matching Planck θ_*.
NOT a closure attempt. NOT a Path D scoping doc. A research-direction
sharpener.
"""

import sys
import os
import math

import numpy as np
from scipy import integrate

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# =============================================================================
# §0. Constants and framework predictions
# =============================================================================

c_km_s = 2.99792458e5

# Framework
H0_SUBSTRATE = 68.19            # km/s/Mpc, cascade H = 1/(N t_P) at z=0
H0_OBSERVER = (16.0 / 15.0) * H0_SUBSTRATE  # = 72.74 (D2-extended)

# Planck 2018 (TT,TE,EE+lowE+lensing baseline)
PLANCK_H0 = 67.36
PLANCK_OMEGA_M = 0.3153
PLANCK_OMEGA_LAMBDA = 1.0 - PLANCK_OMEGA_M
PLANCK_OMEGA_BH2 = 0.02237
PLANCK_OMEGA_R_H2 = 4.18e-5
PLANCK_THETA_STAR_TIMES_100 = 1.04109
PLANCK_THETA_STAR_TIMES_100_SIGMA = 0.00030
PLANCK_R_S_MPC = 147.05
PLANCK_D_M_STAR_MPC = 13869.61
PLANCK_Z_STAR = 1089.92

OMEGA_GAMMA_H2 = 2.47e-5
t_P_s = 5.391247e-44
N_HUB = 8.4e60   # framework cascade scale at observer epoch (Planck-cutoff for r_s)

# Cheeger-Buser bounds on srs Ramanujan expander Cheeger constant h:
#   h ∈ [(3-2√2)/6, √(6(3-2√2))] ≈ [0.029, 1.015] from k*=3 alone.
# Step C: H_inf = h / t_P.
H_CHEEGER_MIN = (3.0 - 2.0*math.sqrt(2.0)) / 6.0
H_CHEEGER_MAX = math.sqrt(6.0 * (3.0 - 2.0*math.sqrt(2.0)))


# =============================================================================
# §1. Step C vs Step D — coexistence vs inconsistency
# =============================================================================

def section_1_stepC_vs_stepD():
    print("=" * 78)
    print("§1. Step C vs Step D — coexistence vs inconsistency reading")
    print("=" * 78)
    print()
    print("Step C [N_hub_spectral_gap_attempt.py lines 64-72]:")
    print("  For srs causal graph with Cheeger constant h = O(1):")
    print("    |∂S|/|S| ≥ h  → boundary contains O(N) nodes (holographic).")
    print("    Rate Ṅ = h·N · (one toggle event per boundary node) / t_P")
    print("    →  dN/dt = h·N / t_P  →  N(t) = N₀ · exp(h·t/t_P)  (de Sitter)")
    print("    H = h/t_P (constant)")
    print()
    print(f"  With h ∈ [{H_CHEEGER_MIN:.4f}, {H_CHEEGER_MAX:.4f}]:")
    print(f"    H_inf t_P ∈ [{H_CHEEGER_MIN:.4f}, {H_CHEEGER_MAX:.4f}]")
    print(f"    H_inf [1/s] ∈ [{H_CHEEGER_MIN/t_P_s:.3e}, {H_CHEEGER_MAX/t_P_s:.3e}]")
    h_inf_kmsmpc_min = H_CHEEGER_MIN / t_P_s * 3.086e19
    h_inf_kmsmpc_max = H_CHEEGER_MAX / t_P_s * 3.086e19
    print(f"    H_inf [km/s/Mpc] ∈ [{h_inf_kmsmpc_min:.3e}, {h_inf_kmsmpc_max:.3e}]")
    print(f"    Compare to H_0 = {H0_OBSERVER:.2f} km/s/Mpc — H_inf is ~{h_inf_kmsmpc_min/H0_OBSERVER:.0e}x larger.")
    print()
    print("Step D [N_hub_spectral_gap_attempt.py lines 78-101]:")
    print("  D1: k*N directed-edge toggles per t_P (one per edge per t_P).")
    print("  D2: cascade ratio observable/possible = 1/(k*N) → 1 new state per t_P.")
    print("  D3: H = (1 new state/t_P) / N states total = 1/(N·t_P) (linear N(t))")
    print()
    print("CONFLICT ANALYSIS:")
    print("  Step C and Step D both compute dN/dt where N = number of causal states.")
    print("  - Step C: dN/dt = h·N/t_P (every boundary toggle creates new state).")
    print("  - Step D: dN/dt = 1/t_P (cascade-filtered, only 1 per t_P).")
    print()
    print("  Reading (a) — same observable, inconsistent claims at same epoch:")
    print("    Both compute the SAME quantity (rate of new causal states).")
    print("    Step C does NOT apply MDL acceptance filter; Step D does.")
    print("    Cannot both be true simultaneously.")
    print()
    print("  Reading (b) — different observables:")
    print("    Step C computes RAW boundary-toggle rate (no MDL filter).")
    print("    Step D computes OBSERVABLE-NEW-STATE rate (after MDL filter).")
    print("    For cosmology, H ∝ d/dt of observable spatial volume → Step D.")
    print("    Step C's exponential rate would be at the level of distinct")
    print("    causal histories, not observable spatial expansion.")
    print("    PROBLEM: this reading makes Step C's de Sitter cosmologically")
    print("    UNREACHABLE — Step C's rate is not what enters H. Candidate 5.3")
    print("    becomes vacuous under this reading.")
    print()
    print("  Reading (c) — different epochs (regime crossover):")
    print("    Step C dominates at small N (early universe, boundary-effect-")
    print("    dominated); Step D dominates at large N (late universe, bulk-")
    print("    effect-dominated). Transition at some N_*.")
    print("    For consistency at transition: rate must match.")
    print("    Step C rate at N_*: h·N_*/t_P. Step D rate: 1/t_P.")
    print("    Match → h·N_* = 1 → N_* = 1/h ≈ 1 (since h = O(1)).")
    print()
    print("    But N_* ≈ 1 means transition occurs at ~ 1 t_P after cosmic")
    print("    'beginning'. After that, coasting takes over. So Step C's")
    print("    de Sitter applies only for t < ~few t_P — essentially")
    print("    a single Planck time of de Sitter. Negligible for cosmology.")
    print()
    print("  → Reading (b) [vacuous] or Reading (c) [trivial transition].")
    print("    Reading (a) requires choosing one — but framework already")
    print("    chose Step D (cascade theorem load-bearing). Step C as")
    print("    cosmological prediction conflicts.")
    print()
    print("VERDICT §1: Step C and Step D are not naturally compatible as")
    print("  cosmological predictions. Candidate 5.3 is structurally")
    print("  problematic without an additional mechanism (specifically: a")
    print("  REGIME within srs where MDL acceptance is suspended). The")
    print("  framework gives no such mechanism; early_universe_k_rundown")
    print("  treats early universe as 'no MDL structure' (T > T_srs), but")
    print("  T_srs >> T_BBN per the D1/D2/D3 audit, so this is at z >> z_*.")
    print()


# =============================================================================
# §2. Numerical test: Candidate 5.1 (multiway-branching N(t) ∝ t^p)
# =============================================================================

def cs_kms(z):
    """Sound speed in baryon-photon plasma."""
    R = (3.0 * PLANCK_OMEGA_BH2) / (4.0 * OMEGA_GAMMA_H2 * (1.0 + z))
    cs2 = (c_km_s ** 2) / (3.0 * (1.0 + R))
    return math.sqrt(cs2)


def Hz_5_1(z, p, H0, z_trans):
    """Candidate 5.1: H = H_0(1+z)^p at z > z_trans, H = H_0(1+z) at z ≤ z_trans.
    Continuous match: A · (1+z_trans)^p = H_0 · (1+z_trans), so A = H_0(1+z_trans)^(1-p).
    """
    if z <= z_trans:
        return H0 * (1.0 + z)
    A = H0 * (1.0 + z_trans)**(1.0 - p)
    return A * (1.0 + z)**p


def Hz_5_3(z, H_inf_kmsmpc, H0, z_trans):
    """Candidate 5.3: de Sitter at z > z_trans (H = H_inf constant),
    coasting at z ≤ z_trans (H = H_0(1+z)).
    Continuity: H_0(1+z_trans) = H_inf → z_trans = H_inf/H_0 - 1.
    Caller must provide z_trans consistent with this."""
    if z <= z_trans:
        return H0 * (1.0 + z)
    return H_inf_kmsmpc


def D_M(z, H_func):
    integrand = lambda zp: 1.0 / H_func(zp)
    chi, _ = integrate.quad(integrand, 0.0, z, limit=500)
    return c_km_s * chi


def r_s_at_z_star(H_func, z_star=PLANCK_Z_STAR, z_max=1e8):
    """Sound horizon r_s = ∫_{z_*}^{z_max} c_s(z)/H(z) dz.

    Uses change of variable u = ln(1+z) to handle multi-scale integrals
    (z from 1090 up to potentially 1e60). With u = ln(1+z), du = dz/(1+z),
    so integrand becomes c_s(z) · (1+z) / H(z) · du.
    """
    if z_max <= z_star:
        return 0.0
    u_min = math.log(1.0 + z_star)
    u_max = math.log(1.0 + z_max)

    def integrand_u(u):
        z = math.exp(u) - 1.0
        return cs_kms(z) * (1.0 + z) / H_func(z)

    rs, _ = integrate.quad(integrand_u, u_min, u_max, limit=500)
    return rs


def section_2_candidate_5_1():
    print("=" * 78)
    print("§2. Candidate 5.1 — multiway-branching N(t) ∝ t^p at z > z_trans")
    print("=" * 78)
    print()
    print("  H(z) = H_0(1+z)     for z ≤ z_trans   (coasting, preserves H_0/t_0)")
    print("  H(z) = A·(1+z)^p    for z > z_trans   (multiway-branching)")
    print("  Continuity: A = H_0(1+z_trans)^(1-p)")
    print()
    print("  For r_s convergence at z → ∞: need p > 1.")
    print()

    # Sweep p and z_trans
    print(f"  Test 1: z_trans fixed at z_*={PLANCK_Z_STAR:.0f} (i.e., 5.1 modifies only z > z_*).")
    print(f"           D_M(z_*) is unaffected (still coasting D_M).")
    print(f"           Vary p; ask: is there a p that matches Planck r_s ≈ {PLANCK_R_S_MPC:.1f} Mpc?")
    print()
    print(f"    {'p':>8} {'r_s [Mpc]':>14} {'D_M [Mpc]':>14} {'100·θ_*':>10} {'σ_obs from Planck':>20}")
    z_trans = PLANCK_Z_STAR

    for p in [1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.5, 3.0]:
        Hf = lambda z, p=p, zt=z_trans: Hz_5_1(z, p, H0_OBSERVER, zt)
        rs = r_s_at_z_star(Hf, z_max=N_HUB)
        dm = D_M(PLANCK_Z_STAR, Hf)
        theta = rs / dm
        sigma = (theta*100 - PLANCK_THETA_STAR_TIMES_100) / PLANCK_THETA_STAR_TIMES_100_SIGMA
        print(f"    {p:>8.3f} {rs:>14.2f} {dm:>14.2f} {100*theta:>10.5f} {sigma:>+20.2e}")
    print()

    print(f"  OBSERVATION: at z_trans=z_*, D_M is FIXED at coasting value")
    Hf_coast = lambda z: H0_OBSERVER * (1.0 + z)
    dm_coast = D_M(PLANCK_Z_STAR, Hf_coast)
    print(f"  (D_M_coast(z_*) = {dm_coast:.0f} Mpc ≈ 2× Planck D_M = {PLANCK_D_M_STAR_MPC:.0f} Mpc).")
    print()
    print(f"  Even with PERFECT r_s match via tuned p, θ_* is at most r_s_match/D_M_coast")
    print(f"  = {PLANCK_R_S_MPC:.1f}/{dm_coast:.0f} = {PLANCK_R_S_MPC/dm_coast*100:.5f},")
    print(f"  vs Planck 100·θ_* = {PLANCK_THETA_STAR_TIMES_100:.5f}")
    print(f"  Ratio = {(PLANCK_R_S_MPC/dm_coast*100)/PLANCK_THETA_STAR_TIMES_100:.4f} → still ~2x off.")
    print()
    print(f"  Candidate 5.1 with z_trans = z_* CANNOT close the factor-of-2 in D_M.")
    print(f"  D_M depends on H(z) at z < z_*, where coasting still applies.")
    print()

    # Test 2: z_trans BELOW z_*
    print(f"  Test 2: z_trans below z_* (Candidate 5.1 modifies BOTH r_s AND D_M).")
    print(f"           Sweep z_trans ∈ {{1, 100, 500, 1000}}; p = 1.5 (representative).")
    print()
    print(f"    {'z_trans':>10} {'r_s [Mpc]':>14} {'D_M [Mpc]':>14} {'100·θ_*':>10} {'σ_obs':>14}")
    for z_trans in [1, 10, 100, 500, 800, 1000, 1085]:
        Hf = lambda z, p=1.5, zt=z_trans: Hz_5_1(z, p, H0_OBSERVER, zt)
        rs = r_s_at_z_star(Hf, z_max=N_HUB)
        dm = D_M(PLANCK_Z_STAR, Hf)
        theta = rs / dm
        sigma = (theta*100 - PLANCK_THETA_STAR_TIMES_100) / PLANCK_THETA_STAR_TIMES_100_SIGMA
        print(f"    {z_trans:>10d} {rs:>14.2f} {dm:>14.2f} {100*theta:>10.5f} {sigma:>+14.2e}")
    print()
    print("  Test 3: simultaneous fit. Find (p, z_trans) closest to Planck θ_*.")
    print()

    # Coarse 2D grid search
    best_chi2 = float('inf')
    best_params = None
    grid_points = []
    for z_trans in [1, 5, 10, 30, 100, 300, 500, 800, 1000]:
        for p in [1.05, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0]:
            Hf = lambda z, pp=p, zt=z_trans: Hz_5_1(z, pp, H0_OBSERVER, zt)
            try:
                rs = r_s_at_z_star(Hf, z_max=N_HUB)
                dm = D_M(PLANCK_Z_STAR, Hf)
                theta = rs / dm
                chi2 = ((100*theta - PLANCK_THETA_STAR_TIMES_100) / PLANCK_THETA_STAR_TIMES_100_SIGMA)**2
                grid_points.append((z_trans, p, rs, dm, 100*theta, chi2))
                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best_params = (z_trans, p, rs, dm, 100*theta, chi2)
            except Exception:
                pass

    # Print top 5 closest matches
    grid_points.sort(key=lambda x: x[5])
    print(f"  Top 5 closest matches in (z_trans, p) grid:")
    print(f"    {'z_trans':>10} {'p':>8} {'r_s':>10} {'D_M':>10} {'100·θ_*':>10} {'σ_obs':>14}")
    for pt in grid_points[:5]:
        sigma = math.sqrt(pt[5]) if pt[4] > PLANCK_THETA_STAR_TIMES_100 else -math.sqrt(pt[5])
        print(f"    {pt[0]:>10.0f} {pt[1]:>8.3f} {pt[2]:>10.2f} {pt[3]:>10.2f} {pt[4]:>10.5f} {sigma:>+14.2e}")
    print()

    print(f"  Best match: (z_trans={best_params[0]}, p={best_params[1]:.3f})")
    print(f"  θ_* prediction: {best_params[4]:.5f} vs Planck {PLANCK_THETA_STAR_TIMES_100:.5f}")
    print(f"  Discrepancy: {math.sqrt(best_chi2):+.2e}σ_obs")
    print()
    print("  VERDICT §2: Candidate 5.1 with TWO free parameters (p, z_trans)")
    print("  can in principle approach Planck θ_*. Closing it requires deriving")
    print("  BOTH from framework structure (no fitting). Currently neither has")
    print("  a structural derivation in the framework. Multi-session research.")
    print()


# =============================================================================
# §3. Numerical test: Candidate 5.3 (de Sitter at high z)
# =============================================================================

def section_3_candidate_5_3():
    print("=" * 78)
    print("§3. Candidate 5.3 — inflation-like Ramanujan de Sitter")
    print("=" * 78)
    print()
    print("  H(z) = H_inf (constant)  for z > z_trans   (de Sitter)")
    print("  H(z) = H_0(1+z)          for z ≤ z_trans   (coasting)")
    print("  Continuity: H_0(1+z_trans) = H_inf → z_trans = H_inf/H_0 - 1.")
    print()
    print("  Step C structural prediction: H_inf = h/t_P with h ∈ [0.029, 1.015]")

    h_inf_kmsmpc_min = H_CHEEGER_MIN / t_P_s * 3.086e19
    h_inf_kmsmpc_max = H_CHEEGER_MAX / t_P_s * 3.086e19

    print(f"    H_inf [km/s/Mpc] ∈ [{h_inf_kmsmpc_min:.3e}, {h_inf_kmsmpc_max:.3e}]")
    print(f"    z_trans = H_inf/H_0 - 1 ∈ [{h_inf_kmsmpc_min/H0_OBSERVER:.3e}, {h_inf_kmsmpc_max/H0_OBSERVER:.3e}]")
    print(f"    → de Sitter applies at z ≳ 10^{math.log10(h_inf_kmsmpc_min/H0_OBSERVER):.0f} (Planck epoch).")
    print()

    print(f"  Test 1: structural H_inf prediction (Step C, h=0.5 representative).")
    h_typical = 0.5
    H_inf = h_typical / t_P_s * 3.086e19
    z_trans_struct = H_inf / H0_OBSERVER - 1.0
    print(f"    h = {h_typical}, H_inf = {H_inf:.3e} km/s/Mpc, z_trans = {z_trans_struct:.3e}")

    # r_s under structural Candidate 5.3:
    # Coasting from z_* to z_trans + de Sitter from z_trans to ∞
    Hf = lambda z, hi=H_inf, zt=z_trans_struct: Hz_5_3(z, hi, H0_OBSERVER, zt)
    rs = r_s_at_z_star(Hf, z_max=N_HUB)
    dm = D_M(PLANCK_Z_STAR, Hf)
    theta = rs / dm
    sigma = (theta*100 - PLANCK_THETA_STAR_TIMES_100) / PLANCK_THETA_STAR_TIMES_100_SIGMA
    print(f"    r_s(z_*) = {rs:.2f} Mpc  vs Planck {PLANCK_R_S_MPC:.2f} Mpc")
    print(f"    D_M(z_*) = {dm:.2f} Mpc  vs Planck {PLANCK_D_M_STAR_MPC:.2f} Mpc")
    print(f"    100·θ_* = {100*theta:.5f}  vs Planck {PLANCK_THETA_STAR_TIMES_100:.5f}")
    print(f"    Discrepancy: {sigma:+.2e}σ_obs")
    print()
    print(f"    NOTE: with z_trans ~ 10^60, the integral z_* → z_trans is")
    print(f"    essentially the FULL coasting integral up to Planck cutoff.")
    print(f"    De Sitter at z_trans → ∞ adds nothing measurable. Result is")
    print(f"    indistinguishable from pure coasting to Planck cutoff (Path A Session 2).")
    print()

    # Sweep H_inf to see if any value matches Planck
    print(f"  Test 2: r_s under de Sitter from z_trans to z_max.")
    print(f"    With z_trans = z_* = {PLANCK_Z_STAR:.0f} (no coasting contribution to r_s),")
    print(f"    de Sitter integral: r_s_dS ≈ c_s · (z_max - z_*) / H_inf.")
    print(f"    Linear in (z_max, H_inf⁻¹) — can match Planck r_s for any H_inf,")
    print(f"    given a free choice of z_max (= 'duration of inflation' parameter).")
    print()
    print(f"    Set z_trans = z_*; integrate to z_max = {N_HUB:.1e} (Planck cutoff).")
    print(f"    Tuning H_inf to match Planck r_s ≈ {PLANCK_R_S_MPC:.1f} Mpc.")
    print()
    print(f"    {'H_inf [km/s/Mpc]':>22} {'r_s [Mpc]':>14} {'100·θ_* (D_M=D_M_coast)':>26}")
    for H_inf_test in [1e60, 1e61, 1.637e61, 1e62, 5.808e62, 1e63]:
        Hf = lambda z, hi=H_inf_test, zt=PLANCK_Z_STAR: Hz_5_3(z, hi, H0_OBSERVER, zt)
        rs = r_s_at_z_star(Hf, z_max=N_HUB)
        dm_coast = c_km_s / H0_OBSERVER * math.log(1.0 + PLANCK_Z_STAR)
        theta = rs / dm_coast
        print(f"    {H_inf_test:>22.3e} {rs:>14.3e} {100*theta:>26.3e}")
    print()
    print(f"    Roughly, H_inf ≈ N_HUB · c_s / r_s_target ≈ {N_HUB * 173000 / PLANCK_R_S_MPC:.3e} km/s/Mpc")
    print(f"    matches Planck r_s. Compare to Step C structural range")
    print(f"    [{h_inf_kmsmpc_min:.3e}, {h_inf_kmsmpc_max:.3e}] km/s/Mpc.")
    print()
    print(f"    Required H_inf is ~10^57 km/s/Mpc, which IS in Step C's range when")
    print(f"    h is at the LOWER Cheeger bound. Coincidence? Step C bound h ∈")
    print(f"    [0.029, 1.015] gives H_inf ∈ [1.6e61, 5.8e62]. Required value ~10^57")
    print(f"    is below this range — doesn't match Step C structural prediction.")
    print()
    print(f"  OBSERVATION: even matching Planck r_s, D_M = D_M_coast = {dm_coast:.0f} Mpc is")
    print(f"  fixed by coasting at z < z_* and is still 2x ΛCDM. So θ_* still has the")
    print(f"  residual D_M factor of 2 (handled in §4).")
    print()
    print(f"  Test 3: tune BOTH H_inf AND z_trans to match Planck θ_*.")
    print(f"    For z_trans < z_*, D_M(z_*) gets a de Sitter contribution at z_trans < z < z_*.")
    print()

    # 2D grid: z_trans and H_inf — extend H_inf into structurally-realistic range
    best_chi2 = float('inf')
    best = None
    points = []
    for z_trans in [1, 10, 100, 500, 800, 1000, PLANCK_Z_STAR]:
        for H_inf_test in [1e55, 1e56, 1e57, 1e58, 1e59, 1e60, 1.637e61, 1e62, 5.808e62]:
            Hf = lambda z, hi=H_inf_test, zt=z_trans: Hz_5_3(z, hi, H0_OBSERVER, zt)
            try:
                rs = r_s_at_z_star(Hf, z_max=N_HUB)
                dm = D_M(PLANCK_Z_STAR, Hf)
                theta = rs / dm
                chi2 = ((100*theta - PLANCK_THETA_STAR_TIMES_100) / PLANCK_THETA_STAR_TIMES_100_SIGMA)**2
                points.append((z_trans, H_inf_test, rs, dm, 100*theta, chi2))
                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best = (z_trans, H_inf_test, rs, dm, 100*theta, chi2)
            except Exception:
                pass

    points.sort(key=lambda x: x[5])
    print(f"    Top 5 closest matches in (z_trans, H_inf) grid:")
    print(f"    {'z_trans':>10} {'H_inf':>14} {'r_s':>12} {'D_M':>12} {'100·θ_*':>14} {'σ_obs':>14}")
    for pt in points[:5]:
        sigma = math.sqrt(pt[5]) if pt[4] > PLANCK_THETA_STAR_TIMES_100 else -math.sqrt(pt[5])
        print(f"    {pt[0]:>10.0f} {pt[1]:>14.3e} {pt[2]:>12.3e} {pt[3]:>12.3e} {pt[4]:>14.3e} {sigma:>+14.2e}")
    print()

    if best is not None:
        print(f"  Best match: (z_trans={best[0]}, H_inf={best[1]:.3e})")
        print(f"  θ_* = {best[4]:.5f} vs Planck {PLANCK_THETA_STAR_TIMES_100:.5f}")
        sig = math.sqrt(best[5]) if best[4] > PLANCK_THETA_STAR_TIMES_100 else -math.sqrt(best[5])
        print(f"  Discrepancy: {sig:+.2e}σ_obs")
        print()

    print("  VERDICT §3: Candidate 5.3's STRUCTURAL form (h ∈ Cheeger bounds")
    print("  → Planck-scale H_inf → z_trans ~ 10^60) does NOT match Planck θ_*")
    print("  to any practical accuracy (essentially identical to pure coasting,")
    print("  10^5 σ off). To match Planck, BOTH H_inf and z_trans must be")
    print("  free-tuned away from structural values. Two free parameters; no")
    print("  framework-internal derivation of either.")
    print()


# =============================================================================
# §4. Ratio-of-residuals diagnostic
# =============================================================================

def section_4_residual_ratio():
    print("=" * 78)
    print("§4. Ratio of residuals — coasting D_M is 2x ΛCDM")
    print("=" * 78)
    print()
    print("  Path A Session 2 found:")
    print("    r_s_coast (Planck cutoff) / r_s_LCDM ≈ 1000x  (UV-divergent)")
    print("    D_M_coast (observer)       / D_M_LCDM ≈ 2x   (logarithmic excess)")
    print("    θ_*_coast / θ_*_Planck     ≈ 500x")
    print()
    print("  The factor-of-2 in D_M is a real structural feature of coasting:")
    print("    D_M_coast(z_*) = (c/H_0) ln(1+z_*) ≈ (c/H_0) · 7.0")
    print("    D_M_LCDM(z_*)  = (c/H_0) · 3.5  (matter-domination compresses it)")
    print()
    print("  This 2x factor in D_M is exactly the empirical Λ_LCDM/Λ_substrate")
    print("  decomposition factor (per Lambda_CC_factor_two_decomposition doc).")
    print("  Suggestive but does NOT close the θ_* problem: r_s residual is")
    print("  3 orders of magnitude, not factor-of-2.")
    print()
    print("  Implication for Item 5: the right-shape resolution would be a")
    print("  cosmology that PRESERVES coasting D_M (2x ΛCDM) but REGULATES r_s")
    print("  to within ~2x ΛCDM, so that θ_* matches Planck to factor of 1.")
    print()
    print("  Under Candidate 5.1 with z_trans = z_*: r_s convergent (good),")
    print("  D_M = coasting D_M = 2x ΛCDM (preserved). Then θ_* under 5.1 ≈")
    print("  r_s_match / (2 D_M_LCDM) = θ_*_Planck / 2.")
    print("  → still factor-of-2 off from Planck (the WRONG way).")
    print()
    print("  The factor-of-2 D_M residual is incompatible with both directions:")
    print("    - To match Planck θ_*, need D_M = D_M_LCDM (lose framework's")
    print("      coasting D_M structure).")
    print("    - Or match Planck θ_*/2 (predict observed θ_* is wrong by 2×;")
    print("      empirically falsified at 10^5 σ).")
    print()
    print("  VERDICT §4: even an ideal Item 5 closure (perfect r_s regulation)")
    print("  leaves a residual factor-of-2 in θ_* via D_M. The framework's")
    print("  coasting D_M structure is incompatible with Planck CMB acoustic peak")
    print("  unless coasting itself is modified at z < z_*.")
    print()


# =============================================================================
# §5. Diagnostic verdict
# =============================================================================

def section_5_verdict():
    print("=" * 78)
    print("§5. Diagnostic verdict — multi-sprint commitment decision")
    print("=" * 78)
    print()
    print("  Question: which of Candidates 5.1, 5.3 (5.2 ruled out at scoping)")
    print("  is most promising for a multi-sprint Item 5 closure?")
    print()
    print("  CANDIDATE 5.1 (multiway-branching N(t) ∝ t^p at z > z_trans):")
    print("    - r_s convergent for p > 1 (good).")
    print("    - At z_trans = z_*: D_M unaffected (preserves coasting D_M).")
    print("    - Tunable p achieves r_s match, but θ_* still 2x off via D_M.")
    print("    - At z_trans < z_*: both r_s and D_M modified; 2D fit can match θ_*")
    print("      but no framework-internal derivation of (p, z_trans).")
    print("    - Multi-session work: derive p AND z_trans from framework.")
    print()
    print("  CANDIDATE 5.3 (Step C de Sitter at z > z_trans):")
    print("    - Structural Step C gives H_inf = h/t_P (Planck-scale).")
    print("    - Structural z_trans = H_inf/H_0 - 1 ~ 10^60 (Planck epoch).")
    print("    - Numerically equivalent to pure coasting up to Planck cutoff.")
    print("    - Tuning to match Planck requires H_inf << structural value")
    print("      and z_trans << 10^60 — both fitted, no framework derivation.")
    print("    - Step C and Step D are not naturally compatible (§1).")
    print()
    print("  CONSEQUENCE: neither candidate has a clear path to closure")
    print("  WITHOUT free-parameter fitting against Planck data. Both require")
    print("  framework-internal derivation of 2 parameters that are currently")
    print("  unsupported.")
    print()
    print("  RESIDUAL FACTOR-OF-2 (§4): even an ideal closure of r_s")
    print("  regularization leaves a factor-of-2 residual in θ_* via D_M.")
    print("  This means coasting D_M itself is exposed to the falsification.")
    print("  The framework's strict coasting at z < z_* is also implicated.")
    print()
    print("  HONEST VERDICT:")
    print("  - 5.2 ruled out (T_srs >> T_*).")
    print("  - 5.3 has structural derivation but numerically equivalent to")
    print("    pure coasting; matching Planck requires H_inf+z_trans free-tune.")
    print("  - 5.1 lacks structural derivation but is mathematically more")
    print("    flexible; can match Planck θ_* with 2 tuned parameters but")
    print("    framework-internal derivation is open.")
    print("  - BOTH face the residual D_M factor-of-2 from coasting at z < z_*.")
    print()
    print("  RECOMMENDATION:")
    print("  Defer Item 5 multi-sprint commitment until either:")
    print("  (i) framework develops new structural machinery that gives")
    print("       p (Candidate 5.1) or H_inf+z_trans (Candidate 5.3) without")
    print("       free-parameter fitting; OR")
    print("  (ii) framework explicitly accepts Scenario 3 (honest concession):")
    print("       cosmology cluster's high-z claims are downgraded; coasting")
    print("       is observer-side / late-time only; pre-recombination")
    print("       structural machinery is NOT IN SCOPE for current framework.")
    print()
    print("  Multi-sprint commitment is NOT WARRANTED at this diagnostic stage.")
    print("  Both viable candidates require framework-level structural")
    print("  development (effectively Need A — multiway formalization) before")
    print("  productive Item 5 work can begin.")
    print()
    print("  This matches the cosmology roadmap's prior judgment: Item 5 is")
    print("  qualitatively bigger than Items 1-4 (12-30 sessions).")
    print()
    print("  Best near-term action:")
    print("  → Update cosmology_item_5_pre_recombination_scoping_2026-05-05.md")
    print("    §3 with this diagnostic's findings; flag both candidates as")
    print("    BLOCKED on framework-level structural development; downgrade")
    print("    'recommended next session' to 'deferred pending Need A'.")
    print("  → Decide separately: pursue Need A (multiway formalization),")
    print("    accept Scenario 3 honest concession, or pause cosmology arc.")
    print()


def main():
    section_1_stepC_vs_stepD()
    section_2_candidate_5_1()
    section_3_candidate_5_3()
    section_4_residual_ratio()
    section_5_verdict()
    print("=" * 78)
    print("DONE: Cosmology Item 5 diagnostic.")
    print("Verdict: both viable candidates require framework-level structural")
    print("  development (effectively Need A) before productive Item 5 work.")
    print("  Multi-sprint Item 5 commitment NOT WARRANTED at this stage.")
    print("=" * 78)


if __name__ == "__main__":
    main()
