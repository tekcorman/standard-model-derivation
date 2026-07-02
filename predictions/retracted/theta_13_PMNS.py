#!/usr/bin/env python3
"""
Canonical prediction file for theta_13 (PMNS reactor mixing angle).

NOTE (post-A3, 2026-04-18): Historical pre-A3 two-axiom derivation,
BLOCKED under B6 and V_us. Under the three-axiom framework (A1+A2+A3;
docs/framework_axioms.md) G.1 and G.5 are DERIVED via CDP 2011
(predictions/observer_hilbert_space.py), but B6 retraction and V_us block
remain load-bearing here.
"""

# ============================================================
# PARAMETER: theta_13 (PMNS reactor mixing angle)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       8.57° ± 0.13° (NuFIT 6.0, normal ordering, 2024;
#              equivalently sin²(θ_13) = 0.02219 ± 0.00062)
# Source:      NuFIT 6.0, September 2024 (consistent with PDG 2024)
# PDG edition: 2024

# --- PREDICTED VALUE -----------------------------------------
# Value:       9.24°  (using V_us = 0.2271 from predictions/V_us.py
#              and sin(θ_13) = V_us / √(k*-1))
# Deviation:   +0.67° absolute, +5.2σ from NuFIT 1σ band.
#              (Open question: alternative route — V_us_bare·(1-α_1)/√2
#               with V_us_bare = (2/3)^(2+√3) — gives 8.61°, within 1σ.
#               The tension between the two routes is the same tension
#               already flagged as Open Question 3 in predictions/V_us_derivation.md
#               and is not specific to θ_13. See the derivation .md.)

# --- DERIVED FORMULA -----------------------------------------
# sin(θ_13) = V_us / √(k*-1)
#
# Derivation chain:
#   1. TBM baseline θ_13 = 0 (theorem).
#      At the P-point the scalar Bloch adjacency A(P) has spectrum
#      (±√3)² (each eigenvalue doubly degenerate).  The C₃-protection
#      of this degeneracy (docs/theorem_BP_doubly_degenerate_h.md
#      Step 3) forces the third column of the mixing matrix to take
#      the tribimaximal form (0, 1/√(k*-1), 1/√(k*-1)) in the k*=3
#      case.  Hence U_TBM(e,3) = 0, i.e. θ_13 = 0 at TBM level.
#   2. PMNS = U_l^† · U_TBM.  Therefore
#        U_PMNS(e,3) = (U_l^†(e,1)·0 + U_l^†(e,2)·1/√2 + U_l^†(e,3)·1/√2)
#                    ≈ U_l(2,1)/√(k*-1)   (small-mixing limit, U_l real).
#      In the charged-lepton basis, |U_l(2,1)| is the 1-2 entry of the
#      down-sector / charged-lepton Cabibbo rotation, identified with
#      V_us by the Pati–Salam SU(4) perpendicularity argument used in
#      predictions/V_us.py.
#   3. Dark correction class: θ_13 is measured at the C₃-symmetric
#      e-row vertex of the TBM third column — an EDGE-LOCAL observable
#      in the W4 catalog Type D Class 3 (docs/W4_identification_catalog.md
#      §2D).  The three C₃ images of the σ_x parity-mixing operator
#      sum to zero at that vertex by character orthogonality
#      (Serre 1977 §2.4 Theorem 3), so Tr(σ_x)=0 kills the Im(h)
#      enhancement that acts on mass²-class Class-2 observables.
#      The surviving correction has coefficient c=1 and is absorbed
#      already into V_us itself (same Class 3 mechanism, see
#      predictions/V_us_derivation.md §Step 4), so no additional
#      (1-α_1) factor is applied on top of V_us here.
#   4. Numerical evaluation uses V_us and k* from the canonical
#      predictions/ files; no fitted parameters.

# --- INPUTS --------------------------------------------------
# symbol | value  | status    | predictions/ file          | meaning
# -------|--------|-----------|----------------------------|--------
# k_star | 3      | [derived] | predictions/k_star.py      | coordination number
# V_us   | 0.2271 | [derived] | predictions/V_us.py        | Cabibbo / (1,2) mixing

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k_star import predict_k_star
from d_spatial import predict_d_spatial
from alpha_1 import predict_alpha_1
from g_girth import predict_g_girth
from V_us import predict_V_us

d = predict_d_spatial()
k = predict_k_star(d)
g = predict_g_girth(k, d)
a1 = predict_alpha_1(k, g)
# V_us is BLOCKED (tree-level = 0 under B3 + Type A); signature updated
# from predict_V_us(k, a1) -> predict_V_us(mu_triv, mu_omega, mu_omega2).
# The BLOCKED numerical value is 0.0; the derivation file documents the
# structural reason.  Kept as an import so the dependency chain remains
# visible for the Sprint 11 re-derivation.
V_us_val = predict_V_us(4, 2, 2)

# TBM third column is (0, 1/√(k*-1), 1/√(k*-1)); charged-lepton mixing
# projects V_us onto (e,3) through the 1/√(k*-1) factor.
sin_theta_13 = V_us_val / math.sqrt(k - 1)
theta_13_rad = math.asin(sin_theta_13)
theta_13_deg = math.degrees(theta_13_rad)

print(f"k*          = {k}")
print(f"V_us        = {V_us_val:.10f}  (from predictions/V_us.py)")
print(f"sin(θ_13)   = V_us / √(k*-1) = {sin_theta_13:.10f}")
print(f"θ_13        = arcsin(V_us / √(k*-1)) = {theta_13_deg:.6f}°")


# --- PURE FUNCTION -------------------------------------------

def predict_theta_13_PMNS(k_star, V_us):
    """
    Computes the PMNS reactor mixing angle theta_13.

    TBM baseline θ_13 = 0 (theorem from the C₃-protected double
    degeneracy of A(P) on srs; see docs/theorem_BP_doubly_degenerate_h.md).
    The reactor angle is generated by the charged-lepton rotation
    U_l applied to the TBM third column (0, 1/√(k*-1), 1/√(k*-1)),
    giving sin(θ_13) = |U_l(1,2)| / √(k*-1) in the small-mixing
    limit.  Under Pati–Salam perpendicularity the (1,2) entry of
    U_l is identified with V_us, and the Class 3 edge-local dark
    correction with coefficient c = 1 (Tr(σ_x) = 0 by character
    orthogonality at the C₃-symmetric vertex) is already carried
    by V_us itself — see predictions/V_us.py and
    docs/W4_identification_catalog.md §2D Class 3.

    Parameters
    ----------
    k_star : int
        Coordination number of the MDL-optimal crystal net (srs).
    V_us : float
        Cabibbo-angle magnitude |V_us|, as produced by
        predictions/V_us.py (includes the Class 3 dark correction).

    Returns
    -------
    float
        theta_13 in degrees.
    """
    return math.degrees(math.asin(V_us / math.sqrt(k_star - 1)))


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("STATUS: BLOCKED under Theorem B6 retraction (2026-04-17)")
    print("See predictions/theta_13_PMNS_derivation.md §Status")
    print("Re-derivation target: Sprint 11 workstream B7.5")
    print("(PMNS under C^3_gen; docs/master_plan.md §Sprint 11)")
    print("Step 1 treats C_3-charged eigenvectors |omega>, |omega^2> as")
    print("generation-mixing components; B6 proves these are color labels.")
    print("Also transitively blocked on V_us (independently BLOCKED).")
    print("=" * 60)
    impl_result = theta_13_deg
    pure_result = predict_theta_13_PMNS(k, V_us_val)
    print(f"\nImplementation: {impl_result:.10f}°")
    print(f"Pure function:  {pure_result:.10f}°")
    assert abs(impl_result - pure_result) < 1e-10, \
        f"Mismatch: {impl_result} vs {pure_result}"
    obs = 8.57
    sigma = 0.13
    dev = (pure_result - obs) / sigma
    print(f"OK: outputs agree.")
    print(f"    θ_13 = {pure_result:.4f}° (obs: {obs} ± {sigma}°, {dev:+.1f}σ)")
