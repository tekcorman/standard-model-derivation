#!/usr/bin/env python3
"""
Canonical prediction file for theta_12 (PMNS solar mixing angle).

NOTE (post-A3, 2026-04-18): Historical pre-A3 two-axiom derivation,
BLOCKED under B6 and V_us. Under the three-axiom framework (A1+A2+A3;
docs/framework_axioms.md) G.1 and G.5 are DERIVED via CDP 2011
(predictions/observer_hilbert_space.py), but the B6 color-vs-generation
retraction and V_us block remain load-bearing here.

STATUS UNDER THE FRAMEWORK RIGOR BAR: BLOCKED (transitively on V_us).

See predictions/theta_12_PMNS_derivation.md for the full audit.
The numerical value produced below follows the Pati-Salam SU(4)
perpendicularity chain (Route B of predictions/V_us_derivation.md),
evaluated at the *bare* V_us = (2/3)^(2+sqrt(3)) to keep the script
upstream-identical to predictions/V_us.py and the Route-B arithmetic.
The value is quoted for downstream continuity; it is NOT theorem-grade.

Every gap inherited from V_us (Route-B gaps B-Gap 1, 2, 3; Route-A
gaps A1, A2) individually BLOCKS theorem-grade closure of theta_12
under docs/parameter_linter.md "Hard quality gate".
"""

# ============================================================
# PARAMETER: theta_12 (PMNS solar mixing angle)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       33.68 +/- 0.73 deg (NuFIT 6.0, normal ordering,
#              September 2024; equivalently sin^2(theta_12) = 0.308
#              +0.012/-0.011).
# Source:      NuFIT 6.0 (Esteban et al.); consistent with PDG 2024
#              global fit.
# PDG edition: 2024

# --- PREDICTED VALUE -----------------------------------------
# Value (Route B, provisional): 33.17 deg
#                               (cos(theta_TBM)/cos(theta_C),
#                                theta_C = arcsin(V_us_bare))
# Deviation:                    -0.70 sigma vs NuFIT 6.0 central
#                               (using sigma = 0.73 deg)
# Status:                       BLOCKED under docs/parameter_linter.md
#                               "Hard quality gate" -- see
#                               predictions/theta_12_PMNS_derivation.md

# --- DERIVED FORMULA -----------------------------------------
# Type D per docs/W4_identification_catalog.md §2D (mixing angle =
# TBM baseline + dark correction).  The derivation chain currently
# implemented is the Pati-Salam SU(4) perpendicularity route of
# predictions/V_us.py ("Route B"):
#
#   theta_TBM = arctan(1/sqrt(k*-1))                         [adopted, B-Gap 4]
#   V_us_bare = ((k*-1)/k*)^(2 + E_P) = (2/3)^(2+sqrt(3))    [adopted, A1]
#   theta_C   = arcsin(V_us_bare)                            [adopted]
#   cos(theta_12) = cos(theta_TBM) / cos(theta_C)            [adopted, B-Gap 2]
#   theta_12  = arccos(cos(theta_TBM)/cos(theta_C))
#
# Each step is explicit arithmetic GIVEN its inputs; the inputs
# themselves are either closed upstream (k*, E_P, alpha_1) or
# flagged as BLOCKED in predictions/V_us_derivation.md §2.3.
#
# Why this file stays BLOCKED:
#   * The spherical Pythagorean step (non-abelian SU(4) manifold;
#     rank-3 not rank-1) is unproven -- see B-Gap 2 in V_us.md.
#   * The walk length L_us = 2 + sqrt(3) inside V_us_bare has no
#     upstream derivation -- see A-Gap 1 in V_us.md.
#   * There is no V_us-free direct spectral computation of
#     theta_12 in the codebase (the 1-2 sector is not
#     C_3-degenerate at TBM the way the 2-3 sector is at P,
#     so the theta_23-style mass^2-class Feshbach split does
#     not apply verbatim -- see .md Section 5 for discussion).
#
# The new closures available since the prior draft --
#   * docs/theorem_uniform_Q_density.md Part A (rho_Q uniform: theorem)
#   * docs/theorem_Feshbach_coupling_strength.md
#     (alpha_1 = (2/3)^(g-2) from the Exponent Principle + Lemma 1)
# -- tighten upstream dependencies for Class-1/Class-2 corrections
# in general, but do NOT close the V_us dependency that this file
# transitively sits on.  They also do not replace the missing
# non-abelian spherical Pythagorean proof.

# --- INPUTS --------------------------------------------------
# symbol      | value           | status    | predictions/ file             | meaning
# ------------|-----------------|-----------|-------------------------------|--------
# k_star      | 3               | [derived] | predictions/k_star.py         | coordination number
# d_spatial   | 3               | [derived] | predictions/d_spatial.py      | spatial dimension
# E_P         | sqrt(k*)=sqrt 3 | [derived] | predictions/srs_E_at_P.py     | A(P) eigenvalue
# V_us_bare   | (2/3)^(2+E_P)   | [BLOCKED] | predictions/V_us.py (Route A  | bare Cabibbo amplitude
#             |                 |           |   numerator; V_us.md §2.1)    |
# theta_TBM   | atan(1/sqrt 2)  | [adopted] | W4 catalog §2D P5 + P1/P2     | TBM solar baseline
#             |                 |           | (theta_12 = hypotenuse)       |

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from k_star import predict_k_star
from d_spatial import predict_d_spatial
from srs_E_at_P import predict_srs_E_at_P

# Upstream (all closed):
d = predict_d_spatial()
k = predict_k_star(d)
E_P = predict_srs_E_at_P(k)  # = sqrt(k*) = sqrt(3)

# V_us_bare: the Route-A "bare" amplitude (2/3)^(2+E_P) used inside
# the theta_12 Route-B chain.  The walk distance L_us = 2 + E_P is
# itself BLOCKED (A-Gap 1 in predictions/V_us_derivation.md); we
# evaluate it here only for continuity with the provisional Route-B
# number that appears in derivations.md §8 and in
# proofs/foundations/theta_12_PMNS_derivation.py.
V_us_bare = ((k - 1) / k) ** (2 + E_P)

# Route-B perpendicularity arithmetic (provisional, BLOCKED).
theta_TBM = math.atan(1.0 / math.sqrt(k - 1))  # adopted; B-Gap 4
theta_C = math.asin(V_us_bare)                 # adopted
cos_ratio = math.cos(theta_TBM) / math.cos(theta_C)   # B-Gap 2
theta_12_rad = math.acos(cos_ratio)
theta_12_deg = math.degrees(theta_12_rad)

print("theta_12 canonical prediction -- STATUS: BLOCKED (transitively on V_us)")
print(f"  k* = {k}, E_P = sqrt(k*) = {E_P:.10f}")
print(f"  V_us_bare  = ((k*-1)/k*)^(2+E_P) = (2/3)^(2+sqrt 3) = {V_us_bare:.10f}")
print(f"  theta_TBM  = arctan(1/sqrt({k-1}))   = {math.degrees(theta_TBM):.6f} deg")
print(f"  theta_C    = arcsin(V_us_bare)       = {math.degrees(theta_C):.6f} deg")
print(f"  cos ratio  = cos(theta_TBM)/cos(theta_C) = {cos_ratio:.10f}")
print(f"  theta_12   = arccos(.)               = {theta_12_deg:.6f} deg"
      "  [provisional; not theorem-grade]")
print("  See predictions/theta_12_PMNS_derivation.md for the chain of open gaps.")


# --- PURE FUNCTION -------------------------------------------
# No hardcoded physical constants: k_star and E_P are named
# parameters; V_us_bare is computed from them via the Route-A
# identification.  The only literals inside the body are pure
# mathematical constants (1, 2) implicit in sqrt / trig / arithmetic.

def predict_theta_12_PMNS(k_star, E_P):
    """
    Provisional Route-B implementation of theta_12 (PMNS solar angle).

    Implements the (currently BLOCKED) chain documented in
    predictions/theta_12_PMNS_derivation.md:

        V_us_bare = ((k_star - 1) / k_star) ** (2 + E_P)
        theta_TBM = arctan(1 / sqrt(k_star - 1))
        theta_C   = arcsin(V_us_bare)
        cos(theta_12) = cos(theta_TBM) / cos(theta_C)
        theta_12  = arccos(.)                   # returned in degrees

    Every one of these steps inherits a gap from
    predictions/V_us_derivation.md (A-Gap 1 for L_us = 2 + E_P;
    B-Gap 2 for the spherical Pythagorean identity; B-Gap 4 for
    sin^2(theta_TBM) = 1/k_star as a Type-D postulate).

    Parameters
    ----------
    k_star : int
        Coordination number of the MDL-optimal regular net
        (derived value: k_star = 3, from predictions/k_star.py).
    E_P : float
        Adjacency eigenvalue of the scalar Bloch matrix A(P) at
        the P-point (derived value: sqrt(k_star) = sqrt(3), from
        predictions/srs_E_at_P.py).

    Returns
    -------
    float
        Route-B provisional value of theta_12 in degrees.
        Not theorem-grade.
    """
    V_us_bare_local = ((k_star - 1) / k_star) ** (2 + E_P)
    theta_TBM_local = math.atan(1.0 / math.sqrt(k_star - 1))
    theta_C_local = math.asin(V_us_bare_local)
    cos_ratio_local = math.cos(theta_TBM_local) / math.cos(theta_C_local)
    return math.degrees(math.acos(cos_ratio_local))


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_result = theta_12_deg
    pure_result = predict_theta_12_PMNS(k, E_P)
    print()
    print(f"Implementation: {impl_result:.10f} deg")
    print(f"Pure function:  {pure_result:.10f} deg")
    assert abs(impl_result - pure_result) < 1e-10, \
        f"Mismatch: {impl_result} vs {pure_result}"
    obs = 33.68
    sigma = 0.73
    dev = (pure_result - obs) / sigma
    print("OK: outputs agree.")
    print(f"    theta_12 (Route B, provisional) = {pure_result:.4f} deg")
    print(f"    NuFIT 6.0                       = {obs} +/- {sigma} deg")
    print(f"    Deviation                       = {dev:+.2f} sigma")
    print("    Rigor status: BLOCKED -- transitively on V_us "
          "(predictions/V_us_derivation.md).")
    print("    Additional B6 blocker (2026-04-17): TBM baseline "
          "treats C_3 irreps as generation labels;")
    print("    B6 (docs/theorem_B6_bridge.md) identifies C_3 as "
          "color-Z_3 of SU(3)_c, not generation.")
    print("    Re-derivation target: Sprint 11 B7.5 "
          "(docs/master_plan.md), PMNS under C^3_gen framework.")
