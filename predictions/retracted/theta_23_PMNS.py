#!/usr/bin/env python3
"""
Canonical prediction file for theta_23 (PMNS atmospheric mixing angle).

Type-D Class-2 (mass-squared) entry of the W4 identification catalog
(docs/W4_identification_catalog.md §2D), now theorem-grade modulo the
adopted Exponent Principle of complete_physics_derivations.md §45.

NOTE (post-A3, 2026-04-18): Historical pre-A3 two-axiom derivation,
BLOCKED under B6 (color-vs-generation retraction). Under the three-axiom
framework (A1+A2+A3; docs/framework_axioms.md) G.1 and G.5 are DERIVED via
CDP 2011 (predictions/observer_hilbert_space.py), but B6 remains
load-bearing here.
"""

# ============================================================
# PARAMETER: theta_23  (PMNS atmospheric mixing angle)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       49.2 +/- 1.0 deg (NuFIT 6.0, normal ordering, September 2024)
# Source:      NuFIT 6.0 (Esteban, Gonzalez-Garcia, Maltoni,
#              Schwetz, Zhou 2024), online tables.
# PDG edition: PDG 2024 consistent within the same band.

# --- PREDICTED VALUE -----------------------------------------
# Value:       theta_23 = arctan((1 + a)/(1 - a)) with a = (5/3)*(2/3)^8
#            = 48.7207 deg
# Deviation:   49.2 - 48.72 = 0.48 deg = 0.48 sigma (within 1 sigma).

# --- DERIVED FORMULA -----------------------------------------
#
# theta_23 = arctan( (1 + alpha_1_full) / (1 - alpha_1_full) )
#
# alpha_1_full = ( Im(h)^2 / Re(h)^2 ) * alpha_1_bare
#              = (5/3) * ((k* - 1)/k*)^(g - 2)
#              = (5/3) * (2/3)^8
#              = 1280 / 19683
#
# Logical chain (each step either an axiom, an upstream predictions
# file, a cited theorem document, or explicit algebra):
#
#   Step A  -- TBM baseline theta_23_TBM = 45 deg.
#              docs/theorem_BP_doubly_degenerate_h.md Step 3: at the
#              P-point the scalar Bloch adjacency A(P) has characteristic
#              polynomial (lambda^2 - 3)^2, so its +sqrt(3) and
#              -sqrt(3) eigenspaces are each C_3-invariant and
#              2-dimensional.  Ihara-Bass (Terras 2011 Thm 2.3) lifts
#              this to B(P): the eigenvalues h = (sqrt(3)+i sqrt(5))/2
#              and h-bar each appear with multiplicity exactly 2,
#              C_3-protected.  The mu_2 / mu_3 ratio is |h-bar| / |h| = 1
#              exactly, forcing tan(theta_23_TBM) = 1.
#
#   Step B  -- Class-2 (mass^2) perturbation structure with coefficient
#              Im(h)^2 / Re(h)^2 = 5/3.  Traced step by step in
#              predictions/dark_extraction_map_derivation.md Class 2:
#              the self-energy Sigma(h) transforms as omega^2 under C_3,
#              so single-insertion first-order contributions vanish by
#              C_3 selection on the (C_3-trivial) angle observable; the
#              leading effect is through the Hermitian decomposition
#              B = B_sym + i B_anti with B_sym ~ Re(h), B_anti ~ Im(h).
#              Degenerate perturbation theory (Sakurai, Modern QM, 3rd
#              ed. Ch 5.2) on the degenerate nu_2/nu_3 block gives
#              Delta theta_23 = (Im(h)^2) / (Re(h)^2) * alpha_1_bare
#              = tan^2(arg h) * alpha_1_bare.
#
#   Step C  -- Density shape of the resolvent integral inside Sigma(h):
#              Theorem A of docs/theorem_uniform_Q_density.md.  The
#              Q-space spectral density is uniform on the Ramanujan
#              circle |lambda|^2 = k* - 1 at MDL optimum, up to
#              O(sqrt(log N / N)).  Uniform density + contour integral
#              (standard Poisson-kernel evaluation, e.g. Stein-Shakarchi
#              Complex Analysis Ch. 2 Ex. 19) yields Sigma(h) = alpha_1/h
#              with no extra shape factors.  This is the step that
#              licenses the Im(h)/|h|^2 factor used in Step B.
#
#   Step D  -- Coupling magnitude alpha_1_bare = ((k*-1)/k*)^(g-2).
#              docs/theorem_Feshbach_coupling_strength.md Lemma 1 proves
#              tree NB-survival on the universal cover gives factor
#              (k*-1)/k* per step; the Exponent Principle
#              (complete_physics_derivations.md Section 45 Result 45.1),
#              an adopted structural theorem, identifies the relevant
#              internal length as g - n_fixed with n_fixed = 2 for a
#              scattering amplitude between two fixed external edges
#              (P-space entry and exit).  So alpha_1_bare = (2/3)^8.
#
#   Step E  -- Symmetric splitting (degenerate perturbation theory,
#              Sakurai Ch 5.2): the sigma_z structure of delta M^2 in
#              Step B splits the degenerate pair as
#              lambda_+ = lambda_0 (1 + a), lambda_- = lambda_0 (1 - a)
#              with a = alpha_1_full.
#
#   Step F  -- Assembling Steps A-E: theta_23 = arctan(lambda_+/lambda_-)
#              = arctan((1 + a)/(1 - a)).

# --- INPUTS --------------------------------------------------
# symbol          | value         | status    | predictions/ file                  | meaning
# ----------------|---------------|-----------|-----------------------------------|-----------------------------
# k_star          | 3             | [derived] | predictions/k_star.py             | coordination number
# d_spatial       | 3             | [derived] | predictions/d_spatial.py          | spatial dimension
# g_girth         | 10            | [derived] | predictions/g_girth.py            | srs girth
# h               | (sqrt(3)+i sqrt(5))/2 | [derived] | predictions/h_walker_eigenvalue.py | Hashimoto eigenvalue at P
# alpha_1_bare    | (2/3)^8       | [derived] | predictions/alpha_1.py            | tree NB survival; Lemma 1 + Exponent Principle
# Class-2 coeff.  | 5/3           | [derived] | predictions/dark_extraction_map_derivation.md Class 2 | Im(h)^2 / Re(h)^2
#
# Adopted structural content (tier below theorem, per W4 catalog section 3):
#   P1  --  physical observables live on the Ramanujan subspace of B(P).
#   P2  --  sqrt(multiplicity) coherent aggregation.
#   Exponent Principle  --  n_steps = g - n_fixed.  Numerically verified
#                           on K_4 and srs (proofs/foundations/hashimoto_exponents.py,
#                           proofs/foundations/exponent_ladder.py).

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from k_star import predict_k_star
from d_spatial import predict_d_spatial
from srs_E_at_P import predict_srs_E_at_P
from g_girth import predict_g_girth
from h_walker_eigenvalue import predict_h_walker_eigenvalue
from alpha_1 import predict_alpha_1
from dark_extraction_map import dark_coefficient_mass_squared

# Upstream values (all axiomatically derived; no numerical inputs).
d = predict_d_spatial()
k = predict_k_star(d)
E = predict_srs_E_at_P(k)
h = predict_h_walker_eigenvalue(k, E)
g = predict_g_girth(k, d)
alpha_1_bare = predict_alpha_1(k, g)          # (2/3)^8  (Lemma 1 + Exponent Principle)
c_mass_sq = dark_coefficient_mass_squared(h)  # Im(h)^2 / Re(h)^2 = 5/3

alpha_1_full = c_mass_sq * alpha_1_bare       # (5/3) * (2/3)^8 = 1280/19683

# TBM baseline (theorem_BP Step 3 + Ihara-Bass => tan(theta_23_TBM) = 1).
theta_23_TBM_rad = math.pi / 4

# Symmetric splitting of the mu_2 / mu_3 degenerate pair
# (Sakurai, Modern QM, 3rd ed. Ch 5.2, degenerate perturbation theory).
theta_23_rad = math.atan2(1 + alpha_1_full, 1 - alpha_1_full)
theta_23_deg = math.degrees(theta_23_rad)

print(f"k*              = {k}")
print(f"g               = {g}")
print(f"h               = Re(h) + i Im(h) = {h.real:.10f} + {h.imag:.10f} i")
print(f"|h|^2           = {abs(h)**2:.10f}  (== k* - 1 = {k - 1})")
print(f"alpha_1_bare    = ((k-1)/k)^(g-2) = {alpha_1_bare:.10f}")
print(f"5/3  coefficient = Im(h)^2 / Re(h)^2 = {c_mass_sq:.10f}")
print(f"alpha_1_full    = (5/3) * alpha_1_bare = {alpha_1_full:.10f}")
print()
print(f"theta_23_TBM    = {math.degrees(theta_23_TBM_rad):.6f} deg")
print(f"Delta theta_23  ~ alpha_1_full rad = {math.degrees(alpha_1_full):.6f} deg")
print(f"theta_23        = arctan((1+a)/(1-a)) = {theta_23_deg:.6f} deg")


# --- PURE FUNCTION -------------------------------------------
# No hard-coded physical constants. All upstream-derived numerical
# inputs (k_star, girth, h, and the Class-2 coefficient implicit in
# Im(h)/Re(h)) enter through the signature.

def predict_theta_23_PMNS(k_star, girth, h_eigenvalue):
    """
    Compute theta_23 (PMNS atmospheric mixing angle) in degrees from
    the srs spectral data at the P-point.

    The formula is
        theta_23 = arctan( (1 + a) / (1 - a) ),
        a = (Im(h)^2 / Re(h)^2) * ((k_star - 1)/k_star)^(girth - 2).

    Derivation sketch (see predictions/theta_23_PMNS_derivation.md):
      -  TBM baseline theta_23_TBM = 45 deg from C_3-protected double
         degeneracy of A(P) (theorem_BP_doubly_degenerate_h.md).
      -  Class-2 Hermitian decomposition of the walker operator gives
         mass^2 perturbation with parity-even / parity-odd strengths
         Re(h)^2 and Im(h)^2; degenerate perturbation theory splits the
         degenerate nu_2/nu_3 pair symmetrically with coefficient
         Im(h)^2 / Re(h)^2.
      -  The resolvent shape factor 1/h in Sigma(h) is licensed by
         uniform Q-space density (theorem_uniform_Q_density.md Theorem A).
      -  The overall magnitude alpha_1_bare = ((k-1)/k)^(g-2) is Lemma 1
         of theorem_Feshbach_coupling_strength.md + the Exponent
         Principle (n_fixed = 2 for scattering).

    Parameters
    ----------
    k_star : int
        Coordination number of the MDL-optimal walker graph.  Derived
        value on srs: 3.
    girth : int
        Girth of the MDL-optimal walker graph.  Derived value on srs: 10.
    h_eigenvalue : complex
        Hashimoto eigenvalue h at the P-point.  Derived value on srs:
        (sqrt(3) + i sqrt(5)) / 2.

    Returns
    -------
    float
        theta_23 in degrees.
    """
    # Bare scattering coupling: (k*-1)/k* per internal NB step, raised
    # to n_steps = girth - n_fixed with n_fixed = 2 (scattering).
    alpha_bare = ((k_star - 1) / k_star) ** (girth - 2)
    # Class-2 coefficient: ratio of parity-odd to parity-even channel
    # strengths, = Im(h)^2 / Re(h)^2.  See dark_extraction_map_derivation.md
    # Class 2 for the b_0 = 1/2 bookkeeping that yields this form.
    c = (h_eigenvalue.imag ** 2) / (h_eigenvalue.real ** 2)
    a = c * alpha_bare
    # Symmetric splitting of the TBM-degenerate nu_2 / nu_3 pair:
    # tan(theta_23) = (1 + a) / (1 - a).
    return math.degrees(math.atan2(1 + a, 1 - a))


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("STATUS: BLOCKED under Theorem B6 retraction (2026-04-17)")
    print("See predictions/theta_23_PMNS_derivation.md §Status")
    print("Re-derivation target: Sprint 11 workstream B7.3/B7.5")
    print("(mass operator + PMNS on C^3_gen; docs/master_plan.md §Sprint 11)")
    print("Step 1 identifies nu_2, nu_3 with h, h* eigenspaces via C_3-charged")
    print("irreps {omega, omega^2}; B6 proves these are color labels.")
    print("Spectral lemma tan(theta) = (1+a)/(1-a) is preserved as color-")
    print("sector arithmetic; atmospheric-angle identification is retracted.")
    print("=" * 60)
    impl_result = theta_23_deg
    pure_result = predict_theta_23_PMNS(k, g, h)
    print()
    print(f"Implementation: {impl_result:.10f} deg")
    print(f"Pure function:  {pure_result:.10f} deg")
    assert abs(impl_result - pure_result) < 1e-10, \
        f"Mismatch: {impl_result} vs {pure_result}"
    obs, sigma = 49.2, 1.0
    dev = (pure_result - obs) / sigma
    print(f"Observed:       {obs} +/- {sigma} deg (NuFIT 6.0, NO)")
    print(f"Deviation:      {pure_result - obs:+.3f} deg = {dev:+.2f} sigma")
    print("OK: outputs agree.")
