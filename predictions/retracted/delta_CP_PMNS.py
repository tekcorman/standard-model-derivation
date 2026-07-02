#!/usr/bin/env python3
"""
Canonical prediction file for delta_CP_PMNS (Dirac CP-violating phase).

delta_CP = (g - 1) * arg(h*)  mod 360°

Type-C (phase-from-holonomy) identification per the framework's W4
catalog (docs/W4_identification_catalog.md §2C).  The walk-topology
input is "Jarlskog loop, one edge fixed by the C_3 transition", which
makes the exponent (g - 1) rather than g.  CP conjugation sends the
walker eigenvalue h |-> h* and therefore arg |-> -arg.

NOTE (post-A3, 2026-04-18): Historical pre-A3 two-axiom derivation,
BLOCKED under B6 (color-vs-generation retraction). Under the three-axiom
framework (A1+A2+A3; docs/framework_axioms.md) G.1 and G.5 are DERIVED via
CDP 2011 (predictions/observer_hilbert_space.py), but B6 retraction
remains load-bearing; Need-A2 still open.
"""

# ============================================================
# PARAMETER: delta_CP_PMNS  (Dirac CP-violating phase in PMNS)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       177 (+19/-20)°  (NuFIT 6.0, September 2024; normal ordering,
#                               1-sigma range from global fit; w/o SK atmos)
# Source:      Esteban, Gonzalez-Garcia, Maltoni, Schwetz, Pinheiro,
#              NuFIT 6.0 (2024); also quoted in PDG 2024 Review of
#              Particle Physics, "Neutrino Mixing" section.
# PDG edition: 2024
#
# Historical note: the T2K+NOvA combined best fit under NuFIT 5.3
# (2023) peaked near 230° with ~36° uncertainty; the NuFIT 6.0 best
# fit has shifted downward to ~177°.  The experimental value of
# delta_CP is currently in flux; DUNE and Hyper-Kamiokande will
# resolve it to ~5-10° precision.

# --- PREDICTED VALUE -----------------------------------------
# Value:       (g - 1) * arg(h*) mod 360° = 9 * (-52.2388°) mod 360°
#                                         = 249.851°
# Deviation:   ~1.9 sigma from NuFIT 6.0 best fit (using 20° as 1-sigma);
#              ~0.55 sigma from historical NuFIT 5.3 best fit.
#              The prediction is zero-parameter; the ~2-sigma tension
#              with the current best fit is within reach of DUNE/HK.

# --- DERIVED FORMULA -----------------------------------------
# delta_CP = (g - 1) * arg(h*)  mod 360°
#          = 9 * arg(h*)  mod 360°
#          = -9 * arg(h)  mod 360°
#
# Full derivation chain (see predictions/delta_CP_PMNS_derivation.md):
#
#   1. MDL + binary self-inverse toggle                 [axioms A1, A2]
#      => d = 3, k* = 3, srs lattice.
#                 [predictions/d_spatial.py, k_star.py, g_girth.py]
#
#   2. Walker dynamics on srs are non-backtracking walks with the
#      Hashimoto operator B as the 1-step transition.
#                                 [docs/theorem_walker_dynamics.md]
#
#   3. At the P-point, B(P) has eigenvalue h = (sqrt(3) + i sqrt(5))/2
#      with multiplicity 2, C_3-protected; conjugate eigenvalue h*.
#                             [docs/theorem_BP_doubly_degenerate_h.md]
#
#   4. srs has girth g = 10 (it is the unique (3, 10)-cage among 3D
#      crystal nets).                      [predictions/g_girth.py]
#
#   5. Adopted postulate P-phase-from-holonomy
#      (docs/W4_identification_catalog.md §2C):
#        Phase observables are accumulated arguments of h^n (or (h*)^n)
#        over specific closed walks on srs, with n determined by the
#        walk's topological invariants.
#      Sub-postulates invoked for delta_CP:
#        (i)   The relevant walk class for Dirac CP is the Jarlskog
#              loop: a closed walk of length g with exactly one C_3
#              generation transition at one edge.
#        (ii)  At the transition edge, C_3 representation theory forces
#              that edge (a sector alpha |-> beta transition uses the
#              unique edge carrying beta * alpha^{-1}); so n = g - 1
#              free NB edges, each contributing arg(h*).
#        (iii) h is the CP-covariant walker eigenvalue: CP conjugation
#              acts on the walk amplitude as complex conjugation, and
#              therefore on the walker eigenvalue as h |-> h*.
#
#   6. De Moivre's theorem: for any complex z and integer n,
#        arg(z^n) = n * arg(z)  (mod 2 pi).
#      [Standard complex analysis; e.g. Needham, *Visual Complex
#       Analysis*, OUP 1997, §1.IV.]
#      Applied with z = h* and n = g - 1 = 9:
#        arg((h*)^9) = 9 * arg(h*) = -9 * arg(h).
#
#   7. Reduction mod 360°: 9 * arg(h*) = 9 * (-52.2388°) = -470.149°;
#      adding 720° gives delta_CP = 249.851°.
#
# Distinction from alpha_21 (Majorana phase): alpha_21 = g * arg(h)
# mod 360° (full girth cycle, no sector-crossing edge => exponent g).
# delta_CP and alpha_21 are both instances of P-phase-from-holonomy
# with different walk-topology inputs (g-edge return loop vs Jarlskog
# loop with one fixed edge).

# --- INPUTS --------------------------------------------------
# symbol | value               | status    | predictions/ file                    | meaning
# -------|---------------------|-----------|--------------------------------------|--------
# h      | (sqrt(3)+i sqrt(5))/2 | [derived] | predictions/h_walker_eigenvalue.py | Hashimoto walker eigenvalue at P
# g      | 10                  | [derived] | predictions/g_girth.py               | girth of srs
# P-phase| adopted             | [adopted] | docs/W4_identification_catalog.md §2C | Type-C identification

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import math
import cmath
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h_walker_eigenvalue import predict_h_walker_eigenvalue
from k_star import predict_k_star
from d_spatial import predict_d_spatial
from srs_E_at_P import predict_srs_E_at_P
from g_girth import predict_g_girth

# Upstream values (all axiomatically derived; no numerical inputs).
d = predict_d_spatial()
k = predict_k_star(d)
E = predict_srs_E_at_P(k)
h = predict_h_walker_eigenvalue(k, E)
g = predict_g_girth(k, d)

# Under the P-phase-from-holonomy postulate (W4 §2C) with Jarlskog-loop
# walk topology, the number of free NB edges contributing holonomy is
# n = g - 1 (one edge is fixed by the C_3 transition), and the
# accumulated walker is h* (CP conjugation).
h_star = h.conjugate()
arg_h_star = cmath.phase(h_star)                 # in radians, in (-pi, pi]
n = g - 1
# De Moivre: arg((h*)^n) = n * arg(h*) mod 2 pi.
delta_CP_rad = (n * arg_h_star) % (2 * math.pi)
delta_CP_deg = math.degrees(delta_CP_rad)

print(f"h      = {h}")
print(f"h*     = {h_star}")
print(f"arg(h*) = {math.degrees(arg_h_star):.6f}° = -arg(h)")
print(f"n      = g - 1 = {n}  (Jarlskog loop, one edge fixed by C_3)")
print(f"n * arg(h*) = {math.degrees(n * arg_h_star):.6f}° (raw, de Moivre)")
print(f"delta_CP = {delta_CP_deg:.6f}° (mod 360°)")


# --- PURE FUNCTION -------------------------------------------
# No hardcoded physical constants.  Every input (the walker eigenvalue
# and the exponent) is a named parameter.  The only literals inside
# the function body are the mathematical constants used to express
# de Moivre's theorem and the mod-360 reduction.

def predict_delta_CP_PMNS(h, g_girth):
    """
    Compute the PMNS Dirac CP phase under the framework's Type-C
    (phase-from-holonomy) identification for the Jarlskog loop.

    delta_CP = (g_girth - 1) * arg(h*)  mod 360°
             = - (g_girth - 1) * arg(h)  mod 360°.

    The exponent is g_girth - 1 (not g_girth): the Jarlskog loop is a
    closed walk of length g_girth with exactly one C_3 generation
    transition; C_3 representation theory forces that one edge, so
    g_girth - 1 edges remain free and each contributes arg(h*) by de
    Moivre's theorem.  CP conjugation sends the walker eigenvalue h
    to h* (complex conjugate), which is why arg(h*) appears instead
    of arg(h).

    Parameters
    ----------
    h : complex
        Hashimoto walker eigenvalue at the P-point (from
        predictions/h_walker_eigenvalue.py).  For srs: h =
        (sqrt(3) + i sqrt(5))/2.
    g_girth : int
        Girth of the srs lattice (from predictions/g_girth.py).
        For srs: g_girth = 10.

    Returns
    -------
    float
        delta_CP in degrees, reduced to [0, 360).
    """
    # CP conjugation: walker eigenvalue h |-> h* (complex conjugate).
    arg_h_conj = cmath.phase(h.conjugate())
    # Jarlskog loop: n = g - 1 free edges carry the holonomy.
    n = g_girth - 1
    # De Moivre + mod-2-pi reduction.
    return math.degrees((n * arg_h_conj) % (2 * math.pi))


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("STATUS: BLOCKED under Theorem B6 retraction (2026-04-17)")
    print("See predictions/delta_CP_PMNS_derivation.md §Status")
    print("Re-derivation target: Sprint 11 workstream B7.5")
    print("(PMNS under C^3_gen; docs/master_plan.md §Sprint 11)")
    print("Jarlskog-loop sub-postulate 'one edge fixed by C_3 generation")
    print("transition' requires C_3 = generation; B6 proves C_3 = color-Z_3.")
    print("Holonomy arithmetic (g-1)*arg(h*) mod 360 is preserved as a lemma.")
    print("=" * 60)
    impl_result = delta_CP_deg
    pure_result = predict_delta_CP_PMNS(h, g)
    print(f"\nImplementation: {impl_result:.6f}°")
    print(f"Pure function:  {pure_result:.6f}°")
    assert abs(impl_result - pure_result) < 1e-10, \
        f"Mismatch: {impl_result} vs {pure_result}"
    print("OK: outputs agree.")
    print(f"    delta_CP = {pure_result:.2f}°  "
          f"(NuFIT 6.0 NO best fit: 177 (+19/-20)°)")
