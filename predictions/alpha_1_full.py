#!/usr/bin/env python3
"""
Canonical prediction file for alpha_1_full (Class 2 dark-sector coupling).

alpha_1_full = (5/3) * alpha_1_bare = (5/3) * (2/3)^8 = 1280/19683

Derived under A1 + A2-T (waterline thm; selective retention) + A5(b) (physical
identification), with the 5/3 coefficient coming from the combinatorial
graph invariant n_g_edge/k = 5/3 of the srs lattice under Jaynes
max-entropy weighting over admissible girth cycles.

Graduation event 2026-04-20 (session 4): 5/3 coefficient moved from
ADOPTED (via tan^2(arg h) structural pattern) to DERIVED (via refined
A2 + cycle enumeration). Supersedes the Rate-Distortion route scoped
in an internal working note
"""

# ============================================================
# PARAMETER: alpha_1_full (Class 2 dark-sector coupling)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       Not directly measured. Verified through downstream
#              parameters: theta_23 (dark correction), lambda_higgs
#              (5/3 channel), v_higgs dark vertex, m_nu3 shape factor.
# PDG edition: N/A (derived graph-theoretic combinatorial constant)

# --- PREDICTED VALUE -----------------------------------------
# Value:       (5/3) * (2/3)^8 = 1280/19683 ~ 0.06503
# Deviation:   N/A (exact rational; verified via downstream checks)

# --- DERIVED FORMULA -----------------------------------------
# alpha_1_full = (n_g_edge / k*) * ((k*-1)/k*)^(g-2)
#
# Derivation chain:
#   1. k* = 3 (predictions/k_star.py)
#   2. g = 10 (predictions/g_girth.py)
#   3. n_g_edge = 5 cycles per ordered edge pair on srs.
#      VERIFIED by proofs/foundations/srs_graph_analysis.py:
#      constructs 3x3x3 srs supercell, enumerates girth-10 cycles,
#      finds 15 per vertex and 5 per ordered edge pair, consistent
#      across multiple vertices (vertex-transitive, Sunada 2012).
#   4. Under A2-T (theorem_A2_mdl_from_finite_register.md, selective retention),
#      all 5 admissible girth cycles per edge pair are retained
#      simultaneously (not just one canonical). Published foundation:
#      A-IT5 Rate-Distortion (Shannon 1959).
#   5. Under Jaynes 1957 max-entropy + I4_1 32 transitivity (Sunada 2012),
#      each admissible cycle is uniformly weighted. Probability of
#      initiating a specific cycle from a random NB step at a vertex
#      = 1/k*.
#   6. Effective Class 2 coefficient = n_g_edge * (1/k*) = 5/3.
#      (Sum over admissibles under refined A2 ensemble weighting.)
#   7. alpha_1_bare = ((k*-1)/k*)^(g-2) = (2/3)^8 from
#      predictions/feshbach_exponent_principle.py (STRICT-SOLID).
#   8. alpha_1_full = (5/3) * (2/3)^8 = 1280/19683.

# --- INPUTS --------------------------------------------------
# symbol      | value   | status    | file/theorem                           | meaning
# ------------|---------|-----------|----------------------------------------|--------
# k_star      | 3       | [derived] | predictions/k_star.py                  | coordination
# g_girth     | 10      | [derived] | predictions/g_girth.py                 | girth of srs
# n_g_edge    | 5       | [derived] | proofs/foundations/srs_graph_analysis  | cycles per edge pair
# alpha_1_bare| (2/3)^8 | [derived] | predictions/feshbach_exponent_principle| NB survival
# A2-T (waterline thm)  | —    | [thm]   | docs/theorems/theorem_A2_mdl_from_finite_register.md | selective retention
# Jaynes 1957 | —       | [cited]   | Jaynes Phys. Rev. 106, 620 (1957)      | max-entropy
# Sunada 2012 | —       | [cited]   | Notices AMS 59(2), 208-215             | srs transitivity
# A5(b)       | —       | [axiom]   | docs/framework/framework_axioms.md §5b           | MDL prob = coupling

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from k_star import predict_k_star
from g_girth import predict_g_girth
from d_spatial import predict_d_spatial
from alpha_1 import predict_alpha_1
from fractions import Fraction
import functools

d = predict_d_spatial()
k = predict_k_star(d)
g = predict_g_girth(k, d)
alpha_1_bare = predict_alpha_1(k, g)

# n_g_edge is a graph invariant of srs, verified numerically by
# proofs/foundations/srs_graph_analysis.py via explicit girth-cycle
# enumeration on the 3x3x3 srs supercell (216 vertices).
# Result: 5 cycles per ordered edge pair (vertex-transitive).
n_g_edge = 5

# Effective coefficient under refined A2: sum over admissible cycles
# weighted by uniform prior 1/k* per specific cycle.
effective_coefficient = Fraction(n_g_edge, k)
alpha_1_full = effective_coefficient * Fraction(alpha_1_bare).limit_denominator(100000)

print(f"alpha_1_full = (n_g_edge / k) * alpha_1_bare")
print(f"            = ({n_g_edge}/{k}) * ({alpha_1_bare})")
print(f"            = {float(alpha_1_full):.6f}")
print(f"            = {alpha_1_full}")


# --- PURE FUNCTION -------------------------------------------
@functools.lru_cache(maxsize=None)
def predict_alpha_1_full(k_star, g_girth, n_g_edge):
    """
    Compute the Class 2 dark-sector coupling alpha_1_full.

    alpha_1_full = (n_g_edge / k_star) * ((k_star - 1) / k_star)^(g_girth - 2)

    Parameters
    ----------
    k_star : int
        Coordination number of the MDL-selected lattice.
    g_girth : int
        Girth of the lattice.
    n_g_edge : int
        Number of distinct girth cycles passing through each ordered edge pair.
        For srs: n_g_edge = 5 (verified by srs_graph_analysis.py).

    Returns
    -------
    Fraction
        alpha_1_full as an exact rational.
    """
    alpha_1_bare = Fraction(k_star - 1, k_star) ** (g_girth - 2)
    return Fraction(n_g_edge, k_star) * alpha_1_bare


# --- VALIDATION ----------------------------------------------
if __name__ == "__main__":
    impl = alpha_1_full
    pure = predict_alpha_1_full(3, 10, 5)
    print(f"\nImplementation: {float(impl):.10f} = {impl}")
    print(f"Pure function:  {float(pure):.10f} = {pure}")
    assert impl == pure, f"Mismatch: {impl} vs {pure}"

    # Exact expected value: (5/3) * (2/3)^8 = 5*256 / (3*6561) = 1280/19683
    expected = Fraction(1280, 19683)
    assert pure == expected, f"Expected {expected}, got {pure}"
    print(f"Expected:       {float(expected):.10f} = {expected}")
    print(f"OK: outputs agree.  alpha_1_full = (5/3) * (2/3)^8 = 1280/19683")
