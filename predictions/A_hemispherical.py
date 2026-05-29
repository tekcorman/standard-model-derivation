#!/usr/bin/env python3
"""
Canonical prediction file for A (CMB hemispherical power asymmetry).

NOTE (post-2026-04-26 demotion): A2 and A3 are derived theorems; structural
slate is {A1} + P1' + A5-mass per docs/framework/framework_axioms.md §10. The closure
chain referenced here is preserved; only the axiomatic-status labels change.
A2-T/A3-T do not affect this file's Bayesian-posterior arithmetic; the
identification of 1/15 with the CMB sky observable remains the
OTHER-SMUGGLE step flagged in an internal strict-gating audit
(Section 4.1).
"""

# ============================================================
# PARAMETER: A (hemispherical power asymmetry)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       0.07 ± 0.02 (Planck 2018, ℓ_max = 64)
# Source:      Planck 2018 VII, A&A 641, A7, 2020
# PDG edition: N/A (CMB anomaly)

# --- PREDICTED VALUE -----------------------------------------
# Value:       1/15 = 0.06667 (exact)
# Deviation:   0.17 sigma

# --- DERIVED FORMULA -----------------------------------------
# A = ε_toggle / k* = (1/5) / 3 = 1/15
#
# Derivation chain:
#   1. k* = 3 (from predictions/k_star.py)
#   2. ε_toggle = 1/5: Bayesian posterior asymmetry for creation
#      vs disruption of toggle events.
#      Standalone substrate-primitives derivation:
#        proofs/foundations/epsilon_toggle_substrate_derivation.py
#      P_create = 1/2 (toggle event under Beta(1,1) prior:
#        Bayesian conjugate, Gelman BDA Ch.2)
#      P_disrupt = 1/3 (disconfirmation under Beta(2,1) posterior:
#        P(different) = (p-1)/(p+1) = 1/3 for p=2)
#      The Bayesian posterior for "creation" vs total:
#        p_creation = P_create/(P_create + P_disrupt) = (1/2)/(5/6) = 3/5
#      The asymmetry (unique linear map from posterior to [-1,1]):
#        ε = 2·p_creation - 1 = 6/5 - 1 = 1/5
#      Equivalently: ε = (P_create - P_disrupt)/(P_create + P_disrupt)
#                      = (1/2 - 1/3)/(1/2 + 1/3) = (1/6)/(5/6) = 1/5
#      This is probability axioms + Bayesian inference. No physics imported.
#   3. Geometric factor: <(e·ẑ)²> = 1/k* = 1/3
#      (From predictions/srs_cubic_moment.py at n=1.)
#   4. A = ε_toggle × <(e·ẑ)²> = (1/5)(1/3) = 1/15

# --- INPUTS --------------------------------------------------
# symbol       | value | status    | predictions/ file             | meaning
# -------------|-------|-----------|------------------------------|--------
# k_star       | 3     | [derived] | predictions/k_star.py        | coordination number
# cubic_moment | 1/3   | [derived] | predictions/srs_cubic_moment.py | <(e·ẑ)²> at n=1

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k_star import predict_k_star
from d_spatial import predict_d_spatial
from srs_cubic_moment import predict_srs_cubic_moment
from fractions import Fraction
import functools

d = predict_d_spatial()
k = predict_k_star(d)

# Toggle asymmetry: epsilon = 1/5
# From Beta(1,1) → Beta(2,1) update after one toggle confirmation
epsilon_toggle = Fraction(1, 5)

# Geometric dilution: <(e·z)²> = 1/k* = 1/3
geometric = Fraction(1, k)
assert float(geometric) == predict_srs_cubic_moment(1, k)

# A = epsilon * geometric
A = epsilon_toggle * geometric
A_float = float(A)

print(f"k* = {k}")
print(f"ε_toggle = {epsilon_toggle} (Bayesian toggle asymmetry)")
print(f"<(e·ẑ)²> = {geometric} (432 cubic moment at n=1)")
print(f"A = {epsilon_toggle} × {geometric} = {A} = {A_float:.10f}")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_A_hemispherical(k_star, epsilon_toggle):
    """
    Computes the CMB hemispherical power asymmetry.

    A = ε_toggle × <(e·ẑ)²> = ε_toggle / k_star.

    Parameters
    ----------
    k_star : int
        Coordination number (from predict_k_star).
    epsilon_toggle : float
        Toggle asymmetry parameter (= 1/5 from Bayesian update).

    Returns
    -------
    float
        A = epsilon_toggle / k_star.
    """
    return epsilon_toggle / k_star


# --- VALIDATION ----------------------------------------------

A_hemispherical_pred = A_float


if __name__ == "__main__":
    impl_result = A_float
    pure_result = predict_A_hemispherical(k, float(epsilon_toggle))
    exact = float(Fraction(1, 15))
    print(f"\nImplementation: {impl_result:.15f}")
    print(f"Pure function:  {pure_result:.15f}")
    print(f"1/15:           {exact:.15f}")
    assert abs(impl_result - pure_result) < 1e-15
    assert abs(pure_result - exact) < 1e-15
    print("OK: outputs agree. A = 1/15 exactly.")
    print(f"    (obs: 0.07 ± 0.02, {abs(exact - 0.07)/0.02:.1f}σ)")
