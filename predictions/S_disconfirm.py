#!/usr/bin/env python3
"""
Edge-disconfirmation surprise S_disconfirm = log_2(3) bits exactly.

Framework-internal quantity: the Shannon self-information of observing
"edge absent" on a pair whose posterior has Beta(2, 1) structure (one
prior confirmation). Equals log_2(3) ~ 1.585 bits. This asymmetry with
S_fresh = 1 bit is the microscopic source of the toggle-process arrow
of time (creation is cheaper than disconfirmation).

Gate grade: THEOREM (Type 2 algebra + Jaynes 1957 + Shannon 1948).

Cross-reference: Stage 2a (docs/theorems/theorem_edge_surprise_thresholds.md §10).
"""

# ============================================================
# PARAMETER: S_disconfirm (disconfirming-observation surprise)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       Framework-internal information quantity. Not directly
#              observed. Sets p_destroy = 2^(-S_disconfirm) = 1/3
#              in the 2-state Markov chain.
# Source:      N/A (framework-derived quantity).
# PDG edition: N/A.

# --- PREDICTED VALUE -----------------------------------------
# Value:       S_disconfirm = log_2(3) bits exactly.
#              Numerically: 1.58496... bits.
# Deviation:   N/A (no direct observation).

# --- DERIVED FORMULA -----------------------------------------
# After one observation of "edge exists" at a pair with Beta(1, 1) prior,
# Bayesian conjugate update (Stage 2a §6) gives posterior Beta(2, 1).
# Under Beta(2, 1), the predictive probabilities are:
#
#   P(exists | Beta(2, 1))  = alpha / (alpha + beta) = 2/3
#   P(absent | Beta(2, 1))  = beta  / (alpha + beta) = 1/3
#
# By Shannon 1948 §I, the surprise of observing "absent" (disconfirming
# the previous observation of "exists") is:
#
#   S_disconfirm = -log_2 P(absent) = -log_2(1/3) = log_2(3) bits.
#
# Asymmetry vs S_fresh:
#   S_disconfirm / S_fresh = log_2(3) / 1 = log_2(3) > 1
#   S_disconfirm - S_fresh = log_2(3) - 1 = log_2(3/2) ≈ 0.585 bits > 0
# This asymmetry is the microscopic source of the arrow of time.
#
# Derivation chain:
#   A1 + A2-T (waterline thm; refined A2)
#     -> Jaynes 1957 MaxEnt uniform prior Beta(1, 1) (see S_fresh).
#     -> Bayesian conjugate update (Stage 2a §6): Beta(1, 1) + "exists"
#        = Beta(2, 1).
#     -> Predictive P(absent) = 1/3 under Beta(2, 1).
#     -> Shannon surprise = -log_2(1/3) = log_2(3).

# --- INPUTS --------------------------------------------------
# symbol | value | status    | source                                   | meaning
# -------|-------|-----------|------------------------------------------|---------
# alpha  | 2     | [derived] | docs/theorems/theorem_edge_surprise_thresholds.md | Beta posterior alpha after 1 confirmation
# beta   | 1     | [derived] | docs/theorems/theorem_edge_surprise_thresholds.md | Beta posterior beta

# --- IMPLEMENTATION ------------------------------------------

from fractions import Fraction
from math import log2
import functools

# Post-one-confirmation Beta posterior
alpha = Fraction(2)
beta = Fraction(1)

# Predictive probability of "absent"
P_absent = beta / (alpha + beta)   # = 1/3

# Shannon surprise
S_disconfirm = -log2(float(P_absent))

print(f"Posterior Beta(alpha={alpha}, beta={beta})")
print(f"P_absent        = {P_absent} = {float(P_absent):.6f}")
print(f"S_disconfirm    = -log_2({P_absent}) = log_2(3) = {S_disconfirm:.15f} bits")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_S_disconfirm(alpha_posterior, beta_posterior):
    """
    Shannon self-information (surprise) of observing "absent" under a
    Beta(alpha, beta) posterior.

    Predictive probability of "absent" is beta/(alpha + beta); surprise
    is -log_2 of that. For the once-confirmed posterior Beta(2, 1),
    P(absent) = 1/3 and surprise = log_2(3) bits.

    Parameters
    ----------
    alpha_posterior : float
        Beta posterior alpha. Stage 2a: alpha = 2 after one confirmation.
    beta_posterior : float
        Beta posterior beta. Stage 2a: beta = 1 after one confirmation.

    Returns
    -------
    float
        Surprise in bits.
    """
    from math import log2
    P = beta_posterior / (alpha_posterior + beta_posterior)
    return -log2(P)


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_result = S_disconfirm
    pure_result = predict_S_disconfirm(float(alpha), float(beta))
    print(f"\nImplementation: {impl_result:.15f}")
    print(f"Pure function:  {pure_result:.15f}")
    assert abs(impl_result - pure_result) < 1e-15
    assert abs(pure_result - log2(3.0)) < 1e-15

    # Sympy exact
    import sympy as sp
    a, b = sp.symbols("alpha beta", positive=True)
    S_sym = -sp.log(b / (a + b), 2)
    S_val = S_sym.subs({a: 2, b: 1})
    assert S_val == sp.log(3, 2), f"Sympy mismatch: {S_val} vs log_2(3)"
    print(f"Sympy exact:    S_disconfirm = {S_val} = {float(S_val):.15f}  OK")

    # Asymmetry check
    S_fresh = 1.0
    asymmetry = pure_result - S_fresh
    import math
    expected_asym = math.log2(3) - 1
    assert abs(asymmetry - expected_asym) < 1e-15
    print(f"Asymmetry:      S_disconfirm - S_fresh = log_2(3/2) = {asymmetry:.6f} bits")

    print("\nOK: S_disconfirm = log_2(3) exactly from Beta(2, 1) posterior.")
