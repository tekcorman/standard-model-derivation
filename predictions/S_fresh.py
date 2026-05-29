#!/usr/bin/env python3
"""
Edge-creation surprise S_fresh = 1 bit exactly.

Framework-internal quantity: the Shannon self-information (surprise) of
the first observation of a pair's existence state, under Jaynes MaxEnt
Beta(1,1) prior. Equals 1 bit exactly — the minimum cost of novelty.

Gate grade: THEOREM (Type 2 algebra + Jaynes 1957 + Shannon 1948).

Cross-reference: Stage 2a (docs/theorems/theorem_edge_surprise_thresholds.md §9).
"""

# ============================================================
# PARAMETER: S_fresh (fresh-observation surprise)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       Framework-internal information quantity. Not directly
#              observed. Sets p_create = 1/2 in the 2-state Markov
#              chain (the predictive probability of "exists" equals
#              2^(-S_fresh) = 1/2).
# Source:      N/A (framework-derived quantity).
# PDG edition: N/A.

# --- PREDICTED VALUE -----------------------------------------
# Value:       S_fresh = 1 bit exactly.
# Deviation:   N/A (no direct observation).

# --- DERIVED FORMULA -----------------------------------------
# Under Jaynes 1957 MaxEnt applied to a Bernoulli parameter with no
# prior information, the observer's prior is uniform on [0, 1], i.e.
# Beta(1, 1). The predictive probability of any first observation
# (edge exists or edge absent) is:
#
#   P(observation | Beta(1, 1)) = alpha/(alpha + beta) = 1/(1+1) = 1/2.
#
# By Shannon 1948 §I (self-information):
#
#   S_fresh = -log_2 P = -log_2(1/2) = 1 bit.
#
# Either outcome ("exists" or "absent") carries the same surprise,
# reflecting the observer's initial uncertainty.
#
# Derivation chain:
#   A1 (toggle binary events) + A2 refined (MDL observer)
#     -> Jaynes 1957 MaxEnt on [0,1] Bernoulli parameter: uniform prior
#        = Beta(1, 1).
#     -> Predictive probability = 1/2 under Beta(1, 1).
#     -> Shannon surprise = -log_2(1/2) = 1.

# --- INPUTS --------------------------------------------------
# symbol       | value | status    | source                                   | meaning
# -------------|-------|-----------|------------------------------------------|---------
# alpha_prior  | 1     | [derived] | docs/theorems/theorem_edge_surprise_thresholds.md | Beta prior alpha
# beta_prior   | 1     | [derived] | docs/theorems/theorem_edge_surprise_thresholds.md | Beta prior beta

# --- IMPLEMENTATION ------------------------------------------

from fractions import Fraction
from math import log2
import functools

# MaxEnt + Jaynes -> uniform Beta(1,1) prior
alpha = Fraction(1)
beta = Fraction(1)

# Predictive probability of fresh observation
P_fresh = alpha / (alpha + beta)   # = 1/2

# Shannon surprise
S_fresh = -log2(float(P_fresh))

print(f"alpha_prior = {alpha}")
print(f"beta_prior  = {beta}")
print(f"P_fresh     = {P_fresh} = {float(P_fresh):.6f}")
print(f"S_fresh     = -log_2({P_fresh}) = {S_fresh:.15f} bit")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_S_fresh(alpha_prior, beta_prior):
    """
    Shannon self-information (surprise) of a fresh observation on a
    pair with Beta(alpha_prior, beta_prior) prior.

    Predictive probability of "exists" is alpha/(alpha + beta);
    predictive probability of "absent" is beta/(alpha + beta). For
    the Jaynes MaxEnt prior alpha = beta = 1 (uniform), both outcomes
    have probability 1/2 and surprise -log_2(1/2) = 1 bit.

    Parameters
    ----------
    alpha_prior : float
        Beta prior alpha. Jaynes MaxEnt gives alpha = 1 exactly.
    beta_prior : float
        Beta prior beta. Jaynes MaxEnt gives beta = 1 exactly.

    Returns
    -------
    float
        S_fresh in bits.
    """
    from math import log2
    P = alpha_prior / (alpha_prior + beta_prior)
    return -log2(P)


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_result = S_fresh
    pure_result = predict_S_fresh(float(alpha), float(beta))
    print(f"\nImplementation: {impl_result:.15f}")
    print(f"Pure function:  {pure_result:.15f}")
    assert abs(impl_result - pure_result) < 1e-15
    assert abs(pure_result - 1.0) < 1e-15, f"Expected 1 bit; got {pure_result}"

    # Sympy exact check
    import sympy as sp
    a, b = sp.symbols("alpha beta", positive=True)
    S_sym = -sp.log(a / (a + b), 2)
    S_val = S_sym.subs({a: 1, b: 1})
    assert S_val == 1, f"Sympy mismatch: {S_val} vs 1"
    print(f"Sympy exact:    S_fresh = {S_val}  OK")

    print("\nOK: S_fresh = 1 bit exactly under Jaynes MaxEnt Beta(1,1) prior.")
