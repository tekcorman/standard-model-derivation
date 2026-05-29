#!/usr/bin/env python3
"""
Toggle rate per edge per Planck step (lambda).

This file ships the framework prediction lambda = 2/5 exactly, derived as the
stationary toggle probability of a 2-state Markov chain whose transition
probabilities (p_create = 1/2, p_destroy = 1/3) come from Stage 2a's
Beta(1,1) and Beta(2,1) posteriors.

Gate grade: THEOREM (Type 2 algebra on Stage 2a Type 4 upstream).

Cross-reference: derived within Stage 3 (docs/theorems/theorem_lorentz_causal_sector.md
§4.1) and independently computed at proofs/lorentz/b1_ags_audit.py.
"""

# ============================================================
# PARAMETER: lambda (toggle rate per edge per Planck step)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       Framework-internal rate. Not directly observed;
#              toggle events at Planck scale are not accessible
#              to current experiments. Downstream observational
#              consequences propagate through Stage 3 dispersion
#              analysis to the ~147 PeV scale energy.
# Source:      N/A (framework-derived quantity).
# PDG edition: N/A.

# --- PREDICTED VALUE -----------------------------------------
# Value:       lambda = 2/5 exactly.
# Deviation:   N/A (no direct observation).

# --- DERIVED FORMULA -----------------------------------------
# lambda = pi_off * p_create + pi_on * p_destroy
#        = (p_create * p_destroy) / (p_create + p_destroy)
#
# where (p_create, p_destroy) are the Bayesian update acceptance
# probabilities from Stage 2a (Beta(1,1) prior, Beta(2,1) posterior
# after one confirmation), and (pi_off, pi_on) are the stationary
# distribution of the resulting 2-state Markov chain.
#
# Derivation chain:
#   A1 (toggle alphabet) + A2 refined (MDL observer)
#     -> Stage 2a (docs/theorems/theorem_edge_surprise_thresholds.md):
#        per-pair Bayesian Beta-Bernoulli model with uniform Beta(1,1)
#        prior. Predictive probabilities:
#           P(exists | Beta(1,1))  = 1/2 = p_create
#           P(absent | Beta(2,1))  = 1/3 = p_destroy
#     -> 2-state Markov chain transition matrix:
#           M = [[1 - p_create, p_destroy ],
#                [  p_create,  1 - p_destroy]]
#             = [[ 1/2, 1/3 ],
#                [ 1/2, 2/3 ]]
#     -> Stationary distribution (detailed balance):
#           pi_on * p_destroy = pi_off * p_create
#           pi_on * (1/3)     = (1 - pi_on) * (1/2)
#           => pi_on  = 3/5,  pi_off = 2/5.
#     -> Stationary toggle rate:
#           lambda = pi_off * p_create + pi_on * p_destroy
#                  = (2/5)(1/2) + (3/5)(1/3)
#                  = 1/5 + 1/5 = 2/5.

# --- INPUTS --------------------------------------------------
# symbol       | value | status    | source                                    | meaning
# -------------|-------|-----------|-------------------------------------------|----------
# p_create     | 1/2   | [derived] | docs/theorems/theorem_edge_surprise_thresholds.md  | Beta(1,1) predictive: P(exists)
# p_destroy    | 1/3   | [derived] | docs/theorems/theorem_edge_surprise_thresholds.md  | Beta(2,1) predictive: P(absent)

# --- IMPLEMENTATION ------------------------------------------

from fractions import Fraction
import functools


# Stage 2a upstream values
p_create_frac = Fraction(1, 2)
p_destroy_frac = Fraction(1, 3)

# Stationary distribution of the 2-state Markov chain
pi_on_frac = p_create_frac / (p_create_frac + p_destroy_frac)   # = 3/5
pi_off_frac = Fraction(1) - pi_on_frac                           # = 2/5

# Stationary toggle rate
lambda_frac = pi_off_frac * p_create_frac + pi_on_frac * p_destroy_frac

assert lambda_frac == Fraction(2, 5), (
    f"Derivation mismatch: lambda = {lambda_frac}, expected 2/5"
)

print(f"p_create      = {p_create_frac} = {float(p_create_frac):.6f}")
print(f"p_destroy     = {p_destroy_frac} = {float(p_destroy_frac):.6f}")
print(f"pi_on         = {pi_on_frac} = {float(pi_on_frac):.6f}")
print(f"pi_off        = {pi_off_frac} = {float(pi_off_frac):.6f}")
print(f"lambda        = {lambda_frac} = {float(lambda_frac):.6f}")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_lambda_toggle_rate(p_create, p_destroy):
    """
    Compute the stationary toggle rate lambda per edge per Planck step
    for a 2-state Bernoulli-Beta Markov chain with given transition
    probabilities.

    The 2-state chain has:
        off -> on with probability p_create
        on  -> off with probability p_destroy

    At stationary distribution:
        pi_on  = p_create / (p_create + p_destroy)
        pi_off = p_destroy / (p_create + p_destroy)

    The stationary toggle rate (probability of a flip on any given step)
    is:
        lambda = pi_off * p_create + pi_on * p_destroy
               = 2 * p_create * p_destroy / (p_create + p_destroy).

    Parameters
    ----------
    p_create : float
        Probability per step of flipping from off to on. Stage 2a
        assigns p_create = 1/2 from the Beta(1,1) prior's predictive.
    p_destroy : float
        Probability per step of flipping from on to off. Stage 2a
        assigns p_destroy = 1/3 from the Beta(2,1) posterior's
        predictive after one confirmation.

    Returns
    -------
    float
        Stationary toggle rate lambda.
    """
    return 2.0 * p_create * p_destroy / (p_create + p_destroy)


# --- VALIDATION ----------------------------------------------

lambda_toggle_rate_pred = float(lambda_frac)


if __name__ == "__main__":
    impl_result = float(lambda_frac)
    pure_result = predict_lambda_toggle_rate(float(p_create_frac), float(p_destroy_frac))
    print(f"\nImplementation: {impl_result:.15f}")
    print(f"Pure function:  {pure_result:.15f}")
    assert abs(impl_result - pure_result) < 1e-15, (
        f"Mismatch: {impl_result} vs {pure_result}"
    )

    # Exact rational check
    assert lambda_frac == Fraction(2, 5), "Exact rational check failed"
    print(f"Exact value: {lambda_frac} = 0.4")

    # Sympy independent verification
    import sympy as sp
    pc, pd = sp.symbols("p_c p_d", positive=True)
    lam_sym = 2 * pc * pd / (pc + pd)
    lam_val = lam_sym.subs({pc: sp.Rational(1, 2), pd: sp.Rational(1, 3)})
    assert lam_val == sp.Rational(2, 5), (
        f"Sympy mismatch: {lam_val} vs 2/5"
    )
    print(f"Sympy exact:    {lam_val}  OK")

    print("\nOK: lambda = 2/5 exactly from Stage 2a thresholds.")
