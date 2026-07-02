#!/usr/bin/env python3
"""
Temporal correlation length xi_t per srs edge.

Framework-internal quantity: the decay length (in Planck units) of the
per-edge toggle Markov chain's connected autocorrelation function.
Derived from Stage 2a thresholds via the 2-state Markov chain spectrum.

Gate grade: THEOREM (Type 2 algebra on Stage 2a Type 4 upstream).

Cross-reference: Stage 3 (docs/theorems/theorem_lorentz_causal_sector.md §4.2).
"""

# ============================================================
# PARAMETER: xi_t (temporal correlation length per edge)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       Framework-internal length. Not directly observed;
#              Planck-scale correlations in per-edge toggle events
#              are below any experimental resolution. Downstream
#              consequence: sets the scale below which toggle
#              correlations are exponentially suppressed in Stage 3's
#              Lorentz-invariance analysis.
# Source:      N/A (framework-derived quantity).
# PDG edition: N/A.

# --- PREDICTED VALUE -----------------------------------------
# Value:       xi_t = 1 / log(6) Planck lengths = 0.5581106... l_P
# Deviation:   N/A (no direct observation).

# --- DERIVED FORMULA -----------------------------------------
# The 2-state Markov chain from Stage 2a has transition matrix
#   M = [[1 - p_create, p_destroy ], [ p_create, 1 - p_destroy]]
#     = [[ 1/2, 1/3 ], [ 1/2, 2/3 ]].
#
# Its eigenvalues solve (lam - 1)(lam - r) = 0 with
#   tr(M) = 1/2 + 2/3 = 7/6  =>  r = tr(M) - 1 = 1/6.
#
# Per standard Markov chain spectral theory, the connected
# autocorrelation function at time separation s decays as r^s.
#
# Therefore the temporal correlation length (in Planck steps) is
#
#   xi_t = 1 / log(1/r) = 1 / log(6).
#
# Numerically: xi_t = 0.5581106... Planck units.
#
# Derivation chain:
#   A1 + A2-T (waterline thm; refined A2)
#     -> Stage 2a (docs/theorems/theorem_edge_surprise_thresholds.md):
#        p_create = 1/2, p_destroy = 1/3.
#     -> transition matrix M (above).
#     -> characteristic polynomial, r = 1/6.
#     -> exponential decay of connected autocorrelations.
#     -> correlation length xi_t = 1/log(6).

# --- INPUTS --------------------------------------------------
# symbol       | value | status    | source                                    | meaning
# -------------|-------|-----------|-------------------------------------------|----------
# p_create     | 1/2   | [derived] | docs/theorems/theorem_edge_surprise_thresholds.md  | Beta(1,1) predictive
# p_destroy    | 1/3   | [derived] | docs/theorems/theorem_edge_surprise_thresholds.md  | Beta(2,1) predictive

# --- IMPLEMENTATION ------------------------------------------

from fractions import Fraction
from math import log
import functools


# Stage 2a upstream values
p_create_frac = Fraction(1, 2)
p_destroy_frac = Fraction(1, 3)

# Second eigenvalue of the transition matrix
# det(M) = p_create * (1 - p_destroy) - p_destroy * (1 - p_create)... wait use trace
# For 2x2 Markov matrix: eigenvalues are 1 and tr(M) - 1
trace_M = (1 - p_create_frac) + (1 - p_destroy_frac)  # = 1/2 + 2/3 = 7/6
r_frac = trace_M - 1                                   # = 1/6
assert r_frac == Fraction(1, 6)

# Correlation length
xi_t = 1.0 / log(float(1 / r_frac))  # = 1 / log(6)

print(f"p_create  = {p_create_frac}")
print(f"p_destroy = {p_destroy_frac}")
print(f"r         = {r_frac} = {float(r_frac):.6f}")
print(f"xi_t      = 1/log({int(1/r_frac)}) = {xi_t:.15f} Planck lengths")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_xi_t(p_create, p_destroy):
    """
    Temporal correlation length (in Planck units) of the 2-state
    Bernoulli-Beta Markov chain per srs edge.

    The transition matrix M has trace tr(M) = (1 - p_create) + (1 - p_destroy)
    = 2 - p_create - p_destroy, and its non-trivial eigenvalue is
    r = tr(M) - 1 = 1 - p_create - p_destroy. The autocorrelation decays
    as r^s with correlation length xi_t = 1 / log(1/r).

    Parameters
    ----------
    p_create : float
        Probability per step of off -> on (Stage 2a: 1/2).
    p_destroy : float
        Probability per step of on -> off (Stage 2a: 1/3).

    Returns
    -------
    float
        Correlation length xi_t in Planck units.
    """
    r = 1.0 - p_create - p_destroy
    if not (0 < r < 1):
        raise ValueError(f"Second eigenvalue r = {r} out of (0, 1).")
    from math import log
    return 1.0 / log(1.0 / r)


# --- VALIDATION ----------------------------------------------

xi_t_temporal_correlation_pred = xi_t


if __name__ == "__main__":
    impl_result = xi_t
    pure_result = predict_xi_t(float(p_create_frac), float(p_destroy_frac))
    print(f"\nImplementation: {impl_result:.15f}")
    print(f"Pure function:  {pure_result:.15f}")
    assert abs(impl_result - pure_result) < 1e-15

    # Cross-check: xi_t = 1/log(6)
    expected = 1.0 / log(6.0)
    assert abs(pure_result - expected) < 1e-15
    print(f"Cross-check:    1/log(6) = {expected:.15f}  OK")

    # Sympy exact
    import sympy as sp
    r_sym = sp.Rational(1, 6)
    xi_sym = 1 / sp.log(1 / r_sym)
    print(f"Sympy exact:    {xi_sym} = {float(xi_sym):.15f}  OK")

    print("\nOK: xi_t = 1/log(6) exactly from Stage 2a thresholds.")
