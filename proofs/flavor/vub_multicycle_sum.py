#!/usr/bin/env python3
"""
proofs/flavor/vub_multicycle_sum.py

PURPOSE
-------
Compute V_ub as the sum of MDL-permitted multi-cycle walk-rep host
amplitudes on H(srs).

CONJECTURE
----------
V_ub = Σ_{m=2}^∞  (2/3)^{6m+2} / (1 - (2/3)^{6m+2})

where each m corresponds to a substrate-level multi-cycle host topology:
  m girth cycles glued in series by (m-1) seams of length 2.
  L_cycle(m) = m·g - 2(m-1)·s = 10m - 4(m-1) = 6m + 4
  L_eff(m)   = L_cycle(m) - n_fixed = 6m + 2

For m = 1: L_eff = 8 → V_cb = 256/6305 (single girth cycle, V_cb host).
For m ≥ 2: multi-cycle compositions, ALL retained by MDL (A2 waterline).

The split:
  V_cb  := m=1 only               (irreducible single-cycle host)
  V_ub  := Σ_{m≥2} (multi-cycle)  (composite hosts, all MDL-positive)

CAS BACKING (this session, 2026-04-25):
  - hashimoto_longcycle_inventory.py:    H(srs) cycle spectrum {10,14,16,...}
  - hashimoto_16cycle_decomposition.py:  100% of L=16 cycles decompose as
                                          two girth cycles + 2-edge seam
  - hashimoto_14cycle_decomposition.py:  100% of L=14 cycles decompose as
                                          two girth cycles + 3-edge seam
                                          (excluded by Feshbach n_fixed=2 cap)

NUMERICAL RESULT
----------------
  V_ub_sum ≈ 3.7670e-3
  PDG 2024 V_ub exclusive: 0.00369 ± 0.00011 → +0.70σ
  PDG 2024 V_ub inclusive: 0.00413 ± 0.00015 → -2.42σ
  (within the well-known excl/incl experimental tension band)

GATE STATUS
-----------
The arithmetic (geometric series at each L_eff = 6m+2; cycle decomposition
on H(srs)) is gate-passing. The LOAD-BEARING structural step that remains
open: "V_cb takes only m=1; V_ub takes all m≥2." That mapping needs a
substrate-level argument before this candidate can be linter-graded
THEOREM. Until then, this script is CAS demonstration of the numerical
match, not a closure.
"""

import sys, os
from fractions import Fraction
import functools

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'predictions'))

from k_star import predict_k_star
from g_girth import predict_g_girth
from d_spatial import predict_d_spatial
from feshbach_exponent_principle import predict_feshbach_coupling


# ──────────────────────────────────────────────────────────────────────────────
# Multi-cycle host walk-rep amplitudes
# ──────────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=None)
def L_eff_at_m(g_girth, s_seam, n_fixed, m):
    """Effective walk length for m girth cycles glued in series by (m-1)
    seams of length s_seam, with n_fixed boundary edges.

    L_cycle(m) = m·g - 2(m-1)·s   (boundary edge count after gluing)
    L_eff(m)   = L_cycle(m) - n_fixed
    """
    L_cycle = m * g_girth - 2 * (m - 1) * s_seam
    return L_cycle - n_fixed


@functools.lru_cache(maxsize=None)
def V_walkrep_at_m(k_star, g_girth, s_seam, n_fixed, m):
    """Geometric-series walk-rep amplitude on the m-cycle host.

    Each m-cycle host has L_eff(m) = 6m + 2 internal NB steps (for k_star=3,
    g=10, s=2, n_fixed=2). The amplitude is the A5(b) walk-rep sum:
       V_m = α/(1-α)   with  α = ((k-1)/k)^L_eff(m)
    """
    L = L_eff_at_m(g_girth, s_seam, n_fixed, m)
    alpha = Fraction(k_star - 1, k_star) ** L
    return alpha / (1 - alpha)


@functools.lru_cache(maxsize=None)
def V_ub_partial_sum(k_star, g_girth, s_seam, n_fixed, m_max):
    """Partial sum Σ_{m=2}^{m_max} V_m of multi-cycle walk-rep contributions."""
    total = Fraction(0)
    for m in range(2, m_max + 1):
        total += V_walkrep_at_m(k_star, g_girth, s_seam, n_fixed, m)
    return total


# ──────────────────────────────────────────────────────────────────────────────
# Pure prediction function
# ──────────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=None)
def predict_V_ub_multicycle(k_star, g_girth, s_seam, n_fixed, m_max=10):
    """Compute V_ub = Σ_{m≥2} (geometric series at L_eff(m)) on H(srs).

    Convergence is geometric in (k-1)/k)^6 = (2/3)^6 ≈ 0.088 per m, so the
    sum converges to ~14 digits by m_max ≈ 10 for k_star=3, g=10, s_seam=2.

    Parameters
    ----------
    k_star : int
        MDL-optimal coordination number (3 for srs).
    g_girth : int
        Girth of the base graph (10 for srs).
    s_seam : int
        Seam length between consecutive girth cycles in multi-cycle host.
        Must equal n_fixed for consistency with the Feshbach principle.
    n_fixed : int
        Number of fixed Feshbach boundary edges (2 for CKM 2-leg vertex).
    m_max : int
        Truncation index for the multi-cycle sum. Series converges fast.

    Returns
    -------
    float
        V_ub prediction = Σ_{m=2}^{m_max} V_m.
    """
    if s_seam != n_fixed:
        raise ValueError(f"seam length {s_seam} must equal n_fixed {n_fixed}")
    return float(V_ub_partial_sum(k_star, g_girth, s_seam, n_fixed, m_max))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    d = predict_d_spatial()
    k = predict_k_star(d)
    g = predict_g_girth(k, d)
    s_seam = 2
    n_fixed = 2

    print("=" * 70)
    print("V_ub multi-cycle host walk-rep sum")
    print("=" * 70)
    print(f"  k* = {k}, g = {g}, s_seam = {s_seam}, n_fixed = {n_fixed}")
    print()

    print(f"  Per-m contributions V_m (with L_eff = 6m + 2):")
    print(f"  {'m':>3s} {'L_eff':>5s}   {'V_m (rational)':>30s}   {'V_m (numeric)':>15s}")
    for m in range(1, 9):
        L = L_eff_at_m(g, s_seam, n_fixed, m)
        V_m = V_walkrep_at_m(k, g, s_seam, n_fixed, m)
        tag = '  ← V_cb' if m == 1 else ''
        print(f"  {m:>3d} {L:>5d}   {str(V_m):>30s}   {float(V_m):>15.10e}{tag}")
    print()

    # V_cb (m=1) check
    V_cb_predicted = V_walkrep_at_m(k, g, s_seam, n_fixed, 1)
    V_cb_obs = 0.04050
    V_cb_unc = 0.00150
    print(f"  V_cb (m=1 only) = {str(V_cb_predicted)} = {float(V_cb_predicted):.6e}")
    print(f"  PDG 2024 excl   = {V_cb_obs} ± {V_cb_unc}")
    sigma_cb = (float(V_cb_predicted) - V_cb_obs) / V_cb_unc
    print(f"  Deviation       = {sigma_cb:+.3f}σ")
    print()

    # V_ub sum
    print("  V_ub = Σ_{m≥2} V_m  (converges fast; m_max=10 saturates):")
    V_ub_predicted = float(V_ub_partial_sum(k, g, s_seam, n_fixed, 10))
    V_ub_obs_excl = 0.00369; V_ub_unc_excl = 0.00011
    V_ub_obs_incl = 0.00413; V_ub_unc_incl = 0.00015

    for m_max in (2, 3, 4, 5, 6, 8, 10):
        V_partial = float(V_ub_partial_sum(k, g, s_seam, n_fixed, m_max))
        sigma_e = (V_partial - V_ub_obs_excl) / V_ub_unc_excl
        sigma_i = (V_partial - V_ub_obs_incl) / V_ub_unc_incl
        print(f"    m_max={m_max:2d}: V_ub = {V_partial:.10e}  excl={sigma_e:+.3f}σ  incl={sigma_i:+.3f}σ")

    print()
    print(f"  Final V_ub = {V_ub_predicted:.6e}")
    print(f"  PDG 2024 excl: {V_ub_obs_excl} ± {V_ub_unc_excl}")
    print(f"  PDG 2024 incl: {V_ub_obs_incl} ± {V_ub_unc_incl}")
    print(f"  Excl deviation: {(V_ub_predicted - V_ub_obs_excl)/V_ub_unc_excl:+.3f}σ")
    print(f"  Incl deviation: {(V_ub_predicted - V_ub_obs_incl)/V_ub_unc_incl:+.3f}σ")
    print()

    # Pure function check
    pure = predict_V_ub_multicycle(k, g, s_seam, n_fixed, m_max=10)
    assert abs(pure - V_ub_predicted) < 1e-15
    print(f"  Pure function:  predict_V_ub_multicycle({k},{g},{s_seam},{n_fixed},10)")
    print(f"                = {pure:.10e}  ✓")

    print()
    print("  GATE STATUS")
    print("  -----------")
    print("  The arithmetic is gate-passing under A1+A2+A5(b)+CAS-verified")
    print("  multi-cycle decomposition (this session). The LOAD-BEARING")
    print("  structural step that remains open:")
    print()
    print("    'V_cb takes only m=1; V_ub takes all m≥2.'")
    print()
    print("  That assignment is currently a 2-point empirical match. A")
    print("  substrate-level argument deriving this split from R3 (3-generation)")
    print("  + A5(b) (MDL retention) is required for theorem-grade closure.")
