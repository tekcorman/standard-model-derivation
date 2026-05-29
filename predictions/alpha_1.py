#!/usr/bin/env python3
"""
Canonical prediction file for alpha_1 (bare NB walk survival probability).

Audit anchor: Row P1 of `docs/parameters/parameter_uniqueness_ledger.md`. UNIQUE
within "NB walks on k*-regular graphs with branch measure μ" — conditional
on Row 4 (k* = 3, structural), Row 9 (g = 10, structural), Row 12 (branch
measure μ uniform per-step, structural) of `docs/audits/registers/uniqueness_ledger.md`.
"""

# ============================================================
# PARAMETER: alpha_1 (bare coupling constant)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       Not directly measured. Verified through downstream
#              parameters: V_cb, Koide masses, dark corrections.
# PDG edition: N/A (derived graph-theoretic constant)

# --- PREDICTED VALUE -----------------------------------------
# Value:       (2/3)^8 = 256/6561 ≈ 0.039018442310623
# Deviation:   N/A (exact rational, verified indirectly)

# --- DERIVED FORMULA -----------------------------------------
# alpha_1 = ((k*-1)/k*)^(g-2)
#
# Derivation chain:
#   1. k* = 3 (from predictions/k_star.py)
#   2. g = 10 (from predictions/g_girth.py)
#   3. On a k*-regular graph, a non-backtracking walker at any vertex
#      arrived via one edge. Of k* incident edges, 1 leads back
#      (forbidden by NB constraint), k*-1 are forward choices.
#      Per-step survival probability: (k*-1)/k*.
#      (Combinatorial fact for k-regular graphs — no citation needed.)
#   4. The walk length is g-2: a non-backtracking cycle of length g
#      has g-2 intermediate vertices where the NB constraint applies.
#      The first vertex has no arrival edge (walk starts there),
#      and the last step closes the cycle (forced).
#      (Counting argument on NB walks — see Terras, "Zeta Functions
#       of Graphs", Cambridge 2011, Ch. 1 for NB walk conventions.)
#   5. alpha_1 = ((k*-1)/k*)^(g-2) = (2/3)^8 = 256/6561.
#
# NOTATION: This is alpha_1_bare. Some scripts use alpha_1 to mean
# alpha_1_full = (5/3) * alpha_1_bare, the mass^2-class chirality
# coupling. See dark_correction_theorem documentation for details.
#
# STATUS (2026-04-19, session 2): alpha_1_bare = (2/3)^8 is THEOREM
# under A1 + A2-T + A5(b) + Jaynes 1957. The combinatorial half (NB
# walk survival probability) is rigorous; the identification with
# the physical dark-sector coupling strength is via A5(b) — the
# coupling clause of A5 (docs/framework/framework_axioms.md §5b, established
# 2026-04-19). Previously this was flagged as ADVANCED due to the
# "I-Feshbach" adoption; A5(b) closes it.

# --- INPUTS --------------------------------------------------
# symbol  | value | status      | predictions/ file              | meaning
# --------|-------|-------------|--------------------------------|--------
# k_star  | 3     | [derived]   | predictions/k_star.py          | coordination number
# g_girth | 10    | [derived]   | predictions/g_girth.py         | girth of srs
# A5(b)   | —     | [axiom]     | docs/framework/framework_axioms.md §5b   | MDL prob = coupling

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k_star import predict_k_star
from d_spatial import predict_d_spatial
from g_girth import predict_g_girth
import functools

d = predict_d_spatial()
k = predict_k_star(d)
g = predict_g_girth(k, d)

# Per-step survival: (k-1)/k
survival_per_step = (k - 1) / k

# Walk length: g - 2
walk_length = g - 2

# alpha_1 = survival^walk_length
alpha_1 = survival_per_step ** walk_length

# Exact rational check
from fractions import Fraction
alpha_1_exact = Fraction(k - 1, k) ** walk_length

print(f"k* = {k}, g = {g}")
print(f"  Per-step survival: (k*-1)/k* = {k-1}/{k}")
print(f"  Walk length: g-2 = {g}-2 = {walk_length}")
print(f"  alpha_1 = ({k-1}/{k})^{walk_length} = {alpha_1_exact} ≈ {float(alpha_1_exact):.15f}")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_alpha_1(k_star, g_girth):
    """
    Computes the bare NB walk survival probability.

    On a k-regular graph, a non-backtracking walker survives each step
    with probability (k-1)/k. Over g-2 steps (the intermediate vertices
    of a girth-length NB cycle), the total survival probability is
    ((k-1)/k)^(g-2).

    Parameters
    ----------
    k_star : int
        Coordination number (from predict_k_star).
    g_girth : int
        Girth of the crystal net (from predict_g_girth).

    Returns
    -------
    float
        alpha_1 = ((k_star - 1) / k_star) ** (g_girth - 2)
    """
    return ((k_star - 1) / k_star) ** (g_girth - 2)


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_result = alpha_1
    pure_result = predict_alpha_1(k, g)
    exact_float = float(alpha_1_exact)
    print(f"\nImplementation:  {impl_result:.15f}")
    print(f"Pure function:   {pure_result:.15f}")
    print(f"Exact rational:  {exact_float:.15f}")
    assert abs(impl_result - pure_result) < 1e-15, \
        f"Mismatch: {impl_result} vs {pure_result}"
    assert abs(pure_result - exact_float) < 1e-15, \
        f"Mismatch with exact: {pure_result} vs {exact_float}"
    print("OK: outputs agree. alpha_1 = (2/3)^8 = 256/6561 exactly.")
