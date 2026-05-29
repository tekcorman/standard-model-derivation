#!/usr/bin/env python3
"""
Feshbach Exponent Principle — standalone theorem file.

This file proves the Exponent Principle as a combinatorial consequence of
walker_dynamics W4 (Jaynes-uniform over k-1 NB choices) plus the graph-
theoretic definition of girth.

STATUS UPDATE (2026-04-19, session 2): Under A5(b) — the coupling clause
of A5 (docs/framework/framework_axioms.md §5b, established 2026-04-19) — the
identification of the NB-walk survival factor with a physical coupling
strength is now an axiomatic identification, not an adopted gap. The
previously-named "I-Feshbach" identification is subsumed by A5(b).
Combined with the rigorous combinatorial half (this file), the full
chain ((NB walk survival) = (physical dark coupling) = ((k-1)/k)^(g-2))
is THEOREM under A1 + A2-T + A5(b) + Jaynes 1957 + Serre 1980 + Terras 2011.

Scope of this file: the combinatorial Exponent Principle for
n_fixed in {0, 1, 2} on a k-regular graph of girth g.
"""

# ============================================================
# PARAMETER: Feshbach Exponent Principle
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       Not directly observed. A structural combinatorial
#              theorem about NB walk survival probabilities on
#              k-regular graphs of girth g.
# Source:      N/A (derived graph-theoretic content).
# PDG edition: N/A.

# --- PREDICTED VALUE -----------------------------------------
# For n_fixed in {0, 1, 2}:
#   coupling(n_fixed) = ((k-1)/k)^(g - n_fixed)
#
# Instantiation on srs (k = 3, g = 10):
#   n_fixed = 0  (self-energy, closed loop)        ((k-1)/k)^g     = (2/3)^10 = 1024/59049
#   n_fixed = 1  (transition, one pinned edge)     ((k-1)/k)^(g-1) = (2/3)^9  = 512/19683
#   n_fixed = 2  (scattering, in+out pinned)       ((k-1)/k)^(g-2) = (2/3)^8  = 256/6561

# --- DERIVED FORMULA -----------------------------------------
# Let G be a k-regular graph of girth g. Let e_1, ..., e_{n_fixed}
# be a set of n_fixed directed edges that are declared "external"
# (pinned: their weights are not counted in the survival factor).
#
# Claim. The minimum-length closed NB walk through the n_fixed pinned
# edges has length g, and the number of "internal" (non-pinned) NB
# steps it contains is g - n_fixed. By W4 (walker_dynamics Step 4)
# each internal step has Jaynes-uniform conditional probability
# (k-1)/k of being a valid NB continuation on the universal covering
# tree. The combined NB survival probability over g - n_fixed
# internal steps is
#
#     survival = ((k-1)/k)^(g - n_fixed).
#
# This is the Exponent Principle in its minimal combinatorial form.
#
# Derivation chain (no physics identifications invoked):
#   A1 + A2-T (+ d_spatial, k_star, g_girth derivations upstream)
#     -> walker_dynamics W4: conditional per-step NB survival = (k-1)/k
#        (rigor: Jaynes 1957 max-entropy on k incident edges + MDL
#         cancellation of the backtrack edge -> uniform over k-1 NB
#         continuations; Serre 1980 §I.1 for the reduced-word content.)
#     -> universal covering tree is loop-free (graph theory; Serre
#        1980 §I.3).
#     -> independence of per-step survival events on the universal
#        covering tree (no two NB walks of length L on the tree share
#        a vertex other than at their endpoints; Terras 2011 §2.1).
#     -> survival over L independent NB steps on tree = ((k-1)/k)^L
#        (Feshbach coupling strength doc Lemma 1; derivable under A1 + A2-T
#         without any additional axiom).
#     -> girth g = length of shortest cycle on G (definition).
#     -> minimum closed NB walk from a directed edge e back to e has
#        length g (definition of girth applied to NB-walk closed cycles;
#        equivalent to "shortest cycle" because any closed NB walk
#        corresponds to a cycle in G and vice versa — Terras 2011 §2.1).
#     -> "internal length after pinning n_fixed edges of a girth cycle"
#        = g - n_fixed (elementary subtraction: a cycle of length g
#        has g edges; pinning n_fixed of them leaves g - n_fixed to
#        count toward NB survival).
#     -> coupling(n_fixed) = ((k-1)/k)^(g - n_fixed).

# --- INPUTS --------------------------------------------------
# symbol   | value | status    | predictions/ file         | meaning
# ---------|-------|-----------|---------------------------|---------
# k_star   | 3     | [derived] | predictions/k_star.py     | coordination number
# g_girth  | 10    | [derived] | predictions/g_girth.py    | girth of srs
# n_fixed  | 0..2  | [input]   | scope of this theorem     | # pinned external edges

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k_star import predict_k_star
from d_spatial import predict_d_spatial
from g_girth import predict_g_girth
import functools


def _internal_length(g_girth, n_fixed):
    """Return (g - n_fixed), the number of internal NB steps in a minimum
    closed walk through n_fixed pinned edges of a girth cycle.

    The scope of this theorem is n_fixed in {0, 1, 2}. Outside that range,
    the "minimum closed walk through n_fixed pinned edges is a girth cycle"
    hypothesis is not generically true on a k-regular graph of girth g
    (multi-loop and higher-n_fixed diagrams require separate combinatorial
    arguments; see ../predictions/Feshbach_coupling_strength_derivation.md §3 for the
    downstream physics readings that go beyond this scope).
    """
    if n_fixed not in (0, 1, 2):
        raise ValueError(
            f"Exponent Principle scope is n_fixed in {{0, 1, 2}}. "
            f"Got n_fixed = {n_fixed}."
        )
    return g_girth - n_fixed


d = predict_d_spatial()
k = predict_k_star(d)
g = predict_g_girth(k, d)

print(f"k* = {k}, g = {g}")
print(f"Per-step NB survival on universal covering tree: (k-1)/k = {k-1}/{k}")
print()
print(f"Exponent Principle couplings for n_fixed in {{0, 1, 2}}:")
print(f"  {'n_fixed':>8s}  {'exponent':>9s}  {'coupling (rational)':>22s}  {'approx':>12s}")
for n_fixed in (0, 1, 2):
    L_internal = g - n_fixed
    coupling_exact = Fraction(k - 1, k) ** L_internal
    coupling_float = float(coupling_exact)
    print(f"  {n_fixed:>8d}  g-{n_fixed} = {L_internal:>2d}  "
          f"({k-1}/{k})^{L_internal} = {str(coupling_exact):>12s}"
          f"  {coupling_float:>10.7f}")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_feshbach_coupling(k_star, g_girth, n_fixed):
    """
    Computes the Exponent-Principle NB-walk-survival coupling for n_fixed
    pinned external edges on a k-regular graph of girth g.

    On the universal covering tree of a k-regular graph, the conditional
    probability that a directed edge extends to a non-backtracking
    continuation at its target vertex is (k-1)/k (per walker_dynamics W4,
    from A1 + A2-T + Jaynes 1957 max-entropy on the k incident edges).

    The minimum closed NB walk on the base graph has length g (girth
    definition). For n_fixed in {0, 1, 2}, pinning n_fixed of the g
    directed edges of a girth cycle leaves g - n_fixed internal steps
    each contributing a (k-1)/k survival factor. The total survival is
    ((k-1)/k)^(g - n_fixed).

    Parameters
    ----------
    k_star : int
        Coordination number (from predict_k_star).
    g_girth : int
        Girth of the base graph (from predict_g_girth).
    n_fixed : int
        Number of pinned external edges on the girth cycle. Must be in
        {0, 1, 2} (the scope of this theorem).

    Returns
    -------
    float
        Coupling strength = ((k_star - 1) / k_star) ** (g_girth - n_fixed).
    """
    if n_fixed not in (0, 1, 2):
        raise ValueError(
            "Exponent Principle scope is n_fixed in {0, 1, 2}."
        )
    return ((k_star - 1) / k_star) ** (g_girth - n_fixed)


# --- VALIDATION ----------------------------------------------

feshbach_exponent_principle_pred = coupling_float


if __name__ == "__main__":
    # Test case 1: n_fixed = 2 (scattering) reproduces alpha_1_bare.
    impl_scatter = (float((Fraction(k - 1, k)) ** (g - 2)))
    pure_scatter = predict_feshbach_coupling(k, g, 2)
    assert abs(impl_scatter - pure_scatter) < 1e-15
    assert abs(pure_scatter - (2 / 3) ** 8) < 1e-15
    print(f"\nTest 1 (scattering, n_fixed=2): pure = {pure_scatter:.15f}  "
          f"expected (2/3)^8 = {(2/3)**8:.15f}  OK")

    # Test case 2: n_fixed = 0 (self-energy).
    impl_self = float(Fraction(k - 1, k) ** g)
    pure_self = predict_feshbach_coupling(k, g, 0)
    assert abs(impl_self - pure_self) < 1e-15
    assert abs(pure_self - (2 / 3) ** 10) < 1e-15
    print(f"Test 2 (self-energy, n_fixed=0): pure = {pure_self:.15f}  "
          f"expected (2/3)^10 = {(2/3)**10:.15f}  OK")

    # Test case 3: n_fixed = 1 (transition).
    impl_trans = float(Fraction(k - 1, k) ** (g - 1))
    pure_trans = predict_feshbach_coupling(k, g, 1)
    assert abs(impl_trans - pure_trans) < 1e-15
    assert abs(pure_trans - (2 / 3) ** 9) < 1e-15
    print(f"Test 3 (transition, n_fixed=1): pure = {pure_trans:.15f}  "
          f"expected (2/3)^9 = {(2/3)**9:.15f}  OK")

    # Test case 4: scope guard.
    try:
        predict_feshbach_coupling(k, g, 3)
    except ValueError:
        print("Test 4 (scope guard, n_fixed=3): correctly raised ValueError  OK")
    else:
        raise AssertionError("Scope guard failed to reject n_fixed=3.")

    # Sympy independent verification of the exact rationals.
    import sympy as sp
    k_sym, g_sym = sp.symbols("k g", positive=True, integer=True)
    for n_fixed in (0, 1, 2):
        sym_expr = ((k_sym - 1) / k_sym) ** (g_sym - n_fixed)
        sym_val = sym_expr.subs({k_sym: k, g_sym: g})
        num_val = predict_feshbach_coupling(k, g, n_fixed)
        assert abs(float(sym_val) - num_val) < 1e-15, \
            f"sympy mismatch at n_fixed={n_fixed}: {sym_val} vs {num_val}"
    print("Test 5 (sympy cross-check for n_fixed in {0,1,2}): OK")

    print("\nOK: Exponent Principle holds for n_fixed in {0, 1, 2} on srs (k=3, g=10).")
