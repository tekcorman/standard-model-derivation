#!/usr/bin/env python3
"""
Canonical prediction file for the dimension counts of the length-graded
multiway Hilbert space H_multiway = ⨁_L H_unred^(L) and its canonical
visible/dark decomposition

NOTE (post-2026-04-26 demotion): A2 and A3 are derived theorems; structural
slate is {A1} + P1' + A5-mass per docs/framework/framework_axioms.md §10. The closure
chain referenced here is preserved; only the axiomatic-status labels change.
This file's dim-count content is purely combinatorial under A1 + A2-T. Under
A3-T, the canonicalization map pi acquires a partial-trace interpretation
(see predictions/observer_hilbert_space.py), but the length-graded dim counts
are unchanged.

    H_unred^(L) = H_visible^(L) ⊕ H_dark^(L),

where H_visible^(L) is the span of length-L reduced words of the free
involutive monoid F_inv(E) (alphabet of |E| undirected edges, e·e = ε),
identified per ../predictions/walker_dynamics_derivation.md Steps 1-3 with length-L
non-backtracking walks of the srs primitive cell (|E| = 6, k* = 3),
and H_dark^(L) is the span of length-L strings containing at least one
adjacent cancellable pair e·e (i.e., strings whose F_inv(E) reduction
strictly shortens them).

The canonicalization map π: H_unred^(L) → H_red is the MDL
canonicalization derived in ../predictions/walker_dynamics_derivation.md Step 2;
H_dark^(L) is the span of basis vectors corresponding to non-reduced
length-L strings (equivalently, strings on which π acts non-trivially
within length L).

The closed-form recursion proved here is

    dim H_unred^(L) = n^L
    dim H_visible^(L) = R_L = n (n-1)^(L-1)         for L ≥ 1,  R_0 = 1
    dim H_dark^(L)    = n^L − R_L                    for L ≥ 1,
                      = n · [n^(L-1) − (n-1)^(L-1)]  (closed form)

with n = |E| = 6 for the srs primitive cell. R_L counts length-L NB
walks on a |E|-letter alphabet without graph-incidence constraint
(i.e., the FREE involutive monoid count, before restricting to walks
compatible with srs incidence — that restriction is the directed-edge
NB walk on the 12-dim Bloch fibre, treated separately at the Bloch
level in predictions/B_P_doubly_degenerate_h.py).

This script is the dim-count lemma F.1-O3-α of
docs/theorem_H_multiway_construction.md. It does NOT close the
companion question of whether the Schur-complement coupling
B_visible(q) ↔ B_dark(q) modifies the small-|q| Bloch dispersion off
|q|^2 scaling — that question is reported as OPEN in the companion
derivation document and the foundations proof
proofs/foundations/H_multiway_construction.py.
"""

# ============================================================
# PARAMETER: H_multiway_dim_count  (length-L dimension recursion
# of H_unred, H_visible = H_red, H_dark = span of non-reduced
# length-L strings on the F_inv(E) alphabet of |E| edges)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       (dim H_unred^(L), dim H_visible^(L), dim H_dark^(L))
#              for L = 0,1,2,3,4,5,6
#              = [(1,1,0), (6,6,0), (36,30,6), (216,150,66),
#                 (1296,750,546), (7776,3750,4026),
#                 (46656,18750,27906)]
#              with n = |E| = 6.
# Source:      Structural prediction from F_inv(E) free-involutive-monoid
#              combinatorics (Serre 1980, Trees, §I.1 Prop. 4) applied
#              to the alphabet of |E| undirected edges of the srs
#              primitive cell (k*=3, d=3 ⇒ srs ⇒ 4 vertices, 6 edges
#              per primitive cell, derived upstream).
# PDG edition: n/a

# --- PREDICTED VALUE -----------------------------------------
# Value:       Closed form
#              dim H_dark^(L) = n · [n^(L-1) − (n-1)^(L-1)]  for L ≥ 1.
#              For n = 6: D_L = 6 · (6^(L-1) − 5^(L-1)).
# Deviation:   sympy verifies the closed form against direct
#              enumeration for L = 0..7 to exact equality (rationals).

# --- DERIVED FORMULA -----------------------------------------
# Per the derivation doc:
#
#   1. Upstream: |E| = 6 in the srs primitive cell.
#                                       [predictions/d_spatial.py,
#                                        predictions/k_star.py,
#                                        predictions/g_girth_derivation.md §2]
#   2. Upstream: F_inv(E) = free involutive monoid on E, derived from
#      axiom A1 (T_e · T_e = 1) via Serre 1980 §I.1 Prop. 4.
#                                       [../predictions/walker_dynamics_derivation.md
#                                        Step 1]
#   3. Upstream: MDL canonicalization (A2-T) selects reduced words.
#                                       [../predictions/walker_dynamics_derivation.md
#                                        Step 2; Grünwald 2007 §5.1-5.3]
#   4. Length-L count of REDUCED words on |E|-letter alphabet:
#         R_0 = 1
#         R_1 = n
#         R_L = (n-1) · R_{L-1}     for L ≥ 2
#         ⇒ R_L = n (n-1)^(L-1)     for L ≥ 1
#      This is the standard count of length-L NB walks on a complete
#      graph K_{n+1} with backtracking forbidden, with the modification
#      that the FREE involutive monoid does not impose graph incidence.
#                                       [Serre 1980 §I.1 Prop. 4;
#                                        Terras 2011 §2.1]
#   5. Length-L count of UNREDUCED strings: U_L = n^L (free monoid).
#                                       [elementary]
#   6. Therefore D_L := dim H_dark^(L) = U_L − R_L
#                = n^L − n (n-1)^(L-1)
#                = n · [n^(L-1) − (n-1)^(L-1)]  for L ≥ 1,
#      D_0 = 0.
#                                       [step 6 of the derivation doc]
#   7. Asymptotic ratio
#         D_L / U_L = 1 − (1 − 1/n)^(L-1) · 1
#                   = 1 − (5/6)^(L-1)   (n=6)
#      ⇒ D_L / U_L → 1 as L → ∞:
#      almost all sufficiently long strings are dark.
#                                       [step 7 of the derivation doc]

# --- INPUTS --------------------------------------------------
# symbol      | value           | status    | predictions/ file                        | meaning
# ------------|-----------------|-----------|------------------------------------------|--------
# n = |E|     | 6               | [derived] | predictions/d_spatial.py + g_girth_derivation.md §2 | undirected edges per srs primitive cell
# k_star      | 3               | [derived] | predictions/k_star.py                    | coordination number; selects srs (used implicitly for |E|=k_star · |V|/2 = 3·4/2 = 6)
# axiom A1    | T_e · T_e = 1   | [axiom]   | (framework axiom)                        | binary self-inverse toggle
# A2-T (waterline thm) | MDL    | [thm]     | docs/theorems/theorem_A2_mdl_from_finite_register.md | minimum description length

# --- IMPLEMENTATION ------------------------------------------
# sympy-symbolic verification of the closed form against direct
# enumeration of length-L reduced strings for L = 0..7.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# moved to proofs/ 2026-05-27: predictions/ siblings live 2 dirs up at <repo>/predictions
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "predictions"))

import sympy as sp
from itertools import product

from k_star import predict_k_star
from d_spatial import predict_d_spatial
import functools


def _count_reduced_strings_brute(n, L):
    """Enumerate length-L strings on {0,..,n-1} with no two adjacent
    equal letters; return the count.  Used as a brute-force ground truth.
    """
    if L == 0:
        return 1
    if L == 1:
        return n
    count = 0
    for word in product(range(n), repeat=L):
        ok = True
        for i in range(L - 1):
            if word[i] == word[i + 1]:
                ok = False
                break
        if ok:
            count += 1
    return count


def _count_dark_strings_brute(n, L):
    """Enumerate length-L strings on {0,..,n-1} containing at least one
    adjacent equal pair; return the count.
    """
    return n ** L - _count_reduced_strings_brute(n, L)


# Upstream framework values: derive |E| = 6 from k_star=3, d=3 ⇒ srs.
d = predict_d_spatial()
k_star = predict_k_star(d)
# srs primitive cell has |V| = 4 vertices, k* = 3, so |E| = k_star * |V| / 2 = 6.
N_VERTICES = 4
n_edges = (k_star * N_VERTICES) // 2
assert n_edges == 6, f"Unexpected |E| for srs primitive cell: {n_edges}"

# Sympy symbolic recursion verification.
n_sym = sp.Symbol('n', positive=True, integer=True)
L_sym = sp.Symbol('L', positive=True, integer=True)

# Closed form for R_L and D_L.
R_closed = n_sym * (n_sym - 1) ** (L_sym - 1)
D_closed = n_sym ** L_sym - R_closed
D_factored = n_sym * (n_sym ** (L_sym - 1) - (n_sym - 1) ** (L_sym - 1))

# Verify factored form algebraically.
assert sp.expand(D_closed - D_factored) == 0, \
    f"Factored form mismatch: D_closed - D_factored = {sp.expand(D_closed - D_factored)}"

print("=" * 70)
print("H_multiway dim-count theorem (length-graded, on F_inv(E))")
print("=" * 70)
print(f"|E| = {n_edges}  (derived from k_star = {k_star}, d_spatial = {d}, srs primitive cell)")
print()
print("Closed form (symbolic, verified by sympy):")
print(f"  R_L = dim H_visible^(L) = n (n-1)^(L-1)        for L ≥ 1")
print(f"        R_0 = 1")
print(f"  U_L = dim H_unred^(L)   = n^L")
print(f"  D_L = dim H_dark^(L)    = U_L − R_L = n · [n^(L-1) − (n-1)^(L-1)]")
print()
print(f"For n = |E| = {n_edges}:")
print(f"  R_L = {n_edges} · {n_edges - 1}^(L-1)")
print(f"  D_L = {n_edges} · ({n_edges}^(L-1) − {n_edges - 1}^(L-1))")
print()

# Tabulate dims for L = 0..7 and verify against brute-force enumeration.
print(f"{'L':>3s} | {'U_L = n^L':>14s} | {'R_L (closed)':>14s} | {'R_L (brute)':>14s} | {'D_L (closed)':>14s} | {'D_L (brute)':>14s}")
print("-" * 90)
all_ok = True
for L in range(0, 8):
    U_L = n_edges ** L
    if L == 0:
        R_L_closed = 1
    else:
        R_L_closed = n_edges * (n_edges - 1) ** (L - 1)
    R_L_brute = _count_reduced_strings_brute(n_edges, L)
    D_L_closed = U_L - R_L_closed
    D_L_brute = _count_dark_strings_brute(n_edges, L)
    ok = (R_L_closed == R_L_brute) and (D_L_closed == D_L_brute)
    all_ok = all_ok and ok
    print(f"{L:>3d} | {U_L:>14d} | {R_L_closed:>14d} | {R_L_brute:>14d} | {D_L_closed:>14d} | {D_L_brute:>14d}")
assert all_ok, "Closed form disagrees with brute-force enumeration."
print()
print("All length-L counts agree exactly between closed form and direct")
print("enumeration over the n^L = 6^L strings (verified for L = 0..7).")
print()

# Asymptotic ratio.
print("Asymptotic dark fraction D_L / U_L = 1 − (1 − 1/n)^(L-1):")
for L in [1, 2, 3, 5, 10, 20, 50]:
    if L == 0:
        frac = 0.0
    else:
        frac = 1.0 - (1.0 - 1.0 / n_edges) ** (L - 1)
    print(f"  L = {L:>3d}:  D_L / U_L ≈ {frac:.6f}")
print()
print("⇒ Almost all sufficiently long strings are dark; the visible (reduced)")
print("  fraction shrinks geometrically as ((n-1)/n)^(L-1) = (5/6)^(L-1).")
print()

# Per-step branching at the GRAPH-INCIDENCE level (walker_dynamics Step 4):
# at each vertex, Jaynes-uniform over the k = 3 incident edges; the visible
# walker takes (k-1)/k = 2/3 (the k-1 = 2 non-backtrack choices); the
# remaining 1/k = 1/3 probability mass is the cancellation event that
# walker_dynamics Step 4 calls "the cancellation case is erased from the
# reduced output."  This 1/k = 1/3 IS the per-step D → V (or rather
# V → D-and-cancel-instantly) exchange rate of the framework's only
# derived stochastic measure.
#
# NOTE: the F_inv(E) free-monoid count above uses alphabet size n = |E| = 6
# (no graph-incidence constraint); the on-graph per-step Jaynes rates use
# k = 3 (local incidence).  The two are consistent: the on-graph walker is
# the F_inv(E) walker projected by graph-incidence, which is a separate
# (already-derived, theorem-grade) restriction handled in
# predictions/B_P_doubly_degenerate_h.py and not duplicated here.
print(f"Per-step Jaynes-uniform branching at the on-graph (k=3) level:")
print(f"  P(NB extension into visible)        = (k-1)/k = 2/3 = {2/3:.6f}")
print(f"  P(cancellation, absorbed into dark) = 1/k     = 1/3 = {1/3:.6f}")
print(f"  (per walker_dynamics Step 4; consistent with this F_inv(E) count)")
print()


# --- PURE FUNCTION -------------------------------------------
# Inputs: n_edges, L_max.  No hardcoded constants.

@functools.lru_cache(maxsize=None)
def predict_H_multiway_dim_count(n_edges, L_max):
    """
    Returns the length-L dimension counts of the length-graded multiway
    Hilbert space H_multiway = ⨁_L H_unred^(L) and its canonical
    visible/dark decomposition for L = 0..L_max.

    Each L-slice splits as
        H_unred^(L) = H_visible^(L) ⊕ H_dark^(L),
    with H_visible^(L) the span of length-L reduced words on the
    n_edges-letter alphabet of F_inv(E) (= the basis of length-L NB
    walks on the primitive cell, before graph-incidence restriction)
    and H_dark^(L) the span of length-L strings containing at least
    one adjacent cancellable pair.

    Parameters
    ----------
    n_edges : int
        Alphabet size = |E| = number of undirected edges per primitive
        cell.  For srs (k* = 3, d = 3) this is 6.
    L_max : int
        Maximum length to tabulate (inclusive).

    Returns
    -------
    list of (int, int, int, int)
        Length-L tuple (L, dim H_unred^(L), dim H_visible^(L), dim H_dark^(L))
        for L = 0..L_max.  Entries are exact Python ints (no floating-point).
    """
    if n_edges < 2:
        raise ValueError(f"Require n_edges >= 2 for non-trivial F_inv(E); got {n_edges}.")
    if L_max < 0:
        raise ValueError(f"Require L_max >= 0; got {L_max}.")
    out = []
    for L in range(L_max + 1):
        U_L = n_edges ** L
        if L == 0:
            R_L = 1
        else:
            R_L = n_edges * (n_edges - 1) ** (L - 1)
        D_L = U_L - R_L
        out.append((L, U_L, R_L, D_L))
    return out


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_table = [
        (L,
         n_edges ** L,
         (1 if L == 0 else n_edges * (n_edges - 1) ** (L - 1)),
         (n_edges ** L) - (1 if L == 0 else n_edges * (n_edges - 1) ** (L - 1)))
        for L in range(0, 8)
    ]
    pure_table = predict_H_multiway_dim_count(n_edges, 7)
    assert impl_table == pure_table, \
        f"Implementation/pure mismatch: {impl_table} vs {pure_table}"
    print("Implementation and pure function tables agree (L = 0..7).")
    print()
    print(f"RESULT: H_multiway dim-count for n=|E|={n_edges}: D_L = n·(n^(L-1) − (n-1)^(L-1)) "
          f"verified against brute-force enumeration up to L=7.")
