#!/usr/bin/env python3
"""
Full-menu Path A ranking audit — does substrate-only MDL pick |E|=3?

Skeptical bridge probe (per menu_to_observation_bridge_scoping_2026-05-07.md
post-cascade-audit): the framework's existing k_star_derivation.md +
d_spatial_derivation.md chains use crystal-net + Gleason + space-group inputs
that are themselves observer-side commitments. Under skepticism, with the
fuller operator set (Path A enumeration of Coxeter quotients |E|=2..8 +
freq-weighted Bayesian retention), what does substrate-only MDL pick?

This probe extends sector_coxeter_freq_weighted_audit.py to:
  1. Enumerate a more comprehensive Coxeter menu (finite + affine, |E|=2..8).
  2. Compute Φ−L+min(freq_factor, 0) at framework scale N_hub = 10^60.
  3. RANK all menu items by combined weight.
  4. Report top-K dominant items.
  5. Specifically check whether |E|=3 (the framework's k*=3 commitment)
     is dominant or buried in the ranking.

If |E|=3 srs-related items are NOT at the top: framework's k*=3 is
observer-side (not MDL-dominant from substrate alone).
If |E|=3 dominates: framework's k*=3 is substrate-internally dominant.

GROWTH-CLASS CONVENTIONS

  Finite Coxeter:    |W(M, N)| ≤ |W| (constant in N)
                     log₂|elements| = log₂|W|
                     Φ = max(0, log₂(F_inv_count) − log₂|W|)

  Affine Coxeter:    |W(M, N)| ~ |W_finite| · N^r where r = finite rank
                     log₂|elements| ≈ r·log₂(N) + log₂|W_finite|
                     (polynomial growth of degree r)

  Free baseline:     no relations, Φ = 0 by definition (substrate's own DL)

ENCODING CONVENTIONS (matching freq-weighted audit):
  L(M) = sum over pairs of L_elias(m_ij)
  L_elias(m) = 1 + 2·floor(log₂ m) for finite m; L_elias(∞) = 1.
  freq_factor = log₂(N) − max(L_r) · log₂(|E|)
  max(L_r) = 2·max(m_ij) (relation length for (T_iT_j)^m).

This is bookkeeping/ranking. No new framework structure proposed.
No theorems / predictions / ledger touched. Pure substrate-menu enumeration.
"""

import math
import os
import sys


# ----------------------------------------------------------------------------
# Encoding / counting primitives (matching freq-weighted audit)
# ----------------------------------------------------------------------------

def L_elias(m):
    if m == float('inf'):
        return 1.0
    return 1 + 2 * math.floor(math.log2(m))


def L_total(m_pairs, E):
    """Sum L_elias over all C(E,2) pairs; default m=2 (commuting) for unspecified."""
    total = 0.0
    for i in range(1, E + 1):
        for j in range(i + 1, E + 1):
            m = m_pairs.get((i, j), 2)
            total += L_elias(m)
    return total


def F_inv_log_count(E, N):
    """log₂ of # reduced words length ≤ N in F_inv(E)."""
    if N == 0 or E == 0:
        return 0.0
    if E == 1:
        return 1.0 if N >= 1 else 0.0
    if E == 2:
        return math.log2(2 * N + 1) if N > 0 else 0.0
    return N * math.log2(E - 1) + math.log2(E / (E - 2))


def Phi_finite(E, order, N):
    f_log = F_inv_log_count(E, N)
    w_log = math.log2(order)
    return max(0.0, f_log - w_log)


def Phi_affine(E, finite_W_order, finite_rank, N):
    """Polynomial growth: log₂|elements ≤ N| ≈ r·log₂(N+1) + log₂|W_finite|."""
    f_log = F_inv_log_count(E, N)
    w_log = finite_rank * math.log2(N + 1) + math.log2(finite_W_order)
    return max(0.0, f_log - w_log)


def Phi_free(E, N):
    """Free baseline: F_inv(|E|) itself, no compression."""
    return 0.0


def max_relation_length(m_pairs):
    max_m = 2
    for m in m_pairs.values():
        if m == float('inf'):
            continue
        if m > max_m:
            max_m = m
    return 2 * max_m


def freq_factor_log(E, max_L_r, N):
    if N <= 0:
        return float('-inf')
    return math.log2(N) - max_L_r * math.log2(E)


def combined_weight(Phi, L, ff):
    return Phi - L + min(ff, 0.0)


# ----------------------------------------------------------------------------
# Comprehensive Path A menu enumeration |E| = 2..8
# Class: 'finite' (with W order), 'affine' (with finite_W_order + finite_rank),
#        'free' (baseline).
# ----------------------------------------------------------------------------

# Helper: build path-graph m_pairs (linear A_n / D_n / E_n style)
def path_pairs(E, m=3):
    return {(i, i + 1): m for i in range(1, E)}


def _Bn_pairs(E, m=4):
    """B_n: m=4 between first two, m=3 elsewhere (T_1 T_2 has m=4, rest m=3)."""
    p = {(i, i + 1): 3 for i in range(2, E)}
    p[(1, 2)] = m
    return p


def _Dn_pairs(E):
    """D_n: branching at the end (rank n). For our purposes, |E| = n."""
    p = {(i, i + 1): 3 for i in range(1, E - 1)}
    p[(E - 2, E)] = 3   # branching node
    return p


# --------- Finite Coxeter / Weyl ---------
finite_systems = [
    # |E|=2: I_2(p) dihedrals
    *[{'E': 2, 'name': f'I_2({p}) dihedral', 'm_pairs': {(1, 2): p}, 'order': 2 * p, 'class': 'finite'}
      for p in [2, 3, 4, 5, 6, 8, 12, 16, 24]],
    # |E|=3
    {'E': 3, 'name': '(Z/2)^3', 'm_pairs': {(1, 2): 2, (1, 3): 2, (2, 3): 2}, 'order': 8, 'class': 'finite'},
    {'E': 3, 'name': 'A_3=S_4 (tetrahedral)', 'm_pairs': {(1, 2): 3, (2, 3): 3}, 'order': 24, 'class': 'finite'},
    {'E': 3, 'name': 'B_3 (octahedral)', 'm_pairs': {(1, 2): 4, (2, 3): 3}, 'order': 48, 'class': 'finite'},
    {'E': 3, 'name': 'H_3 (icosahedral)', 'm_pairs': {(1, 2): 5, (2, 3): 3}, 'order': 120, 'class': 'finite'},
    # |E|=4
    {'E': 4, 'name': '(Z/2)^4', 'm_pairs': {}, 'order': 16, 'class': 'finite'},
    {'E': 4, 'name': 'A_4 = S_5', 'm_pairs': path_pairs(4, 3), 'order': 120, 'class': 'finite'},
    {'E': 4, 'name': 'B_4', 'm_pairs': _Bn_pairs(4), 'order': 384, 'class': 'finite'},
    {'E': 4, 'name': 'D_4', 'm_pairs': _Dn_pairs(4), 'order': 192, 'class': 'finite'},
    {'E': 4, 'name': 'F_4 (rank-4 exceptional)', 'm_pairs': {(1, 2): 3, (2, 3): 4, (3, 4): 3}, 'order': 1152, 'class': 'finite'},
    {'E': 4, 'name': 'H_4 (icosahedral×)', 'm_pairs': {(1, 2): 5, (2, 3): 3, (3, 4): 3}, 'order': 14400, 'class': 'finite'},
    # |E|=5
    {'E': 5, 'name': 'A_5 = S_6', 'm_pairs': path_pairs(5, 3), 'order': 720, 'class': 'finite'},
    {'E': 5, 'name': 'B_5', 'm_pairs': _Bn_pairs(5), 'order': 3840, 'class': 'finite'},
    {'E': 5, 'name': 'D_5', 'm_pairs': _Dn_pairs(5), 'order': 1920, 'class': 'finite'},
    # |E|=6
    {'E': 6, 'name': 'A_6 = S_7', 'm_pairs': path_pairs(6, 3), 'order': 5040, 'class': 'finite'},
    {'E': 6, 'name': 'B_6', 'm_pairs': _Bn_pairs(6), 'order': 46080, 'class': 'finite'},
    {'E': 6, 'name': 'D_6', 'm_pairs': _Dn_pairs(6), 'order': 23040, 'class': 'finite'},
    {'E': 6, 'name': 'E_6 (exceptional)', 'm_pairs': {(1, 2): 3, (2, 3): 3, (3, 4): 3, (4, 5): 3, (3, 6): 3}, 'order': 51840, 'class': 'finite'},
    # |E|=7
    {'E': 7, 'name': 'A_7 = S_8', 'm_pairs': path_pairs(7, 3), 'order': 40320, 'class': 'finite'},
    {'E': 7, 'name': 'D_7', 'm_pairs': _Dn_pairs(7), 'order': 322560, 'class': 'finite'},
    {'E': 7, 'name': 'E_7 (exceptional)', 'm_pairs': {(1, 2): 3, (2, 3): 3, (3, 4): 3, (4, 5): 3, (5, 6): 3, (3, 7): 3}, 'order': 2903040, 'class': 'finite'},
    # |E|=8
    {'E': 8, 'name': 'A_8 = S_9', 'm_pairs': path_pairs(8, 3), 'order': 362880, 'class': 'finite'},
    {'E': 8, 'name': 'B_8', 'm_pairs': _Bn_pairs(8), 'order': 10321920, 'class': 'finite'},
    {'E': 8, 'name': 'D_8', 'm_pairs': _Dn_pairs(8), 'order': 5160960, 'class': 'finite'},
    {'E': 8, 'name': 'E_8 (THE exceptional)', 'm_pairs': {(1, 2): 3, (2, 3): 3, (3, 4): 3, (4, 5): 3, (5, 6): 3, (6, 7): 3, (3, 8): 3}, 'order': 696729600, 'class': 'finite'},
]

# --------- Affine Coxeter (|E| = rank + 1, polynomial growth) ---------
# For affine Ã_n, B̃_n, etc., |E| = n+1 and the finite-Weyl-group rank is n.
# Growth: |elements ≤ N| ~ |W_finite| · N^n.
affine_systems = [
    # |E|=3 affine (rank-2 finite, 2D crystal tilings)
    {'E': 3, 'name': 'Ã_2 (triangular tiling)', 'm_pairs': {(1, 2): 3, (2, 3): 3, (1, 3): 3}, 'finite_order': 6, 'finite_rank': 2, 'class': 'affine'},
    {'E': 3, 'name': 'C̃_2 (square tiling)',   'm_pairs': {(1, 2): 4, (2, 3): 4},            'finite_order': 8, 'finite_rank': 2, 'class': 'affine'},
    {'E': 3, 'name': 'G̃_2 (kagome)',          'm_pairs': {(1, 2): 6, (2, 3): 3},            'finite_order': 12, 'finite_rank': 2, 'class': 'affine'},
    # |E|=4 affine (rank-3, 3D crystal tilings)
    {'E': 4, 'name': 'Ã_3 (3D triangular)', 'm_pairs': {(1, 2): 3, (2, 3): 3, (3, 4): 3, (1, 4): 3}, 'finite_order': 24, 'finite_rank': 3, 'class': 'affine'},
    {'E': 4, 'name': 'B̃_3', 'm_pairs': {(1, 2): 4, (2, 3): 3, (3, 4): 4}, 'finite_order': 48, 'finite_rank': 3, 'class': 'affine'},
    {'E': 4, 'name': 'C̃_3', 'm_pairs': {(1, 2): 4, (2, 3): 3, (3, 4): 4}, 'finite_order': 48, 'finite_rank': 3, 'class': 'affine'},
    {'E': 4, 'name': 'D̃_3', 'm_pairs': _Dn_pairs(4), 'finite_order': 24, 'finite_rank': 3, 'class': 'affine'},
    {'E': 4, 'name': 'F̃_4', 'm_pairs': {(1, 2): 3, (2, 3): 4, (3, 4): 3, (4, 5): 3}, 'finite_order': 1152, 'finite_rank': 4, 'class': 'affine'},
    # |E|=5 affine
    {'E': 5, 'name': 'Ã_4', 'm_pairs': {(1, 2): 3, (2, 3): 3, (3, 4): 3, (4, 5): 3, (1, 5): 3}, 'finite_order': 120, 'finite_rank': 4, 'class': 'affine'},
    # |E|=7 affine
    {'E': 7, 'name': 'Ẽ_6', 'm_pairs': {(1, 2): 3, (2, 3): 3, (3, 4): 3, (4, 5): 3, (3, 6): 3, (3, 7): 3}, 'finite_order': 51840, 'finite_rank': 6, 'class': 'affine'},
    # |E|=8 affine
    {'E': 8, 'name': 'Ẽ_7', 'm_pairs': {(1, 2): 3, (2, 3): 3, (3, 4): 3, (4, 5): 3, (5, 6): 3, (3, 7): 3, (3, 8): 3}, 'finite_order': 2903040, 'finite_rank': 7, 'class': 'affine'},
]

# --------- Free baselines ---------
free_systems = [
    {'E': E, 'name': f'F_inv({E}) free baseline', 'm_pairs': {}, 'class': 'free'}
    for E in range(2, 9)
]

all_systems = finite_systems + affine_systems + free_systems


# ----------------------------------------------------------------------------
# Audit driver
# ----------------------------------------------------------------------------

def compute_W(sys, N):
    E = sys['E']
    L = L_total(sys['m_pairs'], E)
    max_L_r = max_relation_length(sys['m_pairs'])
    n_attest = E ** max_L_r if max_L_r > 0 else 1

    if sys['class'] == 'finite':
        Phi = Phi_finite(E, sys['order'], N)
    elif sys['class'] == 'affine':
        Phi = Phi_affine(E, sys['finite_order'], sys['finite_rank'], N)
    else:
        Phi = Phi_free(E, N)

    ff = freq_factor_log(E, max_L_r, N)
    W = combined_weight(Phi, L, ff)

    return {
        'sys': sys,
        'E': E,
        'name': sys['name'],
        'class': sys['class'],
        'L': L,
        'max_L_r': max_L_r,
        'N_attest': n_attest,
        'Phi': Phi,
        'freq_factor': ff,
        'W': W,
    }


def fmt_big(x, prec=3):
    """Format very large numbers as 10^k or scientific."""
    if abs(x) < 1e10:
        return f"{x:+.{prec}e}"
    sign = '+' if x > 0 else '-'
    return f"{sign}10^{math.log10(abs(x)):.2f}"


def main():
    N_hub = 10 ** 60

    print("=" * 100)
    print(" Path A FULL-MENU ranking audit at framework scale N_hub = 10^60")
    print(" Skeptical bridge probe — does substrate-only MDL pick |E| = 3?")
    print("=" * 100)
    print()

    rows = [compute_W(s, N_hub) for s in all_systems]

    # Sort by combined weight W, descending
    rows_sorted = sorted(rows, key=lambda r: -r['W'])

    print(f" Total menu items audited: {len(rows)}")
    print()
    print(" Top 30 by combined weight W = Φ − L + min(freq_factor, 0):")
    print()
    print(f" {'rank':<5} {'system':<32} {'|E|':>4} {'class':<8} {'Φ':>14} {'L':>5} {'ff':>8} {'W':>14}")
    print(" " + "-" * 96)
    for i, r in enumerate(rows_sorted[:30], 1):
        Phi_s = fmt_big(r['Phi'], 2)
        W_s = fmt_big(r['W'], 2)
        ff_s = f"{r['freq_factor']:+8.1f}"
        print(f" {i:<5} {r['name']:<32} {r['E']:>4} {r['class']:<8} {Phi_s:>14} {r['L']:>5.1f} {ff_s:>8} {W_s:>14}")
    print()

    # Per-|E| top item
    print(" Top item per |E|:")
    print()
    print(f" {'|E|':>4} {'system':<32} {'class':<8} {'Φ':>14} {'L':>5} {'W':>14}")
    print(" " + "-" * 80)
    by_E = {}
    for r in rows:
        if r['E'] not in by_E or r['W'] > by_E[r['E']]['W']:
            by_E[r['E']] = r
    for E in sorted(by_E):
        r = by_E[E]
        print(f" {E:>4} {r['name']:<32} {r['class']:<8} {fmt_big(r['Phi'], 2):>14} {r['L']:>5.1f} {fmt_big(r['W'], 2):>14}")
    print()

    # Where do |E|=3 items rank?
    print(" Where do |E| = 3 items rank in the full menu?")
    print()
    e3_rows = [(i, r) for i, r in enumerate(rows_sorted, 1) if r['E'] == 3]
    print(f" Number of |E|=3 items in audit: {len(e3_rows)}")
    print(f" Top |E|=3 item: rank {e3_rows[0][0]} of {len(rows)} ({e3_rows[0][1]['name']})")
    print(f" Best |E|=3 W = {fmt_big(e3_rows[0][1]['W'], 2)}")
    print()

    # Compare top-1 across all |E|
    top1 = rows_sorted[0]
    e3_top = e3_rows[0][1]
    delta = top1['W'] - e3_top['W']
    print(f" Margin of #1 ({top1['name']}, |E|={top1['E']}) over best |E|=3 ({e3_top['name']}):")
    print(f"   ΔW = {fmt_big(delta, 3)}")
    print()

    # Diagnostic: is the dominant trend "higher |E| wins"?
    print(" Diagnostic — dominant trend by |E|:")
    print()
    print(f" {'|E|':>4} {'top W':>14} {'class of #1':<10}")
    for E in sorted(by_E):
        r = by_E[E]
        print(f" {E:>4} {fmt_big(r['W'], 2):>14} {r['class']:<10}")
    print()

    # Verdict
    print("=" * 100)
    print(" VERDICT")
    print("=" * 100)
    print()
    if top1['E'] == 3:
        print(f" Top item is |E| = 3 ({top1['name']}). Substrate-only MDL ranking PICKS |E|=3.")
        print(" Sub-problem β closes substrate-internally on this audit.")
    elif top1['E'] > 3:
        print(f" Top item is |E| = {top1['E']} ({top1['name']}), NOT |E| = 3.")
        print(" Substrate-only MDL ranking at framework scale prefers HIGHER |E|.")
        print(f" Best |E|=3 item ({e3_top['name']}) is rank {e3_rows[0][0]} of {len(rows)}.")
        print()
        print(" Implication: framework's k* = 3 is NOT MDL-dominant from substrate")
        print(" alone under the freq-weighted Bayesian retention used in Path A.")
        print(" k* = 3 is observer-side commitment, retroactively justified by chains")
        print(" using crystal-net + Gleason + space-group inputs.")
    else:
        print(f" Top item is |E| = {top1['E']} ({top1['name']}). Surprising result.")
    print()
    print(" Honest scope flag:")
    print(" - Φ growth class: finite |W| → log₂|W| static; affine rank-r → r·log₂(N+1).")
    print(" - Hyperbolic Coxeter NOT included (growth-rate computation requires per-system PF eigenvalue).")
    print(" - Sub-menu within each |E| is representative, not exhaustive.")
    print(" - Multi-relator Path B systems NOT included (no closed-form |W|, freq-cutoff axis only).")
    print(" - Could reorder if a different growth-class convention or compression metric")
    print("   were structurally motivated. The freq-weighted methodology is what Path A")
    print("   committed to (commit 30b4bd7); this audit applies it consistently.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
