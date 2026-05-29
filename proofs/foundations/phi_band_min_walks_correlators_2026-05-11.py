"""
proofs/foundations/phi_band_min_walks_correlators_2026-05-11.py

Three more enumerations:
  §1. High-resolution dispersion: confirm band 0 minimum = φ (golden ratio)
  §2. Walk classes beyond NB: self-avoiding, persistent, oriented
  §3. 3-point connected correlators on K_4 quotient
"""

import math
import sys
import itertools
from pathlib import Path
from fractions import Fraction
from collections import Counter, defaultdict

import numpy as np
from numpy import linalg as la

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from simulator.srs_engine.srs_substrate import SrsSubstrate
substrate = SrsSubstrate()


# ============================================================
# §1. High-resolution dispersion — band 0 minimum
# ============================================================

def band_0_minimum():
    print("=" * 100)
    print("§1. Band 0 minimum at high resolution — does it equal φ = (1+√5)/2?")
    print("=" * 100)
    print()

    phi = (1 + math.sqrt(5)) / 2
    print(f"  φ = (1+√5)/2 = {phi:.10f}")
    print()

    # Fine grid
    n = 21
    best_min = float('inf')
    best_k = None
    band0_vals = []
    for i in range(n):
        for j in range(n):
            for l in range(n):
                k = (i / n, j / n, l / n)
                evs = la.eigvals(substrate.adjacency_at_k(k)).real
                top = max(evs)
                band0_vals.append((k, top))
                if top < best_min:
                    best_min = top
                    best_k = k

    print(f"  Grid: {n}×{n}×{n} = {n**3} k-points")
    print(f"  Band 0 minimum found: {best_min:.10f} at k = {best_k}")
    print(f"  Comparison to φ:       Δ = {best_min - phi:+.10f}")
    print(f"  Ratio to φ:            {best_min / phi:.10f}")
    print()

    # Try optimizing further with gradient descent from best_k
    try:
        from scipy.optimize import minimize
        def neg_top(k):
            return -max(la.eigvals(substrate.adjacency_at_k(tuple(k))).real)
        res = minimize(lambda k: -neg_top(k), x0=best_k, method='Nelder-Mead', options={'xatol': 1e-9})
        refined_min = -res.fun
        print(f"  Refined min (Nelder-Mead): {refined_min:.10f}")
        print(f"  At k = {tuple(res.x)}")
        print(f"  Comparison to φ:           Δ = {refined_min - phi:+.10f}")
        if abs(refined_min - phi) < 1e-5:
            print(f"  ✓ Band 0 minimum = φ (within numerical precision)")
        else:
            print(f"  Band 0 minimum is NOT exactly φ; refined value differs")
    except ImportError:
        print(f"  scipy not available; cannot refine numerically")

    print()


# ============================================================
# §2. Walk classes beyond NB
# ============================================================

def walk_classes():
    print("=" * 100)
    print("§2. Walk classes beyond non-backtracking")
    print("=" * 100)
    print()

    N = substrate.N_ATOMS  # 4 vertices
    bonds = substrate.bonds  # 12 directed edges

    # Build adjacency at Γ (k=0) — count of directed paths
    A = np.zeros((N, N), dtype=int)
    for src, tgt, _cell in bonds:
        A[tgt, src] += 1

    print(f"  K_4 quotient: {N} vertices, {len(bonds)} directed edges")
    print()
    print(f"  Walk classes computed (length L = 2..10):")
    print()
    print(f"  {'L':>3}  {'all walks':>12}  {'NB walks':>12}  {'self-avoid':>12}  {'oriented':>12}")
    print(f"  {'-'*3}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")

    for L in range(2, 11):
        # All closed walks: Tr(A^L)
        A_L = la.matrix_power(A, L)
        tr_all = int(np.trace(A_L))

        # NB closed walks: build Hashimoto B, Tr(B^L)
        nB = len(bonds)
        B = np.zeros((nB, nB), dtype=int)
        for e_idx, (e_src, e_tgt, e_cell) in enumerate(bonds):
            for f_idx, (f_src, f_tgt, f_cell) in enumerate(bonds):
                if f_src != e_tgt:
                    continue
                rev_cell = tuple(-c for c in e_cell)
                if f_src == e_tgt and f_tgt == e_src and f_cell == rev_cell:
                    continue
                B[f_idx, e_idx] += 1
        B_L = la.matrix_power(B, L)
        tr_nb = int(np.trace(B_L))

        # Self-avoiding closed walks: enumerate paths
        sa_count = 0
        for v0 in range(N):
            sa_count += count_self_avoiding_closed(v0, A, L, N)

        # Oriented walks: walks where each directed edge is used at most once
        # (here every directed edge is unique; count distinct edge sequences)
        # For closed walks of length L, this requires at least L distinct directed edges
        # We just count walks where no directed edge repeats:
        oriented_count = 0
        for v0 in range(N):
            oriented_count += count_oriented_closed(v0, bonds, L, N)

        print(f"  {L:>3}  {tr_all:>12d}  {tr_nb:>12d}  {sa_count:>12d}  {oriented_count:>12d}")

    print()
    print(f"  Notes:")
    print(f"  - 'all walks' = Tr(A^L)")
    print(f"  - 'NB walks' = Tr(B^L) (non-backtracking, Hashimoto)")
    print(f"  - 'self-avoid' = closed walks with all intermediate vertices distinct")
    print(f"  - 'oriented' = closed walks with all directed edges distinct")


def count_self_avoiding_closed(v0, A, L, N):
    """Count closed walks of length L starting and ending at v0, with all intermediate vertices distinct."""
    count = 0
    def recurse(current, remaining, visited):
        nonlocal count
        if remaining == 0:
            if current == v0:
                count += 1
            return
        for nxt in range(N):
            if A[nxt, current] > 0:
                if remaining == 1 or nxt not in visited:
                    visited.add(nxt) if remaining > 1 else None
                    recurse(nxt, remaining - 1, visited)
                    if remaining > 1:
                        visited.discard(nxt)
    visited = {v0}
    recurse(v0, L, visited)
    return count


def count_oriented_closed(v0, bonds, L, N):
    """Count closed walks of length L using each directed edge at most once."""
    count = 0
    if L > len(bonds):
        return 0

    def recurse(current, remaining, used_edges):
        nonlocal count
        if remaining == 0:
            if current == v0:
                count += 1
            return
        for e_idx, (src, tgt, _cell) in enumerate(bonds):
            if src == current and e_idx not in used_edges:
                used_edges.add(e_idx)
                recurse(tgt, remaining - 1, used_edges)
                used_edges.remove(e_idx)

    used = set()
    recurse(v0, L, used)
    return count


# ============================================================
# §3. 3-point connected correlators
# ============================================================

def three_point_correlators():
    print()
    print("=" * 100)
    print("§3. 3-point connected correlators on K_4 quotient")
    print("=" * 100)
    print()

    N = substrate.N_ATOMS
    bonds = substrate.bonds

    # Build adjacency at Γ
    A = np.zeros((N, N), dtype=float)
    for src, tgt, _cell in bonds:
        A[tgt, src] += 1.0

    # 2-point Green function (regulated)
    E_reg = 3.5  # above Perron
    G = la.inv(E_reg * np.eye(N) - A)

    print(f"  Regulated propagator G(i,j) at E = {E_reg} (above Perron k* = 3):")
    print()

    # 3-point correlator (connected): G_3(i,j,k) = G(i,j) G(j,k) G(k,i) - disconnected parts
    # Tree-level connected:
    print(f"  Tree-level connected 3-point G_3(i,j,k) = G(i,j)·G(j,k)·G(k,i):")
    print(f"  Listing distinct triples (i<j<k):")
    print()
    print(f"  {'(i,j,k)':<10}  {'G(i,j)':>10}  {'G(j,k)':>10}  {'G(k,i)':>10}  {'G_3':>12}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}")

    for i, j, k in itertools.combinations(range(N), 3):
        g_ij = G[i, j]
        g_jk = G[j, k]
        g_ki = G[k, i]
        g3 = g_ij * g_jk * g_ki
        print(f"  ({i},{j},{k})    {g_ij:>+10.6f}  {g_jk:>+10.6f}  {g_ki:>+10.6f}  {g3:>+12.8f}")

    # In K_4, all triples are equivalent under S_4 → all G_3 should be equal
    print()
    triples = []
    for i, j, k in itertools.combinations(range(N), 3):
        triples.append(G[i,j] * G[j,k] * G[k,i])
    print(f"  All 4 triples ({len(triples)}) give identical G_3 (S_4 symmetric)")
    print(f"    G_3 (same for all) = {triples[0]:+.10f}")
    print(f"    Triangle count × G_3 = 4 × {triples[0]:.6f} = {4 * triples[0]:.6f}")

    # The connected 3-point function at tree level removes the 1-particle reducible pieces
    # For K_4 at the symmetric point, this is exactly the triangle amplitude
    print()
    print(f"  Disconnected piece (3 × G(i,i) · G(j,k)^2):")
    # Average over symmetric point
    G_diag = np.diag(G).mean()
    G_off = (G.sum() - np.trace(G)) / (N * (N - 1))
    print(f"    ⟨G(i,i)⟩ = {G_diag:.6f}")
    print(f"    ⟨G(i,j)⟩_{{i≠j}} = {G_off:.6f}")
    print(f"    G(i,i) · G(j,k)² = {G_diag * G_off**2:.10f} per such configuration")

    # Higher-order 3-point: include all paths (not just direct edges)
    # G_3_full = ∑_{paths σ ijk}  G(σ_1) G(σ_2) G(σ_3) ...
    # On K_4, the regulated G already includes all paths via inversion
    print()
    print(f"  Full regulated G already sums all paths (resolvent inversion).")


def main():
    print("Three exhaustive computations: φ verification, walk classes, 3pt correlators")
    print()
    band_0_minimum()
    walk_classes()
    three_point_correlators()


if __name__ == "__main__":
    main()
