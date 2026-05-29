"""
proofs/foundations/dispersion_and_stark_terras_2026-05-11.py

Two computations:
  (1) Generic-k dispersion sampling across 5×5×5 grid in BZ
      - 4 adjacency bands per k-point
      - Find Dirac points, band crossings, gaps
  (2) Stark-Terras factorization at each high-symmetry k-point
      - Explicit V_Ram basis for each saddle
"""

import math
import sys
from pathlib import Path

import numpy as np
from numpy import linalg as la
from collections import Counter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from simulator.srs_engine.srs_substrate import SrsSubstrate

substrate = SrsSubstrate()


# ============================================================
# §1. Generic-k dispersion
# ============================================================

def generic_k_dispersion():
    print("=" * 100)
    print("§1. Generic-k dispersion — 4 adjacency bands on 5×5×5 BZ grid")
    print("=" * 100)
    print()

    n_grid = 5
    k_grid = []
    for i in range(n_grid):
        for j in range(n_grid):
            for l in range(n_grid):
                k_grid.append((i / n_grid, j / n_grid, l / n_grid))

    print(f"  Sampling {len(k_grid)} k-points (5×5×5 grid)")
    print()

    # For each k, compute 4 eigenvalues
    all_evals = []
    for k in k_grid:
        evs = sorted(la.eigvals(substrate.adjacency_at_k(k)).real, reverse=True)
        all_evals.append((k, evs))

    # Find band extrema
    bands = [[], [], [], []]
    for k, evs in all_evals:
        for b, e in enumerate(evs):
            bands[b].append((k, e))

    print(f"  {'band':<6}  {'min':>8}  {'max':>8}  {'width':>8}  {'min @ k':<30}  {'max @ k':<30}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*30}  {'-'*30}")
    for b in range(4):
        vals = [e for _, e in bands[b]]
        lo, hi = min(vals), max(vals)
        lo_k = next(k for k, e in bands[b] if e == lo)
        hi_k = next(k for k, e in bands[b] if e == hi)
        print(f"  {b:<6}  {lo:>+8.4f}  {hi:>+8.4f}  {hi-lo:>8.4f}  "
              f"{str(lo_k):<30}  {str(hi_k):<30}")

    print()
    # Find band crossings (where two bands meet)
    print("  Band-crossing analysis:")
    crossings = []
    for k, evs in all_evals:
        # Check if any two bands are degenerate
        for i in range(len(evs) - 1):
            if abs(evs[i] - evs[i+1]) < 0.01:
                crossings.append((k, evs[i], i, i+1))
    if crossings:
        # Group by approximate location
        by_value = Counter()
        for k, val, b1, b2 in crossings:
            by_value[(round(val, 2), b1, b2)] += 1
        print(f"    Found {len(crossings)} k-points with band degeneracies")
        print(f"    {'eigenvalue':>10}  {'bands':>10}  {'count':>5}")
        for (val, b1, b2), count in sorted(by_value.items(), key=lambda x: -x[1])[:10]:
            print(f"    {val:>+10.2f}  {b1}-{b2:>5}  {count:>5}")
    else:
        print(f"    No band crossings found on this grid")

    print()
    # Find Dirac-like points (linear dispersion)
    # For srs at Γ, there's the Perron eigenvalue + flat 3-fold band at -1
    # Check k-points where band 0 = k* = 3
    perron_count = sum(1 for _, evs in all_evals if abs(evs[0] - 3) < 0.01)
    print(f"  k-points with band-0 = +3 (Perron): {perron_count}")
    # Check k-points where band 3 = -3
    anti_perron = sum(1 for _, evs in all_evals if abs(evs[3] + 3) < 0.01)
    print(f"  k-points with band-3 = -3: {anti_perron}")

    # Find gap (smallest band 0 - band 1 over grid)
    min_gap_01 = min(evs[0] - evs[1] for _, evs in all_evals)
    print(f"  Min gap between band-0 and band-1: {min_gap_01:.4f}")


# ============================================================
# §2. Stark-Terras V_Ram basis at each k-point
# ============================================================

def stark_terras_v_ram():
    print()
    print("=" * 100)
    print("§2. Stark-Terras V_Ram explicit structure at each k-point")
    print("=" * 100)
    print()
    print("  V_Ram is the 8-dim subspace of the 12-dim directed-edge Hashimoto")
    print("  space corresponding to eigenvalues with |λ| = √(k*-1) = √2.")
    print()

    for k_name in ['Gamma', 'P', 'N', 'H']:
        print(f"\n  --- {k_name} ---")
        B = substrate.hashimoto_at_k(k_name)
        evals, evecs = la.eig(B)

        # Sort by |λ|
        order = np.argsort(-np.abs(evals))
        evals_sorted = evals[order]
        evecs_sorted = evecs[:, order]

        # Find saddle eigenvalues
        saddle_mask = np.abs(np.abs(evals_sorted) - math.sqrt(2)) < 0.001
        n_saddles = np.sum(saddle_mask)
        print(f"     {n_saddles} eigenvalues with |λ| = √2 (V_Ram)")
        print(f"     V_Ram dim = {n_saddles}")

        # Distinct arg values
        saddle_evals = evals_sorted[saddle_mask]
        arg_dist = Counter()
        for e in saddle_evals:
            arg = round(math.degrees(math.atan2(e.imag, e.real)), 2)
            arg_dist[arg] += 1
        print(f"     Distinct args + multiplicities:")
        for arg, mult in sorted(arg_dist.items()):
            print(f"       arg = {arg:+.2f}°, mult = {mult}")

        # Eigenvector localization: which directed edges have non-zero weight
        # for each saddle eigenvalue
        if n_saddles > 0:
            # Group eigenvectors by arg
            for arg, mult in sorted(arg_dist.items()):
                # Find all eigenvectors with this arg
                vec_idxs = [i for i, e in enumerate(evals_sorted)
                           if abs(abs(e) - math.sqrt(2)) < 0.001
                           and abs(round(math.degrees(math.atan2(e.imag, e.real)), 2) - arg) < 0.01]
                if not vec_idxs:
                    continue
                # Combine into a subspace and look at its support
                V_sub = evecs_sorted[:, vec_idxs]
                # Total support on each directed edge
                support = np.sqrt(np.sum(np.abs(V_sub)**2, axis=1))
                # Identify edges with significant support
                threshold = 0.01
                edges_with_support = [(i, s) for i, s in enumerate(support) if s > threshold]
                edges_with_support.sort(key=lambda x: -x[1])
                # Print top 4
                top_edges = edges_with_support[:4]
                edges_str = ", ".join(f"edge{idx}({s:.3f})" for idx, s in top_edges)
                print(f"       arg={arg:+.2f}° supported on: {edges_str}...")


# ============================================================
# §3. Aut(K_4) action on Bloch eigenmodes
# ============================================================

def aut_action_on_bloch():
    print()
    print("=" * 100)
    print("§3. Aut(K_4) = S_4 action on adjacency Bloch eigenmodes")
    print("=" * 100)
    print()
    print("  S_4 acts on the 4 vertices. Action on adjacency eigenvectors:")
    print()

    # S_4 generators: τ = (0 1 2 3) cycle, σ = (0 1) transposition
    from itertools import permutations
    auts = list(permutations(range(4)))

    for k_name in ['Gamma', 'P', 'N', 'H']:
        A = substrate.adjacency_at_k(k_name)
        evals, evecs = la.eig(A)
        # Sort
        order = np.argsort(-evals.real)
        evals = evals[order]
        evecs = evecs[:, order]

        # For each eigenvalue subspace, find what S_4 elements preserve it
        # (the stabilizer subgroup of the eigenspace as a set)
        print(f"\n  --- {k_name} (eigenvalues: {[f'{e.real:+.4f}' for e in evals]}) ---")
        # Group eigenvalues
        groups = {}
        for i, e in enumerate(evals):
            key = round(e.real, 4)
            groups.setdefault(key, []).append(i)
        for ev_val, indices in groups.items():
            mult = len(indices)
            V = evecs[:, indices]  # subspace
            # Find stabilizer of this subspace under S_4
            stab = []
            for perm in auts:
                P = np.zeros((4, 4))
                for i, j in enumerate(perm):
                    P[j, i] = 1
                V_permuted = P @ V
                # Does V_permuted span the same subspace as V?
                # Compute projection of V_permuted onto V's span:
                # If V_permuted = V · U for some U, then it's preserved.
                # Use: ||V V^† V_permuted - V_permuted||
                proj = V @ (V.conj().T @ V_permuted)
                if la.norm(proj - V_permuted) < 1e-8:
                    stab.append(perm)
            print(f"     eigenvalue {ev_val:+.4f} (mult {mult}): |Stab(V)| = {len(stab)}")


def main():
    print("Exhaustive: dispersion + Stark-Terras + Aut action")
    print()
    generic_k_dispersion()
    stark_terras_v_ram()
    aut_action_on_bloch()


if __name__ == "__main__":
    main()
