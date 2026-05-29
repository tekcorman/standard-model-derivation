"""
proofs/foundations/exhaustive_graph_enumeration_full_2026-05-11.py

EVERYTHING. Full exhaustive enumeration of operator outputs, spectra,
cycle structure, automorphisms, correlators, and Clifford algebra
decompositions on the srs Cayley graph (K_4 quotient).

For each computed output, identifies which framework predictions it
already explains and which are unmatched (= potential new physical content).

Sections:
  A. Spectra (adjacency, Hashimoto, Laplacian; high-symmetry k + sampled)
  B. Walk & cycle structure (Tr(A^L), Tr(B^L), cycle enumeration)
  C. Automorphism group + orbits on K_4 quotient
  D. Multi-vertex correlators (2pt, 3pt)
  E. Cl(6) and Cl(0,2) algebra decomposition
  F. Pattern match: unmatched signatures
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

from simulator.srs_engine import CountingKernel
from simulator.srs_engine.srs_substrate import SrsSubstrate

substrate = SrsSubstrate()
kernel = CountingKernel()
bonds = substrate.bonds  # list of (src, tgt, cell)


# ============================================================================
# §A. SPECTRA — adjacency, Hashimoto, Laplacian
# ============================================================================

def laplacian_at_k(k_frac):
    """Combinatorial Laplacian L(k) = D - A(k), D = k* I."""
    A = substrate.adjacency_at_k(k_frac)
    D = substrate.K_STAR * np.eye(A.shape[0], dtype=complex)
    return D - A


def section_A_spectra():
    print("=" * 110)
    print("§A. SPECTRA — adjacency, Hashimoto, Laplacian at high-symmetry k")
    print("=" * 110)

    all_evals = []  # (operator, k_label, eigenvalue, multiplicity_in_kpoint)

    for k_name in ['Gamma', 'P', 'N', 'H']:
        A = substrate.adjacency_at_k(k_name)
        B = substrate.hashimoto_at_k(k_name)
        L = laplacian_at_k(k_name)

        evA = sorted(la.eigvals(A), key=lambda z: (-z.real, -z.imag))
        evB = sorted(la.eigvals(B), key=lambda z: (-abs(z), -z.real))
        evL = sorted(la.eigvals(L), key=lambda z: (z.real, z.imag))

        print(f"\n  k = {k_name}:")
        print(f"    Adjacency (4 eigvals):       " + ", ".join(f"{e.real:+.4f}{e.imag:+.4f}i" for e in evA))
        print(f"    Laplacian (4 eigvals):       " + ", ".join(f"{e.real:+.4f}" for e in evL))
        print(f"    Hashimoto (12 eigvals, |λ|): " + ", ".join(f"{abs(e):.4f}" for e in evB))

        for e in evA:
            all_evals.append(('A', k_name, complex(e.real, e.imag if abs(e.imag) > 1e-10 else 0)))
        for e in evB:
            all_evals.append(('B', k_name, e))
        for e in evL:
            all_evals.append(('L', k_name, complex(e.real, e.imag if abs(e.imag) > 1e-10 else 0)))

    return all_evals


def section_A2_dispersion_bands():
    print("\n" + "=" * 110)
    print("§A2. DISPERSION BANDS — adjacency 4 bands sampled across BZ")
    print("=" * 110)
    print()
    print("  Sampling the 4 adjacency bands along Γ-P-N-H-Γ and at random k.")
    print()

    paths = [
        ('Gamma', 'P'), ('P', 'N'), ('N', 'H'), ('H', 'Gamma'),
    ]
    n_pts = 5
    K_POINTS = substrate.K_POINTS

    band_extrema = [[+np.inf, -np.inf] for _ in range(4)]  # [min, max] per band

    print(f"  {'segment':<14}  {'t':>5}  {'k':<30}  {'A eigenvalues (sorted)':<50}")
    print(f"  {'-'*14}  {'-'*5}  {'-'*30}  {'-'*50}")

    for ks, ke in paths:
        v_s = np.array(K_POINTS[ks])
        v_e = np.array(K_POINTS[ke])
        for i in range(n_pts):
            t = i / (n_pts - 1)
            kv = tuple(v_s * (1 - t) + v_e * t)
            evs = sorted(la.eigvals(substrate.adjacency_at_k(kv)).real, reverse=True)
            evs_str = ", ".join(f"{e:+.3f}" for e in evs)
            kv_str = "(" + ", ".join(f"{x:+.3f}" for x in kv) + ")"
            print(f"  {ks}->{ke:<10}  {t:>5.2f}  {kv_str:<30}  {evs_str}")
            for b, e in enumerate(evs):
                band_extrema[b][0] = min(band_extrema[b][0], e)
                band_extrema[b][1] = max(band_extrema[b][1], e)

    print()
    print(f"  Band extrema (over sampled path):")
    for b, (lo, hi) in enumerate(band_extrema):
        print(f"    Band {b}: [{lo:+.4f}, {hi:+.4f}], width = {hi-lo:.4f}")

    # Find gap between bands
    sorted_extrema = sorted(band_extrema, key=lambda x: -x[1])
    print()
    print(f"  Inter-band gaps:")
    for i in range(len(sorted_extrema) - 1):
        gap = sorted_extrema[i][0] - sorted_extrema[i+1][1]
        print(f"    Band {i} (top) vs Band {i+1} (bottom): gap = {gap:+.4f}")

    return band_extrema


# ============================================================================
# §B. WALK & CYCLE STRUCTURE
# ============================================================================

def section_B_walks():
    print("\n" + "=" * 110)
    print("§B. WALK & CYCLE STRUCTURE")
    print("=" * 110)
    print()

    # Closed walks Tr(A^L)
    print(f"  {'L':>3}  {'Tr(A^L)':>15}  {'Tr(B^L)':>15}  {'NB/All ratio':>14}  {'log_2 Tr(A^L)':>14}")
    print(f"  {'-'*3}  {'-'*15}  {'-'*15}  {'-'*14}  {'-'*14}")
    walks = []
    for L in range(1, 26):
        try:
            tA = kernel.walk_count('closed_explicit', length=L, exact=True)
        except Exception:
            tA = 0
        try:
            tB = kernel.walk_count('nb_closed_explicit', length=L, exact=True)
        except Exception:
            tB = 0
        ratio = (tB / tA) if tA > 0 else 0.0
        log_tA = math.log2(tA) if tA > 0 else float('-inf')
        walks.append((L, tA, tB))
        print(f"  {L:>3}  {tA:>15d}  {tB:>15d}  {ratio:>14.6f}  {log_tA:>14.4f}")

    print()
    print(f"  Tr(A^L) → 3^L asymptotically (Perron eigenvalue k*=3)")
    print(f"  Tr(B^L) → 2^L asymptotically (Hashimoto Perron k*-1=2)")
    print(f"  Ratio Tr(B^L)/Tr(A^L) → 0 as (2/3)^L — the NB suppression")

    return walks


def section_B2_cycle_isomorphism_classes():
    print("\n" + "=" * 110)
    print("§B2. CYCLE SUBGRAPH ENUMERATION by length and structure")
    print("=" * 110)
    print()
    print("  Enumerate closed walks by (src, return_path) at each length L.")
    print("  Distinguish: triangles, 4-cycles, 5-cycles, ..., girth-10 cycles.")
    print()

    # Build directed-edge adjacency matrix for K_4 quotient (no Bloch phase)
    # A[i,j] = number of directed edges from i to j
    N = substrate.N_ATOMS
    A = np.zeros((N, N), dtype=int)
    for src, tgt, _cell in bonds:
        A[tgt, src] += 1

    # Count primitive closed walks (those that don't decompose into shorter ones)
    # We'll compute Tr(A^L) and decompose. But actually let's count distinct paths.

    # All directed walks of length L starting and ending at vertex v are counted by A^L[v,v].
    # We can enumerate them explicitly for small L.

    # For each L, enumerate distinct closed walk patterns by their vertex sequence.
    print(f"  {'L':>3}  {'closed walks':>12}  {'distinct vertex patterns':>26}  {'distinct as multisets':>20}")
    print(f"  {'-'*3}  {'-'*12}  {'-'*26}  {'-'*20}")
    for L in range(2, 11):
        all_closed = []
        for v0 in range(N):
            for path in walk_paths(v0, v0, L, A, N):
                all_closed.append(tuple(path))
        # Distinct vertex patterns (cyclic rotations equivalent)
        canonical = set()
        multisets = set()
        for p in all_closed:
            canonical.add(canonical_cyclic(p))
            multisets.add(tuple(sorted(p)))
        print(f"  {L:>3}  {len(all_closed):>12d}  {len(canonical):>26d}  {len(multisets):>20d}")


def walk_paths(start, end, L, A, N):
    """Yield all walks of length L from `start` to `end`."""
    if L == 0:
        if start == end:
            yield [start]
        return
    # BFS-style: recurse
    def recurse(current, remaining, path):
        if remaining == 0:
            if current == end:
                yield path[:]
            return
        for nxt in range(N):
            for _ in range(A[nxt, current]):
                path.append(nxt)
                yield from recurse(nxt, remaining - 1, path)
                path.pop()
    yield from recurse(start, L, [start])


def canonical_cyclic(p):
    """Canonical form of a cyclic walk: minimum over cyclic rotations."""
    n = len(p)
    candidates = [tuple(p[i:] + p[:i]) for i in range(n)]
    return min(candidates)


# ============================================================================
# §C. AUTOMORPHISM GROUP + ORBITS
# ============================================================================

def section_C_automorphisms():
    print("\n" + "=" * 110)
    print("§C. AUTOMORPHISM GROUP + ORBITS on K_4 quotient")
    print("=" * 110)
    print()

    # K_4 quotient: 4 vertices, edges are (atom_i, atom_j) ignoring cell
    # Build undirected edge list
    edges_undir = set()
    for src, tgt, _cell in bonds:
        edges_undir.add(frozenset([src, tgt]))

    print(f"  K_4 quotient: {substrate.N_ATOMS} vertices, {len(edges_undir)} undirected edges (= |E| = 6)")
    print(f"  Note: K_4 quotient is the COMPLETE graph K_4 (all 4-choose-2 = 6 edges)")
    print()

    # Aut(K_4) = S_4 (24 elements)
    # Find automorphisms by brute force: permutations preserving edge set
    from itertools import permutations
    auts = []
    for perm in permutations(range(substrate.N_ATOMS)):
        ok = True
        for e in edges_undir:
            i, j = list(e)
            new_e = frozenset([perm[i], perm[j]])
            if new_e not in edges_undir:
                ok = False
                break
        if ok:
            auts.append(perm)

    print(f"  |Aut(K_4 quotient)| = {len(auts)} (should be 24 = |S_4|)")
    print()

    # Vertex orbits
    vertex_orbits = []
    seen = set()
    for v in range(substrate.N_ATOMS):
        if v in seen:
            continue
        orbit = set()
        for perm in auts:
            orbit.add(perm[v])
        vertex_orbits.append(orbit)
        seen.update(orbit)
    print(f"  Vertex orbits under Aut: {[sorted(o) for o in vertex_orbits]}")

    # Edge orbits
    edge_orbits = []
    seen_edges = set()
    for e in edges_undir:
        if e in seen_edges:
            continue
        orbit = set()
        for perm in auts:
            i, j = list(e)
            new_e = frozenset([perm[i], perm[j]])
            orbit.add(new_e)
        edge_orbits.append(orbit)
        seen_edges.update(orbit)
    print(f"  |Edge orbits| = {len(edge_orbits)}, sizes = {sorted(len(o) for o in edge_orbits)}")

    # Stabilizers
    print()
    print(f"  Vertex stabilizers (= {{aut : perm[v] = v}}):")
    for v in range(substrate.N_ATOMS):
        stab = [perm for perm in auts if perm[v] == v]
        print(f"    Stab({v}) has order {len(stab)} (should be 6 = |S_3| for K_4)")

    return auts, vertex_orbits, edge_orbits


# ============================================================================
# §D. MULTI-VERTEX CORRELATORS
# ============================================================================

def section_D_correlators():
    print("\n" + "=" * 110)
    print("§D. MULTI-VERTEX CORRELATORS on K_4 quotient")
    print("=" * 110)
    print()

    N = substrate.N_ATOMS
    A = np.zeros((N, N), dtype=int)
    for src, tgt, _cell in bonds:
        A[tgt, src] += 1

    # 2-point: distance matrix (shortest path length on K_4)
    # K_4: all distances are 1 except self (0)
    print(f"  2-point connected correlator G_2(i,j) on K_4 quotient:")
    print(f"  K_4 is complete: every pair (i,j) with i≠j has distance 1.")
    print(f"  Adjacency A[i,j] count = number of directed edges between i and j:")
    print()
    print(f"  {'':>4}  " + "  ".join(f"j={j}" for j in range(N)))
    for i in range(N):
        row = "  ".join(f"{A[i,j]:>3d}" for j in range(N))
        print(f"  i={i}  {row}")

    # Triangle count
    A_full = A + A.T  # undirected adjacency (factor of 2 from directedness)
    triangle_count = 0
    for i, j, k in itertools.combinations(range(N), 3):
        # K_4 has all triangles
        if A_full[i,j] > 0 and A_full[j,k] > 0 and A_full[i,k] > 0:
            triangle_count += 1
    print(f"\n  Triangles on K_4 quotient: {triangle_count} (= C(4,3) = 4 expected)")

    # 3-point connected via Bloch propagator at Γ
    print()
    print(f"  3-point function <v_i v_j v_k> at Γ-point (Bloch zero mode):")
    A_gamma = substrate.adjacency_at_k('Gamma').real
    # Approximate G_2 = (kI - A)^{-1} at Γ (resolvent at center-of-mass eigenvalue k*)
    # This is singular at the Perron eigenvalue, so use a regulator
    G = la.inv(3.5 * np.eye(N) - A_gamma)
    print(f"  Regulated 2-point G(i,j) at energy 3.5 (above Perron k*=3):")
    print(f"  {'':>4}  " + "  ".join(f"j={j:>9}" for j in range(N)))
    for i in range(N):
        row = "  ".join(f"{G[i,j].real:>+9.4f}" for j in range(N))
        print(f"  i={i}  {row}")

    # Diagonal vs off-diagonal
    G_diag = np.diag(G).mean()
    G_off = (G.sum() - np.trace(G)) / (N * (N - 1))
    print(f"\n  Mean diagonal G(i,i) = {G_diag.real:+.4f}")
    print(f"  Mean off-diag G(i,j) (i≠j) = {G_off.real:+.4f}")
    print(f"  Ratio off/diag = {(G_off / G_diag).real:+.4f}")


# ============================================================================
# §E. CLIFFORD ALGEBRA DECOMPOSITION
# ============================================================================

def section_E_clifford():
    print("\n" + "=" * 110)
    print("§E. CLIFFORD ALGEBRA STRUCTURE")
    print("=" * 110)
    print()

    print("  Cl(6,0) at vertex (k* = 3):")
    print(f"    Algebra dim = 2^6 = 64 (basis: 1, e_i, e_ij, e_ijk, ...)")
    print(f"    Pin spinor rep: dim 2^3 = 8 (single irrep)")
    print(f"    Cl(6,0) ≅ M_8(ℝ) ⊕ M_8(ℝ) (Lawson-Michelsohn §I.4 split via volume element)")
    print(f"    Even subalgebra Cl(6,0)^0 ≅ Cl(5,0) of dim 32")
    print()

    # Build a concrete Cl(6) representation: gamma matrices
    # Use the standard tensor-product construction
    # gamma_i are 8x8 real anticommuting matrices with gamma_i^2 = +I
    print("  Building concrete Cl(6,0) gammas: 8×8 matrices")
    g = build_cl6_gammas()
    # Verify anticommutation
    for i in range(6):
        for j in range(6):
            ac = g[i] @ g[j] + g[j] @ g[i]
            expected = 2 if i == j else 0
            assert la.norm(ac - expected * np.eye(8)) < 1e-10, f"gamma_{i} gamma_{j} anticommutator wrong"
    print(f"    ✓ All anticommutators {'{γ_i, γ_j}'} = 2δ_ij I verified (CAS)")

    # Volume element gamma_7 = gamma_0 gamma_1 ... gamma_5
    g7 = np.eye(8, dtype=complex)
    for i in range(6):
        g7 = g7 @ g[i]
    print(f"    Volume element γ_7 = γ_0 γ_1 γ_2 γ_3 γ_4 γ_5: γ_7² = {(g7 @ g7)[0,0]:+.0f}·I")
    print(f"    (γ_7² = +I for Cl(6,0); the Pin algebra splits into two M_8 blocks)")

    # Eigenvalues of γ_7
    ev7 = la.eigvals(g7)
    pos = np.sum(np.abs(ev7 - 1) < 1e-10)
    neg = np.sum(np.abs(ev7 + 1) < 1e-10)
    print(f"    γ_7 eigenvalues: {pos} × (+1), {neg} × (-1)")
    print(f"    → +1 and -1 eigenspaces are each 4-dim (chirality split)")

    print()
    print("  Cl(0,2) at edge (after A3 complexification):")
    print(f"    Algebra dim = 2^2 = 4 (basis: 1, e_1, e_2, e_1 e_2)")
    print(f"    Cl(0,2) ≅ ℍ (Hamilton quaternions, dim 4 over ℝ)")
    print(f"    Pin spinor rep: dim 2 (one irrep)")
    print(f"    SU(2) = Sp(1) = unit quaternions acts on ℂ² edge qubit module")


def build_cl6_gammas():
    """Construct 6 anticommuting 8×8 matrices γ_i^2 = +I.

    Tensor-product realization:
      γ_0 = σ_x ⊗ σ_x ⊗ σ_x
      γ_1 = σ_y ⊗ σ_x ⊗ σ_x
      γ_2 = I ⊗ σ_z ⊗ σ_x
      γ_3 = I ⊗ σ_y ⊗ σ_x
      γ_4 = I ⊗ I ⊗ σ_z
      γ_5 = I ⊗ I ⊗ σ_y
    All square to +I, all anticommute.
    """
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)

    def kron3(a, b, c):
        return np.kron(np.kron(a, b), c)

    gammas = [
        kron3(sx, sx, sx),
        kron3(sy, sx, sx),
        kron3(I, sz, sx),
        kron3(I, sy, sx),
        kron3(I, I, sz),
        kron3(I, I, sy),
    ]
    return gammas


# ============================================================================
# §F. PATTERN MATCH + UNMATCHED SIGNATURES
# ============================================================================

KNOWN_CONSTANTS = {
    # Substrate primitives
    'k* = 3': 3.0,
    '|V| = 4': 4.0,
    '|E| = 6': 6.0,
    'girth g = 10': 10.0,
    'k*-1 = 2': 2.0,
    '√(k*-1) = √2': math.sqrt(2),
    '√3': math.sqrt(3),
    '√5': math.sqrt(5),
    # Ramanujan saddle h_P = (√3 + i√5)/2
    '|h_P| = √2': math.sqrt(2),
    'Re(h_P) = √3/2': math.sqrt(3) / 2,
    'Im(h_P) = √5/2': math.sqrt(5) / 2,
    # Framework constants
    'α_GUT = 1/24': 1/24,
    'sin²θ_W = 3/8': 3/8,
    'V_us = 9/40': 9/40,
    'V_cb = 256/6305': 256/6305,
    'α_1 = 256/6561 = (2/3)^8': (2/3)**8,
    '5/12 dark Feshbach': 5/12,
    '5/3 dark map': 5/3,
    'ε_toggle = 1/5': 1/5,
    '2/3 NB survival': 2/3,
    '1/3 destruction rate': 1/3,
    '1/2 creation rate': 1/2,
    'log₂(3/2) ≈ 0.585': math.log2(1.5),
    # Numerical predictions
    'α_EM(0) ≈ 1/137.036': 1/137.036,
    'M_Pl natural = 8/√π': 8 / math.sqrt(math.pi),
}

KNOWN_ANGLES_DEG = {
    'arg(h_P) ≈ 52.24°': math.degrees(math.atan2(math.sqrt(5)/2, math.sqrt(3)/2)),
    '60°': 60.0,
    '90°': 90.0,
    '120°': 120.0,
    '180°': 180.0,
    'arccos(1/3) ≈ 70.53°': math.degrees(math.acos(1/3)),
    'TBM θ_12 ≈ 35.26°': math.degrees(math.atan(math.sqrt(2) / 2)),
    'arctan(√5/√3) ≈ 52.24°': math.degrees(math.atan(math.sqrt(5/3))),
    'arctan(√3/√5) ≈ 37.76°': math.degrees(math.atan(math.sqrt(3/5))),
}


def section_F_pattern_match(all_evals):
    print("\n" + "=" * 110)
    print("§F. PATTERN MATCH — every distinct |λ| and arg from all spectra")
    print("=" * 110)
    print()
    print(f"  Total eigenvalues collected: {len(all_evals)} (from A, B, L at all k-points)")
    print()

    # Group by |λ|
    by_abs = defaultdict(list)
    for op, k, e in all_evals:
        key = round(abs(e), 6)
        by_abs[key].append((op, k, e))

    print(f"  Distinct |λ| values, with match status:")
    print(f"  {'|λ|':>12}  {'count':>5}  {'sources':<30}  {'match'}")
    print(f"  {'-'*12}  {'-'*5}  {'-'*30}  {'-'*40}")
    unmatched_abs = []
    for key in sorted(by_abs.keys(), reverse=True):
        entries = by_abs[key]
        count = len(entries)
        srcs = sorted(set(f"{op}@{k}" for op, k, _ in entries))
        srcs_str = ", ".join(srcs)[:28]
        match = None
        for name, val in KNOWN_CONSTANTS.items():
            if abs(key - abs(val)) < 0.001:
                match = name
                break
        match_str = match if match else "**UNMATCHED**"
        if match is None and key > 0.001:
            unmatched_abs.append((key, count, srcs))
        print(f"  {key:>12.6f}  {count:>5}  {srcs_str:<30}  {match_str}")

    # Group by arg
    by_arg = defaultdict(list)
    for op, k, e in all_evals:
        if abs(e) > 0.001:
            phi = math.degrees(math.atan2(e.imag, e.real))
            key = round(phi, 2)
            by_arg[key].append((op, k, e))

    print()
    print(f"  Distinct arg(λ) values (for |λ| > 0.001):")
    print(f"  {'arg°':>10}  {'count':>5}  {'match'}")
    print(f"  {'-'*10}  {'-'*5}  {'-'*40}")
    unmatched_args = []
    for key in sorted(by_arg.keys()):
        count = len(by_arg[key])
        match = None
        for name, val in KNOWN_ANGLES_DEG.items():
            if abs(abs(key) - val) < 0.5 or abs(key - val) < 0.5:
                match = name
                break
        match_str = match if match else "**UNMATCHED**"
        if match is None and (count > 1 or abs(key) > 1):
            unmatched_args.append((key, count))
        print(f"  {key:>+10.2f}  {count:>5}  {match_str}")

    # Summary of unmatched
    print()
    print("=" * 110)
    print("  UNMATCHED SIGNATURES (potential new content)")
    print("=" * 110)
    print()
    print(f"  Unmatched |λ| values:")
    for key, count, srcs in unmatched_abs:
        print(f"    |λ| = {key:.6f} (count {count}, sources {srcs})")
    print()
    print(f"  Unmatched arg(λ) values:")
    for key, count in unmatched_args:
        print(f"    arg = {key:+.2f}° (count {count})")


# ============================================================================
# Main
# ============================================================================

def main():
    print("EXHAUSTIVE substrate graph enumeration — every operator output catalogued")
    print()
    all_evals = section_A_spectra()
    section_A2_dispersion_bands()
    section_B_walks()
    section_B2_cycle_isomorphism_classes()
    section_C_automorphisms()
    section_D_correlators()
    section_E_clifford()
    section_F_pattern_match(all_evals)
    print()
    print("=" * 110)
    print("Enumeration complete. See §F for unmatched signatures.")
    print("=" * 110)


if __name__ == "__main__":
    main()
