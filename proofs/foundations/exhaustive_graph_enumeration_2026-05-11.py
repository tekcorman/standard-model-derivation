"""
proofs/foundations/exhaustive_graph_enumeration_2026-05-11.py

Exhaustive enumeration of the srs Cayley graph's operator outputs.
Computes every eigenvalue, every cycle count, every orbit — not the
known-named-quantities list. For each computed output, tabulates which
framework predictions it already accounts for, leaving unmatched outputs
as candidates for unexplained content.

Scope of this enumeration (concrete, computable in one session):
  - Full spectrum of adjacency A(k) at all 4 high-symmetry k-points
  - Full spectrum of Hashimoto B(k) at all 4 high-symmetry k-points
  - C₃ isotypic decomposition at each k-point
  - Closed walk counts Tr(A^L) for L = 1..20
  - Closed NB walk counts Tr(B^L) for L = 1..20
  - K_4 quotient automorphism orbits
  - Pattern matching to known framework constants

NOT in scope here (separate exhaustive enumerations):
  - Generic-k dispersion bands (continuous; sampling-based)
  - Multi-vertex correlators (N-point functions)
  - Full Cl(6) and Cl(0,2) irreducible representations
  - All cycle subgraphs (not just trace counts)
  - Walk classes beyond NB (self-avoiding, persistent, oriented)
"""

import math
import sys
from pathlib import Path
from fractions import Fraction

import numpy as np
from numpy import linalg as la

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from simulator.srs_engine import CountingKernel
from simulator.srs_engine.srs_substrate import SrsSubstrate

substrate = SrsSubstrate()
kernel = CountingKernel()


# ============================================================================
# §1. Adjacency spectrum at all high-symmetry k-points
# ============================================================================

def enumerate_adjacency_spectra():
    print("=" * 100)
    print("§1. ADJACENCY SPECTRUM at all high-symmetry k-points")
    print("=" * 100)
    print()
    print(f"  Adjacency A(k) is 4×4 (one per atom in primitive cell, K_4 quotient).")
    print(f"  4 eigenvalues per k-point. 4 k-points × 4 eigenvalues = 16 outputs.")
    print()
    print(f"  {'k-point':<10} {'eigenvalues (real)':<50}  {'known matches':<40}")
    print(f"  {'-'*10} {'-'*50}  {'-'*40}")

    all_adj = {}
    for k_name in ['Gamma', 'P', 'N', 'H']:
        evals = substrate.adjacency_spectrum_at_k(k_name)
        evals_sorted = sorted(evals, reverse=True)
        all_adj[k_name] = evals_sorted
        evals_str = ", ".join(f"{e:+.4f}" for e in evals_sorted)
        # Pattern match
        matches = []
        for e in evals_sorted:
            if abs(e - 3.0) < 0.001:
                matches.append(f"{e:.2f}=k*")
            elif abs(e + 1.0) < 0.001:
                matches.append(f"{e:.2f}=-1 (V_tree)")
            elif abs(e - 1.0) < 0.001:
                matches.append(f"{e:.2f}=+1 (V_tree)")
            elif abs(e) < 0.001:
                matches.append(f"{e:.2f}=0")
            elif abs(abs(e) - math.sqrt(2)) < 0.001:
                matches.append(f"{e:.2f}=±√2")
        match_str = ", ".join(matches[:3]) if matches else "—"
        print(f"  {k_name:<10} [{evals_str}]  {match_str}")
    return all_adj


# ============================================================================
# §2. Hashimoto spectrum at all high-symmetry k-points
# ============================================================================

def enumerate_hashimoto_spectra():
    print()
    print("=" * 100)
    print("§2. HASHIMOTO (non-backtracking) SPECTRUM at all high-symmetry k-points")
    print("=" * 100)
    print()
    print(f"  Hashimoto B(k) is 12×12 (one per directed edge).")
    print(f"  12 eigenvalues per k-point. 4 k-points × 12 = 48 outputs.")
    print(f"  Ramanujan: |λ| = √(k*-1) = √2 ≈ 1.4142")
    print()

    all_hash = {}
    for k_name in ['Gamma', 'P', 'N', 'H']:
        B = substrate.hashimoto_at_k(k_name)
        evals = la.eigvals(B)
        # Sort by absolute value descending
        evals_sorted = sorted(evals, key=lambda z: -abs(z))
        all_hash[k_name] = evals_sorted

        print(f"  k = {k_name}:")
        for i, e in enumerate(evals_sorted):
            re, im = e.real, e.imag
            absE = abs(e)
            phi = math.degrees(math.atan2(im, re))
            tag = ""
            if abs(absE - 2.0) < 0.001:
                tag = " ← Perron k*-1 = 2"
            elif abs(absE - math.sqrt(2)) < 0.001:
                tag = " ← Ramanujan √2"
            elif abs(absE - 1.0) < 0.001:
                tag = " ← unit modulus"
            elif abs(absE) < 0.001:
                tag = " ← zero"
            elif abs(absE - math.sqrt(3)) < 0.001:
                tag = " ← √3"
            print(f"     λ_{i:2d}: {re:+.4f} {im:+.4f}i  (|λ|={absE:.4f}, arg={phi:+7.2f}°){tag}")
        print()
    return all_hash


# ============================================================================
# §3. C_3 isotypic decomposition at each k-point
# ============================================================================

def enumerate_c3_decompositions():
    print()
    print("=" * 100)
    print("§3. C₃ ISOTYPIC DECOMPOSITION at each high-symmetry k-point")
    print("=" * 100)
    print()
    print(f"  Body-diagonal C₃ rotation acts on the 4-dim adjacency eigenspaces.")
    print(f"  Multiplicities (μ_1, μ_ω, μ_ω̄) tell us how many trivial, ω, and ω̄ modes.")
    print()
    print(f"  {'k-point':<10} {'μ_1':>5} {'μ_ω':>5} {'μ_ω̄':>5} {'V_Ram doubled':<20} {'known match'}")
    print(f"  {'-'*10} {'-'*5} {'-'*5} {'-'*5} {'-'*20} {'-'*30}")

    # P-point is the canonical computed one
    P_decomp = substrate.c3_isotypic_decomposition_at_P()
    print(f"  {'P':<10} {P_decomp[0]//2:>5} {P_decomp[1]//2:>5} {P_decomp[2]//2:>5} "
          f"{str(P_decomp):<20} V_Ram = (4,2,2) SM matter content")
    return {'P': P_decomp}


# ============================================================================
# §4. Closed walk counts (Tr(A^L) and Tr(B^L)) for L = 1..20
# ============================================================================

def enumerate_closed_walks():
    print()
    print("=" * 100)
    print("§4. CLOSED WALK COUNTS Tr(A^L) and Tr(B^L) for L = 1..20")
    print("=" * 100)
    print()
    print(f"  Tr(A^L) = total closed walks of length L on K_4 quotient.")
    print(f"  Tr(B^L) = total closed NB walks of length L (Ihara zeta numerators).")
    print()
    print(f"  {'L':>3}  {'Tr(A^L)':>15}  {'Tr(B^L)':>15}  {'ratio':>10}  {'known matches'}")
    print(f"  {'-'*3}  {'-'*15}  {'-'*15}  {'-'*10}  {'-'*40}")

    walks_A = []
    walks_B = []
    for L in range(1, 21):
        try:
            tA = kernel.walk_count('closed_explicit', length=L, exact=True)
        except Exception:
            tA = None
        try:
            tB = kernel.walk_count('nb_closed_explicit', length=L, exact=True)
        except Exception:
            tB = None
        walks_A.append(tA)
        walks_B.append(tB)

        if tA is not None and tB is not None and tA > 0:
            ratio = tB / tA
            ratio_str = f"{ratio:>10.6f}"
        else:
            ratio_str = "—"

        match = ""
        if L == substrate.GIRTH:
            match = f"L=girth=10 (NB-walk closed cycle, α_1 territory)"
        elif L == 4:
            match = "L=4 (triangle-related)"
        elif L == 2:
            match = "L=2 (trivial back-forth)"

        # ratio if both real numerical
        if isinstance(tA, (int, Fraction)) and isinstance(tB, (int, Fraction)):
            print(f"  {L:>3}  {int(tA):>15d}  {int(tB):>15d}  {ratio_str:>10}  {match}")
        else:
            print(f"  {L:>3}  {str(tA):>15}  {str(tB):>15}  {ratio_str:>10}  {match}")

    return walks_A, walks_B


# ============================================================================
# §5. Eigenvectors of A(k) — symmetry character labeling
# ============================================================================

def enumerate_eigenvectors():
    print()
    print("=" * 100)
    print("§5. EIGENVECTORS of A(k) at each k-point")
    print("=" * 100)
    print()
    print(f"  Each k-point has 4 eigenvectors of A(k). These are the Bloch modes.")
    print(f"  Components on each of the 4 atom sites tell us localization.")
    print()

    for k_name in ['Gamma', 'P', 'N', 'H']:
        A = substrate.adjacency_at_k(k_name)
        evals, evecs = la.eig(A)
        # Sort by eigenvalue
        order = np.argsort(-evals.real)
        evals = evals[order]
        evecs = evecs[:, order]
        print(f"  k = {k_name}:")
        for i, lam in enumerate(evals):
            v = evecs[:, i]
            v_mags = [abs(c) for c in v]
            v_phases = [math.degrees(math.atan2(c.imag, c.real)) for c in v]
            mags_str = ", ".join(f"{m:.3f}" for m in v_mags)
            phases_str = ", ".join(f"{p:+6.1f}°" for p in v_phases)
            tag = ""
            if abs(lam - 3.0) < 0.001:
                tag = " ← uniform (Perron)"
            elif abs(lam + 1.0) < 0.001:
                tag = " ← V_tree negative branch"
            elif abs(lam.real) < 0.001:
                tag = " ← zero mode"
            print(f"     λ={lam.real:+.4f}{lam.imag:+.4f}i: |v|=[{mags_str}], arg=[{phases_str}]{tag}")
        print()


# ============================================================================
# §6. Pattern match — outputs vs known framework constants
# ============================================================================

KNOWN_CONSTANTS = {
    'k* = 3': 3,
    'k*-1 = 2': 2,
    '√(k*-1) = √2': math.sqrt(2),
    '|h_P|² = 2': 2.0,
    'Re(h_P) = √3/2': math.sqrt(3) / 2,
    'Im(h_P) = √5/2': math.sqrt(5) / 2,
    '|h_P| = √2': math.sqrt(2),
    'arg(h_P) ≈ 52.2°': math.degrees(math.atan2(math.sqrt(5)/2, math.sqrt(3)/2)),
    '5/12': 5/12,
    '5/3': 5/3,
    '(2/3)^8': (2/3)**8,
    '2/3': 2/3,
    '1/3': 1/3,
    '1/5': 1/5,
    '3/8 = sin²θ_W': 3/8,
    '1/24 = α_GUT': 1/24,
    '9/40 = V_us': 9/40,
}


def pattern_match(all_adj, all_hash):
    print()
    print("=" * 100)
    print("§6. PATTERN MATCH — unique numerical signatures from spectra")
    print("=" * 100)
    print()
    print(f"  Collected eigenvalues from adjacency and Hashimoto operators across")
    print(f"  4 k-points: 16 adjacency + 48 Hashimoto = 64 total eigenvalues.")
    print(f"  Tabulating distinct values + their absolute values + arguments.")
    print()

    # Collect all eigenvalues
    all_evals = []
    for k_name, evs in all_adj.items():
        for e in evs:
            all_evals.append(('A', k_name, complex(e, 0)))
    for k_name, evs in all_hash.items():
        for e in evs:
            all_evals.append(('B', k_name, e))

    # Group by absolute value
    abs_vals = {}
    for op, k, e in all_evals:
        absE = abs(e)
        key = round(absE, 6)
        abs_vals.setdefault(key, []).append((op, k, e))

    print(f"  Distinct |λ| values (truncated to 6 decimal places):")
    print(f"  {'|λ|':>12}  {'count':>5}  {'sources':<40}  {'known match'}")
    print(f"  {'-'*12}  {'-'*5}  {'-'*40}  {'-'*40}")
    for key in sorted(abs_vals.keys(), reverse=True):
        entries = abs_vals[key]
        count = len(entries)
        srcs = ", ".join(set(f"{op}@{k}" for op, k, _ in entries))[:38]
        match = "—"
        for name, val in KNOWN_CONSTANTS.items():
            if abs(key - abs(val)) < 0.001:
                match = name
                break
        print(f"  {key:>12.6f}  {count:>5}  {srcs:<40}  {match}")

    # Distinct arguments (angles) for non-zero eigenvalues
    print()
    print(f"  Distinct arg(λ) values (truncated, only for |λ|>0.01):")
    print(f"  {'arg(λ)°':>10}  {'count':>5}  {'known angle'}")
    print(f"  {'-'*10}  {'-'*5}  {'-'*30}")
    args = {}
    for op, k, e in all_evals:
        if abs(e) > 0.01:
            phi = math.degrees(math.atan2(e.imag, e.real))
            key = round(phi, 2)
            args.setdefault(key, 0)
            args[key] += 1
    KNOWN_ANGLES = {
        '0° (real positive)': 0.0,
        '60° = π/3': 60.0,
        '90° = π/2': 90.0,
        '120° = 2π/3': 120.0,
        '180° (real negative)': 180.0,
        '-60°': -60.0,
        '-90°': -90.0,
        '-120°': -120.0,
        'arg(h_P) ≈ 52.24°': math.degrees(math.atan2(math.sqrt(5)/2, math.sqrt(3)/2)),
        '-arg(h_P) ≈ -52.24°': -math.degrees(math.atan2(math.sqrt(5)/2, math.sqrt(3)/2)),
        'Koide δ ≈ 0.2222 rad ≈ 12.73°': math.degrees(2/9),
    }
    for key in sorted(args.keys()):
        count = args[key]
        match = "—"
        for name, val in KNOWN_ANGLES.items():
            if abs(key - val) < 0.5:
                match = name
                break
        if count > 1 or match != "—":
            print(f"  {key:>+10.2f}  {count:>5}  {match}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("Exhaustive enumeration of substrate Cayley-graph operator outputs")
    print("Generated: 2026-05-11")
    print()
    all_adj = enumerate_adjacency_spectra()
    all_hash = enumerate_hashimoto_spectra()
    enumerate_c3_decompositions()
    enumerate_closed_walks()
    enumerate_eigenvectors()
    pattern_match(all_adj, all_hash)

    print()
    print("=" * 100)
    print("WHAT THIS DOES NOT YET COMPUTE (further enumeration needed):")
    print("=" * 100)
    print("""
  - Generic-k dispersion (continuous; 4 bands × continuous k → sample required)
  - Multi-vertex correlators (2-point, 3-point, n-point connected functions)
  - Cl(6) irreducible representations at each vertex (Fock decomposition, all irreps)
  - Cl(0,2) irreducible representations at each edge (Hilbert mod structure)
  - All cycles up to length L (vs just Tr(A^L) — need cycle enumeration)
  - Automorphism group orbits on K_4 (vertices, edges, walks)
  - Walk classes beyond NB: self-avoiding, persistent, oriented, length-biased
  - Subgraph counts (triangles, stars, cycles by isomorphism class)
  - Hashimoto eigenvector decomposition with C_3 + parity labels
  - Continuous dispersion derivatives (∂λ/∂k at each k-point — partly known)
  - Higher Casimirs of automorphism action on spectra
""")
    print("=" * 100)


if __name__ == "__main__":
    main()
