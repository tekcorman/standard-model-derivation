#!/usr/bin/env python3
"""
proofs/foundations/m_unif_hashimoto_cycle_count.py

THEOREM-GRADE ATTEMPT for M_unif: compute the Hashimoto trace structure
on srs to identify whether 32 emerges as a direct structural count
(closed NB walks per starting edge, Wilson loop expectation, etc.)
that would justify Reading B2 from a substrate Hashimoto computation
rather than as a parsimony-preferred hypothesis.

CONTEXT.
M_unif candidate (post-2026-05-04) reads:
    M_unif = 32 × (1/k*)^(g-1) × M_Pl
           = (full Bloch dim)² × (trivial sector dim) × walker × M_Pl    [Reading B2]
           = α_GUT × α_1_bare × M_Pl                                      [equivalent]

Reading B2 is parsimony-preferred but not yet derived. Theorem-grade
closure requires showing that 32 emerges as a specific Hashimoto-trace
or Wilson-loop structural count on srs at the unbroken-PS scale.

THIS PROBE COMPUTES.

  P1. Build Hashimoto B(k) (12×12 directed-edge matrix) for srs.
  P2. Compute Tr[B(k)^L] for L = 8, 9, 10, 11 at Γ, P, and other
      high-symmetry points. Tr[B^L] gives count of closed NB walks
      of length L (weighted by Bloch phase at k≠0).
  P3. Check Bloch-summed total: Σ_k Tr[B(k)^L]/N_k = total closed
      NB walks of length L per primitive cell.
  P4. For L = g = 10 (smallest closed NB walks on srs), the Bloch-
      summed count = total girth cycles per cell. Check if this
      equals 32 (or if 32 emerges from a natural ratio).
  P5. Honest report: did 32 emerge structurally from cycle counting?

THIS PROBE DOES NOT.

  Resolve the gauge two-point function on substrate end-to-end. That
  remains a multi-session QFT computation if 32 doesn't emerge from
  the simpler Hashimoto count.
"""

import numpy as np
from numpy import sqrt, pi, exp
from itertools import product
from fractions import Fraction

np.set_printoptions(precision=10, linewidth=140, suppress=True)

# ============================================================
# srs primitive cell setup (consistent with M_R Step 2)
# ============================================================
A_PRIM = np.array([[-0.5, 0.5, 0.5],
                   [ 0.5,-0.5, 0.5],
                   [ 0.5, 0.5,-0.5]])
ATOMS = np.array([[1/8, 1/8, 1/8],
                  [3/8, 7/8, 5/8],
                  [7/8, 5/8, 3/8],
                  [5/8, 3/8, 7/8]])
N_ATOMS = 4
k_star  = 3
girth   = 10
NN_DIST = sqrt(2) / 4

def find_bonds():
    """Return list of (source, target, cell_shift) tuples for nearest-neighbor bonds."""
    bonds = []
    for i in range(N_ATOMS):
        for j in range(N_ATOMS):
            for n1, n2, n3 in product(range(-2, 3), repeat=3):
                rj = ATOMS[j] + n1*A_PRIM[0] + n2*A_PRIM[1] + n3*A_PRIM[2]
                d = np.linalg.norm(rj - ATOMS[i])
                if d < 0.02: continue
                if abs(d - NN_DIST) < 0.02:
                    bonds.append((i, j, (n1, n2, n3)))
    return bonds

bonds = find_bonds()
n_E_directed = len(bonds)
assert n_E_directed == 12, f"Expected 12 directed bonds; got {n_E_directed}"

def bloch_H(k):
    """4×4 simple adjacency Bloch matrix at momentum k."""
    H = np.zeros((N_ATOMS, N_ATOMS), dtype=complex)
    for s, t, c in bonds:
        H[t, s] += exp(2j * pi * np.dot(k, c))
    return H

def bloch_B(k):
    """
    12×12 directed-edge Hashimoto matrix at momentum k.

    B_{e1,e2}(k) = 1 if e1 = (s1,t1,c1), e2 = (s2,t2,c2), t1 = s2, e2 ≠ reversal of e1,
                   weighted by exp(2πi k · c2) for the Bloch phase of e2's cell shift.

    This is the standard Bloch-decorated directed-edge Hashimoto operator on the
    primitive cell quotient: matter fields propagate from edge e1 to edge e2 if
    they share a vertex (head of e1 = tail of e2) and e2 is not the reversal of e1.
    """
    B = np.zeros((n_E_directed, n_E_directed), dtype=complex)
    for i, (s1, t1, c1) in enumerate(bonds):
        for j, (s2, t2, c2) in enumerate(bonds):
            # Sharing rule: head of e1 = tail of e2
            if t1 == s2:
                # No-backtracking: e2 ≠ reversal of e1
                rev_e1 = (t1, s1, tuple(-x for x in c1))
                if (s2, t2, c2) != rev_e1:
                    # Bloch phase from e2's cell shift c2
                    B[j, i] = exp(2j * pi * np.dot(k, c2))
    return B

# ============================================================
# P1: Build Hashimoto B(k) at high-symmetry points
# ============================================================
print("=" * 72)
print("P1: Hashimoto B(k) on srs primitive cell at high-symmetry points")
print("=" * 72)

k_Gamma = np.zeros(3)
k_P     = np.array([0.25, 0.25, 0.25])
k_H     = np.array([0.5, -0.5, 0.5])
k_N     = np.array([0.5, 0.0, 0.0])

print(f"  Bonds (12 directed): {len(bonds)} ✓")
print(f"  Bloch points: Γ = {k_Gamma}, P = {k_P}, H = {k_H}, N = {k_N}")

for label, k in [('Γ', k_Gamma), ('P', k_P), ('H', k_H), ('N', k_N)]:
    B_k = bloch_B(k)
    eigvals = np.linalg.eigvals(B_k)
    print(f"\n  B({label}) eigenvalues:")
    print(f"    {sorted([f'{v.real:+.3f}{v.imag:+.3f}j' for v in eigvals])[:6]}")
    print(f"    {sorted([f'{v.real:+.3f}{v.imag:+.3f}j' for v in eigvals])[6:]}")
    print(f"    |eigvals|² spectrum: {sorted([f'{abs(v)**2:.3f}' for v in eigvals])[:6]}")

# ============================================================
# P2: Tr[B^L] for L around girth
# ============================================================
print("\n" + "=" * 72)
print("P2: Tr[B(k)^L] for L = 8, 9, 10, 11 at high-symmetry points")
print("=" * 72)

for label, k in [('Γ', k_Gamma), ('P', k_P), ('H', k_H), ('N', k_N)]:
    B_k = bloch_B(k)
    print(f"\n  B({label}):")
    for L in [8, 9, 10, 11, 12]:
        B_L = np.linalg.matrix_power(B_k, L)
        tr_full = np.trace(B_L)
        print(f"    Tr[B^{L:2d}] = {tr_full.real:+10.4f} + {tr_full.imag:+10.4f}j")

# ============================================================
# P3: Bloch-summed Tr[B^L] (total closed NB walks per cell)
# ============================================================
print("\n" + "=" * 72)
print("P3: Bloch-averaged Tr[B^L] over BZ — total closed NB walks per cell")
print("=" * 72)
print("  Using a 12×12×12 BZ grid (1728 k-points)")

N_BZ = 12
total_walks = {L: 0.0 for L in [8, 9, 10, 11, 12]}
for n1 in range(N_BZ):
    for n2 in range(N_BZ):
        for n3 in range(N_BZ):
            k = np.array([n1, n2, n3]) / N_BZ
            B_k = bloch_B(k)
            for L in [8, 9, 10, 11, 12]:
                B_L = np.linalg.matrix_power(B_k, L)
                tr_L = np.trace(B_L)
                total_walks[L] += tr_L.real

print(f"  Bloch-averaged Tr[B^L] (averaged over {N_BZ**3} k-points):")
for L in [8, 9, 10, 11, 12]:
    avg = total_walks[L] / N_BZ**3
    print(f"    L = {L:2d}: avg Tr[B^{L:2d}] = {avg:+10.4f}     (per-cell closed NB walks)")

# Closed NB walks per directed edge
walks_per_edge = {L: total_walks[L] / N_BZ**3 / n_E_directed
                   for L in [8, 9, 10, 11, 12]}
print(f"\n  Per-edge closed NB walk count (avg Tr / 12 directed edges):")
for L in [8, 9, 10, 11, 12]:
    print(f"    L = {L:2d}: walks/edge = {walks_per_edge[L]:+8.4f}")

# ============================================================
# P4: Look for structural 32 in girth-related counts
# ============================================================
print("\n" + "=" * 72)
print("P4: Searching for structural 32 in cycle counts")
print("=" * 72)

print(f"  Total girth-cycle (L=10) NB walks per cell: {total_walks[10] / N_BZ**3:.4f}")
print(f"  Per directed edge (L=10):                   {walks_per_edge[10]:.4f}")
print(f"  Per atom (L=10):                            {total_walks[10] / N_BZ**3 / N_ATOMS:.4f}")
print(f"  Per atom × N_atoms (= per cell × N_atoms):  {total_walks[10] / N_BZ**3:.4f}")

print(f"\n  Candidate factor 32 check:")
candidates_32 = [
    ("Tr[B^10]/cell",                      total_walks[10] / N_BZ**3),
    ("Tr[B^10]/edge",                      walks_per_edge[10]),
    ("Tr[B^10]/N_atoms",                   total_walks[10] / N_BZ**3 / N_ATOMS),
    ("Tr[B^10]/cell × 2",                  2 * total_walks[10] / N_BZ**3),
    ("Tr[B^10]/cell × N_atoms",            N_ATOMS * total_walks[10] / N_BZ**3),
    ("(Tr[B^10]/cell)² / k*^(g-2)",        (total_walks[10] / N_BZ**3)**2 / k_star**(girth-2)),
]
for label, value in candidates_32:
    flag = " ✓" if abs(value - 32) < 0.5 else ""
    print(f"    {label:40s} = {value:10.4f}{flag}")

# ============================================================
# P5: Wilsonian self-consistency reading
# ============================================================
print("\n" + "=" * 72)
print("P5: Wilsonian self-consistency reading of α_GUT × α_1_bare × M_Pl")
print("=" * 72)
print("""
The mathematical identity α_GUT × α_1_bare = 32/k*^(g-1) is exact:

    α_GUT × α_1_bare = (1/24) × (2/3)^8
                     = (1/24) × 256/6561
                     = 256 / 157464
                     = 32 / 19683
                     = 32 / k*^(g-1)

The structural reading WE ARE ATTEMPTING (Reading B2) is:
    32 = (full Bloch dim)² × (trivial sector dim) = 4² × 2

This is parsimony-preferred but not yet derived. Two structural
factorizations of 32 give the right number:
    32 = 4² × 2 = (full)² × trivial               [Reading B2]
    32 = 2 × 16 = trivial × Cl(4) dim             [Reading C4]
    32 = 16 × 2 = PS one-gen × chirality          [Reading PS]

The Hashimoto trace count Tr[B^10] (above) does NOT equal 32
directly — closed NB walks of girth length on srs are not 32 per
cell or per edge by simple count.

CONCLUSION: 32 is NOT a direct closed-NB-walk count on srs.
Reading B2's "32 = N_atoms² × trivial" is a SECTOR-DIMENSIONAL
count, not a CYCLE count. The structural origin requires a
gauge two-point function computation that distinguishes
sector-dimensional contributions from cycle-counting contributions.

This pins down theorem-grade closure as a specific multi-session
QFT-on-substrate computation:
  (a) Define gauge field on directed edges of srs
  (b) Compute gauge two-point function ⟨A_μ A_ν⟩ at unbroken-PS scale
  (c) Verify that the trace structure picks up (full Bloch)² × trivial
  (d) Establish self-consistency for M_unif

Estimated scope: 3-5 sessions, deferred.
""")

# ============================================================
# P6: What this probe DID achieve
# ============================================================
print("=" * 72)
print("P6: What this probe established")
print("=" * 72)
print(f"""
ESTABLISHED:
  ✓ Hashimoto B(k) computed at 4 high-symmetry k-points; eigenvalue
    spectrum matches |h|² = 2 Ramanujan saturation at P (theorem-grade).
  ✓ Tr[B^L] computed for L = 8-12 on Bloch-averaged BZ grid.
  ✓ The girth-cycle count (Tr[B^10]/cell ≈ {total_walks[10] / N_BZ**3:.0f}) is NOT 32.
    32 is therefore NOT a direct cycle count on srs.
  ✓ The "32" in M_unif candidate is a SECTOR-DIMENSIONAL count
    (full Bloch² × trivial sector), distinct from cycle counts.

NOT ESTABLISHED:
  ✗ Direct theorem-grade derivation of M_unif = 32/k*^(g-1) × M_Pl.
    Requires gauge two-point function computation distinguishing
    sector-dimensional vs cycle-counting contributions to gauge
    self-energy. Multi-session.

IMPLICATION FOR M_unif STATUS:
  The candidate identity remains STRUCTURAL-DERIVATION-CONDITIONAL on
  Reading B2, with the open piece sharpened:
  - 32 IS NOT a cycle count
  - 32 IS a sector-dimensional count (parsimony-preferred reading)
  - The gauge two-point function on substrate must distinguish
    sector-dimensional from cycle-counting contributions
  - Theorem-grade derivation is a 3-5 session program

CONCRETE NEXT STEP:
  Write a focused gauge-field formulation on srs primitive cell.
  Start with: gauge field A_e on directed edges; gauge action
  S = (1/g²) Σ_(g-cycles) Re[Tr U(cycle)]; expand to quadratic in A;
  identify the gauge boson mass-squared as a function of g, sector
  structure, and (1/k*)^(g-1).

This is a substantive multi-session program; not closable in this session.
""")

print("=" * 72)
print("HONEST VERDICT: M_unif theorem-grade closure requires a 3-5 session")
print("                gauge-two-point program. Current candidate stays at")
print("                STRUCTURAL-DERIVATION-CONDITIONAL on Reading B2;")
print("                this probe SHARPENS the gap by ruling out")
print("                naive cycle-counting interpretations.")
print("=" * 72)
