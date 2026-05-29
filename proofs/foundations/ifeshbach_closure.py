#!/usr/bin/env python3
"""
I-Feshbach closure attempt: diagnosis of the eigenspace formulation failure
and the sublattice reformulation.

=============================================================================
FINDINGS (2026-04-18)
=============================================================================

ATTEMPT 1 (FAILED): Eigenspace Feshbach on B(k_P)
---------------------------------------------------
Claim tested: Use P = projector onto V_Ram (eigenvalues h,h*,-h,-h*; 8-dim)
and Q = I-P (V_tree, eigenvalues ±1; 4-dim) in

  C_n = P @ B(k_P) @ (Q @ B(k_P))^n @ Q @ B(k_P) @ P

Expected: C_0 = ... = C_{g-4} = 0, C_{g-2} = C_8 ∝ α₁_bare.
Result:   ALL C_n = 0 to machine precision.

Reason (algebraic, not numerical):
  If P and Q are eigenspace projectors of an operator B, then B commutes
  with P and Q: BPv = P(Bv) = P(λv) = λPv, so PBQ = PQB = 0·B = 0.
  The entire Feshbach series C_n = PB(QB)^n QP vanishes identically.
  This is a theorem, not a numerical accident.

CONSEQUENCE: The eigenspace-Feshbach formulation is STRUCTURALLY BLOCKED.
  It cannot produce a non-zero coupling α₁_bare by any computation on the
  eigenspaces of B(k_P). The document §9.5 of
  ../../predictions/Feshbach_coupling_strength_derivation.md identifies this as a tractable
  computation, but that is incorrect: the computation is trivially zero.

ATTEMPT 2: Sublattice Feshbach on the abstract K4 Hashimoto matrix
-------------------------------------------------------------------
The CORRECT formulation of Feshbach uses P and Q that do NOT diagonalize B.
On K4, the natural physical split is:

  P = directed edges incident to vertex 0 (6-dim)
  Q = directed edges among vertices {1,2,3} only (6-dim)

This splits the K4 edge space by position, not by spectrum. PBQ ≠ 0 in
this basis because B (the walk operator) connects P-edges to Q-edges.

Result of sublattice computation: see below.

INTERPRETATION:
  On K4, the sublattice-Feshbach C_n elements do not directly equal (2/3)^8
  because K4 is a FINITE quotient: walks wrap around K4 (girth 3), not the
  srs lattice (girth 10). The correct computation lives on the srs lattice
  (or its universal covering tree T_3) where the walk of length 8 can avoid
  returning to any vertex.

  On T_3 (infinite 3-regular tree), the sublattice Feshbach at order n = g-2
  involves exactly (k-1)^{g-2} paths, each with survival probability 1/k
  per step (Jaynes-uniform W4). This gives the combinatorial result:
    C_{g-2}[e_out, e_in] / (k-1)^{g-2} = (2/3)^8 = α₁_bare.

  The K4 quotient mixes walks of different LENGTHS on srs (because K4 has
  girth 3, not 10), so the K4 computation at n=8 includes contributions from
  walks that wrap around K4 multiple times, contaminating the result.

STATUS: I-Feshbach remains ADOPTED.
  Closing it at journal grade requires either:
  (A) Direct computation on the srs lattice (band operator on Z³), or
  (B) Proof that the Ihara-Bass u^{g-2} coefficient on srs gives (2/3)^{g-2}
      times a girth-cycle-count orientation factor.

  Both require analytical work beyond this finite matrix computation.
  The combinatorial theorem (Lemma 1 in ../../predictions/Feshbach_coupling_strength_derivation.md)
  proves the (2/3)^8 value on the universal covering tree. The I-Feshbach
  identification is the physical claim that this tree value = physical coupling.
=============================================================================
"""

import numpy as np
from numpy import linalg as la
from numpy.linalg import matrix_power
from fractions import Fraction
from itertools import product as iproduct
import math

np.set_printoptions(precision=6, linewidth=120, suppress=True)

# =============================================================================
# PART 0: CONSTANTS
# =============================================================================

k_star = 3
g = 10
alpha_1_bare = (2/3)**8
alpha_1_bare_exact = Fraction(2, 3)**8

print("=" * 72)
print("I-Feshbach Closure Attempt")
print("=" * 72)
print(f"  k* = {k_star}, g = {g}")
print(f"  α₁_bare = (2/3)^8 = {alpha_1_bare_exact} ≈ {float(alpha_1_bare_exact):.10f}")
print()


# =============================================================================
# PART 1: BUILD ABSTRACT K4 HASHIMOTO MATRIX (integer entries, no Bloch phase)
# =============================================================================

print("=" * 72)
print("PART 1: Abstract K4 Hashimoto Matrix")
print("=" * 72)

vertices = [0, 1, 2, 3]
dir_edges = [(u, v) for u in vertices for v in vertices if u != v]
n_edges = len(dir_edges)
assert n_edges == 12

# B[i,j] = 1 iff edge j can precede edge i in a NB walk: head(j)==tail(i) and tail(j)!=head(i)
B = np.zeros((n_edges, n_edges), dtype=int)
for i, (u, v) in enumerate(dir_edges):
    for j, (w, x) in enumerate(dir_edges):
        if v == w and u != x:
            B[i, j] = 1

row_sums = B.sum(axis=1)
assert all(s == k_star - 1 for s in row_sums), f"Row sums: {row_sums}"
print(f"  K4: {n_edges} directed edges, B is 12×12, row sums = {k_star-1} [verified]")
print()

# Edge index map for reference
print("  Edge indices (index: (tail, head)):")
for i, (u, v) in enumerate(dir_edges):
    print(f"    e{i:2d} = ({u},{v})", end="")
    if (i + 1) % 6 == 0:
        print()
print()


# =============================================================================
# PART 2: EIGENSPACE FESHBACH — DIAGNOSIS OF FAILURE
# =============================================================================

print("=" * 72)
print("PART 2: Eigenspace Feshbach (expected to fail)")
print("=" * 72)

# Build srs Bloch Hashimoto matrix at k_P = (1/4, 1/4, 1/4)
# for comparison with the eigenspace approach

A_PRIM = np.array([
    [-0.5,  0.5,  0.5],
    [ 0.5, -0.5,  0.5],
    [ 0.5,  0.5, -0.5],
])
ATOMS = np.array([
    [1/8, 1/8, 1/8],
    [3/8, 7/8, 5/8],
    [7/8, 5/8, 3/8],
    [5/8, 3/8, 7/8],
])
N_ATOMS = 4
k_P = np.array([0.25, 0.25, 0.25])

def find_bonds():
    tol = 0.02
    bonds = []
    for i in range(N_ATOMS):
        ri = ATOMS[i]
        for j in range(N_ATOMS):
            for n1, n2, n3 in iproduct(range(-2, 3), repeat=3):
                rj = ATOMS[j] + n1*A_PRIM[0] + n2*A_PRIM[1] + n3*A_PRIM[2]
                dist = la.norm(rj - ri)
                if dist < tol:
                    continue
                if abs(dist - math.sqrt(2)/4) < tol:
                    bonds.append((i, j, (n1, n2, n3)))
    return bonds

bonds = find_bonds()
assert len(bonds) == N_ATOMS * k_star

def build_bloch_hashimoto(bonds, k_pt):
    n_bonds = len(bonds)
    B_bloch = np.zeros((n_bonds, n_bonds), dtype=complex)
    for i, (ti, hi, ni) in enumerate(bonds):
        ri_out = ATOMS[hi] + sum(ni[d]*A_PRIM[d] for d in range(3))
        for j, (tj, hj, nj) in enumerate(bonds):
            if hi == tj and ti != hj:
                phase = np.exp(2j * np.pi * np.dot(k_pt, np.array(nj)))
                B_bloch[i, j] = phase
    return B_bloch

B_bloch = build_bloch_hashimoto(bonds, k_P)

# Eigendecomposition of B(k_P)
evals, evecs = la.eigh(B_bloch + B_bloch.T.conj())  # Hermitianize to get stable eigenvectors
evals, evecs = la.eig(B_bloch)

# Sort by |eigenvalue|
idx = np.argsort(-np.abs(evals))
evals = evals[idx]
evecs = evecs[:, idx]

# |h|^2 = 2, so Ramanujan eigenvalues have |lambda|^2 = 2
h_exact = (math.sqrt(3) + 1j * math.sqrt(5)) / 2
thresh_ram = 1.3  # |lambda| > 1.3 selects Ramanujan sector (|h|=sqrt(2)≈1.414)
thresh_tree = 0.8  # |lambda| < 0.8 selects nothing on K4 abstract
# Actually on B(k_P), eigenvalues are h, h*, -h, -h* (|=sqrt(2)) and ±1

n_bloch = len(bonds)
idx_ram = [i for i in range(n_bloch) if abs(abs(evals[i]) - math.sqrt(2)) < 0.1]
idx_tree = [i for i in range(n_bloch) if abs(abs(evals[i]) - 1.0) < 0.1]

print(f"  B(k_P) eigenvalues at P-point ({k_P}):")
for i, ev in enumerate(evals):
    sector = "Ramanujan" if i in idx_ram else ("tree" if i in idx_tree else "other")
    print(f"    λ_{i} = {ev:.6f}  |λ|={abs(ev):.6f}  [{sector}]")
print()

print(f"  Ramanujan sector indices: {idx_ram} (dim={len(idx_ram)})")
print(f"  Tree sector indices:      {idx_tree} (dim={len(idx_tree)})")
print()

# Build CORRECT spectral projectors for a non-Hermitian diagonalizable matrix.
# B_bloch = V @ diag(evals) @ V^{-1}
# Spectral projector onto lambda-sector: P = V @ diag(mask) @ V^{-1}
# This is the Riesz projector; it satisfies P^2 = P and BP = PB = lambda*P.

V = evecs  # right eigenvectors (columns)
V_inv = la.inv(V)  # = left eigenvectors (rows) for a diagonalizable matrix

mask_ram = np.zeros(n_bloch)
mask_tree = np.zeros(n_bloch)
for i in idx_ram:
    mask_ram[i] = 1.0
for i in idx_tree:
    mask_tree[i] = 1.0

P_eig = V @ np.diag(mask_ram) @ V_inv      # spectral projector onto Ramanujan sector
Q_eig = V @ np.diag(mask_tree) @ V_inv     # spectral projector onto tree sector
# Note: P_eig + Q_eig ≠ I in general (there may be other sectors), but here all 12
# eigenvalues split into Ram (8) and tree (4), so P_eig + Q_eig = I.

print("  Spectral projectors (Riesz, V diag(mask) V^{-1}):")
id_check = la.norm(P_eig + Q_eig - np.eye(n_bloch, dtype=complex))
print(f"    ||P_eig + Q_eig - I|| = {id_check:.2e}  (should be ~0)")

print("  Commutator [B, P_eig]:")
comm_BP = B_bloch @ P_eig - P_eig @ B_bloch
print(f"    max |[B,P]| = {np.max(np.abs(comm_BP)):.2e}  (algebraically 0 for spectral projectors)")
print()

print("  Computing PBQ with spectral projectors:")
PBQ = P_eig @ B_bloch @ Q_eig
print(f"    max |PBQ| = {np.max(np.abs(PBQ)):.2e}  (should be ~0 by BP=PB=lambda*P)")
print()
print("  CONCLUSION: Eigenspace Feshbach gives C_n = 0 for ALL n.")
print("  REASON: For spectral projectors, B P = lambda_P * P and P Q = 0,")
print("    so PBQ = P*B*Q = P*(lambda_Q Q) = lambda_Q * (PQ) = 0.")
print("  This is an algebraic identity, confirmed numerically above.")
print()


# =============================================================================
# PART 3: SUBLATTICE FESHBACH — CORRECT NON-TRIVIAL FORMULATION
# =============================================================================

print("=" * 72)
print("PART 3: Sublattice Feshbach on abstract K4 (P=vertex-0 edges, Q=rest)")
print("=" * 72)

# P-edges: touch vertex 0 (either tail=0 or head=0)
# Q-edges: among vertices {1,2,3} only
p_indices = [i for i, (u, v) in enumerate(dir_edges) if u == 0 or v == 0]
q_indices = [i for i, (u, v) in enumerate(dir_edges) if u != 0 and v != 0]

print(f"  P-edges (touching vertex 0): {[(i, dir_edges[i]) for i in p_indices]}")
print(f"  Q-edges (among {{1,2,3}}):    {[(i, dir_edges[i]) for i in q_indices]}")
print()

# Build projector matrices
P_sub = np.zeros((n_edges, n_edges), dtype=float)
Q_sub = np.zeros((n_edges, n_edges), dtype=float)
for i in p_indices:
    P_sub[i, i] = 1.0
for i in q_indices:
    Q_sub[i, i] = 1.0

# Verify PBQ is non-zero
B_float = B.astype(float)
PBQ_sub = P_sub @ B_float @ Q_sub
print(f"  ||P_sub @ B @ Q_sub|| (Frobenius) = {la.norm(PBQ_sub):.4f}")
print(f"  (non-zero: sublattice P/Q do not commute with B)")
print()

# Build B_Q = Q @ B @ Q (walk restricted to Q-space)
B_Q = Q_sub @ B_float @ Q_sub

# Compute C_n = P @ B @ (B_Q)^n @ B @ P for n = 0..12
norm_factor = (k_star - 1)**8  # = 2^8 = 256 (tree-level NB walk count at n=8)

print(f"  C_n = P_sub @ B @ (B_Q)^n @ B @ P_sub")
print(f"  (B_Q = B restricted to Q-edges: walks that stay in {{1,2,3}}-edges)")
print()

# Pick the scattering pair: e_in = (1,0) incoming to vertex 0 from vertex 1
#                           e_out = (0,2) outgoing from vertex 0 to vertex 2
e_in_idx  = dir_edges.index((1, 0))   # arrives at vertex 0 from vertex 1
e_out_idx = dir_edges.index((0, 2))   # departs from vertex 0 to vertex 2

print(f"  Scattering pair: e_in = (1,0) [index {e_in_idx}],  e_out = (0,2) [index {e_out_idx}]")
print()
print(f"  {'n':>3s}   {'C_n[e_out,e_in]':>18s}   {'norm_by_(k-1)^n':>18s}   {'compare_(2/3)^n':>18s}")
print("  " + "-" * 68)

for n in range(0, 13):
    BQn = matrix_power(B_Q, n)
    C_n_mat = P_sub @ B_float @ BQn @ B_float @ P_sub
    c_val = C_n_mat[e_out_idx, e_in_idx]
    norm_n = (k_star - 1)**n if n > 0 else 1
    c_norm = c_val / norm_n
    compare = (2.0/3.0)**n
    print(f"  {n:3d}   {c_val:18.6f}   {c_norm:18.10f}   {compare:18.10f}")

print()

# Interpretation
print(f"  NOTE: C_8[e_out,e_in] / (k-1)^8 ≠ (2/3)^8 = {float(alpha_1_bare_exact):.10f}")
print(f"  REASON: K4 is finite (girth 3); Q-space on K4 is the K3 subgraph on")
print(f"  vertices {{1,2,3}}, which has girth 3. Walks in Q-space wrap around K3")
print(f"  quickly, producing contributions from walks that complete MULTIPLE")
print(f"  short cycles, not one girth-10 cycle. The K4 finite computation")
print(f"  contaminated by short-cycle wrapping.")
print()


# =============================================================================
# PART 4: WHAT THE CORRECT COMPUTATION WOULD LOOK LIKE
# =============================================================================

print("=" * 72)
print("PART 4: Correct closure — what would close I-Feshbach")
print("=" * 72)

print("""
  The combinatorial theorem (Lemma 1, ../../predictions/Feshbach_coupling_strength_derivation.md):
    On the universal covering tree T_3, NB walk survival for L steps = (2/3)^L.
    This is PROVED at journal grade.

  The I-Feshbach identification (ADOPTED):
    The Feshbach self-energy Σ(E) = PBQ(E-QBQ)^{-1}QBP on the PHYSICAL
    Hilbert space of srs has α₁_bare = (2/3)^{g-2} as its leading coefficient.

  Two closure routes:
  (A) Ihara-Bass Green's function route:
      - Write G(u)_{e_out,e_in} = Σ_n (B^n)_{e_out,e_in} u^n on srs.
      - Show u^{g-2} coefficient = (B^{g-2})_{e_out,e_in} equals
        (k-1)^{g-2} · (2/3)^{g-2} / (girth-cycle-count normalisation).
      - This is a spectral graph theory calculation on the srs lattice
        (infinite periodic), not on K4.

  (B) Physical P/Q definition route:
      - Define P = "visible" sector and Q = "dark" sector on physical grounds
        (not via eigenvalue decomposition of B).
      - Show that P and Q do NOT commute with B (so PBQ ≠ 0).
      - Show the Feshbach kernel at leading order equals (2/3)^{g-2}.
      - This requires the dark_sector definition (docs/dark_correction_theorem).

  NEITHER route reduces to the finite K4 matrix computation proposed in §9.5.
  The K4 computation gives either zero (eigenspace) or contaminated values
  (sublattice), because K4 is too small (girth 3 ≪ 10) to see the correct
  long-range girth-10 behaviour.

  STATUS: I-Feshbach remains ADOPTED.
  Closure requires multi-session analytical work, not this computation.
""")


# =============================================================================
# PART 5: WHAT IS NUMERICALLY VERIFIED
# =============================================================================

print("=" * 72)
print("PART 5: What IS numerically verified")
print("=" * 72)

# Verify α₁_bare = (2/3)^8 as NB walk probability on the tree
# (this is the strict-solid part)

# On the (abstract) K4 Hashimoto matrix:
# (B^8)_{e_out, e_in} for a generation-changing pair, normalized by total walks
B8 = matrix_power(B, 8)

in_idx_0  = [i for i, (u, v) in enumerate(dir_edges) if v == 0]
out_idx_0 = [i for i, (u, v) in enumerate(dir_edges) if u == 0]

# Generation-changing scatter at vertex 0: e_in=(a,0), e_out=(0,b), a≠b
scatter_total = 0
scatter_count = 0
for ei in in_idx_0:
    u_in = dir_edges[ei][0]
    for eo in out_idx_0:
        v_out = dir_edges[eo][1]
        if v_out != u_in:
            scatter_total += B8[eo, ei]
            scatter_count += 1
scatter_avg = scatter_total / scatter_count

print(f"  K4 abstract B^8 generation-changing scatter average: {scatter_avg:.6f}")
print(f"  Normalized by (k-1)^8 = 256: {scatter_avg/256:.10f}")
print(f"  Compare (2/3)^8: {float(alpha_1_bare_exact):.10f}")
print()
print(f"  These differ because K4 is finite (girth 3 < g = 10).")
print(f"  The (2/3)^8 value lives on T_3 (infinite tree), not K4.")
print()
print("  STRICT-SOLID (from feshbach_exponent_principle.py):")
print(f"    α₁_bare = (2/3)^8 = {alpha_1_bare_exact} on the universal covering tree.")
print(f"    Proof: W4 Jaynes-uniform per-step survival × g-2 independent steps.")
print()
print("  ADOPTED (I-Feshbach):")
print(f"    The tree survival α₁_bare = the PHYSICAL coupling in Σ(E).")
print(f"    Gap: requires physical definition of P/Q + operator algebra on srs.")

print()
print("=" * 72)
print("FINAL VERDICT")
print("=" * 72)
print("""
  Eigenspace Feshbach: TRIVIALLY ZERO (algebraic impossibility).
  Sublattice Feshbach on K4: contaminated by short cycles (K4 girth = 3 ≠ 10).
  Combinatorial Lemma 1: PROVED. α₁_bare = (2/3)^8 on T_3.
  I-Feshbach identification: ADOPTED. Clear but non-trivial closure route.

  This file closes the question: the K4 finite matrix computation CANNOT close
  I-Feshbach, regardless of the P/Q split chosen. The closure requires
  analytical work on the infinite srs lattice.
""")
