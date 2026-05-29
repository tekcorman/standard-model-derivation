#!/usr/bin/env python3
"""
proofs/_archive/vub_resolvent_kP.py

Bloch-Hashimoto resolvent at k_P, z=2/3, projected onto C3 isotypic
sectors. The 12-dim NB transfer matrix B_kP commutes with the C3
permutation action on bonds (4 orbits × 3 positions), so B_kP block-
diagonalizes into 3 sectors of dimension 4 each (trivial, ω, ω²).

Hypothesis: V_ub matches a specific matrix element / eigenvalue of the
resolvent in the trivial sector at k_P, while V_cb matches the
corresponding quantity in the ω/ω² sectors.

CAS verification only.
"""

import sys, os
import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))
from proofs.common import find_bonds, ATOMS, A_PRIM, C3_PERM, N_ATOMS

bonds_prim = find_bonds()
n_bonds = len(bonds_prim)
omega = np.exp(2j * np.pi / 3)
z = 2 / 3

V_cb_exact = (2/3)**8 / (1 - (2/3)**8)
V_ub_geom14 = (2/3)**14 / (1 - (2/3)**14)
V_us_form = 9/40

C3_CART = np.array([[0,0,1],[1,0,0],[0,1,0]], dtype=float)
c3_atom = {i: int(np.argmax(C3_PERM[:, i])) for i in range(N_ATOMS)}

def bond_disp(src, tgt, cell):
    return (np.array(ATOMS[tgt]) + cell[0]*np.array(A_PRIM[0])
            + cell[1]*np.array(A_PRIM[1]) + cell[2]*np.array(A_PRIM[2])
            - np.array(ATOMS[src]))

prim_disps = [bond_disp(*b) for b in bonds_prim]


def c3_of_bond(i):
    src, _, _ = bonds_prim[i]
    new_src = c3_atom[src]
    rotated = C3_CART @ prim_disps[i]
    for j, (s, t, c) in enumerate(bonds_prim):
        if s == new_src and np.allclose(prim_disps[j], rotated, atol=1e-8):
            return j
    raise ValueError(f"C3 image of bond {i} not found")


c3_map = [c3_of_bond(i) for i in range(n_bonds)]

# Build C3 permutation matrix on the 12-dim bond space
C3_BOND = np.zeros((n_bonds, n_bonds), dtype=complex)
for i in range(n_bonds):
    C3_BOND[c3_map[i], i] = 1.0

# Verify (C3_BOND)^3 = I
diff = np.linalg.norm(C3_BOND @ C3_BOND @ C3_BOND - np.eye(n_bonds))
assert diff < 1e-10, f"(C3_BOND)^3 != I, diff = {diff}"

# C3 isotypic projectors on bond space
P_trivial = (np.eye(n_bonds) + C3_BOND + C3_BOND @ C3_BOND) / 3
P_omega   = (np.eye(n_bonds) + omega**2 * C3_BOND + omega * C3_BOND @ C3_BOND) / 3
P_omega2  = (np.eye(n_bonds) + omega * C3_BOND + omega**2 * C3_BOND @ C3_BOND) / 3

# Verify projectors are idempotent and sum to identity
assert np.linalg.norm(P_trivial @ P_trivial - P_trivial) < 1e-10
assert np.linalg.norm(P_trivial + P_omega + P_omega2 - np.eye(n_bonds)) < 1e-10
print(f"  P_trivial rank: {np.linalg.matrix_rank(P_trivial):d}")
print(f"  P_omega   rank: {np.linalg.matrix_rank(P_omega):d}")
print(f"  P_omega2  rank: {np.linalg.matrix_rank(P_omega2):d}")
print()

# Build Bloch-Hashimoto at k_P
def build_BH(k):
    BH = np.zeros((n_bonds, n_bonds), dtype=complex)
    for j, (sj, tj, dcj) in enumerate(bonds_prim):
        for i, (si, ti, dci) in enumerate(bonds_prim):
            if sj != ti: continue
            dc_sum = tuple(int(dci[d]) + int(dcj[d]) for d in range(3))
            if tj == si and dc_sum == (0, 0, 0): continue
            BH[j, i] = np.exp(2j * np.pi * np.dot(k, dci))
    return BH


k_P = np.array([0.25, 0.25, 0.25])
B_kP = build_BH(k_P)

# Verify [B_kP, C3] = 0 at k_P
commutator = B_kP @ C3_BOND - C3_BOND @ B_kP
print(f"  ||[B_kP, C3]|| = {la.norm(commutator):.2e}")
assert la.norm(commutator) < 1e-9, "C3 should commute with B at k_P"
print()

# Resolvent
G_kP = la.solve(np.eye(n_bonds) - z * B_kP, np.eye(n_bonds))
print(f"=== Bloch-Hashimoto resolvent G(z={z}, k_P) ===")
print(f"  ||G|| = {la.norm(G_kP):.4f}")
print()

# Project to each sector
print(f"=== Sector-projected resolvents ===")
for sector_name, P in [("trivial", P_trivial), ("omega", P_omega), ("omega2", P_omega2)]:
    G_sector = P @ G_kP @ P
    # Eigenvalue spectrum of B_kP within this sector
    B_sector = P @ B_kP @ P
    evals_B = la.eigvals(B_sector)
    evals_B = sorted(evals_B, key=lambda x: -abs(x))
    print(f"  Sector '{sector_name}':")
    print(f"    B_kP eigenvalues (top 4):")
    for ev in evals_B[:4]:
        print(f"      {ev.real:+.5f}{ev.imag:+.5f}i   |1-z·ev| = {abs(1-z*ev):.5f}")

    # Geometric series 1/(1-z·ev) values
    g_vals = sorted([1/(1-z*ev) for ev in evals_B[:4] if abs(1-z*ev) > 1e-6],
                     key=lambda x: -abs(x))
    print(f"    G eigenvalues (= 1/(1-z·ev)):")
    for gv in g_vals:
        print(f"      {gv.real:+.5f}{gv.imag:+.5f}i   |G| = {abs(gv):.5f}")
    print()

# Now the key test: look at (off-diagonal) matrix elements between
# specific bond pairs (b1, b2) at d=8 within girth cycles.
# Then look at (b, u) candidate pairs where u is a different orbit position.
print(f"=== Off-diagonal matrix element checks ===")
print(f"  V_cb       = {V_cb_exact:.6f}")
print(f"  V_ub geom14 = {V_ub_geom14:.6f}")
print(f"  V_us 9/40   = {V_us_form:.6f}")
print()

# Compute |G[i,j]| for all (i,j) and check matches
print(f"  Looking for |G[i,j]| matches in V_ub or V_us window...")
for i in range(n_bonds):
    for j in range(n_bonds):
        if i == j: continue
        v = abs(G_kP[i, j])
        # Check matches
        if 3.4e-3 < v < 3.5e-3:
            print(f"    ** |G[{i},{j}]| = {v:.6f}  matches V_ub_geom14 = {V_ub_geom14:.6f}")
        if 0.220 < v < 0.230:
            print(f"    *  |G[{i},{j}]| = {v:.6f}  in V_us window")
        if 0.040 < v < 0.041:
            print(f"    *  |G[{i},{j}]| = {v:.6f}  in V_cb window")

print()
print(f"  Histogram of |G[i,j]| for off-diagonal entries:")
offdiag_vals = sorted([abs(G_kP[i, j]) for i in range(n_bonds) for j in range(n_bonds) if i != j], reverse=True)
unique_rounded = sorted(set(round(v, 5) for v in offdiag_vals), reverse=True)
for v in unique_rounded[:15]:
    count = sum(1 for x in offdiag_vals if abs(x - v) < 1e-5)
    tag = ""
    if abs(v - V_ub_geom14) < 1e-3: tag = "  ← V_ub geom14 nearby?"
    if abs(v - V_us_form) < 5e-3: tag = "  ← V_us nearby?"
    if abs(v - V_cb_exact) < 1e-3: tag = "  ← V_cb nearby?"
    print(f"    |G| = {v:.5f}  count = {count}{tag}")
