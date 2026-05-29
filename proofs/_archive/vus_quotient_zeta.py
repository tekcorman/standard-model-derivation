#!/usr/bin/env python3
"""
proofs/_archive/vus_quotient_zeta.py

Stark-Terras factorization: ζ_G = ζ_{G/C3, ρ_0} × ζ_{G/C3, ρ_1} × ζ_{G/C3, ρ_2}

The ω-twisted zeta of the QUOTIENT GRAPH G/C3 (NOT a similarity transform of A_H) encodes
which prime cycles in G/C3 have C3 holonomy ω when lifted to G.

Quotient graph structure (from atom orbits):
  - Vertex α: atom 0 (C3-fixed, at (0.125, 0.125, 0.125))
  - Vertex β: atoms {1, 2, 3} (C3 orbit, collapsed to one vertex)
  - 4 bond orbits → 4 directed bonds in the quotient (with multiplicity 3 from orbit)

The Ihara-Bass formula for the quotient with ρ_1 = ω character twist:
  ζ_{G/C3, ω}(u)^{-1} = det(I - u A_{Q,ω} + (k_Q - 1) u² I)

where A_{Q,ω} is the ω-twisted adjacency of the quotient Hashimoto graph.

TEST: Compute ζ_{G/C3, ω}(u=2/3) and check if any derived quantity gives V_us = 0.225.
"""

import sys, os
import numpy as np
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from proofs.common import find_bonds, ATOMS, A_PRIM, C3_PERM, N_ATOMS

# ---------------------------------------------------------------
# Build C3 orbit structure
# ---------------------------------------------------------------

bonds_prim = find_bonds()
n_bonds = len(bonds_prim)

C3_CART = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
c3_atom = {i: int(np.argmax(C3_PERM[:, i])) for i in range(N_ATOMS)}

def bond_disp(src, tgt, cell):
    return (np.array(ATOMS[tgt])
            + cell[0]*np.array(A_PRIM[0])
            + cell[1]*np.array(A_PRIM[1])
            + cell[2]*np.array(A_PRIM[2])
            - np.array(ATOMS[src]))

prim_disps = [bond_disp(src, tgt, cell) for src, tgt, cell in bonds_prim]

def c3_of_bond(i):
    src, _, _ = bonds_prim[i]
    new_src = c3_atom[src]
    rotated = C3_CART @ prim_disps[i]
    for j, (s, t, c) in enumerate(bonds_prim):
        if s == new_src and np.allclose(prim_disps[j], rotated, atol=1e-8):
            return j
    raise ValueError(f"C3 image of bond {i} not found")

c3_map = [c3_of_bond(i) for i in range(12)]

visited = [False]*12
bond_orbits = []
for start in range(12):
    if visited[start]: continue
    b0, b1, b2 = start, c3_map[start], c3_map[c3_map[start]]
    assert c3_map[b2] == b0 and len({b0, b1, b2}) == 3
    bond_orbits.append((b0, b1, b2))
    visited[b0] = visited[b1] = visited[b2] = True
assert len(bond_orbits) == 4

# Atom orbits
atom_fixed = 0   # atom 0 is C3-fixed
atom_orbit = [1, 3, 2]  # C3 orbit: 1→3→2→1

# Quotient vertices: 0=α (fixed atom), 1=β (orbit {1,2,3})
def quotient_atom(a):
    return 0 if a == atom_fixed else 1

print("=== Quotient graph structure ===")
print(f"  Vertex α=0: atom {atom_fixed} (C3-fixed)")
print(f"  Vertex β=1: atoms {atom_orbit} (C3 orbit)")
print()
print("  4 bond orbits in G → 4 directed bonds in G/C3:")
print(f"  {'Orbit':6} {'Rep bond':30} {'Quotient src→tgt':20} {'holonomy'}")
print(f"  {'-'*6} {'-'*30} {'-'*20} {'-'*8}")

# For each bond orbit, the quotient bond is: qsrc = quotient_atom(src_0), qtgt = quotient_atom(tgt_0)
# The C3 holonomy: when lifting the quotient cycle to G, each traversal of the quotient bond
# advances the C3 index. The holonomy per edge = 1 step in the orbit (position change).
# For a bond b0=(src0,tgt0,dc0) in orbit, C3(b0)=b1=(src1,tgt1,dc1):
#   - The C3 action maps the bond's position within the orbit by +1.
# So: traversing the quotient bond once = advancing 1 step in the C3 orbit.
# Holonomy factor per traversal: ω (one step in Z_3).

omega = np.exp(2j * np.pi / 3)
quotient_bonds = []  # (qsrc, qtgt, holonomy_power)

for oi, (b0, b1, b2) in enumerate(bond_orbits):
    src0, tgt0, dc0 = bonds_prim[b0]
    qsrc = quotient_atom(src0)
    qtgt = quotient_atom(tgt0)
    # Holonomy: each traversal of quotient edge b advances C3 position by +1 (mod 3)
    # So the C3 holonomy is ω^1 per traversal.
    hol = 1   # exponent: holonomy = ω^hol
    quotient_bonds.append((qsrc, qtgt, hol, oi))
    print(f"  {oi:6} {'bond '+str(b0)+':'+str(src0)+'→'+str(tgt0):30} "
          f"{qsrc}→{qtgt}{'  ':16} ω^{hol}")

# ---------------------------------------------------------------
# Quotient Hashimoto graph
# ---------------------------------------------------------------
# The quotient G/C3 has 2 vertices (α, β) and 4 directed bonds.
# For the HASHIMOTO graph of G/C3, nodes are directed bonds of G/C3.
# NB successors in G/C3: bond j=(qsj,qtj,hj) is NB successor of i=(qsi,qti,hi)
# if qsj = qti and NOT (qtj=qsi and bond_j is reverse of bond_i).

# The quotient is a multigraph (multiple edges between vertices).
# Let's enumerate all NB successor pairs among the 4 quotient bonds.

print(f"\n=== Quotient Hashimoto adjacency (NB walk on G/C3) ===")
print("  Nodes: 4 directed bonds of G/C3")
print("  The NB condition: successor j of i requires qsrc(j) = qtgt(i), not backtrack")

n_qbonds = 4
AQ = np.zeros((n_qbonds, n_qbonds), dtype=complex)
AQ_untwisted = np.zeros((n_qbonds, n_qbonds), dtype=int)

# Also track the holonomy of each NB step
# When traversing bond i then bond j, the total C3 holonomy contributed = hol(i) + hol(j)...
# Actually the holonomy is per-edge traversal. For the Hashimoto walk,
# each step (following bond i to bond j) contributes the holonomy of bond j.

print("\n  NB adjacency (with ω-holonomy twist):")
for j, (qsj, qtj, hj, oj) in enumerate(quotient_bonds):
    for i, (qsi, qti, hi, oi) in enumerate(quotient_bonds):
        if qsj != qti:
            continue
        # Non-backtracking: j should not reverse i
        # A reverse of i=(qsi,qti,hi) would be (qtj,qsi,...) with opposite direction
        # For the Hashimoto NB condition: explicitly check if j is the reverse of i
        # Reverse bond: same edge traversed backward → (qtj, qsj) with reversed holonomy
        # In our multigraph, we need to check if bond j = reverse(bond i)
        # Simple check: is there a bond from qtj to qsj in the quotient bonds?
        # Since we have 4 bonds and the graph is symmetric, we check explicitly.
        # For now, assume the NB condition eliminates self-reversal (same orbit, opposite dir)
        # We'll handle this properly by checking if bonds i and j are in the same C3-orbit
        # and are reverses of each other.

        # Find the reverse of bond i: it goes from qti back to qsi
        # A bond going qti → qsi would be a bond in our list... let's find it
        is_reverse = False
        for i2, (qsi2, qti2, hi2, oi2) in enumerate(quotient_bonds):
            if qsi2 == qti and qti2 == qsi:
                # This is a candidate reverse. For a k*=3 graph, the backtrack
                # is the unique bond going back to the source.
                # Mark as reverse if it's the only one going qti→qsi.
                # (This depends on multiplicity.)
                pass

        # For the srs lattice with k*=3 (degree 3), each bond has exactly 1 reverse
        # and 2 NB successors. In the quotient, count how many bonds go qti→qsi.
        reverses_to_i = [jj for jj, (qsj2,qtj2,hj2,oj2) in enumerate(quotient_bonds)
                         if qsj2 == qti and qtj2 == qsi]

        # The NB condition eliminates exactly 1 successor per bond (the backtrack)
        # In the quotient, the backtrack may be a specific bond or may not be present
        # if the reverse is in a different C3 orbit.
        # For simplicity, flag the case and handle properly.

        # Include bond j as NB successor of i (will refine backtrack below)
        AQ[j, i] = omega ** hj   # ω-twisted: multiply by holonomy of bond j
        AQ_untwisted[j, i] += 1
        print(f"    bond {i} ({qsi}→{qti}) → bond {j} ({qsj}→{qtj}): "
              f"AQ[{j},{i}] = ω^{hj} = {omega**hj:.4f}")

# ---------------------------------------------------------------
# NOTE: The above includes ALL continuations, not just NB.
# For the correct Hashimoto adjacency, we need the NB (non-backtracking) version.
# Let me build it properly using the actual srs Hashimoto structure.
# ---------------------------------------------------------------

print("\n=== Correct NB Hashimoto for quotient ===")
print("Strategy: each quotient bond = C3-orbit of 3 bonds in G.")
print("Count (j,i) = NB successor in G, weighted by orbit membership.")

# Build the full (12x12) Hashimoto adjacency of G (at k=Gamma, real-space limit)
AH_k0 = np.zeros((12,12), dtype=complex)
for j2, (sj2, tj2, dcj2) in enumerate(bonds_prim):
    for i2, (si2, ti2, dci2) in enumerate(bonds_prim):
        if sj2 != ti2: continue
        dc_sum = tuple(int(dci2[d]) + int(dcj2[d]) for d in range(3))
        if tj2 == si2 and dc_sum == (0,0,0): continue
        AH_k0[j2, i2] = 1  # real-space, k=Gamma (ignore Bloch phase for orbit counting)

# The C3 orbit of bond b0 = {b0, b1=c3_map[b0], b2=c3_map[b1]}
# orbit_of[bond_index] -> which orbit (0-3)
orbit_of = {}
for oi2, (b0,b1,b2) in enumerate(bond_orbits):
    orbit_of[b0] = oi2
    orbit_of[b1] = oi2
    orbit_of[b2] = oi2

# Count NB transitions from orbit j to orbit i (within one C3 orbit group → 1 arrow in quotient)
# M_orbit[oj][oi] = number of NB transitions from any bond in orbit oi to any bond in orbit oj
# Normalized by orbit size (3): gives fractional adjacency
M_orbit = np.zeros((4,4), dtype=float)
for j2 in range(12):
    for i2 in range(12):
        if AH_k0[j2,i2] != 0:
            M_orbit[orbit_of[j2], orbit_of[i2]] += 1
# Each orbit has 3 bonds, so total transitions from orbit oi to orbit oj = 3 * (avg per bond)
# To get the "per-bond" rate for quotient, divide by 3 (orbit size)
M_orbit_per_bond = M_orbit / 3.0  # transitions per source bond per unit time

print("\n  Orbit-to-orbit NB transition matrix M[j_orbit, i_orbit] (raw counts / orbit_size):")
print("  (= expected number of NB successors in orbit j from one bond in orbit i)")
for oj2 in range(4):
    row = "  "
    for oi2 in range(4):
        row += f"{M_orbit_per_bond[oj2,oi2]:.3f}  "
    print(row)

# Sum of each column should be k*-1 = 2 (exactly 2 NB successors per bond)
print(f"\n  Column sums (should = k*-1 = 2): {M_orbit_per_bond.sum(axis=0)}")

# The quotient Hashimoto adjacency in orbit-orbit basis:
# Each quotient node = one C3 orbit (= one directed bond type in G/C3)
# AQ_correct[j,i] = (number of NB transitions from orbit i to orbit j) / 3
# = M_orbit[j,i] / 3 = M_orbit_per_bond[j,i]
# With ω-twist: multiply by omega^(orbit C3 holonomy)
# The C3 holonomy per NB step from bond b (orbit i) to bond b' (orbit j):
# holonomy = orbit position of b' - orbit position of b (mod 3)

# Build holonomy-weighted orbit matrix
orbit_pos = {}
for oi2, (b0,b1,b2) in enumerate(bond_orbits):
    orbit_pos[b0] = 0
    orbit_pos[b1] = 1
    orbit_pos[b2] = 2

M_holonomy = np.zeros((4,4), dtype=complex)
for j2 in range(12):
    for i2 in range(12):
        if AH_k0[j2,i2] != 0:
            delta_pos = (orbit_pos[j2] - orbit_pos[i2]) % 3
            M_holonomy[orbit_of[j2], orbit_of[i2]] += omega ** delta_pos

M_holonomy /= 3.0  # normalize by orbit size

print("\n  ω-twisted orbit transition matrix (with holonomy phases, /3):")
for oj2 in range(4):
    row = "  "
    for oi2 in range(4):
        val = M_holonomy[oj2,oi2]
        row += f"({val.real:+.3f}{val.imag:+.3f}j)  "
    print(row)

# ---------------------------------------------------------------
# Ihara-Bass for the quotient Hashimoto
# ---------------------------------------------------------------

z = 2/3
k_star = 3

print(f"\n=== Ihara-Bass det(I - z M_holonomy + (k*-1)z²I) at z={z} ===")

# Untwisted (M_orbit_per_bond)
M0 = np.eye(4) - z * M_orbit_per_bond + (k_star-1) * z**2 * np.eye(4)
det0 = np.linalg.det(M0)
print(f"  Untwisted:  det = {det0.real:.8f}+{det0.imag:.8f}j")
print(f"  log det (untwisted) = {np.log(det0).real:.8f}+{np.log(det0).imag:.8f}j")

# ω-twisted
Mw = np.eye(4) - z * M_holonomy + (k_star-1) * z**2 * np.eye(4)
detw = np.linalg.det(Mw)
logdetw = np.log(detw)
print(f"  ω-twisted:  det = {detw.real:.8f}+{detw.imag:.8f}j")
print(f"  log det (ω-twisted) = {logdetw.real:.8f}+{logdetw.imag:.8f}j")
print(f"  |log det| = {abs(logdetw):.8f}")

# ω²-twisted
omega2 = np.exp(4j*np.pi/3)
M_holonomy2 = np.zeros((4,4), dtype=complex)
for j2 in range(12):
    for i2 in range(12):
        if AH_k0[j2,i2] != 0:
            delta_pos = (orbit_pos[j2] - orbit_pos[i2]) % 3
            M_holonomy2[orbit_of[j2], orbit_of[i2]] += omega2 ** delta_pos
M_holonomy2 /= 3.0

Mw2 = np.eye(4) - z * M_holonomy2 + (k_star-1) * z**2 * np.eye(4)
detw2 = np.linalg.det(Mw2)
logdetw2 = np.log(detw2)
print(f"  ω²-twisted: det = {detw2.real:.8f}+{detw2.imag:.8f}j")
print(f"  log det (ω²-twisted) = {logdetw2.real:.8f}+{logdetw2.imag:.8f}j")

# Factorization check: log ζ_G(z) should equal sum of three quotient factors
# (The BZ integral of log det for G at Gamma should = log det_Q,ρ_0 + log det_Q,ρ_1 + log det_Q,ρ_2)
# Here we use the k=Gamma (real-space) version for comparison
AH_bloch_G = AH_k0
M_G = np.eye(12) - z * AH_bloch_G + (k_star-1) * z**2 * np.eye(12)
det_G = np.linalg.det(M_G)
logdet_G = np.log(det_G)
print(f"\n  Full G at k=Γ: det = {det_G.real:.6f}, log det = {logdet_G.real:.6f}")
print(f"  Sum of three quotient factors: {(np.log(det0) + logdetw + logdetw2).real:.6f}")
print(f"  (Should be equal by Stark-Terras if the construction is correct)")

# ---------------------------------------------------------------
# Extract V_us candidate from ω-twisted factor
# ---------------------------------------------------------------

g_over_e = 10 / np.e
V_us_pdg = 0.22501

print(f"\n=== V_us candidates from ω-twisted quotient zeta ===")
print(f"  log ζ_Q,ω(z) = -log det_ω = {(-logdetw).real:.8f}+{(-logdetw).imag:.8f}j")
Z_Q_omega = -logdetw
print(f"  |Z_Q,ω| = {abs(Z_Q_omega):.8f}")

print(f"\n  (2/3)^|Z_Q,ω| = {(2/3)**abs(Z_Q_omega):.8f}")
print(f"  (2/3)^Re(Z_Q,ω) = {(2/3)**Z_Q_omega.real:.8f}")
print(f"  exp(-|Z_Q,ω|) = {np.exp(-abs(Z_Q_omega)):.8f}")

print(f"\n  Target V_us = {V_us_pdg}")
print(f"  Target g/e  = {g_over_e:.8f}")
print(f"  (2/3)^{{g/e}} = {(2/3)**g_over_e:.8f}")

# Also look at the imaginary part
print(f"\n  Im(Z_Q,ω) / (√3/2) = {Z_Q_omega.imag / (np.sqrt(3)/2):.8f}")
print(f"  This = Σ_{{[C]: hol=ω}} u^|C| - Σ_{{[C]: hol=ω²}} u^|C|")
print(f"  By C3 symmetry of srs, should = 0.  Value: {Z_Q_omega.imag:.2e}")

# The real part of Z_Q,ω comes from the cos(2π/3)=-1/2 sector:
print(f"\n  Re(Z_Q,ω) = -1/2 × (A_ω + A_ω²) + A_0")
print(f"  where A_k = Σ_{{[C]: hol=k}} Σ_n u^{{n|C|}}/n")
print(f"\n  Comparing magnitudes:")
print(f"  |Z_Q,ω| = {abs(Z_Q_omega):.6f}  vs  g/e = {g_over_e:.6f}")
diff = abs(Z_Q_omega) - g_over_e
print(f"  Difference: {diff:.6f}")
ratio = abs(Z_Q_omega) / g_over_e
print(f"  Ratio |Z_Q,ω| / (g/e) = {ratio:.6f}")
