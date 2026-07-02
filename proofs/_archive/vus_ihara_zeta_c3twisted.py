#!/usr/bin/env python3
"""
proofs/_archive/vus_ihara_zeta_c3twisted.py

Test whether V_us arises from the C3-twisted Ihara zeta function.

HYPOTHESIS
----------
The Ihara zeta for the srs Hashimoto graph, twisted by C3 representation ω,
encodes the amplitude for prime cycles with C3 holonomy ω.

Key formula (Sunada / Hashimoto):
  log ζ_ω(u) = Σ_{[C] prime} Σ_{n≥1} ω^{n·hol(C)} × u^{n|C|} / n

where hol(C) ∈ {0,1,2} is the C3 holonomy (mod 3) of prime cycle C.

The Ihara-Bass formula for the ω-twisted zeta:
  ζ_ω(u)^{-1} = det(I - u × A_H^{(ω)} + (k*-1)u² × I)   [per k-point]

where A_H^{(ω)} is the Hashimoto matrix with each NB step (i→j) weighted by ω^{Δpos},
Δpos = (orbit_pos(j) - orbit_pos(i)) mod 3.

TEST
----
Compute Z_ω(u=2/3) = BZ integral of log det(...).
If |Z_ω(2/3)| = g/e, then V_us = (2/3)^{g/e} follows from the prime cycle density.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from proofs.common import find_bonds, ATOMS, A_PRIM, C3_PERM, N_ATOMS

# ---------------------------------------------------------------
# Lattice + orbit setup (same as vus_hashimoto_bfs.py)
# ---------------------------------------------------------------

bonds_prim = find_bonds()
n_bonds = len(bonds_prim)  # 12

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
orbits = []
for start in range(12):
    if visited[start]:
        continue
    b0, b1, b2 = start, c3_map[start], c3_map[c3_map[start]]
    assert c3_map[b2] == b0 and len({b0, b1, b2}) == 3
    orbits.append((b0, b1, b2))
    visited[b0] = visited[b1] = visited[b2] = True
assert len(orbits) == 4

# Orbit position for each bond index
orbit_pos = {}
for oi, (b0, b1, b2) in enumerate(orbits):
    orbit_pos[b0] = 0
    orbit_pos[b1] = 1
    orbit_pos[b2] = 2

pos_arr = np.array([orbit_pos[i] for i in range(n_bonds)])  # shape (12,)

print("=== Bond orbit positions ===")
for oi, (b0, b1, b2) in enumerate(orbits):
    print(f"  Orbit {oi}: b0={b0}(pos=0), b1={b1}(pos=1), b2={b2}(pos=2)")

# ---------------------------------------------------------------
# C3 twist matrix and verify it commutes correctly
# ---------------------------------------------------------------

omega = np.exp(2j * np.pi / 3)   # ω = e^{2πi/3}

# Phase for each bond: ω^{orbit_pos}
omega_phases = omega ** pos_arr   # shape (12,)

print(f"\nomega = {omega:.6f}+{omega.imag:.6f}i")
print(f"omega^3 = {(omega**3):.6f} (should be 1)")

# ω-twisted Hashimoto matrix: [A_H^(ω)(k)]_{j,i} = ω^{(pos_j - pos_i) mod 3} × [A_H(k)]_{j,i}
# Equivalently: A_H^(ω)(k) = diag(omega_phases) × A_H(k) × diag(omega_phases^{-1})
# Build the twist weight matrix W[j,i] = ω^{(pos_j - pos_i) mod 3}
W = np.outer(omega_phases, omega_phases.conj())   # W[j,i] = ω^{pos_j - pos_i}
print(f"\nTwist matrix W[j,i] = omega^(pos_j - pos_i):")
print(f"  Diagonal elements: all = {W[0,0]:.4f} (should be 1)")

# ---------------------------------------------------------------
# Bloch-Hashimoto matrix builder
# ---------------------------------------------------------------

def build_AH(k_frac):
    """Build 12×12 Bloch-Hashimoto A_H(k)."""
    AH = np.zeros((n_bonds, n_bonds), dtype=complex)
    for j, (sj, tj, dcj) in enumerate(bonds_prim):
        for i, (si, ti, dci) in enumerate(bonds_prim):
            if sj != ti:
                continue
            dc_sum = tuple(int(dci[d]) + int(dcj[d]) for d in range(3))
            if tj == si and dc_sum == (0, 0, 0):
                continue
            phase = np.exp(2j * np.pi * np.dot(k_frac, dci))
            AH[j, i] = phase
    return AH

def build_AH_twisted(k_frac, twist_W):
    """Build ω-twisted Hashimoto: A_H^(ω)(k) = W ⊙ A_H(k) (element-wise)."""
    AH = build_AH(k_frac)
    return twist_W * AH   # element-wise: twist_W[j,i] * AH[j,i]

# ---------------------------------------------------------------
# Sanity check: eigenvalue structure at a few k-points
# ---------------------------------------------------------------

k_star = 3
z = 2/3

print(f"\n=== Eigenvalues of ω-twisted A_H at selected k-points ===")
for k_label, k in [("Γ", [0,0,0]), ("P", [0.25,0.25,0.25]), ("generic", [0.1,0.2,0.3])]:
    AH_tw = build_AH_twisted(k, W)
    eigs = np.linalg.eigvals(AH_tw)
    M = np.eye(n_bonds) - z * AH_tw + (k_star-1) * z**2 * np.eye(n_bonds)
    det_val = np.linalg.det(M)
    print(f"  k={k_label}: |eigenvals| in [{np.min(np.abs(eigs)):.4f}, {np.max(np.abs(eigs)):.4f}],"
          f"  det(I - z A^ω + 2z² I) = {det_val.real:.6f}+{det_val.imag:.6f}i")

# ---------------------------------------------------------------
# BZ integral of log det(I - z A_H^(ω) + (k*-1)z² I)
# ---------------------------------------------------------------
# This equals log ζ_ω(z)^{-1}  [negated = log ζ_ω(z)]
#
# log ζ_ω(z) = Σ_{[C] prime} Σ_{n≥1} ω^{n·hol(C)} z^{n|C|} / n
#
# The ω-sector contribution (holonomy = 1, i.e. hol=0 mod 3):
# Re(log ζ_ω(z)) includes all hol=0 contributions
# The cross terms come from hol ≠ 0 cycles.

print(f"\n=== BZ integral of log det(I - z A_H^ω + 2z²I) at z={z} ===")

N_BZ = 30

acc_logdet = complex(0)
n_k = 0

for i1 in range(N_BZ):
    for i2 in range(N_BZ):
        for i3 in range(N_BZ):
            k = np.array([i1, i2, i3]) / N_BZ
            AH_tw = build_AH_twisted(k, W)
            M = np.eye(n_bonds) - z * AH_tw + (k_star-1) * z**2 * np.eye(n_bonds)
            sign, logabsdet = np.linalg.slogdet(M)
            if sign.real < 0 or abs(sign) < 0.5:
                # Compute full complex log det
                logdet = np.log(np.linalg.det(M))
            else:
                logdet = logabsdet + np.log(sign)
            acc_logdet += logdet
            n_k += 1

Z_inv = acc_logdet / n_k   # = BZ-avg of log det(I - z A^ω + 2z²I) = log ζ_ω(z)^{-1}
Z_ω = -Z_inv               # = log ζ_ω(z)

print(f"  n_k = {n_k}")
print(f"  BZ-avg log det(I - z A^ω + 2z²I) = {Z_inv.real:.8f} + {Z_inv.imag:.8f}i")
print(f"  log ζ_ω(z) = Z_ω = {Z_ω.real:.8f} + {Z_ω.imag:.8f}i")
print(f"  |Z_ω| = {abs(Z_ω):.8f}")
print(f"  Re(Z_ω) = {Z_ω.real:.8f}")
print(f"  Im(Z_ω) = {Z_ω.imag:.8f}")

# ---------------------------------------------------------------
# Compare with target
# ---------------------------------------------------------------

g_over_e = 10 / np.e
V_us_pdg = 0.22501

print(f"\n=== Target comparisons ===")
print(f"  g/e = {g_over_e:.8f}")
print(f"  V_us PDG = {V_us_pdg}")
print(f"  (2/3)^{{g/e}} = {(2/3)**g_over_e:.8f}  (PDG match: 0.0σ)")

print(f"\n  If |Z_ω| is the 'effective L':  V_us_candidate = (2/3)^|Z_ω| = {(2/3)**abs(Z_ω):.8f}")
print(f"  If Re(Z_ω) is the 'effective L': V_us_candidate = (2/3)^Re(Z_ω) = {(2/3)**Z_ω.real:.8f}")
print(f"  If Im(Z_ω) is the 'effective L': V_us_candidate = (2/3)^Im(Z_ω) = {(2/3)**Z_ω.imag:.8f}")
print(f"  exp(-|Z_ω|) = {np.exp(-abs(Z_ω)):.8f}")
print(f"  exp(Z_ω.real) = {np.exp(Z_ω.real):.8f}")

# Also compare untwiisted zeta for reference
print(f"\n=== Untwisted zeta (holonomy = trivial) for comparison ===")
acc_logdet_0 = complex(0)
for i1 in range(N_BZ):
    for i2 in range(N_BZ):
        for i3 in range(N_BZ):
            k = np.array([i1, i2, i3]) / N_BZ
            AH = build_AH(k)
            M0 = np.eye(n_bonds) - z * AH + (k_star-1) * z**2 * np.eye(n_bonds)
            sign, logabsdet = np.linalg.slogdet(M0)
            if sign.real < 0 or abs(sign) < 0.5:
                logdet = np.log(np.linalg.det(M0))
            else:
                logdet = logabsdet + np.log(sign)
            acc_logdet_0 += logdet

Z_inv_0 = acc_logdet_0 / (N_BZ**3)
Z_0 = -Z_inv_0
print(f"  log ζ_0(z) = Z_0 = {Z_0.real:.8f} + {Z_0.imag:.8f}i")
print(f"  |Z_0| = {abs(Z_0):.8f},  Z_0.real = {Z_0.real:.8f}")

# Ratio
if abs(Z_0) > 1e-10:
    print(f"  |Z_ω| / |Z_0| = {abs(Z_ω)/abs(Z_0):.8f}")
    print(f"  Z_ω.real / Z_0.real = {Z_ω.real/Z_0.real:.8f}")

# Also check against V_cb structure
V_cb_exact = (2/3)**8 / (1 - (2/3)**8)
print(f"\n  V_cb exact = {V_cb_exact:.8f}")
print(f"  (2/3)^8 = {(2/3)**8:.8f}")
print(f"  g/e = 10/e = {g_over_e:.8f}")
print(f"  log(V_us)/log(2/3) = {np.log(V_us_pdg)/np.log(2/3):.8f}  (= g/e ?)")
print(f"  Confirmed: log(V_us)/log(2/3) = {np.log(V_us_pdg)/np.log(2/3):.8f} vs g/e = {g_over_e:.8f}")

# Also check omega^2 twist
print(f"\n=== ω²-twisted zeta for comparison ===")
omega2 = np.exp(4j * np.pi / 3)
omega2_phases = omega2 ** pos_arr
W2 = np.outer(omega2_phases, omega2_phases.conj())
acc_logdet_2 = complex(0)
for i1 in range(N_BZ):
    for i2 in range(N_BZ):
        for i3 in range(N_BZ):
            k = np.array([i1, i2, i3]) / N_BZ
            AH_tw2 = W2 * build_AH(k)
            M2 = np.eye(n_bonds) - z * AH_tw2 + (k_star-1) * z**2 * np.eye(n_bonds)
            sign, logabsdet = np.linalg.slogdet(M2)
            if sign.real < 0 or abs(sign) < 0.5:
                logdet = np.log(np.linalg.det(M2))
            else:
                logdet = logabsdet + np.log(sign)
            acc_logdet_2 += logdet
Z_inv_2 = acc_logdet_2 / (N_BZ**3)
Z_2 = -Z_inv_2
print(f"  log ζ_ω²(z) = Z_2 = {Z_2.real:.8f} + {Z_2.imag:.8f}i")
print(f"  |Z_2| = {abs(Z_2):.8f}")
print(f"  Z_ω.imag + Z_2.imag = {Z_ω.imag + Z_2.imag:.8f}  (should be ~0 by C3 symmetry)")
print(f"  Z_ω.real - Z_2.real = {Z_ω.real - Z_2.real:.8f}  (should be ~0)")
