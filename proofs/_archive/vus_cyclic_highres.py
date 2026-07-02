#!/usr/bin/env python3
"""
proofs/_archive/vus_cyclic_highres.py

High-resolution verification of the cross-orbit cyclic amplitude formula.

From vus_cyclic_amplitude.py (N_BZ=30):
  T_cyc   = -0.92949481 (sum over ALL 64 u,d,s orbit triples)
  T_same  = -0.02593491 (sum over 4 SAME-orbit triples)
  T_cross = T_cyc - T_same = -0.90355990
  |T_cross| / 4 = 0.22589  (+1.3σ from V_us = 0.22501)

Key question: does this hold up at higher BZ resolution?
Also: what normalization gives the best match?
Also: does the analogous formula give V_cb?
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from proofs.common import find_bonds, ATOMS, A_PRIM, C3_PERM, N_ATOMS

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
orbits = []
for start in range(12):
    if visited[start]:
        continue
    b0, b1, b2 = start, c3_map[start], c3_map[c3_map[start]]
    assert c3_map[b2] == b0 and len({b0, b1, b2}) == 3
    orbits.append((b0, b1, b2))
    visited[b0] = visited[b1] = visited[b2] = True
assert len(orbits) == 4

u_idx = [orb[0] for orb in orbits]   # position 0
d_idx = [orb[1] for orb in orbits]   # position 1
s_idx = [orb[2] for orb in orbits]   # position 2

def build_AH(k_frac):
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

z = 2/3
V_us_pdg = 0.22501
V_us_err = 0.00068
V_cb_pdg = 0.04050
V_cb_err = 0.00150

print(f"Targets: V_us = {V_us_pdg} ± {V_us_err},  V_cb = {V_cb_pdg} ± {V_cb_err}")
print(f"z = {z}")

for N_BZ in [30, 40, 50]:
    acc_T_cyc = complex(0)
    acc_T_same = complex(0)

    # V_cb analog: same-orbit d->s Green's function (position 1 -> position 2)
    # V_cb uses same-orbit b1->b2; the Green's function analog:
    # G[s_oi, d_oi](z) BZ-averaged for same orbit
    acc_G_ds_same = complex(0)
    acc_G_ds_same_sq = 0.0  # |G[s,d]|^2 analog

    n_k = 0

    for i1 in range(N_BZ):
        for i2 in range(N_BZ):
            for i3 in range(N_BZ):
                k = np.array([i1, i2, i3]) / N_BZ
                AH = build_AH(k)
                G = np.linalg.solve(z * np.eye(n_bonds) - AH, np.eye(n_bonds))

                T = complex(0)
                T_same = complex(0)
                for ui in u_idx:
                    for di in d_idx:
                        for si in s_idx:
                            T += G[di, ui] * G[si, di] * G[ui, si]

                for oi in range(4):
                    ui, di, si = u_idx[oi], d_idx[oi], s_idx[oi]
                    T_same += G[di, ui] * G[si, di] * G[ui, si]
                    # V_cb analog:
                    acc_G_ds_same += G[si, di]
                    acc_G_ds_same_sq += abs(G[si, di])**2

                acc_T_cyc += T
                acc_T_same += T_same
                n_k += 1

    T_cyc   = acc_T_cyc / n_k
    T_same  = acc_T_same / n_k
    T_cross = T_cyc - T_same
    G_ds    = acc_G_ds_same / n_k
    G_ds_sq = acc_G_ds_same_sq / n_k

    V_candidate = abs(T_cross) / 4
    sigma_us = (V_candidate - V_us_pdg) / V_us_err

    print(f"\nN_BZ = {N_BZ} ({n_k} k-points):")
    print(f"  T_cyc   = {T_cyc.real:.10f} + {T_cyc.imag:.2e}i")
    print(f"  T_same  = {T_same.real:.10f} + {T_same.imag:.2e}i")
    print(f"  T_cross = T_cyc - T_same = {T_cross.real:.10f}")
    print(f"  |T_cross| / 4 = {V_candidate:.10f}  ({sigma_us:+.2f}σ from V_us)")
    print(f"  V_us PDG     = {V_us_pdg}")
    print(f"  (2/3)^{{g/e}} = {(2/3)**(10/np.e):.10f}")

    # V_cb analog: what does G[s,d] same-orbit give?
    sigma_cb_direct = (G_ds.real - V_cb_pdg) / V_cb_err
    sigma_cb_sq = (np.sqrt(G_ds_sq/4) - V_cb_pdg) / V_cb_err
    print(f"\n  V_cb analog (G[s,d] same-orbit BZ avg):")
    print(f"    Re(G[s,d]) BZ avg / 4 = {G_ds.real/4:.10f}  ({sigma_cb_direct:.1f}σ from V_cb)")
    print(f"    sqrt(|G[s,d]|^2 / 4)  = {np.sqrt(G_ds_sq/4):.10f}  ({sigma_cb_sq:.1f}σ from V_cb)")
    V_cb_exact = (2/3)**8 / (1-(2/3)**8)
    print(f"    V_cb exact (girth)     = {V_cb_exact:.10f}")

# Also: check if the formula has a clean closed form
print(f"\n=== Limiting analysis ===")
print(f"  g/e = 10/e = {10/np.e:.10f}")
print(f"  V_us = (2/3)^{{10/e}} = {(2/3)**(10/np.e):.10f}")
print(f"  |T_cross|/4 converges to V_us as N_BZ → ∞?")
print(f"  Difference at N_BZ=50 will show convergence rate")
