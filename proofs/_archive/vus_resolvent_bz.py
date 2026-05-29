#!/usr/bin/env python3
"""
proofs/_archive/vus_resolvent_bz.py

Analytical route to V_us: BZ integral of Bloch-Hashimoto resolvent.

The C3 selection rule kills the BZ-averaged G_{us}(z, k) (proven session 17).
Tests:
  (a) G_{us}(z, k) at specific k-points — does it vanish everywhere?
  (b) BZ integral of |G_{us}(z, k)|^2 — always non-negative, bypasses C3 block
  (c) sqrt( integral ) vs V_us = 0.22501

z = (k*-1)/k* = 2/3  (branch measure amplitude)

Species identification:
  u-type bonds: C3-eigenvalue 1, i.e. position 0 within each C3-orbit
  s-type bonds: C3-eigenvalue ω, i.e. position 2 within each C3-orbit
  (4 u-bonds and 4 s-bonds, one per C3-orbit)
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from proofs.common import find_bonds, ATOMS, A_PRIM, C3_PERM, N_ATOMS

# ---------------------------------------------------------------
# Lattice setup
# ---------------------------------------------------------------

bonds_prim = find_bonds()
n_bonds = len(bonds_prim)  # 12
assert n_bonds == 12

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

bond_idx = {}
for oi, (b0, b1, b2) in enumerate(orbits):
    bond_idx[(oi, 0)] = b0   # C3 eigenvalue = 1
    bond_idx[(oi, 1)] = b1   # C3 eigenvalue = ω²
    bond_idx[(oi, 2)] = b2   # C3 eigenvalue = ω

u_idx = [bond_idx[(oi, 0)] for oi in range(4)]   # C3=1
d_idx = [bond_idx[(oi, 1)] for oi in range(4)]   # C3=ω²
s_idx = [bond_idx[(oi, 2)] for oi in range(4)]   # C3=ω

print("=== Bond type indices ===")
for oi in range(4):
    print(f"  Orbit {oi}: u={u_idx[oi]}  d={d_idx[oi]}  s={s_idx[oi]}")

# ---------------------------------------------------------------
# Bloch-Hashimoto matrix
# ---------------------------------------------------------------

def build_AH(k_frac):
    """
    12x12 Bloch-Hashimoto matrix at k_frac (fractional BCC reciprocal coords).

    [A_H(k)]_{j,i} = exp(2πi k·dc_i)  if j is NB successor of i
    where dc_i is the cell displacement of bond i.
    """
    AH = np.zeros((n_bonds, n_bonds), dtype=complex)
    for j, (sj, tj, dcj) in enumerate(bonds_prim):
        for i, (si, ti, dci) in enumerate(bonds_prim):
            if sj != ti:
                continue
            dc_sum = tuple(int(dci[d]) + int(dcj[d]) for d in range(3))
            if tj == si and dc_sum == (0, 0, 0):
                continue  # backtracking
            phase = np.exp(2j * np.pi * np.dot(k_frac, dci))
            AH[j, i] = phase
    return AH

# ---------------------------------------------------------------
# Eigenvalue range check at a few k-points
# ---------------------------------------------------------------

print("\n=== Hashimoto eigenvalue range check ===")
for k_label, k in [("Γ", [0,0,0]), ("P", [0.25,0.25,0.25]), ("H", [0.5,0.5,0.5]),
                    ("N", [0.5,0,0]), ("generic", [0.1,0.2,0.3])]:
    AH = build_AH(k)
    eigs = np.linalg.eigvals(AH)
    print(f"  k={k_label}: |eigenvals| range [{np.min(np.abs(eigs)):.4f}, "
          f"{np.max(np.abs(eigs)):.4f}],  min dist to z=2/3: "
          f"{np.min(np.abs(eigs - 2/3)):.4f}")

# ---------------------------------------------------------------
# C3 check: G_{us} at P-point (should be zero)
# ---------------------------------------------------------------

z = 2/3
print(f"\n=== Resolvent at P-point (k=1/4,1/4,1/4), z={z} ===")
AH_P = build_AH([0.25, 0.25, 0.25])
G_P = np.linalg.solve(z * np.eye(n_bonds) - AH_P, np.eye(n_bonds))
print("  Same-orbit G_{s,u}(k_P):")
for oi in range(4):
    g = G_P[s_idx[oi], u_idx[oi]]
    print(f"    orbit {oi}: G = {g.real:+.6f} + {g.imag:+.6f}i   |G| = {abs(g):.6f}")
print("  Cross-orbit G samples:")
for oi in range(4):
    for oj in range(4):
        if oi == oj: continue
        g = G_P[s_idx[oi], u_idx[oj]]
        if abs(g) > 1e-6:
            print(f"    s-orbit{oi} ← u-orbit{oj}: |G| = {abs(g):.6f}")

# ---------------------------------------------------------------
# BZ integral of |G_{us}(z, k)|^2
# ---------------------------------------------------------------

print(f"\n=== BZ integral of |G_{{us}}(z={z}, k)|^2 ===")

N_BZ = 30   # grid density; 30^3 = 27000 k-points

# Accumulators
acc_same_orbit = 0.0      # Σ_oi |G_{s(oi),u(oi)}|^2
acc_all_us = 0.0          # Σ_{oi,oj} |G_{s(oi),u(oj)}|^2
acc_G_us_re = 0.0         # Re(Σ_oi G_{s(oi),u(oi)}) — check C3 averaging
acc_G_ud_re = 0.0         # Re(Σ_oi G_{d(oi),u(oi)}) — u→d comparison
acc_same_orbit_uu = 0.0   # Σ_oi |G_{u(oi),u(oi)}|^2 — diagonal reference
acc_cb_analog = 0.0       # Σ_oi |G_{d(oi),d(oi+1)}|^2 — same-orbit d(ω²)→s(ω) like V_cb

n_k = 0
n_singular = 0

for i1 in range(N_BZ):
    for i2 in range(N_BZ):
        for i3 in range(N_BZ):
            k = np.array([i1, i2, i3]) / N_BZ
            AH = build_AH(k)
            M = z * np.eye(n_bonds) - AH
            cond = np.linalg.cond(M)
            if cond > 1e10:
                n_singular += 1
                continue
            try:
                G = np.linalg.solve(M, np.eye(n_bonds))
            except np.linalg.LinAlgError:
                n_singular += 1
                continue

            for oi in range(4):
                ui, si, di = u_idx[oi], s_idx[oi], d_idx[oi]
                acc_same_orbit += abs(G[si, ui])**2
                acc_G_us_re += G[si, ui].real
                acc_G_ud_re += G[di, ui].real
                acc_same_orbit_uu += abs(G[ui, ui])**2

            for oi in range(4):
                for oj in range(4):
                    acc_all_us += abs(G[s_idx[oi], u_idx[oj]])**2

            # V_cb analog: same-orbit pos1→pos2 (C3 ω²→ω, same orbit)
            for oi in range(4):
                acc_cb_analog += abs(G[s_idx[oi], d_idx[oi]])**2

            n_k += 1

print(f"  k-points used: {n_k},  skipped (near-singular): {n_singular}")

if n_k == 0:
    print("  ERROR: no valid k-points")
else:
    same = acc_same_orbit / n_k
    allus = acc_all_us / n_k
    uu = acc_same_orbit_uu / n_k
    cb = acc_cb_analog / n_k
    g_us_re = acc_G_us_re / n_k
    g_ud_re = acc_G_ud_re / n_k

    V_us_pdg = 0.22501
    V_cb_pdg = 0.04050

    print(f"\n  Quantity                    | integral    | sqrt(I)     | σ from PDG")
    print(f"  ----------------------------|-------------|-------------|----------")
    print(f"  same-orbit |G_{{su}}|^2 / 4   | {same/4:.8f} | {np.sqrt(same/4):.8f} | "
          f"{(np.sqrt(same/4)-V_us_pdg)/0.00068:+.1f}σ")
    print(f"  same-orbit |G_{{su}}|^2       | {same:.8f} | {np.sqrt(same):.8f} | "
          f"{(np.sqrt(same)-V_us_pdg)/0.00068:+.1f}σ")
    print(f"  all u→s |G|^2 / 16          | {allus/16:.8f} | {np.sqrt(allus/16):.8f} | "
          f"{(np.sqrt(allus/16)-V_us_pdg)/0.00068:+.1f}σ")
    print(f"  all u→s |G|^2               | {allus:.8f} | {np.sqrt(allus):.8f} |")
    print(f"  same-orbit |G_{{uu}}|^2       | {uu:.8f} | {np.sqrt(uu):.8f} | (ref)")
    print(f"  same-orbit |G_{{ds}}|^2 (Vcb) | {cb:.8f} | {np.sqrt(cb):.8f} | "
          f"{(np.sqrt(cb)-V_cb_pdg)/0.00150:+.1f}σ(cb)")
    print(f"  Re(G_us) BZ avg             | {g_us_re:.2e} | (→0 by C3)  |")
    print(f"  Re(G_ud) BZ avg             | {g_ud_re:.2e} | (→0 by C3)  |")

    print(f"\n  PDG references:  V_us={V_us_pdg},  V_cb={V_cb_pdg}")
    print(f"  V_us^2 = {V_us_pdg**2:.8f}")
    print(f"  V_cb^2 = {V_cb_pdg**2:.8f}")
    print(f"  g/e = 10/e = {10/np.e:.6f},  (2/3)^{{g/e}} = {(2/3)**(10/np.e):.8f}")
