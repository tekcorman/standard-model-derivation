#!/usr/bin/env python3
"""
proofs/_archive/hashimoto_kpoint_resolvent.py

Compute the Hashimoto resolvent G(z=2/3, k) at individual high-symmetry
k-points and extract (s_bond, u_bond) matrix elements.

KEY INSIGHT: The BZ-averaged G_us = 0 by C3 selection rule. But the
selection rule acts on the AVERAGE, not on individual k-points. At the
N-point (the only high-symmetry point NOT invariant under C3), the C3
symmetry doesn't pin G_us to zero. This is the natural place to look.

If V_us = G[s,u](z=2/3, k=N) or a simple function thereof, the route is:
  - N is the unique non-C3-invariant high-symmetry point
  - At N the C3-sectors mix
  - The off-diagonal (s,u) resolvent element at N encodes the mixing amplitude

Also compute: C3-decomposed Level 2 crystal overlaps between k-points,
resolving the degeneracy at P properly.
"""

import sys, os
import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from proofs.common import find_bonds, ATOMS, A_PRIM, C3_PERM, N_ATOMS, bloch_H, c3_decompose

# ── Setup ─────────────────────────────────────────────────────────────────────
bonds_prim = find_bonds()
n_bonds = len(bonds_prim)  # 12

omega = np.exp(2j * np.pi / 3)
z = 2/3    # branch measure

V_us_pdg = 0.22501; V_us_err = 0.00068
V_cb_exact = (2/3)**8 / (1-(2/3)**8)
V_us_form = 9/40

# Build C3 orbit structure on bonds
C3_CART = np.array([[0,0,1],[1,0,0],[0,1,0]], dtype=float)
c3_atom = {i: int(np.argmax(C3_PERM[:, i])) for i in range(N_ATOMS)}

def bond_disp(src, tgt, cell):
    return (np.array(ATOMS[tgt]) + cell[0]*np.array(A_PRIM[0])
            + cell[1]*np.array(A_PRIM[1]) + cell[2]*np.array(A_PRIM[2])
            - np.array(ATOMS[src]))

prim_disps = [bond_disp(src, tgt, cell) for src, tgt, cell in bonds_prim]

def c3_of_bond(i):
    src, _, _ = bonds_prim[i]
    new_src = c3_atom[src]
    rotated = C3_CART @ prim_disps[i]
    for j, (s, t, c) in enumerate(bonds_prim):
        if s == new_src and np.allclose(prim_disps[j], rotated, atol=1e-8):
            return j
    raise ValueError(f"bond {i}")

c3_map = [c3_of_bond(i) for i in range(n_bonds)]
visited = [False]*n_bonds
orbits = []
for start in range(n_bonds):
    if visited[start]: continue
    b0, b1, b2 = start, c3_map[start], c3_map[c3_map[start]]
    orbits.append((b0, b1, b2))
    visited[b0] = visited[b1] = visited[b2] = True

u_idx = [orb[0] for orb in orbits]   # position 0
d_idx = [orb[1] for orb in orbits]   # position 1
s_idx = [orb[2] for orb in orbits]   # position 2

print(f"Bond orbits (u=pos0, d=pos1, s=pos2):")
for oi, orb in enumerate(orbits):
    print(f"  Orbit {oi}: u={orb[0]}, d={orb[1]}, s={orb[2]}")

# Hashimoto builder
def build_AH(k):
    AH = np.zeros((n_bonds, n_bonds), dtype=complex)
    for j, (sj, tj, dcj) in enumerate(bonds_prim):
        for i, (si, ti, dci) in enumerate(bonds_prim):
            if sj != ti: continue
            dc_sum = tuple(int(dci[d]) + int(dcj[d]) for d in range(3))
            if tj == si and dc_sum == (0,0,0): continue
            AH[j, i] = np.exp(2j*np.pi*np.dot(k, dci))
    return AH

# High-symmetry k-points
k_points = {
    'Gamma': np.array([0, 0, 0]),
    'H':     np.array([0.5, 0.5, -0.5]),
    'N':     np.array([0, 0, 0.5]),
    'P':     np.array([0.25, 0.25, 0.25]),
}
# The three N-orbit points (C3 images of N)
N_orbit = [np.array([0, 0, 0.5]), np.array([0.5, 0, 0]), np.array([0, 0.5, 0])]

# ── G(z, k) at each high-symmetry k-point ────────────────────────────────────
print(f"\n=== Resolvent G(z={z}, k) — Level 3 Hashimoto ===")
print(f"    Extracting G[s_bond, u_bond] for all (s, u) pairs")
print(f"    Targets: V_us={V_us_pdg}  9/40={V_us_form}  V_cb={V_cb_exact:.5f}")

def resolvent_elements(k, label):
    AH = build_AH(k)
    G = np.linalg.solve(z * np.eye(n_bonds) - AH, np.eye(n_bonds))
    # G[s, u] for each (s-bond, u-bond) pair
    su_same  = [G[s_idx[i], u_idx[i]] for i in range(4)]  # same orbit
    su_cross = [G[s_idx[i], u_idx[j]] for i in range(4) for j in range(4) if i != j]
    sd_same  = [G[s_idx[i], d_idx[i]] for i in range(4)]  # same orbit s-d (→ V_cb)
    print(f"\n  k={label}:")
    print(f"    G[s,u] same-orbit:  " + "  ".join(f"{v.real:+.5f}{v.imag:+.5f}i" for v in su_same))
    print(f"    G[s,u] cross-orbit: " + "  ".join(f"{abs(v):.4f}" for v in su_cross))
    print(f"    G[s,d] same-orbit:  " + "  ".join(f"{abs(v):.5f}" for v in sd_same))
    # Check for matches
    all_su = su_same + su_cross
    for v in all_su:
        for val, name in [(abs(v.real), 'Re(G)'), (abs(v.imag), 'Im(G)'), (abs(v), '|G|')]:
            for tgt, tname in [(V_us_pdg, 'V_us'), (V_us_form, '9/40'), (V_cb_exact, 'V_cb')]:
                sigma = abs(val - tgt) / V_us_err
                if sigma < 5:
                    print(f"    *** {name}[s,u] = {val:.6f}  ({tname}={tgt:.5f}, {sigma:.2f}σ)")
    return G

G_all = {}
for name, k in k_points.items():
    G_all[name] = resolvent_elements(k, name)

# ── N-point in detail: full (s,u) G matrix ───────────────────────────────────
print(f"\n=== Full G[s_i, u_j] matrix at k=N (4×4) ===")
G_N = G_all['N']
print(f"  (rows = s-bond index 0-3, cols = u-bond index 0-3)")
print(f"  Absolute values:")
for i in range(4):
    row = "  ".join(f"{abs(G_N[s_idx[i], u_idx[j]]):.6f}" for j in range(4))
    print(f"    s[{i}]: {row}")
print(f"  Real parts:")
for i in range(4):
    row = "  ".join(f"{G_N[s_idx[i], u_idx[j]].real:+.6f}" for j in range(4))
    print(f"    s[{i}]: {row}")

# Sum over all s,u pairs (like cyclic amplitude decomposition)
G_su_sum_N = sum(G_N[s_idx[i], u_idx[j]] for i in range(4) for j in range(4))
G_su_same_N = sum(G_N[s_idx[i], u_idx[i]] for i in range(4))
G_su_cross_N = G_su_sum_N - G_su_same_N
print(f"\n  Sum over all G[s,u] at N: {G_su_sum_N.real:+.6f}{G_su_sum_N.imag:+.6f}i  |sum|={abs(G_su_sum_N):.6f}")
print(f"  Same-orbit sum: {G_su_same_N.real:+.6f}{G_su_same_N.imag:+.6f}i")
print(f"  Cross-orbit sum: {G_su_cross_N.real:+.6f}{G_su_cross_N.imag:+.6f}i  |cross|={abs(G_su_cross_N):.6f}")
print(f"  |cross|/4 = {abs(G_su_cross_N)/4:.6f}  (target V_us = {V_us_pdg})")

# ── Compare to V_cb: G[s,d] same-orbit at P (the known working case) ──────────
print(f"\n=== Sanity: G[s,d] same-orbit at P — should reproduce V_cb ===")
G_P = G_all['P']
G_sd_P = [G_P[s_idx[i], d_idx[i]] for i in range(4)]
G_sd_mean = np.mean([abs(v) for v in G_sd_P])
print(f"  G[s,d] same-orbit at P: " + "  ".join(f"{abs(v):.5f}" for v in G_sd_P))
print(f"  Mean |G[s,d]| = {G_sd_mean:.6f}")
print(f"  V_cb exact = {V_cb_exact:.6f}")
print(f"  Mean |G[s,d]| / V_cb = {G_sd_mean/V_cb_exact:.4f}")

# ── C3-averaged G over N-orbit ────────────────────────────────────────────────
print(f"\n=== C3-averaged G[s,u] over the full N-orbit: N, C3(N), C3²(N) ===")
print(f"  (Averaging over the 3 C3-images of N kills C3-odd components)")
G_avg = np.zeros((n_bonds, n_bonds), dtype=complex)
for k in N_orbit:
    AH = build_AH(k)
    G = np.linalg.solve(z * np.eye(n_bonds) - AH, np.eye(n_bonds))
    G_avg += G
G_avg /= 3

G_su_avg = np.array([[G_avg[s_idx[i], u_idx[j]] for j in range(4)] for i in range(4)])
print(f"  |G[s,u]| C3-averaged over N-orbit:")
for i in range(4):
    row = "  ".join(f"{abs(G_su_avg[i,j]):.6f}" for j in range(4))
    print(f"    s[{i}]: {row}")
su_cross_avg = sum(G_su_avg[i,j] for i in range(4) for j in range(4) if i != j)
su_same_avg = sum(G_su_avg[i,i] for i in range(4))
print(f"\n  Cross-orbit sum (C3-averaged N): {su_cross_avg.real:+.6f}{su_cross_avg.imag:+.6f}i  |/4|={abs(su_cross_avg)/4:.6f}")
print(f"  Same-orbit sum  (C3-averaged N): {su_same_avg.real:+.6f}{su_same_avg.imag:+.6f}i")
print(f"  Note: C3 averaging SHOULD kill G_us if C3 selection rule applies everywhere.")
print(f"  Non-zero cross-orbit sum means N-orbit breaks the selection rule differently.")

# ── Level 2: C3-decomposed overlaps between k-points ─────────────────────────
print(f"\n=== Level 2: C3-resolved eigenstates and overlaps ===")
print(f"  Using c3_decompose to get proper C3 quantum numbers at each k-point")

c2_data = {}
for name, k in k_points.items():
    evals, evecs, c3_eigs, offdiag = c3_decompose(k, bonds_prim)
    c2_data[name] = (evals, evecs, c3_eigs)
    print(f"\n  k={name}: (eval, C3 label)")
    for i in range(N_ATOMS):
        c3_lab = 'C3=1 ' if abs(c3_eigs[i]-1)<0.2 else ('C3=ω ' if abs(c3_eigs[i]-omega)<0.2 else 'C3=ω²')
        print(f"    [{i}] E={evals[i]:+.5f}  C3={c3_eigs[i].real:+.4f}{c3_eigs[i].imag:+.4f}i ({c3_lab})")

# Overlaps in the C3-eigenstate basis
print(f"\n  C3-resolved overlaps |<P_i|other_j>| (P eigenstates fixed reference):")
_, vP, c3P = c2_data['P']
for other_name in ['Gamma', 'H', 'N']:
    _, vO, c3O = c2_data[other_name]
    ov = np.abs(vP.conj().T @ vO)
    print(f"\n  P ↔ {other_name}:")
    for i in range(N_ATOMS):
        c3_P = 'C3=1 ' if abs(c3P[i]-1)<0.2 else ('C3=ω ' if abs(c3P[i]-omega)<0.2 else 'C3=ω²')
        row = "  ".join(f"{ov[i,j]:.5f}" for j in range(N_ATOMS))
        print(f"    P[{i}]({c3_P}): {row}")
        for j in range(N_ATOMS):
            v = ov[i,j]
            for tgt, tname in [(V_us_pdg,'V_us'),(V_us_form,'9/40'),(V_cb_exact,'V_cb')]:
                sigma = abs(v - tgt) / V_us_err
                if sigma < 5 and (i!=j or other_name=='Gamma'):
                    c3_O = 'C3=1 ' if abs(c3O[j]-1)<0.2 else ('C3=ω ' if abs(c3O[j]-omega)<0.2 else 'C3=ω²')
                    print(f"    *** |<P[{i}]({c3_P})|{other_name}[{j}]({c3_O})>| = {v:.6f}  ({tname}, {sigma:.2f}σ)")

# ── Structural identities ──────────────────────────────────────────────────────
print(f"\n=== Structural identities ===")
k_star = 3; g = 10; N_at = 4
print(f"  k* = {k_star},  g = {g},  N_ATOMS = {N_at}")
print(f"  Crystal eigenvalue at P:  ±√k* = ±{np.sqrt(k_star):.6f}")
print(f"  Crystal eigenvalue at N:  ±√(k*+2) = ±{np.sqrt(k_star+2):.6f}")
print(f"  Crystal eigenvalue at Γ:  k* = {k_star}")
print(f"  Hashimoto phases at P:    ±arctan(√5/√3) = ±{np.degrees(np.arctan(np.sqrt(5)/np.sqrt(3))):.3f}°")
print(f"  Hashimoto phases at N:    ±arctan(√3/√5) = ±{np.degrees(np.arctan(np.sqrt(3)/np.sqrt(5))):.3f}°")
print(f"  Phase diff (P-N):         {np.degrees(np.arctan(np.sqrt(5)/np.sqrt(3)) - np.arctan(np.sqrt(3)/np.sqrt(5))):.3f}°")
print(f"  sin(arctan(√3/√5)) = √3/√8 = {np.sqrt(3)/np.sqrt(8):.6f}")
print(f"  cos(arctan(√3/√5)) = √5/√8 = {np.sqrt(5)/np.sqrt(8):.6f}")
print(f"  Re(h_N × conj(h_P)) / |h|² = Re((√15-i)/4) = {np.sqrt(15)/4:.6f}")
print(f"  Im(h_N × conj(h_P)) / |h|² = Im((√15-i)/4) = {-1/4:.6f}")
print()
print(f"  --- Candidate formulae for V_us ---")
print(f"  9/40                        = {9/40:.8f}")
print(f"  1/√(N_ATOMS × k*)           = {1/np.sqrt(N_at*k_star):.8f}")
print(f"  1/(k* × N_ATOMS/k*)         = {1/(k_star * N_at/k_star):.8f}")
print(f"  (k*-1) / (g-k*+1)           = {(k_star-1)/(g-k_star+1):.8f}")
print(f"  k* / (k*² + N_ATOMS + g/k*) = {k_star/(k_star**2 + N_at + g/k_star):.8f}")
print(f"  V_us PDG                    = {V_us_pdg:.8f}")
