#!/usr/bin/env python3
"""
proofs/_archive/vus_cyclic_amplitude.py

Hypothesis: V_us arises from the BZ-averaged cyclic three-step amplitude.

MOTIVATION
----------
All five previous routes are blocked:
  1. BFS integer-L       — no integer L gives 0.225
  2. BZ-avg G_us         — C3 selection rule: exact 0
  3. BZ-avg |G_us|^2     — gives ~0.39 (hundreds of σ off)
  4. Ihara zeta C3-twist — similarity theorem: Z_ω = Z_0
  5. Stark-Terras quotient— b_pure=0.509, gives 0.812

The C3 selection rule kills BZ-avg G_AB whenever A and B carry
different C3 eigenvalues. But products of three elements can have
TRIVIAL C3 charge and survive the BZ average:

  C3 charges: u=1, d=ω², s=ω
  G_du: charge ω²/1 = ω²
  G_sd: charge ω/ω² = ω^{-1} = ω²   (in Z_3, ω^{-1} = ω²)
  G_us: charge ω/1 = ω (wait: going from u to s, the C3 charge is
        the eigenvalue of the TARGET divided by that of SOURCE = ω/1=ω)

Wait, let me recount. Under C3:
  G_AB → (α_B/α_A)·G_AB where α_X = C3 eigenvalue of species X.
  u: α=1, d: α=ω², s: α=ω.

  G_du: α_u/α_d = 1/ω² = ω (goes FROM d TO u? No, G_du means row d, col u.)
        Actually G[d,u] under C3: C3(G[d,u]) = (α_d/α_u)·G[d,u] = ω²·G[d,u].
  G_sd: C3(G[s,d]) = (α_s/α_d)·G[s,d] = (ω/ω²)·G[s,d] = ω^{-1}·G[s,d] = ω²·G[s,d].
  G_us: C3(G[u,s]) = (α_u/α_s)·G[u,s] = (1/ω)·G[u,s] = ω²·G[u,s].

Product: C3(G_du × G_sd × G_us) = ω² × ω² × ω² = ω^6 = 1 (trivial!).

So T_cyc = BZ-avg Σ_{d,s,u bonds} G[d,u](k) × G[s,d](k) × G[u,s](k)
has trivial C3 charge → BZ average is NOT killed by the selection rule.

This is a fundamentally new quantity. Geometrically: it counts amplitude
for u→(NB walk)→d→(NB walk)→s→(NB walk)→u closed three-step cycle
via the Hashimoto propagator. A "generation cycle amplitude."

TARGET: Check if T_cyc (or some function of it) equals V_us = 0.22501.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from proofs.common import find_bonds, ATOMS, A_PRIM, C3_PERM, N_ATOMS

# ── lattice + orbit setup ────────────────────────────────────────────────────

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

# species: position 0 = u (C3=1), position 1 = d (C3=ω²), position 2 = s (C3=ω)
u_idx = [orb[0] for orb in orbits]   # 4 u-bonds (C3 eigenvalue 1)
d_idx = [orb[1] for orb in orbits]   # 4 d-bonds (C3 eigenvalue ω²)
s_idx = [orb[2] for orb in orbits]   # 4 s-bonds (C3 eigenvalue ω)

print("=== Bond species assignment ===")
for oi, orb in enumerate(orbits):
    print(f"  Orbit {oi}: u={orb[0]}, d={orb[1]}, s={orb[2]}")

# ── Bloch-Hashimoto matrix ────────────────────────────────────────────────────

def build_AH(k_frac):
    AH = np.zeros((n_bonds, n_bonds), dtype=complex)
    for j, (sj, tj, dcj) in enumerate(bonds_prim):
        for i, (si, ti, dci) in enumerate(bonds_prim):
            if sj != ti:
                continue
            dc_sum = tuple(int(dci[d]) + int(dcj[d]) for d in range(3))
            if tj == si and dc_sum == (0, 0, 0):
                continue  # no backtracking
            phase = np.exp(2j * np.pi * np.dot(k_frac, dci))
            AH[j, i] = phase
    return AH

# ── C3 charge verification ────────────────────────────────────────────────────

omega = np.exp(2j * np.pi / 3)

print(f"\n=== C3 charge analysis ===")
print(f"Species: u=C3(1), d=C3(ω²), s=C3(ω)")
print(f"G[d,u]: C3 charge = α_d/α_u = ω²/1 = ω²")
print(f"G[s,d]: C3 charge = α_s/α_d = ω/ω² = ω^{{-1}} = ω²")
print(f"G[u,s]: C3 charge = α_u/α_s = 1/ω = ω²")
print(f"Product G[d,u]·G[s,d]·G[u,s]: C3 charge = ω²·ω²·ω² = ω^6 = 1 ✓ (trivial)")
print(f"→ BZ average of this product is NOT killed by C3 selection rule")

# ── BZ integral ───────────────────────────────────────────────────────────────

z = 2/3  # branch measure amplitude
k_star = 3

print(f"\n=== BZ integral at z={z} ===")
print(f"Computing: T_cyc = BZ-avg Σ_{{ui,dj,sk}} G[dj,ui](k) × G[sk,dj](k) × G[ui,sk](k)")

N_BZ = 30
n_k = 0

# Accumulators
acc_T_cyc = complex(0)      # main cyclic amplitude Tr[G_du G_sd G_us] (full sum)
acc_T_cyc_same = complex(0) # same-orbit only (oi=oj=ok)
acc_T_cyc_diag = complex(0) # diagonal: same oi for all three

# Also compute: G[u,s] alone BZ avg (should be 0), |G[u,s]|^2 (should be ~0.39)
acc_Gus = complex(0)
acc_Gus_sq = 0.0
acc_Gds = complex(0)         # G[d,s] = G between d and s bonds

# New: also check the "reverse" cyclic product G[u,d] G[d,s] G[s,u]
acc_T_rev = complex(0)       # G[u,d]·G[d,s]·G[s,u]: C3 charge = (ω²^{-1})·... check:
                              # G[u,d]: α_u/α_d = 1/ω² = ω
                              # G[d,s]: α_d/α_s = ω²/ω = ω
                              # G[s,u]: α_s/α_u = ω/1 = ω
                              # Product: ω·ω·ω = ω³ = 1 ✓ (also trivial!)

for i1 in range(N_BZ):
    for i2 in range(N_BZ):
        for i3 in range(N_BZ):
            k = np.array([i1, i2, i3]) / N_BZ
            AH = build_AH(k)
            G = np.linalg.solve(z * np.eye(n_bonds) - AH, np.eye(n_bonds))

            # Full cyclic sum: Σ_{u,d,s bonds} G[d,u] × G[s,d] × G[u,s]
            T = complex(0)
            T_rev = complex(0)
            T_same = complex(0)
            for ui in u_idx:
                for di in d_idx:
                    for si in s_idx:
                        T += G[di, ui] * G[si, di] * G[ui, si]
                        T_rev += G[ui, di] * G[di, si] * G[si, ui]
            # Same-orbit contribution (oi=oj=ok)
            for oi in range(4):
                ui, di, si = u_idx[oi], d_idx[oi], s_idx[oi]
                T_same += G[di, ui] * G[si, di] * G[ui, si]

            acc_T_cyc += T
            acc_T_rev += T_rev
            acc_T_cyc_same += T_same

            # Reference quantities
            for oi in range(4):
                ui, si = u_idx[oi], s_idx[oi]
                acc_Gus += G[si, ui]
                acc_Gus_sq += abs(G[si, ui])**2
            for oi in range(4):
                di, si = d_idx[oi], s_idx[oi]
                acc_Gds += G[si, di]

            n_k += 1

n_total = n_k

T_cyc   = acc_T_cyc / n_total
T_rev   = acc_T_rev / n_total
T_same  = acc_T_cyc_same / n_total
G_us_avg = acc_Gus / n_total
G_us_sq_avg = acc_Gus_sq / n_total
G_ds_avg = acc_Gds / n_total

print(f"\n  k-points: {n_total}")
print(f"\n  T_cyc (full, Σ all u,d,s):  {T_cyc.real:.8f} + {T_cyc.imag:.8f}i  (|T|={abs(T_cyc):.8f})")
print(f"  T_rev (reverse cycle):       {T_rev.real:.8f} + {T_rev.imag:.8f}i  (|T|={abs(T_rev):.8f})")
print(f"  T_same (same-orbit only):    {T_same.real:.8f} + {T_same.imag:.8f}i  (|T|={abs(T_same):.8f})")
print(f"  G_us BZ avg (should be 0):   {G_us_avg.real:.2e}")
print(f"  |G_us|^2 BZ avg / 4:         {G_us_sq_avg/4:.8f}")

V_us_pdg = 0.22501
V_us_err = 0.00068

print(f"\n=== Comparison to V_us = {V_us_pdg} ± {V_us_err} ===")

def check(val, label):
    if abs(val) < 1e-15:
        print(f"  {label}: value = 0 (trivially zero)")
        return
    sigma = (val - V_us_pdg) / V_us_err
    print(f"  {label}: {val:.8f}  ({sigma:+.1f}σ)")

# Various functions of T_cyc:
for name, val in [
    ("Re(T_cyc)",         T_cyc.real),
    ("|T_cyc|",           abs(T_cyc)),
    ("Re(T_cyc)/4",       T_cyc.real/4),
    ("|T_cyc|/4",         abs(T_cyc)/4),
    ("Re(T_cyc)/16",      T_cyc.real/16),
    ("|T_cyc|/16",        abs(T_cyc)/16),
    ("Re(T_same)",        T_same.real),
    ("|T_same|",          abs(T_same)),
    ("Re(T_rev)",         T_rev.real),
    ("Re(T_rev)/4",       T_rev.real/4),
    ("|T_rev|/4",         abs(T_rev)/4),
]:
    check(val, name)

# Power/root combinations
print(f"\n=== Power/root candidates ===")
import numpy as np
for name, val in [("Re(T_cyc)", T_cyc.real), ("|T_cyc|", abs(T_cyc)),
                  ("Re(T_same)", T_same.real), ("|T_same|", abs(T_same))]:
    for exp_name, exp in [("^(1/2)", 0.5), ("^(1/3)", 1/3), ("^(1/4)", 0.25), ("^2", 2)]:
        if val > 0:
            v = val ** exp
            sigma = (v - V_us_pdg) / V_us_err
            if abs(sigma) < 30:
                print(f"  ({name}){exp_name} = {v:.8f}  ({sigma:+.1f}σ) ← CLOSE!")

# Compare with g/e structure
g_over_e = 10 / np.e
print(f"\n=== g/e cross-check ===")
print(f"  g/e = {g_over_e:.8f}")
print(f"  V_us = (2/3)^{{g/e}} = {(2/3)**g_over_e:.8f}")
print(f"  PDG  = {V_us_pdg}")

# Check: does the cyclic amplitude encode g/e?
if abs(T_cyc.real) > 1e-10:
    L_eff = np.log(T_cyc.real) / np.log(2/3)
    print(f"  (2/3)^L = Re(T_cyc) → L = {L_eff:.6f}  (g/e = {g_over_e:.6f}?)")
if abs(T_cyc) > 1e-10:
    L_eff2 = np.log(abs(T_cyc)) / np.log(2/3)
    print(f"  (2/3)^L = |T_cyc|   → L = {L_eff2:.6f}")
if abs(T_same.real) > 1e-10:
    L_eff3 = np.log(T_same.real) / np.log(2/3)
    print(f"  (2/3)^L = Re(T_same)→ L = {L_eff3:.6f}")

# Also compare V_cb
V_cb_exact = (2/3)**8 / (1-(2/3)**8)
print(f"\n  V_cb = {V_cb_exact:.8f},  V_us = {V_us_pdg}")
print(f"  V_us/V_cb = {V_us_pdg/V_cb_exact:.4f}  (Wolfenstein 1/Aλ ≈ {1/(0.836*0.225):.4f})")
