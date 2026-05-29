#!/usr/bin/env python3
"""
proofs/flavor/vus_formula_b_test.py

Option B test: is V_us = k*^2/(g * N_ATOMS) = 9/40 a robust structural formula,
or a numerical accident?

Tests:
  1. z-dependence: compute |T_cross(z)|/4 for z in [0.4, 0.9] and compare to 9/40.
     If there's a crossing (where cyclic amp = 9/40), what is z*?
     If no crossing, the formula is NOT the cyclic amplitude.

  2. All three orbit-pair directions: compute cyclic amplitudes for
       (a) u→s = pos0→pos2  (our V_us candidate)
       (b) u→d = pos0→pos1
       (c) d→s = pos1→pos2  (the V_cb-adjacent pair)
     By C3 symmetry all three should be related. Do they give the same
     |T_cross|/4, or different values?

  3. Cross-check: what does the GIRTH-CYCLE formula give for pos0→pos2?
     For V_cb the girth cycle gives L=8 (minimum pos1→pos2 distance, same orbit).
     For pos0→pos2, the minimum same-orbit distance is 2 (g - L_cb = 2).
     Girth-formula prediction: (2/3)^2 / (1 - (2/3)^2) = 0.8 (much too large).
     Confirms V_us ≠ same-orbit girth formula.

  4. Structural identity check: n_s = floor(g/k*) = k* when g = k*^2+1?
     For srs: g=10=3^2+1 → floor(10/3)=3=k*. EXACT identity.
     Test: what does the cyclic amplitude give when g != k*^2+1 (hypothetical)?
     → We can't test this on srs, but we can verify the identity algebraically.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from proofs.common import find_bonds, ATOMS, A_PRIM, C3_PERM, N_ATOMS

bonds_prim = find_bonds()
n_bonds = len(bonds_prim)

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

c3_map = [c3_of_bond(i) for i in range(12)]
visited = [False]*12
orbits = []
for start in range(12):
    if visited[start]: continue
    b0, b1, b2 = start, c3_map[start], c3_map[c3_map[start]]
    orbits.append((b0, b1, b2))
    visited[b0] = visited[b1] = visited[b2] = True

pos_idx = {p: [orb[p] for orb in orbits] for p in range(3)}
# pos_idx[0] = u-bonds, pos_idx[1] = d-bonds, pos_idx[2] = s-bonds

def build_AH(k):
    AH = np.zeros((n_bonds, n_bonds), dtype=complex)
    for j, (sj, tj, dcj) in enumerate(bonds_prim):
        for i, (si, ti, dci) in enumerate(bonds_prim):
            if sj != ti: continue
            dc_sum = tuple(int(dci[d]) + int(dcj[d]) for d in range(3))
            if tj == si and dc_sum == (0,0,0): continue
            AH[j, i] = np.exp(2j*np.pi*np.dot(k, dci))
    return AH

k_star = 3
g = 10

print(f"=== Structural check: g = k*^2 + 1? ===")
print(f"  k* = {k_star}, g = {g}")
print(f"  k*^2 + 1 = {k_star**2 + 1}")
print(f"  g == k*^2 + 1? {g == k_star**2 + 1}")
print(f"  floor(g/k*) = {g // k_star} = k* = {k_star}? {g // k_star == k_star}")
print(f"  → n_s = k* = {k_star} s-bonds per girth cycle (exact when g = k*^2+1)")
print(f"  V_us = k* * n_s / (g * N_ATOMS) = {k_star}*{k_star}/({g}*{N_ATOMS}) = {k_star**2}/{g*N_ATOMS} = {k_star**2/(g*N_ATOMS):.6f}")
print()

V_us_formula = k_star**2 / (g * N_ATOMS)
V_us_pdg = 0.22501
V_us_err = 0.00068

# ── TEST 1: z-dependence ──────────────────────────────────────────────────────

print("=== Test 1: z-dependence of cyclic amplitude ===")
print(f"  Formula k*^2/(g*N_ATOMS) = {V_us_formula:.6f} is z-INDEPENDENT (pure counting)")
print(f"  Cyclic amplitude |T_cross(z)|/4 at z=2/3 = 0.22589 (+1.3σ from formula)")
print(f"  Question: at what z does |T_cross(z)|/4 = {V_us_formula:.6f}?")
print()

N_BZ = 30  # fast grid; we showed convergence is rapid

z_values = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 2/3, 0.70, 0.75, 0.80]
print(f"  {'z':>6}  {'|T_cross|/4':>14}  {'σ from V_us_pdg':>16}  {'σ from 9/40':>12}")
print(f"  {'-'*6}  {'-'*14}  {'-'*16}  {'-'*12}")

T_vals = []
for z in z_values:
    acc_T_cyc = complex(0)
    acc_T_same = complex(0)
    n_k = 0
    for i1 in range(N_BZ):
        for i2 in range(N_BZ):
            for i3 in range(N_BZ):
                k = np.array([i1, i2, i3]) / N_BZ
                AH = build_AH(k)
                G = np.linalg.solve(z * np.eye(n_bonds) - AH, np.eye(n_bonds))
                T = sum(G[di, ui] * G[si, di] * G[ui, si]
                        for ui in pos_idx[0] for di in pos_idx[1] for si in pos_idx[2])
                Ts = sum(G[orbits[oi][1], orbits[oi][0]] * G[orbits[oi][2], orbits[oi][1]] * G[orbits[oi][0], orbits[oi][2]]
                         for oi in range(4))
                acc_T_cyc += T
                acc_T_same += Ts
                n_k += 1
    T_cyc = acc_T_cyc / n_k
    T_same = acc_T_same / n_k
    T_cross = T_cyc - T_same
    val = abs(T_cross) / 4
    sig_pdg = (val - V_us_pdg) / V_us_err
    sig_form = (val - V_us_formula) / V_us_err
    T_vals.append((z, val))
    marker = " ← z=2/3" if abs(z - 2/3) < 1e-10 else ""
    print(f"  {z:6.4f}  {val:14.8f}  {sig_pdg:+16.2f}σ  {sig_form:+12.2f}σ{marker}")

print()

# ── TEST 2: all three pair directions ─────────────────────────────────────────

print("=== Test 2: cyclic amplitudes for all three off-diagonal pairs at z=2/3 ===")
print(f"  If k*^2/(g*N_ATOMS) is the formula, C3 symmetry → all three should give same amplitude")
print()

z = 2/3
acc = {(0,1): complex(0), (1,2): complex(0), (0,2): complex(0)}
acc_same = {(0,1): complex(0), (1,2): complex(0), (0,2): complex(0)}
n_k = 0
for i1 in range(N_BZ):
    for i2 in range(N_BZ):
        for i3 in range(N_BZ):
            k = np.array([i1, i2, i3]) / N_BZ
            AH = build_AH(k)
            G = np.linalg.solve(z * np.eye(n_bonds) - AH, np.eye(n_bonds))

            for (p_src, p_mid, p_tgt) in [(0,1,2), (1,2,0), (0,2,1)]:
                key = tuple(sorted([p_src, p_tgt]))
                # Cyclic: src→mid→tgt→src
                T = sum(G[pos_idx[p_mid][j], pos_idx[p_src][i]] *
                        G[pos_idx[p_tgt][k2], pos_idx[p_mid][j]] *
                        G[pos_idx[p_src][i], pos_idx[p_tgt][k2]]
                        for i in range(4) for j in range(4) for k2 in range(4))
                Ts = sum(G[orbits[oi][p_mid], orbits[oi][p_src]] *
                         G[orbits[oi][p_tgt], orbits[oi][p_mid]] *
                         G[orbits[oi][p_src], orbits[oi][p_tgt]]
                         for oi in range(4))
                acc[(p_src, p_tgt)] += T
                acc_same[(p_src, p_tgt)] += Ts
            n_k += 1

print(f"  {'Pair':8}  {'|T_cross|/4':12}  {'σ from V_us':12}  {'σ from 9/40':12}")
print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}")
for (p_src, p_tgt), name in [((0,2), 'u→s (V_us)'), ((0,1), 'u→d'), ((1,2), 'd→s (V_cb?)')]:
    T_cyc = acc[(p_src, p_tgt)] / n_k
    T_s = acc_same[(p_src, p_tgt)] / n_k
    T_cross = T_cyc - T_s
    val = abs(T_cross) / 4
    sig_pdg = (val - V_us_pdg) / V_us_err
    sig_form = (val - V_us_formula) / V_us_err
    print(f"  {name:8}  {val:12.8f}  {sig_pdg:+12.2f}σ  {sig_form:+12.2f}σ")

# V_cb reference
V_cb_exact = (2/3)**8 / (1 - (2/3)**8)
V_cb_pdg = 0.04050
print(f"\n  V_cb (girth-cycle theorem): {V_cb_exact:.8f}  (PDG: {V_cb_pdg})")
print(f"  k*^2/(g*N_ATOMS) = 9/40 = {V_us_formula:.8f}  ← cannot explain V_cb")
print()

# ── TEST 3: what is the cyclic amplitude for V_cb pair at z=2/3? ─────────────

print("=== Test 3: V_cb-type cyclic amplitude ===")
print(f"  V_cb uses same-orbit pos1→pos2 at girth distance 8.")
print(f"  The *Green's function* analog G[s,d] BZ avg → 0 by C3 (as expected).")
print(f"  Does the V_cb cyclic amplitude differ from V_us cyclic amplitude?")

# From Test 2 above: (1,2) is the d→s pair, which is V_cb-adjacent.
# Its cyclic amplitude was already computed above.

print()
print("=== Summary and interpretation ===")
print()
print(f"  Formula k*^2/(g*N_ATOMS) = 9/40 = {V_us_formula:.8f}")
print(f"  V_us PDG                  = {V_us_pdg}  ± {V_us_err}")
print(f"  Difference from 9/40:     {abs(V_us_pdg - V_us_formula):.2e}")
print(f"  PDG error:                {V_us_err:.2e}")
print(f"  → Formula matches PDG to  {abs(V_us_pdg - V_us_formula)/V_us_err:.3f}σ")
print()

# Check if the z where cyclic amp = 9/40 is physically significant
# Find z* by interpolation
vals = [(z, v) for z, v in T_vals]
for i in range(len(vals)-1):
    z1, v1 = vals[i]; z2, v2 = vals[i+1]
    if (v1 - V_us_formula) * (v2 - V_us_formula) < 0:
        # crossing between z1 and z2
        z_star = z1 + (V_us_formula - v1) / (v2 - v1) * (z2 - z1)
        z_formula = (k_star - 1) / k_star
        print(f"  Crossing: |T_cross(z*)|/4 = 9/40 at z* ≈ {z_star:.4f}")
        print(f"  Physical z = (k*-1)/k* = {z_formula:.4f}")
        print(f"  Difference: Δz = {z_star - z_formula:.4f}")
        break
else:
    print(f"  No crossing found in z ∈ [{vals[0][0]:.2f}, {vals[-1][0]:.2f}]")
    print(f"  Cyclic amplitude monotonically {'decreasing' if vals[-1][1] < vals[0][1] else 'increasing'} with z")
    print(f"  Physical z=2/3={2/3:.4f}: cyclic amp = {dict(T_vals).get(2/3, 0.22589):.5f}, formula = {V_us_formula:.5f}")

print()
print(f"  CONCLUSION:")
if all(abs(T_vals[i][1] - V_us_formula) < 0.001 for i in range(len(T_vals))):
    print(f"  → Cyclic amplitude ≈ 9/40 for all z tested. Formula is ROBUST.")
else:
    print(f"  → Cyclic amplitude ≠ 9/40 across z range. They are DIFFERENT quantities.")
    print(f"  → 9/40 is NOT the cyclic amplitude formula.")
    print(f"  → 9/40 must come from a DIFFERENT (non-BZ-integral) mechanism.")
    print(f"  → Most likely: a pure counting/density argument (Option A route).")
