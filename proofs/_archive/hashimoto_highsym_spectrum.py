#!/usr/bin/env python3
"""
proofs/_archive/hashimoto_highsym_spectrum.py

Enumerate Hashimoto (Level 3) and crystal (Level 2) eigenvalues at all four
BCC high-symmetry k-points and look for CKM/PMNS structure in ratios and
eigenvector overlaps.

BCC BZ high-symmetry k-points (fractional coords of primitive reciprocal):
  Gamma = (0,   0,    0   )  -- zone centre, full C3
  H     = (1/2, 1/2, -1/2)  -- zone face, C3-invariant (mod lattice)
  N     = (0,   0,    1/2)  -- zone edge, C3-orbit of 3 equivalent points
  P     = (1/4, 1/4,  1/4)  -- zone vertex, C3-fixed, Ramanujan-saturating

C3 action on k (fractional): (k1,k2,k3) -> (k3,k1,k2).
  Gamma: fixed.  H: fixed (mod integer lattice).  P: fixed.
  N: (0,0,1/2)->(1/2,0,0)->(0,1/2,0)->(0,0,1/2)  [3-orbit].

The hypothesis (session-20 handoff, Priority 2): CKM matrix elements are
ratios of Hashimoto eigenvalues and/or overlaps of eigenvectors at different
k-points. This is the "rotation matrix between C3 families" route.
"""

import sys, os
import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from proofs.common import find_bonds, ATOMS, A_PRIM, C3_PERM, N_ATOMS, bloch_H

# ── PDG targets ──────────────────────────────────────────────────────────────
V_us_pdg = 0.22501; V_us_err = 0.00068
V_cb_pdg = 0.04050; V_cb_err = 0.00150
V_ub_pdg = 0.00369; V_ub_err = 0.00011
theta12_deg = 33.44; theta12_rad = np.radians(theta12_deg)
theta13_deg =  8.57; theta13_rad = np.radians(theta13_deg)
theta23_deg = 49.20; theta23_rad = np.radians(theta23_deg)

V_us_form = 9/40    # algebraic candidate k*^2/(g*N_ATOMS)
V_cb_exact = (2/3)**8 / (1-(2/3)**8)

print(f"Targets:")
print(f"  V_us = {V_us_pdg}  (9/40 = {V_us_form})")
print(f"  V_cb = {V_cb_pdg}  (theorem = {V_cb_exact:.5f})")
print(f"  V_ub = {V_ub_pdg}")
print(f"  theta12 = {theta12_deg}° = {theta12_rad:.5f} rad")
print(f"  theta13 = {theta13_deg}° = {theta13_rad:.5f} rad")

# ── Lattice setup ─────────────────────────────────────────────────────────────
bonds_prim = find_bonds()
n_bonds = len(bonds_prim)  # 12

omega = np.exp(2j * np.pi / 3)

# C3 action on k (fractional): (k1,k2,k3) -> (k3,k1,k2)
def c3_k(k): return np.array([k[2], k[0], k[1]])

# High-symmetry k-points (fractional coords)
k_points = {
    'Gamma': np.array([0,    0,    0   ]),
    'H':     np.array([0.5,  0.5, -0.5 ]),
    'N':     np.array([0,    0,    0.5 ]),
    'P':     np.array([0.25, 0.25, 0.25]),
}
# N has a 3-orbit under C3
N_orbit = [np.array([0, 0, 0.5]), np.array([0.5, 0, 0]), np.array([0, 0.5, 0])]

print(f"\n=== C3 invariance check ===")
for name, k in k_points.items():
    ck = c3_k(k)
    diff = np.mod(ck - k + 0.5, 1.0) - 0.5   # closest integer shift
    is_inv = np.allclose(diff, 0, atol=1e-9) or np.allclose(diff, np.round(diff), atol=1e-9)
    print(f"  {name}: k={k}, C3(k)={np.round(ck,4)}, diff={np.round(ck-k,4)}, C3-invariant: {is_inv}")

# ── Level 2: crystal Bloch Hamiltonian H(k) ──────────────────────────────────
print(f"\n=== Level 2: Crystal Bloch Hamiltonian eigenvalues (4×4) ===")
L2_eigs = {}
L2_vecs = {}
for name, k in k_points.items():
    H = bloch_H(k, bonds_prim)
    evals, evecs = la.eig(H)
    idx = np.argsort(np.real(evals))[::-1]
    evals = evals[idx]; evecs = evecs[:, idx]
    L2_eigs[name] = evals
    L2_vecs[name] = evecs
    print(f"\n  k={name} {k}:")
    for i, e in enumerate(evals):
        print(f"    eig[{i}] = {e.real:+.6f} + {e.imag:+.6f}i   |e|={abs(e):.6f}  arg={np.degrees(np.angle(e)):+.2f}°")

# ── Level 3: Hashimoto non-backtracking matrix A_H(k) ────────────────────────
def build_AH(k):
    AH = np.zeros((n_bonds, n_bonds), dtype=complex)
    for j, (sj, tj, dcj) in enumerate(bonds_prim):
        for i, (si, ti, dci) in enumerate(bonds_prim):
            if sj != ti: continue
            dc_sum = tuple(int(dci[d]) + int(dcj[d]) for d in range(3))
            if tj == si and dc_sum == (0,0,0): continue   # no backtrack
            AH[j, i] = np.exp(2j*np.pi*np.dot(k, dci))
    return AH

print(f"\n=== Level 3: Hashimoto eigenvalues (12×12) ===")
L3_eigs = {}
L3_vecs = {}
for name, k in k_points.items():
    AH = build_AH(k)
    evals, evecs = la.eig(AH)
    idx = np.argsort(np.abs(evals))[::-1]
    evals = evals[idx]; evecs = evecs[:, idx]
    L3_eigs[name] = evals
    L3_vecs[name] = evecs
    print(f"\n  k={name} {k}:")
    for i, e in enumerate(evals):
        marker = ""
        if abs(abs(e)**2 - 2.0) < 0.01: marker = " ← Ramanujan |h|²=2"
        elif abs(abs(e)**2 - 1.0) < 0.01: marker = " ← |h|²=1"
        print(f"    h[{i:2d}] = {e.real:+.6f} + {e.imag:+.6f}i   |h|={abs(e):.6f}  |h|²={abs(e)**2:.6f}{marker}")

# ── C3 decomposition of Hashimoto at P and H (C3-invariant points) ───────────
print(f"\n=== C3 decomposition of Hashimoto eigenstates ===")
print(f"  (Only well-defined at C3-invariant k-points: Gamma, H, P)")

# Build C3 action on bonds (12×12 permutation matrix)
C3_CART = np.array([[0,0,1],[1,0,0],[0,1,0]], dtype=float)
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

c3_map = [c3_of_bond(i) for i in range(n_bonds)]
C3_bond = np.zeros((n_bonds, n_bonds), dtype=complex)
for i in range(n_bonds):
    C3_bond[c3_map[i], i] = 1.0

for name in ['Gamma', 'H', 'P']:
    print(f"\n  k={name}:")
    evecs = L3_vecs[name]
    for i in range(min(6, n_bonds)):
        v = evecs[:, i]
        c3v = C3_bond @ v
        # C3 eigenvalue = <v|C3|v> / <v|v>
        c3_eig = (np.conj(v) @ c3v) / (np.conj(v) @ v)
        label = '1 ' if abs(c3_eig - 1) < 0.2 else ('ω ' if abs(c3_eig - omega) < 0.2 else 'ω²')
        e = L3_eigs[name][i]
        print(f"    h[{i}] = {abs(e):.5f} ∠{np.degrees(np.angle(e)):+.1f}°  C3≈{c3_eig.real:+.3f}{c3_eig.imag:+.3f}i ({label})")

# ── Ratio table: all pairs of eigenvalue magnitudes across k-points ───────────
print(f"\n=== Ratio table: |h(k_a)|/|h(k_b)| for top eigenvalues ===")
print(f"  Targets: V_us={V_us_pdg}, V_cb={V_cb_pdg:.5f}, V_ub={V_ub_pdg}")
print(f"  9/40={V_us_form}, sin(θ12)={np.sin(theta12_rad):.5f}, sin(θ13)={np.sin(theta13_rad):.5f}")

tgt_vals = [V_us_pdg, V_us_form, V_cb_pdg, V_cb_exact, V_ub_pdg,
            np.sin(theta12_rad), np.sin(theta13_rad),
            1/3, 1/4, 1/np.sqrt(3), np.sqrt(2)-1]

# Collect all distinct |h| values with their k-label
all_h = []
for name in ['Gamma', 'H', 'N', 'P']:
    for i, e in enumerate(L3_eigs[name][:6]):
        all_h.append((name, i, abs(e)))

print(f"\n  All |h| values (top 6 per k-point):")
for nm, i, mag in all_h:
    print(f"    {nm}[{i}]: |h|={mag:.8f}  |h|²={mag**2:.8f}")

print(f"\n  Close ratios (within 3σ of any target):")
hits = []
for i, (nm_a, ia, ha) in enumerate(all_h):
    for j, (nm_b, ib, hb) in enumerate(all_h):
        if i == j: continue
        if hb < 1e-9: continue
        ratio = ha / hb
        for tgt in tgt_vals:
            err_pdg = V_us_err if abs(tgt - V_us_pdg) < 0.01 else V_cb_err if abs(tgt - V_cb_pdg) < 0.002 else V_us_err
            sigma = abs(ratio - tgt) / err_pdg
            if sigma < 5:
                hits.append((sigma, ratio, tgt, nm_a, ia, ha, nm_b, ib, hb))

hits.sort()
seen = set()
for sigma, ratio, tgt, nm_a, ia, ha, nm_b, ib, hb in hits[:30]:
    key = (round(ratio, 6), round(tgt, 6))
    if key in seen: continue
    seen.add(key)
    print(f"    {nm_a}[{ia}]/{nm_b}[{ib}] = {ha:.6f}/{hb:.6f} = {ratio:.6f}  (target={tgt:.5f}, {sigma:.2f}σ)")

# ── Eigenvector overlaps P ↔ N ────────────────────────────────────────────────
print(f"\n=== Eigenvector overlaps: <P_i | N_j> (the 'rotation matrix') ===")
print(f"  P-eigenstates have good C3 quantum numbers; N-eigenstates mix C3.")
print(f"  Off-diagonal overlaps encode cross-C3-sector mixing = CKM elements.")

vP = L3_vecs['P']
vN = L3_vecs['N']

print(f"\n  |<P_i|N_j>| overlap matrix (rows=P, cols=N, top 6x6):")
overlap = np.abs(vP[:, :6].conj().T @ vN[:, :6])
# header
print("  " + "  ".join(f"  N[{j}]  " for j in range(6)))
for i in range(6):
    row = "  ".join(f"{overlap[i,j]:.5f}" for j in range(6))
    e_P = L3_eigs['P'][i]
    print(f"  P[{i}] |h|={abs(e_P):.4f}: {row}")

# ── Phase differences: arg(h_P) - arg(h_N) etc. ──────────────────────────────
print(f"\n=== Phase differences between k-points ===")
print(f"  sin(theta) candidates from arg(h) differences:")
for nm_a in ['P', 'H', 'N']:
    for nm_b in ['Gamma', 'H', 'N', 'P']:
        if nm_a == nm_b: continue
        for ia in range(4):
            for ib in range(4):
                ha = L3_eigs[nm_a][ia]
                hb = L3_eigs[nm_b][ib]
                if abs(ha) < 1e-9 or abs(hb) < 1e-9: continue
                phase_diff = abs(np.angle(ha) - np.angle(hb))
                phase_diff = min(phase_diff, np.pi - phase_diff)   # fold
                s = abs(np.sin(phase_diff))
                for tgt in [V_us_pdg, V_us_form, V_cb_pdg]:
                    err = V_us_err
                    sigma = abs(s - tgt) / err
                    if sigma < 5:
                        print(f"    sin(arg({nm_a}[{ia}])-arg({nm_b}[{ib}])) = {s:.6f}  (target={tgt:.5f}, {sigma:.2f}σ)")

# ── Level 2 cross-k-point overlaps ───────────────────────────────────────────
print(f"\n=== Level 2 crystal: eigenvector overlaps P ↔ N ↔ H ↔ Gamma ===")
for nm_a, nm_b in [('P','N'), ('P','H'), ('P','Gamma'), ('H','N'), ('N','Gamma')]:
    vA = L2_vecs[nm_a]; vB = L2_vecs[nm_b]
    ov = np.abs(vA.conj().T @ vB)
    e_A = L2_eigs[nm_a]; e_B = L2_eigs[nm_b]
    print(f"\n  |<{nm_a}_i|{nm_b}_j>| (|h_A|, |h_B|):")
    for i in range(N_ATOMS):
        row = "  ".join(f"{ov[i,j]:.4f}" for j in range(N_ATOMS))
        print(f"    {nm_a}[{i}] |e|={abs(e_A[i]):.4f}: {row}")
    # look for V_us in off-diagonal
    for i in range(N_ATOMS):
        for j in range(N_ATOMS):
            if i == j: continue
            v = ov[i,j]
            for tgt in [V_us_pdg, V_us_form, V_cb_pdg]:
                sigma = abs(v - tgt) / V_us_err
                if sigma < 5:
                    print(f"    *** |<{nm_a}[{i}]|{nm_b}[{j}]>| = {v:.6f}  (target={tgt:.5f}, {sigma:.2f}σ)")

# ── Squared overlaps (|V|² = probability) ────────────────────────────────────
print(f"\n=== Squared overlaps |<P_i|N_j>|² at Level 3 ===")
ov2 = overlap**2
print(f"  (Should sum to 1 by column/row if eigenvectors are orthonormal)")
for i in range(6):
    row = "  ".join(f"{ov2[i,j]:.5f}" for j in range(6))
    print(f"  P[{i}]: {row}  sum={ov2[i,:].sum():.4f}")

# ── N-point: C3-averaged combination ─────────────────────────────────────────
print(f"\n=== C3-averaged N-point eigenvalues ===")
print(f"  N, C3(N), C3²(N): {[list(np.round(k,3)) for k in N_orbit]}")
eigs_N_avg = []
for k in N_orbit:
    AH = build_AH(k)
    evals = la.eigvals(AH)
    evals = np.sort(np.abs(evals))[::-1]
    eigs_N_avg.append(evals)
eigs_N_avg = np.array(eigs_N_avg)
mean_N = eigs_N_avg.mean(axis=0)
print(f"  Mean |h| over N-orbit (top 6): {mean_N[:6]}")
print(f"  Spread (std):                  {eigs_N_avg.std(axis=0)[:6]}")

# ── Summary: closest matches ──────────────────────────────────────────────────
print(f"\n=== SUMMARY: best matches to CKM targets ===")
print(f"  9/40 = {V_us_form}")
print(f"  V_us PDG = {V_us_pdg} ± {V_us_err}")
print(f"  V_cb exact = {V_cb_exact:.8f}")
print()

# Print top hits
for sigma, ratio, tgt, nm_a, ia, ha, nm_b, ib, hb in hits[:10]:
    print(f"  {nm_a}[{ia}]/{nm_b}[{ib}] = {ratio:.6f}  target={tgt:.5f}  {sigma:.2f}σ")
