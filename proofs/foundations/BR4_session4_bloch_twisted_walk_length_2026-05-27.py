#!/usr/bin/env python3
"""
proofs/foundations/BR4_session4_bloch_twisted_walk_length_2026-05-27.py

BR4 Session 4 — Bloch-twisted walk-length structure at the P-point.

Uses framework's own srs primitive-cell connectivity (`proofs/common.py`)
to build B(k_P) at k_P = (1/4, 1/4, 1/4) (reduced BCC primitive BZ).

PURPOSE
-------
Per `theorem_bloch_lift_mu.md` (theorem-grade), B(k_P) has eigenvalues
±h, ±h* (mult 2 each) + ±1 (mult 2 each), with h = (√3 + i√5)/2.

Per `theorem_multiway_branch_measure.md` §11.4 "the bridge IS μ", the
W-vertex matrix element ⟨gen j | W | gen i⟩ on C³_obs is the μ-moment
of the branch class of NB walks connecting gen i to gen j. The OPEN
question (per §11.3) is: which substrate walk class corresponds to
which (i, j) pair?

Naive structural hypothesis: the minimum L for which ⟨e_j | B(k_P)^L | e_i⟩
is non-zero (between C₃-orbit pairs (orbit_i, orbit_j) on the directed-edge
fibre) gives the L_{i→j} parameter.

If L_min(orbit_i, orbit_j) varies non-trivially over orbit pairs (and
matches the expected L_cb=8, L_ub=14 structure), the BR4 intertwiner
is structurally derivable from B(k_P)^L matrix elements.

Run with:
    python3 proofs/foundations/BR4_session4_bloch_twisted_walk_length_2026-05-27.py
"""

import sys
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
from proofs.common import find_bonds, bloch_H, K_STAR, ATOMS, A_PRIM, NN_DIST

TOL = 1e-9


# ---------------------------------------------------------------------------
# 1. Framework's srs bonds + sanity check A(k_P)
# ---------------------------------------------------------------------------

bonds = find_bonds()
n_verts = 4
n_edges = len(bonds)

print(f"  Framework srs primitive cell: {n_verts} vertices, {n_edges} directed bonds")
assert n_edges == 12, f"Expected 12 directed bonds, got {n_edges}"

K_P = np.array([0.25, 0.25, 0.25])
A_P = bloch_H(K_P, bonds)

# A(k_P) should be Hermitian and have eigenvalues ±√3 (mult 2 each)
A_P_eig = np.linalg.eigvalsh(A_P)
print(f"  A(k_P=(1/4,1/4,1/4)) eigenvalues: {[f'{e:+.4f}' for e in sorted(A_P_eig)]}")
assert np.allclose(sorted(np.abs(A_P_eig)), [np.sqrt(K_STAR)] * n_verts, atol=1e-6), \
    "A(k_P) doesn't match framework theorem ±√3 multiplicity 2"
print(f"  ✓ Matches framework theorem (predictions/srs_E_at_P.py)")
print()


# ---------------------------------------------------------------------------
# 2. Build B(k_P) — Hashimoto matrix on directed edges
# ---------------------------------------------------------------------------

def reverse_edge(e):
    s, t, c = e
    return (t, s, (-c[0], -c[1], -c[2]))


def edges_equal(e1, e2):
    s1, t1, c1 = e1
    s2, t2, c2 = e2
    return s1 == s2 and t1 == t2 and c1 == c2


def edge_idx(e, edges):
    for i, ei in enumerate(edges):
        if edges_equal(e, ei):
            return i
    return None


# Find reverse-edge index (some bonds list both directions; some only one)
rev_idx = [edge_idx(reverse_edge(e), bonds) for e in bonds]
n_pairs = sum(1 for r in rev_idx if r is not None)
print(f"  Reverse-edge pairs found in bond list: {n_pairs}/{n_edges}")


def build_B(k, edges):
    """Build B(k)[e', e]: NB transition matrix at Bloch fiber k.
    B[e', e] = (Bloch phase) if head(e) = tail(e') and e' ≠ rev(e), else 0.
    Convention: Bloch phase of OUTGOING edge e' (so B(k) acts on edge-amplitudes
    with phase exp(2πi·k·cell_out))."""
    n = len(edges)
    B = np.zeros((n, n), dtype=complex)
    for i, ei in enumerate(edges):
        si, ti, ci = ei
        for j, ej in enumerate(edges):
            sj, tj, cj = ej
            if sj != ti:
                continue
            if rev_idx[i] is not None and j == rev_idx[i]:
                continue
            phase = np.exp(2j * np.pi * np.dot(k, cj))
            B[j, i] = phase
    return B


B_P = build_B(K_P, bonds)

# Eigenvalues of B(k_P) — should match theorem ±h, ±h*, ±1
eigs_B = np.linalg.eigvals(B_P)
print()
print(f"  B(k_P) eigenvalues (theorem: ±h, ±h*, ±1; |h|² = 2):")
sorted_eigs = sorted(eigs_B, key=lambda z: (abs(z), z.real, z.imag))
mags = [abs(e) for e in sorted_eigs]

h_target = (np.sqrt(3) + 1j * np.sqrt(5)) / 2
mag_counts = {}
for m in mags:
    rounded = round(m, 4)
    mag_counts[rounded] = mag_counts.get(rounded, 0) + 1

print(f"    Magnitude distribution: {mag_counts}")
for k, e in enumerate(sorted_eigs):
    cls = ""
    if abs(abs(e) - 1) < 1e-6:
        cls = " (trivial ±1)"
    elif abs(abs(e) - np.sqrt(2)) < 1e-6:
        cls = " (Ramanujan ±h, ±h*)"
    print(f"    λ_{k+1}: {e:+.6f}  |λ| = {abs(e):.4f}  arg = {np.degrees(np.angle(e)):+.2f}°{cls}")


# Verify theorem prediction: 4 trivial (|·|=1) + 8 Ramanujan (|·|=√2)
ramanujan_count = sum(1 for m in mags if abs(m - np.sqrt(2)) < 1e-6)
trivial_count = sum(1 for m in mags if abs(m - 1.0) < 1e-6)
print(f"  Ramanujan (|·|=√2): {ramanujan_count} (theorem: 8)")
print(f"  Trivial (|·|=1):    {trivial_count} (theorem: 4)")


# ---------------------------------------------------------------------------
# 3. Identify C_3-equivariant directed-edge orbits
# ---------------------------------------------------------------------------
# Per S1 R-C: substrate C_3 σ = (v_0)(v_1 v_3 v_2) on vertex labels.
# Lifted to directed edges, σ acts as:
#   src → C_3(src), tgt → C_3(tgt), cell-displacement → C_3-rotated cell

def c3_vertex(v):
    """Vertex C_3 per S1 R-C: σ = (v_0)(v_1 v_3 v_2). v_1→v_3→v_2→v_1."""
    return {0: 0, 1: 3, 3: 2, 2: 1}[v]


def c3_edge(e, edges):
    """Apply σ to directed edge. C_3 rotates cell vector (n1,n2,n3) → (n3,n1,n2)
    for the body-diagonal R_(1,1,1) rotation, but in BCC primitive basis the
    cell rotation is different. We compute the C_3-rotated bond using the
    vertex permutation alone and find the matching bond in the list."""
    s, t, c = e
    new_s, new_t = c3_vertex(s), c3_vertex(t)
    # Cyclic permutation of cell components (the BCC primitive vectors
    # also cycle under the body-diagonal C_3)
    new_c = (c[2], c[0], c[1])
    # Find this edge in bonds (cell may shift by lattice translation
    # — accept any (new_s, new_t, c') with the right type)
    for k, ek in enumerate(edges):
        sk, tk, ck = ek
        if sk == new_s and tk == new_t:
            return ek
    return None


# Build orbits
visited = [False] * n_edges
orbits = []
for i in range(n_edges):
    if visited[i]:
        continue
    orbit = []
    cur = bonds[i]
    for _ in range(4):
        idx = edge_idx(cur, bonds)
        if idx is None or visited[idx]:
            break
        orbit.append(idx)
        visited[idx] = True
        nxt = c3_edge(cur, bonds)
        if nxt is None:
            break
        cur = nxt
    orbits.append(orbit)


print()
print(f"  C_3 orbits on directed edges (σ = (v_0)(v_1 v_3 v_2)):")
for k, orb in enumerate(orbits):
    print(f"    Orbit {k} (size {len(orb)}): edges {orb}")
    for idx in orb:
        e = bonds[idx]
        print(f"      edge {idx}: v{e[0]} → v{e[1]} + cell {e[2]}")


# ---------------------------------------------------------------------------
# 4. Compute B(P)^L for various L
# ---------------------------------------------------------------------------

print()
print("  Step 4 — Matrix-element analysis vs walk-length L")
print("  (looking for L_min where C_3-equivariant orbit-pair amplitude > threshold)")
print()

B_powers = [np.eye(n_edges, dtype=complex)]
for L in range(1, 21):
    B_powers.append(B_powers[-1] @ B_P)


def orbit_to_orbit_amp(L, orbit_i, orbit_j):
    """Sum of complex amplitudes ⟨e_j | B^L | e_i⟩ over orbit pairs.
    Returns total signed sum (for phase) and total |·|² for amplitude."""
    total_sum = 0j
    total_sq = 0.0
    for ei in orbit_i:
        for ej in orbit_j:
            entry = B_powers[L][ej, ei]
            total_sum += entry
            total_sq += abs(entry) ** 2
    return total_sum, total_sq


THRESHOLD = 1e-6

print(f"  L_min table (smallest L where Σ |amp|² > {THRESHOLD}):")
print()
header = "    " + "orbit_i \\ orbit_j " + " ".join(f"{j:>8}" for j in range(len(orbits)))
print(header)
for i, oi in enumerate(orbits):
    row = f"    orbit {i:>5}            "
    for j, oj in enumerate(orbits):
        L_min = None
        for L in range(1, 21):
            _, sq = orbit_to_orbit_amp(L, oi, oj)
            if sq > THRESHOLD:
                L_min = L
                break
        row += f"{L_min if L_min is not None else '∞':>8}"
    print(row)
print()


print(f"  Amplitudes at framework's L_cb=8 and L_ub=14:")
print()
print(f"    pair (i,j)       |amp(L=8)|²       arg(sum L=8)°      |amp(L=14)|²      arg(sum L=14)°")
for i in range(len(orbits)):
    for j in range(len(orbits)):
        if i == j:
            continue
        sum8, sq8 = orbit_to_orbit_amp(8, orbits[i], orbits[j])
        sum14, sq14 = orbit_to_orbit_amp(14, orbits[i], orbits[j])
        arg8 = np.degrees(np.angle(sum8)) if abs(sum8) > 1e-9 else 0
        arg14 = np.degrees(np.angle(sum14)) if abs(sum14) > 1e-9 else 0
        print(f"    ({i},{j})         {sq8:.6e}    {arg8:>+8.2f}      {sq14:.6e}    {arg14:>+8.2f}")


# ---------------------------------------------------------------------------
# 5. Specific orbit-pair phase analysis (test for arg(h_P) emergence)
# ---------------------------------------------------------------------------

arg_h = np.degrees(np.arctan(np.sqrt(5/3)))   # = 52.24°

print()
print(f"  Step 5 — Phase analysis (looking for arg(h_P) = arctan(√(5/3)) ≈ {arg_h:.2f}°)")
print()
print(f"    Walker phase at each L: arg(h^L) = L · {arg_h:.2f}° mod 360°")
for L in range(1, 17):
    phase = (L * arg_h) % 360
    flag = ""
    if abs(phase - 12.73) < 1:
        flag = " ← matches δ_lepton = 12.73°"
    if abs(phase - 70.53) < 1:
        flag = " ← matches φ_K4 = arccos(1/3) = 70.53°"
    print(f"    L={L:>2}: {phase:>+7.2f}°{flag}")


# ---------------------------------------------------------------------------
# 6. Test trace formula: Tr(B^L) per theorem L4
# ---------------------------------------------------------------------------

print()
print("  Step 6 — Tr(B(P)^L) per theorem_bloch_lift_mu.md (C2):")
print()
print("    Theorem: Tr(B(P)^L) = 4 + 8·2^(L/2)·cos(L·arg(h)) for L even, else 0")
print()
print(f"    {'L':>3}  {'Tr(B^L) computed':>20}  {'Tr theorem-predicted':>22}")
for L in range(0, 17):
    tr = np.trace(B_powers[L])
    if L == 0:
        predicted = n_edges
    elif L % 2 == 0:
        predicted = 4 + 8 * 2**(L/2) * np.cos(L * np.arctan(np.sqrt(5/3)))
    else:
        predicted = 0
    print(f"    {L:>3}  {tr.real:>+10.4f} + {tr.imag:>+8.4e}j  {predicted:>+12.4f}")


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
print()
print("=" * 76)
print("VERDICT SKELETON")
print("=" * 76)
print()
print("  Inspect L_min table:")
print("    - If L_min varies non-trivially across orbit pairs → STRUCTURAL")
print("      candidate for the (i,j) ↔ L map.")
print("    - If L_min ≈ 8 for ΔGen=1 pairs and ≈ 14 for ΔGen=2 pairs →")
print("      BR4 closure via this route receives strong structural support.")
print("    - If L_min ≈ 1 or 2 for ALL orbit pairs (early connectivity) →")
print("      the C_3-equivariant 'orbit amplitude' is the wrong observable.")
print("      Would need a more refined orbit-pair-projector to extract L_cb=8.")
