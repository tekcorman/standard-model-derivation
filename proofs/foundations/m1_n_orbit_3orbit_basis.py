#!/usr/bin/env python3
"""
proofs/foundations/m1_n_orbit_3orbit_basis.py

PURPOSE
-------
M1 (Bloch eigenmode route) first probe for the substrate-side mass-eigenstate
identification gap — see an internal working note.

CONCEPTUAL REFRAME
------------------
Per `proofs/foundations/n_orbit_c3_multiplicities.py`, the combined C_3 action on
V_Ram(N1) ⊕ V_Ram(N2) ⊕ V_Ram(N3) has uniform isotypic multiplicities (8, 8, 8),
which the existing script flags as "BLOCKED" (does not match P-point's (4, 2, 2)
color shape).

This script reads (8, 8, 8) the OTHER way: it is the necessary signature of an
8-fold structure of disjoint **3-orbits** of the combined C_3, where each
3-orbit (|ψ⟩, C_36|ψ⟩, C_36²|ψ⟩) has support permuted across the three
fibers V(N1) → V(N2) → V(N3) and decomposes as 1·trivial ⊕ 1·ω ⊕ 1·ω̄ in C_3-character.

This is the Z_3 **cyclic basis** structure (not the isotypic decomposition) needed
for a substrate-side image of observer C^3_obs's generation basis (R3).

WHAT THIS PROBE VERIFIES
------------------------
1. The combined operator C_36 acting on V(N1) ⊕ V(N2) ⊕ V(N3) is order 3:
   C_36^3 = I exactly.
2. C_36 commutes with the block-diagonal Hashimoto B_total = B(N1) ⊕ B(N2) ⊕ B(N3).
3. V_Ram(N1) is 8-dimensional with all |eig|² = 2 (Ramanujan saturated).
4. For each of the 8 V_Ram(N1) eigenmodes |ψ⟩, the orbit triplet
   G = (|ψ⟩ in N1-slot, C_36|ψ⟩ in N2-slot, C_36²|ψ⟩ in N3-slot)
   satisfies (a) cyclic closure C_36³|ψ⟩ = |ψ⟩, (b) clean fiber-permuting support
   (no cross-fiber leakage), and (c) Z_3 character on the orbit = {1, ω, ω̄}.

These four checks confirm the conceptual reframe: the (8, 8, 8) result is the
signature of 8 disjoint Z_3-cyclic 3-orbits, any one of which is a candidate
for the substrate image of mass eigenstates {|gen-1⟩, |gen-2⟩, |gen-3⟩}.

Down-stream M1 step (NOT in this probe): for each of the 8 candidate G_i,
compute the position-space walker matrix elements between gen states at natural
walk lengths (L=8 for V_cb, L=14+ for V_ub) using `vcb_hashimoto_bfs.py`'s
supercell apparatus, and check whether any candidate reproduces |V_cb|² and
|V_ub|². See an internal working note §4.

GATE STATUS
-----------
CAS verification only. Confirms structural reframe is computationally well-defined.
NOT a closure of M1 — the matrix-element check is the load-bearing next step.
"""

import sys
import os
import numpy as np
from numpy import linalg as la

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from proofs.common import omega3, find_bonds
from proofs.foundations.theorem_B5_3_core import (
    bloch_hashimoto, build_c3_on_directed_edges, build_directed_edges,
)


N1 = np.array([0.0, 0.0, 0.5])
N2 = np.array([0.5, 0.0, 0.0])
N3 = np.array([0.0, 0.5, 0.0])

RAMANUJAN_MOD_SQ = 2.0  # k* - 1 for k* = 3


def _build_combined_operators():
    bonds = find_bonds()
    directed = build_directed_edges(bonds)
    assert len(directed) == 12

    B_N1 = bloch_hashimoto(N1, directed)
    B_N2 = bloch_hashimoto(N2, directed)
    B_N3 = bloch_hashimoto(N3, directed)
    U_C3 = build_c3_on_directed_edges(directed)
    n = 12

    Z = np.zeros((n, n), complex)
    B_total = np.block([[B_N1, Z, Z], [Z, B_N2, Z], [Z, Z, B_N3]])
    C_36   = np.block([[Z, Z, U_C3], [U_C3, Z, Z], [Z, U_C3, Z]])

    return B_N1, B_total, C_36, n


def _extract_vram(B_k, n=12, tol=1e-6):
    eigs, V = la.eig(B_k)
    ram_idx = [i for i in range(n) if abs(abs(eigs[i]) ** 2 - RAMANUJAN_MOD_SQ) < tol]
    W = V[:, ram_idx]
    W, _ = la.qr(W)
    W = W[:, : len(ram_idx)]
    return eigs[ram_idx], W


def main():
    print("=" * 72)
    print("M1 first probe — N-orbit 3-orbit cyclic basis verification")
    print("=" * 72)

    B_N1, B_total, C_36, n = _build_combined_operators()

    # -------- check 1: C_36 has order 3 --------
    err_c3 = la.norm(C_36 @ C_36 @ C_36 - np.eye(3 * n))
    assert err_c3 < 1e-10, f"C_36 not order-3: ||C_36^3 - I|| = {err_c3}"
    print(f"  [1] ||C_36^3 - I||_F = {err_c3:.2e}    (expect 0)  OK")

    # -------- check 2: C_36 commutes with B_total --------
    err_comm = la.norm(B_total @ C_36 - C_36 @ B_total)
    assert err_comm < 1e-10, f"[B_total, C_36] != 0: norm = {err_comm}"
    print(f"  [2] ||[B_total, C_36]||_F = {err_comm:.2e}    (expect 0)  OK")

    # -------- check 3: V_Ram(N1) is 8-dim, |eig|^2 = 2 --------
    eigs_ram_N1, W1 = _extract_vram(B_N1, n=n)
    assert W1.shape[1] == 8, f"V_Ram(N1) dim {W1.shape[1]} != 8"
    print(f"  [3] V_Ram(N1) dim = {W1.shape[1]}   "
          f"|eig|^2 in [{min(abs(eigs_ram_N1)**2):.4f}, "
          f"{max(abs(eigs_ram_N1)**2):.4f}]  OK")

    # -------- check 4: each of 8 modes generates a clean 3-orbit --------
    n_clean = 0
    n_modes = W1.shape[1]
    closure_max = 0.0
    cross_leak_max = 0.0
    for j in range(n_modes):
        psi = np.zeros(3 * n, complex)
        psi[:n] = W1[:, j]
        psi /= la.norm(psi)

        g0 = psi
        g1 = C_36 @ g0
        g2 = C_36 @ g1
        closure = la.norm(C_36 @ g2 - g0)
        closure_max = max(closure_max, closure)

        # support: g0 in N1, g1 in N2, g2 in N3 (no cross-fiber leakage)
        leak0 = la.norm(g0[n:])
        leak1 = la.norm(g1[:n]) + la.norm(g1[2 * n:])
        leak2 = la.norm(g2[:2 * n])
        leak = max(leak0, leak1, leak2)
        cross_leak_max = max(cross_leak_max, leak)

        if closure < 1e-8 and leak < 1e-8:
            n_clean += 1

    assert n_clean == 8, f"only {n_clean}/8 clean 3-orbits"
    print(f"  [4] Clean 3-orbits: {n_clean}/{n_modes}    "
          f"max closure err = {closure_max:.2e}, max cross-fiber leak = {cross_leak_max:.2e}  OK")

    # -------- check 5: Z_3 character on each orbit = {1, omega, omega_bar} --------
    n_correct_chars = 0
    for j in range(n_modes):
        psi = np.zeros(3 * n, complex)
        psi[:n] = W1[:, j] / la.norm(W1[:, j])
        G = np.column_stack([psi, C_36 @ psi, C_36 @ C_36 @ psi])
        # G is exactly orthonormal because the three vectors have disjoint
        # fiber support (verified in check 4).
        C_restr = G.conj().T @ C_36 @ G  # 3x3
        eigs_restr = sorted(la.eigvals(C_restr), key=lambda z: np.angle(z))

        # Must be {omega_bar, 1, omega} sorted by angle
        targets = sorted([1.0 + 0j, omega3, omega3 ** 2], key=lambda z: np.angle(z))
        diff = sum(abs(a - b) for a, b in zip(eigs_restr, targets))
        if diff < 1e-8:
            n_correct_chars += 1

    assert n_correct_chars == 8, f"{n_correct_chars}/8 orbits have correct Z_3 character"
    print(f"  [5] Z_3-character on each orbit = {{1, omega, omega_bar}}: "
          f"{n_correct_chars}/{n_modes}  OK")

    # -------- summary --------
    print()
    print("-" * 72)
    print("RESULT: 8 disjoint Z_3-cyclic 3-orbits in V_Ram(N1) ⊕ V_Ram(N2) ⊕ V_Ram(N3).")
    print()
    print("Each orbit is a candidate substrate image of observer C^3_obs's")
    print("generation basis {|gen-1>, |gen-2>, |gen-3>}.")
    print()
    print("NOT YET DERIVED: which (if any) of the 8 candidates is THE generation")
    print("basis. Selection rule comes from matching position-space walker matrix")
    print("elements between gen states to |V_cb|^2 = (256/6305)^2 and |V_ub|^2 ~")
    print("1.42e-5 — the load-bearing next probe (cf. m1_m2 entry doc §4).")
    print()
    print("RIGOR STATUS: STRUCTURAL-REFRAME-VERIFIED")
    print("=" * 72)
    print("OK: M1 reframe is computationally well-defined (5/5 checks pass).")


if __name__ == "__main__":
    main()
