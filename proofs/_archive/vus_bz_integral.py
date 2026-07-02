#!/usr/bin/env python3
"""
V_us from BZ integral: three candidate mechanisms.

Result from first attempt: the adjacency resolvent G(k,z) = (zI-H)^{-1}
gives V_us = 0 identically (C₃ selection rule forbids ω→ω² in the
adjacency). This script tries three alternatives:

1. Z₃-TWISTED resolvent: explicitly break C₃ by twist phases
2. NB (Hashimoto) resolvent: the NB constraint may break the selection rule
3. Second-order (two-step) process: ω → trivial → ω² via intermediate state
"""

import numpy as np
from numpy import linalg as la
from itertools import product
import math


def build_srs_cell():
    base = np.array([
        [1/8, 1/8, 1/8], [3/8, 7/8, 5/8],
        [7/8, 5/8, 3/8], [5/8, 3/8, 7/8],
    ])
    bc = (base + 0.5) % 1.0
    return np.vstack([base, bc])


def find_bonds(verts, a=1.0):
    n = len(verts)
    nn_dist = np.sqrt(2) / 4 * a
    tol = 0.01 * a
    bonds = []
    for i in range(n):
        ri = verts[i] * a
        for j in range(n):
            for n1, n2, n3 in product(range(-1, 2), repeat=3):
                if i == j and n1 == n2 == n3 == 0:
                    continue
                rj = verts[j] * a + np.array([n1, n2, n3]) * a
                if abs(la.norm(rj - ri) - nn_dist) < tol:
                    bonds.append((i, j, np.array([n1, n2, n3]), rj - ri))
    return bonds


def bloch_hamiltonian(k, bonds, n_atoms):
    H = np.zeros((n_atoms, n_atoms), dtype=complex)
    for src, tgt, cell, dr in bonds:
        phase = np.exp(2j * np.pi * np.dot(k, cell))
        H[src, tgt] += phase
    return H


def bloch_nb_matrix(k, bonds, n_atoms):
    """
    Bloch NB walk matrix B(k). Rows/columns = directed bonds.
    B_{e1,e2} = phase(e2) if head(e1)=tail(e2) and e2 ≠ reverse(e1).
    """
    n_bonds = len(bonds)
    B = np.zeros((n_bonds, n_bonds), dtype=complex)
    for idx2, (s2, t2, c2, dr2) in enumerate(bonds):
        phase2 = np.exp(2j * np.pi * np.dot(k, c2))
        for idx1, (s1, t1, c1, dr1) in enumerate(bonds):
            if t1 != s2:
                continue
            neg_c1 = -c1
            if t2 == s1 and np.allclose(c2, neg_c1):
                continue
            B[idx1, idx2] = phase2
    return B


def get_c3_structure(verts):
    """Get C₃ permutation and projectors."""
    n = len(verts)
    omega = np.exp(2j * np.pi / 3)

    # C₃ rotation about [111]
    c = np.cos(2 * np.pi / 3)
    s = np.sin(2 * np.pi / 3)
    nx = np.array([1, 1, 1]) / np.sqrt(3)
    nxx = np.array([[0, -nx[2], nx[1]], [nx[2], 0, -nx[0]], [-nx[1], nx[0], 0]])
    R = np.eye(3) * c + s * nxx + (1 - c) * np.outer(nx, nx)

    # Find permutation
    c3_perm = np.zeros(n, dtype=int)
    for i in range(n):
        ri_rot = (R @ verts[i]) % 1.0
        for j in range(n):
            diff = (ri_rot - verts[j]) % 1.0
            if la.norm(diff) < 0.01 or la.norm(diff - 1) < 0.01:
                c3_perm[i] = j
                break
            diff2 = (ri_rot - verts[j] + 1) % 1.0
            if la.norm(diff2) < 0.01:
                c3_perm[i] = j
                break

    C3_mat = np.zeros((n, n), dtype=complex)
    for i in range(n):
        C3_mat[c3_perm[i], i] = 1.0

    eigs, vecs = la.eig(C3_mat)

    idx_triv = [i for i, e in enumerate(eigs) if abs(e - 1) < 0.01]
    idx_w = [i for i, e in enumerate(eigs) if abs(e - omega) < 0.01]
    idx_w2 = [i for i, e in enumerate(eigs) if abs(e - omega**2) < 0.01]

    def projector(indices):
        P = np.zeros((n, n), dtype=complex)
        for i in indices:
            v = vecs[:, i:i+1]
            P += v @ v.conj().T
        return P

    return {
        'perm': c3_perm, 'mat': C3_mat,
        'P_triv': projector(idx_triv),
        'P_omega': projector(idx_w),
        'P_omega2': projector(idx_w2),
        'n_triv': len(idx_triv), 'n_w': len(idx_w), 'n_w2': len(idx_w2),
    }


def main():
    print("=" * 70)
    print("V_us: three candidate mechanisms")
    print("=" * 70)

    verts = build_srs_cell()
    bonds = find_bonds(verts)
    n_atoms = len(verts)
    n_bonds = len(bonds)

    c3 = get_c3_structure(verts)
    P_w = c3['P_omega']
    P_w2 = c3['P_omega2']
    P_triv = c3['P_triv']

    print(f"Cell: {n_atoms} atoms, {n_bonds} directed bonds")
    print(f"C₃ sectors: {c3['n_triv']} trivial, {c3['n_w']} ω, {c3['n_w2']} ω²")

    N = 16  # BZ grid
    dk = 1.0 / N
    omega = np.exp(2j * np.pi / 3)

    # ================================================================
    # METHOD 1: Z₃-TWISTED ADJACENCY RESOLVENT
    # ================================================================
    print(f"\n{'='*70}")
    print("METHOD 1: Z₃-twisted adjacency resolvent")
    print(f"{'='*70}")
    print("H_tw(k) = H(k) with C₃ twist phases on bonds")

    # The Z₃ twist: each bond (i→j) gets phase omega^{gen(j)-gen(i)}
    # In terms of the C₃ matrix: the twist replaces H with C₃†HC₃
    # projected off-diagonally.
    # Actually: the Z₃-twisted Hamiltonian assigns phase omega to
    # bonds that advance the generation and omega* to bonds that
    # go backward.
    #
    # We implement this by multiplying each bond's Bloch phase by
    # omega^{C3_sector_change}.

    for z_try in [2.0, 17/6, 3.0, 3.5]:
        total = 0.0
        for i1 in range(N):
            for i2 in range(N):
                for i3 in range(N):
                    k = np.array([i1+0.5, i2+0.5, i3+0.5]) / N

                    # Twisted Hamiltonian: H_tw_{ij} = omega^{gen(j)-gen(i)} * H_{ij}
                    H = bloch_hamiltonian(k, bonds, n_atoms)
                    C3 = c3['mat']
                    # The generation change for bond i→j: determined by C₃ action
                    # H_tw = C₃† @ H @ C₃ projects to ω-twisted sector
                    # More precisely: the ω-sector resolvent is
                    # G_ω(k,z) = P_ω @ (zI - H)^{-1} @ P_ω
                    # The cross-sector element: P_ω² @ G @ P_ω
                    # But we showed this is zero.
                    #
                    # For the TWISTED version: H → H + δH where δH is the
                    # twist perturbation. But actually, the twist IS what
                    # distinguishes H from G — we want the ω→ω² matrix element.
                    #
                    # Let me try a different approach: multiply H by the
                    # C₃ character to project onto the ω² - ω transition.
                    # G_{ω→ω²}(k) = Tr(P_ω² (zI-H)^{-1} P_ω)
                    # This was zero.
                    #
                    # The TWISTED resolvent: (zI - ω·H)^{-1} or similar?
                    # Try: G_tw = (zI - H·C₃)^{-1}

                    HC3 = H @ C3
                    G_tw = la.inv(z_try * np.eye(n_atoms) - HC3)
                    val = np.trace(G_tw)  # total twisted amplitude
                    total += val

        total *= dk**3
        print(f"  z={z_try:.4f}: Tr(G_tw) = {total:.6f} (|.|={abs(total):.6f})")

    # ================================================================
    # METHOD 2: NB (HASHIMOTO) RESOLVENT
    # ================================================================
    print(f"\n{'='*70}")
    print("METHOD 2: NB walk resolvent (Hashimoto matrix)")
    print(f"{'='*70}")
    print(f"B(k) is {n_bonds}×{n_bonds}. Extracting atom-to-atom amplitudes.")

    # Build C₃ projectors on the BOND space
    # Each bond (src, tgt, cell) maps under C₃ to (c3_perm[src], c3_perm[tgt], R·cell)
    # The bond-space C₃ permutation:
    c3_perm = c3['perm']

    # For each bond idx, find the C₃-image bond
    bond_c3_perm = np.zeros(n_bonds, dtype=int)
    for idx, (s, t, cell, dr) in enumerate(bonds):
        s2 = c3_perm[s]
        t2 = c3_perm[t]
        # The rotated cell displacement
        c_rot = np.cos(2*np.pi/3)
        s_rot = np.sin(2*np.pi/3)
        nx = np.array([1,1,1])/np.sqrt(3)
        nxx = np.array([[0,-nx[2],nx[1]],[nx[2],0,-nx[0]],[-nx[1],nx[0],0]])
        R = np.eye(3)*c_rot + s_rot*nxx + (1-c_rot)*np.outer(nx,nx)
        cell_rot = np.round(R @ cell).astype(int)
        # Find matching bond
        for jdx, (sj, tj, cj, drj) in enumerate(bonds):
            if sj == s2 and tj == t2 and np.allclose(cj, cell_rot):
                bond_c3_perm[idx] = jdx
                break

    C3_bond = np.zeros((n_bonds, n_bonds), dtype=complex)
    for i in range(n_bonds):
        C3_bond[bond_c3_perm[i], i] = 1.0

    # C₃ eigendecomposition on bond space
    eigs_b, vecs_b = la.eig(C3_bond)
    idx_w_b = [i for i, e in enumerate(eigs_b) if abs(e - omega) < 0.01]
    idx_w2_b = [i for i, e in enumerate(eigs_b) if abs(e - omega**2) < 0.01]
    idx_triv_b = [i for i, e in enumerate(eigs_b) if abs(e - 1) < 0.01]

    print(f"Bond C₃ sectors: {len(idx_triv_b)} trivial, {len(idx_w_b)} ω, {len(idx_w2_b)} ω²")

    def bond_projector(indices):
        P = np.zeros((n_bonds, n_bonds), dtype=complex)
        for i in indices:
            v = vecs_b[:, i:i+1]
            P += v @ v.conj().T
        return P

    P_w_b = bond_projector(idx_w_b)
    P_w2_b = bond_projector(idx_w2_b)

    for d_try in [1, 2, 3, 4, 5, 8]:
        total = 0.0
        for i1 in range(N):
            for i2 in range(N):
                for i3 in range(N):
                    k = np.array([i1+0.5, i2+0.5, i3+0.5]) / N
                    B = bloch_nb_matrix(k, bonds, n_atoms)
                    Bd = la.matrix_power(B, d_try)
                    # ω → ω² element
                    val = np.trace(P_w2_b @ Bd @ P_w_b)
                    total += val
        total *= dk**3
        print(f"  B^{d_try}: Tr(P_ω² B^d P_ω) = {total.real:+.6e} {total.imag:+.6e}i "
              f"(|.|={abs(total):.6e})")

    # ================================================================
    # METHOD 3: SECOND-ORDER (TWO-STEP) PROCESS
    # ================================================================
    print(f"\n{'='*70}")
    print("METHOD 3: Second-order ω → trivial → ω²")
    print(f"{'='*70}")
    print("V_us = sum_k Tr(P_ω² G P_triv) × Tr(P_triv G P_ω)")

    for z_try in [2.0, 17/6, 3.0, 3.5, 4.0]:
        total = 0.0
        for i1 in range(N):
            for i2 in range(N):
                for i3 in range(N):
                    k = np.array([i1+0.5, i2+0.5, i3+0.5]) / N
                    H = bloch_hamiltonian(k, bonds, n_atoms)
                    G = la.inv(z_try * np.eye(n_atoms) - H)

                    # Two-step: ω → triv, then triv → ω²
                    step1 = np.trace(P_triv @ G @ P_w)      # ω → trivial
                    step2 = np.trace(P_w2 @ G @ P_triv)     # trivial → ω²
                    total += step1 * step2

        total *= dk**3
        print(f"  z={z_try:.4f}: V_us^(2) = {total.real:+.6e} {total.imag:+.6e}i "
              f"(|.|={abs(total):.6e})")

    # ================================================================
    # METHOD 3b: G² (second order resolvent directly)
    # ================================================================
    print(f"\n{'='*70}")
    print("METHOD 3b: Tr(P_ω² G² P_ω) — second order in resolvent")
    print(f"{'='*70}")

    for z_try in [2.0, 17/6, 3.0, 3.5, 4.0, 5.0]:
        total = 0.0
        for i1 in range(N):
            for i2 in range(N):
                for i3 in range(N):
                    k = np.array([i1+0.5, i2+0.5, i3+0.5]) / N
                    H = bloch_hamiltonian(k, bonds, n_atoms)
                    G = la.inv(z_try * np.eye(n_atoms) - H)
                    G2 = G @ G
                    val = np.trace(P_w2 @ G2 @ P_w)
                    total += val
        total *= dk**3
        print(f"  z={z_try:.4f}: Tr(P_ω² G² P_ω) = {total.real:+.6e} {total.imag:+.6e}i "
              f"(|.|={abs(total):.6e})")

    # ================================================================
    # COMPARISON
    # ================================================================
    print(f"\n{'='*70}")
    print("TARGETS")
    print(f"{'='*70}")
    print(f"  V_us_bare = (2/3)^(2+√3) = {(2/3)**(2+np.sqrt(3)):.10f}")
    print(f"  V_us_obs  = 0.2250")
    print(f"  α₁ = (2/3)^8            = {(2/3)**8:.10f}")


if __name__ == '__main__':
    main()
