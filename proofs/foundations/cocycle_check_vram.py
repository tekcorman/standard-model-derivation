#!/usr/bin/env python3
"""
Cocycle check: does the projective T = A_4 cocycle trivialize on V_Ram?

This is the computational check requested for theorem_need_a2_cocycle_check.md.
It checks whether U_{C_2}|_{V_Ram} and U_{C_3}|_{V_Ram} satisfy the A_4
group relation: U_{C_2} U_{C_3} U_{C_2}^{-1} = U_{C_3}^2.

Steps 1-5 as requested in the handoff prompt.
"""

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
from numpy import linalg as la

from proofs.common import find_bonds, omega3
from proofs.foundations.theorem_B5_3_core import (
    build_directed_edges,
    bloch_hashimoto,
    build_c3_on_directed_edges,
)

K_P = (0.25, 0.25, 0.25)
H_EXACT = (math.sqrt(3) + 1j * math.sqrt(5)) / 2


# -----------------------------------------------------------------------
# Build U_{C_2} on the 12-dim directed-edge fiber
# Implements the (C_2z | tau=(0, 1/2, 0)) space group operation
# as derived in theorem_need_a2_t_equivariance_attempt.md Section 1.
# -----------------------------------------------------------------------

def build_c2_on_directed_edges(directed, k_frac):
    """
    Build U_{C_2} as a phased permutation matrix on directed edges.

    The C_2z rotation in I4_132 acts as:
        (x, y, z) -> (-x, -y, z)
    with fractional translation tau = (0, 1/2, 0) in conventional cubic.

    Atom mapping (Wyckoff 8a, x=1/8, per theorem_need_a2_t_equivariance_attempt.md):
        atom 0 -> atom 1 + cell offset: we compute this from the space group action.

    The Bloch phase factor for U_{C_2} at P is:
        [U_{C_2}]_{e', e} = exp(2*pi*i * k . cs2_cell) * permutation
    where cs2_cell is the cell shift of the target atom under C_2z.

    We derive the atom permutation and cell shifts by applying C_2z to each atom.
    """
    from proofs.common import ATOMS, A_PRIM, N_ATOMS
    from itertools import product as iproduct

    tol = 0.02
    k = np.array(k_frac, dtype=float)

    # Apply C_2z + tau to each atom, find which atom+cell it maps to
    # C_2z in Cartesian: (x,y,z) -> (-x,-y,z); tau in conventional = (0, 0.5, 0)
    # In primitive fractional coords we need to convert tau first.
    # Conventional tau = (0, 0.5, 0) Cartesian.
    # Convert to primitive fractional: solve tau = n1*a1 + n2*a2 + n3*a3
    # a1 = (-0.5, 0.5, 0.5), a2 = (0.5, -0.5, 0.5), a3 = (0.5, 0.5, -0.5)
    # We work in Cartesian and then convert cell offsets to primitive.

    A_PRIM_arr = A_PRIM  # shape (3,3), rows are a1, a2, a3

    # Inverse of A_PRIM for converting Cartesian -> primitive fractional
    # A_PRIM_arr.T . coords_prim = coords_cart
    A_inv = la.inv(A_PRIM_arr.T)

    # C_2z rotation matrix in Cartesian
    C2z_cart = np.diag([-1.0, -1.0, 1.0])

    # tau in Cartesian (conventional a=1): (0, 0.5, 0)
    tau_cart = np.array([0.0, 0.5, 0.0])

    # Find atom permutation and cell shifts
    atom_map = {}   # src_atom -> (dst_atom, cell_shift_primitive)
    for i in range(N_ATOMS):
        ri_cart = ATOMS[i]
        # Apply C_2z + tau
        ri_new_cart = C2z_cart @ ri_cart + tau_cart
        # Find which atom j + cell it equals
        found = False
        for j in range(N_ATOMS):
            rj_cart = ATOMS[j]
            # Check all nearby cells
            for n in iproduct(range(-2, 3), repeat=3):
                rj_shifted = rj_cart + sum(n[k_] * A_PRIM_arr[k_] for k_ in range(3))
                if la.norm(ri_new_cart - rj_shifted) < tol:
                    # ri maps to atom j with cell offset n (in primitive coords)
                    atom_map[i] = (j, np.array(n, dtype=int))
                    found = True
                    break
            if found:
                break
        if not found:
            raise RuntimeError(f"No image found for atom {i} under C_2z+tau")

    # Build phased permutation on directed edges
    n = len(directed)
    edge_to_idx = {de: i for i, de in enumerate(directed)}

    U = np.zeros((n, n), dtype=complex)
    for i, (src, tgt, cell) in enumerate(directed):
        # Image of edge (src -> tgt with cell offset) under C_2z + tau:
        # src -> atom_map[src][0] + cell atom_map[src][1]
        # tgt -> atom_map[tgt][0] + cell atom_map[tgt][1]
        # The cell of the new edge:
        # new_cell = C_2z_cell(cell) + atom_map[tgt][1] - atom_map[src][1]
        # C_2z on primitive cell: need to apply C_2z to the cell vector too
        # Cell vector in Cartesian: cell[0]*a1 + cell[1]*a2 + cell[2]*a3
        cell_cart = sum(cell[k_] * A_PRIM_arr[k_] for k_ in range(3))
        new_cell_cart = C2z_cart @ cell_cart
        # Convert back to primitive fractional
        new_cell_prim = A_inv @ new_cell_cart
        new_cell_int = tuple(int(round(x)) for x in new_cell_prim)

        dst_src, src_cell_shift = atom_map[src]
        dst_tgt, tgt_cell_shift = atom_map[tgt]

        total_cell = tuple(
            new_cell_int[k_] + tgt_cell_shift[k_] - src_cell_shift[k_]
            for k_ in range(3)
        )

        new_edge = (dst_src, dst_tgt, total_cell)
        j = edge_to_idx.get(new_edge)
        if j is None:
            raise RuntimeError(
                f"C_2z mapped {(src, tgt, cell)} -> {new_edge}, not in edge set"
            )

        # Bloch phase: exp(2*pi*i * k . tgt_cell_shift)
        # (the phase arises from the non-symmorphic tau acting on the
        # target atom's position in the fiber; matches the derivation in
        # theorem_need_a2_t_equivariance_attempt.md)
        phase = np.exp(2j * np.pi * np.dot(k, tgt_cell_shift))
        U[j, i] = phase

    return U, atom_map


def project_to_subspace(M, basis):
    """Project M onto the subspace spanned by basis columns.
    Returns the matrix M restricted to that subspace (in basis coordinates).
    basis: n x d matrix with orthonormal columns (or columns to be orthonormalized)
    """
    Q, _ = la.qr(basis)
    d = basis.shape[1]
    Q = Q[:, :d]
    return Q.conj().T @ M @ Q


def find_vram_basis(B_P, h_exact):
    """Find the 8-dim V_Ram subspace: eigenvectors with eigenvalues +-h, +-h*."""
    evals, evecs = la.eig(B_P)
    targets = [h_exact, h_exact.conjugate(), -h_exact, -h_exact.conjugate()]
    indices = []
    for i, ev in enumerate(evals):
        if any(abs(ev - t) < 1e-6 for t in targets):
            indices.append(i)
    assert len(indices) == 8, f"Expected 8 V_Ram eigenvectors, got {len(indices)}"
    return evecs[:, indices]


def group_order(U3, U2, max_iter=200):
    """
    Estimate order of group generated by U3, U2 by enumerating products.
    Returns the number of distinct group elements (up to max_iter).
    Uses exact matrix comparison (norm < 1e-6) without hashing.
    """
    I = np.eye(U3.shape[0], dtype=complex)
    elements = [I]
    queue = [I]
    gens = [U3, U2]
    while queue:
        g = queue.pop(0)
        for gen in gens:
            h = g @ gen
            is_new = all(la.norm(h - e) > 1e-6 for e in elements)
            if is_new:
                elements.append(h)
                queue.append(h)
        if len(elements) > max_iter:
            return len(elements), False  # truncated
    return len(elements), True


def a4_irrep_decomposition(chi_e, chi_c3, chi_c3sq, chi_c2):
    """
    Decompose character into A_4 irreps.
    Classes: e (size 1), C_3 (size 4), C_3^2 (size 4), C_2 (size 3).
    chi_2 and chi_3 use chi(C_3) = omega, omega^2 resp; chi(C_2)=1 for both.
    chi_4: dim 3, chi(C_3)=0, chi(C_2)=-1.
    |A_4| = 12.
    """
    w = omega3
    # m(chi_rho) = (1/|G|) * sum_{classes} |class| * chi_rho(g)^* * chi(g)
    m1 = (1*chi_e + 4*chi_c3 + 4*chi_c3sq + 3*chi_c2) / 12
    m2 = (1*chi_e + 4*np.conj(w)*chi_c3 + 4*np.conj(w**2)*chi_c3sq + 3*1*chi_c2) / 12
    m3 = (1*chi_e + 4*np.conj(w**2)*chi_c3 + 4*np.conj(w)*chi_c3sq + 3*1*chi_c2) / 12
    m4 = (1*3*chi_e + 4*0*chi_c3 + 4*0*chi_c3sq + 3*(-1)*chi_c2) / 12
    return m1, m2, m3, m4


def main():
    print("=" * 72)
    print("COCYCLE CHECK: Does A_4 projective cocycle trivialize on V_Ram?")
    print("=" * 72)

    bonds = find_bonds()
    directed = build_directed_edges(bonds)

    # Build B(P), U_{C_3}, U_{C_2}
    B_P = bloch_hashimoto(K_P, directed)
    U_c3 = build_c3_on_directed_edges(directed)
    U_c2, atom_map = build_c2_on_directed_edges(directed, K_P)

    print("\nAtom mapping under C_2z + tau:")
    for src, (dst, cell) in sorted(atom_map.items()):
        print(f"  atom {src} -> atom {dst} + cell {tuple(cell)}")

    # Verify U_{C_2} properties on the full 12-dim space
    comm_c2 = la.norm(B_P @ U_c2 - U_c2 @ B_P)
    order_c2 = la.norm(U_c2 @ U_c2 - np.eye(12))
    print(f"\nFull 12-dim checks:")
    print(f"  ||[B(P), U_C2]|| = {comm_c2:.2e}  (must be < 1e-10)")
    print(f"  ||U_C2^2 - I||   = {order_c2:.2e}  (must be < 1e-10)")

    # Check the A_4 relation on full 12-dim space
    lhs_full = U_c2 @ U_c3 @ la.inv(U_c2)
    rhs_full = U_c3 @ U_c3
    diff_full = la.norm(lhs_full - rhs_full)
    print(f"  ||U_C2 U_C3 U_C2^-1 - U_C3^2|| (full) = {diff_full:.4f}  (expected ~4.90)")

    # -----------------------------------------------------------------------
    # STEP 1: Extract V_Ram basis and restrict generators
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: V_Ram basis and 8x8 restrictions")

    V_Ram_basis = find_vram_basis(B_P, H_EXACT)
    print(f"  V_Ram dim = {V_Ram_basis.shape[1]}  (expected 8)")

    # Project U_{C_3} and U_{C_2} onto V_Ram
    # Check that V_Ram is stable under both generators first
    Q, _ = la.qr(V_Ram_basis)
    Q = Q[:, :8]

    P_Ram = Q @ Q.conj().T  # Projector onto V_Ram

    # Stability check: does U send V_Ram into V_Ram?
    stab_c3 = la.norm(P_Ram @ U_c3 @ P_Ram - P_Ram @ U_c3)
    stab_c2 = la.norm(P_Ram @ U_c2 @ P_Ram - P_Ram @ U_c2)
    print(f"  V_Ram stability under U_C3: ||P U_C3 P - P U_C3|| = {stab_c3:.2e}")
    print(f"  V_Ram stability under U_C2: ||P U_C2 P - P U_C2|| = {stab_c2:.2e}")

    # 8x8 restrictions
    Ur_c3 = Q.conj().T @ U_c3 @ Q   # 8x8
    Ur_c2 = Q.conj().T @ U_c2 @ Q   # 8x8
    print(f"  Ur_C3 shape: {Ur_c3.shape}")
    print(f"  Ur_C2 shape: {Ur_c2.shape}")

    # -----------------------------------------------------------------------
    # STEP 2: Check A_4 group relation on V_Ram
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: A_4 group relation on V_Ram")

    lhs = Ur_c2 @ Ur_c3 @ la.inv(Ur_c2)
    rhs = Ur_c3 @ Ur_c3
    diff_ram = la.norm(lhs - rhs)
    print(f"  ||U_C2 U_C3 U_C2^-1 - U_C3^2||_VRam = {diff_ram:.6e}")
    if diff_ram < 1e-8:
        print("  -> COCYCLE TRIVIALIZES ON V_Ram (norm < 1e-8)")
        cocycle_trivial = True
    else:
        print(f"  -> COCYCLE DOES NOT TRIVIALIZE ON V_Ram (norm = {diff_ram:.4f})")
        cocycle_trivial = False

    # -----------------------------------------------------------------------
    # STEP 3: Order checks on V_Ram
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Order checks on V_Ram")

    I8 = np.eye(8, dtype=complex)
    order3_check = la.norm(Ur_c3 @ Ur_c3 @ Ur_c3 - I8)
    order2_check = la.norm(Ur_c2 @ Ur_c2 - I8)
    print(f"  ||Ur_C3^3 - I|| = {order3_check:.2e}")
    print(f"  ||Ur_C2^2 - I|| = {order2_check:.2e}")

    print("\n  Estimating group order...")
    grp_order, complete = group_order(Ur_c3, Ur_c2, max_iter=200)
    status = "exact" if complete else "lower bound (truncated at 200)"
    print(f"  Group order generated by Ur_C3, Ur_C2: {grp_order} ({status})")

    # -----------------------------------------------------------------------
    # STEP 4: Character analysis and irrep decomposition
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Character analysis of V_Ram under A_4")

    chi_e = 8
    chi_c3 = np.trace(Ur_c3)
    chi_c3sq = np.trace(Ur_c3 @ Ur_c3)
    chi_c2 = np.trace(Ur_c2)

    print(f"  chi(e)    = {chi_e}")
    print(f"  chi(C_3)  = {chi_c3.real:.6f} {chi_c3.imag:+.6f}i  (expected 2)")
    print(f"  chi(C_3^2)= {chi_c3sq.real:.6f} {chi_c3sq.imag:+.6f}i  (expected 2)")
    print(f"  chi(C_2)  = {chi_c2.real:.6f} {chi_c2.imag:+.6f}i")

    if cocycle_trivial:
        m1, m2, m3, m4 = a4_irrep_decomposition(chi_e, chi_c3, chi_c3sq, chi_c2)
        print(f"\n  A_4 irrep multiplicities:")
        print(f"    m(chi_1) = {m1.real:.4f}  (trivial 1-dim)")
        print(f"    m(chi_2) = {m2.real:.4f}  (1-dim, C_3 eigenvalue omega)")
        print(f"    m(chi_3) = {m3.real:.4f}  (1-dim, C_3 eigenvalue omega^2)")
        print(f"    m(chi_4) = {m4.real:.4f}  (standard 3-dim irrep)")

        m1r = int(round(m1.real))
        m2r = int(round(m2.real))
        m3r = int(round(m3.real))
        m4r = int(round(m4.real))
        print(f"\n  V_Ram = {m1r}*chi_1 + {m2r}*chi_2 + {m3r}*chi_3 + {m4r}*chi_4")
        check_dim = m1r + m2r + m3r + 3*m4r
        print(f"  Dimension check: {m1r} + {m2r} + {m3r} + 3*{m4r} = {check_dim}  (expected 8)")
    else:
        print("\n  Cocycle does not trivialize; Schur decomposition into A_4 irreps invalid.")
        m4r = None

    # -----------------------------------------------------------------------
    # STEP 5: Summary report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 5: SUMMARY REPORT")
    print("=" * 72)
    print(f"  Projective obstruction vanishes on V_Ram: {'YES' if cocycle_trivial else 'NO'}")
    print(f"  ||U_C2 U_C3 U_C2^-1 - U_C3^2||_{'{V_Ram}'} = {diff_ram:.4e}  (threshold 1e-8)")
    if cocycle_trivial:
        print(f"  A_4 content: {m1r}*chi_1 + {m2r}*chi_2 + {m3r}*chi_3 + {m4r}*chi_4")
        print(f"  Copies of 3-dim irrep chi_4 in V_Ram: {m4r}")
    else:
        print(f"  Minimal extension group check needed.")
        print(f"  Group order on V_Ram: {grp_order}")

    return {
        'cocycle_trivial': cocycle_trivial,
        'diff_ram': diff_ram,
        'order3_check': order3_check,
        'order2_check': order2_check,
        'group_order': grp_order,
        'chi_e': chi_e,
        'chi_c3': chi_c3,
        'chi_c3sq': chi_c3sq,
        'chi_c2': chi_c2,
    }


if __name__ == "__main__":
    main()
