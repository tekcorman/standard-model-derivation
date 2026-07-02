#!/usr/bin/env python3
"""
dark_map_theta23_v2.py — Dark-map taxonomy for θ_23 on the 4×4 Bloch H

After dark_map_taxonomy_theta23.py found that the 12×12 Hashimoto B(k_P)
has the wrong C₃ structure for this analysis, this v2 attempts the same
calculation on the 4×4 vertex-space Bloch Hamiltonian H(k_P), where the
framework's claim is that the spectrum decomposes as 2·trivial ⊕ ω ⊕ ω²
(per predictions/theta_23_PMNS.py Step 1).

GOAL: identify the (ω, ω²) doublet at the P-point of the 4×4 Bloch H
and derive the b₀ normalization from explicit eigenvectors.

Status: Step 1 v2 of theorem_dark_map_taxonomy_attempt.md.
"""

import math
import sys
from pathlib import Path

import numpy as np
from numpy import linalg as la

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from proofs.flavor.srs_bloch_ckm import (
    build_unit_cell,
    find_connectivity,
    bloch_hamiltonian,
)


K_P = (0.25, 0.25, 0.25)


def header(title):
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)
    print()


def main():
    header("DARK-MAP TAXONOMY v2: 4×4 BLOCH HAMILTONIAN AT P")

    # Step 1: build the 4-atom unit cell and find bonds
    verts = build_unit_cell()
    n_verts = len(verts)
    bonds = find_connectivity(verts)
    print(f"Unit cell: {n_verts} atoms, {len(bonds)} directed bonds")

    # Step 2: 4×4 Bloch H at the P-point
    H_P = bloch_hamiltonian(K_P, bonds, n_verts)
    print(f"H(k_P) shape: {H_P.shape}")
    print()

    # Diagonalize
    evals, evecs = la.eig(H_P)
    print(f"Eigenvalues of H(k_P):")
    for i, ev in enumerate(sorted(evals, key=lambda z: -np.real(z) if abs(np.imag(z)) < 1e-9 else -abs(z))):
        print(f"  λ_{i} = {ev:+.6f}, |λ| = {abs(ev):.6f}")
    print()

    # Step 3: Build C₃ generator on vertex space (3-fold rotation around [111])
    # The C₃ rotation maps atoms cyclically. In the 4-atom srs primitive cell,
    # atom 0 is invariant (sits on the body diagonal), atoms 1,2,3 cycle.
    # Standard convention: U_C3 maps |1⟩ → |2⟩ → |3⟩ → |1⟩, |0⟩ → |0⟩.
    # But we should derive this from the actual atom positions.

    # Apply C₃ rotation matrix in Cartesian (about [111] axis, 120°)
    axis = np.array([1, 1, 1]) / np.sqrt(3)
    angle = 2 * np.pi / 3
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R_C3 = (cos_a * np.eye(3)
            + sin_a * np.array([[0, -axis[2], axis[1]],
                                 [axis[2], 0, -axis[0]],
                                 [-axis[1], axis[0], 0]])
            + (1 - cos_a) * np.outer(axis, axis))

    print(f"C₃ rotation matrix R about [111]:")
    print(f"  {R_C3}")
    print()

    # Apply R_C3 to each atom and find which atom it maps to (modulo lattice)
    print("Atom positions and C₃ images:")
    for i, v in enumerate(verts):
        v_rot = R_C3 @ np.array(v)
        # Find closest atom (modulo unit cell)
        closest = min(range(n_verts),
                      key=lambda j: la.norm(v_rot - np.array(verts[j])))
        print(f"  atom {i} = {v} → rotated = {v_rot} → atom {closest}")
    print()

    # Build U_C3 as a permutation matrix on vertex space
    perm = []
    for i, v in enumerate(verts):
        v_rot = R_C3 @ np.array(v)
        # Find atom that v maps to (within tolerance, possibly modulo lattice)
        found = None
        for j, w in enumerate(verts):
            if la.norm(v_rot - np.array(w)) < 0.1:
                found = j
                break
        if found is None:
            # Try with lattice translations
            from itertools import product as iproduct
            for shift in iproduct([-1, 0, 1], repeat=3):
                for j, w in enumerate(verts):
                    if la.norm(v_rot - np.array(w) - np.array(shift)) < 0.1:
                        found = j
                        break
                if found is not None: break
        if found is None:
            print(f"  WARNING: atom {i} has no C₃ image in cell")
            return
        perm.append(found)
    print(f"C₃ permutation: atom i → atom {perm}")
    print()

    U_C3 = np.zeros((n_verts, n_verts), dtype=complex)
    for i, j in enumerate(perm):
        U_C3[j, i] = 1.0
    # Verify U_C3^3 = I
    cubic_err = la.norm(la.matrix_power(U_C3, 3) - np.eye(n_verts))
    print(f"||U_C3^3 - I|| = {cubic_err:.2e}")

    # Verify [U_C3, H_P] = 0
    comm_err = la.norm(U_C3 @ H_P - H_P @ U_C3)
    print(f"||[U_C3, H_P]|| = {comm_err:.2e}")
    print()

    # Step 4: C₃ structure of each H_P eigenspace
    omega = np.exp(2j * np.pi / 3)
    print("Step 4: C₃ structure of each H_P eigenspace")
    # Group eigenvalues
    sorted_evals = sorted(evals, key=lambda z: (np.real(z), np.imag(z)))
    for ev in set(complex(round(e.real, 4), round(e.imag, 4)) for e in evals):
        idx = [i for i, e in enumerate(evals) if abs(e - ev) < 1e-3]
        V = evecs[:, idx]
        Q, _ = la.qr(V)
        Q = Q[:, :len(idx)]
        U_block = Q.conj().T @ U_C3 @ Q
        eigs_block = la.eigvals(U_block)
        irreps = []
        for e in eigs_block:
            if abs(e - 1) < 1e-2: irreps.append('1')
            elif abs(e - omega) < 1e-2: irreps.append('ω')
            elif abs(e - omega**2) < 1e-2: irreps.append('ω²')
            else: irreps.append(f'??({e:.3f})')
        print(f"  E = {ev:+.6f} (dim {len(idx)}): C₃ irreps = {irreps}")
    print()

    print("Expected from theta_23_PMNS.py: 'spectrum at P = 2 × trivial ⊕ ω ⊕ ω²'")


if __name__ == "__main__":
    main()
