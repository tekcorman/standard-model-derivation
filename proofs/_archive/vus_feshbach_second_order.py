#!/usr/bin/env python3
"""
V_us from second-order Feshbach: Σ G_triv Σ.

KEY INSIGHT: V_us = 0 on the compressed graph (C₃ selection rule).
The ENTIRE Cabibbo angle comes from the dark sector self-energy Σ.

Σ transforms as ω² under C₃. The ω→ω² transition requires TWO
Σ insertions (each shifts C₃ charge by ω²; two shifts = ω⁴ = ω,
combined with ⟨ω²| giving total ω·ω = ω² which cancels with ω²
from the bra... let me redo this).

Actually: ⟨ω²|Σ|triv⟩ ≠ 0 (selection: ω × ω² × 1 = 1 ✓)
          ⟨triv|Σ|ω⟩ ≠ 0 (selection: 1 × ω² × ω = 1 ✓)

So: V_us = ⟨ω²| Σ G_triv Σ |ω⟩ where G_triv propagates through
the trivial sector intermediate state.

This is a RESONANT second-order process — the trivial sector
propagator G_triv enhances the amplitude near the band edge.
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
        H[src, tgt] += np.exp(2j * np.pi * np.dot(k, cell))
    return H


def get_c3(verts):
    n = len(verts)
    omega = np.exp(2j * np.pi / 3)
    c = np.cos(2 * np.pi / 3)
    s = np.sin(2 * np.pi / 3)
    nx = np.array([1, 1, 1]) / np.sqrt(3)
    nxx = np.array([[0, -nx[2], nx[1]], [nx[2], 0, -nx[0]], [-nx[1], nx[0], 0]])
    R = np.eye(3) * c + s * nxx + (1 - c) * np.outer(nx, nx)

    c3_perm = np.zeros(n, dtype=int)
    for i in range(n):
        ri_rot = (R @ verts[i]) % 1.0
        for j in range(n):
            if la.norm((ri_rot - verts[j]) % 1.0) < 0.01 or \
               la.norm(((ri_rot - verts[j]) % 1.0) - 1) < 0.01:
                c3_perm[i] = j
                break

    C3 = np.zeros((n, n), dtype=complex)
    for i in range(n):
        C3[c3_perm[i], i] = 1.0

    eigs, vecs = la.eig(C3)

    def proj(target_eig):
        P = np.zeros((n, n), dtype=complex)
        for i in range(n):
            if abs(eigs[i] - target_eig) < 0.01:
                v = vecs[:, i:i+1]
                P += v @ v.conj().T
        return P

    return {
        'C3': C3, 'perm': c3_perm,
        'P1': proj(1.0), 'Pw': proj(omega), 'Pw2': proj(omega**2),
    }


def build_sigma_matrix(c3_data, alpha_1, h):
    """
    Build the self-energy Σ as a matrix on the 8-atom cell.

    Σ transforms as ω² under C₃. It connects:
      |ω⟩ → |trivial⟩  (shift by ω²: ω × ω² = ω³ = 1)
      |trivial⟩ → |ω²⟩  (shift by ω²: 1 × ω² = ω²)
      |ω²⟩ → |ω⟩        (shift by ω²: ω² × ω² = ω)

    The simplest ω²-transforming operator on the C₃ eigenspaces:
    Σ_mat acts as the C₃-shift operator (mapping each sector to
    the next one shifted by ω²), scaled by (α₁/2)|h*|.

    Constructing from C₃²: since C₃ has eigenvalue ω on |ω⟩,
    C₃² has eigenvalue ω² on |ω⟩. So C₃² maps |ω⟩→ω²|ω⟩, not
    |ω⟩→|something else⟩. C₃ is diagonal in its own eigenbasis!

    The OFF-DIAGONAL coupling must come from the adjacency structure.
    The operator that connects different C₃ sectors is H itself,
    restricted to the off-diagonal blocks.

    In the C₃ basis, A(Γ) = [[A_triv, 0, 0], [0, A_ω, 0], [0, 0, A_ω²]]
    (block diagonal). The C₃-breaking comes from Σ, which is NOT
    block diagonal.

    The self-energy Σ from the dark sector has matrix elements:
    Σ_{ab} = Σ₀ × M_{ab}
    where M is the coupling matrix between atoms through the dark sector,
    and Σ₀ = α₁/h is the scalar self-energy.

    The simplest model for M: each atom couples to all others equally
    through the dark sector (isotropic). Then M = (J - I)/3 (off-diagonal,
    normalized). But this has all C₃ quantum numbers, not just ω².

    The ω²-projected coupling:
    M_ω² = (1/3)(M + ω² C₃ M C₃† + ω C₃² M C₃†²)

    For M = J-I (all-to-all): C₃ M C₃† = M (since M is C₃-invariant).
    So M_ω² = M × (1/3)(1 + ω² + ω) = 0. The all-to-all coupling
    is C₃-invariant and has no ω² component!

    This is the same zero we found before. The isotropic dark sector
    doesn't break C₃.

    For Σ to have a nonzero ω² component, the dark coupling must be
    ANISOTROPIC — the three edges at each vertex must couple to the
    dark sector with different strengths depending on their C₃ label.
    """
    n = c3_data['C3'].shape[0]
    P1 = c3_data['P1']
    Pw = c3_data['Pw']
    Pw2 = c3_data['Pw2']

    # The self-energy Σ as an operator on atom space:
    # Σ = (α₁/2) h* × (C₃-breaking coupling matrix)
    #
    # The C₃-breaking coupling: each EDGE at a vertex couples to the
    # dark sector. The three edges have C₃ labels {1, ω, ω²}.
    # Edge with label ω² provides the ω²-transforming coupling.
    #
    # On srs: each vertex has 3 edges. Under C₃, these cycle:
    # edge_a → edge_b → edge_c → edge_a.
    # The ω²-component of the edge coupling is:
    # Σ_edge^{ω²} = (1/3)(edge_a + ω·edge_b + ω²·edge_c)
    #
    # This extracts the ω²-Fourier component of the edge sum.
    # On K4: the adjacency A = Σ edges. The ω²-component:
    # A^{ω²} = (1/3)(A + ω·C₃AC₃† + ω²·C₃²A(C₃²)†)

    C3 = c3_data['C3']
    omega = np.exp(2j * np.pi / 3)

    return {
        'scalar': alpha_1 / h,  # Σ(h) = α₁/h
        'P1': P1, 'Pw': Pw, 'Pw2': Pw2,
    }


def main():
    print("=" * 70)
    print("V_us from second-order Feshbach: Σ G_triv Σ")
    print("=" * 70)

    verts = build_srs_cell()
    bonds = find_bonds(verts)
    n = len(verts)
    c3 = get_c3(verts)
    omega = np.exp(2j * np.pi / 3)

    P1 = c3['P1']
    Pw = c3['Pw']
    Pw2 = c3['Pw2']
    C3 = c3['C3']

    h = complex(np.sqrt(3)/2, np.sqrt(5)/2)
    alpha_1 = (2/3)**8
    Sigma_scalar = alpha_1 / h

    print(f"\nΣ(h) = α₁/h = {Sigma_scalar}")
    print(f"|Σ| = {abs(Sigma_scalar):.6f}")
    print(f"α₁ = {alpha_1:.6f}")

    # ================================================================
    # THE C₃-BREAKING OPERATOR
    # ================================================================
    # The dark sector coupling through the ADJACENCY is C₃-symmetric
    # (we proved this — BZ integral gives zero).
    #
    # The C₃ breaking must come from the EDGES, not the vertices.
    # Each edge at a vertex carries a C₃ label. The self-energy Σ
    # couples through specific edges, and the edge labels break C₃.
    #
    # On srs: the 3 edges at each vertex transform as the regular
    # representation of C₃: {1, ω, ω²}. The ω²-component of the
    # edge-weighted adjacency gives the C₃-breaking coupling.
    #
    # A^{ω²} = (1/3)(A + ω·C₃AC₃⁻¹ + ω²·C₃²A(C₃²)⁻¹)
    # where A is the Bloch Hamiltonian.
    # BUT: C₃AC₃⁻¹ = A (since H commutes with C₃ by symmetry).
    # So A^{ω²} = A·(1/3)(1+ω+ω²) = 0. Zero again!
    #
    # The C₃ breaking is MORE SUBTLE. It must come from the
    # BOND DISPLACEMENT VECTORS, not the adjacency matrix.
    # The three bonds at each vertex have displacement vectors
    # d₁, d₂, d₃ in R³. Under C₃, these cycle: d₁→d₂→d₃→d₁.
    # The ω²-component:
    # d^{ω²} = (d₁ + ω·d₂ + ω²·d₃)/3
    #
    # This is a VECTOR in R³ with definite C₃ quantum number ω².
    # The self-energy couples to this vector through the Bloch phase:
    # Σ_{ij}(k) = (α₁/h) × exp(ik·d^{ω²}_{ij})
    #
    # Let me compute d^{ω²} for each vertex.

    print(f"\n{'='*70}")
    print("EDGE C₃ DECOMPOSITION")
    print(f"{'='*70}")

    # For each vertex, find its 3 bonds and decompose under C₃
    for v in range(n):
        v_bonds = [(j, cell, dr) for (s, j, cell, dr) in bonds if s == v]
        assert len(v_bonds) == 3, f"vertex {v}: {len(v_bonds)} bonds"

        # The C₃ rotation permutes these 3 bonds.
        # Find the permutation by rotating the displacement vectors.
        c_rot = np.cos(2 * np.pi / 3)
        s_rot = np.sin(2 * np.pi / 3)
        nx = np.array([1, 1, 1]) / np.sqrt(3)
        nxx = np.array([[0, -nx[2], nx[1]], [nx[2], 0, -nx[0]], [-nx[1], nx[0], 0]])
        R = np.eye(3) * c_rot + s_rot * nxx + (1 - c_rot) * np.outer(nx, nx)

        # Rotate each bond vector and find the matching bond
        dr_list = [dr for (j, cell, dr) in v_bonds]

        # ω²-component: d^{ω²} = (d₀ + ω·d₁ + ω²·d₂)/3
        # where d₀, d₁, d₂ are ordered by C₃ cycling.

        # Find the cycling order: d₀ → R·d₀ should be d₁
        dr0 = dr_list[0]
        dr0_rot = R @ dr0

        # Find which bond dr0_rot matches
        best_match = -1
        for idx, dr in enumerate(dr_list):
            if la.norm(dr0_rot - dr) < 0.01:
                best_match = idx
                break

        if best_match == -1:
            # The rotated bond might go to a different cell image
            # Try with cell shifts
            for idx, (j, cell, dr) in enumerate(v_bonds):
                if la.norm(dr0_rot - dr) < 0.05:
                    best_match = idx
                    break

        if v < 4:  # print first 4 vertices
            print(f"\n  Vertex {v}:")
            for idx, (j, cell, dr) in enumerate(v_bonds):
                print(f"    bond {idx}: → atom {j}, cell {cell}, "
                      f"dr=({dr[0]:+.4f},{dr[1]:+.4f},{dr[2]:+.4f})")

            # Compute ω²-component regardless of ordering
            d_w2 = sum(omega**(2*idx) * dr_list[idx] for idx in range(3)) / 3
            print(f"    d^(ω²) = ({d_w2[0]:.4f}, {d_w2[1]:.4f}, {d_w2[2]:.4f})")
            print(f"    |d^(ω²)| = {la.norm(d_w2):.6f}")

    # ================================================================
    # BZ INTEGRAL WITH C₃-BREAKING BLOCH PHASE
    # ================================================================
    print(f"\n{'='*70}")
    print("BZ INTEGRAL: second-order with C₃-breaking Bloch phases")
    print(f"{'='*70}")

    # The idea: the Bloch phase exp(ik·d) depends on the EDGE,
    # not just the vertex pair. At k ≠ 0, the three edges at a vertex
    # pick up DIFFERENT phases, breaking C₃.
    #
    # The C₃-breaking part of H(k):
    # H^{ω²}(k) = (1/3)[H(k) + ω·C₃H(k)C₃⁻¹ + ω²·C₃²H(k)(C₃²)⁻¹]
    #
    # But wait: C₃H(k)C₃⁻¹ = H(C₃⁻¹·k) (C₃ rotates the momentum).
    # At k = 0: C₃⁻¹·k = 0, so C₃H(0)C₃⁻¹ = H(0), and the ω²
    # projection vanishes (as we found).
    # At k ≠ 0: C₃⁻¹·k ≠ k (unless k is along [111]), so
    # C₃H(k)C₃⁻¹ = H(C₃⁻¹·k) ≠ H(k). The ω² projection is NONZERO!
    #
    # This is the mechanism: the BLOCH PHASES at k ≠ 0 break C₃
    # because the momentum k is not aligned with the C₃ axis.

    # Compute H^{ω²}(k) at each k-point and use it as the coupling
    N = 16
    dk = 1.0 / N

    c_rot = np.cos(2 * np.pi / 3)
    s_rot = np.sin(2 * np.pi / 3)
    nx = np.array([1, 1, 1]) / np.sqrt(3)
    nxx = np.array([[0, -nx[2], nx[1]], [nx[2], 0, -nx[0]], [-nx[1], nx[0], 0]])
    Rmat = np.eye(3) * c_rot + s_rot * nxx + (1 - c_rot) * np.outer(nx, nx)
    C3_inv = c3['C3'].conj().T  # C₃⁻¹

    z_star = 17.0 / 6.0  # spectral parameter

    # Method: V_us = ∫ dk Tr(P_ω² H^{ω²}(k) G_triv(k) H^{ω²}(k)† P_ω)
    # scaled by (α₁/h)²

    total_direct = 0.0
    total_hw2_norm = 0.0

    for i1 in range(N):
        for i2 in range(N):
            for i3 in range(N):
                k = np.array([i1 + 0.5, i2 + 0.5, i3 + 0.5]) / N

                # H(k)
                Hk = bloch_hamiltonian(k, bonds, n)

                # H at C₃-rotated momentum
                k_rot = Rmat @ k  # C₃⁻¹ · k (fractional coords)
                Hk_rot = bloch_hamiltonian(k_rot, bonds, n)

                # H at C₃²-rotated momentum
                k_rot2 = Rmat @ k_rot
                Hk_rot2 = bloch_hamiltonian(k_rot2, bonds, n)

                # C₃-transformed H: C₃ H(C₃⁻¹k) C₃⁻¹
                C3Hk = C3 @ Hk_rot @ C3_inv
                C3sq_Hk = C3 @ C3 @ Hk_rot2 @ C3_inv @ C3_inv

                # ω² projection of H(k)
                Hw2 = (Hk + omega * C3Hk + omega**2 * C3sq_Hk) / 3.0

                # This is the C₃-breaking part of H at momentum k.
                # It transforms as ω² under C₃.

                # Track norm of H^{ω²}
                total_hw2_norm += la.norm(Hw2)**2

                # Resolvent in trivial sector
                G = la.inv(z_star * np.eye(n) - Hk)
                G_triv = P1 @ G @ P1

                # Second-order: ⟨ω²| H^{ω²} G_triv H^{ω²}† |ω⟩
                # Note: H^{ω²}† transforms as ω, which provides
                # the ω needed for the ⟨triv|...|ω⟩ transition.
                val = np.trace(Pw2 @ Hw2 @ G_triv @ Hw2.conj().T @ Pw)
                total_direct += val

    total_direct *= dk**3
    avg_hw2_norm = np.sqrt(total_hw2_norm * dk**3 / N**3)

    print(f"\n  z* = {z_star:.4f}")
    print(f"  Average |H^(ω²)| = {avg_hw2_norm:.6f}")
    print(f"  Raw integral: {total_direct}")
    print(f"  |integral| = {abs(total_direct):.10f}")

    # Scale by (α₁/h)² for the full Feshbach second-order amplitude
    V_us_feshbach = abs(total_direct) * abs(Sigma_scalar)**2
    print(f"\n  Scaled by |Σ|² = {abs(Sigma_scalar)**2:.6e}:")
    print(f"  V_us (Feshbach 2nd order) = {V_us_feshbach:.10f}")
    print(f"  V_us (target (2/3)^(2+√3)) = {(2/3)**(2+np.sqrt(3)):.10f}")
    print(f"  V_us (observed)            = 0.2250")

    # Try different z values
    print(f"\n  z-scan:")
    for z_try in [2.5, 17/6, 3.0, 3.05, 3.1, 3.5, 4.0, 5.0]:
        tot = 0.0
        for i1 in range(N):
            for i2 in range(N):
                for i3 in range(N):
                    k = np.array([i1+0.5, i2+0.5, i3+0.5]) / N
                    Hk = bloch_hamiltonian(k, bonds, n)
                    k_rot = Rmat @ k
                    Hk_rot = bloch_hamiltonian(k_rot, bonds, n)
                    k_rot2 = Rmat @ k_rot
                    Hk_rot2 = bloch_hamiltonian(k_rot2, bonds, n)
                    C3Hk = C3 @ Hk_rot @ C3_inv
                    C3sq_Hk = C3 @ C3 @ Hk_rot2 @ C3_inv @ C3_inv
                    Hw2 = (Hk + omega*C3Hk + omega**2*C3sq_Hk) / 3.0
                    G = la.inv(z_try * np.eye(n) - Hk)
                    G_triv = P1 @ G @ P1
                    val = np.trace(Pw2 @ Hw2 @ G_triv @ Hw2.conj().T @ Pw)
                    tot += val
        tot *= dk**3
        V = abs(tot) * abs(Sigma_scalar)**2
        print(f"    z={z_try:.4f}: |raw|={abs(tot):.6e}, V_us={V:.6e}")


if __name__ == '__main__':
    main()
