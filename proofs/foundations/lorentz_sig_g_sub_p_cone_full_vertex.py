#!/usr/bin/env python3
"""
G_sub session 5 path #6: P-cone matter loop with FULL strain vertex (V_0 + V_1).

Per `g_sub_session5_path4_finding.md` and the path-#5 finding that the
P-cone strain vertex has a non-zero constant piece V_0 = A^{ac}(P)|_proj
(missed by naive Iorio extension), this script computes ζ_P with the full
strain vertex from numerical projection.

Method
------
1. At each numerical δk in the P-cone region:
   - Diagonalize H(P+δk) numerically.
   - Project the 2-dim eigensubspace at energies near +√3.
   - Compute V^{ac}(P+δk) = U_P+δk^† A^{ac}(P+δk) U_P+δk.
2. Compute the matter loop with this FULL vertex (no expansion in δk).
3. Extract ζ_P from Π_TT(p²) at small p.

This is computationally heavier than the cone-effective approximation but
captures the full vertex structure including V_0 piece.
"""
from __future__ import annotations

import numpy as np
from itertools import product


# Bond list (matching srs_dirac_cone_velocities.py + path-#4)
ATOMS = np.array([
    [1/8, 1/8, 1/8],
    [3/8, 7/8, 5/8],
    [7/8, 5/8, 3/8],
    [5/8, 3/8, 7/8],
])

A_PRIM = np.array([
    [-1/2,  1/2,  1/2],
    [ 1/2, -1/2,  1/2],
    [ 1/2,  1/2, -1/2],
])

CELL_EDGES = [
    (0, 1, (1, 1, 1)),
    (0, 2, (1, 1, 1)),
    (0, 3, (1, 1, 1)),
    (1, 2, (-1, 0, 0)),
    (1, 3, (0, 1, 0)),
    (2, 3, (0, 0, -1)),
]

DIRECTED_BONDS = []
for s, t, c in CELL_EDGES:
    DIRECTED_BONDS.append((s, t, np.array(c)))
    DIRECTED_BONDS.append((t, s, -np.array(c)))


def bond_displacement(src, tgt, cell):
    return ATOMS[tgt] - ATOMS[src] + cell @ A_PRIM


BOND_DISPLACEMENTS = [
    (s, t, bond_displacement(s, t, c)) for s, t, c in DIRECTED_BONDS
]


def H_bloch(k_cart):
    H = np.zeros((4, 4), dtype=complex)
    for s, t, rb in BOND_DISPLACEMENTS:
        phase = np.exp(1j * np.dot(k_cart, rb))
        H[t, s] += phase
    return (H + H.conj().T) / 2


def A_strain_full(k_cart, a, c):
    """Strain perturbation matrix A^{ac}(k) at Cartesian k (4x4)."""
    A_mat = np.zeros((4, 4), dtype=complex)
    for s, t, rb in BOND_DISPLACEMENTS:
        phase = np.exp(1j * np.dot(k_cart, rb))
        A_mat[t, s] += 1j * phase * k_cart[a] * rb[c]
    return (A_mat + A_mat.conj().T) / 2


def project_to_p_cone(k_cart, target_ev=np.sqrt(3), tol=0.5):
    """Find the 2-dim eigensubspace at energy near target_ev. Returns U_P (4×2)."""
    H = H_bloch(k_cart)
    eigvals, eigvecs = np.linalg.eigh(H)
    # Find 2 bands closest to target
    distances = np.abs(eigvals - target_ev)
    idx = np.argsort(distances)[:2]
    return eigvecs[:, idx], eigvals[idx]


def Pi_p_cone_at_external_p(p_cart, P_cart, n_radial=15, n_theta=10, n_phi=10,
                              Lambda_cone=1.0, target_ev=np.sqrt(3)):
    """Compute Pi^{ab,cd}(external p) for P-cone with full vertex.

    Spherical sampling of cone-relative momentum δk in sphere of radius Lambda_cone
    around P_cart.
    """
    q_radial = np.linspace(Lambda_cone / n_radial / 2, Lambda_cone - Lambda_cone / n_radial / 2, n_radial)
    dq = Lambda_cone / n_radial
    theta_grid = np.linspace(np.pi / n_theta / 2, np.pi - np.pi / n_theta / 2, n_theta)
    dtheta = np.pi / n_theta
    phi_grid = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi

    Pi = np.zeros((3, 3, 3, 3), dtype=complex)
    n_pts = 0
    for q_mag in q_radial:
        for theta in theta_grid:
            for phi in phi_grid:
                # Cone-relative momentum δk
                dk = q_mag * np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ])
                k_cart = P_cart + dk
                kp_cart = k_cart + p_cart  # external p shift
                # If kp_cart is too far from P, skip (cone-effective region)
                # No: we include all of cone-effective sphere.

                # Diagonalize at k and k+p
                H_k = H_bloch(k_cart)
                H_kp = H_bloch(kp_cart)
                eigs_k, U_k = np.linalg.eigh(H_k)
                eigs_kp, U_kp = np.linalg.eigh(H_kp)

                # Find 2-dim subspace near target_ev at each
                d_k = np.abs(eigs_k - target_ev)
                d_kp = np.abs(eigs_kp - target_ev)
                # P-cone has TWO bands at +√3 (near, splitting by v_F δk).
                # The cone has 2 bands. At energies near +√3 +- v_F |δk|.
                # For 2-band cone: filled = lower of the 2; empty = upper.

                idx_k = np.argsort(d_k)[:2]
                idx_kp = np.argsort(d_kp)[:2]

                eigs_cone_k = eigs_k[idx_k]
                eigs_cone_kp = eigs_kp[idx_kp]

                # Within the 2-band cone manifold, define μ_cone = +√3 (mid-cone)
                # so lower band filled, upper band empty.
                mu_cone = target_ev

                # Compute strain matrices A^{ab}(k) and A^{cd}(k+p) (use symmetric vertex)
                k_mid = (k_cart + kp_cart) / 2
                # Actually use A at k_mid for both vertices (symmetric convention)
                A_kmid = np.zeros((3, 3, 4, 4), dtype=complex)
                for a in range(3):
                    for b in range(3):
                        # symmetric in (a,b) for strain coupling
                        A_ab = A_strain_full(k_mid, a, b)
                        A_ba = A_strain_full(k_mid, b, a)
                        A_kmid[a, b] = (A_ab + A_ba) / 2

                # Project to cone subspace
                # ⟨n,k|A|m,k+p⟩ = U_k[:,n].conj() @ A @ U_kp[:,m]
                vol = q_mag**2 * np.sin(theta) * dq * dtheta * dphi / (2 * np.pi)**3

                for n in idx_k:
                    for m in idx_kp:
                        E_n = eigs_k[n]
                        E_m = eigs_kp[m]
                        # Filled at k means E_n < mu_cone; empty at k+p means E_m > mu_cone
                        f_n = 1.0 if E_n < mu_cone - 1e-10 else (0.5 if abs(E_n - mu_cone) < 1e-10 else 0.0)
                        f_m = 1.0 if E_m < mu_cone - 1e-10 else (0.5 if abs(E_m - mu_cone) < 1e-10 else 0.0)
                        diff = f_n - f_m
                        denom = E_n - E_m
                        if abs(diff) < 1e-12 or abs(denom) < 1e-12:
                            continue

                        # Matrix elements
                        psi_n = U_k[:, n]
                        psi_m = U_kp[:, m]
                        for a in range(3):
                            for b in range(3):
                                ME_nm = (psi_n.conj() @ A_kmid[a, b] @ psi_m)
                                for c in range(3):
                                    for d in range(3):
                                        ME_mn = (psi_m.conj() @ A_kmid[c, d] @ psi_n)
                                        Pi[a, b, c, d] += -2.0 * (ME_nm * ME_mn / denom).real * vol
                n_pts += 1
    return Pi


def TT_project_zhat(Pi):
    """Pi_TT for p along z."""
    Pi_xxxx = Pi[0, 0, 0, 0]
    Pi_xxyy = Pi[0, 0, 1, 1]
    Pi_yyyy = Pi[1, 1, 1, 1]
    Pi_xyxy = Pi[0, 1, 0, 1]
    return ((Pi_xxxx - 2 * Pi_xxyy + Pi_yyyy) / 4 + Pi_xyxy).real


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def main():
    header("G_sub session 5 path #6: P-cone matter loop with FULL strain vertex")
    print()

    P_cart = np.pi * np.array([1.0, 1.0, 1.0])  # P at fractional (1/4,1/4,1/4) → π(1,1,1)
    Lambda_cone = 0.5  # cone-effective sphere radius around P_cart
    print(f"  P_cart = π(1,1,1) = ({P_cart[0]:.4f}, {P_cart[1]:.4f}, {P_cart[2]:.4f})")
    print(f"  Cone-effective sphere radius Λ_cone = {Lambda_cone}")
    print()

    p_z_values = [0.0, 0.05, 0.1, 0.15, 0.2]
    Pi_TT_list = []
    print(f"  {'p_z':>6s}  {'Pi_TT (full vertex)':>22s}")
    for p_z in p_z_values:
        p_cart = np.array([0.0, 0.0, p_z])
        Pi = Pi_p_cone_at_external_p(p_cart, P_cart,
                                      n_radial=15, n_theta=10, n_phi=10,
                                      Lambda_cone=Lambda_cone)
        Pi_TT = TT_project_zhat(Pi)
        Pi_TT_list.append(Pi_TT)
        print(f"  {p_z:>6.4f}  {Pi_TT:>22.6e}")

    p_arr = np.array(p_z_values)
    Pi_arr = np.array(Pi_TT_list)
    coeffs = np.polyfit(p_arr ** 2, Pi_arr, 2)
    a_4, a_2, a_0 = coeffs
    print()
    print(f"  Fit Π_TT(p²) = a_0 + a_2 p² + a_4 (p²)²:")
    print(f"    a_0 = {a_0:.6e}  (static elastic)")
    print(f"    a_2 = {a_2:.6e}  (graviton kinetic; Π_2 = 1/(16π G))")
    print(f"    a_4 = {a_4:.6e}")
    print()

    # ζ_P with full vertex (using v_F = √3/6 for P-cone, Λ²=Λ_cone²)
    v_F_P = np.sqrt(3) / 6
    zeta_P_full = a_2 * v_F_P / Lambda_cone**2
    zeta_P_naive = 2.587e-4  # from path-#4 at sphere Λ=π
    zeta_Gamma = 27 / (512 * np.pi ** 3)

    print(f"  Effective ζ_P (full vertex, Λ_cone={Lambda_cone}):")
    print(f"    ζ_P = a_2 × v_F / Λ_cone² = {zeta_P_full:.6e}")
    print(f"  Comparison:")
    print(f"    ζ_P_naive (path-#4, sphere Λ=π) = {zeta_P_naive:.6e}")
    print(f"    ζ_Γ universal (sphere Λ=π)        = {zeta_Gamma:.6e}")
    print(f"  Ratio ζ_P_full/ζ_P_naive: {zeta_P_full / zeta_P_naive:.4f}  (Λ-rescaling complicates direct)")
    print(f"  Ratio ζ_P_full/ζ_Γ: {zeta_P_full / zeta_Gamma:.4f}")

    print()
    print("  Note: this uses cone-effective sphere Λ_cone (not the BZ-edge Λ=π).")
    print("  At Λ_cone = 0.5: cone-region within ~half lattice unit of P.")
    print("  Comparison to session 4's universal-ζ assumes same convention; rescale by Λ²")

    # If Lambda_cone is the relevant cutoff (cone region), and a_2 is the coefficient of p²:
    # 1/(16π G_P) = a_2  (in path-#4's convention where ζ × Λ²/v_F = a_2 × Λ²/v_F means
    #                     1/(16π G_P) = a_2 × Λ²/v_F = ζ × Λ²/v_F if ζ = a_2)
    # Wait, let me think more carefully.
    # In session 4 form: 1/(16π G_cone) = ζ × Λ²/v_F.
    # ζ has units of (length²) × (1/length) × (energy²) = ... it's dimensionless if Λ has length^-1 units.
    # With our conventions: a_2 = (lattice unit)^{-2} × ... actually a_2 is dimensionless.
    print()
    print(f"  Direct: 1/(16π G_P_full) = {a_2:.6e} (if a_2 is the kinetic coefficient)")
    if a_2 > 0:
        G_P_full = 1.0 / (16 * np.pi * a_2)
        print(f"  G_P (this cone alone, full vertex) = {G_P_full:.6e}")


if __name__ == "__main__":
    main()
