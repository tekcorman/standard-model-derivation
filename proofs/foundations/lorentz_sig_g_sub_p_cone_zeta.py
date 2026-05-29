#!/usr/bin/env python3
"""
G_sub session 5 path #4: ζ_P from 2-band P-cone matter loop.

Tests session 4's universal-ζ assumption: is ζ_Γ = ζ_P at sphere Λ=π?
If yes, structural form 4(√3-1)/27 = (k*-1)²(√k*-1)/k*³ for G_sub stands.
If no, multi-valley sum gives a different value.

Method
------
Cone-effective Sakharov-style computation for the P-cone (2-band Weyl in
3+1D), analogous to `lorentz_sig_g_sub_flat_band_loop_v2.py` for the
spin-1 Γ-cone.

P-cone effective Hamiltonian (per `predictions/srs_dirac_cone_velocities.py`):
  H_P_eff = ε_0 + v_F^P (q · σ)
where σ are Pauli matrices, v_F^P = √3/6.

Bands: ε_± = ε_0 ± v_F^P |q|. At half-filling within the cone, lower
band filled, upper empty.

Strain coupling (Iorio-elastic): δH = v_F^P (∂^a u_b) q^b σ^a.
Strain vertex: V^{ab}(q) = v_F^P q^b σ^a.

Matter polarization at finite p:
  Π^{ab,cd}(p) = ∫ d³q [v_F^P q^b ⟨-,q|σ^a|+,q+p⟩ × v_F^P (q+p)^d ⟨+,q+p|σ^c|-,q⟩]
                 × [f_-(q) - f_+(q+p)] / [ε_-(q) - ε_+(q+p)]

ζ_P extraction: Π_TT(p²) = a_0 + a_2 p² + ..., then
  ζ_P = a_2 × v_F^P / Λ²  (in session 4's normalization)

Compare to ζ_Γ ≈ 1.70e-3 at sphere Λ=π (session 4 universal value).
"""
from __future__ import annotations

import numpy as np


# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma = np.array([sigma_x, sigma_y, sigma_z])  # shape (3, 2, 2)


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def helicity_basis_2band(qhats):
    """For each unit qhat, eigenstates of (qhat · σ).
    Returns eigvals (...,2) sorted descending (+1, -1) and eigvecs (...,2,2)
    with columns being the eigenvectors.
    """
    qS = np.einsum('...a,abc->...bc', qhats, sigma)
    eigvals, eigvecs = np.linalg.eigh(qS)
    idx = np.argsort(-eigvals.real, axis=-1)
    eigvals = np.take_along_axis(eigvals, idx, axis=-1)
    eigvecs = np.take_along_axis(eigvecs, idx[..., np.newaxis, :], axis=-1)
    return eigvals.real, eigvecs


def fermi_T0_batch(E):
    out = np.zeros_like(E)
    out[E < -1e-12] = 1.0
    out[np.abs(E) <= 1e-12] = 0.5
    out[E > 1e-12] = 0.0
    return out


def compute_Pi_2band(p, Lambda=np.pi, v_F=np.sqrt(3)/6, n_radial=40, n_theta=20, n_phi=20):
    """Compute Π^{ab,cd}(p) for 2-band cone with given v_F, sphere Λ.

    Returns (3,3,3,3) complex array.
    """
    # Spherical-grid sampling (sphere of radius Λ)
    q_radial = np.linspace(Lambda / n_radial / 2, Lambda - Lambda / n_radial / 2, n_radial)
    dq = Lambda / n_radial
    theta_grid = np.linspace(np.pi / n_theta / 2, np.pi - np.pi / n_theta / 2, n_theta)
    dtheta = np.pi / n_theta
    phi_grid = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi

    Q_R, Q_T, Q_P = np.meshgrid(q_radial, theta_grid, phi_grid, indexing='ij')
    q_mag = Q_R.flatten()
    sin_theta = np.sin(Q_T.flatten())
    cos_theta = np.cos(Q_T.flatten())
    sin_phi = np.sin(Q_P.flatten())
    cos_phi = np.cos(Q_P.flatten())

    qx = q_mag * sin_theta * cos_phi
    qy = q_mag * sin_theta * sin_phi
    qz = q_mag * cos_theta
    q_vec = np.stack([qx, qy, qz], axis=-1)
    qp_vec = q_vec + p
    qp_mag = np.linalg.norm(qp_vec, axis=-1)

    valid = qp_mag > 1e-12
    q_vec = q_vec[valid]; qp_vec = qp_vec[valid]
    q_mag = q_mag[valid]; qp_mag = qp_mag[valid]
    sin_theta = sin_theta[valid]

    qhat = q_vec / q_mag[:, None]
    qphat = qp_vec / qp_mag[:, None]

    eigvals_q, vecs_q = helicity_basis_2band(qhat)    # (N, 2), (N, 2, 2)
    eigvals_qp, vecs_qp = helicity_basis_2band(qphat)

    vol_per_pt = q_mag ** 2 * sin_theta * dq * dtheta * dphi / (2 * np.pi) ** 3

    # Energies: ε_± = ± v_F |q| (band index 0 = +1 helicity, 1 = -1 helicity)
    E_q = v_F * eigvals_q * q_mag[:, None]      # (N, 2)
    E_qp = v_F * eigvals_qp * qp_mag[:, None]   # (N, 2)

    # Single channel: lower band (h=-1, idx=1) at q, upper band (h=+1, idx=0) at q+p
    # And reverse: lower at q+p, upper at q. Both contribute (sign of denominator differs).
    Pi = np.zeros((3, 3, 3, 3), dtype=complex)
    for (h, hp) in [(1, 0), (0, 1)]:  # (filled@q, empty@q+p) and reverse
        h_state = vecs_q[:, :, h]
        hp_state = vecs_qp[:, :, hp]
        E_h = E_q[:, h]
        E_hp = E_qp[:, hp]
        f_h = fermi_T0_batch(E_h)
        f_hp = fermi_T0_batch(E_hp)
        diff = f_h - f_hp
        denom = E_h - E_hp

        active = (np.abs(diff) > 1e-12) & (np.abs(denom) > 1e-12)
        if not np.any(active):
            continue

        h_state_a = h_state[active]
        hp_state_a = hp_state[active]
        q_a = q_vec[active]
        qp_a = qp_vec[active]
        diff_a = diff[active]
        denom_a = denom[active]
        vol_a = vol_per_pt[active]

        # Matrix elements ⟨h,q̂|σ^b|h',q̂'⟩ and reverse
        ME_h_to_hp = np.einsum('nA,bAB,nB->nb', h_state_a.conj(), sigma, hp_state_a)
        ME_hp_to_h = np.einsum('nA,dAB,nB->nd', hp_state_a.conj(), sigma, h_state_a)

        # Coefficient: v_F² (two vertices, each contributes v_F)
        coeff = (v_F ** 2) * vol_a * diff_a / denom_a

        Pi += np.einsum('n,na,nc,nb,nd->abcd', coeff, q_a, qp_a, ME_h_to_hp, ME_hp_to_h)

    return Pi


def TT_project_zhat(Pi):
    """Extract Π_TT for p along ẑ."""
    Pi_xxxx = Pi[0, 0, 0, 0]
    Pi_xxyy = Pi[0, 0, 1, 1]
    Pi_yyyy = Pi[1, 1, 1, 1]
    Pi_xyxy = Pi[0, 1, 0, 1]
    return ((Pi_xxxx - 2 * Pi_xxyy + Pi_yyyy) / 4 + Pi_xyxy).real


def main():
    header("G_sub session 5 path #4: ζ_P for 2-band P-cone")
    print()
    v_F_P = np.sqrt(3) / 6
    Lambda = np.pi
    print(f"  v_F^P = √3/6 = {v_F_P:.6f}")
    print(f"  Λ = π = {Lambda:.6f}")
    print()

    # Compute Pi_TT(p²) at several p_z values
    p_values = [0.0, 0.05, 0.1, 0.15, 0.2]
    Pi_TT_list = []

    print(f"  Sphere grid (40, 20, 20) = 16000 points")
    print(f"  {'p_z':>6s}  {'Pi_TT':>15s}")
    for p_z in p_values:
        p = np.array([0.0, 0.0, p_z])
        Pi = compute_Pi_2band(p, Lambda=Lambda, v_F=v_F_P,
                                n_radial=40, n_theta=20, n_phi=20)
        Pi_TT = TT_project_zhat(Pi)
        Pi_TT_list.append(Pi_TT)
        print(f"  {p_z:>6.4f}  {Pi_TT:>15.8e}")

    p_arr = np.array(p_values)
    Pi_arr = np.array(Pi_TT_list)
    # Quadratic fit in p²: Pi(p²) = a_0 + a_2 p² + a_4 p^4
    coeffs = np.polyfit(p_arr ** 2, Pi_arr, 2)
    a_4, a_2, a_0 = coeffs
    print()
    print(f"  Fit: Π_TT(p²) = a_0 + a_2 p² + a_4 (p²)²")
    print(f"    a_0 = {a_0:.8e}  (static elastic modulus)")
    print(f"    a_2 = {a_2:.8e}  (graviton kinetic coefficient)")
    print(f"    a_4 = {a_4:.8e}")
    print()

    # ζ_P = a_2 × v_F / Λ² (session 4's normalization)
    zeta_P = a_2 * v_F_P / Lambda ** 2
    zeta_universal = 27 / (512 * np.pi ** 3)
    print(f"  ζ_P = a_2 × v_F^P / Λ² = {zeta_P:.6e}")
    print(f"  ζ_universal (session 4, sphere Λ=π) = 27/(512π³) = {zeta_universal:.6e}")
    print(f"  Ratio ζ_P / ζ_universal = {zeta_P / zeta_universal:.4f}")
    print()

    if abs(zeta_P / zeta_universal - 1) < 0.05:
        print("  → ζ_P ≈ ζ_universal: universal-ζ assumption HOLDS at sphere Λ=π.")
        print("    Session 4's structural form 4(√3-1)/27 stands (within sphere convention).")
    else:
        print("  → ζ_P ≠ ζ_universal: universal-ζ assumption FAILS.")
        print("    Multi-valley sum needs revision.")
        print()
        # Compute corrected G_sub
        zeta_Gamma = 27 / (512 * np.pi ** 3)
        zeta_H = zeta_Gamma  # PH-conjugate
        v_F_Gamma = 1/2
        v_F_H = 1/2

        inv_16piG = (zeta_Gamma * Lambda**2 / v_F_Gamma +
                      zeta_H * Lambda**2 / v_F_H +
                      2 * zeta_P * Lambda**2 / v_F_P)
        G_total = 1.0 / (16 * np.pi * inv_16piG)
        print(f"  Corrected multi-valley G_sub^total:")
        print(f"    1/(16π G) = {inv_16piG:.6e}")
        print(f"    G_sub = {G_total:.6e}")
        print(f"    Compare session 4: 4(√3-1)/27 = {4*(np.sqrt(3)-1)/27:.6f}")
        print(f"    Compare path-#3 numerics: G_sub ≈ 0.107")


if __name__ == "__main__":
    main()
