#!/usr/bin/env python3
"""
G_sub session 5 path #3: matter polarization Π^{ab,cd}(p) at finite external momentum.

Per `g_sub_session5_path1_finding.md`, the substrate's emergent Newton
constant is extracted from the next-to-leading p² coefficient of the
matter polarization:

  1/(16π G_sub) = Π_2  where  Π^{TT}(p²) = Π_0 + Π_2 p² + O(p⁴)

The static piece Π_0 ≈ 15/(8π²) was identified in path #2 as the
substrate's Goldstone phonon kinetic (= ρ_substrate). Path #3 (this script)
extracts Π_2.

Method
------
Generalize the static elastic-modulus computation of
`lorentz_sig_g_sub_elastic_moduli.py` to finite external momentum p:

  Π_para^{ab,cd}(p) = -(2/V_BZ) Re Σ_k Σ_{n filled at k, m unfilled at k+p}
       ⟨n,k| A^{ab}(k+p/2) |m,k+p⟩ ⟨m,k+p| A^{cd}(k+p/2) |n,k⟩
       / (ε_n(k) - ε_m(k+p))

  Π_dia^{ab,cd}(p) = (1/V_BZ) Σ_k Σ_{n filled} ⟨n,k| W^{abcd}(k) |n,k⟩
                     (diamagnetic; depends weakly on p at this order)

Note: the diamagnetic vertex W couples to u² (no derivative-of-u factors),
so its leading p-dependence comes from how the 2-point function of u²
differs from u·u at finite external p. At leading order in p², the dia
contribution is approximately p-independent, and Π_2 comes mainly from
the paramagnetic part.

Convention: external p is a 3-vector chosen as p_z ê_z (along z-axis).
TT polarization is e_×^{xy} (cross-shear). Compute Π^{xy,xy}(p_z) for
several small p_z values, fit the small-p expansion.

At p=0: this reduces to the static C^{xyxy} of `lorentz_sig_g_sub_elastic_moduli.py`.

Status
------
First-principles implementation of finite-p matter polarization. Extracts
Π_2 = leading p² slope. Path-#3 of session 5 — direct G_sub closure.
"""
from __future__ import annotations

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from lorentz_sig_g_sub_elastic_moduli import (
    BOND_DISPLACEMENTS, H_bloch, A_strain_matrix,
    W_diamagnetic_symmetrized,
)


def H_bloch_at(k_cart):
    return H_bloch(k_cart)


def A_at(k_cart, a, b):
    """Strain matrix A^{ab} symmetrized in (a,b)."""
    A_ab = A_strain_matrix(k_cart, a, b)
    A_ba = A_strain_matrix(k_cart, b, a)
    return (A_ab + A_ba) / 2


def Pi_para_at_k(k_cart, p_cart, mu=0.0, tol=1e-8):
    """
    Paramagnetic contribution at single k, external p.

    Returns: 4-tensor Π_para^{abcd}(k, p) (as 3×3×3×3 ndarray).
    """
    k_mid = k_cart + p_cart / 2  # symmetric vertex
    H_k = H_bloch_at(k_cart)
    H_kp = H_bloch_at(k_cart + p_cart)
    eigs_k, U_k = np.linalg.eigh(H_k)
    eigs_kp, U_kp = np.linalg.eigh(H_kp)

    filled_k = eigs_k < mu - tol
    unfilled_kp = eigs_kp > mu + tol

    # Compute 9 strain matrices A^{ac} at k_mid
    A_at_kmid = np.zeros((3, 3, 4, 4), dtype=complex)
    for a in range(3):
        for b in range(3):
            A_at_kmid[a, b] = A_at(k_mid, a, b)

    # Transform to (k, k+p) eigenbasis: A_basis[a,b][m,n] = ⟨m,k+p|A|n,k⟩
    A_basis = np.zeros((3, 3, 4, 4), dtype=complex)
    for a in range(3):
        for b in range(3):
            A_basis[a, b] = U_kp.conj().T @ A_at_kmid[a, b] @ U_k

    K = np.zeros((3, 3, 3, 3), dtype=float)
    for n in np.where(filled_k)[0]:
        for m in np.where(unfilled_kp)[0]:
            denom = eigs_k[n] - eigs_kp[m]
            if abs(denom) < tol:
                continue
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        for d in range(3):
                            term = A_basis[a, b][m, n].conj() * A_basis[c, d][m, n]
                            K[a, b, c, d] += -2.0 * (term / denom).real
    return K


def Pi_dia_at_k(k_cart, mu=0.0, tol=1e-8):
    """
    Diamagnetic contribution at single k. To leading order in external p,
    the diamagnetic vertex doesn't depend on p (it's a contact term in u²),
    so we use the same form as the static computation.
    """
    H_k = H_bloch_at(k_cart)
    eigs, U = np.linalg.eigh(H_k)
    filled = eigs < mu - tol

    K = np.zeros((3, 3, 3, 3), dtype=float)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    W = W_diamagnetic_symmetrized(k_cart, a, b, c, d)
                    W_basis = U.conj().T @ W @ U
                    for n in np.where(filled)[0]:
                        K[a, b, c, d] += np.real(W_basis[n, n])
    return K


def Pi_at_p(p_cart, N_grid=12, mu=0.0, half_extent=2*np.pi):
    """BZ-averaged Π^{ab,cd}(p) = K_dia(p) - K_para(p) (subtractive, per path-#1)."""
    ks = np.linspace(-half_extent, half_extent, N_grid, endpoint=False)
    K_para = np.zeros((3, 3, 3, 3))
    K_dia = np.zeros((3, 3, 3, 3))
    n_pts = 0
    for k1 in ks:
        for k2 in ks:
            for k3 in ks:
                k_cart = np.array([k1, k2, k3])
                K_para += Pi_para_at_k(k_cart, p_cart, mu)
                K_dia += Pi_dia_at_k(k_cart, mu)
                n_pts += 1
    return K_dia / n_pts, K_para / n_pts


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def main():
    header("G_sub session 5 path #3: Π^{ab,cd}(p) at finite external p")
    print()
    print("  Computes the matter polarization at finite external momentum p")
    print("  along z-axis (p = p_z ê_z). Probes Π^{xy,xy}(p_z) — the TT cross")
    print("  channel — for small p_z values to extract the slope Π_2.")
    print()

    N = 8  # smaller grid for finite-p computations (8 times more expensive)
    mu = 0.0
    half_extent = 2 * np.pi

    print(f"  Grid: {N}³ k-points; half-extent = 2π")
    print()

    # Probe several small p_z values
    p_z_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    results = []
    print(f"  {'p_z':>8s} {'Pi_xyxy_dia':>14s} {'Pi_xyxy_para':>14s} {'Pi_xyxy_sub':>14s} {'Pi_xyxy_sub/p²':>17s}")
    print("  " + "-" * 70)

    for p_z in p_z_values:
        p_cart = np.array([0.0, 0.0, p_z])
        K_dia, K_para = Pi_at_p(p_cart, N_grid=N, mu=mu, half_extent=half_extent)
        K_sub = K_dia - K_para
        Pi_dia_xyxy = K_dia[0, 1, 0, 1]
        Pi_para_xyxy = K_para[0, 1, 0, 1]
        Pi_sub_xyxy = K_sub[0, 1, 0, 1]
        if p_z > 0:
            ratio = Pi_sub_xyxy / p_z**2
        else:
            ratio = float('nan')
        results.append((p_z, Pi_dia_xyxy, Pi_para_xyxy, Pi_sub_xyxy, ratio))
        print(f"  {p_z:>8.4f} {Pi_dia_xyxy:>+14.6f} {Pi_para_xyxy:>+14.6f} {Pi_sub_xyxy:>+14.6f} {ratio:>17.6e}")

    header("Fit to Π(p²) = Π_0 + Π_2 p² + Π_4 p^4 + ...")
    print()
    p_arr = np.array([r[0] for r in results])
    Pi_arr = np.array([r[3] for r in results])

    # Fit polynomial Pi(p²) = Pi_0 + Pi_2 × p² + Pi_4 × p⁴
    A_mat = np.array([[1.0, p**2, p**4, p**6] for p in p_arr])
    coeffs, residuals, rank, sv = np.linalg.lstsq(A_mat, Pi_arr, rcond=None)
    Pi_0_fit, Pi_2_fit, Pi_4_fit, Pi_6_fit = coeffs

    print(f"  Pi_0 (= Pi_TT^xy,xy at p=0):  {Pi_0_fit:+.6f}")
    print(f"  Pi_2 (= leading p² slope):    {Pi_2_fit:+.6f}")
    print(f"  Pi_4 (next correction):       {Pi_4_fit:+.6f}")
    print(f"  Pi_6:                          {Pi_6_fit:+.6f}")

    # If 1/(16π G_sub) = Pi_2:
    G_candidate = 1.0 / (16 * np.pi * Pi_2_fit) if Pi_2_fit > 0 else None
    if G_candidate is not None:
        print()
        print(f"  Under 1/(16π G_sub) = Pi_2 hypothesis:")
        print(f"    G_sub = 1/(16π × {Pi_2_fit:.6f}) = {G_candidate:+.6f}")
    else:
        print()
        print(f"  Pi_2 < 0 — sign issue, need to check projection / convention")

    print()
    print(f"  Compare to session 4 candidate G_sub = 4(√3-1)/27 = {4*(np.sqrt(3)-1)/27:.6f}")
    print(f"  Compare to π/30 (path #1 wrong) = {np.pi/30:.6f}")

    header("Sanity check: Pi_xyxy at p=0 should match static C_44")
    print()
    print(f"  Pi_xyxy(p=0) sub: {results[0][3]:.6f}")
    print(f"  Static elastic C_44 (from earlier convergence study, 24³): ~ +0.274")
    print(f"  Note: at N={N}³ static C_44 differs slightly; expected match within grid noise")

    header("Pi(p²) data (for diagnosis)")
    print()
    print(f"  p_z   |  Pi_sub   |  ΔPi from p=0  |  ΔPi/p²")
    Pi_at_0 = results[0][3]
    for r in results[1:]:
        p_z, _, _, Pi_sub, ratio = r
        delta = Pi_sub - Pi_at_0
        print(f"  {p_z:.4f} |  {Pi_sub:+.6f}  |  {delta:+.6e}  |  {delta/p_z**2:+.6e}")


if __name__ == "__main__":
    main()
