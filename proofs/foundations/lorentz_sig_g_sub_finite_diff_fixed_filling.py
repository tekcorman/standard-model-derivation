#!/usr/bin/env python3
"""
G_sub session 5 — refined finite-difference probe with FIXED FILLING.

The naive finite-difference of E_GS(u) = Σ_{eigvals(H[u]) < μ} eigval picks
up a large Fermi-surface (metallic instability) piece that doesn't appear
in the linear-response K_para/K_dia tensors of
`lorentz_sig_g_sub_elastic_moduli.py`. To compare apples-to-apples with the
linear response, we need to track each band CONTINUOUSLY in u and sum only
those bands that are filled at u=0 (regardless of whether they cross μ at
finite u).

This isolates the INTERBAND contribution — the same quantity computed by
the linear-response Kubo-style sum. Its second derivative settles the
K_dia ± K_para sign question.

Method
------
For each k:
  1. Diagonalize H[u=0] — get eigvecs U_0, eigvals λ_0.
  2. For finite u: build H[u], find eigenvectors U_u with maximum overlap
     with U_0 (continuous tracking via permutation matching).
  3. Compute Σ_{n: λ_0(n) < μ} eigval_u_at_position_n.
  4. Finite-difference 2nd derivative.

This is the FIXED-FILLING ∂²E_GS/∂u², which matches the linear-response
sum at second order regardless of metallic/insulator phase.
"""
from __future__ import annotations

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from lorentz_sig_g_sub_finite_diff_elastic import (
    H_bloch_strained, BOND_DISPLACEMENTS,
)


def fixed_filling_energy_at_k(k_cart, E, U_0, lambda_0, mu=0.0):
    """Sum of eigenvalues of H[E](k) at the BAND POSITIONS that were filled at E=0.

    Band tracking via maximum-overlap permutation between U_0 and U_E.
    """
    H = H_bloch_strained(k_cart, E)
    eigvals, eigvecs = np.linalg.eigh(H)
    # Match bands: for each band of U_0, find best-overlap band in eigvecs.
    overlap = np.abs(U_0.conj().T @ eigvecs)**2  # (n_bands_0, n_bands_E)
    # Greedy assignment: largest overlap first.
    n = overlap.shape[0]
    assigned = -np.ones(n, dtype=int)
    used = np.zeros(n, dtype=bool)
    flat_indices = np.argsort(-overlap.ravel())
    for idx in flat_indices:
        i, j = idx // n, idx % n
        if assigned[i] == -1 and not used[j]:
            assigned[i] = j
            used[j] = True
    # For each band-position n_0 with lambda_0[n_0] < mu (filled at u=0),
    # add eigvals[assigned[n_0]] (the corresponding band's energy at finite u).
    energy = 0.0
    for n_0 in range(len(lambda_0)):
        if lambda_0[n_0] < mu:
            energy += eigvals[assigned[n_0]]
    return energy


def bz_average_fixed_filling(E, N_grid=12, half_extent=2*np.pi, mu=0.0):
    """BZ-average of fixed-filling energy."""
    ks = np.linspace(-half_extent, half_extent, N_grid, endpoint=False)
    total = 0.0
    n_pts = 0
    for k1 in ks:
        for k2 in ks:
            for k3 in ks:
                k_cart = np.array([k1, k2, k3])
                # Get reference U_0, lambda_0 at u=0.
                H0 = H_bloch_strained(k_cart, np.zeros((3,3)))
                lambda_0, U_0 = np.linalg.eigh(H0)
                lambda_0 = lambda_0.real
                total += fixed_filling_energy_at_k(k_cart, E, U_0, lambda_0, mu)
                n_pts += 1
    return total / n_pts


def fd_2nd_deriv_fixed(coord_pair, N_grid=12, h=1e-3, half_extent=2*np.pi):
    (a, b), (c, d) = coord_pair

    def make_E(*entries):
        E = np.zeros((3, 3))
        for i, j, val in entries:
            E[i, j] = val
        return E

    if (a, b) == (c, d):
        f_p = bz_average_fixed_filling(make_E((a,b,+h)), N_grid, half_extent)
        f_0 = bz_average_fixed_filling(np.zeros((3,3)), N_grid, half_extent)
        f_m = bz_average_fixed_filling(make_E((a,b,-h)), N_grid, half_extent)
        return (f_p - 2*f_0 + f_m) / h**2

    f_pp = bz_average_fixed_filling(make_E((a,b,+h),(c,d,+h)), N_grid, half_extent)
    f_pm = bz_average_fixed_filling(make_E((a,b,+h),(c,d,-h)), N_grid, half_extent)
    f_mp = bz_average_fixed_filling(make_E((a,b,-h),(c,d,+h)), N_grid, half_extent)
    f_mm = bz_average_fixed_filling(make_E((a,b,-h),(c,d,-h)), N_grid, half_extent)
    return (f_pp - f_pm - f_mp + f_mm) / (4 * h**2)


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def voigt_iso_mu(C_11, C_12, C_44):
    return (C_11 - C_12 + 3 * C_44) / 5


def main():
    header("G_sub session 5: FIXED-FILLING finite-difference probe")
    print()
    print("  Tracks each band continuously in u via overlap matching;")
    print("  sums only bands filled at u=0. Matches linear-response interband")
    print("  contribution (no Fermi-surface rearrangement).")
    print()

    N = 12
    half_extent = 2 * np.pi
    h = 1e-3

    print(f"  Grid: {N}³ k-points; half-extent = 2π; strain step h = {h}")

    diagonal = {}
    for (a, b), label in [((0,0), 'xx'), ((1,1), 'yy'), ((2,2), 'zz'),
                          ((0,1), 'xy'), ((1,0), 'yx')]:
        d2 = fd_2nd_deriv_fixed(((a,b),(a,b)), N_grid=N, h=h, half_extent=half_extent)
        diagonal[(a, b)] = d2
        print(f"    ∂²E_GS/(∂E^{{{label}}})²  (fixed filling)  = {d2:+.6f}")

    # Mixed
    C_xx_yy = fd_2nd_deriv_fixed(((0,0),(1,1)), N_grid=N, h=h, half_extent=half_extent)
    C_xy_yx = fd_2nd_deriv_fixed(((0,1),(1,0)), N_grid=N, h=h, half_extent=half_extent)
    print(f"    ∂²E_GS/∂E^xx ∂E^yy  (fixed filling)            = {C_xx_yy:+.6f}")
    print(f"    ∂²E_GS/∂E^xy ∂E^yx  (fixed filling)            = {C_xy_yx:+.6f}")

    # Symmetric shear: E^xy = E^yx = u → ε^xy = u, ω = 0
    f_zero = bz_average_fixed_filling(np.zeros((3,3)), N, half_extent)
    E_p = np.zeros((3,3)); E_p[0,1]=E_p[1,0]=+h
    E_m = np.zeros((3,3)); E_m[0,1]=E_m[1,0]=-h
    f_p = bz_average_fixed_filling(E_p, N, half_extent)
    f_m = bz_average_fixed_filling(E_m, N, half_extent)
    sym_xy = (f_p - 2*f_zero + f_m) / h**2
    # Antisymmetric: E^xy=-E^yx=u → ε=0, ω=u
    E_p = np.zeros((3,3)); E_p[0,1]=+h; E_p[1,0]=-h
    E_m = np.zeros((3,3)); E_m[0,1]=-h; E_m[1,0]=+h
    f_p = bz_average_fixed_filling(E_p, N, half_extent)
    f_m = bz_average_fixed_filling(E_m, N, half_extent)
    asym_xy = (f_p - 2*f_zero + f_m) / h**2
    print(f"    Symmetric shear (E^xy=E^yx=u):  ∂²E/∂u² = {sym_xy:+.6f}")
    print(f"    Antisymmetric (E^xy=-E^yx=u):   ∂²E/∂u² = {asym_xy:+.6f}")
    print(f"    Decomposition: a + 2b + c = {2*diagonal[(0,1)] + 2*C_xy_yx:+.6f}  (sym predicted)")
    print(f"                   a - 2b + c = {2*diagonal[(0,1)] - 2*C_xy_yx:+.6f}  (antisym predicted)")

    # Cubic Voigt averaging
    C_11_fd = diagonal[(0,0)]  # x-x stretch (cubic-invariant)
    C_12_fd = C_xx_yy
    C_44_fd = sym_xy / 4  # since sym shear gives 4 C^{xyxy} energy

    mu_iso_fd = voigt_iso_mu(C_11_fd, C_12_fd, C_44_fd)
    print()
    print(f"    C_11 = {C_11_fd:+.6f}")
    print(f"    C_12 = {C_12_fd:+.6f}")
    print(f"    C_44 = {C_44_fd:+.6f}")
    print(f"    μ_iso (Voigt) = {mu_iso_fd:+.6f}")

    header("Comparison to linear-response (perturbative)")
    print()
    from lorentz_sig_g_sub_elastic_moduli import bz_average_full, voigt_components
    K_para, K_dia, _ = bz_average_full(N_grid=N, mu=0.0, half_extent=half_extent)
    v_para = voigt_components(K_para)
    v_dia = voigt_components(K_dia)
    K_sub = K_dia - K_para
    K_add = K_dia + K_para
    v_sub = voigt_components(K_sub)
    v_add = voigt_components(K_add)
    mu_sub = voigt_iso_mu(v_sub['C_11'], v_sub['C_12'], v_sub['C_44'])
    mu_add = voigt_iso_mu(v_add['C_11'], v_add['C_12'], v_add['C_44'])

    print(f"  Fixed-filling FD       μ_iso       = {mu_iso_fd:+.6f}")
    print(f"  Linear-response K_dia + K_para     = {mu_add:+.6f}")
    print(f"  Linear-response K_dia - K_para     = {mu_sub:+.6f}")
    print()
    err_add = abs(mu_iso_fd - mu_add)
    err_sub = abs(mu_iso_fd - mu_sub)
    print(f"  |FD - additive|     = {err_add:.6f}")
    print(f"  |FD - subtractive|  = {err_sub:.6f}")
    print()

    if err_sub < err_add:
        print(f"  → SUBTRACTIVE wins: K_dia - K_para is the right interband elastic.")
        print(f"    μ_iso ≈ {mu_iso_fd:.6f} is the substrate's interband shear modulus.")
        print(f"    Compare 15/(8π²) = {15/(8*np.pi**2):.6f}, ratio = {mu_iso_fd / (15/(8*np.pi**2)):.6f}")
    else:
        print(f"  → ADDITIVE wins: K_dia + K_para is the right interband elastic.")
        print(f"    μ_iso ≈ {mu_iso_fd:.6f}.")


if __name__ == "__main__":
    main()
