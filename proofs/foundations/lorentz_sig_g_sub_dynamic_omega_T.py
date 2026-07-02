#!/usr/bin/env python3
"""
G_sub Phase 5: dynamical Π_TT(p, ω_E) with both finite ω_E AND finite T smearing.

Phase 4 (`lorentz_sig_g_sub_dynamic_finite_omega.py`) established that
the finite-ω Kubo prescription is structurally correct, but found grid-
resonance issues: N=12 converged cleanly to G_sub = 1/(4π) within 0.6%,
while N=10 and N=14 diverged due to grid alignment with band-crossing
surfaces (where the sharp T=0 Fermi step changes by ±1 across a single
BZ grid cell).

This script adds finite-T Fermi smearing: replace the sharp T=0 step with
the smooth Fermi-Dirac distribution f_β(E) = 1/(1 + exp(β(E-μ))). With
both ω_E (Drude regulator) AND T (Fermi-step smear) finite, the integrand
becomes smooth across BZ. Uniform grids should then converge.

Both regulators are taken to zero in the OUTER limit AFTER the BZ integral.

Method
------
1. Implement Lindhard at finite (ω_E, T) using smooth Fermi factors.
2. Choose T compatible with ω_E (T ~ ω_E so the Fermi-step smear width
   matches the Lindhard regulator width).
3. Run grid scan (N=10, 12, 14, 16) at fixed (ω_E, T). If a_2 is now
   N-independent: confirms the smearing fix. The ω_E → 0, T → 0 limit
   gives the proper graviton kinetic.
4. If a_2 converges across grids to a clean K[1/π] value: closure.

Phase 4 N=12 result was a_2 ≈ 0.247 (Π_TT convention) / 0.495 (my code's
2× convention), suggesting G_sub = 1/(4π) to 0.6%. Phase 5 verifies.
"""
from __future__ import annotations

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
from lorentz_sig_g_sub_pi_finite_p import H_bloch_at, A_at


def fermi_smooth(E, mu=0.0, T=0.05):
    """Smooth Fermi-Dirac f(E) = 1/(1 + exp((E-μ)/T)). Stable for large argument."""
    arg = (E - mu) / T
    # Stable computation for large |arg|
    return np.where(
        arg > 0,
        np.exp(-arg) / (1 + np.exp(-arg)),
        1 / (1 + np.exp(arg)),
    )


def Pi_at_k_omega_T(k_cart, p_cart, omega_E, T, mu=0.0):
    """Pi^{ab,cd}(k, p) at finite ω_E, T (no sharp Fermi step)."""
    k_mid = k_cart + p_cart / 2
    H_k = H_bloch_at(k_cart)
    H_kp = H_bloch_at(k_cart + p_cart)
    eigs_k, U_k = np.linalg.eigh(H_k)
    eigs_kp, U_kp = np.linalg.eigh(H_kp)

    # Strain matrices at k_mid
    A_at_kmid = np.zeros((3, 3, 4, 4), dtype=complex)
    for a in range(3):
        for b in range(3):
            A_at_kmid[a, b] = A_at(k_mid, a, b)

    # Transform to band basis
    A_basis = np.zeros((3, 3, 4, 4), dtype=complex)
    for a in range(3):
        for b in range(3):
            A_basis[a, b] = U_kp.conj().T @ A_at_kmid[a, b] @ U_k

    # Smooth Fermi factors (finite T smearing)
    f_n_arr = np.array([fermi_smooth(eigs_k[n], mu, T) for n in range(4)])
    f_m_arr = np.array([fermi_smooth(eigs_kp[m], mu, T) for m in range(4)])

    K = np.zeros((3, 3, 3, 3), dtype=float)
    for n in range(4):
        for m in range(4):
            diff = f_n_arr[n] - f_m_arr[m]
            if abs(diff) < 1e-15:
                continue
            Delta = eigs_k[n] - eigs_kp[m]
            denom = Delta ** 2 + omega_E ** 2
            weight = diff * Delta / denom
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        for d in range(3):
                            term = (A_basis[a, b][m, n].conj()
                                    * A_basis[c, d][m, n])
                            K[a, b, c, d] += -2.0 * (term * weight).real
    return K


def Pi_BZ(p_cart, omega_E, T, N=12, mu=0.0, half_extent=2 * np.pi):
    """MP-shifted BZ average."""
    ks = (np.arange(N) + 0.5) * (2 * half_extent / N) - half_extent
    K_total = np.zeros((3, 3, 3, 3))
    for k1 in ks:
        for k2 in ks:
            for k3 in ks:
                K_total += Pi_at_k_omega_T(np.array([k1, k2, k3]), p_cart,
                                            omega_E, T, mu)
    return K_total / N ** 3


def TT_xyxy(K):
    return K[0, 1, 0, 1]


def extract_a2(omega_E, T, N=12, p_z_values=(0.0, 0.05, 0.1, 0.15, 0.2)):
    Pi_xyxy_list = []
    for p_z in p_z_values:
        p_cart = np.array([0.0, 0.0, p_z])
        K = Pi_BZ(p_cart, omega_E, T, N=N)
        Pi_xyxy_list.append(TT_xyxy(K))
    p_arr = np.array(p_z_values)
    Pi_arr = np.array(Pi_xyxy_list)
    coeffs = np.polyfit(p_arr ** 2, Pi_arr, 2)
    a_4, a_2, a_0 = coeffs
    return a_2, a_0, a_4, Pi_arr


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def main():
    header("G_sub Phase 5: Π_TT(p, ω_E, T) with smooth Fermi smearing")
    print()
    print("  Goal: with smooth f_β(E) AND finite ω_E, integrand is smooth across BZ;")
    print("  uniform grid convergence should be restored.")
    print("  Reference target (Phase 4 N=12): G_sub = 1/(4π) ≈ 0.07958.")
    print()

    # Strategy: scan N at FIXED (ω_E, T) to test grid convergence first.
    # Then once converged, scan (ω_E, T) → 0 for outer limit.
    #
    # Choose ω_E = T = 0.1 for first scan: well-regulated, fast compute.
    omega_E = 0.1
    T = 0.1

    print(f"  ===== Step 1: grid convergence at fixed ω_E = {omega_E}, T = {T} =====")
    print(f"  {'N':>4s}  {'time':>5s}  {'a_0':>13s}  {'a_2(my)':>13s}  "
          f"{'Pi_TT_a_2':>11s}  {'G_sub':>9s}")
    grid_results = []
    for N in [8, 10, 12, 14, 16]:
        t0 = time.time()
        a_2, a_0, _, _ = extract_a2(omega_E, T, N=N)
        elapsed = time.time() - t0
        Pi_TT_a_2 = -a_2 / 2  # path-#3 convention: factor 1/2 + sign flip
        if Pi_TT_a_2 > 0:
            G = 1 / (16 * np.pi * Pi_TT_a_2)
            G_str = f"{G:.6f}"
        else:
            G_str = "neg"
        print(f"  {N:>4d}  {elapsed:>4.1f}s  {a_0:>+.6e}  "
              f"{a_2:>+.6e}  {Pi_TT_a_2:>+.6e}  {G_str:>9s}")
        grid_results.append((N, a_2, Pi_TT_a_2, G_str))

    # If grid-converged, scan (ω_E, T) → 0 at the largest stable N
    print()
    print(f"  ===== Step 2: (ω_E, T) → 0 limit at converged N =====")
    # Use N=12 (Phase 4's stable point) for now; expand to N=16 if Phase 5 converges
    N_use = 12

    # Take T = ω_E for consistent regulators
    omega_T_pairs = [(0.5, 0.5), (0.3, 0.3), (0.2, 0.2), (0.15, 0.15),
                      (0.1, 0.1), (0.075, 0.075), (0.05, 0.05), (0.03, 0.03)]
    print(f"  N={N_use}, T = ω_E (consistent regulators)")
    print(f"  {'ω_E=T':>7s}  {'time':>5s}  {'a_0':>13s}  {'a_2(my)':>13s}  "
          f"{'Pi_TT_a_2':>11s}  {'G_sub':>10s}  {'Δ from 1/(4π)':>14s}")
    omega_results = []
    for omega_E, T in omega_T_pairs:
        t0 = time.time()
        a_2, a_0, _, _ = extract_a2(omega_E, T, N=N_use)
        elapsed = time.time() - t0
        Pi_TT_a_2 = -a_2 / 2
        if Pi_TT_a_2 > 0:
            G = 1 / (16 * np.pi * Pi_TT_a_2)
            delta = (G - 1 / (4 * np.pi)) / (1 / (4 * np.pi)) * 100
        else:
            G = None; delta = float('nan')
        G_str = f"{G:.6f}" if G else "neg"
        print(f"  {omega_E:>7.3f}  {elapsed:>4.1f}s  {a_0:>+.6e}  "
              f"{a_2:>+.6e}  {Pi_TT_a_2:>+.6e}  {G_str:>10s}  {delta:>+9.3f}%")
        omega_results.append((omega_E, a_2, Pi_TT_a_2, G))

    # Extrapolate
    if len(omega_results) >= 3:
        omegas = np.array([r[0] for r in omega_results])
        Pi_a2s = np.array([r[2] for r in omega_results])
        c_lin = np.polyfit(omegas, Pi_a2s, 1)
        c_quad = np.polyfit(omegas, Pi_a2s, 2)
        Pi_lin = c_lin[1]
        Pi_quad = c_quad[2]
        G_lin = 1 / (16 * np.pi * Pi_lin) if Pi_lin > 0 else None
        G_quad = 1 / (16 * np.pi * Pi_quad) if Pi_quad > 0 else None
        print()
        print(f"  Extrapolation (ω_E, T → 0):")
        print(f"  Linear in ω: Pi_TT_a_2 = {Pi_lin:.6f}, G_sub = {G_lin}")
        print(f"  Quadratic:   Pi_TT_a_2 = {Pi_quad:.6f}, G_sub = {G_quad}")
        print(f"  1/(4π) target = {1/(4*np.pi):.6f}")

    print()
    print("  INTERPRETATION:")
    print("  Step 1: if a_2 is N-INDEPENDENT (within ~5%) across N=8..16, the Fermi")
    print("    smearing fix has restored grid convergence.")
    print("  Step 2: with grid-converged answer, the (ω_E, T) → 0 extrapolation")
    print("    gives the genuine 1/(16πG_sub).")
    print("  If close to 1/4: confirms G_sub = 1/(4π).")


if __name__ == "__main__":
    main()
