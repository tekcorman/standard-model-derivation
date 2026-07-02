#!/usr/bin/env python3
"""
G_sub Phase 4: dynamical matter polarization at finite Euclidean frequency.

Per `g_sub_routes_ABC_status_2026-04-30.md`, the static (ω=0) extraction of
Π_TT(p²) is blocked by the substrate's metallic non-analyticity — the
band-crossing surfaces at μ=0 give a 1/(E_n − E_m)² singularity in the p²
coefficient that integrates to log(p) divergences and grid-noise.

This script implements the Kubo-style outer-ω limit prescription:

  Π^{ab,cd}(p, ω_E) = ∫_BZ Σ_{n,m}
       (f_n(k) − f_m(k+p)) ⟨n,k|A^{ab}(k+p/2)|m,k+p⟩ ⟨m,k+p|A^{cd}(k+p/2)|n,k⟩
       × (E_n(k) − E_m(k+p)) / [(E_n(k) − E_m(k+p))² + ω_E²]

at finite Euclidean ω_E > 0, then extrapolates ω_E → 0 AFTER the BZ integral.
This regulates the metallic surface (E_n = E_m near μ) by replacing the
singular 1/(E_n − E_m) with the bounded (E_n − E_m)/(...² + ω_E²), giving a
well-defined finite-ω polarization.

The outer ω_E → 0 limit then gives the static elastic plus the regulated
graviton kinetic. Specifically:
- a_0(ω_E) → C_44 (static elastic) as ω_E → 0.
- a_2(ω_E) → ?  this is what we extract.

If a_2(ω_E → 0) converges to a finite value: we have 1/(16π G_sub) cleanly,
and the metallic dressing is benign at low ω.

If a_2(ω_E → 0) diverges (log or 1/ω_E): the metallic dressing is structural
and emergent gravity in this metal genuinely requires a different derivation.

Method
------
1. Compute Π_TT(p_z, ω_E) for small p_z values (scaled to ω_E to stay on-shell).
2. Polynomial fit Π_TT(p²) to extract a_2(ω_E).
3. Sweep ω_E ∈ {0.5, 0.3, 0.2, 0.15, 0.1, 0.05, 0.025}, plot a_2(ω_E).
4. Extrapolate ω_E → 0.
"""
from __future__ import annotations

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
from lorentz_sig_g_sub_pi_finite_p import H_bloch_at, A_at


def Pi_finite_omega_at_k(k_cart, p_cart, omega_E, mu=0.0, tol=1e-10):
    """Return paramagnetic-style Pi^{abcd}(k, p, ω_E) at single k.

    Note: this combines what the static path-#3 called paramagnetic + diamagnetic
    into a single Lindhard expression at finite Euclidean ω. The signs are set
    so that the ω_E → 0 limit reproduces the static dia − para (subtractive)
    convention of path-#1.

    Specifically: Π(p, ω_E) = ∫ Σ (f_n - f_m) × (E_n - E_m) / ((E_n - E_m)² + ω_E²)
                            × ⟨A⟩ × ⟨A⟩  (not divided by E_n - E_m).
    """
    k_mid = k_cart + p_cart / 2
    H_k = H_bloch_at(k_cart)
    H_kp = H_bloch_at(k_cart + p_cart)
    eigs_k, U_k = np.linalg.eigh(H_k)
    eigs_kp, U_kp = np.linalg.eigh(H_kp)

    # Compute strain matrices at k_mid (symmetrized in (a,b))
    A_at_kmid = np.zeros((3, 3, 4, 4), dtype=complex)
    for a in range(3):
        for b in range(3):
            A_at_kmid[a, b] = A_at(k_mid, a, b)

    # Transform to band basis: A_basis[a,b][m,n] = ⟨m,k+p|A|n,k⟩
    A_basis = np.zeros((3, 3, 4, 4), dtype=complex)
    for a in range(3):
        for b in range(3):
            A_basis[a, b] = U_kp.conj().T @ A_at_kmid[a, b] @ U_k

    K = np.zeros((3, 3, 3, 3), dtype=float)
    for n in range(4):
        f_n = 1.0 if eigs_k[n] < mu - tol else (0.5 if abs(eigs_k[n] - mu) < tol else 0.0)
        for m in range(4):
            f_m = 1.0 if eigs_kp[m] < mu - tol else (0.5 if abs(eigs_kp[m] - mu) < tol else 0.0)
            diff = f_n - f_m
            if abs(diff) < 1e-12:
                continue
            Delta = eigs_k[n] - eigs_kp[m]
            denom = Delta ** 2 + omega_E ** 2
            # Static elastic + finite ω_E regulated: (f_n - f_m) × Delta / denom × |ME|²
            weight = diff * Delta / denom
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        for d in range(3):
                            term = (A_basis[a, b][m, n].conj()
                                    * A_basis[c, d][m, n])
                            K[a, b, c, d] += -2.0 * (term * weight).real
    return K


def Pi_finite_omega_BZ(p_cart, omega_E, N=10, mu=0.0, half_extent=2 * np.pi):
    """Monkhorst-Pack-shifted BZ average of Pi at finite ω_E."""
    ks = (np.arange(N) + 0.5) * (2 * half_extent / N) - half_extent
    K_total = np.zeros((3, 3, 3, 3))
    for k1 in ks:
        for k2 in ks:
            for k3 in ks:
                k_cart = np.array([k1, k2, k3])
                K_total += Pi_finite_omega_at_k(k_cart, p_cart, omega_E, mu)
    n_pts = N ** 3
    return K_total / n_pts


def TT_xyxy(K):
    return K[0, 1, 0, 1]


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def extract_a2_at_omega(omega_E, N=10, p_z_values=None):
    """Compute Π_TT^{xyxy}(p_z, ω_E) for several small p_z; polynomial fit gives a_2(ω_E)."""
    if p_z_values is None:
        p_z_values = (0.0, 0.05, 0.1, 0.15, 0.2)
    Pi_xyxy_list = []
    for p_z in p_z_values:
        p_cart = np.array([0.0, 0.0, p_z])
        K = Pi_finite_omega_BZ(p_cart, omega_E, N=N)
        Pi_xyxy_list.append(TT_xyxy(K))
    p_arr = np.array(p_z_values)
    Pi_arr = np.array(Pi_xyxy_list)
    # Quadratic fit Pi(p²) = a_0 + a_2 p² + a_4 p^4
    coeffs = np.polyfit(p_arr ** 2, Pi_arr, 2)
    a_4, a_2, a_0 = coeffs
    return a_2, a_0, a_4, Pi_arr


def main():
    header("G_sub Phase 4: Π_TT(p, ω_E) sweep at finite Euclidean frequency")
    print()
    print("  Goal: regulate metallic Drude/log singularities via finite ω_E.")
    print("  Extract β(ω_E) := a_2(ω_E); take ω_E → 0 OUTER limit.")
    print()
    print("  Convention: Lindhard at finite Euclidean ω_E, MP-shifted N grid.")
    print()

    omega_values = [0.5, 0.3, 0.2, 0.15, 0.1, 0.05]
    N = 8  # small grid for ω-sweep speed
    p_z_values = (0.0, 0.05, 0.1, 0.15, 0.2)

    print(f"  Grid N={N}, p_z values: {p_z_values}")
    print()
    print(f"  {'ω_E':>6s}  {'time':>7s}  {'a_0(stat)':>13s}  "
          f"{'a_2(kin)':>13s}  {'a_4':>13s}  {'1/(16π a_2) ≈ G':>16s}")

    results = []
    for omega in omega_values:
        t0 = time.time()
        a_2, a_0, a_4, Pi_arr = extract_a2_at_omega(omega, N=N, p_z_values=p_z_values)
        elapsed = time.time() - t0
        if a_2 > 1e-10:
            G_eff = 1 / (16 * np.pi * a_2)
            G_str = f"{G_eff:.4f}"
        else:
            G_str = "neg/zero"
        results.append((omega, elapsed, a_0, a_2, a_4, G_str))
        print(f"  {omega:>6.3f}  {elapsed:>5.1f}s  {a_0:>+.6e}  "
              f"{a_2:>+.6e}  {a_4:>+.6e}  {G_str:>16s}")

    print()
    print("  Trend: track a_2(ω_E) as ω_E decreases:")
    omegas = np.array([r[0] for r in results])
    a_2s = np.array([r[3] for r in results])
    print(f"  ω_E   :   {omegas}")
    print(f"  a_2   :   {a_2s}")
    if len(results) >= 3:
        # Try linear extrapolation in ω_E
        coeffs_lin = np.polyfit(omegas, a_2s, 1)
        a_2_at_zero_lin = coeffs_lin[1]
        # Try linear extrapolation in log(ω_E) (in case of log divergence)
        log_omegas = np.log(omegas)
        coeffs_log = np.polyfit(log_omegas, a_2s, 1)
        # Try a + b/ω fit (power-law / Drude)
        inv_omegas = 1 / omegas
        coeffs_inv = np.polyfit(inv_omegas, a_2s, 1)
        print()
        print(f"  Linear extrapolation in ω_E:  a_2(ω→0) → {a_2_at_zero_lin:+.6e}")
        print(f"  Log-fit slope:  d a_2 / d log(ω_E) = {coeffs_log[0]:+.6e}")
        print(f"  Inv-fit (Drude): a_∞ = {coeffs_inv[1]:+.6e}, b_Drude = {coeffs_inv[0]:+.6e}")

    print()
    print("  INTERPRETATION:")
    print("  - If a_2(ω_E) is approximately constant as ω_E → 0: graviton kinetic")
    print("    is well-defined, regulated by ω_E only as a noise filter.")
    print("    G_sub = 1/(16π × a_2(ω → 0)).")
    print("  - If a_2(ω_E) ~ const + b log(ω_E): metallic log-divergence;")
    print("    emergent gravity has a UV anomaly, needs renormalization.")
    print("  - If a_2(ω_E) ~ const + b/ω_E: Drude pole survives; the metallic")
    print("    dressing is structural and dominates the kinetic coefficient.")


if __name__ == "__main__":
    main()
