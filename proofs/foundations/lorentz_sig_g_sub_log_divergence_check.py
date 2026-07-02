#!/usr/bin/env python3
"""
G_sub Phase 5b: confirm log(p²) divergence in the matter polarization.

Phase 5 (`lorentz_sig_g_sub_dynamic_omega_T.py`) showed that with proper
finite-T Fermi smearing AND finite-ω regulator, the polynomial fit a_2
DOESN'T converge as (ω, T) → 0 — it grows monotonically (a_2 ≈ +4.27 at
ω=T=0.1, +11.5 at ω=T=0.03) and even changes sign relative to Phase 4's
sharp-T result.

This script directly tests whether Π_TT(p²) has a log(p²) term:

  Π_TT(p²) ≈ a_0 + a_2 p² + b × p² × log(p²) + a_4 (p²)² + ...

If b ≠ 0, the polynomial fit is wrong and a_2 (kinetic coefficient) is
log-divergent. If b ≈ 0, the polynomial form is right and our results
should converge.

Method
------
1. Compute Π_TT(p) at fixed (ω_E, T) for several small p_z.
2. Fit (a) pure polynomial Π = a_0 + a_2 p² + a_4 (p²)²
       (b) log-corrected   Π = a_0 + a_2 p² + b × p² log(p²) + a_4 (p²)²
3. Compare residuals + Akaike to see which form is preferred.
"""
from __future__ import annotations

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
from lorentz_sig_g_sub_dynamic_omega_T import Pi_BZ, TT_xyxy


def main():
    print("=" * 78)
    print("  G_sub Phase 5b: log(p²) divergence diagnostic")
    print("=" * 78)
    print()

    # Use N=12 (Phase 5 grid-converged), wider range of p_z values
    N = 12
    omega_E = 0.05
    T = 0.05
    p_z_values = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25])

    print(f"  N={N}, ω_E={omega_E}, T={T}")
    print(f"  p_z values: {p_z_values}")
    print()

    Pi_xyxy_list = []
    for p_z in p_z_values:
        t0 = time.time()
        p_cart = np.array([0.0, 0.0, p_z])
        K = Pi_BZ(p_cart, omega_E, T, N=N)
        Pi_xyxy = TT_xyxy(K)
        Pi_xyxy_list.append(Pi_xyxy)
        print(f"    p_z={p_z:.3f}  Pi_TT^xyxy = {Pi_xyxy:+.10e}  ({time.time()-t0:.1f}s)")
    Pi_arr = np.array(Pi_xyxy_list)
    p2 = p_z_values ** 2

    print()
    print("  Subtract a_0 (Pi at smallest p, approximate):")
    Pi_relative = Pi_arr - Pi_arr[0]
    for i, (p_z, dPi) in enumerate(zip(p_z_values, Pi_relative)):
        print(f"    p_z={p_z:.3f}  ΔPi = {dPi:+.6e}  ΔPi/p² = {dPi/p2[i]:+.6e}")
    print()

    # Fit 1: pure polynomial Π = a_0 + a_2 p² + a_4 (p²)²
    A_poly = np.column_stack([np.ones_like(p2), p2, p2 ** 2])
    coeffs_poly, _, _, _ = np.linalg.lstsq(A_poly, Pi_arr, rcond=None)
    a0_poly, a2_poly, a4_poly = coeffs_poly
    Pi_pred_poly = A_poly @ coeffs_poly
    resid_poly = Pi_arr - Pi_pred_poly
    rss_poly = np.sum(resid_poly ** 2)

    # Fit 2: log-corrected Π = a_0 + a_2 p² + b × p² log(p²) + a_4 (p²)²
    A_log = np.column_stack([np.ones_like(p2), p2, p2 * np.log(p2), p2 ** 2])
    coeffs_log, _, _, _ = np.linalg.lstsq(A_log, Pi_arr, rcond=None)
    a0_log, a2_log, b_log, a4_log = coeffs_log
    Pi_pred_log = A_log @ coeffs_log
    resid_log = Pi_arr - Pi_pred_log
    rss_log = np.sum(resid_log ** 2)

    print(f"  Fit 1: Π = a_0 + a_2 p² + a_4 (p²)²  (3 params)")
    print(f"    a_0 = {a0_poly:+.6e}")
    print(f"    a_2 = {a2_poly:+.6e}  ← purported kinetic coefficient")
    print(f"    a_4 = {a4_poly:+.6e}")
    print(f"    RSS = {rss_poly:.6e}")
    print()
    print(f"  Fit 2: Π = a_0 + a_2 p² + b × p² log(p²) + a_4 (p²)²  (4 params)")
    print(f"    a_0 = {a0_log:+.6e}")
    print(f"    a_2 = {a2_log:+.6e}")
    print(f"    b   = {b_log:+.6e}  ← log coefficient (non-zero ⇒ log divergence)")
    print(f"    a_4 = {a4_log:+.6e}")
    print(f"    RSS = {rss_log:.6e}")
    print()
    print(f"  Improvement: RSS_poly/RSS_log = {rss_poly/max(rss_log, 1e-30):.4f}")
    if rss_log < 0.5 * rss_poly:
        print(f"  ⇒ log term is statistically significant: log-divergence is REAL.")
    elif rss_log < 0.9 * rss_poly:
        print(f"  ⇒ log term is marginally significant.")
    else:
        print(f"  ⇒ log term is not improving fit; pure polynomial may be sufficient.")
    print()
    print(f"  Diagnostic: |b| / |a_2|: {abs(b_log)/max(abs(a2_log), 1e-30):.4f}")
    if abs(b_log) > 0.1 * abs(a2_log):
        print("  ⇒ log term comparable to or larger than 'kinetic' a_2 — definitely log-divergent.")
    print()

    # Cross-check at smaller (ω, T) — log divergence should make a_2 grow further
    print("  Cross-check: same fit at smaller regulators (ω=T=0.03):")
    omega_E_2 = 0.03
    T_2 = 0.03
    Pi_xyxy_list_2 = []
    for p_z in p_z_values:
        p_cart = np.array([0.0, 0.0, p_z])
        K = Pi_BZ(p_cart, omega_E_2, T_2, N=N)
        Pi_xyxy_list_2.append(TT_xyxy(K))
    Pi_arr_2 = np.array(Pi_xyxy_list_2)
    coeffs_poly_2, _, _, _ = np.linalg.lstsq(A_poly, Pi_arr_2, rcond=None)
    coeffs_log_2, _, _, _ = np.linalg.lstsq(A_log, Pi_arr_2, rcond=None)
    print(f"    (ω, T) = (0.05, 0.05): poly a_2 = {a2_poly:+.6e}, log b = {b_log:+.6e}")
    print(f"    (ω, T) = (0.03, 0.03): poly a_2 = {coeffs_poly_2[1]:+.6e}, "
          f"log b = {coeffs_log_2[2]:+.6e}")
    print()
    if abs(coeffs_poly_2[1]) > 1.5 * abs(a2_poly):
        print("  ⇒ poly a_2 grows as regulators decrease — confirms log divergence.")
    if abs(coeffs_log_2[2] - b_log) / max(abs(b_log), 1e-30) < 0.3:
        print("  ⇒ log coefficient b is regulator-stable: clean structural piece.")


if __name__ == "__main__":
    main()
