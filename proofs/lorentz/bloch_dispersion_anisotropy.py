#!/usr/bin/env python3
"""
srs Bloch dispersion anisotropy at Gamma-point — O(k^4) expansion.

Fits the acoustic branch of the srs Laplacian L(k) = 3I - H(k):

  lambda_acoustic(k_phys) = D * |k_phys|^2
                           + [D4_iso + D4_aniso * f4(khat)] * |k_phys|^4
                           + ...

where f4(khat) = kx^4 + ky^4 + kz^4  (octahedral-group invariant).

The dimensionless Lorentz-violation coefficient is:
  eta_lattice = D4_aniso / D^2

IMPORTANT: k must be in physical Cartesian units, not BCC fractional coords.
The BCC reciprocal lattice is non-orthogonal; using fractional k directly
gives spuriously different D2 values by direction.

Directions probed (Cartesian):
  [100]:  f4 = 1     (maximum cubic anisotropy)
  [110]:  f4 = 1/2
  [111]:  f4 = 1/3   (P-point direction, Ramanujan-saturated)

eta_lattice = 0 means SO(3) isotropy at O(k^4) — no lattice Lorentz violation.
"""

import sys, os
import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from proofs.common import find_bonds, bloch_H, N_ATOMS, A_PRIM

K_STAR = 3


def build_B_matrix():
    """BCC reciprocal lattice matrix (rows = b_i, in units of 2pi/a)."""
    return la.inv(A_PRIM).T


def laplacian_acoustic_at_k_cart(k_cart, bonds, B_inv):
    """
    Acoustic Laplacian eigenvalue at physical Cartesian k-vector.
    k_cart: Cartesian reciprocal-space vector (in units of 2pi/a)
    B_inv:  inverse of the reciprocal lattice matrix (maps cart -> frac)
    """
    k_frac = B_inv @ k_cart
    H = bloch_H(k_frac, bonds)
    L = K_STAR * np.eye(N_ATOMS) - H
    evals = np.sort(np.real(la.eigvalsh(L)))
    return evals[0]


def fit_D2_D4(k_mags, lambda_vals):
    """Fit lambda = D2*k^2 + D4*k^4. Returns (D2, D4)."""
    A = np.column_stack([k_mags**2, k_mags**4])
    c, _, _, _ = la.lstsq(A, lambda_vals, rcond=None)
    return c[0], c[1]


def main():
    print("=" * 65)
    print("srs Bloch dispersion — O(k^4) anisotropy (Cartesian k)")
    print("=" * 65)

    bonds = find_bonds()
    B     = build_B_matrix()
    B_inv = la.inv(B)

    print(f"\nPrimitive cell: {N_ATOMS} atoms, {len(bonds)} directed bonds")
    print(f"BCC reciprocal lattice B (rows = b_i):\n{B}")

    # Verify acoustic zero at Gamma
    lam0 = laplacian_acoustic_at_k_cart(np.zeros(3), bonds, B_inv)
    print(f"\nL(Gamma) acoustic eigenvalue = {lam0:.2e}  (should be ~0)")

    # Physical Cartesian directions and their f4 values
    cart_dirs = {
        '[100]': (np.array([1., 0., 0.]),           1.0),
        '[110]': (np.array([1., 1., 0.])/np.sqrt(2), 0.5),
        '[111]': (np.array([1., 1., 1.])/np.sqrt(3), 1./3.),
    }

    # k magnitudes (physical, in units of 2pi/a)
    k_mags = np.array([0.002, 0.004, 0.006, 0.008, 0.01,
                       0.015, 0.02,  0.025, 0.03,  0.04])

    print(f"\n{'Dir':8s}  {'D (phys)':>12s}  {'D4_code':>14s}  "
          f"{'D4_code/D^2':>12s}  {'fit_err':>10s}")
    print("-" * 70)

    D_vals  = {}
    D4_vals = {}

    for name, (khat, f4) in cart_dirs.items():
        lam_list = []
        for kk in k_mags:
            k_cart = kk * khat
            lam = laplacian_acoustic_at_k_cart(k_cart, bonds, B_inv)
            lam_list.append(lam)
        lam_arr = np.array(lam_list)

        D2, D4 = fit_D2_D4(k_mags, lam_arr)
        fitted  = D2*k_mags**2 + D4*k_mags**4
        err     = np.max(np.abs(lam_arr - fitted))

        D_vals[name]  = D2
        D4_vals[name] = D4
        print(f"  {name:8s}  {D2:12.8f}  {D4:14.8f}  "
              f"{D4/D2**2:12.6f}  {err:10.2e}")

    # ---------------------------------------------------------------
    # Extract D and eta_lattice from the system of 3 equations:
    #
    #   D4_code[dir] = [D4_iso + D4_aniso * f4(dir)] * (no k factor,
    #                   since k_phys IS the physical magnitude here)
    #
    # So simply:
    #   D4_code[100] = D4_iso + D4_aniso * 1
    #   D4_code[110] = D4_iso + D4_aniso * 0.5
    #   D4_code[111] = D4_iso + D4_aniso * 1/3
    # ---------------------------------------------------------------
    print("\n--- Solving for D4_iso and D4_aniso ---")

    # From [100] - [111]:  D4_aniso * (1 - 1/3) = D4[100] - D4[111]
    #                      D4_aniso * 2/3         = D4[100] - D4[111]
    D4_100 = D4_vals['[100]']
    D4_110 = D4_vals['[110]']
    D4_111 = D4_vals['[111]']

    D4_aniso_a = (D4_100 - D4_111) / (1.0 - 1./3.)   # 3/2 * delta
    D4_aniso_b = (D4_100 - D4_110) / (1.0 - 0.5)      # 2   * delta
    D4_aniso_c = (D4_110 - D4_111) / (0.5 - 1./3.)    # 6   * delta

    D4_aniso = (D4_aniso_a + D4_aniso_b + D4_aniso_c) / 3.0
    D4_iso   = D4_100 - D4_aniso * 1.0    # from [100] equation

    print(f"  D4_aniso ([100]-[111]):  {D4_aniso_a:.8f}")
    print(f"  D4_aniso ([100]-[110]):  {D4_aniso_b:.8f}")
    print(f"  D4_aniso ([110]-[111]):  {D4_aniso_c:.8f}")
    print(f"  D4_aniso (mean):         {D4_aniso:.8f}")
    print(f"  D4_iso:                  {D4_iso:.8f}")

    # Verify
    print(f"\n  Predicted vs actual D4_code:")
    for name, (_, f4) in cart_dirs.items():
        pred = D4_iso + D4_aniso * f4
        actual = D4_vals[name]
        print(f"    {name}: pred = {pred:.6f}, actual = {actual:.6f}, "
              f"resid = {actual-pred:.2e}")

    # D (isotropic diffusion coefficient) — should be same in all directions
    D_avg = np.mean(list(D_vals.values()))
    D_spread = np.std(list(D_vals.values()))
    print(f"\n  D (isotropic, should be same in all dirs):")
    for name, D in D_vals.items():
        print(f"    {name}: D = {D:.8f}")
    print(f"  D mean = {D_avg:.8f}, spread = {D_spread:.2e}")

    # ---------------------------------------------------------------
    # eta_lattice = D4_aniso / D^2
    # ---------------------------------------------------------------
    eta = D4_aniso / D_avg**2
    eta_a = D4_aniso_a / D_avg**2
    eta_b = D4_aniso_b / D_avg**2
    eta_c = D4_aniso_c / D_avg**2

    print(f"\n{'='*65}")
    print("RESULT: eta_lattice = D4_aniso / D^2")
    print(f"{'='*65}")
    print(f"  eta ([100]-[111]):  {eta_a:.6f}")
    print(f"  eta ([100]-[110]):  {eta_b:.6f}")
    print(f"  eta ([110]-[111]):  {eta_c:.6f}")
    print(f"  eta (mean):         {eta:.6f}")
    spread = max(abs(eta_a-eta), abs(eta_b-eta), abs(eta_c-eta))
    print(f"  eta spread:         ±{spread:.6f}")

    # Interpret
    print()
    if abs(eta) < 0.01:
        verdict = "CONSISTENT WITH ZERO — dispersion isotropic to O(k^4)."
    elif abs(eta) < 0.5:
        verdict = f"NONZERO, sub-half: eta ~ {eta:.3f}. Dimension-6 Lorentz violation."
    else:
        verdict = f"O(1): eta ~ {eta:.3f}. Substantial dimension-6 Lorentz violation."
    print(f"  Interpretation: {verdict}")

    # Sign interpretation
    if eta > 0:
        sign_text = "SUBLUMINAL (eta > 0): high-energy photons slower. No photon decay."
        threshold_type = "pair-production threshold raised; universe more transparent above"
    else:
        sign_text = "SUPERLUMINAL (eta < 0): high-energy photons faster. Photon decay enabled."
        threshold_type = "photon decay threshold at"

    print(f"\n  Sign: {sign_text}")

    m_e_eV = 0.511e6
    E_P_eV = 1.22e28
    E_th_eV = (m_e_eV**2 * E_P_eV**2 / abs(eta))**0.25
    print(f"  Scale energy E_th ~ (m_e * E_P / |eta|^(1/2))^(1/2):")
    print(f"    {threshold_type} {E_th_eV:.2e} eV = {E_th_eV/1e15:.0f} PeV")

    print(f"\n  Dimension-5 note:")
    print(f"  Time-reversal symmetry at Planck scale forces O(k^1) and O(k^3)")
    print(f"  terms to zero. eta_5 = 0 follows from T-invariance, not")
    print(f"  vertex-transitivity (correction to earlier claim).")

    print(f"\n  NOTE ON SCOPE:")
    print(f"  This is the LAPLACIAN (adjacency) acoustic branch — the diffusive")
    print(f"  mode. The physical photon in a separate private derivation by the author is the NB walk propagator (Hashimoto")
    print(f"  matrix). eta_lattice for the Hashimoto dispersion may differ by an")
    print(f"  O(1) factor. This result gives the correct ORDER OF MAGNITUDE.")


if __name__ == '__main__':
    main()
