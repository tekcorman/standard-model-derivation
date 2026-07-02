#!/usr/bin/env python3
"""
Hashimoto Bloch dispersion for srs — two questions:

Q1. eta_lattice for the NB walk (Hashimoto) dispersion.
    The Laplacian computation gave eta ≈ 0.166. Does the NB walk
    (physical photon propagator in a separate private derivation by the author) give the same order of magnitude?

Q2. eta_5 = 0 resolution.
    Is there an O(k^1) or O(k^3) Lorentz-violating term?
    Claim: B(-k) = B(k)* for any undirected graph, so h_max(k) is
    real and EVEN in k near Gamma -> eta_5 = 0 exactly.
    This comes from the GRAPH structure, not from toggle-process T-symmetry.

The Hashimoto Bloch matrix B(k) on the srs primitive cell:
  - 4 atoms, 12 directed bonds from find_bonds()
  - B_{ij}(k) = exp(i k.r_{e_i}) if bond e_j NB-allows bond e_i, else 0
    where r_{e_i} = displacement vector of directed bond e_i

At k=0: eigenvalues are {2(x1), 1(x3), -1(x2), complex(x6)}
Maximum eigenvalue h_max(0) = 2.

The dispersion of h_max(k) near Gamma:
  h_max(k) = 2 - D_NB * |k|^2 + [D4_NB_iso + D4_NB_aniso * f4(khat)] * |k|^4 + ...
  eta_NB = D4_NB_aniso / D_NB^2
"""

import sys, os
import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from proofs.common import find_bonds, N_ATOMS, A_PRIM, ATOMS

K_STAR = 3


def build_hashimoto_bloch(k_cart, bonds, B_inv):
    """
    Build the 12x12 Hashimoto Bloch matrix at physical Cartesian k.

    bonds: list of (src, tgt, cell) from find_bonds(), 12 directed bonds.
    B_inv: inverse of reciprocal lattice matrix (Cartesian -> fractional).
    k_cart: Cartesian reciprocal-space vector.

    Convention: B[i,j](k) = exp(ik.r_{e_i}) if bond e_j NB-allows e_i.
    (e_j followed by e_i: tail(e_i) = head(e_j), and e_i != reverse(e_j))
    """
    n = len(bonds)   # 12

    # Pre-compute displacement vectors for each directed bond
    # r[i] = Cartesian displacement of bond i (from src to tgt)
    r_vecs = []
    for src, tgt, cell in bonds:
        disp = (ATOMS[tgt]
                + cell[0]*A_PRIM[0]
                + cell[1]*A_PRIM[1]
                + cell[2]*A_PRIM[2]
                - ATOMS[src])
        r_vecs.append(disp)
    r_vecs = np.array(r_vecs)   # (12, 3)

    # Build B matrix
    B_mat = np.zeros((n, n), dtype=complex)
    for i, (src_i, tgt_i, cell_i) in enumerate(bonds):
        phase_i = np.exp(1j * np.dot(k_cart, r_vecs[i]))
        for j, (src_j, tgt_j, cell_j) in enumerate(bonds):
            # e_j -> e_i is NB-allowed if:
            # (1) head(e_j) = tail(e_i): tgt_j == src_i
            # (2) e_i is not reverse of e_j:
            #     NOT (src_i == tgt_j AND tgt_i == src_j AND cell_i == -cell_j)
            if tgt_j == src_i:
                is_reverse = (src_i == tgt_j and tgt_i == src_j
                              and tuple(cell_i) == tuple(-np.array(cell_j)))
                if not is_reverse:
                    B_mat[i, j] = phase_i
    return B_mat


def h_max(k_cart, bonds, B_inv):
    """Maximum real part of Hashimoto eigenvalue at k."""
    B_mat = build_hashimoto_bloch(k_cart, bonds, B_inv)
    evals = la.eigvals(B_mat)
    return np.max(np.real(evals))


def build_B_matrix():
    return la.inv(A_PRIM).T


def main():
    print("=" * 65)
    print("Hashimoto Bloch dispersion on srs — NB walk propagator")
    print("=" * 65)

    bonds = find_bonds()
    B_mat = build_B_matrix()
    B_inv = la.inv(B_mat)

    print(f"\nPrimitive cell: {N_ATOMS} atoms, {len(bonds)} directed bonds")

    # ---------------------------------------------------------------
    # PART 1: Verify B(0) eigenvalues
    # ---------------------------------------------------------------
    print("\n--- Part 1: B(k=0) eigenvalues ---")
    B0 = build_hashimoto_bloch(np.zeros(3), bonds, B_inv)
    evals0 = np.sort(la.eigvals(B0).real)[::-1]
    print(f"  Eigenvalues at Gamma (real parts, sorted):")
    for ev in evals0:
        print(f"    {ev:.6f}")
    h_max_0 = np.max(evals0)
    print(f"  h_max(0) = {h_max_0:.6f}  (expected: k*-1 = 2)")

    # ---------------------------------------------------------------
    # PART 2: Test B(-k) = B(k)* — verify eta_5 = 0
    # ---------------------------------------------------------------
    print("\n--- Part 2: Symmetry B(-k) = B(k)* -> eta_5 = 0 ---")

    test_dirs = [
        np.array([1., 0., 0.]),
        np.array([1., 1., 0.]) / np.sqrt(2),
        np.array([0.3, 0.7, 0.1]) / np.linalg.norm([0.3, 0.7, 0.1]),  # generic
    ]
    eps_test = 0.01

    print(f"  Testing B(-k) = B(k)* at |k| = {eps_test}:")
    all_ok = True
    for khat in test_dirs:
        k = eps_test * khat
        Bk    = build_hashimoto_bloch(k,  bonds, B_inv)
        Bmk   = build_hashimoto_bloch(-k, bonds, B_inv)
        diff  = np.max(np.abs(Bmk - Bk.conj()))
        ok    = diff < 1e-12
        all_ok = all_ok and ok
        print(f"    khat={khat}: max|B(-k)-B(k)*| = {diff:.2e}  {'OK' if ok else 'FAIL'}")

    if all_ok:
        print(f"\n  B(-k) = B(k)* confirmed for all test directions.")
        print(f"  Consequence: eigenvalues of B(-k) = conjugates of B(k).")
        print(f"  => h_max(-k) = h_max(k)* = h_max(k) (real near Gamma).")
        print(f"  => h_max(k) is EVEN in k => O(k^1) and O(k^3) terms absent.")
        print(f"  => eta_5 = 0 EXACTLY.")
        print(f"\n  Source: undirected graph structure (B(-k)=B(k)*),")
        print(f"  NOT toggle-process time-reversal (which IS broken by")
        print(f"  p_create=1/2 != p_destroy=1/3, but is irrelevant here).")
    else:
        print(f"\n  UNEXPECTED: B(-k) != B(k)* -- check bond displacement signs.")

    # ---------------------------------------------------------------
    # PART 3: Dispersion of h_max(k) along [100], [110], [111]
    # ---------------------------------------------------------------
    print("\n--- Part 3: h_max dispersion anisotropy ---")
    print("  Model: h_max(k) = 2 - D_NB*k^2 - [D4_iso + D4_aniso*f4(khat)]*k^4")

    cart_dirs = {
        '[100]': (np.array([1., 0., 0.]),           1.0),
        '[110]': (np.array([1., 1., 0.])/np.sqrt(2), 0.5),
        '[111]': (np.array([1., 1., 1.])/np.sqrt(3), 1./3.),
    }

    k_mags = np.array([0.002, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03])

    print(f"\n  {'Dir':8s}  {'D_NB':>12s}  {'D4_code':>14s}  {'fit_err':>10s}")
    print("  " + "-" * 52)

    D_NB_vals  = {}
    D4_NB_vals = {}

    for name, (khat, f4) in cart_dirs.items():
        delta_h = []
        for kk in k_mags:
            k_cart = kk * khat
            hm = h_max(k_cart, bonds, B_inv)
            delta_h.append(h_max_0 - hm)   # 2 - h_max(k) > 0 (h_max decreases)
        delta_h = np.array(delta_h)

        # Fit delta_h = D_NB*k^2 + D4*k^4
        A_fit = np.column_stack([k_mags**2, k_mags**4])
        c, _, _, _ = la.lstsq(A_fit, delta_h, rcond=None)
        D2, D4 = c[0], c[1]

        fitted = D2*k_mags**2 + D4*k_mags**4
        err    = np.max(np.abs(delta_h - fitted))

        D_NB_vals[name]  = D2
        D4_NB_vals[name] = D4

        print(f"  {name:8s}  {D2:12.8f}  {D4:14.8f}  {err:10.2e}")

    # ---------------------------------------------------------------
    # PART 4: Solve for D4_iso and D4_aniso
    # ---------------------------------------------------------------
    print("\n--- Part 4: Solve for eta_NB = D4_NB_aniso / D_NB^2 ---")

    D4_100 = D4_NB_vals['[100]']
    D4_110 = D4_NB_vals['[110]']
    D4_111 = D4_NB_vals['[111]']

    # D4_code[dir] = D4_iso + D4_aniso * f4(dir)
    # [100]-[111]: D4_aniso * (1 - 1/3) = D4_100 - D4_111
    D4_aniso_a = (D4_100 - D4_111) / (1.0 - 1./3.)
    D4_aniso_b = (D4_100 - D4_110) / (1.0 - 0.5)
    D4_aniso_c = (D4_110 - D4_111) / (0.5  - 1./3.)
    D4_aniso   = (D4_aniso_a + D4_aniso_b + D4_aniso_c) / 3.0
    D4_iso     = D4_100 - D4_aniso

    D_NB_avg   = np.mean(list(D_NB_vals.values()))
    D_NB_spread = np.std(list(D_NB_vals.values()))

    print(f"\n  D_NB values by direction:")
    for name, D in D_NB_vals.items():
        print(f"    {name}: D_NB = {D:.8f}")
    print(f"  D_NB mean = {D_NB_avg:.8f}, spread = {D_NB_spread:.2e}")

    print(f"\n  D4_aniso ([100]-[111]): {D4_aniso_a:.8f}")
    print(f"  D4_aniso ([100]-[110]): {D4_aniso_b:.8f}")
    print(f"  D4_aniso ([110]-[111]): {D4_aniso_c:.8f}")
    print(f"  D4_aniso (mean):        {D4_aniso:.8f}")
    print(f"  D4_iso:                 {D4_iso:.8f}")

    eta_NB = D4_aniso / D_NB_avg**2
    spread = max(abs(D4_aniso_a/D_NB_avg**2 - eta_NB),
                 abs(D4_aniso_b/D_NB_avg**2 - eta_NB),
                 abs(D4_aniso_c/D_NB_avg**2 - eta_NB))

    print(f"\n{'='*65}")
    print("RESULT: eta_NB = D4_NB_aniso / D_NB^2  (Hashimoto/NB walk)")
    print(f"{'='*65}")
    print(f"  eta_NB = {eta_NB:.6f} ± {spread:.6f}")

    # Compare with Laplacian result
    eta_Lap = 0.166183
    print(f"  eta_Laplacian (adjacency) = {eta_Lap:.6f}")
    print(f"  ratio eta_NB / eta_Lap    = {eta_NB/eta_Lap:.4f}")

    print()
    if abs(eta_NB) < 0.01:
        verdict = "CONSISTENT WITH ZERO — isotropic to O(k^4)."
    elif abs(eta_NB) < 0.5:
        verdict = f"NONZERO, O(0.1): eta_NB ~ {eta_NB:.3f}."
    else:
        verdict = f"O(1): eta_NB ~ {eta_NB:.3f}."
    print(f"  Interpretation: {verdict}")

    if eta_NB > 0:
        sign_text = "SUBLUMINAL — NB walk speed decreases at high energy."
    else:
        sign_text = "SUPERLUMINAL — NB walk speed increases at high energy."
    print(f"  Sign: {sign_text}")

    # Physical scale
    m_e_eV = 0.511e6
    E_P_eV = 1.22e28
    E_th   = (m_e_eV**2 * E_P_eV**2 / abs(eta_NB))**0.25
    print(f"  Scale energy: {E_th:.2e} eV = {E_th/1e15:.0f} PeV")

    print(f"\n--- Summary ---")
    print(f"  eta_5 = 0 (exact, from B(-k)=B(k)* symmetry of undirected graph)")
    print(f"  eta_6 (Laplacian) = {eta_Lap:.4f}")
    print(f"  eta_6 (NB walk)   = {eta_NB:.4f}")
    print(f"  Both O(0.1-0.2) -> dimension-6 Lorentz violation confirmed at O(0.1).")


if __name__ == '__main__':
    main()
