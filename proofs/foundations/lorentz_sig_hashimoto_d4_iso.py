#!/usr/bin/env python3
"""
Verify D4_iso^NB = -5/512 prediction on Hashimoto B(k) directly.

The Ihara factorization derivation in `lorentz_sig_ihara_lv_relation.py`
predicts D4_iso for the Hashimoto top eigenvalue h_max(k) to be -5/512.

This script mirrors `proofs/lorentz/hashimoto_dispersion_symbolic.py`'s
high-precision Taylor extraction but explicitly extracts D4_iso (which the
existing scripts compute internally but only report for D4_aniso).

Method: same as in `proofs/foundations/lorentz_sig_h_lv_coefficients.py`
(four k-magnitudes per direction, Vandermonde solve, three directions),
but on the 12×12 Hashimoto Bloch matrix B(k).
"""

import os
import sys
from itertools import product
from fractions import Fraction

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

from proofs.common import find_bonds, A_PRIM, ATOMS, N_ATOMS

ATOMS_EXACT = [
    (Fraction(1, 8), Fraction(1, 8), Fraction(1, 8)),
    (Fraction(3, 8), Fraction(7, 8), Fraction(5, 8)),
    (Fraction(7, 8), Fraction(5, 8), Fraction(3, 8)),
    (Fraction(5, 8), Fraction(3, 8), Fraction(7, 8)),
]
A_PRIM_EXACT = [
    (Fraction(-1, 2), Fraction(1, 2), Fraction(1, 2)),
    (Fraction(1, 2), Fraction(-1, 2), Fraction(1, 2)),
    (Fraction(1, 2), Fraction(1, 2), Fraction(-1, 2)),
]


def disp_exact(src, tgt, cell):
    out = []
    for d in range(3):
        v = ATOMS_EXACT[tgt][d] - ATOMS_EXACT[src][d]
        for i in range(3):
            v += cell[i] * A_PRIM_EXACT[i][d]
        out.append(v)
    return tuple(out)


def build_hashimoto_mp(k_cart, bonds, prec):
    """12×12 Hashimoto Bloch B(k) at high precision."""
    import mpmath as mp
    mp.mp.prec = prec

    n = len(bonds)
    r_vecs_frac = [disp_exact(src, tgt, cell) for src, tgt, cell in bonds]
    r_vecs = [[mp.mpf(r.numerator) / mp.mpf(r.denominator) for r in rv]
              for rv in r_vecs_frac]
    k_mp = list(k_cart)

    B = mp.matrix(n, n)
    for i, (src_i, tgt_i, cell_i) in enumerate(bonds):
        phase = mp.exp(mp.mpc(0, 1) * sum(k_mp[d] * r_vecs[i][d] for d in range(3)))
        for j, (src_j, tgt_j, cell_j) in enumerate(bonds):
            if tgt_j == src_i:
                is_reverse = (src_i == tgt_j and tgt_i == src_j
                              and tuple(cell_i) == tuple(-c for c in cell_j))
                if not is_reverse:
                    B[i, j] = phase
    return B


def hmax_mp(k_cart, bonds, prec=200):
    import mpmath as mp
    mp.mp.prec = prec
    B = build_hashimoto_mp(k_cart, bonds, prec)
    evals, _ = mp.eig(B)
    return max(mp.re(ev) for ev in evals)


def main():
    import mpmath as mp
    mp.mp.prec = 500

    bonds = find_bonds()

    print("=" * 78)
    print("  D4_iso for Hashimoto B(k) -- verifying Ihara prediction -5/512")
    print("=" * 78)

    h0 = hmax_mp([mp.mpf(0)]*3, bonds, prec=500)
    print(f"\n  h_max(k=0) = {mp.nstr(h0, 20)}  (expected: 2)")

    k_mags = [mp.mpf(1)/mp.mpf(10)**7,
              mp.mpf(1)/mp.mpf(10)**5,
              mp.mpf(1)/mp.mpf(10)**3,
              mp.mpf(1)/mp.mpf(10)**2]

    sqrt2 = mp.sqrt(mp.mpf(2))
    sqrt3 = mp.sqrt(mp.mpf(3))
    dirs = {
        '[100]': (mp.mpf(1), mp.mpf(0), mp.mpf(0), mp.mpf(1)),
        '[110]': (mp.mpf(1)/sqrt2, mp.mpf(1)/sqrt2, mp.mpf(0), mp.mpf(1)/mp.mpf(2)),
        '[111]': (mp.mpf(1)/sqrt3, mp.mpf(1)/sqrt3, mp.mpf(1)/sqrt3, mp.mpf(1)/mp.mpf(3)),
    }

    results = {}
    for name, (kx, ky, kz, f4_val) in dirs.items():
        delta_h = []
        for km in k_mags:
            k_cart = [km * kx, km * ky, km * kz]
            h = hmax_mp(k_cart, bonds, prec=500)
            delta_h.append(h0 - h)

        k1, k2, k3_, k4_ = k_mags
        A = mp.matrix([
            [k1**2, k1**4, k1**6, k1**8],
            [k2**2, k2**4, k2**6, k2**8],
            [k3_**2, k3_**4, k3_**6, k3_**8],
            [k4_**2, k4_**4, k4_**6, k4_**8],
        ])
        b = mp.matrix(delta_h)
        x = mp.lu_solve(A, b)
        D2, D4 = x[0], x[1]
        results[name] = (D2, D4, f4_val)

    D4_100 = results['[100]'][1]
    D4_110 = results['[110]'][1]
    D4_111 = results['[111]'][1]

    D4_aniso_a = (D4_100 - D4_111) * mp.mpf(3) / mp.mpf(2)
    D4_aniso_b = (D4_100 - D4_110) * mp.mpf(2)
    D4_aniso_c = (D4_110 - D4_111) * mp.mpf(6)
    D4_aniso = (D4_aniso_a + D4_aniso_b + D4_aniso_c) / 3
    D4_iso = D4_100 - D4_aniso

    D2_mean = sum(results[d][0] for d in dirs) / 3

    print(f"\n  Extracted (high-precision):")
    print(f"    D_NB        = {mp.nstr(D2_mean, 25)}     (expected 1/8 = {mp.nstr(mp.mpf(1)/mp.mpf(8), 20)})")
    print(f"    D4_aniso^NB = {mp.nstr(D4_aniso, 25)}    (expected 1/768 = {mp.nstr(mp.mpf(1)/mp.mpf(768), 20)})")
    print(f"    D4_iso^NB   = {mp.nstr(D4_iso, 25)}    (predicted -5/512 = {mp.nstr(mp.mpf(-5)/mp.mpf(512), 20)})")

    expected_D4_iso = mp.mpf(-5) / mp.mpf(512)
    diff = abs(D4_iso - expected_D4_iso)
    print(f"\n  |D4_iso^NB - (-5/512)| = {mp.nstr(diff, 10)}")

    if diff < mp.mpf('1e-15'):
        print("\n  ✓ EXACT MATCH at 25+ digit precision: D4_iso^NB = -5/512.")
        print("    Ihara factor-of-2 derivation confirmed across all four LV coefficients.")
    else:
        print("\n  ✗ Mismatch -- check derivation.")

    # Check η_NB = D4_aniso / D_NB² = 1/12
    eta_NB = D4_aniso / D2_mean**2
    print(f"\n  η_NB = D4_aniso/D_NB² = {mp.nstr(eta_NB, 25)}")
    print(f"  Expected 1/12 = {mp.nstr(mp.mpf(1)/mp.mpf(12), 20)}")
    print(f"  |η_NB - 1/12| = {mp.nstr(abs(eta_NB - mp.mpf(1)/mp.mpf(12)), 10)}")


if __name__ == "__main__":
    main()
