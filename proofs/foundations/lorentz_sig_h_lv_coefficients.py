#!/usr/bin/env python3
"""
LV coefficients of the scalar Bloch H Perron band — high-precision Taylor extraction.

Mirrors `proofs/lorentz/hashimoto_bloch_dispersion.py` but on the 4×4
scalar Bloch H instead of the 12×12 Hashimoto B.

Goal
----
Compute the D2, D4_iso, D4_aniso, D6_iso, D8_iso coefficients of the
small-|k| expansion

    λ_0(k) = 3 - D2 |k|^2 + (D4_iso + D4_aniso · f4(k̂)) |k|^4
            + D6 |k|^6 + D8 |k|^8 + O(|k|^{10})

where λ_0 is the Perron eigenvalue of H(k). f4(k̂) = k̂_x^4 + k̂_y^4 + k̂_z^4
is the cubic-anisotropy function (for [100]: f4=1; [110]: f4=1/2; [111]: f4=1/3).

Then compute η^H_NB = D4_aniso / D2² and compare to:
  - the framework's existing η_NB (Hashimoto) = 1/12
    (predictions/eta_lattice_lorentz_dim6.py).

Match ⇒ "universal LV across walkers" (operator-independent).
Mismatch ⇒ operator-distinct LV; then η^H_NB itself is a new prediction.

Method
------
At each direction k̂ ∈ {[100], [110], [111]}, sample λ_0 at four magnitudes
k_mag spanning ~4 orders of magnitude. Solve the Vandermonde
4-point system to extract D2, D4, D6, D8 simultaneously (eliminating
higher-order contamination of the D4 extraction). Use mpmath at 500-bit
precision.

The k_cart convention here matches `predictions/srs_bloch_dispersion_gamma.py`:
the headline result is in physical Cartesian k. (For convenience, we sample in
Cartesian k and use the bonds list as in proofs.common; this gives D2 = 1/16
matching S3.)
"""

import os
import sys
from itertools import product
from fractions import Fraction

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

from proofs.common import find_bonds, A_PRIM, ATOMS, N_ATOMS


# Exact-rational atom positions and primitive vectors (same as
# proofs/lorentz/hashimoto_dispersion_symbolic.py)
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
    """Exact rational displacement vector (src→tgt with cell offset)."""
    out = []
    for d in range(3):
        v = ATOMS_EXACT[tgt][d] - ATOMS_EXACT[src][d]
        for i in range(3):
            v += cell[i] * A_PRIM_EXACT[i][d]
        out.append(v)
    return tuple(out)


def build_scalar_bloch_mp(k_cart, bonds, prec):
    """Sympy-style Bloch builder in mpmath at high precision.

    H_scalar[tgt, src] = sum_{bonds (src→tgt, cell)} exp(i k_cart · r_disp)
    where r_disp is the exact displacement of bond (src, tgt, cell).
    """
    import mpmath as mp
    mp.mp.prec = prec

    r_vecs_frac = [disp_exact(src, tgt, cell) for src, tgt, cell in bonds]
    r_vecs = [[mp.mpf(r.numerator) / mp.mpf(r.denominator) for r in rv]
              for rv in r_vecs_frac]
    k_mp = list(k_cart)

    H = mp.matrix(N_ATOMS, N_ATOMS)
    for i, (src, tgt, _) in enumerate(bonds):
        phase = mp.exp(mp.mpc(0, 1) * sum(k_mp[d] * r_vecs[i][d] for d in range(3)))
        H[tgt, src] = H[tgt, src] + phase
    return H


def lambda_max_mp(k_cart, bonds, prec=200):
    import mpmath as mp
    mp.mp.prec = prec
    H = build_scalar_bloch_mp(k_cart, bonds, prec)
    evals, _ = mp.eig(H)
    return max(mp.re(ev) for ev in evals)


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def main():
    import mpmath as mp
    mp.mp.prec = 500  # ~150 digits

    bonds = find_bonds()

    header("Scalar Bloch H Perron-band LV coefficients")
    print(f"\n  Precision: {mp.mp.dps} decimal digits")

    # Reference: λ_0(k=0) = 3 (K_4 Perron)
    h0 = lambda_max_mp([mp.mpf(0), mp.mpf(0), mp.mpf(0)], bonds, prec=500)
    print(f"  λ_0(k=0) = {mp.nstr(h0, 20)}  (expected: 3)")

    # Sample k magnitudes
    k_mags = [
        mp.mpf(1) / mp.mpf(10)**7,
        mp.mpf(1) / mp.mpf(10)**5,
        mp.mpf(1) / mp.mpf(10)**3,
        mp.mpf(1) / mp.mpf(10)**2,
    ]

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
            h = lambda_max_mp(k_cart, bonds, prec=500)
            delta_h.append(h0 - h)  # 3 - λ_0(k) > 0

        # Solve: delta_h[i] = D2 k^2 + D4 k^4 + D6 k^6 + D8 k^8
        k1, k2, k3_, k4_ = k_mags
        A = mp.matrix([
            [k1**2, k1**4, k1**6, k1**8],
            [k2**2, k2**4, k2**6, k2**8],
            [k3_**2, k3_**4, k3_**6, k3_**8],
            [k4_**2, k4_**4, k4_**6, k4_**8],
        ])
        b = mp.matrix(delta_h)
        x = mp.lu_solve(A, b)
        D2, D4, D6, D8 = x[0], x[1], x[2], x[3]

        results[name] = (D2, D4, D6, D8, f4_val)

    print()
    for name, (D2, D4, D6, D8, f4) in results.items():
        print(f"  Direction {name} (f4 = {mp.nstr(f4, 8)}):")
        print(f"    D2 = {mp.nstr(D2, 30)}")
        print(f"    D4 = {mp.nstr(D4, 30)}")
        print(f"    D6 = {mp.nstr(D6, 25)}")
        print(f"    D8 = {mp.nstr(D8, 20)}")

    # Cross-direction consistency: D2 should be direction-independent
    D2s = [results[d][0] for d in ['[100]', '[110]', '[111]']]
    D2_mean = sum(D2s) / 3
    D2_spread = max(abs(D - D2_mean) for D in D2s)
    print(f"\n  D2 consistency: spread = {mp.nstr(D2_spread, 10)}  (should be ≈ 0)")

    one_sixteenth = mp.mpf(1) / mp.mpf(16)
    print(f"  D2 - 1/16 = {mp.nstr(D2_mean - one_sixteenth, 15)}  (target: 0)")
    if abs(D2_mean - one_sixteenth) < mp.mpf('1e-50'):
        print("  ✓ D2 = 1/16 confirmed (matches predictions/srs_bloch_dispersion_gamma.py).")

    # Solve for D4_iso, D4_aniso
    # D4_dir = D4_iso + D4_aniso · f4_dir
    # [100]: D4 = D4_iso + 1 · D4_aniso
    # [110]: D4 = D4_iso + (1/2) D4_aniso
    # [111]: D4 = D4_iso + (1/3) D4_aniso
    D4_100 = results['[100]'][1]
    D4_110 = results['[110]'][1]
    D4_111 = results['[111]'][1]

    # D4_aniso from [100] - [111]: factor (1 - 1/3) = 2/3
    D4_aniso_a = (D4_100 - D4_111) * mp.mpf(3) / mp.mpf(2)
    D4_aniso_b = (D4_100 - D4_110) * mp.mpf(2)
    D4_aniso_c = (D4_110 - D4_111) * mp.mpf(6)

    D4_aniso_mean = (D4_aniso_a + D4_aniso_b + D4_aniso_c) / 3
    D4_iso_mean   = D4_100 - D4_aniso_mean

    print()
    header("D4 decomposition")
    print(f"  D4_aniso ([100]-[111]):  {mp.nstr(D4_aniso_a, 25)}")
    print(f"  D4_aniso ([100]-[110]):  {mp.nstr(D4_aniso_b, 25)}")
    print(f"  D4_aniso ([110]-[111]):  {mp.nstr(D4_aniso_c, 25)}")
    print(f"  D4_aniso (mean):          {mp.nstr(D4_aniso_mean, 25)}")
    print(f"  D4_iso (mean):            {mp.nstr(D4_iso_mean, 25)}")

    # Compute η^H_NB = D4_aniso / D2²
    eta_H = D4_aniso_mean / (D2_mean * D2_mean)
    print()
    header("LV coefficient η^H_NB = D4_aniso / D2²")
    print(f"  η^H_NB = {mp.nstr(eta_H, 25)}")

    one_twelfth = mp.mpf(1) / mp.mpf(12)
    print(f"\n  Hashimoto reference: η_NB = 1/12 = {mp.nstr(one_twelfth, 25)}")
    print(f"  η^H_NB - 1/12 = {mp.nstr(eta_H - one_twelfth, 15)}")

    # Try matching to a few clean rationals
    print()
    print("  Rational candidates near η^H_NB:")
    candidates = [
        (mp.mpf(1)/mp.mpf(12),  "1/12"),
        (mp.mpf(1)/mp.mpf(24),  "1/24"),
        (mp.mpf(1)/mp.mpf(8),   "1/8"),
        (mp.mpf(1)/mp.mpf(16),  "1/16"),
        (mp.mpf(1)/mp.mpf(4),   "1/4"),
        (mp.mpf(1)/mp.mpf(6),   "1/6"),
        (mp.mpf(1)/mp.mpf(3),   "1/3"),
        (mp.mpf(1)/mp.mpf(48),  "1/48"),
        (mp.mpf(2)/mp.mpf(3),   "2/3"),
        (mp.mpf(1)/mp.mpf(2),   "1/2"),
        (mp.mpf(3)/mp.mpf(16),  "3/16"),
        (mp.mpf(5)/mp.mpf(48),  "5/48"),
    ]
    for val, name in candidates:
        diff = abs(eta_H - val)
        if diff < mp.mpf('1e-15'):
            print(f"    *** EXACT MATCH: η^H_NB = {name} (residual {mp.nstr(diff, 10)}) ***")
        elif diff < mp.mpf('1e-3'):
            print(f"    near match: {name}: {mp.nstr(val, 10)}, |diff| = {mp.nstr(diff, 8)}")

    # Same for D4_iso to see if it has clean rational form
    print()
    header("D4_iso candidates")
    print(f"  D4_iso = {mp.nstr(D4_iso_mean, 25)}")
    candidates_iso = [
        (mp.mpf(0),                    "0"),
        (mp.mpf(1)/mp.mpf(96),         "1/96"),
        (mp.mpf(1)/mp.mpf(192),        "1/192"),
        (mp.mpf(1)/mp.mpf(384),        "1/384"),
        (mp.mpf(1)/mp.mpf(768),        "1/768"),
        (mp.mpf(-1)/mp.mpf(768),       "-1/768"),
        (mp.mpf(1)/mp.mpf(1536),       "1/1536"),
        (mp.mpf(1)/mp.mpf(3072),       "1/3072"),
        (mp.mpf(-1)/mp.mpf(96),        "-1/96"),
    ]
    for val, name in candidates_iso:
        diff = abs(D4_iso_mean - val)
        if diff < mp.mpf('1e-15'):
            print(f"    *** EXACT MATCH: D4_iso = {name} (residual {mp.nstr(diff, 10)}) ***")
        elif diff < mp.mpf('1e-5'):
            print(f"    near match: {name}: {mp.nstr(val, 10)}, |diff| = {mp.nstr(diff, 8)}")

    # And D4_aniso
    print()
    header("D4_aniso candidates")
    print(f"  D4_aniso = {mp.nstr(D4_aniso_mean, 25)}")
    candidates_aniso = [
        (mp.mpf(0),                    "0"),
        (mp.mpf(1)/mp.mpf(96),         "1/96"),
        (mp.mpf(1)/mp.mpf(192),        "1/192"),
        (mp.mpf(1)/mp.mpf(384),        "1/384"),
        (mp.mpf(1)/mp.mpf(768),        "1/768"),
        (mp.mpf(1)/mp.mpf(1536),       "1/1536"),
        (mp.mpf(1)/mp.mpf(3072),       "1/3072"),
        (mp.mpf(1)/mp.mpf(48),         "1/48"),
        (mp.mpf(-1)/mp.mpf(48),        "-1/48"),
    ]
    for val, name in candidates_aniso:
        diff = abs(D4_aniso_mean - val)
        if diff < mp.mpf('1e-15'):
            print(f"    *** EXACT MATCH: D4_aniso = {name} (residual {mp.nstr(diff, 10)}) ***")
        elif diff < mp.mpf('1e-5'):
            print(f"    near match: {name}: {mp.nstr(val, 10)}, |diff| = {mp.nstr(diff, 8)}")


if __name__ == "__main__":
    main()
