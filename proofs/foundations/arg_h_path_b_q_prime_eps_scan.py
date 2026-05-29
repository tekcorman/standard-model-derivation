#!/usr/bin/env python3
"""
arg_h_path_b_q_prime_eps_scan.py — clean eps-scan of the Wilson holonomy
at high precision to settle the 269° vs 240° discrepancy.

Two convergence-monitoring scripts gave different answers:
  - q_prime_su2_convergence.py (numpy, eps in [1e-2, 1e-6], M=32):
      θ_∞ ≈ 269.030°
  - q_prime_pslq.py (mpmath 30 dps, eps=1e-8, M=64):
      θ ≈ 240.368°

Use mpmath at 30 dps with M=128 segments and scan eps from 1e-3 to 1e-10
to determine whether the eps→0 limit is 269°, 240°, or something else.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mpmath as mp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "proofs" / "cosmology"))

from srs_photon_bloch_primitive import (
    build_primitive_unit_cell,
    find_primitive_connectivity,
)
from arg_h_path_b_q_prime_pslq import (
    build_B_directed_mp,
    find_h_band_mp,
    wilson_loop_mp,
    make_c3_triangle_mp,
    discretize_mp,
)


mp.mp.dps = 30
H_MP = (mp.sqrt(3) + mp.j * mp.sqrt(5)) / 2


def main():
    print("=" * 78)
    print(f"Q' SU(2) Wilson holonomy: eps-scan at {mp.mp.dps} dps")
    print("=" * 78)

    verts, lat = build_primitive_unit_cell()
    bonds = find_primitive_connectivity(verts, lat)
    k_P = [mp.mpf(1) / 4, mp.mpf(1) / 4, mp.mpf(1) / 4]
    M = 32   # match the convergence script

    print(f"\n  M = {M}, scanning eps:")
    print(f"  {'eps':>10}  {'cos(θ/2)':>15}  {'|cos(θ/2)|':>14}  {'θ (deg)':>14}  "
          f"{'2π−θ (deg)':>14}")
    print(f"  {'-'*10}  {'-'*15}  {'-'*14}  {'-'*14}  {'-'*14}")

    eps_list = [mp.mpf(s) for s in
                ["1e-3", "1e-4", "1e-5", "1e-6", "1e-7", "1e-8", "1e-9", "1e-10"]]
    for eps in eps_list:
        try:
            tri_vs = make_c3_triangle_mp(k_P, eps)
            tri_closed = list(tri_vs) + [tri_vs[0]]
            path = discretize_mp(tri_closed, M)
            W = wilson_loop_mp(bonds, path)
            det_W = W[0, 0] * W[1, 1] - W[0, 1] * W[1, 0]
            sqrt_det_W = mp.sqrt(det_W)
            tr_norm = (W[0, 0] + W[1, 1]) / sqrt_det_W
            cos_half = tr_norm / 2
            cos_h_real = cos_half.real
            if abs(cos_h_real) > 1:
                cos_h_real = mp.sign(cos_h_real) * mp.mpf(1)
            theta_half = mp.acos(cos_h_real)
            theta_deg = 2 * mp.degrees(theta_half)
            theta_other = 360 - theta_deg
            cos_str = mp.nstr(cos_h_real, 8)
            cos_abs_str = mp.nstr(abs(cos_h_real), 8)
            theta_str = mp.nstr(theta_deg, 8)
            other_str = mp.nstr(theta_other, 8)
            print(f"  {mp.nstr(eps, 4):>10}  {cos_str:>15}  {cos_abs_str:>14}  "
                  f"{theta_str:>14}  {other_str:>14}")
        except Exception as e:
            print(f"  {eps}: failed ({e})")

    print(f"\n" + "=" * 78)
    print("OK")


if __name__ == "__main__":
    main()
