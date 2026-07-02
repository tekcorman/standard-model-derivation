#!/usr/bin/env python3
"""
arg_h_path_b_q_prime_su2_convergence.py — Convergence check for the SU(2)
Wilson holonomy angle around small loops at k_P.

The Q' diagnostic showed:
  - Degeneracy splits linearly off k_P (true band crossing).
  - U(1) Berry phase ~ eps² → 0 (no Abelian monopole).
  - SU(2) Wilson holonomy θ ≈ 257° at eps=0.001, eps-independent for small eps.

This script checks whether θ → 2π − 2·arg(h) ≈ 255.52° as eps → 0
exactly (would identify the SU(2) Berry phase as the structural angle
needed for β closure).

Run with:
    PYTHONPATH=. python3 proofs/foundations/arg_h_path_b_q_prime_su2_convergence.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from numpy import linalg as la

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "proofs" / "cosmology"))

from srs_photon_bloch_primitive import (
    build_primitive_unit_cell,
    find_primitive_connectivity,
    canonical_edges_primitive,
)
from srs_photon_c3_chainmap import K_P_RED
from srs_photon_chirality_coefficient import build_B_directed


H_EXACT = (math.sqrt(3) + 1j * math.sqrt(5)) / 2
SIN_ARG_H = math.sqrt(5.0 / 8.0)
ARG_H = math.atan2(math.sqrt(5), math.sqrt(3))
ARG_H_DEG = math.degrees(ARG_H)
PRINT_WIDTH = 78


def find_h_band(B_at_k, h_target=H_EXACT, n_band=2):
    evs, evecs = la.eig(B_at_k)
    distances = np.abs(evs - h_target)
    idx = np.argsort(distances)[:n_band]
    band = evecs[:, idx]
    Q, _ = la.qr(band)
    return Q[:, :n_band]


def make_c3_triangle(k_center, eps):
    axis = np.array([1.0, 1.0, 1.0]) / math.sqrt(3)
    perp = np.cross(axis, np.array([1.0, 0.0, 0.0]))
    perp = perp / la.norm(perp)
    v0 = eps * perp
    v1 = np.array([v0[2], v0[0], v0[1]])
    v2 = np.array([v1[2], v1[0], v1[1]])
    return [k_center + v0, k_center + v1, k_center + v2]


def discretize(verts_closed, M):
    out = []
    for e in range(len(verts_closed) - 1):
        v0, v1 = verts_closed[e], verts_closed[e + 1]
        for s in range(M):
            t = s / M
            out.append((1 - t) * v0 + t * v1)
    out.append(verts_closed[-1])
    return out


def wilson_loop(bonds, path_points, h_target=H_EXACT, n_band=2):
    bands = [find_h_band(build_B_directed(bonds, np.array(k)),
                          h_target=h_target, n_band=n_band)
             for k in path_points]
    W = np.eye(n_band, dtype=complex)
    for i in range(len(path_points) - 1):
        M = bands[i].conj().T @ bands[i + 1]
        W = M @ W
    return W


def main():
    print("=" * PRINT_WIDTH)
    print("Q' SU(2) Wilson holonomy convergence: does θ → 2π − 2·arg(h)?")
    print("=" * PRINT_WIDTH)

    verts, lat = build_primitive_unit_cell()
    bonds = find_primitive_connectivity(verts, lat)
    edges = canonical_edges_primitive(bonds)
    k_P = np.array(K_P_RED)

    target_2pi_2argh = 2 * math.pi - 2 * ARG_H
    target_deg = math.degrees(target_2pi_2argh)
    print(f"\n  arg(h) = {ARG_H_DEG:.6f}°")
    print(f"  2·arg(h) = {2*ARG_H_DEG:.6f}°")
    print(f"  2π − 2·arg(h) = {target_deg:.6f}° (candidate for SU(2) holonomy)")
    print(f"  2·arg(h) = {2*ARG_H_DEG:.6f}° (alternative candidate)")
    print(f"  sin(arg h) = √(5/8) = {SIN_ARG_H:.6f}")

    print(f"\n  Convergence scan: SU(2) angle as eps → 0, M = 32 (fine discretization)")
    print(f"  {'eps':>10}  {'tr(W)/2 = cos(θ/2)':>22}  "
          f"{'θ (deg)':>14}  {'Δ vs 2π−2·arg(h)':>20}  {'Δ vs 2·arg(h)':>16}")
    print(f"  {'-'*10}  {'-'*22}  {'-'*14}  {'-'*20}  {'-'*16}")

    M = 32
    results = []
    eps_list = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    for eps in eps_list:
        tri_vs = make_c3_triangle(k_P, eps)
        tri_closed = list(tri_vs) + [tri_vs[0]]
        path = discretize(tri_closed, M)
        W = wilson_loop(bonds, path)
        det_W = la.det(W)
        if abs(det_W) < 0.99:
            print(f"  WARN: |det W| = {abs(det_W):.4f} at eps={eps}, band crossing?")
            continue
        # SU(2) part: divide W by sqrt(det)
        sqrt_det = np.sqrt(det_W)
        W_SU2 = W / sqrt_det
        tr_SU2 = np.trace(W_SU2).real
        # Force trace into [-2, 2] for arccos
        tr_clamped = min(max(tr_SU2 / 2, -1), 1)
        theta_SU2 = 2 * math.acos(tr_clamped)
        # SU(2) angle has sign ambiguity; check both branches
        theta_deg = math.degrees(theta_SU2)
        delta_2pi2argh = abs(theta_SU2 - target_2pi_2argh)
        delta_2argh = abs(theta_SU2 - 2 * ARG_H)
        results.append((eps, tr_SU2 / 2, theta_deg, delta_2pi2argh, delta_2argh))
        print(f"  {eps:>10.5g}  {tr_SU2 / 2:>+22.10f}  "
              f"{theta_deg:>+14.6f}  {delta_2pi2argh:>20.4e}  "
              f"{delta_2argh:>16.4e}")

    # -----------------------------------------------------------------------
    # Convergence analysis
    # -----------------------------------------------------------------------
    print(f"\n  Convergence analysis:")
    if len(results) >= 2:
        eps_finest, tr_finest, theta_finest_deg, d1_finest, d2_finest = results[-1]
        eps_coarsest, tr_coarsest, theta_coarsest_deg, d1_coarsest, d2_coarsest = results[0]
        print(f"    Finest grid (eps={eps_finest}): θ = {theta_finest_deg:.6f}°")
        print(f"    Coarsest (eps={eps_coarsest}):  θ = {theta_coarsest_deg:.6f}°")
        print(f"    Drift: {abs(theta_finest_deg - theta_coarsest_deg):.6f}°")
        print(f"\n    Closest of two candidates:")
        if d1_finest < d2_finest:
            print(f"      θ → 2π − 2·arg(h) at finest (Δ = {d1_finest:.4e})")
            asymp_target = target_2pi_2argh
        else:
            print(f"      θ → 2·arg(h) at finest (Δ = {d2_finest:.4e})")
            asymp_target = 2 * ARG_H

        # Polynomial extrapolation to eps → 0 by quadratic fit (last 3 points)
        if len(results) >= 3:
            last3 = results[-3:]
            eps_arr = np.array([r[0] for r in last3])
            theta_arr_rad = np.array([math.radians(r[2]) for r in last3])
            # Fit θ(eps) = θ_0 + a·eps²
            X = np.column_stack([np.ones_like(eps_arr), eps_arr**2])
            coef, _, _, _ = la.lstsq(X, theta_arr_rad, rcond=None)
            theta_0 = coef[0]
            a = coef[1]
            print(f"\n    Quadratic fit θ(eps) = θ_0 + a·eps²:")
            print(f"      θ_0 = {math.degrees(theta_0):+.6f}°")
            print(f"      a   = {a:+.4e}")
            print(f"      candidate target 2π−2·arg(h) = {target_deg:+.6f}°")
            print(f"      |θ_0 − (2π−2·arg(h))| = "
                  f"{abs(math.degrees(theta_0) - target_deg):.4e}°")

    # -----------------------------------------------------------------------
    # Sign convention check: SU(2) angle has ±θ ambiguity
    # The arccos branch gives θ ∈ [0, 2π]. If actual is negative, we'd want 2π − θ.
    # -----------------------------------------------------------------------
    print(f"\n  SU(2) sign convention check at finest eps:")
    eps_min = eps_list[-1]
    tri_vs = make_c3_triangle(k_P, eps_min)
    tri_closed = list(tri_vs) + [tri_vs[0]]
    path = discretize(tri_closed, M)
    W = wilson_loop(bonds, path)
    det_W = la.det(W)
    sqrt_det_pos = np.sqrt(det_W)
    sqrt_det_neg = -sqrt_det_pos
    W_SU2_pos = W / sqrt_det_pos
    W_SU2_neg = W / sqrt_det_neg
    tr_pos = np.trace(W_SU2_pos)
    tr_neg = np.trace(W_SU2_neg)
    print(f"    tr(W/+√det) = {tr_pos:.6f}")
    print(f"    tr(W/-√det) = {tr_neg:.6f}")
    print(f"    The SU(2) holonomy θ has ±θ ambiguity (±sin), but |cos(θ/2)| is canonical.")

    # -----------------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------------
    print(f"\n" + "=" * PRINT_WIDTH)
    print(f"VERDICT")
    print(f"=" * PRINT_WIDTH)

    # Compare θ_finest to candidates
    if len(results) >= 1:
        eps_f, tr_f, theta_f, d1_f, d2_f = results[-1]
        target_match = "2π − 2·arg(h)" if d1_f < d2_f else "2·arg(h)"
        target_val = target_deg if d1_f < d2_f else 2 * ARG_H_DEG
        delta_min = min(d1_f, d2_f)
        print(f"\n  At eps = {eps_f}: θ = {theta_f:.6f}°")
        print(f"  Closest candidate: {target_match} = {target_val:.6f}°")
        print(f"  Δ = {math.degrees(delta_min):.6f}° ({delta_min:.4e} rad)")
        if delta_min < math.radians(0.1):
            print(f"  ✓ θ → {target_match} converges within 0.1°")
            print(f"  Q' SU(2) Berry phase = topological angle related to arg(h).")
            if target_match == "2π − 2·arg(h)":
                print(f"\n  Structural interpretation: SU(2) Wilson holonomy = "
                      f"2π − 2·arg(h) per loop.")
                print(f"  This is a non-Abelian topological phase encoding the walker's")
                print(f"  per-step phase angle. Connection to c = 1 in β = c·sin(arg h)·α_EM")
                print(f"  requires next step: how does the SU(2) angle map to photon")
                print(f"  polarization rotation? (Spin-1 photon vs SU(2) on rank-2 walker band.)")
        else:
            print(f"  Δ > 0.1° — convergence still drifting. Try smaller eps or higher M.")

    print(f"\n" + "=" * PRINT_WIDTH)
    print(f"OK: arg_h_path_b_q_prime_su2_convergence completed without errors")


if __name__ == "__main__":
    main()
