#!/usr/bin/env python3
"""
arg_h_path_b_q_prime_diagnostics.py — diagnostics for Q' Berry phase result.

The Q' Berry phase attempt found γ_B → 0 as the loop shrinks (γ_B ~ eps²),
suggesting non-topological (no Berry monopole at k_P). Before declaring
Q' falsified, this script checks:

1. Does the doubly-degenerate +h eigenvalue actually SPLIT off k_P, or
   does it persist as a 2-fold band? (If persistent, U(1) Berry phase is
   trivially 0 — but SU(2) holonomy might still encode topology.)

2. What does the full 2×2 Wilson matrix look like at small eps? Is it
   close to identity (no winding) or close to a non-trivial SU(2)
   rotation (potentially winding)?

3. Does γ_B reach a quantized value (2π, π, 2π/3, etc.) at any
   intermediate eps before the small-loop limit kills it? (i.e., is
   there a topological feature at finite distance from k_P that gets
   crossed?)

4. Try alternative loops not enclosing k_P (e.g., a loop on the BZ boundary
   between k_P and another high-symmetry point). Berry phase along these
   could carry framework-relevant topology.

Run with:
    PYTHONPATH=. python3 proofs/foundations/arg_h_path_b_q_prime_diagnostics.py
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
    HIGH_SYM_POINTS,
)
from srs_photon_c3_chainmap import K_P_RED
from srs_photon_chirality_coefficient import build_B_directed


H_EXACT = (math.sqrt(3) + 1j * math.sqrt(5)) / 2
ABS_H = math.sqrt(2.0)
SIN_ARG_H = math.sqrt(5.0 / 8.0)
ARG_H = math.atan2(math.sqrt(5), math.sqrt(3))
ARG_H_DEG = math.degrees(ARG_H)
PRINT_WIDTH = 78


def fmt_z(z, prec=6):
    return f"{z.real:+.{prec}f}{z.imag:+.{prec}f}j"


def main():
    print("=" * PRINT_WIDTH)
    print("Q' diagnostics: degeneracy splitting + SU(2) Wilson + alt loops")
    print("=" * PRINT_WIDTH)

    verts, lat = build_primitive_unit_cell()
    bonds = find_primitive_connectivity(verts, lat)
    edges = canonical_edges_primitive(bonds)
    k_P = np.array(K_P_RED)

    # -----------------------------------------------------------------------
    # Diagnostic 1: Does the degeneracy split off k_P?
    # -----------------------------------------------------------------------
    print(f"\nDiagnostic 1 — Degeneracy splitting near k_P")
    print(f"-" * PRINT_WIDTH)
    print(f"  At k_P + eps · ê_x for various eps:")
    print(f"  {'eps':>10}  {'h-band eigenvalue 1':>26}  {'h-band eigenvalue 2':>26}  "
          f"{'split':>12}")
    direction = np.array([1.0, 0.0, 0.0])
    for eps in [0.0, 1e-6, 1e-4, 1e-3, 0.01, 0.05, 0.1]:
        k_test = k_P + eps * direction
        B_k = build_B_directed(bonds, k_test)
        evs = la.eigvals(B_k)
        # Find 2 eigenvalues closest to h
        idx = np.argsort([abs(ev - H_EXACT) for ev in evs])[:2]
        e1, e2 = evs[idx[0]], evs[idx[1]]
        split = abs(e1 - e2)
        print(f"  {eps:>10.5g}  {fmt_z(e1, 6):>26}  {fmt_z(e2, 6):>26}  "
              f"{split:>12.4e}")

    # -----------------------------------------------------------------------
    # Diagnostic 2: Full SU(2) Wilson matrix at various eps
    # -----------------------------------------------------------------------
    print(f"\nDiagnostic 2 — Full 2×2 Wilson matrix and SU(2) holonomy")
    print(f"-" * PRINT_WIDTH)

    def find_h_band(B_at_k, h_target=H_EXACT, n_band=2):
        evs, evecs = la.eig(B_at_k)
        distances = np.abs(evs - h_target)
        idx = np.argsort(distances)[:n_band]
        band = evecs[:, idx]
        Q, _ = la.qr(band)
        return Q[:, :n_band]

    def make_c3_triangle(k_center, eps):
        # Vertices at k_center + eps · perp(1,1,1) rotated by C_3
        axis = np.array([1.0, 1.0, 1.0]) / math.sqrt(3)
        perp = np.cross(axis, np.array([1.0, 0.0, 0.0]))
        perp = perp / la.norm(perp)
        v0 = eps * perp
        v1 = np.array([v0[2], v0[0], v0[1]])
        v2 = np.array([v1[2], v1[0], v1[1]])
        return [k_center + v0, k_center + v1, k_center + v2]

    def wilson_loop(path_points, h_target=H_EXACT, n_band=2):
        bands = [find_h_band(build_B_directed(bonds, np.array(k)),
                              h_target=h_target, n_band=n_band)
                 for k in path_points]
        W = np.eye(n_band, dtype=complex)
        for i in range(len(path_points) - 1):
            M = bands[i].conj().T @ bands[i + 1]
            W = M @ W
        return W, bands

    def discretize(verts_closed, M):
        out = []
        for e in range(len(verts_closed) - 1):
            v0, v1 = verts_closed[e], verts_closed[e + 1]
            for s in range(M):
                t = s / M
                out.append((1 - t) * v0 + t * v1)
        out.append(verts_closed[-1])
        return out

    for eps in [0.001, 0.01, 0.1, 0.3]:
        tri = make_c3_triangle(k_P, eps) + [make_c3_triangle(k_P, eps)[0]]
        path = discretize(tri, 16)
        W, bands = wilson_loop(path)
        det_W = la.det(W)
        phase_U1 = np.angle(det_W)
        print(f"\n  eps = {eps}:")
        print(f"    Wilson W =")
        for row in W:
            print("      " + "  ".join(f"{x.real:+.4f}{x.imag:+.4f}j" for x in row))
        print(f"    det(W) = {fmt_z(det_W, 6)}, |det| = {abs(det_W):.4f}, "
              f"arg = {math.degrees(phase_U1):.4f}°")
        # SU(2) part: W / sqrt(det(W)) — has det = 1 by construction
        # but only well-defined up to sign of sqrt
        sqrt_det = np.sqrt(det_W)
        W_SU2 = W / sqrt_det
        # SU(2) angle: trace = 2 cos(θ/2) where θ is rotation angle
        tr_SU2 = np.trace(W_SU2)
        if abs(tr_SU2) <= 2.0:
            theta = 2 * math.acos(min(max(tr_SU2.real / 2, -1), 1))
        else:
            theta = float('nan')
        print(f"    SU(2) part: tr(W/√det) = {fmt_z(tr_SU2, 4)}, "
              f"rotation angle = {math.degrees(theta):.4f}°")

    # -----------------------------------------------------------------------
    # Diagnostic 3: Berry phase at intermediate eps
    # -----------------------------------------------------------------------
    print(f"\nDiagnostic 3 — Berry phase scan over wide eps range")
    print(f"-" * PRINT_WIDTH)
    print(f"  Looking for quantized values at any eps; check eps² vs other scaling.")
    print(f"  {'eps':>10}  {'γ_B (rad)':>16}  {'γ_B/eps² scale':>18}  "
          f"{'γ_B (deg)':>12}")
    for eps in [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        tri = make_c3_triangle(k_P, eps) + [make_c3_triangle(k_P, eps)[0]]
        path = discretize(tri, 16)
        W, _ = wilson_loop(path)
        phase_U1 = np.angle(la.det(W))
        print(f"  {eps:>10.4g}  {phase_U1:>+16.10f}  "
              f"{phase_U1 / eps**2:>+18.6f}  {math.degrees(phase_U1):>+12.4f}")

    # -----------------------------------------------------------------------
    # Diagnostic 4: Loop along a BZ-spanning path Γ → P → H → Γ (closed)
    # -----------------------------------------------------------------------
    print(f"\nDiagnostic 4 — Berry phase along Γ → P → H → Γ closed BZ path")
    print(f"-" * PRINT_WIDTH)
    Gamma = np.array(HIGH_SYM_POINTS["Γ"])
    P = np.array(HIGH_SYM_POINTS["P"])
    H_pt = np.array(HIGH_SYM_POINTS["H"])
    N = np.array(HIGH_SYM_POINTS["N"])
    print(f"  Γ = {Gamma}, P = {P}, H = {H_pt}, N = {N}")

    # Try several closed paths spanning high-symmetry points.
    paths = {
        "Γ → P → H → Γ":   [Gamma, P, H_pt, Gamma],
        "Γ → P → N → Γ":   [Gamma, P, N, Gamma],
        "Γ → P → −P → Γ":  [Gamma, P, -P, Gamma],
        "P → H → −P → −H → P":  [P, H_pt, -P, -H_pt, P],
        "P → −P → P (one trip)": [P, -P, P],
    }

    for label, verts_path in paths.items():
        path_disc = discretize(verts_path, 32)
        try:
            W, _ = wilson_loop(path_disc, h_target=H_EXACT, n_band=2)
            det_W = la.det(W)
            phase = np.angle(det_W)
            print(f"  {label:<28}: γ_B = {phase:+.6f} rad "
                  f"= {math.degrees(phase):+.4f}°,  |det W| = {abs(det_W):.4f}")
        except Exception as ex:
            print(f"  {label:<28}: FAILED ({ex})")

    # -----------------------------------------------------------------------
    # Diagnostic 5: Compare γ_B to the perimeter (rules out trivial phase
    # from "+h band" being undefined when the band crosses other bands).
    # -----------------------------------------------------------------------
    print(f"\nDiagnostic 5 — Compare γ_B to (eps², eps, perimeter) scaling")
    print(f"-" * PRINT_WIDTH)
    eps_test = [0.001, 0.005, 0.01, 0.03]
    print(f"  {'eps':>10}  {'γ_B':>14}  {'γ_B/eps':>12}  "
          f"{'γ_B/eps²':>12}  {'γ_B/perim²':>14}")
    for eps in eps_test:
        tri = make_c3_triangle(k_P, eps) + [make_c3_triangle(k_P, eps)[0]]
        path = discretize(tri, 16)
        W, _ = wilson_loop(path)
        phase = np.angle(la.det(W))
        # Triangle side length = eps · √3 (equilateral with circumradius eps)
        side = eps * math.sqrt(3)
        perim = 3 * side
        print(f"  {eps:>10.4g}  {phase:>+14.6e}  {phase/eps:>+12.4f}  "
              f"{phase/eps**2:>+12.4f}  {phase/perim**2:>+14.4f}")

    print(f"\n" + "=" * PRINT_WIDTH)
    print(f"OK: arg_h_path_b_q_prime_diagnostics completed without errors")


if __name__ == "__main__":
    main()
