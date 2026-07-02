#!/usr/bin/env python3
"""
arg_h_path_b_q_prime_pslq.py — high-precision numerical Wilson holonomy +
PSLQ integer-relation search to identify the cleanly converging 269.030°.

Method.
1. Use mpmath at 50 decimal digits to compute the Wilson holonomy at
   eps = 10⁻¹⁰ around a small C_3 triangle, with M = 256 path
   discretization. The non-Hermitian B(k) eigenvectors at each path
   point are computed via mpmath linear algebra.
2. Extract cos(θ/2) to ~30 significant digits.
3. PSLQ search over the framework's structural basis:
   {1, √3, √5, √15, π, π², ...} for an integer relation matching cos(θ/2).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from numpy import linalg as la
import mpmath as mp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "proofs" / "cosmology"))

from srs_photon_bloch_primitive import (
    build_primitive_unit_cell,
    find_primitive_connectivity,
)
from srs_photon_c3_chainmap import K_P_RED


mp.mp.dps = 30

H_MP = (mp.sqrt(3) + mp.j * mp.sqrt(5)) / 2
SIN_ARG_H_MP = mp.sqrt(mp.mpf(5) / 8)
ARG_H_MP = mp.atan2(mp.sqrt(5), mp.sqrt(3))


def build_B_directed_mp(bonds, k_red):
    """B(k) symbolic with mpmath complex precision."""
    n = len(bonds)
    B = mp.matrix(n, n)
    two_pi_i = 2 * mp.pi * mp.j
    for e_idx, (e_src, e_tgt, e_cell, _dr) in enumerate(bonds):
        for f_idx, (f_src, f_tgt, f_cell, _dr2) in enumerate(bonds):
            if f_src != e_tgt:
                continue
            rev_cell = tuple(-c for c in e_cell)
            if f_src == e_tgt and f_tgt == e_src and f_cell == rev_cell:
                continue
            phase = -two_pi_i * (k_red[0]*e_cell[0] + k_red[1]*e_cell[1] + k_red[2]*e_cell[2])
            B[f_idx, e_idx] = B[f_idx, e_idx] + mp.exp(phase)
    return B


def find_h_band_mp(B_at_k, h_target, n_band=2):
    """Return 12×n_band matrix of (orthonormalized) right eigenvectors of
    B closest to eigenvalue h_target."""
    # mpmath has eig
    evs, evecs_R = mp.eig(B_at_k)
    # Sort by distance to h_target.
    distances = [abs(ev - h_target) for ev in evs]
    idx_sorted = sorted(range(len(evs)), key=lambda i: distances[i])
    idx = idx_sorted[:n_band]
    band_cols = mp.matrix([[evecs_R[i, j] for j in idx] for i in range(12)])
    # QR orthonormalize via Gram-Schmidt.
    out = mp.matrix(12, n_band)
    for k in range(n_band):
        v = mp.matrix([band_cols[i, k] for i in range(12)])
        for kk in range(k):
            w = mp.matrix([out[i, kk] for i in range(12)])
            # ⟨w|v⟩
            dot = mp.mpc(0)
            for i in range(12):
                dot = dot + mp.conj(w[i]) * v[i]
            for i in range(12):
                v[i] = v[i] - dot * w[i]
        # normalize
        nrm = mp.sqrt(sum(mp.conj(v[i]) * v[i] for i in range(12)))
        for i in range(12):
            out[i, k] = v[i] / nrm
    return out


def wilson_loop_mp(bonds, path_points, h_target=H_MP, n_band=2):
    """Compute Wilson product around path."""
    bands = []
    for k in path_points:
        B = build_B_directed_mp(bonds, k)
        Q = find_h_band_mp(B, h_target, n_band=n_band)
        bands.append(Q)
    # Initial W = I_2
    W = mp.eye(n_band)
    for i in range(len(path_points) - 1):
        Q_a = bands[i]
        Q_b = bands[i + 1]
        # M = Q_a^† · Q_b   (n_band × n_band)
        M = mp.matrix(n_band, n_band)
        for r in range(n_band):
            for c in range(n_band):
                s = mp.mpc(0)
                for k in range(12):
                    s = s + mp.conj(Q_a[k, r]) * Q_b[k, c]
                M[r, c] = s
        W = M * W
    return W


def make_c3_triangle_mp(k_center, eps):
    """Same convention as the numerical script."""
    perp = [mp.mpf(0), mp.mpf(1) / mp.sqrt(2), -mp.mpf(1) / mp.sqrt(2)]
    v0 = [eps * perp[i] for i in range(3)]
    # Cyclic shift v -> (v[2], v[0], v[1])
    v1 = [v0[2], v0[0], v0[1]]
    v2 = [v1[2], v1[0], v1[1]]
    return [
        [k_center[i] + v0[i] for i in range(3)],
        [k_center[i] + v1[i] for i in range(3)],
        [k_center[i] + v2[i] for i in range(3)],
    ]


def discretize_mp(verts_closed, M):
    out = []
    for e in range(len(verts_closed) - 1):
        v0 = verts_closed[e]
        v1 = verts_closed[e + 1]
        for s in range(M):
            t = mp.mpf(s) / M
            out.append([(1 - t) * v0[i] + t * v1[i] for i in range(3)])
    out.append(verts_closed[-1])
    return out


def main():
    print("=" * 78)
    print("Q' SU(2) Wilson holonomy at high precision + PSLQ identification")
    print("=" * 78)

    verts, lat = build_primitive_unit_cell()
    bonds = find_primitive_connectivity(verts, lat)
    k_P = [mp.mpf(1) / 4, mp.mpf(1) / 4, mp.mpf(1) / 4]

    # Try moderate precision first to keep run time tractable.
    eps = mp.mpf("1e-8")
    M = 64
    print(f"\n  Computing Wilson loop at eps = {eps}, M = {M} ...")
    print(f"  (this is slow at mpmath precision {mp.mp.dps} dps; may take minutes)")

    tri_vs = make_c3_triangle_mp(k_P, eps)
    tri_closed = list(tri_vs) + [tri_vs[0]]
    path = discretize_mp(tri_closed, M)
    W = wilson_loop_mp(bonds, path)

    det_W = W[0, 0] * W[1, 1] - W[0, 1] * W[1, 0]
    sqrt_det_W = mp.sqrt(det_W)
    tr_W_norm = (W[0, 0] + W[1, 1]) / sqrt_det_W
    cos_half = tr_W_norm / 2
    print(f"\n  tr(W/√det(W)) / 2 = {cos_half}")
    print(f"  Re(cos(θ/2)) = {cos_half.real}")
    print(f"  Im(cos(θ/2)) = {cos_half.imag}")

    cos_h = cos_half.real
    if abs(cos_h) > 1:
        cos_h = mp.sign(cos_h) * mp.mpf(1)
    theta_half = mp.acos(cos_h)
    theta_deg = 2 * mp.degrees(theta_half)
    print(f"  θ = {theta_deg}°")

    # PSLQ: try to find integer relation cos(θ/2) = a₀ + a₁√3 + a₂√5 + a₃√15 + ...
    print(f"\n— PSLQ integer-relation search —")
    basis_names = ["1", "sqrt(3)", "sqrt(5)", "sqrt(15)",
                   "1/π", "π", "1/sqrt(2)",
                   "(√5+√3)/sqrt(2)/4",
                   "sqrt(3/8)+sqrt(5/8)"]
    basis_vals = [
        mp.mpf(1),
        mp.sqrt(3),
        mp.sqrt(5),
        mp.sqrt(15),
        1 / mp.pi,
        mp.pi,
        1 / mp.sqrt(2),
        (mp.sqrt(5) + mp.sqrt(3)) / (4 * mp.sqrt(2)),
        mp.sqrt(mp.mpf(3) / 8) + mp.sqrt(mp.mpf(5) / 8),
    ]
    target = cos_h
    print(f"  Target cos(θ/2) = {mp.nstr(target, 30)}")
    # Form vector (target, basis) and run PSLQ.
    vec = [target] + basis_vals
    try:
        rel = mp.pslq(vec, maxcoeff=10**6)
        print(f"  PSLQ relation: {rel}")
        if rel is not None and rel[0] != 0:
            # rel[0]·target + sum(rel[i+1]·basis_vals[i]) = 0
            # → target = - sum(rel[i+1] · basis_vals[i]) / rel[0]
            print(f"  Interpretation:")
            num_terms = []
            for i, name in enumerate(basis_names):
                coef = -rel[i + 1]
                if coef != 0:
                    num_terms.append(f"{coef}·{name}")
            print(f"    cos(θ/2) = ({' + '.join(num_terms)}) / {rel[0]}")
    except Exception as e:
        print(f"  PSLQ failed: {e}")

    # Try θ itself in PSLQ
    print(f"\n  PSLQ on θ (radians) for relation with arg(h), π, etc.:")
    theta_rad = 2 * theta_half
    target2 = theta_rad
    basis2_names = ["1", "π", "arg(h)", "arctan(√3)", "arctan(√5)", "arctan(√15)", "π/2"]
    basis2_vals = [
        mp.mpf(1),
        mp.pi,
        ARG_H_MP,
        mp.atan(mp.sqrt(3)),  # = π/3
        mp.atan(mp.sqrt(5)),
        mp.atan(mp.sqrt(15)),
        mp.pi / 2,
    ]
    print(f"  Target θ (rad) = {mp.nstr(target2, 30)}")
    vec2 = [target2] + basis2_vals
    try:
        rel2 = mp.pslq(vec2, maxcoeff=10**6)
        print(f"  PSLQ relation: {rel2}")
        if rel2 is not None and rel2[0] != 0:
            num_terms = []
            for i, name in enumerate(basis2_names):
                coef = -rel2[i + 1]
                if coef != 0:
                    num_terms.append(f"{coef}·{name}")
            print(f"    θ = ({' + '.join(num_terms)}) / {rel2[0]}")
    except Exception as e:
        print(f"  PSLQ failed: {e}")

    print(f"\n" + "=" * 78)
    print("OK")


if __name__ == "__main__":
    main()
