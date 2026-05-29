#!/usr/bin/env python3
"""
arg_h_path_b_q_prime_c3_split.py — split the rank-2 band into ω and ω²
C_3 irrep components at the loop's start, then track each as a SEPARATE
U(1) Berry phase around the loop.

Motivation. Q' found:
  - U(1) det(W) phase → 0 as eps → 0 (no Abelian monopole)
  - SU(2) Wilson holonomy θ ≈ 269.03° (topological but not matching
    obvious structural targets)

The full SU(2) holonomy is too noisy. But the rank-2 band is naturally
C_3-decomposable into (ω, ω²) irreps at k_P. If the loop is C_3-invariant,
the C_3 character is preserved along the loop (up to gauge). Then each
1-dim C_3 irrep gives its own U(1) Berry phase γ_ω, γ_ω². These are
gauge-canonical (no SU(2) basis ambiguity within each 1-dim irrep).

  γ_ω + γ_ω² = γ_U(1)_Abelian = 0 (we measured)
  γ_ω − γ_ω² = chirality difference (THE structural angle?)

Hypothesis. γ_ω − γ_ω² should equal a clean structural angle related to
arg(h). Specifically: if γ_ω = +(some_angle) and γ_ω² = −(some_angle),
then γ_ω − γ_ω² = 2·some_angle. Candidates: 2·arg(h), 2π·sin(arg h),
4·arg(h), etc.

Run with:
    PYTHONPATH=. python3 proofs/foundations/arg_h_path_b_q_prime_c3_split.py
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
from srs_photon_chirality_coefficient import build_B_directed, build_C3_directed


H_EXACT = (math.sqrt(3) + 1j * math.sqrt(5)) / 2
ABS_H = math.sqrt(2.0)
SIN_ARG_H = math.sqrt(5.0 / 8.0)
ARG_H = math.atan2(math.sqrt(5), math.sqrt(3))
ARG_H_DEG = math.degrees(ARG_H)
omega = np.exp(2j * math.pi / 3)
omega2 = omega.conjugate()
PRINT_WIDTH = 78


def find_h_band(B_at_k, h_target=H_EXACT, n_band=2):
    evs, evecs = la.eig(B_at_k)
    distances = np.abs(evs - h_target)
    idx = np.argsort(distances)[:n_band]
    band = evecs[:, idx]
    Q, _ = la.qr(band)
    return Q[:, :n_band]


def find_h_band_c3_resolved(B_at_k, C3_op, h_target=H_EXACT):
    """Find the doubly-degenerate +h band and split it into ω, ω² C_3 irreps.

    Returns:
       Q_ω : 12-dim vector, ω-irrep state in the +h band
       Q_ω2: 12-dim vector, ω²-irrep state in the +h band

    If the band is not exactly C_3-decomposable (e.g., off k_P), we identify
    each component by best C_3-eigenvalue match.
    """
    evs, evecs = la.eig(B_at_k)
    distances = np.abs(evs - h_target)
    idx = np.argsort(distances)[:2]
    band = evecs[:, idx]
    Q, _ = la.qr(band)
    Q_band = Q[:, :2]   # 12 × 2 orthonormal basis for +h band
    # Diagonalize C_3 in the band
    C3_in_band = Q_band.conj().T @ C3_op @ Q_band
    evs_c3, evecs_c3 = la.eig(C3_in_band)
    # Find ω and ω² components
    omega_idx = int(np.argmin(np.abs(evs_c3 - omega)))
    omega2_idx = int(np.argmin(np.abs(evs_c3 - omega2)))
    Q_omega = Q_band @ evecs_c3[:, omega_idx]
    Q_omega2 = Q_band @ evecs_c3[:, omega2_idx]
    Q_omega = Q_omega / la.norm(Q_omega)
    Q_omega2 = Q_omega2 / la.norm(Q_omega2)
    # Check c_3 character is correct
    c3_omega = (Q_omega.conj() @ C3_op @ Q_omega)
    c3_omega2 = (Q_omega2.conj() @ C3_op @ Q_omega2)
    return Q_omega, Q_omega2, c3_omega, c3_omega2


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


def main():
    print("=" * PRINT_WIDTH)
    print("Q' C_3-resolved Berry phases: γ_ω, γ_ω², γ_ω − γ_ω²")
    print("=" * PRINT_WIDTH)

    verts, lat = build_primitive_unit_cell()
    bonds = find_primitive_connectivity(verts, lat)
    edges = canonical_edges_primitive(bonds)
    k_P = np.array(K_P_RED)
    C3_dir = build_C3_directed(bonds)

    print(f"\n  arg(h) = {ARG_H_DEG:.6f}°")
    print(f"  2·arg(h) = {2*ARG_H_DEG:.6f}°")
    print(f"  sin(arg h) = {SIN_ARG_H:.6f}")

    print(f"\n  C_3-resolved Berry phases of (ω, ω²) components of +h band")
    print(f"  {'eps':>10}  {'M':>4}  {'γ_ω (deg)':>14}  {'γ_ω² (deg)':>14}  "
          f"{'γ_ω+γ_ω² (deg)':>16}  {'γ_ω−γ_ω² (deg)':>16}")
    print(f"  {'-'*10}  {'-'*4}  {'-'*14}  {'-'*14}  {'-'*16}  {'-'*16}")

    # NOTE: At points off k_P, the band is no longer exactly C_3-symmetric
    # (since the loop point isn't C_3-fixed except at center). The C_3
    # decomposition only makes sense at k_P. So we decompose at the START
    # of the loop only, then transport WITHOUT re-decomposing — that gives
    # a 2-component basis tracked smoothly.
    #
    # But wait — at the start of loop (k = k_P + eps·v0), the C_3 stabilizer
    # is broken. We can't decompose the rank-2 band at non-k_P points
    # via C_3 irreps directly.
    #
    # Alternative strategy: decompose the band at k_0 (loop start) using
    # whatever DIAGONALIZES some natural operator (e.g., C_3 lifted to
    # k_0, even if not symmetry of B(k_0)). Track these basis vectors
    # smoothly around the loop by parallel-transport.

    for eps in [0.001, 0.0001, 1e-5, 1e-6]:
        M = 32
        tri_vs = make_c3_triangle(k_P, eps)
        tri_closed = list(tri_vs) + [tri_vs[0]]
        path = discretize(tri_closed, M)

        # At loop start: use C_3-lifted decomposition (even if C_3 isn't
        # exact symmetry at k_0). The "(ω, ω²) at k_0" basis just specifies
        # initial conditions for parallel transport.
        B_0 = build_B_directed(bonds, np.array(path[0]))
        # Find the 2-dim band; diagonalize C_3 within it (treats C_3 as
        # an external operator we decompose by, not as a true symmetry).
        Q0_band = find_h_band(B_0)
        C3_in_band_0 = Q0_band.conj().T @ C3_dir @ Q0_band
        evs_c3_0, evecs_c3_0 = la.eig(C3_in_band_0)
        # Identify ω, ω²
        omega_idx = int(np.argmin(np.abs(evs_c3_0 - omega)))
        omega2_idx = int(np.argmin(np.abs(evs_c3_0 - omega2)))
        # Get ω, ω² eigenstates at k_0 in 12-dim basis
        psi_omega_0 = Q0_band @ evecs_c3_0[:, omega_idx]
        psi_omega2_0 = Q0_band @ evecs_c3_0[:, omega2_idx]
        psi_omega_0 = psi_omega_0 / la.norm(psi_omega_0)
        psi_omega2_0 = psi_omega2_0 / la.norm(psi_omega2_0)

        # Parallel-transport ω, ω² around loop separately.
        # The transport rule: at each step, project the previous eigenstate
        # onto the current band, then renormalize. This is the "U(1)
        # restriction" of the SU(2) parallel transport, projected onto
        # whatever the parallel-transported state happens to land in.
        psi_omega = psi_omega_0
        psi_omega2 = psi_omega2_0
        psi_omega_prev = psi_omega_0
        psi_omega2_prev = psi_omega2_0
        accumulated_phase_omega = 0
        accumulated_phase_omega2 = 0

        for i in range(1, len(path)):
            B_i = build_B_directed(bonds, np.array(path[i]))
            Q_band_i = find_h_band(B_i)
            P_band_i = Q_band_i @ Q_band_i.conj().T   # projector onto band

            # Transport ω-component
            psi_omega_new = P_band_i @ psi_omega_prev
            norm_ω = la.norm(psi_omega_new)
            if norm_ω > 1e-10:
                psi_omega_new = psi_omega_new / norm_ω
            # Phase accumulated this step
            overlap_ω = psi_omega_prev.conj() @ psi_omega_new
            accumulated_phase_omega += np.angle(overlap_ω)

            # Transport ω²-component
            psi_omega2_new = P_band_i @ psi_omega2_prev
            norm_ω2 = la.norm(psi_omega2_new)
            if norm_ω2 > 1e-10:
                psi_omega2_new = psi_omega2_new / norm_ω2
            overlap_ω2 = psi_omega2_prev.conj() @ psi_omega2_new
            accumulated_phase_omega2 += np.angle(overlap_ω2)

            psi_omega_prev = psi_omega_new
            psi_omega2_prev = psi_omega2_new

        # Closing overlap: from end back to start
        close_ω = psi_omega_prev.conj() @ psi_omega_0
        close_ω2 = psi_omega2_prev.conj() @ psi_omega2_0
        accumulated_phase_omega += np.angle(close_ω)
        accumulated_phase_omega2 += np.angle(close_ω2)

        γ_ω_deg = math.degrees(accumulated_phase_omega)
        γ_ω2_deg = math.degrees(accumulated_phase_omega2)
        γ_sum_deg = γ_ω_deg + γ_ω2_deg
        γ_diff_deg = γ_ω_deg - γ_ω2_deg
        # mod 360
        γ_sum_deg = ((γ_sum_deg + 180) % 360) - 180
        γ_diff_deg = ((γ_diff_deg + 180) % 360) - 180
        print(f"  {eps:>10.5g}  {M:>4d}  {γ_ω_deg:>+14.6f}  "
              f"{γ_ω2_deg:>+14.6f}  {γ_sum_deg:>+16.6f}  {γ_diff_deg:>+16.6f}")

    # -----------------------------------------------------------------------
    # Compare γ_ω − γ_ω² to structural targets
    # -----------------------------------------------------------------------
    print(f"\n  Structural target candidates (mod 360°):")
    candidates = [
        ("0", 0.0),
        ("180° = π", 180.0),
        ("2·arg(h)", 2 * ARG_H_DEG),
        ("4·arg(h)", 4 * ARG_H_DEG % 360),
        ("2π·sin(arg h)", 360 * SIN_ARG_H),
        ("90°", 90.0),
        ("270°", 270.0),
        ("120° = 2π/3", 120.0),
        ("240° = 4π/3", 240.0),
        ("256.97° (SU(2) result from earlier)", 256.97),
        ("269.03° (SU(2) converged)", 269.03),
        ("2π − 2·arg(h)", 360 - 2 * ARG_H_DEG),
    ]
    for label, val in candidates:
        # Wrap val to [-180, 180]
        val_w = ((val + 180) % 360) - 180
        print(f"    {label:<40}  {val_w:+.4f}°")

    print(f"\n" + "=" * PRINT_WIDTH)
    print(f"OK: c_3_split completed without errors")


if __name__ == "__main__":
    main()
