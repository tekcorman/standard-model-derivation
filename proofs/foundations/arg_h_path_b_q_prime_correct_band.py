#!/usr/bin/env python3
"""
arg_h_path_b_q_prime_correct_band.py — Q' on the CORRECT walker band.

Key correction. Earlier Q' attempts tracked the +h band (B-eigenvalue h
at k_P, doubly degenerate). But the C_3 structure of V_Ram is:

  ω-sector  (dim 2) contains {−h, −h̄}   (Re(λ) < 0)
  ω²-sector (dim 2) contains {+h, +h̄}   (Re(λ) > 0)
  trivial   (dim 4) contains 1 of each {±h, ±h̄}

The +h band (eigenvalue h, mult 2) thus decomposes as
  +h = (1 in trivial) ⊕ (1 in ω²)

The L photon (ω-irrep at k_P) does NOT couple to the +h band — it
couples to the ω-component of either the (-h)-band or the (-h̄)-band
(both of which have ω-character).

This script re-runs Q' Wilson holonomy on the (-h, -h̄) band (the
"left-chirality" band with negative Re part), where the ω/ω² C_3
characters live. If the SU(2) holonomy of THIS band has the structural
angle related to arg(h), Q' closes c=1 differently.

Run with:
    PYTHONPATH=. python3 proofs/foundations/arg_h_path_b_q_prime_correct_band.py
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
SIN_ARG_H = math.sqrt(5.0 / 8.0)
ARG_H = math.atan2(math.sqrt(5), math.sqrt(3))
ARG_H_DEG = math.degrees(ARG_H)
omega = np.exp(2j * math.pi / 3)
omega2 = omega.conjugate()
PRINT_WIDTH = 78


def fmt_z(z, prec=6):
    return f"{z.real:+.{prec}f}{z.imag:+.{prec}f}j"


def find_band(B_at_k, target, n_band=2):
    """Find n_band eigenstates closest to target."""
    evs, evecs = la.eig(B_at_k)
    distances = np.abs(evs - target)
    idx = np.argsort(distances)[:n_band]
    band = evecs[:, idx]
    Q, _ = la.qr(band)
    return Q[:, :n_band], evs[idx]


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


def wilson_loop(bonds, path_points, target, n_band=2):
    bands = [find_band(build_B_directed(bonds, np.array(k)), target, n_band)[0]
             for k in path_points]
    W = np.eye(n_band, dtype=complex)
    for i in range(len(path_points) - 1):
        M = bands[i].conj().T @ bands[i + 1]
        W = M @ W
    return W


def main():
    print("=" * PRINT_WIDTH)
    print("Q' on CORRECT band: -h band (which contains ω + trivial C_3 chars)")
    print("=" * PRINT_WIDTH)

    verts, lat = build_primitive_unit_cell()
    bonds = find_primitive_connectivity(verts, lat)
    edges = canonical_edges_primitive(bonds)
    k_P = np.array(K_P_RED)
    C3_dir = build_C3_directed(bonds)

    # -----------------------------------------------------------------------
    # Step 1: Verify C_3 character of each band at k_P.
    # -----------------------------------------------------------------------
    print(f"\nStep 1 — Verify C_3 character of each B-eigenvalue band at k_P")
    print(f"-" * PRINT_WIDTH)
    B_kP = build_B_directed(bonds, k_P)
    targets = {
        "+h":     H_EXACT,
        "+h̄":    H_EXACT.conjugate(),
        "-h":    -H_EXACT,
        "-h̄":   -H_EXACT.conjugate(),
    }
    for label, tgt in targets.items():
        Q_band, evs = find_band(B_kP, tgt, n_band=2)
        # Compute C_3 character of the band
        C3_in_band = Q_band.conj().T @ C3_dir @ Q_band
        evs_c3 = la.eigvals(C3_in_band)
        chars = []
        for ev in evs_c3:
            if abs(ev - 1.0) < 0.1:
                chars.append("trivial")
            elif abs(ev - omega) < 0.1:
                chars.append("ω")
            elif abs(ev - omega2) < 0.1:
                chars.append("ω²")
            else:
                chars.append(f"?({ev:.3f})")
        print(f"  Band {label}: B-eigenvalues = "
              f"{[fmt_z(e, 4) for e in evs]}, C_3 chars = {chars}")

    # -----------------------------------------------------------------------
    # Step 2: Wilson holonomy on each of the 4 bands
    # -----------------------------------------------------------------------
    print(f"\nStep 2 — Wilson SU(2) holonomy on each band, eps=1e-5, M=32")
    print(f"-" * PRINT_WIDTH)
    eps = 1e-5
    M = 32
    tri_vs = make_c3_triangle(k_P, eps)
    tri_closed = list(tri_vs) + [tri_vs[0]]
    path = discretize(tri_closed, M)

    print(f"  {'Band':>8}  {'tr(W)/2':>20}  {'|det W|':>12}  "
          f"{'SU(2) angle (deg)':>20}")
    for label, tgt in targets.items():
        W = wilson_loop(bonds, path, tgt, n_band=2)
        det_W = la.det(W)
        if abs(det_W) < 0.01:
            print(f"  {label:>8}: |det W| = {abs(det_W):.4f} (band crosses, skip)")
            continue
        sqrt_det = np.sqrt(det_W)
        W_SU2 = W / sqrt_det
        tr_SU2 = np.trace(W_SU2).real
        tr_clamped = min(max(tr_SU2 / 2, -1), 1)
        theta_SU2 = 2 * math.acos(tr_clamped)
        print(f"  {label:>8}  {tr_SU2/2:>+20.10f}  {abs(det_W):>12.6f}  "
              f"{math.degrees(theta_SU2):>+20.6f}")

    # -----------------------------------------------------------------------
    # Step 3: Wilson holonomy on the COMBINED (h + (-h̄)) band that hosts
    # ω in V_Ram. This is rank 2 with 1 ω-state (in -h) and 1 ω²-state (in +h).
    # Wait, ω-sector at k_P contains BOTH -h and -h̄, both 1-dim. So the
    # ω-sector's full band is rank 2 (-h + -h̄). Let's track that.
    # -----------------------------------------------------------------------
    print(f"\nStep 3 — Wilson SU(2) holonomy on the FULL ω-sector at k_P (-h, -h̄)")
    print(f"-" * PRINT_WIDTH)
    print(f"  ω-sector at k_P spans both -h (1 state) and -h̄ (1 state)")
    print(f"  Track the rank-2 ω-irrep band around the loop.")

    # ω-sector at k_P: extract by combining -h and -h̄ eigenstates that are in ω.
    Q_minus_h, _ = find_band(B_kP, -H_EXACT, n_band=2)
    # Within -h band, find the ω-irrep state
    C3_in_minh = Q_minus_h.conj().T @ C3_dir @ Q_minus_h
    evs_c3_minh, evecs_c3_minh = la.eig(C3_in_minh)
    omega_idx = int(np.argmin(np.abs(evs_c3_minh - omega)))
    omega_state_in_minh = Q_minus_h @ evecs_c3_minh[:, omega_idx]
    omega_state_in_minh = omega_state_in_minh / la.norm(omega_state_in_minh)
    print(f"  -h band's ω-component at k_P: "
          f"⟨ψ|C3|ψ⟩ = {fmt_z(omega_state_in_minh.conj() @ C3_dir @ omega_state_in_minh, 4)}")

    Q_minus_hbar, _ = find_band(B_kP, -H_EXACT.conjugate(), n_band=2)
    C3_in_minhbar = Q_minus_hbar.conj().T @ C3_dir @ Q_minus_hbar
    evs_c3_minhbar, evecs_c3_minhbar = la.eig(C3_in_minhbar)
    # -h̄ band's ω-component
    omega_idx_hbar = int(np.argmin(np.abs(evs_c3_minhbar - omega)))
    omega_state_in_minhbar = Q_minus_hbar @ evecs_c3_minhbar[:, omega_idx_hbar]
    omega_state_in_minhbar = omega_state_in_minhbar / la.norm(omega_state_in_minhbar)
    print(f"  -h̄ band's ω-component at k_P: "
          f"⟨ψ|C3|ψ⟩ = {fmt_z(omega_state_in_minhbar.conj() @ C3_dir @ omega_state_in_minhbar, 4)}")

    # Build the ω-irrep 2-dim subspace at k_P (1 from -h, 1 from -h̄)
    omega_irrep_kP = np.column_stack([omega_state_in_minh, omega_state_in_minhbar])
    Q_omega_irrep_kP, _ = la.qr(omega_irrep_kP)
    Q_omega_irrep_kP = Q_omega_irrep_kP[:, :2]

    # Now parallel-transport this 2-dim subspace around the loop.
    # Approach: at each k_i, project the previous basis onto the lowest-eigenvalue
    # 2-dim band that PROJECTS LARGEST onto the previous basis. (Tracks
    # the "ω-irrep band" smoothly.)
    bands_omega = [Q_omega_irrep_kP]
    Q_prev = Q_omega_irrep_kP
    for k in path[1:]:
        B_k = build_B_directed(bonds, np.array(k))
        evs, evecs = la.eig(B_k)
        # Project Q_prev onto each eigenvector; take the 2 with largest projection
        projections = np.abs(evecs.conj().T @ Q_prev).sum(axis=1)
        idx = np.argsort(-projections)[:2]
        Q_new = evecs[:, idx]
        Q_new, _ = la.qr(Q_new)
        Q_new = Q_new[:, :2]
        bands_omega.append(Q_new)
        Q_prev = Q_new

    # Compute Wilson holonomy
    W_omega = np.eye(2, dtype=complex)
    for i in range(len(path) - 1):
        Mi = bands_omega[i].conj().T @ bands_omega[i + 1]
        W_omega = Mi @ W_omega
    M_close = bands_omega[-1].conj().T @ bands_omega[0]
    W_omega = M_close @ W_omega
    det_W_omega = la.det(W_omega)
    sqrt_det = np.sqrt(det_W_omega)
    if abs(det_W_omega) > 0.01:
        W_omega_SU2 = W_omega / sqrt_det
        tr = np.trace(W_omega_SU2).real
        tr_clamp = min(max(tr / 2, -1), 1)
        theta = 2 * math.acos(tr_clamp)
        print(f"\n  ω-sector Wilson holonomy:")
        print(f"    tr(W)/2 = {tr/2:+.10f}")
        print(f"    |det W| = {abs(det_W_omega):.6f}")
        print(f"    SU(2) angle = {math.degrees(theta):+.6f}°")
    else:
        print(f"  ω-sector tracking failed (|det W| = {abs(det_W_omega):.4f})")

    # -----------------------------------------------------------------------
    # Step 4: Compare to all bands' Wilson holonomies and to predicted
    # structural angles
    # -----------------------------------------------------------------------
    print(f"\n  Predicted structural angle candidates:")
    print(f"    arg(h)      = {ARG_H_DEG:.4f}°")
    print(f"    2·arg(h)    = {2*ARG_H_DEG:.4f}°")
    print(f"    π − arg(h)  = {180 - ARG_H_DEG:.4f}°")
    print(f"    π + arg(h)  = {180 + ARG_H_DEG:.4f}°")
    print(f"    2π·sin(arg h) = {360*SIN_ARG_H:.4f}°")
    print(f"    sin(arg h)·360° = {360*SIN_ARG_H:.4f}°")

    print(f"\n" + "=" * PRINT_WIDTH)
    print(f"OK: q_prime_correct_band completed without errors")


if __name__ == "__main__":
    main()
