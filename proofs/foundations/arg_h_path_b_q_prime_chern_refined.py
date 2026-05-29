#!/usr/bin/env python3
"""
arg_h_path_b_q_prime_chern_refined.py — high-precision numerical Berry-charge
estimate using the FULL non-Hermitian B(k) eigenvectors (bi-orthogonal).

Method. For each k = k_P + R·n̂(θ, φ) on a small 2-sphere of radius R,
compute the rank-2 spectral projector P_+h(k) onto eigenvalues closest
to h. Then compute first Chern number on the sphere via:

  c_1 = (i/2π) ∫_S² Tr(P · dP ∧ dP)

discretized as a sum of "Wilson plaquettes" log[Tr(P_n P_e P_s P_w P_n)]
for each plaquette. This is the standard non-Abelian Chern formula
(Resta-Sgiarovello-Smirnov), valid for non-normal P (uses spectral
projector, not orthogonal).

Goal. Settle whether the numerical 1.2570 is exactly 5/4 (Im(h)² = 1.25)
or something else. Compare against symbolic targets:
  5/4         = 1.25
  φ²/2        = 1.309 (golden ratio squared / 2)
  4(φ² − 1)/(...)
  etc.
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
)
from srs_photon_c3_chainmap import K_P_RED
from srs_photon_chirality_coefficient import build_B_directed


H_EXACT = (math.sqrt(3) + 1j * math.sqrt(5)) / 2


def spectral_projector(B, lam, eigvals, eigvecs):
    """Project onto the eigenspace of B closest to eigenvalue lam, dim 2.

    Uses the spectral projector via Lagrange interpolation when the
    distinct eigenvalue pattern is known. For B near a degenerate point,
    the 2 closest eigenvalues to lam form the band.
    """
    # Find the 2 indices closest to lam.
    dists = np.abs(eigvals - lam)
    idx = np.argsort(dists)[:2]
    V = eigvecs[:, idx]   # 12×2 right eigenvectors (non-orthogonal in general)
    # Find left eigenvectors via biorthogonal partner: W·B = λ W means W = (eigvecs of B^T at λ)†
    # Or equivalently, W = (V^(-1))[idx, :] for the full eigendecomposition.
    # Using full eigendecomposition: B = V_full · D · V_full^(-1)
    # P = V_full[:, idx] @ V_full^(-1)[idx, :]
    V_full_inv = la.inv(eigvecs)
    P = eigvecs[:, idx] @ V_full_inv[idx, :]
    return P


def chern_via_wilson(bonds, k_P, R, N_theta, N_phi):
    """Compute first Chern number via Wilson-plaquette product on small sphere.

    Standard non-Abelian Chern (Fukui-Hatsugai-Suzuki):
      c_1 = (1/2π) Σ_plaquette Im[log det U_plaq]
    where U_plaq = U_n,e · U_e,s · U_s,w · U_w,n,
    U_a,b = det(P_a · P_b · P_a) approximated by det(orthonormalized overlap).

    But here we want the FULL non-Abelian (rank-2) Chern, computed via
    overlap matrices instead of det.
    """
    # Build mesh on sphere of radius R around k_P.
    # Parametrize with (θ, φ) ∈ ([0, π], [0, 2π)).
    # Compute spectral projector P_+h at each grid point.
    # Then compute Wilson plaquette holonomies and sum their imaginary log-det.
    P_grid = np.empty((N_theta + 1, N_phi), dtype=object)
    for i in range(N_theta + 1):
        theta = math.pi * i / N_theta
        for j in range(N_phi):
            phi = 2 * math.pi * j / N_phi
            n_hat = np.array([math.sin(theta) * math.cos(phi),
                              math.sin(theta) * math.sin(phi),
                              math.cos(theta)])
            k = np.array(k_P) + R * n_hat
            B = build_B_directed(bonds, k)
            evs, evecs = la.eig(B)
            P = spectral_projector(B, H_EXACT, evs, evecs)
            P_grid[i, j] = P

    # Wilson plaquette: at each (i, j) plaquette spanning (i, j), (i+1, j),
    # (i+1, j+1), (i, j+1).
    chern_sum = 0.0
    for i in range(N_theta):
        for j in range(N_phi):
            j_next = (j + 1) % N_phi
            P_a = P_grid[i, j]
            P_b = P_grid[i, j_next]
            P_c = P_grid[i + 1, j_next]
            P_d = P_grid[i + 1, j]
            # Wilson holonomy = trace of plaquette projector product
            U = P_a @ P_b @ P_c @ P_d @ P_a
            tr_U = np.trace(U)
            # log of trace gives log of "det" in projector space.
            # For rank-2 P, tr(U) is the Wilson holonomy contribution.
            phase = np.angle(tr_U)
            chern_sum += phase
    chern = chern_sum / (2 * math.pi)
    return chern


def chern_via_sigma_winding(bonds, k_P, R, N_theta, N_phi):
    """Compute σ-winding via Hermitian-part decomposition on small sphere.

    Σ̂_H(δk) = (h_x, h_y, h_z) of (M(δk) + M(δk)†)/2 where M = Q† ∂B Q.
    Then winding = (1/4π) Σ over triangles of σ̂·(σ̂_b × σ̂_c) where
    a, b, c are σ̂-values at triangle vertices.

    This matches the q_prime_perturbation.py method but with finer mesh.
    """
    # First, build orthonormal Q at k_P using QR on +h eigenvectors.
    B_kP = build_B_directed(bonds, np.array(k_P))
    evs_kP, evecs_kP = la.eig(B_kP)
    h_idx = [i for i, e in enumerate(evs_kP) if abs(e - H_EXACT) < 1e-10]
    assert len(h_idx) == 2
    Q, _ = la.qr(evecs_kP[:, h_idx])
    Q = Q[:, :2]

    # Compute ∂B/∂k_a at k_P.
    eps = 1e-7
    dB = []
    for axis in range(3):
        kp = np.array(k_P, dtype=float).copy()
        km = np.array(k_P, dtype=float).copy()
        kp[axis] += eps; km[axis] -= eps
        dB.append((build_B_directed(bonds, kp) - build_B_directed(bonds, km)) / (2 * eps))

    # M^a = Q† · dB^a · Q
    M_per_axis = [Q.conj().T @ dB[a] @ Q for a in range(3)]
    # σ_H per axis: (h_x, h_y, h_z) of (M + M†)/2
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    Lambda = np.zeros((3, 3), dtype=float)
    for a in range(3):
        M_H = (M_per_axis[a] + M_per_axis[a].conj().T) / 2
        # Pauli decomp
        h_x = np.trace(M_H @ sigma_x).real / 2
        h_y = np.trace(M_H @ sigma_y).real / 2
        h_z = np.trace(M_H @ sigma_z).real / 2
        Lambda[:, a] = (h_x, h_y, h_z)

    # σ̂_H field on sphere
    sigma_field = np.zeros((N_theta + 1, N_phi, 3))
    for i in range(N_theta + 1):
        theta = math.pi * i / N_theta
        for j in range(N_phi):
            phi = 2 * math.pi * j / N_phi
            n_hat = np.array([math.sin(theta) * math.cos(phi),
                              math.sin(theta) * math.sin(phi),
                              math.cos(theta)])
            sigma = Lambda @ n_hat
            mag = np.linalg.norm(sigma)
            if mag > 1e-15:
                sigma_field[i, j] = sigma / mag

    # Sum signed solid angles via Van Oosterom-Strang formula.
    chern = 0.0
    for i in range(N_theta):
        for j in range(N_phi):
            j_next = (j + 1) % N_phi
            v1 = sigma_field[i, j]
            v2 = sigma_field[i, j_next]
            v3 = sigma_field[i + 1, j_next]
            v4 = sigma_field[i + 1, j]
            for tri in [(v1, v2, v3), (v1, v3, v4)]:
                a_v, b_v, c_v = tri
                # Van Oosterom-Strang signed solid angle
                num = a_v[0] * (b_v[1] * c_v[2] - b_v[2] * c_v[1])
                num -= a_v[1] * (b_v[0] * c_v[2] - b_v[2] * c_v[0])
                num += a_v[2] * (b_v[0] * c_v[1] - b_v[1] * c_v[0])
                denom = 1.0 + np.dot(a_v, b_v) + np.dot(b_v, c_v) + np.dot(a_v, c_v)
                # signed solid angle Ω = 2 atan2(num, denom)
                omega = 2 * math.atan2(num, denom)
                chern += omega
    return chern / (4 * math.pi), Lambda


def main():
    print("=" * 78)
    print("Q' Berry-charge refined numerical (Wilson + σ-winding)")
    print("=" * 78)

    verts, lat = build_primitive_unit_cell()
    bonds = find_primitive_connectivity(verts, lat)
    # Keep 4-tuple format for build_B_directed compatibility.
    k_P = list(K_P_RED)

    print("\n— σ_H winding (Van Oosterom-Strang exact spherical solid angle) —")
    for R, N_theta, N_phi in [(1e-3, 16, 32), (1e-4, 32, 64), (1e-5, 60, 120),
                              (1e-6, 60, 120), (1e-7, 120, 240)]:
        c, Lam = chern_via_sigma_winding(bonds, k_P, R, N_theta, N_phi)
        print(f"  R = {R:.0e}, mesh = {N_theta}×{N_phi}: c_σH = {c:+.6f}")
    print(f"\n  Numerical Λ_H matrix:")
    for row in Lam:
        print(f"    " + "  ".join(f"{x:+.6f}" for x in row))
    print(f"  det(Λ_H) numerical = {np.linalg.det(Lam):.6f}")
    print(f"  Compare symbolic det(Λ_H) = 2π³(7√3 − 3√15)/243 ≈ "
          f"{2 * math.pi**3 * (7*math.sqrt(3) - 3*math.sqrt(15)) / 243:.6f}")

    print("\n— Non-Abelian Chern via Wilson plaquettes (full non-Hermitian) —")
    for R, N_theta, N_phi in [(1e-3, 16, 32), (1e-4, 32, 64), (1e-5, 60, 120)]:
        try:
            c = chern_via_wilson(bonds, k_P, R, N_theta, N_phi)
            print(f"  R = {R:.0e}, mesh = {N_theta}×{N_phi}: c_Wilson = {c:+.6f}")
        except Exception as e:
            print(f"  R = {R:.0e}: failed ({e})")

    print("\n— Identification candidates —")
    print(f"  5/4  = {5/4:.6f}")
    print(f"  φ²/2 = {((1+math.sqrt(5))/2)**2 / 2:.6f}")
    print(f"  3π/2 / 4π = 3/8 = {3/8:.6f}")
    print(f"  (3 − √5)/π² · const?")

    print(f"\n" + "=" * 78)
    print("OK")


if __name__ == "__main__":
    main()
