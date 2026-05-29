#!/usr/bin/env python3
"""
arg_h_path_b_q_prime_perturbation.py — analytic Bloch perturbation for the
+h band at k_P, to derive the SU(2) Berry curvature symbolically and
extract the topological invariant.

Background. Q' Wilson loop gave SU(2) angle ≈ 269° per BZ loop around k_P.
The exact converged value isn't matching obvious structural targets
(2π·sin(arg h), 2π−2·arg(h), 249.85°, etc.). Maybe a structural derivation
will reveal what 269° actually represents.

Method. At k_P, the +h band is doubly degenerate. Off k_P, B(k) splits
the degeneracy linearly: ΔB ≈ (∂_k B|_{k_P})·δk. Within the +h band
subspace, the splitting is encoded by the 2×2 reduced matrix:

  H_eff(δk) = P_+h · (∂_k B · δk) · P_+h    on the rank-2 +h subspace

H_eff is a 2×2 traceless matrix (since the trace shift is just a shift
of band center). Its eigenvalues are ±|H_eff|, splitting the degeneracy
linearly in δk.

The non-Abelian Berry curvature on the band is
  F_ab = i [P, ∂_a P, ∂_b P]    where P = projector onto +h band.

For a band crossing with effective Hamiltonian H_eff = δk·σ (a Pauli
σ-vector decomposition), the Berry curvature has a known monopole-like
structure with charge ±1 per pair of bands. For us, the rank-2 band
crossing is more complex: it's like a HIGHER-ORDER monopole.

This script computes H_eff symbolically (using the framework's exact
B(k) at k_P + δk to first order in δk), extracts the σ-vector
decomposition, and derives the topological invariant of the SU(2)
holonomy around a small loop encircling the crossing.

Run with:
    PYTHONPATH=. python3 proofs/foundations/arg_h_path_b_q_prime_perturbation.py
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


def main():
    print("=" * PRINT_WIDTH)
    print("Q' analytic perturbation: H_eff for +h band crossing at k_P")
    print("=" * PRINT_WIDTH)

    verts, lat = build_primitive_unit_cell()
    bonds = find_primitive_connectivity(verts, lat)
    edges = canonical_edges_primitive(bonds)
    k_P = np.array(K_P_RED)

    # -----------------------------------------------------------------------
    # Step 1: Find the +h band's 2-dim subspace at k_P explicitly.
    # -----------------------------------------------------------------------
    print(f"\nStep 1 — +h band subspace at k_P")
    print(f"-" * PRINT_WIDTH)
    B_kP = build_B_directed(bonds, k_P)
    evs_kP, evecs_kP = la.eig(B_kP)
    h_indices = [i for i, ev in enumerate(evs_kP) if abs(ev - H_EXACT) < 1e-10]
    print(f"  Found {len(h_indices)} eigenvectors with eigenvalue h at k_P")
    assert len(h_indices) == 2

    # Orthonormalize the +h subspace
    Q_h = evecs_kP[:, h_indices]
    Q_h, _ = la.qr(Q_h)
    Q_h = Q_h[:, :2]   # 12 × 2 orthonormal basis for +h band

    # Compute ∂_k B at k_P for each axis
    eps = 1e-7
    dB_dk = []
    for axis in range(3):
        kp = list(k_P).copy(); km = list(k_P).copy()
        kp[axis] += eps; km[axis] -= eps
        Bp = build_B_directed(bonds, np.array(kp))
        Bm = build_B_directed(bonds, np.array(km))
        dB_dk.append((Bp - Bm) / (2 * eps))

    # -----------------------------------------------------------------------
    # Step 2: First-order perturbation: H_eff(δk) = Q_h† · (Σ_a δk_a · ∂_a B) · Q_h
    # This gives a 2×2 matrix per direction.
    # -----------------------------------------------------------------------
    print(f"\nStep 2 — H_eff per axis (2×2 in +h band basis)")
    print(f"-" * PRINT_WIDTH)
    H_eff_per_axis = []
    for axis in range(3):
        H_a = Q_h.conj().T @ dB_dk[axis] @ Q_h
        H_eff_per_axis.append(H_a)
        print(f"\n  ∂_a H_eff for axis {axis}:")
        for row in H_a:
            print("    " + "  ".join(f"{x.real:+.4f}{x.imag:+.4f}j" for x in row))
        # Diagonal: shifts the band center; off-diagonal: splits the degeneracy
        tr_H = np.trace(H_a)
        traceless = H_a - (tr_H / 2) * np.eye(2)
        print(f"    tr(H_a) = {fmt_z(tr_H, 4)}, |traceless part| = "
              f"{la.norm(traceless):.4f}")

    # -----------------------------------------------------------------------
    # Step 3: Express H_eff in σ-basis (decompose 2×2 traceless Hermitian as
    # H = h_x σ_x + h_y σ_y + h_z σ_z). This is the Berry monopole
    # representation.
    # -----------------------------------------------------------------------
    print(f"\nStep 3 — Decompose H_eff in Pauli σ-basis")
    print(f"-" * PRINT_WIDTH)

    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    def decompose_2x2(H):
        """Decompose 2×2 Hermitian H as h_0 I + h_x σ_x + h_y σ_y + h_z σ_z.

        Returns h_0, h_x, h_y, h_z (real for Hermitian H).
        """
        # Ensure Hermitian
        H_h = (H + H.conj().T) / 2
        h_0 = np.trace(H_h).real / 2
        h_x = (H_h[0, 1] + H_h[1, 0]).real / 2
        h_y = (H_h[1, 0] - H_h[0, 1]).imag / (-2)   # σ_y = [[0,-i],[i,0]]
        h_z = (H_h[0, 0] - H_h[1, 1]).real / 2
        return h_0, h_x, h_y, h_z

    # H_eff is generally not Hermitian (B is not Hermitian), so decompose
    # into anti-Hermitian + Hermitian parts.
    print(f"\n  Decomposition of (H_eff + H_eff†)/2 (Hermitian part):")
    for axis in range(3):
        H_h = (H_eff_per_axis[axis] + H_eff_per_axis[axis].conj().T) / 2
        h_0, h_x, h_y, h_z = decompose_2x2(H_h)
        norm_sigma = math.sqrt(h_x**2 + h_y**2 + h_z**2)
        print(f"    axis {axis}: h_0 = {h_0:+.4f}, σ-vector = "
              f"({h_x:+.4f}, {h_y:+.4f}, {h_z:+.4f}), |σ| = {norm_sigma:.4f}")

    print(f"\n  Decomposition of (H_eff − H_eff†)/(2i) (anti-Hermitian part / i):")
    for axis in range(3):
        H_a = (H_eff_per_axis[axis] - H_eff_per_axis[axis].conj().T) / (2j)
        h_0, h_x, h_y, h_z = decompose_2x2(H_a)
        norm_sigma = math.sqrt(h_x**2 + h_y**2 + h_z**2)
        print(f"    axis {axis}: h_0 = {h_0:+.4f}, σ-vector = "
              f"({h_x:+.4f}, {h_y:+.4f}, {h_z:+.4f}), |σ| = {norm_sigma:.4f}")

    # -----------------------------------------------------------------------
    # Step 4: Compute "Berry monopole charge" of the +h band crossing.
    # For a 2-band crossing, this is the Chern number of the σ-vector map
    # from a small 2-sphere around the crossing to S² (= unit sphere of σ-space).
    # -----------------------------------------------------------------------
    print(f"\nStep 4 — Estimate Berry monopole charge from σ-vector winding")
    print(f"-" * PRINT_WIDTH)

    # Build the σ-vector field on a small 2-sphere around k_P
    # parameterized by (u, v) ∈ [0, 1]² for some small radius.
    # For each (u, v), compute σ-vector of H_eff(δk) and map to unit sphere.
    # The Chern number = (1/4π) ∫ dA · sign(d^a σ × d^b σ) (winding number).

    R = 1e-4   # small sphere radius
    n_theta, n_phi = 16, 32
    # On a sphere of radius R centered at k_P:
    # δk(θ, φ) = R · (sin θ cos φ, sin θ sin φ, cos θ)
    # H_eff(θ, φ) = sum_axes δk_a · H_eff_per_axis[a]

    # For each grid point, compute H_eff(δk) and decompose into σ-vector
    # (using the Hermitian part — most relevant for Berry curvature).
    sigma_field = np.zeros((n_theta, n_phi, 3))
    for i in range(n_theta):
        theta = math.pi * (i + 0.5) / n_theta
        for j in range(n_phi):
            phi = 2 * math.pi * j / n_phi
            dk = R * np.array([math.sin(theta) * math.cos(phi),
                                math.sin(theta) * math.sin(phi),
                                math.cos(theta)])
            H_total = sum(dk[a] * H_eff_per_axis[a] for a in range(3))
            H_h = (H_total + H_total.conj().T) / 2
            _, h_x, h_y, h_z = decompose_2x2(H_h)
            mag = math.sqrt(h_x**2 + h_y**2 + h_z**2)
            if mag > 1e-15:
                sigma_field[i, j] = np.array([h_x, h_y, h_z]) / mag

    # Compute winding number (Chern) by triangulating the sphere and
    # summing signed solid-angle contributions.
    chern = 0.0
    for i in range(n_theta - 1):
        for j in range(n_phi):
            j_next = (j + 1) % n_phi
            v1 = sigma_field[i, j]
            v2 = sigma_field[i, j_next]
            v3 = sigma_field[i + 1, j]
            v4 = sigma_field[i + 1, j_next]
            # Two triangles per quad: (v1, v2, v3) and (v2, v4, v3)
            for tri in [(v1, v2, v3), (v2, v4, v3)]:
                a, b, c = tri
                # Signed solid angle (Eriksson formula)
                cross = np.cross(b - a, c - a)
                triple = np.dot(a, cross)
                chern += triple
    # Normalize to Chern number (oriented spherical winding)
    chern_estimate = chern / (4 * math.pi)
    print(f"  σ-vector winding around small 2-sphere at k_P: "
          f"{chern_estimate:.4f}")
    print(f"  (Should be integer for a true Berry monopole; non-integer = "
          f"non-monopole topology)")

    # -----------------------------------------------------------------------
    # Step 5: Closer look at H_eff structure
    # -----------------------------------------------------------------------
    print(f"\nStep 5 — Examine H_eff structural pattern")
    print(f"-" * PRINT_WIDTH)
    print(f"  H_eff(δk = (1,0,0)):")
    H1 = H_eff_per_axis[0]
    for row in H1:
        print(f"    " + "  ".join(f"{x.real:+.4f}{x.imag:+.4f}j" for x in row))
    print(f"\n  H_eff(δk = (1,1,1)/√3):  (along C_3 axis)")
    H_c3 = (H_eff_per_axis[0] + H_eff_per_axis[1] + H_eff_per_axis[2]) / math.sqrt(3)
    for row in H_c3:
        print(f"    " + "  ".join(f"{x.real:+.4f}{x.imag:+.4f}j" for x in row))
    print(f"\n  Singular values of H_eff(δk = (1,1,1)/√3): "
          f"{la.svd(H_c3, compute_uv=False)}")

    print(f"\n" + "=" * PRINT_WIDTH)
    print(f"OK: q_prime_perturbation completed without errors")


if __name__ == "__main__":
    main()
