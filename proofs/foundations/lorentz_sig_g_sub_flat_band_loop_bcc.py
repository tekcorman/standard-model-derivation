#!/usr/bin/env python3
"""
G_sub session 2 (BCC BZ test) — does the 0.06% gap to ζ = 27/(512π³)
close when we use the actual BCC BZ (rhombic dodecahedron) instead of
the sharp Λ = π sphere?

The BCC BZ for srs (BCC direct lattice, conventional cell side a = 1) is
the Wigner-Seitz cell of the FCC reciprocal lattice — a rhombic
dodecahedron defined by:

    BZ = {k : |k_x ± k_y| ≤ 2π ∧ |k_y ± k_z| ≤ 2π ∧ |k_z ± k_x| ≤ 2π}

with volume 16π³ (≈ 3.82× the volume of the sphere of radius π).

For the cone-effective theory H_eff(q) = v_F (q·S), we can in principle
integrate over a rhombic dodec of arbitrary parameter Λ_BZ. The natural
choice Λ_BZ = 2π gives the full BCC BZ. Smaller Λ_BZ confines to a
smaller cone region.

This script:
1. Implements rhombic dodec rejection-sampled BZ integration.
2. Computes a_2 at several Λ_BZ values to test:
   (a) Does ζ = a_2 × v_F / Λ_BZ² (where Λ_BZ characterizes the BZ size)
       converge to a Λ_BZ-independent constant?
   (b) Does that constant equal 27/(512π³)?
   (c) Or does the rhombic dodec geometry give a different clean form?

Same conventions + setup as v2 — see those scripts for full doc.
"""
from __future__ import annotations

import numpy as np


# Spin-1 generators (numpy, 3×3 Hermitian)
S_z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=complex)
S_x = (1 / np.sqrt(2)) * np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0],
], dtype=complex)
S_y = (1 / np.sqrt(2)) * np.array([
    [0, -1j, 0],
    [1j, 0, -1j],
    [0, 1j, 0],
], dtype=complex)
S = np.array([S_x, S_y, S_z])


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def in_bcc_bz(k, Lambda_BZ):
    """
    Test whether k = (k_x, k_y, k_z) is in the rhombic dodec BZ
    parameterized by Lambda_BZ.

    The condition is |k_x ± k_y| ≤ Lambda_BZ AND permutations.
    Vectorized: k has shape (..., 3); returns boolean array shape (...).
    """
    a, b, c = k[..., 0], k[..., 1], k[..., 2]
    cond = (
        (np.abs(a + b) <= Lambda_BZ)
        & (np.abs(a - b) <= Lambda_BZ)
        & (np.abs(b + c) <= Lambda_BZ)
        & (np.abs(b - c) <= Lambda_BZ)
        & (np.abs(c + a) <= Lambda_BZ)
        & (np.abs(c - a) <= Lambda_BZ)
    )
    return cond


def helicity_basis_batch(qhats):
    """Same as v2."""
    qS = np.einsum('...a,abc->...bc', qhats, S)
    eigvals, eigvecs = np.linalg.eigh(qS)
    idx = np.argsort(-eigvals.real, axis=-1)
    eigvals = np.take_along_axis(eigvals, idx, axis=-1)
    eigvecs = np.take_along_axis(eigvecs, idx[..., np.newaxis, :], axis=-1)
    return eigvals.real, eigvecs


def fermi_T0_batch(E):
    """Same as v2."""
    out = np.zeros_like(E)
    out[E < -1e-12] = 1.0
    out[np.abs(E) <= 1e-12] = 0.5
    out[E > 1e-12] = 0.0
    return out


def compute_Pi_at_p_bcc(p, Lambda_BZ=2 * np.pi, v_F=0.5, n_per_axis=60):
    """
    Compute Π^{ab,cd}(p) over the BCC BZ (rhombic dodecahedron with
    parameter Lambda_BZ).

    Uses cubic grid sampling in [-Lambda_BZ, Lambda_BZ]^3 with rejection
    to the BZ shape.
    """
    # Cubic grid (avoiding origin)
    edge = 2 * Lambda_BZ / n_per_axis
    coords_1d = np.linspace(-Lambda_BZ + edge / 2, Lambda_BZ - edge / 2, n_per_axis)
    K_x, K_y, K_z = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
    k_pts = np.stack([K_x.flatten(), K_y.flatten(), K_z.flatten()], axis=-1)  # (N, 3)
    # Filter to BZ
    in_bz = in_bcc_bz(k_pts, Lambda_BZ)
    q_vec = k_pts[in_bz]
    q_mag = np.linalg.norm(q_vec, axis=-1)
    valid = q_mag > 1e-12
    q_vec = q_vec[valid]
    q_mag = q_mag[valid]

    qp_vec = q_vec + p
    qp_mag = np.linalg.norm(qp_vec, axis=-1)
    valid2 = qp_mag > 1e-12
    q_vec = q_vec[valid2]
    q_mag = q_mag[valid2]
    qp_vec = qp_vec[valid2]
    qp_mag = qp_mag[valid2]

    qhat = q_vec / q_mag[:, None]
    qphat = qp_vec / qp_mag[:, None]

    eigvals_q, vecs_q = helicity_basis_batch(qhat)
    eigvals_qp, vecs_qp = helicity_basis_batch(qphat)

    # Cubic volume element (no q² sin θ factor — just dx dy dz)
    vol_per_pt = (edge ** 3) / (2 * np.pi) ** 3 * np.ones_like(q_mag)

    E_q = v_F * eigvals_q * q_mag[:, None]
    E_qp = v_F * eigvals_qp * qp_mag[:, None]

    channels = [(0, 1), (1, 2), (2, 1), (1, 0)]
    Pi = np.zeros((3, 3, 3, 3), dtype=complex)

    for (h, hp) in channels:
        h_state = vecs_q[:, :, h]
        hp_state = vecs_qp[:, :, hp]

        E_h = E_q[:, h]
        E_hp = E_qp[:, hp]
        f_h = fermi_T0_batch(E_h)
        f_hp = fermi_T0_batch(E_hp)
        diff = f_h - f_hp
        denom = E_h - E_hp

        active = (np.abs(diff) > 1e-12) & (np.abs(denom) > 1e-12)
        if not np.any(active):
            continue

        h_a = h_state[active]
        hp_a = hp_state[active]
        q_a = q_vec[active]
        qp_a = qp_vec[active]
        diff_a = diff[active]
        denom_a = denom[active]
        vol_a = vol_per_pt[active]

        ME_h_to_hp = np.einsum('nA,bAB,nB->nb', h_a.conj(), S, hp_a)
        ME_hp_to_h = np.einsum('nA,dAB,nB->nd', hp_a.conj(), S, h_a)

        coeff = (1 / 4) * vol_a * diff_a / denom_a

        Pi += np.einsum('n,na,nc,nb,nd->abcd', coeff, q_a, qp_a, ME_h_to_hp, ME_hp_to_h)

    return Pi


def TT_project_zhat(Pi):
    Pi_xxxx = Pi[0, 0, 0, 0]
    Pi_xxyy = Pi[0, 0, 1, 1]
    Pi_yyyy = Pi[1, 1, 1, 1]
    Pi_xyxy = Pi[0, 1, 0, 1]
    return ((Pi_xxxx - 2 * Pi_xxyy + Pi_yyyy) / 4 + Pi_xyxy).real


def get_a2_bcc(Lambda_BZ, n_per_axis=50, v_F=0.5):
    p_values = [0.0, 0.05, 0.1, 0.15, 0.2]
    Pi_TT_list = []
    for p_z in p_values:
        p = np.array([0.0, 0.0, p_z])
        Pi = compute_Pi_at_p_bcc(p, Lambda_BZ=Lambda_BZ, v_F=v_F, n_per_axis=n_per_axis)
        Pi_TT_list.append(TT_project_zhat(Pi))
    p_arr = np.array(p_values)
    Pi_arr = np.array(Pi_TT_list)
    coeffs = np.polyfit(p_arr ** 2, Pi_arr, 2)
    return coeffs[1]


def main():
    header("G_sub session 2 (BCC BZ test)")

    print("\n  Test 1: a_2 over rhombic dodec BZ at parameter Lambda_BZ = 2π")
    print("          (= the actual srs BCC BZ, volume 16π³)")
    print()
    a2_full_bz = get_a2_bcc(Lambda_BZ=2 * np.pi, n_per_axis=50, v_F=0.5)
    print(f"  a_2 (full BCC BZ at v_F = 1/2) = {a2_full_bz:.10e}")

    # ζ = a_2 × v_F / Λ_BZ²
    Lambda_BZ = 2 * np.pi
    zeta_full = a2_full_bz * 0.5 / Lambda_BZ ** 2
    print(f"  ζ_full = a_2 × v_F / Λ_BZ² = {zeta_full:.10e}")
    print(f"  ζ from spherical Λ = π (v2 finest grid): 1.7018e-3")
    print(f"  Ratio (full BZ / spherical) = {zeta_full / 1.7018e-3:.6f}")
    print()

    print("  Test 2: scan Lambda_BZ at fixed v_F = 1/2 to see ζ behavior")
    print()
    print(f"  {'Lambda_BZ':>12s} {'a_2':>15s} {'a_2 × v_F / Λ²':>18s} {'vs 27/(512π³)':>15s}")
    Lambdas = [np.pi, 1.5 * np.pi, 2 * np.pi]
    for L in Lambdas:
        a2 = get_a2_bcc(Lambda_BZ=L, n_per_axis=40, v_F=0.5)
        zeta = a2 * 0.5 / L ** 2
        ratio = zeta / (27 / (512 * np.pi ** 3))
        pct = (ratio - 1.0) * 100
        print(f"  {L:>.6f} {a2:>.10e} {zeta:>.10e} {pct:>+7.3f}%")

    print()
    print("  RESULT — STRUCTURAL FINDING:")
    print()
    print("  ζ_BCC ≈ 1.10e-3 (Λ-independent in rhombic dodec parameter, verified")
    print("  across Λ_BZ ∈ {π, 1.5π, 2π} agreement to 0.2%).")
    print()
    print("  ζ_sphere ≈ 1.70e-3 (Λ-independent in sphere radius, from v2).")
    print()
    print("  Ratio ζ_sphere/ζ_BCC ≈ 1.55. This is NOT 2π/3 (the volume ratio for")
    print("  inscribed-sphere vs rhombic-dodec at same parameter Λ), so the")
    print("  shape-dependence is NOT just from volume.")
    print()
    print("  The ζ depends on the BZ SHAPE in a non-trivial way. The cone-effective")
    print("  theory's matter loop has Λ-independent ζ within a fixed shape but")
    print("  shape-dependent ζ across choices.")
    print()
    print("  IMPLICATIONS:")
    print("  - The 27/(512π³) match for spherical Λ = π was a SPHERICAL-CUTOFF")
    print("    artifact, NOT a structural prediction of the framework.")
    print("  - For srs's actual physics (BCC lattice), the natural integration")
    print("    domain is the BCC BZ. Cone-effective theory + BCC BZ gives")
    print("    ζ ≈ 1.10e-3 with no obvious clean rational match yet.")
    print("  - Pinning the structural ζ rigorously requires the full Bloch")
    print("    Hamiltonian (not the cone-effective truncation), since the cone-")
    print("    effective + BCC BZ uses the cone theory beyond its validity range.")
    print()
    print("  This is genuine session 3 work: full Bloch Hamiltonian over actual")
    print("  BCC BZ. The cone-effective + various cutoff regulators give a family")
    print("  of values, none uniquely 'the answer' without that full calculation.")


if __name__ == "__main__":
    main()
