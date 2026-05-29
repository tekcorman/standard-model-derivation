#!/usr/bin/env python3
"""
G_sub closure Phase 1: Γ-cone full-vertex matter loop.

Per an internal working note. Phase 1 of the
3-session K-meta-theorem-constrained closure path. Builds the analogue of
path-#6's P-cone full-vertex code (`lorentz_sig_g_sub_p_cone_full_vertex.py`)
for the 3-band Γ cone, using FULL Bloch projection (not the cone-effective
approximation that v2 used).

Method
------
1. At each δk inside a sphere of radius Λ_cone around Γ:
   - Diagonalize H(δk) numerically.
   - Identify the 3-band cone subspace (3 lowest of 4 bands; T-irrep).
   - Identify the 3-band cone subspace at k+p similarly.
2. Compute strain perturbation matrix A^{ab}(k_mid) (full 4×4) at the symmetric
   midpoint k_mid = (k + k+p)/2.
3. Compute matter polarization Π^{ab,cd}(p) by Lindhard-style sum over the
   3×3 = 9 band-pair channels (cone-cone) at each grid point.
4. Extract a_2 = leading p² coefficient via polynomial fit.
5. Compute ζ_Γ_full = a_2 × v_F^Γ / Λ_cone² for several Λ_cone ∈ {0.2, 0.3, 0.5, 0.7, 1.0}.

Validation target
-----------------
At small Λ_cone (cone-effective regime), ζ_Γ_full should match v2's
ζ_universal = 27/(512π³) ≈ 1.701e-3 (since at Γ, P_cart = 0 so V_0 = 0
and the strain vertex is purely linear-in-δk like cone-effective Iorio).

If the validation succeeds at small Λ_cone but ζ_Γ_full(Λ_cone) deviates
appreciably at larger Λ_cone, that quantifies the band-curvature corrections
to the universal-ζ value at sphere Λ=π in v2.

Conventions (per K-meta-theorem § 3 + path-#6)
- Sphere of radius Λ_cone around Γ in Cartesian k-space.
- μ_cone = -1 (T-irrep band center; h=0 flat band sits at -1, half-filled).
- Spherical sampling.

Conditional on Phase 2 result for the P-cone, the multi-valley sum will be
assembled in Phase 3.
"""
from __future__ import annotations

import numpy as np


# Atom positions and bond list (matching path-#6 + path-#5 + lichnerowicz_closure)
ATOMS = np.array([
    [1/8, 1/8, 1/8],
    [3/8, 7/8, 5/8],
    [7/8, 5/8, 3/8],
    [5/8, 3/8, 7/8],
])

A_PRIM = np.array([
    [-1/2,  1/2,  1/2],
    [ 1/2, -1/2,  1/2],
    [ 1/2,  1/2, -1/2],
])

CELL_EDGES = [
    (0, 1, (1, 1, 1)),
    (0, 2, (1, 1, 1)),
    (0, 3, (1, 1, 1)),
    (1, 2, (-1, 0, 0)),
    (1, 3, (0, 1, 0)),
    (2, 3, (0, 0, -1)),
]

DIRECTED_BONDS = []
for s, t, c in CELL_EDGES:
    DIRECTED_BONDS.append((s, t, np.array(c)))
    DIRECTED_BONDS.append((t, s, -np.array(c)))


def bond_displacement(src, tgt, cell):
    return ATOMS[tgt] - ATOMS[src] + cell @ A_PRIM


BOND_DISPLACEMENTS = [
    (s, t, bond_displacement(s, t, c)) for s, t, c in DIRECTED_BONDS
]


def H_bloch(k_cart):
    """4×4 Bloch Hamiltonian at Cartesian k."""
    H = np.zeros((4, 4), dtype=complex)
    for s, t, rb in BOND_DISPLACEMENTS:
        phase = np.exp(1j * np.dot(k_cart, rb))
        H[t, s] += phase
    return (H + H.conj().T) / 2


def A_strain_full(k_cart, a, c):
    """Strain perturbation A^{ac}(k) = i Σ_bonds e^{i k·r} k_a r^c (4×4, Hermitized)."""
    A_mat = np.zeros((4, 4), dtype=complex)
    for s, t, rb in BOND_DISPLACEMENTS:
        phase = np.exp(1j * np.dot(k_cart, rb))
        A_mat[t, s] += 1j * phase * k_cart[a] * rb[c]
    return (A_mat + A_mat.conj().T) / 2


def Pi_gamma_cone_at_external_p(
    p_cart,
    n_radial=15,
    n_theta=10,
    n_phi=10,
    Lambda_cone=0.5,
    target_ev=-1.0,
):
    """Compute Π^{ab,cd}(p) for the Γ-cone (3-band) with the full strain vertex.

    Spherical sampling of momentum δk in sphere of radius Lambda_cone around Γ.
    """
    q_radial = np.linspace(
        Lambda_cone / n_radial / 2,
        Lambda_cone - Lambda_cone / n_radial / 2,
        n_radial,
    )
    dq = Lambda_cone / n_radial
    theta_grid = np.linspace(np.pi / n_theta / 2, np.pi - np.pi / n_theta / 2, n_theta)
    dtheta = np.pi / n_theta
    phi_grid = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi

    Pi = np.zeros((3, 3, 3, 3), dtype=complex)
    n_pts = 0
    for q_mag in q_radial:
        for theta in theta_grid:
            for phi in phi_grid:
                dk = q_mag * np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ])
                k_cart = dk        # Γ at origin
                kp_cart = k_cart + p_cart

                H_k = H_bloch(k_cart)
                H_kp = H_bloch(kp_cart)
                eigs_k, U_k = np.linalg.eigh(H_k)
                eigs_kp, U_kp = np.linalg.eigh(H_kp)

                # 3-band T-irrep: 3 bands closest to -1, then sort by energy
                # (lowest = filled, middle = half-filled flat, upper = empty).
                d_k = np.abs(eigs_k - target_ev)
                d_kp = np.abs(eigs_kp - target_ev)
                idx_k_3 = np.argsort(d_k)[:3]
                idx_kp_3 = np.argsort(d_kp)[:3]
                # Re-sort by energy ascending so band-position fixes f
                idx_k = idx_k_3[np.argsort(eigs_k[idx_k_3])]
                idx_kp = idx_kp_3[np.argsort(eigs_kp[idx_kp_3])]

                # Band-position-based Fermi factors (matches v2 cone-effective
                # convention: lowest cone band filled, middle (flat) half-filled,
                # upper empty — so the cone manifold is half-filled overall).
                f_table = [1.0, 0.5, 0.0]

                # Vertex convention follows v2 (cone-effective spin-1) which uses
                # the non-symmetric V^{ab}(q) = q^a × S^b form. This convention
                # is the one normalized by 27/(512π³) ≈ 1.701e-3 at Γ.
                k_mid = (k_cart + kp_cart) / 2
                A_kmid = np.zeros((3, 3, 4, 4), dtype=complex)
                for a in range(3):
                    for b in range(3):
                        A_kmid[a, b] = A_strain_full(k_mid, a, b)

                vol = q_mag ** 2 * np.sin(theta) * dq * dtheta * dphi / (2 * np.pi) ** 3

                for n_pos, n in enumerate(idx_k):
                    f_n = f_table[n_pos]
                    psi_n = U_k[:, n]
                    E_n = eigs_k[n]
                    for m_pos, m in enumerate(idx_kp):
                        f_m = f_table[m_pos]
                        E_m = eigs_kp[m]
                        diff = f_n - f_m
                        denom = E_n - E_m
                        if abs(diff) < 1e-12 or abs(denom) < 1e-12:
                            continue
                        psi_m = U_kp[:, m]
                        for a in range(3):
                            for b in range(3):
                                ME_nm = psi_n.conj() @ A_kmid[a, b] @ psi_m
                                for c in range(3):
                                    for d in range(3):
                                        ME_mn = psi_m.conj() @ A_kmid[c, d] @ psi_n
                                        Pi[a, b, c, d] += diff * (
                                            ME_nm * ME_mn / denom
                                        ) * vol
                n_pts += 1
    return Pi


def TT_project_zhat(Pi):
    """Π_TT for p along ẑ (matches path-#6 + v2 conventions)."""
    Pi_xxxx = Pi[0, 0, 0, 0]
    Pi_xxyy = Pi[0, 0, 1, 1]
    Pi_yyyy = Pi[1, 1, 1, 1]
    Pi_xyxy = Pi[0, 1, 0, 1]
    return ((Pi_xxxx - 2 * Pi_xxyy + Pi_yyyy) / 4 + Pi_xyxy).real


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def extract_a2_at_lambda(Lambda_cone, n_radial=15, n_theta=10, n_phi=10,
                          p_z_values=None):
    """Compute Π_TT(p²) at given Lambda_cone, polynomial fit, return a_2.

    p_z values are scaled to Lambda_cone so the cone-effective expansion stays
    valid (p ≪ Λ_cone). Default scaling: p_z ∈ {0, 0.1, 0.2, 0.3, 0.4} × Λ_cone.
    """
    if p_z_values is None:
        p_z_values = tuple(c * Lambda_cone for c in (0.0, 0.1, 0.2, 0.3, 0.4))
    Pi_TT_list = []
    for p_z in p_z_values:
        p_cart = np.array([0.0, 0.0, p_z])
        Pi = Pi_gamma_cone_at_external_p(
            p_cart, n_radial=n_radial, n_theta=n_theta, n_phi=n_phi,
            Lambda_cone=Lambda_cone,
        )
        Pi_TT = TT_project_zhat(Pi)
        Pi_TT_list.append(Pi_TT)
    p_arr = np.array(p_z_values)
    Pi_arr = np.array(Pi_TT_list)
    coeffs = np.polyfit(p_arr ** 2, Pi_arr, 2)
    a_4, a_2, a_0 = coeffs
    return a_2, a_0, a_4, Pi_TT_list


def main():
    header("G_sub closure Phase 1: Γ-cone full-vertex matter loop")
    print()
    print("  Method: full 4×4 Bloch projection onto 3-band cone subspace at Γ.")
    print("  Sphere of radius Λ_cone around Γ. μ_cone = -1.")
    print(f"  v_F^Γ = 1/2 (cone-effective dispersion E_h = -1 + (1/2) h |q|).")
    print()

    v_F_Gamma = 0.5
    zeta_universal = 27 / (512 * np.pi ** 3)

    Lambda_values = [0.2, 0.3, 0.5, 0.7, 1.0]
    print(f"  {'Λ_cone':>8s}  {'a_2':>15s}  {'a_0 (static)':>16s}  "
          f"{'ζ_Γ_full':>15s}  {'ratio to ζ_univ':>18s}")
    print(f"  {'-' * 8}  {'-' * 15}  {'-' * 16}  {'-' * 15}  {'-' * 18}")

    results = []
    for Lambda in Lambda_values:
        a_2, a_0, a_4, Pi_TT_list = extract_a2_at_lambda(
            Lambda, n_radial=12, n_theta=10, n_phi=10
        )
        zeta = a_2 * v_F_Gamma / Lambda ** 2
        ratio = zeta / zeta_universal
        results.append((Lambda, a_2, a_0, zeta, ratio))
        print(f"  {Lambda:>8.3f}  {a_2:>+.6e}  {a_0:>+.6e}  "
              f"{zeta:>+.6e}  {ratio:>+12.4f}")

    print()
    print(f"  Reference: ζ_universal = 27/(512π³) = {zeta_universal:.6e}")
    print()

    # Trend analysis
    if len(results) >= 3:
        zetas = [r[3] for r in results]
        ratios = [r[4] for r in results]
        delta_ratios = np.diff(ratios)
        print(f"  Δ(ζ/ζ_univ) between successive Λ: {[f'{d:+.3f}' for d in delta_ratios]}")
        if abs(ratios[0] - 1.0) < 0.10:
            print(f"  ✓ At smallest Λ_cone={Lambda_values[0]}, ratio = {ratios[0]:.3f} → "
                  f"ζ_Γ_full ≈ ζ_universal within 10%.")
            print(f"    → Methodology validated. Universal-ζ holds at the Γ cone.")
        else:
            print(f"  ✗ At smallest Λ_cone={Lambda_values[0]}, ratio = {ratios[0]:.3f} ≠ 1.")
            print(f"    → Methodology issue OR Γ-cone full vertex differs from cone-effective.")
            print(f"    → Investigate: sign convention, normalization, or band selection.")

    print()
    print("  Status: PHASE 1 OUTPUT (run with finer grid for production-quality result).")
    print("  Next: Phase 2 (P-cone full vertex at matched Λ_cone) — see entry-point doc.")


if __name__ == "__main__":
    main()
