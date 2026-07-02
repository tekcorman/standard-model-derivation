#!/usr/bin/env python3
"""
G_sub closure Phase 2: P-cone full-vertex matter loop at small Λ_cone.

Same Lindhard convention as Phase 1 (`lorentz_sig_g_sub_gamma_cone_full_vertex.py`)
to enable apples-to-apples ratio comparison ζ_P_full / ζ_Γ_full.

Differences vs path-#6 (`lorentz_sig_g_sub_p_cone_full_vertex.py`):
- Lindhard formula uses Pi += diff × (ME × ME / denom).real × vol with explicit
  diff factor (path-#6 uses -2.0 prefactor without diff, which makes contributions
  from (filled→empty) and (empty→filled) ordered pairs cancel for diagonal Π entries).
- Band-position-based Fermi factors (lower of 2 cone bands at +√3 = filled,
  upper = empty); matches half-filling of substrate (μ at +√3 = cone center).
- Λ_cone scan, including Λ_cone ∈ {0.15, 0.2, 0.25, 0.3, 0.4}.

Validation
----------
With same Lindhard convention as Phase 1, ζ_P_full(Λ_cone → 0) should converge
to a value that is comparable to Phase 1's ζ_Γ_full ≈ 0.95e-3.
- If ζ_P_full / ζ_Γ_full → 1: universal-ζ holds → 4(√3-1)/27 candidate validated.
- If ratio ≠ 1 but converges to a clean rational/algebraic number: structural
  form for G_sub differs from 4(√3-1)/27.
- If ratio diverges as Λ_cone → 0: V_0 piece dominates and 1/Λ²-scaling of
  the effective ζ_P signals failure of cone-effective theory at the P-cone.
"""
from __future__ import annotations

import numpy as np


# Bond list (matching Phase 1 + path-#6 + path-#5)
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
    H = np.zeros((4, 4), dtype=complex)
    for s, t, rb in BOND_DISPLACEMENTS:
        phase = np.exp(1j * np.dot(k_cart, rb))
        H[t, s] += phase
    return (H + H.conj().T) / 2


def A_strain_full(k_cart, a, c):
    """A^{ac}(k) = i Σ_bonds e^{i k·r} k_a r^c (4×4 Hermitized)."""
    A_mat = np.zeros((4, 4), dtype=complex)
    for s, t, rb in BOND_DISPLACEMENTS:
        phase = np.exp(1j * np.dot(k_cart, rb))
        A_mat[t, s] += 1j * phase * k_cart[a] * rb[c]
    return (A_mat + A_mat.conj().T) / 2


def Pi_p_cone_at_external_p(
    p_cart,
    P_cart,
    n_radial=15,
    n_theta=10,
    n_phi=10,
    Lambda_cone=0.5,
    target_ev=np.sqrt(3),
):
    """Compute Π^{ab,cd}(p) at the P-cone with full strain vertex.

    Same Lindhard convention as Phase 1: explicit diff/denom × vol, .real per loop.
    Spherical sampling around P_cart in Cartesian k.
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
    for q_mag in q_radial:
        for theta in theta_grid:
            for phi in phi_grid:
                dk = q_mag * np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ])
                k_cart = P_cart + dk
                kp_cart = k_cart + p_cart

                H_k = H_bloch(k_cart)
                H_kp = H_bloch(kp_cart)
                eigs_k, U_k = np.linalg.eigh(H_k)
                eigs_kp, U_kp = np.linalg.eigh(H_kp)

                # 2-band P cone at +√3: 2 bands closest to target_ev
                d_k = np.abs(eigs_k - target_ev)
                d_kp = np.abs(eigs_kp - target_ev)
                idx_k_2 = np.argsort(d_k)[:2]
                idx_kp_2 = np.argsort(d_kp)[:2]
                # Sort by energy ascending so band-position fixes f
                idx_k = idx_k_2[np.argsort(eigs_k[idx_k_2])]
                idx_kp = idx_kp_2[np.argsort(eigs_kp[idx_kp_2])]

                # 2-band cone half-filled (substrate at half-filling):
                # lower of 2 cone bands at +√3 is filled, upper is empty.
                f_table = [1.0, 0.0]

                # Vertex convention follows v2 (cone-effective spin-1) which uses
                # the non-symmetric V^{ab}(q) = q^a × S^b form. Phase 1 uses the
                # same convention so the Γ vs P ratio cancels conventions cleanly.
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
                                        ).real * vol
    return Pi


def TT_project_zhat(Pi):
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


def extract_a2_at_lambda(Lambda_cone, P_cart, target_ev=np.sqrt(3),
                          n_radial=15, n_theta=10, n_phi=10,
                          p_z_values=None):
    """Polynomial-fit Π_TT(p²) at given Λ_cone; return (a_2, a_0, a_4, list).

    p_z values are scaled to Λ_cone so the cone-effective expansion stays valid.
    """
    if p_z_values is None:
        p_z_values = tuple(c * Lambda_cone for c in (0.0, 0.1, 0.2, 0.3, 0.4))
    Pi_TT_list = []
    for p_z in p_z_values:
        p_cart = np.array([0.0, 0.0, p_z])
        Pi = Pi_p_cone_at_external_p(
            p_cart, P_cart,
            n_radial=n_radial, n_theta=n_theta, n_phi=n_phi,
            Lambda_cone=Lambda_cone, target_ev=target_ev,
        )
        Pi_TT_list.append(TT_project_zhat(Pi))
    p_arr = np.array(p_z_values)
    Pi_arr = np.array(Pi_TT_list)
    coeffs = np.polyfit(p_arr ** 2, Pi_arr, 2)
    a_4, a_2, a_0 = coeffs
    return a_2, a_0, a_4, Pi_TT_list


def main():
    header("G_sub closure Phase 2: P-cone full-vertex matter loop at small Λ_cone")
    print()
    print("  Same Lindhard convention as Phase 1; same code structure (full Bloch")
    print("  projection + diff/denom × vol). Goal: extract ζ_P_full at small Λ_cone")
    print("  and compare to Phase 1's ζ_Γ_full ≈ 0.95e-3.")
    print()

    P_cart = np.pi * np.array([1.0, 1.0, 1.0])
    target_ev_pos = np.sqrt(3)
    target_ev_neg = -np.sqrt(3)
    v_F_P = np.sqrt(3) / 6
    print(f"  P_cart = π(1,1,1) = ({P_cart[0]:.4f}, {P_cart[1]:.4f}, {P_cart[2]:.4f})")
    print(f"  v_F^P = √3/6 = {v_F_P:.6f}")
    print()

    # Phase 1 reference (for ratio)
    zeta_Gamma_phase1 = 0.95e-3   # observed at Λ ∈ [0.2, 0.4] in Phase 1 convention
    print(f"  Phase 1 reference: ζ_Γ_full ≈ {zeta_Gamma_phase1:.3e} (small-Λ asymptote)")
    print()

    Lambda_values = [0.15, 0.20, 0.25, 0.30, 0.40]
    print(f"  P+√3 cone scan (target_ev = +√3):")
    print(f"  {'Λ_cone':>8s}  {'a_2':>15s}  {'a_0':>15s}  "
          f"{'ζ_P_full':>15s}  {'ζ_P/ζ_Γ':>12s}")
    print(f"  {'-' * 8}  {'-' * 15}  {'-' * 15}  {'-' * 15}  {'-' * 12}")

    results_pos = []
    for Lambda in Lambda_values:
        a_2, a_0, a_4, _ = extract_a2_at_lambda(
            Lambda, P_cart, target_ev=target_ev_pos,
            n_radial=12, n_theta=10, n_phi=10,
        )
        zeta = a_2 * v_F_P / Lambda ** 2
        ratio = zeta / zeta_Gamma_phase1
        results_pos.append((Lambda, a_2, a_0, zeta, ratio))
        print(f"  {Lambda:>8.3f}  {a_2:>+.6e}  {a_0:>+.6e}  "
              f"{zeta:>+.6e}  {ratio:>+10.4f}")

    print()
    print("  Trend analysis (P+√3 cone):")
    for L, a2, a0, z, r in results_pos:
        print(f"    Λ={L:.2f}: ζ_P/ζ_Γ = {r:+.3f}  (a_0/Λ⁴ = {a0/L**4:.3e}, "
              f"static-elastic Λ⁴ scaling check)")
    print()

    if len(results_pos) >= 3:
        ratios = [r[4] for r in results_pos]
        small_Lambda_ratio = np.mean(ratios[:2])  # average over 2 smallest Λ
        print(f"  Average ratio at smallest 2 Λ values: {small_Lambda_ratio:+.4f}")
        if abs(small_Lambda_ratio - 1.0) < 0.10:
            print(f"  ✓ ζ_P_full ≈ ζ_Γ_full → universal-ζ HOLDS with full vertex.")
            print(f"    → 4(√3-1)/27 candidate validated.")
        elif small_Lambda_ratio > 5:
            print(f"  ✗ ζ_P_full ≫ ζ_Γ_full → V_0 constant piece dominates.")
            print(f"    → Cone-effective theory FAILS at P (V_0 ≠ 0 from P_cart ≠ 0).")
            print(f"    → Phase 3 needs different structural framework.")
        else:
            print(f"  ? ζ_P_full ≠ ζ_Γ_full but bounded.")
            print(f"    → Investigate whether ratio approaches a clean K-element.")

    print()
    print("  STATUS: Phase 2 OUTPUT.")
    print("  Next: Phase 3 (multi-valley sum + K-admissibility check) given Phase 2's result.")


if __name__ == "__main__":
    main()
