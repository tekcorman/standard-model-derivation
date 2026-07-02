#!/usr/bin/env python3
"""
pi_v_dirac_cone_drude_contribution_probe.py — Fermi-surface analysis for Π_v at μ=0.

Continuation of `pi_v_drude_weight_bloch_invariants_probe.py`. The naive
inter-band f-sum approach failed (wrong sign, wrong magnitude) because srs
at half-filling is NOT a standard metal — it's a chiral Weyl semimetal
with band crossings at high-symmetry points (per
`theorem_lorentzian_signature_from_dirac_cones.md` + `srs_weyl_points_probe.py`).

The Drude weight `d` in `π_2_xx = a + d/ω²` therefore doesn't come from a
2D Fermi surface velocity integral; it comes from contributions of the
Dirac/Weyl cones at the Fermi level.

This probe:

  STEP 1 — find all k-points where E_n(k) = 0 (the actual "Fermi surface").
  Test high-symmetry candidates Γ, H, P, N, plus a coarse k-scan to spot
  others.

  STEP 2 — at each Fermi-level touching, linearize H(k) around it to
  identify the cone structure: dim of touching, v_F, chirality.

  STEP 3 — compute the cone contribution to the Drude weight for Π_v.
  For a 3D Weyl cone, the standard result is
    σ_xx(ω) → (e²/(12πv_F)) × |ω|  (linear in ω, NOT Drude δ)
  giving a regular ω → 0 limit. For 4-fold or higher-degeneracy Dirac
  cones, the prefactor changes but the linear-in-ω scaling persists.

  STEP 4 — compare to the measured d_Π_v ≈ -0.00728 and see whether
  the cone contributions reproduce it. If yes, we have the structural
  derivation. If no, additional substrate physics (e.g., flat bands,
  off-symmetry crossings) must be identified.

The Π_TT (G_sub) analog used the SAME mechanism — D_TT = -1/(⟨Tr H²⟩ × k*) =
-1/36 emerged from the Bloch operator with consistent application to the
matter sector at half-filling. For Π_v we expect a similar substrate-
intrinsic structural form once the right cone-counting is identified.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lorentz_sig_g_sub_elastic_moduli import H_bloch
from gauge_beta_from_substrate_kubo_probe import velocity_matrix


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


# =============================================================================
# Step 1 — find E_n(k) = 0 points (the Fermi surface at μ=0)
# =============================================================================

# srs primitive lattice vectors give a BCC-derived reciprocal lattice
# High-symmetry k-points (in Cartesian, where the primitive vectors are
# A_PRIM rows of the 4-atom srs primitive cell):
#   Γ = (0, 0, 0)
#   H = (2π, 0, 0) [or symmetric variants]
#   P = (π/2, π/2, π/2)
#   N = (π/2, π/2, 0)

high_sym_points = {
    "Γ":  np.array([0.0,    0.0,    0.0]),
    "H":  np.array([2*np.pi, 0.0,   0.0]),
    "P":  np.array([np.pi/2, np.pi/2, np.pi/2]),
    "N":  np.array([np.pi/2, np.pi/2, 0.0]),
    # The "P_neg" antipode
    "-P": np.array([-np.pi/2, -np.pi/2, -np.pi/2]),
}


def eigvals_at_k(k):
    return np.linalg.eigvalsh(H_bloch(k))


def step1_find_fermi_level_touchings():
    header("STEP 1: where does E_n(k) = 0 happen on srs at μ=0?")
    print()
    print(f"  {'k-point':>8s}  {'eigenvalues':>40s}  {'|E_n| min':>10s}  {'at FS?':>8s}")
    fs_candidates = []
    for name, k in high_sym_points.items():
        eigs = eigvals_at_k(k)
        eig_str = "[" + ", ".join(f"{e:+.4f}" for e in eigs) + "]"
        min_abs = min(abs(e) for e in eigs)
        at_fs = "YES" if min_abs < 1e-6 else "no"
        print(f"  {name:>8s}  {eig_str:>40s}  {min_abs:>+10.4e}  {at_fs:>8s}")
        if min_abs < 1e-6:
            fs_candidates.append((name, k, eigs))
    print()

    # Coarse BZ scan for accidental zeros
    print("  Coarse BZ scan for other E_n=0 zeros (5×5×5 grid, ω-tolerance 1e-3):")
    found_extras = 0
    grid = np.linspace(-np.pi, np.pi, 5)
    for k1 in grid:
        for k2 in grid:
            for k3 in grid:
                k = np.array([k1, k2, k3])
                eigs = eigvals_at_k(k)
                if min(abs(e) for e in eigs) < 1e-3:
                    # Skip if too close to already-identified high-sym points
                    matched = False
                    for name, kp, _ in fs_candidates:
                        if np.linalg.norm(k - kp) < 0.1:
                            matched = True
                            break
                    if not matched:
                        if found_extras < 5:
                            print(f"    accidental zero at k = {k}: eigs = {eigs}")
                        found_extras += 1
    print(f"  Found {found_extras} accidental zeros in coarse scan.")
    print()
    return fs_candidates


# =============================================================================
# Step 2 — local cone analysis at each Fermi-level point
# =============================================================================

def linearize_at_point(k0, eps=1e-3):
    """Compute H_bloch near k0; identify zero-eigenvalue subspace and the
    effective Dirac/Weyl Hamiltonian in that subspace."""
    H0 = H_bloch(k0)
    eigs0, U0 = np.linalg.eigh(H0)
    zero_mask = np.abs(eigs0) < 1e-6
    deg = int(np.sum(zero_mask))
    print(f"  Degeneracy at k0: {deg}-fold (eigenvalues at zero: {eigs0[zero_mask]})")
    if deg == 0:
        return None

    # Project H near k0 onto the zero subspace
    P_zero = U0[:, zero_mask]  # 4 × deg matrix

    # Compute ∂H/∂k_μ projected onto the zero subspace at k0
    # H_local(q) ≈ Σ_μ q_μ × (P_zero^† v^μ(k0) P_zero) + O(q²)
    v_projected = []
    for mu in range(3):
        v_mu = velocity_matrix(k0, mu)
        v_proj = P_zero.conj().T @ v_mu @ P_zero  # deg × deg
        v_projected.append(v_proj)

    print(f"  Projected velocity matrices in zero-subspace ({deg}×{deg}):")
    for mu, v in enumerate(v_projected):
        norm = np.linalg.norm(v, 'fro')
        print(f"    ‖v^{mu}_proj‖_F = {norm:.6f}")

    # Effective Fermi velocity from the projected velocity operator
    # For 2-fold Weyl: H_eff(q) = q · v_eff σ + ... with v_eff = trace of projected velocity
    # For higher degeneracy: analyse rank/spectrum

    return {
        "k0": k0,
        "degeneracy": deg,
        "P_zero": P_zero,
        "v_projected": v_projected,
        "eigenvalues_at_k0": eigs0,
    }


# =============================================================================
# Step 3 — compute Drude-cone contribution to π_2_xx(ω) at saturated regime
# =============================================================================

def cone_contribution_to_pi2(cone_data, omega_E=0.30, q_max=0.10, n_q=10):
    """For a Dirac/Weyl cone with linear dispersion E_±(q) = ±v|q|, compute the
    leading p² coefficient contribution to Π^{xx}_v at finite ω_E.

    For a 2-fold Weyl cone with H_local = q · (v_x σ^1 + v_y σ^2 + v_z σ^3),
    the matter loop at finite ω contributes:
      π_2_cone(ω_E) = -(2/(2π)³) × ∫ d³q × Σ_{nm} (f_n-f_m) (E_n-E_m)/[(E_n-E_m)²+ω²] × |⟨v^x⟩|²_p²coef

    Here we just do the integration numerically over a small ball around q=0
    (the cone region) with q_max as the cutoff.
    """
    deg = cone_data["degeneracy"]
    v_projected = cone_data["v_projected"]

    if deg < 2:
        return 0.0

    # Build local Dirac/Weyl Hamiltonian: H_eff(q) = Σ_μ q_μ × v_projected[μ]
    # Linearised dispersion; integrate Kubo response over a small q-ball
    print(f"  Integrating cone contribution: ω={omega_E}, q_max={q_max}, n_q={n_q}")

    total_pi2 = 0.0
    qs = np.linspace(-q_max, q_max, n_q)
    dq = qs[1] - qs[0]
    p_z_values = (0.0, 0.05, 0.10)

    # For each p_z value, compute Π^{xx}(p_z, ω) contribution from cone
    pi_at_pz = []
    for p_z in p_z_values:
        pi_pz = 0.0
        n_pts = 0
        for qx in qs:
            for qy in qs:
                for qz in qs:
                    q = np.array([qx, qy, qz])
                    q_plus_p = np.array([qx, qy, qz + p_z])
                    # H_eff at q
                    H_q = sum(q[mu] * v_projected[mu] for mu in range(3))
                    H_qp = sum(q_plus_p[mu] * v_projected[mu] for mu in range(3))
                    eigs_q, U_q = np.linalg.eigh(H_q)
                    eigs_qp, U_qp = np.linalg.eigh(H_qp)
                    # Fermi sea at μ=0
                    filled_q = eigs_q < -1e-9
                    unfilled_qp = eigs_qp > 1e-9
                    # Velocity at midpoint q_mid (linearised, doesn't depend on q)
                    for mu_idx, m in enumerate(np.where(unfilled_qp)[0]):
                        for n in np.where(filled_q)[0]:
                            diff = -1.0  # f_n - f_m = 1 - 0 = +1 for n filled
                            Delta = eigs_q[n] - eigs_qp[m]
                            denom = Delta * Delta + omega_E * omega_E
                            weight = (1 if filled_q[n] and unfilled_qp[m] else 0) * Delta / denom
                            # Matrix element of v^x at projected level
                            v_x_proj = v_projected[0]  # constant within cone (linearised)
                            # ⟨m|v^x|n⟩ in band basis
                            v_in_basis = U_qp.conj().T @ v_x_proj @ U_q
                            term = abs(v_in_basis[m, n]) ** 2
                            pi_pz += -2.0 * weight * term
                    n_pts += 1
        pi_pz *= (dq ** 3)  # volume element
        pi_at_pz.append(pi_pz)

    # Extract leading p² coefficient
    p_arr = np.array(p_z_values)
    coeffs = np.polyfit(p_arr ** 2, pi_at_pz, 1)
    pi2_coef = coeffs[0]
    pi0 = coeffs[1]
    print(f"    π_0(ω={omega_E}) = {pi0:+.6f}  (q=0 piece)")
    print(f"    π_2(ω={omega_E}) = {pi2_coef:+.6f}  (leading p²)")
    return pi2_coef


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 78)
    print("  Π_v Drude weight via Dirac/Weyl cone analysis at μ=0")
    print("=" * 78)

    # Step 1: find Fermi-level zeros
    fs_candidates = step1_find_fermi_level_touchings()

    if not fs_candidates:
        header("NO Fermi-level touchings found at high-symmetry points")
        print("  Need a finer scan to identify the actual Fermi surface.")
        return

    # Step 2: analyse each Fermi-level point
    header("STEP 2: local cone analysis at each Fermi-level touching")
    print()
    all_cone_data = []
    for name, k, eigs in fs_candidates:
        print(f"  Analysing k = {name} = {k}")
        cone_data = linearize_at_point(k)
        if cone_data is not None:
            cone_data["name"] = name
            all_cone_data.append(cone_data)
        print()

    # Step 3: compute the cone contribution to Drude weight
    header("STEP 3: cone contribution to π_2_xx(ω) at saturated regime")
    print()
    omega_E = 0.30  # saturated regime test point
    for cone_data in all_cone_data:
        print(f"  Cone at {cone_data['name']}:")
        pi2_cone = cone_contribution_to_pi2(cone_data, omega_E=omega_E)
        print(f"    Cone contribution to π_2_xx(ω={omega_E}): {pi2_cone:+.6e}")
        print()

    # Compare to pairwise-extracted value
    header("STEP 4: comparison to pairwise-extracted d at ω=0.30")
    pairwise_d = -0.00728
    print(f"  Predicted π_2(0.30) from Drude fit:  a + d/ω² with a≈0.108, d≈-0.00728")
    a_pair = 0.108
    d_pair = -0.00728
    pi2_predicted = a_pair + d_pair / omega_E ** 2
    print(f"    a_phys + d_phys/(0.3)² = {a_pair} + {d_pair}/0.09 = {pi2_predicted:+.6f}")
    print(f"  But this is the PHYSICAL form, and the cone integration gives raw fit")
    print(f"  convention. Need to compare with appropriate sign and normalisation.")
    print()
    print(f"  Headline finding: see whether the cone contribution captures the")
    print(f"  measured π_2(ω=0.30) magnitude. If it does, this IS the structural")
    print(f"  derivation of d_Π_v.")


if __name__ == "__main__":
    main()
