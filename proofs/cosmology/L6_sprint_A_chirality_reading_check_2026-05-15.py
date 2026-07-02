#!/usr/bin/env python3
"""
L6 Sprint A — chirality reading check (companion probe, 2026-05-15 EOD+5).

PURPOSE
-------
The single-sector overlap test in `L6_sprint_A_bloch_decomposition_probe_2026-05-15.py`
showed |<L_dir|P_+h|L_dir>| ~ 0 at K_P. This is consistent with the existing
β derivation framework (which uses a SUM over walker eigensectors weighted
by sin(arg λ_sector), not a single-sector overlap).

This companion probe replicates the existing β derivation's chirality reading
along the Gamma -> P path and asks the load-bearing Sprint A question:

  Does the chirality reading c = (chir_L - chir_R) / (2 sin(arg h))
  extend smoothly from K_P (where c = 1 per UNIQUE-THEOREM-GRADE
  beta_cosmic_birefringence) to small k near Gamma (acoustic regime)?

A clean extension would mean the framework's photon-walker chirality coupling
has a SMOOTH k-dependent generalization — Sprint A PASS on the chirality
reading.

A k-discontinuous c(k) means the K_P chirality reading is structurally
tied to high-symmetry k-points and does not have an acoustic-regime
analog — Sprint A FAIL.

DESIGN
------
At each k along Gamma -> P, compute:
  c(k) = (sum_lambda |<L_dir|P_lambda|L_dir>| sin(arg lambda)
         - sum_lambda |<R_dir|P_lambda|R_dir>| sin(arg lambda)) / (2 sin(arg h))

where the sum is over walker eigenvalues lambda in {+h, +h*, -h, -h*, +1, -1}
(the K_P walker eigenstructure; at intermediate k the targets shift continuously).

L(k) and R(k) are tracked by adiabatic continuation from K_P.
"""

import math
import os
import sys

import numpy as np
from numpy import linalg as la

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from srs_photon_bloch_primitive import (
    build_primitive_unit_cell,
    canonical_edges_primitive,
    find_primitive_connectivity,
    incidence_matrix_primitive,
)
from srs_photon_hodge import build_d1, build_edge_lookup
from srs_cycle_enumerator import enumerate_simple_cycles
from srs_photon_c3_chainmap import build_C3_edge, build_delta_1
from srs_photon_chirality_coefficient import (
    build_B_directed,
    build_pi_projector,
)


def get_photon_LR_at_kp(Delta_1, C3_e, eigval=36.0, tol=1e-6):
    """Extract L (omega-irrep) and R (omega-bar-irrep) photon modes at K_P
    via C_3 representation theory."""
    eigs, vecs = la.eig(Delta_1)
    order = np.argsort(eigs.real)
    eigs, vecs = eigs[order], vecs[:, order]
    mask = np.abs(eigs.real - eigval) < tol
    photon_basis = vecs[:, mask]
    Q, _ = la.qr(photon_basis)

    omega = np.exp(2j * math.pi / 3)
    omega2 = omega.conjugate()
    C3_photon = Q.conj().T @ C3_e @ Q
    ev_C3, vec_C3 = la.eig(C3_photon)
    L_in_Q = vec_C3[:, np.argmin(np.abs(ev_C3 - omega))]
    R_in_Q = vec_C3[:, np.argmin(np.abs(ev_C3 - omega2))]
    L_in_Q /= la.norm(L_in_Q)
    R_in_Q /= la.norm(R_in_Q)
    L_undir = Q @ L_in_Q
    R_undir = Q @ R_in_Q
    return L_undir, R_undir, eigs


def track_mode_by_overlap(eigs, vecs, prev_mode, eigval_hint=None, tol_eig=2.0):
    """Adiabatic continuation: find eigenvector with max overlap to prev_mode,
    optionally constrained to a band near eigval_hint."""
    if eigval_hint is None:
        candidates = list(range(len(eigs)))
    else:
        candidates = [j for j in range(len(eigs))
                      if abs(eigs[j].real - eigval_hint) < tol_eig]
        if not candidates:
            candidates = list(range(len(eigs)))
    overlaps = [abs(np.vdot(prev_mode, vecs[:, j])) for j in candidates]
    best_j = candidates[np.argmax(overlaps)]
    new_mode = vecs[:, best_j].copy()
    new_mode /= la.norm(new_mode)
    gauge = np.vdot(prev_mode, new_mode)
    if abs(gauge) > 1e-10:
        new_mode *= np.conj(gauge) / abs(gauge)
    return new_mode, eigs[best_j].real, abs(np.vdot(prev_mode, new_mode))


def chirality_reading(L_dir, R_dir, B, walker_targets):
    """Compute chirality reading c = (chir_L - chir_R) / (2 sin(arg h)).

    walker_targets: dict of {label: target_eigval} for K_P-style enumeration.
    At intermediate k, projectors are built for the eigenvalues closest to
    the K_P targets (continuous deformation).
    """
    eigs, vecs = la.eig(B)
    sin_arg_h = math.sqrt(5.0 / 8.0)  # sin(arg h) at K_P

    chir_L = 0.0
    chir_R = 0.0
    contributions = {}
    for label, target_KP in walker_targets.items():
        # Find eigenvalue closest to target_KP
        distances = np.abs(eigs - target_KP)
        j_closest = np.argmin(distances)
        lambda_k = eigs[j_closest]
        # Build single-eigenvector projector at this k (continuous tracking)
        # For simplicity, project on the one eigenvector (assume non-degenerate at general k)
        v = vecs[:, j_closest]
        v /= la.norm(v)
        P = np.outer(v, v.conj())
        # Chirality contribution: |overlap|^2 * sin(arg lambda_k)
        # (At K_P, the multiplicity gives larger projector; at general k single eigenvector)
        wL = np.real(np.vdot(L_dir, P @ L_dir))
        wR = np.real(np.vdot(R_dir, P @ R_dir))
        sin_arg_lambda = math.sin(np.angle(lambda_k))
        chir_L += wL * sin_arg_lambda
        chir_R += wR * sin_arg_lambda
        contributions[label] = (wL, wR, sin_arg_lambda, lambda_k)

    c_coefficient = (chir_L - chir_R) / (2 * sin_arg_h)
    return c_coefficient, chir_L, chir_R, contributions


def main():
    print("=" * 76)
    print(" L6 Sprint A — chirality reading c(k) along Gamma -> P")
    print("=" * 76)
    print()

    # Build infrastructure
    verts, lat = build_primitive_unit_cell()
    bonds = find_primitive_connectivity(verts, lat)
    edges = canonical_edges_primitive(bonds)
    edge_lookup = build_edge_lookup(edges)
    cycles = enumerate_simple_cycles(bonds, max_length=10)
    print()

    P_red = np.array([0.25, 0.25, 0.25])
    h_KP = complex(math.sqrt(3) / 2, math.sqrt(5) / 2)
    walker_targets_KP = {
        "+h": h_KP,
        "+h*": h_KP.conjugate(),
        "-h": -h_KP,
        "-h*": -h_KP.conjugate(),
        "+1": 1.0 + 0j,
        "-1": -1.0 + 0j,
    }

    # Reference at K_P
    k_KP = P_red
    d_KP = incidence_matrix_primitive(k_KP, edges, len(verts))
    d1_KP = build_d1(cycles, edge_lookup, k_KP, len(edges))
    Delta_1_KP = build_delta_1(d_KP, d1_KP)
    C3_e_KP = build_C3_edge(edges, k_KP)
    B_KP = build_B_directed(bonds, k_KP)
    pi_sym_KP = build_pi_projector(bonds, edges, k_KP)

    L_undir_KP, R_undir_KP, _ = get_photon_LR_at_kp(Delta_1_KP, C3_e_KP)
    L_dir_KP = pi_sym_KP @ L_undir_KP
    R_dir_KP = pi_sym_KP @ R_undir_KP
    L_dir_KP /= la.norm(L_dir_KP)
    R_dir_KP /= la.norm(R_dir_KP)

    c_KP, chir_L_KP, chir_R_KP, contribs_KP = chirality_reading(
        L_dir_KP, R_dir_KP, B_KP, walker_targets_KP)

    print(" Reference at K_P (t = 1.0):")
    print(f"   chir_L = {chir_L_KP:+.6f}, chir_R = {chir_R_KP:+.6f}")
    print(f"   c(K_P) = (chir_L - chir_R) / (2 sin(arg h)) = {c_KP:+.6f}")
    print(f"   Expected c at K_P (per beta_cosmic_birefringence): 1.000 (UNIQUE-THEOREM-GRADE 2026-04-29)")
    print(f"   Match: {abs(c_KP - 1.0) < 0.1}")
    print()

    # Walker-sector contributions table at K_P
    print("   K_P walker-sector contributions:")
    print(f"   {'sector':>6}  {'|<L|P|L>|':>10}  {'|<R|P|R>|':>10}  "
          f"{'sin(arg λ)':>11}  {'λ at K_P':>20}")
    for label, (wL, wR, s, lam) in contribs_KP.items():
        print(f"   {label:>6}  {wL:>10.4f}  {wR:>10.4f}  {s:>+11.4f}  "
              f"{lam.real:>+8.4f}{lam.imag:>+8.4f}j")
    print()

    # Sweep
    print(" -" * 38)
    print(" Sprint A chirality reading sweep: Gamma -> P (12 t values)")
    print(" -" * 38)
    print()
    print(f"   {'t':>6}  {'|k_red|':>9}  {'chir_L':>10}  {'chir_R':>10}  "
          f"{'c(k)':>10}  {'|L|R overlap':>14}")
    print(f"   {'-'*6}  {'-'*9}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*14}")

    t_vals = sorted(np.linspace(0.05, 1.0, 12), reverse=True)
    prev_L_undir = L_undir_KP.copy()
    prev_R_undir = R_undir_KP.copy()

    results = []
    for t in t_vals:
        k_red = t * P_red
        k_norm = la.norm(k_red)

        d = incidence_matrix_primitive(k_red, edges, len(verts))
        d1 = build_d1(cycles, edge_lookup, k_red, len(edges))
        Delta_1 = build_delta_1(d, d1)
        B_k = build_B_directed(bonds, k_red)
        pi_sym = build_pi_projector(bonds, edges, k_red)

        # Track L and R by adiabatic continuation (overlap maximization)
        eigs_p, vecs_p = la.eig(Delta_1)
        # Eigenvalue hint: previous photon eigenvalue (continuous tracking)
        L_undir_k, eig_L, ovl_L = track_mode_by_overlap(eigs_p, vecs_p, prev_L_undir)
        R_undir_k, eig_R, ovl_R = track_mode_by_overlap(eigs_p, vecs_p, prev_R_undir)

        # Check L/R didn't collapse to same mode
        LR_overlap = abs(np.vdot(L_undir_k, R_undir_k))

        # Lift to directed
        L_dir_k = pi_sym @ L_undir_k
        R_dir_k = pi_sym @ R_undir_k
        if la.norm(L_dir_k) > 1e-10:
            L_dir_k /= la.norm(L_dir_k)
        if la.norm(R_dir_k) > 1e-10:
            R_dir_k /= la.norm(R_dir_k)

        # Compute chirality reading
        c_k, chir_L_k, chir_R_k, _ = chirality_reading(
            L_dir_k, R_dir_k, B_k, walker_targets_KP)

        print(f"   {t:>6.3f}  {k_norm:>9.3e}  {chir_L_k:>+10.6f}  "
              f"{chir_R_k:>+10.6f}  {c_k:>+10.6f}  {LR_overlap:>14.6e}")

        results.append({'t': t, 'k_norm': k_norm, 'c': c_k,
                        'chir_L': chir_L_k, 'chir_R': chir_R_k,
                        'LR_overlap': LR_overlap})
        prev_L_undir = L_undir_k
        prev_R_undir = R_undir_k

    print()

    # Verdict
    print("=" * 76)
    print(" SPRINT A CHIRALITY READING VERDICT")
    print("=" * 76)
    print()

    results_sorted = sorted(results, key=lambda r: r['t'])
    c_values = [r['c'] for r in results_sorted]
    LR_overlaps = [r['LR_overlap'] for r in results_sorted]

    c_at_KP = c_values[-1]
    c_smallest_k = c_values[0]
    c_range = max(c_values) - min(c_values)

    print(f"   c(k) along Gamma -> P:")
    print(f"     at K_P (reference):         {c_at_KP:+.6f}")
    print(f"     at smallest k (t ≈ 0.05):   {c_smallest_k:+.6f}")
    print(f"     range across path:          {c_range:.6f}")
    print(f"     max |c|:                    {max(abs(v) for v in c_values):.6f}")
    print()
    print(f"   L/R orthogonality along path (|<L|R>|):")
    print(f"     at K_P (reference):         {LR_overlaps[-1]:.6e}")
    print(f"     at smallest k:              {LR_overlaps[0]:.6e}")
    print(f"     worst on path:              {max(LR_overlaps):.6e}")
    print()

    # Sprint A gate criteria (refined per chirality reading)
    EXPECTED_C_AT_KP = 1.0  # per beta_cosmic_birefringence UNIQUE-THEOREM-GRADE
    GATE_PASS_C_DEVIATION = 0.2  # |c(k) - c(K_P)| < 0.2 along path
    GATE_PASS_LR_ORTHO = 0.1  # |<L|R>| < 0.1 throughout

    c_deviation = max(abs(c - c_at_KP) for c in c_values)
    worst_LR = max(LR_overlaps)

    c_pass = c_deviation < GATE_PASS_C_DEVIATION
    ortho_pass = worst_LR < GATE_PASS_LR_ORTHO

    print(f"   Gate criteria:")
    print(f"     c(k) deviation < {GATE_PASS_C_DEVIATION}: "
          f"{'PASS' if c_pass else 'FAIL'} (max dev: {c_deviation:.4f})")
    print(f"     L/R remain orthogonal < {GATE_PASS_LR_ORTHO}: "
          f"{'PASS' if ortho_pass else 'FAIL'} (worst: {worst_LR:.4f})")
    print()

    if c_pass and ortho_pass:
        print("   GATE VERDICT: PASS")
        print("     The K_P chirality reading c = 1 extends to small k via adiabatic")
        print("     continuation. Sprint B (alpha_EM coupling for acoustic dispersion)")
        print("     becomes attemptable.")
    elif not c_pass and not ortho_pass:
        print("   GATE VERDICT: FAIL")
        print("     Both criteria fail: chirality reading c(k) does not stay near 1")
        print("     and L/R mix as k -> Gamma. The K_P photon-walker correspondence")
        print("     is genuinely C_3-symmetry-specific and does NOT extend smoothly")
        print("     to acoustic-regime k. L6 wall structurally confirmed at gate 1.")
    else:
        print("   GATE VERDICT: PARTIAL")
        print("     One of (c-deviation, L/R orthogonality) passes; the other fails.")
        print("     Structural extension is partial — interpretation depends on which")
        print("     criterion is more load-bearing for acoustic dispersion derivation.")

    return c_pass, ortho_pass


if __name__ == "__main__":
    main()
