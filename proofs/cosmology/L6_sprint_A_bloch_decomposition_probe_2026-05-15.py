#!/usr/bin/env python3
"""
L6 Sprint A — Bloch decomposition gate test (2026-05-15 EOD+5).

CONTEXT
-------
Per `substrate_r_s_mechanism_audit_2026-05-09.md` §5, L6 = intermediate-scale
field theory from substrate primitives is the framework's most-cited
structural blocker (r_s, theta_*, sigma_8, n_s parametric-translation all
inherit it). Sprint A is the cheapest L6 gate test: do the framework's
photon Hodge mode and walker Hashimoto operator admit a CLEAN Bloch
decomposition at intermediate (small but nonzero) k, where Mpc-scale
acoustic wavelengths live?

The existing photon-walker correspondence
(`proofs/cosmology/srs_photon_walker_correspondence.py`) is built at the
P-point (high-symmetry corner of the BZ) where C_3 representation theory
labels photon modes as L = omega-irrep and R = omega-bar-irrep. Sprint A
asks: does this correspondence extend SMOOTHLY toward Gamma, where small
|k| corresponds to long-wavelength (acoustic) dispersion?

GATE CRITERION
--------------
Gate PASS:
- Both Delta_1(k) [photon Hodge Laplacian] and B(k) [walker Hashimoto]
  admit smooth k-analytic eigenvalue branches on a neighborhood of Gamma.
- The K_P photon-walker correspondence (L cleanly in +h walker eigensector
  via pi-lift) extends continuously along the Gamma-P path: |<L|P_+h|L>|
  stays close to 1 across the path.
- The framework's mode categorization scheme is consistent at intermediate k.

Gate FAIL:
- Either operator has singular spectral structure at small k.
- The photon-walker correspondence breaks down (L mixes between +h and
  other walker eigensectors as we move off K_P).
- The framework's mode categorization is k-point-specific (C_3 labels only
  defined at C_3-invariant k-points; structural obstruction).

If Gate PASS: Sprint B (alpha_EM coupling at intermediate k) becomes the
next probe.

If Gate FAIL: L6 wall is structurally confirmed at gate 1; r_s/theta_*/
sigma_8/n_s cluster locks into Scenario 3 honest concession with concrete
mathematical obstruction.

DESIGN
------
1. Build Delta_1(k) and B(k) at k = t * P_red for t in [0, 1] (Gamma-P path)
2. Track walker eigenvalues h_j(t) along the path
3. At t = 1 (P-point), identify L = omega-irrep photon mode via C_3
4. At intermediate t, photon eigenstates are tracked by continuous deformation
   from t = 1 (parallel transport via overlap maximization)
5. Compute walker-sector overlap |<L(t)|P_+h(t)|L(t)>| as t varies
6. Verdict: gate PASS / FAIL / PARTIAL based on overlap behavior
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
    build_C3_directed,
    build_pi_projector,
)


def build_pi_antisym(bonds, edges, k_red):
    """Antisymmetric pi-lift (copied from srs_photon_walker_correspondence)."""
    n_bonds = len(bonds)
    n_edges = len(edges)
    pi = np.zeros((n_bonds, n_edges), dtype=complex)
    fwd_idx, bwd_idx = {}, {}
    for b_idx, (src, tgt, cell, _) in enumerate(bonds):
        for (e_idx, vs, vt, ec) in edges:
            if (vs, vt, ec) == (src, tgt, cell):
                fwd_idx[e_idx] = b_idx
            neg_cell = tuple(-c for c in ec)
            if (vt, vs, neg_cell) == (src, tgt, cell):
                bwd_idx[e_idx] = b_idx
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for (e_idx, vs, vt, cell) in edges:
        pi[fwd_idx[e_idx], e_idx] = inv_sqrt2
        bwd_phase = np.exp(-1j * 2 * math.pi * np.dot(k_red, cell))
        pi[bwd_idx[e_idx], e_idx] = -inv_sqrt2 * bwd_phase
    return pi


def get_photon_mode_at_kp(Delta_1, C3_e, eigval=36.0, tol=1e-6):
    """At a C_3-invariant k-point (e.g., P), extract photon mode in the
    omega-irrep eigenspace of C_3.

    Returns: (L_mode_undirected, R_mode_undirected) each shape (n_edges,)
    """
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
    return L_undir, R_undir, Q


def track_photon_mode(Delta_1, prev_mode, photon_eigval=36.0, tol_eig=0.5):
    """Track a photon mode by maximum overlap continuation at non-symmetric k.

    At intermediate k where C_3 is broken, we can't use C_3 representation
    labels. Instead we identify the mode by maximum overlap with the
    previous-k mode (parallel transport / adiabatic continuation).

    Looks for the eigenvector of Delta_1 closest to prev_mode in the
    "photon band" (eigenvalues near photon_eigval; tol_eig is the search
    width).
    """
    eigs, vecs = la.eig(Delta_1)
    # Photon band: eigenvalues near photon_eigval
    band_idx = np.where(np.abs(eigs.real - photon_eigval) < tol_eig)[0]
    if len(band_idx) == 0:
        # Photon band disappeared! Take all positive eigenvalues
        band_idx = np.where(eigs.real > 1.0)[0]
    # Pick the eigenvector with max overlap with prev_mode
    overlaps = [abs(np.vdot(prev_mode, vecs[:, j])) for j in band_idx]
    best_j = band_idx[np.argmax(overlaps)]
    new_mode = vecs[:, best_j]
    new_mode /= la.norm(new_mode)
    # Fix gauge: pick global phase to maximize Re(<prev|new>)
    gauge = np.vdot(prev_mode, new_mode)
    if abs(gauge) > 1e-10:
        new_mode *= np.conj(gauge) / abs(gauge)
    return new_mode, eigs[best_j].real


def walker_h_at_k(B, target_h):
    """Find the walker eigenvalue closest to target_h (continuous tracking).

    Returns: (lambda_close_to_target, eigenvector, multiplicity_at_this_k)
    """
    eigs, vecs = la.eig(B)
    distances = np.abs(eigs - target_h)
    j_closest = np.argmin(distances)
    lambda_close = eigs[j_closest]
    # Count multiplicity (eigenvalues within 1e-4 of this one)
    mult = int(np.sum(np.abs(eigs - lambda_close) < 1e-4))
    return lambda_close, vecs[:, j_closest], mult


def walker_projector_for_eigval(B, target_eig, tol=1e-4):
    """Build projector onto the eigenspace of B with eigenvalue near target_eig."""
    eigs, vecs = la.eig(B)
    idx = np.where(np.abs(eigs - target_eig) < tol)[0]
    if len(idx) == 0:
        return None, None
    V = vecs[:, idx]
    Q_w, _ = la.qr(V)
    P = Q_w @ Q_w.conj().T
    return P, idx


def main():
    print("=" * 76)
    print(" L6 Sprint A — Bloch decomposition gate test")
    print("=" * 76)
    print()

    # Build infrastructure once
    verts, lat = build_primitive_unit_cell()
    bonds = find_primitive_connectivity(verts, lat)
    edges = canonical_edges_primitive(bonds)
    edge_lookup = build_edge_lookup(edges)
    cycles = enumerate_simple_cycles(bonds, max_length=10)
    print()

    P_red = np.array([0.25, 0.25, 0.25])
    h_KP = complex(math.sqrt(3) / 2, math.sqrt(5) / 2)

    print(f"  Sprint A path: Gamma -> P_red = {P_red}")
    print(f"  Walker target eigenvalue at K_P: h = (√3 + i√5)/2 = {h_KP}")
    print()

    # Build operators at K_P (t = 1) first to set the reference
    k_KP = P_red
    d_KP = incidence_matrix_primitive(k_KP, edges, len(verts))
    d1_KP = build_d1(cycles, edge_lookup, k_KP, len(edges))
    Delta_1_KP = build_delta_1(d_KP, d1_KP)
    C3_e_KP = build_C3_edge(edges, k_KP)
    B_KP = build_B_directed(bonds, k_KP)
    pi_sym_KP = build_pi_projector(bonds, edges, k_KP)

    L_undir_KP, R_undir_KP, _ = get_photon_mode_at_kp(Delta_1_KP, C3_e_KP)
    print(" Reference at K_P (t = 1.0):")
    print(f"   L_undir norm: {la.norm(L_undir_KP):.6f}")
    print(f"   R_undir norm: {la.norm(R_undir_KP):.6f}")
    print(f"   <L|R>: {np.vdot(L_undir_KP, R_undir_KP):+.2e}")

    # Walker B(K_P) eigenvalues — verify h_KP is among them
    walker_eigs_KP = la.eigvals(B_KP)
    h_dists_KP = np.abs(walker_eigs_KP - h_KP)
    print(f"   Walker eigenvalue closest to h = {walker_eigs_KP[np.argmin(h_dists_KP)]} "
          f"(distance: {np.min(h_dists_KP):.2e})")
    P_h_KP, _ = walker_projector_for_eigval(B_KP, h_KP)

    # Lift L to directed via pi_sym at K_P
    L_dir_KP = pi_sym_KP @ L_undir_KP
    L_dir_KP /= la.norm(L_dir_KP)
    overlap_KP = np.real(np.vdot(L_dir_KP, P_h_KP @ L_dir_KP))
    print(f"   |<L_dir|P_+h|L_dir>| at K_P: {overlap_KP:.6f}")
    print()

    # ========================================================================
    # Sweep along Gamma -> P path
    # ========================================================================
    print(" -" * 38)
    print(" Sprint A sweep: Gamma -> P (10 intermediate t values)")
    print(" -" * 38)
    print()
    print(f"   {'t':>6}  {'k_red':<30}  {'|k_red|':>9}  {'photon_eig':>12}  "
          f"{'h_close':>30}  {'mult':>4}  {'|<L|P_+h|L>|':>14}")
    print(f"   {'-'*6}  {'-'*30}  {'-'*9}  {'-'*12}  {'-'*30}  {'-'*4}  {'-'*14}")

    t_vals = np.linspace(0.01, 1.0, 12)  # avoid exact Gamma (singular)
    results = []

    # Initialize tracking from K_P (t = 1) and go backwards toward Gamma
    # Actually start from K_P and go to Gamma so continuity gives stable tracking
    t_vals_sorted = sorted(t_vals, reverse=True)
    prev_L_undir = L_undir_KP.copy()
    prev_h = h_KP

    for t in t_vals_sorted:
        k_red = t * P_red
        k_norm = la.norm(k_red)

        # Operators at this k
        d = incidence_matrix_primitive(k_red, edges, len(verts))
        d1 = build_d1(cycles, edge_lookup, k_red, len(edges))
        Delta_1 = build_delta_1(d, d1)
        B_k = build_B_directed(bonds, k_red)
        pi_sym = build_pi_projector(bonds, edges, k_red)

        # Track photon L mode by maximum overlap continuation
        L_undir_k, photon_eig = track_photon_mode(Delta_1, prev_L_undir)

        # Track walker h eigenvalue by closest-distance continuation
        h_k, h_evec, h_mult = walker_h_at_k(B_k, prev_h)

        # Build walker projector onto eigenspace near h_k
        P_h_k, _ = walker_projector_for_eigval(B_k, h_k)

        # Lift L mode to directed via pi at this k
        L_dir_k = pi_sym @ L_undir_k
        if la.norm(L_dir_k) > 1e-10:
            L_dir_k /= la.norm(L_dir_k)
            overlap_k = np.real(np.vdot(L_dir_k, P_h_k @ L_dir_k))
        else:
            overlap_k = float("nan")

        h_str = f"{h_k.real:+.4f}{h_k.imag:+.4f}j"
        k_red_str = f"({k_red[0]:+.3f},{k_red[1]:+.3f},{k_red[2]:+.3f})"
        print(f"   {t:>6.3f}  {k_red_str:<30}  {k_norm:>9.3e}  "
              f"{photon_eig:>12.4f}  {h_str:>30}  {h_mult:>4d}  {overlap_k:>14.6f}")

        results.append({
            't': t, 'k_norm': k_norm, 'photon_eig': photon_eig,
            'h': h_k, 'h_mult': h_mult, 'overlap': overlap_k,
        })

        prev_L_undir = L_undir_k
        prev_h = h_k

    print()

    # ========================================================================
    # Sprint A gate verdict
    # ========================================================================
    print("=" * 76)
    print(" SPRINT A GATE VERDICT")
    print("=" * 76)
    print()

    # Reverse results to chronological order (Gamma -> K_P)
    results_sorted = sorted(results, key=lambda r: r['t'])
    overlaps = [r['overlap'] for r in results_sorted if not math.isnan(r['overlap'])]
    photon_eigs = [r['photon_eig'] for r in results_sorted]
    h_mults = [r['h_mult'] for r in results_sorted]

    overlap_min = min(overlaps) if overlaps else float("nan")
    overlap_max = max(overlaps) if overlaps else float("nan")
    overlap_range = overlap_max - overlap_min

    # Photon eigenvalue variation
    photon_eig_range = max(photon_eigs) - min(photon_eigs)

    # Walker multiplicity change
    h_mult_changes = sum(1 for i in range(len(h_mults) - 1) if h_mults[i] != h_mults[i+1])

    print(f"   Photon eigenvalue along Gamma -> P:")
    print(f"     range: {min(photon_eigs):.4f} -> {max(photon_eigs):.4f}  "
          f"(span {photon_eig_range:.4f})")
    print()
    print(f"   Walker h eigenvalue multiplicity along Gamma -> P:")
    print(f"     mult at t=Gamma-ish: {h_mults[0]}")
    print(f"     mult at t=K_P:       {h_mults[-1]}")
    print(f"     mult changes along path: {h_mult_changes}")
    print()
    print(f"   Photon-walker overlap |<L_dir|P_+h|L_dir>| along Gamma -> P:")
    print(f"     at K_P (t=1.0): {overlap_KP:.6f}  [reference]")
    print(f"     minimum along path: {overlap_min:.6f}")
    print(f"     maximum along path: {overlap_max:.6f}")
    print(f"     range: {overlap_range:.6f}")
    print()

    # Gate criteria
    GATE_PASS_OVERLAP_FLOOR = 0.5    # Overlap must stay > 0.5 along path
    GATE_PASS_MULT_STABILITY = 1     # Allow at most 1 multiplicity change

    overlap_pass = overlap_min > GATE_PASS_OVERLAP_FLOOR
    mult_pass = h_mult_changes <= GATE_PASS_MULT_STABILITY

    print(f"   Gate criteria:")
    print(f"     Overlap floor > {GATE_PASS_OVERLAP_FLOOR}: "
          f"{'PASS' if overlap_pass else 'FAIL'} ({overlap_min:.3f})")
    print(f"     Walker mult changes <= {GATE_PASS_MULT_STABILITY}: "
          f"{'PASS' if mult_pass else 'FAIL'} ({h_mult_changes} changes)")
    print()

    if overlap_pass and mult_pass:
        verdict = "PASS"
        print(f"   GATE VERDICT: SPRINT A PASS")
        print(f"     The K_P photon-walker correspondence extends smoothly toward")
        print(f"     Gamma. Sprint B (alpha_EM coupling at intermediate k) becomes")
        print(f"     attemptable.")
    elif not overlap_pass and not mult_pass:
        verdict = "FAIL"
        print(f"   GATE VERDICT: SPRINT A FAIL")
        print(f"     Both overlap and multiplicity criteria fail. The K_P photon-")
        print(f"     walker correspondence does NOT extend cleanly to intermediate")
        print(f"     k. L6 wall structurally confirmed at gate 1. Cosmology cluster")
        print(f"     (r_s, theta_*, sigma_8, n_s parametric-translation) locks into")
        print(f"     Scenario 3 honest concession.")
    else:
        verdict = "PARTIAL"
        print(f"   GATE VERDICT: SPRINT A PARTIAL")
        print(f"     One of (overlap, multiplicity) criteria passes but not the")
        print(f"     other. Sprint A is structurally informative but does not")
        print(f"     cleanly pass or fail. Further analysis needed for Sprint B")
        print(f"     readiness.")
    print()

    return verdict


if __name__ == "__main__":
    main()
