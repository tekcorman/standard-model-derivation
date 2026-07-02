#!/usr/bin/env python3
"""
arg_h_path_b_q_qed_1loop_attempt.py — Approach Q for arg(h) Path B''.

Tests the c = 1 closure of β = c · sin(arg h) · α_EM via a 1-loop photon
self-energy calculation that uses ONLY B(k) and ∂_k B (no γ_7 transfer
through V_Ram). This bypasses the gauge-ambiguity obstacle that ruled
out P4 + L3-tree + L3-trace-survey.

Background
----------
The 12-dim directed-bond space at k_P decomposes as
    V_Ram (8-dim, Ramanujan, |B|²=2; walker excitations)
       ⊕
    V_kernel (4-dim, |B|²=1; gauge/photon sector)
with photon Hodge bundle ⊂ V_kernel entirely. The off-diagonal blocks
of B(P), Im(B(P)), C3_dir between V_Ram and V_kernel are all 0 at k_P.
But ∂_k B(k_P) (kinematic gradient of B(k) in the Bloch BZ) DOES bridge
V_Ram and V_kernel with Frobenius ~11.66 per axis (per
arg_h_path_b_l3_trace_survey.py Section F).

This script tests the standard QED-style 1-loop photon self-energy
diagram with photon-walker vertex = ∂_k B and walker propagator chosen
in several structurally-motivated forms.

1-loop self-energy ansatz
-------------------------
For each photon basis vector γ_α (α ∈ {L=ω, R=ω²}) and each propagator
form P:

    M_αβ(P) = Σ_axes ⟨γ_β| ∂_k B^a · P_walker · ∂_k B^a† |γ_α⟩

where P_walker is one of:
    (1) Vacuum resolvent at z=0:   B(k_P)^{-1} (defined on V_Ram only)
    (2) Unit-phasor propagator:    B(k_P)/|B(k_P)| = B/√2 on V_Ram
    (3) Anti-Hermitian projection: Im(B(k_P))/|B(k_P)| = Im(B)/√2
    (4) Phase extractor:           e^{i arg(B)} on V_Ram eigenmodes
    (5) Standard SD propagator:    (B(k_P)†)·(B(k_P) B(k_P)†)^{-1}
                                    = (k*−1)·B(k_P)† on V_Ram

Chirality coefficient: c = (M_LL − M_RR) / (2 · sin(arg h) · ⟨norm⟩)
where ⟨norm⟩ is whatever the natural normalization is (we'll inspect).

Hypothesis (Approach Q)
-----------------------
For SOME structurally-natural choice of P_walker, the 1-loop diagram
gives M_LL − M_RR = (vertex Frobenius)² · sin(arg h) · 2 (factor 2 from L/R
splitting, dim-2 photon bundle), exactly. Per-photon-state c = 1.

If c = 1: Approach Q closes Lemma 3 of theorem_dark_correction_mdl.md;
β graduates A− → THEOREM-GRADE; 4 P-rows graduate.

If no propagator form gives c = 1: the 1-loop with this vertex doesn't
close at tree level either; pivot to fully-momentum-integrated 1-loop
or fall back to ADOPTED-ARG-H-PROJECTION.

Run with:
    PYTHONPATH=. python3 proofs/foundations/arg_h_path_b_q_qed_1loop_attempt.py
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
    incidence_matrix_primitive,
)
from srs_photon_hodge import build_d1, build_edge_lookup
from srs_cycle_enumerator import enumerate_simple_cycles
from srs_photon_c3_chainmap import (
    build_C3_edge,
    build_delta_1,
    K_P_RED,
)
from srs_photon_chirality_coefficient import (
    build_pi_projector,
    build_C3_directed,
    build_B_directed,
)


TOL = 1e-10
H_EXACT = (math.sqrt(3) + 1j * math.sqrt(5)) / 2
ABS_H = math.sqrt(2.0)
SIN_ARG_H = math.sqrt(5.0 / 8.0)
ARG_H_DEG = math.degrees(math.atan2(math.sqrt(5), math.sqrt(3)))
omega = np.exp(2j * math.pi / 3)
omega2 = omega.conjugate()
PRINT_WIDTH = 78


def fmt_z(z, prec=6):
    return f"{z.real:+.{prec}f}{z.imag:+.{prec}f}j"


def main():
    print("=" * PRINT_WIDTH)
    print("Approach Q — 1-loop photon self-energy via ∂_k B vertex")
    print("Tests c = 1 closure of β = c · sin(arg h) · α_EM without γ_7 transfer")
    print("=" * PRINT_WIDTH)

    # -----------------------------------------------------------------------
    # Step 1: Geometry and apparatus.
    # -----------------------------------------------------------------------
    verts, lat = build_primitive_unit_cell()
    bonds = find_primitive_connectivity(verts, lat)
    edges = canonical_edges_primitive(bonds)
    edge_lookup = build_edge_lookup(edges)
    cycles = enumerate_simple_cycles(bonds, max_length=10)
    n_verts, n_edges, n_bonds = len(verts), len(edges), len(bonds)
    k_red = K_P_RED

    B = build_B_directed(bonds, k_red)
    Bevs, Bevecs = la.eig(B)
    ram_idx = [i for i, ev in enumerate(Bevs) if abs(abs(ev)**2 - 2.0) < 1e-5]
    ker_idx = [i for i, ev in enumerate(Bevs) if abs(abs(ev) - 1.0) < 1e-5]
    V_Ram, _ = la.qr(Bevecs[:, ram_idx])
    V_Ram = V_Ram[:, :8]
    V_ker, _ = la.qr(Bevecs[:, ker_idx])
    V_ker = V_ker[:, :4]

    print(f"\nStep 1: Apparatus")
    print(f"  V_Ram: 12×8 (|B|²=2),  V_kernel: 12×4 (|B|=1)")

    # -----------------------------------------------------------------------
    # Step 2: ∂_k B at k_P via finite difference (3 axes).
    # -----------------------------------------------------------------------
    print(f"\nStep 2: Compute ∂_k B(k_P) for axes 0, 1, 2")
    eps = 1e-6
    dB_dk = []
    for axis in range(3):
        k_plus = list(k_red).copy()
        k_minus = list(k_red).copy()
        k_plus[axis] += eps
        k_minus[axis] -= eps
        Bp = build_B_directed(bonds, np.array(k_plus))
        Bm = build_B_directed(bonds, np.array(k_minus))
        dB_dk.append((Bp - Bm) / (2 * eps))

    # Verify Frobenius norm of V_Ram ↔ V_kernel block (should be ~11.66 per axis)
    for axis in range(3):
        block = V_Ram.conj().T @ dB_dk[axis] @ V_ker
        print(f"  axis {axis}: ||V_Ram† · ∂B/∂k_{axis} · V_kernel||_F = "
              f"{la.norm(block):.4f}")

    # -----------------------------------------------------------------------
    # Step 3: Build photon basis {γ_L, γ_R} in V_kernel.
    # -----------------------------------------------------------------------
    print(f"\nStep 3: Build photon basis L = ω-irrep, R = ω²-irrep at P")
    d = incidence_matrix_primitive(k_red, edges, n_verts)
    d1 = build_d1(cycles, edge_lookup, k_red, n_edges)
    Delta_1 = build_delta_1(d, d1)
    eigs_full, vecs_full = la.eig(Delta_1)
    order = np.argsort(eigs_full.real)
    eigs_full = eigs_full[order]
    vecs_full = vecs_full[:, order]
    mask = np.abs(eigs_full.real - 36.0) < 1e-6
    photon_basis_undir = vecs_full[:, mask]   # 6 × 2 (undirected edges)
    Q_phot_undir, _ = la.qr(photon_basis_undir)
    Q_phot_undir = Q_phot_undir[:, :photon_basis_undir.shape[1]]

    pi = build_pi_projector(bonds, edges, k_red)
    Q_phot = pi @ Q_phot_undir   # 12 × 2 in directed-bond space

    # Diagonalize C_3 on photon (in undirected) and lift to L/R
    C3_e = build_C3_edge(edges, k_red)
    C3_photon = Q_phot_undir.conj().T @ C3_e @ Q_phot_undir
    eigvals_C3, eigvecs_C3 = la.eig(C3_photon)
    L_idx = int(np.argmin(np.abs(eigvals_C3 - omega)))
    R_idx = int(np.argmin(np.abs(eigvals_C3 - omega2)))
    L_vec_undir = eigvecs_C3[:, L_idx] / la.norm(eigvecs_C3[:, L_idx])
    R_vec_undir = eigvecs_C3[:, R_idx] / la.norm(eigvecs_C3[:, R_idx])
    # Lift to 12-dim directed bonds via π (and Q_phot_undir basis)
    L_vec = pi @ Q_phot_undir @ L_vec_undir   # 12-dim
    R_vec = pi @ Q_phot_undir @ R_vec_undir   # 12-dim
    print(f"  ||L||={la.norm(L_vec):.4f}, ||R||={la.norm(R_vec):.4f}")
    print(f"  ⟨L|R⟩ = {fmt_z(L_vec.conj() @ R_vec, 4)}  "
          f"(should be ~0)")
    L_vec = L_vec / la.norm(L_vec)
    R_vec = R_vec / la.norm(R_vec)

    # Also verify L lives in V_kernel (not V_Ram) per prior finding.
    L_in_VRam = la.norm(V_Ram.conj().T @ L_vec)
    L_in_Vker = la.norm(V_ker.conj().T @ L_vec)
    R_in_VRam = la.norm(V_Ram.conj().T @ R_vec)
    R_in_Vker = la.norm(V_ker.conj().T @ R_vec)
    print(f"  L overlap: V_Ram={L_in_VRam:.4f}, V_kernel={L_in_Vker:.4f}")
    print(f"  R overlap: V_Ram={R_in_VRam:.4f}, V_kernel={R_in_Vker:.4f}")

    # -----------------------------------------------------------------------
    # Step 4: Walker propagators. Define several candidate forms.
    # -----------------------------------------------------------------------
    print(f"\nStep 4: Build candidate walker propagators on V_Ram")
    B_VRam = V_Ram.conj().T @ B @ V_Ram   # 8×8
    Bevs_VRam, Bvec_VRam = la.eig(B_VRam)

    # P1: Vacuum resolvent at z=0: B^{-1} (well-defined since |B|² > 0 on V_Ram)
    P1 = la.inv(B_VRam)

    # P2: Unit-phasor propagator: B/|B| = B/√2 on V_Ram (since |B|² = 2 uniformly)
    P2 = B_VRam / ABS_H

    # P3: Anti-Hermitian unit-phasor: Im(B)/|B| = (B−B†)/(2i √2)
    Im_BVRam = (B_VRam - B_VRam.conj().T) / (2j)
    P3 = Im_BVRam / ABS_H

    # P4: Phase extractor: e^{i arg(B)} on each B-eigenmode
    #     = sum_n e^{i arg(λ_n)} |n⟩⟨n|
    P4 = np.zeros((8, 8), dtype=complex)
    for n in range(8):
        phase = Bevs_VRam[n] / abs(Bevs_VRam[n])
        # eigenvector (right): Bvec_VRam[:, n].
        # need biorthogonal left eigenvector: solve B† u = λ̄ u → u = (B†)-eigvec at λ̄
        # for diagonalizable B with distinct eigenvalues, use inverse of right-eigvec matrix
    # Use spectral decomposition via diagonalization: B_VRam = R · D · R^{-1}
    R_mat = Bvec_VRam
    D_mat = np.diag(Bevs_VRam)
    R_inv = la.inv(R_mat)
    # Verify: R · D · R^{-1} ≈ B_VRam
    assert la.norm(R_mat @ D_mat @ R_inv - B_VRam) < 1e-8
    # Phase-extractor propagator: replace D with diag(e^{i arg λ_n}) = D / |D|
    D_phase = np.diag([ev / abs(ev) for ev in Bevs_VRam])
    P4 = R_mat @ D_phase @ R_inv

    # P5: Standard SD propagator B†·(BB†)^{-1} = B† / |B|² = B† / 2 on V_Ram
    P5 = B_VRam.conj().T / 2.0

    # P6: Im of phase extractor: (P4 − P4†)/(2i)
    P6 = (P4 - P4.conj().T) / (2j)
    P6 = (P6 + P6.conj().T) / 2.0   # Hermitize numerically

    # P7: Resolvent at z = i·η (small imaginary): retarded propagator
    #     1/(B − iη) = (B + iη)/((B − iη)(B + iη)) = (B + iη)/(BB† + η²)
    # On V_Ram with |B|² = 2: 1/(B − iη) → at η→0+, this is just 1/B = P1.
    # For η small but nonzero, use full inverse.
    eta = 1e-3
    P7 = la.inv(B_VRam - 1j * eta * np.eye(8))

    # Lift each Pn to 12-dim (zero outside V_Ram).
    def lift_to_12(Pn):
        return V_Ram @ Pn @ V_Ram.conj().T

    propagators = {
        "P1: B^{-1}":              lift_to_12(P1),
        "P2: B/|B|":               lift_to_12(P2),
        "P3: Im(B)/|B|":           lift_to_12(P3),
        "P4: phase exp(i·arg(B))": lift_to_12(P4),
        "P5: B†/|B|²":             lift_to_12(P5),
        "P6: Im(phase)":           lift_to_12(P6),
        "P7: 1/(B-iη)":            lift_to_12(P7),
    }

    # -----------------------------------------------------------------------
    # Step 5: Compute 1-loop M_αβ for each propagator + axis.
    # -----------------------------------------------------------------------
    print(f"\nStep 5: 1-loop M_αβ = Σ_axes ⟨γ_β| ∂_k B^a · P_walker · ∂_k B^a† |γ_α⟩")
    print(f"-" * PRINT_WIDTH)

    def compute_M(P_walker_12):
        """Sum over 3 axes."""
        M_LL = 0
        M_LR = 0
        M_RL = 0
        M_RR = 0
        for axis in range(3):
            V = dB_dk[axis]
            # The "outgoing" leg is V·v from initial photon state v;
            # propagator P; "incoming" leg is V†·v back to photon basis.
            # Diagram: V → P → V†.
            # ⟨γ_β| V · P · V† |γ_α⟩ — full 12×12 multiplication.
            op = V @ P_walker_12 @ V.conj().T
            M_LL += L_vec.conj() @ op @ L_vec
            M_LR += L_vec.conj() @ op @ R_vec
            M_RL += R_vec.conj() @ op @ L_vec
            M_RR += R_vec.conj() @ op @ R_vec
        return M_LL, M_LR, M_RL, M_RR

    # Baseline: |∂_k B|² Frobenius norm on V_Ram → V_kernel block
    F_total = sum(la.norm(V_Ram.conj().T @ dB_dk[a] @ V_ker)**2 for a in range(3))
    print(f"  Reference: Σ_axes ||V_Ram† · ∂_k B^a · V_kernel||_F² = {F_total:.4f}")
    print(f"  sin(arg h) = √(5/8) = {SIN_ARG_H:.6f}")
    print()

    print(f"  {'Propagator':<30}  {'M_LL':>14}  {'M_RR':>14}  "
          f"{'(M_LL−M_RR)/2':>16}  {'/sin(arg h)':>14}")
    print(f"  {'-'*30}  {'-'*14}  {'-'*14}  {'-'*16}  {'-'*14}")
    results = {}
    for name, P12 in propagators.items():
        M_LL, M_LR, M_RL, M_RR = compute_M(P12)
        diff_half = (M_LL - M_RR) / 2.0
        # Take Im part for chirality (β is a phase, parity-odd)
        # But sometimes the full split is in real or imag — print both
        results[name] = (M_LL, M_LR, M_RL, M_RR, diff_half)
        ratio_re = diff_half.real / SIN_ARG_H
        ratio_im = diff_half.imag / SIN_ARG_H
        ratio_abs = abs(diff_half) / SIN_ARG_H
        print(f"  {name:<30}  {fmt_z(M_LL, 4):>14}  {fmt_z(M_RR, 4):>14}  "
              f"{fmt_z(diff_half, 4):>16}  Re={ratio_re:+.4f} "
              f"Im={ratio_im:+.4f} |·|={ratio_abs:.4f}")

    # -----------------------------------------------------------------------
    # Step 6: Off-diagonal M_LR, M_RL — should be 0 by Schur (ω vs ω² inequiv).
    # -----------------------------------------------------------------------
    print(f"\nStep 6: Schur off-diagonal check (ω vs ω² inequivalent → M_LR=M_RL=0)")
    print(f"-" * PRINT_WIDTH)
    schur_max = 0.0
    for name, (M_LL, M_LR, M_RL, M_RR, _) in results.items():
        m = max(abs(M_LR), abs(M_RL))
        schur_max = max(schur_max, m)
        print(f"  {name:<30}  |M_LR|={abs(M_LR):.3e}  |M_RL|={abs(M_RL):.3e}  "
              f"{'PASS' if m < 1e-8 else 'FAIL'}")
    print(f"  max |off-diag| over all propagators: {schur_max:.3e}")

    # -----------------------------------------------------------------------
    # Step 7: Verdict.
    # -----------------------------------------------------------------------
    print(f"\n" + "=" * PRINT_WIDTH)
    print(f"VERDICT")
    print(f"=" * PRINT_WIDTH)

    landed = []
    for name, (_, _, _, _, diff_half) in results.items():
        # Look for c = ±1 (Re or Im or |·|) modulo Frobenius normalization
        for measure_name, val in [("Re", diff_half.real), ("Im", diff_half.imag),
                                  ("|·|", abs(diff_half))]:
            ratio = val / SIN_ARG_H
            # Test c = ±1, ±F_total, ±F_total·integer, ±sin(arg h)·integer
            for target, label in [(1.0, "c=1"), (-1.0, "c=-1"),
                                   (F_total, "c=F_total"),
                                   (-F_total, "c=-F_total"),
                                   (2.0, "c=2"), (-2.0, "c=-2")]:
                if abs(ratio - target) < 1e-6:
                    landed.append((name, measure_name, target, label, val, ratio))

    if landed:
        print(f"\n  Possible structural matches:")
        for entry in landed:
            print(f"    {entry[0]} · {entry[1]} = {entry[4]:+.6f} = "
                  f"{entry[5]:+.4f}·sin(arg h) ≈ {entry[3]}")
        print(f"\n  CANDIDATE LANDED — verify carefully against gauge invariance and Frobenius scaling.")
    else:
        print(f"\n  No clean c = ±1 match found across propagator catalog.")
        print(f"  Best |Re/sin(arg h)| value: {max(abs(d.real) / SIN_ARG_H for _,_,_,_,d in results.values()):.4f}")
        print(f"  Best |Im/sin(arg h)| value: {max(abs(d.imag) / SIN_ARG_H for _,_,_,_,d in results.values()):.4f}")
        print(f"  Σ Frobenius normalization F_total = {F_total:.4f}")
        print(f"\n  Candidate not landed at this 1-loop ansatz.")
        print(f"  Possible follow-ons: (a) full BZ momentum integral, (b) different vertex")
        print(f"  structure (not ∂_k B), (c) anomaly-style Fujikawa regularized trace.")

    print(f"\n" + "=" * PRINT_WIDTH)
    print(f"OK: arg_h_path_b_q_qed_1loop_attempt completed without errors")


if __name__ == "__main__":
    main()
