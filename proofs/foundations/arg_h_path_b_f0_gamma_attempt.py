#!/usr/bin/env python3
"""
arg_h_path_b_f0_gamma_attempt.py — F0_γ for arg(h) Path B''.

Attempts the F0_γ–F3_γ+Spectral_γ recipe for c = 1 in β = c·sin(arg h)·α_EM,
following the structural pattern of `proofs/foundations/dark_feshbach_a2_closure.py`
(Higgs c = 5/12 chain).

Higgs analog:
  F0:  vertex couples to all k* outgoing × k* incoming directed-edge pairs
  F1:  H_PQ · H_QP = adjacency  →  k*² = 9 pair multiplicity
  F2:  backtrack pairs (i,i) contribute 0 girth cycles
  F3:  C/C̄ MDL-equivalent  →  n_g = 15 unoriented (not 30)
  Spec: H(k_P)² = k*·I_{N_ATOMS}  →  1/N_ATOMS = 1/4
  c = n_g / (k*² · N_ATOMS) = 15/(9·4) = 5/12

Photon analog needs:
  F0_γ:  edge process for photon Hodge mode at k_P
  F1_γ:  algebraic identity giving edge-pair multiplicity for photon
  F2_γ:  zero-contribution theorem analog
  F3_γ:  counting reduction analog
  Spec_γ: per-mode normalization on photon Hodge bundle
  c = 1

Constraints from prior session falsifications (per `theorem_beta_coefficient_unity.md`):
  C1: F0_γ vertex CANNOT be Im(B) projected via π (parity-filtered)
  C2: F0_γ must be a V_kernel↔V_Ram cross-sector coupling (photon ⊥ V_Ram at k_P)
  C3: chirality projector must be gauge-CANONICAL (no γ_7 via B6)
  C4: vertex cannot reduce to ∂_k B at single momentum (parity-EVEN)

This script tests the candidate chirality projector
    γ_7^B := sign(Im(B(P)))   on V_Ram
which is intrinsic to B's spectrum (gauge-canonical, satisfies C3) and
combines with various spectral structures to test for c = 1.

Mathematical content:
  - γ_7^B = +1 on B-eigenstates with Im(λ) > 0 (h, -h̄)
  - γ_7^B = -1 on B-eigenstates with Im(λ) < 0 (h̄, -h)
  - On C_3 sector ω of V_Ram (containing -h, -h̄):
      γ_7^B|V_Ram^ω = diag(-1, +1) in (-h, -h̄) eigenbasis
      tr_ω[γ_7^B · Im(B)/|h|] = (-1)(-sin(arg h)) + (+1)(+sin(arg h)) = 2·sin(arg h)
  - This trace is gauge-INVARIANT (γ_7^B intrinsic to B; no A intertwiner)

The script:
  1. Builds γ_7^B from B(P) spectral decomposition.
  2. Verifies γ_7^B is gauge-invariant (function of B only, not A).
  3. Checks γ_7^B's algebraic properties: γ_7^B² = I, tr = 0, [γ_7^B, U_C3] = 0.
  4. Computes the photon 1-loop self-energy with γ_7^B inserted in the
     chirality-projected diagram, in multiple ansatz forms.
  5. Reports whether (M_LL − M_RR)/2 = sin(arg h)·c at c = 1.

Run with:
    PYTHONPATH=. python3 proofs/foundations/arg_h_path_b_f0_gamma_attempt.py
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
from srs_photon_c3_chainmap import build_C3_edge, build_delta_1, K_P_RED
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


def build_gamma7_B(B_VRam):
    """Build γ_7^B := sign(Im(B(P))) on V_Ram via spectral decomposition.

    γ_7^B has eigenvalue +1 on B-eigenstates with Im(λ) > 0,
    and eigenvalue -1 on B-eigenstates with Im(λ) < 0.

    Hermitian, satisfies (γ_7^B)² = I, intrinsic to B(P) spectrum
    (no A intertwiner used → gauge-canonical).
    """
    evals, evecs = la.eig(B_VRam)
    n = len(evals)
    G = np.zeros((n, n), dtype=complex)
    # Diagonalize: B = R · D · R^{-1}, build sign(Im(D)) and similarity-transform.
    R = evecs
    R_inv = la.inv(R)
    sgn_diag = np.diag([1.0 if ev.imag > 0 else -1.0 for ev in evals])
    G = R @ sgn_diag @ R_inv
    # Hermitize numerically
    G = (G + G.conj().T) / 2.0
    return G


def main():
    print("=" * PRINT_WIDTH)
    print("F0_γ — γ_7^B chirality projector intrinsic to B(P) spectrum")
    print("Tests c = 1 in β = c·sin(arg h)·α_EM via gauge-canonical chirality")
    print("=" * PRINT_WIDTH)

    # -----------------------------------------------------------------------
    # Apparatus.
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

    print(f"\nV_Ram: 12×8 (|B|²=2),  V_kernel: 12×4 (|B|=1)")

    B_VRam = V_Ram.conj().T @ B @ V_Ram

    # -----------------------------------------------------------------------
    # Step 1: Build γ_7^B = sign(Im(B)) and verify properties.
    # -----------------------------------------------------------------------
    print("\nStep 1 — Build γ_7^B := sign(Im(B|V_Ram)) and verify")
    print("-" * PRINT_WIDTH)
    G7B = build_gamma7_B(B_VRam)
    err_h = la.norm(G7B - G7B.conj().T)
    err_sq = la.norm(G7B @ G7B - np.eye(8))
    tr_G7B = np.trace(G7B)
    print(f"  ||γ_7^B − (γ_7^B)†|| = {err_h:.2e}    (Hermiticity)")
    print(f"  ||(γ_7^B)² − I_8||   = {err_sq:.2e}    (involution)")
    print(f"  tr(γ_7^B)            = {tr_G7B:+.4e}    (expect 0)")

    # Eigenvalues should be ±1 with equal multiplicity 4+4
    evs_G7B = la.eigvalsh(G7B)
    from collections import Counter
    cc = Counter(int(round(ev)) for ev in evs_G7B)
    print(f"  γ_7^B eigenvalues on V_Ram: +1 mult={cc[+1]}, -1 mult={cc[-1]}")
    assert cc[+1] == 4 and cc[-1] == 4, f"γ_7^B does not split V_Ram as 4+4: {dict(cc)}"

    # Verify [γ_7^B, U_C3] = 0 (B commutes with C_3 → γ_7^B should too)
    C3_dir = build_C3_directed(bonds)
    U_C3_VRam = V_Ram.conj().T @ C3_dir @ V_Ram
    comm = la.norm(G7B @ U_C3_VRam - U_C3_VRam @ G7B)
    print(f"  ||[γ_7^B, U_C3]|| = {comm:.2e}    (expect ≈ 0; both commute with B)")

    # -----------------------------------------------------------------------
    # Step 2: γ_7^B is gauge-canonical (function of B only).
    # -----------------------------------------------------------------------
    print("\nStep 2 — Gauge invariance check (vs B6-via-A γ_7 ambiguity)")
    print("-" * PRINT_WIDTH)
    # Apply random unitary on V_Ram (simulates basis-of-V_Ram gauge);
    # γ_7^B should be COVARIANT under change-of-basis (not change in spectral content).
    rng = np.random.default_rng(2026)
    for trial in range(3):
        Z = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
        U_basis, _ = la.qr(Z)
        # Rotate V_Ram basis: B_VRam → U_basis · B_VRam · U_basis†
        B_p = U_basis @ B_VRam @ U_basis.conj().T
        G7B_p = build_gamma7_B(B_p)
        # Check that γ_7^B transforms covariantly: U·γ_7^B·U† = γ_7^B'
        # (i.e., the spectral function is well-defined regardless of basis)
        G7B_expected = U_basis @ G7B @ U_basis.conj().T
        err_cov = la.norm(G7B_p - G7B_expected)
        print(f"  Trial {trial+1}: ||γ_7^B(rotated B) − U·γ_7^B·U†|| = {err_cov:.2e}")
        assert err_cov < 1e-8, f"γ_7^B not basis-covariant: {err_cov}"
    print(f"  ✓ γ_7^B is intrinsic to B's spectrum (basis-covariant, gauge-canonical)")

    # -----------------------------------------------------------------------
    # Step 3: γ_7^B per C_3 sector — is it traceless on each sector?
    # -----------------------------------------------------------------------
    print("\nStep 3 — γ_7^B restricted to each C_3 isotypic sector of V_Ram")
    print("-" * PRINT_WIDTH)

    def c3_isotypic_basis(M, dim, tol=0.1):
        evals, evecs = la.eig(M)
        groups = {'1': [], 'w': [], 'w2': []}
        for i, ev in enumerate(evals):
            if abs(ev - 1.0) < tol:
                groups['1'].append(evecs[:, i])
            elif abs(ev - omega) < tol:
                groups['w'].append(evecs[:, i])
            elif abs(ev - omega2) < tol:
                groups['w2'].append(evecs[:, i])
        bases = {}
        for label, vecs in groups.items():
            if vecs:
                mat = np.column_stack(vecs)
                Q, _ = la.qr(mat)
                bases[label] = Q[:, :len(vecs)]
            else:
                bases[label] = np.zeros((dim, 0), dtype=complex)
        return bases

    bases_VRam = c3_isotypic_basis(U_C3_VRam, 8)

    for label in ['1', 'w', 'w2']:
        Q = bases_VRam[label]
        d = Q.shape[1]
        if d == 0:
            continue
        G7B_sec = Q.conj().T @ G7B @ Q
        evs_sec = sorted(la.eigvalsh(G7B_sec))
        tr_sec = np.trace(G7B_sec).real
        B_sec = Q.conj().T @ B_VRam @ Q
        B_evs = sorted(la.eigvals(B_sec), key=lambda z: (z.real, z.imag))
        print(f"  Sector C_3='{label}' (dim={d}):")
        print(f"    γ_7^B|sector eigenvalues: {[f'{ev:+.4f}' for ev in evs_sec]}")
        print(f"    γ_7^B|sector trace      : {tr_sec:+.4f}")
        print(f"    B|sector eigenvalues    : {[fmt_z(ev, 4) for ev in B_evs]}")

    # -----------------------------------------------------------------------
    # Step 4: Candidate trace tr_ω[γ_7^B · Im(B)/|h|] = ?
    # -----------------------------------------------------------------------
    print("\nStep 4 — Candidate trace on each C_3 sector")
    print("-" * PRINT_WIDTH)

    Im_B = (B_VRam - B_VRam.conj().T) / (2j)
    Im_B = (Im_B + Im_B.conj().T) / 2.0
    target_op = Im_B / ABS_H   # Im(B)/|h| (unit-phasor parity-odd projection)

    for label in ['1', 'w', 'w2']:
        Q = bases_VRam[label]
        if Q.shape[1] == 0:
            continue
        G7B_sec = Q.conj().T @ G7B @ Q
        F_sec = Q.conj().T @ target_op @ Q
        tr = np.trace(G7B_sec @ F_sec)
        ratio = tr / SIN_ARG_H
        print(f"  Sector '{label}': tr[γ_7^B · Im(B)/|h|] = {fmt_z(tr, 5)}  "
              f"(/sin(arg h) = {fmt_z(ratio, 4)})")

    # Sum and difference over ω, ω²
    Q_w = bases_VRam['w']
    Q_w2 = bases_VRam['w2']
    Q_nontriv = np.column_stack([Q_w, Q_w2])
    G7B_nt = Q_nontriv.conj().T @ G7B @ Q_nontriv
    F_nt = Q_nontriv.conj().T @ target_op @ Q_nontriv
    tr_sum = np.trace(G7B_nt @ F_nt)
    tr_diff = (np.trace(Q_w.conj().T @ G7B @ Q_w @ Q_w.conj().T @ target_op @ Q_w)
               - np.trace(Q_w2.conj().T @ G7B @ Q_w2 @ Q_w2.conj().T @ target_op @ Q_w2))
    print(f"\n  Sum (ω+ω²):       {fmt_z(tr_sum, 5)}    (/sin(arg h) = "
          f"{(tr_sum / SIN_ARG_H).real:+.4f})")
    print(f"  Difference (ω-ω²): {fmt_z(tr_diff, 5)}    (/sin(arg h) = "
          f"{(tr_diff / SIN_ARG_H).real:+.4f})")

    # -----------------------------------------------------------------------
    # Step 5: Gauge-invariance test.
    # -----------------------------------------------------------------------
    print("\nStep 5 — Gauge-invariance of the trace under U(8) on V_Ram basis")
    print("-" * PRINT_WIDTH)
    # Apply C_3-equivariant random unitary (acts on each C_3 sector independently)
    rng2 = np.random.default_rng(7)
    for trial in range(5):
        # C_3-equivariant unitary: independent random U on each C_3 sector basis
        Z1 = rng2.standard_normal((4, 4)) + 1j * rng2.standard_normal((4, 4))
        Zw = rng2.standard_normal((2, 2)) + 1j * rng2.standard_normal((2, 2))
        Zw2 = rng2.standard_normal((2, 2)) + 1j * rng2.standard_normal((2, 2))
        U1, _ = la.qr(Z1)
        Uw, _ = la.qr(Zw)
        Uw2, _ = la.qr(Zw2)
        # Build C_3-equivariant rotation on V_Ram coordinates
        U_full = (bases_VRam['1'] @ U1 @ bases_VRam['1'].conj().T
                  + bases_VRam['w'] @ Uw @ bases_VRam['w'].conj().T
                  + bases_VRam['w2'] @ Uw2 @ bases_VRam['w2'].conj().T)
        # Verify unitary and C_3-equivariant
        assert la.norm(U_full @ U_full.conj().T - np.eye(8)) < 1e-7
        assert la.norm(U_full @ U_C3_VRam - U_C3_VRam @ U_full) < 1e-7
        # Rotate B → U·B·U† and recompute γ_7^B
        B_p = U_full @ B_VRam @ U_full.conj().T
        G7B_p = build_gamma7_B(B_p)
        Im_B_p = (B_p - B_p.conj().T) / (2j)
        target_p = (Im_B_p + Im_B_p.conj().T) / 2.0 / ABS_H
        # Re-extract C_3 sectors after rotation
        U_C3_p = U_full @ U_C3_VRam @ U_full.conj().T
        bases_p = c3_isotypic_basis(U_C3_p, 8)
        # Compute the trace difference on ω-ω²
        tr_w_p = np.trace(bases_p['w'].conj().T @ G7B_p @ bases_p['w']
                          @ bases_p['w'].conj().T @ target_p @ bases_p['w'])
        tr_w2_p = np.trace(bases_p['w2'].conj().T @ G7B_p @ bases_p['w2']
                           @ bases_p['w2'].conj().T @ target_p @ bases_p['w2'])
        tr_diff_p = tr_w_p - tr_w2_p
        print(f"  Trial {trial+1}: ω trace = {fmt_z(tr_w_p, 4)}  "
              f"ω² trace = {fmt_z(tr_w2_p, 4)}  Δ = {fmt_z(tr_diff_p, 4)}  "
              f"(/sin(arg h) = {tr_diff_p.real / SIN_ARG_H:+.4f})")

    # -----------------------------------------------------------------------
    # Step 6: Photon 1-loop with γ_7^B chirality projector.
    # -----------------------------------------------------------------------
    print("\nStep 6 — Photon 1-loop with γ_7^B inserted as chirality projector")
    print("-" * PRINT_WIDTH)

    # Build photon basis L = ω-irrep, R = ω²-irrep
    d = incidence_matrix_primitive(k_red, edges, n_verts)
    d1 = build_d1(cycles, edge_lookup, k_red, n_edges)
    Delta_1 = build_delta_1(d, d1)
    eigs_full, vecs_full = la.eig(Delta_1)
    order = np.argsort(eigs_full.real)
    eigs_full = eigs_full[order]
    vecs_full = vecs_full[:, order]
    mask = np.abs(eigs_full.real - 36.0) < 1e-6
    Q_phot_undir, _ = la.qr(vecs_full[:, mask])
    Q_phot_undir = Q_phot_undir[:, :int(mask.sum())]

    pi = build_pi_projector(bonds, edges, k_red)
    C3_e = build_C3_edge(edges, k_red)
    C3_photon = Q_phot_undir.conj().T @ C3_e @ Q_phot_undir
    eigvals_C3, eigvecs_C3 = la.eig(C3_photon)
    L_idx = int(np.argmin(np.abs(eigvals_C3 - omega)))
    R_idx = int(np.argmin(np.abs(eigvals_C3 - omega2)))
    L_vec = pi @ Q_phot_undir @ (eigvecs_C3[:, L_idx] / la.norm(eigvecs_C3[:, L_idx]))
    R_vec = pi @ Q_phot_undir @ (eigvecs_C3[:, R_idx] / la.norm(eigvecs_C3[:, R_idx]))
    L_vec = L_vec / la.norm(L_vec)
    R_vec = R_vec / la.norm(R_vec)

    # ∂_k B
    eps = 1e-6
    dB_dk = []
    for axis in range(3):
        k_plus = list(k_red).copy(); k_minus = list(k_red).copy()
        k_plus[axis] += eps; k_minus[axis] -= eps
        Bp = build_B_directed(bonds, np.array(k_plus))
        Bm = build_B_directed(bonds, np.array(k_minus))
        dB_dk.append((Bp - Bm) / (2 * eps))

    # γ_7^B lifted to 12-dim
    G7B_12 = V_Ram @ G7B @ V_Ram.conj().T

    # 1-loop diagrams to test:
    # Q1: (chirality-projected vertex) ∂_k B → γ_7^B → ∂_k B†
    # Q2: ∂_k B → γ_7^B · Im(B)/|h| → ∂_k B†
    # Q3: ∂_k B → γ_7^B · B/|B| → ∂_k B†
    diagrams = {
        "Q1: ∂B · γ_7^B · ∂B†": G7B_12,
        "Q2: ∂B · γ_7^B · Im(B)/|h| · ∂B†":
            G7B_12 @ V_Ram @ ((B_VRam - B_VRam.conj().T) / (2j) / ABS_H) @ V_Ram.conj().T,
        "Q3: ∂B · γ_7^B · B/|B| · ∂B†":
            G7B_12 @ V_Ram @ (B_VRam / ABS_H) @ V_Ram.conj().T,
        "Q4: ∂B · Im(B)/|h| · γ_7^B · ∂B† (γ_7^B last)":
            V_Ram @ ((B_VRam - B_VRam.conj().T) / (2j) / ABS_H) @ V_Ram.conj().T @ G7B_12,
    }

    print(f"  {'Diagram':<45}  {'M_LL':>12}  {'M_RR':>12}  "
          f"{'(M_LL−M_RR)/2':>16}  {'/sin(arg h)':>14}")
    print(f"  {'-'*45}  {'-'*12}  {'-'*12}  {'-'*16}  {'-'*14}")

    F_total = sum(la.norm(V_Ram.conj().T @ dB_dk[a] @ V_ker)**2 for a in range(3))

    for name, P12 in diagrams.items():
        M_LL = M_RR = M_LR = M_RL = 0
        for axis in range(3):
            V_ax = dB_dk[axis]
            op = V_ax @ P12 @ V_ax.conj().T
            M_LL += L_vec.conj() @ op @ L_vec
            M_LR += L_vec.conj() @ op @ R_vec
            M_RL += R_vec.conj() @ op @ L_vec
            M_RR += R_vec.conj() @ op @ R_vec
        diff = (M_LL - M_RR) / 2.0
        ratio = diff / SIN_ARG_H
        off = max(abs(M_LR), abs(M_RL))
        print(f"  {name:<45}  {fmt_z(M_LL, 3):>12}  {fmt_z(M_RR, 3):>12}  "
              f"{fmt_z(diff, 3):>16}  Re={ratio.real:+.4f} "
              f"Im={ratio.imag:+.4f} |off|={off:.1e}")

    # -----------------------------------------------------------------------
    # Step 7: Verdict.
    # -----------------------------------------------------------------------
    print("\n" + "=" * PRINT_WIDTH)
    print(f"VERDICT")
    print(f"=" * PRINT_WIDTH)
    print(f"\n  sin(arg h) = √(5/8) = {SIN_ARG_H:.6f}")
    print(f"  Σ Frobenius normalization F_total = {F_total:.4f}")
    print(f"\n  γ_7^B properties verified gauge-canonical (Step 2).")
    print(f"  Trace tr_ω[γ_7^B · Im(B)/|h|] result and gauge invariance to be inspected.")
    print(f"  Photon 1-loop with γ_7^B insertion: read (M_LL−M_RR)/2 column above.")
    print(f"\n  If c = 1 lands gauge-invariantly: F0_γ closure candidate.")
    print(f"  If still gauge-dependent or c ≠ 1: further sub-step recipe required.")

    print("\n" + "=" * PRINT_WIDTH)
    print(f"OK: arg_h_path_b_f0_gamma_attempt completed without errors")


if __name__ == "__main__":
    main()
