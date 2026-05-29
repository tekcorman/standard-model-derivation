#!/usr/bin/env python3
"""
arg_h_path_b_l3_gauge_check.py — gauge-invariance check on the candidate
chirality trace tr_ω[γ_7 · Im(B)/|h|] = 2·sin(arg h) discovered in
arg_h_path_b_l3_trace_survey.py.

The earlier survey found:
  tr_V_Ram^ω[γ_7 · Im(B|V_Ram)/|h|] = 2 · sin(arg h)
exactly on the ω-sector (2-dim) of V_Ram. Per-state, c = 1.

But the C_3-intertwining isomorphism A: ℂ^8 → V_Ram has gauge freedom on
each C_3-isotypic sector: U(4) on trivial, U(2) on ω, U(2) on ω². The
P4 attempt only checked U(4) gauge invariance on the trivial sector (and
got gauge-invariant c = 0 for the photon channel, because photon ⊥ V_Ram).

For the trace survey result to be a genuine structural prediction, we
need it to be invariant under the U(2) gauge on the ω sector specifically.

This script tests: does tr_V_Ram^ω[γ_7 · F(B)] depend on the U(2) gauge
on the ω-sector of S (i.e., on the choice of A within the ω-sector)?

Run with:
    PYTHONPATH=. python3 proofs/foundations/arg_h_path_b_l3_gauge_check.py
"""

from __future__ import annotations

import itertools
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from numpy import linalg as la
from scipy.linalg import expm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "proofs" / "cosmology"))

from srs_photon_bloch_primitive import (
    build_primitive_unit_cell,
    find_primitive_connectivity,
    canonical_edges_primitive,
)
from srs_photon_c3_chainmap import K_P_RED, ATOM_PERM
from srs_photon_chirality_coefficient import (
    build_C3_directed,
    build_B_directed,
)


TOL = 1e-10
H_EXACT = (math.sqrt(3) + 1j * math.sqrt(5)) / 2
SIN_ARG_H = math.sqrt(5.0 / 8.0)
omega = np.exp(2j * math.pi / 3)
omega2 = omega.conjugate()
PRINT_WIDTH = 78

I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)


def kron(*mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


Gamma = [None] * 7
Gamma[1] = kron(sx, I2, I2)
Gamma[2] = kron(sy, I2, I2)
Gamma[3] = kron(sz, sx, I2)
Gamma[4] = kron(sz, sy, I2)
Gamma[5] = kron(sz, sz, sx)
Gamma[6] = kron(sz, sz, sy)
I8 = np.eye(8, dtype=complex)
G7 = -1j * Gamma[1] @ Gamma[2] @ Gamma[3] @ Gamma[4] @ Gamma[5] @ Gamma[6]


def biv(a, b):
    return 0.5 * (Gamma[a] @ Gamma[b] - Gamma[b] @ Gamma[a])


def build_U_C3_S():
    K4_VERTICES = [0, 1, 2, 3]
    K4_EDGES = [(i, j) for i in K4_VERTICES for j in K4_VERTICES if i < j]
    SIGMA = ATOM_PERM
    edge_to_idx = {e: i for i, e in enumerate(K4_EDGES)}
    P_so6 = np.zeros((6, 6), dtype=float)
    for e in K4_EDGES:
        i = edge_to_idx[e]
        j = edge_to_idx[tuple(sorted((SIGMA[e[0]], SIGMA[e[1]])))]
        P_so6[j, i] = 1.0
    evals_so6, evecs_so6 = la.eig(P_so6)
    log_evals = np.array([np.log(ev) for ev in evals_so6])
    L_so6 = (evecs_so6 @ np.diag(log_evals) @ np.linalg.inv(evecs_so6))
    L_so6_real = L_so6.real
    X_spin = np.zeros((8, 8), dtype=complex)
    for i in range(6):
        for j in range(i + 1, 6):
            X_spin += L_so6_real[i, j] * biv(i + 1, j + 1)
    U_C3_S = expm(0.5 * X_spin)
    if np.allclose(U_C3_S @ U_C3_S @ U_C3_S, -I8, atol=1e-8):
        U_C3_S = np.exp(1j * math.pi / 3) * U_C3_S
    return U_C3_S


def classify_c3(ev, tol=0.1):
    if abs(ev - 1.0) < tol:
        return '1'
    if abs(ev - omega) < tol:
        return 'w'
    if abs(ev - omega2) < tol:
        return 'w2'
    return '?'


def c3_isotypic_basis(M, dim, tol=0.1):
    evals, evecs = la.eig(M)
    groups = {'1': [], 'w': [], 'w2': []}
    for i, ev in enumerate(evals):
        label = classify_c3(ev, tol)
        if label in groups:
            groups[label].append(evecs[:, i])
    bases = {}
    for label, vecs in groups.items():
        if vecs:
            mat = np.column_stack(vecs)
            Q, _ = la.qr(mat)
            bases[label] = Q[:, :len(vecs)]
        else:
            bases[label] = np.zeros((dim, 0), dtype=complex)
    return bases


def main():
    print("=" * PRINT_WIDTH)
    print("Gauge-invariance check on tr_ω[γ_7 · F(B)] candidate trace")
    print("=" * PRINT_WIDTH)

    verts, lat = build_primitive_unit_cell()
    bonds = find_primitive_connectivity(verts, lat)
    edges = canonical_edges_primitive(bonds)
    k_red = K_P_RED

    B = build_B_directed(bonds, k_red)
    Bevs, Bevecs = la.eig(B)
    ram_idx = [i for i, ev in enumerate(Bevs) if abs(abs(ev)**2 - 2.0) < 1e-5]
    V_Ram, _ = la.qr(Bevecs[:, ram_idx])
    V_Ram = V_Ram[:, :8]

    C3_dir = build_C3_directed(bonds)
    U_C3_VRam = V_Ram.conj().T @ C3_dir @ V_Ram
    bases_VRam = c3_isotypic_basis(U_C3_VRam, 8)

    U_C3_S = build_U_C3_S()
    bases_S = c3_isotypic_basis(U_C3_S, 8)

    B_VRam = V_Ram.conj().T @ B @ V_Ram
    Im_B = (B_VRam - B_VRam.conj().T) / (2j)
    Im_B = (Im_B + Im_B.conj().T) / 2.0

    target_op = Im_B / math.sqrt(2.0)   # Im(B)/|h| (unit-phasor normalized)

    def make_A(W_triv, W_w, W_w2):
        """Build A = sum_α Q_VR_α · W_α · Q_S_α†, with W_α gauge unitaries."""
        A = np.zeros((8, 8), dtype=complex)
        A += bases_VRam['1'] @ W_triv @ bases_S['1'].conj().T
        A += bases_VRam['w'] @ W_w @ bases_S['w'].conj().T
        A += bases_VRam['w2'] @ W_w2 @ bases_S['w2'].conj().T
        return A

    def random_unitary(dim, rng):
        Z = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        Q, _ = la.qr(Z)
        return Q

    def trace_omega(A_choice):
        G7_VRam = A_choice @ G7 @ A_choice.conj().T
        Q_w = bases_VRam['w']
        G7_w = Q_w.conj().T @ G7_VRam @ Q_w
        F_w = Q_w.conj().T @ target_op @ Q_w
        return np.trace(G7_w @ F_w)

    def trace_omega2(A_choice):
        G7_VRam = A_choice @ G7 @ A_choice.conj().T
        Q_w2 = bases_VRam['w2']
        G7_w2 = Q_w2.conj().T @ G7_VRam @ Q_w2
        F_w2 = Q_w2.conj().T @ target_op @ Q_w2
        return np.trace(G7_w2 @ F_w2)

    def trace_omega_plus_omega2(A_choice):
        G7_VRam = A_choice @ G7 @ A_choice.conj().T
        Q_nontriv = np.column_stack([bases_VRam['w'], bases_VRam['w2']])
        G7_nt = Q_nontriv.conj().T @ G7_VRam @ Q_nontriv
        F_nt = Q_nontriv.conj().T @ target_op @ Q_nontriv
        return np.trace(G7_nt @ F_nt)

    # --- Original A (identity gauge) ---
    A0 = make_A(np.eye(4), np.eye(2), np.eye(2))
    tr_w0 = trace_omega(A0)
    tr_w20 = trace_omega2(A0)
    tr_sum0 = trace_omega_plus_omega2(A0)
    print(f"\nOriginal gauge (W_α = I):")
    print(f"  tr_ω[γ_7·Im(B)/|h|]   = {tr_w0:.6f}    "
          f"(/sin(arg h) = {tr_w0.real / SIN_ARG_H:+.6f})")
    print(f"  tr_ω²[γ_7·Im(B)/|h|]  = {tr_w20:.6f}    "
          f"(/sin(arg h) = {tr_w20.real / SIN_ARG_H:+.6f})")
    print(f"  tr_(ω+ω²)[γ_7·Im(B)/|h|] = {tr_sum0:.6f}  "
          f"(/sin(arg h) = {tr_sum0.real / SIN_ARG_H:+.6f})")

    # --- Test 1: Random U(2) on ω sector only ---
    print(f"\n--- Test 1: Random U(2) on ω sector (W_w varies, W_triv = I, W_w2 = I) ---")
    rng = np.random.default_rng(42)
    tr_w_list, tr_w2_list, tr_sum_list = [], [], []
    for trial in range(5):
        W_w = random_unitary(2, rng)
        A_p = make_A(np.eye(4), W_w, np.eye(2))
        assert la.norm(A_p @ A_p.conj().T - np.eye(8)) < 1e-7
        tr_w_list.append(trace_omega(A_p))
        tr_w2_list.append(trace_omega2(A_p))
        tr_sum_list.append(trace_omega_plus_omega2(A_p))
        print(f"  Trial {trial+1}: tr_ω = {tr_w_list[-1]:+.5f}  "
              f"tr_ω² = {tr_w2_list[-1]:+.5f}  tr_sum = {tr_sum_list[-1]:+.5f}")
    var_w = max(abs(t - tr_w0) for t in tr_w_list)
    var_w2 = max(abs(t - tr_w20) for t in tr_w2_list)
    var_sum = max(abs(t - tr_sum0) for t in tr_sum_list)
    print(f"  max |Δtr_ω|  = {var_w:.3e}")
    print(f"  max |Δtr_ω²| = {var_w2:.3e}")
    print(f"  max |Δtr_sum|= {var_sum:.3e}")

    # --- Test 2: Random U(2) on ω² sector only ---
    print(f"\n--- Test 2: Random U(2) on ω² sector (W_w2 varies, others = I) ---")
    tr_w_list2, tr_w2_list2, tr_sum_list2 = [], [], []
    for trial in range(5):
        W_w2 = random_unitary(2, rng)
        A_p = make_A(np.eye(4), np.eye(2), W_w2)
        tr_w_list2.append(trace_omega(A_p))
        tr_w2_list2.append(trace_omega2(A_p))
        tr_sum_list2.append(trace_omega_plus_omega2(A_p))
        print(f"  Trial {trial+1}: tr_ω = {tr_w_list2[-1]:+.5f}  "
              f"tr_ω² = {tr_w2_list2[-1]:+.5f}  tr_sum = {tr_sum_list2[-1]:+.5f}")
    var_w_2 = max(abs(t - tr_w0) for t in tr_w_list2)
    var_w2_2 = max(abs(t - tr_w20) for t in tr_w2_list2)
    var_sum_2 = max(abs(t - tr_sum0) for t in tr_sum_list2)
    print(f"  max |Δtr_ω|  = {var_w_2:.3e}")
    print(f"  max |Δtr_ω²| = {var_w2_2:.3e}")
    print(f"  max |Δtr_sum|= {var_sum_2:.3e}")

    # --- Test 3: Random U(2) ⊕ U(2) on both ω, ω² (full non-trivial gauge) ---
    print(f"\n--- Test 3: Random U(2) ⊕ U(2) on ω + ω² (both gauges varied) ---")
    tr_w_list3, tr_w2_list3, tr_sum_list3 = [], [], []
    for trial in range(5):
        W_w = random_unitary(2, rng)
        W_w2 = random_unitary(2, rng)
        A_p = make_A(np.eye(4), W_w, W_w2)
        tr_w_list3.append(trace_omega(A_p))
        tr_w2_list3.append(trace_omega2(A_p))
        tr_sum_list3.append(trace_omega_plus_omega2(A_p))
        print(f"  Trial {trial+1}: tr_ω = {tr_w_list3[-1]:+.5f}  "
              f"tr_ω² = {tr_w2_list3[-1]:+.5f}  tr_sum = {tr_sum_list3[-1]:+.5f}")
    var_w_3 = max(abs(t - tr_w0) for t in tr_w_list3)
    var_w2_3 = max(abs(t - tr_w20) for t in tr_w2_list3)
    var_sum_3 = max(abs(t - tr_sum0) for t in tr_sum_list3)
    print(f"  max |Δtr_ω|  = {var_w_3:.3e}")
    print(f"  max |Δtr_ω²| = {var_w2_3:.3e}")
    print(f"  max |Δtr_sum|= {var_sum_3:.3e}")

    # --- Test 4: Full gauge group (U(4) ⊕ U(2) ⊕ U(2)) ---
    print(f"\n--- Test 4: Full gauge group U(4) ⊕ U(2) ⊕ U(2) ---")
    tr_w_list4, tr_w2_list4, tr_sum_list4 = [], [], []
    for trial in range(5):
        W_triv = random_unitary(4, rng)
        W_w = random_unitary(2, rng)
        W_w2 = random_unitary(2, rng)
        A_p = make_A(W_triv, W_w, W_w2)
        tr_w_list4.append(trace_omega(A_p))
        tr_w2_list4.append(trace_omega2(A_p))
        tr_sum_list4.append(trace_omega_plus_omega2(A_p))
        print(f"  Trial {trial+1}: tr_ω = {tr_w_list4[-1]:+.5f}  "
              f"tr_ω² = {tr_w2_list4[-1]:+.5f}  tr_sum = {tr_sum_list4[-1]:+.5f}")
    var_w_4 = max(abs(t - tr_w0) for t in tr_w_list4)
    var_w2_4 = max(abs(t - tr_w20) for t in tr_w2_list4)
    var_sum_4 = max(abs(t - tr_sum0) for t in tr_sum_list4)
    print(f"  max |Δtr_ω|  = {var_w_4:.3e}")
    print(f"  max |Δtr_ω²| = {var_w2_4:.3e}")
    print(f"  max |Δtr_sum|= {var_sum_4:.3e}")

    # --- Verdict ---
    print("\n" + "=" * PRINT_WIDTH)
    print("VERDICT")
    print("=" * PRINT_WIDTH)
    test1_inv = (var_w < 1e-8)
    test2_inv = (var_w2_2 < 1e-8)
    test3_inv = (var_w_3 < 1e-8 and var_w2_3 < 1e-8 and var_sum_3 < 1e-8)
    test4_inv = (var_sum_4 < 1e-8)

    print(f"\n  Test 1 (U(2) on ω only) — tr_ω invariant: {test1_inv}")
    print(f"  Test 2 (U(2) on ω² only) — tr_ω² invariant: {test2_inv}")
    print(f"  Test 3 (U(2)⊕U(2) on ω+ω²) — all sector traces invariant: {test3_inv}")
    print(f"  Test 4 (full gauge) — tr_(ω+ω²) invariant: {test4_inv}")

    if test3_inv:
        print("\n  ✓ tr_(ω+ω²)[γ_7·Im(B)/|h|] = "
              f"{tr_sum0.real:.6f} = {tr_sum0.real / SIN_ARG_H:.4f}·sin(arg h)")
        print("    is GAUGE-INVARIANT under the U(2)⊕U(2) freedom on non-trivial sectors.")
    else:
        print("\n  ✗ Trace varies with gauge — not gauge-invariant.")
        print("    The C_3-canonical sector trace is not a structural prediction.")

    print("\n  individual sector traces:")
    print(f"    tr_ω[γ_7·Im(B)/|h|]  = {tr_w0.real:+.6f} = "
          f"{tr_w0.real / SIN_ARG_H:.4f}·sin(arg h)  "
          f"{'[GAUGE-INV]' if test1_inv else '[GAUGE-DEP]'}")
    print(f"    tr_ω²[γ_7·Im(B)/|h|] = {tr_w20.real:+.6f} = "
          f"{tr_w20.real / SIN_ARG_H:.4f}·sin(arg h)  "
          f"{'[GAUGE-INV]' if test2_inv else '[GAUGE-DEP]'}")

    print("\n" + "=" * PRINT_WIDTH)


if __name__ == "__main__":
    main()
