#!/usr/bin/env python3
"""
arg_h_path_b_l3_trace_survey.py — exploratory survey for arg(h) Path B''
Routes L3-anomaly / L3-loop.

Background
----------
Both tree-level routes (L3-tree direct π†·Im(B)·π and P4 Cl(6,0) γ_7
via B6) are FALSIFIED with c = 0 because:
  - Im(B) is parity-ODD on bond reversal but photon Hodge bundle is parity-EVEN.
  - Photon Hodge bundle is orthogonal to V_Ram at k_P (overlap ~10⁻¹⁵);
    γ_7 transferred via B6 lives entirely on V_Ram, cannot reach V_kernel.

This survey explores the C_3-canonical, gauge-invariant chirality traces on
V_Ram and the off-diagonal block structure of V_Ram ↔ V_kernel — the two
ingredients needed for any L3-anomaly (chiral fermion loop → F·F̃) or
L3-loop (photon self-energy bridging V_Ram and V_kernel) closure of
β = c · sin(arg h) · α_EM.

What this script computes
-------------------------
1. C_3-isotypic decomposition of V_Ram (trivial 4, ω 2, ω² 2).
2. Spectral content of B|V_Ram in each C_3 sector (which of {h, h̄, -h, -h̄}
   land in which sector).
3. γ_7 (= Γ_7^Ram) restricted to each C_3 sector — eigenvalue split.
4. γ_7-weighted spectral traces on the C_3-canonical (ω, ω²) sectors:
   tr_V_Ram^ω[γ_7 · F(B|V_Ram)] for F ∈ {B, B², B−B†, Im(B), |B|², resolvent}.
5. Off-diagonal block of B(P) between V_Ram and V_kernel (should be 0
   exactly since V_Ram and V_kernel are eigenspace decomposition of B(P)).
6. Off-diagonal block of (B(P) − B(P)†)/(2i) = Im(B(P)) between V_Ram
   and V_kernel — this might be non-zero (Im(B) is not block-diagonal
   in B's eigenbasis).
7. Off-diagonal block of γ_7 (lifted via B6) between V_Ram and V_kernel —
   note γ_7 lift is 0 outside V_Ram, so this is automatically 0; but
   alternative chirality structures might exist on V_kernel directly.

Run with:
    PYTHONPATH=. python3 proofs/foundations/arg_h_path_b_l3_trace_survey.py
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
    incidence_matrix_primitive,
)
from srs_photon_hodge import build_d1, build_edge_lookup
from srs_cycle_enumerator import enumerate_simple_cycles
from srs_photon_c3_chainmap import (
    build_C3_edge,
    build_delta_1,
    K_P_RED,
    ATOM_PERM,
)
from srs_photon_chirality_coefficient import (
    build_pi_projector,
    build_C3_directed,
    build_B_directed,
)


TOL = 1e-10
H_EXACT = (math.sqrt(3) + 1j * math.sqrt(5)) / 2
SIN_ARG_H = math.sqrt(5.0 / 8.0)
ARG_H_DEG = math.degrees(math.atan2(math.sqrt(5), math.sqrt(3)))
omega = np.exp(2j * math.pi / 3)
omega2 = omega.conjugate()
PRINT_WIDTH = 78

# Cl(6,0) Brauer-Weyl
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
assert np.allclose(G7 @ G7, I8, atol=TOL)


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


def fmt_z(z, prec=6):
    return f"{z.real:+.{prec}f}{z.imag:+.{prec}f}j"


def main():
    print("=" * PRINT_WIDTH)
    print("arg(h) Path B'' L3 trace survey")
    print("Goal: identify gauge-invariant chirality traces giving sin(arg h)·c")
    print("=" * PRINT_WIDTH)

    # ---- Setup ----
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
    print(f"  V_Ram·V_Ram† + V_ker·V_ker† − I_12: max = "
          f"{np.max(np.abs(V_Ram @ V_Ram.conj().T + V_ker @ V_ker.conj().T - np.eye(12))):.2e}")

    # ---- C_3 ----
    C3_dir = build_C3_directed(bonds)
    U_C3_VRam = V_Ram.conj().T @ C3_dir @ V_Ram
    U_C3_Vker = V_ker.conj().T @ C3_dir @ V_ker
    print(f"\nC_3 isotypic on V_Ram: "
          f"{Counter(classify_c3(ev) for ev in la.eigvals(U_C3_VRam))}")
    print(f"C_3 isotypic on V_kernel: "
          f"{Counter(classify_c3(ev) for ev in la.eigvals(U_C3_Vker))}")

    bases_VRam = c3_isotypic_basis(U_C3_VRam, 8)

    # ---- γ_7 transfer ----
    U_C3_S = build_U_C3_S()
    bases_S = c3_isotypic_basis(U_C3_S, 8)
    A = np.zeros((8, 8), dtype=complex)
    for label in ['1', 'w', 'w2']:
        A += bases_VRam[label] @ bases_S[label].conj().T
    G7_VRam = A @ G7 @ A.conj().T
    G7_VRam = (G7_VRam + G7_VRam.conj().T) / 2.0   # Hermitize

    B_VRam = V_Ram.conj().T @ B @ V_Ram
    print(f"\nB|V_Ram eigenvalues: "
          f"{[fmt_z(ev, 3) for ev in sorted(la.eigvals(B_VRam), key=lambda z: (z.real, z.imag))]}")

    # =======================================================================
    # Section A: γ_7 split inside each C_3 sector of V_Ram
    # =======================================================================
    print("\n" + "=" * PRINT_WIDTH)
    print("Section A — γ_7 split within each C_3 sector of V_Ram")
    print("=" * PRINT_WIDTH)
    for label in ['1', 'w', 'w2']:
        Q = bases_VRam[label]
        d = Q.shape[1]
        if d == 0:
            continue
        G7_sector = Q.conj().T @ G7_VRam @ Q
        G7_sector = (G7_sector + G7_sector.conj().T) / 2.0
        evs = sorted(la.eigvalsh(G7_sector))
        B_sector = Q.conj().T @ B_VRam @ Q
        B_evs = sorted(la.eigvals(B_sector), key=lambda z: (z.real, z.imag))
        print(f"\n  C_3 sector '{label}' (dim={d}):")
        print(f"    γ_7|sector eigenvalues: {[f'{ev:+.4f}' for ev in evs]}")
        print(f"    B|sector eigenvalues:   {[fmt_z(ev, 4) for ev in B_evs]}")

    # =======================================================================
    # Section B: γ_7-weighted spectral traces on each C_3 sector
    # =======================================================================
    print("\n" + "=" * PRINT_WIDTH)
    print("Section B — γ_7-weighted traces tr[γ_7 · F(B|V_Ram^α)] per sector")
    print("(ω and ω² sectors are gauge-canonical; trivial sector is U(4)-ambiguous)")
    print("=" * PRINT_WIDTH)

    Im_B = (B_VRam - B_VRam.conj().T) / (2j)
    Im_B = (Im_B + Im_B.conj().T) / 2.0

    operators = {
        "I_8": np.eye(8, dtype=complex),
        "B": B_VRam,
        "B†": B_VRam.conj().T,
        "Im(B)": Im_B,
        "Re(B) = (B+B†)/2": (B_VRam + B_VRam.conj().T) / 2.0,
        "B²": B_VRam @ B_VRam,
        "BB†": B_VRam @ B_VRam.conj().T,
        "(B−B†)·B": (B_VRam - B_VRam.conj().T) @ B_VRam,
        "B·B − B†·B†": B_VRam @ B_VRam - B_VRam.conj().T @ B_VRam.conj().T,
        "(B−B†)/(2i√2) [unit phasor]": Im_B / math.sqrt(2.0),
    }

    for label in ['1', 'w', 'w2']:
        Q = bases_VRam[label]
        if Q.shape[1] == 0:
            continue
        print(f"\n--- Sector C_3='{label}' (dim={Q.shape[1]}) ---")
        G7_sec = Q.conj().T @ G7_VRam @ Q
        for name, F in operators.items():
            F_sec = Q.conj().T @ F @ Q
            tr = np.trace(G7_sec @ F_sec)
            ratio = tr / SIN_ARG_H if abs(SIN_ARG_H) > 0 else float('nan')
            print(f"    tr[γ_7·{name:30s}]  = {fmt_z(tr, 5)}  "
                  f"(/sin(arg h) = {fmt_z(ratio, 4)})")

    # Summed over ω+ω² (gauge-invariant)
    print("\n--- Sum over ω+ω² sectors (gauge-canonical) ---")
    Q_w = bases_VRam['w']
    Q_w2 = bases_VRam['w2']
    Q_nontriv = np.column_stack([Q_w, Q_w2])
    G7_nt = Q_nontriv.conj().T @ G7_VRam @ Q_nontriv
    for name, F in operators.items():
        F_nt = Q_nontriv.conj().T @ F @ Q_nontriv
        tr = np.trace(G7_nt @ F_nt)
        ratio = tr / SIN_ARG_H if abs(SIN_ARG_H) > 0 else float('nan')
        print(f"  tr_(ω+ω²)[γ_7·{name:28s}] = {fmt_z(tr, 5)}  "
              f"(/sin(arg h) = {fmt_z(ratio, 4)})")

    # Difference ω − ω²
    print("\n--- Difference ω − ω² sectors (chirality-discriminating) ---")
    G7_w = Q_w.conj().T @ G7_VRam @ Q_w
    G7_w2 = Q_w2.conj().T @ G7_VRam @ Q_w2
    for name, F in operators.items():
        F_w = Q_w.conj().T @ F @ Q_w
        F_w2 = Q_w2.conj().T @ F @ Q_w2
        tr_diff = np.trace(G7_w @ F_w) - np.trace(G7_w2 @ F_w2)
        ratio = tr_diff / SIN_ARG_H if abs(SIN_ARG_H) > 0 else float('nan')
        print(f"  Δtr[γ_7·{name:28s}]   = {fmt_z(tr_diff, 5)}  "
              f"(/sin(arg h) = {fmt_z(ratio, 4)})")

    # =======================================================================
    # Section C: V_Ram ↔ V_kernel off-diagonal blocks
    # =======================================================================
    print("\n" + "=" * PRINT_WIDTH)
    print("Section C — V_Ram ↔ V_kernel off-diagonal blocks at k_P")
    print("=" * PRINT_WIDTH)

    bridge_ops_12 = {
        "B(P)": B,
        "Im(B(P)) [(B−B†)/(2i)]": (B - B.conj().T) / (2j),
        "Re(B(P)) [(B+B†)/2]": (B + B.conj().T) / 2.0,
        "B(P)†·B(P)": B.conj().T @ B,
        "B(P)·B(P)†": B @ B.conj().T,
        "C3_dir": C3_dir,
    }

    for name, M in bridge_ops_12.items():
        block = V_Ram.conj().T @ M @ V_ker
        ker_to_ram = V_ker.conj().T @ M @ V_Ram
        f_norm = la.norm(block)
        f_norm_kr = la.norm(ker_to_ram)
        print(f"\n  {name}:")
        print(f"    ||V_Ram† · M · V_kernel||_F = {f_norm:.3e}")
        print(f"    ||V_kernel† · M · V_Ram||_F = {f_norm_kr:.3e}")
        if f_norm > 1e-6:
            print(f"    NON-TRIVIAL bridge — singular values: "
                  f"{[f'{s:.4f}' for s in la.svd(block, compute_uv=False)]}")

    # =======================================================================
    # Section D: γ_7-weighted V_Ram ↔ V_kernel chirality bridge
    # =======================================================================
    print("\n" + "=" * PRINT_WIDTH)
    print("Section D — Try chirality-bridging operators for L3-loop ansatz")
    print("=" * PRINT_WIDTH)

    # Lift γ_7^Ram to 12-dim, restricted to V_Ram (zero on V_kernel by construction)
    G7_12 = V_Ram @ G7_VRam @ V_Ram.conj().T

    # Composite ops that mix V_Ram & V_kernel
    composites = {
        "G7_12 · Im(B)": G7_12 @ ((B - B.conj().T) / (2j)),
        "Im(B) · G7_12": ((B - B.conj().T) / (2j)) @ G7_12,
        "G7_12 · B − B · G7_12": G7_12 @ B - B @ G7_12,
        "{G7_12, Im(B)}": G7_12 @ ((B - B.conj().T) / (2j))
                          + ((B - B.conj().T) / (2j)) @ G7_12,
        "[G7_12, Im(B)]": G7_12 @ ((B - B.conj().T) / (2j))
                          - ((B - B.conj().T) / (2j)) @ G7_12,
    }
    for name, M in composites.items():
        block = V_Ram.conj().T @ M @ V_ker
        f_norm = la.norm(block)
        print(f"  ||V_Ram† · ({name}) · V_kernel||_F = {f_norm:.3e}")

    # =======================================================================
    # Section E: Photon basis projection onto V_kernel sectors
    # =======================================================================
    print("\n" + "=" * PRINT_WIDTH)
    print("Section E — Photon Hodge bundle decomposition within V_kernel")
    print("=" * PRINT_WIDTH)

    d = incidence_matrix_primitive(k_red, edges, n_verts)
    d1 = build_d1(cycles, edge_lookup, k_red, n_edges)
    Delta_1 = build_delta_1(d, d1)
    eigs_full, vecs_full = la.eig(Delta_1)
    order = np.argsort(eigs_full.real)
    eigs_full = eigs_full[order]
    vecs_full = vecs_full[:, order]
    mask = np.abs(eigs_full.real - 36.0) < 1e-6
    photon_basis = vecs_full[:, mask]
    Q_phot, _ = la.qr(photon_basis)
    Q_phot = Q_phot[:, :photon_basis.shape[1]]

    pi = build_pi_projector(bonds, edges, k_red)
    photon_12 = pi @ Q_phot   # 12 × 2 in directed-bond space
    photon_VRam_overlap = la.norm(V_Ram.conj().T @ photon_12, axis=0)
    photon_Vker_overlap = la.norm(V_ker.conj().T @ photon_12, axis=0)
    print(f"\n  Photon overlap with V_Ram:   {photon_VRam_overlap}")
    print(f"  Photon overlap with V_kernel: {photon_Vker_overlap}")

    # Decompose photon within V_kernel sector by C_3
    photon_in_Vker = V_ker.conj().T @ photon_12   # 4 × 2 in V_kernel coords
    bases_Vker = c3_isotypic_basis(U_C3_Vker, 4)
    print(f"\n  V_kernel C_3 isotypic dims: "
          f"{[(lab, bases_Vker[lab].shape[1]) for lab in ['1','w','w2']]}")
    for label in ['1', 'w', 'w2']:
        if bases_Vker[label].shape[1] == 0:
            continue
        proj = bases_Vker[label] @ bases_Vker[label].conj().T
        photon_sector = proj @ photon_in_Vker
        norms = la.norm(photon_sector, axis=0)
        print(f"  Photon Hodge bundle in V_kernel C_3 sector '{label}': "
              f"norms = {norms}")

    # =======================================================================
    # Section F: Try a B(k)-derivative bridge (k near k_P)
    # =======================================================================
    print("\n" + "=" * PRINT_WIDTH)
    print("Section F — B(k)-derivative bridge (numerical ∂_k B at k_P)")
    print("=" * PRINT_WIDTH)
    eps = 1e-5
    dB_dk = []
    for axis in range(3):
        k_plus = list(k_red).copy()
        k_minus = list(k_red).copy()
        k_plus[axis] += eps
        k_minus[axis] -= eps
        Bp = build_B_directed(bonds, np.array(k_plus))
        Bm = build_B_directed(bonds, np.array(k_minus))
        dB_dk.append((Bp - Bm) / (2 * eps))

    # Project ∂_k B onto V_Ram ↔ V_kernel block
    for axis in range(3):
        block = V_Ram.conj().T @ dB_dk[axis] @ V_ker
        f_norm = la.norm(block)
        print(f"  ||V_Ram† · (∂B/∂k_{axis}) · V_kernel||_F = {f_norm:.3e}")
        if f_norm > 1e-3:
            svds = la.svd(block, compute_uv=False)
            print(f"    sing.vals = {[f'{s:.4f}' for s in svds]}")

    # γ_7-weighted ∂_k B trace
    print(f"\n  γ_7-weighted ∂B/∂k bridge integrals:")
    for axis in range(3):
        block_kp = V_Ram.conj().T @ dB_dk[axis] @ V_ker
        # 1-loop diagram: tr[γ_7 · block · block†]
        loop1 = np.trace(G7_VRam @ block_kp @ block_kp.conj().T)
        # 2-loop diagram: tr[γ_7 · block · block†]²
        ratio = loop1 / SIN_ARG_H if abs(SIN_ARG_H) > 0 else float('nan')
        print(f"    axis {axis}: tr[γ_7·∂B·∂B†] on V_Ram = "
              f"{fmt_z(loop1, 4)}  (/sin(arg h) = {fmt_z(ratio, 4)})")

    print("\n" + "=" * PRINT_WIDTH)
    print("END OF SURVEY — inspect outputs for sin(arg h) · structural-constant patterns")
    print("=" * PRINT_WIDTH)


if __name__ == "__main__":
    main()
