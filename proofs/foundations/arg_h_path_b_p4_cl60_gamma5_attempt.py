#!/usr/bin/env python3
"""
arg_h_path_b_p4_cl60_gamma5_attempt.py — Route P4 of arg(h) Path B'' scoping.

Tests whether the Cl(6,0) γ_7 chirality projection (transferred to V_Ram
via the B6 isomorphism) supplies the c = 1 coefficient for
β = c · sin(arg h) · α_EM where the L3-tree direct V_chir = π†·Im(B)·π
attempt failed (`proofs/cosmology/srs_photon_chirality_coefficient.py`,
c = 0 because Im(B) is parity-ODD under bond reversal but the photon
Hodge bundle is parity-EVEN).

Hypothesis (Route P4)
---------------------
Substituting the Cl(6,0) chirality operator γ_7 = -i Γ_1 Γ_2 ... Γ_6
(transferred to V_Ram via the B6 C_3-intertwining isomorphism A) for
Im(B(P)) routes the chirality through spinor algebra rather than through
the parity-ODD anti-Hermitian part of B(P). The hope: γ_7's lift through
V_Ram avoids the parity-EVEN filtering that killed Route L3-tree.

V_chir^{P4}(k_P) := π† · Γ_7^{12} · π     (6×6 on undirected edges)

where Γ_7^{12} = V_Ram_basis · A · γ_7 · A† · V_Ram_basis† embeds the
Cl(6,0) chirality into the 12-dim directed-bond space via the
8-dim Ramanujan subspace V_Ram of B(P).

Restrict V_chir^{P4} to the 2-dim photon ω² = 36 Hodge eigenspace at
k_P; diagonalize C_3 on this subspace; read the L (= ω-irrep) and R
(= ω²-irrep) diagonal entries. Hypothesis: V_chir^{P4} = diag(+c·sin(arg h),
−c·sin(arg h)) with c = 1.

Outcomes
--------
- c = 1: Route P4 closes Lemma 3 of `theorem_dark_correction_mdl.md`;
  β graduates A− → THEOREM-GRADE; 4 P-rows graduate (P34, P35, P36, P44).
- c = 0: P4 falsified at tree level (analog of L3-tree failure); pivot
  to Route L3-anomaly / L3-loop or fall back to ADOPTED-ARG-H-PROJECTION.
- c ≠ 0, 1: identify the structural quantity c is and reframe.

Gauge ambiguity caveat
----------------------
Per `proofs/foundations/gamma7_chirality.py` Step 11: A has residual
U(4) gauge freedom in the C_3-trivial sector of S (4-dim trivial), which
mixes γ_7 = ±1 sub-sectors. The ω and ω² C_3-sectors of S are 2-dim
each with γ_7 split 1+1, so gauge there is U(2) ⊃ U(1)×U(1) preserving
γ_7's split. The photon's L = ω-irrep, R = ω²-irrep restricts to
gauge-canonical sectors. The trivial sector contributes to
V_chir^{P4} via the projection π's overlap with the trivial-isotypic
part of V_Ram — that contribution is gauge-dependent.

The script reports both:
  (a) full V_chir^{P4} (gauge-dependent),
  (b) gauge-invariant subset by restricting to ω/ω² C_3 sectors only.

Run with:
    PYTHONPATH=. python3 proofs/foundations/arg_h_path_b_p4_cl60_gamma5_attempt.py
"""

from __future__ import annotations

import itertools
import math
import os
import sys
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
    HIGH_SYM_POINTS,
)
from srs_photon_hodge import build_d1, build_edge_lookup
from srs_cycle_enumerator import enumerate_simple_cycles
from srs_photon_c3_chainmap import (
    build_C3_vertex,
    build_C3_edge,
    build_delta_1,
    K_P_RED,
    ATOM_PERM,
    c3_cell,
)
from srs_photon_chirality_coefficient import (
    build_pi_projector,
    build_C3_directed,
    build_B_directed,
)


# ===========================================================================
# Constants
# ===========================================================================

TOL = 1e-10
H_EXACT = (math.sqrt(3) + 1j * math.sqrt(5)) / 2
ABS_H = math.sqrt(2.0)
ARG_H = math.atan2(math.sqrt(5), math.sqrt(3))
ARG_H_DEG = math.degrees(ARG_H)
SIN_ARG_H = math.sqrt(5.0 / 8.0)   # = Im(h)/|h| = √(5/8) ≈ 0.7906
PRINT_WIDTH = 78
omega = np.exp(2j * math.pi / 3)
omega2 = omega.conjugate()


# ===========================================================================
# Cl(6,0) Brauer-Weyl gamma matrices and γ_7 (B3 convention)
# Self-contained copy of proofs/foundations/gamma7_chirality.py setup.
# ===========================================================================

I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)


def kron(*mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


# Brauer-Weyl Γ_1...Γ_6 on S = ℂ^8 (B3 convention)
Gamma = [None] * 7
Gamma[1] = kron(sx, I2, I2)
Gamma[2] = kron(sy, I2, I2)
Gamma[3] = kron(sz, sx, I2)
Gamma[4] = kron(sz, sy, I2)
Gamma[5] = kron(sz, sz, sx)
Gamma[6] = kron(sz, sz, sy)
I8 = np.eye(8, dtype=complex)

# Verify Clifford relations
for _a, _b in itertools.product(range(1, 7), repeat=2):
    _lhs = Gamma[_a] @ Gamma[_b] + Gamma[_b] @ Gamma[_a]
    _rhs = 2.0 * (1.0 if _a == _b else 0.0) * I8
    assert np.allclose(_lhs, _rhs, atol=TOL), f"Clifford fails {_a},{_b}"

# γ_7 = -i Γ_1...Γ_6
G7 = -1j * Gamma[1] @ Gamma[2] @ Gamma[3] @ Gamma[4] @ Gamma[5] @ Gamma[6]
assert np.allclose(G7 @ G7, I8, atol=TOL)
assert np.allclose(G7, G7.conj().T, atol=TOL)


def biv(a, b):
    return 0.5 * (Gamma[a] @ Gamma[b] - Gamma[b] @ Gamma[a])


def build_U_C3_S():
    """Spin(6) lift of body-diagonal C_3 onto the B3 8-dim spinor (per B6)."""
    K4_VERTICES = [0, 1, 2, 3]
    K4_EDGES = [(i, j) for i in K4_VERTICES for j in K4_VERTICES if i < j]
    SIGMA = ATOM_PERM   # {0:0, 1:3, 2:1, 3:2}

    def apply_sigma_to_edge(edge):
        a, b = edge
        return tuple(sorted((SIGMA[a], SIGMA[b])))

    edge_to_idx = {e: i for i, e in enumerate(K4_EDGES)}
    P_so6 = np.zeros((6, 6), dtype=float)
    for e in K4_EDGES:
        i = edge_to_idx[e]
        j = edge_to_idx[apply_sigma_to_edge(e)]
        P_so6[j, i] = 1.0

    assert np.allclose(P_so6.T @ P_so6, np.eye(6), atol=TOL)
    assert np.allclose(np.linalg.matrix_power(P_so6, 3), np.eye(6), atol=TOL)

    evals_so6, evecs_so6 = la.eig(P_so6)
    log_evals = np.array([np.log(ev) for ev in evals_so6])
    L_so6 = (evecs_so6 @ np.diag(log_evals) @ np.linalg.inv(evecs_so6))
    L_so6_real = L_so6.real
    assert np.allclose(expm(L_so6_real), P_so6, atol=1e-10)

    X_spin = np.zeros((8, 8), dtype=complex)
    for i in range(6):
        for j in range(i + 1, 6):
            X_spin += L_so6_real[i, j] * biv(i + 1, j + 1)
    U_C3_S = expm(0.5 * X_spin)

    U3 = U_C3_S @ U_C3_S @ U_C3_S
    if np.allclose(U3, -I8, atol=1e-8):
        U_C3_S = np.exp(1j * np.pi / 3) * U_C3_S
        assert np.allclose(U_C3_S @ U_C3_S @ U_C3_S, I8, atol=1e-8)
    else:
        assert np.allclose(U3, I8, atol=1e-8)

    assert la.norm(U_C3_S @ G7 - G7 @ U_C3_S) < 1e-8
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


# ===========================================================================
# Driver
# ===========================================================================

def main():
    print("=" * PRINT_WIDTH)
    print("Route P4 — Cl(6,0) γ_7 chirality projection on photon Hodge bundle")
    print("β = c · sin(arg h) · α_EM, hypothesis c = 1 via γ_7 transfer through V_Ram")
    print("=" * PRINT_WIDTH)

    # -----------------------------------------------------------------------
    # Step 0: Geometry — bonds, edges, cycles (srs_photon_bloch_primitive).
    # -----------------------------------------------------------------------
    print("\nStep 0 — Build srs primitive cell geometry")
    print("-" * PRINT_WIDTH)
    verts, lat = build_primitive_unit_cell()
    bonds = find_primitive_connectivity(verts, lat)
    edges = canonical_edges_primitive(bonds)
    edge_lookup = build_edge_lookup(edges)
    cycles = enumerate_simple_cycles(bonds, max_length=10)
    n_verts, n_edges, n_bonds = len(verts), len(edges), len(bonds)
    print(f"  verts={n_verts}, undirected edges={n_edges}, "
          f"directed bonds={n_bonds}, length-10 cycles={len(cycles)}")

    k_red = K_P_RED

    # -----------------------------------------------------------------------
    # Step 1: Walker B(P), V_Ram (8-dim Ramanujan subspace).
    # -----------------------------------------------------------------------
    print("\nStep 1 — Build B(P) on directed bonds and extract V_Ram")
    print("-" * PRINT_WIDTH)
    B = build_B_directed(bonds, k_red)
    print(f"  B(P) shape: {B.shape}")
    Bevs, Bevecs = la.eig(B)
    print(f"  B(P) eigenvalues:")
    for ev in sorted(Bevs, key=lambda z: (-abs(z), -z.imag)):
        print(f"    {ev.real:+.4f}{ev.imag:+.4f}j   |·|={abs(ev):.4f}   "
              f"arg={math.degrees(np.angle(ev)):+.2f}°")
    ram_idx = [i for i, ev in enumerate(Bevs) if abs(abs(ev)**2 - 2.0) < 1e-5]
    assert len(ram_idx) == 8, f"Expected 8 Ramanujan eigenvectors, got {len(ram_idx)}"
    V_Ram, _ = la.qr(Bevecs[:, ram_idx])
    V_Ram = V_Ram[:, :8]
    assert la.matrix_rank(V_Ram) == 8
    print(f"  V_Ram: 12×8 orthonormal basis (||V†V−I_8|| = "
          f"{la.norm(V_Ram.conj().T @ V_Ram - np.eye(8)):.2e})")

    # B(P) restricted to V_Ram, in V_Ram coords (8×8)
    B_VRam = V_Ram.conj().T @ B @ V_Ram
    print(f"  B|V_Ram eigenvalues: "
          f"{sorted([f'{ev.real:+.3f}{ev.imag:+.3f}j' for ev in la.eigvals(B_VRam)])}")

    # -----------------------------------------------------------------------
    # Step 2: C_3 on directed bonds, restrict to V_Ram, verify (4,2,2).
    # -----------------------------------------------------------------------
    print("\nStep 2 — C_3 on directed bonds; isotypic decomp on V_Ram")
    print("-" * PRINT_WIDTH)
    C3_dir = build_C3_directed(bonds)
    assert np.allclose(C3_dir @ C3_dir @ C3_dir, np.eye(n_bonds), atol=TOL)
    comm = la.norm(B @ C3_dir - C3_dir @ B)
    print(f"  ||[B(P), C3_dir]|| = {comm:.2e}")
    assert comm < 1e-10
    U_C3_VRam = V_Ram.conj().T @ C3_dir @ V_Ram
    from collections import Counter
    cc = Counter(classify_c3(ev) for ev in la.eigvals(U_C3_VRam))
    print(f"  C_3-isotypic on V_Ram: trivial={cc.get('1',0)}, "
          f"ω={cc.get('w',0)}, ω²={cc.get('w2',0)}")
    assert (cc.get('1', 0), cc.get('w', 0), cc.get('w2', 0)) == (4, 2, 2)

    # -----------------------------------------------------------------------
    # Step 3: U_C3_S Spin(6) lift on Cl(6,0) S = ℂ^8; γ_7 splits as 4+4.
    # -----------------------------------------------------------------------
    print("\nStep 3 — U_C3_S Spin(6) lift on S = ℂ^8 (per B6); γ_7 splits S as 4+4")
    print("-" * PRINT_WIDTH)
    U_C3_S = build_U_C3_S()
    cc_S = Counter(classify_c3(ev) for ev in la.eigvals(U_C3_S))
    print(f"  U_C3_S isotypic: trivial={cc_S.get('1',0)}, "
          f"ω={cc_S.get('w',0)}, ω²={cc_S.get('w2',0)}")
    assert (cc_S.get('1', 0), cc_S.get('w', 0), cc_S.get('w2', 0)) == (4, 2, 2)
    cc_G7 = Counter(int(round(ev)) for ev in la.eigvalsh(G7))
    print(f"  γ_7 eigenvalues on S: +1 mult={cc_G7[+1]}, -1 mult={cc_G7[-1]}")
    assert cc_G7[+1] == 4 and cc_G7[-1] == 4

    # -----------------------------------------------------------------------
    # Step 4: C_3-intertwining isomorphism A: ℂ^8 → V_Ram coords.
    # -----------------------------------------------------------------------
    print("\nStep 4 — Build A: ℂ^8 → V_Ram coords (C_3-intertwining)")
    print("-" * PRINT_WIDTH)
    bases_S = c3_isotypic_basis(U_C3_S, 8)
    bases_VRam = c3_isotypic_basis(U_C3_VRam, 8)
    A = np.zeros((8, 8), dtype=complex)
    for label in ['1', 'w', 'w2']:
        A += bases_VRam[label] @ bases_S[label].conj().T
    err_u = la.norm(A @ A.conj().T - np.eye(8))
    err_i = la.norm(A @ U_C3_S - U_C3_VRam @ A)
    print(f"  ||AA†−I_8|| = {err_u:.2e},  ||A·U_C3_S − U_C3_VRam·A|| = {err_i:.2e}")
    assert err_u < 1e-8 and err_i < 1e-8

    # -----------------------------------------------------------------------
    # Step 5: Transfer γ_7 to V_Ram and lift to 12-dim directed bonds.
    # -----------------------------------------------------------------------
    print("\nStep 5 — Transfer γ_7 to V_Ram and lift to 12-dim directed bonds")
    print("-" * PRINT_WIDTH)
    G7_VRam = A @ G7 @ A.conj().T
    G7_12 = V_Ram @ G7_VRam @ V_Ram.conj().T
    err_h = la.norm(G7_VRam - G7_VRam.conj().T)
    err_sq = la.norm(G7_VRam @ G7_VRam - np.eye(8))
    print(f"  ||Γ_7^Ram − (Γ_7^Ram)†|| = {err_h:.2e}   (Hermiticity)")
    print(f"  ||(Γ_7^Ram)² − I_8|| = {err_sq:.2e}   (γ² = 1)")
    assert err_h < 1e-8 and err_sq < 1e-8
    cc_G7_VR = Counter(int(round(ev)) for ev in la.eigvalsh(G7_VRam))
    print(f"  Γ_7^Ram eigenvalues on V_Ram: +1 mult={cc_G7_VR[+1]}, "
          f"-1 mult={cc_G7_VR[-1]}")
    assert cc_G7_VR[+1] == 4 and cc_G7_VR[-1] == 4
    # Γ_7^12 acts as 0 outside V_Ram. Check Hermiticity.
    err_h12 = la.norm(G7_12 - G7_12.conj().T)
    print(f"  ||Γ_7^12 − (Γ_7^12)†|| = {err_h12:.2e}")
    assert err_h12 < 1e-8

    # -----------------------------------------------------------------------
    # Step 6: Build π: undirected → directed bonds; verify equivariance.
    # -----------------------------------------------------------------------
    print("\nStep 6 — Build π: 6-dim undirected edges → 12-dim directed bonds")
    print("-" * PRINT_WIDTH)
    pi = build_pi_projector(bonds, edges, k_red)
    assert np.allclose(pi.conj().T @ pi, np.eye(n_edges), atol=TOL)
    C3_e = build_C3_edge(edges, k_red)
    eqv = la.norm(pi @ C3_e - C3_dir @ pi)
    print(f"  ||π·C3_e − C3_dir·π|| = {eqv:.2e}")
    assert eqv < 1e-8

    # -----------------------------------------------------------------------
    # Step 7: V_chir^{P4} = π† · Γ_7^12 · π on undirected edges (6×6).
    # -----------------------------------------------------------------------
    print("\nStep 7 — Compute V_chir^{P4} = π† · Γ_7^12 · π (6×6 on undirected)")
    print("-" * PRINT_WIDTH)
    V_chir_undir = pi.conj().T @ G7_12 @ pi
    V_chir_undir = (V_chir_undir + V_chir_undir.conj().T) / 2.0
    print(f"  V_chir^{{P4}} eigenvalues: "
          f"{sorted(la.eigvalsh(V_chir_undir).real)}")
    print(f"  ||V_chir^{{P4}}||_F = {la.norm(V_chir_undir):.6f}")
    print(f"  trace(V_chir^{{P4}}) = {np.trace(V_chir_undir).real:+.6f}")

    # -----------------------------------------------------------------------
    # Step 8: Restrict to photon ω² = 36 Hodge eigenspace.
    # -----------------------------------------------------------------------
    print("\nStep 8 — Restrict to photon ω² = 36 Hodge eigenspace (2-dim)")
    print("-" * PRINT_WIDTH)
    d = incidence_matrix_primitive(k_red, edges, n_verts)
    d1 = build_d1(cycles, edge_lookup, k_red, n_edges)
    Delta_1 = build_delta_1(d, d1)
    eigs_full, vecs_full = la.eig(Delta_1)
    order = np.argsort(eigs_full.real)
    eigs_full = eigs_full[order]
    vecs_full = vecs_full[:, order]
    target = 36.0
    mask = np.abs(eigs_full.real - target) < 1e-6
    photon_basis = vecs_full[:, mask]
    Q_phot, _ = la.qr(photon_basis)
    Q_phot = Q_phot[:, :photon_basis.shape[1]]
    print(f"  Photon ω²={target} eigenspace dim: {Q_phot.shape[1]}")
    assert Q_phot.shape[1] == 2

    V_chir_photon = Q_phot.conj().T @ V_chir_undir @ Q_phot
    V_chir_photon = (V_chir_photon + V_chir_photon.conj().T) / 2.0
    print(f"  V_chir^{{P4}} | photon (2×2 in arbitrary photon basis):")
    for row in V_chir_photon:
        print("   ", "  ".join(f"{x.real:+.6f}{x.imag:+.6f}j" for x in row))
    print(f"  trace = {np.trace(V_chir_photon).real:+.6f}")
    print(f"  eigenvalues = "
          f"{sorted(la.eigvalsh(V_chir_photon).real)}")

    # -----------------------------------------------------------------------
    # Step 9: Diagonalize C_3 on photon space; pull V_chir into L/R basis.
    # -----------------------------------------------------------------------
    print("\nStep 9 — L/R photon basis (= ω/ω² C_3 eigenstates) + read V_chir^{P4}")
    print("-" * PRINT_WIDTH)
    C3_photon = Q_phot.conj().T @ C3_e @ Q_phot
    eigvals_C3, eigvecs_C3 = la.eig(C3_photon)
    print(f"  C_3|photon eigenvalues: {eigvals_C3}")
    L_idx = int(np.argmin(np.abs(eigvals_C3 - omega)))
    R_idx = int(np.argmin(np.abs(eigvals_C3 - omega2)))
    L_vec = eigvecs_C3[:, L_idx] / la.norm(eigvecs_C3[:, L_idx])
    R_vec = eigvecs_C3[:, R_idx] / la.norm(eigvecs_C3[:, R_idx])
    LR = np.column_stack([L_vec, R_vec])

    V_chir_LR = LR.conj().T @ V_chir_photon @ LR
    V_chir_LR = (V_chir_LR + V_chir_LR.conj().T) / 2.0
    print(f"  V_chir^{{P4}} in L/R basis:")
    print(f"    [⟨L|V|L⟩  ⟨L|V|R⟩]")
    print(f"    [⟨R|V|L⟩  ⟨R|V|R⟩]")
    for row in V_chir_LR:
        print("   ", "  ".join(f"{x.real:+.6f}{x.imag:+.6f}j" for x in row))

    cL, cR = V_chir_LR[0, 0].real, V_chir_LR[1, 1].real
    off_diag = max(abs(V_chir_LR[0, 1]), abs(V_chir_LR[1, 0]))
    splitting_half = (cL - cR) / 2.0

    print(f"\n  c_L = ⟨L|V_chir^{{P4}}|L⟩ = {cL:+.6f}")
    print(f"  c_R = ⟨R|V_chir^{{P4}}|R⟩ = {cR:+.6f}")
    print(f"  sin(arg h) = √(5/8)        = {SIN_ARG_H:+.6f}")
    print(f"  c_L / sin(arg h)            = {cL / SIN_ARG_H:+.6f}")
    print(f"  c_R / sin(arg h)            = {cR / SIN_ARG_H:+.6f}")
    print(f"  splitting half (c_L−c_R)/2  = {splitting_half:+.6f}")
    print(f"  splitting / sin(arg h)      = "
          f"{splitting_half / SIN_ARG_H:+.6f}    (target c = ±1)")
    print(f"  max off-diagonal |V_LR|     = {off_diag:.2e}")

    coef_match_one = abs(abs(splitting_half / SIN_ARG_H) - 1.0) < 1e-6
    coef_match_zero = abs(splitting_half) < 1e-8
    schur_ok = off_diag < 1e-8

    if schur_ok:
        print("  PASS — Schur (V_chir^{P4} diagonal in L/R basis at machine prec)")
    else:
        print("  WARN — V_chir^{P4} has off-diagonal terms in L/R "
              "(Schur expected diagonal for ω≠ω² inequivalent irreps)")

    # -----------------------------------------------------------------------
    # Step 10: Gauge invariance check on the chosen quantities.
    # Test whether c_L, c_R depend on the U(4) gauge choice in trivial sector.
    # -----------------------------------------------------------------------
    print("\nStep 10 — Gauge-invariance check (random U(4) on trivial sector of A)")
    print("-" * PRINT_WIDTH)
    np.random.seed(2026)
    cL_trials, cR_trials, off_trials = [], [], []
    for trial in range(5):
        Z = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4))
        W4, _ = la.qr(Z)   # random 4×4 unitary on trivial sector
        Q_S1, Q_VR1 = bases_S['1'], bases_VRam['1']
        A_p = A.copy()
        A_p -= Q_VR1 @ Q_S1.conj().T
        A_p += Q_VR1 @ W4 @ Q_S1.conj().T
        # Check still unitary + intertwining
        assert la.norm(A_p @ A_p.conj().T - np.eye(8)) < 1e-7
        assert la.norm(A_p @ U_C3_S - U_C3_VRam @ A_p) < 1e-7
        G7_VRam_p = A_p @ G7 @ A_p.conj().T
        G7_12_p = V_Ram @ G7_VRam_p @ V_Ram.conj().T
        Vc_p = pi.conj().T @ G7_12_p @ pi
        Vc_p = (Vc_p + Vc_p.conj().T) / 2.0
        Vc_phot_p = Q_phot.conj().T @ Vc_p @ Q_phot
        Vc_phot_p = (Vc_phot_p + Vc_phot_p.conj().T) / 2.0
        Vc_LR_p = LR.conj().T @ Vc_phot_p @ LR
        Vc_LR_p = (Vc_LR_p + Vc_LR_p.conj().T) / 2.0
        cL_trials.append(Vc_LR_p[0, 0].real)
        cR_trials.append(Vc_LR_p[1, 1].real)
        off_trials.append(max(abs(Vc_LR_p[0, 1]), abs(Vc_LR_p[1, 0])))
    print(f"  Original           cL = {cL:+.6f},  cR = {cR:+.6f},  "
          f"|off| = {off_diag:.2e}")
    for i in range(5):
        print(f"  Random U(4) #{i+1}    cL = {cL_trials[i]:+.6f},  "
              f"cR = {cR_trials[i]:+.6f},  |off| = {off_trials[i]:.2e}")
    cL_var = max(abs(c - cL) for c in cL_trials)
    cR_var = max(abs(c - cR) for c in cR_trials)
    print(f"  max |Δc_L| over trials = {cL_var:.2e}")
    print(f"  max |Δc_R| over trials = {cR_var:.2e}")
    gauge_invariant = cL_var < 1e-8 and cR_var < 1e-8

    if gauge_invariant:
        print(f"  PASS — c_L, c_R are gauge-invariant under U(4) on trivial sector")
    else:
        print(f"  STRUCTURAL: c_L, c_R DEPEND on gauge choice in trivial sector A")
        print(f"  (= the photon Hodge bundle has overlap with the trivial-C_3")
        print(f"  sub-sector of V_Ram, where γ_7's lift is gauge-ambiguous)")

    # -----------------------------------------------------------------------
    # Step 11: Photon-bundle overlap with C_3 sectors of V_Ram.
    # Diagnostic: where in V_Ram does the photon Hodge bundle sit?
    # -----------------------------------------------------------------------
    print("\nStep 11 — Decompose photon Hodge bundle by C_3 sector of V_Ram")
    print("-" * PRINT_WIDTH)
    # Lift photon basis from undirected edges (6) to directed bonds (12).
    photon_12 = pi @ Q_phot     # 12 × 2
    # Project onto V_Ram (8-dim) inside 12-dim directed bonds.
    photon_VRam = V_Ram.conj().T @ photon_12     # 8 × 2 (in V_Ram coords)
    # Norm of photon inside V_Ram (vs total norm via π):
    norm_total = la.norm(photon_12, axis=0)
    norm_VRam = la.norm(photon_VRam, axis=0)
    print(f"  Photon basis in 12-dim (after π): col-norms = {norm_total}")
    print(f"  Photon basis in V_Ram (8-dim):    col-norms = {norm_VRam}")
    print(f"  Fraction of photon norm in V_Ram: "
          f"{norm_VRam[0]/norm_total[0]:.4f}, {norm_VRam[1]/norm_total[1]:.4f}")
    # Decompose photon (in V_Ram coords) by C_3 sector.
    for label, basis_label in zip(['trivial', 'ω', 'ω²'], ['1', 'w', 'w2']):
        proj = bases_VRam[basis_label] @ bases_VRam[basis_label].conj().T
        photon_in_sector = proj @ photon_VRam
        norm_sector = la.norm(photon_in_sector, axis=0)
        frac = norm_sector / np.maximum(norm_VRam, 1e-12)
        print(f"  Photon sector '{label}' overlap (in V_Ram): "
              f"L={frac[0]:.4f}, R={frac[1]:.4f}")

    # -----------------------------------------------------------------------
    # Step 12: Verdict.
    # -----------------------------------------------------------------------
    print("\n" + "=" * PRINT_WIDTH)
    print("VERDICT")
    print("=" * PRINT_WIDTH)
    print(f"\n  c_split = (c_L − c_R)/2 = {splitting_half:+.6f}")
    print(f"  sin(arg h)              = {SIN_ARG_H:+.6f}")
    print(f"  c = c_split / sin(arg h) = "
          f"{splitting_half / SIN_ARG_H:+.6f}")
    print(f"  schur off-diagonal       = {off_diag:.2e}")
    print(f"  gauge-invariant          = {gauge_invariant}")

    if coef_match_one and schur_ok and gauge_invariant:
        print("\n  ✓ ROUTE P4 LANDS — c = 1 in β = c · sin(arg h) · α_EM")
        print("  Lemma 3 of theorem_dark_correction_mdl.md closes via Cl(6,0) γ_7.")
        print("  β graduates A− → THEOREM-GRADE.")
        print("  P34/P35/P36/P44 graduate from BLOCKED → THEOREM-GRADE.")
        verdict = "P4_LANDS"
    elif coef_match_zero:
        print("\n  ✗ ROUTE P4 FALSIFIED — c = 0 (analog of L3-tree failure)")
        print("  γ_7 transferred via B6 also filters out chirality on photon bundle.")
        print("  Pivot to Route L3-anomaly / L3-loop, or fall to ADOPTED-ARG-H-PROJECTION.")
        verdict = "P4_FALSIFIED_C_ZERO"
    elif coef_match_one and not gauge_invariant:
        print("\n  ⚠ ROUTE P4 GAUGE-AMBIGUOUS at c ≈ 1 — needs canonical gauge fix")
        print("  Splitting half matches sin(arg h) in this gauge but depends on the")
        print("  U(4) freedom in the trivial-C_3 sector of A. Either (a) identify a")
        print("  canonical gauge from additional structure, or (b) accept a gauge-")
        print("  free measurement (e.g. tr(γ_7 · F) on ω⊕ω² sectors only).")
        verdict = "P4_GAUGE_AMBIGUOUS"
    else:
        print(f"\n  ⚠ ROUTE P4 — c is neither 0 nor 1; identify what it is")
        print(f"  c = {splitting_half / SIN_ARG_H:+.6f}    (some structural quantity)")
        print(f"  splitting_half = {splitting_half:+.6f}")
        verdict = "P4_OTHER"

    print(f"\n  verdict tag: {verdict}")
    print("\n" + "=" * PRINT_WIDTH)
    print("OK: arg_h_path_b_p4_cl60_gamma5_attempt completed without errors")


if __name__ == "__main__":
    main()
