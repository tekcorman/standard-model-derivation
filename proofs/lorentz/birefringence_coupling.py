#!/usr/bin/env python3
"""
Birefringence coupling — screw-axis Peierls perturbation test.

Tests how the L vs R photon-polarization eigenstates of B(P) split at first
order under a U(1) gauge field aligned with the I4_132 4_1 screw axis (z =
[001]).  Reports the coefficient of g in the first-order phase splitting
Δarg/g and compares it against candidate algebraic forms.

Sister script: proofs/lorentz/birefringence_c3_irrep.py runs the same test
with a C₃-projector perturbation V_proj = g·(P_ω − P_ω²) and additional
L/R identifications.  V_proj is the diagnostic; the script here focuses on
the screw-axis Peierls (V_screw) and its ((r × k_P)·ẑ) variant (V_CFJ).

L/R identification (verified numerically: see birefringence_c3_irrep.py):
    L = ω-irrep eigenstate at walker eigenvalue +h
    R = ω-irrep eigenstate at walker eigenvalue +h*

The h-eigenspace decomposes under the C₃ stabilizer of P as
trivial ⊕ ω; the h*-eigenspace also decomposes as trivial ⊕ ω.  L and R
are at different walker eigenvalues but the same C₃ irrep — they're the
T-symmetry-conjugate pair.  This is the L1 identification in the sister
script's notation.

NOTE.  Earlier versions of this file mis-described the L/R structure as
"ω and ω* irreps" both within the h-eigenspace.  That description is
structurally incorrect: ω* (= ω²) lives at -h and -h*, not +h or +h*.
The corrected description is in the docstring above and in the
companion handoff doc.

Earlier bug history (since fixed): the chirality definition
(r × k_P) · k_P/|k_P| was identically zero by construction; k_P was at
(1/2, 1/2, 1/2) which is the H-point not the P-point; and the missing
2π factor in k_P_cart sent the eigenvalues to the wrong location.  All
three are corrected.  L/R selection by argmin distance to (h, h*) was
also replaced — argmin selects an arbitrary state from each multiplicity-2
eigenspace; we now project onto the ω-irrep specifically.
"""

import sys
import os
import math
import cmath
import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))
from proofs.common import find_bonds, A_PRIM, ATOMS, N_ATOMS
from proofs.lorentz.hashimoto_bloch_dispersion import (
    build_hashimoto_bloch, build_B_matrix
)


# =============================================================================
# Setup: B(P), C₃ stabilizer of P, eigenspace bases.
# =============================================================================

def setup():
    bonds = find_bonds()
    B_mat = build_B_matrix()
    B_inv = la.inv(B_mat)
    k_P_frac = np.array([0.25, 0.25, 0.25])
    k_P_cart = 2 * math.pi * (B_mat.T @ k_P_frac)
    BP = build_hashimoto_bloch(k_P_cart, bonds, B_inv)
    return bonds, BP, k_P_cart


def build_C3(bonds):
    R3 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    atom_perm = {0: 0, 1: 3, 2: 1, 3: 2}
    bond_perm = {}
    for i, (src, tgt, cell) in enumerate(bonds):
        r_old = (ATOMS[tgt] + cell[0] * A_PRIM[0] + cell[1] * A_PRIM[1]
                 + cell[2] * A_PRIM[2] - ATOMS[src])
        r_new = R3 @ r_old
        new_src, new_tgt = atom_perm[src], atom_perm[tgt]
        rhs = r_new - ATOMS[new_tgt] + ATOMS[new_src]
        new_cell_cont = la.solve(np.array(A_PRIM).T, rhs)
        new_cell = tuple(int(round(x)) for x in new_cell_cont)
        for j, (s, t, c) in enumerate(bonds):
            if s == new_src and t == new_tgt and tuple(c) == new_cell:
                bond_perm[i] = j
                break
    C3 = np.zeros((12, 12), dtype=complex)
    for i in range(12):
        C3[bond_perm[i], i] = 1.0
    return C3


def find_eigenspace(BP, eigval, tol=1e-8):
    evals, evecs = la.eig(BP)
    idx = np.where(np.abs(evals - eigval) < tol)[0]
    Q, _ = la.qr(evecs[:, idx])
    return Q


def project_omega_state(V, C3, target):
    """Eigenstate of C3 in V with eigenvalue target. Returns unit vector or None."""
    C3_in_V = V.conj().T @ C3 @ V
    e, v = la.eig(C3_in_V)
    i = int(np.argmin(np.abs(e - target)))
    if abs(e[i] - target) > 1e-8:
        return None
    psi = V @ v[:, i]
    return psi / la.norm(psi)


# =============================================================================
# Peierls perturbation: rebuild B(P) with bond phase += g · chir(r).
# =============================================================================

def build_B_perturbed(bonds, k_P_cart, g, chir_func):
    n = len(bonds)
    r_vecs = []
    for src, tgt, cell in bonds:
        disp = (ATOMS[tgt] + cell[0] * A_PRIM[0] + cell[1] * A_PRIM[1]
                + cell[2] * A_PRIM[2] - ATOMS[src])
        r_vecs.append(disp)
    r_vecs = np.array(r_vecs)
    chir = np.array([chir_func(r) for r in r_vecs])
    chir = chir - np.mean(chir)         # remove gauge-trivial uniform shift
    B_pert = np.zeros((n, n), dtype=complex)
    for i, (src_i, tgt_i, cell_i) in enumerate(bonds):
        phase_i = np.exp(1j * (np.dot(k_P_cart, r_vecs[i]) + g * chir[i]))
        for j, (src_j, tgt_j, cell_j) in enumerate(bonds):
            if tgt_j == src_i:
                is_reverse = (src_i == tgt_j and tgt_i == src_j
                              and tuple(cell_i) == tuple(-np.array(cell_j)))
                if not is_reverse:
                    B_pert[i, j] = phase_i
    return B_pert


# =============================================================================
# Driver.
# =============================================================================

def main():
    print("=" * 70)
    print("Birefringence coupling — screw-axis Peierls perturbation test")
    print("=" * 70)

    bonds, BP, k_P_cart = setup()
    h = complex(math.sqrt(3) / 2, math.sqrt(5) / 2)
    h_star = h.conjugate()
    omega = np.exp(2j * math.pi / 3)

    print(f"\nP-point in fractional reciprocal coords: (1/4, 1/4, 1/4)")
    print(f"k_P_cart = {k_P_cart}")

    # Walker eigenspaces and L/R states (L1 identification)
    V_h_plus = find_eigenspace(BP, h)
    V_hs_plus = find_eigenspace(BP, h_star)
    psi_L = project_omega_state(V_h_plus,  C3 := build_C3(bonds), omega)
    psi_R = project_omega_state(V_hs_plus, C3,                    omega)

    print(f"\nL = ω-irrep state at +h:  ⟨L|B(P)|L⟩ = "
          f"{np.vdot(psi_L, BP @ psi_L):+.6f}  (target h = {h:+.6f})")
    print(f"R = ω-irrep state at +h*: ⟨R|B(P)|R⟩ = "
          f"{np.vdot(psi_R, BP @ psi_R):+.6f}  (target h* = {h_star:+.6f})")
    print(f"⟨L|C₃|L⟩ = {np.vdot(psi_L, C3 @ psi_L):+.6f}  (target ω = {omega:+.6f})")
    print(f"⟨R|C₃|R⟩ = {np.vdot(psi_R, C3 @ psi_R):+.6f}  (target ω = {omega:+.6f})")
    print(f"⟨L|R⟩    = {np.vdot(psi_L, psi_R):+.2e}  "
          f"(non-orthogonal: B(P) is non-normal so eigenstates at"
          f" distinct eigenvalues need not be orthogonal under the Hermitian"
          f" inner product)")

    d_arg_baseline = np.angle(h) - np.angle(h_star)
    print(f"\nUnperturbed Δarg₀ = arg(h) − arg(h*) = "
          f"{math.degrees(d_arg_baseline):+.4f}°")

    # Two Peierls perturbation models
    models = [
        ("V_screw  = g · (r · ẑ)         [axial gauge field along [001]]",
         lambda r: r[2]),
        ("V_CFJ    = g · ((r × k_P) · ẑ) [screw-axis curl term]",
         lambda r: np.cross(r, k_P_cart)[2]),
    ]
    g_vals = np.array([1e-6, 1e-5, 1e-4, 1e-3])

    candidates = [
        ("1                   ", 1.0),
        ("2                   ", 2.0),
        ("sin(arg h) = √(5/8) ", math.sqrt(5 / 8)),
        ("Im(h) = √5/2        ", math.sqrt(5) / 2),
        ("Im(h)/|h|² = √5/4   ", math.sqrt(5) / 4),
        ("Re(h) = √3/2        ", math.sqrt(3) / 2),
        ("|h| = √2            ", math.sqrt(2)),
        ("√5/18 (=screw-norm) ", math.sqrt(5) / 18),
    ]

    print("\nFirst-order linear coefficient |Δarg/g| under each model:\n")
    for label, chir_func in models:
        print(f"--- {label} ---")
        coefs = []
        for g in g_vals:
            B_pert = build_B_perturbed(bonds, k_P_cart, g, chir_func)
            V = B_pert - BP
            dhL = np.vdot(psi_L, V @ psi_L)
            dhR = np.vdot(psi_R, V @ psi_R)
            argL = np.angle(h + dhL)
            argR = np.angle(h_star + dhR)
            d_arg = argL - argR
            coefs.append(((d_arg - d_arg_baseline) / g, dhL, dhR))
        c_d_arg = coefs[0][0]            # smallest g gives best linearization
        c_d_arg_abs = abs(c_d_arg)

        # Diagnostic: stability of coefficient across g
        all_coefs = [c[0] for c in coefs]
        stable = (max(all_coefs) - min(all_coefs)) / max(abs(c) for c in all_coefs) \
            if max(abs(c) for c in all_coefs) > 0 else 0
        print(f"  Coefficient stability across g: {stable:.2e} (relative spread)")
        print(f"  Δarg/g = {c_d_arg:+.10f}  rad/(unit g)")
        print(f"  |Δarg/g| = {c_d_arg_abs:.10f}")
        if c_d_arg_abs < 1e-8:
            print(f"  → ZERO (perturbation gauge-trivial in this L/R basis)")
        else:
            best = None
            for name, val in candidates:
                if val == 0:
                    continue
                ratio = c_d_arg_abs / val
                err = abs(ratio - round(ratio))
                if 0 < ratio <= 4 and err < 0.01:
                    best = (name, val, ratio)
                    break
            if best:
                name, val, ratio = best
                print(f"  → CLEAN MATCH: {round(ratio):.0f} × {name.strip()} "
                      f"= {round(ratio) * val:.6f}")
            else:
                print(f"  → No clean integer/half-integer multiple of any "
                      f"candidate; closest:")
                sorted_c = sorted(
                    candidates, key=lambda nv: abs(c_d_arg_abs/nv[1] - round(c_d_arg_abs/nv[1])) if nv[1] else 1e9)
                for name, val in sorted_c[:3]:
                    if val > 0:
                        ratio = c_d_arg_abs / val
                        print(f"      {name}: ratio = {ratio:.6f}")

    print("\n" + "=" * 70)
    print("Summary: the screw-axis Peierls test is gauge-equivalent to no")
    print("perturbation under the L1 identification, OR it gives a coefficient")
    print("of √5/18 = (1/9)·(√5/2) which is a lattice-geometry factor, not a")
    print("fundamental algebraic form.  The cleaner test is V_proj in")
    print("birefringence_c3_irrep.py, which gives Im(h) directly.")
    print("=" * 70)


if __name__ == "__main__":
    main()
