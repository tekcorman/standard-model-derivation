#!/usr/bin/env python3
"""
β.E follow-up — second-order chiral perturbation with helicity-correct
L/R identification.

NOTE (2026-04-25 PM).  The interpretive framing in (a)/(b)/(c) below is
superseded by the MDL synthesis in `docs/theorems/theorem_dark_correction_mdl.md`.
The L3 V_proj zero result is EXPECTED, not falsification-adjacent: V_proj
isolates a more-expensive MDL term, while the framework's leading prediction
(Lemma 2 + polar decomposition) takes the cheaper MDL term sin(arg h).
Both are MDL-permitted (see Lemma 1 in the synthesis doc).  This script
remains a useful diagnostic for "which MDL term does naive perturbation
isolate?" but its outputs are not the framework's leading β prediction.

Motivation: standard QED identifies L (left circular polarization,
helicity +1) and R (right circular, helicity -1) via opposite C₃
eigenvalues ω, ω² for C₃ = exp(2πi·J_[111]/3) on a spin-1 rotation.
On the srs walker, the ω-irrep states live at walker eigenvalues +h,
+h*; the ω²-irrep states live at -h, -h*.

Therefore physically correct L/R identification:
    L = ω-state at +h    (helicity +1)
    R = ω²-state at -h   (helicity -1)
which is option L3 in the earlier β.E test.  Under L3, the V_proj
perturbation gave |Δarg/g| = 0 at first order (exact cancellation).

This script extends to second order.  Three questions:

(a) Does any perturbation model give a non-zero β at higher order
    under L3?
(b) If yes, is the coefficient sin(arg h)·α_EM, or some other
    functional form?
(c) If no, the framework's canonical β = sin(arg h)·α_EM may need
    re-examination — specifically the L/R identification on which
    the canonical formula relies.

Method: numerical diagonalization at multiple g values, polynomial fit
in g, extract c₁ (linear), c₂ (quadratic), c₃ (cubic) coefficients.
Track which perturbed eigenvalue connects continuously to which
unperturbed eigenstate.

Comparison:  for each (option, model), report the polynomial
coefficients and whether they match clean algebraic candidates.
"""

import sys
import os
import math
import cmath
import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))
from proofs.common import find_bonds, A_PRIM, ATOMS
from proofs.lorentz.hashimoto_bloch_dispersion import (
    build_hashimoto_bloch, build_B_matrix
)


# =============================================================================
# Setup utilities (shared with birefringence_c3_irrep.py).
# =============================================================================

def setup():
    bonds = find_bonds()
    B_mat = build_B_matrix()
    B_inv = la.inv(B_mat)
    k_P_cart = 2 * math.pi * (B_mat.T @ np.array([0.25, 0.25, 0.25]))
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


def project_irrep(V, C3, target):
    """ω-irrep state in eigenspace V."""
    C3_in_V = V.conj().T @ C3 @ V
    e, v = la.eig(C3_in_V)
    i = int(np.argmin(np.abs(e - target)))
    if abs(e[i] - target) > 1e-8:
        return None
    psi = V @ v[:, i]
    return psi / la.norm(psi)


def make_proj_perturbation(C3):
    """V_proj = (P_ω − P_ω²) — global projector chiral perturbation."""
    omega = np.exp(2j * math.pi / 3)
    omega2 = omega.conjugate()
    c_evals, c_evecs = la.eig(C3)
    P_omega = np.zeros((12, 12), dtype=complex)
    P_omega2 = np.zeros((12, 12), dtype=complex)
    for j, ev in enumerate(c_evals):
        v = c_evecs[:, j:j + 1]
        v = v / la.norm(v)
        if abs(ev - omega) < 1e-8:
            P_omega += v @ v.conj().T
        elif abs(ev - omega2) < 1e-8:
            P_omega2 += v @ v.conj().T
    V_unit = P_omega - P_omega2
    return lambda g: g * V_unit


def make_peierls_perturbation(bonds, k_P_cart, chir_func):
    """Peierls phase perturbation (rebuilt B at each g)."""
    BP_baseline = build_hashimoto_bloch(
        k_P_cart, bonds,
        la.inv(build_B_matrix()))
    n = len(bonds)
    r_vecs = np.array([
        ATOMS[tgt] + cell[0]*A_PRIM[0] + cell[1]*A_PRIM[1]
        + cell[2]*A_PRIM[2] - ATOMS[src]
        for src, tgt, cell in bonds])
    chir = np.array([chir_func(r) for r in r_vecs])
    chir = chir - np.mean(chir)

    def make_B_pert(g):
        B_pert = np.zeros((n, n), dtype=complex)
        for i, (src_i, tgt_i, cell_i) in enumerate(bonds):
            phase_i = np.exp(1j * (np.dot(k_P_cart, r_vecs[i])
                                   + g * chir[i]))
            for j, (src_j, tgt_j, cell_j) in enumerate(bonds):
                if tgt_j == src_i:
                    is_reverse = (src_i == tgt_j and tgt_i == src_j
                                  and tuple(cell_i) == tuple(-np.array(cell_j)))
                    if not is_reverse:
                        B_pert[i, j] = phase_i
        return B_pert - BP_baseline

    return make_B_pert


def follow_eigenvalue(BP, V_at_g, h_unperturbed, psi_unperturbed):
    """Find the perturbed eigenvalue that continuously connects to
    h_unperturbed.  Uses largest-projection criterion: among all
    perturbed eigenvalues, pick the one whose eigenvector has the
    largest overlap with psi_unperturbed.
    """
    BP_pert = BP + V_at_g
    e, v = la.eig(BP_pert)
    # normalize v
    for j in range(v.shape[1]):
        v[:, j] /= la.norm(v[:, j])
    overlaps = np.abs(v.conj().T @ psi_unperturbed)
    idx = int(np.argmax(overlaps))
    return e[idx], v[:, idx]


def fit_polynomial(gs, ys, degree=3):
    """Fit y = c0 + c1*g + c2*g^2 + ... by least squares."""
    A = np.vstack([gs**k for k in range(degree + 1)]).T
    coefs, *_ = la.lstsq(A, ys, rcond=None)
    return coefs


# =============================================================================
# Driver: run all three options at second order.
# =============================================================================

def main():
    print("=" * 70)
    print("β.E O(g²) — chiral perturbation at second order, three L/R options")
    print("=" * 70)

    bonds, BP, k_P_cart = setup()
    C3 = build_C3(bonds)
    h = complex(math.sqrt(3)/2, math.sqrt(5)/2)
    h_star = h.conjugate()
    omega = np.exp(2j * math.pi / 3)
    omega2 = omega.conjugate()

    # All four eigenspaces
    V_ph = find_eigenspace(BP, h)
    V_phs = find_eigenspace(BP, h_star)
    V_mh = find_eigenspace(BP, -h)
    V_mhs = find_eigenspace(BP, -h_star)

    # ω-states at +h, +h*; ω²-states at -h, -h*
    psi_w_at_ph = project_irrep(V_ph, C3, omega)
    psi_w_at_phs = project_irrep(V_phs, C3, omega)
    psi_w2_at_mh = project_irrep(V_mh, C3, omega2)
    psi_w2_at_mhs = project_irrep(V_mhs, C3, omega2)

    # Three options
    options = [
        ("L1: L = ω at +h,  R = ω at +h*",
         psi_w_at_ph, psi_w_at_phs, h, h_star),
        ("L2: L = ω at +h,  R = ω² at −h*",
         psi_w_at_ph, psi_w2_at_mhs, h, -h_star),
        ("L3: L = ω at +h,  R = ω² at −h   (helicity-correct)",
         psi_w_at_ph, psi_w2_at_mh, h, -h),
    ]

    # Three perturbation builders
    pert_builders = [
        ("V_proj  = g·(P_ω − P_ω²)",
         make_proj_perturbation(C3)),
        ("V_screw = Peierls g·(r·ẑ)",
         make_peierls_perturbation(bonds, k_P_cart, lambda r: r[2])),
        ("V_CFJ   = Peierls g·((r×k_P)·ẑ)",
         make_peierls_perturbation(bonds, k_P_cart,
                                   lambda r: np.cross(r, k_P_cart)[2])),
    ]

    # g sample grid (need both small and moderate g for polynomial fit)
    g_vals = np.array([1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2])

    # Candidates (linear and quadratic forms)
    sin_arg_h = math.sqrt(5/8)
    Im_h = math.sqrt(5)/2
    Re_h = math.sqrt(3)/2
    abs_h = math.sqrt(2)
    Im_h_over_abs_h_sq = math.sqrt(5)/4
    candidates_linear = [
        ("1                 ", 1.0),
        ("sin(arg h) = √(5/8)", sin_arg_h),
        ("Im(h) = √5/2      ", Im_h),
        ("Im(h)/|h|² = √5/4 ", Im_h_over_abs_h_sq),
        ("Re(h) = √3/2      ", Re_h),
        ("|h| = √2          ", abs_h),
    ]
    candidates_quadratic = [
        ("1                  ", 1.0),
        ("sin²(arg h) = 5/8  ", 5/8),
        ("sin·cos = √15/8    ", math.sqrt(15)/8),
        ("Im²(h) = 5/4       ", 5/4),
        ("Im(h)·Re(h) = √15/4", math.sqrt(15)/4),
        ("Re²(h) = 3/4       ", 3/4),
    ]

    for opt_label, psi_L, psi_R, h_L, h_R in options:
        print("\n" + "=" * 70)
        print(f"{opt_label}")
        print("=" * 70)
        print(f"  h_L = {h_L:+.6f}  h_R = {h_R:+.6f}")
        print(f"  Δarg₀ = arg(h_L) − arg(h_R) = "
              f"{math.degrees(np.angle(h_L) - np.angle(h_R)):+.4f}°")

        for pert_label, builder in pert_builders:
            print(f"\n  --- {pert_label} ---")
            d_args = []
            dh_Ls = []
            dh_Rs = []
            for g in g_vals:
                V_g = builder(g)
                eL, _ = follow_eigenvalue(BP, V_g, h_L, psi_L)
                eR, _ = follow_eigenvalue(BP, V_g, h_R, psi_R)
                d_arg = np.angle(eL) - np.angle(eR) - (np.angle(h_L) - np.angle(h_R))
                # Wrap to (-π, π]
                d_arg = (d_arg + math.pi) % (2 * math.pi) - math.pi
                d_args.append(d_arg)
                dh_Ls.append(eL - h_L)
                dh_Rs.append(eR - h_R)
            d_args = np.array(d_args)
            dh_Ls = np.array(dh_Ls)
            dh_Rs = np.array(dh_Rs)
            gs = g_vals

            # Polynomial fit Δarg = c0 + c1·g + c2·g² + c3·g³
            coefs = fit_polynomial(gs, d_args, degree=3)
            c0, c1, c2, c3 = coefs[:4]

            # Linear coefficient analysis
            c1_abs = abs(c1)
            print(f"    Δarg(g) ≈ {c0:+.2e} + ({c1:+.6e})·g "
                  f"+ ({c2:+.4e})·g² + ({c3:+.4e})·g³")
            if c1_abs < 1e-8:
                print(f"    Linear: ZERO (machine zero)")
            else:
                best_l = None
                for name, val in candidates_linear:
                    if val == 0:
                        continue
                    r = c1_abs / val
                    err = abs(r - round(r))
                    if 0 < r <= 5 and err < 0.01:
                        best_l = (name, val, r)
                        break
                if best_l:
                    nm, vl, r = best_l
                    print(f"    Linear: c₁ = {c1:+.6e}, "
                          f"|c₁| ≈ {round(r):.0f}·{nm.strip()} = {round(r)*vl:.6f}")
                else:
                    print(f"    Linear: c₁ = {c1:+.6e}  (no clean candidate)")

            # Quadratic coefficient analysis
            c2_abs = abs(c2)
            if c2_abs < 1e-8:
                print(f"    Quad:   ZERO")
            else:
                best_q = None
                for name, val in candidates_quadratic:
                    if val == 0:
                        continue
                    r = c2_abs / val
                    err = abs(r - round(r))
                    if 0 < r <= 5 and err < 0.05:
                        best_q = (name, val, r)
                        break
                if best_q:
                    nm, vl, r = best_q
                    print(f"    Quad:   c₂ = {c2:+.6e}, "
                          f"|c₂| ≈ {round(r):.0f}·{nm.strip()} = {round(r)*vl:.6f}")
                else:
                    print(f"    Quad:   c₂ = {c2:+.6e}  (no clean candidate)")

    print("\n" + "=" * 70)
    print("END β.E O(g²) test")
    print("=" * 70)


if __name__ == "__main__":
    main()
