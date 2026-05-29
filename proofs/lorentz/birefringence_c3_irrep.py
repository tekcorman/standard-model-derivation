#!/usr/bin/env python3
"""
β.E — numerical diagnostic of the cosmic-birefringence MDL terms.

NOTE (2026-04-25 PM).  See `docs/theorems/theorem_dark_correction_mdl.md` for the
canonical reading.  This script is a diagnostic that identifies which
MDL-permitted parity-odd functional of h naive chiral perturbation isolates
under a given L/R identification — it is NOT a direct test of the leading
prediction β = sin(arg h)·α_EM.  Multiple MDL terms (Im(h), sin(arg h),
arg(h), Im(h)/|1−h|, etc.) are MDL-permitted; this script's V_proj
projects onto Im(h)·g (a higher-bit-cost MDL term), while sin(arg h) is
the leading (lowest-bit-cost) MDL term that the framework's prediction
actually takes (Lemma 1 in the synthesis doc).

Tests how the L vs R photon-polarization eigenstates of B(P) split at first
order under a chiral U(1) perturbation that breaks T-symmetry.  Reports
the linear coefficient |Δarg/g| under three L/R identifications and three
perturbation models.

Physical model
--------------
At the P-point of the I4_132 BZ, the directed Hashimoto B(P) has the
doubly-degenerate eigenvalue h = (√3 + i√5)/2 and its complex conjugate
h* = (√3 - i√5)/2, each multiplicity 2.  The 2D h-eigenspace decomposes
under the C₃ stabilizer of P (rotation by 2π/3 about [111]) as

    V_h = (trivial-irrep) ⊕ (ω-irrep)

where ω = exp(2πi/3).  The conjugate eigenspace V_{h*} decomposes by
T-symmetry (complex conjugation) as

    V_{h*} = (trivial-irrep) ⊕ (ω²-irrep).

NOTE.  predictions/B_P_doubly_degenerate_h_derivation.md L131 states this
(trivial⊕ω) / (trivial⊕ω²) split correctly.  docs/theorems/theorem_cosmic_birefringence.md
states "ω and ω* irreps in the h-eigenspace" — this is incorrect (see
companion task to flag).

The L (left circular) and R (right circular) photon-polarization
eigenstates of B(P) are identified as

    |L⟩  = ω-irrep eigenstate at walker eigenvalue h
    |R⟩  = ω²-irrep eigenstate at walker eigenvalue h*

|L⟩ and |R⟩ are related by complex conjugation (T-symmetry).  This is the
"L and R polarizations are at h and h*, distinguished by C₃ irrep within
their respective eigenspaces" framing.

Test protocol
-------------
1. Build B(P) at k_P = (1/4, 1/4, 1/4) · 2π.
2. Find the h-eigenspace (mult 2) and h*-eigenspace (mult 2).
3. Build the C₃ permutation operator on bonds, verify [C₃, B(P)] = 0.
4. Diagonalize C₃ within V_h, V_{h*} to identify ω-state, ω²-state.
5. Define L = ω-state at h, R = ω²-state at h*.
6. Apply a chiral U(1) perturbation V (several models tested):
     V_proj   = g · (P_ω - P_ω²) acting globally on all 12 bonds
     V_screw  = Peierls phase g · (r · ẑ) on each bond (axial-vector
                gauge field along [001] screw axis)
     V_CFJ    = Peierls phase g · ((r × k_P) · ẑ) (screw-axis curl term)
7. Compute first-order eigenvalue shift δh_L, δh_R and phase shift
   δarg(L), δarg(R).
8. Extract the coefficient c in δβ = c · g · (functional of h).
9. Compare c against candidates: 1, sin(arg h) = √(5/8), Im(h) = √5/2,
   Im(h)/|h|² = √5/4, etc.
10. Report which V model + functional gives a clean rational/algebraic
    coefficient.

The output is numerical evidence for or against the canonical claim
β = sin(arg h) · α_EM.  It does not by itself constitute a CAS-grade
derivation of the coefficient, but it can rule out coefficient = 1 if
the data is inconsistent.
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
# Step 1: Build B(P), find h- and h*-eigenspaces.
# =============================================================================

def build_BP():
    """Return B(P) at the I4_132 P-point in the 12-bond basis."""
    bonds = find_bonds()
    B_mat = build_B_matrix()
    B_inv = la.inv(B_mat)
    k_P_frac = np.array([0.25, 0.25, 0.25])
    k_P_cart = 2 * math.pi * (B_mat.T @ k_P_frac)
    BP = build_hashimoto_bloch(k_P_cart, bonds, B_inv)
    return BP, bonds


def find_eigenspace(BP, target_eigval, tol=1e-8):
    """Return orthonormal basis (12 × m) of eigenspace at target_eigval."""
    evals, evecs = la.eig(BP)
    indices = np.where(np.abs(evals - target_eigval) < tol)[0]
    V = evecs[:, indices]
    # QR-orthonormalize
    Q, _ = la.qr(V)
    # Verify
    err = la.norm(BP @ Q - target_eigval * Q) / la.norm(Q)
    assert err < 1e-10, f"Eigenspace verification failed: {err}"
    return Q


# =============================================================================
# Step 2: Build the C3 stabilizer of P on the 12-bond space.
# =============================================================================

def build_C3_op(bonds):
    """C₃ rotation by 2π/3 about [111], represented on the 12 directed bonds."""
    R3 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    # Atom permutation: 0 (on diagonal) is fixed; 1→3→2→1
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
        else:
            raise RuntimeError(
                f"C₃ image of bond {i} not found in bonds list")
    C3 = np.zeros((12, 12), dtype=complex)
    for i in range(12):
        C3[bond_perm[i], i] = 1.0
    return C3


def project_to_irrep(V, C3, target):
    """Project V (12×m basis) onto target-eigenspace of C3|V.

    Returns the unit-norm eigenstate in the full 12-dim basis, or None if
    the target eigenvalue is not present in C3|V (within tol).
    """
    C3_in_V = V.conj().T @ C3 @ V       # m×m
    evals, evecs = la.eig(C3_in_V)
    idx = int(np.argmin(np.abs(evals - target)))
    if abs(evals[idx] - target) > 1e-8:
        return None
    psi_in_V = evecs[:, idx]
    psi = V @ psi_in_V
    return psi / la.norm(psi)


# =============================================================================
# Step 3: Apply a chiral perturbation V(g) and compute first-order shifts.
# =============================================================================

def apply_perturbation_proj(BP, C3, g):
    """V_proj = g · (P_ω - P_ω²) on the global 12-bond space."""
    omega = np.exp(2j * math.pi / 3)
    omega2 = omega.conjugate()
    # Eigendecomposition of C3 (unitary) → spectral projectors
    c_evals, c_evecs = la.eig(C3)
    # Normalize eigenvectors
    for j in range(c_evecs.shape[1]):
        c_evecs[:, j] /= la.norm(c_evecs[:, j])
    P_omega = np.zeros_like(BP)
    P_omega2 = np.zeros_like(BP)
    for j, ev in enumerate(c_evals):
        v = c_evecs[:, j:j + 1]
        if abs(ev - omega) < 1e-8:
            P_omega += v @ v.conj().T
        elif abs(ev - omega2) < 1e-8:
            P_omega2 += v @ v.conj().T
    V = g * (P_omega - P_omega2)
    return BP + V


def apply_perturbation_peierls(BP, bonds, k_P_cart, g, chir_func):
    """Peierls perturbation: rebuild B(P) with bond phase += g · chir_func(r)."""
    n = len(bonds)
    # bond displacements
    r_vecs = []
    for src, tgt, cell in bonds:
        disp = (ATOMS[tgt] + cell[0] * A_PRIM[0] + cell[1] * A_PRIM[1]
                + cell[2] * A_PRIM[2] - ATOMS[src])
        r_vecs.append(disp)
    r_vecs = np.array(r_vecs)
    chir = np.array([chir_func(r) for r in r_vecs])
    # Optional: subtract mean to remove gauge-trivial uniform shift
    chir = chir - np.mean(chir)
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


def first_order_shift(B_pert, psi, h_unperturbed):
    """⟨psi|B_pert|psi⟩ - h_unperturbed (first-order eigenvalue shift)."""
    return np.vdot(psi, B_pert @ psi) - h_unperturbed


# =============================================================================
# Driver.
# =============================================================================

def main():
    print("=" * 70)
    print("β.E — chiral-perturbation test of β = c · sin(arg h) · α_EM")
    print("=" * 70)

    BP, bonds = build_BP()
    h = complex(math.sqrt(3) / 2, math.sqrt(5) / 2)
    h_star = h.conjugate()

    print("\nStep 1 — Walker eigenspaces at ±h, ±h*")
    V_h_plus = find_eigenspace(BP, h)
    V_hs_plus = find_eigenspace(BP, h_star)
    V_h_minus = find_eigenspace(BP, -h)
    V_hs_minus = find_eigenspace(BP, -h_star)
    print(f"  dim V_+h  = {V_h_plus.shape[1]}, "
          f"dim V_+h* = {V_hs_plus.shape[1]}")
    print(f"  dim V_-h  = {V_h_minus.shape[1]}, "
          f"dim V_-h* = {V_hs_minus.shape[1]}")
    print(f"  h  = {h}")
    print(f"  h* = {h_star}")
    print(f"  arg(h)  = {math.degrees(np.angle(h)):.4f}°")
    print(f"  sin(arg h) = √(5/8) = {math.sqrt(5/8):.6f}")
    print(f"  |h|        = √2 = {math.sqrt(2):.6f}")
    print(f"  Im(h)      = √5/2 = {math.sqrt(5)/2:.6f}")
    print(f"  Im(h)/|h|² = √5/4 = {math.sqrt(5)/4:.6f}")

    print("\nStep 2 — C₃ stabilizer of P on 12-bond space")
    C3 = build_C3_op(bonds)
    print(f"  C₃³ = I?  max|·| = {np.max(np.abs(C3 @ C3 @ C3 - np.eye(12))):.2e}")
    print(f"  [C₃, B(P)] = 0?  max|·| = "
          f"{np.max(np.abs(C3 @ BP - BP @ C3)):.2e}")

    print("\nStep 3 — Project onto C₃ irreps inside each walker eigenspace")
    omega = np.exp(2j * math.pi / 3)
    omega2 = omega.conjugate()

    # Each walker eigenspace decomposes as trivial + (ω or ω²).
    # ω-states live at +h, +h*.  ω²-states live at -h, -h*.
    psi_omega_at_plus_h   = project_to_irrep(V_h_plus,   C3, omega)
    psi_omega_at_plus_hs  = project_to_irrep(V_hs_plus,  C3, omega)
    psi_omega2_at_minus_h = project_to_irrep(V_h_minus,  C3, omega2)
    psi_omega2_at_minus_hs = project_to_irrep(V_hs_minus, C3, omega2)

    # Sanity: confirm cross-eigenspace projections are absent
    print(f"  ω-state at +h   : {'present' if psi_omega_at_plus_h is not None else 'ABSENT'}")
    print(f"  ω-state at +h*  : {'present' if psi_omega_at_plus_hs is not None else 'ABSENT'}")
    print(f"  ω-state at -h   : "
          f"{'present' if project_to_irrep(V_h_minus, C3, omega) is not None else 'ABSENT'}")
    print(f"  ω²-state at +h  : "
          f"{'present' if project_to_irrep(V_h_plus, C3, omega2) is not None else 'ABSENT'}")
    print(f"  ω²-state at -h  : {'present' if psi_omega2_at_minus_h is not None else 'ABSENT'}")
    print(f"  ω²-state at -h* : {'present' if psi_omega2_at_minus_hs is not None else 'ABSENT'}")

    # Three L/R identification options to test:
    options = []
    options.append((
        "L1: L = ω at +h,  R = ω at +h*  (same irrep, conjugate walker eigenvalue)",
        psi_omega_at_plus_h,  psi_omega_at_plus_hs,  h, h_star))
    options.append((
        "L2: L = ω at +h,  R = ω² at −h*  (T-conjugate: sign-flip + irrep-flip)",
        psi_omega_at_plus_h,  psi_omega2_at_minus_hs, h, -h_star))
    options.append((
        "L3: L = ω at +h,  R = ω² at −h   (irrep-flip, sign-flip, no conjugation)",
        psi_omega_at_plus_h,  psi_omega2_at_minus_h,  h, -h))

    # Sanity check first option (we use this to confirm structure)
    psi_L_opt, psi_R_opt = options[0][1], options[0][2]
    print(f"\n  Sanity (L1): ⟨L|B(P)|L⟩ = {np.vdot(psi_L_opt, BP @ psi_L_opt):.6f}  "
          f"(should be h = {h:.6f})")
    print(f"  Sanity (L1): ⟨R|B(P)|R⟩ = {np.vdot(psi_R_opt, BP @ psi_R_opt):.6f}  "
          f"(should be h* = {h_star:.6f})")
    print(f"  Sanity (L1): ⟨L|C₃|L⟩ = {np.vdot(psi_L_opt, C3 @ psi_L_opt):.6f}  "
          f"(should be ω = {omega:.6f})")
    print(f"  Sanity (L1): ⟨R|C₃|R⟩ = {np.vdot(psi_R_opt, C3 @ psi_R_opt):.6f}  "
          f"(should be ω = {omega:.6f})")

    # k_P_cart for Peierls test
    B_mat = build_B_matrix()
    k_P_cart = 2 * math.pi * (B_mat.T @ np.array([0.25, 0.25, 0.25]))

    # Models: list of (label, perturbation function returning B_pert)
    models = []
    models.append((
        "V_proj  = g·(P_ω − P_ω²) on 12-bond space",
        lambda g: apply_perturbation_proj(BP, C3, g)))
    models.append((
        "V_screw = Peierls phase g·(r·ẑ), z = [001]",
        lambda g: apply_perturbation_peierls(
            BP, bonds, k_P_cart, g, lambda r: r[2])))
    models.append((
        "V_CFJ   = Peierls phase g·((r × k_P)·ẑ), z = [001]",
        lambda g: apply_perturbation_peierls(
            BP, bonds, k_P_cart, g,
            lambda r: np.cross(r, k_P_cart)[2])))

    candidates = [
        ("1                   ", 1.0),
        ("2                   ", 2.0),
        ("sin(arg h) = √(5/8) ", math.sqrt(5 / 8)),
        ("Im(h) = √5/2        ", math.sqrt(5) / 2),
        ("Im(h)/|h| = √(5/8)  ", math.sqrt(5 / 8)),
        ("Im(h)/|h|² = √5/4   ", math.sqrt(5) / 4),
        ("2·sin(arg h)        ", 2 * math.sqrt(5 / 8)),
        ("2·Im(h) = √5        ", math.sqrt(5)),
        ("Re(h) = √3/2        ", math.sqrt(3) / 2),
        ("|h| = √2            ", math.sqrt(2)),
    ]

    g_vals = np.array([1e-6, 1e-5, 1e-4, 1e-3])

    print("\nStep 4 — First-order eigenvalue shifts under each (option, model)")
    print(f"\n  Conventions:")
    print(f"    δh_L      = ⟨L|V|L⟩  where V = (B_pert − BP), first-order PT")
    print(f"    Δarg(g)   = arg(h_L + δh_L) − arg(h_R + δh_R)")
    print(f"    Δarg₀     = arg(h_L) − arg(h_R) (g=0 baseline)")
    print(f"    coefficient = lim_{{g→0}} (Δarg(g) − Δarg₀) / g")

    for opt_label, psi_L, psi_R, h_L, h_R in options:
        print("\n" + "=" * 70)
        print(f"OPTION  {opt_label}")
        print("=" * 70)
        if psi_L is None or psi_R is None:
            print("  SKIP (one of |L⟩, |R⟩ is absent in target eigenspace)")
            continue
        print(f"  ⟨L|B(P)|L⟩ = {np.vdot(psi_L, BP @ psi_L):+.6f}, "
              f"⟨R|B(P)|R⟩ = {np.vdot(psi_R, BP @ psi_R):+.6f}")
        print(f"  ⟨L|C₃|L⟩ = {np.vdot(psi_L, C3 @ psi_L):+.6f}, "
              f"⟨R|C₃|R⟩ = {np.vdot(psi_R, C3 @ psi_R):+.6f}")
        print(f"  ⟨L|R⟩    = {np.vdot(psi_L, psi_R):+.2e}")

        d_arg_unperturbed = np.angle(h_L) - np.angle(h_R)
        print(f"  Δarg₀ = arg(h_L) − arg(h_R) = "
              f"{math.degrees(d_arg_unperturbed):+.4f}°")

        for label, pert_fn in models:
            print(f"\n  --- {label} ---")
            rows = []
            for g in g_vals:
                B_pert = pert_fn(g)
                V = B_pert - BP
                dhL = np.vdot(psi_L, V @ psi_L)
                dhR = np.vdot(psi_R, V @ psi_R)
                argL = np.angle(h_L + dhL)
                argR = np.angle(h_R + dhR)
                d_arg = argL - argR
                rows.append((g, dhL, dhR, d_arg))
            gs = np.array([r[0] for r in rows])
            dhL_re = np.array([r[1].real for r in rows])
            dhL_im = np.array([r[1].imag for r in rows])
            dhR_re = np.array([r[2].real for r in rows])
            dhR_im = np.array([r[2].imag for r in rows])
            d_arg = np.array([r[3] for r in rows])

            cL_re = (dhL_re / gs).mean()
            cL_im = (dhL_im / gs).mean()
            cR_re = (dhR_re / gs).mean()
            cR_im = (dhR_im / gs).mean()
            # For Δarg coefficient, fit (Δarg(g) − Δarg₀) / g; smallest g
            # is most accurate
            d_arg_coef = (d_arg - d_arg_unperturbed) / gs
            c_d_arg = d_arg_coef[0]   # use smallest g for best linearization

            print(f"    Re(δh_L)/g = {cL_re:+.8f}, Im(δh_L)/g = {cL_im:+.8f}")
            print(f"    Re(δh_R)/g = {cR_re:+.8f}, Im(δh_R)/g = {cR_im:+.8f}")
            print(f"    Δarg/g     = {c_d_arg:+.8f}  rad/(unit g)")

            # Magnitudes of interest
            c_dhL = math.hypot(cL_re, cL_im)
            c_d_arg_abs = abs(c_d_arg)
            best_match = None
            best_ratio_err = float('inf')
            for name, val in candidates:
                if val == 0:
                    continue
                ratio = c_d_arg_abs / val
                err = abs(ratio - round(ratio))
                if 0.5 <= ratio <= 4 and err < best_ratio_err:
                    best_ratio_err = err
                    best_match = (name, val, ratio)
            if best_match and best_ratio_err < 0.02:
                name, val, ratio = best_match
                print(f"    |Δarg/g| = {c_d_arg_abs:.8f} ≈ "
                      f"{round(ratio):.0f} × {name.strip()} = "
                      f"{round(ratio) * val:.6f}  ←  CLEAN MATCH")
            else:
                print(f"    |Δarg/g| = {c_d_arg_abs:.8f}  (no clean candidate match)")
                # Show top few candidates for diagnosis
                sorted_cands = sorted(
                    candidates, key=lambda nv: abs(c_d_arg_abs/nv[1] - 1) if nv[1] else 1e9)
                print(f"    Closest candidates:")
                for name, val in sorted_cands[:3]:
                    if val > 0:
                        print(f"      |Δarg/g| / ({name.strip()}) = "
                              f"{c_d_arg_abs / val:.6f}")

    print("\n" + "=" * 70)
    print("END β.E test.  Interpret the numerical coefficients above against")
    print("the candidates listed; clean integer/algebraic ratios are evidence")
    print("for the corresponding analytic form.")
    print("=" * 70)


if __name__ == "__main__":
    main()
