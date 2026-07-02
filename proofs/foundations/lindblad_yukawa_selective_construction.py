#!/usr/bin/env python3
"""
Construction proof for the Yukawa-selective Lindblad on the 96-dim Hilbert
space H = H_visible (12-dim Bloch fibre at P) tensor S (8-dim Cl(6,0) spinor).

This script is the foundational construction file: it builds the 96 x 96
operators (Hamiltonian, Yukawa-like jump operators, projectors) and verifies
their algebraic properties (commutation, unitality, projector idempotency,
species/isotypic intersections) numerically before the prediction layer
(predictions/lindblad_yukawa_selective.py) consumes them to compute the
steady state and mass-flux table.

Construction (per task plan steps A-E):

  Step A. H_total = H_visible (x) S, dim = 12 * 8 = 96.

  Step B. H_full = H_visible (x) I_S + I_visible (x) H_spinor, with
          H_spinor = 0 (canonical baseline; the Lindblad dissipator carries
          all sector-mixing structure).

  Step C. Two jump-operator families, BOTH preserving C_3 on the visible
          side (commute with U_{C_3} (x) I_S):

          Family I (visible C_3-isotypic dephasing on spinor identity):
              L_{alpha, vis} = sqrt(gamma_1) * P_{alpha, vis} (x) I_S,
              alpha in {trivial, omega, omegabar}.
              [Preserves C_3 trivially; same as the prior degenerate-steady-
              state construction predictions/lindblad_isotypic_at_P.py
              embedded in the 96-dim Hilbert space.]

          Family II (Yukawa-like species-mixing on visible identity):
              L_{Y, s} = sqrt(gamma_2) * I_visible (x) X_s,
              s in {e, nu, u, d}.
              X_s is the species-internal L<->R chirality swap on Pi_s
              (rank-2 species projector from theorem_B3_spinor_fermion):
                  X_s = |s_R><s_L| + |s_L><s_R|
              where |s_L> and |s_R> are the chirality-eigenstate basis of
              the 2-dim Pi_s subspace (G_7 |s_L> = +|s_L>, G_7 |s_R> = -|s_R>).
              [Yukawa-like content: this is the dissipator analogue of an
              ELECTROWEAK-induced L<->R coupling for one species at a time;
              the canonical Standard-Model Yukawa Hamiltonian H_Y =
              sum_s y_s (s_L^dag s_R + h.c.) when treated as a Lindblad jump
              channel produces exactly this X_s structure.]

          Both families commute with U_{C_3} (x) I_S, so the dissipator
          PRESERVES the visible-side C_3 isotypic block structure.

          CONJECTURE C1 (PHYSICAL MOTIVATION FLAGGED, NOT DERIVED). The
          Yukawa-like jump operators of family II are NOT derived from MDL
          + binary self-inverse toggle. The motivation comes from the
          physical Standard-Model Yukawa interaction (electroweak Higgs
          mechanism), reframed as a Lindblad dissipator. This is a
          structural conjecture beyond the framework's two axioms;
          flagged as such here and in the companion .md.

  Step D. Vectorize the Lindblad superoperator (96^2 = 9216 dim) and
          compute its kernel via SVD.

  Step E. For each species s and each C_3 isotypic sector alpha, compute
              m_{s, alpha} = Tr[(P_{alpha, vis} P_h) (x) Pi_s
                                * (sum_jump L^dag L) * rho_ss]
          giving a 4 x 3 mass-flux table.

  Step F. Per-species Koide check.

This script does only the algebraic construction and small-dim verifications.
The 9216 x 9216 SVD and the steady-state computation are done in the
companion prediction file predictions/lindblad_yukawa_selective.py.

References:
  Lindblad 1976 (CMP 48, 119); Gorini-Kossakowski-Sudarshan 1976
  (J. Math. Phys. 17, 821); Wolf 2012 "Quantum Channels and Operations"
  Theorem 6.1 (unital channels) and §6 fixed-point set characterization;
  Breuer-Petruccione 2002 Ch. 3.
  Theorem B3 (proofs/foundations/theorem_B3_spinor_fermion.py) for the
  species-x-chirality basis on the 8-dim Cl(6,0) spinor; chirality
  operator G_7 = -i Gamma_1 ... Gamma_6.

Prints "OK:" on success.
"""

from __future__ import annotations

import sys
import os
import itertools

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, REPO_ROOT)

# Reuse the spinor-coupled construction's algebraic infrastructure for the
# visible Bloch fibre, the spinor build, the C_3 projectors, the species
# projectors, and the chirality operator G_7. This is a chain-import per
# parameter_linter rules: lindblad_yukawa_selective_construction is a
# DERIVED downstream construction.
from proofs.foundations import lindblad_spinor_coupled_construction as cstr  # noqa: E402

TOL = 1e-10

# --------------------------------------------------------------------------
# Step A. Total Hilbert space H = H_visible (x) S, dim = 96
# --------------------------------------------------------------------------

I_VIS = cstr.I_VIS
N_VIS = cstr.N_VIS
assert N_VIS == 12
I_S = cstr.I_S
H_vis = cstr.H_vis
DIM = N_VIS * 8
assert DIM == 96
I_TOT = np.eye(DIM, dtype=complex)

# Visible-side C_3 isotypic projectors and h-eigenspace projector
P_vis_triv = cstr.P_vis_triv
P_vis_om = cstr.P_vis_om
P_vis_omb = cstr.P_vis_omb
P_h = cstr.P_h
U_C3_vis = cstr.U_C3_vis

# Spinor-side species projectors (Pi_e, Pi_nu, Pi_u, Pi_d each rank-2 by
# theorem_B3_spinor_fermion Step 4) and chirality operator G_7
Pi_e = cstr.Pi_e
Pi_nu = cstr.Pi_nu
Pi_u = cstr.Pi_u
Pi_d = cstr.Pi_d
G7 = cstr.G7

# Verify chirality operator: G_7^2 = I_S, G_7 = G_7^dag
assert np.allclose(G7 @ G7, I_S, atol=TOL), "G_7^2 != I_S"
assert np.allclose(G7, G7.conj().T, atol=TOL), "G_7 not Hermitian"

# Each species projector commutes with G_7 (each Pi_s contains one L state
# and one R state; theorem_B3_spinor_fermion Step 4)
for name, Pi_s in [('e', Pi_e), ('nu', Pi_nu), ('u', Pi_u), ('d', Pi_d)]:
    err = np.linalg.norm(Pi_s @ G7 - G7 @ Pi_s)
    assert err < TOL, f"[Pi_{name}, G_7] = {err}"

# --------------------------------------------------------------------------
# Step B. Total Hamiltonian
# --------------------------------------------------------------------------

H_spinor = np.zeros((8, 8), dtype=complex)  # H_spinor = 0 per task plan
H_full = np.kron(H_vis, I_S) + np.kron(I_VIS, H_spinor)
assert np.allclose(H_full, H_full.conj().T, atol=TOL), "H_full not Hermitian"
assert H_full.shape == (96, 96)

# --------------------------------------------------------------------------
# Step C. Yukawa-like jump operators
# --------------------------------------------------------------------------

K_STAR = 3  # Hard-coded locally; the prediction-layer file imports k_star.py.
gamma_1 = 1.0 / K_STAR  # Family I (W4 cancellation rate per step)
gamma_2 = 1.0 / K_STAR  # Family II (Yukawa rate; CONJECTURAL: matched to gamma_1)


def yukawa_X(Pi_s, G7_op):
    """Build the L<->R chirality swap operator X_s on the rank-2 Pi_s.

    Within Pi_s, G_7 has eigenvalues +1 (L state) and -1 (R state). The
    swap operator is X_s = |s_R><s_L| + |s_L><s_R|, which is Hermitian
    and idempotent on Pi_s (X_s^2 = Pi_s).

    By theorem_B3_spinor_fermion Step 4, each species contains exactly
    one L state and one R state, so the swap is well-defined.
    """
    # Project G_7 onto Pi_s
    G7_s = Pi_s @ G7_op @ Pi_s
    # Find chirality eigenvectors within Pi_s (the 6 zero eigenvalues correspond
    # to the orthogonal complement; the +1 and -1 eigenvalues give |L>, |R>)
    ew, ev = np.linalg.eigh(G7_s)
    # Sort by eigenvalue descending; the +1 eigenvector is the L state
    order = np.argsort(-ew)
    v_L = ev[:, order[0]]
    v_R = ev[:, order[-1]]
    # Confirm eigenvalues are ~ +/- 1 (within Pi_s subspace)
    assert abs(ew[order[0]] - 1.0) < 1e-8, f"L eigenvalue {ew[order[0]]}, expected +1"
    assert abs(ew[order[-1]] - (-1.0)) < 1e-8, f"R eigenvalue {ew[order[-1]]}, expected -1"
    # Build swap: |R><L| + |L><R|
    X = np.outer(v_R, v_L.conj()) + np.outer(v_L, v_R.conj())
    return X


X_e = yukawa_X(Pi_e, G7)
X_nu = yukawa_X(Pi_nu, G7)
X_u = yukawa_X(Pi_u, G7)
X_d = yukawa_X(Pi_d, G7)

# Verify each X_s is Hermitian and X_s^2 = Pi_s (idempotent on Pi_s; zero
# off Pi_s)
for name, X, Pi_s in [('e', X_e, Pi_e), ('nu', X_nu, Pi_nu),
                        ('u', X_u, Pi_u), ('d', X_d, Pi_d)]:
    assert np.allclose(X, X.conj().T, atol=TOL), f"X_{name} not Hermitian"
    err = np.linalg.norm(X @ X - Pi_s)
    assert err < 1e-8, f"X_{name}^2 - Pi_{name}: {err}"
    # Also verify Pi_s X Pi_s = X (X is supported on Pi_s)
    err2 = np.linalg.norm(Pi_s @ X @ Pi_s - X)
    assert err2 < 1e-8, f"Pi_{name} X_{name} Pi_{name} - X_{name}: {err2}"

# Verify sum_s X_s^2 = sum_s Pi_s = I_S (since species projectors partition
# the spinor by theorem_B3_spinor_fermion Step 4)
sum_X_sq = X_e @ X_e + X_nu @ X_nu + X_u @ X_u + X_d @ X_d
assert np.linalg.norm(sum_X_sq - I_S) < 1e-8, \
    f"sum X^2 != I_S: ||residual|| = {np.linalg.norm(sum_X_sq - I_S)}"

# Build the family I and family II jump operators
L_family_I = []
for name, P_a in [('triv', P_vis_triv), ('omega', P_vis_om), ('omegabar', P_vis_omb)]:
    L_family_I.append(np.sqrt(gamma_1) * np.kron(P_a, I_S))

L_family_II = []
for name, X_s in [('e', X_e), ('nu', X_nu), ('u', X_u), ('d', X_d)]:
    L_family_II.append(np.sqrt(gamma_2) * np.kron(I_VIS, X_s))

L_all = L_family_I + L_family_II
print(f"Family I jump operators: {len(L_family_I)} (visible C_3 dephasing)")
print(f"Family II jump operators: {len(L_family_II)} (Yukawa species L<->R swap)")
print(f"Total: {len(L_all)}")

# Verify unitality of total dissipator
S_check = sum(L.conj().T @ L for L in L_all)
S_expected = (gamma_1 + gamma_2) * I_TOT
err_unital = np.linalg.norm(S_check - S_expected)
assert err_unital < 1e-9, \
    f"Total dissipator not unital: ||sum L^dL - (g1+g2) I||= {err_unital}"
print(f"Total dissipator: sum L^dL = {gamma_1 + gamma_2:.6f} I_96 (unital, residual {err_unital:.3e})")

# Verify that BOTH families preserve C_3 (commute with U_C3 (x) I_S)
U_C3_full = np.kron(U_C3_vis, I_S)
max_comm_I = max(np.linalg.norm(U_C3_full @ L - L @ U_C3_full) for L in L_family_I)
max_comm_II = max(np.linalg.norm(U_C3_full @ L - L @ U_C3_full) for L in L_family_II)
print(f"max ||[L_family_I, U_C3 (x) I]||: {max_comm_I:.3e} (should be ~0)")
print(f"max ||[L_family_II, U_C3 (x) I]||: {max_comm_II:.3e} (should be ~0)")
assert max_comm_I < 1e-10, "Family I jumps do NOT preserve C_3"
assert max_comm_II < 1e-10, "Family II jumps do NOT preserve C_3"
print("BOTH families preserve C_3: dissipator is C_3-symmetric on the visible side.")

# --------------------------------------------------------------------------
# Verify h-eigenspace C_3 content (1, 1, 0) (theorem_B5_3_core Step 5)
# --------------------------------------------------------------------------
d_th = float(np.trace(P_vis_triv @ P_h).real)
d_oh = float(np.trace(P_vis_om @ P_h).real)
d_obh = float(np.trace(P_vis_omb @ P_h).real)
assert abs(d_th - 1.0) < TOL
assert abs(d_oh - 1.0) < TOL
assert abs(d_obh - 0.0) < TOL
print(f"h-eigenspace C_3 content: ({d_th:.0f}, {d_oh:.0f}, {d_obh:.0f}) = (1, 1, 0)")

print()
print("OK: lindblad_yukawa_selective_construction algebraic verifications complete.")
