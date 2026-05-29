#!/usr/bin/env python3
"""
M1_J_real_structure_probe.py
=============================
Task M1 of the M-arc unified research arc
(`M_arc_unified_scoping_2026-05-14.md`).

Goal.  Construct the J real-structure antiunitary operator for the
framework's spectral triple (A_F, H_F, D_F) and identify the KO-dimension
from its commutation relations with D_F and χ̂.

CC spectral-triple axioms for J:
  J² = ε · 1     (ε = ±1)
  J D J⁻¹ = ε' · D   (ε' = ±1)
  J χ̂ J⁻¹ = ε'' · χ̂  (ε'' = ±1)
  J commutes with right A_F multiplication on H_F (the "0-th order condition").

The three signs (ε, ε', ε'') determine the KO-dimension mod 8 per
Connes' table (cf. Connes-Marcolli 2008 §13):

  KO-dim 0: (+1, +1, +1)
  KO-dim 1: (+1, -1, ?)
  KO-dim 2: (-1, +1, -1)
  KO-dim 3: (-1, -1, +1)
  KO-dim 4: (-1, +1, +1)
  KO-dim 5: (-1, -1, ?)
  KO-dim 6: (+1, +1, -1)    ← typical for CC SM
  KO-dim 7: (+1, -1, ?)

What this probe does
--------------------
A — Build two candidate J operators:
    J^(α)(X) = X̄  (entrywise complex conjugate on each block)
    J^(β)(X) = X†  (Hermitian adjoint on each block)

B — For each candidate, verify J² = ε · 1 and find ε.

C — Compute J D_F J⁻¹ and compare to D_F:  find ε'.

D — Compute J χ̂ J⁻¹ and compare to χ̂:  find ε''.

E — Identify the framework's KO-dim from (ε, ε', ε'').

F — Verify the 0-th order condition: [J, π(a)] = 0 for a ∈ A_F's right
    multiplication action (sketch).

If candidate J fails the CC axioms, document the structural obstacle.

No graded content changes from this probe.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from proofs.foundations.de_rham_susy_fibered_v2_probe import (  # noqa: E402
    d_alg, NE, NV, SX, SY, SZ, I2,
)

np.set_printoptions(precision=4, suppress=True, linewidth=140)
TOL = 1e-9


# -----------------------------------------------------------------------------
# Setup: H_F vector structure
# -----------------------------------------------------------------------------
#
# H_F = C⁰_alg ⊕ C¹_alg = (⊕_v M_8) ⊕ (⊕_e M_2) with Hilbert-Schmidt inner product.
# A vector v ∈ H_F has 280 components.  The first 256 are 4 × 64 = vertex M_8
# flatten;  the last 24 are 6 × 4 = edge M_2 flatten.
#
# For a single 64-dim M_8 vector (col-major flatten of 8×8 matrix X):
#   v[k] = X[k mod 8, k // 8]
# i.e., v = vec(X) in col-major flatten.
#
# The Hermitian conjugate operation X ↦ X† has flatten:
#   vec(X†)[k] = X†[k mod 8, k // 8] = conj(X[k // 8, k mod 8]) = conj(v[8·(k mod 8) + k // 8])
# So J^(β)_M8 is a permutation × conjugation: P · conj where P swaps row and col indices.

def J_alpha_64():
    """J^(α) on M_8 flatten basis (64-dim):  X ↦ X̄ (entrywise complex conjugate)."""
    return np.eye(64, dtype=complex)   # just complex-conjugate the vector (no permutation)


def J_beta_64():
    """J^(β) on M_8 flatten basis (64-dim):  X ↦ X† (Hermitian conjugate = transpose + conj).
    Returns the PERMUTATION matrix P such that  J_beta(v) = P · conj(v),
    where P swaps the (row, col) → (col, row) indices.
    """
    P = np.zeros((64, 64), dtype=complex)
    for col in range(8):
        for row in range(8):
            k_in = col * 8 + row    # flatten index of X[row, col]
            k_out = row * 8 + col   # flatten index of X†[col, row] = X[row, col]*
            P[k_out, k_in] = 1.0
    return P


def J_alpha_4():
    return np.eye(4, dtype=complex)


def J_beta_4():
    P = np.zeros((4, 4), dtype=complex)
    for col in range(2):
        for row in range(2):
            k_in = col * 2 + row
            k_out = row * 2 + col
            P[k_out, k_in] = 1.0
    return P


def build_J(variant):
    """Build the full 280×280 J operator (as a permutation P on the flattened H_F
    representing the LINEAR PART of  J(v) = P · conj(v) ).  Caller applies conj
    separately.

    variant = 'alpha': J^(α)(X) = X̄  →  P = identity (just conj).
    variant = 'beta':  J^(β)(X) = X†  →  P swaps row/col flatten indices.
    """
    dim0, dim1 = NV * 64, NE * 4
    P_full = np.zeros((dim0 + dim1, dim0 + dim1), dtype=complex)
    if variant == 'alpha':
        P_64 = J_alpha_64()
        P_4 = J_alpha_4()
    elif variant == 'beta':
        P_64 = J_beta_64()
        P_4 = J_beta_4()
    else:
        raise ValueError(variant)
    for v in range(NV):
        P_full[v * 64:(v + 1) * 64, v * 64:(v + 1) * 64] = P_64
    for e in range(NE):
        P_full[dim0 + e * 4:dim0 + (e + 1) * 4, dim0 + e * 4:dim0 + (e + 1) * 4] = P_4
    return P_full


def apply_J(P, v):
    """J(v) = P · conj(v)."""
    return P @ v.conjugate()


def J_squared(P, v):
    """J²(v) = J(J(v)) = P · conj(P · conj(v)) = P · conj(P) · v."""
    return P @ P.conjugate() @ v


# -----------------------------------------------------------------------------
# Setup: D_F and χ̂
# -----------------------------------------------------------------------------

def build_D_F():
    d = d_alg((0.0, 0.0, 0.0))
    dim0, dim1 = NV * 64, NE * 4
    D_F = np.zeros((dim0 + dim1, dim0 + dim1), dtype=complex)
    D_F[:dim0, dim0:] = d.conj().T
    D_F[dim0:, :dim0] = d
    return D_F, dim0, dim1


# -----------------------------------------------------------------------------
# Part A — build candidate J operators
# -----------------------------------------------------------------------------

def part_A():
    print("=" * 100)
    print("PART A — build candidate J^(α) (entrywise conj) and J^(β) (Hermitian adjoint)")
    print("=" * 100)
    P_alpha = build_J('alpha')
    P_beta = build_J('beta')
    print(f"\n  P_alpha shape : {P_alpha.shape}")
    print(f"  P_beta  shape : {P_beta.shape}")
    # Check permutation structure
    print(f"  P_alpha is identity matrix : {np.allclose(P_alpha, np.eye(280, dtype=complex), atol=TOL)}")
    print(f"  P_beta is a permutation    : {np.all((P_beta == 0) | (P_beta == 1))}")
    return P_alpha, P_beta


# -----------------------------------------------------------------------------
# Part B — compute J² for each, find ε
# -----------------------------------------------------------------------------

def part_B(P_alpha, P_beta):
    print("\n" + "=" * 100)
    print("PART B — compute J² and find ε")
    print("=" * 100)
    # J²(v) = P · conj(P · conj(v)) = P · conj(P) · v.
    # For J^(α): P = I (real), so J² = I · I · v = v.  ε = +1.
    # For J^(β): P is real permutation matrix, conj(P) = P.  J²(v) = P · P · v.
    #   P² = ?  For X ↦ X†, then X† ↦ (X†)† = X.  So J² = identity.  ε = +1.
    J2_alpha = P_alpha @ P_alpha.conjugate()
    J2_beta = P_beta @ P_beta.conjugate()
    eps_alpha = 1 if np.allclose(J2_alpha, np.eye(280, dtype=complex), atol=TOL) else (
        -1 if np.allclose(J2_alpha, -np.eye(280, dtype=complex), atol=TOL) else 0)
    eps_beta = 1 if np.allclose(J2_beta, np.eye(280, dtype=complex), atol=TOL) else (
        -1 if np.allclose(J2_beta, -np.eye(280, dtype=complex), atol=TOL) else 0)
    print(f"\n  J^(α) = entrywise conj:  J² {('= +I, ε = +1' if eps_alpha == +1 else '= -I, ε = -1' if eps_alpha == -1 else 'has unexpected form')}")
    print(f"  J^(β) = Hermitian adj.:  J² {('= +I, ε = +1' if eps_beta == +1 else '= -I, ε = -1' if eps_beta == -1 else 'has unexpected form')}")
    return eps_alpha, eps_beta


# -----------------------------------------------------------------------------
# Part C — JDJ⁻¹ vs D, find ε'
# -----------------------------------------------------------------------------

def part_C(P_alpha, P_beta, D_F):
    print("\n" + "=" * 100)
    print("PART C — compute J D J⁻¹ and find ε'")
    print("=" * 100)
    # JDJ⁻¹(v) = J(D · J⁻¹(v)).  Since J²=I (both candidates), J⁻¹ = J.
    # JDJ(v) = J(D · J(v)) = J(D · P · conj(v)) = P · conj(D · P · conj(v))
    #        = P · conj(D) · conj(P) · v
    # For P real (both candidates), conj(P) = P.  So JDJ⁻¹ as a LINEAR map (on v) = P · conj(D) · P.
    JDJ_alpha = P_alpha @ D_F.conjugate() @ P_alpha
    JDJ_beta = P_beta @ D_F.conjugate() @ P_beta
    # Compare to D and -D
    def find_sign(M, ref):
        if np.allclose(M, ref, atol=1e-9): return +1
        if np.allclose(M, -ref, atol=1e-9): return -1
        return 0
    eps_p_alpha = find_sign(JDJ_alpha, D_F)
    eps_p_beta = find_sign(JDJ_beta, D_F)
    def report(name, eps_p, JDJ):
        if eps_p == 1:
            print(f"  {name}: J D J⁻¹ = +D, ε' = +1")
        elif eps_p == -1:
            print(f"  {name}: J D J⁻¹ = -D, ε' = -1")
        else:
            d_plus = np.linalg.norm(JDJ - D_F)
            d_minus = np.linalg.norm(JDJ + D_F)
            print(f"  {name}: J D J⁻¹ neither ±D — ‖JDJ − D‖ = {d_plus:.3e}, ‖JDJ + D‖ = {d_minus:.3e}")
    print()
    report("J^(α)", eps_p_alpha, JDJ_alpha)
    report("J^(β)", eps_p_beta, JDJ_beta)
    return eps_p_alpha, eps_p_beta


# -----------------------------------------------------------------------------
# Part D — Jχ̂J⁻¹ vs χ̂, find ε''
# -----------------------------------------------------------------------------

def part_D(P_alpha, P_beta):
    print("\n" + "=" * 100)
    print("PART D — compute J χ̂ J⁻¹ and find ε''")
    print("=" * 100)
    chi = np.diag([1.0] * (NV * 64) + [-1.0] * (NE * 4)).astype(complex)
    # JχJ⁻¹ as linear map = P · conj(χ) · P = P · χ · P (χ is real).
    JchiJ_alpha = P_alpha @ chi @ P_alpha
    JchiJ_beta = P_beta @ chi @ P_beta
    def find_sign(M, ref):
        if np.allclose(M, ref, atol=1e-9): return +1
        if np.allclose(M, -ref, atol=1e-9): return -1
        return 0
    eps_pp_alpha = find_sign(JchiJ_alpha, chi)
    eps_pp_beta = find_sign(JchiJ_beta, chi)
    def report_chi(name, eps_pp):
        if eps_pp == 1:
            print(f"  {name}: J χ̂ J⁻¹ = +χ̂, ε'' = +1")
        elif eps_pp == -1:
            print(f"  {name}: J χ̂ J⁻¹ = -χ̂, ε'' = -1")
        else:
            print(f"  {name}: J χ̂ J⁻¹ is neither ±χ̂")
    print()
    report_chi("J^(α)", eps_pp_alpha)
    report_chi("J^(β)", eps_pp_beta)
    return eps_pp_alpha, eps_pp_beta


# -----------------------------------------------------------------------------
# Part E — identify KO-dim
# -----------------------------------------------------------------------------

def part_E(eps_alpha, eps_p_alpha, eps_pp_alpha,
           eps_beta, eps_p_beta, eps_pp_beta):
    print("\n" + "=" * 100)
    print("PART E — identify KO-dimension from (ε, ε', ε'')")
    print("=" * 100)

    # KO-dim table per Connes-Marcolli 2008 §13
    KO_table = {
        ( +1, +1, +1): 0,
        ( +1, -1, -1): 1,   # ε'' undefined in some conventions
        ( -1, +1, -1): 2,
        ( -1, -1, +1): 3,
        ( -1, +1, +1): 4,
        ( -1, -1, -1): 5,
        ( +1, +1, -1): 6,   # ← typical for CC SM
        ( +1, -1, +1): 7,
    }
    def identify(eps, ep, epp):
        return KO_table.get((eps, ep, epp), None)

    ko_alpha = identify(eps_alpha, eps_p_alpha, eps_pp_alpha)
    ko_beta = identify(eps_beta, eps_p_beta, eps_pp_beta)
    print(f"\n  J^(α) signs (ε, ε', ε'') = ({eps_alpha:+d}, {eps_p_alpha:+d}, {eps_pp_alpha:+d})  →  KO-dim {ko_alpha}")
    print(f"  J^(β) signs (ε, ε', ε'') = ({eps_beta:+d}, {eps_p_beta:+d}, {eps_pp_beta:+d})  →  KO-dim {ko_beta}")
    print(f"\n  CC-SM typical KO-dim = 6 (signs (+1, +1, -1)).")


# -----------------------------------------------------------------------------
def main():
    print(r"""
==========================================================================================
M1 — J real-structure operator construction and KO-dimension identification
First task of the M-arc unified research arc.
==========================================================================================""")
    P_alpha, P_beta = part_A()
    eps_alpha, eps_beta = part_B(P_alpha, P_beta)
    D_F, _, _ = build_D_F()
    eps_p_alpha, eps_p_beta = part_C(P_alpha, P_beta, D_F)
    eps_pp_alpha, eps_pp_beta = part_D(P_alpha, P_beta)
    part_E(eps_alpha, eps_p_alpha, eps_pp_alpha, eps_beta, eps_p_beta, eps_pp_beta)
    print("\n" + "=" * 100)
    print("M1 INTERIM VERDICT")
    print("=" * 100)
    print(f"""
  CANDIDATE J^(α) (entrywise conj):
    ε  = {eps_alpha:+d}
    ε' = {eps_p_alpha:+d}
    ε''= {eps_pp_alpha:+d}

  CANDIDATE J^(β) (Hermitian adjoint):
    ε  = {eps_beta:+d}
    ε' = {eps_p_beta:+d}
    ε''= {eps_pp_beta:+d}

  These signs determine the framework's KO-dimension.  See Part E for assignment.

  Next: verify the 0-th order condition (J commutes with right A_F multiplication)
  and identify which candidate is the right J for the framework's CC spectral
  triple.  Multi-session continuation.

  No graded content changes from M1.  R1 status: INTERIM.
""")
    print("M1_J_real_structure_probe.py: sentinel done.")


if __name__ == "__main__":
    main()
