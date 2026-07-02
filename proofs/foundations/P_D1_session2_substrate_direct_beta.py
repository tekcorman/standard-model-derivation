#!/usr/bin/env python3
"""
P_D1_session2_substrate_direct_beta.py
=======================================

P-D1 session 2: compute the framework's substrate-direct β-function
coefficients and identify whether they give SM-β or MSSM-β naturally.

Approach:
  1. Build SM gauge generators (SU(3)_c, SU(2)_L, U(1)_Y) on H_F = 280.
  2. Decompose H_F into per-vertex Fock states (32 per cell) + bilinears (224 matter) + edges (24).
  3. Per-vertex Fock states (32): these are the "physical fermion" candidates.
     Compute T_i(F_32) per gauge factor.
  4. Bilinears (224 matter) + edges (24) = 248 "bosonic" candidates.
     Compute T_i(S_248) per gauge factor.
  5. Compute candidate b_i = -11/3 C_2(adj_i) + (2/3) T_i(F_32) + (1/3) T_i(S_248).
  6. Compare to SM b_i and MSSM b_i.

Caveats acknowledged upfront:
  - This assumes a clean fermion/scalar split which the framework doesn't
    naturally provide (Z probe showed CC inner-fluctuation order-1 fails).
  - The SM gauge embedding (M2) doesn't commute with B3 SU(2)_L, so the
    decomposition is approximate at best.

The cleanest deliverable: a numerical answer that identifies whether
substrate b_i is closer to SM or MSSM (or neither).

No graded content changes.  P-D1 session 2 computational probe.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from fractions import Fraction

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from proofs.foundations.de_rham_susy_fibered_v2_probe import (  # noqa: E402
    d_alg, NV, NE,
)
from proofs.foundations.spectral_beta_from_H_F_probe import (  # noqa: E402
    build_su2L_8, build_su2R_8, build_su3_8, build_u1Y_8,
    lift_to_HF_adjoint, trace_dynkin, trace_dynkin_u1, build_D_F,
)

np.set_printoptions(precision=4, suppress=True, linewidth=140)
TOL = 1e-9


# ---------------------------------------------------------------------------
# Reference β-coefficients (with sign convention: dα_i⁻¹/d ln µ = +b_i/(2π))
# ---------------------------------------------------------------------------

SM_b = {1: Fraction(41, 10), 2: Fraction(-19, 6), 3: Fraction(-7)}
MSSM_b = {1: Fraction(33, 5), 2: Fraction(1), 3: Fraction(-3)}
MSSM_delta_b = {1: Fraction(5, 2), 2: Fraction(25, 6), 3: Fraction(4)}

# C_2(adj_i) — Casimir of adjoint rep
C2_adj = {1: Fraction(0), 2: Fraction(2), 3: Fraction(3)}


# ---------------------------------------------------------------------------
# Decompose H_F into Fock subspace and complement
# ---------------------------------------------------------------------------

def build_fock_subspace_per_vertex():
    """Per-vertex Cl(6) Fock = ℂ^8 sitting inside M_8 = ℂ^{64} as the
    first column (or diagonal-state-vector embedding).

    Convention: the 64-dim M_8 = ℂ^8 ⊗ ℂ^8* in col-major flatten.
    The "Fock state" subspace is the subspace where the bra is fixed
    (e.g., ⟨0|).  Concretely: |i⟩⟨0| for i = 0..7.  These are 8 specific
    flat vectors in ℂ^{64}.

    Alternative: |i⟩⟨i| (number-operator-like).  Or diagonal states.

    For simplicity, use the "first column" convention: state-bras-fixed-at-|0⟩.
    These span an 8-dim subspace of M_8.

    Returns: (8, 64) matrix whose columns are basis vectors of the Fock subspace.
    """
    Q = np.zeros((64, 8), dtype=complex)
    for i in range(8):
        # |i⟩⟨0| has 1 in entry (i, 0) and 0 elsewhere
        # Col-major flatten of M_8: index = col*8 + row = 0*8 + i = i
        Q[i, i] = 1.0
    return Q


def build_fock_subspace_HF(dim0, dim1):
    """Build the per-cell Fock subspace inside H_F = 280.

    32 fermion states per cell = 4 vertices × 8 Fock-per-vertex.
    Embed each vertex's 8 Fock states into the corresponding 64-block of H_F.
    """
    dim_tot = dim0 + dim1
    Q_per_vertex = build_fock_subspace_per_vertex()  # 64×8
    Q = np.zeros((dim_tot, 32), dtype=complex)
    for v in range(NV):
        Q[v*64:(v+1)*64, v*8:(v+1)*8] = Q_per_vertex
    # Edge sector contributes 0 (Fock states are vertex-only)
    return Q


# ---------------------------------------------------------------------------
# Compute substrate-direct b_i
# ---------------------------------------------------------------------------

def compute_T_on_subspace(T_lift, Q):
    """Compute (1/dim(adj)) Σ_a Tr(T^a T^a) restricted to subspace spanned
    by columns of Q (orthonormal).  For a single generator T (no adjoint
    sum), returns Tr(T^2 |_Q)."""
    # P_Q = Q Q† is the projector
    # T restricted to Q: Q† T Q (an 8x8 if Q is 280x8)
    T_proj = Q.conj().T @ T_lift @ Q
    return np.trace(T_proj @ T_proj).real


def main():
    print(r"""
==========================================================================================
P-D1 session 2 — Substrate-direct β-function from H_F
==========================================================================================""")
    print(f"\n  Reference values:")
    print(f"    SM   b = {SM_b[1]}, {SM_b[2]}, {SM_b[3]}")
    print(f"    MSSM b = {MSSM_b[1]}, {MSSM_b[2]}, {MSSM_b[3]}")
    print(f"    Δb   = b_MSSM - b_SM = {MSSM_delta_b[1]}, {MSSM_delta_b[2]}, {MSSM_delta_b[3]}")

    # Build framework data
    D_F, dim0, dim1 = build_D_F()
    print(f"\n  H_F dim = {dim0 + dim1} = {dim0} (matter) + {dim1} (gauge)")

    # Build SM gauge generators on H_F
    su2L_8 = build_su2L_8()
    su2L_lift = [lift_to_HF_adjoint(T, dim0, dim1) for T in su2L_8]

    su3_8 = build_su3_8()
    su3_lift = [lift_to_HF_adjoint(T, dim0, dim1) for T in su3_8]

    Y_8 = build_u1Y_8()
    Y_lift = lift_to_HF_adjoint(Y_8, dim0, dim1)

    # Build Fock subspace (32-dim, per-cell fermion states)
    Q_fock = build_fock_subspace_HF(dim0, dim1)
    Q_fock, _ = np.linalg.qr(Q_fock)
    print(f"  Fock subspace dim = {Q_fock.shape[1]} (= 32 per cell, fermion states)")

    # ---- T_i over full H_F ----
    print("\n" + "=" * 100)
    print("PART A — T_i(F) over full H_F vs over Fock-only subspace (32)")
    print("=" * 100)

    # Compute T_2 over full H_F and over Fock-only
    T2_full = trace_dynkin(su2L_lift, dim_adj=3)
    T2_fock = sum(compute_T_on_subspace(T, Q_fock) for T in su2L_lift) / 3
    print(f"\n  SU(2)_L:")
    print(f"    T_2 over full H_F (280): {T2_full:.4f}")
    print(f"    T_2 over Fock only (32): {T2_fock:.4f}")
    print(f"    T_2 over bilinears+gauge (248): {T2_full - T2_fock:.4f}")

    T3_full = trace_dynkin(su3_lift, dim_adj=8)
    T3_fock = sum(compute_T_on_subspace(T, Q_fock) for T in su3_lift) / 8
    print(f"\n  SU(3)_c:")
    print(f"    T_3 over full H_F (280): {T3_full:.4f}")
    print(f"    T_3 over Fock only (32): {T3_fock:.4f}")
    print(f"    T_3 over bilinears+gauge (248): {T3_full - T3_fock:.4f}")

    T1_full = trace_dynkin_u1(Y_lift)
    T1_fock = compute_T_on_subspace(Y_lift, Q_fock)
    print(f"\n  U(1)_Y (raw Σ Y², no GUT factor):")
    print(f"    T_1 over full H_F (280): {T1_full:.4f}")
    print(f"    T_1 over Fock only (32): {T1_fock:.4f}")
    print(f"    T_1 over bilinears+gauge (248): {T1_full - T1_fock:.4f}")

    # ---- Compute candidate b_i ----
    print("\n" + "=" * 100)
    print("PART B — Candidate b_i interpretations")
    print("=" * 100)

    def b_i_interpretation(name, T1, T2, T3, F_type):
        """Compute candidate b_i assuming F_type = 'fermion' (Weyl) or 'scalar' (complex)."""
        # GUT-normalised T_1
        T1_gut = T1 * 3 / 5
        if F_type == 'fermion':
            factor = Fraction(2, 3)
        elif F_type == 'scalar':
            factor = Fraction(1, 3)
        # Fraction conversion for clarity (approximate)
        b1 = -C2_adj[1]*Fraction(11,3) + float(factor) * T1_gut
        b2 = -C2_adj[2]*Fraction(11,3) + float(factor) * T2
        b3 = -C2_adj[3]*Fraction(11,3) + float(factor) * T3
        print(f"\n  {name} (treating as {F_type}):")
        print(f"    T_1 (GUT) = {T1_gut:.4f},  T_2 = {T2:.4f},  T_3 = {T3:.4f}")
        print(f"    b_1 = -{C2_adj[1]}·11/3 + ({factor})·{T1_gut:.4f} = {b1:.4f}  (SM={float(SM_b[1]):.3f}, MSSM={float(MSSM_b[1]):.3f})")
        print(f"    b_2 = -{C2_adj[2]}·11/3 + ({factor})·{T2:.4f} = {b2:.4f}  (SM={float(SM_b[2]):.3f}, MSSM={float(MSSM_b[2]):.3f})")
        print(f"    b_3 = -{C2_adj[3]}·11/3 + ({factor})·{T3:.4f} = {b3:.4f}  (SM={float(SM_b[3]):.3f}, MSSM={float(MSSM_b[3]):.3f})")

    # Interpretation 1: Fock as Weyl fermions, rest as scalars
    print("\n  ---- INTERPRETATION 1: Fock (32) as Weyl fermions; rest (248) ignored ----")
    b_i_interpretation("(Fock fermions only)", T1_fock, T2_fock, T3_fock, 'fermion')

    print("\n  ---- INTERPRETATION 2: Fock (32) as fermions + bilinears+gauge (248) as scalars ----")
    print("\n    Fermion contribution from Fock:")
    b_i_interpretation("Fock fermion part", T1_fock, T2_fock, T3_fock, 'fermion')
    print("\n    PLUS scalar contribution from bilinears (NOT including gauge -11/3 again):")
    T1_S = T1_full - T1_fock
    T2_S = T2_full - T2_fock
    T3_S = T3_full - T3_fock
    T1_S_gut = T1_S * 3 / 5
    print(f"    Scalar T's: T_1_S (GUT) = {T1_S_gut:.4f},  T_2_S = {T2_S:.4f},  T_3_S = {T3_S:.4f}")
    # Combined (single -11/3 from gauge bosons, plus fermion (2/3) plus scalar (1/3))
    b1_total = -float(C2_adj[1])*11/3 + (2/3)*T1_fock*3/5 + (1/3)*T1_S_gut
    b2_total = -float(C2_adj[2])*11/3 + (2/3)*T2_fock + (1/3)*T2_S
    b3_total = -float(C2_adj[3])*11/3 + (2/3)*T3_fock + (1/3)*T3_S
    print(f"\n    Combined (Fock-fermion + bilinear-scalar):")
    print(f"      b_1 = {b1_total:.4f}  (SM={float(SM_b[1]):.3f}, MSSM={float(MSSM_b[1]):.3f}, Δ_SM={b1_total - float(SM_b[1]):.3f}, Δ_MSSM={b1_total - float(MSSM_b[1]):.3f})")
    print(f"      b_2 = {b2_total:.4f}  (SM={float(SM_b[2]):.3f}, MSSM={float(MSSM_b[2]):.3f}, Δ_SM={b2_total - float(SM_b[2]):.3f}, Δ_MSSM={b2_total - float(MSSM_b[2]):.3f})")
    print(f"      b_3 = {b3_total:.4f}  (SM={float(SM_b[3]):.3f}, MSSM={float(MSSM_b[3]):.3f}, Δ_SM={b3_total - float(SM_b[3]):.3f}, Δ_MSSM={b3_total - float(MSSM_b[3]):.3f})")

    print("\n  ---- INTERPRETATION 3: Full H_F (280) as Weyl fermions ----")
    b_i_interpretation("Full H_F as Weyl", T1_full, T2_full, T3_full, 'fermion')

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    print(r"""
  Substrate-direct β depends crucially on the FERMION/SCALAR assignment of H_F.

  Three interpretations give wildly different b_i values:
    1. Only Fock (32) as fermions: too small (Fock is 32, way less than SM 48-Weyl-effective).
    2. Fock + bilinears as fermions+scalars: depends on splitting; not standard.
    3. Full H_F as Weyl: wildly too big (~85 for b_2 vs MSSM 1).

  The FRAMEWORK does NOT provide a canonical fermion/scalar split of H_F.
  Path E showed Cl(6) Fock is purely fermionic (all 32 vertex states are
  Weyls); Z probe showed order-1 CC inner-fluctuation reduction fails.

  Conclusion: substrate-direct β can NOT be cleanly extracted from H_F
  alone.  An additional structural ingredient is needed to identify
  the scalar/fermion content correctly.

  This is the same gap that's been blocking ADOPTED-MSSM-Sb closure.
  P-D1 session 2 confirms: substrate-direct β computation requires
  resolving the fermion/scalar split — which is precisely the missing
  ingredient.""")
    print("\nP_D1_session2_substrate_direct_beta.py: sentinel done.")


if __name__ == "__main__":
    main()
