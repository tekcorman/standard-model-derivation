#!/usr/bin/env python3
"""
Z_CC_reductions_omega_D1_phase2_probe.py
========================================

Phase 2 of the refined Z probe.  Phase 1 (Z_CC_reductions_omega_D1_probe.py)
found that Hermitian + J-real reductions of the framework's raw 1-form module
Ω_D¹ = 1536 give:

  J^α (entrywise conj):       1536 → 768 real DOF
  J^β (Hermitian adjoint):    1536 → 1344 real DOF  ← canonical GNS modular conj

Both are much larger than MSSM Higgs + sfermion + sneutrino bosonic content
per cell (~98–104 real DOF).  This Phase 2 probe asks:

  Q1.  Does the framework satisfy the CC ORDER-1 condition
       [[D_F, π(a)], JπbJ⁻¹] = 0  ∀ a, b ∈ A_F ?
       If YES: Hermitian + J-real is the full CC reduction (1344 real for J^β).
       If NO: there is a structural obstruction, and the standard CC scalar
       sector doesn't apply to this spectral triple as-is.

  Q2.  How does the J-real Hermitian Ω_D¹ decompose under the SM gauge
       subgroups SU(2)_L × SU(2)_R × U(1)_Y that ARE consistent on Cl(6) Fock
       (skipping SU(3) since it doesn't commute with SU(2)_L per M2)?

  Q3.  What β-function contribution Δb_i does that decomposition give, and
       does it match MSSM Δb = (+5/2, +25/6, +4) (the extras needed to flow
       from α_GUT⁻¹=24 + sin²θ_W=3/8 down to PDG at M_Z)?

This is the test of whether the framework's spectral-triple inner-fluctuation
content REPRODUCES the MSSM β-coefficient deltas — which would graduate
ADOPTED-MSSM-Sb structurally.

No graded content changes from this probe.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from proofs.foundations.de_rham_susy_fibered_v2_probe import (  # noqa: E402
    d_alg, NV, NE,
)
from proofs.foundations.M1_J_real_structure_probe import build_J  # noqa: E402
from proofs.foundations.M2_SM_gauge_embedding_probe import (  # noqa: E402
    build_gamma, biv,
)

np.set_printoptions(precision=4, suppress=True, linewidth=140)
TOL = 1e-9


def build_D_F():
    d = d_alg((0.0, 0.0, 0.0))
    dim0, dim1 = NV * 64, NE * 4
    D_F = np.zeros((dim0 + dim1, dim0 + dim1), dtype=complex)
    D_F[:dim0, dim0:] = d.conj().T
    D_F[dim0:, :dim0] = d
    return D_F, dim0, dim1


# ---------------------------------------------------------------------------
# Part A — Order-1 condition check
# ---------------------------------------------------------------------------

def part_A_order_1():
    """Test [[D_F, L_a], R_c] = 0 for a sample of (a, c) ∈ A_F × A_F.

    For J = J^β (Hermitian adjoint), the modular conjugation maps L_b → R_{b†}.
    So order-1 [[D_F, L_a], JL_bJ⁻¹] = [[D_F, L_a], R_{b†}].  Renaming c = b†
    (which ranges over all of A_F since A_F is *-closed): order-1 reduces to
    [[D_F, L_a], R_c] = 0 for all a, c ∈ A_F.
    """
    print("=" * 100)
    print("PART A — Order-1 condition [[D_F, L_a], R_c] = 0 for all a, c ∈ A_F")
    print("=" * 100)
    D_F, dim0, dim1 = build_D_F()
    dim_tot = dim0 + dim1

    # Build sample of L_a and R_c operators.
    # L_a (left mult by a on H_F): for a in vertex M_8(v) elementary E^{ij}:
    #   L_a (X) maps each block of X via left-mult-by-a in vertex-v block of M_8 flatten.
    #   In matrix form: L_a = a_full ⊗ I on H_F if we use the right convention,
    #   but H_F is itself End(H_F)?  Let me reframe.
    #
    # H_F = A_F vectorized via HS-inner-product.  L_a: H_F → H_F is left mult by a in A_F.
    # In flatten basis of A_F (= M_8 blocks + M_2 blocks, each col-major flattened),
    # L_a for a ∈ M_8(v) is: (np.kron(I_8, a) on the 64-dim block of vertex v;
    #                         trivial on other blocks).
    # R_c for c ∈ M_8(v) is: (np.kron(c.T, I_8) on the 64-dim block of vertex v;
    #                         trivial on other blocks).
    # Similarly for edges.
    #
    # CRUCIAL: L_a and R_c always commute (left and right mult on an algebra).
    # So [L_a, R_c] = 0 ALWAYS.
    # Order-1 asks whether [D_F, L_a] commutes with R_c.

    def build_L(M_block, sector, idx):
        op = np.zeros((dim_tot, dim_tot), dtype=complex)
        if sector == 'vertex':
            op[idx*64:(idx+1)*64, idx*64:(idx+1)*64] = np.kron(np.eye(8, dtype=complex), M_block)
        else:
            op[dim0+idx*4:dim0+(idx+1)*4, dim0+idx*4:dim0+(idx+1)*4] = np.kron(np.eye(2, dtype=complex), M_block)
        return op

    def build_R(M_block, sector, idx):
        op = np.zeros((dim_tot, dim_tot), dtype=complex)
        if sector == 'vertex':
            op[idx*64:(idx+1)*64, idx*64:(idx+1)*64] = np.kron(M_block.T, np.eye(8, dtype=complex))
        else:
            op[dim0+idx*4:dim0+(idx+1)*4, dim0+idx*4:dim0+(idx+1)*4] = np.kron(M_block.T, np.eye(2, dtype=complex))
        return op

    # Build sample: 4 vertices × 4 randomly-chosen M_8 elementary + 6 edges × 1 elementary
    rng = np.random.default_rng(42)
    sample_a = []
    sample_c = []
    sector_idx_pairs = []
    for v in range(NV):
        for _ in range(2):
            i, j = rng.integers(0, 8), rng.integers(0, 8)
            E = np.zeros((8, 8), dtype=complex); E[i, j] = 1
            sample_a.append((build_L(E, 'vertex', v), f'L_V{v}E_{i}{j}'))
            sample_c.append((build_R(E, 'vertex', v), f'R_V{v}E_{i}{j}'))
            sector_idx_pairs.append(('vertex', v, i, j))
    for e in range(NE):
        i, j = rng.integers(0, 2), rng.integers(0, 2)
        E = np.zeros((2, 2), dtype=complex); E[i, j] = 1
        sample_a.append((build_L(E, 'edge', e), f'L_E{e}E_{i}{j}'))
        sample_c.append((build_R(E, 'edge', e), f'R_E{e}E_{i}{j}'))
        sector_idx_pairs.append(('edge', e, i, j))

    # Test [[D_F, L_a], R_c] = 0 for all sample (a, c) pairs.
    print(f"\n  Sample size: {len(sample_a)} L_a × {len(sample_c)} R_c = {len(sample_a)*len(sample_c)} pairs")
    max_norm = 0.0
    n_zero = 0
    n_nonzero = 0
    for L_a, name_a in sample_a:
        comm_DLa = D_F @ L_a - L_a @ D_F
        for R_c, name_c in sample_c:
            order1_op = comm_DLa @ R_c - R_c @ comm_DLa
            nrm = np.linalg.norm(order1_op)
            if nrm < TOL:
                n_zero += 1
            else:
                n_nonzero += 1
                if nrm > max_norm:
                    max_norm = nrm
    print(f"\n  Pairs with ‖[[D_F, L_a], R_c]‖ < TOL: {n_zero}")
    print(f"  Pairs with ‖[[D_F, L_a], R_c]‖ > TOL: {n_nonzero}")
    print(f"  Max norm of order-1 commutator: {max_norm:.3e}")

    if n_nonzero == 0:
        print(f"\n  ⇒ ORDER-1 CONDITION HOLDS (on this sample, machine precision).")
        print(f"    Standard CC reductions apply.  Hermitian + J-real ≈ physical scalar count.")
        return True
    else:
        print(f"\n  ⇒ ORDER-1 CONDITION FAILS ({n_nonzero} of {n_zero+n_nonzero} pairs nonzero).")
        print(f"    Framework's spectral triple violates the CC order-1 axiom.")
        print(f"    Implication: the standard CC reduction Ω_D¹ → physical scalars does not directly apply.")
        return False


# ---------------------------------------------------------------------------
# Part B — diagnostic on the order-1 violation structure
# ---------------------------------------------------------------------------

def part_B_diagnostic_violation():
    """If order-1 fails, characterise the failure structurally."""
    print("\n" + "=" * 100)
    print("PART B — diagnostic on order-1 violation")
    print("=" * 100)
    D_F, dim0, dim1 = build_D_F()
    dim_tot = dim0 + dim1

    # Specifically: order-1 says D acts as a 1st-order differential operator
    # w.r.t. A_F.  For the framework, D_F = Q̂_alg = fibered de Rham
    # supercharge built from partial trace.  Partial trace is NOT a derivation.
    # Specifically: tr_⊥(ab) ≠ tr_⊥(a) tr_⊥(b) in general.
    # So [D_F, L_a] doesn't factor as a left mult of [D_F a derived element],
    # and the order-1 axiom is structurally violated.

    # Let me give one explicit example with simple a, c and report the norm.
    # Take a = E^{00} on vertex 0 (left-mult); c = E^{00} on vertex 0 (right-mult).
    E = np.zeros((8, 8), dtype=complex); E[0, 0] = 1
    L = np.zeros((dim_tot, dim_tot), dtype=complex)
    L[:64, :64] = np.kron(np.eye(8, dtype=complex), E)  # L_E on vertex 0
    R = np.zeros((dim_tot, dim_tot), dtype=complex)
    R[:64, :64] = np.kron(E.T, np.eye(8, dtype=complex))  # R_E on vertex 0
    LR_commute = np.linalg.norm(L @ R - R @ L)
    print(f"\n  Sanity check: ‖[L_a, R_c]‖ for vertex-internal mults = {LR_commute:.3e}  (should be 0)")
    DL = D_F @ L - L @ D_F
    DR = D_F @ R - R @ D_F
    order1 = DL @ R - R @ DL
    print(f"  ‖[D_F, L]‖    = {np.linalg.norm(DL):.3e}")
    print(f"  ‖[D_F, R]‖    = {np.linalg.norm(DR):.3e}  (should equal ‖[D_F, L]‖ by L↔R sym)")
    print(f"  ‖[[D_F, L], R]‖ = {np.linalg.norm(order1):.3e}  ← order-1 obstruction")

    # Maybe useful: rank of order1 operator
    sv = np.linalg.svd(order1, compute_uv=False)
    rank_order1 = int(np.sum(sv > TOL))
    print(f"  rank of [[D_F, L], R] = {rank_order1}")

    print(r"""
  Structural reason: D_F = Q̂_alg is built from partial trace tr_⊥: M_8 → M_2.
  Partial trace is NOT a derivation: tr_⊥(ab) ≠ tr_⊥(a) tr_⊥(b) in general.
  So [D_F, L_a] doesn't act "to first order" in a (it has higher-order parts
  in a that the order-1 axiom forbids).""")


# ---------------------------------------------------------------------------
def main():
    print(r"""
==========================================================================================
Z Phase 2 — Order-1 check + structural diagnosis
==========================================================================================""")
    order1_holds = part_A_order_1()
    if not order1_holds:
        part_B_diagnostic_violation()
    print("\n" + "=" * 100)
    print("PHASE 2 verdict")
    print("=" * 100)
    if order1_holds:
        print("""
  Order-1 condition holds.  Hermitian + J-real reduction (Phase 1) gives the
  CC physical scalar count: J^β → 1344 real DOF.  Phase 2 next step: decompose
  under SU(2)_L × SU(2)_R × U(1)_Y; compute β contribution; compare to MSSM Δb.""")
    else:
        print("""
  Order-1 condition FAILS for this spectral triple.

  Implication: the standard CC Ω_D¹ → physical scalar reduction does NOT
  directly apply.  The framework's D_F (built from non-derivation partial-trace
  tr_⊥: M_8 → M_2) is not a 1st-order differential operator in the A_F
  algebra structure.

  This is consistent with M-arc enumeration findings (C2 fails for all SM
  gauge tuples).  Both stem from the same structural fact:  D_F is not a
  standard CC derivation-type Dirac operator.

  Strategic re-orientation needed:
   - The framework's matter and gauge content are theorem-grade per their
     existing derivations (B3 + C_3 + Cl(0,2) + walks).
   - Bosonic content (Higgs derived; W mass derived; gauge bosons via edge
     sector 12=8+3+1) is theorem-grade for the EXISTING bosons.
   - MSSM β-coefficient contribution (+5/2, +25/6, +4) needs a DIFFERENT
     mechanism than CC inner-fluctuation reduction.
   - Possible route: substrate's actual β contribution comes from operator
     spectrum or walk-density mechanisms, not Ω_D¹ scalar count.""")
    print("\nZ_CC_reductions_omega_D1_phase2_probe.py: sentinel done.")


if __name__ == "__main__":
    main()
