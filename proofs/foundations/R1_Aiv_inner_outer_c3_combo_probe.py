#!/usr/bin/env python3
"""
R1_Aiv_inner_outer_c3_combo_probe.py
=====================================
A.iv slow-path probe — the inner × outer C_3 combination.

Per the framework's B3-B6 reconciliation + M1.B Galois closure
(`theorem_41_screw_wigner.md` §6), the body-diagonal C_3 acts in TWO
DISTINCT ways:

  - INNER  (B6):  Spin(6) lift U_{C_3}^S on each vertex's Cl(6) Fock,
                  inducing color-Z_3.
  - OUTER  (M1.B):  permutation of the 4 K_4 vertices (v_0 fixed,
                    v_1→v_3, v_2→v_1, v_3→v_2) as an order-3 outer
                    automorphism α of the operator algebra,
                    inducing generation-Z_3.

These are at DIFFERENT structural levels:
  inner acts on STATES at each vertex (intra-vertex action)
  outer acts on OPERATORS by permuting blocks (inter-vertex action)

A.iv tests whether their COMBINATION on H_F has a Z_2 sub-structure
relevant to the boson/fermion split that gates b_i derivation.

Specifically, decompose H_F under the joint Z_3 × Z_3 = Z_3_inner × Z_3_outer
action.  9 sectors emerge.  Look for Z_2 patterns among them.

What this probe does
--------------------
A — Reuse U_{C_3}^S from `theorem_B3_B6_reconciliation.py` (inner C_3 lift on ℂ^8).
B — Build inner C_3 on H_F: per-vertex block-diagonal adjoint by U_{C_3}^S.
C — Build outer C_3 on H_F: vertex permutation (block-permute the M_8 blocks
    per R1.2's body-diagonal σ).
D — Verify the inner and outer C_3 actions commute (or compute commutator).
E — Decompose H_F under joint eigenvalues of (inner, outer):
    9 sectors {(inner=1,ω,ω²) × (outer=1,ω,ω²)}.  Tabulate dim of each.
F — Test Z_2 candidates among the 9 sectors:
    e.g., is there a parity (i, j) → sign such that sectors split into
    two equal-dim families?

No graded content changes.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm, logm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from proofs.foundations.de_rham_susy_fibered_v2_probe import (  # noqa: E402
    d_alg, NE, NV, SX, SY, SZ, I2,
)

np.set_printoptions(precision=4, suppress=True, linewidth=140)
TOL = 1e-9
omega3 = np.exp(2j * np.pi / 3)


# -----------------------------------------------------------------------------
# Cl(6) on ℂ^8 (Brauer-Weyl per B3)
# -----------------------------------------------------------------------------

def build_gamma_cl6():
    Gamma = [None] * 7
    Gamma[1] = np.kron(np.kron(SX, I2), I2)
    Gamma[2] = np.kron(np.kron(SY, I2), I2)
    Gamma[3] = np.kron(np.kron(SZ, SX), I2)
    Gamma[4] = np.kron(np.kron(SZ, SY), I2)
    Gamma[5] = np.kron(np.kron(SZ, SZ), SX)
    Gamma[6] = np.kron(np.kron(SZ, SZ), SY)
    return Gamma


def biv(G, a, b):
    return 0.5 * (G[a] @ G[b] - G[b] @ G[a])


# -----------------------------------------------------------------------------
# Part A — build U_{C_3}^S (inner C_3 lift) per framework's B3-B6 reconciliation
# -----------------------------------------------------------------------------

def part_A_inner_c3():
    print("=" * 100)
    print("PART A — U_{C_3}^S (inner C_3 = Spin(6) lift on Cl(6) Fock ≅ ℂ^8)")
    print("=" * 100)
    Gamma = build_gamma_cl6()
    # K_4 edges in B6 ordering
    K4_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    SIGMA = {0: 0, 1: 3, 2: 1, 3: 2}
    def apply_sigma_to_edge(e):
        a, b = e
        return tuple(sorted((SIGMA[a], SIGMA[b])))
    edge_to_idx = {e: i for i, e in enumerate(K4_EDGES)}
    P_so6 = np.zeros((6, 6), dtype=float)
    for e in K4_EDGES:
        i = edge_to_idx[e]
        j = edge_to_idx[apply_sigma_to_edge(e)]
        P_so6[j, i] = 1.0

    L_so6 = logm(P_so6).real
    L_so6 = 0.5 * (L_so6 - L_so6.T)
    X_spin = np.zeros((8, 8), dtype=complex)
    for i in range(6):
        for j in range(i + 1, 6):
            X_spin += L_so6[i, j] * biv(Gamma, i + 1, j + 1)
    X_spin_half = 0.5 * X_spin
    U_inner = expm(X_spin_half)

    # Resolve sign
    U3 = U_inner @ U_inner @ U_inner
    I8 = np.eye(8, dtype=complex)
    if np.allclose(U3, -I8, atol=1e-9):
        U_inner = np.exp(1j * np.pi / 3) * U_inner
        U3 = U_inner @ U_inner @ U_inner

    assert np.allclose(U3, I8, atol=1e-9), f"||U³ − I|| = {np.linalg.norm(U3 - I8)}"
    print(f"\n  U_{{C_3}}^S ∈ SU(8) (Spin(6) ≅ SU(4) chiral spinor rep):  built")
    print(f"  U^3 = I  :  {np.allclose(U_inner @ U_inner @ U_inner, I8, atol=TOL)}")
    eigs = np.linalg.eigvals(U_inner)
    print(f"  eigenvalues of U_inner (sorted by phase): {sorted(np.angle(eigs))}")
    # Should be {0, 0, 2π/3, 2π/3, 0, 0, -2π/3, -2π/3} for the (4,4̄) of SU(4) with eigvals (1,1,ω,ω²) ⊕ (1,1,ω²,ω)
    return U_inner


# -----------------------------------------------------------------------------
# Part B — inner C_3 lifted to A_F as per-vertex adjoint
# -----------------------------------------------------------------------------

def adjoint_on_C0(J8, vertex):
    """ad action of 8x8 unitary J on vertex's M_8 block in C⁰_alg, lifted to 280×280.
    For X ∈ M_8 (col-major flatten): ad_J(X) = J X J† = (J ⊗ J^*) flatten(X).
    Wait — adjoint action by UNITARY: X ↦ U X U†.  Flatten: (U^* ⊗ U) flatten(X).

    Caveat: for Hermitian generator a (Lie alg level), ad_a(X) = aX − Xa,
    but for GROUP ELEMENT U, the action is Ad_U(X) = U X U†, NOT a commutator.
    """
    dim0, dim1 = NV * 64, NE * 4
    op = np.zeros((dim0 + dim1, dim0 + dim1), dtype=complex)
    # Action on M_8 = col-major flatten ℂ^64:  Ad_U(X) = U X U†
    # In col-major flatten:  vec(U X U†) = (U^* ⊗ U) · vec(X)
    block = np.kron(J8.conj(), J8)   # 64×64
    op[vertex * 64:(vertex + 1) * 64, vertex * 64:(vertex + 1) * 64] = block
    return op


def part_B_inner_c3_on_AF(U_inner):
    print("\n" + "=" * 100)
    print("PART B — Inner C_3 lifted to A_F (per-vertex adjoint by U_{C_3}^S)")
    print("=" * 100)
    # Per-vertex inner C_3 = Ad_{U_inner} on each M_8 block
    # All vertices get the SAME U_inner action (since each vertex has same Cl(6) structure)
    inner_C3_AF = np.zeros((280, 280), dtype=complex)
    for v in range(NV):
        inner_C3_AF += adjoint_on_C0(U_inner, v)
    # The edge sector (C¹_alg) is INVARIANT under inner C_3 (no Spin(6) action there)
    # → add identity on the edge block
    for e in range(NE):
        inner_C3_AF[NV*64 + e*4:NV*64 + (e+1)*4, NV*64 + e*4:NV*64 + (e+1)*4] = np.eye(4, dtype=complex)
    # Verify (inner_C3_AF)^3 = I
    cube = inner_C3_AF @ inner_C3_AF @ inner_C3_AF
    err = np.linalg.norm(cube - np.eye(280, dtype=complex))
    print(f"\n  ‖inner_C3_AF^3 − I_{{280}}‖ = {err:.3e}")
    assert err < 1e-9, "inner C_3 lifted to A_F should have order 3"
    # Eigvalues
    eigs = np.linalg.eigvals(inner_C3_AF)
    n_t = sum(1 for e in eigs if abs(e - 1) < TOL)
    n_w = sum(1 for e in eigs if abs(e - omega3) < TOL)
    n_w2 = sum(1 for e in eigs if abs(e - omega3 ** 2) < TOL)
    print(f"  eigenvalue counts:  trivial = {n_t}, ω = {n_w}, ω² = {n_w2}  (total {n_t + n_w + n_w2})")
    return inner_C3_AF


# -----------------------------------------------------------------------------
# Part C — outer C_3 on A_F (vertex permutation, from R1.2)
# -----------------------------------------------------------------------------

def part_C_outer_c3_on_AF():
    print("\n" + "=" * 100)
    print("PART C — Outer C_3 on A_F (vertex permutation; from R1.2)")
    print("=" * 100)
    # σ: v_0→v_0, v_1→v_3, v_2→v_1, v_3→v_2
    SIGMA = [0, 3, 1, 2]
    outer_C3_AF = np.zeros((280, 280), dtype=complex)
    I_64 = np.eye(64, dtype=complex)
    # vertex blocks
    for j in range(4):
        i = SIGMA[j]
        outer_C3_AF[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64] = I_64

    # Edge sector also permutes — recall R1.3's edge permutation
    from proofs.foundations.de_rham_susy_fibered_v2_probe import EDGES
    def find_edge(a, b):
        a, b = (min(a, b), max(a, b))
        for k, (u, v, _) in enumerate(EDGES):
            if (min(u, v), max(u, v)) == (a, b):
                return k
        raise ValueError(f"edge ({a}, {b}) not found")
    edge_perm = []
    for i, (u, v, _) in enumerate(EDGES):
        new_u = SIGMA[u]; new_v = SIGMA[v]
        edge_perm.append(find_edge(new_u, new_v))
    I_4 = np.eye(4, dtype=complex)
    dim0 = NV * 64
    for j in range(NE):
        i = edge_perm[j]
        outer_C3_AF[dim0 + i * 4:dim0 + (i + 1) * 4, dim0 + j * 4:dim0 + (j + 1) * 4] = I_4

    cube = outer_C3_AF @ outer_C3_AF @ outer_C3_AF
    err = np.linalg.norm(cube - np.eye(280, dtype=complex))
    print(f"\n  ‖outer_C3_AF^3 − I_{{280}}‖ = {err:.3e}")
    assert err < 1e-9, "outer C_3 on A_F should have order 3"
    eigs = np.linalg.eigvals(outer_C3_AF)
    n_t = sum(1 for e in eigs if abs(e - 1) < TOL)
    n_w = sum(1 for e in eigs if abs(e - omega3) < TOL)
    n_w2 = sum(1 for e in eigs if abs(e - omega3 ** 2) < TOL)
    print(f"  eigenvalue counts:  trivial = {n_t}, ω = {n_w}, ω² = {n_w2}")
    return outer_C3_AF


# -----------------------------------------------------------------------------
# Part D — commutation check
# -----------------------------------------------------------------------------

def part_D_commutation(inner_C3, outer_C3):
    print("\n" + "=" * 100)
    print("PART D — do inner and outer C_3 commute on H_F?")
    print("=" * 100)
    comm = inner_C3 @ outer_C3 - outer_C3 @ inner_C3
    nrm = np.linalg.norm(comm)
    print(f"\n  ‖[inner, outer]‖ = {nrm:.3e}")
    if nrm < TOL:
        print(f"  ⇒ inner and outer C_3 COMMUTE  →  Z_3 × Z_3 abelian joint action")
    else:
        print(f"  ⇒ inner and outer C_3 DO NOT commute  →  Z_3 ⋊ Z_3 joint structure")
    return nrm < TOL


# -----------------------------------------------------------------------------
# Part E — joint Z_3 × Z_3 decomposition
# -----------------------------------------------------------------------------

def part_E_joint_decomposition(inner_C3, outer_C3):
    print("\n" + "=" * 100)
    print("PART E — joint Z_3 × Z_3 eigenvalue decomposition of H_F")
    print("=" * 100)
    # Diagonalize joint action.  Since inner and outer commute (verified D), they're
    # simultaneously diagonalizable.  Use simultaneous eigenvalue assignment.
    # In practice: get eigenvectors of inner, then within each inner eigenspace,
    # diagonalize outer.
    # Computational shortcut: just compute eigenvalues of both on the same vector by
    # using ω-combos.
    # Use: combined = inner + 5 * outer (some non-degenerate combination), diagonalize once.
    combined = inner_C3 + np.exp(1j * 0.7) * outer_C3
    eig_c, vecs = np.linalg.eig(combined)
    # For each eigenvector, evaluate inner and outer eigenvalues:
    print(f"\n  Tabulating joint (inner, outer) sector dims:")
    print(f"    {'inner':>8} | {'outer':>8} | count")
    print("  " + "-" * 35)
    inner_eigs = []
    outer_eigs = []
    for k in range(280):
        v = vecs[:, k]
        inner_val = (v.conj() @ inner_C3 @ v) / (v.conj() @ v)
        outer_val = (v.conj() @ outer_C3 @ v) / (v.conj() @ v)
        # Bin to {1, ω, ω²}
        def bin_omega(z):
            phases = {0: 1, 1: omega3, -1: omega3 ** 2}
            best = min(phases.values(), key=lambda p: abs(z - p))
            return best
        inner_eigs.append(bin_omega(inner_val))
        outer_eigs.append(bin_omega(outer_val))

    # Count joint sectors
    counts = {}
    for i_e, o_e in zip(inner_eigs, outer_eigs):
        # Round complex eigenvalues to {1, ω, ω²}
        def label(e):
            if abs(e - 1) < 0.1: return '1'
            if abs(e - omega3) < 0.1: return 'ω'
            return 'ω²'
        key = (label(i_e), label(o_e))
        counts[key] = counts.get(key, 0) + 1
    for inner_lab in ['1', 'ω', 'ω²']:
        for outer_lab in ['1', 'ω', 'ω²']:
            c = counts.get((inner_lab, outer_lab), 0)
            print(f"    {inner_lab:>8} | {outer_lab:>8} | {c}")
    return counts


# -----------------------------------------------------------------------------
# Part F — Z_2 indicator search
# -----------------------------------------------------------------------------

def part_F_z2_search(counts):
    print("\n" + "=" * 100)
    print("PART F — searching for Z_2 sub-structure in the 9 joint sectors")
    print("=" * 100)
    print(f"\n  9 joint sectors with dims:")
    print(f"      |  outer=1 |  outer=ω | outer=ω²")
    print(f"  ------------------------------------")
    for i_lab in ['1', 'ω', 'ω²']:
        row = "  "
        row += f"{i_lab:>3} | "
        for o_lab in ['1', 'ω', 'ω²']:
            row += f"{counts.get((i_lab, o_lab), 0):>8} | "
        print(row)

    # Candidate Z_2 indicators:
    # (i)  parity = (inner_idx + outer_idx) mod 2, where idx is 0,1,2 for 1,ω,ω²
    #      → only works if 3 not a divisor issue;  parity here is from 0+0=0 even, 1+0=1 odd, etc.
    # (ii) diagonal vs off-diagonal:  inner=outer (== diagonal Z_3 subgroup) vs not.
    # (iii) trivial × trivial (= 1, 1) vs everything else.

    idx = {'1': 0, 'ω': 1, 'ω²': 2}
    print(f"\n  Z_2 indicator (i) — parity (inner_idx + outer_idx) mod 2:")
    even_total = 0
    odd_total = 0
    for (i_lab, o_lab), c in counts.items():
        s = (idx[i_lab] + idx[o_lab]) % 2
        if s == 0:
            even_total += c
        else:
            odd_total += c
    print(f"    even (i+j even): {even_total}")
    print(f"    odd  (i+j odd):  {odd_total}")
    if abs(even_total - odd_total) < 4 and even_total + odd_total == 280:
        print(f"    → BALANCED Z_2!  (Difference {abs(even_total - odd_total)} of 280.)")

    print(f"\n  Z_2 indicator (ii) — diagonal (inner = outer) vs off-diagonal:")
    diag = sum(c for (i, o), c in counts.items() if i == o)
    offd = 280 - diag
    print(f"    diagonal (inner=outer): {diag}")
    print(f"    off-diagonal:           {offd}")

    print(f"\n  Z_2 indicator (iii) — (1, 1) (totally trivial) vs everything else:")
    triv = counts.get(('1', '1'), 0)
    nontriv = 280 - triv
    print(f"    totally trivial (1, 1):  {triv}")
    print(f"    non-trivial:             {nontriv}")


# -----------------------------------------------------------------------------
def main():
    print(r"""
==========================================================================================
A.iv — joint inner × outer C_3 action on H_F + Z_2 sub-structure search
Speculative slow-path probe per `R1_Z2_grading_slow_path_scoping_2026-05-14.md`.
==========================================================================================""")
    U_inner_C8 = part_A_inner_c3()
    inner_C3 = part_B_inner_c3_on_AF(U_inner_C8)
    outer_C3 = part_C_outer_c3_on_AF()
    commute = part_D_commutation(inner_C3, outer_C3)
    counts = part_E_joint_decomposition(inner_C3, outer_C3)
    part_F_z2_search(counts)

    print("\n" + "=" * 100)
    print("A.iv VERDICT")
    print("=" * 100)
    print(f"""
  ESTABLISHED (this probe):
   (i)   Inner C_3 (B6's Spin(6) lift U_{{C_3}}^S) verified on Cl(6) Fock + lifted to A_F adjoint.
   (ii)  Outer C_3 (vertex permutation per M1.B) verified on A_F.
   (iii) Inner and outer COMMUTE on H_F: {commute}  →  joint Z_3 × Z_3 abelian action.
   (iv)  H_F = 280 decomposes into 9 joint sectors with the dim table above.

  Z_2 INDICATORS tested:
   - Parity (i + j) even/odd:  see above table.
   - Diagonal (inner = outer) vs off-diagonal:  see above.
   - Totally trivial (1, 1) vs non-trivial:  see above.

  HONEST READING:
   The joint Z_3 × Z_3 structure exists, but whether ANY of the candidate Z_2 indicators
   has STRUCTURAL meaning (i.e., distinguishes physical fermions from bosons or scalars)
   requires further bookkeeping with the framework's gauge group + matter content.
   The bare numerical decomposition above doesn't immediately give a boson/fermion split.

   A.iv = STRUCTURAL DATA POINT, not direct closure of the Z_2 grading question.

  ADOPTED-MSSM-Sb stands.  R1 status: INTERIM.  No graded content changes.
""")
    print("R1_Aiv_inner_outer_c3_combo_probe.py: sentinel done.")


if __name__ == "__main__":
    main()
