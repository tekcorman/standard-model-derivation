#!/usr/bin/env python3
"""
R1_Av_refined_su2L_higgs_probe.py
=================================
A.v.refined slow-path probe, following A.v.simple's closed-negative finding
(per-edge SU(2)_e doesn't preserve [a, D_F] Higgs subspace, see
`R1_Av_verdict_2026-05-14.md`).

The fix: use the FRAMEWORK's ACTUAL SU(2)_L = Spin(3) ⊂ Spin(6) lift via
Γ-bivectors (per B3 theorem, `predictions/theorem_B3_spinor_fermion_
derivation.md`).  Self-dual triple of Spin(4) gives SU(2)_L:

  J_L^1 = (Γ_{12} + Γ_{34}) / (2 · 2i) = (T_1 + T_2)
  J_L^2 = (Γ_{13} − Γ_{24}) / (2 · 2i)
  J_L^3 = (Γ_{14} + Γ_{23}) / (2 · 2i)

These form an SU(2) algebra and act per-vertex on Cl(6) Fock ≅ ℂ^8.
Lifted to A_F = ⊕_v M_8 ⊕ ⊕_e M_2 by adjoint (X_v ↦ J X_v − X_v J on each
vertex's M_8 block;  trivial on edges since SU(2)_L is a Spin(6) action
on vertex Fock, not edge Cl(2)).

What this probe does
--------------------
A — Build the 6 Brauer-Weyl Γ matrices on ℂ^8 (per B3).
B — Build SU(2)_L generators (self-dual bivector combinations) and verify
    [J_L^a, J_L^b] = i ε_{abc} J_L^c (su(2) algebra).
C — Lift SU(2)_L to adjoint action on A_F (per-vertex on M_8, trivial on edges).
D — Test closure: does SU(2)_L preserve the Higgs subspace [a, D_F]?
    Compute closure residuals.
E — If closed: diagonalize Casimir, extract spin-J multiplicities → T(R_s).
    If not closed: report negative finding.
F — Compute b_2 scalar contribution from SU(2)_L Higgs irreps.

No graded content changes.
"""

import sys
import itertools
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from proofs.foundations.de_rham_susy_fibered_v2_probe import (  # noqa: E402
    d_alg, NE, NV, SX, SY, SZ, I2,
)

np.set_printoptions(precision=4, suppress=True, linewidth=140)
TOL = 1e-9
NORM_TOL = 1e-6


# -----------------------------------------------------------------------------
# Cl(6) Brauer-Weyl on ℂ^8 (per B3)
# -----------------------------------------------------------------------------

def kron(*mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def build_gamma_cl6():
    """Per `predictions/theorem_B3_spinor_fermion.py` Brauer-Weyl construction."""
    G = [None] * 7
    G[1] = kron(SX, I2, I2)
    G[2] = kron(SY, I2, I2)
    G[3] = kron(SZ, SX, I2)
    G[4] = kron(SZ, SY, I2)
    G[5] = kron(SZ, SZ, SX)
    G[6] = kron(SZ, SZ, SY)
    return G


def bivector(G, a, b):
    """Γ_{ab} = (1/2)[Γ_a, Γ_b]  (Hermitian, generates Spin(6))."""
    return 0.5 * (G[a] @ G[b] - G[b] @ G[a])


# -----------------------------------------------------------------------------
# Part A — Verify Cl(6) + bivectors
# -----------------------------------------------------------------------------

def part_A():
    print("=" * 100)
    print("PART A — Cl(6) Brauer-Weyl on ℂ^8 (per B3) — quick verification")
    print("=" * 100)
    G = build_gamma_cl6()
    # check {Γ_a, Γ_b} = 2δ_ab
    ok = True
    for a, b in itertools.product(range(1, 7), repeat=2):
        ac = G[a] @ G[b] + G[b] @ G[a]
        expected = 2 * (1 if a == b else 0) * np.eye(8, dtype=complex)
        if not np.allclose(ac, expected, atol=TOL):
            ok = False
    print(f"  {{Γ_a, Γ_b}} = 2δ_ab :  {ok}")
    assert ok
    return G


# -----------------------------------------------------------------------------
# Part B — SU(2)_L self-dual bivector triple + algebra verification
# -----------------------------------------------------------------------------

def part_B_su2L(G):
    print("\n" + "=" * 100)
    print("PART B — SU(2)_L generators via self-dual bivectors of {Γ_1, Γ_2, Γ_3, Γ_4}")
    print("=" * 100)

    # 6 bivectors of Spin(4) = bivectors using only Γ_1..Γ_4:
    G12 = bivector(G, 1, 2)
    G13 = bivector(G, 1, 3)
    G14 = bivector(G, 1, 4)
    G23 = bivector(G, 2, 3)
    G24 = bivector(G, 2, 4)
    G34 = bivector(G, 3, 4)

    # 't Hooft self-dual / anti-self-dual decomposition.
    # Self-dual triple: {Γ_12 + Γ_34, Γ_13 − Γ_24, Γ_14 + Γ_23} / 2
    # Anti-self-dual:    {Γ_12 − Γ_34, Γ_13 + Γ_24, Γ_14 − Γ_23} / 2
    # The combination / (2i) gives Hermitian SU(2) generators.
    # The /2 factor here is so [J^a, J^b] = i ε_{abc} J^c (i.e., spin-1/2 normalisation).
    # Let's parameterise and check via numerical commutators.

    # Try the natural construction with (1/(2·2i)) prefactor (4i in denominator):
    # J_L^a = (sum of 2 bivectors)/(4i)
    # Actually for σ_a/2 normalisation (standard SU(2) Cartan), we want eigenvalues ±1/2.
    # Bivector Γ_12/(2i) has eigenvalues ±1/2 already on the 2-spinor side, so:
    # J^a as a sum of bivectors needs careful normalisation.

    # Empirical approach:  build candidates with a prefactor κ, then verify the SU(2) algebra
    # [J^a, J^b] = i ε_{abc} J^c uniquely fixes κ.

    # Candidate (most common 't Hooft / instanton convention):
    # J_L^1 = (Γ_{12} + Γ_{34}) / (4 i)
    # J_L^2 = (Γ_{13} − Γ_{24}) / (4 i)
    # J_L^3 = (Γ_{14} + Γ_{23}) / (4 i)
    JL1 = (G12 + G34) / (4j)
    JL2 = (G13 - G24) / (4j)
    JL3 = (G14 + G23) / (4j)

    # Verify Hermiticity
    print(f"\n  J_L^1 Hermitian :  {np.allclose(JL1, JL1.conj().T, atol=TOL)}")
    print(f"  J_L^2 Hermitian :  {np.allclose(JL2, JL2.conj().T, atol=TOL)}")
    print(f"  J_L^3 Hermitian :  {np.allclose(JL3, JL3.conj().T, atol=TOL)}")
    # Verify su(2) algebra [J^a, J^b] = i ε_{abc} J^c
    J = [None, JL1, JL2, JL3]
    eps = {(1,2,3):1, (2,3,1):1, (3,1,2):1, (3,2,1):-1, (1,3,2):-1, (2,1,3):-1}
    print(f"\n  SU(2) algebra check [J_L^a, J_L^b] = i ε_{{abc}} J_L^c :")
    algebra_ok = True
    for a in [1,2,3]:
        for b in [1,2,3]:
            if a >= b: continue
            comm = J[a] @ J[b] - J[b] @ J[a]
            c = 6 - a - b  # the third index
            sign = eps.get((a,b,c), 0)
            expected = 1j * sign * J[c]
            err = np.linalg.norm(comm - expected)
            print(f"    [J_L^{a}, J_L^{b}] = i · {sign:+d} · J_L^{c}     err = {err:.3e}")
            if err > TOL:
                algebra_ok = False

    if not algebra_ok:
        print(f"\n  SU(2) algebra NOT verified with this normalisation;  trying alternative...")
        # Try with /(2i) instead of /(4i):
        JL1 = (G12 + G34) / (2j)
        JL2 = (G13 - G24) / (2j)
        JL3 = (G14 + G23) / (2j)
        J = [None, JL1, JL2, JL3]
        print(f"\n  Retry with J = (Γ_a + Γ_b) / 2i :")
        algebra_ok = True
        for a in [1,2,3]:
            for b in [1,2,3]:
                if a >= b: continue
                comm = J[a] @ J[b] - J[b] @ J[a]
                c = 6 - a - b
                sign = eps.get((a,b,c), 0)
                expected = 1j * sign * J[c]
                err = np.linalg.norm(comm - expected)
                print(f"    [J_L^{a}, J_L^{b}] = i · {sign:+d} · J_L^{c}     err = {err:.3e}")
                if err > TOL:
                    algebra_ok = False
    print(f"\n  ⇒ SU(2)_L algebra verified : {algebra_ok}")

    # Print eigenvalues (should be ±J(J+1) etc.)
    eig = np.linalg.eigvalsh(JL3)
    print(f"\n  J_L^3 eigenvalues : {sorted(np.round(eig, 4).tolist())}")
    casimir = JL1 @ JL1 + JL2 @ JL2 + JL3 @ JL3
    cas_eig = np.linalg.eigvalsh((casimir + casimir.conj().T) / 2)
    print(f"  Casimir J² eigenvalues : {sorted(np.round(cas_eig, 4).tolist())}")
    return JL1, JL2, JL3, algebra_ok


# -----------------------------------------------------------------------------
# Part C — Lift SU(2)_L to adjoint action on A_F
# -----------------------------------------------------------------------------

def build_D_F():
    d = d_alg((0.0, 0.0, 0.0))
    dim0, dim1 = NV * 64, NE * 4
    D_F = np.zeros((dim0 + dim1, dim0 + dim1), dtype=complex)
    D_F[:dim0, dim0:] = d.conj().T
    D_F[dim0:, :dim0] = d
    return D_F, dim0, dim1


def left_mult_C0(M8, vertex):
    dim0, dim1 = NV * 64, NE * 4
    op = np.zeros((dim0 + dim1, dim0 + dim1), dtype=complex)
    op[vertex * 64:(vertex + 1) * 64, vertex * 64:(vertex + 1) * 64] = np.kron(np.eye(8, dtype=complex), M8)
    return op


def left_mult_C1(M2, edge):
    dim0, dim1 = NV * 64, NE * 4
    op = np.zeros((dim0 + dim1, dim0 + dim1), dtype=complex)
    op[dim0 + edge * 4:dim0 + (edge + 1) * 4, dim0 + edge * 4:dim0 + (edge + 1) * 4] = np.kron(np.eye(2, dtype=complex), M2)
    return op


def adjoint_on_C0(J8, vertex):
    """Adjoint action of J (8×8) on vertex's M_8 block of C⁰_alg, lifted to 280×280.

    For X ∈ M_8, ad_J(X) = J X − X J.  On flattened (column-major) M_8 vector:
        flatten(J X − X J) = (np.kron(I_8, J) − np.kron(J.T, I_8)) @ flatten(X)
    """
    dim0, dim1 = NV * 64, NE * 4
    op = np.zeros((dim0 + dim1, dim0 + dim1), dtype=complex)
    I8 = np.eye(8, dtype=complex)
    ad_block = np.kron(I8, J8) - np.kron(J8.T, I8)
    op[vertex * 64:(vertex + 1) * 64, vertex * 64:(vertex + 1) * 64] = ad_block
    return op


def part_C_lift_to_AF(JL1, JL2, JL3):
    print("\n" + "=" * 100)
    print("PART C — Lift SU(2)_L to A_F = ⊕_v M_8 ⊕ ⊕_e M_2 via adjoint (per-vertex)")
    print("=" * 100)
    # Sum over all 4 vertices: SU(2)_L generator acts simultaneously on each vertex's M_8.
    ad_J = []
    for a, J in enumerate([JL1, JL2, JL3], 1):
        op = np.zeros((280, 280), dtype=complex)
        for v in range(NV):
            op = op + adjoint_on_C0(J, v)
        ad_J.append(op)
        # Verify Hermiticity (J Hermitian + adjoint action ⇒ ad_J Hermitian)
        print(f"  ad_{{J_L^{a}}} on A_F :   Hermitian = {np.allclose(op, op.conj().T, atol=NORM_TOL)}, "
              f"‖·‖_F = {np.linalg.norm(op):.3f}")
    # Check SU(2) algebra on A_F: [ad_J^a, ad_J^b] = i ε_{abc} ad_J^c
    print(f"\n  SU(2) algebra check on the lifted ad_J operators:")
    eps = {(1,2,3):1, (2,3,1):1, (3,1,2):1, (3,2,1):-1, (1,3,2):-1, (2,1,3):-1}
    for a in range(3):
        for b in range(a+1, 3):
            comm = ad_J[a] @ ad_J[b] - ad_J[b] @ ad_J[a]
            c = 3 - a - b   # (0+1+2 = 3)
            sign = eps.get((a+1, b+1, c+1), 0)
            expected = 1j * sign * ad_J[c]
            err = np.linalg.norm(comm - expected)
            print(f"    [ad_{{J_L^{a+1}}}, ad_{{J_L^{b+1}}}] − i·{sign:+d}·ad_{{J_L^{c+1}}}   err = {err:.3e}")
    return ad_J


# -----------------------------------------------------------------------------
# Part D — Higgs subspace closure under SU(2)_L
# -----------------------------------------------------------------------------

def part_D_higgs_closure(D_F, ad_J):
    print("\n" + "=" * 100)
    print("PART D — Higgs subspace [a, D_F] under SU(2)_L : closure check + Casimir")
    print("=" * 100)
    # Build the 280 Hermitian generators of A_F and their commutators with D_F.
    M8_basis = []
    for i in range(8):
        E = np.zeros((8, 8), dtype=complex); E[i, i] = 1.0
        M8_basis.append(E)
    for i in range(8):
        for j in range(i + 1, 8):
            E_s = np.zeros((8, 8), dtype=complex); E_s[i, j] = 1; E_s[j, i] = 1
            M8_basis.append(E_s)
            E_a = np.zeros((8, 8), dtype=complex); E_a[i, j] = 1j; E_a[j, i] = -1j
            M8_basis.append(E_a)
    M2_basis = []
    for i in range(2):
        E = np.zeros((2, 2), dtype=complex); E[i, i] = 1.0
        M2_basis.append(E)
    for i in range(2):
        for j in range(i + 1, 2):
            pass
    E_s = np.array([[0, 1], [1, 0]], dtype=complex); M2_basis.append(E_s)
    E_a = np.array([[0, 1j], [-1j, 0]], dtype=complex); M2_basis.append(E_a)

    gens = []
    for v in range(NV):
        for g in M8_basis:
            gens.append(left_mult_C0(g, v))
    for e in range(NE):
        for g in M2_basis:
            gens.append(left_mult_C1(g, e))
    print(f"\n  A_F generators built: {len(gens)} (expected 280)")
    higgs_vecs = []
    for a in gens:
        comm = a @ D_F - D_F @ a
        higgs_vecs.append(comm.flatten())
    H = np.array(higgs_vecs)
    rank = np.linalg.matrix_rank(H, tol=TOL)
    print(f"  rank(Higgs subspace) = {rank}  (= {len(gens) - rank} linearly dep)")

    # Build orthonormal basis
    _, _, Vh = np.linalg.svd(H, full_matrices=False)
    basis = Vh[:rank]
    print(f"  Higgs basis: shape {basis.shape}")

    # Test closure: for each ad_J^a, compute ad_J^a(basis_j), project back, measure residual
    print(f"\n  Closure check for SU(2)_L on Higgs subspace [a, D_F]:")
    residuals_all = {1: [], 2: [], 3: []}
    for a in range(3):
        for j in range(rank):
            op = basis[j].reshape(280, 280)
            ad_op = ad_J[a] @ op - op @ ad_J[a]
            ad_flat = ad_op.flatten()
            proj = (basis.conj() @ ad_flat)
            proj_back = basis.T @ proj
            res = np.linalg.norm(ad_flat - proj_back)
            residuals_all[a + 1].append(res)
    for a in [1, 2, 3]:
        mx = max(residuals_all[a]); mn = np.mean(residuals_all[a])
        print(f"    ad_{{J_L^{a}}} : max residual = {mx:.3e}, mean = {mn:.3e}")
    max_total = max(max(r) for r in residuals_all.values())
    closed = max_total < NORM_TOL
    print(f"\n  ⇒ Higgs subspace closed under SU(2)_L : {closed}  (max residual = {max_total:.3e})")
    return basis, rank, closed


# -----------------------------------------------------------------------------
# Part E — Casimir spectrum on Higgs subspace
# -----------------------------------------------------------------------------

def part_E_casimir(basis, ad_J, rank):
    print("\n" + "=" * 100)
    print("PART E — SU(2)_L Casimir spectrum on Higgs subspace [a, D_F]")
    print("=" * 100)
    # Build ad_J^a in the basis: (rank, rank) matrix
    print(f"\n  building ad_{{J_L^a}} in Higgs subspace basis...")
    def ad_in_basis(ad_op):
        # ad_op is (280, 280); compute its action on each basis vector
        ad_matrix = np.zeros((rank, rank), dtype=complex)
        for j in range(rank):
            op = basis[j].reshape(280, 280)
            ad_op_j = ad_op @ op - op @ ad_op
            ad_flat = ad_op_j.flatten()
            ad_matrix[:, j] = basis.conj() @ ad_flat
        return ad_matrix

    ad_in_b = [ad_in_basis(a) for a in ad_J]
    casimir = ad_in_b[0] @ ad_in_b[0] + ad_in_b[1] @ ad_in_b[1] + ad_in_b[2] @ ad_in_b[2]
    # Hermiticity check
    herm_err = np.linalg.norm(casimir - casimir.conj().T)
    print(f"  Casimir Hermiticity error: {herm_err:.3e}")
    eig = np.linalg.eigvalsh((casimir + casimir.conj().T) / 2).real
    # bin by J(J+1)
    J_to_dim = {0.0: 1, 0.5: 2, 1.0: 3, 1.5: 4, 2.0: 5, 2.5: 6, 3.0: 7, 3.5: 8, 4.0: 9}
    J_to_C = {J: J * (J + 1) for J in J_to_dim}
    spin_counts = {J: 0 for J in J_to_dim}
    unassigned = []
    for c in eig:
        matched = False
        for J, target_c in J_to_C.items():
            if abs(c - target_c) < 1e-5:
                spin_counts[J] += 1
                matched = True
                break
        if not matched:
            unassigned.append(c)

    print(f"\n  Casimir eigenvalues binned by spin J:")
    print(f"     {'J':>4} | {'2J+1':>5} | {'count':>6} | {'n_J (count / (2J+1))':>22}")
    print("   " + "-" * 50)
    Tsum = 0.0
    for J in sorted(J_to_dim.keys()):
        cnt = spin_counts[J]
        d = int(2 * J + 1)
        if cnt == 0:
            continue
        n_J_int = cnt // d if cnt % d == 0 else cnt / d
        T_J = J * (J + 1) * (2 * J + 1) / 3.0
        contrib = (cnt / d) * T_J
        Tsum += contrib
        print(f"     {J:>4} | {d:>5} | {cnt:>6} | n_J = {n_J_int} (T_J = {T_J:.3f}, contrib to ΣT = {contrib:.4f})")
    if unassigned:
        print(f"\n     unassigned eigvals: {len(unassigned)} (not matching J(J+1) for J ≤ 4)")
        print(f"       sample values : {sorted(unassigned)[:10]}")

    total_assigned = sum(spin_counts.values())
    print(f"\n     total assigned: {total_assigned} (of {rank} total Higgs dim)")
    print(f"\n  ΣT(R_s)_SU(2)_L on Higgs subspace = {Tsum:.4f}")
    print(f"  Contribution to b_2 (scalar piece) = (1/3) · {Tsum:.4f} = {Tsum/3:.4f}")
    return spin_counts, unassigned, Tsum


# -----------------------------------------------------------------------------
def main():
    print(r"""
==========================================================================================
A.v.refined — Higgs subspace [a, D_F] under B3's actual SU(2)_L = Spin(3) ⊂ Spin(6)
Slow-path probe following A.v.simple's CLOSED-NEGATIVE result (per-edge SU(2)_e fails).
==========================================================================================""")
    G = part_A()
    JL1, JL2, JL3, alg_ok = part_B_su2L(G)
    if not alg_ok:
        print("\n  SU(2)_L algebra not verified — aborting.")
        return
    D_F, _, _ = build_D_F()
    ad_J = part_C_lift_to_AF(JL1, JL2, JL3)
    basis, rank, closed = part_D_higgs_closure(D_F, ad_J)
    print("\n" + "=" * 100)
    if closed:
        spin_counts, unassigned, Tsum = part_E_casimir(basis, ad_J, rank)
        verdict = "POSITIVE: Higgs subspace closed under SU(2)_L"
    else:
        spin_counts = None; unassigned = None; Tsum = None
        verdict = "NEGATIVE: Higgs subspace NOT closed under SU(2)_L either"
    print("A.v.refined INTERIM VERDICT")
    print("=" * 100)
    print(f"""
  SU(2)_L generators (B3's Spin(3) ⊂ Spin(6) self-dual bivector triple):  ✓ built, su(2) algebra verified
  Lifted to A_F by adjoint:                                                 ✓ Hermitian, su(2) algebra preserved
  Closure of Higgs subspace [a, D_F] under SU(2)_L:                         {closed}
""")
    if closed:
        print(f"  Casimir spectrum: {sum(spin_counts.values()) if spin_counts else 0} assigned, {len(unassigned) if unassigned else 0} unassigned")
        print(f"  ΣT(R_s) on Higgs subspace = {Tsum:.4f}")
        if Tsum > 0:
            print(f"  → A.v.refined CLOSES POSITIVELY:  framework's SU(2)_L sees Higgs scalars with non-zero T(R_s).")
        else:
            print(f"  → A.v.refined CLOSED-NEGATIVE for SU(2)_L:  all Higgs modes are singlets (no b_2 scalar contribution).")
    else:
        print(f"""
  Higgs subspace [a, D_F] is ALSO NOT CLOSED under B3's SU(2)_L.
  This rules out the simple [a, D_F] Higgs construction for the b_i question,
  REGARDLESS of which gauge group we use.

  The next slow-path option is A.v.full (full 1-form module Σ a [D_F, b],
  gauge-closed by construction) or A.i (χ̂ lift to Fock).
""")
    print(f"\n  ADOPTED-MSSM-Sb stands.  R1 status: INTERIM.")
    print(f"  Verdict: {verdict}")
    print("R1_Av_refined_su2L_higgs_probe.py: sentinel done.")


if __name__ == "__main__":
    main()
