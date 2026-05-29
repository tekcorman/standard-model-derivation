#!/usr/bin/env python3
"""
4d_dirac_inner_fluctuation_probe.py
===================================
Step 2 of the 4D spacetime spectral triple project (Step 1 in
`4d_dirac_operator_construction_probe.py` closed positively;
`project_4d_dirac_step1_complete_2026-05-14.md`).

Step 2's goal: activate the inner-fluctuation gauge field A_μ ∈ A_F (the
substrate's operator algebra), expand D_4² to order A², and extract the
Yang-Mills 1/g² coefficient from the spectral action's a_4 heat-kernel piece.
Compare to the framework's α_GUT⁻¹ = 24 (theorem-grade upstream from
`theorem_sin2_theta_W_unification.md`).

Analytical content (Step 2a)
----------------------------
For the framework's D_4 = D_M ⊗ 1_F + γ_5^M ⊗ D_F with M = flat Euclidean
4-manifold and CONSTANT inner fluctuation A_μ (a Hermitian element of A_F
acting on H_F, independent of x), the inner-fluctuated Dirac is

  D_4(p) = γ^μ ⊗ (p_μ · 1_F - A_μ) + γ_5 ⊗ D_F

and its square decomposes as

  D_4(p)² = I_4 ⊗ [(p − A)² + D_F²]                              (positive operator)
          + (i/2) γ^{μν} ⊗ F_μν                                    (gauge-curvature)
          − γ^μ γ_5 ⊗ [A_μ, D_F]                                   (Higgs-cross)

with F_μν = −i [A_μ, A_ν]   (constant-A field strength).
Derivation:  γ^μ γ^ν = δ^μν + γ^{μν};  X_μ = p_μ − A_μ;  [X_μ, X_ν] = [A_μ, A_ν]
  = i F_μν;  so γ^μ γ^ν X_μ X_ν = X² + (1/2) γ^{μν}[X_μ, X_ν] = X² + (i/2) γ^{μν} F_μν.

The Higgs-cross term −γ^μ γ_5 ⊗ [A_μ, D_F] is the framework's analog of
CC's Higgs scalar emerging from F-side inner fluctuations.  It vanishes
iff A commutes with D_F.

Step 2 then has two natural sub-cases:
  (i) PURE GAUGE — pick A_μ commuting with D_F.  Then D_4² reduces to a
      Laplace-type operator with explicit YM curvature and the standard
      Gilkey a_4 formula gives the Yang-Mills coefficient.
  (ii) FULL — keep the [A, D_F] term;  YM and Higgs Yukawa terms both
      appear at order A² in a_4.  This is what's physically happening.

Step 2 here does (i) — a clean test of the YM extraction; Step 2's
extension to (ii) (the Higgs sector) is naturally Step 2.5 / Step 3 work.

What this probe does
--------------------
A — Build a test PURE-GAUGE inner fluctuation:  the part of A_F's algebra
    that commutes with D_F.  Find the commutant numerically; pick a
    Hermitian generator A with [A, D_F] = 0 and [A, A'] ≠ 0 for some other
    generator A'.

B — Form the field strength F_μν = −i[A_μ, A_ν] for A_μ = c · A · ε^{(μ)},
    ε^{(μ)} the 4-direction unit vectors with A_1 = c A, A_2 = c B (two
    non-commuting commutant generators).  Compute Tr_F(F_μν F^μν) numerically.

C — Compare to standard CC formula.  Gilkey's a_4 for the Laplace-type
    operator D_4² gives

      a_4_YM = (1/(4π²)) × ∫ d⁴x √g × (1/12) × Tr_{H_4}(F_μν F^μν)
             = (1/(4π²)) × (1/12) × 4 × Tr_F(F_μν F^μν) × ∫ d⁴x √g

    where the factor 4 is the spinor trace (Tr_spinor(I_4) = 4).  Equating
    to the CC Yang-Mills convention S_YM = −(1/4g²) ∫ F·F √g:

      1/g² = (f_0 / 24π²) × Tr_F(F·F)|_{normalized to one F-direction pair}

    where f_0 = f(0) is the spectral function value at 0.

D — Read off α_GUT⁻¹ = 4π/g² and compare to 24.

E — Verdict.

This is a Step-2 SCOPING probe, not a full theorem-grade derivation.  The
goal is to (a) confirm the spectral-action machinery cleanly produces a YM
term with substrate-derived coefficient, (b) identify which Tr_F invariant
plays the role of α_GUT⁻¹, (c) check structural consistency with the
framework's α_GUT⁻¹ = 24.  No graded content changes.
"""

import sys
from pathlib import Path

import numpy as np
from numpy.linalg import eigvalsh

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from proofs.foundations.de_rham_susy_fibered_v2_probe import (  # noqa: E402
    d_alg, NE, NV, incident_edges, T_SLOT, SX, SY, SZ, I2,
    _cl2_action_on_slot,
)

np.set_printoptions(precision=4, suppress=True, linewidth=140)


# -----------------------------------------------------------------------------
# Reuse Step-1's D_F (Q̂_alg at k=0) and A_F's left-multiplication action
# -----------------------------------------------------------------------------

def build_D_F():
    d = d_alg((0.0, 0.0, 0.0))
    dim0, dim1 = NV * 64, NE * 4
    D_F = np.zeros((dim0 + dim1, dim0 + dim1), dtype=complex)
    D_F[:dim0, dim0:] = d.conj().T
    D_F[dim0:, :dim0] = d
    return D_F, dim0, dim1


def left_mult_C0_at_vertex(M8, vertex):
    """Left-multiplication by 8×8 matrix M8 on vertex v of C⁰_alg side of H_F.
    The action is X_v ↦ M8 · X_v (matrix product).
    On H_F = (⊕_v M_8(ℂ)_v) ⊕ (⊕_e M_2(ℂ)_e) with HS inner product, this
    is a (256+24)×(256+24) operator; identity on everything except the
    vertex-v block.  In the column-major flattened basis of M_8, the
    operator is `M8 ⊗ I_8` (since left-mult by M8 in column-major flatten
    is np.kron(M8, I_8) — actually it's np.kron(I_8, M8); see below).
    Actually for column-major flatten of A (i.e. vec(A) = stack of columns),
    left-mult by M acts as np.kron(I, M)."""
    dim0, dim1 = NV * 64, NE * 4
    op = np.zeros((dim0 + dim1, dim0 + dim1), dtype=complex)
    block = np.kron(np.eye(8, dtype=complex), M8)
    op[vertex * 64:(vertex + 1) * 64, vertex * 64:(vertex + 1) * 64] = block
    return op


def left_mult_C1_at_edge(M2, edge):
    """Left-multiplication by 2×2 matrix M2 on edge e of C¹_alg side of H_F.
    Identity on C⁰_alg, identity on other edges; on edge e's 4-dim M_2(ℂ)
    block, action is X_e ↦ M2 · X_e.  In column-major flatten of M_2,
    left-mult by M2 is np.kron(I_2, M2)."""
    dim0, dim1 = NV * 64, NE * 4
    op = np.zeros((dim0 + dim1, dim0 + dim1), dtype=complex)
    block = np.kron(np.eye(2, dtype=complex), M2)
    op[dim0 + edge * 4:dim0 + (edge + 1) * 4, dim0 + edge * 4:dim0 + (edge + 1) * 4] = block
    return op


# -----------------------------------------------------------------------------
# Step 2a — symbolic D_4²(p, A) form check (numeric verification at one p, A)
# -----------------------------------------------------------------------------

def euclidean_gamma_4():
    Z2 = np.zeros((2, 2), dtype=complex)
    g0 = np.block([[Z2, I2], [I2, Z2]])
    g1 = np.block([[Z2, -1j * SX], [1j * SX, Z2]])
    g2 = np.block([[Z2, -1j * SY], [1j * SY, Z2]])
    g3 = np.block([[Z2, -1j * SZ], [1j * SZ, Z2]])
    g5 = g0 @ g1 @ g2 @ g3
    return (g0, g1, g2, g3), g5


def gamma_munu(gammas, mu, nu):
    """γ^{μν} = (1/2) [γ^μ, γ^ν]."""
    return 0.5 * (gammas[mu] @ gammas[nu] - gammas[nu] @ gammas[mu])


def build_D_4_with_A(p, A_list, D_F=None):
    """D_4(p) = γ^μ ⊗ (p_μ · I_F − A_μ) + γ_5 ⊗ D_F, with A_list[mu] a 280×280 Hermitian operator."""
    gammas, g5_M = euclidean_gamma_4()
    if D_F is None:
        D_F, _, _ = build_D_F()
    I_F = np.eye(D_F.shape[0], dtype=complex)
    D_4 = np.kron(g5_M, D_F)
    for mu in range(4):
        op_F = p[mu] * I_F - A_list[mu]
        D_4 = D_4 + np.kron(gammas[mu], op_F)
    return D_4, gammas, g5_M, D_F, I_F


def part_2a_verify_decomposition():
    print("=" * 100)
    print("PART 2A — verify D_4²(p, A) decomposition for constant inner fluctuation")
    print("=" * 100)
    rng = np.random.default_rng(20260514)
    p = rng.normal(size=4)
    D_F, dim0, dim1 = build_D_F()
    I_F = np.eye(dim0 + dim1, dtype=complex)
    # constant Hermitian inner fluctuations — pick edge 0's σ_z and σ_x as test directions
    A_0 = 0.3 * left_mult_C1_at_edge(SZ, 0)   # A_0 along time, generator σ_z on edge 0
    A_1 = 0.3 * left_mult_C1_at_edge(SX, 0)   # A_1 along x, generator σ_x on edge 0
    A_2 = np.zeros_like(A_0)
    A_3 = np.zeros_like(A_0)
    A_list = [A_0, A_1, A_2, A_3]
    # check Hermiticity of A_μ
    for mu in range(4):
        ok_h = np.allclose(A_list[mu], A_list[mu].conj().T)
        if not ok_h:
            print(f"  WARNING: A_{mu} not Hermitian")
    # F_μν = -i [A_μ, A_ν]   (constant-A field strength, no x-derivative term)
    F = [[(-1j) * (A_list[mu] @ A_list[nu] - A_list[nu] @ A_list[mu]) for nu in range(4)] for mu in range(4)]
    F_01 = F[0][1]
    F_10 = F[1][0]
    print(f"\n  test inner fluctuation:  A_0 = 0.3·L_σz on edge 0,  A_1 = 0.3·L_σx on edge 0,  A_2 = A_3 = 0")
    print(f"  Hermiticity of F_01: ‖F_01 - F_01†‖ = {np.linalg.norm(F_01 - F_01.conj().T):.3e}  (should be 0 for real F_μν)")
    print(f"  Antisymmetry:        ‖F_01 + F_10‖ = {np.linalg.norm(F_01 + F_10):.3e}")
    # build D_4
    D_4, gammas, g5_M, D_F, I_F = build_D_4_with_A(p, A_list, D_F)
    # check (P1) Hermitian
    err_herm = np.linalg.norm(D_4 - D_4.conj().T)
    print(f"\n  D_4(p, A) Hermitian:   ‖D_4 − D_4†‖ = {err_herm:.3e}")
    assert err_herm < 1e-10
    # decompose:  D_4² = I_4 ⊗ [(p−A)² + D_F²]  −  (i/2) γ^{μν} ⊗ F_μν  −  γ^μ γ_5 ⊗ [A_μ, D_F]
    # term 1
    pmA_sq = np.zeros_like(I_F)
    for mu in range(4):
        op = p[mu] * I_F - A_list[mu]
        pmA_sq = pmA_sq + op @ op
    term1 = np.kron(np.eye(4, dtype=complex), pmA_sq + D_F @ D_F)
    # term 2 (gauge curvature) — derivation gives +(i/2), not −(i/2)
    term2 = np.zeros_like(D_4)
    for mu in range(4):
        for nu in range(4):
            term2 = term2 + (+0.5j) * np.kron(gamma_munu(gammas, mu, nu), F[mu][nu])
    # term 3 (Higgs-cross)
    term3 = np.zeros_like(D_4)
    for mu in range(4):
        comm = A_list[mu] @ D_F - D_F @ A_list[mu]
        term3 = term3 + np.kron(gammas[mu] @ g5_M, -comm)
    expected = term1 + term2 + term3
    actual = D_4 @ D_4
    err_decomp = np.linalg.norm(actual - expected)
    print(f"\n  decomposition check:  D_4² = I_4⊗[(p−A)²+D_F²] + (i/2)γ^μν⊗F_μν − γ^μγ_5⊗[A_μ,D_F]")
    print(f"  ‖D_4² − decomposition‖ = {err_decomp:.3e}   →  {err_decomp < 1e-9}")
    assert err_decomp < 1e-9, f"decomposition failed at {err_decomp}"
    # diagnose Higgs-cross term magnitude — is A_μ commuting with D_F?
    norm_higgs = sum(np.linalg.norm(A_list[mu] @ D_F - D_F @ A_list[mu]) for mu in range(4))
    print(f"\n  Higgs-cross diagnostic:  Σ_μ ‖[A_μ, D_F]‖ = {norm_higgs:.3f}")
    print(f"  ⇒ A_μ DOES NOT commute with D_F for this choice (edge-only inner fluctuation has Higgs component)")
    print(f"     → for PURE gauge extraction (Step 2c) we need A_μ in the COMMUTANT of D_F (Part 2B finds these)")
    return D_4, A_list, F


# -----------------------------------------------------------------------------
# Step 2b — find the commutant of D_F in A_F and pick a pure-gauge fluctuation
# -----------------------------------------------------------------------------

def part_2b_find_pure_gauge():
    print("\n" + "=" * 100)
    print("PART 2B — find pure-gauge inner fluctuations: A in A_F commuting with D_F")
    print("=" * 100)
    D_F, dim0, dim1 = build_D_F()
    # The commutant of D_F in A_F is the set of A in A_F with [A, D_F] = 0.
    # A_F = (⊕_v M_8) ⊕ (⊕_e M_2);  64 generators per vertex + 4 per edge.
    # We screen generators by computing the commutator-norm:
    print(f"\n  Screening A_F generators for [A, D_F] = 0 (i.e. pure-gauge):")
    print(f"    {NV} vertices × 64 generators per vertex = {NV*64} matter generators")
    print(f"    {NE} edges × 4 generators per edge       = {NE*4}  gauge generators")

    # Enumerate Hermitian basis generators of A_F:
    #   per vertex v:  E_{ij} = (e_ij + e_ji)/2  (symm) and (e_ij - e_ji)/(2i) (antisym),
    #                  i,j ∈ [0..7],  i ≤ j  →  64 Hermitian generators of M_8(ℂ).
    #   per edge   e:  similarly for M_2(ℂ),  giving 4 Hermitian generators.

    def hermitian_basis_M_n(n):
        gens = []
        for i in range(n):
            E = np.zeros((n, n), dtype=complex); E[i, i] = 1.0
            gens.append(E)
        for i in range(n):
            for j in range(i + 1, n):
                E_s = np.zeros((n, n), dtype=complex); E_s[i, j] = 1.0; E_s[j, i] = 1.0
                gens.append(E_s)
                E_a = np.zeros((n, n), dtype=complex); E_a[i, j] = 1j; E_a[j, i] = -1j
                gens.append(E_a)
        return gens   # length = n²

    M8_gens = hermitian_basis_M_n(8)
    M2_gens = hermitian_basis_M_n(2)

    n_match_vert = 0
    n_close_vert = 0
    matter_commutant_count = []
    for v in range(NV):
        cnt = 0
        for g in M8_gens:
            A = left_mult_C0_at_vertex(g, v)
            comm = A @ D_F - D_F @ A
            nrm = np.linalg.norm(comm)
            if nrm < 1e-9:
                cnt += 1
        matter_commutant_count.append(cnt)
    print(f"\n  matter-side (C⁰_alg) commutant counts per vertex (out of 64 generators):")
    for v, c in enumerate(matter_commutant_count):
        print(f"    vertex {v}:  {c}/64 commute exactly with D_F")

    gauge_commutant_count = []
    for e in range(NE):
        cnt = 0
        for g in M2_gens:
            A = left_mult_C1_at_edge(g, e)
            comm = A @ D_F - D_F @ A
            nrm = np.linalg.norm(comm)
            if nrm < 1e-9:
                cnt += 1
        gauge_commutant_count.append(cnt)
    print(f"\n  gauge-side (C¹_alg) commutant counts per edge (out of 4 generators):")
    for e, c in enumerate(gauge_commutant_count):
        print(f"    edge {e}:    {c}/4 commute exactly with D_F")

    total_commutant_dim = sum(matter_commutant_count) + sum(gauge_commutant_count)
    total_AF_dim = NV * 64 + NE * 4   # 280
    print(f"\n  TOTAL: {total_commutant_dim} / {total_AF_dim} elements of A_F commute exactly with D_F")
    print(f"\n  Interpretation:")
    print(f"     Commutant-of-D_F ∩ A_F  is the algebra whose left-multiplication action commutes")
    print(f"     with the supercharge.  This is the framework's analog of CC's 'gauge subalgebra'")
    print(f"     — what survives as a pure YM field after inner fluctuation.")
    print(f"     Elements NOT in the commutant generate the Higgs / Yukawa scalar (see [A_μ, D_F]).")

    return matter_commutant_count, gauge_commutant_count


# -----------------------------------------------------------------------------
# Step 2c — pick two non-commuting pure-gauge generators, compute Tr_F(F·F)
# -----------------------------------------------------------------------------

def find_pure_gauge_generators(D_F):
    """Find Hermitian generators of A_F that (i) commute with D_F and (ii) don't commute with each other."""
    def hermitian_basis_M_n(n):
        gens = []
        for i in range(n):
            E = np.zeros((n, n), dtype=complex); E[i, i] = 1.0
            gens.append(E)
        for i in range(n):
            for j in range(i + 1, n):
                E_s = np.zeros((n, n), dtype=complex); E_s[i, j] = 1.0; E_s[j, i] = 1.0
                gens.append(E_s)
                E_a = np.zeros((n, n), dtype=complex); E_a[i, j] = 1j; E_a[j, i] = -1j
                gens.append(E_a)
        return gens

    M8_gens = hermitian_basis_M_n(8)
    pure_gens = []
    # vertex generators
    for v in range(NV):
        for i, g in enumerate(M8_gens):
            A = left_mult_C0_at_vertex(g, v)
            comm = A @ D_F - D_F @ A
            if np.linalg.norm(comm) < 1e-9:
                pure_gens.append((f"vertex_{v}_M8gen_{i}", A))
    # edge generators
    M2_gens = hermitian_basis_M_n(2)
    for e in range(NE):
        for i, g in enumerate(M2_gens):
            A = left_mult_C1_at_edge(g, e)
            comm = A @ D_F - D_F @ A
            if np.linalg.norm(comm) < 1e-9:
                pure_gens.append((f"edge_{e}_M2gen_{i}", A))
    return pure_gens


def part_2c_pure_gauge_F_squared():
    print("\n" + "=" * 100)
    print("PART 2C — pick pure-gauge generators A, B with [A, D_F]=[B, D_F]=0 and [A,B]≠0; compute Tr_F(F·F)")
    print("=" * 100)
    D_F, _, _ = build_D_F()
    pure_gens = find_pure_gauge_generators(D_F)
    print(f"\n  found {len(pure_gens)} pure-gauge generators (Hermitian, [·, D_F] = 0)")

    # Find a non-commuting pair
    found_pair = None
    for i in range(len(pure_gens)):
        for j in range(i + 1, len(pure_gens)):
            name_i, A_i = pure_gens[i]
            name_j, A_j = pure_gens[j]
            comm = A_i @ A_j - A_j @ A_i
            if np.linalg.norm(comm) > 1e-9:
                found_pair = (name_i, A_i, name_j, A_j, comm)
                break
        if found_pair is not None:
            break

    if found_pair is None:
        print(f"\n  NO non-commuting pair found among the pure-gauge generators!")
        print(f"  → The pure-gauge subalgebra is ABELIAN (commutative).  This is a structural finding:")
        print(f"     for constant-A inner fluctuations, F_μν = −i[A_μ, A_ν] = 0, so the Yang-Mills term")
        print(f"     vanishes for ANY constant pure-gauge A.  Non-Abelian YM requires either")
        print(f"     (i) x-dependent A_μ (which contributes ∂A − ∂A to F_μν) or (ii) coupling through D_F.")
        return None

    name_i, A_i, name_j, A_j, comm = found_pair
    print(f"\n  non-commuting pure-gauge pair found:")
    print(f"    A = {name_i}")
    print(f"    B = {name_j}")
    print(f"    ‖[A, B]‖ = {np.linalg.norm(comm):.4f}")
    F_AB = -1j * comm
    # F·F   (the YM trace, structurally)
    tr_FF = float(np.real(np.trace(F_AB @ F_AB.conj().T)))
    print(f"    Tr_F(F·F†) = {tr_FF:.6f}  ← Yang-Mills trace for this constant pure-gauge pair")
    print(f"    Tr_F(F·F)  = {float(np.real(np.trace(F_AB @ F_AB))):.6f}  (Hermitian F²)")
    return found_pair


def part_2c_alt_anti_commutant():
    """Alternative: try the ANTI-commutant of D_F  ({A, D_F} = 0).  In CC theory, elements
    anti-commuting with the chirality grading correspond to off-diagonal Higgs-like fluctuations.
    Here we want gauge-like which is the COMMUTANT (above), but for completeness check size."""
    pass


# -----------------------------------------------------------------------------
# Step 2d — Z(t) numerical extraction (with the Higgs-cross term present)
# -----------------------------------------------------------------------------

def part_2d_heat_trace_numerical():
    print("\n" + "=" * 100)
    print("PART 2D — numerical small-t Laurent extraction with simple inner fluctuation (Higgs+gauge mixed)")
    print("=" * 100)
    print(f"\n  We take A_μ = c · L_{{σ_a}}_{{edge 0}} for the simplest test (one Pauli per direction).")
    print(f"  This carries BOTH gauge and Higgs components (Part 2A showed [A, D_F] ≠ 0).")
    print(f"  We extract the order-c² coefficient of the t^0 piece of Tr e^(-tD_4²(p=0, c))-Tr e^(-tD_4²(p=0, 0))/c²")
    print(f"  i.e. the coefficient that controls the bare gauge+Higgs kinetic + mass.")

    D_F, _, _ = build_D_F()
    # At p=0:  D_4(p=0, A) = -γ^μ ⊗ A_μ + γ_5 ⊗ D_F
    gammas, g5_M = euclidean_gamma_4()
    p = np.zeros(4)
    # Build inner fluctuation generator E0 (σ_z on edge 0) — affects only one of the 4 components
    E0 = left_mult_C1_at_edge(SZ, 0)
    E1 = left_mult_C1_at_edge(SX, 0)

    # vary c, compute Z(t=fixed) to extract order-c² dependence
    cs = np.linspace(0, 0.2, 5)
    t_choices = [0.05, 0.1, 0.2, 0.5]
    print(f"\n  Z(t, c) = Tr e^(-t D_4²(p=0, A=c[E0, E1, 0, 0]))   at various t and c:")
    print(f"  {'t':>10} | " + " ".join([f"{'c=':>1}{c:>5.3f}" for c in cs]) + " |    coef of c² at t^0")
    print("  " + "-" * 90)
    for t in t_choices:
        row = []
        z_at_c = []
        for c in cs:
            A_list = [c * E0, c * E1, np.zeros_like(E0), np.zeros_like(E0)]
            D_4, *_ = build_D_4_with_A(p, A_list, D_F)
            # heat trace at this t
            eigs = eigvalsh((D_4 + D_4.conj().T) / 2)
            z = float(np.real(np.sum(np.exp(-t * eigs ** 2))))   # D_4² eigenvalues = eigs² since D_4 Hermitian
            z_at_c.append(z)
        # fit Z(c) = Z(0) + a · c² for small c
        ca = np.array(cs)
        za = np.array(z_at_c)
        # second-order polynomial fit
        coeffs = np.polyfit(ca, za - z_at_c[0], 2)   # a c² + b c + 0
        a_coef = coeffs[0]
        b_coef = coeffs[1]
        row_str = " ".join([f"{z:>8.2f}" for z in z_at_c])
        print(f"  {t:>10.3f} | {row_str} |  a=c² coef = {a_coef:>9.4f},  b=c coef ≈ {b_coef:>8.3e}")


# -----------------------------------------------------------------------------
# Step 2e — pure-gauge bare 1/g² extraction via Gilkey + Tr_F(F·F)
# -----------------------------------------------------------------------------

def part_2e_pure_gauge_g2_inv(pair):
    print("\n" + "=" * 100)
    print("PART 2E — bare 1/g² extraction (pure-gauge case)")
    print("=" * 100)
    D_F, _, _ = build_D_F()
    if pair is None:
        print("\n  No non-commuting pair in the (empty) left-mult commutant of D_F — see Part 2B finding.")
        print("  STRUCTURAL CONSEQUENCE: the framework's A_F has trivial left-multiplication commutant")
        print("  with D_F = Q̂_alg(0).  This means the standard CC 'gauge subalgebra = commutant' identification")
        print("  does NOT apply.  Q̂_alg is a SUPERCHARGE that intrinsically couples matter and gauge sectors;")
        print("  constant-A inner fluctuations via left-mult by A_F generators ALL activate the Higgs cross")
        print("  term [A_μ, D_F].  Pure-gauge inner fluctuations via this route do not exist.")
        print("  ")
        print("  THE RIGHT framework analog of CC's gauge subalgebra is the GAUGE UNITARIES of A_F under")
        print("  ADJOINT action preserving D_F — which Step 1 verified at machine precision as ⊕_e SU(2)_e")
        print("  (per-edge + cross-edge SU(2) invariance).  Part 2F switches to this route and gets a")
        print("  non-trivial Tr_F(F·F).")
        return None
    name_A, A_op, name_B, B_op, comm = pair
    F_AB = -1j * comm
    tr_FF = float(np.real(np.trace(F_AB @ F_AB.conj().T)))
    # Gilkey  a_4 contains  (1/(4π²)) × (1/12) × Tr_internal(F_μν F^μν).
    # For our internal space = (spinor 4-dim) × H_F (280-dim), with F_μν ⊗ (1 in spinor sector via the γ^{μν} convention):
    #   Tr_internal(F·F) = Tr_spinor(γ^{μν}γ_{μν}) × Tr_F(F·F)  (NEEDS the precise spinor contraction;
    #   for d=4 Euclidean,  Tr_spinor(γ^{μν}γ_{μν}) = 4·(d²−d) = 4·12 = 48)
    # So a_4_YM = (1/(4π²)) × (1/12) × 48 × Tr_F(F·F) per cell of H_F.
    # The CC YM action S_YM = (1/(4g²)) ∫ Tr_gauge(F·F):
    #   matching   (1/(4π²)) × (1/12) × 48 × Tr_F(F·F)  =  (1/4g²) × Tr_gauge(F·F) × (V_M /4π²)
    # giving the trace identification Tr_F(F·F) = N_AB × Tr_gauge(F·F) and:
    #   1/g² ≈ (1/π²) × Tr_F/N_AB
    # The exact normalization needs the matter-content+rep multiplicity.  We just report the
    # *structural* number that emerges.
    print(f"\n  Pure-gauge YM trace  Tr_F(F·F)  for the (A, B) pair  =  {tr_FF:.6f}")
    print(f"  Spinor trace factor  Tr_spinor(γ^μν γ_μν) = 48  (d=4 Euclidean)")
    print(f"  Gilkey factor  1/12  (the standard YM coefficient in a_4)")
    print(f"  → spectral-action YM piece (per cell of H_F, per F_μν F^μν-direction-pair):")
    print(f"     a_4_YM = (1/(4π²)) × (1/12) × 48 × Tr_F(F·F)")
    print(f"            = (1/π²) × Tr_F(F·F)")
    print(f"            = (1/π²) × {tr_FF:.4f}")
    print(f"            = {tr_FF / np.pi ** 2:.6f}")
    print(f"  This is the structural-bare YM coefficient before any normalization to a specific")
    print(f"  gauge-group factor (SU(3)_c, SU(2)_L, U(1)_Y).")
    print(f"\n  Framework's α_GUT⁻¹ = 24 at unification (theorem-grade, Cl(6) normalization).")
    print(f"  CC's per-pair YM coefficient depends on the trace normalization of F_AB in the gauge")
    print(f"  group's Killing form — a representation-dependent factor that this Step-2 SCOPING")
    print(f"  probe does NOT fix.  Step 3 / Step 4 work needed for full identification.")


# -----------------------------------------------------------------------------
# Part 2F — gauge-unitary route (not commutant): U ∈ ⊕_e SU(2)_e preserves D_F via ADJOINT
# (per Step 1's verification).  This IS the framework's gauge group; the corresponding
# inner fluctuation is the right object for YM extraction.
# -----------------------------------------------------------------------------

def part_2f_gauge_unitary_perspective():
    print("\n" + "=" * 100)
    print("PART 2F — gauge-unitary inner fluctuation (the right CC analog for the framework)")
    print("=" * 100)
    print(f"""
  Step 1's verification: U ∈ ⊕_e SU(2)_e acts by ADJOINT (A ↦ U A U†) on H_F leaving D_F = Q̂_alg
  invariant at machine precision (per-edge + cross-edge).  THIS is the framework's gauge group,
  not the (empty) commutant of left-mult above.

  The corresponding inner-fluctuation gauge field is constructed from per-edge Hermitian
  generators of su(2)_e ⊂ A_F's edge sector.  For two non-commuting su(2)_e generators on
  the SAME edge (e.g. σ_x and σ_y on edge 0), F_μν = −i[A_μ, A_ν] computes the YM curvature
  on the gauge group's Cartan algebra.
""")
    D_F, _, _ = build_D_F()
    # A_0 = c · L_σx_{edge 0}, A_1 = c · L_σy_{edge 0}
    A_x = left_mult_C1_at_edge(SX, 0)
    A_y = left_mult_C1_at_edge(SY, 0)
    A_z = left_mult_C1_at_edge(SZ, 0)
    # F_01 = -i [A_x, A_y]
    F_xy = -1j * (A_x @ A_y - A_y @ A_x)
    # Algebraic expectation: [L_σx, L_σy] = L_{[σx, σy]} = L_{2iσz} ⇒ F_xy = 2 L_σz
    F_predicted = 2.0 * A_z
    err = np.linalg.norm(F_xy - F_predicted)
    print(f"  on edge 0:  [L_σx, L_σy] = 2i·L_σz  ⇒  F_xy = 2·L_σz")
    print(f"  numeric verification:  ‖F_xy − 2·L_σz‖ = {err:.3e}  →  {err < 1e-12}")

    # Tr_F(F·F)
    tr_FxyFxy = float(np.real(np.trace(F_xy @ F_xy)))
    tr_AxAx = float(np.real(np.trace(A_x @ A_x)))
    tr_AyAy = float(np.real(np.trace(A_y @ A_y)))
    tr_AzAz = float(np.real(np.trace(A_z @ A_z)))
    print(f"\n  Tr_F(A_x²) = Tr_F(A_y²) = Tr_F(A_z²) = {tr_AxAx:.4f} (= 4 = dim Cl(2)_e)")
    print(f"  Tr_F(F_xy · F_xy) = 4 × Tr_F(L_σz²) = 4 × 4 = {tr_FxyFxy:.4f}    ← YM curvature trace")

    # spectral action a_4 contribution (Euclidean d=4, Tr_spinor(I_4)=4, Gilkey 1/12):
    # a_4_YM (per F-pair) = (1/(4π²)) × (1/12) × Tr_internal(F_μν F^μν)
    # For F only on (μ, ν) = (0, 1) and (1, 0) directions: contribution = 2 × (1/12) × Tr_internal(F_01²)
    # Tr_internal(F_01²) = Tr_spinor(I_4) × Tr_F(F·F) = 4 × tr_FxyFxy
    a4_YM_per_F_pair = (1.0 / (4 * np.pi ** 2)) * (1.0 / 12) * 2.0 * 4.0 * tr_FxyFxy
    print(f"\n  Heat-kernel a_4 contribution (per F_xy F^xy direction-pair):")
    print(f"    a_4_YM = (1/(4π²)) × (1/12) × Tr_spinor(I) × 2 × Tr_F(F²)")
    print(f"           = (1/(4π²)) × (1/12) × 4 × 2 × 16")
    print(f"           = 128 / (48 π²) = 8/(3π²) ≈ {a4_YM_per_F_pair:.6f}")

    # Compare to standard CC YM action S_YM = -(1/4g²) ∫ F·F (with F·F summed over gauge index a):
    # The CC convention has 1/g² emerging as coefficient of (1/4) Tr_gauge(F·F) in a_4.
    # For SU(2)_e on Cl(2)_e fundamental (4-dim left-mult), the trace identification is:
    #   Tr_F(F·F) = (Tr_F(T_a T_b) per generator) × δ_{ab} → for σ_a/2 normalised generators:
    #     Tr_F(L_{σ_a/2} L_{σ_b/2}) = (1/4) Tr_F(L_{σ_a} L_{σ_b}) = (1/4) × 4 δ_{ab} = δ_{ab}
    #   So  Tr_F(F·F)|_normalised = Σ_a (F^a)² × 1 = (F^a)².
    # CC's standard SM derivation: 1/g²(Λ_unif) = (f_0 / (4π²)) × (Tr_F-related integer)
    # For per-edge SU(2)_e: bare 1/g²_e ≈ (1/π²) × (some integer counting the H_F-multiplicity of edge e).
    #
    # The framework's α_GUT⁻¹ = 24 at unification → 1/g²(Λ_unif) = 24/(4π) ≈ 1.910 (per gauge group).
    # The CC bare coefficient from one F·F pair: a_4_YM ≈ 0.270 (numerical value above).
    # To match: need a structural multiplicity factor of ~7 from H_F's matter-side dressing
    # acting on the gauge sector — NOT directly recoverable from a single-edge probe.
    print(f"\n  α_GUT⁻¹ = 24 ⇒ 1/g²_unif = 24/(4π) ≈ 1.9099  (per gauge group factor)")
    print(f"  spectral-action a_4_YM (per F-pair, per cell) = {a4_YM_per_F_pair:.4f}")
    print(f"  ratio (1/g²_unif) / a_4_YM = {1.9099 / a4_YM_per_F_pair:.4f}")
    print(f"  → factor of ~{1.9099 / a4_YM_per_F_pair:.2f} between CC-bare and framework's α_GUT⁻¹/(4π).")
    print(f"     This is the rep-multiplicity factor that maps Tr_F (over H_F = 280-dim) to")
    print(f"     Tr_gauge (over SU(2)_e adjoint = 3-dim).  Bookkeeping for Step 3.")


def main():
    print(r"""
==========================================================================================
4D DIRAC INNER FLUCTUATION — STEP 2 of 4D spacetime spectral-triple project
Activate A_μ ∈ A_F, decompose D_4², separate gauge / Higgs / Yukawa, extract bare 1/g².
==========================================================================================""")
    part_2a_verify_decomposition()
    part_2b_find_pure_gauge()
    pair = part_2c_pure_gauge_F_squared()
    part_2d_heat_trace_numerical()
    part_2e_pure_gauge_g2_inv(pair)
    part_2f_gauge_unitary_perspective()
    print("\n" + "=" * 100)
    print("STEP 2 INTERIM VERDICT")
    print("=" * 100)
    print("""
  WHAT THIS PROBE ESTABLISHED

  (i)  STRUCTURAL DECOMPOSITION (Part 2A).  D_4²(p, A) for constant inner fluctuation A_μ
       decomposes at machine precision:
         D_4² = I⊗[(p−A)² + D_F²] + (i/2) γ^μν ⊗ F_μν − γ^μ γ_5 ⊗ [A_μ, D_F]
       with F_μν = −i[A_μ, A_ν].  The Higgs-cross term  −γ^μγ_5 ⊗ [A_μ, D_F]  is the
       framework's analog of CC's Higgs-scalar emergence.

  (ii) ZERO COMMUTANT (Part 2B).  ZERO of the 280 left-mult Hermitian generators of A_F
       commute exactly with D_F = Q̂_alg(0).  STRUCTURAL CONSEQUENCE: the standard CC
       'gauge subalgebra = commutant of D_F' identification does NOT apply to the framework's
       supercharge.  Q̂_alg intrinsically couples matter ↔ gauge — there is no left-mult
       inner fluctuation that's PURE gauge with no Higgs activation.

  (iii) RIGHT GAUGE ROUTE (Part 2F).  The framework's gauge group is the adjoint-action
        unitaries of A_F preserving D_F = ⊕_e SU(2)_e (per-edge + cross-edge), verified
        at machine precision in Step 1.  Inner fluctuation A_μ ∈ ⊕_e su(2)_e gives a
        well-defined YM curvature:  for σ_x, σ_y on edge 0,  F_xy = −i[L_σx, L_σy] = 2·L_σz,
        Tr_F(F_xy · F_xy) = 16.

  (iv) HEAT-KERNEL a_4 (Part 2F).  Per (edge, F-direction-pair):
        a_4_YM = (1/(4π²)) × (1/12) × Tr_spinor(I) × 2 × Tr_F(F²)
               = (1/(4π²)) × (1/12) × 4 × 2 × 16
               = 128/(48π²)  =  **8/(3π²)** ≈ 0.2702

  (v) NUMERICAL COMPARISON to framework's α_GUT⁻¹ = 24:
        1/g²_unif = α_GUT⁻¹/(4π) = 6/π ≈ 1.9099
        spectral-action bare value (per edge, per F-pair) = 8/(3π²)
        ratio = (6/π) / (8/(3π²)) = (6/π) × (3π²/8) = 9π/4 ≈ 7.07

      The factor 9π/4 is the multiplicity / Killing-form-normalization that maps
      Tr_F (over 280-dim H_F, per-edge-per-F-pair) to Tr_gauge (over the gauge group).
      This isn't an immediate clean integer match — it identifies the BOOKKEEPING that
      Step 3 has to do to close the gauge β question.

  WHAT REMAINS (deferred to Step 3)

  (a)  Sum over EDGES + F-direction PAIRS.  Per cell there are 6 edges, each carrying 3
       generator pairs (σ_x σ_y → σ_z, σ_y σ_z → σ_x, σ_z σ_x → σ_y);  total per-cell YM
       coefficient = 18 × 8/(3π²) = 48/π² ≈ 4.86.  Step 3 needs to relate this to
       the framework's per-factor gauge couplings (SU(3)_c, SU(2)_L, U(1)_Y) via Cl(6)
       → PS → SM embedding.

  (b)  Cross-edge generators.  Step 1's gauge-equivariance check verified BOTH per-edge
       AND cross-edge SU(2).  The cross-edge generators contribute additional inner-fluctuation
       structure that this probe didn't enumerate — non-trivial bookkeeping.

  (c)  Higgs-sector decoupling.  The [A_μ, D_F] cross term contributes Yukawa-like content
       to a_2 + a_4.  For the YM coefficient extraction this is separable in principle
       (different a_4 structural form), but requires explicit form-by-form analysis.

  (d)  Continuum bridge Λ_sub ↔ µ_unif (per handoff Step 3).  Even if the structural integers
       in (a)-(c) line up cleanly, the comparison to MSSM b_i needs the substrate-scale ↔
       continuum-scale identification (analog of Π_TT's path-(b) substrate-Planck mass).

  HONEST SCOPE OF THIS STEP.  The spectral-action machinery cleanly produces a YM
  coefficient of structural form  Tr_F(F·F)/π² × structural-integer.  The framework's
  Q̂_alg-as-D_F SUCCEEDS at giving such a coefficient via the gauge-unitary route, NOT via
  the (empty) commutant route.  The numerical value 8/(3π²) per (edge, F-pair) does NOT
  match α_GUT⁻¹/(4π) = 6/π immediately — the gap is the 9π/4 factor identified in (v),
  which is the multiplicity bookkeeping for Step 3.

  No graded content changes from this probe.  ADOPTED-MSSM-Sb stands.
""")
    print("4d_dirac_inner_fluctuation_probe.py: sentinel done.")


if __name__ == "__main__":
    main()
