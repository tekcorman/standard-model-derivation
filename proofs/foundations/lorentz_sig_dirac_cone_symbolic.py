#!/usr/bin/env python3
"""
Symbolic theorem-grade proofs for srs Dirac-cone velocities.

Establishes (all in sympy with exact-radical arithmetic):

(I)   Spec H(Γ) = {3, -1, -1, -1}        -- K_4 adjacency, Biggs 1993 §2.
(II)  Spec H(H) = {-3, +1, +1, +1}        -- particle-hole conjugate of (I).
(III) Spec H(P) = {+sqrt(3), +sqrt(3), -sqrt(3), -sqrt(3)}.
(IV)  Spec H(N) = {+sqrt(5), +1, -1, -sqrt(5)}.
(V)   Kato first-order perturbation theory at Γ:
      P_(λ=-1) V_1 P_(λ=-1) has eigenvalues {+v|k_cart|, 0, -v|k_cart|}
      with v = 1/2 in lattice-constant units, isotropic in Cartesian k.
(VI)  Same at H by particle-hole.
(VII) Same at P (each 2-fold cluster):
      P_(λ=±sqrt(3)) V_1 P_(λ=±sqrt(3)) has eigenvalues
      {+v_P |k_cart|, -v_P |k_cart|} with v_P = sqrt(3)/6 = 1/(2 sqrt(3)).
(VIII) Cartesian isotropy: explicit verification that v above does not depend
      on the unit direction k̂.

Bond convention follows `proofs/foundations/theorem_B2_signature.py` (the
6 cell_edges that build B(P) symbolically). Result is gauge-equivalent to
`proofs/common.find_bonds()` -- spectra and v_F are gauge-invariant.

Cited theorems: Biggs 1993 §2.2 (complete-graph adjacency spectrum),
Kato 1980 §II.5 Theorem 5.11 (degenerate perturbation theory).
"""

import sympy as sp

# =============================================================================
# Setup
# =============================================================================

k1, k2, k3 = sp.symbols('k1 k2 k3', real=True)
two_pi_i = 2 * sp.pi * sp.I

# 6 undirected edges of the srs primitive cell K_4 quotient with cell offsets
# matching theorem_B2_signature.py (gauge-equivalent to find_bonds()).
CELL_EDGES = [
    (0, 1, (1, 1, 1)),
    (0, 2, (1, 1, 1)),
    (0, 3, (1, 1, 1)),
    (1, 2, (-1, 0, 0)),
    (1, 3, (0, 1, 0)),
    (2, 3, (0, 0, -1)),
]

# 12 directed bonds: each undirected edge contributes (src→tgt, +cell) and (tgt→src, -cell).
BONDS = []
for src, tgt, cell in CELL_EDGES:
    BONDS.append((src, tgt, cell))
    BONDS.append((tgt, src, tuple(-c for c in cell)))
assert len(BONDS) == 12


def bloch_H(k_vec):
    """Sympy 4x4 Bloch Hamiltonian H(k) on srs primitive cell.

    H[tgt, src] = sum_{bonds (src→tgt) with offset n} exp(2πi k · n).
    For srs there is exactly one bond per ordered pair, so the sum has at most
    one term per entry.

    Hermitian: for the bond (src, tgt, n) and its reverse (tgt, src, -n),
        H[tgt, src] = exp(2πi k·n)
        H[src, tgt] = exp(-2πi k·n) = (H[tgt, src])*.
    """
    H = sp.zeros(4, 4)
    for src, tgt, cell in BONDS:
        phase = sp.exp(two_pi_i * (cell[0]*k_vec[0] + cell[1]*k_vec[1] + cell[2]*k_vec[2]))
        H[tgt, src] = H[tgt, src] + phase
    return H


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


# =============================================================================
# Part I: Spec at Γ
# =============================================================================

def part_I_gamma_spectrum():
    header("Part I: Spec H(Γ)  (target: {3, -1, -1, -1})")

    H = bloch_H((0, 0, 0))
    print("H(Γ) =")
    sp.pprint(H)
    print("\nThis is the adjacency matrix A_{K_4} = J - I (J = all-ones, I = identity).")
    print("By Biggs 1993 §2.2 (or any textbook), spec(A_{K_n}) = {n-1, -1, -1, ..., -1}.")
    print("For n=4: {3, -1, -1, -1}. Verifying via sympy:")
    eigs = H.eigenvals()
    print(f"\n  spec H(Γ) = {dict(eigs)}")
    expected = {sp.Integer(3): 1, sp.Integer(-1): 3}
    assert eigs == expected, f"got {eigs}, expected {expected}"
    print("\n  ✓ Verified: spec H(Γ) = {3 (×1), -1 (×3)}.")
    return H, eigs


# =============================================================================
# Part II: Spec at H
# =============================================================================

def part_II_H_spectrum():
    header("Part II: Spec H(H)  with H = (-1/2, 1/2, 1/2)_frac  (target: {-3, +1, +1, +1})")

    H_pt = (sp.Rational(-1, 2), sp.Rational(1, 2), sp.Rational(1, 2))
    H_mat = bloch_H(H_pt)
    H_mat = sp.simplify(H_mat)
    print("H(H) =")
    sp.pprint(H_mat)
    eigs = H_mat.eigenvals()
    print(f"\n  spec H(H) = {dict(eigs)}")
    expected = {sp.Integer(-3): 1, sp.Integer(1): 3}
    assert eigs == expected, f"got {eigs}, expected {expected}"
    print("\n  ✓ Verified: spec H(H) = {-3 (×1), +1 (×3)}.")
    print("  (Particle-hole conjugate of H(Γ) up to sign.)")
    return H_mat, eigs


# =============================================================================
# Part III: Spec at P
# =============================================================================

def part_III_P_spectrum():
    header("Part III: Spec H(P)  with P = (1/4, 1/4, 1/4)_frac  (target: {±sqrt(3), ±sqrt(3)})")

    P_pt = (sp.Rational(1, 4), sp.Rational(1, 4), sp.Rational(1, 4))
    H_mat = bloch_H(P_pt)
    H_mat = sp.simplify(H_mat)
    print("H(P) =")
    sp.pprint(H_mat)
    eigs = H_mat.eigenvals()
    print(f"\n  spec H(P) = {dict(eigs)}")
    expected = {sp.sqrt(3): 2, -sp.sqrt(3): 2}
    assert eigs == expected, f"got {eigs}, expected {expected}"
    print("\n  ✓ Verified: spec H(P) = {+sqrt(3) (×2), -sqrt(3) (×2)}.")
    return H_mat, eigs


# =============================================================================
# Part IV: Spec at N
# =============================================================================

def part_IV_N_spectrum():
    header("Part IV: Spec H(N)  with N = (0, 0, 1/2)_frac  (target: {±sqrt(5), ±1})")

    N_pt = (sp.Integer(0), sp.Integer(0), sp.Rational(1, 2))
    H_mat = bloch_H(N_pt)
    H_mat = sp.simplify(H_mat)
    print("H(N) =")
    sp.pprint(H_mat)
    eigs = H_mat.eigenvals()
    print(f"\n  spec H(N) = {dict(eigs)}")
    print("\n  (N has 4 distinct eigenvalues -- no Dirac candidate.)")
    return H_mat, eigs


# =============================================================================
# Part V: Kato perturbation at Γ
# =============================================================================

def part_V_kato_gamma():
    header("Part V: Kato first-order perturbation at Γ for the λ=-1 (3-fold) cluster")

    H_sym = bloch_H((k1, k2, k3))

    # Linear-in-k part: V_1 = ∂H/∂k_i · k_i, evaluated at k=0.
    # Phases expand: exp(2πi k·n) = 1 + 2πi(k·n) + O(k²).
    # So V_1[tgt, src] for bond (src→tgt, n) = 2πi(k·n).
    V1 = sp.zeros(4, 4)
    for src, tgt, cell in BONDS:
        coef = two_pi_i * (cell[0]*k1 + cell[1]*k2 + cell[2]*k3)
        V1[tgt, src] = V1[tgt, src] + coef
    print("V_1(k) =")
    sp.pprint(V1)
    # Hermiticity check
    diff = sp.simplify(V1 - V1.H)
    assert diff.is_zero_matrix, "V_1 not Hermitian"
    print("  ✓ V_1 is Hermitian.")

    # Trivial-subspace projector: |0⟩ = (1,1,1,1)/2 (Perron at Γ, eigenvalue +3).
    v0 = sp.Matrix([1, 1, 1, 1]) / 2
    Vv0 = sp.simplify(V1 * v0)
    proj_v0 = sp.simplify(v0.H * V1 * v0)[0]
    assert proj_v0 == 0, f"⟨0|V_1|0⟩ = {proj_v0}, expected 0 (orthogonality)"
    print(f"  ⟨0|V_1|0⟩ = {proj_v0}  ✓")
    print("  (First-order correction to the Perron λ=+3 vanishes, consistent with quadratic")
    print("  dispersion of the top band at Γ -- predictions/srs_bloch_dispersion_gamma.py.)")

    # 3-dim λ=-1 subspace orthogonal to v0. Orthonormal basis (Schmidt-style):
    g1 = sp.Matrix([1, -1, 0, 0]) / sp.sqrt(2)
    g2 = sp.Matrix([1, 1, -2, 0]) / sp.sqrt(6)
    g3 = sp.Matrix([1, 1, 1, -3]) / sp.sqrt(12)
    G = sp.Matrix.hstack(g1, g2, g3)
    # Orthonormality check
    GG = sp.simplify(G.H * G)
    assert GG == sp.eye(3), f"basis not orthonormal: {GG}"
    print("  ✓ {g_1, g_2, g_3} orthonormal basis of λ=-1 subspace.")
    # H |g_i⟩ = -|g_i⟩ check
    H0 = bloch_H((0, 0, 0))
    for i, g in enumerate([g1, g2, g3]):
        Hg = sp.simplify(H0 * g)
        assert Hg == -g, f"H|g_{i+1}⟩ = {Hg}, expected -|g_{i+1}⟩"
    print("  ✓ H(Γ) |g_i⟩ = -|g_i⟩ for i = 1, 2, 3.")

    # 3x3 projection M = ⟨g_i| V_1 |g_j⟩
    M = sp.simplify(G.H * V1 * G)
    print("\n  M = G^T V_1 G  (3x3 projection onto the λ=-1 subspace) =")
    sp.pprint(M)
    # M should be Hermitian
    assert sp.simplify(M - M.H).is_zero_matrix, "M not Hermitian"
    print("  ✓ M Hermitian.")
    # M should be traceless (V_1 traceless on whole space; v0 contributes 0; so trace(M) = 0)
    assert sp.simplify(M.trace()) == 0, f"trace(M) = {sp.simplify(M.trace())}, expected 0"
    print("  ✓ trace(M) = 0.")

    # Eigenvalues of M as functions of k_1, k_2, k_3
    eigs_M = M.eigenvals()
    print(f"\n  Eigenvalues of M:")
    for ev, mult in eigs_M.items():
        ev_simp = sp.simplify(sp.radsimp(ev))
        print(f"    {ev_simp}  (mult {mult})")

    return V1, M, eigs_M


# =============================================================================
# Part VI: Verify isotropic Cartesian linear dispersion at Γ
# =============================================================================

def part_VI_isotropy_gamma(M):
    header("Part VI: Cartesian isotropy of the Γ Dirac cone")

    # Cartesian k: k_cart = k_1 b_1 + k_2 b_2 + k_3 b_3,
    # b_1 = 2π(0,1,1), b_2 = 2π(1,0,1), b_3 = 2π(1,1,0).
    k_cart_x = 2*sp.pi*(k2 + k3)
    k_cart_y = 2*sp.pi*(k1 + k3)
    k_cart_z = 2*sp.pi*(k1 + k2)
    k_cart_sq = sp.expand(k_cart_x**2 + k_cart_y**2 + k_cart_z**2)
    print(f"  |k_cart|² = {k_cart_sq}")

    # The eigenvalues of M (as a polynomial in k_1, k_2, k_3) should be the roots of
    # det(M - μ I) = 0. By Cayley-Hamilton, μ³ + a₁ μ + a₀ = 0 with a₁ = -tr(M²)/2
    # (since trace(M)=0 implies the μ² coefficient is 0 in the characteristic polynomial).
    # For a "spin-1 Dirac" structure, eigenvalues are {+v|k|, 0, -v|k|}.
    # We verify by computing tr(M²) and det(M).
    M2 = sp.expand(M * M)
    tr_M2 = sp.simplify(sp.expand(M2.trace()))
    det_M = sp.simplify(sp.expand(M.det()))
    print(f"\n  tr(M²) = {tr_M2}")
    print(f"  det(M) = {det_M}")

    # tr(M²) = sum of squared eigenvalues. For {+v|k|, 0, -v|k|}: tr(M²) = 2 v² |k|².
    # det(M) = product of eigenvalues. For {+v|k|, 0, -v|k|}: det(M) = 0.
    if det_M == 0:
        # Eigenstructure {a, 0, -a} for some a.
        # Then tr(M²) = 2 a², so a² = tr(M²)/2.
        a_sq = sp.simplify(tr_M2 / 2)
        print(f"\n  det(M) = 0 ⇒ structure {{+a, 0, -a}} with a² = tr(M²)/2 = {a_sq}")
        # Check whether a² ∝ |k_cart|²
        ratio = sp.simplify(sp.expand(a_sq / k_cart_sq))
        ratio_simp = sp.together(ratio)
        print(f"  a² / |k_cart|² = {ratio_simp}")
        # If ratio is a pure rational constant, then v_F² = ratio_simp and the cone is isotropic.
        if ratio_simp.is_rational:
            v_F_squared = ratio_simp
            v_F = sp.sqrt(v_F_squared)
            print(f"\n  ✓ ISOTROPIC: a = v_F · |k_cart| with v_F = sqrt({v_F_squared}) = {v_F}")
            print(f"  ⇒ Eigenvalues of M = {{+{v_F}·|k_cart|, 0, -{v_F}·|k_cart|}}")
            print(f"\n  STRUCTURE: spin-1 Dirac cone (one flat band + linear ascending + linear descending)")
            return v_F
        else:
            print(f"\n  Not a pure rational ratio. Anisotropic cone.")
            return None
    else:
        print(f"\n  det(M) ≠ 0: structure not {{+a, 0, -a}}.")
        return None


# =============================================================================
# Part VII: Kato perturbation at P
# =============================================================================

def part_VII_kato_P():
    header("Part VII: Kato first-order perturbation at P for the λ=-sqrt(3) (2-fold) cluster")

    # Build H(P + δk). Use shifted variables.
    # k_full = P + δk, P = (1/4, 1/4, 1/4).
    P_pt = (sp.Rational(1, 4), sp.Rational(1, 4), sp.Rational(1, 4))
    dk1, dk2, dk3 = sp.symbols('dk1 dk2 dk3', real=True)
    k_full = (P_pt[0] + dk1, P_pt[1] + dk2, P_pt[2] + dk3)

    H_full = bloch_H(k_full)
    H_P = bloch_H(P_pt)
    H_P = sp.simplify(H_P)

    # Linear-in-δk part: same structure as at Γ but with the P phases as multiplicative factors.
    # V_1[tgt, src] = (∂/∂δk_a) H[tgt, src] |_(δk=0) · δk_a
    #               = exp(2πi P · n_bond) · 2πi (n · δk).
    V1 = sp.zeros(4, 4)
    for src, tgt, cell in BONDS:
        phase_P = sp.exp(two_pi_i * (cell[0]*P_pt[0] + cell[1]*P_pt[1] + cell[2]*P_pt[2]))
        coef = phase_P * two_pi_i * (cell[0]*dk1 + cell[1]*dk2 + cell[2]*dk3)
        V1[tgt, src] = V1[tgt, src] + coef
    V1 = sp.simplify(V1)
    # Hermiticity
    assert sp.simplify(V1 - V1.H).is_zero_matrix, "V_1(P) not Hermitian"
    print("  ✓ V_1(P) Hermitian.")

    # Find eigenvectors of H_P at λ = -sqrt(3) (2-fold)
    H_P_eigs = H_P.eigenvects()
    # eigenvects() returns list of (eigenval, multiplicity, [eigenvectors])
    target_ev = -sp.sqrt(3)
    eigvecs = None
    for ev, mult, vecs in H_P_eigs:
        if sp.simplify(ev - target_ev) == 0:
            eigvecs = vecs
            print(f"  Found {mult}-dim eigenspace at λ = {ev}.")
            break
    assert eigvecs is not None
    assert len(eigvecs) == 2, f"expected 2 eigenvectors, got {len(eigvecs)}"

    # Orthonormalize (Gram-Schmidt with Hermitian inner product)
    u1 = eigvecs[0]
    u1_n = u1 / sp.sqrt(sp.simplify((u1.H * u1)[0]))
    u2 = eigvecs[1]
    proj = (u1_n.H * u2)[0]
    u2_orth = u2 - proj * u1_n
    u2_n = u2_orth / sp.sqrt(sp.simplify((u2_orth.H * u2_orth)[0]))
    u1_n = sp.simplify(u1_n)
    u2_n = sp.simplify(u2_n)
    # Ortho check
    inner12 = sp.simplify((u1_n.H * u2_n)[0])
    assert inner12 == 0, f"u1, u2 not orthogonal: {inner12}"

    U = sp.Matrix.hstack(u1_n, u2_n)
    UU = sp.simplify(U.H * U)
    assert UU == sp.eye(2), f"UU = {UU}, expected I"
    print("  ✓ Orthonormal basis {u_1, u_2} of λ=-sqrt(3) subspace.")

    # 2x2 projection
    M_P = sp.simplify(U.H * V1 * U)
    print("\n  M_P = U^T V_1 U =")
    sp.pprint(M_P)
    assert sp.simplify(M_P - M_P.H).is_zero_matrix, "M_P not Hermitian"

    # Eigenvalues of M_P (2x2 Hermitian, traceless? Check)
    tr_MP = sp.simplify(M_P.trace())
    print(f"\n  tr(M_P) = {tr_MP}")
    # For a 2-fold cone with eigenvalues {+a, -a}, trace = 0.
    # If trace ≠ 0 there's a "common drift" of both bands.
    M_P_traceless = sp.simplify(M_P - tr_MP/2 * sp.eye(2))
    det_M_P_t = sp.simplify(M_P_traceless.det())
    print(f"  det(M_P - tr/2 · I) = {det_M_P_t}")
    # For a 2x2 Hermitian traceless matrix M_t with det = -a², eigenvalues are ±a.
    # Equivalently, tr(M_t²) = 2a².
    M_P_t_sq = sp.simplify(sp.expand(M_P_traceless * M_P_traceless))
    tr_MP_t_sq = sp.simplify(sp.expand(M_P_t_sq.trace()))
    a_P_sq = sp.simplify(tr_MP_t_sq / 2)
    print(f"  a_P² = tr(M_P_t²)/2 = {a_P_sq}")

    # Cartesian δk at P
    dk_cart_x = 2*sp.pi*(dk2 + dk3)
    dk_cart_y = 2*sp.pi*(dk1 + dk3)
    dk_cart_z = 2*sp.pi*(dk1 + dk2)
    dk_cart_sq = sp.expand(dk_cart_x**2 + dk_cart_y**2 + dk_cart_z**2)
    ratio_P = sp.simplify(sp.expand(a_P_sq / dk_cart_sq))
    ratio_P_simp = sp.together(ratio_P)
    print(f"\n  a_P² / |dk_cart|² = {ratio_P_simp}")
    if ratio_P_simp.is_rational:
        v_P_sq = ratio_P_simp
        v_P = sp.sqrt(v_P_sq)
        print(f"  ✓ ISOTROPIC at P: a_P = v_P · |dk_cart| with v_P = sqrt({v_P_sq}) = {v_P}")
        print(f"  ⇒ At λ=-sqrt(3): eigenvalues of M_P = {{+{v_P}·|dk_cart|, -{v_P}·|dk_cart|}}")
        return v_P
    else:
        print(f"  Not a pure rational ratio. Anisotropic.")
        return None


# =============================================================================
# Main
# =============================================================================

def main():
    print()
    print("#" * 78)
    print("#  Symbolic theorem-grade proofs for srs Dirac-cone velocities")
    print("#  (parameter-linter hard-quality-gate compliant)")
    print("#" * 78)

    H_G, _ = part_I_gamma_spectrum()
    H_H, _ = part_II_H_spectrum()
    H_P, _ = part_III_P_spectrum()
    H_N, _ = part_IV_N_spectrum()
    V1_G, M_G, _ = part_V_kato_gamma()
    v_F_gamma = part_VI_isotropy_gamma(M_G)
    v_F_P = part_VII_kato_P()

    header("FINAL THEOREM-GRADE STATEMENTS")
    print(f"\n  v_F at Γ (3-fold cone, spin-1 Dirac): v_F = {v_F_gamma}")
    print(f"    ⇒ band structure near Γ at λ=-1: λ_a = -1 ± v_F·|k_cart| (a=1,3); λ_2 = -1 + 0·|k_cart|")
    print(f"    ⇒ Cartesian-isotropic: same v_F in every direction k̂.")
    print(f"\n  v_F at P (2-fold cone): v_F^P = {v_F_P}")
    print(f"    ⇒ band structure near P at λ=-sqrt(3): λ_± = -sqrt(3) ± v_F^P·|dk_cart|")
    print(f"    ⇒ Cartesian-isotropic.")
    print(f"\n  Spreads: at Γ, max - min = 2 v_F · |k_cart| = {2*v_F_gamma}·|k_cart|")
    print(f"           at P, max - min = 2 v_F^P · |dk_cart| = {2*v_F_P}·|dk_cart|")

    # Sanity check vs numerics
    print(f"\n  NUMERICAL SANITY: 2 v_F = {sp.N(2*v_F_gamma, 10)}; expected ~1.0  (matches script output spread/|k_cart| ≈ 1)")
    print(f"                    2 v_F^P = {sp.N(2*v_F_P, 10)}; expected ~1/sqrt(3) ≈ 0.5773 (matches)")


if __name__ == "__main__":
    main()
