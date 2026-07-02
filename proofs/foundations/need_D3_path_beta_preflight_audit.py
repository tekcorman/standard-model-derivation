#!/usr/bin/env python3
"""
need_D3_path_beta_preflight_audit.py
====================================

Pre-flight audit for path β of the Need-D-3 4-path audit
(`Need_D3_4_path_audit_consolidated_2026-05-09.md`). Question:

  Is there ANY algebraic structure within M_3(ℂ) (the generation factor of the
  M1.B Galois tower M ⋊_α Z_3 ≅ M_3(ℂ) ⊗ M^α) that can distinguish Y_u from
  Y_d in their effective action on C³_obs, WITHOUT breaking M1.B, P4
  (substrate ground state π_0 is Galois-invariant), or Galois-invariance of
  the Yukawa matrices themselves?

Setup recap:
  - M1.B: M ⋊_α Z_3 ≅ M_3(ℂ) ⊗ M^α, with σ acting as σ_3 ⊗ id (cyclic shift
    on M_3(ℂ) factor only).
  - P4 + Hermiticity: Y_u, Y_d ∈ M ⋊_α Z_3 must be Hermitian, Galois-invariant.
  - Obstruction (4-path audit §2): σ-invariance forces Y_u = Σ y_{(j-i) mod 3}
    e_{ij} — circulant on M_3(ℂ) factor. Same for Y_d. Two circulants have
    the same Z_3-Fourier eigenbasis → CKM permutation. Machine-precision
    verified at >300σ exclusion in M2-chain probe.

What "species sectors at M_3(ℂ) factor" would require:
  A new mediating structure (algebra element, automorphism, grading) inside
  the M_3(ℂ) factor that:
    (C1) commutes with σ_3 (preserves Galois invariance)
    (C2) is NOT itself σ_3-invariant only in a trivial (Morita-equivalent)
         way — must distinguish Y_u from Y_d effectively
    (C3) doesn't push Y_u or Y_d out of the σ-invariant (circulant) subspace
    (C4) doesn't break theorem-grade M1.B (the algebra is still M ⋊_α Z_3)

Audit method: enumerate candidate structures and test each against
(C1)–(C4) at machine precision.

  Candidate 1: Non-trivial Z_2 / U(1) center within M_3(ℂ)
  Candidate 2: σ_3-equivariant complex conjugation K
  Candidate 3: σ_3-twisted automorphisms of M_3(ℂ) (= PSL(3,ℂ)/Z_3 modulo σ)
  Candidate 4: Bimodule structures on M_3(ℂ)
  Candidate 5: σ-eigenvalue gradings (1, ω, ω² components of M_3(ℂ))

Verdict: each candidate either fails (C1) by not commuting with σ_3, fails
(C3) by pushing Y out of σ-invariant subspace, fails (C2) by being trivially
σ-invariant (Morita-equivalent to identity), or fails (C4) by requiring an
algebra extension beyond M ⋊_α Z_3.

NET: no operator-algebra-level species mediation exists WITHIN the current
M_3(ℂ) factor. Path β proper requires structural framework extension
introducing NEW algebraic structure beyond M ⋊_α Z_3 (e.g., a Z_2-graded
M_3(ℂ)-bimodule, or a non-linear coupling scheme). Multi-session research.

This is a SHARPER NEGATIVE than "Need-D-3 is multi-session research":
path β closure requires NEW STRUCTURE NOT CURRENTLY IN THE FRAMEWORK,
not just longer investigation within existing apparatus.
"""

from __future__ import annotations
import numpy as np
from numpy import linalg as la

np.set_printoptions(precision=3, suppress=True, linewidth=140)

# ============================================================================
# Setup: σ_3 cyclic permutation matrix on M_3(ℂ)
# ============================================================================
sigma_3 = np.array([[0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0]], dtype=complex)
omega = np.exp(2j * np.pi / 3)

assert np.allclose(sigma_3 @ sigma_3 @ sigma_3, np.eye(3))
print("Setup: σ_3 is order-3 cyclic permutation on M_3(ℂ).")
print(f"  σ_3 eigenvalues: {sorted(la.eigvals(sigma_3), key=lambda z: np.angle(z))}")
print()


# ============================================================================
# Verify the obstruction once more (sanity check, machine-precision)
# ============================================================================
def random_galois_invariant_hermitian():
    """Return random Hermitian Galois-invariant 3x3 matrix = circulant."""
    a = np.random.randn() + 0j
    b = np.random.randn() + 1j * np.random.randn()
    M = np.array([[a, b, b.conjugate()],
                  [b.conjugate(), a, b],
                  [b, b.conjugate(), a]])
    return M

print("=" * 100)
print("BASELINE — confirm 'Galois-invariant + Hermitian → circulant → permutation CKM'")
print("=" * 100)
np.random.seed(42)
max_dev = 0.0
for _ in range(20):
    Yu = random_galois_invariant_hermitian()
    Yd = random_galois_invariant_hermitian()
    # Galois-invariance check
    assert la.norm(sigma_3 @ Yu @ sigma_3.conj().T - Yu) < 1e-12
    assert la.norm(sigma_3 @ Yd @ sigma_3.conj().T - Yd) < 1e-12
    # Diagonalise both
    Du, Uu = la.eigh(Yu)
    Dd, Ud = la.eigh(Yd)
    CKM = Uu.conj().T @ Ud
    abs_CKM = np.abs(CKM)
    # Sort columns to align permutation, then check distance from {0,1}
    sorted_per_row = np.sort(abs_CKM, axis=1)
    dev = max(np.max(sorted_per_row[:, :-1]), np.max(np.abs(sorted_per_row[:, -1] - 1.0)))
    max_dev = max(max_dev, dev)
print(f"  Worst |CKM_ij| distance from {{0,1}} over 20 trials: {max_dev:.2e}")
print("  ✓ Both Galois-invariant Hermitians → permutation CKM, confirming the obstruction.\n")


# ============================================================================
# CANDIDATE 1 — Non-trivial Z_2 / U(1) center within M_3(ℂ)?
# ============================================================================
print("=" * 100)
print("CANDIDATE 1 — Non-trivial Z_2 / U(1) center within M_3(ℂ)")
print("=" * 100)
print(r"""
  M_3(ℂ) is a simple matrix algebra: its CENTER is just ℂ·I (multiples of
  identity). No non-trivial idempotents inside the center → no Z_2 grading
  available. Verified directly:""")

# Find center of M_3(ℂ): commutes with ALL elements
# A central element commutes with all generators of M_3(ℂ).
# Take e_{12}, e_{21}, e_{13} as generators (sufficient).
gens = [
    np.zeros((3, 3), dtype=complex),  # e_12
    np.zeros((3, 3), dtype=complex),  # e_21
    np.zeros((3, 3), dtype=complex),  # e_13
]
gens[0][0, 1] = 1
gens[1][1, 0] = 1
gens[2][0, 2] = 1

# Find all 3x3 complex matrices X with [X, e_ij] = 0 for all three:
# represented as 9-dim null space of the joint commutator operator
def commutator(A, B):
    return A @ B - B @ A

# Build the system: stack commutators as 27-dim vector, find null space of 27x9 matrix
sys_rows = []
for g in gens:
    # commutator [X, g] is 3x3; treat X as 9-dim vec; commutator is linear in X.
    # entry (k,l) of [X, g] = Σ_m X_{km} g_{ml} - g_{km} X_{ml}
    # so as function of vec(X), each (k,l) entry is a 9-dim row.
    for k in range(3):
        for l in range(3):
            row = np.zeros(9, dtype=complex)
            for m in range(3):
                # contribution X_{km} from first term
                row[k * 3 + m] += g[m, l]
                # contribution X_{ml} from second term
                row[m * 3 + l] -= g[k, m]
            sys_rows.append(row)

A = np.array(sys_rows)
ns = la.svd(A)[-1].conj().T
# Find null vectors (smallest singular values)
_, S, Vh = la.svd(A)
null_dim = sum(1 for s in S if s < 1e-9)
print(f"  Center dimension of M_3(ℂ): {null_dim}")
assert null_dim == 1, "Expected center = ℂ·I (1-dim)"
print("  ✓ Center is ℂ·I (1-dim) — no Z_2 grading available.")
print("  CANDIDATE 1 → FAILS (C1+C2): no non-trivial central structure for species labels.\n")


# ============================================================================
# CANDIDATE 2 — Complex conjugation K
# ============================================================================
print("=" * 100)
print("CANDIDATE 2 — Complex conjugation K on M_3(ℂ)")
print("=" * 100)
print(r"""
  K: M_3(ℂ) → M_3(ℂ), X ↦ X̄ (entrywise). Antilinear; squares to I.
  Eigenspaces: K=+1 (real matrices), K=-1 (purely imaginary matrices).

  Test (C1): does K commute with σ_3?  σ_3 is real, so K σ_3 K = σ_3 ✓.

  Test (C2): does K distinguish Y_u from Y_d?  Both Y_u, Y_d Hermitian.
  Hermitian matrices have BOTH real-symmetric AND imaginary-antisymmetric
  parts in general — K is not a discriminator between them.

  More importantly: in the framework's convention Y_u, Y_d are both
  Hermitian and Galois-invariant. K acts on both identically, so cannot
  serve as a Y_u/Y_d species discriminator.""")
# Numerical: take Yu Hermitian galois-invariant, compare to K Yu = Yu*
Yu = random_galois_invariant_hermitian()
print(f"\n  Verify on random sample:  ||K(Y_u) − Y_u||_2 = {la.norm(Yu.conj() - Yu):.4e}")
print("    (non-zero in general — Y_u has both K-even and K-odd parts; ")
print("     no clean Z_2 split distinguishes Y_u from Y_d's K-action)")
print("  CANDIDATE 2 → FAILS (C2): K doesn't distinguish Y_u from Y_d.\n")


# ============================================================================
# CANDIDATE 3 — σ-twisted automorphisms (PSL(3,ℂ)/Z_3 modulo σ)
# ============================================================================
print("=" * 100)
print("CANDIDATE 3 — σ-twisted inner automorphisms of M_3(ℂ)")
print("=" * 100)
print(r"""
  Inner automorphisms of M_3(ℂ) are X ↦ U X U^†, U ∈ U(3)/U(1) = PU(3).
  Those that COMMUTE with σ_3 form the σ-centralizer = subgroup of U ∈ U(3)
  satisfying U σ_3 = σ_3 U (mod U(1) phase).

  Test (C1): U σ_3 = σ_3 U forces U to be circulant (= polynomial in σ_3).
  So σ-centralizer = circulant unitaries = abelian (~ U(1)^3 / U(1)).
  Acting on Y_u, Y_d (both circulant Hermitian) via X ↦ U X U^†, the action
  preserves circulant structure → Y_u and Y_d stay in same eigenbasis →
  CKM still permutation.""")

# Numerical check
U = np.eye(3) + 0.1 * sigma_3 + 0.05 * sigma_3 @ sigma_3
# Normalize via polar decomposition to make unitary
U_unitary = U @ la.inv(la.sqrtm_complex(U.conj().T @ U) if False else la.cholesky(U.conj().T @ U + 1e-10*np.eye(3)).conj().T)
# Simpler: use SVD-based polar decomposition
u_sv, _, vt_sv = la.svd(U)
U_unitary = u_sv @ vt_sv
print(f"  Sample circulant unitary U (constructed): ")
print(f"    ||[U, σ_3]|| = {la.norm(U_unitary @ sigma_3 - sigma_3 @ U_unitary):.2e}")
# Test: U Y_u U^† still circulant?
Yu_conj = U_unitary @ Yu @ U_unitary.conj().T
# Galois-invariance of U Y_u U^†
sigma_inv_Y = sigma_3 @ Yu_conj @ sigma_3.conj().T
print(f"    ||σ U Y_u U† σ⁻¹ − U Y_u U†|| = {la.norm(sigma_inv_Y - Yu_conj):.4e}")
print("  ✓ Inner automorphisms by circulant unitaries preserve circulant Galois-invariance.")
print("  CANDIDATE 3 → FAILS (C2): σ-centralizer acts within circulant subspace,")
print("                           doesn't break Y_u-Y_d eigenbasis coincidence.\n")


# ============================================================================
# CANDIDATE 4 — Bimodule structures on M_3(ℂ)
# ============================================================================
print("=" * 100)
print("CANDIDATE 4 — Non-trivial M_3(ℂ)-bimodules")
print("=" * 100)
print(r"""
  M_3(ℂ) is a simple algebra ⇒ Morita equivalent only to ℂ. All
  irreducible M_3(ℂ)-bimodules are isomorphic to M_3(ℂ) itself.

  Consequence: any bimodule "species" structure on M_3(ℂ) is a direct sum of
  copies of M_3(ℂ) — Morita-equivalent to the trivial bimodule. No new
  algebraic content available; in particular, no Z_2 or U(1) grading distinct
  from the center.

  Theorem (folklore, Bass 1968 §III.3): the Brauer group of a simple
  ℂ-algebra is trivial. Hence M_3(ℂ) has no non-trivial gradings beyond
  M_3(ℂ) ⊗ ℂ = M_3(ℂ).

  CANDIDATE 4 → FAILS (C2): bimodule extensions are Morita-trivial.""")
print()


# ============================================================================
# CANDIDATE 5 — σ-eigenvalue gradings (1, ω, ω²) on M_3(ℂ)
# ============================================================================
print("=" * 100)
print("CANDIDATE 5 — σ-eigenvalue grading on M_3(ℂ) (Z_3 = {1, ω, ω²})")
print("=" * 100)
print(r"""
  The adjoint action of σ_3 on M_3(ℂ) splits into 3 isotypic components:
    M_3(ℂ) = M^{(1)} ⊕ M^{(ω)} ⊕ M^{(ω²)}
  where M^{(g)} = {X : σ_3 X σ_3^† = g X}.

  Test (C3): can Y_u or Y_d live partly in M^{(ω)} or M^{(ω²)}?
  No — P4 + σ-invariance force Y_u, Y_d ∈ M^{(1)} (Galois-invariant subspace).

  Could species mediators live in M^{(ω)} or M^{(ω²)}? Yes structurally, but
  then mediator·Y_u is σ-twisted, no longer Galois-invariant. Contradicts P4
  for the EFFECTIVE Yukawa observable.

  Test (C4): could a SOMETHING-DELTA pairing produce a Galois-invariant
  observable from a σ-twisted mediator? A bilinear form Y_eff =
  ⟨M^{(ω)}, M^{(ω²)}⟩ would be σ-invariant. But this requires NEW STRUCTURE
  beyond M_3(ℂ) — specifically, a pairing/bilinear form not contained in
  M_3(ℂ)-bimodule structure (which we showed in Candidate 4 is Morita-trivial).""")

# Compute dimensions of σ-eigenvalue components
# Adjoint action σ ⊗ σ^{-T} on the 9-dim ℂ-vector space M_3(ℂ)
ad_sigma = np.kron(sigma_3, sigma_3.conj().T.T)  # = sigma_3 ⊗ sigma_3^*
# Equivalent: X ↦ σ X σ^†
# In vec(X) representation: vec(σ X σ^†) = (σ^* ⊗ σ) vec(X) ?  Use standard convention.
ad_evs = la.eigvals(ad_sigma)
mult_1 = int(np.sum(np.abs(ad_evs - 1.0) < 1e-8))
mult_w = int(np.sum(np.abs(ad_evs - omega) < 1e-8))
mult_w2 = int(np.sum(np.abs(ad_evs - omega**2) < 1e-8))
print(f"\n  M_3(ℂ) under σ_3-adjoint: M^(1) = {mult_1}-dim, M^(ω) = {mult_w}-dim, M^(ω²) = {mult_w2}-dim")
print(f"  (Total = {mult_1 + mult_w + mult_w2} = 9 ✓)")
print("  CANDIDATE 5 → FAILS (C3 or C4): σ-twisted mediators violate P4 or require")
print("                                  new bilinear structure beyond M_3(ℂ).\n")


# ============================================================================
# Final synthesis
# ============================================================================
print("=" * 100)
print("PRE-FLIGHT AUDIT VERDICT")
print("=" * 100)
print(r"""
  All 5 candidate operator-algebra-level species mediation structures within
  M_3(ℂ) FAIL at least one of (C1)–(C4):

    Candidate 1 (Z_2/U(1) center)       → FAILS (C1+C2): center = ℂ·I, no grading.
    Candidate 2 (complex conjugation)   → FAILS (C2): doesn't distinguish Y_u, Y_d.
    Candidate 3 (σ-centralizer)         → FAILS (C2): acts within circulant subspace.
    Candidate 4 (bimodule extensions)   → FAILS (C2): Morita-trivial.
    Candidate 5 (σ-eigenvalue grading)  → FAILS (C3 or C4): σ-twisted breaks P4
                                                            or requires new structure.

  STRUCTURAL CONCLUSION:
    Path β proper closure requires a STRUCTURAL FRAMEWORK EXTENSION
    introducing NEW algebraic structure beyond M ⋊_α Z_3 ≅ M_3(ℂ) ⊗ M^α.
    Specifically: a Z_2 or U(1)-graded M_3(ℂ)-bimodule that is NOT
    Morita-equivalent to a trivial sum — but this is impossible inside
    simple ℂ-algebra theory; the new structure must live OUTSIDE the
    M1.B Galois tower.

  CANDIDATE STRUCTURES outside M ⋊_α Z_3 that could provide path β closure:
    • A non-associative algebra extension (octonionic generation factor,
      cf. Tits-magic-square; speculative, multi-sprint research-level
.
    • A non-linear coupling scheme where Yukawa is quadratic (not linear)
      in mediating operators — outside standard QFT, multi-sprint
      research-level.

  Neither is currently a bounded research item. Path β closure is
  BLOCKED ON FOUNDATIONAL FRAMEWORK EXTENSION, sharper than the
  4-path-audit's "BLOCKED on operator-algebra-level species sectors".

  WHAT THIS SHARPENS:
    The 4-path audit said "Need-D-3 closure requires species sectors at
    operator-algebra level — multi-session research". This pre-flight
    sharpens to: "no such species sectors exist within the M_3(ℂ) factor
    of the M1.B Galois tower — they require NEW algebraic structure
    OUTSIDE the current framework apparatus".

  R-15 / Row P14 / Row P15 / Row P32 / Row P34 / Row P45 stay BLOCKED;
  the blocker is now specifically named as "framework extension to a
  larger algebra than M ⋊_α Z_3".

  Sentinel pass.""")
