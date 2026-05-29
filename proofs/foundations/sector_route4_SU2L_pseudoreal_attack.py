#!/usr/bin/env python3
"""
Route 4 attack: SU(2)_L Higgs doublet partner mechanism for Y_u ≠ Y_d.

CONTEXT
=======
Route 4 (identified EOD+3 Need-D audit, `Need_D_audit_and_SU2L_higgs_partner_candidate_2026-05-05.md`):
  Claim: Y_u and Y_d differ by conjugate-SU(2)_L representation (H̃ vs H),
  inheriting framework's edge-qubit Cl(0,2) ≅ ℍ structure. If C³_gen carries
  SU(2)_L action lifted from edge qubit, Y_u and Y_d eigenbases on C³_gen
  differ → derivable non-trivial CKM.

User said "attack route 4" — direct attempt rather than audit-only.

CRITICAL TEST (the audit framing was incomplete): SU(2) is pseudoreal —
the fundamental and conjugate representations are EQUIVALENT via the iσ_2
intertwiner. If H and H̃ are SAME SU(2)_L rep (just intertwined), then the
"different rep" framing of Route 4 is structurally incorrect.

This probe:
1. Verifies SU(2) pseudoreal structure rigorously (machine-precision).
2. Identifies what the actual Y_u vs Y_d distinction is (HYPERCHARGE U(1)_Y).
3. Audits whether framework can distinguish u from d WITHOUT full U(1)_Y
   derivation (G2-D blocked).
4. Tests alternative bridges: edge-qubit Cl(0,2) ≅ ℍ structure gives a
   COMPLEX CONJUGATION distinction between H and H̃ components — does this
   lift to a non-trivial Y_u/Y_d eigenbasis distinction on C³_gen?
5. Honest verdict.

EXPECTED OUTCOME
================
Route 4's "different SU(2)_L rep" framing is structurally INCORRECT —
H ≡ H̃ as SU(2)_L reps via iσ_2. The actual distinction is U(1)_Y
hypercharge, which is BLOCKED in the framework (G2-D, theorem_g2_edge_qubit_su2 §7).

Route 4 is therefore BLOCKED on G2-D, NOT Need-A2 as claimed in the EOD+3
audit. This is a sharper closure target.

The complex-conjugation distinction (H vs H̃ components h⁰ vs h⁰*) gives a
SIGN-FLIP on Cl(0,2) generators but doesn't directly give a structural
eigenbasis rotation on C³_gen without additional content.
"""

from __future__ import annotations

import math
import numpy as np

TOL = 1e-12

# ============================================================================
# 1. Define SU(2)_L Higgs doublet H and conjugate H̃
# ============================================================================
print("=" * 78)
print("Step 1: SU(2)_L Higgs doublet H and conjugate H̃")
print("=" * 78)
print()

# Pauli matrices
sigma_1 = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_3 = np.array([[1, 0], [0, -1]], dtype=complex)
i_sigma_2 = 1j * sigma_2  # = [[0, 1], [-1, 0]]

# Test Higgs doublet: H = (h_+, h_0)^T (column vector)
# After EWSB: ⟨H⟩ = (0, v/√2). Test with general components.
np.random.seed(42)
h_plus = np.random.randn() + 1j * np.random.randn()
h_zero = np.random.randn() + 1j * np.random.randn()
H = np.array([[h_plus], [h_zero]], dtype=complex)

# Conjugate Higgs: H̃ = iσ_2 H^*
H_tilde = i_sigma_2 @ H.conj()

print(f"  Test H components:    h_+ = {h_plus:.4f},  h_0 = {h_zero:.4f}")
print(f"  H̃ = iσ_2 H^* components:")
print(f"    H̃_+ = h_0^* = {H_tilde[0, 0]:.4f}    (expected: {h_zero.conjugate():.4f})")
print(f"    H̃_0 = -h_+^* = {H_tilde[1, 0]:.4f}    (expected: {-h_plus.conjugate():.4f})")
print()
assert abs(H_tilde[0, 0] - h_zero.conjugate()) < TOL
assert abs(H_tilde[1, 0] - (-h_plus.conjugate())) < TOL
print(f"  ✓ H̃ structure verified: H̃ = (h_0^*, -h_+^*)^T")
print()


# ============================================================================
# 2. Test SU(2) pseudoreal structure: do H and H̃ transform identically?
# ============================================================================
print("=" * 78)
print("Step 2: SU(2)_L pseudoreal test — same rep or different rep?")
print("=" * 78)
print()
print(f"  For SU(2): the FUNDAMENTAL and CONJUGATE reps are EQUIVALENT via")
print(f"  the iσ_2 intertwiner. Test claim: U H̃ = (iσ_2)(U H)^* for all U ∈ SU(2).")
print()

# Generate random SU(2) matrix
def random_SU2():
    """Random element of SU(2) via random axis-angle."""
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)
    angle = np.random.uniform(0, 2*np.pi)
    return np.cos(angle/2) * np.eye(2, dtype=complex) - 1j * np.sin(angle/2) * (
        axis[0]*sigma_1 + axis[1]*sigma_2 + axis[2]*sigma_3
    )


# Test: does U·H̃ = i_sigma_2 · (U·H)^* hold?
n_tests = 5
for trial in range(n_tests):
    U = random_SU2()
    UH = U @ H
    UH_tilde_direct = U @ H_tilde
    UH_tilde_via_intertwiner = i_sigma_2 @ UH.conj()

    # Verify U is in SU(2)
    assert abs(np.linalg.det(U) - 1) < TOL
    assert np.allclose(U @ U.conj().T, np.eye(2))

    diff = np.linalg.norm(UH_tilde_direct - UH_tilde_via_intertwiner)
    print(f"  Trial {trial+1}: ||U·H̃ - iσ_2·(U·H)^*|| = {diff:.2e}", end="")
    assert diff < TOL, f"SU(2)_L equivalence fails: {diff}"
    print(" ✓")

print()
print(f"  RESULT: H and H̃ transform IDENTICALLY under SU(2)_L (after iσ_2")
print(f"  intertwiner). They are the SAME representation of SU(2)_L (pseudoreal).")
print()
print(f"  CONSEQUENCE: Route 4's 'different SU(2)_L rep' framing is")
print(f"  STRUCTURALLY INCORRECT. H and H̃ are not different reps; they're")
print(f"  the same rep with components related by iσ_2 conjugation.")
print()


# ============================================================================
# 3. What IS the actual structural distinction between H and H̃?
# ============================================================================
print("=" * 78)
print("Step 3: Actual structural distinction — U(1)_Y hypercharge")
print("=" * 78)
print()
print(f"  Under U(1)_Y:")
print(f"    H has hypercharge Y_H = +1/2")
print(f"    H̃ has hypercharge Y_H̃ = -1/2")
print()
print(f"  This is FUNDAMENTALLY DIFFERENT from SU(2)_L (where they're same rep).")
print(f"  Y_H ≠ Y_H̃ is what allows Y_d Q̄_L H d_R + Y_u Q̄_L H̃ u_R to be")
print(f"  U(1)_Y-INVARIANT for both terms (with d_R having Y_d_R = -1/3 and")
print(f"  u_R having Y_u_R = +2/3, plus Q_L having Y_Q_L = +1/6).")
print()
print(f"  Bilinear hypercharges:")
print(f"    Q̄_L H d_R: Y_total = -1/6 + 1/2 - 1/3 = 0  ✓")
print(f"    Q̄_L H̃ u_R: Y_total = -1/6 - 1/2 + 2/3 = 0  ✓")
print()

bilinear_d = -1/6 + 1/2 - 1/3
bilinear_u = -1/6 - 1/2 + 2/3
assert abs(bilinear_d) < TOL and abs(bilinear_u) < TOL

print(f"  Without H̃, the up-Yukawa term Q̄_L H u_R would have")
print(f"    Y_total = -1/6 + 1/2 + 2/3 = 1  ≠ 0   →  violates U(1)_Y")
violation = -1/6 + 1/2 + 2/3
print(f"    Computed: Y_total = {violation:.4f}")
assert abs(violation - 1.0) < TOL
print()
print(f"  CONCLUSION: H̃ is REQUIRED for U(1)_Y gauge invariance of up-Yukawa.")
print(f"  The H vs H̃ distinction = U(1)_Y HYPERCHARGE distinction.")
print()


# ============================================================================
# 4. Status of U(1)_Y in the framework — is it derived?
# ============================================================================
print("=" * 78)
print("Step 4: Status of U(1)_Y in the framework — G2-D BLOCKED")
print("=" * 78)
print()
print(f"  Per `theorem_g2_edge_qubit_su2.md §7` 'What G2 does not give':")
print(f"    'G2-D: hypercharge U(1)_Y — requires ADOPTED-B3 or independent")
print(f"     derivation. The edge qubit carries SU(2) but hypercharge is not")
print(f"     yet derived.'")
print()
print(f"  STATUS: U(1)_Y is BLOCKED in the framework's current apparatus.")
print()
print(f"  CONSEQUENCE for Route 4: the H vs H̃ distinction (= hypercharge")
print(f"  distinction) cannot be derived without G2-D closure. Route 4's")
print(f"  bridge from edge-qubit SU(2)_L to a Y_u ≠ Y_d distinction on C³_gen")
print(f"  REQUIRES U(1)_Y to be derivable, which is not currently the case.")
print()
print(f"  REVISED Route 4 status: BLOCKED on G2-D (hypercharge), NOT on")
print(f"  Need-A2 (canonical generation-Z_3) as the EOD+3 audit claimed.")
print()


# ============================================================================
# 5. Alternative: Cl(0,2) ≅ ℍ complex conjugation as the f-direction distinction
# ============================================================================
print("=" * 78)
print("Step 5: Alternative — Cl(0,2) ≅ ℍ complex conjugation as candidate")
print("=" * 78)
print()
print(f"  Per `theorem_ytau_corollary §7 L13`: framework identifies")
print(f"    h⁰ ↔ f_1 direction of Cl(0,2)")
print(f"    h^+ ↔ f_2 direction of Cl(0,2)")
print()
print(f"  For up-type Yukawa via H̃ = (h_0^*, -h_+^*):")
print(f"    h_0^* would correspond to 'f_1 direction with conjugate-i'")
print(f"    h_+^* would correspond to 'f_2 direction with conjugate-i'")
print()
print(f"  In ℍ ≅ Cl(0,2): complex conjugation w.r.t. i_1 (one quaternion")
print(f"  generator) maps:")
print(f"    1 → 1   (real, unchanged)")
print(f"    i_1 → -i_1 (conjugated)")
print(f"    j → j (orthogonal generator, unchanged)")
print(f"    k → k (orthogonal generator, unchanged)")
print()

# Quaternion conjugation: q = a + bi + cj + dk → q̄ = a - bi - cj - dk
# Test: is the iσ_2 conjugation of H equivalent to a SIGN FLIP on f_1 in Cl(0,2)?

# Represent ℍ via Pauli matrices: 1 = I, i = -i*sigma_1, j = -i*sigma_2, k = -i*sigma_3
# Convention check: σ_1 σ_2 = iσ_3, so (-iσ_1)(-iσ_2) = i² σ_1 σ_2 = -i σ_3 = -i*sigma_3 = ham_k ✓
ham_1 = np.eye(2, dtype=complex)
ham_i = -1j * sigma_1
ham_j = -1j * sigma_2
ham_k = -1j * sigma_3

# Verify quaternion relations: i^2 = j^2 = k^2 = -1; ij = k, jk = i, ki = j
assert np.allclose(ham_i @ ham_i, -ham_1)
assert np.allclose(ham_j @ ham_j, -ham_1)
assert np.allclose(ham_k @ ham_k, -ham_1)
assert np.allclose(ham_i @ ham_j, ham_k)
assert np.allclose(ham_j @ ham_k, ham_i)
assert np.allclose(ham_k @ ham_i, ham_j)
print(f"  Quaternion algebra verified: i^2 = j^2 = k^2 = -1, ij = k, jk = i, ki = j")
print()


# ============================================================================
# 6. Lift attempt: SU(2)_L action on C³_gen via complex conjugation
# ============================================================================
print("=" * 78)
print("Step 6: Does Cl(0,2) complex conjugation lift to a C³_gen rotation?")
print("=" * 78)
print()
print(f"  Hypothesis: if C³_gen inherits the edge-qubit Cl(0,2) structure")
print(f"  somehow, then the H ↔ H̃ complex conjugation (i_1 → -i_1 in ℍ)")
print(f"  could induce a non-trivial transformation on C³_gen.")
print()
print(f"  Test: assume C³_gen carries a C_3-Fourier basis (per Need-A2's")
print(f"  expected derivation). What does 'complex conjugation' do on C³_gen?")
print()

omega = np.exp(2j * np.pi / 3)

# C_3-Fourier basis on C³_gen
e_0 = np.array([1, 1, 1], dtype=complex) / np.sqrt(3)
e_1 = np.array([1, omega, omega**2], dtype=complex) / np.sqrt(3)
e_2 = np.array([1, omega**2, omega], dtype=complex) / np.sqrt(3)

# Complex conjugate Fourier basis
e_0_conj = e_0.conj()
e_1_conj = e_1.conj()
e_2_conj = e_2.conj()

print(f"  C_3-Fourier basis on C³_gen (assuming Need-A2 closure):")
print(f"    e_0 = (1, 1, 1)/√3 (trivial, real)")
print(f"    e_1 = (1, ω, ω²)/√3 (ω-rep)")
print(f"    e_2 = (1, ω², ω)/√3 (ω̄-rep)")
print()
print(f"  Under complex conjugation (i_1 → -i_1):")
print(f"    e_0 → e_0 (real, unchanged)")
print(f"    e_1 → e_1^* = e_2 (swap ω ↔ ω²)")
print(f"    e_2 → e_2^* = e_1 (swap ω² ↔ ω)")
print()
print(f"  RESULT: complex conjugation on C³_gen Fourier basis acts as the")
print(f"  PERMUTATION e_1 ↔ e_2 (swap of two non-trivial Z_3-Fourier modes).")
print()

# Verify swap
assert np.allclose(e_1.conj(), e_2)
assert np.allclose(e_2.conj(), e_1)
print(f"  Verified at machine precision: e_1^* = e_2, e_2^* = e_1.")
print()


# ============================================================================
# 7. Does the swap give a non-trivial CKM?
# ============================================================================
print("=" * 78)
print("Step 7: CKM from Y_u (= U_d^*) vs Y_d under e_1 ↔ e_2 swap")
print("=" * 78)
print()
print(f"  Under the hypothesis: Y_d diagonal in C_3-Fourier basis (m_d, m_s, m_b)")
print(f"  with eigenvectors {{e_0, e_1, e_2}}.")
print()
print(f"  Y_u is the conjugate (complex conjugation acting on Y_d's matrix elements).")
print(f"  For a HERMITIAN matrix Y_d with REAL eigenvalues, Y_d^* has the SAME")
print(f"  eigenvalues but eigenvectors {{e_0^*, e_1^*, e_2^*}} = {{e_0, e_2, e_1}}.")
print()

# Form U_d (eigenvectors of Y_d as columns) in standard basis
U_d = np.column_stack([e_0, e_1, e_2])

# Form U_u (eigenvectors of Y_u = Y_d^* as columns)
U_u = np.column_stack([e_0_conj, e_1_conj, e_2_conj])

# CKM = U_u^† U_d
CKM_test = U_u.conj().T @ U_d

# Check unitarity
unitarity_err = np.linalg.norm(CKM_test @ CKM_test.conj().T - np.eye(3))
print(f"  CKM = U_u^† U_d:")
print(f"  {CKM_test}")
print()
print(f"  |CKM|² (entrywise):")
print(np.abs(CKM_test)**2)
print()

# Check if it's a permutation matrix
abs_CKM = np.abs(CKM_test)
is_permutation = (
    np.all(np.isclose(abs_CKM.sum(axis=0), [1, 1, 1])) and
    np.all(np.isclose(abs_CKM.sum(axis=1), [1, 1, 1]))
)
zeros_count = np.sum(np.isclose(abs_CKM, 0))
ones_count = np.sum(np.isclose(abs_CKM, 1))
print(f"  Unitarity: ||CKM·CKM^† - I|| = {unitarity_err:.2e}  (should be 0)")
print(f"  Permutation matrix? entries 0 or 1: zeros={zeros_count}, ones={ones_count}")
print()


# Compare to observed CKM
V_us_obs = 0.22501
V_cb_obs = 0.0410
V_ub_obs = 0.00382
print(f"  Observed CKM:")
print(f"    |V_us| = {V_us_obs:.4f}    (small mixing, NOT permutation)")
print(f"    |V_cb| = {V_cb_obs:.4f}    (small mixing)")
print(f"    |V_ub| = {V_ub_obs:.4e}  (very small mixing)")
print()
print(f"  COMPARISON: predicted permutation matrix has |V_us| ∈ {{0, 1}},")
print(f"  not 0.225. The naive 'Y_u = Y_d^*' reading gives a PERMUTATION CKM,")
print(f"  not the small-mixing pattern observed in nature.")
print()


# ============================================================================
# 8. Honest verdict on Route 4
# ============================================================================
print("=" * 78)
print("Step 8: Honest verdict on Route 4")
print("=" * 78)
print()
print(f"""  ROUTE 4 ATTACK OUTCOME:

  STRUCTURAL FINDINGS:

  (1) [VERIFIED] H and H̃ are SAME SU(2)_L representation (pseudoreal).
      Route 4's 'different SU(2)_L rep' framing in the EOD+3 audit was
      structurally incorrect — verified at machine precision via 5 random
      SU(2) matrices that U·H̃ = iσ_2·(U·H)^*.

  (2) [IDENTIFIED] The actual H vs H̃ distinction is U(1)_Y HYPERCHARGE,
      not SU(2)_L. Y_H = +1/2 vs Y_H̃ = -1/2; this hypercharge difference
      is what makes both Y_d Q̄_L H d_R and Y_u Q̄_L H̃ u_R U(1)_Y-invariant.

  (3) [BLOCKER IDENTIFIED] U(1)_Y is BLOCKED in framework's current
      apparatus per `theorem_g2_edge_qubit_su2 §7` (G2-D, requires
      ADOPTED-B3 or independent derivation). Route 4 is therefore BLOCKED
      on G2-D, NOT on Need-A2 as claimed in the EOD+3 audit.

  (4) [TESTED] Naive 'Y_u = Y_d^*' (complex conjugation reading of H ↔ H̃)
      gives a PERMUTATION CKM (entries ∈ {{0, 1}}), not the small-mixing
      CKM observed in nature. Complex conjugation alone is NOT the right
      structural mechanism.

  REVISED ROUTE 4 STATUS:

  Route 4 in its naive 'SU(2)_L pseudoreal partner' framing is OBSTRUCTED.
  The pseudoreal property of SU(2) means H and H̃ are the same SU(2)_L rep,
  so they cannot directly give a Y_u ≠ Y_d distinction via different reps.

  The actual distinction is U(1)_Y hypercharge, which is BLOCKED in the
  framework. Closing Route 4 requires either:
  (a) Deriving U(1)_Y in the framework (G2-D closure, currently blocked).
  (b) Finding an alternative mechanism not requiring U(1)_Y.

  ESCALATION: Route 4 is now correctly characterized as needing G2-D
  closure (= hypercharge derivation) rather than just Need-A2 closure.
  This is a SHARPER closure target than the EOD+3 audit identified.

  G2-D CLOSURE WOULD INVOLVE:
  - Deriving U(1)_Y as a structural feature of the edge qubit Cl(0,2) ≅ ℍ
    or Cl(6) Fock structure.
  - Currently per ADOPTED-B3 (Pati-Salam labeling) or independent derivation
    (unattempted).
  - Multi-session research-level work.

  HONEST READ:

    The EOD+3 audit's identification of Route 4 as "bounded conditional on
    Need-A2" was structurally incorrect. The actual blocker is G2-D
    (hypercharge derivation), which is a separate gap from Need-A2.

    Route 4 status: STRUCTURALLY OBSTRUCTED in naive form; closure
    requires NEW framework content (G2-D / hypercharge derivation).

    The structural progress this probe makes: sharpens the closure target
    from "Need-A2 → Route 4 → CKM" to "G2-D + Need-A2 → Route 4 → CKM",
    correctly identifying TWO required upstream closures rather than one.
""")

print("=" * 78)
print("END")
print("=" * 78)
