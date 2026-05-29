#!/usr/bin/env python3
"""
proofs/foundations/BR4_AB6_circulant_commutation_check_2026-05-27.py

BR4 Session 1 — AB6 pre-flight: commutation-obstruction check.

PURPOSE
-------
Per entry-point doc an internal working note
§5 AB6: "Does the proposed intertwiner break [W, P_C3] = 0? If yes
(intertwiner is C_3-anomalous), good candidate. If no (commutes with C_3),
inherits 10-category obstruction."

This script formalizes AB6 for the NAIVE BR4 candidate:

  Naive bridge claim (entry-point §1):
    ⟨gen j | W | gen i⟩ depends only on (j - i) mod 3,
    via L = |i-j|·g - 2(|i-j|-1)·s + n_fixed-correction.

If W's matrix elements depend only on (j-i) mod 3, then W is CIRCULANT on
C³_obs. We verify:

  1. Any circulant matrix on C³ commutes with the C_3 cyclic-shift σ.
  2. σ's spectral projectors P_1, P_ω, P_ω² are the C_3 isotypic projections.
  3. Any circulant W commutes with each P_C3 isotypic projector.

This means the naive bridge candidate INHERITS the substrate-side
10-category obstruction lemma (W75 finding: C_3 isotypic decomposition
of B_NB preserves complex-conjugate-pair closure at any Bloch fiber).

AB6 VERDICT for the naive candidate: FAILS (commutes with P_C3, inherits
obstruction). Viable BR4 candidates must be NON-CIRCULANT on C³_obs.

Companion CAS finding (vub_bridge_z3_shift_classifier.py, run 2026-05-27):
  Test 2 ("ΔGen=k ↔ m=k host" segregation) REFUTED at all m values:
  Z₃¹ vs Z₃² pair counts balanced (645 vs 647 total, ratio 0.499) across
  m ∈ {1,2,3,4} and d ∈ {8,14,20,26}. Substrate's natural C_3 symmetry
  forces circulant pattern → obstruction inherited.

Run with:
    python3 proofs/foundations/BR4_AB6_circulant_commutation_check_2026-05-27.py
"""

import numpy as np

OMEGA = np.exp(2j * np.pi / 3)
TOL = 1e-12


# ---------------------------------------------------------------------------
# C_3 cyclic-shift σ on C³_obs (per R3, Halmos 1958 §83)
# ---------------------------------------------------------------------------

sigma = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
], dtype=complex)

# Verify σ³ = I and σ is unitary
assert np.allclose(sigma @ sigma @ sigma, np.eye(3), atol=TOL), "σ³ ≠ I"
assert np.allclose(sigma @ sigma.conj().T, np.eye(3), atol=TOL), "σ not unitary"


# ---------------------------------------------------------------------------
# C_3 isotypic projectors P_1, P_ω, P_ω² (spectral projections of σ)
# ---------------------------------------------------------------------------
# σ has eigenvalues {1, ω, ω²}; eigenvectors form the DFT_3 basis.

def isotypic_projector(omega_power):
    """Project onto the σ-eigenspace with eigenvalue ω^omega_power."""
    eval_target = OMEGA ** omega_power
    # P = (1/3) · Σ_{k=0,1,2} ω^(-k·omega_power) · σ^k
    P = np.zeros((3, 3), dtype=complex)
    for k in range(3):
        P += (OMEGA ** (-k * omega_power)) * np.linalg.matrix_power(sigma, k)
    return P / 3

P_1 = isotypic_projector(0)
P_omega = isotypic_projector(1)
P_omegabar = isotypic_projector(2)

# Verify P_i² = P_i, P_i orthogonal, Σ P_i = I
for i, P in enumerate([P_1, P_omega, P_omegabar]):
    assert np.allclose(P @ P, P, atol=TOL), f"P_{i} not idempotent"
    assert np.allclose(P, P.conj().T, atol=TOL), f"P_{i} not Hermitian"

assert np.allclose(P_1 @ P_omega, np.zeros((3,3)), atol=TOL), "P_1, P_ω not orthogonal"
assert np.allclose(P_1 + P_omega + P_omegabar, np.eye(3), atol=TOL), "Σ P_i ≠ I"

# Verify σ acts as ω^i on P_i
assert np.allclose(sigma @ P_1, P_1, atol=TOL), "σ P_1 ≠ P_1"
assert np.allclose(sigma @ P_omega, OMEGA * P_omega, atol=TOL), "σ P_ω ≠ ω P_ω"
assert np.allclose(sigma @ P_omegabar, (OMEGA ** 2) * P_omegabar, atol=TOL), "σ P_ω̄ ≠ ω² P_ω̄"


# ---------------------------------------------------------------------------
# Naive BR4 candidate W: circulant on C³_obs
# ---------------------------------------------------------------------------
# Per entry-point §1, the naive bridge claim is
#   W_{ij} = f((j-i) mod 3)
# for some function f. This is exactly the circulant pattern.

def circulant(a, b, c):
    """Build the circulant matrix with first row (a, b, c)."""
    return np.array([
        [a, b, c],
        [c, a, b],
        [b, c, a],
    ], dtype=complex)


# Test with framework-relevant amplitudes: a (diagonal, ΔGen=0 within-species),
# b (ΔGen=1, V_cb-analog), c (ΔGen=2, V_ub-analog)
a = 1.0 + 0j                # within-species (placeholder)
b = (2/3)**8 / (1 - (2/3)**8)   # V_cb-analog
c = (2/3)**14 / (1 - (2/3)**14) # V_ub-leading-analog (m=2)
W_naive = circulant(a, b, c)


# ---------------------------------------------------------------------------
# AB6 check: [W_naive, σ] = 0?
# ---------------------------------------------------------------------------

commutator_with_sigma = W_naive @ sigma - sigma @ W_naive
sigma_violation = np.abs(commutator_with_sigma).max()

# ---------------------------------------------------------------------------
# AB6 check: [W_naive, P_C3] = 0 for each isotypic projector?
# ---------------------------------------------------------------------------

violations = {}
for name, P in [("P_1", P_1), ("P_ω", P_omega), ("P_ω̄", P_omegabar)]:
    comm = W_naive @ P - P @ W_naive
    violations[name] = np.abs(comm).max()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

print("=" * 72)
print("BR4 Session 1 — AB6 commutation-obstruction check on naive intertwiner")
print("=" * 72)
print()
print("Setup:")
print(f"  σ = C_3 cyclic-shift on C³_obs (R3 Halmos §83, theorem-grade)")
print(f"  P_1, P_ω, P_ω̄ = σ's spectral projectors (C_3 isotypic projections)")
print(f"  W_naive = circulant(a, b, c) with")
print(f"      a = {a}  (ΔGen=0, within-species placeholder)")
print(f"      b = (2/3)^8 / (1-(2/3)^8) ≈ {b:.6e}  (V_cb-analog, ΔGen=1)")
print(f"      c = (2/3)^14 / (1-(2/3)^14) ≈ {c:.6e}  (V_ub-leading, ΔGen=2)")
print()
print("AB6 checks:")
print(f"  ||[W_naive, σ]||_∞    = {sigma_violation:.3e}")
for name, v in violations.items():
    print(f"  ||[W_naive, {name}]||_∞ = {v:.3e}")
print()

all_commute = (
    sigma_violation < TOL
    and all(v < TOL for v in violations.values())
)

print("=" * 72)
print("VERDICT")
print("=" * 72)
if all_commute:
    print("  W_naive (circulant on C³_obs) COMMUTES WITH σ and all P_C3.")
    print()
    print("  This means W_naive is C_3-equivariant on the observer side and")
    print("  inherits the substrate-side 10-category obstruction (W75 finding:")
    print("  C_3 isotypic decomposition preserves complex-conjugate-pair")
    print("  closure at any Bloch fiber).")
    print()
    print("  AB6: NAIVE candidate FAILS — inherits obstruction.")
    print()
    print("  STRUCTURAL CONSEQUENCE: the BR4 intertwiner cannot be purely")
    print("  cyclic in the (j-i) mod 3 index. Viable candidates must be")
    print("  NON-CIRCULANT on C³_obs — i.e., depend on (i, j) more finely")
    print("  than just |i-j| mod 3.")
    print()
    print("  Candidates surviving AB6:")
    print("    (i) Species-coupled phase (W65-style Higgs-induced)")
    print("    (ii) Bloch-fiber-specific phase at P-point")
    print("         (W73 0.3% lepton near-match is a hint in this direction)")
    print("    (iii) Chirality-flip-coupled intertwiner (M_persistence's")
    print("          chir-5/3 structure on the observer side)")
else:
    print(f"  W_naive does NOT commute with C_3 structure.")
    print(f"  Naive candidate is non-trivially C_3-anomalous → AB6 PASSES.")
    print(f"  Continue BR4 work with this candidate.")

print()
print("Companion CAS finding (vub_bridge_z3_shift_classifier.py, 2026-05-27):")
print("  Z₃¹/Z₃² same-orbit pair counts BALANCED at all (m, d):")
print("    Total Z₃¹ = 645, Total Z₃² = 647 (ratio 0.499)")
print("  Substrate's natural C_3 symmetry forces circulant pattern →")
print("  combinatorial Z₃-shift segregation REFUTED across m ∈ {1,2,3,4}.")
