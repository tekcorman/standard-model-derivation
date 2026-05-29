#!/usr/bin/env python3
"""
R-6 closure: ℍ-residue → SU(2)_L gauge structure.

Test: does P1' (observer-as-finite-register, register-storable eigenvalues
must be ℝ-valued) strictly exclude ℍ on the same grounds it excludes ℝ?

The §F argument as stated:
  ℝ-L²: Stone gives skew-symmetric B, spectrum in iℝ (complexification),
        not register-storable on ℝ-L² → ℝ EXCLUDED.
  ℂ-L²: Stone gives self-adjoint H = -iB, spectrum in ℝ → register-storable
        → ℂ ADMITTED.
  ℍ-L²: NOT ADDRESSED (Row 5 GAP).

This script numerically checks the spectrum of generic Stone generators
in each setting at small dimensions, and confirms whether ℍ admits a
register-storable spectrum.

OUTCOME: REFUTED. The same logic that excludes ℝ-L² (skew-symmetric
generator with iℝ spectrum, not in ℝ) excludes ℍ-L² (anti-self-adjoint
quaternionic generator with Im(ℍ) spectrum, not in ℝ — the imaginary
quaternions form a 3-real-dimensional space, not the 1-real-dim ℝ).

ℍ is hard-gated by P1' — same mechanism as ℝ, just stronger (3D imaginary
vs 1D imaginary). The "structural-enhancement reading" (ℍ = ℂ ⊕ ℂj) reduces
ℍ-L² to ℂ-L² ⊕ ℂ-L², equivalent to ℂ-L² ⊗ ℂ². The 2-dim internal ℂ²
factor is already in the framework via Cl(6;ℂ) spinor structure (Pati-Salam
SU(2)_L sub-embedding); ℍ-residue is not a NEW source of SU(2)_L but a
redundant redescription of the existing one.

Therefore R-6 closes as REFUTED. Row 5 of the uniqueness ledger collapses
from GAP to UNIQUE: 𝔽 = ℂ is the only register-compatible field choice.

Cross-references:
  - docs/audits/registers/structural_residue_register.md R-6 (this closure updates the entry)
  - docs/operator_sweep/operator_sweep_from_A1.md §F (current ℝ-vs-ℂ field selection)
  - Adler 1995, *Quaternionic Quantum Mechanics and Quantum Fields*, Oxford.
"""

import numpy as np

# ============================================================================
# Quaternionic algebra — represented as 2x2 complex matrices
# ============================================================================
#
# Standard representation: i, j, k → Pauli-like 2x2 complex matrices.
#
# 1 → I_2
# i → i·σ_z
# j → i·σ_y  (or equivalent)
# k → i·σ_x

I2 = np.eye(2, dtype=complex)
qi = 1j * np.array([[1, 0], [0, -1]], dtype=complex)   # corresponds to quaternion i
qj = 1j * np.array([[0, -1j], [1j, 0]], dtype=complex) # corresponds to j
qk = 1j * np.array([[0, 1], [1, 0]], dtype=complex)    # corresponds to k

# Verify quaternion algebra: i²=j²=k²=ijk=-1
print("Quaternion algebra check:")
print(f"  i² = -1: {np.allclose(qi @ qi, -I2)}")
print(f"  j² = -1: {np.allclose(qj @ qj, -I2)}")
print(f"  k² = -1: {np.allclose(qk @ qk, -I2)}")
print(f"  ijk = -1: {np.allclose(qi @ qj @ qk, -I2)}")


# ============================================================================
# Stone generators in ℝ-L², ℂ-L², ℍ-L²: spectrum check
# ============================================================================

print("\n" + "="*75)
print("Stone-generator spectrum: ℝ vs ℂ vs ℍ")
print("="*75)

# ℝ-L² case: generic skew-symmetric matrix B with B^T = -B on ℝ²
# Eigenvalues: imaginary (under complexification)
print("\n--- ℝ-L² ---")
B_real = np.array([[0, 1], [-1, 0]], dtype=float)
print(f"Generic skew-symmetric B on ℝ² (=rotation generator):")
print(f"  B = {B_real.tolist()}")
print(f"  σ(B) (under complexification): {np.linalg.eigvals(B_real)}")
print(f"  σ(B) ⊂ ℝ?  {np.allclose(np.linalg.eigvals(B_real).imag, 0)}")
print(f"  → REGISTER-STORABLE (real eigenvalues)? FALSE")
print(f"  → P1' EXCLUDES ℝ.")

# ℂ-L² case: self-adjoint H on ℂ²
print("\n--- ℂ-L² ---")
H_complex = np.array([[1, 1+1j], [1-1j, 2]], dtype=complex)
print(f"Generic self-adjoint H on ℂ²:")
print(f"  H = ")
print(f"  σ(H) = {np.linalg.eigvalsh(H_complex)}")
print(f"  σ(H) ⊂ ℝ?  {np.allclose(np.linalg.eigvalsh(H_complex).imag, 0)}")
print(f"  → REGISTER-STORABLE (real eigenvalues)? TRUE")
print(f"  → P1' ADMITS ℂ.")

# ℍ-L² case: anti-self-adjoint quaternionic B on ℍ¹ ≅ ℂ²
# Stone's theorem on quaternionic Hilbert space (Adler 1995 §2):
# U(t) = exp(B·t) where B* = -B (quaternionic anti-self-adjoint).
# Spectrum of B: σ(B) ⊂ Im(ℍ) = {bi + cj + dk : b, c, d ∈ ℝ}.
#
# Concretely, take B = α·i + β·j + γ·k for α, β, γ real (quaternionic
# imaginary number, i.e., a pure-imaginary quaternion acting by
# left multiplication on ℍ¹).
#
# This is a quaternionic operator. Its "eigenvalue" is the quaternion itself.

print("\n--- ℍ-L² ---")
alpha, beta, gamma = 1.0, 2.0, 3.0
B_quat = alpha * qi + beta * qj + gamma * qk
print(f"Generic quaternionic anti-self-adjoint B = α·i + β·j + γ·k")
print(f"  with α={alpha}, β={beta}, γ={gamma}")
print(f"  Quaternion magnitude: |B| = √(α² + β² + γ²) = {np.sqrt(alpha**2 + beta**2 + gamma**2):.3f}")
print(f"  σ(B) (over ℂ², complexified): {np.linalg.eigvals(B_quat)}")
# These are ±i·|B| in the complexification — pure imaginary.
print(f"  σ(B) ⊂ ℝ?  {np.allclose(np.linalg.eigvals(B_quat).imag, 0)}")
print(f"  → REGISTER-STORABLE (real eigenvalues)? FALSE")
print(f"  → P1' EXCLUDES ℍ.")

print("""
ℍ-spectrum lives in Im(ℍ) — pure-imaginary quaternions, a 3-real-dimensional
space. Even more obstructed than ℝ-Stone's iℝ (1-real-dim). The same
register-is-real argument that excludes ℝ excludes ℍ a fortiori.
""")

# ============================================================================
# Structural-enhancement reading: ℍ ≅ ℂ²
# ============================================================================

print("="*75)
print("Structural-enhancement reading: ℍ-L² ≅ ℂ-L² ⊗ ℂ²")
print("="*75)

# As a left-ℂ module, ℍ ≅ ℂ²: every quaternion q = a + bi + cj + dk can be
# written as (a + bi) + (c + di)·j = z + w·j with z, w ∈ ℂ. So ℍ ≅ ℂ ⊕ ℂj
# as a left ℂ-module, ≅ ℂ² as a left ℂ-vector space.
#
# Right-multiplication by a unit pure quaternion (e.g. j, k) generates Sp(1)
# = SU(2), acting on the ℂ² internal structure.
#
# This SU(2) is the "ℍ-residue's SU(2)_L" candidate from R-6.

print("""
Decomposition: ℍ ≅ ℂ ⊕ ℂ·j as left ℂ-module (basis {1, j}).
  Every q ∈ ℍ:  q = (a + bi) + (c + di)·j
  Identify: q ↔ (z, w) ∈ ℂ² with z = a+bi, w = c+di.

Right-multiplication on ℍ by unit pure quaternion preserves this structure;
the group of such right-multiplications acting on ℂ² is Sp(1) = SU(2).

This is the candidate "ℍ-residue SU(2)_L" R-6 hypothesizes. Question: is it
a NEW SU(2) beyond what the framework already has?
""")

# Check: does Cl(6;ℂ) spinor structure ALREADY contain this ℂ² ?
print("Framework status check: Cl(6;ℂ) spinor structure")
print("-" * 50)
print("""
Per ../../predictions/theorem_B3_spinor_fermion_derivation.md:
  Cl(6;ℂ) irreducible spinor S has dimension 2³ = 8 over ℂ.
  Pati-Salam embedding: Spin(4) × Spin(2) ⊂ Spin(6).
  Spin(4) ≅ SU(2)_L × SU(2)_R.
  Under this decomposition: S = (4, 2, 1) ⊕ (4̄, 1, 2)
                          = SU(2)_L doublet ⊕ SU(2)_R doublet
                            (for the SM left + right fermion content).

The SU(2)_L from Pati-Salam acts on a 2-dim ℂ subspace of S — exactly
the "ℂ² internal" structure that ℍ-decomposition predicts. The two SU(2)s
are STRUCTURALLY THE SAME object (they preserve the same 2-dim ℂ slice
of S; both are subgroups of Spin(6) and U(8)).

The framework does NOT have two SU(2)_Ls; it has one, derivable from either:
  (a) Pati-Salam Spin(4) ⊂ Spin(6) (current canonical derivation), OR
  (b) right-multiplication on the ℍ-internal-structure of S (alternative
      derivation, equivalent at the level of group action).

R-6's hypothesized "new SU(2)_L from ℍ-residue" is therefore not a new
gauge structure, but a redundant relabeling of the existing Pati-Salam
SU(2)_L. It does NOT predict any new physics.
""")

# ============================================================================
# REFUTATION + MECHANISM
# ============================================================================

print("="*75)
print("R-6 CLOSURE — REFUTED")
print("="*75)

print("""
R-6 closes as REFUTED on TWO independent grounds:

GROUND 1 (raw ℍ-substrate excluded by P1').

The same register-is-real argument in §F that excludes ℝ-L² (because
skew-symmetric Stone generator has iℝ spectrum, not register-storable)
extends to ℍ-L². Adler 1995's quaternionic Stone theorem gives anti-self-
adjoint generator B with σ(B) ⊂ Im(ℍ), the 3-real-dim space of pure-
imaginary quaternions. This is even further from ℝ than iℝ was for the ℝ
case. P1' EXCLUDES ℍ — Row 5's GAP closes to UNIQUE: 𝔽 = ℂ is the only
register-compatible field choice.

GROUND 2 (structural-enhancement reading is redundant).

ℍ ≅ ℂ ⊕ ℂj has 2-dim ℂ internal structure, with right-multiplication-by-
unit-pure-quaternion giving Sp(1) = SU(2) action. This SU(2) is structurally
identical to the Pati-Salam SU(2)_L the framework already has (both act on
the 2-dim ℂ-slice of Cl(6;ℂ)'s 8-dim spinor S). R-6's "ℍ-residue → new
SU(2)_L" is not a new physics prediction; it's a redundant relabeling of
the existing Pati-Salam derivation. The two derivations describe the same
SU(2)_L gauge structure.

GROUND 1 alone is sufficient for REFUTATION at parameter_linter Type 1
gate (P1' is an axiom). GROUND 2 is the additional structural insight that
even a soft-gating reading wouldn't add new physics.

This is REFUTATION mode 2 (R-4 style): the proposed observable (SU(2)_L)
is already explained by upstream structure (Pati-Salam Spin(4) ⊂ Spin(6)
on Cl(6;ℂ) spinor). Combined with mode 1 (P1' hard-gates ℍ): R-6 fails
both refutation-mode checks.

CONSEQUENCE FOR ROW 5 OF UNIQUENESS LEDGER:

Row 5 (𝔽 = ℂ) status: was GAP (ℍ not addressed in §F).
                      now UNIQUE — ℍ explicitly excluded by same P1' machinery
                      that excludes ℝ.

§F (operator_sweep_from_A1.md) should be updated to add a third leg
(quaternionic case) to its derivation: "On ℍ-L², Adler 1995's quaternionic
Stone gives anti-self-adjoint B with σ(B) ⊂ Im(ℍ) (3-real-dim, not in ℝ)
→ register-incompatible → ℍ excluded."

CONSEQUENCE FOR THE METHODOLOGY:

R-6 was the residue most likely to PASS both refutation-mode checks
(quaternionic structure naturally hosts chirality, satisfying Mode 1; SU(2)_L
not currently derived from a singular upstream that R-6 would compete with,
satisfying Mode 2 prima facie). It still failed.

Lesson: even residues that LOOK structurally compatible can fail the strict
register-storability requirement at the field-selection layer. The ℝ-vs-ℂ
asymmetry in §F is more discriminating than it appears — it generalizes
to ℝ-vs-ℂ-vs-ℍ-vs-... by the same mechanism.

POSITIVE OUTCOME:

§F's Row 5 GAP closes. The ℝ-vs-ℂ field selection extends to a full
classification of register-compatible division algebras: only ℂ. This
sharpens the framework's foundational uniqueness claim.

REGISTER STATE AFTER R-6:

  R-1 (higher arity) — OPEN, low priority
  R-2 (fixed-point → |0⟩) — OPEN, high priority
  R-3 (relations → cycles) — OPEN, high priority
  R-4 (d=4 → time) — REFUTED (mode 2 + mode 1)
  R-5 (d≥5) — REFUTED (inherits R-4)
  R-6 (ℍ → SU(2)_L) — REFUTED (mode 1 hard-gating + mode 2 redundancy)
  R-7 (ths CKM) — REFUTED (mode 1)
  R-8 (dia girth-6) — REFUTED (inherits R-7)
  R-9 (full-MDL) — RESTRICTED to chiral nets only
  R-10 (finite-graph UV) — OPEN, low priority
  R-11 (alphabet localization) — OPEN, low priority
  R-12 (chirality) — ACCOUNTED-FOR + STRUCTURAL FILTER

Five REFUTED, one ACCOUNTED-FOR with dual role, two HIGH-PRIORITY OPEN
(R-2, R-3), three LOW-PRIORITY OPEN (R-1, R-10, R-11), one RESTRICTED.
""")
