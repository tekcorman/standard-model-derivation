#!/usr/bin/env python3
"""
NA-4 / F5 Path B — AB5 sufficiency check (obstruction-inheritance pre-flight)

Pre-flight check before committing to the multi-session NA-4 first-probe per
the F5 articulation spec
(an internal working note §6).

The articulation spec specializes O4 = Tr[B^L · (P_{τ_L} − P_{τ_R})] to
Λ_CC Path B (cross-bracketing correlator at multiway-DAG Layer-1). Before
investing 2-3 sessions on the main probe, we must check whether the natural
framework-internal multiway-DAG operators inherit the commutation-obstruction
lemma (theorem-grade 2026-05-23,
`docs/theorems/lemma_commutation_obstruction_spectral_galois_2026-05-23.md`)
in the same way W65 (F3 Higgs phase) and W66 (F1 observer Bayesian-walk)
inherited it.

This mirrors the W65/W66 pattern: enumerate the natural operators in the
target sector and test [U, ρ_3] = 0 and tensor-factorization.

SETUP (per F5 articulation §3):

  Free magma over E at length L: basis vectors are (w; τ) where
    - w ∈ E^L is a length-L unreduced toggle word
    - τ is a binary tree on L leaves (C_{L-1} bracketings)

  ℋ_freemagma^(L) := (ℂ^|E|)^⊗L ⊗ ℂ^{C_{L-1}}

  For the minimum non-trivial associator slice: |E| = 3, L = 3.
    - dim alphabet factor: 3^3 = 27 (length-3 sequences over {e_0, e_1, e_2})
    - dim bracketing factor: C_2 = 2 (bracketings ((ab)c) and (a(bc)))
    - total dim ℋ = 27 · 2 = 54

  C_3 acts on E by cyclic permutation: e_0 → e_1 → e_2 → e_0. This is the
  natural C_3-equivariance of the srs site stabilizer (acts on the 3
  incident edges per vertex per Row 4 audit-v2). On ℋ_freemagma^(3):

  ρ_3 = R_3^⊗3 ⊗ I_2

  where R_3 is the 3-dim cyclic shift permutation matrix and the bracketing
  factor I_2 is C_3-invariant (C_3 doesn't act on bracketings — bracketings
  are syntactic structures on POSITIONS, not on alphabet labels).

PRE-DECLARED GATES (mirror W65/W66, generalized to bracketing level):

  G1 (NARROW commutation check):
      For each natural multiway-DAG operator U, does [U, ρ_3] = 0?
      Mirrors the W65/W66 commutation gate.

  G2 (BROAD factorization check):
      Does U factorize as U_alpha ⊗ U_bracket (alphabet-acting ⊗
      bracketing-acting)? If yes, the cross-bracketing trace
      Tr[U · (P_{τ_L} − P_{τ_R})] vanishes identically because
      Tr[U_alpha] · Tr[U_bracket · (|τ_L⟩⟨τ_L| − |τ_R⟩⟨τ_R|)] = 0 via
      Tr[|τ_L⟩⟨τ_L| − |τ_R⟩⟨τ_R|] = 1 - 1 = 0 (for the bracketing-blind
      part) or because U_alpha trace factors out (alphabet-blind case).
      G2 FAILS for a factorizing U; G2 PASSES if U non-factorizes.

  G3 (ESCAPE survey, structural):
      Does the framework's natural toolkit supply a multiway-DAG operator
      that is NEITHER C_3-equivariant NOR factorizable? If yes, F5 Path B
      has a real escape candidate. If no, AB5 FIRES → F5 inherits the
      obstruction.

PASS CONDITION (sufficiency to proceed with main probe):
  G1 FAIL for at least one natural U (some U doesn't commute with ρ_3), OR
  G2 PASS for at least one natural U (some U non-factorizes), AND
  G3 PASS (escape operator identifiable).

  Otherwise AB5 FIRES.
"""

from __future__ import annotations
import numpy as np
from numpy import linalg as la

np.set_printoptions(precision=4, suppress=True, linewidth=140)
TOL = 1e-10

results = []
def gate(name, passed, detail=""):
    results.append((name, bool(passed)))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    if detail:
        for line in detail.strip("\n").split("\n"):
            print(f"         {line}")
    print()


print("=" * 78)
print("NA-4 AB5 sufficiency check — F5 Path B obstruction-inheritance pre-flight")
print("=" * 78)
print()
print("Test parameters: |E| = 3, L = 3 (minimum slice for associator + C_3)")
print()


# ------------------------------------------------------------------------
# Setup: alphabet, bracketing, and C_3 representation
# ------------------------------------------------------------------------
E = 3   # alphabet size
L = 3   # word length

dim_alpha = E ** L      # 27
dim_bracket = 2         # C_{L-1} = C_2 = 2 bracketings: ((ab)c), (a(bc))
dim_total = dim_alpha * dim_bracket  # 54

# C_3 cyclic shift on alphabet: e_0 → e_1 → e_2 → e_0
R3 = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
], dtype=complex)
assert np.allclose(R3 @ R3 @ R3, np.eye(E))

# C_3 action on ℂ^|E|^⊗L = ℂ^E^L: R3 ⊗ R3 ⊗ R3
R3_alpha = R3.copy()
for _ in range(L - 1):
    R3_alpha = np.kron(R3_alpha, R3)
assert R3_alpha.shape == (dim_alpha, dim_alpha)
assert np.allclose(R3_alpha @ R3_alpha @ R3_alpha, np.eye(dim_alpha))

# C_3 action on bracketing factor: TRIVIAL (bracketings are alphabet-blind)
I_bracket = np.eye(dim_bracket, dtype=complex)

# Full ρ_3 on ℋ_freemagma^(3):
rho_3 = np.kron(R3_alpha, I_bracket)
assert rho_3.shape == (dim_total, dim_total)
assert np.allclose(rho_3 @ rho_3 @ rho_3, np.eye(dim_total))

print(f"Setup verified:")
print(f"  dim alphabet factor: {dim_alpha} (= |E|^L = 3^3)")
print(f"  dim bracketing factor: {dim_bracket} (= C_{{L-1}} = C_2)")
print(f"  dim total ℋ_freemagma^(3): {dim_total}")
print(f"  ρ_3 = R_3^⊗3 ⊗ I_2 (C_3 on alphabet only; bracket is C_3-invariant)")
print(f"  ||ρ_3^3 - I|| = {la.norm(rho_3 @ rho_3 @ rho_3 - np.eye(dim_total)):.2e}")
print()


# ------------------------------------------------------------------------
# Bracketing projectors (the load-bearing structure for O4)
# ------------------------------------------------------------------------
tau_L = np.array([1, 0], dtype=complex)  # |((ab)c)⟩
tau_R = np.array([0, 1], dtype=complex)  # |(a(bc))⟩

P_tau_L = np.outer(tau_L, tau_L.conj())
P_tau_R = np.outer(tau_R, tau_R.conj())
X_bracket = P_tau_L - P_tau_R  # the bracketing-cross-correlator operator

# Lift to ℋ_freemagma^(3): bracketing-only operator
X_full = np.kron(np.eye(dim_alpha, dtype=complex), X_bracket)
print(f"Bracketing-cross-correlator X = |τ_L⟩⟨τ_L| − |τ_R⟩⟨τ_R|:")
print(f"  Tr[X_bracket] = {np.trace(X_bracket).real:.3f} (expected 0)")
print(f"  Tr[X_full]   = {np.trace(X_full).real:.3f} (expected 0)")
print()


# ------------------------------------------------------------------------
# Construct the natural multiway-DAG operators (the framework's toolkit)
# ------------------------------------------------------------------------
print("=" * 78)
print("Natural multiway-DAG operator candidates")
print("=" * 78)

# (a) Identity (trivial dynamics)
U_a = np.eye(dim_total, dtype=complex)

# (b) Alphabet-acting Hashimoto-like (NB-walk-style C_3-equivariant operator
#     on alphabet, lifted with trivial bracketing action). Concretely, take a
#     C_3-equivariant matrix on ℂ^3 (R_3 itself is the simplest), apply it
#     iteratively per-letter:
B_alpha_atom = (R3 + R3.T) / 2.0  # symmetric C_3-equivariant operator
# Lift to length-L alphabet factor: apply B_alpha_atom at each of L positions
# (here just take tensor power for simplicity; the structural property
# matters, not the dynamics specifics)
B_alpha_full = B_alpha_atom.copy()
for _ in range(L - 1):
    B_alpha_full = np.kron(B_alpha_full, B_alpha_atom)
U_b = np.kron(B_alpha_full, I_bracket)

# (c) Bracketing-projection-only (alphabet-blind, bracketing-aware)
U_c = np.kron(np.eye(dim_alpha, dtype=complex), X_bracket)

# (d) Natural factorized "O4 candidate operator" =
#     B_alpha_full ⊗ X_bracket (this IS the natural alphabet ⊗ bracketing
#     combination that the framework's existing apparatus would naturally
#     supply for a multiway-DAG observable)
U_d = np.kron(B_alpha_full, X_bracket)

constructions = [
    ("U_a = I (trivial dynamics)",                                                U_a),
    ("U_b = B_alpha^⊗L ⊗ I_bracket (alphabet-only, bracket-blind)",               U_b),
    ("U_c = I_alpha ⊗ X_bracket (bracketing-only, alphabet-blind)",               U_c),
    ("U_d = B_alpha^⊗L ⊗ X_bracket (natural O4 candidate; factorized)",           U_d),
]


# ------------------------------------------------------------------------
# G1 — narrow commutation check: [U, ρ_3] = 0?
# ------------------------------------------------------------------------
print("=" * 78)
print("G1 — does each natural multiway-DAG operator commute with ρ_3?")
print("=" * 78)

any_escape_g1 = False
for name, U in constructions:
    comm = U @ rho_3 - rho_3 @ U
    comm_norm = la.norm(comm)
    U_norm = la.norm(U)
    rel = comm_norm / max(U_norm, 1e-12)
    status = "ESCAPES" if rel > 1e-9 else "COMMUTES"
    print(f"  {name}")
    print(f"    ||[U, ρ_3]||      = {comm_norm:.2e}")
    print(f"    ||[U, ρ_3]||/||U|| = {rel:.2e}  → {status}")
    if rel > 1e-9:
        any_escape_g1 = True
    print()

# G1 PASSES (= obstruction inheritance) if ALL commute. G1 escape = at least
# one operator non-commutes. To match W65/W66 framing, we report G1 as
# "obstruction inheritance confirmed" when all commute.
g1_obstruction_inherits = not any_escape_g1
gate("G1 (NARROW) — all natural multiway-DAG operators commute with ρ_3",
     g1_obstruction_inherits,
     "PASS means obstruction inherits via commutation; ESCAPE means some operator non-commutes")


# ------------------------------------------------------------------------
# G2 — broad factorization check: O4 trace vanishes for factorized U?
# ------------------------------------------------------------------------
print("=" * 78)
print("G2 — is the O4 cross-bracketing trace Tr[U · X_full] nonzero?")
print("=" * 78)

print(f"  X_full = I_alpha ⊗ (|τ_L⟩⟨τ_L| − |τ_R⟩⟨τ_R|)")
print()

any_nonzero_g2 = False
for name, U in constructions:
    # O4 candidate trace
    O4 = np.trace(U @ X_full).real
    # Also detect factorization: a factorized U = A_alpha ⊗ A_bracket
    # has Tr[U · X_full] = Tr[A_alpha] · Tr[A_bracket · X_bracket]
    factorized_O4 = None
    if name.startswith("U_a"):
        # Identity: A_alpha = I_27, A_bracket = I_2
        factorized_O4 = np.trace(np.eye(dim_alpha)) * np.trace(I_bracket @ X_bracket)
    elif name.startswith("U_b"):
        factorized_O4 = np.trace(B_alpha_full) * np.trace(I_bracket @ X_bracket)
    elif name.startswith("U_c"):
        factorized_O4 = np.trace(np.eye(dim_alpha)) * np.trace(X_bracket @ X_bracket)
    elif name.startswith("U_d"):
        factorized_O4 = np.trace(B_alpha_full) * np.trace(X_bracket @ X_bracket)
    print(f"  {name}")
    print(f"    Tr[U · X_full] = {O4:+.6f}")
    if factorized_O4 is not None:
        print(f"    (factorized check: Tr[A_alpha] · Tr[A_bracket · X_bracket] = {factorized_O4.real:+.6f})")
    if abs(O4) > 1e-9:
        any_nonzero_g2 = True
        print(f"    NONZERO — non-factorizing or genuinely bracketing-aware")
    else:
        print(f"    ZERO — factorization or alphabet-trace=0 collapses O4")
    print()

# G2 PASSES (= sufficiency to escape obstruction) if at least one U gives
# nonzero O4. If all are zero, the natural toolkit doesn't supply a
# bracketing-aware operator, so O4 ≡ 0 identically.
gate("G2 (BROAD) — at least one natural multiway-DAG operator gives nonzero O4",
     any_nonzero_g2,
     "PASS means O4 has nonzero support somewhere; FAIL means natural toolkit forces O4=0")


# ------------------------------------------------------------------------
# G2.b — note the special case of U_c (the bracketing-projection-only)
# ------------------------------------------------------------------------
print("=" * 78)
print("G2.b — pure bracketing operator U_c: nontrivial O4 but C_3-invariant")
print("=" * 78)

O4_c = np.trace(U_c @ X_full).real
print(f"  Tr[U_c · X_full] = {O4_c:+.6f}   (expected {np.trace(X_bracket @ X_bracket).real:+.0f} · dim_alpha = {(np.trace(X_bracket @ X_bracket).real) * dim_alpha:+.0f})")
print()
print(f"  Interpretation: U_c is the trivial 'no alphabet content, all in")
print(f"  bracketing' operator — it gives the maximum O4 by construction")
print(f"  (just measures the bracketing-class difference). But U_c is")
print(f"  ALPHABET-BLIND: it provides NO substrate dynamics content. Per F5")
print(f"  articulation §3, a meaningful Layer-1 escape needs an operator")
print(f"  that COUPLES substrate dynamics (alphabet) to bracketing.")
print()
print(f"  G2.b is INFORMATIONAL ONLY: U_c is trivially nonzero on O4 but")
print(f"  doesn't constitute an NA-4 escape because it lacks alphabet-")
print(f"  bracketing entanglement.")
print()


# ------------------------------------------------------------------------
# G3 — structural escape survey (mirror W66 §G2)
# ------------------------------------------------------------------------
print("=" * 78)
print("G3 — does the framework's natural toolkit supply an entangling operator?")
print("=" * 78)

print("""  STRUCTURAL ARGUMENT (mirror W66 §G2):

  For a multiway-DAG operator U to (i) NOT commute with ρ_3 = R_3^⊗L ⊗ I_2
  AND (ii) NOT factorize into U_alpha ⊗ U_bracket, U must ENTANGLE alphabet
  letters with bracketing tree structure in a way that breaks C_3 symmetry.

  Surveying the framework's natural multiway-DAG toolkit:

  (a) The Hashimoto B is defined on srs directed edges. It has NO bracketing
      structure — B operates on flat sequences. Natural lift to ℋ_freemagma is
      B^⊗L ⊗ I_bracket (purely factorizing).

  (b) The free-magma walker (proposed Phase 3 deliverable per the Phase 2
      verdict) operates on tree-structured words but ENUMERATES bracketings
      without weighting them by an alphabet-dependent measure. The induced
      operator on ℋ_freemagma factorizes as (alphabet-walker) ⊗ (tree-projector).

  (c) Per the F5 articulation §3 candidates O1-O4:
      - O1 (bracketing entropy): scalar, alphabet-dependent only via the prior
        p(τ|w); the framework's natural Jaynes-uniform prior is C_3-invariant
        → scalar is C_3-invariant.
      - O2 (tree-weighted h-power): factorizes as h^L (alphabet) · w(τ)
        (bracketing).
      - O3 (associator amplitude): inner product between two bracketings of
        same w; in free magma the states are orthogonal → amplitude = 0
        identically.
      - O4 (cross-bracketing correlator): the operator B^L · X factorizes
        for any framework-natural B → trace vanishes.

  (d) The 2026-05-06 sharpening (associator [a,b,c] = (ab)c − a(bc)) names
      the LAYER-1 escape FORM but does NOT supply a framework-anchored
      entangling operator. The associator is a STRUCTURAL FEATURE of
      non-associative algebras; lifting it to an operator on ℋ_freemagma
      requires a non-associative multiplication law, which is NOT in the
      framework's natural toolkit (Path A's edge-algebra approach was ruled
      out 2026-05-15 for disrupting calibration).

  CONCLUSION: under the framework's natural multiway-DAG construction, the
  natural operators (a)-(d) ALL inherit the obstruction via either
  commutation (G1) or factorization (G2). NO natural entangling operator
  is identifiable without introducing structure OUTSIDE the existing
  apparatus (which is exactly what NA-4 Phase 3 Path B was supposed to be
  about — but no concrete candidate has been articulated yet at the
  bracketing-aware operator level).
""")

g3_escape_identified = False
gate("G3 — framework-natural entangling multiway-DAG operator identifiable",
     g3_escape_identified,
     "FAIL: no natural operator entangles alphabet × bracketing in a "
     "C_3-asymmetric, framework-anchored way")


# ------------------------------------------------------------------------
# VERDICT
# ------------------------------------------------------------------------
print("=" * 78)
print("NA-4 AB5 sufficiency check — VERDICT")
print("=" * 78)

# Pass condition: G1 FAIL or (G2 PASS and G3 PASS)
ab5_passes = (not g1_obstruction_inherits) or (any_nonzero_g2 and g3_escape_identified)
# But G2 alone (without G3) doesn't escape; G2.b's U_c shows nonzero O4 only
# because it has no alphabet content, which isn't a real NA-4 escape

if not ab5_passes:
    print()
    print("AB5 FIRES — F5 Path B INHERITS THE OBSTRUCTION at the natural-")
    print("construction level.")
    print()
    print("Summary:")
    print(f"  G1 (commutation):     ALL natural operators commute with ρ_3")
    print(f"  G2 (factorization):   ALL natural alphabet-coupled operators factorize")
    print(f"                        → cross-bracketing trace vanishes identically")
    print(f"  G2.b (pure-bracket):  U_c has nonzero O4 but is alphabet-blind")
    print(f"                        → not a real Layer-1 escape (no substrate dynamics)")
    print(f"  G3 (escape survey):   no framework-natural entangling operator identified")
    print()
    print("IMPLICATION:")
    print("  Same family as W65 (F3) and W66 (F1). The framework's natural")
    print("  multiway-DAG toolkit doesn't supply an operator that:")
    print("    (i) breaks C_3 equivariance, AND")
    print("    (ii) entangles alphabet × bracketing,")
    print("  both of which are required for a non-trivial O4 cross-bracketing")
    print("  correlator that escapes the commutation-obstruction class.")
    print()
    print("HONEST F5 PATH B STATUS UPDATE:")
    print("  - The articulation spec's first-probe target (Λ_CC Path B via O4)")
    print("    is BLOCKED at the natural-construction level by AB5.")
    print("  - F5 Path B does NOT escape merely by going from edge-algebra")
    print("    (Path A, closed-NEG 2026-05-15) to bracketing-DAG-level: the")
    print("    natural bracketing-aware operators all factorize.")
    print("  - The associator sharpening (2026-05-06) names the escape FORM")
    print("    but doesn't yet supply a framework-anchored OPERATOR.")
    print()
    print("WHAT REMAINS FOR F5:")
    print("  - The 2026-05-15 Path B scoping's recommended sequencing (NA-2'")
    print("    consolidation + multiway-DAG simulator + Layer-1 residue")
    print("    enumeration) requires articulating a genuinely entangling")
    print("    operator FIRST — this is a structural-articulation question,")
    print("    not a numerical-probe question.")
    print("  - Per the W66 G2 analysis pattern, an entangling operator would")
    print("    require either:")
    print("      (a) A non-associative multiplication law (ruled out by Path A")
    print("          closed-NEG 2026-05-15 for breaking calibration), OR")
    print("      (b) A bracketing-aware substrate dynamics distinct from")
    print("          existing Hashimoto B (no framework anchor identified), OR")
    print("      (c) Framework EXTENSION beyond A-IT + k*=3 (new axioms).")
    print()
    print("  Honest probability of F5 Path B closing positive is now")
    print("  significantly below the 20-30% estimate of the entry-point doc")
    print("  §9: AB5 fires at the natural-construction level pre-flight,")
    print("  before any first-probe code runs.")
else:
    print()
    print("AB5 PASSES — F5 Path B has structural room for the main probe.")
    print()
    print("Some natural operator either non-commutes with ρ_3 OR a framework-")
    print("anchored entangling operator was identified in the G3 survey.")
    print("Proceed to the main probe (multiway-DAG simulator scaffold).")

print()
print("=" * 78)
sentinel = "AB5 FIRES (obstruction inherits)" if not ab5_passes else "AB5 PASSES"
print(f"AB5 sentinel: {sentinel}")
print("=" * 78)

# Numerical exit-code style summary for downstream wrappers
print()
print("Gate results:")
for name, passed in results:
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
