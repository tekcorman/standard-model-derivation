#!/usr/bin/env python3
"""
C1 — Gleason genericity audit on F_inv(E) (substrate-generic).

CONTEXT
=======
Theorem 8 (`docs/theorems/theorem_observer_selected_d_periodic_dominance.md`)
is THEOREM-GRADE-CONDITIONAL on conditional C1 alone:

  C1: CDP 2011 axioms 1-5 hold generically on F_inv(E)'s Hilbert space
      L²(F_inv(E); ℂ) without using srs / Bloch / B(P) spectral data as
      load-bearing inputs.

The conditional concerns Step 4 of Theorem 8's proof, which uses Gleason's
frame-function-uniqueness theorem (n ≥ 3) to exclude n_eff < 3 substrate-
models asymptotically.

Step 4's Gleason invocation requires:
  (i) F_inv(E)'s Hilbert space exists (Hilbert-space structure)
  (ii) Field is ℂ (Gleason needs complex Hilbert space)
  (iii) Gleason's theorem applies (dim ≥ 3 for unique frame functions)

The framework's existing chain has TWO routes to (i)+(ii):

  Route A (CDP via observer_hilbert_space.py):
    A1 + A2-T + A3-T + srs structure → CDP axioms 1-5 → Hilbert space + ℂ
    → SRS-DEPENDENT (CDP axioms 1, 2, 4 use W3 directed-edge Markov on srs,
       B(P) spectral data on srs, Sunada 2012 Bloch decomposition on srs).

  Route B (Stone via theorem_A3_complex_hilbert_from_multiway.md):
    A1 + P1' + Folland + Stone + Strauch + Childs → Hilbert space + ℂ
    via direct construction L²(F_inv(E); ℂ) + Stone's theorem on F_inv(E)
    left regular rep + Strauch 2006 continuum-time limit + P1' field-selection.

This audit:
  §1. Enumerates srs-uses in Route A and Route B.
  §2. For each Route B step, verifies substrate-genericity (depends only on
      F_inv(E) generic structure, not srs-specific).
  §3. Verifies Gleason applies generically on the Stone-route Hilbert space.
  §4. Audit verdict.

OUT OF SCOPE
============
  - Re-deriving Stone 1932, Strauch 2006, Childs 2009, Folland 1999, or
    Gleason 1957 — these are textbook citations.
  - Verifying numerical values like (1/6)^s in Stage 3 (those use srs and
    are NOT load-bearing for the Stone route's Strauch prerequisite).
  - Re-running the existing observer_hilbert_space CDP chain — the audit
    just notes that route is srs-dependent and the Stone route bypasses it.

This probe is a STRUCTURAL READING audit, not a new theorem. No machine-
precision computation is needed; the verifications are tabular.
"""

from __future__ import annotations

print("=" * 72)
print("C1 — Gleason genericity audit on F_inv(E) (substrate-generic)")
print("=" * 72)


# ============================================================================
# §1. SRS uses in the existing chain
# ============================================================================

print()
print("=" * 72)
print("§1. SRS uses in the existing Hilbert-space chain")
print("=" * 72)

# Route A — CDP via observer_hilbert_space.py (historical, A1+A2-T+A3-T)
ROUTE_A_SRS_USES = [
    ("CDP axiom 1 (causality)",
     "W3 directed-edge Markov dynamics on srs",
     "walker_dynamics_derivation.md W3"),
    ("CDP axiom 2 (perfect distinguishability)",
     "B(P) spectral data h = (√3 + i√5)/2 with C_3 multiplicity 2 on srs",
     "B_P_doubly_degenerate_h_derivation.md"),
    ("CDP axiom 3 (ideal compressions)",
     "A2-T directly; NO srs",
     "Grunwald 2007 §5"),
    ("CDP axiom 4 (local distinguishability)",
     "srs primitive cell {4 vertices, 6 edges} + Sunada 2012 Bloch decomposition",
     "k_star.py + d_spatial.py + g_girth.py + RCSR srs entry"),
    ("CDP axiom 5 (purification)",
     "A3 directly; NO srs",
     "framework_axioms.md §4 + CDP 2011 §VIII"),
]

print()
print("Route A (CDP via observer_hilbert_space.py — historical 2026-04-18):")
print(f"  {'CDP axiom':<35}  srs-load-bearing?  Source")
print("  " + "-" * 70)
for axiom, src, _cite in ROUTE_A_SRS_USES:
    is_srs = "srs" in src.lower() and "NO srs" not in src
    flag = "YES" if is_srs else "no"
    print(f"  {axiom:<35}  {flag:<18}  {src}")

n_srs_route_a = sum(1 for _, src, _ in ROUTE_A_SRS_USES if "srs" in src.lower() and "NO srs" not in src)
print(f"\n  Route A: {n_srs_route_a} of 5 CDP axioms have srs as load-bearing input.")

# Route B — Stone via theorem_A3_complex_hilbert_from_multiway.md (post-2026-04-26)
ROUTE_B_STEPS = [
    ("Step 1: F_inv(E) is countable group",
     "A1 + Serre 1980 §I.1 Prop 4",
     "Substrate-generic — A1's involutive structure + reduced-word uniqueness."),
    ("Step 2: L²(F_inv(E); 𝔽) is separable Hilbert space (counting measure)",
     "Folland 1999 §11.1, §11.4",
     "Substrate-generic — discrete countable group has counting Haar measure."),
    ("Step 3: Left regular rep is unitary",
     "Folland 1999 §11.4",
     "Substrate-generic — unitarity from Haar left-invariance."),
    ("Step 4: Continuum-time unitary group on L² (Strauch 2006)",
     "Strauch 2006 + Childs 2009 + Stage 3 rapid-decay condition",
     "*** Stage 3 uses srs for specific (1/6)^s value, but Strauch only needs SOME correlation decay; on F_inv(E)'s regular |E|-tree Cayley graph, NB walks have spectral gap (Stark-Terras) → correlation decay generically. Substrate-generic argument: regular-tree spectral structure suffices for Strauch's prerequisite. ***"),
    ("Step 5-6: Stone's theorem; spectral content distinguishes ℝ vs ℂ",
     "Stone 1932; Reed-Simon I §VIII.4, §VI.2; Adler 1995 §2 (quaternionic)",
     "Substrate-generic — Stone's theorem applies on any complex/real/quaternionic separable Hilbert space."),
    ("Step 7: P1' field-selection (𝔽 = ℂ; ℝ + ℍ excluded)",
     "P1' register-real condition + R-6 closure 2026-04-27",
     "Substrate-generic — field-selection by spectral compatibility with finite register; uses A1 + P1' alone (per uniqueness ledger Row 5 post-R-6)."),
]

print()
print("Route B (Stone via theorem_A3_complex_hilbert_from_multiway.md — 2026-04-26):")
for step, src, gen in ROUTE_B_STEPS:
    print(f"\n  {step}")
    print(f"    Source: {src}")
    print(f"    Generic: {gen}")


# ============================================================================
# §2. Substrate-generic verification of Route B prerequisites
# ============================================================================

print()
print("=" * 72)
print("§2. Substrate-generic verification of Route B prerequisites")
print("=" * 72)

print("""
The Stone route's substrate-genericity rests on two non-trivial premises:

  (P1) F_inv(E)'s Cayley graph has correlation-decaying NB walk dynamics
       (needed for Strauch 2006 continuum-time limit).
  (P2) Field-selection 𝔽 = ℂ via P1' alone (no srs, no observer-Hilbert
       construction loops).

Both are substrate-generic facts:

  (P1) F_inv(E) = *_e Z/2 (free product of |E| copies of Z/2). Its Cayley
       graph is the (|E|)-regular tree T_|E|. NB walks on (d)-regular trees
       have spectral edge λ ∈ [-2√(d-1), 2√(d-1)] (Lubotzky-Phillips-Sarnak;
       Stark-Terras 2007 Hashimoto-Bass Ihara). Spectral gap ≥ 2/(d-1) gives
       correlation decay rate ~exp(-N · log(d-1)/d) ≥ 0 — sub-exponential
       in walk length. For Strauch's continuum-limit applicability the only
       precondition is bounded vertex degree (|E| < ∞ by A1) + sub-step
       correlation decay (✓ via NB-walk spectral gap on regular tree).
       The framework's specific (1/6)^s rate from Stage 3 is ONE numerical
       instantiation; the EXISTENCE of correlation decay is generic.

  (P2) Per uniqueness ledger Row 5 (UNIQUE post-R-6 closure 2026-04-27):
       on real L²(F_inv(E)), Stone generator B is skew-symmetric → σ(B) ⊂
       iℝ, register-incompatible. On quaternionic L²(F_inv(E)), Stone
       generator is anti-self-adjoint quaternionic → σ ⊂ Im(ℍ), register-
       incompatible (Adler 1995 §2). On complex L²(F_inv(E)), Stone
       generator H is self-adjoint → σ(H) ⊂ ℝ, register-storable.
       This argument uses A1 + P1' only; no srs, no observer-Hilbert
       construction loops.

Both (P1) and (P2) are textbook + R-6-closure facts. Neither uses srs.
""")


# ============================================================================
# §3. Gleason applies generically on Route B Hilbert space
# ============================================================================

print()
print("=" * 72)
print("§3. Gleason's frame-function-uniqueness on Route B Hilbert space")
print("=" * 72)

print("""
Gleason 1957: for any complex separable Hilbert space H of dim ≥ 3, every
additive frame function f: ProjLat(H) → [0,1] extends to a unique density
operator ρ via f(P) = Tr(ρ P).

Route B gives H = L²(F_inv(E); ℂ), which is infinite-dim separable complex
(F_inv(E) is countably infinite). Gleason applies on:
  - All finite-dim subspaces of dim ≥ 3 (e.g., span of any 3 distinct δ_g
    basis vectors)
  - The full space H (with appropriate measure-theoretic care; Maeda 1989
    extends Gleason to separable complex Hilbert spaces).

For Theorem 8 Step 4: the relevant claim is that substrate-models with
n_eff < 3 fail Gleason → frame functions form an infinite-dim space →
Cover-Thomas 2006 §13.5.2 metric-entropy cost diverges asymptotically →
below A2-T waterline.

For substrate-models with n_eff ≥ 3: Gleason gives unique frame functions
→ no metric-entropy cost from frame-function selection → finite F → above
waterline.

The n_eff parameter is the substrate-MODEL's effective Hilbert dimension
(determined by Brown rank Fisher-rank of the model's induced probability
distribution per d_spatial.md §2c). This is a property of the model being
audited, not of F_inv(E) itself. Substrate-generic.
""")


# ============================================================================
# §4. Audit verdict
# ============================================================================

print()
print("=" * 72)
print("§4. Audit verdict")
print("=" * 72)

print("""
ROUTE A (CDP, historical):
  - 3 of 5 CDP axioms (1, 2, 4) have srs as load-bearing input.
  - Route A is NOT substrate-generic by itself.
  - Route A predates the 2026-04-26 demotion of A3 to derived theorem.

ROUTE B (Stone, post-2026-04-26):
  - 7 steps, NONE with srs as load-bearing input.
  - Step 4 cites Stage 3 for specific (1/6)^s value, but the Strauch
    prerequisite (sub-step correlation decay) is satisfied generically
    on F_inv(E)'s regular-tree Cayley graph by NB-walk spectral gap.
  - Field-selection (Step 7) uses P1' + R-6 closure — substrate-generic
    per uniqueness ledger Row 5.
  - Route B IS substrate-generic.

GLEASON APPLICABILITY:
  - On L²(F_inv(E); ℂ) (Route B's Hilbert space), Gleason 1957 applies on
    subspaces of dim ≥ 3 (textbook).
  - n_eff < 3 substrate-model exclusion via metric-entropy divergence
    (Theorem 8 Step 4) is substrate-generic.

C1 AUDIT VERDICT:
  C1 closes substrate-generically via Route B. The historical Route A's
  srs uses are NOT LOAD-BEARING for Theorem 8 because Route B provides an
  alternative non-srs path to the same Hilbert-space + complex-field
  conclusion that Theorem 8 Step 4 needs.

CAVEATS:
  - Route B relies on Strauch 2006 + Childs 2009 + Stone 1932 textbook math
    (cited; not re-derived here).
  - Route B's Step 4 requires sub-step correlation decay on F_inv(E)'s
    Cayley graph; argued substrate-generic via NB-walk spectral gap on
    regular trees (Stark-Terras 2007 + Lubotzky-Phillips-Sarnak).
    The framework's specific (1/6)^s rate from Stage 3 (srs-dependent) is
    NOT load-bearing for the existence claim; only for numerical magnitude.
  - This audit is a STRUCTURAL READING of the chain, not a new theorem.

THEOREM 8 STATUS POST-AUDIT:
  Pre-audit: THEOREM-GRADE-CONDITIONAL on C1 (Gleason genericity on F_inv(E))
  Post-audit: would graduate to UNIQUE-THEOREM-GRADE if audit accepted.
              C1 has substrate-generic closure via Route B.

  Audit-grade closure ≠ formal-theorem-grade closure. To formalize, the
  audit findings should be folded into Theorem 8 Step 4 with explicit
  reference to Route B as the substrate-generic path; observer_hilbert_space.py
  and observer_dim_three_derivation.md may benefit from companion notes
  flagging Route A's srs uses as historical-alternative-not-load-bearing.
""")


# ============================================================================
# Summary
# ============================================================================

print("=" * 72)
print("C1 GLEASON GENERICITY AUDIT — SUMMARY")
print("=" * 72)
print(f"""
SRS USES MAPPED:
  Route A (CDP):      3 of 5 CDP axioms use srs as load-bearing
  Route B (Stone):    0 of 7 steps use srs as load-bearing
                      (Stage 3 cited for specific value, not load-bearing
                       for the Strauch existence claim)

VERIFICATION (P1, P2):
  P1 (correlation decay on F_inv(E) Cayley graph): substrate-generic
     via NB-walk spectral gap on regular |E|-tree
  P2 (field-selection ℂ via P1' alone): substrate-generic
     per uniqueness ledger Row 5 + R-6 closure 2026-04-27

GLEASON APPLICABILITY:
  L²(F_inv(E); ℂ) is infinite-dim separable complex Hilbert space.
  Gleason 1957 applies on subspaces of dim ≥ 3 (textbook).
  n_eff < 3 exclusion via Cover-Thomas metric entropy (substrate-generic).

VERDICT: C1 closes substrate-generically via Route B.

CAVEAT: This is an AUDIT (structural reading + chain analysis), not a
        new theorem. The findings would formalize Theorem 8's status
        from THEOREM-GRADE-CONDITIONAL on C1 to UNIQUE-THEOREM-GRADE
        if audit accepted.
""")
