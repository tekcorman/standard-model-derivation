#!/usr/bin/env python3
"""
W19 — Formal Clause-6 derivation attempt for W18 candidate (2026-05-26).

W18 PROPOSED: c_F^(rep)_j = -α₁²·c_S·(1/μ_rep_j − 1/μ_t)

For this to pass parameter_linter.md Clause 6 (K-meta-theorem), the
derivation must express it as a valid L-expression with selection step
being EITHER canonical_encoding (within encoding-equivalence class)
OR channel_select (across physically distinct K-candidates).

THIS PROBE: attempts the formal Clause-6 two-step derivation and reports
honestly whether it closes.

REFERENCE: master-doc Family-D Clause-6 derivation at α₁² (universal piece):
  Step 1 — channel_select(S, c="single_edge_spectral"):
    excludes gauge-singlet 1/(2|E|)² (the δ_r channel) and vertex-local
    1/k*² (tree Yukawa-norm channel); selects single-edge-spectral channel
    candidates.
  Step 2 — canonical_encoding({1/(2|E|), 1/(N_atoms·k*)}):
    encoding-equivalent via handshake lemma 2|E| = N_atoms·k* = 12.
    Canonical (min-bit) representative: 1/(N_atoms·k*) = 1/12.
  Result: c_F_universal = -α₁²/(N_atoms·k*) = -α₁²/12 per leg.

EXTENSION ATTEMPT: rep-resolved Clause-6 two-step at α₁² order.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'predictions'))

from alpha_1 import predict_alpha_1
from Q_Koide import chain_import_ramanujan_multiplicities

alpha_1 = float(predict_alpha_1(3, 10))
mu_t, mu_o, mu_w = chain_import_ramanujan_multiplicities()
N_atoms = 4
two_E = 12
c_S = 1.0 / two_E

print("=" * 76)
print("W19 — Formal Clause-6 derivation attempt for W18 candidate")
print("=" * 76)
print()
print("Target form: c_F^(rep)_j = -α₁²·c_S·(1/μ_rep_j − 1/μ_t)")
print()
print(f"Framework primitives (theorem-grade):")
print(f"  α₁² (Family-D scale): leading from Route H joint walker, master-doc §3 D")
print(f"  c_S = 1/(2|E|) = 1/12: Perron-residue singlet projection, unified-oblique §3.2")
print(f"  μ_rep_j: C₃-rep multiplicities (4, 2, 2) on V_Ram, Q_Koide.py")
print(f"  μ_t = N_atoms = 4: trivial-rep multiplicity = atom count by arc-transitivity")
print()
print("=" * 76)
print("ATTEMPTED Clause-6 two-step for the rep-resolved channel")
print("=" * 76)
print()

print("""
STEP 1: channel_select(S, c="rep-j-resolved single_edge_spectral")
─────────────────────────────────────────────────────────────────
The fermion leg in generation j sits in the C₃ rep-j subspace of V_Ram.
At the Yukawa vertex, the per-leg dark correction at α₁² order has a
SUB-LEADING channel beyond the master-doc rep-universal piece.

Channel constraint c: the rep-j subspace of V_Ram (dim μ_rep_j), within
the walker-active sector identified by W45 mode-count theorem
("V_Ram is walker-active; non-Ramanujan trivial sector is walker-
inactive").

K-candidates above the waterline (rep-resolved):

  (a) 1/μ_rep_j — per-rep V_Ram channel density (rep-j subspace projector)
  (b) 1/(N_atoms·μ_rep_j) — rep-j × atom count
  (c) 1/(μ_rep_j · k*) — rep-j × coordination
  (d) μ_rep_j/N_atoms² — multiplicity ratio
  (e) (μ_t − μ_rep_j)/μ_rep_j — rep-deviation from trivial reference

WHICH IS THE CANONICAL CHANNEL c?

The master-doc Family-D universal c_F = -α₁²/12 is independently the
"single-edge-spectral at the FULL B(P) directed-edge structure." At the
rep-resolved level, the canonical channel must reduce to this for the
trivial rep (where the rep-j subspace coincides with the trivial reference).

(a) gives c_F^(rep)_τ = -α₁²/4 → too big AND doesn't reduce to 0 at τ
    (the trivial rep wouldn't be DOUBLE-counted with the universal piece).

(b) gives c_F^(rep)_τ = -α₁²/16. Same issue.

(c) gives c_F^(rep)_τ = -α₁²/12 = universal piece. Reduces correctly
    AT τ, but doesn't have rep-deviation form.

(d) gives c_F^(rep)_τ = α₁²·(4/16) = α₁²/4. Wrong sign and magnitude.

(e) gives c_F^(rep)_τ = 0 (since (μ_t − μ_t)/μ_t = 0). Vanishes correctly
    at τ; reduces to "rep-deviation" form. THIS is the W18 candidate.

The form (e) = (μ_t − μ_rep_j)/μ_rep_j is structurally interpretable as
the "fractional rep-deviation from trivial." But is it a single
channel_select output? Let's check.

""")

# Structural readings
print("STRUCTURAL READINGS for each candidate:")
print()
print("  (a) 1/μ_rep_j: V_Ram rep-j projector density (single channel) ✓")
print("                 - YES this is a single channel_select output.")
print("                 - Reduces to c_S? At τ (μ=4): 1/4 ≠ 1/12 = c_S. NO.")
print()
print("  (e) (μ_t/μ_rep_j − 1) ≡ μ_t·(1/μ_rep_j − 1/μ_t): rep-deviation")
print("                 - NOT a single channel_select output. It is a DIFFERENCE")
print("                   of two channel_selects: rep-j-density MINUS")
print("                   trivial-reference-density.")
print("                 - This subtraction is NOT a canonical_encoding step")
print("                   (the two channels have DIFFERENT numerical values).")
print()
print("WHY THE SUBTRACTION ISN'T NATURAL CLAUSE-6:")
print()
print("  Per parameter_linter.md Clause 6c (REFORMULATED 2026-05-05):")
print("    'canonical_encoding(S)' — applied within encoding-equivalence class")
print("      (every element of S evaluates to the same numerical value)")
print("    'channel_select(S, c)' — applied across physically distinct K-candidates")
print("      in different structural channels")
print()
print("  Both selection steps OUTPUT a single K-rational, not a difference.")
print("  The (1/μ_rep_j − 1/μ_t) form requires SUBTRACTING two channel_selects,")
print("  which is NOT a primitive Clause-6 operation.")
print()
print("=" * 76)
print("STEP 2: canonical_encoding attempt")
print("=" * 76)
print("""
For canonical_encoding to give the W18 form, we'd need two K-candidates
that are encoding-equivalent (same numerical value):

  1/μ_rep_j           and           ???

For τ (μ=4):       1/4    =?=    1/N_atoms = 1/4 ✓ (trivially equal)
For ω (μ=2):       1/2    =?=    1/N_atoms = 1/4 ✗ (different)
For ω̄ (μ=2):      1/2    =?=    1/4 ✗

So 1/μ_rep_j and 1/N_atoms are NOT encoding-equivalent across all reps.
canonical_encoding doesn't directly apply.

HOWEVER: for the TRIVIAL rep only, 1/μ_t = 1/N_atoms holds via the
arc-transitivity identity (μ_t = N_atoms = 4). At trivial rep, the
candidates ARE encoding-equivalent, and canonical_encoding gives the
universal master-doc form 1/(N_atoms·k*) = 1/12.

For Ramanujan reps, 1/μ_rep ≠ 1/N_atoms — the candidates are physically
distinct, requiring channel_select. But channel_select selects ONE, not
a difference.
""")

print("=" * 76)
print("HONEST VERDICT")
print("=" * 76)
print("""
The W18 form (1/μ_rep_j − 1/μ_t) does NOT arise from a single Clause-6
two-step at the rep-resolved channel:

  • channel_select gives a single K-candidate (e.g., 1/μ_rep_j), NOT a
    difference between two candidates.
  • canonical_encoding requires encoding-equivalence (same numerical
    value), which fails for Ramanujan reps where 1/μ_rep ≠ 1/N_atoms.
  • The "subtraction" structure of (1/μ_rep_j − 1/μ_t) is NOT a
    primitive operation in the framework's L-grammar.

POSSIBLE INTERPRETATIONS:

  (A) The form requires EXTENDING the framework's L-grammar to include
      "rep-deviation difference" as a new primitive. This would be a
      framework-level structural extension, not a derivation within the
      existing framework. Speculative; multi-session research.

  (B) The W18 numerical match is COINCIDENCE — the (1/μ_rep_j − 1/μ_t)
      form arises by accident, not by structural mechanism. This is the
      framework's named "numerology" failure mode per master doc §6 Step 6.

  (C) The actual mechanism is DIFFERENT from W18, and the numerical
      match happens to coincide with this specific form. Without
      identifying the actual mechanism, we can't distinguish (B) from (C).

  (D) The framework's L-grammar IS expressible as a difference but I'm
      missing the right channel_select c-tag. Possible but I've enumerated
      the standard channels.

CUMULATIVE STATUS AFTER W19:

The W18 candidate matches m_e/m_μ Koide-ratio common-mode at 97% but
its formal Clause-6 derivation DOES NOT CLOSE in this session.

Per framework discipline (master doc §6 Step 6 + parameter_linter.md
Clause 6c), this means:
  - W18 is NOT theorem-grade.
  - W18 is NOT linter-eligible for predictions/ modification.
  - W18 falls into the "intriguing numerical match without mechanism"
    bucket that the framework explicitly warns against treating as closure.

THE PREDICTIONS DAG REMAINS THE AUTHORITY. No modifications to
predictions/m_e.py, predictions/m_mu.py, or predictions/y_tau.py.

The framework's existing grade for m_e/m_μ predictions
(mathematically-complete-conditional, with un-derived sub-leading
Feshbach analog per master doc §8b at ~0.5% Yukawa systematic budget)
stands. The W18 candidate is preserved as research-WIP in proofs/
but is NOT promoted.

WHAT WOULD CLOSE IT (still research-level):
  • A formal L-grammar extension that admits "rep-deviation differences"
    as primitives, parallel to how canonical_encoding admits
    encoding-equivalence classes.
  • OR: identification of the actual mechanism that produces the W18
    numerical pattern, with its own structural derivation.

Both are multi-session research targets. Not closeable in this session.

HONEST CUMULATIVE SESSION RESULT:

I attempted multiple derivations of the m_e/m_μ Koide-ratio residual:
  W4-W10: α₁³ rep-resolved Family-D sketch (FALSIFIED at Steps 1, 1b, 1c)
  W11: 2 rigorous meta-theorems (Born rule + non-factorization)
  W16: α₁²/54 from A_s mechanism (87% match, no derivation)
  W17: α₁²·c_S/μ_rep_j (breaks m_τ closure)
  W18: α₁²·c_S·(1/μ_rep_j − 1/μ_t) (97% match, preserves m_τ closure,
       no formal derivation)
  W19: Clause-6 attempt for W18 (DOES NOT CLOSE)

The most honest position: the m_e/m_μ residuals are real per-C₃-rep
observations, but the framework's current structural toolkit does NOT
admit a theorem-grade derivation of the correction mechanism. The
candidate W18 numerical match is suggestive but lacks mechanism.

The framework's existing predictions DAG grade for m_e/m_μ
(mathematically-complete-conditional within ~0.5% systematic) is the
correct current grade. No modifications.
""")
