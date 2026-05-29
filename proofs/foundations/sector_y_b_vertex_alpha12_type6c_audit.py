#!/usr/bin/env python3
"""
Down-sector y_b closure: Type 6c gate audit of the (3k*-2)/k* candidate.

CONTEXT
=======
Picking up the R-14 arc handoff (`session_handoff_2026-05-05_EOD+2_R14_arc_complete.md`).
The recommended "highest-EV next-session probe" was: derive vertex-level
α_12_vertex / α_1 = (k*-2)/k* via explicit substrate-walk computation
analogous to a separate private derivation by the author (cycle-level (g-2)/g identity).

Per the parameter_linter.md hard quality gate Type 6c (strict-minimum-smuggle
reformulation 2026-05-05 EOD+1):

  "Selection step is waterline-consistent. Every selection step in the
  derivation must be one of the two A2-T-waterline-consistent forms:
    canonical_encoding(S) — applied within an encoding-equivalence class
                            (every element of S evaluates to the SAME value)
    channel_select(S, c)  — applied across physically distinct candidates
                            in DIFFERENT structural channels
  The strict-minimum framing 'MDL bit-cost minimum across all K-candidates'
  is NOT acceptable. A derivation citing only 'MDL bit-cost minimum' without
  naming whether the selection is canonical_encoding or channel_select is
  a smuggle and BLOCKS Type 6 closure."

The handoff explicitly flagged: "current 'by analogy to Row P37' framing is
closer to bare-MDL-minimum than to canonical_encoding or channel_select."

This probe AUDITS, BEFORE attempting derivation, whether the (3k*-2)/k*
candidate has any Type 6c-compliant structural reading in the framework's
L expression, OR is post-hoc spotting at k*=3 where multiple K-rationals
coincide.

THREE QUESTIONS THE AUDIT ANSWERS
=================================
Q1: At k*=3, do multiple structurally-natural K-rationals collapse to
    α_12_vertex/α_1 = 1/3? If so, the candidate set is encoding-equivalent
    AT k*=3 — but does the equivalence class itself have a structural
    derivation, or is it post-hoc?

Q2: Does the Row P37 "many-body Hamming-weight" ansatz (n one-body +
    n(n-1)/2 two-body terms at trivalent vertex) applied to the YUKAWA
    sector predict the OBSERVED hierarchy y_τ < y_b < y_top, or does it
    predict the WRONG sign?

Q3: Is there a structural identity on srs at a trivalent vertex (analog
    of a separate private derivation by the author "pair-correlation length = g-2 internal edges of
    girth-g cycle") that would derive α_12_vertex/α_1 from substrate
    properties? Or does "vertex-level pair correlation" lack a topological
    setting on srs?

VERDICT (computed below)
========================
- Q1: Multiple candidates collapse at k*=3 — encoding-equivalence holds NUMERICALLY
  at the framework's specific k*=3, but the equivalence class is not
  derived independently. Type 6c canonical_encoding requires the equivalence
  class to be structurally identified BEFORE bit-cost ranking, which we
  cannot do without a derivation of α_12_vertex/α_1 itself.

- Q2: The many-body Hamming-weight ansatz predicts the WRONG SIGN for the
  Yukawa hierarchy: it gives y_τ ≫ y_b under any positive r = α_12/α_1,
  contradicting observed y_τ < y_b. The Row P37 cycle-level analogy does
  NOT transfer to the vertex-level Yukawa hierarchy.

- Q3: At a trivalent vertex with two of three edges occupied, there is NO
  topological "internal-edge path" between the two occupied edges (they
  share the vertex; pair-correlation length = 0 in any natural reading,
  not k*-2 = 1). The "vertex-level α_12" is structurally underspecified.

NET: the (3k*-2)/k* candidate at k*=3 is POST-HOC NUMEROLOGY.
     Type 6c gate BLOCKS closure of Row P39 down-sector via this candidate.

The R-14 partial closure target for Row P39 down-sector REQUIRES new
structural content: a sector-DEPENDENT extension of theorem_ytau_corollary
that gives the OBSERVED hierarchy direction (y_τ < y_b). Multi-session
research-level work; no bounded one-session closure path identified.
"""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import combinations

# =============================================================================
# Framework constants (Type 4 upstream)
# =============================================================================
K_STAR = 3                                              # predictions/k_star.py
G_GIRTH = 10                                            # predictions/g_girth.py
ALPHA_1_BARE = Fraction(2, 3) ** 8                      # predictions/alpha_1.py
ALPHA_1_FULL_FRAC = Fraction(5, 3) * ALPHA_1_BARE       # predictions/alpha_1_full.py
ALPHA_1_FULL = float(ALPHA_1_FULL_FRAC)
Y_TAU_PRED = ALPHA_1_FULL / K_STAR**2                   # theorem_ytau_corollary

# Observation (PDG 2024)
V_HIGGS = 246.22                                         # GeV
M_TAU_PDG = 1.77686
M_BOTTOM_PDG = 4.18
M_TOP_PDG = 172.69

Y_TAU_OBS = M_TAU_PDG / V_HIGGS
Y_BOTTOM_OBS = M_BOTTOM_PDG / V_HIGGS
Y_TOP_OBS = M_TOP_PDG / V_HIGGS

print("=" * 78)
print("Type 6c gate audit: y_b candidate (3k*-2)/k* · y_τ")
print("=" * 78)
print()
print(f"  Framework constants:")
print(f"    k*       = {K_STAR}     (predictions/k_star.py, theorem-grade)")
print(f"    g        = {G_GIRTH}    (predictions/g_girth.py, STRICT-SOLID)")
print(f"    α_1_full = (5/3)·(2/3)^8 = {float(ALPHA_1_FULL_FRAC):.6f}")
print(f"    y_τ_pred = α_1_full/k*² = {Y_TAU_PRED:.6e}    (theorem_ytau_corollary)")
print()
print(f"  Observation:")
print(f"    y_τ_obs  = m_τ/v = {Y_TAU_OBS:.6e}")
print(f"    y_b_obs  = m_b/v = {Y_BOTTOM_OBS:.6e}")
print(f"    y_t_obs  = m_t/v = {Y_TOP_OBS:.6e}")
print(f"    y_b/y_τ  = {Y_BOTTOM_OBS/Y_TAU_OBS:.5f} (observed)")
print(f"    y_t/y_τ  = {Y_TOP_OBS/Y_TAU_OBS:.5f} (observed)")
print(f"    y_t/y_b  = {Y_TOP_OBS/Y_BOTTOM_OBS:.5f} (observed)")
print()


# =============================================================================
# Q1 — encoding-equivalence audit at k*=3
# =============================================================================
print("=" * 78)
print("Q1: at k*=3, do multiple structurally-natural K-rationals give 1/3?")
print("=" * 78)
print()
print("  Test set S of 'pair-correlation modulation' candidates, each given by")
print("  a structurally distinct expression that COULD be the right object at")
print("  vertex level.")
print()

candidate_set = [
    ("(k*-2)/k*",                     "cycle-level analogy substituting k* for g",
     Fraction(K_STAR - 2, K_STAR)),
    ("1/k*",                          "uniform MDL on k* indistinguishable edges",
     Fraction(1, K_STAR)),
    ("1/binomial(k*, 2)",             "uniform MDL over (k*,2) edge-pair choices",
     Fraction(1, len(list(combinations(range(K_STAR), 2))))),
    ("(k*-1)/(k*(k*-1))",             "alt: 1 unoccupied / (k* · (k*-1) free pairs)",
     Fraction(K_STAR - 1, K_STAR * (K_STAR - 1))),
    ("1/3 (literal K-rational)",      "bare 1/3 in K = ℚ(√2,√3,√5)",
     Fraction(1, 3)),
    ("(g-2)/g · k*/g",                "double-rescale of cycle-level identity",
     Fraction(G_GIRTH - 2, G_GIRTH) * Fraction(K_STAR, G_GIRTH)),
]

print(f"  {'expression':<30}  {'value':>10}  {'reading':<46}")
print(f"  {'-'*30}  {'-'*10}  {'-'*46}")
all_one_third = True
for name, reading, val in candidate_set:
    fval = float(val)
    matches_one_third = abs(fval - 1.0/3.0) < 1e-12
    print(f"  {name:<30}  {fval:>10.6f}  {reading:<46}{'  ✓' if matches_one_third else '  ✗'}")
    if not matches_one_third:
        all_one_third = False
print()
n_matching = sum(1 for _, _, v in candidate_set if abs(float(v) - 1.0/3.0) < 1e-12)
print(f"  RESULT: {n_matching} of {len(candidate_set)} candidates evaluate to 1/3 at k*=3.")
print(f"          Multiple structurally-distinct K-expressions COLLAPSE at k*=3.")
print()
print("  Type 6c canonical_encoding(S) interpretation:")
print(f"    S is encoding-equivalent at k*=3 (all elements yield the same value).")
print(f"    Bit-cost ranking would select '1/k*' (cheapest at 4 chars + arithmetic)")
print(f"    over '(k*-2)/k*' (8 chars). But bit-cost ranking is NOT the issue —")
print(f"    we lack a structural derivation of which member of S applies AT ALL.")
print()
print("  Why the canonical_encoding reading does NOT close Type 6c here:")
print("    canonical_encoding(S) requires the equivalence-class S to be derived")
print("    independently. Without a structural derivation of α_12_vertex/α_1 = 1/3")
print("    from any structural mechanism, we have a SET-OF-CANDIDATES, not a")
print("    derived encoding class. Picking any element by bit-cost is exactly")
print("    the strict-minimum smuggle that Type 6c forbids.")
print()


# =============================================================================
# Q2 — many-body Hamming-weight ansatz predicts the WRONG SIGN
# =============================================================================
print("=" * 78)
print("Q2: does Row P37 many-body ansatz predict observed Yukawa hierarchy?")
print("=" * 78)
print()
print(f"  Row P37 ansatz: at trivalent vertex with sector Hamming weight n,")
print(f"  the many-body coupling is")
print(f"    A_n = n · α_1 + binomial(n, 2) · α_12_vertex")
print(f"        = α_1 · (n + n(n-1)/2 · r)   where r = α_12_vertex/α_1")
print()
print(f"  Sector mapping (theorem_charge_before_color):")
print(f"    n = 3: charged lepton (τ, μ, e).  Reference: y_τ_pred = α_1_full/k*²")
print(f"    n = 2: up-type quark (t, c, u)")
print(f"    n = 1: down-type quark (b, s, d)")
print(f"    n = 0: neutrino (ν₃, ν₂, ν₁)")
print()
print(f"  Naive Yukawa-sector reading: y_(sector n) ∝ A_n.")
print(f"  Then y_(sector n) / y_τ = A_n / A_3 with r free parameter.")
print()
print(f"  For each candidate r, compute predicted vs observed ratios:")
print()
print(f"  {'r':<24}  {'y_b/y_τ pred':>12}  {'y_t/y_τ pred':>12}  {'observed direction match?':<28}")
print(f"  {'-'*24}  {'-'*12}  {'-'*12}  {'-'*28}")

def A_n(n, r):
    return n + (n * (n - 1) / 2) * r

candidate_rs = [
    ("(k*-2)/k* = 1/3",   1.0/3.0),
    ("(g-2)/g = 4/5",     4.0/5.0),
    ("1 (equal weights)", 1.0),
    ("0 (no pair corr)",  0.0),
]
for name, r in candidate_rs:
    A1 = A_n(1, r)
    A2 = A_n(2, r)
    A3 = A_n(3, r)
    y_b_over_tau = A1 / A3
    y_t_over_tau = A2 / A3
    direction_obs = (Y_BOTTOM_OBS / Y_TAU_OBS > 1)  # observed: y_b > y_τ
    direction_pred = (y_b_over_tau > 1)
    match = "MATCHES" if direction_pred == direction_obs else "INVERTED"
    print(f"  {name:<24}  {y_b_over_tau:>12.4f}  {y_t_over_tau:>12.4f}  {match:<28}")

print()
print(f"  Observed: y_b/y_τ ≈ {Y_BOTTOM_OBS/Y_TAU_OBS:.3f} > 1 (down quark heavier than τ)")
print(f"            y_t/y_τ ≈ {Y_TOP_OBS/Y_TAU_OBS:.3f} > 1 (top quark heavier than τ)")
print()
print(f"  RESULT: Row P37 ansatz applied to Yukawa gives WRONG SIGN.")
print(f"          For any r ≥ 0, A_3 = 3 + 3r > A_1 = 1, so predicted y_b/y_τ < 1.")
print(f"          But observed y_b/y_τ > 1 by factor ~2.4.")
print()
print(f"  Why the cycle-level Koide ansatz transfers cleanly but the vertex-level")
print(f"  Yukawa ansatz does NOT: in Row P37, the prefactor f(n) = n(3-n)/3 is")
print(f"  symmetric in n ↔ 3-n and CANCELS in the up/down RATIO (both at f=2/3).")
print(f"  The cycle-level α_12/α_1 then enters only as 2 + r in the n=2 sum.")
print(f"  For ABSOLUTE Yukawa hierarchy across n ∈ {{1, 3}} sectors, no such")
print(f"  cancellation: A_n itself is the load-bearing quantity, and A_3 > A_1")
print(f"  for r ≥ 0 makes the lepton (n=3) sector heavier than the down sector,")
print(f"  contradicting observation. The Row P37 mechanism does not transfer.")
print()


# =============================================================================
# Q3 — is "vertex-level pair-correlation length" a derivable substrate object?
# =============================================================================
print("=" * 78)
print("Q3: is α_12_vertex defined from a substrate-walk identity on srs?")
print("=" * 78)
print()
print("  Cycle-level α_12 (a separate private derivation by the author, theorem-grade):")
print(f"    α_12_cycle / α_1 = (g-2)/g = 8/10 at g=10.")
print(f"    Source: 'pair correlation length = g-2 internal edges of girth-g cycle'.")
print(f"    Topology: two NB walkers on disjoint edges of one girth cycle pair-")
print(f"    correlate via the (g-2) internal-edge path connecting them. The")
print(f"    cycle has g vertices and g edges; (g-2) is the path length excluding")
print(f"    the two endpoint edges where the walkers sit.")
print()
print(f"  Vertex-level analog needed:")
print(f"    Two NB walkers on two of the k*=3 edges incident at a single trivalent")
print(f"    vertex. Path length connecting them: ZERO (they share the vertex).")
print()
print(f"  Possible 'pair-correlation length' readings at vertex level:")
print(f"    (i)   Direct shared-vertex coupling:  length = 0  →  α_12 = ?")
print(f"    (ii)  Through unoccupied edge then around:  length = 2 (two-step walk)")
print(f"          if we read the unoccupied edge as a mediator")
print(f"    (iii) Through the trivalent vertex itself:  length = 1 (the vertex)")
print(f"    (iv)  k* - 2 = 1 (number of unoccupied edges):  by analogy alone")
print()
print(f"  Reading (iv) is the candidate from the handoff. It matches (iii) at")
print(f"  k*=3 by coincidence (k*-2 = 1 = single-vertex-step at k*=3). At general")
print(f"  k* > 3, (iv) ≠ (iii) and the candidate gives different values.")
print()
print(f"  None of (i)-(iv) is a DERIVED structural identity — each is a candidate")
print(f"  reading without an explicit substrate-walk computation that picks one")
print(f"  channel over the others. a separate private derivation by the author cycle-level identity has")
print(f"  topological backing (girth-cycle path length is unambiguous); the vertex-")
print(f"  level case lacks this structural backing.")
print()
print(f"  RESULT: 'vertex-level pair-correlation length' is structurally")
print(f"          underspecified on srs at a trivalent vertex.")
print()


# =============================================================================
# Type 6c verdict
# =============================================================================
print("=" * 78)
print("Type 6c gate verdict for y_b = (3k*-2)/k* · y_τ candidate")
print("=" * 78)
print()
print(f"  (6a) L-expression:")
print(f"       (3k*-2)/k* ∈ ℚ ⊂ K = ℚ(√2,√3,√5).  ✓")
print(f"       Expressible in framework's structural derivation language L.")
print()
print(f"  (6b) K-membership:")
print(f"       (3k*-2)/k* = 7/3 at k*=3.  7/3 ∈ K.  ✓")
print()
print(f"  (6c) Selection step waterline-consistent:")
print(f"       Q1: encoding-equivalence class S = {{(k*-2)/k*, 1/k*, 1/binomial,")
print(f"           ...}} all evaluate to 1/3 at k*=3.  Encoding-equivalence holds")
print(f"           NUMERICALLY at k*=3, but no structural derivation identifies S")
print(f"           independently. Picking 1/k* by bit-cost without a derivation of")
print(f"           α_12_vertex/α_1 itself is the strict-minimum smuggle.")
print(f"       Q2: many-body Hamming-weight reading from Row P37 predicts WRONG")
print(f"           SIGN — would give y_τ heaviest, contradicting observed y_b > y_τ.")
print(f"           Row P37 cancellation mechanism (f(n) symmetric, cancels in")
print(f"           up/down ratio) does NOT transfer to Yukawa hierarchy.")
print(f"       Q3: substrate-walk identity at vertex level is structurally")
print(f"           underspecified — no derivation of α_12_vertex/α_1 from srs.")
print(f"       VERDICT: BLOCKED. ✗")
print()
print(f"  TYPE 6c GATE VERDICT: BLOCKED.")
print(f"    The (3k*-2)/k* candidate has no canonical_encoding or channel_select")
print(f"    derivation. The numerical match at k*=3 is a coincidence: multiple")
print(f"    structurally-distinct K-expressions collapse to 1/3 at k*=3, none of")
print(f"    them DERIVED. The Row P37 analogy DOES NOT TRANSFER (wrong sign).")
print()


# =============================================================================
# What WOULD close the down-sector y_b
# =============================================================================
print("=" * 78)
print("What WOULD close down-sector y_b (research-level open problem)")
print("=" * 78)
print()
print(f"""  The framework's existing y_τ derivation (theorem_ytau_corollary §3-§7)
  is GENUINELY sector-blind: cycle amplitude, fermion edge factors, Higgs
  edge factor, and Cl(2) channel factor each derive from sector-independent
  structural arguments (graph cycle, MDL on indistinguishable edges, vertex
  bijection at k*=n=3, per-process A2-T waterline).

  For y_b ≠ y_τ to be derivable, the framework needs ONE of:

  (R1) Sector-DEPENDENT vertex factorization
       Extend theorem_ytau_corollary §5-§7 with a Hamming-weight-n-dependent
       factor that gives the OBSERVED hierarchy direction (y_τ smallest, y_b
       intermediate, y_top largest). The C_3 isotypic decomposition of Cl(6)
       Fock at trivalent vertex (proven theorem-grade per
       cl6_fock_z3_breaking_decomposition.py) gives different rep content per
       Hamming weight — could a Yukawa amplitude 'see' this isotypic content?
       Currently the y_τ vertex factorization does not invoke isotypic content.

  (R2) Sector-DEPENDENT Cl(2) channel structure
       theorem_ytau_corollary §7 L13-15 picks ONE Cl(0,2) direction (h⁰ paired
       with τ̄_L τ_R) for y_τ. For quark sector, the SU(2)_L doublet partner
       of b is t (not ν as for τ). Could the structural difference between
       (ν, τ) doublet and (t, b) doublet enter through a sector-dependent
       channel selection? Speculative; would need NEW framework content.

  (R3) Distinct mechanism for quark sector entirely
       Quark sector might not be Yukawa-derivable at all; instead the quark
       masses inherit from a different mechanism (RG running between unification
       and EW scales applied to up/down sectors, or environmental selection).
       This route accepts that y_τ is the only Yukawa fully derivable in the
       framework's current apparatus.

  Each route is multi-session research-level work. R1 is the most natural
  framework-internal direction (uses existing C_3 isotypic theorem-grade
  apparatus). R2 requires new content. R3 is a retreat.

  HONEST READ:
    The (3k*-2)/k* candidate at -0.81% from PDG y_b is post-hoc numerology
    rather than a closure path. The R-14 down-sector closure target needs
    NEW STRUCTURAL CONTENT, not a single-session ansatz fit.
""")


# =============================================================================
# R-14 arc final state under linter-aware audit
# =============================================================================
print("=" * 78)
print("R-14 arc state after Type 6c-aware audit")
print("=" * 78)
print()
print(f"""  CLOSED:
    Row P37 (Koide quark deviation ratio): f(n) Open Question 1 closed
    (commit 3f2efae) via Z_3 isotypic decomposition on Cl(6) Fock.

  PARTIAL CLOSURE:
    Rows P15 + P34 (δ_CP_CKM, δ_CP_PMNS): V_{{-1}}-T_{{B-L}} structural
    identity (commit 13595d2) at 0.68σ + 0.15σ; conditional on existing CKM
    Other-Smuggle precedent. Sector selection rule pending.

  OPEN — Type 6c BLOCKS at present:
    Row P39 down-sector (m_b): (3k*-2)/k* candidate at -0.81% is post-hoc
    numerology under the audit of this probe. Type 6c gate blocks. Closing
    requires new sector-DEPENDENT structural content (R1 / R2 / R3 above);
    multi-session research-level work.

    Row P39 up-sector (m_top, m_c, m_u): no clean candidate from probes 1-3.
    Genuinely research-level.

    Row P38 (m_top): inherits up-sector status.

  R-14 SESSION NEXT-PICKUP RECOMMENDATIONS (revised):

    HIGHEST-EV NOW: explore R1 (sector-dependent vertex factorization via
                    C_3 isotypic content). This is the only route with
                    framework-internal precedent (cl6_fock_z3_breaking
                    is theorem-grade, used in Row P37). Test whether the
                    Yukawa amplitude can be enhanced/suppressed by the
                    isotypic content of the Hamming-weight-n Fock subspace.
                    Estimated 2-3 sessions of bounded structural work.

    ALTERNATIVE: strengthen V_{{-1}}-T_{{B-L}} closure (Rows P15 + P34) by
                 deriving the bridge from V_{{-1}} angle to W-vertex 4-walk
                 Jarlskog phase structurally (1-2 sessions, would graduate
                 P34 from THEOREM-GRADE-CONDITIONAL to UNIQUE-THEOREM-GRADE).

    NOT RECOMMENDED: pushing (3k*-2)/k* further as down-sector closure under
                     current framework apparatus. The candidate fails Type 6c
                     and the Row P37 analogy demonstrably does not transfer.
""")

print("=" * 78)
print("END")
print("=" * 78)
