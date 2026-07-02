#!/usr/bin/env python3
"""
Row 2 closure: F_inv(E) is the substrate algebra (no extra relations).

Question: among the operator-permitted alternatives F_inv(E)/N for normal
N ⊴ F_inv(E), is the trivial choice N = ⟨ε⟩ (i.e., F_inv(E) itself) uniquely
selected?

Two readings checked:

  (a) STRICT-A1 reading: A1's algebraic content (`framework_axioms.md` §2)
      directly asserts "raw streams quotient to F_inv(E)." Under this reading,
      F_inv(E) is part of A1's letter; no further selection needed.
      → Row 2 closure: UNIQUE-by-stipulation.

  (b) OPERATOR-PERMITTED reading: F_inv(E)/N for any normal N is operator-
      permitted (it still satisfies T_e² = id). Selection between F_inv(E)
      and its quotients requires an MDL-on-relations argument.

This script computes the MDL bookkeeping for reading (b).

OUTCOME:

  Reading (a): UNIQUE-by-stipulation. A1's algebraic content includes the
               free-product structure. No further argument required at the
               framework's foundational layer.

  Reading (b): DOMINANT, not UNIQUE. F_inv(E) has the smallest model
               description length L(M) among algebras realizing A1. However,
               for short relations |r| ≥ 2 at typical observation lengths
               N ≳ |E|^|r|, the quotient F_inv(E)/⟨⟨r⟩⟩ has positive MDL
               savings and is soft-retained per A2-T waterline (Row 11).
               These soft-retained quotient algebras are R-3-style relation
               residues. R-3 was REFUTED for srs-cycle-matching relations
               (no physical signature beyond the existing geometric
               embedding); generic short relations remain soft-retained
               with no observable consequence.

  Combined closure: Row 2 closes as **UNIQUE under reading (a) /
                    DOMINANT under reading (b).** The framework's foundational
                    presentation uses reading (a). Status: UNIQUE.

  This is the same kind of definitional-vs-derived distinction Row 1 carries:
  Row 1's p = 1 exclusion is "A1's spirit" but not strictly in A1's letter.
  Row 2's "no extra relations" is in A1's algebraic content explicitly
  (per `framework_axioms.md` §2: "raw streams quotient to F_inv(E)").

  The alternative, weaker DOMINANT closure is honest about the operator-
  permitted-but-soft-retained quotients. Either is gate-passing under
  parameter_linter; the framework picks the strict reading.

Cross-references:
  - docs/audits/registers/uniqueness_ledger.md Row 2
  - docs/framework/framework_axioms.md §2 (A1 algebraic content)
  - docs/audits/registers/structural_residue_register.md R-3 (relation residues; closed REFUTED)
  - Serre 1980 *Trees* §I.1 Prop 4 (reduced-word uniqueness within F_inv(E))
  - Mac Lane 1998 *Categories for the Working Mathematician* §I.5 (universal
    property of free constructions)
  - Rissanen 1978 / 1983 (universal prefix-code prior L*(n) for relation cost)
"""

from math import log2

# ============================================================================
# Computation: MDL bookkeeping under reading (b)
# ============================================================================

def L_star(n):
    """Rissanen 1983 universal prefix-code length for positive integer n."""
    if n <= 0:
        return 0.0
    if n == 1:
        return 1.0
    total = 1.0
    x = float(n)
    while x > 1.0:
        lx = log2(x)
        total += lx
        x = lx
        if x <= 0:
            break
    return total


def relation_cost(E_size, r_length):
    """Bits to encode a relation of length r_length over alphabet of size E_size."""
    return r_length * log2(E_size) + L_star(r_length)


def relation_savings(N, E_size, r_length):
    """Expected savings (in bits) over a length-N reduced-word stream from
    introducing a relation of length r_length, assuming i.i.d. uniform stream
    per branch measure μ (Row 12).

    Per-position probability that the next r_length letters match a specific
    reduced word: (E_size - 1)^{-(r_length - 1)} / E_size.
    Per match, savings: r_length * log2(E_size - 1) bits.
    Total savings: N * (per-position prob) * (per-match savings).
    """
    if r_length < 2:
        return 0.0
    per_position_prob = (E_size - 1) ** -(r_length - 1) / E_size
    per_match_savings = r_length * log2(E_size - 1)
    return N * per_position_prob * per_match_savings


# Framework values
E_SIZE = 6   # Row 7: |E| = 6 (UNIQUE, post-R-11 closure)


print("="*75)
print("Row 2 — MDL-on-relations bookkeeping")
print("="*75)
print(f"\nFramework alphabet: |E| = {E_SIZE}")
print(f"\nFor each candidate relation length |r|, compute the threshold")
print(f"stream length N* above which F_inv(E)/⟨⟨r⟩⟩ has positive MDL savings:")
print()

print(f"  {'|r|':<5s} {'Cost (bits)':<12s} {'Savings rate (bits/N)':<22s} {'N* threshold':<15s} {'Note':<35s}")
print(f"  {'---':<5s} {'---':<12s} {'---':<22s} {'---':<15s} {'---':<35s}")

for r_len in range(2, 8):
    cost = relation_cost(E_SIZE, r_len)
    savings_rate = relation_savings(1, E_SIZE, r_len)  # savings per unit N
    if savings_rate > 0:
        N_threshold = cost / savings_rate
    else:
        N_threshold = float("inf")
    note = ""
    if r_len == 2:
        note = "Hard-gated by Row 4 (k* = 3)"
    elif r_len == 3:
        note = "Generic relation; R-3 territory"
    elif r_len == 10:
        note = "= srs girth; R-3 case (REFUTED)"

    print(f"  {r_len:<5d} {cost:<12.2f} {savings_rate:<22.4f} {N_threshold:<15.0f} {note:<35s}")

print(f"""

Reading (b) interpretation:

  - For |r| = 2: relations identifying generators (e_i e_j = ε ⇒ e_i = e_j).
    These collapse the alphabet, hard-gated by Row 4 (k* = 3 needs 3
    distinguishable generators per Brown 1986 Fisher rank for d = 3).

  - For |r| ≥ 3: above-waterline at N ≳ {relation_cost(E_SIZE, 3)/relation_savings(1, E_SIZE, 3):.0f}–{relation_cost(E_SIZE, 5)/relation_savings(1, E_SIZE, 5):.0f}.
    Generic short relations would be soft-retained at typical observation
    lengths. No physical observable matches their cycle structure (per R-3
    REFUTATION for srs-matching relations); residual soft-retention has no
    known downstream signature.

  - Asymptotic |r| → ∞: relation cost grows linearly while savings shrink
    exponentially. Long relations are below waterline.

Reading (a) interpretation:

  A1's algebraic content (`framework_axioms.md` §2) directly states "raw
  streams quotient to F_inv(E)." This is part of A1's letter, not a derived
  theorem. The free involutive monoid is selected by definitional commitment.
  No MDL argument needed.

============================================================================
ROW 2 CLOSURE — UNIQUE under reading (a) / DOMINANT under reading (b)
============================================================================

The framework's foundational presentation uses reading (a) (F_inv(E) is part
of A1's algebraic content, not separately derived). Under this reading,
Row 2 closes as **UNIQUE-by-stipulation** — same kind of definitional move
as Row 1's p = 1 exclusion.

Under reading (b), Row 2 closes as **DOMINANT** — F_inv(E) has the smallest
L(M) among algebras realizing A1, and short-relation quotients are soft-
retained per A2-T waterline. The soft-retained quotients are R-3-style
relation residues; R-3 is REFUTED for srs-matching relations and the residue
mechanism produces no known observable.

Both readings are gate-passing under parameter_linter:

  Reading (a): Type 1 (A1 axiom — F_inv(E) is part of A1's algebraic content).
  Reading (b): Type 2 (explicit MDL bookkeeping above) + Type 3 (Rissanen
               1978/1983 universal prior + Mac Lane 1998 universal property
               of free constructions) + Type 4 (Row 12 branch measure μ for
               i.i.d. uniformity assumption).

The framework picks reading (a). Row 2 status: **UNIQUE**.

The honest backup is reading (b) DOMINANT — Row 2 falls back to DOMINANT
if A1's "algebraic content" is read more narrowly (as just T_e² = id,
without the free-product specification).

CASCADE: Row 2's closure does not affect downstream rows (Rows 1, 3-22 do
not directly cite Row 2's free-product status; they cite F_inv(E) directly,
which is what Row 2 confirms either way).

UNIQUENESS LEDGER UPDATE: Row 2 status GAP → UNIQUE. The framework's
structural pass now has 18 UNIQUE / 2 DOMINANT / 1 mathematically-complete /
2 GAP / 1 OPEN across 22 rows. The two remaining GAPs are Rows 14b + 21
(smooth-manifold continuum limit; research-level Lorentzian-signature
routes); the OPEN row is Row 15b (B1 ordering workstream).
""")
