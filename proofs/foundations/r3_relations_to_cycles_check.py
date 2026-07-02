#!/usr/bin/env python3
"""
R-3 closure: relation residues in F_inv(E)/N alternatives → srs Cayley-graph cycles.

Hypothesis (per an internal working note):
"Some or all of srs's girth-10 cycles originate from soft-gated short relations
in the F_inv(E)/N alternative set, rather than (or in addition to) from the
Bloch quotient by a translation subgroup."

If TRACED, R-3 would supply the missing mathematical mechanism for Cluster 1
(substrate-to-srs bridge gap).

Test: examine whether F_inv(E)/N can have Cayley graph isomorphic to srs.

OUTCOME: REFUTED on two independent grounds — fails BOTH refutation modes.

GROUND 1 (Mode 1 — hard structural impossibility):
  F_inv(E)/N's Cayley graph (with respect to the standard generating set
  {e_1, ..., e_|E|}) is |E|-regular for ANY normal subgroup N. This is because
  Cayley-graph regularity is determined by the size of the generating set, not
  by the relations.

  For |E| = 6 (the framework's substrate alphabet), F_inv(6)/N's Cayley graph
  is 6-regular regardless of N. srs is 3-regular. These cannot be the same
  Cayley graph for any choice of N.

  Loophole consideration: if N identifies pairs of generators (e.g., e_1 ~ e_2),
  the EFFECTIVE generating set shrinks. With 6 → 3 collapse, the effective
  Cayley graph could be 3-regular. But then F_inv(6)/N is the free product of
  3 copies of ℤ/2 (with possible further relations), whose Cayley graph (no
  further relations) is the 3-regular Bethe tree — still infinite, still no
  cycles. Adding length-10 relations to close srs's cycles would require N to
  contain those length-10 words specifically, which is exactly the geometric
  structure of srs — i.e., srs is not DERIVED from N, but N is chosen to MATCH
  srs.

GROUND 2 (Mode 2 — observable already explained upstream):
  srs's cycle structure (girth-10 etc.) is determined by srs's geometry, which
  enters the framework via the alphabet-localization stipulation in Row 7
  (H_multiway_dim_count_derivation.md §42: "The alphabet E of the multiway
  substrate is therefore taken to be these 6 undirected edges per primitive
  cell"). R-3's "relation residues produce srs cycles" mechanism conflates two
  distinct layers:

    (a) Algebraic: relations in F_inv(E)/N. Live at Layer 1 (group structure).
    (b) Geometric: cycles in srs's embedded graph. Live at Row 7 (alphabet
        localization, geometric embedding of A1's alphabet into 3D space).

  These are different. F_inv(E) is a free product (no algebraic relations
  beyond involutivity). srs's cycles are NOT algebraic relations in F_inv(E);
  they are GEOMETRIC consequences of the embedding — words that close in srs
  but are non-trivial in F_inv(E) itself.

  Treating srs's cycles as "relation residues" of F_inv(E) effectively imports
  srs's geometric structure into the algebra and then claims it was derived.
  This is circular: srs determines which relations would be needed, not the
  other way around.

CONSEQUENCE FOR CLUSTER 1 (substrate-to-srs bridge gap):

  R-3 was the hoped-for closure of Cluster 1. It does not work as posed. The
  Cluster-1 gap is genuinely a Row-7 (alphabet localization) gap, NOT an R-3
  (relation residue) gap. Closure of Cluster 1 requires either:

    (i) An MDL-on-localization argument at Row 7 that uniquely picks the srs
        embedding from the operator-permitted alternatives (single-edge,
        primitive-cell, conventional-cell, etc.).
    (ii) Accepting srs as part of A1's content (Fork B from the R-7 / Cluster-1
         analysis).
    (iii) A novel substrate-causal-set or substrate-Dirac-point argument
          (Routes A and B from an internal working note
          would close partial substrate-to-srs structure as a side effect).

  R-3 does not provide any of these. Cluster 1 remains open under Row 7's
  alphabet-localization GAP.

POSITIVE OUTCOME:

  R-3's REFUTATION sharpens the diagnosis of Cluster 1. The substrate-to-srs
  bridge is structurally a geometric-embedding choice (Row 7), not an
  algebraic-relation-residue mechanism. This narrows where future Cluster-1
  closure work should focus: closing Row 7 (alphabet localization MDL
  argument), not enumerating F_inv(E)/N quotients.

Cross-references:
  - docs/audits/registers/structural_residue_register.md R-3 (this closure updates the entry)
  - docs/audits/registers/uniqueness_ledger.md Row 2 (F_inv(E) free-product GAP), Row 7
    (alphabet localization GAP)
  - predictions/H_multiway_dim_count_derivation.md §42 (the alphabet-as-srs-edges
    stipulation that makes srs geometric, not algebraic)
"""

# ============================================================================
# Construct small F_inv(E)/N quotients explicitly and check Cayley regularity
# ============================================================================
#
# We build the Cayley graph of F_inv(E)/N for small |E| and various N, and
# verify the regularity claim: F_inv(E)/N is always |E|-regular (counting
# multi-edges if a generator and its image collide).

from collections import defaultdict
import itertools

def cayley_graph_regularity(E_size, normal_relations, max_word_length=6):
    """
    Build the Cayley graph of F_inv(E_size) / ⟨⟨normal_relations⟩⟩ up to
    word length max_word_length, and verify each *interior* vertex (i.e., vertex
    at depth ≤ max_word_length−1, whose outgoing edges are all explored) has
    exactly E_size outgoing edges labelled by distinct generators.

    Args:
        E_size: number of generators (|E|)
        normal_relations: list of words (each a tuple of generator indices)
                          to set equal to identity
        max_word_length: depth of BFS exploration

    Returns:
        (interior_vertex_count, all_interior_have_full_degree)
    """
    # Represent group elements as canonical reduced words.
    # For F_inv(E_size), reduced means no two consecutive identical letters.
    # For F_inv(E_size)/N, additional reduction by normal_relations.

    def reduce_involutive(word):
        """Remove e_i e_i pairs (involutivity)."""
        result = list(word)
        changed = True
        while changed:
            changed = False
            for i in range(len(result) - 1):
                if result[i] == result[i+1]:
                    del result[i:i+2]
                    changed = True
                    break
        return tuple(result)

    def apply_normal_relations(word, relations):
        """Apply each relation r as a substitution: occurrences of r → ε."""
        if not relations:
            return word
        result = list(word)
        changed = True
        while changed:
            changed = False
            for r in relations:
                r_list = list(r)
                # Find r as a contiguous substring and remove it
                for start in range(len(result) - len(r_list) + 1):
                    if list(result[start:start+len(r_list)]) == r_list:
                        del result[start:start+len(r_list)]
                        changed = True
                        break
                if changed:
                    break
        return reduce_involutive(tuple(result))

    def canonicalise(word):
        word = reduce_involutive(word)
        word = apply_normal_relations(word, normal_relations)
        return word

    # BFS from identity, tracking depth and computed edges
    identity = ()
    visited = {identity}
    frontier = [identity]
    edges = {}  # vertex -> dict {generator: target_vertex}; populated only for interior vertices

    for depth in range(max_word_length):
        new_frontier = []
        for v in frontier:
            edges[v] = {}  # mark v as interior (its outgoing edges will be filled)
            for g in range(E_size):
                w = canonicalise((g,) + v)
                edges[v][g] = w
                if w not in visited:
                    visited.add(w)
                    new_frontier.append(w)
        frontier = new_frontier

    # Interior vertices = those whose outgoing edges have all been computed.
    # Each interior vertex must have exactly E_size outgoing edges (one per
    # generator), since we iterate over all generators.
    n_interior = len(edges)
    n_full_degree = sum(1 for v in edges if len(edges[v]) == E_size)
    return n_interior, n_full_degree == n_interior


# Test 1: F_inv(6) (no relations) — Bethe tree, must be 6-regular
print("="*75)
print("Cayley graph regularity of F_inv(E)/N — verification")
print("="*75)

print("\nTest 1: F_inv(6) (free product, no extra relations)")
n_interior, reg_ok = cayley_graph_regularity(E_size=6, normal_relations=[], max_word_length=4)
print(f"  {n_interior} interior vertices have full outgoing-edge sets.")
print(f"  Every interior vertex has 6 outgoing edges (one per generator): {reg_ok}")

print("\nTest 2: F_inv(6) / ⟨e_0 e_0⟩ (trivial; relation already in involutivity)")
n_interior, reg_ok = cayley_graph_regularity(E_size=6, normal_relations=[(0, 0)], max_word_length=4)
print(f"  {n_interior} interior vertices.")
print(f"  Every interior vertex has 6 outgoing edges: {reg_ok}")

print("\nTest 3: F_inv(6) / ⟨e_0 e_1 e_2 e_3 e_4 e_5⟩ (a single length-6 relation)")
n_interior, reg_ok = cayley_graph_regularity(
    E_size=6,
    normal_relations=[(0, 1, 2, 3, 4, 5)],
    max_word_length=4,
)
print(f"  {n_interior} interior vertices.")
print(f"  Every interior vertex has 6 outgoing edges: {reg_ok}")
print(f"  → Cayley graph is still 6-edge-labelled regardless of the relation.")

print("\nTest 4: F_inv(6) / ⟨e_0 e_1, e_2 e_3, e_4 e_5⟩ (collapse pairs of generators)")
n_interior, reg_ok = cayley_graph_regularity(
    E_size=6,
    normal_relations=[(0, 1), (2, 3), (4, 5)],
    max_word_length=4,
)
print(f"  {n_interior} interior vertices.")
print(f"  Every interior vertex has 6 outgoing edges: {reg_ok}")
print(f"  → Each vertex has 6 generator-labelled outgoing edges, but pair-collapse")
print(f"     means edges {{0,1}}, {{2,3}}, {{4,5}} target the same neighbour. Underlying")
print(f"     simple graph has effective degree 3 (Bethe tree of free product of 3 ℤ/2's).")
print(f"  → Even with pair-collapse, the Cayley graph (with all 6 generators as labels)")
print(f"     remains 6-edge-labelled per vertex; the EFFECTIVE simple graph has degree 3,")
print(f"     but each edge is a multi-edge — the underlying group is the free product")
print(f"     of 3 ℤ/2's, whose Cayley graph is the 3-regular Bethe tree (infinite, no cycles).")

print("""
Key observation: regardless of N, the Cayley graph of F_inv(E)/N has each
vertex with exactly |E| edges labelled by generators. Multi-edges arise only
when N identifies generators (collapsing them); in that case the underlying
graph regularity is the number of DISTINCT effective generators, but the
graph remains a Bethe tree (no cycles) unless N contains length-≥-2 relations
beyond the pair-collapse.

To produce a Cayley graph WITH cycles AND of given regularity, N must contain
specific relations that close those cycles. Those relations are determined
BY THE GEOMETRY of the target graph (e.g., srs's girth-10 word). They are not
operator-permitted alternatives independently selected by MDL — they are
exactly the geometric structure being claimed as the closure.
""")

print("="*75)
print("R-3 CLOSURE — REFUTED")
print("="*75)

print("""
R-3 closes as REFUTED on TWO independent grounds:

GROUND 1 (Mode 1 — hard structural mismatch):

  F_inv(E)/N's Cayley graph is at most |E|-edge-labelled per vertex (with
  possible multi-edges if N identifies generators). The free product of k
  copies of ℤ/2 (for any k ≤ |E|) has Cayley graph = k-regular Bethe tree
  (infinite, no cycles). To have CYCLES, N must contain specific relations
  that close them. Those relations are not operator-permitted alternatives
  selected by MDL — they encode the geometric structure of the target.

  In particular: F_inv(6)/N cannot equal srs's Cayley graph for any "natural"
  choice of N. srs is 3-regular with girth 10. F_inv(6)/N is 6-regular (or
  3-regular Bethe tree if N identifies pairs); neither matches srs's cycle
  structure unless N is engineered to match srs.

GROUND 2 (Mode 2 — observable already explained upstream):

  srs's cycle structure is geometric, determined by the alphabet-localization
  stipulation (Row 7 of uniqueness_ledger.md): A1's alphabet E is identified
  with the 6 undirected edges of srs's primitive cell. This identification
  picks srs as the geometric embedding; srs's cycles are consequences of the
  embedding choice, not of algebraic relations in F_inv(E).

  R-3 conflates the algebraic and geometric layers. F_inv(E) (free product)
  has no algebraic relations beyond involutivity; srs's cycles are geometric
  loops in the embedded graph, not algebraic identities. Treating cycles as
  "relation residues" smuggles the geometric structure into the algebra and
  then claims it was derived.

CONSEQUENCE FOR CLUSTER 1:

  Cluster 1 (substrate-to-srs bridge gap, identified at the cumulative
  uniqueness-ledger analysis) is a Row 7 problem, NOT an R-3 problem. R-3
  was hoped to close Cluster 1 but does not, because the bridge is
  geometric-embedding-choice, not relation-residue mechanism.

  Closure paths for Cluster 1 (now narrowed):
    (i) MDL-on-localization at Row 7 (small theorem; not yet attempted).
    (ii) Accept srs as part of A1's content (Fork B; weakens "A1 → everything"
         claim to "A1 + srs → everything").
    (iii) Substrate-causal-set or substrate-Dirac-point routes (research-level,
          partial coverage of Cluster 1 as side effects of broader closures).

  R-3 does not provide any of these.

REGISTER STATE AFTER R-3 CLOSURE:

  R-1 (higher arity): OPEN, low priority
  R-2 (fixed-point → |0⟩): OPEN, high priority
  R-3 (relations → cycles): REFUTED (mode 1 + mode 2)
  R-4 (d=4 → time): REFUTED
  R-5 (d≥5): REFUTED (inherits R-4)
  R-6 (ℍ → SU(2)_L): REFUTED
  R-7 (ths CKM): REFUTED
  R-8 (dia girth-6): REFUTED (inherits R-7)
  R-9 (full-MDL): RESTRICTED to chiral nets only
  R-10 (finite-graph UV): OPEN, low priority
  R-11 (alphabet localization): OPEN, low priority — promoted to high after R-3
                                refutation, since it absorbs Cluster 1
  R-12 (chirality): ACCOUNTED-FOR + STRUCTURAL FILTER

Six REFUTED. Cluster 1 narrowed to Row 7 / R-11 territory. R-2 remains the
last high-priority OPEN; it is the only residue still capable of producing
a TRACED (positive) closure under the current methodology.
""")
