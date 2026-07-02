#!/usr/bin/env python3
"""
f_3 explicit computation on srs — fraction of NB-walker 3-letter windows
that hit non-Fano (non-associative) octonion triples under the MDL-preferred
Fano-line embedding.

CONTEXT
=======
Per Theorem 9 PARTIAL (`proofs/foundations/theorem9_f3_quantification_on_srs.py`,
revised 2026-05-07): at the k*=3 dominant slice, MDL prefers Fano-line
embedding of the 3 incident edges at each srs vertex into 3-of-7 octonion
units (forming a Fano line of the Fano plane). This local-vertex argument
gives effective f_3 = 0 ON A SINGLE-VERTEX 3-LETTER WINDOW.

But walker 3-letter windows generally span 2 or 3 vertices (since each step
moves to a different vertex). So the actual f_3 value depends on how the
local Fano lines at adjacent vertices combine globally.

This probe COMPUTES f_3 on srs explicitly:
  - srs primitive cell = K_4 quotient (4 vertices, 6 edges, K_4 adjacency)
  - Fano-deletion embedding: 4 vertices → 4 of 7 Fano lines covering 6 of 7
    points (one octonion unit excluded; 7 such deletion configurations exist)
  - Enumerate NB walks of length 3 with proper Jaynes per-step weighting
  - For each walk, get the 3 toggle labels (edges); check if collinear in
    Fano plane
  - f_3 = weighted fraction of non-Fano (non-collinear) triples

DYNAMIC_ZOO DOC PREDICTION
==========================
"f_3 = 28/35 = 4/5 if srs walks visit all triples uniformly"
But "srs has constraints (girth g=10, vertex-transitive, edge-transitive).
Not all 3-letter triples occur with equal frequency. The actual f_3 on srs
is a closed combinatorial computation."

This probe gives the closed combinatorial value.

DELIVERABLE
===========
Numerical f_3 value with the following implications:
  - f_3 ≈ 0: Theorem 9 PARTIAL would close to CO-DOMINANT-with-constant-
    suppression. Layer-1 octonion plurally retained at ~exp(-7) ≈ percent
    level. Could explain unexplained residues (caveat: M2 partial echo).
  - f_3 ≈ 4/5 (high): Theorem 9 SHARP-DOMINANT branch. Layer-1 octonion
    suppressed astronomically. Layer-1 escape via this channel ruled out.
  - Intermediate: requires careful interpretation.

OUT OF SCOPE
============
  - The audit doesn't test access mechanisms M1-M7 (those were audited in
    `M_mechanisms_synthesis_2026-05-07.md`; mostly NEGATIVE).
  - The audit doesn't compute observable cosmology blocker corrections
    (Phase 3+ work, M5/M6 framework extensions).
  - The audit takes the MDL-preferred Fano-line embedding as given (per
    Theorem 9 PARTIAL); doesn't audit alternative embeddings.
"""

from __future__ import annotations
from itertools import combinations, permutations
from collections import Counter, defaultdict
from fractions import Fraction


# ============================================================================
# §1. The Fano plane — 7 points, 7 lines, each pair of points on unique line
# ============================================================================
# Standard Fano plane (Cayley/Lounesto §23.1):
#   Points: {1, 2, 3, 4, 5, 6, 7} (octonion imaginary unit indices)
#   Lines (triples, 7 of C(7,3)=35):
#     {1,2,3}, {1,4,5}, {1,6,7}, {2,4,6}, {2,5,7}, {3,4,7}, {3,5,6}

FANO_LINES = [
    frozenset([1, 2, 3]),
    frozenset([1, 4, 5]),
    frozenset([1, 6, 7]),
    frozenset([2, 4, 6]),
    frozenset([2, 5, 7]),
    frozenset([3, 4, 7]),
    frozenset([3, 5, 6]),
]
assert len(FANO_LINES) == 7
assert len(set(FANO_LINES)) == 7  # all distinct

# Verify Fano plane axioms
def verify_fano_plane():
    """Each pair of points lies on exactly 1 line; each pair of lines
    intersects in exactly 1 point."""
    points = set(range(1, 8))
    # Each pair of points → exactly 1 line
    for p1, p2 in combinations(points, 2):
        containing = [L for L in FANO_LINES if p1 in L and p2 in L]
        assert len(containing) == 1, f"pair {p1},{p2}: {len(containing)} lines"
    # Each pair of lines → exactly 1 point intersection
    for L1, L2 in combinations(FANO_LINES, 2):
        intersection = L1 & L2
        assert len(intersection) == 1, f"lines {L1}, {L2}: |∩| = {len(intersection)}"
    # Each point lies on exactly 3 lines
    for p in points:
        on = [L for L in FANO_LINES if p in L]
        assert len(on) == 3, f"point {p}: on {len(on)} lines"

verify_fano_plane()
print("=" * 72)
print("f_3 explicit computation on srs")
print("=" * 72)
print()
print("§1. Fano plane: 7 points, 7 lines verified (each pair of points on")
print("    exactly 1 line; each pair of lines intersects in exactly 1 point;")
print("    each point on 3 lines).")


# ============================================================================
# §2. Fano-deletion configuration — 4 lines covering 6 of 7 points
# ============================================================================
# Per the dynamic_zoo doc + Theorem 9 PARTIAL: each srs vertex's 3 incident
# edges form a Fano line. srs primitive cell has 4 vertices and 6 edges
# (K_4 quotient, each pair of vertices shares 1 edge).
#
# Embedding: 4 vertex-edge-sets → 4 Fano lines covering 6 of 7 points.
# This is a "Fano-deletion" configuration: pick 1 point P* to exclude;
# then 4 lines NOT through P* cover the other 6 points exactly.
# Each point is on 3 Fano lines; excluding P* removes exactly 3 lines;
# remaining 4 lines automatically cover 6 of 7 points.
# 7 deletion configurations, one per excluded point.

def find_deletion_config(excluded_point: int) -> list[frozenset]:
    """Return the 4 Fano lines NOT through excluded_point."""
    return [L for L in FANO_LINES if excluded_point not in L]

print()
print("=" * 72)
print("§2. Fano-deletion configuration (for srs primitive-cell K_4 embedding)")
print("=" * 72)

for p_excl in range(1, 8):
    lines_kept = find_deletion_config(p_excl)
    points_covered = set()
    for L in lines_kept:
        points_covered |= L
    assert len(lines_kept) == 4
    assert len(points_covered) == 6
    assert p_excl not in points_covered
    # Each kept point on exactly 2 of the 4 kept lines (3 - 1 = 2)
    for p in points_covered:
        on = [L for L in lines_kept if p in L]
        assert len(on) == 2

# Pick excluded_point = 7 by convention (arbitrary; symmetric over the 7
# choices by Aut(Fano) = PSL(2,7) = GL(3,2) = 168-element group)
EXCLUDED = 7
LINES_4 = find_deletion_config(EXCLUDED)
EDGE_POINTS = sorted({p for L in LINES_4 for p in L})  # 6 points used
assert EDGE_POINTS == [1, 2, 3, 4, 5, 6]

print(f"\n  Convention: exclude point {EXCLUDED} (excluding any other point gives")
print(f"  isomorphic configuration via Aut(Fano) = GL(3,2), order 168).")
print(f"  4 vertex-lines (each is the Fano line of its vertex's 3 incident edges):")
for i, L in enumerate(LINES_4):
    print(f"    Vertex v_{i}: edges = {sorted(L)}")
print(f"  6 edge-points (octonion imaginary units): {EDGE_POINTS}")


# ============================================================================
# §3. srs primitive cell K_4 quotient — vertex-edge incidence
# ============================================================================
# srs primitive cell: 4 vertices, 6 edges, K_4 adjacency
# (each pair of vertices shares 1 edge per |E|=k*·|V|/2 = 3·4/2 = 6).
#
# Identify vertices with the 4 Fano lines (LINES_4 above) and edges with
# their points (EDGE_POINTS). Two vertices v_i, v_j are adjacent in K_4
# via the unique edge in L_i ∩ L_j.

print()
print("=" * 72)
print("§3. srs primitive cell K_4 quotient: vertex-edge incidence")
print("=" * 72)

# Build vertex → incident edges map (just LINES_4)
VERTEX_EDGES = {i: LINES_4[i] for i in range(4)}

# Build edge → 2 endpoints map (each edge between exactly 2 vertices)
EDGE_VERTICES = defaultdict(list)
for v_idx, edges in VERTEX_EDGES.items():
    for e in edges:
        EDGE_VERTICES[e].append(v_idx)

# Verify K_4: each edge between exactly 2 vertices; each pair of vertices
# shares exactly 1 edge
for e, vs in EDGE_VERTICES.items():
    assert len(vs) == 2, f"Edge {e} between {len(vs)} vertices, expected 2"

for v_i, v_j in combinations(range(4), 2):
    shared = VERTEX_EDGES[v_i] & VERTEX_EDGES[v_j]
    assert len(shared) == 1, f"Vertices {v_i}, {v_j} share {len(shared)} edges"

print(f"  4 vertices, 6 edges, K_4 adjacency verified")
print(f"  Each pair of vertices shares exactly 1 edge ✓")
print(f"  Each edge is incident to exactly 2 vertices ✓")
print()
print(f"  Edge → endpoints:")
for e in sorted(EDGE_VERTICES):
    print(f"    edge {e}: between vertices {EDGE_VERTICES[e]}")


# ============================================================================
# §4. Enumerate NB walks of length 3 (3 edges) on srs primitive cell
# ============================================================================
# Walker state: (current_vertex, last_edge_taken)
# At each step, walker takes one of the 3 incident edges, BUT cannot take
# the reverse of the last edge (NB constraint).
#
# Since each edge is between 2 specific vertices (not directed in the graph;
# the toggle T_e is involutive so it traverses both ways), "reverse of last
# edge" = the same edge label. NB constraint excludes 1 of the 3 options.
# So 2 NB choices per step.
#
# Walker takes 3 steps → traverses 3 edges → visits 4 vertices (no cycle
# possible at length 3 < g = 10).
#
# Per framework's μ:
#   - 1st step: any of 6 edges, weight 1/6 each (uniform per Jaynes)
#   - But we condition on a starting vertex; at each vertex, 3 incident
#     edges → weight 1/3 each (the per-vertex conditional measure).
#   - 2nd step (NB): 2 NB-allowed edges at v_1, weight 1/2 each.
#   - 3rd step (NB): 2 NB-allowed edges at v_2, weight 1/2 each.
# Since srs is vertex-transitive, walks from any starting vertex have the
# same statistics. Average over starting vertex = uniform per starting
# vertex = 1/4 each.

print()
print("=" * 72)
print("§4. NB walks of length 3 — enumeration with Jaynes weighting")
print("=" * 72)

walks_with_weight = []  # list of ((e_1, e_2, e_3), weight)

for start_vertex in range(4):
    incident_edges_v0 = sorted(VERTEX_EDGES[start_vertex])
    for e1 in incident_edges_v0:
        # 1st edge
        # Move to next vertex
        v1 = [v for v in EDGE_VERTICES[e1] if v != start_vertex][0]
        # NB constraint: can't take e1 as 2nd edge
        incident_edges_v1 = sorted(VERTEX_EDGES[v1] - {e1})
        assert len(incident_edges_v1) == 2
        for e2 in incident_edges_v1:
            v2 = [v for v in EDGE_VERTICES[e2] if v != v1][0]
            incident_edges_v2 = sorted(VERTEX_EDGES[v2] - {e2})
            assert len(incident_edges_v2) == 2
            for e3 in incident_edges_v2:
                # Per Jaynes (uniform per step at each vertex's incidence
                # set, with NB filter): weight = 1/3 × 1/2 × 1/2 = 1/12
                # Plus 1/4 for uniform starting vertex
                weight = Fraction(1, 4) * Fraction(1, 3) * Fraction(1, 2) * Fraction(1, 2)
                walks_with_weight.append(((e1, e2, e3), weight))

total_weight = sum(w for _, w in walks_with_weight)
print(f"  Total NB walks of length 3: {len(walks_with_weight)}")
print(f"  Sum of weights: {total_weight} (expected 1)")
assert total_weight == 1


# ============================================================================
# §5. Classify each walk's edge triple — Fano vs non-Fano
# ============================================================================
# A walk's 3 edges (e_1, e_2, e_3) form a Fano line ⇔ collinear in Fano plane
# ⇔ the set {e_1, e_2, e_3} ∈ FANO_LINES.
#
# But edges in a walk can REPEAT (e.g., (T_a, T_b, T_a) if vertex incidences
# allow). For repeated edges, the "triple" has fewer than 3 distinct points,
# so the Fano-line check requires distinct edges. If e_1 = e_3, the walk's
# 3-letter window has only 2 distinct toggles — the associator [e_a, e_b, e_a]
# is something specific in 𝕆. Let me handle this case separately.

print()
print("=" * 72)
print("§5. Classify NB-walk 3-letter windows: Fano vs non-Fano")
print("=" * 72)

n_distinct = 0
n_repeat = 0
fano_weight = Fraction(0)
non_fano_distinct_weight = Fraction(0)
repeat_weight = Fraction(0)

for triple, w in walks_with_weight:
    distinct = len(set(triple))
    if distinct == 3:
        n_distinct += 1
        if frozenset(triple) in FANO_LINES:
            fano_weight += w
        else:
            non_fano_distinct_weight += w
    else:
        # Repeats: e_1 = e_3 (since e_2 ≠ e_1 and e_2 ≠ e_3 by NB)
        # So distinct = 2; 3-letter window is (e_a, e_b, e_a)
        n_repeat += 1
        repeat_weight += w

print(f"  Walks with 3 distinct edges: {n_distinct}, total weight = {n_distinct/len(walks_with_weight):.4f}")
print(f"  Walks with repeated edges (e_1 = e_3): {n_repeat}, total weight = {n_repeat/len(walks_with_weight):.4f}")
print()
print(f"  Among 3-distinct walks:")
print(f"    Fano line (associative): weight = {fano_weight} = {float(fano_weight):.4f}")
print(f"    Non-Fano (non-associative): weight = {non_fano_distinct_weight} = {float(non_fano_distinct_weight):.4f}")
print(f"  Repeats (2 distinct edges, e_1 = e_3): weight = {repeat_weight} = {float(repeat_weight):.4f}")
print()


# ============================================================================
# §6. Compute f_3 — fraction of non-associative triples
# ============================================================================

# The associator [a, b, c] for a, b, c ∈ 𝕆:
#   - If {a, b, c} are 3 distinct Fano-collinear units: [a, b, c] = 0 (associative on this Fano line, ℍ-like)
#   - If {a, b, c} are 3 distinct non-collinear units: [a, b, c] ≠ 0 (the canonical octonion non-associativity)
#   - If a = c (repeat): [a, b, a] = (ab)a − a(ba). For octonion units e_a, e_b: this is in general non-zero
#     unless e_b commutes with e_a (which only holds for e_b = e_a).
#     For e_a ≠ e_b: associator [e_a, e_b, e_a] = (e_a e_b) e_a - e_a (e_b e_a) = ...
#     For octonions, e_a e_b = sign · e_c where e_c = e_a × e_b (Fano-line product) if they're on a Fano line, or non-associative if not.
#     Actually for e_a, e_b on the SAME Fano line (with e_c = e_a · e_b), the algebra restricted to {e_a, e_b, e_c} ≅ ℍ is associative.
#     For e_a ≠ e_b NOT on a Fano line — well, any 2 distinct units are ALWAYS on a unique Fano line per Fano-plane axioms. So e_a and e_b are always on SOME Fano line; the third element of that line is e_c = e_a · e_b.
#   So [e_a, e_b, e_a] computation in octonion algebra:
#     e_a · e_b = ε_{ab} · e_c where c = a·b (in Fano-line product), ε is sign
#     (e_a · e_b) · e_a = ε_{ab} · (e_c · e_a) = ε_{ab} · ε_{ca} · e_b
#                       (since e_c, e_a are on the same Fano line as e_b)
#     e_a · (e_b · e_a) = e_a · (-e_b · e_a)  [since e_b · e_a = -e_a · e_b for distinct]
#                       wait that's not right either; e_a · e_b = -e_b · e_a only for distinct imaginary units
#     ...
#   For SAFETY, treat {e_a, e_b, e_a} repeat case as ASSOCIATIVE (associator = 0)
#   because in any subalgebra generated by 2 distinct units {e_a, e_b}, the algebra is associative
#   (it's the ℍ subalgebra ⟨e_a, e_b, e_a·e_b⟩ which is the Fano line through them; closure within ℍ → associative).
#   So associator [e_a, e_b, e_a] = 0 since all elements are in a single ℍ ⊂ 𝕆.

# Therefore: NON-associative ⇔ 3 distinct edges forming a non-Fano triple.
#            Associative ⇔ either Fano triple, OR repeat (in 2-element subalgebra).

f_3 = non_fano_distinct_weight  # weight of non-Fano-3-distinct walks
associative_weight = fano_weight + repeat_weight

print("=" * 72)
print("§6. f_3 verdict")
print("=" * 72)
print()
print(f"  Total weight: {fano_weight + non_fano_distinct_weight + repeat_weight} = 1 ✓")
print()
print(f"  f_3 (NB-walker windows with 3 distinct edges forming non-Fano triple):")
print(f"    = {f_3} = {float(f_3):.6f}")
print()
print(f"  Associative weight (Fano + repeats):")
print(f"    = {associative_weight} = {float(associative_weight):.6f}")
print()
print(f"  Naive uniform prediction (per dynamic_zoo doc): f_3 = 28/35 = {28/35:.4f}")
print(f"  srs combinatorial constraint: f_3 = {float(f_3):.4f}")
print()


# ============================================================================
# §7. Implications for Theorem 9
# ============================================================================

print("=" * 72)
print("§7. Implications for Theorem 9 PARTIAL")
print("=" * 72)
print()

if f_3 == 0:
    verdict = """
  f_3 = 0 EXACTLY — Theorem 9 PARTIAL closes at CO-DOMINANT-INACCESSIBLE-AT-
  3-LETTER-WALKS:

  All NB-walker 3-letter windows on srs primitive cell are either:
    (a) associative Fano lines (all 3 edges on a single vertex's Fano line)
    (b) repeats with e_1 = e_3 (2-distinct case, in ℍ subalgebra)

  → No 3-letter window samples a non-Fano (non-associative) octonion triple
    DIRECTLY. The Fano-deletion embedding's structure ensures associator-
    invisible direct mechanism.

  → Theorem 9 closes ONLY for direct-mechanism octonion access via 3-letter
    windows. Indirect access via M2-M7 (Albert algebra E_6, Aut(𝕆) = G_2,
    cooling-cascade transients, etc.) was audited NEGATIVE in
    `M_mechanisms_synthesis_2026-05-07.md`.

  → BUT: Theorem 9 PARTIAL framing (per user correction 79e9406) preserves
    formal A2-T plural retention of non-Cl content even though direct-
    walker access is f_3 = 0. The non-closure of Theorem 9 was specifically
    about NOT making non-Cl content provably-inaccessible. f_3 = 0 at the
    direct walker level is consistent with non-closure: subdominant 𝕆 / E_8
    content remains formally retained even though direct walker mechanism
    doesn't sample it.
"""
elif float(f_3) < 0.05:
    verdict = f"""
  f_3 = {float(f_3):.4f} (very small but nonzero) — Theorem 9 PARTIAL leans
  toward CO-DOMINANT-with-near-zero-direct-suppression. Some 3-letter windows
  on srs do hit non-Fano triples, but at very low rate.

  → Layer-1 octonion content has a small but nonzero direct-walker access
    rate. Could in principle contribute observable corrections at order f_3
    × exp(-7) ~ small percent level. Worth checking against framework's
    residual deviations (y_τ +0.13%, λ_Higgs +0.52%, J_CKM +2.56%).
"""
elif float(f_3) > 0.5:
    verdict = f"""
  f_3 = {float(f_3):.4f} (large) — Theorem 9 SHARP-DOMINANT branch. Most
  walker 3-letter windows hit non-Fano (non-associative) triples. Direct
  octonion access is the dominant 3-letter sampling mode.

  → BUT Theorem 9 (REVISED, non-closure framing) is still PARTIAL: even
    though f_3 is large, the framework's PS predictions match observation
    (so any direct access is somehow not affecting them). The PARTIAL
    framing preserves room for non-Cl content; whether direct access at
    high f_3 is suppressed by additional mechanisms is the open question.
"""
else:
    verdict = f"""
  f_3 = {float(f_3):.4f} (intermediate) — Theorem 9 PARTIAL framing unchanged.

  → Direct octonion access at moderate rate. Whether Layer-1 escape
    candidates manifest at this rate depends on additional access-mechanism
    audit (M2-M7 mostly NEGATIVE per `M_mechanisms_synthesis_2026-05-07.md`).
"""

print(verdict)


# ============================================================================
# §8. Summary
# ============================================================================

print("=" * 72)
print("f_3 SRS EXPLICIT COMPUTATION — SUMMARY")
print("=" * 72)
print(f"""
SETUP:
  - srs primitive cell K_4 quotient (4 vertices, 6 edges)
  - MDL-preferred Fano-line embedding per Theorem 9 PARTIAL
  - Fano-deletion configuration (one octonion imaginary unit excluded)
  - NB walks of length 3 with Jaynes per-step weighting

ENUMERATION:
  - {len(walks_with_weight)} NB walks of length 3 (4 starts × 3 first-edges × 2 NB × 2 NB = 48)
  - Walks with 3 distinct edges: {n_distinct} ({n_distinct/len(walks_with_weight):.2%})
  - Walks with repeated edge (e_1 = e_3): {n_repeat} ({n_repeat/len(walks_with_weight):.2%})

CLASSIFICATION:
  - Fano line (associative within ℍ ⊂ 𝕆): weight {float(fano_weight):.4f}
  - Non-Fano (non-associative, octonion content): weight {float(non_fano_distinct_weight):.4f}
  - Repeats (associative in 2-generator ℍ subalgebra): weight {float(repeat_weight):.4f}

f_3 RESULT:
  - srs-specific value: f_3 = {f_3} = {float(f_3):.6f}
  - Naive uniform prediction (28/35 = {28/35:.4f}) NOT matched

VERDICT:
  - Theorem 9 PARTIAL closure status (per dynamic_zoo doc gate):
    {"f_3 = 0 EXACTLY (CO-DOMINANT direct-inaccessible)" if f_3 == 0 else f"f_3 = {float(f_3):.4f} (intermediate)"}
""")
