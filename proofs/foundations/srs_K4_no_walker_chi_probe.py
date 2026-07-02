#!/usr/bin/env python3
"""
P1.4 — Verify γ_7^A on srs's K_4 IS trivial (no walker-level Z_2 supercharge).

The χ̃ ≡ ±γ_7^A unification (`srs_z_gamma7_lift_recovers_chi.py`) showed that
on srs-z's bipartite Q_3 quotient, the half-bipartite product γ_7^A acts on
the walker as ±χ̃, anti-commuting with B(k) at all k. The structural claim
"no walker-level Z_2 on srs because K_4 isn't bipartite" rests on the absence
of a canonical bipartition.

This probe verifies the claim numerically. The question of whether ANY sign
assignment χ̃ on the 12 directed arcs of K_4 anti-commutes with B(k) is
k-INDEPENDENT: B[a',a] is nonzero (with some Bloch phase) iff a' is a
non-backtracking continuation of a, and that pattern is fixed by the abstract
arc digraph. Anti-commutation requires χ̃[a'] = −χ̃[a] for EVERY continuation
pair (a, a'). So the test reduces to a structural 2-coloring question on the
non-backtracking continuation digraph.

Three layers of verification:

  (1) **Vertex-induced lifts.** For each nonempty proper subset A ⊂ V(K_4),
      define χ̃^A on the directed-arc space by χ̃^A_a := +1 if tail(a) ∈ A,
      else −1. Check that the resulting χ̃ has at least one continuation
      pair (a, a') with χ̃[a'] = +χ̃[a]. Show all 14 subsets fail.

  (2) **Brute-force 2-coloring search.** Enumerate ALL 2^12 = 4096 sign
      assignments on the 12 directed arcs. For each, count violating
      continuation pairs. Show NO non-trivial assignment gives zero
      violations.

  (3) **Structural odd-cycle obstruction.** Anti-commutation requires
      consecutive arcs (a, a' continuation) to have OPPOSITE signs — a
      proper 2-coloring of the continuation digraph viewed as an
      undirected constraint graph. K_4 contains triangles → the
      Hashimoto continuation digraph contains 3-cycles → odd cycle in
      the constraint graph → no proper 2-coloring exists.

Outcome: γ_7^A on srs's K_4 cannot recover χ̃ on the walker. The walker-
level Z_2 supercharge structure REQUIRES the bipartite primitive quotient.
"""

import numpy as np
from itertools import combinations


def build_K4_arcs():
    """K_4: 4 vertices, every pair adjacent → 6 undirected edges → 12 directed arcs.

    Returns:
      arcs: list of (tail, head) over 12 arcs (no shifts — abstract K_4).
      cont: set of (i, j) where arc j is a non-backtracking continuation of arc i,
            i.e., tail(j) = head(i) AND (tail(j), head(j)) != (head(i), tail(i)).
    """
    arcs = []
    for u in range(4):
        for v in range(4):
            if u != v:
                arcs.append((u, v))
    n_arcs = len(arcs)
    assert n_arcs == 12

    cont = set()
    for i, (ti, hi) in enumerate(arcs):
        for j, (tj, hj) in enumerate(arcs):
            if tj == hi and not (tj == hi and hj == ti):
                cont.add((i, j))
    return arcs, cont


def violations(signs, cont):
    """Number of continuation pairs (i, j) with signs[i] == signs[j] (should be opposite)."""
    return sum(1 for (i, j) in cont if signs[i] == signs[j])


def main():
    print("=" * 78)
    print("P1.4 — γ_7^A on srs's K_4 walker: NO non-trivial Z_2 anti-commutes with B")
    print("=" * 78)

    # --- Step 1: Build K_4 abstract walker -----------------------------------
    arcs, cont = build_K4_arcs()
    n_arcs = len(arcs)
    n_cont = len(cont)
    print(f"\n--- Step 1: Abstract K_4 directed-arc walker ---")
    print(f"  |V| = 4, |E| = 6 (every pair adjacent), |arcs| = {n_arcs}")
    print(f"  Non-backtracking continuation pairs: {n_cont}")
    print(f"  (Each arc has 2 continuations: 3 outgoing from head minus 1 reverse)")

    # --- Step 2: Confirm K_4 has 3-cycles (non-bipartite) -------------------
    print(f"\n--- Step 2: K_4 is non-bipartite (contains 3-cycles) ---")
    # Triangle 0-1-2: arcs (0→1), (1→2), (2→0). Each consecutive pair is a continuation.
    triangle = [(0, 1), (1, 2), (2, 0)]
    arc_idx = {arc: i for i, arc in enumerate(arcs)}
    tri_idx = [arc_idx[a] for a in triangle]
    print(f"  Triangle vertex sequence: 0 → 1 → 2 → 0")
    print(f"  Arc indices: {tri_idx} = {triangle}")
    for k in range(3):
        i, j = tri_idx[k], tri_idx[(k + 1) % 3]
        is_cont = (i, j) in cont
        print(f"    arc {triangle[k]} → arc {triangle[(k+1)%3]}: continuation? {is_cont}")
    print(f"  → 3-cycle in continuation digraph confirmed (forces odd-cycle constraint).")

    # --- Step 3: Vertex-induced χ̃^A — enumerate all 14 subsets -------------
    print(f"\n--- Step 3: Vertex-induced χ̃^A (χ̃[a] = +1 if tail(a) ∈ A else −1) ---")
    print(f"  Test: count continuation pairs (i, j) with χ̃[i] = χ̃[j] (violations)")
    print(f"  Anti-commutation requires zero violations.\n")
    print(f"  {'A':<14s} {'violations / total':<22s} {'verdict'}")
    print(f"  " + "-" * 58)
    all_subsets = []
    for size in (1, 2, 3):
        for combo in combinations(range(4), size):
            all_subsets.append(set(combo))

    min_violations_vertex = n_cont
    for A_set in all_subsets:
        signs = np.array([+1 if t in A_set else -1 for (t, h) in arcs])
        v = violations(signs, cont)
        if v < min_violations_vertex:
            min_violations_vertex = v
        verdict = "✓ ANTI-COMMUTES" if v == 0 else f"✗ {v} violations"
        print(f"  {str(sorted(A_set)):<14s} {v:>3d} / {n_cont:<14d}  {verdict}")
    print(f"\n  Min violations across all 14 vertex-induced χ̃^A: {min_violations_vertex}")
    if min_violations_vertex > 0:
        print(f"  → No vertex-induced χ̃^A on K_4 anti-commutes with B.")

    # --- Step 4: Brute-force search over ALL 2^12 sign assignments -----------
    print(f"\n--- Step 4: Brute-force search over all 2^12 = 4096 sign assignments ---")
    best_violations = n_cont
    best_signs = None
    nontrivial_count = 0
    perfect_assignments = 0
    for mask in range(2 ** n_arcs):
        signs = np.array([+1 if (mask >> i) & 1 else -1 for i in range(n_arcs)])
        # Skip trivial χ̃ = ±I (uniform)
        if np.all(signs == signs[0]):
            continue
        nontrivial_count += 1
        v = violations(signs, cont)
        if v == 0:
            perfect_assignments += 1
        if v < best_violations:
            best_violations = v
            best_signs = signs.copy()

    print(f"  Searched: {nontrivial_count} non-trivial sign assignments")
    print(f"  Assignments with ZERO violations (true anti-commutation): {perfect_assignments}")
    print(f"  Best (minimum) violations achieved: {best_violations} / {n_cont}")
    if best_signs is not None:
        plus = int(np.sum(best_signs > 0))
        minus = int(np.sum(best_signs < 0))
        print(f"  Best assignment counts: +1: {plus},  −1: {minus}")
    if perfect_assignments == 0:
        print(f"  ✓ NO non-trivial 2-coloring of K_4's arcs anti-commutes with B at any k.")

    # --- Step 5: Structural odd-cycle obstruction ---------------------------
    print(f"\n--- Step 5: Odd-cycle obstruction (structural proof) ---")
    # Continuation digraph contains a 3-cycle: arc(0→1) → arc(1→2) → arc(2→0) → arc(0→1)
    a, b, c = arc_idx[(0, 1)], arc_idx[(1, 2)], arc_idx[(2, 0)]
    print(f"  3-cycle in continuation digraph: arc {arcs[a]} → arc {arcs[b]} → arc {arcs[c]} → arc {arcs[a]}")
    print(f"  Anti-commutation forces opposite signs on every continuation edge:")
    print(f"     χ̃[a] = −χ̃[b],  χ̃[b] = −χ̃[c],  χ̃[c] = −χ̃[a]")
    print(f"  Composing around the 3-cycle:")
    print(f"     χ̃[a] = −χ̃[b] = +χ̃[c] = −χ̃[a]  →  2χ̃[a] = 0  →  χ̃[a] = 0.")
    print(f"  But χ̃² = I requires χ̃[a] ∈ {{±1}}, contradicting χ̃[a] = 0.")
    print(f"  → No nonzero 2-coloring of K_4's arcs anti-commutes with B.")

    # --- Final summary ------------------------------------------------------
    print(f"\n" + "=" * 78)
    print(f"Conclusion")
    print(f"=" * 78)
    print(f"""
  Three independent verifications of the structural claim:

    (1) NO vertex-induced lift χ̃^A (over all 14 nonempty proper subsets A ⊂
        V(K_4)) anti-commutes with B. Best (minimum) violation count:
        {min_violations_vertex} / {n_cont} continuation pairs.

    (2) Brute-force search over all 4094 non-trivial sign assignments to the
        12 arcs finds NO assignment with anti-commutation. Best (minimum)
        violations: {best_violations} / {n_cont}. ZERO assignments achieve
        anti-commutation.

    (3) Structural odd-cycle obstruction: the Hashimoto continuation digraph
        on K_4 contains 3-cycles (one per triangle in K_4 × NB direction),
        which force any anti-commuting sign assignment to satisfy
        χ̃[a] = −χ̃[a], i.e., χ̃[a] = 0. Contradicts χ̃² = I.

  Therefore γ_7^A on srs's K_4 is trivial — there is NO non-trivial walker-
  level Z_2 grading that anti-commutes with B(k). The walker-level
  supercharge structure is genuinely a srs-z-specific phenomenon (bipartite-
  quotient substrates only). The single Cl(6) chirality γ_7 lives on srs
  only at the per-vertex level, where it is trivial on the walker's
  F_total = 1 sector — exactly as the original Layer-5 trace concluded,
  and consistent with the χ̃ ≡ ±γ_7^A unification framing on srs-z.

  Roadmap item P1.4 closed.
""")


if __name__ == '__main__':
    main()
