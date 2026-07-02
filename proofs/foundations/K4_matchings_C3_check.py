#!/usr/bin/env python3
# ============================================================
# Sprint gamma (3b) spike: K_4 perfect matchings under C_3
# ============================================================
#
# Setup. The srs K_4 quotient has 4 vertices {v_0, v_1, v_2, v_3} and
# 6 edges e_{ij} = {v_i, v_j} for 0 <= i < j <= 3. The body-diagonal
# C_3 action on the primitive cell induces the vertex permutation
#   sigma = (v_0)(v_1 v_3 v_2).
# (Per ../../predictions/B_P_doubly_degenerate_h_derivation.md Step 2, verified.)
#
# Claim 1. K_4 has exactly 3 perfect matchings. A perfect matching is
# a set of disjoint edges that covers all 4 vertices.
#
# Claim 2. sigma permutes the three perfect matchings cyclically (as
# a 3-cycle). This means matching-space carries the regular Z_3
# representation.
#
# Significance for 3b chirality bridge. The Brauer-Weyl construction
# of B3 (../../predictions/theorem_B3_spinor_fermion_derivation.md Step 1) realizes Cl(6,0)
# via three nested Pauli factors, producing 3 Cartan bivector pairs
# (Gamma_1, Gamma_2), (Gamma_3, Gamma_4), (Gamma_5, Gamma_6) -> the
# Cartan generators T_1, T_2, Y. If these three pairs = the three
# perfect matchings of K_4, then sigma cyclically permutes (T_1, T_2, Y),
# which is consistent with (and sharpens) the docs/framework/B3_B6_reconciliation.md
# Finding 2: ||[T_a, U_{C_3}^S]|| = 2 (nonzero) for each T_a.
#
# Consequence candidate. T_1 (the up/down weight sign generator) is NOT
# C_3-invariant — it is one vertex of a C_3-3-cycle inside the Cartan.
# This is exactly the type of non-trivial walker-level spinor/graph
# coupling called for in V_us_derivation.md §3 input (iii).

import itertools
import sys


VERTICES = (0, 1, 2, 3)


def all_edges():
    """Return the 6 unordered edges of K_4 as frozensets."""
    return [frozenset(e) for e in itertools.combinations(VERTICES, 2)]


def perfect_matchings():
    """Return the list of perfect matchings of K_4.

    A perfect matching of K_4 is a set of 2 disjoint edges covering
    all 4 vertices.
    """
    edges = all_edges()
    matchings = []
    for combo in itertools.combinations(edges, 2):
        covered = set().union(*combo)
        if len(covered) == 4:
            matchings.append(frozenset(combo))
    # dedup
    return list(set(matchings))


def apply_sigma_to_vertex(v):
    """sigma = (v_0)(v_1 v_3 v_2): v_0 fixed; v_1 -> v_3, v_3 -> v_2, v_2 -> v_1."""
    return {0: 0, 1: 3, 2: 1, 3: 2}[v]


def apply_sigma_to_edge(edge):
    """sigma acts on edges by sigma({u, v}) = {sigma(u), sigma(v)}."""
    u, v = edge
    return frozenset({apply_sigma_to_vertex(u), apply_sigma_to_vertex(v)})


def apply_sigma_to_matching(matching):
    """sigma acts on matchings by component-wise action on edges."""
    return frozenset({apply_sigma_to_edge(e) for e in matching})


def verify():
    results = {}
    edges = all_edges()
    results["num_edges"] = len(edges)

    matchings = perfect_matchings()
    results["num_perfect_matchings"] = len(matchings)
    results["matchings"] = [
        sorted(tuple(sorted(e)) for e in m) for m in matchings
    ]

    # C1: K_4 has exactly 3 perfect matchings
    c1 = (len(matchings) == 3)
    results["C1_three_matchings"] = c1

    # C2: sigma maps matchings to matchings (matching-space is sigma-invariant)
    sigma_matchings = [apply_sigma_to_matching(m) for m in matchings]
    c2 = all(sm in matchings for sm in sigma_matchings)
    results["C2_sigma_preserves_matchings"] = c2

    # C3: sigma acts on matching-space as a 3-cycle
    # (not identity, not a transposition)
    permutation = [matchings.index(apply_sigma_to_matching(m)) for m in matchings]
    results["C3_sigma_permutation_on_matchings"] = permutation

    # order of the permutation
    order = 1
    p = list(permutation)
    ident = list(range(len(matchings)))
    while p != ident:
        p = [permutation[p[i]] for i in range(len(permutation))]
        order += 1
        if order > 10:
            break
    results["C3_sigma_order_on_matchings"] = order

    # C3: expect order 3 AND permutation is not identity
    c3 = (order == 3 and permutation != ident)
    results["C3_sigma_is_3cycle"] = c3

    # C4: edges are partitioned into two C_3-orbits under sigma
    # (star around v_0 vs rim edges)
    edge_orbits = []
    seen = set()
    for e in edges:
        if e in seen:
            continue
        orbit = [e]
        seen.add(e)
        next_e = apply_sigma_to_edge(e)
        while next_e != e:
            orbit.append(next_e)
            seen.add(next_e)
            next_e = apply_sigma_to_edge(next_e)
        edge_orbits.append(orbit)
    results["edge_orbits"] = [
        sorted(tuple(sorted(e)) for e in orbit) for orbit in edge_orbits
    ]
    results["num_edge_orbits"] = len(edge_orbits)
    results["edge_orbit_lengths"] = sorted([len(o) for o in edge_orbits])

    # Expected: two orbits of length 3 each (6 = 3+3)
    c4 = (len(edge_orbits) == 2 and sorted([len(o) for o in edge_orbits]) == [3, 3])
    results["C4_two_3orbits"] = c4

    # C5: verify each matching is a TRANSVERSAL of the two edge orbits
    # (exactly 1 edge from each orbit in each matching)
    transversal_ok = True
    for m in matchings:
        count_per_orbit = [0] * len(edge_orbits)
        for e in m:
            for i, orbit in enumerate(edge_orbits):
                if e in orbit:
                    count_per_orbit[i] += 1
        if count_per_orbit != [1, 1]:
            transversal_ok = False
            break
    results["C5_matchings_are_transversals"] = transversal_ok

    all_passed = (c1 and c2 and c3 and c4 and transversal_ok)
    results["ALL_PASSED"] = all_passed

    return results


if __name__ == "__main__":
    print("=" * 72)
    print("Sprint gamma (3b) spike: K_4 perfect matchings under C_3")
    print("sigma = (v_0)(v_1 v_3 v_2): body-diagonal C_3 on srs primitive cell")
    print("=" * 72)
    print()

    r = verify()

    print(f"K_4 total edges:                  {r['num_edges']} (expected 6)")
    print(f"K_4 perfect matchings:            {r['num_perfect_matchings']} (expected 3)")
    print()
    print("The three perfect matchings:")
    for i, m in enumerate(r["matchings"]):
        print(f"  M_{i+1}: {m}")
    print()
    print("Edge orbits under sigma:")
    for i, orbit in enumerate(r["edge_orbits"]):
        print(f"  orbit {i+1}: {orbit}")
    print(f"  total orbits: {r['num_edge_orbits']}, lengths: {r['edge_orbit_lengths']}")
    print()
    print(f"sigma permutation on (M_1, M_2, M_3) as indices: "
          f"{r['C3_sigma_permutation_on_matchings']}")
    print(f"order of sigma on matching-space: "
          f"{r['C3_sigma_order_on_matchings']}")
    print()
    print("Check results:")
    print(f"  C1 (3 matchings):               {r['C1_three_matchings']}")
    print(f"  C2 (sigma preserves matchings): {r['C2_sigma_preserves_matchings']}")
    print(f"  C3 (sigma is 3-cycle):          {r['C3_sigma_is_3cycle']}")
    print(f"  C4 (two edge 3-orbits):         {r['C4_two_3orbits']}")
    print(f"  C5 (matchings are transversals):{r['C5_matchings_are_transversals']}")
    print()

    assert r["ALL_PASSED"], f"Some checks failed: {r}"

    print("=" * 72)
    print("RESULT:")
    print()
    print("  K_4 has exactly 3 perfect matchings. Each is a transversal of")
    print("  the two C_3 edge orbits (star + rim). sigma = (v_0)(v_1 v_3 v_2)")
    print("  permutes the 3 matchings as a 3-cycle.")
    print()
    print("  Matching-space of K_4 is thus a C_3-equivariant 3-element set,")
    print("  realizing the regular Z_3 representation (by the R3 result:")
    print("  faithful Z_3 on a 3-element set is unique up to iso).")
    print()
    print("  Significance for 3b: if Brauer-Weyl pair (Gamma_{2a-1}, Gamma_{2a})")
    print("  corresponds canonically to perfect matching M_a of K_4, then the")
    print("  three B3 Cartan generators (T_1, T_2, Y) are cyclically permuted")
    print("  by sigma, which matches (and sharpens) B3_B6_reconciliation.md's")
    print("  finding that [T_a, U_{C_3}^S] = 2.0 (nonzero) for each a.")
    print()
    print("  This does NOT yet close 3b — a derivation 'MDL forces Brauer-Weyl")
    print("  pair = K_4 perfect matching' is still needed. But it is a new")
    print("  candidate route that places the spinor Cartan and the graph C_3")
    print("  in a single C_3-equivariant object (matching-space).")
    print()
    print("OK: K_4_matchings_C3_check computation complete.")
    print("=" * 72)
    sys.exit(0)
