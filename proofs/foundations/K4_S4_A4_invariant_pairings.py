#!/usr/bin/env python3
# ============================================================
# Sprint gamma (3b) spike: S_4 / A_4 invariance of K_4 pair-partitions
#                         (gap (G.a'.β))
# ============================================================
#
# Context. K4_C3_equivariant_pairings.py showed: among the 15 three-way
# pair-partitions of the 6 K_4 edges, exactly 3 are C_3-equivariant
# under sigma = (v_0)(v_1 v_3 v_2) — the matching-partition P_M and
# two vertex-sharing partitions P_A, P_B.
#
# Question (G.a'.β). Among these three C_3-equivariant partitions,
# which are invariant under larger subgroups of S_4 = Sym({v_0,...,v_3})?
#
# Specifically: A_4 (alternating group, order 12, = rotation subgroup
# of the tetrahedron's full symmetry group) and S_4 (full permutation
# group, order 24, = octahedral rotation group 432 acting on the 4
# cube-body-diagonals).
#
# Physical significance. The srs primitive cell has 4 K_4 vertices at
# Wyckoff 8a positions of I4_1 32. The point group is 432 = O ≅ S_4,
# acting on the 4 body-diagonal directions (= 4 primitive-cell
# vertices) as the full symmetric group S_4. If the matching-partition
# is the UNIQUE S_4-invariant pair-partition among the 3 C_3-equivariant
# candidates, then the Brauer-Weyl pairing of Cl(6,0) generators on
# H_{K_4} is forced to be the matching-partition, whenever the
# Brauer-Weyl construction respects the full primitive-cell vertex
# symmetry S_4 (not just C_3).

import itertools
import sys


VERTICES = (0, 1, 2, 3)


def all_edges():
    return [frozenset(e) for e in itertools.combinations(VERTICES, 2)]


def permute_edge(edge, perm):
    """perm is a tuple (perm[0], perm[1], perm[2], perm[3]) giving the
    image of each vertex."""
    return frozenset({perm[v] for v in edge})


def permute_partition(partition, perm):
    return frozenset(
        frozenset({permute_edge(e, perm) for e in pair})
        for pair in partition
    )


def is_invariant(partition, group_elements):
    return all(permute_partition(partition, perm) == partition for perm in group_elements)


def permutation_sign(perm):
    """Sign of a permutation on {0,1,2,3}: +1 for even, -1 for odd."""
    n = len(perm)
    sign = 1
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                sign = -sign
    return sign


def S4_elements():
    return list(itertools.permutations(VERTICES))


def A4_elements():
    return [p for p in itertools.permutations(VERTICES) if permutation_sign(p) == 1]


def c3_equivariant_partitions():
    """The 3 C_3-equivariant partitions from K4_C3_equivariant_pairings.py."""
    E = {}
    for e in all_edges():
        key = tuple(sorted(e))
        E[key] = e

    P_M = frozenset({
        frozenset({E[(0, 3)], E[(1, 2)]}),
        frozenset({E[(0, 1)], E[(2, 3)]}),
        frozenset({E[(0, 2)], E[(1, 3)]}),
    })
    P_A = frozenset({
        frozenset({E[(0, 1)], E[(1, 2)]}),
        frozenset({E[(0, 2)], E[(2, 3)]}),
        frozenset({E[(0, 3)], E[(1, 3)]}),
    })
    P_B = frozenset({
        frozenset({E[(0, 1)], E[(1, 3)]}),
        frozenset({E[(0, 2)], E[(1, 2)]}),
        frozenset({E[(0, 3)], E[(2, 3)]}),
    })
    return {"matching P_M": P_M, "vertex-sharing P_A": P_A, "vertex-sharing P_B": P_B}


def verify():
    results = {}
    partitions = c3_equivariant_partitions()
    A4 = A4_elements()
    S4 = S4_elements()
    results["A4_size"] = len(A4)
    results["S4_size"] = len(S4)

    details = {}
    for name, P in partitions.items():
        A4_inv = is_invariant(P, A4)
        S4_inv = is_invariant(P, S4)

        # find which S_4 elements DON'T preserve P (counterexample)
        counterexample = None
        if not S4_inv:
            for perm in S4:
                if permute_partition(P, perm) != P:
                    counterexample = perm
                    break

        details[name] = {
            "A4_invariant": A4_inv,
            "S4_invariant": S4_inv,
            "S4_counterexample_perm": counterexample,
        }

    results["partition_details"] = details

    # Counts
    A4_invariant_partitions = [n for n, d in details.items() if d["A4_invariant"]]
    S4_invariant_partitions = [n for n, d in details.items() if d["S4_invariant"]]
    results["A4_invariant"] = A4_invariant_partitions
    results["S4_invariant"] = S4_invariant_partitions

    # (G.a'.β) decision: is matching unique S_4-invariant?
    gab_closes_S4 = (
        len(S4_invariant_partitions) == 1
        and "matching P_M" in S4_invariant_partitions
    )
    gab_closes_A4 = (
        len(A4_invariant_partitions) == 1
        and "matching P_M" in A4_invariant_partitions
    )
    results["Gab_closes_via_S4"] = gab_closes_S4
    results["Gab_closes_via_A4"] = gab_closes_A4

    return results


if __name__ == "__main__":
    print("=" * 72)
    print("Gap (G.a'.β): S_4 / A_4 invariance of K_4 pair-partitions")
    print("=" * 72)
    print()

    r = verify()

    print(f"A_4 (order {r['A4_size']}): alternating group, = I4_1 32 primitive")
    print(f"     cell vertex-permutation subgroup (rotation-only).")
    print(f"S_4 (order {r['S4_size']}): full symmetric group on 4 vertices,")
    print(f"     = 432 point group acting on 4 body-diagonals of cube.")
    print()

    print("Per-partition invariance:")
    for name, d in r["partition_details"].items():
        print(f"  {name}:")
        print(f"    A_4 invariant? {d['A4_invariant']}")
        print(f"    S_4 invariant? {d['S4_invariant']}")
        if d["S4_counterexample_perm"] is not None:
            print(f"    S_4 counterexample permutation: {d['S4_counterexample_perm']}")
    print()
    print(f"A_4-invariant C_3-equivariant partitions: {r['A4_invariant']}")
    print(f"S_4-invariant C_3-equivariant partitions: {r['S4_invariant']}")
    print()
    print(f"(G.a'.β) closes via S_4 invariance? {r['Gab_closes_via_S4']}")
    print(f"(G.a'.β) closes via A_4 invariance? {r['Gab_closes_via_A4']}")
    print()

    print("=" * 72)
    if r["Gab_closes_via_A4"]:
        print("RESULT: (G.a'.β) CLOSES via A_4 invariance.")
        print()
        print("  Among the 3 C_3-equivariant pair-partitions of K_4 edges,")
        print("  ONLY the matching-partition P_M is A_4-invariant.")
        print("  (P_A and P_B fail A_4 invariance because they single out")
        print("   the center vertex v_0, which A_4 moves.)")
        print()
        print("  Since the srs primitive-cell vertex-permutation group is at")
        print("  least A_4 (from I4_1 32 rotations), the Brauer-Weyl pairing")
        print("  of Cl(6,0) generators — if required to be invariant under the")
        print("  ambient vertex-symmetry — is FORCED to be the matching-")
        print("  partition.")
        print()
        print("  Gap (G) refines sharply: closing 3b now requires only")
        print("  'Brauer-Weyl pairing inherits the primitive-cell vertex")
        print("  symmetry group (A_4 or larger).' That is a much cleaner")
        print("  structural claim than the original 'JW ordering from A4'.")
    elif r["Gab_closes_via_S4"]:
        print("RESULT: (G.a'.β) CLOSES via S_4 invariance only.")
        print()
        print("  Matching is unique under S_4, but A_4 preserves other")
        print("  partitions too. To force matching, need ambient symmetry to")
        print("  include at least one S_4 \\ A_4 (odd) element.")
    else:
        print("RESULT: (G.a'.β) DOES NOT close.")
        print()
        print("  Multiple partitions are A_4-invariant. Matching is not")
        print("  uniquely forced even under the full rotation subgroup.")
    print("=" * 72)
    sys.exit(0)
