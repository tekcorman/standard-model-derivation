#!/usr/bin/env python3
# ============================================================
# Sprint gamma (3b) spike: C_3-equivariant 3-way pair-partitions
#                         of the 6 K_4 edges (gap (G.a))
# ============================================================
#
# Setup. K_4 has 6 unordered edges. A 3-way pair-partition is a
# partition of the edge set into 3 disjoint pairs. The total number
# of such partitions is 6!/(2^3 * 3!) = 15.
#
# C_3 acts on edges via sigma = (v_0)(v_1 v_3 v_2). A partition P is
# C_3-equivariant iff sigma sends P to itself setwise, i.e.
# sigma({P_1, P_2, P_3}) = {P_1, P_2, P_3} as an unordered set.
#
# Question (G.a). Among the 15 pair-partitions, which are
# C_3-equivariant? If the matching-partition (perfect matchings of
# K_4) is the unique C_3-equivariant partition, then the Brauer-Weyl
# pairing of Cl(6,0) in B3 -- if required to be C_3-equivariant --
# is forced to be the matching-partition, closing gap (G).
#
# Extension. Even if the matching-partition is not UNIQUE among
# C_3-equivariant partitions, we want to know: which C_3-equivariant
# partitions are "regular-rep" on pair-space (sigma acts as a 3-cycle
# on the 3 pairs) vs "trivial-rep" (sigma fixes each pair setwise)?
# The regular-rep ones are the ones that give the desired spinor/
# graph coupling.

import itertools
import sys


VERTICES = (0, 1, 2, 3)


def all_edges():
    return [frozenset(e) for e in itertools.combinations(VERTICES, 2)]


def sigma_vertex(v):
    return {0: 0, 1: 3, 2: 1, 3: 2}[v]


def sigma_edge(edge):
    return frozenset({sigma_vertex(u) for u in edge})


def all_pair_partitions(edges):
    """Enumerate all partitions of `edges` into 3 unordered pairs.

    Uses the standard recursion: fix one element in a pair, recurse
    on the rest.
    """
    edges = list(edges)
    if len(edges) == 0:
        return [frozenset()]
    if len(edges) == 2:
        return [frozenset({frozenset(edges)})]
    if len(edges) % 2 != 0:
        return []

    first = edges[0]
    rest = edges[1:]
    partitions = []
    for i in range(len(rest)):
        partner = rest[i]
        remaining = rest[:i] + rest[i + 1:]
        for sub_partition in all_pair_partitions(remaining):
            new_partition = frozenset({frozenset({first, partner})}) | sub_partition
            partitions.append(new_partition)
    return partitions


def apply_sigma_to_partition(partition):
    """sigma sends {{a,b}, {c,d}, ...} to {{sigma(a), sigma(b)}, ...}."""
    new = set()
    for pair in partition:
        new_pair = frozenset({sigma_edge(e) for e in pair})
        new.add(new_pair)
    return frozenset(new)


def is_c3_equivariant(partition):
    """sigma permutes the 3 pairs of `partition` setwise."""
    return apply_sigma_to_partition(partition) == partition


def sigma_action_on_pairs(partition):
    """Return the permutation induced by sigma on the 3 pairs.

    Only meaningful if partition is C_3-equivariant.
    Returns a list [idx_0, idx_1, idx_2] where idx_i is the index of
    sigma(pair_i) in the enumerated pair list.
    """
    pair_list = list(partition)
    result = []
    for p in pair_list:
        new_p = frozenset({sigma_edge(e) for e in p})
        result.append(pair_list.index(new_p))
    return result


def permutation_order(perm):
    """Order of a permutation given as list [p(0), p(1), ...]."""
    n = len(perm)
    identity = list(range(n))
    if perm == identity:
        return 1
    current = list(perm)
    for order in range(2, n * n + 1):
        current = [perm[current[i]] for i in range(n)]
        if current == identity:
            return order
    return -1


def is_perfect_matching(pair):
    """A pair of edges is a matching iff the edges are disjoint."""
    edge_list = list(pair)
    if len(edge_list) != 2:
        return False
    e1, e2 = edge_list
    return len(e1 & e2) == 0


def is_matching_partition(partition):
    """All 3 pairs are K_4 perfect matchings."""
    return all(is_perfect_matching(p) for p in partition)


def verify():
    edges = all_edges()
    all_parts = all_pair_partitions(edges)
    results = {}

    results["total_pair_partitions"] = len(all_parts)

    # Filter by C_3-equivariance
    c3_equi = [p for p in all_parts if is_c3_equivariant(p)]
    results["c3_equivariant_count"] = len(c3_equi)

    # For each C_3-equivariant partition, record the sigma-action order on pair-space
    equi_details = []
    for part in c3_equi:
        perm = sigma_action_on_pairs(part)
        order = permutation_order(perm)
        is_matching = is_matching_partition(part)
        equi_details.append({
            "partition": [
                sorted(tuple(sorted(e)) for e in pair)
                for pair in part
            ],
            "sigma_perm_on_pairs": perm,
            "order_on_pair_space": order,
            "is_matching_partition": is_matching,
        })

    results["c3_equivariant_partitions"] = equi_details

    # Count by type
    matching_equi = [d for d in equi_details if d["is_matching_partition"]]
    non_matching_equi = [d for d in equi_details if not d["is_matching_partition"]]
    results["matching_partitions_that_are_c3_equi"] = len(matching_equi)
    results["non_matching_c3_equi_partitions"] = len(non_matching_equi)

    # Regular-rep vs trivial-rep by order
    regular_rep = [d for d in equi_details if d["order_on_pair_space"] == 3]
    trivial_rep = [d for d in equi_details if d["order_on_pair_space"] == 1]
    results["regular_rep_partitions"] = len(regular_rep)
    results["trivial_rep_partitions"] = len(trivial_rep)

    # Decision
    uniqueness_matching = (len(matching_equi) == 1 and len(non_matching_equi) == 0)
    results["matching_is_unique_c3_equi"] = uniqueness_matching

    uniqueness_regular = (len(regular_rep) == 1 and regular_rep[0]["is_matching_partition"] if regular_rep else False)
    results["matching_is_unique_regular_rep"] = uniqueness_regular

    return results


if __name__ == "__main__":
    print("=" * 72)
    print("Gap (G.a): C_3-equivariant 3-way pair-partitions of K_4 edges")
    print("sigma = (v_0)(v_1 v_3 v_2)")
    print("=" * 72)
    print()

    r = verify()

    print(f"Total 3-way pair-partitions of 6 K_4 edges: "
          f"{r['total_pair_partitions']} (expected 15)")
    print(f"C_3-equivariant among these:                "
          f"{r['c3_equivariant_count']}")
    print()

    print("Listing of C_3-equivariant partitions:")
    for i, d in enumerate(r["c3_equivariant_partitions"]):
        print(f"  [{i}] partition: {d['partition']}")
        print(f"      sigma on pairs: {d['sigma_perm_on_pairs']}  "
              f"(order {d['order_on_pair_space']})")
        print(f"      is matching-partition? {d['is_matching_partition']}")
    print()

    print(f"matching-partitions that are C_3-equivariant: "
          f"{r['matching_partitions_that_are_c3_equi']} / 1 possible")
    print(f"non-matching C_3-equivariant partitions:      "
          f"{r['non_matching_c3_equi_partitions']}")
    print(f"regular-rep (order-3) partitions:             "
          f"{r['regular_rep_partitions']}")
    print(f"trivial-rep (order-1) partitions:             "
          f"{r['trivial_rep_partitions']}")
    print()
    print(f"Matching is UNIQUE C_3-equivariant partition?    "
          f"{r['matching_is_unique_c3_equi']}")
    print(f"Matching is UNIQUE regular-rep C_3-partition?    "
          f"{r['matching_is_unique_regular_rep']}")
    print()

    print("=" * 72)
    if r["matching_is_unique_c3_equi"]:
        print("RESULT: (G.a) CLOSES.")
        print()
        print("  Among the 15 pair-partitions of the 6 K_4 edges, the UNIQUE")
        print("  C_3-equivariant partition is the matching-partition (the")
        print("  3 perfect matchings of K_4).")
        print()
        print("  Consequence: if the Brauer-Weyl pairing of Cl(6,0) generators")
        print("  on H_{K_4} is required to be C_3-equivariant, it is forced to")
        print("  be the matching-partition. Under any C_3-equivariant refinement")
        print("  of B1.b's invariant Clifford construction, (T_1, T_2, Y) are")
        print("  cyclically permuted by sigma.")
        print()
        print("  This closes gap (G) IF C_3-equivariance of the Brauer-Weyl")
        print("  pairing can be derived from A1-A5. Gap (G) refines to (G'):")
        print("  is C_3-equivariance of the Brauer-Weyl pairing forced by A1-A5?")
    elif r["matching_is_unique_regular_rep"]:
        print("RESULT: (G.a) partially CLOSES.")
        print()
        print("  The matching-partition is the UNIQUE C_3-regular-rep (order-3)")
        print("  equivariant partition, but there are other equivariant")
        print("  partitions (those where sigma fixes pairs individually).")
        print("  (G) refines to: 'among C_3-equivariant partitions, why the")
        print("  regular-rep one rather than the trivial-rep one?'")
    else:
        print("RESULT: (G.a) does NOT close.")
        print()
        print("  The matching-partition is one of several C_3-equivariant")
        print("  partitions. Combinatorial uniqueness does not force the")
        print("  Brauer-Weyl pairing to match. Gap (G) remains open via (G.a).")
    print("=" * 72)
    sys.exit(0)
