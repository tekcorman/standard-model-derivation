# ============================================================
# THEOREM: B1 — No MDL-canonical ordering of K_4 quotient edges
# ============================================================
#
# Audit anchor: Row 15b of `docs/audits/registers/uniqueness_ledger.md` (B1 ordering OPEN
# globally; CAR/JW closure UNIQUE locally). This theorem documents the
# obstruction to a global canonical ordering — load-bearing for the A4
# axiom-elimination workstream.

# --- THEOREM STATEMENT ---------------------------------------
# S_4 = Aut(K_4) partitions the 720 orderings of the 6 edges of K_4 into
# 30 free orbits of size 24; all orbits have identical two-part MDL cost;
# K_4 has no Eulerian trail; therefore no MDL-canonical ordering of the
# K_4-quotient edges is forced by the framework axiom set A1 + A2-T, and the
# Clifford algebra Cl(V,Q) must be defined via the S_6-equivariant
# tensor-algebra quotient.  Verdict: B1.b.
# Status: STRICT-SOLID (theorem-grade, computationally verified)

# --- FRAMEWORK AXIOMS INVOKED --------------------------------
# A1 (self-inverse binary toggle): enters via the srs NB-walker dynamics
#    and the K_4 quotient of the srs primitive cell.
# A2 (MDL): the two-part code-length criterion (Grunwald 2007 §5.3) that
#    is claimed to select a canonical ordering; Step 3 shows it does not.

# --- INPUTS --------------------------------------------------
# K_4 graph: the 4-vertex complete graph arising as the primitive-cell
# quotient of the srs lattice (I4_1 32, Wyckoff 8a).
# Aut(K_4) = S_4 (Dummit & Foote 2004 §2.2 Example 4).
# Line graph L(K_4) = K_{2,2,2} = octahedron (Harary 1969 §8).
# NB walk on srs projected to K_4 (theorem_walker_dynamics).

# --- IMPLEMENTATION ------------------------------------------
# The proof script proofs/foundations/theorem_B1_ordering.py contains the
# full enumeration.  This predictions entry imports and re-runs the four
# proof steps to produce a verifiable True/False result for the DAG.

import math
import sys
from collections import Counter
from itertools import permutations
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # repo root (file now under proofs/foundations/)
sys.path.insert(0, str(REPO))

# K_4 combinatorics — vertex / edge counts sourced from leaf primitives
sys.path.insert(0, str(REPO / "predictions"))
from V_count import V_count_pred as N_V   # = 4
from E_count import E_count_pred as N_E   # = 6 via handshake 2|E| = k·|V|
EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
FACT_6 = 720


def apply_sigma_to_edge(sigma, edge):
    a, b = sigma[edge[0]], sigma[edge[1]]
    return (min(a, b), max(a, b))


def edge_permutation(sigma):
    return tuple(EDGES.index(apply_sigma_to_edge(sigma, e)) for e in EDGES)


def build_S4_action():
    S4_vertex = list(permutations(range(N_V)))
    assert len(S4_vertex) == 24
    S4_edge = [edge_permutation(s) for s in S4_vertex]
    assert len(set(S4_edge)) == 24
    return S4_edge


def act_on_ordering(pi_edge, ordering):
    return tuple(pi_edge[o] for o in ordering)


def enumerate_S4_orbits():
    S4_edge = build_S4_action()
    orderings = list(permutations(range(N_E)))
    seen = set()
    orbits = []
    for o in orderings:
        if o in seen:
            continue
        orb = set()
        for pe in S4_edge:
            orb.add(act_on_ordering(pe, o))
        seen |= orb
        orbits.append(sorted(orb))
    return orbits


def shares_vertex(e1, e2):
    return len(set(e1) & set(e2)) >= 1


def is_lg_hamiltonian(ordering):
    for i in range(N_E - 1):
        if not shares_vertex(EDGES[ordering[i]], EDGES[ordering[i + 1]]):
            return False
    return True


def classify_orbits_by_lg(orbits):
    result = []
    for i, orb in enumerate(orbits):
        n_lg = sum(1 for o in orb if is_lg_hamiltonian(o))
        assert n_lg == 0 or n_lg == len(orb)
        result.append({
            'index': i, 'size': len(orb),
            'lg_ham_count': n_lg,
            'is_lg_ham_orbit': n_lg > 0,
            'representative': orb[0],
        })
    return result


def mdl_cost_bits(orbit_info, n_orbits, n_lg_orbits):
    model_bits_all = math.log2(n_orbits)
    model_bits_lg = math.log2(n_lg_orbits) if orbit_info['is_lg_ham_orbit'] else float('inf')
    data_bits = math.log2(orbit_info['size'])
    return {
        'total_bits_all': model_bits_all + data_bits,
        'total_bits_lg_restricted': model_bits_lg + data_bits,
        **orbit_info,
    }


def count_eulerian_trails_K4():
    total = 0
    for start in range(N_V):
        cnt = 0

        def rec(v, used):
            nonlocal cnt
            if len(used) == N_E:
                cnt += 1
                return
            for i, e in enumerate(EDGES):
                if i in used or v not in e:
                    continue
                nxt = e[0] if e[1] == v else e[1]
                used.add(i)
                rec(nxt, used)
                used.remove(i)

        rec(start, set())
        total += cnt
    return total


# --- PURE FUNCTION -------------------------------------------
def verify_theorem_B1_ordering():
    """Verify Theorem B1.b: no MDL-canonical ordering of K_4 edges exists.

    Returns True iff all four proof steps pass:
      1. S_4 partitions 720 orderings into 30 free orbits of size 24.
      2. Exactly 240 LG-Hamiltonian orderings (10 orbits of size 24).
      3. All orbits have identical two-part MDL cost (no unique minimum).
      4. K_4 has zero Eulerian trails (NB-walker cannot canonicalise).
    """
    # Step 1: orbit structure
    orbits = enumerate_S4_orbits()
    orbit_sizes = Counter(len(orb) for orb in orbits)
    n_orbits = len(orbits)
    if n_orbits != 30:
        return False
    if not all(s == 24 for s in orbit_sizes.elements()):
        return False

    # Step 2: LG-Hamiltonian classification
    orbit_class = classify_orbits_by_lg(orbits)
    lg_orbits = [oi for oi in orbit_class if oi['is_lg_ham_orbit']]
    total_lg = sum(oi['lg_ham_count'] for oi in orbit_class)
    if total_lg != 240 or len(lg_orbits) != 10:
        return False

    # Step 3: MDL costs are identical
    n_lg = len(lg_orbits)
    ranked = [mdl_cost_bits(oi, n_orbits, n_lg) for oi in orbit_class]
    all_totals = set(round(oi['total_bits_all'], 10) for oi in ranked)
    lg_totals = set(
        round(oi['total_bits_lg_restricted'], 10)
        for oi in ranked if oi['is_lg_ham_orbit']
    )
    if len(all_totals) != 1 or len(lg_totals) != 1:
        return False

    # Step 4: no Eulerian trails on K_4
    total_euler = count_eulerian_trails_K4()
    if total_euler != 0:
        return False

    return True


# --- VALIDATION ----------------------------------------------
if __name__ == "__main__":
    result = verify_theorem_B1_ordering()
    print(f"Result: {result}")
    assert result, "Theorem B1 verification failed"
    print(
        "Theorem B1.b verified: no MDL-canonical ordering of K_4 edges exists.\n"
        "  - 30 free S_4-orbits of size 24 (all MDL-equivalent)\n"
        "  - 240 LG-Hamiltonian orderings in 10 orbits (all MDL-equivalent)\n"
        "  - 0 Eulerian trails on K_4 (Euler obstruction)\n"
        "  => Clifford algebra must be defined invariantly (S_6-equivariant)."
    )
    print("OK")
