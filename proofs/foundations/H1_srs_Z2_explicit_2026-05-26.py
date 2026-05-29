#!/usr/bin/env python3
"""
H_1(srs; Z_2) explicit calculation — Target 1 of intra-srs-z bosonic hypothesis.

GROUND-THEORY SESSION: compute H_1 of the srs primitive cell with Z_2
voltage assignment, identify the trivial-Z_2 class (deck-symmetric walks)
vs non-trivial-Z_2 class (deck-antisymmetric walks), and check whether the
framework's M_persistence girth-10 walks fall into the candidate "bosonic"
or "fermionic" class.

INPUTS:
  - srs primitive cell graph: 4 atoms, 6 undirected bonds (= 12 directed
    arcs), 3-regular. From proofs/common.py find_bonds().
  - Bipartite double cover: voltage 1 on every edge → Z_2 Galois cover.
  - Voltage homomorphism: closed walk → length mod 2.

CALCULATION:
  1. K_4 = 4 vertices, 6 edges, β_1 = 6 - 4 + 1 = 3.
     H_1(K_4; Z_2) = (Z_2)^3 = 8 cycle classes.
  2. Pick spanning tree {0-1, 0-2, 0-3}. Fundamental cycles:
     C_1 = triangle (0,1,2) [from edge 1-2]
     C_2 = triangle (0,1,3) [from edge 1-3]
     C_3 = triangle (0,2,3) [from edge 2-3]
     Each length 3 (odd).
  3. Z_2 voltage map sends each basis cycle to 1 (odd length).
     Map: (a,b,c) → (a+b+c) mod 2.
  4. Kernel = {(a,b,c) : a+b+c = 0 mod 2}: 4 elements (Z_2-trivial classes).
  5. Image = {0,1}: surjective. Non-trivial classes also size 4.

USER HYPOTHESIS TEST:
  - Z_2-trivial classes (kernel): even-length walks. Candidate "bosonic" walks.
  - Z_2-non-trivial classes (image-1): odd-length walks. Walks used by M_persistence.

CRITICAL CHECK:
  Does M_persistence's girth-10 cycle (full srs lattice, NOT primitive K_4)
  fall into Z_2-trivial (length 10, even) or Z_2-non-trivial?

  Length 10 is EVEN → Z_2-TRIVIAL → deck-symmetric class.

  But M_persistence uses these for FERMION mass. If deck-symmetric = bosonic
  per user's hypothesis, this is a CONTRADICTION with the framework's existing
  M_persistence theorem.
"""

# ============================================================
# K_4 EXPLICIT CYCLE STRUCTURE
# ============================================================

# Vertices
V = [0, 1, 2, 3]

# Edges (undirected, as frozensets for uniqueness)
edges = [frozenset((u, v)) for u in V for v in V if u < v]
E = edges  # 6 edges total

# Spanning tree (rooted at 0)
tree_edges = [frozenset((0, 1)), frozenset((0, 2)), frozenset((0, 3))]
non_tree_edges = [e for e in E if e not in tree_edges]
# Non-tree: {1-2, 1-3, 2-3}


def fundamental_cycle(extra_edge):
    """Return cycle as set of edges. For K_4 with star tree at 0,
    each non-tree edge {u, v} gives triangle (0, u, v)."""
    u, v = list(extra_edge)
    return frozenset({frozenset((0, u)), frozenset((u, v)), frozenset((0, v))})


def cycle_length(cycle_edges):
    """Number of edges in cycle (= length)."""
    return len(cycle_edges)


def xor_cycles(c1, c2):
    """Z_2 sum of two cycles (symmetric difference of edge sets)."""
    return frozenset(c1.symmetric_difference(c2))


# Fundamental cycles
C1 = fundamental_cycle(frozenset((1, 2)))  # triangle 0-1-2
C2 = fundamental_cycle(frozenset((1, 3)))  # triangle 0-1-3
C3 = fundamental_cycle(frozenset((2, 3)))  # triangle 0-2-3

# All Z_2 classes (8 total)
all_classes = {}
for a in (0, 1):
    for b in (0, 1):
        for c in (0, 1):
            cyc = frozenset()
            if a: cyc = xor_cycles(cyc, C1)
            if b: cyc = xor_cycles(cyc, C2)
            if c: cyc = xor_cycles(cyc, C3)
            all_classes[(a, b, c)] = cyc


# ============================================================
# Z_2 VOLTAGE HOMOMORPHISM
# ============================================================
def voltage_class(cycle):
    """Voltage = (length) mod 2."""
    return cycle_length(cycle) % 2


# Classify all 8 Z_2 classes
trivial_classes = []
nontrivial_classes = []
for coords, cyc in all_classes.items():
    v = voltage_class(cyc)
    if v == 0:
        trivial_classes.append((coords, cyc))
    else:
        nontrivial_classes.append((coords, cyc))


# ============================================================
# REPORT
# ============================================================
def cycle_str(cyc):
    """Pretty-print cycle as sequence of vertices."""
    if not cyc:
        return "∅ (length 0)"
    return f"{{{', '.join(sorted(str(set(e)) for e in cyc))}}} (length {len(cyc)})"


def report():
    print("=" * 78)
    print("  H_1(K_4; Z_2) explicit — Target 1 of intra-srs-z bosonic hypothesis")
    print("=" * 78)

    print("\n  K_4 (srs primitive cell graph):")
    print(f"    V = 4, E = 6, β_1 = E - V + 1 = 3")
    print(f"    H_1(K_4; Z_2) = (Z_2)^3 = 8 cycle classes")
    print(f"    Spanning tree: {{0-1, 0-2, 0-3}} (star from vertex 0)")
    print(f"    Non-tree edges: {{1-2, 1-3, 2-3}}")

    print("\n  Fundamental cycles (basis of H_1 over Z_2):")
    for name, c in [('C_1', C1), ('C_2', C2), ('C_3', C3)]:
        print(f"    {name} = {cycle_str(c)}")
    print("  All three basis cycles are TRIANGLES of length 3 (odd).")

    print("\n  Z_2 voltage homomorphism (length mod 2):")
    print("    φ(C_1) = φ(C_2) = φ(C_3) = 3 mod 2 = 1  → SURJECTIVE map to Z_2")
    print("    Kernel = {(a,b,c) ∈ Z_2^3 : a+b+c = 0 mod 2}")
    print("    |Kernel| = 4 ; |Image| = 2 (Z_2)")

    print(f"\n  Z_2-TRIVIAL classes (kernel, |·| = {len(trivial_classes)}):")
    for coords, cyc in trivial_classes:
        print(f"    {coords}: {cycle_str(cyc)}")

    print(f"\n  Z_2-NON-TRIVIAL classes (image-1, |·| = {len(nontrivial_classes)}):")
    for coords, cyc in nontrivial_classes:
        print(f"    {coords}: {cycle_str(cyc)}")

    print("\n  STRUCTURAL SUMMARY (K_4 primitive cell H_1):")
    print("    Z_2-trivial classes: 0 cycle + 3 four-cycles (length 4 each)")
    print("                          → deck-symmetric lifts in srs-z")
    print("    Z_2-non-trivial classes: 4 triangles (length 3 each)")
    print("                              → deck-antisymmetric lifts in srs-z")

    # ============================================================
    # CRITICAL CHECK: M_PERSISTENCE GIRTH-10 CLASSIFICATION
    # ============================================================
    print("\n" + "=" * 78)
    print("  CRITICAL CHECK — M_persistence girth-10 cycle classification")
    print("=" * 78)
    GIRTH_FULL_LATTICE = 10
    g10_voltage = GIRTH_FULL_LATTICE % 2
    print(f"\n  Full srs lattice girth = {GIRTH_FULL_LATTICE} (per predictions/g_girth_derivation.md)")
    print(f"  φ(girth-10 cycle) = {GIRTH_FULL_LATTICE} mod 2 = {g10_voltage}")
    print(f"  Z_2 class: {'TRIVIAL (deck-symmetric)' if g10_voltage == 0 else 'NON-TRIVIAL (deck-antisymmetric)'}")
    print()
    print("  M_persistence uses girth-10 cycles for FERMION MASS GENERATION.")
    print("  Girth-10 cycle has EVEN length → Z_2-TRIVIAL → DECK-SYMMETRIC.")

    # ============================================================
    # VERDICT
    # ============================================================
    print("\n" + "=" * 78)
    print("  VERDICT — does user's 'deck-symmetric = bosonic' hypothesis hold?")
    print("=" * 78)
    print("""
  USER'S HYPOTHESIS:
    Deck-symmetric walks on srs-z → bosonic content (SUSY scalar partners)
    Deck-antisymmetric walks on srs-z → fermionic content (M_persistence)

  FRAMEWORK FACT:
    M_persistence uses GIRTH-10 cycles for fermion mass.
    Girth-10 has EVEN length → DECK-SYMMETRIC under the Z_2 voltage cover.

  CONSEQUENCE:
    Under the user's hypothesis, girth-10 cycles would generate BOSONIC
    content. But the framework's M_persistence theorem (THEOREM-GRADE)
    uses these EXACT cycles to generate FERMION mass eigenvalues.

  CONTRADICTION:
    Both candidate "bosonic" walks (deck-symmetric per H_1 trivial class)
    AND the existing "fermionic" walks (M_persistence girth-10) are in
    the SAME Z_2 class (trivial). The H_1(srs; Z_2) decomposition does
    NOT separate them.

  HONEST CONCLUSION:
    The "deck-symmetric vs deck-antisymmetric" walk class distinction does
    NOT match the "boson vs fermion" distinction at the framework level.
    M_persistence's fermion-mass-generating cycles are in the SAME
    homology class as the proposed bosonic walks.

    The user's intuition was structurally sharp but the specific H_1
    decomposition doesn't carry the Bose/Fermi distinction at the cycle-
    class level. Some OTHER distinction is needed.

  WHAT THIS PROBE RULES OUT:
    The H_1(srs; Z_2) trivial-vs-non-trivial cycle class decomposition
    cannot supply the Bose/Fermi distinction needed for Path B closure.

  WHAT'S STILL POSSIBLE:
    Other structural distinctions might supply Bose/Fermi at the walker-
    state level (independent of cycle homology class). Candidates:
      (a) The walker's INTERNAL state algebra (Cl(6,0) Fock vs Cl(0,2)
          edge state) might encode Bose/Fermi independent of which cycle
          it traverses.
      (b) The CYCLE LENGTH distinct from parity might matter — e.g.,
          length-10 (girth) vs length-12 (longer cycles in same Z_2
          class) might carry different physics.
      (c) The TOPOLOGY of the cycle (homotopy class on the periodic
          torus, not just Z_2 homology) might supply additional structure.

    None of these is concrete enough to pursue without further structural
    input. The 'intra-srs-z bosonic walks' candidate via H_1 decomposition
    is closed-as-negative.

  LAYER 5 SUSY STATUS — same as before this session's reformulation:
    GENUINELY EXTERNAL to current substrate axioms. Path C (Gorard causal
    invariance) remains the only theoretical option not yet exhausted.
""")
    print("=" * 78)


if __name__ == "__main__":
    report()
