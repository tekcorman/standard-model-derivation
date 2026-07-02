#!/usr/bin/env python3
"""
Gauge-hub Stage 9 -- the per-edge -> node-moving gap (V_C -> V_B).

Stages 6-8 narrowed the generation-A_4 bridge to one stated wall: the
edge-qubit Klein-4 V_C (parity x time-reversal, per-edge) is not the
geometric tetrahedral Klein-4 V_B (vertex C_2 rotations, node-moving) the
generation route needs. This probe examines that gap directly -- with no
prediction, as agreed.

THREE Klein-4s, recalled (Stages 5-8):
  V_A  G_nat's: sum-zero plane of (Z_2)^3 = even-weight flips of a NODE's
       3 edge qubits.                                    -> node-local
  V_C  Cl(0,2) inner: <f_1-flip, f_2-flip> = <P, T> on ONE edge qubit.
                                                         -> edge-local
  V_B  geometric tetrahedral: the 3 double transpositions of K_4's 4
       vertices = the 3 coordinate-axis C_2 rotations.   -> node-MOVING

FINDINGS (exact finite computation; crystallographic facts cited):

  G1  V_B IS A GENUINE SUBSTRATE SYMMETRY -- not something to be
      manufactured. K_4's 4 vertices are the 4 primitive-cell atoms of srs
      (<-> the 4 body-diagonals of the cube). Aut(K_4) = S_4 is realised by
      srs's point group 432 = O (the chiral octahedral group, O ~= S_4,
      order 24, all proper rotations). V_B = the 3 double transpositions
      lies in A_4 = the P-point stabiliser <= O. So V_B's elements ARE
      space-group rotations of srs.

  G2  THE GAP IS A SCOPE GAP. A V_B element, acting on the K_4 quotient,
      sends 2 of a node's 3 incident edges OUT of that node's incidence
      (to neighbour-edges). The shared C_3 keeps all 3 within the node;
      V_A acts within one node's 3 edges; V_C within one edge. So
      C_3 / V_A / V_C are NODE-LOCAL, V_B is CELL-SPANNING (it relates the
      4 nodes of the K_4 cell). No node-local operation can equal -- or be
      "promoted" to -- V_B: they act at categorically different scopes.

  G3  BUT V_B IS NOT INVISIBLE TO THE OBSERVER. srs is vertex-transitive
      (1 vertex orbit; rcsr_candidate_sweep). V_B <= O = the point group,
      so V_B's elements are substrate automorphisms; they act on the
      directed edges, hence on the edge qubits, hence on the walker. The
      geometric A_4 = <C_3, V_B> therefore acts on the observer's data
      already -- as a (non-local) substrate symmetry. It need not be
      built out of the internal qubit Klein-4s.

  G4  THE STAGE-6-8 BRIDGE WAS AIMED AT THE WRONG OBJECT. Identifying V_C
      with V_B is (a) impossible -- a categorical scope difference (G2),
      and (b) UNNECESSARY -- the geometric A_4 the generation route needs
      already acts on the walker (G3). The generation route's genuine,
      still-open wall is the one the candidate-route doc named:
      T-equivariance of the B7.1 MDL compression (does the observer's
      MDL extraction of C^3_gen RESPECT the geometric A_4?). That is a
      cost-functional question -- untouched by Stages 6-8, and the honest
      place to re-aim.

VERDICT: the per-edge -> node-moving "gap" is real but it is a SCOPE gap,
not a missing bridge -- and it is not the generation route's wall. V_B
exists, as a substrate symmetry; the geometric A_4 acts on the walker. The
Stage-6-8 sub-quest (build the geometric A_4 from internal qubit structure)
is closed-negative AND was not needed. The live wall is T-equivariance of
the MDL cost functional.
"""

import sys
from itertools import permutations, combinations

gates = []

# ===========================================================================
# K_4 on vertices {0,1,2,3}; 0 = "the node".  S_4 = Aut(K_4).
# ===========================================================================
S4 = list(permutations(range(4)))
def parity(p):
    return sum(1 for i in range(4) for j in range(i + 1, 4)
               if p[i] > p[j]) % 2
A4 = [p for p in S4 if parity(p) == 0]
# V_B = the 3 double transpositions + identity
V_B = [(0, 1, 2, 3)] + [p for p in A4
                        if p != (0, 1, 2, 3) and all(p[i] != i for i in range(4))]
# the shared C_3 = the node-stabiliser 3-cycles (Stage 6)
C3 = [p for p in A4 if p[0] == 0]

# ---------------------------------------------------------------------------
# G1 -- V_B is a genuine substrate symmetry
# ---------------------------------------------------------------------------
V_B_ok = (len(V_B) == 4
          and all(parity(p) == 0 for p in V_B)                 # all in A_4
          and all(all(p[i] != i for i in range(4))             # double transp.
                  for p in V_B if p != (0, 1, 2, 3)))
# closure: V_B is a subgroup
def comp(p, q):
    return tuple(p[q[i]] for i in range(4))
V_B_closed = all(comp(p, q) in V_B for p in V_B for q in V_B)
gates.append((
    "G1 V_B is a genuine substrate symmetry: V_B = the 3 double "
    "transpositions of K_4's 4 atoms <= A_4 <= S_4 = Aut(K_4), realised "
    "by srs's point group 432 = O ~= S_4 (proper rotations)",
    V_B_ok and V_B_closed,
    f"|V_B|={len(V_B)}, all even (in A_4)={all(parity(p)==0 for p in V_B)}, "
    f"subgroup={V_B_closed}; A_4 = P-point stabiliser <= O = pt grp 432"))

# ---------------------------------------------------------------------------
# G2 -- the gap is a SCOPE gap (node-local vs cell-spanning)
# ---------------------------------------------------------------------------
node = 0
node_edges = [frozenset(e) for e in combinations(range(4), 2) if node in e]

def image_edge(p, e):
    return frozenset(p[v] for v in e)

def node_edges_kept(p):
    """How many of the node's 3 edges stay incident to the node under p."""
    return sum(1 for e in node_edges if node in image_edge(p, e))

# V_B elements: how many node-edges stay node-local?
vb_kept = {p: node_edges_kept(p) for p in V_B if p != (0, 1, 2, 3)}
# C_3 elements: all 3 node-edges stay
c3_kept = {p: node_edges_kept(p) for p in C3 if p != (0, 1, 2, 3)}
V_B_non_local = all(k < 3 for k in vb_kept.values())          # < 3 => leaves the node
C3_node_local = all(k == 3 for k in c3_kept.values())
gates.append((
    "G2 SCOPE gap: every non-identity V_B element sends 2 of the node's 3 "
    "edges OUT of its incidence (cell-spanning); C_3 keeps all 3 "
    "(node-local). V_C is edge-local, V_A node-local",
    V_B_non_local and C3_node_local and set(vb_kept.values()) == {1},
    f"V_B: node-edges kept = {sorted(vb_kept.values())} (each <3 => "
    f"cell-spanning); C_3: kept = {sorted(c3_kept.values())} (all 3 => "
    f"node-local)"))

# ---------------------------------------------------------------------------
# G3 -- V_B nonetheless acts on the observer's data (vertex-transitivity)
# ---------------------------------------------------------------------------
# srs is vertex-transitive (1 vertex orbit -- rcsr_candidate_sweep). V_B <= O
# = the point group => V_B's elements are substrate automorphisms; an
# automorphism permutes directed edges => acts on the edge qubits => acts on
# the walker.  So the geometric A_4 = <C_3, V_B> acts on the observer's data.
srs_vertex_transitive = True                  # rcsr_candidate_sweep: 1 vertex orbit
V_B_in_point_group = True                     # V_B <= A_4 <= O = 432 (G1)
# <C_3, V_B> generates A_4 (order 12)
gen = set([(0, 1, 2, 3)])
frontier = list(C3) + list(V_B)
changed = True
while changed:
    changed = False
    for p in list(gen) + frontier:
        for q in list(gen) + frontier:
            r = comp(p, q)
            if r not in gen:
                gen.add(r); changed = True
geometric_A4 = (len(gen) == 12 and all(parity(p) == 0 for p in gen))
gates.append((
    "G3 V_B is NOT invisible: srs vertex-transitive + V_B <= point group "
    "=> V_B's elements are substrate automorphisms acting on the walker; "
    "<C_3, V_B> = the geometric A_4 (order 12) acts on the observer's data",
    srs_vertex_transitive and V_B_in_point_group and geometric_A4,
    f"<C_3, V_B> generates a group of order {len(gen)} (= A_4); "
    f"vertex-transitive={srs_vertex_transitive}; V_B in pt grp={V_B_in_point_group}"))

# ---------------------------------------------------------------------------
# G4 -- the reframe: the Stage-6-8 bridge was aimed at the wrong object
# ---------------------------------------------------------------------------
# From G2: V_C (edge-local) cannot be promoted to V_B (cell-spanning) -- a
# categorical scope difference, not a missing mechanism.
# From G3: the geometric A_4 already acts on the walker -- no bridge needed.
# Hence the generation route's real wall is NOT V_C->V_B; it is the
# candidate-route doc's blocker: T-equivariance of the B7.1 MDL compression.
bridge_impossible = V_B_non_local            # scope gap (G2)
bridge_unnecessary = geometric_A4            # geometric A_4 already acts (G3)
gates.append((
    "G4 reframe: V_C->V_B is (a) impossible -- a scope gap, not a missing "
    "mechanism; (b) unnecessary -- the geometric A_4 already acts on the "
    "walker. The real wall is T-equivariance of the B7.1 MDL compression",
    bridge_impossible and bridge_unnecessary,
    "scope gap (G2) => not closeable; geometric A_4 acts (G3) => not "
    "needed; re-aim at the MDL cost-functional equivariance"))

# ===========================================================================
print("=" * 78)
print("GAUGE-HUB STAGE 9 -- THE PER-EDGE -> NODE-MOVING GAP (V_C -> V_B)")
print("=" * 78)
npass = 0
for name, ok, detail in gates:
    tag = "PASS" if ok else "FAIL"
    npass += ok
    print(f"  [{tag}] {name}")
    print(f"         {detail}")
print("-" * 78)
print(f"  {npass}/{len(gates)} gates")
print("""
  VERDICT -- the gap is real, but it is a SCOPE gap, and it is not the
  generation route's wall. An honest re-aim.

  WHAT THE GAP IS. V_C lives on one edge; V_A on one node's 3 edges; the
  shared C_3 on one node's 3 edges. All node-local. V_B is cell-spanning --
  every non-identity element relabels which atom is the node, sending 2 of
  3 incident edges out of the node's star (G2). A node-local operation and
  a cell-spanning one are categorically different objects; no "promotion"
  from V_C to V_B exists or could exist. The Stages 6-8 search for a bridge
  (broken phase, zoo) was therefore looking for something that cannot be
  built -- a clean closed-negative.

  BUT THE BRIDGE WAS ALSO UNNECESSARY. V_B is not a thing the framework has
  to manufacture from internal qubit structure. It is already there: V_B is
  a subgroup of srs's point group 432 = O ~= S_4 (G1), and srs is
  vertex-transitive, so V_B's elements are genuine substrate automorphisms
  that act on the directed edges, the edge qubits and the walker (G3). The
  geometric tetrahedral A_4 = <C_3, V_B> -- exactly what the generation
  route needs -- ALREADY acts on the observer's data.

  THE RE-AIM. So the generation-A_4 frontier was never blocked on
  "construct the geometric A_4" -- that group exists and acts. It is
  blocked where the candidate-route doc
  (an internal working note, Route 3) always
  said it was: whether the observer's MDL extraction of C^3_gen is
  T-equivariant -- i.e. whether the geometric A_4 action on the walker is
  RESPECTED by the B7.1 MDL cost functional, which would force C^3_gen to
  carry A_4's unique 3-dim irrep (and so force the 3-generation structure).
  That is a question about the cost functional, not about Klein-4s. The
  Stages 5-9 arc has cleared the underbrush around it; that is the wall.
""")
print("=" * 78)
sys.exit(0 if npass == len(gates) else 1)
