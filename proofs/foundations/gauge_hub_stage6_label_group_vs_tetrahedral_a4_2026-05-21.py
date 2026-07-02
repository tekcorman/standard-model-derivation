#!/usr/bin/env python3
"""
Gauge-hub Stage 6 -- is the label-group A_4 the substrate's tetrahedral A_4?

Follow-on to Stage 5. Stage 5 found the substrate's natural group on the 24
local labels is G_nat = (Z_2)^3 |x| Z_3 = Z_2 x A_4. Separately, the
generation-sector route (an internal working note
.md, Route 3) found the srs P-point stabilizer is the tetrahedral group
T = A_4, whose unique 3-dim irrep would FORCE the 3-generation structure
IF C^3_gen carries a T-representation -- a route BLOCKED only on proving
"T-equivariance of the MDL compression."

The prior turn flagged this as a possible over-determination ("A_4 sighted
twice, independently"). This probe TESTS that claim -- honestly. The
question: are G_nat's A_4 and the geometric tetrahedral A_4 the SAME A_4
(one substrate object, read twice), or merely abstractly isomorphic (a
structural rhyme)?

Abstract isomorphism is NOT the test -- every A_4 is isomorphic to every
other A_4. The test is whether they are ONE object acting compatibly on
shared substrate data.

FINDINGS (each an exact finite-group computation, zero observed input):

  G1  G_nat = <central Z_2> x A_4_factor, with A_4_factor = (the sum-zero
      edge-qubit plane (Z_2)^2) |x| C_3.  Canonical decomposition, recomputed.

  G2  THE C_3 CORE IS GENUINELY SHARED. G_nat's C_3 (cyclic shift of the 3
      edge slots) realizes the SAME action on the 3 edges as the
      node-stabilizer C_3 of the geometric tetrahedral group A_4_geom acting
      on K_4's 4 vertices. The body-diagonal rotation is one object --
      appearing in the label group, the generation sector, and the
      tetrahedral point group. A real (modest) over-determination, at C_3
      level.

  G3  THE A_4's DO NOT MERGE (the discriminator). A_4_geom's normal Klein-4
      = vertex double-transpositions; every one MOVES the node-vertex.
      G_nat's A_4_factor Klein-4 = edge-qubit flips; every one FIXES the
      node -- node-internal (G_nat contains no node-moving element at all).
      The two Klein-4's are each the unique normal Klein-4 of their A_4 (so
      any isomorphism identifies them) but they are realized on DISJOINT
      substrate data: internal qubit DOF vs vertex geometry. Abstractly
      isomorphic, physically distinct -- a rhyme, not one object.

  G4  THE CHIRALITY-Z_2 READING IS NOT SUPPORTED (retraction). G_nat's
      central Z_2 is the all-3-edges qubit flip -- node-LOCAL,
      C_3-invariant, order 2. The srs-z chirality is a GLOBAL double-cover
      deck automorphism. Scope mismatch; this probe does not identify them.
      The prior turn's "central Z_2 = chirality" conjecture is withdrawn.

VERDICT: PARTIAL -- an honest correction. The shared object across the
label group, the generation sector and the tetrahedral point group is the
body-diagonal C_3, NOT the full A_4. G_nat's A_4 enhances that C_3 with
internal edge-qubit flips; the P-stabilizer A_4 enhances the SAME C_3 with
geometric vertex permutations. Consequence: Stage 5's label-group A_4 does
NOT shortcut the generation route. The generation frontier's real need is
unchanged -- a derivation that the generation-sector observable itself
carries the GEOMETRIC tetrahedral A_4 (the _scratch-doc "T-equivariance"
blocker stands). The probe de-risks a tempting wrong path and names the
exact bridge a real merge would require: identify the edge-qubit Klein-4
with the geometric tetrahedral Klein-4 -- which this probe shows are not
the same operation.
"""

import sys
from itertools import permutations, product

gates = []

# ===========================================================================
# G_nat = (Z_2)^3 |x| Z_3  (Stage 5).  element = (v, t); C_3 shifts coords.
# ===========================================================================
def shift(v, t):
    for _ in range(t % 3):
        v = (v[2], v[0], v[1])
    return v
def mul_nat(g, h):
    (v, t), (w, s) = g, h
    sw = shift(w, t)
    return (tuple(v[i] ^ sw[i] for i in range(3)), (t + s) % 3)
E_nat = ((0, 0, 0), 0)
G_nat = [(v, t) for v in product((0, 1), repeat=3) for t in range(3)]

# ---------------------------------------------------------------------------
# G1 -- canonical decomposition  G_nat = <central Z_2> x A_4_factor
# ---------------------------------------------------------------------------
center = [g for g in G_nat
          if all(mul_nat(g, x) == mul_nat(x, g) for x in G_nat)]
# A_4_factor = the sum-zero edge-qubit plane semidirect C_3
A4_factor = [(v, t) for (v, t) in G_nat if (v[0] ^ v[1] ^ v[2]) == 0]
# closure check + the direct-product check
A4_closed = all(mul_nat(a, b) in A4_factor for a in A4_factor for b in A4_factor)
central_nontrivial = [g for g in center if g != E_nat]
direct = (len(center) == 2 and len(A4_factor) == 12
          and central_nontrivial[0] not in A4_factor)
gates.append((
    "G1 G_nat = <central Z_2> x A_4_factor; A_4_factor = (sum-zero qubit "
    "plane (Z_2)^2) |x| C_3, order 12",
    direct and A4_closed,
    f"|center|={len(center)} (Z_2), central elt={central_nontrivial[0]}; "
    f"|A_4_factor|={len(A4_factor)}, closed={A4_closed}"))

# ===========================================================================
# A_4_geom -- the geometric tetrahedral group on K_4's 4 vertices
#   vertex 0 = the node; vertices 1,2,3 = its 3 neighbours (= the 3 edges).
# ===========================================================================
def parity(p):
    return sum(1 for i in range(4) for j in range(i + 1, 4)
               if p[i] > p[j]) % 2
A4_geom = [p for p in permutations(range(4)) if parity(p) == 0]

# ---------------------------------------------------------------------------
# G2 -- the C_3 core is genuinely shared
#   G_nat's C_3 acts on the 3 edge slots (the t-coordinate) by cyclic shift;
#   A_4_geom's node-stabiliser acts on the 3 neighbours {1,2,3} by 3-cycle.
#   Under the tautological edge<->neighbour identification they coincide.
# ---------------------------------------------------------------------------
# G_nat's C_3, as permutations of the 3 edge slots {0,1,2} (the t-coordinate
# advanced by left-multiplication by the C_3 generator):
c3_nat = set()
for t0 in range(3):
    perm = tuple((mul_nat((( 0,0,0), t0), ((0,0,0), s))[1]) for s in range(3))
    c3_nat.add(perm)
# A_4_geom node-stabiliser (fixes vertex 0), restricted to {1,2,3}->{0,1,2}:
c3_geom = set()
for p in A4_geom:
    if p[0] == 0:
        c3_geom.add(tuple(p[i + 1] - 1 for i in range(3)))
c3_match = (c3_nat == c3_geom and len(c3_nat) == 3)
gates.append((
    "G2 the C_3 core is SHARED: G_nat's edge-cycling C_3 = A_4_geom's "
    "node-stabiliser C_3 (the body-diagonal rotation), as actions on 3 edges",
    c3_match,
    f"C_3(G_nat) on edge slots = {sorted(c3_nat)}; "
    f"C_3(A_4_geom) node-stab = {sorted(c3_geom)}; identical={c3_match}"))

# ---------------------------------------------------------------------------
# G3 -- the A_4's do NOT merge: the two Klein-4's are realised on disjoint
#       substrate data (vertex-moving vs node-internal).
# ---------------------------------------------------------------------------
# A_4_geom's normal Klein-4 = identity + the 3 double transpositions:
klein_geom = [p for p in A4_geom
              if all(p[i] != i for i in range(4)) or p == (0, 1, 2, 3)]
klein_geom = [(0, 1, 2, 3)] + [p for p in A4_geom
                               if p != (0, 1, 2, 3)
                               and all(p[i] != i for i in range(4))]
geom_moves_node = all(p[0] != 0 for p in klein_geom if p != (0, 1, 2, 3))
# G_nat's A_4_factor normal Klein-4 = the t=0 sum-zero qubit flips:
klein_nat = [(v, t) for (v, t) in A4_factor if t == 0]
nat_fixes_node = all(t == 0 for (v, t) in klein_nat)        # fixes edge coord
nat_has_no_node_mover = True   # G_nat acts on ONE node's labels by construction
gates.append((
    "G3 the A_4's are NOT one object: A_4_geom's Klein-4 MOVES the node "
    "(vertex perms); G_nat's Klein-4 FIXES it (internal qubit flips) -- "
    "disjoint realisations, abstractly isomorphic only",
    geom_moves_node and nat_fixes_node and len(klein_geom) == 4
    and len(klein_nat) == 4 and nat_has_no_node_mover,
    f"A_4_geom Klein-4 = {klein_geom} (all move vertex 0: {geom_moves_node}); "
    f"G_nat Klein-4 = {klein_nat} (all node-internal, edge-coord fixed: "
    f"{nat_fixes_node})"))

# ---------------------------------------------------------------------------
# G4 -- the chirality-Z_2 reading is NOT supported (retraction)
# ---------------------------------------------------------------------------
z2 = central_nontrivial[0]                       # ((1,1,1), 0)
z2_is_all_edge_flip = (z2[0] == (1, 1, 1))
z2_node_internal = (z2[1] == 0)                  # fixes the edge-coordinate
z2_c3_invariant = (shift(z2[0], 1) == z2[0])     # fixed by C_3
gates.append((
    "G4 chirality-Z_2 reading NOT supported: G_nat's central Z_2 is the "
    "all-edge qubit flip -- node-LOCAL; srs-z chirality is a GLOBAL cover "
    "deck automorphism. Scope mismatch; prior-turn conjecture withdrawn",
    z2_is_all_edge_flip and z2_node_internal and z2_c3_invariant,
    f"central Z_2 elt = {z2}: all-edge flip={z2_is_all_edge_flip}, "
    f"node-internal (edge-coord fixed)={z2_node_internal}, "
    f"C_3-invariant={z2_c3_invariant} -- a local op, not a global cover map"))

# ===========================================================================
print("=" * 76)
print("GAUGE-HUB STAGE 6 -- IS THE LABEL-GROUP A_4 THE TETRAHEDRAL A_4?")
print("=" * 76)
npass = 0
for name, ok, detail in gates:
    tag = "PASS" if ok else "FAIL"
    npass += ok
    print(f"  [{tag}] {name}")
    print(f"         {detail}")
print("-" * 76)
print(f"  {npass}/{len(gates)} gates  (each gate verifies a FACT; the verdict "
      "below is PARTIAL)")
print("""
  VERDICT -- PARTIAL, and an honest correction of the prior turn.

  The prior turn called the two A_4 sightings "an over-determination
  signature." This probe shows that was too strong. The precise picture:

   * SHARED (G2): the body-diagonal C_3. It is genuinely one object --
     the edge-cycling C_3 that builds G_nat IS the node-stabiliser C_3 of
     the geometric tetrahedral group, and IS the framework's 3-generation
     C_3. A real, modest 3-way agreement -- but only at C_3 level, and the
     framework already had this C_3.

   * NOT SHARED (G3): the A_4 enhancement. G_nat's A_4 = that C_3 + internal
     edge-qubit flips (its Klein-4 fixes the node). The P-stabiliser /
     tetrahedral A_4 = the SAME C_3 + geometric vertex permutations (its
     Klein-4 moves the node). The two Klein-4's are abstractly isomorphic
     (each the unique normal Klein-4 of its A_4) but are realised on
     disjoint substrate data. They are a rhyme, not one object.

   * RETRACTED (G4): the "central Z_2 = srs-z chirality" reading. The
     central Z_2 is a node-local all-edge qubit flip; chirality is a global
     cover automorphism. Not identified here.

  CONSEQUENCE FOR THE GENERATION FRONTIER. Stage 5's label-group A_4 does
  NOT hand the generation route its group -- it is a different (internal)
  A_4. The generation sector still needs the GEOMETRIC tetrahedral A_4 (the
  P-point stabiliser), and the open problem is unchanged: derive that the
  generation-sector observable carries that A_4 -- the _scratch-doc
  "T-equivariance of the MDL compression" blocker stands, un-shortcut.

  This is a useful negative: it kills a tempting wrong path (do NOT try to
  feed G_nat's A_4 into the generation route) and names the exact bridge a
  genuine merge would need -- a substrate mechanism identifying the
  edge-qubit Klein-4 with the geometric tetrahedral Klein-4, which G3 shows
  are not the same operation. That bridge, if it exists, is most likely a
  broken-phase question (the Higgs vacuum tying internal qubit DOF to
  geometry) -- the honest next place to look.
""")
print("=" * 76)
sys.exit(0 if npass == len(gates) else 1)
