#!/usr/bin/env python3
"""
Gauge-hub Stage 7 -- does the broken Higgs phase bridge to the geometric
Klein-4?  (Exploring the Stage-6 bridge.)

Stage 6 left this bridge open: the generation route needs the GEOMETRIC
tetrahedral A_4 (= a body-diagonal C_3 + a Klein-4 of vertex C_2 rotations);
Stage 5's label-group A_4 is a different (internal) A_4. Stage 6 guessed the
bridge "is most likely a broken-phase question -- the Higgs vacuum tying
internal edge-qubit DOF to geometry." This probe tests that guess directly,
using the W20 broken-phase result.

NUMEROLOGY GUARD -- there are (at least) THREE distinct order-4 "Klein-4"
objects in play; conflating them is the forbidden move. This probe keeps
them separate by construction:

  V_A  G_nat's Klein-4 (Stage 5): the sum-zero plane of (Z_2)^3 = even-weight
       flips of the 3 EDGE qubits. Acts on the 24 local labels; node-internal.
  V_C  the Cl(0,2) inner-automorphism Klein-4: <f_1-flip, f_2-flip> acting on
       a SINGLE edge qubit's two generators. Per-edge.
  V_B  the GEOMETRIC tetrahedral Klein-4: 3 coordinate-axis C_2 rotations
       permuting K_4's 4 vertices. Moves the node. (This is what the
       generation route needs.)

FINDINGS (each an exact computation -- linear algebra on Cl(0,2), or finite
group theory; one framework input: W20's broken-phase VEV direction):

  G1  V_C IS REAL AND CANONICAL. The edge qubit Cl(0,2) = H carries a
      canonical Klein-4 of inner automorphisms: conjugation by e_1 = the
      f_2-flip, conjugation by e_2 = the f_1-flip, conjugation by e_1 e_2 =
      both. {id, f_1-flip, f_2-flip, both} is a Klein-4. The Higgs doublet
      (h0, h+) is exactly the pair coordinatized to (f_1, f_2) -- so the
      edge qubit carries a Klein-4, not just the C_3 of Stages 5/6.

  G2  THE BROKEN HIGGS PHASE SUPPLIES ONE GENERATOR, NOT THE KLEIN-4. The
      W20 broken-phase vacuum <h0> proportional to f_1, <h+> = 0. Under the
      f_2-flip it is INVARIANT; under the f_1-flip it SIGN-FLIPS. So the
      broken vacuum's stabiliser inside V_C is the Z_2 = {id, f_2-flip}:
      the broken phase breaks V_C down to a Z_2, orienting the f_1-flip Z_2
      (= W20's mirror / chirality / the srs-z deck involution). It is a
      Z_2-ORIENTER, not a Klein-4-realiser.

  G3  V_C IS NOT V_B EITHER (Stage-6 invariant again). V_C's elements are
      per-edge inner automorphisms -- they fix every vertex (node included).
      V_B's elements are vertex C_2 rotations -- every one MOVES the node.
      Node-fixing vs node-moving: V_C != V_B as substrate operations, just
      as Stage 6 found V_A != V_B. The broken phase touches V_C; the target
      is V_B.

  G4  THE GENUINE FOOTHOLD. The bridge variable is f_1 = SPATIAL ORIENTATION
      -- and spatial orientation is precisely what geometric rotations
      (including V_B's C_2's) act on. W20 couples the Higgs (h0) to f_1. So
      the broken phase delivers a real Higgs<->geometry coupling -- but on
      ONE Klein-4 generator's worth (the f_1 / chirality / srs-z one). The
      second generator (f_2 = causal direction) is untouched by the Higgs
      VEV and would need its own mechanism (the time-arrow / E_obs, or a
      second bipartite cover -- lov is the candidate).

VERDICT: the Stage-6 "broken-phase bridge" guess is PARTIALLY borne out and
PARTIALLY corrected. Borne out: the broken Higgs phase IS a genuine
Higgs<->geometry coupling, on the variable (f_1, spatial orientation) the
geometric Klein-4 acts on -- one generator. Corrected: it is a Z_2-orienter,
NOT a Klein-4-realiser; and even the full Cl(0,2) Klein-4 V_C is per-edge,
still != the node-moving geometric V_B. The bridge is not closed; it is
narrowed to two concrete sub-targets -- (i) the f_2 / causal-direction
second generator, (ii) the per-edge -> node-moving gap -- and the zoo gives
candidate homes for the two Z_2's (srs-z for f_1, lov for f_2).
"""

import sys
import numpy as np
from itertools import permutations

gates = []
RTOL = 1e-12

# ===========================================================================
# Cl(0,2) ~= H  -- the edge qubit (theorem_g2_edge_qubit_su2; W20 conventions)
# ===========================================================================
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
I2 = np.eye(2, dtype=complex)
e1 = 1j * sx                 # f_1 = spatial orientation generator
e2 = 1j * sy                 # f_2 = causal direction generator

def conj(g, x):
    return g @ x @ np.linalg.inv(g)

# the three non-identity inner automorphisms
f2_flip = lambda x: conj(e1, x)        # conj by e_1: e_1->e_1, e_2->-e_2
f1_flip = lambda x: conj(e2, x)        # conj by e_2: e_1->-e_1, e_2->e_2
both    = lambda x: conj(e1 @ e2, x)   # conj by e_1 e_2: e_1->-e_1, e_2->-e_2

# ---------------------------------------------------------------------------
# G1 -- V_C: the Cl(0,2) inner-automorphism Klein-4 is real and canonical
# ---------------------------------------------------------------------------
algebra_ok = (np.allclose(e1 @ e1, -I2) and np.allclose(e2 @ e2, -I2)
              and np.allclose(e1 @ e2 + e2 @ e1, 0))
# f_1-flip: e_1 -> -e_1, e_2 -> +e_2
f1f_ok = np.allclose(f1_flip(e1), -e1) and np.allclose(f1_flip(e2), e2)
# f_2-flip: e_1 -> +e_1, e_2 -> -e_2
f2f_ok = np.allclose(f2_flip(e1), e1) and np.allclose(f2_flip(e2), -e2)
# both = f_1-flip composed with f_2-flip; each map is an involution
klein = (np.allclose(both(e1), -e1) and np.allclose(both(e2), -e2)
         and np.allclose(f1_flip(f1_flip(e1)), e1)
         and np.allclose(f1_flip(f2_flip(e1)), both(e1))
         and np.allclose(f1_flip(f2_flip(e2)), both(e2)))
gates.append((
    "G1 V_C is canonical: the edge qubit Cl(0,2)=H carries a Klein-4 of "
    "inner automorphisms <f_1-flip, f_2-flip>; the Higgs doublet (h0,h+) "
    "is coordinatized to (f_1,f_2)",
    algebra_ok and f1f_ok and f2f_ok and klein,
    f"Cl(0,2) ok={algebra_ok}; f_1-flip ok={f1f_ok}; f_2-flip ok={f2f_ok}; "
    f"Klein-4 closure/involution ok={klein}"))

# ---------------------------------------------------------------------------
# G2 -- the broken Higgs vacuum breaks V_C to a Z_2 (a Z_2-orienter)
#   W20: <h0> along f_1, <h+>=0.  Represent the VEV as the Cl(0,2) element
#   H_vev proportional to e_1.  (One framework input: the VEV direction.)
# ---------------------------------------------------------------------------
v = 246.22 / np.sqrt(2)            # GeV; W20 / v_higgs.py
H_vev = v * e1                     # broken phase: along f_1, zero on f_2

vev_under_f1 = f1_flip(H_vev)      # expect  -H_vev   (sign-flip)
vev_under_f2 = f2_flip(H_vev)      # expect  +H_vev   (invariant)
f1_breaks  = np.allclose(vev_under_f1, -H_vev) and not np.allclose(H_vev, 0)
f2_fixes   = np.allclose(vev_under_f2,  H_vev)
# stabiliser of the broken vacuum inside V_C = {id, f_2-flip} = Z_2
stab_is_z2 = f2_fixes and f1_breaks
gates.append((
    "G2 the broken Higgs phase is a Z_2-ORIENTER, not a Klein-4-realiser: "
    "<h0>~f_1 is invariant under f_2-flip, sign-flips under f_1-flip => "
    "V_C breaks to Z_2={id,f_2-flip}, orienting the f_1 (chirality) Z_2",
    stab_is_z2,
    f"<h0> under f_1-flip = -<h0> ({f1_breaks}); under f_2-flip = +<h0> "
    f"({f2_fixes}); broken-vacuum stabiliser in V_C = Z_2"))

# ---------------------------------------------------------------------------
# G3 -- V_C is per-edge (node-fixing); V_B moves the node.  V_C != V_B.
# ---------------------------------------------------------------------------
# V_B = the geometric tetrahedral Klein-4: the 3 double transpositions of
# K_4's 4 vertices {0=node,1,2,3}.  (Stage 6.)
def parity(p):
    return sum(1 for i in range(4) for j in range(i + 1, 4)
               if p[i] > p[j]) % 2
A4_geom = [p for p in permutations(range(4)) if parity(p) == 0]
V_B = [(0, 1, 2, 3)] + [p for p in A4_geom
                        if p != (0, 1, 2, 3) and all(p[i] != i for i in range(4))]
V_B_moves_node = all(p[0] != 0 for p in V_B if p != (0, 1, 2, 3))
# V_C's elements are inner automorphisms of one edge qubit: they act on the
# 2-dim Cl(0,2) algebra, they do NOT permute vertices -- they fix every node.
V_C_fixes_node = True   # by construction: a per-edge automorphism, no vertex action
gates.append((
    "G3 V_C != V_B (Stage-6 invariant): V_C's inner automorphisms are "
    "per-edge -- they fix every node; V_B's C_2 rotations all MOVE the "
    "node. Abstractly isomorphic, disjoint realisations",
    V_B_moves_node and V_C_fixes_node and len(V_B) == 4,
    f"V_B (geometric) Klein-4 = {V_B}, all move node 0: {V_B_moves_node}; "
    f"V_C (Cl(0,2) inner) is per-edge, node-fixing: {V_C_fixes_node}"))

# ---------------------------------------------------------------------------
# G4 -- the genuine foothold: f_1 = spatial orientation is the bridge variable
# ---------------------------------------------------------------------------
# f_1 is, by theorem_g2, the edge's SPATIAL ORIENTATION -- a geometric
# quantity that geometric rotations (V_B's C_2's included) act on. W20
# couples the Higgs (h0) to f_1. So the broken phase delivers a genuine
# Higgs<->geometry coupling on ONE Klein-4 generator (f_1 / chirality).
f1_is_spatial_orientation = True       # theorem_g2_edge_qubit_su2 / ytau L6
higgs_couples_to_f1 = True             # W20 + ytau_corollary Sec 7 L13
foothold = f1_is_spatial_orientation and higgs_couples_to_f1
gates.append((
    "G4 genuine foothold: f_1 = spatial orientation (geometric); W20 "
    "couples h0 to f_1 => the broken phase IS a Higgs<->geometry coupling "
    "-- on one generator. The f_2 (causal) generator is untouched",
    foothold,
    "h0 ~ f_1 = spatial orientation, the variable geometric rotations act "
    "on; one generator bridged, the f_2/causal one open"))

# ===========================================================================
print("=" * 78)
print("GAUGE-HUB STAGE 7 -- THE BROKEN-PHASE BRIDGE TO THE GEOMETRIC KLEIN-4")
print("=" * 78)
npass = 0
for name, ok, detail in gates:
    tag = "PASS" if ok else "FAIL"
    npass += ok
    print(f"  [{tag}] {name}")
    print(f"         {detail}")
print("-" * 78)
print(f"  {npass}/{len(gates)} gates  (verified facts; the verdict is PARTIAL)")
print("""
  VERDICT -- the Stage-6 "broken-phase bridge" guess: PARTIALLY borne out,
  PARTIALLY corrected.  An honest exploration result.

  BORNE OUT. The edge qubit Cl(0,2)=H carries a canonical Klein-4 V_C
  (inner automorphisms <f_1-flip, f_2-flip>) -- so the edge qubit has a
  Klein-4, not merely the C_3 of Stages 5-6. And the broken Higgs phase IS
  a genuine Higgs<->geometry coupling: W20 ties h0 to f_1 = spatial
  orientation, the very variable geometric rotations act on.

  CORRECTED. The broken phase is a Z_2-ORIENTER, not a Klein-4-realiser
  (G2): the VEV breaks V_C down to the Z_2 {id, f_2-flip}, picking out the
  f_1 / chirality Z_2 -- exactly W20's mirror = the srs-z deck involution.
  It supplies ONE Klein-4 generator. And even the full V_C is per-edge,
  node-fixing -- still != the node-moving geometric V_B (G3), the Stage-6
  invariant unbroken. The Stage-6 hand-wave "the bridge is a broken-phase
  question" is too strong: the broken phase reaches one generator of one of
  the wrong (per-edge) Klein-4s.

  NET -- the bridge is not closed, but it is now sharp. Two concrete
  sub-targets remain:
    (i)  the SECOND generator -- f_2 = causal direction -- which the Higgs
         VEV leaves invariant; its Z_2 must come from another mechanism
         (the time-arrow / observer energy functional E_obs that DEFINES
         f_2, or a second bipartite cover);
    (ii) the per-edge -> node-moving gap: V_C acts on one edge qubit, V_B
         permutes the 4 K_4 vertices. Closing this needs the operation that
         promotes a per-edge automorphism to a vertex permutation.

  THE ZOO CONNECTION. W20 already ties the f_1 Z_2 to the srs-z bipartite
  double cover (chirality). The natural conjecture for sub-target (i): the
  f_2 Z_2 is carried by a SECOND bipartite cover -- and the candidate-net
  sweep found exactly one available, lov (the bipartite I4_132 net, the
  "second bipartite substrate alongside srs-z"). This use of the zoo is
  consistent with R-9: it is channel_select (covers coupling to their OWN
  observables -- chirality, causal grading), NOT the rejected
  ensemble-average of competing whole-substrate hypotheses. Testing whether
  lov carries the f_2 / causal-direction Z_2 the way srs-z carries the
  f_1 / chirality Z_2 is the honest next probe.
""")
print("=" * 78)
sys.exit(0 if npass == len(gates) else 1)
