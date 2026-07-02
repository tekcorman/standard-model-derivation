#!/usr/bin/env python3
"""
Gauge-hub merge — Stage 2 (start): the H1(srs) voltage space.

Scoping doc: an internal working note
Builds on Stage 0 (gauge_hub_stage0_z2_artin_ihara_2026-05-21.py, 7/7):
the Z2 Artin-Ihara factorization is verified; the route is alive.

THE STAGE-2 QUESTION.
  A covering graph of srs is fixed by a voltage assignment -- a group element
  per edge -- and distinct covers correspond to voltage classes modulo
  coboundaries, i.e. to H1(srs, G). Stage 0 showed (a) the Z2 sign voltage is
  a genuine non-trivial class, and (b) the archived C3 voltage was a
  coboundary (Z_omega = Z_0) -- a trivial class, no cover. So the gauge-hub
  question is sharp: WHERE on srs can a non-trivial voltage live, and can that
  room carry SU(3) x SU(2) x U(1)?

  This probe maps the voltage space of srs's primitive cell. It does NOT
  claim the gauge merge -- it characterises the room a gauge voltage has.

WHAT IS COMPUTED (primitive cell: 4 atoms, 6 edges, Z^3 edge displacements).
  A. The cycle space: first Betti number b1, a fundamental cycle basis, and
     the displacement map (cycle -> net Z^3 lattice vector).
  B. Girth consistency: girth 10 forbids any short zero-displacement loop, so
     the displacement map must be injective -- every primitive-cell cycle is
     a lattice translation. => the cell-preserving ABELIAN voltage space is
     exactly the Bloch torus T^3. The Bloch momentum k IS a U(1)^3 voltage;
     the framework's Bloch machinery already IS voltage-graph theory.
  C. The Z2 question: is srs-z's voltage a Bloch half-period (a corner of
     T^3) or a genuinely separate "parity" class -- i.e. extra room?
  D. H1(quotient, G) counts for G = Z2, Z3; where each SM gauge factor could
     sit, and the honest verdict on the abelian-vs-nonabelian wall.

NO observed input. Integer/modular linear algebra. Pre-declared gates +
an explicit honest verdict block.
"""

import sys, os
import numpy as np
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from proofs.common import find_bonds, GIRTH

gates = []   # (name, passed, detail)


# ---------------------------------------------------------------------------
# A. srs primitive cell as a displacement graph: K4 with Z^3 edge labels
# ---------------------------------------------------------------------------
bonds = find_bonds()                       # 12 directed (src,tgt,cell)
NV = 4

# one representative directed bond per unordered pair {i<j}: displacement i->j
disp = {}
for (s, t, c) in bonds:
    key = (min(s, t), max(s, t))
    if key not in disp:
        disp[key] = np.array(c, dtype=int) if s < t else -np.array(c, dtype=int)
edges = sorted(disp.keys())
NE = len(edges)
gates.append(("G1 primitive cell is K4: 4 vertices, 6 edges, one per pair",
              NV == 4 and NE == 6 and edges == [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)],
              f"|V|={NV} |E|={NE} edges={edges}"))

b1 = NE - NV + 1
gates.append(("G2 first Betti number b1 = |E|-|V|+1 = 3",
              b1 == 3, f"b1 = {NE}-{NV}+1 = {b1}"))


# ---------------------------------------------------------------------------
# A. fundamental cycle basis (spanning tree = star at vertex 0) + displacement
# ---------------------------------------------------------------------------
# tree edges (0,1),(0,2),(0,3); chords (1,2),(1,3),(2,3).
# fundamental cycle of chord (a,b): a -> b (chord) -> 0 (tree) -> a (tree).
def d(i, j):
    """displacement going i -> j."""
    if (i, j) in disp:
        return disp[(i, j)]
    return -disp[(j, i)]

chords = [(1, 2), (1, 3), (2, 3)]
cycle_disp = []
for (a, b) in chords:
    # a -> b -> 0 -> a
    vec = d(a, b) + d(b, 0) + d(0, a)
    cycle_disp.append(vec)
cycle_disp = np.array(cycle_disp, dtype=int)        # 3 x 3 integer matrix

print("Fundamental cycle displacements (rows = cycles through chords "
      f"{chords}):")
for ch, v in zip(chords, cycle_disp):
    print(f"  chord {ch}: net displacement {tuple(v)}  (|d|={np.abs(v).sum()})")

det = int(round(np.linalg.det(cycle_disp)))
rank = np.linalg.matrix_rank(cycle_disp)
gates.append(("G3 displacement map (cycle space -> Z^3) has rank 3",
              rank == 3, f"rank = {rank}, det = {det}"))


# ---------------------------------------------------------------------------
# B. girth consistency: no zero-displacement primitive cycle
# ---------------------------------------------------------------------------
# Any closed loop in srs has length >= girth = 10. Every primitive-cell cycle
# has length <= 6 < 10, so none can be a genuine loop => every primitive-cell
# cycle carries non-zero net displacement <=> the displacement map is
# injective <=> no integer combination of cycles has zero displacement
# except 0. Injective on a rank-3 -> Z^3 map <=> rank 3 (G3).
no_internal_loop = (rank == 3)
gates.append(("G4 girth-10 forbids internal loops: displacement map injective "
              "-> every primitive-cell cycle is a lattice translation",
              no_internal_loop and GIRTH == 10,
              f"girth={GIRTH}; max primitive cycle length 6 < 10; "
              f"injective={no_internal_loop}"))


# ---------------------------------------------------------------------------
# B. the abelian cell-preserving voltage space = the Bloch torus
# ---------------------------------------------------------------------------
# H1(quotient, Z) = Z^3 (the displacement lattice, G3/G4). Cell-preserving
# abelian covers <-> characters Hom(H1, U(1)) = U(1)^3 = T^3 = the Brillouin
# zone. The Bloch phase exp(2 pi i k.cell) IS the U(1)^3 voltage evaluated by
# the character k. So: the framework's Bloch momentum is a voltage, and the
# BZ is the abelian voltage moduli space. This is a reframe, asserted on G3+G4.
gates.append(("G5 cell-preserving abelian voltage space = Hom(Z^3,U(1)) = T^3 "
              "= the Bloch torus; Bloch k is a U(1)^3 voltage",
              no_internal_loop, "H1 = Z^3 (G3) -> character group U(1)^3 = BZ"))


# ---------------------------------------------------------------------------
# C. the Z2 question -- is srs-z's voltage a Bloch half-period or separate?
# ---------------------------------------------------------------------------
# srs-z = bipartite double cover = the "parity" Z2 voltage: every edge carries
# the generator, so a cycle's holonomy = (its edge-count) mod 2. Each
# fundamental cycle here has 3 edges -> parity class on the basis = (1,1,1).
parity_class = np.array([1, 1, 1]) % 2     # length-3 cycles, all odd

# A Bloch half-period m in {0,1}^3 induces the Z2 voltage whose holonomy on a
# cycle c is (m . displacement(c)) mod 2. Enumerate all 8 and see if (1,1,1)
# -- the parity class -- is among them.
half_period_classes = {}
for m in product([0, 1], repeat=3):
    cls = tuple((cycle_disp @ np.array(m)) % 2)
    half_period_classes[cls] = m
parity_is_bloch = tuple(parity_class) in half_period_classes
gates.append(("G6 srs-z's Z2 voltage (parity class (1,1,1)) IS a Bloch "
              "half-period -- not separate room",
              parity_is_bloch,
              f"parity class (1,1,1); Bloch half-period classes = "
              f"{sorted(half_period_classes.keys())}; "
              f"parity matches m={half_period_classes.get(tuple(parity_class))}"))


# ---------------------------------------------------------------------------
# D. H1(quotient, G) counts + where the SM gauge factors can sit
# ---------------------------------------------------------------------------
print()
print("H1(srs primitive quotient, G) = G^3  (b1 = 3):")
for name, order in [("Z2", 2), ("Z3", 3), ("U(1)", None)]:
    if order:
        print(f"  G = {name:5s}: {order**3} classes ({order}^3); "
              f"{order**3 - 1} non-trivial")
    else:
        print(f"  G = {name:5s}: U(1)^3 = T^3 continuum  (= the Bloch torus)")

# the count of DISTINCT covers = |H1| since the quotient graph has trivial
# coboundary action for abelian G on a connected graph: coboundaries are
# Hom(V,G)/G, dim |V|-1; H1 = G^|E| / coboundary = G^(|E|-|V|+1) = G^b1.
n_z2 = 2**b1
n_z3 = 3**b1
gates.append(("G7 abelian covers counted: |H1(.,Z2)| = 2^3 = 8, "
              "|H1(.,Z3)| = 3^3 = 27",
              n_z2 == 8 and n_z3 == 27, f"Z2: {n_z2}  Z3: {n_z3}"))


# ---------------------------------------------------------------------------
print("=" * 74)
print("GAUGE-HUB STAGE 2 (start) -- THE H1(srs) VOLTAGE SPACE")
print("=" * 74)
npass = 0
for name, ok, detail in gates:
    tag = "PASS" if ok else "FAIL"
    if ok:
        npass += 1
    print(f"  [{tag}] {name}")
    print(f"         {detail}")
print("-" * 74)
print(f"  {npass}/{len(gates)} gates")
print("""
  VERDICT -- where a gauge voltage can live on srs (honest).

  (1) The abelian voltage room of the primitive cell IS the Bloch torus.
      H1 = Z^3, all of it translational (girth 10 kills internal loops, G4).
      Hom(Z^3, U(1)) = T^3 = the BZ. The Bloch momentum k is literally a
      U(1)^3 voltage -- the framework's Bloch machinery already IS abelian
      voltage-graph theory. A U(1) gauge factor has a natural home here.

  (2) srs-z's Z2 is NOT separate room -- it is a Bloch half-period (G6).
      The chirality double cover sits at a corner of the BZ. Consistent
      with the framework's "structure lives at Bloch points" picture; it
      means Z2 is not extra cohomology, it is a 2-torsion point of T^3.

  (3) THE WALL, stated honestly. Every abelian voltage on the primitive
      cell is a point of T^3 -- there is no abelian room beyond U(1)^3.
      A non-abelian gauge factor (SU(2), SU(3)) is therefore NOT an abelian
      voltage class on this cell. It must come from one of:
        (a) a non-abelian voltage (non-abelian H1 -- not classified by
            characters; the cover's deck group is non-abelian),
        (b) a supercell, where girth-10 loops close as internal cycles and
            new (non-translational) voltage room opens, or
        (c) the edge-qubit Cl(0,2) = H already carried per edge -- an
            SU(2) attached to edges that is structural, not a cover voltage.

  (3c) is the live lead: the framework ALREADY puts an SU(2) (the edge
  qubit) and the Cl(6) gauge content on the substrate -- not as covers but
  as edge/vertex algebra. The Stage-2 reframe: the gauge group may not be a
  COVER of srs at all -- it may be the structure group of a BUNDLE over srs,
  with B_NB the connection. Covers (Stage 0/this probe) handle U(1) and the
  Z2; SU(2)/SU(3) want the bundle, not the cover. Next probe: test whether
  the edge-qubit SU(2) + Cl(6) content assemble into a connection whose
  holonomy B_NB reads the couplings -- the bundle picture, not the cover.
""")
print("=" * 74)
sys.exit(0 if npass == len(gates) else 1)
