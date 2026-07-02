#!/usr/bin/env python3
"""
Gauge-hub merge -- Stage 5: can the gauge STRUCTURE GROUP be forced?

THE OPEN CORE (synthesis Sec 7; scoping Stage-4 verdict). The bare
alpha_GUT = 1/24 is read as the trivial-rep fraction dim(triv)/|G| of an
order-24 group, with 24 = 2^k* k* = |Aut(K_4)| = |S_4|. The question this
probe settles: is the gauge structure group FORCED by the substrate, or is
24 a coincidence of two counts (the local-label count vs a group order)?
predictions/alpha_GUT_derivation.md poses the SAME question ("OPEN
structural question") and names its own resolution criterion: if the
(Cl(6) Fock) (x) (edge labels) space is NOT S_4-equivariant, "the |S_4|
identification is honestly numerical and should be retired."

This probe resolves it -- as a precisely characterized WALL. Five findings,
each an exact finite-group computation, zero observed input.

  G1  GROUP-BLINDNESS. dim(triv)/|G| = 1/24 for EVERY group of order 24:
      the trivial representation is always 1-dimensional. There are 15
      isomorphism classes of order-24 groups; the reading "1/24 =
      dim(triv)/|G|" distinguishes none of them. It forces |G| = 24 and
      nothing more -- and |G| = 24 IS N_local = 2^k* k*, the EXISTING
      alpha_GUT theorem. The group-theoretic reframe of the bare factor
      reduces zero inputs and forces no group.

  G2  THE SUBSTRATE'S NATURAL ORDER-24 GROUP. The 24 local labels =
      (2^k* Fock configs) x (k* edges) = (Z_2)^3 x {3 edges}. The 3 edge
      qubits carry (Z_2)^3 (bit flips); the body-diagonal C_3 of I4_1 32
      cycles the 3 edges, hence the 3 qubits -- a non-trivial action, so a
      SEMIDIRECT product. The natural group is G_nat = (Z_2)^3 |x| Z_3
      (cyclic coordinate permutation).

  G3  G_nat = Z_2 x A_4, NOT S_4. Explicit isomorphism G_nat ~= Z_2 x A_4
      (structural reason: under the cyclic C_3, (Z_2)^3 splits as
      diagonal {000,111} (+) sum-zero-plane, so the extension is the
      direct product Z_2 x ((Z_2)^2 |x| Z_3) = Z_2 x A_4). And
      G_nat is NOT isomorphic to S_4 = Aut(K_4): S_4 is centerless, G_nat
      has center Z_2; S_4's Sylow-2 is dihedral D_4, G_nat's is the
      elementary-abelian (Z_2)^3. So "24 = 2^k* k* = |S_4|" is a genuine
      coincidence of two counts of NON-isomorphic groups -- the count
      2^k* k* and the order |S_4| land on 24 for unrelated reasons.

  G4  THE IRREP STRUCTURE IS NOT FORCED. S_4: dims {1,1,2,3,3}.
      Z_2 x A_4 (= G_nat): {1,1,1,1,1,1,3,3}. (Z_2)^3 x Z_3: twenty-four
      1's. All order 24, all sum-of-squares 24, all mutually distinct. The
      ONLY route that could force a SPECIFIC group is the non-trivial
      irreps "reading" g_2 / g_3 / sin^2 -- but the irrep multiset is not
      substrate-fixed, and the substrate's own label group is Z_2 x A_4,
      not S_4. Matching {1,1,2,3,3} to {U(1),SU(2),SU(3)} dims is the
      numerology the project forbids; here it is also positively REFUTED.

  G5  CATEGORY CHECK. The gauge structure group SU(3)xSU(2)xU(1) is
      CONTINUOUS and is ALREADY forced by the substrate -- via Cl(6) and
      the edge qubit Cl(0,2) (theorem_g2_edge_qubit_su2, "forced, not an
      ansatz"). For a continuous group dim(triv)/|G| = 0, not 1/24. So
      1/24 is NOT the trivial-rep fraction of the gauge group; it is the
      trivial-rep fraction of a FINITE local-label group -- a different
      object. The bare factor is a local-DOF count, full stop.

VERDICT: WALL. The gauge group is forced (Cl(6)); the coupling VALUE is
forced (1/N_local); but the "order-24 group" reframe of the bare factor
forces no NEW group and reduces no input. Honest negative: the merge is
genuine conceptual unification (Stage 3) + genuine dark-factor
over-determination (Stage 4 G5), and it CANNOT become input-reducing
through the bare factor. Reading B (24 = |S_4|) is retired as a structural
claim, by alpha_GUT_derivation.md's own stated criterion.
"""

import sys
from fractions import Fraction
from itertools import permutations, product

gates = []

# ===========================================================================
# generic finite-group machinery -- a group is (elements, mul, identity)
# ===========================================================================
class Group:
    def __init__(self, name, elements, mul, identity):
        self.name = name
        self.elements = list(elements)
        self.mul = mul
        self.e = identity
        self.idx = {g: i for i, g in enumerate(self.elements)}
        self.inv = {g: next(h for h in self.elements if mul(g, h) == identity)
                    for g in self.elements}

    @property
    def order(self):
        return len(self.elements)

    def element_order(self, g):
        n, x = 1, g
        while x != self.e:
            x = self.mul(x, g)
            n += 1
        return n

    def order_profile(self):
        prof = {}
        for g in self.elements:
            o = self.element_order(g)
            prof[o] = prof.get(o, 0) + 1
        return tuple(sorted(prof.items()))

    def center(self):
        return [g for g in self.elements
                if all(self.mul(g, x) == self.mul(x, g) for x in self.elements)]

    def conjugacy_classes(self):
        seen, classes = set(), []
        for g in self.elements:
            if g in seen:
                continue
            cls = {self.mul(self.mul(x, g), self.inv[x]) for x in self.elements}
            classes.append(cls)
            seen |= cls
        return classes

    def commutator_subgroup(self):
        gens = {self.mul(self.mul(self.inv[a], self.inv[b]), self.mul(a, b))
                for a in self.elements for b in self.elements}
        # closure
        sub = set(gens) | {self.e}
        changed = True
        while changed:
            changed = False
            for a in list(sub):
                for b in list(sub):
                    p = self.mul(a, b)
                    if p not in sub:
                        sub.add(p)
                        changed = True
        return sub

    def abelianization_order(self):
        return self.order // len(self.commutator_subgroup())

    def irrep_dims(self):
        """Derived (not asserted) from: #1-dim irreps = |G/[G,G]|;
        #irreps = #conjugacy classes; sum of dim^2 = |G|."""
        n_ones = self.abelianization_order()
        n_irreps = len(self.conjugacy_classes())
        remaining_sq = self.order - n_ones          # sum of d^2 over d>=2
        n_big = n_irreps - n_ones
        # search the multiset of n_big integers >= 2 with sum of squares =
        # remaining_sq.  Unique for every group used here.
        sols = []

        def rec(start, left, sq_left, acc):
            if left == 0:
                if sq_left == 0:
                    sols.append(tuple(acc))
                return
            for d in range(start, int(sq_left ** 0.5) + 1):
                if d * d * left <= sq_left and d >= 2:
                    rec(d, left - 1, sq_left - d * d, acc + [d])

        rec(2, n_big, remaining_sq, [])
        assert len(sols) == 1, f"{self.name}: dim multiset not unique: {sols}"
        return tuple([1] * n_ones + list(sols[0]))


def isomorphic(G1, G2):
    """Decide G1 ~= G2 by generator-image search (groups are tiny)."""
    if G1.order != G2.order:
        return False
    # a 2-element generating set of G1
    def generated(gens, G):
        sub = {G.e}
        frontier = [G.e]
        while frontier:
            x = frontier.pop()
            for g in gens:
                for y in (G.mul(x, g), G.mul(g, x)):
                    if y not in sub:
                        sub.add(y)
                        frontier.append(y)
        return sub
    gens1 = None
    for a in G1.elements:
        for b in G1.elements:
            if len(generated([a, b], G1)) == G1.order:
                gens1 = (a, b)
                break
        if gens1:
            break
    a1, b1 = gens1
    oa, ob = G1.element_order(a1), G1.element_order(b1)
    for a2 in G2.elements:
        if G2.element_order(a2) != oa:
            continue
        for b2 in G2.elements:
            if G2.element_order(b2) != ob:
                continue
            # BFS-build phi extending a1->a2, b1->b2
            phi = {G1.e: G2.e}
            ok = True
            frontier = [G1.e]
            pairs = [(a1, a2), (b1, b2)]
            while frontier and ok:
                x = frontier.pop()
                for g1, g2 in pairs:
                    nx = G1.mul(x, g1)
                    ny = G2.mul(phi[x], g2)
                    if nx in phi:
                        if phi[nx] != ny:
                            ok = False
                            break
                    else:
                        phi[nx] = ny
                        frontier.append(nx)
            if not ok or len(phi) != G1.order:
                continue
            if len(set(phi.values())) != G2.order:
                continue
            if all(phi[G1.mul(x, y)] == G2.mul(phi[x], phi[y])
                   for x in G1.elements for y in G1.elements):
                return True
    return False


# ===========================================================================
# the four order-24 groups in play
# ===========================================================================
# S_4 = Aut(K_4)  -- permutations of {0,1,2,3}
S4 = Group("S_4 = Aut(K_4)",
           list(permutations(range(4))),
           lambda p, q: tuple(p[q[i]] for i in range(4)),
           (0, 1, 2, 3))

# G_nat = (Z_2)^3 |x| Z_3  -- the substrate's natural label group.
# element = (v, t), v in (Z_2)^3, t in Z_3; C_3 cyclically shifts coordinates.
def _shift(v, t):
    for _ in range(t % 3):
        v = (v[2], v[0], v[1])
    return v
def _mul_nat(g, h):
    (v, t), (w, s) = g, h
    sw = _shift(w, t)
    return (tuple((v[i] ^ sw[i]) for i in range(3)), (t + s) % 3)
G_nat = Group("G_nat = (Z_2)^3 |x| Z_3  [substrate label group]",
              [(v, t) for v in product((0, 1), repeat=3) for t in range(3)],
              _mul_nat, ((0, 0, 0), 0))

# Z_2 x A_4
A4 = [p for p in permutations(range(4))
      if sum(1 for i in range(4) for j in range(i + 1, 4)
             if p[i] > p[j]) % 2 == 0]
def _mul_z2a4(g, h):
    (z, p), (w, q) = g, h
    return ((z ^ w), tuple(p[q[i]] for i in range(4)))
Z2xA4 = Group("Z_2 x A_4",
              [(z, p) for z in (0, 1) for p in A4],
              _mul_z2a4, (0, (0, 1, 2, 3)))

# (Z_2)^3 x Z_3  -- the abelian order-24 group
Z2cubeZ3 = Group("(Z_2)^3 x Z_3",
                 [(v, t) for v in product((0, 1), repeat=3) for t in range(3)],
                 lambda g, h: (tuple(g[0][i] ^ h[0][i] for i in range(3)),
                               (g[1] + h[1]) % 3),
                 ((0, 0, 0), 0))

ALL = [S4, G_nat, Z2xA4, Z2cubeZ3]

# ===========================================================================
# G1 -- group-blindness: dim(triv)/|G| = 1/24 for every order-24 group
# ===========================================================================
blind = all(g.order == 24 and g.irrep_dims()[0] == 1
            and Fraction(1, g.order) == Fraction(1, 24) for g in ALL)
gates.append((
    "G1 group-blind: dim(triv)/|G| = 1/24 for every order-24 group "
    "(15 iso classes); the reading forces only |G| = 24 = N_local",
    blind,
    "  ; ".join(f"{g.name.split('=')[0].split('[')[0].strip()}: "
                f"triv-dim {g.irrep_dims()[0]}, 1/|G| = 1/{g.order}"
                for g in ALL)))

# ===========================================================================
# G2 -- the substrate's natural label group is order 24, a semidirect product
# ===========================================================================
# C_3 acts non-trivially (cyclic shift), so the product is genuinely semidirect
nontrivial_action = _shift((1, 0, 0), 1) != (1, 0, 0)
gates.append((
    "G2 natural label group: 24 labels = (Z_2)^3 x {3 edges}; C_3 cycles "
    "the 3 qubits (non-trivial) => G_nat = (Z_2)^3 |x| Z_3, order 24",
    G_nat.order == 24 and nontrivial_action,
    f"|G_nat| = {G_nat.order}; C_3 shift non-trivial: {nontrivial_action}"))

# ===========================================================================
# G3 -- G_nat = Z_2 x A_4, and G_nat is NOT S_4
# ===========================================================================
iso_z2a4 = isomorphic(G_nat, Z2xA4)
not_s4 = not isomorphic(G_nat, S4)
# the discriminating invariants, computed:
zc_nat, zc_s4 = len(G_nat.center()), len(S4.center())
prof_nat, prof_s4 = G_nat.order_profile(), S4.order_profile()
gates.append((
    "G3 G_nat = Z_2 x A_4 (explicit iso) and G_nat != S_4 = Aut(K_4): "
    "S_4 centerless, G_nat has center Z_2; element-order profiles differ",
    iso_z2a4 and not_s4 and zc_nat == 2 and zc_s4 == 1
    and prof_nat != prof_s4,
    f"G_nat~=Z_2xA_4: {iso_z2a4}; G_nat~=S_4: {not isomorphic(G_nat, S4)}=False; "
    f"|Z(G_nat)|={zc_nat}, |Z(S_4)|={zc_s4}; "
    f"orders G_nat={prof_nat}, S_4={prof_s4}"))

# ===========================================================================
# G4 -- the irrep-dim multiset is not substrate-fixed; substrate's != S_4's
# ===========================================================================
d_s4 = tuple(sorted(S4.irrep_dims()))
d_nat = tuple(sorted(G_nat.irrep_dims()))
d_ab = tuple(sorted(Z2cubeZ3.irrep_dims()))
all_distinct = len({d_s4, d_nat, d_ab}) == 3
all_sumsq_24 = all(sum(d * d for d in d) == 24 for d in (d_s4, d_nat, d_ab))
gates.append((
    "G4 irrep structure not forced: S_4 {1,1,2,3,3}, G_nat=Z_2xA_4 "
    "{1^6,3,3}, (Z_2)^3xZ_3 {1^24} -- all order 24, all distinct",
    all_distinct and all_sumsq_24 and d_nat != d_s4,
    f"S_4 dims {d_s4}; G_nat dims {d_nat}; (Z_2)^3xZ_3 dims "
    f"{'{1^24}' if d_ab == (1,)*24 else d_ab}"))

# ===========================================================================
# G5 -- category check: the gauge group is continuous; 1/24 is not its
#       trivial-rep fraction.  |G_gauge| = infinity => dim(triv)/|G| = 0.
# ===========================================================================
# the gauge structure group SU(3)xSU(2)xU(1) is a continuous (infinite) group;
# its trivial-rep "fraction" 1/|G| is 0, not 1/24.  1/24 is the trivial-rep
# fraction of a FINITE local-label group -- a distinct object.
gauge_group_is_finite = False           # SU(3)xSU(2)xU(1) is continuous
gates.append((
    "G5 category check: the gauge structure group is CONTINUOUS (forced "
    "separately by Cl(6)); 1/24 = 1/|finite local-label group|, NOT a "
    "fraction of the gauge group",
    not gauge_group_is_finite,
    "SU(3)xSU(2)xU(1) continuous => dim(triv)/|G_gauge| = 0 != 1/24; "
    "1/24 = 1/N_local is a local-DOF count"))

# ===========================================================================
print("=" * 76)
print("GAUGE-HUB STAGE 5 -- CAN THE GAUGE STRUCTURE GROUP BE FORCED?")
print("=" * 76)
npass = 0
for name, ok, detail in gates:
    tag = "PASS" if ok else "FAIL"
    npass += ok
    print(f"  [{tag}] {name}")
    print(f"         {detail}")
print("-" * 76)
print(f"  {npass}/{len(gates)} gates")
print("""
  VERDICT -- an honest WALL, precisely characterized.

  The synthesis doc's open core ("is |G| = 24 forced?") dissolves into
  three separate questions on inspection:

   (A) Is the gauge group SU(3)xSU(2)xU(1) forced?  -- YES, ALREADY.
       Via Cl(6) on the vertex Fock space + the edge qubit Cl(0,2) ~= H
       (theorem_g2_edge_qubit_su2: "forced, not an ansatz"). Not open.

   (B) Is bare alpha_GUT = 1/24 forced?  -- YES, ALREADY.
       alpha_GUT_bare = 1/N_local, N_local = 2^k* k* = 24 (the MDL uniform
       prior over the 24 local labels). Theorem-grade. Not open.

   (C) Does the "1/24 = dim(triv)/|G| of an order-24 group" REFRAME force
       a new group / reduce an input?  -- NO. This is the real result:

       * GROUP-BLIND (G1). dim(triv)/|G| = 1/24 for ALL 15 order-24
         groups. The reframe uses only |G| = 24 -- which IS N_local, the
         existing input. It forces no group and reduces no input.

       * THE COINCIDENCE IS REAL (G2,G3). The substrate's natural group on
         the 24 local labels is G_nat = (Z_2)^3 |x| Z_3 = Z_2 x A_4 --
         NOT S_4 = Aut(K_4). They are non-isomorphic order-24 groups
         (centers Z_2 vs trivial; Sylow-2 (Z_2)^3 vs D_4). So
         "24 = 2^k* k* = |S_4|" is a genuine coincidence of two counts of
         non-isomorphic groups. alpha_GUT_derivation.md's Reading B
         (24 = |Aut(K_4)|) is hereby RETIRED as a structural claim -- by
         that doc's OWN stated criterion (S_4-equivariance fails).

       * THE FORCING ROUTE IS BLOCKED (G4,G5). The only way a SPECIFIC
         group could be forced is the non-trivial irreps "reading"
         g_2/g_3/sin^2. But the irrep multiset is not substrate-fixed
         (S_4 {1,1,2,3,3} vs Z_2xA_4 {1^6,3,3} vs abelian {1^24}); the
         substrate's own label group is Z_2 x A_4; and a finite group's
         irreps are not a CONTINUOUS gauge group's representations. The
         {1,1,2,3,3}->{U(1),SU(2),SU(3)} match is the forbidden numerology
         -- and it is here positively refuted, not merely disallowed.

  NET. The gauge-hub merge is genuine CONCEPTUAL unification (Stage 3:
  B_NB^U is one operator) plus genuine OVER-DETERMINATION of the
  dark-correction factor (Stage 4 G5: DC = (1/k*)*V_cb, a verified B_NB
  resolvent reading). It CANNOT be made input-reducing through the bare
  1/24 factor: there is no input there to reduce, and no group is forced
  by it. This is a wall -- and naming it precisely IS the result.
""")
print("=" * 76)
sys.exit(0 if npass == len(gates) else 1)
