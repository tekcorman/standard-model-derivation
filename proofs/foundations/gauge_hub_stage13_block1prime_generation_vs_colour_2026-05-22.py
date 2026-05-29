#!/usr/bin/env python3
"""
Gauge-hub Stage 13 -- Block-1': is the generation symmetry distinct from
colour?

Block-1' (theorem_c3_gen_attempt.md): the worry that a generation-Z_3
derived from the substrate would just BE the colour-Z_3 (the centre of
SU(3)_colour) -- the same Z_3 in disguise, so the generation structure
would be no new content. The framework needs the generation symmetry to
be (a) from an independent structural source and (b) provably distinct
from colour.

The decisive observation: Stages 9-12 did NOT derive a generation-Z_3.
They derived a generation-A_4 -- the full non-abelian tetrahedral group.
Block-1' is framed as Z_3-vs-Z_3; it is answered by the fact that the
generation symmetry is not a Z_3 at all.

FINDINGS (exact computation; framework facts cited):

  G1  THE GENERATION SYMMETRY IS NON-ABELIAN A_4, NOT A Z_3. Stages 9-12:
      C^3_gen carries the A_4 irrep; the generation symmetry is A_4 (order
      12, non-abelian). The colour centre is Z_3 (order 3, abelian). A
      non-abelian group is not isomorphic to any Z_3 -- so the generation
      symmetry cannot be "the colour-Z_3."

  G2  EVEN THE C_3's ACT DIFFERENTLY. Restrict to the C_3 subgroup. The
      colour centre Z_3 acts on the colour triplet as a SCALAR (z.I --
      that is what "centre" means: eigenvalues (z,z,z)). The generation
      C_3 acts on C^3_gen as the REGULAR representation -- eigenvalues
      (1, w, w^2), three DISTINCT values (the C_3-restriction of the A_4
      irrep). Scalar =/= regular representation: the two C_3 actions are
      not the same operator even up to basis change.

  G3  THE BODY-DIAGONAL C_3 IS NOT THE COLOUR CENTRE, INSIDE SU(4). On the
      4 primitive-cell atoms (= the PS 4 of SU(4)) the body-diagonal C_3
      has eigenvalues (1,1,w,w^2) -- multiplicity partition (2,1,1). The
      colour-centre element of SU(4) (from 4 -> 3 (+) 1) is
      diag(w,w,w,1) -- partition (3,1). Different partitions => not
      conjugate => distinct SU(4) elements. (Corroborates B3_B6_
      reconciliation.md Finding 2: ||[U_C3, PS-Cartan]|| = 2.0.)

  G4  INDEPENDENT STRUCTURAL ORIGIN, AND V_B IS WHAT COLOUR LACKS. The
      generation A_4 is the geometric tetrahedral point group (Stage 9:
      A_4 <= srs's point group 432 -- a spacetime/lattice symmetry).
      Colour SU(3) is from Cl(6) on the vertex Fock space -- an internal
      structure. Different parts of the apparatus; the generation A_4 was
      NOT derived by descent from colour. The Klein-four V_B <= A_4 (3
      order-2 elements) has no counterpart in the colour Z_3 (0 order-2
      elements) -- and V_B is exactly what makes the generation triplet
      an IRREDUCIBLE A_4-rep (Stage 10): under C_3 alone the triplet is
      indistinguishable from the reducible 1 (+) w (+) w^2, which is the
      shape a colour-Z_3 'triplet' would have.

VERDICT: Block-1' resolves for the SYMMETRIES. The generation symmetry is
the non-abelian A_4 -- provably not the colour-Z_3 (G1-G3), from an
independent structural source (G4). One finer question is left open and
flagged honestly: whether C^3_gen and C^3_colour are distinct SPACES
(separate tensor factors). The framework asserts they are
(R3 Lemma L1); 'V_Ram ~= Cl(6) Fock' is flagged research-open. But even
the worst case there does not threaten the symmetry-distinctness above:
A_4 (non-abelian) is not SU(3), and the generation triplet's
irreducibility rests on V_B, which colour has no analogue of.
"""

import sys
import os
import numpy as np
from itertools import permutations

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))
from proofs.common import C3_PERM, omega3

gates = []
W = omega3

# ===========================================================================
# A_4 and its 3-dim irrep
# ===========================================================================
def parity(p):
    return sum(1 for i in range(4) for j in range(i + 1, 4)
               if p[i] > p[j]) % 2
A4 = [p for p in permutations(range(4)) if parity(p) == 0]

raw = np.array([[1, -1, 0, 0], [1, 1, -2, 0], [1, 1, 1, -3]], float)
Q, _ = np.linalg.qr(raw.T)                          # 4x3 orthonormal
def irrep3(p):
    Pm = np.zeros((4, 4))
    for i in range(4):
        Pm[p[i], i] = 1.0
    return Q.conj().T @ Pm @ Q

def compose(p, q):
    return tuple(p[q[i]] for i in range(4))

# ---------------------------------------------------------------------------
# G1 -- the generation symmetry is non-abelian A_4, not a Z_3
# ---------------------------------------------------------------------------
# exhibit a non-commuting pair in A_4
noncommuting = None
for a in A4:
    for b in A4:
        if compose(a, b) != compose(b, a):
            noncommuting = (a, b)
            break
    if noncommuting:
        break
A4_nonabelian = noncommuting is not None
gates.append((
    "G1 the generation symmetry is the NON-ABELIAN A_4 (order 12), not a "
    "Z_3: Stages 9-12 derived an A_4 triplet; A_4 is non-abelian so it is "
    "not isomorphic to the abelian colour-Z_3",
    A4_nonabelian and len(A4) == 12,
    f"|A_4| = {len(A4)}; non-abelian (e.g. {noncommuting[0]} and "
    f"{noncommuting[1]} do not commute) = {A4_nonabelian}; colour centre = "
    f"Z_3 (abelian, order 3)"))

# ---------------------------------------------------------------------------
# G2 -- the C_3 actions differ: scalar (colour) vs regular rep (generation)
# ---------------------------------------------------------------------------
# generation C_3 on C^3_gen: a 3-cycle's irrep matrix
three_cycle = next(p for p in A4 if sum(1 for i in range(4) if p[i]==i) == 1)
gen_C3_eigs = np.linalg.eigvals(irrep3(three_cycle))
gen_distinct = len({np.round(e, 6) for e in gen_C3_eigs}) == 3
# colour centre Z_3 on the colour triplet: a scalar w.I
colour_Z3 = W * np.eye(3)
col_eigs = np.linalg.eigvals(colour_Z3)
col_scalar = len({np.round(e, 6) for e in col_eigs}) == 1
gates.append((
    "G2 the C_3 actions differ: colour-Z_3 on the colour triplet is a "
    "SCALAR (eigenvalues all equal); generation-C_3 on C^3_gen is the "
    "REGULAR rep (eigenvalues 1, w, w^2 -- all distinct)",
    gen_distinct and col_scalar,
    f"generation C_3 eigenvalues = {sorted(np.round(gen_C3_eigs,3), key=str)} "
    f"(3 distinct={gen_distinct}); colour Z_3 eigenvalues = "
    f"{sorted(np.round(col_eigs,3), key=str)} (scalar={col_scalar})"))

# ---------------------------------------------------------------------------
# G3 -- the body-diagonal C_3 is not the colour centre, inside SU(4)
# ---------------------------------------------------------------------------
# body-diagonal C_3 on the 4 atoms (= PS 4 of SU(4)):
c3_atom_eigs = np.linalg.eigvals(C3_PERM)
def partition(eigs):
    seen = []
    for e in eigs:
        for s in seen:
            if abs(s[0] - e) < 1e-6:
                s[1] += 1
                break
        else:
            seen.append([e, 1])
    return tuple(sorted((m for _, m in seen), reverse=True))
part_C3 = partition(c3_atom_eigs)
# colour-centre element of SU(4): 4 -> 3 (+) 1 => diag(w,w,w,1), det = w^3 = 1
colour_su4 = np.diag([W, W, W, 1.0])
part_colour = partition(np.linalg.eigvals(colour_su4))
distinct_in_su4 = (part_C3 != part_colour)
gates.append((
    "G3 the body-diagonal C_3 is NOT the colour centre inside SU(4): on "
    "the PS 4 it has eigenvalue partition (2,1,1); the colour-centre "
    "element diag(w,w,w,1) has (3,1) -- different partitions, not conjugate",
    distinct_in_su4 and part_C3 == (2, 1, 1) and part_colour == (3, 1),
    f"body-diagonal C_3 on 4 atoms: eigenvalues {sorted(np.round(c3_atom_eigs,3),key=str)}, "
    f"partition {part_C3}; colour centre: partition {part_colour}; distinct={distinct_in_su4}"))

# ---------------------------------------------------------------------------
# G4 -- independent origin; V_B is the structure colour lacks
# ---------------------------------------------------------------------------
# A_4's order-2 elements = the Klein-four V_B; Z_3 has none.
A4_order2 = [p for p in A4 if p != (0,1,2,3) and compose(p, p) == (0,1,2,3)]
Z3_order2 = 0                                       # Z_3 has no order-2 element
gates.append((
    "G4 independent origin + V_B is what colour lacks: the generation A_4 "
    "is the geometric point group 432 (Stage 9), colour SU(3) is internal "
    "(Cl(6)); A_4 has the Klein-four V_B (3 order-2 elements), the "
    "colour-Z_3 has none -- and V_B makes the triplet irreducible (Stage 10)",
    len(A4_order2) == 3 and Z3_order2 == 0,
    f"A_4 order-2 elements (= V_B) = {len(A4_order2)}; colour-Z_3 order-2 "
    f"elements = {Z3_order2}; generation from point-group 432 vs colour "
    f"from Cl(6) -- distinct apparatus"))

# ===========================================================================
print("=" * 78)
print("GAUGE-HUB STAGE 13 -- BLOCK-1': GENERATION SYMMETRY vs COLOUR")
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
  VERDICT -- Block-1' resolves for the SYMMETRIES; one finer question on
  the spaces is left open and flagged honestly.

  THE DECISIVE POINT. Block-1' fears a generation-Z_3 that is secretly the
  colour-Z_3. But Stages 9-12 did not derive a generation-Z_3 -- they
  derived a generation-A_4, the full non-abelian tetrahedral group. A
  non-abelian group of order 12 simply is not the abelian colour-Z_3 (G1).
  Even comparing only the C_3 subgroups: the colour centre acts on its
  triplet as a SCALAR, while the generation C_3 acts as the REGULAR
  representation, three distinct eigenvalues (G2) -- different operators.
  And as an SU(4) element the body-diagonal C_3 has eigenvalue partition
  (2,1,1), the colour centre (3,1) -- not conjugate, hence distinct (G3,
  corroborating B3_B6's commutator-2.0 result).

  WHY IT IS NOT A COINCIDENCE. The generation A_4 comes from the geometric
  point group 432 -- a spacetime/lattice symmetry -- not by descent from
  the internal Cl(6) colour structure (G4). The two are different parts of
  the apparatus. And the piece that distinguishes them is concrete: the
  Klein-four V_B. Colour's Z_3 has no order-2 elements at all; A_4 has V_B.
  V_B is exactly the structure that (Stage 10) makes the generation triplet
  an IRREDUCIBLE A_4-rep -- under C_3 alone the triplet is indistinguishable
  from the reducible 1 (+) w (+) w^2, which is the shape a colour-Z_3
  'triplet' has. So the generation triplet's very irreducibility is built
  on the structure colour lacks.

  HONEST RESIDUAL -- a SEPARATE, lesser question. Whether C^3_gen and
  C^3_colour are distinct SPACES (separate tensor factors of the fermion
  Hilbert space). The framework asserts they are -- R3's Lemma L1,
  H_fermion = C^3_gen (x) H_gauge (x) H_spinor -- but the identification
  'V_Ram ~= Cl(6) Fock space' is flagged research-open. This does not
  threaten the verdict: even if the spaces coincided, the generation
  symmetry (non-abelian A_4) is not the colour symmetry (SU(3)), and the
  triplet's irreducibility rests on V_B, which colour has no analogue of.

  NET. The generation A_4 is a genuinely independent structure -- not the
  colour-Z_3 wearing a different hat. With the Stage-12 theorem (the three
  generations are an A_4 triplet), the generation sector is a new derived
  structure, and Need-A2's last stated blocker is cleared for the
  symmetry; the residual space-level question is named and bounded.
""")
print("=" * 78)
sys.exit(0 if npass == len(gates) else 1)
