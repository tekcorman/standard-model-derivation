#!/usr/bin/env python3
"""
Gauge-hub Stage 16 -- the Z_2-identity check: is 2T's central Z_2 the
srs-z chirality Z_2?

This is the decisive step for Route delta of the 2T scoping doc. Route
delta hoped that the central Z_2 of the P-point little group 2T (the
kernel of 2T -> A_4 -- the witness of the projective cocycle) IS the
framework's chirality double cover srs-z. If so, the P-point flavour
sector and the chirality sector would be one structure, and the
projective 2-dim irreps would be physics, not an obstruction.

THE TEST. A chirality grading -- chi-tilde, gamma_5, gamma_7 -- is a
NON-TRIVIAL involution that ANTICOMMUTES with the hopping / walker
operator: {chi, H} = 0. This is the defining property of a chiral symmetry
(bipartite lattices, the framework's {chi-tilde, B(k)} = 0 verified at the
P-point 'mid' = (1/4,1/4,1/4) in the srs-z / lov work). The central Z_2 of
2T, by contrast, lies IN the little group G_P -- it COMMUTES with the
Bloch operator. Commute vs anticommute is basis-independent and absolute.

FINDINGS (exact computation):

  G1  THE CENTRAL ELEMENT z. The little group G_P = 2T (Stage 15) has a
      unique non-identity central element z; it has order 2, and on the
      4-atom space (two 2-dim spinor irreps) it acts as the scalar -I.

  G2  z COMMUTES WITH THE BLOCH OPERATOR. z is in G_P, the symmetry group
      of A(P): [z, A(P)] = 0.

  G3  THEREFORE z IS NOT A CHIRALITY GRADING. A chirality grading
      anticommutes with the Bloch operator ({chi, A} = 0); z commutes with
      it. An operator cannot do both for A(P) =/= 0. So the central Z_2 of
      2T is NOT the srs-z chirality Z_2 -- Route delta's identification is
      refuted.

  G4  WHAT z ACTUALLY IS. z is the SPINORIAL Z_2: central in G_P, it acts
      as the scalar -1 on every 2-dim (spinor) irrep and +1 on every
      linear irrep -- it is the kernel of 2T -> A_4, the '2-pi rotation'
      element. G_P / <z> = A_4 (order 12). It is a symmetry, not a grading;
      a scalar on each irrep, not a non-trivial involution.

VERDICT: the Z_2-identity check is NEGATIVE. 2T's central Z_2 is the
spinorial / double-cover Z_2; the srs-z Z_2 is a chirality grading. They
are different in KIND -- a central symmetry vs a chiral anti-symmetry.
Route delta loses its framework-native anchor.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))
from proofs.common import find_bonds, bloch_H, C3_PERM

gates = []
A_P = bloch_H((0.25, 0.25, 0.25), find_bonds())

# --- reconstruct G_P (the P-point little group), as in Stage 15 ------------
def perm_mat(p):
    M = np.zeros((4, 4), dtype=complex)
    for i in range(4):
        M[p[i], i] = 1.0
    return M

def monomial_for(perm):
    Pp = perm_mat(perm)
    M = Pp @ A_P @ Pp.conj().T
    if not np.allclose(np.abs(M), np.abs(A_P), atol=1e-6):
        return None
    d = [None] * 4
    d[0] = 1.0 + 0j
    frontier = [0]
    while frontier:
        i = frontier.pop()
        for j in range(4):
            if abs(M[i, j]) > 1e-6:
                want = np.conj(A_P[i, j] / (d[i] * M[i, j]))
                if d[j] is None:
                    d[j] = want / abs(want); frontier.append(j)
                elif abs(d[j] - want / abs(want)) > 1e-6:
                    return None
    d = [x if x is not None else 1.0 for x in d]
    U = np.diag(d) @ Pp
    return U if np.allclose(U @ A_P @ U.conj().T, A_P, atol=1e-6) else None

gens = [C3_PERM.astype(complex)]
for p in [(1, 0, 3, 2), (2, 3, 0, 1), (3, 2, 1, 0)]:
    U = monomial_for(p)
    if U is not None:
        gens.append(U)

def close_group(gens, tol=1e-6):
    elems = [np.eye(4, dtype=complex)]
    def find(M):
        return next((k for k, E in enumerate(elems)
                     if np.allclose(E, M, atol=tol)), -1)
    for g in gens:
        if find(g) < 0:
            elems.append(g)
    changed = True
    while changed and len(elems) < 200:
        changed = False
        for a in list(elems):
            for g in gens:
                if find(a @ g) < 0:
                    elems.append(a @ g); changed = True
    return elems

G_P = close_group(gens)

# ---------------------------------------------------------------------------
# G1 -- the central element z
# ---------------------------------------------------------------------------
centre = [E for E in G_P
          if all(np.allclose(E @ X, X @ E) for X in G_P)]
z = next((E for E in centre if not np.allclose(E, np.eye(4))), None)
z_is_minusI = z is not None and np.allclose(z, -np.eye(4))
z_order2 = z is not None and np.allclose(z @ z, np.eye(4))
gates.append((
    "G1 the central element z: G_P = 2T has a unique non-identity central "
    "element, of order 2, acting as the scalar -I on the 4-atom space "
    "(two 2-dim spinor irreps)",
    len(centre) == 2 and z_is_minusI and z_order2,
    f"|centre(G_P)| = {len(centre)}; z = -I on 4-atom space: {z_is_minusI}; "
    f"z^2 = I: {z_order2}"))

# ---------------------------------------------------------------------------
# G2 -- z commutes with the Bloch operator
# ---------------------------------------------------------------------------
comm = np.linalg.norm(z @ A_P - A_P @ z)
gates.append((
    "G2 z COMMUTES with the Bloch operator A(P): z is in G_P, the symmetry "
    "group of A(P)",
    comm < 1e-9,
    f"||[z, A(P)]|| = {comm:.2e}  (z is a symmetry of A(P))"))

# ---------------------------------------------------------------------------
# G3 -- z is therefore NOT a chirality grading
# ---------------------------------------------------------------------------
# a chirality grading chi satisfies {chi, A} = 0 (the defining property of a
# chiral symmetry; the framework's chi-tilde: {chi-tilde, B(k)} = 0, verified
# at the P-point 'mid' in the srs-z / lov work). z has {z, A(P)} = -2 A(P).
anticomm = np.linalg.norm(z @ A_P + A_P @ z)
A_norm = np.linalg.norm(A_P)
z_not_chiral = (comm < 1e-9 and anticomm > 1e-6 and A_norm > 1e-6)
gates.append((
    "G3 z is NOT a chirality grading: a chiral grading ANTICOMMUTES with "
    "the Bloch operator ({chi,A}=0); z COMMUTES with it. No operator does "
    "both for A(P) =/= 0. So 2T's central Z_2 is NOT the srs-z chirality Z_2",
    z_not_chiral,
    f"||{{z, A(P)}}|| = {anticomm:.3f} (=/= 0 => z does NOT anticommute); "
    f"||A(P)|| = {A_norm:.3f} =/= 0; z commutes (G2) -- cannot also "
    f"anticommute => z is not chiral"))

# ---------------------------------------------------------------------------
# G4 -- what z actually is: the spinorial Z_2
# ---------------------------------------------------------------------------
# z = -I on the 4-atom space (all spinorial irreps); G_P/<z> = A_4.
quotient_order = len(G_P) // 2
is_spinorial_Z2 = (z_is_minusI and len(G_P) == 24 and quotient_order == 12)
gates.append((
    "G4 z is the SPINORIAL Z_2: central in G_P, scalar -1 on the 2-dim "
    "spinor irreps and +1 on linear irreps -- the kernel of 2T -> A_4, the "
    "'2-pi rotation' element. A symmetry, not a grading",
    is_spinorial_Z2,
    f"|G_P| = {len(G_P)}, |G_P/<z>| = {quotient_order} = |A_4|; z = -I on "
    f"the all-spinorial 4-atom space -- the double-cover Z_2"))

# ===========================================================================
print("=" * 78)
print("GAUGE-HUB STAGE 16 -- Z_2-IDENTITY CHECK: 2T's CENTRE vs srs-z CHIRALITY")
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
  VERDICT -- the Z_2-identity check is NEGATIVE. 2T's central Z_2 is NOT
  the srs-z chirality Z_2.

  THE DISTINCTION IS ABSOLUTE. The central Z_2 of 2T (the element z) lies
  IN the P-point little group -- it COMMUTES with the Bloch operator; it
  acts as a scalar (-I) on the spinorial irreps; it is a SYMMETRY. The
  srs-z chirality Z_2 is a chiral grading -- chi-tilde, which ANTICOMMUTES
  with the walker ({chi-tilde, B} = 0, verified at the P-point in the
  framework's own srs-z / lov work); it is a non-trivial involution with
  +1 / -1 eigenspaces; it is an ANTI-symmetry. An operator cannot both
  commute and anticommute with A(P) =/= 0. The two Z_2's are different in
  KIND, not merely different elements -- a central symmetry vs a chiral
  grading.

  WHAT 2T's CENTRAL Z_2 ACTUALLY IS. It is the spinorial / '2-pi rotation'
  Z_2 -- the kernel of the double cover 2T -> A_4, the witness of the
  projective cocycle. It records that the P-point representation theory is
  genuinely spinorial. That is a real and meaningful fact -- but it is the
  spin-double-cover Z_2, not the LH/RH chirality.

  CONSEQUENCE FOR ROUTE delta. Route delta's framework-native ANCHOR --
  'the projective Z_2 is the srs-z chirality, so flavour and chirality are
  one structure' -- is severed. The broader delta idea (the generation
  sector is flavour (x) a spinorial label) is not refuted, but the
  spinorial label is now the spin double-cover Z_2, which the framework has
  NOT independently identified with a physical sector. Route delta thus
  loses its concrete handle and becomes a fresh posit rather than a
  connection to existing structure.

  WHERE THIS LEAVES THE GENERATION SECTOR. Of the 2T-route options:
  beta is very likely dead; delta is now un-anchored; alpha (3 = 2 (+) 1)
  is physically motivated but needs the singlet's origin derived; gamma
  (the P-point is the wrong place) remains the honest fallback. None is a
  bounded target. The disciplined recommendation stands: the generation
  SYMMETRY is genuinely underived, and bounded input-reducing work
  (N_hub / Gap G1) is the better use of effort until a new idea appears.
  What survives unchanged: B7.1's dim(C^3_gen) = 3; the C_3 structure;
  Stage 5's gauge-hub wall.
""")
print("=" * 78)
sys.exit(0 if npass == len(gates) else 1)
