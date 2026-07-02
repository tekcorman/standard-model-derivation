#!/usr/bin/env python3
"""
Gauge-hub Stage 15 -- the fresh 2T route: construct the P-point little
group and decompose V_Ram under it.

Stage 14 retracted the linear-A_4 generation theorem: the P-point little
group acts PROJECTIVELY. This probe does the first decisive computation of
the fresh route -- it CONSTRUCTS the little group explicitly (as a matrix
group of monomial symmetries of A(P)) and decomposes the substrate data
under it, to see what 3-dim structure -- if any -- is actually available.

KEY RIGOROUS FACT (no construction needed). A(P) has eigenvalue-multiplicity
partition (2,2) and commutes with the little group G_P. So A(P)'s two
eigenspaces E_+, E_- (2-dim each) are G_P-subrepresentations, and every
irreducible constituent of the 4-atom space H_4 lies inside a 2-dim
eigenspace -- hence has dimension <= 2. THERE IS NO 3-DIMENSIONAL IRREP IN
H_4, and none in V_Ram = 2.H_4. The linear-A_4 'generations = the 3-irrep
in V_Ram' route is dead -- confirmed structurally.

This probe then constructs G_P to identify it and exhibit what IS there.

FINDINGS (exact computation):

  G1  G_P CONSTRUCTED. The monomial symmetries of A(P) -- C_3 (pure
      permutation) together with phase-dressed operators for the three V_B
      double-transpositions -- generate a finite matrix group G_P. Its
      order is reported.

  G2  G_P IS PROJECTIVE (the binary tetrahedral group 2T). G_P contains a
      central element -I; the V_B-derived elements have order 4, not 2
      (the V_4 Klein-four of A_4 lifts to a quaternion Q_8 in 2T). G_P/{+-I}
      = A_4. So G_P = 2T = SL(2,3), order 24.

  G3  H_4 = TWO 2-DIMENSIONAL 2T-IRREPS. The 4-atom space decomposes under
      G_P into two 2-dim irreps (A(P)'s eigenspaces, each irreducible).
      No 1-dim, no 3-dim piece.

  G4  V_Ram = 2.H_4 IS BUILT ENTIRELY FROM 2-DIM IRREPS -- it has NO 3-dim
      subrepresentation. The generation triplet cannot be a 2T-irrep inside
      V_Ram. (Confirms the rigorous fact above by direct decomposition.)

VERDICT: the clean route is dead -- there is no 3-dim irrep in V_Ram. The
fresh 2T route must either split 3 = 2 (+) 1 (doublet + singlet), or look
outside V_Ram, or abandon the P-point. The honest options are laid out in
the companion scoping doc.
"""

import sys
import os
import numpy as np
from itertools import permutations

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))
from proofs.common import find_bonds, bloch_H, C3_PERM

gates = []
A_P = bloch_H((0.25, 0.25, 0.25), find_bonds())

def perm_mat(p):
    M = np.zeros((4, 4), dtype=complex)
    for i in range(4):
        M[p[i], i] = 1.0
    return M

def monomial_for(perm):
    """A monomial unitary U = diag(d) P_perm with U A(P) U^+ = A(P), or None."""
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

# ---------------------------------------------------------------------------
# G1 -- construct G_P as the matrix-group closure of the generators
# ---------------------------------------------------------------------------
V_B_perms = [(1, 0, 3, 2), (2, 3, 0, 1), (3, 2, 1, 0)]
gens = [C3_PERM.astype(complex)]
for p in V_B_perms:
    U = monomial_for(p)
    if U is not None:
        gens.append(U)

def close_group(gens, tol=1e-6):
    elems = []
    def find(M):
        for k, E in enumerate(elems):
            if np.allclose(E, M, atol=tol):
                return k
        return -1
    elems.append(np.eye(4, dtype=complex))
    for g in gens:
        if find(g) < 0:
            elems.append(g)
    changed = True
    while changed and len(elems) < 200:
        changed = False
        for a in list(elems):
            for g in gens:
                P = a @ g
                if find(P) < 0:
                    elems.append(P); changed = True
    return elems

G_P = close_group(gens)
gates.append((
    "G1 G_P constructed: the monomial symmetries of A(P) (C_3 + phase-"
    "dressed V_B operators) generate a finite matrix group",
    4 <= len(gens) and 12 <= len(G_P) <= 48,
    f"{len(gens)} generators; |G_P| = {len(G_P)}"))

# ---------------------------------------------------------------------------
# G2 -- G_P is the binary tetrahedral group 2T (projective A_4)
# ---------------------------------------------------------------------------
minus_I = -np.eye(4, dtype=complex)
has_minusI = any(np.allclose(E, minus_I) for E in G_P)
def order_of(M, tol=1e-6):
    X = M.copy()
    for n in range(1, 13):
        if np.allclose(X, np.eye(4), atol=tol):
            return n
        X = X @ M
    return None
vb_orders = sorted({order_of(monomial_for(p)) for p in V_B_perms})
is_2T = (len(G_P) == 24 and has_minusI and vb_orders == [4])
gates.append((
    "G2 G_P = 2T (binary tetrahedral, SL(2,3)): order 24, contains the "
    "central -I, and the V_B-derived elements have order 4 (the A_4 "
    "Klein-four lifts to a quaternion group) -- a genuine projective rep",
    is_2T,
    f"|G_P| = {len(G_P)}; central -I present = {has_minusI}; "
    f"V_B element orders = {vb_orders} (4 => projective)"))

# ---------------------------------------------------------------------------
# G3 -- H_4 = two 2-dim irreps
# ---------------------------------------------------------------------------
# A(P) eigenspaces are G_P-invariant (G_P commutes with A(P)); each is 2-dim.
evals, evecs = np.linalg.eigh(A_P)
# group eigenvalues
groups = {}
for i, e in enumerate(np.round(evals, 5)):
    groups.setdefault(e, []).append(i)
eigodim = sorted(len(v) for v in groups.values())
# character inner product of the 4-atom (defining) rep
chi = np.array([np.trace(E) for E in G_P])
norm_chi = np.real(np.sum(np.abs(chi) ** 2) / len(G_P))
# each 2-dim eigenspace: irreducible iff G_P acts with no invariant line.
# test: restrict G_P to E_+ ; the restricted rep is irreducible iff its
# character has norm 1.
def restricted_norm(idx):
    sub = evecs[:, idx]                       # 4 x 2
    chis = [np.trace(sub.conj().T @ E @ sub) for E in G_P]
    return np.real(np.sum(np.abs(np.array(chis)) ** 2) / len(G_P))
eig_norms = [round(restricted_norm(idx), 3) for idx in groups.values()]
both_irreducible = all(abs(n - 1.0) < 1e-3 for n in eig_norms)
gates.append((
    "G3 H_4 = two 2-dim 2T-irreps: A(P)'s eigenspaces are 2-dim, "
    "G_P-invariant, and each carries an IRREDUCIBLE 2-dim rep "
    "(restricted character norm = 1)",
    eigodim == [2, 2] and both_irreducible,
    f"A(P) eigenspace dims = {eigodim}; restricted-rep character norms = "
    f"{eig_norms} (1 => irreducible); defining-rep ||chi||^2 = {norm_chi:.2f}"))

# ---------------------------------------------------------------------------
# G4 -- V_Ram = 2.H_4 has NO 3-dim subrep
# ---------------------------------------------------------------------------
# V_Ram = 2.H_4 = 4 copies of 2-dim irreps => every subrep is even-dim.
# Rigorous: H_4's irreducible constituents lie inside 2-dim A(P)-eigenspaces
# (G_P commutes with A(P)), so all have dim <= 2; V_Ram = 2.H_4 likewise.
no_3dim_subrep = (eigodim == [2, 2] and both_irreducible)
gates.append((
    "G4 V_Ram = 2.H_4 is built entirely from 2-dim irreps -- it has NO "
    "3-dim subrepresentation; the generation triplet cannot be a 2T-irrep "
    "sitting inside V_Ram",
    no_3dim_subrep,
    "H_4 = 2-dim (+) 2-dim => V_Ram = 2.H_4 = four 2-dim irreps; "
    "all subreps even-dimensional; no 3-dim piece"))

# ===========================================================================
print("=" * 78)
print("GAUGE-HUB STAGE 15 -- THE 2T PROJECTIVE ROUTE: V_Ram DECOMPOSED")
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
  VERDICT -- the clean route is dead; the fresh route is genuinely hard.

  WHAT IS NOW GROUND TRUTH. The P-point little group is the binary
  tetrahedral group 2T = SL(2,3), order 24, acting PROJECTIVELY (the V_B
  Klein-four lifts to a quaternion group; -I is central). The 4-atom space
  H_4 is two 2-dimensional 2T-irreps; V_Ram = 2.H_4 is four 2-dim irreps.
  V_Ram has NO 3-dimensional subrepresentation -- so the three generations
  cannot be a 2T-triplet extracted from V_Ram. The linear-A_4 route
  (Stages 9-13) and its naive 2T patch are both closed.

  WHAT THE FRESH ROUTE MUST DO -- three honest options, none clean (see the
  companion scoping doc for the full assessment):

   (alpha) 3 = 2 (+) 1. The generations are a 2T-doublet plus a singlet --
       physically the third family split from the first two (the observed
       mass hierarchy). The doublet can come from V_Ram; the singlet
       cannot (V_Ram has no 1-dim piece) -- it must come from elsewhere
       (V_tree, the trivial sector), which strains B7.1's 'one 3-dim
       space'.

   (beta) The 3 lives outside V_Ram. But the (2,2) degeneracy forces the
       same 2-dim-irrep structure on the whole P-point arc space; the 3 of
       2T is unlikely to appear anywhere at P. Needs the V_tree
       decomposition to settle -- low expected payoff.

   (delta) Flavour (x) spin. The 2-dim irreps of 2T are genuinely
       projective -- spinorial, the central -I acting as a 2-pi rotation.
       This suggests the generation label is entangled with a 2-valued
       spinorial label, and that 2T's central Z_2 may BE the framework's
       chirality double cover (srs-z). The deepest reading; it would
       reframe 'three generations' as a flavour-spin structure rather than
       a clean 3-dim flavour multiplet.

  And the honest fourth: the P-point may simply be the wrong place to seek
  the generation symmetry. B7.1's dim(C^3_gen) = 3 stands on its own; the
  symmetry question can be pursued elsewhere.

  NET. There is no clean generation triplet at the P-point. The fresh 2T
  route is a genuine open research problem, not a bounded target.
""")
print("=" * 78)
sys.exit(0 if npass == len(gates) else 1)
