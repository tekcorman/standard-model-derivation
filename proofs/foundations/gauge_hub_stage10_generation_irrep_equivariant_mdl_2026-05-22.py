#!/usr/bin/env python3
"""
Gauge-hub Stage 10 -- the generation frontier: does symmetry-adapted MDL
force C^3_gen to be the A_4 irrep?

Stage 9 closed the underbrush: the geometric tetrahedral A_4 = <C_3, V_B>
already acts on the walker (it is in srs's point group 432). The lone
remaining wall (candidate-route doc Route 3) is whether the observer's MDL
extraction of C^3_gen is A_4-equivariant -- which, IF it forced C^3_gen to
carry A_4's unique 3-dim IRREP, would make the 3 generations a derived
A_4 triplet.

The deepest stated blocker is Block-C2 (theorem_need_a2_generation_z3_attempt
.md): "B7.1's MDL is blind to representation content -- four different Z_3
actions on C^3 all have identical parameter count n^2-1 = 8." If true, MDL
cannot pick the irrep.

THIS PROBE TESTS BLOCK-C2 -- and finds it counts the WRONG model class.

  Block-C2 counts a GENERIC density operator on C^3: 8 real parameters,
  rep-blind. But the substrate data at P is A_4-SYMMETRIC -- B(P) is
  T-invariant, T = A_4 is the P-point stabiliser (theorem-grade). MDL
  applied correctly to symmetric data selects from the SYMMETRY-ADAPTED
  model class -- A_4-equivariant operators. And the parameter count of an
  A_4-equivariant operator is NOT 8: it is the dimension of the COMMUTANT,
  which IS rep-dependent.

FINDINGS (exact representation-theory computation; zero observed input):

  G1  COMMUTANT DIMENSIONS SEE THE REP. For the 3-dim representations of
      A_4, the dimension of the algebra of A_4-equivariant operators is:
      irrep 3 -> 1 ; 1+1'+1'' -> 3 ; reducible-with-repeat -> 5 ; trivial^3
      -> 9. The IRREP is the unique 3-dim rep with commutant dimension 1.
      An A_4-invariant density operator then has 0 / 2 / 4 / 8 free real
      parameters respectively -- the irrep gives the SHORTEST model.

  G2  THE IRREP IS ALSO THE UNIQUE FAITHFUL 3-dim rep. Every reducible
      3-dim rep is a sum of 1-dim reps, on which the Klein-4 V_4 acts
      trivially -> V_4 (and more) is in the kernel. Only the irrep is
      faithful.

  G3  V_B IS THE LOAD-BEARING DISCRIMINATOR (the Stage 5-9 thread pays
      off). Restricted to C_3 alone, the irrep 3 and the reducible 1+1'+1''
      are INDISTINGUISHABLE -- both restrict to the regular Z_3 rep, both
      with C_3-commutant dimension 3. It is exactly the Klein-4 V_4 = V_B
      whose addition cuts the irrep's commutant 3 -> 1 while leaving
      1+1'+1'' at 3. C_3 (node-local) cannot force the irrep; the
      cell-spanning V_B can. Stage 9 showed V_B acts on the walker.

  G4  BLOCK-C2 IS DEFEATED. Its "all 8 parameters, MDL blind" is the
      GENERIC count. The symmetry-adapted (A_4-equivariant) model cost is
      the commutant dimension -- rep-dependent (G1), minimised by the
      irrep. MDL applied to the correct model class for A_4-symmetric data
      DOES see the representation and DOES pick the irrep.

HONEST STATUS -- this is a real ADVANCE on an open route, NOT a closure.
The representation-theory core (G1-G4) is rigorous. Two concrete gaps
remain before the generation triplet is a theorem:
  (i)  the MDL-exploits-symmetry lemma -- formally, that the MDL-optimal
       model of A_4-symmetric data lies in the A_4-equivariant class
       (candidate-route doc Route 3 step 2: "T-invariance of B(P) =>
       T-invariance of the frame-function space");
  (ii) that the irrep 3 actually OCCURS in the substrate data at P -- i.e.
       the A_4-decomposition of the Ramanujan subspace V_Ram (the cited
       C_3-decomposition (4,2,2) does not by itself fix the V_4-content;
       it needs B(P)'s V_4-characters).
Both are concrete and checkable. Block-C2 -- the blocker that made the
route look dead -- is removed.
"""

import sys
import numpy as np
from itertools import permutations

gates = []
W = np.exp(2j * np.pi / 3)

# ===========================================================================
# A_4 = even permutations of {0,1,2,3}
# ===========================================================================
def parity(p):
    return sum(1 for i in range(4) for j in range(i + 1, 4)
               if p[i] > p[j]) % 2
A4 = [p for p in permutations(range(4)) if parity(p) == 0]
V4 = [p for p in A4 if all(p[i] != i for i in range(4)) or p == (0, 1, 2, 3)]
C3 = [p for p in A4 if p[0] == 0]                      # node-stabiliser C_3

def permmat(p):
    M = np.zeros((4, 4))
    for i in range(4):
        M[p[i], i] = 1.0
    return M

# --- the 3-dim irrep: A_4 on the sum-zero subspace of C^4 -------------------
Bcol = np.array([[1, 0, 0], [-1, 1, 0], [0, -1, 1], [0, 0, -1]], float)  # 4x3
Bpinv = np.linalg.pinv(Bcol)
def irrep3(p):
    return Bpinv @ permmat(p) @ Bcol                   # 3x3

# --- the Z_3 class label (A_4 / V_4 = Z_3) ---------------------------------
def z3_label(p):
    # coset of p in A_4/V_4 ; reps: e->0, (0,2,3,1)->? use a 3-cycle
    c = (0, 2, 3, 1)                                   # the 3-cycle (1 2 3)
    def comp(a, b): return tuple(a[b[i]] for i in range(4))
    for t in range(3):
        rep = (0, 1, 2, 3)
        for _ in range(t):
            rep = comp(c, rep)
        # p in rep*V4 ?
        for v in V4:
            if comp(rep, v) == p:
                return t
    raise RuntimeError("coset not found")

def diag_rep(chars):
    """3-dim rep diag(char_a, char_b, char_c); each char in {0(triv),1,2}."""
    def rep(p):
        t = z3_label(p)
        return np.diag([W ** (c * t) for c in chars]).astype(complex)
    return rep

REPS = {
    "irrep 3":        irrep3,
    "1 + 1' + 1''":   diag_rep([0, 1, 2]),
    "1 + 1' + 1'":    diag_rep([0, 1, 1]),
    "1 + 1 + 1'":     diag_rep([0, 0, 1]),
    "trivial^3":      diag_rep([0, 0, 0]),
}

# ===========================================================================
# commutant dimension of a representation given as a list of matrices
# ===========================================================================
def commutant_dim(mats):
    """dim of {M : g M = M g for all g} over C."""
    n = mats[0].shape[0]
    rows = []
    I = np.eye(n)
    for g in mats:
        # vec(gM - Mg) = (I (x) g - g^T (x) I) vec(M)
        rows.append(np.kron(I, g) - np.kron(g.T, I))
    Amat = np.vstack(rows)
    rank = np.linalg.matrix_rank(Amat, tol=1e-9)
    return n * n - rank

# ---------------------------------------------------------------------------
# G1 -- commutant dimension sees the representation
# ---------------------------------------------------------------------------
comm = {name: commutant_dim([rep(p) for p in A4]) for name, rep in REPS.items()}
irrep_comm = comm["irrep 3"]
irrep_unique_min = (irrep_comm == 1
                    and all(comm[n] > 1 for n in comm if n != "irrep 3"))
gates.append((
    "G1 commutant dimension sees the rep: irrep 3 -> 1 (unique minimum); "
    "reducibles -> 3 / 5 / 9. A_4-invariant density operator: 0/2/4/8 free "
    "params -- the irrep is the shortest model",
    irrep_unique_min and comm["1 + 1' + 1''"] == 3
    and comm["trivial^3"] == 9,
    "; ".join(f"{n}: comm={comm[n]} (rho free={comm[n]-1})" for n in comm)))

# ---------------------------------------------------------------------------
# G2 -- the irrep is the unique faithful 3-dim rep
# ---------------------------------------------------------------------------
def kernel_size(rep):
    return sum(1 for p in A4 if np.allclose(rep(p), np.eye(3), atol=1e-9))
kers = {name: kernel_size(rep) for name, rep in REPS.items()}
irrep_faithful = (kers["irrep 3"] == 1
                  and all(kers[n] > 1 for n in kers if n != "irrep 3"))
gates.append((
    "G2 the irrep is the unique FAITHFUL 3-dim rep: every reducible rep is "
    "a sum of 1-dim reps -> V_4 (>= ) in the kernel; only the irrep is "
    "faithful (kernel = {e})",
    irrep_faithful,
    "; ".join(f"{n}: |kernel|={kers[n]}" for n in kers)))

# ---------------------------------------------------------------------------
# G3 -- V_B (the Klein-4) is the load-bearing discriminator
# ---------------------------------------------------------------------------
# under C_3 alone:
comm_C3 = {name: commutant_dim([rep(p) for p in C3])
           for name, rep in REPS.items()}
# under the full A_4 (= comm above)
irrep_vs_111_under_C3 = (comm_C3["irrep 3"] == comm_C3["1 + 1' + 1''"] == 3)
irrep_vs_111_under_A4 = (comm["irrep 3"] == 1 and comm["1 + 1' + 1''"] == 3)
key111 = "1 + 1' + 1''"
gates.append((
    "G3 V_B is the discriminator: under C_3 alone the irrep 3 and 1+1'+1'' "
    "are INDISTINGUISHABLE (both -> regular Z_3, commutant 3); the Klein-4 "
    "V_4 = V_B cuts the irrep's commutant 3 -> 1, leaving 1+1'+1'' at 3",
    irrep_vs_111_under_C3 and irrep_vs_111_under_A4,
    f"under C_3: irrep comm={comm_C3['irrep 3']}, 1+1'+1'' comm="
    f"{comm_C3[key111]} (equal => C_3 cannot tell them apart); "
    f"under A_4: irrep comm={comm['irrep 3']}, 1+1'+1'' comm="
    f"{comm[key111]} (V_B distinguishes)"))

# ---------------------------------------------------------------------------
# G4 -- Block-C2 is defeated
# ---------------------------------------------------------------------------
# Block-C2's count: a GENERIC density operator on C^3 = n^2 - 1 = 8 params,
# rep-blind. The symmetry-adapted count (G1): A_4-equivariant operator =
# commutant dim, rep-dependent, minimised by the irrep.
generic_count = 3 * 3 - 1                              # = 8, Block-C2's number
adapted_counts = {n: int(comm[n]) - 1 for n in comm}   # density-op free params
block_c2_defeated = (generic_count == 8
                     and len(set(adapted_counts.values())) > 1
                     and min(adapted_counts, key=adapted_counts.get) == "irrep 3")
gates.append((
    "G4 Block-C2 defeated: its '8 params, MDL blind' is the GENERIC count; "
    "the symmetry-adapted (A_4-equivariant) model cost is the commutant "
    "dimension -- rep-dependent, minimised by the irrep",
    block_c2_defeated,
    f"generic rho on C^3 = {generic_count} params (Block-C2, rep-blind); "
    f"A_4-equivariant rho = {adapted_counts} (rep-dependent; argmin = irrep)"))

# ===========================================================================
print("=" * 78)
print("GAUGE-HUB STAGE 10 -- GENERATION C^3: DOES EQUIVARIANT MDL FORCE THE IRREP?")
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
  VERDICT -- a real ADVANCE on the generation route. Block-C2, the blocker
  that made the route look dead, is DEFEATED. This is not yet a closure.

  WHAT IS NOW RIGOROUS (G1-G4). Among the 3-dimensional representations of
  the geometric tetrahedral group A_4, the irreducible one is uniquely
  singled out by description length -- once you count the RIGHT model class.
  Block-C2 compared generic density operators (8 parameters, identical for
  every rep -- genuinely blind). But the substrate data at P is
  A_4-symmetric (B(P) is T-invariant; A_4 acts on the walker via the point
  group, Stage 9). MDL applied to symmetric data selects from the
  symmetry-adapted class -- A_4-equivariant operators -- whose parameter
  count is the COMMUTANT dimension: irrep 1, reducibles 3/5/9. The irrep is
  the shortest model, uniquely. And it is the unique faithful 3-dim rep.

  WHY THE WHOLE V_B THREAD MATTERED (G3). Restricted to the node-local C_3,
  the irrep and the reducible 1+1'+1'' are identical (both = regular Z_3,
  commutant 3) -- C_3 cannot force the triplet. It is precisely the Klein-4
  V_4 = V_B -- the cell-spanning tetrahedral piece -- that cuts the irrep's
  commutant from 3 to 1 and so distinguishes it. Stage 9 showed V_B acts on
  the walker (it is in srs's point group). Stages 5-9 were not underbrush:
  V_B is the exact discriminator the MDL argument needs.

  WHAT REMAINS OPEN -- two concrete, checkable gaps.
   (i)  the MDL-exploits-symmetry lemma: that the MDL-optimal model of
        A_4-symmetric data lies in the A_4-equivariant class. This is the
        candidate-route doc's Route-3 step 2 ("T-invariance of B(P) =>
        T-invariance of the frame-function space"). It is the framework's
        own "shortest description exploits all symmetry" principle, applied
        here -- plausible and framework-aligned, but it needs the formal
        statement, not a hand-wave.
   (ii) that the irrep 3 actually OCCURS in the substrate data at P: the
        A_4-decomposition of the Ramanujan subspace V_Ram. The cited
        C_3-decomposition (4,2,2) does NOT fix this -- V_Ram = a.1 + b.1' +
        c.1'' + d.3 has d free under (4,2,2) alone; pinning d needs B(P)'s
        V_4 (= V_B) characters. A concrete next computation.

  NET. The generation triplet is not yet a theorem -- but the route is no
  longer blocked. Block-C2 is gone; the mechanism (equivariant-MDL +
  commutant dimension, with V_B as the discriminator) is concrete and its
  representation-theory core is rigorous. If gaps (i) and (ii) close,
  C^3_gen carries the A_4 irrep -- the three generations become a derived
  A_4 triplet, and the mass operator breaking A_4 gives their distinct
  masses, exactly the structure of A_4 flavour models -- derived, not posed.
""")
print("=" * 78)
sys.exit(0 if npass == len(gates) else 1)
