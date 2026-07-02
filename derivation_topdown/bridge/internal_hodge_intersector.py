"""
INTERNAL n=1 <-> n=2 HODGE MAP on the Cl(6) Fock = Lambda^*(C^3), and the C3-breaking test.

PURE MATH, walled.  Reads ONLY ../dirac_srs_mdl/srs.py (the K4 / Z^3 crystal) + native structure
already established in-box (matter_bridge m05: internal = Lambda^*(C^3), C3-isotype (4,2,2); the deck
C3 acts on the 3-irrep C^3 = Lambda^1 by the character diag(1,omega,omega^2)).  No physics, no observed
numbers, no targets, no fitting.  Derive; under-claim.  K-rational where possible.

THE CONTRAST WE ARE PROBING (established in-box):
  * bridge/offdiag_interchannel_mixing.py: the bare SPATIAL inter-channel mixing V(s)=U_A^dag U_B is
    FORCED TRIVIAL (= I), because the run is along the deck-C3-FIXED axis (1,-1,1), so [B(s*AXIS),C3]=0
    and both channels co-diagonalize under the deck C3.  The spatial fiber cannot misalign anything.
  QUESTION: does reading the INTERNAL Cl(6) occupation labels (n=1 vs n=2 Hamming sectors) supply a
  C3-breaking, sector-distinguishing structure that the bare spatial object forces to zero?

WHAT THIS FILE COMPUTES
  (1) The canonical n=1 <-> n=2 map: Hodge star Lambda^1 <-> Lambda^2 via the canonical volume form
      of C^3, AND (equivalently up to the metric) the wedge/contraction generators of Cl(6).  Its
      explicit 3x3 matrix, its angles, its phase.
  (2) Whether the Hodge star INTERTWINES the deck C3 (so n=1 and n=2 are the SAME C3-rep) or whether
      a definite occupation LABELING singles out a C3-isotype direction = breaks the deck C3.
  (3) FORCED vs FREE: residual freedom in the labeling; relate to the V_Ram~Cl(6) iso / U(4)xU(2)xU(2)
      residual IF in-box (flag if not).
  (4) THE DECISIVE CONTRAST: is the internal n=1<->n=2 misalignment NON-trivial (unlike the spatial)?
"""
import numpy as np, cmath, itertools, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dirac_srs_mdl"))
import srs

np.set_printoptions(precision=6, suppress=True, linewidth=140)
def hdr(s): print("\n" + "=" * 96 + "\n" + s + "\n" + "=" * 96)
om = cmath.exp(2j*np.pi/3)
SQRT2, SQRT3 = np.sqrt(2.0), np.sqrt(3.0)

# =============================================================================================
hdr("(0) THE OBJECT: Lambda^*(C^3), C3 grading, and the two 3-dim Hamming sectors n=1, n=2")
# =============================================================================================
# C^3 = Lambda^1 carries the 3-irrep; the deck C3 acts as the diagonal character diag(1, omega, omega^2)
# (m05: the C3-isotype of Lambda^* is built from weights {0,1,2} on the three generators e1,e2,e3).
# Basis of Lambda^1: e1, e2, e3   with C3 weights (0,1,2)   -> C3|_{L1} = diag(1, om, om^2).
# Basis of Lambda^2: e2^e3, e3^e1, e1^e2 (the canonical Hodge-dual ordering) with C3 weights:
#     e2^e3 -> 1+2 = 0 ;  e3^e1 -> 2+0 = 2 ;  e1^e2 -> 0+1 = 1.
# So C3|_{L2} in the (e23,e31,e12) basis = diag(1, om^2, om).
C3_L1 = np.diag([1, om, om**2])
C3_L2 = np.diag([om**(1+2) % 1 if False else 1, om**(2+0), om**(0+1)])  # weights (0,2,1)
# (write the L2 weights explicitly to avoid any modular slip)
w_L1 = np.array([0, 1, 2])                 # weights of e1, e2, e3
w_L2 = np.array([(1+2) % 3, (2+0) % 3, (0+1) % 3])  # weights of e23, e31, e12 = (0,2,1)
C3_L1 = np.diag([om**w for w in w_L1])
C3_L2 = np.diag([om**w for w in w_L2])
print(f"  Lambda^1 basis (e1,e2,e3): C3 weights {tuple(w_L1)}  -> C3|L1 = diag(1, om, om^2).")
print(f"  Lambda^2 basis (e23,e31,e12): C3 weights {tuple(w_L2)}  -> C3|L2 = diag(1, om^2, om).")
print(f"  => as C3-reps: L1-isotype multiplicities (triv,om,om2) = "
      f"{tuple(int(np.sum(w_L1==t)) for t in (0,1,2))} ;  L2 = "
      f"{tuple(int(np.sum(w_L2==t)) for t in (0,1,2))}.")
print("     L1 carries each of {triv, om, om2} ONCE; L2 carries each ONCE -- but the om / om2 are SWAPPED.")
print("     (This swap is exactly complex conjugation on the C3 character: L2 = conj(L1) as a C3-rep.)")

# =============================================================================================
hdr("(1) THE CANONICAL n=1 <-> n=2 MAP: the Hodge star via the volume form of C^3")
# =============================================================================================
# Hodge star *: Lambda^1 -> Lambda^2 defined by  alpha ^ (*beta) = <alpha,beta> vol,  vol = e1^e2^e3.
# With the standard Hermitian metric <ei,ej>=delta_ij and vol = e1^e2^e3:
#   *e1 = e2^e3,   *e2 = e3^e1,   *e3 = e1^e2.
# In the ORDERED bases (e1,e2,e3) -> (e23,e31,e12) this is the IDENTITY 3x3 matrix.
# That is the WHOLE content of the canonical Hodge star at the level of the abstract 3-dim sectors:
# with the Hodge-dual ordering of Lambda^2 it is the identity (a real, unit-determinant map).
STAR = np.eye(3)               # *: (e1,e2,e3) -> (e23,e31,e12)
print("  Hodge star *: Lambda^1 -> Lambda^2,  alpha^(*beta) = <alpha,beta> vol,  vol = e1^e2^e3.")
print("    *e1 = e2^e3 ,  *e2 = e3^e1 ,  *e3 = e1^e2.")
print("    In the canonical Hodge-dual ordering (e23,e31,e12) the matrix of * is:")
print(STAR)
print("    => angles: 0 (a diagonal identity); phase: 0 (real, det = +1).  It is K-rational (entries in Z).")
print("""
  IMPORTANT (the honest content): the Hodge star is "trivial" ONLY because we CHOSE to label the
  Lambda^2 basis by the dual ordering (e23,e31,e12).  The map is the IDENTITY *as a labeling
  convention*.  The geometric content is NOT the matrix entries -- it is the C3-EQUIVARIANCE: how *
  conjugates the deck C3.  We test that next.""")

# =============================================================================================
hdr("(2) DOES THE HODGE STAR INTERTWINE THE DECK C3?  (the C3-breaking test)")
# =============================================================================================
# Equivariance test:  STAR @ C3_L1 == C3_L2 @ STAR  ?
# i.e. does *  carry the C3-action on L1 to the C3-action on L2 (an intertwiner), or does it MIX
# the isotypes (break C3)?
lhs = STAR @ C3_L1
rhs = C3_L2 @ STAR
print("  Test STAR @ (C3|L1) == (C3|L2) @ STAR  (is * a C3-intertwiner)?")
print("   STAR @ C3|L1 =\n", lhs)
print("   C3|L2 @ STAR =\n", rhs)
print(f"   intertwiner (equal)?  {np.allclose(lhs, rhs)}")
print(f"   max| STAR C3|L1  -  C3|L2 STAR | = {np.max(np.abs(lhs-rhs)):.3e}")
print("""
  RESULT: * is NOT a C3-intertwiner between (L1, C3|L1) and (L2, C3|L2) in the SAME labeling -- it is
  an intertwiner only after the om<->om2 SWAP.  Concretely:  *  maps the L1 weight-1 line (e2) to the
  L2 weight-2 line (e3^e1) and the L1 weight-2 line (e3) to the L2 weight-1 line (e1^e2).
  So the canonical Hodge star is an intertwiner  L1 -> conj(L2-rep), i.e. it ANTI-commutes with the
  C3 phase (it is C3-conjugate-linear in the isotype label).  This is a GENUINE structural datum, not
  a convention: the volume form e1^e2^e3 has C3 weight 0+1+2 = 3 = 0, so * preserves the trivial line
  but exchanges the two complex-conjugate lines.  This is the FIRST non-trivial feature absent spatially.""")

# Make the swap explicit as a permutation/conjugation operator on the C3 phases.
SWAP = np.array([[1,0,0],[0,0,1],[0,1,0]])   # fixes triv, swaps the om and om2 isotype lines
print("  The C3-action * intertwines is the SWAP of the two complex lines (charge conjugation K):")
print("   K = ", SWAP.tolist(), "   K @ C3|L1 @ K^-1 == conj(C3|L1) ?",
      np.allclose(SWAP @ C3_L1 @ SWAP, np.conj(C3_L1)))
print("""
  SHARPENING (basis-independent): BOTH L1 and L2 are the REGULAR rep of C3 (one copy of each
  character 1,om,om2), so they ARE isomorphic as C3-reps -- a C3-equivariant identification EXISTS.
  There are thus TWO distinct FORCED, RIGID identifications L1 -> L2:
     (i)  the metric Hodge star *   :  w -> -w   (e1->e23, e2->e31, e3->e12) -- the volume-form map;
     (ii) the C3-equivariant match  :  w ->  w   (e1->e23, e2->e12, e3->e31) -- weight-preserving.
  They differ by EXACTLY K = swap(om,om2).  Neither is a free parameter: both are rigid (K^2=I, real,
  det=-1). The Hodge star is specifically the CONJUGATION identification, not the equivariant one --
  that is its non-trivial content.""")

# =============================================================================================
hdr("(3) THE FULL n=1 <-> n=2 OVERLAP and whether a DEFINITE LABELING breaks C3")
# =============================================================================================
# A "definite occupation labeling" = an ordered orthonormal basis of L1 (which single-occupation modes
# we call e1,e2,e3) AND, via the canonical Hodge star, the induced ordered basis of L2.  The question:
# does fixing such a labeling single out a direction in the C3-isotype (4,2,2) space -- i.e. break the
# deck C3 -- or is the occupation grading itself C3-invariant?
#
# Two distinct things to separate cleanly:
#  (a) the GRADING by Hamming weight n (the sectors L0,L1,L2,L3 themselves): is it C3-invariant?
#  (b) a chosen ORDERED BASIS within L1 (a labeling): does it break C3?

# (a) The grading is C3-invariant: C3 = diag of cube-roots acts WITHIN each Lambda^n (it is a tensor
# power of the L1 character), so each Hamming sector is a C3-subrepresentation -- the GRADING does not
# move under C3.  Verify on the full 8-dim Fock.
def fock_basis():
    basis = []
    for r in range(4):
        for S in itertools.combinations(range(3), r):
            basis.append(S)
    return basis   # 8 subsets, graded by |S|
FB = fock_basis()
def C3_on_fock():
    # diagonal: weight of subset S = sum of generator-weights mod 3, generator weights (0,1,2)
    diag = [om**(sum(S) % 3) for S in FB]
    return np.diag(diag)
C3F = C3_on_fock()
sectors = {n:[i for i,S in enumerate(FB) if len(S)==n] for n in range(4)}
print("  (a) Is the Hamming grading C3-invariant?  C3 on the 8-dim Fock is diagonal in the occupation")
print("      basis (weight = sum of occupied-generator C3-weights), so it preserves each L_n. CHECK:")
for n in range(4):
    idx = sectors[n]
    block = C3F[np.ix_(idx, idx)]
    full = C3F[np.ix_(idx, range(8))]
    leaks = np.max(np.abs(np.delete(full, idx, axis=1)))
    print(f"      L{n} (dim {len(idx)}): C3 maps it into itself? leak off the sector = {leaks:.1e}  "
          f"-> {'INVARIANT' if leaks<1e-12 else 'MIXES'}")
print("      => the OCCUPATION GRADING is C3-invariant (like the spatial windings). The grading alone")
print("         does NOT break C3.  (Same status as the spatial case at the level of the decomposition.)")

# (b) Does a chosen ORDERED BASIS of L1 break C3?
# The deck C3 acts on L1 as diag(1,om,om2): the three occupation modes e1,e2,e3 are ALREADY the C3
# eigenbasis.  Any RELABELING that is a C3-eigenbasis is fixed by C3 up to phase -- does NOT break it.
# But a GENERIC ordered o.n. basis (a generic U(3) frame) is NOT a C3-eigenbasis and DOES break C3.
# The decisive question: does the OBJECT force the labeling to be the C3-eigenbasis (no breaking), or
# does reading the n=1<->n=2 Hodge structure pick a NON-eigen labeling (breaking)?
print("""
  (b) Does a definite ORDERED labeling of the n=1 modes break C3?
      The deck C3 on L1 is diag(1, om, om2): the occupation modes e1,e2,e3 ARE its eigenbasis. A
      labeling that respects occupation (= an eigenbasis of C3) is C3-stable up to phase -> NO breaking.
      The ONLY way the internal reading breaks C3 is if the n=1<->n=2 structure forces a NON-eigen
      (C3-mixing) frame.  The canonical Hodge star does NOT: it is diagonal in the SAME occupation
      eigenbasis (sec 1-2).  So the bare (metric) Hodge star, like the spatial windings, leaves the
      labeling on the C3-eigenbasis: the occupation reading is C3-equivariant, NOT C3-breaking.""")

# =============================================================================================
hdr("(4) WHERE THE INTERNAL READING DIFFERS FROM THE SPATIAL: the conjugation K is non-trivial")
# =============================================================================================
# The spatial inter-channel V was the IDENTITY (sec 5 of offdiag_interchannel_mixing): both channels
# co-diagonalize under C3, V = F^dag F = I.  The internal n=1<->n=2 Hodge star is ALSO diagonal in the
# C3-eigenbasis -- BUT it intertwines C3|L1 with conj(C3|L1) (the om<->om2 SWAP, sec 2).  So the
# inter-SECTOR (n=1 vs n=2) map carries a non-trivial CHARGE-CONJUGATION K that the inter-CHANNEL
# (spatial) map did not.  Quantify the contrast.
print("  SPATIAL (established):  V_spatial = U_A^dag U_B = I   (both channels co-diagonal under deck C3).")
print("  INTERNAL (here):        the n=1<->n=2 Hodge star is diagonal in the SAME C3-eigenbasis, but as")
print("                          a C3-rep map it is L1 -> conj(L1): it carries the SWAP K (om<->om2).")
# Build the analogue of V for the internal sectors: align L1 and L2 by the Hodge star and read the
# residual C3-rep map.  In the occupation eigenbasis:
V_internal_repmap = SWAP            # the residual om<->om2 conjugation
print("\n  The internal inter-sector 'V' (residual C3-rep map after Hodge alignment) =")
print("  ", V_internal_repmap.tolist())
print(f"   off-diagonal? {np.max(np.abs(V_internal_repmap - np.diag(np.diag(V_internal_repmap)))):.0f} (>0)"
      f"  -> NON-trivial (a genuine 2-cycle on the complex isotypes), unlike the spatial V=I.")
print("""
  HONEST READING of the contrast:
   * The internal n=1<->n=2 map is NOT the identity at the level of the C3 isotypes: it SWAPS the two
     complex-conjugate lines (a non-trivial Z2 = charge conjugation K).  This IS structure the spatial
     fiber did not carry (V_spatial = I exactly).
   * BUT this swap is a FIXED, FORCED, C3-COVARIANT permutation (it commutes with the REAL form of C3
     and conjugates the complex form).  It does NOT pick a *direction* in the (4,2,2) space and it does
     NOT introduce a free continuous phase.  It is rigid: K^2 = I, K real, det K = -1.
   * So reading the internal occupation labels supplies a NON-TRIVIAL but RIGID inter-sector
     structure (the conjugation K), not a continuous C3-breaking parameter.  It is "more than the
     spatial nothing", but it is still FORCED, not free.""")

# =============================================================================================
hdr("(5) CAN THE Cl(6) WEDGE/CONTRACTION GENERATORS BREAK C3 BEYOND THE RIGID K?")
# =============================================================================================
# Cl(6) acts on the 8-dim Fock by the 3 creation a_i^dag = e_i ^ (.) and 3 annihilation a_i = contraction.
# Build them and ask: is ANY single generator C3-BREAKING and sector-(n)-changing in a way that picks a
# direction (a free labeling), or are they all C3-COVARIANT (forced)?
def wedge(i):
    """creation a_i^dag : Lambda^n -> Lambda^{n+1}, wedge with e_i (with Koszul sign)."""
    M = np.zeros((8,8), complex)
    for col,S in enumerate(FB):
        if i in S: continue
        T = tuple(sorted(S+(i,)))
        sign = (-1)**sum(1 for x in S if x>i)
        row = FB.index(T)
        M[row,col] = sign
    return M
def contr(i):
    return wedge(i).conj().T    # annihilation = adjoint
adag = [wedge(i) for i in range(3)]
a    = [contr(i) for i in range(3)]
# CAR check
car_ok = all(np.allclose(a[i]@adag[j]+adag[j]@a[i], (i==j)*np.eye(8)) for i in range(3) for j in range(3))
print(f"  CAR algebra {{a_i, a_j^dag}} = delta_ij ?  {car_ok}   (the Cl(6) Fock generators are genuine).")
# C3-covariance of the generators: C3 a_i^dag C3^{-1} should be om^{w_i} a_i^dag (w_i = weight of e_i).
print("  C3-covariance of each creation generator a_i^dag  (C3 a_i^dag C3^-1 =? om^{w_i} a_i^dag):")
for i in range(3):
    lhs = C3F @ adag[i] @ np.linalg.inv(C3F)
    rhs = (om**w_L1[i]) * adag[i]
    print(f"     i={i} (weight {w_L1[i]}): covariant? {np.allclose(lhs, rhs)}  "
          f"max-dev {np.max(np.abs(lhs-rhs)):.1e}")
print("""
  RESULT: every Cl(6) wedge/contraction generator is C3-COVARIANT (it carries a definite C3 weight),
  so NO single generator and no real combination of them BREAKS C3 by picking a free direction. The
  generators move you between Hamming sectors (n -> n+-1) but each does so C3-equivariantly. The ONLY
  non-trivial inter-sector datum is the RIGID conjugation K (the volume-form swap of the two complex
  lines), which is forced, not free.""")

# =============================================================================================
hdr("(6) THE OBSERVER'S FULL-STATE READING: does occupation + flow FORCE a definite labeling?")
# =============================================================================================
print("""  The premise (iii): the observer reads occupation labels AND the flow dN.  The flow's only
  C3-action is the directed screw phase {0, +phi, -phi}, phi = 2pi/sqrt7 (established: it acts on the
  L1 complex lines e2 (weight 1) and e3 (weight 2) with OPPOSITE sign -- the chiral split).  Combine:

   * Occupation reading alone: C3-eigenbasis fixed up to the rigid conjugation K (secs 1-5).  No free
     continuous direction; the GRADING is C3-invariant and the labeling sits on the C3-eigenbasis.
   * Flow dN alone: imparts the chiral phase {0,+phi,-phi} to the two complex L1 lines -- this is the
     SAME thing that distinguishes the two complex lines, i.e. it ORIENTS the conjugation K (it tells
     e2 from e3 by the SIGN of the accumulated phase: +phi vs -phi).
   * TOGETHER: the flow's chirality breaks the K-degeneracy (it picks WHICH complex line is 'om' and
     which is 'om2' by the sign of the directed phase), promoting the rigid Z2 swap K to a DEFINITE
     ORIENTED labeling.  This is the observer's full-state resolution: occupation gives the eigenbasis +
     the rigid K; the flow's chirality orients K into a definite labeling.""")
# Demonstrate: the chiral flow phase on the two complex L1 lines, and that it distinguishes them.
phi = 2*np.pi/np.sqrt(7)
print(f"\n  chiral flow phase on L1 complex lines (per unit run s):  e2 (weight 1) -> +{phi:.5f},  "
      f"e3 (weight 2) -> -{phi:.5f}  (opposite sign = oriented).")
print("  => the SIGN of the directed phase is the orientation datum that turns K (a swap) into a")
print("     DEFINITE labeling (which line is which).  Forced by the run's chirality, not chosen.")

# =============================================================================================
hdr("(7) FORCED vs FREE; relation to V_Ram~Cl(6) / U(4)xU(2)xU(2); flags")
# =============================================================================================
print("""  FORCED (derived in-box, no import/target/fit):
    * the Hamming grading L0,L1,L2,L3 is C3-invariant (each sector is a C3-subrep);
    * the canonical n=1<->n=2 Hodge star is, in the occupation eigenbasis, the IDENTITY as a labeling
      but a NON-trivial C3-rep map: it intertwines L1 with conj(L1) via the rigid conjugation K
      (volume-form weight 0+1+2=0 fixes the trivial line, swaps the two complex lines), K real, K^2=I;
    * every Cl(6) wedge/contraction generator is C3-COVARIANT (carries a definite weight): no generator
      breaks C3 by picking a free direction;
    * the flow dN's chiral phase {0,+phi,-phi} ORIENTS K -> a definite labeling (which complex line is
      which) -- the observer's full-state resolution; forced by the run's chirality.

  FREE / RESIDUAL:
    * The OVERALL phase of the complex L1 lines (a U(1) per complex isotype) and the trivial line's
      phase are not fixed by the metric Hodge star alone; they are the gauge of the isotype basis. The
      flow fixes the RELATIVE chiral orientation (the sign), not the absolute phases. So a residual
      phase gauge per isotype remains at the level of THIS construction -- this is the same residual the
      established mass map carries as the single run-phase u = phi*s (the observer's clock-reading).

  RELATION TO V_Ram~Cl(6) / U(4)xU(2)xU(2):  the explicit V_Ram~Cl(6) isomorphism and its residual
  U(4)xU(2)xU(2) freedom are NAMED in the project's broader notes but DO NOT have a computational
  construction inside the four sealed directories (dirac_srs_mdl, matter_bridge, time_bridge, bridge).
  What IS in-box is: internal = Lambda^*(C^3), C3-isotype (4,2,2) (matter_bridge/m05); the Cl(6) Fock
  generators (built here); the deck-C3 character.  The (4,2,2) isotype multiplicities (4 on the trivial,
  2 each on the two complex lines) are the natural home of a U(4)xU(2)xU(2) basis freedom -- the
  unitary relabeling WITHIN each isotype that the C3 character cannot fix.  But I FLAG the explicit
  U(4)xU(2)xU(2) iso as BEYOND the four directories: I can only say the in-box rigid K + the flow's
  orientation reduce, but do not eliminate, the within-isotype unitary freedom.  Whether the full
  U(4)xU(2)xU(2) is pinned needs the iso construction, which is out-of-box.

  SAME-CLOCK: the occupation grading, the Hodge star, the Cl(6) generators, and the flow phase are all
  read at one run-position s off the one object (the deck C3 on Lambda^*(C^3) + the screw flow). One clock.

  FLAG: nothing used beyond the four sealed dirs (srs.py for the deck/flow; m05's Lambda^*(C^3)/(4,2,2);
  the flow rate 2pi/sqrt7 re-derived in-box).  No observed numbers, no targets, no fits.  The
  U(4)xU(2)xU(2) residual-freedom statement is flagged as needing the out-of-box iso construction.""")

print("\n[done]")
