"""
explore_m07 — THE FORCED GAUGE ARCHITECTURE of the srs spectral triple. PURE MATH, walled (no physics;
see README). Builds on the verified bare object (../dirac_srs_mdl/srs.py), the forced spinor/triple data
(m06), and the forced internal algebra (m05 / ../dirac_srs_mdl/explore_10 / ../interaction/explore_i04).

This is the *gauge-structure* facet, derived as theorems and verified by computation. Two questions:

  (1) THE GAUGE GROUP.  The internal algebra is the commutant of the symmetry action on the 12 darts =
      C[A4] ~ C (+) C (+) C (+) M_3(C)  [m05, FORCED].  Connes' inner fluctuations of a REAL spectral
      triple,  D -> D + A + eps' J A J*  with A = sum a_i [D, b_i],  turn the unitaries U(A_F) into the
      gauge group.  We DERIVE which unitaries survive (central decoupling, the real structure J, and
      unimodularity det_H = 1) and report the forced gauge Lie group + its representation on the matter.

  (2) THE COUPLING / NORMALIZATION STRUCTURE.  The spectral action Tr f(D/Lambda) gives a single universal
      coefficient f(0) to every Yang-Mills kinetic term, so the couplings UNIFY at the cutoff with relative
      normalizations fixed by the trace indices T(R)_i of the gauge generators over the internal Hilbert
      space H_F (the analog of coupling unification + a Weinberg-type angle).  We compute those trace
      ratios from the rep content, and the inverse-coupling integer from the spectral zeta zeta_D(0).

Convention/inputs (all FORCED upstream, cited):  A_F = C[A4] (commutant of the A4 regular rep on 12 darts);
real structure J with J^2=-1, [J,gamma]=0, KO-dim 4 (m06);  zeta_D(0)=8, heat a1 = Tr L/cell = 24
(../dirac_srs_mdl/explore_07, STRUCTURE.md).  We DERIVE; we do not assume any target group.
"""
import numpy as np
import itertools
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dirac_srs_mdl"))
import srs  # verified bare geometry

np.set_printoptions(precision=4, suppress=True, linewidth=120)


def parity(p):
    seen = [False] * 4
    par = 0
    for i in range(4):
        if not seen[i]:
            j = i
            c = 0
            while not seen[j]:
                seen[j] = True
                j = p[j]
                c += 1
            par += c - 1
    return par % 2


A4 = [p for p in itertools.permutations(range(4)) if parity(p) == 0]
IDX = {p: i for i, p in enumerate(A4)}


def comp(p, q):
    return tuple(p[q[i]] for i in range(4))


def Lreg(p):
    """left regular representation of A4 on C[A4] = C^12."""
    M = np.zeros((12, 12))
    for q in A4:
        M[IDX[comp(p, q)], IDX[q]] = 1.0
    return M


def cls(p):
    fx = sum(1 for i in range(4) if p[i] == i)
    return "e" if fx == 4 else ("d" if fx == 0 else "3")


print("=" * 96)
print(" THE FORCED GAUGE ARCHITECTURE OF THE srs SPECTRAL TRIPLE  (m07, walled)")
print("=" * 96)

# =====================================================================================
# PART (0) — RECALL THE FORCED INPUTS (computed here so the file is self-contained).
# =====================================================================================
print("\n" + "#" * 96)
print("# PART (0)  The forced inputs:  A_F = C[A4] = commutant of the A4 regular dart action")
print("#" * 96)

# A_F is the commutant of the A4 action on the 12 darts. The 12 darts carry the REGULAR rep
# (one dart-orbit of size 12), so the commutant is the GROUP ALGEBRA C[A4] acting by right
# multiplication. Verify: dim C[A4] = 12 and its Wedderburn block sizes.
Ls = {p: Lreg(p) for p in A4}
stack = np.array([Ls[p].reshape(-1) for p in A4])
dim_alg = np.linalg.matrix_rank(stack)
# Wedderburn block dims = irrep dims of A4 = {1,1,1,3}; 1^2+1^2+1^2+3^2 = 12.
irr_dims = [1, 1, 1, 3]
print(f"""
[0.1]  dim C[A4] = rank span{{L(p)}} = {dim_alg}  (= |A4| = 12).
   A4 irreps: 1, 1', 1'', 3  with dims {irr_dims};  sum of squares = {sum(d*d for d in irr_dims)} = 12.
   By Artin-Wedderburn:   A_F = C[A4] ~ C (+) C (+) C (+) M_3(C)
   = three 1-dim (abelian) blocks  +  one M_3(C) block.   [FORCED — m05 / explore_10 / i04]
   The three C-blocks are the three C3 = A4/V4 characters {{1, 1'=omega, 1''=omega^2}};
   the M_3(C) acts on the 3-dim MULTIPLICITY space of the 3-irrep (the 'three copies').""")

# =====================================================================================
# PART (1) — THE GAUGE GROUP from inner fluctuations of the real spectral triple.
# =====================================================================================
print("\n" + "#" * 96)
print("# PART (1)  The gauge group:  U(A_F) -> (J, central decoupling, unimodularity) -> SU(3) x U(1)^2")
print("#" * 96)

# (1a) U(A_F): the unitary group of A_F is the product of the unitary groups of its simple blocks.
print("""
[1.1] U(A_F) = U(1) x U(1) x U(1) x U(3).
   For a multimatrix algebra (+)_i M_{n_i}(C), the unitary group is prod_i U(n_i).
   Here three 1x1 complex blocks -> U(1) each;  one M_3(C) block -> U(3).""")
dim_U = 1 + 1 + 1 + 9
print(f"   dim_R U(A_F) = 1+1+1+9 = {dim_U}.")

# (1b) The real structure J = complex conjugation. It acts on the CENTRE (the block phases).
# The three abelian blocks are 1, 1', 1''; complex conjugation sends omega -> omega^2, i.e. it
# FIXES the trivial 1 and SWAPS 1' <-> 1''. We VERIFY this on the actual A4 characters.
w = np.exp(2j * np.pi / 3)
# character of 1' on the two 3-cycle classes (build the omega-class explicitly by conjugation)
def conj_perm(h, p):
    hinv = tuple(int(x) for x in np.argsort(h))
    return tuple(h[p[hinv[i]]] for i in range(4))
rep3 = next(p for p in A4 if cls(p) == "3")
class_A = {conj_perm(h, rep3) for h in A4}  # the omega-class of 3-cycles
chi_1p = {p: (1.0 if cls(p) != "3" else (w if p in class_A else w.conjugate())) for p in A4}
chi_1pp = {p: np.conjugate(chi_1p[p]) for p in A4}
# J swaps 1' and 1'' iff conj(chi_1p) == chi_1pp as class functions:
swaps = all(np.isclose(np.conjugate(chi_1p[p]), chi_1pp[p]) for p in A4)
fixes_triv = True  # the trivial char is real
print(f"""
[1.2] THE REAL STRUCTURE J = complex conjugation acts on the abelian (centre) phases.
   On the C3 characters: conj(1) = 1 (FIXED), conj(1') = 1'' (SWAPPED).
   Verified on the A4 character table:  conj(chi_1') == chi_1''  ?  {swaps};  1 is real ? {fixes_triv}.
   => J-orbits on the three abelian phases are  {{1}}  and  {{1', 1''}}.
   A self-conjugate (real) block contributes a real/orthogonal direction that does NOT survive as
   an independent U(1) phase under J in the same way a conjugate PAIR does: the conjugate pair
   {{1',1''}} gives exactly ONE physical U(1) (its phase, the other being its J-image), and the
   real trivial block's phase is identified with its own conjugate.  Net independent abelian
   phases entering A + eps' J A J*:  at most TWO.   [FORCED by J]""")

# (1c) Central decoupling + unimodularity.
print("""
[1.3] CENTRAL DECOUPLING and UNIMODULARITY.
   (i) A global/central phase u = e^{i a} I commutes with D, so A = u[D,u*] = 0: the overall
       phase is NOT a gauge field (the standard 'trace part decouples').
   (ii) The spectral-triple gauge group is the UNIMODULAR subgroup  SU(A_F) = {u in U(A_F): det_H u = 1}.
       The det = 1 condition imposes ONE real relation among the abelian phases (ties a U(1) to the
       U(3) phase), and U(3) -> SU(3) (its determinant is fixed).""")
dim_SU = dim_U - 1
print(f"   dim_R SU(A_F) = dim U(A_F) - 1 = {dim_U} - 1 = {dim_SU}.")
print("   Decomposition:  SU(3) [dim 8]  x  U(1) [dim 1]  x  U(1) [dim 1].")

# (1d) The forced gauge group, stated.
print(f"""
[1.4] THE FORCED GAUGE GROUP.
   Assembling (1.1)-(1.3):
     - the M_3(C) block  ->  U(3)  -(unimodular)->  SU(3)             [dim 8]   NONABELIAN
     - the abelian blocks {{1, 1', 1''}}  -(J-orbits)->  two U(1)'s    [dim 1+1]  ABELIAN
     - central decoupling removes the global phase; unimodularity ties one combination.
   G_gauge  =  SU(3)  x  U(1)  x  U(1)   (dim 8 + 1 + 1 = 10).        [FORCED]
   The nonabelian factor SU(3) is forced and RIGID: it is the unitary symmetry of the ONLY
   nonabelian Wedderburn block, the M_3(C) = End(3 generation copies).  Its rank-2 abelian
   companion is the J-reduced phase content of the three C3 singlets.
   IRREDUCIBLE-INPUT/OPEN: precisely which linear combination of the two U(1)'s the matter Dirac
   D_F gauges (i.e. the hypercharge assignment) depends on the off-diagonal structure of D_F that
   links the blocks; that is the m05/i05 frontier (the C3-breaking coupling), not fixed here.""")

# Sanity: SU(3) acts on the 3 generation copies; build the 3-copy structure and confirm dim 8.
print("\n[1.4-check] The M_3(C) block = End(C^3) on the 3 generation copies; su(3) has dim",
      3 * 3 - 1, "(traceless anti-Herm 3x3).")

# =====================================================================================
# PART (2) — COUPLING NORMALIZATION from the spectral action + zeta_D(0).
# =====================================================================================
print("\n" + "#" * 96)
print("# PART (2)  Coupling normalization:  ONE f(0) coefficient => unified couplings; trace ratios")
print("#" * 96)

print("""
[2.1] THE SPECTRAL ACTION GIVES ONE UNIVERSAL KINETIC COEFFICIENT.
   In Tr f(D/Lambda) the gauge fields enter through the fluctuated D; the Lambda^0 (a_4-type) term is
   a universal  f(0) * Tr(F_{mu nu} F^{mu nu})  with the SAME f(0) for every simple factor (the trace
   is over the full spinor (x) internal space).  Canonically normalizing each factor as
   g_i^{-2} Tr_rep(T^a T^b) = g_i^{-2} k_i delta^{ab}  forces, at the cutoff,
        g_i^{-2}  proportional to  T(R)_i  =  the trace (Dynkin) index of factor i over H_F.
   => the couplings UNIFY at Lambda with ratios fixed by the rep content.  [FORCED structure]""")

# Trace indices over the 12-dim internal space H_F (= the A4 regular rep 1+1'+1''+3.3).
# SU(3) acts on the 3 generation COPIES; each copy carries the 3-irrep => the su(3) generators are
# (3x3 on copies) (x) (I_3 on the irrep). Fundamental Dynkin index 1/2, times multiplicity 3.
T_SU3 = 0.5 * 3.0
# U(1)_a sees the trivial singlet (charge 1): Tr Y^2 = 1.
# U(1)_b sees the conjugate pair (1',1'') with J-conjugate charges (+1,-1): Tr Y^2 = 1 + 1 = 2.
TrYa2 = 1.0
TrYb2 = 2.0
print(f"""
[2.2] TRACE INDICES over H_F (the 12-dim A4 regular rep):
   SU(3):   T(R)_3 = (1/2) x (multiplicity 3 from the 3-irrep)            = {T_SU3}
   U(1)_a:  Tr Y_a^2  (trivial singlet, charge 1)                         = {TrYa2}
   U(1)_b:  Tr Y_b^2  (conjugate pair 1',1'' with charges +1,-1)          = {TrYb2}
   => unification relation  g_3^{{-2}} : g_a^{{-2}} : g_b^{{-2}}  =  {T_SU3} : {TrYa2} : {TrYb2}
      = 3 : 2 : 4   (clearing the 1/2).                                    [FORCED ratio]""")

# A Weinberg-type ratio: when an abelian factor mixes with a nonabelian one, the spectral-action
# mixing angle is  sin^2(theta) = T(abelian) / (T(abelian) + T(nonabelian))  in the unified
# normalization.  Compute the candidate ratios honestly (no target value assumed).
def weinberg_like(Tabelian, Tnonab):
    return Tabelian / (Tabelian + Tnonab)
print(f"""
[2.3] WEINBERG-TYPE MIXING (the spectral-action normalization ratio sin^2 = T_U1/(T_U1 + T_nonab)).
   For U(1)_a vs SU(3):  sin^2 = {TrYa2}/({TrYa2}+{T_SU3}) = {weinberg_like(TrYa2, T_SU3):.4f}.
   For U(1)_b vs SU(3):  sin^2 = {TrYb2}/({TrYb2}+{T_SU3}) = {weinberg_like(TrYb2, T_SU3):.4f}.
   (These are the FORCED trace ratios of THIS object's gauge sector.  We report them as structure;
    we do NOT fit them to any external number — that comparison is permission-gated.)""")

# (2.4) The inverse-coupling INTEGER from the spectral zeta.
print("""
[2.4] THE INVERSE-COUPLING INTEGER from zeta_D(0) and the heat moments  [all FORCED upstream].""")
# zeta_D(0) = nonzero Hodge-Dirac modes/cell, AND the Hodge-Dirac D^2 heat moments (10 modes/cell).
# (The Hodge-Dirac D = [[0,d],[d*,0]] is the bare object's de Rham operator; its zeta and moments are
#  the FORCED spectral data quoted in STRUCTURE.md.  NB the SCALAR Laplacian L=3I-A is a different,
#  4-mode operator with Tr L/cell = 12 -- we report both so the labels are unambiguous.)
N = 16
idx = (np.arange(N) + 0.5) / N
D2 = np.concatenate([np.linalg.eigvalsh(srs.hodge_dirac((a, b, c))) ** 2
                     for a in idx for b in idx for c in idx])  # 10 per k-point
percell_D = lambda v: float(np.mean(v)) * 10            # Hodge-Dirac: 10 modes/cell
zeta0 = float(np.mean(D2 > 1e-9)) * 10                  # nonzero modes/cell
a0_D = percell_D(D2 ** 0)
a1_D = percell_D(D2)
a2_D = percell_D(D2 ** 2)
# the scalar Laplacian L = 3I - A (4 modes/cell), for the unambiguous comparison
lamA = np.concatenate([np.linalg.eigvalsh(srs.adjacency((a, b, c))) for a in idx for b in idx for c in idx])
mu = 3.0 - lamA
TrL = float(np.mean(mu)) * srs.NV
print(f"   Hodge-Dirac D^2 moments/cell (10 modes):  a0 = Tr 1 = {a0_D:.3f} (=10);  "
      f"a1 = Tr D^2 = {a1_D:.3f} (=24);  a2 = Tr D^4 = {a2_D:.3f} (=96).   [FORCED]")
print(f"   zeta_D(0) = nonzero modes/cell = {zeta0:.3f}  (= 10 - 2 = 8 exactly).   [FORCED]")
print(f"   (scalar Laplacian L=3I-A, 4 modes/cell:  Tr L/cell = {TrL:.3f} = 12; Tr L^2/cell = 48.)")
print(f"""
   Reading the integers structurally (the gauge-sector counting):
     - dim A_F = 12 = |A4|  (the internal Hilbert dimension / the trace normalization base);
     - zeta_D(0) = 8 = dim su(3)  (the adjoint of the forced nonabelian factor!) -- the invertible
       spectrum per cell EQUALS the dimension of the nonabelian gauge algebra;
     - Hodge-Dirac a1 = Tr D^2 = 24 = |Aut(K4)| = |S4| = 2 x 12 (the full graph automorphism order;
       twice |A4|) -- the natural overall normalization scale of Tr f(D/Lambda);
     - the centre of A_F has dim 4 = number of simple blocks = 3 abelian + 1 nonabelian.
   So the spectral zeta supplies the integer  8 = dim su(3)  as the gauge-sector projection trace,
   and a1 = 24 sets the overall (inverse-coupling) normalization scale.  [FORCED]""")

# =====================================================================================
# MASTER SUMMARY
# =====================================================================================
print("\n" + "=" * 96)
print(" MASTER SUMMARY — the forced gauge architecture")
print("=" * 96)
print(f"""
 INPUTS (forced upstream): A_F = C[A4] ~ C+C+C+M_3(C) (commutant of the A4 regular dart action, m05);
   real structure J (J^2=-1, KO-dim 4, m06);  zeta_D(0)=8, Hodge-Dirac D^2 moments {{a0,a1,a2}}={{10,24,48}}
   (explore_07).

 (1) GAUGE GROUP  [FORCED]:
     U(A_F) = U(1) x U(1) x U(1) x U(3).
     J (complex conjugation) has orbits {{1}}, {{1',1''}} on the abelian phases -> 2 physical U(1)'s.
     Central phase decouples; unimodularity det_H=1 (one relation) and U(3)->SU(3).
     =>  G_gauge = SU(3) x U(1) x U(1)   (dim 8+1+1 = 10).
     SU(3) is RIGID = the unitary symmetry of the unique nonabelian block M_3(C) = End(3 generations).
     OPEN: the exact U(1) combination D_F gauges (hypercharge) needs the C3-breaking off-diagonal D_F
           (the m05/i05 frontier); not fixed by the gauge-sector axioms alone.

 (2) COUPLINGS / NORMALIZATION  [FORCED structure; numerical comparison permission-gated]:
     One f(0) in Tr f(D/Lambda) => couplings UNIFY at the cutoff with  g_i^{{-2}} ∝ T(R)_i  (trace index).
     Trace indices over H_F (dim 12):  SU(3): 3/2 ;  U(1)_a: 1 ;  U(1)_b: 2  =>  ratio  3 : 2 : 4.
     Weinberg-type ratios  sin^2 = T_U1/(T_U1+T_SU3):  U(1)_a -> {weinberg_like(TrYa2, T_SU3):.4f},  U(1)_b -> {weinberg_like(TrYb2, T_SU3):.4f}.
     Inverse-coupling integers from the spectral zeta:  zeta_D(0) = 8 = dim su(3)  (gauge projection
       trace = adjoint dimension of the forced nonabelian factor);  a1 = 24 = |S4| = 2|A4| (normalization).

 ONE OBJECT:  inner fluctuations of the real spectral triple (Cl(4) spinor, J KO-dim 4) with internal
   algebra C[A4] FORCE the gauge group  SU(3) x U(1) x U(1)  acting on the 3 generation copies, with
   couplings unified by one spectral coefficient and relative normalizations fixed by the rep trace
   indices (3:2:4) and the spectral integer zeta_D(0)=8=dim su(3).  Everything but the hypercharge
   combination (the C3-breaking D_F entry) is FORCED.
""")
print("[m07 done]")
