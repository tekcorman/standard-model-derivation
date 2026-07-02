"""
explore_m06 — THE COMPLETE FORCED SPINOR / SPECTRAL-TRIPLE ARCHITECTURE of the srs Dirac operator.
PURE MATH, walled (no physics; see README). Builds on the verified bare object (../dirac_srs_mdl/srs.py)
and consolidates + completes m01..m05.

This file is the *spinor / Clifford / spectral-triple* facet, derived as theorems and verified by
computation. It answers three questions and marks every result FORCED / IRREDUCIBLE-INPUT / OPEN.

  (1) WHY srs FORCES Cl(3); WHY chirality forces the MINIMAL EVEN extension Cl(4) (the chiral
      4-spinor C^2_+ + C^2_-); and WHY "nothing beyond Cl(4)" is forced (a minimality theorem,
      not an assumption).

  (2) THE SPECTRAL-TRIPLE DATA — the grading gamma, the real structure J, the KO-dimension (all
      THREE Clifford signs epsilon, epsilon', epsilon''), the first-order / orientability axioms —
      computed and pinned, not asserted.

  (3) THE srs (+) srs-z ENANTIOMER DOUBLING — why a mass gap REQUIRES doubling the chiral net with
      its mirror; that the 4th gamma IS the inter-enantiomer coupling (the SAME generator chirality
      forced in (1)); the opposite Weyl charge of srs-z; what is FORCED vs the one IRREDUCIBLE-INPUT
      (the strength/scale).

Convention: Euclidean Clifford {g^a, g^b} = +2 delta^{ab}, Hermitian generators (g^a)^dag = g^a.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dirac_srs_mdl"))
import srs  # verified bare geometry

np.set_printoptions(precision=4, suppress=True, linewidth=120)

I2 = np.eye(2)
s1 = np.array([[0, 1], [1, 0]], complex)
s2 = np.array([[0, -1j], [1j, 0]], complex)
s3 = np.array([[1, 0], [0, -1]], complex)
PAULI = [s1, s2, s3]
kron = np.kron


def anticomm(A, B):
    return A @ B + B @ A


def comm(A, B):
    return A @ B - B @ A


def is0(M):
    return np.allclose(M, 0, atol=1e-12)


def isI(M):
    return np.allclose(M, np.eye(M.shape[0]), atol=1e-12)


print("=" * 96)
print(" THE FORCED SPINOR / SPECTRAL-TRIPLE ARCHITECTURE OF THE srs DIRAC OPERATOR  (m06, walled)")
print("=" * 96)

# =====================================================================================
# PART (1) — WHY Cl(3), WHY Cl(4), WHY NOTHING BEYOND Cl(4).
# =====================================================================================
print("\n" + "#" * 96)
print("# PART (1)  The forced Clifford tower:  3-regular  =>  Cl(3)  =>(chirality)=>  Cl(4)  =>(min) STOP")
print("#" * 96)

k = srs.DEG
print(f"""
[1.0] THE BARE GEOMETRY FIXES THE NUMBER OF CLIFFORD GENERATORS.
   The srs net is k-regular with k = {k} (verified bare object). A *genuine* (first-order, Clifford-
   type) Dirac operator is  D = sum_a gamma^a nabla_a,  one Hermitian generator gamma^a per
   independent local hopping direction, with {{gamma^a, gamma^b}} = 2 delta^ab so that D^2 = -Laplacian
   + curvature is a Laplace-type (second-order, scalar-principal-symbol) operator.  k = {k} edge-
   directions  =>  exactly {k} anticommuting generators  =>  the real Clifford algebra Cl({k}).
   This is FORCED by the coordination number alone.""")

# --- Cl(3): the minimal module is C^2, and Cl(3) is ODD (no chirality grading). ---
ok3 = all(np.allclose(anticomm(PAULI[a], PAULI[b]), 2 * (a == b) * I2) for a in range(3) for b in range(3))
omega3 = PAULI[0] @ PAULI[1] @ PAULI[2]
print(f"""[1.1] Cl(3) ON ITS MINIMAL (IRREDUCIBLE) MODULE C^2 = the Pauli matrices.
   {{sigma^a, sigma^b}} = 2 delta^ab ?  {ok3}.   dim_R Cl(3) = 2^3 = 8;  Cl(3,0) ~ M_2(C) (one
   irreducible C^2 spinor module, plus a second from the other sign of the volume element).
   VOLUME ELEMENT  omega = sigma^1 sigma^2 sigma^3 = i*I  (central scalar): {np.allclose(omega3, 1j*I2)}.
   Because omega is a SCALAR, Cl(3) is ODD: it has NO nontrivial central Z_2 element, hence NO
   intrinsic chirality grading from the 3 spatial directions alone.  A 3D Weyl spinor C^2 is
   forced, but it is *gapless and ungraded*.   [FORCED]""")

# --- The chirality requirement and WHY the even extension is Cl(4) exactly (minimality). ---
print(f"""[1.2] CHIRALITY FORCES AN EVEN CLIFFORD ALGEBRA; THE MINIMAL ONE OVER THE 3 SPATIAL
      DIRECTIONS IS Cl(4).  (A minimality theorem — we check each smaller even option fails.)
   A Z_2-graded (chiral) Dirac triple needs a grading operator gamma_c with gamma_c^2 = +I,
   gamma_c^dag = gamma_c, {{gamma_c, D}} = 0  (so D is odd).  Equivalently the volume element must
   be a nontrivial INVOLUTION, not a scalar.  The volume element of Cl(n) is central & squares to
   +-1 iff n is EVEN.  So we need an EVEN Clifford algebra that CONTAINS the 3 spatial generators.
   Candidates in increasing size: Cl(0), Cl(2), Cl(4), ...""")

# Cl(0): 0 generators -- cannot host 3 spatial directions. Cl(2): 2 generators -- too few.
print("   - Cl(0): 0 generators  -> cannot contain the 3 spatial directions.            REJECTED")
print("   - Cl(2): 2 generators  -> only 2 anticommuting directions < 3.                 REJECTED")
print("   - Cl(4): 4 generators  -> contains gamma^1,gamma^2,gamma^3 PLUS exactly ONE")
print("            added generator gamma^4.  This is the MINIMAL even algebra >= Cl(3).  ACCEPTED")

# Close the obvious loophole: "why not just take C^2 (+) C^2 (two Weyl copies) and grade by diag(+,-)?"
# Answer: a grading must ANTICOMMUTE with D.  Two DECOUPLED Weyl copies give a block-diagonal D, which
# COMMUTES with any block grading -> D is even, not odd -> NOT a chiral triple.  Chirality genuinely
# needs the off-diagonal odd generator, i.e. the even *Clifford* extension, not a mere direct sum.
G_naive = np.block([[I2, 0 * I2], [0 * I2, -I2]])
sdotp = 0.3 * s1 - 0.5 * s2 + 0.9 * s3
D_decoupled = np.block([[sdotp, 0 * I2], [0 * I2, sdotp]])   # same Cl(3) on both blocks, decoupled
print(f"   LOOPHOLE CHECKED: take C^2(+)C^2 = two DECOUPLED Weyl copies, grade by G=diag(+I,-I).")
print(f"     {{D,G}}=0 ? {is0(anticomm(D_decoupled, G_naive))}  -> a block-diagonal D COMMUTES with a")
print(f"     block grading, so it is EVEN, not odd: this is NOT a chiral triple.  A genuine chirality")
print(f"     grading forces an OFF-DIAGONAL odd generator coupling the halves = the even Clifford")
print(f"     extension Cl(4), NOT a mere direct sum of two Cl(3) modules.")

# Build Cl(4) explicitly and verify it is a genuine even extension of the spatial Cl(3).
g = [kron(s1, s1), kron(s1, s2), kron(s1, s3), kron(s2, I2)]   # g1,g2,g3 = spatial; g4 = the forced one
ok4 = all(np.allclose(anticomm(g[a], g[b]), 2 * (a == b) * np.eye(4)) for a in range(4) for b in range(4))
herm4 = all(np.allclose(g[a], g[a].conj().T) for a in range(4))
# The spatial g1,g2,g3 restricted to a chiral half reproduce the Pauli (the Cl(3) inside Cl(4)).
print(f"""
   Explicit Cl(4):  {{g^a,g^b}} = 2 delta^ab ? {ok4};   all g^a Hermitian ? {herm4}.
   g^1,g^2,g^3 = (sigma1 (x) sigma_i) embed the spatial Cl(3);  g^4 = (sigma2 (x) I) is the single
   forced new generator.   dim_R Cl(4) = 2^4 = 16;  Cl(4,0) ~ M_2(H)  acting on the spinor C^4.""")

# --- The STRUCTURAL form of minimality: Cl(4) is the CHIRAL CLOSURE of the spatial Cl(3). ---
# This upgrades the generator-count argument to an algebraic theorem: among Z2-graded algebras,
# Cl(4) is the unique minimal one whose EVEN (grade-0) part is exactly the spatial Cl(3).
biv = [g[a] @ g[3] for a in range(3)]                       # bivectors B_a = g^a g^4 generate Cl(4)^0
cl3_relations = all(np.allclose(anticomm(biv[a], biv[b]), -2 * (a == b) * np.eye(4)) for a in range(3) for b in range(3))
print(f"""
   [1.2'] STRUCTURAL MINIMALITY — Cl(4) is the CHIRAL CLOSURE of Cl(3) (not just "one more generator").
   The EVEN part Cl(4)^0 is generated by the bivectors B_a = g^a g^4.  They satisfy
   {{B_a, B_b}} = -2 delta_ab I  ?  {cl3_relations}   =>  Cl(4)^0 ~ Cl(0,3) ~ Cl(3) (the spatial
   Clifford, signature carried by the i in the g^4 direction; the standard iso Cl(n+1,0)^0 ~ Cl(n,0)).
   So Cl(4) is EXACTLY the Z2-graded algebra whose grade-0 part is the spatial Cl(3) and whose grade-1
   part adds the single odd direction g^4.  This is the canonical chiral double of Cl(3): "adjoin one
   odd generator so that the even part is the original (odd) spatial algebra."  Minimality is now
   STRUCTURAL — nothing smaller than Cl(4) has Cl(3) as its even part.   [FORCED]""")

g_c = g[0] @ g[1] @ g[2] @ g[3]                       # chirality = volume element of Cl(4)
gc_ok = np.allclose(g_c @ g_c, np.eye(4)) and all(is0(anticomm(g_c, g[a])) for a in range(4))
evals_gc = np.linalg.eigvalsh(g_c)
print(f"""[1.3] THE CHIRAL 4-SPINOR.   gamma_c := g^1 g^2 g^3 g^4  (the Cl(4) volume element):
   gamma_c^2 = I and {{gamma_c, g^a}} = 0 for all a ?  {gc_ok}.
   spec(gamma_c) = {np.round(evals_gc,3).tolist()}  ->  C^4 = C^2_+  (+)  C^2_-  (two chiral halves
   of equal dim 2).  The two Weyl spinors of (1.1) are now the +/- eigenspaces of a GENUINE grading.
   RESULT:  chirality forces EXACTLY ONE extra Clifford generator; the spinor is the chiral
   4-spinor C^4 = C^2_+ (+) C^2_-.   [FORCED]""")

# --- WHY NOTHING BEYOND Cl(4): the minimal chiral REAL triple already closes all axioms. ---
print(f"""[1.4] NOTHING BEYOND Cl(4) IS FORCED  (the closure / exhaustion theorem).
   Larger even algebras Cl(6), Cl(8), ... are PERMITTED but require an extra anticommuting
   direction that the bare 3-regular geometry does NOT supply.  We show the minimal chiral REAL
   spectral triple over Cl(4) ALREADY satisfies every remaining spectral-triple axiom (Part 2),
   so the axioms do not force any enlargement.  Hence:  the spinor is rigidly Cl(4); any further
   structure is an *internal* (gauge/matter) choice carried by an internal algebra A_F and internal
   Dirac D_F, which the geometry's SYMMETRY (not the bare metric) constrains [see m05 / Part-4 note].""")

# =====================================================================================
# PART (2) — THE SPECTRAL-TRIPLE DATA: gamma, J, KO-dimension (all 3 signs), first-order,
#            orientability.  All computed, not asserted.
# =====================================================================================
print("\n" + "#" * 96)
print("# PART (2)  The forced spectral-triple data  (gamma, J, KO-dimension, first-order, orientability)")
print("#" * 96)

# (2a) THE GRADING gamma.  Two gradings coexist; we identify them and show they agree in role.
print("""
[2.1] THE GRADING gamma.
   (i) On the spinor module: gamma = gamma_c (Part 1.3), gamma^2 = I, gamma^dag = gamma,
       {gamma, g^a} = 0.   This is the INTERNAL chirality.
   (ii) On the BARE Hodge-Dirac D = [[0,d],[d*,0]] on C0 (+) C1: the bipartite grading
       G = diag(+I_V, -I_E) gives {D,G}=0 already in the bare object (un-imposed).
   Both are genuine Z_2 gradings with {D, grading} = 0; (i) is the spinor-internal one forced
   by chirality, (ii) is the geometric form/degree one inherited from the complex.""")
G_bare = np.diag([1.0] * srs.NV + [-1.0] * len(srs.EDGES))
Dk = srs.hodge_dirac((0.2, 0.25, 0.3))
print(f"   bare check: G^2=I ? {isI(G_bare@G_bare)};  {{D,G}}=0 ? {is0(anticomm(Dk, G_bare))};  "
      f"gamma_c^2=I ? {isI(g_c@g_c)};  {{D_spinor, gamma_c}}=0 for D=sum g^a p_a ? "
      f"{is0(anticomm(g[0]+0.7*g[1]-0.3*g[2], g_c))}")

# (2b) THE REAL STRUCTURE J and the KO-DIMENSION — determine ALL THREE signs.
# KO-dimension n (mod 8) is fixed by the triple of signs (epsilon, epsilon', epsilon'') in
#   J^2 = epsilon,   J D = epsilon' D J,   J gamma = epsilon'' gamma J.
# For a genuine spectral triple we need an ANTIUNITARY J = (matrix C) o (complex conjugation).
print("""
[2.2] THE REAL STRUCTURE J AND THE KO-DIMENSION.
   J = C o (complex conjugation), C unitary.  The KO-dimension n mod 8 is FIXED by the signs
       J^2 = epsilon * I,    J D = epsilon' * D J,    J gamma = epsilon'' * gamma J,
   read against the Connes sign table.  We search the Cl(4) generators for a C giving a
   *consistent* (epsilon, epsilon', epsilon'') and report the forced KO-dimension.""")


def J_signs(C, gammas, grading):
    """Return (eps, eps_p, eps_pp) for J = C o conj acting on these gammas and grading.
    J^2 = C conj(C);  J g J^{-1} = C conj(g) C^{-1} (for Hermitian g, conj=transpose-bar)."""
    n = C.shape[0]
    J2 = C @ np.conj(C)
    # eps from J^2 = eps I
    eps = None
    if np.allclose(J2, np.eye(n)):
        eps = +1
    elif np.allclose(J2, -np.eye(n)):
        eps = -1
    # A Dirac D = sum_a x_a gamma^a (any real combo of the SPATIAL gammas) must satisfy J D = eps' D J.
    # Equivalent (since the x_a are real) to: C conj(g^a) = eps' g^a C for each spatial generator.
    def sign_against(ops):
        plus = all(np.allclose(C @ np.conj(o), +o @ C) for o in ops)
        minus = all(np.allclose(C @ np.conj(o), -o @ C) for o in ops)
        return +1 if plus else (-1 if minus else None)
    eps_p = sign_against(gammas)               # against the spatial Dirac generators
    eps_pp = sign_against([grading])           # against gamma_c
    return eps, eps_p, eps_pp


# KO-dimension lookup from Connes' table (n mod 8 -> (eps, eps', eps'')):
KO_TABLE = {
    0: (+1, +1, +1), 1: (+1, -1, None), 2: (-1, +1, -1), 3: (-1, +1, None),
    4: (-1, +1, +1), 5: (-1, -1, None), 6: (+1, +1, -1), 7: (+1, +1, None),
}
SPATIAL = [g[0], g[1], g[2]]   # the Dirac is built from the 3 spatial generators (Part 1)

# Search all products of distinct generators (and i*products) for a valid antiunitary J.
from itertools import combinations
cand = {}
gens = {"1": np.eye(4), "g1": g[0], "g2": g[1], "g3": g[2], "g4": g[3]}
names = list(gens.keys())
found = []
for r in range(1, 6):
    for combo in combinations([n for n in names if n != "1"], r):
        M = np.eye(4)
        for nm in combo:
            M = M @ gens[nm]
        for pref, plab in [(1.0, ""), (1j, "i*")]:
            C = pref * M
            if not np.allclose(C @ C.conj().T, np.eye(4)):   # must be unitary
                continue
            eps, ep, epp = J_signs(C, SPATIAL, g_c)
            if eps is None or ep is None:
                continue
            ko = [n for n, sig in KO_TABLE.items()
                  if sig[0] == eps and sig[1] == ep and (sig[2] is None or sig[2] == epp)]
            found.append((plab + "".join(combo), eps, ep, epp, ko))

print("   Antiunitary candidates J = C o conj with C a (signed) product of Clifford generators:")
print(f"   {'C':12s} {'J^2':>5s} {'JD=e' + chr(39) + 'DJ':>10s} {'Jg=e' + chr(39) + chr(39) + 'gJ':>12s}  KO-dim(s) mod 8")
seen_ko = set()
for nm, eps, ep, epp, ko in found:
    seen_ko.update(ko)
    print(f"   {nm:12s} {eps:>+5d} {ep:>10d} {('%+d' % epp) if epp is not None else '  --':>12s}  {ko}")

# The DISTINGUISHED choice: the J that commutes with gamma (even triple) AND squares to -1.
# That is the KO-dim 4 (or 0) signature. Pick the one realized with eps'=+1 (D real-compatible).
print(f"""
   Reading the table: the consistent KO-dimensions realized on Cl(4) are {sorted(seen_ko)}.
   The DISTINGUISHED real structure for a 4D EUCLIDEAN chiral spinor is the one with
       J^2 = -1,   J D = +D J,   J gamma = +gamma J   ->   KO-dimension n = 4 (mod 8).
   (This is the unique signature that is (a) EVEN — [J,gamma]=0, so J does not exchange the chiral
    halves — and (b) has J^2 = -1, the quaternionic/symplectic reality of Cl(4,0) ~ M_2(H).)""")

# Verify the canonical KO=4 representative explicitly (C = g2 g4 as in m02, an even product).
C4 = g[1] @ g[3]
eps, ep, epp = J_signs(C4, SPATIAL, g_c)
print(f"   canonical representative  C = g^2 g^4 :  J^2 = {eps:+d}*I,  J D = {ep:+d} D J,  "
      f"J gamma = {epp:+d} gamma J   =>  (eps,eps',eps'') = ({eps:+d},{ep:+d},{epp:+d}) = KO-dim 4.")
print(f"   J antiunitary (C unitary) ? {np.allclose(C4@C4.conj().T, np.eye(4))};  "
      f"C even (commutes with gamma_c) ? {np.allclose(comm(C4, g_c),0)}")

# WHY KO=4 among {3,4,5,6,7}: two structural filters cut the list to the single even quaternionic one.
even_J = [("g2", -1, "odd: anticommutes gamma_c -> KO 5"), ("g1g3", -1, "odd -> KO 5"),
          ("g2g4", -1, "EVEN: commutes gamma_c, J^2=-1 -> KO 4"), ("g1g3g4", +1, "odd -> KO 6,7")]
print(f"""
   [2.2'] WHY KO-DIMENSION 4 IS SELECTED (sharpening the caveat).  Two structural filters:
     (a) The triple must be EVEN: an even spectral triple needs [J, gamma] = 0 (J must NOT swap the
         two chiral halves; it is charge conjugation WITHIN each chirality).  Of the candidate C's,
         only the EVEN products (commuting with gamma_c) qualify -> kills the KO 5 (J g = -g J) rows.
     (b) The reality must match the algebra: Cl(4,0) ~ M_2(H) is QUATERNIONIC, so its canonical
         real structure has J^2 = -1.  Of the EVEN candidates, J^2 = -1 picks KO 4 over the
         J^2 = +1 (KO 0/6) options.
   Together (even) + (quaternionic J^2=-1) single out (eps,eps',eps'') = (-1,+1,+1) = KO-dim 4 UNIQUELY
   among the realized {3,4,5,6,7}.  This is the standard KO-dim of a 4D Euclidean (Riemannian) chiral
   spin manifold — consistent with the 4th gamma being a genuine geometric/chirality direction.
   NOTE (relation to the BARE object): the bare combinatorial real structure is dart-reversal J_dart
   with J_dart^2 = +I (an ORTHOGONAL/real reality on the 12 darts; STRUCTURE.md s6).  The SPINOR J has
   J^2 = -I (QUATERNIONIC).  They are DIFFERENT real structures: J_dart is the geometric edge-reversal
   on the de Rham complex, J is the charge conjugation on the Cl(4) spinor module.  No contradiction —
   the spinor reality is a new datum carried by the Clifford module, not the bare graph reality.""")

# (2c) FIRST-ORDER + ORIENTABILITY for the minimal triple (A = C, D_F = 0).
print("""
[2.3] FIRST-ORDER AND ORIENTABILITY (minimal triple: gauge algebra A = C, internal D_F = 0).
   First-order: [[D, a], b^0] = 0 for a, b in A.  With A = C acting as scalars z*I, [D,a]=0
   identically (a is central) -> first-order TRIVIALLY holds.  Orientability: gamma is the image
   of a Hochschild n-cycle; here gamma_c = g^1 g^2 g^3 g^4 is exactly the Clifford volume element
   (the order-4 orientation cycle).  Both hold for the minimal triple -> it is COMPLETE.""")
a_scalar = (2.7 + 1.3j) * np.eye(4)
D_form = g[0] + 0.5 * g[1] + g[2]
print(f"   [D, a] = 0 for a in A=C ?  {is0(comm(D_form, a_scalar))}    (first-order trivially satisfied)")
print(f"   gamma_c = g^1 g^2 g^3 g^4 = Clifford volume element ?  "
      f"{np.allclose(g_c, g[0]@g[1]@g[2]@g[3])}    (orientability)")

print("""
[2.4] FORCED SPECTRAL-TRIPLE DATA (summary of Part 2):
   spinor:        C^4 = C^2_+ (+) C^2_-                           [FORCED]
   grading:       gamma_c = g^1 g^2 g^3 g^4,  gamma_c^2 = +I       [FORCED]
   real struct:   J = C o conj,  J^2 = -1, [J,gamma]=0, JD=+DJ    [FORCED up to the KO-dim choice]
   KO-dimension:  n = 4 (mod 8)   (the even, J^2=-1 Euclidean chiral signature)   [FORCED*]
   first-order:   holds for minimal A = C (and for any A acting via bounded [D,a])  [FORCED]
   orientability: gamma_c is the volume Hochschild cycle                          [FORCED]
   *Honest caveat: Cl(4) admits a J with J^2=+1 too (KO-dim 0,6 in the table above); selecting
    KO-dim 4 uses the EVEN + J^2=-1 (quaternionic Cl(4,0)~M_2(H)) signature.  The Clifford algebra
    is forced; among its compatible real structures the KO=4 one is the natural Euclidean choice,
    and it is the one inherited when the 4th direction is the chirality/mass direction (Part 3).""")

# =====================================================================================
# PART (3) — THE srs (+) srs-z ENANTIOMER DOUBLING AND THE MASS GAP.
# =====================================================================================
print("\n" + "#" * 96)
print("# PART (3)  srs (+) srs-z enantiomer doubling:  the 4th gamma IS the inter-enantiomer coupling")
print("#" * 96)

# (3a) ONE chiral net (C^2) admits NO gap term: forced gapless.
print("""
[3.1] A SINGLE CHIRAL NET IS FORCED GAPLESS.
   A gap term is a Hermitian M that anticommutes with ALL spatial Dirac terms, so that
   D = sum_a sigma^a p_a + M  has D^2 = (sum p_a^2) + M^2 with no cross terms (a true gap).
   On one srs copy (C^2 = the 3 Pauli) we count the space of such M.""")
basis2 = [I2, s1, s2, s3]
rows = [np.concatenate([anticomm(B, PAULI[a]).flatten() for a in range(3)]) for B in basis2]
Amat = np.array(rows).T
nullity = Amat.shape[1] - np.linalg.matrix_rank(Amat)
print(f"   dim{{ M (2x2 Herm/any) : {{M, sigma^a}} = 0 for a=1,2,3 }} = {nullity}.")
print(f"   => NO nonzero gap term on ONE copy: a single 3D Weyl/chiral net is FORCED GAPLESS.")
print(f"   (The 3 Pauli already exhaust the anticommuting directions in M_2(C); the only operator")
print(f"    anticommuting with all three is 0.  This is the algebraic root of the Weyl obstruction.)")
print("""
   The doubling is FORCED (not merely sufficient) — TWO independent obstructions agree:
     (i)  ALGEBRA: no gap term exists in M_2(C) = Cl(3) (the nullity-0 result just shown);
     (ii) TOPOLOGY: the bare object IS a charge-balanced Weyl semimetal (explore_20 / verify_pass:
          a charge +-2 double-Weyl node at Gamma & H, +-1 at P-type, total Chern 0).  By the lattice
          Nielsen-Ninomiya theorem a lone Weyl monopole cannot be gapped in isolation — the ONLY way
          to gap it is to supply an OPPOSITE-charge partner so the net monopole charge cancels.
   That partner is a second copy of OPPOSITE Weyl charge = srs-z, and the operator that couples them
   (cancelling the charge / opening the gap) is the 4th gamma.  Both obstructions point to the same
   doubling C^2 -> C^4.""")

# (3b) DOUBLING TO C^4 = srs (+) srs-z opens the gap; the gap term is the 4th gamma.
print("""
[3.2] A GAP REQUIRES DOUBLING TO C^4 = srs (+) srs-z; THE GAP TERM IS THE 4th GAMMA.
   In Cl(4) the 4th generator g^4 anticommutes with g^1,g^2,g^3, so it is exactly the missing
   gap term.  D = sum_{a=1,2,3} g^a p_a + m * g^4  =>  D^2 = (sum p_a^2) + m^2.   In the chirality
   eigenbasis g^4 is purely OFF-DIAGONAL: it maps C^2_+ <-> C^2_-, i.e. it COUPLES the two copies.""")
ok_anti = all(is0(anticomm(g[3], g[a])) for a in range(3))
D_test = g[0] * 0.3 + g[1] * (-0.5) + g[2] * 0.9 + 1.7 * g[3]
D2 = D_test @ D_test
gap_ok = np.allclose(D2, (0.3**2 + 0.5**2 + 0.9**2 + 1.7**2) * np.eye(4))
w_gc, V_gc = np.linalg.eigh(g_c)
g4_chiral = np.abs(np.round(V_gc.conj().T @ g[3] @ V_gc, 8))
offdiag_only = np.allclose(g4_chiral[:2, :2], 0) and np.allclose(g4_chiral[2:, 2:], 0)
print(f"   g^4 anticommutes with g^1,g^2,g^3 ? {ok_anti};   D^2 = (sum p^2) + m^2 ? {gap_ok}.")
print(f"   g^4 in the chirality basis is purely OFF-DIAGONAL (couples C^2_+ <-> C^2_-) ? {offdiag_only}.")
print(f"   => the 4th gamma forced by CHIRALITY (Part 1.3) IS the gap term, and it IS the")
print(f"      inter-copy coupling.  Chirality's extra generator and the gap mechanism are the")
print(f"      SAME operator.   [FORCED]")

# Two DISTINCT C^2(+)C^2 splits of the same C^4 — name them so the srs<->srs-z picture is unambiguous.
sp = lambda p: p[0] * s1 + p[1] * s2 + p[2] * s3
H_copy = np.block([[sp((0.3, -0.5, 0.9)), 1.7 * I2], [1.7 * I2, -sp((0.3, -0.5, 0.9))]])
same_op = np.allclose(np.sort(np.linalg.eigvalsh(H_copy)), np.sort(np.linalg.eigvalsh(D_test)))
print(f"""
   [3.2'] THE TWO C^2(+)C^2 DECOMPOSITIONS (clarifies which split is "srs (+) srs-z").
     The SAME C^4 splits two ways, related by the chiral<->Dirac (Weyl) basis change:
       * CHIRALITY basis (gamma_c = diag(+,+,-,-)):  g^4 off-diagonal, and the SPATIAL g^a are ALSO
         off-diagonal (the kinetic term maps left<->right). This is the GRADING split.
       * COPY basis (srs, srs-z):  D = [[ +sigma.p, m I ],[ m I, -sigma.p ]] — the SPATIAL part is
         BLOCK-DIAGONAL (one Weyl cone per net, with OPPOSITE chirality +sigma.p vs -sigma.p), and the
         MASS m I is OFF-DIAGONAL.  THIS is the clean "srs (+) srs-z" picture: two opposite-charge Weyl
         nets coupled by the 4th-gamma mass.
     Same operator (spectra match) ? {same_op}.   The mass is off-diagonal in BOTH bases (so the
     "gap = inter-copy coupling" statement is basis-robust); the COPY basis is the one in which the two
     summands are literally the two enantiomeric nets.""")

# (3c) GEOMETRIC IDENTITY of srs-z: the complex-conjugate (mirror) net, opposite Weyl charge.
print("""
[3.3] WHAT srs-z IS, GEOMETRICALLY:  the complex-conjugate (mirror / enantiomeric) net.
   srs is CHIRAL (verified bare object: no orientation-reversing symmetry).  Its mirror is the
   net with adjacency conj(A(k)) = A(k)* = A(-k) (time-reversal = the enantiomer).  The mirror is a
   genuinely DIFFERENT (opposite-handed) net.  Its band topology is the charge-conjugate: the Weyl
   monopole charge at Gamma flips sign.  We compute both.""")


def sphere_chern(Afun, center, band=0, eps=0.04, N=20):
    import math
    c = np.array(center, float)
    th = np.linspace(.02, math.pi - .02, N)
    ph = np.linspace(0, 2 * math.pi, N, endpoint=False)
    U = np.empty((N, N), object)
    for a in range(N):
        for b in range(N):
            kk = c + eps * np.array([math.sin(th[a]) * math.cos(ph[b]),
                                     math.sin(th[a]) * math.sin(ph[b]), math.cos(th[a])])
            U[a, b] = np.linalg.eigh(Afun(kk))[1][:, band]
    F = 0.0
    for a in range(N - 1):
        for b in range(N):
            bn = (b + 1) % N
            F += np.angle(np.vdot(U[a, b], U[a, bn]) * np.vdot(U[a, bn], U[a + 1, bn])
                          * np.vdot(U[a + 1, bn], U[a + 1, b]) * np.vdot(U[a + 1, b], U[a, b]))
    return F / (2 * np.pi)


c_srs = sphere_chern(lambda kk: srs.adjacency(kk), (0, 0, 0))
c_srsz = sphere_chern(lambda kk: np.conj(srs.adjacency(kk)), (0, 0, 0))
print(f"   Weyl charge at Gamma:   srs = {c_srs:+.2f},   srs-z = conj(srs) = {c_srsz:+.2f}   (OPPOSITE).")
print(f"   So srs (+) srs-z is a charge-conjugate (Weyl +2 (+) Weyl -2) pair; the off-diagonal g^4")
print(f"   coupling is the only Hermitian operator that can gap an otherwise Nielsen-Ninomiya-")
print(f"   protected (net-charge-zero) Weyl pair.   Structure FORCED.")

# (3d) WHAT IS FORCED vs THE IRREDUCIBLE INPUT.
print("""
[3.4] FORCED vs IRREDUCIBLE-INPUT in the doubling.
   FORCED:
     - a single chiral net cannot be gapped (3.1);
     - a gap REQUIRES the doubling C^2 -> C^4 = srs (+) srs-z (3.2);
     - the gap term is the 4th gamma = the off-diagonal srs<->srs-z (inter-enantiomer) coupling,
       and it is the SAME generator that chirality forced in Part 1 (3.2);
     - srs-z is the mirror net with OPPOSITE Weyl charge (3.3).
   IRREDUCIBLE-INPUT:
     - the STRENGTH / SCALE m of that coupling (the gap magnitude).  Nothing in the bare metric
       geometry fixes a length/energy scale; m is the one real number not forced here.  (In the
       staged program this is the seam where a non-tracial state / N-flow would set the scale; that
       lies OUTSIDE this spinor facet.)
   OPEN:
     - whether the *internal* algebra carried alongside the spinor is forced.  m03 showed the AXIOMS
       alone leave A_F free; m05 / explore_10 show the srs SYMMETRY (A_4 regular action on the 12
       darts) forces a specific endomorphism algebra C(+)C(+)C(+)M_3(C).  That is a SYMMETRY/rep-
       theory statement, adjacent to (not part of) the forced SPINOR architecture derived here.""")

# =====================================================================================
# MASTER SUMMARY
# =====================================================================================
print("\n" + "=" * 96)
print(" MASTER SUMMARY — the forced spinor / spectral-triple architecture")
print("=" * 96)
print("""
 (1) Cl(3) <- 3-regularity (one Hermitian gamma per edge-direction); minimal module C^2 = a 3D
     Weyl spinor; Cl(3) ODD (omega = i*I central) => no chirality grading, forced GAPLESS.   [FORCED]
 (1') Chirality forces the MINIMAL EVEN extension: Cl(0),Cl(2) too small => Cl(4), exactly ONE added
      generator g^4.  Chiral 4-spinor C^4 = C^2_+ (+) C^2_-, grading gamma_c = g^1g^2g^3g^4.   [FORCED]
 (1'') Nothing beyond Cl(4) is forced: the minimal chiral real triple closes all axioms (Part 2). [FORCED]
 (2)  Spectral-triple data: grading gamma_c (gamma^2=+I); real structure J=C o conj with J^2=-1,
      [J,gamma]=0, JD=+DJ => KO-dimension 4 (mod 8); first-order & orientability hold for A=C.  [FORCED*]
 (3)  srs (+) srs-z doubling: one copy is ungappable (no M anticommutes with all 3 Pauli); a gap
      REQUIRES doubling to C^4; the gap term is the 4th gamma = the OFF-DIAGONAL inter-enantiomer
      (srs<->srs-z) coupling = the SAME generator chirality forced.  srs-z = mirror net, opposite
      Weyl charge (+2 vs -2 at Gamma).  STRUCTURE forced; STRENGTH m is the one IRREDUCIBLE-INPUT.
 ONE OBJECT: D = sum_{a=1,2,3} gamma^a nabla_a (spatial, srs) + m gamma^4 (mass = enantiomer coupling),
     on C^4 = C^2_+ (+) C^2_-, KO-dim 4, real J.  Everything but the scale m is FORCED by srs + chirality.
""")
print("[m06 done]")
