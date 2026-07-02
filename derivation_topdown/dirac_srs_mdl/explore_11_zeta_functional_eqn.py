"""
explore_11 — the FUNCTIONAL EQUATION of the Ihara zeta, and the Bloch decomposition AS the
Artin-Ihara L-function factorization over the deck group Z^3.  Pure math, walled off.

Three parts, all verified numerically against srs.py (the Z^3 cover of K_4, q = k-1 = 2):

  (1) Functional equation of the finite-K_4 Ihara zeta under  u -> 1/(q u).
  (2) The same functional equation for the per-cell (Bloch / cover) zeta at every fixed k.
  (3) Bloch decomposition  =  Artin-Ihara (Stark-Terras) factorization:
      zeta_cover(u) = prod over characters chi of the chi-twisted L-zeta of the base K_4,
      and the Bloch parameter k IS the character  chi_k(n) = exp(2 pi i k.n).

Math facts cited:
  - Bass's determinant formula for the Ihara zeta (Bass 1992; Hashimoto 1989 for the edge form).
  - The functional equation / completed zeta for a (q+1)-regular graph (Stark-Terras,
    "Zeta functions of finite graphs and coverings," Adv. Math. 121 (1996) 124; Terras,
    "Zeta Functions of Graphs," Cambridge 2010, Thm 10.4).
  - Artin-Ihara L-functions and the factorization of a Galois cover's zeta over the characters
    of the deck group (Stark-Terras, same; Terras Ch. 18-21).  For an abelian cover the
    irreducible reps are the 1-dim characters chi; here G = Z^3 and chi_k is the Bloch phase.
"""
import numpy as np, srs

q = srs.DEG - 1                      # = 2  (the (q+1)=3-regular graph K_4)
n = srs.NV                           # = 4 vertices
m = len(srs.EDGES)                   # = 6 edges
r = m - n + 1                        # = 3  = first Betti number = rank of the cover Z^r
crit = 1/np.sqrt(q)                  # 1/sqrt(2)  self-dual ('central') radius

US = [0.13, 0.31, 0.27+0.11j, 0.5j, 0.2-0.3j, 0.61]   # test points (avoid u=0 and the poles)


# ---------- the known closed-form finite-K_4 Ihara zeta inverse ----------
def zinv_K4(u):
    """zeta_{K_4}(u)^{-1} = (1-u^2)^2 (1+u+2u^2)^3 (1-3u+2u^2)   (Terras 2010)."""
    return (1 - u**2)**2 * (1 + u + 2*u**2)**3 * (1 - 3*u + 2*u**2)


def zeta_K4(u):
    return 1.0 / zinv_K4(u)


# ============================================================================
# (1) FUNCTIONAL EQUATION of the finite-K_4 Ihara zeta
# ============================================================================
print("="*78)
print("(1) Functional equation of the finite-K_4 Ihara zeta  (q = k-1 = %d)" % q)
print("="*78)

# sanity: the closed form equals the Bass determinant at the trivial fiber Gamma
maxd = max(abs(zinv_K4(u) - srs.ihara_zeta_inv(u, (0, 0, 0))) for u in US)
print(f"  closed form  ==  Bass det(I - uA + q u^2 I)(1-u^2)^(m-n) at Gamma   (max |diff| = {maxd:.1e})")
print(f"  Bass exponent (1-u^2)^(m-n) :  m-n = {m-n} = r-1 = {r-1}  (= genus-1 = rank of H_1 minus 1)\n")

# The Stark-Terras completed ('Lambda') zeta.  For a (q+1)-regular graph the canonical,
# fully symmetric completion is
#       Lambda(u) = (1-u^2)^(r-1) (1 - q u^2)^(r-1) u^(r-1) * zeta(u),
# invariant under u -> 1/(q u).  Here all three exponents equal r-1 = 2.
A_, B_, C_ = (r-1), (r-1), (r-1)
def Lambda(u, zeta=zeta_K4):
    return (1-u**2)**A_ * (1-q*u**2)**B_ * u**C_ * zeta(u)

print(f"  Completed zeta (Stark-Terras, symmetric form):")
print(f"      Lambda(u) = (1-u^2)^{A_} (1 - q u^2)^{B_} u^{C_} * zeta(u)        [all exponents = r-1 = {r-1}]")
print(f"      claim:  Lambda(u) = Lambda(1/(q u))   exactly\n")
maxd = 0.0
for u in US:
    ud = 1/(q*u)
    L, Ld = Lambda(u), Lambda(ud)
    maxd = max(maxd, abs(L-Ld))
    print(f"      u = {u!s:>12}:  Lambda(u) = {L: .6f}   Lambda(1/qu) = {Ld: .6f}   |diff| = {abs(L-Ld):.1e}")
print(f"\n  => functional equation VERIFIED  (max |Lambda(u) - Lambda(1/qu)| = {maxd:.1e}).")
print(f"  The map u -> 1/(q u) fixes the circle |u| = 1/sqrt(q) = {crit:.6f}  (the SELF-DUAL /")
print(f"  'central' radius).  This is the SAME circle on which the non-trivial zeros lie")
print(f"  (graph Riemann hypothesis, explore_04): the Ramanujan shell IS the self-dual line.")

# Note: the completion is a one-parameter family.  (1-q u^2)^2 / u^2 is itself invariant
# under u -> 1/(q u), so one may trade (1-qu^2)^2 <-> u^2.  We display the symmetric member.
fam = lambda u: (1-q*u**2)**2 / u**2
print(f"\n  (The completion is unique up to the invariant factor (1-q u^2)^2 / u^2,")
print(f"   itself fixed by u -> 1/(qu):  max|f(u)-f(1/qu)| = "
      f"{max(abs(fam(u)-fam(1/(q*u))) for u in US):.1e}.  We show the symmetric member.)")


# ============================================================================
# (2) the per-cell (Bloch / cover) zeta -- same functional equation at every k
# ============================================================================
print("\n" + "="*78)
print("(2) The Bloch / per-cell cover zeta  zeta(u,k)^-1 = (1-u^2)^2 det(I - uA(k) + 2u^2 I)")
print("="*78)
print("  Each fiber has the SAME (n=4, m=6) combinatorics, so the SAME completion applies:")
print(f"      Lambda(u,k) = (1-u^2)^{A_} (1 - q u^2)^{B_} u^{C_} * zeta(u,k),  invariant under u -> 1/(qu).\n")
assert A_ == B_ == C_ == r-1

def Lambda_k(u, k):
    return Lambda(u, zeta=lambda v: 1.0/srs.ihara_zeta_inv(v, k))

worst = 0.0
for kname, k in [('Gamma', (0, 0, 0)), ('P=(1/4)^3', (.25, .25, .25)),
                 ('H=(1/2)^3', (.5, .5, .5)), ('generic', (.2, .25, .3)),
                 ('random', (.137, .611, .29))]:
    d = max(abs(Lambda_k(u, k) - Lambda_k(1/(q*u), k)) for u in US)
    worst = max(worst, d)
    print(f"      k = {kname:12}: max |Lambda(u,k) - Lambda(1/qu,k)| = {d:.1e}")
print(f"\n  => the functional equation holds at EVERY fiber k  (worst |diff| = {worst:.1e}).")
print("  The fixed circle |u| = 1/sqrt(q) is k-independent: the graph-RH critical circle is")
print("  the self-dual line of the Bloch zeta uniformly across the Brillouin zone.")
print("  (Time reversal acts separately:  A(-k) = conj A(k)  =>  zeta(u,-k) = conj-coeff zeta;")
print("   it relates k to -k, the functional equation relates u to 1/(qu) at fixed k.)")


# ============================================================================
# (3) Bloch decomposition  =  Artin-Ihara L-function factorization over Z^3
# ============================================================================
print("\n" + "="*78)
print("(3) Bloch decomposition  =  Artin-Ihara (Stark-Terras) L-factorization over G = Z^3")
print("="*78)
print("  Deck group of the cover srs -> K_4 is G = Z^3 (the homology lattice).  Its characters")
print("  are  chi_k(n) = exp(2 pi i k.n),  k in [0,1)^3  =  the Bloch/Floquet parameter.")
print("  Claim A:  the Bloch zeta at k IS the chi_k-TWISTED Ihara L-zeta of the base K_4,")
print("            and the twisted Ihara-Bass identity holds:")
print("              L_{K4}(u, chi_k)^-1 = det(I - u B(k))  =  (1-u^2)^2 det(I - uA(k) + 2u^2 I).")
print("            (B(k) = the Hashimoto edge operator with the character phase on each dart.)\n")

maxd = 0.0
for kname, k in [('Gamma', (0, 0, 0)), ('P', (.25, .25, .25)),
                 ('generic', (.2, .25, .3)), ('random', (.137, .611, .29))]:
    Bk = srs.hashimoto(k)                       # twisted edge-adjacency = chi_k phases
    for u in [0.13, 0.27+0.11j, 0.3]:
        edge = np.linalg.det(np.eye(2*m) - u*Bk)            # L(u,chi_k)^-1  (Hashimoto edge zeta)
        bass = srs.ihara_zeta_inv(u, k)                     # (1-u^2)^2 det(I-uA+qu^2)  (vertex/Bass)
        maxd = max(maxd, abs(edge-bass))
    print(f"      k = {kname:8}:  det(I - uB(k)) == (1-u^2)^2 det(I-uA(k)+2u^2 I)  "
          f"(max over u: {max(abs(np.linalg.det(np.eye(2*m)-uu*Bk)-srs.ihara_zeta_inv(uu,k)) for uu in [0.13,0.27+0.11j,0.3]):.1e})")
print(f"\n  => twisted Ihara-Bass VERIFIED (edge form = vertex form), max |diff| = {maxd:.1e}.")
print("     So  ihara_zeta_inv(u,k)  =  L_{K4}(u, chi_k)^-1  exactly.  The trivial character")
print("     k=0 (chi_0 = 1) gives the ordinary base zeta zeta_{K4}.")

print("\n  Claim B (Artin-Ihara factorization):  for a FINITE abelian cover Y -> K_4 with deck")
print("  group G = (Z_N)^3, the cover's own Ihara zeta factors over the N^3 characters:")
print("      zeta_Y(u)^-1  =  prod_{chi in G^}  L_{K4}(u, chi)^-1")
print("                    =  prod_{k in (1/N)Z^3 / Z^3}  ihara_zeta_inv(u, k).")
print("  We BUILD the cover graph Y = (Z_N)^3 x K_4 explicitly and compare its Bass zeta to")
print("  the product of the N^3 Bloch fibers:\n")

def build_cover_adjacency(N):
    """Explicit (Z_N)^3 cover of K_4: 4 N^3 vertices, Bass on the real adjacency."""
    sites = [(a, b, c) for a in range(N) for b in range(N) for c in range(N)]
    idx = {s: i for i, s in enumerate(sites)}
    NVc = len(sites) * 4
    A = np.zeros((NVc, NVc))
    for (i, j, v) in srs.EDGES:
        for s in sites:
            sc = tuple((np.array(s) + np.array(v)) % N)
            a_, b_ = idx[s]*4 + i, idx[sc]*4 + j
            A[a_, b_] += 1; A[b_, a_] += 1
    return A

for N in [2, 3]:
    A = build_cover_adjacency(N)
    NVc = A.shape[0]; Ec = int(round(A.sum()/2))
    def zinv_Y(u):
        return (1-u**2)**(Ec-NVc) * np.linalg.det(np.eye(NVc) - u*A + q*u**2*np.eye(NVc))
    def prod_L(u):
        p = 1.0 + 0j
        for a in range(N):
            for b in range(N):
                for c in range(N):
                    p *= srs.ihara_zeta_inv(u, (a/N, b/N, c/N))
        return p
    rels = []
    for u in [0.13, 0.21+0.07j]:
        zy, pl = zinv_Y(u), prod_L(u)
        rels.append(abs(zy-pl)/abs(zy))
        print(f"      N={N}  (Y: {NVc} verts, {Ec} edges,  G = (Z_{N})^3, |G| = {N**3}):  "
              f"u={u!s:>10}  zeta_Y^-1 = {zy: .4e}   prod_chi L^-1 = {pl: .4e}   rel.diff {abs(zy-pl)/abs(zy):.1e}")
    print(f"        => factorization VERIFIED for N={N}  (max rel.diff = {max(rels):.1e})\n")

print("  => Bloch decomposition over the Brillouin zone IS the Artin-Ihara factorization of the")
print("     cover's Ihara zeta over the characters of the deck group Z^3.  The infinite cover is")
print("     the N -> infinity limit: a continuous 'product' (log zeta = integral over the BZ of")
print("     log L(u, chi_k) dk).  Each Bloch fiber is one Artin L-function; the spectrum lives")
print("     character-by-character, exactly as Stark-Terras predicts for a Galois cover.")

print("\n" + "="*78)
print("SUMMARY")
print("="*78)
print(f"  (1) Functional equation:  Lambda(u) = (1-u^2)^2 (1-2u^2)^2 u^2 zeta(u) = Lambda(1/(2u)).")
print(f"      All three exponents = r-1 = 2, r = b_1 = 3.  Self-dual radius 1/sqrt(2).")
print(f"  (2) The Bloch/cover zeta obeys the same equation at every fiber k; the critical circle")
print(f"      |u|=1/sqrt(2) is k-independent and coincides with the graph-RH / Ramanujan shell.")
print(f"  (3) ihara_zeta_inv(u,k) = L_{{K4}}(u, chi_k)^-1 (twisted Ihara-Bass), and the finite-cover")
print(f"      zeta factors as prod over characters: Bloch decomposition = Artin-Ihara factorization.")
print(f"  Facts: Bass (1992); Hashimoto (1989); Stark-Terras (Adv. Math. 1996); Terras (2010).")
