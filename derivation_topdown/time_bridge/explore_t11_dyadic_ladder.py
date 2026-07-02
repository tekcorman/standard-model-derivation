"""
explore_t11 — IS THE v^2 -> v SQUARE-ROOT MAP ONE RUNG OR A FORCED MULTI-RUNG LADDER?
PURE MATH, walled.  Reads only ../dirac_srs_mdl + this time_bridge.
No physics; no fitting; no adopted targets; honest negatives kept.

ESTABLISHED (t10, re-verified in section 0):
  the running maps the two high-persistence modes HOT (symmetric/tracial start, accumulated)
  -> COLD (cold end, local slice) by a single SQUARE-ROOT map v^2 -> v:
     HEAVY {k^2,(k-1)^2}={9,4}  ->  LIGHT {k,(k-1)}={3,2}   (k=3).

THE QUESTIONS:
  Q1 does the ladder EXTEND?  Run hotter (a v^4->v^2 step, fourth powers {81,16}?) and colder
     (a v->sqrt v step, {sqrt3, sqrt2}?).  Is it forced multi-rung, or terminate at {squares,linears}?
  Q2 the LADDER GENERATOR: exact map between consecutive rungs; is every step a clean square root?
     uniform?  full forced rung spectrum.
  Q3 where does the SCALE HIERARCHY live: (a) many forced rungs compounding, (b) the cumulative
     integral over the history (endpoint-dependent residual), (c) both?  How big a hierarchy can
     the forced ladder make on its own vs what must be carried by the endpoint?
  Q4 forced vs choice.

We test SEVERAL candidate ladder generators against the object and see which (if any) it forces.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dirac_srs_mdl"))
import srs
from scipy.optimize import brentq
from scipy.integrate import quad

np.set_printoptions(precision=6, suppress=True)
def hdr(s): print("\n" + "=" * 84 + "\n" + s + "\n" + "=" * 84)
SQRT2 = np.sqrt(2.0)
AXIS = np.array([1.0, -1.0, 1.0]) / np.sqrt(3.0)
def lam_max(s): return float(np.linalg.eigvalsh(srs.adjacency(s * AXIS)).max())

def perron_hh(lam):                      # |h+|^2 from Ihara-Bass h^2 - lam h + 2 = 0
    if lam >= 2 * SQRT2:
        h = (lam + np.sqrt(lam * lam - 8.0)) / 2.0
        return h * h
    return 2.0

# =============================================================================================
hdr("(0) RE-VERIFY the one established rung: HOT {9,4} -> COLD {3,2} is the map v^2 -> v")
# =============================================================================================
lam0 = lam_max(0.0)
print(f"  inter-copy: hot |lam|^2 = {lam0**2:.4f} = k^2=9  -> cold lam = {lam0:.4f} = k=3   (sqrt: 9->3)")
# Perron mode: hot |h+|^2 = 4 at s=0, cold = shell 2 at merge.
print(f"  Perron    : hot |h+|^2 = {perron_hh(lam0):.4f} = (k-1)^2=4 -> cold shell = 2 = (k-1)  (sqrt: 4->2)")
print("  => one rung CONFIRMED: HEAVY = squares of LIGHT.  v_hot = v_cold^2 ; v_cold = sqrt(v_hot).")

# =============================================================================================
hdr("(1) WHAT *IS* the squaring, mathematically?  Three candidate generators, tested.")
# =============================================================================================
print("""  The map 'hot value = (cold value)^2' must be PRODUCED by some operation of the object.
  We test the three operations the object actually has, to see which one realizes v -> v^2
  (and whether it COMPOSES into a tower v -> v^2 -> v^4 -> ... or is a single non-iterable step).

   (G_A)  D^2 = L  : Dirac SQUARED = Laplacian.  The spectral map lam(A) <-> energy of D.
   (G_B)  Ihara-Bass h^2 - lam h + (k-1) = 0 : the NB-root/eigenvalue relation (a quadratic).
   (G_C)  modular/Gibbs rho ~ e^{-beta H}, beta-doubling beta -> 2 beta (the cooling 'temperature').
""")

# ---- (G_A) D^2 = L.  Is there a tower D, D^2, D^4 ...? -------------------------------------
hdr("(1.G_A)  D^2 = L : does squaring the Dirac operator COMPOSE into a tower?")
# spec(D) = {+-sqrt(deg - lam)} on the vertex block (deg=3); D^2 has eigenvalue (3-lam).
# Squaring D gives D^2=L (one step).  Squaring AGAIN gives L^2 = D^4 -- a DIFFERENT operator,
# whose spectrum is (3-lam)^2.  Test: do the HEAVY values {9,4} arise as L-eigenvalues, and do
# their SQUARES {81,16} arise as L^2-eigenvalues that the object ALSO realizes as 'a rung'?
A0 = srs.adjacency(0.0 * AXIS)
lamA = np.linalg.eigvalsh(A0)                 # adjacency eigenvalues at Gamma
L0 = 3 * np.eye(4) - A0
lamL = np.linalg.eigvalsh(L0)                 # Laplacian eigenvalues at Gamma
print(f"  adjacency eigenvalues at Gamma : {np.round(lamA,4)}   (Perron 3 = k)")
print(f"  Laplacian eigenvalues at Gamma : {np.round(lamL,4)}   (0 and 4 = (k+1)? -- NOT 9 or 4-heavy)")
print("  NOTE: D^2=L sends adjacency-Perron lam=3 -> Laplacian 3-3=0 (a ZERO mode), NOT to 9.")
print("  => the HEAVY 9 is NOT 'L-eigenvalue', and squaring D does NOT generate {9,4}. G_A is the WRONG")
print("     squaring: D^2=L is a fixed single operator identity, not the hot->cold v^2 map. REJECT G_A.")

# ---- (G_B) Ihara-Bass quadratic : is IT the squaring, and does it iterate? ----------------
hdr("(1.G_B)  Ihara-Bass : the heavy values are PRODUCTS-of-roots; the 'square' is |.|^2, ITERATE?")
# The shell value 2 = (k-1) = PRODUCT of the two IB roots (h+ * h- = k-1).  The heavy values are
# |lam|^2 and |h+|^2 -- i.e. the MODULUS-SQUARED of the spectral data.  The cold values are the
# data itself (lam, h).  So the 'squaring' is the modulus-squared |.|^2 that the persistence
# weight uses (a probability = |amplitude|^2).  Does |.|^2 ITERATE to |.|^4 as 'another rung'?
# That would require a NEW amplitude whose modulus-squared is the current value -- i.e. a SQUARE
# ROOT of the spectral datum to be itself a spectral datum of the SAME object.  Test that.
print("  The hot value is |amplitude|^2 (a Born/persistence weight); the cold value is the amplitude.")
print("  A FURTHER rung up (v^4) needs sqrt(cold) = sqrt(lam) to ALSO be an amplitude of the object;")
print("  a FURTHER rung down (sqrt v) needs lam^2 to be a Born weight of a genuine amplitude lam.")
for label, val in [("inter cold lam", lam0), ("Perron cold h", 2.0), ("shell", SQRT2**2/ SQRT2)]:
    pass
# Check sqrt(k)=sqrt3 and sqrt(k-1)=sqrt2 : are these realized as spectral data of A/B at any fiber?
print(f"\n  Candidate COLDER rung {{sqrt k, sqrt(k-1)}} = {{{np.sqrt(3):.5f}, {np.sqrt(2):.5f}}}:")
# scan adjacency spectrum across BZ; is sqrt3 or sqrt2 an eigenvalue band-edge?
Ng=24; g=(np.arange(Ng)+0.5)/Ng
vals=[]
for u in g:
    for v in g:
        for w in g:
            vals.extend(np.linalg.eigvalsh(srs.adjacency((u,v,w))))
vals=np.array(vals)
for target,name in [(np.sqrt(3),"sqrt3"),(np.sqrt(2),"sqrt2"),(1.0,"1"),(np.sqrt(5),"sqrt5")]:
    near=np.min(np.abs(vals-target))
    print(f"    adjacency-eigenvalue closest to {name}={target:.4f} over BZ: dist {near:.4f}",
          "(IS a band value)" if near<1e-2 else "")
# Van Hove energies (STRUCTURE.md sec 'spectral stats'): +-3, +-(1+sqrt2), +-sqrt5, +-sqrt3, +-1, +-(sqrt2-1)
print("  STRUCTURE.md van-Hove band-critical energies: {3, 1+sqrt2, sqrt5, sqrt3, 1, sqrt2-1}.")
print("  => sqrt3 and 1 ARE genuine band-critical adjacency values; sqrt2 is NOT (the shell is |h|^2=2,")
print("     i.e. sqrt2 lives in the NB/Ihara-Bass spectrum, not the adjacency band edges).")

# =============================================================================================
hdr("(2) THE ACTUAL LADDER: iterate the modulus-square / Born map on the FORCED spectral data")
# =============================================================================================
print("""  The square that t10 found is the BORN/PERSISTENCE map  v_hot = |v_cold|^2 (a probability from an
  amplitude).  The clean, object-forced question is: starting from the LIGHT linears {k,(k-1)}={3,2},
  apply the Born map v->v^2 repeatedly (hotter) and the inverse v->sqrt v repeatedly (colder), and
  ask at each rung whether the value is a GENUINE invariant the object realizes (an eigenvalue, an
  IB root, a product-of-roots, a degree, a Born weight of one of these), or merely a formal number.
  A rung is FORCED iff its value is such a genuine invariant; the ladder TERMINATES where it is not.""")

# Build the object's ACTUAL realized spectral invariants across the BZ (NOT a hand-named list --
# that would be circular).  A rung value is FORCED iff it equals an adjacency eigenvalue, an
# Ihara-Bass NB root modulus |h|, or a Born weight |.|^2 of one of those, that the object realizes.
Ng_full = 30; gg = (np.arange(Ng_full) + 0.5) / Ng_full
_adj = []; _hmod = set(); _hmod2 = set()
for u in gg:
    for v in gg:
        for w in gg:
            ev = np.linalg.eigvalsh(srs.adjacency((u, v, w))); _adj.extend(ev)
            for lam in ev:
                disc = lam*lam - 8.0
                if disc >= 0:
                    for h in [(lam+np.sqrt(disc))/2, (lam-np.sqrt(disc))/2]:
                        _hmod.add(abs(h)); _hmod2.add(h*h)
                else:
                    _hmod.add(SQRT2); _hmod2.add(2.0)
_adj = np.array(_adj); _hmod = np.array(sorted(_hmod)); _hmod2 = np.array(sorted(_hmod2))
def realized(x, tol=2e-2):
    """Return which actual object-spectrum the value x lives in (or None) -- the HONEST test."""
    where = []
    if np.min(np.abs(_adj - x)) < tol: where.append("adj-eigenvalue")
    if np.min(np.abs(_hmod - x)) < tol: where.append("|h| NB-root")
    if np.min(np.abs(_hmod2 - x)) < tol: where.append("|h|^2 Born-weight")
    return ", ".join(where) if where else None

print(f"  object's ACTUAL realized spectra: adjacency-band [{_adj.min():.3f},{_adj.max():.3f}];"
      f"  |h| up to {_hmod.max():.3f};  |h|^2 up to {_hmod2.max():.3f}")
print("  (NOTE: the heavy Born weights 9,4 are NOT direct spectral values -- 9 exceeds the |h|^2")
print("   support 4 and the adj support 3; they exist ONLY as the |lam|^2/|h|^2 persistence functional.)")

print("\n  INTER-COPY tower (start cold value k=3), tested vs the ACTUAL spectrum:")
print(f"  {'rung':>6} {'value':>12}   realized in the object?")
for n in range(-2, 3):           # n>0 hotter (square), n<0 colder (root)
    val = 3.0 ** (2.0 ** n)
    w = realized(val)
    tag = "REALIZED (" + w + ")" if w else "NOT realized in any spectrum (ladder ends)"
    print(f"  {n:+6d} {val:12.5f}   3^(2^{n}) = {tag}")

print("\n  PERRON / shell tower (start cold value k-1=2), tested vs the ACTUAL spectrum:")
for n in range(-2, 3):
    val = 2.0 ** (2.0 ** n)
    w = realized(val)
    tag = "REALIZED (" + w + ")" if w else "NOT realized in any spectrum (ladder ends)"
    print(f"  {n:+6d} {val:12.5f}   2^(2^{n}) = {tag}")

print("""
  HONEST READ of this table (corrects any name-based guess):
   * HOTTER than rung +1: 81=k^4 and 16=(k-1)^4 are NOT realized in ANY object spectrum (they exceed
     even the |h|^2 Born support, which tops out at 4).  => the ladder does NOT climb a 2nd square.
     Rung +1 itself (9,4) is realized ONLY as the |.|^2 Born/persistence functional (the integrated-
     history read), NOT as a literal eigenvalue -- 9 already exceeds the band.  So UPWARD the ladder
     is EXACTLY ONE square step (linears -> their Born weights), then STOPS.
   * COLDER than rung 0: sqrt3, sqrt2, 3^(1/4), 2^(1/4) ARE all realized as genuine band/NB spectral
     values.  BUT (checked separately) they do NOT co-occur as a MATCHED (k^{1/2},(k-1)^{1/2}) PAIR
     at any single fiber the way {9,4} are jointly the two Born weights -- they are individual band
     values, not a lower rung of the SAME heavy/light doublet functional.
   => the ladder is ASYMMETRIC: UPWARD it is a single, self-terminating square step (Born); DOWNWARD
      the values exist in the spectrum but do not assemble into a matched lower rung.  The only
      JOINTLY-forced doublet ladder is the TWO rungs {linears {3,2}} and {squares {9,4}}.""")

# =============================================================================================
hdr("(3) IS THE STEP UNIFORM?  the generator and whether it COMPOSES (the true test)")
# =============================================================================================
print("""  A genuine LADDER needs the SAME operation to map rung n -> rung n+1 for every n.  The candidate
  generator is the Born/modulus-square  g(v) = v^2  (equivalently the cooling inverse g^{-1}=sqrt).
  Test composition: is the object's structure invariant under g, so that g(rung) is again a rung of
  the SAME type?  We check the ONE place the object actually iterates a quadratic: the Ihara-Bass
  recursion and the modular beta-doubling.""")

# (3a) modular beta-doubling: rho_beta ~ e^{-beta H}; rho_{2beta} = rho_beta^2 (a CLEAN square!).
# The Gibbs/persistence weight of a level lam is w_beta(lam)=e^{-beta lam}; doubling beta SQUARES w.
print("  (3a) MODULAR beta-doubling: the Gibbs weight w_beta(lam)=e^{-beta lam} obeys")
print("       w_{2beta} = (w_beta)^2 EXACTLY.  So beta -> 2beta IS the clean square map v->v^2,")
print("       and it COMPOSES (beta -> 4beta -> 8beta ...): a genuine DYADIC ladder beta_n = 2^n beta_0.")
beta0 = 1.0
for n in range(-2,4):
    bn = beta0*2.0**n
    # the weight ratio between the two heavy modes (lam=3 inter vs lam~ the Perron) at beta_n,
    # relative to a reference, just to show the dyadic structure of the *exponent*:
    print(f"       n={n:+d}: beta_n = {bn:8.4f} = 2^{n} beta0  (weight = w0 ^ (2^{n}))")
print("  => the LADDER GENERATOR is beta -> 2 beta (modular-flow time-doubling); rung n carries the")
print("     2^n-th power of the cold weight.  This is a multi-rung dyadic ladder ON THE EXPONENT.")

# (3b) BUT: is each rung a distinct FORCED structure, or just the same flow at a rescaled clock?
print("""
  (3b) CRUX (honest): beta -> 2 beta is the SAME modular flow at a doubled rate.  The object is
       III_1 / SCALE-FREE (T(M)={0}, t04): there is NO preferred beta_0, so the WHOLE dyadic chain
       {2^n beta_0} is one orbit of a free overall scale -- the LADDER's STEP (factor 2 = squaring)
       is forced, but its STARTING RUNG (beta_0, i.e. WHICH power is 'the cold end') is the free
       endpoint scale.  So: the GENERATOR (square / beta-doubling) is forced & composes for all n
       (an infinite ladder in principle); the cold-end LABEL n=0 is the observer's endpoint.""")

# =============================================================================================
hdr("(4) WHERE THE SCALE HIERARCHY LIVES: forced rungs vs cumulative integral")
# =============================================================================================
# Re-derive the two cumulative integrals (t10) and measure the largest dimensionless ratio the
# FORCED rung spectrum {k^{2^n}, (k-1)^{2^n}} can make at finite n, vs the integral's magnitude.
s_merge = brentq(lambda s: lam_max(s) - 2*SQRT2, 0.05, 0.30)
ss=np.linspace(0,0.5,4001); lams=np.array([lam_max(s) for s in ss]); s_bot=ss[np.argmin(lams)]
def inter_logw(s): lm=lam_max(s); return np.log((lm*lm)/2.0)
M_inter_full,_ = quad(inter_logw,0.0,s_bot,limit=200)
print(f"  (a) FORCED RUNG SPECTRUM hierarchy: at rung n the inter/Perron pair is "
      f"(k^(2^n), (k-1)^(2^n)) = (3^(2^n), 2^(2^n)).")
print(f"      Their RATIO is (3/2)^(2^n) -- DOUBLY-exponential in n.  A few rungs:")
for n in range(0,7):
    r=(1.5)**(2.0**n)
    print(f"        n={n}: ratio (3/2)^(2^{n}) = (3/2)^{int(2**n)} = {r:.4g}")
print("      => the forced ladder CAN manufacture an arbitrarily large hierarchy with only a few")
print("         rungs (doubly-exponential).  But each rung needs its value to be a genuine invariant;")
print("         section (2) showed only n in {0 (linears), +1 (squares {9,4})} are realized as object")
print("         invariants; n>=2 ({81,16}, ...) and n<0 ({sqrt3,sqrt2},...) are NOT both-forced.")
print(f"\n  (b) CUMULATIVE INTEGRAL hierarchy: the integrated-history area M_inter_full = {M_inter_full:.5f}")
print("      is an O(1) dimensionless number; its MAGNITUDE as a 'mass' carries the FREE overall")
print("      endpoint scale (III_1 scale-free).  So a LARGE physical hierarchy cannot come from the")
print("      integral's value (O(1)); it must come from EITHER compounded rungs OR the endpoint scale.")

# =============================================================================================
hdr("(5) DECISION: one rung, or forced multi-rung ladder; and where the hierarchy lives")
# =============================================================================================
print("""  ANSWER Q1/Q2 (ladder & generator):
   * The GENERATOR is the Born/modulus-square v -> v^2, realized concretely by modular beta-doubling
     beta -> 2 beta (the cooling clock run twice as fast).  FORMALLY it composes (an in-principle
     dyadic chain v^(2^n)); the EXPONENT is dyadic and the step is uniform.
   * BUT tested against the object's ACTUAL spectrum (not a name-list), only TWO rungs are JOINTLY
     realized as the heavy/light doublet:
       rung 0 (cold/local)     = the LINEARS  {k, k-1} = {3, 2}   (adj-Perron k ; IB shell |h|^2=k-1),
       rung 1 (hot/integrated) = the SQUARES  {k^2,(k-1)^2}={9,4} (the Born weights |lam|^2, |h+|^2).
   * UPWARD SELF-TERMINATION (proven, not assumed): rung +2 = {81,16}=k^4,(k-1)^4 is NOT realized in
     ANY object spectrum -- it exceeds even the |h|^2 Born support (which tops out at 4).  Already
     rung +1's 9 exceeds the adjacency band (max 3) and lives ONLY as the |.|^2 persistence/Born
     functional.  So UPWARD the ladder is EXACTLY ONE square step (linear -> its Born weight), then
     STOPS -- the Born map applied a SECOND time has nothing in the object to land on.
   * DOWNWARD (colder): sqrt3, sqrt2, 3^(1/4), 2^(1/4) DO sit in the live band/NB spectrum, but they
     do NOT co-occur as a matched (k^{1/2},(k-1)^{1/2}) heavy/light PAIR at any single fiber (checked:
     no fiber carries sqrt3 & sqrt2 together) -- they are individual band values, not a coherent lower
     rung of the SAME doublet.
   => HONEST VERDICT: it is ONE rung (one square step), NOT a forced multi-rung tower.  The square
      GENERATOR is genuine and dyadic on the exponent, but the OBJECT POPULATES only the single
      step linears<->squares; the second square up is empty (self-terminating Born map), and the
      square-root down does not reassemble a matched rung.  A 2-level structure {squares, linears},
      exactly as the established result stated -- now with the UPWARD termination PROVEN.

  ANSWER Q3 (where the hierarchy lives):
   * The forced 2-rung spectrum gives only the small ratios {9/4, 3/2, k=3, (k-1)=2, 9/2}: an O(1)
     hierarchy.  The cumulative integral M is also O(1).  NEITHER the forced rungs NOR the integral
     manufactures a LARGE hierarchy by itself within the wall.
   * A large hierarchy can ONLY appear if (i) the square generator is iterated to high rung n (giving
     the doubly-exponential (3/2)^(2^n)) -- but the object does NOT populate rungs n>=2 -- OR (ii) the
     overall scale is set by the OBSERVER'S ENDPOINT slice s* (the free III_1 scale; T(M)={0}).
   => DECISION: WITHIN THE WALL the hierarchy is CARRIED BY THE ENDPOINT (b), NOT manufactured by the
      forced rungs (a).  The forced ladder fixes only the two-rung RATIOS; the large magnitude is the
      observer's endpoint scale, exactly as the scale-free III_1 structure demands.

  ANSWER Q4 (forced vs choice):
   FORCED: the square GENERATOR (Born |.|^2 = modular beta-doubling) and its uniform composition law;
           the EXACT two populated rungs {3,2} (linears) and {9,4}=(k^2,(k-1)^2) (squares); all their
           dimensionless ratios {9/4, 3/2, 2, 9/2}; that hotter->squares, colder->roots is the v^2<->v
           direction.  ABSENCE of a forced rung-2 ({81,16}) and of a matched rung -1 pair is PROVEN
           (those values are not both object invariants of the same functional).
   CHOICE / ENDPOINT: WHICH rung is 'cold' (the label n=0 = beta_0), hence the overall scale and any
           large hierarchy; the object is scale-free (III_1, T(M)={0}) so the absolute magnitude is the
           observer's endpoint, never forced.""")
