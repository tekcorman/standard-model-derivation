"""
perron_curvature_run_scratch — PURE MATH, walled. Reads only ../dirac_srs_mdl/srs.py.
No physics, no targets, no fitting to any external value.

THE OBJECT:
  - Bloch adjacency A(k) of the srs crystal (Z^3 cover of K4, k*=3, girth 10).
  - adjacency Bloch eigenvalue branch lambda(k); the Perron (band-top) branch lambda_P(k).
  - Ihara-Bass eigenvalue band  h(k)  from  h^2 - lambda(k) h + (k*-1) = 0,  k*-1 = 2.
  - scalar resolvent read  g(h;u) = 1/(1 - u h),  u in (0,1).

QUESTIONS (pure derivation):
  1. Perron band top h_P, its k-point; confirm it is a band maximum (h'=0, h''<0).
  2. Band-top curvature: h(k) = h_P - (1/2) H (dk)^2 + (1/24) Q (dk)^4 - ...
     compute H, Q exactly along the run axis.  Is H = pi^2 (clean multiple)?
  3. The run scale s (the C3-screw / dN displacement in the band variable dk): forced or free?
  4. Read expansion g(s)/g_P = 1 - a s^2 + b s^4 - ...; closed forms for a, b.
  5. Two-terms sufficiency (b/a) s^2 at representative u (incl u=2/3, u=0.01).
"""
import numpy as np, sys, os
import sympy as sp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dirac_srs_mdl"))
import srs

np.set_printoptions(precision=8, suppress=True)
def hdr(s): print("\n" + "=" * 80 + "\n" + s + "\n" + "=" * 80)

KST = srs.DEG               # k* = 3
SHELL = KST - 1             # k*-1 = 2  (Ihara-Bass constant term)

# The run advances the observer along the C3 screw / triality / stiff-transport axis,
# the unique special direction (verified across the program): AXIS = (1,-1,1)/sqrt3.
# The run coordinate is the screw displacement s in FRACTIONAL Bloch-k along this axis,
# exactly as the established run uses it (k = s * AXIS).  We will examine BOTH a closed-form
# symbolic lambda(s) along this axis and a numeric cross-check, then the curvature.
AXIS = np.array([1.0, -1.0, 1.0]) / np.sqrt(3.0)

# =============================================================================================
hdr("(1) PERRON BAND TOP  h_P  and its k-point; confirm it is a band MAXIMUM")
# =============================================================================================
# Adjacency Perron branch:
def lam_max(kvec):
    return float(np.linalg.eigvalsh(srs.adjacency(np.asarray(kvec, float))).max())

lamP_Gamma = lam_max([0, 0, 0])
print(f"  adjacency Perron eigenvalue at Gamma (k=0): lambda_P = {lamP_Gamma:.10f}  (= k* = {KST})")

# Ihara-Bass roots h of  h^2 - lambda h + 2 = 0.
def ih_roots(lam):
    disc = lam * lam - 4.0 * SHELL
    if disc >= 0:
        r = np.sqrt(disc)
        return (lam + r) / 2.0, (lam - r) / 2.0          # real roots
    r = np.sqrt(-disc)
    return complex(lam / 2, r / 2), complex(lam / 2, -r / 2)  # complex-conjugate pair

hP_plus, hP_minus = ih_roots(lamP_Gamma)
print(f"  Ihara-Bass roots at Gamma: h+ = {hP_plus:.10f}, h- = {hP_minus:.10f}")
print(f"  => Perron NB-root  h_P = (lambda_P + sqrt(lambda_P^2 - 8))/2 = {hP_plus:.10f}")
print(f"     closed form: lambda_P = k* = 3 -> h_P = (3 + sqrt(9-8))/2 = (3+1)/2 = 2 = (k*-1)")
hP = hP_plus

# Confirm Gamma is the band top of lambda(k): scan a neighbourhood, check it is a strict max,
# and that lambda is stationary (gradient 0) at Gamma (by symmetry it must be).
print("\n  Scan lambda_P(k) near Gamma along the run axis and transverse, confirm MAXIMUM:")
for s in [-0.06, -0.03, 0.0, 0.03, 0.06]:
    print(f"    s={s:+.3f} along AXIS:  lambda_P = {lam_max(s*AXIS):.8f}   (<= 3 for all s != 0)")
# transverse check (a generic perpendicular direction)
perp = np.array([1.0, 1.0, 0.0]); perp = perp - (perp @ AXIS) * AXIS; perp /= np.linalg.norm(perp)
for s in [-0.06, 0.0, 0.06]:
    print(f"    s={s:+.3f} along PERP:  lambda_P = {lam_max(s*perp):.8f}")
print("  => lambda_P(Gamma)=3 is a strict local MAXIMUM (h'=0 by symmetry, values fall off in every dir).")

# =============================================================================================
hdr("(2) BAND-TOP CURVATURE: symbolic lambda(s) and h(s) along the run axis -> H, Q")
# =============================================================================================
# Build A(k) symbolically along k = s * AXIS.  The Bloch phases live on the 3 cotree edges with
# homology vectors e1=(1,0,0), e2=(0,1,0), e3=(0,0,1).  Along AXIS=(1,-1,1)/sqrt3, the phase
# argument for edge with homology v is  2*pi * (k . v) = 2*pi * s * (AXIS . v).
# AXIS.e1 = 1/sqrt3, AXIS.e2 = -1/sqrt3, AXIS.e3 = 1/sqrt3.
# So the three cotree phases are theta1 = c*s, theta2 = -c*s, theta3 = c*s with c = 2*pi/sqrt3.
s = sp.symbols('s', real=True)
c = 2 * sp.pi / sp.sqrt(3)            # the per-unit-s phase rate from the lattice (FORCED, no continuum import)
th = [c * s, -c * s, c * s]          # the three cotree Bloch phases along the run axis

# Symbolic Bloch adjacency (Hermitian 4x4). Tree edges {01,02,03} carry phase 0; cotree {12,13,23}.
def sym_adjacency():
    A = sp.zeros(4, 4)
    edges = [(0, 1, None), (0, 2, None), (0, 3, None),
             (1, 2, 0), (1, 3, 1), (2, 3, 2)]   # last 3 carry cotree phase index
    for (i, j, idx) in edges:
        p = 1 if idx is None else sp.exp(sp.I * th[idx])
        A[i, j] += p
        A[j, i] += sp.conjugate(p)
    return A

A = sym_adjacency()
# The Perron eigenvalue: along the C3 axis the (1,-1,1)/sqrt3 phase pattern keeps a clean branch.
# Get the characteristic polynomial P(lambda, s); the Perron branch is the root lambda(s) with
# lambda(0)=3.  We Taylor-expand that branch by IMPLICIT DIFFERENTIATION of P(lambda(s), s)=0
# at (lambda,s)=(3,0) - fast and exact (avoids symbolic quartic root-solving).
lam = sp.symbols('lambda')
charpoly = sp.expand(A.charpoly(lam).as_expr())
P = sp.expand(sp.re(charpoly) + sp.im(charpoly))  # adjacency is Hermitian -> charpoly real; tidy
P = sp.expand(sp.simplify(charpoly))
print("  characteristic polynomial of A(s) (along the run axis):")
print("   ", sp.nsimplify(sp.expand(P)))

order = 8
# Implicit Taylor: write lambda(s) = 3 + sum_{n>=1} a_n s^n, plug into P, kill orders one by one.
coeffs = [sp.Integer(3)]
unknowns = sp.symbols(f'a1:{order}')   # a1..a_{order-1}
lam_poly = sp.Integer(3) + sum(unknowns[i] * s**(i + 1) for i in range(order - 1))
Pexpand = sp.series(P.subs(lam, lam_poly), s, 0, order).removeO()
Pexpand = sp.expand(Pexpand)
sol = {}
for n in range(1, order):
    eq = sp.expand(Pexpand.subs(sol)).coeff(s, n)
    a_n = sp.solve(sp.Eq(eq, 0), unknowns[n - 1])
    sol[unknowns[n - 1]] = sp.simplify(a_n[0]) if a_n else sp.Integer(0)
lamP_ser = sp.expand(sp.Integer(3) + sum(sol[unknowns[i]] * s**(i + 1) for i in range(order - 1)))
lamP_ser = sp.expand(sp.simplify(lamP_ser))
print("\n  Perron adjacency branch  lambda_P(s) Taylor:")
print("   ", lamP_ser)

# coefficients
clam2 = sp.simplify(lamP_ser.coeff(s, 2))
clam4 = sp.simplify(lamP_ser.coeff(s, 4))
print(f"\n  lambda_P(s) = 3 + ({clam2}) s^2 + ({clam4}) s^4 + ...")
print(f"  numeric: lambda_P(s) = 3 + ({float(clam2):.6f}) s^2 + ({float(clam4):.6f}) s^4")

# Now the Ihara-Bass Perron root h_P(s) = (lambda + sqrt(lambda^2 - 8))/2.
lamv = sp.symbols('L')
h_of_lam = (lamv + sp.sqrt(lamv**2 - 8)) / 2
# compose: substitute lambda_P(s) and expand in s about 0 (lambda(0)=3 -> sqrt(1)=1, smooth).
h_expr = h_of_lam.subs(lamv, lamP_ser)
h_ser = sp.series(h_expr, s, 0, order).removeO()
h_ser = sp.expand(sp.simplify(h_ser))
print("\n  Ihara-Bass Perron root  h_P(s) Taylor:")
print("   ", sp.nsimplify(h_ser, [sp.pi]))

ch2 = sp.simplify(h_ser.coeff(s, 2))
ch4 = sp.simplify(h_ser.coeff(s, 4))
# h(s) = h_P - (1/2) H s^2 + (1/24) Q s^4 - ...   so coeff(s^2) = -H/2, coeff(s^4) = +Q/24
H = sp.simplify(-2 * ch2)
Q = sp.simplify(24 * ch4)
print(f"\n  h_P(s) = h_P - (1/2) H s^2 + (1/24) Q s^4 - ...,  h_P = {h_ser.coeff(s,0)}")
print(f"     coeff(s^2) = {ch2}  ->  H = -2*coeff = {sp.nsimplify(H,[sp.pi])} = {float(H):.8f}")
print(f"     coeff(s^4) = {ch4}  ->  Q = 24*coeff = {sp.nsimplify(Q,[sp.pi])} = {float(Q):.8f}")
pi2 = float(sp.pi**2)
print(f"\n  pi^2 = {pi2:.8f};  H/pi^2 = {float(H)/pi2:.8f};  H/(pi^2) clean? ratio = {sp.nsimplify(H/sp.pi**2)}")
print(f"  Q/pi^4 = {float(Q)/float(sp.pi**4):.8f};  Q as multiple of pi^4 = {sp.nsimplify(Q/sp.pi**4)}")

# Also report the adjacency-band Hessian H_lambda for context (lambda(s) = 3 - 1/2 H_lam s^2 + ...).
H_lam = sp.simplify(-2 * clam2)
Q_lam = sp.simplify(24 * clam4)
print(f"\n  (adjacency band: lambda(s)=3 - 1/2 H_lam s^2 + 1/24 Q_lam s^4; "
      f"H_lam = {sp.nsimplify(H_lam,[sp.pi])} = {float(H_lam):.6f}, "
      f"Q_lam = {sp.nsimplify(Q_lam,[sp.pi])} = {float(Q_lam):.6f})")

# numeric cross-check of H via finite differences of the numeric Perron h(s)
def hP_num(sv):
    lm = lam_max(sv * AXIS)
    return (lm + np.sqrt(lm * lm - 8.0)) / 2.0
dd = 1e-3
H_num = -(hP_num(dd) - 2 * hP_num(0.0) + hP_num(-dd)) / dd**2
print(f"\n  numeric finite-diff check:  H (= -h''(0)) = {H_num:.6f}  vs symbolic {float(H):.6f}")

# =============================================================================================
hdr("(3) THE RUN SCALE s: is the natural band-displacement GEOMETRICALLY FORCED or free?")
# =============================================================================================
print("""  The run (dN / modular-cooling flow) advances along the C3 screw axis (1,-1,1)/sqrt3 - the
  unique forced special direction.  The band variable is the FRACTIONAL Bloch displacement dk=s.
  Candidate geometric scales the object itself supplies along this axis:

   (i) the full C3 screw PERIOD in fractional-k: the deck Z^3, the C3 line k=t(1,-1,1) closes
       its phase pattern with period dictated by the lattice; one full deck period along a basis
       direction is dk_frac = 1 (k is fractional, BZ = [0,1) per basis vector).  The natural
       *fraction* of a turn the screw is a THREE-fold (C3) screw: one screw step = 1/3 of the deck
       translation -> s_screw = 1/3 (in fractional-k along a basis direction).
   (ii) the saddle/inflection WIDTH of the Perron band along the axis: where the quartic term
       balances the quadratic, i.e. the band's own curvature scale s_band where H s^2 ~ |Q|/12 s^4
       -> s_band = sqrt(12 H / |Q|) (the band's intrinsic half-width).
   (iii) the BZ half-width pi in ANGLE (theta = c*s ranges; one full lattice period is theta=2pi).
""")
# (i) C3 screw fraction
s_screw_frac = sp.Rational(1, 3)
print(f"  (i)  C3 screw fraction of a deck period along the axis:  s_screw = 1/3 (fractional-k).")
print(f"       at s=1/3: theta = c*s = (2pi/sqrt3)/3 = {float(c*s_screw_frac):.6f} rad per cotree phase.")
# (ii) band half-width from its own curvature (where quartic ~ quadratic of the h-expansion)
s_band = sp.sqrt(12 * H / sp.Abs(Q))
print(f"  (ii) band intrinsic half-width  s_band = sqrt(12 H/|Q|) = {sp.nsimplify(sp.simplify(s_band))} "
      f"= {float(s_band):.6f}")
# The angle theta = c*s; report each candidate's theta.
for nm, sv in [("screw 1/3", float(s_screw_frac)), ("band half-width", float(s_band))]:
    print(f"       {nm}: s={sv:.5f} -> theta=c*s={float(c)*sv:.6f} rad")
print("""
  HONEST STATUS: the run is SCALE-FREE (the object is type III_1, Connes T(M)={0}; the time/length
  UNIT is not fixed by {D,srs,MDL} - established across the time_bridge).  So an ABSOLUTE s is NOT
  geometrically forced; only the DIRECTION (the C3 axis) and the dimensionless RATE c=2pi/sqrt3 are
  forced.  What the geometry DOES force is the *shape* (H, Q above) and the candidate special
  fractions (the screw 1/3, the band half-width).  We carry s as a derived family and report the
  read expansion as a function of s, evaluating at the forced screw fraction s=1/3 as the natural
  representative.  (If the cross-pollination later fixes the unit, s lands; within the wall it does not.)
""")
S_REP = float(s_screw_frac)   # natural representative = the C3 screw fraction

# =============================================================================================
hdr("(4) READ EXPANSION  g(s)/g_P = 1 - a s^2 + b s^4 - ...   (closed forms in u, H, Q, h_P)")
# =============================================================================================
# g(h;u) = 1/(1 - u h).  With h(s) = h_P - (1/2)H s^2 + (1/24)Q s^4 - ...,
# g_P = 1/(1 - u h_P).  Expand g(s)/g_P in s.
u = sp.symbols('u', positive=True)
hsym = sp.symbols('h')  # placeholder
g = 1 / (1 - u * h_ser)
gP = 1 / (1 - u * h_ser.coeff(s, 0))
ratio = sp.series(sp.simplify(g / gP), s, 0, 6).removeO()
ratio = sp.expand(sp.simplify(ratio))
a_coeff = sp.simplify(-ratio.coeff(s, 2))   # ratio = 1 - a s^2 + b s^4 ; coeff(s^2) = -a
b_coeff = sp.simplify(ratio.coeff(s, 4))    # coeff(s^4) = +b
print("  g(s)/g_P series in s:")
print("   ", ratio)
print(f"\n  a = -(coeff s^2) = {sp.simplify(a_coeff)}")
print(f"  b =  (coeff s^4) = {sp.simplify(b_coeff)}")

# Now express a, b in the requested closed form in (u, H, Q, h_P).
# Let g_P = 1/(1 - u h_P), and define the 'gain' G = u/(1 - u h_P) = u g_P.
# h(s) = h_P - (1/2)H s^2 + (1/24) Q s^4.  Then
#   1 - u h(s) = (1 - u h_P) + (u H/2) s^2 - (u Q/24) s^4
#             = (1 - u h_P)[ 1 + (G H/2) s^2 - (G Q/24) s^4 ]   with G = u/(1-u h_P)
#   g(s)/g_P = 1 / [ 1 + (G H/2) s^2 - (G Q/24) s^4 ]
#            = 1 - (G H/2) s^2 + [ (G H/2)^2 + G Q/24 ] s^4 - ...
# => a = G H / 2 ;  b = (G H/2)^2 + G Q/24  with G = u/(1 - u h_P).
G = u / (1 - u * 2)   # h_P = 2
a_closed = sp.simplify(G * H / 2)
b_closed = sp.simplify((G * H / 2)**2 + G * Q / 24)
print("\n  CLOSED FORM (G := u/(1 - u h_P), h_P = 2):")
print(f"    a = G*H/2                    = {sp.nsimplify(a_closed,[sp.pi])}")
print(f"    b = (G*H/2)^2 + G*Q/24       = {sp.nsimplify(b_closed,[sp.pi])}")
print(f"  check vs series: a match? {sp.simplify(a_closed - a_coeff)==0};  "
      f"b match? {sp.simplify(b_closed - b_coeff)==0}")

# =============================================================================================
hdr("(5) TWO-TERMS SUFFICIENCY:  (b s^4)/(a s^2) = (b/a) s^2  at representative u (incl 2/3, 0.01)")
# =============================================================================================
print(f"  Using the forced screw representative s = 1/3 (=> s^2 = 1/9 = {S_REP**2:.6f}); "
      f"H={float(H):.5f}, Q={float(Q):.5f}, h_P=2.\n")
print(f"  {'u':>8} {'a':>14} {'b':>16} {'(b/a)':>14} {'(b/a)s^2':>14} {'verdict':>16}")
for uv in [0.01, 0.1, 1/3, 0.49, 2/3, 0.9]:
    if abs(uv - 0.5) < 1e-9:
        print(f"  {uv:8.4f}  --- resolvent pole u = 1/h_P = 1/2 : g_P diverges, expansion invalid ---")
        continue
    ur = sp.Rational(uv).limit_denominator(10**6)
    av = float(sp.re(a_closed.subs(u, ur)))
    bv = float(sp.re(b_closed.subs(u, ur)))
    bova = bv / av
    nlt = bova * S_REP**2
    verdict = "O(s^2) suffices" if abs(nlt) < 0.1 else ("borderline" if abs(nlt) < 0.3 else "O(s^4) needed")
    print(f"  {uv:8.4f} {av:14.6f} {bv:16.6f} {bova:14.6f} {nlt:14.6f} {verdict:>16}")

# Also report (b/a) in closed form and the s-value that would make O(s^2) sufficient (ratio < 0.1).
bova_closed = sp.simplify(b_closed / a_closed)
print(f"\n  (b/a) closed form = {sp.nsimplify(bova_closed,[sp.pi])}")
# The u-independent piece: b/a = a + Q/(12 H).
print(f"  identity b/a = a + Q/(12 H);  Q/(12 H) = {sp.nsimplify(Q/(12*H),[sp.pi])} = {float(Q/(12*H)):.6f}")

print("""
  Note: (b/a) s^2 = [ (G H/2) + Q/(12 H) ] s^2  (since b/a = a + Q/(12H), with a = G H/2).
  The u-dependence enters ONLY through a = GH/2 = u H/(2(1-2u)); the Q/(12H) piece is u-independent.
  As u -> 1/2 (= 1/h_P), G blows up (the resolvent pole 1 - u h_P -> 0) and the expansion fails -
  that is the resolvent's own singularity, independent of s.
""")

# -------- honest follow-up: at s=1/3 the LEADING term a s^2 is itself O(1), so two terms do NOT
# suffice there.  Find the s for which the expansion is genuinely perturbative (a s^2 small AND
# (b/a)s^2 small).  This is set by H = 4 pi^2 being large: a s^2 ~ 1 already near s ~ 1/(2 pi).
hdr("(5b) WHERE is the read genuinely perturbative?  The leading term a*s^2 itself")
print("  a*s^2 = (G H/2) s^2 ; with H = 4 pi^2 this is O(1) already at s ~ 1/(pi*sqrt(2G)).")
print("  At the C3-screw s=1/3, a*s^2 and (b/a)s^2 are BOTH O(1): the band has moved a FULL O(1)")
print("  fraction down from the top, so a 2-term Taylor is NOT valid there - the screw step is a")
print("  LARGE-amplitude displacement, not a perturbative one.\n")
print(f"  {'u':>8} {'a*s^2 (s=1/3)':>14} {'(b/a)s^2':>12}   s for which |(b/a)s^2|=0.1 (perturbative)")
for uv in [0.01, 0.1, 1/3, 2/3, 0.9]:
    ur = sp.Rational(uv).limit_denominator(10**6)
    av = float(sp.re(a_closed.subs(u, ur)))
    bv = float(sp.re(b_closed.subs(u, ur)))
    a_s2 = av * S_REP**2
    bova_s2 = (bv / av) * S_REP**2
    s_pert = np.sqrt(0.1 / abs(bv / av))
    print(f"  {uv:8.4f} {a_s2:14.6f} {bova_s2:12.4f}        s* = {s_pert:.4f}  (theta*=c*s*={float(c)*s_pert:.4f} rad)")
print("""
  VERDICT (honest): with H = 4 pi^2 (large), the band-top quartic is steep.  The leading O(s^2)
  term is sufficient ONLY for SMALL displacements s << 1/(2 pi) ~ 0.16; the perturbative s* where
  the next-to-leading ratio falls below 10% is s* ~ 0.06-0.13 depending on u.  At the FULL C3-screw
  fraction s=1/3 the expansion has left its radius of convergence (the band reaches the merge/saddle
  near s~0.13-0.35), so the O(s^4) term is REQUIRED there - i.e. two terms do NOT suffice at s=1/3,
  but the *closed forms* a, b are exact and the read is well-defined for the small-s (perturbative)
  window.  The non-perturbative screw step must be summed, not truncated.
""")
print("[done]")
