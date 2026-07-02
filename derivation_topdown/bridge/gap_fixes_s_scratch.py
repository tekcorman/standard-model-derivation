"""
gap_fixes_s_scratch.py  --  SEALED derivation (no physics, no targets, no tuning).
Reads ONLY ../dirac_srs_mdl/srs.py and the in-wall interaction/ machinery.

QUESTION (pure math):
  The bare object leaves the run-displacement `s` off the Perron band top a FREE
  type-III_1 scale.  h(s) = h_P - (1/2)(4 pi^2) s^2 + ...  (curvature H = 4 pi^2 exact).
  The interaction sector (i01) generates a scale by dimensional transmutation:
        1 = g * I(m),   I(m) = int DOS(eps)/(2 sqrt(eps^2+m^2)) d eps,
        m ~ W * exp(-1/(g N0)).
  i04 corrects i01/i02: the coupling is NOT free -- it is spectral data of the forced
  internal Dirac D_F (algebra C[A4] = C^3 (+) M3; spectrum on the Ramanujan shell |h|^2=2).

  DOES the gap equation FIX s, and what is its value?
  Test: is s the dynamically-generated scale-ratio  s = m/W = exp(-1/(g N0)),
  or (the analogous self-consistent quantity for the displacement off the band top)?

  Every step: forced vs chosen, stated explicitly.  No tuning, no target value.
"""
import numpy as np, sys, os
import sympy as sp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dirac_srs_mdl"))
import srs

np.set_printoptions(precision=8, suppress=True)
def hdr(t): print("\n" + "=" * 84 + "\n" + t + "\n" + "=" * 84)

KST = srs.DEG        # k* = 3   FORCED (srs.py:12)
SHELL = KST - 1      # k*-1 = 2 FORCED (Ihara-Bass constant term; Ramanujan shell)

# =====================================================================================
hdr("(0) THE FORCED BAND DATA  (W, DOS, N0)  -- exactly as interaction/i01 builds them")
# =====================================================================================
# Reproduce the i01 band DOS verbatim (explore_i01_gap_equation.py:14-21), all FORCED:
N = 22; idx = (np.arange(N) + 0.5) / N
eps = np.concatenate([np.linalg.eigvalsh(srs.adjacency((a, b, c)))
                      for a in idx for b in idx for c in idx])
eps = eps - np.mean(eps)
W = eps.max() - eps.min()
hist, edges = np.histogram(eps, bins=400, density=True)
ec = 0.5 * (edges[:-1] + edges[1:]); de = edges[1] - edges[0]
N0_center = hist[np.argmin(np.abs(ec))]
print(f"  srs adjacency band  (FORCED, from srs.adjacency over the BZ):")
print(f"    bandwidth      W        = {W:.6f}   (= eps_max - eps_min; geometric, FORCED)")
print(f"    DOS at band CENTER N(0) = {N0_center:.6f}   (i01's N0; sampling/binning dependent)")
print(f"    Perron band TOP eps_top = {eps.max():.6f}   (= k* shifted by the band mean)")
print(f"    Perron eigenvalue at Gamma (unshifted) = {float(np.linalg.eigvalsh(srs.adjacency([0,0,0])).max()):.6f}  (= k* = {KST}, FORCED)")

# =====================================================================================
hdr("(1) THE GAP EQUATION  1 = g I(m)  AND THE TRANSMUTATION SCALE  m/W = exp(-1/(g N0))")
# =====================================================================================
I = lambda m: float(np.sum(hist * de * 0.5 / np.sqrt(ec**2 + m**2)))  # i01 kernel (FORCED form)
def solve_gap(g):
    if g * I(1e-11) < 1.0: return 0.0
    lo, hi = 1e-12, 8.0
    for _ in range(300):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if g * I(mid) > 1.0 else (lo, mid)
    return np.sqrt(lo * hi)

print("  Reproduce i01 transmutation (g free here, pinned in step 3):")
print(f"    {'g':>6} {'m=gap':>14} {'m/W':>14} {'exp(-1/(g N0))':>16}")
gs = np.array([1.3, 1.7, 2.2, 3.0])
ms = np.array([solve_gap(g) for g in gs])
for g, m in zip(gs, ms):
    print(f"    {g:6.2f} {m:14.6e} {m/W:14.6e} {W*np.exp(-1/(g*N0_center))/W:16.6e}")
slope, icpt = np.polyfit(-1.0/gs, np.log(ms/W), 1)
print(f"  fit ln(m/W) = {slope:.4f}*(-1/g) + {icpt:.4f};  slope vs 1/N0 = {1/N0_center:.4f}")
print(f"  => transmutation confirmed: the OVERALL leading scale m sits exp-below W.")

# =====================================================================================
hdr("(2) THE LINK  s <-> band displacement.  Is s the SAME object as m/W?")
# =====================================================================================
# Established (perron_curvature_run_scratch): h(s) = h_P - (1/2) H s^2,  H = 4 pi^2 exact,
# h_P = 2.  So the displacement of the NB-root off the band top is
#     delta_h(s) = h_P - h(s) = (1/2) H s^2 = 2 pi^2 s^2.
# The transmutation gap m is a displacement of the FERMION POLE below the band EDGE (W),
# i.e. a fraction  m/W  of the bandwidth.  These are TWO DIFFERENT displacements:
#   - s:    fractional Bloch-k displacement of the eigen-branch off the Perron top (a CRYSTAL-
#           MOMENTUM coordinate; dimensionless, enters the read g=1/(1-u h) via h(s));
#   - m/W:  energy gap of the self-consistent fermion pole below the band, in units of W
#           (an ENERGY-ratio; the order parameter of the gap equation).
# They are commensurable ONLY through an identification.  Two candidate identifications,
# both tested below:
#   (A)  s = m/W   (the prompt's literal hypothesis: s IS the transmutation ratio)
#   (B)  the gap's ENERGY displacement below the top equals the band-variable displacement
#        delta_h = 2 pi^2 s^2 matched to the gap m measured in the SAME (h) units.
sym_s = sp.symbols('s', real=True)
H_sym = 4 * sp.pi**2
delta_h = sp.simplify(sp.Rational(1, 2) * H_sym * sym_s**2)
print(f"  band-variable displacement:  delta_h(s) = (1/2) H s^2 = {delta_h}  (H = 4 pi^2 exact)")
print(f"  transmutation energy gap:    m/W = exp(-1/(g N0))   (an ENERGY ratio, the order param)")
print(f"  => these are DIFFERENT displacements (crystal-momentum s vs energy-gap m/W).")
print(f"     The gap equation determines m/W; it does NOT, by itself, determine the")
print(f"     crystal-momentum s -- UNLESS an extra identification ties them.  Test both (A),(B).")

# =====================================================================================
hdr("(3) i04 FORCING: the coupling is spectral data of D_F.  Put in its FORCED value.")
# =====================================================================================
# i04: the internal algebra = commutant of the A4 regular action on the 12 darts
#      = C[A4] = C^3 (+) M3(C) (dim 12).  The internal Dirac spectrum sits on the
#      Ramanujan shell |h|^2 = k*-1 = 2.  So the only forced dimensionless coupling
#      magnitude available to the gap equation is the EFFECTIVE coupling lambda = g*N0,
#      and i04 says g is NOT free: it is read off D_F's spectrum.
# The forced spectral magnitudes the object supplies (no tuning, all from srs):
hs = []
for kk in [(.13, .27, .41), (.55, .21, .89), (.33, .61, .07), (.70, .20, .50), (.10, .50, .90)]:
    hs.extend(np.linalg.eigvals(srs.hashimoto(kk)))
hs = np.array(hs); bulk = hs[np.abs(hs.imag) > 1e-6]
shell = float(np.mean(np.abs(bulk)**2))
print(f"  forced Hashimoto shell |h|^2 = {shell:.6f}  (= k*-1 = {SHELL}, FORCED; i04)")
print(f"  forced band-top NB root h_P = (k* + sqrt(k*^2 - 4(k*-1)))/2 = (3+1)/2 = 2  (FORCED)")
print(f"  forced effective-coupling candidates the spectrum supplies (dimensionless):")
print(f"    |h|^2 = 2  (shell);   h_P = 2  (band top);   k* = 3;   k*-1 = 2.")
# The gap equation's natural dimensionless object is lambda = g*N0.  i04 forces the
# coupling to be a spectral magnitude.  The ONLY spectral magnitudes are {2,2,3,...}.
# Crucially: the gap equation needs lambda = g*N0; N0 is the DOS (a density, regularization-
# dependent number, NOT a clean spectral invariant).  We carry lambda = g*N0 as ONE object.

# =====================================================================================
hdr("(4) SOLVE the self-consistency for s under each identification, FORCED inputs.")
# =====================================================================================
print("""  (A) s = m/W = exp(-1/(g N0)) = exp(-1/lambda).
      With the forced effective coupling lambda = g*N0 read as a SPECTRAL magnitude.
      The closed form is  s = exp(-1/lambda).  Evaluate at the forced shell magnitudes:""")
for lam_name, lam_val in [("|h|^2 = 2", 2.0), ("k* = 3", 3.0), ("k*-1 = 2", 2.0),
                          ("h_P = 2", 2.0)]:
    s_A = np.exp(-1.0/lam_val)
    dh = 2*np.pi**2 * s_A**2
    print(f"      lambda = {lam_name:10s}:  s = exp(-1/lambda) = {s_A:.6f}"
          f"   (then delta_h = 2 pi^2 s^2 = {dh:.4f})")

print("""
  (B) Match the gap's displacement IN BAND-VARIABLE units to delta_h = 2 pi^2 s^2.
      The gap equation, linearized at the BAND TOP (not center), gives the standard
      BCS-edge result m_edge ~ W exp(-1/lambda).  Equate the dimensionless displacement
      of the read:  delta_h/h_P = (gap as a fraction of the shell) = exp(-1/lambda):
          2 pi^2 s^2 / h_P = exp(-1/lambda)   (h_P = 2)
          => s = sqrt( (h_P/(2 pi^2)) exp(-1/lambda) ) = sqrt( exp(-1/lambda)/pi^2 ).""")
for lam_val in [2.0, 3.0]:
    s_B = np.sqrt(np.exp(-1.0/lam_val)/np.pi**2)
    print(f"      lambda = {lam_val:.1f}:  s = sqrt(exp(-1/lambda)/pi^2) = {s_B:.6f}")

# =====================================================================================
hdr("(5) SENSITIVITY: vary each input, find what is forced vs free/regularization-dependent")
# =====================================================================================
# Re-extract N0 at the band CENTER and at the band EDGE for several bin counts to expose
# the regularization (binning) dependence -- the crux of whether s is FORCED.
print("  DOS N0 vs binning (exposes regularization dependence of the gap-eq input):")
print(f"    {'bins':>6} {'N0(center)':>12} {'N0(edge~top)':>14}")
for nb in [100, 200, 400, 800, 1600]:
    h2, e2 = np.histogram(eps, bins=nb, density=True)
    c2 = 0.5*(e2[:-1]+e2[1:])
    n0c = h2[np.argmin(np.abs(c2))]
    # DOS near the Perron top: average of the top few populated bins
    top_region = c2 > (eps.max() - 0.15*W)
    n0e = float(np.mean(h2[top_region])) if top_region.any() else float('nan')
    print(f"    {nb:6d} {n0c:12.5f} {n0e:14.5f}")
print("  => N0 is a DENSITY: it DRIFTS with bin count / sampling. It is NOT a clean")
print("     spectral invariant.  Whatever 's' the gap eq yields inherits this drift")
print("     through lambda = g*N0 in the exponent exp(-1/(g N0)).")

print("\n  Sensitivity of s_A = exp(-1/lambda) to lambda (d ln s / d ln lambda = 1/lambda):")
for lam_val in [1.5, 2.0, 2.5, 3.0]:
    s_A = np.exp(-1/lam_val)
    dlns_dlnlam = 1.0/lam_val
    print(f"    lambda={lam_val:.1f}: s={s_A:.5f},  d ln s/d ln lambda = {dlns_dlnlam:.3f}"
          f"  (10% in lambda -> {10*dlns_dlnlam:.1f}% in s)")

# =====================================================================================
hdr("(6) DOES THE BAND-TOP GAP EQUATION EVEN HAVE A NONTRIVIAL EDGE SOLUTION?")
# =====================================================================================
# The transmutation form m ~ W exp(-1/lambda) is the BCS/NJL result for a pairing
# instability at a band CENTER with FINITE DOS N0.  At the Perron band TOP the DOS of a
# d=3 (Weyl-law, i03) band VANISHES like sqrt(eps_top - eps) (a 3D band edge).  Test
# whether the gap eq linearized AT THE TOP gives an exponential (finite-N0) or a
# power-law (vanishing-N0) threshold -- this decides if 's off the top' is transmutation
# at all.
print("  d=3 band edge: DOS(eps) ~ sqrt(eps_top - eps) near the Perron top (vanishes).")
# numerically: DOS in the top 10% window vs distance from top
top = eps.max()
dist = top - ec
mask = (dist > 0) & (dist < 0.25*W) & (hist > 0)
if mask.sum() > 5:
    p = np.polyfit(np.log(dist[mask]), np.log(hist[mask]), 1)
    print(f"    fit DOS ~ (eps_top-eps)^p near top:  p = {p[0]:.3f}"
          f"   (p ~ +0.5 => vanishing 3D band-edge DOS)")
print("""  CONSEQUENCE: at a VANISHING-DOS band edge the pairing kernel
      I_top(m) = int_0^W sqrt(x)/(2 sqrt(x^2+m^2)) dx  is FINITE as m->0
  (no log divergence), so 1 = g I(m) has a solution only for g above a THRESHOLD g_c,
  and near threshold m is a POWER of (g-g_c), NOT exp(-1/(g N0)).  There is no
  transmutation (no exponential) at the band TOP -- transmutation needs the finite-DOS
  band CENTRE (where i01 actually solved it).  So 's off the top' is NOT the
  transmutation gap m: the gap lives at the band centre, s lives at the top.""")
# verify the kernel is finite at m->0 at the top:
def I_top(m):
    x = np.linspace(1e-6, W, 40000)      # edge variable x = eps_top - eps
    dos = np.sqrt(x)                     # d=3 band-edge DOS ~ sqrt(x) (vanishes at the top)
    return float(np.trapezoid(dos/(2*np.sqrt(x**2 + m**2)), x))
print(f"    I_top(m->0) = {I_top(1e-8):.4f}  (FINITE => threshold g_c = 1/I_top, no exp law)")
print(f"    => 1 = g I_top(m) solvable only for g > g_c = {1/I_top(1e-8):.4f}; near g_c, m ~ power, not exp.")

# =====================================================================================
hdr("(7) HONEST VERDICT")
# =====================================================================================
print("""  FORCED inputs:
    - H = 4 pi^2 (band-top curvature, exact)           FORCED  [perron_curvature_run_scratch]
    - h_P = 2, |h|^2 = 2, k* = 3                         FORCED  [srs; i04]
    - W = bandwidth (geometric)                          FORCED  [srs band ~ 5.99]
  AMBIGUOUS / not-forced inputs:
    - N0 (DOS): a DENSITY, drifts with binning/sampling  NOT a clean invariant (step 5)
    - g: i04 says 'spectral data', but the gap eq needs lambda = g*N0, and N0 is not a
         clean invariant; the spectral magnitudes {2,3} are candidates, not a unique pin.
    - the IDENTIFICATION s <-> m/W: there are >=2 inequivalent ways to relate the
         crystal-momentum s to the energy gap m/W (forms (A),(B)); they give different s.

  STRUCTURAL OBSTRUCTION (step 6): the transmutation gap m is a band-CENTRE instability
    (finite DOS -> exponential exp(-1/(g N0))).  The displacement s is a band-TOP
    (Perron) coordinate, where the d=3 DOS VANISHES -> the gap equation there has a
    THRESHOLD (power-law), NOT a transmutation exponential.  So s is NOT the
    transmutation ratio m/W: they live at opposite ends of the band.

  => The interaction's gap equation does NOT cleanly fix s.
     - It fixes a band-CENTRE order parameter m/W = exp(-1/(g N0)); s is a band-TOP
       crystal-momentum displacement; the two are different objects.
     - Even granting an identification, s would inherit N0's regularization drift and
       the choice of identification -- so no single forced number emerges.
     This is a CLEAN NEGATIVE: the bare object's verdict stands -- s is a free
     type-III_1 scale that the gap equation does not pin.""")
print("[done]")
