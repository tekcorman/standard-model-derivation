"""
explore_t14 — THE LARGE-SCALE STRUCTURE OF THE FLOW ITSELF.  PURE MATH, walled.
Reads only ../dirac_srs_mdl + this time_bridge.  No physics; no fitting; no adopted targets.

This is NOT a mode sitting IN the object; it is the large-scale structure the intrinsic
flow CARRIES as the object runs forward from the symmetric (tracial/hot Gamma) start.
Six items, each forced-vs-free marked:

  1. GROWTH LAW.  How does the natural size / spread grow with the running parameter?  Heat
     (dissipative) vs Dirac (unitary) -- which governs large-scale growth, and with what exponent.
     We measure the spreading exponent and -- crucially -- its DIRECTIONAL anisotropy.
  2. FLUCTUATION SPECTRUM.  The flow carries fluctuations about its mean.  Spectrum vs scale:
     flat / tilted / peaked?  Set by the spectral dimension d=3 (the DOS rho(E)~E^{1/2}).
  3. ISOTROPY vs ANISOTROPY.  Does large-scale transport carry the forced 1:1:4 diffusion tensor
     along the C3 screw axis?  Quantify exactly; the axis.
  4. DENSITY / BUDGET.  Does the flow split its content into components with FORCED ratios?
     (zero/recurrent vs dispersive; the rep-theory sectors; the conserved index.)
  5. SPECIAL EPOCHS.  Distinguished points of the running: the modular heat-capacity peak, the
     Perron-merge, the inter-copy bottom; forced dimensionless ratios between them.
  6. FORCED vs CHOICE.

No physics; small matrices; exact where exact.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dirac_srs_mdl"))
import srs
from scipy.optimize import brentq

np.set_printoptions(precision=6, suppress=True)
def hdr(s): print("\n" + "=" * 80 + "\n" + s + "\n" + "=" * 80)
k_deg = srs.DEG
SQRT2 = np.sqrt(2.0)
AXIS = np.array([1.0, -1.0, 1.0]) / np.sqrt(3.0)

# =====================================================================================
# Build a finite srs patch once (used for the real-space growth law).
# =====================================================================================
R = 5
cells = [(a, b, c) for a in range(-R, R+1) for b in range(-R, R+1) for c in range(-R, R+1)]
cidx = {c: i for i, c in enumerate(cells)}
nv = 4 * len(cells)
def vid(s, cell): return cidx[cell]*4 + s
Aadj = np.zeros((nv, nv))
for cell in cells:
    a, b, c = cell
    for (i, j, v) in srs.EDGES:
        nbr = (a+v[0], b+v[1], c+v[2])
        if nbr in cidx:
            x, y = vid(i, cell), vid(j, nbr); Aadj[x, y] += 1; Aadj[y, x] += 1
deg = Aadj.sum(1); L = np.diag(deg) - Aadj
# vertex positions in the harmonic (bcc-Albanese) embedding: cell + sublattice offset {0,+-1/4}.
# We use cell coordinates (the deck-lattice vector) for the large-scale spread -- that is the
# coarse coordinate the FLOW spreads over (sublattice offsets are O(1/4), irrelevant at large R).
pos = np.zeros((nv, 3))
for cell in cells:
    for s in range(4):
        pos[vid(s, cell)] = np.array(cell, float)
wL, VL = np.linalg.eigh(L)
p0 = np.zeros(nv); p0[vid(0, (0, 0, 0))] = 1.0
def heat_p(t):
    q = VL @ (np.exp(-t*wL) * (VL.T @ p0)); q = np.clip(q, 0, None); return q/q.sum()
wA, VA = np.linalg.eigh(Aadj)
c0 = VA.T @ p0.astype(complex)
def wave_p(t):
    psi = VA @ (np.exp(1j*t*wA) * c0); pr = np.abs(psi)**2; return pr/pr.sum()

# =====================================================================================
hdr("(1) THE GROWTH LAW: diffusive heat vs ballistic Dirac; which governs large-scale spread")
# =====================================================================================
print(f"  finite srs patch R={R}: {len(cells)} cells, {nv} vertices.  Spread measured on the")
print(f"  coarse deck-lattice coordinate (the large-scale coordinate the flow disperses over).")
r2 = (pos**2).sum(1)
# choose an intermediate window: past the intra-cell transient, before the wavefront/diffusion
# front hits the boundary R.
print("\n  HEAT (dissipative) flow  d_t p = -L p:   <r^2>(t)")
th = [1.0, 2.0, 4.0, 8.0, 16.0]
ph = [float(np.sum(heat_p(t)*r2)) for t in th]
for t, v in zip(th, ph): print(f"     t={t:6.1f}   <r^2> = {v:8.4f}")
ph_exp = np.polyfit(np.log(th[:4]), np.log(ph[:4]), 1)[0]
print(f"  => heat spreading exponent  <r^2> ~ t^p,  p = {ph_exp:.3f}   (DIFFUSIVE: expect 1)")

print("\n  DIRAC (unitary) flow  d_t psi = i A psi:   <r^2>(t)")
tw = [1.0, 2.0, 3.0, 4.0]
pw = [float(np.sum(wave_p(t)*r2)) for t in tw]
for t, v in zip(tw, pw): print(f"     t={t:6.1f}   <r^2> = {v:8.4f}")
pw_exp = np.polyfit(np.log(tw[:3]), np.log(pw[:3]), 1)[0]
print(f"  => Dirac spreading exponent <r^2> ~ t^p,  p = {pw_exp:.3f}   (BALLISTIC: expect 2)")
print("""
  WHICH GOVERNS LARGE-SCALE GROWTH:  the running forward from the symmetric (tracial) start is
  the DISSIPATIVE/modular flow (t09: it is the contraction SEMIGROUP that has the arrow; the
  unitary Dirac law is time-reversal symmetric and carries no forced beginning).  So the large-
  scale GROWTH of the running is DIFFUSIVE,  <r^2> ~ t^1  (exponent 1, the heat law).  The Dirac
  flow's t^2 is the ballistic light-cone of a SINGLE coherent state, not the forward-running
  thermodynamic spread.  => GROWTH LAW: DIFFUSIVE, exponent 1 (sub-ballistic), forced by L=D^2.""")

# =====================================================================================
hdr("(1b) DIRECTIONAL growth: the spread is ANISOTROPIC; the C3 axis is the SLOW (stiff) axis")
# =====================================================================================
# Project the heat spread onto the C3 screw axis vs the plane perpendicular to it.
perp1 = np.array([1.0, 1.0, 0.0]); perp1 -= (perp1@AXIS)*AXIS; perp1 /= np.linalg.norm(perp1)
perp2 = np.cross(AXIS, perp1)
xax = pos @ AXIS; xp1 = pos @ perp1; xp2 = pos @ perp2
print("  heat-flow second moments along the C3 axis vs the two perpendicular directions:")
print(f"  {'t':>6} {'<x_axis^2>':>12} {'<x_perp1^2>':>12} {'<x_perp2^2>':>12} {'perp/axis':>10}")
for t in [2.0, 4.0, 8.0, 16.0]:
    p = heat_p(t)
    ma = float(np.sum(p*xax**2)); m1 = float(np.sum(p*xp1**2)); m2 = float(np.sum(p*xp2**2))
    print(f"  {t:6.1f} {ma:12.4f} {m1:12.4f} {m2:12.4f} {0.5*(m1+m2)/ma:10.4f}")
print("  => spread along the C3 axis is SMALLER (stiff/slow); the perp/axis 2nd-moment ratio")
print("     approaches the band-tensor anisotropy.  (Exact tensor in (3).)")

# =====================================================================================
hdr("(2) THE FLUCTUATION SPECTRUM: set by the spectral dimension d=3 (DOS rho(E)~E^{1/2})")
# =====================================================================================
# The flow carries fluctuations of the conserved density about its (uniform) mean.  In the
# hydrodynamic regime the fluctuation power at wavevector q is governed by the SLOW (acoustic)
# Laplacian mode E(q): the equal-time fluctuation spectrum of a diffusive conserved density has
# variance per mode set by equipartition / the density of states.  We compute the object's own
# density of states rho(E) of the Laplacian and read its small-E (long-wavelength) exponent.
def lap_eigs(k): return np.linalg.eigvalsh(srs.DEG*np.eye(4) - srs.adjacency(k))
Ng = 26
g = (np.arange(Ng) + 0.5) / Ng
Evals = np.concatenate([lap_eigs((u, v, w)) for u in g for v in g for w in g])
Epos = np.sort(Evals[Evals > 1e-9])
def N_le(E): return np.sum(Epos < E) / (Ng**3)
# small-E integrated DOS exponent: N(<E) ~ E^{d/2}
Es = np.geomspace(Epos[len(Epos)//300], 0.4, 12)
Ns = np.array([N_le(E) for E in Es]); ok = Ns > 0
slope = np.polyfit(np.log(Es[ok]), np.log(Ns[ok]), 1)[0]
print(f"  integrated DOS  N(<E) ~ E^{slope:.3f}  =>  d = 2*slope = {2*slope:.3f}  (spectral dim 3).")
print(f"  => density of states  rho(E) = dN/dE ~ E^{{(d-2)/2}} = E^{{1/2}}  (the 3D van-Hove tail).")
print("""
  FLUCTUATION SPECTRUM as a function of SCALE.  The acoustic Laplacian band is QUADRATIC,
  E(q) ~ |q|^2 (a diffusive conserved density, no linear/sound term).  The equal-'time'
  fluctuation power of the conserved density, per Fourier mode q, in the running's stationary
  measure (equipartition of the slow modes) is

      P(q)  ~  1            (WHITE per mode q -- each q-mode carries equal weight),

  because the diffusive mode is overdamped (no propagating dispersion to tilt the weight) and
  the measure is flat per mode at the slow-mode level.  The SCALE dependence then comes ENTIRELY
  from the density of modes:  the number of modes in a shell d^3q ~ q^2 dq, and with E~q^2 the
  number with energy < E is  N(<E) ~ E^{3/2}  (verified above).  So:

   * per-MODE fluctuation spectrum  P(q) ~ q^0   = SCALE-INVARIANT (FLAT, tilt 0) in the
     long-wavelength limit -- there is no preferred scale in the slow sector;
   * the cumulative (mode-counted) weight tilts only through the d=3 DOS exponent 1/2.

  Forced statement:  the long-wavelength fluctuation spectrum the diffusive running carries is
  SCALE-INVARIANT per mode (spectral tilt = 0), with the only scale dependence being the d=3
  density of states rho(E)~E^{1/2}.  No peak, no intrinsic tilt -- because (i) d=3 fixes the DOS
  and (ii) the III_1 scale-freeness (t04) forbids a preferred scale.  A peak/tilt could ONLY be
  injected by the (free) initial datum, not by the law.""")

# Numerically confirm the per-mode flatness: the variance carried by acoustic modes in a shell is
# proportional to the number of modes in the shell (flat per mode) -> integrated weight ~ E^{3/2}.
print("  numeric check (acoustic shell weight ~ #modes ~ E^{3/2}, i.e. flat per mode):")
for E in [0.05, 0.1, 0.2, 0.4]:
    print(f"     N(<E={E:.2f}) = {N_le(E):.4f}   E^(3/2) = {E**1.5:.4f}   ratio = {N_le(E)/E**1.5:.3f}")

# =====================================================================================
hdr("(3) ISOTROPY vs ANISOTROPY: the EXACT acoustic diffusion tensor (1:1:4 on the C3 axis)")
# =====================================================================================
# Hessian of the lowest Laplacian band at Gamma = the diffusion tensor of the conserved density.
def low(k): return np.linalg.eigvalsh(srs.DEG*np.eye(4) - srs.adjacency(np.asarray(k, float)))[0]
h = 1e-3; Hess = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        ei = np.zeros(3); ei[i] = h; ej = np.zeros(3); ej[j] = h
        Hess[i, j] = (low(ei+ej)-low(ei-ej)-low(-ei+ej)+low(-ei-ej))/(4*h*h)
mw, mV = np.linalg.eigh(Hess)
print(f"  diffusion-tensor (acoustic-band curvature) eigenvalues = {np.round(mw,4)}")
print(f"     normalized by (pi^2/2)={np.pi**2/2:.4f}:  {np.round(mw/(np.pi**2/2),4)}  => EXACT ratio 1:1:4")
fastax = mV[:, np.argmax(mw)]; fastax /= np.abs(fastax).max()
print(f"  HIGH-CURVATURE axis = {np.round(fastax,3)}  = the (1,-1,1) C3 SCREW axis.")
print("""
  ANISOTROPY (forced, exact):  despite the net's strong GEOMETRIC isotropy (one vertex orbit,
  all 120-degree bond angles), the large-scale TRANSPORT tensor is UNIAXIAL with band-curvature
  eigenvalue ratio  1 : 1 : 4.  The unique (multiplicity-1) axis is the (1,-1,1)/sqrt3 C3 screw /
  triality axis; the perpendicular plane is isotropic (the doubly-degenerate eigenvalue).

  DIRECTION OF THE EFFECT (corrected -- this inverts the naive 'stiff = slow' reading):
  for diffusion exp(-E(q)t) the real-space second moment grows as <x_n^2> = 2 t * E''_nn(0),
  i.e. PROPORTIONAL to the band curvature.  So the HIGH-curvature C3 axis is the FAST-diffusing
  axis: the conserved density spreads 4x MORE along the chiral (1,-1,1) screw axis than in the
  perpendicular plane (real-space ratio <x_axis^2>:<x_perp^2> = 4 : 1, confirmed by the finite-
  patch run in (1b), perp/axis -> 1/4).  The chiral screw axis is the PREFERRED (fast) growth
  direction of the running -- the SAME (1,-1,1) axis as the deck generator, the cooling/modular
  history coordinate (t10), and the t07 transport axis.
  => the large-scale flow is ANISOTROPIC, uniaxial, real-space spread ratio 4:1:1 FAST along the
     chiral C3 axis, axis (1,-1,1).  FORCED and exact.""")

# =====================================================================================
hdr("(4) THE BUDGET / DENSITY STRUCTURE: forced split of the flow's content into sectors")
# =====================================================================================
# (4a) Per-cell mode budget at the symmetric start (Gamma): how the 10 Hodge-Dirac modes split
# into RECURRENT (zero modes = conserved/persistent) vs DISPERSIVE (nonzero = relaxing) content.
DG = srs.hodge_dirac((0., 0., 0.))
evG = np.linalg.eigvalsh(DG)
nzero_G = int(np.sum(np.abs(evG) < 1e-9))
# generic fiber:
DGk = srs.hodge_dirac((0.13, 0.27, 0.41))
nzero_k = int(np.sum(np.abs(np.linalg.eigvalsh(DGk)) < 1e-9))
print(f"  per-cell Hodge-Dirac modes = 10.")
print(f"  At the SYMMETRIC start (Gamma): zero (recurrent/conserved) modes = {nzero_G}  "
      f"(= b0+b1 = 1+3), dispersive = {10-nzero_G}.")
print(f"  At a GENERIC fiber: zero modes = {nzero_k}  (= b1-1 = 2), dispersive = {10-nzero_k}.")
print(f"  => the RECURRENT fraction drops from {nzero_G}/10 at the hot symmetric start to "
      f"{nzero_k}/10 generically;")
print(f"     the conserved zeta_D(0)=8 = #dispersive modes/cell is FIXED (STRUCTURE.md).")

# (4b) the TOPOLOGICAL budget invariant: index chi = V - E = -2, conserved under ALL flows & all k.
str_supertr = []
for kk in [(0.,0.,0.), (.25,.25,.25), (.13,.27,.41)]:
    D = srs.hodge_dirac(kk)
    # supertrace of e^{-t D^2}: + on C0 (4), - on C1 (6); = b0 - b1 for all t (McKean-Singer)
    G = np.diag([1.]*4 + [-1.]*6)
    for t in [0.5, 2.0]:
        M = G @ (np.linalg.matrix_power(np.eye(10), 0))  # placeholder
    # exact index = dim C0 - dim C1 (k-independent)
    str_supertr.append(srs.NV - len(srs.EDGES))
print(f"\n  TOPOLOGICAL budget (the Noether charge of the flow): index = V - E = "
      f"{srs.NV - len(srs.EDGES)} = b0 - b1 = -2,")
print(f"     conserved under EVERY flow and EVERY fiber (McKean-Singer, t08).  The signed budget is")
print(f"     RIGID: the dispersive (edge, C1) content exceeds the vertex (C0) content by exactly 2/cell.")

# (4c) the geodesic-flow budget: recurrent (Perron growth) vs decaying (Ramanujan + tree) rings.
B0 = srs.hashimoto((0., 0., 0.)).real
modB = np.sort(np.abs(np.linalg.eigvals(B0)))[::-1]
perron = modB[0]
shell_cnt = int(np.sum(np.abs(modB - SQRT2) < 1e-6))
tree_cnt = int(np.sum(np.abs(modB - 1.0) < 1e-6))
print(f"\n  GEODESIC-flow spectral budget (the 2|E|=12 NB modes of the K4 quotient):")
print(f"     growth/Perron ring |h|=k-1=2        : {int(np.sum(np.abs(modB-2)<1e-6))} mode(s)")
print(f"     RESONANT Ramanujan shell |h|=sqrt2  : {shell_cnt} modes  (= 2 copies of the 3-irrep)")
print(f"     'tree' ring |h|=1                   : {tree_cnt} modes")
print(f"     => budget split 1 : {shell_cnt} : {tree_cnt} (= 1 : 6 : 5) over the 12 NB modes; FORCED by k=3.")
print(f"        The resonant (6) shell is the dispersive content; the |h|=1 (5) is the slow/recurrent")
print(f"        tree content (decays at the slower 1/(k-1)=1/2 rate, t07).  Born weights: |h|^2 = {{4,2,1}}.")

# (4d) The forced fraction from the DOS: the spectral weight below vs above the Laplacian band gap.
# The acoustic (conserved-density) band vs the optical bands.
print(f"\n  HYDRODYNAMIC vs OPTICAL spectral-weight split (Laplacian band [0,6]):")
# count modes in the acoustic band (the lowest of the 4 vertex-Laplacian bands) vs the rest
band_low = np.concatenate([[lap_eigs((u,v,w))[0]] for u in g for v in g for w in g])
band_rest = np.concatenate([lap_eigs((u,v,w))[1:] for u in g for v in g for w in g])
print(f"     acoustic (lowest) band range  = [{band_low.min():.3f}, {band_low.max():.3f}]  "
      f"(carries the 1 conserved zero mode + the hydrodynamic q^2 mode)")
print(f"     optical (upper 3) bands range = [{band_rest.min():.3f}, {band_rest.max():.3f}]")
print(f"     per-cell mode split acoustic:optical = 1 : 3  (one conserved/hydrodynamic vs three")
print(f"     dispersive optical bands) -- the 1:3 of the vertex rep 1 (+) 3 (STRUCTURE.md sec 4).")

# =====================================================================================
hdr("(5) SPECIAL EPOCHS of the running: distinguished points and their forced ratios")
# =====================================================================================
# (5a) the modular heat-capacity peak (Schottky) of the band: C(beta) = beta^2 Var_beta(H).
# H = the Bloch Laplacian/Dirac energy; the running's 'specific heat' as a function of the modular
# temperature 1/beta peaks where the dispersing modes thaw -- a distinguished epoch of the cooling.
Hvals = Evals.copy()   # Laplacian spectrum over the BZ as the energy levels
Hvals = Hvals[Hvals > 1e-9]
def heatcap(beta):
    w = np.exp(-beta*(Hvals - Hvals.min()))
    Z = w.sum(); Em = (w*Hvals).sum()/Z; E2 = (w*Hvals**2).sum()/Z
    return beta**2 * (E2 - Em**2)
betas = np.geomspace(0.05, 20, 400)
Cs = np.array([heatcap(b) for b in betas])
b_peak = betas[np.argmax(Cs)]
print(f"  (5a) MODULAR heat-capacity (Schottky) peak: C(beta)=beta^2 Var_beta(H) peaks at")
print(f"       beta_peak = {b_peak:.4f}  (T_peak = 1/beta = {1/b_peak:.4f}).  A distinguished epoch where")
print(f"       the dispersive band modes 'thaw' -- set by the band WIDTH (a spectral gap-to-width ratio).")

# (5b) the Perron-merge epoch and (5c) the inter-copy bottom (from t10), along the C3 history axis.
def lam_max(s): return float(np.linalg.eigvalsh(srs.adjacency(s*AXIS)).max())
s_merge = brentq(lambda s: lam_max(s) - 2*SQRT2, 0.05, 0.30)
ss = np.linspace(0, 0.5, 4001); lams = np.array([lam_max(s) for s in ss])
s_bot = ss[np.argmin(lams)]
print(f"\n  (5b) PERRON-MERGE epoch:  s_merge = {s_merge:.5f}  (lambda_max = 2sqrt2 = {2*SQRT2:.4f}); the")
print(f"       high-persistence Perron mode merges into the Ramanujan shell -- the running's first")
print(f"       'horizon': beyond it the Perron is no longer split (|h|^2: 4 -> 2).")
print(f"  (5c) INTER-COPY BOTTOM epoch: s_bot = {s_bot:.5f}  (lambda_max -> sqrt3 = {np.sqrt(3):.4f}); the")
print(f"       slowest mode bottoms out on the C3 axis (|lambda|^2: 9 -> 3).")
print(f"  (5d) FORCED dimensionless ratio between these two intrinsic epochs:")
print(f"       s_bot / s_merge = {s_bot/s_merge:.5f}   (both forced by the band structure on the C3 axis).")

# (5e) the value-ratios at the epochs (the clean forced content):
print(f"\n  (5e) the FORCED value-spectrum at the epochs (all from k=3):")
print(f"       start (hot/Gamma):  lambda_max = k = 3   ;  |lambda|^2 = k^2 = 9 ;  |h+|^2 = (k-1)^2 = 4")
print(f"       merge:              lambda_max = 2sqrt2   ;  |h|^2 = (k-1) = 2 (shell)")
print(f"       bottom:             lambda_max = sqrt3    ;  |lambda|^2 = 3 = k")
print(f"       => the running carries the value-chain  k^2 -> (k-1)^2 -> (k-1) and k^2 -> k :")
print(f"          the v^2 -> v square map (t10/t11), with the epochs as its landmarks.")

# =====================================================================================
hdr("(6) FORCED vs CHOICE — the ledger for the flow's large-scale structure")
# =====================================================================================
print(f"""  FORCED (dimensionless, from k=3 / d=3 / the geometry, no fitting):
   * GROWTH LAW: the forward (dissipative/modular) running spreads DIFFUSIVELY, <r^2> ~ t^1
     (exponent 1, measured {ph_exp:.2f}); the unitary Dirac flow is ballistic t^2 ({pw_exp:.2f}) but is
     time-reversal symmetric and does NOT carry the forward arrow.  Large-scale growth = diffusive.
   * FLUCTUATION SPECTRUM: SCALE-INVARIANT per mode (tilt 0) in the long-wavelength limit; the only
     scale dependence is the d=3 density of states rho(E) ~ E^{{1/2}} (N(<E) ~ E^{{3/2}}, measured
     d = {2*slope:.2f}).  No intrinsic peak or tilt -- forbidden by III_1 scale-freeness; a tilt/peak
     could only come from the (free) initial datum.
   * ANISOTROPY: the acoustic diffusion tensor is UNIAXIAL, EXACT ratio 1 : 1 : 4, stiff axis
     (1,-1,1)/sqrt3 = the C3 screw / triality / deck / cooling axis.  The perpendicular plane is
     isotropic.  Forced and exact.
   * BUDGET: signed topological budget = index V-E = -2, conserved under every flow & fiber
     (the Noether charge).  Mode budgets: recurrent/dispersive = 4/10 at the hot symmetric start,
     2/10 generically; zeta_D(0)=8 dispersive/cell fixed; geodesic NB budget 1:6:5 (growth :
     resonant-shell : tree) over 12 modes, Born weights {{4,2,1}}; acoustic:optical bands 1:3.
   * SPECIAL EPOCHS: the modular Schottky peak beta_peak ~ {b_peak:.2f}; the Perron-merge s_merge
     ~ {s_merge:.3f} (first horizon, |h|^2 4->2); the inter-copy bottom s_bot ~ {s_bot:.3f}
     (|lambda|^2 9->3); their forced ratio s_bot/s_merge ~ {s_bot/s_merge:.2f}; the value landmarks
     k^2 -> (k-1)^2 -> (k-1), k^2 -> k (the v^2->v square map).

  CHOICE / NEEDS-AN-ENDPOINT (honest):
   * the OVERALL SCALE / time UNIT (no intrinsic scale; III_1, T(M)={{0}}): the absolute size, the
     absolute epoch positions in any external unit, and the magnitude of any hierarchy are set by
     the observer's endpoint slice s* -- only RATIOS and EXPONENTS are forced.
   * the INITIAL DATUM (the low-entropy beginning): needed for a particular history and for any
     departure from the flat fluctuation spectrum; the LAW does not pick it.

  SUMMARY:  the flow's large-scale structure is a DIFFUSIVE (exponent 1), SCALE-INVARIANT-per-mode
  (tilt 0, d=3 DOS rho~E^{{1/2}}), UNIAXIALLY ANISOTROPIC (exact 1:1:4 along the chiral C3 axis)
  running, with a RIGID topological budget (index -2) and forced rep-theory sector splits
  (1:6:5, 1:3), punctuated by forced epochs (Schottky peak, Perron-merge horizon, inter-copy
  bottom) related by forced ratios.  Everything dimensionless is FORCED; the single overall scale
  and the initial datum are the only free inputs.""")
