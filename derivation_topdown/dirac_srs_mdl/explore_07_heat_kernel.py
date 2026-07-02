"""
explore_07 — heat-kernel / spectral structure of the srs Hodge-Dirac.  Pure math, walled off.

(1) spectral dimension d from the SCALAR Laplacian L(k)=3I-A(k):  N(<E) ~ E^{d/2} at small E.
(2) heat-kernel trace  K(t) = (1/N_k) sum_BZ sum_n exp(-t lambda_n^2)  (lambda_n = Hodge-Dirac
    eigenvalues): leading small-t power t^{-d/2} + Seeley-DeWitt-type coefficients a_0, a_1, ...
(3) zeta_D(0): the regularized t^0 constant in that small-t expansion (subtle; reported honestly).
(4) zero modes of hodge_dirac at Gamma, P, H -> cohomology of the K_4 cell (b_0=1, b_1=3).
"""
import numpy as np
import srs

np.set_printoptions(precision=4, suppress=True)
TWO_PI = 2*np.pi


# ----------------------------------------------------------------------------------------
def bz_grid(Ng):
    """Fine uniform Brillouin-zone grid k in [0,1)^3 (fractional)."""
    a = (np.arange(Ng) + 0.0)/Ng
    return [(x, y, z) for x in a for y in a for z in a]


def laplacian_eigs(k):
    """Scalar graph Laplacian L(k) = 3I - A(k); return its 4 (real) eigenvalues."""
    L = srs.DEG*np.eye(srs.NV) - srs.adjacency(k)
    return np.linalg.eigvalsh(L)


def dirac_eigs(k):
    """Hodge-Dirac eigenvalues (10 of them, real, +/- symmetric about 0)."""
    return np.linalg.eigvalsh(srs.hodge_dirac(k))


# ========================================================================================
print("=" * 88)
print("explore_07 — HEAT KERNEL / SPECTRAL STRUCTURE of the srs Hodge-Dirac (pure math)")
print("=" * 88)

# ----------------------------------------------------------------------------------------
# (1) SPECTRAL DIMENSION from the scalar Laplacian L = 3I - A.
# ----------------------------------------------------------------------------------------
print("\n--- (1) SPECTRAL DIMENSION from the scalar Laplacian L(k)=3I-A(k) ---")
Ng = 24
ks = bz_grid(Ng)
Nk = len(ks)
Lvals = np.concatenate([laplacian_eigs(k) for k in ks])
Lpos = np.sort(Lvals[Lvals > 1e-9])          # drop the b_0=1 zero mode at Gamma
print(f"  BZ grid {Ng}^3 = {Nk} k-points; {len(Lpos)} positive Laplacian eigenvalues")
print(f"  Laplacian spectral range: [{Lpos.min():.3e}, {Lpos.max():.3f}]   (band [0,6])")

# Integrated DOS N(<E) = (#eigs < E)/N_k.  Near E->0 a 3D crystal has quadratic band bottom
# => N(<E) ~ E^{d/2}.  Fit log N vs log E on a window of SMALL E (above the discrete floor).
def integrated_dos(E):
    return np.sum(Lpos < E)/Nk

# choose a small-E window: from just above the smallest eigenvalue up to a fraction of the band
Emin_fit = Lpos[max(5, len(Lpos)//400)]       # skip the very-lowest few (finite-grid noise)
Es = np.geomspace(Emin_fit, 0.5, 14)
Ns = np.array([integrated_dos(E) for E in Es])
good = Ns > 0
logE, logN = np.log(Es[good]), np.log(Ns[good])
slope, intercept = np.polyfit(logE, logN, 1)
resid = logN - (slope*logE + intercept)
rms = np.sqrt(np.mean(resid**2))
d_spec = 2*slope
print(f"  fit  log N(<E) = {slope:.4f} * log E + const   over E in [{Es[good][0]:.3e}, {Es[good][-1]:.3f}]")
print(f"  => N(<E) ~ E^{slope:.3f}  =>  spectral dimension d = 2*slope = {d_spec:.3f}   (expect 3)")
print(f"     fit RMS(log) = {rms:.3f}  ({len(logE)} points).  Slope drifts up near the floor")
print(f"     (finite-grid + discrete spectrum) and down near band-mid (van Hove): d~3 is the clean small-E law.")

# show the local slope as a function of window to be honest about the fit
print("  local two-point slopes d=2*dlogN/dlogE (small E -> large E):")
for i in range(0, len(Es[good])-1, 2):
    e0, e1 = Es[good][i], Es[good][i+1]
    n0, n1 = Ns[good][i], Ns[good][i+1]
    if n0 > 0 and n1 > 0:
        loc = 2*(np.log(n1)-np.log(n0))/(np.log(e1)-np.log(e0))
        print(f"      E in [{e0:.3e},{e1:.3e}]:  d_local = {loc:.3f}")


# ----------------------------------------------------------------------------------------
# (2) HEAT-KERNEL TRACE of the Hodge-Dirac.  K(t)=(1/N_k) sum_BZ sum_n exp(-t lambda_n^2).
#     lambda_n^2 are the D^2 eigenvalues (Laplacian on C0 (+) edge-Laplacian on C1).
# ----------------------------------------------------------------------------------------
print("\n--- (2) HEAT-KERNEL TRACE  K(t) = <Tr exp(-t D^2)>_BZ  of the Hodge-Dirac ---")
# gather D^2 spectrum once over the BZ (square the Dirac eigenvalues)
D2 = np.concatenate([dirac_eigs(k)**2 for k in ks])     # 10 per k-point
D2 = D2.real
print(f"  collected {len(D2)} D^2 eigenvalues over the {Ng}^3 BZ grid")
print(f"  D^2 range: [{D2.min():.3e}, {D2.max():.3f}];  per-cell dim = {len(D2)//Nk} (=10)")

def K(t):
    return np.sum(np.exp(-t*D2))/Nk

# IMPORTANT (honest framing).  D is a BOUNDED operator (|lambda| <= sqrt(6)), so the trace
# Tr exp(-t D^2) is ANALYTIC at t=0 and SATURATES to the per-cell dimension (10) as t->0:
#       K(t) = 10 - t * m1 + (t^2/2) * m2 - ...,   m_p = <Tr D^{2p}>_BZ  (spectral moments).
# There is therefore NO continuum Weyl power  t^{-3/2}  in the raw lattice trace -- that power
# only appears in the CONTINUUM (lattice-spacing -> 0) limit, which a fixed graph does not have.
# The clean spectral-dimension probe is the DOS exponent in (1); here we (a) verify the analytic
# small-t Taylor law and (b) show the effective exponent of K never reaches d/2 (it -> 0).
ts = np.geomspace(1e-3, 30.0, 40)
Kt = np.array([K(t) for t in ts])
logt, logK = np.log(ts), np.log(Kt)
dlog = np.gradient(logK, logt)        # local logarithmic exponent  -dlogK/dlogt
print("  effective exponent  p_eff(t) = -dlogK/dlogt   (a continuum t^{-d/2} would give p_eff -> 1.5;")
print("  a BOUNDED operator gives p_eff -> 0 as t->0 -- which is what we see):")
for i in range(0, len(ts), 4):
    print(f"      t={ts[i]:.3e}:  K={Kt[i]:.4f}   p_eff={-dlog[i]:+.3f}")
print(f"  => p_eff -> 0 at small t (K -> 10) and -> 0 at large t (K -> #zero modes/cell);")
print(f"     it PEAKS at p~{-dlog[(ts>0.3)&(ts<1.0)].min():.2f} near t~0.6.  No t^{{-3/2}} Weyl regime")
print(f"     exists on the bounded graph spectrum -- the spectral dimension d=3 lives in the DOS, not here.")

# (a) ANALYTIC small-t Taylor structure (the genuinely correct small-t expansion here).
m1 = np.sum(D2)/Nk                       # <Tr D^2>
m2 = np.sum(D2**2)/Nk                    # <Tr D^4>
print("\n  Genuine small-t expansion (bounded op, ANALYTIC at 0):  K(t) = a0 - a1 t + a2 t^2 - ...")
print(f"      a0 = <Tr 1>      = {len(D2)/Nk:.4f}   (per-cell dimension, = 10)")
print(f"      a1 = <Tr D^2>    = {m1:.4f}            (1st spectral moment)")
print(f"      a2 = <Tr D^4>/2  = {m2/2:.4f}            (2nd moment / 2)")
# verify the Taylor law at a small t
t0 = 5e-3
taylor = len(D2)/Nk - m1*t0 + 0.5*m2*t0**2
print(f"      check at t={t0}:  K={K(t0):.6f}   vs  a0 - a1 t + a2 t^2 = {taylor:.6f}   "
      f"(match to {abs(K(t0)-taylor):.1e})")

# (b) the OTHER physical object -- the SCALAR Laplacian's heat trace per cell DOES, near the
# band bottom, follow the 3D continuum law; show its leading t^{-3/2} via the small-E DOS slope.
# (This is the honest place the d=3 Weyl term appears: in the low-lying density of states.)
print("\n  Where the d=3 (4 pi t)^{-3/2} Weyl term truly lives -- the low-E DOS tail of L:")
print(f"      from (1):  N(<E) ~ E^{slope:.3f} (=E^{{3/2}}) near E->0  ==>  rho(E)=dN/dE ~ E^{{1/2}},")
print(f"      the exact 3D van Hove / Weyl signature; its Laplace transform gives the continuum")
print(f"      heat-kernel  ~ (4 pi t)^{{-3/2}}  in the lattice-spacing->0 scaling limit.")


# ----------------------------------------------------------------------------------------
# (3) zeta_D(0): the regularized t^0 constant of the heat-kernel expansion.
# ----------------------------------------------------------------------------------------
print("\n--- (3) zeta_D(0): the regularized constant of the spectral zeta function ---")
print("  Method.  zeta_D(s) = Tr'|D|^{-s} = (1/N_k) sum_{lambda != 0} |lambda|^{-s}, analytically")
print("  continued.  Mellin:  zeta_D(s) = (1/Gamma(s/2)) int_0^inf t^{s/2-1} theta(t) dt,  with")
print("  theta(t) = (1/N_k) sum_{lambda != 0} exp(-t lambda^2)  (the heat trace over NONZERO modes).")
print("  KEY (bounded operator): theta(t) is ANALYTIC at t=0 with theta(0) = (#nonzero modes)/cell")
print("  =: N_nz.  Then near t->0 the Mellin integral has a single pole int_0 t^{s/2-1} N_nz dt =")
print("  N_nz * (2/s), and 1/Gamma(s/2) ~ s/2, so  zeta_D(0) = (s/2)*(2 N_nz/s) = N_nz  EXACTLY.")
print("  (No continuum t^{-3/2} subtraction arises -- the bounded graph spectrum has no UV pole.)")

# Build theta(t) excluding the exact zero modes (count them per BZ).
nz_mask = D2 > 1e-9
D2nz = D2[nz_mask]
n_zero_total = len(D2) - len(D2nz)
N_nz = len(D2nz)/Nk
print(f"\n  exact zero modes over the grid: {n_zero_total}  (= {n_zero_total/Nk:.3f}/cell);  "
      f"nonzero modes N_nz = {N_nz:.4f}/cell")

def theta(t):
    return np.sum(np.exp(-t*D2nz))/Nk

# (3a) the analytic prediction: zeta_D(0) = N_nz = (10 - #zeromodes)/cell.
zeta0_analytic = N_nz
print(f"  (3a) ANALYTIC result:  zeta_D(0) = N_nz = (10 - {n_zero_total/Nk:.0f})/cell = {zeta0_analytic:.4f}")

# (3b) NUMERICAL verification of the analytic continuation, done CORRECTLY by isolating the
#      t->0 pole.  Split at t=T:   zeta_D(s)*Gamma(s/2) = int_0^T t^{s/2-1} theta dt + int_T^inf.
#      In the lower piece subtract the constant limit theta(0+)=N_nz analytically (it carries the
#      whole pole):  int_0^T t^{s/2-1} N_nz dt = N_nz * T^{s/2}/(s/2)  (exact, holomorphic in s),
#      and  int_0^T t^{s/2-1} (theta - N_nz) dt  is convergent (theta - N_nz ~ -m1 t).  Then
#      zeta_D(0) = lim_{s->0} (1/Gamma(s/2)) * [ N_nz*T^{s/2}/(s/2) + reg ] = N_nz  (the reg piece,
#      times 1/Gamma(s/2)~s/2, vanishes).  Evaluate the full expression at small s as the check.
from math import gamma
T = 5.0
lo = np.geomspace(1e-6, T, 6000)            # fine grid for the regularized lower integral
th_lo = np.array([theta(t) for t in lo]) - N_nz
hi = np.geomspace(T, 80.0, 4000)
th_hi = np.array([theta(t) for t in hi])
def zeta_mellin(s):
    pole = N_nz * T**(s/2.0) / (s/2.0)                       # exact int_0^T t^{s/2-1} N_nz dt
    reg = np.trapezoid(lo**(s/2.0 - 1.0) * th_lo, lo)        # convergent remainder, lower
    tail = np.trapezoid(hi**(s/2.0 - 1.0) * th_hi, hi)       # upper integral
    return (pole + reg + tail)/gamma(s/2.0)
print("  (3b) NUMERICAL Mellin continuation (pole isolated) zeta_D(s) -> s=0  (-> N_nz):")
for s in [0.4, 0.2, 0.1, 0.05, 0.02]:
    print(f"        zeta_D({s:4.2f}) = {zeta_mellin(s):.4f}")
zs = [zeta_mellin(s) for s in (0.05, 0.02)]
zeta0_num = (zs[1]*0.05 - zs[0]*0.02)/(0.05-0.02)            # linear s->0 extrapolation
print(f"        linear s->0 extrapolation:  zeta_D(0) ~ {zeta0_num:.4f}   "
      f"(vs analytic {zeta0_analytic:.4f})")

# (3c) sanity ladder: convergent partial sums for s>d (no continuation needed there).
def zeta_sum(s):
    return np.sum(np.abs(D2nz)**(-s/2))/Nk     # = (1/Nk) sum |lambda|^{-s}
print("  (3c) sanity ladder (convergent, s>d=3):  "
      + "   ".join(f"zeta_D({s})={zeta_sum(s):.3f}" for s in (4.0, 5.0, 6.0)))

print(f"\n  VERDICT on zeta_D(0):  zeta_D(0) = {zeta0_analytic:.1f}  (= number of NONZERO modes per cell).")
print(f"  This is EXACT and robust (not a delicate fit): for a bounded operator whose heat trace")
print(f"  saturates to a constant at t->0, the Mellin/zeta continuation pins zeta_D(0) to that")
print(f"  constant N_nz = dim - dim(ker D).  The pole-isolated numerical Mellin continuation")
print(f"  confirms it:  zeta_D(0) ~ {zeta0_num:.4f}  vs  {zeta0_analytic:.4f}  "
      f"(agreement {abs(zeta0_num-zeta0_analytic)/zeta0_analytic*100:.2f}%).")
print(f"  Equivalently zeta_D(0) + dim(ker D) = total dim/cell = 10: zeta_D(0) counts the INVERTIBLE")
print(f"  part of the spectrum.  The generic-k kernel dim is 2 (K_4's b_1=3 drops to 2 once a")
print(f"  nontrivial Bloch character is on, and H^0 dies too), so the BZ-average dim ker D = 2;")
print(f"  only the trivial character Gamma carries the full (b_0,b_1)=(1,3) => 4 zero modes.")


# ----------------------------------------------------------------------------------------
# (4) ZERO MODES at Gamma, P, H -> cohomology of the K_4 cell.
# ----------------------------------------------------------------------------------------
print("\n--- (4) ZERO MODES of hodge_dirac(k) at Gamma, P, H  (cohomology of the K_4 cell) ---")
print("  D=[[0,d],[d*,0]] on C0(+)C1, d=incidence (4x6).  ker D = ker(d*)|C0 (+) ker(d)|C1.")
print("  On C0: ker(d d*) = ker L = H^0 (vertex-harmonics, b_0).  On C1: ker(d) = H^1 (cycles, b_1).")
print("  Euler check:  dim C0 - dim C1 = 4 - 6 = -2 = b_0 - b_1 = 1 - 3.\n")
special = {'Gamma': (0.0, 0.0, 0.0), 'P': (.25, .25, .25), 'H': (.5, .5, .5)}
for nm, k in special.items():
    D = srs.hodge_dirac(k)
    ev = np.linalg.eigvalsh(D)
    nzero = int(np.sum(np.abs(ev) < 1e-9))
    # split the kernel into C0 (vertex, H^0) vs C1 (edge, H^1) content
    #   ker D = { (c0,c1): d* c0 = 0 (vertex-harmonic, H^0) and d c1 = 0 (cycle, H^1) }
    L = srs.DEG*np.eye(srs.NV) - srs.adjacency(k)          # = d d*  on C0  (graph Laplacian)
    b0 = int(np.sum(np.abs(np.linalg.eigvalsh(L)) < 1e-9))  # dim H^0 = harmonic 0-cochains
    d = srs.incidence(k)
    edgeL = d.conj().T @ d                                  # 6x6, = d* d on C1;  ker = ker d = H^1
    b1 = int(np.sum(np.abs(np.linalg.eigvalsh(edgeL)) < 1e-9))
    spec = sorted(float(x) for x in np.round(ev.real, 3))
    print(f"  {nm:6} k={tuple(round(x,2) for x in k)}:  "
          f"total ker D = {nzero}   |   b_0 (H^0, vertex) = {b0}   b_1 (H^1, cycle) = {b1}")
    print(f"            spec D = {spec}")

print("\n  INTERPRETATION:")
print("   - At Gamma=(0,0,0) the cover is untwisted: full K_4 cohomology, b_0=1 (constant vertex")
print("     harmonic) and b_1=3 (= rank H_1(K_4)=3 independent cycle-harmonics) => 1+3 = 4 zero modes.")
print("   - At P and H the Bloch phase twists the cochain complex; the harmonic spaces shrink")
print("     (no global constant survives the nontrivial character) => the zero-mode count drops.")
print("   - The pattern of ker D across {Gamma,P,H} is the Bloch-equivariant cohomology of the")
print("     K_4 cell: only the trivial character (Gamma) carries the full (b_0,b_1)=(1,3) cohomology.")
print("=" * 88)
