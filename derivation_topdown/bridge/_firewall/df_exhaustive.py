"""
df_exhaustive — the internal-Dirac mass sector, computed exhaustively (controlled, no agents).

Structure FORCED by the zero-adoption blind construction (construction-space mathematician):
  D_F's odd part = a forced CIRCULANT Y on the 3-fold generation space, with C3-isotype amplitudes
  c = (c0, c1, c2).  First-order + A_F-bimodule => circulant.  J-reality => c0 real, c2 = conj(c1).
  Born weights from Lambda^*(C^3) C3-content (4,2,2)/8 = (1/2,1/4,1/4) => |c0|^2=1/2, |c1|^2=|c2|^2=1/4.
  Residual freedom: ONE external scale (set to 1 here) + ONE physical phase delta in [0, 2pi/3).
  sqrt(mass) eigenvalues:  a_j = c0 + c1 w^j + c2 wbar^j   (w = e^{2pi i/3}); J forces a_j REAL.

We harvest EVERYTHING (per the exhaustiveness mandate), not just the physics targets.
"""
import numpy as np
np.set_printoptions(precision=6, suppress=True)
w = np.exp(2j*np.pi/3)
mult = np.array([4.,2.,2.]); wt = mult/mult.sum()          # Born weights (1/2,1/4,1/4)

def amps(delta, reading="born"):
    c0, r = (np.sqrt(wt[0]), np.sqrt(wt[1])) if reading=="born" else (wt[0], wt[1])
    return c0, r*np.exp(1j*delta), r*np.exp(-1j*delta)      # c2 = conj(c1)  [J-reality]

def spectrum(delta, reading="born"):
    c0,c1,c2 = amps(delta,reading)
    a = np.array([(c0 + c1*w**j + c2*w**(-j)) for j in range(3)])
    return a.real                                          # forced real by J

def invariants(a):
    m = a**2
    p1,p2,p3 = a.sum(), (a**2).sum(), (a**3).sum()
    det = np.prod(a)
    Q_koide   = p2/p1**2                                   # Sum m / (Sum signed a)^2
    Q_magnit  = (a**2).sum()/(np.abs(a).sum())**2          # TRUE Koide: uses |a_j|=sqrt(m)
    return m,p1,p2,p3,det,Q_koide,Q_magnit

print("="*78); print(" D_F MASS SECTOR — exhaustive (Born reading, scale=1)"); print("="*78)
print("C3 content (4,2,2) -> Born weights", wt, "  amplitudes |c|=", np.sqrt(wt))

print("\n--- (1) FORCED vs delta-DEPENDENT: scan delta in [0, 2pi/3) ---")
print(f"{'delta':>8} | {'a0':>8}{'a1':>8}{'a2':>8} | {'p1=Sa':>7}{'p2=Sa2':>7} | {'Q=p2/p1^2':>10}{'Q_|a|':>8} | {'detY':>8}")
for d in np.linspace(0, 2*np.pi/3, 9):
    a = spectrum(d); m,p1,p2,p3,det,Qk,Qm = invariants(a)
    print(f"{d:8.4f} | {a[0]:8.4f}{a[1]:8.4f}{a[2]:8.4f} | {p1:7.4f}{p2:7.4f} | {Qk:10.5f}{Qm:8.5f} | {det:8.4f}")

print("\n  => p1=Sum a = 3*c0 = 3/sqrt2 and p2=Sum a^2 = 3 are delta-INVARIANT (Parseval): FORCED.")
print("     Q = p2/p1^2 = 2/3 EXACTLY for all delta (signed power-sum Koide): FORCED.")
print("     BUT the TRUE Koide Q_|a| (using sqrt(m)=|a_j|) DRIFTS with delta -> only =2/3 where all a_j>0.")
print("     det Y is the FIRST invariant the phase touches (delta-dependent): the hierarchy lives here.")

print("\n--- (2) the mass HIERARCHY as a function of delta (the one physical knob) ---")
print(f"{'delta':>8} | {'m0:m1:m2 (normalized to m_min)':>40} | {'m_mid/m_min':>11}{'m_max/m_min':>11}")
for d in [0.0, 1/9, 2/9, 1/3, 2*np.pi/9, 0.6]:
    a = spectrum(d); m = np.sort(a**2)
    mmin = m[m>1e-12].min() if (m>1e-12).any() else m.min()
    r = m/mmin
    print(f"{d:8.4f} | {str(np.round(r,2)):>40} | {r[1]:11.2f}{r[2]:11.2f}")

print("\n  Observed charged-lepton ratios for reference: m_mu/m_e=206.77, m_tau/m_e=3477.2")
# find delta best matching the observed lepton ratios
target = np.array([1.0, 206.77, 3477.2])
best=None
for d in np.linspace(1e-4, 2*np.pi/3, 20000):
    a=spectrum(d); m=np.sort(a**2)
    if (m>1e-9).all():
        r=m/m.min()
        err=np.sum((np.log(r)-np.log(target))**2)
        if best is None or err<best[1]: best=(d,err,r)
print(f"  best-fit delta = {best[0]:.5f} rad (= {best[0]:.5f}; 2/9={2/9:.5f}) -> ratios {np.round(best[2],2)}")

print("\n--- (3) BOTH readings at delta=0 (the construction fork) ---")
for rd in ["born","direct"]:
    a=spectrum(0.0,rd); m,p1,p2,p3,det,Qk,Qm=invariants(a)
    print(f"  {rd:7}: sqrt(m)={np.round(a,4)}  m-ratio={np.round((a**2)/(a**2).max(),5)}  Koide={Qk:.5f}")

print("\n--- (4) SYMMETRY / CHARGE byproducts ---")
# Y is circulant => commutes with the C3 cyclic shift S
S = np.roll(np.eye(3),1,axis=0)
c0,c1,c2 = amps(2/9)
Y = np.array([[c0,c2,c1],[c1,c0,c2],[c2,c1,c0]])           # circulant first column (c0,c1,c2)
print("  [Y,S]=0 (circulant commutes with C3 shift)?", np.allclose(Y@S,S@Y))
print("  charges = C3 character labels of the eigenvectors:")
for j in range(3):
    print(f"    eigenvector j={j}: C3 charge = {j}/3 = {j/3:.4f}   (eigenphase w^{j})")
print("  => residual symmetry of the mass operator = C3 (the triality); charges quantized in 1/3.")
print("     The single phase delta is C3-covariant (defined mod 2pi/3) -> physical value in [0,2pi/3).")

print("\n" + "="*78)
print(" LANDING: the whole mass sector = {(4,2,2) Born weights [forced], ONE scale [external],")
print("          ONE phase delta in [0,2pi/3) [physical]}.  Forced dimensionless: Koide 2/3 (signed).")
print("          Everything else (the hierarchy, det, true Koide) is a function of delta ALONE.")
print("="*78)
