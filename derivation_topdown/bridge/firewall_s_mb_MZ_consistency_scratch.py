"""
firewall_s_mb_MZ_consistency_scratch.py  --  FIREWALL consistency check (READ-ONLY math).

QUESTION: does ONE run-displacement s simultaneously account for the residuals of
m_b and M_Z, given the DERIVED Perron band-top correction?

NOT sealed (targets visible) but NO TUNING: we only COMPUTE the two implied s values
and report. No search over N_hub / constants for an s that fits. Nothing in predictions/
is modified.

DERIVED INPUT (from perron_curvature_run_scratch.py, this same dir):
  At the Perron band top h_P = 2 the band-top curvature is EXACT  H = 4*pi^2.
  Resolvent read g = 1/(1 - u h) under a run displacement s:
      g(s)/g_P = 1 - a s^2 + b s^4 - ...,   a = G H/2,  G = u/(1 - u h_P) = u/(1-2u)
                                          => a = -2 pi^2 u/(2u-1)   (prompt's general form)
  Mass-as-(2/3)^L read: leading rel shift = -(4 pi^2 L/(k*-1)) s^2, k*-1 = 2
                                          => = -2 pi^2 L s^2        (prompt's mass-read form)
  The correction is NEGATIVE for u<1/2 (a>0): the read DECREASES with s^2.
  Both live predictions are TOO HIGH (positive residual) -> a negative shift CAN cancel them.

LIVE VALUES (file:line):
  m_b:  pred 4.269825 GeV  (predictions/m_b.py:101, m_b_pred = v*(2/3)^10)
        obs  4.18 +/- 0.03 (predictions/m_b.py:104-105)   resid = +2.1489%
  M_Z:  tree pred 91.5135 GeV (predictions/M_Z.py:149 M_Z_tree)
        pole pred 91.2039 GeV (predictions/M_Z.py:164 M_Z_GeV = M_Z_tree*(1-delta_r))
        obs  91.1876 +/- 0.0021 (predictions/M_Z.py:168-169)
        delta_r = +0.338356% (predictions/delta_r.py:131, c_S=1/12, u=alpha_1=(2/3)^8)
        tree resid = +0.3574%;  POLE (delta_r applied) resid = +0.017862%

ASSUMPTIONS / parametrization choice (stated, not tuned):
  * m_b read is literally (2/3)^L, L = g = 10  => use the mass-read form  2 pi^2 L s^2.
  * M_Z: the leading delta_r oblique is ALREADY in the pole prediction (M_Z.py:164).
    Per the prompt, the run-displacement s accounts for the *REMAINING* pole residual
    (+0.017862%). The delta_r channel sits at u = alpha_1 = (2/3)^8 (delta_r.py:130),
    so we use the general resolvent form a = -2 pi^2 u/(2u-1) at u=(2/3)^8.
  * An ALT interpretation (s REPLACES delta_r, closing the full +0.3574% tree residual)
    is computed too, for honesty -- it does NOT agree (see bottom).
"""
import math

pi2 = math.pi**2
H = 4*pi2                 # DERIVED band-top curvature (exact)
Q = -116*math.pi**4/3     # DERIVED quartic (exact), for the b*s^4 window check

def a_b(u):
    """resolvent-form a, b at h_P=2."""
    G = u/(1-2*u)
    a = G*H/2
    b = (G*H/2)**2 + G*Q/24
    return a, b

# ---------------- m_b ----------------
res_mb = (4.269825440112393 - 4.18)/4.18          # +2.1489%
L_mb = 10
coef_mb = 2*pi2*L_mb                               # = 4 pi^2 L/(k*-1), k*-1=2
s_mb = math.sqrt(res_mb/coef_mb)
# u-equivalent of the L=10 mass read: G = u/(1-2u) = L => u = L/(2L+1)
u_mb = L_mb/(2*L_mb+1)
_, b_mb = a_b(u_mb)

# ---------------- M_Z (remaining pole residual; canonical/live reading) ----------------
res_MZ = (91.20388789194071 - 91.1876)/91.1876    # +0.017862%
u_MZ = (2/3)**8
a_MZ, b_MZ = a_b(u_MZ)
s_MZ = math.sqrt(res_MZ/a_MZ)

print("="*70)
print("IMPLIED RUN-DISPLACEMENT s, BACKED OUT INDEPENDENTLY")
print("="*70)
print(f"m_b : residual +{res_mb*100:.4f}%  (mass-read 2 pi^2 L s^2, L=10)")
print(f"      s_mb = {s_mb:.6f}")
print(f"M_Z : remaining pole residual +{res_MZ*100:.6f}%  (resolvent a=-2pi^2 u/(2u-1), u=(2/3)^8)")
print(f"      s_MZ = {s_MZ:.6f}")
print()
print(f"RATIO  s_mb/s_MZ = {s_mb/s_MZ:.4f}   (s_MZ/s_mb = {s_MZ/s_mb:.4f})")
agree = (1/1.5) <= (s_mb/s_MZ) <= 1.5
print(f"AGREE within factor 1.5?  {agree}")
print()
print("perturbative window (need s <~ 0.13 and |b s^4| < residual):")
print(f"  m_b: s={s_mb:.5f} (<0.13:{s_mb<0.13}); a s^2={coef_mb*s_mb**2:.6f}; "
      f"|b s^4|={abs(b_mb*s_mb**4):.3e} < res({res_mb:.3e})? {abs(b_mb*s_mb**4)<res_mb}")
print(f"  M_Z: s={s_MZ:.5f} (<0.13:{s_MZ<0.13}); a s^2={a_MZ*s_MZ**2:.3e}; "
      f"|b s^4|={abs(b_MZ*s_MZ**4):.3e} < res({res_MZ:.3e})? {abs(b_MZ*s_MZ**4)<res_MZ}")

print()
print("ALT (honesty): if s instead REPLACES delta_r and must close the FULL tree")
res_MZ_tree = (91.51352922794857-91.1876)/91.1876
s_MZ_tree = math.sqrt(res_MZ_tree/a_MZ)
print(f"  M_Z tree residual +{res_MZ_tree*100:.4f}%  -> s_MZ_tree={s_MZ_tree:.6f}; "
      f"s_mb/s_MZ_tree={s_mb/s_MZ_tree:.4f}  (does NOT agree)")
