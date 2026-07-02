#!/usr/bin/env python3
"""
m_nu_close_y_nu_and_sqrtR_scratch_2026-06-23.py   (SCRATCH — derives, no fits)

GOAL (two open neutrino masses):
  m_nu3 = +0.87% (+2.18 sig_PDG),  m_nu2 = +2.37% (+1.91 sig_PDG)

TASK 1 — retire y_nu = 1.  Is the neutrino Dirac Yukawa a FORCED shell read
         (analogue of y_tau = alpha1_full/k*^2), or genuinely the natural-scale
         unit 1?  Compute the forced y_nu, the resulting m_nu3, and its sigma.

TASK 2 — pin the sqrt(R) step.  Is the extra ~1.5% on m_nu2 (beyond m_nu3's own
         residual) an observable-convention error (sqrt applied to m vs m^2),
         a sub-leading R-correction, or a genuine formula gap?

TASK 3 — net: does (forced y_nu) + (sqrt-R fix) bring both to <= 1 sig_PDG?

DISCIPLINE: derive, do NOT fit to targets.  Every load-bearing value is
file:line-sourced.  If a thing is genuinely 1 or genuinely uncloseable, say so.

All forced inputs READ from the framework (no new structure):
  h = (sqrt3 + i sqrt5)/2,  |h|^2 = k*-1 = 2          predictions/h_walker_eigenvalue.py:45,95
  Im(h)/|h|^2 = sqrt5/4  (Class-1 amplitude DC coeff) predictions/dark_extraction_map.py:108,161,186
  alpha1_bare = (2/3)^8                               predictions/alpha_1.py:21
  alpha1_full = (5/3)(2/3)^8                          predictions/alpha_1_full.py:5
  y_tau       = alpha1_full / k*^2                    predictions/y_tau.py:21
  m_nu3 = 12 * M_Pl / sqrt(N_hub) = y_nu^2 v^2 / M_R  predictions/m_nu3.py:168,Step3
  R = 228/7  (Ihara, K4 topological, NO dark corr)    predictions/R_nu_splitting.py:47
  m_nu2 = m_nu3 / sqrt(R)                              predictions/m_nu2.py:118
"""
import math, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'predictions'))

# ---- forced shell reads --------------------------------------------------
k = 3
a1b = (2/3)**8                 # alpha_1_full.py / alpha_1.py
a1f = (5/3)*a1b
h = complex(math.sqrt(3)/2, math.sqrt(5)/2)
assert abs(abs(h)**2 - 2) < 1e-14            # |h|^2 = k*-1 = 2
s = h.imag/abs(h)**2                          # Im(h)/|h|^2 = sqrt5/4 (Class-1 amplitude)
assert abs(s - math.sqrt(5)/4) < 1e-14
y_tau = a1f/k**2

# ---- observed (NuFIT 6.0, normal ordering, m_nu1 = 0) --------------------
dm21, sdm21 = 7.49e-5, 0.19e-5                # m_nu2_derivation.md:18
dm31, sdm31 = 2.513e-3, 0.020e-3             # m_nu3.py:191,271   (= Dm2_3l, the NO ref splitting)
m3_obs = math.sqrt(dm31); s3 = 0.5*sdm31/m3_obs
m2_obs = math.sqrt(dm21); s2 = 0.5*sdm21/m2_obs

# ---- framework prediction at y_nu = 1 ------------------------------------
m3_y1 = 0.0505651            # m_nu3.py:168 numeric (12 M_Pl/sqrt(N_hub), G_F-pinned N_hub)
R_fw = 228/7
m2_y1 = m3_y1/math.sqrt(R_fw)

def sig(pred, obs, sg): return (pred-obs)/sg
def pct(pred, obs): return (pred/obs-1)*100

print("="*78)
print("BASELINE (y_nu = 1, R = 228/7)")
print("="*78)
print(f"  m_nu3 = {m3_y1*1e3:8.4f} meV  obs {m3_obs*1e3:.4f}+-{s3*1e3:.4f}  "
      f"{pct(m3_y1,m3_obs):+.3f}%  {sig(m3_y1,m3_obs,s3):+.2f} sig")
print(f"  m_nu2 = {m2_y1*1e3:8.4f} meV  obs {m2_obs*1e3:.4f}+-{s2*1e3:.4f}  "
      f"{pct(m2_y1,m2_obs):+.3f}%  {sig(m2_y1,m2_obs,s2):+.2f} sig")
print(f"  y_tau (charged-lepton forced shell read) = alpha1_full/k*^2 = {y_tau:.6e}")

print("\n"+"="*78)
print("TASK 1 — is y_nu a FORCED shell read (like y_tau), or the unit 1?")
print("="*78)
# The needed y_nu (m_nu3 ∝ y_nu^2):
need_y2 = m3_obs/m3_y1                         # factor on y_nu^2 (= factor on m_nu3)
need_y  = math.sqrt(need_y2)
print(f"  observed m_nu3 needs y_nu = {need_y:.5f}  (y_nu^2 factor {need_y2:.5f}, "
      f"{(need_y2-1)*100:+.3f}% on m_nu3)")
print(f"  charged-lepton shell read y_tau = {y_tau:.4e}  ~15x BELOW 1.")
print()
print("  (a) Is y_nu a SMALL shell read (alpha1-scale, like y_tau)?  NO.")
print("      The framework's own persistence-Yukawa law (y_nu_persistence_ceiling")
print("      _2026-05-19.py) has a STRUCTURAL CEILING = alpha1_full = "
      f"{a1f:.4f}; required y_nu~1 is ~{1/a1f:.0f}x above it.  y_t(GUT)=1 is the")
print("      UN-SUPPRESSED NATURAL-SCALE UNIT the whole Koide ladder is measured")
print("      AGAINST, not an amplitude on the shell.  => leading y_nu = 1 is GENUINE.")
print()
print("  (b) Is there a forced SUB-LEADING shell correction to y_nu = 1?")
print("      The framework classifies m_nu2,m_nu3 as Class-1 AMPLITUDE observables")
print("      (dark_extraction_map.py:108,161): coeff = Im(h)/|h|^2 = sqrt5/4.")
print("      Test the Class-1 amplitude DC read at every structurally-clean leg form:")
def report_m3(label, factor_on_m3):
    m = m3_y1*factor_on_m3
    print(f"      {label:42s} m_nu3={m*1e3:7.4f} meV  {pct(m,m3_obs):+6.3f}%  "
          f"{sig(m,m3_obs,s3):+5.2f} sig")
report_m3("y_nu=1                       (baseline)", 1.0)
report_m3("y_nu=(1-s*a1b)^1 on y, ^2 on m", (1-s*a1b)**2)
report_m3("y_nu=(1-s*a1f)^1 on y, ^2 on m", (1-s*a1f)**2)
report_m3("m*(1-s*a1b)  (mass-level, 1x)", (1-s*a1b))
report_m3("m*(1-s*a1f)  (mass-level, 1x)", (1-s*a1f))
report_m3("m*(1-s*a1f/(1-a1f)) [m_nu3.py:81 '-1.4%']", (1-s*a1f/(1-a1f)))
report_m3("m*(1-a1f^2)  (Family-D Higgs-leg style)", (1-a1f**2))
report_m3("m*(1-a1b^2)  (Family-D Higgs-leg style)", (1-a1b**2))
need_on_m = m3_obs/m3_y1
print(f"\n      EXACT factor needed on m_nu3 = {need_on_m:.5f} ({(need_on_m-1)*100:+.3f}%);")
print(f"      as (1-x)^2 on y: x={1-math.sqrt(need_on_m):.5f} (cf s*a1b={s*a1b:.5f}, s*a1f={s*a1f:.5f})")
print(f"      as (1-x)   on m: x={1-need_on_m:.5f} (cf a1b={a1b:.5f}, a1f^2={a1f**2:.5f})")
print()
print("  VERDICT (Task 1): the full Class-1 sqrt5/4*alpha amplitude DC is FAR TOO BIG")
print("    (overshoots to -1.3%..-6%).  It is NOT a separate small correction on")
print("    y_nu=1; it is already structurally implicated in the bare spectral-gap")
print("    scale (the m_nu3.py 'double-count' warning is correct).  NO forced shell")
print("    read produces y_nu ~ 0.9957.  Leading y_nu = 1 is the genuine natural-")
print("    scale UNIT; the +0.87% on m_nu3 is the N_hub-anchor / unit residual,")
print("    NOT a missing forced Yukawa factor.")
print(f"    [For reference: N_hub from G_F-pin (8.395e60) gives +0.87%; the")
print(f"     m_tau-anchored N_hub (8.435e60) gives +0.63% (+1.58 sig) — same datum,")
print(f"     different SI calibration of the ONE adopted scale.]")

print("\n"+"="*78)
print("TASK 2 — pin the sqrt(R) step")
print("="*78)
# (a) convention check: is sqrt applied to m or m^2 consistently?
R_obs = (m3_obs/m2_obs)**2
print(f"  (a) CONVENTION.  R := Dm2_31/Dm2_21 = m_nu3^2/m_nu2^2 (m_nu1=0).")
print(f"      File computes m_nu2 = m_nu3/sqrt(R)  (m_nu2.py:118).  This is")
print(f"      DIMENSIONALLY CORRECT: R is on m^2, sqrt(R) brings it to m.  Cross-")
print(f"      check m_nu2.py:208 asserts (m_nu3/m_nu2)^2 == R by construction.")
print(f"      => NOT an m-vs-m^2 convention error.  R_obs(from sqrt-masses)={R_obs:.3f}.")
print()
# Exact error decomposition
f_m3 = m3_y1/m3_obs
f_R  = math.sqrt(R_obs/R_fw)
sgR  = R_obs*math.sqrt((sdm31/dm31)**2+(sdm21/dm21)**2)
print(f"  EXACT decomposition of the m_nu2 error (+{pct(m2_y1,m2_obs):.3f}%):")
print(f"      inherited from m_nu3 : {(f_m3-1)*100:+.3f}%")
print(f"      from R being low     : {(f_R-1)*100:+.3f}%   (R_fw={R_fw:.4f} vs "
      f"R_obs={R_obs:.3f}+-{sgR:.3f})")
print(f"      product              : {(f_m3*f_R-1)*100:+.3f}%  (= m_nu2 dev, check)")
print(f"      => the 'extra ~1.5%' IS exactly the R discrepancy.  R = 228/7 sits")
print(f"         {(R_fw-R_obs)/sgR:+.2f} sig LOW vs the observed ratio.")
print()
# (b)/(c): is there a forced sub-leading R correction?
print(f"  (b/c) SUB-LEADING R-CORRECTION?  R_theorem.md (cited R_nu_splitting.py:47)")
print(f"        states R is a TOPOLOGICAL invariant of K4, IMMUNE to dark")
print(f"        corrections (R_nu_splitting_derivation.md:229).  Adding an O(alpha1)")
print(f"        bump to hit data would CONTRADICT the framework's own theorem and")
print(f"        be a fit.  (For the record, R*(1+a1b) -> {R_fw*(1+a1b):.3f}, +0.33 sig;")
print(f"        but a1b has NO derivation as an R-correction — REJECTED as a fit.)")
print(f"        => R stays 228/7.  The 'extra 1.5%' is the leading-Ihara-vs-data")
print(f"           gap (-1.1 sig on R), a GENUINE residual, NOT a formula/convention")
print(f"           bug and NOT closeable by a forced read.")
print()
# The honest m_nu2 comparison: isolate from the m_nu3 anchor.
m2_from_obs_m3 = m3_obs/math.sqrt(R_fw)
print(f"  ISOLATED R-only m_nu2 (anchor m_nu3 to truth): {m2_from_obs_m3*1e3:.4f} meV  "
      f"{pct(m2_from_obs_m3,m2_obs):+.3f}%  {sig(m2_from_obs_m3,m2_obs,s2):+.2f} sig")

print("\n"+"="*78)
print("TASK 3 — NET before/after")
print("="*78)
print("  m_nu3:")
print(f"    before (y_nu=1, G_F-N_hub) : {pct(m3_y1,m3_obs):+.3f}%  {sig(m3_y1,m3_obs,s3):+.2f} sig")
m3_mt = 12* (1.22089e19)*1e9 / math.sqrt(8.435e60)  # m_tau-anchored N_hub
print(f"    after  (y_nu=1, m_tau-N_hub): {pct(m3_mt,m3_obs):+.3f}%  {sig(m3_mt,m3_obs,s3):+.2f} sig")
print(f"      (y_nu retirement: leading y_nu=1 is FORCED as the unit; no forced")
print(f"       shell read shifts it; residual is the ONE adopted scale N_hub.)")
print("  m_nu2:")
print(f"    before (full chain)        : {pct(m2_y1,m2_obs):+.3f}%  {sig(m2_y1,m2_obs,s2):+.2f} sig")
m2_mt = m3_mt/math.sqrt(R_fw)
print(f"    after  (m_tau-N_hub, R=228/7): {pct(m2_mt,m2_obs):+.3f}%  {sig(m2_mt,m2_obs,s2):+.2f} sig")
print(f"    R-only (m_nu3 exact)       : {pct(m2_from_obs_m3,m2_obs):+.3f}%  "
      f"{sig(m2_from_obs_m3,m2_obs,s2):+.2f} sig  <- irreducible (R is 228/7, no DC)")
print()
print("  CONCLUSION: the sector does NOT close to <=1 sig by retiring y_nu=1.")
print("   - Task 1: y_nu=1 is GENUINE (the natural-scale unit), not a missing shell")
print("     read; the +0.87% m_nu3 residual is the N_hub anchor (1.58 sig at m_tau-N_hub).")
print("   - Task 2: the sqrt(R) step is CORRECT (no convention bug); the extra 1.5%")
print("     on m_nu2 is the leading R=228/7 sitting -1.1 sig below the data ratio,")
print("     and R takes NO dark correction by the framework's own theorem.")
print("   - Honest endpoint: m_nu3 ~ +1.6..2.2 sig (N_hub-anchor), m_nu2 ~ +1.2..1.9 sig")
print("     (R-gap-dominated).  Both residuals are KNOWN, NAMED, and not fits; neither")
print("     is closed by a forced y_nu or a sqrt-R fix.")
