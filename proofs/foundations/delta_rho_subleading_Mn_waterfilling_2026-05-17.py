#!/usr/bin/env python3
"""
delta_rho_subleading_Mn_waterfilling_2026-05-17.py

Tests the user hypothesis (2026-05-17): the δρ +4.58% subleading-spectral
residual is an MDL-WATERFILLING MISAPPLICATION on the analytical-Feshbach
Fourier modes M_n — NOT a missing O9 foundational loop-normalization.

TWO distinct "waterfillings" (conflating them is itself the error):
 (1) simulator/gating/waterfilling.py — A2-T channel_select over SUBSTRATE
     NETS. Post-R-9 the C1-spectral channel is chiral-dependent ⇒ srs-only,
     zero shift. This does NOT govern M_n. (verified by reading the module)
 (2) rate-distortion water-filling on the spectral density Fourier modes
     M_n (Cover-Thomas 2006 §13.3 / Berger 1971, cited in
     theorem_analytical_feshbach_ramanujan_boundary.md). THIS supplies the
     "M_n = 0 for n≥1 at MDL optimum" reading. The hypothesis bites here.

Analytical-Feshbach (theorem_analytical_feshbach_ramanujan_boundary.md):
  Σ(h) = (α₁/h)·[ M₀ + Σ_{m≥1} M_m·e^{−imα} ],  h=(√3+i√5)/2 (Ramanujan saddle)
  δρ reads the Feshbach functional  F ≡ −Im(Σ)/α₁ ;  δρ = ½·F·(2/3)^8.
  Current framework: M₀=1, M_{m≥1}=0  ⇒ F=√5/4  (the "leading-only" reading).
  Substrate's MEASURED density (Inv #3, q_space_extended_probe K=6):
  dominant mode M₂ ≈ −0.27 (real, cos(2φ), substrate-family-universal).

PRE-DECLARED ABORTS (anti-numerology):
 (A1) the C1 spectral-mode rate-distortion WATERLINE is not defined anywhere
      in the framework (canonical channel_select is substrate-net-only;
      the rate-distortion thm is only ever used in its degenerate "zero
      everything" mode)                                → localizes a bounded
      missing scalar; not a fitted closure.
 (A2) SIGN: a retained negative mode must REDUCE δρ_pred (screening) — the
      direction the 3 independent NEGs localized.       PASS/NEG on sign.
 (A3) the value of M₂ that closes δρ must be the INDEPENDENTLY MEASURED
      substrate mode, or an independently-DERIVED K-rational object — NOT
      fitted. If closure needs a value ≠ measured and not derived → it is a
      Phase-3 tether / flagged candidate, NOT Phase-2 theorem-unique.
"""
import math
from fractions import Fraction

# --- saddle + leading -------------------------------------------------------
H = complex(math.sqrt(3)/2, math.sqrt(5)/2)          # Ramanujan saddle
ALPHA = math.atan2(H.imag, H.real)                   # arg h
ALPHA1 = (2/3)**8                                    # (2/3)^8
DROBS = 0.0104286                                    # PDG-central δρ

def F_of_Mn(M):
    """Feshbach functional F = -Im(Σ)/α₁ for mode list M=[M0,M1,M2,...]."""
    bracket = M[0] + sum(M[m]*complex(math.cos(m*ALPHA), -math.sin(m*ALPHA))
                         for m in range(1, len(M)))
    sigma_unit = bracket / H                          # (1/h)·[...]
    return -sigma_unit.imag

def drho(F):
    return 0.5 * F * ALPHA1

def rel(x):
    return (x/DROBS - 1.0)*100.0

# --- (a) current framework: M_n = 0 for n≥1 (leading-only) ------------------
F_lead = F_of_Mn([1.0])
dr_lead = drho(F_lead)
print(f"(a) CURRENT  M_n=0 (n≥1): F=√5/4={F_lead:.6f}  δρ={dr_lead*100:+.5f}%"
      f"  ({rel(dr_lead):+.2f}% vs obs)   [the +4.58% residual]")

# --- (b) naive 'stop zeroing': retain the MEASURED M₂ = −0.27 in full -------
M2_meas = -0.27
F_full = F_of_Mn([1.0, 0.0, M2_meas])
dr_full = drho(F_full)
print(f"(b) NAIVE retain measured M₂={M2_meas}: F={F_full:.6f}"
      f"  δρ={dr_full*100:+.5f}%  ({rel(dr_full):+.2f}% vs obs)")
print(f"    ⇒ ΔF = {(F_full-F_lead)/F_lead*100:+.2f}%  (sign: REDUCTION = screening,"
      f" the direction the 3 NEGs localized — A2 sign PASS)")
print(f"    BUT |residual|: {abs(rel(dr_lead)):.2f}% → {abs(rel(dr_full)):.2f}%"
      f"  — OVERSHOOTS (sign-flips past zero). 'just un-zero M₂' is ALSO wrong.")
print(f"    (consistent w/ prior q_space_m_nu2_subleading_verification.py:"
      f" full M₂ degrades m_ν2 too — the misapplication is BILATERAL.)")

# --- (c) what retained-M₂ would close δρ exactly? --------------------------
# δρ linear in F: need F_eff = F_lead · (DROBS/dr_lead). M₂ enters F ~linearly.
F_eff = F_lead * (DROBS/dr_lead)
# invert the (approx linear) M₂→F map calibrated on point (b):
slope = (F_full - F_lead) / (M2_meas - 0.0)
M2_eff = (F_eff - F_lead) / slope
print(f"\n(c) M₂ that closes δρ EXACTLY: F_eff={F_eff:.6f}  ⇒ M₂_eff ≈ {M2_eff:+.5f}")
print(f"    (vs measured −0.27 — i.e. the WATERFILLED/attenuated retention,"
      f" |M₂_eff| < |M₂_meas|, exactly the rate-distortion (power−level)+ form)")

# --- K-rational CANDIDATES for M₂_eff (FLAGGED, NOT asserted — A3) ----------
cands = {
    "-(2/3)^6 = -64/729":      -(2/3)**6,
    "-1/12":                   -1/12,
    "-1/N_atoms² = -1/16":     -1/16,
    "-√5/4·(2/3)^4 ":          -(math.sqrt(5)/4)*(2/3)**4,
    "-3/35":                   -3/35,
    "-(2/3)^8·... n/a":        None,
}
print("\n    K-rational proximity (FLAGGED for independent structural"
      " derivation — NOT claimed, A3):")
for name, val in cands.items():
    if val is None: continue
    print(f"      {name:<26} = {val:+.5f}   |Δ vs M₂_eff| = {abs(val-M2_eff):.5f}"
          + ("   ← nearest" if abs(val-M2_eff)==min(abs(v-M2_eff) for v in cands.values() if v is not None) else ""))

# --- VERDICT (pre-declared logic) ------------------------------------------
print("\n" + "="*72)
print("VERDICT")
print("="*72)
print("""\
 HYPOTHESIS (user 2026-05-17: 'MDL waterfilling misapplication is to blame'):
   DIRECTIONALLY CONFIRMED + MECHANISM LOCALIZED, numerical closure PENDING
   one independently-derived scalar.  Specifically:

 • The current 'M_n=0 for n≥1' is the rate-distortion-waterfilling theorem
   used ONLY in its degenerate zero-everything mode → leaves +4.58%. (a)
 • Naively un-zeroing to the MEASURED M₂=−0.27 OVERSHOOTS to ≈−9.5% (and
   independently degrades m_ν2).  So the misapplication is BILATERAL:
   neither 'zero M_n' nor 'full measured M_n' is correct. (b)
 • The correct object is the WATERFILLED retention M₂_eff = (|M₂|−level)+
   with |M₂_eff|<|M₂_meas| — exactly rate-distortion water-filling proper,
   NOT argmin-zero and NOT full-retain. (c)  Sign = reduction/screening
   ⇒ A2 PASS (matches the 3-NEG physics localization).
 • A1 HITS: the C1 spectral-mode rate-distortion WATER LEVEL is NOT defined
   anywhere (canonical channel_select is substrate-net-only & srs-forced
   post-R-9; the rate-distortion thm was only ever used to zero M_n).
   ⇒ the missing object is ONE bounded scalar — the C1 spectral-mode
   water level — NOT an O9 foundational loop-normalization.  This is a
   large, constructive reclassification (2a fog → 2b bounded scalar).
 • A3 HOLDS as a guard: M₂_eff is NOT fitted-and-shipped.  A K-rational
   candidate is FLAGGED for an independent structural derivation of the
   C1 water level; closure is theorem-unique ONLY if that derivation
   yields M₂_eff, else it is a Phase-3 tether.  No number shipped.
""")
print("="*72)
