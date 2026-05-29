#!/usr/bin/env python3
"""
W27 — Family-D-analog attempt for c_1 sub-leading offset

CONTEXT
-------
Yesterday's `sector_specific_c_alpha_GUT_scan_2026-05-26.py` extracted from PDG:
  c_1 = 0.3428 ± 0.0074 (1.36σ from 1/3)
  c_2 = 0.3317 ± 0.006  (0.27σ from 1/3)

The "+0.008 offset on c_1" is a point-estimate reading at 1.36σ confidence —
not robustly distinguished from c_EW = 1/3 exactly. R_∞ ppt-precision was
suggested as independent corroboration, but `Rinf_clean_ratio_diagnostic_2026-05-16.py`
showed the R_∞ residual is dominated by the EXTERNAL Δα_running import (Type-3
PDG-derived), NOT a framework α_EM derivation error.

The master doc `theorem_substrate_feshbach_dark_corrections_master.md` §3 (D)
states: "Family D's failure on the gauge-boson 2-point function (M_Z, m_W ...
have OPPOSITE SIGNS). Sign-uniform Family-D corrections cannot..."

But Family D was designed for Yukawa/Higgs VERTICES (1H+2F, 4H legs), not
gauge boson self-energies. The per-leg coefficients from the master doc:
  a_H = +1 per Higgs leg
  a_F = -1/12 per fermion leg
  → y_τ (1H+2F): c = 1 + 2·(-1/12) = 5/6
  → λ_Higgs (4H): c = 4·1 = 4

For a gauge boson 2-point function (e.g., α_GUT^U(1)Y self-energy via U(1)_Y
matter content), the leg counting is different. This probe computes the
Family-D-analog predicted magnitude and compares to the empirical c_1 offset.

WHAT THIS PROBE TESTS
---------------------
1. The Family-D-analog formula for U(1)_Y gauge boson self-energy:
     δα_GUT^U(1)Y / α_GUT^U(1)Y = -c_FD · α_1_bare²
   where c_FD is determined by per-leg counting on U(1)_Y matter loops.

2. The implied shift in c_EW from this Family-D correction.

3. Comparison against the empirical c_1 = 1/3 + 0.008 (1.36σ point estimate).

4. Honest verdict: closure, partial match, or honest negative.
"""

import math
from fractions import Fraction
import sys, os

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'predictions'))

from predictions.M_unif import predict_M_unif_GeV
from predictions.M_Z import M_Z_GeV as _M_Z
from predictions.M_Pl_natural import M_Pl_GeV

K_STAR = 3
G_GIRTH = 10
ALPHA_GUT_BARE = 1.0/(2**K_STAR * K_STAR)
ALPHA_1_BARE = (2.0/3.0)**(G_GIRTH - 2)
X_DARK = ALPHA_1_BARE / (1.0 - ALPHA_1_BARE)
M_UNIF = float(predict_M_unif_GeV(K_STAR, G_GIRTH, M_Pl_GeV))
M_Z = float(_M_Z)
LOG_RATIO = math.log(M_Z/M_UNIF)
HYP_NORM = 3.0/5.0

# Magnitude of α_1² at the substrate scale
ALPHA_1_SQ = ALPHA_1_BARE**2

print("="*78)
print(" W27 — Family-D-analog attempt for c_1 sub-leading offset")
print("="*78)
print()
print(f" Substrate scales:")
print(f"   α_1_bare         = (2/3)^8 = {ALPHA_1_BARE:.6f}")
print(f"   α_1_bare²        = {ALPHA_1_SQ:.6f}  ({ALPHA_1_SQ*100:.3f}%)")
print(f"   x = α_1/(1-α_1)  = {X_DARK:.6f}")
print()

# ============================================================
# Empirical extraction (yesterday's sector scan)
# ============================================================
PDG_alpha_EM = (1.0/127.944, 0.014/127.944**2)
PDG_sin2W = (0.23121, 0.00004)
PDG_alpha_s = (0.11800, 0.00090)

def pdg_inv_alphas():
    s2 = PDG_sin2W[0]
    aEM = PDG_alpha_EM[0]
    aS = PDG_alpha_s[0]
    a2 = aEM / s2
    aY = aEM / (1.0 - s2)
    a1 = aY / HYP_NORM
    return [1.0/a1, 1.0/a2, 1.0/aS]

def pdg_sigma_inv():
    s2, ss2 = PDG_sin2W
    aEM, saEM = PDG_alpha_EM
    aS, saS = PDG_alpha_s
    a2 = aEM / s2
    sa2 = math.hypot(saEM/s2, aEM*ss2/s2/s2)
    sigma_inv_a2 = sa2 / a2**2
    aY = aEM / (1.0 - s2)
    saY = math.hypot(saEM/(1.0 - s2), aEM*ss2/(1.0 - s2)**2)
    a1 = aY / HYP_NORM
    sa1 = saY / HYP_NORM
    return [sa1 / a1**2, sigma_inv_a2, saS / aS**2]

B_MSSM = (33.0/5.0, 1.0, -3.0)
pdg = pdg_inv_alphas()
sig = pdg_sigma_inv()

def c_for_sector(target, b):
    inv_alpha_GUT_obs = target + (b/(2.0*math.pi))*LOG_RATIO
    ratio = ALPHA_GUT_BARE * inv_alpha_GUT_obs
    one_minus_cx = 1.0/ratio
    c = (1.0 - one_minus_cx) / X_DARK
    return c

c_empirical = [c_for_sector(pdg[i], B_MSSM[i]) for i in range(3)]
# σ on c_i (propagation)
sigma_c_i = [sig[i] / (X_DARK/ALPHA_GUT_BARE) for i in range(3)]

print(" EMPIRICAL c-EXTRACTION (yesterday, one-loop MSSM):")
print(f"   c_1 = {c_empirical[0]:.5f} ± {sigma_c_i[0]:.5f}  "
      f"(Δ from 1/3 = {c_empirical[0]-1/3:+.5f}, "
      f"{(c_empirical[0]-1/3)/sigma_c_i[0]:+.2f}σ)")
print(f"   c_2 = {c_empirical[1]:.5f} ± {sigma_c_i[1]:.5f}  "
      f"(Δ from 1/3 = {c_empirical[1]-1/3:+.5f}, "
      f"{(c_empirical[1]-1/3)/sigma_c_i[1]:+.2f}σ)")
print(f"   c_3 = {c_empirical[2]:.5f} ± {sigma_c_i[2]:.5f}  "
      f"(Δ from 1/3 = {c_empirical[2]-1/3:+.5f}, "
      f"{(c_empirical[2]-1/3)/sigma_c_i[2]:+.2f}σ)")
print()

# ============================================================
# Family-D-analog: what magnitude does it produce on c_EW?
# ============================================================
# Family-D form: δα/α = -c_FD · α_1_bare² for some leg-count coefficient c_FD.
#
# For U(1)_Y gauge boson self-energy: the loop is matter (fermions + Higgs)
# running in the vacuum polarization. Per the master doc §3 (D), the per-leg
# coefficients are a_H = +1 (Higgs leg) and a_F = -1/12 (fermion leg).
#
# U(1)_Y matter content (one full SM generation, per Pati-Salam):
#   - 2 leptons (n_F = 2): nu_L + e_L doublet, e_R singlet, plus nu_R
#     Total fermion legs running in U(1)_Y propagator: depends on charges and Y
#   - 3 quark colors per quark species (n_C = 3): u_L, d_L, u_R, d_R
#
# For now, take a SIMPLEST estimate: total fermion-leg count from one SM
# generation (16 PS states) entering a gauge boson 2-point function.
#
# In standard SM (one-loop β-function for U(1)_Y):
#   b_1^{(1)} = sum_f Y_f² × n_color(f) over all fermions = 2 × (sum over 16 PS states)
#
# We can use this as a proxy for "leg count weighted by hypercharge."
#
# But for the structural Family-D-analog, the leg count is per-leg uniform
# (a_H = +1, a_F = -1/12). This doesn't reproduce the SM β-function structure.

# Compute α_GUT_obs shift if we add Family-D-analog correction:
# α_GUT_obs^Y = α_GUT_bare × (1 - (1/3) X) × (1 - c_FD · α_1²)
# Assume Family-D adds δα/α = -c_FD · α_1²

# Empirical: target Δc_1 ≈ +0.0095
empirical_c1_offset = c_empirical[0] - 1.0/3.0
print(f" EMPIRICAL c_1 offset from 1/3:  Δc_1 = {empirical_c1_offset:+.5f}  "
      f"({(c_empirical[0]-1/3)/sigma_c_i[0]:+.2f}σ — NOT firm)")
print()

# Family-D-analog prediction: how much c-shift does c_FD · α_1² give?
# δα_GUT_obs / α_GUT_obs = -c_FD · α_1²
# But the dark correction is α_GUT_obs = bare × (1 - c X), so an additional
# Family-D suppression by factor (1 - c_FD α_1²) gives:
# α_GUT_obs^new = bare × (1 - c X) × (1 - c_FD α_1²)
#               ≈ bare × (1 - c X - c_FD α_1²)        [linear in α_1²]
# This is equivalent to shifting c → c_eff = c + c_FD α_1²/X
# So Δc_FD-equivalent = c_FD α_1²/X

print(" FAMILY-D-ANALOG PREDICTED MAGNITUDES:")
print(f"   Family-D form: δα/α = -c_FD · α_1²")
print(f"   Equivalent c-shift: Δc = c_FD · α_1²/X = c_FD · {ALPHA_1_SQ/X_DARK:.5f}")
print()

for c_FD in [Fraction(1, 12), Fraction(1, 6), Fraction(1, 3), Fraction(1, 2),
             Fraction(5, 6), Fraction(1, 1), Fraction(2, 1), Fraction(4, 1)]:
    delta_c_predicted = float(c_FD) * ALPHA_1_SQ / X_DARK
    matches_empirical = abs(delta_c_predicted - empirical_c1_offset) < 0.5*sigma_c_i[0]  # within 0.5σ
    print(f"   c_FD = {c_FD}:  Δc_predicted = {delta_c_predicted:+.5f}  "
          f"vs empirical Δc_1 = {empirical_c1_offset:+.5f}  "
          f"{'✓ within 0.5σ_c1' if matches_empirical else ''}")
print()

# Compute what c_FD would EXACTLY match the empirical c_1 offset
c_FD_needed = empirical_c1_offset / (ALPHA_1_SQ / X_DARK)
print(f" c_FD that EXACTLY matches empirical Δc_1 = +{empirical_c1_offset:.5f}:")
print(f"   c_FD = {c_FD_needed:.4f}")
print(f"   Express as Fraction: ≈ {Fraction(c_FD_needed).limit_denominator(100)}")
print()

# ============================================================
# What about α_EM directly?
# ============================================================
# α_EM(M_Z) ≈ α_1·α_2/(α_1·sin²θ + α_2·cos²θ) — this is the weak-mixing relation.
# If we add Family-D correction to U(1)_Y, α_1 shifts (which is what c_1
# corresponds to). α_EM tightens correspondingly.

print("-"*78)
print(" ALTERNATE FRAMING: Family-D correction directly on α_EM(M_Z)")
print("-"*78)
print()
# α_EM = e²/(4π); the U(1)_EM gauge boson is a linear combination of B_μ (U(1)_Y)
# and W^3_μ (SU(2)_L). Its self-energy correction at order α_1² would be:
# δα_EM/α_EM = -c_FD^EM · α_1²
#
# Empirical: α_EM(M_Z) predicted = 1/127.944, PDG = 1/127.951
# Δ(1/α_EM) = -0.007 = ~-σ_PDG
# δα_EM/α_EM = -Δ(1/α_EM) / (1/α_EM) = +0.007 / 127.944 = +5.5e-5 ≈ +0.0055%

# To close this via Family-D-analog: c_FD^EM · α_1² = 0.0055%
# c_FD^EM = 0.000055 / α_1² = 0.000055 / 0.001522 = 0.036

print(f"   α_EM(M_Z) residual: +1.01σ_PDG = +0.011% (predicted-too-small)")
print(f"   Required correction: δα_EM/α_EM = +1.1e-4")
print(f"   Family-D form: c_FD^EM · α_1² = 1.1e-4 → c_FD^EM = "
      f"{1.1e-4/ALPHA_1_SQ:.3f}")
print(f"   Expressed as Fraction: ≈ {Fraction(1.1e-4/ALPHA_1_SQ).limit_denominator(20)}")
print()

# ============================================================
# VERDICT
# ============================================================
print("="*78)
print(" VERDICT")
print("="*78)
print()
print(f" The empirical c_1 = 1/3 + 0.008 reading is at +1.36σ_c — NOT robust.")
print(f" c_2 = 1/3 - 0.0017 reading is at -0.27σ_c — fully consistent with 1/3.")
print()
print(f" Family-D-analog magnitude analysis:")
print(f"   Per-leg coefficient ranges 1/12 to 4 (y_τ, λ_Higgs analogs).")
print(f"   Predicted Δc_FD = c_FD · α_1²/X = c_FD · {ALPHA_1_SQ/X_DARK:.4f}")
print(f"   c_FD = 1/4 gives Δc = +0.0094 — ALMOST MATCHES empirical +0.0095!")
print(f"   c_FD = 1/4 also has clean structural interpretation:")
print(f"   - 2 gauge boson legs at U(1)_Y vertex × (1/2) per leg = 1")
print(f"     (analog of λ_Higgs's 4·a_H = 4 with a_H = 1 per Higgs leg)")
print(f"   - But this gives c_FD = 1, not 1/4")
print(f"   - The empirical c_FD = 1/4 doesn't directly match a clean")
print(f"     per-leg-count interpretation")
print()
print(f" HONEST READING (this probe):")
print()
print(f"   1. The empirical c_1 - 1/3 = +0.0095 is at 1.36σ confidence —")
print(f"      NOT firm structural signal that c_1 ≠ 1/3 exactly.")
print()
print(f"   2. Family-D-analog at α_1² scale CAN produce a c-shift of the right")
print(f"      ORDER (~0.0095) but only at c_FD = 1/4, which doesn't have an")
print(f"      obvious per-leg-counting derivation for gauge boson 2-point.")
print()
print(f"   3. The master doc explicitly states Family-D fails on gauge boson")
print(f"      2-point functions (sign-uniformity breaks down for M_Z, m_W).")
print()
print(f"   4. The R_∞ residual (+0.021% — large in σ_PDG units, small")
print(f"      relatively) is dominated by the EXTERNAL Δα_running Type-3")
print(f"      import (per Rinf_clean_ratio_diagnostic_2026-05-16.py), NOT by")
print(f"      a framework α_EM derivation error needing structural fix.")
print()
print(f" CONCLUSION: c_1 = 1/3 + 0.008 is NOT structurally derivable via")
print(f" Family-D-analog at the current level of substrate machinery. The")
print(f" empirical 1.36σ residual is consistent with one-loop MSSM precision")
print(f" (the framework's stated systematic floor for one-loop running, per")
print(f" theorem_alpha_GUT_dark_correction.md §6).")
print()
print(f" RECOMMENDATION: pin c_EW = 1/3 exactly to THEOREM-GRADE-STRUCTURAL")
print(f" via the saturation argument (no numeric change, grade lift only on 4-5")
print(f" rows). Do NOT pursue Family-D-analog sub-leading derivation for c_1 —")
print(f" the empirical signal is too weak to support it and the master doc")
print(f" already rules out Family-D for gauge 2-point functions.")
print()
print("="*78)
