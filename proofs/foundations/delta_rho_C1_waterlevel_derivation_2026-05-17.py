#!/usr/bin/env python3
"""
delta_rho_C1_waterlevel_derivation_2026-05-17.py

DERIVE the C1 spectral-mode water level for the δρ W/h_P channel from the
framework's OWN derived MDL apparatus — and apply it. NO fitting; N and Δφ
are taken from the channel's structural definition, the threshold from an
existing theorem, the verdict is whatever falls out.

DERIVED THRESHOLD (predictions/uniform_Q_density_derivation.md, Theorem A;
Rissanen 1978 two-part MDL + Pinsker/χ², Cover-Thomas Lemma 17.3.2 —
THEOREM-GRADE, not introduced here):

  A Q-space Fourier mode of amplitude ε on angular support Δφ is RETAINED
  (above the MDL waterline) iff   ε²·Δφ  >  log(N)/N .
  Otherwise the density is uniform there (mode → 0).

  ⇒ This is a *binary* model-selection threshold (include the parameter or
    not — Rissanen/BIC), NOT a reverse-water-filling soft (|M|−θ)₊
    attenuation. The soft form sketched in the prior probe
    (delta_rho_subleading_Mn_waterfilling_2026-05-17.py) was MY sketch,
    not the framework's derived rule. This probe uses the derived rule.

δρ-CHANNEL STRUCTURAL SCALE N (NOT fitted — derived from the channel def):
  δρ is read off the srs primitive-cell NON-BACKTRACKING (Hashimoto)
  spectrum. The cell's spectral sample size is its NB dimension
  2|E| = 12 — the SAME canonical constant as c_S = 1/(2|E|) = 1/12,
  Route F-2 dim(B|cell)=12, etc. So N_cell = 2|E| = 12.
  Pre-declared comparison scales (also structural, not fitted):
   - N=6  (Inv #3 measured M_n at K_GRID=6 — the measurement's own N)
   - N→∞  (cosmological observer, N_hub~1e60 — the m_ν / cosmology read)

PRE-DECLARED VERDICT LOGIC:
  • If at the canonical N_cell the derived threshold ZEROES M₂ → the
    framework's current M_n=0 is CORRECT for δρ → +4.58% is a genuine
    residual, NOT a waterfilling-misapplication artifact → hypothesis
    REFUTED for δρ; correct the prior probe's 'directionally confirmed'.
  • Binary ⇒ the only achievable δρ are M₂=0 (current, +4.58%) or M₂=full
    (−9.54% overshoot). The closing M₂_eff≈−0.0876 is NOT reachable at
    ANY N (binary admits no partial). Robust.
  • Hypothesis SURVIVES only if a structurally-DERIVED (not fitted) N puts
    M₂ exactly at retention AND that branch closes δρ — pre-declared.
"""
import math

# ---- measured mode + δρ context -------------------------------------------
M2_meas   = -0.27           # Inv #3 measured Fourier mode (q_space_extended K=6)
M2_pow    = M2_meas**2      # = 0.0729  (ε² in the threshold)
DR_obs    = 0.0104286
F_lead    = math.sqrt(5)/4
ALPHA1    = (2/3)**8
dr_lead   = 0.5*F_lead*ALPHA1                       # +1.0906%, +4.58%
# δρ with M₂ fully retained (binary 'retain' branch) — from prior probe:
dr_full   = +0.0094337                              # −9.54% (overshoot)

def rel(x): return (x/DR_obs - 1.0)*100.0

print(f"measured M₂ = {M2_meas}  (ε² = |M₂|² = {M2_pow:.5f})")
print(f"δρ leading (M₂=0)        = {dr_lead*100:+.5f}%  ({rel(dr_lead):+.2f}% vs obs)")
print(f"δρ with M₂ fully retained= {dr_full*100:+.5f}%  ({rel(dr_full):+.2f}% vs obs)")
print(f"(binary threshold ⇒ ONLY these two are reachable; the closing "
      f"M₂_eff≈−0.0876 is NOT a binary outcome at any N)\n")

# ---- derived threshold at the structural N choices ------------------------
def threshold(N, base):
    """log(N)/N in the stated log base (nats=BIC/Rissanen; bits=DL_struct)."""
    lg = math.log(N) if base == 'nats' else math.log2(N)
    return lg / N

print("DERIVED retention test:  RETAIN M₂ iff  |M₂|²·Δφ  >  log(N)/N\n")
print(f"{'N (scale)':<34}{'log(N)/N nats':>14}{'|M₂|²·Δφ (Δφ=π/6)':>20}"
      f"{'|M₂|²·Δφ (Δφ=1)':>18}  verdict")
for label, N in [("N_cell = 2|E| = 12  (canonical)", 12),
                 ("N = 6   (Inv #3 K_GRID)",          6),
                 ("N = 4   (N_atoms)",                4),
                 ("N → 1e60 (cosmological/N_hub)",    1e60)]:
    thr = threshold(N, 'nats')
    lhs_a = M2_pow * (math.pi/6)        # Δφ = 2π/(2|E|) = π/6
    lhs_b = M2_pow * 1.0               # Δφ = O(1) = 1 (theorem's stated form)
    retain = lhs_b > thr               # use the more generous Δφ=1
    print(f"{label:<34}{thr:>14.6f}{lhs_a:>20.5f}{lhs_b:>18.5f}  "
          f"{'RETAIN→overshoot' if retain else 'ZERO→ +4.58% (current)'}")

# crossover N where M₂ would just be retained (Δφ=1): log(N)/N = |M₂|²
# (reported only to show it is NOT a framework constant — anti-numerology)
def lnN_over_N(N): return math.log(N)/N
# bisection for log(N)/N = 0.0729 on N>e
lo, hi = math.e+0.01, 1e6
for _ in range(200):
    mid = math.sqrt(lo*hi)
    (lo, hi) = (mid, hi) if lnN_over_N(mid) > M2_pow else (lo, mid)
N_cross = math.sqrt(lo*hi)
print(f"\ncrossover N (log N/N = |M₂|², Δφ=1): N ≈ {N_cross:.1f} "
      f"— NOT a framework structural constant (4,6,12,N_hub… none match); "
      f"choosing it would be fitting ⇒ refused.")

# ---- verdict --------------------------------------------------------------
print("\n" + "="*72)
print("VERDICT — derived, not fitted")
print("="*72)
print(f"""\
 At the CANONICAL structural scale N_cell = 2|E| = 12:
   log(12)/12 = {math.log(12)/12:.5f}   vs   |M₂|²·Δφ ≤ {M2_pow:.5f}
   ⇒ |M₂|²·Δφ  <<  log(N)/N   ⇒  M₂ is BELOW the derived waterline
   ⇒ M₂ is CORRECTLY ZEROED → δρ = +1.0906% (+4.58%).

 The framework's current 'M_n = 0 for n≥1' IS the correct C1 waterfilling
 for the δρ channel at its own structural scale. The +4.58% is therefore
 a GENUINE residual — NOT an MDL-waterfilling-misapplication artifact.

 ⇒ HYPOTHESIS ('mdl waterfilling misapplication to blame') is REFUTED
   for the δρ channel by the framework's OWN derived threshold.
   The prior probe's 'directionally confirmed' is CORRECTED: a closing
   M₂_eff exists numerically, but the derived threshold is BINARY and at
   the canonical N it zeroes M₂ outright (and at any larger N it would
   FULLY retain → −9.54% overshoot). No N yields the closing value;
   reaching it requires a non-framework soft attenuation OR a fitted N
   (N≈{N_cross:.0f}, not a structural constant) — refused as numerology.

 ⇒ The +4.58% reverts to the §2 object: the continuum/dispersive
   (Fano-type) self-energy on the McKay cut = O9-class deep layer, OR a
   Phase-3 empirical tether — but it is NOT repairable by 'correct the
   M_n waterfilling'. That route is now closed-NEGATIVE, derived.

 Honest: this is the 4th reframe-or-flip this session and it flips
 AGAINST both my earlier deferral AND the prior probe's tentative
 confirmation. The discipline (derive, don't fit) produced it.
""")
print("="*72)
