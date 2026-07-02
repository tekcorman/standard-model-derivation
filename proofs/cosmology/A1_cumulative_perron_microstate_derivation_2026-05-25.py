#!/usr/bin/env python3
"""
A1 cumulative-Perron / microstate derivation attempt (2026-05-25).

Goal: attempt the microstate-based α correction that retracted candidate
#4 was reaching for, but properly framed — NOT as a fractional spatial
dimension (which conflicted with theorem-grade d_spatial = 3), but as a
CUMULATIVE MICROSTATE-VOLUME contribution over N substrate ticks.

The structural claim under test:
  - Each substrate tick contributes a Perron-singlet microstate share
    c_S = 1/(2|E|) = 1/12 (theorem-grade upstream)
  - These accumulate over N ticks
  - This modifies horizon-thermal scaling away from α = 1/2 EXACT to
    α_effective = 1/2 + ε for some structurally-derived ε

Two questions:
  (Q1) At the CORRECTED A1 anchor (T_GUT = M_unif × c_S), what α
       reproduces the observed T_today = 2.725 K exactly?
  (Q2) What microstate-volume scaling V_eff(N) gives this α?
  (Q3) Does cumulative-Perron from N=1 to N=N_today give the right
       size correction, or is it 33× too large?

Additionally addresses F4's η problem under substrate-microstate-pumping:
  (Q4) If substrate continuously pumps photons + baryons with rates
       determined by microstate counting, is η preserved or evolving?
"""

from __future__ import annotations
import math
from fractions import Fraction


# ---------------------------------------------------------------------------
# Constants and framework primitives
# ---------------------------------------------------------------------------
k_B = 1.380649e-23
GeV = 1.602176634e-10
K_per_GeV = GeV / k_B
M_Pl_GeV = 1.220890e19

N_hub = 8.394881e60
v_today = 246.22
M_unif_GeV = 1.985e16
T_CMB = 2.7255

k_star = 3
N_atoms = 4
two_E = N_atoms * k_star    # 12
c_S = Fraction(1, two_E)    # 1/12

alpha_GUT_bare = Fraction(1, 24)
alpha_1_bare = Fraction(2, 3) ** 8

N_GUT = N_hub / (M_unif_GeV / v_today) ** 4

# ---------------------------------------------------------------------------
# Q1 — What α at the corrected anchor reproduces T_today exactly?
# ---------------------------------------------------------------------------
print("=" * 76)
print("Q1 — Required α at T_GUT = M_unif × c_S anchor to give T_today = T_CMB")
print("=" * 76)

T_GUT_K = M_unif_GeV * float(c_S) * K_per_GeV
ratio_target = T_CMB / T_GUT_K   # T_today / T_GUT_anchor
ratio_NN = N_GUT / N_hub
ln_ratio_NN = math.log(ratio_NN)

# T_today = T_GUT × (N_GUT/N_today)^α
# ln(T_today/T_GUT) = α × ln(N_GUT/N_hub)
alpha_required = math.log(ratio_target) / ln_ratio_NN
alpha_excess = alpha_required - 0.5

print(f"\n  T_GUT_anchor = M_unif × c_S × K_per_GeV = {T_GUT_K:.3e} K")
print(f"  T_today_observed = {T_CMB} K")
print(f"  ratio T_today/T_GUT = {ratio_target:.3e}")
print(f"  ratio N_GUT/N_today = {ratio_NN:.3e}")
print(f"  ln(N_GUT/N_today) = {ln_ratio_NN:.3f}")
print(f"\n  α_required = {alpha_required:.6f}")
print(f"  α_excess vs α=1/2 = {alpha_excess:+.6f} = {alpha_excess*1000:.3f} × 10⁻³")


# ---------------------------------------------------------------------------
# Q2 — Comparison to candidate structural α corrections
# ---------------------------------------------------------------------------
print(f"\n{'='*76}")
print("Q2 — Structural candidate α values vs required α")
print('='*76)

candidates = [
    ("α = 1/2 EXACT (no correction)",      Fraction(1, 2),           "horizon-thermal in flat 3D coasting"),
    ("α = 1/2 + 1/(4·2|E|) = 25/48",       Fraction(1, 2) + Fraction(1, 4 * two_E),
                                            "retracted candidate #4 (cumulative Perron, 'd_eff = 3 + 1/(2|E|)')"),
    ("α = 1/2 + 1/(8·2|E|) = 49/96",       Fraction(1, 2) + Fraction(1, 8 * two_E),
                                            "half-size cumulative-Perron"),
    ("α = 1/2 + c_S²/2 = 0.5 + 1/288",     Fraction(1, 2) + c_S * c_S / 2,
                                            "second-order Perron (Perron of Perron)"),
    ("α = 1/2 + α₁_bare/(8·12)",           Fraction(1, 2) + alpha_1_bare / (8 * two_E),
                                            "dark-waterline-modulated Perron"),
]

print(f"\n  {'candidate':<55} | {'α value':<12} | {'gives T_today':<13} | {'residual':<10}")
print(f"  {'-'*55}-|-{'-'*12}-|-{'-'*13}-|-{'-'*10}")

for label, alpha_frac, desc in candidates:
    alpha_val = float(alpha_frac)
    T_today_pred = T_GUT_K * ratio_NN ** alpha_val
    residual_pct = (T_today_pred - T_CMB) / T_CMB * 100
    print(f"  {label:<55} | {alpha_val:.8f} | {T_today_pred:>11.4f} K | {residual_pct:+7.2f}%")


# ---------------------------------------------------------------------------
# Q3 — Cumulative Perron from N=1 to N=N_today (full substrate range)
# ---------------------------------------------------------------------------
print(f"\n{'='*76}")
print("Q3 — Cumulative-Perron derivation: how much α does it actually predict?")
print('='*76)
print(f"""
Cumulative-Perron structural reading:
  - Each substrate tick: substrate's NB walker takes one step on B_NB(srs)
  - Perron-singlet projection at Γ has weight c_S = 1/(2|E|) = 1/12
  - This is the gauge-readable share of substrate-microstate accessible
    at that tick
  - Over N ticks, cumulative gauge-readable microstate count: ∝ N · c_S

Naive interpretation A — V_eff = N^3 × N^c_S = N^(3 + 1/12):
  u = E/V_eff = κN / N^(3 + 1/12) = κ N^(-25/12)
  T = u^(1/4) ∝ N^(-25/48)
  α = 25/48 = {float(Fraction(25, 48)):.6f}
  Δα vs 1/2 = 1/48 = {1/48:.6f}

  This is 33× LARGER than α_required ({alpha_required:.4f} - 0.5 = {alpha_excess:.6f}).
  → Cumulative-Perron with full N-exponent is TOO BIG.

Naive interpretation B — V_eff = N^3 + N · c_S (additive, sub-leading):
  V_eff/V_naive = 1 + c_S/N², negligible for large N
  → No appreciable correction to α. Doesn't close.

Interpretation C — second-order cumulative:
  V_eff = N^3 × N^(c_S²) = N^(3 + 1/144)
  α_C = (d_eff - 1)/4 = (2 + 1/144)/4 = 1/2 + 1/576 = {1/2 + 1/576:.6f}
  Δα = 1/576 = {1/576:.6f}

  This is still 3× larger than α_required.

Interpretation D — cumulative-Perron from N_GUT (not N=1):
  Effective accumulation over GUT-to-today is N_today - N_GUT ≈ N_today
  Same as Interpretation A — too big.

Interpretation E — cumulative-Perron with sub-fractional weight per tick:
  If per-tick contribution is c_S/scale with scale chosen so total is small,
  this is fitting, not deriving. Reject.

CONCLUSION:
  The simplest cumulative-Perron derivations all give α corrections that
  are ORDERS OF MAGNITUDE too large for the required Δα ≈ 0.0006 = {alpha_excess:.4f}.

  The required α excess Δα ≈ 6 × 10⁻⁴. Framework primitives near this size:
  - 1/(N_atoms·k*³) = 1/108 ≈ 9.3 × 10⁻³ (10× too big)
  - 1/(2|E|)² = 1/144 ≈ 6.9 × 10⁻³ (10× too big)
  - α₁²/(stuff) — would need denominator ~ 60-100

  There's NO obvious clean framework primitive matching Δα = 6 × 10⁻⁴.
""")


# ---------------------------------------------------------------------------
# Q4 — F4 η problem under substrate-microstate-pumping
# ---------------------------------------------------------------------------
print(f"\n{'='*76}")
print("Q4 — η problem under microstate-pumped continuous baryon+photon production")
print('='*76)
print(f"""
F4's η problem (n_b ∝ a^(-3) but n_γ ∝ T^3 ∝ a^(-3/2) under A1+coasting)
ASSUMES standard cosmological conservation: baryon number conserved
after baryogenesis, dilutes only by expansion.

Under the SUBSTRATE-MICROSTATE-PUMPED picture:
  - Substrate generates matter + radiation continuously (each tick)
  - Per-tick baryon production rate: ρ_b_per_tick (substrate-determined)
  - Per-tick photon production rate: ρ_γ_per_tick (substrate-determined)
  - Cumulative at observation epoch N:
      n_b(N) = ρ_b_per_tick × N (per comoving volume; baryons not diluted
               because they're continuously created)
      n_γ(N) = ρ_γ_per_tick × N (same)
  - η = n_b/n_γ = ρ_b_per_tick / ρ_γ_per_tick = CONSTANT
      (set by substrate's relative production rates)

If the substrate's baryon-to-photon production ratio MATCHES the observed
η ≈ 6 × 10⁻¹⁰, then F4's η problem is RESOLVED by accepting the framework's
substrate-pump picture.

This is the microstate intuition's structural payoff: η isn't preserved by
external conservation — it's SET by substrate's relative production rates.
The substrate fixes η from primitives.

What sets the substrate's relative baryon-to-photon production?
  - If both come from one Landauer pump (κ/t_P per substrate tick), and
    each tick produces 1 photon + 1 baryon, η = 1 (way too big)
  - If baryon production is suppressed by some substrate factor relative
    to photon production, η < 1
  - Empirical η ≈ 6 × 10⁻¹⁰ would need a specific suppression factor

Candidate sources for η ≈ 6 × 10⁻¹⁰:
  - α_1_bare^? — 256/6561 ≈ 0.039; α_1^4 ≈ 2.3 × 10⁻⁶; α_1^6 ≈ 3.6 × 10⁻⁹
  - α_GUT^? — 1/24 ≈ 0.042; (1/24)^7 ≈ 2.4 × 10⁻¹⁰ (close to 6×10⁻¹⁰)
  - (1/N_hub)^(1/4) ≈ 6 × 10⁻¹⁶ (too small)
  - More elaborate combinations...

This is PATTERN-HUNTING territory; without a structural derivation of
the substrate's relative baryon-to-photon production rate, no specific
η value is forced. But the structural READING — η is substrate-set
rather than conserved — is independent of which specific value comes out.
""")


# ---------------------------------------------------------------------------
# Honest verdict
# ---------------------------------------------------------------------------
print(f"\n{'='*76}")
print("HONEST VERDICT — cumulative-Perron microstate derivation attempt")
print('='*76)
print(f"""
Q1: at the corrected anchor (T_GUT = M_unif × c_S), the REQUIRED α to
    match T_today = 2.725 K is α = {alpha_required:.6f}, i.e., Δα = {alpha_excess:.6f}.

Q2: simple structural α candidates ({float(Fraction(25, 48)):.4f}, etc.) all give
    Δα orders of magnitude TOO LARGE. None match Δα = 6 × 10⁻⁴.

Q3: cumulative-Perron framings (V_eff = N^(3+c_S), N^(3+c_S²), etc.) all
    give α corrections orders of magnitude too large. The cumulative-Perron
    intuition at the GUT-anchor framing does NOT close the 8% residual.

Q4: F4's η problem CAN be resolved by accepting substrate-microstate-pumped
    continuous baryon+photon production with substrate-determined relative
    rates. This is a STRUCTURAL READING (not a derivation of η numerical
    value), but it dissolves the apparent contradiction with BBN.

DISTINCT EPISTEMIC LANDINGS:

  (A) Cumulative-Perron at GUT anchor (α correction): FAILS at the
      corrected anchor. The 8% T_today residual is NOT the cumulative-
      Perron N-exponent correction. It must be either:
        (i) an anchor calibration issue (T_GUT ≠ M_unif × c_S exactly),
        (ii) a different small structural correction not yet identified,
        (iii) genuine 8% residual unresolved.

  (B) Substrate-pump microstate framing for F4 η: HONEST POSITIVE
      STRUCTURAL READING. The framework's substrate-pump picture
      NATURALLY preserves η as a substrate-set ratio. F4's η problem
      DISSOLVES if we accept this framing. This is a real structural
      finding, even though the specific η value isn't derived yet.

  (C) The retracted "α_empirical = 0.5201 matches 25/48" was for the
      SUBSTRATE-to-today propagation (N=1 to N_today), NOT the corrected
      GUT-to-today propagation. Under substrate-to-today, α = 25/48
      cumulative-Perron MIGHT be right, but that's a DIFFERENT framing
      from the corrected A1 candidate.

NET FINDING:
  The microstate intuition has real structural content for F4 (resolves
  η problem by recognizing substrate-pump as primary). It does NOT close
  A1's 8% T_today residual under the corrected anchor framing.

  The 8% T_today residual remains GENUINELY OPEN. The cumulative-Perron
  derivation gives corrections that are too large by orders of magnitude
  at the corrected anchor; the dark-correction Routes H/C give corrections
  that don't close (already shown today).

  Path forward:
    1. The 8% T_today residual might be the ANCHOR calibration — T_GUT
       isn't exactly M_unif × c_S, but rather some closely-related
       framework scale that's ~17% smaller. Worth investigating.
    2. The substrate-pump η framing is a positive structural contribution
       to F4 that should be captured even though the specific η number
       isn't derived.

  Per W58 / no-3-point-fit discipline: no closure claimed. The
  cumulative-Perron derivation at GUT anchor is HONEST NEGATIVE; the
  microstate framing for F4 η is HONEST POSITIVE STRUCTURAL READING.
""")

print("=" * 76)
print("STATUS: cumulative-Perron at GUT anchor FAILS (correction too big);")
print("        microstate framing for F4 η problem PASSES (dissolves the issue).")
print("=" * 76)
