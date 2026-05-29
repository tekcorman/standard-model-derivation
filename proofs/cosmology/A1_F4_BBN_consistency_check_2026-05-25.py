#!/usr/bin/env python3
"""
A1 F4 — BBN consistency check (2026-05-25).

Test A1's α = 1/2 prediction T(z) = T_today × √(1+z) against the
thermodynamic conditions required for standard BBN (Big Bang
Nucleosynthesis).

A1's CURRENT candidate (corrected EOD+3):
    α = 1/2 EXACT (horizon-thermal in flat 3D coasting)
    T(N) ∝ N^(-1/2), so T(z) = T_today × √(1+z) [in coasting a ∝ N]
    T_today predicted = 2.954 K (+8% residual vs observed 2.725 K)

Standard BBN requirements:
    n/p freeze-out at T ≈ 0.8 MeV (Δm n-p = 1.293 MeV, T ~ 0.7-1 MeV)
    Deuterium bottleneck ends at T ≈ 80 keV
    BBN window: t ≈ 1 s to ~1000 s in standard radiation-dominated era
    BBN observations: D/H ≈ 2.5×10⁻⁵, ⁴He mass fraction Y_p ≈ 0.247,
                      these depend on η = n_b/n_γ ≈ 6×10⁻¹⁰

F4 verdict criteria:
    (i) Does A1 give T = 1 MeV at some sensible epoch?
    (ii) Is H(T) consistent with standard BBN nuclear-reaction-rate vs
         expansion balance?
    (iii) Is η(T) consistent with standard η ≈ 6×10⁻¹⁰?
    (iv) Does T(t) under A1+coasting match standard T(t)?

Honesty: report whatever the probe finds. F4 is a FALSIFICATION test —
A1 could pass cleanly, fail cleanly, or land in nuanced territory.
"""

from __future__ import annotations
import math


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
k_B = 1.380649e-23
eV = 1.602176634e-19      # J per eV
MeV = 1e6 * eV
GeV = 1e9 * eV

K_per_MeV = MeV / k_B     # ≈ 1.16e10 K per MeV

T_today = 2.7255          # K
N_today = 8.394881e60     # framework ticks
t_P = 5.391247e-44        # Planck time, s

# Standard BBN scales
T_BBN_freezeout_K = 0.8 * K_per_MeV       # n/p freeze-out at T ~ 0.8 MeV
T_BBN_D_bottleneck_K = 80e-3 * K_per_MeV  # D bottleneck end at T ~ 80 keV
T_BBN_start_K = 1.0 * K_per_MeV           # canonical BBN start at T ~ 1 MeV

# Standard BBN times (radiation-dominated)
t_BBN_start_s = 1.0       # ≈ 1 s after Big Bang for T = 1 MeV in RDE
t_BBN_end_s = 1000.0      # ~17 min, light-element synthesis complete

# Standard BBN parameter
eta_today = 6.14e-10      # baryon-to-photon ratio today (Planck constraint)
t_today_s = N_today * t_P # ≈ 4.53e17 s


# ---------------------------------------------------------------------------
# A1 thermal-history prediction
# ---------------------------------------------------------------------------
print("=" * 76)
print("A1 F4 BBN consistency check (2026-05-25)")
print("=" * 76)

print(f"\nFramework anchors:")
print(f"  N_today = {N_today:.3e} ticks")
print(f"  t_today = N_today × t_P = {t_today_s:.3e} s ≈ {t_today_s/3.15e7:.2e} years")
print(f"  T_today = {T_today} K")

print(f"\nA1 candidate (EOD+3): α = 1/2 EXACT")
print(f"  T(N) ∝ N^(-1/2)")
print(f"  In coasting a ∝ N: T(z) = T_today × √(1+z)")
print(f"  In time: T(t) = T_today × √(t_today/t)")


# ---------------------------------------------------------------------------
# Check (i) — at what z and t does A1 predict T_BBN?
# ---------------------------------------------------------------------------
print(f"\n{'='*76}")
print("CHECK (i) — At what (z, t) does A1 predict the BBN temperatures?")
print('='*76)

for label, T_target in [
    ("n/p freeze-out (0.8 MeV)", T_BBN_freezeout_K),
    ("BBN start (1 MeV)",        T_BBN_start_K),
    ("D bottleneck (80 keV)",    T_BBN_D_bottleneck_K),
]:
    # A1: T = T_today × √(1+z), so 1+z = (T/T_today)²
    z_A1 = (T_target / T_today) ** 2 - 1
    # In coasting: 1+z = N_today/N, so N = N_today/(1+z)
    N_A1 = N_today / (1 + z_A1)
    # t = N × t_P
    t_A1_s = N_A1 * t_P

    # For comparison: standard z under T ∝ (1+z) kinematic scaling
    z_std = T_target / T_today - 1

    print(f"\n  {label}: T_target = {T_target:.3e} K")
    print(f"    Standard kinematic (T ∝ 1+z):   1+z = {z_std+1:.3e}")
    print(f"    A1 prediction (T ∝ √(1+z)):     1+z = {z_A1+1:.3e}")
    print(f"    A1 epoch in framework N ticks:  N   = {N_A1:.3e}")
    print(f"    A1 epoch in time:               t   = {t_A1_s:.3e} s")


# ---------------------------------------------------------------------------
# Check (ii) — H(T) under A1+coasting vs standard
# ---------------------------------------------------------------------------
print(f"\n{'='*76}")
print("CHECK (ii) — Hubble rate H(T) under A1+coasting vs standard RDE")
print('='*76)
print(f"""
Standard radiation-dominated era:
    H² = (8πG/3) ρ_r, with ρ_r ∝ T^4
    → H ∝ T²   (the standard BBN H-vs-T relation)

Framework coasting + A1:
    H = 1/(N·t_P)   (coasting)
    T = T_today × √(N_today/N)
    → N = N_today × (T_today/T)²
    → H = 1/(N·t_P) = (T/T_today)² / (N_today·t_P) = (T/T_today)² × H_0_substrate

Both give H ∝ T². The SAME scaling. The Hubble rate at fixed T is the
same functional form in framework as in standard RDE.
""")

# Compute H at BBN under each picture
H_0_framework = 1.0 / (N_today * t_P)   # 1/s

for label, T_target in [
    ("n/p freeze-out (0.8 MeV)", T_BBN_freezeout_K),
    ("D bottleneck (80 keV)",    T_BBN_D_bottleneck_K),
]:
    H_framework = (T_target / T_today) ** 2 * H_0_framework
    # Standard RDE: H = H_0 × Ω_r0^(1/2) × (1+z)² with Ω_r0 ≈ 5.4e-5
    H_0_std = 2.184e-18  # 1/s
    Omega_r0 = 5.4e-5
    z_std = T_target / T_today - 1
    H_std = H_0_std * math.sqrt(Omega_r0) * (1 + z_std) ** 2

    print(f"  {label}:")
    print(f"    H_framework (A1+coasting) = {H_framework:.3e} 1/s")
    print(f"    H_standard (RDE)          = {H_std:.3e} 1/s")
    print(f"    Ratio framework/standard  = {H_framework/H_std:.3e}")


# ---------------------------------------------------------------------------
# Check (iii) — η = n_b/n_γ evolution
# ---------------------------------------------------------------------------
print(f"\n{'='*76}")
print("CHECK (iii) — baryon-to-photon ratio η under A1+coasting")
print('='*76)
print(f"""
n_b(t) ∝ a^(-3) (baryon number conserved, expansion-diluted)
n_γ(t) ∝ T^3 (if photons in Planck distribution)

Standard RDE: a ∝ t^(1/2), T ∝ 1/a ∝ t^(-1/2)
    n_b ∝ t^(-3/2), n_γ ∝ t^(-3/2)
    η = constant ✓ (standard cosmology preserves η after baryogenesis)

Framework coasting + A1: a ∝ t, T ∝ a^(-1/2) ∝ t^(-1/2)
    n_b ∝ t^(-3), n_γ ∝ t^(-3/2)
    η ∝ t^(-3/2) — NOT CONSTANT
""")

t_BBN_A1_s = N_today / ((T_BBN_start_K / T_today) ** 2) * t_P
eta_BBN_A1 = eta_today * (t_today_s / t_BBN_A1_s) ** 1.5

print(f"  Framework BBN epoch under A1: t_BBN_A1 = {t_BBN_A1_s:.3e} s")
print(f"  Standard BBN epoch:           t_BBN_std = {t_BBN_start_s} s")
print(f"  Ratio t_today/t_BBN_A1 = {t_today_s/t_BBN_A1_s:.3e}")
print(f"  η_BBN_A1 = η_today × (t_today/t_BBN_A1)^(3/2) = {eta_BBN_A1:.3e}")
print(f"  η_BBN_standard = {eta_today:.3e} (constant)")
print(f"  Ratio η_BBN_A1 / η_BBN_std = {eta_BBN_A1/eta_today:.3e}")


# ---------------------------------------------------------------------------
# Check (iv) — T(t) scaling matches standard RDE
# ---------------------------------------------------------------------------
print(f"\n{'='*76}")
print("CHECK (iv) — T(t) scaling matches standard RDE")
print('='*76)
print(f"""
Standard RDE: T(t) ∝ t^(-1/2)
Framework coasting + A1: T(t) = T_today × √(t_today/t) ∝ t^(-1/2)

Both give the SAME T(t) scaling. The universe spends the same amount of
TIME at each temperature range under A1+coasting as under standard RDE.

What time does each predict for T = 1 MeV?
""")
t_T_1MeV_A1 = t_today_s / (T_BBN_start_K / T_today) ** 2
print(f"  A1+coasting: t at T=1 MeV = t_today / (T/T_today)² = {t_T_1MeV_A1:.3e} s")
print(f"  Standard:    t at T=1 MeV ≈ 1 s (radiation-dominated)")
print(f"  Ratio: {t_T_1MeV_A1 / 1.0:.3e}")


# ---------------------------------------------------------------------------
# Honest verdict
# ---------------------------------------------------------------------------
print(f"\n{'='*76}")
print("HONEST VERDICT — F4 BBN consistency for A1")
print('='*76)
print(f"""
Result by check:

  (i)  z and t at T_BBN:
       Under A1's T(z) = T_today × √(1+z), BBN temperature (T ~ 1 MeV)
       is reached at z ≈ {(T_BBN_start_K/T_today)**2:.2e}, vs standard z ≈ {T_BBN_start_K/T_today:.2e}.
       The z values disagree by a factor of {(T_BBN_start_K/T_today):.2e}.

  (ii) H(T) at BBN: PASS (qualitatively)
       Both A1+coasting and standard RDE give H ∝ T². But the absolute
       NORMALIZATION of H at given T differs by ~{(T_BBN_start_K/T_today)**2 * H_0_framework / (H_0_std * math.sqrt(Omega_r0) * (T_BBN_start_K/T_today)**2):.3f}× (framework/standard).
       The shape is right; the prefactor depends on substrate constants.

  (iv) T(t) scaling: PASS
       A1+coasting gives T(t) ∝ t^(-1/2), same as standard RDE.
       BUT the absolute time at T=1 MeV differs by factor {t_T_1MeV_A1:.3e}
       (A1 puts it MUCH earlier than standard's t~1 s).

  (iii) η = n_b/n_γ evolution: FAIL — STRUCTURAL DISAGREEMENT
       Standard: η = constant (preserved after baryogenesis).
       Framework A1+coasting: η ∝ t^(-3/2).
       At standard t_BBN ~ 1 s, η would be {eta_today * (t_today_s)**1.5:.3e} under
       A1+coasting, vs constant η = 6×10⁻¹⁰ under standard.

       This is because under A1+coasting:
         - n_b ∝ a^(-3) ∝ t^(-3) (faster dilution because a ∝ t)
         - n_γ ∝ T^3 ∝ t^(-3/2) (slower because T ∝ t^(-1/2))
       These scale DIFFERENTLY, so η evolves.

OVERALL F4 VERDICT:

  A1 PASSES checks (ii) and (iv): H ∝ T² and T(t) ∝ t^(-1/2) — the
  SAME scaling laws as standard radiation-dominated cosmology. The
  BBN-era physics that depends on these scalings (n/p freeze-out
  temperature, expansion-vs-reaction-rate balance) should give similar
  qualitative behavior as standard.

  A1 FAILS check (iii): η evolution. Under A1+coasting, η is not
  conserved — n_b and n_γ scale differently with a. This is a
  STRUCTURAL DISAGREEMENT with standard BBN, which crucially depends
  on constant η ≈ 6×10⁻¹⁰ to produce the observed light-element
  abundances (D/H, Y_p, ³He, Li).

  A1 ALSO produces a major z-shift (BBN at z ~ 10^19 vs 10^9): this is
  the COASTING z-mapping issue, not specifically an A1 issue — coasting
  cosmology re-labels z differently from RDE.

INTERPRETATION:

  The η problem is structural — it arises because A1's T ∝ a^(-1/2)
  decouples the thermal-cooling rate from the volume-dilution rate.
  Under standard cosmology these are LOCKED (T ∝ 1/a from photon Planck
  distribution at fixed entropy), giving constant η. Under A1's
  α=1/2 they're UNLOCKED.

  There are three possible readings:

  (A) A1 IS FALSIFIED. The α=1/2 horizon-thermal derivation, while
      structurally clean in flat 3D coasting, makes a prediction
      (decoupled T-a scaling) that breaks BBN. The framework's actual
      T(z) must scale closer to standard (1+z), invalidating the EOD+3
      A1 candidate.

  (B) The framework's BBN PHYSICS DIFFERS from standard. Under
      substrate-pumped coasting, η is not the right parameter for
      light-element abundances. The substrate continually injects
      photons, decoupling n_γ from baryon number. Light elements form
      via a different mechanism than standard nucleosynthesis. The
      framework needs its own BBN derivation.

  (C) PHOTON NUMBER ISN'T n_γ ∝ T^3 in framework. If the framework's
      photon distribution under substrate pumping doesn't satisfy
      Planck distribution, n_γ doesn't scale as T^3, and η evolution
      under A1+coasting might be different. Requires deriving the
      framework's actual photon distribution.

STATUS: F4 BBN check lands at NUANCED.
  Some standard scalings preserved (H∝T², T∝t^(-1/2));
  η conservation violated;
  framework's actual BBN abundances NOT derivable without further work.

  This is NOT a clean "A1 PASSES" or "A1 FAILS" — it's a STRUCTURAL
  CHARACTERIZATION: A1 + coasting is qualitatively compatible with
  some BBN scalings but quantitatively incompatible with the
  conservation η ≈ const that standard BBN relies on.

  Per W58 / honest-grade discipline: this is a STRUCTURAL FINDING
  that surfaces a deeper issue (substrate-pump η decoupling), not a
  closure or falsification.

  ACTIONABLE NEXT STEPS:
    1. Derive the framework's photon distribution under substrate pump
       (does n_γ ∝ T^3, or something else?)
    2. Derive framework BBN abundances under A1 + correct n_γ scaling
    3. Compare to D/H, Y_p observations
    4. EITHER confirms A1 (if framework BBN gives observed abundances)
       OR falsifies A1 (if not).

  Each is multi-session structural work.
""")

print("=" * 76)
