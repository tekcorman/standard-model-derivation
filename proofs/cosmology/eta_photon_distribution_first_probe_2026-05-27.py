#!/usr/bin/env python3
"""
η / photon-distribution under coasting — first probe (2026-05-27).

Scoping doc: an internal working note

PURPOSE
-------
Before the BBN reaction-network harness (proofs/cosmology/lib/bbn_network.py)
can be trusted, the framework must say what η = n_b/n_γ is *through the BBN
window*. There is an internal tension:

  (1) predictions/eta_B.py asserts η_B = ε_CP·Re(h)·α₁^M = 6.11e-10, built
      from dimensionless structural constants with NO N → η is N-INVARIANT.

  (2) The F4 kinematic argument (A1_F4_BBN_consistency_check_2026-05-25.py):
      n_b ∝ a⁻³ (baryon conservation) and n_γ ∝ T³ (Planck) with T ∝ a⁻¹ᐟ²
      (α=1/2 thermal law) → η ∝ a⁻³ᐟ², NOT constant.

Exactly one input to (2) is wrong inside the framework. This probe:

  P1 — quantifies the comoving-entropy growth (coasting is non-adiabatic).
  P2 — makes the kinematic η drift across the BBN window explicit, against
       the LIVE eta_B.py value.
  P3 — tests reading (B): if baryon number is CO-PUMPED with entropy,
       dn_b/dt = η_B·dn_γ/dt, is η pinned to η_B identically?
  P4 — states the A/B/C discriminator.

This is a SCOPING probe — it sharpens the question and tests reading B's
bookkeeping. It does NOT close the question (that is sessions 2-5).

Run:
    python3 proofs/cosmology/eta_photon_distribution_first_probe_2026-05-27.py
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import sys

# --- import the LIVE framework η_B (no hardcoded value) ---------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PRED_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "predictions"))
sys.path.insert(0, _PRED_DIR)

_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    # eta_B.py exposes the closed substrate-Sakharov prediction.
    from eta_B import eta_B_pred  # dimensionless, N-invariant by construction

ETA_B = float(eta_B_pred)

# --- BBN window (temperatures in MeV; standard landmarks) -------------------
# These are the *thermal* endpoints of nucleosynthesis, used only to size the
# scale-factor span Δln a over which η would drift. Not framework-derived.
T_START_MeV = 10.0     # network start, weak interactions in equilibrium
T_NPFREEZE_MeV = 0.8   # n/p weak freeze-out
T_BOTTLENECK_MeV = 0.07  # deuterium bottleneck (4He assembly)

ALPHA_THERMAL = 0.5    # framework instantaneous T-N exponent (T ∝ a^(-α))

ZETA3 = 1.2020569


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def a_ratio_from_T(T_hi_MeV: float, T_lo_MeV: float, alpha: float) -> float:
    """Scale-factor ratio a(T_lo)/a(T_hi) for T ∝ a^(-alpha): a ∝ T^(-1/alpha)."""
    return (T_hi_MeV / T_lo_MeV) ** (1.0 / alpha)


# ============================================================================
banner("η / PHOTON-DISTRIBUTION FIRST PROBE (2026-05-27)")
print(f"""
  Live inputs:
    η_B (predictions/eta_B.py) = {ETA_B:.4e}   [N-INVARIANT by construction]
    α_thermal (T ∝ a^-α)       = {ALPHA_THERMAL}
  BBN window (thermal landmarks, MeV):
    start = {T_START_MeV}, n/p freeze-out = {T_NPFREEZE_MeV}, bottleneck = {T_BOTTLENECK_MeV}
""")

# ----------------------------------------------------------------------------
# P1 — comoving entropy growth: s·a³ ∝ T³a³ ∝ a^(3(1-α)) = a^{3/2} for α=1/2
# ----------------------------------------------------------------------------
banner("P1 — Comoving photon entropy is NOT conserved under α=1/2")

entropy_exponent = 3.0 * (1.0 - ALPHA_THERMAL)  # power of a in s·a³ ∝ a^{...}
a_span_window = a_ratio_from_T(T_START_MeV, T_BOTTLENECK_MeV, ALPHA_THERMAL)
entropy_growth = a_span_window ** entropy_exponent

print(f"""
  Standard (adiabatic): s·a³ = const  ⟺  T ∝ 1/a  (α=1). Entropy conserved.

  Framework α={ALPHA_THERMAL}:  s ∝ g_*T³,  T ∝ a^(-{ALPHA_THERMAL})
     s·a³ ∝ T³·a³ ∝ a^(3·(1-α)) = a^{entropy_exponent:.2f}

  Across the BBN window:
     a(bottleneck)/a(start) = (T_start/T_bot)^(1/α) = ({T_START_MeV}/{T_BOTTLENECK_MeV})^{1/ALPHA_THERMAL:.0f}
                            = {a_span_window:.3e}
     comoving entropy grows by  a^{entropy_exponent:.2f} = {entropy_growth:.3e}×

  ⇒ Under a LITERAL α=1/2 bath-cooling law the substrate pumps ~10^{math.log10(entropy_growth):.0f}×
    the comoving entropy across BBN. Coasting is strongly NON-ADIABATIC.
    This is the SAME missing physics as the √g_* leading factor in H
    (substrate↔bath coupling); see scoping §2/§5.
""")

# ----------------------------------------------------------------------------
# P2 — kinematic η drift across the window, vs the live constant η_B
# ----------------------------------------------------------------------------
banner("P2 — Kinematic η ∝ a^(-3/2) drifts by orders of magnitude")

# If we PIN η = η_B at the bottleneck (end of the window) and run the kinematic
# law backward, η at the start would have been η_B · (a_start/a_bot)^(-3/2)...
# i.e. η_kin(a) = η_ref · (a/a_ref)^(-3/2). Show the start/bottleneck contrast.
eta_drift_exponent = -3.0 * (1.0 - ALPHA_THERMAL)  # = -3/2 for α=1/2 (η ∝ a^{-3(1-α)})
# η_start / η_bottleneck under the kinematic law:
eta_start_over_bottleneck = a_span_window ** eta_drift_exponent

print(f"""
  Kinematic law (F4): η ∝ a^(-3(1-α)) = a^{eta_drift_exponent:.2f}  (= a^-3/2 at α=1/2)
  [n_b ∝ a⁻³ conserved; n_γ ∝ T³ ∝ a^(-3α) Planck]

  Pin η(bottleneck) = η_B = {ETA_B:.4e}.  Then:
     η(start)/η(bottleneck) = a_span^{eta_drift_exponent:.2f} = {eta_start_over_bottleneck:.3e}
     η(start, kinematic)    = {ETA_B * eta_start_over_bottleneck:.4e}

  The kinematic law makes η vary by {1.0/eta_start_over_bottleneck:.2e}× across the
  window. eta_B.py says η is CONSTANT. These are flatly incompatible:
  a network run with a drifting η of this magnitude is meaningless.
""")

# ----------------------------------------------------------------------------
# P3 — reading (B): co-pumping pins η to η_B identically
# ----------------------------------------------------------------------------
banner("P3 — Reading B (co-pumping): does dn_b/dt = η_B·dn_γ/dt pin η?")

# Bookkeeping test. Discretise a few steps of expansion. Under co-pumping the
# substrate creates baryons at η_B times the photon-creation rate, so at every
# step n_b = η_B · n_γ exactly, regardless of how n_γ scales. We DEMONSTRATE
# the bookkeeping is self-consistent (not that the rate is derived — that is
# session 3). Compare against the "conservation" bookkeeping that drifts.

# Arbitrary positive photon-number track n_γ(a) ∝ a^(-3α) (Planck) over steps.
a_grid = [1.0, 2.0, 4.0, 8.0, 16.0]
n_gamma = [a ** (-3.0 * ALPHA_THERMAL) for a in a_grid]   # Planck n_γ ∝ a^{-3α}

# (i) conservation bookkeeping: n_b ∝ a^-3 (independent of n_γ)
n_b_conserved = [a ** (-3.0) for a in a_grid]
# normalise so η(a=1) = η_B in both schemes
norm_cons = ETA_B * n_gamma[0] / n_b_conserved[0]
n_b_conserved = [norm_cons * x for x in n_b_conserved]

# (ii) co-pumping bookkeeping: n_b(a) = η_B · n_γ(a) by construction
n_b_copump = [ETA_B * ng for ng in n_gamma]

print(f"\n  {'a/a0':>6} {'n_γ(arb)':>12} {'η (conserve)':>14} {'η (co-pump)':>14}")
print("  " + "-" * 50)
for a, ng, nbc, nbp in zip(a_grid, n_gamma, n_b_conserved, n_b_copump):
    eta_cons = nbc / ng
    eta_cop = nbp / ng
    print(f"  {a:>6.1f} {ng:>12.4e} {eta_cons:>14.4e} {eta_cop:>14.4e}")

print(f"""
  Conservation bookkeeping: η drifts (∝ a^-3/2) — the tension.
  Co-pumping bookkeeping:   η ≡ η_B = {ETA_B:.4e} at every step, identically.

  ⇒ Reading B is INTERNALLY CONSISTENT: if the Sakharov chain runs as an
    ONGOING co-production locked at η_B, the tension vanishes by construction.
    The open work (session 3) is to DERIVE the baryon-creation rate as
    η_B × (entropy-creation rate) from the NB-walker Sakharov mechanism —
    NOT to fix the bookkeeping, which is already self-consistent.
""")

# ----------------------------------------------------------------------------
# P4 — the A/B/C discriminator
# ----------------------------------------------------------------------------
banner("P4 — Discriminator for readings A / B / C")
print(f"""
  η_BBN value:
    (A) adiabatic α=1 bath:  η_BBN = η_B = {ETA_B:.4e}  (const; standard BBN)
    (B) co-pumping:          η_BBN = η_B = {ETA_B:.4e}  (const; +ongoing baryogenesis)
    (C) non-Planck n_γ:      η_BBN ≠ η_B  (must be re-derived from n_γ(T))

  ⇒ At BBN, A and B are IDENTICAL (both feed η_B to the harness). They differ
    only at late times (B predicts continuous baryon creation — a distinct,
    potentially falsifiable signature). C is already distinguished AT BBN.

  HARNESS IMPLICATION: default the framework run to η = η_B (readings A/B).
  Only build photon statistical mechanics (reading C) if A and B both fail
  in sessions 2-3.
""")

banner("VERDICT — first probe")
print(f"""
  • Tension confirmed quantitatively: comoving entropy grows ~{entropy_growth:.0e}× and
    kinematic η would drift ~{1.0/eta_start_over_bottleneck:.0e}× across the BBN window —
    incompatible with the live N-invariant η_B = {ETA_B:.4e}.
  • Reading B (ongoing co-pumping) removes the tension by construction; its
    open piece is a RATE derivation, not a bookkeeping fix.
  • Readings A and B both deliver η_BBN = η_B → the harness may use the live
    η_B as a STATED ASSUMPTION (not yet a closure).
  • Next: session 2 (reading A adiabaticity) and session 3 (reading B rate);
    deeper target is the unified substrate-thermal-coupling mechanism (scoping §5)
    that would close this AND the √g_* leading factor together.

  Per W58: this is a SCOPING result. No closure claimed; no free parameter
  introduced. η_B is consumed LIVE from predictions/eta_B.py.
""")
