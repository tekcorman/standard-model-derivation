#!/usr/bin/env python3
"""
*** RETRACTED 2026-05-25 EOD+3 — WRONG PREMISE ***

This probe attempted to derive a mechanism for d_eff = 3 + 1/(2|E|).
The premise itself was wrong: d_spatial = 3 (theorem-grade via Cencov-
Fisher) rules out a fractional-dimension correction. The probe could
not find a mechanism because the mechanism doesn't exist — the d_eff
deviation it was trying to explain doesn't exist either.

The corrected framing (`A1_perron_anchor_at_GUT_2026-05-25.py`):
  d_eff = 3 exact, α = 1/2 exact, c_S enters at the thermal anchor
  T_GUT = M_unif × c_S, not in d_eff.

The text below is preserved for record but its premise is incorrect.
Don't waste sessions on the six candidates enumerated below — they
all fail because they're trying to derive a non-existent fractional
dimension.

---

A1 d_eff = 3 + 1/(2|E|) — mechanism derivation attempt (2026-05-25).

The previous probe (`A1_d_eff_derivation_attempt_2026-05-25.py`) found that
d_eff = 3 + 1/(2|E|) = 3.0833 matches empirical α (GUT anchor, no g* corr)
within 0.1%. The 1/(2|E|) factor is structurally meaningful: it's the
framework's c_S Perron-residue projection scale, theorem-grade upstream.

This probe attempts to DERIVE why d_eff gets this specific 1/(2|E|)
correction from the framework's cosmological horizon scaling.

HONEST OUTCOME (to be reported below): I cannot find a clean mechanism.
Each candidate either gives wrong N-scaling, wrong sign, or requires
unjustified additional assumptions. The numerical match is striking; the
structural derivation is open.

This probe documents what was attempted and where each candidate fails,
so a future session doesn't repeat the same dead-ends.
"""

from __future__ import annotations

import math
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, REPO_ROOT)

print("=" * 76)
print("A1 d_eff mechanism derivation attempt (2026-05-25)")
print("=" * 76)

# Setup
k_star = 3
N_atoms = 4
E_edges = 6   # |E| = N·k*/2 = 4·3/2 = 6
two_E = 2 * E_edges  # = 12
c_S = 1 / two_E   # = 1/12, Perron-residue

# Empirical target
d_eff_target = 3 + c_S  # = 3.0833...
alpha_target = (d_eff_target - 1) / 4   # = 0.5208

print(f"\nTarget: d_eff = 3 + 1/(2|E|) = 3 + 1/{two_E} = {d_eff_target:.6f}")
print(f"        α = (d_eff − 1)/4 = {alpha_target:.6f}")
print(f"        Empirical α (GUT anchor) = 0.5201 (match within 0.1%)")
print(f"\nMechanism needed: V_horizon(N) ∝ N^(3 + 1/(2|E|))")


# ------------------------------------------------------------------------
# Mechanism candidates and where each fails
# ------------------------------------------------------------------------
print(f"\n{'='*76}")
print("MECHANISM CANDIDATES — each attempted, each failing for the stated reason")
print('='*76)

print(f"""
═══ CANDIDATE 1: Multiplicative substrate-microstate factor ═══

  Idea: Cosmological horizon contains N³ substrate cells, each with 2|E|
  directed-edge microstates. Total microstates: W = 2|E| × N³.

  T equilibrium: u = E_pumped / V_eff. For E_pump ∝ N, V_eff = N³:
    T = (κN/V)^(1/4) = (κN / (2|E| × N³))^(1/4) ∝ (1/(2|E|))^(1/4) × N^(-1/2)

  FAILS: gives a CONSTANT multiplicative factor (1/(2|E|))^(1/4), not a
  power-law correction. T still scales as N^(-1/2). d_eff = 3 exactly,
  not 3 + 1/12.

═══ CANDIDATE 2: Perron-projected pump fraction ═══

  Idea: Only the Perron-singlet fraction of the substrate's pump contributes
  to gauge-readable thermal photons. Effective pump rate = κ × c_S = κ/(2|E|).
  Non-singlet modes go to dark/non-thermal channels.

  T equilibrium: u_thermal = (κ × c_S × N) / (N³)
    T ∝ (c_S × N / N³)^(1/4) = c_S^(1/4) × N^(-1/2)

  FAILS: again a constant multiplicative factor, not a fractional-dim
  correction. Same issue as Candidate 1.

═══ CANDIDATE 3: Multiway-DAG accessible-branch count ═══

  Idea: Effective microstates = unique multiway-DAG branches accessible
  by epoch N. NB walks on k-regular graph: 2^N walks of length N.

  V_eff = N³ × 2^N (substrate volume × branch count)

  This gives EXPONENTIAL volume growth: V_eff ∝ 2^N × N³.
  T ∝ N^(1/4) × 2^(-N/4) → exponentially decaying T.

  FAILS: way too fast cooling. The framework's actual cooling is power-law,
  not exponential. The multiway-DAG branch count doesn't directly enter
  thermal equilibrium because Bayesian compression reduces it.

═══ CANDIDATE 4: Cumulative Perron-projection contributions over time ═══

  Idea: At each substrate tick, the Perron projector contributes c_S to
  some accumulating quantity. Over N ticks: total = N × c_S = N/(2|E|).

  If V_eff ∝ N³ × exp(N × c_S):
    V grows much faster than power-law, gives exponential cooling.

  If V_eff ∝ N³ × (1 + N × c_S)^something:
    For small N × c_S, this is ≈ N³ × const. For large N × c_S, depends on
    the exponent.

  Trying V_eff ∝ N³ × N^c_S = N^(3 + c_S):  ✓ matches target.

  But what gives this scaling? The Perron projector at Γ on B_NB(srs) is
  N-INDEPENDENT. To get an N^c_S contribution to V, the Perron projector
  needs to contribute a SMALL N-dependence — and I don't see what that is.

  FAILS: works numerically but I can't derive the mechanism that gives
  N^c_S from a constant Perron projector.

═══ CANDIDATE 5: Fractional walker-dimension from spectral analysis ═══

  Idea: NB walker on srs has spectral dimension d_s determined by the
  Hashimoto spectrum. Standard srs spectral analysis gives d_s = 3
  (3D crystal). But the Perron eigenvalue (= k-1 = 2) and Ramanujan bulk
  (|h| = √2) suggest a logarithmic correction.

  Spectral dim more carefully: d_s = 2 log_2(k-1) / log_2(k) × (3 for spatial)
  = 2 × log(2)/log(3) × 3 = 2 × 0.631 × 3 = 3.79

  FAILS: gives 3.79, not 3.08. Not the right correction.

═══ CANDIDATE 6: First-order corrections to standard 3D horizon ═══

  Idea: V_horizon = N³ × (1 + δ(N)) where δ(N) is a small correction
  from substrate microstructure.

  For V ∝ N^(3+c_S): δ(N) = N^c_S − 1. At small N, δ ≈ c_S × log(N). At
  large N, δ ≈ N^c_S.

  WHAT WOULD GIVE δ(N) = N^c_S − 1?

  At each substrate tick, some small amount of phase-space "leaks" into
  a higher-dimensional manifold. Over N ticks, the leakage compounds.
  Rate of leakage per tick × N steps = N × leak_rate.

  If leak_rate ∝ c_S, total leakage ∝ N × c_S, but raised to what power?

  For V ∝ exp(c_S × log N) = N^c_S: the leakage is fractional, like a
  Wiener process. Each tick contributes ln(something) × c_S.

  This is consistent with a stochastic-fractal-dimension picture, but
  the framework doesn't have a derived c_S × log(N) per-tick contribution.

  STATUS: undetermined; needs derivation of substrate-microstructure
  fractal contribution.
""")


# ------------------------------------------------------------------------
# What I think is actually going on
# ------------------------------------------------------------------------
print(f"\n{'='*76}")
print("HONEST INTERPRETATION")
print('='*76)
print(f"""
The d_eff = 3 + 1/(2|E|) fit is structurally suggestive but I CANNOT
cleanly derive the mechanism. Each candidate either gives:
  - Wrong N-scaling (Candidates 1, 2, 5)
  - Wrong functional form (Candidate 3 is exponential not power-law)
  - Right form but unjustified assumption (Candidates 4, 6)

THREE HONEST POSSIBILITIES:

(A) The match is real and there IS a clean derivation I'm not seeing.
    The structural elegance of 1/(2|E|) being c_S suggests this. Some
    candidate ideas I didn't fully explore:
      - The cosmological MaxEnt over substrate microstates with a
        log-divergent contribution from the Perron singlet
      - Anomalous dimension from the renormalization-group-like flow of
        the cosmological cascade clock
      - Topological contribution from the substrate's homology

(B) The match is real but the mechanism is more subtle. The 1/(2|E|)
    enters not via horizon volume directly, but via SOME OTHER
    relationship (e.g., the relationship between substrate emission rate
    and horizon temperature, modified by Perron projection of the
    emission spectrum).

(C) The match is numerological in the right range. 1/12 is a small
    number, structurally meaningful primitives in [0.05, 0.15] all give
    similar matches. The empirical α might land near 1/2 + 1/48 by
    accident.

I cannot distinguish (A) from (C) without finding the mechanism.

WHAT WOULD CLOSE A1 PROPERLY:

  A theorem-grade derivation of either:
  - V_horizon(N) ∝ N^(3 + 1/(2|E|)) from substrate primitives, OR
  - T(N) ∝ N^(-1/2 - 1/(8|E|)) directly from horizon-thermal balance
    modified by Perron projection

  Neither is in hand. The empirical fit suggests the answer; the
  mechanism is open.

REMAINING PATHS FORWARD:

  1. Look at how the Perron projector at Γ enters the cosmological
     scale-factor evolution (instead of horizon volume directly).
     The Friedmann analog in the framework might have a c_S factor.

  2. Compute the substrate's MULTIWAY-DAG dimension in coasting more
     carefully. The branching factor might give an effective dimension
     correction.

  3. Investigate whether the 9% T_today residual (predicted 2.47 K vs
     observed 2.725 K) is informative — it might indicate the mechanism
     needs an additional small correction.

  4. Look at substrate-anchor discrepancy (Δα = 0.015 between substrate
     and GUT). This might be a calibration issue or evidence that d_eff
     is N-dependent.

Each of these is bounded scoping work (~2-3 sessions). None guaranteed.

CURRENT STATE OF A1:

  Closest structural match: d_eff = 3 + 1/(2|E|), α match within 0.1%
  Mechanism derivation: OPEN

  The frontier is sharper than at start of session, but A1 has not closed.
  The handoff doc should be updated to reflect: numerical fit found,
  mechanism remains open.
""")
print("=" * 76)
