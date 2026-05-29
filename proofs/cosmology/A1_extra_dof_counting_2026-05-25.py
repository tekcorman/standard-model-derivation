#!/usr/bin/env python3
"""
A1 closure attempt — possibility (B): count framework-native DOFs systematically (2026-05-25).

Following `A1_thermal_scale_handoff_2026-05-25_thread.md` §"What to try":
possibility (B) is 'extra DOFs from substrate-cell counting + multiway-DAG
branch counting could reach ~10^5 cumulative.' This probe tries it.

Target: explain the 39× residual at GUT-anchored horizon-thermal + SM g*_S
via framework-specific DOF count.

Required factor: from the entropy formula
  T_today = T_GUT × (g_GUT/g_today)^(1/3) × √(N_GUT/N_today)
to match observed T_today = 2.725 K (vs SM-only prediction 107 K):

  Required (g_GUT/g_today)^(1/3) = 2.725 / (T_GUT × √(N_GUT/N_today))
                                 = 2.725 / (2.3e29 × 1.54e-28) = 0.0770
  Required g_GUT/g_today = 0.0770^3 = 4.56e-4

Hmm, g_GUT/g_today < 1 means FEWER DOFs at GUT than today. That's
the OPPOSITE of standard SM, where g* increases going back in time.

THIS IS A STRUCTURALLY IMPORTANT FINDING: in the framework's coasting +
horizon-thermal picture, the entropy correction goes the WRONG way
relative to standard SM. With SM g_GUT = 106.75 > g_today = 3.94, the
correction HEATS the prediction (going from 35 K to 107 K), but observed
is COLDER (2.725 K).

So possibility (B) — adding MORE DOFs to g_GUT — makes the discrepancy
WORSE, not better. Unless we add DOFs to g_today, which is bounded by
photon + neutrino content.

This probe verifies the sign issue numerically and reports the verdict.
"""

from __future__ import annotations

import math
import os
import sys
from fractions import Fraction

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, REPO_ROOT)

print("=" * 76)
print("A1 closure attempt — possibility (B): DOF counting (2026-05-25)")
print("=" * 76)

# Constants
k_B = 1.380649e-23
hbar = 1.054571817e-34
eV = 1.602176634e-19
GeV = 1e9 * eV
t_P = 5.391247e-44
N_hub = 8.394881e60
v_today = 246.22
M_unif = 1.98e16
T_CMB_today_K = 2.7255

T_substrate_K = hbar * (2*math.pi/t_P) / (k_B * math.log(2))
T_GUT_K = M_unif * GeV / k_B
N_GUT = N_hub * (v_today / M_unif)**4

# Pure horizon-thermal prediction at GUT anchor
T_today_no_g_correction = T_GUT_K * math.sqrt(N_GUT/N_hub)
print(f"\nWithout any g*_S correction (pure α=1/2 from GUT):")
print(f"  T_today = T_GUT × √(N_GUT/N_hub) = {T_today_no_g_correction:.3f} K")
print(f"  vs observed: {T_CMB_today_K} K")
print(f"  Ratio: {T_today_no_g_correction/T_CMB_today_K:.2f}× too hot")


# ------------------------------------------------------------------------
# The sign issue
# ------------------------------------------------------------------------
print(f"\n{'='*76}")
print("THE SIGN ISSUE")
print('='*76)

required_factor_X = T_CMB_today_K / T_today_no_g_correction  # = (g_GUT/g_today)^(1/3)
required_g_ratio = required_factor_X ** 3

print(f"""
Entropy conservation in coasting + horizon-thermal baseline:
  T_today = T_GUT × (g_GUT/g_today)^(1/3) × √(N_GUT/N_today)

To match observed T_today = 2.725 K, need:
  (g_GUT/g_today)^(1/3) = {required_factor_X:.4f}
  g_GUT/g_today        = {required_g_ratio:.4e}

With g_today (SM at 2.725 K) ≈ 3.94, we'd need:
  g_GUT = {required_g_ratio * 3.94:.4f}

  -- WAY less than 1 DOF, unphysical.

Equivalently: g_today (= today's DOFs) would need to be MUCH LARGER than
g_GUT (= GUT-epoch DOFs). This is the OPPOSITE of standard SM cosmology,
where g* DECREASES going forward in time.

**SM-style adding more DOFs to g_GUT makes the discrepancy WORSE, not
better.** Possibility (B) as originally framed (extra DOFs at the GUT
anchor) goes the wrong direction.
""")


# ------------------------------------------------------------------------
# Reframing: what if extra DOFs are at TODAY's epoch, not GUT?
# ------------------------------------------------------------------------
print(f"{'='*76}")
print("REFRAMING — extra DOFs at TODAY's epoch")
print('='*76)

# If g_today_effective is much larger than 3.94 (standard SM):
g_today_required = 106.75 / required_g_ratio  # ratio reversal
print(f"""
For consistency, if g_GUT is 106.75 (SM standard, no extra GUT DOFs):
  g_today_required = g_GUT / (g_GUT/g_today_required)
                   = 106.75 / {required_g_ratio:.4e}
                   = {g_today_required:.4e}

That's ~10^5 effective DOFs TODAY (vs SM g_today = 3.94).

This is structurally different from standard cosmology. Where could 10^5
'today DOFs' come from in the framework?

CANDIDATE SOURCES (framework-native at the OBSERVER scale, not the
substrate scale):

  (i) Observer's accessible cells at present epoch. At z=0 the observer's
      cosmological-horizon volume contains ~ (c·t_today)³ / (lattice
      cell volume). For c·t_today = c·N_hub·t_P = c × 8.4e60 × 5.4e-44 s
      = c × 4.5e17 s ≈ 1.4e26 m (cosmological horizon). Lattice cell volume
      ~ (Planck length)³ × N_atoms = 4 × (1.6e-35 m)³ ~ 1.6e-105 m³.
      Cells in horizon = 1.4e26³ / 1.6e-105 = 1.8e183. Way more than 10^5.

      So 'observer's horizon contains many cells' is true. The question is
      which of those cells contribute to the OBSERVED thermal radiation.

  (ii) Each cell's substrate microstate count: 2^|E| = 2^6 = 64 per cell.
       But these microstates are coherent (not independent thermal DOFs)
       under A2-T (uniform measure).

  (iii) Multiway-DAG branch states: 2^N_observations. At N_hub, this is
        2^(8.4e60), astronomically large. But branches collapse under
        observer's Bayesian-walk integration.

  (iv) Cl(6) Fock × walker types × generation: 16 × 4 × 3 = 192. Modest;
       roughly 50× more than SM g_today = 3.94. Times 10^3 still short.

NONE of these gives a clean 10^5 without large speculative factors.
""")


# ------------------------------------------------------------------------
# Sharper alternative: maybe the formula is wrong
# ------------------------------------------------------------------------
print(f"{'='*76}")
print("ALTERNATIVE: maybe the entropy-conservation + horizon-thermal formula is wrong")
print('='*76)
print(f"""
The standard entropy-conservation formula T·g*^(1/3)·a = const assumes
adiabatic evolution: total entropy is conserved, just redistributed across
the changing DOFs.

The framework's coasting cosmology with horizon-thermal pumping is NOT
adiabatic. The substrate continuously pumps energy into the horizon at
rate κ/t_P. This is a NON-CONSERVATIVE source term.

Under non-adiabatic conditions:
  ds/dt = (substrate pump entropy rate) - (entropy expansion dilution)

If the substrate pump compensates the expansion (steady-state):
  T(N) is set by RATE BALANCE, not by entropy conservation.

In that picture, T(N) is set by ratio (pump rate / expansion rate):
  T^4 ∝ (κ/t_P) / (c³·N²) → T ∝ N^(-1/2)   [horizon-thermal, no g*]

g*_S corrections to entropy DON'T apply in the rate-balance picture
because there's no conserved S to redistribute.

THIS WOULD EXPLAIN WHY ADDING g* CORRECTIONS GOES THE WRONG DIRECTION
in our analysis: the framework's coasting cosmology operates in rate-
balance regime, not entropy-conservation regime.

Under pure rate balance: T_today = T_GUT × √(N_GUT/N_hub) ≈ 35 K.
That's 13× off observation, not 39×. Better than SM g* application.

But still 13× off. Need additional mechanism for the residual.

POSSIBILITY (A) revisited — d_eff > 3:
  T ∝ N^((1-d_eff)/4) — for T_today = 2.725 K with T_GUT = 2.3e29 K:
  log(2.725/2.3e29) = (1-d_eff)/4 × log(N_hub/N_GUT)
  log(1.185e-29) / log(4.2e55) = (1-d_eff)/4
  -28.926 / 55.624 = (1-d_eff)/4
  d_eff = 1 + 4 × 0.520 = 3.080

  So d_eff = 3.08, very close to standard 3D. The 0.08 deviation could
  come from anomalous geometric dimension of the substrate or the
  observer's horizon shape.
""")

# Verify
d_eff_empirical = 1 + 4 * math.log(T_CMB_today_K / T_GUT_K) / math.log(N_GUT / N_hub)
print(f"Empirical d_eff (no g* correction, GUT anchor): {d_eff_empirical:.4f}")
print(f"Deviation from 3: {d_eff_empirical - 3:.4f}")


# ------------------------------------------------------------------------
# Verdict
# ------------------------------------------------------------------------
print(f"\n{'='*76}")
print("VERDICT")
print('='*76)
print(f"""
Possibility (B) — extra framework DOFs — DOES NOT CLOSE A1 cleanly.

The standard entropy-conservation formula, when applied with SM g*_S
freeze-outs, produces a sign-mismatch: adding DOFs at GUT makes the
prediction HOTTER, not colder. So extra DOFs at GUT cannot explain the
overshoot.

Reframing as 'extra DOFs at today's epoch' requires ~10^5 today DOFs,
which has no clean framework-native source.

THE BETTER FRAMING IS:
  - Framework cosmology is NOT adiabatic — substrate pumping is a
    non-conservative source.
  - Entropy conservation formula T·g^(1/3)·a = const DOES NOT APPLY.
  - In the rate-balance regime, T(N) is set by horizon-thermal balance
    alone: T ∝ N^(-1/2), no g*_S corrections.
  - Pure rate-balance from GUT anchor: T_today = 35 K, off by 13×.

The 13× residual under rate-balance is closer to closure than the 39×
under entropy-conservation+SM. It points to possibility (A) — anomalous
effective dimensionality d_eff = 3.08, where the 0.08 deviation captures
the 13× factor.

POSSIBILITY (A) IS NOW THE LIVE CANDIDATE for A1 closure:
  - Derive d_eff for framework substrate cosmological horizon
  - The 0.08 deviation should come from a small structural correction
  - If derivable, A1 closes WITHOUT needing extra DOFs

The DOF-counting program (possibility B) is RETIRED as a primary attack.
The dimensionality program (possibility A) is the surviving direction.

CONCRETE NEXT-STEP for possibility (A):
  - Look at the framework's substrate horizon volume scaling as N → ∞.
    Standard coasting + flat 3D gives V ∝ N^3.
    Framework's substrate could have small fractional correction from:
      * Multiway-DAG branching factor
      * srs lattice's effective dimension (k*=3, but spectral dim varies)
      * Cosmological-cascade non-isotropy
  - A 0.08 deviation in d_eff is small but specific. Bounded scoping
    target: derive it (or rule it out) from substrate primitives.
""")
print("=" * 76)
