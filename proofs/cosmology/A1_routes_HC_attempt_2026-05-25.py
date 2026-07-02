#!/usr/bin/env python3
"""
A1 Routes H/C derivation attempt for c_g(T_propagation) (2026-05-25).

Goal: derive c_g for the temperature observable T via Routes H + C in
parallel to α_GUT (c=1/k*) and v_Higgs (c=5/12), with the strict
discipline that BOTH routes must give the same value, AND the same
mechanism must reproduce v_Higgs c=5/12 on calibration check.

Target: c_g(T) = N_trivial = 2 (the candidate from D3) — if Routes H+C
confirm this, A1 closes at theorem-grade-numerical.

Template (master doc §4):

  Route H: c = dim(observable's selected marginal sector) / dim(NB total)
  Route C: c = (observable's substrate-coupling numerator) / (N_atoms × k*²)

For srs (|V|=4, |E|=6, k*=3, |E|-|V|=2, 2|E|=12, N_atoms·k*² = 36):

  Mode catalogue from Stark-Terras factorization:
    - Bipartite-factor marginal at u=±1: dim 2(|E|-|V|) = 4
      (cycle modes, H¹ Wilson-loop content, gauge-CHARGED)
    - Adjacency-Perron marginal at u=+1: dim 1
      (uniform-on-directed-edges Perron vector, gauge-SINGLET / scalar)
    - Adjacency-Perron visible at u=k*-1=2: dim 1
    - Oscillatory (λ_A=-1 factors): dim 6
    - TOTAL NB: 2|E| = 12

  Observable-class selection rules (from master doc):
    - α_GUT (gauge 1-pt): selects bipartite marginal only → c = 4/12 = 1/3
    - v_Higgs (scalar 2-pt): selects bipartite + scalar → c = 5/12
    - δ_r (Z channel): selects Perron-residue singlet → c_S = 1/12

QUESTION: which selection rule applies to T (temperature)?

Three observable-class candidates for T:

  T1. Thermal-bath observable (sees ALL pump-pumped modes)
      → if substrate pump touches all 2|E| edges, c_g(T) = 12/12 = 1
      → single c=1 application gives only -4% residual (P2, P4)

  T2. Substrate-thermal-pump propagator (single-mode trivial sector)
      → if substrate pump goes through trivial-isotypic propagator,
        c_g(T) = N_trivial / (something)
      → need to identify the denominator

  T3. Photon-thermal-channel observable (gauge-readable thermal radiation
      coupling to U(1)_EM gauge boson)
      → photon = gauge boson, observable class = gauge 1-pt
      → c_g(T) = 1/k* (same as α_GUT) → gives -7% (P3, doesn't close)

This probe attempts each in turn, with the strict discipline:
  - Both Routes H + C must give same value
  - Calibration check: same mechanism must give v_Higgs 5/12
  - K-rational test (master doc §8 rule 4)
  - Falsify if any check fails
"""

from __future__ import annotations
import math
from fractions import Fraction


# Framework primitives
k_B = 1.380649e-23
GeV = 1.602176634e-10
K_per_GeV = GeV / k_B
M_Pl_GeV = 1.220890e19

N_hub = 8.394881e60
v_today = 246.22
T_CMB = 2.7255

k_star = 3
N_atoms = 4
n_E = 6  # |E| = 6 for srs
N_trivial = 2
n_g = 15  # girth cycles per vertex (Sunada)

two_E = 2 * n_E   # = 12
N_couplings = N_atoms * k_star ** 2  # = 36

# Mode catalogue (from Stark-Terras for srs)
dim_bipartite_marginal = 2 * (n_E - N_atoms)  # = 4 (cycle modes, H¹)
dim_perron_singlet = 1                         # uniform Perron vector
dim_perron_visible = 1                         # u = k*-1 = 2
dim_oscillatory = 6                            # u = (-1 ± i√7)/2

assert dim_bipartite_marginal + dim_perron_singlet + dim_perron_visible + dim_oscillatory == two_E

alpha_GUT_bare = Fraction(1, 24)
alpha_1_bare = Fraction(2, 3) ** 8
c_S = Fraction(1, two_E)
waterline = alpha_1_bare / (1 - alpha_1_bare)
w = float(waterline)

# Baseline T_today
M_unif = float(alpha_GUT_bare * alpha_1_bare) * M_Pl_GeV
N_GUT = N_hub * (v_today / M_unif) ** 4
T_baseline = M_unif * float(c_S) * K_per_GeV * math.sqrt(N_GUT / N_hub)

X_target = T_CMB / T_baseline  # ≈ 0.9228
required_c_single = (1 - X_target) / w  # ≈ 1.90 if single application
required_c_squared_root = (1 - math.sqrt(X_target)) / w  # ≈ 0.97 if (1-cw)²

print("=" * 76)
print("A1 ROUTES H/C ATTEMPT — c_g(T) derivation for propagation closure")
print("=" * 76)
print(f"\nFramework primitives:")
print(f"  k* = {k_star}, N_atoms = {N_atoms}, |E| = {n_E}, 2|E| = {two_E}")
print(f"  N_trivial = {N_trivial} (single-mode propagator, M_unif Stage 4)")
print(f"  N_couplings = N_atoms × k*² = {N_couplings}")
print(f"  n_g = {n_g} (girth cycles per vertex, Sunada)")
print(f"  α₁ = (2/3)^8 = {float(alpha_1_bare):.6f}, w = α₁/(1-α₁) = {w:.6f}")

print(f"\nMode catalogue (Stark-Terras factorization on srs):")
print(f"  Bipartite marginal (cycle / H¹):     dim = {dim_bipartite_marginal}")
print(f"  Perron-adjacency singlet (scalar):   dim = {dim_perron_singlet}")
print(f"  Perron-adjacency visible:             dim = {dim_perron_visible}")
print(f"  Oscillatory:                          dim = {dim_oscillatory}")
print(f"  Total NB:                             dim = {two_E}")

print(f"\nCALIBRATION (the target c_g values from the master doc catalogue):")
print(f"  α_GUT (gauge 1-pt):     c = (cycle only) / 2|E| = {dim_bipartite_marginal}/{two_E} = {Fraction(dim_bipartite_marginal, two_E)} = 1/k*")
print(f"  v_Higgs (scalar 2-pt):  c = (cycle + scalar) / 2|E| = {dim_bipartite_marginal+dim_perron_singlet}/{two_E} = {Fraction(dim_bipartite_marginal+dim_perron_singlet, two_E)} = 5/12")
print(f"  δ_r (Z-channel):        c = (Perron singlet) / 2|E| = {dim_perron_singlet}/{two_E} = {Fraction(dim_perron_singlet, two_E)} = 1/12")

print(f"\nClosure target for T:")
print(f"  Need X = {X_target:.5f} on T_today")
print(f"  Single application:  c = (1-X)/w = {required_c_single:.4f}")
print(f"  Squared application: c = {required_c_squared_root:.4f}")


# ---------------------------------------------------------------------------
# T-class candidate T1: T sees ALL pump-pumped modes (substrate pump = thermal bath)
# ---------------------------------------------------------------------------
print(f"\n{'='*76}")
print("T1 — T sees all pump-pumped modes (substrate-thermal-bath class)")
print('='*76)
print(f"""
Hypothesis: the substrate pump κ touches every edge every tick. T at
the substrate level is determined by total energy / horizon volume, with
contributions from EVERY NB mode (no selection — thermal bath sees all).

Route H: c = (all NB modes) / 2|E| = {two_E}/{two_E} = 1
Route C: c = (all walker steps) / N_couplings
         = {two_E}/{N_couplings} = {Fraction(two_E, N_couplings)} = 1/k*

ROUTES DISAGREE: Route H gives 1, Route C gives 1/k*. FALSIFIED by the
two-route discipline (master doc §8 rule 2: 'any new c_g must close via
two routes giving the same value').

This rules out the "T sees all modes" reading as the dark-correction
mechanism for T.
""")
T_T1_H = T_baseline * (1 - 1 * w)
T_T1_C = T_baseline * (1 - (1/3) * w)
print(f"  T_today (T1 Route H, c=1):    {T_T1_H:.4f} K  ({(T_T1_H-T_CMB)/T_CMB*100:+.2f}%)")
print(f"  T_today (T1 Route C, c=1/3):  {T_T1_C:.4f} K  ({(T_T1_C-T_CMB)/T_CMB*100:+.2f}%)")
print(f"  STATUS: FALSIFIED — Routes disagree.")


# ---------------------------------------------------------------------------
# T-class candidate T2: T as substrate-thermal-pump propagator (trivial sector)
# ---------------------------------------------------------------------------
print(f"\n{'='*76}")
print("T2 — T as substrate-thermal-pump propagator (trivial-isotypic class)")
print('='*76)
print(f"""
Hypothesis: T is set by the substrate pump rate κ, which flows through
the trivial-isotypic sector of the K_4 vertex stabilizer (the same
sector that gives M_R = N_trivial × (1/k*)^(g-1) × M_Pl in M_unif
Stage 4). The trivial isotypic has N_trivial = 2 dim.

Route H: c = N_trivial / 2|E| = {N_trivial}/{two_E} = {Fraction(N_trivial, two_E)} = 1/6
   (selecting the trivial-isotypic modes from the full NB sector)

Route C: c = N_trivial / N_couplings = {N_trivial}/{N_couplings} = {Fraction(N_trivial, N_couplings)} = 1/18
   (per-cell trivial-isotypic mode count over A2 edge-process count)

ROUTES DISAGREE: Route H gives 1/6, Route C gives 1/18. FALSIFIED.

NEITHER gives N_trivial = 2 — both give a FRACTION involving N_trivial,
not N_trivial itself. The D3 candidate c_g(T) = N_trivial = 2 is NOT
recoverable via the standard Routes H/C templates (which always give
fractions ≤ 1).

The c_g value of 2 from D3 was actually a SHORTHAND for (1-2w) ≈ (1-w)²,
NOT a clean Route-H/C c_g. So D3's "c_g = N_trivial" reading is
structurally MISIDENTIFIED.
""")
T_T2_H = T_baseline * (1 - (1/6) * w)
T_T2_C = T_baseline * (1 - (1/18) * w)
print(f"  T_today (T2 Route H, c=1/6):  {T_T2_H:.4f} K  ({(T_T2_H-T_CMB)/T_CMB*100:+.2f}%)")
print(f"  T_today (T2 Route C, c=1/18): {T_T2_C:.4f} K  ({(T_T2_C-T_CMB)/T_CMB*100:+.2f}%)")
print(f"  STATUS: FALSIFIED — Routes disagree, neither closes.")


# ---------------------------------------------------------------------------
# T-class candidate T3: T as photon-thermal observable (gauge 1-pt class)
# ---------------------------------------------------------------------------
print(f"\n{'='*76}")
print("T3 — T as photon-thermal-channel observable (gauge 1-pt class)")
print('='*76)
print(f"""
Hypothesis: the CMB photon is a U(1)_EM gauge boson; T as the
thermal scale of the photon bath inherits α_GUT's observable class
(gauge 1-pt). c_g(T) = 1/k* (same as α_GUT, both routes match).

Route H: c = (cycle marginal) / 2|E| = {dim_bipartite_marginal}/{two_E} = 1/k* ✓
Route C: c = 2|E| / N_couplings = 1/k* ✓

ROUTES AGREE: c = 1/k* = 1/3. Calibration check: this is α_GUT's c_g,
which already passes the v_Higgs 5/12 calibration via the selection rule
(c_v includes +1 Perron singlet for scalar 2-pt, c_α_GUT excludes it for
gauge 1-pt).

But: c = 1/3 gives only -7% residual (P3, doesn't close to 0.5%).
""")
T_T3 = T_baseline * (1 - (1/3) * w)
print(f"  T_today (T3, c=1/k*): {T_T3:.4f} K  ({(T_T3-T_CMB)/T_CMB*100:+.2f}%)")
print(f"  STATUS: Routes AGREE but NUMERICAL RESIDUAL +6.9% (does not close)")


# ---------------------------------------------------------------------------
# T-class candidate T4: T sees scalar 2-pt class (like v_Higgs, c=5/12)
# ---------------------------------------------------------------------------
print(f"\n{'='*76}")
print("T4 — T as scalar dim-1 scale (v_Higgs class)")
print('='*76)
print(f"""
Hypothesis: T is dim-1 (units of energy = mass) and so is v_Higgs. If
T inherits v_Higgs's observable class (scalar 2-pt), then c_g(T) = 5/12
via the same Stark-Terras+scalar-zero-mode mechanism.

Route H: c = (cycle + scalar) / 2|E| = {dim_bipartite_marginal+dim_perron_singlet}/{two_E} = 5/12
Route C: c = n_g / N_couplings = {n_g}/{N_couplings} = 5/12

ROUTES AGREE: c = 5/12. Calibration: this IS v_Higgs's calibration. ✓

But: c = 5/12 = 0.417 still doesn't close.
""")
T_T4 = T_baseline * (1 - (5/12) * w)
print(f"  T_today (T4, c=5/12): {T_T4:.4f} K  ({(T_T4-T_CMB)/T_CMB*100:+.2f}%)")
print(f"  STATUS: Routes AGREE but NUMERICAL RESIDUAL +6.6% (does not close)")


# ---------------------------------------------------------------------------
# Honest verdict
# ---------------------------------------------------------------------------
print(f"\n{'='*76}")
print("HONEST VERDICT — Routes H/C derivation for c_g(T)")
print('='*76)
print(f"""
Tested four T observable-class candidates:

  T1 (substrate-thermal-bath, c=1 if Route H):
     Route H gives 1, Route C gives 1/k* — FALSIFIED by two-route discipline.

  T2 (trivial-isotypic propagator, c=N_trivial=2):
     Route H gives 1/6, Route C gives 1/18 — FALSIFIED. Neither gives 2.
     The D3 "c_g = N_trivial" reading was structurally MISIDENTIFIED:
     the standard Routes H/C always produce fractions ≤ 1, not c=2.

  T3 (gauge 1-pt / photon class, c=1/k*):
     Routes AGREE on 1/k* (calibrated by α_GUT). Calibration ✓.
     But numerically gives only -7% residual — does NOT close to 0.5%.

  T4 (scalar dim-1 / v_Higgs class, c=5/12):
     Routes AGREE on 5/12 (calibrated by v_Higgs). Calibration ✓.
     But numerically gives only -6% residual — does NOT close to 0.5%.

KEY FINDING — the (1-w)² ≈ (1-2w) numerical closure of A1 CANNOT BE
RECOVERED via Routes H/C with any standard observable class:

  - Standard Routes H/C produce c_g ∈ [1/12, 5/12] for the catalogued
    classes (Z-channel 1/12, gauge 1-pt 1/3, scalar 2-pt 5/12). All
    fractions, none give c=2.
  - The (1-w)² form requires either c=2 single application (not in
    Routes H/C range) or two distinct (1-w) factors from independent
    mechanisms.

So the Routes H/C derivation for A1 does NOT close at theorem-grade.

POSSIBLE INTERPRETATIONS:

  (a) The (1-w)² numerical match is a COINCIDENCE — A1 has a real
      structural residual (~8%) that is NOT a dark correction, and the
      0.25% closure of (1-w)² is numerology.

  (b) A1 needs a NEW observable class beyond the master-doc catalogue —
      perhaps "thermodynamic conjugate variable" (T is conjugate to S via
      dE=TdS) requires a different selection rule on B_NB.

  (c) The anchor T_GUT = M_unif × c_S is structurally WRONG, and the
      correct anchor gives different baseline that closes via a standard
      c_g — but then the c_S identification (the third c_S reading
      alongside δ_r and A_s) loses its appeal.

  (d) The propagation is NOT pure α=1/2 horizon-thermal — there's
      additional N-dependent dark correction along the way that integrates
      to (1-w) over many e-folds. Requires a separate cumulative-dark-
      sector derivation, not the universal template.

VERDICT:
  Routes H/C derivation attempt: HONEST NEGATIVE.

  The clean structural framework via the dark-correction master doc
  does NOT close A1's propagation residual. The (1-w)² numerical match
  remains a CANDIDATE without theorem-grade derivation.

  This is a STRUCTURAL FINDING: A1's propagation correction (if it
  exists structurally) is OUTSIDE the dark-correction universal
  template's reach with the existing observable-class catalogue.

  Per the W58/3-point-fit feedback discipline, the (1-w)² numerical
  hit should NOT be promoted to a "structural candidate" without a
  clear mechanism. It stays as a SUGGESTIVE NUMERICAL OBSERVATION.

NEXT STEPS (honest options):

  1. Pivot to candidate (b): scope a "thermodynamic conjugate variable"
     observable class with its own selection rule on B_NB. Multi-session
     research, no guarantee. Same epistemic class as the original A1
     candidate survey.

  2. Pivot to candidate (c): re-examine the anchor T_GUT = M_unif × c_S
     to see if a different structural identification (e.g., the §4.5
     unified-oblique S, U, or Δκ relatives) gives a better baseline.

  3. Park A1 at the current state ("CANDIDATE — 8% residual at
     theorem-grade-structural anchor, propagation closure mechanism
     open") and pivot to a different open item.

  4. Accept the (1-w)² match as a numerical coincidence (the most
     honest reading given Routes H/C FALSIFY the clean derivations).

This is what the Routes H/C attempt delivers: an HONEST NEGATIVE on the
straightforward structural derivation, sharpening the open question
from "find c_g(T)" to "find a mechanism OUTSIDE the universal template
that gives the right propagation correction."
""")

print("=" * 76)
print("STATUS: HONEST NEGATIVE — Routes H/C do not close c_g(T)")
print("=" * 76)
