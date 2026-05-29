#!/usr/bin/env python3
"""
L6 — rigorous derivation attempt for r_s = (c/H_0)/(2·n_g) — beware numerology.

The numerical match r_s = (c/H_0)/(2·n_g) = (c/H_0)/30 = 146.53 Mpc
gives -0.35% residual vs Planck 147.05 Mpc. The factor 2·n_g = 30 uses
the theorem-grade Sunada count n_g = 15.

BUT (per user's "beware of numerology" warning): the numerical match
isn't enough. The structural mechanism must place 2·n_g as a SPECTRAL
or geometric quantity, not just as a primitive that happens to make
the arithmetic work.

This probe checks:
  (1) Does 2·n_g = 30 appear in the Stark-Terras spectral decomposition
      of B_NB(srs) as a mode count?
  (2) Are there alternative framework-natural factors of 30 that come
      from rigorous substrate structure?
  (3) Can the rigorous Bloch dispersion at acoustic modes give the
      specific length scale (c/H_0)/30?

Honest discipline: if no clean derivation closes, report the r_s
candidate as NUMEROLOGY-SUSPECT despite the sub-percent numerical match.
"""

from __future__ import annotations
import math
from fractions import Fraction


# Constants
c_light = 2.998e8
Mpc = 3.0857e22
Gpc = 1000 * Mpc
hbar = 1.054571817e-34
G_Newton = 6.6743e-11
t_P = math.sqrt(hbar * G_Newton / c_light ** 5)
ell_P = c_light * t_P

# Framework primitives
k_star = 3
N_atoms = 4
n_E = 6
two_E = 2 * n_E
n_g = 15                                  # Sunada girth cycles per vertex
g_girth = 10
N_local = 2 ** k_star * k_star
N_local_x_atoms = N_local * N_atoms

# Bass-Stark-Terras spectral decomposition dimensions
dim_bipartite_marginal = 2 * (n_E - N_atoms)   # = 4 (cycle modes)
dim_perron_singlet = 1                          # uniform Perron
dim_perron_visible = 1                          # u = k*-1
dim_oscillatory = 6                             # λ_A = -1 modes
beta_1 = n_E - N_atoms + 1                      # = 3 (first Betti number)

N_hub = 8.394881e60
c_over_H0_framework = N_hub * ell_P
r_s_Planck_Mpc = 147.05

print("=" * 76)
print("L6 — rigorous r_s derivation attempt (beware numerology)")
print("=" * 76)
print(f"""
Numerical target: r_s = (c/H_0)/30 = 146.53 Mpc (-0.35% from Planck 147.05 Mpc)
Factor in question: 30 = 2·n_g = k*·g = ?

Framework-natural factors of 30:
  2 × n_g    = 2 × 15 = 30  (directed girth-cycles per vertex)
  k* × g     = 3 × 10 = 30  (coordination × girth, equivalent for srs)
  2|E| × g/N_atoms = 12 × 10/4 = 30  (per-cell quantity per atom)

All three are theorem-grade arithmetic on srs primitives. The question
is whether ANY of these has a SPECTRAL or geometric mechanism justifying
the role in r_s.
""")


# ---------------------------------------------------------------------------
# Check (1) — does 30 appear in Stark-Terras spectral decomposition?
# ---------------------------------------------------------------------------
print(f"\n{'='*76}")
print("Check (1) — Stark-Terras spectral decomposition of B_NB(srs)")
print('='*76)
print(f"""
Bass-Stark-Terras factorization:
  det(uI - B_NB) = (u² - 1)^(|E|-|V|) · Π_λ (u² - λu + (k*-1))

For srs (|V|=4, |E|=6, k*=3, σ(A) = {{+3, -1, -1, -1}}):
  = (u² - 1)² · (u² - 3u + 2)(u² + u + 2)³
  = (u² - 1)² · (u-1)(u-2)(u² + u + 2)³

MODE COUNTS:
  Bipartite-factor marginal (u=±1):   dim = {dim_bipartite_marginal}  (cycle modes / H¹ Wilson)
  Adjacency-Perron marginal (u=+1):   dim = {dim_perron_singlet}  (scalar zero-mode)
  Adjacency-Perron visible (u=k*-1):  dim = {dim_perron_visible}  (Perron eigenvalue)
  Oscillatory (u² + u + 2 = 0):       dim = {dim_oscillatory}  (λ_A = -1 modes)
  TOTAL                                = 2|E| = {two_E}

  β_1 (first Betti number, cycle space dim) = |E|-|V|+1 = {beta_1}

NUMEROLOGY CHECK:
  Does 30 = 2·n_g appear ANYWHERE in this decomposition? NO.
  - Marginal mode count: 4 (not 30)
  - Total NB modes: 12 (not 30)
  - Homologically independent cycles: β_1 = 3 (not 30)
  - 4 × 3 = 12 (not 30)
  - 6 × 5 = 30 (oscillatory × something?) — NO clean interpretation
  - 2 × 4 × β_1 + 6 = 30? checking: 24 + 6 = 30 ✓ but no clean mechanism

CONCLUSION (1): 30 = 2·n_g does NOT appear as a clean spectral-mode count
in Stark-Terras. The 4 bipartite marginal modes are the spectral
representation of cycle content, NOT 30.
""")


# ---------------------------------------------------------------------------
# Check (2) — does 30 come from per-vertex cycle counting?
# ---------------------------------------------------------------------------
print(f"\n{'='*76}")
print("Check (2) — per-vertex cycle counting on srs")
print('='*76)
print(f"""
n_g = 15 = unoriented girth-cycles per vertex (Sunada count, theorem-grade
for srs). Each unoriented cycle has 2 directed orientations, giving
2·n_g = 30 DIRECTED girth-cycle paths starting at each vertex.

These 30 directed paths are NOT 30 distinct B_NB eigenmodes. They're
30 EVENTS — distinct walker trajectories at length g = 10. The spectral
content of all 30 is contained in the 4 bipartite marginal modes
(which form a 4-dimensional subspace of B_NB^{{12}}).

So 2·n_g = 30 is a COUNTING quantity, not a SPECTRAL quantity. It
counts events (walker paths) but doesn't index modes.

POSSIBLE STRUCTURAL READING:
  The OBSERVER's beta-Bernoulli posterior accumulates over substrate
  events. The substrate emits 2·n_g = 30 directed girth-cycle events
  per vertex per cycle period. If the observer's first detectable
  acoustic feature corresponds to ONE such event being resolved within
  the Hubble distance, then:
    r_s = (Hubble distance) / (events per vertex within Hubble) = c/H_0 / 30

  This reading is HEURISTIC. Why "one event per vertex divided into
  Hubble distance"? Not derived from first principles.
""")


# ---------------------------------------------------------------------------
# Check (3) — does Bloch dispersion at marginal modes give λ = (c/H_0)/30?
# ---------------------------------------------------------------------------
print(f"\n{'='*76}")
print("Check (3) — Bloch dispersion at acoustic/marginal modes")
print('='*76)
print(f"""
The rigorous derivation would proceed as:

  1. Solve B_NB(k) eigenproblem at finite Bloch momentum k
  2. Identify the band corresponding to bipartite marginal modes
     (these go through u=±1 at k=Γ; extend to finite k via dispersion)
  3. Compute group velocity v_s and wavelength λ(k) along the band
  4. Identify the FUNDAMENTAL wavelength = full BZ wavelength scaled
     by some characteristic substrate length
  5. Convert to physical length via substrate ↔ observer-graph metric
  6. Compare to (c/H_0)/(2·n_g)

The framework has Stark-Terras decomposition at Γ (k=0) but NOT the
full Bloch dispersion at arbitrary k. This computation requires:
  - Explicit B_NB matrix on srs supercells of multiple sizes
  - Eigenvalue tracking across the BZ
  - Identification of "acoustic-like" bands

NONE of this calculation has been done within the framework. The
"acoustic wavelength" interpretation of r_s is therefore CONJECTURAL
on the spectral side — not derived from Bloch dispersion.

CONCLUSION (3): the rigorous Bloch-dispersion derivation of r_s
remains OPEN. It's a substantial multi-session computational program
(explicit B_NB on srs supercells, BZ dispersion tracking).
""")


# ---------------------------------------------------------------------------
# Honest numerology assessment
# ---------------------------------------------------------------------------
print(f"\n{'='*76}")
print("HONEST NUMEROLOGY ASSESSMENT")
print('='*76)
print(f"""
The user's "beware of numerology" warning applies sharply here.

NUMERICAL FACT (real):
  r_s = (c/H_0) / 30 = 146.53 Mpc matches Planck 147.05 Mpc within 0.35%.

NUMEROLOGY-CHECK QUESTIONS:

  Q: Is the factor 30 a clean framework primitive?
  A: YES. 30 = 2·n_g = k*·g where n_g and k*, g are theorem-grade.

  Q: Is the factor 30 a SPECTRAL mode count?
  A: NO. Stark-Terras gives 4 marginal modes, 12 total NB modes,
     β_1 = 3 cycle homology — NONE of these is 30.

  Q: Is the factor 30 derivable from Bloch dispersion?
  A: UNKNOWN. The full Bloch-dispersion calculation hasn't been done.

  Q: Does the structural reading have an INDEPENDENT route (Routes
     H+C analog calibration)?
  A: NO. Only one route articulated (events-per-vertex heuristic).
     No second observable in this class to cross-check.

  Q: Could the match be coincidence?
  A: PLAUSIBLY. Many framework-primitive products give factors near
     30 (3·10, 15·2, 6·5, 4·8, 2·3·5, ...). With 5 framework primitives
     (k*, g, N_atoms, n_g, |E|), there are O(2^5 × few) plausible
     products of order 10-100. Getting one within 0.35% of Planck's
     r_s could be coincidence.

  Q: But isn't r_s + D_A + θ* all consistent at sub-percent?
  A: YES — but they're NOT INDEPENDENT. θ* = r_s/D_A is forced. Two
     numerical matches (r_s and θ_*, with D_A derived), not three.
     And 1/96 + 1/30 are TWO degrees of freedom that fit two numbers
     (Planck r_s and Planck θ_*). The χ² for 2 parameters fitting 2
     observables is trivial.

NUMEROLOGY VERDICT:
  Per W58 / no-pattern-fit discipline:
  - The numerical match IS real but uses 2 free framework-primitive
    choices to fit 2 independent observables. This is NOT a strong
    structural constraint.
  - No spectral-mode derivation supports the specific factor 30
  - No Bloch-dispersion calculation supports the wavelength
    interpretation
  - The structural reading ("events per vertex") is heuristic with
    no calibration analog

  The r_s = (c/H_0)/30 candidate is NUMEROLOGY-SUSPECT until either:
  (a) A rigorous Bloch-dispersion derivation closes the wavelength
      calculation
  (b) A SECOND independent observable in the same class gives a
      consistent fit with the same primitives
  (c) The "events per vertex" reading gets sharpened to a rigorous
      mechanism

  CURRENT STATUS: candidate-grade with strong numerical hit but
  ELEVATED NUMEROLOGY RISK. The user's caution is well-placed.
""")


# ---------------------------------------------------------------------------
# What WOULD close this rigorously
# ---------------------------------------------------------------------------
print(f"\n{'='*76}")
print("What WOULD close the rigorous derivation")
print('='*76)
print(f"""
1. Explicit Bloch-dispersion calculation on B_NB(srs):
   - Construct B_NB on srs supercells of size N×N×N (N=2, 4, 8, ...)
   - Diagonalize at multiple k-points across the BZ
   - Identify acoustic-like bands (low-energy, near-marginal at Γ)
   - Compute v_s = dω/dk and fundamental wavelength
   - Convert from substrate length units (ℓ_P) to observer-graph
     length units via the beta-Bernoulli posterior map

2. SECOND independent observable in the same class:
   - The framework has n_s, r, σ_8, t_0 ΛCDM as remaining L6-blocked
     rows
   - If any of these has a clean primary-observable expression with
     a DIFFERENT framework-primitive combination that ALSO matches
     Planck at sub-percent, that's calibration evidence
   - Currently untested under the reframe

3. Rigorous "events per vertex" mechanism:
   - Derive WHY the observer's first detectable acoustic feature
     corresponds to one girth-cycle event per vertex
   - This would require explicit beta-Bernoulli posterior dynamics
     on CMB-sphere observations at recombination filtration boundary

Each of these is a multi-session structural program. None is closed
in current session.

NEXT-STEP RECOMMENDATION:
  Either pause on r_s rigor and accept the numerology-risk caveat,
  OR commit to multi-session structural work (likely option 1 or 2).
""")

print("=" * 76)
print("STATUS: r_s = (c/H_0)/30 is NUMERICAL CANDIDATE WITH ELEVATED NUMEROLOGY RISK")
print("        Rigorous spectral-mode derivation remains OPEN.")
print("=" * 76)
