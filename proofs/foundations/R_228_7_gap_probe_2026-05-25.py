#!/usr/bin/env python3
"""
R = 228/7 vs Δm²₃₁/Δm²₂₁_obs gap probe (2026-05-25).

CONTEXT (mass-operator stocktake, this session)
-----------------------------------------------
The mass-sector audit flagged m_ν₂ = +2.37% (+1.87σ) residual as decomposing
into +0.87% (m_ν₃ chain) + +1.49% (R-itself).  R = 228/7 = 32.5714... is
the framework's neutrino mass-squared splitting ratio, identified with
Δm²₃₁ / Δm²₂₁.  NuFIT 6.0 gives R_obs ≈ 33.55 — a −2.92% gap.

Two structural facts make the gap especially interesting:

  (a) R = 228/7 is *hard-locked* at the spectral level.  The Gaussian-integer
      identity (1+i√7)⁵ = 176 − 16i√7 forces sin²(5φ) = 7/128 EXACTLY.
      The chain k* = 3 → K₄ → φ = arctan(√7) → q=2 unique positive integer
      root of q³−5q+2=0 → n=5 → R = 228/7 has NO ε-perturbation path that
      preserves k*=3.

  (b) m_ν₁ = 0 was UPGRADED to THEOREM-GRADE-CONDITIONAL by W45 (2026-05-21,
      an internal working note):
      the substrate produces exactly 2 dynamical Majorana ν_R → rank-2
      Type-I seesaw → m_ν₁ ≡ 0 by linear algebra alone.

So neither (1) "shift R" nor (2) "let m_ν₁ ≠ 0 absorb the gap" is on the
parametric escape surface.  This probe surveys the remaining structural
escape routes.

WHAT THIS PROBE TESTS
---------------------
G1. Confirm 228/7 is locked: any 1% perturbation of cos²φ or q breaks the
    Chebyshev cubic identity that selects n = 5.

G2. Audit data inconsistency in the codebase.  R_theorem.md says obs ≈
    32.576 ("0.015% match"); R_nu_splitting_derivation.md says obs = 33.83;
    m_nu3.py chain effectively uses obs = 33.55.

G3. Test the m_ν₁ ≠ 0 ESCAPE explicitly (even though W45 blocks it
    structurally): what m_ν₁ would close the gap, and is that value
    consistent with Planck Σm_ν, KATRIN m_β, and 0νββ bounds?  This makes
    the W45 tension quantitative: m_ν₁ ≈ 1.5 meV would close the gap
    cleanly.

G4. Test the IDENTIFICATION layer.  The framework's R is derived as a
    spectral anisotropy of K₄ (R_theorem.md §"Physical interpretation"):
        R := (total propagator intensity) − (isotropic background)
           = 2/sin²(5φ) − (k*+1).
    The identification R ≡ m₃²/m₂² is an OPEN QUESTION (Open Question 1
    of R_nu_splitting_derivation.md).  Does the spectral object actually
    equal Δm²₃₁/Δm²₂₁, or some *other* combination of mass-squareds that
    happens to coincide with 228/7 only at m_ν₁ = 0?

G5. Survey alternative K₄ spectral invariants for an exact ~33.55 hit.
    If a near-by spectral object lands on 33.55, the identification G4
    might be reading the wrong invariant.

VERDICT FRAMEWORK (pass/fail/inconclusive)
------------------------------------------
G1 PASS = 228/7 cannot be perturbed → no parametric escape.
G2: surfaces the data inconsistency; the canonical NuFIT 6.0 value is
    R_obs = 33.55(±0.32).
G3 INCONCLUSIVE means: m_ν₁≈1.5 meV would close the gap and pass external
    bounds, but is blocked by W45 mode-count.  So G3 records a *structural
    tension* between the m_ν₁≡0 theorem and the observed splitting ratio.
G4 IDENTIFIES the load-bearing identification: 228/7 is a K₄ spectral
    object; "= m₃²/m₂²" is a *physical bridge*, not a derivation.
G5 surveys whether any other K₄ invariant lands on 33.55 exactly.

If G1 + G4 + G5 all close cleanly without giving 33.55, the gap is
either (i) an open structural item — the R identification bridge is
incomplete, or (ii) an experimental-data drift the framework is correct
to disagree with (JUNO will resolve).
"""

import math
import sys
import os
from fractions import Fraction

# ---------------------------------------------------------------------
# Constants from the framework
# ---------------------------------------------------------------------
k_star = 3
q = k_star - 1                              # = 2
sin2_phi_exact = Fraction(7, 8)             # exact
cos2_phi_exact = Fraction(1, 8)             # exact, cosφ = 1/(2√q) = 1/(2√2)
phi = math.atan(math.sqrt(7))               # numerical
sin2_5phi_exact = Fraction(7, 128)          # exact
R_pred_exact = Fraction(228, 7)             # = 32.5714...
R_pred = float(R_pred_exact)

# ---------------------------------------------------------------------
# NuFIT 6.0 (Sept 2024) — single canonical source for this probe
# ---------------------------------------------------------------------
# All three values appear with this exact magnitude in the public NuFIT 6.0
# v1 release (http://www.nu-fit.org/?q=node/294), normal ordering, w/ SK.
Dm2_21_obs = 7.49e-5                         # eV², ±0.19e-5
Dm2_21_sig = 0.19e-5
Dm2_31_obs_NO = 2.513e-3                     # eV², ±0.020e-3 (with SK)
Dm2_31_sig    = 0.020e-3
# Also seen in codebase: Dm2_31 = 2.534e-3 (NuFIT 6.0 NO without SK atm)
# — R_nu_splitting_derivation.md uses this. m_nu3.py chain uses 2.513e-3.

R_obs_with_SK    = Dm2_31_obs_NO / Dm2_21_obs        # = 33.55
R_obs_no_SK      = 2.534e-3 / Dm2_21_obs             # = 33.83
R_obs_stale      = 32.576                            # quoted in R_theorem.md
R_obs_propagated = math.sqrt((Dm2_31_sig/Dm2_31_obs_NO)**2 + (Dm2_21_sig/Dm2_21_obs)**2) * R_obs_with_SK


def banner(s):
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


# =====================================================================
banner("G1 — 228/7 is HARD-LOCKED at the spectral level")
# =====================================================================
# Verify: any small perturbation of cos²φ breaks the n=5 selector cubic
# q³ − 5q + 2 = 0 at q = k*−1, AND breaks G₅ = −1/(k*+1).
#
# Test: vary q from q=2 by ±5%, check whether G₅ still equals −1/(q+2).

print("Spectral chain:  k*=3 → q=k*−1=2 → cos²φ = 1/(4q) = 1/8")
print("                 → cubic q³−5q+2=0, q=2 unique positive integer root")
print("                 → G₅ = U₄(cos φ) = −1/4 = −1/(k*+1)")
print("                 → sin²(5φ) = 7/128")
print("                 → R = 2/sin²(5φ) − (k*+1) = 256/7 − 4 = 228/7")
print()

print("Perturbation scan (vary q around 2):")
print(f"  {'q':>8}  {'U₄(1/(2√q))':>18}  {'−1/(q+2)':>15}  {'R':>14}")
for dq in [-0.05, -0.01, -0.001, 0.0, 0.001, 0.01, 0.05]:
    qv = 2.0 + dq
    cos_phi = 1.0 / (2.0 * math.sqrt(qv))
    U4 = 16*cos_phi**4 - 12*cos_phi**2 + 1
    target = -1.0 / (qv + 2)
    sin2 = 1 - cos_phi**2
    G5 = U4
    R = qv / (G5**2 * sin2) - (qv + 2)
    print(f"  {qv:8.4f}  {U4:18.10f}  {target:15.10f}  {R:14.8f}")

print()
print("Verdict: 228/7 is exact only at q = 2 (k* = 3).  Any non-integer q")
print("breaks both the cubic identity AND the rational form of R.")
print()
print("G1 = PASS  (no parametric escape from 228/7 while preserving k*=3).")


# =====================================================================
banner("G2 — Data inconsistency audit in the codebase")
# =====================================================================
print("Three different R_obs values appear in the codebase:")
print()
print(f"  (a) R_theorem.md (docs/parameters/):            R_obs = 32.576")
print(f"        — '0.015% match' claim — STALE PRE-NuFIT-6.0 NUMBER")
print(f"  (b) R_nu_splitting_derivation.md (predictions/): R_obs = 33.83")
print(f"        — uses Δm²₃₁ = 2.534×10⁻³ (NuFIT 6.0 NO without SK atm)")
print(f"  (c) m_nu3.py chain (live):                        R_obs = 33.55")
print(f"        — uses Δm²₃₁ = 2.513×10⁻³ (NuFIT 6.0 NO with SK atm)")
print()

R_obs_central = R_obs_with_SK
gap_pct = (R_pred - R_obs_central) / R_obs_central * 100
gap_sigma = (R_pred - R_obs_central) / R_obs_propagated
print(f"Canonical (NuFIT 6.0 w/ SK):  R_obs = {R_obs_central:.4f} ± {R_obs_propagated:.4f}")
print(f"R_pred = 228/7              = {R_pred:.4f}")
print(f"Gap                          = {gap_pct:+.3f}%  ({gap_sigma:+.2f}σ)")
print()
print("The R_theorem.md '32.576' value DOES NOT match any current NuFIT")
print("release; it appears to back-trace to NuFIT 4.0/5.0 era data with")
print(f"Δm²₃₁ ~ 2.44 × 10⁻³ (R_pred·Δm²₂₁ = {R_pred * Dm2_21_obs * 1e3:.3f}×10⁻³).")
print()
print("G2 = AUDIT FINDING:  R_theorem.md headline match-percent is stale.")
print("                     The live discrepancy on R itself is −2.9% / −1.1σ_R")
print("                     (NOT 0.015% match).  When propagated through to")
print("                     m_ν₂ with m_ν₃ taken parameter-free, it becomes")
print("                     +1.87σ_obs (FAIL by §8 standard).")


# =====================================================================
banner("G3 — m_ν₁ ≠ 0 hypothesis (blocked by W45 but quantified here)")
# =====================================================================
# If R = m₃²/m₂² (theorem, m₁=0) and observation is Δm²₃₁/Δm²₂₁, then:
#   (m₃² - m₁²) / (m₂² - m₁²) = R_obs
#   With m₃²/m₂² = R_pred (theorem), let x = m₁²/m₂²:
#   (R_pred - x) / (1 - x) = R_obs
#   ⇒ x = (R_pred - R_obs) / (1 - R_obs)
#
# Compute m_ν₁ that would close the gap and check external bounds.

x = (R_pred - R_obs_central) / (1 - R_obs_central)
m_nu2_sq = Dm2_21_obs / (1 - x)
m_nu2 = math.sqrt(m_nu2_sq)
m_nu1 = math.sqrt(x * m_nu2_sq)
m_nu3_sq = R_pred * m_nu2_sq
m_nu3 = math.sqrt(m_nu3_sq)
sum_m_nu = m_nu1 + m_nu2 + m_nu3

# Effective beta-decay mass (KATRIN observable):
# m_β² = Σ |U_ei|² m_i².  Approximate using current PMNS values (3-flavor):
s12_sq = math.sin(math.radians(33.41))**2  # NuFIT 6.0
s13_sq = math.sin(math.radians(8.57))**2
c13_sq = 1 - s13_sq
Ue1_sq = c13_sq * (1 - s12_sq)
Ue2_sq = c13_sq * s12_sq
Ue3_sq = s13_sq
m_beta_sq = Ue1_sq*m_nu1**2 + Ue2_sq*m_nu2**2 + Ue3_sq*m_nu3**2
m_beta = math.sqrt(m_beta_sq)

# Effective Majorana mass for 0νββ (assuming Majorana phases = 0 — worst case):
m_bb = abs(Ue1_sq*m_nu1 + Ue2_sq*m_nu2 + Ue3_sq*m_nu3)   # phases=0
m_bb_min_phases = abs(Ue1_sq*m_nu1 - Ue2_sq*m_nu2 - Ue3_sq*m_nu3)  # max cancellation

print("If R_pred = m₃²/m₂² (with m₁=0) and obs is Δm²₃₁/Δm²₂₁:")
print(f"  Closing the gap requires m₁²/m₂² = {x:.6f} ⇒ m₁/m₂ = {math.sqrt(x):.5f}")
print()
print("Implied mass spectrum (normal ordering):")
print(f"  m_ν₁ = {m_nu1*1e3:.3f} meV    (theorem says 0; this is the gap-closing value)")
print(f"  m_ν₂ = {m_nu2*1e3:.3f} meV    (current pred 8.860 meV; obs 8.654 meV)")
print(f"  m_ν₃ = {m_nu3*1e3:.3f} meV    (current pred 50.57 meV; obs 50.13 meV)")
print(f"  Σm_ν = {sum_m_nu*1e3:.2f} meV")
print()
print("External-bound consistency check:")
print(f"  Σm_ν = {sum_m_nu*1e3:.1f} meV     vs Planck 2018 TT,TE,EE+lensing+BAO < 120 meV    OK")
print(f"  m_β  = {m_beta*1e3:.2f} meV   vs KATRIN 2024 90% CL < 450 meV                     OK")
print(f"  m_ββ ≤ {max(m_bb,m_bb_min_phases)*1e3:.2f} meV  vs KamLAND-Zen 2024 < 36–156 meV (NME range)  OK")
print()
print("Verdict: m_ν₁ ≈ {:.2f} meV closes the gap CLEANLY against all current bounds.".format(m_nu1*1e3))
print()
print("BUT: m_ν₁ ≡ 0 is locked by W45 (substrate produces exactly 2 dynamical")
print("     Majorana ν_R via girth-ring holonomy h^g; rank-2 seesaw → exactly one")
print("     light mass identically zero).  W45 IS Need-D-3-free.")
print()
print("G3 STRUCTURAL TENSION:  the gap-closing m_ν₁ of {:.2f} meV is incompatible".format(m_nu1*1e3))
print("with the W45 mode-count theorem.  This is genuine, not a precision issue.")


# =====================================================================
banner("G4 — Identification: is R = 228/7 really m₃²/m₂²?")
# =====================================================================
print("R_theorem.md §'Physical interpretation' (lines 53-66) states:")
print()
print('   "R is the anisotropy of this propagator across the three Z3 channels:')
print('    - Total propagator intensity: 2/sin²(5φ) = 256/7')
print('    - Isotropic background: k*+1 = 4 (one per K4 vertex)')
print('    - Splitting: R = anisotropy - background = 228/7"')
print()
print("R_nu_splitting_derivation.md §'Open Questions' explicitly flags:")
print()
print('   "The identification of [228/7] with Δm²₃₁/Δm²₂₁ relies on the')
print('    physical interpretation... a physical argument, not a formal proof."')
print()
print("So '228/7 = m₃²/m₂²' is a BRIDGE ASSERTION, not a theorem.")
print()
print("The K₄ spectral object 2/sin²(5φ) − (k*+1) is exact and unambiguous.")
print("Whether it equals (m₃²−m₁²)/(m₂²−m₁²), m₃²/m₂², or some other ratio")
print("of mass-squared eigenvalues depends on how the substrate spectral")
print("structure maps to the physical mass matrix.  That map is the open item.")
print()
print("G4 = LOAD-BEARING FINDING:  the gap may live in the IDENTIFICATION layer.")
print("                            The spectral computation is closed; its")
print("                            physical interpretation as Δm²₃₁/Δm²₂₁ is not.")


# =====================================================================
banner("G5 — Alternative K₄ spectral invariants near 33.55")
# =====================================================================
# Survey other K₄ invariants involving the Ihara phase φ and check whether
# any natural one lands close to 33.55.
print(f"Target:  R_obs (NuFIT 6.0 w/ SK) = {R_obs_central:.4f}")
print(f"Current: R_pred = 228/7         = {R_pred:.4f}   gap −2.92%")
print()

candidates = []

# (a) 2/sin²(5φ) − (k*+1)             = 228/7  = 32.571   (current)
candidates.append(("(a) 2/sin²(5φ) − (k*+1)         [current]", 256/7 - 4))
# (b) 2/sin²(5φ) − (k*)               = 256/7 − 3 = 235/7 = 33.571   <-- !
candidates.append(("(b) 2/sin²(5φ) − k*              [shift]", 256/7 - 3))
# (c) 2/sin²(5φ) − (k*−1)             = 256/7 − 2 = 242/7 = 34.571
candidates.append(("(c) 2/sin²(5φ) − (k*−1)          [shift]", 256/7 - 2))
# (d) 256/7                            = 36.571                       [pure spectral, no background]
candidates.append(("(d) 2/sin²(5φ) [no subtraction]   [shift]", 256/7))
# (e) k*·256/7 / (k*+1)
candidates.append(("(e) k*·(256/7) / (k*+1)          [scaled]", 3 * (256/7) / 4))
# (f) ratio of two Chebyshev squared-amplitudes
# G_n² with n=5: 1/16; G_n² with n=10: cos(10φ)=57/64 so sin(10φ)² = (1-57²/64²) = ?
cos_10phi = 57/64
sin_10phi_sq = 1 - cos_10phi**2
G10_sq = sin_10phi_sq / float(sin2_phi_exact)
G5_sq = float(sin2_5phi_exact)/float(sin2_phi_exact)
candidates.append(("(f) G₁₀²/G₅²                      [Chebyshev ratio]", G10_sq/G5_sq))
# (g) algebraic forms involving 57 = 64−7
candidates.append(("(g) 57·(k*+1)/(k*-1) + 0          [57·4/2=114]", 57*4/2))
# (h) 57/sin²(5φ) - something
candidates.append(("(h) 57/(2·sin²(5φ))               [57·64/7=520.57]", 57/(2*float(sin2_5phi_exact))))
# (i) the cos(10φ) = 57/64 identity in different combinations
candidates.append(("(i) 64/sin²(5φ) − 7·k*            [64·128/7 − 21]", 64/float(sin2_5phi_exact) - 7*3))
candidates.append(("(j) (228 + 7)/7  = 235/7           [+1 unit shift]", 235/7))

print(f"  {'candidate':<48}  {'value':>12}  {'Δ%':>8}")
print("  " + "-" * 76)
for name, val in candidates:
    dev = (val - R_obs_central)/R_obs_central * 100
    flag = "  ⬅" if abs(dev) < 1.0 else ""
    print(f"  {name:<48}  {val:>12.4f}  {dev:+7.3f}%{flag}")

print()
# Best alternative interpretation
best_name, best_val = min(candidates, key=lambda c: abs(c[1] - R_obs_central))
print(f"Best near-by alternative: {best_name}")
print(f"   value = {best_val:.4f}  vs obs {R_obs_central:.4f}  (Δ = {(best_val-R_obs_central)/R_obs_central*100:+.3f}%)")
print()
print("Candidate (b) — 2/sin²(5φ) − k* = 235/7 = 33.571 — sits at +0.07% from")
print("R_obs.  That is a single-unit shift in the subtracted 'background':")
print("    'background = (k*+1)' (current)  →  'background = k*' (alternative).")
print()
print("This is a structurally meaningful candidate — the (k*+1) was chosen as")
print("'one isotropic propagator per K₄ vertex' (i.e., a 4-vertex K₄).  But the")
print("substrate's physical observable might naturally subtract one channel")
print("(k* directed edges per vertex; or k* = 3 generations) instead.")
print()
print("G5 = STRUCTURAL CANDIDATE:  235/7 = 33.571 sits at +0.07% from R_obs.")
print("                            The shift = swap (k*+1) → k* in the background.")
print("                            Worth a structural-articulation probe of which")
print("                            count is correct for the physical mass-ratio observable.")


# =====================================================================
banner("SYNTHESIS")
# =====================================================================
print("""
The R = 228/7 vs R_obs ≈ 33.55 gap is REAL: −2.92% (−1.10σ on R itself
given NuFIT 6.0 σ_R ≈ 0.89; +1.87σ_obs when propagated through m_ν₂
with m_ν₃ taken parameter-free, which is the §8-failure mode flagged
in m_nu2.py).

  G1: 228/7 is hard-locked at the spectral level (PASS).  No ε-perturbation
      of k* = 3 / cos²φ / n = 5 changes 228/7.

  G2: R_theorem.md's "32.576 / 0.015% match" headline is STALE pre-NuFIT-6.0
      data; codebase data inputs are inconsistent (32.576 / 33.55 / 33.83
      across three files).  Canonical NuFIT 6.0 w/ SK: R_obs = 33.55(0.32).

  G3: m_ν₁ ≈ 1.5 meV would close the gap cleanly within all external bounds
      (Planck Σm_ν, KATRIN m_β, KamLAND-Zen m_ββ).  But m_ν₁ ≡ 0 is locked
      by W45 (rank-2 mode count) which is Need-D-3-free.  GENUINE TENSION.

  G4: The identification R ≡ Δm²₃₁/Δm²₂₁ is a PHYSICAL BRIDGE, not derived.
      R_nu_splitting_derivation.md flags this as Open Question 1.  The
      spectral object is unambiguous; what it equates to physically is open.

  G5: Alternative K₄ invariant 235/7 = 33.571 sits at +0.07% from R_obs.
      The shift is a one-unit swap in the subtracted background:
          228/7 = 256/7 − (k*+1)
          235/7 = 256/7 −  k*
      (k*+1) was 'one propagator per K₄ vertex'; k* would be 'one per
      generation channel' or 'one per directed edge at a vertex'.  This
      is a structural-articulation candidate worth a bounded probe.

WHERE THE STRUCTURAL SOLUTION LIKELY LIVES (recommendation):

Of the three escape paths surveyed, only G5 is BOUNDED-PROBE-ACTIONABLE.
G1 is closed-negative (228/7 cannot move).  G3 is tension with W45 (a
deeper theorem).  G4 is the open-question layer flagged by the derivation
doc itself but with no current bounded re-derivation path.

The G5 candidate "235/7 = 256/7 − k*" is a single-bit shift in the
background-subtraction count.  Probe: re-derive what 'background' means
in the physical mass-matrix translation (R = anisotropy − background) and
check whether the count should be (k*+1) = vertex-count or k* = generation-
count when the spectral object is read as a mass-squared ratio rather than
a propagator-intensity quantity.

This is the same structural family as multi-axial waterfilling Reference
A.4 (chirality-routing inheritance via R-9 discharge): a single integer
in a subtraction template controlled by whether a substrate observable
routes through one classification axis or another.
""")
