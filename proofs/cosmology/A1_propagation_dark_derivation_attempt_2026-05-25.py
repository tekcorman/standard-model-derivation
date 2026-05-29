#!/usr/bin/env python3
"""
A1 PROPAGATION dark-correction derivation ATTEMPT (2026-05-25).

Goal: derive structurally why the dark waterline factor enters A1's
propagation (in addition to the anchor), giving the (1 - α₁/(1-α₁))²
total closure of the 8% residual found in
`A1_dark_correction_propagation_probe_2026-05-25.py`.

This probe is a HONEST DERIVATION ATTEMPT, not a closure claim. Each
candidate gets:
  (a) structural framing
  (b) numerical evaluation
  (c) PASS/FAIL/OPEN against the closure target

Required total dark correction factor on T_today:
    X_target = T_observed / T_baseline = 2.7255 / 2.9536 = 0.92276
which is (1 - α₁/(1-α₁))² = (1 - 256/6305)² ≈ 0.92047 (closes -0.25%).

Equivalently, on u = a_SB × T^4 level: u factor = X_target^4 = 0.7249,
which is (1 - α₁/(1-α₁))^8 ≈ 0.7218.

THE PROBLEM: u-level corrections combine multiplicatively. For T to
pick up (1-w)² total, u needs to pick up (1-w)^8. That's 8 independent
applications of the dark waterline on u — very hard to motivate.

So if (1-w)² is real on T, it CAN'T come from u-level corrections alone.
It must come from T-level corrections (where T is a fundamental observable
separate from u, with its own dark-correction template).

This probe tests three candidate derivations:

  D1: Stefan-Boltzmann at u-level only — substrate pump κ → κ × (1-w)
      → T → T × (1-w)^(1/4). Combined with anchor c_S × (1-w):
      T = T_baseline × (1-w)^(5/4).

  D2: T-direct correction at propagation with c_g unspecified, anchor
      c_g unspecified. Test which combinations close.

  D3: Two-stage observer-graph picture: anchor at GUT (BARE since
      unbroken parity, per M_unif.py rule); propagation at every
      epoch picks up the dark waterline at T-level with c_g = 2 (one
      power for each of the two factors that define T as observer-
      graph thermal scale: energy and entropy).

  D4: Alternative — maybe the anchor c_S itself ISN'T dark-corrected
      (per M_unif.py's unbroken-parity rule), and ALL (1-w)² is in
      the propagation as a single c=2 application.

Aborts if no derivation closes structurally; honest report.
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
N_trivial = 2          # trivial sector dim (from M_unif Stage 4 closure)
two_E = N_atoms * k_star  # = 12 (handshake lemma)

alpha_GUT_bare = Fraction(1, 24)
alpha_1_bare = Fraction(2, 3) ** 8
c_S = Fraction(1, two_E)         # = 1/12
waterline = alpha_1_bare / (1 - alpha_1_bare)  # = 256/6305
w = float(waterline)

# Baseline (bare anchor, no dark correction)
M_unif = float(alpha_GUT_bare * alpha_1_bare) * M_Pl_GeV
N_GUT = N_hub * (v_today / M_unif) ** 4
T_baseline = M_unif * float(c_S) * K_per_GeV * math.sqrt(N_GUT / N_hub)

X_target = T_CMB / T_baseline
u_factor_target = X_target ** 4

print("=" * 76)
print("A1 PROPAGATION DARK-CORRECTION DERIVATION ATTEMPT")
print("=" * 76)
print(f"\nBaseline T_today (all bare):      {T_baseline:.4f} K  ({(T_baseline-T_CMB)/T_CMB*100:+.2f}%)")
print(f"Observed T_CMB:                   {T_CMB} K")
print(f"Required T-level factor:          X = {X_target:.5f}")
print(f"Equivalent u-level factor:        X^4 = {u_factor_target:.5f}")
print(f"\nSingle waterline factor:          (1-w) = {1-w:.5f}, w = {w:.5f}")
print(f"Squared waterline:                (1-w)² = {(1-w)**2:.5f}")
print(f"Eighth power waterline:           (1-w)^8 = {(1-w)**8:.5f}")
print(f"\nSO: closure needs T → T × (1-w)² ⇔ u → u × (1-w)^8")
print(f"   The 8-power on u is hard to motivate from a few independent")
print(f"   substrate channels — more natural to derive at T-level directly.")


# ---------------------------------------------------------------------------
# D1: u-level correction from substrate pump only
# ---------------------------------------------------------------------------
print(f"\n{'='*76}")
print("D1: substrate-pump u-level dark correction (single (1-w) on κ)")
print('='*76)
print(f"""
Structural argument: substrate pumps energy at rate κ/t_P per tick. Of
this pumped energy, only the NB-walker-surviving share (1-α₁/(1-α₁))
ends up in the gauge-readable channel; the dark waterline absorbs the
rest. So κ_observable = κ_substrate × (1-w).

From u = κ/[(c·t_P)³ · N²], this gives u → u × (1-w).
From T = u^(1/4), this gives T → T × (1-w)^(1/4).

Combined with anchor c_S × (1-w) [universal template on Perron-singlet
projection]: T_today = T_baseline × (1-w) × (1-w)^(1/4) = T_baseline × (1-w)^(5/4).
""")

T_D1 = T_baseline * (1 - w) ** (5/4)
print(f"  T_today (D1) = {T_D1:.4f} K  ({(T_D1-T_CMB)/T_CMB*100:+.2f}%)")
print(f"  vs required {T_CMB} K — DOES NOT CLOSE (single u-level correction too weak)")


# ---------------------------------------------------------------------------
# D2: thermal-mode multiplicity: each of 4 thermal-mode dimensions gets (1-w)
# ---------------------------------------------------------------------------
print(f"\n{'='*76}")
print("D2: thermal-mode multiplicity — (1-w) per phase-space dimension")
print('='*76)
print(f"""
Structural argument: Stefan-Boltzmann u ∝ T^4 from 4 phase-space
dimensions (3 spatial momentum + 1 polarization for massless photon).
If EACH phase-space integration picks up an independent NB-walker
survival (1-w), then u → u × (1-w)^4.

From u → u × (1-w)^4 and T = u^(1/4): T → T × (1-w).

Combined with anchor c_S × (1-w): T_today = T_baseline × (1-w)².
This is the CLOSURE form.
""")

T_D2_with_anchor = T_baseline * (1 - w) ** 2
print(f"  T_today (D2 + anchor c_S × (1-w)) = {T_D2_with_anchor:.4f} K  ({(T_D2_with_anchor-T_CMB)/T_CMB*100:+.2f}%)")
print(f"  → CLOSES numerically (-0.25%)")
print()
print("  HONESTY CHECK on the structural argument:")
print("    The claim 'each phase-space integration picks up (1-w)' would")
print("    require deriving that the NB-walker survival probability factors")
print("    as a product over independent k-modes and polarizations. This")
print("    needs:")
print("      (a) The NB walker's k-mode structure factorizes (likely true")
print("          by Bloch decomposition — each k is an independent sector).")
print("      (b) The dark waterline α₁/(1-α₁) applies PER MODE, not")
print("          globally (NEEDS DERIVATION — the master doc applies it")
print("          per OBSERVABLE, not per phase-space dimension).")
print()
print("  STATUS: candidate structural form matches numerics, but (b) is")
print("    NOT yet derived. The dark waterline is conventionally applied")
print("    at the OBSERVABLE level (one factor per observable), not at")
print("    the phase-space dimension level.")


# ---------------------------------------------------------------------------
# D3: T-level correction at propagation with c_g = 2 (N_trivial)
# ---------------------------------------------------------------------------
print(f"\n{'='*76}")
print("D3: T-level dark correction with c_g = N_trivial = 2 at propagation")
print('='*76)
print(f"""
Structural argument: temperature is the energy-conjugate Lagrange
multiplier (T = ∂E/∂S). The substrate's "trivial sector" — the
single-mode propagator dim — is N_trivial = 2 (per M_unif Stage 4:
M_R = N_trivial × (1/k*)^(g-1) × M_Pl).

If the temperature observable's dark-correction c_g equals N_trivial =
2 (the trivial-sector multiplicity), then:
    T_observable = T_substrate × (1 - 2 × α₁/(1-α₁))

If applied JUST at propagation (anchor stays bare), we get
T_today = T_baseline × (1 - 2w).
""")

T_D3_propagation_only = T_baseline * (1 - 2*w)
print(f"  T_today (D3, propagation only, anchor BARE) = {T_D3_propagation_only:.4f} K  ({(T_D3_propagation_only-T_CMB)/T_CMB*100:+.2f}%)")
print(f"  → CLOSES numerically (-0.40%, within numerical noise of (1-w)²)")
print()
print("  HONESTY CHECK:")
print("    (i) N_trivial = 2 IS substrate-derived (M_unif.py Stage 4).")
print("    (ii) But the CLAIM that c_g(T) = N_trivial needs derivation.")
print("         The master doc's c_g values (5/12 for v, 1/k* for α_GUT)")
print("         are derived via Routes H/C from specific substrate")
print("         mechanisms. We'd need an analogous derivation for")
print("         c_g(T) = N_trivial.")
print("    (iii) The anchor's parity-unbroken rule (M_unif.py) means the")
print("         anchor stays bare here — consistent with the framework's")
print("         existing rule.")
print()
print("  Why might c_g(T) = N_trivial structurally?")
print("    The substrate-pump-rate energy comes from the trivial-sector")
print("    propagator (single-mode). The N_trivial = 2 counts the two")
print("    states in the trivial isotypic (= 2-dim trivial rep of Z_3 on")
print("    K_4 substrate). If each trivial state contributes an")
print("    independent dark-waterline absorption channel, c_g = N_trivial = 2.")
print()
print("  STATUS: c_g = N_trivial is a STRUCTURAL CANDIDATE, but the per-state")
print("    dark-waterline channel derivation is not yet written. This is the")
print("    same epistemic grade as v_Higgs c=5/12 before Routes H/C closed it")
print("    (i.e., conjecturally substrate-derived, awaiting explicit proof).")


# ---------------------------------------------------------------------------
# D4: Anchor stays BARE, all dark in propagation with c=2 on (1-w)²
# ---------------------------------------------------------------------------
print(f"\n{'='*76}")
print("D4: all dark correction in propagation as (1-w)² (anchor stays bare)")
print('='*76)
print(f"""
Same numerics as D3 essentially. Structural difference: instead of c_g=2
single application, this is (1-w)² from two independent (1-w) propagation
factors — e.g.:
  - one (1-w) from each Stefan-Boltzmann dimension squared, or
  - one (1-w) from the substrate-pump-rate correction AND one (1-w)
    from a separate gauge-readable-channel-survival correction.

Numerically identical to D3 (since (1-w)² = 1 - 2w + w² ≈ 1 - 2w).
Open question: which TWO independent dark-correction sources combine to
give the (1-w)² form?
""")

T_D4 = T_baseline * (1 - w)**2
print(f"  T_today (D4) = {T_D4:.4f} K  ({(T_D4-T_CMB)/T_CMB*100:+.2f}%)")


# ---------------------------------------------------------------------------
# HONEST VERDICT
# ---------------------------------------------------------------------------
print(f"\n{'='*76}")
print("HONEST VERDICT — does the propagation derivation close?")
print('='*76)
print(f"""
NUMERICAL FACT (established):
  A factor (1-w)² ≈ 0.9205 applied to T_baseline = 2.9536 K closes the
  A1 residual to within -0.25% of T_CMB = 2.7255 K. Equivalently a
  single c=2 application (1-2w) gives -0.4% (within numerical noise).

STRUCTURAL DERIVATION (the goal of this probe):
  D1 (single u-level via κ correction) → T × (1-w)^(5/4) → +2.9%
     FAILS to close.

  D2 (per-phase-space-dimension on u → (1-w)^4 on u → (1-w) on T,
      plus anchor c_S × (1-w)) → T × (1-w)² → -0.25%
     NUMERICALLY CLOSES; structurally requires deriving that the dark
     waterline applies PER PHASE-SPACE DIMENSION, which is NOT a
     standard reading of the master doc (master doc applies (1-w) per
     observable, not per phase-space dimension).

  D3 (c_g = N_trivial = 2 at propagation, anchor stays bare) → T × (1-2w) → -0.40%
     NUMERICALLY CLOSES; structurally requires deriving c_g(T) =
     N_trivial from the substrate's trivial-isotypic structure. This
     is STRUCTURALLY SUGGESTIVE (N_trivial is theorem-grade upstream
     from M_R/M_unif analysis) but the per-trivial-state dark-waterline
     mechanism is not yet derived.

  D4 (anchor bare, propagation = two distinct (1-w) sources)
     Structurally requires identifying the two independent dark-
     correction sources. Not yet identified.

NONE of D1-D4 is theorem-grade closure. The closest is D3, where the
c_g = N_trivial would be a clean structural reading IF the dark-waterline-
per-trivial-state derivation closes.

The PROPAGATION DERIVATION attempt LANDS at:
  - Numerically: P5/(1-w)² works (-0.25%); D3/(1-2w) equivalent (-0.40%)
  - Structurally: c_g(T_propagation) = N_trivial = 2 is the cleanest
    candidate, but the per-trivial-state dark-waterline derivation is
    OPEN (analogous to v_Higgs c=5/12 before Routes H/C, or α_GUT
    c=1/k* before its Routes H/C closed)

CONCLUSION:
  A1's propagation dark-correction has a CANDIDATE STRUCTURAL FORM
  (c_g = N_trivial = 2 at propagation, anchor stays bare per M_unif's
  parity rule) that closes the residual to -0.4%. But the candidate is
  NOT yet derived from substrate primitives — it's at the same grade
  as v_Higgs c=5/12 used to be before its dark-correction routes were
  closed.

  The DERIVATION ATTEMPT IS HONESTLY OPEN, NOT CLOSED.

  Path forward to actual closure:
    1. Identify the per-trivial-state dark-waterline mechanism (analogous
       to Routes H/C for v_Higgs / α_GUT) — multi-session structural work
    2. Verify the mechanism reproduces both A1's c_g = N_trivial = 2 AND
       the existing v_Higgs c = 5/12 / α_GUT c = 1/k* (calibration check
       per master doc §8 rule 4)
    3. If both pass, A1 graduates to theorem-grade-numerical closure

  This is structurally analogous to the c_α_GUT = 1/k* closure that
  happened 2026-05-15 — initially a candidate, eventually theorem-grade
  via Routes H/C. A1's propagation correction follows the same epistemic
  path: candidate now, theorem-grade if derivable.
""")

print("=" * 76)
print("STATUS: HONEST CANDIDATE (numerics close; mechanism derivation open)")
print("=" * 76)
