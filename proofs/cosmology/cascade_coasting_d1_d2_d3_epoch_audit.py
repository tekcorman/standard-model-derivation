#!/usr/bin/env python3
"""
proofs/cosmology/cascade_coasting_d1_d2_d3_epoch_audit.py

Pre-Path-D Step 1: cascade-theorem D1/D2/D3 epoch-validity audit.

Setup
-----
Path A Session 2 falsified the cascade theorem's strict claim "H · t_P · N
= 1 for any epoch N" at the CMB acoustic-peak level by ~10⁵σ. Per the
pre-Path-D scoping doc (`cascade_coasting_high_z_falsification_scoping_
2026-05-05.md`), three scenarios:

  Scenario 1: framework correct, missing pre-recombination machinery (Path D).
  Scenario 2: cascade theorem's "any epoch" claim needs revision.
  Scenario 3: framework genuinely falsified.

This audit examines each cascade-theorem step (D1, D2, D3) for hidden
epoch-dependent assumptions. The verdict determines which scenario applies.

Cascade theorem statement (per `predictions/N_hub.py` lines 64-80)
------------------------------------------------------------------

  D1 [A1, Type 1]: Each of the k*N directed edges in the toggle graph is
    toggled once per Planck time. Each toggle modifies 1/(k*N) of the
    universe's causal structure.

  D2 [A2 + algebra, Type 1+2]: MDL surprise threshold θ* = log₂(k*).
    Acceptance probability per toggle: 2^{-θ*} = 1/k*.
    "Observable" options (new causal states per t_P) = k*N × (1/k*N) = 1
    exactly. Coefficient is identically k*N × [1/(k*N)].

  D3 [algebra, Type 2]: Cascade ratio ε = 1/(k*N). New states per t_P
    = k*N × ε = 1. H = (1 new state per t_P) / (N states total) = 1/(N t_P).

  Result: H · t_P · N = 1 exactly, for any epoch N.

Audit method
------------
For each D_i, identify the hidden assumptions. Test which assumptions are
genuinely epoch-independent vs which break at very early cosmic epochs.
"""

import sys
import os
import math
from fractions import Fraction

import numpy as np

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# =============================================================================
# §0. Setup
# =============================================================================
print("=" * 78)
print("CASCADE-THEOREM D1/D2/D3 EPOCH-VALIDITY AUDIT")
print("Pre-Path-D Step 1 (per cascade_coasting_high_z_falsification_scoping_2026-05-05.md)")
print("=" * 78)
print()


# =============================================================================
# §1. D1 — "Each of the k*N directed edges is toggled once per t_P"
# =============================================================================
print("§1. D1 — toggle graph has k*·N directed edges; each toggled once per t_P")
print("-" * 78)
print("""
  STATED CLAIM. Per N_hub.py line 67-69:
    "Each of the k*N directed edges in the toggle graph is toggled once
     per Planck time (time mapping: one t_P = k*N toggles)."

  HIDDEN ASSUMPTIONS:
    (A1.a) Substrate IS the srs net with k* = 3.
    (A1.b) Substrate has N stable vertices at the epoch in question.
    (A1.c) The "toggle clock" runs at rate 1 toggle per t_P per directed edge.
    (A1.d) t_P is the substrate's Landauer quantum at substrate temperature
           T_substrate = T_Planck; this defines the clock.

  EPOCH-DEPENDENCE:
    (A1.a) FAILS at T > T_srs. Per `early_universe_k_rundown.py`:
       "In the early universe (T >> T_srs): ALL nets below threshold —
        no MDL structure, pure random toggles (Planck epoch)."
       The substrate is NOT in srs at T > T_srs; it has no stable structure.

    (A1.b) FAILS at T > T_srs (corollary of A1.a — no stable structure
       means no stable vertex count).

    (A1.c) is internal to the framework's Landauer-quantum identification
       and doesn't depend on cosmic temperature. Holds at all epochs.

    (A1.d) is the framework's natural-units convention. Holds by definition
       at all epochs.

  RUNDOWN TEMPERATURE ESTIMATE (from `early_universe_k_rundown.py`):
    C(srs) = n_g × (g-2) × log₂(3/2) = 15 × 8 × log₂(3/2) ≈ 70.18 bits/vertex
    Thermal noise floor: ε_T = k_B T × ln 2 (Landauer + ln 2 base conversion)
    T_srs is set by ε_T = C(srs), giving k_B T_srs ≈ 70.18/ln(2) ≈ 101.3
    in framework natural units (Planck units).
""")

C_SRS = 70.18  # bits/vertex (n_g × (g-2) × log₂(3/2) for srs with k=3, g=10)
T_SRS_PLANCK_UNITS = C_SRS / math.log(2)
print(f"  T_srs ≈ {T_SRS_PLANCK_UNITS:.1f} × T_Planck")
print(f"  i.e., srs is MDL-active (above thermal noise) for T < {T_SRS_PLANCK_UNITS:.0f} × T_Planck.")

# Compare to the maximum cosmic temperature in observable history:
# universe never exceeded T_Planck (Planck epoch is the upper boundary).
# CMB peak temperature: T_*≈ 2.725 K × (1+z*) ≈ 2970 K at z* = 1090.
# In Planck units: T_Planck ≈ 1.42e32 K, so T_*/T_Planck ≈ 2.1e-29.
T_Planck_K = 1.417e32
T_today_K = 2.725
T_recomb_K = T_today_K * 1090
T_BBN_K = T_today_K * 4e8  # approximate BBN temperature ~ MeV
T_recomb_planck = T_recomb_K / T_Planck_K
T_BBN_planck = T_BBN_K / T_Planck_K

print(f"\n  Cosmic temperatures in Planck units:")
print(f"    T_today (z=0)        = {T_today_K:.3f} K = {T_today_K/T_Planck_K:.2e} × T_Planck")
print(f"    T_* (z = 1090, CMB)  = {T_recomb_K:.0f} K = {T_recomb_planck:.2e} × T_Planck")
print(f"    T_BBN (z ≈ 4e8)      ~{T_BBN_K:.1e} K = {T_BBN_planck:.2e} × T_Planck")
print(f"    T_Planck epoch       = {T_Planck_K:.2e} K = 1 × T_Planck")
print()
print(f"  T_srs ≈ {T_SRS_PLANCK_UNITS:.0f} × T_Planck >> T_Planck epoch >> T_BBN >> T_* >> T_today.")
print(f"  ⇒ srs is MDL-active throughout ALL OBSERVABLE COSMIC HISTORY.")
print(f"  ⇒ A1.a and A1.b hold at z = z_* = 1090 (and at all observable z).")
print()
print("  D1 epoch-validity: HOLDS throughout observable cosmology.")
print("  No epoch-restriction surfaces from D1 alone.")
print()


# =============================================================================
# §2. D2 — "MDL surprise threshold = log₂ k*; acceptance prob = 1/k*"
# =============================================================================
print("§2. D2 — MDL acceptance probability per toggle = 1/k*")
print("-" * 78)
print("""
  STATED CLAIM. Per N_hub.py line 70-74:
    "MDL surprise threshold θ* = log₂(k*) [from S_fresh.py and S_disconfirm.py].
     Acceptance probability per toggle: 2^{-θ*} = 1/k*.
     Observable options (new causal states per t_P) = k*N × (1/k*N) = 1 exactly."

  HIDDEN ASSUMPTIONS:
    (A2.a) k* = 3 (substrate is srs).
    (A2.b) MDL threshold θ* = log₂ k* applies at all toggle events.
    (A2.c) S_fresh + S_disconfirm framing (Beta(1,1) → Beta(2,1) update)
           applies to every toggle event.

  EPOCH-DEPENDENCE:
    (A2.a) inherits from D1 — fails at T > T_srs. But T_srs > T_Planck epoch,
       so A2.a holds throughout observable cosmology.

    (A2.b) THIS IS THE KEY HIDDEN ASSUMPTION. The MDL threshold is a
       structural property of the observer's compression apparatus, NOT
       a cosmological parameter. The threshold sets the acceptance rate
       of NEW causal states. But what counts as "new" depends on the
       observer's prior state count.

       For the cascade theorem, the assumption is: at every t_P, the
       observer compresses the substrate's accumulated state up to that
       point, and 1/(k*N) of toggles cross the threshold to enter the
       observable register.

       At very early times (small N): the threshold 1/(k*N) is very high
       (small N → 1/(k*N) is order 1). This means almost EVERY toggle
       event produces a new observable state. The cascade theorem says
       the new-state rate is k*N · 1/(k*N) = 1 per t_P, exactly because
       the high threshold (small N) is multiplicatively cancelled by the
       high toggle count (k*N is large at later N, but at early N it's
       small).

       Actually the algebra k*N · 1/(k*N) = 1 is identity for ANY N >= 1.
       So D2 is internally consistent at all N. No epoch restriction here.

    (A2.c) S_fresh + S_disconfirm applies to Beta-conjugate updating, which
       is structurally true at all epochs. No epoch restriction.

  D2 epoch-validity: HOLDS at all N >= 1 by algebraic identity.
""")
print()


# =============================================================================
# §3. D3 — "Cascade ratio ε = 1/(k*N), giving 1 new state per t_P"
# =============================================================================
print("§3. D3 — cascade gives exactly 1 new state per t_P")
print("-" * 78)
print("""
  STATED CLAIM. Per N_hub.py line 76-78:
    "Cascade ratio ε = 1/(k*N). New states per t_P = k*N × ε = 1.
     H = (1 new state per t_P) / (N states total) = 1/(N t_P)."

  HIDDEN ASSUMPTIONS:
    (A3.a) "New states per t_P" is identified with the universe's expansion
           rate — i.e., dN/dt_cosmic = 1/t_P at all epochs.
    (A3.b) The substrate clock t_P matches cosmic time t at all epochs.
    (A3.c) "1 new state" maps directly to "1 unit of cosmic expansion".

  EPOCH-DEPENDENCE — THE KEY ISSUE:

    (A3.a) is the load-bearing identification. It says: when the substrate's
       observer-side compression accepts 1 new state per substrate-clock
       tick (t_P), the cosmic universe expands such that ȧ/a = 1/t_cosmic.

       This identification conflates TWO time scales:
         (i)  substrate clock: 1 tick = t_P (Planck-scale duration).
         (ii) cosmic time:    measured by FLRW expansion ȧ/a.

       At observer's epoch, the framework asserts these are SYNCHRONIZED.
       This synchronization is plausible at low cosmic energies (T <<
       T_Planck) where the substrate's intrinsic dynamics dominate over
       cosmic-expansion effects.

       AT HIGH COSMIC ENERGIES (T → T_Planck epoch, i.e., very early cosmic
       times): the substrate's clock and the cosmic clock may NOT be
       synchronized. The substrate's intrinsic time scale (t_P) is fixed,
       but cosmic time is whatever the metric expansion produces. In an
       inflationary or radiation-dominated regime, cosmic time runs
       differently from substrate-clock time.

       This is the framework's HIDDEN EPOCH-DEPENDENT ASSUMPTION: that the
       substrate's state-counting rate (1 per t_P) translates directly
       into the cosmic Hubble rate H = 1/t at all epochs.

    (A3.b) is a special case of A3.a — same issue.

    (A3.c) is the structural identification of "new substrate state" with
       "cosmic expansion event". This holds when each new state corresponds
       to one Planck-volume of cosmic space; at high T, this might break
       down if the substrate can produce multiple states per Planck-volume
       (multiway branching during inflation, e.g.).

  EPOCH-RESTRICTION: A3.a is the load-bearing assumption. It is a
  STRUCTURAL identification that the framework treats as tautological,
  but it implicitly assumes:

     "the observer-side compression rate (substrate state acceptances)
      equals the cosmic expansion rate."

  This holds at observer's epoch (definitionally — that's how H_obs is
  measured). It is NOT obviously true at z >> 0 unless the substrate's
  state-counting machinery operates the SAME WAY at all cosmic
  temperatures.

  D3 epoch-validity: assumes substrate clock ↔ cosmic clock synchronization
  at all epochs. This is a STRONG assumption that is not separately derived
  in the framework.
""")
print()


# =============================================================================
# §4. The audit verdict
# =============================================================================
print("§4. Audit verdict — which scenario applies?")
print("=" * 78)
print(f"""
  Per-step epoch-validity:
    D1: HOLDS throughout observable cosmology (substrate is srs at all
        T < T_srs ≈ {T_SRS_PLANCK_UNITS:.0f} × T_Planck, which is everywhere observable).
    D2: HOLDS by algebraic identity (k*N · 1/(k*N) = 1 at any N ≥ 1).
    D3: ASSUMES substrate-clock ↔ cosmic-clock synchronization at all
        epochs. This is the load-bearing implicit assumption that is NOT
        separately derived.

  The cascade theorem's strict claim "H · t_P · N = 1 exactly, for any
  epoch N" is therefore CONDITIONAL on the synchronization assumption in
  D3. At low cosmic temperatures (T << T_Planck), this is plausible. At
  very high cosmic temperatures (T → T_Planck epoch), the cosmic-time
  metric may run differently from the substrate clock, breaking
  synchronization.

  At z = z_* = 1090, T_cosmic ≈ {T_recomb_planck:.2e} × T_Planck — far below
  Planck temperature. The synchronization assumption is plausible at
  this scale. Yet Session 2 showed coasting θ_* misses Planck by ~10⁵σ.

  Conclusion: D1 and D2 are epoch-robust. D3's synchronization assumption
  is implicit and undefended, but it's not OBVIOUSLY broken at z = z_*
  where T_cosmic << T_Planck.

  This SHARPENS the falsification: the cascade theorem's strict claim
  isn't just "a bit overstated" — it appears to be applicable at z = z_*
  per the audit, yet observation falsifies it. So the falsification is
  not resolved by epoch-restriction alone (Scenario 2). The framework's
  cascade theorem as currently derived genuinely PREDICTS coasting at
  z = z_* and is FALSIFIED there.

  WHICH SCENARIO?

    Scenario 1 (Path D — non-coasting early-universe regulator):
      Requires identifying a NEW mechanism that breaks coasting at
      z > z_eq while preserving it at z < z_eq. The audit doesn't find
      such a mechanism in the existing cascade-theorem derivation. So
      Path D would have to introduce something NEW (e.g., multiway
      branching dynamics that produce more states per t_P at high z).
      Multi-session research; not a quick fix.

    Scenario 2 (cascade theorem's "any epoch" claim needs revision):
      D3's synchronization assumption is implicit but not OBVIOUSLY
      broken at observable z. So Scenario 2 in its simple form (just
      add an epoch-restriction) doesn't resolve the falsification.

    Scenario 3 (framework genuinely falsified):
      The audit suggests the framework's derivation is internally
      consistent and the falsification is REAL. This is the
      uncomfortable but honest conclusion if Scenarios 1 and 2 don't
      resolve it.

  HONEST VERDICT FROM AUDIT:

  - D1 and D2 are epoch-robust under the audit.
  - D3 has an implicit synchronization assumption, but it's plausible at
    z = z_* where T << T_Planck.
  - The cascade theorem's coasting prediction at z = z_* is therefore
    NOT rescued by an epoch-restriction. Either:
      (a) The framework needs a NEW mechanism (Path D / Scenario 1)
          beyond the current cascade-theorem derivation.
      (b) The framework's prediction at z = z_* is genuinely wrong
          (Scenario 3).

  Cosmology cluster exposure is therefore SERIOUS. The audit doesn't
  identify a quick out. Path D is the only constructive option, but its
  motivation is now structural FALSIFICATION RECOVERY, not "factor-of-2
  closure".

  Item-5 status (per scoping doc §7): the cosmology arc's "fifth item"
  (pre-recombination physics) is now confirmed as a NECESSARY structural
  research direction, not an optional follow-on. The framework owes a
  quantitative early-universe story OR an honest concession that the
  cascade theorem at high z is wrong.
""")


# =============================================================================
# §5. What the audit clarifies for Path D
# =============================================================================
print("§5. What this means for Path D scoping")
print("-" * 78)
print("""
  Path D as motivated by factor-of-2: NOT viable (factor-of-2 is
  empirical relabeling per Path F audit).

  Path D as motivated by Session 2 falsification: still motivated, but
  scope is BROADER than originally framed. It's not "find a regulator
  that preserves factor-of-2"; it's "find a mechanism that reconciles
  the cascade theorem with Planck CMB at z = z_*".

  Concrete sub-targets for Path D:
    - Multiway-branching pre-recombination phase: at z >> z_eq, the
      substrate's state-counting may follow a polynomial different from
      N(t) ∝ t (e.g., N(t) ∝ t^p with p > 1). This would give H ∝ t^{-p},
      possibly reducing to radiation-domination H ∝ (1+z)² for some p.
    - Two-phase substrate dynamics: the substrate transitions from a
      "thermal soup" phase (T > T_srs) to "stable srs" (T < T_srs) at
      some z_transition. The transition itself produces a non-coasting
      H(z). But T_srs >> T_Planck epoch under this audit's estimate, so
      this transition isn't accessible at observable z.
    - Inflation-like regime: at very early times, exponential expansion
      could give H = const while the substrate's state count is in a
      different regime. The framework's existing inflation analog
      (`predictions/N_hub.py` mentions "de Sitter exponential growth"
      via Ramanujan expander) doesn't quantitatively connect to CMB θ_*.

  ESTIMATED SCOPE OF PATH D (revised after this audit):
    - Identify the right pre-recombination dynamics: 4-8 sessions.
    - Connect to CMB observables (θ_*, BBN abundances): 4-8 sessions.
    - Cross-check with current cosmology cluster (P17, P19, P20, P22, P24): 2-4 sessions.
    Total: 10-20 sessions, multi-sprint research direction.

  PRE-PATH-D RECOMMENDATION: Before opening Path D, audit which downstream
  rows are genuinely insulated from this issue (P19, P20 use cascade only
  at z = 0; P22 is frame-invariant; P24 uses coasting at z = 0 only). The
  ROWS most exposed are P25 (n_s, inflationary epoch), P26 (A_s,
  inflationary epoch), and any others derived through high-z cosmology.

  Item-5 of cosmology roadmap: open as "pre-recombination physics
  reconciliation" — NECESSARY structural research direction.
""")


# =============================================================================
# §6. Recommendations
# =============================================================================
print("§6. Recommendations")
print("=" * 78)
print("""
  A. Update parameter ledger Rows P17, P19, P20, P22, P24 with a NEW
     conditional gate: "valid at observer's epoch (z = 0); cascade theorem
     at z >> z_eq is structurally falsified per Session 2; pre-recombination
     reconciliation is open Item 5".

  B. Audit Rows P25, P26 (and any inflationary-epoch predictions) to check
     whether their derivations use cascade-theorem coasting at high z. If
     so, those rows are MORE exposed than P17-P22 to the falsification.

  C. Open Item 5 of cosmology roadmap as "pre-recombination physics".
     Recommended scope: derive H(z) at z > z_eq from substrate dynamics
     (multiway branching, k-cooling, or inflation-like phase). Multi-
     session research.

  D. Path D scoping doc to be written AFTER A and B are complete — Path D's
     concrete attack vector depends on which downstream rows are exposed.

  This is a real structural finding. The cascade-theorem coasting at all
  epochs is the framework's most ambitious cosmological claim, and it
  doesn't survive the simplest CMB test. Honest disclosure of this in the
  ledger and roadmap is the necessary first step before Path D research.
""")
print("=" * 78)
print("DONE: cascade-theorem D1/D2/D3 epoch-validity audit.")
print("Verdict: D1+D2 epoch-robust; D3 has implicit synchronization but")
print("  not obviously broken at z=z*; falsification is REAL not resolved")
print("  by epoch-restriction alone. Path D needed (Scenario 1) OR honest")
print("  concession (Scenario 3). Item 5 of cosmology roadmap REQUIRED.")
print("=" * 78)
