#!/usr/bin/env python3
"""
proofs/cosmology/p25_p26_high_z_exposure_audit.py

Pre-Path-D Step 1, Recommendation B from
`cascade_coasting_d1_d2_d3_epoch_audit.py` §6: audit P25 (n_s) and P26
(A_s, treated as cosmology-arc Item 3 with formula upgrade per
As_feshbach_exponent_upgrade.py) for high-z exposure to the Session-2
CMB θ_* falsification.

Question: do their derivations actually use cascade-theorem coasting at
z >> 0, or are they insulated like P19/P20?

Method
------
For each row, identify:
  (1) The structural input primitives. Are they framework-internal
      (k*, g, α_GUT, M_GUT, M_P, etc.) and z-independent?
  (2) The mapping to the observed quantity (n_s_observed at CMB scales,
      A_s_observed at CMB scales). Does the mapping use cascade H(z) at
      some specific z?
  (3) Any IMPLICIT cosmic-evolution assumption in the derivation chain.
      Most importantly: do they use slow-roll inflation identification,
      which requires de Sitter-like H ≈ const at inflation epoch — which
      conflicts with the framework's strict cascade-coasting H ∝ (1+z)?
"""

import sys
import os
import math


print("=" * 78)
print("P25 (n_s) and P26/A_s high-z EXPOSURE AUDIT")
print("Pre-Path-D Recommendation B")
print("=" * 78)
print()


# =============================================================================
# §1. P25 — n_s scalar spectral index ≈ 0.968
# =============================================================================
print("§1. Row P25 — n_s ≈ 0.968")
print("-" * 78)
print("""
  CLAIM (per ledger Row P25):
    n_s ≈ 0.968 from "branching statistics on the toggle graph" plus
    standard slow-roll identification.

  STATUS: MATHEMATICALLY-COMPLETE / ADVANCED. Theorem-grade closure
    pending the formal branching-statistics derivation per
    `theorem_n_s_scoping.md`.

  CURRENT-DERIVATION INPUTS:
    - branching multiplicity statistics on the substrate toggle graph
    - slow-roll inflation identification (standard cosmology mapping)

  z-DEPENDENCE AUDIT:
    (1) Branching statistics: structural property of the substrate.
        z-INDEPENDENT.
    (2) Slow-roll identification: maps framework's branching statistics
        to a slow-roll inflation parameterization at horizon crossing
        (k = aH at some inflationary epoch z_inflation ~ 10²⁶ or higher).

        The slow-roll formalism ASSUMES the inflationary epoch has
        H ≈ const (de Sitter / quasi-de Sitter). It does NOT match
        cascade-theorem coasting H ∝ (1+z) at high z.

  EXPOSURE LEVEL:
    HIGH. The slow-roll identification implicitly assumes de Sitter-like
    H(z) at inflation epoch, conflicting with the framework's strict
    cascade-coasting prediction. The derivation is INTERNALLY
    INCONSISTENT under cascade-theorem coasting at all epochs.

    This was already noted in `As_feshbach_exponent_upgrade.py` Item 4
    discussion: "with the white-noise (uncorrelated reconnection)
    identification, n_s = 1 is the framework's natural prediction."
    The 0.965 deviation requires structural input the framework hasn't
    derived — and the deviation magnitude depends on the inflationary
    H(z) profile, which under cascade-coasting differs from slow-roll's
    de Sitter assumption.

  VERDICT: P25 is EXPOSED to the CMB θ_* falsification. The slow-roll
    identification used in its derivation is incompatible with cascade-
    theorem coasting at inflation epoch. The empirical match (+0.75σ)
    is real but is via a parameterization the framework can't internally
    justify.
""")
print()


# =============================================================================
# §2. P26 / A_s — scalar amplitude
# =============================================================================
print("§2. Row P26 / cosmology-arc Item 3 — A_s ≈ 2.07e-9")
print("-" * 78)
print("""
  Note: ledger Row P26 is "r (tensor-to-scalar) < 0.01" not A_s
  directly. The A_s prediction is in the cosmology-arc Item 3 work
  (As_promotion.py + As_feshbach_exponent_upgrade.py 2026-05-05). I'll
  audit both since they share the high-z exposure question.

  A_s CLAIM (per `As_promotion.py` and Item 3 closure):
    A_s = α_GUT × (2/3)^g × (M_GUT/M_P)²
        = 1.94×10⁻⁹ pre-(16/15) correction
        = 2.07×10⁻⁹ post-(16/15) D2-extended correction

  STATUS (post-Item 3 Session 2): THEOREM-GRADE-CONDITIONAL on three
    structural identifications: (a) n_fixed = 0 self-energy via Feshbach
    Exponent Principle for (2/3)^g; (b) Bernoulli variance argument for
    α_GUT; (c) standard gravitational coupling for (M_GUT/M_P)².

  CURRENT-DERIVATION INPUTS:
    - α_GUT = reconnection DL ≈ 1/24.1 (structural; MDL on substrate)
    - (2/3)^g where g = 10 (NB walk survival via Feshbach Exponent
      Principle, n_fixed = 0)
    - (M_GUT/M_P)² gravitational suppression (mass scale ratio,
      dimensional)
    - (16/15) cascade-D2-extended observer-rate correction (Item 1)

  z-DEPENDENCE AUDIT:
    (1) α_GUT: structural MDL property; z-INDEPENDENT.
    (2) (2/3)^g: NB walk survival on srs graph; z-INDEPENDENT.
    (3) (M_GUT/M_P)²: ratio of mass scales (M_GUT from MSSM unification,
        M_P from Planck mass). z-INDEPENDENT.
    (4) (16/15) D2-extended: applied at observer's epoch (z ≈ 0); does
        NOT propagate to z = z_inflation. z-INDEPENDENT.

    BUT: identifying A_s_framework with the OBSERVED CMB scalar amplitude
    requires:
      - A_s is a power spectrum normalization at a pivot scale k_* (in
        comoving wavenumber units).
      - The pivot scale k_* corresponds to a specific physical length at
        recombination, which corresponds to a specific cosmic time at
        horizon crossing during inflation.
      - Mapping framework's (M_GUT/M_P)² gravitational coupling to the
        observed A_s requires assuming the inflationary epoch had a
        de-Sitter-like H ≈ M_GUT/M_P (or similar) at horizon crossing.

    Under cascade-theorem coasting H ∝ (1+z) at all epochs, H at
    z_inflation ~ 10²⁶ would be H_0 × (1+z_inflation) ~ 10²⁶ H_0 ~
    10⁷ km/s/Mpc — far above any de Sitter scale needed. The framework's
    formula A_s = α_GUT × (2/3)^g × (M_GUT/M_P)² doesn't EVALUATE H(z)
    at z_inflation, but its identification as "the inflationary scalar
    amplitude" relies on standard slow-roll machinery.

  EXPOSURE LEVEL:
    HIGH. Same issue as P25: the formula's connection to OBSERVED A_s
    relies on slow-roll identification, which requires de Sitter at
    inflation. Cascade-coasting at all z is incompatible with this
    identification's underlying assumption.

  VERDICT: A_s prediction is EXPOSED to the CMB θ_* falsification. The
    slow-roll identification of A_s with the inflationary scalar
    amplitude is incompatible with cascade-theorem coasting at inflation
    epoch. The empirical match (+1σ post-correction) is real but uses
    a parameterization the framework can't internally justify.
""")
print()


# =============================================================================
# §3. r (tensor-to-scalar)
# =============================================================================
print("§3. Row P26 — r (tensor-to-scalar) < 0.01")
print("-" * 78)
print("""
  CLAIM: r < 0.01, framework prediction within current observational
    bound (r_obs < 0.036, BICEP/Keck 2023).

  STATUS: CONSISTENT bound, not a tight prediction. Same scoping-doc
    gap as Row P25.

  z-DEPENDENCE AUDIT:
    Framework prediction uses slow-roll consistency relation r = 16ε
    with framework's small ε from branching statistics. Slow-roll
    consistency relation is itself a slow-roll formula, requiring
    de Sitter at inflation. Same exposure as P25/A_s.

  VERDICT: P26 is EXPOSED via slow-roll consistency relation. Same
    issue as P25/A_s.
""")
print()


# =============================================================================
# §4. Comparison with insulated rows (P17/P19/P20/P22/P24)
# =============================================================================
print("§4. Comparison with insulated rows P17/P19/P20/P22/P24")
print("-" * 78)
print("""
  Why are P17, P19, P20, P22, P24 INSULATED but P25, P26, A_s EXPOSED?

  P17 (N_hub): N_hub at z = 0 is t_0/t_P ≈ 8.5×10⁶⁰. The cascade theorem
    is evaluated only at observer's epoch. No high-z extrapolation.

  P19 (H_0): H_0 = 1/(N_hub · t_P) at z = 0. Same as P17.

  P20 (t_0): t_0 = N_hub · t_P at z = 0. Same as P17.

  P22 (Ω_DM/Ω_m): frame-invariant ratio (factor-of-2 cancels) using
    Poisson(2k*) tail at the per-vertex level. No H(z) evaluation
    needed.

  P24 (Λ_CC): Λ = 3 H_0² (Friedmann coasting condition at z = 0). Uses
    cascade theorem only at observer's epoch.

  P25, P26, A_s: DIFFER because they're predictions about OBSERVABLES
  AT INFLATION EPOCH (z >> z_*). The slow-roll identification assumes
  de Sitter H(z) at inflation, which is incompatible with cascade-
  coasting H ∝ (1+z) at all z. The framework's strict cosmological
  prediction is structurally INCONSISTENT with the parameterization
  used to derive these rows.

  This means P25, P26, and the A_s prediction (Item 3 of cosmology arc)
  are MORE EXPOSED than the late-time cosmology cluster. The cosmology
  arc declared "Item 3 closed at THEOREM-GRADE-CONDITIONAL" but the
  conditional gates didn't include "compatible with framework's
  inflation-epoch H(z)" — which under cascade-coasting is inconsistent.
""")
print()


# =============================================================================
# §5. Recommendations for ledger update
# =============================================================================
print("§5. Recommendations for ledger update")
print("=" * 78)
print(f"""
  After this audit, ledger row exposure to the Session-2 CMB θ_*
  falsification is:

  | Row | Quantity | Exposure | Reason |
  |-----|----------|----------|--------|
  | P17 | N_hub | INSULATED | cascade only at z = 0 |
  | P19 | H_0 | INSULATED | cascade only at z = 0; observer-rate (16/15) at z = 0 |
  | P20 | t_0 | INSULATED | cascade only at z = 0 |
  | P22 | Ω_DM/Ω_m | INSULATED | frame-invariant ratio |
  | P23 | Ω_DM, Ω_b | EXPOSED via P24 | factor-of-2 closure blocked (Path A) |
  | P24 | Λ_CC | EXPOSED | factor-of-2 closure blocked; CMB-θ_* tension |
  | P25 | n_s | EXPOSED | slow-roll identification incompatible with cascade-coasting at inflation |
  | P26 | r | EXPOSED | slow-roll consistency relation; same as P25 |
  | A_s (Item 3) | A_s ≈ 2.07e-9 | EXPOSED | slow-roll identification inconsistent with cascade-coasting at inflation |

  Net change: P25, P26, and A_s (Item 3) join the exposed set. The
  cosmology arc's "completion" verdict on Item 3 (THEOREM-GRADE-
  CONDITIONAL via Feshbach Exponent Principle citation) needs an
  ADDITIONAL conditional gate: "compatible with framework's inflation-
  epoch H(z) profile, which is currently undefined and may be
  inconsistent with cascade-theorem coasting at all z".

  Recommended ledger updates:
    P25: add note "EXPOSED to Session 2 CMB θ_* falsification via
         slow-roll identification; framework's strict cascade-coasting
         at z = z_inflation conflicts with de-Sitter-like slow-roll
         assumption. See cascade_coasting_high_z_falsification_scoping
         doc. Awaiting Item 5 closure (pre-recombination physics
         reconciliation)."
    P26: same note as P25.

  For A_s (Item 3): the cosmology roadmap doc should be updated to add
  a SIXTH conditional gate (alongside the three named in Session 2's
  closure) — "compatible with framework's inflation-epoch H(z)".
""")
print("=" * 78)
print("DONE: P25/P26/A_s high-z exposure audit.")
print("Verdict: P25, P26, A_s all EXPOSED via slow-roll identification.")
print("=" * 78)
