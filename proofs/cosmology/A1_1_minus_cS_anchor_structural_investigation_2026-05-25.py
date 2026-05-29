#!/usr/bin/env python3
"""
A1 — structural investigation of T_GUT = M_unif × c_S × (1 − c_S) anchor.

The numerical observation: at the α=1/2 EXACT framework, the anchor

    T_GUT = M_unif × c_S × (1 − c_S) = M_unif × (1/12)(11/12) = M_unif × 11/144

gives T_today = 2.715 K vs observed 2.725 K (−0.36% residual). This is
a NUMERICAL HIT closer than the simple M_unif × c_S anchor (+8.4%).

This probe investigates whether (1 − c_S) has a structural derivation,
following the same calibration discipline that established v_Higgs c=5/12
and α_GUT c=1/k*: ANY new structural mechanism must reproduce v_Higgs
c=5/12 on the same machinery (master doc §8 rule 4).

Five candidate structural readings tested:
  R1. Bernoulli variance of Perron projection: c_S(1−c_S)
  R2. Fermi's golden rule cross-amplitude: |⟨gauge|sub⟩⟨non-gauge|sub⟩|² = c_S(1−c_S)
  R3. Mode-pair counting: (singlet × non-singlet) modes / total² = 11/144
  R4. Stark-Terras off-diagonal: cross-correlation between Perron and rest
  R5. Born-rule projector overlap: P(gauge) × P(dark)

For each, check the v_Higgs calibration: applied to v_Higgs, does the same
mechanism give c_v = 5/12 (or whatever the analogous mode-pair gives)?
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
M_unif_GeV = 1.985e16
T_CMB = 2.7255

k_star = 3
N_atoms = 4
n_E = 6
two_E = 2 * n_E

c_S = Fraction(1, two_E)         # 1/12 (Perron-singlet projection)
c_v = Fraction(5, two_E)         # 5/12 (v_Higgs from Routes H+C)
c_alpha_GUT = Fraction(1, k_star)  # 1/3 (α_GUT from Routes H+C)

N_GUT = N_hub / (M_unif_GeV / v_today) ** 4

# Mode dimensions per Stark-Terras catalogue on srs
dim_bipartite_marginal = 4   # cycle modes / H¹ Wilson loops
dim_perron_singlet = 1       # gauge-singlet uniform Perron
dim_perron_visible = 1       # u = k*-1
dim_oscillatory = 6          # λ_A = -1 factors
dim_NB_total = two_E         # = 12

dim_scalar_v_Higgs = dim_bipartite_marginal + dim_perron_singlet  # = 5 (c_v numerator)
dim_alpha_GUT = dim_bipartite_marginal                            # = 4 (c_α_GUT numerator)


print("=" * 76)
print("Structural investigation: T_GUT = M_unif × c_S × (1 − c_S)")
print("=" * 76)

# Numerical hit
T_GUT_candidate = M_unif_GeV * float(c_S) * float(1 - c_S)
T_today_pred = T_GUT_candidate * K_per_GeV * math.sqrt(N_GUT / N_hub)
print(f"\n  c_S = 1/12 = {float(c_S):.6f}  (Perron-singlet weight, theorem-grade)")
print(f"  1 - c_S = 11/12 = {float(1-c_S):.6f}")
print(f"  c_S × (1 - c_S) = 11/144 = {float(c_S * (1-c_S)):.6f}")
print(f"\n  T_GUT_candidate = M_unif × c_S × (1-c_S) = {T_GUT_candidate:.4e} GeV")
print(f"  T_today (α=1/2) = {T_today_pred:.4f} K  vs observed {T_CMB} K")
print(f"  Residual = {(T_today_pred - T_CMB)/T_CMB*100:+.3f}%")


# ---------------------------------------------------------------------------
# Calibration discipline check: does any structural reading of (1 - c_S)
# also give v_Higgs's c=5/12 via the same mechanism?
# ---------------------------------------------------------------------------
print(f"\n{'='*76}")
print("CALIBRATION DISCIPLINE — any new mechanism must reproduce v_Higgs c=5/12")
print('='*76)
print(f"""
The dark-correction master doc §8 rule 4: any new structural derivation
of a c_g value must ALSO reproduce v_Higgs's c=5/12 via the same machinery.
This is the discipline that catches numerology.

For T_GUT = M_unif × c_S × (1 - c_S), test each candidate reading.
""")


# ---------------------------------------------------------------------------
# R1 — Bernoulli variance of Perron projection
# ---------------------------------------------------------------------------
print(f"{'='*76}")
print("R1 — Bernoulli variance c_S × (1 - c_S)")
print('='*76)
print(f"""
Interpretation: treating the Perron-singlet projection as a Bernoulli
random variable with probability c_S, the variance is c_S × (1 - c_S).
For srs: 1/12 × 11/12 = 11/144 = 0.0764.

Calibration: applied to v_Higgs, the analog Bernoulli variance is
c_v × (1 - c_v) = (5/12)(7/12) = 35/144 ≈ 0.243.

  T_GUT analog under R1: M_unif × 11/144 = {float(M_unif_GeV * 11/144):.4e} GeV
  v_Higgs analog under R1: c_v × (1 - c_v) = {Fraction(5, 12) * Fraction(7, 12)} = {float(Fraction(5,12) * Fraction(7,12)):.4f}
  Actual v_Higgs c: {c_v} = {float(c_v):.4f}

  Routes H+C gives v_Higgs c = 5/12, NOT 35/144. R1's "Bernoulli variance"
  reading would PREDICT v_Higgs c = 35/144, which is wrong by factor
  {float(c_v) / float(Fraction(5,12) * Fraction(7,12)):.2f}.

  STATUS: FAILS calibration check. R1 is NOT the framework mechanism.
""")


# ---------------------------------------------------------------------------
# R2 — Fermi's golden rule cross-amplitude
# ---------------------------------------------------------------------------
print(f"{'='*76}")
print("R2 — Fermi's golden rule cross-amplitude: matrix element squared")
print('='*76)
print(f"""
Interpretation: substrate state decomposes as
   |substrate⟩ = √c_S |gauge⟩ + √(1-c_S) |non-gauge⟩
Matrix element ⟨gauge|V|non-gauge⟩ has |amplitude| = √(c_S × (1-c_S)).
Fermi's golden rule rate Γ ∝ |matrix element|² = c_S × (1-c_S).

Calibration: applied to v_Higgs, the analog rate would be c_v × (1-c_v)
under the same Fermi-rule framing. Same number 35/144 as R1.

  Same result as R1 — gives 35/144 for v_Higgs analog, not 5/12.
  STATUS: FAILS calibration check. R2 is NOT the framework mechanism.
""")


# ---------------------------------------------------------------------------
# R3 — Mode-pair counting
# ---------------------------------------------------------------------------
print(f"{'='*76}")
print("R3 — Mode-pair counting: (singlet modes × non-singlet modes) / total²")
print('='*76)
print(f"""
Interpretation: count ORDERED PAIRS of (singlet mode, non-singlet mode)
in the NB sector. For srs: 1 singlet × 11 non-singlet = 11 ordered pairs.
Normalize by total NB² = 12² = 144. Gives 11/144.

Calibration: applied to v_Higgs's mode count (5 = bipartite-marginal +
singlet), the analog is (5 modes × 7 non-modes) / 144 = 35/144.

Same as R1, R2. Fails calibration.

UNLESS the relevant pair count is DIFFERENT for T vs v_Higgs:
  - For T (thermal exchange): maybe count Perron-singlet × everything-else
    pairs only — 1 × 11 = 11.
  - For v_Higgs (scalar 2-pt): maybe count scalar-cluster (5 modes) ×
    something else.

But the master doc's c_v = 5/12 is NOT a pair count — it's a SINGLE mode
count (cycle + scalar) divided by 2|E|. Different structure.

  STATUS: FAILS — R3's mode-pair denominator (144) doesn't match Routes H+C
  for v_Higgs (denominator 12). Inconsistent framework normalization.
""")


# ---------------------------------------------------------------------------
# R4 — Stark-Terras off-diagonal: cross-mode correlation
# ---------------------------------------------------------------------------
print(f"{'='*76}")
print("R4 — Stark-Terras off-diagonal: cross-correlation matrix element")
print('='*76)
print(f"""
Interpretation: the Hashimoto operator's spectral decomposition has
diagonal entries (the eigenvalues) and off-diagonal entries (mode mixing).
For an off-diagonal between Perron-singlet and a non-Perron mode, the
matrix element might scale as √(c_S × (1-c_S)) (amplitude) or as
c_S × (1-c_S) (squared).

This would be the substrate's natural "mixing strength" between gauge-
readable and dark sectors at gauge unification.

Calibration: for v_Higgs's c=5/12, the same mixing would predict v_Higgs
c=5/12 × 7/12 = 35/144 — same failure as R1.

UNLESS the relevant mode-pair is SPECIFIC to T (e.g., Perron-singlet
mixing with scalar-Perron-visible mode at u=2, the singlet's nearest
spectral neighbor).

For srs: Perron-singlet at u=1 and Perron-visible at u=2 form a
2-dim Perron-adjacency block. Off-diagonal in this 2×2 block: dim 1.

If T_GUT relates to this Perron-block off-diagonal coupling: c = 1/(2|E|) ×
(something) = 1/12 × 1 = 1/12. Just c_S, NOT c_S × (1-c_S).

  STATUS: FAILS — can't structurally derive the (1-c_S) factor from a
  specific spectral off-diagonal without ad-hoc reading.
""")


# ---------------------------------------------------------------------------
# R5 — Born-rule projector overlap as gauge-dark exchange
# ---------------------------------------------------------------------------
print(f"{'='*76}")
print("R5 — Born-rule projector overlap (gauge × dark sector amplitudes)")
print('='*76)
print(f"""
Interpretation: substrate has projectors P_gauge (onto gauge-readable
modes) and P_dark = 1 - P_gauge (onto dark/non-gauge modes). The
expected value of P_gauge × P_dark in a Perron-projected state is
c_S × (1 - c_S).

Calibration: for v_Higgs, the same Born-rule overlap of P_scalar ×
P_non-scalar would give c_v × (1 - c_v) = 35/144. Fails.

UNLESS the (1-c_S) refers SPECIFICALLY to "the share NOT seen as
gauge-readable" — i.e., the OBSERVATIONAL COMPLEMENT, not a Born-rule
overlap. In that case, for v_Higgs the "observational complement" of
scalar-Perron projection might be... well, what?

The framework's v_Higgs = δ²M_Pl/(√2·N_hub^(1/4)) IS the scalar Perron-
projected scale ALREADY. There's no obvious "observational complement"
that gives the (1-c_v) factor multiplicatively.

  STATUS: Reading R5 IS specific to thermal/observational class but lacks
  a clean machinery that calibrates against v_Higgs's known c=5/12.
""")


# ---------------------------------------------------------------------------
# Honest verdict
# ---------------------------------------------------------------------------
print(f"\n{'='*76}")
print("HONEST VERDICT — (1 - c_S) structural investigation")
print('='*76)
print(f"""
Tested five candidate structural readings (R1-R5) for the (1 - c_S) =
11/12 factor in T_GUT = M_unif × c_S × (1 - c_S).

ALL FIVE FAIL the calibration discipline: applied to v_Higgs via the
same machinery, they predict v_Higgs c = c_v × (1 - c_v) = 35/144 (or
similar), NOT the established c_v = 5/12.

KEY OBSERVATION:
  Routes H + C for v_Higgs give c_v = 5/12 by counting modes (bipartite-
  marginal + scalar = 5 modes / 12 total). This is a LINEAR count, not
  a Bernoulli-variance / Born-rule product.

  c_S × (1 - c_S) has a DIFFERENT structural form than c_v = 5/12. They
  can't both come from the same mechanism unless the mechanism has
  observable-class-specific selection rules that we haven't derived.

  An ad-hoc "for T use Bernoulli, for v_Higgs use linear count" reading
  is NOT structural derivation — it's fitting via mechanism choice.

INTERPRETATIONS:

  (A) The (1 - c_S) numerical match is COINCIDENCE.
      Per an internal note and the
      strict calibration discipline, a numerical match without a
      mechanism that passes the v_Higgs calibration check is numerology.
      The (1 - c_S) factor should NOT be promoted to a structural candidate.

  (B) The (1 - c_S) reflects a NEW observable class for T (thermal
      observables) not yet catalogued in the master doc.
      This is possible — temperature is thermodynamic-conjugate, not a
      coupling. The framework's master doc §6 step 1 classifies tensor
      character of observables (gauge 1-pt, scalar 2-pt, etc.) but
      doesn't have an entry for "thermal exchange rate." Maybe T has
      its own selection rule that gives c_T = c_S × (1 - c_S), distinct
      from c_v's linear count and α_GUT's gauge-cycle count.

      To make (B) work as a CLEAN derivation, we'd need:
        1. An explicit observable class for T (thermodynamic conjugate)
        2. A selection rule on B_NB modes specific to this class
        3. A calibration check against another known T-like observable
           (but the framework doesn't have one!)

      Without a SECOND known T-like observable to calibrate against,
      we can't structurally distinguish (B) from (A). The 1-observable
      situation makes (B) unfalsifiable in the master doc's framework.

  (C) M_unif × c_S is the WRONG anchor entirely.
      The right anchor might be a different structural object that
      happens to have value M_unif × 11/144. Need to identify what.

      Candidates to investigate (other than (1-c_S) factor):
        - M_unif × c_S² × scaling factor
        - M_unif × N_trivial/N_couplings × scaling = M_unif × 2/36 ×
          scaling = M_unif × scaling/18. For T_GUT ≈ M_unif × 11/144,
          need scaling × 1/18 = 11/144, so scaling = 11/8. No clean
          framework primitive.
        - M_unif × n_g/something. n_g = 15 girth cycles. For T_GUT ≈
          M_unif × 11/144 = M_unif × 0.0764. If T_GUT = M_unif × n_g/X,
          X = 15/0.0764 = 196. Not a clean primitive.

      None of these are obviously cleaner than (1-c_S).

VERDICT: (1 - c_S) anchor investigation lands at HONEST NEGATIVE for
  structural derivation.

  The numerical match is real but the calibration discipline FAILS:
  no mechanism that gives T's c_T = c_S(1-c_S) also gives v_Higgs's
  c_v = 5/12 (which is theorem-grade upstream).

  Per the same discipline that caught the d_eff misidentification and
  the W58 δ-pattern fit, this should be reported as a NUMERICAL
  OBSERVATION, not promoted to a candidate.

  The 8% T_today residual remains GENUINELY OPEN. The (1 - c_S) form
  is suggestive but lacks a structural derivation that passes v_Higgs
  calibration.

NEXT STEPS (if anchor investigation continues):
  - Search for a T observable class definition that's structurally
    motivated and calibrates against an existing framework observable
    (NOT v_Higgs / α_GUT / δ_r — they're already explained by different
    mechanisms)
  - Or accept the 8% T_today residual as a precision-floor for A1 and
    move on
  - Or reframe T_GUT entirely (not anchored at M_unif × c_S)
""")

print("=" * 76)
print("STATUS: HONEST NEGATIVE — (1-c_S) factor lacks structural derivation")
print("        that passes v_Higgs c=5/12 calibration discipline.")
print("=" * 76)
