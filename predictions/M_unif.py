#!/usr/bin/env python3
"""
M_unif — gauge unification scale.

NEW DERIVATION (2026-05-04 EOD): M_unif as a substrate-derived structural
scale, parallel to the m_ν₃ closure pattern.

THE FORMULA:

    M_unif = α_GUT × α_1_bare × M_Pl
           = (1/24) × (2/3)^8 × M_Pl
           = (32 / k*^(g-1)) × M_Pl                 [equivalent rational form]
           = N_atoms² × M_R                          [equivalent geometric form]

In framework-natural units (M_Pl = 8/√π, see docs/framework/framework_natural_units.md):
    M_unif = (32/k*^(g-1)) × (8/√π) ≈ 7.34 × 10⁻³

In GeV via CODATA M_Pl unit translation:
    M_unif ≈ 1.985 × 10¹⁶ GeV

KEY PROPERTIES:
- All inputs framework-internal: α_GUT (theorem-grade Class C), α_1_bare
  (theorem-grade Class A), M_Pl (theorem-grade structural per Drude+Planck
  convention, untethered structural prediction in framework-natural units).
- N-INDEPENDENT (substrate-local family): no cosmological N_hub dependence.
- Match: -0.76% vs MSSM single-regime unification benchmark (~2 × 10¹⁶ GeV).

STRUCTURAL READING (B2 — "Reading B2" per
an internal working note):

    M_unif = (full Bloch sector dim)² × (trivial sector dim) × (Markov return per step)^(g-1) × M_Pl
           =        4²                ×          2          ×           (1/k*)              ^(g-1) × M_Pl
           = 32 / k*^(g-1) × M_Pl

PHYSICAL READING:
- Gauge bosons couple to all matter (no sector restriction in unbroken
  PS phase) → bilinear in full Bloch sector → factor (N_atoms)² = 16.
- Walker excursion mediating gauge propagation is the same trivial-mode
  closed walk over the girth cycle that gave M_R → factor 2 × (1/k*)^(g-1).
- Combining: gauge-bilinear × trivial-walker = 16 × M_R_factor = 32 × (1/k*)^(g-1).

This is the substrate-only parsimony-preferred reading. Competing readings
(Cl(4) × chirality, PS one-generation × chirality) coincidentally also
give 32 due to the algebraic accident N_atoms² = dim Cl(4) = PS one-gen
dim = 16, but use additional structural assumptions.

STATUS (2026-05-04 EOD+1): THEOREM-GRADE-CONDITIONAL post-5-stage closure
program.

Stage breakdown:
  Stage 1: gauge field formalism on srs (proofs/gauge/srs_gauge_field_definition.py)
  Stage 2: Wilson action quadratic form, M² eigenspectrum {0×6, 2×5, 50×1}
           (srs_wilson_action_quadratic.py) — ruled out cycle-incidence as 32 source
  Stage 3: matter loop trace DERIVES 32 = N_atoms² × N_trivial structurally
           (srs_gauge_self_energy.py) — Reading B2 promoted candidate → derived
  Stage 4: linear form justified via substrate-local family pattern + Wilsonian
           saturation (srs_M_unif_self_consistency.py)
  Stage 5: audit v2 + ledger graduation (this status banner update)

THEOREM-GRADE-CONDITIONAL on:
  (a) Stage 3 matter trace structural derivation (derived)
  (b) Substrate-local family LINEAR pattern (parallelism with m_ν₃, M_R, v BZJ)
  (c) Wilsonian-on-substrate Type 3 standard machinery

Numerical match -0.76% vs the MSSM single-regime benchmark (M_unif_obs
itself is not a measurement, so Clause 8 is a benchmark consistency
check rather than a PDG comparison).

Remaining gap: full QFT-on-substrate formalism for Wilsonian saturation
(established by parallelism, not yet from a single explicit RG equation).
Same grade as m_ν₃ closure 2026-05-04.

LEVERAGE: 6+1 cluster targets {sin²θ_W(M_Z), g_1, g_2, g_3, α_EM, α_s, R∞}
graduate to UNIQUE-THEOREM-GRADE (or STRUCTURAL-DERIVATION-CONDITIONAL,
depending on M_unif's eventual grade) via standard SM/MSSM RG running from
M_unif to M_Z.

COMPANION DOCS:
- proofs/foundations/m_unif_candidate_identity.py (numerical verification)
- proofs/foundations/m_unif_full_bloch_bilinear.py (Reading B2 analysis)
- predictions/M_Pl_natural.py (untethered structural M_Pl)
- predictions/alpha_GUT.py (theorem-grade α_GUT = 1/24)
- predictions/alpha_1.py (theorem-grade α_1_bare = (2/3)^8)
- predictions/alpha_EM.py (downstream consumer; RG-run from M_unif)
"""

# ============================================================
# PARAMETER: M_unif (gauge unification scale)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       M_unif ≈ 2 × 10¹⁶ GeV (canonical MSSM unification scale)
# Source:      Standard MSSM RG analysis inverting measured g_1, g_2, g_3 at M_Z.
# Note:        M_unif is NOT directly observed; it's inferred from RG running.
#              The single-regime MSSM-style running (framework's native picture,
#              no M_SUSY threshold; see ADOPTED-MSSM-Sb 2026-05-14 PM revision)
#              gives a single-valued reference point ~2 × 10¹⁶ GeV.

# --- PREDICTED VALUE -----------------------------------------
# Value:       M_unif = 32/k*^(g-1) × M_Pl
#                    = (32/3⁹) × 1.22089 × 10¹⁹ GeV
#                    ≈ 1.985 × 10¹⁶ GeV
# Deviation:   -0.76% vs MSSM benchmark (~2 × 10¹⁶ GeV).

# --- DERIVED FORMULA -----------------------------------------
# M_unif = α_GUT × α_1_bare × M_Pl
#        = (1/24) × (k*-1)^(g-2)/k*^(g-2) × M_Pl
#        = (1/24) × (2/3)^8 × M_Pl
#        = (32/k*^(g-1)) × M_Pl                         [rational simplification]
#        = N_atoms² × M_R                               [geometric form via 2026-05-04 M_R closure]
#
# Logical chain:
#   Step 1: α_GUT = 1/24 (theorem-grade Class C; predictions/alpha_GUT.py)
#   Step 2: α_1_bare = (k*-1)^(g-2)/k*^(g-2) = (2/3)^8 (theorem-grade Class A; predictions/alpha_1.py)
#   Step 3: M_Pl substrate-anchored (theorem-grade per G_sub Drude closure;
#           untethered structural in framework-natural units per
#           predictions/M_Pl_natural.py)
#   Step 4: Reading B2 — gauge two-point function bilinear-in-full-Bloch ×
#           trivial-walker gives (N_atoms)² × (trivial sector dim) × (1/k*)^(g-1)
#           = 32 × (1/k*)^(g-1) [STRUCTURAL HYPOTHESIS, parsimony-preferred]
#   Step 5: Combining: M_unif = α_GUT × α_1_bare × M_Pl = 32/k*^(g-1) × M_Pl

# --- INPUTS --------------------------------------------------
# symbol     | value        | status            | predictions/ file              | meaning
# -----------|--------------|-------------------|--------------------------------|-----
# k_star     | 3            | [derived]         | predictions/k_star.py          | Hashimoto Perron, theorem-grade
# g_girth    | 10           | [derived]         | predictions/g_girth.py         | srs girth, theorem-grade
# N_atoms    | 4            | [derived]         | (structural, srs)              | atoms per primitive cell
# alpha_GUT  | 1/24         | [derived]         | predictions/alpha_GUT.py       | unified gauge coupling (theorem-grade)
# alpha_1    | (2/3)^8      | [derived]         | predictions/alpha_1.py         | bare NB walker survival (theorem-grade)
# M_Pl       | 8/√π M_subs  | [derived]         | predictions/M_Pl_natural.py    | Planck mass (untethered structural)

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import math
import functools
from fractions import Fraction

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from k_star import predict_k_star
from d_spatial import predict_d_spatial
from g_girth import predict_g_girth
from alpha_1 import predict_alpha_1

# Substrate primitives (theorem-grade)
d_val = predict_d_spatial()
k_val = predict_k_star(d_val)
g_val = predict_g_girth(k_val, d_val)
alpha_1_bare = (k_val - 1)**(g_val - 2) / k_val**(g_val - 2)  # = (2/3)^8 for srs

from V_count import V_count_pred as N_atoms  # = 4, srs primitive cell |V| / K_4 quotient (predict_V_count)
alpha_GUT = Fraction(1, 24)

# Reading B2 structural counting (rational form)
M_unif_factor = Fraction(32, k_val**(g_val - 1))  # = 32/3^9 = 32/19683

# Equivalent forms
M_unif_factor_v2 = alpha_GUT * Fraction(2, 3)**(g_val - 2)         # α_GUT × α_1_bare
M_R_factor = Fraction(2, k_val**(g_val - 1))
M_unif_factor_v3 = N_atoms**2 * M_R_factor                          # N²·(M_R/M_Pl)

assert M_unif_factor == M_unif_factor_v2 == M_unif_factor_v3

# Anthropocentric SI translation (single-source from M_Pl_natural.py)
from M_Pl_natural import M_Pl_GeV   # CODATA — comparison/display only

# Predictions
M_unif_GeV = float(M_unif_factor) * M_Pl_GeV
M_unif_natural = float(M_unif_factor) * (8.0 / math.sqrt(math.pi))   # In framework-natural units

# Module-level exports
M_unif_pred = M_unif_GeV
M_unif_obs = 2.0e16          # MSSM single-regime reference value (~2 × 10¹⁶ GeV).

print("=" * 68)
print("  M_unif  --  Gauge unification scale  --  THEOREM-GRADE-CONDITIONAL")
print("    (Stage 4 closed 2026-05-14 PM under framework's substrate-spectral")
print("     mass-as-flux definition — same template as M_R with matter-bilinear")
print("     coefficient from Stage 3.  See:")
print("     proofs/gauge/srs_M_unif_step4_substrate_spectral.py)")
print("=" * 68)
print(f"  α_GUT      = {alpha_GUT} = {float(alpha_GUT):.6f}              [theorem-grade]")
print(f"  α_1_bare   = (2/3)^{g_val-2} = {alpha_1_bare:.6f}    [theorem-grade]")
print(f"  M_Pl       = 8/√π × M_substrate = {8.0/math.sqrt(math.pi):.4f} (framework-natural)")
print(f"             = {M_Pl_GeV:.4e} GeV [via CODATA unit translation]")
print()
print(f"  Rational form:  M_unif = (32/k*^(g-1)) × M_Pl")
print(f"                = {M_unif_factor} × M_Pl")
print(f"                = {float(M_unif_factor):.4e} × M_Pl")
print(f"                ≈ {M_unif_GeV:.4e} GeV  (via CODATA M_Pl)")
print()
print(f"  Geometric form: M_unif = N_atoms² × M_R = 16 × M_R")
print(f"                   M_R    = (2/k*^(g-1)) × M_Pl ≈ 1.241 × 10¹⁵ GeV")
print(f"                   M_unif = 16 × 1.241 × 10¹⁵ = {16 * 2/k_val**(g_val-1) * M_Pl_GeV:.4e} GeV")
print()
print(f"  STAGE 4 CLOSURE (2026-05-14 PM): substrate-spectral mass template")
print(f"    M_X = (coefficient) × (1/k*)^(g-1) × M_Pl")
print(f"    M_R    : coefficient = N_trivial            = 2  (single-mode propagator)")
print(f"    M_unif : coefficient = N_atoms² × N_trivial = 32 (matter-bilinear gauge two-point)")
print(f"  Linear form is NATIVE under framework's mass-as-substrate-spectral-quantity")
print(f"  definition (mass-as-flux); no Wilsonian-saturation smuggle. Dark corrections")
print(f"  NOT applicable (parity not violated at unbroken-PS scale).")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_M_unif_factor(k_star, g_girth, p_toggle):
    """
    Predict M_unif/M_Pl as a pure rational fraction.

    Formula
    -------
        M_unif/M_Pl = α_GUT × α_1_bare = (1/24) × ((k*-1)/k*)^(g-2)
                    = 32/k*^(g-1) for k* = 3, g = 10

    All numeric coefficients sourced from framework primitives:
      24 (label count = 2^k*·k*) = p_toggle^k_star · k_star
       1 (NB constraint)          = p_toggle - 1
       2 (Feshbach exponent g-2)  = p_toggle

    Parameters
    ----------
    k_star : int
        Substrate coordination (= 3 for srs).
    g_girth : int
        Substrate girth (= 10 for srs).
    p_toggle : int
        Toggle arity (from predict_p_toggle).

    Returns
    -------
    Fraction
        M_unif/M_Pl as exact rational.
    """
    one_nb = p_toggle - 1                                       # = 1
    label_count = p_toggle**k_star * k_star                      # = 24
    feshbach_n_fixed = p_toggle                                  # = 2
    alpha_GUT = Fraction(one_nb, label_count)                    # = 1/24
    alpha_1 = Fraction(k_star - one_nb, k_star)**(g_girth - feshbach_n_fixed)
    return alpha_GUT * alpha_1


@functools.lru_cache(maxsize=None)
def predict_M_unif_GeV(k_star, g_girth, M_Pl_GeV):
    """
    Predict M_unif in GeV, given the structural ratio and M_Pl in GeV.

    Parameters
    ----------
    k_star : int
        Substrate coordination.
    g_girth : int
        Substrate girth.
    M_Pl_GeV : float
        Planck mass in GeV (unit translation; CODATA).

    Returns
    -------
    float
        M_unif in GeV.
    """
    from p_toggle import predict_p_toggle
    p = predict_p_toggle()
    return float(predict_M_unif_factor(k_star, g_girth, p)) * M_Pl_GeV


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_result = M_unif_GeV
    pure_result = predict_M_unif_GeV(k_val, g_val, M_Pl_GeV)
    print()
    print("=" * 68)
    print("STATUS (parameter linter clauses):")
    print("  Clauses 1-5 (axiom/algebra/theorem/predictions chain):")
    print("    Step 1 [α_GUT]       = Type 4 (predictions/alpha_GUT.py)")
    print("    Step 2 [α_1_bare]    = Type 4 (predictions/alpha_1.py)")
    print("    Step 3 [M_Pl]        = Type 4 (predictions/M_Pl_natural.py + G_N.py)")
    print("    Step 4 [Reading B2]  = STRUCTURAL HYPOTHESIS, parsimony-preferred")
    print("    Step 5 [combination] = Type 2 (rational arithmetic)")
    print("  Clause 6 (K-meta-theorem):")
    print("    M_unif/M_Pl = 32/k*^(g-1) ∈ ℚ ⊂ K = ℚ(√2,√3,√5)  ✓")
    print("  Clause 7 (uniqueness):")
    print("    Inherits Row 4 closure for k*=3 (srs lattice).")
    print("  Clause 8 (benchmark consistency — M_unif is not directly measured):")
    print(f"    Benchmark = ~2 × 10¹⁶ GeV (MSSM single-regime inference from PDG α_i(M_Z)).")
    dev_rel_ = (M_unif_GeV - M_unif_obs) / M_unif_obs * 100
    print(f"    Deviation = {dev_rel_:+.2f}% vs benchmark.")
    print("=" * 68)

    print()
    print(f"  Implementation:  M_unif = {impl_result:.6e} GeV")
    print(f"  Pure function:   M_unif = {pure_result:.6e} GeV")
    assert abs(impl_result - pure_result) / impl_result < 1e-12
    print(f"  OK: outputs agree.")
    print()
    print(f"    M_unif predicted = {M_unif_GeV:.4e} GeV")
    print(f"    M_unif benchmark = {M_unif_obs:.4e} GeV  (MSSM single-regime inversion)")
    print(f"    Deviation        = {(M_unif_GeV - M_unif_obs)/M_unif_obs*100:+.2f}%")
    print()
    print(f"    In framework-natural units: M_unif = {M_unif_natural:.6f}")
    print(f"    (where M_substrate = 1, M_Pl = 8/√π ≈ 4.514)")

    # Sympy verification
    import sympy as sp
    k_sym = 3
    g_sym = 10
    alpha_GUT_sym = sp.Rational(1, 24)
    alpha_1_sym = sp.Rational(k_sym - 1, k_sym)**(g_sym - 2)
    M_unif_factor_sym = sp.simplify(alpha_GUT_sym * alpha_1_sym)
    expected = sp.Rational(32, k_sym**(g_sym - 1))
    diff = sp.simplify(M_unif_factor_sym - expected)
    assert diff == 0, f"Sympy: expected diff=0, got {diff}"
    print(f"\n  Sympy exact: α_GUT × α_1_bare = {M_unif_factor_sym} = {expected}")
    print(f"  OK: sympy confirms M_unif/M_Pl = 32/k*^(g-1) = 32/19683 exactly.")

    print()
    print("OK: M_unif structural prediction passes all checks at candidate-grade.")
    print("    Reading B2 (gauge bilinear × trivial walker) is parsimony-preferred.")
    print("    Numerical match within MSSM single-regime benchmark precision.")
    print("    Theorem-grade upgrade requires explicit gauge two-point computation.")
