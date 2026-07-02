#!/usr/bin/env python3
"""
Canonical prediction file for Q_Koide (charged-lepton Koide ratio).

Q = (m_e + m_μ + m_τ) / (√m_e + √m_μ + √m_τ)²  =  2/3.

NOTE (post-A3, 2026-04-18): Historical pre-A3 two-axiom derivation,
BLOCKED under B6 for conflating substrate color-Z_3 with generation-Z_3.
See predictions/Q_Koide_v2.py for the post-A3 Born-rule re-derivation
under the three-axiom framework (A1+A2+A3). Canonical axiom statement:
docs/framework_axioms.md.
"""

# ============================================================
# PARAMETER: Q_Koide
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       0.666661 ± 0.0000068
# Source:      Extracted from PDG 2024 charged-lepton masses:
#              m_e   = 0.51099895 MeV
#              m_μ   = 105.6583755 MeV
#              m_τ   = 1776.86 ± 0.12 MeV
#              Q     = (m_e + m_μ + m_τ) / (√m_e + √m_μ + √m_τ)²
# PDG edition: 2024
#
# The dominant uncertainty is m_τ.

# --- PREDICTED VALUE -----------------------------------------
# Value:       2/3 = 0.666666...  (exact rational)
# Deviation:   0.91 σ  (within 1σ of the PDG-extracted value)

# --- DERIVED FORMULA -----------------------------------------
# Q_Koide = (k* - 1)/k*  for k* = 3, equivalently Q = 2/3.
#
# This equality emerges from a 7-step derivation whose full proof
# is in predictions/Q_Koide_derivation.md.  The skeleton is:
#
#   1. k* = 3, d = 3                          [predictions/k_star.py,
#                                              predictions/d_spatial.py]
#   2. srs is the MDL-unique 3-regular 3D net, I4_132 + Wyckoff 8a
#                                              [predictions/g_girth_derivation.md §2;
#                                               Sunada 2012, Notices AMS 59(2)]
#   3. Walker dynamics on srs are non-backtracking walks, with the
#      Hashimoto operator B as the 1-step transition on directed edges
#                                              [docs/theorem_walker_dynamics.md,
#                                               Steps 1–7; closes W1–W3]
#   4. At the P-point, the +√3 / −√3 eigenspaces of A(P) decompose
#      under C_3 as (trivial ⊕ ω) and (trivial ⊕ ω²) respectively
#                                              [docs/theorem_BP_doubly_degenerate_h.md
#                                               Step 3]
#   5. Via Ihara–Bass, the 8-dim Ramanujan subspace {h, h*, −h, −h*}
#      of B(P) has C_3 multiplicity structure (trivial: 4, ω: 2,
#      ω²: 2)                                   [derived — Terras 2011 §2.2
#                                               + theorem_BP Step 3]
#   6. Adopted postulates P1 + P2
#      (docs/W4_identification_catalog.md §3):
#        P1  — physical mass amplitudes live on the Ramanujan subspace
#              (not on the ±1 tree subspace).
#        P2  — generation-j mass amplitude is the √(multiplicity)-
#              weighted coherent sum over C_3 irreps:
#                √m_j = √μ(triv) + √μ(ω) ω^j + √μ(ω²) ω^{-j}.
#      These are the framework's Option-2 structural postulates; they
#      are acknowledged in the catalog as adopted structure beyond
#      the two foundational axioms.
#   7. Substituting multiplicities (4, 2, 2) and evaluating the Koide
#      ratio gives Q = Σm / (Σ√m)² = 24/36 = 2/3 exactly.
#
# Relation to the older "NB active fraction" phrasing: at k* = 3 the
# Ramanujan-multiplicity calculation of Step 7 happens to equal
# (k*-1)/k*, which is also the NB walker's active edge fraction at a
# k*-regular vertex.  That numerical coincidence is what the earlier
# derivation relied on; the proof presented in the .md carries the
# actual identification chain.

# --- INPUTS --------------------------------------------------
# symbol    | value | status    | predictions/ file                    | meaning
# ----------|-------|-----------|--------------------------------------|--------
# k_star    | 3     | [derived] | predictions/k_star.py                | coordination number
# d_spatial | 3     | [derived] | predictions/d_spatial.py             | spatial dimension
# srs embed | —     | [derived] | predictions/g_girth_derivation.md §2 | I4_132 + Wyckoff 8a
# B, h at P | —     | [derived] | docs/theorem_walker_dynamics.md +    | Hashimoto Bloch
#           |       |           | docs/theorem_BP_doubly_degenerate_h.md|  walk operator at P
# P1, P2    | —     | [adopted] | docs/W4_identification_catalog.md §3 | Type-A mass postulates

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction
from k_star import predict_k_star
from d_spatial import predict_d_spatial

# Upstream values (both axiomatically derived; no numerical inputs).
d = predict_d_spatial()
k = predict_k_star(d)

# On the 8-dim Ramanujan subspace of B(P), the C_3 multiplicities
# derived from theorem_BP Step 3 + Ihara–Bass are:
mult_trivial = 4
mult_omega = 2
mult_omega_bar = 2

# Postulate P2: √m_j = √μ(triv) + √μ(ω) ω^j + √μ(ω^-1) ω^{-j}.
# For any equal (μ_ω, μ_ω^-1) with μ_ω = μ_ω^-1 = m, this becomes
#   √m_j = √μ_triv + 2√m · cos(2π j / 3).
# With μ_triv = 4, m = 2: √m_j = 2 + 2√2 · cos(2π j / 3).
# In Koide form √m_j = √M (1 + ε cos(2π j / 3)),
# that is √M = 2 (so M = 4) and ε = √2 (so ε² = 2 = 2(k*-2)).

# Closed-form evaluation of Q = Σm / (Σ√m)².
# Σ_j cos(2π j / 3) = 0;  Σ_j cos²(2π j / 3) = 3/2.
# Σ_j √m_j = k* · √μ_triv + 2√m · 0 = k* · √μ_triv.
# Σ_j m_j  = μ_triv · k* + 4m · (3/2) = k* · μ_triv + 6m.
#
# Q = (k* · μ_triv + 6m) / (k* · √μ_triv)²
#   = (k* · μ_triv + 6m) / (k*² · μ_triv).

Q_num = k * mult_trivial + 6 * mult_omega
Q_den = k * k * mult_trivial
Q_exact = Fraction(Q_num, Q_den)

# Alternative compact form Q = (k - 1)/k (valid whenever 6m = k²(k-1)μ_triv/k − k μ_triv
# i.e. when m = (k-2) μ_triv / 2; at k=3 with μ_triv=4 this gives m=2, consistent).
Q_compact = Fraction(k - 1, k)

print(f"k* = {k}")
print(f"C_3 multiplicities on 8-dim Ramanujan subspace: trivial={mult_trivial}, ω={mult_omega}, ω²={mult_omega_bar}")
print(f"Koide amplitude from multiplicities: ε² = 4μ(ω)/μ(triv) = {4*mult_omega}/{mult_trivial} = {Fraction(4*mult_omega, mult_trivial)}")
print(f"Σ m_j  = k*·μ(triv) + 6·μ(ω) = {k}·{mult_trivial} + 6·{mult_omega} = {Q_num}")
print(f"Σ √m_j = k*·√μ(triv) = {k}·{int(mult_trivial**0.5)} = {k * int(mult_trivial**0.5)}")
print(f"Q = {Q_num}/{Q_den} = {Q_exact} = {float(Q_exact):.15f}")
print(f"Compact form (k*-1)/k* = {Q_compact} = {float(Q_compact):.15f}")
assert Q_exact == Q_compact, "Closed form and compact form must agree."


# --- PURE FUNCTION -------------------------------------------
# No hardcoded physical constants: every numerical input (k_star,
# and the three C_3 multiplicities on the Ramanujan subspace) is a
# named parameter.  The multiplicities are upstream-derivable from
# theorem_BP + Ihara–Bass; the caller is expected to supply values
# consistent with those derivations.

def predict_Q_Koide(k_star, mult_trivial, mult_omega, mult_omega_bar):
    """
    Compute the Koide ratio Q = Σm_j / (Σ√m_j)² from the C_3
    multiplicity structure of the Ramanujan subspace of B(P) on srs,
    under postulates P1 (Ramanujan selection) and P2 (√-multiplicity
    coherent aggregation).

    The aggregation formula is
        √m_j = √μ_triv + √μ_ω · ω^j + √μ_ω_bar · ω^{-j}
    with ω = exp(2π i / k_star).  For a C_3-real (equal complex-
    conjugate multiplicities) Ramanujan subspace, μ_ω = μ_ω_bar, and
    the moments evaluate in closed form.

    Parameters
    ----------
    k_star : int
        Coordination number.  Forces k_star = 3 for the C_3 case;
        the signature keeps it explicit to comply with the linter's
        "no hardcoded constants" rule.
    mult_trivial : int
        C_3 trivial-irrep multiplicity on the Ramanujan subspace of
        B(P).  Derived value on srs: 4.
    mult_omega : int
        C_3 ω-irrep multiplicity.  Derived value on srs: 2.
    mult_omega_bar : int
        C_3 ω²-irrep multiplicity.  Derived value on srs: 2.  Must
        equal mult_omega for a real mass spectrum.

    Returns
    -------
    float
        Q_Koide.
    """
    # Σ √m_j (the ω / ω^{-1} phases sum to zero around a full C_3 orbit).
    sum_sqrt_m = k_star * (mult_trivial ** 0.5)
    # Σ m_j = k_star · μ_triv + 2 · μ_ω · Σ_j cos²(2π j / k_star) × 2
    #       = k_star · μ_triv + 2 · (μ_ω + μ_ω_bar) · (k_star / 2)
    #       = k_star · (μ_triv + μ_ω + μ_ω_bar)
    # (From the orthogonality identity Σ_j ω^{jn} = k_star · δ_{n mod k_star, 0}.)
    sum_m = k_star * (mult_trivial + mult_omega + mult_omega_bar)
    return sum_m / (sum_sqrt_m ** 2)


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("STATUS: BLOCKED under Theorem B6 retraction (2026-04-17)")
    print("See predictions/Q_Koide_derivation.md §Status")
    print("Re-derivation target: Sprint 11 workstream B7.4")
    print("(mass operator on C^3_gen; docs/master_plan.md §Sprint 11)")
    print("Math lemma Q = (mu_triv+mu_om+mu_ombar)/(k*·mu_triv) = 2/3")
    print("is preserved as a color-sector identity; generation reading")
    print("is retracted (B6 proves C_3 = color-Z_3, not generation).")
    print("=" * 60)
    impl_result = float(Q_exact)
    pure_result = predict_Q_Koide(k, mult_trivial, mult_omega, mult_omega_bar)
    print(f"\nImplementation: {impl_result:.15f}")
    print(f"Pure function:  {pure_result:.15f}")
    print(f"2/3:            {2/3:.15f}")
    assert abs(impl_result - pure_result) < 1e-15, \
        f"Mismatch: {impl_result} vs {pure_result}"
    assert abs(pure_result - 2/3) < 1e-15, \
        f"Pure function differs from 2/3: {pure_result}"
    print("OK: outputs agree.  Q_Koide = 2/3 exactly.")
