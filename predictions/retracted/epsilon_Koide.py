#!/usr/bin/env python3
"""
Canonical prediction file for epsilon_Koide (Koide amplitude parameter).

ε = √2, the amplitude of the charged-lepton Koide mass parametrisation.

NOTE (post-A3, 2026-04-18): Historical pre-A3 two-axiom derivation,
cascades on Q_Koide (BLOCKED under B6 for conflating substrate color-Z_3
with generation-Z_3). Post-A3 re-derivation path via predictions/Q_Koide_v2.py.
Canonical axiom statement: docs/framework_axioms.md.
"""

# ============================================================
# PARAMETER: epsilon_Koide
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       1.414209 ± 0.000011
# Source:      Extracted from PDG 2024 charged-lepton masses via
#              the relation ε = √(6Q − 2), with Q extracted from
#              (m_e, m_μ, m_τ).
# PDG edition: 2024

# --- PREDICTED VALUE -----------------------------------------
# Value:       √2 = 1.41421356...  (exact)
# Deviation:   0.43 σ

# --- DERIVED FORMULA -----------------------------------------
# ε² = 4 · μ(ω) / μ(trivial)   =   2(k* − 2)  at srs
#
# The full proof is in predictions/epsilon_Koide_derivation.md.  It shares
# Steps 1–5 of predictions/Q_Koide_derivation.md verbatim (upstream k*=3,
# srs identification, walker_dynamics theorem, theorem_BP C_3 decomposition,
# Ihara–Bass lift to the 8-dim Ramanujan subspace with multiplicities
# (μ_triv, μ_ω, μ_ω²) = (4, 2, 2)), then extracts ε from the Koide-form
# matching:
#
#   √m_j = √μ_triv + √μ_ω · ω^j + √μ_ω² · ω^{-j}
#        = √M · (1 + ε · cos(2π j / k*))             (Koide form)
#
# giving  √M = √μ_triv  and  ε² = 4 · μ_ω / μ_triv.
#
# For srs multiplicities (4, 2, 2):
#   ε² = 4 · 2 / 4 = 2 = 2(k* − 2)   →   ε = √2.

# --- INPUTS --------------------------------------------------
# symbol    | value | status    | predictions/ file                    | meaning
# ----------|-------|-----------|--------------------------------------|--------
# k_star    | 3     | [derived] | predictions/k_star.py                | coordination number
# srs embed | —     | [derived] | predictions/g_girth_derivation.md §2 | I4_132 + Wyckoff 8a
# B at P    | —     | [derived] | docs/theorem_walker_dynamics.md +    | Hashimoto Bloch
#           |       |           | docs/theorem_BP_doubly_degenerate_h.md|  at P
# (μ_t,μ_ω,μ_ω²) = (4,2,2)      | [derived, Ihara–Bass + theorem_BP]   | C_3 mults on Ramanujan
# P1, P2    | —     | [adopted] | docs/W4_identification_catalog.md §3 | mass-amplitude postulates

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction
from k_star import predict_k_star
from d_spatial import predict_d_spatial

d = predict_d_spatial()
k = predict_k_star(d)

mult_trivial = 4
mult_omega = 2
mult_omega_bar = 2
assert mult_omega == mult_omega_bar, "Real mass spectrum requires μ_ω = μ_ω²."

eps_sq = Fraction(4 * mult_omega, mult_trivial)   # = 4·2/4 = 2
eps = math.sqrt(float(eps_sq))

# Cross-check: the same value is given by ε² = 2(k − 2) at srs.
eps_sq_compact = Fraction(2 * (k - 2), 1)
assert eps_sq == eps_sq_compact, \
    f"Ramanujan-multiplicity form {eps_sq} and compact form {eps_sq_compact} must agree."

print(f"k* = {k}")
print(f"Ramanujan-subspace C_3 multiplicities: (μ_triv, μ_ω, μ_ω²) = ({mult_trivial}, {mult_omega}, {mult_omega_bar})")
print(f"ε² = 4 · μ_ω / μ_triv = 4·{mult_omega}/{mult_trivial} = {eps_sq}")
print(f"Equivalently ε² = 2(k*-2) = 2·{k-2} = {eps_sq_compact}")
print(f"ε = √{eps_sq} = {eps:.15f}   (√2 = {math.sqrt(2):.15f})")


# --- PURE FUNCTION -------------------------------------------
# No hardcoded physical constants.  Both mult_trivial and mult_omega
# are named parameters, derived from theorem_BP + Ihara-Bass upstream.

def predict_epsilon_Koide(mult_trivial, mult_omega):
    """
    Koide amplitude ε from the C_3 multiplicity structure of the
    Ramanujan subspace of B(P), under postulate P2 (√-multiplicity
    coherent aggregation).

    Matching √m_j = √μ_triv + 2 √μ_ω · cos(2π j / 3) to the Koide form
    √m_j = √M · (1 + ε cos(2π j / 3)) gives

        √M = √μ_triv,   ε = 2 √μ_ω / √μ_triv,   ε² = 4 μ_ω / μ_triv.

    Parameters
    ----------
    mult_trivial : int
        C_3 trivial-irrep multiplicity on the Ramanujan subspace.
        Derived value on srs: 4.
    mult_omega : int
        C_3 ω-irrep multiplicity.  Derived value on srs: 2.

    Returns
    -------
    float
        ε = √(4 · mult_omega / mult_trivial).
    """
    return math.sqrt(4 * mult_omega / mult_trivial)


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("STATUS: BLOCKED under Theorem B6 retraction (2026-04-17)")
    print("See predictions/epsilon_Koide_derivation.md §Status")
    print("Re-derivation target: Sprint 11 workstream B7.4")
    print("(mass operator on C^3_gen; docs/master_plan.md §Sprint 11)")
    print("Math lemma eps^2 = 4 mu_omega / mu_triv = 2 is preserved as")
    print("a color-sector identity; generation reading is retracted")
    print("(B6 proves C_3 = color-Z_3, not generation).")
    print("=" * 60)
    impl_result = eps
    pure_result = predict_epsilon_Koide(mult_trivial, mult_omega)
    print(f"\nImplementation: {impl_result:.15f}")
    print(f"Pure function:  {pure_result:.15f}")
    print(f"√2:             {math.sqrt(2):.15f}")
    assert abs(impl_result - pure_result) < 1e-15, \
        f"Mismatch: {impl_result} vs {pure_result}"
    assert abs(pure_result - math.sqrt(2)) < 1e-15, \
        f"Pure function differs from √2: {pure_result}"
    print("OK: outputs agree.  epsilon_Koide = √2 exactly.")
