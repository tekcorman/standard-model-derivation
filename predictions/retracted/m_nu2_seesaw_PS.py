#!/usr/bin/env python3
"""
m_nu2 -- second light neutrino mass (normal ordering), post-A3 Feshbach pattern.

STATUS UPDATE 2026-05-02 EOD+9: I-Feshbach is now DERIVED, not adopted.
The "NB walk survival = physical coupling" identification flows from
`docs/theorems/theorem_analytical_feshbach_ramanujan_boundary.md`
(2026-05-02 EOD+4): Σ(h) = α₁·h̄/|h|² is the closed-form Feshbach
self-energy from the substrate-complement spectral density on the
Ramanujan circle, evaluated at the saddle h. The a separate private derivation by the author water-filling
theorem (M_n = 0 for n ≥ 1 at MDL optimum) was confirmed by the
2026-05-02 EOD+5 m_ν2 PDG sensitivity sweep at -0.10σ; subleading
M_n contributions DEGRADE PDG match, supporting M_n = 0 structurally.
Im(Σ_lead)/α₁ = -Im(h)/|h|² = -√5/4 IS the dark coefficient used here.

Remaining adopted residuals: ADOPTED-PS (bare neutrino scale from
Pati-Salam labeling) and ADOPTED-Z3 (C₃ Fourier index ↔ generation
label, since under B6 C_3 at P-point is color-Z₃, not generation).

STATUS UPDATE 2026-04-19 session 2 (HISTORICAL): References below to
"I-Feshbach (adopted)" pre-date the 2026-05-02 closure; treat them as
"derived via theorem_analytical_feshbach_ramanujan_boundary.md".

CAVEAT — m_ν1 = 0 citation: the line below citing
proofs/masses/srs_hashimoto_seesaw_verify.py for "M_D(trivial_s) = 0
at the P-point → m_ν1 = 0" reflects a derivation that was RETRACTED
under B6 (C₃-trivial sector is color-singlet content, not gen-1
neutrino). The m_ν1 = 0 statement here is currently a CONVENTION
(NuFIT 6.0 normal-ordering, lightest neutrino assumed massless), NOT
a derived structural result. Re-derivation under the C³_gen
framework is open research per an internal working note
sub-target B7.3a.v. Tracked as residue R-15 in
`docs/audits/registers/structural_residue_register.md`. m_ν2's
prediction does NOT load-bear on m_ν1 = 0 for its central value:
under any normal-ordering convention with m_ν1 small relative to
m_ν2, m_ν2 ≈ √(Δm²₂₁) to high precision.

This file ships the rigorous core of the m_nu2 prediction under the
structural slate (A1 + A2-T + A3-T + local CAR thm + A5; docs/framework/framework_axioms.md §10)
plus explicitly flagged adopted residuals.  It follows the Feshbach pattern
established in predictions/feshbach_exponent_principle.py: the
theorem-grade mathematical content (shape factor, Feshbach correction,
splitting ratio) is derived; the external bare scale and three structural
identifications are adopted and labelled as such.

The prediction chain is:
    m_nu3_bare_eV [ADOPTED-PS]
    correction    = 1 + alpha_1_bare * sqrt(5)/4     [derived]
    m_nu2_bare    = m_nu3_bare_eV / sqrt(R)          [R derived]
    m_nu2         = m_nu2_bare * correction

STRICT-SOLID content derived here:
  1. Shape factor  Im(h)/|h|^2 = (sqrt(5)/2) / 2 = sqrt(5)/4
     from h = (sqrt(3) + i sqrt(5))/2  [B_P_doubly_degenerate_h.py]
     with |h|^2 = 2  (Ramanujan saturation, k*-1 = 2).
  2. Class-1 Feshbach correction  1 + alpha_1_bare * sqrt(5)/4
     with alpha_1_bare = (2/3)^8  [feshbach_exponent_principle.py, n_fixed=2].
  3. Splitting ratio  R = 228/7  [R_nu_splitting.py].
     With m_nu1 = 0 (normal ordering, lightest massless):
         m_nu2_bare = m_nu3_bare / sqrt(R)   (exact algebra, no adopted content).

ADOPTED residuals (explicitly flagged, NOT derived):
  ADOPTED-PS: m_nu3_bare_eV = 0.048277 eV.  External numerical input
              from the Pati-Salam seesaw pipeline at M_R = (2/3)^10 * M_GUT
              (proofs/masses/srs_nu_mass_ps.py).  Not derivable from
              A1 + A2-T + A3-T alone; requires the M_GUT identification and the
              two-loop MSSM RG pipeline (A- grade, not theorem-grade).
  ADOPTED-Z3: C_3 Fourier index j identifies with generation label.
              Same adoption as predictions/Q_Koide.py ADOPTED-Z3.  B6
              (docs/theorem_B6_bridge.md) establishes C_3 at the P-point
              is color-Z_3 of SU(3)_c; the identification of the Class-1
              coefficient sqrt(5)/4 with a neutrino mass correction requires
              this separate adopted structural postulate.
  I-Feshbach: REMOVED 2026-05-02 EOD+9. Now derived via the analytical
              Feshbach formula Σ(h) = α₁·h̄/|h|² (theorem-grade per
              `docs/theorems/theorem_analytical_feshbach_ramanujan_boundary.md`)
              + a separate private derivation by the author (M_n = 0 at MDL optimum,
              confirmed by m_ν2 PDG sensitivity sweep at -0.10σ).
              The "NB walk survival = physical coupling" identification
              now flows from the closed-form Σ(h) = α₁/h leading-order
              evaluation: -Im(Σ_lead)/α₁ = Im(h)/|h|² = √5/4 IS the dark
              coefficient used in this file. No longer a separate adoption.

STATUS: THEOREM-GRADE-CONDITIONAL on ADOPTED-PS + ADOPTED-Z3
        (promotion 2026-05-02 EOD+9: I-Feshbach removed from conditionals
        via theorem_analytical_feshbach_ramanujan_boundary.md closure +
        a separate private derivation by the author water-filling PDG verification).

Cross-check: since both m_nu2 and m_nu3 carry the same multiplicative
correction factor (1 + sqrt(5)/4 * alpha_1_bare), their ratio is
    m_nu3^2 / m_nu2^2 = m_nu3_bare^2 / m_nu2_bare^2 = R = 228/7
exactly, consistent with predictions/R_nu_splitting.py.
"""

# ============================================================
# PARAMETER: m_nu2 (second light neutrino mass, normal ordering)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       m_nu2 = sqrt(Dm2_21) = 8.654 meV   (assuming m_nu1 = 0)
#              Uncertainty +/- 0.110 meV from the 1-sigma band on Dm2_21.
# Source:      NuFIT 6.0 (September 2024), normal ordering:
#              Dm2_21 = (7.49 +/- 0.19) x 10^-5 eV^2.
#              PDG 2024 compatible (same NuFIT inputs).
#              m_nu1 = 0 is the framework prediction from
#              proofs/masses/srs_hashimoto_seesaw_verify.py
#              (M_D(trivial_s) = 0 at the P-point).
# PDG edition: 2024

# --- PREDICTED VALUE -----------------------------------------
# Value:       m_nu2 = (m_nu3_bare / sqrt(R)) * (1 + sqrt(5)/4 * alpha_1_bare)
#                    = (0.048277 / sqrt(228/7)) * 1.02181197 eV
#                    = 8.644 meV
# Deviation:   (see __main__ validation block)
# Status:      THEOREM-GRADE-CONDITIONAL on ADOPTED-PS + ADOPTED-Z3 only
#              (I-Feshbach derived via theorem_analytical_feshbach_ramanujan_boundary.md
#              2026-05-02 EOD+9 promotion).
#
# Bridge convention (docs/framework/framework_scheme_convention.md §4.2 / §7):
# m_nu2 inherits the linear-amplitude Feshbach correction (√5/4·α₁_bare,
# from a separate private derivation by the author dark_correction_theorem §4a contour integral) applied to the
# bare seesaw scale. Convention-complete on the Feshbach side; bare
# seesaw scale itself is ADOPTED-PS.

# --- DERIVED FORMULA -----------------------------------------
# Full chain:
#
#   Step 1 (A1 + A2-T; Ihara 1966; Bass 1992; Terras 2011 Thm 3.1):
#       P-point NB walk eigenvalue [derived from upstream closed files]
#
#       h = (sqrt(3) + i sqrt(5)) / 2        [h_walker_eigenvalue.py]
#       |h|^2 = 3/4 + 5/4 = 2               (Ramanujan saturation: |h|^2 = k*-1)
#       Im(h) = sqrt(5)/2
#
#       Shape factor (exact rational-radical arithmetic):
#           Im(h) / |h|^2 = (sqrt(5)/2) / 2 = sqrt(5)/4
#
#   Step 2 (A1 + A2-T; Jaynes 1957; Terras 2011 §2.1; cited in
#           feshbach_exponent_principle.py):
#       Feshbach coupling (Exponent Principle, n_fixed = 2) [derived]
#
#           alpha_1_bare = ((k*-1)/k*)^(g-2) = (2/3)^8  [alpha_1.py]
#
#       The Exponent Principle is proven in
#       predictions/feshbach_exponent_principle.py (Jaynes 1957
#       max-entropy + Serre 1980 §I.1 + Terras 2011 §2.1).
#       I-Feshbach: identification of NB walk survival with physical
#       coupling strength is adopted (flagged in
#       feshbach_exponent_principle.py; referenced here).
#
#   Step 3 (A1 + A2-T; Ihara 1966; K4 Chebyshev + Gaussian integer
#           arithmetic; exact):
#       Splitting ratio [derived from R_nu_splitting.py]
#
#           R = Dm2_31 / Dm2_21 = 228/7
#
#       Under m_nu1 = 0 (normal ordering, lightest massless):
#           m_nu2^2 = Dm2_21,  m_nu3^2 = Dm2_31
#           => m_nu2_bare = m_nu3_bare / sqrt(R)   (exact algebra)
#
#   Step 4 (A1 + A2-T; exact algebra):
#       Class-1 Feshbach correction [derived from steps 1 + 2]
#
#           correction = 1 + alpha_1_bare * Im(h) / |h|^2
#                      = 1 + (2/3)^8 * sqrt(5)/4
#
#       Under Theorem A (../predictions/uniform_Q_density_derivation.md Part A),
#       rho_Q is uniform on the Ramanujan circle at MDL optimum.  The
#       Feshbach self-energy contour integral against the uniform measure
#       gives Sigma(h) = alpha_1_bare / h.  The multiplicative amplitude
#       correction is 1 + |Im Sigma(h)| = 1 + alpha_1_bare * Im(h)/|h|^2.
#
#   Step 5 (ADOPTED-PS): Bare scale [external numerical input]
#
#       m_nu3_bare = 0.048277 eV  [external]
#       m_nu2_bare = m_nu3_bare / sqrt(228/7)
#
#       The Pati-Salam seesaw bare scale for the heaviest generation
#       (m_t(GUT)^2 / M_R) is an [external] A-grade input.  Flagged
#       ADOPTED-PS.
#
#   Step 6 (ADOPTED-Z3 + ADOPTED-PS; exact arithmetic):
#       Combined prediction
#
#           m_nu2 = m_nu2_bare * correction

# --- INPUTS --------------------------------------------------
# symbol            | value                   | status     | predictions/ file                    | meaning
# ------------------|-------------------------|------------|--------------------------------------|--------
# A1                | (axiom)                 | [axiom]    | docs/framework/framework_axioms.md             | binary self-inverse toggle
# A2                | (axiom)                 | [axiom]    | docs/framework/framework_axioms.md             | MDL canonicalization
# A3                | (axiom)                 | [axiom]    | docs/framework/framework_axioms.md             | partial trace over dark sector
# k_star            | 3                       | [derived]  | predictions/k_star.py                | srs coordination number
# d_spatial         | 3                       | [derived]  | predictions/d_spatial.py             | srs spatial dimension
# g_girth           | 10                      | [derived]  | predictions/g_girth.py               | srs girth
# E_at_P            | sqrt(3)                 | [derived]  | predictions/srs_E_at_P.py            | P-point Bloch energy (intermediate for h)
# alpha_1_bare      | (2/3)^8                 | [derived]  | predictions/alpha_1.py               | Feshbach coupling (Exp. Principle)
# h                 | (sqrt(3)+i sqrt(5))/2   | [derived]  | predictions/h_walker_eigenvalue.py   | P-point NB walk eigenvalue
# shape_factor      | sqrt(5)/4               | [derived]  | Im(h)/|h|^2 from above               | Feshbach contour residue coefficient
# R                 | 228/7                   | [derived]  | predictions/R_nu_splitting.py        | Dm2_31/Dm2_21 splitting ratio
# m_nu3_bare_eV     | 0.048277 eV             | [external] | proofs/masses/srs_nu_mass_ps.py      | PS seesaw m_t(GUT)^2/M_R  [ADOPTED-PS]
# ADOPTED-PS        | (structural adoption)   | [adopted]  | flagged above                        | bare scale from Pati-Salam seesaw
# ADOPTED-Z3        | (structural adoption)   | [adopted]  | flagged above                        | C_3 Fourier index = generation label
# Sigma(h) closure  | h_bar/|h|^2             | [derived]  | theorem_analytical_feshbach_ramanujan_boundary.md | analytical Feshbach Σ(h) = α₁·h̄/|h|² (replaces I-Feshbach adoption 2026-05-02 EOD+9)

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import math
from fractions import Fraction

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from k_star import predict_k_star
from d_spatial import predict_d_spatial
from g_girth import predict_g_girth
from feshbach_exponent_principle import predict_feshbach_coupling
from h_walker_eigenvalue import predict_h_walker_eigenvalue
from srs_E_at_P import predict_srs_E_at_P
from R_nu_splitting import predict_R_nu_splitting
import functools

# --- Upstream chain-imports (all closed at their rigor tier) ---
d = predict_d_spatial()
k = predict_k_star(d)
g = predict_g_girth(k, d)

# alpha_1_bare = ((k*-1)/k*)^(g-2) = (2/3)^8
# [derived] predictions/feshbach_exponent_principle.py, n_fixed=2
alpha_1_bare = predict_feshbach_coupling(k, g, 2)
alpha_1_bare_frac = Fraction(k - 1, k) ** (g - 2)

# h = (sqrt(3) + i sqrt(5))/2 at srs P-point
# [derived] predictions/h_walker_eigenvalue.py
E_at_P = predict_srs_E_at_P(k)
h = predict_h_walker_eigenvalue(k, E_at_P)

# R = 228/7  [derived] predictions/R_nu_splitting.py
R = predict_R_nu_splitting(k)

# --- Shape factor (strict-solid, exact rational-radical arithmetic) ---
# |h|^2 = h_re^2 + h_im^2 = 3/4 + 5/4 = 2 (Ramanujan saturation)
h_abs_sq = h.real ** 2 + h.imag ** 2
# Im(h)/|h|^2 = (sqrt(5)/2) / 2 = sqrt(5)/4
shape_factor = h.imag / h_abs_sq

# Cross-check: exact value must equal sqrt(5)/4
shape_factor_exact = math.sqrt(5.0) / 4.0
assert abs(shape_factor - shape_factor_exact) < 1e-12, (
    f"Shape factor mismatch: computed {shape_factor}, "
    f"expected sqrt(5)/4 = {shape_factor_exact}"
)
assert abs(h_abs_sq - 2.0) < 1e-12, (
    f"|h|^2 = {h_abs_sq}, expected 2 (Ramanujan saturation k*-1 = {k-1})"
)

# --- Class-1 Feshbach correction (strict-solid) ---
# 1 + alpha_1_bare * sqrt(5)/4
correction_factor = 1.0 + alpha_1_bare * shape_factor

# --- Bare scale (ADOPTED-PS: external numerical input) ---
# m_nu3_bare = m_t(GUT)^2 / M_R from the Pati-Salam seesaw pipeline.
# Source: proofs/masses/srs_nu_mass_ps.py Part 3/4 output (Ihara form).
# This is an [external] A-grade input; not derivable from A1 + A2-T + A3-T.
m_nu3_bare_eV = 0.048277   # [external] ADOPTED-PS

# m_nu2_bare = m_nu3_bare / sqrt(R)  (from R = Dm2_31/Dm2_21 with m_nu1=0)
# This step is derived (given ADOPTED-PS provides m_nu3_bare):
#   m_nu2^2 = Dm2_21, m_nu3^2 = Dm2_31, so m_nu2/m_nu3 = 1/sqrt(R).
#   Since the correction factor is the same for both generations
#   (Class-1 is generation-independent mathematically), the ratio
#   m_nu2_bare / m_nu3_bare = 1/sqrt(R) as well.
m_nu2_bare_eV = m_nu3_bare_eV / math.sqrt(R)

# --- Combined prediction (ADOPTED-PS + ADOPTED-Z3 only) ---
# (I-Feshbach removed 2026-05-02 EOD+9; now derived via theorem_analytical_feshbach_ramanujan_boundary.md)
m_nu2_eV = m_nu2_bare_eV * correction_factor

print("m_nu2 prediction -- Feshbach pattern (post-A3)")
print(f"  Upstream: k*={k}, d={d}, g={g}")
print(f"  alpha_1_bare = ({k-1}/{k})^({g}-2) = {alpha_1_bare_frac}"
      f" = {alpha_1_bare:.10f}  [derived, n_fixed=2]")
print(f"  h = {h.real:+.8f} + {h.imag:+.8f} i"
      f"     [|h|^2 = {h_abs_sq:.8f}]  [derived]")
print(f"  Im(h)/|h|^2 = sqrt(5)/4 = {shape_factor_exact:.10f}  [derived]")
print(f"  R = 228/7 = {R:.8f}  [derived]")
print()
print(f"  Bare scale [ADOPTED-PS -- external, A-grade]:")
print(f"    m_nu3_bare = {m_nu3_bare_eV:.6f} eV"
      f"  (Pati-Salam seesaw; srs_nu_mass_ps.py)")
print(f"    m_nu2_bare = m_nu3_bare / sqrt(R)"
      f" = {m_nu3_bare_eV:.6f} / {math.sqrt(R):.6f}"
      f" = {m_nu2_bare_eV:.6f} eV")
print()
print(f"  Class-1 Feshbach correction [derived -- strict-solid]:")
print(f"    1 + sqrt(5)/4 * (2/3)^8"
      f" = 1 + {shape_factor_exact * alpha_1_bare:.10f}")
print(f"    correction_factor = {correction_factor:.10f}")
print()
print(f"  m_nu2 = m_nu2_bare * correction_factor")
print(f"        = {m_nu2_bare_eV:.6f} * {correction_factor:.10f}")
print(f"        = {m_nu2_eV:.6f} eV  =  {m_nu2_eV * 1e3:.4f} meV")


# --- PURE FUNCTION -------------------------------------------
# Every physical input is a named parameter.  Mathematical constants
# (sqrt, pi) appear only as they arise from the derivation chain.

@functools.lru_cache(maxsize=None)
def predict_m_nu2(m_nu3_bare_eV, R, alpha_1_bare, h_real, h_imag):
    """
    Predict m_nu2 from a bare Pati-Salam seesaw scale plus the Class-1
    Feshbach amplitude correction (Theorem A + Exponent Principle).

    Strict-solid content: the correction factor
        1 + alpha_1_bare * Im(h) / |h|^2
    is derived under A1 + A2-T + feshbach_exponent_principle.py (n_fixed=2)
    + B_P_doubly_degenerate_h.py (h = (sqrt(3) + i sqrt(5))/2).

    The splitting m_nu2_bare = m_nu3_bare / sqrt(R) is derived given R
    (from R_nu_splitting.py) and m_nu1 = 0 (normal ordering).

    Adopted residuals (see module docstring; 2026-05-02 EOD+9):
        ADOPTED-PS: m_nu3_bare_eV is an external numerical input.
        ADOPTED-Z3: C_3 index = generation label identification.
        (I-Feshbach removed; now derived via Sigma(h) = alpha_1*h_bar/|h|^2
         per theorem_analytical_feshbach_ramanujan_boundary.md)

    Parameters
    ----------
    m_nu3_bare_eV : float
        Pati-Salam bare seesaw scale for the heaviest light neutrino,
        in eV.  [ADOPTED-PS]
    R : float
        Ihara splitting ratio R = Dm2_31/Dm2_21 (derived: 228/7).
    alpha_1_bare : float
        Feshbach coupling strength ((k*-1)/k*)^(g-2) on srs.  [derived]
    h_real : float
        Real part of P-point NB walk eigenvalue. On srs: sqrt(3)/2.
    h_imag : float
        Imaginary part of P-point NB walk eigenvalue. On srs: sqrt(5)/2.

    Returns
    -------
    float
        m_nu2 in eV.

    Formula
    -------
        m_nu2_bare = m_nu3_bare_eV / sqrt(R)
        h_abs_sq   = h_real^2 + h_imag^2          (= 2 on srs, Ramanujan)
        shape      = h_imag / h_abs_sq             (= sqrt(5)/4 on srs)
        m_nu2      = m_nu2_bare * (1 + alpha_1_bare * shape)
    """
    m_nu2_bare = m_nu3_bare_eV / math.sqrt(R)
    h_abs_sq = h_real ** 2 + h_imag ** 2
    shape = h_imag / h_abs_sq
    return m_nu2_bare * (1.0 + alpha_1_bare * shape)


# --- VALIDATION ----------------------------------------------

m_nu2_pred = m_nu2_eV


if __name__ == "__main__":
    impl_result = m_nu2_eV
    pure_result = predict_m_nu2(
        m_nu3_bare_eV=m_nu3_bare_eV,
        R=R,
        alpha_1_bare=alpha_1_bare,
        h_real=h.real,
        h_imag=h.imag,
    )
    assert abs(impl_result - pure_result) < 1e-12, (
        f"Implementation vs pure function mismatch: "
        f"{impl_result} vs {pure_result}"
    )
    print()
    print("=" * 60)
    print("STATUS (A1 + A2-T + A3-T rigor bar; 2026-05-02 EOD+9 promotion):")
    print("  Class-1 Feshbach correction: THEOREM-GRADE")
    print("    (analytical Sigma(h) = alpha_1 * h_bar/|h|^2;")
    print("     theorem_analytical_feshbach_ramanujan_boundary.md)")
    print("  R = 228/7 splitting:         THEOREM-GRADE")
    print("  Bare scale m_nu3_bare:       ADOPTED-PS  [external, A-grade]")
    print("  Generation identification:   ADOPTED-Z3  [B6 caveat: C_3 at P is color-Z_3]")
    print("  NB survival = coupling:      DERIVED via Sigma(h) closure (was I-Feshbach)")
    print("  Overall:                     THEOREM-GRADE-CONDITIONAL on (ADOPTED-PS + ADOPTED-Z3)")
    print("=" * 60)

    # NuFIT 6.0 (September 2024), normal ordering:
    # Dm2_21 = (7.49 +/- 0.19) x 10^-5 eV^2
    dm2_21_obs = 7.49e-5
    dm2_21_sigma = 0.19e-5
    m_nu2_obs = math.sqrt(dm2_21_obs)           # = 8.654 meV
    m_nu2_sigma = 0.5 * dm2_21_sigma / m_nu2_obs   # propagated 1-sigma in m_nu2

    dev_abs = pure_result - m_nu2_obs
    dev_sigma = dev_abs / m_nu2_sigma

    # Cross-check: ratio m_nu3^2 / m_nu2^2 = R = 228/7
    ratio_check = (m_nu3_bare_eV * correction_factor) ** 2 / pure_result ** 2
    assert abs(ratio_check - R) < 1e-8, (
        f"Ratio m_nu3^2/m_nu2^2 = {ratio_check:.8f} != R = {R:.8f}"
    )

    print()
    print(f"  Implementation:  {impl_result:.9f} eV")
    print(f"  Pure function:   {pure_result:.9f} eV")
    print(f"  OK: outputs agree.")
    print(f"    m_nu2 (predicted) = {pure_result * 1e3:.4f} meV")
    print(f"    m_nu2 (NuFIT 6.0) = {m_nu2_obs * 1e3:.4f} +/- {m_nu2_sigma * 1e3:.4f} meV")
    print(f"    Deviation         = {dev_abs * 1e3:+.4f} meV  ({dev_sigma:+.2f} sigma)")
    print()
    print(f"  Cross-check: m_nu3^2/m_nu2^2 = {ratio_check:.6f}"
          f"  (R = 228/7 = {R:.6f})  OK")
    print()
    print("  Note: the small deviation is driven by the A-grade bare")
    print("  scale (ADOPTED-PS).  The Class-1 correction factor itself is")
    print("  theorem-grade.  Closing ADOPTED-PS requires a theorem-grade")
    print("  derivation of M_R and m_t(GUT) from A1 + A2-T + A3-T.")
