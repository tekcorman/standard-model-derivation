#!/usr/bin/env python3
"""
m_nu3 -- heaviest light neutrino mass (normal ordering), post-A3 Feshbach pattern.

STATUS UPDATE 2026-05-02 EOD+9: I-Feshbach is now DERIVED, not adopted.
The "NB walk survival = physical coupling" identification flows from
`docs/theorems/theorem_analytical_feshbach_ramanujan_boundary.md`
(2026-05-02 EOD+4): Σ(h) = α₁·h̄/|h|² is the closed-form Feshbach
self-energy from the substrate-complement spectral density on the
Ramanujan circle, evaluated at the saddle h. The a separate private derivation by the author water-filling
theorem (M_n = 0 for n ≥ 1 at MDL optimum) was confirmed by the
2026-05-02 EOD+5 m_ν2 PDG sensitivity sweep at -0.10σ; subleading
M_n contributions DEGRADE PDG match. -Im(Σ_lead)/α₁ = √5/4 IS the
dark coefficient used here. m_ν3 inherits the same correction factor
as m_ν2; the m_ν3 absolute value carries the larger PDG residual
(-4σ) which is driven entirely by the ADOPTED-PS bare scale, NOT by
the dark correction.

Remaining adopted residuals: ADOPTED-PS (bare neutrino scale from
Pati-Salam labeling) and ADOPTED-Z3 (C₃ Fourier index ↔ generation
label, since under B6 C_3 at P-point is color-Z₃, not generation).

STATUS UPDATE 2026-04-19 session 2 (HISTORICAL): References below to
"I-Feshbach (adopted)" pre-date the 2026-05-02 closure; treat them as
"derived via theorem_analytical_feshbach_ramanujan_boundary.md".

This file ships the rigorous core of the m_nu3 prediction under the
structural slate (A1 + A2-T + A3-T + local CAR thm + A5; docs/framework/framework_axioms.md §10)
plus explicitly flagged adopted residuals.  It follows the Feshbach pattern
established in predictions/feshbach_exponent_principle.py: the
theorem-grade mathematical content (shape factor, Feshbach correction,
splitting ratio) is derived; the external bare scale and three structural
identifications are adopted and labelled as such.

STRICT-SOLID content derived here:
  1. Shape factor  Im(h)/|h|^2 = (sqrt(5)/2) / 2 = sqrt(5)/4
     from h = (sqrt(3) + i sqrt(5))/2  [B_P_doubly_degenerate_h.py]
     with |h|^2 = 2  (Ramanujan saturation, k*-1 = 2).
  2. Class-1 Feshbach correction  1 + alpha_1_bare * sqrt(5)/4
     with alpha_1_bare = (2/3)^8  [feshbach_exponent_principle.py, n_fixed=2].
  3. Splitting ratio  R = 228/7  [R_nu_splitting.py].

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
        a separate private derivation by the author water-filling PDG verification on m_ν2).
"""

# ============================================================
# PARAMETER: m_nu3 (heaviest light neutrino mass, normal ordering)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       m_nu3 = sqrt(Dm2_31) = 50.13 meV   (assuming m_nu1 = 0)
#              Uncertainty +/- 0.20 meV from the 1-sigma band on Dm2_31.
# Source:      NuFIT 6.0 (September 2024), normal ordering:
#              Dm2_31 = (2.513 +/- 0.020) x 10^-3 eV^2.
#              PDG 2024 compatible (same NuFIT inputs).
#              m_nu1 = 0 is the framework prediction from
#              proofs/masses/srs_hashimoto_seesaw_verify.py
#              (M_D(trivial_s) = 0 at the P-point).
# PDG edition: 2024

# --- PREDICTED VALUE -----------------------------------------
# Value:       m_nu3 = m_nu3_bare * (1 + sqrt(5)/4 * alpha_1_bare)
#                    = 0.048277 * 1.02181197 eV
#                    = 49.33 meV
# Deviation:   (see __main__ validation block)
# Status:      THEOREM-GRADE-CONDITIONAL on ADOPTED-PS + ADOPTED-Z3 only
#              (I-Feshbach derived via theorem_analytical_feshbach_ramanujan_boundary.md
#              2026-05-02 EOD+9 promotion).
#
# Bridge convention (docs/framework/framework_scheme_convention.md §4.2 / §7):
# m_nu3 IS the canonical worked example of the convention's linear-
# amplitude Feshbach correction (Im(h)/|h|² · α₁_bare = √5/4 · α₁_bare,
# applied to m_nu3_bare). The (1 + √5/4 · α₁_bare) factor is the
# Feshbach self-energy from the substrate complement of the MDL
# projection (a separate private derivation by the author dark_correction_theorem_2026-04-14.md §4a contour
# integral on the Ramanujan circle). Convention-complete on the
# Feshbach side; residual lives in the bare seesaw scale (ADOPTED-PS).

# --- DERIVED FORMULA -----------------------------------------
# Full chain:
#
#   Step 1 (ADOPTED-PS): Bare seesaw scale [external numerical input]
#
#       m_nu3_bare = 0.048277 eV  [external]
#
#       Pati-Salam type-I seesaw at M_R = ((k*-1)/k*)^g * M_GUT:
#           m_nu3_bare = m_t(GUT)^2 / M_R
#       from proofs/masses/srs_nu_mass_ps.py (Part 3/4 output).
#       The girth-cycle identification M_R = (2/3)^10 * M_GUT and the
#       two-loop MSSM RG pipeline for m_t(GUT) are A-grade (not
#       theorem-grade under A1 + A2-T + A3-T).  Flagged ADOPTED-PS.
#
#   Step 2 (A1 + A2-T; Ihara 1966; Bass 1992; Terras 2011 Thm 3.1):
#       P-point NB walk eigenvalue [derived from upstream closed files]
#
#       h = (sqrt(3) + i sqrt(5)) / 2        [h_walker_eigenvalue.py]
#       |h|^2 = 3/4 + 5/4 = 2               (Ramanujan saturation: |h|^2 = k*-1)
#       Im(h) = sqrt(5)/2
#
#       Shape factor (exact rational-radical arithmetic):
#           Im(h) / |h|^2 = (sqrt(5)/2) / 2 = sqrt(5)/4
#
#   Step 3 (A1 + A2-T; Jaynes 1957; Terras 2011 §2.1; cited in
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
#   Step 4 (A1 + A2-T; exact algebra):
#       Class-1 Feshbach correction [derived from steps 2 + 3]
#
#           correction = 1 + alpha_1_bare * Im(h) / |h|^2
#                      = 1 + (2/3)^8 * sqrt(5)/4
#
#       Under Theorem A (../predictions/uniform_Q_density_derivation.md Part A),
#       rho_Q is uniform on the Ramanujan circle at MDL optimum.  The
#       Feshbach self-energy contour integral against the uniform measure
#       gives Sigma(h) = alpha_1_bare / h (residue at the pole h inside
#       the unit disk).  The multiplicative amplitude correction is
#       1 + |Im Sigma(h)|.
#
#   Step 5 (ADOPTED-Z3 + ADOPTED-PS; exact arithmetic):
#       Combined prediction
#
#           m_nu3 = m_nu3_bare * correction

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
# Source: proofs/masses/srs_nu_mass_ps.py Part 3/4 output.
# This is an [external] A-grade input; not derivable from A1 + A2-T + A3-T.
m_nu3_bare_eV = 0.048277   # [external] ADOPTED-PS

# --- Combined prediction (ADOPTED-PS + ADOPTED-Z3 only) ---
# (I-Feshbach removed 2026-05-02 EOD+9; now derived via theorem_analytical_feshbach_ramanujan_boundary.md)
m_nu3_eV = m_nu3_bare_eV * correction_factor

print("m_nu3 prediction -- Feshbach pattern (post-A3)")
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
print()
print(f"  Class-1 Feshbach correction [derived -- strict-solid]:")
print(f"    1 + sqrt(5)/4 * (2/3)^8"
      f" = 1 + {shape_factor_exact * alpha_1_bare:.10f}")
print(f"    correction_factor = {correction_factor:.10f}")
print()
print(f"  m_nu3 = m_nu3_bare * correction_factor")
print(f"        = {m_nu3_bare_eV:.6f} * {correction_factor:.10f}")
print(f"        = {m_nu3_eV:.6f} eV  =  {m_nu3_eV * 1e3:.4f} meV")


# --- PURE FUNCTION -------------------------------------------
# Every physical input is a named parameter.  Mathematical constants
# (sqrt, pi) appear only as they arise from the derivation chain.

@functools.lru_cache(maxsize=None)
def predict_m_nu3(m_nu3_bare_eV, alpha_1_bare, h_real, h_imag):
    """
    Predict m_nu3 from a bare Pati-Salam seesaw scale plus the Class-1
    Feshbach amplitude correction (Theorem A + Exponent Principle).

    Strict-solid content: the correction factor
        1 + alpha_1_bare * Im(h) / |h|^2
    is derived under A1 + A2-T + feshbach_exponent_principle.py (n_fixed=2)
    + B_P_doubly_degenerate_h.py (h = (sqrt(3) + i sqrt(5))/2).

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
    alpha_1_bare : float
        Feshbach coupling strength ((k*-1)/k*)^(g-2) on srs.  [derived]
    h_real : float
        Real part of P-point NB walk eigenvalue. On srs: sqrt(3)/2.
    h_imag : float
        Imaginary part of P-point NB walk eigenvalue. On srs: sqrt(5)/2.

    Returns
    -------
    float
        m_nu3 in eV.

    Formula
    -------
        h_abs_sq = h_real^2 + h_imag^2          (= 2 on srs, Ramanujan)
        shape    = h_imag / h_abs_sq             (= sqrt(5)/4 on srs)
        m_nu3    = m_nu3_bare_eV * (1 + alpha_1_bare * shape)
    """
    h_abs_sq = h_real ** 2 + h_imag ** 2
    shape = h_imag / h_abs_sq
    return m_nu3_bare_eV * (1.0 + alpha_1_bare * shape)


# --- VALIDATION ----------------------------------------------

m_nu3_pred = m_nu3_eV


if __name__ == "__main__":
    impl_result = m_nu3_eV
    pure_result = predict_m_nu3(
        m_nu3_bare_eV=m_nu3_bare_eV,
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
    print("  Bare scale m_nu3_bare:       ADOPTED-PS  [external, A-grade]")
    print("  Generation identification:   ADOPTED-Z3  [B6 caveat: C_3 at P is color-Z_3]")
    print("  NB survival = coupling:      DERIVED via Sigma(h) closure (was I-Feshbach)")
    print("  Overall:                     THEOREM-GRADE-CONDITIONAL on (ADOPTED-PS + ADOPTED-Z3)")
    print("=" * 60)

    # NuFIT 6.0 (September 2024), normal ordering:
    # Dm2_31 = (2.513 +/- 0.020) x 10^-3 eV^2 (task-spec observed values)
    dm2_31_obs = 2.513e-3
    dm2_31_sigma = 0.020e-3
    m_nu3_obs = math.sqrt(dm2_31_obs)           # = 50.13 meV
    m_nu3_sigma = 0.5 * dm2_31_sigma / m_nu3_obs   # propagated 1-sigma in m_nu3

    dev_abs = pure_result - m_nu3_obs
    dev_sigma = dev_abs / m_nu3_sigma

    print()
    print(f"  Implementation:  {impl_result:.9f} eV")
    print(f"  Pure function:   {pure_result:.9f} eV")
    print(f"  OK: outputs agree.")
    print(f"    m_nu3 (predicted) = {pure_result * 1e3:.4f} meV")
    print(f"    m_nu3 (NuFIT 6.0) = {m_nu3_obs * 1e3:.4f} +/- {m_nu3_sigma * 1e3:.4f} meV")
    print(f"    Deviation         = {dev_abs * 1e3:+.4f} meV  ({dev_sigma:+.2f} sigma)")
    print()
    print("  Note: the large deviation is driven entirely by the A-grade bare")
    print("  scale (ADOPTED-PS).  The Class-1 correction factor itself is")
    print("  theorem-grade.  Closing ADOPTED-PS requires a theorem-grade")
    print("  derivation of M_R and m_t(GUT) from A1 + A2-T + A3-T.")
