#!/usr/bin/env python3
"""
Canonical prediction file for m_nu2 (second-generation light neutrino mass).

NOTE (post-A3, 2026-04-18): Historical pre-A3 two-axiom derivation,
BLOCKED under B6 (color-vs-generation retraction). Under the three-axiom
framework (A1+A2+A3; docs/framework_axioms.md) G.1 and G.5 are DERIVED via
CDP 2011 (predictions/observer_hilbert_space.py); B6 retraction and
Feshbach/uniform-Q-density formalism remain separately load-bearing.

Type-D Class-1 observable (amplitude, coefficient sqrt(5)/4) per
docs/W4_identification_catalog.md §2D:

    m_nu2 = m_nu2_bare * (1 + sqrt(5)/4 * alpha_1_bare)

where m_nu2_bare is the Pati-Salam seesaw value at the second-generation
slot (propagated via the Ihara splitting ratio R = 228/7 from m_nu3),
and the Class-1 multiplicative correction follows from

    Theorem A  (docs/theorem_uniform_Q_density.md, Part A)
        rho_Q(phi) is uniform on the Ramanujan circle at MDL optimum,
        with TV-distance remainder O(sqrt(log N / N)).

    Exponent Principle + Lemma 1  (docs/theorem_Feshbach_coupling_strength.md)
        alpha_1_bare = ((k*-1)/k*)**(g-2) = (2/3)**8 on srs.

    Feshbach contour integral
        Under Theorem A, Sigma(h) = alpha_1_bare / h (closed-form
        residue evaluation at the pole h inside the unit disk).
        |Im Sigma(h)| = alpha_1_bare * Im(h) / |h|**2
                      = alpha_1_bare * (sqrt(5)/2) / 2
                      = alpha_1_bare * sqrt(5)/4
        using h = (sqrt(3) + i sqrt(5))/2 from
        predictions/h_walker_eigenvalue.py.
"""

# ============================================================
# PARAMETER: m_nu2 (second light neutrino mass, normal ordering)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       m_nu2 = sqrt(Dm2_21) = 0.008654 eV   (assuming m_nu1 = 0)
#              Uncertainty +/- 0.000110 eV from the 1-sigma band on Dm2_21.
# Source:      NuFIT 6.0 (September 2024), normal ordering:
#              Dm2_21 = (7.49 +/- 0.19) x 10**-5 eV**2.
#              PDG 2024 is compatible (same NuFIT inputs).
#              m_nu1 = 0 is the framework prediction from
#              proofs/masses/srs_hashimoto_seesaw_verify.py
#              (M_D(trivial_s) = 0 at the P-point; m_nu1 = 0 to the
#              precision of the Pati-Salam seesaw on srs).
# PDG edition: 2024
#
# Under m_nu1 = 0 (normal ordering, lightest neutrino massless), the
# solar mass-squared difference Dm2_21 = m_nu2**2 - m_nu1**2 = m_nu2**2,
# so sqrt(Dm2_21) IS m_nu2 up to the small m_nu1 correction.

# --- PREDICTED VALUE -----------------------------------------
# Value:       m_nu2 = 0.008707 eV   (bare 0.008521 eV times 1.02181)
# Deviation:   +0.000053 eV, ~0.48 sigma (within 1-sigma of NuFIT 6.0).
# Status:      Class-1 amplitude correction is now theorem-grade
#              (Theorem A + Exponent Principle, 2026-04-15 upgrade).
#              BLOCKED at the bare scale: m_nu2_bare depends on
#              M_R = (2/3)**g * M_GUT (girth-cycle identification,
#              A- grade, not theorem) and on m_c(GUT) from two-loop
#              MSSM RG running with tan(beta) = 44.73 (A- grade).
#              See "Open questions" in m_nu2_derivation.md.

# --- DERIVED FORMULA -----------------------------------------
# Full chain:
#
#   1. Bare seesaw (Pati-Salam, M_D(nu) = M_u^T from Cl(6) SU(4)_PS):
#        m_nu3_bare = m_t(GUT)**2 / M_R                        [PS]
#        m_nu2_bare = m_nu3_bare / sqrt(R)                     [R-split]
#      with M_R = ((k*-1)/k*)**g * M_GUT = (2/3)**10 * M_GUT.
#      Equivalent form: m_nu2_bare = m_c(GUT)**2 / M_R using the
#      diagonal seesaw in the mass eigenbasis.
#
#   2. Ihara splitting ratio (theorem):
#        R = Dm2_31/Dm2_21 = 228/7                             [R-thm]
#      from predictions/R_nu_splitting.py.  With m_nu1 = 0 this
#      reduces to m_nu2_bare = m_nu3_bare / sqrt(R).
#
#   3. Class-1 dark correction (theorem-grade, Sprint 4 upgrade):
#        m_nu2 = m_nu2_bare * (1 + sqrt(5)/4 * alpha_1_bare)   [Class 1]
#      derivation:
#        - Theorem A:   rho_Q uniform on Ramanujan circle (MDL).
#        - Exp. Princ.: alpha_1_bare = (2/3)**(g-2).
#        - Feshbach:    Sigma(h) = alpha_1_bare / h under Theorem A
#                       (contour integral of 1/(h - sqrt(k-1) e**i*phi)
#                       against uniform measure gives 1/h inside the
#                       disk).  Multiplicative amplitude correction
#                       factor is 1 + |Im Sigma(h)| = 1 + alpha_1_bare
#                       * Im(h)/|h|**2.  At h = (sqrt(3)+i sqrt(5))/2,
#                       |h|**2 = 2 and Im(h) = sqrt(5)/2, giving
#                       Im(h)/|h|**2 = sqrt(5)/4.

# --- INPUTS --------------------------------------------------
# symbol        | value            | status     | predictions/ file                    | meaning
# --------------|------------------|------------|--------------------------------------|--------
# k_star        | 3                | [derived]  | predictions/k_star.py                | coordination number
# g_girth       | 10               | [derived]  | predictions/g_girth.py               | srs girth
# alpha_1_bare  | (2/3)**8         | [theorem*] | predictions/alpha_1.py               | Feshbach coupling (Exp. Principle)
# h             | (sqrt(3)+i sqrt(5))/2 | [derived] | predictions/h_walker_eigenvalue.py | P-point NB walk eigenvalue
# R             | 228/7            | [theorem]  | predictions/R_nu_splitting.py        | Dm2_31/Dm2_21 ratio
# m_nu3_bare_eV | 0.04828 eV       | [external] | proofs/masses/srs_nu_mass_ps.py      | PS seesaw m_t(GUT)**2/M_R (A-)
#
# *alpha_1_bare is theorem-grade modulo the Exponent Principle
#  (adopted structural theorem, see theorem_Feshbach_coupling_strength.md §3);
#  Lemma 1 (tree NB survival) is fully theorem-grade.
#
# m_nu3_bare is taken as an external numerical input from the PS seesaw
# pipeline; its construction uses observed M_Z gauge couplings and the
# pole top-quark mass, and is not theorem-grade under the present
# rigor bar.  Its upstream status is A- (see .md §Open questions).

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import math
from fractions import Fraction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k_star import predict_k_star
from d_spatial import predict_d_spatial
from g_girth import predict_g_girth
from alpha_1 import predict_alpha_1
from h_walker_eigenvalue import predict_h_walker_eigenvalue
from srs_E_at_P import predict_srs_E_at_P
from R_nu_splitting import predict_R_nu_splitting

# Upstream (all closed at their own rigor tier):
d = predict_d_spatial()
k = predict_k_star(d)
g = predict_g_girth(k, d)
alpha_1_bare = predict_alpha_1(k, g)               # (2/3)**8
E_at_P = predict_srs_E_at_P(k)                      # sqrt(3)
h = predict_h_walker_eigenvalue(k, E_at_P)          # (sqrt(3)+i sqrt(5))/2
R = predict_R_nu_splitting(k)                       # 228/7

# --- Bare scale (A-grade external numerical input) -----------
# m_nu3_bare from the Pati-Salam seesaw at M_R = (2/3)**10 * M_GUT
# with m_t(GUT) from the two-loop MSSM RG pipeline
# (proofs/masses/srs_nu_mass_ps.py).  The Ihara ratio R = 228/7
# then fixes m_nu2_bare = m_nu3_bare / sqrt(R) exactly.
#
# This script does NOT re-run the RG pipeline; it imports the bare
# value as a documented numerical input so the Class-1 correction can
# be isolated and audited.  See m_nu2_derivation.md §3 and
# srs_nu_mass_ps.py for the full bare derivation.
m_nu3_bare_eV = 0.048277   # documented output of srs_nu_mass_ps.py part4_ihara

m_nu2_bare_eV = m_nu3_bare_eV / math.sqrt(float(R))

# --- Class-1 amplitude correction (theorem-grade) ------------
# |h|**2 = k*-1 = 2 (Ramanujan saturation, h_walker_eigenvalue.py)
# Im(h) = sqrt(k*-1 - (k*-1)**2/k*) ... no, simpler: Im(h) = sqrt(5)/2.
# The closed form from Theorem A + Exponent Principle:
#   correction_factor = 1 + alpha_1_bare * Im(h) / |h|**2
#                     = 1 + alpha_1_bare * sqrt(5)/4      (at k*=3)

h_abs_sq = h.real**2 + h.imag**2                   # = 2 exactly
shape_factor = h.imag / h_abs_sq                   # Im(h)/|h|**2 = sqrt(5)/4
correction_factor = 1.0 + alpha_1_bare * shape_factor

m_nu2_eV = m_nu2_bare_eV * correction_factor

# --- Symbolic cross-check -----------------------------------
# Class-1 coefficient sqrt(5)/4 (exact):
shape_factor_exact = math.sqrt(5.0) / 4.0
assert abs(shape_factor - shape_factor_exact) < 1e-12, \
    f"shape factor mismatch: {shape_factor} vs sqrt(5)/4 = {shape_factor_exact}"

# Exact Class-1 correction using rational alpha_1_bare:
alpha_1_bare_frac = Fraction(k - 1, k) ** (g - 2)
# correction_factor = 1 + sqrt(5)/4 * (2/3)**8 (symbolic mixed form)

print("m_nu2 canonical prediction")
print(f"  Upstream: k*={k}, g={g}")
print(f"  alpha_1_bare = ({k-1}/{k})**({g}-2) = {alpha_1_bare_frac} "
      f"= {alpha_1_bare:.10f}")
print(f"  h            = ({h.real:+.6f}) + ({h.imag:+.6f}) i "
      f"     [|h|**2 = {h_abs_sq:.6f}]")
print(f"  Im(h)/|h|**2 = sqrt(5)/4 = {shape_factor_exact:.10f}")
print(f"  R            = 228/7 = {float(R):.6f}")
print()
print(f"  Bare scale (external input from srs_nu_mass_ps.py):")
print(f"    m_nu3_bare = {m_nu3_bare_eV:.6f} eV")
print(f"    m_nu2_bare = m_nu3_bare / sqrt(R) = {m_nu2_bare_eV:.6f} eV")
print()
print(f"  Class-1 correction (Theorem A + Exponent Principle):")
print(f"    factor = 1 + sqrt(5)/4 * alpha_1_bare "
      f"= 1 + {shape_factor_exact*alpha_1_bare:.8f}")
print(f"           = {correction_factor:.8f}")
print()
print(f"  m_nu2 = m_nu2_bare * correction_factor = {m_nu2_eV:.6f} eV")


# --- PURE FUNCTION -------------------------------------------
# Every physical quantity is a named parameter.  The only literals in
# the body are pure-mathematical (sqrt, arithmetic).

def predict_m_nu2(m_nu3_bare_eV, R, alpha_1_bare, h_real, h_imag):
    """
    Predict m_nu2 from a bare Pati-Salam seesaw scale plus the Class-1
    Feshbach amplitude correction (Theorem A + Exponent Principle).

    Parameters
    ----------
    m_nu3_bare_eV : float
        Pati-Salam bare seesaw scale for the heaviest light neutrino
        (m_t(GUT)**2 / M_R), in eV.
    R : float
        Ihara splitting ratio R = Dm2_31/Dm2_21 (theorem: 228/7).
    alpha_1_bare : float
        Feshbach coupling strength, ((k*-1)/k*)**(g-2) on srs.
    h_real, h_imag : float
        Real and imaginary parts of the P-point NB walk eigenvalue.
        On srs: (sqrt(3)/2, sqrt(5)/2).

    Returns
    -------
    float
        m_nu2 in eV.

    Formula
    -------
        m_nu2_bare = m_nu3_bare_eV / sqrt(R)
        shape      = h_imag / (h_real**2 + h_imag**2)       # sqrt(5)/4 on srs
        m_nu2      = m_nu2_bare * (1 + alpha_1_bare * shape)
    """
    m_nu2_bare = m_nu3_bare_eV / math.sqrt(R)
    h_abs_sq = h_real * h_real + h_imag * h_imag
    shape = h_imag / h_abs_sq
    return m_nu2_bare * (1.0 + alpha_1_bare * shape)


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("STATUS: BLOCKED under Theorem B6 retraction (2026-04-17)")
    print("See predictions/m_nu2_derivation.md §Status")
    print("Re-derivation target: Sprint 11 workstream B7.3")
    print("(mass operator on C^3_gen; docs/master_plan.md §Sprint 11)")
    print("Bare scale uses m_nu2_bare = m_nu3_bare/sqrt(R), indexed by a")
    print("generation slot; Class-1 coefficient identifies with gen-2")
    print("neutrino correction. B6 proves C_3 = color-Z_3, not generation.")
    print("Color-sector lemma sqrt(5)/4 * (2/3)^8 is preserved as math.")
    print("=" * 60)
    impl_result = m_nu2_eV
    pure_result = predict_m_nu2(
        m_nu3_bare_eV=m_nu3_bare_eV,
        R=float(R),
        alpha_1_bare=alpha_1_bare,
        h_real=h.real,
        h_imag=h.imag,
    )
    obs = 8.654e-3                         # eV, sqrt(7.49e-5)
    sigma = 0.5 * 0.19e-5 / obs            # propagated uncertainty in m_nu2
    dev_abs = pure_result - obs
    dev_sigma = dev_abs / sigma
    print()
    print(f"Implementation: {impl_result:.9f} eV")
    print(f"Pure function:  {pure_result:.9f} eV")
    assert abs(impl_result - pure_result) < 1e-12, \
        f"Mismatch: {impl_result} vs {pure_result}"
    print("OK: outputs agree.")
    print(f"    m_nu2 (predicted) = {pure_result*1e3:.4f} meV")
    print(f"    m_nu2 (NuFIT 6.0) = {obs*1e3:.4f} +/- {sigma*1e3:.4f} meV")
    print(f"    Deviation         = {dev_abs*1e3:+.4f} meV "
          f"({dev_sigma:+.2f} sigma)")
    print("    Class-1 coefficient sqrt(5)/4 * alpha_1_bare: theorem-grade")
    print("    (Theorem A uniform rho_Q + Exponent Principle + Feshbach).")
    print("    Bare scale m_nu2_bare: A- (depends on M_GUT, m_t(GUT); see .md).")
