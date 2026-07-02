#!/usr/bin/env python3
"""
Canonical prediction file for alpha_21 (PMNS first Majorana phase).

alpha_21 = g * arg(h) mod 360°  with  h = (sqrt(3) + i*sqrt(5))/2  and  g = 10.

Full derivation: predictions/alpha_21_PMNS_derivation.md.

NOTE (post-A3, 2026-04-18): Historical pre-A3 two-axiom derivation,
BLOCKED under B6 (color-vs-generation retraction). Under the three-axiom
framework (A1+A2+A3; docs/framework_axioms.md) G.1 and G.5 are DERIVED via
CDP 2011 (predictions/observer_hilbert_space.py), but B6 retraction
remains load-bearing; Need-A2 (generation-Z_3 on C^3_gen) still open.
"""

# ============================================================
# PARAMETER: alpha_21 (first Majorana phase in the PMNS matrix)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       unconstrained
# Source:      The PMNS Majorana phases are not measured in oscillation
#              experiments; they enter only lepton-number-violating
#              observables (principally <m_beta_beta> in 0nu beta beta
#              decay).  Current bounds on <m_beta_beta>:
#                - KamLAND-Zen 2023 (Xe-136): <m_bb> < 36–156 meV (90% CL)
#                - GERDA-II 2020 (Ge-76):     <m_bb> < 79–180 meV (90% CL)
#              Global PMNS fits (NuFIT 5.3, 2024) do not constrain
#              alpha_21 independently.
# PDG edition: 2024 (status: unconstrained)

# --- PREDICTED VALUE -----------------------------------------
# Value:       g * arg(h) mod 360° = 10 * arctan(sqrt(5/3)) mod 360°
#                                  = 162.388° (to 6 s.f.)
# Deviation:   N/A (no current measurement)

# --- DERIVED FORMULA -----------------------------------------
# alpha_21 = g * arg(h) mod 360°
#          = g * arctan(sqrt(5/3)) mod 360°
#
# Logical chain (full proof: predictions/alpha_21_PMNS_derivation.md):
#
#   1. k* = 3, d = 3                              [predictions/k_star.py,
#                                                  predictions/d_spatial.py]
#   2. srs is the MDL-unique 3-regular 3D net,
#      I4_132 + Wyckoff 8a                        [predictions/g_girth_derivation.md §2;
#                                                  Sunada 2012, Notices AMS 59(2)]
#   3. Walker dynamics on srs = non-backtracking
#      walks; B is the Hashimoto operator         [docs/theorem_walker_dynamics.md
#                                                  Steps 1–7; closes W1–W3]
#   4. At the P-point B(P) has h = (sqrt(3)+i sqrt(5))/2
#      as a C_3-protected doubly-degenerate
#      eigenvalue; |h|^2 = k*-1 = 2 (Ramanujan)   [docs/theorem_BP_doubly_degenerate_h.md]
#   5. g = 10 is the girth of srs                 [predictions/g_girth.py;
#                                                  Sunada 2012, RCSR entry srs]
#   6. Adopted postulate P-phase-from-holonomy    [docs/W4_identification_catalog.md §2C, §3]
#      "Physical Majorana/Dirac phases are accumulated arguments
#       of h^n over specific closed walks on srs, with n set by
#       walk-topology invariants."
#      For alpha_21 (row 1 of catalog §2C) the assigned walk is
#      a full girth cycle, giving n = g = 10.
#      This is ADOPTED structure, not a theorem of MDL+toggle
#      (see catalog §4).  Every other step below is upstream-
#      closed, cited mathematics, or explicit arithmetic.
#   7. By de Moivre's theorem (Ahlfors 1978 §1.2.3):
#         arg(h^g) = g * arg(h)  (mod 2*pi).
#   8. arg(h) = arctan((sqrt(5)/2) / (sqrt(3)/2)) = arctan(sqrt(5/3))
#            = 52.23875609...° (sympy-verifiable).
#   9. 10 * 52.23875609...° = 522.38756...°; reducing mod 360°
#      gives alpha_21 = 162.38756...°.

# --- INPUTS --------------------------------------------------
# symbol | value          | status    | predictions/ file                   | meaning
# -------|----------------|-----------|-------------------------------------|--------
# h      | (sqrt(3)+i*sqrt(5))/2 | [derived] | predictions/h_walker_eigenvalue.py + docs/theorem_BP_doubly_degenerate_h.md | Hashimoto eigenvalue at P
# g      | 10             | [derived] | predictions/g_girth.py              | girth of srs
# P-phase-from-holonomy | — | [adopted] | docs/W4_identification_catalog.md §2C, §3 | Type-C identification postulate (incl. n = g assignment for alpha_21)

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import math
import cmath

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from d_spatial import predict_d_spatial
from k_star import predict_k_star
from srs_E_at_P import predict_srs_E_at_P
from h_walker_eigenvalue import predict_h_walker_eigenvalue
from g_girth import predict_g_girth

# Upstream derived values (all closed under their own predictions/ files).
d = predict_d_spatial()
k = predict_k_star(d)
E = predict_srs_E_at_P(k)
h = predict_h_walker_eigenvalue(k, E)
g = predict_g_girth(k, d)

# Sanity check: h matches the closed form (sqrt(3) + i*sqrt(5))/2 from
# docs/theorem_BP_doubly_degenerate_h.md.  This is a cross-check of the
# upstream closed-form value, not an input.
h_expected_re = math.sqrt(3) / 2
h_expected_im = math.sqrt(5) / 2
assert abs(h.real - h_expected_re) < 1e-14 and abs(h.imag - h_expected_im) < 1e-14, (
    f"Upstream h disagrees with theorem_BP closed form: {h} vs "
    f"({h_expected_re} + {h_expected_im}i)"
)
# And the Ramanujan identity |h|^2 = k*-1 = 2.
assert abs(abs(h) ** 2 - (k - 1)) < 1e-14, (
    f"Ramanujan saturation violated: |h|^2 = {abs(h) ** 2}, expected {k - 1}"
)

# Closed-form argument (principal branch, first quadrant):
#   arg(h) = arctan(Im/Re) = arctan(sqrt(5)/sqrt(3)) = arctan(sqrt(5/3)).
arg_h_rad = cmath.phase(h)  # principal value in (-pi, pi]
arg_h_closed_form = math.atan(math.sqrt(5.0 / 3.0))
assert abs(arg_h_rad - arg_h_closed_form) < 1e-14, (
    f"arg(h) != arctan(sqrt(5/3)): {arg_h_rad} vs {arg_h_closed_form}"
)

# De Moivre + mod 360°.
TWO_PI = 2.0 * math.pi
alpha_21_rad = (g * arg_h_rad) % TWO_PI
alpha_21_deg = math.degrees(alpha_21_rad)

print(f"k* = {k}, d = {d}, E(P) = sqrt(k*) = {E:.15f}")
print(f"h  = {h}  (= (sqrt(3) + i*sqrt(5))/2 from theorem_BP)")
print(f"|h|^2 = {abs(h) ** 2:.15f} = k*-1 = {k - 1}  (Ramanujan saturation)")
print(f"arg(h) = arctan(sqrt(5/3)) = {math.degrees(arg_h_rad):.10f}°")
print(f"g = {g}  (girth of srs, Sunada 2012)")
print(f"g * arg(h) = {math.degrees(g * arg_h_rad):.10f}°  (before mod 360°)")
print(f"alpha_21 = g * arg(h) mod 360° = {alpha_21_deg:.10f}°")


# --- PURE FUNCTION -------------------------------------------
# Contract: every physical quantity is a named parameter; the ONLY
# literals inside the body are mathematical constants (2, pi and 360).

def predict_alpha_21_PMNS(h_eigenvalue, g_girth):
    """
    Compute the first PMNS Majorana phase alpha_21 in degrees.

    Under the adopted Type-C postulate P-phase-from-holonomy
    (docs/W4_identification_catalog.md §2C), alpha_21 is the holonomy of
    the Hashimoto eigenvalue h around a full girth-cycle closed walk on
    srs:  alpha_21 = arg(h^g) mod 360°.  By de Moivre's theorem
    (Ahlfors 1978, §1.2.3), arg(h^g) = g * arg(h) (mod 2*pi), so

        alpha_21 = (g_girth * arg(h_eigenvalue)) mod 360°.

    Parameters
    ----------
    h_eigenvalue : complex
        Hashimoto (non-backtracking walk) eigenvalue at the P-point of
        the srs Brillouin zone.  Closed form (from
        docs/theorem_BP_doubly_degenerate_h.md) is (sqrt(3) + i*sqrt(5))/2,
        but the function does not assume this — it works for any complex h
        whose principal argument lies in (-pi, pi].
    g_girth : int
        Girth of the srs lattice (shortest NB cycle length).  Closed-form
        value from predictions/g_girth.py is 10.  No default; caller
        supplies it explicitly.

    Returns
    -------
    float
        alpha_21 reduced to the interval [0, 360) degrees.
    """
    # cmath.phase returns the principal argument in (-pi, pi].
    arg_h = cmath.phase(h_eigenvalue)
    return math.degrees((g_girth * arg_h) % (2 * math.pi))


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("STATUS: BLOCKED under Theorem B6 retraction (2026-04-17)")
    print("See predictions/alpha_21_PMNS_derivation.md §Status")
    print("Re-derivation target: Sprint 11 workstream B7.5")
    print("(PMNS under C^3_gen; docs/master_plan.md §Sprint 11)")
    print("P-phase sub-postulate 'walk class = gen 1 -> gen 2 transition'")
    print("requires C_3 = generation; B6 proves C_3 = color-Z_3.")
    print("Holonomy arithmetic g*arg(h) mod 360 is preserved as a math lemma.")
    print("=" * 60)
    impl_result = alpha_21_deg
    pure_result = predict_alpha_21_PMNS(h, g)
    print(f"\nImplementation: {impl_result:.10f}°")
    print(f"Pure function:  {pure_result:.10f}°")
    assert abs(impl_result - pure_result) < 1e-10, (
        f"Mismatch: {impl_result} vs {pure_result}"
    )
    # Sanity: the exact closed-form value is 10 * arctan(sqrt(5/3)) - 360°.
    alpha_21_closed = math.degrees(10 * math.atan(math.sqrt(5.0 / 3.0))) - 360.0
    assert abs(pure_result - alpha_21_closed) < 1e-10, (
        f"Pure function disagrees with closed form: "
        f"{pure_result} vs {alpha_21_closed}"
    )
    print("OK: outputs agree.")
    print(f"    alpha_21 = {pure_result:.4f}°  (experimentally unconstrained)")
