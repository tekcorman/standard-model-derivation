#!/usr/bin/env python3
"""
Canonical prediction file for alpha_31 (second Majorana phase in the PMNS
matrix).

alpha_31 = 2g * arg(h) mod 360°, where h = (sqrt(3) + i*sqrt(5))/2 is the
Bloch non-backtracking walk eigenvalue at the P-point of srs and g = 10 is
the girth of srs.  Every step of the chain is an upstream-closed theorem,
a cited mathematical identity, or an adopted structural postulate of the
framework (P-phase-from-holonomy, W4 Type C; see
docs/W4_identification_catalog.md §2C, §3).

NOTE (post-A3, 2026-04-18): Historical pre-A3 two-axiom derivation,
BLOCKED under B6 (color-vs-generation retraction). Under the three-axiom
framework (A1+A2+A3; docs/framework_axioms.md) G.1 and G.5 are DERIVED via
CDP 2011 (predictions/observer_hilbert_space.py), but B6 retraction
remains load-bearing; Need-A2 still open.
"""

# ============================================================
# PARAMETER: alpha_31 (second Majorana phase in PMNS matrix)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       unconstrained
# Source:      Majorana phases are not directly measured.  They enter
#              the amplitude of neutrinoless double-beta decay
#              (0νββ); current experiments (KamLAND-Zen, GERDA,
#              LEGEND, nEXO) bound |m_ββ| but do not isolate α_31.
# PDG edition: 2024 (PDG "Neutrino Masses, Mixing, and Oscillations"
#              review lists the Majorana phases as "undetermined").
#
# CAVEAT: This is a framework prediction with no experimental
# discriminator.  It is included for completeness of the PMNS
# parameter set, not because there is a measurement to match.

# --- PREDICTED VALUE -----------------------------------------
# Value:       2g · arg(h) mod 360°  ≈  324.78°
# Deviation:   N/A (unconstrained)

# --- DERIVED FORMULA -----------------------------------------
# alpha_31 = arg(h^{2g}) mod 360° = 2g · arg(h) mod 360°
#
# Derivation chain (full proof: predictions/alpha_31_PMNS_derivation.md):
#
#   1. h = (sqrt(3) + i*sqrt(5))/2  — B(P) eigenvalue with multiplicity 2,
#      C_3-protected.  [docs/theorem_BP_doubly_degenerate_h.md, theorem]
#   2. g = 10  — srs girth.  [predictions/g_girth.py, theorem]
#   3. P-phase-from-holonomy (W4 Type C, adopted postulate):
#        A physical phase observable is the accumulated argument of h^n
#        (or h*^n) along a specific closed walk on srs, with n fixed by
#        a walk-topology invariant of the observable.
#      [docs/W4_identification_catalog.md §2C, §3]
#   4. Sub-postulate (within P-phase, specific to α_31):
#        The walk class for the inter-generation Majorana phase α_31
#        (the generation-1 to generation-3 transition, skipping
#        generation 2) is two girth cycles, so n = 2g.
#        α_21 uses n = g (one girth cycle);  α_31 uses n = 2g (two
#        girth cycles).
#   5. De Moivre's theorem (standard complex analysis; e.g. Ahlfors
#      *Complex Analysis* 3rd ed. §1.2):
#        arg(h^{2g}) = 2g · arg(h)  (mod 2π).
#   6. arg(h) = arctan(sqrt(5)/sqrt(3)) = arctan(sqrt(5/3))
#             ≈ 0.91167 rad ≈ 52.23876°.
#   7. 2g · arg(h) = 20 · 52.23876° = 1044.7752°;
#      reduce mod 360°:  1044.7752° − 2·360° = 324.7752°.
#
# The only freely postulated content is step 3 (P-phase-from-holonomy,
# catalog-documented as adopted) and the n = 2g sub-choice in step 4,
# which is flagged as an identification-layer postulate, not a theorem.

# --- INPUTS --------------------------------------------------
# symbol     | value          | status     | predictions/ file                        | meaning
# -----------|----------------|------------|-------------------------------------------|--------
# h          | (√3+i√5)/2     | [derived]  | predictions/h_walker_eigenvalue.py       | Hashimoto eigenvalue
# g          | 10             | [derived]  | predictions/g_girth.py                   | girth of srs
# P-phase    | —              | [adopted]  | docs/W4_identification_catalog.md §2C,§3 | Type-C phase postulate
# n = 2g     | 20             | [adopted]  | docs/W4_identification_catalog.md §2C    | sub-postulate: gen-1→gen-3 transition

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import math
import cmath
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h_walker_eigenvalue import predict_h_walker_eigenvalue
from k_star import predict_k_star
from d_spatial import predict_d_spatial
from srs_E_at_P import predict_srs_E_at_P
from g_girth import predict_g_girth

# Upstream values.
d = predict_d_spatial()
k = predict_k_star(d)
E = predict_srs_E_at_P(k)
h = predict_h_walker_eigenvalue(k, E)
g = predict_g_girth(k, d)

# n = 2g: two girth cycles (W4 §2C sub-postulate for the
# generation-1 to generation-3 inter-generation transition).
n = 2 * g

# De Moivre: arg(h^n) = n · arg(h) mod 2π.
arg_h = cmath.phase(h)
alpha_31_rad = (n * arg_h) % (2 * math.pi)
alpha_31_deg = math.degrees(alpha_31_rad)

print(f"h = {h}")
print(f"|h|^2 = {abs(h)**2:.6f}  (Ramanujan bound: k*-1 = {k-1})")
print(f"arg(h) = {math.degrees(arg_h):.6f}°")
print(f"g = {g}  (srs girth)")
print(f"n = 2g = {n}  (W4 §2C: inter-generation 1→3 walk class)")
print(f"n · arg(h) = {math.degrees(n * arg_h):.6f}° (raw, pre-reduction)")
print(f"alpha_31 = {alpha_31_deg:.6f}° (mod 360°)")


# --- PURE FUNCTION -------------------------------------------
# No hardcoded physical constants.  Both h (a complex number) and the
# walk length n = 2g are named parameters.  The only literals in the
# body are pi (2π for modular reduction) and 180/pi (degree conversion,
# handled by math.degrees).

def predict_alpha_31_PMNS(h, g_girth):
    """
    Compute the PMNS second Majorana phase α_31 under the W4-catalog
    Type-C postulate (P-phase-from-holonomy) and the sub-postulate
    that the generation-1 to generation-3 inter-generation transition
    traverses n = 2·g_girth edges on srs.

    By de Moivre's theorem,
        α_31 = arg(h^{2g}) mod 2π = 2·g · arg(h) mod 2π.

    Parameters
    ----------
    h : complex
        Bloch non-backtracking walk eigenvalue at the P-point of srs.
        The framework's canonical value is (√3 + i√5)/2, derived in
        docs/theorem_BP_doubly_degenerate_h.md.
    g_girth : int
        Girth of srs.  The framework's derived value is 10, from
        predictions/g_girth.py.

    Returns
    -------
    float
        α_31 in degrees, reduced to [0, 360).
    """
    arg_h = cmath.phase(h)
    return math.degrees((2 * g_girth * arg_h) % (2 * math.pi))


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("STATUS: BLOCKED under Theorem B6 retraction (2026-04-17)")
    print("See predictions/alpha_31_PMNS_derivation.md §Status")
    print("Re-derivation target: Sprint 11 workstream B7.5")
    print("(PMNS under C^3_gen; docs/master_plan.md §Sprint 11)")
    print("P-phase sub-postulate 'walk class = gen 1 -> gen 3 transition,")
    print("n = 2g' requires C_3 = generation; B6 proves C_3 = color-Z_3.")
    print("Holonomy arithmetic 2g*arg(h) mod 360 is preserved as a math lemma.")
    print("=" * 60)
    impl_result = alpha_31_deg
    pure_result = predict_alpha_31_PMNS(h, g)
    print(f"\nImplementation: {impl_result:.6f}°")
    print(f"Pure function:  {pure_result:.6f}°")
    assert abs(impl_result - pure_result) < 1e-10, \
        f"Mismatch: {impl_result} vs {pure_result}"
    print("OK: outputs agree.")
    print(f"    alpha_31 = {pure_result:.2f}° (obs: unconstrained)")
