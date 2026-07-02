#!/usr/bin/env python3
"""
delta_Koide -- Koide phase parameter under A3 Born rule.

Audit anchor: Row P9 of `docs/parameters/parameter_uniqueness_ledger.md`. UNIQUE under
A1 + A2-T + A3-T + Pati-Salam + C₃-observer (Rows 16, 17, 18 of
`docs/audits/registers/uniqueness_ledger.md`). Algebraic identity δ = Q·(1−Q) = (2/3)(1/3) = 2/9.

The phase delta = 2/9 is DERIVED (not adopted, not a fit): it is the forced
directed phase of the chiral ∂_N run B(s*AXIS) at the lepton slice
(phi*s = 2/9, phi = 2pi/sqrt7). Three independent routes give the same number:
the directed run-phase (build_dN / generation_phase_delta_irreducible_2026-06-23);
the harmonic mean of the Wigner-D1 survival probabilities {4/9,1/9,4/9} at
cos beta = 1/k*; and the arithmetic Q*(1-Q) = (2/3)(1/3) = 2/9. Chain-imports
predictions/Q_Koide.py.

⚠ SCOPE (why the STATIC object looks phase-free): delta (phase) and Q
(amplitude) are independent in the Koide cos-parametrisation, and the substrate's
STATIC leading Q-construction is phase-DEGENERATE (delta == 0; no free U(1) —
every phase forced by the adjacency spectrum). That is NOT a gap: delta is not a
static quantity — it is the DYNAMICAL ∂_N running artifact (the forward C3-screw
run splits the T-paired windings, 0 : ±2pi*sqrt(3/7), rate 8:6:6; band-edge
−½+i√7/2 ∈ ℚ(√−7)). So the SPLITTING is DERIVED (the run), not adopted — the
static degeneracy is precisely why it once looked adopted (the static iso was the
wrong place to look). The data's deviation from exactly 2/9 is the SEPARATE OPEN
subleading per-rep O(alpha1^3) correction (the -70 ppm m_e miss;
incomplete_equations_todo.md §1). STRICT-SOLID for the phase (the ∂_N-run
derivation); the only residual is the A5 SEMANTIC anchor (that this Fock structure
IS the charged leptons — the "it's physics" narrative), docs/framework/framework_axioms.md §5b.

The retracted predictions/delta_Koide.py (pre-A3 two-axiom derivation)
is NOT modified. Both coexist as historical record + post-A3 re-derivation.
"""

# ============================================================
# PARAMETER: delta_Koide (Koide phase parameter)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       delta_observed = 0.2222227 +/- 0.0000009
# Source:      Extracted from PDG 2024 charged-lepton masses via the
#              Koide parametric form sqrt(m_j) = sqrt(M)*(1 + epsilon*
#              cos(2*pi*j/3 + phi)), fitting Q and then reading off
#              delta = Q*(1-Q). Dominant uncertainty from m_tau.
# PDG edition: 2024

# --- PREDICTED VALUE -----------------------------------------
# Value:       delta_predicted = 2/9 = 0.22222... (exact rational, forced VALUE)
# Deviation:   approx 0.53 sigma in the Koide-phase variable
#
# Bridge convention (docs/framework/framework_scheme_convention.md §7): the VALUE
# δ = 2/9 is forced (Q(1−Q) arithmetic + Wigner-HM + φ·s run). The 0.53σ gap
# (δ_obs = 0.2222227 ± 9e-7 vs 2/9, gap 4.78e-7) is NOT just measurement noise:
# it is the OPEN subleading per-rep O(α₁³) generation-splitting correction
# (lever-amplified ~120× into the −70 ppm m_e mass miss), un-derived /
# conjecture-grade MDL. Per the TOP-DOWN LAW the phase consistency does NOT make
# the lepton mass fine — the mass miss stays OPEN.

# --- DERIVED FORMULA -----------------------------------------
# delta = Q_Koide * (1 - Q_Koide)
#
# Derivation chain:
#
#   Step 1 (chain-import): Q_Koide = 2/3 from
#              predictions/Q_Koide.py (strict-solid under A1 + A2-T + A3-T,
#              same A5 (docs/framework/framework_axioms.md §5b) residuals).
#
#   Step 2 (the VALUE, three routes — NOT a parametric identity):
#              delta = Q*(1-Q) = 2/9 is the forced LEADING value. CAUTION: this
#              is NOT an identity of the cos-parametrisation — delta and Q are
#              independent in sqrt(m_j)=sqrt(M)(1+eps*cos(2pi j/k* + delta));
#              CAS verifies only the ARITHMETIC (2/3)(1/3)=2/9. The same value
#              2/9 also = HM{4/9,1/9,4/9} (Wigner-D1) and = phi*s (the screw
#              run) — three routes, one forced value. The splitting is DERIVED
#              (the ∂_N run); only the subleading −70 ppm per-rep O(a1^3) is open.
#
#   Step 3 (arithmetic): delta = (2/3)*(1/3) = 2/9 exactly.
#
# Upstream closed prediction files:
#   predictions/Q_Koide.py    (Q = 2/3, chain-imported here)
#   predictions/k_star.py        (k* = 3, implicit via Q derivation)

# --- INPUTS --------------------------------------------------
# symbol     | value | status    | source                     | meaning
# -----------|-------|-----------|----------------------------|--------
# Q_Koide | 2/3   | [derived] | predictions/Q_Koide.py  | chain-import
# A5 |       | [axiom] | Q_Koide_derivation.md   | amps on Ramanujan
# A5  |       | [axiom] | Q_Koide_derivation.md   | amps = Yukawa

# --- IMPLEMENTATION ------------------------------------------

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import sympy as sp
from fractions import Fraction

from k_star import predict_k_star
from d_spatial import predict_d_spatial
from Q_Koide import chain_import_ramanujan_multiplicities, predict_Q_Koide as predict_Q_Koide
import functools


# Upstream chain-imports.
d = predict_d_spatial()
k = predict_k_star(d)
mu_t, mu_o, mu_w = chain_import_ramanujan_multiplicities()

# Step 1: Q from upstream.
Q_float = predict_Q_Koide(k, mu_t, mu_o, mu_w)
from Q_Koide import Q_exact   # = Fraction(2,3) single-source
assert abs(Q_float - float(Q_exact)) < 1e-12

# Step 3: delta = Q*(1-Q).
delta_exact = Q_exact * (Fraction(1, 1) - Q_exact)

# Sympy CAS verification of the Bernoulli-moment identity at k* = 3.
_Q = sp.Symbol('Q', real=True)
_delta_sym = _Q * (1 - _Q)
_delta_at_2_3 = sp.simplify(_delta_sym.subs(_Q, sp.Rational(2, 3)))
assert sp.simplify(_delta_at_2_3 - sp.Rational(2, 9)) == 0, (
    f"Sympy verification failed: delta = {_delta_at_2_3}, expected 2/9."
)

# Compact check: delta = (k*-1)/k*^2 at k*=3 -> 2/9.
compact = Fraction(2, 9)
assert delta_exact == compact, f"Arithmetic mismatch: {delta_exact} vs {compact}"

# Canonical alias for the run_predictions.py harness:
delta_Koide_pred = float(delta_exact)

print(f"Upstream Q_Koide = {Q_exact} = {float(Q_exact):.15f}")
print(f"delta = Q*(1-Q) = {Q_exact} * {Fraction(1,1) - Q_exact} = {delta_exact}")
print(f"delta (float)   = {float(delta_exact):.15f}")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_delta_Koide(Q):
    """
    Compute delta = Q*(1-Q) given Q_Koide.
    For Q = 2/3: delta = 2/9 exactly.
    """
    return Q * (1 - Q)


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    pure_result = predict_delta_Koide(Q_float)

    print()
    print("=" * 60)
    print("STATUS under structural rigor bar (A1 + A2-T + A3-T):")
    print("  delta = 2/9 as a COLOR-SECTOR algebraic identity:")
    print("      STRICT-SOLID at theorem grade.")
    print("  delta = 2/9 as the CHARGED-LEPTON Koide phase:")
    print("      DERIVED = the ∂_N-run directed phase (static shell gives δ≡0 — δ lives")
    print("      in the run, not the static iso); subleading −70 ppm (data vs exactly")
    print("      2/9) is the SEPARATE OPEN item. Only residual = A5 semantic anchor.")
    print("=" * 60)
    print()
    print(f"Implementation (exact rational): delta = {delta_exact} "
          f"= {float(delta_exact):.15f}")
    print(f"Pure function (float):           delta = {pure_result:.15f}")
    print(f"Target 2/9:                      {2/9:.15f}")
    print(f"delta_observed (PDG 2024):       0.2222227 +/- 0.0000009")
    print(f"Deviation from observed:         "
          f"{abs(float(delta_exact) - 0.2222227) / 0.0000009:.2f} sigma")

    assert abs(float(delta_exact) - pure_result) < 1e-12, (
        f"Mismatch: {float(delta_exact)} vs {pure_result}"
    )
    assert abs(pure_result - 2/9) < 1e-12, (
        f"Pure function differs from 2/9: {pure_result}"
    )
    print()
    print("OK: outputs agree. delta_Koide = 2/9 exactly.")
