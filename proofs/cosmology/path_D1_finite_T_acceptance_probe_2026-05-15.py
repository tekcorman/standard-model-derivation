#!/usr/bin/env python3
"""
Path D probe (D.1) — MDL acceptance rate at finite T (2026-05-15 EOD+1).

HYPOTHESIS
----------
The cascade theorem's "MDL acceptance probability = 1/k*" (D2 step) holds
exactly at T = 0. At finite T, the Stage-2c noise floor ε_T might suppress
acceptance below 1/k*, modifying H(z) from coasting.

If suppression is significant in the recombination-to-T_srs range (~3000 K
to ~10³³ K), the framework's H(z) profile differs from coasting at high z
and could mimic ΛCDM radiation-domination (H ∝ (1+z)²).

This probe checks whether the natural Boltzmann/Landauer-form suppression
gives an appreciable effect at observationally-relevant redshifts.

APPROACH
--------
1. Compute Stage-2c noise floor ε_T = k_B T ln(2) (Planck units).
2. Compare to srs MDL compression saving per vertex C_srs ≈ 87.7 bits.
3. Compute suppression factor p_accept(T)/p_accept(0) at multiple z.
4. Derive resulting H(z) modification.
5. Check whether H(z) has p > 1 in some observationally-relevant range.

PRE-COMPUTATION EXPECTATION
---------------------------
Thermal noise at recombination (T ~ 3000 K = ~10⁻²⁹ Planck units) is
~30 orders of magnitude BELOW C_srs ~ 88 Planck units.  So the
natural Boltzmann-form suppression is negligible at recombination.

For the suppression to be ~10% (giving H(z) deviation from coasting of
~10%, near radiation-domination strength), we need T ~ C_srs × 0.1, i.e.,
T ~ 9 Planck units = 9 × 1.4 × 10³² K = 1.3 × 10³³ K.  At T_CMB(z) ~
T_CMB(0) × (1+z), this means z ~ 5 × 10³² — way above any
observationally-relevant epoch.

So the probe is expected NEGATIVE: framework-natural Boltzmann suppression
doesn't reach observationally-relevant z range.  Confirming this
quantitatively is the value of this probe.
"""

from __future__ import annotations
import math
import sys
import os
import numpy as np

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ============================================================================
# Framework primitives (theorem-grade upstream)
# ============================================================================
K_STAR = 3                       # Row 4 theorem-grade
G_GIRTH = 10                     # Row 9 / Row P50 theorem-grade

# Stage-2c noise floor: per-bit Landauer floor at temperature T is k_B T ln 2.
# In Planck units (k_B = 1, T in Planck temperature), ε_T = T · ln(2).
# Actually for Landauer's bound, the energy cost of erasing one bit at
# temperature T is k_B T ln 2.  In Planck units (k_B = 1, ℏ = c = 1), the
# energy is T · ln 2 per bit.

LN2 = math.log(2)

# srs MDL compression saving per vertex (per `early_universe_k_rundown.py`)
# C_srs = n_g · log₂(1/α_1) bits where α_1 = (2/3)^(g-2) = (2/3)^8 and n_g = 15
ALPHA_1 = (2.0 / 3.0) ** (G_GIRTH - 2)
N_G_PER_VERTEX = 15  # Sunada cycle count on srs primitive cell (DFS verified)
C_SRS_BITS = N_G_PER_VERTEX * math.log2(1.0 / ALPHA_1)  # ≈ 87.74 bits/vertex

# Substrate-stability temperature (where C_srs is comparable to ε_T per bit)
# At T_srs, ε_T · ln(2) = C_srs · k_B ⇒ T_srs (Planck units) ≈ C_srs / ln(2)
# Per `early_universe_k_rundown.py`, this is order Planck scale.
T_SRS_PLANCK = C_SRS_BITS / LN2  # ≈ 126.6 Planck units

# Planck temperature in Kelvin (CODATA)
T_PLANCK_K = 1.416784e32

# Today's CMB temperature
T_CMB_TODAY_K = 2.7255

# Convert today's CMB temperature to Planck units
T_CMB_TODAY_PLANCK = T_CMB_TODAY_K / T_PLANCK_K

# Recombination temperature (standard cosmology)
T_RECOMB_K = 3000.0
T_RECOMB_PLANCK = T_RECOMB_K / T_PLANCK_K

# Z values of interest (cosmological epochs)
EPOCHS = [
    ("Today (z=0)",           0.0),
    ("Matter-DE equality",    0.32),       # ΛCDM
    ("Recombination",         1089.0),     # CMB last-scattering
    ("Matter-rad equality",   3400.0),     # ΛCDM z_eq
    ("BBN",                   3.9e8),      # T ~ 0.1 MeV
    ("QCD transition",        7e11),       # T ~ 200 MeV
    ("Electroweak",           1e15),       # T ~ 100 GeV
    ("T = 0.01 × T_srs",      0.01 * T_SRS_PLANCK / T_CMB_TODAY_PLANCK),
    ("T = 0.1 × T_srs",       0.1  * T_SRS_PLANCK / T_CMB_TODAY_PLANCK),
    ("T = T_srs",                  T_SRS_PLANCK / T_CMB_TODAY_PLANCK),
]


# ============================================================================
# Suppression form
# ============================================================================
def p_accept_suppression(T_Planck: float, form: str = "boltzmann") -> float:
    """
    Multiplicative suppression of MDL acceptance probability at finite T.

    p_accept(T) / p_accept(0) = (this function)

    Multiple candidate functional forms; default is Boltzmann-weighted
    relative to substrate stability scale.
    """
    if form == "boltzmann":
        # Boltzmann factor: at T > T_srs, srs falls below MDL waterline
        # Suppression = exp(-(T_srs - T)/T_srs) ... no, that's wrong direction.
        # Correct: at T → T_srs from below, acceptance → 0.
        # Use sigmoid-like form: suppression = 1/(1 + exp((T - T_srs/2)/T_scale))
        # Simplest physical form: p_accept ∝ exp(-T·ln2 / C_srs) (Boltzmann factor
        # for clearing the MDL waterline)
        x = T_Planck * LN2 / C_SRS_BITS
        if x > 100:
            return 0.0
        return math.exp(-x)
    elif form == "linear":
        # Linear interpolation: p_accept = (1 - T/T_srs) for T < T_srs, else 0
        if T_Planck >= T_SRS_PLANCK:
            return 0.0
        return 1.0 - T_Planck / T_SRS_PLANCK
    elif form == "power":
        # Power-law: p_accept = (1 - T/T_srs)^q for some q
        q = 0.5
        if T_Planck >= T_SRS_PLANCK:
            return 0.0
        return (1.0 - T_Planck / T_SRS_PLANCK) ** q
    else:
        raise ValueError(f"Unknown suppression form: {form}")


def H_modification_from_acceptance(suppression: float, p_eff: float = 1.0) -> float:
    """
    Convert acceptance suppression to H(z) modification factor.

    If p_accept(T) = (1/k*) · suppression, then the rate of state-counting
    is suppressed by `suppression`. Cascade theorem H = dN/dt / N gives:
      H(T) / H_coasting(T) ≈ suppression

    So this returns the H modification factor.

    Note: p_eff is an "effective exponent" placeholder if we wanted to
    consider models where acceptance suppression has a non-linear effect
    on H (e.g., via state-counting saturation). Default p_eff = 1 means
    H scales linearly with acceptance.
    """
    return suppression ** p_eff


# ============================================================================
# Main probe
# ============================================================================

def main():
    print("=" * 80)
    print(" Path D probe (D.1) — MDL acceptance rate at finite T")
    print("=" * 80)
    print()
    print(f"  Substrate primitives (theorem-grade):")
    print(f"    k*               = {K_STAR}")
    print(f"    g                = {G_GIRTH}")
    print(f"    α_1_bare         = (2/3)^8 = {ALPHA_1:.6e}")
    print(f"    n_g (per vertex) = {N_G_PER_VERTEX} (Sunada DFS)")
    print(f"    C_srs            = {C_SRS_BITS:.4f} bits/vertex")
    print()
    print(f"  Substrate-stability scale (Planck units):")
    print(f"    T_srs ≈ C_srs / ln(2) = {T_SRS_PLANCK:.4f}")
    print(f"    T_srs in Kelvin       = {T_SRS_PLANCK * T_PLANCK_K:.3e} K")
    print()
    print(f"  Today's CMB temperature (calibration):")
    print(f"    T_CMB(0) = {T_CMB_TODAY_K} K = {T_CMB_TODAY_PLANCK:.3e} Planck units")
    print()
    print(f"  Hypothesis: at finite T, p_accept(T) < 1/k* due to thermal noise")
    print(f"  approaching the MDL waterline. Suppression form: Boltzmann.")
    print()

    # ---------------------------------------------------------------------
    # § A. Probe across cosmological epochs
    # ---------------------------------------------------------------------
    print("-" * 80)
    print("§A. Suppression at cosmological epochs (Boltzmann form)")
    print("-" * 80)
    print()
    print(f"  {'Epoch':<28} {'z':>12} {'T_CMB(z) [K]':>14} {'T/T_srs':>14} {'p_acc/p_0':>14}")
    print(f"  {'-'*28} {'-'*12} {'-'*14} {'-'*14} {'-'*14}")
    for name, z in EPOCHS:
        T_K = T_CMB_TODAY_K * (1.0 + z)
        T_Planck = T_K / T_PLANCK_K
        T_ratio = T_Planck / T_SRS_PLANCK
        supp = p_accept_suppression(T_Planck, "boltzmann")
        print(f"  {name:<28} {z:>12.3e} {T_K:>14.3e} {T_ratio:>14.3e} {supp:>14.6e}")

    print()
    print(f"  Verdict for cosmological epochs:")
    print(f"    Across all observationally-relevant z (including BBN at z ~ 4×10⁸ and")
    print(f"    QCD at z ~ 7×10¹¹), T_CMB/T_srs is overwhelmingly below 1.")
    print(f"    Boltzmann suppression p_accept/p_0 ≈ 1 to many decimal places.")
    print(f"    NO modification of H(z) from MDL acceptance suppression in any")
    print(f"    epoch where observations exist (BBN onwards).")
    print()

    # ---------------------------------------------------------------------
    # § B. Threshold analysis — what z is needed for suppression?
    # ---------------------------------------------------------------------
    print("-" * 80)
    print("§B. Threshold analysis — what z reaches significant suppression?")
    print("-" * 80)
    print()
    print(f"  Suppression threshold table (Boltzmann form):")
    print(f"  {'Target p_acc/p_0':<24} {'Required T/T_srs':<24} {'Required z':<24}")
    print(f"  {'-'*24} {'-'*24} {'-'*24}")
    for target in [0.999, 0.99, 0.9, 0.5, 0.1, 0.01, 1e-3]:
        # Solve: exp(-T·ln2/C_srs) = target  →  T·ln2/C_srs = -ln(target)
        # T (Planck) = -ln(target) · C_srs / ln2 = -log2(target) · ...
        # Actually: exp(-x) = target → x = -ln(target)
        # x = T · ln(2) / C_srs → T = -ln(target) · C_srs / ln(2)
        if target >= 1.0:
            T_req = 0
        else:
            T_req = -math.log(target) * C_SRS_BITS / LN2
        z_req = T_req * T_PLANCK_K / T_CMB_TODAY_K - 1.0
        print(f"  {target:<24g} {T_req/T_SRS_PLANCK:<24.4f} {z_req:<24.3e}")
    print()
    print(f"  Verdict for threshold:")
    print(f"    To get even 0.1% suppression (p_acc/p_0 = 0.999), T must be ~10⁻³")
    print(f"    Planck units — requires z ~ 10²⁹.  Way beyond observational reach")
    print(f"    (recombination at z ~ 10³, matter-rad eq at z ~ 3400).")
    print()

    # ---------------------------------------------------------------------
    # § C. Try other functional forms
    # ---------------------------------------------------------------------
    print("-" * 80)
    print("§C. Alternative suppression forms (sensitivity check)")
    print("-" * 80)
    print()
    print(f"  Test: do other functional forms (linear, power-law) give different")
    print(f"  conclusions at recombination (z = 1089, T = 3000 K)?")
    print()
    T_at_recomb = T_CMB_TODAY_K * (1.0 + 1089.0) / T_PLANCK_K
    for form in ["boltzmann", "linear", "power"]:
        supp = p_accept_suppression(T_at_recomb, form)
        deviation = 1.0 - supp
        print(f"    {form:<14}: p_acc/p_0 = {supp:.6e}, deviation from 1 = {deviation:.3e}")
    print()
    print(f"  All forms give negligible suppression at recombination.  The factor")
    print(f"  T_recomb/T_srs ~ 2×10⁻³¹ dominates regardless of functional form.")
    print()

    # ---------------------------------------------------------------------
    # § D. Verdict
    # ---------------------------------------------------------------------
    print("=" * 80)
    print("VERDICT — Path D probe (D.1) NEGATIVE")
    print("=" * 80)
    print()
    print(f"  Stage-2c noise-floor-based MDL acceptance suppression p_accept(T)")
    print(f"  does NOT produce appreciable H(z) deviation from coasting at any")
    print(f"  observationally-relevant epoch.")
    print()
    print(f"  Quantitative finding:")
    print(f"    T_srs (Planck units)             ≈ {T_SRS_PLANCK:.2f}")
    print(f"    T_CMB(z=1089) (Planck units)     ≈ {T_RECOMB_PLANCK:.3e}")
    print(f"    Ratio                            ≈ {T_RECOMB_PLANCK/T_SRS_PLANCK:.3e}")
    print(f"    Suppression at z = 1089 (Boltzmann): {p_accept_suppression(T_RECOMB_PLANCK):.6e}")
    print(f"    Deviation from 1                 ≈ {1.0 - p_accept_suppression(T_RECOMB_PLANCK):.3e}")
    print()
    print(f"  Even at the matter-radiation-equality epoch (z ~ 3400, T ~ 9000 K),")
    print(f"  the ratio T/T_srs ≈ 4.5×10⁻³¹ — Boltzmann factor is essentially 1.")
    print()
    print(f"  STRUCTURAL CONCLUSION:")
    print(f"    The framework's substrate-stability scale T_srs ~ 10³³ K is")
    print(f"    enormously above any observationally-accessible cosmological epoch.")
    print(f"    Boltzmann-form thermal suppression of MDL acceptance does NOT close")
    print(f"    Item 5.  The natural starting candidate for Path D yields nothing in")
    print(f"    the relevant range.")
    print()
    print(f"  PATH D DECISION TREE — next step:")
    print(f"    (D.1) is closed-negative.  Per `path_D_pre_recombination_research_scoping_2026-05-15.md`,")
    print(f"    next bounded probe is (D.4) — re-audit cascade theorem D3 derivation")
    print(f"    for sub-T_srs epoch restrictions (besides the T < T_srs one already")
    print(f"    confirmed).  If D3 has additional sub-T_srs structure, that could")
    print(f"    introduce a substrate-derived intermediate scale.")
    print()
    print(f"    Otherwise, (D.5) sound-speed modification is the remaining bounded")
    print(f"    probe; (D.2) and (D.3) require Need A multiway formalization, currently")
    print(f"    blocked.")
    print()
    print(f"  HONEST ASSESSMENT:")
    print(f"    Path D's most-natural starting candidate (MDL acceptance suppression")
    print(f"    from Stage-2c noise floor) gives no leverage in the recombination-to-")
    print(f"    T_srs range because the temperatures involved are too low to perturb")
    print(f"    the MDL waterline.  The framework's substrate is fully stable from")
    print(f"    z = 0 to T near Planck scale; no thermal-suppression mechanism is")
    print(f"    available for cosmological-epoch H(z) modification.")
    print()
    print(f"    Item 5 remains research-level.  The closure path (if any) requires")
    print(f"    a mechanism other than thermal MDL suppression — most likely a")
    print(f"    substrate-internal scale (electroweak, QCD, or BBN-related) that")
    print(f"    the framework currently doesn't structurally derive at sub-percent")
    print(f"    precision.")


if __name__ == "__main__":
    main()
