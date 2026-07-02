#!/usr/bin/env python3
"""
Canonical prediction file for the tau Yukawa coupling y_τ.
"""

# ============================================================
# PARAMETER: y_tau (tau Yukawa coupling)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       y_tau = m_tau / v = 1.77686 / 246.22 GeV ≈ 7.2160e-3
# Source:      PDG 2024 (m_tau); PDG 2022 electroweak precision fits (v)
# PDG edition: 2024
# Note:        y_tau is not measured directly; it is inferred from m_tau
#              and v via the SM mass relation m_tau = y_tau × v / √2,
#              giving y_tau = √2 × m_tau / v. The "m_tau/v" form above
#              uses the convention where Yukawa is defined as the
#              DIMENSIONLESS coupling to the full Higgs field (not /√2).

# --- PREDICTED VALUE -----------------------------------------
# Value:       y_tau = α₁_full / k*² = (5/3)(2/3)^8 / 9 = 1280/177147 ≈ 7.2256e-3
# Deviation:   +1.33e-5 absolute, +0.13% relative
#
# Bridge convention (docs/framework/framework_scheme_convention.md §4.4): the framework's
# y_tau is a tree-level α₁-dependent coupling, NOT MS̄-at-any-scale. Comparison
# to y_tau_obs = m_tau/v uses the bridge convention "bare + Feshbach = SM
# pole-mass-equivalent." The +0.13% residual corresponds to a Feshbach analog
# on the fermion-Higgs vertex that has not been investigated under the
# convention (Priority 4.4 step 2.2; see master_plan.md).

# --- DERIVED FORMULA -----------------------------------------
# y_τ = α₁_full / k*²
#
# Gate-first derivation chain (docs/theorems/theorem_ytau_corollary.md):
#   §4 Cycle amplitude α₁_full = (5/3)(2/3)^8  [T4: alpha_1_full.py]
#   §5 Fermion edge projection (ψ)   = 1/k*    [T1+T3+T2: A5(b) + I4₁32]
#   §5 Fermion edge projection (ψ̄)  = 1/k*    [same]
#   §6 Higgs edge (deterministic complement)   = 1 [T2+T4: theorem_g2_edge_qubit_su2.md]
#   §7 Cl(0,2) channel (per-process waterline) = 1 [T3+T1: Peskin-Schroeder §20.2 + A2]
#
# Product: y_τ = α₁_full × (1/k*) × (1/k*) × 1 × 1 = α₁_full/k*²
#
# Status: THEOREM-GRADE under A1 + A3-T + A5(a) + A5(b). Zero adoptions.
# Supersedes the 4/5 corollary grade of proofs/masses/ytau_corollary.py
# Part 9. Premise (c.ii) resolved via the per-process waterline reading:
# the two Cl(0,2) directions in the Higgs doublet (f₁, f₂) pair with
# DIFFERENT fermion bilinears under SU(2)_L × U(1)_Y gauge structure,
# so they contribute to distinct couplings — not both to y_τ.

# --- INPUTS --------------------------------------------------
# symbol       | value             | status      | predictions/ file          | meaning
# -------------|-------------------|-------------|----------------------------|--------
# alpha_1_full | (5/3)(2/3)^8      | [derived]   | predictions/alpha_1_full.py| Class-2 dark-sector coupling
# k_star       | 3                 | [derived]   | predictions/k_star.py      | srs coordination number
# A5(b)        | —                 | [axiom]     | docs/framework/framework_axioms.md §5b | MDL prob = coupling
# theorem      | —                 | [upstream]  | docs/theorems/theorem_ytau_corollary.md | full gate-first proof

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from k_star import predict_k_star
from d_spatial import predict_d_spatial
from g_girth import predict_g_girth
from alpha_1 import predict_alpha_1
from alpha_1_full import predict_alpha_1_full, n_g_edge
from dark_extraction_map import family_D_per_leg_correction
import functools

# --- chain imports ---
d = predict_d_spatial()
k = predict_k_star(d)
g = predict_g_girth(k, d)
alpha_1_bare_val = float(predict_alpha_1(k, g))             # = (2/3)^8 for Family D
alpha_1_full_exact = predict_alpha_1_full(k, g, n_g_edge)   # Fraction
alpha_1_full = float(alpha_1_full_exact)

# --- y_τ tree-level (theorem-grade core) ---
y_tau_tree = alpha_1_full / (k ** 2)

# --- Family D per-leg multiway dark-disruption correction ---
#   STATUS (corrected W1 2026-05-18): THEOREM-GRADE-STRUCTURAL, conditional.
#   c_H = α₁² structurally derived (Route H; Route C corrob.). c_F via the
#   parameter_linter Clause-6 two-step channel_select → canonical_encoding
#   (single_edge_spectral channel fixed by theorem_car_local_jordan_wigner §1;
#   the historical "Routes F-1/F-2" are canonical_encoding-EQUIVALENT, not
#   independent — see predictions/dark_extraction_map.py
#   _c_F_denominator_channel_select + master doc §3 (D); verified via the real
#   simulator/gating/mdl.channel_select gate in proofs/foundations/
#   c_F_channel_select_waterfilling_2026-05-18.py). NOT UNIQUE-THEOREM-GRADE.
#   Numeric value UNCHANGED.
# Per docs/theorems/theorem_substrate_feshbach_dark_corrections_master.md §3 (D):
# Yukawa vertex y_τ φ ψ̄ψ has 1 Higgs leg + 2 fermion legs.
# Per-Higgs-leg c_H = α₁_bare² (Route H derived; Route C corrob.).
# Per-fermion-leg c_F = -α₁_bare²/(N_atoms·k*) (Clause-6 two-step:
#   channel_select[single_edge_spectral] → canonical_encoding; value unchanged).
# Combined: δy_τ/y_τ = -(5/6)·α₁_bare² ≈ -0.127%.
from V_count import V_count_pred as N_atoms_srs  # = 4, srs primitive cell |V| / K_4 quotient (predict_V_count)
n_H_legs_yuk = 1      # 1 Higgs leg at the Yukawa vertex
n_F_legs_yuk = 2      # 2 fermion legs at the Yukawa vertex
family_D_factor = family_D_per_leg_correction(alpha_1_bare_val, n_H_legs_yuk,
                                                n_F_legs_yuk, N_atoms_srs, k)
y_tau_pred = y_tau_tree * family_D_factor

# --- observed value ---
m_tau_obs = 1.77686   # GeV (PDG 2024)
v_obs     = 246.22    # GeV (PDG 2022 EW fits)
y_tau_obs = m_tau_obs / v_obs

dev_abs   = y_tau_pred - y_tau_obs
dev_rel   = dev_abs / y_tau_obs

print("=" * 68)
print("  y_tau (tau Yukawa coupling) -- THEOREM-GRADE under A1 + A3-T + A5(a) + A5(b)")
print("  + Family D per-leg multiway dark-disruption (master doc §3 (D), 2026-05-15)")
print("=" * 68)
print(f"  k*           = {k}")
print(f"  g            = {g}")
print(f"  alpha_1_full = (5/3)(2/3)^8 = {alpha_1_full:.15f}")
print()
print(f"  y_tau_tree   = alpha_1_full / k*^2 = {y_tau_tree:.15f}")
print(f"  Family D     = 1 - (5/6)·α₁_bare²  = {family_D_factor:.15f}")
print(f"                 (vertex: 1 Higgs + 2 fermion legs)")
print(f"  y_tau_pred   = y_tau_tree × Family D = {y_tau_pred:.15f}")
print(f"  y_tau_obs    = m_tau/v = {m_tau_obs}/{v_obs} = {y_tau_obs:.15f}")
print(f"  Deviation    = {dev_abs:+.3e}  ({dev_rel*100:+.4f}%)")
print()
print("  Tree derivation: docs/theorems/theorem_ytau_corollary.md")
print("  Family D:        docs/theorems/theorem_substrate_feshbach_dark_corrections_master.md §3 (D)")
print("                   c_H = α₁²: Route H derived (Route C corrob.)")
print("                   c_F = -α₁²/12: Clause-6 channel_select →")
print("                                  canonical_encoding (W1 2026-05-18;")
print("                                  dark_extraction_map _c_F_denominator_channel_select)")
print("  Status:          THEOREM-GRADE-STRUCTURAL conditional; m_τ -0.17σ_PDG (value unchanged)")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_y_tau(alpha_1_full, alpha_1_bare, k_star, n_H_legs, n_F_legs, N_atoms):
    """
    Computes the tau Yukawa coupling from the framework's graph-QFT
    structure at a trivalent srs vertex, with Family D per-leg dark
    correction applied.

    Formula:
        y_τ_physical = y_τ_tree × family_D_factor
                     = (α₁_full / k*²) × (1 - n_H·α₁_bare² + n_F·α₁_bare²/(N_atoms·k*))

    Tree-level (theorem-grade, docs/theorems/theorem_ytau_corollary.md):
      y_τ_tree = α₁_full / k*²
      where α₁_full = (5/3)(2/3)^8 is the full Class-2 dark-sector coupling
      and k* = 3 is the srs coordination number.

    Family D correction (THEOREM-GRADE-STRUCTURAL conditional, W1
    2026-05-18; master doc §3 (D)):
      - Per-Higgs-leg rate c_H = α₁_bare² (Route H derived; Route C corrob.)
      - Per-fermion-leg rate c_F = -α₁_bare²/(N_atoms·k*) via Clause-6
        channel_select → canonical_encoding (value unchanged)
      - Yukawa vertex: n_H = 1, n_F = 2 → factor = 1 - (5/6)·α₁_bare² ≈ 0.999

    Parameters
    ----------
    alpha_1_full : float
        Full Class-2 dark-sector coupling (5/3)(2/3)^8, from alpha_1_full.py.
    alpha_1_bare : float
        Bare NB walk survival (2/3)^8, from alpha_1.py (theorem-grade upstream).
    k_star : int
        Coordination number of the srs net (= 3), from k_star.py.
    n_H_legs : int
        Higgs legs at the Yukawa vertex (structural: 1).
    n_F_legs : int
        Fermion legs at the Yukawa vertex (structural: 2).
    N_atoms : int
        Wyckoff 8a atoms per primitive cell of srs (= 4).

    Returns
    -------
    float
        y_τ_physical = (α₁_full / k*²) × (1 - (5/6)·α₁_bare²)
                     ≈ 7.2165e-3  (vs observed 7.2166e-3, -0.17σ_PDG on m_τ)
    """
    y_tau_tree = alpha_1_full / (k_star * k_star)
    family_D_factor = family_D_per_leg_correction(alpha_1_bare, n_H_legs, n_F_legs,
                                                    N_atoms, k_star)
    return y_tau_tree * family_D_factor


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_result = y_tau_pred
    pure_result = predict_y_tau(alpha_1_full, alpha_1_bare_val, k, n_H_legs_yuk,
                                  n_F_legs_yuk, N_atoms_srs)
    print()
    print(f"Implementation: {impl_result:.15f}")
    print(f"Pure function:  {pure_result:.15f}")
    assert abs(impl_result - pure_result) < 1e-15, \
        f"Mismatch: {impl_result} vs {pure_result}"
    print("OK: outputs agree.")
    print(f"    y_tau_tree           = alpha_1_full/k*^2 = 1280/177147 = {y_tau_tree:.10f}")
    print(f"    y_tau_physical (FD)  = y_tau_tree × (1 - (5/6)·α₁²)   = {pure_result:.10f}")
    print(f"    y_tau_obs            = m_τ/v                          = {y_tau_obs:.10f}")
    print(f"    σ_PDG match:         {dev_rel*100:+.5f}%  ({(pure_result-y_tau_obs)/(0.00012/v_obs):+.2f}σ_PDG)")
    print()
    print("    Rigor status: THEOREM-GRADE-STRUCTURAL conditional (W1 2026-05-18)")
    print("      Tree: A1 + A3-T + A5(a) + A5(b) (theorem_ytau_corollary.md)")
    print("      Family D: master doc §3 (D) — c_H Route H derived;")
    print("        c_F = -α₁²/(N·k*) via Clause-6 channel_select→canonical_encoding")
    print("        (F-1≡F-2 encoding-equivalent, not independent; value unchanged)")
