#!/usr/bin/env python3
"""
Canonical prediction file for G_F (Fermi constant) — the observable that CALIBRATES
the framework's one adopted dimensional input, N_hub.

G_F is NOT an anchor. The "N_hub anchored from G_F" framing was RETRACTED 2026-05-12:
the framework adopts ONE dimensional parameter — N_hub ≈ 8.394881e60 — and everything
dimensional is DERIVED from it. G_F sits in that downstream set:

    G_F = 1 / (√2 v²),   v = δ² M_Pl dark / (√2 N_hub^{1/4})   [BZJ cascade ← the adopted N_hub]

But the *value* of the adopted N_hub is currently fixed by requiring exactly this
chain to reproduce the measured G_F (0.51 ppm; `predictions/N_hub.py:n_hub_from_g_f_consistency`
inverts the BZJ relation so the predicted v = (√2·G_F_measured)^{-1/2}). So G_F is
"the observable used to calibrate the one adopted parameter" — its prediction matches
the measurement BY CONSTRUCTION (a calibration round-trip, not an independent test),
exactly as v_Higgs does. The GENUINE independent predictions from the adopted N_hub are
H_0 = 1/(N·t_P) (= 68.0 km/s/Mpc, +1σ vs CMB — `predictions/H_0.py`), t_0 = N·t_P
(`predictions/t_0.py`), and — combined with more structure — the particle masses
(m_τ, m_e, …, with their own deviations: the y_τ chain's +0.13% etc.).

Nothing in the framework "is tied to G_F" structurally: N_hub is the adopted parameter;
G_F is merely the precision instrument that reads off its value. (Closure of Gap G1 —
deriving N_hub from the substrate alone — would remove even that.) See
`simulator.axioms.n_hub_pivot()`, Row P17 of `docs/parameters/parameter_uniqueness_ledger.md`.

NOTE: ADOPTED-DARK-MAP is THEOREM-GRADE (session 18; dark_feshbach_a2_closure.py).

NOTE (2026-05-15 sweep item 5): The G_F round-trip identity v_pred(N_hub) = v_obs is
preserved by construction REGARDLESS of which dark corrections are explicitly applied to
v_higgs.py vs absorbed into N_hub's calibration.  Currently v_higgs.py applies leading
Class C (5/12) explicitly and absorbs Family D sub-leading (-α₁² ≈ -0.152% from the 1H+0F
v_Higgs vertex per master doc §3 (D)) into N_hub.  If Family D were exposed explicitly,
N_hub would shift by +4α₁² ≈ +0.61% (well within the existing G_F-vs-m_τ anchor spread).
Consistency verified by `proofs/foundations/v_higgs_family_D_absorption_check_2026-05-15.py`.
"""

# ============================================================
# PARAMETER: Fermi constant (G_F)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       1.1663787 ± 0.0000006 × 10⁻⁵ GeV⁻²
# Source:      PDG 2024 Review of Particle Physics (Navas et al. 2024,
#              Phys. Rev. D 110, 030001).  Value derived from muon
#              lifetime measurement by the MuLan experiment (Webber
#              et al. 2011, Phys. Rev. Lett. 106, 041803; 0.6 ppm).
# PDG edition: 2024

# --- PREDICTED VALUE -----------------------------------------
# Value:       matches the measured 1.1663787e-5 GeV⁻² by construction (G_F calibrates N_hub)
# Deviation:   +5.173 × 10⁻⁸ GeV⁻²  (+0.4435% relative)
#              Note: the experimental uncertainty on G_F is 0.6 ppm
#              (6 × 10⁻¹² GeV⁻²), so the sigma pull is ~8600 sigma
#              in the experimental sense.  The dominant source of
#              [historical note: pre-2026-04-22 the anchor was H_0 and G_F came out ~0.44% off; with G_F as the calibrating observable since session 19 it is a round-trip]
#              via v^2; the 0.44% residual lies well within that band.

# --- DERIVED FORMULA -----------------------------------------
# G_F = 1 / (√2 × v²)
#
# This is the tree-level SM relation between the Fermi constant and
# the Higgs VEV.  At tree level in the electroweak theory, the
# four-Fermi operator 4G_F/√2 (J†J) arises from integrating out the
# W boson.  The muon-decay amplitude gives G_F/√2 = g₂²/(8M_W²).
# Using M_W = g₂v/2:
#
#   G_F/√2 = g₂² / (8 (g₂v/2)²) = g₂² / (2 g₂² v²) = 1/(2v²)
#
# Hence G_F = 1/(√2 v²)  (exact at tree level).
#
# Ref: Peskin & Schroeder, §20.1 (W propagator at q²=0);
#      Donoghue, Golowich & Holstein, §IV.1.
#
# Chain: A2 (MDL) → MDL mean-field (R≥48) → Curie-Weiss →
#        BZJ finite-size scaling N^{−1/4} at T_c →
#        MDL criticality (μ²=0, R_μ²≥2.88×10⁶) →
#        dark vertex correction (5/12)α₁/(1−α₁) [THEOREM-GRADE; sessions 18+21] →
#        v_higgs ≈ 246.22 GeV  [predictions/v_higgs.py; round-trip via G_F anchor] →
#        G_F = 1/(√2 v²)
#
# Grade: EXTERNAL ANCHOR (calibration role; N_hub anchored from observed G_F
#        per session 19, 2026-04-22). G1 gap closed 2026-04-28 PM via G1b R2
#        path; ADOPTED-DARK-MAP closed 2026-04-28 via Class-2 taxonomy.
#        Residual ±0.13% propagates from y_τ chain (Row P7 Clause 8 numerical
#        residual) via v_Higgs.

# --- INPUTS --------------------------------------------------
# symbol   | value                  | status     | predictions/ file              | meaning
# ---------|------------------------|------------|--------------------------------|--------
# delta    | 2/9                    | [derived]  | predictions/h_walker_eigenvalue.py | Koide phase from Z_3 rate-distortion
# M_P      | M_Pl_natural.M_Pl_GeV | [derived] | predictions/M_Pl_natural.py | M_Pl/M_subst=8/√π theorem-grade; GeV=single declared SI-anchor, Gap-G1 (CODE imports it — line 119 — NOT hardcoded; was falsely "[external]|none")
# N_hub    | ~8.49e60               | [adopted]  | predictions/N_hub.py           | Hubble-Planck inverse (H_0 t_P)^{-1}; adopted scale anchor (Gap G1)
# alpha_1  | (2/3)^8                | [derived]  | predictions/alpha_1.py         | bare NB walk survival, k*=3, g=10

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v_higgs import predict_v_higgs
from alpha_1 import predict_alpha_1
from k_star import predict_k_star
from d_spatial import predict_d_spatial
from g_girth import predict_g_girth
from p_toggle import predict_p_toggle
from V_count import predict_V_count
from N_hub import predict_N_hub
import functools

# --- chain imports ---
d       = predict_d_spatial()
k       = predict_k_star(d)
g       = predict_g_girth(k, d)
alpha_1 = predict_alpha_1(k, g)

# --- the adopted dimensional input + the unit-setting constant ---
from N_hub import N_hub as N_HUB_ADOPTED   # THE adopted dimensional input (≈8.394881e60); predictions/N_hub.py
from M_Pl_natural import M_Pl_GeV as M_P   # CODATA single-source — the unit-setting constant (SI translation)
from delta_Koide import delta_Koide_pred as delta  # = 2/9 (Q*(1-Q) at Q=2/3, predict_delta_Koide)          # Koide phase [derived]
G_F_obs = 1.1663787e-5       # the MEASURED Fermi constant — comparison target only (and pins N_hub's value to ppm; predictions/N_hub.py); NOT a structural input

# --- the chain: adopted N_hub → v_Higgs (BZJ) → G_F ---
N_hub  = N_HUB_ADOPTED

# --- compute v_pred (will ≈ v_obs by construction) ---
v_pred = predict_v_higgs(delta, M_P, N_hub, alpha_1)

# --- G_F round-trip consistency check ---
G_F_pred  = 1.0 / (math.sqrt(2) * v_pred**2)
G_F_sigma = 0.0000006e-5    # GeV^-2  (0.6 ppm experimental)

dev_abs   = G_F_pred - G_F_obs
dev_rel   = dev_abs / G_F_obs
dev_sigma = dev_abs / G_F_sigma

print("=" * 68)
print("  G_F  --  the observable that CALIBRATES the adopted N_hub (matches by construction)")
print("=" * 68)
print(f"  N_hub (adopted)    = {N_hub:.6e}   [the framework's one adopted dimensional parameter]")
print(f"  v_pred (BZJ ← N_hub) = {v_pred:.6f} GeV")
print(f"  G_F = 1/(√2 v²)    = {G_F_pred:.7e} GeV^-2")
print(f"  PDG 2024 measured  = {G_F_obs:.7e} ± {G_F_sigma:.1e} GeV^-2")
print(f"  Round-trip residual = {dev_abs:+.2e} GeV^-2  ({dev_rel*100:+.5f}%)  — ≈0 by construction")
print()
print("  G_F is NOT an anchor (the 'N_hub anchored from G_F' framing was RETRACTED")
print("  2026-05-12). N_hub is the framework's one adopted dimensional parameter; G_F")
print("  is the precision observable used to fix its value — so the G_F prediction")
print("  matches by construction (like v_Higgs). The GENUINE independent predictions")
print("  from the adopted N_hub are H_0 (+1σ vs CMB), t_0, and the particle masses.")
print("  Tree-level G_F = 1/(√2 v²) assumes no beyond-SM corrections. simulator.axioms.n_hub_pivot().")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_G_F(G_F_GeV2, M_P_GeV, alpha_1, delta):
    """G_F = 1/(√2 v²), v ← the adopted N_hub via BZJ — the observable that CALIBRATES N_hub.

    G_F is NOT an anchor (the "N_hub anchored from G_F" framing was RETRACTED
    2026-05-12). N_hub is the framework's one adopted dimensional parameter; its
    *value* is fixed by requiring this very chain to reproduce the measured G_F, so
    the returned G_F matches `G_F_GeV2` by construction (a calibration round-trip,
    like v_Higgs — not an independent test). The genuine independent predictions
    from the adopted N_hub are H_0, t_0, and the particle masses. `G_F_GeV2` is the
    measured value (the calibration target / comparison reference); `M_P_GeV` is the
    unit-setting constant.

    Parameters
    ----------
    G_F_GeV2 : float
        Fermi constant in GeV^{-2} (external anchor; PDG 2024/MuLan 2011).
    M_P_GeV : float
        Planck mass in GeV (CODATA 2018; external).
    alpha_1 : float
        Bare NB walk survival ((k*-1)/k*)^{g-2} = (2/3)^8.
    delta : float
        Koide phase (2/9 exactly).

    Returns
    -------
    float
        G_F recovered from BZJ chain (≈ G_F_GeV2 by construction).
    """
    import math
    k_star_ = predict_k_star(predict_d_spatial())
    p_ = predict_p_toggle()
    V_ = predict_V_count(k_star_, predict_d_spatial())
    N = predict_N_hub(G_F_GeV2, M_P_GeV, alpha_1, delta, k_star_, p_, V_)
    v = predict_v_higgs(delta, M_P_GeV, N, alpha_1)
    return 1.0 / (math.sqrt(2) * v**2)


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_result = G_F_pred
    pure_result = predict_G_F(G_F_obs, M_P, alpha_1, delta)
    print()
    print(f"Implementation:  {impl_result:.10e} GeV^-2")
    print(f"Pure function:   {pure_result:.10e} GeV^-2")
    assert abs(impl_result - pure_result) < 1e-20, \
        f"Mismatch: {impl_result} vs {pure_result}"
    print("OK: round-trip consistent (≈ anchor by construction).")
    print(f"    G_F round-trip = {pure_result:.7e} GeV^-2  "
          f"(anchor: {G_F_obs:.7e}, {dev_rel*100:+.6f}%)")
    print("    Status: EXTERNAL ANCHOR; v_Higgs is calibration check.")
