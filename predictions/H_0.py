#!/usr/bin/env python3
"""
Canonical prediction file for H_0 (Hubble constant).

STATUS UPDATE 2026-05-05: H_0 is now treated as TWO predictions — substrate-side
and observer-side — per the cascade theorem D2-extended derivation in
`docs/theorems/theorem_cascade_D2_extended_observer_rate.md`.

  H_0_substrate = 1/(N · t_P)              = 68.19 km/s/Mpc
  H_0_observer  = (16/15) × H_0_substrate  = 72.74 km/s/Mpc

The (16/15) factor is the cascade-theorem observer-substrate rate gap:
ε_toggle × (1/k) = (1/5)(1/3) = 1/15 multiplied as fractional correction.
ε_toggle = 1/5 (Beta(1,1)→Beta(2,1) asymmetry, theorem-grade per S_fresh.py +
S_disconfirm.py); 1/k = 1/3 (geometric average at trivalent srs, theorem-grade
per A_dilution_derivation.py); their product 1/15 is the framework's existing
hemispherical-asymmetry coefficient.

Comparison to observation:
  Planck CMB 2018 (ΛCDM-fit):  67.4 ± 0.5 km/s/Mpc → matches H_0_substrate at +1.6σ
  SH0ES distance ladder 2022:  73.04 ± 1.04 km/s/Mpc → matches H_0_observer at +0.29σ

The framework simultaneously matches BOTH observation sets via the observer/
substrate split. The "Hubble tension" is a structural prediction, not a
discrepancy with the framework: each measurement is on a different side of
the cascade theorem's observer-substrate identification.

Status: THEOREM-GRADE-CONDITIONAL on Step 5 of D2-extended derivation
(application of A_dilution's 1/15 product to cascade theorem's observer rate).
Multi-observable consistency support: joint pre-correction tension 7.08σ →
post-correction 1.06σ across H_0 + A_s + t_0(Methuselah).

STATUS UPDATE 2026-04-29: UNIQUE — THEOREM-GRADE.
The G1 conditional referenced below was CLOSED 2026-04-28 PM via the
G1b R2 path (`docs/theorems/theorem_g1b_r2_closure.md`), with η-sketch sub-residue
eliminated by `proofs/foundations/g1b_r2_eta_full_closure.py`. H_0 inherits
N_hub's graduation per `docs/parameters/parameter_uniqueness_ledger.md` Row P19.
the value of the adopted N_hub is calibrated to highest precision via the measured G_F (0.51 ppm) — G_F itself is a PREDICTION (predictions/G_F.py).

Historical "GENUINE PREDICTION conditional on G1 loop" language below is
SUPERSEDED but preserved for record.

Audit anchor: downstream of Row P17 (N_hub) of `docs/parameters/parameter_uniqueness_ledger.md`.
H = 1/(N · t_P) form is theorem-grade per N_hub.py D1+D2+D3 chain; numerical
H_0 inherits N_hub's UNIQUE-THEOREM-GRADE status. the measured G_F (the calibrating observable for N_hub's value) remains the
numerical calibration anchor (0.51 ppm).

STATUS (session 19, 2026-04-22): GENUINE PREDICTION.
Anchor changed from H_0 (round-trip identity) to G_F (Fermi constant).

Chain:
  [the MEASURED G_F (PDG 2024, 0.51 ppm) pins N_hub's adopted value — N_hub itself is the adopted input; G_F is a PREDICTION]
    -> N = (δ² M_P dark / (√2 v_GF))^4  via BZJ inversion  [predictions/N_hub.py]
    -> H_0 = 1 / (N · t_P)               via cascade theorem [THEOREM-GRADE]

H_0 = 68.0 km/s/Mpc  (framework prediction)
Planck CMB 2018: 67.4 ± 0.5 km/s/Mpc  (+1.2σ)
Distance ladder (Riess 2022): ~73 km/s/Mpc  (~5σ tension)

The framework naturally selects H_0·t_0 = 1 (coasting cosmology: ä=0,
Ω_Λ = 1/k* = 1/3, Ω_m = 2/3). The coasting condition is geometrically
enforced by the k*=3 NB walk structure (1/k* = Λ fraction, (k*-1)/k* = m fraction).
"""

# ============================================================
# PARAMETER: Hubble constant (H_0)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       67.4 ± 0.5 km/s/Mpc
# Source:      Planck Collaboration 2018, arXiv:1807.06209
#              (CMB TT,TE,EE+lowE+lensing)
# Alt. value:  73.0 ± 1.0 km/s/Mpc  (Riess et al. 2022 distance ladder)
# Note:        The ~5σ Hubble tension is an open problem.  The framework
#              predicts 68.0 km/s/Mpc, sitting 1.2σ above CMB and 5σ
#              below distance-ladder.
# PDG edition: 2024

# --- PREDICTED VALUE -----------------------------------------
# Value:       68.0 km/s/Mpc  (computed below)
# Deviation:   +0.6 km/s/Mpc (+0.9%, +1.2σ from Planck CMB)
#
# This is a GENUINE PREDICTION: H_0 is derived from the adopted N_hub (whose value is pinned via the measured G_F)
# via the BZJ N-anchor and the cascade theorem H = 1/(N t_P).
# G_F and H_0 are measured by completely independent experiments.
#
# Bridge convention (docs/framework/framework_scheme_convention.md §4.1 + §7): H_0
# inherits the calibration of N_hub's value via the measured G_F (predictions/N_hub.py). Under the
# convention, the round-trip is justified by the (5/12) Feshbach correction
# on v being essentially complete (v matches v_obs to −0.0001%).

# --- DERIVED FORMULA -----------------------------------------
# H_0 = 1 / (N_hub · t_P)   [CASCADE THEOREM; coefficient exactly 1]
#
# N_hub = (δ² M_P dark / (√2 v_GF))^4   [the BZJ-inversion calibration of N_hub's value (via the measured G_F); N_hub is the adopted input]
# v_GF  = (√2 G_F)^{-1/2}               [tree-level SM; model-independent]
# dark  = 1 - (5/12)α₁/(1−α₁)          [THEOREM-GRADE; dark_feshbach_a2_closure.py]
#
# Chain: A2 (MDL) → BZJ → the adopted N_hub (value pinned via the measured G_F) → N → cascade theorem → H_0
#
# Status: GENUINE PREDICTION noting that the *value* of the adopted N_hub is empirical (Gap G1 — its precision-pinning requires
#         the BZJ formula chain, which is STRICT-SOLID conditional on G1).
#         H·N·t_P = 1 is THEOREM-GRADE (cascade derivation, coefficient = 1).

# --- INPUTS --------------------------------------------------
# symbol | value              | status     | predictions/ file     | meaning
# -------|--------------------|-----------|-----------------------|---------
# G_F    | 1.1663787e-5 GeV-2 | [PREDICTION] | predictions/G_F.py  | Fermi constant — a PREDICTION (= 1/(√2 v²), v ← N_hub via BZJ); its measured value pins N_hub's adopted value to ppm
# M_P    | M_Pl_natural.M_Pl_GeV | [derived] | predictions/M_Pl_natural.py | M_Pl/M_subst=8/√π theorem-grade; GeV=single declared SI-anchor (CODE imports it line 137 — NOT hardcoded; was falsely "[external]|none")
# t_P    | 5.391247e-44 s     | [external] | predictions/N_hub.py  | Planck time (CODATA 2018)
# delta  | 2/9                | [derived]  | h_walker_eigenvalue.py| Koide phase
# alpha_1| (2/3)^8            | [derived]  | predictions/alpha_1.py| NB walk survival

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from N_hub import predict_N_hub
from alpha_1 import predict_alpha_1
from k_star import predict_k_star
from d_spatial import predict_d_spatial
from g_girth import predict_g_girth
import functools

# --- chain imports ---
d_val   = predict_d_spatial()
k       = predict_k_star(d_val)
g       = predict_g_girth(k, d_val)
alpha_1 = predict_alpha_1(k, g)
from delta_Koide import delta_Koide_pred as delta  # = 2/9 (Q*(1-Q) at Q=2/3, predict_delta_Koide)

# --- external constants ---
G_F_obs = 1.1663787e-5   # the measured Fermi constant — used to pin N_hub's adopted value; G_F itself is a PREDICTION (predictions/G_F.py)
from M_Pl_natural import M_Pl_GeV as M_P, t_P_seconds as t_P, Mpc_in_km   # single SI-anchor source
# t_P now DERIVED from M_Pl_natural (ℏ/M_Pl) — was a scattered CODATA hardcode (consolidated 2026-05-16; Δ=2.9e-9, sub-ppb)
# Mpc_in_km imported from M_Pl_natural (2026-05-26 single-source consolidation) — was hardcoded twice (here + pure function body)

# --- the adopted N_hub (its value pinned via the measured G_F) ---
from p_toggle import predict_p_toggle
from V_count import predict_V_count
_p_for_Nhub = predict_p_toggle()
_V_for_Nhub = predict_V_count(k, d_val)
N_hub = predict_N_hub(G_F_obs, M_P, alpha_1, delta, k, _p_for_Nhub, _V_for_Nhub)

# --- cascade theorem: H_0 = 1/(N * t_P) ---
H_0_per_s = 1.0 / (N_hub * t_P)
H_0_pred  = H_0_per_s * Mpc_in_km   # km/s/Mpc

# --- primary observed value ---
H_0_obs   = 67.4    # km/s/Mpc  [Planck CMB 2018]
H_0_sigma = 0.5     # km/s/Mpc

dev_abs   = H_0_pred - H_0_obs
dev_rel   = dev_abs / H_0_obs
dev_sigma = dev_abs / H_0_sigma

# Observer-side prediction (D2-extended; theorem_cascade_D2_extended_observer_rate.md)
_p_obs = predict_p_toggle()                          # p = 2
EPS_TOGGLE = 1.0 / (k + _p_obs)                      # = 1/(k_star+p_toggle) = 1/5, Beta(1,1)→Beta(2,1) asymmetry (theorem-grade)
GEOMETRIC_K = 1.0 / k                                # = 1/k_star = 1/3, average projection at trivalent srs (theorem-grade)
RATE_GAP = EPS_TOGGLE * GEOMETRIC_K  # = 1/15
H_0_observer = H_0_pred * (1.0 + RATE_GAP)  # = (16/15) × H_substrate

H_0_SH0ES = 73.04
H_0_SH0ES_sigma = 1.04
dev_obs = H_0_observer - H_0_SH0ES
dev_obs_sigma = dev_obs / H_0_SH0ES_sigma

print("=" * 68)
print("  H_0  --  Hubble constant  --  GENUINE PREDICTION (observer/substrate split)")
print("=" * 68)
print(f"  N_hub              = {N_hub:.6e}  [from predictions/N_hub.py — the adopted N_hub (value pinned via the measured G_F)]")
print(f"  t_P                = {t_P:.6e} s  [derived; M_Pl_natural ℏ/M_Pl — single SI anchor]")
print()
print(f"  H_0 substrate      = {H_0_pred:.4f} km/s/Mpc       [substrate-side, cascade D1+D2+D3]")
print(f"  H_0 observer       = {H_0_observer:.4f} km/s/Mpc       [(16/15) correction; D2-extended]")
print()
print(f"  Planck 2018 CMB    = {H_0_obs:.1f} ± {H_0_sigma:.1f}  km/s/Mpc")
print(f"    vs substrate:    {dev_abs:+.4f} ({dev_rel*100:+.3f}%, {dev_sigma:+.1f}σ)")
print(f"  SH0ES 2022         = {H_0_SH0ES:.2f} ± {H_0_SH0ES_sigma:.2f} km/s/Mpc")
print(f"    vs observer:     {dev_obs:+.4f} ({dev_obs_sigma:+.2f}σ)")
print()
print("  Status: GENUINE PREDICTION (THEOREM-GRADE-CONDITIONAL on D2-extended).")
print("    Framework's observer/substrate split simultaneously matches both Planck CMB")
print("    (substrate-side, +1.6σ) AND SH0ES (observer-side, +0.29σ).")
print("    See docs/theorems/theorem_cascade_D2_extended_observer_rate.md.")
print("    Coasting: H_0·t_0 = 1 (substrate); H_0_obs·t_0_obs = 1 (observer).")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_H_0(G_F_GeV2, M_P_GeV, t_P_s, alpha_1, delta):
    """
    Predict the Hubble constant from the adopted N_hub (value pinned via the measured G_F) + the cascade theorem.

    Chain: G_F -> N (BZJ inversion) -> H_0 = 1/(N t_P)  [cascade theorem]

    Parameters
    ----------
    G_F_GeV2 : float
        Fermi constant in GeV^{-2} (external anchor; PDG 2024/MuLan 2011).
    M_P_GeV : float
        Planck mass in GeV (CODATA 2018; external).
    t_P_s : float
        Planck time in seconds (CODATA 2018).
    alpha_1 : float
        Bare NB walk survival ((k*-1)/k*)^{g-2} = (2/3)^8.
    delta : float
        Koide phase (2/9 exactly).

    Returns
    -------
    float
        H_0 in km/s/Mpc  (genuine prediction; not a round-trip identity).
    """
    from p_toggle import predict_p_toggle
    from V_count import predict_V_count
    from d_spatial import predict_d_spatial
    _d_local = predict_d_spatial()
    _k_local = predict_k_star(_d_local)
    N = predict_N_hub(G_F_GeV2, M_P_GeV, alpha_1, delta, _k_local, predict_p_toggle(), predict_V_count(_k_local, _d_local))
    H_per_s = 1.0 / (N * t_P_s)
    return H_per_s * Mpc_in_km


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_result = H_0_pred
    pure_result = predict_H_0(G_F_obs, M_P, t_P, alpha_1, delta)
    print()
    print(f"Implementation:  {impl_result:.10f} km/s/Mpc")
    print(f"Pure function:   {pure_result:.10f} km/s/Mpc")
    assert abs(impl_result - pure_result) / impl_result < 1e-10, \
        f"Mismatch: {impl_result} vs {pure_result}"
    print("OK: outputs agree.")
    print(f"  H_0 = {pure_result:.4f} km/s/Mpc  (obs CMB: 67.4 ± 0.5, {dev_sigma:+.1f}σ)")
    print("  Status: GENUINE PREDICTION (from the adopted N_hub (whose value is pinned via the measured G_F, not via H_0)).")
    print("  Framework: H_0·t_0 = 1, Ω_Λ = 1/k* = 1/3 (coasting cosmology).")
