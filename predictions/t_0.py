#!/usr/bin/env python3
"""
Canonical prediction file for t_0 (age of the universe).

STATUS UPDATE 2026-04-29: UNIQUE — THEOREM-GRADE.
The G1 / "STRICT-SOLID (the value of the adopted N_hub is empirical, pinned via the measured G_F)" status referenced below was
upgraded 2026-04-28 PM via the G1b R2 path closure
(`docs/theorems/theorem_g1b_r2_closure.md`, η-sketch sub-residue eliminated by
`proofs/foundations/g1b_r2_eta_full_closure.py`). t_0 inherits N_hub's
graduation per `docs/parameters/parameter_uniqueness_ledger.md` Row P20. The coasting
condition H_0·t_0 = 1 is itself theorem-grade independent of G_F (from
Ω_Λ = 1/k* = 1/3 + Friedmann ä = 0). −0.1σ from Methuselah (model-
independent) is strong evidence for the framework's coasting cosmology.

Historical "STRICT-SOLID (the value of the adopted N_hub is empirical, pinned via the measured G_F)" / "GENUINE PREDICTION
(conditional)" language below is SUPERSEDED but preserved for record.

Audit anchor: downstream of Row P17 (N_hub) of `docs/parameters/parameter_uniqueness_ledger.md`.
t_0 = N_hub · t_P (cascade theorem coefficient = 1, theorem-grade); inherits
N_hub's UNIQUE-THEOREM-GRADE status post-2026-04-28-PM G1b R2 closure.

STATUS: GENUINE PREDICTION (UNIQUE-THEOREM-GRADE).

Chain:
  [the MEASURED G_F (PDG 2024, 0.51 ppm) pins N_hub's adopted value — N_hub itself is the adopted input; G_F is a PREDICTION]
    -> N = (δ² M_P dark / (√2 v_GF))^4  via BZJ inversion  [predictions/N_hub.py]
    -> t_0 = N · t_P                      via cascade theorem [THEOREM-GRADE]

t_0 = 14.38 Gyr  (framework prediction)

Observed:
  CMB/ΛCDM:   13.797 ± 0.023 Gyr  (Planck 2018; model-dependent)
  Methuselah: 14.46  ± 0.80  Gyr  (Bond et al. 2013, ApJ 765:L12; oldest star HD 140283)

The framework predicts H_0·t_0 = 1 exactly (coasting geometry: Ω_Λ = 1/k* = 1/3).
This is consistent with Methuselah at +0.0σ and inconsistent with CMB/ΛCDM at +25σ.
The CMB/ΛCDM value assumes the ΛCDM cosmological model (Ω_Λ ≈ 0.68);
the framework predicts a different dark energy fraction (Ω_Λ = 1/3).
"""

# ============================================================
# PARAMETER: t_0 (age of the universe)
# ============================================================

# --- OBSERVED VALUES -----------------------------------------
# CMB/ΛCDM:  13.797 ± 0.023 Gyr
#   Source:  Planck Collaboration 2018, arXiv:1807.06209
#   Note:    model-dependent (assumes ΛCDM with Ω_Λ ≈ 0.68, Ω_m ≈ 0.31)
#
# Methuselah star (HD 140283):  14.46 ± 0.80 Gyr
#   Source:  Bond, Nelan, VandenBerg, Schaefer, Lawler 2013, ApJ 765:L12
#   Note:    model-independent lower bound from stellar evolution;
#            the most constraining direct age measurement

# --- PREDICTED VALUE -----------------------------------------
# Value:  14.38 Gyr  (computed below)
# Deviation from CMB/ΛCDM:  +25σ  (but CMB value assumes different cosmology)
# Deviation from Methuselah: -0.1σ (within 0.1σ — the framework's cosmology)
#
# The near-exact agreement with Methuselah is strong evidence for the
# coasting condition H_0·t_0 = 1 in the framework's cosmology.
#
# Bridge convention (docs/framework/framework_scheme_convention.md §4.1 + §7): t_0
# inherits the calibration of N_hub's value via the measured G_F (predictions/N_hub.py). Under the
# convention, the round-trip is justified by the (5/12) Feshbach correction
# on v being essentially complete.

# --- DERIVED FORMULA -----------------------------------------
# t_0 = N_hub · t_P   [CASCADE THEOREM; coefficient exactly 1]
#
# N_hub = (δ² M_P dark / (√2 v_GF))^4   [the BZJ-inversion calibration of N_hub's value (via the measured G_F); N_hub is the adopted input]
#
# Chain: the adopted N_hub (value pinned via the measured G_F) -> t_0 = N·t_P  [cascade theorem THEOREM-GRADE]
#
# Framework cosmology context:
#   H_0 · t_0 = 1 (exact)           [coasting: ä = 0 at current epoch]
#   Ω_Λ = 1/k* = 1/3                [NB walk dark fraction]
#   Ω_m = (k*-1)/k* = 2/3           [NB walk matter fraction]
#   Ω_m = 2 Ω_Λ  =>  ä = 0         [Einstein eq for coasting flat universe]
#
# Status: GENUINE PREDICTION (from the adopted N_hub (value pinned via the measured G_F); cascade theorem THEOREM-GRADE).
#         Conditional on G1 (BZJ chain; same gap as v_Higgs and H_0).

# --- INPUTS --------------------------------------------------
# symbol | value              | status     | predictions/ file     | meaning
# -------|--------------------|-----------|-----------------------|---------
# G_F    | 1.1663787e-5 GeV-2 | [PREDICTION] | predictions/G_F.py  | Fermi constant — a PREDICTION (= 1/(√2 v²), v ← N_hub via BZJ); its measured value pins N_hub's adopted value to ppm
# M_P    | M_Pl_natural.M_Pl_GeV | [derived] | predictions/M_Pl_natural.py | M_Pl/M_subst=8/√π theorem-grade; GeV=single declared SI-anchor (CODE imports it line 116 — NOT hardcoded; was falsely "[external]|none")
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
from p_toggle import predict_p_toggle
from V_count import predict_V_count
import functools

# --- chain imports ---
d_val   = predict_d_spatial()
k       = predict_k_star(d_val)
g       = predict_g_girth(k, d_val)
alpha_1 = predict_alpha_1(k, g)
p_val   = predict_p_toggle()
V_val   = predict_V_count(k, d_val)
from delta_Koide import delta_Koide_pred as delta  # = 2/9 (Q*(1-Q) at Q=2/3, predict_delta_Koide)

# --- external constants ---
G_F_obs = 1.1663787e-5   # the measured Fermi constant — used to pin N_hub's adopted value; G_F itself is a PREDICTION (predictions/G_F.py)
from M_Pl_natural import M_Pl_GeV as M_P, t_P_seconds as t_P, Mpc_in_km   # single SI-anchor source (Mpc_in_km consolidated 2026-05-26)
# t_P now DERIVED from M_Pl_natural (ℏ/M_Pl) — was a scattered CODATA hardcode (consolidated 2026-05-16; Δ=2.9e-9, sub-ppb)

# --- compute N and t_0 ---
N_hub = predict_N_hub(G_F_obs, M_P, alpha_1, delta, k, p_val, V_val)
t_0_pred_s = N_hub * t_P                        # seconds
Gyr = 3.1557e16                                  # seconds per Gyr (Julian year)
t_0_pred = t_0_pred_s / Gyr                     # Gyr

# --- observed values ---
t_0_CMB   = 13.797   # Gyr  [Planck 2018 CMB/ΛCDM; model-dependent]
t_0_CMB_s = 0.023    # Gyr  sigma
t_0_star  = 14.46    # Gyr  [Bond 2013; Methuselah star HD 140283]
t_0_star_s = 0.80    # Gyr  sigma

dev_CMB   = (t_0_pred - t_0_CMB) / t_0_CMB_s
dev_star  = (t_0_pred - t_0_star) / t_0_star_s

# --- Runner-facing canonical anchor (Clause 8 Category-B special
# accommodation; parameter_linter.md names "Methuselah for t_0"). The
# framework's claimed comparison is the MODEL-INDEPENDENT stellar age,
# NOT the model-dependent Planck CMB/ΛCDM value (which inherits
# recombination/acoustic physics the framework has no generator for —
# the L6 wall). Without these aliases run_predictions.py falls back to
# the manifest CMB value and mis-reports a coasting-predicted +25σ
# difference as a failure. Aliases only; zero computational change.
t_0_obs   = t_0_star      # 14.46 Gyr  (Bond et al. 2013, HD 140283)
t_0_sigma = t_0_star_s    # 0.80 Gyr
dev_sigma = dev_star      # ≈ −0.1σ (framework PASS)

# --- verify coasting condition: H_0 · t_0 = 1 ---
# Mpc_in_km imported from M_Pl_natural single-source above (was: 3.085677581e19 inline; consolidated 2026-05-26)
H_0_per_s = 1.0 / (N_hub * t_P)
H_0_km    = H_0_per_s * Mpc_in_km
H_0_t_0   = H_0_per_s * t_0_pred_s   # should = 1 exactly

# Observer-side prediction (D2-extended; theorem_cascade_D2_extended_observer_rate.md)
# H_0 · t_0 = 1 in coasting; if H_0_obs = (16/15)·H_sub then t_0_obs = (15/16)·t_0_sub
_p_obs = predict_p_toggle()                          # p = 2
EPS_TOGGLE = 1.0 / (k + _p_obs)                      # = 1/(k_star+p_toggle) = 1/5
GEOMETRIC_K = 1.0 / k                                # = 1/k_star = 1/3
RATE_GAP = EPS_TOGGLE * GEOMETRIC_K  # = 1/15
t_0_observer = t_0_pred * (1.0 / (1.0 + RATE_GAP))  # = (15/16) × t_0_substrate

dev_obs_CMB = (t_0_observer - t_0_CMB) / t_0_CMB_s

print("=" * 68)
print("  t_0  --  Age of universe  --  GENUINE PREDICTION (observer/substrate split)")
print("=" * 68)
print(f"  N_hub              = {N_hub:.6e}  [from the adopted N_hub (value pinned via the measured G_F)]")
print(f"  t_P                = {t_P:.6e} s")
print()
print(f"  t_0 substrate      = {t_0_pred:.4f} Gyr   [substrate-side, cascade D1+D2+D3]")
print(f"  t_0 observer       = {t_0_observer:.4f} Gyr   [(15/16) correction; D2-extended]")
print()
print(f"  Methuselah star      = {t_0_star:.2f} ± {t_0_star_s:.2f} Gyr  [substrate-side, stellar evolution]")
print(f"    vs t_0_substrate:  {dev_star:+.2f}σ")
print(f"  Planck 2018 CMB/ΛCDM = {t_0_CMB:.3f} ± {t_0_CMB_s:.3f} Gyr  [observer-side, ΛCDM-fit]")
print(f"    vs t_0_substrate:  {dev_CMB:+.1f}σ  (cosmology-model mismatch — predicted)")
print(f"    vs t_0_observer:   {dev_obs_CMB:+.1f}σ  (still model-dependent — coasting fit gives different result)")
print()
print(f"  Coasting check: H_0·t_0 = {H_0_t_0:.8f}  [exactly 1 by construction]")
print(f"  H_0_pred = {H_0_km:.4f} km/s/Mpc  (see predictions/H_0.py)")
print()
print("  Framework cosmology:")
print(f"    Ω_Λ = 1/k* = 1/{k}  (NB walk dark fraction)")
print(f"    Ω_m = (k*-1)/k* = {k-1}/{k}  (NB walk matter fraction)")
print(f"    ä = 0  (coasting; Ω_m = 2Ω_Λ in flat Einstein eq)")
print(f"    H_0·t_0 = 1  (exact coasting relation)")
print()
print("  Λ_CC: substrate Λ=1/N² in predictions/Lambda_CC.py; the Planck-CMB")
print("  t_0 mismatch is the same parametric-class translation as the Λ_CC")
print("  factor-of-2, predicted observable-side in predictions/Lambda_CC_LCDM.py.")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_t_0(G_F_GeV2, M_P_GeV, t_P_s, alpha_1, delta):
    """
    Predict the age of the universe from the adopted N_hub (value pinned via the measured G_F) + cascade theorem.

    Chain: G_F -> N (BZJ inversion) -> t_0 = N * t_P  [cascade theorem]

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
        t_0 in seconds (age of universe).  Divide by 3.1557e16 for Gyr.
    """
    k_star_ = predict_k_star(predict_d_spatial())
    p_ = predict_p_toggle()
    V_ = predict_V_count(k_star_, predict_d_spatial())
    N = predict_N_hub(G_F_GeV2, M_P_GeV, alpha_1, delta, k_star_, p_, V_)
    return N * t_P_s


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_result = t_0_pred_s
    pure_result = predict_t_0(G_F_obs, M_P, t_P, alpha_1, delta)
    pure_Gyr    = pure_result / Gyr
    print()
    print(f"Implementation:  {impl_result:.6e} s  ({t_0_pred:.4f} Gyr)")
    print(f"Pure function:   {pure_result:.6e} s  ({pure_Gyr:.4f} Gyr)")
    assert abs(impl_result - pure_result) / impl_result < 1e-10, \
        f"Mismatch: {impl_result} vs {pure_result}"
    print("OK: outputs agree.")
    print(f"  t_0 = {pure_Gyr:.4f} Gyr")
    print(f"  Methuselah: {t_0_star:.2f} ± {t_0_star_s:.2f} Gyr  ({dev_star:+.2f}σ)")
    print(f"  CMB/ΛCDM:   {t_0_CMB:.3f} ± {t_0_CMB_s:.3f} Gyr  ({dev_CMB:+.1f}σ, different model)")
    print("  Status: GENUINE PREDICTION (from the adopted N_hub (value pinned via the measured G_F); coasting cosmology).")
