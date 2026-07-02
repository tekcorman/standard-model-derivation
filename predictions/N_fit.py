#!/usr/bin/env python3
"""
Canonical identification file for N_fit (best-fit toggle graph node count).

Audit anchor: alternative anchoring for Row P17 (N_hub) of
`docs/parameters/parameter_uniqueness_ledger.md`. N_fit aggregates multiple observables
to estimate N; superseded by Session 19's G_F-calibration of N_hub's value (700× precision; N_hub is the adopted input
improvement). Retained as historical / cross-check.

Performs a weighted least-squares combination of all observables that depend
on N (the toggle graph node count) via the BZJ formula, to produce a best-fit
N with propagated uncertainty and a residuals table.

Grade: identification (combines adopted N from multiple observables; not a
new prediction).

STATUS: tracking (N_hub adopted; this file consolidates the evidence).

STATUS UPDATE 2026-04-19 session 2: References below to
"ADOPTED-I-FESHBACH" should be read as "AXIOM A5(b)" — the
coupling clause of A5 (docs/framework/framework_axioms.md §5b) subsumes this
identification. ADOPTED-DARK-MAP and N_hub G1 gap remain separate.
"""

# ============================================================
# PARAMETER: N_fit (weighted least-squares best-fit node count)
# ============================================================

# --- THEOREM STATEMENT ---------------------------------------
# N_fit is the weighted least-squares estimate of the toggle graph node count N
# from all observables that constrain it via the BZJ formula:
#
#   v = δ² M_P / (√2 N^{1/4}) × (1 - (5/12)α₁)      [BZJ + dark correction]
#
# Inverted:
#   N = (δ² M_P × dark / (√2 × v))^4
#
# where dark = 1 - (5/12)α₁.
#
# For each observable i, we compute N_i and σ_{N_i} via error propagation,
# then form:
#   w_i   = 1 / σ_{N_i}²
#   N_fit = Σ(w_i × N_i) / Σ(w_i)
#   σ_fit = 1 / sqrt(Σ(w_i))
#   χ²    = Σ(w_i × (N_i - N_fit)²)
#
# Status: identification (N_hub is the adopted input; the measured G_F provides
#         the tightest constraint; H_0 rows and m_H row add context).

# --- OBSERVED VALUES AND OBSERVABLES -------------------------
# 1. G_F (MuLan/PDG 2024)
#    G_F = 1.1663787e-5 GeV^{-2}, σ_GF = 6e-12 GeV^{-2} (0.51 ppm)
#    the *value* of the adopted N_hub is calibrated via the measured G_F (so its uncertainty inherits 0.5·σ_GF/G_F via the v↔G_F relation; G_F is downstream of N_hub)
#    σ_N/N = 4 σ_v/v = 2 σ_GF/G_F ≈ 1.03 ppm
#
# 2. H_0 (Planck 2018 CMB)
#    N = 1/(H_0 × t_P),  σ_N/N = σ_H0/H0 = 0.742%
#
# 3. H_0 (Riess et al. distance ladder)
#    N = 1/(H_0 × t_P),  σ_N/N = σ_H0/H0 = 1.37%
#    [included to show H_0 tension]
#
# 4. m_H (LHC, λ-contaminated)
#    v = m_H / sqrt(2λ),  σ_N/N = 4 σ_mH/mH ≈ 0.352%
#    [experimental uncertainty only; λ carries additional systematic]

# --- INPUTS --------------------------------------------------
# symbol     | value                | status     | predictions/ file              | meaning
# -----------|----------------------|------------|--------------------------------|--------
# delta      | 2/9                  | [derived]  | h_walker_eigenvalue.py         | Koide phase from Z_3 rate-distortion
# alpha_1    | (2/3)^8              | [derived]  | alpha_1.py                     | bare NB walk survival, k*=3, g=10
# M_P        | M_Pl_natural.M_Pl_GeV | [derived] | predictions/M_Pl_natural.py | M_Pl/M_subst=8/√π theorem-grade; GeV=single declared SI-anchor, Gap-G1 (CODE imports it line 110 — NOT hardcoded; was falsely "[external]|none")
# t_P        | 5.391247e-44 s       | [external] | N_hub.py                       | Planck time (CODATA 2018)
# G_F        | 1.1663787e-5 GeV^-2  | [PREDICTION] | G_F.py                       | Fermi constant — a PREDICTION (= 1/(√2 v²), v ← N_hub via BZJ); its measured value pins N_hub's adopted value to ppm
# H0_CMB     | 67.4 km/s/Mpc        | [external] | N_hub.py                       | Hubble constant (Planck 2018 CMB)
# H0_ladder  | 73.0 km/s/Mpc        | [external] | —                              | Hubble constant (distance ladder; Riess et al.)
# m_H        | 125.20 GeV           | [external] | m_H.py                         | Higgs boson mass (PDG 2025)
# lambda     | 2560/19683           | [derived]  | lambda_higgs.py                | Higgs quartic coupling (ADOPTED-I-FESHBACH + ADOPTED-DARK-MAP)

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_1 import predict_alpha_1
from k_star import predict_k_star
from d_spatial import predict_d_spatial
from g_girth import predict_g_girth
from h_walker_eigenvalue import predict_h_walker_eigenvalue
from srs_E_at_P import predict_srs_E_at_P
from lambda_higgs import predict_lambda_higgs
from p_toggle import predict_p_toggle
import functools

# --- chain imports ---
d       = predict_d_spatial()
k       = predict_k_star(d)
E       = predict_srs_E_at_P(k)
p       = predict_p_toggle()
h       = predict_h_walker_eigenvalue(k, E, p)
g       = predict_g_girth(k, d)
alpha_1 = predict_alpha_1(k, g)
from V_count import predict_V_count
N_atoms_srs = predict_V_count(k, d)   # = 4, Wyckoff 8a per srs primitive cell
lam     = predict_lambda_higgs(alpha_1, h, n_channels=2, n_H_legs=4, n_F_legs=0,
                                  N_atoms=N_atoms_srs, k_star=k)

# --- external constants (Gap G1) ---
from delta_Koide import delta_Koide_pred as delta  # = 2/9 (Q*(1-Q) at Q=2/3, predict_delta_Koide)       # Koide phase [derived]
from M_Pl_natural import M_Pl_GeV as M_P, t_P_seconds as t_P, Mpc_in_km   # single SI-anchor source
# t_P + Mpc_in_km now sourced from M_Pl_natural single-source (consolidated
# 2026-05-16 for t_P, 2026-05-26 for Mpc_in_km).

# --- dark correction factor (geometric series: all windings, A2-waterline) ---
# c_vertex = 5/12: 5 = k_star + p_toggle (framework Hashimoto sum);
# 12 = k_star · V_count (= 2|E| handshake).
c_vertex = (k + p) / (k * N_atoms_srs)
dark = float(p - 1) - c_vertex * alpha_1 / (float(p - 1) - alpha_1)

# --- BZJ pivot factor ---
# v = (delta^2 * M_P * dark) / (sqrt(2) * N^{1/4})
# => N^{1/4} = delta^2 * M_P * dark / (sqrt(2) * v)
# => N = (delta^2 * M_P * dark / (sqrt(2) * v))^4

def _v_to_N(v_GeV, delta_, M_P_, dark_):
    """Convert Higgs VEV to N using the BZJ inversion."""
    factor = (delta_**2 * M_P_ * dark_) / (math.sqrt(2) * v_GeV)
    return factor**4

def _sigma_N_from_sigma_v_rel(N_val, sigma_v_rel):
    """σ_N = 4 * N * σ_v/v  (logarithmic propagation: N ∝ v^{-4})."""
    return 4.0 * N_val * sigma_v_rel

# ---- Observable 1: G_F (MuLan/PDG 2024) ----
G_F_obs   = 1.1663787e-5    # GeV^{-2}
G_F_sigma = 6.0e-12         # GeV^{-2}  (0.51 ppm)
# v ← the adopted N_hub via BZJ; the tree-level SM relation G_F = 1/(sqrt(2)*v^2) then gives the predicted G_F (which matches the measured value by construction)  => v = 1/(2 G_F sqrt(2))^{1/2}
v_from_GF = 1.0 / math.sqrt(math.sqrt(2) * G_F_obs)
# σ_v/v = 0.5 * σ_GF/G_F
sigma_v_rel_GF = 0.5 * (G_F_sigma / G_F_obs)
N_GF    = _v_to_N(v_from_GF, delta, M_P, dark)
sigma_N_GF = _sigma_N_from_sigma_v_rel(N_GF, sigma_v_rel_GF)

# ---- Observable 2: H_0 (Planck 2018 CMB) ----
H0_CMB       = 67.4    # km/s/Mpc
sigma_H0_CMB = 0.5     # km/s/Mpc
H0_CMB_per_s = H0_CMB / Mpc_in_km  # convert km/s/Mpc -> s^{-1} (km cancels)
N_H0_CMB    = 1.0 / (H0_CMB_per_s * t_P)
sigma_N_H0_CMB = N_H0_CMB * (sigma_H0_CMB / H0_CMB)

# ---- Observable 3: H_0 (distance ladder; Riess et al.) ----
H0_ladder       = 73.0    # km/s/Mpc
sigma_H0_ladder = 1.0     # km/s/Mpc
H0_ladder_per_s = H0_ladder / Mpc_in_km  # s^{-1} (km cancels)
N_H0_ladder    = 1.0 / (H0_ladder_per_s * t_P)
sigma_N_H0_ladder = N_H0_ladder * (sigma_H0_ladder / H0_ladder)

# ---- Observable 4: m_H (LHC, λ-contaminated) ----
m_H_obs   = 125.20   # GeV
m_H_sigma = 0.11     # GeV
# v from m_H: m_H = sqrt(2*lambda)*v  => v = m_H / sqrt(2*lambda)
v_from_mH = m_H_obs / math.sqrt(2.0 * lam)
sigma_v_rel_mH = m_H_sigma / m_H_obs   # σ_v/v = σ_mH/mH
N_mH    = _v_to_N(v_from_mH, delta, M_P, dark)
sigma_N_mH = _sigma_N_from_sigma_v_rel(N_mH, sigma_v_rel_mH)

# ---- Weighted least squares ----
def _wls(N_vals, sigma_N_vals):
    """Weighted least-squares combination. Returns (N_fit, sigma_fit, chi2)."""
    w = [1.0 / s**2 for s in sigma_N_vals]
    W = sum(w)
    N_fit_ = sum(wi * Ni for wi, Ni in zip(w, N_vals)) / W
    sigma_fit_ = 1.0 / math.sqrt(W)
    chi2_ = sum(wi * (Ni - N_fit_)**2 for wi, Ni in zip(w, N_vals))
    return N_fit_, sigma_fit_, chi2_

observables = [
    ("G_F (MuLan)",    N_GF,        sigma_N_GF),
    ("H0 (Planck CMB)",N_H0_CMB,    sigma_N_H0_CMB),
    ("H0 (dist. lad.)",N_H0_ladder, sigma_N_H0_ladder),
    ("m_H (LHC)",      N_mH,        sigma_N_mH),
]

N_vals      = [o[1] for o in observables]
sigma_N_vals = [o[2] for o in observables]

N_fit_val, sigma_N_fit, chi2 = _wls(N_vals, sigma_N_vals)

# ---- Weights and weight fractions ----
weights      = [1.0 / s**2 for s in sigma_N_vals]
W_total      = sum(weights)
weight_fracs = [w / W_total for w in weights]

# ---- Residuals ----
residuals = [(Ni - N_fit_val) / sNi for Ni, sNi in zip(N_vals, sigma_N_vals)]

# ---- Implied observable values at N_fit ----
# H_0 from N_fit
H0_pred_per_s = 1.0 / (N_fit_val * t_P)
H0_pred_km_s_Mpc = H0_pred_per_s * Mpc_in_km
# v from N_fit
v_pred = delta**2 * M_P * dark / (math.sqrt(2) * N_fit_val**0.25)
# G_F from v
G_F_pred = 1.0 / (math.sqrt(2) * v_pred**2)
# m_H from v and lambda
m_H_pred = math.sqrt(2.0 * lam) * v_pred

# ---- σ propagation for implied values at N_fit ----
# v ∝ N^{-1/4}  =>  σ_v/v = (1/4) σ_N/N
sigma_v_pred = v_pred * (sigma_N_fit / N_fit_val) / 4.0
# G_F ∝ v^{-2}  =>  σ_GF/GF = 2 σ_v/v
sigma_GF_pred = G_F_pred * 2.0 * (sigma_v_pred / v_pred)
# m_H ∝ v       =>  σ_mH = σ_v × (m_H/v)
sigma_mH_pred = sigma_v_pred * (m_H_pred / v_pred)

# ---- Print output ----
print("=" * 65)
print("  N_fit — weighted least squares over N-dependent observables")
print("=" * 65)
print()
print("  BZJ formula:  v = δ² M_P / (√2 N^{1/4}) × dark")
print(f"  dark = 1 - (5/12)α₁/(1−α₁) = {dark:.10f}")
print(f"  δ = 2/9 = {delta:.10f},  M_P = {M_P:.5e} GeV")
print(f"  α₁ = (2/3)^8 = {alpha_1:.10f}")
print(f"  λ  = 2560/19683 = {lam:.10f}  [for m_H row]")
print()
print(f"  {'Observable':<17s}  {'N_i (×10^60)':>14s}  {'σ_N/N':>9s}  {'weight frac':>12s}")
print(f"  {'-'*17}  {'-'*14}  {'-'*9}  {'-'*12}")
labels = ["G_F (MuLan)",
          "H₀ (Planck CMB)",
          "H₀ (dist. lad.)",
          "m_H (LHC)"]
notes  = ["dominant",
          "small",
          "tiny",
          "moderate (λ-sys)"]
for i, (lbl, Ni, sNi, wf) in enumerate(zip(labels, N_vals, sigma_N_vals, weight_fracs)):
    sig_rel = sNi / Ni
    print(f"  {lbl:<17s}  {Ni/1e60:>14.6f}  {sig_rel*100:>8.4f}%  {wf*100:>10.4f}%"
          f"  [{notes[i]}]")
print()
print(f"  Best-fit N_fit = {N_fit_val/1e60:.6f} ± {sigma_N_fit/1e60:.6f} × 10^60")
print(f"  Dominated by: G_F  (weight fraction: {weight_fracs[0]*100:.1f}%)")
print()
print(f"  χ² = {chi2:.4f}  (3 d.o.f.  [4 obs − 1 param])")
print()
print("  Residuals (N_i − N_fit) / σ_{N_i}:")
for lbl, r, note in zip(labels, residuals, notes):
    print(f"    {lbl:<17s}  {r:+.3f} σ")
print(f"    [H₀ tension: CMB vs distance ladder residuals differ by "
      f"{abs(residuals[1]-residuals[2]):.1f} σ]")
print()
print("  Implied values at N_fit:")
print(f"    H₀_pred   = {H0_pred_km_s_Mpc:.4f} ± {sigma_N_fit/N_fit_val/4*H0_pred_km_s_Mpc:.4f} km/s/Mpc"
      f"   (obs CMB: 67.4 ± 0.5)")
print(f"    v_pred    = {v_pred:.4f} ± {sigma_v_pred:.4f} GeV"
      f"     (obs PDG: 246.22 ± 0.12)")
print(f"    G_F_pred  = {G_F_pred:.7e} GeV^-2"
      f"  (obs: 1.1663787 × 10^-5)")
print(f"    m_H_pred  = {m_H_pred:.4f} ± {sigma_mH_pred:.4f} GeV"
      f"      (obs PDG: 125.20 ± 0.11)")
print()
print("  Notes:")
print("    - v_direct (PDG 246.22 GeV) corresponds to the adopted N_hub via BZJ; the predicted G_F matches the measured value by construction (not an independent test).")
print("    - m_H row uses λ = 2560/19683 (ADOPTED-I-FESHBACH + ADOPTED-DARK-MAP).")
print("    - H₀ tension visible in residuals: CMB vs distance-ladder rows.")
print("    - σ on H₀_pred is propagated from σ_N only; not a new H₀ prediction.")
print("=" * 65)


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_N_fit(G_F, sigma_GF, H0_CMB, sigma_H0_CMB, m_H, sigma_mH,
                  H0_ladder, sigma_H0_ladder,
                  delta, M_P, alpha_1,
                  lam, t_P_s,
                  H0_km_s_Mpc_conversion,
                  k_star, V_count, p_toggle):
    # parameter_linter.md: NO default argument values — every input
    # explicitly passed by the caller (was: H0_ladder=73.0,
    # M_P=1.22089e19, delta=2/9, alpha_1=None, lam=None,
    # t_P_s=5.39e-44, conv=3.0857e19 — all removed 2026-05-16).
    """
    Weighted least-squares estimate of the toggle graph node count N.

    The BZJ formula v = δ² M_P / (√2 N^{1/4}) × (1 - (5/12)α₁/(1−α₁)) is
    inverted for each observable to obtain N_i and σ_{N_i}, then combined
    via weighted least squares.

    Parameters
    ----------
    G_F : float
        Fermi constant in GeV^{-2} (PDG 2024 / MuLan).
    sigma_GF : float
        Uncertainty on G_F in GeV^{-2}.
    H0_CMB : float
        Hubble constant from Planck CMB in km/s/Mpc.
    sigma_H0_CMB : float
        Uncertainty on H0_CMB in km/s/Mpc.
    m_H : float
        Higgs boson mass in GeV (PDG 2025).
    sigma_mH : float
        Uncertainty on m_H in GeV.
    H0_ladder : float, optional
        Hubble constant from distance ladder in km/s/Mpc (default: 73.0).
    sigma_H0_ladder : float, optional
        Uncertainty on H0_ladder in km/s/Mpc (default: 1.0).
    delta : float, optional
        Koide phase (default: 2/9 exactly).
    M_P : float, optional
        Planck mass in GeV (default: 1.22089e19; CODATA 2018).
    alpha_1 : float or None, optional
        Bare NB walk survival probability (default: (2/3)^8).
    lam : float or None, optional
        Higgs quartic coupling (default: 2560/19683).
    t_P_s : float, optional
        Planck time in seconds (default: 5.391247e-44).
    H0_km_s_Mpc_conversion : float, optional
        Conversion factor: 1 Mpc in km (default: 3.085677581e19).

    Returns
    -------
    tuple
        (N_fit, sigma_N_fit, residuals_dict, chi2)
        N_fit, sigma_N_fit : floats (dimensionless node count and 1-sigma uncertainty)
        residuals_dict : dict mapping observable name -> (N_i, sigma_N_i, pull)
        chi2 : float
    """
    import math

    # (no internal defaults: alpha_1 and lam are required params,
    #  passed by the caller from the framework-derived module values —
    #  the prior `if … is None: <hardcoded>` fallbacks were a
    #  pure-function violation and are removed.)

    # derived constants — sourced from leaves
    from c_vertex_dark import predict_c_vertex_dark
    one_nb = p_toggle - 1                          # = 1, NB constraint
    c_vertex = float(predict_c_vertex_dark(k_star, V_count, p_toggle))  # = 5/12
    dark_ = one_nb - c_vertex * alpha_1 / (one_nb - alpha_1)   # geometric series

    def v_to_N_(v_):
        factor_ = (delta**p_toggle * M_P * dark_) / (math.sqrt(p_toggle) * v_)
        return factor_**V_count                    # = ⁴√ inverse: V_count = 4 BZJ exponent

    def sigma_N_from_rel_(N_, rel_):
        return float(V_count) * N_ * rel_           # = 4·N·σ_rel (BZJ Δσ propagation)

    # --- Observable 1: G_F ---
    v_gf  = float(one_nb) / math.sqrt(math.sqrt(p_toggle) * G_F)
    sv_gf = (float(one_nb)/p_toggle) * (sigma_GF / G_F)   # 0.5 = (p-1)/p
    N1    = v_to_N_(v_gf)
    sN1   = sigma_N_from_rel_(N1, sv_gf)

    # --- Observable 2: H_0 CMB ---
    H0c_per_s = H0_CMB / H0_km_s_Mpc_conversion  # km cancels (was *1e3/(*1e3))
    N2        = 1.0 / (H0c_per_s * t_P_s)
    sN2       = N2 * (sigma_H0_CMB / H0_CMB)

    # --- Observable 3: H_0 distance ladder ---
    H0l_per_s = H0_ladder / H0_km_s_Mpc_conversion  # km cancels
    N3        = 1.0 / (H0l_per_s * t_P_s)
    sN3       = N3 * (sigma_H0_ladder / H0_ladder)

    # --- Observable 4: m_H ---
    v_mh  = m_H / math.sqrt(2.0 * lam)
    sv_mh = m_H_sigma / m_H if False else sigma_mH / m_H  # σ_v/v = σ_mH/mH
    N4    = v_to_N_(v_mh)
    sN4   = sigma_N_from_rel_(N4, sv_mh)

    # --- weighted least squares ---
    N_list  = [N1, N2, N3, N4]
    sN_list = [sN1, sN2, sN3, sN4]
    names   = ["G_F", "H0_CMB", "H0_ladder", "m_H"]

    w_list = [1.0 / s**2 for s in sN_list]
    W      = sum(w_list)
    Nfit   = sum(w * N for w, N in zip(w_list, N_list)) / W
    sigma_ = 1.0 / math.sqrt(W)
    chi2_  = sum(w * (N - Nfit)**2 for w, N in zip(w_list, N_list))

    resid = {name: (Ni, sNi, (Ni - Nfit) / sNi)
             for name, Ni, sNi in zip(names, N_list, sN_list)}

    return Nfit, sigma_, resid, chi2_


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    pure_Nfit, pure_sigma, pure_resid, pure_chi2 = predict_N_fit(
        G_F=1.1663787e-5,
        sigma_GF=6e-12,
        H0_CMB=67.4,
        sigma_H0_CMB=0.5,
        m_H=125.20,
        sigma_mH=0.11,
        H0_ladder=73.0,
        sigma_H0_ladder=1.0,
        delta=delta,
        M_P=M_P,
        alpha_1=alpha_1,
        lam=lam,
        t_P_s=t_P,
        H0_km_s_Mpc_conversion=Mpc_in_km,
        k_star=k, V_count=N_atoms_srs, p_toggle=p,
    )

    print()
    print(f"Implementation N_fit = {N_fit_val:.6e}")
    print(f"Pure function  N_fit = {pure_Nfit:.6e}")
    assert abs(N_fit_val - pure_Nfit) / N_fit_val < 1e-10, \
        f"N_fit mismatch: {N_fit_val} vs {pure_Nfit}"
    assert abs(sigma_N_fit - pure_sigma) / sigma_N_fit < 1e-10, \
        f"sigma mismatch: {sigma_N_fit} vs {pure_sigma}"
    print("OK: implementation and pure function agree.")
    print()
    print(f"  N_fit        = {pure_Nfit/1e60:.6f} × 10^60")
    print(f"  σ_N_fit      = {pure_sigma/1e60:.6f} × 10^60")
    print(f"  χ²           = {pure_chi2:.4f}  (3 d.o.f.)")
    print(f"  G_F pull     = {pure_resid['G_F'][2]:+.3f} σ")
    print(f"  H₀ CMB pull  = {pure_resid['H0_CMB'][2]:+.3f} σ")
    print(f"  H₀ dist pull = {pure_resid['H0_ladder'][2]:+.3f} σ")
    print(f"  m_H pull     = {pure_resid['m_H'][2]:+.3f} σ  (λ-sys not included)")
    print()
    print("  Grade: identification (adopted N from multiple observables).")
    print("  Dominant constraint: G_F (MuLan 2024, 0.51 ppm).")
    print("  Open gap G1: genuine N derivation from A1-A4 required to")
    print("  convert this identification into a theorem-grade prediction.")
