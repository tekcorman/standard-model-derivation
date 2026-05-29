#!/usr/bin/env python3
"""
run_predictions.py — Import all prediction modules and write predicted_parameters.md.

Mirrors the sector structure of docs/parameters/target_parameters.md.
With @functools.lru_cache on every predict_* function, shared sub-expressions
are computed once regardless of how many modules import them.

Usage:
    python3 run_predictions.py
Output:
    predicted_parameters.md  (in repo root)
"""

import importlib
import sys
import os
import math
import re
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------
# Manifest: (display_symbol, module_slug, observed, sigma, units, notes)
# observed / sigma = None for structural / framework-internal entries.
# module_slug = None for parameters without a predictions/ file yet.
# ---------------------------------------------------------------

SECTORS = [
    ("Standard Model — Gauge couplings", [
        ("α_GUT",    "alpha_GUT",    1/24.3,          None,    "(dimensionless)", "dark-corrected 1/24.329 vs MSSM back-extrap 1/24.3±0.5 (bare substrate value 1/24)"),
        ("sin²θ_W (M_unif)", "sin2_theta_W", None,    None,    "(dimensionless)", "3/8 EXACT THEOREM at M_unif (no direct obs; session 25)"),
        ("sin²θ_W (M_Z)", "sin2_theta_W_MZ", 0.23121, 0.00004, "(dimensionless)", "RG from 3/8 at M_unif (THEOREM-GRADE-CONDITIONAL 2026-05-04 EOD+1)"),
        ("g_1 (GUT-norm)", "g_1",   0.46144,         0.0001,  "(dimensionless)", "U(1)_Y RG from α_GUT (THEOREM-GRADE-CONDITIONAL 2026-05-04 EOD+1). Observed = PDG-derived √(4π·(5/3)·α_Y); manifest fallback corrected W2 2026-05-18 from stale 0.4626 (runner uses g_1.py g_1_obs regardless; +0.37σ honest)."),
        ("g_2",      "g_2",          0.6520,          0.0001,  "(dimensionless)", "SU(2)_L RG from α_GUT (THEOREM-GRADE-CONDITIONAL 2026-05-04 EOD+1)"),
        ("g_3",      "g_3",          1.218,           0.005,   "(dimensionless)", "SU(3)_c RG from α_GUT (THEOREM-GRADE-CONDITIONAL 2026-05-04 EOD+1)"),
        ("α_s (M_Z)","alpha_s",      0.1180,          0.0009,  "(dimensionless)", "α_3 = g_3²/4π (THEOREM-GRADE-CONDITIONAL 2026-05-04 EOD+1)"),
        ("α_EM (M_Z)", "alpha_EM",   1.0/127.944,     0.014/127.944**2, "(dimensionless)", "RG run from M_unif (THEOREM-GRADE-CONDITIONAL 2026-05-04 EOD+1)"),
        ("M_Z",      "M_Z",          91.1876,         0.0021,  "GeV",   "Z-boson mass; self-consistent (THEOREM-GRADE-CONDITIONAL 2026-05-04 EOD+1)"),
        ("m_W",      "m_W",          80.369,          0.013,   "GeV",   "W-boson mass; m_W = M_Z·cos(θ_W) (Row P71, THEOREM-GRADE-CONDITIONAL 2026-05-04 EOD+2)"),
        ("δ_r",      "delta_r",      None,            None,    "(dimensionless)", "M_Z tree→pole oblique correction (Row P64-sibling; substrate Δr-analog); δ_r = (1/12)·α₁/(1−α₁); Z/Perron eigen-channel of the unified-oblique G_NB resolvent (theorem_unified_oblique.md 2026-05-16, c_S derived as 1/(2|E|)); THEOREM-GRADE-STRUCTURAL — Clause-7 fully closed via §6.1 resummation derivation. Companion of δρ."),
        ("δρ",       "delta_rho",    0.010429,        0.00063, "(dimensionless)", "ρ−1 custodial-breaking shift (Row P73); δρ = (1/2)·(√5/4)·(2/3)⁸ = +1.0906%. W/h_P eigen-channel of the unified-oblique G_NB resolvent. obs = m_W²/(M_Z²cos²θ_W)−1 from PDG 2024. MATHEMATICALLY COMPLETE (Clause 7 PASS via 2026-05-17 Leading-Order Uniqueness Closure); Clause 8 +0.76σ_obs PASS. +4.58% relative is the deep-layer §2 object, NOT a residual of the prediction."),
        ("M_unif",   "M_unif",       2.0e16,          0.5e16,  "GeV",   "Gauge unification scale (THEOREM-GRADE-CONDITIONAL 2026-05-04 EOD+1, 5-stage closure)"),
        # R∞ REMOVED FROM predictions/ 2026-05-28 — not theorem-grade. R∞ = α(0)²·m_e·c/(2h)
        # needs the atomic-frame α(0); the α(M_Z)→α(0) bridge (Δα) is un-derivable
        # (continuum/Clause-9; substrate analog BLOCKED). The shipped file reached its
        # "match" only by hardcoding delta_alpha_running=9.092 (a Clause-9 smuggle). Moved
        # to proofs/cosmology/R_infinity.py pending a derived Δα. Clean in-scope EM test = α_EM(M_Z).
    ]),
    ("Standard Model — Higgs sector", [
        ("v_higgs",  "v_higgs",      246.22,          0.12,    "GeV",  "Higgs VEV"),
        ("m_H",      "m_H",          125.20,          0.11,    "GeV",  "Higgs mass"),
        ("λ_H",      "lambda_higgs", 0.1294,          None,    "(dimensionless)", "Higgs quartic"),
        ("λ_3",      "lambda_3_higgs", 31.8314,       0.0714,  "GeV",   "Higgs trilinear self-coupling λ_3 = m_H²/(2v) (theorem-grade descendant of m_H + v with Family D)"),
        ("G_F",      "G_F",          1.1663787e-5,    6e-12,   "GeV⁻²", "Fermi constant"),
    ]),
    ("Standard Model — Charged fermion masses", [
        ("m_u",      "m_u",          2.16e-3,         0.49e-3, "GeV",  "M_persistence-shipped 2026-05-26 (commit c9fba27); THEOREM-GRADE-STRUCTURAL-CONDITIONAL via the 12×12 fermion mass operator (Row P39). Within 1σ_PDG."),
        ("m_d",      "m_d",          4.67e-3,         0.48e-3, "GeV",  "M_persistence-shipped 2026-05-26; THEOREM-GRADE-STRUCTURAL-CONDITIONAL (Row P39). Within 1σ_PDG."),
        ("m_s",      "m_s",          93.4e-3,         8.6e-3,  "GeV",  "M_persistence-shipped 2026-05-26; THEOREM-GRADE-STRUCTURAL-CONDITIONAL (Row P39). Within 1σ_PDG."),
        ("m_c",      "m_c",          1.27,            0.02,    "GeV",  "M_persistence-shipped 2026-05-26; THEOREM-GRADE-STRUCTURAL-CONDITIONAL (Row P39). Within 1σ_PDG."),
        ("m_b",      "m_b",          4.18,            0.03,    "GeV",  "M_persistence-shipped 2026-05-26; THEOREM-GRADE-STRUCTURAL-CONDITIONAL (Row P39). Relative +2.1%, Clause-8 borderline (+2.99σ_PDG); residual is MSSM-threshold + two-loop class."),
        ("m_t",      "m_t",          172.69,          0.30,    "GeV",  "M_persistence-shipped 2026-05-26 (Row P38; previously slug=None with retracted Koide-waterfall in predictions/retracted/m_top.py). Live chain ships via M_persistence + Type-II saturation y_t(GUT)=1 + MSSM RGE: THEOREM-GRADE-STRUCTURAL-CONDITIONAL. Relative +0.82%; Clause-8 FAIL on σ_PDG (+4.71σ), residual is MSSM-threshold + two-loop class."),
        ("m_e",      "m_e",          0.51099895e-3,   1.5e-13, "GeV",  "PRECISION-FLOOR: framework predicts via Koide ratio m_τ × (f_min/f_max)²; σ_PDG ~10⁻⁹ unreachable from integers, framework matches to ~10⁻⁴ relative (−0.008%). THEOREM-GRADE-STRUCTURAL conditional on G1 + y_τ Family-D c_F (W1 2026-05-18)."),
        ("m_μ",      "m_mu",         0.1056583755,    2.3e-9,  "GeV",  "PRECISION-FLOOR: same Koide-ratio chain m_τ × (f_mid/f_max)²; σ_PDG ~10⁻⁸ unreachable; relative −0.007%. THEOREM-GRADE-STRUCTURAL conditional (W1 2026-05-18)."),
        ("m_τ",      "m_tau",        1.77686,         0.00012, "GeV",  "m_τ = v × y_τ × Family-D; live 1.7768 GeV (−0.19σ_PDG). THEOREM-GRADE-STRUCTURAL conditional (W1 2026-05-18 reinstatement; prior 'UNIQUE-THEOREM-GRADE-NUMERICAL' Family-D was a Clause-6c smuggle per Row P11; value unchanged)."),
        ("y_τ",      "y_tau",        7.2165543e-3,    None,    "(dimensionless)", "α₁_full/k*² × Family-D; THEOREM-GRADE-STRUCTURAL conditional (W1 2026-05-18; c_F Clause-6 channel_select → canonical_encoding via dark_extraction_map _c_F_denominator)."),
    ]),
    ("Standard Model — CKM", [
        ("V_ud",     "V_ud",         0.97435,         0.00016, "(dimensionless)", "unitarity Type-4"),
        ("V_us",     "V_us",         0.22501,         0.00068, "(dimensionless)", ""),
        ("V_ub",     "V_ub",         0.00382,         0.00020, "(dimensionless)", "PDG combined exc+inc"),
        ("V_cd",     "V_cd",         0.22487,         0.00068, "(dimensionless)", "unitarity Type-4"),
        ("V_cs",     "V_cs",         0.97349,         0.00016, "(dimensionless)", "unitarity Type-4"),
        ("V_cb",     "V_cb",         0.0406,          0.0009,  "(dimensionless)", "PDG 2024 exclusive (Belle); ~3.3σ excl/incl tension"),
        ("V_td",     "V_td",         0.00854,         0.00023, "(dimensionless)", "unitarity Type-4"),
        ("V_ts",     "V_ts",         0.04110,         0.00083, "(dimensionless)", "unitarity Type-4"),
        ("V_tb",     "V_tb",         0.999118,        0.000031,"(dimensionless)", "unitarity Type-4"),
        ("J_CKM",    "J_CKM",        3.08e-5,         0.13e-5, "(dimensionless)", "Jarlskog Type-4"),
        ("δ_CP_CKM", "delta_CP_CKM", 68.5,            3.0,     "°",    ""),
        ("δ_CP_geom","delta_CP_CKM_geometry", 68.5,   3.0,     "°",    "geometry route"),
    ]),
    ("Standard Model — QCD", [
        ("θ_QCD",    "theta_QCD",    0.0,             1e-10,   "(dimensionless)", "strong CP"),
    ]),
    ("Neutrino sector", [
        ("m_ν2",     "m_nu2",        math.sqrt(7.49e-5),   0.5*0.19e-5/math.sqrt(7.49e-5),   "eV",   "√Δm²₂₁ (NuFIT 6.0)"),
        ("m_ν3",     "m_nu3",        math.sqrt(2.513e-3),  0.5*0.020e-3/math.sqrt(2.513e-3), "eV",   "√Δm²₃₁ (NuFIT 6.0)"),
        ("R_ν",      "R_nu_splitting", 32.576,        None,    "(dimensionless)", "Δm²₃₁/Δm²₂₁"),
        ("θ_12_PMNS","theta_12_PMNS", 33.41,          0.75,    "°",    "PS perp identity"),
        ("θ_13_PMNS","theta_13_PMNS", 8.57,           0.11,    "°",    "TBM + edge-local dark; +2σ canonical chain"),
        ("θ_23_PMNS","theta_23_PMNS", 49.2,           1.3,     "°",    ""),
        ("δ_CP_PMNS","delta_CP_PMNS", 177.0,          20.0,    "°",    "δ_CP_PMNS = arccos(T_{B-L,lepton}) = arccos(−1) = 180° via the parameter-free V_{−1}–T_{B-L} geometric identity (same identity gives δ_CP_CKM=arccos(1/3)=70.53°, +0.68σ — independent corroboration). THEOREM-GRADE-STRUCTURAL-CONDITIONAL (on Need-D-3 + the geometric↔Jarlskog adoption, Row P15-shared). SUPERSEDES the Hashimoto-phase (g−1)·arg(h*)≈249.85° route, which WAS falsified at +3.83σ vs NuFIT 6.0 IC19 (2026-05-02) — see honest_assessment.md item 3 (reconciled W3 2026-05-18). obs NuFIT 6.0 IC19 = 177°⁺¹⁹₋₂₀, +0.16σ; sigma=20° upper used."),
        ("α_21_PMNS","alpha_21_PMNS", None,           None,    "°",    "g·arg(h); UNIQUE-THEOREM-GRADE-CONDITIONAL via Path B (2026-05-04 EOD+1, unconstrained obs)"),
        ("α_31_PMNS","alpha_31_PMNS", None,           None,    "°",    "2g·arg(h); UNIQUE-THEOREM-GRADE-CONDITIONAL via Path B (2026-05-04 EOD+1, unconstrained obs)"),
        ("Q_Koide",  "Q_Koide",      2/3,             None,    "(dimensionless)", ""),
        ("ε_Koide",  "epsilon_Koide",math.sqrt(2),    None,    "(dimensionless)", ""),
        ("δ_Koide",  "delta_Koide",  2/9,             None,    "(dimensionless)", ""),
        ("N_eff",    "N_eff",        2.99,            0.17,    "(dimensionless)", "effective number of relativistic neutrino species; framework predicts N_eff = observer Hilbert-space dim = 3 exactly; Planck 2018 2.99±0.17 → +0.06σ"),
    ]),
    ("Cosmology", [
        ("Ω_DM",     "Omega_DM",     0.2645,          0.0050,  "(dimensionless)", "Row P23. Promoted to predictions/ 2026-05-15 EOD+5 under adopted-z_eff framing; MATHEMATICALLY-COMPLETE-CONDITIONAL-ON-ADOPTED-z_eff (N_hub-class). Live = Ω_m_LCDM × P22 ratio. Prior PS-seesaw chain retracted 2026-05-04 (see `predictions/retracted/Omega_DM.py`); current live chain is the z_eff-conditional adoption."),
        ("Ω_DM/Ω_m", "Omega_DM_over_Omega_m", 0.846, 0.016,   "(dimensionless)", "Row P22; UNIQUE-THEOREM-GRADE. Cl(2k*) Fock + Poisson(2k*) waterline; 1 − P(k≤k*|Poisson(2k*))."),
        ("Ω_b",      "Omega_b",      0.04930,         0.00046, "(dimensionless)", "Row P23 companion. Promoted to predictions/ 2026-05-15 EOD+5; MATHEMATICALLY-COMPLETE-CONDITIONAL-ON-ADOPTED-z_eff. Carries the known ~0.7% P22-partition residual not movable by z_eff."),
        ("Ω_m_LCDM", "Omega_m_LCDM", 0.3153,          0.0073,  "(dimensionless)", "Row P24 primary. MATHEMATICALLY-COMPLETE-CONDITIONAL-ON-ADOPTED-z_eff. Bias-function FORM Ω_m(z)=(u+1)/(u²+u+1) is theorem-grade K-rational; the framework's ONE adopted z_eff (N_hub-class) replaces ΛCDM's free Ω_m. K-rational anchor z=√3 → 1/3 exact."),
        ("Ω_Λ_LCDM", "Omega_Lambda_LCDM", 0.6847,     0.0073,  "(dimensionless)", "Row P24 sibling (Type-4 = 1 − Ω_m_LCDM). K-rational anchor z=√3 → 2/3 exact."),
        ("z_eff",    "z_eff",        1.916,           0.079,    "(dimensionless)", "ADOPTED cosmology parameter (N_hub-class) — value from survey Fisher GEOMETRY (SN+BAO survey design, not fit to distances). Observation-implied = invert bias function at Planck Ω_m; framework-adopted = 1.832 (SN+BAO Fisher first-moment). Cross-check vs obs-implied: −1.06σ."),
        ("H_0 (CMB side)", "H_0",     67.4,            0.5,     "km/s/Mpc", "Row P19. UNIQUE-THEOREM-GRADE post G1b R2 closure (2026-05-07 PM); substrate-side from coasting H_0·t_0=1. Framework's Clause-8 Category-B CMB-side anchor."),
        ("H_0 (observer)", "H_0_observer", 73.04,      1.04,    "km/s/Mpc", "Row P19 sibling. Observer-side via the D2-extended observer/substrate rate gap (16/15) = ε_toggle·(1/k*) = (1/5)(1/3) = 1/15. The Hubble tension is a STRUCTURAL PREDICTION, not an anomaly."),
        ("Λ_CC",     "Lambda_CC",    2.850e-122,      None,    "(Planck units)", "Row P24. UNIQUE-THEOREM-GRADE (graduated 🟡→✅ 2026-05-16). Clean substrate Λ = 1/N²; carries only the coasting + ADOPTED-N_HUB conditional (G1-cluster class). The factor-of-2 vs ΛCDM-fit is the parametric-class translation Λ_LCDM = 3·Ω_Λ_LCDM·Λ_substrate, handled in the sibling row Λ_LCDM below."),
        ("Λ_LCDM",   "Lambda_CC_LCDM", 2.84852e-122,  5.204e-124, "(Planck units)", "Row P24-sibling (NEW 2026-05-16, observable-side). MATHEMATICALLY-COMPLETE-CONDITIONAL-ON-ADOPTED-z_eff; Type-4 inheritance from Omega_Lambda_LCDM × Lambda_CC. Clause 8 +0.77σ_obs at adopted z_eff; −0.20σ_obs at K-rational anchor z=√3 (factor exactly 2)."),
        ("w_DE",     "w_DE",         -1.03,           0.03,    "(dimensionless)", "Row P21. UNIQUE-THEOREM-GRADE. Ratio of rate²-scaling quantities; (16/15)² cancels in ratio."),
        ("A_hemis",  "A_hemispherical", 0.07,         0.02,    "(dimensionless)", "CMB asymmetry"),
        ("η_B",      "eta_B",        6.12e-10,        0.04e-10, "(dimensionless)", "baryon-to-photon ratio (UNIQUE-THEOREM-GRADE 2026-04-30)"),
        ("ε_CP",     "epsilon_CP",   None,            None,    "(dimensionless)", "per-process baryon CP asymmetry = 1/5 exactly (ε_toggle Bayesian Beta(2,1)); feeds η_B (Row P28, UNIQUE-THEOREM-GRADE)"),
        ("t_0",      "t_0",          14.46,           0.80,    "Gyr",   "Age vs Methuselah HD 140283 (Bond 2013, model-independent — framework's Clause-8 Category-B anchor); coasting H_0·t_0=1. ΛCDM/CMB-frame t_0 is a tracked OPEN target (target_parameters + ledger Row P20-sibling) — no predictions/ file until the theorem exists (directive 2026-05-17)"),
        ("T_e±_ann", "T_e_ann",      None,            None,    "MeV",   "e⁺e⁻ annihilation temperature T = m_e/k* = m_e/3 ≈ 0.170 MeV; equals the Boltzmann m_e/3 convention exactly (k*=3 makes m_e/k* = m_e/3). No precise measurement — convention-anchored thermal-history milestone."),
        ("G_N·M_Pl²","G_N",          1.0,             None,    "(dimensionless)", "Newton's constant identity (THEOREM-GRADE-CONDITIONAL 2026-04-30)"),
    ]),
    ("Framework-adjacent observables", [
        ("β_birefringence", "beta_cosmic_birefringence", 0.342, 0.094, "°", "CMB cosmic birefringence (Eskilt 2022); UNIQUE-THEOREM-GRADE 2026-04-29 — β = c·sin(arg h)·α_EM"),
        ("E_transparency", "universe_transparency", None, None, "PeV", "Universe transparency onset at framework Hashimoto scale (~147 PeV); tentative GRB 221009A evidence"),
    ]),
    ("Structural / definitional", [
        ("k*",       "k_star",       3,               None,    "(dimensionless)", "coordination"),
        ("d_spatial","d_spatial",    3,               None,    "(dimensionless)", "spatial dim"),
        ("g_girth",  "g_girth",      10,              None,    "(dimensionless)", "girth"),
        ("p_toggle", "p_toggle",     2,               None,    "(dimensionless)", ""),
        ("|V|",      "V_count",      4,               None,    "(dimensionless)", "srs primitive cell vertex count / K_4 quotient (Sunada 2012 + MDL + Gleason chain)"),
        ("|E|",      "E_count",      6,               None,    "(dimensionless)", "srs primitive cell edge count = k·|V|/2 via handshake lemma"),
        ("N_gen",    "R3_observer_c3_generation", 3,  None,    "(dimensionless)", "generations"),
        ("H_obs (=C³)", "observer_dim_three", 3,      None,    "(dimensionless)", "observer Hilbert-space dimension; MDL + Gleason 1957 + A3 (CDP 2011)"),
    ]),
    ("Framework-internal", [
        ("M_Pl (lattice)", "M_Pl_natural", 8.0/math.sqrt(math.pi), None, "M_substrate units", "Planck mass as untethered structural prediction (8/√π exact); GeV value is unit conversion"),
        ("h_walker", "h_walker_eigenvalue", None,     None,    "complex",  "P-point eigenvalue"),
        ("srs_E_P",  "srs_E_at_P",   None,            None,    "(dimensionless)", "√3"),
        ("α₁_bare",  "alpha_1",      None,            None,    "(dimensionless)", "(2/3)^8"),
        ("α₁_full",  "alpha_1_full", None,            None,    "(dimensionless)", "(5/3)(2/3)^8"),
        ("λ_toggle", "lambda_toggle_rate", None,      None,    "(dimensionless)", ""),
        ("ξ_t",      "xi_t_temporal_correlation", None, None,  "ℓ_P",     ""),
        ("S_fresh",  "S_fresh",      None,            None,    "bits",    ""),
        ("S_disconf","S_disconfirm", None,            None,    "bits",    ""),
        ("η_5",      "eta_5_lorentz_dim5", 0.0,       0.1,    "(dimensionless)", "LIV dim-5"),
        ("η_lattice","eta_lattice_lorentz_dim6", None, None,   "(dimensionless)", "LIV dim-6"),
        ("E_scale",  "scale_energy_hashimoto", None,  None,    "PeV",     "Hashimoto scale"),
        ("srs_cubic","srs_cubic_moment", None,        None,    "(dimensionless)", ""),
        ("koide_qr", "koide_quark_ratio", 2.800,      None,    "(dimensionless)", "14/5"),
        ("N_hub",    "N_hub",        None,            None,    "(dimensionless)", "Hubble-Planck"),
        ("feshbach_exp", "feshbach_exponent_principle", None, None, "(dimensionless)", ""),
        ("e_bit",    "e_bit",        1.0,             None,    "(dimensionless)", "substrate edge-toggle energy primitive (definition-equivalent)"),
        ("GJ ratio", "georgi_jarlskog", 3.0,          1.0,     "(dimensionless)", "Georgi-Jarlskog = k* = 3 exact; ±1 GUT-RGE uncertainty"),
        ("η^H_NB",   "srs_bloch_lv_dim6", None,       None,    "(dimensionless)", "scalar Bloch dim-6 LV coefficient = 1/6 (CAS-verified; sister to η_lattice = 1/12)"),
        ("tan β",    "tan_beta",     None,            None,    "(dimensionless)", "MSSM ratio of Higgs VEVs. **Live RGE chain (predictions/tan_beta.py) computes tan β ≈ 60.07**; the documented framework value 44.73 from proofs/masses/srs_tan_beta.py disagrees by ~35%. The previous `except: return 44.73` fallback was masking this disagreement — surfaced 2026-05-26 by the literal-fallback audit. THEOREM-GRADE-STRUCTURAL-CONDITIONAL on RGE consistency, but the documented-proof-vs-live-chain reconciliation is open work (Row P46). Not directly observed."),
        ("c_vertex_dark", "c_vertex_dark", None,       None,    "(dimensionless)", "dark-correction vertex factor = (k+p)/(k·|V|) = 5/12; consumed by N_fit + N_hub chains"),
        ("b_i MSSM", "mssm_beta_coefficients", None,   None,    "(dimensionless)", "MSSM one-loop β-coefficients (b_1=33/5, b_2=1, b_3=−3) + hypercharge_norm (=3/5); GUT-normalized; consumed by alpha_EM, sin2_theta_W_MZ, M_Z, alpha_s, g_1/2/3"),
        ("Σ(h) map", "dark_extraction_map", None,      None,    "(dimensionless)", "C₃ × parity rep-theory map: HOW each observable couples to the dark self-energy Σ(h) = α₁/h; consumed by lambda_higgs, y_tau, theta_23_PMNS"),
        ("h(B,P)",   "B_P_doubly_degenerate_h", None,  None,    "complex", "P-point Bloch NB-walk doubly-degenerate eigenvalue h_P = (√3 + i√5)/2; consumed by beta_cosmic_birefringence"),
        ("Hilb(obs)","observer_hilbert_space", None,   None,    "(dimensionless)", "observer Hilbert-space structure under axioms A1+A2-T+A3-T; upstream of observer_dim_three"),
    ]),
]


# Plain-language names for each prediction slug. Single source of truth — used
# by viz/build_data.py and any other consumer that wants a non-jargon label.
# Keep in sync with SECTORS as new predictions are added.
PLAIN_NAMES: dict[str, str] = {
    "alpha_1":                    "bare survival probability of substrate walker",
    "alpha_1_full":               "dressed survival probability (geometric series sum)",
    "alpha_GUT":                  "unified gauge coupling at high-energy unification",
    "sin2_theta_W":               "weak mixing angle (squared) at unification",
    "v_higgs":                    "Higgs vacuum expectation value",
    "m_H":                        "Higgs boson mass",
    "lambda_higgs":               "Higgs self-interaction strength",
    "G_F":                        "Fermi constant (weak-interaction strength)",
    "m_top":                      "top quark mass",
    "m_e":                        "electron mass",
    "m_mu":                       "muon mass",
    "m_tau":                      "tau lepton mass",
    "y_tau":                      "tau Yukawa coupling",
    "V_us":                       "CKM up–strange mixing element",
    "V_cb":                       "CKM charm–bottom mixing element",
    "V_ub":                       "CKM up–bottom mixing element",
    "delta_CP_CKM":               "CP-violating phase (quark sector)",
    "delta_CP_CKM_geometry":      "CP-violating phase via tetrahedral geometry",
    "theta_QCD":                  "strong CP-violation angle (predicted exactly zero)",
    "m_nu2":                      "second neutrino mass eigenstate",
    "m_nu3":                      "third neutrino mass eigenstate",
    "R_nu_splitting":             "ratio of neutrino mass-squared splittings",
    "theta_23_PMNS":              "atmospheric neutrino mixing angle",
    "theta_12_PMNS":              "solar neutrino mixing angle",
    "theta_13_PMNS":              "reactor neutrino mixing angle",
    "Q_Koide":                    "Koide quadratic ratio for charged leptons",
    "epsilon_Koide":              "Koide spinor-amplitude parameter",
    "delta_Koide":                "Koide complementary parameter",
    "N_eff":                      "effective number of relativistic neutrino species",
    "T_e_ann":                    "e+e- annihilation temperature (thermal history)",
    "Omega_DM":                   "dark-matter fraction of universe energy",
    "Omega_DM_over_Omega_m":      "dark-matter fraction of total matter",
    "H_0":                        "Hubble constant (universe expansion rate today)",
    "w_DE":                       "dark energy equation-of-state parameter",
    "A_hemispherical":            "CMB hemispherical asymmetry amplitude",
    "eta_B":                      "baryon-to-photon ratio (matter–antimatter asymmetry)",
    "k_star":                     "lattice coordination number",
    "d_spatial":                  "number of spatial dimensions",
    "g_girth":                    "shortest cycle length (girth) of substrate",
    "p_toggle":                   "alphabet size of substrate states",
    "V_count":                    "srs primitive-cell vertex count (=4)",
    "E_count":                    "srs primitive-cell edge count (=6, via handshake)",
    "c_vertex_dark":              "dark-correction vertex factor (=5/12)",
    "mssm_beta_coefficients":     "MSSM one-loop β-function coefficients (b_1, b_2, b_3)",
    "dark_extraction_map":        "dark-sector coupling map per observable",
    "B_P_doubly_degenerate_h":    "P-point Bloch walk eigenvalue h_P",
    "observer_hilbert_space":     "observer Hilbert-space structure",
    "R3_observer_c3_generation":  "number of fermion generations from observer C₃ symmetry",
    "h_walker_eigenvalue":        "complex eigenvalue of substrate walker at high-symmetry point",
    "srs_E_at_P":                 "edge-amplitude eigenvalue at saddle point",
    "lambda_toggle_rate":         "substrate state-toggle rate",
    "xi_t_temporal_correlation":  "temporal correlation length",
    "S_fresh":                    "Shannon entropy of fresh substrate state",
    "S_disconfirm":               "disconfirmation entropy",
    "eta_5_lorentz_dim5":         "dimension-5 Lorentz-violation coefficient",
    "eta_lattice_lorentz_dim6":   "dimension-6 Lorentz-violation coefficient",
    "scale_energy_hashimoto":     "characteristic Hashimoto-operator energy scale",
    "srs_cubic_moment":           "third moment on substrate lattice",
    "koide_quark_ratio":          "Koide-style ratio for quark masses",
    "N_hub":                      "Hubble-to-Planck ratio (Planck times in age of universe)",
    "feshbach_exponent_principle":"Feshbach exponent unification principle",
    "G_N":                        "Newton's gravitational constant (unit-setting identity)",
    "M_Pl_natural":               "Planck mass in framework-natural lattice units (untethered structural prediction)",
    "M_unif":                     "gauge unification scale (where SU(3)×SU(2)×U(1) couplings meet)",
    "alpha_EM":                   "fine-structure constant at M_Z (electromagnetic coupling)",
    "M_Z":                        "Z-boson mass (self-consistent electroweak matching)",
    "sin2_theta_W_MZ":            "weak mixing angle squared at the Z-pole",
    "g_1":                        "U(1)_Y gauge coupling at M_Z (GUT-normalized)",
    "g_2":                        "SU(2)_L gauge coupling at M_Z",
    "g_3":                        "SU(3)_c gauge coupling at M_Z",
    "alpha_s":                    "strong coupling at M_Z",
    "alpha_21_PMNS":              "first PMNS Majorana phase",
    "alpha_31_PMNS":              "second PMNS Majorana phase",
    "beta_cosmic_birefringence":  "cosmic-microwave-background birefringence rotation angle",
    "m_W":                        "W-boson mass",
    "delta_CP_PMNS":              "CP-violating phase (lepton sector)",
    "t_0":                        "age of the universe",
    "universe_transparency":      "energy onset of universe transparency to UHE photons",
    "observer_dim_three":         "minimum viable observer Hilbert-space dimension",
    "e_bit":                      "substrate edge-toggle energy primitive",
    "georgi_jarlskog":            "Georgi-Jarlskog mass ratio (= k*)",
    "srs_bloch_lv_dim6":          "scalar-Bloch dim-6 Lorentz-violation coefficient",
}


def _find_result_vars(mod, slug):
    """
    Introspect a module for predicted/observed/sigma values.
    Returns (predicted, observed, sigma, dev_sigma) — any may be None.

    Prefix-matching strategy (fixes fallback bugs where a module has
    multiple *_obs variables — e.g., v_higgs.py has both v_obs and
    G_F_obs for its external G_F anchor):
      1. Try canonical {slug}_pred / {slug}_obs / {slug}_sigma exactly.
      2. If predicted is found, strip its "_pred" suffix to get a prefix
         (e.g. "v_pred" → "v"), then prefer {prefix}_obs and {prefix}_sigma.
      3. If still missing, fall back to any *_pred / *_obs / *_sigma
         (shortest name wins — heuristic for "main quantity").
    """
    mv = vars(mod)
    slug_upper = slug.upper()

    def _get_float(name):
        if name in mv:
            v = mv[name]
            if isinstance(v, complex):
                return v
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        return None

    def _first_float_matching(suffix, exclude=()):
        """Among module vars ending in `suffix`, try by shortest name first."""
        cands = [k for k in mv if k.endswith(suffix) and not k.startswith("_")
                 and k not in exclude]
        cands.sort(key=len)
        for k in cands:
            try:
                return float(mv[k]), k
            except (TypeError, ValueError):
                continue
        return None, None

    def _first_not_none(*names):
        for n in names:
            v = _get_float(n)
            if v is not None:
                return v
        return None

    # --- Step 1: canonical names ---
    # Clause 8 is evaluated against σ_PDG only; we use *_sigma (PDG-only).
    predicted = _first_not_none(f"{slug}_pred", f"{slug_upper}_pred")
    observed  = _first_not_none(f"{slug}_obs",  f"{slug_upper}_obs")
    sigma     = _first_not_none(f"{slug}_sigma", f"{slug_upper}_sigma")
    dev_sig   = _get_float("dev_sigma")

    # --- Step 1b: bare slug-name (e.g. theta_QCD = 0, alpha_1 = ..., w_DE = -1) ---
    if predicted is None:
        predicted = _first_not_none(slug, slug_upper)

    # --- Step 2: predicted fallback (shortest *_pred) ---
    pred_prefix = None
    if predicted is not None and f"{slug}_pred" in mv:
        pred_prefix = slug
    elif predicted is None:
        val, name = _first_float_matching("_pred")
        if val is not None:
            predicted = val
            pred_prefix = name[:-5]  # strip "_pred"

    # --- Step 3: prefix-matched observed / sigma (uses pred_prefix) ---
    if observed is None and pred_prefix:
        observed = _get_float(f"{pred_prefix}_obs")
    if sigma is None and pred_prefix:
        sigma = _get_float(f"{pred_prefix}_sigma")

    # (Step 4 last-resort fallback removed: it cross-grabbed unrelated *_obs
    # like G_F_obs from N_hub.py. Modules with a legitimate observed value
    # expose it as <slug>_obs (Step 1) or share a prefix with <slug>_pred
    # (Step 3). Other observed values come from the manifest.)

    # --- Step 5: recompute dev_sigma if we have all three (real-only) ---
    if (dev_sig is None and predicted is not None and observed is not None
            and sigma and not isinstance(predicted, complex)
            and not isinstance(observed, complex)):
        dev_sig = (predicted - observed) / sigma

    return predicted, observed, sigma, dev_sig


def _load_module(slug):
    """Import predictions/{slug}.py, return module or None on failure.
    Suppresses stdout/stderr during import (modules print on load)."""
    mod_name = f"predictions.{slug}"
    try:
        devnull = open(os.devnull, "w")
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            mod = importlib.import_module(mod_name)
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            devnull.close()
        return mod
    except Exception:
        return None


def _fmt(val, observed=None):
    """Format a float for display."""
    if val is None:
        return "—"
    if isinstance(val, complex):
        return f"{val.real:.6f}+{val.imag:.6f}i"
    try:
        f = float(val)
    except (TypeError, ValueError):
        return str(val)
    if observed is not None and observed != 0:
        mag = abs(observed)
    else:
        mag = abs(f) if f != 0 else 1.0
    if mag == 0:
        return "0"
    if 1e-4 < mag < 1e5:
        sig = max(4, -int(math.floor(math.log10(mag))) + 4)
        return f"{f:.{sig}g}"
    return f"{f:.6e}"


def _sigma_str(dev, predicted=None, observed=None):
    """Format sigma deviation. When |σ| > 10 (dominated by tight PDG error,
    as for lepton masses), also show relative % for readability."""
    if dev is None:
        return "—"
    if abs(dev) > 10 and predicted is not None and observed is not None and observed != 0:
        rel_pct = (predicted - observed) / observed * 100
        return f"{dev:+.1f}σ ({rel_pct:+.3f}%)"
    return f"{dev:+.2f}σ"


def build_report():
    lines = [
        "# Predicted Parameters",
        "",
        "Generated by `run_predictions.py` from all `predictions/*.py` modules.",
        "Columns: **Predicted** (computed), **Observed** (PDG/experiment), **Δ/σ** (sigma pull).",
        "Mirroring structure of `docs/parameters/target_parameters.md`.",
        "",
    ]

    for sector_name, params in SECTORS:
        lines.append(f"## {sector_name}")
        lines.append("")
        lines.append("| Symbol | Predicted | Observed | Δ/σ | Units | Notes |")
        lines.append("|--------|-----------|----------|-----|-------|-------|")

        for entry in params:
            symbol, slug, obs_manifest, sigma_manifest, units, notes = entry

            predicted = None
            observed  = obs_manifest
            sigma     = sigma_manifest
            dev_sig   = None
            status    = ""

            if slug is not None:
                mod = _load_module(slug)
                if mod is None:
                    status = "⚠️ import error"
                else:
                    p, o, s, d = _find_result_vars(mod, slug)
                    if p is not None:
                        predicted = p
                    if o is not None:
                        observed = o  # prefer module-level over manifest
                    if s is not None:
                        sigma = s
                    if d is not None:
                        dev_sig = d

                    # recompute dev_sigma if missing; or recover predicted from dev_sigma
                    if dev_sig is None and predicted is not None and observed is not None and sigma:
                        dev_sig = (predicted - observed) / sigma
                    elif predicted is None and dev_sig is not None and observed is not None and sigma:
                        predicted = observed + dev_sig * sigma
            else:
                status = "↩ retracted" if notes.startswith("RETRACTED") else "❌ no file"

            pred_str = _fmt(predicted, observed)
            obs_str  = _fmt(observed)
            sig_str  = _sigma_str(dev_sig, predicted, observed)
            if status:
                pred_str = status

            lines.append(f"| {symbol} | {pred_str} | {obs_str} | {sig_str} | {units} | {notes} |")

        lines.append("")

    lines.append("---")
    lines.append(f"*Auto-generated. Do not edit manually.*")
    return "\n".join(lines)


if __name__ == "__main__":
    print("Running predictions... (lru_cache avoids redundant computation)")
    report = build_report()

    out_path = os.path.join(ROOT, "predicted_parameters.md")
    with open(out_path, "w") as f:
        f.write(report)

    print(f"Written: {out_path}")
    print()
    # Quick summary
    lines = report.split("\n")
    for line in lines:
        if line.startswith("## ") or ("|" in line and "Symbol" not in line and "---" not in line):
            print(line)
