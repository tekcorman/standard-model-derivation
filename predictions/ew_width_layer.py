# ============================================================
# PARAMETER: δ_EW^width — the EW radiative layer on the α-form
#            golden rule (Z-width and W-width variants)
# ============================================================
# THE LOOP PROGRAM's R-V output (2026-07-02), registered by user gate.
# This leaf is the SINGLE SOURCE for the layer applied by
# predictions/Gamma_Z_over_M_Z.py and predictions/Gamma_W_over_Gamma_Z.py.
#
# --- WHAT THIS IS -------------------------------------------
# The multiplicative correction (1 + δ) between the frozen S3 α-form
# tree×QCD golden-rule assembly and the physical width fraction, for the
# Z (δ_Z) and W (δ_W) channels. NOT an independently-observed parameter
# (registered with obs = None, like δ_r): it is a derived LAYER of the
# width rows.
#
# --- DERIVATION CHAIN (every step pre-registered, git-witnessed) ------
#   C2  (DN_C2_vertex_loop_class_2026-07-02.py, pre-reg 2188fbe): R-V's
#       class = the CAR-KMS matter loop on the P3 vertex forms; conditional
#       on the P3/PS identification its content is STANDARD EW ⟹
#       SM-REPRODUCTION-CONDITIONAL.
#   V1  (LOOP_V1_car_kms_calibration_2026-07-02.py, pre-reg a5287f4): the
#       machinery calibrated (Veltman Δρ symbolic-exact; 1/(12π) lock
#       1e-14) and the EVALUATION RULE DERIVED — the KMS loop family has
#       exactly two parameter-free evaluations; the arrow (the
#       already-counted bit) selects the retarded VACUUM loop; thermality
#       enters as statistics only.
#   V2  (LOOP_V2_rv_blind_evaluation_2026-07-02.py, pre-reg d37a679): the
#       layer computed BLIND — extracted from the certified PDG-2024
#       worked example against the SHIPPED α-form tree at the PDG MS̄
#       point, applied at framework leaves with all input-drift
#       sensitivities bounded (|ΔS| < 0.012 loop units). LANDED on the
#       pre-registered demand (−1.81 vs −1.62 ± 0.34 loop units, pull
#       −0.54); all falsification surfaces held.
#   USER GATE 2026-07-02: registration approved ("Let's do the
#       registration properly").
#
# --- GRADE (parameter_linter.md vocabulary) ------------------
# STRUCTURAL-DERIVATION-CONDITIONAL / SM-REPRODUCTION-CONDITIONAL,
# Clause 9(9b) bridge tag EXPLICIT: the layer's numerical content is
# continuum-loop (π-transcendental over K) — K-RATIONALITY OF THE LAYER IS
# BROKEN AND ACKNOWLEDGED. This row is NOT theorem-grade and can never be
# promoted past bridge-conditional until the loop coefficient is derived
# natively (the interacting sector coupling / walk↔Fock dictionary at
# theorem grade — incomplete_equations_todo.md §7).
#
# DISTINCTION from the retracted Clause-9 exemplar (f878f82, retracted
# 4ce4d5c — Sirlin Δr pasted onto M_Z as a closure): that was a VALUE
# import with no derivation chain ("importing a value that moves a value
# is an oxymoron" — user ruling 2026-07-02). Here the MEASURE (C0), the
# EVALUATION RULE (V1: the vacuum loop, forced), the CLASS (C2 +
# T-ID1/T-ID2: standard EW on the derived site table), and the VALUE (V2:
# blind, pre-registered tier rule, surfaces gated) are each derived or
# certified, and the residual served as the pre-registered falsification
# target — the class could have died and did not. What remains imported
# is the standard-EW loop-formula content certified on ONE named worked
# example — the same Type-3 class as the golden rule's 1/(48π) and 1.409
# already in the frozen assembly.
#
# --- INPUTS --------------------------------------------------
# [external] — the certified worked example (PDG 2024 EW review,
# rev-standard-model, 31 May 2024; archived at
# docs/references/pdg2024_rev_standard_model.pdf; fetched 2026-07-02):
# symbol         | value    | where
# ---------------|----------|---------------------------------
# Γ_Z^SM         | 2.4940 GeV ± 0.0009 | Eq. (10.78); Table 10.6 (2494.00 ± 0.87 MeV)
# Γ_W^SM         | 2.0892 GeV ± 0.0008 | Eq. (10.78)
# M_Z^fit        | 91.1884 GeV         | Table 10.7 / SM-fit column
# m_W^SM         | 80.356 GeV          | SM-fit column
# ŝ²_Z           | 0.23129 ± 0.00004   | Table 10.2 (MS̄)
# 1/α̂⁽⁵⁾(M_Z)   | 127.930 ± 0.008     | §10.2 (MS̄, with α_s = 0.1187)
# α_s(M_Z)^fit   | 0.1187 ± 0.0017     | global fit
# ρ_t scaling    | 0.00934·(m_t/172.61)² | Eq. (10.23); fit-SM m_t = 172.85
# Γ_bb̄^SM       | 375.73 MeV          | Table 10.6 (b-share for ΔS)
# Table 10.6 per-channel rows are asserted to reassemble Γ_had and Γ_Z
# below (< 0.1 MeV) — the transcription certification.
# [derived] m_t | 172.41 | m_t.py | framework top mass (ΔS b-vertex drift)
# Formula-structure constants (48π, 2Qs² couplings, 1.409) = the SAME
# declared Type-3 import already carried by the width rows.
#
# --- DERIVED FORMULA -----------------------------------------
# δ_Z = [Γ_Z^SM / M_Z^fit] / [α-form tree×QCD at the PDG MS̄ point] − 1 + ΔS
# δ_W = [Γ_W^SM / m_W^SM]  / [g²·9/(48π) × QCD_W at the PDG MS̄ point] − 1
# ΔS  = −(4/3)·ρ_t^ref·[(m_t^fw/172.61)² − (172.85/172.61)²]·(Γ_b/Γ_Z)^SM
#       (the b-vertex m_t² drift — the only input-difference term above
#        1e-5; the s²-curvature, α_s-tail, α̂ and M_H-log terms are
#        BOUNDED < 0.012 loop units total in the V2 probe and not applied)
# The α-form tree here is a REPLICA of the frozen S3 assembly; the two
# width files assert replica ≡ their own pure functions at 1e-14 on
# import (the anti-drift weld).

import functools
import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from m_t import m_t_pred                                # noqa: E402

# --- the certified worked example ([external]; provenance in header) ---
GAMMA_Z_SM_GEV = 2.4940
GAMMA_W_SM_GEV = 2.0892
MZ_FIT_GEV     = 91.1884
MW_SM_GEV      = 80.356
S2_HAT_PDG     = 0.23129
INV_ALPHA_HAT  = 127.930
ALPHA_S_FIT    = 0.1187
RHO_T_REF      = 0.00934
MT_REF_GEV     = 172.61
MT_FITSM_GEV   = 172.85
GAMMA_BB_SM_MEV = 375.73
# Table 10.6 rows (MeV) for the transcription certification:
_T106 = dict(ee=83.955, mumu=83.955, tautau=83.772, inv=501.435,
             uu=299.87, cc=299.81, dd=382.75, ss=382.75, bb=375.73,
             had=1740.88, total=2494.00)
assert abs(_T106['uu'] + _T106['cc'] + _T106['dd'] + _T106['ss'] + _T106['bb']
           - _T106['had']) < 0.1, "Table 10.6 hadronic transcription drift"
assert abs(_T106['ee'] + _T106['mumu'] + _T106['tautau'] + _T106['inv']
           + _T106['had'] - _T106['total']) < 0.1, "Table 10.6 total transcription drift"

G2SQ_PDG = 4 * math.pi * (1.0 / INV_ALPHA_HAT) / S2_HAT_PDG   # MS̄ g₂² at the PDG point


def _tree_ratio_alpha_form(g2sq, s2, a_s, k_star=3, n_gen=3, n_up_open=2):
    """REPLICA of the frozen S3 α-form tree×QCD Γ_Z/M_Z assembly (the exact
    Gamma_Z_over_M_Z.py structure; consumers assert equality at 1e-14)."""
    tot, had = 0.0, 0.0
    for n in range(k_star + 1):
        sgn = (-1) ** n
        T3, Q, Nc = sgn / 2, sgn * n / k_star, math.comb(k_star, n)
        gens = n_up_open if n == 2 else n_gen
        w = gens * Nc * ((T3 - 2 * Q * s2) ** 2 + T3 ** 2)
        tot += w
        if 0 < n < k_star:
            had += w
    x = a_s / math.pi
    return (g2sq / (1 - s2)) * tot / (48 * math.pi) \
        * (1 + (had / tot) * (x + 1.409 * x * x))


def _tree_ratio_W(g2sq, a_s, n_channels=9, n_had=6):
    """the frozen S3 W-side tree: Γ_W/m_W = g²·n_ch/(48π) × QCD_W."""
    x = a_s / math.pi
    return g2sq * n_channels / (48 * math.pi) \
        * (1 + (n_had / n_channels) * (x + 1.409 * x * x))


# --- PURE FUNCTIONS ------------------------------------------
@functools.lru_cache(maxsize=None)
def predict_ew_width_layer_Z(gamma_z_sm_over_mz, tree_ratio_pdg, delta_s):
    """δ_Z: the EW radiative layer on the α-form Z golden rule.

    Parameters
    ----------
    gamma_z_sm_over_mz : float  the worked example's Γ_Z^SM/M_Z^fit
    tree_ratio_pdg : float      the frozen tree×QCD assembly at the PDG MS̄ point
    delta_s : float             the framework-vs-PDG input-drift correction

    Returns
    -------
    float : δ_Z (apply as tree×QCD×(1+δ_Z))
    """
    return gamma_z_sm_over_mz / tree_ratio_pdg - 1 + delta_s


@functools.lru_cache(maxsize=None)
def predict_ew_width_layer_W(gamma_w_sm_over_mw, tree_w_ratio_pdg):
    """δ_W: the EW radiative layer on the α-form W golden rule."""
    return gamma_w_sm_over_mw / tree_w_ratio_pdg - 1


@functools.lru_cache(maxsize=None)
def predict_delta_s_bvertex(rho_t_ref, mt_fw, mt_ref, mt_fitsm, b_share):
    """ΔS: the b-vertex m_t² drift of the layer between the PDG-fit point
    and the framework top mass (Eq. 10.55 structure: ρ̂_b ≈ 1 − 4/3·ρ_t)."""
    return -(4.0 / 3.0) * rho_t_ref * ((mt_fw / mt_ref) ** 2
                                       - (mt_fitsm / mt_ref) ** 2) * b_share


# --- IMPLEMENTATION ------------------------------------------
_tree_pdg_Z = _tree_ratio_alpha_form(G2SQ_PDG, S2_HAT_PDG, ALPHA_S_FIT)
_tree_pdg_W = _tree_ratio_W(G2SQ_PDG, ALPHA_S_FIT)
_delta_s = predict_delta_s_bvertex(RHO_T_REF, m_t_pred, MT_REF_GEV, MT_FITSM_GEV,
                                   GAMMA_BB_SM_MEV / (_T106['total']))

ew_width_layer_Z_pred = predict_ew_width_layer_Z(
    GAMMA_Z_SM_GEV / MZ_FIT_GEV, _tree_pdg_Z, _delta_s)
ew_width_layer_W_pred = predict_ew_width_layer_W(
    GAMMA_W_SM_GEV / MW_SM_GEV, _tree_pdg_W)
# the registry/lock headline value = the Z layer (the R-V number); the W layer
# is locked implicitly through Gamma_W_over_Gamma_Z_pred
ew_width_layer_pred = ew_width_layer_Z_pred

# the V2 gate carried over: the applied drift term must stay far under band
_LOOP_UNIT_SCALE = G2SQ_PDG / (4 * math.pi) / (4 * math.pi)   # α₂/4π at the PDG point
assert abs(_delta_s) / _LOOP_UNIT_SCALE < 0.1, "Delta_S left its V2 gate — re-audit"

if __name__ == "__main__":
    print("δ_EW^width — the α-form EW radiative layer (LOOP program R-V, registered)")
    print(f"  δ_Z = {ew_width_layer_Z_pred*100:+.4f}%   "
          f"(= {ew_width_layer_Z_pred/_LOOP_UNIT_SCALE:+.2f} loop units α₂/4π)")
    print(f"  δ_W = {ew_width_layer_W_pred*100:+.4f}%")
    print(f"  ΔS (b-vertex m_t² drift, applied) = {_delta_s:+.2e}")
    print("  Grade: STRUCTURAL-DERIVATION-CONDITIONAL / SM-REPRODUCTION-CONDITIONAL")
    print("  (Clause 9b bridge tag; K-rationality of the layer broken, acknowledged;")
    print("   native derivation = the interacting sector coupling, todo §7 — OPEN).")
    # regression guards against the V2-banked values (comparison-only literals)
    assert abs(ew_width_layer_Z_pred - (-0.4864e-2)) < 2e-5, "δ_Z drifted from the V2-banked value"
    assert abs(ew_width_layer_W_pred - (-0.0787e-2)) < 2e-5, "δ_W drifted from the V2-banked value"
    print("OK: layer values match the V2-banked probe outputs.")
