# ============================================================
# PARAMETER: Γ_W / Γ_Z — ratio of the W and Z total decay widths
# ============================================================
# The framework's FIRST WIDTH OBSERVABLE (F4 S3, 2026-07-02).
#
# --- OBSERVED VALUE ------------------------------------------
# Value:       Γ_W/Γ_Z = 0.83560 ± 0.01685   (±2.0%)
# Source:      PDG 2024: Γ_W = 2.085 ± 0.042 GeV, Γ_Z = 2.4952 ± 0.0023 GeV
# PDG edition: 2024
#
# --- PREDICTED VALUE -----------------------------------------
# Value:       0.83801   (tree ratio 0.83460 × (1+δ_W)/(1+δ_Z);
#              δ_W = −0.0787%, δ_Z = −0.4864% — ew_width_layer.py)
# Deviation:   +0.00241 absolute, +0.29% relative, +0.14σ_PDG
#              (pre-layer −0.06σ; the layer differential is +0.41% — the
#              κ̂/b-vertex content has no W analog; the sub-σ criterion
#              holds comfortably against the ±2.0% measurement)
#
# --- GRADE (parameter_linter.md vocabulary) ------------------
# MATHEMATICALLY COMPLETE / STRUCTURAL-DERIVATION-CONDITIONAL per Clause 9(9b):
# the golden-rule STRUCTURE (tree width Γ_f = g²·M·N_c(v²+a²)/(48π), Peskin &
# Schroeder §20/PDG EW review, and the QCD series 1 + a_s/π + 1.409(a_s/π)²,
# Chetyrkin–Kühn–Kwiatkowski 1996) is a Type-3 SM import whose 1/(48π) is
# π-transcendental over K — NOT theorem-grade closable until the Clifford-native
# phase space is derived (incomplete_equations_todo.md §7; the band-geometric
# route is CLOSED — F4_cone_spectral_function probe, kill branch). Every NUMBER
# feeding the formula is a framework read. Clause 8: +0.14σ ≤ 1σ_PDG PASS (8c).
# EW LAYER REGISTERED 2026-07-02 (user gate; joint with the Γ_Z/M_Z companion):
# the LOOP program's derived rate-side layer applies to BOTH widths; in this
# ratio the common normalization cancels and only the differential
# (1+δ_W)/(1+δ_Z) survives — the S4-pattern surface gated in the V2 probe
# (proofs/foundations/LOOP_V2_rv_blind_evaluation_2026-07-02.py, Row 3).
#
# --- WIDTH-SIDE DARK: NONE, BY THEOREM -----------------------
# proofs/foundations/F4_S2b_width_ratio_dark_lemma_2026-07-02.py (CAS): a REAL
# multiplicative dressing cancels in Γ/M and in common width ratios EXACTLY,
# and the gauge sector's matching-point dark reads the exactly-real Perron
# channel; a complex-pole shell dressing is stability-excluded (×1.6e16 vs μ).
# Applying dark corrections to this ratio is therefore FORBIDDEN, not omitted.
# (The registered EW layer is NOT a dark dressing: it is per-channel loop
# content — exactly the part that does NOT cancel in the ratio.)
#
# --- ASSEMBLY (frozen S3 tree×QCD + the REGISTERED derived layer) -----
# tree × own-α_s QCD (frozen 2026-07-02 BEFORE comparison; probe S6 of
# F4_width_math_verification_2026-07-02.py) × (1+δ_W)/(1+δ_Z). The layer
# differential REPLACES the former stated-not-applied "EW ρ_f/vertex layer
# (largely common ⟹ suppressed)" estimate — that suppression estimate was
# WRONG on size (actual differential +0.41%, not ≲0.1%: the κ̂-shift and
# Z→bb̄ vertex have no W analog) and is corrected by the derived layer;
# the miss is recorded in the V2 probe and the loop-kickoff banner.
# Still stated-not-applied inside the ratio (bundled per-width in δ):
# |V_CKM| beyond row unitarity (= 0 exactly by unitarity).
#
# --- DERIVED FORMULA -----------------------------------------
# Γ_W/Γ_Z = [ N_W · c² / Σ_Z(s²) ] · (m_W/M_Z) · [ QCD_W / QCD_Z ]
#   N_W    = n_gen + N_c·(n_gen − 1)  open W channels (lepton doublets + CKM
#            row-unitary quark doublets; the top row is CLOSED: m_t > m_W)
#   Σ_Z(s²)= Σ_species,gens N_c·(v_f² + a_f²), v = T₃ − 2Q s², a = T₃,
#            top channel closed (m_t > M_Z/2, framework's own m_t)
#   fermion content from the Cl(6)-Fock read: for occupation n ∈ {0..k*}:
#            Q = (−1)ⁿ·n/k*, T₃ = (−1)ⁿ/2, multiplicity N(n) = C(k*, n)
#            (n = 0 ν, 1 d, 2 u, 3 e; color triplet = C(3,1) = C(3,2) = 3)
#   NOTE g₂ cancels in the ratio (g² in W vs g²/c² in Z).
#
# --- INPUTS --------------------------------------------------
# symbol   | value      | status    | predictions/ file            | meaning
# ---------|------------|-----------|------------------------------|-----------------
# s²(M_Z)  | 0.23125    | [derived] | sin2_theta_W_MZ.py           | weak mixing angle at M_Z
# α_s(M_Z) | 0.1179     | [derived] | alpha_s.py                   | strong coupling at M_Z
# m_W      | 80.4010    | [derived] | m_W.py                       | W pole mass (GeV)
# M_Z      | 91.2039    | [derived] | M_Z.py                       | Z pole mass (GeV)
# m_t      | 172.41     | [derived] | m_t.py                       | top mass (channel closure only)
# k*       | 3          | [derived] | k_star.py                    | coordination (Cl(6) Fock)
# n_gen    | 3          | [derived] | R3_observer_c3_generation.py | generation count
# δ_W, δ_Z | −0.0787%, −0.4864% | [derived-bridge] | ew_width_layer.py | the registered EW layer
#          |            |           |                              | ([external] certified PDG-2024
#          |            |           |                              | worked example inside the leaf)
# Formula-structure constants (48π, the 2Q s² vector coupling, 1.409) belong to
# the imported Type-3 SM structure declared above — they are part of the FORMULA,
# not tunable inputs.  G_F is NOT used (non-circular vs the width data; G_F is
# itself calibrated from τ_μ, untouched here).

import functools
import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from sin2_theta_W_MZ import sin2_theta_W_MZ            # noqa: E402
from alpha_s import alpha_s_MZ                          # noqa: E402
from m_W import m_W_pred                                # noqa: E402
from M_Z import M_Z_GeV                                 # noqa: E402
from m_t import m_t_pred                                # noqa: E402
from g_2 import g_2_MZ                                  # noqa: E402  (weld assert only)
from k_star import predict_k_star                       # noqa: E402
from d_spatial import predict_d_spatial                 # noqa: E402
from R3_observer_c3_generation import R3_observer_c3_generation_pred  # noqa: E402
import ew_width_layer as _ewl                           # noqa: E402

# --- IMPLEMENTATION ------------------------------------------
_k_star = predict_k_star(predict_d_spatial())
_n_gen = int(R3_observer_c3_generation_pred)

# channel-closure reads (framework's own masses; booleans, not fits)
assert m_t_pred > M_Z_GeV / 2 and m_t_pred > m_W_pred, "top closure premise failed"
_n_up_open = _n_gen - 1                                 # u, c open; t closed


def _species(k_star):
    """Cl(6)-Fock species read: (T3, Q, N_color) for occupation n = 0..k*."""
    out = []
    for n in range(k_star + 1):
        sgn = (-1) ** n
        out.append((sgn / 2, sgn * n / k_star, math.comb(k_star, n)))
    return out


def _sum_Z(s2, k_star, n_gen, n_up_open):
    """Σ_f N_c(v²+a²) over open Z channels (top closed)."""
    tot, had = 0.0, 0.0
    for n, (T3, Q, Nc) in enumerate(_species(k_star)):
        gens = n_up_open if n == 2 else n_gen           # up-type: top closed
        w = gens * Nc * ((T3 - 2 * Q * s2) ** 2 + T3 ** 2)
        tot += w
        if 0 < n < k_star:                               # colored (quark) channels
            had += w
    return tot, had / tot


def _qcd(a_s, had_frac):
    x = a_s / math.pi
    return 1 + had_frac * (x + 1.409 * x * x)


def _implementation():
    S_Z, had_Z = _sum_Z(sin2_theta_W_MZ, _k_star, _n_gen, _n_up_open)
    c2 = 1 - sin2_theta_W_MZ
    Nc_quark = math.comb(_k_star, 1)
    n_W = _n_gen + Nc_quark * _n_up_open                 # 3 + 3·2 = 9 open W channels
    had_W = Nc_quark * _n_up_open / n_W
    tree = n_W * c2 / S_Z * (m_W_pred / M_Z_GeV)
    tree_qcd = tree * _qcd(alpha_s_MZ, had_W) / _qcd(alpha_s_MZ, had_Z)
    # the registered EW layer: only the W-vs-Z DIFFERENTIAL survives in the ratio
    return tree_qcd * (1 + _ewl.ew_width_layer_W_pred) \
        / (1 + _ewl.ew_width_layer_Z_pred), tree_qcd


# --- PURE FUNCTION -------------------------------------------
@functools.lru_cache(maxsize=None)
def predict_Gamma_W_over_Gamma_Z(s2_MZ, a_s_MZ, mW_over_MZ, k_star, n_gen, n_up_open):
    """
    Γ_W/Γ_Z from the frozen tree×QCD golden-rule assembly (Type-3 structure)
    with all numerical content from framework reads.

    Parameters
    ----------
    s2_MZ : float        sin²θ_W at M_Z (framework RG endpoint)
    a_s_MZ : float       α_s at M_Z (framework)
    mW_over_MZ : float   framework m_W / M_Z pole-mass ratio
    k_star : int         coordination number (Cl(6) Fock rank)
    n_gen : int          generation count
    n_up_open : int      open up-type generations at these energies

    Returns
    -------
    float : predicted Γ_W/Γ_Z
    """
    c2 = 1 - s2_MZ
    tot, had_num = 0.0, 0.0
    for n in range(k_star + 1):
        sgn = (-1) ** n
        T3, Q, Nc = sgn / 2, sgn * n / k_star, math.comb(k_star, n)
        gens = n_up_open if n == 2 else n_gen
        w = gens * Nc * ((T3 - 2 * Q * s2_MZ) ** 2 + T3 ** 2)
        tot += w
        if 0 < n < k_star:
            had_num += w
    Nc_q = math.comb(k_star, 1)
    n_W = n_gen + Nc_q * n_up_open
    x = a_s_MZ / math.pi
    qcd_W = 1 + (Nc_q * n_up_open / n_W) * (x + 1.409 * x * x)
    qcd_Z = 1 + (had_num / tot) * (x + 1.409 * x * x)
    return n_W * c2 / tot * mW_over_MZ * qcd_W / qcd_Z


@functools.lru_cache(maxsize=None)
def predict_Gamma_W_over_Gamma_Z_dressed(tree_qcd_ratio, ew_layer_W, ew_layer_Z):
    """the registered full prediction: the EW-layer DIFFERENTIAL on the ratio."""
    return tree_qcd_ratio * (1 + ew_layer_W) / (1 + ew_layer_Z)


# --- ANTI-DRIFT WELD -----------------------------------------
# the layer leaf's W-side tree replica must equal this file's W-channel
# structure: Γ_W/m_W = g²·n_W/(48π)·QCD_W (same 9-channel, 6-hadronic form)
assert abs(_ewl._tree_ratio_W(g_2_MZ ** 2, alpha_s_MZ)
           / (g_2_MZ ** 2 * 9 / (48 * math.pi) * _qcd(alpha_s_MZ, 6.0 / 9.0)) - 1) < 1e-14, \
    "layer-leaf W tree replica drifted from the shipped W-channel structure"

# --- VALIDATION ----------------------------------------------
Gamma_W_over_Gamma_Z_obs = 2.085 / 2.4952
Gamma_W_over_Gamma_Z_sigma = Gamma_W_over_Gamma_Z_obs * math.sqrt(
    (0.042 / 2.085) ** 2 + (0.0023 / 2.4952) ** 2)
Gamma_W_over_Gamma_Z_pred, Gamma_W_over_Gamma_Z_tree_pred = _implementation()

if __name__ == "__main__":
    impl = Gamma_W_over_Gamma_Z_pred
    pure = predict_Gamma_W_over_Gamma_Z_dressed(
        predict_Gamma_W_over_Gamma_Z(sin2_theta_W_MZ, alpha_s_MZ,
                                     m_W_pred / M_Z_GeV, _k_star, _n_gen, _n_up_open),
        _ewl.ew_width_layer_W_pred, _ewl.ew_width_layer_Z_pred)
    sig = (impl - Gamma_W_over_Gamma_Z_obs) / Gamma_W_over_Gamma_Z_sigma
    sig_tree = (Gamma_W_over_Gamma_Z_tree_pred - Gamma_W_over_Gamma_Z_obs) \
        / Gamma_W_over_Gamma_Z_sigma
    print("Γ_W/Γ_Z — first width observable (F4 S3 assembly + the LOOP-program EW")
    print("          layer differential, registered 2026-07-02; Clause 9b)")
    print(f"  Implementation: {impl:.5f}   (tree×QCD {Gamma_W_over_Gamma_Z_tree_pred:.5f}; "
          f"layer differential {((1+_ewl.ew_width_layer_W_pred)/(1+_ewl.ew_width_layer_Z_pred)-1)*100:+.2f}%)")
    print(f"  Pure function:  {pure:.5f}")
    assert abs(impl - pure) < 1e-12, f"Mismatch: {impl} vs {pure}"
    print(f"  Observed:       {Gamma_W_over_Gamma_Z_obs:.5f} ± {Gamma_W_over_Gamma_Z_sigma:.5f}")
    print(f"  Deviation:      {(impl/Gamma_W_over_Gamma_Z_obs-1)*100:+.2f}%  ({sig:+.2f}σ_PDG)"
          f"  [pre-layer: {sig_tree:+.2f}σ]")
    assert abs(sig) < 1.0, "Γ_W/Γ_Z left the 1σ band — investigate before relabeling"
    print("OK: within 1σ_PDG (Clause 8c PASS). Grade: MATHEMATICALLY COMPLETE /")
    print("    bridge-conditional (Clause 9b: 1/(48π) + EW layer Type-3; native open).")
    print("    No dark applied — forbidden by the S2b lemma, not omitted.")
