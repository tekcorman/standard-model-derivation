#!/usr/bin/env python3
"""
Canonical prediction file for the electron mass m_e (ratio prediction from m_tau).

Audit anchor: Row P11 of `docs/parameters/parameter_uniqueness_ledger.md`.
m_e = m_τ × (f_min/f_max)² is exact-rational GIVEN the Koide structure
(Q_Koide = 2/3 ⟺ ε = √2; phase δ = 2/9). The substrate-ABSOLUTE route
(mass as a B_NB persistence/flux object, the framework's intended
mechanism, proofs/masses/lindblad_steady_state_at_P) does NOT supply the
generation hierarchy: the 2026-05-18 first-step test showed the unified-
oblique resolvent's spectral channels span at most ~4× whereas the e/μ/τ
hierarchy needs 3477× — falsified by three orders of magnitude. The
generation splitting is provably NOT in the spectral channels; it lives
in the C³_gen / multiway index structure (R-15 / Need-D-3), which is
unsolved. The Koide relation is therefore an ADOPTED phenomenological
input that supplies the hierarchy the substrate does not derive — not a
substrate derivation.

Dark correction: NONE applies (master doc §5). All three charged-lepton
Yukawa vertices are identical 1H+2F, so the Family-D factor −(5/6)α₁² is
generation-independent and divides out EXACTLY in any mass ratio
(verified this session). The residual is therefore not a missing /
Feshbach dark correction.

Clause 8 (σ_PDG only): FAIL — see PREDICTED VALUE for the honest
decomposition. Grade: ratio exact-rational given adopted Koide;
absolute scale mathematically-complete-conditional on G1 (v←N_hub←G_F,
circular calibration).
"""

# ============================================================
# PARAMETER: m_e (electron mass) -- ratio prediction from m_tau
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       m_e = 0.51099895000 ± 0.00000000015 MeV = 5.1100e-4 GeV
# Source:      PDG 2024 (Particle Data Group, Review of Particle Physics)
# PDG edition: 2024

# --- PREDICTED VALUE -----------------------------------------
# Value:       m_e = m_tau × (f_min/f_max)² = 510.9563 keV  (live, this run)
# Observed:    m_e = 510.998950 keV  (PDG 2024)
# Deviation:   -0.0426 keV, -0.0083% relative, ≈ -2.8e5 σ_PDG  (Clause 8 FAIL).
#
# HONEST RESIDUAL DECOMPOSITION (2026-05-18 lint — supersedes the prior
# "+0.12% inherited from m_τ / un-derived y_τ Feshbach analog" text,
# which was BOTH stale (live is -0.0083%, not +0.12%) AND wrong):
#
#   m_e residual  -0.0083%  =  m_τ absolute scale  -0.0013%
#                            +  Koide ratio        -0.0070%   (~84%, dominant)
#
# The dominant term is NOT inherited from m_τ (m_τ alone is only
# -0.0013%) and is NOT a Feshbach / dark-correction analog. It is the
# gap between the asserted rational δ = 2/9 = 0.2222222 and the physical
# Koide phase (measured 0.2222227 ± 9e-7, i.e. 0.53σ_δ from 2/9),
# lever-amplified ~120× by the f_min ≈ 0.04 near-cancellation. This is a
# wrong-term in an adopted phenomenological relation — not a precision
# floor, not a missing dark correction (master doc §5: the identical
# 1H+2F Family-D vertex cancels by construction in the ratio).

# --- DERIVED FORMULA -----------------------------------------
# m_e = m_τ × (f_min / f_max)²
#
# where f_j = 1 + ε · cos(2πj/k* + δ) are the Koide triplet factors on k*=3
# with ε = √2 (from Q_Koide = 2/3) and δ = 2/9 (asserted exact rational).
#
# f_min ≈ 0.04 is the smallest Koide factor (electron). The near-
# cancellation makes m_e a ~120× LEVER on δ: the asserted δ = 2/9 differs
# from the measured Koide phase by 0.53σ_δ, and that small gap is
# amplified into the dominant -0.0070% m_e residual. This is the
# structural reason the absolute m_e cannot be a parameter-free
# prediction while δ is an adopted (not substrate-derived) constant.
#
# Status of parts (honest, post-2026-05-18 lint):
#   - m_e/m_τ RATIO: exact-rational GIVEN the adopted Koide structure
#     (Q=2/3 ⟺ ε=√2; δ=2/9). The Koide RELATION itself is empirical
#     phenomenology, not substrate-derived (absolute route falsified /
#     C³_gen-blocked); δ=2/9 is a wrong-term carrying the residual.
#   - Absolute scale m_τ = v·y_τ: mathematically-complete-conditional —
#     v←N_hub←measured G_F is a circular calibration, NOT a prediction.
#   - Absolute m_e: NO clean grade. Clause 7 (uniqueness of the
#     multiplicative form) holds; Clause 8 FAILS vs σ_PDG. NOT
#     THEOREM-GRADE-NUMERICAL.

# --- INPUTS --------------------------------------------------
# symbol        | value        | status          | predictions/ file              | meaning
# --------------|--------------|-----------------|--------------------------------|--------
# m_tau         | 1.77684 GeV  | [G1-cond/circ]  | predictions/m_tau.py           | tau mass (v←N_hub←G_F)
# epsilon_Koide | √2           | [adopted-Koide] | predictions/epsilon_Koide.py   | Koide amplitude (Q=2/3)
# delta_Koide   | 2/9          | [adopted-Koide] | predictions/delta_Koide.py     | Koide phase (wrong-term, 0.53σ_δ)
# k_star        | 3            | [derived]       | predictions/k_star.py          | coordination

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from k_star import predict_k_star
from d_spatial import predict_d_spatial
from g_girth import predict_g_girth
from alpha_1 import predict_alpha_1
from alpha_1_full import predict_alpha_1_full, n_g_edge
from v_higgs import predict_v_higgs, delta as vh_delta, M_P, N_hub, alpha_1
from y_tau import predict_y_tau
from m_tau import predict_m_tau
from Q_Koide import chain_import_ramanujan_multiplicities
from epsilon_Koide import predict_epsilon_Koide
from delta_Koide import predict_delta_Koide
import functools

# --- chain imports ---
d = predict_d_spatial()
k = predict_k_star(d)
g = predict_g_girth(k, d)
alpha_1_bare_val = float(predict_alpha_1(k, g))
alpha_1_full_exact = predict_alpha_1_full(k, g, n_g_edge)
alpha_1_full = float(alpha_1_full_exact)
from V_count import V_count_pred as N_atoms_srs  # = 4, srs primitive cell |V| / K_4 quotient (predict_V_count)
y_tau = predict_y_tau(alpha_1_full, alpha_1_bare_val, k, n_H_legs=1, n_F_legs=2,
                       N_atoms=N_atoms_srs)
v_pred = predict_v_higgs(vh_delta, M_P, N_hub, alpha_1)
m_tau_pred = predict_m_tau(v_pred, y_tau)

mu_t, mu_o, mu_w = chain_import_ramanujan_multiplicities()
from p_toggle import predict_p_toggle
epsilon = predict_epsilon_Koide(k, mu_t, mu_o, mu_w, predict_p_toggle())
from Q_Koide import Q_Koide_pred as _Q_K   # = 2/3 (Born-rule sqrt-multiplicity)
delta_k = predict_delta_Koide(_Q_K)

# --- Koide f_j factors on k*=3 ---
k_int = int(round(k))
factors = [
    1.0 + epsilon * math.cos(2.0 * math.pi * j / k_int + delta_k)
    for j in range(k_int)
]
factors_sorted = sorted(factors)  # ascending: [min, mid, max]
f_min, f_mid, f_max = factors_sorted

# --- m_e from ratio ---
ratio_e_sq = (f_min / f_max) ** 2
m_e_pred = m_tau_pred * ratio_e_sq

# --- observed value ---
m_e_obs   = 0.00051099895   # GeV (PDG 2024)
m_e_sigma = 1.5e-13         # GeV (PDG 2024, extremely small)

dev_abs   = m_e_pred - m_e_obs
dev_rel   = dev_abs / m_e_obs

print("=" * 68)
print("  m_e  --  RATIO PREDICTION from m_tau via Koide f_j structure")
print("=" * 68)
print(f"  epsilon_Koide   = sqrt(2) = {epsilon:.10f}  [THEOREM]")
print(f"  delta_Koide     = 2/9    = {delta_k:.10f}  [THEOREM]")
print(f"  k*              = {k_int}")
print()
print(f"  f_j factors (ascending):")
for i, f in enumerate(factors_sorted):
    label = ['min (electron)', 'mid (muon)', 'max (tau)'][i]
    print(f"    f[{i}] = {f:.10f}  ({label})")
print()
print(f"  (f_min/f_max)^2 = {ratio_e_sq:.12f}")
print(f"  m_tau           = {m_tau_pred:.6f} GeV")
print(f"  m_e_pred        = m_tau × (f_min/f_max)² = {m_e_pred*1e6:.4f} keV")
print(f"  m_e_obs         = {m_e_obs*1e6:.6f} keV  (PDG 2024)")
print(f"  Deviation       = {dev_abs*1e6:+.4f} keV  ({dev_rel*100:+.4f}%)")
print()
print("  Grade chain (honest, 2026-05-18 lint):")
print("    Ratio (f_min/f_max)² = exact-rational GIVEN adopted Koide (ε,δ)")
print("    Koide relation       = empirical phenomenology (NOT substrate-")
print("                           derived; absolute route falsified/C³_gen)")
print("    m_tau abs scale      = math-complete-cond (v←N_hub←G_F circular)")
print(f"    Clause 8 vs σ_PDG    = FAIL ({dev_rel*100:+.4f}%, ≈ {dev_abs/m_e_sigma:+.1e} σ)")
print()
print("  Residual decomposition: -0.0083% = -0.0013% (m_τ scale)")
print("                                   + -0.0070% (Koide δ=2/9 wrong-term,")
print("                                     0.53σ_δ vs measured phase, ~120× lever)")
print("  Dark correction: NONE (master §5: identical 1H+2F Family-D")
print("  vertex cancels by construction in the ratio).")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_m_e(m_tau, epsilon_Koide, delta_Koide, k_star):
    """
    Compute the electron mass as a ratio prediction from the tau mass.

    Formula:
        m_e = m_τ × (f_min/f_max)²

    where f_j = 1 + ε · cos(2πj/k* + δ) are the Koide triplet factors.

    The ratio (f_min/f_max)² is theorem-grade under ε, δ, k* all being
    theorem-grade. Absolute scale inherits m_τ's G1 conditional status.

    Parameters
    ----------
    m_tau : float
        Tau lepton mass in GeV. From predictions/m_tau.py.
    epsilon_Koide : float
        Koide amplitude parameter. From predictions/epsilon_Koide.py (= √2).
    delta_Koide : float
        Koide phase parameter. From predictions/delta_Koide.py (= 2/9).
    k_star : int
        Coordination number of srs (= 3). From predictions/k_star.py.

    Returns
    -------
    float
        Predicted electron mass in GeV.
    """
    import math
    k_int = int(round(k_star))
    factors = [
        1.0 + epsilon_Koide * math.cos(2.0 * math.pi * j / k_int + delta_Koide)
        for j in range(k_int)
    ]
    f_sorted = sorted(factors)
    f_min_local = f_sorted[0]
    f_max_local = f_sorted[2]
    return m_tau * (f_min_local / f_max_local) ** 2


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_result = m_e_pred
    pure_result = predict_m_e(m_tau_pred, epsilon, delta_k, k)
    print()
    print(f"Implementation: {impl_result*1e6:.8f} keV")
    print(f"Pure function:  {pure_result*1e6:.8f} keV")
    assert abs(impl_result - pure_result) < 1e-15, \
        f"Mismatch: {impl_result} vs {pure_result}"
    print("OK: outputs agree.")
    print(f"    m_e = {pure_result*1e6:.4f} keV  "
          f"(obs: {m_e_obs*1e6:.6f} keV, {dev_rel*100:+.4f}%)")
    print("    Rigor: ratio exact-rational GIVEN adopted Koide (ε,δ);")
    print("           absolute math-complete-cond (G_F-circular); Clause 8")
    print("           FAILS vs σ_PDG. NOT theorem-grade-numerical.")
