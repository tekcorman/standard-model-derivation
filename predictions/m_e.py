#!/usr/bin/env python3
"""
Canonical prediction file for the electron mass m_e (ratio prediction from m_tau).

Audit anchor: Row P11 of `docs/parameters/parameter_uniqueness_ledger.md`.
m_e = m_τ × (f_min/f_max)² is exact-rational given the Koide structure
(Q_Koide = 2/3 ⟺ ε = √2) and the phase δ. BOTH are DERIVED:
  • Q=2/3, ε²=2 — forced by the (4,2,2) Ramanujan/Spin^c Born weights.
  • the phase δ — DERIVED as the forced directed phase of the chiral ∂_N run.
    The STATIC shell is DEGENERATE (δ=0; every phase forced by the adjacency
    spectrum, no free U(1)) — so δ is correctly NOT in the static spectral
    channels; running forward along the C₃ screw FORCES the split
    0 : ±2π√(3/7) (rate 8:6:6; band-edge −½+i√7/2 ∈ ℚ(√−7)). Reproduced
    independently by the walker-length δ(L) and the Wigner-HM cosβ map →
    the sector rationals {2/9, 1/9, 2/27}. δ is NOT a free parameter, NOT an
    adoption, NOT a fit (an internal working note;
    RESTART §row-8, the 2026-06-21 win). This file hardcodes δ=2/9 (the
    run-phase value at the lepton slice) for the read; the derivation is the run.
  (The old "spectral channels span only ~4× / hierarchy un-derived / C³_gen
   unsolved / Koide ADOPTED" note was the pre-2026-06-21 static-route framing —
   superseded: the split is the RUN, not the static channels.)

Dark correction: the species-common Family-D factor −(5/6)α₁² is
generation-independent (all three charged-lepton vertices are identical
1H+2F) and divides out EXACTLY in any mass ratio — so THAT dark term is
not the residual. But its generation-RESOLVED sibling does NOT cancel:
the −70 ppm ratio residual is the OPEN O(α₁³) per-rep dark Dyson diagram
(the Family-D 16-cycle bubble × first-girth-return = q²⁴, allocated per
C₃-isotype). It IS a dark object — UN-derived, conjecture-grade (the MDL
water-filling ceiling, same as Q=2/3 / c_F); all 11 operator-routes
explored and ruled out (transport/band/curvature/resolvent/cover/
enantiomer/scale/cascade/degenerate-PT/Berry-holonomy/continuum-D₄ —
internal working notes,
docs/incomplete_equations_todo.md §1). An OPEN miss — NOT a floor, NOT
"fine"; the spectral/operator route to it is exhausted, the miss stands.

Clause 8 (σ_PDG only): FAIL — see PREDICTED VALUE for the honest
decomposition. Grade: ratio exact-rational given the Koide structure
(leading content forced by the C₃-screw run; subleading −70 ppm OPEN);
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
# -0.0013%). In the LIVE single-δ read it APPEARS as the gap between the
# leading Koide phase δ = 2/9 = 0.2222222 and the effective phase the data
# prefers (0.2222227 ± 9e-7, 0.53σ_δ), lever-amplified ~120× by the
# f_min ≈ 0.04 near-cancellation. But δ = 2/9 is NOT a wrong term — it is
# the FORCED leading phase (φ·s = (2π/√7)·s = 2/9 falls out of the C₃-screw
# run B(s·AXIS); an internal working note). The 0.53σ "phase gap" is the
# subleading O(α₁³) per-rep dark correction RE-EXPRESSED as an effective
# single-δ shift: the leading read freezes the Γ moduli {2,√2,√2} + the
# forced δ and drops the per-rep α₁³ Dyson diagram, which IS the −70 ppm.
# This is the species-common Family-D's generation-RESOLVED sibling (does
# NOT cancel in the ratio) — an OPEN dark miss (conjecture-grade MDL), not
# a precision floor and not a wrong δ.

# --- DERIVED FORMULA -----------------------------------------
# m_e = m_τ × (f_min / f_max)²
#
# where f_j = 1 + ε · cos(2πj/k* + δ) are the Koide triplet factors on k*=3
# with ε = √2 (from Q_Koide = 2/3) and δ = 2/9 (the DERIVED ∂_N-run phase).
#
# f_min ≈ 0.04 is the smallest Koide factor (electron). The near-
# cancellation makes m_e a ~120× LEVER on δ: the derived δ = 2/9 (the ∂_N-run
# directed phase at the lepton slice) differs from the effective phase the data
# prefers by 0.53σ_δ, and that small gap is amplified into the dominant -0.0070%
# m_e residual — which IS the OPEN subleading O(α₁³) per-rep dark correction
# (below), a separate object from δ. δ itself is derived (the run), not adopted.
#
# Status of parts (honest):
#   - m_e/m_τ RATIO: exact-rational given the Koide structure — ALL DERIVED:
#     Q=2/3 ⟺ ε²=2 (the (4,2,2) Born weights); δ = the forced directed phase of
#     the ∂_N run (=2/9 at the lepton slice — the static shell gives δ=0, the
#     chiral run forces the split; derive_generation_spectrum / build_dN /
#     generation_phase_delta_irreducible). The file HARDCODES δ=2/9 (the run-phase
#     value) for the read; the derivation is the run, not an adoption. The residual
#     is the OPEN subleading per-rep O(α₁³) dark correction the leading read drops
#     (un-derived, conjecture-grade MDL) — a separate object.
#   - Absolute scale m_τ = v·y_τ: mathematically-complete-conditional —
#     v←N_hub←measured G_F is a circular calibration, NOT a prediction.
#   - Absolute m_e: NO clean grade. Clause 7 (uniqueness of the
#     multiplicative form) holds; Clause 8 FAILS vs σ_PDG. NOT
#     THEOREM-GRADE-NUMERICAL.

# --- INPUTS --------------------------------------------------
# symbol        | value        | status          | predictions/ file              | meaning
# --------------|--------------|-----------------|--------------------------------|--------
# m_tau         | 1.77684 GeV  | [G1-cond/circ]  | predictions/m_tau.py           | tau mass (v←N_hub←G_F)
# epsilon_Koide | √2           | [derived: Q=2/3]  | predictions/epsilon_Koide.py   | Koide amplitude (ε²=2 from (4,2,2))
# delta_Koide   | 2/9          | [derived: ∂_N run]| predictions/delta_Koide.py     | Koide phase = forced ∂_N-run directed phase (hardcoded 2/9 for the read; subleading −70ppm OPEN)
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
print(f"  delta_Koide     = 2/9    = {delta_k:.10f}  [DERIVED: ∂_N-run directed phase; subleading −70ppm OPEN]")
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
print("    Ratio (f_min/f_max)² = exact-rational GIVEN derived Koide (ε²=2; δ=∂_N-run phase)")
print("    Koide relation       = DERIVED (Q=2/3 Born weights; δ=∂_N-run phase);")
print("                           substrate-ABSOLUTE route falsified, but the Koide")
print("                           RATIO route IS the derivation")
print("    m_tau abs scale      = math-complete-cond (v←N_hub←G_F circular)")
print(f"    Clause 8 vs σ_PDG    = FAIL ({dev_rel*100:+.4f}%, ≈ {dev_abs/m_e_sigma:+.1e} σ)")
print()
print("  Residual decomposition: -0.0083% = -0.0013% (m_τ scale)")
print("                                   + -0.0070% (the OPEN subleading O(α₁³) per-rep")
print("                                     dark item, 0.53σ_δ vs the derived δ, ~120× lever)")
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
    print("    Rigor: ratio exact-rational GIVEN derived Koide (ε²=2; δ=∂_N-run phase);")
    print("           absolute math-complete-cond (G_F-circular); Clause 8")
    print("           FAILS vs σ_PDG. NOT theorem-grade-numerical.")
