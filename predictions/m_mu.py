#!/usr/bin/env python3
"""
Canonical prediction file for the muon mass m_mu (ratio prediction from m_tau).

Audit anchor: Row P11 of `docs/parameters/parameter_uniqueness_ledger.md`. UNIQUE for the
multiplicative structure; CONDITIONAL on Row P10 (v_Higgs) + Row P7 (y_τ).
m_μ is theorem-grade as a Koide-ratio prediction from m_τ via the f_j
structure (Q_Koide = 2/3, ε² = 2, δ = 2/9 algebraic identities). Inherits
P10's G1 conditional status.
"""

# ============================================================
# PARAMETER: m_mu (muon mass) -- ratio prediction from m_tau
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       m_mu = 105.6583755 ± 0.0000023 MeV = 0.10566 GeV
# Source:      PDG 2024 (Particle Data Group, Review of Particle Physics)
# PDG edition: 2024

# --- PREDICTED VALUE -----------------------------------------
# Value:       m_mu = m_tau × (f_mid/f_max)² ≈ 105.78 MeV (with v=246.22 GeV)
# Deviation:   +0.13 MeV absolute, +0.13% relative (systematic inherited from m_tau)
#
# Bridge convention (docs/framework/framework_scheme_convention.md §7): m_μ inherits
# its scheme treatment from m_τ via the Koide ratio structure (f_mid/f_max)².
# The Koide ratios themselves are exact algebraic predictions (Q=2/3, ε=√2,
# δ=2/9). The +0.13% residual tracks the same un-derived Feshbach analog on
# y_τ that drives m_τ's residual (Priority 4.4 step 2.2). When that analog
# is derived, m_μ's residual is expected to shrink commensurately.

# --- DERIVED FORMULA -----------------------------------------
# m_μ = m_τ × (f_mid / f_max)²
#
# where f_j = 1 + ε · cos(2πj/k* + δ), j = 0, 1, 2
#   ε = √2 (theorem-grade: predictions/epsilon_Koide.py)
#   δ = 2/9 (theorem-grade: predictions/delta_Koide.py)
#   k* = 3 (theorem-grade: predictions/k_star.py)
#
# f_j are the Koide triplet factors on k*=3:
#   j=0: f_max (tau)
#   j=1: f_mid (mu)
#   j=2: f_min (electron)
#
# Status of parts:
#   - m_μ/m_τ RATIO: theorem-grade (Koide f_j structure via ε, δ exact rationals).
#     ε² = 2 and δ = 2/9 are both STRICT-SOLID from Wigner D¹ on k*=3.
#     No free parameters enter the ratio.
#   - Absolute scale m_τ: STRICT-SOLID conditional on G1 (predictions/m_tau.py).
#   - Therefore absolute m_μ: THEOREM-GRADE-STRUCTURAL conditional, inheriting
#     m_τ's two conditionals — G1 (v←N_hub) AND the y_τ Family-D c_F Clause-6
#     channel argument (W1 2026-05-18; see predictions/m_tau.py). The Koide
#     RATIO stays theorem-grade (ε,δ exact); only the absolute scale inherits.
#
# The Koide identity Q = (m_e + m_μ + m_τ) / (√m_e + √m_μ + √m_τ)² = 2/3
# is satisfied BY CONSTRUCTION of the f_j parametrization. It is NOT an
# independent verification.

# --- INPUTS --------------------------------------------------
# symbol        | value       | status          | predictions/ file              | meaning
# --------------|-------------|-----------------|--------------------------------|--------
# m_tau         | 1.7791 GeV  | [STRICT-SOLID]  | predictions/m_tau.py           | tau mass (G1 conditional)
# epsilon_Koide | √2          | [THEOREM]       | predictions/epsilon_Koide.py   | Koide amplitude
# delta_Koide   | 2/9         | [THEOREM]       | predictions/delta_Koide.py     | Koide phase
# k_star        | 3           | [derived]       | predictions/k_star.py          | coordination

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

# --- m_mu from ratio ---
ratio_mu_sq = (f_mid / f_max) ** 2
m_mu_pred = m_tau_pred * ratio_mu_sq

# --- observed value ---
m_mu_obs   = 0.1056583755   # GeV (PDG 2024)
m_mu_sigma = 2.3e-9         # GeV (PDG 2024, very small)

dev_abs   = m_mu_pred - m_mu_obs
dev_rel   = dev_abs / m_mu_obs

print("=" * 68)
print("  m_mu  --  RATIO PREDICTION from m_tau via Koide f_j structure")
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
print(f"  (f_mid/f_max)^2 = {ratio_mu_sq:.10f}")
print(f"  m_tau           = {m_tau_pred:.6f} GeV")
print(f"  m_mu_pred       = m_tau × (f_mid/f_max)² = {m_mu_pred*1000:.4f} MeV")
print(f"  m_mu_obs        = {m_mu_obs*1000:.6f} MeV  (PDG 2024)")
print(f"  Deviation       = {dev_abs*1000:+.4f} MeV  ({dev_rel*100:+.4f}%)")
print()
print("  Grade chain:")
print("    Ratio (f_mid/f_max)² = THEOREM (ε, δ both STRICT-SOLID on k*=3)")
print("    m_tau abs scale      = STRICT-SOLID conditional on G1 (via v)")
print("    m_mu abs scale       = inherits m_tau (STRICT-SOLID cond G1)")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_m_mu(m_tau, epsilon_Koide, delta_Koide, k_star):
    """
    Compute the muon mass as a ratio prediction from the tau mass.

    Formula:
        m_μ = m_τ × (f_mid/f_max)²

    where f_j = 1 + ε · cos(2πj/k* + δ) are the Koide triplet factors.

    The ratio (f_mid/f_max)² is theorem-grade under ε, δ, k* all being
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
        Predicted muon mass in GeV.
    """
    import math
    k_int = int(round(k_star))
    factors = [
        1.0 + epsilon_Koide * math.cos(2.0 * math.pi * j / k_int + delta_Koide)
        for j in range(k_int)
    ]
    f_sorted = sorted(factors)
    f_mid_local = f_sorted[1]
    f_max_local = f_sorted[2]
    return m_tau * (f_mid_local / f_max_local) ** 2


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_result = m_mu_pred
    pure_result = predict_m_mu(m_tau_pred, epsilon, delta_k, k)
    print()
    print(f"Implementation: {impl_result*1000:.6f} MeV")
    print(f"Pure function:  {pure_result*1000:.6f} MeV")
    assert abs(impl_result - pure_result) < 1e-12, \
        f"Mismatch: {impl_result} vs {pure_result}"
    print("OK: outputs agree.")
    print(f"    m_mu  = {pure_result*1000:.4f} MeV  "
          f"(obs: {m_mu_obs*1000:.6f} MeV, {dev_rel*100:+.4f}%)")
    print("    Rigor: Ratio THEOREM (ε,δ exact); absolute THEOREM-GRADE-STRUCTURAL")
    print("           conditional — G1 AND y_τ Family-D c_F (W1 2026-05-18).")
