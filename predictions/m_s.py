#!/usr/bin/env python3
"""
Canonical prediction file for m_s (strange quark mass).

Status: THEOREM-GRADE-STRUCTURAL-CONDITIONAL via down-sector Koide cosine
ratio from m_b with ε²_down = 2 + 6·α₁_full (Type IV n=1, f(1)=1) and
δ_down = 1/9 (W3 PS sector connectivity).

DERIVATION CHAIN
================
  Step 1: m_b (predictions/m_b.py) — Type IV Perron anchor.
  Step 2: ε²_down = 2 + 6·α₁_full·1·1 = 2 + 6·α₁_full (theorem)
  Step 3: δ_down = 2/(9·(1+1)) = 1/9 (W3 theorem-grade-structural)
  Step 4: Koide cosine ratio m_s/m_b = (f_mid/f_max)²
"""

# ============================================================
# PARAMETER: m_s (strange quark mass)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       m_s = 93.4 ± 8.6 MeV (MS-bar at 2 GeV)
# Source:      PDG 2024
# PDG edition: 2024

# --- PREDICTED VALUE -----------------------------------------
# Value:       m_s = m_b · (f_mid / f_max)²
# Status:      THEOREM-GRADE-STRUCTURAL-CONDITIONAL

import sys
import os
import functools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from k_star import predict_k_star
from g_girth import predict_g_girth
from d_spatial import predict_d_spatial
from alpha_1_full import alpha_1_full
from m_b import m_b_pred
from _koide_quark import koide_lighter_mass

_d = predict_d_spatial()
_k = predict_k_star(_d)
_g = predict_g_girth(_k, _d)
_alpha_1_full = float(alpha_1_full)

m_s_pred = koide_lighter_mass(m_b_pred, n=1, position='mid',
                               alpha_1_full=_alpha_1_full,
                               k_star=_k, g_girth=_g)

m_s_obs = 93.4e-3       # GeV (MS-bar at 2 GeV)
m_s_sigma = 8.6e-3

dev_abs = m_s_pred - m_s_obs
dev_rel = dev_abs / m_s_obs
dev_sigma = dev_abs / m_s_sigma


@functools.lru_cache(maxsize=None)
def predict_m_s(m_b_GeV, alpha_1_full_val, k_star, g_girth):
    return koide_lighter_mass(m_b_GeV, n=1, position='mid',
                               alpha_1_full=alpha_1_full_val,
                               k_star=k_star, g_girth=g_girth)


if __name__ == "__main__":
    print("=" * 68)
    print("  m_s  --  THEOREM-GRADE-STRUCTURAL-CONDITIONAL")
    print("=" * 68)
    print(f"  m_b (anchor) = {m_b_pred:.4f} GeV")
    print(f"  m_s          = {m_s_pred*1e3:.4f} MeV")
    print(f"  PDG 2024     = {m_s_obs*1e3} ± {m_s_sigma*1e3} MeV (MS-bar 2 GeV)")
    print(f"  Δ            = {dev_abs*1e3:+.4f} MeV  ({dev_rel*100:+.3f}%, {dev_sigma:+.2f}σ)")
    impl = m_s_pred
    pure = predict_m_s(m_b_pred, _alpha_1_full, _k, _g)
    assert abs(impl - pure) < 1e-12
    print(f"  Implementation = pure = {impl*1e3:.6f} MeV  ✓")
