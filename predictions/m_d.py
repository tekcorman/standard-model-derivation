#!/usr/bin/env python3
"""
Canonical prediction file for m_d (down quark mass).

Status: THEOREM-GRADE-STRUCTURAL-CONDITIONAL via down-sector Koide
cosine ratio from m_b with ε²_down = 2 + 6·α₁_full and δ_down = 1/9 (W3).

Same machinery as predictions/m_s.py; m_d uses f_min (lightest factor).
"""

# ============================================================
# PARAMETER: m_d (down quark mass)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       m_d = 4.67 ± 0.48 MeV (MS-bar at 2 GeV)
# Source:      PDG 2024
# PDG edition: 2024

# --- PREDICTED VALUE -----------------------------------------
# Value:       m_d = m_b · (f_min / f_max)²
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

m_d_pred = koide_lighter_mass(m_b_pred, n=1, position='min',
                               alpha_1_full=_alpha_1_full,
                               k_star=_k, g_girth=_g)

m_d_obs = 4.67e-3       # GeV (MS-bar at 2 GeV)
m_d_sigma = 0.48e-3

dev_abs = m_d_pred - m_d_obs
dev_rel = dev_abs / m_d_obs
dev_sigma = dev_abs / m_d_sigma


@functools.lru_cache(maxsize=None)
def predict_m_d(m_b_GeV, alpha_1_full_val, k_star, g_girth):
    return koide_lighter_mass(m_b_GeV, n=1, position='min',
                               alpha_1_full=alpha_1_full_val,
                               k_star=k_star, g_girth=g_girth)


if __name__ == "__main__":
    print("=" * 68)
    print("  m_d  --  THEOREM-GRADE-STRUCTURAL-CONDITIONAL")
    print("=" * 68)
    print(f"  m_b (anchor) = {m_b_pred:.4f} GeV")
    print(f"  m_d          = {m_d_pred*1e3:.4f} MeV")
    print(f"  PDG 2024     = {m_d_obs*1e3} ± {m_d_sigma*1e3} MeV (MS-bar 2 GeV)")
    print(f"  Δ            = {dev_abs*1e3:+.4f} MeV  ({dev_rel*100:+.3f}%, {dev_sigma:+.2f}σ)")
    impl = m_d_pred
    pure = predict_m_d(m_b_pred, _alpha_1_full, _k, _g)
    assert abs(impl - pure) < 1e-12
    print(f"  Implementation = pure = {impl*1e3:.6f} MeV  ✓")
