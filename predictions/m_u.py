#!/usr/bin/env python3
"""
Canonical prediction file for m_u (up quark mass).

Status: THEOREM-GRADE-STRUCTURAL-CONDITIONAL via up-sector Koide cosine
ratio from m_t with ε²_up = 2 + 28·α₁_full and δ_up = 2/27 (W3).

Same machinery as predictions/m_c.py; m_u uses f_min (lightest factor).
"""

# ============================================================
# PARAMETER: m_u (up quark mass)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       m_u = 2.16 ± 0.49 MeV (MS-bar at 2 GeV)
# Source:      PDG 2024
# PDG edition: 2024

# --- PREDICTED VALUE -----------------------------------------
# Value:       m_u = m_t · (f_min / f_max)²
# Status:      THEOREM-GRADE-STRUCTURAL-CONDITIONAL (sensitive to ε,δ via
#              small f_min, expected ~5-10% scale; cascade not free-parameter)

# --- INPUTS --------------------------------------------------
# m_t, alpha_1_full, k_star, g_girth (all framework-internal)

import sys
import os
import functools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from k_star import predict_k_star
from g_girth import predict_g_girth
from d_spatial import predict_d_spatial
from alpha_1_full import alpha_1_full
from m_t import m_t_pred
from _koide_quark import koide_lighter_mass

_d = predict_d_spatial()
_k = predict_k_star(_d)
_g = predict_g_girth(_k, _d)
_alpha_1_full = float(alpha_1_full)

m_u_pred = koide_lighter_mass(m_t_pred, n=2, position='min',
                               alpha_1_full=_alpha_1_full,
                               k_star=_k, g_girth=_g)

m_u_obs = 2.16e-3       # GeV (MS-bar at 2 GeV)
m_u_sigma = 0.49e-3

dev_abs = m_u_pred - m_u_obs
dev_rel = dev_abs / m_u_obs
dev_sigma = dev_abs / m_u_sigma


@functools.lru_cache(maxsize=None)
def predict_m_u(m_t_GeV, alpha_1_full_val, k_star, g_girth):
    return koide_lighter_mass(m_t_GeV, n=2, position='min',
                               alpha_1_full=alpha_1_full_val,
                               k_star=k_star, g_girth=g_girth)


if __name__ == "__main__":
    print("=" * 68)
    print("  m_u  --  THEOREM-GRADE-STRUCTURAL-CONDITIONAL")
    print("=" * 68)
    print(f"  m_t (anchor) = {m_t_pred:.4f} GeV")
    print(f"  m_u          = {m_u_pred*1e3:.4f} MeV")
    print(f"  PDG 2024     = {m_u_obs*1e3} ± {m_u_sigma*1e3} MeV (MS-bar 2 GeV)")
    print(f"  Δ            = {dev_abs*1e3:+.4f} MeV  ({dev_rel*100:+.3f}%, {dev_sigma:+.2f}σ)")
    impl = m_u_pred
    pure = predict_m_u(m_t_pred, _alpha_1_full, _k, _g)
    assert abs(impl - pure) < 1e-12
    print(f"  Implementation = pure = {impl*1e3:.6f} MeV  ✓")
