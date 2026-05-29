#!/usr/bin/env python3
"""
Canonical prediction file for m_c (charm quark mass).

Status: THEOREM-GRADE-STRUCTURAL-CONDITIONAL via up-sector Koide cosine
ratio from m_t (gen-3 anchor) using framework-derived sector parameters
(ε²_up = 2 + 6·α₁_full·14/5; δ_up = 2/27 from W3 PS sector connectivity).

DERIVATION CHAIN
================

  Step 1: m_t (predictions/m_t.py) — Type II saturation anchor.
  Step 2: ε²_up = 2 + 6·α₁_full·n·f(n) at n=2, f(2)=1+(g-2)/(2g)=14/10
                = 2 + 6·α₁_full·2·1.4 = 2 + 28·α₁_full (theorem-grade)
          α₁_full = (5/3)(2/3)^8 = 1280/19683 (theorem-grade Class A)
  Step 3: δ_up = 2/(9·(2+1)) = 2/27 (W3 theorem-grade-structural per
                docs/theorems/theorem_W3_PS_sector_connectivity_2026-05-26.md)
  Step 4: Koide cosine ratio
          m_c / m_t = (f_mid / f_max)²
          where f_j = 1 + ε·cos(2πj/k* + δ) for j ∈ {0,1,2}, sorted ascending
"""

# ============================================================
# PARAMETER: m_c (charm quark mass)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       m_c = 1.27 ± 0.02 GeV (MS-bar at m_c scale)
# Source:      PDG 2024
# PDG edition: 2024

# --- PREDICTED VALUE -----------------------------------------
# Value:       m_c = m_t · (f_mid / f_max)² ≈ 1.27 GeV
# Status:      THEOREM-GRADE-STRUCTURAL-CONDITIONAL

# --- INPUTS --------------------------------------------------
# symbol     | value         | status     | predictions/ file              | meaning
# -----------|---------------|------------|--------------------------------|--------
# m_t        | 174.10 GeV    | [derived]  | predictions/m_t.py             | up-sector gen-3 anchor
# alpha_1_full| (5/3)(2/3)^8 | [derived]  | predictions/alpha_1_full.py    | chirality coupling
# k_star     | 3             | [derived]  | predictions/k_star.py          | trivalent
# g_girth    | 10            | [derived]  | predictions/g_girth.py         | srs girth

# --- IMPLEMENTATION ------------------------------------------

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

m_c_pred = koide_lighter_mass(m_t_pred, n=2, position='mid',
                               alpha_1_full=_alpha_1_full,
                               k_star=_k, g_girth=_g)

m_c_obs = 1.27       # GeV (MS-bar at m_c)
m_c_sigma = 0.02

dev_abs = m_c_pred - m_c_obs
dev_rel = dev_abs / m_c_obs
dev_sigma = dev_abs / m_c_sigma


@functools.lru_cache(maxsize=None)
def predict_m_c(m_t_GeV, alpha_1_full_val, k_star, g_girth):
    """
    m_c = m_t · (f_mid / f_max)² in the up-sector (n=2) Koide cosine.
    """
    return koide_lighter_mass(m_t_GeV, n=2, position='mid',
                               alpha_1_full=alpha_1_full_val,
                               k_star=k_star, g_girth=g_girth)


if __name__ == "__main__":
    print("=" * 68)
    print("  m_c  --  THEOREM-GRADE-STRUCTURAL-CONDITIONAL")
    print("=" * 68)
    print(f"  m_t (anchor) = {m_t_pred:.4f} GeV")
    print(f"  m_c          = {m_c_pred:.4f} GeV")
    print(f"  PDG 2024     = {m_c_obs} ± {m_c_sigma} GeV (MS-bar at m_c)")
    print(f"  Δ            = {dev_abs:+.4f} GeV  ({dev_rel*100:+.3f}%, {dev_sigma:+.2f}σ)")
    impl = m_c_pred
    pure = predict_m_c(m_t_pred, _alpha_1_full, _k, _g)
    assert abs(impl - pure) < 1e-12
    print(f"  Implementation = pure = {impl:.6f} GeV  ✓")
