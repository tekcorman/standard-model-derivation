#!/usr/bin/env python3
"""
heavy_quark_anchor_dark — the ONE unified Feshbach channel-read dark factor (B′),
consumed by BOTH m_b and m_t (per the "linter done right": one dressing read, not
a per-file patch).

    m_q_phys = m_q_bare · (1 − α₁_bare / h_P**p)

with the Perron channel h_P = k* − 1 (REAL) and the L-power p:

    p = 2  if L = 0   (saturation ceiling: m_t, y_t = 1 — only |a₀|² dressable)
    p = 1  if L > 0   (propagating amplitude: m_b, L = g)

This is the Family-(B) Feshbach channel read Σ(h) = α₁/h (the leading first-girth-
return of the visible NB resolvent G_NB = (I − uB)⁻¹ at u = α₁) evaluated at the
PERRON channel, NOT at the shell (the shell read gives the ν parity-odd √5/4).

Authority: docs/theorems/theorem_substrate_feshbach_dark_corrections_master.md §(B′)
(added 2026-06-25, reconciling an internal working note).

GRADE: THEOREM-GRADE-STRUCTURAL-CONDITIONAL. The coefficient α₁/h_P**p ∈ ℚ ⊂ K is
forced GIVEN the channel (Perron) and the L-power; the channel/sign ASSIGNMENT is
the open conditional ("narrative-open" per the source doc §6 — NOT a §8-rule-1
two-independent-routes closure). NOT UNIQUE-THEOREM-GRADE. This is DISTINCT from
Family (D) (vertex α₁² dark, blocked for quarks via Need-D-3) and from the
koide_quark_ratio (P37) residual (a Cl(6) Fock-expansion artifact). It does not
overturn the 2026-05-15 "Family D non-applicability to quark sector" audit — that
audit is about the vertex dark and the ratio, not this absolute-anchor channel read.
"""

import sys
import os
import functools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from k_star import predict_k_star
from g_girth import predict_g_girth
from d_spatial import predict_d_spatial
from alpha_1 import predict_alpha_1

# --- IMPLEMENTATION ------------------------------------------
_d = predict_d_spatial()
_k = predict_k_star(_d)
_g = predict_g_girth(_k, _d)
_alpha_1 = predict_alpha_1(_k, _g)   # ((k*-1)/k*)^(g-2) = (2/3)^8
_h_P = _k - 1                         # Perron channel = k* − 1 = 2


# --- PURE FUNCTION -------------------------------------------
@functools.lru_cache(maxsize=None)
def predict_heavy_quark_anchor_dark(alpha_1_bare, h_P, power):
    """
    The (B′) Feshbach channel-read dark factor at the Perron channel.

    Returns the multiplicative factor applied to a heavy-quark absolute-mass
    anchor: m_q_phys = m_q_bare · predict_heavy_quark_anchor_dark(...).

    Parameters
    ----------
    alpha_1_bare : float   first-girth-return amplitude ((k*-1)/k*)^(g-2) = (2/3)^8.
    h_P : float            Perron channel eigenvalue k*-1 (= 2).
    power : int            L-power rank: 1 if L>0 (m_b), 2 if L=0 saturation (m_t).

    Returns
    -------
    float                  the dark factor (1 − α₁_bare / h_P**power).
    """
    return 1.0 - alpha_1_bare / (h_P ** power)


# --- VALIDATION ----------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("  heavy_quark_anchor_dark (B′) — Feshbach channel read at Perron channel")
    print("=" * 72)
    print(f"  α₁_bare = {_alpha_1:.6f}   h_P = k*-1 = {_h_P}")
    for p, who in [(1, "m_b  (L = g > 0 → power 1)"), (2, "m_t  (L = 0 saturation → power 2)")]:
        f = predict_heavy_quark_anchor_dark(_alpha_1, _h_P, p)
        print(f"  {who:32} factor = 1 − α₁/h_P**{p} = {f:.6f}  ({(f - 1.0) * 100:+.3f}%)")
    print("  GRADE: THEOREM-GRADE-STRUCTURAL-CONDITIONAL (master theorem §(B′))")
