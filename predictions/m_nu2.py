#!/usr/bin/env python3
"""
m_nu2 — second light neutrino mass (normal ordering)

NEW DERIVATION (supersedes 2026-05-04): m_ν₂ inherits from m_ν₃ via the
theorem-grade Ihara R = 228/7 splitting:

    m_ν₂ = m_ν₃ / √R

with m_ν₃ from `predictions/m_nu3.py` — **DOMINANT-CONDITIONAL** (re-graded
2026-05-18 chain audit; was overstated UNIQUE-THEOREM-GRADE-CONDITIONAL).
m_ν₃'s absolute scale rests on the ADOPTED y_ν = 1 (Dirac/top Yukawa —
the undischarged anchor that retracted m_top), G_F-circular N_hub,
engineered δ⁴ cancellation, and a §7.6-off-support normalization. m_ν₂
**inherits all of these** through m_ν₃.

The "ZERO adopted inputs" claim below is **FALSE and retracted**: the
chain adopts y_ν = 1 (undisclosed in the prior framing, exposed in
m_nu3_derivation.md Step 3). The numerical match (m_ν₂ +2.4% vs NuFIT
6.0; +1.91σ_PDG; Clause 8 FAIL vs σ_PDG) is conditional on y_ν = 1.
The ONLY anchor-free neutrino prediction is the splitting R = 228/7
(Ihara, theorem-grade); the absolute m_ν₂ is NOT parameter-free.

Old chain preserved at predictions/retracted/m_nu2_seesaw_PS{,_derivation}.{py,md}.

COMPANION DOCS:
- predictions/m_nu3.py — m_ν₃ derivation
- predictions/m_nu3_derivation.md — full structural chain
- predictions/R_nu_splitting.py — R = 228/7 (theorem-grade Ihara)
"""

# ============================================================
# PARAMETER: m_ν₂ (second light neutrino mass, normal ordering)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       m_ν₂ = √Δm²₂₁ = 8.654 ± 0.110 meV  (assuming m_ν₁ = 0)
# Source:      NuFIT 6.0 (September 2024), normal ordering;
#              Δm²₂₁ = (7.49 ± 0.19) × 10⁻⁵ eV².
# PDG edition: 2024

# --- PREDICTED VALUE -----------------------------------------
# Value:       m_ν₂ = m_ν₃ / √R
#                    = 50.57 / √(228/7) meV
#                   ≈ 8.86 meV  (with m_ν₃ from new global formula)
# Deviation:   +0.21 meV (+2.40%, +1.91 σ_PDG) — Clause 8 FAIL vs σ_PDG alone.
#
# NOTE (2026-05-15 dark-correction sweep): inherits m_ν₃'s Feshbach-baked-in
# treatment.  No separate multiplicative DC factor is applied here either:
# the spectral-gap mechanism in m_ν₃ already evaluates the Feshbach residue,
# and R = 228/7 is a pure Ihara cycle-counting splitting ratio (no DC needed).
# The +2.40% residual inherits the N_hub anchor sensitivity of m_ν₃; see
# m_nu3.py NOTE and master doc §3 (B) "Application clarification".

# --- DERIVED FORMULA -----------------------------------------
# m_ν₂ = m_ν₃ / √R
#      = [(k* × N_atoms) × M_Pl × N_hub^(-1/2)] / √R
#
# Logical chain:
#   Step 1: m_ν₃ = (k* × N_atoms) × M_Pl × N_hub^(-1/2)
#           [Type 4: predictions/m_nu3.py — UNIQUE-THEOREM-GRADE-CONDITIONAL]
#   Step 2: R = 228/7 from Ihara 5-step Chebyshev recurrence
#           [Type 4: predictions/R_nu_splitting.py — theorem-grade]
#   Step 3: m_ν₂² / m_ν₃² = 1/R (with m_ν₁ = 0; R = Δm²₃₁/Δm²₂₁)
#           [Type 2 algebra: definition of R]
#   Step 4: m_ν₂ = m_ν₃ / √R [Type 2 algebra]

# --- INPUTS --------------------------------------------------
# symbol     | value     | status     | predictions/ file       | meaning
# -----------|-----------|------------|-------------------------|----
# k_star     | 3         | [derived]  | predictions/k_star.py   | Hashimoto Perron
# N_atoms    | 4         | [derived]  | (structural, srs)        | atoms per primitive cell
# M_P        | M_Pl_natural.M_Pl_GeV | [derived] | predictions/M_Pl_natural.py | M_Pl/M_subst=8/√π theorem; GeV=single declared SI-anchor (CODE imports it line 100; NOT "[external]")
# N_hub      | ~8.4e60   | [derived]  | predictions/N_hub.py     | substrate site count
# R          | 228/7     | [derived]  | predictions/R_nu_splitting.py | Ihara splitting

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import math
import functools

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from k_star import predict_k_star
from d_spatial import predict_d_spatial
from g_girth import predict_g_girth
from alpha_1 import predict_alpha_1
from N_hub import predict_N_hub
from R_nu_splitting import predict_R_nu_splitting
from m_nu3 import predict_m_nu3
from p_toggle import predict_p_toggle
from V_count import predict_V_count

# Substrate primitives
d_val = predict_d_spatial()
k_val = predict_k_star(d_val)
g_val = predict_g_girth(k_val, d_val)
alpha_1 = predict_alpha_1(k_val, g_val)
from V_count import V_count_pred as N_atoms  # = 4, srs primitive cell |V| / K_4 quotient (predict_V_count)

# External anchors
from M_Pl_natural import M_Pl_GeV as M_P_GeV   # CODATA single-source — ANTHROPOCENTRIC SI TRANSLATION
G_F_obs = 1.1663787e-5                          # the measured Fermi constant — used to pin N_hub's adopted value; G_F itself is a PREDICTION (predictions/G_F.py)
from delta_Koide import delta_Koide_pred as delta  # = 2/9 (Q*(1-Q) at Q=2/3, predict_delta_Koide)
_p_val = predict_p_toggle()
_V_val = predict_V_count(k_val, d_val)
N_hub = predict_N_hub(G_F_obs, M_P_GeV, alpha_1, delta, k_val, _p_val, _V_val)
R_split = predict_R_nu_splitting(k_val, _p_val, _V_val)

# m_ν₃ from the new global formula
m_nu3_eV = predict_m_nu3(k_val, N_atoms, M_P_GeV, N_hub)

# m_ν₂ = m_ν₃ / √R
m_nu2_eV = m_nu3_eV / math.sqrt(R_split)

# Module-level observation exports for run_predictions.py introspection.
# NuFIT 6.0 normal ordering: Δm²₂₁ = (7.49 ± 0.19) × 10⁻⁵ eV².
m_nu2_pred = m_nu2_eV
m_nu2_obs = math.sqrt(7.49e-5)
m_nu2_sigma = 0.5 * 0.19e-5 / m_nu2_obs                       # PDG-only ~0.11 meV

print("m_ν₂ prediction — global spectral-gap chain via R = 228/7 splitting")
print(f"  Upstream:  k*={k_val}, N_atoms={N_atoms}, N_hub={N_hub:.4e}")
print(f"  m_ν₃     = {m_nu3_eV * 1e3:.4f} meV   [from predictions/m_nu3.py]")
print(f"  R        = 228/7 = {R_split:.6f}      [theorem-grade Ihara]")
print()
print(f"  m_ν₂ = m_ν₃ / √R = {m_nu2_eV * 1e3:.4f} meV")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_m_nu2(k_star, N_atoms, M_P_GeV, N_hub, R_split):
    """
    Predict m_ν₂ from m_ν₃ via Ihara R-splitting.

    Formula
    -------
        m_ν₂ = m_ν₃ / √R
        m_ν₃ = (k* × N_atoms) × M_Pl × N_hub^(-1/2)

    Parameters
    ----------
    k_star : int
        Substrate coordination number (= 3 for srs).
    N_atoms : int
        Atoms per primitive cell (= 4 for srs).
    M_P_GeV : float
        Planck mass in GeV.
    N_hub : float
        Substrate site count.
    R_split : float
        Ihara mass-squared splitting ratio (= 228/7 for srs).

    Returns
    -------
    float
        m_ν₂ in eV.
    """
    m_nu3 = predict_m_nu3(k_star, N_atoms, M_P_GeV, N_hub)
    return m_nu3 / math.sqrt(R_split)


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_result = m_nu2_eV
    pure_result = predict_m_nu2(k_val, N_atoms, M_P_GeV, N_hub, R_split)
    print()
    print("=" * 60)
    print("STATUS (parameter linter clauses):")
    print("  Clauses 1-5: chain theorem-grade")
    print("    Step 1 [m_ν₃]    = Type 4 (predictions/m_nu3.py)")
    print("    Step 2 [R=228/7] = Type 4 (predictions/R_nu_splitting.py)")
    print("    Steps 3-4         = Type 2 algebra")
    print("  Clause 8 (numerical match, σ_PDG only):")
    print("    σ_obs      = 1.27% (NuFIT 6.0)")
    print("    Deviation  = +2.40% (+1.91 σ_PDG) ⇒ Clause 8 FAIL.")
    print("=" * 60)

    # NuFIT 6.0
    dm2_21_obs = 7.49e-5
    dm2_21_sigma = 0.19e-5
    m_nu2_obs = math.sqrt(dm2_21_obs)
    m_nu2_sigma = 0.5 * dm2_21_sigma / m_nu2_obs

    dev_abs = pure_result - m_nu2_obs
    dev_rel = dev_abs / m_nu2_obs
    dev_sigma = dev_abs / m_nu2_sigma

    print()
    print(f"  Implementation:  {impl_result:.9f} eV")
    print(f"  Pure function:   {pure_result:.9f} eV")
    assert abs(impl_result - pure_result) < 1e-12, (
        f"Implementation vs pure function mismatch: "
        f"{impl_result} vs {pure_result}"
    )
    print(f"  OK: outputs agree.")
    print()
    print(f"    m_ν₂ predicted = {pure_result * 1e3:.4f} meV")
    print(f"    m_ν₂ NuFIT 6.0 = {m_nu2_obs * 1e3:.4f} ± {m_nu2_sigma * 1e3:.4f} meV")
    print(f"    Deviation       = {dev_abs * 1e3:+.4f} meV  ({dev_rel*100:+.3f}%, {dev_sigma:+.2f} σ_obs)")
    print()
    print(f"  Cross-check: m_ν₃²/m_ν₂² = {(m_nu3_eV/m_nu2_eV)**2:.6f}")
    print(f"               R = 228/7  = {R_split:.6f}")
    print(f"               (must match by construction; OK if equal)")
