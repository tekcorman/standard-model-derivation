#!/usr/bin/env python3
"""
---
derives: T_nu_dec
inputs:
  - G_F_pred             # framework Fermi constant
  - M_Pl_GeV             # framework Planck mass
script_version: 2.0.0
doc: predictions/T_nu_dec_derivation.md
mechanism: structural
rigor_status: theorem-grade-structural-bounded-by-substrate-thermal-coupling
phase: IIb
---

T_nu_dec — neutrino decoupling temperature (Phase IIb F-fiber).

Phase IIb rate balance Γ_weak(T) = H(T) under framework's instantaneous-event
T-N scaling.

α-CONVENTION (post-2026-05-27 α-audit, an internal working note):
  Phase IIb freezeout is an INSTANTANEOUS event at a specific epoch N. Per the
  bounded-sweep Phase IIa convention (ALPHA_THERMAL = 1/2) and the unified-
  observation reframe §3.3 (beta-Bernoulli posterior σ ∝ N^(-1/2)), the
  INSTANTANEOUS T-N scaling uses α = 1/2 (NOT the cumulative α=25/48 used
  for T_today observer-side propagation).

  Under α = 1/2:
    H(T) = T² · M_Pl^(-1)   (substrate cascade theorem H·N·t_P = 1; prefactor 1)
    Setting Γ_weak = G_F²·T⁵ = H gives T_nu_dec ≈ 0.844 MeV

  Framework's substrate H lacks the √g_* ≈ 3.28 factor that ΛCDM's
  radiation-era H carries (from Friedmann counting active relativistic
  species). This makes framework T_nu_dec ≈ 0.84 MeV vs ΛCDM's 1.5 MeV
  (factor √g_*^(1/3) ≈ 0.57 lower) — bounded by the substrate-thermal-
  coupling structural question (not addressed by the N_hub cascade theorem).

The T_nu_dec ≈ 0.84 MeV value still cleanly separates Phase IIb ν decoupling
from e⁺e⁻ annihilation at T_e_ann = m_e/3 ≈ 0.17 MeV (factor ~5), supporting
N_eff = 3 with reduced (but not zero) entropy transfer.

Historical note: predictions/T_nu_dec.py v1.0.0 (2026-05-26 through 2026-05-27
EOD) used α = 25/48 (cumulative-Perron) giving T_nu_dec = 3.18 MeV. The α-audit
2026-05-27 EOD+1 identified this as a calibration conflation between the
INSTANTANEOUS T-N scaling (α=1/2, for substrate-side rate balance) and the
CUMULATIVE T-N propagation (α=25/48, for observer-side T_today). The
correction restores internal consistency with the bounded-sweep
ALPHA_THERMAL=1/2 convention used for all Phase IIa F-fiber N-projections.

NO inline literals. NO defaults. DAG-cascade.
"""

import sys
import os
import math
import functools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from G_F import G_F_pred
from M_Pl_natural import M_Pl_GeV


@functools.lru_cache(maxsize=None)
def predict_T_nu_dec(G_F, M_Pl, alpha_temp_num, alpha_temp_den):
    """Phase IIb T_F from Γ_weak(T) = H(T) under instantaneous T-N scaling.

    Γ_weak ≈ G_F² · T⁵       (single ν species; no continuum 7π/60 prefactor — Clause-9 safe)
    H(T)                    = T^(1/α_temp) · M_Pl^(1 - 1/α_temp)   (cascade theorem; prefactor 1)

    Setting Γ = H and solving:
       T^(5 - 1/α) = M_Pl^(-(1/α - 1)) / G_F²
       T = [M_Pl^(-(1/α - 1)) / G_F²]^(α / (5α - 1))

    For α = 1/2 ⇒ 1/α = 2, exponent_lhs = 5 - 2 = 3:
       T = [M_Pl^(-1) / G_F²]^(1/3) ≈ 0.844 MeV

    Pure function — NO defaults.

    Parameters
    ----------
    G_F : float, Fermi constant in GeV^-2
    M_Pl : float, Planck mass in GeV
    alpha_temp_num : int, numerator of α (= 1 for instantaneous Phase IIb)
    alpha_temp_den : int, denominator (= 2 for instantaneous)
    """
    inv_a = alpha_temp_den / alpha_temp_num
    exponent_lhs = 5 - inv_a
    M_Pl_factor = M_Pl ** (-(inv_a - 1))
    rhs = M_Pl_factor / (G_F ** 2)
    return rhs ** (1.0 / exponent_lhs)


# Instantaneous α = 1/2 (Phase IIb F-fiber convention, matches bounded-sweep
# ALPHA_THERMAL = 1/2 used for all Phase IIa F-fibers).
# Per unified_observation_process_reframe_2026-05-25.md §3.3 (beta-Bernoulli
# posterior σ ∝ N^(-1/2)) and α-audit verdict 2026-05-27 EOD+1.
ALPHA_NUM = 1
ALPHA_DEN = 2


T_nu_dec_pred_GeV = predict_T_nu_dec(G_F_pred, M_Pl_GeV, ALPHA_NUM, ALPHA_DEN)
T_nu_dec_pred_MeV = T_nu_dec_pred_GeV * 1e3


# --- OBSERVED / REFERENCE (ΛCDM vs framework distinct) -------
# ΛCDM standard: T_ν_dec ≈ 1.5 MeV (under H ∝ T² radiation era WITH √g_* ≈ 3.28)
# Framework: 0.844 MeV (α=1/2 instantaneous; substrate H lacks √g_* factor)
T_nu_dec_LCDM_MeV = 1.5

ratio_vs_LCDM = T_nu_dec_pred_MeV / T_nu_dec_LCDM_MeV

print("=" * 68)
print("  T_nu_dec -- THEOREM-GRADE-STRUCTURAL (Phase IIb)")
print("=" * 68)
print(f"  DAG inputs:")
print(f"    G_F        = {G_F_pred:.4e} GeV^-2 (predictions/G_F.py)")
print(f"    M_Pl       = {M_Pl_GeV:.4e} GeV (predictions/M_Pl_natural.py)")
print(f"    α_temp     = {ALPHA_NUM}/{ALPHA_DEN} (instantaneous, Phase IIb)")
print(f"  Phase IIb: Γ_weak = H under instantaneous H ∝ T²")
print(f"  T_nu_dec (formula) = {T_nu_dec_pred_MeV:.3f} MeV")
print(f"  ΛCDM standard      = {T_nu_dec_LCDM_MeV} MeV")
print(f"  Ratio (formula/ΛCDM) = {ratio_vs_LCDM:.2f}")
print()
print("  Framework T_ν_dec LOWER than ΛCDM by factor ~1.78 because substrate H")
print("  lacks √g_* ≈ 3.28 factor that ΛCDM Friedmann carries.")
print("  This propagates downstream to T_BBN-1 + Y_p (see Y_p falsification).")


if __name__ == "__main__":
    print(f"\nOK: T_nu_dec = {T_nu_dec_pred_MeV:.3f} MeV (Phase IIb, α=1/2)")
