#!/usr/bin/env python3
"""
---
derives: T_BBN_D_bottleneck
inputs:
  - eta_B_pred       # framework η_B
  - B_D              # external nuclear physics input (D binding 2.22 MeV)
  - m_nucleon        # external nuclear physics input (avg nucleon mass)
script_version: 1.0.0
doc: predictions/T_BBN_D_bottleneck_derivation.md
mechanism: structural
rigor_status: mathematically-complete-conditional
phase: III
---

T_BBN_D_bottleneck — BBN Stage 2 deuterium formation freezeout (Phase III).

Phase III F-fiber with E_bind = B_D (deuterium binding) and m_thermal = m_nucleon.
Both are NUCLEAR PHYSICS EXTERNAL INPUTS (not framework-derived).

Grade: **MATHEMATICALLY-COMPLETE-CONDITIONAL** on (B_D, m_nucleon) external
nuclear physics. The Phase III structural form is theorem-grade.

Distinct from Phase IIb Stage 1 weak freeze-out at 0.7 MeV
(`predictions/T_BBN_weak_freezeout.py`).
"""

import sys
import os
import math
import functools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eta_B import eta_B_pred


@functools.lru_cache(maxsize=None)
def predict_T_BBN_D_bottleneck(E_bind_GeV, m_thermal_GeV, eta_B, n_iter):
    """Phase III F-fiber T_F = E_bind / N_thermal with nucleon thermal mass.

    Pure function — NO defaults.
    """
    ZETA3 = 1.2020569
    T = E_bind_GeV / 30.0
    for _ in range(n_iter):
        prefac = (m_thermal_GeV * T / (2 * math.pi)) ** 1.5
        n_gamma = (2 * ZETA3 / math.pi ** 2) * T ** 3
        N_thermal = math.log(prefac / (eta_B * n_gamma))
        T = E_bind_GeV / N_thermal
    return T


# --- EXTERNAL INPUTS (nuclear physics) ----------------------
# B_D = 2.22452 MeV (D binding; nuclear-physics measurement)
# m_nucleon ≈ (m_p + m_n)/2 ≈ 0.93892 GeV (PDG)
# Both NOT framework-derived; external inputs.
B_D_MeV = 2.22452
B_D_GeV = B_D_MeV * 1e-3
m_nucleon_GeV = 0.93892   # external nuclear physics


T_BBN_D_pred_GeV = predict_T_BBN_D_bottleneck(B_D_GeV, m_nucleon_GeV, eta_B_pred, 10)
T_BBN_D_pred_MeV = T_BBN_D_pred_GeV * 1e3


# --- OBSERVED VALUE ----------------------------------------
T_BBN_D_obs_MeV = 0.07

dev_pct = (T_BBN_D_pred_MeV - T_BBN_D_obs_MeV) / T_BBN_D_obs_MeV * 100

print("=" * 68)
print("  T_BBN_D_bottleneck -- MATHEMATICALLY-COMPLETE-CONDITIONAL (Phase III)")
print("=" * 68)
print(f"  B_D = 2.22452 MeV (nuclear external)")
print(f"  m_nucleon = 0.93892 GeV (nuclear external)")
print(f"  T_BBN_D (Phase III, framework η_B)  = {T_BBN_D_pred_MeV:.4f} MeV")
print(f"  Observed                            = {T_BBN_D_obs_MeV} MeV")
print(f"  Match                               = {dev_pct:+.2f}%")
print()
print("  Phase III structural form theorem-grade; B_D + m_nucleon nuclear external")
print("  Phase IIb Stage 1 (weak freeze-out) at 0.7 MeV is SEPARATE")


if __name__ == "__main__":
    print(f"\nOK: T_BBN_D = {T_BBN_D_pred_MeV:.4f} MeV")
