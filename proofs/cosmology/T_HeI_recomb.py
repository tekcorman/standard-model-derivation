#!/usr/bin/env python3
"""
---
derives: T_HeI_recomb
inputs:
  - m_e_pred         # framework electron mass (for thermal mass scale)
  - eta_B_pred       # framework η_B
  - E_bind_HeI       # atomic-physics input (24.6 eV He first IP)
script_version: 1.0.0
doc: predictions/T_HeI_recomb_derivation.md
mechanism: structural
rigor_status: mathematically-complete-conditional
phase: III
---

T_HeI_recomb — Helium I recombination (He⁺ + e⁻ → He + γ).

Phase III F-fiber. Neutral He first ionization potential 24.6 eV is NOT
hydrogenic (2-electron QM); it's an ATOMIC-PHYSICS EXTERNAL INPUT applied
to framework's Phase III structural form.

Grade: **MATHEMATICALLY-COMPLETE-CONDITIONAL** on E_bind external input
(atomic-physics measurement / multi-electron QM result, NOT framework-derivable).

The Phase III structural form is theorem-grade; the SPECIFIC E_bind value
for neutral He is an external input (similar to G_F being downstream
calibration in v_higgs / N_hub).
"""

import sys
import os
import math
import functools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m_e import m_e_pred
from eta_B import eta_B_pred


@functools.lru_cache(maxsize=None)
def predict_T_HeI_recomb(E_bind_GeV, m_e_GeV, eta_B, n_iter):
    """Phase III F-fiber T_F = E_bind / N_thermal.

    Pure function — NO defaults.
    """
    ZETA3 = 1.2020569
    T = E_bind_GeV / 40.0
    for _ in range(n_iter):
        prefac = (m_e_GeV * T / (2 * math.pi)) ** 1.5
        n_gamma = (2 * ZETA3 / math.pi ** 2) * T ** 3
        N_thermal = math.log(prefac / (eta_B * n_gamma))
        T = E_bind_GeV / N_thermal
    return T


# --- EXTERNAL INPUT (atomic physics — He first IP) ---------
# Value: 24.587 eV (CODATA / NIST atomic-physics database)
# Status: external; not framework-derivable (multi-electron QM)
E_bind_HeI_eV = 24.587
E_bind_HeI_GeV = E_bind_HeI_eV * 1e-9


T_HeI_pred_GeV = predict_T_HeI_recomb(E_bind_HeI_GeV, m_e_pred, eta_B_pred, 10)
T_HeI_pred_eV  = T_HeI_pred_GeV * 1e9


# --- OBSERVED VALUE -----------------------------------------
T_HeI_obs_eV = 0.60

dev_pct = (T_HeI_pred_eV - T_HeI_obs_eV) / T_HeI_obs_eV * 100

print("=" * 68)
print("  T_HeI_recomb -- MATHEMATICALLY-COMPLETE-CONDITIONAL (Phase III)")
print("=" * 68)
print(f"  E_bind = 24.587 eV (atomic-physics external; multi-electron QM)")
print(f"  T_HeI (Phase III, framework m_e + η_B) = {T_HeI_pred_eV:.4f} eV")
print(f"  Observed                              = {T_HeI_obs_eV} eV")
print(f"  Match                                 = {dev_pct:+.2f}%")
print()
print("  Phase III structural form theorem-grade; E_bind atomic-physics external")


if __name__ == "__main__":
    print(f"\nOK: T_HeI = {T_HeI_pred_eV:.4f} eV (mathematically-complete-conditional)")
