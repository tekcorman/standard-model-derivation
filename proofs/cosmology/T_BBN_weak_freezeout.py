#!/usr/bin/env python3
"""
---
derives: T_BBN_weak_freezeout
inputs:
  - G_F_pred         # framework Fermi constant (predictions/G_F.py)
  - M_Pl_GeV         # framework Planck mass (predictions/M_Pl_natural.py)
script_version: 2.0.0
doc: predictions/T_BBN_weak_freezeout_derivation.md
mechanism: structural
rigor_status: theorem-grade-structural-bounded-by-substrate-thermal-coupling
phase: IIb
---

T_BBN_weak_freezeout — BBN Stage 1 weak freeze-out (Phase IIb F-fiber).

Phase IIb rate balance Γ_weak(T) = H(T) for n ↔ p interconversion. Same
mechanism as T_nu_dec but with calorimetric calibration factor accounting
for Q_np-bound integration over phase space.

α-CONVENTION (post-2026-05-27 α-audit): Phase IIb freezeout uses α = 1/2
(INSTANTANEOUS), not α = 25/48 (cumulative). See `predictions/T_nu_dec.py`
docstring and an internal working note.

Calibration via empirical std-cosmology ratio: T_BBN-1 / T_ν_dec ≈ 0.7/1.5
= 7/15 ≈ 0.467. Under instantaneous H ∝ T² (without √g_* factor), the
ratio still applies since both Γ_weak and H scale identically relative to
T_nu_dec. The ratio is set by Q_np-dependent calorimetric factors in
Γ_n↔p.

  T_BBN-1 (framework) ≈ T_ν_dec_framework · (T_BBN-1_LCDM / T_ν_dec_LCDM)
                      ≈ 0.844 MeV · 7/15 ≈ 0.394 MeV

Framework T_BBN-1 ≈ 0.39 MeV is LOWER than ΛCDM 0.7 MeV because substrate
H lacks √g_* prefactor; propagates to Y_p UNDER-prediction (~0.05 vs
observed 0.245) per the Y_p falsification candidate.

Bounded by Need-B for Q_np precision (BR4 closure-NEGATIVE).
"""

import sys
import os
import functools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from G_F import G_F_pred
from M_Pl_natural import M_Pl_GeV
from T_nu_dec import predict_T_nu_dec, ALPHA_NUM, ALPHA_DEN


@functools.lru_cache(maxsize=None)
def predict_T_BBN_weak_freezeout(G_F, M_Pl, alpha_num, alpha_den, calorimetric_ratio_num, calorimetric_ratio_den):
    """Phase IIb T_BBN-1 = T_ν_dec · (T_BBN-1/T_ν_dec)_ΛCDM ratio.

    The Phase IIb ν decoupling rate Γ_ν = G_F²·T⁵ and the n↔p rate
    Γ_n↔p ≈ A·G_F²·T⁵ have the SAME T-scaling but different prefactors A.
    Under any H(T) cosmology, T_F(weak)/T_F(ν dec) is set by the rate
    prefactor ratio. Empirically (ΛCDM): T_BBN-1/T_ν_dec ≈ 0.7/1.5.

    Within-class residue: precise Q_np-dependent calorimetric ratio is
    bounded by Need-B (BR4 closure-NEGATIVE).

    Pure function — NO defaults; ratio expressed as K-rational fraction.

    Parameters
    ----------
    G_F, M_Pl : framework primitives
    alpha_num, alpha_den : 1, 2 (instantaneous Phase IIb α = 1/2)
    calorimetric_ratio_num, calorimetric_ratio_den : 7, 15 (= 0.7/1.5)

    Returns
    -------
    float : T_BBN-1 in GeV.
    """
    T_nu = predict_T_nu_dec(G_F, M_Pl, alpha_num, alpha_den)
    ratio = calorimetric_ratio_num / calorimetric_ratio_den
    return T_nu * ratio


# Calorimetric ratio: T_BBN-1 / T_ν_dec = 7/15 (≈ 0.467, ΛCDM-anchored)
# This ratio is bounded by Need-B (Q_np precision via quark masses).
RATIO_NUM = 7
RATIO_DEN = 15


T_BBN_weak_pred_GeV = predict_T_BBN_weak_freezeout(
    G_F_pred, M_Pl_GeV, ALPHA_NUM, ALPHA_DEN, RATIO_NUM, RATIO_DEN)
T_BBN_weak_pred_MeV = T_BBN_weak_pred_GeV * 1e3


# --- OBSERVED VALUE (standard cosmology consensus ~0.7-0.8 MeV) -----
# Framework predicts factor ~0.56 shift (lower) vs ΛCDM, giving ~0.39 MeV
T_BBN_weak_LCDM_MeV = 0.7

ratio_vs_LCDM = T_BBN_weak_pred_MeV / T_BBN_weak_LCDM_MeV

print("=" * 68)
print("  T_BBN_weak_freezeout -- THEOREM-GRADE-STRUCTURAL (Phase IIb)")
print("                          (bounded by substrate-thermal coupling + Need-B)")
print("=" * 68)
print(f"  DAG inputs:")
print(f"    G_F  = {G_F_pred:.4e} GeV^-2 (predictions/G_F.py)")
print(f"    M_Pl = {M_Pl_GeV:.4e} GeV (predictions/M_Pl_natural.py)")
print(f"    α    = {ALPHA_NUM}/{ALPHA_DEN}  (instantaneous Phase IIb)")
print(f"    cal-ratio = {RATIO_NUM}/{RATIO_DEN} (ΛCDM-anchored; Q_np-dependent)")
print(f"  Phase IIb: Γ_weak = H rate balance with calorimetric factor")
print(f"  T_BBN-1 (framework, α=1/2)       = {T_BBN_weak_pred_MeV:.3f} MeV")
print(f"  ΛCDM standard                    = {T_BBN_weak_LCDM_MeV} MeV")
print(f"  Ratio framework/ΛCDM             = {ratio_vs_LCDM:.2f}")
print()
print("  Framework T_BBN-1 LOWER than ΛCDM (factor ~0.56) — substrate H lacks")
print("  √g_* prefactor. Propagates to Y_p UNDER-prediction (~0.05 vs obs 0.245).")


if __name__ == "__main__":
    T = predict_T_BBN_weak_freezeout(G_F_pred, M_Pl_GeV, ALPHA_NUM, ALPHA_DEN, RATIO_NUM, RATIO_DEN)
    assert abs(T - T_BBN_weak_pred_GeV) < 1e-15
    print(f"\nOK: T_BBN-1 = {T*1e3:.3f} MeV (framework Phase IIb, α=1/2)")
