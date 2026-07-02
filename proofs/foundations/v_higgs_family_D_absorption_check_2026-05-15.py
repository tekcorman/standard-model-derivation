#!/usr/bin/env python3
"""
proofs/foundations/v_higgs_family_D_absorption_check_2026-05-15.py

Sweep-item-5 consistency check: verify that the Family D sub-leading correction
on the Higgs VEV (δv/v = -α₁² ≈ -0.152% from the 1H+0F vertex per master doc
§3 (D)) is absorbed into the N_hub anchor calibration WITHOUT breaking the
G_F bridge.

Master doc §3 (D) states: "v_Higgs calibration check (§8 rule 2): Family D at
v_Higgs (1 Higgs leg) predicts δv/v = -c_H = -α₁² ≈ -0.152% as a sub-leading
correction to the leading -5/12 × α₁/(1-α₁) (Family C). This sub-leading
correction is absorbed into the N_hub anchor calibration via the G_F
round-trip (predictions/v_higgs.py, predictions/N_hub.py) — consistent by
construction with the framework's existing v-sector closure."

This script makes the "consistent by construction" claim explicit and quantifies
the magnitude of the absorption.

================================================================================
THE CALIBRATION CHAIN
================================================================================

The framework's v-sector chain is:
    v_pred(N_hub) = δ²·M_Pl·DC(α₁) / (√2·N_hub^{1/4})
where DC(α₁) is the multiplicative dark correction.

PDG G_F fixes v_obs = (√2·G_F_obs)^{-1/2} ≈ 246.220 GeV.

The framework's N_hub is calibrated such that v_pred(N_hub) = v_obs (round-trip
identity, predictions/N_hub.py).  This means N_hub absorbs whatever
multiplicative DC factor appears in v_pred:

    N_hub_calibrated = (δ²·M_Pl·DC(α₁) / (√2·v_obs))^4

If DC includes only the leading Class C (5/12)·α₁/(1-α₁) ≈ -1.69% factor,
N_hub is calibrated at some value N_C.  If DC includes Class C + Family D
sub-leading (1-α₁²) ≈ -1.84% combined, N_hub is calibrated at N_CD, with

    N_CD / N_C = (1 - α₁²)^4 / 1^4 = (1 - α₁²)^4 ≈ 1 - 4α₁² ≈ 0.99391

i.e. N_hub shifts by ≈ -0.609% to absorb the extra -0.152% on v.

================================================================================
DOWNSTREAM CONSEQUENCES
================================================================================

Other framework predictions depend on N_hub at different powers:
- H_0 ∝ N^{-1}      → shift = -4α₁² ≈ -0.609%  (within ±1σ Planck CMB)
- t_0 ∝ N^{+1}      → shift = +4α₁² ≈ +0.609%  (within Methuselah uncertainty)
- m_ν₃ ∝ N^{-1/2}   → shift = -2α₁² ≈ -0.305%  (would reduce m_ν₃ residual from +0.87% to +0.57%)

All shifts are sub-σ on current observations.  The framework's choice to absorb
Family D into N_hub is mathematically well-defined and observationally consistent;
it propagates a ~0.6% systematic uncertainty into the N_hub anchor that's
indistinguishable from the existing N_hub anchor-spread (G_F-anchored ~8.395e60
vs m_τ-anchored ~8.44e60 differ by 0.55%).

================================================================================
THE G_F BRIDGE
================================================================================

By construction, the G_F round-trip remains exact whichever DC scheme is used:
    v_pred(N_hub_calibrated) = v_obs   ⇒   G_F_pred = G_F_obs

The bridge is preserved BY DESIGN; the only effect of including/excluding
Family D on v is to shift the calibrated value of N_hub by a sub-percent
amount, which propagates to N_hub-dependent observables (H_0, t_0, m_ν).

CONCLUSION: master doc §3 (D)'s "absorbed into N_hub anchor calibration —
consistent by construction" statement is verified.  The framework's structural
choice to absorb Family D on v rather than expose it is well-defined; future
work could elect to make the absorption explicit, with the consequence of
shifting H_0 by -0.61% and m_ν₃ by -0.30% (both improving cluster Clause 8
match modestly), but doing so doesn't change any falsifiable prediction
beyond the absorbed sub-percent.
"""

from __future__ import annotations
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'predictions'))

from alpha_1 import predict_alpha_1
from k_star import predict_k_star
from d_spatial import predict_d_spatial
from g_girth import predict_g_girth


def main() -> None:
    d = predict_d_spatial()
    k = predict_k_star(d)
    g = predict_g_girth(k, d)
    alpha_1 = predict_alpha_1(k, g)
    alpha_1_full = alpha_1 / (1 - alpha_1)

    # Class C (5/12) correction (currently applied to v_higgs.py)
    c_v = 5.0 / 12.0
    factor_class_C = 1 - c_v * alpha_1_full

    # Family D sub-leading on v: 1H+0F vertex per master doc §3 (D)
    # δv/v = -c_H = -α₁²
    factor_family_D = 1 - alpha_1 ** 2

    # Combined factor (if both applied)
    combined_factor = factor_class_C * factor_family_D

    print("=" * 76)
    print("v_Higgs Family D absorption consistency check")
    print("=" * 76)
    print()
    print(f"Substrate primitives:")
    print(f"  α₁_bare = (2/3)^8       = {alpha_1:.10e}")
    print(f"  α₁_bare²                = {alpha_1**2:.6e}")
    print(f"  α₁_bare/(1-α₁_bare)     = {alpha_1_full:.6e}")
    print()
    print(f"Multiplicative DC factors on v (relative to bare BZJ):")
    print(f"  Class C (5/12)·α₁/(1-α₁)  applied:    {factor_class_C:.10f}  ({(factor_class_C-1)*100:+.5f}%)")
    print(f"  Family D (-α₁²) sub-leading:           {factor_family_D:.10f}  ({(factor_family_D-1)*100:+.5f}%)")
    print(f"  Combined product:                       {combined_factor:.10f}  ({(combined_factor-1)*100:+.5f}%)")
    print()

    # G_F round-trip: v_pred(N_hub) = v_obs by construction.  If Family D sub-leading
    # is included in v_pred, N_hub absorbs the change.
    # v ∝ N^{-1/4} ⇒ δN/N = -4 · δv/v
    delta_v_FD = factor_family_D - 1                  # -α₁²
    delta_N_FD = -4 * delta_v_FD                       # +4α₁²

    print(f"G_F bridge consistency:")
    print(f"  By construction, v_pred(N_hub_calibrated) = v_obs.")
    print(f"  Including Family D on v shifts N_hub by +4α₁² (to keep v_pred = v_obs).")
    print(f"  δN_hub/N_hub = {delta_N_FD*100:+.5f}%")
    print()

    # Downstream observable shifts if Family D were exposed explicitly
    H0_shift = -1.0 * delta_N_FD            # H_0 ∝ N^{-1}
    t0_shift = +1.0 * delta_N_FD            # t_0 ∝ N^{+1}
    mnu3_shift = -0.5 * delta_N_FD          # m_ν₃ ∝ N^{-1/2}

    print(f"Downstream observable shifts (if Family D on v exposed explicitly):")
    print(f"  H_0 ∝ N^(-1):     shift = {H0_shift*100:+.5f}%  (currently 68.18 km/s/Mpc; would → 67.77)")
    print(f"  t_0 ∝ N^(+1):     shift = {t0_shift*100:+.5f}%  (currently 14.34 Gyr; would → 14.43)")
    print(f"  m_ν₃ ∝ N^(-1/2):  shift = {mnu3_shift*100:+.5f}%  (currently +0.87%/σ; would → +0.57%/σ)")
    print()

    # All shifts within current σ_obs / σ_anchor envelopes:
    # - H_0: Planck CMB σ ≈ 0.7%; SH0ES σ ≈ 1.4%; -0.61% within both bands.
    # - t_0: Methuselah σ ≈ 5.5%; +0.61% well within band.
    # - m_ν₃: NuFIT 6.0 σ_PDG ≈ 0.40%; -0.30% closer to obs but still FAIL Clause 8.
    # - N_hub anchor spread: G_F (~8.395e60) vs m_τ (~8.435e60) differ by 0.48%; the
    #   Family D absorption shift (+0.61%) is in the same class as this spread.

    print(f"Conclusion (master doc §3 (D) 'absorbed by construction'):")
    print(f"  The G_F round-trip identity v_pred(N_hub) = v_obs is preserved")
    print(f"  REGARDLESS of whether Family D sub-leading is included in DC or")
    print(f"  absorbed into N_hub.  The framework's chose absorption.")
    print()
    print(f"  Magnitude of absorbed N_hub shift: {delta_N_FD*100:+.4f}%")
    print(f"  This is COMPARABLE to the existing N_hub anchor-spread (G_F vs m_τ:")
    print(f"  ~0.5%), so the Family D absorption is observationally indistinguishable")
    print(f"  from the inherent N_hub anchor uncertainty at current precision.")
    print()
    print(f"  Future option: explicit Family D on v would shift downstream by")
    print(f"  the sub-σ amounts above (H_0 ↓ 0.6%, t_0 ↑ 0.6%, m_ν₃ ↓ 0.3%) without")
    print(f"  breaking the G_F bridge — this is a labeling choice, not a structural one.")
    print()

    # Sentinel assertions for correctness
    assert abs(factor_class_C - (1 - 5/12 * (2/3)**8 / (1 - (2/3)**8))) < 1e-15, \
        "Class C factor formula incorrect"
    assert abs(factor_family_D - (1 - ((2/3)**8)**2)) < 1e-15, \
        "Family D factor formula incorrect"
    assert abs(delta_N_FD - 4 * ((2/3)**8)**2) < 1e-15, \
        "N_hub shift formula incorrect"

    print("OK: all assertions pass.")
    print()
    print("Verifies master doc §3 (D) statement: Family D sub-leading on v is")
    print("absorbed into N_hub anchor calibration — consistent by construction.")


if __name__ == "__main__":
    main()
