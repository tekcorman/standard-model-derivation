#!/usr/bin/env python3
"""
---
derives: Y_p
inputs:
  - T_BBN_weak_freezeout       # framework T_BBN-1 (predictions/T_BBN_weak_freezeout.py)
  - Q_np                       # neutron-proton mass splitting (atomic-physics anchor)
script_version: 2.0.0
doc: predictions/Y_p_derivation.md
mechanism: structural
rigor_status: theorem-grade-structural-bounded-by-substrate-thermal-coupling
phase: III-derivative
---

Y_p — primordial ⁴He mass fraction (BBN abundance).

Simple analytic estimate via n/p ratio at weak freeze-out + Boltzmann
suppression:

    Y_p ≈ 2 · (n/p)_final / [1 + (n/p)_final]

Where (n/p)_freeze = exp(-Q_np / T_BBN-1) at weak freezeout, and
(n/p)_final accounts for neutron β-decay during the cosmic time from
T_BBN-1 to T_BBN-D bottleneck.

α-CONVENTION (post-2026-05-27 α-audit): Framework Phase IIb uses α = 1/2
(instantaneous), giving T_BBN-1 ≈ 0.39 MeV (vs ΛCDM 0.7 MeV) per
`predictions/T_BBN_weak_freezeout.py`. Framework T_BBN-1 LOWER than
ΛCDM because substrate H lacks √g_* prefactor that ΛCDM Friedmann carries.

This predicts a DIFFERENT n/p ratio at freeze-out → different Y_p.

Within-class residues:
  (i) Substrate-thermal coupling: framework H lacks √g_*-equivalent factor
  (ii) Need-B: Q_np precision via quark masses (BR4 closure-NEGATIVE)
  (iii) Neutron β-decay correction during cosmic time
  (iv) Full BBN reaction-network for precision

Simple framework estimate (with decay factor 0.7) gives Y_p ≈ 0.05 —
**far below observed Y_p ≈ 0.245 (Aver et al. 2020)**.

This is a SIGNIFICANT framework-distinct prediction: framework's
substrate H (without √g_* factor) shifts T_BBN-1 DOWN, which lowers
n/p_freeze, which lowers Y_p.

If framework Y_p ≈ 0.05 is the correct prediction (without √g_*-
equivalent substrate-thermal coupling factor), this is a
**falsification test** distinguishing framework from observation.

The standard cosmology Y_p ≈ 0.245 ± 0.003 (PDG 2024) is one of the
most precisely measured cosmological observables. A framework-distinct
Y_p prediction at ~0.05 would be FALSIFIED by current data unless
the substrate-thermal coupling structurally produces a √g_*-equivalent
factor in H_coasting.

Honest disposition: this prediction is a FALSIFICATION CANDIDATE for
framework's coasting cosmology + Phase IIb T_BBN-1 + substrate-thermal-
coupling gap. Per α-audit verdict:
the +69σ → -65σ flip when correcting α from 25/48 to 1/2 confirms the Y_p
falsification is structurally driven by the missing √g_* factor in
substrate H, not by α-choice ambiguity. Open per Need-B + substrate-
thermal coupling.
"""

import sys
import os
import math
import functools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from T_BBN_weak_freezeout import T_BBN_weak_pred_GeV


@functools.lru_cache(maxsize=None)
def predict_Y_p(T_BBN_weak_GeV, Q_np_GeV, neutron_decay_factor):
    """Y_p from n/p ratio at weak freezeout + decay correction.

    Pure function — NO defaults.

    Parameters
    ----------
    T_BBN_weak_GeV : float, weak freeze-out temperature (framework)
    Q_np_GeV : float, m_n - m_p mass splitting
    neutron_decay_factor : float, multiplicative correction for n β-decay
                          during cosmic time from T_BBN-1 to T_BBN-D
    """
    n_over_p_freeze = math.exp(-Q_np_GeV / T_BBN_weak_GeV)
    n_over_p_final = n_over_p_freeze * neutron_decay_factor
    Y_p = 2 * n_over_p_final / (1 + n_over_p_final)
    return Y_p


# --- INPUTS ----------------------------------------------------
# Q_np = m_n - m_p ≈ 1.2933 MeV (PDG; framework: precision-bounded by Need-B)
# This is an EXTERNAL ANCHOR (not framework-derivable at high precision per BR4).
Q_np_MeV = 1.2933
Q_np_GeV = Q_np_MeV * 1e-3

# Neutron β-decay correction during cosmic time from T_BBN-1 to T_BBN-D.
# Standard cosmology: ~exp(-Δt/τ_n) with Δt ~ 200-300s and τ_n ≈ 880s
# → factor ~0.7. Under framework instantaneous T-N scaling (α=1/2), this
# factor differs. Schematic: use ΛCDM-anchored factor 0.7 (framework
# correction is multi-sprint per coasting-cosmic-time question).
NEUTRON_DECAY_FACTOR_LCDM = 0.7   # multiplicative; framework correction = open


Y_p_pred = predict_Y_p(T_BBN_weak_pred_GeV, Q_np_GeV, NEUTRON_DECAY_FACTOR_LCDM)

# Also compute "no-decay" baseline for comparison
Y_p_no_decay = predict_Y_p(T_BBN_weak_pred_GeV, Q_np_GeV, 1.0)


# --- OBSERVED VALUE -------------------------------------------
# Y_p = 0.245 ± 0.003 (Aver et al. 2020 / PDG 2024 BBN consensus)
Y_p_obs = 0.245
Y_p_sigma = 0.003

dev_sigma = (Y_p_pred - Y_p_obs) / Y_p_sigma
dev_pct = (Y_p_pred - Y_p_obs) / Y_p_obs * 100

print("=" * 68)
print("  Y_p (BBN ⁴He mass fraction) -- FALSIFICATION CANDIDATE")
print("=" * 68)
print(f"  DAG inputs:")
print(f"    T_BBN-1 = {T_BBN_weak_pred_GeV*1e3:.3f} MeV (framework, α=1/2 instantaneous)")
print(f"    Q_np    = {Q_np_MeV} MeV (external, bounded by Need-B)")
print(f"    n β-decay factor = {NEUTRON_DECAY_FACTOR_LCDM} (ΛCDM-anchored; framework correction OPEN)")
print()
print(f"  n/p_freeze (framework) = exp(-{Q_np_MeV}/{T_BBN_weak_pred_GeV*1e3:.2f}) = {math.exp(-Q_np_GeV/T_BBN_weak_pred_GeV):.4f}")
print(f"  n/p_final (framework)  = {math.exp(-Q_np_GeV/T_BBN_weak_pred_GeV) * NEUTRON_DECAY_FACTOR_LCDM:.4f}")
print()
print(f"  Y_p (framework, with decay)   = {Y_p_pred:.4f}")
print(f"  Y_p (framework, no decay)     = {Y_p_no_decay:.4f}")
print(f"  Y_p (observed, Aver 2020)     = {Y_p_obs} ± {Y_p_sigma}")
print(f"  Deviation                     = {dev_sigma:+.2f}σ ({dev_pct:+.2f}%)")
print()
print("  Framework predicts Y_p LOWER than observed because substrate H lacks")
print("  √g_* prefactor (≈ 3.28 in ΛCDM). T_BBN-1 = 0.39 MeV (vs ΛCDM 0.7 MeV)")
print("  → low n/p_freeze ≈ 0.04 → low Y_p ≈ 0.05.")
print("  Per α-audit verdict (2026-05-27 EOD+1): falsification is structurally")
print("  robust — both natural α choices fail (α=1/2 → -65σ; α=25/48 → +69σ).")
print("  Resolution requires substrate-thermal-coupling structural extension or")
print("  full BBN reaction network under coasting (multi-sprint).")


if __name__ == "__main__":
    Y = predict_Y_p(T_BBN_weak_pred_GeV, Q_np_GeV, NEUTRON_DECAY_FACTOR_LCDM)
    print(f"\nOK: Y_p = {Y:.4f} (framework Phase III-derivative)")
