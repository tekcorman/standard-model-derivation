#!/usr/bin/env python3
"""
Canonical prediction file for θ_12_PMNS (PMNS solar mixing angle).

STATUS UNDER PARAMETER LINTER (created 2026-05-02 via combined cleanup walk):
UNIQUE-THEOREM-GRADE for structural form via SU(4)_PS perpendicular-rotation
identity (theorem-grade per an internal working note
P1-P7 + `proofs/flavor/srs_theta12_perp.py` CAS-passing). Labeling layer
data-anchored / non-blocking via inheritance from Row P14 (Angle D verdict).
Clause 7 PASS-CITED; Clause 8 PASS at −0.45σ.

Audit anchor: Row P32 of `docs/parameters/parameter_uniqueness_ledger.md`.

    cos θ_12_PMNS = cos θ_TBM / cos θ_C
                  = √(2/3) / √(1 − V_us²)

where:
    θ_TBM = arctan(1/√2)            (tribimaximal mixing, cos θ_TBM = √(2/3))
    θ_C   = arcsin V_us              (Cabibbo angle from CKM)
    V_us  = 9/40                     (Row P4 theorem-grade)

The structural derivation comes from SU(4)_PS sector orthogonality:
the Cabibbo generator T_C and the TBM generator T_TBM lie in
orthogonal-Killing-form sectors of SU(4)_PS, and spherical Pythagoras
gives the perpendicular-rotation formula above.

Theorem chain (an internal working note §2):
  P1: SU(4)_PS = Spin(6) acts on 8-dim Cl(6,0) Fock space.    [B3]
  P2: 15 = 8 ⊕ 1 ⊕ 3 ⊕ 3̄ orthogonal under Killing form.    [Slansky 1981]
  P3: T_C ≡ (a_1†a_2 + a_2†a_1)/2 lies in the 8 (SU(3)_c).    [direct]
  P4: T_TBM ≡ Σ_i (a_i† + a_i)/(2√3) lies in 3 ⊕ 3̄.           [direct]
  P5: B(T_C, T_TBM) = 0.                                       [P2-P4 + CAS]
  P6: Spherical Pythagoras gives cos θ_TBM = cos θ_12 · cos θ_C.
                                                              [Berger 1987 §18]
  P7: V_us = 9/40 → θ_12 = 33.07°.                            [arithmetic]

Labeling layer (color ≡ generation) is OTHER-SMUGGLE residue inherited
from Row P14, NON-BLOCKING for predictive content per the (Z/2)^3 Angle D
verdict (commit e5ef667). Per an internal working note
§3.1, the identification "T_C = SM Cabibbo generator" requires color ≡
generation; the (Z/2)^3 Angle D audit verifies all 77 prediction VALUES
are invariant under the relabeling, so the residue is empirical anchoring
of names — not a predictive gap.

History:
  - Pre-2026-04-25: BLOCKED on color≡generation gap.
  - 2026-04-25 → 2026-04-30: tracked as 🟡 in target_parameters.md, no
    prediction file. Algebraic content theorem-grade per scoping doc.
  - 2026-04-30: ledger Row P32 graduated to UNIQUE-THEOREM-GRADE for
    structural form; labeling data-anchored / non-blocking via inheritance
    from Row P14 (M1 amplitude-form closure + Angle D verdict).
  - 2026-05-02: prediction file shipped via parameter_linter combined
    cleanup walk. Numerical value 33.07° vs PDG 33.41° = -0.45σ.
"""

# ============================================================
# PARAMETER: θ_12_PMNS (PMNS solar mixing angle)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       θ_12_PMNS = 33.41° ± 0.75°
#              (sin²θ_12 = 0.307 ± 0.013)
# Source:      PDG 2024 Review of Particle Physics, Neutrino mixing
#              (NuFIT 5.3 / 6.0 global fit).
# PDG edition: 2024.

# --- PREDICTED VALUE -----------------------------------------
# Value:       θ_12_PMNS = arccos(√(3200/4557)) ≈ 33.0723°
# Deviation:   −0.34° absolute, −1.02% relative, −0.45σ
# Status:      UNIQUE-THEOREM-GRADE for structural form (SU(4)_PS perp);
#              labeling data-anchored / non-blocking via Row P14 inheritance.
#              Systematic floor: zero (pure structural prediction).
#              Clause 8 PASS.

# --- DERIVED FORMULA -----------------------------------------
# cos θ_12_PMNS = cos θ_TBM / cos θ_C
#
# where:
#   cos θ_TBM = √(2/3) (from tribimaximal mixing's solar angle, exact)
#   cos θ_C   = √(1 - V_us²) = √(1 - 81/1600) = √(1519/1600) = √1519 / 40
#
# Therefore:
#   cos θ_12 = √(2/3) · 40 / √1519 = 40√2 / (√3 · √1519) = 40√2 / √4557
#            = √(3200/4557)
#   θ_12     = arccos(√(3200/4557)) ≈ 33.0723°
#
# Chain:
#   A1 + A2 + B3 (Pati-Salam Cl(6) embedding)
#     → SU(4)_PS = Spin(6) on 8-dim Cl(6,0) Fock space (P1)
#     → 15 = 8 ⊕ 1 ⊕ 3 ⊕ 3̄ orthogonal Killing decomposition (P2, Slansky 1981)
#     → T_C ∈ 8, T_TBM ∈ 3⊕3̄ (P3, P4 direct construction)
#     → B(T_C, T_TBM) = 0 (P5, sector orthogonality)
#     → Spherical Pythagoras: cos θ_TBM = cos θ_12 · cos θ_C (P6, Berger 1987)
#     → V_us = 9/40 (Row P4, theorem-grade Level 2)
#     → θ_TBM = arctan(1/√2), so cos θ_TBM = √(2/3)
#     → θ_12 = arccos(√(2/3) / cos θ_C) = arccos(√(3200/4557)) ≈ 33.07°

# --- INPUTS --------------------------------------------------
# symbol      | value          | status     | predictions/ file               | meaning
# ------------|----------------|------------|----------------------------------|--------
# V_us        | 9/40           | [derived]  | predictions/V_us.py              | Cabibbo CKM entry
# theta_TBM   | arctan(1/√2)   | [derived]  | (tribimaximal mixing, B3-PS)     | TBM solar angle, cos θ_TBM = √(2/3)
# Pati-Salam  |                | [derived]  | predictions/sin2_theta_W.py      | SU(4)_PS embedding (Row 17)

# --- IMPLEMENTATION ------------------------------------------

import math
import functools
import sys
import os
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from V_us import predict_V_us, k_star as _k_us, g as _g_us, N_ATOMS as _N_us


V_us = predict_V_us(_k_us, _g_us, _N_us)

# Tribimaximal solar angle: cos θ_TBM = √(2/3)
cos_theta_TBM_sq = Fraction(2, 3)
cos_theta_TBM = math.sqrt(2.0 / 3.0)

# Cabibbo angle: cos θ_C = √(1 − V_us²)
V_us_sq = V_us ** 2
cos_theta_C = math.sqrt(1 - V_us_sq)

# Perpendicular-rotation identity: cos θ_12 = cos θ_TBM / cos θ_C
cos_theta_12 = cos_theta_TBM / cos_theta_C
theta_12_rad = math.acos(cos_theta_12)
theta_12_deg = math.degrees(theta_12_rad)

# Exact symbolic form: V_us = 9/40 → cos θ_12 = √(3200/4557)
V_us_exact = Fraction(9, 40)
cos_theta_C_sq_exact = 1 - V_us_exact ** 2  # = 1519/1600
cos_theta_12_sq_exact = cos_theta_TBM_sq / cos_theta_C_sq_exact  # = (2/3) / (1519/1600) = 3200/4557

# Observed (PDG 2024 / NuFIT)
theta_12_obs_deg = 33.41
theta_12_unc_deg = 0.75
dev_abs = theta_12_deg - theta_12_obs_deg
dev_rel = dev_abs / theta_12_obs_deg
dev_sigma = dev_abs / theta_12_unc_deg

# Runner-facing canonical aliases (slug = "theta_12_PMNS"): without these
# run_predictions._find_result_vars reverse-engineers predicted from
# dev_sigma. Aliases only; zero computational change.
theta_12_PMNS_pred  = theta_12_deg
theta_12_PMNS_obs   = theta_12_obs_deg
theta_12_PMNS_sigma = theta_12_unc_deg

print("=" * 68)
print("  θ_12_PMNS  --  UNIQUE-THEOREM-GRADE for structural form (SU(4)_PS perp)")
print("=" * 68)
print(f"  V_us            = {V_us:.6f} = 9/40 (Row P4 theorem-grade)")
print(f"  cos θ_TBM       = √(2/3) = {cos_theta_TBM:.10f} (tribimaximal exact)")
print(f"  cos θ_C         = √(1−V_us²) = √({1-float(V_us_sq):.10f}) = {cos_theta_C:.10f}")
print(f"  cos θ_12        = cos θ_TBM / cos θ_C = {cos_theta_12:.10f}")
print(f"  cos² θ_12 (exact) = (2/3)·(1600/1519) = {cos_theta_12_sq_exact} ≈ {float(cos_theta_12_sq_exact):.10f}")
print(f"  θ_12_PMNS       = arccos(√(3200/4557)) = {theta_12_deg:.6f}°")
print()
print(f"  PDG 2024 (NuFIT): {theta_12_obs_deg}° ± {theta_12_unc_deg}°")
print(f"  Deviation       : {dev_abs:+.4f}° ({dev_rel*100:+.3f}%, {dev_sigma:+.2f}σ)")
print()
print("  Gate chain:")
print("    Step 1 [B3 + Slansky 1981]: SU(4)_PS sector orthogonality 15 = 8 ⊕ 1 ⊕ 3 ⊕ 3̄")
print("    Step 2 [Type 2]: T_C ∈ 8, T_TBM ∈ 3⊕3̄ → B(T_C, T_TBM) = 0 (CAS srs_theta12_perp.py)")
print("    Step 3 [Type 3 — Berger 1987 §18]: spherical Pythagoras cos θ_TBM = cos θ_12 · cos θ_C")
print("    Step 4 [Type 4]: V_us = 9/40 from predictions/V_us.py (Row P4)")
print("    Step 5 [Type 2]: θ_12 = arccos(√(3200/4557)) ≈ 33.0723°")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_theta_12_PMNS(V_us, cos_theta_TBM_sq):
    """
    Compute θ_12_PMNS from the SU(4)_PS perpendicular-rotation identity.

    Formula:
        cos θ_12 = cos θ_TBM / cos θ_C
        cos θ_C  = √(1 − V_us²)
        θ_12    = arccos(cos θ_12)  in degrees

    Parameters
    ----------
    V_us : float
        |V_us| (Row P4, framework-derived).
    cos_theta_TBM_sq : float
        cos²(θ_TBM) (B3-PS-derived = 2/3 exact for tribimaximal).

    Returns
    -------
    float
        Predicted θ_12_PMNS in degrees.
    """
    cos_theta_C_sq = 1 - V_us ** 2
    cos_theta_12_sq = cos_theta_TBM_sq / cos_theta_C_sq
    cos_theta_12 = math.sqrt(cos_theta_12_sq)
    theta_12_rad = math.acos(cos_theta_12)
    return math.degrees(theta_12_rad)


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_result = theta_12_deg
    pure_result = predict_theta_12_PMNS(V_us, 2.0 / 3.0)
    print()
    print(f"Implementation: {impl_result:.10f}°")
    print(f"Pure function:  {pure_result:.10f}°")
    assert abs(impl_result - pure_result) < 1e-10, \
        f"Mismatch: {impl_result} vs {pure_result}"
    print("OK: outputs agree.")
    print(f"    θ_12_PMNS = {pure_result:.4f}°  "
          f"(PDG: {theta_12_obs_deg}° ± {theta_12_unc_deg}°, {dev_sigma:+.2f}σ)")
    print("    Rigor status: UNIQUE-THEOREM-GRADE for structural form (SU(4)_PS perp);")
    print("    labeling data-anchored / non-blocking via Row P14 inheritance.")
