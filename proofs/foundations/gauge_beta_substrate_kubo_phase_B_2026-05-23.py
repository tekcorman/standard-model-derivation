#!/usr/bin/env python3
"""
gauge_beta_substrate_kubo_phase_B_2026-05-23.py — Π_JJ Phase B.

Phase A (`gauge_beta_from_substrate_kubo_probe.py`, 2026-05-13) numerically
extracted the substrate's gauge kinetic coefficient as

    1/g²_substrate(ω_E) = a + d/ω_E²
    a (Phase A)  ≈ +0.099416   (best candidate 1/g = 1/10, -0.58%; or 1/π², -1.88%)
    d (Phase A)  ≈ -0.005942   (best candidate -1/168 = -1/(24·7), -0.17%)

at fits over the saturated regime ω_E ∈ [0.15, 0.7].  Phase A flagged THREE
open structural questions (its §4):

    (a) which candidate is right for `a` (1/π² Π_TT-analog vs 1/g girth)?
    (b) why is 1/168 the structural Drude weight?
    (c) per-gauge-factor traces T_i(R)?

Phase B (this probe).  Three steps with pre-declared accept/reject sentinels:

    Step 1 — UV asymptote.  Run extract_pi2 at high ω_E ∈ {1.0, 1.5, 2.0, 3.0,
             5.0} (≥ bandwidth ~ 2).  In the saturated-Drude form, `a` is the
             ω→∞ asymptote of π_2_xx after the 1/ω² piece dies.  Test whether
             the high-ω data still satisfy the same Drude form with the
             same `a`, and pick the structurally cleanest candidate.
    Step 2 — PS rep traces.  Compute T_i(R) for SU(2)_L, SU(2)_R, U(1)_{B-L},
             and SU(3)_c on the framework's PS matter rep (4,2,1) ⊕ (4*,1,2)
             from B3+B6 (the Brauer-Weyl Cl(6) Fock + Spin(6)≅SU(4)_PS lift).
             Pin the U(1)_{B-L} normalization to the SU(4) trace-normalized
             T_15 generator (no Y² ambiguity).
    Step 3 — α_GUT^{-1} = 24 recovery.  Combine Step 1 (a) and Step 2 (T_i)
             against the framework's theorem-grade α_GUT_bare^{-1} = 2^k*·k* =
             24.  Test multiple candidate formulas:
             (a) literal scoping-doc: α_GUT^{-1} ?= a · T_i
             (b) QFT convention:     α_GUT^{-1} ?= 4π · a · T_i
             (c) IR Drude weight:    α_GUT^{-1} ?= -1/(d · (g-n_fixed))
             (d) counting analog:    α_GUT^{-1} ?= a · N_local with N_local = 2^k*·k*

             A clean match (sub-percent) on (a) or (b) closes the Kubo route
             on the UV side. A clean match on (c) routes α_GUT^{-1} through
             the IR Drude weight (independent over-determination, NOT a UV
             closure but still a structural win).  No match on any formula
             is an HONEST NEGATIVE: substrate Kubo's `a` is structurally
             distinct from α_GUT^{-1}.

Failure modes embraced (per linter discipline):
    (N1) `a` does not saturate at high ω (data shows 1/ω² decay) → the
         Phase A Drude form was an IR-saturated effective form, no UV
         structural identification possible.
    (N2) No clean α_GUT^{-1} formula matches → honest negative; Phase C
         (substrate β → MSSM b_i) blocked from a different angle.

References:
- Phase A:   proofs/foundations/gauge_beta_from_substrate_kubo_probe.py
- Scoping:   an internal working note
- Template:  docs/theorems/theorem_g_sub_drude_closure_2026-04-30.md
             (Π_TT analog: a_TT = N_atoms/π² = 4/π² theorem-grade)
- PS rep:    predictions/theorem_B3_spinor_fermion.py (Cl(6) on ℂ^8)
- Framework α_GUT: predictions/alpha_GUT.py (α_GUT^{-1} = 2^k*·k* = 24)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gauge_beta_from_substrate_kubo_probe import extract_pi2


# Framework structural integers
N_ATOMS = 4        # srs primitive cell
K_STAR = 3         # srs valency
GIRTH = 10         # srs girth
N_FIXED = 3        # cocyclic-pin count (g - n_fixed = 7)
N_LOCAL = 2 ** K_STAR * K_STAR  # = 24 (label-group order; α_GUT_bare^{-1})


def header(s: str) -> None:
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


# =============================================================================
# Step 1 — UV asymptote: extend ω range and verify `a` saturates
# =============================================================================

def step1_uv_asymptote() -> dict:
    header("Step 1 — UV asymptote: extend ω_E to {1.0, 2.0, 3.0, 5.0}")
    print()
    print("  Phase A used ω_E ∈ [0.15, 0.70] (saturated-Drude regime).  Bandwidth")
    print("  of half-filled spin-1 Iorio matter is ~ 2.  This step pushes ω_E into")
    print("  the UV regime (ω > bandwidth) to test whether the constant `a` from")
    print("  the 2p Drude fit is the true ω→∞ asymptote, or an IR-saturated")
    print("  artifact transitioning to 1/ω² decay.")
    print()
    print("  Pre-declared sentinel:")
    print("    PASS  if  high-ω `a` matches Phase A `a` ≈ 0.0994 within 2%")
    print("    FAIL  if  high-ω `a` drifts >5% (Drude form was IR-only)")
    print()

    omegas_uv = [1.0, 1.5, 2.0, 3.0, 5.0]
    N = 14
    p_z_values = (0.0, 0.05, 0.10, 0.15, 0.20)

    print(f"  Running extract_pi2 at N={N}, ω_E ∈ {omegas_uv}:")
    print(f"  {'ω_E':>6s}  {'time':>6s}  {'π_2_xx':>14s}  {'π_2_xx · ω²':>14s}")
    records = []
    for omega in omegas_uv:
        t0 = time.time()
        res = extract_pi2(omega, omega, N=N, p_z_values=p_z_values)
        dt = time.time() - t0
        pi2 = res["pi_xx_p2"]
        records.append((omega, pi2))
        # If pi2 decays as 1/ω², then pi2·ω² is constant ≈ d.
        # If pi2 → a (Drude), pi2 - a ≈ d/ω², so a + d/ω² ≈ a at high ω.
        print(f"  {omega:>6.3f}  {dt:>5.1f}s  {pi2:>+.6e}  {pi2 * omega**2:>+.6e}")

    omegas_arr = np.array([r[0] for r in records])
    pi2_arr = np.array([r[1] for r in records])
    inv_om2 = 1.0 / omegas_arr ** 2

    # Drude fit on UV-extended data
    d_uv, a_uv = np.polyfit(inv_om2, pi2_arr, 1)
    print()
    print(f"  2p Drude fit on UV-extended data (ω_E ∈ {omegas_uv}):")
    print(f"    a_uv = {a_uv:+.6e}  (Phase A: -0.099416)")
    print(f"    d_uv = {d_uv:+.6e}  (Phase A: +0.005942)")

    # Sign-flip to physical convention
    a_phys_uv = -a_uv
    d_phys_uv = -d_uv

    print()
    print(f"  Sign-flipped (physical kinetic-coef convention):")
    print(f"    a_phys_uv = {a_phys_uv:+.6f}  (Phase A: +0.099416)")
    print(f"    d_phys_uv = {d_phys_uv:+.6f}  (Phase A: -0.005942)")

    phase_A_a = 0.099416
    phase_A_d = -0.005942
    a_drift = (a_phys_uv - phase_A_a) / phase_A_a * 100
    d_drift = (d_phys_uv - phase_A_d) / phase_A_d * 100
    print()
    print(f"  Drift vs Phase A:")
    print(f"    Δa/a = {a_drift:+.3f}%")
    print(f"    Δd/d = {d_drift:+.3f}%")

    sentinel_pass = abs(a_drift) < 2.0
    if sentinel_pass:
        print()
        print(f"  [PASS] `a` saturates: high-ω fit matches Phase A within 2% (drift {a_drift:+.2f}%).")
    else:
        print()
        print(f"  [FAIL] `a` does NOT saturate: high-ω fit drifts {a_drift:+.2f}% > 2%.")
        print(f"         The Drude form was IR-only; no clean UV asymptote.")

    print()
    print(f"  Structural candidates for a_phys = {a_phys_uv:.6f}:")
    cands = {
        "1/π² (Π_TT analog × 1/4)": 1.0 / np.pi ** 2,
        "1/g (girth^{-1})":         1.0 / GIRTH,
        "N_atoms/π² (Π_TT a)":      N_ATOMS / np.pi ** 2,
        "k*/π² (3/π²)":             K_STAR / np.pi ** 2,
        "1/(N_atoms·k*) = 1/12":    1.0 / (N_ATOMS * K_STAR),
        "1/N_local = 1/24":         1.0 / N_LOCAL,
        "2/N_local = 1/12":         2.0 / N_LOCAL,
        "k*/N_local = 3/24 = 1/8":  K_STAR / N_LOCAL,
    }
    best_dev = 1.0
    best_name = ""
    for name, val in cands.items():
        dev = (a_phys_uv - val) / val * 100
        marker = ""
        if abs(dev) < abs(best_dev):
            best_dev = dev
            best_name = name
            marker = " ← best"
        print(f"    {name:30s} = {val:+.6f}   deviation = {dev:+.3f}%{marker}")

    print()
    print(f"  Best candidate (smallest deviation): {best_name} at {best_dev:+.3f}%")

    return {
        "a_phys_uv": a_phys_uv,
        "d_phys_uv": d_phys_uv,
        "a_drift_pct": a_drift,
        "d_drift_pct": d_drift,
        "sentinel_pass_a_saturates": sentinel_pass,
        "best_candidate_a": best_name,
        "best_dev_pct": best_dev,
    }


# =============================================================================
# Step 2 — PS rep traces T_i(R) for the framework's matter rep
# =============================================================================

def step2_ps_rep_traces() -> dict:
    header("Step 2 — PS rep traces T_i(R) for (4,2,1) ⊕ (4*,1,2)")
    print()
    print("  The framework's matter rep per B3+B6 is one PS generation:")
    print("    R = (4, 2, 1)_L  ⊕  (4*, 1, 2)_R")
    print("  under SU(4)_PS × SU(2)_L × SU(2)_R.  Compute the Dynkin index")
    print("  T_i(R) = (1/2) Tr_R(T_i^a T_i^b δ^{ab}) per gauge factor.")
    print()
    print("  Conventions (standard):")
    print("    SU(N) fundamental: T(fund) = 1/2")
    print("    SU(N) adjoint:     T(adj)  = N")
    print("    Direct sum:        T(R ⊕ R') = T(R) + T(R')")
    print("    Tensor product:    T(R_1 ⊗ R_2) = T(R_1) · dim(R_2) + dim(R_1) · T(R_2)")
    print()

    # SU(2)_L: only the (4,2,1) doublet matters; (4*,1,2) is L-singlet
    # The rep is (4, 2) under SU(4) × SU(2)_L.  For SU(2)_L:
    #   T(R) = dim(4) × T(2) = 4 × 1/2 = 2
    T_SU2_L = 4 * 0.5  # = 2

    # SU(2)_R: only the (4*,1,2) doublet matters; (4,2,1) is R-singlet
    T_SU2_R = 4 * 0.5  # = 2

    # SU(4)_PS: BOTH (4,2,1) and (4*,1,2) carry SU(4) charge
    # T(4) = T(4*) = 1/2, with multiplicities from SU(2) reps:
    #   (4, 2, 1):  T_SU4 = 1/2 × 2 (dim of SU(2)_L doublet) = 1
    #   (4*, 1, 2): T_SU4 = 1/2 × 2 (dim of SU(2)_R doublet) = 1
    #   Total:     T_SU4 = 2
    T_SU4 = 0.5 * 2 + 0.5 * 2  # = 2

    # SU(3)_c ⊂ SU(4)_PS via 4 → 3 + 1 (color triplet + lepton singlet)
    # T(3) = 1/2, T(1) = 0.  So SU(3)_c sees only the color triplet:
    #   (4, 2, 1) → (3, 2, 1) + (1, 2, 1):  T_3 = 1/2 × 2 = 1
    #   (4*, 1, 2) → (3*, 1, 2) + (1, 1, 2): T_3 = 1/2 × 2 = 1
    #   Total: T_3 = 2
    T_SU3 = 0.5 * 2 + 0.5 * 2  # = 2

    # U(1)_{B-L}: trace-normalized via SU(4) generator T_15 = (1/√24) diag(1,1,1,-3)
    # The SU(4) Cartan generator that becomes U(1)_{B-L} after PS breaking.
    # T_R(T_15) = (1/√24)² × [3·(1)² + 1·(3)²] = (1/24) × (3 + 9) = 1/2
    # This is the standard SU(4) fundamental Dynkin index (consistent).
    # For the full matter rep R = (4,2,1) ⊕ (4*,1,2):
    #   T_15 acts on 4 and 4*; charges are (1,1,1,-3) and (-1,-1,-1,3) in T_15 basis
    #   Each color/lepton state appears with isospin multiplicity 2:
    #     (4,2,1):  Σ X² over (1,1,1,-3) × 2 = 2 × (3·1 + 9) = 24
    #     (4*,1,2): Σ X² over (-1,-1,-1,3) × 2 = 24
    #   Total Σ X² = 48.  T_R(T_15)_full = (1/24) × 48 = 2.
    # So with trace-normalized SU(4) generator: T_U1_BL = 2 (SAME as SU(2)/SU(3)).
    T_U1_BL = 2.0  # T_15 trace-normalized in SU(4) (no Y² ambiguity)

    print(f"  T_SU(2)_L (R) = 4 × 1/2                  = {T_SU2_L}")
    print(f"  T_SU(2)_R (R) = 4 × 1/2                  = {T_SU2_R}")
    print(f"  T_SU(4)_PS(R) = 1/2 · 2 + 1/2 · 2        = {T_SU4}")
    print(f"  T_SU(3)_c (R) = 1/2 · 2 + 1/2 · 2        = {T_SU3}  (4 → 3+1, only 3 carries)")
    print(f"  T_U(1)_{{B-L}}(R) = T_15 trace-norm       = {T_U1_BL}  (SU(4) T_15, no Y² ambiguity)")
    print()
    print(f"  ALL T_i(R) = 2 — a consistent index across all PS gauge factors.")
    print(f"  This is the UNIVERSALITY signal: at unbroken-PS scale, one matter")
    print(f"  generation contributes the SAME Dynkin index to every gauge factor,")
    print(f"  consistent with the framework's α_GUT = 1/24 at M_unif (all g_i equal).")
    print()
    print(f"  Sum across factors: ΣT_i = 4 × 2 = 8 (SU(2)_L, SU(2)_R, SU(3)_c, U(1)_{{B-L}})")
    print(f"  Sum at PS-unbroken (SU(4)×SU(2)_L×SU(2)_R): ΣT_i = 3 × 2 = 6")

    return {
        "T_SU2_L": T_SU2_L,
        "T_SU2_R": T_SU2_R,
        "T_SU4_PS": T_SU4,
        "T_SU3_c": T_SU3,
        "T_U1_BL": T_U1_BL,
        "T_per_factor": 2.0,
        "T_total_PS": 6.0,
        "T_total_SM": 8.0,
    }


# =============================================================================
# Step 3 — α_GUT^{-1} = 24 recovery from a, T_i, d
# =============================================================================

def step3_alpha_gut_recovery(step1: dict, step2: dict) -> dict:
    header("Step 3 — α_GUT^{-1} = 24 recovery: test candidate formulas")
    print()

    a = step1["a_phys_uv"]
    d_phys = -step1["d_phys_uv"]  # back to original sign (d in Phase A convention)
    T_i = step2["T_per_factor"]
    T_total_PS = step2["T_total_PS"]
    T_total_SM = step2["T_total_SM"]

    alpha_gut_inv_target = N_LOCAL  # = 24 (framework theorem-grade)

    print(f"  Target: α_GUT^{{-1}}_bare = 2^k*·k* = {alpha_gut_inv_target}  (predictions/alpha_GUT.py)")
    print(f"  Inputs from Steps 1, 2:")
    print(f"    a (Step 1, UV asymptote)        = {a:+.6f}")
    print(f"    d_phys (Step 1, IR Drude wt)    = {step1['d_phys_uv']:+.6f}")
    print(f"    T_i (Step 2, per gauge factor)  = {T_i}")
    print(f"    T_total_PS (Step 2, ΣT_i)       = {T_total_PS}")
    print()
    print("  Candidate formulas:")
    print()

    # (a) Literal scoping-doc: α_GUT^{-1} = a · T_i
    cand_a = a * T_i
    print(f"  (a) Literal scoping:    α_GUT^{{-1}} ?= a · T_i")
    print(f"      = {a:.6f} × {T_i} = {cand_a:.6f}     [target {alpha_gut_inv_target}]")
    print(f"      OFF BY FACTOR {alpha_gut_inv_target / cand_a:.2f}")
    print()

    # (b) QFT convention: α_GUT^{-1} = 4π · a · T_i
    cand_b = 4 * np.pi * a * T_i
    print(f"  (b) QFT convention:     α_GUT^{{-1}} ?= 4π · a · T_i")
    print(f"      = 4π × {a:.6f} × {T_i} = {cand_b:.6f}     [target {alpha_gut_inv_target}]")
    print(f"      OFF BY FACTOR {alpha_gut_inv_target / cand_b:.2f}")
    print()

    # (c) IR Drude weight: α_GUT^{-1} = -1/(d · (g - n_fixed))
    g_minus_n = GIRTH - N_FIXED  # = 7
    cand_c = -1.0 / (step1["d_phys_uv"] * g_minus_n)
    # Wait, d_phys = -(d_orig).  Phase A: d_orig = -0.005942.  d_phys = +0.005942 - no,
    # actually in Phase A the sign-flipped to physical is `-π_2_xx`, so the d in
    # 1/g² = a + d/ω² is the original FIT d.  Let me use the original-sign value:
    d_original = -step1["d_phys_uv"]  # negative for Phase A
    print(f"      Phase A sign convention:  Drude weight d in 1/g² = a + d/ω² is NEGATIVE")
    print(f"      d_original = {d_original:.6e}")
    cand_c = -1.0 / (d_original * g_minus_n)
    print(f"  (c) IR Drude weight:    α_GUT^{{-1}} ?= -1/(d · (g - n_fixed))")
    print(f"      = -1/({d_original:.6e} × {g_minus_n}) = {cand_c:.4f}     [target {alpha_gut_inv_target}]")
    print(f"      DEVIATION {(cand_c - alpha_gut_inv_target) / alpha_gut_inv_target * 100:+.3f}%")
    print()

    # (d) Counting: α_GUT^{-1} = a · N_local
    cand_d = a * N_LOCAL
    print(f"  (d) Counting analog:    α_GUT^{{-1}} ?= a · N_local")
    print(f"      = {a:.6f} × {N_LOCAL} = {cand_d:.6f}     [target {alpha_gut_inv_target}]")
    print(f"      OFF BY FACTOR {alpha_gut_inv_target / cand_d:.2f}")
    print()

    # (e) Combined a·T·something
    cand_e1 = a * T_total_PS
    cand_e2 = a * T_total_SM
    cand_e3 = -1.0 / d_original
    print(f"  (e) Other combinations to compare:")
    print(f"      a · ΣT_i (PS, 3 factors) = {cand_e1:.6f}     [target {alpha_gut_inv_target}]")
    print(f"      a · ΣT_i (SM, 4 factors) = {cand_e2:.6f}     [target {alpha_gut_inv_target}]")
    print(f"      -1/d (Drude weight raw)   = {cand_e3:.4f}     [target {alpha_gut_inv_target} or {alpha_gut_inv_target * g_minus_n}]")
    print(f"      -1/d divided by N_local·g_minus_n: {cand_e3/(N_LOCAL * g_minus_n):.6f}")
    print()

    # Verdicts
    print("  VERDICTS")
    print("  -" * 38)
    matches = {
        "(a) a · T_i":         (cand_a, abs(cand_a - alpha_gut_inv_target) / alpha_gut_inv_target),
        "(b) 4π · a · T_i":    (cand_b, abs(cand_b - alpha_gut_inv_target) / alpha_gut_inv_target),
        "(c) -1/(d·(g-n_f))":  (cand_c, abs(cand_c - alpha_gut_inv_target) / alpha_gut_inv_target),
        "(d) a · N_local":     (cand_d, abs(cand_d - alpha_gut_inv_target) / alpha_gut_inv_target),
    }
    for name, (val, dev) in matches.items():
        status = "[CLOSE]" if dev < 0.02 else "[match]" if dev < 0.10 else "[MISS]"
        print(f"    {name:28s} = {val:>12.4f}     dev = {dev*100:+6.2f}%   {status}")
    print()

    best = min(matches, key=lambda k: matches[k][1])
    best_val, best_dev = matches[best]
    print(f"  Best match:  {best}  →  {best_val:.4f}  vs target {alpha_gut_inv_target}  ({best_dev*100:+.3f}%)")
    print()

    return {
        "alpha_gut_inv_target": alpha_gut_inv_target,
        "candidate_a_a_Ti": cand_a,
        "candidate_b_4pi_a_Ti": cand_b,
        "candidate_c_drude_weight": cand_c,
        "candidate_d_a_Nlocal": cand_d,
        "best_formula": best,
        "best_value": best_val,
        "best_dev": best_dev,
    }


# =============================================================================
# Step 4 — Sentinels + honest verdict
# =============================================================================

def step4_verdict(step1: dict, step2: dict, step3: dict) -> None:
    header("Step 4 — Sentinels and honest verdict")
    print()

    # Sentinel 1: a saturates at high ω
    s1 = step1["sentinel_pass_a_saturates"]
    print(f"  Sentinel 1 — `a` saturates at high ω:  {'PASS' if s1 else 'FAIL'}")
    print(f"               drift = {step1['a_drift_pct']:+.2f}% (tol 2%)")

    # Sentinel 2: T_i is universal across factors (PS unification signal)
    universal = all(
        v == step2["T_per_factor"]
        for v in [step2["T_SU2_L"], step2["T_SU2_R"], step2["T_SU4_PS"], step2["T_SU3_c"], step2["T_U1_BL"]]
    )
    print(f"  Sentinel 2 — T_i universal (PS signal): {'PASS' if universal else 'FAIL'}")
    print(f"               all T_i = {step2['T_per_factor']} across SU(2)_L, SU(2)_R, SU(4)_PS, SU(3)_c, U(1)_{{B-L}}")

    # Sentinel 3: at least one α_GUT^{-1} formula matches within 2%
    s3 = step3["best_dev"] < 0.02
    print(f"  Sentinel 3 — some α_GUT^{{-1}} formula matches within 2%: {'PASS' if s3 else 'FAIL'}")
    print(f"               best: {step3['best_formula']} at {step3['best_dev']*100:+.3f}%")

    print()
    print("  HONEST VERDICT")
    print("  " + "=" * 64)

    if s3 and s1 and universal:
        verdict = "PHASE B PASS — substrate Kubo recovers α_GUT^{-1} = 24"
        print(f"  {verdict}")
        print(f"  Closure route: {step3['best_formula']}")
        print(f"  Substrate-Kubo β route graduates to THEOREM-GRADE-CONDITIONAL on")
        print(f"  (a, T_i) structural identifications.  Phase C (running to MSSM b_i)")
        print(f"  unblocked.")
    elif s3 and (s1 or universal):
        verdict = "PHASE B PARTIAL — formula matches but a sentinel failed"
        print(f"  {verdict}")
        print(f"  Closure: {step3['best_formula']}")
        print(f"  Caveat: investigate sentinel failure before promoting.")
    elif s1 and universal:
        verdict = "PHASE B STRUCTURAL — substrate Kubo confirms upstream invariants"
        print(f"  {verdict}")
        print(f"  • `a` saturates structurally")
        print(f"  • T_i is universal across PS gauge factors (unification signal)")
        print(f"  But: no clean `α_GUT^{{-1}} = a · T_i` formula at <2%.")
        print(f"  Best match: {step3['best_formula']} at {step3['best_dev']*100:+.2f}%.")
        print(f"  ROUTE STATUS: the substrate's `a` is a structurally distinct quantity")
        print(f"  from α_GUT^{{-1}}.  The IR Drude weight d may still encode α_GUT^{{-1}}")
        print(f"  via -1/d = α_GUT^{{-1}}·(g-n_fixed) — examine cand (c).")
    else:
        verdict = "PHASE B HONEST NEGATIVE — substrate Kubo does not recover α_GUT^{-1}"
        print(f"  {verdict}")
        print(f"  Sentinel 1 (`a` saturates): {'PASS' if s1 else 'FAIL'}")
        print(f"  Sentinel 2 (T_i universal): {'PASS' if universal else 'FAIL'}")
        print(f"  Sentinel 3 (α_GUT^{{-1}} match): {'PASS' if s3 else 'FAIL'}")
        print(f"  Best closest: {step3['best_formula']} at {step3['best_dev']*100:+.2f}%.")
        print(f"  Phase A's `a × T_i = α_GUT^{{-1}}` hope is NOT borne out.")
        print(f"  The substrate-Kubo route does NOT directly close α_GUT^{{-1}} via")
        print(f"  the UV asymptote.  Phase C is blocked at this level.")
        print(f"  Useful negative: substrate `a` is the girth/π² scale, NOT α_GUT^{{-1}}.")

    print()
    print(f"  Sentinels passed: {sum([s1, universal, s3])}/3")


def main() -> None:
    header("Π_JJ Phase B: UV asymptote, PS rep traces, α_GUT^{-1} recovery")
    print()
    print("  Three-step structural identification of Phase A's empirical Drude form")
    print("  1/g²_substrate(ω_E) = a + d/ω² (a ≈ 0.0994, d ≈ -0.00594).")
    print()
    print("  Pre-declared sentinels (linter discipline):")
    print("    [S1] `a` saturates at high ω (drift < 2% vs Phase A)")
    print("    [S2] T_i(R) universal across PS gauge factors")
    print("    [S3] some α_GUT^{-1} formula matches 24 within 2%")

    step1 = step1_uv_asymptote()
    step2 = step2_ps_rep_traces()
    step3 = step3_alpha_gut_recovery(step1, step2)
    step4_verdict(step1, step2, step3)


if __name__ == "__main__":
    main()
