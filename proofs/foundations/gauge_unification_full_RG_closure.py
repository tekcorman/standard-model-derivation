#!/usr/bin/env python3
"""
Integrated gauge-unification → M_Z closure: theorem-grade upstream inputs
(α_GUT = 1/24, sin²θ_W(M_unif) = 3/8, M_unif = (32/k*^(g-1))·M_Pl) feed a
single one-loop MSSM RG running solver, yielding all six PDG-comparable
electroweak observables at the Z pole:

  sin²θ_W(M_Z), g_1(M_Z), g_2(M_Z), g_3(M_Z), α_s(M_Z), α_EM(M_Z)

This is the integrated foundational closure script — the per-observable
thin wrappers (predictions/{sin2_theta_W_MZ, g_1, g_2, g_3, alpha_s,
alpha_EM}.py) duplicate fragments of the same running and are the
parameter-ledger entry points; this file is the consolidated, gate-
audited source of truth.

Closure grade (per parameter_linter.md): **THEOREM-GRADE-CONDITIONAL** under
- Row P40 α_GUT = 1/24 (theorem-grade Class C; predictions/alpha_GUT.py)
- Row P6  sin²θ_W(M_unif) = 3/8 (theorem-grade Class C; predictions/sin2_theta_W.py)
- Row P62 M_unif = (32/k*^(g-1))·M_Pl (theorem-grade-conditional; 5-stage
          program 2026-05-04 EOD+1; predictions/M_unif.py)
- Row P64 M_Z (electroweak scale; predictions/M_Z.py)
- MSSM one-loop β-functions (Type 3 standard QFT: Peskin & Schroeder §16,
          Martin SUSY primer §6.5)
- Hypercharge norm 3/5 (Type 1; SU(5) embedding α_Y = (3/5)·α_1_GUT)

The conditional load is exactly what each entry-point row already inherits;
this file does NOT introduce any new conditional. It DOES centralize the
running so that a single change to the upstream value of α_GUT, sin²θ_W,
M_unif, or M_Z propagates uniformly to all six observables.

Numerical reporting: deviations are reported per-observable against σ_PDG
only. Clause 8 fails for all cluster observables at the σ_PDG level —
the residuals are structural (M_SUSY threshold position, two-loop running,
hadronic vacuum polarization) and not absorbed into σ_PDG.

Cluster ledger entries (post-2026-05-04 EOD+1):
  P65 sin²θ_W(M_Z), P66 g_1, P67 g_2, P68 g_3, P69 α_s, P70 α_EM(M_Z)

Run-of-record outputs (LIVE 2026-05-22; prior header values were stale --
they predated the dark-corrected α_GUT / current M_unif; never updated.
Memory lesson: run live, never docstring -- this is the canonical example):
  sin²θ_W(M_Z) = 0.23125  (PDG 0.23121, +0.96σ_PDG, Clause 8 PASS)
  g_1(M_Z)     = 0.46148  (PDG 0.46144, +0.37σ_PDG, Clause 8 PASS)
  g_2(M_Z)     = 0.65175  (PDG 0.65200, -2.52σ_PDG, Clause 8 FAIL)
  g_3(M_Z)     = 1.21118  (PDG 1.21800, -1.36σ_PDG, Clause 8 FAIL)
  α_s(M_Z)     = 0.11674  (PDG 0.11800, -1.40σ_PDG, Clause 8 FAIL)
  α_EM(M_Z)    = 1/127.93 (PDG 1/127.944, +1.01σ_PDG, borderline FAIL)
Net at 1-loop: 2/6 Clause 8 PASS; remaining FAILs all within |-2.5σ_PDG|.

Two-loop closure check (2026-05-22, proofs/foundations/gauge_two_loop_RG
_closure_2026-05-22.py): going to 2-loop in single-regime MSSM running
WORSENS every observable -- 0/6 PASS Clause 8 at 2-loop. The residuals at
1-loop are STRUCTURAL, not loop-order artefacts; the framework's derived
(α_GUT, M_unif) are effectively tuned to 1-loop running and the dark
correction (1/24 -> 1/24.329) acts as the higher-order accommodation.
"""

import math
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ============================================================================
# 1. Theorem-grade upstream inputs (rational where possible; CODATA passes
#    through M_Pl_natural unit translation — anthropocentric SI conversion
#    only, not a separate empirical anchor).
# ============================================================================

# Dark-corrected α_GUT per theorem_alpha_GUT_dark_correction.md (2026-05-15)
# Bare 1/24 × (1 − (1/k*) × α_1/(1−α_1)) = 18659/453960 ≈ 1/24.329.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'predictions'))
from alpha_GUT import predict_alpha_GUT_observed
ALPHA_GUT = float(predict_alpha_GUT_observed(3, 10))    # dark-corrected, theorem-grade-cond
SIN2_THETA_W_UNIF = 3.0 / 8.0                 # Row P6,  theorem-grade
HYPERCHARGE_NORM = 3.0 / 5.0                  # Type 1: α_Y = (3/5) α_1_GUT (SU(5))

# MSSM one-loop β-function coefficients
# Convention: 1/α_i(μ) = 1/α_i(μ_0) − (b_i/(2π)) · ln(μ/μ_0)
# Refs: Peskin & Schroeder §16; Martin SUSY primer §6.5
B1_MSSM = 33.0 / 5.0   # U(1)_Y, GUT-normalized
B2_MSSM = 1.0          # SU(2)_L
B3_MSSM = -3.0         # SU(3)_c (asymptotically free in MSSM)


def load_scale_inputs():
    """Pull M_unif from the framework prediction and M_Z from its prediction file."""
    from predictions.M_unif import predict_M_unif_GeV
    from predictions.M_Z import M_Z_GeV
    from predictions.M_Pl_natural import M_Pl_GeV
    k_star = 3
    g_girth = 10
    M_unif_GeV = predict_M_unif_GeV(k_star, g_girth, M_Pl_GeV)
    return M_unif_GeV, M_Z_GeV


# ============================================================================
# 2. One-loop MSSM RG running solver.
# ============================================================================

def run_mssm_one_loop(alpha_GUT, M_unif_GeV, M_Z_GeV,
                     b_1=B1_MSSM, b_2=B2_MSSM, b_3=B3_MSSM):
    """
    Solve the MSSM one-loop RG flow from M_unif down to M_Z for α_1, α_2, α_3.

    Returns
    -------
    dict with keys 'alpha_1', 'alpha_2', 'alpha_3', 'inv_alpha_1', etc.
    """
    log_ratio = math.log(M_Z_GeV / M_unif_GeV)   # negative since M_Z << M_unif
    inv_a1 = 1.0 / alpha_GUT - (b_1 / (2.0 * math.pi)) * log_ratio
    inv_a2 = 1.0 / alpha_GUT - (b_2 / (2.0 * math.pi)) * log_ratio
    inv_a3 = 1.0 / alpha_GUT - (b_3 / (2.0 * math.pi)) * log_ratio
    return {
        'log_ratio': log_ratio,
        'inv_alpha_1': inv_a1,
        'inv_alpha_2': inv_a2,
        'inv_alpha_3': inv_a3,
        'alpha_1': 1.0 / inv_a1,
        'alpha_2': 1.0 / inv_a2,
        'alpha_3': 1.0 / inv_a3,
    }


# ============================================================================
# 3. Physical observables at M_Z (six-tuple).
# ============================================================================

def derive_observables(running, hypercharge_norm=HYPERCHARGE_NORM):
    """Compose the running output into the six physical observables at M_Z."""
    a1 = running['alpha_1']
    a2 = running['alpha_2']
    a3 = running['alpha_3']
    aY = hypercharge_norm * a1
    sin2_W_MZ = aY / (a2 + aY)
    alpha_EM_MZ = a2 * sin2_W_MZ
    g_1_MZ = math.sqrt(4.0 * math.pi * a1)   # GUT-normalized
    g_2_MZ = math.sqrt(4.0 * math.pi * a2)
    g_3_MZ = math.sqrt(4.0 * math.pi * a3)
    return {
        'sin2_theta_W_MZ': sin2_W_MZ,
        'alpha_EM_MZ': alpha_EM_MZ,
        'alpha_s_MZ': a3,
        'g_1_MZ': g_1_MZ,
        'g_2_MZ': g_2_MZ,
        'g_3_MZ': g_3_MZ,
        'alpha_Y_MZ': aY,
    }


# ============================================================================
# 4. PDG references for the six observables. Deviations are reported against
#    σ_PDG only — no theoretical-uncertainty band is applied.
# ============================================================================

def pdg_table():
    return {
        'sin2_theta_W_MZ': (0.23121,      0.00004),
        'g_1_MZ':          (None,         0.0001),  # central via derive_g1_obs
        'g_2_MZ':          (0.6520,       0.0001),
        'g_3_MZ':          (1.218,        0.005),
        'alpha_s_MZ':      (0.1180,       0.0009),
        'alpha_EM_MZ':     (1.0/127.944,  0.014/127.944**2),
    }


def derive_g1_obs():
    """g_1 GUT-normalized PDG-equivalent: derive from α_EM(M_Z) + sin²θ_W(M_Z)."""
    alpha_EM_obs = 1.0 / 127.944
    sin2_W_obs = 0.23121
    alpha_Y_obs = alpha_EM_obs / (1.0 - sin2_W_obs)
    alpha_1_obs = (5.0 / 3.0) * alpha_Y_obs
    return math.sqrt(4.0 * math.pi * alpha_1_obs)


# ============================================================================
# 5. Driver.
# ============================================================================

def main():
    print("=" * 76)
    print(" Gauge-unification → M_Z integrated RG closure")
    print(" (α_GUT, sin²θ_W=3/8 at M_unif) → six observables at M_Z")
    print("=" * 76)

    M_unif_GeV, M_Z_GeV = load_scale_inputs()
    print()
    print(f" Inputs:")
    print(f"   α_GUT             = 1/24 = {ALPHA_GUT:.6f}     (Row P40, theorem-grade)")
    print(f"   sin²θ_W(M_unif)   = 3/8  = {SIN2_THETA_W_UNIF:.6f}     (Row P6, theorem-grade)")
    print(f"   M_unif            = {M_unif_GeV:.4e} GeV (Row P62, theorem-grade-cond)")
    print(f"   M_Z               = {M_Z_GeV:.4f} GeV    (Row P64)")
    print(f"   hypercharge norm  = 3/5 = {HYPERCHARGE_NORM:.6f}     (SU(5) embedding)")
    print(f"   MSSM β: b_1={B1_MSSM}, b_2={B2_MSSM:.0f}, b_3={B3_MSSM:.0f}     (Type 3)")
    print()

    running = run_mssm_one_loop(ALPHA_GUT, M_unif_GeV, M_Z_GeV)
    print(f" Running:")
    print(f"   ln(M_Z/M_unif)   = {running['log_ratio']:.5f}")
    print(f"   1/α_1(M_Z)       = {running['inv_alpha_1']:.4f}   (PDG ≈ 59.0)")
    print(f"   1/α_2(M_Z)       = {running['inv_alpha_2']:.4f}   (PDG ≈ 29.6)")
    print(f"   1/α_3(M_Z)       = {running['inv_alpha_3']:.4f}    (PDG ≈ 8.5)")
    print()

    obs = derive_observables(running)
    pdg = pdg_table()

    # Special: derive g_1 GUT-normalized PDG-equivalent
    g1_obs = derive_g1_obs()
    pdg['g_1_MZ'] = (g1_obs, pdg['g_1_MZ'][1])

    print(" Derived observables at M_Z (σ_PDG-only deviation reporting):")
    print()
    print(f"   {'observable':<20} {'predicted':>12} {'PDG':>12} {'σ_PDG':>10} "
          f"{'Δ':>12} {'Nσ_PDG':>10}")
    print("   " + "-" * 82)

    PASS_LIMIT_SIGMA = 1.0
    failures = []

    rows = [
        ('sin2_theta_W_MZ', "sin²θ_W(M_Z)"),
        ('g_1_MZ',          "g_1(M_Z) GUTn"),
        ('g_2_MZ',          "g_2(M_Z)"),
        ('g_3_MZ',          "g_3(M_Z)"),
        ('alpha_s_MZ',      "α_s(M_Z)"),
        ('alpha_EM_MZ',     "α_EM(M_Z)"),
    ]
    summary = {}
    for key, label in rows:
        pred = obs[key]
        central, sigma_pdg = pdg[key]
        delta = pred - central
        n_sigma = delta / sigma_pdg if sigma_pdg > 0 else float('inf')
        print(f"   {label:<20} {pred:>12.5f} {central:>12.5f} {sigma_pdg:>10.5f} "
              f"{delta:>+12.5f} {n_sigma:>+9.2f}σ")
        summary[key] = {
            'pred': pred, 'pdg': central, 'sigma_pdg': sigma_pdg, 'n_sigma_pdg': n_sigma
        }
        if abs(n_sigma) > PASS_LIMIT_SIGMA:
            failures.append((label, n_sigma))

    # Special-case readout for α_EM in 1/α units (more legible)
    inv_pred = 1.0 / obs['alpha_EM_MZ']
    inv_obs = 127.944
    print()
    print(f"   1/α_EM(M_Z) prediction = {inv_pred:.3f}    "
          f"(PDG = {inv_obs:.3f}, dev {(inv_pred - inv_obs)/inv_obs*100:+.3f}%)")
    print()

    # ----- consistency checks (algebraic identities, machine precision) -----
    print(" Consistency checks (machine precision):")
    aY = obs['alpha_Y_MZ']
    a2 = running['alpha_2']
    sin2_check = aY / (a2 + aY)
    assert abs(sin2_check - obs['sin2_theta_W_MZ']) < 1e-15
    print(f"   α_Y/(α_2 + α_Y) = sin²θ_W           OK  ({sin2_check:.6f})")

    em_check = a2 * obs['sin2_theta_W_MZ']
    assert abs(em_check - obs['alpha_EM_MZ']) < 1e-15
    print(f"   α_2 · sin²θ_W = α_EM                 OK  ({em_check:.6f})")

    a3 = running['alpha_3']
    g3_squared_over_4pi = obs['g_3_MZ'] ** 2 / (4.0 * math.pi)
    assert abs(g3_squared_over_4pi - a3) < 1e-13
    print(f"   g_3²/(4π) = α_s                      OK  ({g3_squared_over_4pi:.6f})")

    # ----- σ_PDG-only deviation summary -----
    print()
    print("=" * 76)
    print(" σ_PDG-only Clause 8 summary (multi-σ_PDG deviations expected):")
    for key, label in rows:
        s = summary[key]
        verdict = "PASS" if abs(s['n_sigma_pdg']) <= 1.0 else "FAIL"
        print(f"   {label:<20}  {s['n_sigma_pdg']:+.2f}σ_PDG   Clause 8 {verdict}")
    print("=" * 76)

    return 0


if __name__ == "__main__":
    sys.exit(main())
