#!/usr/bin/env python3
"""
Two-loop MSSM RG closure for the six gauge-cluster observables at M_Z.

The framework's existing closure
(`proofs/foundations/gauge_unification_full_RG_closure.py`) is ONE-loop and
its six observables at M_Z all FAIL Clause 8 vs sigma_PDG by +0.3% to
+2.8%. M_SUSY is eliminated (no threshold scale), so the conventional
"M_SUSY threshold position" attribution of the residual is unavailable.
The remaining structural attributions named in the closure header are
two-loop running and hadronic vacuum polarisation.

The framework's own Fork-1 lesson is explicit: never assert what a
higher-order correction does without doing the calculation. So: redo the
closure at TWO loops -- single-regime MSSM, no thresholds, the framework's
derived alpha_GUT and M_unif -- and see whether the +0.3%-+2.8% residuals
tighten, worsen, or stay.

INPUTS (all theorem-grade-conditional from the framework):
  - alpha_GUT (dark-corrected) -- predictions/alpha_GUT.py
  - sin^2 theta_W(M_unif) = 3/8 -- follows from gauge unification +
        hypercharge norm 3/5 (not an independent input)
  - M_unif = (32 / k*^(g-1)) M_Pl -- predictions/M_unif.py
  - M_Z -- predictions/M_Z.py
  - MSSM single-regime running, M_SUSY eliminated -- per ADOPTED-MSSM-Sb
  - Two-loop MSSM beta-matrix b_ij from the framework's own
        mssm_two_loop_RG_envelope.py (Martin SUSY primer Sec 6.5.2-3).

TWO-LOOP RGE (gauge sector, Yukawa contributions dropped at leading order):
        dx_i / dt  =  -b_i / (2 pi)  -  Sum_j  b_ij / (8 pi^2) * (1 / x_j)
where x_i = 1/alpha_i (GUT-normalised alpha_1 = (5/3) alpha_Y), t = ln(mu).

GATES (each compared to the framework's 1-loop result and PDG):

  G1  REPRODUCE THE FRAMEWORK'S 1-LOOP CLOSURE (sanity check on inputs +
      conventions). Six observables at M_Z must match the run-of-record.

  G2  TWO-LOOP CLOSURE. Integrate the 2-loop RGE from M_unif down to M_Z
      with the framework's derived inputs. Report the six observables.

  G3  RESIDUAL COMPARISON. For each observable: how does the deviation
      from PDG change going from 1-loop to 2-loop? Tightens, worsens, or
      flips sign?

  G4  CLAUSE-8 STATUS. Count how many observables PASS Clause 8 vs
      sigma_PDG at 2-loop, vs 1-loop's 0/6.

VERDICT IS THE OUTCOME, NOT PRE-DECLARED.

NOTE ON STALE DOCSTRING. The 1-loop file's docstring header lists
run-of-record numbers (sin2_theta_W = 0.23027, g_3 = 1.2349, alpha_s =
0.1213, etc.) that DO NOT match the live output. The current live 1-loop
result is sin2_theta_W = 0.23125, g_3 = 1.2112, alpha_s = 0.11674 -- much
closer to PDG (2/6 PASS Clause 8, not 0/6 as the header implies). The
header is stale, presumably from an older M_unif or alpha_GUT input;
the survey that read the header propagated the staleness. Memory lesson
('run live, never docstring') applies here exactly.
"""

import math
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'predictions'))

from alpha_GUT import predict_alpha_GUT_observed
from M_unif import predict_M_unif_GeV
from M_Z import M_Z_GeV
from M_Pl_natural import M_Pl_GeV

# ===========================================================================
# inputs (theorem-grade-conditional from the framework)
# ===========================================================================
K_STAR = 3
GIRTH = 10
ALPHA_GUT = float(predict_alpha_GUT_observed(K_STAR, GIRTH))   # ~ 1/24.329
M_UNIF = predict_M_unif_GeV(K_STAR, GIRTH, M_Pl_GeV)
M_Z = M_Z_GeV
HYPERCHARGE_NORM = 3.0 / 5.0                                    # SU(5)

# MSSM beta-coefficients (the framework's own values, from
# mssm_two_loop_RG_envelope.py; Martin SUSY primer Sec 6.5.2-3)
B = np.array([33.0/5.0, 1.0, -3.0])
B_IJ = np.array([
    [199.0/25.0, 27.0/5.0, 88.0/5.0],
    [9.0/5.0,    25.0,     24.0    ],
    [11.0/5.0,   9.0,      14.0    ],
])

# PDG anchors (per gauge_unification_full_RG_closure.py header)
PDG = {
    "sin2_theta_W":   (0.23121, 0.00018),
    "alpha_EM_inv":   (127.944, 0.014),
    "alpha_s":        (0.1180,  0.0009),
    "g_1_derived":    (0.4614,  0.4614 * 0.0002),    # derived from PDG sin2 + alpha_em
    "g_2":            (0.6520,  0.6520 * 0.001),
    "g_3":            (1.2180,  0.003),
}

# ===========================================================================
# observable extraction (same convention as the framework's 1-loop closure)
# ===========================================================================
def observables(alpha_1, alpha_2, alpha_3):
    alpha_Y = HYPERCHARGE_NORM * alpha_1
    sin2 = alpha_Y / (alpha_2 + alpha_Y)
    cos2 = 1.0 - sin2
    alpha_EM = alpha_2 * sin2                                   # = alpha_Y cos2
    g_1 = math.sqrt(4 * math.pi * alpha_1)
    g_2 = math.sqrt(4 * math.pi * alpha_2)
    g_3 = math.sqrt(4 * math.pi * alpha_3)
    return {
        "sin2_theta_W": sin2,
        "alpha_EM_inv": 1.0 / alpha_EM,
        "alpha_s":      alpha_3,
        "g_1_derived":  g_1,
        "g_2":          g_2,
        "g_3":          g_3,
    }

# ===========================================================================
# one-loop running (closed form)
# ===========================================================================
def run_one_loop():
    log_r = math.log(M_Z / M_UNIF)
    inv = [1.0/ALPHA_GUT - (B[i] / (2*math.pi)) * log_r for i in range(3)]
    return observables(1.0/inv[0], 1.0/inv[1], 1.0/inv[2])

# ===========================================================================
# two-loop running (numerical integration)
# ===========================================================================
def run_two_loop():
    def rhs(t, x):
        # dx_i/dt = -b_i/(2 pi) - sum_j b_ij/(8 pi^2) / x_j
        return -B/(2*math.pi) - (B_IJ @ (1.0/x)) / (8*math.pi*math.pi)
    t0 = math.log(M_UNIF)
    tf = math.log(M_Z)
    x0 = np.array([1.0/ALPHA_GUT, 1.0/ALPHA_GUT, 1.0/ALPHA_GUT])
    sol = solve_ivp(rhs, [t0, tf], x0, method='DOP853',
                    rtol=1e-12, atol=1e-14, dense_output=False)
    assert sol.success, sol.message
    x_MZ = sol.y[:, -1]
    return observables(1.0/x_MZ[0], 1.0/x_MZ[1], 1.0/x_MZ[2])

# ===========================================================================
# probe gates
# ===========================================================================
gates = []
ONE = run_one_loop()
TWO = run_two_loop()

# expected 1-loop values from the framework's run-of-record (the header)
# framework's CURRENT live 1-loop output (verified by running
# gauge_unification_full_RG_closure.py 2026-05-22). The docstring header in
# that file gives DIFFERENT, STALE values (0.23027, 0.4628, 0.6554, 1.2349,
# 0.1213) -- run-of-record never updated. The live numbers below are the
# truth. Memory lesson: run live, never docstring.
EXPECTED_1L = {
    "sin2_theta_W": 0.23125,
    "g_1_derived":  0.46148,
    "g_2":          0.65175,
    "g_3":          1.21118,
    "alpha_s":      0.11674,
}

# ---------------------------------------------------------------------------
# G1 -- reproduce the framework's 1-loop closure
# ---------------------------------------------------------------------------
ok_1L = all(abs(ONE[k] - EXPECTED_1L[k]) / EXPECTED_1L[k] < 1e-3
            for k in EXPECTED_1L)
gates.append((
    "G1 reproduce the framework's 1-loop closure (sanity check on inputs "
    "and conventions)",
    ok_1L,
    "; ".join(f"{k} 1L={ONE[k]:.5g} (expected {EXPECTED_1L[k]:.5g})"
              for k in EXPECTED_1L)))

# ---------------------------------------------------------------------------
# G2 -- 2-loop closure (the actual computation)
# ---------------------------------------------------------------------------
gates.append((
    "G2 two-loop closure: integrate the MSSM 2-loop RGE from M_unif to M_Z "
    "with the framework's derived inputs (single-regime; no thresholds)",
    True,                                          # diagnostic
    "; ".join(f"{k}={TWO[k]:.5g}" for k in TWO)))

# ---------------------------------------------------------------------------
# G3 -- residual comparison
# ---------------------------------------------------------------------------
rows = []
n_tightened = 0
n_pass_1l = 0
n_pass_2l = 0
for k, (val, sig) in PDG.items():
    d1 = ONE[k] - val
    d2 = TWO[k] - val
    pct1 = 100.0 * d1 / val
    pct2 = 100.0 * d2 / val
    sig1 = d1 / sig
    sig2 = d2 / sig
    rows.append((k, val, sig, ONE[k], TWO[k], pct1, pct2, sig1, sig2))
    if abs(d2) < abs(d1):
        n_tightened += 1
    if abs(sig1) < 1.0:
        n_pass_1l += 1
    if abs(sig2) < 1.0:
        n_pass_2l += 1
gates.append((
    "G3 residual comparison 1-loop -> 2-loop (per observable, vs PDG)",
    True,                                          # diagnostic
    f"{n_tightened}/6 observables tightened going to 2-loop"))

# ---------------------------------------------------------------------------
# G4 -- Clause-8 status
# ---------------------------------------------------------------------------
gates.append((
    "G4 Clause-8 status: how many of the 6 observables pass |dev| < sigma_PDG",
    True,                                          # diagnostic
    f"1-loop PASS: {n_pass_1l}/6 ; 2-loop PASS: {n_pass_2l}/6"))

# ===========================================================================
print("=" * 80)
print(" TWO-LOOP MSSM GAUGE-RG CLOSURE -- 2026-05-22")
print("=" * 80)
print(f"  alpha_GUT (dark-corrected) = {ALPHA_GUT:.6f}  (1/{1/ALPHA_GUT:.4f})")
print(f"  M_unif = {M_UNIF:.3e} GeV ; M_Z = {M_Z:.4f} GeV ; "
      f"ln(M_Z/M_unif) = {math.log(M_Z/M_UNIF):+.4f}")
print()
print(f"  {'observable':<14s} {'PDG':>12s} {'sigma':>10s} "
      f"{'1-loop':>12s} {'2-loop':>12s} "
      f"{'1L dev':>9s} {'2L dev':>9s} {'1L sig':>8s} {'2L sig':>8s}")
for k, val, sig, v1, v2, p1, p2, s1, s2 in rows:
    print(f"  {k:<14s} {val:>12.5g} {sig:>10.3g} "
          f"{v1:>12.5g} {v2:>12.5g} "
          f"{p1:>+8.3f}% {p2:>+8.3f}% "
          f"{s1:>+7.2f}σ {s2:>+7.2f}σ")
print()
for name, ok, detail in gates:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}")
    print(f"         {detail}")
print("-" * 80)
print()

# verdict assembled from the actual numbers, not pre-declared
tighten_total_sig = sum(abs(s2) - abs(s1) for _, _, _, _, _, _, _, s1, s2 in rows)
print("  VERDICT.\n")
if n_pass_2l == 6:
    print("  All six gauge-cluster observables PASS Clause 8 at two loops.\n"
          "  The 1-loop FAIL was a loop-order artefact, exactly as the Fork-1\n"
          "  lesson warned. The framework's gauge unification is now consistent\n"
          "  with the PDG values at sigma_PDG level, with zero free parameters.")
elif n_pass_2l > n_pass_1l:
    print(f"  PARTIAL TIGHTENING. Two-loop running tightens {n_tightened}/6\n"
          f"  observables; {n_pass_2l}/6 pass Clause 8 at 2-loop vs\n"
          f"  {n_pass_1l}/6 at 1-loop. Total signed sigma improvement = "
          f"{-tighten_total_sig:+.2f}.\n"
          f"  Some genuine residual remains -- candidates: higher orders,\n"
          f"  hadronic vacuum polarisation, MSSM-content adoption.")
elif n_tightened >= 3:
    print(f"  MIXED RESULT. Two-loop tightens {n_tightened}/6 observables but\n"
          f"  worsens the others. {n_pass_2l}/6 pass Clause 8 (vs {n_pass_1l}/6\n"
          f"  at 1-loop). Total signed sigma improvement = "
          f"{-tighten_total_sig:+.2f}.\n"
          f"  Residuals are not purely loop-order; structural attention needed.")
else:
    print(f"  TWO-LOOP DOES NOT FIX THE RESIDUAL. Only {n_tightened}/6\n"
          f"  observables tightened; {n_pass_2l}/6 pass Clause 8.\n"
          f"  The 1-loop FAIL was NOT a loop-order artefact -- the deviation\n"
          f"  is structural and persists at two loops. Honest finding: the\n"
          f"  framework's gauge sector has a genuine ~few-% residual that\n"
          f"  loop order alone cannot explain. Next suspects: hadronic vacuum\n"
          f"  polarisation in alpha_EM(M_Z) matching, the ADOPTED-MSSM-Sb\n"
          f"  matter content adoption, or the M_unif scale (P62 conditional).")
print()
print("=" * 80)
n_real_gates = sum(1 for name, ok, _ in gates if "G1" in name)
sys.exit(0 if ok_1L else 1)
