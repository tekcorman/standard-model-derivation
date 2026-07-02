#!/usr/bin/env python3
"""
Two-stage RG running probe: scan M_SUSY with framework boundary conditions
and ask "what M_SUSY value closes the gauge cluster at the σ_PDG level?"

CONTEXT
-------
The framework currently runs single-regime MSSM b_i = (33/5, 1, -3) from M_unif
to M_Z and reports 2/6 Clause 8 PASS with residuals -2.52σ to +1.01σ_PDG:

  observable       predicted      PDG     Nσ_PDG     direction
  sin²θ_W(M_Z)     0.23125        0.23121 +0.96σ     slightly high
  g_1(M_Z)         0.46148        0.46144 +0.37σ     fine
  g_2(M_Z)         0.65175        0.65200 -2.52σ     LOW
  g_3(M_Z)         1.21118        1.21800 -1.36σ     LOW
  α_s(M_Z)         0.11674        0.11800 -1.40σ     LOW
  1/α_EM(M_Z)      127.930        127.944 +1.01σ     slightly high

In INVERSE COUPLING space:
  1/α_1 essentially matches PDG (≈ 0)
  1/α_2 slightly too HIGH by ~0.016 (α_2 low)
  1/α_3 too HIGH by ~+0.09 (α_3 low)

Required Δb to close:  Δb_1≈0, Δb_2≈-0.003, Δb_3≈+0.017.

This is the canonical signature of MSSM-breaking threshold corrections at the
TeV scale: between M_SUSY and M_Z one runs with SM b_i = (41/10, -19/6, -7),
producing exactly this sign pattern (less-negative b_3, slight Δb_2, ~zero
Δb_1) integrated over the M_Z-to-M_SUSY span.

The framework's ADOPTED-MSSM-Sb 2026-05-14 PM revision explicitly eliminates
M_SUSY as a parameter to avoid free-parameter fitting. This probe asks the
diagnostic question:

  IF the discipline were relaxed and M_SUSY allowed as a free knob, what
  value closes the cluster?

If optimal M_SUSY lands in the natural SUSY-scale window [≈500 GeV, ≈50 TeV],
the residual pattern is structurally consistent with a missing TeV-scale
threshold — a target for substrate derivation (e.g., via the L_r selection
rule arc that already gives EWSB at L_r = 17 → 173 GeV; an analogous
substrate L_r in [17, 19] would land at the SUSY scale).

If optimal M_SUSY lands outside this window, the residual diagnosis is wrong
and the M_SUSY-threshold hypothesis is ruled out cleanly.

NOT theorem-grade. This is a diagnostic probe. M_SUSY remains a fit
parameter unless and until a substrate derivation closes it independently.

METHOD
------
Two-stage one-loop matching (Martin SUSY primer §6.5, eq. 6.43):

  1/α_i(M_Z) = 1/α_GUT
             - (b_i^MSSM / 2π) · ln(M_SUSY / M_unif)
             - (b_i^SM    / 2π) · ln(M_Z    / M_SUSY)

with b^MSSM = (33/5, 1, -3) above M_SUSY and b^SM = (41/10, -19/6, -7) below.
One-loop β doesn't depend on α itself; matching is just continuity of 1/α_i.
"""

import math
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'predictions'))

from alpha_GUT import predict_alpha_GUT_observed
from predictions.M_unif import predict_M_unif_GeV
from predictions.M_Z import M_Z_GeV as _M_Z
from predictions.M_Pl_natural import M_Pl_GeV

# ------------------------------------------------------------------
# Framework inputs (all theorem-grade or theorem-grade-conditional)
# ------------------------------------------------------------------
ALPHA_GUT = float(predict_alpha_GUT_observed(3, 10))   # dark-corrected
M_UNIF    = float(predict_M_unif_GeV(3, 10, M_Pl_GeV))
M_Z       = float(_M_Z)
HYP_NORM  = 3.0 / 5.0

# ------------------------------------------------------------------
# β-coefficients (Type 3 standard QFT)
# ------------------------------------------------------------------
B_MSSM = (33.0/5.0,  1.0,        -3.0)         # above M_SUSY
B_SM   = (41.0/10.0, -19.0/6.0,  -7.0)         # below M_SUSY (3 gen, 1H)

# ------------------------------------------------------------------
# PDG cluster (mirror of framework's σ_PDG reporting at gauge_unification_full_RG_closure.py)
# ------------------------------------------------------------------
PDG = {
    'sin2_W':       (0.23121,  0.00004),
    'g_1':          (0.46144,  0.00010),     # GUT-norm
    'g_2':          (0.65200,  0.00010),
    'g_3':          (1.21800,  0.00500),
    'alpha_s':      (0.11800,  0.00090),
    'alpha_EM_inv': (127.944,  0.014),
}


def two_stage_run(M_SUSY):
    """1/α_i(M_Z) under two-stage one-loop matching at M_SUSY."""
    log_susy_unif = math.log(M_SUSY / M_UNIF)
    log_z_susy    = math.log(M_Z    / M_SUSY)
    return tuple(
        1.0/ALPHA_GUT
        - (b_high/(2.0*math.pi)) * log_susy_unif
        - (b_low /(2.0*math.pi)) * log_z_susy
        for b_high, b_low in zip(B_MSSM, B_SM)
    )


def observables(inv_alpha):
    inv1, inv2, inv3 = inv_alpha
    a1, a2, a3 = 1.0/inv1, 1.0/inv2, 1.0/inv3
    aY   = HYP_NORM * a1
    sin2 = aY / (a2 + aY)
    aem  = a2 * sin2
    return {
        'sin2_W':       sin2,
        'g_1':          math.sqrt(4*math.pi*a1),
        'g_2':          math.sqrt(4*math.pi*a2),
        'g_3':          math.sqrt(4*math.pi*a3),
        'alpha_s':      a3,
        'alpha_EM_inv': 1.0/aem,
        'inv_a1':       inv1,
        'inv_a2':       inv2,
        'inv_a3':       inv3,
    }


def cluster_residuals(obs):
    """σ_PDG residual on each of the 6 framework observables."""
    out = {}
    for key, (v, sig) in PDG.items():
        out[key] = (obs[key] - v) / sig
    return out


def chi2_independent(obs):
    """χ² over the 3 INDEPENDENT couplings g_1, g_2, g_3 (sin²θ_W, α_s, 1/α_EM
    are derived from these — counting them again would double-count)."""
    s = 0.0
    keep = []
    for key in ('g_1', 'g_2', 'g_3'):
        v, sig = PDG[key]
        r = (obs[key] - v) / sig
        s += r*r
        keep.append((key, r))
    return s, keep


# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------
print("="*78)
print("  Two-stage RG: M_SUSY scan with framework boundary conditions  (2026-05-26)")
print("="*78)
print(f"  α_GUT = {ALPHA_GUT:.6f}  (1/{1.0/ALPHA_GUT:.4f}, dark-corrected)")
print(f"  M_unif = {M_UNIF:.4e} GeV   (theorem-grade-conditional)")
print(f"  M_Z   = {M_Z:.4f} GeV")
print(f"  b^MSSM (above M_SUSY) = {B_MSSM}")
print(f"  b^SM   (below M_SUSY) = {B_SM}")
print()

# ---- Baseline: framework's single-regime (M_SUSY = M_Z) ----
print("-"*78)
print("BASELINE — framework single-regime MSSM (equivalent to M_SUSY = M_Z):")
print("-"*78)
inv_base = two_stage_run(M_Z + 1e-9)   # avoid log(1)=0 edge, behaviorally identical
obs_base = observables(inv_base)
res_base = cluster_residuals(obs_base)
chi2_base, keep_base = chi2_independent(obs_base)
print(f"  1/α_1(M_Z) = {obs_base['inv_a1']:8.4f}    1/α_2(M_Z) = {obs_base['inv_a2']:8.4f}    "
      f"1/α_3(M_Z) = {obs_base['inv_a3']:8.4f}")
for k in ('sin2_W', 'g_1', 'g_2', 'g_3', 'alpha_s', 'alpha_EM_inv'):
    print(f"  {k:>14}: pred={obs_base[k]:10.5f}  PDG={PDG[k][0]:10.5f}  Nσ={res_base[k]:+6.2f}")
print(f"  χ² over independent (g_1,g_2,g_3) = {chi2_base:.3f}   √χ² = {math.sqrt(chi2_base):.3f}")
print()

# ---- Scan ----
print("-"*78)
print("SCAN over M_SUSY ∈ [100 GeV, 10^7 GeV]:")
print("-"*78)
print(f"  {'M_SUSY (GeV)':>14}   {'1/α_1':>7}  {'1/α_2':>7}  {'1/α_3':>7}   "
      f"{'Nσ g1':>6} {'Nσ g2':>6} {'Nσ g3':>6}   {'√χ²':>6}")

scan = [10.0**(2.0 + 0.1*i) for i in range(51)]   # 100 GeV to 10^7 GeV, 0.1 dex
best = None
all_results = []
for m in scan:
    inv_a = two_stage_run(m)
    obs = observables(inv_a)
    chi2, keep = chi2_independent(obs)
    all_results.append((m, obs, chi2, keep))
    if best is None or chi2 < best[2]:
        best = (m, obs, chi2, keep)

for m, obs, chi2, keep in all_results[::2]:   # every 0.2 dex
    rs = dict(keep)
    print(f"  {m:>14.3e}   {obs['inv_a1']:7.3f}  {obs['inv_a2']:7.3f}  {obs['inv_a3']:7.3f}   "
          f"{rs['g_1']:+6.2f} {rs['g_2']:+6.2f} {rs['g_3']:+6.2f}   {math.sqrt(chi2):6.3f}")

# Fine scan around the optimum (Brent-like, 5 refinements)
lo, hi = best[0]/3.0, best[0]*3.0
for _ in range(20):
    a, b = lo + (hi-lo)/3, lo + 2*(hi-lo)/3
    obs_a = observables(two_stage_run(a)); chi2_a, _ = chi2_independent(obs_a)
    obs_b = observables(two_stage_run(b)); chi2_b, _ = chi2_independent(obs_b)
    if chi2_a < chi2_b:
        hi = b
    else:
        lo = a
M_SUSY_opt = 0.5 * (lo + hi)
inv_opt = two_stage_run(M_SUSY_opt)
obs_opt = observables(inv_opt)
res_opt = cluster_residuals(obs_opt)
chi2_opt, keep_opt = chi2_independent(obs_opt)

print()
print("-"*78)
print(f"OPTIMAL M_SUSY = {M_SUSY_opt:.3e} GeV  ({M_SUSY_opt/1000:.2f} TeV)")
print("-"*78)
print(f"  1/α_1(M_Z) = {obs_opt['inv_a1']:8.4f}    1/α_2(M_Z) = {obs_opt['inv_a2']:8.4f}    "
      f"1/α_3(M_Z) = {obs_opt['inv_a3']:8.4f}")
for k in ('sin2_W', 'g_1', 'g_2', 'g_3', 'alpha_s', 'alpha_EM_inv'):
    print(f"  {k:>14}: pred={obs_opt[k]:10.5f}  PDG={PDG[k][0]:10.5f}  Nσ={res_opt[k]:+6.2f}")
print(f"  χ² over independent (g_1,g_2,g_3) = {chi2_opt:.4f}   √χ² = {math.sqrt(chi2_opt):.3f}")

# Diagnosis
print()
print("="*78)
print("DIAGNOSIS")
print("="*78)
if 500.0 <= M_SUSY_opt <= 50000.0:
    print(f"  Optimal M_SUSY = {M_SUSY_opt:.2e} GeV ({M_SUSY_opt/1000:.1f} TeV) sits in the")
    print(f"  NATURAL SUSY-scale window [500 GeV, 50 TeV].")
    print()
    print("  → The residual pattern is structurally consistent with a missing")
    print("    TeV-scale threshold (the canonical MSSM-Sb interpretation).")
    print(f"  → χ² reduction: {chi2_base:.2f} → {chi2_opt:.3f} ({chi2_base/max(chi2_opt,1e-9):.0f}× drop).")
    print("  → A substrate derivation of M_SUSY landing in this range would close")
    print("    the gauge cluster while respecting linter discipline.")
    print()
    print("  Candidate substrate origins worth scoping:")
    print("    (a) L_r selection rule arc: EWSB lives at L_r=17→173 GeV. M_SUSY at")
    print(f"        {M_SUSY_opt:.1f} GeV corresponds to L_r ≈ {math.log(M_SUSY_opt/M_UNIF)/math.log((2.0/3.0)**1):.1f}")
    print( "        under N_attest = 96^L_r (preliminary, exact formula TBD).")
    print("    (b) N_hub × M_Z structural product.")
    print("    (c) Pati-Salam → SM breaking scale (M_R or related).")
elif M_SUSY_opt < 200.0:
    print(f"  Optimal M_SUSY = {M_SUSY_opt:.2e} GeV — below 200 GeV, no two-stage benefit.")
    print()
    print("  → Residual diagnosis WRONG: M_SUSY-threshold hypothesis RULED OUT.")
    print("  → Residuals must originate elsewhere — sector-specific dark correction,")
    print("    M_unif conditional, or another structural feature.")
else:
    print(f"  Optimal M_SUSY = {M_SUSY_opt:.2e} GeV outside the natural SUSY window.")
    if M_SUSY_opt > 50000.0:
        print("  → Residuals push toward HIGH-SCALE SUSY (Split SUSY) or PQ-symmetry scale.")
        print("  → Still a candidate diagnosis but harder to motivate from substrate.")
    print(f"  → χ² reduction: {chi2_base:.2f} → {chi2_opt:.3f}.")

print()
print("Probe outputs are diagnostic only. M_SUSY remains a fit parameter unless")
print("and until a substrate derivation closes it independently.")
print("="*78)
