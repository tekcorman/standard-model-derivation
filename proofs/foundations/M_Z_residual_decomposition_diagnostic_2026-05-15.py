#!/usr/bin/env python3
"""
proofs/foundations/M_Z_residual_decomposition_diagnostic_2026-05-15.py

DIAGNOSTIC — decompose the live M_Z +0.357% residual into its actual
upstream contributors, using the predictions/ DAG as the AUTHORITY
(run the real predict_* functions and perturb them; do NOT trust the
ledger's "M_unif Stage-5" prose attribution).

Per user (2026-05-15 EOD+16): "the DAG predictions are the authority."
Per an internal note / [[audit-for-smuggled-
parameters-2026-05-14]]: diagnose which input actually drives the
residual BEFORE proposing any fix.

M_Z (predictions/M_Z.py) = √π · v · √(α_2(M_Z) + (3/5)α_1(M_Z)),
self-consistent; α_i from 1-loop MSSM single-regime RG run of α_GUT
(dark-corrected) from M_unif down to M_Z.  Upstream inputs:
  v       = predict_v_higgs(...)        [N_hub-anchored to measured G_F]
  α_GUT   = predict_alpha_GUT_observed  [dark-corrected ≈ 1/24.329]
  M_unif  = predict_M_unif_GeV(...)     [≈1.985e16, Row P62 −0.76% vs bench]
  b_1,b_2 = 33/5, 1                     [1-loop MSSM, NO 2-loop/M_SUSY]

This probe:
 (A) reproduce the live baseline M_Z; confirm residual vs PDG.
 (B) one-at-a-time: what value would each upstream need so M_Z = PDG?
     Express as the % shift required of that input.  The input needing
     the SMALLEST plausible shift (or the one whose KNOWN residual
     already accounts for it) is the load-bearing contributor.
 (C) sensitivity dM_Z/M_Z per d(input)/input (log-derivative).
 (D) explicit 2-loop-β test: does adding the standard 2-loop MSSM
     term to the RG running move M_Z toward PDG by ~the residual?
 (E) honest verdict: name the load-bearing contributor from the
     numbers, not from ledger prose.
"""
from __future__ import annotations
import os
import sys
import math

PRED = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "predictions")
sys.path.insert(0, PRED)

from d_spatial import predict_d_spatial
from k_star import predict_k_star
from g_girth import predict_g_girth
from alpha_1 import predict_alpha_1
from p_toggle import predict_p_toggle
from V_count import predict_V_count
from N_hub import predict_N_hub
from v_higgs import predict_v_higgs
from M_unif import predict_M_unif_GeV
from M_Pl_natural import M_Pl_GeV
from alpha_GUT import predict_alpha_GUT_observed

d = predict_d_spatial(); k = predict_k_star(d); g = predict_g_girth(k, d)
a1s = predict_alpha_1(k, g)
p_val = predict_p_toggle(); V_val = predict_V_count(k, d)
G_F_obs = 1.1663787e-5
delta = 2.0 / 9.0
N_hub = predict_N_hub(G_F_obs, M_Pl_GeV, a1s, delta, k, p_val, V_val)
v0 = predict_v_higgs(delta, M_Pl_GeV, N_hub, a1s)
aGUT0 = float(predict_alpha_GUT_observed(3, 10))
Munif0 = predict_M_unif_GeV(k, g, M_Pl_GeV)
b1, b2, hY = 33.0 / 5.0, 1.0, 3.0 / 5.0
M_Z_PDG = 91.1876

print("=" * 78)
print("  M_Z residual decomposition — DAG is the authority")
print("=" * 78)
print()
print(f"  Live DAG upstream values:")
print(f"    v       = {v0:.6f} GeV   [N_hub-anchored to measured G_F]")
print(f"    α_GUT   = {aGUT0:.8f} = 1/{1/aGUT0:.4f}  [dark-corrected]")
print(f"    M_unif  = {Munif0:.6e} GeV")
print(f"    b_1,b_2 = {b1}, {b2}  [1-loop MSSM, single-regime]")
print()


def running(MZ, aGUT, Munif, B1=b1, B2=b2, two_loop=False):
    L = math.log(MZ / Munif)
    inv1 = 1.0 / aGUT - (B1 / (2 * math.pi)) * L
    inv2 = 1.0 / aGUT - (B2 / (2 * math.pi)) * L
    if two_loop:
        # Standard MSSM 2-loop leading term: d(1/α_i)/dt gains
        #  -(1/(8π²)) Σ_j b_ij α_j ; use the well-known MSSM b_ij row
        #  for the gauge-gauge part (Martin SUSY primer Eq. 8.x).
        # MSSM 2-loop b_ij (i,j ∈ {1,2,3}); we use the 1↔2 block.
        bij = {(1, 1): 199 / 25, (1, 2): 27 / 5, (2, 1): 9 / 5, (2, 2): 25}
        a1g, a2g = 1.0 / inv1, 1.0 / inv2
        inv1 += (1.0 / (8 * math.pi ** 2)) * (bij[(1, 1)] * a1g + bij[(1, 2)] * a2g) * L
        inv2 += (1.0 / (8 * math.pi ** 2)) * (bij[(2, 1)] * a1g + bij[(2, 2)] * a2g) * L
    return 1.0 / inv1, 1.0 / inv2


def MZ_of(v, aGUT, Munif, B1=b1, B2=b2, two_loop=False):
    MZ = 91.18
    for _ in range(200):
        a1, a2 = running(MZ, aGUT, Munif, B1, B2, two_loop)
        new = math.sqrt(math.pi) * v * math.sqrt(a2 + hY * a1)
        if abs(new - MZ) < 1e-10:
            return new
        MZ = new
    return MZ


# ---- (A) baseline -----------------------------------------------------------
MZ0 = MZ_of(v0, aGUT0, Munif0)
res0 = (MZ0 - M_Z_PDG) / M_Z_PDG
print("=" * 78)
print("(A) Baseline (live DAG)")
print("=" * 78)
print(f"  M_Z = {MZ0:.6f} GeV   residual = {res0*100:+.4f}%   "
      f"({(MZ0-M_Z_PDG)/0.0021:+.1f}σ_PDG)")
print()

# ---- (B) what shift of each input alone makes M_Z = PDG? --------------------
print("=" * 78)
print("(B) Single-input shift required to hit M_Z = PDG (91.1876)")
print("=" * 78)

# v: M_Z ∝ v linearly → need v scaled by PDG/MZ0
v_need = v0 * (M_Z_PDG / MZ0)
print(f"  v:       {v0:.4f} → {v_need:.4f} GeV   "
      f"({(v_need-v0)/v0*100:+.4f}% shift in v)")
print(f"           [but v is N_hub-anchored to measured G_F — essentially")
print(f"            fixed by construction; a 0.36% v shift would break the")
print(f"            G_F round-trip.  So v is unlikely the free culprit.]")

# α_GUT: bisect
lo, hi = aGUT0 * 0.9, aGUT0 * 1.1
for _ in range(80):
    mid = 0.5 * (lo + hi)
    if MZ_of(v0, mid, Munif0) > M_Z_PDG:
        hi = mid
    else:
        lo = mid
aG_need = 0.5 * (lo + hi)
print(f"  α_GUT:   1/{1/aGUT0:.4f} → 1/{1/aG_need:.4f}   "
      f"({(aG_need-aGUT0)/aGUT0*100:+.4f}% shift in α_GUT)")

# M_unif: bisect (M_Z increases as M_unif decreases — more running)
lo, hi = Munif0 * 0.2, Munif0 * 5.0
for _ in range(100):
    mid = math.sqrt(lo * hi)
    if MZ_of(v0, aGUT0, mid) > M_Z_PDG:
        lo = mid
    else:
        hi = mid
Mu_need = math.sqrt(lo * hi)
print(f"  M_unif:  {Munif0:.4e} → {Mu_need:.4e} GeV   "
      f"({(Mu_need-Munif0)/Munif0*100:+.2f}% shift in M_unif)")
print(f"           [Row P62 quotes M_unif at −0.76% vs the (not-measured)")
print(f"            MSSM benchmark; the shift needed here is the real test.]")
print()

# ---- (C) log-sensitivities --------------------------------------------------
print("=" * 78)
print("(C) Log-sensitivity  d ln M_Z / d ln(input)")
print("=" * 78)
eps = 1e-4
s_v = (math.log(MZ_of(v0*(1+eps), aGUT0, Munif0)) - math.log(MZ0)) / eps
s_aG = (math.log(MZ_of(v0, aGUT0*(1+eps), Munif0)) - math.log(MZ0)) / eps
s_Mu = (math.log(MZ_of(v0, aGUT0, Munif0*(1+eps))) - math.log(MZ0)) / eps
print(f"  ∂lnM_Z/∂ln v      = {s_v:+.4f}   (≈ +1, linear in v)")
print(f"  ∂lnM_Z/∂ln α_GUT  = {s_aG:+.4f}")
print(f"  ∂lnM_Z/∂ln M_unif = {s_Mu:+.4f}")
print(f"  → M_unif needs {res0*100:+.3f}% / {s_Mu:+.4f} = "
      f"{res0/s_Mu*100:+.2f}% shift to absorb the whole residual")
print()

# ---- (D) explicit 2-loop-β test --------------------------------------------
print("=" * 78)
print("(D) Does adding standard MSSM 2-loop β running close the residual?")
print("=" * 78)
MZ_2L = MZ_of(v0, aGUT0, Munif0, two_loop=True)
res_2L = (MZ_2L - M_Z_PDG) / M_Z_PDG
print(f"  1-loop (live):  M_Z = {MZ0:.5f}  ({res0*100:+.4f}%)")
print(f"  +2-loop term:   M_Z = {MZ_2L:.5f}  ({res_2L*100:+.4f}%)")
print(f"  2-loop moves M_Z by {(MZ_2L-MZ0)/MZ0*100:+.4f}%  "
      f"(residual to close: {-res0*100:+.4f}%)")
frac = (MZ0 - MZ_2L) / (MZ0 - M_Z_PDG) if (MZ0 - M_Z_PDG) != 0 else float('nan')
print(f"  → 2-loop accounts for {frac*100:.0f}% of the 1-loop residual")
print()

# ---- (E) verdict ------------------------------------------------------------
print("=" * 78)
print("(E) Verdict — load-bearing contributor (from numbers, not prose)")
print("=" * 78)
print(f"  Residual to explain: {res0*100:+.4f}% (M_Z too high).")
print(f"  v-route:      needs {(v_need-v0)/v0*100:+.3f}% — but v is G_F-anchored")
print(f"                (breaking it breaks G_F); NOT a free culprit.")
print(f"  α_GUT-route:  needs {(aG_need-aGUT0)/aGUT0*100:+.3f}% on a theorem-grade")
print(f"                dark-corrected quantity.")
print(f"  M_unif-route: needs {(Mu_need-Munif0)/Munif0*100:+.2f}% (vs Row P62's")
print(f"                quoted −0.76% vs benchmark).")
print(f"  2-loop-β:     accounts for {frac*100:.0f}% of the residual by itself.")
print()
print(f"  The decomposition (not the ledger) names the driver.  Read the")
print(f"  four lines above: whichever needs the smallest / already-known")
print(f"  shift, or which the 2-loop term already supplies, is the fix")
print(f"  target.  No culprit is asserted here beyond what the numbers show.")
print()
print("=" * 78)
print("End of M_Z residual decomposition diagnostic.")
print("=" * 78)
