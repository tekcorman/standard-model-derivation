#!/usr/bin/env python3
"""
proofs/_archive/family_E_phase_AB_CORRECTED_base_predictions_2026-05-15.py

CORRECTION probe: Phase A/B used STALE base predictions.

ERROR IDENTIFIED (via parameter_linter Checkpoint 1 input verification):
  Phase A (commit e1466db) and Phase B (commit 05b2cbb) used:
    M_Z_pred = 91.97 GeV, m_W_pred = 80.69 GeV
  These are the PRE-α_GUT-DC values from the M_Z.py docstring header.

  Live predictions/M_Z.py + predictions/m_W.py output:
    M_Z = 91.5135 GeV  (residual +0.3574%, +155σ_PDG)
    m_W = 80.2373 GeV  (residual -0.1642%, -9.92σ_PDG)

  Critical: with CORRECT predictions the residuals have OPPOSITE SIGNS
  (M_Z too high, m_W too low) — the custodial-breaking signature per
  ledger Row P64/P71.  The stale values were both-positive, which is
  why the Phase A/B decomposition gave a spurious "0.007% δρ match".

This probe RECOMPUTES the Family C + Family E decomposition with the
CORRECT live base predictions, and honestly reports whether the
c_S = 1/12, c_E = 1/18 structural forms still close the cluster.

Per an internal note + [[audit-for-smuggled-
parameters]]: the prior "essentially exact" claim is RETRACTED pending
this recomputation.
"""
from __future__ import annotations
from fractions import Fraction
import numpy as np

# ---------------------------------------------------------------------------
# Framework constants
# ---------------------------------------------------------------------------
k_star = 3
g = 10
N_ATOMS = 4
N_EDGES = 6
alpha_1_bare = Fraction(k_star - 1, k_star) ** (g - 2)
alpha_factor = float(alpha_1_bare) / (1 - float(alpha_1_bare))

# CORRECT live predictions (verified by running predictions/M_Z.py, m_W.py)
M_Z_pred = 91.5135  # GeV  (live output, post α_GUT-DC)
m_W_pred = 80.2373  # GeV  (live output)

# PDG 2024
M_Z_PDG = 91.1876
m_W_PDG = 80.3692
sin2_W_MS = 0.23122
cos2_W_MS = 1 - sin2_W_MS

# Phase A/B structural forms
c_S = Fraction(1, 12)
c_E = Fraction(1, 18)

print("=" * 78)
print("  CORRECTED Phase A/B — live base predictions")
print("=" * 78)
print()
print(f"  STALE (Phase A/B used):  M_Z = 91.97,   m_W = 80.69")
print(f"  LIVE  (correct):         M_Z = {M_Z_pred}, m_W = {m_W_pred}")
print()

# Residuals (predicted - observed)/observed
dM_Z = (M_Z_pred - M_Z_PDG) / M_Z_PDG
dm_W = (m_W_pred - m_W_PDG) / m_W_PDG
print(f"  Live residuals:")
print(f"    M_Z: {dM_Z*100:+.4f}%  (predicted {'too high' if dM_Z>0 else 'too low'})")
print(f"    m_W: {dm_W*100:+.4f}%  (predicted {'too high' if dm_W>0 else 'too low'})")
print(f"    → OPPOSITE SIGNS: {'YES (custodial-breaking signature)' if dM_Z*dm_W < 0 else 'NO'}")
print()

# δρ
rho_obs = (m_W_PDG**2) / (M_Z_PDG**2 * cos2_W_MS)
print(f"  Observed δρ (vs SM tree ρ=1): {(rho_obs-1)*100:+.4f}%")
print(f"  Framework predicted ρ (m_W = M_Z·cosθ by construction): 1.0000 (δρ=0)")
print()

# ---------------------------------------------------------------------------
# Decomposition with CORRECT base predictions
# ---------------------------------------------------------------------------
print("=" * 78)
print("Decomposition with correct live base predictions")
print("=" * 78)
print()

# multiplicative shifts NEEDED (observed/predicted - 1)
shift_M_Z = M_Z_PDG / M_Z_pred - 1
shift_m_W = m_W_PDG / m_W_pred - 1
print(f"  Shifts needed (observed/predicted - 1):")
print(f"    δ_Z = {shift_M_Z*100:+.4f}%")
print(f"    δ_W = {shift_m_W*100:+.4f}%")
print()

S = (shift_M_Z + shift_m_W) / 2  # sign-uniform
R = (shift_m_W - shift_M_Z) / 2  # asymmetric (m_W rel to M_Z)
print(f"  S = (δ_Z + δ_W)/2 = {S*100:+.4f}%  (sign-uniform, Family C)")
print(f"  R = (δ_W - δ_Z)/2 = {R*100:+.4f}%  (asymmetric, Family E)")
print()
print(f"  → S is SMALL ({abs(S)*100:.3f}%), R is DOMINANT ({abs(R)*100:.3f}%)")
print(f"    (residuals are mostly opposite-sign = pure custodial breaking)")
print()

target_c_S = abs(S) / alpha_factor
target_c_E = abs(R) / alpha_factor
print(f"  Required c values:")
print(f"    c_S target = {target_c_S:.6f}")
print(f"    c_E target = {target_c_E:.6f}")
print()
print(f"  Phase A/B structural forms:")
print(f"    c_S = 1/12 = {float(c_S):.6f}  ({abs(float(c_S)-target_c_S)/target_c_S*100:.1f}% off live target)")
print(f"    c_E = 1/18 = {float(c_E):.6f}  ({abs(float(c_E)-target_c_E)/target_c_E*100:.1f}% off live target)")
print()

# K-rational neighborhoods
print(f"  K-rational neighborhood of c_S target {target_c_S:.4f}:")
for n, d in [(1, 36), (1, 42), (1, 48), (1, 54), (1, 32), (5, 216)]:
    print(f"    {n}/{d} = {n/d:.6f}  ({abs(n/d-target_c_S)/target_c_S*100:.1f}% off)")
print(f"  K-rational neighborhood of c_E target {target_c_E:.4f}:")
for n, d in [(1, 16), (1, 15), (1, 18), (5, 72), (1, 12), (2, 27)]:
    print(f"    {n}/{d} = {n/d:.6f}  ({abs(n/d-target_c_E)/target_c_E*100:.1f}% off)")
print()

# ---------------------------------------------------------------------------
# Apply Phase A/B forms (c_S=1/12, c_E=1/18) to CORRECT base
# ---------------------------------------------------------------------------
print("=" * 78)
print("Apply Phase A/B forms (c_S=1/12, c_E=1/18) to LIVE base predictions")
print("=" * 78)
print()

sC = float(c_S) * alpha_factor
sE = float(c_E) * alpha_factor
M_Z_corr = M_Z_pred * (1 - sC - sE)
m_W_corr = m_W_pred * (1 - sC + sE)
rM = (M_Z_corr - M_Z_PDG) / M_Z_PDG
rW = (m_W_corr - m_W_PDG) / m_W_PDG
rho_corr = (m_W_corr**2) / (M_Z_corr**2 * cos2_W_MS)

print(f"  Family C shift: c_S × α₁/(1-α₁) = -{sC*100:.4f}% (both)")
print(f"  Family E shift: c_E × α₁/(1-α₁) = ±{sE*100:.4f}% (M_Z down, m_W up)")
print()
print(f"  M_Z: {M_Z_pred:.4f} → {M_Z_corr:.4f}  residual {(M_Z_pred-M_Z_PDG)/M_Z_PDG*100:+.4f}% → {rM*100:+.4f}%")
print(f"  m_W: {m_W_pred:.4f} → {m_W_corr:.4f}  residual {(m_W_pred-m_W_PDG)/m_W_PDG*100:+.4f}% → {rW*100:+.4f}%")
print()
print(f"  δρ: predicted {(rho_corr-1)*100:+.4f}%  vs observed {(rho_obs-1)*100:+.4f}%")
print(f"  δρ gap: {abs((rho_corr-1)-(rho_obs-1))*100:.4f}% absolute")
print()

# Try sign flip on Family E (M_Z up, m_W down)
M_Z_corr2 = M_Z_pred * (1 - sC + sE)
m_W_corr2 = m_W_pred * (1 - sC - sE)
rM2 = (M_Z_corr2 - M_Z_PDG) / M_Z_PDG
rW2 = (m_W_corr2 - m_W_PDG) / m_W_PDG
rho_corr2 = (m_W_corr2**2) / (M_Z_corr2**2 * cos2_W_MS)
print(f"  [Sign-flip Family E: M_Z up, m_W down]")
print(f"  M_Z: → {M_Z_corr2:.4f}  residual {rM2*100:+.4f}%")
print(f"  m_W: → {m_W_corr2:.4f}  residual {rW2*100:+.4f}%")
print(f"  δρ: predicted {(rho_corr2-1)*100:+.4f}%  vs observed {(rho_obs-1)*100:+.4f}%")
print()

# ---------------------------------------------------------------------------
# Honest verdict
# ---------------------------------------------------------------------------
print("=" * 78)
print("HONEST VERDICT (post-correction)")
print("=" * 78)
print()
print(f"  Phase A/B used STALE base (M_Z=91.97, m_W=80.69) → spurious 0.007% δρ match.")
print()
print(f"  With CORRECT live base (M_Z={M_Z_pred}, m_W={m_W_pred}):")
print(f"    - Residuals are OPPOSITE-SIGN (custodial-breaking dominant)")
print(f"    - c_S target ≈ {target_c_S:.4f} (small, NOT 1/12=0.083)")
print(f"    - c_E target ≈ {target_c_E:.4f} (NOT 1/18=0.056)")
print(f"    - Phase A/B forms give δρ_pred = {(rho_corr-1)*100:+.3f}% vs obs {(rho_obs-1)*100:+.3f}%")
print()
print(f"  CONCLUSION: The Phase A/B 'closure to 0.007%' is RETRACTED.")
print(f"  It was an artifact of stale base predictions.  With correct live")
print(f"  predictions, c_S=1/12 and c_E=1/18 do NOT close the cluster at")
print(f"  sub-percent.")
print()
print(f"  WHAT SURVIVES:")
print(f"    - Structural insight: residuals ARE opposite-sign (custodial-")
print(f"      breaking), confirming Family E (asymmetric) is the right")
print(f"      mechanism CLASS (consistent with master doc + Phase A.2/PIVOT")
print(f"      findings).")
print(f"    - c_E target ≈ {target_c_E:.4f}: closest clean K-rational is")
print(f"      1/16 = 0.0625 (Route H: 'W^± pair / 2|E|² ' style) — needs")
print(f"      independent structural derivation, NOT yet done.")
print(f"    - The Family C piece is SMALL (c_S ≈ {target_c_S:.3f}); the")
print(f"      cluster residual is DOMINANTLY custodial-breaking, not")
print(f"      sign-uniform.")
print()
print(f"  CLUSTER STATUS UNCHANGED: M_Z (P64), m_W (P71) remain")
print(f"  STRUCTURAL-DERIVATION-CONDITIONAL.  No graduation.  Documentation")
print(f"  must NOT be updated with the retracted Phase A/B closure.")
print()
print("=" * 78)
print("End of correction probe.")
print("=" * 78)
