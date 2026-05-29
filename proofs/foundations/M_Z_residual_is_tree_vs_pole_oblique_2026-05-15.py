#!/usr/bin/env python3
"""
proofs/foundations/M_Z_residual_is_tree_vs_pole_oblique_2026-05-15.py

DECOMPOSITION Part 2 (follows M_Z_residual_decomposition_diagnostic
2026-05-15.py, commit ffa89dc).  Identifies WHAT the M_Z +0.357%
residual actually is — using the predictions/ DAG + exact PDG inputs as
the authority, not ledger/theorem prose.

ESTABLISHED in Part 1 (ffa89dc):
  - M_Z is M_unif-INSENSITIVE (∂lnM_Z/∂lnM_unif ≈ −0.004) → the ledger's
    "upstream M_unif Stage-5" attribution is FALSE (corrected, 1c9722f).
  - 2-loop MSSM β makes M_Z WORSE (+0.357% → +0.868%) → the
    theorem_gauge_unification_RG_closure.md §4 "needs two-loop running"
    attribution is ALSO false.
  - Residual lives in the √(α_2+(3/5)α_1) electroweak-coupling factor.

THIS PROBE — the decisive test.  The post-α_GUT-DC cluster (the DONE
work in gauge_unification_full_RG_closure.py) gives g_2(M_Z), sin²θ_W,
matching PDG to <0.1%.  v=246.22 is the PDG Higgs VEV (G_F-anchored).
predictions/M_Z.py computes the SM TREE relation
    M_Z = √π·v·√(α_2+(3/5)α_1)  ≡  √(g_2²+g_Y²)·v/2  ≡  g_2·v/(2cosθ_W).
Question: with PDG-CONSISTENT (g_2, sin²θ_W, v), does the SM tree
relation ITSELF reproduce the PDG POLE M_Z, or is it intrinsically
high?

If the SM tree relation with EXACT PDG inputs already over-predicts
M_Z by ~the same +0.36%, then the residual is the SM TREE-vs-POLE
OBLIQUE radiative correction (the Δr / ρ-parameter family) — NOT any
framework input error.  predictions/M_Z.py uses ρ=1 tree and is high
BY CONSTRUCTION, exactly as the SM tree relation is.

CLAUSE-9 DISCIPLINE (critical).  Identifying the residual AS the
tree-vs-pole oblique correction is a STRUCTURAL CLASSIFICATION
(verified numerically here).  It is NOT a closure.  Closing it by
citing the SM Sirlin Δr number is the bridge-attribution-as-closure
anti-pattern explicitly RETRACTED (commit 4ce4d5c, Clause 9
violation: Δr is continuum 2-loop QFT, not K-rational).  The
legitimate closure path is the SAME as δρ (Row P73): derive the
SUBSTRATE spectral analog (Phase-C Hashimoto h_P Feshbach residue),
K-rational, O9-respecting — NOT import the SM loop number.  This probe
only DIAGNOSES; it does not close.
"""
from __future__ import annotations
import math

# ── post-α_GUT-DC cluster (DONE work; live from gauge_unification_full_RG_closure.py)
g2_fw  = 0.65175      # PDG 0.652    (−0.038%)
s2w_fw = 0.23126      # PDG 0.23121  (+0.020%)
v      = 246.22       # = PDG Higgs VEV (G_F-anchored)
MZ_PDG = 91.1876
g2_PDG, s2w_PDG = 0.652, 0.23121


def MZ_tree(g2, s2w, vev):
    """SM tree:  M_Z = √(g1²+g2²)·v/2 = g_2·v/(2 cosθ_W)."""
    return g2 * vev / (2.0 * math.sqrt(1.0 - s2w))


print("=" * 78)
print("  M_Z residual = SM tree-vs-pole OBLIQUE correction (decomposition Pt 2)")
print("=" * 78)
print()
print("  predictions/M_Z.py computes the SM TREE relation (ρ=1, no oblique):")
print("    M_Z = √π·v·√(α_2+(3/5)α_1) ≡ g_2·v/(2cosθ_W)")
print()

mz_fw  = MZ_tree(g2_fw,  s2w_fw,  v)
mz_pdg = MZ_tree(g2_PDG, s2w_PDG, v)

print(f"  (1) framework post-DC inputs (each matches PDG to <0.1%):")
print(f"      g_2={g2_fw} sin²θ_W={s2w_fw} v={v}")
print(f"      → SM tree M_Z = {mz_fw:.5f}  vs PDG pole {MZ_PDG}"
      f"  =  {(mz_fw-MZ_PDG)/MZ_PDG*100:+.4f}%")
print()
print(f"  (2) EXACT PDG inputs (ZERO framework error):")
print(f"      g_2={g2_PDG} sin²θ_W={s2w_PDG} v={v}")
print(f"      → SM tree M_Z = {mz_pdg:.5f}  vs PDG pole {MZ_PDG}"
      f"  =  {(mz_pdg-MZ_PDG)/MZ_PDG*100:+.4f}%")
print()
print(f"  ⇒ the +0.36–0.39% PERSISTS with exact PDG inputs.  It is INTRINSIC")
print(f"    to the SM tree M_Z relation — the tree-vs-pole OBLIQUE radiative")
print(f"    correction (Δr / ρ-parameter family).  NOT a framework input")
print(f"    error: not M_unif (Pt 1: insensitive), not α_GUT-magnitude, not")
print(f"    1-loop-vs-2-loop (Pt 1: 2-loop is worse), not v (PDG VEV).")
print()

# ── connection to the δρ derived THIS session (Row P73)
drho = 0.5 * (math.sqrt(5) / 4.0) * (2.0 / 3.0) ** 8
print("=" * 78)
print("  Connection: this is the SAME oblique family as δρ (Row P73)")
print("=" * 78)
print(f"  δρ  (Row P73, derived this session) = {drho*100:+.4f}%   "
      f"[W/Z custodial-breaking, Δρ]")
print(f"  M_Z tree→pole gap                   ≈ +0.36%        "
      f"[M_Z absolute, Δr-family]")
print(f"  Both are SM OBLIQUE corrections (Δρ / Δr / S,T).  δρ is the FIRST")
print(f"  member closed at substrate level (Phase C: Hashimoto h_P Feshbach")
print(f"  residue).  The M_Z absolute residual is its SIBLING — the natural")
print(f"  next target of the SAME substrate-spectral mechanism.")
print()
print("=" * 78)
print("  Verdict (DIAGNOSIS only — NOT a closure)")
print("=" * 78)
print(f"  The M_Z +0.357% is the SM tree-vs-pole oblique radiative correction")
print(f"  (Δr family), sibling of the δρ (Δρ) closed this session.  The")
print(f"  ledger's 'M_unif Stage-5' and theorem-doc's 'needs 2-loop' prose")
print(f"  attributions are BOTH numerically false (Pt 1 + this probe).")
print()
print(f"  Closure path (NOT done here): derive the substrate spectral analog")
print(f"  of the M_Z tree→pole oblique correction via the SAME Phase-C")
print(f"  Hashimoto-h_P Feshbach mechanism that gave δρ — K-rational,")
print(f"  O9-respecting.  Citing the SM Sirlin Δr number is the RETRACTED")
print(f"  Clause-9 bridge-attribution anti-pattern (commit 4ce4d5c) and is")
print(f"  NOT an acceptable closure.")
print()
print("=" * 78)
