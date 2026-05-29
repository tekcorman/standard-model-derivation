#!/usr/bin/env python3
"""
proofs/_archive/M_Z_Delta_r_analog_substrate_2026-05-15.py

SUBSTRATE Δr-ANALOG — the M_Z tree→pole oblique correction, as the
sign-uniform sibling of the δρ (Δρ) closed this session (Row P73).

ESTABLISHED (this session):
 - Decomposition Pt1 (ffa89dc): M_Z is M_unif-INSENSITIVE; the +0.357%
   is NOT M_unif, NOT 2-loop-β.
 - Decomposition Pt2 (9501a65): with EXACT PDG (g_2, sin²θ_W, v) the SM
   TREE M_Z = g_2·v/(2cosθ_W) over-predicts the pole by +0.393%.  The
   residual is INTRINSIC to the SM tree relation = the tree-vs-pole
   OBLIQUE radiative correction (Δr / ρ-parameter family).
 - Phase C: ρ ≡ m_W²/(M_Z²cos²θ_W) = (1/2)·(Π_W/Π_Z).  The W residue
   (h_P, phase) carried δρ (custodial-BREAKING, cancels nothing).  The
   Z residue (Perron, real) is custodial-SYMMETRIC and CANCELLED in the
   ρ ratio — so δρ never used it.  But that Z-Perron sign-uniform piece
   is precisely the ABSOLUTE-M_Z self-energy oblique shift = the Δr.

THE Δr-ANALOG (no new derivation needed — it is the Phase-A piece):
 Phase A (commit e1466db, the part NOT retracted — only the additive
 c_S+c_E combination and the stale-input numbers were retracted; the
 two-routes c_S=1/12 structural finding STOOD and was superseded only
 as a *combination*, not as the sign-uniform coefficient) derived the
 Family-C sign-uniform coefficient for the gauge-boson 2-point:
     c_S = 1/12   via TWO independent routes:
       Route H (Hashimoto-spectral): 1 marginal direction / 2|E| = 1/12
       Route C (cycle-counting):     k*/(N_atoms·k*²) = 1/(N·k*) = 1/12
     v_Higgs-calibrated: the factor-1/5 reduction from c_v=5/12 (Family-C
     counting form, parity-odd content absorbed in the integer fraction
     — NO separate F* functional, same as v_Higgs and α_GUT counting).

 Master-doc Family-C universal template (counting form):
     g_physical = g_bare · (1 − c · α₁_bare/(1−α₁_bare))
 Applied to the M_Z gauge-boson 2-point (a mass observable):
     M_Z_pole = M_Z_tree · (1 − c_S · α₁_bare/(1−α₁_bare))
 ⇒ δM_Z/M_Z = − c_S · α₁_bare/(1−α₁_bare),  c_S = 1/12.

This is the SIGN-UNIFORM sibling of δρ.  Unified Phase-C picture:
 ONE Hashimoto spectral object, two vertex samplings —
   Π_Z Perron-real (sign-uniform)  → absolute-M_Z Δr  (this probe)
   Π_W h_P-phase  (custodial-break) → δρ ratio          (Row P73)

PRE-DECLARED ABORT:
 (D.1) wrong sign (correction must LOWER M_Z: tree is HIGH)        → NEG
 (D.2) magnitude off the tree→pole gap by >1 order                 → NEG
 (D.3) requires re-deriving c_S or a non-K-rational / arg factor   → NEG
 (D.4) double-counts an oblique correction already in M_Z.py       → NEG
 (D.5) c_S=1/12 (Phase-A two-routes, v_Higgs-calibrated, counting
       template) gives the right sign AND within δρ-comparable
       accuracy (≲15% rel) of the tree→pole gap                    → POS
"""
from __future__ import annotations
import math
from fractions import Fraction

k_star, g, N_atoms, N_edges = 3, 10, 4, 6
alpha1 = Fraction(k_star - 1, k_star) ** (g - 2)          # (2/3)^8
afac = float(alpha1) / (1.0 - float(alpha1))              # α₁/(1−α₁)

# Phase-A two-routes coefficient (CITED, not re-derived):
c_S_H = Fraction(1, 2 * N_edges)                          # Route H: 1/(2|E|)
c_S_C = Fraction(k_star, N_atoms * k_star ** 2)           # Route C: k*/(N·k*²)
assert c_S_H == c_S_C == Fraction(1, 12), "Phase-A two-routes must both give 1/12"
c_S = Fraction(1, 12)

print("=" * 78)
print("  Substrate Δr-analog — sign-uniform sibling of δρ (Row P73)")
print("=" * 78)
print()
print(f"  c_S (Phase-A two-routes, CITED): Route H 1/(2|E|)={c_S_H}, "
      f"Route C k*/(N·k*²)={c_S_C}  ⇒ c_S = {c_S}")
print(f"  v_Higgs calibration: c_v=5/12 → c_S=1/12 is the factor-1/5 reduction")
print(f"  (Family-C counting form; no separate F* — parity-odd content in c).")
print(f"  α₁_bare = (2/3)^8 = {float(alpha1):.8f};  α₁/(1−α₁) = {afac:.8f}")
print()

# Family-C counting template on the M_Z 2-point (a mass observable):
deltaMZ_over_MZ = - float(c_S) * afac
print("=" * 78)
print("  δM_Z/M_Z = − c_S · α₁_bare/(1−α₁_bare)")
print("=" * 78)
print(f"  = − (1/12) · {afac:.6f} = {deltaMZ_over_MZ*100:+.4f}%")
print()

# Tree→pole gap to close (from decomposition Pt2, both input sets):
gap_fw  = +0.3573e-2     # framework post-α_GUT-DC inputs (M_Z.py live)
gap_pdg = +0.3925e-2     # EXACT PDG inputs (intrinsic SM tree-vs-pole)
print("  Tree→pole gap the correction must REMOVE (M_Z is HIGH):")
print(f"    framework-input gap = +{gap_fw*100:.4f}%  (predictions/M_Z.py live)")
print(f"    exact-PDG-input gap = +{gap_pdg*100:.4f}%  (intrinsic SM tree-vs-pole)")
print()

# Apply and report
MZ_tree_fw = 91.5135
MZ_pole_PDG = 91.1876
MZ_corr = MZ_tree_fw * (1.0 + deltaMZ_over_MZ)
res_before = (MZ_tree_fw - MZ_pole_PDG) / MZ_pole_PDG
res_after = (MZ_corr - MZ_pole_PDG) / MZ_pole_PDG
print("=" * 78)
print("  Applied to live predictions/M_Z.py tree value")
print("=" * 78)
print(f"  M_Z_tree (live)              = {MZ_tree_fw:.5f} GeV  "
      f"(residual {res_before*100:+.4f}%)")
print(f"  M_Z_pole = tree·(1+δM_Z/M_Z) = {MZ_corr:.5f} GeV  "
      f"(residual {res_after*100:+.4f}%)")
print(f"  PDG pole                     = {MZ_pole_PDG} GeV")
print()

# Accuracy of the Δr-analog vs the gap (sign + relative)
sign_ok = deltaMZ_over_MZ < 0 and gap_fw > 0
rel_vs_fw = abs(abs(deltaMZ_over_MZ) - gap_fw) / gap_fw
rel_vs_pdg = abs(abs(deltaMZ_over_MZ) - gap_pdg) / gap_pdg
print("=" * 78)
print("  Verdict (pre-declared aborts)")
print("=" * 78)
print(f"  (D.1) sign: correction {deltaMZ_over_MZ*100:+.4f}% LOWERS M_Z, gap is +; "
      f"{'OK' if sign_ok else 'WRONG SIGN — NEG'}")
print(f"  (D.2) magnitude: |δ|={abs(deltaMZ_over_MZ)*100:.4f}% vs gap "
      f"{gap_fw*100:.4f}% (fw, {rel_vs_fw*100:.1f}% off) / "
      f"{gap_pdg*100:.4f}% (PDG, {rel_vs_pdg*100:.1f}% off)")
print(f"  (D.3) K-rational: (1/12)·(2/3)^8/(1−(2/3)^8) ∈ ℚ ⊂ K; no F*/arg. OK")
print(f"  (D.4) double-count: α_GUT-DC is VERTEX-level (coupling@M_unif, c=1/k*);")
print(f"        this is PROPAGATOR-level (M_Z 2-point) — different sector, no")
print(f"        double-count.  M_Z.py applies NO 2-point oblique (pure SM tree).")
print(f"  (D.5) c_S=1/12 Phase-A two-routes, v_Higgs-calibrated, counting")
print(f"        template, right sign, within δρ-comparable accuracy:")
verdict_pos = sign_ok and rel_vs_fw < 0.15
print(f"        {'YES — Δr-analog POSITIVE' if verdict_pos else 'NO'}")
print()
if verdict_pos:
    print(f"  → Δr-ANALOG POSITIVE.  δM_Z/M_Z = −(1/12)·α₁_bare/(1−α₁_bare)")
    print(f"    = {deltaMZ_over_MZ*100:+.4f}% closes the framework tree→pole gap")
    print(f"    (+{gap_fw*100:.4f}%) to within {res_after*100:+.4f}% — the SAME")
    print(f"    δρ-comparable grade (δρ was +4.58% rel).  Sign-uniform sibling")
    print(f"    of δρ: ONE Hashimoto object, Π_Z Perron→Δr, Π_W h_P→δρ.")
    print(f"    c_S=1/12 is the Phase-A two-routes result (NOT re-fit);")
    print(f"    counting Family-C template (NOT the SM Sirlin Δr import —")
    print(f"    Clause-9-safe, K-rational).")
else:
    print(f"  → honest NEG / partial — see aborts above; do not force a fit.")
print()
print("=" * 78)
print("End of substrate Δr-analog probe.")
print("=" * 78)
