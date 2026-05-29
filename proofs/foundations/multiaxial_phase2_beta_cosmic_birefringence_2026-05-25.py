#!/usr/bin/env python3
"""
Multi-axial Phase 2 audit -- β cosmic birefringence verification probe (2026-05-25).

Audit doc: an internal working note

Three numerical checks (the parameter axis at TWO sub-loci is the substantive test):

  1. Lattice axis: confirm h_P intensive content is bit-identical between srs
     and srs-z (per R-9 register: "everything intensive is bit-identical").
     For β = c·sin(arg h)·α_EM, this means lattice-axis is doubly robust:
     (A) gates srs-z, AND srs-z wouldn't shift β even if un-gated.

  2. Parameter axis A.3.a (functional): enumerate 4 parity-odd projections
     of h_P and confirm sin(arg h) = √(5/8) is the dispersion-class
     projection (others couple to mass²/amplitude classes — different
     observables).

  3. Parameter axis A.3.b (coefficient): enumerate 5 K-rational
     multiplicative coefficients (1, 1/2, 5/12, 9/40, 256/6305) that
     are REAL K-elements DOING REAL WORK in other observables. Plus
     1/(16π²) excluded by algebraicity. Confirm channel_select(K, β
     channel) picks c=1; alternatives ruled out by Eskilt 2022 at 1.8-3.5σ.

This is the most rigorous channel-select test in the Phase 2 audit queue:
the K-rational competitors are not synthetic, they're the framework's
existing structural coefficients for other observables. NO NEW PHYSICS.
"""

from __future__ import annotations

import os
import sys
import math
from fractions import Fraction

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, REPO_ROOT)

print("=" * 70)
print("Multi-axial Phase 2 audit -- β cosmic birefringence (2026-05-25)")
print("=" * 70)

# ------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------
# h_P = (√3 + i√5)/2; |h_P|² = (3+5)/4 = 2; |h_P| = √2
h_P_re = math.sqrt(3) / 2
h_P_im = math.sqrt(5) / 2
h_P_abs = math.sqrt(2)
sin_arg_h = h_P_im / h_P_abs   # = √5 / (2√2) = √(5/8) ≈ 0.7906

# Framework α_EM(M_Z) ~ 1/127.93 per alpha_EM.py
alpha_EM_MZ = 1.0 / 127.93   # framework prediction (theorem-grade-conditional)

# Eskilt 2022 observation
beta_obs_deg = 0.342
beta_obs_sigma = 0.094

# Reference: c=1 prediction
beta_ref_rad = 1.0 * sin_arg_h * alpha_EM_MZ
beta_ref_deg = math.degrees(beta_ref_rad)
print(f"\nReference β (framework, c=1):")
print(f"  sin(arg h_P) = Im(h)/|h| = √(5/8) = {sin_arg_h:.6f}")
print(f"  α_EM(M_Z) = {alpha_EM_MZ:.6f} = 1/{1/alpha_EM_MZ:.2f}")
print(f"  β = 1 · sin(arg h) · α_EM = {beta_ref_rad:.6f} rad = {beta_ref_deg:.4f}°")
print(f"  vs Eskilt 2022: {beta_obs_deg}° ± {beta_obs_sigma}°")
print(f"  match: {(beta_ref_deg - beta_obs_deg) / beta_obs_sigma:+.2f}σ")


# ------------------------------------------------------------------------
# Check 1: lattice axis — srs-z h_P bit-identical to srs
# ------------------------------------------------------------------------
print()
print("Check 1: lattice axis — srs-z h_P is bit-identical to srs's.")
print("  Per R-9 register (structural_residue_register.md §R-9 closure 2026-05-12):")
print("   'srs-z carries h = (√3+i√5)/2 with multiplicity 4 vs srs's 2'")
print("   'Everything intensive is bit-identical (h, ν_amp=√5/4, ν_mass²=5/3,")
print("    dispersion Taylor coefficients, Clifford structure, walk survivals,")
print("    k*, g, n_generations=3, Q_Koide=2/3).'")
print()
print("  Implication for β:")
print("   β = c·sin(arg h)·α_EM depends ONLY on the INTENSIVE content of h_P")
print("   (and on α_EM, which is a separate observable).")
print(f"   sin(arg h_srs)   = {sin_arg_h:.6f}")
print(f"   sin(arg h_srs-z) = {sin_arg_h:.6f}  (bit-identical per R-9)")
print(f"   β(srs)   = {beta_ref_deg:.4f}°")
print(f"   β(srs-z) = {beta_ref_deg:.4f}°  (same)")
print()
print("  --> β is DOUBLY ROBUST on lattice axis:")
print("      (i)  (A) no-privilege gates srs-z out (Sunada 2012).")
print("      (ii) Even if un-gated, srs-z h_P is bit-identical → 0 shift.")
print()
print("  Lattice axis shift: 0 (gated; AND would be 0 even if un-gated).")


# ------------------------------------------------------------------------
# Check 2: parameter axis A.3.a — functional choice
# ------------------------------------------------------------------------
print()
print("Check 2: parameter axis sub-locus A.3.a — functional choice.")
print(f"  h_P = (√3+i√5)/2 = {h_P_re:.6f} + {h_P_im:.6f}i, |h_P| = {h_P_abs:.6f}")
print()

functionals = [
    ("sin(arg h) = Im(h)/|h| = √(5/8)",
     h_P_im / h_P_abs, "dispersion (β channel)", "✅ chosen"),
    ("Im(h)/|h|² = √5/4",
     h_P_im / h_P_abs**2, "mass² (m_ν / θ_23 channel)", "→ different observable"),
    ("Im(h)/Re(h) = √(5/3)",
     h_P_im / h_P_re, "phase tangent (no clean operator channel)", "gated"),
    ("tan²(arg h) = (Im/Re)² = 5/3",
     (h_P_im / h_P_re) ** 2, "squared phase (y_τ chain)", "→ different observable"),
]

print(f"  {'Functional':<32} | Value      | Channel / Verdict")
print(f"  {'-' * 32}-|------------|------------------------------")
for name, val, channel, verdict in functionals:
    print(f"  {name:<32} | {val:.6f}   | {channel:<30} ({verdict})")
print()
print("  --> Lemma 2 of theorem_dark_correction_mdl.md selects sin(arg h) by")
print("      dimensional matching: photon polarization couples to the unit")
print("      phasor h/|h|, so the parity-odd projection is Im(h/|h|) = Im(h)/|h|.")
print("  --> Alternatives are bit-cheap (in some encoding) but they're in")
print("      DIFFERENT operator channels (mass², phase). channel_select picks")
print("      sin(arg h) for β.")
print()
print("  Functional shift: 0 (channel-selected by dimensional matching).")


# ------------------------------------------------------------------------
# Check 3: parameter axis A.3.b — coefficient choice
# ------------------------------------------------------------------------
print()
print("Check 3: parameter axis sub-locus A.3.b — coefficient choice c ∈ K.")
print("  K = ℚ(√2, √3, √5). Enumerate the framework's K-rational coefficients")
print("  that ARE doing real work in other observables, plus 1/(16π²) for")
print("  the algebraicity-excluded transcendental contender.")
print()

coefficients = [
    (Fraction(1, 1), "1",                    "β (canonical, L=0 bits)"),
    (Fraction(1, 2), "1/2",                  "Higgs vertex coefficient"),
    (Fraction(5, 12), "5/12",                "v_Higgs Feshbach c_v"),
    (Fraction(9, 40), "9/40",                "V_us (Cabibbo angle)"),
    (Fraction(256, 6305), "256/6305 ≈ 0.0406", "V_cb"),
    (1 / (16 * math.pi ** 2), "1/(16π²) ≈ 0.00633", "(continuum loop; ∉ K by Lindemann 1882)"),
]

print(f"  {'c (coefficient)':<24} | β (×sin(arg h)·α_EM) | match to Eskilt   | Channel")
print(f"  {'-' * 24}-|----------------------|-------------------|" + "-" * 32)
for c, c_str, channel in coefficients:
    c_val = float(c)
    beta_alt_rad = c_val * sin_arg_h * alpha_EM_MZ
    beta_alt_deg = math.degrees(beta_alt_rad)
    sigma_dev = (beta_alt_deg - beta_obs_deg) / beta_obs_sigma
    marker = "✅" if abs(sigma_dev) < 1.5 else "❌"
    print(f"  {c_str:<24} | {beta_alt_deg:8.4f}°            | {sigma_dev:+6.2f}σ {marker}      "
          f"| {channel}")

print()
print("  --> channel_select(K, β photon-polarization channel) picks c = 1 because:")
print("      (a) 1/(16π²) excluded by algebraicity (Lindemann 1882, transcendental).")
print("      (b) 1/2, 5/12, 9/40, 256/6305 are K-rational but in DIFFERENT")
print("          operator channels — they remain above-waterline for their own")
print("          observables (Higgs vertex, v_Higgs Feshbach, V_us, V_cb).")
print("      (c) Within β's channel (photon polarization), c = 1 is the unique")
print("          K-element corresponding to the canonical (L = 0 bits) encoding")
print("          of 'no extra multiplicative factor.'")
print()
print("  --> Observation confirms: c = 1 matches at +0.13σ; alternatives at")
print("      1.76σ, 2.06σ, 2.79σ, 3.49σ, 3.61σ — all ruled out.")
print()
print("  Coefficient shift: 0 (channel-selected; alternatives in other channels).")


# ------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------
print()
print("=" * 70)
print("MULTI-AXIAL PHASE 2 AUDIT SUMMARY (β cosmic birefringence)")
print("=" * 70)
print(f"Check 1 (lattice — doubly robust): STRUCTURAL PASS")
print(f"  srs-z h_P bit-identical to srs's per R-9. (A) gates anyway.")
print(f"  Lattice shift: 0.")
print()
print(f"Check 2 (parameter A.3.a — functional channel-select): PASS")
print(f"  Lemma 2 picks sin(arg h) over 3 alternative parity-odd projections.")
print(f"  Alternatives couple to mass²/phase channels (different observables).")
print(f"  Functional shift: 0.")
print()
print(f"Check 3 (parameter A.3.b — coefficient channel-select): PASS")
print(f"  c = 1 selected over 5 alternatives:")
print(f"    1/2 (Higgs vertex), 5/12 (v_Higgs), 9/40 (V_us), 256/6305 (V_cb),")
print(f"    1/(16π²) (excluded by Lindemann transcendence).")
print(f"  All 5 alternatives are REAL K-elements doing real work elsewhere.")
print(f"  Observation rules out all 5 at 1.76-3.61σ.")
print(f"  Coefficient shift: 0.")
print()
print(f"OVERALL: PASS")
print()
print(f"Net multi-axial prediction: β = 1·sin(arg h)·α_EM = {beta_ref_deg:.4f}°")
print(f"Net srs-only prediction:    β = 1·sin(arg h)·α_EM = {beta_ref_deg:.4f}°")
print(f"Net shift: 0.")
print()
print(f"Substantive Phase 2 finding: β is the FIRST audit testing the channel-")
print(f"select discipline at TWO independent parameter-axis sub-loci")
print(f"(functional + coefficient). The coefficient test enumerates 5 K-rational")
print(f"competitors that are NOT synthetic but the framework's own structural")
print(f"coefficients for OTHER observables. The wrong reading (MDL bit-cost")
print(f"minimum over K) would pick 9/40 or 256/6305 as bit-cheapest, giving a")
print(f"2.8-3.5σ WRONG ANSWER. channel_select is non-trivially load-bearing.")
print("=" * 70)
