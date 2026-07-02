#!/usr/bin/env python3
"""
W24 — δ_Koide structural identity via Wigner D¹ HM (4₁ screw route).

Date: 2026-05-26
Context: continues the Audit B verdict from `W23_audits_ABC_koide_2026-05-26.py`.
Audit B identified D3 as living in the parametric cos-phase δ = 2/9 in
`predictions/delta_Koide.py`. The current derivation there (δ_Bernoulli =
Q(1-Q) = 2/9) is acknowledged as numerical coincidence (Need-B).

THIS PROBE: traces an alternative substrate-mechanical derivation of
δ_Koide = 2/9 that already exists at THEOREM-GRADE in the framework but
is NOT the one chained into `predictions/delta_Koide.py`. Specifically:

    `docs/theorems/theorem_41_screw_wigner.md` + `proofs/masses/wigner_d1_screw_41.py`
    derive δ = 2/9 via TWO routes:
      Route A: HM of Wigner D¹ diagonal survival probabilities at the
               4₁ screw projected onto the [111] C₃ axis.
      Route B: δ = Q / n_gen = (2/3)/3 = 2/9 (algebraic identity given
               substrate-derived Q and n_gen = 3 from Spin(8) triality).

And `proofs/masses/srs_delta_n_derivation.py` uses the SAME Wigner D¹ HM
machinery to derive δ(n) = 2/(9(n+1)) for ALL THREE generation bands
(leptons n=0, down quarks n=1, up quarks n=2), with sub-percent empirical
match for all three species.

If the framework accepts the substrate Wigner D¹ HM route for quark δ(n)
(which it does — currently THEOREM-GRADE-STRUCTURAL per docs/honest_assessment.md),
then by CONSISTENCY δ_Koide at n=0 IS substrate-derived = 2/9 via THE
SAME machinery.

PROBE QUESTIONS:
(1) Verify the substrate derivation of δ_Koide = 2/9 via Wigner D¹ HM.
(2) Compare to the current predictions/delta_Koide.py Bernoulli-moment route.
(3) Quantify D3 residual at the δ-level given exact substrate δ = 2/9.
(4) Identify what (if anything) remains structurally open for Need-B.
"""

import math
from fractions import Fraction
import numpy as np
import sympy as sp

print("=" * 76)
print("W24 — δ_Koide structural identity via Wigner D¹ HM (4₁ screw)")
print("Date: 2026-05-26")
print("=" * 76)

# ============================================================
# Step 1 — Verify the 4₁ screw Wigner D¹ HM derivation
# ============================================================
print()
print("=" * 76)
print("Step 1 — 4₁ screw + [111] C₃ → Wigner D¹ → HM = 2/9")
print("=" * 76)
print()
print("From docs/theorems/theorem_41_screw_wigner.md SS-1 through SS-3:")
print()
print("  Substrate: srs lattice, space group I4₁32 (No. 214), Wyckoff 8a")
print("  4₁ screw axis: R₄ = 90° rotation about [001]")
print("  C₃ site axis: [111]")
print("  Tilt angle: cos β = 1/√3 ≈ 0.5774 → β ≈ 54.74°")
print()
print("In the [111]-quantization frame, Wigner D¹(R₄) has diagonal survival")
print("probabilities (Type 2 = exact algebra, sympy-verified per wigner_d1_screw_41.py):")
print()
print("  |D¹_{+1,+1}|² = 4/9")
print("  |D¹_{ 0, 0}|² = 1/9")
print("  |D¹_{-1,-1}|² = 4/9")
print()
print("Harmonic mean:")
P_plus = Fraction(4, 9)
P_zero = Fraction(1, 9)
P_minus = Fraction(4, 9)
HM = Fraction(3, 1) / (Fraction(1) / P_plus + Fraction(1) / P_zero + Fraction(1) / P_minus)
print(f"  HM(4/9, 1/9, 4/9) = 3 / (9/4 + 9 + 9/4) = 3 / (54/4) = 12/54 = {HM}")
print()
print(f"  ROUTE A: δ = HM(P_+1, P_0, P_-1) = {HM} ≈ {float(HM):.10f}")

# Route B: δ = Q/n_gen
Q = Fraction(2, 3)
n_gen = 3
delta_routeB = Q / n_gen
print()
print(f"  ROUTE B: δ = Q/n_gen = (2/3)/3 = {delta_routeB} ≈ {float(delta_routeB):.10f}")
print()
print(f"  Both routes agree: δ = 2/9 = {float(HM):.10f}")
print()
print("Status per `theorem_41_screw_wigner.md` §6 + §9 (2026-05-08 + 2026-05-17 updates):")
print("  • SS-1 through SS-3 STRICT-SOLID (substrate algebra)")
print("  • Need-A (C₃ irreps = generation labels): CLOSED 2026-05-08 via M1.B Galois closure")
print("  • §6(i) HM=δ_Koide via mass∝1/inverse-propagator postulate: CLOSED 2026-05-17 b1'")
print("    via Landauer saturation + A-IT (theorem_mass_propagator_overdetermination.md §9)")
print("  • Need-B (δ-as-cos-phase-in-radians): STILL OPEN")


# ============================================================
# Step 2 — n-DEPENDENCE: framework's claim does NOT extend via Wigner HM
# ============================================================
print()
print("=" * 76)
print("Step 2 — n-dependence: Wigner d^j HM ≠ 2/(9(n+1)) for n > 0")
print("=" * 76)
print()
print("DIRECT NUMERICAL CHECK at cos(β) = 1/3 (computed in this session):")
print()

import scipy.special as _sp
from scipy.special import factorial as _fact

def _wigner_d_diag(j, m, beta):
    cos_half = math.cos(beta / 2)
    sin_half = math.sin(beta / 2)
    result = 0.0
    for s in range(int(2 * j) + 2):
        n1 = j + m - s
        n2 = s
        n3 = m - m + s
        n4 = j - m - s
        if n1 < 0 or n2 < 0 or n3 < 0 or n4 < 0:
            continue
        n1, n2, n3, n4 = int(n1), int(n2), int(n3), int(n4)
        sign = (-1) ** int(s)
        numer = math.sqrt(_fact(int(j + m), exact=True) * _fact(int(j - m), exact=True)
                          * _fact(int(j + m), exact=True) * _fact(int(j - m), exact=True))
        denom = float(_fact(n1, exact=True) * _fact(n2, exact=True)
                      * _fact(n3, exact=True) * _fact(n4, exact=True))
        power_cos = int(2 * j - 2 * s)
        power_sin = int(2 * s)
        term = sign * numer / denom
        if power_cos > 0:
            term *= cos_half ** power_cos
        if power_sin > 0:
            term *= sin_half ** power_sin
        result += term
    return result

beta = math.acos(1.0/3.0)
print(f"  cos(β) = 1/3, β = {math.degrees(beta):.4f}°")
print()
print(f"  {'j':>3} {'HM(full d^j)':>15} {'HM(C_3 m=+1,0,-1)':>22} {'2/(9(j+1))':>15} {'match?':>10}")

for j_test in [1, 2, 3, 4]:
    diags = []
    for m_idx in range(int(2*j_test) + 1):
        m_val = j_test - m_idx
        diags.append(_wigner_d_diag(j_test, m_val, beta))
    probs = [d*d for d in diags]
    nz = [p for p in probs if p > 1e-12]
    hm_full = len(nz) / sum(1/p for p in nz)
    # C_3 sub-block at m = +1, 0, -1
    mid = j_test
    sub_probs = [diags[mid-1]**2, diags[mid]**2, diags[mid+1]**2]
    hm_sub = 3 / sum(1/p for p in sub_probs if p > 1e-12)
    target = 2.0 / (9 * (j_test))  # The framework's claim δ(n) = 2/(9(n+1)) with n+1 = j
    match_str = "✓" if abs(hm_full - target) < 1e-6 else "✗"
    print(f"  {j_test:>3} {hm_full:>15.10f} {hm_sub:>22.10f} {target:>15.10f} {match_str:>10}")

print()
print("→ Wigner d¹ HM = 2/9 (n=0 leptons): MATCHES")
print("→ Wigner d^j HM for j>1: does NOT match the 2/(9(n+1)) formula")
print()
print("The framework's δ(n) = 2/(9(n+1)) for n>0 DOES NOT come from direct")
print("Wigner d^j HM extension. It comes from MDL ALLOCATION:")
print()
print("  srs_delta_n_derivation.py Approach 2 (THE actual framework derivation):")
print("    'TOTAL symmetry-breaking information is still 2/9 (from n=0 case)")
print("     but SHARED among n+1 independent Koide triples (one per Fock mode).")
print("     By MDL equal-allocation: δ(n) = 2/9 / (n+1).'")
print()
print("EMPIRICAL CONSEQUENCES of the heuristic for n>0:")
species = [
    ("leptons (n=0)", 0, 2/9, "~0.003% (≈ 2 ppm at δ-level)"),
    ("down quarks (n=1)", 1, 0.1102, "0.84% (percent-level)"),
    ("up quarks (n=2)", 2, 0.0744, "0.41% (percent-level)"),
]
for label, n, observed, match in species:
    pred = 2.0 / (9 * (n + 1))
    print(f"  n={n} ({label:18s}): δ_pred = {pred:.6f}, δ_obs = {observed:.6f}  ({match})")
print()
print("The n=0 lepton case empirically matches at PPM precision.")
print("The n>0 quark cases match at PERCENT precision.")
print("This 100× precision gap suggests the MDL allocation argument has")
print("sub-leading structural content not in the simple δ_0/(n+1) formula.")


# ============================================================
# Step 3 — predictions/delta_Koide.py uses the obsolete framing
# ============================================================
print()
print("=" * 76)
print("Step 3 — predictions/delta_Koide.py uses inferior derivation framing")
print("=" * 76)
print()
print("Current `predictions/delta_Koide.py` derives δ via:")
print()
print("  δ = Q · (1 - Q) = (2/3) · (1/3) = 2/9   [Bernoulli moment]")
print()
print("This is a PURELY ALGEBRAIC identity GIVEN the cos-form parametrization.")
print("It produces the same value but its dimensional interpretation is the")
print("'numerical coincidence' Need-B framing (delta_Koide_derivation.md line 3):")
print()
print('  "the IDENTIFICATION of δ_Bernoulli (variance, dimensionless) with the')
print('   Koide cosine PHASE δ in radians ... is a NUMERICAL coincidence"')
print()
print("The Wigner D¹ HM route has IDENTICAL dimensional problem but with")
print("SUBSTANTIAL substrate-mechanical structure underneath:")
print()
print("  • 4₁ screw axis is a substrate-derived symmetry (Wyckoff 8a + I4₁32)")
print("  • [111] C₃ axis is the substrate site symmetry")
print("  • Wigner D¹ at the tilt angle β = arccos(1/√3) is exact algebra")
print("  • HM = 2/9 is exact rational arithmetic")
print("  • Need-A (C₃ irreps = generations) is closed via M1.B")
print("  • Mass-propagator interpretation closed via Landauer + A-IT")
print()
print("→ The Wigner route promotes δ_Koide from 'numerical coincidence' to")
print("  'substrate-derived value with ONE OPEN dimensional convention (Need-B)'")


# ============================================================
# Step 4 — D3 residual quantification given exact substrate δ = 2/9
# ============================================================
print()
print("=" * 76)
print("Step 4 — D3 residual at δ-level given substrate δ = 2/9 EXACT")
print("=" * 76)
print()

delta_substrate = 2.0 / 9.0
delta_PDG_central = 0.2222227
delta_PDG_sigma = 0.0000009

print(f"  Substrate value:  δ = 2/9 = {delta_substrate:.10f}  (exact)")
print(f"  PDG fit:          δ_obs = {delta_PDG_central} ± {delta_PDG_sigma}")
print(f"  Gap:              δ_obs - 2/9 = {delta_PDG_central - delta_substrate:+.10f}")
print(f"  Gap in ppm:       {(delta_PDG_central - delta_substrate)/delta_substrate * 1e6:+.2f} ppm at δ-level")
print(f"  Gap in σ_PDG_δ:   {(delta_PDG_central - delta_substrate)/delta_PDG_sigma:+.2f} σ")

# Sensitivity of m_e and m_e/m_μ to δ near δ = 2/9
epsilon = math.sqrt(2)
delta_0 = 2.0/9.0
factors_0 = sorted([1 + epsilon * math.cos(2*math.pi*j/3 + delta_0) for j in range(3)])
f_min, f_mid, f_max = factors_0

# Sensitivities
def f_of_delta(delta):
    return sorted([1 + epsilon * math.cos(2*math.pi*j/3 + delta) for j in range(3)])

h = 1e-9
fs_plus = f_of_delta(delta_0 + h)
fs_minus = f_of_delta(delta_0 - h)
df_min_dd = (fs_plus[0] - fs_minus[0]) / (2*h)
df_mid_dd = (fs_plus[1] - fs_minus[1]) / (2*h)
df_max_dd = (fs_plus[2] - fs_minus[2]) / (2*h)

# d(m_e/m_μ)/dδ where m_e/m_μ = (f_min/f_mid)²
ratio_emu = (f_min/f_mid)**2
d_ratio_dd = 2 * (f_min/f_mid) * (df_min_dd*f_mid - f_min*df_mid_dd)/(f_mid**2)
print()
print(f"  Sensitivity of m_e/m_μ to δ near 2/9:")
print(f"    d(m_e/m_μ)/dδ = {d_ratio_dd:.4f}")
print(f"    m_e/m_μ at δ=2/9: {ratio_emu:.6e}")
print(f"    Relative sensitivity: d ln(m_e/m_μ) / dδ = {d_ratio_dd/ratio_emu:.4f}")

D3_ppm = 9.83  # from W23 / W22
d_delta_for_D3 = D3_ppm * 1e-6 * ratio_emu / d_ratio_dd
print()
print(f"  D3 residual (from W23) = {D3_ppm} ppm in m_e/m_μ direct test")
print(f"  Implied δ_shift from D3: {d_delta_for_D3:+.4e} absolute = {d_delta_for_D3/delta_substrate*1e6:+.3f} ppm at δ")
print()
print(f"  COMPARISON:")
print(f"    PDG δ_obs - 2/9    = {delta_PDG_central - delta_substrate:+.4e}")
print(f"    Implied from D3    = {d_delta_for_D3:+.4e}")
print(f"    Match (same order, signs may differ): both are O(ppm) at δ-level")

# ============================================================
# Step 5 — Synthesis
# ============================================================
print()
print("=" * 76)
print("SYNTHESIS")
print("=" * 76)
print()
print("""
HONEST SYNTHESIS (post user pushback on n-dependence):

(1) For LEPTONS (n=0) ONLY: the framework has a SUBSTRATE-MECHANICAL
    derivation of δ = 2/9 that is genuinely theorem-grade:
      • Route A (Wigner D¹ HM at 4₁ screw + [111] frame): SS-1 through
        SS-3 strict-solid per theorem_41_screw_wigner.md
      • Route B (δ = Q/n_gen): theorem-grade given M1.B closure of Need-A
      • Both give δ = 2/9 EXACTLY at substrate level for the LEPTON case
    Empirical match for leptons at ~2 ppm (within ~2σ_PDG_δ).

(2) For QUARKS (n > 0): the framework's δ(n) = 2/(9(n+1)) is HEURISTIC
    per the MDL allocation argument. It does NOT come from Wigner d^j
    HM at higher j (verified numerically above: HM ≠ 2/(9(n+1)) for j>1).
    Empirical match: 0.84% (down), 0.41% (up) — PERCENT precision, not
    ppm. The "δ(n) is theorem-grade for all n" framing in docs/honest_
    assessment.md / M_persistence is overreach in this specific respect:
    the n-dependence formula is heuristic, even though the n=0 case is
    substrate-derived.

(3) D3 RESIDUAL (m_e/m_μ direct test +9.83 ppm → ~1 ppm at δ-level for
    leptons): consistent with the framework's n=0 substrate δ = 2/9
    at the ~2 σ_PDG_δ level. NOT a structural defect distinct from PDG
    precision in the lepton case.

(4) The n=0 → n>0 precision gap (ppm → percent) is suspicious:
    • If n=0 were truly clean (substrate-exact via Wigner D¹ HM) and the
      MDL allocation truly captured n>0, the n>0 cases should match
      better than percent.
    • Quark Koide-phase observation has its own uncertainties from
      running masses at common scale; ~1% is plausible measurement
      precision, not structural defect.
    • Or: the n-dependence has sub-leading structural content the
      framework hasn't derived.

(5) REVISED NEED-B (sharpened):
    Two open structural questions:
    (B-rad) Why is the dimensionless Wigner-HM value 2/9 interpreted
            as cos-phase in RADIANS? (dimensional convention)
    (B-n)   What is the EXACT substrate derivation of δ(n) for n>0?
            The current MDL allocation argument is heuristic.

    Closing (B-rad) for leptons would not automatically close (B-n) for
    quarks. They are structurally distinct sub-questions of Need-B.

(6) IMPLICATIONS FOR D1/D3:
    • D3 (m_e/m_μ residual) is at the precision floor of n=0 substrate
      δ = 2/9 vs PDG measurement. Sub-σ_PDG closure unlikely without
      external precision improvement OR sub-leading substrate term.
    • D1 (y_τ residual −10.8 ppm) is a SEPARATE issue from δ — it lives
      in Family-D α₁² leading-only derivation (master doc §3). Audits
      A/B/C didn't address D1.

The W23 Audit B verdict needs amendment: "δ is parametric numerical
coincidence" is correct ONLY in the sense that δ(0) = 2/9 has dimensional
identification problem (Need-B (B-rad)). The Wigner HM at n=0 IS substrate-
derived. The user's pushback identified my overreach: I conflated n=0
clean derivation with the broader n>0 picture which IS heuristic.
""")
