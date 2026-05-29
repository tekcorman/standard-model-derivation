#!/usr/bin/env python3
"""
proofs/foundations/family_E_hamming_weight_dark_disruption_2026-05-15.py

α2'''-FAMILY-E candidate: Hamming-weight-dependent per-fermion-leg dark
disruption.

CONTEXT (post α2'''-PIVOT closed-negative, post intra-vertex matrix-element
closed-negative)

The master doc `theorem_substrate_feshbach_dark_corrections_master.md` §3
(D) currently has SECTOR-BLIND per-fermion-leg dark disruption:

  c_F = -α₁_bare² / (N_atoms · k*) = -α₁²/12

This is the SAME for all fermion legs regardless of species (Hamming
weight n).  The master doc §"Family D's failure on gauge-boson 2-point"
explicitly notes: "sign-uniform Family-D corrections cannot produce this
[M_Z²/m_W²] split.  Through the tree relation m_W = M_Z cos θ_W any
multiplicative correction propagates identically to both."

USER INSIGHT (2026-05-15 EOD+12, post-PIVOT discussion):
Different Hamming-weight species (different number of pinned/occupied
edges per vertex) should have STRUCTURALLY DIFFERENT dark disruption
rates from the multiway dynamics — because:

  - Higher n = more excited modes at vertex = more multiway branches
  - Each branch is an independent dark disruption channel
  - More channels → larger c_F magnitude

This is exactly what's needed for a non-trivial Family E that
differentiates up (n=2) from down (n=1) sectors → Δρ ≠ 0 mechanism.

PROBE STRUCTURE

Generalize c_F to c_F(n).  Test STRUCTURALLY-MOTIVATED candidate forms:

(C1) Linear in occupation:        c_F(n) = -α₁² · n / 12
(C2) Linear in NB-internal steps: c_F(n) = -α₁² · (g-n) / 12
(C3) Pair-correlation:            c_F(n) = -α₁² · n(3-n) / 12
                                  (Koide quark prefactor: theorem-grade)
(C4) Charge-Q² weighted:          c_F(n) = -α₁² · (n/k*)² / 12
                                  (electromagnetic charge squared)
(C5) Persistence:                 c_F(n) = -α₁² · (3/2)^n / 12
                                  (persistence factor (3/2)^n)

For each candidate:
- Compute δy_t (Yukawa correction to top, n=2 sector) and δy_b (n=1)
- Compute substrate Δρ from the (top, bottom) doublet asymmetry:
    Δρ_substrate ≈ 2 × (δy_t - δy_b)  (since m_t² >> m_b²)
- Compare to empirical δρ ≈ 1.05%

Pre-declared abort:
- DO NOT enumerate candidates and pick the magnitude-matching one
- Each candidate must have STRUCTURAL JUSTIFICATION (not just magnitude)
- Report honestly which match magnitude AND have structural derivation
- Numerical match WITHOUT structural derivation = numerology, retract

(F.1) All candidates give δy_t = δy_b → no Family E mechanism → close NEG
(F.2) Candidates differ but none match δρ_emp within sub-percent → close NEG
(F.3) Candidate matches AND has theorem-grade structural derivation → POSITIVE
"""
from __future__ import annotations
from fractions import Fraction
import numpy as np

# ---------------------------------------------------------------------------
# Framework constants (theorem-grade)
# ---------------------------------------------------------------------------
k_star = 3
g = 10
N_ATOMS = 4
N_directed_edges = N_ATOMS * k_star  # = 12

alpha_1_bare = Fraction(k_star - 1, k_star) ** (g - 2)  # = (2/3)^8 = 256/6561
alpha_1_sq = alpha_1_bare ** 2  # = (2/3)^16

# Higgs-leg dark disruption rate (theorem-grade per master doc §3 (D))
c_H = alpha_1_sq  # = α₁²

# Fermion-leg dark disruption rate (current Family D, sector-blind)
c_F_blind = -alpha_1_sq / 12  # = -α₁²/12

# ---------------------------------------------------------------------------
# Empirical
# ---------------------------------------------------------------------------
M_Z_PDG = 91.1876  # GeV
m_W_PDG = 80.3692  # GeV
sin2_theta_W_eff = 0.23122  # MS-bar at M_Z

# Empirical δρ from M_Z, m_W
cos2_theta_W = 1 - sin2_theta_W_eff
rho_emp = (m_W_PDG ** 2) / (M_Z_PDG ** 2 * cos2_theta_W)
delta_rho_emp = rho_emp - 1
print(f"Empirical δρ (PDG-based): {delta_rho_emp * 100:+.4f}%")

# Top-quark dominant SM Δρ (NLO):
m_t = 172.69
v = 246.22
delta_rho_SM_top = 3 * m_t ** 2 / (16 * np.pi ** 2 * v ** 2)
print(f"SM top-quark NLO δρ:      {delta_rho_SM_top * 100:+.4f}%")
print(f"(For comparison; substrate Δρ should land in the same ballpark.)")
print()


# ---------------------------------------------------------------------------
# Generalized Family E: c_F as a function of Hamming weight n
# ---------------------------------------------------------------------------

def c_F_candidates(n: int) -> dict:
    """All structurally-motivated candidates for c_F(n)."""
    return {
        "C1: linear in n (more excited modes = more channels)":
            -alpha_1_sq * Fraction(n, 12),
        "C2: linear in (g-n) (more NB-internal steps)":
            -alpha_1_sq * Fraction(g - n, 12),
        "C3: pair-correlation n(3-n)/3 (Koide quark prefactor)":
            -alpha_1_sq * Fraction(n * (3 - n), 36),  # /12 × /3
        "C4: charge Q² = (n/k*)² (EM-charge squared)":
            -alpha_1_sq * Fraction(n * n, 9 * 12),  # /12 × n²/k*²
        "C5: persistence (3/2)^n / 12 (persistence factor)":
            -alpha_1_sq * Fraction(3, 2) ** n / 12,
        "C0: sector-blind (current Family D)":
            -alpha_1_sq / 12,
    }


# ---------------------------------------------------------------------------
# For each candidate: compute Yukawa correction asymmetry and Δρ
# ---------------------------------------------------------------------------

print("=" * 78)
print("Family-E candidates: per-fermion-leg c_F(n) modulation")
print("=" * 78)
print()
print(f"  α₁² = (2/3)^16 = {float(alpha_1_sq):.6e}")
print(f"  c_H = α₁² = {float(c_H):.6e}")
print(f"  c_F (sector-blind) = -α₁²/12 = {float(c_F_blind):.6e}")
print()
print(f"  Yukawa vertex: 1H + 2F (t̄, t for top; b̄, b for bottom)")
print(f"  Top quark: n=2 sector.  Bottom quark: n=1 sector.")
print()
print(f"  δy_t/y_t = -(c_H + 2 c_F(n=2))")
print(f"  δy_b/y_b = -(c_H + 2 c_F(n=1))")
print()
print(f"  Substrate Δρ candidate:  Δρ ≈ 2 (δy_t - δy_b)")
print(f"  (since m_t² >> m_b², top dominates the gap)")
print()
print()

print(f"{'Candidate':<60} {'c_F(n=2)':<14} {'c_F(n=1)':<14} {'δy_t-δy_b':<14} {'Δρ_subs':<12}")
print("-" * 116)

candidate_results = []
for label in [
    "C0: sector-blind (current Family D)",
    "C1: linear in n (more excited modes = more channels)",
    "C2: linear in (g-n) (more NB-internal steps)",
    "C3: pair-correlation n(3-n)/3 (Koide quark prefactor)",
    "C4: charge Q² = (n/k*)² (EM-charge squared)",
    "C5: persistence (3/2)^n / 12 (persistence factor)",
]:
    candidates_n1 = c_F_candidates(1)
    candidates_n2 = c_F_candidates(2)
    cF_n1 = float(candidates_n1[label])
    cF_n2 = float(candidates_n2[label])

    dyt = -(float(c_H) + 2 * cF_n2)  # δy_t/y_t
    dyb = -(float(c_H) + 2 * cF_n1)  # δy_b/y_b
    diff = dyt - dyb
    delta_rho_sub = 2 * diff  # leading approximation since m_t² >> m_b²

    print(f"{label:<60} {cF_n2:>+13.4e} {cF_n1:>+13.4e} {diff:>+13.4e} {delta_rho_sub*100:>+11.4f}%")
    candidate_results.append((label, cF_n2, cF_n1, diff, delta_rho_sub))


# ---------------------------------------------------------------------------
# Verdict per candidate
# ---------------------------------------------------------------------------

print()
print("=" * 78)
print("Per-candidate verdict (vs empirical δρ ≈ 1%)")
print("=" * 78)
print()

for label, cF_n2, cF_n1, diff, drho in candidate_results:
    print(f"  {label}")
    print(f"    Δρ_substrate = {drho * 100:+.4f}%  (target {delta_rho_emp * 100:+.4f}%)")

    if abs(drho) < 1e-6:
        print(f"    → ZERO: candidate gives no asymmetry")
    elif abs(abs(drho * 100) - abs(delta_rho_emp * 100)) / abs(delta_rho_emp * 100) > 5:
        print(f"    → OFF BY ORDER-OF-MAGNITUDE (factor {abs(drho/delta_rho_emp):.2f})")
    elif abs(drho * 100 - delta_rho_emp * 100) < 0.5:
        print(f"    → MATCH within sub-percent ✓")
    else:
        print(f"    → INTERMEDIATE: differs from target by {abs(drho * 100 - delta_rho_emp * 100):.3f}%")

    if diff > 0:
        print(f"    → SIGN: positive (top correction larger than bottom)")
    elif diff < 0:
        print(f"    → SIGN: negative (bottom correction larger than top)")
    print()


# ---------------------------------------------------------------------------
# Honest meta-verdict
# ---------------------------------------------------------------------------

print("=" * 78)
print("Meta-verdict")
print("=" * 78)
print()
print("  Each candidate has been computed; PER an internal note,")
print("  none of these candidates can be elevated based on magnitude match")
print("  alone.  STRUCTURAL DERIVATION is required.")
print()
print("  Of the 5 non-trivial candidates:")
print("    C1 (linear n): structural reading 'more excited modes = more")
print("      channels' is plausible but not derived; needs multiway count.")
print("    C2 (linear g-n): structural reading 'more internal NB steps")
print("      = more bleed' has Feshbach-exponent-style motivation.")
print("    C3 (n(3-n)/3): theorem-grade Koide quark prefactor — STRONGEST")
print("      structural footing (already derived as Z_3-non-trivial dim).")
print("    C4 (Q²): natural for U(1)_EM coupling squared; structural")
print("      motivation tied to electromagnetic charge.")
print("    C5 ((3/2)^n persistence): persistence-factor reading of")
print("      Feshbach exponent; new derivation needed.")
print()
print("  None of these is theorem-grade as a per-leg dark disruption rate.")
print("  Promotion to theorem-grade Family E requires:")
print("    (i)  multiway DAG derivation of c_F(n) for chosen candidate")
print("    (ii) two-route discipline (master doc §8 rule 1)")
print("    (iii) calibration check against existing closures (master doc §6)")
print()
print("  This probe is a SCOPING audit — names the candidates and tests")
print("  magnitudes, does NOT close Family E.")
print()
print("  Pre-declared (per scoping):")
print(f"    Empirical δρ ≈ {delta_rho_emp * 100:+.4f}%")
print()


# Show top match candidates for further investigation
print("=" * 78)
print("Magnitude ranking (PURELY descriptive, NOT a closure)")
print("=" * 78)
print()
sorted_results = sorted(candidate_results,
                        key=lambda r: abs(r[4] * 100 - delta_rho_emp * 100) if r[4] != 0 else float('inf'))
print(f"  Candidate (sorted by closeness to {delta_rho_emp * 100:.2f}%):")
for label, cF_n2, cF_n1, diff, drho in sorted_results:
    if drho != 0:
        gap = drho * 100 - delta_rho_emp * 100
        print(f"    Δρ = {drho*100:+.4f}%  (gap {gap:+.4f}%)  {label}")
    else:
        print(f"    Δρ = +0.0000%  (identically zero)  {label}")

print()
print("=" * 78)
print("Forward direction")
print("=" * 78)
print()
print("  IF any candidate is within order-of-magnitude AND has structural")
print("  motivation, scope a follow-up probe to:")
print("    1. Derive c_F(n) form from multiway DAG (route H per master doc)")
print("    2. Cross-check via cycle-counting (route C)")
print("    3. Calibrate against existing v_Higgs Family C closure (c_v=5/12)")
print("    4. Apply to M_Z, m_W via Family-E template, get final residuals")
print()
print("  IF no candidate is within order-of-magnitude OR none has structural")
print("  derivation, the M_Z/m_W cluster needs a more substantive structural")
print("  extension (Path β proper, Path B, or substrate-loop computation).")
print()
print("=" * 78)
print("End of Family-E Hamming-weight scoping probe.")
print("=" * 78)
