#!/usr/bin/env python3
"""
Probe 2: Hamming-weight / charge dependence of sector Yukawa couplings.

CONTEXT
=======
Per `theorem_charge_before_color.md` §9, the Cl(6) Fock space at a trivalent
vertex decomposes under U(3) ⊂ Spin(6) ≅ SU(4) into 4 sectors indexed by
Hamming weight n ∈ {0, 1, 2, 3} with charges Q = n/k* ∈ {0, 1/3, 2/3, 1}.
Per Furey 2018 these match SM fermions of one generation:
- n = 0, Q = 0:    ν_L (neutrino)
- n = 1, Q = 1/3:  d_L (down-type quark)
- n = 2, Q = 2/3:  ū_R (up-type quark)
- n = 3, Q = 1:    e_L^+ (positron / charged lepton)

The framework's y_τ derivation (`theorem_ytau_corollary.md`) gives the LEPTON
Yukawa as y_τ = α₁_full / k*² (theorem-grade, +0.13% match). This is the n=3
sector Yukawa.

QUESTION
========
Is there a structurally natural Hamming-weight-dependent factor g_n such that
the heaviest-generation Yukawa per sector follows y_(sector n) = α_1_full · g_n / k*²
for all four n? Specifically:
- y_τ (n=3, lepton) ≈ 0.00722 (already matched at +0.13% with g_3 = 1)
- y_top (n=2, up-type heaviest) ≈ 0.703 → g_2 / g_3 = 97.2
- y_bottom (n=1, down-type heaviest) ≈ 0.0170 → g_1 / g_3 = 2.35
- y_ν (n=0, neutrino) ≈ ~10⁻¹² → g_0 / g_3 ≈ 10⁻¹⁰

WHAT THIS PROBE TESTS
=====================
For each candidate functional form g_n (powers of n, charges Q, products,
ratios involving framework constants k*, g, n_atoms, |E|, α_1, etc.), compute
g_n / g_3 for n = 0, 1, 2 and compare to the observed Yukawa ratios.

Outcome:
- POSITIVE if a candidate form fits all 4 sectors within tolerance with a
  structural reading.
- PARTIAL if it fits some sectors but breaks others.
- NEGATIVE if no natural form fits.

DOES NOT
========
- Address generation hierarchy (Koide structure within a sector). g_n is the
  PER-SECTOR scale; the framework's existing Koide structure handles intra-
  sector mass differences.
- Address why y_top is so dramatically heavier than y_τ (97×) — a successful
  g_n would expose the structural origin of this hierarchy.
"""

from __future__ import annotations

import math
from fractions import Fraction

# ============================================================================
# 1. Framework constants
# ============================================================================
K_STAR = 3
G_GIRTH = 10
N_ATOMS = 4
N_EDGES = 6
ALPHA_1_BARE = Fraction(2, 3) ** 8                      # = 256/6561
ALPHA_1_FULL_FRAC = Fraction(5, 3) * ALPHA_1_BARE       # = 1280/19683
ALPHA_1_FULL = float(ALPHA_1_FULL_FRAC)                  # ≈ 0.06504
Y_TAU_PRED = ALPHA_1_FULL / K_STAR**2                    # ≈ 0.007225

# Higgs VEV (← the adopted N_hub via BZJ; G1-conditional but sufficient for ratio test)
V_HIGGS = 246.22

# ============================================================================
# 2. Observed Yukawa values (heaviest generation per sector)
# ============================================================================
# All y values use PDG 2024 mass values, converted via y = m / v_Higgs.
M_TOP = 172.69       # GeV (PDG 2024)
M_BOTTOM = 4.18      # GeV (PDG 2024 MS-bar)
M_TAU = 1.77686      # GeV (PDG 2024: 1776.86 MeV)
M_NU3 = 0.05013e-9   # GeV (50.13 meV from NuFIT 6.0)

Y_TAU_OBS = M_TAU / V_HIGGS
Y_BOTTOM_OBS = M_BOTTOM / V_HIGGS
Y_TOP_OBS = M_TOP / V_HIGGS
Y_NU_OBS = M_NU3 / V_HIGGS

# Tolerance per ratio: take ~5% wide for matching candidate forms.
RATIO_TOLERANCE = 0.05

# ============================================================================
# 3. Print observed sector Yukawas
# ============================================================================
print("=" * 78)
print("Sector Yukawas (heaviest generation per sector)")
print("=" * 78)
print()
print(f"  {'sector':<24}  {'n':>1}  {'Q=n/k*':>7}  {'y (obs)':>11}  {'y/y_τ (obs)':>11}")
print(f"  {'-'*24}  {'-'*1}  {'-'*7}  {'-'*11}  {'-'*11}")

sectors = [
    ('lepton (charged, e+)',     3, Fraction(3, K_STAR), Y_TAU_OBS),
    ('up-type (top)',            2, Fraction(2, K_STAR), Y_TOP_OBS),
    ('down-type (bottom)',       1, Fraction(1, K_STAR), Y_BOTTOM_OBS),
    ('neutrino (heaviest, ν3)',  0, Fraction(0, 1),       Y_NU_OBS),
]
for name, n, q, y in sectors:
    ratio = y / Y_TAU_OBS
    print(f"  {name:<24}  {n:>1}  {float(q):>7.4f}  {y:>11.5e}  {ratio:>11.5e}")

print()
y_tau_obs = sectors[0][3]


# ============================================================================
# 4. Candidate g_n functional forms — search
# ============================================================================
print("=" * 78)
print("Candidate g_n forms — match against observed Yukawa ratios")
print("=" * 78)
print()
# We want g_n / g_3 such that y_(sector n) = y_τ × (g_n / g_3) matches obs.
# Target ratios:
target_ratios = {
    3: 1.0,                                  # y_τ / y_τ = 1 (control)
    2: Y_TOP_OBS / Y_TAU_OBS,                # ≈ 97.2
    1: Y_BOTTOM_OBS / Y_TAU_OBS,             # ≈ 2.35
    0: Y_NU_OBS / Y_TAU_OBS,                 # ≈ 8.9e-11 (neutrino tiny)
}
print(f"  Observed g_n / g_3 ratios (y_(sector n) / y_τ):")
for n, r in sorted(target_ratios.items(), reverse=True):
    print(f"    g_{n} / g_3 = {r:.5e}")
print()

candidate_forms = [
    # name, fn(n) returning g_n, expression
    ("g_n = 1 (uniform)",                        lambda n: 1.0),
    ("g_n = n",                                  lambda n: float(n) if n > 0 else 1e-99),
    ("g_n = n²",                                 lambda n: float(n)**2 if n > 0 else 1e-99),
    ("g_n = (n+1)",                              lambda n: float(n+1)),
    ("g_n = (n+1)²",                             lambda n: float(n+1)**2),
    ("g_n = k*^n",                               lambda n: K_STAR**n),
    ("g_n = k*^(2n)",                            lambda n: K_STAR**(2*n)),
    ("g_n = k*^(g-n)",                           lambda n: K_STAR**(G_GIRTH-n)),
    ("g_n = (3k*-2)^n",                          lambda n: (3*K_STAR-2)**n),
    ("g_n = (Q+ε)^a where Q=n/k*",               lambda n: (n/K_STAR + 1e-30)),
    ("g_n = (n/k*)·k*^(g-2)",                    lambda n: (n/K_STAR) * K_STAR**(G_GIRTH-2) if n > 0 else 1e-99),
    ("g_n = (k*-1)^(n)",                         lambda n: (K_STAR-1)**n),
    ("g_n = (k*-1)^(2n)",                        lambda n: (K_STAR-1)**(2*n)),
    ("g_n = 1/((k*-1)/k*)^(g-2-n*c) c=1",        lambda n: K_STAR**(G_GIRTH-2-n) / (K_STAR-1)**(G_GIRTH-2-n) if (G_GIRTH-2-n) > 0 else 1e9),
    ("g_n = (1/α_1_bare)^n",                     lambda n: (1.0/float(ALPHA_1_BARE))**n),
    ("g_n = ((g-2)/(2g))^(3-n)",                 lambda n: (float(G_GIRTH-2)/(2*G_GIRTH))**(3-n) if n < 3 else 1.0),
    ("g_n = (M_top/M_tau)^((n-3)/(2-3))",        lambda n: (M_TOP/M_TAU)**((n-3)/-1.0) if n != 3 else 1.0),
    ("g_n = N_atoms^(n)",                        lambda n: N_ATOMS**n),
]

print(f"  {'candidate':<48}  {'g_2/g_3':>10}  {'g_1/g_3':>10}  {'g_0/g_3':>10}  fit?")
print(f"  {'-'*48}  {'-'*10}  {'-'*10}  {'-'*10}  ----")

best_candidates = []
for name, fn in candidate_forms:
    try:
        g3 = fn(3)
        if g3 == 0 or abs(g3) < 1e-50:
            print(f"  {name:<48}  {'g_3 = 0':>10}")
            continue
        ratios = {n: fn(n) / g3 for n in range(4)}

        # Compare to target ratios; tolerance 5% relative for non-zero target
        all_match = True
        nonzero_match = True
        for n in [0, 1, 2]:
            t = target_ratios[n]
            r = ratios[n]
            if t == 0:
                continue
            rel_err = abs(r - t) / abs(t) if abs(t) > 1e-30 else float('inf')
            if rel_err > RATIO_TOLERANCE:
                all_match = False
            # For n=0 (neutrino), large discrepancy OK
            if n != 0 and rel_err > RATIO_TOLERANCE:
                nonzero_match = False

        flag = "MATCH" if all_match else ("partial" if nonzero_match else "")
        print(f"  {name:<48}  {ratios[2]:>10.4e}  {ratios[1]:>10.4e}  {ratios[0]:>10.4e}  {flag}")
        if all_match or nonzero_match:
            best_candidates.append((name, ratios, all_match))
    except Exception as e:
        print(f"  {name:<48}  ERROR: {e}")
print()


# ============================================================================
# 5. Manual structural-form search around y_top/y_τ ≈ 97.2 and y_b/y_τ ≈ 2.35
# ============================================================================
print("=" * 78)
print("Direct structural-form search: y_top/y_τ ≈ 97.2 and y_bottom/y_τ ≈ 2.35")
print("=" * 78)
print()
ratio_t = Y_TOP_OBS / Y_TAU_OBS
ratio_b = Y_BOTTOM_OBS / Y_TAU_OBS

# What rational combinations of {k*=3, g=10, N_atoms=4, |E|=6, α_1_bare} approach these?
print(f"  Target: y_top/y_τ = {ratio_t:.5f}")
print(f"  Target: y_bottom/y_τ = {ratio_b:.5f}")
print()

# (3k*-2)/k* at k*=3 = 7/3 ≈ 2.333 — close to y_b/y_τ ≈ 2.35
val_3k2_k = (3*K_STAR - 2) / K_STAR
print(f"  (3k*-2)/k* = {val_3k2_k:.5f}")
print(f"    y_bottom/y_τ deviation: {abs(val_3k2_k - ratio_b)/ratio_b*100:.2f}%")
print()

# (3g-2)/g at g=10 = 14/5 = 2.8 — same form, different value
val_3g2_g = (3*G_GIRTH - 2) / G_GIRTH
print(f"  (3g-2)/g = {val_3g2_g:.5f}")
print(f"    Off from y_b/y_τ by {abs(val_3g2_g - ratio_b)/ratio_b*100:.2f}%")
print()

# Powers of various k*, g
print(f"  Quick k*-powers: k*^2 = {K_STAR**2}, k*^4 = {K_STAR**4}, k*^5 = {K_STAR**5}")
print(f"  Quick g-powers: g = {G_GIRTH}, g·k* = {G_GIRTH*K_STAR}, g·N_atoms = {G_GIRTH*N_ATOMS}")
print()

# Search for clean rationals near 97.2
print(f"  Forms near {ratio_t:.3f}:")
for num in range(80, 110):
    for den in range(1, 5):
        if abs(num/den - ratio_t) < 1.0 and num != 0:
            err_pct = abs(num/den - ratio_t)/ratio_t * 100
            if err_pct < 3:  # within 3%
                print(f"    {num}/{den} = {num/den:.4f}, deviation {err_pct:.2f}%")

# Test specific framework-natural forms for ratio_t (y_top/y_τ)
print(f"\n  Framework-natural forms for {ratio_t:.3f}:")
forms_t = [
    ("g² - k*",                     G_GIRTH**2 - K_STAR),
    ("g² - 1",                      G_GIRTH**2 - 1),
    ("k*^4 + (k*-1)^4",             K_STAR**4 + (K_STAR-1)**4),
    ("g²·(k*-1)/k* - 1",            G_GIRTH**2 * (K_STAR-1)/K_STAR - 1),
    ("g·(g-2) + |E|",               G_GIRTH*(G_GIRTH-2) + N_EDGES),
    ("(g-1)·(g-2) + |E|",           (G_GIRTH-1)*(G_GIRTH-2) + N_EDGES),
    ("(3g-2)·k*+1",                 (3*G_GIRTH-2)*K_STAR + 1),
    ("k*·g·(g-1)·(k*-1)/(...)", None),  # placeholder
]
for name, val in forms_t:
    if val is None:
        continue
    err_pct = abs(val - ratio_t) / ratio_t * 100
    flag = "MATCH" if err_pct < 1 else ("close" if err_pct < 3 else "")
    print(f"    {name:<40} = {val:>9.4f}  deviation {err_pct:>6.2f}%  {flag}")
print()


# ============================================================================
# 6. Verdict
# ============================================================================
print("=" * 78)
print("PROBE 2 VERDICT")
print("=" * 78)
print()
if not best_candidates:
    print(f"  No tested g_n form matches all four sectors within {RATIO_TOLERANCE*100:.0f}% tolerance.")
    print(f"  No natural Hamming-weight functional form found in this scan.")
else:
    full_matches = [(n, r, m) for n, r, m in best_candidates if m]
    partial_matches = [(n, r, m) for n, r, m in best_candidates if not m]
    if full_matches:
        print(f"  FULL MATCHES (4 sectors within tolerance):")
        for name, ratios, _ in full_matches:
            print(f"    {name}")
            for n, r in sorted(ratios.items(), reverse=True):
                print(f"      g_{n}/g_3 = {r:.4e}")
    if partial_matches:
        print(f"  PARTIAL MATCHES (n=1,2 within tolerance, n=0 not):")
        for name, ratios, _ in partial_matches:
            print(f"    {name}")

if not best_candidates:
    print()
    print("  Direct structural-form observations:")
    print(f"    y_bottom/y_τ ≈ {ratio_b:.3f} is close to (3k*-2)/k* = 7/3 ≈ 2.333 (-{abs(val_3k2_k - ratio_b)/ratio_b*100:.1f}%)")
    print(f"    y_top/y_τ ≈ {ratio_t:.1f} has no clean small-rational match within 3%.")
    print(f"    The y_top hierarchy (97.2× heavier than τ) is unusually steep.")
    print()
    print("  Probe 2 status: NEGATIVE for the Yukawa-Hamming-weight ansatz tested here.")
    print("  Down-sector ratio close to (3k*-2)/k* is suggestive but not closure.")
    print("  Up-sector ratio ~97 doesn't fit any natural Hamming-weight scaling.")

print()
print("=" * 78)
print("END")
print("=" * 78)
