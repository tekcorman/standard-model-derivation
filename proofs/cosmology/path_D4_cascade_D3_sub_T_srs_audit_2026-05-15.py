#!/usr/bin/env python3
"""
Path D probe (D.4) — Cascade theorem D3 sub-T_srs epoch audit (2026-05-15 EOD+1).

HYPOTHESIS
----------
The 2026-05-09 audit confirmed cascade theorem holds for T < T_srs.  But
D1/D2/D3 each have FURTHER implicit assumptions that might break in some
sub-T_srs regime, introducing a substrate-derived intermediate scale where
H(z) deviates from coasting.

This probe audits D3 (and adjacent D1/D2 dependencies) for sub-T_srs
epoch restrictions.

If D3 has additional sub-T_srs structure, that could yield a
substrate-derived intermediate scale (e.g., the framework's electroweak,
QCD, or BBN scales) where the cascade theorem breaks naturally.

APPROACH
--------
1. Catalog D1/D2/D3 assumptions beyond "T < T_srs."
2. For each assumption, check whether the framework's substrate-internal
   scales (v_higgs, T_QCD, T_BBN) would break it.
3. Quantify any resulting H(z) modification.
"""

from __future__ import annotations
import math
import sys
import os

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ============================================================================
# Substrate primitives + framework's intermediate scales
# ============================================================================
K_STAR = 3
G_GIRTH = 10
ALPHA_1 = (2.0 / 3.0) ** (G_GIRTH - 2)
N_G_PER_VERTEX = 15
C_SRS_BITS = N_G_PER_VERTEX * math.log2(1.0 / ALPHA_1)
LN2 = math.log(2)
T_SRS_PLANCK = C_SRS_BITS / LN2
T_PLANCK_K = 1.416784e32
T_CMB_TODAY_K = 2.7255

# Framework substrate-internal scales (theorem-grade upstream)
V_HIGGS_GEV = 246.22  # Row P10
M_PL_GEV = 1.22089e19  # CODATA
T_EW_K = V_HIGGS_GEV / M_PL_GEV * T_PLANCK_K  # T_EW ≈ v_higgs
T_EW_PLANCK = V_HIGGS_GEV / M_PL_GEV

# QCD scale (Row P? — not a framework-derived scale, observational)
T_QCD_GEV = 0.2  # ΛQCD ~ 200 MeV
T_QCD_K = T_QCD_GEV / M_PL_GEV * T_PLANCK_K
T_QCD_PLANCK = T_QCD_GEV / M_PL_GEV

# BBN onset
T_BBN_GEV = 1e-4  # T ~ 0.1 MeV at BBN start
T_BBN_PLANCK = T_BBN_GEV / M_PL_GEV

# Cl(6) Fock dimension per vertex
CL6_FOCK_DIM = 2 ** 6  # = 64 states per vertex


# ============================================================================
# § 1. D3 derivation — assumptions catalog
# ============================================================================
print("=" * 80)
print(" Path D probe (D.4) — Cascade theorem D3 sub-T_srs epoch audit")
print("=" * 80)
print()
print("CASCADE THEOREM (from predictions/N_hub.py):")
print()
print("  D1 [A1, Type 1]: Each of k*N directed edges toggled once per t_P.")
print("       Each toggle modifies 1/(k*N) of causal structure.")
print("       → time mapping: 1 t_P = k*N toggle events.")
print()
print("  D2 [A2, Type 1+2]: MDL surprise threshold θ* = log₂(k*) = log₂(3).")
print("       Acceptance probability per toggle: 2^{-θ*} = 1/k*.")
print()
print("  D3 [algebra, Type 2]: Cascade ratio ε = 1/(k*N).")
print("       New states per t_P: k*N · ε = 1.")
print("       H = (1 / t_P) / N = 1/(N · t_P).")
print()
print("  Result: H · t_P · N = 1 exactly, FOR T < T_srs (per 2026-05-09 audit).")
print()
print(f"  Substrate-internal scales (Planck units):")
print(f"    T_srs           ≈ {T_SRS_PLANCK:.3e}  (= 101 Planck units, ~10³⁴ K)")
print(f"    T_EW = v_higgs  ≈ {T_EW_PLANCK:.3e}  (= 2×10⁻¹⁷, ~10¹⁵ K)")
print(f"    T_QCD           ≈ {T_QCD_PLANCK:.3e}  (= 1.6×10⁻²⁰, ~10¹² K)")
print(f"    T_BBN           ≈ {T_BBN_PLANCK:.3e}  (= 8×10⁻²⁴, ~10⁹ K)")
print(f"    Cl(6) Fock dim per vertex = {CL6_FOCK_DIM}")
print()


# ============================================================================
# § 2. Assumption audit
# ============================================================================
print("-" * 80)
print("§2. D1/D2/D3 assumption audit beyond T < T_srs")
print("-" * 80)
print()

assumptions = [
    {
        "step": "D1",
        "assumption": "Each substrate vertex has k* = 3 directed edges",
        "epoch_dependence": "NONE — k* is structural, Row 4 Brown-rank theorem-grade",
        "fails_at": "N/A (always k* = 3 for T < T_srs)",
    },
    {
        "step": "D1",
        "assumption": "Toggle rate is k*N per t_P (each edge toggled once per t_P)",
        "epoch_dependence": "Possible: if some edges are dormant at low T, toggle rate < k*N",
        "fails_at": "No substrate-internal scale identified where edges become dormant",
    },
    {
        "step": "D1",
        "assumption": "Each toggle modifies exactly 1/(k*N) of causal structure",
        "epoch_dependence": "NONE — pure combinatorial counting; structural at all T < T_srs",
        "fails_at": "N/A",
    },
    {
        "step": "D2",
        "assumption": "MDL surprise threshold θ* = log₂(k*) = log₂(3)",
        "epoch_dependence": "Possible: at very low N (Planck-era), the Bayesian posterior hasn't converged. At cosmological N >> 1, converged.",
        "fails_at": "N ~ 1 (Planck epoch), irrelevant at recombination N ~ 10⁶¹",
    },
    {
        "step": "D2",
        "assumption": "Acceptance probability = 2^{-θ*} = 1/k* uniformly per toggle",
        "epoch_dependence": "Possible if MDL waterline shifts with epoch.  Per Stage-2c + S_fresh + S_disconfirm, the waterline depends on k* (constant) and thermal noise (negligible per probe D.1 verdict).",
        "fails_at": "Confirmed negligible variation across cosmological epochs (probe D.1 verdict)",
    },
    {
        "step": "D3",
        "assumption": "Each accepted toggle adds 1 new distinguishable microstate",
        "epoch_dependence": "Possible: if Cl(6) Fock species thermalize, each accepted toggle could add Cl(6)-multiplicity new substrate configurations",
        "fails_at": "Cl(6) species thresholds — see § 3 quantification below",
    },
    {
        "step": "D3",
        "assumption": "N grows linearly: N(t) = t/t_P",
        "epoch_dependence": "Follows from D1+D2; no additional assumption",
        "fails_at": "N/A (derived)",
    },
]

print(f"  {'#':<3} {'Step':<5} {'Assumption':<55} {'Fails at?':<30}")
print(f"  {'-'*3} {'-'*5} {'-'*55} {'-'*30}")
for i, a in enumerate(assumptions, 1):
    print(f"  {i:<3} {a['step']:<5} {a['assumption'][:55]:<55} {a['fails_at'][:30]:<30}")
print()


# ============================================================================
# § 3. Cl(6) Fock species thermal population check (most-promising candidate)
# ============================================================================
print("-" * 80)
print("§3. Cl(6) Fock species thermal population — quantitative check")
print("-" * 80)
print()
print("  Hypothesis: at T > T_threshold_n for some Fock state n, that state")
print("  is thermally populated and contributes to substrate microstate count.")
print()
print("  Cl(6) Fock has dim 2^6 = 64 states per vertex.  At T = 0, only the")
print("  vacuum is populated.  At T > E_n, Fock state n becomes thermally")
print("  populated (Boltzmann factor e^{-E_n/T} ~ O(1)).")
print()
print("  If each accepted toggle adds m(T) microstates (counting accessible Fock")
print("  states), then dN/dt = m(T)/t_P, and H = m(T)/(N · t_P) > coasting at high T.")
print()

# Fock energy levels (rough estimate: equally spaced from 0 to v_higgs)
print("  Rough Cl(6) Fock energy spectrum (vacuum → first excited at v_higgs):")
print(f"    E_0 = 0,   E_1 ~ v_higgs = {V_HIGGS_GEV} GeV")
print(f"    E_n = n · v_higgs for n = 1, ..., 63 (very rough)")
print()
print("  When does each state become thermally populated?")
print()
print(f"  {'n':<5} {'E_n (GeV)':<15} {'T_pop (K)':<15} {'z_pop':<15}")
print(f"  {'-'*5} {'-'*15} {'-'*15} {'-'*15}")

for n in [1, 2, 5, 10, 20, 63]:
    E_n_GeV = n * V_HIGGS_GEV
    T_pop_K = E_n_GeV / M_PL_GEV * T_PLANCK_K
    z_pop = T_pop_K / T_CMB_TODAY_K
    print(f"  {n:<5} {E_n_GeV:<15.1f} {T_pop_K:<15.3e} {z_pop:<15.3e}")

print()
print("  Earliest Fock-population epoch (n=1, T_pop = T_EW = 10¹⁵ K):")
T_EW_K_val = V_HIGGS_GEV / M_PL_GEV * T_PLANCK_K
z_EW = T_EW_K_val / T_CMB_TODAY_K
print(f"    z_EW ≈ {z_EW:.3e}")
print(f"    Way above recombination (z ~ 10³) and matter-rad equality (z ~ 3400).")
print(f"    Also above BBN (z ~ 4×10⁸).")
print()
print("  Quantitative consequence: between z_BBN ~ 4×10⁸ and z_EW ~ 4×10¹²,")
print("  the first excited Fock state is thermally populated (Boltzmann factor 1).")
print("  At z << z_EW, only vacuum populated; at z >> z_EW, more Fock states.")
print()
print("  IF D3's 'one microstate per accepted toggle' implicitly counts vacuum")
print("  only, then between z_BBN and z_EW (and especially above z_EW),")
print("  effective m(T) > 1 and H(z) > coasting.")
print()

# Compute m(T) Boltzmann factor sum
def cl6_fock_count_at_T(T_planck: float, max_n: int = 63) -> float:
    """Estimate effective number of accessible Cl(6) Fock states at T (Planck units)."""
    if T_planck <= 0:
        return 1.0
    # Boltzmann factor for state n: e^{-E_n/T} = e^{-n·v_h/T}
    # For T >> v_h, all states accessible; for T << v_h, only vacuum
    total = 1.0  # vacuum
    for n in range(1, max_n + 1):
        E_n = n * T_EW_PLANCK
        bf = math.exp(-E_n / T_planck) if T_planck > 1e-30 else 0.0
        total += bf
    return total


# Compute m(T) at observable epochs
print(f"  {'Epoch':<28} {'z':<12} {'T_CMB (K)':<14} {'T (Planck)':<14} {'m(T)':<12}")
print(f"  {'-'*28} {'-'*12} {'-'*14} {'-'*14} {'-'*12}")
EPOCHS = [
    ("Today (z=0)",           0.0),
    ("Recombination",         1089.0),
    ("Matter-rad equality",   3400.0),
    ("BBN",                   3.9e8),
    ("QCD",                   7e11),
    ("Electroweak (T_EW)",    z_EW),
    ("T = 10 × T_EW",         10 * z_EW),
    ("T = 100 × T_EW",        100 * z_EW),
]
for name, z in EPOCHS:
    T_K = T_CMB_TODAY_K * (1.0 + z)
    T_Planck = T_K / T_PLANCK_K
    m = cl6_fock_count_at_T(T_Planck)
    print(f"  {name:<28} {z:<12.3e} {T_K:<14.3e} {T_Planck:<14.3e} {m:<12.6f}")

print()
print("  Verdict for § 3:")
print("    At observationally-relevant z (BBN through recombination), T_CMB << v_higgs,")
print("    so Cl(6) Fock states beyond vacuum are NOT thermally populated.  m(T) ≈ 1.")
print("    The Cl(6)-species-thermal-population mechanism doesn't activate in the")
print("    relevant z range.")
print()


# ============================================================================
# § 4. What if D3 has structure at lower scales (T_BBN, T_QCD)?
# ============================================================================
print("-" * 80)
print("§4. Lower-scale structure — would T_BBN or T_QCD work?")
print("-" * 80)
print()
print("  T_QCD ≈ 200 MeV ≈ 2×10¹² K corresponds to z_QCD ~ 7×10¹¹.")
print("  T_BBN ≈ 0.1 MeV ≈ 10⁹ K corresponds to z_BBN ~ 4×10⁸.")
print()
print("  Both are ABOVE recombination (z = 1089) — so a D3 modification at")
print("  T > T_BBN would give a sub-coasting H(z) at z > 4×10⁸ but NOT at")
print("  recombination.")
print()
print("  The CMB θ_* sensitivity is dominated by the integral from BBN to")
print("  recombination — not from above BBN.  So even if D3 changes at T > T_BBN,")
print("  the r_s integral from z=BBN downwards is the dominant contribution and")
print("  remains coasting-shaped (since z << z_BBN until recombination).")
print()
print("  Conclusion: D3 modification at T_BBN or higher doesn't help r_s")
print("  regulation — wrong epoch.")
print()
print("  Specifically, what would help r_s:")
print("    r_s = ∫ c_s/H · da/a from a_BBN to a_recomb")
print("    If H(z) > coasting in z = 1089 to z ~ 10⁵ range, r_s shrinks.")
print("    But D3's natural candidates (Cl(6) Fock at T_EW, or below T_QCD/T_BBN)")
print("    don't activate in this range — too low to populate Fock states,")
print("    too high to be the structural-transition scales.")
print()


# ============================================================================
# § 5. Verdict
# ============================================================================
print("=" * 80)
print("VERDICT — Path D probe (D.4)")
print("=" * 80)
print()
print("  D3's 'one microstate per accepted toggle' has TWO candidate sub-T_srs")
print("  modification mechanisms:")
print()
print("  (a) Cl(6) Fock species thermal population (T ≈ v_higgs ≈ 10¹⁵ K).")
print("      This is the most-natural framework-internal scale.  But T_EW corresponds")
print("      to z ~ 10¹² — WAY ABOVE the recombination-to-z_eq range that matters")
print("      for r_s.  In the relevant range (z ~ 10³ to 10⁵), only vacuum is")
print("      populated, m(T) ≈ 1, no D3 modification.")
print()
print("  (b) Low-T transitions (T_QCD ~ 10¹² K, T_BBN ~ 10⁹ K).")
print("      These are observationally-relevant for SM physics but they're")
print("      MATTER-CONTENT transitions, not substrate-structural transitions.")
print("      The framework's substrate (k*=3 srs Cayley graph) doesn't change")
print("      at T_QCD or T_BBN — only the SM particle content does.  D3 is")
print("      substrate-combinatorial, not SM-particle-content-dependent.")
print()
print("  Neither mechanism gives a D3 modification in the recombination-to-z_eq")
print("  range where r_s integration dominates the CMB θ_* constraint.")
print()
print("  STRUCTURAL CONCLUSION: D3 does NOT have a sub-T_srs epoch restriction")
print("  that activates in the r_s-relevant range.  The cascade theorem's H(z) =")
print("  H_0(1+z) coasting prediction holds throughout the recombination-to-z_eq")
print("  range, giving θ_* ~ 0.05 rad vs Planck 0.01 rad (factor ~5 mismatch).")
print()
print("  Path D probe (D.4) is CLOSED-NEGATIVE.")
print()
print("  PATH D DECISION TREE STATUS:")
print("    (D.1) CLOSED-NEGATIVE — thermal MDL acceptance suppression negligible")
print("    (D.4) CLOSED-NEGATIVE — D3 has no sub-T_srs structural transition (this probe)")
print("    (D.5) sound-speed c_s(z) modification — REMAINING BOUNDED PROBE")
print("    (D.2)/(D.3) — blocked by Need A of MS.1 (multiway formalization)")
print()
print("  NEXT BOUNDED PROBE: (D.5) — test whether substrate-derived c_s(z) gives")
print("  r_s ~ 150 Mpc under coasting H(z) without H(z) modification.")
print()
print("  HONEST ASSESSMENT:")
print("    Two of three bounded Path D probes now CLOSED-NEGATIVE.  The framework's")
print("    cascade theorem D1+D2+D3 is structurally uniform across the cosmological")
print("    epochs that matter for r_s — no thermal effects, no substrate phase")
print("    transitions, no D3-modification mechanisms in the relevant range.")
print()
print("    If (D.5) also closes negative, Path D becomes research-level only via")
print("    (D.2)/(D.3), pending Need A of MS.1 unblock.  At that point Item 5")
print("    is honestly a multi-session structural research direction with no")
print("    near-term bounded closure path identified.")
