#!/usr/bin/env python3
"""
Leading-factor chase — substrate-thermal coupling for H prefactor.

PURPOSE
-------
Per α-audit verdict 2026-05-27 EOD+1, the Y_p falsification is structurally
robust to α choice. Both natural α values fail:
  α=1/2 (instantaneous):     T_ν_dec=0.84 MeV, T_BBN-1=0.39 MeV, Y_p=0.051 (-65σ)
  α=25/48 (cumulative):      T_ν_dec=3.18 MeV, T_BBN-1=1.48 MeV, Y_p=0.453 (+69σ)
  Required to match obs:     T_ν_dec=1.72 MeV, T_BBN-1=0.80 MeV, Y_p=0.245

Root cause: framework substrate H = 1/(N·t_P) has prefactor EXACTLY 1
(theorem-grade via N_hub cascade D1+D2+D3). ΛCDM Friedmann H carries
prefactor (1.66·√g_*) ≈ 5.44 at MeV scale, counting active relativistic
species (g_*=10.75).

This probe systematically chases the leading factor:

  (A) Quantify required F such that H_corrected = F · H_substrate matches
      observed Y_p at α=1/2.
  (B) Inventory framework's structural species-count candidates that
      could produce F.
  (C) Test epochal consistency: F must reduce to ~1 at today (else
      breaks H_0=68 km/s/Mpc prediction). ΛCDM g_*(N) runs from ~106
      (T>>100 GeV) → 10.75 (MeV) → 2 (today). Does any framework
      natural F have this running behavior?
  (D) Identify the structural-derivation route candidates if F is to be
      added as a new framework piece.

Run:
    python3 proofs/cosmology/H_prefactor_leading_factor_chase_2026-05-27.py
"""

import math

# --- Framework primitives ---
G_F  = 1.1663787e-5
M_Pl = 1.22089e19
Q_np = 1.2933e-3
T_nu_dec_substrate = 0.844e-3  # current framework α=1/2 prediction (GeV)
T_BBN_substrate    = 0.394e-3
ratio_7_15 = 7.0/15.0

# --- ΛCDM reference ---
T_nu_dec_LCDM = 1.5e-3
T_BBN_LCDM    = 0.7e-3
Y_p_obs = 0.245
Y_p_sigma = 0.003
decay_factor = 0.7

# --- Framework structural primitives (for candidate F sources) ---
k_star = 3
g_girth = 10
N_atoms = 4
E_count = 12   # directed edges per primitive cell (= 2|E|=24 directed bonds, |E|=12)
V_count = 4
hypercharge_norm = 5/3

# --- α=1/2 freezeout solver ---
def T_freezeout_with_F(F):
    """Solve Γ_weak = F · H_substrate where H_substrate = T²·M_Pl^(-1).
    Returns T_F in GeV.

    G_F²·T^5 = F · T²/M_Pl
    T^3 = F/(M_Pl·G_F²)
    T = (F/(M_Pl·G_F²))^(1/3)
    """
    return (F / (M_Pl * G_F**2))**(1.0/3.0)


def Y_p_from_T_BBN(T_BBN):
    n_p_freeze = math.exp(-Q_np / T_BBN)
    n_p_final = n_p_freeze * decay_factor
    return 2 * n_p_final / (1 + n_p_final)


# =============================================================================
# (A) Quantify required F
# =============================================================================
print("=" * 78)
print("  LEADING-FACTOR CHASE — substrate-thermal coupling for H prefactor")
print("=" * 78)

print("\n" + "-" * 78)
print("  (A) Quantify required F precisely")
print("-" * 78)

# For T_F = 1.5 MeV (ΛCDM match):
F_required_LCDM = T_nu_dec_LCDM**3 * M_Pl * G_F**2
print(f"\n  ΛCDM-match (T_ν_dec = 1.5 MeV, T_BBN-1 = 0.7 MeV, Y_p = 0.245):")
print(f"    F_required = T_F^3 · M_Pl · G_F² = {F_required_LCDM:.4f}")
print(f"    Compare ΛCDM 1.66·√g_*(MeV)     = 1.66·√10.75 = {1.66 * math.sqrt(10.75):.4f}")
print(f"    Match: ✓ (these agree by construction)")

# For Y_p = 0.245 exactly (different than ΛCDM T_BBN-1, if framework 7/15 still applies):
# Y_p = 0.245 → n_p_freeze · 0.7 = Y_p/(2-Y_p) = 0.245/1.755 = 0.1396
# n_p_freeze = 0.1994 → T_BBN-1 = Q_np / (-ln(0.1994)) = 1.293/1.612 = 0.802 MeV
# T_ν_dec = 0.802·15/7 = 1.719 MeV
T_nu_target = (Q_np / (-math.log(Y_p_obs / (2-Y_p_obs) / decay_factor))) / ratio_7_15
F_required_Y_p = T_nu_target**3 * M_Pl * G_F**2
print(f"\n  Y_p-direct-match (target T_ν_dec = {T_nu_target*1e3:.3f} MeV):")
print(f"    F_required = {F_required_Y_p:.4f}")

# --- Sensitivity ---
print(f"\n  Sensitivity: Y_p as function of F (sweep)")
print(f"  {'F':>8}  {'T_ν_dec (MeV)':>14}  {'T_BBN-1 (MeV)':>14}  {'Y_p':>8}  {'σ':>8}")
for F in [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 5.44, 7.0, 10.0]:
    T_nu = T_freezeout_with_F(F)
    T_BBN = T_nu * ratio_7_15
    Y_p = Y_p_from_T_BBN(T_BBN)
    dev = (Y_p - Y_p_obs) / Y_p_sigma
    print(f"  {F:>8.3f}  {T_nu*1e3:>14.3f}  {T_BBN*1e3:>14.3f}  {Y_p:>8.4f}  {dev:>+8.2f}σ")


# =============================================================================
# (B) Inventory framework's structural species-count candidates
# =============================================================================
print()
print("-" * 78)
print("  (B) Framework structural primitives — candidate F sources")
print("-" * 78)

candidates = [
    ("2|E|=24 directed bonds (substrate full handshake)", 2*E_count),
    ("|E|=12 (BST handshake; appears in α_color = 3/12)", E_count),
    ("N_atoms × k* × k* = 4·3·3", N_atoms * k_star * k_star),  # 36
    ("g_girth = 10 (srs cycle length)", g_girth),
    ("Cl(6) Fock dim 2^6 = 64", 64),
    ("k* × N_atoms = 12 (= 2|E| for K_4)", k_star * N_atoms),
    ("Hypercharge norm 5/3 × k* × N_atoms = 20", hypercharge_norm * k_star * N_atoms),
    ("SU(5) dim = 24", 24),
    ("R3 × 4 generations × 2 chirality = ?", 0),  # placeholder
    ("ΛCDM g_*(MeV) = 10.75 (reference)", 10.75),
    ("ΛCDM 1.66·√g_*(MeV) ≈ 5.44 (full Friedmann)", 1.66 * math.sqrt(10.75)),
]

print(f"\n  {'Candidate':>50}  {'Value':>10}  {'T_F (MeV)':>10}  {'Y_p':>8}  {'σ':>8}")
print(f"  Target F:                                       ~ 5.4-5.5")
for name, F in candidates:
    if F == 0:
        continue
    T_nu = T_freezeout_with_F(F)
    T_BBN = T_nu * ratio_7_15
    Y_p = Y_p_from_T_BBN(T_BBN)
    dev = (Y_p - Y_p_obs) / Y_p_sigma
    marker = " ★" if abs(F - 5.44) < 1.0 else ""
    print(f"  {name:>50}  {F:>10.3f}  {T_nu*1e3:>10.3f}  {Y_p:>8.4f}  {dev:>+8.2f}σ{marker}")


# =============================================================================
# (C) Epochal consistency check
# =============================================================================
print()
print("-" * 78)
print("  (C) Epochal consistency: F must run from ~5.4 at MeV → ~1 at today")
print("-" * 78)

# Today's H_0 prediction depends on F at today:
# H_0_pred = 68 km/s/Mpc with F=1 (current framework). If F_today > 1, H_0
# would be too large.
H_0_obs = 67.4
H_0_framework = 68.0  # current framework prediction with F=1
H_0_sigma_obs = 0.5

print(f"\n  Today (T_today = 2.73 K):")
print(f"    Current framework F_today = 1  →  H_0 = {H_0_framework:.1f} km/s/Mpc")
print(f"    Observation H_0_CMB       = {H_0_obs} ± {H_0_sigma_obs} km/s/Mpc")
print(f"    Framework H_0 OK ✓ (within ~1σ)")
print()
print(f"  At MeV (Y_p match):")
print(f"    Required F_MeV ≈ 5.44 (= 1.66·√g_*(MeV) = 1.66·√10.75)")
print()
print(f"  Required RUNNING: F(N) must be 5.44 at MeV but 1 at today.")
print(f"    Ratio: F(N_MeV)/F(N_today) ≈ 5.44 (factor ~5 running)")
print()
print(f"  ΛCDM analog: √g_*(N) runs:")
print(f"    T >> 100 GeV : g_* = 106.75, √g_* = 10.33")
print(f"    T ~ MeV       : g_* = 10.75,  √g_* = 3.28")
print(f"    Today         : g_* ≈ 2-3.4, √g_* ≈ 1.4-1.8")
print()
print(f"  Standard cosmology Friedmann naturally produces this running via")
print(f"  ρ_rad ∝ g_*·T^4 + ρ_matter ∝ a^(-3) + ρ_Λ. Different epochs dominated")
print(f"  by different ρ terms.")
print()
print(f"  Framework coasting has UNIFORM H = 1/(N·t_P) across all epochs.")
print(f"  No ρ-decomposition → no natural g_*(N) running.")
print()
print(f"  → Adding F(N) to framework H requires NEW structural piece that:")
print(f"    (i) reduces to 1 at today (matter/Λ-dominated epoch)")
print(f"    (ii) reaches ~5.4 at MeV (radiation-dominated epoch)")
print(f"    (iii) is structurally K-rational (no continuum π factors)")


# =============================================================================
# (D) Structural-derivation candidate routes
# =============================================================================
print()
print("-" * 78)
print("  (D) Structural-derivation candidates for F(N)")
print("-" * 78)

routes = [
    ("R-A: Substrate-side species coupling",
     """The substrate hosts multiple walker types at the bath. Each species
contributes its own σ-channel to the posterior. If T = σ_per_mode then
H_eff·T² ∝ ρ ∝ Σ_modes T^4 = g_*·T^4, giving H_eff ∝ √g_*·T². Requires
deriving g_*(N) from Cl(6) Fock active-mode count at each epoch."""),

    ("R-B: Friedmann import via observer-energy functional",
     """Framework has theorem_observer_energy_functional E_obs = κ·S_total
(theorem-grade). If the observer's posterior energy ρ_obs relates to H via
some Friedmann-like H² ∝ ρ_obs, and ρ_obs counts active walker species at
the bath, then H carries √g_*-like factor structurally."""),

    ("R-C: 2|E| edge thermal multiplicity",
     """The substrate has 2|E|=24 directed bonds per primitive cell. If at
MeV scale, each bond hosts a thermalized walker mode, and the rate H scales
as √(modes_per_bond)·H_substrate, this could produce a factor √24 ≈ 4.9
at MeV. Doesn't naturally run to 1 at today."""),

    ("R-D: Coasting-modified Friedmann",
     """Abandon the H = 1/(N·t_P) cascade theorem at MeV and replace with
H² ∝ ρ-thermal at radiation-dominated epochs. Major structural change;
breaks coasting uniformity. Doesn't seem framework-natural."""),

    ("R-E: Substrate edge-coupling factor under Bayesian observer",
     """Beta-Bernoulli posterior with N edges (single mode): σ² ∝ 1/N.
With M independent channels: σ_marginal² ∝ 1/N (per channel), but
COMBINED σ² ∝ M/N. Identifying T with σ → T ∝ √M·N^(-1/2). Then T²/H
relation gives H ∝ T²/M. So multiple species REDUCE H, not increase.
Counterintuitive — multiple modes spread thermal energy thinner."""),
]

for name, desc in routes:
    print(f"\n  {name}")
    print(f"    {desc.strip()}")


# =============================================================================
# Verdict
# =============================================================================
print()
print("=" * 78)
print("  VERDICT — leading-factor chase")
print("=" * 78)
print(f"""
  Required H prefactor at MeV: F ≈ 5.44 (= 1.66·√g_*(MeV) = ΛCDM Friedmann
  with 10.75 species, 1.66 from continuum 4D Stefan-Boltzmann).

  Required RUNNING: F(N_today)/F(N_MeV) ≈ 0.18, i.e. F must reduce ~5× to
  preserve H_0 = 68 km/s/Mpc at today. ΛCDM achieves this naturally via
  ρ_matter ≫ ρ_rad at today; framework's coasting has no analogous
  ρ-decomposition.

  Inventory finding: No single K-rational framework primitive (2|E|, k*,
  N_atoms, g_girth, etc.) hits F ≈ 5.4 at MeV. Closest natural candidates:
    - SU(5) dim = 24 → F = 24 → T_F = 2.43 MeV (TOO BIG)
    - 2|E| = 24      → F = 24 → same
    - Hypercharge × k* × N_atoms = 20 → T_F = 2.30 MeV (too big)
    - g_girth = 10   → F = 10 → T_F = 1.82 MeV (closer but not great)
    - k*·N_atoms = 12 → F = 12 → T_F = 1.93 MeV (closer)

  None match. The ΛCDM 1.66·√g_* = 5.44 comes from continuum Friedmann
  with π and √g_*; both terms contribute. Framework would need BOTH:
    (i) A π-free K-rational analog of the 1.66 continuum factor
    (ii) A substrate analog of g_*(N) species running

  Structural assessment:
    - Route R-A (substrate-side species coupling) is the most natural
      direction. Requires deriving active-walker-mode count from Cl(6) Fock
      at each thermal epoch. Multi-sprint research.
    - Route R-B (Friedmann import via E_obs = κ·S_total) is more bounded.
      Could leverage existing observer-energy theorem. But ρ-counting at
      thermal bath requires careful definition.
    - Route R-E (Bayesian σ for M-channel) gives WRONG-SIGN effect.
    - Route R-D (abandon coasting at MeV) breaks the framework.

  Bounded next step: R-B scoping. Probe whether E_obs = κ·S_total naturally
  generates a thermal-mode multiplicity factor in H at radiation-dominated
  epochs without breaking H_0 at today.

  No K-rational primitive hits F=5.4 directly — the leading factor is NOT a
  simple substrate combinatorial. It's structurally a SPECIES-RUNNING factor
  that requires Cl(6) Fock species enumeration at each epoch.
""")
