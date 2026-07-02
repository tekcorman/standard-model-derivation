#!/usr/bin/env python3
"""
F-deactivation mechanism probe — closing Gate 2 for F = √(k_star · g_*).

PURPOSE
-------
The √(k_star·g_*) candidate matches ΛCDM 1.66·√g_* across two independent
epochs (BBN + post-e+e- annihilation) with consistent +4.3% offset, but
applied uniformly it FAILS at today: gives H_0 = 216 km/s/Mpc instead of
observed 67.4.

ΛCDM has automatic deactivation via ρ-decomposition: at late times ρ_Λ + ρ_m
≫ ρ_rad, so the √g_*·T² term doesn't dominate H². Framework's uniform
coasting H = 1/(N·t_P) lacks this structure.

This probe explores candidate F-deactivation mechanisms:

  MECH-1: Additive Friedmann-style: H² = H_substrate² + H_rad² where
          H_rad = √(k·g_*)·T²/M_Pl. Question: does H_rad naturally
          become negligible at low z?

  MECH-2: Two-regime T(N): α=1/2 (radiation regime, instantaneous);
          α=25/48 (cumulative regime, late). Transition at some N_eq.

  MECH-3: Active-species density threshold: F engages only when
          ρ_rad > threshold (e.g. ρ_rad > ρ_substrate or ρ_rad > ρ_Λ).

  MECH-4: Substrate-coupling fraction f(N): F → F · f(N) where f(N)
          encodes "fraction of substrate states thermally coupled to
          relativistic bath." Decays as N grows.

For each, check: (i) gives correct H at MeV; (ii) gives correct H_0 today;
(iii) is framework-derivable from existing primitives.

Run:
    python3 proofs/cosmology/F_deactivation_mechanism_probe_2026-05-27.py
"""

import math

# Constants
k_star = 3
g_girth = 10
M_Pl = 1.22089e19
N_hub = 8.394881e60
G_F = 1.1663787e-5
t_P_seconds = 5.391247e-44

# Observed H_0
H_0_obs = 67.4  # km/s/Mpc
H_0_substrate_pred = 68.0  # framework prediction with F = 1 at today

# Compute H_substrate at any T under α=1/2 in natural units (GeV)
def H_substrate_alpha_half(T_GeV):
    """Under α=1/2: T = M_Pl·N^(-1/2), so N = (M_Pl/T)², H = M_Pl/N = T²/M_Pl."""
    return T_GeV**2 / M_Pl

def H_substrate_alpha_25_48(T_GeV):
    """Under α=25/48: H = T^(48/25) · M_Pl^(-23/25)."""
    return T_GeV**(48/25) * M_Pl**(-23/25)


# Active species (ΛCDM g_*(T) accounting)
def g_star_active(T_MeV):
    if T_MeV > 0.5:
        return 10.75
    elif T_MeV > 0.05:
        return 3.36
    else:
        return 3.36  # neutrinos + photons in radiation, but matter/Λ dominates total ρ


# H_radiation under K-rational √(k·g_*) form
def H_rad_K_rational(T_GeV):
    g = g_star_active(T_GeV * 1e3)
    return math.sqrt(k_star * g) * T_GeV**2 / M_Pl


print("=" * 78)
print("  F-DEACTIVATION MECHANISM PROBE")
print("=" * 78)


# =============================================================================
# MECH-1: Additive Friedmann-style
# =============================================================================
print()
print("-" * 78)
print("  MECH-1: H² = H_substrate² + H_rad²  (additive Friedmann-style)")
print("-" * 78)

epochs = [
    ("BBN ν decoupling (T_ν_dec)", 1.5e-3),       # GeV
    ("BBN weak freeze-out", 0.7e-3),
    ("e+e- annihilation (m_e/3)", 0.17e-3),
    ("CMB recombination", 0.32e-9),
    ("Today (T_CMB = 2.73 K)", 2.349e-13),
]

print(f"\n  {'Epoch':<40}  {'T (GeV)':>10}  {'H_sub':>10}  {'H_rad':>10}  {'ratio':>10}")
for name, T_GeV in epochs:
    H_sub = H_substrate_alpha_half(T_GeV)
    H_rad = H_rad_K_rational(T_GeV)
    ratio = H_rad / H_sub
    print(f"  {name:<40}  {T_GeV:>10.3e}  {H_sub:>10.3e}  {H_rad:>10.3e}  {ratio:>10.4f}")

print(f"""
  Critical insight under α=1/2:
    T² ∝ N^(-1) (since T = M_Pl·N^(-1/2)) → T²/M_Pl = M_Pl/N = H_sub.
    Therefore H_rad/H_sub = √(k_star · g_*) — INDEPENDENT of N.

    The ratio is CONSTANT across all epochs:
      MeV epoch:  H_rad/H_sub = √(3·10.75) = √32.25 = {math.sqrt(3*10.75):.3f}
      Today:      H_rad/H_sub = √(3·3.36)  = √10.08 = {math.sqrt(3*3.36):.3f}

  → MECH-1 under α=1/2 gives CONSTANT multiplicative factor, NO automatic
    deactivation. Same problem as before. FAILS Gate 2.

  Verdict: additive Friedmann-style with √(k·g_*)·T²/M_Pl form does NOT
  produce automatic deactivation under α=1/2 instantaneous T-N scaling.
""")


# =============================================================================
# MECH-2: Two-regime T(N) crossover
# =============================================================================
print()
print("-" * 78)
print("  MECH-2: Two-regime α=1/2 (radiation) ↔ α=25/48 (late) crossover")
print("-" * 78)

print(f"""
  Hypothesis: framework uses α=1/2 when radiation dominant ρ_rad ≫ other,
  α=25/48 when other components dominate. Crossover at some N_eq.

  Under α=25/48: T ∝ N^(-25/48), H = 1/(N·t_P) ∝ T^(48/25).
  At today, this gives T_today = 2.66 K matching observation ✓.

  Problem: when is the crossover? In ΛCDM, matter-radiation equality at
  z_eq ≈ 3400. Framework REJECTS matter-radiation equality structurally
  (per cosmic_history_bounded_sweep §2 — substrate has no ρ_m / ρ_rad
  decomposition).

  ⇒ No structural mechanism for an α crossover in current framework.
  Adding one = Axiom-class framework extension (recover an analog of
  rad-matter equality WITHOUT actual rad-matter species decomposition).

  Verdict: structurally INCOMPATIBLE with framework's coasting + Ω_Λ=1/3
  uniformity. Would require major restructuring.
""")


# =============================================================================
# MECH-3: Active-species density threshold
# =============================================================================
print()
print("-" * 78)
print("  MECH-3: F engages only when ρ_rad > ρ_substrate threshold")
print("-" * 78)

# Compute ρ_rad / ρ_substrate ratio at each epoch under α=1/2
# ρ_rad = g_*·T^4 (without π factors, K-rational)
# ρ_substrate analog: derive from H_substrate via Friedmann-like H² ∝ ρ
# H_sub² = M_Pl⁻²·T⁴ ⇒ ρ_substrate ∝ T⁴ as well (under α=1/2)
# Constant ratio under α=1/2 again.

# Try alternative: ρ_substrate = M_Pl²·H_sub² × (1) = M_Pl²·T⁴/M_Pl² = T⁴
# Then ρ_rad/ρ_sub = k_star·g_*(T) — depends only on g_*(T), not on T directly.

print(f"""
  Under α=1/2, ρ_substrate ∝ T⁴ (since H_sub ∝ T²), so ρ_rad/ρ_substrate
  is independent of T. ρ_rad/ρ_substrate = k_star · g_*(T):
    MeV:   k·g_* = 32.25
    Today: k·g_* = 10.08  (with g_*_rad = 3.36 at today)

  Even at today, k·g_*_rad > 1. So under this scheme, radiation is
  "always dominant" — but this contradicts ΛCDM observations where ρ_rad
  is ~5×10⁻⁵ of total ρ at today.

  Issue: g_*_rad at today is 3.36 only if we count photons + neutrinos as
  relativistic. But their energy density is tiny because T_today is tiny
  in absolute units. ρ_rad ∝ T⁴ · g_* — and T⁴ is the suppressor.

  Under framework α=1/2, T⁴ scaling matches H_sub² ∝ T⁴ exactly. So the
  ratio is constant. NO deactivation.

  Verdict: under α=1/2, no T-dependent ρ-threshold provides deactivation.

  IF we use α=25/48 for T(N) and α=1/2 for H_sub-from-rate-balance, the
  scalings differ and ρ_rad/ρ_sub becomes N-dependent. Let's check:
""")

# Under α=25/48: T = M_Pl·N^(-25/48). T^4 = M_Pl^4·N^(-100/48) = M_Pl^4·N^(-25/12).
# H_sub = M_Pl/N. ρ_sub ∝ H_sub² ∝ N^(-2).
# ρ_rad ∝ T^4 ∝ N^(-25/12).
# ρ_rad/ρ_sub ∝ N^(-25/12+2) = N^(-1/12) = decreases with N ✓

# Compute at each epoch
print(f"  Hybrid: T(N) ∝ N^(-25/48), H_sub ∝ N^(-1):")
print(f"  {'Epoch':<30}  {'N (under α=25/48)':>22}  {'ρ_rad/ρ_sub':>14}")
for name, T_GeV in epochs:
    N_25_48 = (M_Pl / T_GeV)**(48/25)
    ratio = k_star * g_star_active(T_GeV * 1e3) * N_25_48**(-1/12)
    print(f"  {name:<30}  {N_25_48:>22.3e}  {ratio:>14.4e}")

print(f"""
  Under hybrid α-mixing (T uses α=25/48, H_sub from cascade theorem):
    ρ_rad/ρ_sub ∝ g_*·N^(-1/12) → decreases with N
    At today (N=N_hub): ratio = k_star·g_*_rad·N_hub^(-1/12)
                              = 3·3.36·{N_hub**(-1/12):.4e}
                              = {3*3.36*N_hub**(-1/12):.4e}
    At MeV (N≈10^41):   ratio = k_star·g_*_MeV·N_MeV^(-1/12)
                              ≈ 32.25·10^(-1/12·41) ≈ {32.25*10**(-41/12):.4e}

  Hmm: ratio at today (≈ 9e-5) < ratio at MeV (≈ 0.012). BOTH small.
  This says ρ_rad is ALWAYS subdominant under α=25/48-for-T + α=1-for-H_sub.
  No epoch where radiation dominates. Doesn't match BBN physics.

  Verdict: hybrid α-mixing doesn't produce the right structure either.
""")


# =============================================================================
# MECH-4: Substrate-coupling fraction f(N)
# =============================================================================
print()
print("-" * 78)
print("  MECH-4: Substrate-coupling fraction f(N) → 0 at late N")
print("-" * 78)

print(f"""
  Hypothesis: F = √(k_star · g_*) · f(N) where f(N) is "fraction of
  substrate states thermally coupled to relativistic bath." At early N,
  f(N) ≈ 1 (high temperature; bath dense; coupling strong). At late N,
  f(N) → 0 (dilute bath; weak coupling).

  Candidate functional form (substrate-natural):
    f(N) = 1/(1 + N/N_critical)  (logistic)
    f(N) = exp(-N/N_critical)    (exponential)
    f(N) = (T/T_critical)^p      (power-law)

  Required behavior:
    f(N=N_MeV) ≈ 1 (MeV; coupling on; T_MeV ≈ 1 MeV)
    f(N=N_hub) ≈ 0 (today; coupling off; T_today ≈ 0.2 meV)

  Per framework, what N_critical? Some structural epoch — perhaps when
  the bath energy density falls below some substrate-derivable threshold:
    - ρ_rad ≈ ρ_Λ (framework's structural Ω_Λ = 1/3)?
    - T < some characteristic substrate energy?
    - The "matter-radiation equality" analog under coasting (which the
      framework structurally rejects)?

  Verdict: f(N) introduces a NEW substrate-coupling parameter. Without a
  framework-derivable form for f(N), this is just CURVE FITTING. To be
  candidate-grade per W58, the form needs structural derivation.

  Tentative: maybe f(N) connects to observer-energy functional
  theorem_observer_energy_functional.md (E_obs = κ·S_total). The
  bath-substrate coupling could be derived from observer's accumulated
  information rate. Multi-sprint research.
""")


# =============================================================================
# VERDICT — Gate 2 deactivation remains open
# =============================================================================
print()
print("=" * 78)
print("  VERDICT — F-deactivation mechanism remains an OPEN STRUCTURAL QUESTION")
print("=" * 78)
print(f"""
  Probed 4 mechanism candidates:

    MECH-1 (additive H² Friedmann-style under α=1/2): FAILS — gives
      constant multiplicative factor, no deactivation.
    MECH-2 (two-regime α crossover): INCOMPATIBLE with framework's
      coasting + Ω_Λ=1/3 uniformity. Requires axiom.
    MECH-3 (active-species density threshold): under α=1/2 constant
      ratio; under hybrid α-mixing, ρ_rad never dominates. Doesn't work.
    MECH-4 (f(N) coupling fraction): introduces new parameter without
      structural derivation. NOT framework-natural without further work.

  ⇒ None of the natural mechanisms produces the required behavior.

  Implication: the F = √(k_star · g_*) candidate's CANDIDATE-GRADE status
  (per W58, after the +4.3% offset match at two epochs) is preserved, but
  the FULL closure to theorem-grade requires either:

    (a) An Axiom-A framework extension producing F(N) running with proper
        deactivation at late times (multi-sprint structural research).
    (b) Acceptance that framework's H is uniform (no √(k·g_*) factor) and
        Y_p = 0.05 is the framework's HONEST PREDICTION (falsified by
        observation at -65σ).
    (c) A subtler mechanism — perhaps the framework's H_substrate ITSELF
        encodes effective species count via N_hub structure, in which
        case the √(k·g_*) factor would need to be ABSORBED into N_hub
        derivation. This would require revisiting the N_hub cascade theorem.

  HONEST FINAL POSTURE:
    - F=√(k_star·g_*) candidate: 4.3% match across 2 epochs is provocative.
    - W58 anti-numerology: candidate stays CANDIDATE-GRADE pending derivation.
    - Gate 2 deactivation: OPEN; no natural mechanism in current framework.
    - The Y_p falsification candidate (-65σ at α=1/2) STANDS as the
      framework's structural prediction unless a deactivation derivation
      closes Gate 2.

  Per an internal note Option F + A axioms:
    - Option F (log-transcendence acceptance): closes Phase III numerical,
      not Y_p.
    - Option A (substrate color-coupling): could in principle close
      Y_p via species-derived H correction. Multi-sprint.

  Next bounded direction: Option A scoping (substrate species coupling)
  with explicit consideration of how √(k_star·g_*) would emerge AND
  deactivate.
""")
