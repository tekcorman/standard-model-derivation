#!/usr/bin/env python3
"""
T_BBN-1 calibration recheck — α-choice audit + Y_p propagation.

PURPOSE
-------
Y_p falsification (+69σ) under current predictions/T_BBN_weak_freezeout.py is
driven by T_BBN-1 = 1.48 MeV, which in turn comes from T_ν_dec = 3.18 MeV
computed with α = 25/48 in predictions/T_nu_dec.py.

Per unified_observation_process_reframe_2026-05-25.md, the framework has TWO
distinct α values:
  - α_inst = 1/2  (instantaneous, d_spatial = 3, beta-Bernoulli σ-scaling)
  - α_cum  = 25/48 (cumulative, d_eff_horizon = 37/12, T_today propagation)

Phase IIa F-fibers (EWSB, QCD, PS→SM) in the cosmic-history bounded sweep
explicitly use ALPHA_THERMAL = 0.5 = α_inst. By analogy, Phase IIb freezeout
events (ν decoupling, BBN-1) ought to use α_inst as well, since they are
INSTANTANEOUS events at specific N, not cumulative observables.

The current predictions/T_nu_dec.py uses α_cum = 25/48 — a potential
calibration mistake.

This probe:
  (1) Compute T_ν_dec under both α choices and report.
  (2) Project T_BBN-1 = T_ν_dec · 7/15 under each.
  (3) Compute n/p_freeze and Y_p under each.
  (4) Check internal consistency: for each α, find N at T_ν_dec via T(N) =
      T_P·N^(-α). The bounded sweep claims ν decoupling at N ~10^43; check
      which α reproduces that.
  (5) Compare to observed Y_p = 0.245 ± 0.003.

The result informs whether Y_p falsification is robust to α choice or
sensitive to a single calibration ambiguity.

Run:
    python3 proofs/cosmology/T_BBN_alpha_choice_audit_2026-05-27.py
"""

import math


# --- Framework primitives (theorem-grade upstream) ---
G_F  = 1.1663787e-5    # GeV^-2 (predictions/G_F.py)
M_Pl = 1.22089e19      # GeV (CODATA via predictions/M_Pl_natural.py)
Q_np_MeV = 1.2933      # MeV (PDG, bounded by Need-B per BR4 closure-neg)
Q_np = Q_np_MeV * 1e-3 # GeV

# ΛCDM reference
T_nu_dec_LCDM = 1.5e-3      # GeV (= 1.5 MeV)
T_BBN_LCDM    = 0.7e-3      # GeV (= 0.7 MeV)
RATIO_BBN_NU  = 7.0/15.0    # ΛCDM-empirical kinematic ratio
DECAY_FACTOR  = 0.7         # n β-decay during cosmic time from T_BBN-1 to T_BBN-D
Y_p_obs = 0.245
Y_p_sigma = 0.003


def T_freezeout(alpha, G_F, M_Pl, prefactor=1.0):
    """Solve Γ_weak(T) = H(T) under coasting H(T) = prefactor · T^(1/α) · M_Pl^(1-1/α).

    Γ_weak = G_F² · T^5 (relativistic limit, no continuum 7π/60 — Clause-9 safe).

    Returns T_F in GeV.
    """
    inv_a = 1.0 / alpha
    exponent_lhs = 5 - inv_a
    M_Pl_factor = M_Pl ** (-(inv_a - 1))
    rhs = M_Pl_factor / (G_F ** 2) / prefactor
    return rhs ** (1.0 / exponent_lhs)


def N_at_T(T, alpha, M_Pl):
    """Project N at which T(N) = T_P · N^(-α) equals T."""
    return (M_Pl / T) ** (1.0 / alpha)


def Y_p_from_T_BBN(T_BBN, Q_np_GeV, decay):
    """Y_p from n/p ratio at weak freezeout + β-decay correction."""
    n_p_freeze = math.exp(-Q_np_GeV / T_BBN)
    n_p_final = n_p_freeze * decay
    return 2 * n_p_final / (1 + n_p_final), n_p_freeze, n_p_final


print("=" * 78)
print("  T_BBN-1 CALIBRATION RECHECK — α-choice audit + Y_p propagation")
print("=" * 78)
print(f"\n  Framework primitives:")
print(f"    G_F      = {G_F:.4e} GeV^-2")
print(f"    M_Pl     = {M_Pl:.4e} GeV")
print(f"    Q_np     = {Q_np_MeV} MeV")
print(f"  Reference (ΛCDM): T_ν_dec = 1.5 MeV, T_BBN-1 = 0.7 MeV, Y_p_obs = {Y_p_obs}")

# --- (1)+(2)+(3) Two α choices ---
print()
print("-" * 78)
print("  Two α choices under unified-observation reframe")
print("-" * 78)

for label, alpha_num, alpha_den in [
    ("α_inst = 1/2 (instantaneous; Phase IIa-style F-fiber)", 1, 2),
    ("α_cum  = 25/48 (cumulative; T_today propagation)", 25, 48),
]:
    alpha = alpha_num / alpha_den
    T_nu = T_freezeout(alpha, G_F, M_Pl)
    T_BBN = T_nu * RATIO_BBN_NU
    Y_p, n_p_freeze, n_p_final = Y_p_from_T_BBN(T_BBN, Q_np, DECAY_FACTOR)
    dev_sigma = (Y_p - Y_p_obs) / Y_p_sigma
    N_at_T_nu = N_at_T(T_nu, alpha, M_Pl)

    print(f"\n  {label}")
    print(f"    H(T) form: prefactor=1, T^({alpha_den}/{alpha_num}) · M_Pl^(1-{alpha_den}/{alpha_num})")
    print(f"    T_ν_dec (from Γ=H)    = {T_nu*1e3:.3f} MeV")
    print(f"    T_BBN-1 (× 7/15 ratio) = {T_BBN*1e3:.3f} MeV")
    print(f"    n/p_freeze             = {n_p_freeze:.4f}")
    print(f"    n/p_final (× decay)    = {n_p_final:.4f}")
    print(f"    Y_p prediction         = {Y_p:.4f}")
    print(f"    deviation              = {dev_sigma:+.2f}σ from observed {Y_p_obs}")
    print(f"    N at T_ν_dec           = {N_at_T_nu:.3e}  (log₁₀ = {math.log10(N_at_T_nu):.2f})")

# --- (4) Cross-consistency check: which α matches bounded sweep N ~10^43 ---
print()
print("-" * 78)
print("  (4) Bounded-sweep consistency check")
print("-" * 78)
print(f"\n  cosmic_history_bounded_sweep_consolidation_2026-05-27.py lists:")
print(f"    ν decoupling (T_ν_dec) | N ~10⁴³ | 3.18 MeV")
print()
print(f"  Required to match BOTH (T_ν_dec = 3.18 MeV AND N ~10⁴³):")
print(f"    From Γ=H with α=25/48:   T = 3.18 MeV ✓; N = {N_at_T(3.18e-3, 25/48, M_Pl):.2e} (log₁₀={math.log10(N_at_T(3.18e-3, 25/48, M_Pl)):.2f}) ✗")
print(f"    From Γ=H with α=1/2:     T = 0.84 MeV ✗; N = {N_at_T(3.18e-3, 1/2, M_Pl):.2e} (log₁₀={math.log10(N_at_T(3.18e-3, 1/2, M_Pl)):.2f}) ✓")
print()
print(f"  INTERNAL INCONSISTENCY: predictions/T_nu_dec.py uses α=25/48 to get 3.18 MeV,")
print(f"  but the bounded-sweep N=10^43 projection requires α=1/2.")
print(f"  The 3.18 MeV value and the 10^43 N value are produced by DIFFERENT α choices.")

# --- (5) What α matches observed Y_p? ---
print()
print("-" * 78)
print("  (5) What α is needed to match observed Y_p = 0.245?")
print("-" * 78)

# For Y_p = 0.245, with decay factor 0.7:
#   Y_p = 2·n_p_final / (1 + n_p_final) → n_p_final = Y_p/(2-Y_p) = 0.245/1.755 = 0.1396
#   n_p_freeze = n_p_final/0.7 = 0.1994
#   T_BBN-1 = -Q_np / ln(n_p_freeze) = 1.293 / 1.612 = 0.802 MeV
#   T_ν_dec = T_BBN-1 / (7/15) = 0.802 · 15/7 = 1.719 MeV
T_BBN_target = Q_np / (-math.log(Y_p_obs / (2 - Y_p_obs) / DECAY_FACTOR))
T_nu_target = T_BBN_target / RATIO_BBN_NU
print(f"\n  Target T_BBN-1 = {T_BBN_target*1e3:.3f} MeV")
print(f"  Target T_ν_dec = {T_nu_target*1e3:.3f} MeV")

# Find α such that T_freezeout(α) = T_nu_target
# T^(5-1/α) = M_Pl^(-(1/α - 1)) / G_F²
# log T · (5 - 1/α) = -(1/α - 1)·log M_Pl - log G_F²
# Solve numerically
log_T = math.log10(T_nu_target)
log_M = math.log10(M_Pl)
log_GF2 = math.log10(G_F ** 2)
# log_T · (5 - 1/α) = -(1/α - 1) · log_M - log_GF2
# (5 - 1/α)·log_T + (1/α - 1)·log_M + log_GF2 = 0
# 5·log_T - (1/α)·log_T + (1/α)·log_M - log_M + log_GF2 = 0
# (1/α)·(log_M - log_T) = log_M - 5·log_T - log_GF2
# 1/α = (log_M - 5·log_T - log_GF2) / (log_M - log_T)
inv_alpha = (log_M - 5*log_T - log_GF2) / (log_M - log_T)
alpha_target = 1.0 / inv_alpha
print(f"  Required α     = {alpha_target:.4f}  (between α_inst=0.500 and α_cum=0.521)")
print()
print(f"  Note: neither α_inst (0.5) nor α_cum (25/48 ≈ 0.521) hits {alpha_target:.4f}.")
print(f"  The framework's two structural α values do NOT include this value.")

# --- Y_p sensitivity to α ---
print()
print("-" * 78)
print("  Y_p as a function of α (sweep)")
print("-" * 78)
print(f"  {'α':>8}  {'T_ν_dec (MeV)':>14}  {'T_BBN-1 (MeV)':>14}  {'Y_p':>8}  {'Δσ':>8}")
for a in [0.470, 0.480, 0.490, 0.500, 0.505, 0.510, 0.515, 0.521, 0.530, 0.540]:
    T_nu = T_freezeout(a, G_F, M_Pl)
    T_BBN = T_nu * RATIO_BBN_NU
    Y_p, _, _ = Y_p_from_T_BBN(T_BBN, Q_np, DECAY_FACTOR)
    dev = (Y_p - Y_p_obs) / Y_p_sigma
    print(f"  {a:>8.4f}  {T_nu*1e3:>14.3f}  {T_BBN*1e3:>14.3f}  {Y_p:>8.4f}  {dev:>+8.2f}σ")

# --- VERDICT ---
print()
print("=" * 78)
print("  VERDICT — path 3 (T_BBN-1 calibration recheck)")
print("=" * 78)
print(f"""
  (1) Y_p is HIGHLY sensitive to α: a change from 0.500 to 0.521 (the two
      structural α values) shifts Y_p from ~0.05 to ~0.45 — spanning the
      entire range from below-observed to far-above-observed.

  (2) The α that matches observed Y_p = 0.245 is α ≈ {alpha_target:.4f}, which is
      between α_inst and α_cum but NOT EQUAL to either.

  (3) The framework's structural choice of α for Phase IIb freezeout is
      currently AMBIGUOUS:
        - predictions/T_nu_dec.py uses α_cum = 25/48 (→ 3.18 MeV, Y_p=0.45)
        - bounded sweep N-projection uses α_inst = 1/2 (→ 0.84 MeV, Y_p=0.05)
      The 3.18 MeV value and the 10^43 N projection are produced by DIFFERENT
      α choices; this is an internal inconsistency.

  (4) Resolving the inconsistency requires either:
      (a) A structural argument that Phase IIb uses α_cum (justifying current
          T_nu_dec=3.18 MeV, accepting Y_p=0.45 falsification)
      (b) A structural argument that Phase IIb uses α_inst (lowering T_nu_dec
          to 0.84 MeV but UNDERPREDICTING Y_p to 0.05 — also falsified)
      (c) A different α that lies between 1/2 and 25/48 (no current framework
          justification)
      (d) Additional structural pieces (e.g., √g_*-like prefactor in H, or
          proper coasting-modified BBN reaction network)

  Path 3 finding: T_BBN-1 calibration recheck does NOT admit a clean
  resolution within the framework's existing structural commitments. Both
  natural α choices are FALSIFIED by Y_p.

  This shifts the open frontier to path 2 (full BBN reaction network under
  coasting) or path 4 (Axiom F / framework extension).
""")
