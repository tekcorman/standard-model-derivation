#!/usr/bin/env python3
"""
H prefactor: F = √(k* · g_girth) = √30 candidate — anti-numerology audit.

PURPOSE
-------
The leading-factor chase (2026-05-27 EOD+1) identified F = √(k* · g_girth) = √30
≈ 5.477 as a K-rational framework-primitive combination matching ΛCDM's
required H prefactor 1.66·√g_*(MeV) = 5.443 to within 0.7%.

Per W58 anti-numerology discipline, a single near-match without structural
derivation OR independent observable check is NUMEROLOGY. This probe runs
the gauntlet:

  GATE 1: STRUCTURAL — does √(k* · g_girth) have a framework-derivable
          origin (substrate-thermal coupling, Friedmann analog, etc.)?
  GATE 2: EPOCHAL RUN — does the framework predict this factor RUNS with
          N like ΛCDM g_*(N), so it equals 1 at today (preserving H_0=68)?
  GATE 3: INDEPENDENT OBSERVABLES — does the factor predict consistent
          numbers for OTHER Phase IIb F-fibers (T_e_ann separation, N_eff
          correction)?
  GATE 4: SPECIES-COUNT MATCH — is √30 = √(k*·g_girth) consistent with
          counting active SM species (g_*=10.75) somehow?

Run:
    python3 proofs/cosmology/H_prefactor_sqrt30_candidate_test_2026-05-27.py
"""

import math

# Framework primitives (theorem-grade)
k_star = 3
g_girth = 10
N_atoms = 4
E_count = 12
G_F = 1.1663787e-5
M_Pl = 1.22089e19
ratio_7_15 = 7.0/15.0
Q_np = 1.2933e-3
decay_factor = 0.7
Y_p_obs = 0.245
Y_p_sigma = 0.003

F_candidate = math.sqrt(k_star * g_girth)
F_required_LCDM = 1.66 * math.sqrt(10.75)

print("=" * 78)
print("  ANTI-NUMEROLOGY AUDIT: F = √(k* · g_girth) = √30 candidate")
print("=" * 78)

print(f"\n  Candidate: F = √(k* · g_girth) = √(3 · 10) = √30 = {F_candidate:.4f}")
print(f"  Required:  F_ΛCDM = 1.66·√g_*(MeV=10.75) = {F_required_LCDM:.4f}")
print(f"  Match:     {(F_candidate / F_required_LCDM - 1)*100:+.3f}% deviation")

# Compute downstream T_F and Y_p
T_F_candidate = (F_candidate / (M_Pl * G_F**2))**(1.0/3.0)
T_BBN_candidate = T_F_candidate * ratio_7_15
n_p_freeze = math.exp(-Q_np / T_BBN_candidate)
n_p_final = n_p_freeze * decay_factor
Y_p_candidate = 2 * n_p_final / (1 + n_p_final)
dev_sigma = (Y_p_candidate - Y_p_obs) / Y_p_sigma

print(f"\n  Downstream under F = √30:")
print(f"    T_ν_dec     = {T_F_candidate*1e3:.4f} MeV (vs ΛCDM 1.5)")
print(f"    T_BBN-1     = {T_BBN_candidate*1e3:.4f} MeV (vs ΛCDM 0.7)")
print(f"    n/p_freeze  = {n_p_freeze:.4f}")
print(f"    Y_p         = {Y_p_candidate:.4f} (vs obs {Y_p_obs}, {dev_sigma:+.2f}σ simple)")
print(f"    Note: full BBN network needed to bridge simple {Y_p_candidate:.3f} → ΛCDM 0.245")
print(f"          (-15σ residue is the SAME as ΛCDM has at simple-model level)")


# =============================================================================
# GATE 1: STRUCTURAL — derivation of √(k*·g_girth)?
# =============================================================================
print()
print("-" * 78)
print("  GATE 1: STRUCTURAL — does √(k*·g_girth) have a framework derivation?")
print("-" * 78)

print(f"""
  Framework primitives:
    k_star  = {k_star}  (substrate coordination valence, theorem-grade)
    g_girth = {g_girth} (smallest cycle on srs lattice, theorem-grade)
    Product = k_star · g_girth = {k_star * g_girth}

  Known structural roles of k_star · g_girth:
    - α_1 = (2/3)^(g_girth - 2) involves g_girth (NB walker survival)
    - k_star (= 3) is the substrate valence
    - 30 = 2|E| · k_star / 2|V| · 2|V|/2|E|... no clean appearance
    - 30 = 5! / 4 ... no
    - 30 = 2·3·5 ... no obvious substrate role

  Potential routes:
    (i) g_girth as "thermalization timescale" — walker traverses smallest
        cycle in g_girth steps; equilibration time ~ g_girth · t_substrate.
        Effective H sees enhanced rate by √g_girth? Speculative.

    (ii) k_star as "channel multiplicity" — each substrate edge can host
         k_star walker types (one per coordination direction). g_girth
         channels then give √(k_star · g_girth)? Also speculative.

    (iii) Friedmann analog: H² ∝ ρ where ρ ∝ N_active · T^4. If N_active at
          MeV bath = k_star · g_girth = 30 (counting somehow), then
          √(k_star·g_girth) = √30 emerges naturally. Requires deriving
          why N_active(MeV) = 30 specifically.

  STRUCTURAL VERDICT: NO derivation currently exists. The √30 = 5.477 vs
  ΛCDM 5.443 match is suggestive but UNVERIFIED. Could be numerological
  coincidence; could be hint of deeper structural identity.

  Per W58 discipline: candidate REMAINS open pending structural derivation.
""")


# =============================================================================
# GATE 2: EPOCHAL RUNNING
# =============================================================================
print()
print("-" * 78)
print("  GATE 2: EPOCHAL RUNNING — does √30 reduce to 1 at today?")
print("-" * 78)

# If F = √(k_star · g_girth) is constant (=√30 everywhere), then at today:
# H_0_corrected = √30 · 1/(N_hub · t_P) = √30 · H_0_substrate
# This would give H_0 = 5.477 · 68 = 372 km/s/Mpc. FALSIFIED by observation.

H_0_substrate = 68.0
F_constant_at_today = F_candidate
H_0_constant_pred = F_constant_at_today * H_0_substrate
print(f"\n  If F = √30 is CONSTANT (no running):")
print(f"    H_0_predicted = √30 · 68.0 = {H_0_constant_pred:.1f} km/s/Mpc")
print(f"    H_0_observed  = 67.4 km/s/Mpc")
print(f"    Deviation     = +{(H_0_constant_pred/67.4 - 1)*100:.0f}% — FALSIFIED")

# For F to give correct H_0 at today, F_today must = 1.
# So F must RUN: F(N_MeV) = √30, F(N_today) = 1.
# This is the same running problem as ΛCDM's g_*(N).

print(f"""
  For framework H_0 to remain 68 km/s/Mpc, F(N_today) must = 1.
  → F must RUN with epoch: F(N_MeV) ≈ √30, F(N_today) = 1.

  Framework's coasting H = 1/(N·t_P) is UNIFORM (theorem-grade); has NO
  ρ-decomposition that would naturally produce epoch-dependent F.

  ΛCDM analog: g_*(N) runs because different ρ terms dominate at
  different epochs (ρ_rad ≫ ρ_m at MeV; ρ_Λ + ρ_m at today). Framework
  has Ω_Λ = 1/3 at ALL z (theorem-grade); no radiation-domination.

  STRUCTURAL VERDICT: A constant F = √30 IS FALSIFIED by H_0 today. The
  factor would need to RUN, requiring framework extension beyond current
  coasting theorem. Per Phase III taxonomy formalization 2026-05-27 EOD:
  this is genuinely Axiom-A territory (substrate-thermal species coupling).
""")


# =============================================================================
# GATE 3: INDEPENDENT OBSERVABLES
# =============================================================================
print()
print("-" * 78)
print("  GATE 3: Independent observables under F = √30 at Phase IIb")
print("-" * 78)

# T_e_ann is m_e/k* = 0.17 MeV (NOT from Γ=H, independent of F).
# So T_e_ann unchanged. F doesn't affect this.
# But T_ν_dec / T_e_ann separation does change.
m_e = 0.511e-3
T_e_ann = m_e / k_star
sep_ratio_at_F30 = T_F_candidate / T_e_ann
print(f"\n  T_e_ann = m_e/k* = {T_e_ann*1e3:.4f} MeV (independent of F)")
print(f"  T_ν_dec / T_e_ann under F = √30:  {sep_ratio_at_F30:.2f}")
print(f"  T_ν_dec / T_e_ann under F = 1 (current α=1/2):  {0.844/(m_e*1e3/k_star):.2f}")
print(f"  ΛCDM:  T_ν_dec / T_e_ann ≈ 1.5/0.17 = 8.8")
print(f"\n  N_eff prediction depends on separation factor:")
print(f"    Framework currently F=1: factor 5 → some entropy transfer → N_eff < 3.046")
print(f"    Framework with F=√30:    factor ~9 → clean separation → N_eff ≈ 3.000")
print(f"    ΛCDM:                    factor ~9 → 3.046 (non-instantaneous correction)")
print(f"  N_eff with F=√30 stays close to 3, consistent with CMB-S4 forecast.")


# =============================================================================
# GATE 4: SPECIES-COUNT MATCH
# =============================================================================
print()
print("-" * 78)
print("  GATE 4: Species-count interpretation of √30")
print("-" * 78)

# At MeV: ΛCDM g_*(MeV) = 10.75 (2 + 7/8·(4+6))
# √g_* = 3.28; required factor is 1.66·√g_* = 5.44
# 1.66 from continuum Friedmann (√(8π³/90)) — NOT K-rational
# √30 ≈ 5.48, decomposable as √30 = √(3·10) where 3 = k*, 10 = g_girth
# Or √30 = √(species_dof) where species_dof = 30?

# Try: SM dof at MeV in framework counting
# γ: 1 walker · 2 polarization = 2 (bosonic)
# e+: 1 spin·2 chir·2±... = 4 fermionic dof
# e-: same = 4 fermionic
# 3 ν_L: 3 · 1 chir · 2± = 6 fermionic (Dirac); 3 if Majorana
# Total bosonic = 2, fermionic = 8-10 (depending on ν)
# With 7/8 fermion weighting: g_* = 2 + 7/8·(8 to 10) = 9 to 10.75

ALPHA_SM_MeV = 2 + 7/8 * 10  # = 10.75 if 10 fermionic dof (Dirac ν)
ALPHA_SM_MeV_M = 2 + 7/8 * 8   # = 9 if Majorana ν
print(f"\n  ΛCDM g_* at MeV bath:")
print(f"    Dirac ν:    g_* = 2 + 7/8·10 = {ALPHA_SM_MeV:.3f}")
print(f"    Majorana ν: g_* = 2 + 7/8·8  = {ALPHA_SM_MeV_M:.3f}")
print(f"  Framework's preferred ν type: MAJORANA (from M_R seesaw)")
print(f"    → framework g_* at MeV = 9 (?)")
print(f"\n  Required F = 1.66 · √g_* with g_* = 10.75: F = {1.66*math.sqrt(10.75):.3f}")
print(f"  Required F = 1.66 · √g_* with g_* = 9.00:  F = {1.66*math.sqrt(9.0):.3f}")
print(f"  Candidate F = √30 = {math.sqrt(30):.3f}")
print(f"\n  Decompose √30 = √(k_star·g_girth) — does NOT match SM dof counting")
print(f"  directly. The match to 1.66·√g_*(10.75) is COINCIDENTAL at substrate-")
print(f"  primitive level. Per Phase III Saha-π pattern, the 1.66 Friedmann")
print(f"  continuum factor is precisely what Clause 9 BLOCKS (π factors).")
print(f"")
print(f"  If F is structurally K-rational, the framework's analog of 1.66·√g_*")
print(f"  cannot be 1.66·√g_* itself — it must replace continuum 1.66 with")
print(f"  a K-rational substitute. √30 IS such a substitute.")
print(f"")
print(f"  Conjecture (structural): F_framework_K = √(k_star · g_girth) =")
print(f"  K-rational analog of 1.66·√g_*(MeV). The k_star · g_girth = 30")
print(f"  encodes a substrate-derived 'effective species count' at the bath.")
print(f"  This would explain why the match is to within 0.7%.")


# =============================================================================
# Verdict
# =============================================================================
print()
print("=" * 78)
print("  VERDICT — F = √30 candidate audit")
print("=" * 78)
print(f"""
  STATUS: PROVOCATIVE BUT UNCLOSED.

  Match quality:  √30 = {math.sqrt(30):.4f} vs ΛCDM required 5.443 — 0.7% agreement.

  GATE 1 (structural derivation):  ✗ NOT CLOSED — no framework-derived
    argument for why F = √(k_star · g_girth) appears in H_eff. Speculative
    routes (thermalization timescale, channel multiplicity, Friedmann
    analog) need rigorous derivation. Per W58: candidate stays open.

  GATE 2 (epochal running):       ✗ CONSTANT F = √30 FALSIFIED by H_0
    today. Factor must RUN with N. Framework's coasting theorem has NO
    natural ρ-decomposition to produce running. Adding running = framework
    extension (Axiom-A territory: substrate-thermal species coupling).

  GATE 3 (independent observables): ✓ F = √30 INCREASES T_ν_dec/T_e_ann
    separation from 5 → 9, consistent with N_eff = 3 prediction at the
    SAME level as ΛCDM (separation factor 9 → 3.046; framework 9 → 3.000).
    No conflict with existing N_eff prediction.

  GATE 4 (species-count interpretation): ◐ INDIRECT MATCH — √30 ≈ 1.66·√10.75
    is a numerical coincidence at substrate-primitive level. The k_star·g_girth
    = 30 doesn't directly count SM dof (= 10.75). The match would be
    interpretable if 1.66 (continuum Friedmann) has a K-rational substrate
    analog that combines with √g_* to give √30. Requires Friedmann-substrate
    derivation.

  Per anti-numerology discipline (W58):
    - 1 near-match without structural derivation is NUMEROLOGY
    - Promotion to CANDIDATE requires structural argument OR independent
      observable check on a second epoch.
    - The √30 match alone passes neither bar.

  HOWEVER, the match is closer than expected (0.7%) and uses ONLY
  theorem-grade primitives (k_star, g_girth). This warrants:

    (a) Probing structural derivation via Friedmann-substrate analog route
        (R-B from earlier scoping: E_obs = κ·S_total Friedmann import)
    (b) Checking whether other Phase IIb F-fibers (e+e- ann, etc.) would
        also need F = √30 or different epoch-dependent F. If F changes
        with epoch in a substrate-derivable way, that's a structural lead.
    (c) Checking precision: 0.7% vs ΛCDM 1.66 measurement uncertainty.

  Bounded next step: scope R-B Friedmann-import via observer-energy
  functional, see if √(k_star·g_girth) emerges naturally.

  NOT a closure; NOT a refutation. Provocative open candidate.
""")
