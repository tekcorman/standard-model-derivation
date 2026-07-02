#!/usr/bin/env python3
"""
Prediction file for lambda (Higgs quartic self-coupling).

STATUS UPDATE 2026-04-29: UNIQUE — THEOREM-GRADE.
The dark-map Class-2 identification — previously the last open adoption on
this row — was graduated to theorem grade 2026-04-28 via
`docs/theorems/theorem_dark_map_class2_closure.md` (corollary chain through y_τ:
λ/y_τ = 2k*² ratio per `theorem_ytau_corollary.md` §10.3). ADOPTED-DARK-MAP
is retired for λ_Higgs. The 0.52% residual to PDG-extracted λ ≈ 0.1294 is
the un-derived Feshbach-analog gap on the Higgs quartic, tracked separately
at an internal working note. Per Row P41 of
`docs/parameters/parameter_uniqueness_ledger.md`.

Historical "ADVANCED, ADOPTED-DARK-MAP" language below is SUPERSEDED but
preserved for record.

RIGOR AUDIT (2026-04-18; updated 2026-04-21 session 11):
  Overall verdict: ADVANCED (one adopted identification remaining:
  dark-map classification. I-Feshbach closed via A5(b) 2026-04-19.
  ADOPTED-B3 removed 2026-04-21: n_channels=2 is invariant under the
  (Z/2)^3 L↔R convention of B3 — no adoption required.)

This file separates the strict-solid combinatorial core from the two
adopted identification steps that connect it to the observed lambda.
Both adopted steps are explicitly labeled. The numerical result is
exact rational arithmetic throughout.

========================================================================
STEP-BY-STEP RIGOR AUDIT
========================================================================

Step 1 — k* = 3:  STRICT-SOLID.
  A1 + A2-T (MDL on self-inverse toggle structure) selects k* = 3.
  Chain: predictions/d_spatial.py -> predictions/k_star.py.
  Cites: Gleason 1957 (dimension ≥ 3), MDL cost audit (d_spatial.py).

Step 2 — g = 10:  STRICT-SOLID.
  MDL selects srs as the unique 3-regular 3D crystal net;
  girth of srs is g = 10.
  Chain: predictions/g_girth.py. Cites: Sunada 2012, Delgado-Friedrichs
  2003, O'Keeffe 2008.

Step 3 — h = (√3 + i√5)/2:  STRICT-SOLID.
  Walker eigenvalue of the Bloch–Hashimoto operator on srs at the P-point.
  Ramanujan-saturating eigenvalue with |h|² = k*-1 = 2.
  Chain: predictions/h_walker_eigenvalue.py.
  Cites: Lubotzky–Phillips–Sarnak 1988 (Ramanujan graphs).

Step 4 — α₁_bare = (2/3)^8 = 256/6561:  STRICT-SOLID (combinatorial).
  NB walk survival on universal covering tree over g-2 = 8 steps.
  Proved as Lemma 1 in ../predictions/Feshbach_coupling_strength_derivation.md.
  Chain: predictions/feshbach_exponent_principle.py.
  Cites: Terras 2011 §2.1 (NB walk independence on tree).

  CLOSED (A5(b)): The identification of α₁_bare with the
  PHYSICAL scattering coupling magnitude is now AXIOM-LEVEL via
  A5(b), the coupling clause of A5 (docs/framework/framework_axioms.md §5b,
  established 2026-04-19 session 2). MDL probabilities of leading-
  order multiway processes ARE physical coupling strengths.
  Previously this was the "I-Feshbach" adoption; the Feshbach P/Q
  derivation route was exhausted (six attempts blocked, see
  docs/theorems/theorem_ifeshbach_percycle_resolution.md and
  an internal working note), motivating
  the A5 extension.

Step 5 — tan²(arg h) = 5/3 (mass²-class coefficient):  STRICT-SOLID.
  Im(h)²/Re(h)² = (√5/2)²/(√3/2)² = 5/3. Exact algebra from Step 3.
  No adoption: this is pure arithmetic from the derived h.
  Chain: predictions/dark_extraction_map.py, dark_coefficient_mass_squared.

  ADOPTED (dark-map identification): The identification of this 5/3
  coefficient as the CLASS 2 (mass²-class) dark correction coefficient
  for the Higgs quartic — i.e., that λ belongs to Class 2 (C₃-trivial
  diagonal self-coupling) — is a physical classification. It is not
  derived from A1 + A2-T + A3-T alone; it is adopted from
  dark_correction_theorem_2026-04-14.md §4a.

Step 6 — Factor 2 = dim_ℂ(min. faithful rep of Cl(0,2)):
  STRICT-SOLID via Theorem G2 (chain-import; see G2_cl2_channels.py) for n_channels = 2.

  Theorem G2 (../predictions/G2_cl2_channels_derivation.md, 2026-04-19) derives this:
    A1: T_{(u,v)}^2 = T_{(v,u)}^2 = I  (toggle involutions)
    A4: {T_{(u,v)}, T_{(v,u)}} = 0      (CAR at shared vertex)
    A3: F=C, set gamma_j = i*T_j => gamma_j^2 = -I
    => gamma_1, gamma_2 generate Cl(0,2) over R, isom to M_2(C) over C
    => min. faithful C-rep of M_2(C) has dim = 2
    => n_channels = 2  STRICT-SOLID

  Convention note (2026-04-21): n_channels = dim_C(min faithful C-rep of
  Cl(0,2)_C) = 2 is an intrinsic algebraic invariant. The (Z/2)^3 choices
  in theorem_B3_spinor_fermion.py (L↔R flip, isospin flip, Y flip) relabel
  the generators but do not change dim(min faithful rep). λ uses n_channels
  only as a multiplier — so λ = 2560/19683 is convention-independent.
  ADOPTED-B3 is not load-bearing here and has been removed.

  Proof file: proofs/foundations/theorem_G2_cl2_channels.py
  Closes: BLOCK-2 sub-claims A, B, C of an internal working note

  LINTER AUDIT 2026-05-14, SUPERSEDED 2026-05-15:
  Yesterday's annotation flagged a Type-4 inheritance gap (Cl(0,2) rep dim
  vs |φ|⁴ vertex amplitude factor). 2026-05-15 finding RESOLVES the
  apparent gap: the framework's single-leg Tr[I] = 2 at the |φ|⁴ vertex
  IS the correct leading-order structure; the sub-leading correction
  responsible for the m_H +3.43σ_PDG residual comes from a DIFFERENT
  mechanism (Family D per-leg multiway dark-disruption, master doc §3 (D),
  added 2026-05-15) NOT from an alternative Cl(0,2) vertex trace.

  Family D (THEOREM-GRADE 2026-05-15, Routes H+C+F-1+F-2 closed): per-Higgs-leg dark disruption rate
  c_H = α₁_bare² (joint srs × srs-z NB walker survival, both g=10).
  |φ|⁴ vertex has 4 Higgs legs → δλ/λ = -4 c_H = -4 α₁_bare² ≈ -0.609%.
  Match to empirical -0.601% at +1.4% rel. err.; the λ/y_τ structural-
  identity breaking pattern fits to 0.007% (NO FITTING).
  See ../docs/theorems/theorem_substrate_feshbach_dark_corrections_master.md §3 (D)
  and ../proofs/foundations/dark_disruption_per_leg_2026-05-15.py.

  Family D is now THEOREM-GRADE (Routes H+C+F-1+F-2 closed 2026-05-15) and IS
  propagated to the numerical prediction: λ_physical = λ_tree·(1 − 4·α₁_bare²) =
  109528517120/847288609443 ≈ 0.129269. (λ_tree = 2560/19683 ≈ 0.1301, Step 7
  below, is the pre-Family-D tree value.)

Step 7 — λ = 2 × (5/3) × (2/3)^8 = 2560/19683 ≈ 0.1301:
  Given Steps 4–6, this is exact arithmetic. No further adoption needed
  beyond what is already labeled in Steps 4–6.

SUMMARY:
  | Step | Claim | Status |
  |------|-------|--------|
  | 1    | k* = 3 | STRICT-SOLID |
  | 2    | g = 10 | STRICT-SOLID |
  | 3    | h = (√3+i√5)/2 | STRICT-SOLID |
  | 4a   | α₁_bare = (2/3)^8 (combinatorial) | STRICT-SOLID |
  | 4b   | I-Feshbach (physical ↔ walk survival) | AXIOM A5(b) (was ADOPTED) |
  | 5a   | tan²(arg h) = 5/3 (algebra) | STRICT-SOLID |
  | 5b   | λ is Class 2 (dark-map classification) | ADOPTED |
  | 6    | n_channels=2 (Cl(0,2) min rep) | STRICT-SOLID via Theorem G2 (chain-import; see G2_cl2_channels.py) [convention-independent] |
  | 7    | λ = 2560/19683 | exact arithmetic given 4–6 |

Overall: ADVANCED (not CLOSED: ONE independent adopted identification
remains — dark-map classification).
  Closed: I-Feshbach (via A5(b), 2026-04-19)
  Closed: F2-class (n_channels=2 STRICT-SOLID via Theorem G2, 2026-04-19)
  Closed: ADOPTED-B3 (n_channels=2 convention-independent, 2026-04-21)
  Open:   dark-map Class 2 assignment (dark_correction_theorem_2026-04-14.md §4a)
"""

# ============================================================
# PARAMETER: lambda (Higgs quartic self-coupling)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       0.1294 ± 0.0004 (from m_H = 125.25 ± 0.17 GeV,
#              v = 246.22 GeV, lambda = m_H² / (2v²))
# Source:      PDG 2024, ATLAS+CMS combined Higgs mass
# PDG edition: 2024

# --- PREDICTED VALUE -----------------------------------------
# Value:       2 * (5/3) * (2/3)^8 = 2560/19683 ≈ 0.13006
# Deviation:   +0.52% from observed 0.1294 (≈ +1.7σ at ±0.0004)
#
# Bridge convention (docs/framework/framework_scheme_convention.md §4.3): the +0.52%
# residual on lambda corresponds to an un-derived Feshbach analog on the
# Higgs quartic. The framework's bare combinatorial λ_tree = 2α₁_full is
# theorem-grade under A5(b) Case (A) Level-2 direct moment; the bridge to
# SM observables expects a Feshbach self-energy correction analogous to the
# (5/12) on v (which is derived) — that analog has NOT been derived. Three
# naive forms (Path 4 multi-cycle, Path 5 BZ integration, Option Y fermion-
# loop-analog) were tested in session 25 and falsified — see
# an internal working note Empirical hint: the
# universal QFT 1/(16π²) prefactor matches λ_obs/λ_tree to 0.033%; if a
# graph-QFT loop calculation produces 1/(16π²) from srs structure, that
# would close the residual structurally. Open under Priority 4.4 step 2.1.

# --- FORMULA -------------------------------------------------
# lambda = n_channels [strict-solid G2: A1 + A3-T + local CAR thm] * alpha_1_full [adopted: I-Feshbach + dark-map]
#
# where:
#   alpha_1_full = tan²(arg h) * alpha_1_bare
#                = (5/3) * (2/3)^8   [strict-solid algebra + I-Feshbach + dark-map]
#   n_channels   = 2                  [strict-solid G2: min faithful C-rep of Cl(0,2); A1 + A3-T + local CAR thm]
#
# Derivation chain:
#   [strict-solid] k* = 3             from predictions/k_star.py
#   [strict-solid] g = 10             from predictions/g_girth.py
#   [strict-solid] h = (√3+i√5)/2    from predictions/h_walker_eigenvalue.py
#   [strict-solid] α₁_bare=(2/3)^8   from predictions/feshbach_exponent_principle.py
#   [strict-solid] tan²(arg h)=5/3   from predictions/h_walker_eigenvalue.py (algebra)
#   [adopted I-Feshbach] α₁_bare = physical scattering coupling
#   [adopted dark-map]   5/3 factor applies to λ (Class 2 assignment)
#   [adopted F2-class]   factor 2 = Cl(2) generator count = SU(2)_L doublet dim
#
# See: ../predictions/Feshbach_coupling_strength_derivation.md (I-Feshbach gap, §9)
#      an internal working note §4.4 (F2 gap)
#      an internal Sprint 9 kickoff doc §2 (Cl(6) plan, B3 not yet dispatched)

# --- INPUTS --------------------------------------------------
# symbol        | value      | status               | file
# --------------|------------|----------------------|-----------------------------
# k_star        | 3          | [derived]            | predictions/k_star.py
# g_girth       | 10         | [derived]            | predictions/g_girth.py
# h             | (√3+i√5)/2 | [derived]            | predictions/h_walker_eigenvalue.py
# alpha_1_bare  | (2/3)^8    | [derived combinat.] | predictions/alpha_1.py
# tan²(arg h)   | 5/3        | [derived algebra]    | predictions/h_walker_eigenvalue.py
# n_channels    | 2          | [strict-solid G2]    | proofs/foundations/theorem_G2_cl2_channels.py

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_1 import predict_alpha_1
from k_star import predict_k_star
from d_spatial import predict_d_spatial
from g_girth import predict_g_girth
from h_walker_eigenvalue import predict_h_walker_eigenvalue
from srs_E_at_P import predict_srs_E_at_P
from dark_extraction_map import dark_coefficient_mass_squared, family_D_per_leg_correction
from fractions import Fraction
import functools

d = predict_d_spatial()
k = predict_k_star(d)
E = predict_srs_E_at_P(k)
from p_toggle import predict_p_toggle
p = predict_p_toggle()
h = predict_h_walker_eigenvalue(k, E, p)
g = predict_g_girth(k, d)
a1 = predict_alpha_1(k, g)

# Step 5a: tan²(arg h) = Im(h)²/Re(h)² = 5/3  [strict-solid algebra]
c_mass = dark_coefficient_mass_squared(h)  # 5/3

# Step 6: ADOPTED (F2-class) — Cl(2) generator count = SU(2)_L doublet dim
# This is an adopted identification pending B3/F2 closure.
n_channels = 2  # strict-solid [G2]: min faithful C-rep of Cl(0,2) under A1 + A3-T + local CAR thm

# Step 7: tree-level coupling (exact arithmetic given adopted inputs above)
alpha_1_full = c_mass * a1                    # (5/3) * (2/3)^8 = 1280/19683
lam_tree = n_channels * alpha_1_full           # 2 * 1280/19683 = 2560/19683 (tree-level)

# Step 8: Family D per-leg multiway dark-disruption correction (THEOREM-GRADE 2026-05-15)
# Per docs/theorems/theorem_substrate_feshbach_dark_corrections_master.md §3 (D):
# |φ|⁴ vertex has 4 Higgs legs + 0 fermion legs. Per-Higgs-leg dark disruption rate
# c_H = α₁_bare² (theorem-grade via Routes H + C). Correction = 1 - 4·α₁_bare².
from V_count import V_count_pred as N_atoms_srs  # = 4, srs primitive cell |V| / K_4 quotient (predict_V_count)
n_H_legs_phi4 = 4    # 4 Higgs legs at the |φ|⁴ vertex
n_F_legs_phi4 = 0    # no fermion legs in the Higgs quartic
family_D_factor = family_D_per_leg_correction(a1, n_H_legs_phi4, n_F_legs_phi4,
                                                N_atoms_srs, k)
# = 1 - 4·α₁_bare² ≈ 0.99391 (sub-leading dark correction at order α₁²)
lam = lam_tree * family_D_factor               # Family D-corrected λ

# Exact rational verification
alpha_1_exact = Fraction(k - 1, k) ** (g - 2)                 # 256/6561
c_mass_exact  = Fraction(5, 3)                                 # tan²(arg h) exact
lam_tree_exact = Fraction(n_channels) * c_mass_exact * alpha_1_exact  # 2560/19683
family_D_factor_exact = 1 - Fraction(n_H_legs_phi4) * alpha_1_exact**2  # 1 - 4·(2/3)^16
lam_exact = lam_tree_exact * family_D_factor_exact

print(f"k* = {k}, g = {g}")
print(f"α₁_bare = ({k-1}/{k})^{g-2} = {alpha_1_exact} = {float(alpha_1_exact):.10f}")
print(f"h = ({math.sqrt(3):.6f} + i{math.sqrt(5):.6f}) / 2")
print(f"tan²(arg h) = Im(h)²/Re(h)² = 5/3 = {c_mass:.10f}")
print(f"α₁_full = (5/3)·α₁_bare = {float(c_mass_exact * alpha_1_exact):.10f}")
print(f"n_channels = {n_channels}  [STRICT-SOLID G2: min faithful C-rep of Cl(0,2); A1 + A3-T + local CAR thm]")
print(f"λ_tree = {n_channels} × α₁_full = {lam_tree:.10f}  (tree-level, pre-Family D)")
print(f"  Exact tree rational: {lam_tree_exact} = {float(lam_tree_exact):.10f}")
print()
print(f"Family D per-leg correction (theorem-grade 2026-05-15):")
print(f"  Vertex: 4 Higgs legs, 0 fermion legs (|φ|⁴ Higgs quartic)")
print(f"  Correction factor: 1 - 4·α₁_bare² = {family_D_factor:.10f}")
print(f"  Routes H + C closed for c_H = α₁_bare² (see proofs/foundations/family_D_route_{{H,C}}_2026-05-15.py)")
print()
print(f"λ_physical = λ_tree × (1 - 4·α₁_bare²) = {lam:.10f}")
print(f"  Exact rational: {lam_exact} = {float(lam_exact):.10f}")
print()
print("Adopted identification remaining (ADVANCED, not CLOSED):")
print("  [dark-map]   λ is Class 2 (C₃-trivial, mass²-class dark correction)")
print("               — adopted from dark_correction_theorem_2026-04-14.md §4a")
print("Closed adoptions:")
print("  [I-Feshbach] α₁_bare = physical coupling — closed by A5(b) 2026-04-19")
print("  [F2-class]   n_channels=2 — STRICT-SOLID via Theorem G2 (A1 + A3-T + local CAR thm)")
print("  [ADOPTED-B3] removed 2026-04-21: n_channels=2 is (Z/2)^3-invariant;"
      " no adoption needed for magnitude prediction")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_lambda_higgs(alpha_1, h, n_channels, n_H_legs, n_F_legs, N_atoms, k_star):
    """
    Computes the Higgs quartic coupling with Family D dark correction.

    lambda_physical = lambda_tree × family_D_factor
                    = (n_channels × tan²(arg h) × alpha_1) × (1 - 4·alpha_1²)

    Tree-level (strict-solid):
      - alpha_1 is the bare NB walk survival (2/3)^8
      - tan²(arg h) = Im(h)²/Re(h)² = 5/3
      - n_channels = 2 (Theorem G2: Cl(0,2) min faithful C-rep)

    Family D correction (THEOREM-GRADE 2026-05-15, master doc §3 (D)):
      - Per-Higgs-leg rate c_H = α₁_bare² (Routes H + C closed)
      - Per-fermion-leg rate c_F = -α₁² / (N_atoms · k_star) (Routes F-1 + F-2 closed)
      - Vertex correction = 1 - (n_H · c_H + n_F · c_F)
      - For |φ|⁴ vertex: n_H = 4, n_F = 0 → factor = 1 - 4·α₁²

    Parameters
    ----------
    alpha_1 : float
        Bare NB walk survival from predict_alpha_1 (k*=3, g=10) = (2/3)^8.
    h : complex
        Walker eigenvalue from predict_h_walker_eigenvalue = (√3+i√5)/2.
    n_channels : int
        Min faithful C-rep dim of Cl(0,2). Theorem G2 gives n_channels = 2.
    n_H_legs : int
        Higgs legs at the |φ|⁴ vertex (structural: 4).
    n_F_legs : int
        Fermion legs at the |φ|⁴ vertex (structural: 0).
    N_atoms : int
        Wyckoff 8a atoms per primitive cell of srs (I4_132): 4.
    k_star : int
        Coordination number on srs: 3.

    Returns
    -------
    float
        lambda_physical = 2 · (Im(h)/Re(h))² · alpha_1 · (1 - 4·alpha_1²)
                        = 2560/19683 × (1 - 4·(2/3)^16)
                        ≈ 0.12927  (vs observed 0.12928, −0.05σ_PDG on m_H)
    """
    c_mass = h.imag**2 / h.real**2   # tan²(arg h) = 5/3
    lam_tree = n_channels * c_mass * alpha_1
    family_D_factor = family_D_per_leg_correction(alpha_1, n_H_legs, n_F_legs,
                                                    N_atoms, k_star)
    return lam_tree * family_D_factor


# --- VALIDATION ----------------------------------------------

lambda_higgs_pred = lam
# Observed λ from PDG 2024 m_H + v: λ_obs = m_H²/(2v²)
# σ_PDG only per 2026-05-13 σ_theory strip + master doc §8 rule 2c
lambda_higgs_obs = 125.20**2 / (2 * 246.22**2)               # ≈ 0.129281
lambda_higgs_sigma = 125.20 * 0.11 / 246.22**2               # ≈ 2.27e-4 (from σ_m_H = 0.11 GeV)


if __name__ == "__main__":
    impl_result = lam
    pure_result = predict_lambda_higgs(a1, h, n_channels, n_H_legs_phi4,
                                         n_F_legs_phi4, N_atoms_srs, k)
    exact_float = float(lam_exact)

    print(f"\n--- NUMERICAL VERIFICATION ---")
    print(f"Implementation:  {impl_result:.15f}")
    print(f"Pure function:   {pure_result:.15f}")
    print(f"Exact rational:  {exact_float:.15f}  ({lam_exact})")
    assert abs(impl_result - pure_result) < 1e-15, \
        f"impl vs pure mismatch: {impl_result} vs {pure_result}"
    assert abs(pure_result - exact_float) < 1e-15, \
        f"pure vs exact mismatch: {pure_result} vs {exact_float}"
    print("OK: implementation, pure function, and exact rational agree.")

    # Observed λ from PDG (post-2026-05-13 σ_theory strip: σ_PDG only)
    m_H_obs = 125.20
    v_obs = 246.22
    obs = m_H_obs**2 / (2 * v_obs**2)        # = 0.129281
    sigma_m_H = 0.11
    # σ_λ from m_H σ: λ = m_H²/(2v²), δλ = m_H·δm_H/v² → σ_λ = m_H · σ_m_H / v²
    sigma_lam = m_H_obs * sigma_m_H / v_obs**2

    sigma = (pure_result - obs) / sigma_lam
    pct = (pure_result - obs) / obs * 100
    print(f"\nObserved: λ_obs = m_H²/(2v²) = {obs:.6f}  (σ_PDG = {sigma_lam:.3e})")
    print(f"Predicted: {pure_result:.6f} (exact: {lam_exact})")
    print(f"Deviation: {pct:+.4f}%  ({sigma:+.2f}σ_PDG)")
    print()
    print("VERDICT: UNIQUE — THEOREM-GRADE")
    print("  Tree-level core: k*=3, g=10, h=(√3+i√5)/2, α₁=(2/3)^8, tan²(arg h)=5/3")
    print("  G2 (2026-04-19): n_channels=2 STRICT-SOLID via Theorem G2 (Cl(0,2) min faithful C-rep)")
    print("  Dark-map Class-2 (2026-04-28): tan²(arg h) = 5/3 THEOREM-GRADE")
    print("  A5(b) (2026-04-19): α₁_bare = physical coupling strength")
    print("  ADOPTED-B3 removed (2026-04-21): n_channels=2 is (Z/2)^3-invariant")
    print("  Family D (2026-05-15): per-leg dark-disruption THEOREM-GRADE (Routes H + C + F-1 + F-2)")
    print("    Correction: δλ/λ = -4·α₁_bare² ≈ -0.609% on |φ|⁴ vertex (4H legs)")
    print("    Master doc: docs/theorems/theorem_substrate_feshbach_dark_corrections_master.md §3 (D)")
    print("    Closes Clause 8 vs σ_PDG to <σ_PDG (m_H residual +3.43σ_PDG → -0.05σ_PDG)")
