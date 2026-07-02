#!/usr/bin/env python3
"""
PS β coefficients from substrate-derived matter content — Session 1 probe.

Scoping doc: an internal working note (P1).

GOAL: derive the one-loop β-function coefficients (b_4, b_2L, b_2R) for the
Pati-Salam gauge group SU(4)_PS × SU(2)_L × SU(2)_R from the substrate's
fermion multiplet content, and cross-check against canonical non-SUSY PS
literature values.

If this lands, the framework can run gauge couplings from M_unif down to M_R
(~10^15 GeV) using PS β coefficients above M_R and SM β coefficients below,
removing the dependency on MSSM β coefficients that 7 predictions currently use.

SCOPE OF SESSION 1:
- Matter content (3 generations × [(4, 2, 1) + (4̄, 1, 2)]): substrate-derived
  via Cl(6,0) ⊕ Cl(0,2) Fock decomposition. Theorem-grade matter assignment.
- Higgs content: NOT substrate-derived in this session. Reported as a separate
  contribution with model-dependence noted.

ONE-LOOP β CONVENTION:
  μ ∂g/∂μ = b · g³ / (16π²)
  b = -(11/3) C_2(G) + (2/3) Σ_Weyl T(R) + (1/3) Σ_complex_scalar T(R)

GROUP-THEORY INPUTS (canonical, no fitting):
  C_2(SU(N)) = N            (adjoint Casimir)
  T(fundamental SU(N)) = 1/2 (Dynkin index)
"""

# ============================================================
# GROUP-THEORY CONSTANTS
# ============================================================
C2_SU4 = 4   # adjoint Casimir of SU(4)
C2_SU2 = 2   # adjoint Casimir of SU(2)
T_FUND = 0.5  # Dynkin index of fundamental representation, any SU(N)


# ============================================================
# SUBSTRATE-DERIVED MATTER CONTENT
# ============================================================
# Per generation, Pati-Salam embeds the SM matter as:
#
#   ψ_L = (4, 2, 1) — left-handed quarks + lepton (lepton-as-fourth-color)
#         8 Weyl fermions: (u_L, d_L) × 3 colors + (ν_L, e_L) × 1 "lepton color"
#
#   ψ_R = (4̄, 1, 2) — right-handed quarks + lepton (+ ν_R)
#         8 Weyl fermions: (u_R, d_R) × 3 colors + (ν_R, e_R) × 1 "lepton color"
#
# This multiplet structure is substrate-derived via Cl(6,0) ⊕ Cl(0,2) Fock
# decomposition. See proofs/foundations/R1_1_cl6_fock_su4_PS_decomposition_probe.py
# for the structural derivation. N_GENERATIONS = 3 is the framework's
# theorem-grade generation count.
N_GENERATIONS = 3


def T_index_matter():
    """
    Compute the total Dynkin index T(R) over all matter multiplets, for each
    gauge factor.

    For each multiplet:
      T_SU4 = (multiplicity of SU(4) fundamentals) × T_FUND
      T_SU2L = (multiplicity of SU(2)_L fundamentals) × T_FUND
      T_SU2R = (multiplicity of SU(2)_R fundamentals) × T_FUND

    (4, 2, 1): under SU(4), is 2 copies of fundamental (SU(2)_L doubles it);
               under SU(2)_L, is 4 copies of fundamental (SU(4) doubles by 4);
               singlet under SU(2)_R.
    (4̄, 1, 2): under SU(4), 2 copies of antifundamental (T = T_FUND);
               singlet under SU(2)_L;
               under SU(2)_R, 4 copies of fundamental.
    """
    # (4, 2, 1) contributions per generation
    T_SU4_421 = 2 * T_FUND     # 2 copies of SU(4) fundamental
    T_SU2L_421 = 4 * T_FUND    # 4 copies of SU(2)_L fundamental
    T_SU2R_421 = 0             # singlet

    # (4̄, 1, 2) contributions per generation
    T_SU4_412bar = 2 * T_FUND  # antifundamental has same T as fundamental
    T_SU2L_412bar = 0          # singlet
    T_SU2R_412bar = 4 * T_FUND # 4 copies of SU(2)_R fundamental

    # Total per generation
    T_SU4_per_gen = T_SU4_421 + T_SU4_412bar
    T_SU2L_per_gen = T_SU2L_421 + T_SU2L_412bar
    T_SU2R_per_gen = T_SU2R_421 + T_SU2R_412bar

    # Over N_GENERATIONS
    return (
        N_GENERATIONS * T_SU4_per_gen,
        N_GENERATIONS * T_SU2L_per_gen,
        N_GENERATIONS * T_SU2R_per_gen,
    )


def beta_coefficients_matter_only():
    """
    Compute (b_4, b_2L, b_2R) from matter content only (no Higgs).

    b = -(11/3) C_2(G) + (2/3) Σ_Weyl T(R)
    """
    T4, T2L, T2R = T_index_matter()
    b_4_matter = -(11.0 / 3.0) * C2_SU4 + (2.0 / 3.0) * T4
    b_2L_matter = -(11.0 / 3.0) * C2_SU2 + (2.0 / 3.0) * T2L
    b_2R_matter = -(11.0 / 3.0) * C2_SU2 + (2.0 / 3.0) * T2R
    return b_4_matter, b_2L_matter, b_2R_matter


# ============================================================
# HIGGS CONTRIBUTIONS (NOT substrate-derived — model-dependent)
# ============================================================
# The minimal PS Higgs sector contains:
#   Φ = (1, 2, 2)     — bidoublet for EWSB (1 copy)
#   Δ_R = (4̄, 1, 2)   — for SU(4)×SU(2)_R → SU(3)×U(1)_Y breaking (1 copy)
#                       (alternatively (10̄, 1, 3) for explicit see-saw)
#
# This Higgs sector is the conventional minimal choice. Substrate-side
# derivation of which scalars exist is a separate open question.

def T_index_higgs_minimal():
    """
    Dynkin index sums for the minimal PS Higgs sector:
      (1, 2, 2) + (4̄, 1, 2)
    """
    # (1, 2, 2): SU(4) singlet, 2 copies of SU(2)_L fund, 2 copies of SU(2)_R fund
    T_SU4_122 = 0
    T_SU2L_122 = 2 * T_FUND
    T_SU2R_122 = 2 * T_FUND

    # (4̄, 1, 2): 2 copies of SU(4) antifund, singlet under SU(2)_L, 4 copies SU(2)_R
    T_SU4_412 = 2 * T_FUND
    T_SU2L_412 = 0
    T_SU2R_412 = 4 * T_FUND

    return (
        T_SU4_122 + T_SU4_412,
        T_SU2L_122 + T_SU2L_412,
        T_SU2R_122 + T_SU2R_412,
    )


def beta_coefficients_with_minimal_higgs():
    """
    Matter + minimal Higgs (1,2,2) + (4̄,1,2).
    Complex scalars contribute (1/3) T(R) each.
    """
    b_4_m, b_2L_m, b_2R_m = beta_coefficients_matter_only()
    T4_h, T2L_h, T2R_h = T_index_higgs_minimal()
    b_4 = b_4_m + (1.0 / 3.0) * T4_h
    b_2L = b_2L_m + (1.0 / 3.0) * T2L_h
    b_2R = b_2R_m + (1.0 / 3.0) * T2R_h
    return b_4, b_2L, b_2R


# ============================================================
# CANONICAL LITERATURE VALUES (non-SUSY Pati-Salam)
# ============================================================
# Mohapatra-Pal, "Massive Neutrinos in Physics and Astrophysics" §6.4;
# also Hewett-Rizzo PRD 1989; Mohapatra-Senjanovic 1981.
#
# Matter-only (3 generations):
#   b_4 = -32/3
#   b_2L = -10/3
#   b_2R = -10/3
#
# With minimal Higgs (1,2,2) + (4̄,1,2):
#   b_4 = -31/3
#   b_2L = -3
#   b_2R = -7/3

LIT_MATTER_ONLY = (-32.0 / 3.0, -10.0 / 3.0, -10.0 / 3.0)
LIT_WITH_HIGGS = (-31.0 / 3.0, -3.0, -7.0 / 3.0)


# ============================================================
# PROBE EXECUTION
# ============================================================
def report():
    b_m = beta_coefficients_matter_only()
    b_h = beta_coefficients_with_minimal_higgs()

    print("=" * 72)
    print("  PS β coefficients (one-loop) from substrate matter content")
    print("  Session 1 of PS → SM observer-graph transition scoping")
    print("=" * 72)

    print("\n  Inputs (substrate-derived, theorem-grade):")
    print(f"    N_generations    = {N_GENERATIONS}")
    print(f"    matter content   = 3 × [(4, 2, 1) + (4̄, 1, 2)]")
    print(f"    multiplet source = Cl(6,0) ⊕ Cl(0,2) Fock (R1_1 probe)")

    print("\n  Group-theory primitives:")
    print(f"    C_2(SU(4)) = {C2_SU4}")
    print(f"    C_2(SU(2)) = {C2_SU2}")
    print(f"    T(fund)    = {T_FUND}")

    T4, T2L, T2R = T_index_matter()
    print("\n  Dynkin index sums (matter, all 3 gens):")
    print(f"    Σ T(R)_SU(4)    = {T4}")
    print(f"    Σ T(R)_SU(2)_L  = {T2L}")
    print(f"    Σ T(R)_SU(2)_R  = {T2R}")

    print("\n  Derived β coefficients (matter only, no Higgs):")
    print(f"    b_4   = {b_m[0]:+.6f}  (= {b_m[0]*3:+.1f}/3)")
    print(f"    b_2L  = {b_m[1]:+.6f}  (= {b_m[1]*3:+.1f}/3)")
    print(f"    b_2R  = {b_m[2]:+.6f}  (= {b_m[2]*3:+.1f}/3)")

    print("\n  Literature (non-SUSY PS, matter only, 3 gens):")
    print(f"    b_4   = {LIT_MATTER_ONLY[0]:+.6f}  (= -32/3)")
    print(f"    b_2L  = {LIT_MATTER_ONLY[1]:+.6f}  (= -10/3)")
    print(f"    b_2R  = {LIT_MATTER_ONLY[2]:+.6f}  (= -10/3)")

    print("\n  Match (matter-only):")
    eps = 1e-10
    match_4 = abs(b_m[0] - LIT_MATTER_ONLY[0]) < eps
    match_2L = abs(b_m[1] - LIT_MATTER_ONLY[1]) < eps
    match_2R = abs(b_m[2] - LIT_MATTER_ONLY[2]) < eps
    print(f"    b_4   : {'EXACT MATCH' if match_4 else 'MISMATCH'}")
    print(f"    b_2L  : {'EXACT MATCH' if match_2L else 'MISMATCH'}")
    print(f"    b_2R  : {'EXACT MATCH' if match_2R else 'MISMATCH'}")

    print("\n  --- WITH MINIMAL HIGGS ((1,2,2) + (4̄,1,2)) ---")
    print(f"    b_4   = {b_h[0]:+.6f}  vs literature -31/3 = {LIT_WITH_HIGGS[0]:+.6f}")
    print(f"    b_2L  = {b_h[1]:+.6f}  vs literature -3    = {LIT_WITH_HIGGS[1]:+.6f}")
    print(f"    b_2R  = {b_h[2]:+.6f}  vs literature -7/3  = {LIT_WITH_HIGGS[2]:+.6f}")

    match_h_4 = abs(b_h[0] - LIT_WITH_HIGGS[0]) < eps
    match_h_2L = abs(b_h[1] - LIT_WITH_HIGGS[1]) < eps
    match_h_2R = abs(b_h[2] - LIT_WITH_HIGGS[2]) < eps
    print(f"    b_4   : {'EXACT MATCH' if match_h_4 else 'MISMATCH'}")
    print(f"    b_2L  : {'EXACT MATCH' if match_h_2L else 'MISMATCH'}")
    print(f"    b_2R  : {'EXACT MATCH' if match_h_2R else 'MISMATCH'}")

    print("\n" + "=" * 72)
    print("  VERDICT")
    print("=" * 72)

    all_matter = match_4 and match_2L and match_2R
    all_higgs = match_h_4 and match_h_2L and match_h_2R

    if all_matter:
        print("  [MATTER]  PS β coefficients from substrate-derived matter content")
        print("            MATCH canonical non-SUSY PS literature values EXACTLY.")
        print("            Grade: THEOREM-GRADE-DERIVED (matter-only contribution).")
    else:
        print("  [MATTER]  MISMATCH — check Dynkin index computation.")

    if all_higgs:
        print("  [HIGGS]   Matter + minimal Higgs reproduces canonical values.")
        print("            But Higgs content is NOT substrate-derived — model-dependent.")
        print("            Grade: PHENOMENOLOGICALLY-CORRECT, STRUCTURALLY OPEN.")
    else:
        print("  [HIGGS]   Mismatch in Higgs contribution — check assumed Higgs sector.")

    print("\n  Session 1 conclusion:")
    print("    - Substrate matter content (3 × [(4,2,1)+(4̄,1,2)]) suffices to derive")
    print("      the matter-only contribution to b_4, b_2L, b_2R for non-SUSY PS.")
    print("    - The matter-only values match standard literature exactly:")
    print("      b_4 = -32/3, b_2L = b_2R = -10/3.")
    print("    - Higgs sector contributions are quantitatively known but require")
    print("      separate substrate-side derivation (Session 1 does NOT close this).")
    print("    - Net Session 1 status: matter contribution structurally derived;")
    print("      Higgs contribution model-dependent → β coefficients overall not")
    print("      yet at theorem-grade until Higgs sector is substrate-grounded.")

    print("\n  Next step (Session 2 P2): threshold matching at M_R from m_nu3.py,")
    print("  two-regime running PS-above / SM-below, compare α_i(M_Z) to PDG.")
    print("=" * 72)

    return {
        "matter_only": b_m,
        "with_higgs": b_h,
        "matter_match": all_matter,
        "higgs_match": all_higgs,
    }


if __name__ == "__main__":
    report()
