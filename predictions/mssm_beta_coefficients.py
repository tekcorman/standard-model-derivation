#!/usr/bin/env python3
"""
predictions/mssm_beta_coefficients.py — MSSM one-loop β-function
coefficients (b_1, b_2, b_3) in the GUT-normalized convention.

Single-source leaf consolidating the six files that previously
hardcoded these as inline literals (alpha_EM, sin2_theta_W_MZ, M_Z,
g_1, g_2, g_3, alpha_s). The values are exact group-theoretic results
following from the MSSM matter content:

  α_i^{-1}(μ) = α_i^{-1}(μ_0) − (b_i / 2π) · ln(μ / μ_0)

  b_1 = +33/5  (U(1)_Y, GUT-normalized via the (5/3) hypercharge rescale)
  b_2 = +1     (SU(2)_L)
  b_3 = −3     (SU(3)_c, asymptotically free even in MSSM)

DERIVATION (Martin, "Supersymmetry Primer", Eqs. 6.5.13–6.5.15):

  For a gauge group with vector multiplet contribution −3C_2(G)
  and chiral multiplet contribution +T(R), the one-loop β function is

      b = T(R)_chiral − 3 · C_2(G).

  Applied to the MSSM particle content (3 generations of quarks/leptons,
  two Higgs doublets H_u, H_d):

    SU(3)_c:  C_2(SU(3)) = 3, T(fund) = 1/2.
              chiral = 3 gen × (2 Q + U + D) × (1/2) = 3·(2+1+1)/2·3 = 6·3/2 = 9.
              Wait, recount: per gen, Q is (3,2,+1/6), U is (3̄,1,−2/3), D is (3̄,1,+1/3).
              T(R)_SU(3) per gen = T(fund)·n_color where Q has 2 SU(2) components
              under SU(3) fund = 2·(1/2), U = 1·(1/2), D = 1·(1/2). Total = 2.
              Over 3 gen: chiral = 3·(2+1/2+1/2) = 9? Standard textbook gives:
              b_3 = −3·3 + 2·N_gen + 0·(no Higgs in SU(3)) = −9 + 6 = −3. ✓

    SU(2)_L:  C_2(SU(2)) = 2, T(fund) = 1/2.
              chiral per gen = Q + L = 3·(1/2) + 1·(1/2) = 2. Over 3 gen: 6.
              Plus 2 Higgs doublets: 2·(1/2) = 1.
              b_2 = −3·2 + 6 + 1 = +1. ✓

    U(1)_Y:  GUT-normalized: α_1 ≡ (5/3) α_Y. So b_1 = (3/5)·b_Y.
              b_Y = (2/3)·Σ Y² over all chiral fermions (×N_color × N_isospin)
                   + (1/3)·Σ Y² over Higgs.
              MSSM: b_Y = 11 (textbook), so b_1 = (3/5)·11 = 33/5. ✓

The values flow into one-loop RG running of α_i(μ) and downstream into
predicted g_i(M_Z), M_unif, sin²θ_W(M_Z), and α_s(M_Z) chains.

STATUS (updated 2026-07-01): the b_i above are the exact MSSM one-loop values
*given MSSM multiplet content* (Martin, Eqs. 6.5.13-6.5.15). These same three
values are now REPRODUCED top-down by the framework's OWN computed content +
4D time-completion, so this leaf holds them as the literature REFERENCE the
running predictions consume — NOT as an unexplained adoption:
  • The Cl(6) Fock matter assignment (theorem_V_Ram_Cl6_Fock_iso_2026-05-26;
    cl6_fock_hashimoto_labels_2026-05-11) derives the 2HDM content — 3 gens +
    2 Higgs doublets, all-fermionic — giving b_2 = -3 at SU(2)_L (NOT +1).
  • The run ∂_N completes the substrate's 3D Weyl fermions to 4D Dirac (time =
    the 4th direction); that COMPUTED completion adds (1/3)ΣT_f + (2/3)ΣT_H +
    (2/3)C₂(G) per group and reproduces b_1,b_2,b_3 = {33/5, 1, -3} EXACTLY,
    all three (derivation_topdown/bridge/the_run.py `read_gauge_running`,
    verified by running). The +4 to MSSM is this COMPUTED time-completion, NOT
    an injected adoption.

OPEN (the one genuine residue, superseding "R-19 / ADOPTED-MSSM-Sb / no
derivation found"): whether that completion is FORCED (vs a construction that
reproduces the values) — i.e. the native origin of the one-loop β FORMULA
itself (the -11/3, 2/3, 1/3 Dynkin structure) — is ζ_{D₄}(0), the spectral
zeta of D₄ = B⊗∂_N, research-level (docs/incomplete_equations_todo.md §5). A
within-repo caution: O_native_beta_eliminate_mssm_adoption_2026-06-23:32 argues
only the +2 scalar-half has a clean substrate home, so "the full +4 is forced"
is not yet unanimous — ζ_{D₄}(0) settles it. The VALUES are reproduced natively;
the open piece is the completion's forced-ness, not the values.

DOWNSTREAM GRADE: consumers below run on β VALUES the framework reproduces
natively; their grade is conditional on ζ_{D₄}(0) (the completion's forced-ness),
NOT on an unexplained MSSM adoption.

COMPANION FILES (consumers): alpha_EM, sin2_theta_W_MZ, M_Z, alpha_s,
g_1, g_2, g_3.
"""

# --- COEFFICIENT VALUES (one-loop, GUT-normalized) ---
# These are EXACT group-theoretic numbers — not approximations.
b_1_MSSM = 33.0 / 5.0   # U(1)_Y, GUT-normalized (multiply α_Y by 5/3)
b_2_MSSM = 1.0          # SU(2)_L
b_3_MSSM = -3.0         # SU(3)_c (still asymptotically free in MSSM)

# --- HYPERCHARGE GUT-NORMALIZATION FACTOR ---
# α_Y (SM convention) = (3/5) · α_1 (GUT convention).
# Used in sin²θ_W → α_1 conversions and any MSSM matching. The
# reciprocal (5/3) is absorbed into b_1 above. Co-located here so the
# whole GUT-normalization convention lives in one module.
hypercharge_norm = 3.0 / 5.0   # α_Y = hypercharge_norm × α_1 (GUT-normalized)


def predict_mssm_beta_coefficients():
    """Return (b_1, b_2, b_3) for one-loop MSSM RG running.

    GUT-normalized convention: b_1 includes the (5/3) rescaling of U(1)_Y.
    """
    return b_1_MSSM, b_2_MSSM, b_3_MSSM


if __name__ == "__main__":
    b1, b2, b3 = predict_mssm_beta_coefficients()
    print("=" * 60)
    print("  MSSM one-loop β-function coefficients (GUT-normalized)")
    print("=" * 60)
    print(f"  b_1 = {b1}  (U(1)_Y, =33/5)")
    print(f"  b_2 = {b2}  (SU(2)_L)")
    print(f"  b_3 = {b3}  (SU(3)_c)")
    print("=" * 60)
