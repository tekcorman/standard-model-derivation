#!/usr/bin/env python3
"""
4d_dirac_continuum_bridge_probe.py
==================================
Step 3 of the 4D spacetime spectral triple project.  Steps 1 and 2 closed
positively (D_4 well-formed almost-commutatively, inner-fluctuation YM
coefficient extracted at 8/(3π²) per (edge, F-pair), with residual 9π/4 to
the framework's α_GUT⁻¹/(4π) = 6/π).

Step 3 has two main jobs:

  (A) STRUCTURAL: verify the 9π/4 residual decomposes into framework
      theorem-grade quantities (algebraic identity), giving the structural
      bookkeeping that Step 2 deferred to Step 3.

  (B) CONTINUUM BRIDGE: identify the substrate UV cutoff Λ_sub via the
      Π_TT path-(b) substrate-Planck reframing (`theorem_g_sub_drude_closure_
      2026-04-30.md`), and the framework's gauge unification scale M_unif
      (from the 2026-05-04 theorem-grade program).  These are different
      scales — Step 4 (next session) will run between them via MSSM β.

Bounded scope.  This probe does NOT compute the running 1/g²(µ) from Λ_sub
to M_Z — that's Step 4's job, requires choosing SM vs MSSM matter content
+ running RG via standard 1-loop or 2-loop β-functions.  Step 3's deliverable
is the structural set-up: where each scale sits, what the residual factor's
algebraic form is, and what the next-session running needs as inputs.

What this probe does
--------------------
A — Verify the 9π/4 = sin²θ_W × α_GUT⁻¹ × π/N_atoms algebraic identity.
B — Compute Λ_sub in framework-natural, M_Pl, and GeV units via Π_TT's path-(b).
C — Compute M_unif via the framework's substrate-derived formula and compare.
D — Identify what's needed for Step 4 (MSSM b_i match).

No graded content changes from this probe.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

np.set_printoptions(precision=8, suppress=True, linewidth=120)

# Framework theorem-grade values (upstream)
SIN2_THETA_W_UNIF = 3.0 / 8.0          # theorem_sin2_theta_W_unification.md
ALPHA_GUT_INV = 24.0                   # theorem-grade Class C, proofs/gauge/alpha_GUT_derivation.py
N_ATOMS = 4                            # srs primitive cell
N_EDGES = 6                            # srs primitive cell
K_STAR = 3                             # Hashimoto Perron eigenvalue
GIRTH = 10                             # srs girth

# Π_TT path-(b) substrate-Planck reframing (theorem_g_sub_drude_closure_2026-04-30.md)
# M_substrate / M_Pl = √π / 8   (substrate UV scale)
M_SUB_OVER_M_PL = np.sqrt(np.pi) / 8.0
# Equivalently, M_Pl / M_substrate = 8/√π
M_PL_OVER_M_SUB = 8.0 / np.sqrt(np.pi)

# CODATA M_Pl
M_PL_GEV = 1.22e19


# -----------------------------------------------------------------------------
# Part A — verify 9π/4 = sin²θ_W × α_GUT⁻¹ × π / N_atoms
# -----------------------------------------------------------------------------

def part_A_residual_decomposition():
    print("=" * 100)
    print("PART A — Step 2's 9π/4 residual decomposes into framework theorem-grade quantities")
    print("=" * 100)
    target = 9.0 * np.pi / 4.0
    decomp = SIN2_THETA_W_UNIF * ALPHA_GUT_INV * np.pi / N_ATOMS
    print(f"\n  Step 2 residual:  ratio = (α_GUT⁻¹/(4π)) / (8/(3π²)) = (6/π) / (8/(3π²)) = 9π/4")
    print(f"  numerical:        9π/4                              = {target:.10f}")
    print(f"\n  Decomposition:    sin²θ_W × α_GUT⁻¹ × π / N_atoms")
    print(f"                  = (3/8)   ×    24   × π / 4")
    print(f"                  = {decomp:.10f}")
    print(f"  match:            {abs(target - decomp) < 1e-12}")
    assert abs(target - decomp) < 1e-12
    print(f"\n  Per-factor framework theorem-grade upstream sources:")
    print(f"    sin²θ_W = 3/8     →  docs/theorems/theorem_sin2_theta_W_unification.md")
    print(f"    α_GUT⁻¹ = 24      →  docs/theorems/theorem_sin2_theta_W_unification.md §11,")
    print(f"                          proofs/gauge/alpha_GUT_derivation.py")
    print(f"    π                 →  BZ-edge frequency in lattice units (heat-kernel measure)")
    print(f"    N_atoms = 4       →  srs primitive cell (theorem-grade structural integer)")
    print(f"\n  HONEST INTERPRETATION.  This is an algebraic identity (numerator 3·24 = 72,")
    print(f"  denominator 8·4 = 32, simplifies to 9/4).  It's MEANINGFUL — vs a coincidence —")
    print(f"  only if each factor enters INDEPENDENTLY in the spectral-action expansion via")
    print(f"  the framework's Cl(6) → PS → SM embedding bookkeeping.  Verifying that each")
    print(f"  factor IS extractable independently from the CC machinery is multi-session.")
    print(f"\n  What this DOES show:  9π/4 is NOT some random number;  it admits a clean")
    print(f"  framework-theorem-grade rewrite, supporting the hypothesis that the spectral-")
    print(f"  action route correctly reproduces the framework's α_GUT⁻¹ = 24.")


# -----------------------------------------------------------------------------
# Part B — continuum bridge Λ_sub from Π_TT path-(b)
# -----------------------------------------------------------------------------

def part_B_continuum_bridge():
    print("\n" + "=" * 100)
    print("PART B — substrate UV cutoff Λ_sub via Π_TT path-(b) (theorem_g_sub_drude_closure)")
    print("=" * 100)
    print(f"\n  Π_TT theorem (path-(b) substrate-Planck closure, 2026-04-30):")
    print(f"    M_substrate / M_Pl     = √π / 8 ≈ {M_SUB_OVER_M_PL:.6f}")
    print(f"    M_Pl       / M_substrate = 8 / √π ≈ {M_PL_OVER_M_SUB:.6f}")
    print(f"\n  Substrate UV cutoff identification (per handoff Step 3a):")
    print(f"    Λ_sub = π × M_substrate  (the BZ-edge frequency in lattice units)")
    print(f"    Λ_sub / M_Pl  = π × √π / 8  =  π^(3/2) / 8")

    L_sub_over_M_Pl = np.pi ** 1.5 / 8.0
    print(f"                  = {L_sub_over_M_Pl:.6f}")
    print(f"    Λ_sub (M_Pl units) = {L_sub_over_M_Pl:.4f}")
    L_sub_GeV = L_sub_over_M_Pl * M_PL_GEV
    print(f"    Λ_sub (GeV)        = {L_sub_GeV:.4e}")
    print(f"                       ≈ 8.49 × 10¹⁸ GeV")
    print(f"\n  This is the substrate's natural UV cutoff:  the BZ-edge frequency scaled by")
    print(f"  the substrate-Planck mass ratio.  In framework-natural units (M_substrate = 1),")
    print(f"  Λ_sub = π ≈ {np.pi:.6f}.")


# -----------------------------------------------------------------------------
# Part C — framework's M_unif (substrate-derived) vs Λ_sub
# -----------------------------------------------------------------------------

def part_C_M_unif_comparison():
    print("\n" + "=" * 100)
    print("PART C — framework's M_unif (theorem-grade substrate formula) vs Λ_sub")
    print("=" * 100)
    print(f"\n  Framework's M_unif (predictions/M_unif.py, 2026-05-04 theorem-grade-conditional):")
    print(f"    M_unif = (32 / k*^(g-1)) × M_Pl   [substrate Markov-return formula]")
    print(f"           = α_GUT × α_1_bare × M_Pl  [equivalent form, with α_1_bare = (2/3)^8]")

    # Compute M_unif via the substrate formula
    M_unif_over_M_Pl = 32.0 / K_STAR ** (GIRTH - 1)
    M_unif_GeV = M_unif_over_M_Pl * M_PL_GEV
    print(f"    M_unif / M_Pl     = 32 / 3^9 = 32 / {K_STAR ** (GIRTH - 1)} = {M_unif_over_M_Pl:.6e}")
    print(f"    M_unif (GeV)      = {M_unif_GeV:.4e}")
    print(f"                      ≈ 1.99 × 10¹⁶ GeV")

    L_sub_over_M_Pl = np.pi ** 1.5 / 8.0
    ratio = L_sub_over_M_Pl / M_unif_over_M_Pl
    print(f"\n  Scale comparison:")
    print(f"    Λ_sub  / M_Pl   ≈ {L_sub_over_M_Pl:.4f}              (substrate UV)")
    print(f"    M_unif / M_Pl   ≈ {M_unif_over_M_Pl:.4e}              (framework gauge unification)")
    print(f"    Λ_sub / M_unif  ≈ {ratio:.4f}                  ← spectral-action's CC scale is ~430× above M_unif")
    print(f"\n  INTERPRETATION.  The spectral-action's 1/g² is computed at the substrate's UV scale")
    print(f"  Λ_sub ≈ 0.696 × M_Pl, NOT at the framework's gauge unification scale M_unif ≈")
    print(f"  1.6×10⁻³ × M_Pl.  Comparison to MSSM b_i (Step 4) requires running 1/g² from Λ_sub")
    print(f"  down through M_unif and on to M_Z.  Between Λ_sub and M_unif the running is governed")
    print(f"  by the framework's matter content (substrate-derived);  between M_unif and M_Z by")
    print(f"  standard SM or MSSM β-functions.")
    print(f"\n  The α_GUT⁻¹ = 24 ALWAYS holds at M_unif (definition + the framework's structural value).")
    print(f"  The spectral-action's bare 1/g²(Λ_sub) needs to RUN to give 24/(4π) at µ = M_unif.")
    print(f"  Whether the spectral-action prediction matches α_GUT⁻¹ = 24 numerically AT Λ_sub")
    print(f"  is a different (and prior) question from whether it matches AT M_unif.")
    # Show the ratio in a few alternative forms
    print(f"\n  The ratio Λ_sub/M_unif ≈ {ratio:.2f}.  In alternative substrate quantities:")
    print(f"    π^(3/2) / 8        × k*^(g-1) / 32  =  π^(3/2) × k*^(g-1) / 256")
    a = (np.pi ** 1.5) * (K_STAR ** (GIRTH - 1)) / 256.0
    print(f"                                          =  π^(3/2) × {K_STAR**(GIRTH-1)} / 256")
    print(f"                                          =  {a:.4f}")
    print(f"    not a clean small integer — the substrate UV scale and gauge unification scale")
    print(f"    are structurally DIFFERENT objects in the framework's existing derivations.")


# -----------------------------------------------------------------------------
# Part D — what's needed for Step 4 (MSSM b_i match)
# -----------------------------------------------------------------------------

def part_D_step4_inputs():
    print("\n" + "=" * 100)
    print("PART D — inputs needed for Step 4 (MSSM b_i match, next session)")
    print("=" * 100)
    print(f"""
  Step 4 needs to test:  does the spectral-action's bare 1/g²(Λ_sub), run via SM or MSSM
  β-functions down to M_Z, match the observed coupling values  α_1(M_Z), α_2(M_Z), α_3(M_Z)?
  And does it pass through α_GUT⁻¹ = 24 at µ = M_unif?

  Specific inputs Step 4 needs assembled:

  (i)  BARE 1/g² AT Λ_sub.  From Step 2 + Part A here:
       bare 1/g²_i (Λ_sub) = (something) × Tr_F per-factor
       where "(something)" decomposes through the 9π/4 = sin²θ_W × α_GUT⁻¹ × π/N_atoms
       structural identity, IF the Cl(6) → SM embedding is bookkeeping-clean (multi-session).

  (ii) RUNNING from Λ_sub to M_unif.  Between substrate UV and gauge unification,
       what matter content runs?  Standard CC SM: continuum matter content (3 generations
       of fermions + Higgs).  Framework: substrate-derived matter (Cl(6) Fock per vertex,
       Iorio sector).  This identifies the relevant β-functions in the high-µ window.

  (iii) MATCHING AT M_unif.  At µ = M_unif, the framework's structural value
        α_GUT⁻¹ = 24, sin²θ_W = 3/8, gives the unification ratios.  This is the IR
        boundary condition for the high-µ running.

  (iv) RUNNING from M_unif to M_Z.  Standard SM or MSSM β-functions (existing
        infrastructure: `proofs/gauge/_mssm_rge.py`, `gauge_unification_full_RG_closure.py`).
        Compare predicted α_i(M_Z) to observation.

  Failure modes pre-registered for Step 4:
    N1 — matching to MSSM b_i without checking SM b_i first (the framework's matter content
         between Λ_sub and M_unif is the open question, NOT whether MSSM is right).
    N2 — claiming closure if Λ_sub-to-M_unif running needs new structural content
         (e.g. additional matter thresholds derived from substrate) that the existing
         framework hasn't yet established.

  Step 4 effort: ≥ 1 session.  Could close MSSM-Sb question OR identify a specific
  residual obstacle that needs further structural work.
""")


# -----------------------------------------------------------------------------
def main():
    print(r"""
==========================================================================================
4D DIRAC CONTINUUM BRIDGE — STEP 3 of 4D spacetime spectral-triple project
Identify Λ_sub via Π_TT path-(b), compare to M_unif, document Step 2's 9π/4 decomposition.
==========================================================================================""")
    part_A_residual_decomposition()
    part_B_continuum_bridge()
    part_C_M_unif_comparison()
    part_D_step4_inputs()
    print("\n" + "=" * 100)
    print("STEP 3 INTERIM VERDICT")
    print("=" * 100)
    print("""
  WHAT THIS PROBE ESTABLISHED

  (A) ALGEBRAIC DECOMPOSITION:  Step 2's residual 9π/4 = sin²θ_W × α_GUT⁻¹ × π / N_atoms.
      All four factors are framework theorem-grade upstream.  This is a clean rewrite,
      consistent with the hypothesis that spectral action reproduces the framework's
      α_GUT⁻¹ = 24 via the Cl(6) embedding.  Verifying each factor's independent emergence
      from the spectral-action machinery is multi-session bookkeeping.

  (B) CONTINUUM BRIDGE:  Λ_sub = π × M_substrate = M_Pl × π^(3/2)/8 ≈ 0.696 M_Pl ≈
      8.49 × 10¹⁸ GeV.  This is the substrate UV cutoff inherited from Π_TT's path-(b)
      substrate-Planck reframing.

  (C) SCALE IDENTIFICATION:  Λ_sub (substrate UV ≈ 0.7 M_Pl) and M_unif (gauge unification
      ≈ 1.6e-3 M_Pl) are STRUCTURALLY DIFFERENT scales in the framework, separated by
      ≈ 430x.  The spectral-action prediction is at Λ_sub;  the framework's α_GUT⁻¹ = 24
      is at M_unif.  Comparison requires RG running from Λ_sub through M_unif to M_Z.

  (D) STEP 4 INPUTS:  4 specific items identified for the next-session MSSM b_i match.

  WHAT REMAINS

  Step 4 — RG running from Λ_sub to M_Z, compare predicted α_i(M_Z) to observation.  This
  is the final closure test for the MSSM β-question.  Effort: ≥1 session.

  HONEST SCOPE OF THIS STEP.  Step 3 sets up the continuum-bridge mathematics + validates
  the structural decomposition of Step 2's residual.  Step 3 does NOT close MSSM β
  (Step 4's job) and does NOT independently derive α_GUT⁻¹ = 24 from CC (multi-session
  bookkeeping per Step 3a's note).  ADOPTED-MSSM-Sb stands.

  No graded content changes from this probe.
""")
    print("4d_dirac_continuum_bridge_probe.py: sentinel done.")


if __name__ == "__main__":
    main()
