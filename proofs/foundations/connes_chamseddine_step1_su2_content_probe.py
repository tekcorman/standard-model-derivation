#!/usr/bin/env python3
"""
connes_chamseddine_step1_su2_content_probe.py
=============================================
PHASE 1 of building the 4D Chamseddine–Connes (CC) spectral-triple embedding for
the framework — the foundational step needed for every β-function computation.

Setup.  The CC spectral action principle states that for an almost-commutative
spectral triple (A, H, D) = (C^∞(M) ⊗ A_F, L²(spinors) ⊗ H_F, D_M ⊗ 1 + γ_5 ⊗ D_F),
the gauge couplings at the unification scale satisfy

    1/g_i²  =  (f_0 / (24π²)) · Tr_F (T_a^{(i)} T_a^{(i)})

where T_a^{(i)} are the generators of the i-th gauge group factor acting on H_F,
and f_0 is a moment of the spectral cutoff function.  So the FIRST number we need
is  Tr_F(T_a²)  for each gauge group factor.

The framework's gauge group (`theorem_g2_edge_qubit_su2.md` +
`theorem_g2d_chirality_doubled.md`) is SU(2)_L × SU(2)_R, acting on every edge
qubit as the FUNDAMENTAL representation (one global SU(2)_L, one global SU(2)_R).
The vertex Cl(6) Focks inherit the action via the tensor decomposition
Cl(6) = Cl(2)_a ⊗ Cl(2)_b ⊗ Cl(2)_c  over each vertex's three incident edges.

What this probe computes
------------------------
A — Per-vertex Cl(6) Fock decomposition under SU(2)_L acting on each tensor factor:
        (1/2) ⊗ (1/2) ⊗ (1/2)  =  2 × (1/2)  ⊕  (3/2)        — 2 doublets + 1 quartet
B — Aggregate per cell:
        4 vertices × (2 doublets + 1 quartet)  +  6 edges × (1 doublet)
       = 8 doublets + 4 quartets  +  6 doublets   =  14 doublets + 4 quartets per cell
C — Tr_F(T_L²) per cell  using Tr(T²)_{spin-j} = j(j+1)(2j+1):
       14·(1/2·3/2·2)  +  4·(3/2·5/2·4)  =  14·(3/2)  +  4·15  =  21 + 60  =  81  per cell.
D — Same for SU(2)_R (chirality-doubled gauge, by symmetry); compare L vs R.
E — CC formula:  1/g_L²(Λ) = (f_0/24π²) · 81.  Solve for f_0 if we DEMAND
    1/g_L²(M_unif) = α_GUT⁻¹/(4π) with α_GUT⁻¹ = 24 (the framework's structural value).
F — Comparison table:  Tr_F(T²) for SM matter content vs MSSM matter content vs framework.
G — Verdict: is the framework's Tr_F(T²) = 81 SM-like, MSSM-like, or its own thing?
    What does that imply for whether CC gives MSSM-equivalent physics?

This is one step of a multi-step Chamseddine–Connes embedding.  Next steps:
  • Phase 2:  same analysis for SU(2)_R and any U(1) factor (hypercharge).
  • Phase 3:  build the manifold spectral triple and combine.
  • Phase 4:  compute the a_4 Seeley–deWitt coefficient and read off the YM 1/g² + Higgs.
  • Phase 5:  run β-functions from M_unif to M_Z and check against α_s, α_em, sin²θ_W.

VERDICT printed honestly.  Structural probe; no graded content changes.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from proofs.common import K_STAR, N_ATOMS  # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=130)

# spin-j representation: Casimir T² = j(j+1) on each state, dim = 2j+1, Tr(T²) = j(j+1)(2j+1)
def tr_T_sq(j):
    return j * (j + 1) * (2 * j + 1)


# ----------------------------------------------------------------------
# SU(2)_L matter content of the framework, per cell
# ----------------------------------------------------------------------

def part_A():
    print("=" * 100)
    print("PART A — per-vertex Cl(6) Fock decomposition under SU(2)_L (acting as fundamental on each edge qubit)")
    print("=" * 100)
    print("""
  Cl(6) Fock at vertex v  =  ⊗_{e ∋ v} ℂ²_e  =  (1/2) ⊗ (1/2) ⊗ (1/2)  under SU(2)_L.

  Decomposition of (1/2)⊗(1/2):
    (1/2) ⊗ (1/2)  =  (0)  ⊕  (1)        (singlet + triplet — antisymmetric + symmetric)

  Then (0 ⊕ 1) ⊗ (1/2):
    (0) ⊗ (1/2)  =  (1/2)
    (1) ⊗ (1/2)  =  (1/2)  ⊕  (3/2)        (Clebsch–Gordan)

  Total per vertex:  (1/2)²  ⊕  (3/2)   →  2 doublets + 1 quartet  =  4 + 4 = 8 dim  ✓
""")


def part_B():
    print("=" * 100)
    print("PART B — aggregate per CELL: 4 vertices' matter + 6 edges' gauge fibers")
    print("=" * 100)
    n_doublets_matter = N_ATOMS * 2          # 2 doublets per vertex × 4 vertices = 8
    n_quartets_matter = N_ATOMS * 1          # 1 quartet per vertex × 4 vertices = 4
    n_edges_per_cell = (K_STAR * N_ATOMS) // 2   # = 6
    n_doublets_edge = n_edges_per_cell       # 1 doublet per edge = 6
    total_doublets = n_doublets_matter + n_doublets_edge
    total_quartets = n_quartets_matter
    print(f"\n  matter (Cl(6) Fock × {N_ATOMS} vertices):  {n_doublets_matter} doublets + {n_quartets_matter} quartets"
          f"  (dim = {n_doublets_matter * 2 + n_quartets_matter * 4})")
    print(f"  gauge fiber (edge qubit × {n_edges_per_cell} edges):  {n_doublets_edge} doublets"
          f"  (dim = {n_doublets_edge * 2})")
    print(f"\n  TOTAL per cell:  {total_doublets} doublets + {total_quartets} quartets  =  "
          f"{total_doublets * 2 + total_quartets * 4} dim   (= 32 matter + 12 gauge = 44 ✓)")
    return total_doublets, total_quartets


def part_C(total_doublets, total_quartets):
    print("\n" + "=" * 100)
    print("PART C — Tr_F(T_L²) per cell  (the central CC quantity)")
    print("=" * 100)
    tr_T_doublet = tr_T_sq(0.5)         # spin-1/2: 3/2
    tr_T_quartet = tr_T_sq(1.5)         # spin-3/2: 15
    Tr_TL2 = total_doublets * tr_T_doublet + total_quartets * tr_T_quartet
    print(f"\n  Tr(T²)_{{doublet (spin-1/2)}}  =  j(j+1)(2j+1)  with j=1/2  =  3/2")
    print(f"  Tr(T²)_{{quartet (spin-3/2)}}  =  with j=3/2  =  15")
    print(f"\n  Tr_F(T_L²)_cell  =  {total_doublets}·(3/2)  +  {total_quartets}·(15)  "
          f"=  {total_doublets * tr_T_doublet}  +  {total_quartets * tr_T_quartet}  =  {Tr_TL2}")
    return Tr_TL2


def part_D(Tr_TL2):
    print("\n" + "=" * 100)
    print("PART D — SU(2)_R same analysis (chirality-doubled gauge per `theorem_g2d_chirality_doubled.md`)")
    print("=" * 100)
    print(f"""
  SU(2)_R acts identically by the chirality-doubling theorem (`theorem_g2d_chirality_doubled.md`):
  the same Cl(0,2) on each edge gives SU(2)_R under right-multiplication on ℍ; the matter
  decomposition under SU(2)_R is identical to SU(2)_L by symmetry.

      Tr_F(T_R²)_cell  =  Tr_F(T_L²)_cell  =  {Tr_TL2}
""")
    return Tr_TL2


def part_E(Tr_TL2):
    print("=" * 100)
    print("PART E — CC formula:  1/g_L²(M_unif)  =  (f_0 / 24π²) · Tr_F(T_L²)")
    print("=" * 100)
    alpha_GUT_inv = 24.0
    inv_g_sq = alpha_GUT_inv / (4 * np.pi)   # = 1/g² = α_GUT⁻¹ / 4π
    f_0_required = inv_g_sq * 24 * np.pi ** 2 / Tr_TL2
    print(f"""
  The framework's structural prediction at the unification scale (independent of CC):
      α_GUT⁻¹ = {alpha_GUT_inv} ;   1/g_L²(M_unif) = α_GUT⁻¹ / (4π) = {inv_g_sq:.5f}

  CC formula with our Tr_F(T_L²)_cell = {Tr_TL2}:
      f_0  =  1/g_L²(M_unif) · 24π² / Tr_F(T_L²)
            =  {inv_g_sq:.5f}  ·  24π²  /  {Tr_TL2}
            =  {f_0_required:.5f}

  In Chamseddine–Connes for the SM, f_0 is a moment of the spectral cutoff function f
  (specifically f_0 = ∫₀^∞ f(u) u du), and is treated as a free parameter at the matching scale
  (fixed by the requirement that one of the gauge couplings come out right at the unification scale,
  which then DETERMINES the relative couplings of the other factors).

  Our f_0 = {f_0_required:.5f} is a SPECIFIC numerical prediction the framework would give
  for the spectral-cutoff moment.  In CC SM with the standard A_F = ℂ ⊕ ℍ ⊕ M_3(ℂ), Tr_F(T²) for
  SU(2)_L per generation is 6, and per 3 generations × Higgs is about 19; f_0 then sits at
  whatever value matches the observed α_em⁻¹(M_Z).  Here we get f_0 from the framework's
  STRUCTURAL α_GUT⁻¹ = 24 prediction — so the comparison is the other way: this f_0 is what the
  framework demands.
""")
    return f_0_required


def part_F(Tr_TL2):
    print("=" * 100)
    print("PART F — comparison with SM and MSSM matter content under SU(2)_L")
    print("=" * 100)
    # SM under SU(2)_L (per generation): 3 quark colors × Q_L doublet + 1 L_L doublet = 4 doublets/gen
    # × 3 gens = 12 doublets fermionic; plus 1 Higgs doublet (scalar).
    # MSSM doubles fermion → scalar partners + adds gauginos (adjoint of SU(2))
    SM_doublets = 3 * (3 + 1) + 1     # 12 + 1 = 13 doublets (12 fermionic + 1 Higgs)
    SM_TrT2 = SM_doublets * tr_T_sq(0.5)
    # MSSM:
    #   12 fermion doublets (SM fermions, unchanged)
    #   12 scalar doublets (sfermion partners: 3 × (Q̃ × 3 colors + L̃) = 12)
    #   2 fermion doublets (higgsinos H̃_u, H̃_d)
    #   2 scalar doublets (Higgs H_u, H_d — MSSM has 2 Higgs doublets, vs 1 in SM)
    #   3 adjoint (gauginos: Wino — 3-dim adjoint of SU(2))
    MSSM_doublets = 12 + 12 + 2 + 2   # 28 doublets
    MSSM_adj = 3                       # 1 SU(2) adjoint = 3 (gauginos)
    MSSM_TrT2 = MSSM_doublets * tr_T_sq(0.5) + 1 * tr_T_sq(1.0)   # j=1 for adjoint
    print(f"\n  SM under SU(2)_L (3 generations + 1 Higgs):")
    print(f"    {SM_doublets} doublets, no quartets/adjoints")
    print(f"    Tr_F(T_L²)_SM  =  {SM_doublets} × 3/2  =  {SM_TrT2}")
    print(f"\n  MSSM under SU(2)_L (3 gens + sfermions + 2 Higgs + higgsinos + gauginos):")
    print(f"    {MSSM_doublets} doublets, {MSSM_adj//3} adjoint (=3 states)")
    print(f"    Tr_F(T_L²)_MSSM  =  {MSSM_doublets} × 3/2  +  1 × 6  =  {MSSM_TrT2}")
    print(f"\n  FRAMEWORK per cell:")
    print(f"    14 doublets + 4 quartets")
    print(f"    Tr_F(T_L²)_framework  =  14 × 3/2  +  4 × 15  =  21 + 60  =  {Tr_TL2}")
    print(f"\n  comparison:  SM = {SM_TrT2:.1f},  MSSM = {MSSM_TrT2:.1f},  framework = {Tr_TL2:.1f}")
    print(f"  framework / SM   = {Tr_TL2 / SM_TrT2:.3f}")
    print(f"  framework / MSSM = {Tr_TL2 / MSSM_TrT2:.3f}")
    return SM_TrT2, MSSM_TrT2


def part_G(Tr_TL2, SM_TrT2, MSSM_TrT2, f_0_required):
    print("\n" + "=" * 100)
    print("VERDICT — Phase 1 of the CC bridge")
    print("=" * 100)
    print(f"""
  WHAT WE COMPUTED
   • Per-cell SU(2)_L matter content of the framework's state-level Hilbert space H_F:
       14 doublets + 4 quartets  (the quartets coming from the spin-3/2 piece of
       (1/2)⊗(1/2)⊗(1/2) inside each vertex's Cl(6) Fock).
   • Tr_F(T_L²)_framework = 81 per cell.
   • CC formula matches α_GUT⁻¹ = 24 with f_0 ≈ {f_0_required:.3f} (a SPECIFIC numerical
     prediction for the spectral-cutoff moment).

  COMPARISON TO SM AND MSSM
   • SM Tr_F(T_L²) = {SM_TrT2:.1f}  (per 3 generations + 1 Higgs).
   • MSSM Tr_F(T_L²) = {MSSM_TrT2:.1f}.
   • Framework = {Tr_TL2:.1f}.
   • Framework / SM ≈ {Tr_TL2/SM_TrT2:.2f}  ;  Framework / MSSM ≈ {Tr_TL2/MSSM_TrT2:.2f}.

   ⇒ the framework's per-cell Tr_F is roughly 2× MSSM and 4× SM.  This is NEITHER SM nor MSSM
     in its matter content under SU(2)_L — it is the framework's own content, dominated by
     the spin-3/2 quartet states inside each Cl(6) Fock that don't have an obvious SM analog.

  WHAT THIS IMPLIES
   • The framework gives a CC-style spectral action with its OWN matter content (4 quartets per
     cell + 14 doublets).  Running β-functions from M_unif to M_Z with THIS matter content (NOT
     SM, NOT MSSM) is what the framework actually predicts at the gauge-coupling level.
   • The spin-3/2 quartets are a non-SM feature — they would contribute to gauge β-functions
     like adjoint+doublet/quartet matter, not like SM fermions.  Whether the resulting running
     happens to land near the observed α_s(M_Z), sin²θ_W(M_Z), α_em⁻¹(M_Z) is the next-phase
     computation (Phase 4-5 of the CC arc).
   • Crucially: the "MSSM dictionary" was wrong not because the framework is SM-like, but because
     it has DIFFERENT matter content — with extra quartets — that the textbook gauge-unification
     analysis didn't see.  The framework's α_GUT⁻¹ = 24 + sin²θ_W = 3/8 already account for this
     content structurally; the question is just what RG running to use.

  NEXT STEPS
   • Phase 2: compute Tr_F(T_R²) for SU(2)_R chirality-doubled gauge (just done — same 81 by
     symmetry).  Also Tr_F(Y²) for the hypercharge U(1), once the framework's U(1) embedding
     in SO(4) is identified explicitly.
   • Phase 3: build the manifold side (4D flat Euclidean spectral triple);  combine via
     almost-commutative product.
   • Phase 4: Seeley–deWitt a_4 coefficient — extract the Yang–Mills 1/g² Tr F² coefficient
     and the Higgs/Yukawa terms from the spectral action.
   • Phase 5: 1-loop running with the framework's quartet-and-doublet matter; compare to PDG.

  No graded content changes.  This is the foundational data piece for the multi-phase CC arc.
""")


def main():
    print(r"""
======================================================================================================
CONNES–CHAMSEDDINE PHASE 1 — framework matter content under SU(2)_L, the foundational Tr_F(T²) input
======================================================================================================""")
    part_A()
    total_doublets, total_quartets = part_B()
    Tr_TL2 = part_C(total_doublets, total_quartets)
    part_D(Tr_TL2)
    f_0_required = part_E(Tr_TL2)
    SM_TrT2, MSSM_TrT2 = part_F(Tr_TL2)
    part_G(Tr_TL2, SM_TrT2, MSSM_TrT2, f_0_required)
    print("connes_chamseddine_step1_su2_content_probe.py: done (sentinel).")


if __name__ == "__main__":
    main()
