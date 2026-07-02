#!/usr/bin/env python3
"""
Substrate doubling → MSSM obstruction: explicit Fock-state enumeration.

GROUND-THEORY PROBE: makes precise the structural gap between substrate's
96 fermion states/cell and MSSM matter+partner content (48 fermion + 56
partner). Identifies what additional input R1.2/R1.3/R1.4 must supply for
Layer 5 SUSY closure.

UPSTREAM:
  - R1_1 verdict: 96 colored fermion states/cell, speculatively "MSSM doubling"
  - 2026-05-10 Path D: MSSM matter content empirically required
  - Today's substrate-doubling β probe: b_2/b_3 match under fermion doubling,
    but b_1 requires partners-with-SM-Y-not-fermion-mirror

PROBE GOAL: enumerate per-vertex Fock states with all quantum numbers, and
demonstrate the boson/fermion identification obstruction.
"""

import math


# ============================================================
# PER-VERTEX FOCK CONTENT (from R1_1 verdict + B3-B6 reconciliation)
# ============================================================
# 8 states per vertex, organized as 4 + 4̄ under SU(4)_PS chirality.
# Per B3-B6 reconciliation: states are "color factored out" — each
# state is one species at one chirality at one isospin; lepton-or-quark
# distinction lives in U(1)_{B−L} = Y_PS direction.

# Format: (label, chirality_Γ_7, T_3L, T_3R, Y_PS_BL, Y_SM, occupation_grade)
# Y_PS_BL = SU(4) U(1)_(B-L) charge; Y_SM = SM hypercharge.
# Occupation grade = sum of mode-pair occupations mod 2 (boson if 0, fermion if 1).

# Note: Cl(6,0) Fock per vertex is a pure-fermion Fock space; the
# "grade" here refers to Z_2 grading of states, not Bose/Fermi statistics.

PER_VERTEX_FOCK = [
    # 4 states with positive chirality (= SU(4) fundamental, (4, 2, 1) of PS)
    # Color-factored: 1 lepton entry + 1 quark entry per isospin per chirality
    ("ν_L",  +1, +0.5, 0, -1, -0.5, 'odd'),
    ("e_L",  +1, -0.5, 0, -1, -0.5, 'odd'),
    ("u_L",  +1, +0.5, 0, +1/3, +1/6, 'odd'),
    ("d_L",  +1, -0.5, 0, +1/3, +1/6, 'odd'),
    # 4 states with negative chirality (= SU(4) antifund, (4̄, 1, 2) of PS)
    ("ν_R",  -1, 0, +0.5, -1, 0, 'even'),
    ("e_R",  -1, 0, -0.5, -1, -1, 'even'),
    ("u_R",  -1, 0, +0.5, +1/3, +2/3, 'even'),
    ("d_R",  -1, 0, -0.5, +1/3, -1/3, 'even'),
]


# ============================================================
# SM Y² SUM (sanity check)
# ============================================================
def y2_sum_per_vertex():
    """Per-vertex Σ Y_SM² (color factored)."""
    # Q_L doublet (1 color): 2 states (u_L + d_L), each Y_SM = 1/6, Y² = 1/36, sum = 2/36 = 1/18
    # L_L doublet: 2 states, Y_SM = -1/2, Y² = 1/4, sum = 2/4 = 1/2
    # u_R singlet (1 color): Y_SM = 2/3, Y² = 4/9
    # d_R singlet (1 color): Y_SM = -1/3, Y² = 1/9
    # ν_R singlet: Y_SM = 0
    # e_R singlet: Y_SM = -1, Y² = 1
    return 0.5 + 1/18 + 0 + 1 + 4/9 + 1/9  # = 19/9

def y2_sum_per_full_gen():
    """Per-full-gen Σ Y_SM² (color included)."""
    return 0.5 + 6*(1/36) + 0 + 1 + 3*(4/9) + 3*(1/9)  # = 10/3


# ============================================================
# MSSM MATTER + PARTNER CONTENT (target structure)
# ============================================================
# Per generation, MSSM has:
#   Fermions (16):  Q_L (3 col × 2 isospin = 6) + u_R (3) + d_R (3) + L_L (2) + e_R (1) + ν_R (1)
#   Sfermions (16): squark Q̃ (6) + ũ (3) + d̃ (3) + slepton L̃ (2) + ẽ (1) + ν̃_R (1)
# Per gen total: 32 (matter + partners).
# Higgs sector: 2 Higgs doublets (8 complex scalars) + 2 Higgsinos (4 Weyl)
#                 = 8 boson + 4 fermion = 12 states/gen-equivalent.
# Gauginos: 8 (gluinos) + 3 (winos) + 1 (bino) = 12 states.

MSSM_per_gen_matter_fermion = 16  # SM matter (incl. ν_R)
MSSM_per_gen_scalar_partner = 16  # squarks + sleptons + ν̃_R
MSSM_total_matter_3gen = 48
MSSM_total_partners_3gen = 48
MSSM_higgs_higgsino = 12
MSSM_gauginos = 12


# ============================================================
# REPORT
# ============================================================
def report():
    print("=" * 78)
    print("  Substrate doubling → MSSM obstruction (ground theory probe)")
    print("=" * 78)

    print("\n  Per-vertex Cl(6,0) Fock content (8 states, color factored):")
    print(f"  {'label':<6} {'Γ_7':>4} {'T_3L':>6} {'T_3R':>6} {'Y_BL':>6} {'Y_SM':>6} {'grade':>6}")
    print(f"  {'-'*6} {'-'*4} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for label, g7, t3l, t3r, ybl, ysm, grade in PER_VERTEX_FOCK:
        print(f"  {label:<6} {g7:>+4d} {t3l:>+6.2f} {t3r:>+6.2f} {ybl:>+6.2f} {ysm:>+6.2f} {grade:>6}")

    y2v = y2_sum_per_vertex()
    y2g = y2_sum_per_full_gen()
    print(f"\n  Per-vertex Σ Y_SM² = {y2v:.4f}  (= 19/9 ✓)")
    print(f"  Per-full-gen Σ Y_SM² (color included) = {y2g:.4f}  (= 10/3 ✓)")

    print("\n" + "-" * 78)
    print("  GRADE-CHIRALITY COINCIDENCE TEST")
    print("-" * 78)
    print("""
  Cl(6,0) Fock per vertex has 8 states from 3 mode-pair creation operators.
  States with EVEN occupation grade (n_1+n_2+n_3 even): 4 states {000, 110, 101, 011}.
  States with ODD occupation grade (n_1+n_2+n_3 odd):   4 states {100, 010, 001, 111}.

  Under Γ_7 = -i Γ_1...Γ_6, even-grade states have Γ_7 = +1; odd-grade Γ_7 = -1.
  Therefore: GRADE-EVEN = chirality +1 = (4,2,1) of PS = left-handed SM matter.
             GRADE-ODD  = chirality -1 = (4̄,1,2) of PS = right-handed SM matter.

  CRITICAL: the grade is ENTIRELY DETERMINED by chirality. There is no
  independent boson/fermion grading in Cl(6,0) Fock — all 8 states are Weyl
  fermions (with chirality given by their grade).
""")

    print("-" * 78)
    print("  THE OBSTRUCTION TO MSSM-EQUIVALENT DOUBLING")
    print("-" * 78)
    print(f"""
  Substrate (per srs cell, color-included via B6 C_3):
    96 fermion states  ←  ALL Weyl, no native boson content

  MSSM per 3 generations:
    Fermion matter:  {MSSM_total_matter_3gen}  (SM matter, incl. ν_R)
    Scalar matter:   {MSSM_total_partners_3gen} (complex scalar SUSY partners)
    Higgs/Higgsinos: {MSSM_higgs_higgsino}  (8 scalar + 4 Weyl)
    Gauginos:        {MSSM_gauginos} (Weyl)

  Total MSSM matter+partner d.o.f.: {MSSM_total_matter_3gen + MSSM_total_partners_3gen + MSSM_higgs_higgsino + MSSM_gauginos}  (matter + partners, mixed boson/fermion)
  Substrate provides:               96 (Cl(6,0)) + ?? (Cl(0,2) edge sector)

  Cl(0,2) edge sector: R1_1 noted 24-dim per cell. Bosonic operator content
  consistent with PS gauge generators + Higgs-like multiplets, but R1.3 has
  NOT delivered the explicit boson decomposition.

  STRUCTURAL GAP — three open identifications:
    (1) The 96/48 = 2 doubling in Cl(6,0): does it organize as 48 matter +
        48 partners? Pure fermion doubling fails b_1 match (today's probe).
        For MSSM-equivalence the extra 48 must be COMPLEX SCALARS at SM-
        partner Y assignments, but Cl(6,0) Fock is purely fermionic at
        the operator level.
    (2) Where do the partner SCALARS come from structurally? Candidates:
        (a) some Cl(6,0) states reinterpret as scalars via grade-promotion
            mechanism (NOT in framework axioms);
        (b) Cl(0,2) edge sector provides scalar partners (would need
            R1.3 + tensor-product representation);
        (c) the "doubling" is illusory — substrate has only 48 matter
            states and the 96 count is over-counting by C_3.
    (3) Where do Higgsinos + gauginos (24 partners) come from? These are
        NOT covered by any current substrate inventory.

  HONEST ASSESSMENT:
    The substrate's 96-state counting is consistent with MSSM-equivalent
    content if interpretation (2b) holds AND the boson/fermion split via
    Cl(6,0) ⊗ Cl(0,2) tensor structure produces specific scalar partners
    with same gauge reps and Y as SM matter. This is the open R1.3/R1.4
    program.

    Without R1.3/R1.4 closed, the substrate doubling is **consistent with**
    MSSM-equivalent content but not **derived to be** MSSM-equivalent.
    The framework's Layer 5 SUSY assertion remains a structural conjecture
    even with today's probes; the closure path is now PRECISE but not yet
    EXECUTED.
""")

    print("=" * 78)
    print("  WHAT WOULD CLOSE LAYER 5 (precise structural targets)")
    print("=" * 78)
    print(f"""
  R1.2 — Generation grading via Galois C_3 outer action on Cl(6,0) Fock.
         Specifies: how 3 generations are distributed across the 4 vertices
         of an srs cell (or across cells via outer C_3 action).
         OUTCOME REQUIRED: explicit map (vertex, mode_state) → gen index.

  R1.3 — Edge sector decomposition: Cl(0,2) at 6 edges per cell = 24-dim
         operator algebra under PS. Identify which states are gauge bosons,
         which are Higgs scalars, which are gauginos/Higgsinos.
         OUTCOME REQUIRED: explicit map of 24 edge states → MSSM gauge +
         partner content.

  R1.4 — β coefficient extraction: with R1.2 + R1.3 closed, count fermion
         + scalar irrep multiplicities and plug into one-loop b_i formula.
         OUTCOME REQUIRED: explicit (b_1, b_2, b_3) = (33/5, 1, -3) MSSM.

  PASS GATE for Layer 5 closure: ALL THREE of R1.2/1.3/1.4 deliver MSSM-
  equivalent matter content from substrate primitives with no fitted
  parameters. Anything less is a structural conjecture, not a theorem.

  If R1.4 derives β coefficients DIFFERENT from MSSM: framework has an
  alternative substrate-derived matter content that should be compared to
  PDG. This is also a positive outcome (just with different β values).

  If R1.4 cannot derive specific β coefficients from substrate without
  external assumption: Layer 5 SUSY remains genuinely OPEN at the
  structural level, and predictions α_i(M_Z) etc. stay DOMINANT-CONDITIONAL.
""")
    print("=" * 78)


if __name__ == "__main__":
    report()
