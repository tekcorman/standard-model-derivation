#!/usr/bin/env python3
"""
Substrate-doubling β-coefficient test — unified-story follow-on probe.

CONTEXT:
  - PS→SM Session 2 (Probe 1 today): non-SUSY PS matter content fails PDG α_i(M_Z).
  - 2026-05-10 probe `mssm_matter_content_required.py`: SM + 2HDM both catastrophic;
    only MSSM matter content gives PDG-consistent running. Path D structurally established.
  - R1_1 probe (`R1_1_cl6_fock_su4_PS_decomposition_probe.py`) noted in §"COUNTING
    CONCERN": substrate Cl(6,0) Fock gives 96 fermion states per srs cell vs 48 for
    3 SM gens — a factor-of-2 doubling. Speculatively flagged as "potentially MSSM
    boson/fermion doubling."

QUESTION: under what organization does the substrate's 96-state content reproduce
MSSM β coefficients (b_1, b_2, b_3) = (33/5, 1, -3)?

This probe tests three candidate organizations of the substrate's doubled content:
  H1 — Mirror SM (6 generations of pure SM fermions, no scalars)
  H2 — 3 gen + 3 'sterile gen' with hypercharge-zero (sterile mirror)
  H3 — 3 gen SM + 3 gen 'flipped' (Y → -Y mirror)

We do NOT claim H1/H2/H3 are correct organizations — we test which (if any) reproduces
MSSM β coefficients structurally. The aim is to identify what structural feature beyond
counting must hold for substrate-doubling to be equivalent to MSSM partners.

Method: one-loop β formula b = -(11/3)·C_2(G) + (2/3)·Σ_Weyl T(R) + (1/3)·Σ_scalar T(R).
For each hypothesis, compute (b_1, b_2, b_3) and run α_i to M_Z.
"""

import math

# ============================================================
# FRAMEWORK & PDG INPUTS
# ============================================================
ALPHA_GUT = 1.0/24.0       # framework theorem-grade
M_UNIF = 2.0e16            # GeV
M_Z = 91.1876              # GeV

INV_ALPHA_1_PDG = 59.02
INV_ALPHA_2_PDG = 29.58
INV_ALPHA_3_PDG = 8.48

MSSM = (33.0/5.0, 1.0, -3.0)
SM   = (41.0/10.0, -19.0/6.0, -7.0)


# ============================================================
# β COEFFICIENT FORMULAS
# ============================================================
def b_pure_SM(N_gen, N_higgs):
    """SM β coefficients with N_gen Weyl-fermion generations + N_higgs Higgs doublets.

    Standard one-loop:
      b_3 = -11 + (4/3)·N_gen
      b_2 = -22/3 + (4/3)·N_gen + N_higgs/6
      b_1_GUT = (4/3)·N_gen + N_higgs/10
    """
    b3 = -11.0 + (4.0/3.0) * N_gen
    b2 = -22.0/3.0 + (4.0/3.0) * N_gen + N_higgs/6.0
    b1 = (4.0/3.0) * N_gen + N_higgs/10.0
    return (b1, b2, b3)


def b_sterile_mirror(N_gen_SM, N_gen_sterile, N_higgs):
    """SM gens + 'sterile' gens that are SU(2)×SU(3) gauge singlets (only contribute
    to gravitational anomaly), so they don't enter b_1/b_2/b_3.
    This is the 'maximally cosmologically-quiet' doubling hypothesis."""
    return b_pure_SM(N_gen_SM, N_higgs)


def b_flipped_mirror(N_gen_SM, N_gen_flipped, N_higgs):
    """SM gens + 'flipped' gens where Y → -Y on each multiplet.
    Since β coefficients depend on Y² (not Y), flipped mirror is IDENTICAL to mirror SM
    for β. (Sanity check that confirms b_1 same as 6-gen SM.)"""
    return b_pure_SM(N_gen_SM + N_gen_flipped, N_higgs)


def b_mssm():
    return MSSM


# ============================================================
# RG RUNNING
# ============================================================
def run_inv_alpha(alpha_init, mu_init, mu_final, b):
    return 1.0/alpha_init - (b/(2.0*math.pi)) * math.log(mu_final/mu_init)


def alpha_at_M_Z(b_triple):
    b1, b2, b3 = b_triple
    return (
        run_inv_alpha(ALPHA_GUT, M_UNIF, M_Z, b1),
        run_inv_alpha(ALPHA_GUT, M_UNIF, M_Z, b2),
        run_inv_alpha(ALPHA_GUT, M_UNIF, M_Z, b3),
    )


# ============================================================
# REPORT
# ============================================================
def report():
    print("=" * 78)
    print("  Substrate doubling β-coefficient test — unified-story probe")
    print("=" * 78)

    print(f"\n  Substrate counting (R1_1 verdict):")
    print(f"    Per srs cell: 4 vertices × 8 = 32 fermion states (color factored)")
    print(f"    × 3 colors via B6 C_3 action = 96 colored fermion states/cell")
    print(f"    3 SM gen × 16 states = 48                  (matter content)")
    print(f"    Substrate-to-SM ratio = 96/48 = 2 (doubling)")

    print(f"\n  Framework targets (PDG, GUT-normalized):")
    print(f"    1/α_1(M_Z) = {INV_ALPHA_1_PDG:.3f}, 1/α_2(M_Z) = {INV_ALPHA_2_PDG:.3f}, 1/α_3(M_Z) = {INV_ALPHA_3_PDG:.3f}")
    print(f"    MSSM β:    (b_1, b_2, b_3) = ({MSSM[0]:.3f}, {MSSM[1]:.3f}, {MSSM[2]:.3f})")

    hypotheses = [
        ("SM (3 gen + 1 H)",                b_pure_SM(3, 1),               "baseline; what substrate gives if doubling NOT realized"),
        ("2HDM (3 gen + 2 H)",              b_pure_SM(3, 2),               "framework-derived: 1 PS bidoublet = 2 SM doublets"),
        ("MSSM (3 gen + 2 H + partners)",   b_mssm(),                       "current adopted assumption"),
        ("Mirror SM (6 gen, 1 H)",          b_pure_SM(6, 1),               "doubling H1: pure fermion doubling"),
        ("Mirror SM (6 gen, 2 H)",          b_pure_SM(6, 2),               "H1 + 2HDM Higgs sector"),
        ("Sterile mirror (3+3 sterile, 2 H)", b_sterile_mirror(3, 3, 2),  "H2: doubling is gauge-singlet → cosmologically dark"),
        ("Flipped Y mirror (3+3, 2 H)",     b_flipped_mirror(3, 3, 2),    "H3: Y→-Y mirror = same Y², equivalent to H1"),
    ]

    print(f"\n  {'Hypothesis':<40} {'b_1':>8} {'b_2':>8} {'b_3':>8}    {'1/α_1':>8} {'1/α_2':>8} {'1/α_3':>8}    |Δ_3|")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8}    {'-'*8} {'-'*8} {'-'*8}    {'-'*6}")
    for label, b, note in hypotheses:
        inv = alpha_at_M_Z(b)
        delta_3 = inv[2] - INV_ALPHA_3_PDG
        print(f"  {label:<40} {b[0]:>+8.3f} {b[1]:>+8.3f} {b[2]:>+8.3f}    "
              f"{inv[0]:>8.3f} {inv[1]:>8.3f} {inv[2]:>8.3f}    {delta_3:+.3f}")

    print(f"\n  Notes per hypothesis:")
    for label, b, note in hypotheses:
        print(f"    {label:<40}  {note}")

    # ============================================================
    # ANALYSIS
    # ============================================================
    print("\n" + "=" * 78)
    print("  ANALYSIS")
    print("=" * 78)

    print("""
  KEY OBSERVATIONS:

  (1) 6-gen SM (H1) gives b_2 = -11/3, b_3 = -3  — partial MSSM match!
      But b_1 ≠ MSSM b_1; SM hypercharge sum doubles to ~8.2 vs MSSM 6.6.

  (2) 6-gen 2-Higgs SM matches MSSM b_2 = +1 AND b_3 = -3 EXACTLY, but
      b_1 = 41/5 ≠ 33/5 = MSSM b_1. Hypercharge sum differs.

  (3) Sterile mirror (H2: 3 gens + 3 sterile generations) is IDENTICAL
      to 3-gen SM for β (sterile contributes nothing to gauge β). So
      H2 cannot explain MSSM β-coefficient matching.

  (4) Flipped Y mirror (H3) is identical to mirror SM (Y² doesn't care
      about sign), so same as H1.

  STRUCTURAL CONCLUSION:
    Pure FERMIONIC doubling (96 = 2 × 48 Weyl states) cannot reproduce MSSM
    β coefficients across all three gauge factors. The b_1 sector hypercharge
    sum is structurally different between (6 SM gen + 2 H) and MSSM.

    For substrate doubling to MATCH MSSM, the "extra 48" states would need to
    be organized as MSSM SUSY partners — i.e., 48 = 48 COMPLEX SCALARS at
    specific representations (squarks, sleptons, Higgsinos as scalar-paired
    Weyl) PLUS gauginos. The COUNTING matches MSSM (96 = SM·2 = MSSM matter)
    but the STRUCTURE is what determines β. Counting alone does not.

  WHAT WOULD CLOSE LAYER 5:
    A substrate-derived demonstration that the extra 48 states organize
    as MSSM SUSY partners (squarks at (3,2,1/6), sleptons at (1,2,-1/2),
    Higgsinos at (1,2,±1/2), gauginos for SU(3)/SU(2)/U(1)). This is what
    R1.2-R1.4 was speculatively flagged for in R1_1 verdict — NOT yet closed.

  HONEST READING:
    The substrate-counting result (96 = 2×48) is a NECESSARY condition for
    MSSM-equivalence, but NOT a SUFFICIENT condition. The β-coefficient
    match across all three sectors requires specific structural assignments
    that counting alone doesn't provide. The 2026-05-10 path-D finding
    stands: framework needs MSSM matter content for PDG running; the
    substrate counting is consistent but doesn't yet derive the structure.

  STILL OPEN: Layer 5 substrate-derivation of SUSY-partner structure.
  This is a multi-session research target (R1.2 / R1.3 / R1.4 from R1_1
  verdict). The unified story should acknowledge the structural opening
  but not claim it's closed.
""")
    print("=" * 78)


if __name__ == "__main__":
    report()
