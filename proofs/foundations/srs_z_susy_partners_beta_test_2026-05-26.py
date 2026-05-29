#!/usr/bin/env python3
"""
srs-z as substrate origin of MSSM scalar partners — β coefficient test.

USER HYPOTHESIS (2026-05-26 EOD+5):
  "srs to srs-z walks are ultimately responsible for mass in our framework.
   Is something like that happening here?"

The framework's M_persistence mechanism (predictions/M_persistence.py + W46)
generates the 12 SM fermion masses as the holonomy of self-sustaining L↔R
chirality oscillation on the srs ↔ srs-z double cover. srs-z is the
bipartite Z₂ double cover of srs (gauge_hub_stage0_z2_artin_ihara_2026-05-21).
The multi-axial dark sector theorem (2026-05-24) identifies srs-z as the
dark sector content under R-9 closure.

CANDIDATE STRUCTURAL READING:
  - srs vertex Fock = SM matter (3 SM gens via S1 today's R-C reading)
  - srs-z vertex Fock = MSSM SCALAR SUPERPARTNERS (sfermions: squarks, sleptons)
  - Both contribute to β coefficient running:
      srs Weyl fermions contribute (2/3)·T(R)
      srs-z scalars contribute (1/3)·T(R)
  - Edge sector (Cl(0,2)) gives SU(2)_L + Higgs doublets (existing theorem)
  - Higgsinos come from Cl(0,2) edge structure paired with srs↔srs-z duality
  - Gauginos: hypothesis — from edge sector's "fermion partner" of the gauge
    boson, via srs↔srs-z duality on edges (parallel to vertex matter doubling)

THIS PROBE TESTS the central numerical prediction: does
"srs fermions (2/3) + srs-z scalars (1/3) + Higgs supermultiplet + gauginos"
reproduce MSSM β = (33/5, 1, -3) EXACTLY?

If YES: Path B of `theorem_susy_requirement_scoping.md` (dark-sector
necessity for SUSY) closes structurally. srs-z PROVIDES the SUSY partner
content as theorem-grade substrate primitive.
"""

import math

# ============================================================
# GROUP-THEORY INPUTS
# ============================================================
# Casimirs
C2_SU3 = 3
C2_SU2 = 2
C2_U1 = 0   # Abelian, no adjoint
# Adjoint Dynkin indices for gaugino contribution
T_ADJ_SU3 = 3   # T(adj_SU(3)) = N for SU(N)
T_ADJ_SU2 = 2   # T(adj_SU(2)) = 2


# ============================================================
# PER-GENERATION DYNKIN INDICES (SM matter content)
# ============================================================
# Per SM generation (color INCLUDED), summed T(R) over all Weyl fermions:
#   SU(3)_c: Q (×2 isospin), u_R, d_R contribute T=1/2 each, summed: 2
#   SU(2)_L: L (×3 colors implicit), Q (×3 colors) → T_2 = 2
#   U(1)_Y (GUT-norm): Σ Y_GUT² per gen = 2
T_PER_GEN_SU3 = 2
T_PER_GEN_SU2 = 2
T_PER_GEN_U1_GUT = 2   # GUT-normalized Σ Y² per gen


# Per Higgs doublet (2 components, complex scalar)
T_HIGGS_SU3 = 0    # SU(3) singlet
T_HIGGS_SU2 = 0.5  # T(2_fund) × 1 doublet
T_HIGGS_U1_GUT = 3/10   # (3/5)·(2·(1/4)) GUT-norm

# Per Higgsino doublet (Weyl partners of Higgs)
T_HIGGSINO_SU3 = 0
T_HIGGSINO_SU2 = 0.5
T_HIGGSINO_U1_GUT = 3/10


# ============================================================
# BETA COEFFICIENT COMPUTATION
# ============================================================
def beta_from_content(T_F_SU3, T_F_SU2, T_F_U1,
                       T_S_SU3, T_S_SU2, T_S_U1,
                       gauginos_present=False):
    """
    One-loop β:
      b_i = -(11/3)·C_2(G_i) + (2/3)·Σ_Weyl T_i + (1/3)·Σ_scalar T_i
            + (2/3)·T_adj_gauginos    [gaugino contribution if present]

    Gaugino contribution: each gauge factor's gaugino (1 Weyl Majorana in
    adjoint) contributes (2/3)·T(adj_G) = (2/3)·C_2(G_i) (since T(adj) = C_2
    for any G).
    """
    b_3 = -(11/3)*C2_SU3 + (2/3)*T_F_SU3 + (1/3)*T_S_SU3
    b_2 = -(11/3)*C2_SU2 + (2/3)*T_F_SU2 + (1/3)*T_S_SU2
    b_1 = (2/3)*T_F_U1 + (1/3)*T_S_U1   # U(1) has no gauge self-coupling

    if gauginos_present:
        b_3 += (2/3) * T_ADJ_SU3
        b_2 += (2/3) * T_ADJ_SU2
        # U(1) gaugino (bino) is a singlet — no T(adj) contribution

    return (b_1, b_2, b_3)


# ============================================================
# HYPOTHESES
# ============================================================
def hypothesis_srs_only_SM():
    """srs alone (S1 R-C reading): 3 gens SM matter + 2 Higgs doublets only."""
    T_F_3 = 3 * T_PER_GEN_SU3
    T_F_2 = 3 * T_PER_GEN_SU2
    T_F_1 = 3 * T_PER_GEN_U1_GUT
    T_S_3 = 2 * T_HIGGS_SU3   # 2 Higgs doublets
    T_S_2 = 2 * T_HIGGS_SU2
    T_S_1 = 2 * T_HIGGS_U1_GUT
    return beta_from_content(T_F_3, T_F_2, T_F_1, T_S_3, T_S_2, T_S_1, gauginos_present=False)


def hypothesis_srs_plus_srsz_no_gauginos():
    """srs (Weyl) + srs-z (scalar) for matter, 2 Higgs doublets, NO gauginos."""
    # 3 gens fermions from srs
    T_F_3 = 3 * T_PER_GEN_SU3
    T_F_2 = 3 * T_PER_GEN_SU2
    T_F_1 = 3 * T_PER_GEN_U1_GUT
    # 3 gens scalar partners from srs-z (same group reps, T values identical)
    T_S_3 = 3 * T_PER_GEN_SU3 + 2 * T_HIGGS_SU3
    T_S_2 = 3 * T_PER_GEN_SU2 + 2 * T_HIGGS_SU2
    T_S_1 = 3 * T_PER_GEN_U1_GUT + 2 * T_HIGGS_U1_GUT
    return beta_from_content(T_F_3, T_F_2, T_F_1, T_S_3, T_S_2, T_S_1, gauginos_present=False)


def hypothesis_srs_plus_srsz_plus_higgsinos_plus_gauginos():
    """
    FULL hypothesis: srs (Weyl) + srs-z (scalar) for matter, plus
      - 2 Higgs doublets (scalars)
      - 2 Higgsino doublets (Weyl, from edge-sector srs↔srs-z duality)
      - Gauginos (Weyl in adjoint, from edge-sector srs↔srs-z duality)
    This reproduces MSSM if the partner mechanism works structurally.
    """
    # Matter fermions (srs):
    T_F_3 = 3 * T_PER_GEN_SU3
    T_F_2 = 3 * T_PER_GEN_SU2 + 2 * T_HIGGSINO_SU2  # + 2 Higgsino doublets
    T_F_1 = 3 * T_PER_GEN_U1_GUT + 2 * T_HIGGSINO_U1_GUT
    # Scalar matter (srs-z) + Higgs scalars:
    T_S_3 = 3 * T_PER_GEN_SU3   # squarks
    T_S_2 = 3 * T_PER_GEN_SU2 + 2 * T_HIGGS_SU2   # sfermions + 2 Higgs doublets
    T_S_1 = 3 * T_PER_GEN_U1_GUT + 2 * T_HIGGS_U1_GUT
    return beta_from_content(T_F_3, T_F_2, T_F_1, T_S_3, T_S_2, T_S_1, gauginos_present=True)


MSSM_TARGET = (33/5, 1.0, -3.0)


# ============================================================
# REPORT
# ============================================================
def report():
    print("=" * 78)
    print("  srs / srs-z as substrate origin of MSSM partners — β test")
    print("=" * 78)

    print("\n  Hypothesis: substrate-derived MSSM β coefficients via")
    print("    - srs sector  = 3 SM gens (Weyl fermions, S1 R-C reading)")
    print("    - srs-z sector = 3 gens of SCALAR partners (sfermions, dark)")
    print("    - Cl(0,2) edges = SU(2)_L + Higgs doublets (theorem g2_edge_qubit)")
    print("    - 2 Higgsinos (Weyl) + gauginos (Weyl in adjoint)")
    print("      from edge-sector srs↔srs-z duality")

    print(f"\n  Target: MSSM β = (b_1, b_2, b_3) = ({MSSM_TARGET[0]:.4f}, {MSSM_TARGET[1]:.4f}, {MSSM_TARGET[2]:.4f})")

    cases = [
        ("srs only (3 SM gens + 2H, no partners)", hypothesis_srs_only_SM()),
        ("srs + srs-z (sfermions, NO Higgsinos/gauginos)", hypothesis_srs_plus_srsz_no_gauginos()),
        ("srs + srs-z + Higgsinos + gauginos (full MSSM hypothesis)",
         hypothesis_srs_plus_srsz_plus_higgsinos_plus_gauginos()),
    ]

    print(f"\n  {'Hypothesis':<55} {'b_1':>10} {'b_2':>10} {'b_3':>10}")
    print(f"  {'-'*55} {'-'*10} {'-'*10} {'-'*10}")
    for label, (b1, b2, b3) in cases:
        print(f"  {label:<55} {b1:>+10.4f} {b2:>+10.4f} {b3:>+10.4f}")
    print(f"  {'MSSM target':<55} {MSSM_TARGET[0]:>+10.4f} {MSSM_TARGET[1]:>+10.4f} {MSSM_TARGET[2]:>+10.4f}")

    # Check exact match for full hypothesis
    full = cases[-1][1]
    eps = 1e-10
    matches = (abs(full[0] - MSSM_TARGET[0]) < eps,
               abs(full[1] - MSSM_TARGET[1]) < eps,
               abs(full[2] - MSSM_TARGET[2]) < eps)

    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)

    if all(matches):
        print("""
  >>> FULL HYPOTHESIS REPRODUCES MSSM β COEFFICIENTS EXACTLY <<<

  The substrate's srs ↔ srs-z double cover + edge-sector content gives,
  under the proposed boson/fermion attribution, the EXACT MSSM β-function
  coefficients (b_1, b_2, b_3) = (33/5, 1, -3).

  Structural mapping:
    Visible sector (compressed, above MDL waterline):
      - srs vertices → 3 SM gens of Weyl matter fermions
      - Cl(0,2) edges → SU(2)_L gauge + Higgs doublets

    Hidden sector (dark, srs-z, below MDL waterline):
      - srs-z vertices → 3 gens of COMPLEX SCALAR sfermion partners
        (squarks, sleptons, ν̃_R)
      - Edge-sector srs↔srs-z duality → Higgsinos + gauginos

  This identifies the MSSM SUSY partner sector with the framework's
  EXISTING DARK SECTOR (per multi-axial dark sector theorem, 2026-05-24).
  Layer 5 SUSY closure proceeds via PATH B (dark-sector consistency).

  CONSEQUENCES if validated by full structural derivation:
    - MSSM β coefficients DERIVED from substrate (Layer 5 closes).
    - Framework predictions α_i(M_Z), g_i, M_Z, sin²θ_W, α_s graduate
      from DOMINANT-CONDITIONAL to UNIQUE-THEOREM-GRADE.
    - SUSY scalar partners are STRUCTURALLY DARK (not at colliders),
      explaining LHC non-observation.
    - Framework + LSP-as-dark-matter standard SUSY framing replaced by
      framework-where-ALL-partners-are-dark.
""")
    else:
        print("  Hypothesis does NOT exactly reproduce MSSM β. Mismatches:")
        for i, (got, target, name) in enumerate(zip(full, MSSM_TARGET, ['b_1', 'b_2', 'b_3'])):
            if abs(got - target) > eps:
                print(f"    {name}: got {got:.4f}, expected {target:.4f}, Δ = {got-target:+.4f}")

    print("""
  OPEN STRUCTURAL ELEMENTS (even if numerical match):

  1. WHY does srs contribute as Weyl (2/3·T) but srs-z as scalar (1/3·T)?
     The framework's existing M_persistence uses srs↔srs-z as L↔R chirality
     oscillation, NOT as fermion↔boson promotion. The "boson promotion" of
     srs-z in this hypothesis is a NEW structural claim that needs
     foundational justification beyond just numerical β matching.

     Candidate justifications to investigate:
       (a) MDL waterline mechanism — dark content (srs-z) projects as
           scalar under observer compression, even if substrate states
           are operator-fermionic.
       (b) Galois Z_2 sheet structure — sheet 0 (srs) = matter fermion,
           sheet 1 (srs-z) = scalar via some structural pairing.
       (c) The bipartite double cover's even/odd grading enforces a
           Bose/Fermi alternation between sheets.

  2. Edge-sector duality (gauginos + Higgsinos): edge primitives are
     Cl(0,2) bivectors. Does srs↔srs-z extend to edges naturally? The
     existing theorem_g2_edge_qubit_su2 doesn't discuss this.

  3. Cross-check with M_persistence: if srs-z is scalar-promoted, does
     this break the M_persistence holonomy (which assumes both srs and
     srs-z carry fermion states for L↔R oscillation)?

  These three are the natural next-session targets if numerical β match
  holds. If foundational justifications close, this is theorem-grade
  closure of Layer 5 via Path B (dark sector).
""")
    print("=" * 78)


if __name__ == "__main__":
    report()
