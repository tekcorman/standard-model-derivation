#!/usr/bin/env python3
"""
2026-05-10 evening — structural finding: framework's α_GUT = 1/24 + sin²θ_W(M_unif) = 3/8
theorems STRUCTURALLY REQUIRE MSSM-like matter content for cluster predictions to match PDG.

Method: run α_i(M_Z) from α_GUT = 1/24 at M_unif under three matter-content scenarios.
Compare to PDG.

Three scenarios:
  - SM    (3 gen + 1 Higgs):  b_1 = 41/10, b_2 = -19/6, b_3 = -7
  - 2HDM  (3 gen + 2 Higgs):  b_1 = 21/5,  b_2 = -3,    b_3 = -7
  - MSSM  (3 gen + 2 Higgs + SUSY partners): b_1 = 33/5, b_2 = +1, b_3 = -3

The framework's CURRENTLY DERIVED matter content is 3 generations + 2 Higgs doublets:
  - 3 generations: theorem-grade via Galois Z_3 of M^α ⊂ M ⋊ Z_3
  - 2 Higgs doublets: theorem-grade via PS (1, 2, 2) bidoublet (2026-05-05)
  - SUSY partners: NOT YET DERIVED, currently ADOPTED (per
    docs/framework/framework_architecture.md Layer 5 + Sprint 11 B7.6 Thread A)

This probe verifies: 2HDM running gives catastrophically wrong PDG predictions, while
MSSM running gives ~1-3% match. Therefore Layer 5 SUSY closure is REQUIRED for the
framework's existing α_GUT and sin²θ_W theorems to be consistent with PDG.

This is the structural finding behind the 2026-05-10 audit ledger downgrade
(Rows P63-P70 to DOMINANT-CONDITIONAL on Layer 5 SUSY closure).
"""
from __future__ import annotations
import math


def banner(title):
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def run_one_loop(alpha_GUT, M_unif, M_Z, b1, b2, b3):
    """One-loop RG from α_GUT at M_unif down to M_Z."""
    log_ratio = math.log(M_Z / M_unif)
    inv_a1 = 1/alpha_GUT - (b1/(2*math.pi)) * log_ratio
    inv_a2 = 1/alpha_GUT - (b2/(2*math.pi)) * log_ratio
    inv_a3 = 1/alpha_GUT - (b3/(2*math.pi)) * log_ratio
    a1 = 1/inv_a1
    a2 = 1/inv_a2
    a3 = 1/inv_a3
    aY = (3/5) * a1
    sin2_W = aY/(a2 + aY)
    aEM = a2 * sin2_W
    return {
        'inv_a1': inv_a1, 'inv_a2': inv_a2, 'inv_a3': inv_a3,
        'sin2_W': sin2_W, 'aEM': aEM, 'a3': a3,
    }


def main():
    banner("2026-05-10 — MSSM matter content REQUIRED by framework")

    print("""
  Inputs (framework theorem-grade):
    α_GUT = 1/24                    (single-node Fock × edges, theorem-grade)
    sin²θ_W(M_unif) = 3/8           (GQW trace identity, theorem-grade)
    M_unif = 2×10¹⁶ GeV             (lattice scale, theorem-grade-conditional)
    M_Z = 91.1876 GeV                (PDG external)

  Question: which matter content makes one-loop RG from α_GUT at M_unif
  match PDG α_EM(M_Z), sin²θ_W(M_Z), α_s(M_Z)?
""")

    alpha_GUT = 1.0 / 24.0
    M_unif = 2.0e16
    M_Z = 91.1876

    # PDG 2024
    pdg = {
        'inv_a1': 59.0,
        'inv_a2': 29.6,
        'inv_a3': 8.5,
        'sin2_W': 0.23121,
        'aEM': 1/127.944,
        'a3': 0.1180,
    }

    scenarios = [
        ('SM (3 gen + 1 Higgs)', 41/10, -19/6, -7, 'no Higgs doublet partners'),
        ('2HDM (3 gen + 2 Higgs) — framework derived matter',
         21/5, -3, -7, 'second Higgs doublet adds 1/10 to b_1, 1/6 to b_2'),
        ('MSSM (3 gen + 2 Higgs + SUSY) — required adoption',
         33/5, 1, -3, 'SUSY partners change all three β-coefficients drastically'),
    ]

    for name, b1, b2, b3, note in scenarios:
        print(f"\n  === {name} ===")
        print(f"    β-coefficients: b_1 = {b1}, b_2 = {b2}, b_3 = {b3}")
        print(f"    Note: {note}")
        result = run_one_loop(alpha_GUT, M_unif, M_Z, b1, b2, b3)
        print(f"    1/α_1(M_Z)  = {result['inv_a1']:.3f}    PDG {pdg['inv_a1']:.3f}    "
              f"dev {(result['inv_a1'] - pdg['inv_a1'])/pdg['inv_a1']*100:+.2f}%")
        print(f"    1/α_2(M_Z)  = {result['inv_a2']:.3f}    PDG {pdg['inv_a2']:.3f}    "
              f"dev {(result['inv_a2'] - pdg['inv_a2'])/pdg['inv_a2']*100:+.2f}%")
        print(f"    1/α_3(M_Z)  = {result['inv_a3']:.3f}    PDG {pdg['inv_a3']:.3f}    "
              f"dev {(result['inv_a3'] - pdg['inv_a3'])/pdg['inv_a3']*100:+.2f}%")
        print(f"    sin²θ_W(MZ) = {result['sin2_W']:.5f}    PDG {pdg['sin2_W']:.5f}    "
              f"dev {(result['sin2_W'] - pdg['sin2_W'])/pdg['sin2_W']*100:+.2f}%")
        print(f"    α_EM(M_Z)   = 1/{1/result['aEM']:.3f}    PDG 1/{1/pdg['aEM']:.3f}    "
              f"dev {(result['aEM'] - pdg['aEM'])/pdg['aEM']*100:+.2f}%")
        print(f"    α_s(M_Z)    = {result['a3']:.4f}    PDG {pdg['a3']:.4f}    "
              f"dev {(result['a3'] - pdg['a3'])/pdg['a3']*100:+.2f}%")
        if result['a3'] < 0:
            print(f"    ⚠ α_s NEGATIVE — asymptotic non-freedom, structurally inconsistent")

    print()
    banner("Diagnostic verdict")
    print("""
  RESULT:
    SM running:    α_s comes out NEGATIVE (asymptotic non-freedom).
    2HDM running:  α_s comes out NEGATIVE (same problem).
    MSSM running:  α_s = +0.121, matches PDG 0.118 within 3%.

  The framework's α_GUT = 1/24 + sin²θ_W(M_unif) = 3/8 STRUCTURALLY REQUIRE
  MSSM-like matter content (3 gen + 2 Higgs + SUSY partners) for cluster
  predictions to match PDG.

  With the matter content the framework currently derives at theorem-grade
  (3 generations + 2 Higgs doublets, no SUSY), one-loop RG from α_GUT gives
  catastrophically wrong predictions — α_s is negative, sin²θ_W is 0.097
  not 0.231.

  Therefore: SUSY partners (gauginos, sfermions, higgsinos, gravitino) MUST
  exist in the framework's spectrum. Per docs/framework/framework_architecture.md:
    Line 89: 'Why SUSY is non-optional in this framework'
    Line 144: 'SUSY is framework-required, not assumed. If Sprint 11 B7.6
              closes, this becomes a theorem rather than an adopted result.'

  Sprint 11 B7.6 Thread A is the open chain that would close this. Until
  then, MSSM matter content is ADOPTED (not derived); cluster rows P63-P70
  carry DOMINANT-CONDITIONAL on Layer 5 SUSY closure.
""")


if __name__ == "__main__":
    main()
