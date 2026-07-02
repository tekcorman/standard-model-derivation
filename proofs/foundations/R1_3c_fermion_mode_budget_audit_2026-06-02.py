#!/usr/bin/env python3
"""
R1_3c_fermion_mode_budget_audit_2026-06-02.py
=============================================
RUN-TO-GROUND: is the gaugino fermion LATENT in A1's existing structure, or
does it genuinely require new degrees of freedom?

Before naming any new axiom ("fermionize the 1-skeleton" / "corner toggles"),
this probe closes EVERY escape hatch by which the gaugino+higgsino fermions
could already be present in the substrate without new DOF.

THE FERMION-STATE BUDGET A1 GENERATES (from theorem_car_local_jordan_wigner):
  • A1 puts ONE binary toggle on each directed edge mode.
  • Jordan-Wigner assembles the 3 edge modes incident to a vertex into Cl(6),
    Fock = Λ•(ℂ³), dim 8, AT EACH VERTEX (the 0-skeleton).
  • # independent FERMIONIC (single-particle) modes = # edge modes. There is
    no second, independent fermion mode anywhere.
  • walker-matter unification: 12 directed-edge modes × 4 Ramanujan saddles =
    48 = exactly the 48 SM Weyl spinors (3 gen × 16). 48↔48 SATURATION.

ESCAPE HATCHES TESTED (each must be closed to conclude "new DOF required"):
  (H1) Are the gauge generators (adjoint) already fermionic STATES we can read
       as gauginos?  → They are fermion BILINEARS c_i† c_j (number-conserving
       operators on the matter Fock), NOT states. Verified: the 15 SU(4)
       bivectors preserve fermion parity (map even→even, odd→odd) → operators.
  (H2) Do the higher Fock sectors Λ²,Λ³ give spare fermion slots? → No: the
       full 8-dim Fock Λ⁰⊕Λ¹⊕Λ²⊕Λ³ is ALREADY the matter 4+4̄ (all consumed).
  (H3) Do corners / 2-paths (where the Hashimoto dynamics lives) give new
       fermion modes? → No: a corner operator is a PRODUCT T_e T_e' of existing
       toggles = an even-grade element of the SAME algebra, not a new odd
       (fermionic) generator. Reassembling the same toggles at line-graph nodes
       gives an ISOMORPHIC CAR algebra (ordering is a gauge), not a new sector.
  (H4) Could the gaugino live in the DARK sector (outside Cl(6), the ths/srs-z
       bipartite partner where sfermions live)? → No: that sector is BIPARTITE
       → no χ̃ walk → SCALAR (Lemma B). It cannot host a chiral FERMION.

If H1-H4 all close, the gaugino is NOT latent: it needs a genuinely new
fermion mode per edge (= corner/2-cell toggles), i.e. a new axiom. The probe
then states the HONEST FORK without accepting that axiom.

REUSE: R1_1 (Cl(6) γ's, bivectors); substrate_selection_theorem (Lemma B).
"""

import sys
from pathlib import Path
from fractions import Fraction as F

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from R1_1_cl6_fock_su4_PS_decomposition_probe import build_gamma, bivector, TOL
from substrate_selection_theorem import quotient_bipartite

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def section(t):
    print("\n" + "=" * 92 + f"\n {t}\n" + "=" * 92)


def main():
    G = build_gamma()

    section("BUDGET — the fermion-state DOF A1 generates")
    print("  A1: one binary toggle per directed edge mode.")
    print("  srs cell: 4 vertices, 6 undirected = 12 directed edge modes.")
    print("  JW (CAR theorem): 3 edge modes per vertex → Cl(6), Fock Λ•(ℂ³)=8, AT VERTICES.")
    print("  # independent fermionic single-particle modes = # edge modes = 12 (per cell).")
    print("  walker-matter unification: 12 modes × 4 Ramanujan saddles = 48 = SM Weyl spinors.")
    print("  → 48 ↔ 48 SATURATION: every fermion-state slot A1 makes is already MATTER.")

    section("H1 — Are the gauge generators (adjoint) fermionic STATES (= gauginos)?")
    # The 15 SU(4)_PS generators are the bivectors M_ab = (1/2i)[γ_a,γ_b]; in fermion
    # terms these are bilinears c_i† c_j. A bilinear is number-conserving → preserves
    # fermion parity (-1)^N = Γ_7. Test: every bivector commutes with Γ_7.
    G7 = -1j * G[1] @ G[2] @ G[3] @ G[4] @ G[5] @ G[6]   # fermion parity / chirality
    pairs = [(a, b) for a in range(1, 7) for b in range(a + 1, 7)]
    parity_preserving = all(
        np.allclose(bivector(G, a, b) @ G7 - G7 @ bivector(G, a, b), 0, atol=TOL)
        for (a, b) in pairs
    )
    # contrast: a creation operator γ_a (would-be new mode) ANTI-commutes with parity
    creation_flips = all(np.allclose(G[a] @ G7 + G7 @ G[a], 0, atol=TOL) for a in range(1, 7))
    print(f"  15 gauge bivectors all commute with parity Γ_7 (= number-conserving OPERATORS): {parity_preserving}")
    print(f"  single γ_a (a creation/mode operator) ANTI-commutes with Γ_7 (would add/remove a fermion): {creation_flips}")
    assert parity_preserving and creation_flips
    print("  → the adjoint = fermion BILINEARS = operators ON the matter Fock, NOT states.")
    print("    A gaugino requires promoting these operators to STATES → a NEW Fock. H1 closed.")

    section("H2 — Do higher Fock sectors Λ²,Λ³ give spare fermion slots?")
    eig = np.round(np.linalg.eigvalsh(G7)).astype(int)
    even, odd = int((eig == 1).sum()), int((eig == -1).sum())
    print(f"  vertex Fock Λ•(ℂ³) dims: Λ⁰,Λ¹,Λ²,Λ³ = 1,3,3,1 (total 8).")
    print(f"  parity even (Λ⁰⊕Λ²)= {even}  →  SU(4) 4 ;  parity odd (Λ¹⊕Λ³)= {odd}  →  4̄.")
    print(f"  the full 8 = 4 + 4̄ = the matter species per vertex (R1.1). ALL consumed.")
    assert (even, odd) == (4, 4)
    print("  → no spare Fock sector. H2 closed.")

    section("H3 — Do corners / 2-paths give NEW fermion modes? (the Hashimoto carrier)")
    # A 'corner' operator is a product of two existing edge operators. In fermion terms
    # c_i c_j (or c_i† c_j): even grade. It is NOT a new anticommuting generator.
    # Demonstrate: γ_1 γ_2 (a corner / 2-path) commutes with parity → even → operator.
    corner = G[1] @ G[2]
    corner_even = np.allclose(corner @ G7 - G7 @ corner, 0, atol=TOL)
    print(f"  corner operator γ_1γ_2 (= T_e∘T_e' product) commutes with parity (even-grade): {corner_even}")
    assert corner_even
    print("  → a corner is a PRODUCT of existing toggles (even-grade element of the SAME")
    print("    CAR algebra), not a new odd generator. # fermion modes stays = # edges.")
    print("  Reassembling the same toggles at line-graph nodes (different JW ordering)")
    print("  yields an ISOMORPHIC CAR algebra (the CAR theorem: ordering is a gauge) —")
    print("  the same operators, not an independent second fermion sector. H3 closed.")

    section("H4 — Could the gaugino live in the DARK sector (ths/srs-z partner)?")
    for net in ('ths', 'srs-z'):
        nq, bip, tri = quotient_bipartite(net)
        print(f"  {net}: quotient |V|={nq}, bipartite={bip} → "
              f"{'SCALAR (no χ̃ walk, Lemma B)' if bip else 'fermion-capable'}")
        assert bip
    print("  → the dark/partner sector is BIPARTITE → it carries SCALARS (sfermions),")
    print("    not chiral fermions. It cannot host the gaugino. H4 closed.")

    section("VERDICT — run to ground")
    print("""\
  ALL FOUR ESCAPE HATCHES CLOSED. The gaugino is NOT latent in A1:

   • Every fermion-STATE slot A1 generates is consumed by matter (48↔48).
   • The gauge adjoint is fermion BILINEARS (operators on the matter Fock),
     not states (H1).
   • The full vertex Fock is already 4+4̄ matter; no spare sector (H2).
   • Corners/2-paths are products of existing toggles — even-grade operators,
     not new fermion modes; reassembly gives the same CAR algebra (H3).
   • The dark partner sector is bipartite → scalar; cannot host a fermion (H4).

  Therefore a gaugino REQUIRES a genuinely new fermionic mode per edge — a
  second toggle layer on corners/2-cells (promoting the substrate graph to a
  2-complex). It is NOT derivable from A1. No new axiom is adopted here.

  THE HONEST FORK (stated, not resolved):
   (A) ADOPT the new fermion-mode layer (corner/2-cell toggles) → gauginos
       appear on the (non-bipartite) line graph → supply (0,4/3,2); with the
       analogous Higgs fermionization, the substrate β → MSSM (33/5,1,−3) and
       α_GUT⁻¹=24 graduates. Cost: ONE new structural axiom, sharply named.
   (B) DECLINE it → the substrate's honest 1-loop content is matter fermions
       (srs) + scalar partners (dark, bipartite) + gauge bosons (edges), with
       NO gauginos/higgsinos → b = (31/5, −1, −5). Then MSSM / α_GUT⁻¹=24 is an
       OVER-ADOPTION, and (31/5,−1,−5) is the framework's actual prediction —
       a falsifiable, distinct gauge-unification statement to test vs data.

  This is the precise place the keystone rests: NOT 'is MSSM derivable?' but
  'does the substrate carry a second fermion-mode layer (corner toggles)?'.
  The framework's own 48↔48 saturation + bilinear-gauge + bipartite-dark
  structure make the absence of that layer the DEFAULT; its presence is the
  thing that must be independently motivated, derived, or measured (via which
  of b=(31/5,−1,−5) vs MSSM the running actually follows).
""")
    return True


if __name__ == "__main__":
    main()
    raise SystemExit(0)
