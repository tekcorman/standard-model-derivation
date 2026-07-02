#!/usr/bin/env python3
"""
σ_S orbit-to-B3-species map — does σ_S label generations or colors?

CONTEXT
-------
`proofs/foundations/matching_brauer_weyl_sigma.py` established that σ_S
permutes the 8 matching-basis weight states of S as 1 + 1 + 3 + 3:

  2 fixed points: |+,+,+⟩_M, |-,-,-⟩_M
  2 3-orbits:    {|+,+,-⟩, |+,-,+⟩, |-,+,+⟩}_M (single negative)
                 {|+,-,-⟩, |-,+,-⟩, |-,-,+⟩}_M (single positive)

This is suggestive of "3 + 3" generation structure, BUT the matching-basis
weight states are eigenstates of the matching Cartans (T_M1, T_M2, T_M3)
— NOT the B3 species Cartans (T_L, T_R, Y_PS) which assign physical
SM labels (ν, e, u, d) × (L, R).

QUESTION
--------
Map each matching-basis weight to its B3 species content. Specifically,
for each matching weight |e₁,e₂,e₃⟩_M, compute the expectation values
of (T_L, T_R, Y_PS, Γ_7) and identify which physical species it
corresponds to (or whether it's a superposition of multiple species).

Then interpret σ_S's 1+1+3+3 orbit structure in physical-species terms:

  Hypothesis A (σ_S = generation):  3-orbits = generation triplets of a
                                    fixed species. Each 3-orbit cycles
                                    through 3 mass-eigenstates of one
                                    species (e.g., {ν₁, ν₂, ν₃}).

  Hypothesis B (σ_S = color):       3-orbits = color triplets of a fixed
                                    quark species. Same physical species
                                    in 3 different colors. (Same as
                                    C₃_body but a different element of
                                    SU(3)_c maximal torus.)

  Hypothesis C (σ_S = mixed):       3-orbits cross species/chirality
                                    boundaries — not a clean SM label.

VERDICT (filled by run output)
------------------------------
[Reported by run.]

Run with:
    PYTHONPATH=. python3 proofs/foundations/sigma_S_orbit_to_species.py

Upstream:
    proofs/foundations/matching_brauer_weyl_sigma.py
    docs/framework/B3_B6_reconciliation.md (B3 species labels in standard basis)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
from numpy import linalg as la

from proofs.foundations.matching_brauer_weyl_sigma import (
    brauer_weyl_gammas,
    bivector,
    hermitian_cartan,
    sigma_permutation_on_gammas,
    build_sigma_S,
    simultaneous_eigenbasis,
)


TOL = 1e-8


def section(s):
    print()
    print(s)
    print("-" * 76)


def main():
    print("=" * 76)
    print("σ_S orbit-to-B3-species map — generation vs color vs mixed?")
    print("=" * 76)

    # --- Build infrastructure -----------------------------------------------
    Gs = brauer_weyl_gammas()

    # Standard B3 Cartans (T_1, T_2, T_3) on the spinor S
    T_1 = hermitian_cartan(Gs[0], Gs[1])    # = Γ_1 Γ_2 / 2i
    T_2 = hermitian_cartan(Gs[2], Gs[3])    # = Γ_3 Γ_4 / 2i
    T_3 = hermitian_cartan(Gs[4], Gs[5])    # = Γ_5 Γ_6 / 2i

    # Matching Cartans (T_M1, T_M2, T_M3)
    # Per MATCHING_LABELING:
    #   Γ_1 ↔ e_03,  Γ_2 ↔ e_12  → matching pair 1
    #   Γ_3 ↔ e_01,  Γ_4 ↔ e_23  → matching pair 2
    #   Γ_5 ↔ e_02,  Γ_6 ↔ e_13  → matching pair 3
    # T_Mk = Γ_(2k-1) · Γ_(2k) / 2i (same as standard Cartans, but the labels
    # are interpreted as matching pairs of K_4 edges, not lex-pair edges.)
    # Numerically: T_M1 = T_1, T_M2 = T_2, T_M3 = T_3 (same Cartan structure
    # in the same Brauer-Weyl realization!) — the "matching basis" terminology
    # is about which K_4 edge each Γ_a is identified with, not about the
    # numerical Cartan operators.
    #
    # So in our explicit Cl(6,0) realization, the matching basis weights are
    # EQUAL to the standard basis weights — they're both eigenstates of
    # (Γ_1Γ_2/2i, Γ_3Γ_4/2i, Γ_5Γ_6/2i). The MATCHING vs STANDARD distinction
    # only enters in the K_4 edge-labeling and hence in σ_S vs C₃_body
    # construction.
    #
    # Conclusion: the matching-basis weight states ARE the standard-basis
    # weight states (as 8 vectors in C^8). σ_S permutes them as 1+1+3+3
    # in this same basis. So we just need to map (e_1, e_2, e_3) weight
    # labels to B3 species labels.

    # B3 PS Cartans:
    T_L = T_1 + T_2          # = (Γ_1Γ_2 + Γ_3Γ_4)/2i
    T_R = T_1 - T_2          # = (Γ_1Γ_2 - Γ_3Γ_4)/2i
    Y_PS = T_3               # = Γ_5Γ_6 / 2i  (B-L generator in PS embedding)

    # Chirality
    G7 = -1j * Gs[0] @ Gs[1] @ Gs[2] @ Gs[3] @ Gs[4] @ Gs[5]
    assert la.norm(G7 @ G7 - np.eye(8)) < TOL
    assert la.norm(G7 - G7.conj().T) < TOL

    # Build σ_S
    perm = sigma_permutation_on_gammas()
    sigma_S = build_sigma_S(Gs, perm)

    # --- Compute weight basis (eigenstates of T_1, T_2, T_3) ---------------
    section("Step 1 — Weight basis (eigenstates of T_1, T_2, T_3)")

    weight_basis = simultaneous_eigenbasis([T_1, T_2, T_3])
    print(f"  8 weight states identified (labels = (sign(T_1), sign(T_2), sign(T_3))):")
    for lbl in sorted(weight_basis.keys()):
        print(f"    {lbl}")

    # Verify σ_S permutes these as |e₁,e₂,e₃⟩ → |e₂,e₃,e₁⟩
    section("Step 2 — Verify σ_S permutation rule on weights")
    print("  Per matching_brauer_weyl_sigma.py: σ_S maps |e₁,e₂,e₃⟩ → |e₂,e₃,e₁⟩")
    sigma_orbit_map = {}
    for lbl, vec in weight_basis.items():
        v_mapped = sigma_S @ vec
        # Find which weight this corresponds to (up to phase)
        best_match = None
        best_overlap = 0.0
        for lbl2, vec2 in weight_basis.items():
            overlap = abs(vec2.conj() @ v_mapped)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = lbl2
        sigma_orbit_map[lbl] = (best_match, best_overlap)

    expected_rule = lambda e: (e[1], e[2], e[0])
    correct = all(sigma_orbit_map[lbl][0] == expected_rule(lbl)
                  for lbl in weight_basis)
    print(f"  Rule |e₁,e₂,e₃⟩ → |e₂,e₃,e₁⟩ holds: {correct}")
    assert correct

    # --- Compute B3 species content of each weight -------------------------
    section("Step 3 — B3 species content of each weight state")
    print("  For each weight |e₁,e₂,e₃⟩ in the σ_S-permuted basis:")
    print("  compute (⟨T_L⟩, ⟨T_R⟩, ⟨Y_PS⟩, ⟨Γ_7⟩) — eigenvalues, since they're")
    print("  simultaneously diagonalized with T_1, T_2, T_3.")
    print()
    print(f"  {'weight (e₁,e₂,e₃)':<22s} {'T_L':>8s} {'T_R':>8s} {'Y_PS':>8s} "
          f"{'Γ_7':>8s}  {'PS species':>20s}")

    species_map = {}
    for lbl in sorted(weight_basis.keys()):
        v = weight_basis[lbl]
        tl = np.real(v.conj() @ T_L @ v)
        tr = np.real(v.conj() @ T_R @ v)
        ypsl = np.real(v.conj() @ Y_PS @ v)
        g7 = np.real(v.conj() @ G7 @ v)

        # Identify PS chirality via SU(2)_L vs SU(2)_R doublet membership:
        #   - L-chirality: T_R = 0 (SU(2)_R singlet), T_L ∈ {±1/2}
        #   - R-chirality: T_L = 0 (SU(2)_L singlet), T_R ∈ {±1/2}
        #
        # NOTE: this is the PS-doublet chirality, NOT Γ_7. In Cl(6,0) S,
        # Γ_7 eigenvalue = -e_1·e_2·e_3 does not directly track which SU(2)
        # doublet a state belongs to. The PS chirality is determined by
        # whether the state is in (2,1) or (1,2) of SU(2)_L × SU(2)_R.
        #
        # Species naming (B3 PS convention):
        #   L-chirality + Y_PS=+1/2 + T_L=+1/2: u_L
        #   L-chirality + Y_PS=+1/2 + T_L=-1/2: d_L
        #   L-chirality + Y_PS=-1/2 + T_L=+1/2: ν_L
        #   L-chirality + Y_PS=-1/2 + T_L=-1/2: e_L
        #   R-chirality + Y_PS=+1/2 + T_R=+1/2: u_R
        #   R-chirality + Y_PS=+1/2 + T_R=-1/2: d_R
        #   R-chirality + Y_PS=-1/2 + T_R=+1/2: ν_R
        #   R-chirality + Y_PS=-1/2 + T_R=-1/2: e_R

        if abs(tr) < 0.1 and abs(tl) > 0.1:
            chirality = "L"
            doublet_axis = tl
        elif abs(tl) < 0.1 and abs(tr) > 0.1:
            chirality = "R"
            doublet_axis = tr
        else:
            chirality = "?"
            doublet_axis = 0

        is_quark = ypsl > 0
        if is_quark:
            species = "u" if doublet_axis > 0 else "d"
        else:
            species = "ν" if doublet_axis > 0 else "e"
        full_label = f"{species}_{chirality}"

        species_map[lbl] = full_label
        print(f"  {str(lbl):<22s} {tl:>+8.3f} {tr:>+8.3f} {ypsl:>+8.3f} "
              f"{g7:>+8.3f}  {full_label:>20s}")

    # --- Interpret σ_S orbits in species terms -----------------------------
    section("Step 4 — σ_S orbits in B3 species terms")

    fixed_pts = [lbl for lbl in weight_basis if expected_rule(lbl) == lbl]
    non_fixed = [lbl for lbl in weight_basis if expected_rule(lbl) != lbl]
    orbits = []
    seen = set(fixed_pts)
    for lbl in non_fixed:
        if lbl in seen:
            continue
        orb = [lbl]
        cur = expected_rule(lbl)
        seen.add(lbl)
        while cur != lbl:
            orb.append(cur)
            seen.add(cur)
            cur = expected_rule(cur)
        orbits.append(orb)

    print(f"  Fixed points ({len(fixed_pts)}):")
    for lbl in fixed_pts:
        print(f"    {str(lbl):<22s}  → species {species_map[lbl]}")

    print(f"\n  3-orbits ({len(orbits)}):")
    for i, orb in enumerate(orbits):
        species_in_orbit = [species_map[lbl] for lbl in orb]
        print(f"    Orbit {i+1}:")
        for lbl in orb:
            print(f"      {str(lbl):<22s}  → species {species_map[lbl]}")
        print(f"    (species set: {set(species_in_orbit)})")

    # --- Hypothesis testing ------------------------------------------------
    section("Step 5 — Hypothesis classification")

    # Determine: do orbits cycle within ONE species (Hypothesis B: σ_S = color
    # within fixed species), or DIFFERENT species (mixed)?
    fixed_species = set(species_map[lbl] for lbl in fixed_pts)
    orbit_species_sets = [set(species_map[lbl] for lbl in orb) for orb in orbits]

    print(f"  Fixed-point species: {fixed_species}")
    for i, sp_set in enumerate(orbit_species_sets):
        print(f"  Orbit {i+1} species set: {sp_set}  (size {len(sp_set)})")

    print()
    if all(len(sp) == 1 for sp in orbit_species_sets):
        print(f"  HYPOTHESIS B: σ_S 3-orbits each cycle WITHIN ONE PHYSICAL SPECIES.")
        print(f"  This means σ_S labels SOMETHING WITHIN A FIXED SPECIES — possibly")
        print(f"  color-internal (3 components of a single quark species in 3 colors).")
        verdict = "B_color_internal"
    elif all(len(sp) == 3 for sp in orbit_species_sets):
        print(f"  HYPOTHESIS C: σ_S 3-orbits cycle across 3 DIFFERENT SPECIES with")
        print(f"  DIFFERENT SM gauge quantum numbers.")
        print(f"  σ_S permutes physical species across SU(2)_L vs SU(2)_R chirality,")
        print(f"  and across lepton vs quark Y_PS values. The orbits do NOT preserve")
        print(f"  electric charge, T_L, T_R, Y_PS, or chirality.")
        print()
        print(f"  σ_S is therefore NOT a SM generation Z₃ (which by SM construction")
        print(f"  must preserve all gauge quantum numbers — generations differ only in")
        print(f"  mass), and NOT a SM color Z₃ (which preserves species and chirality).")
        print()
        print(f"  σ_S is a CYCLIC PERMUTATION OF THE THREE Cl(6,0) CARTAN AXES")
        print(f"  (T₁ → T₂ → T₃ → T₁). This is an abstract structural symmetry of")
        print(f"  the Cl(6,0) algebra under matching/standard ambiguity in B3, not")
        print(f"  a physical SM symmetry.")
        verdict = "C_cartan_permutation_NOT_generation"
    else:
        print(f"  HYPOTHESIS C': σ_S 3-orbits have mixed species sets (some 1, some 3).")
        print(f"  Mixed structure — needs case-by-case analysis.")
        verdict = "C_prime_mixed"

    print()
    print(f"  Final classification: {verdict}")

    # --- Verdict and implications ------------------------------------------
    print()
    print("=" * 76)
    print("VERDICT")
    print("=" * 76)
    print()
    if verdict.startswith("C_cartan"):
        print("  σ_S = generation hypothesis is FALSIFIED.")
        print()
        print("  σ_S 3-orbits cycle across (species, chirality) pairs with DIFFERENT")
        print("  SM gauge quantum numbers. By SM construction, generation Z₃ must")
        print("  preserve all gauge quantum numbers (Q, T_L, T_R, Y, color, chirality)")
        print("  since generations are mass-eigenstate copies of the same gauge")
        print("  multiplet. σ_S violates this requirement.")
        print()
        print("  Specifically observed orbits (this run):")
        for i, orb in enumerate(orbits):
            sp_str = " → ".join(species_map[lbl] for lbl in orb)
            print(f"    Orbit {i+1}: {sp_str}")
        print()
        print("  σ_S is a CARTAN PERMUTATION SYMMETRY of Cl(6,0) — an abstract Z₃")
        print("  rotation in the (T₁, T₂, T₃) Cartan axes, distinct from C₃_body")
        print("  which permutes (T₁, T₂, T₃) under a DIFFERENT cyclic ordering")
        print("  determined by standard vs matching K_4 edge labeling.")
        print()
        print("  IMPLICATION FOR THE CHOKE-POINT:")
        print("  • σ_S = generation Z₃: FALSIFIED (orbits don't preserve gauge labels).")
        print("  • V_ub via σ_S phase mechanism: ALREADY RULED OUT (sigma_S_phase_walks).")
        print("  • V_ub via icosahedral apex φ⁻²: structural handle remains, but its")
        print("    physical interpretation is NOT 'σ_S = generation'. Some other 2I-")
        print("    equivariant structure on S ⊗ S, S ⊗ C³_obs, or another tensor")
        print("    factor would need to provide the generation labeling.")
        print()
        print("  Generation Z₃ in this framework most likely lives in C³_obs (the")
        print("  observer factor, per docs/audits/registers/adoption_register.md), NOT in Spin(6) on S.")
        print()
        print("  The 2I = SL(2,5) finding is genuinely structural (the framework's")
        print("  Cl(6,0) Cartan algebra carries hidden icosahedral symmetry under")
        print("  matching/standard-basis ambiguity) but its physical role is NOT")
        print("  'generation'. Its physical role is OPEN.")
    elif verdict.startswith("B"):
        print("  σ_S labels INTERNAL STRUCTURE within fixed B3 species.")
    elif verdict.startswith("C_prime"):
        print("  σ_S has mixed structure. Detailed analysis needed.")

    return verdict, species_map, orbits, fixed_pts


if __name__ == "__main__":
    main()
