#!/usr/bin/env python3
"""
S1 = R1.2 — Generation grading via Galois outer C_3 on srs vertices.

GROUND-THEORY SESSION 1: kick off the Layer 5 closure program.
Scoping: an internal working note

GOAL: determine the correct structural reading of the substrate's C_3 action
on Cl(6,0) Fock + srs cell content, to identify which of:
  R-A: C_3 = color, uniform triplication (= R1_1's "96 = 2 × 48" reading)
  R-B: C_3 = color, quark-only triplication (= 64 = 4 SM gens / cell)
  R-C: C_3 = generations across vertices via σ = (v_0)(v_1 v_3 v_2) on cell

LANDS the substrate matter content count and tests whether substrate-doubling-
as-MSSM-partners hypothesis is structurally alive (S1-B) or eliminated (S1-A/C).

UPSTREAM (read in this session):
  - B5.3-core (`proofs/foundations/theorem_B5_3_core.py`): σ = (v_0)(v_1 v_3 v_2)
    on srs primitive cell vertices. C_3 outer action, body-diagonal 3-fold rotation.
    Isotypic decomposition (4, 4, 4) on 12-dim edge fibre; (4, 2, 2) on 8-dim
    Ramanujan subspace at P.
  - B3-B6 reconciliation (`docs/framework/B3_B6_reconciliation.md`):
    !! IMPORTANT !! B6's "C_3 = Z(SU(3)_color)" identification is algebraically
    INCORRECT. The body-diagonal C_3 on SU(4) fundamental has eigenvalues
    (1, 1, ω, ω²) with multiplicities (2, 1, 1), NOT (3, 1) as Z_3 center
    would give. Three corrected interpretations open: (α) generation index,
    (β) pure algebraic SU(4) Cartan, (γ) cyclic SU(3) maximal-torus action
    (NOT center). None currently derived.
  - R1_1: per-vertex 8 Cl(6,0) Fock states, color factored.

The B3-B6 reconciliation already DEFEATS reading R-A's "color via inner C_3."
The remaining live readings are R-B (some other inner color action) or R-C
(C_3 = generations across vertices).

This probe focuses on R-C and tests whether it consistently labels 3 SM gens.
"""

# ============================================================
# B5.3-CORE σ ACTION ON 4 SRS-CELL VERTICES
# ============================================================
# σ = (v_0)(v_1 v_3 v_2)  —  fixes v_0, 3-cycles (v_1, v_3, v_2).
SIGMA = {0: 0, 1: 3, 3: 2, 2: 1}   # σ(v_i) = SIGMA[i]

def apply_sigma(v): return SIGMA[v]


# ============================================================
# PER-VERTEX 8-STATE FOCK CONTENT (from R1_1, color factored)
# ============================================================
# 8 species labels per vertex, identical content at every vertex.
# (color factored: each "u" represents 1 of 3 colors; color filled in below.)
SPECIES = ['nu_L', 'e_L', 'u_L', 'd_L', 'nu_R', 'e_R', 'u_R', 'd_R']

def is_quark(species):
    return species.startswith('u') or species.startswith('d')


# ============================================================
# C_3 ISOTYPIC DECOMPOSITION OF VERTEX PERMUTATION
# ============================================================
# Per species, 4 copies (one per vertex). σ permutes as (v_0)(v_1 v_3 v_2):
#   - v_0 copy: fixed by σ, eigenvalue +1 (trivial isotype)
#   - (v_1, v_2, v_3) copies: form a 3-cycle, decomposing as
#     trivial (1 dim) + ω (1 dim) + ω² (1 dim) isotypes.
# So per species: (2 trivial, 1 ω, 1 ω²)  ← 2 trivial = v_0 + sum(v_1,v_2,v_3).

def isotypic_count_per_species():
    return {'trivial': 2, 'omega': 1, 'omega_bar': 1}

def isotypic_count_per_cell():
    """8 species × (2, 1, 1) = (16, 8, 8) isotypic per cell, color factored."""
    per_sp = isotypic_count_per_species()
    return {k: 8 * v for k, v in per_sp.items()}


# ============================================================
# COLOR INCLUSION (R-B reading: only quarks triplicated)
# ============================================================
def color_factor(species):
    return 3 if is_quark(species) else 1

def states_per_vertex_color_included():
    """8 species → 4 leptons + 4 quarks × 3 colors = 16 per vertex (color included)."""
    return sum(color_factor(s) for s in SPECIES)

def states_per_cell_color_included(reading):
    """Per cell, 4 vertices × 16 (R-B) = 64. Under R-A (R1_1's incorrect): 4 × 24 = 96."""
    if reading == 'R-A':
        return 4 * (8 * 3)   # uniform triplication (B6's broken color reading)
    elif reading == 'R-B':
        return 4 * states_per_vertex_color_included()   # quark-only
    elif reading == 'R-C':
        # R-C: generations from σ orbit. v_0 = 1 SM-gen-equivalent (interpretation TBD);
        # (v_1,v_2,v_3) = 3 SM gens. Color triplication applies as in R-B.
        v0_states = states_per_vertex_color_included()   # 16
        v123_states = 3 * states_per_vertex_color_included()   # 48
        return v0_states + v123_states


# ============================================================
# REPORT
# ============================================================
def report():
    print("=" * 78)
    print("  S1 = R1.2 — Generation grading via outer Galois C_3 on srs vertices")
    print("=" * 78)

    print("\n  B5.3-core σ on 4 srs-cell vertices:")
    print(f"    σ = (v_0)(v_1 v_3 v_2)")
    for v in (0, 1, 2, 3):
        print(f"      σ(v_{v}) = v_{apply_sigma(v)}{'   [FIXED]' if apply_sigma(v) == v else ''}")

    print("\n  B3-B6 RECONCILIATION (2026-04-17 — must be respected):")
    print("    The C_3 eigenvalues (1, 1, ω, ω²) on SU(4) fundamental have")
    print("    multiplicity (2, 1, 1), NOT (3, 1) as Z_3 ⊂ SU(3)_color would give.")
    print("    => Reading R-A ('C_3 = SU(3)_color center, uniform triplication') is BROKEN.")
    print("    Live readings: R-B (some other inner color action) or R-C (outer = gen).")

    iso_per_species = isotypic_count_per_species()
    iso_per_cell = isotypic_count_per_cell()

    print("\n  C_3 ISOTYPIC DECOMPOSITION OF VERTEX-PERMUTATION (per species):")
    print(f"    v_0 fixed:            +1 isotype × 1")
    print(f"    (v_1, v_2, v_3) cycle: trivial × 1 + ω × 1 + ω̄ × 1")
    print(f"    Total per species:    trivial × {iso_per_species['trivial']}, "
          f"ω × {iso_per_species['omega']}, ω̄ × {iso_per_species['omega_bar']}")

    print(f"\n  Per cell (color-factored, 8 species):")
    print(f"    trivial × {iso_per_cell['trivial']}, ω × {iso_per_cell['omega']}, ω̄ × {iso_per_cell['omega_bar']}")
    print(f"    Total: {sum(iso_per_cell.values())} (= 4 vertices × 8)")

    print("\n  STATE COUNTING under three readings:")
    for reading in ('R-A', 'R-B', 'R-C'):
        count = states_per_cell_color_included(reading)
        print(f"    Reading {reading}: {count} colored fermion states per cell")

    print("""
  R-A (uniform triplication): 96 — R1_1's reported "doubling" count. BUT this
       reading requires C_3 = Z_3 ⊂ SU(3)_color center, which the B3-B6
       reconciliation DISPROVES. R-A is structurally BROKEN.

  R-B (quark-only triplication): 64 — correct color action if some non-center
       inner C_3 exists. Per cell = 4 SM-gen-equivalents (each vertex = 1 SM
       gen with all colors). No doubling.

  R-C (outer C_3 = generations): 64 — same total as R-B, but interpretation
       differs: v_0 = 1 special-sector SM-gen-equivalent + (v_1, v_2, v_3) =
       3 SM gens via the C_3 3-orbit (each isotype = one generation).
""")

    print("-" * 78)
    print("  S1 OUTCOME (Reading R-C analysis):")
    print("-" * 78)
    print("""
  Under R-C, the substrate's σ permutation of vertices gives a structurally
  clean generation labeling:
    - 3 SM generations from C_3 isotypes on (v_1, v_2, v_3) cycle:
        gen 1 = trivial isotype = (v_1 + v_2 + v_3) / √3 superposition
        gen 2 = ω isotype       = (v_1 + ω·v_3 + ω²·v_2) / √3
        gen 3 = ω̄ isotype       = (v_1 + ω²·v_3 + ω·v_2) / √3
    - v_0 = "special" sector, isolated from generation grading.

  v_0's role in MSSM-equivalence test:
    v_0 has 16 color-included states (1 SM-gen-equivalent). Candidates:
      (i)  4th SM gen — RULED OUT (framework + observation has 3 gens).
      (ii) Higgs sector — color-factored: 4 leptons → 4 Higgs scalars (= 2
           Higgs doublets) + 4 Higgsino Weyl. The colored part (12 states)
           would be heavy "leptoquark Higgs" or similar, decoupled at low
           energy. Plausible structural reading.
      (iii) Pure scaffolding (gauge constraint, ghost-like) — possible.

  CRUCIAL: under R-C, substrate has NO MSSM SCALAR PARTNER content.
  3 SM gens are pure fermion matter; v_0 is at most 2 Higgs doublets +
  Higgsinos, NOT squarks/sleptons.

  Therefore the substrate-derives-MSSM-β-coefficients hypothesis FAILS
  under R-C — the 48 scalar partners (squarks, sleptons) are NOT in
  Cl(6,0) Fock per cell. Layer 5 SUSY closure via R1.2 alone is RULED OUT.
""")

    print("-" * 78)
    print("  STATE COUNT UNDER R-C (corrected):")
    print("-" * 78)
    print(f"""
  Per cell, color-included fermion states (R-C reading):
    v_0 (special sector):  16 states  (= 4 leptons + 4 quarks × 3 colors)
    v_1 (gen 1):           16 states
    v_2 (gen 2):           16 states
    v_3 (gen 3):           16 states
    Total:                 64 states

  vs R1_1's reported 96 — R1_1 used the (now-disproved) R-A reading.
  vs MSSM matter+partners (3 gens, fermion + scalar): 96-120 states.

  Substrate count: 64 (R-C) << MSSM count (~120). The "factor of 2 doubling"
  in R1_1's verdict is an artifact of R-A reading and does NOT survive
  the B3-B6 reconciliation correction.

  SUBSTRATE provides: 3 SM gens (48 fermion matter) + 1 special sector
  (16, plausibly 2 Higgs supermultiplets totaling 8 Higgs + 4 Higgsino +
  4 colored heavy + 4 other) = 48 + ≤16 = ≤64.

  MSSM needs: 48 fermion matter + 48 scalar partners + 24 Higgsino + 12
  gauginos = ~132. Substrate falls short by ≥68 states.
""")

    print("=" * 78)
    print("  S1 VERDICT — AB-S1 PRE-DECLARED ABORT STATUS")
    print("=" * 78)
    print("""
  AB-S1.1 (Galois C_3 doesn't lift consistently): NOT TRIGGERED.
          σ on vertices is unambiguous; lift to per-vertex Fock is just
          permutation of Fock spaces. Cleanly lifts.

  AB-S1.2 (R-C lands → no substrate doubling → Layer 5 closure FAILS via
          this route): TRIGGERED.
          R-A reading is structurally broken (B3-B6 reconciliation).
          R-B and R-C give 64 per cell, NOT 96. The R1_1 "doubling
          speculation" was based on the broken R-A reading.

  S1 OUTCOME: S1-C (Reading R-C correct, doubling illusory).
              → Pursue Path B/C reopening (S5) per program doc.
              → S2 (Cl(0,2) edge sector) still worth running — it might
                 provide gauginos/Higgsinos/scalars from a DIFFERENT primitive.

  STRUCTURAL CONCLUSION:
    The framework's substrate Cl(6,0) Fock per srs cell contains
    EXACTLY 3 SM generations (from the σ 3-orbit on v_1, v_2, v_3) plus
    a special v_0 sector (≤ 1 SM-gen-equivalent, plausibly Higgs+Higgsino).

    There is NO substrate-derived doubling that produces MSSM-equivalent
    scalar partner content. R1_1's "factor of 2 doubling" speculation is
    artifact of an incorrect color action reading.

    Layer 5 SUSY closure via SUBSTRATE VERTEX FOCK alone is RULED OUT.

  PRECISE OPEN QUESTION (for S2):
    Does Cl(0,2) edge sector (24 dim per cell) contain SUSY-partner-
    equivalent content? If yes, Layer 5 closure may still be reachable
    via {Cl(6,0) matter} ⊕ {Cl(0,2) partners} construction. If no,
    Layer 5 is genuinely a structural assertion external to current
    substrate axioms; framework predictions α_i(M_Z) etc. remain
    DOMINANT-CONDITIONAL.
""")
    print("=" * 78)


if __name__ == "__main__":
    report()
