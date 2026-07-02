#!/usr/bin/env python3
"""
walker_48_multiplicity_audit_2026-06-02.py
===========================================
Chase the 24→48 matter multiplicity (the "iso-redundancy" factor) in the
walker-matter "48↔48" unification.

CONTEXT. `theorem_walker_matter_unification` claims 48 Hashimoto saddle modes
= 48 SM Weyl spinors. Digging in (2026-06-02) showed: only 24 of the 48 modes
are fermionic matter (h_P:8, h_Γ:8, h_H:8); the companion probe admits "the
1-to-1 isn't mode-to-spinor"; and `walker_class_dictionary` §5 reaches 48 via
"8 P-saddle modes × 4 vertices / iso-redundancy(=2) × 3 gen = 48", with the
"iso-redundancy" factor unexplained. This probe audits that factor.

WHAT THIS RESOLVES:
  • The MATTER COUNT is clean and fully sourced — it does NOT need an
    "iso-redundancy" factor:
        48 = 4(PS color) × 2(chirality) × 2(isospin) × 3(generation)
    sources: color = Cl(6) Fock SU(4)_PS; chirality = γ_7; isospin = per-edge
    Cl(0,2)→SU(2); generation = observer C³ (R3). (per_weyl_spinor_dictionary
    §2.3/2.4: the 8 Cl(6) Fock states ARE 4 colors × 2 chiralities.)
  • The "iso-redundancy = 2" in walker_class_dictionary §5 is SPURIOUS: it
    multiplied by "4 vertices" — but srs has ONE vertex orbit (Wyckoff 8a), so
    the 4 cell-atoms are symmetry-equivalent (one orbit), NOT a ×4 dof
    multiplicity. Multiplying by 4 then dividing by 2 is a double-count then
    half-correct; the clean per-orbit count never introduces either factor.

WHAT REMAINS GENUINELY OPEN (the real Phase-1b):
  • The explicit MAP between the per-cell Hashimoto mode structure (12 directed
    edges × 4 saddles = 48 modes, of which 24 are fermionic) and the matter
    content (4×2×2×3 = 48). The two 48-totals are derived INDEPENDENTLY and
    factor INCOMPATIBLY (edges×saddles vs color×chir×isospin×gen). The
    per-family mode→spinor ratios are wildly non-uniform — h_P: 8→42,
    h_Γ: 8→3, h_H: 8→3 — and no worked-out assignment exists.
    (per_weyl_spinor_dictionary line 17: "research-level structural work the
    framework's iso theorem doesn't yet do explicitly.")

So: the COUNT is clean; the "iso-redundancy" is an artifact; the MODE↔SPINOR
MAP is the actual undone work, and the "48↔48" is a coincidence of two
independently-derived totals, not a worked-out correspondence.
"""

import json
import os
from fractions import Fraction as F


def section(t):
    print("\n" + "=" * 88 + f"\n {t}\n" + "=" * 88)


def main():
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    SNAP = json.load(open(os.path.join(repo, "simulator", "menus", "data",
                                       "rcsr_candidates_snapshot.json")))["entries"]
    e = SNAP["srs"]

    section("STEP 1 — srs has ONE vertex orbit (the 4 cell-atoms are NOT 4 copies)")
    n_orbits = len(e["vertex_orbits"])
    wyck = e["vertex_orbits"][0]["wyckoff_label"]
    print(f"  srs space group : {e['sg_name']}")
    print(f"  # vertex orbits  : {n_orbits}   (Wyckoff {wyck})")
    assert n_orbits == 1, "srs must have a single vertex orbit"
    print("  → the 4 atoms in the primitive cell are ONE symmetry orbit (equivalent).")
    print("    Matter content is PER-ORBIT; '×4 vertices' is NOT a degree-of-freedom factor.")

    section("STEP 2 — The clean matter factorization (fully sourced, no redundancy factor)")
    factors = [("PS color (Cl(6) Fock SU(4)_PS: ℓ,r,g,b)", 4),
               ("chirality (γ_7 = ±1)",                     2),
               ("isospin (per-edge Cl(0,2) → SU(2))",       2),
               ("generation (observer C³, R3)",             3)]
    prod = 1
    for name, v in factors:
        prod *= v
        print(f"    × {v:<2}  {name}")
    print(f"    -----")
    print(f"    = {prod}   (per generation: {prod//3} = the SM 16-spinor generation)")
    assert prod == 48 and prod // 3 == 16
    print("  The '8 Cl(6) Fock states' = 4 color × 2 chirality (per_weyl_spinor §2.3).")
    print("  No '4 vertices' and no 'iso-redundancy' appear in this count.")

    section("STEP 3 — The walker_class §5 route reaches 48 only via a spurious /2")
    p_modes, n_vertices, iso_redund, n_gen = 8, 4, 2, 3
    val = F(p_modes * n_vertices, iso_redund) * n_gen
    print(f"    walker_class §5:  8 P-modes × 4 vertices / iso-redundancy(2) × 3 gen")
    print(f"                    = {p_modes}×{n_vertices}/{iso_redund}×{n_gen} = {val}")
    assert val == 48
    print("  Reaches 48 — but the ×4 treats the single vertex-orbit as 4 dof (STEP 1 says no),")
    print("  and the ÷2 'iso-redundancy' is the ad-hoc correction for that overcount.")
    print("  Net: ×4/2 = ×2 is doing the work of (chirality) OR (isospin) in STEP 2,")
    print("  but mislabeled as a vertex/redundancy effect. The factor is SPURIOUS bookkeeping.")

    section("STEP 4 — The genuine mismatch: 24 fermionic modes ↔ 48 spinors, NON-uniform")
    # 48 Hashimoto modes per cell = 12 directed edges × 4 saddles. Of these, fermionic-matter:
    fermionic = {"h_P (charged)": (8, 42), "h_Γ (ν_L)": (8, 3), "h_H (ν_R)": (8, 3)}
    nonmatter = {"Trivial |λ|=1 (gauge/cycle)": 18, "Perron (Higgs/VEV)": 2, "h_N (dark/inert)": 4}
    tot_modes = sum(m for m, _ in fermionic.values()) + sum(nonmatter.values())
    tot_ferm_modes = sum(m for m, _ in fermionic.values())
    tot_spinors = sum(s for _, s in fermionic.values())
    print(f"  48 Hashimoto modes/cell = 12 directed edges × 4 saddles. Breakdown:")
    for k, (m, s) in fermionic.items():
        print(f"    {k:<22} {m:>2} modes  →  {s:>2} Weyl spinors   (ratio {s}/{m} = {F(s,m)})")
    for k, m in nonmatter.items():
        print(f"    {k:<22} {m:>2} modes  →   0 Weyl spinors   (non-matter)")
    print(f"    {'-'*22} {'--':>2}        {'--':>2}")
    print(f"    total modes = {tot_modes}; fermionic modes = {tot_ferm_modes}; spinors = {tot_spinors}")
    assert tot_modes == 48 and tot_ferm_modes == 24 and tot_spinors == 48
    print()
    print(f"  → only {tot_ferm_modes} of the 48 modes are fermionic, and they map to {tot_spinors} spinors")
    print(f"    with WILDLY non-uniform per-family ratios (h_P 42/8, h_Γ 3/8, h_H 3/8).")
    print(f"    There is no uniform 'multiplicity'; the explicit per-mode→per-spinor map is")
    print(f"    undone (per_weyl_spinor_dictionary line 17: research-level work not yet done).")

    section("VERDICT — the iso-redundancy audit")
    print("""\
  RESOLVED:
   • The matter COUNT is clean and fully sourced: 48 = 4(color) × 2(chirality)
     × 2(isospin) × 3(generation). No "iso-redundancy" factor is needed.
   • srs has ONE vertex orbit ⇒ the "×4 vertices" in walker_class §5 is not a
     dof multiplicity, and the compensating "÷2 iso-redundancy" is a spurious
     bookkeeping artifact. The genuine ×2 in the count is chirality (or
     isospin), not a vertex/redundancy effect.

  STILL OPEN (the real Phase-1b):
   • The "48↔48" is a COINCIDENCE OF TWO INDEPENDENTLY-DERIVED TOTALS:
       matter  = 4×2×2×3       (Cl(6) Fock + γ_7 + edge Cl(0,2) + observer C³)
       modes   = 12 edges × 4 saddles   (Hashimoto spectrum)
     They factor incompatibly. Only 24 of the 48 modes are fermionic, and the
     per-family mode→spinor ratios (8→42, 8→3, 8→3) are non-uniform. The
     explicit map — which Hashimoto eigenmode is which Weyl spinor — is NOT
     worked out. That map (R1.2 generation-grading across the 4-vertex cell)
     is what the "iso-redundancy" was papering over.

  BEARING ON THE UNIFICATION CLAIM:
   The corrected §1 statement (sector-level, not bijection) is the honest one.
   The matter count stands on its own (4×2×2×3, fully sourced). The walker-mode
   side is a separate, equally-48 structure whose detailed identification with
   the matter side remains the open per-spinor expansion — NOT a closed iso.
""")
    return True


if __name__ == "__main__":
    main()
    raise SystemExit(0)
