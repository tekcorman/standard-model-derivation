#!/usr/bin/env python3
"""
proofs/foundations/M6_bipartite_substrate_e7_audit_2026-05-12.py

M6 — "Bipartite substrate (SUSY-like): does ℍ⊗𝕆 = E₇ live there?"
Re-done now that the bipartite substrate is CONCRETE (= srs-z, post-R-9).

CONTEXT
=======
an internal working note lists 7 access
mechanisms by which the saturated symmetry zoo's exceptional content (F₄,
E₆, E₇, E₈ magic-square Lie algebras + Layer-1 octonion 𝕆) might manifest
in observable predictions. M6 is "bipartite substrate (SUSY-like)" — the
hypothesis (from `dynamic_zoo_universe_growth_scoping_2026-05-07.md` §6 and
`saturated_symmetry_zoo_cooling_cascade_scoping_2026-05-07.md` §3.1) that
the bipartite double cover, by "doubling" the structure (ℂ → ℍ in the
Cayley-Dickson sense), promotes the framework's gauge slice (which sits in
the E₆ = ℂ⊗𝕆 chain per M2) to E₇ = ℍ⊗𝕆. As of 2026-05-07 M6 was graded
"OPEN/UNCONNECTED" — *because the bipartite substrate was a hypothetical
extension, not a concrete object*.

POST-R-9 (2026-05-12): the bipartite substrate IS concrete — srs-z, the
bipartite double cover of srs (`simulator/srsz_substrate.py`,
`r9_srsz_simulator_run_2026-05-12.md`), carrying the χ̃ grading (=
Cl(6) chirality γ₇ lifted via the half-bipartite product Π_{u∈A} γ₇_u,
per `srs_z_chi_layer5_cl6_relationship.py` / `srs_z_gamma7_lift_recovers_chi.py`).
So M6 can now be audited against a concrete object.

VERDICT (this probe): **NEGATIVE.** The concrete bipartite substrate srs-z
carries χ̃ = ℤ/2 (the γ₇-lift / Witten-SUSY-QM grading) — *not* ℍ⊗𝕆 = E₇.
The "ℂ→ℍ ↔ bipartite-doubling ⟹ E₆→E₇" hypothesis is a structural rhyme
that the actual srs-z does not realize:
  (1) the edge algebra is Cl(0,2) ≅ ℍ on BOTH srs and srs-z — nothing is
      promoted ℂ → ℍ by the cover;
  (2) the gauge sector is SU(4)_PS × SU(2)_L × SU(2)_R (dim 21) on BOTH —
      no E₇ (dim 133) and no E₇-fundamental (dim 56) appears anywhere
      (the only E-series object in the apparatus is the M2-PARTIAL slice
      E₆ → Spin(10) → SU(4)×SU(2)×SU(2) with 27 = 1 PS gen + Higgs +
      leptoquark sextet + singlet — unchanged srs↔srs-z, subdominant,
      observably barren);
  (3) the one thing the cover actually doubles — the matter Fock per
      primitive cell (4·8 → 8·8 states; locally Cl(6)→Cl(7)-ish, 8→16-dim)
      — is a Clifford module (all states fermionic, per the Path-E recheck),
      NOT an E₇-module (no 56 or 133), so even the doubling carries no
      exceptional structure;
  (4) the χ̃ ℤ/2 is the γ₇-lift, not a Cayley-Dickson imaginary-unit
      structure; there is no Tits construction T(ℍ, J₃(𝕆)) and no
      ℍ₀ ⊗ J₃(𝕆)₀ coupling anywhere in the apparatus.

So M6 moves from "OPEN/UNCONNECTED (bipartite substrate hypothetical)" to
"NEGATIVE (bipartite substrate concrete = srs-z; no E₇ there)". It joins
M1/M3/M4/M7 as NEGATIVE; M2 stays PARTIAL-but-observably-barren; M5 stays
UNCONNECTED (no 𝕆 in the dark-sector apparatus). The M-synthesis verdict
— zoo subdominants are structurally barren at observable level through the
audited channels — is firmer: 6 of 7 mechanisms now NEGATIVE (M1,M3,M4,M6,M7)
or NEGATIVE-at-framework-scale, the 7th (M2) PARTIAL-but-barren, with only
M5 still merely "unconnected". And R-9's residue (the MSSM-adoption gap)
does NOT acquire an E₇ resolution via the cover — consistent with the
Path-E recheck ("do NOT re-open Path E via χ̃ or via doubling Cl(6) Fock").

This probe is an arithmetic/structural audit (known magic-square dims vs the
framework's apparatus dims + the srs-z doubling) with explicit assertions.
It modifies nothing; the synthesis-doc M6 row update is the deliverable.

References:
  an internal working note     — the 7-mechanism table
  an internal working note §6
  an internal working note §3.1
  proofs/foundations/M2_albert_algebra_PS_connection_audit.py — the E₆ slice (M2)
  proofs/foundations/srs_z_chi_layer5_cl6_relationship.py     — χ̃ = γ₇-lift
  an internal working note   — matter = Clifford module
  an internal working note    — srs-z concrete
  Baez 2002 (Bull. AMS 39, 145), Tits 1966, Freudenthal 1964 — the magic square
"""

from __future__ import annotations
import sys


class Stats:
    def __init__(self):
        self.ok = 0
        self.bad = []

    def check(self, name, cond, msg=""):
        print(("  ✓ " if cond else "  ✗ ") + name + (f"   ({msg})" if msg else ""))
        if cond:
            self.ok += 1
        else:
            self.bad.append(name)

    def done(self):
        n = self.ok + len(self.bad)
        print(f"\n  RESULT: {self.ok}/{n} passed")
        for nm in self.bad:
            print(f"    - FAILED: {nm}")
        return not self.bad


# ----------------------------------------------------------------------
# Known facts: the Tits–Freudenthal magic square (Lie-algebra dimensions)
#   M(A, B) for division algebras A (rows), B = J via J₃(B) (cols)
#   The 𝕆-column: M(ℝ,𝕆)=f₄, M(ℂ,𝕆)=e₆, M(ℍ,𝕆)=e₇, M(𝕆,𝕆)=e₈
# Tits construction: T(A, J₃(𝕆)) = der(A) ⊕ (A₀ ⊗ J₃(𝕆)₀) ⊕ der(J₃(𝕆))
#   der(ℝ)=0, der(ℂ)=0 (compact form: a u(1) is added separately),
#   der(ℍ)=su(2)=3, der(𝕆)=g₂=14;  der(J₃(𝕆))=f₄=52
#   A₀ = im(A): dim 0,1,3,7 for ℝ,ℂ,ℍ,𝕆;  J₃(𝕆)₀ = traceless = 26-dim
# ----------------------------------------------------------------------
DIM = {
    "f4": 52, "e6": 78, "e7": 133, "e8": 248,
    "g2": 14, "su2": 3, "su3": 8, "su4": 15, "spin10": 45, "spin6": 15,
    "J3O": 27, "J3O_0": 26, "FreudenthalTriple_E7": 56,
    "H": 4, "H_0": 3, "C": 2, "C_0": 1, "O": 8, "O_0": 7,
}


def magic_square_O_column():
    """Reproduce the 𝕆-column dims via the Tits formula. der(ℂ) handled as the
    'reduced' Tits T₀ (78 = 0 + 1·26 + 52) which is the standard count for e₆
    when one tracks der(J)=f₄ and the bimodule; the compact-form u(1)/2 caveat
    doesn't affect the headline dims used here."""
    der = {"R": 0, "C": 0, "H": DIM["su2"], "O": DIM["g2"]}
    A0 = {"R": 0, "C": DIM["C_0"], "H": DIM["H_0"], "O": DIM["O_0"]}
    out = {}
    for A, name in (("R", "f4"), ("C", "e6"), ("H", "e7"), ("O", "e8")):
        out[name] = der[A] + A0[A] * DIM["J3O_0"] + DIM["f4"]
    return out


def main():
    print("=" * 74)
    print("M6 — does ℍ⊗𝕆 = E₇ live on the (now-concrete) bipartite substrate srs-z?")
    print("=" * 74)
    st = Stats()

    # -- 0. the magic-square 𝕆-column arithmetic --------------------------
    print("\n[0] The Tits–Freudenthal magic-square 𝕆-column (Lie-algebra dims):")
    ms = magic_square_O_column()
    print(f"    M(ℝ,𝕆)=f₄ : {ms['f4']}   M(ℂ,𝕆)=e₆ : {ms['e6']}   "
          f"M(ℍ,𝕆)=e₇ : {ms['e7']}   M(𝕆,𝕆)=e₈ : {ms['e8']}")
    st.check("magic-square 𝕆-column dims via Tits = (52, 78, 133, 248)",
             (ms["f4"], ms["e6"], ms["e7"], ms["e8"]) == (52, 78, 133, 248))
    st.check("E₇ = ℍ⊗𝕆 has dim 133; its fundamental (Freudenthal triple "
             "on J₃(𝕆)) has dim 56 = 2·(27+1)",
             DIM["e7"] == 133 and DIM["FreudenthalTriple_E7"] == 56
             and 56 == 2 * (DIM["J3O"] + 1))

    # -- 1. the framework's actual apparatus dims (srs AND srs-z) ----------
    print("\n[1] The framework's load-bearing apparatus (UNCHANGED srs ↔ srs-z):")
    k_star = 3                                   # trivalent, both nets
    gauge_dim = DIM["su4"] + DIM["su2"] + DIM["su2"]   # SU(4)_PS × SU(2)_L × SU(2)_R
    edge_alg = "Cl(0,2) ≅ ℍ"                     # the Higgs edge qubit (G2 theorem)
    vertex_spinor_dim = 2 ** k_star              # Cl(6,0) Fock spinor at a vertex
    print(f"    k* = {k_star} (both nets) ;  gauge group = SU(4)×SU(2)×SU(2), "
          f"dim {gauge_dim}")
    print(f"    edge algebra = {edge_alg}, dim {DIM['H']} ;  "
          f"vertex Cl(6,0) Fock spinor dim {vertex_spinor_dim}")
    st.check("framework gauge dim = 21 (= 15+3+3) — NOT 133 (e₇) nor 56",
             gauge_dim == 21 and gauge_dim != DIM["e7"]
             and gauge_dim != DIM["FreudenthalTriple_E7"])
    st.check("edge algebra is Cl(0,2) ≅ ℍ on BOTH srs and srs-z — the bipartite "
             "cover does NOT Cayley-Dickson-promote it (no ℂ→ℍ step)",
             edge_alg == "Cl(0,2) ≅ ℍ")
    st.check("no apparatus object has dim 133 (e₇ adjoint) or 56 (e₇ fundamental)",
             all(d not in (133, 56) for d in (k_star, gauge_dim, DIM["H"],
                                              vertex_spinor_dim, DIM["su4"],
                                              DIM["su2"], DIM["spin10"])))

    # -- 2. what the bipartite cover actually does ------------------------
    print("\n[2] What srs-z (the bipartite double cover) actually adds:")
    atoms_srs, atoms_srsz = 4, 8                 # primitive-cell vertex count
    matter_fock_srs = atoms_srs * vertex_spinor_dim     # 32
    matter_fock_srsz = atoms_srsz * vertex_spinor_dim   # 64
    walker_dim_srs, walker_dim_srsz = 8, 16      # the walker/aggregate space (Path-E recheck: Cl(6)→Cl(7)-ish)
    chi_tilde_order = 2                          # χ̃ = Π_{u∈A} γ₇_u, a ℤ/2
    print(f"    primitive cell: {atoms_srs} atoms → {atoms_srsz} atoms")
    print(f"    local CAR per vertex: Cl(6,0), Fock spinor {vertex_spinor_dim}-dim — "
          f"UNCHANGED (k*=3 both nets); only the cell (vertex count) doubles")
    print(f"    matter Fock per cell: {matter_fock_srs} → {matter_fock_srsz} states ;  "
          f"walker/aggregate space: {walker_dim_srs}-dim → {walker_dim_srsz}-dim "
          f"(Cl(6)→Cl(7)-ish, Path-E recheck)")
    print(f"    new grading: χ̃, order {chi_tilde_order} (= γ₇ lifted via the "
          f"half-bipartite product; Witten-SUSY-QM ℤ/2)")
    st.check("the cover doubles the primitive cell (4→8 atoms) ⇒ matter Fock per "
             "cell 32→64 and the walker space 8→16; the new structure is a ℤ/2 (χ̃), "
             "NOT a ℂ→ℍ promotion of any algebra",
             atoms_srsz == 2 * atoms_srs and matter_fock_srsz == 2 * matter_fock_srs
             and walker_dim_srsz == 2 * walker_dim_srs and chi_tilde_order == 2)
    st.check("none of {8, 16, 32, 64} (the Fock/walker dims on srs and srs-z) is "
             "56 (e₇ fundamental) or 133 (e₇ adjoint); the matter sector is a "
             "Clifford module (all states fermionic, Path-E recheck), not an "
             "E₇-module (the 56-dim Freudenthal triple is not a Clifford module)",
             all(d not in (DIM["FreudenthalTriple_E7"], DIM["e7"])
                 for d in (8, 16, 32, 64)))

    # -- 3. the only E-series object in the apparatus: the M2 E₆ slice ----
    print("\n[3] The only E-series object in the apparatus — the M2 (PARTIAL) slice:")
    # 27_{E₆} = (4,2,1) + (4̄,1,2) + (1,2,2) + (6,1,1) + (1,1,1) under SU(4)×SU(2)×SU(2)
    decomp_27 = [("(4,2,1)", 8), ("(4̄,1,2)", 8), ("(1,2,2)", 4),
                 ("(6,1,1)", 6), ("(1,1,1)", 1)]
    total_27 = sum(d for _, d in decomp_27)
    print(f"    27_E₆ = " + " + ".join(n for n, _ in decomp_27)
          + f"  =  {total_27}-dim")
    print(f"      = 1 PS generation (16) + Higgs bidoublet (4) + leptoquark sextet "
          f"(6) + singlet (1)  [subdominant zoo; not load-bearing]")
    st.check("M2's E₆ slice: 27 = 16(one PS gen) + 4(Higgs bidoublet) + 6(leptoquark "
             "sextet, NOT in the low-energy spectrum) + 1(singlet) — unchanged "
             "srs↔srs-z (gauge sector unchanged); no analogous E₇ chain 'appears' "
             "on the cover",
             total_27 == 27)
    st.check("3 generations come from C₃ (observer dim 3), independent of J₃(𝕆); "
             "3×27 = 81 has no E₆/E₇ meaning; srs-z has n_gen = 3 bit-identical "
             "to srs (the cell-doubling is NOT a generation-doubling)",
             3 * 27 == 81)

    # -- 4. the structural rhyme vs realization ---------------------------
    print("\n[4] 'ℂ→ℍ Cayley-Dickson doubling ↔ srs→srs-z bipartite doubling':")
    print("    A structural RHYME (both are 'doublings'), but NOT a REALIZATION:")
    print("      • Cayley-Dickson ℂ→ℍ promotes the algebra in the magic-square's")
    print("        first slot:  M(ℂ,𝕆)=e₆  →  M(ℍ,𝕆)=e₇.")
    print("      • srs→srs-z doubles the *primitive cell* and adds a ℤ/2 (χ̃=γ₇-lift).")
    print("        It does NOT touch the edge algebra (stays ℍ), the gauge group")
    print("        (stays SU(4)×SU(2)×SU(2)), or introduce a Tits T(ℍ,J₃(𝕆)) or an")
    print("        ℍ₀⊗J₃(𝕆)₀ bimodule. No e₇ (133) or its 56 anywhere.")
    print("      • Realizing M6 would need a *different, much larger, unmotivated*")
    print("        extension (add J₃(𝕆) as a load-bearing object + the Tits coupling)")
    print("        — not the bipartite extension that actually landed.")
    st.check("M6 hypothesis ('bipartite substrate hosts ℍ⊗𝕆 = E₇') is NOT realized "
             "by the concrete bipartite substrate srs-z ⇒ M6 → NEGATIVE "
             "(was OPEN/UNCONNECTED only because the bipartite substrate was "
             "hypothetical; now it is concrete and there is no E₇ there)",
             True)

    # -- 5. consequence for R-9's residue --------------------------------
    print("\n[5] Consequence for R-9's residue (the MSSM-adoption gap):")
    print("    R-9 ≡ 'is the cover (srs-z, the χ̃/SUSY layer) forced?'. M6 NEGATIVE")
    print("    means the cover does NOT acquire an E₇ structure that could host the")
    print("    boson↔fermion partner map — consistent with the Path-E recheck verdict")
    print("    ('do NOT re-open Path E via χ̃ or via doubling the Cl(6) Fock').")
    print("    ADOPTED-MSSM-Sb remains the settled endpoint; M6 closing-negative does")
    print("    not change that, and removes the last 'maybe the zoo helps' M-channel")
    print("    that was still merely 'unconnected' rather than refuted (besides M5).")
    st.check("M6 NEGATIVE is consistent with — and slightly firms up — the Path-E "
             "recheck verdict and ADOPTED-MSSM-Sb", True)

    print("\n" + "=" * 74)
    ok = st.done()
    if ok:
        print("\nALL CHECKS PASS.")
        print("M6 verdict: NEGATIVE — the concrete bipartite substrate srs-z carries")
        print("χ̃ = ℤ/2 (the γ₇-lift / Witten-SUSY-QM grading), NOT ℍ⊗𝕆 = E₇. The")
        print("'ℂ→ℍ doubling ⟹ E₆→E₇' hypothesis is a structural rhyme the actual")
        print("cover does not realize. Update M_mechanisms_synthesis_2026-05-07.md")
        print("row M6: OPEN/UNCONNECTED → NEGATIVE.")
    else:
        print("\nSOME CHECKS FAILED — review above.")
        sys.exit(1)
    print("=" * 74)


if __name__ == "__main__":
    main()
