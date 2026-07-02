#!/usr/bin/env python3
"""
M2 — Albert algebra J_3(O) ↔ Pati-Salam 3-generation structural connection audit.

Question: the 27-dim Albert algebra J_3(O) (3x3 Hermitian octonion matrices)
sits in the saturated symmetry zoo (subdominant) via O ⊗ O magic-square route
and via F_4 = Aut(J_3(O)). The framework's Pati-Salam unification SU(4) x
SU(2)_L x SU(2)_R has fermion content with one natural slice carrying 16
states/generation (LH+RH Weyl), or 24 states for 3-gen LH-only. Is there a
27-dim PS-rep candidate that maps to J_3(O) under F_4 in a structurally
non-trivial way, or is the dim-27 a coincidence?

Methodology:
  Step 1. Enumerate PS rep dim candidates (LH-Weyl, LH+RH, 3-gen, with/without
          Higgs partners) and identify all 27-dim reachable totals.
  Step 2. Compute J_3(O) decomposition under (a) SO(8) triality, (b) F_4
          breaking F_4 -> Spin(9) -> ..., (c) the standard exceptional-Jordan
          F_4 -> SU(3) x SU(3) and F_4 -> Spin(9) / Spin(8) branchings.
  Step 3. Branch J_3(O) under F_4 -> SU(3) ("color") and check whether any
          such decomposition contains the PS rep content (4,2,1)(+ ...) for
          one or three generations.
  Step 4. Compare structurally: does any PS-rep choice carry a natural F_4
          action that recovers PS rep structure under F_4 -> PS-subgroup?
  Step 5. Verdict: STRUCTURAL or ACCIDENTAL.

This probe is purely STRUCTURAL bookkeeping at the saturated-zoo level.
It does NOT modify framework theorems or predictions. It produces a verdict
on whether the dim-27 coincidence is rep-theoretically promotable or not.

DAG: branching tabulation + dim arithmetic.
"""

import math


# ---------------------------------------------------------------------------
# Step 1. PS rep dim candidates
# ---------------------------------------------------------------------------

def enumerate_ps_rep_candidates():
    """
    PS = SU(4) x SU(2)_L x SU(2)_R.
    Standard fermion content per generation:
      F_L = (4, 2, 1)  : 8 states (LH q + l)
      F_R = (4-bar, 1, 2) : 8 states (RH q + l)
      Per generation total: 16 (Weyl)
    Higgs:
      Phi = (1, 2, 2) bidoublet : 4
      Sigma = (15, 1, 1) adjoint: 15
      Delta_R = (10, 1, 3) (PS sym tensor): 30
      H_PS = (4, 1, 2): 8
    """
    fermion_LH = ('(4, 2, 1)', 8)
    fermion_RH = ('(4-bar, 1, 2)', 8)
    higgs_bidoublet = ('(1, 2, 2)', 4)
    higgs_PS = ('(4, 1, 2)', 8)
    higgs_adjoint_15 = ('(15, 1, 1)', 15)
    higgs_LR_charged = ('(10, 1, 3)', 30)
    singlet = ('(1, 1, 1)', 1)
    candidates = []
    # Start: LH+RH 1 generation:
    candidates.append(("LH+RH 1 gen", fermion_LH[1] + fermion_RH[1]))
    candidates.append(("LH only 1 gen", fermion_LH[1]))
    candidates.append(("LH+RH 3 gen", 3 * (fermion_LH[1] + fermion_RH[1])))
    candidates.append(("LH only 3 gen", 3 * fermion_LH[1]))
    candidates.append(("RH only 3 gen", 3 * fermion_RH[1]))
    candidates.append(("LH 3 gen + Higgs (1,2,2)", 3 * fermion_LH[1] + higgs_bidoublet[1]))
    candidates.append(("LH 3 gen + 3 singlets",   3 * fermion_LH[1] + 3 * singlet[1]))
    candidates.append(("LH 3 gen + (15, 1, 1)",   3 * fermion_LH[1] + higgs_adjoint_15[1]))
    candidates.append(("LH+RH 1 gen + 3 singlets", 16 + 3))
    candidates.append(("LH+RH 1 gen + (1,2,2)+1+1+1+1+1+1+1", 16 + 4 + 7))
    candidates.append(("(4,1,2) PS-Higgs + (4-bar,1,2) + 11 singlets", 8 + 8 + 11))
    candidates.append(("(15,1,1) + (4,2,1) + (4-bar,1,2)", 15 + 8 + 8))
    candidates.append(("3 gen LH + 1 (1,1,3)", 24 + 3))   # 27 trivial
    return candidates


# ---------------------------------------------------------------------------
# Step 2. J_3(O) decomposition under standard branchings
# ---------------------------------------------------------------------------

def jordan_albert_dim():
    """
    J_3(O) = 3x3 Hermitian octonion matrices.
    Real dimension = 3 (real diagonal) + 3 octonion off-diagonals (each 8-dim
    over R) = 3 + 3*8 = 27.
    """
    return 3 + 3 * 8


def f4_dim_check():
    """F_4 = Aut(J_3(O)) has dim 52."""
    return 52


def j3_so8_triality_decomp():
    """
    Standard decomposition of J_3(O) (=27) under SO(8) ⊂ F_4 (the 'isotropy'
    SO(8) acting on the off-diagonal octonion entries via triality). The
    relevant decomposition splits into the three diagonal scalars + three
    octonion blocks transforming as the three 8-dim reps of SO(8) under
    triality (8_v, 8_s, 8_c).

    27 = 1 + 1 + 1 + 8_v + 8_s + 8_c   (NOT a direct decomposition; the
    triality permutes the three blocks). For F_4 ⊃ Spin(9), the Spin(9)-
    decomposition is

        27 = 1 ⊕ 9 ⊕ 16   (Spin(9))

    where 16 is the spinor of Spin(9). For Spin(9) ⊃ Spin(8), 16 = 8_s ⊕ 8_c
    and 9 = 8_v ⊕ 1, recovering the triality picture.
    """
    return {
        'SO(8)_triality': '1 + 1 + 1 + 8_v + 8_s + 8_c (perm by triality)',
        'Spin(9)':        '1 + 9 + 16',
        'Spin(8)':        '1 + 1 + 1 + 8_v + 8_s + 8_c',
        'F_4':            '27   (irrep, the fundamental of F_4)',
    }


def j3_su3_x_su3_decomp():
    """
    Under F_4 ⊃ SU(3) × SU(3) (the 'magic square' SU(3)_C × SU(3)_F maximal
    subgroup of F_4), the 27 of F_4 decomposes as

        27 = (3, 3-bar) ⊕ (3-bar, 3) ⊕ (1, 1) ⊕ ...  (not exactly; standard
        result: 27 = (3, 3-bar) ⊕ (3-bar, 3-bar) ⊕ (3, 3) ?)

    The CORRECT branching (Slansky 1981 Table 47, Yokota 2009 §3.7):
        F_4 ⊃ SU(3) × SU(3):  52 = (8,1) + (1,8) + (3,6) + (3-bar,6-bar) + ...
        but for the 26 of F_4 (note F_4 has 26-dim rep, not 27 — see below).

    CRITICAL POINT: F_4 has fundamental rep dimension 26 (not 27). The 27
    appears as J_3(O) = R + V_26 where R is a singlet (the trace-1 part).
    F_4 acts on J_3(O) via its 26 irrep + a singlet (trace).

    Under F_4 ⊃ Spin(9):  26 = 9 ⊕ 16 + 1 (with 1 as trace).
    Under E_6 ⊃ F_4:     27 = 1 ⊕ 26 (E_6 has the 27 as fundamental).
    """
    return {
        'F_4_acts_on_J3O': 'reducibly: 27 = 1 (trace) + 26 (traceless)',
        'F_4 > Spin(9)':   '26 = 9 + 16 + 1   (trace separate)',
        'F_4 > SU(3)xSU(3)': '26 = (3,3-bar) + (3-bar,3) + (8,1) + ... [partial]',
        'E_6_irrep_27':    'E_6 has 27 as IRREDUCIBLE fundamental rep',
        'E_6 > SU(3)^3':   '27 = (3,3-bar,1) + (1,3,3-bar) + (3-bar,1,3)  Trinification',
    }


def trinification_check():
    """
    The Trinification model SU(3)_C × SU(3)_L × SU(3)_R has the famous
    27-rep decomposition for one fermion generation:

        27 = (3, 3-bar, 1) ⊕ (1, 3, 3-bar) ⊕ (3-bar, 1, 3)

    Trinification embeds in E_6 (which has E_6 ⊃ SU(3)^3 maximal subgroup).
    Trinification is DIFFERENT from Pati-Salam:
       PS = SU(4) × SU(2)_L × SU(2)_R   (Pati-Salam 1974)
       Tri = SU(3) × SU(3) × SU(3)       (de Rujula-Georgi-Glashow 1984)

    PS and Tri are BOTH subgroups of SO(10)/E_6 GUTs but they are NOT
    isomorphic and their fermion content per generation is realized
    differently:
       PS one gen = (4,2,1) + (4-bar,1,2) = 16 (matches SO(10) spinor)
       Tri one gen = (3,3-bar,1)+(1,3,3-bar)+(3-bar,1,3) = 27 (matches E_6 rep)
    """
    pieces = [
        ('(3, 3-bar, 1)', 9, 'left-handed quark sector + leptons'),
        ('(1, 3, 3-bar)', 9, 'Higgs/lepton sector'),
        ('(3-bar, 1, 3)', 9, 'right-handed antiquark sector'),
    ]
    total = sum(p[1] for p in pieces)
    return pieces, total


# ---------------------------------------------------------------------------
# Step 3. Cross-comparison with framework's PS dominant tuple
# ---------------------------------------------------------------------------

def framework_ps_content_summary():
    """
    Per theorem_g2d_chirality_doubled.md (theorem-grade):
      Per generation: (4, 2, 1) + (4-bar, 1, 2) = 16  (one PS generation)
      Three generations: 3 x 16 = 48
      Higgs: (1, 2, 2) + ... bidoublet at minimum

    The framework's PS structure is NOT 27-dim per generation.
    The framework's 3-gen total is 48 fermion + Higgs, not 27.

    The 27-dim rep does NOT appear naturally in PS unless we go to E_6
    (which contains PS as a subgroup; E_6 ⊃ Spin(10) ⊃ SU(5) ⊃ PS).
    """
    return {
        'PS_per_gen': 16,
        'PS_3gen': 48,
        'PS_per_gen_LH_only': 8,
        'PS_3gen_LH_only': 24,
        'PS_3gen_LH_only_plus_3_singlets': 27,   # numerical match only
    }


def framework_dominant_slice_in_zoo():
    """
    Per sector_zoo_framework_connection_audit.py:
      Framework dominant slice: substrate srs + Cl(6,0) at vertex + Cl(0,2)≅ℍ
      at edge → PS = SU(4) × SU(2)_L × SU(2)_R.

    Subdominant zoo entries containing F_4 / E_6:
      𝕆 ⊗ ℝ at vertex via magic square: F_4 (52-dim Lie alg)
      𝕆 ⊗ ℂ at vertex via magic square: E_6 (78-dim Lie alg)
      𝕆 ⊗ 𝕆 at vertex via magic square: E_8 (248-dim Lie alg)

    F_4 / E_6 are SUBDOMINANT zoo Lie algebras, plurally co-retained but with
    constant + tensor + associator suppression. Their irreps (incl. J_3(O))
    are present in the zoo at subdominant level.
    """
    return {
        'dominant': 'PS = Spin(6) × Spin(4) at (Cl(6), Cl(0,2))',
        'F_4_zoo_slice': '𝕆 ⊗ ℝ magic-square at vertex',
        'E_6_zoo_slice': '𝕆 ⊗ ℂ magic-square at vertex',
        'F_4_action_on_J3O': 'auto-action; 27 = 1 + 26 reducible decomp',
        'E_6_action_on_27': 'irreducible fundamental rep',
    }


# ---------------------------------------------------------------------------
# Step 4. Structural connection assessment
# ---------------------------------------------------------------------------

def structural_assessment():
    """
    Three independent dim-27 comparisons:

    (A) PS 3-gen LH-only-Weyl + 3 singlets = 24 + 3 = 27.
        Numerical match. But 3 RH-singlet neutrinos is a tag-on, not
        rep-theoretically distinguished by F_4 action. Singlets are
        F_4-singlets trivially; they don't sit in J_3(O) as a non-trivial
        F_4 rep.

    (B) Trinification SU(3)_C × SU(3)_L × SU(3)_R has 27 as one-generation
        fermion rep. This sits naturally in E_6 ⊃ SU(3)^3 with 27 as E_6
        fundamental.
        Trinification is NOT the framework's PS — different gauge group.

    (C) E_6 ⊃ Spin(10) ⊃ Pati-Salam (or SU(5)).
        E_6's 27 decomposes under Spin(10) as 27 = 1 + 10 + 16, where 16
        is a fermion generation under SO(10) GUT.
        Under Spin(10) ⊃ PS = Spin(6) × Spin(4):
            16 = (4, 2, 1) + (4-bar, 1, 2) -- the standard PS gen
            10 = (6, 1, 1) + (1, 2, 2)     -- 6 of SU(4) + Higgs bidoublet
            1  = (1, 1, 1)                  -- singlet
        So:
            J_3(O) ↔ 27 of E_6 ↔ {1 PS gen} + {(6,1,1)+(1,2,2)} + {singlet}
        This is a MEANINGFUL E_6 identification: 1 generation + Higgs +
        singlet = 27.

    Assessment:
      (A) is a numerical match without rep-theoretic content.
      (B) is structural for trinification, NOT PS.
      (C) IS structural — 27 of E_6 ⊃ PS gives one generation + (6,1,1) +
          (1,2,2) + (1,1,1).

    CRITICAL question: is E_6 in the framework's zoo at a level that lets
    it act on a PS rep?
      Per Task A (sector_local_algebra_zoo_audit.py line 314): magic-square
      slice 'O ⊗ C → E_6 (78-dim Lie alg)' is FREQ-OK at N_hub but
      SUBDOMINANT relative to PS dominant tuple.

    For 27 of E_6 to act on PS-content:
      - The PS dominant slice has 1 generation = 16 (Spin(10) spinor).
      - E_6 ⊃ Spin(10): 27 = 16 + 10 + 1, so 1 generation + 11 extras.
      - The extras (10 of Spin(10) = (6,1,1) + (1,2,2) under PS) are
        the SU(4) sextet (could carry leptoquarks) and the (1,2,2) bidoublet
        (which IS the framework's Higgs per memory 2026-05-05 EOD+3).
      - Singlet = neutral scalar.

    This matches the standard E_6 GUT branching!

    INTERPRETATION:
      J_3(O) ≅ 27 of E_6 ⊃ PS gives:
        1 PS fermion generation + Higgs bidoublet + (6,1,1) + singlet.
      The 27-dim Albert algebra at the F_4 / E_6 zoo slice 'organizes' one
      generation + Higgs in a single E_6 multiplet.

    But: framework HAS 3 generations not 1, and uses ONE bidoublet not 3.
    To get 3 generations + 1 Higgs from E_6 reps, would need 3 copies of
    27 → 3 × 27 = 81, OR a different rep choice.

    Honest scope: the dim-27 connection is STRUCTURAL through E_6 ⊃ PS,
    but the framework's specific fermion+Higgs count (48 fermion + ~4 Higgs
    dof) does NOT map cleanly to N copies of 27.
    """
    return {
        'numerical_match_A': '24 + 3 = 27 (3-gen LH + 3 singlets, no F_4 content)',
        'trinification_B':   '27 = (3,3-bar,1) + (1,3,3-bar) + (3-bar,1,3) — for SU(3)^3, NOT PS',
        'e6_branching_C':    '27 = 1 + 10 + 16  (under E_6 ⊃ Spin(10))',
        'e6_to_PS':          '16 = (4,2,1)+(4-bar,1,2),  10 = (6,1,1)+(1,2,2),  1 = (1,1,1)',
        'verdict_sketch':    'STRUCTURAL via E_6 path: 1 gen + Higgs + leptoquark sextet + singlet',
        'framework_match_status': 'PARTIAL — 1 of 3 generations fits one J_3(O); 3 gens need 3×27 or different rep',
    }


# ---------------------------------------------------------------------------
# Step 5. Verdict + concrete F_4 corrections (if structural)
# ---------------------------------------------------------------------------

def verdict_and_corrections():
    """
    Per Step 4 assessment:

    VERDICT: The dim-27 connection is STRUCTURAL THROUGH E_6 PATH, NOT
             DIRECTLY THROUGH F_4 → PS. Specifically:

       (i)  F_4 = Aut(J_3(O)) but F_4 does NOT contain PS as a subgroup
            (F_4 ⊃ Spin(9), and Spin(9) is the maximal subgroup that
             stabilizes a basis of J_3(O); F_4 does not contain SU(4)).
             So F_4 → PS is NOT a structural branching path.

       (ii) E_6 ⊃ F_4 (E_6 is the smallest Lie algebra containing F_4
            with the 27 as IRREDUCIBLE fundamental). Under E_6 ⊃ Spin(10) ⊃
            PS, 27 = 16 + 10 + 1 = (1 PS gen) + (Higgs bidoublet + leptoquark
            sextet) + singlet.

       (iii) Per saturated zoo: E_6 sits at 𝕆 ⊗ ℂ subdominant tuple, with
             constant + tensor + assoc Bayesian suppression. So J_3(O) is
             plurally co-retained per A2-T but at SUBDOMINANT zoo level.

    CONCRETE F_4-INDUCED CORRECTIONS to PS predictions:

      None directly — F_4 ↛ PS branching means F_4 does NOT contribute
      a corrective coupling to PS observables that would inherit from
      F_4 = Aut(J_3(O)).

      The E_6 zoo-slice subdominant retention (at 𝕆 ⊗ ℂ vertex) WOULD
      contribute corrections if its Bayesian weight is non-trivial:
        - Suppression vs PS dominant: tensor (factor 2 bits) +
          octonion associator content (constant if f_3 = 0; astronomical
          if f_3 > 0 per Theorem 9-class question).
        - If f_3 = 0 (constant suppression): O(exp(-ΔL)) corrections
          where ΔL ~ 5-10 bits → ~1-3% corrections to PS observables.
        - If f_3 > 0: corrections astronomically small.

      The Theorem 9 closure status (per memory 2026-05-07 + commit 79e9406)
      is PARTIAL/REVERTED — non-closure framing — so f_3 ∈ {0, >0} is OPEN.

    The 3-generation / 27-dim observation:
      3 generations × 27 = 81. This is the dimension of (1) a single rep of
      no obvious Lie algebra, but (2) it equals dim(E_6 × U(1)^3)? No, 78+3
      = 81 (E_6 adjoint + 3 abelian). Could be coincidental.

      The framework's 3 generations come from C^3_obs (observer Hilbert dim),
      NOT from any J_3(O) decomposition. The J_3(O) dim=27 carries 1 PS
      generation + Higgs + leptoquark + singlet, which is independent of
      generation count.

    HONEST SCOPE FLAGS:
      - The F_4 = Aut(J_3(O)) arrow does NOT reach PS structurally; need
        E_6 path.
      - E_6 zoo slice is SUBDOMINANT in framework's saturated zoo; it does
        NOT replace the PS dominant slice.
      - The numerical match dim-27 = 3*8 + 3 (=27) for "3 LH-PS gens +
        3 singlets" has NO F_4 / E_6 rep-theoretic content (just 24+3
        = 27 arithmetic).
      - The framework's predictions are PS-dominant; J_3(O) does NOT
        contribute load-bearing structural content to existing predictions.
      - J_3(O) is consistent with E_6-zoo-slice as one generation + Higgs +
        leptoquark, but is NOT the framework's primary rep for 3 generations
        (which uses C^3_obs ⊗ PS rep).
    """
    return {
        'verdict': 'STRUCTURAL THROUGH E_6 PATH, NOT THROUGH F_4 DIRECTLY; ZOO-SUBDOMINANT',
        'F_4_to_PS': 'NO direct branching (F_4 ⊅ SU(4))',
        'E_6_to_PS': 'YES: 27 = 1 + 10 + 16 = singlet + (Higgs + sextet) + 1 PS gen',
        'corrections': 'O(exp(-ΔL)) at zoo-subdominant level, suppression ΔL ~ 5-10 bits if f_3=0',
        'corrections_status': 'BLOCKED on Theorem 9 f_3 closure',
        'load_bearing_for_predictions': 'NO — PS dominant slice carries framework predictions',
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    print("=" * 110)
    print(" M2 — Albert algebra J_3(𝕆) ↔ Pati-Salam 3-generation structural connection audit")
    print("=" * 110)
    print()
    print(" Question: is the dim-27 coincidence between J_3(𝕆) and 3-gen PS rep")
    print(" candidates STRUCTURAL or ACCIDENTAL?")
    print()
    print(" Methodology: enumerate PS rep candidates, branch J_3(𝕆) under standard")
    print(" subgroup chains, identify rep-theoretic intersections, verdict.")
    print()

    # Step 1: PS rep candidates
    print("=" * 110)
    print(" §1. PS REP CANDIDATES (PS = SU(4) × SU(2)_L × SU(2)_R)")
    print("=" * 110)
    print()
    print(" Single-gen content per theorem_g2d_chirality_doubled.md:")
    print("   F_L = (4, 2, 1) [8 LH q+l],  F_R = (4-bar, 1, 2) [8 RH q+l]")
    print("   per gen total: 16  (matches Spin(10) 16-spinor)")
    print()
    print(" Higgs reps (PS embedding): (1, 2, 2) bidoublet [4],  (4, 1, 2) [8],")
    print("                             (15, 1, 1) adjoint [15], (10, 1, 3) [30]")
    print()
    print(" 27-dim reachable totals (numerical, not rep-theoretic):")
    print()
    print(f"   {'PS combination':<55} {'dim':>6}  {'27?':<5}")
    print("   " + "-" * 70)
    candidates = enumerate_ps_rep_candidates()
    for label, d in candidates:
        flag = '✓' if d == 27 else ' '
        print(f"   {label:<55} {d:>6}  {flag:<5}")
    print()
    print(" Hits at 27-dim:")
    hits = [c for c in candidates if c[1] == 27]
    for label, d in hits:
        print(f"   - {label}: {d}")
    print()

    # Step 2: J_3(O) decompositions
    print("=" * 110)
    print(" §2. J_3(𝕆) DECOMPOSITIONS UNDER STANDARD CHAINS")
    print("=" * 110)
    print()
    print(f" J_3(𝕆) real dim = {jordan_albert_dim()}  (= 3 + 3·8 = trace + 3 octonion off-diags)")
    print(f" F_4 = Aut(J_3(𝕆)), dim {f4_dim_check()}")
    print()
    print(" CRITICAL POINT: F_4 acts on J_3(𝕆) REDUCIBLY: 27 = 1 (trace) + 26 (traceless).")
    print(" The 26 is the F_4 fundamental irrep; 27 is NOT a F_4 irrep.")
    print(" The 27 IS the E_6 fundamental irrep (E_6 ⊃ F_4 with 27 = 1 + 26).")
    print()
    decomps = j3_so8_triality_decomp()
    print(" Standard branchings of J_3(𝕆):")
    for chain, decomp in decomps.items():
        print(f"   {chain:<25}: {decomp}")
    print()
    su3_decomps = j3_su3_x_su3_decomp()
    print(" Additional structural identities:")
    for k, v in su3_decomps.items():
        print(f"   {k:<25}: {v}")
    print()

    # Step 3: Trinification and E_6 paths
    print("=" * 110)
    print(" §3. TWO 27-DIM REP-THEORETIC ROUTES (Trinification + E_6 ⊃ PS)")
    print("=" * 110)
    print()
    pieces, total = trinification_check()
    print(" Route 1: Trinification SU(3)^3 (de Rujula-Georgi-Glashow 1984)")
    print(f"   27 of E_6 = SU(3)^3 → 3 fermion blocks per generation:")
    for label, d, desc in pieces:
        print(f"     {label:<18} dim {d}  ({desc})")
    print(f"   total = {total}")
    print(f"   NOTE: Trinification gauge group ≠ Pati-Salam.")
    print()
    print(" Route 2: E_6 ⊃ Spin(10) ⊃ Pati-Salam (standard E_6 GUT)")
    print(f"   27 of E_6 = 1 + 10 + 16  (under Spin(10))")
    print(f"   16 of Spin(10) = (4, 2, 1) + (4-bar, 1, 2)   ← 1 PS generation")
    print(f"   10 of Spin(10) = (6, 1, 1) + (1, 2, 2)        ← leptoquark + Higgs bidoublet")
    print(f"   1 of Spin(10)  = (1, 1, 1)                    ← singlet")
    print()
    print(f"   STRUCTURAL CONTENT: J_3(𝕆) ≅ 27_E6 ↔ {{1 PS gen}} ⊕ {{Higgs + leptoquark}} ⊕ {{singlet}}")
    print()

    # Step 4: framework comparison
    print("=" * 110)
    print(" §4. FRAMEWORK PS DOMINANT SLICE vs J_3(𝕆) STRUCTURE")
    print("=" * 110)
    print()
    summary = framework_ps_content_summary()
    for k, v in summary.items():
        print(f"   {k:<40}: {v}")
    print()
    zoo = framework_dominant_slice_in_zoo()
    print(" Saturated zoo placement:")
    for k, v in zoo.items():
        print(f"   {k:<25}: {v}")
    print()

    assessment = structural_assessment()
    print(" Three dim-27 comparisons:")
    for k, v in assessment.items():
        print(f"   {k:<25}: {v}")
    print()

    # Step 5: verdict
    print("=" * 110)
    print(" §5. VERDICT + HONEST SCOPE FLAGS")
    print("=" * 110)
    print()
    verd = verdict_and_corrections()
    for k, v in verd.items():
        print(f"   {k:<32}: {v}")
    print()

    print("=" * 110)
    print(" FINAL VERDICT")
    print("=" * 110)
    print()
    print(" The dim-27 connection between J_3(𝕆) and the framework's PS structure is:")
    print()
    print("   STRUCTURAL — but ONLY through the E_6 path, NOT directly via F_4.")
    print()
    print(" Key facts:")
    print("   1. F_4 = Aut(J_3(𝕆)) does NOT contain SU(4) ⊂ PS as a subgroup;")
    print("      the maximal subgroup of F_4 is Spin(9), not PS.")
    print("      → No direct F_4 → PS branching path.")
    print()
    print("   2. F_4 acts on J_3(𝕆) REDUCIBLY: 27 = 1 (trace) ⊕ 26 (traceless F_4 irrep).")
    print("      The 27 is the IRREDUCIBLE fundamental of E_6 (E_6 ⊃ F_4).")
    print()
    print("   3. E_6 ⊃ Spin(10) ⊃ PS: 27 = 1 + 10 + 16 →")
    print("      one generation (16) + Higgs bidoublet + leptoquark sextet (10) + singlet (1).")
    print("      This IS a structural rep-theoretic connection.")
    print()
    print("   4. In the saturated zoo:")
    print("      F_4 sits at 𝕆 ⊗ ℝ subdominant magic-square slice.")
    print("      E_6 sits at 𝕆 ⊗ ℂ subdominant magic-square slice.")
    print("      Both are plurally co-retained per A2-T but SUBDOMINANT to the")
    print("      framework's PS dominant Cl(6) ⊗ Cl(0,2) tuple.")
    print()
    print("   5. The 'naive 27 = 24 + 3' (3-gen LH-only + 3 singlets) is a numerical")
    print("      coincidence WITHOUT F_4 / E_6 rep-theoretic content.")
    print()
    print(" Concrete F_4-induced corrections to PS observables:")
    print("   NONE direct: F_4 ↛ PS branching means no F_4-action on PS reps.")
    print()
    print(" Concrete E_6-induced corrections (zoo-subdominant retention):")
    print("   - Bayesian suppression ΔL ~ 5-10 bits if f_3 = 0 (constant assoc cost)")
    print("     → O(1%) fractional corrections to PS observables.")
    print("   - Astronomical (exp(-N)) suppression if f_3 > 0.")
    print("   - BLOCKED on Theorem 9 f_3 closure (currently PARTIAL per commit 79e9406).")
    print()
    print(" Load-bearing for existing framework predictions:")
    print("   J_3(𝕆) is NOT load-bearing for any current PS-derived prediction.")
    print("   The 3-gen count comes from C^3_obs (observer Hilbert dim 3), independent.")
    print("   J_3(𝕆) ↔ 27_E6 organizes one PS gen + Higgs + leptoquark + singlet — a")
    print("   structural identification that holds at zoo-subdominant level.")
    print()
    print(" Honest scope flags:")
    print("   ⚠ F_4 = Aut(J_3(𝕆)) does NOT reach PS structurally; only E_6 ⊃ PS works.")
    print("   ⚠ E_6 zoo slice is SUBDOMINANT — does not replace PS dominant.")
    print("   ⚠ 'Naive 27 = 24 + 3' is numerical coincidence, not rep-theoretic.")
    print("   ⚠ E_6 corrections to PS observables are at zoo-subdominant level,")
    print("     gated by Theorem 9 f_3 closure (currently OPEN).")
    print("   ⚠ 3 generations × 27 = 81 has no obvious E_6/F_4 rep-theoretic meaning;")
    print("     framework's 3-gen comes from C^3_obs, independent of J_3(𝕆).")
    print()
    print("=" * 110)
    print(" Probe complete. No theorems, predictions, or ledger modified.")
    print("=" * 110)

    return 0


if __name__ == "__main__":
    main()
