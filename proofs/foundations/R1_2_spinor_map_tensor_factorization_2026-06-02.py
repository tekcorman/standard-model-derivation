#!/usr/bin/env python3
"""
R1_2_spinor_map_tensor_factorization_2026-06-02.py
==================================================
The spinor map (R1.2 / "Phase 1b"): what the 48-mode ↔ 48-spinor map actually
IS, and why a per-mode bijection is ill-posed.

CONTEXT. Phase 1 (`per_weyl_spinor_dictionary_2026-05-27`) tagged each of the
48 SM Weyl spinors by (γ_7, Q-pattern→color, isospin, generation) and assigned
each to a walker FAMILY (h_P / h_Γ / h_H). It explicitly left "Phase 1b" open:
(1) which specific mode within a family, (2) per-vertex localization, (3) which
edge supplies isospin. The multiplicity audit (2026-06-02) showed the two
48-totals factor incompatibly and the per-family ratios are non-uniform
(h_P 8→42, h_Γ 8→3, h_H 8→3). This probe resolves what the map is.

TWO STRUCTURAL FACTS that fix the answer:
  (O1) The V_Ram ≅ Cl(6) Fock iso (T1) intertwines ONLY C_3, and is unique
       only up to **U(4) × U(2) × U(2)** within-isotype basis freedom. So an
       individual Hashimoto mode is NOT canonically assigned to an individual
       Cl(6) (color, chirality) state — that assignment is a gauge choice
       inside each C_3-isotype, not a derived fact.
  (O2) The fermionic Hashimoto modes carry quantum numbers {C_3-isotype,
       saddle/arg (= chirality-TYPE)} and, via the iso, {color, γ_7}. They do
       NOT carry **isospin** (an edge Cl(0,2)→SU(2) label) or **generation**
       (an observer C³ / R3 label). Those two factors live on structures
       ORTHOGONAL to the directed-edge Hashimoto modes.

CONSEQUENCE. The 48-spinor labeling is a TENSOR FACTORIZATION over three
independent structures:

    spinor  =  [ saddle-mode: color × chirality × fermion-class ]      (Hashimoto)
            ⊗  [ isospin: up / down ]                                  (edge Cl(0,2))
            ⊗  [ generation: 1 / 2 / 3 ]                               (observer C³)

It is canonical at the (color × chirality × isospin × generation × family)
grain — which is exactly Phase 1. A finer per-MODE bijection is ILL-POSED:
(O1) leaves the within-isotype mode↔(color,chir) assignment gauge-free, and
(O2) means isospin & generation are not mode quantum numbers at all. The
"48↔48" is a coincidence of dimensions plus a C_3-level iso, refined by two
external factors — never a mode-by-mode identification.

This probe (a) reconstructs the canonical map and verifies the Phase-1 tallies,
and (b) states the two obstructions explicitly.
"""

import sys
from pathlib import Path
from itertools import product

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from R1_1_cl6_fock_su4_PS_decomposition_probe import build_gamma, TOL

np.set_printoptions(precision=4, suppress=True, linewidth=140)


def section(t):
    print("\n" + "=" * 90 + f"\n {t}\n" + "=" * 90)


# ---------------------------------------------------------------------------
# Cl(6) Fock: label the 8 states by (color via Q-pattern, chirality via γ_7)
# ---------------------------------------------------------------------------

def cl6_labels():
    """Return the 8 Cl(6) Fock basis states labeled (γ_7, color), built from the
    common eigenbasis of (γ_7, Q_1, Q_2, Q_3). Verifies the Q_i quaternion
    algebra and the 4-valid-sign-pattern → {ℓ,r,g,b} rule (per_weyl_spinor §2.3)."""
    G = build_gamma()
    G7 = -1j * G[1] @ G[2] @ G[3] @ G[4] @ G[5] @ G[6]
    Q1 = G[3] @ G[4] @ G[5] @ G[6]
    Q2 = G[1] @ G[2] @ G[5] @ G[6]
    Q3 = G[1] @ G[2] @ G[3] @ G[4]
    # algebra checks (V_Ram-iso T4)
    I8 = np.eye(8, dtype=complex)
    assert np.allclose(Q1 @ Q1, I8, atol=TOL) and np.allclose(Q2 @ Q2, I8, atol=TOL)
    assert np.allclose(Q1 @ Q2 @ Q3, -I8, atol=TOL), "Q1Q2Q3 = -I (only 4 sign patterns)"
    assert all(np.allclose(Qi @ G7 - G7 @ Qi, 0, atol=TOL) for Qi in (Q1, Q2, Q3))
    # simultaneous eigenbasis of commuting Hermitian (G7, Q1, Q2)
    M = 1.0 * G7 + 3.7 * Q1 + 11.3 * Q2     # generic combo splits the joint spectrum
    _, vecs = np.linalg.eigh(M)
    color_of = {(+1, +1, -1): 'r', (+1, -1, +1): 'g', (-1, +1, +1): 'b', (-1, -1, -1): 'ℓ'}
    states = []
    for k in range(8):
        v = vecs[:, k]
        g7 = int(round(np.real(v.conj() @ G7 @ v)))
        q1 = int(round(np.real(v.conj() @ Q1 @ v)))
        q2 = int(round(np.real(v.conj() @ Q2 @ v)))
        q3 = int(round(np.real(v.conj() @ Q3 @ v)))
        states.append({'g7': g7, 'color': color_of[(q1, q2, q3)]})
    return states


# ---------------------------------------------------------------------------
def main():
    section("STEP 1 — Cl(6) Fock: 8 states = 4 colors × 2 chiralities (verified)")
    states = cl6_labels()
    from collections import Counter
    col_count = Counter(s['color'] for s in states)
    chir_count = Counter(s['g7'] for s in states)
    print(f"  Q_i quaternion algebra + Q1Q2Q3=-I verified (only 4 sign patterns → 4 colors).")
    print(f"  colors present: {dict(col_count)}  (expect each of ℓ,r,g,b twice)")
    print(f"  γ_7 split: {dict(chir_count)}  (expect +1:4, -1:4)")
    assert set(col_count) == {'ℓ', 'r', 'g', 'b'} and all(c == 2 for c in col_count.values())
    assert chir_count[1] == 4 and chir_count[-1] == 4

    section("STEP 2 — Build the canonical spinor map by TENSOR FACTORIZATION")
    print("  spinor = [color × chirality] (Cl(6) Fock, via V_Ram iso)")
    print("         ⊗ [isospin up/down] (edge Cl(0,2)→SU(2))")
    print("         ⊗ [generation 1/2/3] (observer C³, R3)")
    print("  saddle/family assignment rule (chir-7 theorem): a color-singlet (ℓ),")
    print("  up-isospin state = a NEUTRINO → chir-7 saddle (γ_7=+1 → Γ as ν_L;")
    print("  γ_7=−1 → H as ν_R^c). Everything else = charged → P (h_P).")
    print()

    def sm_name(color, g7, iso):
        L = 'L' if g7 == +1 else 'R'
        if color == 'ℓ':
            base = ('ν' if iso == 'up' else 'e')
        else:
            base = ('u' if iso == 'up' else 'd')
        # right-handed (γ_7=-1) SM Weyl appear as conjugates
        return f"{base}_{L}" + ("" if g7 == +1 else "^c") + (f"({color})" if color != 'ℓ' else "")

    def saddle(color, g7, iso):
        if color == 'ℓ' and iso == 'up':          # neutrino
            return ('Γ', 'h_Γ') if g7 == +1 else ('H', 'h_H')
        return ('P', 'h_P')

    rows = []
    for gen in (1, 2, 3):
        for s in states:
            for iso in ('up', 'down'):
                sd, fam = saddle(s['color'], s['g7'], iso)
                rows.append({'gen': gen, 'color': s['color'], 'g7': s['g7'],
                             'iso': iso, 'name': sm_name(s['color'], s['g7'], iso),
                             'saddle': sd, 'family': fam})
    assert len(rows) == 48
    # show generation 1
    print("  Generation-1 block (16 rows):")
    print(f"    {'name':<10} {'color':<5} {'γ_7':>4} {'isospin':<7} {'saddle':<6} {'family'}")
    for r in rows[:16]:
        print(f"    {r['name']:<10} {r['color']:<5} {r['g7']:>+4} {r['iso']:<7} {r['saddle']:<6} {r['family']}")

    section("STEP 3 — Verify the Phase-1 tallies fall out of the factorization")
    fam_count = Counter(r['family'] for r in rows)
    chir = Counter('L' if r['g7'] == +1 else 'R' for r in rows)
    colr = Counter(r['color'] for r in rows)
    print(f"  walker-family tally: {dict(fam_count)}")
    print(f"    → h_P:42 charged, h_Γ:3 ν_L, h_H:3 ν_R  (matches walker_class_dictionary §3.4)")
    print(f"  chirality tally: {dict(chir)}  (expect 24 L + 24 R)")
    print(f"  color tally:     {dict(colr)}  (expect 12 each of ℓ,r,g,b)")
    assert fam_count['h_P'] == 42 and fam_count['h_Γ'] == 3 and fam_count['h_H'] == 3
    assert chir['L'] == 24 and chir['R'] == 24
    assert all(c == 12 for c in colr.values())
    print("  ✓ all Phase-1 tallies reproduced from the tensor factorization.")

    section("STEP 4 — Why a per-MODE bijection is ILL-POSED (the two obstructions)")
    print("""\
  (O1) ISO GAUGE FREEDOM. The V_Ram ≅ Cl(6) Fock iso (T1 construction) is fixed
       only at the C_3 level: U ρ_V U* = ρ_C, unique up to U(4)×U(2)×U(2)
       within-isotype basis choice. So WHICH individual P-saddle mode is the
       'green-chiral' Cl(6) state (vs blue-chiral, etc.) is a free rotation
       inside each C_3 isotype — not a derived assignment. Pinning it would
       require intertwining a LARGER symmetry than C_3 (e.g. full Aut(K_4)=S_4
       or the Spin(6) action), which the iso program has not done.

  (O2) ORTHOGONAL QUANTUM NUMBERS. The fermionic Hashimoto modes carry
       {C_3-isotype, saddle/arg (chirality-type)} → {color, γ_7} via the iso.
       They do NOT carry isospin or generation:
         • isospin  lives on the per-edge Cl(0,2) (theorem_g2_edge_qubit_su2),
         • generation lives on the observer C³ (R3),
       both ORTHOGONAL to the directed-edge Hashimoto eigenspace. So the mode
       label set is strictly coarser than the spinor label set — there are not
       enough mode quantum numbers to index 48 spinors one-to-one.

  COUNTING THE OBSTRUCTION (O2): the 24 fermionic modes carry
       (color×γ_7 = 8) × (3 fermion-families) = 24 worth of {color,chir,class},
  while the 48 spinors need additionally (isospin ×2)(generation ×3)/(class
  already counted) — the missing factor is exactly the isospin and generation
  that are NOT mode degrees of freedom. A bijection cannot exist.""")

    section("VERDICT — what 'the spinor map' is")
    print("""\
  THE MAP IS A TENSOR FACTORIZATION, canonical at the Phase-1 grain:

     48 spinors  =  [ color × chirality ]_(Cl(6) Fock, via C_3-level V_Ram iso)
                 ⊗  [ isospin ]_(per-edge Cl(0,2))
                 ⊗  [ generation ]_(observer C³),
     with saddle/family fixed by chirality-type (ν = color-singlet up-isospin
     → Γ/H; all else → P).

  This reproduces every Phase-1 tally (42/3/3 families, 24L/24R, 12 per color)
  directly from the three factors — built and verified above.

  A finer per-MODE bijection (the open 'Phase 1b') is NOT merely undone — it is
  ILL-POSED:
    • (O1) the V_Ram iso leaves the within-isotype mode↔(color,chir) assignment
      gauge-free (U(4)×U(2)×U(2)); only a larger intertwined symmetry could fix
      it, and none is established;
    • (O2) isospin and generation are not Hashimoto-mode quantum numbers (they
      live on edges and on the observer), so modes cannot index spinors 1-to-1.

  This dissolves the apparent puzzle of the non-uniform ratios (h_P 8→42,
  h_Γ 8→3, h_H 8→3): those are NOT per-mode multiplicities. They are the
  chirality-type PARTITION of the 48 tensor-factorized spinors across the 3
  fermion saddle-families — 42 charged, 3 ν_L, 3 ν_R — independent of the
  8-mode dimension of each family's V_Ram eigenspace.

  Bottom line for the unification claim: '48↔48' is genuine as a tensor
  factorization with each factor sourced (Cl(6) Fock, edge Cl(0,2), observer
  C³); it is NOT a mode-by-mode isomorphism, and cannot be made one without
  new intertwined-symmetry input that the framework does not currently have.
""")
    return True


if __name__ == "__main__":
    main()
    raise SystemExit(0)
