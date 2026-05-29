#!/usr/bin/env python3
"""
Theorem-grade verification of the substrate Bloch invariants used in the
G_sub structural form.

Context. `proofs/foundations/lorentz_sig_g_sub_lichnerowicz_closure.py`
historically declared structural identities G_sub = 1/(8π³) and later
1/(16π³), both based on the paramagnetic-only static elastic susceptibility.
Per 2026-04-28 PM Update 2 of an internal working note,
both identifications were RETRACTED: static elastic modulus (paramagnetic +
diamagnetic ≈ 0.26 near-cancellation) ≠ graviton kinetic coefficient. The
correct G_sub comes from the dynamic matter 1-loop polarization (multi-
session). This script's role is now narrower: it CAS-verifies the substrate-
side Bloch invariants ⟨Tr(H²)⟩, ⟨Tr(H⁴)⟩, ⟨Tr(R_4²)⟩ exhaustively, which
are theorem-grade ingredients regardless of how G_sub finally closes:
    (A) ⟨Tr(H(k)²)⟩_BZ = 2|E| = 12 (bond-count sum rule, trivial).
    (B) ⟨Tr(H(k)⁴)⟩_BZ = 60 (closed-walk count with zero net displacement).
    (C) v_F = 1/2 (Γ-cone Fermi velocity, separately theorem-grade).

The previous script numerically confirmed (A) and (B) on a 30³ grid, and
asserted the analytic walk decomposition (3 bounces + 12 three-vertex + 0
four-cycles per atom = 15 per atom × 4 = 60). This script promotes (B) to
theorem-grade rigor by exhaustive enumeration of every length-4 walk on
the srs primitive cell, with the zero-net-displacement filter applied
explicitly. The walk classification (bounces / three-vertex i→j→i→j'→i /
three-vertex i→j→k→j→i / four-cycles) is computed and cross-checked.

Purpose. Tightens the substrate-side ingredient of G_sub's structural
form from "asserted" to "exhaustively enumerated and CAS-verified". The
*bridge* to G_sub via Sakharov polarization tensor (the structural-fit
gap) is unaffected — that remains the research-level closure step per
an internal working note.

Reads:
- Row 6 (srs lattice, theorem-grade) → CELL_EDGES bond list.
- Row 7 (|E|=6, theorem-grade) → number of undirected edges.
- Row 16 (Cl(6)/|V|=4, theorem-grade) → number of atoms in primitive cell.

Outputs:
- ⟨Tr(H²)⟩_BZ = 12 verified by walk enumeration (bond reversal).
- ⟨Tr(H⁴)⟩_BZ = 60 verified by walk enumeration with displacement filter.
- Walk-type decomposition: 12 + 24 + 24 + 0 = 60.
- ⟨Tr(R_4²)⟩_BZ = 24 derived analytically from (A) + (B).
- Substrate-side Bloch invariants verified at theorem grade (G_sub closure path is via dynamic matter loop, not these invariants directly — see retraction above).
"""
from __future__ import annotations

import sympy as sp
from itertools import product

# srs primitive cell: 4 atoms (Wyckoff 8a), 6 undirected edges with the
# explicit cell-offset vectors below. This is the same bond list used in
# `lorentz_sig_g_sub_lichnerowicz_closure.py` and `srs_dirac_cone_velocities.py`,
# inherited from Row 6 (srs identification) at theorem grade.
N_ATOMS = 4
CELL_EDGES = [
    (0, 1, (1, 1, 1)),
    (0, 2, (1, 1, 1)),
    (0, 3, (1, 1, 1)),
    (1, 2, (-1, 0, 0)),
    (1, 3, (0, 1, 0)),
    (2, 3, (0, 0, -1)),
]

# Build the directed bond list: for each undirected edge (s, t, c) we
# include both (s, t, c) and (t, s, -c). The Bloch sum rule
# ⟨e^{2πi k·Δ}⟩_BZ = δ_{Δ, 0} then projects onto walks with zero net
# cell offset.
DIRECTED_BONDS = []
for s, t, c in CELL_EDGES:
    DIRECTED_BONDS.append((s, t, c))
    DIRECTED_BONDS.append((t, s, tuple(-x for x in c)))

# Adjacency-style outgoing bond table indexed by source atom.
OUTGOING = {a: [] for a in range(N_ATOMS)}
for s, t, c in DIRECTED_BONDS:
    OUTGOING[s].append((t, c))


def closed_walks(length: int):
    """Yield all length-`length` closed walks (i₀, i₁, …, i_{length}=i₀)
    with the directed-bond list above, returning (vertex sequence, total
    cell-offset vector). No displacement filtering yet — we project later."""
    for start in range(N_ATOMS):
        # Enumerate (length)-step walks beginning at `start` and returning
        # to `start` on the last step.
        def recurse(path, offset, steps_left):
            if steps_left == 0:
                if path[-1] == start:
                    yield list(path), offset
                return
            for nxt, c in OUTGOING[path[-1]]:
                yield from recurse(
                    path + [nxt],
                    tuple(offset[i] + c[i] for i in range(3)),
                    steps_left - 1,
                )

        yield from recurse([start], (0, 0, 0), length)


def filter_zero_displacement(walks):
    """Keep only walks whose cell-offset sum is (0, 0, 0)."""
    return [w for w in walks if w[1] == (0, 0, 0)]


def classify_length4_walk(path):
    """Bucket a length-4 closed walk by its vertex-pattern.

    Possible patterns for i₀ → i₁ → i₂ → i₃ → i₀ (closed; i₀ = i_4 fixed):
      'bounce'         — i₁ = i₃ AND i₂ = i₀ (visits 2 atoms, alternating).
      '3v_outback_a'   — i₂ = i₀ AND i₁ ≠ i₃ (visits 3 atoms; i₀–i₁–i₀–i₃–i₀).
      '3v_outback_b'   — i₁ = i₃ AND i₂ ≠ i₀ (visits 3 atoms; i₀–i₁–i₂–i₁–i₀).
      '4cycle'         — all of i₀, i₁, i₂, i₃ distinct.
    """
    i0, i1, i2, i3, i4 = path
    assert i4 == i0, "expected closed walk"
    distinct = {i0, i1, i2, i3}
    if i1 == i3 and i2 == i0:
        return "bounce"
    if i2 == i0 and i1 != i3:
        return "3v_outback_a"
    if i1 == i3 and i2 != i0:
        return "3v_outback_b"
    if len(distinct) == 4:
        return "4cycle"
    raise AssertionError(f"unexpected length-4 walk pattern: {path}")


def section(title):
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def step_A_length2():
    section("Step A — ⟨Tr(H²)⟩_BZ via length-2 walk enumeration (theorem-grade)")
    walks2 = list(closed_walks(2))
    zero2 = filter_zero_displacement(walks2)
    print()
    print(f"  Total length-2 closed walks (any displacement):            {len(walks2)}")
    print(f"  Length-2 closed walks with zero net cell offset:           {len(zero2)}")
    print(f"  Bond-count sum rule predicts: 2|E| = {2 * len(CELL_EDGES)} (each undirected edge"
          " contributes")
    print(f"     forward and backward, both at zero displacement).")
    print()
    assert len(zero2) == 12, f"expected 12 zero-displacement length-2 walks, got {len(zero2)}"
    print("  ✓ ⟨Tr(H²)⟩_BZ = 12 verified by exhaustive walk enumeration.")
    return len(zero2)


def step_B_length4():
    section("Step B — ⟨Tr(H⁴)⟩_BZ via length-4 walk enumeration (theorem-grade)")
    walks4 = list(closed_walks(4))
    zero4 = filter_zero_displacement(walks4)

    classified = {"bounce": [], "3v_outback_a": [], "3v_outback_b": [], "4cycle": []}
    for path, _ in zero4:
        classified[classify_length4_walk(path)].append(path)

    print()
    print(f"  Total length-4 closed walks (any displacement):            {len(walks4)}")
    print(f"  Length-4 closed walks with zero net cell offset:           {len(zero4)}")
    print()
    print(f"  Classification of zero-displacement length-4 walks:")
    print(f"    bounces (i→j→i→j→i, visit 2 atoms):                      {len(classified['bounce'])}")
    print(f"    3-vertex i→j→i→j'→i (visit 3 atoms, j ≠ j'):              {len(classified['3v_outback_a'])}")
    print(f"    3-vertex i→j→k→j→i (visit 3 atoms, k ≠ i):                {len(classified['3v_outback_b'])}")
    print(f"    4-cycles (visit 4 distinct atoms):                       {len(classified['4cycle'])}")
    print()

    # Per-atom decomposition cross-check
    print(f"  Per-atom counts (uniform across atoms by symmetry):")
    for cls in ("bounce", "3v_outback_a", "3v_outback_b", "4cycle"):
        per_atom = len(classified[cls]) / N_ATOMS
        print(f"    {cls:18s} = {len(classified[cls])} = {N_ATOMS} × {int(per_atom)}")
    print()

    # Check 4-cycles closure under displacement
    walks4_4cyc_any = [
        (path, off) for path, off in walks4
        if len({path[0], path[1], path[2], path[3]}) == 4
    ]
    print(f"  4-cycle structural finding:")
    print(f"    Total length-4 walks with 4 distinct vertices (any displ.): {len(walks4_4cyc_any)}")
    print(f"    Those that close at zero net displacement:                  {len(classified['4cycle'])}")
    print(f"    → srs's BCC geometry suppresses ALL 4-cycle closures from")
    print(f"      contributing to the Bloch sum rule. This is a structural")
    print(f"      property of the I4₁32 cubic lattice with the given Wyckoff")
    print(f"      8a embedding — different chiral-cubic lattices would give")
    print(f"      a different ⟨Tr(H⁴)⟩.")
    print()

    assert len(zero4) == 60, f"expected 60 zero-displacement length-4 walks, got {len(zero4)}"
    assert len(classified["bounce"]) == 12
    assert len(classified["3v_outback_a"]) == 24
    assert len(classified["3v_outback_b"]) == 24
    assert len(classified["4cycle"]) == 0
    print("  ✓ ⟨Tr(H⁴)⟩_BZ = 60 verified by exhaustive walk enumeration with")
    print("    displacement filter. Decomposition: 12 + 24 + 24 + 0 = 60.")
    return len(zero4)


def step_C_R4_norm(trH2: int, trH4: int):
    section("Step C — ⟨Tr(R_4²)⟩_BZ derived from ⟨Tr(H²)⟩, ⟨Tr(H⁴)⟩")
    n = N_ATOMS
    # R_4(k) := H(k)² - (Tr(H(k)²)/n_atoms) I = H(k)² - 3 I
    # ⟨Tr(R_4²)⟩ = ⟨Tr(H⁴)⟩ - 6 ⟨Tr(H²)⟩ + 9 n_atoms (since (a I) trace = a · n)
    # Using Tr(H²)/n = 12/4 = 3 ⇒ R_4 = H² - 3 I, and (H² - 3I)² = H⁴ - 6H² + 9I.
    trR4_sq = trH4 - 6 * trH2 + 9 * n
    print()
    print(f"  Definition: R_4(k) := H(k)² − (⟨Tr(H²)⟩/n_atoms) I_{{n_atoms}}")
    print(f"  For srs: Tr(H²)/n_atoms = 12/4 = 3, so R_4(k) = H(k)² − 3·I_4.")
    print()
    print(f"  ⟨Tr(R_4²)⟩_BZ = ⟨Tr(H⁴)⟩ − 6·⟨Tr(H²)⟩ + 9·n_atoms")
    print(f"                = {trH4} − 6·{trH2} + 9·{n}")
    print(f"                = {trH4} − {6*trH2} + {9*n}")
    print(f"                = {trR4_sq}")
    assert trR4_sq == 24
    print()
    print("  ✓ ⟨Tr(R_4²)⟩_BZ = 24 (theorem-grade combinatorial integer).")
    return trR4_sq


def step_D_structural_form(trH2: int, trH4: int, trR4_sq: int):
    section("Step D — G_sub structural form using verified Bloch invariants")
    pi = sp.pi
    v_F = sp.Rational(1, 2)                          # Row 6 + Γ-cone (theorem-grade)
    # V_BZ for srs's BCC primitive cell:
    # V_primitive = 1/2 (BCC primitive vectors a_1, a_2, a_3 with |det| = 1/2)
    # V_BZ_BCC = (2π)³ / V_primitive = (2π)³ / (1/2) = 16π³
    V_BZ_BCC = sp.Integer(16) * pi**3
    G_sub_form = sp.Rational(trR4_sq, 1) * v_F / (sp.Rational(trH2, 1) * V_BZ_BCC)
    G_sub_form = sp.simplify(G_sub_form)
    print()
    print(f"  Structural form (per `lorentz_sig_g_sub_lichnerowicz_closure.py`,")
    print(f"  CORRECTED 2026-04-28 PM with proper BCC V_BZ):")
    print(f"    G_sub_form = ⟨Tr(R_4²)⟩_BZ · v_F / (⟨Tr(H²)⟩_BZ · V_BZ_BCC)")
    print(f"               = {trR4_sq} · ({v_F}) / ({trH2} · 16π³)")
    print(f"               = {G_sub_form}")
    print(f"               ≈ {float(G_sub_form):.8f}")
    print()
    expected = sp.Rational(1, 16) / pi**3
    print(f"  Result: G_sub = 1/(16π³) ≈ {float(expected):.8f}")
    print(f"  Match: {sp.simplify(G_sub_form - expected) == 0}")
    print()
    print(f"  Numerical ratio simplification: ⟨Tr(R_4²)⟩/⟨Tr(H²)⟩ = {trR4_sq}/{trH2} = {trR4_sq // trH2}.")
    print(f"  So G_sub_form = ({trR4_sq // trH2})·v_F / V_BZ_BCC = 2·(1/2)/(16π³) = 1/(16π³).")
    print()
    print(f"  Earlier (uncorrected) result used V_BZ = (2π)³ = 8π³ (simple-cubic")
    print(f"  convention), giving G_sub = 1/(8π³). The correction to V_BZ_BCC = 16π³")
    print(f"  for srs's BCC primitive (V_primitive = 1/2) halves the value.")
    print()
    print(f"  This ratio depends on srs's specific 4-cycle suppression:")
    print(f"    ⟨Tr(R_4²)⟩/⟨Tr(H²)⟩ = ⟨Tr(H⁴)⟩/⟨Tr(H²)⟩ − 6 + 9·n_atoms/⟨Tr(H²)⟩")
    print(f"                        = {trH4}/{trH2} − 6 + 9·{N_ATOMS}/{trH2}")
    print(f"                        = 5 − 6 + 3 = 2")
    print(f"  The {trH4}/{trH2} = 5 term is srs-specific (different cubic")
    print(f"  lattices with closing 4-cycles would push it higher).")
    return G_sub_form


def step_E_status():
    section("Step E — Theorem-grade status update")
    print("""
  Substrate-side Bloch invariants of G_sub's structural form:
    ✓ ⟨Tr(H(k)²)⟩_BZ = 2|E| = 12      (this script, exhaustive walk enum.)
    ✓ ⟨Tr(H(k)⁴)⟩_BZ = 60             (this script, exhaustive walk enum.)
    ✓ ⟨Tr(R_4(k)²)⟩_BZ = 24           (algebraic from above two)
    ✓ v_F^Γ = 1/2                     (predictions/srs_dirac_cone_velocities.py)
    ✓ V_BZ_BCC = 16π³                 (BCC primitive: (2π)³/V_primitive = (2π)³/(1/2))

  Closed-form structural identity (CORRECTED 2026-04-28 PM):
    G_sub_form = 1/(16π³) ≈ 0.002016  (this script, with proper BCC V_BZ)
    Earlier 1/(8π³) used simple-cubic V_BZ = 8π³ (wrong for BCC).

  Remaining theorem-grade gap:
    Why this specific combination of Bloch invariants equals G_sub
    (rather than fitting it from Sakharov). Closure requires the
    operator-level R_sub → geometric R^{ab}(x) bridge, scoped at
    an internal working note Sessions 1-4.
    Multi-session research-level item; not closed by the present script.

  This script's contribution:
    Promotes ⟨Tr(H⁴)⟩_BZ = 60 (and hence ⟨Tr(R_4²)⟩_BZ = 24) from
    "claimed via decomposition argument" to "exhaustively verified by
    walk enumeration with explicit displacement filtering". The walk-type
    decomposition (12 bounces + 24 + 24 + 0 = 60) is itself a structural
    finding documenting srs's specific 4-cycle suppression.
""")


def main():
    section("G_sub Bloch invariants — theorem-grade walk enumeration")
    print()
    print("  Setup: srs primitive cell (Row 6) with directed bond list.")
    print(f"    n_atoms     = {N_ATOMS}  (Wyckoff 8a, Row 16)")
    print(f"    n_undirected_edges = {len(CELL_EDGES)}  (Row 7: |E| = 6)")
    print(f"    n_directed_bonds   = {len(DIRECTED_BONDS)}  (= 2|E| = 12)")

    trH2 = step_A_length2()
    trH4 = step_B_length4()
    trR4_sq = step_C_R4_norm(trH2, trH4)
    step_D_structural_form(trH2, trH4, trR4_sq)
    step_E_status()


if __name__ == "__main__":
    main()
