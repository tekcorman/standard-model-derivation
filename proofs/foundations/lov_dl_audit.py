#!/usr/bin/env python3
"""
lov vs {srs, srs-z} structural-DL audit (R-9 closure-path probe — extension).

Follow-on to `srs_vs_srs_z_dl_audit.py`. The 2026-05-01 candidate sweep
(`rcsr_candidate_sweep.py`) found that lov is a SECOND bipartite-primitive
substrate alongside srs-z (γ_7^A → −χ̃ exact, ‖{χ̃, B(k)}‖ = 0 at all 5 k).
This audit answers: under the same M2a structural-DL accounting that gave
ΔDL(srs-z − srs) = +3.25 bits (Level 2), what is ΔDL(lov − srs) and
ΔDL(lov − srs-z)?

The point is to determine whether lov enters the Boltzmann-weighted ensemble
at a comparable level to srs-z, or whether its larger Wyckoff multiplicity
(24f vs srs's 8a) suppresses it more strongly.

LOV STRUCTURAL DATA (from RCSR /data/3dall.txt and probe results)
-----------------------------------------------------------------
  Space group:      I4(1)32 (#214) — SAME as srs
  Wyckoff:          24f, site symmetry .2., 1 free parameter
                    (vs srs at 8a .32, 0 free params; srs-z at 8c .3., 1 free param)
  Conventional:     |V|=24, |E|=36 (E1 mult 12 + Eq mult 24)
  Primitive:        |V|=12, |E|=18 (after I-centering quotient)
  V+E-transitive:   YES (1 vertex orbit + 1 main edge orbit, with Eq aux)
  Bipartite prim:   YES (|A|=|B|=6) — verified by `rcsr_candidate_sweep.py`
  γ_7^A → −χ̃:      EXACT
  ‖{χ̃, B(k)}‖:     0 at all 5 k-points {Γ, R, X, M, mid}
  k* = 3:           YES (Pati-Salam compatible)

NOTE on existing dl_comparison.py I4_132 Wyckoff data
------------------------------------------------------
The existing WYCKOFF_DATA[214] lists W=5 positions (8a, 8b, 12c, 12d, 24e),
but per International Tables Vol. A, I4_132 (#214) actually has W=9
positions (8a, 8b, 12c, 12d, 16e, 24f, 24g, 24h, 48i). The 24f position
that lov occupies is missing from the existing data — a data gap that also
means the existing `dl_srs()` understates W (srs at 8a from W=9, not W=5).
This audit reports both:
  - "as-published" convention: W=5 for I4_132 (matches `srs_vs_srs_z_dl_audit.py`
    so our number is comparable);
  - "corrected" convention: W=9 for I4_132 (more accurate; affects srs and lov
    symmetrically).
The `dl_comparison.py` data gap should eventually be patched, but that's
out of scope for this single-substrate audit.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from math import log2

from dl_comparison import dl_integer, dl_choice
from srs_vs_srs_z_dl_audit import (
    dl_srs_z,
    primitive_cell_atom_count as primitive_cell_atom_count_srs_srs_z,
    directed_edge_orbit_count as directed_edge_orbit_count_srs_srs_z,
)
from dl_comparison import dl_srs


# =============================================================================
# I4_132 (#214) WYCKOFF DATA — corrected
# =============================================================================
# Source: International Tables for Crystallography Vol. A, space group #214,
# cross-referenced against Bilbao Cryst Server.
#
# I4_132 has W=9 Wyckoff positions:
#   8a   .32   (1/8, 1/8, 1/8)         0 free params  ← srs, srs-c27 here
#   8b   .32   (7/8, 7/8, 7/8)         0 free params
#   12c  .2.   (1/8, 0, 1/4) etc.      0 free params
#   12d  .2.   (5/8, 0, 1/4) etc.      0 free params
#   16e  .3.   (x, x, x)               1 free param
#   24f  .1.   (-x, x+1/4, 1/8)        1 free param   ← lov here
#   24g  ..2   (x, 0, 1/4)             1 free param   ← lou, okw here
#   24h  .1.   (1/8, y, -y+1/4)        1 free param
#   48i  1     (x, y, z)               3 free params
#
# Existing dl_comparison.py WYCKOFF_DATA[214] only lists 5 positions —
# missing 16e, 24f, 24g, 24h, 48i. Existing data is INCOMPLETE.

I4132_WYCKOFF_CORRECTED_W = 9

P4132_WYCKOFF_W = 5  # Confirmed correct: P4_132 has 5 positions
                     # (4a, 4b, 8c, 12d, 24e).


# =============================================================================
# DL(lov) under same accounting as dl_srs(), dl_srs_z()
# =============================================================================

def dl_lov(W_I4132=I4132_WYCKOFF_CORRECTED_W):
    """lov net: I4_132 (#214), Wyckoff 24f.

    Same encoding pattern as dl_srs() and dl_srs_z(): space group + n_orbits +
    Wyckoff + coordinates + edges + chirality.

    24f has 1 free parameter (.2. site symmetry). For an edge-transitive net,
    the topology determines whether the free parameter has a constraining
    value (if so, coordinates=0; otherwise coordinates would be nonzero).
    Per RCSR data, lov's bond orbits include both E1 (mult 12) and Eq aux
    (mult 24) — TWO directed-edge orbits at the conventional cell level.

    For comparability with dl_srs_z(), we use coordinates=0 (assume topology
    determines the param within the stable range). The ADDITIONAL edge orbit
    is captured by the Eq-aux contribution to edges count: undirected edges
    = 1 main + 1 aux = 2 orbits → log2(2) = 1 bit beyond srs/srs-z's
    edges=0.
    """
    bits = {}
    bits['space_group'] = dl_choice(230)               # 7.85 (same)
    bits['n_orbits']    = dl_integer(1)                # 1.00 (1 vertex orbit)
    bits['wyckoff']     = dl_choice(W_I4132)           # log2(W_I4132) bits
    bits['coordinates'] = 0.0                          # assume topology-determined
    bits['edges']       = 1.0                          # 2 edge orbits (E1 + Eq aux) → log2(2) = 1
    bits['chirality']   = 1.0                          # chiral (I4_132 is Sohncke)
    return sum(bits.values()), bits


def dl_srs_corrected(W_I4132=I4132_WYCKOFF_CORRECTED_W):
    """dl_srs() with corrected I4_132 W=9 (instead of W=5 in the existing
    dl_comparison.py data). Same Wyckoff (8a) + chirality + edges as
    `dl_srs()`, just `wyckoff = dl_choice(9)` instead of `dl_choice(5)`.
    """
    bits = {}
    bits['space_group'] = dl_choice(230)
    bits['n_orbits']    = dl_integer(1)
    bits['wyckoff']     = dl_choice(W_I4132)
    bits['coordinates'] = 0.0
    bits['edges']       = 0.0                          # arc-transitive → 1 edge orbit
    bits['chirality']   = 1.0
    return sum(bits.values()), bits


# =============================================================================
# M2a refinements (α) primitive-cell atom count, (β) directed-edge orbits
# =============================================================================

def primitive_cell_atom_count(name):
    """Same convention as srs_vs_srs_z_dl_audit.py + lov."""
    if name == 'srs':
        return 4    # I4_132 + 8a → 8/2 = 4
    elif name == 'srs-z':
        return 8    # P4_132 + 8c → 8/1 = 8
    elif name == 'lov':
        return 12   # I4_132 + 24f → 24/2 = 12
    raise ValueError(name)


def directed_edge_orbit_count(name):
    """Number of automorphism orbits of DIRECTED edges (arcs).

    srs:   arc-transitive (Sunada-unique strongly isotropic) → 1 directed-edge orbit
    srs-z: V+E-transitive but NOT arc-transitive → 2 directed-edge orbits
           (1 undirected × 2 orientations not swapped by stab(uv))
    lov:   V+E-transitive at the conventional level (1 vertex orbit, 1 main
           edge orbit) but with auxiliary 'Eq' orbit on the directed-edge
           level. The conventional structure has 2 UNDIRECTED edge orbits
           (E1 + Eq), each splitting into 2 directed orbits if not arc-transitive.
           Conservative count: 2 undirected × 2 = 4 directed-edge orbits.
           (If lov turns out to be arc-transitive on each undirected orbit,
           this becomes 2; but lov is NOT in the Sunada strongly-isotropic
           class — that is uniquely srs.)
    """
    if name == 'srs':
        return 1
    elif name == 'srs-z':
        return 2
    elif name == 'lov':
        return 4    # 2 undirected orbits × 2 (1/2-arc-transitive within each)
    raise ValueError(name)


def refinement_alpha(name):
    return dl_integer(primitive_cell_atom_count(name))


def refinement_beta(name):
    return dl_integer(directed_edge_orbit_count(name))


# =============================================================================
# AUDIT
# =============================================================================

def main():
    print("=" * 84)
    print("lov vs {srs, srs-z} structural-DL audit — R-9 closure-path probe (extension)")
    print("=" * 84)

    # ---- Convention A: as-published (matches existing dl_srs() with W=5 for I4_132)
    dl_srs_A,   srs_A_bits   = dl_srs()
    dl_srs_z_A, srs_z_A_bits = dl_srs_z()
    dl_lov_A,   lov_A_bits   = dl_lov(W_I4132=5)  # match existing dl_srs() under-count

    # ---- Convention B: corrected (W=9 for I4_132, both srs and lov)
    dl_srs_B,   srs_B_bits   = dl_srs_corrected(W_I4132=9)
    dl_srs_z_B, srs_z_B_bits = dl_srs_z()         # P4_132 W=5 unchanged
    dl_lov_B,   lov_B_bits   = dl_lov(W_I4132=9)

    print("\n" + "-" * 84)
    print("LEVEL 0 — DL component breakdown (existing dl_comparison.py accounting)")
    print("-" * 84)
    print()
    print("  Convention A (as-published, W=5 for I4_132 per existing dl_comparison.py):")
    print(f"    {'component':<14s} {'srs':>8s} {'srs-z':>8s} {'lov':>8s}")
    keys = ['space_group', 'n_orbits', 'wyckoff', 'coordinates', 'edges', 'chirality']
    for k in keys:
        print(f"    {k:<14s} {srs_A_bits[k]:>8.3f} {srs_z_A_bits[k]:>8.3f} {lov_A_bits[k]:>8.3f}")
    print(f"    {'TOTAL':<14s} {dl_srs_A:>8.3f} {dl_srs_z_A:>8.3f} {dl_lov_A:>8.3f}")
    print(f"\n    ΔDL(srs-z − srs) = {dl_srs_z_A - dl_srs_A:+.3f} bits")
    print(f"    ΔDL(lov   − srs) = {dl_lov_A   - dl_srs_A:+.3f} bits")
    print(f"    ΔDL(lov   − srs-z) = {dl_lov_A - dl_srs_z_A:+.3f} bits")

    print()
    print("  Convention B (corrected, W=9 for I4_132 per ITA Vol. A):")
    print(f"    {'component':<14s} {'srs':>8s} {'srs-z':>8s} {'lov':>8s}")
    for k in keys:
        print(f"    {k:<14s} {srs_B_bits[k]:>8.3f} {srs_z_B_bits[k]:>8.3f} {lov_B_bits[k]:>8.3f}")
    print(f"    {'TOTAL':<14s} {dl_srs_B:>8.3f} {dl_srs_z_B:>8.3f} {dl_lov_B:>8.3f}")
    print(f"\n    ΔDL(srs-z − srs) = {dl_srs_z_B - dl_srs_B:+.3f} bits")
    print(f"    ΔDL(lov   − srs) = {dl_lov_B   - dl_srs_B:+.3f} bits")
    print(f"    ΔDL(lov   − srs-z) = {dl_lov_B - dl_srs_z_B:+.3f} bits")

    # ---- Level 1: + α (primitive-cell atom count)
    a_srs   = refinement_alpha('srs')
    a_srs_z = refinement_alpha('srs-z')
    a_lov   = refinement_alpha('lov')

    dl_srs_A_1   = dl_srs_A   + a_srs
    dl_srs_z_A_1 = dl_srs_z_A + a_srs_z
    dl_lov_A_1   = dl_lov_A   + a_lov
    dl_srs_B_1   = dl_srs_B   + a_srs
    dl_srs_z_B_1 = dl_srs_z_B + a_srs_z
    dl_lov_B_1   = dl_lov_B   + a_lov

    print("\n" + "-" * 84)
    print("LEVEL 1 — + (α) primitive-cell atom count (Rissanen prior L*(N_prim))")
    print("-" * 84)
    print(f"  N_prim:  srs={primitive_cell_atom_count('srs')}, srs-z={primitive_cell_atom_count('srs-z')}, "
          f"lov={primitive_cell_atom_count('lov')}")
    print(f"  L*(N_prim):  srs={a_srs:.3f}, srs-z={a_srs_z:.3f}, lov={a_lov:.3f}")
    print()
    print(f"  Convention A:")
    print(f"    DL: srs={dl_srs_A_1:.3f}, srs-z={dl_srs_z_A_1:.3f}, lov={dl_lov_A_1:.3f}")
    print(f"    ΔDL(srs-z − srs) = {dl_srs_z_A_1 - dl_srs_A_1:+.3f} bits")
    print(f"    ΔDL(lov   − srs) = {dl_lov_A_1   - dl_srs_A_1:+.3f} bits")
    print(f"    ΔDL(lov   − srs-z) = {dl_lov_A_1 - dl_srs_z_A_1:+.3f} bits")
    print(f"  Convention B:")
    print(f"    DL: srs={dl_srs_B_1:.3f}, srs-z={dl_srs_z_B_1:.3f}, lov={dl_lov_B_1:.3f}")
    print(f"    ΔDL(srs-z − srs) = {dl_srs_z_B_1 - dl_srs_B_1:+.3f} bits")
    print(f"    ΔDL(lov   − srs) = {dl_lov_B_1   - dl_srs_B_1:+.3f} bits")
    print(f"    ΔDL(lov   − srs-z) = {dl_lov_B_1 - dl_srs_z_B_1:+.3f} bits")

    # ---- Level 2: + β (directed-edge orbits)
    b_srs   = refinement_beta('srs')
    b_srs_z = refinement_beta('srs-z')
    b_lov   = refinement_beta('lov')

    dl_srs_A_2   = dl_srs_A_1   + b_srs
    dl_srs_z_A_2 = dl_srs_z_A_1 + b_srs_z
    dl_lov_A_2   = dl_lov_A_1   + b_lov
    dl_srs_B_2   = dl_srs_B_1   + b_srs
    dl_srs_z_B_2 = dl_srs_z_B_1 + b_srs_z
    dl_lov_B_2   = dl_lov_B_1   + b_lov

    print("\n" + "-" * 84)
    print("LEVEL 2 — + (β) directed-edge orbit count L*(N_arcs_orbits)")
    print("-" * 84)
    print(f"  arc-orbits:  srs={directed_edge_orbit_count('srs')} (arc-transitive), "
          f"srs-z={directed_edge_orbit_count('srs-z')} (1/2-arc-transitive), "
          f"lov={directed_edge_orbit_count('lov')} (1/2-arc-transitive × 2 undirected orbits)")
    print(f"  L*(arc-orbits):  srs={b_srs:.3f}, srs-z={b_srs_z:.3f}, lov={b_lov:.3f}")
    print()
    print(f"  Convention A:")
    print(f"    DL: srs={dl_srs_A_2:.3f}, srs-z={dl_srs_z_A_2:.3f}, lov={dl_lov_A_2:.3f}")
    print(f"    ΔDL(srs-z − srs) = {dl_srs_z_A_2 - dl_srs_A_2:+.3f} bits")
    print(f"    ΔDL(lov   − srs) = {dl_lov_A_2   - dl_srs_A_2:+.3f} bits")
    print(f"    ΔDL(lov   − srs-z) = {dl_lov_A_2 - dl_srs_z_A_2:+.3f} bits")
    print(f"  Convention B:")
    print(f"    DL: srs={dl_srs_B_2:.3f}, srs-z={dl_srs_z_B_2:.3f}, lov={dl_lov_B_2:.3f}")
    print(f"    ΔDL(srs-z − srs) = {dl_srs_z_B_2 - dl_srs_B_2:+.3f} bits")
    print(f"    ΔDL(lov   − srs) = {dl_lov_B_2   - dl_srs_B_2:+.3f} bits")
    print(f"    ΔDL(lov   − srs-z) = {dl_lov_B_2 - dl_srs_z_B_2:+.3f} bits")

    # ---- Empirical-inverse threshold calibration
    print("\n" + "=" * 84)
    print("EMPIRICAL-INVERSE THRESHOLD CALIBRATION (M2b supplementary)")
    print("=" * 84)
    print()
    print("  V_us prediction shifts under Boltzmann-weighted ensemble {srs + lov} or")
    print("  {srs + srs-z} or {srs + srs-z + lov}. V_us = k² / (g · N_prim) =")
    print(f"    V_us(srs)   = 9 / (10 · 4)  = 9/40 = {9/40:.5f}")
    print(f"    V_us(srs-z) = 9 / (10 · 8)  = 9/80 = {9/80:.5f}")
    print(f"    V_us(lov)   = 9 / (10 · 12) = 9/120 = {9/120:.5f}")
    print(f"    PDG V_us    = 0.22501 ± 0.00067 → σ/V_us = {0.00067/0.22501*100:.3f}%")
    print()

    sigma_pdg = 0.00067
    pdg_central = 0.22501
    v_us = {'srs': 9/40, 'srs-z': 9/80, 'lov': 9/120}
    delta_2_lov_srs   = dl_lov_B_2 - dl_srs_B_2
    delta_2_srs_z_srs = dl_srs_z_B_2 - dl_srs_B_2

    # Two-substrate ensemble {srs, lov}
    w_lov = 2 ** (-delta_2_lov_srs)
    v_us_mix_srs_lov = (v_us['srs'] + w_lov * v_us['lov']) / (1.0 + w_lov)
    shift_srs_lov = (v_us['srs'] - v_us_mix_srs_lov) / sigma_pdg

    # Two-substrate ensemble {srs, srs-z}
    w_srs_z = 2 ** (-delta_2_srs_z_srs)
    v_us_mix_srs_srs_z = (v_us['srs'] + w_srs_z * v_us['srs-z']) / (1.0 + w_srs_z)
    shift_srs_srs_z = (v_us['srs'] - v_us_mix_srs_srs_z) / sigma_pdg

    # Three-substrate ensemble {srs, srs-z, lov}
    v_us_mix_3 = (v_us['srs'] + w_srs_z * v_us['srs-z'] + w_lov * v_us['lov']) / (1.0 + w_srs_z + w_lov)
    shift_3 = (v_us['srs'] - v_us_mix_3) / sigma_pdg

    print(f"  Convention B Level 2 ΔDL(lov − srs) = {delta_2_lov_srs:+.3f} bits → "
          f"w(lov)/w(srs) = 2^(−ΔDL) = {w_lov:.4e}")
    print(f"  Convention B Level 2 ΔDL(srs-z − srs) = {delta_2_srs_z_srs:+.3f} bits → "
          f"w(srs-z)/w(srs) = 2^(−ΔDL) = {w_srs_z:.4e}")
    print()
    print(f"  Boltzmann-weighted V_us under various ensembles:")
    print(f"    {{srs only}}:                V_us = {v_us['srs']:.6f}, "
          f"PDG shift = {(v_us['srs']-pdg_central)/sigma_pdg:+.2f}σ")
    print(f"    {{srs, srs-z}} (w_z={w_srs_z:.2e}): V_us = {v_us_mix_srs_srs_z:.6f}, "
          f"shift from srs alone = {shift_srs_srs_z:+.2f}σ")
    print(f"    {{srs, lov}}   (w_lov={w_lov:.2e}): V_us = {v_us_mix_srs_lov:.6f}, "
          f"shift from srs alone = {shift_srs_lov:+.2f}σ")
    print(f"    {{srs, srs-z, lov}}:          V_us = {v_us_mix_3:.6f}, "
          f"shift from srs alone = {shift_3:+.2f}σ")
    print()
    print(f"  Threshold ΔDL for sub-3σ V_us match (vs srs's −0.015σ baseline):")
    print(f"    For lov   alone: w · (V_us(srs)−V_us(lov))   = 3σ_PDG → ΔDL > "
          f"{-log2(3*sigma_pdg/(v_us['srs']-v_us['lov'])):.2f} bits")
    print(f"    For srs-z alone: w · (V_us(srs)−V_us(srs-z)) = 3σ_PDG → ΔDL > "
          f"{-log2(3*sigma_pdg/(v_us['srs']-v_us['srs-z'])):.2f} bits")

    # ---- Verdict
    print("\n" + "=" * 84)
    print("VERDICT (M2a structural alone)")
    print("=" * 84)
    print(f"""
  Convention B Level 2 (corrected I4_132 W=9, all M2a refinements):
    ΔDL(srs-z − srs)   = {dl_srs_z_B_2 - dl_srs_B_2:+.2f} bits  (was {dl_srs_z_A_2 - dl_srs_A_2:+.2f} as-published)
    ΔDL(lov   − srs)   = {dl_lov_B_2 - dl_srs_B_2:+.2f} bits
    ΔDL(lov   − srs-z) = {dl_lov_B_2 - dl_srs_z_B_2:+.2f} bits

  Threshold for sub-3σ V_us closure:
    srs-z: {-log2(3*sigma_pdg/(v_us['srs']-v_us['srs-z'])):.2f} bits  → gap {-log2(3*sigma_pdg/(v_us['srs']-v_us['srs-z'])) - (dl_srs_z_B_2 - dl_srs_B_2):+.2f} bits
    lov:   {-log2(3*sigma_pdg/(v_us['srs']-v_us['lov'])):.2f} bits  → gap {-log2(3*sigma_pdg/(v_us['srs']-v_us['lov'])) - (dl_lov_B_2 - dl_srs_B_2):+.2f} bits

  Headline: lov enters the M2a structural-DL story at ~{dl_lov_B_2 - dl_srs_B_2:.1f} bits suppression
  vs srs (Convention B Level 2). This is {'LARGER' if (dl_lov_B_2 - dl_srs_B_2) > (dl_srs_z_B_2 - dl_srs_B_2) else 'SMALLER'} than srs-z's
  +{dl_srs_z_B_2 - dl_srs_B_2:.1f} bits suppression, so lov is {'less' if (dl_lov_B_2 - dl_srs_B_2) > (dl_srs_z_B_2 - dl_srs_B_2) else 'more'} weighty in the
  Boltzmann ensemble than srs-z. {'Both' if max(dl_lov_B_2 - dl_srs_B_2, dl_srs_z_B_2 - dl_srs_B_2) < -log2(3*sigma_pdg/(v_us['srs']-v_us['srs-z'])) else 'Neither'} fall{'s' if max(dl_lov_B_2 - dl_srs_B_2, dl_srs_z_B_2 - dl_srs_B_2) < -log2(3*sigma_pdg/(v_us['srs']-v_us['srs-z'])) else ''} above the sub-3σ
  V_us-match threshold under M2a alone (consistent with the existing srs-z
  finding that M2a is 2.56 bits short for the srs-z sub-3σ case).

  Important caveats:
    1. I4_132 Wyckoff data in dl_comparison.py is INCOMPLETE (lists 5 of 9
       positions); fixing this affects srs and lov symmetrically but inflates
       both DL totals by log2(9/5) = 0.85 bits. Pre-existing bug, out of
       scope here.
    2. lov's directed-edge orbit count (4) is conservative. If lov turns out
       to be arc-transitive within each undirected orbit (i.e., the
       automorphism stabilizing each undirected edge swaps its orientations),
       the count drops to 2 and ΔDL decreases by L*(4)−L*(2) = {dl_integer(4)-dl_integer(2):.2f} bits.
       Verifying this requires explicit P4_132/I4_132 stabilizer analysis on
       lov's arc orbits — deferred.
    3. lov's bipartite primitive (12 atoms, 6+6) confirms γ_7^A → −χ̃ exactly
       per `rcsr_candidate_sweep.py`; the algebraic SUSY-Q structure
       reproduces. Whether the structural-DL-suppressed weight is enough to
       yield observable sub-leading effects in V_us / V_cb / Q_Koide /
       η_B is a quantitative ensemble question (open).
""")


if __name__ == '__main__':
    main()
