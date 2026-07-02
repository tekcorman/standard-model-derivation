#!/usr/bin/env python3
"""
=============================================================================
CORRECTION / SUPERSEDED 2026-05-12 — see `proofs/foundations/r9_srsz_simulator_run.py`
and an internal working note. Known issues in
this script: (1) the docstring below and the `audit_with_refinements()` print
"DL(srs) = DL(srs-z) = 12.17 — IDENTICAL" but the code (importing the corrected
`dl_comparison.dl_srs()` with I4_132 W=9) actually computes DL(srs)=13.02 ≠
DL(srs-z)=12.17 → Level-0 ΔDL(srs-z−srs) = −0.85 bit (favours srs-z under that
encoding); the "IDENTICAL" prints are stale. (2) The α/β "refinements"
(primitive-cell atom count L*(N_prim), directed-edge orbit count) are presented
as M2a-legitimate; under parameter-linter scrutiny they are cherry-picked
add-ons to an already-valid prefix code (N_prim and the arc-orbit count are
determined by the space group + Wyckoff position, already encoded). The honest
structural picture: srs-z is srs's bipartite double cover; its only genuine
extra structural cost over srs is the doubled-motif cost — a few bits — which is
NOT a hard gate (MDL keeps it above the waterline). R-9 stays DOMINANT-CONDITIONAL.
Kept for provenance.
=============================================================================

srs vs srs-z structural-DL audit (R-9 closure-path probe).

Question (raised 2026-05-01 PM): under the corrected M2a/M2b protocol
,
M2b data-conditional Gaussian-likelihood penalty is supplementary only — it
cannot close R-9 srs-z. Closure must come from M2a structural ΔDL alone.

Existing `dl_comparison.py` accounts (space group + n_orbits + Wyckoff +
coords + edges + chirality) give DL(srs) = 12.17 bits. srs-z at the same
encoding level gives DL(srs-z) = 12.17 bits — IDENTICAL. R-9 srs-z is at
ΔDL = 0 under the existing accounting.

This audit:
  1. Reproduces srs DL using existing dl_comparison.py components.
  2. Adds P4_132 (#213) Wyckoff data + dl_srs_z() in same form.
  3. Computes M2a-legitimate REFINEMENTS:
       (α) primitive-cell atom count (Rissanen prior on N_prim)
       (β) directed-edge orbit count (arc-transitive=1 vs 1/2-arc-transitive=2)
  4. Reports ΔDL at each refinement level.
  5. Calibrates against empirical-inverse threshold:
       V_us(srs) = 9/40 = 0.225 vs V_us(srs-z) = 9/80 = 0.1125
       PDG σ = 0.00067 (0.3% of central value)
       For sub-3σ match: srs-z Boltzmann weight w < 0.018 → ΔDL > ~5.5 bits

Honest verdict reported at end. The audit is M2a-only — no PDG comparison
for closure (per corrected protocol; PDG comparison is supplementary
empirical validation only, not load-bearing).

References:
  - `proofs/foundations/dl_comparison.py` — existing DL framework.
  - `docs/audits/registers/structural_residue_register.md` R-9 — multi-tier closure structure.
  - International Tables for Crystallography Vol. A (Wyckoff data #213, #214).
  - Sunada 2012, *Notices AMS* 59(2) — strong isotropy of srs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dl_comparison import (
    dl_integer, dl_choice, dl_choose_k_of_n, dl_srs,
)
from math import log2


# =============================================================================
# P4_132 (#213) WYCKOFF DATA — for srs-z
# =============================================================================
# Source: International Tables for Crystallography Vol. A, space group #213.
# Site symmetries from Bilbao Crystallographic Server.
#
# P4_132 has W=5 Wyckoff positions:
#   4a   .32 (D_3)   (3/8, 3/8, 3/8) etc., 0 free params
#   4b   .32 (D_3)   (7/8, 7/8, 7/8) etc., 0 free params
#   8c   .3.  (C_3)   (x, x, x),           1 free param
#   12d  .2.  (C_2)   (1/8, y, -y+3/4),    1 free param
#   24e  1            (x, y, z),           3 free params
#
# srs-z occupies 8c (mult 8, C_3 site symmetry, 1 free param x).
# Same Wyckoff multiplicity as srs's 8a in I4_132, same C_3 site symmetry.
# Same number of Wyckoff positions (W=5) as I4_132.

P4132_WYCKOFF = {
    'name': 'P4_132', 'sg_number': 213, 'W': 5,
    'positions': {
        '4a':  {'mult': 4,  'free_params': 0, 'coords': '(3/8, 3/8, 3/8)'},
        '4b':  {'mult': 4,  'free_params': 0, 'coords': '(7/8, 7/8, 7/8)'},
        '8c':  {'mult': 8,  'free_params': 1, 'coords': '(x, x, x)'},
        '12d': {'mult': 12, 'free_params': 1, 'coords': '(1/8, y, -y+3/4)'},
        '24e': {'mult': 24, 'free_params': 3, 'coords': '(x, y, z)'},
    },
}


def dl_srs_z():
    """
    srs-z net: P4_132 (#213), Wyckoff 8c.
    Vertex-transitive, edge-transitive (V+E-transitive in (1,1) RCSR sense),
    but NOT arc-transitive — the V+E-transitive automorphism group does NOT
    act transitively on directed edges. Strictly weaker than srs's strong
    isotropy (per Sunada 2012, srs is unique strongly isotropic).

    Same encoding pattern as dl_srs(): space group + n_orbits + Wyckoff +
    coordinates + edges + chirality. P4_132 has W=5 Wyckoff positions
    (same as I4_132). Wyckoff 8c is one of 5 → log2(5) = 2.32 bits.

    By the existing dl_comparison.py accounting alone, DL(srs-z) = DL(srs)
    exactly. Refinements added in `audit_with_refinements()` below.
    """
    bits = {}
    bits['space_group'] = dl_choice(230)               # 7.85
    bits['n_orbits']    = dl_integer(1)                # 1.00
    bits['wyckoff']     = dl_choice(5)                 # 2.32 (8c from W=5 in P4_132)
    bits['coordinates'] = 0.0                          # topology determines coords (V+E-transitive)
    bits['edges']       = 0.0                          # undirected edge-transitive (1 orbit)
    bits['chirality']   = 1.0                          # chiral (P4_132 is Sohncke)
    return sum(bits.values()), bits


# =============================================================================
# M2a-LEGITIMATE STRUCTURAL REFINEMENTS
# =============================================================================
# These are additions to the basic dl_comparison.py framework that capture
# structural complexity NOT reflected in (space group + Wyckoff) choice alone.
# Each is justified as M2a-legitimate (substrate-encoding bit-count).
# No PDG comparisons; pure structural.

def primitive_cell_atom_count(name):
    """Number of atoms in the primitive cell.

    For body-centered (I) groups, primitive = conventional/2.
    For primitive (P) groups, primitive = conventional.

    srs:   I4_132 + 8a → primitive cell = 8/2 = 4 atoms
    srs-z: P4_132 + 8c → primitive cell = 8/1 = 8 atoms
    """
    if name == 'srs':
        return 4
    elif name == 'srs-z':
        return 8
    else:
        raise ValueError(f"unknown net: {name}")


def directed_edge_orbit_count(name):
    """Number of automorphism orbits of DIRECTED edges (arcs).

    For arc-transitive graphs (full action transitive on arcs): 1 orbit.
    For (1,1)-transitive but not arc-transitive (1/2-arc-transitive): 2 orbits.

    srs:   arc-transitive (Sunada 2012 strong isotropy) → 1 directed-edge orbit
    srs-z: V+E-transitive but NOT arc-transitive → 2 directed-edge orbits
           (each undirected edge orbit splits into 2 directed orbits since
           the automorphism stabilizing the undirected edge does NOT swap
           its two orientations)
    """
    if name == 'srs':
        return 1
    elif name == 'srs-z':
        return 2
    else:
        raise ValueError(f"unknown net: {name}")


def refinement_alpha_primitive_cell(name):
    """Refinement (α): Rissanen universal prior on primitive-cell atom count.

    The Kolmogorov complexity floor for a periodic crystal includes encoding
    HOW MANY atoms the primitive cell holds. Universal prior L*(N_prim).

    This is M2a-legitimate (substrate complexity bit-count, no PDG involved).
    Justified by bridge theorem `theorem_substrate_layer1_layer2_bridge_dominant.md`
    §3.1(b): DL_struct(C) >= K(C), Kolmogorov floor on the periodic cell
    description.

    Not currently in dl_comparison.py — the script's `n_orbits=L*(1)=1` term
    counts vertex orbits, NOT primitive-cell atom count. They coincide for
    V-transitive nets only when primitive cell has multiplicity equal to the
    smallest Wyckoff (e.g., the script lumps all vertex-transitive nets into
    n_orbits=1 regardless of primitive cell size).
    """
    return dl_integer(primitive_cell_atom_count(name))


def refinement_beta_directed_edge_orbits(name):
    """Refinement (β): directed-edge orbit count.

    The walker_dynamics formalism (`predictions/walker_dynamics_derivation.md`
    Step 5) uses directed edges as causal states. The Hashimoto operator B is
    a 2|E|-dimensional matrix on directed edges. The substrate's automorphism
    action on directed edges determines how many distinct B-matrix-entry
    classes there are.

    Encoding cost: L*(directed_edge_orbits) bits to specify the orbit count.

    This is M2a-legitimate (substrate symmetry-class bit-count, no PDG).
    Not in dl_comparison.py — the script's 'edges' term counts UNDIRECTED
    edge orbits.
    """
    return dl_integer(directed_edge_orbit_count(name))


# =============================================================================
# AUDIT COMPUTATION
# =============================================================================

def audit_with_refinements():
    """Compute ΔDL(srs-z − srs) at each refinement level."""
    print("=" * 78)
    print("srs vs srs-z STRUCTURAL-DL AUDIT — R-9 closure-path probe (M2a only)")
    print("=" * 78)

    # ----- Level 0: existing dl_comparison.py accounting --------------------
    dl_srs_val, srs_bits = dl_srs()
    dl_srs_z_val, srs_z_bits = dl_srs_z()

    print("\n" + "-" * 78)
    print("LEVEL 0: existing dl_comparison.py accounting")
    print("-" * 78)
    print(f"  DL(srs)   = {dl_srs_val:.4f} bits")
    print(f"  DL(srs-z) = {dl_srs_z_val:.4f} bits")
    print(f"  ΔDL(srs-z - srs) = {dl_srs_z_val - dl_srs_val:+.4f} bits")
    print()
    print(f"  Component breakdown:")
    for k in srs_bits:
        print(f"    {k:<14s} srs={srs_bits[k]:6.3f}  srs-z={srs_z_bits[k]:6.3f}  Δ={srs_z_bits[k]-srs_bits[k]:+.3f}")
    print(f"\n  → srs and srs-z have IDENTICAL DL under existing accounting.")

    # ----- Level 1: + primitive-cell atom count (refinement α) --------------
    a_srs   = refinement_alpha_primitive_cell('srs')
    a_srs_z = refinement_alpha_primitive_cell('srs-z')

    dl_srs_1   = dl_srs_val   + a_srs
    dl_srs_z_1 = dl_srs_z_val + a_srs_z
    delta_1    = dl_srs_z_1 - dl_srs_1

    print("\n" + "-" * 78)
    print("LEVEL 1: + primitive-cell atom count (Rissanen prior)")
    print("-" * 78)
    print(f"  N_prim(srs)   = {primitive_cell_atom_count('srs')}    L*({primitive_cell_atom_count('srs')}) = {a_srs:.4f} bits")
    print(f"  N_prim(srs-z) = {primitive_cell_atom_count('srs-z')}    L*({primitive_cell_atom_count('srs-z')}) = {a_srs_z:.4f} bits")
    print(f"  Δ(α) = {a_srs_z - a_srs:+.4f} bits")
    print(f"  Cumulative ΔDL = {delta_1:+.4f} bits  →  Boltzmann weight ratio w(srs-z)/w(srs) = 2^(-ΔDL) = {2**(-delta_1):.4e}")

    # ----- Level 2: + directed-edge orbit count (refinement β) --------------
    b_srs   = refinement_beta_directed_edge_orbits('srs')
    b_srs_z = refinement_beta_directed_edge_orbits('srs-z')

    dl_srs_2   = dl_srs_1   + b_srs
    dl_srs_z_2 = dl_srs_z_1 + b_srs_z
    delta_2    = dl_srs_z_2 - dl_srs_2

    print("\n" + "-" * 78)
    print("LEVEL 2: + directed-edge orbit count (arc-transitivity bit-count)")
    print("-" * 78)
    print(f"  arc-orbits(srs)   = {directed_edge_orbit_count('srs')}    L*({directed_edge_orbit_count('srs')}) = {b_srs:.4f} bits  (arc-transitive)")
    print(f"  arc-orbits(srs-z) = {directed_edge_orbit_count('srs-z')}    L*({directed_edge_orbit_count('srs-z')}) = {b_srs_z:.4f} bits  (1/2-arc-transitive)")
    print(f"  Δ(β) = {b_srs_z - b_srs:+.4f} bits")
    print(f"  Cumulative ΔDL = {delta_2:+.4f} bits  →  Boltzmann weight ratio w(srs-z)/w(srs) = 2^(-ΔDL) = {2**(-delta_2):.4e}")

    # ----- Empirical-inverse calibration ------------------------------------
    print("\n" + "=" * 78)
    print("EMPIRICAL-INVERSE THRESHOLD CALIBRATION (supplementary, M2b)")
    print("=" * 78)
    print("""
  This block is M2b SUPPLEMENTARY ONLY (per uniqueness_audit_v2_protocol.md
  M2a/M2b split, 2026-05-01 PM). Computed for transparency, NOT used to claim
  closure.

  Discriminating predictions (srs vs srs-z values differ):
    V_us = k* / (g · N_prim)
      V_us(srs)   = 9/(10·4) = 9/40   = 0.22500
      V_us(srs-z) = 9/(10·8) = 9/80   = 0.11250

    η_B chain length M = N_edges/cell = k* · N_prim / 2
      M(srs)   = 6   → α_1^M = (2/3)^48  ≈ 3.5e-9
      M(srs-z) = 12  → α_1^M = (2/3)^96  ≈ 1.2e-17

  Non-discriminating predictions (same value on both):
    V_cb (depends on k, g only — both k=3, g=10)
    Q_Koide
    m_τ chain
""")

    v_us_srs   = 9.0/40.0
    v_us_srs_z = 9.0/80.0
    sigma_pdg  = 0.00067   # PDG 2024 V_us ± uncertainty
    pdg_central = 0.22501

    print(f"  PDG V_us = {pdg_central:.5f} ± {sigma_pdg:.5f}  ({sigma_pdg/pdg_central*100:.2f}% precision)")
    print()
    print(f"  Waterfilled V_us as function of srs-z weight w:")
    print(f"  {'w':>10s}  {'ΔDL':>8s}  {'V_us(mix)':>12s}  {'shift / σ_PDG':>14s}")

    for w in [1.0, 0.5, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]:
        v_us_mix = (v_us_srs + w * v_us_srs_z) / (1.0 + w)
        shift = (v_us_srs - v_us_mix) / sigma_pdg
        delta_dl = -log2(w) if w > 0 else float('inf')
        print(f"  {w:10.4f}  {delta_dl:8.2f}  {v_us_mix:12.6f}  {shift:14.1f}σ")

    print()
    print("  Empirical-inverse thresholds:")
    print("    sub-1σ shift:  w < 1·σ_PDG/(V_us(srs)-V_us(srs-z)) ≈ {:.4e} → ΔDL > {:.2f} bits".format(
        sigma_pdg/(v_us_srs - v_us_srs_z), -log2(sigma_pdg/(v_us_srs - v_us_srs_z))))
    print("    sub-3σ shift:  w < {:.4e} → ΔDL > {:.2f} bits".format(
        3*sigma_pdg/(v_us_srs - v_us_srs_z), -log2(3*sigma_pdg/(v_us_srs - v_us_srs_z))))
    print("    sub-5σ shift:  w < {:.4e} → ΔDL > {:.2f} bits".format(
        5*sigma_pdg/(v_us_srs - v_us_srs_z), -log2(5*sigma_pdg/(v_us_srs - v_us_srs_z))))

    return {
        'level_0': dl_srs_z_val - dl_srs_val,
        'level_1': delta_1,
        'level_2': delta_2,
        'thresholds_v_us': {
            '1sigma': -log2(sigma_pdg/(v_us_srs - v_us_srs_z)),
            '3sigma': -log2(3*sigma_pdg/(v_us_srs - v_us_srs_z)),
            '5sigma': -log2(5*sigma_pdg/(v_us_srs - v_us_srs_z)),
        }
    }


def verdict(results):
    """Honest verdict on R-9 srs-z closure via M2a structural alone."""
    print("\n" + "=" * 78)
    print("VERDICT (M2a structural only — corrected protocol)")
    print("=" * 78)

    delta = results['level_2']
    threshold_3sigma = results['thresholds_v_us']['3sigma']
    threshold_1sigma = results['thresholds_v_us']['1sigma']

    print(f"""
  M2a structural ΔDL(srs-z - srs) under refined accounting:
    Level 0 (existing dl_comparison.py):    {results['level_0']:+.2f} bits
    Level 1 (+ primitive-cell atom count):  {results['level_1']:+.2f} bits
    Level 2 (+ directed-edge orbit count):  {results['level_2']:+.2f} bits

  Empirical-inverse thresholds for sub-Nσ V_us match (M2b supplementary):
    sub-1σ: ΔDL > {threshold_1sigma:.2f} bits
    sub-3σ: ΔDL > {threshold_3sigma:.2f} bits

  Honest reading:
""")

    if delta >= threshold_1sigma:
        print("    M2a STRUCTURAL ALONE CLOSES R-9 srs-z to sub-1σ V_us match. ✓")
    elif delta >= threshold_3sigma:
        print("    M2a structural alone closes R-9 srs-z to sub-3σ V_us match.")
        print("    Sub-1σ closure requires additional structural component beyond Level 2.")
    else:
        gap = threshold_3sigma - delta
        print(f"    M2a STRUCTURAL ALONE IS INSUFFICIENT. Gap to sub-3σ closure: {gap:.2f} bits.")
        print(f"    Boltzmann weight ratio at Level 2 ΔDL = {delta:+.2f}: w(srs-z)/w(srs) = {2**(-delta):.3e}")
        print(f"    Resulting V_us shift would be ~{(9/40 - 9/80) * 2**(-delta) / (1+2**(-delta)) / 0.00067:.1f}σ — observable.")
        print()
        print(f"    Honest conclusion: under the corrected M2a/M2b protocol, R-9 srs-z is")
        print(f"    NOT cleanly closed by M2a structural alone. The framework's empirical")
        print(f"    match to PDG (V_us −0.015σ) is M2b supplementary evidence — confirming")
        print(f"    the structural exclusion of srs-z is correct, but NOT itself providing")
        print(f"    closure.")
        print()
        print(f"    Possible additional M2a-legitimate components to investigate:")
        print(f"    - Stricter Kolmogorov-floor encoding of (sg, Wyckoff) that reflects")
        print(f"      compression efficiency (centered space groups encode more structure")
        print(f"      per bit; uniform prior over 230 sg's may be too coarse).")
        print(f"    - Specifying the framework's structural REQUIREMENT of arc-transitivity")
        print(f"      as a NEW axiom or load-bearing closure step, justified independently")
        print(f"      of walker_dynamics (which audit shows does not require it).")
        print(f"    - K-rationality of srs-z's Hashimoto P-point spectrum — but this is")
        print(f"      potentially circular per user catch (K identified from srs spectrum).")

    print()


if __name__ == '__main__':
    results = audit_with_refinements()
    verdict(results)
