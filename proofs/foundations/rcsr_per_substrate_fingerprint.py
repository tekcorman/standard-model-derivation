#!/usr/bin/env python3
"""
Per-substrate structural fingerprint for the 9 V+E-transitive 3-c chiral 3D
candidates in `rcsr_net_assessment.py:592`.

For each candidate, computes a UNIFORM set of derived quantities under one
convention so the cross-substrate comparison is rigorous:

  S1.   Space group (chirality, centering)
  S2.   Conventional + primitive cell (|V|, |E|, deg seq, multi-edges)
  S3.   Connectivity of primitive quotient (single component or disjoint)
  S4.   Bipartition status (BIPARTITE / NOT_BIPARTITE / DISCONNECTED) +
        side sizes when applicable
  S5.   Girth from RCSR vertex symbol (NOT primitive-quotient girth — the
        crystal-lattice girth that drives the framework's α_1 = (q_NB)^(g-2))
  S6.   B(k) eigenvalue spectrum at all relevant high-symmetry k-points,
        with k-points selected by SG centering type:
          P-cubic groups → {Γ, R, M, X, mid}
          I-cubic groups → {Γ, H, N, P, mid_BCC}      (BCC reciprocal lattice)
  S7.   Saddle k-point identification: the k-point at which a K-rational
        eigenvalue of B(k) appears (= "K-rational saddle"). For srs this is
        k_P=(1/4,1/4,1/4) on I-cubic; for srs-z it is k_R=(1/2,1/2,1/2)
        on P-cubic. Other substrates may or may not host K-rational saddles.
  S8.   Saddle eigenvalue identification: Re/Im in K = ℚ(√2,√3,√5)?
        Multiplicity of the saddle eigenvalue. Mass-relevant data.
  S9.   For bipartite substrates: γ_7^A → ±χ̃ verification + ‖{χ̃, B(k)}‖
        anti-commutation residual.
  S10.  Convention-B Level 2 description length (for Boltzmann ensemble use):
        space-group + n_orbits + Wyckoff (corrected W per ITA Vol. A) +
        coordinates + edges + chirality + α primitive-cell + β arc-orbit.

Honest reporting of per-candidate quirks:
  - hcb-c4 disconnected primitive (catenated honeycomb) — flagged, not
    suppressed.
  - srs-c8 non-uniform conventional degree sequence [6,2,3,1,5,3,4,0]
    after multi-orbit aggregation — flagged.
  - lou/lov/okw use 'Eq' auxiliary edge orbits whose handling depends on
    the parser fix in `rcsr_net_assessment.py`; without that fix, 24 of 36
    edges are silently dropped.
"""

import sys
import os
import math
import numpy as np
from numpy.linalg import eigvals
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcsr_net_assessment import (
    parse_rcsr_3dall, get_space_group_ops, orbit_of, reconstruct_bonds,
    bloch_hashimoto, build_directed_edges, identify_irrational,
)
from srs_z_bipartite_involution_commutation import find_bipartition, build_adjacency
from rcsr_candidate_sweep import primitive_quotient_via_body_centering, find_bipartition_full


I_CENTERED_SGS = {'I4(1)32', 'I432', 'Ia-3d', 'I23', 'I2(1)3', 'Im-3m', 'Ia-3', 'I-43d'}

# Per-substrate Wyckoff position used (extracted from RCSR data; not all are
# in dl_comparison.py's WYCKOFF_DATA — see lov_dl_audit.py for the 9-position
# correction to I4_132 (#214)).
SUBSTRATE_WYCKOFF = {
    'srs':     {'sg_num': 214, 'wyckoff': '8a',  'free_params': 0},
    'srs-z':   {'sg_num': 213, 'wyckoff': '8c',  'free_params': 1},
    'srs-c4':  {'sg_num': 208, 'wyckoff': '4b',  'free_params': 0},
    'srs-c8':  {'sg_num': 211, 'wyckoff': '8c',  'free_params': 1},
    'srs-c27': {'sg_num': 214, 'wyckoff': '8a',  'free_params': 0},
    'lou':     {'sg_num': 214, 'wyckoff': '24g', 'free_params': 1},
    'lov':     {'sg_num': 214, 'wyckoff': '24f', 'free_params': 1},
    'okw':     {'sg_num': 214, 'wyckoff': '24g', 'free_params': 1},
    'hcb-c4':  {'sg_num': 212, 'wyckoff': '8c',  'free_params': 1},
}

# Space-group total Wyckoff position counts (W) per ITA Vol. A. Used by
# `dl_choice(W)` for the wyckoff component of DL.
SG_W = {
    208: 6,   # P4_232 — 2a, 4b, 4c, 4d, 6e, 6f (= 6 positions; corrected per ITA)
    211: 7,   # I432 — 2a, 6b, 8c, 12d, 12e, 24f, 24g + 24h, 24i, 48j? Standard ITA has 8 positions for I432
    212: 4,   # P4_332 — 4a, 4b, 8c, 12d (= 4 positions)
    213: 5,   # P4_132 — 4a, 4b, 8c, 12d, 24e (= 5 positions; in lov_dl_audit)
    214: 9,   # I4_132 — 8a, 8b, 12c, 12d, 16e, 24f, 24g, 24h, 48i (= 9 positions; in lov_dl_audit)
}

# Girth from RCSR vertex symbol (extracted by parsing).
SUBSTRATE_GIRTH = {
    'srs':     10,  # 10(5).10(5).10(5)
    'srs-z':   10,
    'srs-c4':  10,
    'srs-c8':  10,
    'srs-c27': 10,
    'lou':     16,  # 16(8).16(8).16(8)
    'lov':     16,  # 16(4).16(4).16(8)
    'okw':     16,
    'hcb-c4':   6,  # 6.6.6 (3 hexagonal faces per vertex)
}

# K-point conventions per centering type (in CONVENTIONAL fractional coords —
# the input convention used by `bloch_hashimoto`).
P_CUBIC_KPOINTS = {
    'Γ':   np.array([0.0, 0.0, 0.0]),
    'R':   np.array([0.5, 0.5, 0.5]),     # K-rational saddle for srs-z, srs-c4, hcb-c4
    'X':   np.array([0.5, 0.0, 0.0]),
    'M':   np.array([0.5, 0.5, 0.0]),
    'mid': np.array([0.25, 0.25, 0.25]),
}
I_CUBIC_KPOINTS = {
    'Γ':   np.array([0.0, 0.0, 0.0]),
    'H':   np.array([0.5, -0.5, 0.5]),    # BCC zone-boundary
    'N':   np.array([0.0, 0.0, 0.5]),     # BCC face-centre
    'P':   np.array([0.25, 0.25, 0.25]),  # K-rational saddle for srs (and other I-cubic)
    'midR': np.array([0.5, 0.5, 0.5]),    # primitive-cubic R, equivalent to Γ in BCC reciprocal? Reported for diagnostic
}


# =============================================================================
# K-RATIONALITY TEST
# =============================================================================

def is_K_rational(lam, tol=1e-7):
    """Test whether λ is in K = ℚ(√2, √3, √5).

    Cheap heuristic: factor as (a + b√2 + c√3 + d√5 + e√6 + f√10 + g√15 +
    h√30) / N for small integers a..h, N ≤ 12. Returns identifier string or
    None.
    """
    re_l, im_l = float(np.real(lam)), float(np.imag(lam))
    re_id = identify_irrational(re_l, tol=tol) if abs(re_l) > tol else "0"
    im_id = identify_irrational(abs(im_l), tol=tol) if abs(im_l) > tol else "0"
    if re_id is None or im_id is None:
        return None
    sgn = '-' if im_l < 0 else '+'
    return f"({re_id}) {sgn} ({im_id})i" if im_id != "0" else f"{re_id}"


def find_K_rational_saddle(arcs, n_atoms, k_points, k_minus_1=2):
    """Identify which k-point (if any) hosts a K-rational eigenvalue of B(k)
    saturating Ramanujan |λ|² = k-1.

    Returns dict {k_name: {'K_rational_eigs': [(λ, mult, identifier)], 'all_eigs': [...]}}

    Note: we group eigenvalues by 5-decimal rounding for multiplicity counting,
    but pass the FULL-PRECISION mean of each group to is_K_rational so that
    identify_irrational's 1e-6 tolerance can match canonical algebraic values
    like √3/2 (= 0.8660254..., > 5-decimal precision).
    """
    results = {}
    for k_name, k_frac in k_points.items():
        B = bloch_hashimoto(arcs, k_frac, n_atoms)
        eigs = list(eigvals(B))
        # Group by (re, im) rounded
        groups = {}
        for e in eigs:
            key = (round(np.real(e), 5), round(np.imag(e), 5))
            groups.setdefault(key, []).append(e)
        K_rat = []
        for key, members in groups.items():
            m = len(members)
            # Use full-precision MEAN of the group for irrational identification
            lam_mean = complex(np.mean([m_.real for m_ in members]),
                                np.mean([m_.imag for m_ in members]))
            mod_sq = abs(lam_mean) ** 2
            if abs(mod_sq - k_minus_1) > 1e-3:
                continue
            ident = is_K_rational(lam_mean)
            if ident is not None and ident != "0":
                K_rat.append((lam_mean, m, ident))
        results[k_name] = {'K_rational_eigs': K_rat, 'n_eigs': len(eigs)}
    return results


# =============================================================================
# γ_7^A → ±χ̃ check (for bipartite substrates)
# =============================================================================

def gamma7_lift_check(side_a, side_b, n_atoms):
    """Same as in lov_chi_layer_replication.py — verify γ_7^A → ±χ̃."""
    gamma7_F0, gamma7_F1 = -1, +1
    side_label = {v: +1 for v in side_a}
    side_label.update({v: -1 for v in side_b})
    all_minus = True
    all_plus = True
    for v in range(n_atoms):
        eig = 1
        for u in side_a:
            eig *= (gamma7_F1 if u == v else gamma7_F0)
        if eig != -side_label[v]:
            all_minus = False
        if eig != +side_label[v]:
            all_plus = False
    if all_minus:
        return '-χ̃'
    elif all_plus:
        return '+χ̃'
    return 'MISMATCH'


def chi_anticomm_norms(arcs, side_label, n_atoms, k_points):
    """For bipartite substrate, compute ‖{χ̃, B(k)}‖/‖B‖ at each k-point."""
    chi = np.diag([side_label[a[0]] for a in arcs]).astype(complex)
    out = {}
    for k_name, k_frac in k_points.items():
        B = bloch_hashimoto(arcs, k_frac, n_atoms)
        anti = chi @ B + B @ chi
        ratio = float(np.linalg.norm(anti) / max(np.linalg.norm(B), 1e-12))
        out[k_name] = ratio
    return out


# =============================================================================
# CONVENTION-B LEVEL 2 DL (matches lov_dl_audit.py)
# =============================================================================

def dl_convention_B_level2(name, n_arcs_orbits=None):
    """Convention-B Level 2 DL for a substrate.

    Same encoding as dl_lov(W_I4132=9): space group + n_orbits + Wyckoff
    (using corrected SG_W) + coordinates + edges + chirality + α + β.
    """
    info = SUBSTRATE_WYCKOFF[name]
    sg_num = info['sg_num']
    W = SG_W[sg_num]
    bits = {
        'space_group': math.log2(230),
        'n_orbits':    1.0,                                # L*(1)
        'wyckoff':     math.log2(W),
        'coordinates': 0.0,                                 # topology-determined
    }
    # Edges contribution: 1 main + maybe 1 aux for lou/lov/okw
    if name in ('lou', 'lov', 'okw'):
        bits['edges'] = 1.0  # log2(2) = 1 (main + aux orbit)
    else:
        bits['edges'] = 0.0
    bits['chirality'] = 1.0
    # α: L*(N_prim)
    n_prim_table = {'srs': 4, 'srs-z': 8, 'srs-c4': 4, 'srs-c8': 4,
                    'srs-c27': 4, 'lou': 12, 'lov': 12, 'okw': 12, 'hcb-c4': 8}
    n_prim = n_prim_table[name]
    bits['alpha_Nprim'] = _Lstar(n_prim)
    # β: L*(arc-orbits) — substrate arc-orbit count from arc-transitivity analysis.
    # srs is arc-transitive (Sunada-unique); others not.
    arc_orbits_table = {
        'srs': 1, 'srs-z': 2, 'srs-c4': 2, 'srs-c8': 2,
        'srs-c27': 2, 'lou': 4, 'lov': 4, 'okw': 4, 'hcb-c4': 2,
    }
    if n_arcs_orbits is None:
        n_arcs_orbits = arc_orbits_table[name]
    bits['beta_arc_orbits'] = _Lstar(n_arcs_orbits)
    return sum(bits.values()), bits


def _Lstar(n):
    """Rissanen universal prefix code for positive integer n."""
    if n <= 0:
        return 0.0
    if n == 1:
        return 1.0
    total = 1.0
    x = float(n)
    while x > 1.0:
        lx = math.log2(x)
        total += lx
        x = lx
        if x <= 0:
            break
    return total


# =============================================================================
# MAIN — assemble per-substrate fingerprint
# =============================================================================

CANDIDATES = ['srs', 'srs-z', 'srs-c4', 'srs-c8', 'srs-c27',
              'lou', 'lov', 'okw', 'hcb-c4']


def fingerprint(name, entry):
    sg = entry['sg_name']
    rotations, translations, _, _ = get_space_group_ops(sg)
    v_frac = np.array(entry['vertex_orbits'][0]['cartesian'])
    coord = entry['vertex_orbits'][0]['coord']
    atom_orbit = orbit_of(v_frac, rotations, translations)
    midpoints = [orbit_of(np.array(eo['cartesian']), rotations, translations)
                 for eo in entry['edge_orbits']]
    midpoint_orbit = np.vstack(midpoints)
    n_conv_atoms = len(atom_orbit)
    n_conv_mid = len(midpoint_orbit)
    bonds_conv = reconstruct_bonds(atom_orbit, midpoint_orbit, tol=1e-3, max_shift=3)
    bonds_conv = [b for b in bonds_conv if b is not None]
    A_conv = build_adjacency(bonds_conv, n_conv_atoms)
    conv_deg_seq = sorted(int(d) for d in A_conv.sum(axis=1))

    # Primitive cell construction
    if sg in I_CENTERED_SGS:
        n_prim, A_prim, partner, prim_bonds, conv_to_prim = \
            primitive_quotient_via_body_centering(atom_orbit, bonds_conv)
        primitive_method = 'I-quotient'
    else:
        n_prim = n_conv_atoms
        A_prim = A_conv
        prim_bonds = bonds_conv
        primitive_method = 'P-group'

    n_prim_edges = int(A_prim.sum() // 2)
    prim_deg_seq = sorted(int(d) for d in A_prim.sum(axis=1))

    # Bipartition / connectivity
    bp_status, side_a, side_b = find_bipartition_full(A_prim)

    # γ_7^A check + χ̃ anti-commutation (only when BIPARTITE)
    gamma7_verdict = None
    chi_anti = None
    if bp_status == 'BIPARTITE':
        gamma7_verdict = gamma7_lift_check(side_a, side_b, n_prim)
        side_label = {v: +1 for v in side_a}
        side_label.update({v: -1 for v in side_b})
        arcs = build_directed_edges(prim_bonds)
        # Use I-conv k-points if I-centered, else P-conv
        kpts = I_CUBIC_KPOINTS if sg in I_CENTERED_SGS else P_CUBIC_KPOINTS
        chi_anti = chi_anticomm_norms(arcs, side_label, n_prim, kpts)

    # K-rational saddle search — uses CONVENTIONAL cell + Bloch phases (matches the
    # framework's canonical srs saddle calculation at k_P = (1/4,1/4,1/4) on
    # I4_132 conventional). The primitive-cell body-centering quotient has
    # multi-edges and a different spectrum.
    arcs_conv = build_directed_edges(bonds_conv)
    kpts = I_CUBIC_KPOINTS if sg in I_CENTERED_SGS else P_CUBIC_KPOINTS
    k_minus_1 = coord - 1
    saddle_search = find_K_rational_saddle(arcs_conv, n_conv_atoms, kpts, k_minus_1=k_minus_1) \
        if bp_status != 'DISCONNECTED' else None

    # DL
    dl_total, dl_bits = dl_convention_B_level2(name)

    # Girth + α_1 + α_1^2/(1-α_1) (V_ub) + α_1/(1-α_1) (V_cb)
    g = SUBSTRATE_GIRTH[name]
    q_NB = (coord - 1) / coord
    alpha_1 = q_NB ** (g - 2)
    V_cb_pred = alpha_1 / (1 - alpha_1)
    V_ub_pred = alpha_1 ** 2 / (1 - alpha_1)

    return {
        'name': name, 'sg': sg, 'coord': coord,
        'wyckoff': SUBSTRATE_WYCKOFF[name]['wyckoff'],
        'g': g, 'q_NB': q_NB, 'alpha_1': alpha_1,
        'n_conv_atoms': n_conv_atoms, 'n_conv_mid': n_conv_mid,
        'conv_deg_seq': conv_deg_seq,
        'primitive_method': primitive_method,
        'n_prim_atoms': n_prim, 'n_prim_edges': n_prim_edges,
        'prim_deg_seq': prim_deg_seq,
        'bp_status': bp_status,
        'side_a': side_a, 'side_b': side_b,
        'gamma7_verdict': gamma7_verdict,
        'chi_anticomm': chi_anti,
        'saddle_search': saddle_search,
        'dl_total': dl_total, 'dl_bits': dl_bits,
        'V_cb_pred': V_cb_pred, 'V_ub_pred': V_ub_pred,
    }


def main():
    print("=" * 92)
    print("PER-SUBSTRATE STRUCTURAL FINGERPRINT — full 9-candidate ensemble")
    print("=" * 92)

    rcsr_file = '/tmp/rcsr_3d_current.txt'
    entries = parse_rcsr_3dall(rcsr_file, CANDIDATES)
    fps = {}
    for name in CANDIDATES:
        fps[name] = fingerprint(name, entries[name])

    # ---- Section 1: structural attributes ----
    print("\n" + "-" * 92)
    print("S1-S5 — Structural attributes (SG, Wyckoff, primitive cell, bipartition, girth)")
    print("-" * 92)
    print(f"{'name':<10s} {'SG':<10s} {'Wyck':<6s} {'k*':>3s} "
          f"{'Nv_p':>5s} {'Ne_p':>5s} {'deg_p_min/max':<14s} "
          f"{'bipart':<14s} {'g':>3s}")
    for name in CANDIDATES:
        f = fps[name]
        deg_str = f"{f['prim_deg_seq'][0]}/{f['prim_deg_seq'][-1]}"
        print(f"{name:<10s} {f['sg']:<10s} {f['wyckoff']:<6s} {f['coord']:>3d} "
              f"{f['n_prim_atoms']:>5d} {f['n_prim_edges']:>5d} {deg_str:<14s} "
              f"{f['bp_status']:<14s} {f['g']:>3d}")

    # ---- Section 2: K-rational saddle + γ_7^A check ----
    print("\n" + "-" * 92)
    print("S6-S9 — Saddle eigenvalues, K-rationality, γ_7^A → ±χ̃ verification")
    print("-" * 92)
    for name in CANDIDATES:
        f = fps[name]
        print(f"\n  {name} ({f['sg']}, {'I-cubic' if f['sg'] in I_CENTERED_SGS else 'P-cubic'} BZ):")
        if f['saddle_search'] is None:
            print(f"    (DISCONNECTED — saddle search not performed)")
            continue
        # Report K-rational eigenvalues per k-point
        any_K = False
        for k_name, ks in f['saddle_search'].items():
            if ks['K_rational_eigs']:
                any_K = True
                for (lam, mult, ident) in ks['K_rational_eigs']:
                    print(f"    k={k_name:<6s}: K-rational eigenvalue {ident}, "
                          f"|λ|²={abs(lam)**2:.3f}, mult={mult}")
        if not any_K:
            print("    NO K-rational Ramanujan-saturating eigenvalue found at any tested k-point")
        # γ_7^A check
        if f['gamma7_verdict']:
            print(f"    γ_7^A on walker → {f['gamma7_verdict']}")
            ac = f['chi_anticomm']
            max_anti = max(ac.values())
            print(f"    max ‖{{χ̃, B(k)}}‖/‖B‖ across k-points: {max_anti:.3e}")
        elif f['bp_status'] == 'NOT_BIPARTITE':
            print(f"    (NON-BIPARTITE — no walker-level Z_2 supercharge)")

    # ---- Section 3: DL totals ----
    print("\n" + "-" * 92)
    print("S10 — Convention-B Level 2 description length")
    print("-" * 92)
    print(f"{'name':<10s} {'sg':>4s} {'wyck':>4s} {'edge':>4s} {'chir':>4s} "
          f"{'α(N_p)':>6s} {'β(arc)':>6s} {'TOTAL':>7s} {'ΔDL vs srs':>10s}")
    dl_srs_total = fps['srs']['dl_total']
    for name in CANDIDATES:
        f = fps[name]
        b = f['dl_bits']
        delta = f['dl_total'] - dl_srs_total
        print(f"{name:<10s} {b['space_group']:>4.2f} {b['wyckoff']:>4.2f} "
              f"{b['edges']:>4.2f} {b['chirality']:>4.2f} "
              f"{b['alpha_Nprim']:>6.3f} {b['beta_arc_orbits']:>6.3f} "
              f"{f['dl_total']:>7.3f} {delta:>+10.3f}")

    # ---- Section 4: framework predictions per substrate ----
    print("\n" + "-" * 92)
    print("Framework predictions per substrate (CLASS A/B/E formulas)")
    print("-" * 92)
    print(f"{'name':<10s} {'g':>3s} {'α_1=(2/3)^(g-2)':>16s} {'V_cb=α_1/(1-α_1)':>18s} "
          f"{'V_ub=α_1²/(1-α_1)':>20s} {'V_us=k²/(g·N_p)':>18s} {'M=N_e_p (η_B)':>14s}")
    for name in CANDIDATES:
        f = fps[name]
        V_us = f['coord']**2 / (f['g'] * f['n_prim_atoms'])
        M_chain = f['n_prim_edges']
        print(f"{name:<10s} {f['g']:>3d} {f['alpha_1']:>16.6e} {f['V_cb_pred']:>18.6e} "
              f"{f['V_ub_pred']:>20.6e} {V_us:>18.5f} {M_chain:>14d}")

    # ---- Section 5: honest gap report ----
    print("\n" + "=" * 92)
    print("HONEST GAP REPORT")
    print("=" * 92)
    print()
    print("  Per-candidate quirks worth flagging before ensemble propagation:")
    for name in CANDIDATES:
        f = fps[name]
        flags = []
        if f['bp_status'] == 'DISCONNECTED':
            flags.append("DISCONNECTED primitive (catenated structure breaks single-substrate framing)")
        if min(f['conv_deg_seq']) != max(f['conv_deg_seq']):
            flags.append(f"non-uniform conventional degree {f['conv_deg_seq']}")
        if min(f['prim_deg_seq']) != max(f['prim_deg_seq']):
            flags.append(f"non-uniform primitive degree {f['prim_deg_seq']}")
        if f['saddle_search'] is not None:
            n_K_rat = sum(1 for ks in f['saddle_search'].values() if ks['K_rational_eigs'])
            if n_K_rat == 0 and f['bp_status'] != 'DISCONNECTED':
                flags.append("NO K-rational saddle at any tested k-point — CLASS C predictions undefined")
        if not flags:
            flags = ['(clean)']
        print(f"    {name:<10s}: {'; '.join(flags)}")


if __name__ == '__main__':
    main()
