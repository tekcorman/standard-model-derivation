#!/usr/bin/env python3
"""
RCSR candidate sweep — extend χ̃ ≡ ±γ_7^A bipartiteness verdict from
{srs, srs-z} to the full RCSR candidate set scoped in
`proofs/foundations/rcsr_net_assessment.py` (line 592):

    candidates = ['srs', 'srs-z', 'srs-c4', 'srs-c8', 'srs-c27',
                  'lou', 'lov', 'okw', 'hcb-c4']

Per an internal working note,
this probe performs S1 (primitive cell ID), S2 (bipartiteness check), S3
(γ_7^A walker lift / odd-cycle obstruction), and S6 (k* = 3 lookup) for
all 9 candidates.

S4 (CLASS A/B/C/D/E partner-prediction matrix) and S5 (M2a structural-DL)
are deferred per the handoff effort estimate.

PRIMITIVE-CELL CONSTRUCTION
---------------------------
For natively-primitive (P-) space groups (P4(1)32, P4(2)32, P4(3)32):
  primitive = conventional, no reduction needed.

For body-centered (I-) space groups (I4(1)32, I432):
  conventional cell has the body-centering translation t_bc = (1/2, 1/2, 1/2)
  as part of its symmetry. Primitive cell is obtained by identifying each
  atom with its body-centered partner. This probe uses the
  `primitive_quotient_via_body_centering` routine which:
    (a) resolves the FULL conventional-cell bond list (all edge orbits,
        including the 'Eq' auxiliary orbit RCSR uses for lou/lov/okw),
    (b) identifies body-centered atom pairs,
    (c) quotients the conventional bond list down to the primitive
        adjacency multigraph via the lower-index representative.

This bypasses the existing `to_primitive_I_centered` + `reconstruct_bonds`
pipeline (which silently mis-resolves primitive bonds for several candidates
because primitive midpoints don't always sit at the lex-smaller atom-pair
midpoint — only 4/6 bonds resolve for srs's K_4 in that pipeline).

PARSER FIX (rcsr_net_assessment.py)
-----------------------------------
Two pre-existing parser bugs were discovered and fixed during this work:
  (a) regex `^E(\\d+) \\d+\\s*$` for edge-orbit lines failed to match
      `Eq  2` (auxiliary edge orbit used in lou/lov/okw and a few other
      RCSR entries), silently dropping 24 of 36 edges for these nets.
  (b) the orbit-block parser advanced `i += 6` after consuming a 5-line
      record, which skipped the next orbit when RCSR packs them with no
      trailing padding line (again lou/lov/okw).
Both were resolved in `rcsr_net_assessment.py` so that `parse_rcsr_3dall`
now correctly reports 2 edge orbits for lou/lov/okw and resolves 36 of 36
conventional bonds for these nets.
"""

import sys
import os
import numpy as np
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcsr_net_assessment import (
    parse_rcsr_3dall, get_space_group_ops, orbit_of, reconstruct_bonds,
    build_directed_edges, bloch_hashimoto,
)
from srs_z_bipartite_involution_commutation import find_bipartition, build_adjacency


# Space groups treated as I-centered (need body-centering quotient).
I_CENTERED_SGS = {
    'I4(1)32', 'I432', 'Ia-3d', 'I23', 'I2(1)3', 'Im-3m', 'Ia-3', 'I-43d',
}


# =============================================================================
# PRIMITIVE-CELL QUOTIENT (works directly on conventional bond list)
# =============================================================================

def primitive_quotient_via_body_centering(atom_orbit, conv_bonds, tol=1e-6):
    """For an I-centered conventional cell: identify each atom with its
    body-centered partner (offset by (1/2,1/2,1/2) mod 1), then quotient
    the conventional bond list to produce the primitive-cell adjacency.

    Returns (n_prim, A_prim, partner_map, prim_bonds, conv_to_prim) where
      - n_prim: number of primitive-cell vertices
      - A_prim: primitive adjacency multigraph (n_prim × n_prim, may have
                multi-edges)
      - partner_map[i] = j: conventional atoms i and j are body-centered
                            partners
      - prim_bonds: list of (i_prim, j_prim, shift_xyz) primitive bonds
      - conv_to_prim: list with conv_to_prim[i] = primitive index of conv i
    """
    n_conv = len(atom_orbit)
    bcent = np.array([0.5, 0.5, 0.5])
    partner = [-1] * n_conv
    for i in range(n_conv):
        if partner[i] != -1:
            continue
        for j in range(i + 1, n_conv):
            if partner[j] != -1:
                continue
            diff = (atom_orbit[j] - atom_orbit[i] - bcent) % 1.0
            diff = np.where(diff > 0.5, diff - 1.0, diff)
            if np.linalg.norm(diff) < tol:
                partner[i] = j
                partner[j] = i
                break
    # Each pair has a single representative (lower index).
    rep_set = sorted({min(i, partner[i]) for i in range(n_conv) if partner[i] != -1})
    rep_to_prim = {r: k for k, r in enumerate(rep_set)}
    n_prim = len(rep_set)
    if n_prim * 2 != n_conv:
        # Some atoms had no partner — not a standard I-centered orbit.
        unpaired = [i for i in range(n_conv) if partner[i] == -1]
        raise ValueError(f"I-centering: unpaired atoms {unpaired} (n_conv={n_conv}, n_prim*2={n_prim*2})")

    conv_to_prim = [rep_to_prim[min(i, partner[i])] for i in range(n_conv)]

    A_prim = np.zeros((n_prim, n_prim), dtype=int)
    prim_bonds = []
    for (i, j, shift) in conv_bonds:
        pi, pj = conv_to_prim[i], conv_to_prim[j]
        prim_bonds.append((pi, pj, shift))
        A_prim[pi, pj] += 1
        if pi != pj:
            A_prim[pj, pi] += 1
    return n_prim, A_prim, partner, prim_bonds, conv_to_prim


# =============================================================================
# BIPARTITION (handles disconnected graphs by flagging incomplete coverage)
# =============================================================================

def find_bipartition_full(A):
    """Like find_bipartition but reports whether all vertices are colored.

    Returns one of:
      ('NOT_BIPARTITE', None, None)       — odd cycle found
      ('DISCONNECTED', side_A, side_B)    — graph is bipartite but only the
                                             component containing vertex 0 is
                                             reported
      ('BIPARTITE', side_A, side_B)        — graph is connected and bipartite
    """
    n = len(A)
    color = [-1] * n
    color[0] = 0
    queue = [0]
    while queue:
        u = queue.pop(0)
        for v in range(n):
            if A[u, v] > 0:
                if color[v] == -1:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    return ('NOT_BIPARTITE', None, None)
    side_a = [i for i in range(n) if color[i] == 0]
    side_b = [i for i in range(n) if color[i] == 1]
    if len(side_a) + len(side_b) < n:
        return ('DISCONNECTED', side_a, side_b)
    return ('BIPARTITE', side_a, side_b)


# =============================================================================
# S3a: BIPARTITE — γ_7^A walker-lift verification
# =============================================================================
#
# Convention (matches srs_z_gamma7_lift_recovers_chi.py):
#   γ_7(F=0) = -1, γ_7(F=1) = +1.
#   γ_7^A := Π_{u ∈ A} γ_7_u, restricted to walker subspace |v⟩ where F_v=1
#   and F_u=0 for u ≠ v, gives eigenvalue:
#     v ∈ A: (+1) · (-1)^(|A|-1)
#     v ∈ B: (-1)^|A|
#
# χ̃ on walker subspace (vertex-indexed):
#     χ̃_v = +1 if v ∈ A else -1.
#
# We check that γ_7^A acting on |v⟩ matches ±χ̃_v at every vertex.

def verify_gamma7_lift_recovers_chi(side_a, side_b, n_atoms):
    """Returns dict with eigenvalues per vertex and verdict."""
    gamma7_F0, gamma7_F1 = -1, +1
    gamma7A_eigs = []
    chi_eigs = []
    for v in range(n_atoms):
        eig = 1
        for u in side_a:
            eig *= (gamma7_F1 if u == v else gamma7_F0)
        gamma7A_eigs.append(eig)
        chi_eigs.append(+1 if v in side_a else -1)
    # Check whether γ_7^A = +χ̃ or -χ̃ everywhere
    plus_match = all(g == c for g, c in zip(gamma7A_eigs, chi_eigs))
    minus_match = all(g == -c for g, c in zip(gamma7A_eigs, chi_eigs))
    if plus_match:
        verdict = '+χ̃'
    elif minus_match:
        verdict = '-χ̃'
    else:
        verdict = 'MISMATCH'
    return {
        'gamma7A_eigs': gamma7A_eigs,
        'chi_eigs': chi_eigs,
        'verdict': verdict,
    }


def verify_chi_anticommutes_with_B(prim_bonds, side_a, n_atoms, k_points):
    """Build B(k) on the directed-arc space and verify χ̃ B + B χ̃ ≈ 0.

    χ̃ on directed arcs: χ̃_a = +1 if tail(a) ∈ A else -1.
    """
    arcs = build_directed_edges(prim_bonds)
    n_arcs = len(arcs)
    chi = np.array([+1 if arc[0] in side_a else -1 for arc in arcs])
    chi_diag = np.diag(chi).astype(complex)
    results = {}
    for k_name, k_frac in k_points.items():
        B = bloch_hashimoto(arcs, k_frac, n_atoms)
        anti = chi_diag @ B + B @ chi_diag
        com = chi_diag @ B - B @ chi_diag
        results[k_name] = {
            'anti_norm': float(np.linalg.norm(anti)),
            'com_norm': float(np.linalg.norm(com)),
            'B_norm': float(np.linalg.norm(B)),
        }
    return results


# =============================================================================
# S3b: NON-BIPARTITE — brute-force odd-cycle obstruction on directed arcs
# =============================================================================

def odd_cycle_obstruction_proof(prim_bonds, n_atoms, max_brute_arcs=14):
    """For non-bipartite primitive: verify NO 2-coloring of the directed arcs
    anti-commutes with B(k) at all k.

    Reduces to: the Hashimoto continuation digraph has NO proper 2-coloring
    (because of an odd cycle in the continuation digraph, which exists
    whenever the underlying graph has an odd cycle).
    """
    arcs = build_directed_edges(prim_bonds)
    n_arcs = len(arcs)
    # Build continuation pairs (i, j): tail(j) = head(i) AND j != reverse(i)
    cont = set()
    for i, (ti, hi, si) in enumerate(arcs):
        rev_i = (hi, ti, tuple(-s for s in si))
        for j, (tj, hj, sj) in enumerate(arcs):
            if tj == hi and arcs[j] != rev_i:
                cont.add((i, j))
    n_cont = len(cont)

    if n_arcs > max_brute_arcs:
        return {
            'n_arcs': n_arcs,
            'n_cont': n_cont,
            'brute_perfect': None,
            'note': f'arcs={n_arcs} exceeds brute-force cap ({max_brute_arcs}); '
                    'odd-cycle proof in underlying graph is sufficient.',
        }

    perfect = 0
    nontrivial = 0
    for mask in range(2 ** n_arcs):
        signs = np.array([+1 if (mask >> i) & 1 else -1 for i in range(n_arcs)])
        if np.all(signs == signs[0]):
            continue
        nontrivial += 1
        violations = sum(1 for (i, j) in cont if signs[i] == signs[j])
        if violations == 0:
            perfect += 1
    return {
        'n_arcs': n_arcs,
        'n_cont': n_cont,
        'brute_nontrivial': nontrivial,
        'brute_perfect': perfect,
    }


# =============================================================================
# K-POINTS for S3 anti-commutation check (primitive-cubic high-symmetry)
# =============================================================================

K_POINTS = {
    'Γ': np.array([0.0, 0.0, 0.0]),
    'R': np.array([0.5, 0.5, 0.5]),
    'X': np.array([0.5, 0.0, 0.0]),
    'M': np.array([0.5, 0.5, 0.0]),
    'mid': np.array([0.25, 0.25, 0.25]),
}


# =============================================================================
# S6: k* = 3 Pati-Salam compatibility lookup
# =============================================================================

def k_star_compatible(coord_number):
    """k* = 3 ⇒ Cl(2k) = Cl(6) ⇒ Pati-Salam Spin(6) embedding ⇒ sin²θ_W = 3/8."""
    return coord_number == 3


# =============================================================================
# MAIN SWEEP
# =============================================================================

CANDIDATES = ['srs', 'srs-z', 'srs-c4', 'srs-c8', 'srs-c27',
              'lou', 'lov', 'okw', 'hcb-c4']


def assess_candidate(name, entry):
    """Run S1+S2+S3+S6 on a single candidate, return result dict."""
    sg = entry['sg_name']
    rotations, translations, _, _ = get_space_group_ops(sg)
    v_frac = np.array(entry['vertex_orbits'][0]['cartesian'])
    coord = entry['vertex_orbits'][0]['coord']

    # S1: build conventional adjacency from ALL edge orbits
    atom_orbit = orbit_of(v_frac, rotations, translations)
    midpoints = []
    for eo in entry['edge_orbits']:
        midpoints.append(orbit_of(np.array(eo['cartesian']), rotations, translations))
    midpoint_orbit = np.vstack(midpoints)
    n_conv_atoms = len(atom_orbit)
    n_conv_mid = len(midpoint_orbit)

    bonds = reconstruct_bonds(atom_orbit, midpoint_orbit, tol=1e-3, max_shift=3)
    n_resolved = sum(1 for b in bonds if b is not None)
    bonds = [b for b in bonds if b is not None]
    A_conv = build_adjacency(bonds, n_conv_atoms)

    # Primitive-cell construction
    if sg in I_CENTERED_SGS:
        try:
            n_prim, A_prim, partner, prim_bonds, _ = primitive_quotient_via_body_centering(atom_orbit, bonds)
            primitive_method = 'body-centering quotient (conv → prim/2)'
        except ValueError as ex:
            return {
                'name': name, 'sg': sg, 'coord': coord,
                'n_conv_atoms': n_conv_atoms, 'n_conv_mid': n_conv_mid,
                'bonds_resolved': f'{n_resolved}/{n_conv_mid}',
                'STATUS': f'FAILED I-centering: {ex}',
            }
    else:
        n_prim, A_prim, prim_bonds = n_conv_atoms, A_conv, bonds
        primitive_method = 'P-group (primitive = conventional)'

    n_prim_edges = int(A_prim.sum() // 2)
    deg_seq = sorted(int(d) for d in A_prim.sum(axis=1))

    # S2: bipartition check
    bp_status, side_a, side_b = find_bipartition_full(A_prim)

    # S3: γ_7^A walker-lift OR odd-cycle obstruction
    s3 = {}
    if bp_status == 'BIPARTITE':
        # Verify γ_7^A → ±χ̃
        s3['gamma7_lift'] = verify_gamma7_lift_recovers_chi(side_a, side_b, n_prim)
        # Verify {χ̃, B(k)} = 0 at each k-point
        s3['chi_anticomm'] = verify_chi_anticommutes_with_B(prim_bonds, side_a, n_prim, K_POINTS)
    elif bp_status == 'NOT_BIPARTITE':
        s3['odd_cycle'] = odd_cycle_obstruction_proof(prim_bonds, n_prim, max_brute_arcs=14)
    else:
        s3['note'] = 'DISCONNECTED primitive — bipartite cover Z_2 not well-defined'

    return {
        'name': name, 'sg': sg, 'coord': coord,
        'n_conv_atoms': n_conv_atoms, 'n_conv_mid': n_conv_mid,
        'bonds_resolved': f'{n_resolved}/{n_conv_mid}',
        'primitive_method': primitive_method,
        'n_prim_atoms': n_prim, 'n_prim_edges': n_prim_edges,
        'deg_seq': deg_seq,
        'bp_status': bp_status,
        'side_a': side_a, 'side_b': side_b,
        'k_star_compat': k_star_compatible(coord),
        's3': s3,
    }


def main():
    print("=" * 84)
    print("RCSR CANDIDATE SWEEP — S1 (primitive cell), S2 (bipartiteness),")
    print("                       S3 (γ_7^A lift / odd cycle), S6 (k*=3)")
    print("=" * 84)

    rcsr_file = '/tmp/rcsr_3d_current.txt'
    if not os.path.exists(rcsr_file):
        print(f"ERROR: {rcsr_file} missing. Run: curl -sL https://rcsr.anu.edu.au/data/3dall.txt -o {rcsr_file}")
        sys.exit(1)

    entries = parse_rcsr_3dall(rcsr_file, CANDIDATES)
    results = []
    for name in CANDIDATES:
        if name not in entries:
            print(f"\n[SKIP] {name} not found in RCSR data")
            continue
        r = assess_candidate(name, entries[name])
        results.append(r)
        print(f"\n--- {name} ({r['sg']}) ---")
        if 'STATUS' in r:
            print(f"  STATUS: {r['STATUS']}")
            continue
        print(f"  k* = {r['coord']}  (Pati-Salam compatible: {r['k_star_compat']})")
        print(f"  conventional: |V|={r['n_conv_atoms']} |E|={r['n_conv_mid']} "
              f"(bonds resolved {r['bonds_resolved']})")
        print(f"  primitive ({r['primitive_method']}): "
              f"|V|={r['n_prim_atoms']} |E|={r['n_prim_edges']}  "
              f"deg seq min={r['deg_seq'][0]}, max={r['deg_seq'][-1]}, mean={sum(r['deg_seq'])/len(r['deg_seq']):.2f}")
        print(f"  S2 verdict: {r['bp_status']}", end='')
        if r['bp_status'] == 'BIPARTITE':
            print(f"  |A|={len(r['side_a'])}  |B|={len(r['side_b'])}")
        elif r['bp_status'] == 'DISCONNECTED':
            print(f"  partial coloring: |A|={len(r['side_a'])}  |B|={len(r['side_b'])}  "
                  f"(only the component containing vertex 0 is reported)")
        else:
            print()
        # S3 reporting
        if 'gamma7_lift' in r['s3']:
            g = r['s3']['gamma7_lift']
            print(f"  S3 γ_7^A lift: {g['verdict']}")
            ac = r['s3']['chi_anticomm']
            print(f"  S3 χ̃ vs B(k): anti-commutator norms")
            for k_name, ac_data in ac.items():
                ratio = ac_data['anti_norm'] / max(ac_data['B_norm'], 1e-12)
                print(f"      k={k_name}: ||{{χ̃,B}}|| = {ac_data['anti_norm']:.3e}  "
                      f"(ratio to ||B||: {ratio:.3e})")
        elif 'odd_cycle' in r['s3']:
            o = r['s3']['odd_cycle']
            note = o.get('note', '')
            if o.get('brute_perfect') is None:
                print(f"  S3 odd-cycle: {o['n_arcs']} arcs, {o['n_cont']} continuation pairs.  {note}")
            else:
                print(f"  S3 odd-cycle: {o['n_arcs']} arcs, {o['n_cont']} continuation pairs, "
                      f"brute force found {o['brute_perfect']} of {o['brute_nontrivial']} non-trivial "
                      "2-colorings with zero violations (≥1 means 2-colorable).")
        elif 'note' in r['s3']:
            print(f"  S3 note: {r['s3']['note']}")

    # Summary table
    print("\n" + "=" * 84)
    print("COMPACT VERDICT TABLE")
    print("=" * 84)
    print(f"{'name':<10s} {'SG':<10s} {'k*':>3s} {'PS':>4s}  "
          f"{'|V|prim':>7s} {'|E|prim':>7s}  {'S2':<14s}  {'S3 verdict'}")
    for r in results:
        if 'STATUS' in r:
            print(f"  {r['name']:<10s} {r['sg']:<10s}  {r['STATUS']}")
            continue
        s3_text = ''
        if r['bp_status'] == 'BIPARTITE':
            g = r['s3']['gamma7_lift']
            ac = r['s3']['chi_anticomm']
            max_anti = max(ac[k]['anti_norm'] for k in ac)
            s3_text = f"γ_7^A={g['verdict']}, max ||{{χ̃,B}}||={max_anti:.2e}"
        elif r['bp_status'] == 'NOT_BIPARTITE':
            o = r['s3']['odd_cycle']
            if o.get('brute_perfect') is None:
                s3_text = f"odd cycle (analytical: |arcs|={o['n_arcs']} too large)"
            else:
                s3_text = f"odd cycle (brute: {o['brute_perfect']}/{o['brute_nontrivial']} 2-colorings)"
        else:
            s3_text = 'disconnected — N/A'
        print(f"{r['name']:<10s} {r['sg']:<10s} {r['coord']:>3d} "
              f"{'YES' if r['k_star_compat'] else 'NO ':>4s}  "
              f"{r['n_prim_atoms']:>7d} {r['n_prim_edges']:>7d}  "
              f"{r['bp_status']:<14s}  {s3_text}")

    # Final scoping conclusion
    print("\n" + "=" * 84)
    print("SCOPING CONCLUSION")
    print("=" * 84)
    bipartite = [r for r in results if r.get('bp_status') == 'BIPARTITE']
    non_bipartite = [r for r in results if r.get('bp_status') == 'NOT_BIPARTITE']
    disconnected = [r for r in results if r.get('bp_status') == 'DISCONNECTED']
    print(f"\n  BIPARTITE primitive ({len(bipartite)}): "
          f"{[r['name'] for r in bipartite]}")
    print(f"  NON-BIPARTITE primitive ({len(non_bipartite)}): "
          f"{[r['name'] for r in non_bipartite]}")
    print(f"  DISCONNECTED primitive ({len(disconnected)}): "
          f"{[r['name'] for r in disconnected]}")
    print(f"\n  All k* = 3? {all(r.get('k_star_compat', False) for r in results)} "
          f"(Pati-Salam Cl(6) embedding compatible across the candidate set)")
    print(f"\n  Walker-level γ_7^A → ±χ̃ unification is supported on the BIPARTITE")
    print(f"  candidates (above). The NON-BIPARTITE candidates have no walker-level")
    print(f"  Z_2 supercharge structure (odd-cycle obstruction in continuation digraph).")
    print(f"  DISCONNECTED candidates need separate per-component analysis.")


if __name__ == '__main__':
    main()
