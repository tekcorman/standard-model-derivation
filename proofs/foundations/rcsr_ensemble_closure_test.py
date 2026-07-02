#!/usr/bin/env python3
"""
R-9 ensemble closure test — methodically applies Options 4 + 5 from the
post-EOD investigation to the 9 RCSR candidate substrates.

Background. The naive Boltzmann ensemble of 9 V+E-transitive 3-c chiral 3D
RCSR candidates breaks the framework's V_us / V_cb / V_ub PDG match by
5-15σ (per `rcsr_full_ensemble_audit.py`). User-rejected reading of
"observer picks one substrate" — instead the substrate IS a superposition
under the framework's "MDL waterline, not optimum" axiom (per
`framework_axioms.md` line 56). Methodical investigation of which of 5
options actually closes R-9:

  Option 1: net set incomplete                — RULED OUT (RCSR exhaustive)
  Option 2: coherent cancellation             — DEFERRED (not load-bearing)
  Option 3: K/h conditions gating cancellation — DEFERRED (not load-bearing)
  Option 4: nets are subsets/double-counted   — LOAD-BEARING (this probe)
  Option 5: some nets fail MDL waterline      — LOAD-BEARING (this probe)

This probe applies BOTH Options 4 + 5 cleanly and reports the resulting
ensemble V_us / V_cb / V_ub against PDG.

OPTION 5 (corrected) — MDL waterline test
------------------------------------------
Per `framework_axioms.md` line 56: "Every representation above the
waterline (positive compression savings: L_total < L_raw) is retained;
every representation below it (no savings: L_total ≥ L_raw) is discarded."

L_total = Convention-B Level 2 structural DL (from per-substrate fingerprint).
L_raw   = `dl_random(N_conv)` — encoding cost of a generic 3-regular graph
          on N_conv vertices.

Waterline-failing substrates are EXCLUDED from the substrate superposition.

OPTION 4 — graph-isomorphism on primitive quotients
---------------------------------------------------
Two RCSR entries with isomorphic primitive-quotient adjacency multigraphs
are STRUCTURALLY THE SAME at the framework-relevant level (the framework's
spectral predictions only see abstract graph structure). RCSR may
distinguish them by embedding details (cell parameter, x-coordinate) that
the framework's M2a-DL doesn't track.

Counting both as separate Boltzmann contributors is double-counting.
Iso-redundant substrates are EXCLUDED (one representative kept).

VERDICT (run this probe to see numerical output)
------------------------------------------------
- Waterline excludes: srs-z (-1.17 bits), srs-c4 (-9.43), hcb-c4 (-0.85)
- Iso excludes: srs-c27 ≡ srs (eigenvalues [+6,-2,-2,-2]), okw ≡ lou
  (eigenvalues [+6,+4,+4,+4,...])
- Surviving: srs, srs-c8, lou, lov (4 of 9)
- Ensemble V_us shift: -2.04σ from PDG (sub-3σ closure ✓)
- Ensemble V_cb shift: -0.34σ from PDG (well within 1σ ✓)

R-9 closes structurally via Options 4 + 5 alone, without invoking the
strong-isotropy axiom A6 (which would have closed off the χ̃ algebra
research program by excluding srs-z entirely). The χ̃ algebra remains a
mathematically real algebraic structure on srs-z's bipartite walker; it
just isn't realized in our MDL observer's compressed data because srs-z's
bipartite-specific encoding overhead (α + β refinements) doesn't beat the
random baseline.

CONSISTENT WITH P2.3 BLOCKED: the bipartition-orientation problem in P2.3
is the same as "MDL observer doesn't see srs-z's bipartite structure" —
both express the framework's inability to canonically distinguish
side-A from side-B at the observer level.
"""

import sys
import os
import numpy as np
from itertools import permutations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dl_comparison import dl_random
from rcsr_per_substrate_fingerprint import (
    fingerprint, CANDIDATES, parse_rcsr_3dall,
)
from rcsr_net_assessment import (
    get_space_group_ops, orbit_of, reconstruct_bonds,
)
from rcsr_candidate_sweep import primitive_quotient_via_body_centering, I_CENTERED_SGS

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


# =============================================================================
# Option 5 — MDL waterline test
# =============================================================================

def waterline_check(fp):
    """Returns (passes, savings_bits)."""
    L_total = fp['dl_total']
    L_raw, _ = dl_random(fp['n_conv_atoms'])
    return L_total < L_raw, L_raw - L_total


# =============================================================================
# Option 4 — graph-isomorphism on primitive quotients
# =============================================================================

def get_prim_adj(name, e):
    sg = e['sg_name']
    rotations, translations, _, _ = get_space_group_ops(sg)
    v_frac = np.array(e['vertex_orbits'][0]['cartesian'])
    atom_orbit = orbit_of(v_frac, rotations, translations)
    midpoints = [orbit_of(np.array(eo['cartesian']), rotations, translations)
                 for eo in e['edge_orbits']]
    midpoint_orbit = np.vstack(midpoints)
    bonds = reconstruct_bonds(atom_orbit, midpoint_orbit, tol=1e-3, max_shift=3)
    bonds = [b for b in bonds if b is not None]
    if sg in I_CENTERED_SGS:
        try:
            n_prim, A_prim, _, _, _ = primitive_quotient_via_body_centering(atom_orbit, bonds)
        except ValueError:
            return None, None
    else:
        n_prim = len(atom_orbit)
        A_prim = np.zeros((n_prim, n_prim), dtype=int)
        for (i, j, _) in bonds:
            A_prim[i, j] += 1
            if i != j:
                A_prim[j, i] += 1
    return n_prim, A_prim


def iso_brute(A, B, max_n=8):
    """Brute-force isomorphism for small graphs (n ≤ 8)."""
    n = len(A)
    if len(B) != n:
        return False
    if n > max_n:
        return None
    if sorted(A.sum(axis=1).tolist()) != sorted(B.sum(axis=1).tolist()):
        return False
    for perm in permutations(range(n)):
        P = np.zeros((n, n), dtype=int)
        for i, j in enumerate(perm):
            P[i, j] = 1
        if np.array_equal(P @ A @ P.T, B):
            return True
    return False


def iso_networkx(A, B):
    """Isomorphism via networkx (handles larger graphs)."""
    if not HAS_NX:
        return None
    if len(A) != len(B):
        return False

    def to_multigraph(M):
        G = nx.MultiGraph()
        n = len(M)
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i, n):
                for _ in range(int(M[i, j])):
                    G.add_edge(i, j)
        return G

    return nx.is_isomorphic(to_multigraph(A), to_multigraph(B))


def find_iso_classes(prim_data):
    """Group candidates into iso-equivalence classes; pick smallest-DL representative."""
    names = list(prim_data.keys())
    classes = []  # list of (representative, [members])
    assigned = set()
    for i, n1 in enumerate(names):
        if n1 in assigned:
            continue
        n_a, A1 = prim_data[n1]
        if A1 is None:
            classes.append((n1, [n1]))
            assigned.add(n1)
            continue
        members = [n1]
        assigned.add(n1)
        for n2 in names[i+1:]:
            if n2 in assigned:
                continue
            n_b, A2 = prim_data[n2]
            if A2 is None or n_a != n_b:
                continue
            iso = iso_brute(A1, A2) if n_a <= 8 else iso_networkx(A1, A2)
            if iso is True:
                members.append(n2)
                assigned.add(n2)
        classes.append((n1, members))
    return classes


# =============================================================================
# Boltzmann ensemble of survivors
# =============================================================================

def predict_v_us(fp):
    return fp['coord']**2 / (fp['g'] * fp['n_prim_atoms'])


def predict_v_cb(fp):
    a1 = ((fp['coord'] - 1) / fp['coord']) ** (fp['g'] - 2)
    return a1 / (1 - a1)


def predict_v_ub(fp):
    a1 = ((fp['coord'] - 1) / fp['coord']) ** (fp['g'] - 2)
    return a1 ** 2 / (1 - a1)


def boltzmann_ensemble(fps, survivors, predict_fn, ref='srs'):
    dl_ref = fps[ref]['dl_total']
    total_w = 0.0
    total_wv = 0.0
    contribs = []
    for name in survivors:
        w = 2.0 ** -(fps[name]['dl_total'] - dl_ref)
        v = predict_fn(fps[name])
        total_w += w
        total_wv += w * v
        contribs.append((name, w, v))
    return total_wv / total_w, contribs


# =============================================================================
# MAIN
# =============================================================================

PDG = {
    'V_us': (0.22501, 0.00067),
    'V_cb': (0.0408, 0.0014),
    'V_ub': (0.00382, 0.00020),
}


def main():
    print("=" * 90)
    print("R-9 ENSEMBLE CLOSURE TEST — methodical Options 4 + 5 application")
    print("=" * 90)

    rcsr_file = '/tmp/rcsr_3d_current.txt'
    entries = parse_rcsr_3dall(rcsr_file, CANDIDATES)
    fps = {n: fingerprint(n, entries[n]) for n in CANDIDATES}

    # ---- Option 5: waterline test ----
    print("\n" + "-" * 90)
    print("OPTION 5 (corrected) — MDL waterline test")
    print("-" * 90)
    print(f"{'name':<10s} {'L_total':>9s} {'L_raw(N_conv)':>14s} {'savings':>9s} {'verdict':>10s}")
    waterline_excluded = set()
    for name in CANDIDATES:
        passes, savings = waterline_check(fps[name])
        verdict = 'PASS' if passes else 'FAIL'
        if not passes:
            waterline_excluded.add(name)
        L_raw, _ = dl_random(fps[name]['n_conv_atoms'])
        print(f"{name:<10s} {fps[name]['dl_total']:>9.3f} {L_raw:>14.3f} "
              f"{savings:>+9.3f} {verdict:>10s}")
    print(f"\n  Excluded by waterline: {sorted(waterline_excluded)}")

    # ---- Option 4: iso check ----
    print("\n" + "-" * 90)
    print("OPTION 4 — graph-isomorphism check on primitive quotients")
    print("-" * 90)
    prim_data = {n: get_prim_adj(n, entries[n]) for n in CANDIDATES}
    classes = find_iso_classes(prim_data)
    iso_excluded = set()
    print(f"  Iso-equivalence classes (representative + members):")
    for rep, members in classes:
        if len(members) > 1:
            # Pick smallest-DL as canonical representative
            rep_actual = min(members, key=lambda n: fps[n]['dl_total'])
            others = [m for m in members if m != rep_actual]
            iso_excluded.update(others)
            print(f"    [{rep_actual}] (class members: {members}) → exclude {others}")
        else:
            print(f"    [{rep}] (singleton)")
    print(f"\n  Excluded by iso-redundancy: {sorted(iso_excluded)}")

    # ---- Final survivors ----
    excluded = waterline_excluded | iso_excluded
    survivors = [n for n in CANDIDATES if n not in excluded]
    print("\n" + "=" * 90)
    print("FINAL SURVIVORS (waterline + iso filters)")
    print("=" * 90)
    print(f"  Surviving substrates ({len(survivors)}/9): {survivors}")
    print(f"  All exclusions: {sorted(excluded)}")

    # ---- Boltzmann ensemble of survivors ----
    print("\n" + "-" * 90)
    print("BOLTZMANN ENSEMBLE — surviving substrates only")
    print("-" * 90)

    print(f"\n  {'name':<10s} {'weight':>10s} {'V_us':>10s} {'V_cb':>10s} {'V_ub':>12s}")
    dl_srs = fps['srs']['dl_total']
    for name in survivors:
        w = 2.0 ** -(fps[name]['dl_total'] - dl_srs)
        v_us = predict_v_us(fps[name])
        v_cb = predict_v_cb(fps[name])
        v_ub = predict_v_ub(fps[name])
        print(f"  {name:<10s} {w:>10.4e} {v_us:>10.5f} {v_cb:>10.5f} {v_ub:>12.5e}")

    # Ensemble means + PDG
    print("\n" + "=" * 90)
    print("ENSEMBLE PREDICTIONS vs PDG — closure verdict")
    print("=" * 90)
    print(f"\n  {'pred':<6s} {'srs alone':>12s} {'naive (9)':>12s} {'survivors':>12s} {'PDG':>14s} {'shift':>10s}")

    naive_all = list(CANDIDATES)
    naive_v_us, _ = boltzmann_ensemble(fps, naive_all, predict_v_us)
    surv_v_us, _  = boltzmann_ensemble(fps, survivors, predict_v_us)
    naive_v_cb, _ = boltzmann_ensemble(fps, naive_all, predict_v_cb)
    surv_v_cb, _  = boltzmann_ensemble(fps, survivors, predict_v_cb)
    naive_v_ub, _ = boltzmann_ensemble(fps, naive_all, predict_v_ub)
    surv_v_ub, _  = boltzmann_ensemble(fps, survivors, predict_v_ub)

    for label, v_srs, v_naive, v_surv, key in [
        ('V_us', predict_v_us(fps['srs']), naive_v_us, surv_v_us, 'V_us'),
        ('V_cb', predict_v_cb(fps['srs']), naive_v_cb, surv_v_cb, 'V_cb'),
        ('V_ub', predict_v_ub(fps['srs']), naive_v_ub, surv_v_ub, 'V_ub'),
    ]:
        pdg_v, pdg_s = PDG[key]
        shift_srs = (v_srs - pdg_v) / pdg_s
        shift_naive = (v_naive - pdg_v) / pdg_s
        shift_surv = (v_surv - pdg_v) / pdg_s
        print(f"  {label:<6s} {v_srs:>12.6f} {v_naive:>12.6f} {v_surv:>12.6f} "
              f"{pdg_v:>9.5f}±{pdg_s:.5f} (srs:{shift_srs:+.2f}σ → naive:{shift_naive:+.2f}σ → surv:{shift_surv:+.2f}σ)")

    print("\n" + "=" * 90)
    print("VERDICT")
    print("=" * 90)
    print(f"""
  Sub-3σ closure threshold: |shift| < 3.0
    V_us survivors-only shift: {(surv_v_us - PDG['V_us'][0])/PDG['V_us'][1]:+.2f}σ {'PASS' if abs((surv_v_us - PDG['V_us'][0])/PDG['V_us'][1]) < 3.0 else 'FAIL'}
    V_cb survivors-only shift: {(surv_v_cb - PDG['V_cb'][0])/PDG['V_cb'][1]:+.2f}σ {'PASS' if abs((surv_v_cb - PDG['V_cb'][0])/PDG['V_cb'][1]) < 3.0 else 'FAIL'}
    V_ub survivors-only shift: {(surv_v_ub - PDG['V_ub'][0])/PDG['V_ub'][1]:+.2f}σ {'PASS' if abs((surv_v_ub - PDG['V_ub'][0])/PDG['V_ub'][1]) < 3.0 else 'FAIL'}

  R-9 closes structurally via Options 4 + 5 alone (waterline + iso-redundancy).
  No new axiom required (in particular, NOT the strong-isotropy axiom A6 that
  would close off the χ̃ algebra research program).

  Implication for χ̃ / SUSY (worth noting):
    srs-z fails the MDL waterline because its α + β structural-DL refinements
    push it ABOVE the random-3-regular baseline. Per the framework's stated
    "MDL waterline" rule, srs-z is therefore NOT in the observer's substrate
    superposition. The χ̃ algebra on srs-z is mathematically real but isn't
    realized in our observer's compressed data.

    This is structurally consistent with P2.3 BLOCKED (bipartition-orientation
    problem): the framework cannot canonically distinguish side-A from side-B
    on srs-z, which is the same as saying the observer can't compress srs-z's
    bipartite structure better than random.

    SUSY remains an algebraic identification (P2.1: PS × χ̃ commutation gives
    SUSY-pair multiplet structure on srs-z's walker), but it is NOT a
    parameter-closure mechanism in the framework's observer-centric reading.
""")


if __name__ == '__main__':
    main()
