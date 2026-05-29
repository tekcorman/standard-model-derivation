#!/usr/bin/env python3
"""
proofs/flavor/srs_girth_cycle_homology.py

PURPOSE
-------
Classify the 120 girth-10 NB cycles on srs by their H_1 class in the
primitive-cell quotient (with periodic identification). The quotient
has 4 vertices and 6 undirected bonds, so

    rank H_1(srs / translations) = E - V + 1 = 6 - 4 + 1 = 3.

If the 120 girth cycles span a 3-dim subspace of H_1(quotient, Z) (or
its real lift), this is suggestive of an ISOMORPHISM

    C^3_obs (generation Hilbert space, R3 theorem) ≅ H_1(srs / Z, R)

and the conjectural rule

    ΔGen = k  ⇔  k independent girth-cycle homology generators

would have a natural structural realization. This is the load-bearing
candidate for G-Vub-1.

GATE STATUS
-----------
CAS verification of homology rank only. The isomorphism with C^3_obs
remains a conjecture pending an independent identification.
"""

import sys
import os
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vcb_hashimoto_bfs as vcb

bonds_prim     = vcb.bonds_prim          # 12 directed bonds in primitive cell
prim_disps     = vcb.prim_disps          # cell-displacement vectors
N_SUPER        = vcb.N_SUPER
nb_successors  = vcb.nb_successors
in_bounds      = vcb.in_bounds
edge_prim_type = vcb.edge_prim_type
type_label     = vcb.type_label
g              = vcb.g                   # 10


# ── Map directed bond → undirected primitive bond index ───────────────────────

def bond_undirected_index(src, tgt, dc):
    """Map a directed bond (src, tgt, dc) to a canonical undirected index in
    {0, ..., 5} representing the 6 undirected bonds of the primitive cell."""
    # Pair up (i, j, c) with (j, i, -c) as the same undirected edge.
    forward = (src, tgt, tuple(dc))
    backward = (tgt, src, tuple(-d for d in dc))
    forward_idx = None
    backward_idx = None
    for k, (s, t, c) in enumerate(bonds_prim):
        if (s, t, tuple(c)) == forward:
            forward_idx = k
        if (s, t, tuple(c)) == backward:
            backward_idx = k
    return min(forward_idx, backward_idx) if (forward_idx is not None and backward_idx is not None) else None


# Build the canonical undirected bond list (6 entries)
canonical_bonds = []
seen = set()
for k, (s, t, c) in enumerate(bonds_prim):
    fwd = (s, t, tuple(c))
    bwd = (t, s, tuple(-d for d in c))
    key = min(fwd, bwd)
    if key not in seen:
        seen.add(key)
        canonical_bonds.append(fwd)
assert len(canonical_bonds) == 6, f"Expected 6 undirected bonds, got {len(canonical_bonds)}"

print(f"Primitive cell undirected bond inventory:")
for k, (s, t, c) in enumerate(canonical_bonds):
    print(f"  [{k}] (a{s} → a{t}, dc={c})")


# ── For each cycle, compute its homology class as Z^6 → quotient ──────────────
# A directed walk along the cycle traverses each undirected bond a number of
# times with sign (forward = +1, backward = -1). The "net flux" through each
# undirected bond gives a 6-vector. The quotient H_1 = Z^6 / (boundary image).

def directed_to_signed_undirected(src_a, tgt_a, dc):
    """Return (undirected_index, sign) for a directed bond (src, tgt, dc)."""
    fwd = (src_a, tgt_a, tuple(dc))
    bwd = (tgt_a, src_a, tuple(-d for d in dc))
    for k, ucb in enumerate(canonical_bonds):
        if ucb == fwd:
            return (k, +1)
        if ucb == bwd:
            return (k, -1)
    return None


def cycle_flux_vector(cycle):
    """A cycle is a list of directed Hashimoto edges. Each is
    (src_a, src_c, tgt_a, tgt_c). Compute the net signed count of each
    of the 6 undirected primitive bonds."""
    flux = np.zeros(6, dtype=int)
    for (sa, sc, ta, tc) in cycle:
        dc = tuple(tc[d] - sc[d] for d in range(3))
        result = directed_to_signed_undirected(sa, ta, dc)
        if result is None:
            return None
        idx, sign = result
        flux[idx] += sign
    return flux


def cycle_translation(cycle):
    """Net translation vector of the cycle in the supercell (should be 0
    for a closed cycle in the unbounded lattice)."""
    if not cycle:
        return None
    start_cell = cycle[0][1]
    last_cell = cycle[-1][3]
    return tuple(last_cell[d] - start_cell[d] for d in range(3))


# ── Find girth cycles ─────────────────────────────────────────────────────────

def find_girth_cycles(start_edge, girth=10, max_cycles=20):
    found = []
    path_set = {start_edge}
    def dfs(current, path, depth):
        if len(found) >= max_cycles: return
        if depth == girth:
            for s in nb_successors(*current):
                if s == start_edge:
                    found.append(list(path)); return
            return
        for s in nb_successors(*current):
            if s == start_edge:
                if depth == girth - 1:
                    found.append(list(path))
                continue
            if s in path_set: continue
            path_set.add(s); path.append(s)
            dfs(s, path, depth + 1)
            path.pop(); path_set.discard(s)
    dfs(start_edge, [start_edge], 1)
    return found


if __name__ == '__main__':
    print()
    print("=" * 70)
    print("Girth-cycle homology analysis on H(srs)")
    print("=" * 70)
    print(f"  Primitive cell: 4 atoms, 6 undirected bonds, k={3}, g={g}")
    print(f"  Quotient H_1 rank (E - V + 1) = 6 - 4 + 1 = 3")
    print()

    center = (N_SUPER // 2,) * 3
    flux_vectors = []
    translations = []
    for bond_idx in range(12):
        prim_bond = bonds_prim[bond_idx]
        dc = prim_bond[2]
        tgt_cell = tuple(center[d] + dc[d] for d in range(3))
        if not in_bounds(tgt_cell): continue
        start = (prim_bond[0], center, prim_bond[1], tgt_cell)
        for cyc in find_girth_cycles(start, girth=g, max_cycles=20):
            if len(cyc) != g: continue
            flux = cycle_flux_vector(cyc)
            if flux is not None:
                flux_vectors.append(flux)
                translations.append(cycle_translation(cyc))

    print(f"  Found {len(flux_vectors)} girth cycles. Computing homology...")
    print()

    M = np.array(flux_vectors, dtype=int)
    print(f"  Cycle flux matrix shape: {M.shape} (n_cycles, n_undirected_bonds)")
    print(f"  Rank over Q: {np.linalg.matrix_rank(M)}")
    print()

    # Distinct flux vectors
    flux_set = set(tuple(v) for v in flux_vectors)
    print(f"  Distinct flux vectors: {len(flux_set)}")

    # Translations distribution
    trans_counter = Counter(translations)
    print(f"  Translation vectors of girth cycles (should be (0,0,0) for ALL):")
    for t in sorted(trans_counter.keys()):
        print(f"    translation {t}: {trans_counter[t]} cycles")

    # Sample distinct flux vectors
    print()
    print(f"  First 12 distinct flux vectors (each row = signed counts of 6 undirected bonds):")
    distinct = sorted(flux_set)
    for v in distinct[:12]:
        print(f"    {v}")

    # Rank under closing relations
    # Quotient H_1 has rank 3 in primitive cell with periodic identification.
    # The directed-bond cycle space has rank n_directed_edges - n_vertices + 1
    # but we're working in undirected (6-d) space without periodic identification.
    # For a finite supercell, the quotient relations come from "closing" each
    # vertex's incoming = outgoing. We want H_1 of the QUOTIENT graph (4 atoms,
    # 6 undirected bonds, all bonds wrap around translations).

    # Since we have flux vectors of girth cycles in 6-d space (where each
    # component = signed count of one of the 6 undirected bonds), and we
    # observe that cycles satisfy "in = out" at each vertex (= boundary of
    # 0-chain), the cycle vectors automatically lie in the kernel of the
    # boundary operator ∂: Z^6 → Z^4, which has rank E - V + 1 = 3 (since
    # ∂ has rank V - 1 = 3 due to one redundant relation).

    # So girth cycles SHOULD span a subspace of rank ≤ 3.
    rank = np.linalg.matrix_rank(M)
    print()
    if rank == 3:
        print(f"  ✓ Girth cycles span a rank-3 subspace of the quotient cycle space.")
        print(f"  This is the EXPECTED rank for H_1(srs / Z^3, Z) = Z^3.")
        print(f"  Suggestive isomorphism C^3_obs ≅ H_1(srs / Z^3, R) is consistent")
        print(f"  with this dimensional match.")
    elif rank < 3:
        print(f"  ⚠ Girth cycles span only a rank-{rank} subspace, less than the")
        print(f"  expected rank-3. Some H_1 generators are NOT representable by")
        print(f"  girth cycles alone.")
    else:
        print(f"  ⚠ Girth cycles span a rank-{rank} subspace, exceeding the expected 3.")

    print()
    print("  Translations check: girth cycles in the FULL graph have non-zero")
    print("  translation in general. The quotient H_1 (with periodic identification)")
    print("  is what we want for the generation conjecture.")
