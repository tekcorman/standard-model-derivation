#!/usr/bin/env python3
"""
Enumerate short simple cycles in the srs primitive cell with
periodic boundary conditions.

Strategy:
- Work in the infinite srs graph via (primitive_vertex_idx, cell_vector)
  tuples.
- DFS from each primitive vertex at cell (0,0,0), up to depth = max_length.
- A cycle closes when we return to (start_vertex, (0,0,0)).
- Simple cycle: no repeated intermediate (vertex, cell) pair.
- Canonicalize each cycle (lex-min rotation + optional reversal) and
  deduplicate across DFS starts.

Prints:
- Number of simple cycles of each length from 3 to max_length.
- For each length, the set of inequivalent cycles (up to rotation/reversal).
- For each cycle, the signed edge list in the primitive-cell basis with
  cell displacements. This is the raw material for d_1: C^1 -> C^2.
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from srs_photon_bloch_primitive import (
    build_primitive_unit_cell,
    find_primitive_connectivity,
    canonical_edges_primitive,
)


def build_neighbor_map(bonds):
    """
    Map: for each primitive vertex v, list of (neighbor_v, cell_displacement).
    Note cell_displacement is stored as a tuple of ints (n1, n2, n3) in the
    primitive lattice basis.
    """
    nbrs = [[] for _ in range(4)]
    for (src, tgt, cell, _) in bonds:
        nbrs[src].append((tgt, cell))
    return nbrs


def enumerate_simple_cycles(bonds, max_length):
    """
    Enumerate all simple cycles starting and ending at
    (v=0, cell=(0,0,0)) with length <= max_length.

    Returns: list of (length, path) pairs, where path is a list of
    (vertex, cell) tuples [start, step1, step2, ..., start].
    """
    n_verts = 4
    nbrs = build_neighbor_map(bonds)
    found = []

    def canonicalize_cycle(path):
        """
        Canonical form: the sequence of (vertex, cell - cell_0) tuples
        starting at the lexicographically smallest (vertex, cell) in the
        cycle, with the direction chosen so the second element is
        smaller than the last.

        Since the cycle is closed (path[0] == path[-1]), we work with
        path[:-1] (L distinct positions for an L-cycle).
        """
        L = len(path) - 1
        positions = path[:-1]  # L positions
        # Try all rotations and both directions
        candidates = []
        for start in range(L):
            rot = tuple(positions[(start + i) % L] for i in range(L))
            rev = tuple(rot[::-1])
            # Normalize so cell offsets are relative to the starting position
            for seq in [rot, rev]:
                # Shift cells so start's cell is (0,0,0)
                c0 = seq[0][1]
                shifted = tuple(
                    (v, tuple(c[j] - c0[j] for j in range(3)))
                    for (v, c) in seq
                )
                candidates.append(shifted)
        return min(candidates)

    def dfs(path, visited, depth):
        if depth > max_length:
            return
        current_v, current_cell = path[-1]
        for (next_v, bond_cell) in nbrs[current_v]:
            new_cell = tuple(current_cell[j] + bond_cell[j] for j in range(3))
            new_pos = (next_v, new_cell)
            # Closing condition
            if new_pos == path[0]:
                if depth + 1 >= 3:  # cycle of length >= 3
                    found.append((depth + 1, path + [new_pos]))
                continue
            if new_pos in visited:
                continue
            # Recurse
            visited.add(new_pos)
            path.append(new_pos)
            dfs(path, visited, depth + 1)
            path.pop()
            visited.discard(new_pos)

    # DFS from each primitive vertex starting at cell (0,0,0).
    # This finds all cycles (with duplication across starting vertices);
    # dedupe by canonical form afterwards.
    for start_v in range(n_verts):
        start = (start_v, (0, 0, 0))
        dfs([start], {start}, 0)

    # Deduplicate
    uniq = {}
    for (length, path) in found:
        key = canonicalize_cycle(path)
        if key not in uniq:
            uniq[key] = (length, path)
    return [v for v in uniq.values()]


def main():
    print("=" * 70)
    print("  srs short-cycle enumerator (primitive bcc cell)")
    print("=" * 70)

    verts, lat_vecs = build_primitive_unit_cell()
    bonds = find_primitive_connectivity(verts, lat_vecs)
    edges = canonical_edges_primitive(bonds)

    print(f"\n  {len(bonds)} directed bonds, {len(edges)} undirected edges")

    # Enumerate cycles up to length 12 (just past the expected girth 10)
    max_len = 10
    print(f"\n  Enumerating simple cycles with length up to {max_len}...")
    cycles = enumerate_simple_cycles(bonds, max_len)

    # Group by length
    by_length = {}
    for (length, path) in cycles:
        by_length.setdefault(length, []).append(path)

    print(f"\n  Found {len(cycles)} inequivalent simple cycles.")
    for L in sorted(by_length.keys()):
        print(f"    length {L:2d}: {len(by_length[L])} cycles")

    # Display cycles by length with edge lists
    print("\n  Cycles (showing vertices and cumulative cell displacements):")
    for L in sorted(by_length.keys()):
        print(f"\n  === Length {L} cycles ({len(by_length[L])}) ===")
        for i, path in enumerate(by_length[L]):
            positions_str = " -> ".join(
                f"v{v}@{list(c)}" for (v, c) in path
            )
            # Compute net cell displacement over the cycle
            net = tuple(path[-1][1][j] - path[0][1][j] for j in range(3))
            contractible = net == (0, 0, 0)
            print(f"    cycle #{i+1} ({'contractible' if contractible else 'WRAPS '+str(net)}):")
            print(f"      {positions_str}")


if __name__ == "__main__":
    main()
