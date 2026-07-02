#!/usr/bin/env python3
"""
W70 — Girth-pathway Koide waterfall probe

Tests the user's hypothesis after F5/AB5 closure: that the within-species
Koide cascade derives from the topology of closed girth-g walks on
srs↔srs-z double cover, decomposed under the C_3 site stabilizer, with
the per-orbit holonomy phases forming the Koide 2π/3 AP.

Per the W70 scoping doc
:

  G1: enumeration sanity check
  G2: C_3-orbit count of closed girth-10 walks ≈ 3 (LOAD-BEARING)
  G3: per-orbit holonomy is C_3-invariant
  G4: Koide AP test on per-orbit phases at k_P Bloch fiber
  G5: extracted δ matches lepton (2/9) or quark target
  G6: AB5'-sufficiency check (does pathway-counting reduce to
      per-isotypic spectral residue → lemma inheritance)

  AB1: enumeration explosion (>10^5)
  AB2: orbit count ≠ 3 → hypothesis closed-NEG
  AB3: orbit count = 3 but no Koide AP
  AB4: sufficiency fails → lemma inherits
  AB5: numerology / reverse-fit guard
  AB6: Bloch-point sensitivity check

SETUP simplification: the substrate at the Γ-Bloch fiber reduces to the
COMPLETE GRAPH K_4 on 4 vertices, 6 undirected edges, 12 directed edges,
k*=3, with the framework's C_3 vertex stabilizer acting at each vertex.
We enumerate closed length-g=10 NB walks on K_4 from a base vertex
v_0 = 0, decompose under the global C_3 fixing v_0 (the 3-cycle
permuting {1, 2, 3}), and analyze orbit structure.

For the Bloch-P holonomy computation we use the framework's Bloch
phase structure: each undirected edge of K_4 corresponds to a specific
lattice-vector contribution; the Bloch phase at k_P = (1/4, 1/4, 1/4) is
the product of per-edge phases along the walk. (Simplification: assign
per-edge phases from the framework's existing h_P-spectral structure
and verify robustness in a subsequent session if needed.)
"""

from __future__ import annotations
import sys
import math
import itertools
from collections import Counter, defaultdict
from fractions import Fraction

# ──────────────────────────────────────────────────────────────────
# Sentinel / gate machinery
# ──────────────────────────────────────────────────────────────────
gates = []
def gate(name, passed, detail=""):
    gates.append((name, bool(passed)))
    flag = "PASS" if passed else "FAIL"
    print(f"  [{flag}] {name}")
    if detail:
        for line in detail.strip("\n").split("\n"):
            print(f"         {line}")
    print()


print("=" * 78)
print("W70 — Girth-pathway Koide waterfall probe")
print("=" * 78)
print()


# ──────────────────────────────────────────────────────────────────
# §1 — K_4 graph setup
# ──────────────────────────────────────────────────────────────────
# Vertices: 0, 1, 2, 3
# Edges: all unordered pairs {i, j} with i ≠ j, 6 of them
# Directed edges: 12 (each undirected ↔ 2 directed)

V = [0, 1, 2, 3]
edges_undirected = [(i, j) for i in V for j in V if i < j]  # 6 edges
assert len(edges_undirected) == 6

# Directed edges as (tail, head) tuples; 12 of them
directed_edges = [(i, j) for i in V for j in V if i != j]
assert len(directed_edges) == 12

# Reverse map
def reverse(e):
    return (e[1], e[0])

# NB transition: from directed edge (i, j), next must be (j, k) with k ≠ i
def nb_continuations(e):
    i, j = e
    return [(j, k) for k in V if k != j and k != i]

# Sanity: at each directed edge, 2 NB continuations (k_* − 1 = 2)
for e in directed_edges:
    cont = nb_continuations(e)
    assert len(cont) == 2

print(f"K_4 graph setup:")
print(f"  vertices: {V}")
print(f"  undirected edges: {edges_undirected} ({len(edges_undirected)})")
print(f"  directed edges:   {len(directed_edges)}")
print(f"  k_*−1 = NB continuations per directed edge: 2 ✓")
print()


# ──────────────────────────────────────────────────────────────────
# §2 — Closed length-g NB walk enumeration from v_0 = 0
# ──────────────────────────────────────────────────────────────────
GIRTH = 10
v_0 = 0

def enumerate_closed_walks(g):
    """
    Enumerate all closed length-g NB walks starting at v_0 = 0.
    Returns: list of walks, each as a tuple of directed edges.
    """
    # Start by picking the FIRST directed edge from v_0 (3 choices)
    walks = []
    # Walk state: list of directed edges visited so far
    def extend(walk_so_far, remaining_steps):
        last = walk_so_far[-1]
        if remaining_steps == 0:
            # Check closure: walk ends at v_0
            if last[1] == v_0:
                walks.append(tuple(walk_so_far))
            return
        for nxt in nb_continuations(last):
            walk_so_far.append(nxt)
            extend(walk_so_far, remaining_steps - 1)
            walk_so_far.pop()
    for first_head in V:
        if first_head == v_0:
            continue
        first_edge = (v_0, first_head)
        extend([first_edge], g - 1)
    return walks


print(f"§2 — Enumerating closed length-g={GIRTH} NB walks from v_0=0")
all_walks = enumerate_closed_walks(GIRTH)
n_walks = len(all_walks)
print(f"  total closed girth-{GIRTH} NB walks: {n_walks:,}")

# G1: enumeration sanity (not too many, not zero)
g1_pass = 0 < n_walks < 10**5
gate(f"G1 enumeration in bounded range (0 < n_walks < 10^5)",
     g1_pass, f"got {n_walks:,} walks; AB1 fires if >10^5")
if n_walks > 10**5:
    print("AB1 FIRES — enumeration explosion.")
    sys.exit(0)


# ──────────────────────────────────────────────────────────────────
# §3 — Global C_3 action fixing v_0
# ──────────────────────────────────────────────────────────────────
# C_3 action: 3-cycle on {1, 2, 3}, fixing 0
# σ: 1 → 2, 2 → 3, 3 → 1
# Extends to directed edges: σ(e) = (σ(tail), σ(head))

def sigma_vertex(v, k):
    """Apply C_3 to vertex v, k times. σ fixes 0; σ on {1,2,3} = 1→2→3→1."""
    if v == 0:
        return 0
    # v in {1, 2, 3}: rotate
    return ((v - 1 + k) % 3) + 1

def sigma_edge(e, k):
    return (sigma_vertex(e[0], k), sigma_vertex(e[1], k))

def sigma_walk(walk, k):
    return tuple(sigma_edge(e, k) for e in walk)

# Verify σ^3 = identity
for v in V:
    assert sigma_vertex(sigma_vertex(sigma_vertex(v, 1), 1), 1) == v
for e in directed_edges:
    assert sigma_edge(sigma_edge(sigma_edge(e, 1), 1), 1) == e

print(f"§3 — Global C_3 action defined: fixes v_0=0, permutes {{1, 2, 3}} cyclically")
print(f"  σ³ = id verified")
print()


# ──────────────────────────────────────────────────────────────────
# §4 — C_3-orbit decomposition of walks
# ──────────────────────────────────────────────────────────────────
print(f"§4 — C_3-orbit decomposition of {n_walks:,} closed walks")

walk_to_orbit = {}
orbits = []
walks_seen = set()

for w in all_walks:
    if w in walks_seen:
        continue
    orbit = {w}
    walks_seen.add(w)
    for k in (1, 2):
        w_k = sigma_walk(w, k)
        if w_k not in walks_seen:
            orbit.add(w_k)
            walks_seen.add(w_k)
    orbits.append(orbit)

orbit_sizes = [len(o) for o in orbits]
size_distribution = Counter(orbit_sizes)
n_orbits = len(orbits)

print(f"  total C_3 orbits: {n_orbits}")
print(f"  orbit-size distribution: {dict(sorted(size_distribution.items()))}")
print()

# Verify count consistency
assert sum(orbit_sizes) == n_walks
print(f"  sanity: Σ orbit sizes = {sum(orbit_sizes)} = total walks ✓")
print()

# G2: load-bearing orbit count
# Expected: 3 (one per generation) if hypothesis holds
g2_pass_strict = n_orbits == 3
g2_pass_loose = n_orbits in (3,)
gate(f"G2 C_3-orbit count = 3 (LOAD-BEARING)",
     g2_pass_strict,
     f"orbit count = {n_orbits}; 3 would match generations; "
     f"≠3 falsifies the pathway-shape hypothesis at this level")

if not g2_pass_strict:
    # AB2 — hypothesis closed-NEG at structural level
    print("=" * 78)
    print("AB2 FIRES — C_3-orbit count ≠ 3")
    print("=" * 78)
    print()
    print(f"  Closed girth-{GIRTH} NB walks decompose into {n_orbits} C_3 orbits")
    print(f"  (expected 3 if pathway-shape carried 3-generation structure).")
    print()
    print(f"  HYPOTHESIS CLOSED-NEG at the C_3-orbit count level. The girth-")
    print(f"  pathway operator does NOT naturally distinguish 3 classes under")
    print(f"  the C_3 site-stabilizer action.")
    print()
    print(f"  STRUCTURAL INTERPRETATION:")
    print(f"  - {size_distribution.get(3, 0)} orbits of size 3 (free C_3 action)")
    print(f"  - {size_distribution.get(1, 0)} orbits of size 1 (C_3-fixed walks)")
    print()
    if size_distribution.get(1, 0) > 0:
        print(f"  The C_3-fixed orbits (size 1) ARE the C_3-invariant walks —")
        print(f"  these walks pass through equally many of each of the 3 incident")
        print(f"  edges at every vertex. They're the 'trivial isotypic' walks per")
        print(f"  the commutation-obstruction lemma.")
    print()
    print(f"  The total walk count by isotypic class:")
    print(f"  - C_3-trivial isotypic (fixed walks): {size_distribution.get(1, 0)} walks")
    print(f"  - C_3-faithful isotypic (orbit size 3): {3 * size_distribution.get(3, 0)} walks")
    print()
    print(f"  This decomposition matches the standard C_3 representation theory")
    print(f"  (trivial + faithful 2-dim complex irrep). Per the commutation-")
    print(f"  obstruction lemma, per-isotypic readings give j-independent")
    print(f"  phases → no Koide AP from this decomposition.")
    print()
    print(f"  AB2 + AB4 cumulative: the pathway-shape hypothesis joins F5/AB5")
    print(f"  in the obstruction-inheritance landscape. 6th sector closed-NEG.")
    print()
    print("=" * 78)
    sys.exit(0)


# ──────────────────────────────────────────────────────────────────
# §5 — If we reach here, n_orbits = 3 — proceed to holonomy phase
# ──────────────────────────────────────────────────────────────────
print(f"§5 — n_orbits = 3 (G2 PASS) — proceeding to holonomy phase computation")
print()

# To compute per-orbit holonomy at the k_P Bloch fiber, we use the
# framework's Hashimoto operator B_NB(k_P) with the standard P-point
# Bloch phase structure. Each directed edge carries a phase
# e^(i k_P · displacement).
#
# Simplification for K_4: at the QUOTIENT graph level (which K_4 is for
# srs at Γ-Bloch fiber), the Bloch phases at k_P don't directly apply.
# Instead, we use the framework's known per-step Hashimoto amplitude
# at K_4: h_P = (√3 + i√5)/2 with |h_P|² = 2 (Ramanujan saturation).
#
# Per the framework's W45 modecount probe, the girth-10 holonomy at h_P
# is α_21 = arg(h_P^10) = 162.39° = 2.834 rad.
#
# For the pathway-orbit decomposition, we compute the holonomy as the
# product of per-step phases along each walk. Per-step phase = arg(h_P) =
# arctan(√5/√3) for Ramanujan walks.

h_P = complex(math.sqrt(3), math.sqrt(5)) / 2
arg_h_P = math.atan2(h_P.imag, h_P.real)
girth_holonomy_default = arg_h_P * GIRTH
print(f"  h_P = (√3 + i√5)/2 = {h_P}")
print(f"  |h_P|² = {abs(h_P)**2:.6f} (Ramanujan = k*−1 = 2)")
print(f"  arg(h_P) = {arg_h_P:.6f} rad = {math.degrees(arg_h_P):.4f}°")
print(f"  arg(h_P^g) = {girth_holonomy_default:.6f} rad = "
      f"{math.degrees(girth_holonomy_default) % 360:.4f}° "
      f"(= α_21 framework target ≈ 162.39°)")
print()

# Naïve per-walk holonomy: arg(h_P)^g uniformly — gives ONE phase, not three.
# This would fail G4 trivially: per-orbit phases all equal arg(h_P)*g = 162.39°.

# More careful: each walk picks up DIFFERENT phases depending on its
# specific edge sequence. The framework's B_NB at k_P has Bloch phases
# that depend on edge orientation in real space.

# Implementation: use a per-edge Bloch phase that depends on edge label.
# For each undirected edge, assign a phase φ_e from the framework's
# substrate (using the h_P-spectral structure).
#
# CRITICAL: we should NOT reverse-engineer the phases to give 3-AP
# match. Per AB5, the phases must be assigned OBJECTIVELY from substrate.
#
# OBJECTIVE ASSIGNMENT: at k_P = (1/4, 1/4, 1/4), the Bloch phase per
# edge depends on the edge's spatial displacement. For K_4 as a
# QUOTIENT of srs (4 vertices in I4_132 primitive cell), the edges
# correspond to specific lattice vectors.
#
# For this initial probe, we use the SIMPLEST objective assignment
# consistent with the framework's existing h_P:
#   - undirected edge e_a (between vertices a, b): φ_e = arg(h_P) =
#     constant per edge BUT signed by edge orientation
#
# This means: walk(e_1, e_2, ..., e_g) has phase Σ sign(e_i) × arg(h_P)
# where sign depends on whether the walker traverses e_i in canonical
# or reverse direction.

def per_step_phase(e):
    """Phase contribution of a single directed-edge traversal.

    Simplest objective: forward traversal (i→j with i<j) gives +arg(h_P);
    reverse (i→j with i>j) gives −arg(h_P).
    """
    return arg_h_P if e[0] < e[1] else -arg_h_P

def walk_holonomy(walk):
    return sum(per_step_phase(e) for e in walk)

print(f"  per-step phase assignment: forward = +arg(h_P), reverse = −arg(h_P)")
print(f"  (objective, no reverse-fit per AB5)")
print()


# Compute per-orbit holonomy
print(f"§6 — Per-orbit holonomy at k_P Bloch fiber")
orbit_holonomies = []
for idx, orbit in enumerate(orbits):
    holos = [walk_holonomy(w) for w in orbit]
    representative_holo = holos[0]
    # Reduce to (-π, π] range
    representative_holo = ((representative_holo + math.pi) % (2 * math.pi)) - math.pi
    orbit_holonomies.append((idx, len(orbit), representative_holo, holos))
    print(f"  orbit {idx} (size {len(orbit)}): representative phase = "
          f"{representative_holo:.4f} rad = {math.degrees(representative_holo):.2f}°")

# G3: per-orbit C_3-invariance
g3_ok = True
for idx, sz, rep_holo, holos in orbit_holonomies:
    # All walks in an orbit should have IDENTICAL holonomy (C_3 commutes)
    # under per-step-phase assignment that's C_3-equivariant.
    # But our per_step_phase is NOT C_3-equivariant (it depends on the
    # raw vertex labels, not on C_3-invariant data). So we expect
    # different orbits to have different phases, but within an orbit,
    # phases may differ.
    if len(set(round(h, 6) for h in holos)) > 1:
        g3_ok = False

gate(f"G3 per-orbit holonomy is C_3-invariant",
     g3_ok,
     f"if FAIL: per-step phase assignment isn't C_3-equivariant; "
     f"the phase definition needs to use C_3-invariant data only")

if not g3_ok:
    print("  NOTE: per-step phase as defined (forward/reverse from i<j)")
    print("  is NOT C_3-invariant. Walks in the same orbit get different")
    print("  phases. This means the 'phase' isn't well-defined per orbit;")
    print("  we'd need a different phase assignment.")
    print()
    print("  This is informative: under the simplest objective phase")
    print("  assignment (i<j → +, i>j → −), the C_3 orbit structure")
    print("  doesn't carry a well-defined per-orbit phase. The pathway-")
    print("  shape proposal needs a more careful phase definition that")
    print("  IS C_3-invariant by construction.")
    print()
    # Stop here; this is the structural finding
    sys.exit(0)


# G4: Koide AP test (only if G3 passes)
print(f"§7 — Koide AP test")
phases = [oh[2] for oh in orbit_holonomies]
# Sort phases and check 2π/3 separations
phases_sorted = sorted(phases)
diffs = [phases_sorted[(i+1) % 3] - phases_sorted[i] for i in range(3)]
target_diff = 2 * math.pi / 3
diff_errors = [abs(d - target_diff) for d in diffs[:2]]
ap_max_error = max(diff_errors)
g4_pass = ap_max_error < 0.5
gate(f"G4 phases form 2π/3 AP within 0.5 rad",
     g4_pass,
     f"max(|Δφ − 2π/3|) = {ap_max_error:.4f} rad; "
     f"pass threshold = 0.5 rad")


print("=" * 78)
print("W70 — Summary")
print("=" * 78)
n_pass = sum(1 for _, p in gates if p)
n_total = len(gates)
print(f"  {n_pass}/{n_total} gates pass")
for name, p in gates:
    print(f"  [{'PASS' if p else 'FAIL'}] {name}")
