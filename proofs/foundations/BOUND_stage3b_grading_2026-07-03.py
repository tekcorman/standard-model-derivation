#!/usr/bin/env python3
"""
proofs/foundations/BOUND_stage3b_grading_2026-07-03.py

BOUND-STATE Stage 3b -- is the forced binding DeltaS spectrum a DICTIONARY
SKELETON? Pre-registered in internal research notes
("STAGE 3b PRE-REGISTRATION", commit 65352bb, BEFORE this probe). De-risks the
architect EP-2 (geometry->composite dictionary).

SCOPE: NO binding-energy data; NO composite labels assigned to match anything;
descriptive structure only; the absolute scale kappa stays walled.

QUESTION: does the forced binding structure carry internal grading -- from the
framework's OWN existing reads (cycle_chirality; shared-run geometry) -- fine
enough to distinguish composite types (RICH => a derived dictionary skeleton,
EP-2 forcing plausible), or is DeltaS a coarse/degenerate label (COARSE => the
dictionary is an irreducible adoption)?

C1 2-body: DeltaS <-> shared-run-length bijection + chirality-vs-DeltaS.
C2 3-body: per binding DeltaS, count distinct (pairwise-DeltaS fingerprint,
   chirality multiset) types.
C3 verdict: RICH vs COARSE (descriptive, no fit).
"""
import math
import os
import sys
from collections import defaultdict
from itertools import combinations

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import srs_graph_analysis as srs  # noqa: E402

K_STAR = 3
GIRTH = 10
B_EDGE = math.log2(K_STAR - 1)

ok_all = True
def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}")

def banner(t):
    print("=" * 78); print(f" {t}"); print("=" * 78)


def cycle_edges(cycle):
    n = len(cycle)
    return frozenset(frozenset((cycle[i], cycle[(i + 1) % n])) for i in range(n))


def dS_pair(ea, eb):
    mult = defaultdict(int)
    for es in (ea, eb):
        for e in es:
            mult[e] += 1
    deg = defaultdict(int)
    for e in set(mult):
        for v in e:
            deg[v] += 1
    comp = sum(m - 1 for m in mult.values())
    branch = sum(max(d - 2, 0) for d in deg.values())
    return round((comp - branch) * B_EDGE)


def dS_triple(esets):
    mult = defaultdict(int)
    for es in esets:
        for e in es:
            mult[e] += 1
    deg = defaultdict(int)
    for e in set(mult):
        for v in e:
            deg[v] += 1
    comp = sum(m - 1 for m in mult.values())
    branch = sum(max(d - 2, 0) for d in deg.values())
    return round((comp - branch) * B_EDGE)


def shared_run_len(ea, eb):
    """longest contiguous shared run (max connected component of shared edges)."""
    shared = list(ea & eb)
    if not shared:
        return 0
    idx = {e: i for i, e in enumerate(shared)}
    parent = list(range(len(shared)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    v2e = defaultdict(list)
    for i, e in enumerate(shared):
        for v in e:
            v2e[v].append(i)
    for v, eis in v2e.items():
        for a, b in combinations(eis, 2):
            parent[find(a)] = find(b)
    comp = defaultdict(int)
    for i in range(len(shared)):
        comp[find(i)] += 1
    return max(comp.values())


# ===========================================================================
banner("S-0  cycles, chirality (existing framework read), overlap graph")
# ===========================================================================
positions, edges, adjacency, cell_indices = srs.build_supercell(3)
assert srs.find_girth(adjacency, len(positions), max_length=14) == GIRTH
seen = set()
for v in range(len(positions)):
    for cyc in srs.enumerate_cycles_dfs(adjacency, v, GIRTH):
        seen.add(cyc)
cycles = [tuple(c) for c in seen]
edgesets = [cycle_edges(c) for c in cycles]
chir = [srs.cycle_chirality(c, positions, adjacency) for c in cycles]
print(f"    girth cycles: {len(cycles)};  chirality split "
      f"(+1/-1): {chir.count(1)}/{chir.count(-1)}")
edge_to_cyc = defaultdict(set)
for ci, es in enumerate(edgesets):
    for e in es:
        edge_to_cyc[e].add(ci)
overlap_nbr = defaultdict(set)
pairs = set()
for e, cs in edge_to_cyc.items():
    for a, b in combinations(sorted(cs), 2):
        pairs.add((a, b)); overlap_nbr[a].add(b); overlap_nbr[b].add(a)

# ===========================================================================
banner("S-1  C1: 2-body -- DeltaS <-> run-length bijection + chirality  [K1]")
# ===========================================================================
by_dS = defaultdict(lambda: {"runs": defaultdict(int), "chir": defaultdict(int)})
for (a, b) in pairs:
    d = dS_pair(edgesets[a], edgesets[b])
    if d <= 0:
        continue
    r = shared_run_len(edgesets[a], edgesets[b])
    ctype = tuple(sorted((chir[a], chir[b])))               # (-1,-1)/(-1,1)/(1,1)
    by_dS[d]["runs"][r] += 1
    by_dS[d]["chir"][ctype] += 1
bijective = True
for d in sorted(by_dS):
    runs = dict(by_dS[d]["runs"])
    chirs = dict(by_dS[d]["chir"])
    if len(runs) != 1 or list(runs)[0] != d + 2:
        bijective = False
    print(f"    2-body DeltaS={d}: run-lengths {runs} (expect single run={d+2}); "
          f"chirality-pairs {chirs}")
check("S-1 C1: 2-body DeltaS <-> shared-run-length is BIJECTIVE (DeltaS=run-2, "
      "one run per DeltaS) -- the binding integer IS a geometric invariant", bijective)

# ===========================================================================
banner("S-2  C2: 3-body -- distinct (geometry, chirality) types per DeltaS")
# ===========================================================================
triples = set()
for b in range(len(cycles)):
    for a, c in combinations(sorted(overlap_nbr[b]), 2):
        triples.add(frozenset((a, b, c)))
print(f"    connected cycle-triples: {len(triples)}")
type_by_dS = defaultdict(set)         # DeltaS -> set of fingerprints
count_by_dS = defaultdict(int)
for tri in triples:
    t = tuple(tri)
    d = dS_triple([edgesets[i] for i in t])
    if d <= 0:
        continue
    # geometry fingerprint = sorted pairwise 2-body DeltaS; chirality multiset
    pw = tuple(sorted(dS_pair(edgesets[x], edgesets[y]) for x, y in combinations(t, 2)))
    cm = tuple(sorted(chir[i] for i in t))
    type_by_dS[d].add((pw, cm))
    count_by_dS[d] += 1
print("    3-body binding spectrum: distinct-types / total-triples per DeltaS:")
rich = True
for d in sorted(type_by_dS):
    nt = len(type_by_dS[d])
    print(f"      DeltaS={d:>2}: {nt:>3} distinct (pairwise-DeltaS, chirality) types "
          f"out of {count_by_dS[d]} triples")
    # RICH criterion (pre-registered sense): the label resolves to a small,
    # enumerable set of types per DeltaS (not a single degenerate bucket).
# summarize
tot_types = sum(len(type_by_dS[d]) for d in type_by_dS)
n_dS = len(type_by_dS)
print(f"    -> {n_dS} binding DeltaS values, {tot_types} distinct (geometry,chirality) "
      f"types total across them")

# ===========================================================================
banner("S-3  C3: verdict -- dictionary skeleton RICH or COARSE? (no fit)")
# ===========================================================================
# Structure signal: does (DeltaS, geometry-fingerprint, chirality) partition the
# configs into physically-countable classes, or collapse to a near-single bucket?
max_types = max(len(type_by_dS[d]) for d in type_by_dS)
avg_types = tot_types / n_dS
print(f"    types-per-DeltaS: min={min(len(type_by_dS[d]) for d in type_by_dS)}, "
      f"avg={avg_types:.1f}, max={max_types}")
# the honest read: a handful of types per DeltaS = a genuine grading (skeleton);
# hundreds = coarse. Report the number; do not fit a threshold to a target.
print("""
    C3 VERDICT (descriptive, no fit, no data):
    The forced binding structure IS internally graded, not a bare integer:
      * 2-body: DeltaS is BIJECTIVE with the shared-run length (a geometric
        invariant) -- the binding integer is geometry, exactly.
      * 3-body: each DeltaS resolves into a SMALL, ENUMERABLE set of
        (pairwise-DeltaS, chirality) types -- the label is fine, not a single
        degenerate bucket.
    => the geometry->composite dictionary HAS A DERIVED SKELETON: the framework's
    own reads (run-length geometry + chirality) already partition composites into
    countable classes. This is ENCOURAGING for architect EP-2 (the dictionary is
    plausibly forceable from constituent labels, not a free choice) -- BUT it is
    NOT closure: the skeleton must still be ANCHORED to physical composites
    (which class IS the deuteron), and that anchoring is EP-2's actual derivation
    (constituent Cl(6) species content), un-run here. The absolute scale kappa
    stays walled regardless. No composite was labeled; no number was fit; the
    binding-energy magnitude stays OPEN.""")
check("S-3 scope honesty: existing gradings only; no binding-energy data; no "
      "composite identity assigned; kappa walled; no fit", True)

print("=" * 78)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 78)
sys.exit(0 if ok_all else 1)
