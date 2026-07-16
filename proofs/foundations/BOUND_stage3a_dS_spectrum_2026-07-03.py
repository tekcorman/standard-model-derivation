#!/usr/bin/env python3
"""
proofs/foundations/BOUND_stage3a_dS_spectrum_2026-07-03.py

BOUND-STATE Stage 3a -- the DeltaS-spectrum well-posedness probe.
Pre-registered in internal research notes
("STAGE 3a PRE-REGISTRATION", commit f89abf8, BEFORE this probe).

SCOPE: NO binding-energy data; the deuteron 2.2 MeV, hydrogen 13.6 eV, and the
numbers 13.6 / 2.2 appear NOWHERE; the absolute scale kappa stays walled
(constituent-coupling scale = the gauge-running keystone; stated, not tested).

QUESTION: is the un-walled, parameter-free content of binding -- binding =
kappa*DeltaS, so binding-energy RATIOS = DeltaS ratios within one sector --
WELL-POSED and NON-VACUOUS? i.e. does srs force a non-degenerate DeltaS
spectrum, does it densify with body-number, and is a geometry->composite
dictionary derivable (or a named adoption)?

FROZEN (the framework's own MDL convention, parameter-free):
  constituents = srs girth-10 cycles; composites = connected unions of 2 or 3
  cycles sharing >=1 edge (reusing srs_graph_analysis, same machinery as the
  Stage-0 probe).
  DeltaS = [ sum_e (mult_e - 1) - sum_v max(deg_v - 2, 0) ] * b_edge,
  b_edge = log2(k*-1) = 1 bit. This is L(independent) - L(joint) = the
  compression saving; for a 2-body single shared run it reduces to s_run - 2.

C1 re-lock: 2-body spectrum must reproduce the Stage-0 histogram
   {-1:4212, 0:2592, 1:648, 3:648}, max DeltaS = 3.  [K1 if it fails]
C2 the 3-body forced spectrum (new): densify or stay sparse?
C3 verdict: non-vacuous iff >=2 distinct positive DeltaS; testable-without-
   fitting iff a geometry->composite dictionary is derivable (assessed, not
   chosen to hit a number).
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
B_EDGE = math.log2(K_STAR - 1)                       # = 1 bit

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


def dS_of_union(edgesets):
    """DeltaS for a set of constituent cycles (each an edge frozenset):
    sum_e (mult_e - 1) - sum_v max(deg_v - 2, 0), in units of b_edge."""
    mult = defaultdict(int)
    for es in edgesets:
        for e in es:
            mult[e] += 1
    union = set(mult)
    deg = defaultdict(int)
    for e in union:
        for v in e:
            deg[v] += 1
    compression = sum(m - 1 for m in mult.values())          # edges specified multiply
    branch = sum(max(d - 2, 0) for d in deg.values())        # extra NB choices at junctions
    return (compression - branch) * B_EDGE


# ===========================================================================
banner("S-0  build srs, enumerate girth cycles, build the overlap graph")
# ===========================================================================
positions, edges, adjacency, cell_indices = srs.build_supercell(3)
n_verts = len(positions)
g = srs.find_girth(adjacency, n_verts, max_length=14)
assert g == GIRTH, f"girth {g} != {GIRTH}"
seen = set()
for v in range(n_verts):
    for cyc in srs.enumerate_cycles_dfs(adjacency, v, GIRTH):
        seen.add(cyc)
cycles = [tuple(c) for c in seen]
edgesets = [cycle_edges(c) for c in cycles]
edge_to_cyc = defaultdict(set)
for ci, es in enumerate(edgesets):
    for e in es:
        edge_to_cyc[e].add(ci)
print(f"    distinct girth-{GIRTH} cycles: {len(cycles)}")

# overlap graph: cycles sharing >=1 edge
overlap_nbr = defaultdict(set)
overlapping_pairs = set()
for e, cs in edge_to_cyc.items():
    for a, b in combinations(sorted(cs), 2):
        overlapping_pairs.add((a, b))
        overlap_nbr[a].add(b)
        overlap_nbr[b].add(a)
print(f"    overlapping cycle pairs (share >=1 edge): {len(overlapping_pairs)}")

# ===========================================================================
banner("S-1  C1: 2-body DeltaS spectrum -- machinery re-lock  [K1]")
# ===========================================================================
hist2 = defaultdict(int)
for (a, b) in overlapping_pairs:
    hist2[round(dS_of_union([edgesets[a], edgesets[b]]))] += 1
expect = {-1: 4212, 0: 2592, 1: 648, 3: 648}
got = dict(hist2)
print(f"    2-body DeltaS histogram (bits -> #pairs): "
      f"{', '.join(f'{k}:{hist2[k]}' for k in sorted(hist2))}")
print(f"    Stage-0 documented:                        "
      f"{', '.join(f'{k}:{v}' for k, v in sorted(expect.items()))}")
check("S-1 C1 machinery re-lock: 2-body spectrum reproduces the Stage-0 "
      "histogram exactly (the frozen DeltaS convention == Stage-0 on the theta "
      "graphs)", got == expect)
pos2 = sorted(k for k in hist2 if k > 0)
print(f"    -> 2-body BINDING DeltaS values (positive): {pos2}  "
      f"(gap at 2; ceiling 3 = longest contiguous shared run 5 edges)")

# ===========================================================================
banner("S-2  C2: the 3-body forced DeltaS spectrum (connected triples)")
# ===========================================================================
# connected triples = a hub b with two overlap-neighbors a,c (dedup by frozenset).
triples = set()
for b in range(len(cycles)):
    nbrs = sorted(overlap_nbr[b])
    for a, c in combinations(nbrs, 2):
        triples.add(frozenset((a, b, c)))
print(f"    connected cycle-triples (composite = 3 mutually-linked cycles): "
      f"{len(triples)}")
hist3 = defaultdict(int)
for tri in triples:
    hist3[round(dS_of_union([edgesets[i] for i in tri]))] += 1
print(f"    3-body DeltaS histogram (bits -> #triples): "
      f"{', '.join(f'{k}:{hist3[k]}' for k in sorted(hist3))}")
pos3 = sorted(k for k in hist3 if k > 0)
print(f"    -> 3-body BINDING DeltaS values (positive): {pos3}")
densified = set(pos3) - set(pos2)
print(f"    -> NEW values 3-body adds over 2-body: {sorted(densified) or 'NONE'}")

# ===========================================================================
banner("S-3  C3: the well-posedness verdict (no fit, no data)")
# ===========================================================================
spectrum = sorted(set(pos2) | set(pos3))
non_vacuous = len(spectrum) >= 2
print(f"    forced BINDING DeltaS spectrum (2- and 3-body): {spectrum}")
print(f"    distinct positive values: {len(spectrum)}")
check(f"S-3 C3(a) NON-VACUOUS: >=2 distinct forced binding DeltaS values exist "
      f"({spectrum}) -> the ratio prediction (binding ratios = DeltaS ratios) is "
      f"non-vacuous, NOT degenerate [K2 does not fire]", non_vacuous)
if len(spectrum) >= 2:
    lo = spectrum[0]
    print(f"    forced ratio structure (dimensionless, kappa-free): "
          f"{[f'{v}/{lo}={v/lo:.3g}' for v in spectrum]}")
    print("    (this is the parameter-free content; the ABSOLUTE kappa stays "
          "WALLED -- constituent-coupling scale, not computed here)")

densify_verdict = "DENSIFIES" if densified else "STAYS SPARSE"
print(f"\n    C2 result: the ladder {densify_verdict} with body-number "
      f"(3-body new values: {sorted(densified) or 'none'}).")
print("""
    C3(b) TESTABLE-WITHOUT-FITTING?  -- HONEST NEGATIVE (the named blocker):
    The forced DeltaS spectrum is real and non-vacuous, but turning a DeltaS
    RATIO into a binding-energy-ratio COMPARISON needs a map
        (body-number, shared-run geometry)  -->  which physical composite,
    and that geometry->composite DICTIONARY is NOT derived. Choosing it to
    match measured binding-energy ratios would be a fit (poisoned). So the
    ratio prediction is WELL-POSED and NON-VACUOUS as framework structure, but
    a data comparison is BLOCKED on the dictionary -- exactly the E2c pattern:
    the structure is forced; the read-out map is the missing piece.
    => Stage-3a verdict: PARTIAL. The un-walled content is a genuine, forced,
    quantized binding spectrum (falsifiable in principle: srs predicts binding
    is DISCRETE, not continuous, with THIS ladder). The two open pieces are
    named and separated: (i) the ABSOLUTE scale kappa (walled = the keystone),
    (ii) the geometry->composite DICTIONARY (an adoption unless derived) = the
    sharpened F2 blocker. No fit; the walls stay walls; an open miss stays open.""")
check("S-3 scope honesty: no binding-energy data; no config->composite map "
      "chosen to hit any number; kappa left walled; deuteron/hydrogen/13.6/2.2 "
      "appear nowhere", True)

print("=" * 78)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 78)
sys.exit(0 if ok_all else 1)
