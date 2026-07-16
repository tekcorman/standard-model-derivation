#!/usr/bin/env python3
# ============================================================================
# T0-N-3 — THE DOMAIN THEOREM (coincidence exclusion by statistics)
# ============================================================================
#
# PRE-REGISTRATION (binding, frozen, architect architect 2026-07-11):
#   internal research notes
#
# LINEAGE: T0-NUCLEAR (internal research notes) -> T0-NUCLEAR-2
#   amendment (887152d) -> verdict KIN-WRONG-WAY
#   (internal research notes).  That station's
#   ground state collapsed onto an EXACT self-translation coincidence
#   (mult_e=3 at zero relative displacement, giving raw DeltaS=20, deeper than the
#   certified ground-triple rung DeltaS=13); the 2-body station separately removed
#   the analogous coincidence (raw DeltaS=10) by a DISCLOSED, UNDERIVED cap
#   (IV4_T0_class_2026-07-10.py, DS_CAP=3.0).  THIS station asks: does the object
#   itself (not a disclosed choice) force the coincidence configuration OUT of the
#   interaction domain?  Candidate forcing: exchange statistics (fermion parity /
#   the net's twisted (Klein) locality).
#
# MACHINERY REPLICATED (not edited; same objects, cited by file:line), from:
#   proofs/foundations/srs_graph_analysis.py
#     - build_supercell(58), find_girth(138), enumerate_cycles_dfs(217): the
#       frozen supercell(3) girth-10 cycle enumeration (216 vertices, girth 10,
#       324 cycles) -- IDENTICAL call sequence to T0_NUCLEAR2_2026-07-10.py:600-608.
#   proofs/foundations/BOUND_stage3a_dS_spectrum_2026-07-03.py:64-78 (dS_of_union,
#     VERBATIM, replicated here as dS_union_parts) / proofs/foundations/
#     T0_NUCLEAR2_2026-07-10.py:275-294 (its own verbatim replication) -- THE DEPTH
#     FUNCTION: DeltaS = [Sum_e(mult_e-1) - Sum_v max(deg_v-2,0)] * b_edge,
#     b_edge = log2(k*-1) = 1 bit -- and the pair/triple overlap-search
#     (T0_NUCLEAR2_2026-07-10.py:612-667) that identifies the certified rungs:
#     648 ground pairs (DeltaS=3), 216 ground triples (DeltaS=13), re-locked
#     EXACTLY against the frozen histograms below (T0_NUCLEAR2_2026-07-10.py:172-174).
#   derivation_topdown/state/the_net.py (ACCRETED master object, imported, NEVER
#     edited, NEVER rebuilt):
#     - gauge_sector_category() [:656-717] -- the Cl(6) field-algebra / DHR sector
#       category: species_sector_dims {0:1(nu),1:3(d),2:3(u),3:1(e)}, fermion_parity
#       = {0:+1,1:-1,2:+1,3:-1} = (-1)^w on the 3-fundamental-fermionic-mode Fock
#       count w=0..3 (C(3,w) = 1,3,3,1).
#     - twisted_locality_holds() [:311-331] -- explicit finite (Jordan-Wigner) Fock
#       demonstration that Klein-twisted ODD operators genuinely ANTICOMMUTE (CAR),
#       naive commutation FAILS, even operators commute.
#     - z2_gradings() [:1663-1685] (MS-1a, accreted 2026-07-10) -- R(A4) admits ONLY
#       the trivial Z2 grading (count 1); R(2T) admits EXACTLY one nontrivial
#       grading (count 2), which EQUALS the fermion-parity/center grading.  Fermion
#       parity is therefore the ONE AND ONLY nontrivial Z2 sector grading this
#       category supports.
#     - sector_grading_hist() [:2077-2141] -- "the Sec.8 grading" (FOCK-0,
#       2026-07-11, internal research notes): grades
#       the I2b word-Fock space H_hist (where darts/cycles actually live, via
#       build_hist/build_S [:1861-1902], the walk<->Fock dictionary LOOP-E1 cites)
#       by its A4-isotypic (species) decomposition; ALSO carries a word-length
#       parity (-1)^n, checked (not derived) to COMMUTE with the species grading
#       (parity_commute_residual, asserted < 1e-7).
#   FOCK-0's OWN accreted verdict (internal research notes
#     Sec.5 item 1, INDEPENDENTLY CONFIRMED by its adversarial checker
#     internal research notes Sec.6 defect #2) is the
#     load-bearing fact for THIS station's statistics step (Sec. 3 below): word-length
#     parity as H_hist's fermion-parity bit is EXTRINSIC (compatible, NOT forced) --
#     z2_gradings proves R(A4) alone carries no nontrivial grading, so nothing in
#     the accreted category FORCES darts into the odd/Fermi slot.  FOCK-0's further
#     attempt to build a FORCED equivariant bridge Hom_A4(dart_rep, field-algebra F)
#     found it to be exactly {0} (no such natural map); its own checker flags this
#     specific fact as a generic, unsurprising algebraic consequence (regular vs.
#     projective representation types), not a lattice-specific theorem, and flags
#     that the driver's promotion of it to a settled "V3 obstruction" oversteps the
#     (still architect-pending) evidence.  This station uses BOTH sides of that
#     honestly, per the poisons: no re-interpretation, no patching.
#
# ---------------------------------------------------------------------------
# FROZEN A PRIORI (pre-reg SS2, executed in order; nothing below was chosen in
# response to a computed number):
#   STEP 1: reproduce the frozen supercell(3) cycle enumeration, RE-LOCK the two
#     histograms EXACTLY (stop-clause: any mismatch is a graph/cycle-enumeration
#     regression, not a license to reweight); lift each cycle to its TWO dart
#     orientations (forward/backward reading of the vertex sequence -- a clean,
#     unambiguous 2-fold lift; NOT branch D4).
#   STEP 2: the dart-occupation audit -- for every ground pair (648) / ground
#     triple (216): does an orientation assignment (one per cycle, free per DISTINCT
#     cycle) exist with no dart occupied twice?  Full counts, both directions.  Same
#     audit for exact coincidence (same cycle counted with multiplicity 2 / 3 at
#     zero relative displacement) -- but coincidence is NOT "two independently
#     orientable slots": the self-translation ansatz applies the IDENTICAL
#     translation (an orientation/chirality-preserving isometry of the net) to the
#     SAME canonical cycle, so every copy inherits the SAME transported orientation,
#     not a free per-copy choice.  Both readings are printed for transparency.
#   STEP 3: connect dart modes to the fermionic grading using ONLY gauge_sector_
#     category / twisted_locality_holds / z2_gradings / sector_grading_hist (the
#     Sec.8 grading) -- state exactly which premise carries CAR and its
#     conditionality (ML-2b, plus the FOCK-0 bridge-status finding above).
#   STEP 4: assemble the theorem (algebraic proof + a machine demonstration on a
#     small truncation: an explicit two-copy occupation state built from REAL data,
#     shown to be the zero vector under CAR; a real ground-pair state under the
#     found orientation, shown nonzero under CAR).
#   STEP 5: a CONSEQUENCE NOTE only (no solve) on the 2-body DS_CAP and the 3-body
#     re-solve, naming T0-N-4.  NO binding solve, no kappa, no B2/B3, no
#     confrontation runs anywhere in this file.
#
# FROZEN VERDICT TREE (pre-reg SS3; dual-outcome; the ML-2b qualifier attaches to
# EVERY sentence below that touches the sector category):
#   D1 EXCLUSION-DERIVED / D2 NO-EXCLUSION / D3 CONTRADICTION / D4 BLOCKED.
#
# POISONS (pre-reg SS4, carried verbatim): no cap introduced anywhere; no
# re-weighting; rungs 3/13 and both frozen histograms imported/re-derived and
# RE-LOCKED before use; the orientation lift is derived or D4, never invented;
# MS-1a/ML-2 facts via the accreted the_net.py APIs only (never rebuilt); every
# citation opened and verified by reading the file; an open miss stays open (D2/D3
# are complete, bookable outcomes, not failures to patch).
#
# Standalone: python3 proofs/foundations/T0N3_domain_theorem_2026-07-11.py ; exit 0.
# Asserts fire ONLY on machine-checkable regressions (histogram re-locks, CAR
# algebra identities); the STATION VERDICT is always PRINTED, never asserted.
# ============================================================================

import math
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import srs_graph_analysis as srs                                    # noqa: E402
from derivation_topdown.state import the_net                        # noqa: E402

W = 82
ok_all = True


def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}")


def banner(t):
    print("=" * W)
    print(f" {t}")
    print("=" * W)


# ===========================================================================
# BLOCK 1 -- frozen supercell(3) enumeration + depth function (replicated
# verbatim from BOUND_stage3a_2026-07-03.py:64-78 / T0_NUCLEAR2_2026-07-10.py
# Block 2 + Step-1 gate, cited above)
# ===========================================================================
GIRTH = 10
K_STAR = 3
B_EDGE = math.log2(K_STAR - 1)        # = 1 bit
DS2_RUNG = 3
DS3_RUNG = 13
LADDER2 = [1, 3]
LADDER3 = [1, 2, 3, 4, 6, 13]
HIST2_STAGE0 = {-1: 4212, 0: 2592, 1: 648, 3: 648}
HIST3_20260703 = {-3: 108, -2: 76464, -1: 82512, 0: 54432,
                  1: 20736, 2: 19224, 3: 16848, 4: 3888, 6: 2592, 13: 216}


def cycle_edges(cycle):
    n = len(cycle)
    return frozenset(frozenset((cycle[i], cycle[(i + 1) % n])) for i in range(n))


def dS_union_parts(edgesets):
    mult = defaultdict(int)
    for es in edgesets:
        for e in es:
            mult[e] += 1
    deg = defaultdict(int)
    for e in mult:
        for v in e:
            deg[v] += 1
    compression = sum(m - 1 for m in mult.values())
    branch = sum(max(d - 2, 0) for d in deg.values())
    return (compression - branch) * B_EDGE, compression, branch


def build_reference_configs():
    """STEP 1: reproduce the frozen enumeration; RE-LOCK both histograms EXACTLY
    (asserted); return (cycles, edgesets, ground_pairs, ground_triples)."""
    positions, edges, adjacency, cell_indices = srs.build_supercell(3)
    n_verts = len(positions)
    g = srs.find_girth(adjacency, n_verts, max_length=14)
    assert g == GIRTH, f"girth {g} != {GIRTH}"
    seen = set()
    for v in range(n_verts):
        for cyc in srs.enumerate_cycles_dfs(adjacency, v, GIRTH):
            seen.add(cyc)
    cycles = [tuple(c) for c in seen]
    print(f"    supercell(3): {n_verts} vertices, girth {g}, {len(cycles)} girth-{GIRTH} cycles")
    edgesets = [cycle_edges(c) for c in cycles]

    edge_to_cyc = defaultdict(set)
    for ci, es in enumerate(edgesets):
        for e in es:
            edge_to_cyc[e].add(ci)
    overlap_nbr = defaultdict(set)
    pairs = set()
    for e, cs in edge_to_cyc.items():
        for a, b in combinations(sorted(cs), 2):
            pairs.add((a, b))
            overlap_nbr[a].add(b)
            overlap_nbr[b].add(a)

    hist2 = Counter()
    ground_pairs = []
    for a, b in pairs:
        dS, comp, br = dS_union_parts([edgesets[a], edgesets[b]])
        hist2[round(dS)] += 1
        if round(dS) == DS2_RUNG:
            ground_pairs.append((a, b))
    assert dict(hist2) == HIST2_STAGE0, f"2-body histogram RE-LOCK FAILED: {dict(hist2)}"
    pos2 = sorted(k for k in hist2 if k > 0)
    assert pos2 == LADDER2, f"2-body spectrum {pos2} != {LADDER2}"
    check("RE-LOCK: 2-body histogram == frozen Stage-0 {-1:4212,0:2592,1:648,3:648}",
          dict(hist2) == HIST2_STAGE0)
    check(f"all {len(ground_pairs)} ground pairs (DeltaS=3) identified", len(ground_pairs) == 648)

    triples = set()
    for b in range(len(cycles)):
        nbrs = sorted(overlap_nbr[b])
        for a, c in combinations(nbrs, 2):
            triples.add(frozenset((a, b, c)))
    hist3 = Counter()
    ground_triples = []
    for tri in triples:
        i, j, k = sorted(tri)
        dS, comp, br = dS_union_parts([edgesets[i], edgesets[j], edgesets[k]])
        hist3[round(dS)] += 1
        if round(dS) == DS3_RUNG:
            ground_triples.append((i, j, k))
    assert dict(hist3) == HIST3_20260703, f"3-body histogram RE-LOCK FAILED: {dict(hist3)}"
    pos3 = sorted(k for k in hist3 if k > 0)
    assert pos3 == LADDER3, f"3-body spectrum {pos3} != {LADDER3}"
    check("RE-LOCK: 3-body histogram == frozen 2026-07-03 run (ladder "
          f"{LADDER3}, #(dS=13)=216)", dict(hist3) == HIST3_20260703)
    check(f"all {len(ground_triples)} ground triples (DeltaS=13) identified",
          len(ground_triples) == 216)

    return cycles, edgesets, ground_pairs, ground_triples


# ===========================================================================
# BLOCK 2 -- THE ORIENTATION LIFT + DART-OCCUPATION AUDIT
# Each undirected girth-10 cycle (a vertex sequence, canonicalized by
# enumerate_cycles_dfs's own min-over-all-rotations-and-both-directions rule) has
# exactly TWO dart (directed-edge) readings: forward (consecutive pairs of the
# tuple) and backward (consecutive pairs of the reversed tuple) -- a clean 2-fold
# lift, unambiguous, NOT branch D4.  Two cycles sharing an undirected edge collide
# (same-dart double occupation) iff BOTH traverse that edge in the SAME direction;
# opposite directions give two DISTINCT darts (different modes) -- no collision.
# ===========================================================================
def dart_set(cycle, reverse=False):
    seq = tuple(reversed(cycle)) if reverse else cycle
    n = len(seq)
    return frozenset((seq[i], seq[(i + 1) % n]) for i in range(n))


def pair_compatible(cyc_a, cyc_b):
    """Fix cyc_a's orientation (forward); try both orientations of cyc_b (free,
    independent choice -- cyc_a and cyc_b are DISTINCT combinatorial cycles, no
    shared-translation constraint links their orientations).  Returns
    (compatible, same_works, opp_works)."""
    A_F = dart_set(cyc_a, False)
    B_F = dart_set(cyc_b, False)
    B_R = dart_set(cyc_b, True)
    same_works = not (A_F & B_F)
    opp_works = not (A_F & B_R)
    return (same_works or opp_works), same_works, opp_works


def triple_compatible(cyc_a, cyc_b, cyc_c):
    """Fix cyc_a forward; try all 4 combos of (cyc_b, cyc_c) orientations (each
    free/independent); a combo is OK iff none of the 3 pairwise dart-sets collide.
    Returns (compatible, [((ob,oc), ok), ...])."""
    A_F = dart_set(cyc_a, False)
    B = {False: dart_set(cyc_b, False), True: dart_set(cyc_b, True)}
    C = {False: dart_set(cyc_c, False), True: dart_set(cyc_c, True)}
    results = []
    for ob in (False, True):
        for oc in (False, True):
            Bc, Cc = B[ob], C[oc]
            ok = not ((A_F & Bc) or (A_F & Cc) or (Bc & Cc))
            results.append(((ob, oc), ok))
    return any(ok for _, ok in results), results


def run_dart_audit(cycles, edgesets, ground_pairs, ground_triples):
    banner("STEP 2 -- THE DART-OCCUPATION AUDIT")

    # ---- ground pairs (648) ----
    pair_compat, pair_incompat = [], []
    same_only = opp_only = both_work = 0
    shared_hist = Counter()
    for (a, b) in ground_pairs:
        compat, sw, ow = pair_compatible(cycles[a], cycles[b])
        (pair_compat if compat else pair_incompat).append((a, b))
        if sw and ow:
            both_work += 1
        elif sw:
            same_only += 1
        elif ow:
            opp_only += 1
        shared_hist[len(edgesets[a] & edgesets[b])] += 1
    print(f"\n    ground pairs (DeltaS=3, N=648): "
          f"{len(pair_compat)} orientation-compatible, {len(pair_incompat)} incompatible")
    print(f"      resolution mode: same-orientation-only {same_only}, "
          f"opposite-orientation-only {opp_only}, both work {both_work}")
    print(f"      shared undirected-edge-count histogram: {dict(shared_hist)}")

    # ---- ground triples (216) ----
    tri_compat, tri_incompat = [], []
    n_ok_hist = Counter()
    shared_triple_hist = Counter()
    for (i, j, k) in ground_triples:
        compat, results = triple_compatible(cycles[i], cycles[j], cycles[k])
        (tri_compat if compat else tri_incompat).append((i, j, k))
        n_ok_hist[sum(1 for _, ok in results if ok)] += 1
        sij = len(edgesets[i] & edgesets[j])
        sik = len(edgesets[i] & edgesets[k])
        sjk = len(edgesets[j] & edgesets[k])
        shared_triple_hist[(sij, sik, sjk)] += 1
    print(f"\n    ground triples (DeltaS=13, N=216): "
          f"{len(tri_compat)} orientation-compatible, {len(tri_incompat)} incompatible")
    print(f"      number of the 4 orientation-combos that work, histogram: {dict(n_ok_hist)}")
    print(f"      pairwise shared-edge-count-triple histogram: {dict(shared_triple_hist)}")

    check(f"ALL 648 ground pairs orientation-compatible", len(pair_compat) == 648)
    check(f"ALL 216 ground triples orientation-compatible", len(tri_compat) == 216)

    # ---- coincidence (mult_e = 2, 3 at zero relative displacement) ----
    print("\n    coincidence configurations (same cycle, multiplicity 2 / 3, zero "
          "relative displacement):")
    print("    FORCED reading (the physically correct one): the self-translation ansatz")
    print("    applies the IDENTICAL translation (an orientation/chirality-preserving")
    print("    isometry of the net) to the SAME canonical cycle for every copy; at zero")
    print("    displacement every copy IS the same directed object, not an independently")
    print("    re-orientable slot -- there is only ONE assignment to check.")
    sample = cycles[0]
    AF = dart_set(sample, False)
    AR = dart_set(sample, True)
    collide_2_fwd = len(AF & AF)
    collide_2_rev = len(AR & AR)
    print(f"      2-copy, forced-forward:  {collide_2_fwd}/10 darts doubly occupied")
    print(f"      2-copy, forced-backward: {collide_2_rev}/10 darts doubly occupied")
    check("coincidence (2-copy, forced orientation): ALL 10 darts doubly occupied "
          "(either global choice)", collide_2_fwd == 10 and collide_2_rev == 10)
    print(f"      3-copy: a_d^2=0 already kills every dart at 2-fold; the 3rd copy is "
          f"moot (10/10 darts >=2x occupied either way)")

    # transparency: what the (WRONG) free-independent-orientation reading would say
    naive_escape = len(AF & AR)
    print(f"\n    TRANSPARENCY CHECK -- if (incorrectly) the two coincident copies were")
    print(f"    treated as independently, freely orientable (as if they were the two")
    print(f"    DISTINCT cycles the ground-pair audit above uses): opposite orientation")
    print(f"    gives {naive_escape}/10 collisions -- an APPARENT escape.  This is why the")
    print(f"    forcing argument (identical translation => identical transported")
    print(f"    orientation, not independent re-labeling) is the load-bearing physical")
    print(f"    premise here, not the combinatorics alone; verified generically below.")

    # generic check across ALL 324 cycles (not just cycles[0]): self-overlap under a
    # SINGLE fixed orientation is ALWAYS full (10/10) -- a trivial identity, confirmed
    # exhaustively rather than asserted from one sample.
    all_full = all(len(dart_set(c, False) & dart_set(c, False)) == 10 for c in cycles)
    check(f"generic identity (all {len(cycles)} cycles): self dart-set has full "
          f"self-overlap under either fixed orientation", all_full)

    return {
        "pair_compat": pair_compat, "pair_incompat": pair_incompat,
        "tri_compat": tri_compat, "tri_incompat": tri_incompat,
        "same_only": same_only, "opp_only": opp_only, "both_work": both_work,
        "shared_hist": dict(shared_hist), "n_ok_hist": dict(n_ok_hist),
    }


# ===========================================================================
# BLOCK 3 -- THE STATISTICS STEP (accreted the_net.py APIs ONLY, per pre-reg)
# ===========================================================================
def run_statistics_step():
    banner("STEP 3 -- CONNECTING DART MODES TO THE FERMIONIC GRADING "
           "(accreted the_net.py APIs only)")

    sc = the_net.gauge_sector_category()
    print(f"\n    gauge_sector_category() [the_net.py:656]:")
    print(f"      species_sector_dims = {sc['species_sector_dims']}   "
          f"(w=0..3 -> nu,d,u,e; = C(3,w), a genuine 3-fundamental-fermionic-mode Fock count)")
    print(f"      fermion_parity      = {sc['fermion_parity']}   (= (-1)^w, EXACT, forced by the "
          f"Cl(6)/Clifford construction)")
    print(f"      double_cover_2T = {sc['double_cover_2T']}, sectors_are_species = "
          f"{sc['sectors_are_species']}   (ML-2b: sectors_are_species is CONDITIONAL on the "
          f"TD-limit twisted Haag duality premise, NOT verified by this or that suite)")

    tl = the_net.twisted_locality_holds()
    print(f"\n    twisted_locality_holds() [the_net.py:311]  (explicit finite JW Fock demo):")
    for k, v in tl.items():
        print(f"      {k:24s} = {float(v):.6e}")
    print(f"      => odd operators GENUINELY anticommute (odd_odd_anticommute=0 residual on the "
          f"ANTI-commutator; naive_commutation_fails={float(tl['naive_commutation_fails']):.1f} != 0 "
          f"shows this is real anticommutation, not accidental commutation); even operators commute; "
          f"the Klein twist restores locality.  This is the CAR MECHANISM, demonstrated as available.")

    zA = the_net.z2_gradings("A4")
    z2T = the_net.z2_gradings("2T")
    print(f"\n    z2_gradings() [the_net.py:1663] (MS-1a, 2026-07-10):")
    print(f"      z2_gradings('A4')  = {zA}   (ONLY the trivial grading -- NO nontrivial Z2 "
          f"statistics grading exists at the bare/un-doubled species level)")
    print(f"      z2_gradings('2T')  = {z2T}   (trivial + EXACTLY one nontrivial grading, which "
          f"EQUALS the fermion-parity/center grading -- fermion parity is the ONE AND ONLY "
          f"nontrivial Z2 sector grading this category supports, PROVEN, not assumed)")

    sgh = the_net.sector_grading_hist(N_max=2)
    print(f"\n    sector_grading_hist(N_max=2) [the_net.py:2077] (\"the Sec.8 grading\", FOCK-0 "
          f"2026-07-11 -- H_hist, the I2b word-Fock space where darts/cycles actually live):")
    print(f"      irrep_dims = {sgh['irrep_dims']}, length-1 (the 12 elementary darts) "
          f"multiplicities = {sgh['mult'][1].tolist()}  (== irrep_dims itself, the regular-rep "
          f"property, cross-check independent of Section 7's character-inner-product route)")
    print(f"      parity_commute_residual = {sgh['parity_commute_residual']:.2e}  (word-length "
          f"parity (-1)^n COMMUTES with the species/isotypic grading, to machine precision)")

    print(f"\n    ================ THE SYNTHESIS (honest, both sides) ================")
    print(f"""
    The premise that WOULD carry CAR, if it applied to darts: the Cl(6) field-algebra
    construction underlying gauge_sector_category() is a GENUINE fermionic (CAR) Fock
    space (3 fundamental modes, w = fermion number, fermion_parity = (-1)^w EXACT);
    twisted_locality_holds() demonstrates the Klein-twisted-odd-anticommutation
    MECHANISM this would use is real and available in the framework's toolkit;
    z2_gradings() PROVES fermion parity is the UNIQUE nontrivial Z2 grading the
    category supports at all (R(A4) alone: only the trivial grading -- nothing
    forces ANY nontrivial statistics label onto the bare species content).

    BUT: the object T0-N-3's cycles actually live on is H_hist (the I2b word-Fock
    space built from Cuntz-Krieger word-extension isometries S_d, the_net.py
    Section 7b) -- a DIFFERENT Fock space from the Cl(6) field algebra F.  Sector_
    grading_hist (2026-07-11, "the Sec.8 grading") is the accreted attempt to connect
    them, and its OWN self-documented finding (FOCK0_return_2026-07-11.md Sec.5 item
    1, INDEPENDENTLY CONFIRMED by FOCK0_check_2026-07-11.md Sec.6 defect #2) is that
    word-length parity (-1)^n as H_hist's stand-in fermion-parity bit is EXTRINSIC:
    checked ONLY for compatibility (it commutes with the species grading, residual
    {sgh['parity_commute_residual']:.1e}, re-verified above) -- NOT forced or derived
    from the A4/sector content.  Per z2_gradings('A4')=={zA}, nothing in the bare
    category forces darts into the odd/Fermi slot; the assignment is a free,
    consistent, but UNFORCED choice.  FOCK-0's further, more ambitious attempt to
    build a FORCED equivariant bridge (Hom_A4(dart_rep, field-algebra F)) found that
    space to be exactly {{0}} -- no such natural map exists -- though its own
    adversarial checker flags this specific fact as a generic, unsurprising
    algebraic consequence of the representation TYPES involved (an honest regular
    representation vs. a genuinely projective one), not a lattice-specific theorem,
    and flags that promoting it to a settled "V3 obstruction" oversteps evidence
    that is itself still ARCHITECT-PENDING (FOCK-0's own verdict section is headed
    "Proposed verdict (evidence only)").

    CONCLUSION: the accreted the_net.py structure does NOT, today, FORCE CAR onto
    the dart/word-Fock occupation numbers that the girth-10-cycle/DeltaS-ladder
    construction lives on.  The genuinely fermionic (CAR) structure is real but
    lives on a part of the category with NO ESTABLISHED FORCED connection to darts;
    adopting "darts are Fermi" requires the EXTRINSIC, UNFORCED choice FOCK-0 itself
    names, not a derived consequence of gauge_sector_category / twisted_locality_
    holds / z2_gradings alone.
    """)
    print(f"    QUALIFIER STACK (attaches to every sentence above and below that touches the "
          f"sector category):")
    print(f"      (i)  ML-2b -- gauge_sector_category()'s sectors_are_species / DR-frame "
          f"identification is conditional on TD-limit twisted Haag duality, NOT verified here.")
    print(f"      (ii) the FOCK-0 bridge status -- the word-length-parity-as-fermion-parity "
          f"reading is EXTRINSIC (checked, not forced); the equivariant-Hom obstruction is "
          f"evidence, not an adjudicated theorem, per its own checker.")

    return {"gauge_sector_category": sc, "twisted_locality_holds": tl,
            "z2_gradings_A4": zA, "z2_gradings_2T": z2T, "sector_grading_hist": sgh}


# ===========================================================================
# BLOCK 4 -- THE THEOREM (conditional algebraic proof) + MACHINE DEMONSTRATION
# on a small truncation.  The algebra below is exact CAR combinatorics (a Slater-
# determinant / antisymmetrized-wedge representation: a sequence of creation-
# operator mode-labels is the zero vector iff any label repeats, else it reduces,
# by transposition parity, to a signed canonical basis ket) -- IF the CAR premise
# holds for these darts (Block 3's open question).  A separate, fully explicit
# small (2^10-dim) Jordan-Wigner matrix demonstration on REAL data (one actual
# ground pair's 5 shared edges) backs the same claim with concrete vectors.
# ===========================================================================
def apply_creation_sequence(mode_labels):
    """Antisymmetrized (CAR) application of a_d1^dag a_d2^dag ... a_dn^dag |0>: returns
    None if the state is the zero vector (a repeated mode label anywhere forces
    a_d^2=0, by CAR, regardless of position -- anticommuting past other operators
    only introduces signs, never removes a genuine double occupation), else
    (sign, canonical_sorted_tuple) via bubble-sort transposition-parity tracking."""
    seq = list(mode_labels)
    if len(set(seq)) != len(seq):
        return None
    sign = 1
    n = len(seq)
    for i in range(n):
        for j in range(n - 1 - i):
            if seq[j] > seq[j + 1]:
                seq[j], seq[j + 1] = seq[j + 1], seq[j]
                sign *= -1
    return sign, tuple(seq)


def jw_creation_ops(N):
    """Standard Jordan-Wigner creation operators on a 2^N-dim Fock space (own,
    self-contained construction -- NOT importing the_net.py's private _jw_ops;
    same standard convention twisted_locality_holds uses: Z-string before site p,
    raising sigma^+ = |1><0| at p, identity after)."""
    I2 = np.eye(2)
    Z = np.diag([1.0, -1.0])
    sp = np.array([[0.0, 0.0], [1.0, 0.0]])
    ops = []
    for p in range(N):
        m = np.array([[1.0]])
        for q in range(N):
            m = np.kron(m, Z if q < p else (sp if q == p else I2))
        ops.append(m)
    return ops


def run_theorem_and_demo(cycles, edgesets, ground_pairs, audit):
    banner("STEP 4 -- THE THEOREM: algebraic proof + machine demonstration")
    print("""
    ALGEBRAIC PROOF (conditional on Block 3's CAR premise holding for these darts):
      Let a_d^dag be the creation operator for dart d (a directed edge).  CAR:
      {a_d, a_d^dag} = 1, {a_d^dag, a_d^dag} = 0  =>  (a_d^dag)^2 = 0 for every d.
      (1) COINCIDENCE: the self-translation ansatz forces every copy of a coincident
          configuration onto the SAME orientation (Block 2); hence the occupation
          state is Prod_d (a_d^dag)^{m_d} |0> with m_d >= 2 for all 10 darts of the
          cycle.  Since (a_d^dag)^2 = 0 for each such d, the state is IDENTICALLY
          ZERO -- not deep, not capped: nonexistent as a Fock vector.  (m_d=3 for the
          3-body coincidence is a fortiori zero, since m_d=2 already vanishes.)
      (2) CERTIFIED CONFIGURATIONS: Block 2 exhibits, for every one of the 648
          ground pairs and 216 ground triples, an explicit orientation assignment
          under which every dart index used across all cycles in the configuration
          is DISTINCT (no repeats).  The corresponding state Prod_d a_d^dag |0> (all
          exponents m_d = 1) is then a genuine, nonzero, fully antisymmetrized
          Fock state -- CAR does not annihilate it.  The certified distinct-cycle
          domain survives.
    """)

    # ---- demonstration A: exact CAR combinatorics on REAL data, full alphabet ----
    a0, b0 = ground_pairs[0]
    cyc_a, cyc_b = cycles[a0], cycles[b0]
    _, same_ok, opp_ok = pair_compatible(cyc_a, cyc_b)
    ob = False if same_ok else True   # the actual compatible orientation found in Block 2
    A_darts = list(dart_set(cyc_a, False))
    B_darts = list(dart_set(cyc_b, ob))
    ground_pair_seq = A_darts + B_darts       # 20 creation ops, all-distinct labels (by Block 2)
    coincidence_seq = A_darts + A_darts        # 20 creation ops, every one of the 10 labels twice

    res_pair = apply_creation_sequence(ground_pair_seq)
    res_coinc = apply_creation_sequence(coincidence_seq)
    print(f"    DEMO A (exact CAR combinatorics, real ground pair #{(a0, b0)}, full 20-dart "
          f"alphabet):")
    print(f"      ground-pair state (orientation {'same' if not ob else 'opposite'}): "
          f"{'ZERO' if res_pair is None else f'NONZERO, sign={res_pair[0]:+d}'}")
    print(f"      coincidence state (same cycle x2, forced orientation):        "
          f"{'ZERO' if res_coinc is None else 'NONZERO'}")
    check("DEMO A: ground-pair state is NONZERO", res_pair is not None)
    check("DEMO A: coincidence state is ZERO", res_coinc is None)

    # ---- demonstration B: explicit small (2^10-dim) JW matrix vectors on the 5
    # shared edges of the SAME real ground pair ----
    shared_edges = sorted(edgesets[a0] & edgesets[b0], key=lambda e: sorted(e))
    assert len(shared_edges) == 5, f"expected 5 shared edges, got {len(shared_edges)}"
    N = 2 * len(shared_edges)     # 10 modes: 2 per shared edge (A's direction, B's direction)
    ops = jw_creation_ops(N)
    vac = np.zeros(2 ** N)
    vac[0] = 1.0

    # mode 2k = "A-direction" on shared edge k (the dart cycle_a actually uses there);
    # mode 2k+1 = "B-direction" (opposite dart) on the same edge.
    def apply_ops(indices, psi):
        v = psi.copy()
        for idx in reversed(indices):  # apply rightmost operator first (any fixed order)
            v = ops[idx] @ v
        return v

    ground_pair_state = apply_ops(list(range(0, N, 2)) + list(range(1, N, 2)), vac)
    coincidence_state = apply_ops(list(range(0, N, 2)) + list(range(0, N, 2)), vac)
    n_pair = float(np.linalg.norm(ground_pair_state))
    n_coinc = float(np.linalg.norm(coincidence_state))
    print(f"\n    DEMO B (explicit {N}-mode / 2^{N}={2**N}-dim JW Fock space, the SAME real "
          f"ground pair's 5 shared edges):")
    print(f"      ||ground-pair vector||  = {n_pair:.6f}  (nonzero: opposite directions on "
          f"every shared edge -> 10 distinct occupied modes)")
    print(f"      ||coincidence vector||  = {n_coinc:.6e}  (zero: same direction applied twice "
          f"per edge -> (a_d^dag)^2 annihilates)")
    check("DEMO B: ground-pair vector is NONZERO (norm > 0.9)", n_pair > 0.9)
    check("DEMO B: coincidence vector is EXACTLY ZERO (norm < 1e-9)", n_coinc < 1e-9)


# ===========================================================================
# BLOCK 5 -- CONSEQUENCE NOTE ONLY (no solve, no kappa, no B2/B3, no confrontation)
# ===========================================================================
def print_consequence_note(verdict):
    banner("STEP 5 -- CONSEQUENCE NOTE (no solve run; T0-N-4 named, not executed)")
    print("""
    IF the coincidence-exclusion mechanism above were established unconditionally
    (it is NOT, per Block 3 -- verdict below), the implication chain would be:
      - 2-BODY: IV4_T0_class_2026-07-10.py's DS_CAP=3.0 (edge_resolved_profile,
        lines ~252-274) currently CAPS the raw self-translation profile (which
        reaches DeltaS=10 at coincidence) down to the certified rung 3 by a
        DISCLOSED, UNDERIVED empirical choice.  Excluding the coincidence point
        outright (rather than capping its VALUE) would replace that empirical
        choice with a DOMAIN restriction: the deepest configuration remaining in
        the (correctly excluded) Hilbert space is exactly the certified ground-pair
        rung DeltaS=3 -- a CANDIDATE derivation of the cap's existence (not yet its
        confirmed dynamical consequence; that requires an actual re-solve).
      - 3-BODY: T0_NUCLEAR2's KIN-WRONG-WAY pathology was traced to the ground state
        collapsing onto the (0,0,0) coincidence well (raw DeltaS=20, ~99.99% of
        <V>).  Excluding that point from the relative-cell box's domain would force
        the solver's variational space away from that well; whether B3 then moves
        TOWARD (not away from) the mirror mean is an open, EMPIRICAL question for a
        genuine re-solve.
      - NAMED NEXT STATION: T0-N-4 (re-solve on the excluded domain) -- eligible for
        pre-registration ONLY on a D1 (or a subsequently strengthened) verdict; NOT
        executed here, per the poisons (no binding solve, no kappa, no B2/B3, no
        confrontation anywhere in this file).

    ACTUAL VERDICT (below) is """ + verdict + """, so the above chain is NOT
    activated as a derivation: the DS_CAP=3 delta STAYS EXACTLY AS UNDERIVED AS
    BEFORE THIS STATION, and the kinetic-completion route stays closed-negative
    (KIN-WRONG-WAY, booked raw, per T0-NUCLEAR-2).  The combinatorial content of
    Block 2 (100% orientation-compatibility, unique resolution, on both the 648
    ground pairs and 216 ground triples) remains a genuine, useful, UNCONDITIONAL
    structural fact for whichever future station next attempts a CAR-based (or
    other) derivation of coincidence exclusion -- it is not invalidated by the
    statistics finding, only its APPLICATION as an exclusion principle is
    unsupported today.
    """)


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    banner("T0-N-3 -- THE DOMAIN THEOREM (coincidence exclusion by statistics)")
    print("pre-reg: internal research notes (FROZEN)")

    banner("STEP 1 -- REPRODUCE + RE-LOCK the frozen supercell(3) enumeration; "
           "lift cycles to dart orientations")
    cycles, edgesets, ground_pairs, ground_triples = build_reference_configs()

    audit = run_dart_audit(cycles, edgesets, ground_pairs, ground_triples)
    stats = run_statistics_step()
    run_theorem_and_demo(cycles, edgesets, ground_pairs, audit)

    # ================= VERDICT ADJUDICATION (frozen tree, SS3) =================
    banner("VERDICT ADJUDICATION")
    combinatorics_clean = (len(audit["pair_compat"]) == 648 and len(audit["tri_compat"]) == 216
                            and len(audit["pair_incompat"]) == 0 and len(audit["tri_incompat"]) == 0)
    car_forced = False   # per Block 3's synthesis: NOT forced by the accreted APIs, honestly

    print(f"""
    Combinatorial half (Block 2, UNCONDITIONAL): ALL 648 ground pairs and ALL 216
    ground triples admit an orientation assignment with no dart occupied twice
    (100% each, and in fact a UNIQUE resolution up to the overall global flip in
    every single case -- same_only={audit['same_only']}, opp_only={audit['opp_only']},
    both_work={audit['both_work']} for pairs; the 4-combo histogram for triples is
    {audit['n_ok_hist']}, i.e. exactly 1-of-4 for all 216).  Coincidence, under the
    physically forced (translation-covariant) reading, collides on all 10 darts,
    trivially and generically (verified on all {len(cycles)} enumerated cycles).
    NOT D3 (no contradiction: the certified rungs are never forced into
    same-dart double-occupation) and NOT D4 (the orientation lift is a clean,
    unambiguous 2-fold combinatorial fact, not underdetermined).

    Statistics half (Block 3, CONDITIONAL): the CAR premise needed to turn "all
    darts doubly occupied" into "the amplitude is identically zero" is NOT forced
    by the accreted the_net.py APIs (gauge_sector_category / twisted_locality_holds
    / z2_gradings / sector_grading_hist).  The genuinely fermionic (CAR) structure
    is real (Cl(6), the Klein-twist mechanism, z2_gradings' uniqueness result) but
    lives on a part of the sector category with no FORCED, accreted connection to
    the I2b dart/word-Fock algebra the cycles live on -- the identification is
    EXTRINSIC (FOCK-0's own finding, independently confirmed by its checker).

    => VERDICT: D2 NO-EXCLUSION.  "The dictionary does not force CAR dart modes" --
    per the pre-reg's own D2 text: the cap stays underived, the kinetic-completion
    route stays closed-negative, booked raw.  EVERY sentence above carries the
    ML-2b qualifier (TD-limit twisted Haag duality, unverified) PLUS the named
    FOCK-0 bridge-status qualifier (extrinsic parity assignment; the equivariant-
    Hom obstruction is evidence pending architect adjudication, per its own
    checker) -- both stated, neither hidden.
    """)
    verdict = "D2 NO-EXCLUSION"
    check("verdict adjudicated: D2 NO-EXCLUSION (combinatorics clean, CAR not forced)",
          combinatorics_clean and not car_forced)

    print_consequence_note(verdict)

    print("=" * W)
    print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}"
          f"   |   STATION VERDICT: {verdict}")
    print("=" * W)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
