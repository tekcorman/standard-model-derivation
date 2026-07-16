#!/usr/bin/env python3
"""
derivation_topdown/adapters/aqft_net.py

G4 ADAPTER — the HAAG-KASTLER / DHR contract suite on the net {A(O)}.
Pre-registered in internal research notes (frozen BEFORE this file).
Companion charter: internal research notes (G4 = "aqft_net");
protocol: internal research notes (this file = pipeline step 3,
IMPLEMENTATION). Sibling adapters (G1-G3, G5-G6) are scaffolded in adapters/README.md.

WHAT THIS FILE IS: an ADAPTER, not a new derivation. It imports the one master net object
(derivation_topdown/state/the_net.py, the {A(O)} of physics = (D, omega, {A(O)})) and asserts,
on that existing object, the defining axioms of Algebraic Quantum Field Theory (AQFT) in the
Haag-Kastler / Doplicher-Haag-Roberts sense:

    Haag & Kastler, "Algebraic approach to quantum field theory", J. Math. Phys. 5 (1964) 848.
    Doplicher, Haag & Roberts, "Local observables and particle statistics I/II",
        Commun. Math. Phys. 23 (1971) 199 and 35 (1974) 49  [DHR superselection sectors].
    Haag, "Local Quantum Physics: Fields, Particles, Algebras", Springer (2nd ed. 1996).

CLAIM = INSTANTIATION, NOT EQUIVALENCE: a green contract means "the object the framework
already built satisfies these axioms at the stated (finite, cell-duality) scope; run it and
see." It does not claim the object equals, or is the unique instance of, AQFT/DHR theory.

THE CONTRACTS (frozen tolerances; plain language; the_net / ML0-ML2 station each re-expresses):
  HK-0  ANCHORS (regression)        -- net.anchor_cell_projector(), net.anchor_tick_2pi() both True.
                                        [ML-0 cell-projector anchor; M0-2R tick-2pi anchor]
  HK-1  ISOTONY                     -- nested forward diamonds O1 subset O2 (same base, depth d<d')
                                        give A(O1) subset A(O2) as mode-sets, on >=3 base points,
                                        depths 1..3, Patch(M=4).             [ML0-1 isotony]
  HK-2  EXACT CAUSAL LOCALITY       -- {alpha_a(t), a_c^dagger} = (B^t)_ca is IDENTICALLY zero
                                        strictly below the geometric horizon t < 1+dist(head a,
                                        tail c); T in {1,2,3} on Patch(M=4), and T in {1,2,3,4} on
                                        Patch(M=5) if the time budget allows.  [ML0-2 the strict cone]
  HK-3  TWISTED (KLEIN) LOCALITY    -- even-even commute, even-odd commute, odd-odd anticommute all
                                        < 1e-12; naive (untwisted) commutation FAILS (forced twist);
                                        the Klein-twisted commutant commutes < 1e-12. [ML0-3]
  HK-4  Z^3 COVARIANCE (all 3 dirs) -- B[b,a] = B[T_e b, T_e a] identically (<1e-13) on interior
                                        dart pairs, cloned for e1, e2, AND e3 (ML0-5 only checked
                                        e1).                                  [ML0-5, generalized]
  HK-5  CELL-LEVEL TWISTED HAAG
        DUALITY (full 62-subset
        family)                    -- for EVERY region R of the 6 cell edges, 1<=|R|<=5 (all 62
                                        subsets): S(R)=S(R^c) within 1e-8 AND the nontrivial
                                        single-particle modular spectra of R, R^c agree within
                                        1e-8.                                 [ML0-4, full family]
  HK-6  DHR SECTORS == SPECIES      -- net.gauge_sector_category() gives species_sector_dims ==
                                        {0:1,1:3,2:3,3:1}, double_cover_2T == True,
                                        sectors_are_species == True, fermion_parity ==
                                        {0:+1,1:-1,2:+1,3:-1} (grading = the ML0 Klein twist).
                                                                                [ML-2 DHR sectors]
  HK-7  SCOPE DECLARATION           -- printed, not computed: what this suite does NOT claim
                                        (see the printed block below).  Never gates PASS/FAIL.

REUSE MAP (zero physics added; every symbol below is imported, none redefined):
  net.anchor_cell_projector, net.anchor_tick_2pi   -- HK-0
  net.Patch(M).diamond(base, depth)                -- HK-1 (isotony via mode-set nesting;
                                                       clones ML0_history_net_2026-07-08.py:~290)
  net.Patch(M).anticommutator_below_cone(T)         -- HK-2 (the strict combinatorial light cone)
  net.twisted_locality_holds(...)                   -- HK-3 (explicit JW Fock-space residuals)
  net.Patch(M).RD / .dpos / .B                      -- HK-4 (Z^3 covariance; clones
                                                       ML0_history_net_2026-07-08.py:~381-404 for
                                                       e1, adds e2, e3 mechanically)
  net.vacuum_covariance(), net.region_data(C, A)    -- HK-5 (cell duality, full subset family)
  net.gauge_sector_category()                       -- HK-6 (DHR sector category == species)

POISONS (binding, per pre-reg): no engine edits (bridge/the_run.py, state/the_net.py, proofs/ are
untouched); no new physics; no constants introduced beyond pure bookkeeping (tolerances declared
in the pre-reg; the one derived bookkeeping constant used below, the region_data clip-artifact
magnitude, is computed on-screen from the SAME 1e-12 clip the reused function already applies --
it classifies "pinned by the clip" vs "a genuine entangled mode", it is not a physics input); no
loosening of any tolerance or region family after seeing results.  A FAILING contract is reported
as a finding, not massaged.
"""
import itertools
import math
import os
import sys
import time

import numpy as np

_T0 = time.time()

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "state"))
import the_net as net  # noqa: E402

np.set_printoptions(precision=6, suppress=True)
ok_all = True          # gates HK-0..HK-6 only (HK-7 is a declaration, never a gate)


def check(name, cond, detail=""):
    global ok_all
    if not cond:
        ok_all = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    return cond


def banner(t):
    print("=" * 88)
    print(f" {t}")
    print("=" * 88)


print("=" * 88)
print(" G4 ADAPTER -- Haag-Kastler / DHR contract suite on the net {A(O)}")
print(" (Haag-Kastler 1964; Doplicher-Haag-Roberts 1971/74; Haag, Local Quantum Physics)")
print("=" * 88)

# ===========================================================================
banner("HK-0  ANCHORS  (regression: the two known degenerate region-shapes)")
# ===========================================================================
a_cell = net.anchor_cell_projector()
a_tick = net.anchor_tick_2pi()
check("HK-0a net.anchor_cell_projector(): cell C=(I+iJ6)/2 is an exact rank-3 projector",
      a_cell, detail=f"anchor_cell_projector() = {a_cell}")
check("HK-0b net.anchor_tick_2pi(): tick modular flow is a compact U(1) of minimal period 2pi",
      a_tick, detail=f"anchor_tick_2pi() = {a_tick}")

# ===========================================================================
banner("HK-1  ISOTONY  (nested forward diamonds; A(O1) subset A(O2) as mode-sets)")
# ===========================================================================
patch4 = net.Patch(M=4)
central4 = patch4.central_dart()
bases = sorted(set([0, patch4.Nd // 3, patch4.Nd // 2, central4]))
check(f"HK-1 pre-req: >=3 distinct base darts chosen on Patch(M=4)", len(bases) >= 3,
      detail=f"bases={bases} (Nd={patch4.Nd})")

iso_ok = True
iso_detail = []
for base in bases:
    d1 = patch4.diamond(base, 1)
    d2 = patch4.diamond(base, 2)
    d3 = patch4.diamond(base, 3)
    nested = (d1 <= d2) and (d2 <= d3) and (d1 <= d3)
    iso_ok = iso_ok and nested
    iso_detail.append(f"base={base}:|O1,2,3|={[len(d1), len(d2), len(d3)]}:nested={nested}")
check("HK-1 isotony: O1 subset O2 subset O3 (mode-sets) for every base, depths 1<2<3",
      iso_ok, detail="; ".join(iso_detail))

# ===========================================================================
banner("HK-2  EXACT CAUSAL LOCALITY  (the strict combinatorial light cone)")
# ===========================================================================
hk2_ok = True
worst4 = 0.0
for T in (1, 2, 3):
    w = patch4.anticommutator_below_cone(T)
    worst4 = max(worst4, w)
    hk2_ok = check(f"HK-2 Patch(M=4) T={T}: max|{{alpha_a(t),a_c^dag}}| below cone < 1e-13",
                   w < 1e-13, detail=f"measured max|below cone| = {w:.3e}") and hk2_ok

elapsed_so_far = time.time() - _T0
BUDGET_S = 600.0
if elapsed_so_far < 0.5 * BUDGET_S:
    patch5 = net.Patch(M=5)
    worst5 = 0.0
    for T in (1, 2, 3, 4):
        if time.time() - _T0 > 0.9 * BUDGET_S:
            print(f"  [SKIP] Patch(M=5) T={T}: remaining budget too small, stopping optional block")
            break
        w = patch5.anticommutator_below_cone(T)
        worst5 = max(worst5, w)
        hk2_ok = check(f"HK-2(optional) Patch(M=5) T={T}: max|{{alpha_a(t),a_c^dag}}| below "
                        f"cone < 1e-13", w < 1e-13, detail=f"measured max|below cone| = {w:.3e}") and hk2_ok
else:
    print("  [SKIP] Patch(M=5) optional block dropped: insufficient remaining time budget "
          f"(elapsed {elapsed_so_far:.1f}s of {BUDGET_S:.0f}s cap)")

# ===========================================================================
banner("HK-3  TWISTED (KLEIN) LOCALITY")
# ===========================================================================
tl = net.twisted_locality_holds()
check("HK-3a even(R1) commutes with even(R2)", tl["even_even_commute"] < 1e-12,
      detail=f"max|[e_A,e_B]| = {tl['even_even_commute']:.3e}")
check("HK-3b even(R1) commutes with odd(R2)", tl["even_odd_commute"] < 1e-12,
      detail=f"max|[e_A,o_B]| = {tl['even_odd_commute']:.3e}")
check("HK-3c odd(R1) anticommutes with odd(R2)", tl["odd_odd_anticommute"] < 1e-12,
      detail=f"max|{{o_A,o_B}}| = {tl['odd_odd_anticommute']:.3e}")
check("HK-3d naive (untwisted) commutation FAILS for odd-odd => the twist is FORCED",
      tl["naive_commutation_fails"] > 0.5,
      detail=f"max|[o_A,o_B]| = {tl['naive_commutation_fails']:.3e} (>0.5, not numerical noise)")
check("HK-3e the Klein-twisted commutant commutes", tl["klein_twist_commutes"] < 1e-12,
      detail=f"max|[o_A, P1.o_B]| = {tl['klein_twist_commutes']:.3e}")

# ===========================================================================
banner("HK-4  Z^3 COVARIANCE  (all three lattice directions; ML0-5 generalized to e2, e3)")
# ===========================================================================
def shifted(dart, e):
    (ti, tx), (hi, hx) = dart
    return ((ti, tuple(np.array(tx) + e)), (hi, tuple(np.array(hx) + e)))


for name, e in (("e1", np.array([1, 0, 0])), ("e2", np.array([0, 1, 0])), ("e3", np.array([0, 0, 1]))):
    interior = [n for n, d in enumerate(patch4.RD) if shifted(d, e) in patch4.dpos]
    Tsh = {n: patch4.dpos[shifted(patch4.RD[n], e)] for n in interior}
    cov_ok = True
    worst_cov = 0.0
    checked = 0
    for a in interior:
        for b in interior:
            if patch4.B[b, a] != 0 or patch4.B[Tsh[b], Tsh[a]] != 0:
                diff = abs(patch4.B[b, a] - patch4.B[Tsh[b], Tsh[a]])
                worst_cov = max(worst_cov, diff)
                if diff > 1e-13:
                    cov_ok = False
                checked += 1
    check(f"HK-4 Z^3 covariance along {name}: B[b,a] = B[T_{name} b, T_{name} a] identically",
          cov_ok and worst_cov < 1e-13,
          detail=f"max diff = {worst_cov:.3e}, {checked} nonzero dart-pairs matched, "
                 f"{len(interior)} interior darts")

# ===========================================================================
banner("HK-5  CELL-LEVEL TWISTED HAAG DUALITY  (full 62-subset family, 1<=|R|<=5)")
# ===========================================================================
C = net.vacuum_covariance()
NE = net.NE
edges = list(range(NE))
# the exact clip-artifact magnitude produced by the REUSED region_data's internal 1e-12 clip
# (log((1-1e-12)/1e-12)); an entry pinned at this magnitude is a clip artifact of the reused
# function, not a genuine entangled mode. This is pure bookkeeping derived from the same
# constant region_data already uses -- no new physics constant.
CLIP_PIN = math.log((1 - 1e-12) / 1e-12)
NT_CUT = CLIP_PIN - 1.0

subsets = [R for k in range(1, NE) for R in itertools.combinations(edges, k)]
check(f"HK-5 pre-req: family = all subsets R with 1<=|R|<=5 of the 6 cell edges",
      len(subsets) == 62, detail=f"|family| = {len(subsets)} (expected 62)")

worst_S = 0.0
worst_eps = 0.0
count_mismatch_len = 0
count_fail_S = 0
count_fail_eps = 0
for R in subsets:
    Rc = tuple(e for e in edges if e not in R)
    zR, epsR, SR = net.region_data(C, list(R))
    zRc, epsRc, SRc = net.region_data(C, list(Rc))
    dS = abs(SR - SRc)
    worst_S = max(worst_S, dS)
    if dS >= 1e-8:
        count_fail_S += 1
    ntR = np.sort(np.abs(epsR[np.abs(epsR) < NT_CUT]))
    ntRc = np.sort(np.abs(epsRc[np.abs(epsRc) < NT_CUT]))
    if len(ntR) != len(ntRc):
        count_mismatch_len += 1
        continue
    d_eps = np.max(np.abs(ntR - ntRc)) if len(ntR) else 0.0
    worst_eps = max(worst_eps, d_eps)
    if d_eps >= 1e-8:
        count_fail_eps += 1

check("HK-5a entropy duality: S(R) = S(R^c) within 1e-8, over all 62 subsets",
      worst_S < 1e-8 and count_fail_S == 0,
      detail=f"worst |S(R)-S(R^c)| = {worst_S:.3e} over 62 subsets; {count_fail_S} failing")
check("HK-5b nontrivial single-particle modular spectra agree (sorted |eps|) within 1e-8, "
      "same nontrivial mode count, over all 62 subsets",
      count_mismatch_len == 0 and worst_eps < 1e-8 and count_fail_eps == 0,
      detail=f"worst sorted-|eps| mismatch = {worst_eps:.3e}; {count_mismatch_len} count-mismatches, "
             f"{count_fail_eps} value-failures (nontrivial cutoff |eps|<{NT_CUT:.3f}, "
             f"clip-pin magnitude {CLIP_PIN:.3f})")

# ===========================================================================
banner("HK-6  DHR SECTORS == SPECIES  (superselection category of the observable algebra)")
# ===========================================================================
sc = net.gauge_sector_category()
check("HK-6a species_sector_dims == {0:1, 1:3, 2:3, 3:1}",
      sc["species_sector_dims"] == {0: 1, 1: 3, 2: 3, 3: 1},
      detail=f"measured dims = {sc['species_sector_dims']}")
check("HK-6b double_cover_2T == True (spinorial gauge group)",
      sc["double_cover_2T"] is True, detail=f"double_cover_2T = {sc['double_cover_2T']}")
check("HK-6c sectors_are_species == True",
      sc["sectors_are_species"] is True, detail=f"sectors_are_species = {sc['sectors_are_species']}")
check("HK-6d fermion_parity == {0:+1, 1:-1, 2:+1, 3:-1} (grading = the ML0 Klein twist)",
      sc["fermion_parity"] == {0: 1, 1: -1, 2: 1, 3: -1},
      detail=f"measured fermion_parity = {sc['fermion_parity']}")

# ===========================================================================
banner("HK-7  SCOPE DECLARATION  (printed, NOT computed; never gates PASS/FAIL)")
# ===========================================================================
print("""  This suite does NOT claim, and none of HK-0..HK-6 establishes:
    (i)   THERMODYNAMIC-LIMIT (infinite-lattice) Haag duality. Every duality check here (HK-5)
          is CELL-LEVEL only (the 6-edge static vacuum). ML-2b's DR-frame argument is CONDITIONAL
          on the TD-limit duality holding, which is NOT verified by this suite.
    (ii)  Local DHR charge transporters / braiding statistics at GENERAL regions. HK-6 verifies the
          sector CATEGORY (dimensions, double cover, grading) but no intertwiner/braiding
          construction at general causal regions is built here.
    (iii) Past-cone / diamond-intersection closure beyond FORWARD diamonds. HK-1/HK-2 use forward
          cones (net.Patch.diamond, net.Patch.anticommutator_below_cone) only; the full
          past-cone-intersected causal diamond and its closure properties are not checked.
  These remain OPEN and are carried into adapters/README.md as declared, unclaimed scope.""")

# ===========================================================================
banner("SUMMARY")
# ===========================================================================
elapsed = time.time() - _T0
print(f"""  HK-0  anchors                       : {'PASS' if (a_cell and a_tick) else 'FAIL'}
  HK-1  isotony                       : {'PASS' if iso_ok else 'FAIL'}
  HK-2  exact causal locality         : {'PASS' if hk2_ok else 'FAIL'}
  HK-3  twisted (Klein) locality      : {'PASS' if (tl['even_even_commute'] < 1e-12 and tl['even_odd_commute'] < 1e-12 and tl['odd_odd_anticommute'] < 1e-12 and tl['naive_commutation_fails'] > 0.5 and tl['klein_twist_commutes'] < 1e-12) else 'FAIL'}
  HK-4  Z^3 covariance (e1,e2,e3)     : see per-direction lines above
  HK-5  cell-level twisted Haag dual. : {'PASS' if (worst_S < 1e-8 and count_fail_S == 0 and count_mismatch_len == 0 and worst_eps < 1e-8 and count_fail_eps == 0) else 'FAIL'}
  HK-6  DHR sectors == species        : {'PASS' if (sc['species_sector_dims'] == {0: 1, 1: 3, 2: 3, 3: 1} and sc['double_cover_2T'] and sc['sectors_are_species'] and sc['fermion_parity'] == {0: 1, 1: -1, 2: 1, 3: -1}) else 'FAIL'}
  HK-7  scope declaration             : printed above (declaration only, not a gate)
  wall time: {elapsed:.1f}s""")
print("RESULT:", "ALL HK-0..HK-6 CONTRACTS PASS (Haag-Kastler instantiation at the stated scope)"
      if ok_all else "AT LEAST ONE CONTRACT FAILED -- see per-contract detail above (a finding, not a bug)")
sys.exit(0 if ok_all else 1)
