#!/usr/bin/env python3
"""Phase 5.2/R1 -- Leg A of the ratification case: only Perron carries a mean.

Spec: phase5_2_repricing_spec_2026-06-11.md, "R1 RATIFICATION ATTEMPT --
FROZEN BEFORE WORK" (hash 173f8329, register row in the freezing commit).

THE CASE (two legs; this probe is Leg A, exact math):
  Leg B (in-repo, pre-existing, verify-gated): the framework's Higgs order
  parameter IS the uniform zero mode m = (1/N) sum_i Phi_i -- derived,
  STRICT-SOLID, by the MDL mean-field theorem (predictions/
  v_higgs_derivation.md Step 2; suite entry "MDL mean-field uniquely
  optimal"); the field<->Higgs identification is G3, closed under A5
  (axiom slate, already priced). Dictionary-independent, pre-Phase-5.
  Leg A (here): the spatial mean functional is EXACTLY the Gamma-Perron
  coordinate -- the uniform edge vector is the Gamma-Perron mode, and
  every other mode of the Bloch decomposition has zero spatial mean.
  Composition: the framework's own order parameter can only be carried by
  the family containing the Gamma-Perron mode => the Higgs placement
  among {h_N, h_N_neg, Perron} is FORCED to Perron.

THIS PROBE DOES NOT MOVE THE LEDGER ROW. The 5.2 panel priced the Higgs
placement (1.585 bits); overriding a panel pricing requires the ultracode
ratification ordered in the frozen attempt spec. Gates here lock the
mathematical leg only.

[R1 PANEL VERDICT 2026-06-11 (4 refuters + judge, wf_ecc5e682): PARTIAL
under K-R1-3 -- row STAYS 2.0, sensitivity NOT executed. K-R1-1 clear
(Leg B provenance git-verified: srs_mdl_meanfield_theorem.py 2026-04-11;
v_higgs Step 2 + A5/G3 paragraph verbatim at bf2bb67 2026-05-12,
pre-dictionary). K-R1-2 clear (this probe's math airtight after the
ordered rewordings below, volume-independent at L=6). Composition gaps
(the refile path, NOT discharged here): (1) mode-selection -- v_higgs
Step 2 excludes pairwise couplings only, does not adjudicate staggered
single-mode CW vacua (uniform-modulus zero-mean h_N torus configurations
EXIST); the closing argument (k != 0 quadratic action strictly positive
at criticality under the banked action + MDL mode-naming asymmetry) was
identified by the panel but NOT filed; (2) the vertex<->edge mean
transfer must be stated + gated; (3) G3's zero-cost home is the A5
clause in v_higgs itself + the panel-banked Higgs/VEV(2) role label, NOT
the axiom-slate row (which prices (A)+(B)+(I) only); hosting semantics
must be stated and reconciled with the dictionary's iso-mediated fermion
hosting. Panel rewordings 5a-5c applied in place (V4 annotation, V5
biorthogonality justification, PF lemma re-scope); no result changed.]

Gates:
  V1 fiber: B(Gamma) 1 = 2*1 exactly (row sums k*-1); eigenvalue 2 simple;
     1 is also a LEFT eigenvector (column sums 2).
  V2 fiber: all 11 non-Perron Gamma modes have zero mean (biorthogonality
     to the left-Perron 1), < 1e-12.
  V3 torus (4x4x4 primitive cells, 768 modes, full real-space NB matrix):
     exactly one eigenvalue +2; its eigenvector is the uniform vector; the
     spatial mean of EVERY other eigenvector is < 1e-9; spectral radius 2.
  V4 the rival families: the N-star phase sums vanish exactly on the torus
     (sum_d e^{2pi i N.d} = 0 for all 6 star members) -> every h_N /
     h_N_neg mode has IDENTICALLY zero spatial mean; bonus: max |lambda|
     at N = sqrt2 < 2 (no condensate-competitive Perron value at N).
  V5 exact biorthogonality (panel rewording 5a): 1^T B_T = 2*1^T
     integer-exact => (2-lambda)^k 1^T x = 0 kills the mean of every
     generalized eigenvector at lambda != 2, all Jordan structure;
     lambda = 2 simple by fiber census; B_T irreducible. The load-bearing
     lemma is LINEAR (nonzero mean <=> nonzero Perron amplitude,
     componentwise for complex/doublet fields); PF enters only as
     uniqueness of the non-negative eigendirection (rewording 5b).
  V6 Leg-B tether: predictions/v_higgs_derivation.md present and contains
     the load-bearing phrases ("uniform zero mode", "Curie-Weiss"); the
     verify suite carries the MDL mean-field theorem entry.
  V7 Leg A locked (spectral leg only): the R1 panel ruled the COMPOSITION
     PARTIAL under K-R1-3 (see verdict block above); row stays 2.0.
"""
import os
import sys
from itertools import product

import numpy as np
from numpy import linalg as la

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
from proofs.common import find_bonds, ATOMS, A_PRIM  # noqa: E402

FAILURES = []
SQ2 = np.sqrt(2)


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


bonds = find_bonds()
EDGES = [(i, j, tuple(int(x) for x in c)) for (i, j, c) in bonds]
NE = len(EDGES)
E_INDEX = {e: a for a, e in enumerate(EDGES)}
REV = {a: E_INDEX[(j, i, tuple(-x for x in c))] for a, (i, j, c) in enumerate(EDGES)}
FOLLOW = {a: [b for b, (i2, j2, c2) in enumerate(EDGES)
              if i2 == EDGES[a][1] and b != REV[a]] for a in range(NE)}


def B_of(k):
    B = np.zeros((NE, NE), dtype=complex)
    for a in range(NE):
        for b in FOLLOW[a]:
            B[b, a] = np.exp(2j * np.pi * np.dot(k, np.asarray(EDGES[b][2], float)))
    return B


print("=" * 72)
print(" PHASE 5.2/R1 Leg A -- only the Perron mode carries a spatial mean")
print("=" * 72)

# V1: fiber statement at Gamma
B0 = B_of(np.zeros(3))
one = np.ones(NE)
ev0, V0 = la.eig(B0)
n_two = int(np.sum(np.abs(ev0 - 2.0) < 1e-9))
gate("V1 B(Gamma) 1 = 2*1 (right) and 1^T B = 2*1^T (left); eigenvalue 2 "
     "simple",
     la.norm(B0 @ one - 2 * one) < 1e-12
     and la.norm(one @ B0 - 2 * one) < 1e-12 and n_two == 1)

# V2: all other Gamma modes have zero mean
idx2 = int(np.argmin(np.abs(ev0 - 2.0)))
means = [abs(np.sum(V0[:, i])) / la.norm(V0[:, i])
         for i in range(NE) if i != idx2]
gate("V2 the 11 non-Perron Gamma modes: spatial mean = 0 (< 1e-12)",
     max(means) < 1e-12, f"max={max(means):.1e}")

# V3: explicit 4x4x4 torus, full real-space NB operator
L = 4
cells = list(product(range(L), repeat=3))
cidx = {c: i for i, c in enumerate(cells)}
NT = len(cells) * NE
BT = np.zeros((NT, NT))
for d in cells:
    for a in range(NE):
        d2 = tuple((d[m] + EDGES[a][2][m]) % L for m in range(3))
        col = cidx[d] * NE + a
        for b in FOLLOW[a]:
            BT[cidx[d2] * NE + b, col] = 1.0
evT, VT = la.eig(BT)
uni = np.ones(NT) / np.sqrt(NT)
i2 = int(np.argmin(np.abs(evT - 2.0)))
n_plus2 = int(np.sum(np.abs(evT - 2.0) < 1e-9))
vperron = VT[:, i2]
vperron = vperron / (np.sum(vperron) / abs(np.sum(vperron)))  # fix phase
align = abs(np.vdot(uni, VT[:, i2])) / la.norm(VT[:, i2])
other_means = [abs(np.sum(VT[:, i])) / la.norm(VT[:, i])
               for i in range(NT) if i != i2]
gate("V3 torus 4x4x4 (768 modes): exactly ONE eigenvalue +2; its mode IS "
     "the uniform vector; all 767 others zero-mean; spectral radius 2",
     n_plus2 == 1 and align > 1 - 1e-9 and max(other_means) < 1e-9
     and abs(np.max(np.abs(evT)) - 2.0) < 1e-9,
     f"align={align:.12f}, max other mean={max(other_means):.1e}")

# V4: rival families -- N-star phase sums vanish; N spectral radius sqrt2
kN_star = [np.array(p) for p in
           ([0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5],
            [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5])]
# the N star in primitive-fractional coords: all order-2 k vectors of the
# BCC reciprocal cell that fold to the cubic N star; phase-sum test below
# is the load-bearing statement (k != 0 on the torus => zero mean).
sums = [abs(sum(np.exp(2j * np.pi * (k @ np.asarray(d, float)))
              for d in cells)) for k in kN_star]
radN = np.max(np.abs(la.eigvals(B_of(A_PRIM @ np.array([0.0, 0.5, 0.5])))))
gate("V4 N-star torus phase sums = 0 exactly (h_N / h_N_neg modes have "
     "identically zero mean); bonus (N-SPECIFIC, panel annotation 5c: the "
     "H fiber reaches modulus 2, but H hosts no size-2 candidate family): "
     "max |lambda| at N = sqrt2 < 2",
     max(sums) < 1e-9 and abs(radN - SQ2) < 1e-9,
     f"max phase sum={max(sums):.1e}, rad(N)={radN:.6f}")

# V5 (panel-corrected justification, ordered 2026-06-11 R1 panel item 5a):
# the load-bearing statement is EXACT BIORTHOGONALITY, valid for all
# Jordan structure of the non-normal B_T:
#   1^T B_T = 2 * 1^T  (integer-exact: every column has exactly k*-1 = 2
#   ones), hence 1^T (B_T - lambda I) = (2 - lambda) 1^T, so for ANY
#   generalized eigenvector x at lambda != 2:
#   0 = 1^T (B_T - lambda I)^k x = (2 - lambda)^k 1^T x  =>  1^T x = 0.
# No appeal to numerical eigenvector completeness is needed. Simplicity
# of lambda = 2 is established fiber-exactly: spec(B_T) = union of the 64
# fiber spectra, and +2 occurs in exactly one fiber (Gamma), simply.
# (Fiber-level "constant phase" was the WRONG criterion -- first-run
# lesson: the antiperiod makes the H fiber's lambda = -2 mode uniform
# within its fiber while staggered in real space; recorded.)
M = (BT > 0)
R = np.eye(NT, dtype=bool) | M
for _ in range(10):                      # (I|M)^(2^10) covers diameter
    R = R @ R
col_sums = BT.sum(axis=0)
n_two_fibers = 0
for d in cells:
    kf = np.asarray(d, float) / L
    n_two_fibers += int(np.sum(np.abs(la.eigvals(B_of(kf)) - 2.0) < 1e-9))
gate("V5 exact biorthogonality: 1^T B_T = 2*1^T (integer-exact column "
     "sums) => 1^T x = 0 for every generalized eigenvector at lambda != 2 "
     "(all Jordan structure); lambda = 2 simple by fiber census (occurs "
     "in exactly one of 64 fibers); B_T irreducible (PF applies)",
     bool(R.all()) and np.all(col_sums == 2.0) and n_two_fibers == 1,
     f"fibers with +2: {n_two_fibers}")
print("  Load-bearing linear lemma (panel rewording 5b): nonzero spatial")
print("  mean <=> nonzero Gamma-Perron amplitude -- exact componentwise,")
print("  valid for complex/vector (doublet) fields. PF enters ONLY as")
print("  uniqueness of the non-negative eigendirection of B_T; entrywise")
print("  non-negativity of a doublet configuration is NOT gauge-meaningful")
print("  and is NOT load-bearing for the case.")

# V6: Leg-B tether
vh_path = os.path.join(ROOT, "predictions", "v_higgs_derivation.md")
verify_path = os.path.join(ROOT, "verify.py")
vh = open(vh_path).read() if os.path.exists(vh_path) else ""
vf = open(verify_path).read() if os.path.exists(verify_path) else ""
gate("V6 Leg-B tether: v_higgs_derivation.md carries the MDL "
     "uniform-zero-mode theorem; suite carries 'MDL mean-field uniquely "
     "optimal'",
     "uniform zero mode" in vh and "Curie-Weiss" in vh
     and "MDL mean-field uniquely optimal" in vf)

# V7: composition lock
gate("V7 LEG A ESTABLISHED (the spectral leg only; the COMPOSITION was "
     "ruled PARTIAL under K-R1-3 by the R1 panel -- row stays 2.0)",
     not FAILURES)

print("\n  NOTE: the R1 panel (2026-06-11) ruled PARTIAL: Leg A stands,")
print("  Leg B provenance stands, but the composition has named gaps")
print("  (mode-selection closure, vertex<->edge bridge, G3 pricing home).")
print("  The A5-mass row STAYS at 2.0; refile path recorded in the spec.")

print("\n" + "=" * 72)
if FAILURES:
    print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
    sys.exit(1)
print(" RESULT: ALL GATES PASS -- Leg A locked (panel verdict: PARTIAL; "
      "row stays 2.0)")
print("=" * 72)
sys.exit(0)
