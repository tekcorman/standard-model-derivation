#!/usr/bin/env python3
"""
proofs/foundations/MS1a_fusion_grading_2026-07-10.py

MS-1a -- THE FINITE-FUSION-RING NO-ADDITIVE-CHARGE THEOREM (matter stability, net side).
Milestone III.2, build-task MS-1a per
internal research notes sec 1.4.
This is the NON-vertex half only; the operator-classification half MS-1b is gated on the
interaction layer (I-0 RECONCILIATION / IV.4) and is NOT this station.

THEOREM (MS-1a).  Let G be the derived gauge sector group of the net: G = A4 (forced
J-covariance), with the Fock representation spinorial => the double cover 2T carrying the
fermion parity (derivation_topdown/state/the_net.py:656-717 gauge_sector_category(); HK-6
contract, derivation_topdown/adapters/aqft_net.py:265-277: species_sector_dims
{0:1,1:3,2:3,3:1}, double_cover_2T, sectors_are_species, fermion_parity {0:+1,1:-1,2:+1,3:-1}).
The superselection sectors as built are finitely many G-irreps, with the fusion ring of Rep(G).
THEN: any additive Z-valued charge q on this fusion ring -- additive in the sense that for
EVERY irreducible c appearing in the fusion product a (x) b, q(c) = q(a) + q(b) -- is
IDENTICALLY ZERO.  Hence no baryon-number-like unbounded additive conservation law exists at
the sector level, and exact-conservation protection of the proton is STRUCTURALLY IMPOSSIBLE
in the category as built.  This upgrades the assessment-grade inference of the scoping doc
sec 1.2 point 3(iv) ("a finite fusion ring cannot carry an additive Z-valued charge") to a
machine-checked theorem at the stated scope, and is CONSISTENT with the eta_B closure's
Sakharov skeleton, which REQUIRES effective B-violation
(docs/theorems/theorem_eta_B_substrate_sakharov_closure_2026-04-30.md).

MACHINE CHECKS (all computed, none asserted):
  (a) The A4 and 2T character tables are COMPUTED from the groups themselves (Dixon/Burnside
      class-matrix algorithm) and verified by full row+column orthogonality; the fusion /
      tensor-decomposition tables N_ab^c are computed from the characters and verified to be
      non-negative integers, commutative, associative, with unit and unique duals.  ALL Z2
      gradings of each fusion ring are enumerated by brute force (2^r sign assignments):
      R(A4) admits ONLY the trivial Z2 grading; R(2T) admits exactly ONE nontrivial Z2
      grading, and it coincides with the center grading s_a = chi_a(-1)/chi_a(1), i.e. the
      spinoriality / fermion-parity grading of the DR frame (F,2T).
  (b) ALL additive Z-valued charges are enumerated by solving the linear constraint system
      { q(a)+q(b)-q(c) = 0 : N_ab^c > 0 } EXACTLY over Q (Fraction Gaussian elimination):
      the solution space is {0} for R(A4), for R(2T), and for the fusion closure of the
      sectors as built.  THIS IS THE THEOREM, computed not asserted.
  (c) The winding-Z3 check: the deck-winding screw FAILS the gauge-charge vacuum-fixing test
      (<0|U_pi^2|0> = i/2 != 1, reproduced here by the identical construction as
      the_net.py:534-587 dr_frame_audit(); all 12 A4 lifts PASS the same test) => the winding
      adds no sectors (ML-2b verdict, the_net.py:535-538; corroborated FS5,
      derivation_topdown/adapters/furey_stoica_labels.py:455-460) => no charge can be
      smuggled in via the generation screw.  Belt-and-suspenders lemma (computed): even IF a
      Z3 grading were adjoined, the additive-Z-charge solution space of the Z3 group ring is
      {0} (torsion kills additivity).
  (d) DECLARED PREMISE (DR-frame qualifier): the finite-G premise -- that the sector category
      is Rep(G) for the FINITE group G = A4/2T and nothing bigger -- inherits ML-2b's
      conditionality on the THERMODYNAMIC-LIMIT twisted Haag duality
      (proofs/foundations/ML2b_dr_frame_2026-07-08.py:11-12,128-140; aqft_net.py HK-7 scope
      declaration: cell-level duality only is verified).  Printed below as a premise, per the
      checker's note in the scoping doc.

SCOPE GUARDS (binding):
  * This proves NO EXACT sector-level conservation law.  It does NOT prove the proton decays,
    does NOT estimate any rate or lifetime, and does NOT touch suppression magnitudes -- all
    of that is MS-1b's job (gated on the interaction layer; in the free theory nothing
    decays, vacuously).  BLOCKED-2/BLOCKED-3 of
    docs/_scratch/theorem_matter_stability_attempt.md stand; the exp(-girth/2) heuristic is
    NOT used.
  * HK-7 single-cell scope declared: the sector category is verified at the single-cell
    (cell-duality) scope; no intertwiner/braiding construction at general regions is built.
  * The lattice-level prior art (theorem_matter_stability_attempt.md Steps 1-3: I4_132
    chiral, no geometric Z2 R-parity, symmorphic C3 candidate) is COMPLEMENTARY lattice-side
    structure; this station is the FUSION-level argument and does not adjudicate BLOCKED-1
    (the spatial-C3 <-> triality identification).  Note the assessment's checker fixes:
    binding does NOT select color (BOUND_EP2 C4d NEGATIVE) -- one less candidate protection.
  * No goal-seek (no target value exists); no existing file edited; standalone against
    imports -- the the_net.py accretion of this ring is deferred to integration.

VERDICT (computed below): MS1a-THEOREM if every additive-charge solution space is {0};
MS1a-SURPRISE (reported exactly, not suppressed) if any nonzero additive grading exists.
"""
import itertools
import math
import os
import sys
from fractions import Fraction

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "state"))
import the_net as net  # noqa: E402  (the ONE master Layer-3 object; imported, not edited)
from simulator.srs_engine.utils import AlgebraicUtility  # noqa: E402

np.set_printoptions(precision=6, suppress=True)
ok_all = True


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


# ===========================================================================
# PART 0 -- DECLARED PREMISES (printed FIRST, per the checker's note; check (d))
# ===========================================================================
banner("MS-1a  PART 0 -- DECLARED PREMISES (the DR-frame qualifier; check (d))")
print("""  P1 (FINITE G, conditional): G = A4 with double cover 2T is the gauge group of the DR
     field-algebra frame (F, 2T).  ML-2b's DR-frame argument -- that the sector category is
     Rep(G) for THIS finite G and is NOT bigger -- is CONDITIONAL on the THERMODYNAMIC-LIMIT
     twisted Haag duality; only CELL-LEVEL duality is verified (ML0-4 / HK-5).  Sources:
     proofs/foundations/ML2b_dr_frame_2026-07-08.py:11-12 ("DR conclusions CONDITIONAL on the
     TD-limit duality"), :128-140; aqft_net.py HK-7 scope declaration (i).  Everything below
     inherits this conditionality: the theorem is about the category AS BUILT.
  P2 (SINGLE-CELL SCOPE): the sector category is machine-verified at the single-cell scope
     (HK-6); no charge transporters / braiding at general regions are built (HK-7 (ii)).
  P3 (BOOKED INPUTS, imported not re-derived): species_sector_dims {0:1,1:3,2:3,3:1},
     double_cover_2T, sectors_are_species, fermion_parity {0:+1,1:-1,2:+1,3:-1} (HK-6a-d);
     the winding is NOT a DHR charge (ML-2b, the_net.py:535-538).  The derived U(1) charge
     Q = N-hat/3 (the_run.py:255-256) lives on the FIELD algebra and is bounded on the cell
     (n <= 3); whether any Z-valued charge survives at SECTOR-fusion level is exactly what
     this station computes.""")


# ===========================================================================
# GROUP MACHINERY (generic; both groups built from scratch, nothing imported)
# ===========================================================================
def group_A4():
    """A4 as the even permutations of {0,1,2,3} (same convention as the_net.py:689-690)."""
    elems = [p for p in itertools.permutations(range(4))
             if sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j]) % 2 == 0]
    mul = lambda p, q: tuple(p[q[i]] for i in range(4))   # (p o q)(i) = p(q(i)), the_net comp()

    def inv(p):
        v = [0] * 4
        for i, pi in enumerate(p):
            v[pi] = i
        return tuple(v)
    return elems, mul, inv, (0, 1, 2, 3)


def group_2T():
    """2T (binary tetrahedral) as the 24 unit Hurwitz quaternions, doubled-integer coords
    (2a,2b,2c,2d) so all arithmetic is EXACT.  Generated by closure from i and (1+i+j+k)/2."""
    def mul(x, y):
        a0, a1, a2, a3 = x
        b0, b1, b2, b3 = y
        z = (a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
             a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
             a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
             a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0)
        assert all(t % 2 == 0 for t in z), "non-Hurwitz product"
        return tuple(t // 2 for t in z)

    e = (2, 0, 0, 0)
    elems, frontier = {e}, [e]
    gens = [(0, 2, 0, 0), (1, 1, 1, 1)]
    while frontier:
        new = []
        for g in frontier:
            for h in gens:
                x = mul(g, h)
                if x not in elems:
                    elems.add(x)
                    new.append(x)
        frontier = new
    inv = lambda x: (x[0], -x[1], -x[2], -x[3])           # unit quaternion inverse = conjugate
    return sorted(elems), mul, inv, e


def elt_order(g, mul, e):
    n, x = 1, g
    while x != e:
        x = mul(x, g)
        n += 1
    return n


def conjugacy_classes(elems, mul, inv, e):
    """Classes ordered: identity first, then by (element order, class size, representative)."""
    seen, classes = set(), []
    for g in elems:
        if g in seen:
            continue
        cl = sorted({mul(mul(h, g), inv(h)) for h in elems})
        classes.append(cl)
        seen |= set(cl)
    classes.sort(key=lambda c: (elt_order(c[0], mul, e), len(c), c[0]))
    assert classes[0] == [e]
    return classes


def character_table(elems, mul, inv, e, label):
    """Dixon/Burnside: the class-sum multiplication matrices (A_i)_{jk} = a_ij^k commute and
    their common eigenvectors are the central characters w_a(Z_i) = |C_i| chi_a(g_i)/chi_a(1);
    dims from the norm relation.  Returns (classes, sizes, cls_of, X, dims); X[a][j] = chi_a
    on class j.  Everything is VERIFIED downstream by orthogonality -- computed, not asserted."""
    G = len(elems)
    classes = conjugacy_classes(elems, mul, inv, e)
    r = len(classes)
    sizes = [len(c) for c in classes]
    cls_of = {g: i for i, c in enumerate(classes) for g in c}
    A = np.zeros((r, r, r))
    for i in range(r):
        for k in range(r):
            zk = classes[k][0]
            for x in classes[i]:
                A[i, cls_of[mul(inv(x), zk)], k] += 1.0   # x*y = z_k with y = x^-1 z_k
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23][:r]
    M = sum(math.sqrt(p) * A[i] for i, p in enumerate(primes))
    evals, evecs = np.linalg.eig(M)
    gap = min(abs(evals[i] - evals[j]) for i in range(r) for j in range(i + 1, r))
    assert gap > 1e-6, f"{label}: degenerate Dixon combination (gap={gap:.2e})"
    X, dims = [], []
    for a in range(r):
        w = evecs[:, a] / evecs[0, a]                      # w_0 = w(Z_e) = 1
        d = math.sqrt(G / sum(abs(w[j]) ** 2 / sizes[j] for j in range(r)))
        X.append([w[j] * d / sizes[j] for j in range(r)])
        dims.append(d)
    order = sorted(range(r), key=lambda a: (round(dims[a], 6),
                                            tuple((round(X[a][j].real, 6), round(X[a][j].imag, 6))
                                                  for j in range(r))))
    X = [X[a] for a in order]
    dims = [dims[a] for a in order]
    triv = next(a for a in range(r) if all(abs(X[a][j] - 1) < 1e-8 for j in range(r)))
    perm = [triv] + [a for a in range(r) if a != triv]
    return classes, sizes, cls_of, [X[a] for a in perm], [dims[a] for a in perm]


def fusion_table(X, sizes, G):
    """N[a,b,c] = <chi_a chi_b, chi_c> = (1/|G|) sum_j |C_j| chi_a chi_b conj(chi_c)."""
    r = len(X)
    N = np.zeros((r, r, r))
    for a in range(r):
        for b in range(r):
            for c in range(r):
                N[a, b, c] = sum(sizes[j] * (X[a][j] * X[b][j] * np.conj(X[c][j])).real
                                 for j in range(r)) / G
    return N


def verify_ring(tag, X, sizes, G, N, names):
    """Full verification battery: character orthogonality (rows+columns), integer dims,
    sum d^2 = |G|; fusion integrality/non-negativity, unit, commutativity, associativity,
    unique duals, dimension homomorphism."""
    r = len(X)
    row = max(abs(sum(sizes[j] * X[a][j] * np.conj(X[b][j]) for j in range(r)) / G
                  - (1 if a == b else 0)) for a in range(r) for b in range(r))
    col = max(abs(sum(X[a][i] * np.conj(X[a][j]) for a in range(r))
                  - (G / sizes[i] if i == j else 0)) for i in range(r) for j in range(r))
    check(f"{tag}: character rows orthonormal <chi_a,chi_b>=delta_ab", row < 1e-9,
          detail=f"max dev {row:.2e}")
    check(f"{tag}: character columns orthogonal (completeness)", col < 1e-8,
          detail=f"max dev {col:.2e}")
    dims = [X[a][0].real for a in range(r)]
    check(f"{tag}: dims are positive integers, sum d^2 = |G| = {G}",
          all(abs(d - round(d)) < 1e-9 and round(d) >= 1 for d in dims)
          and abs(sum(d * d for d in dims) - G) < 1e-6,
          detail=f"dims = {[int(round(d)) for d in dims]}")
    Nint = np.round(N).astype(int)
    check(f"{tag}: fusion coefficients are non-negative integers",
          float(np.max(np.abs(N - Nint))) < 1e-7 and int(Nint.min()) >= 0,
          detail=f"max dev from integer {float(np.max(np.abs(N - Nint))):.2e}")
    unit_ok = all(Nint[0, a, c] == (1 if a == c else 0) for a in range(r) for c in range(r))
    check(f"{tag}: irrep '{names[0]}' is the fusion unit", unit_ok)
    check(f"{tag}: fusion commutative N_ab^c = N_ba^c",
          bool(np.all(Nint == np.transpose(Nint, (1, 0, 2)))))
    assoc = max(abs(int(sum(Nint[a, b, e_] * Nint[e_, c, d] for e_ in range(r))
                        - sum(Nint[b, c, f] * Nint[a, f, d] for f in range(r))))
                for a in range(r) for b in range(r) for c in range(r) for d in range(r))
    check(f"{tag}: fusion associative ((a x b) x c = a x (b x c))", assoc == 0,
          detail=f"max |mismatch| = {assoc}")
    duals_ok = all(int(np.sum(Nint[a, :, 0])) == 1 for a in range(r))
    check(f"{tag}: unique duals (N_ab^1 = delta_b,a*)", duals_ok)
    dh = max(abs(sum(Nint[a, b, c] * dims[c] for c in range(r)) - dims[a] * dims[b])
             for a in range(r) for b in range(r))
    check(f"{tag}: dimension homomorphism sum_c N_ab^c d_c = d_a d_b", dh < 1e-6,
          detail=f"max dev {dh:.2e}")
    return Nint, [int(round(d)) for d in dims]


def name_irreps(dims):
    out, seen = [], {}
    for d in dims:
        k = seen.get(d, 0)
        out.append(str(d) + "'" * k)
        seen[d] = k + 1
    return out


def fusion_str(Nint, names, a, b):
    parts = []
    for c in range(len(names)):
        m = Nint[a, b, c]
        if m == 1:
            parts.append(names[c])
        elif m > 1:
            parts.append(f"{m}*{names[c]}")
    return " + ".join(parts)


def z2_gradings(Nint):
    """ALL maps s: Irr -> {+1,-1} with s(a)s(b) = s(c) whenever N_ab^c > 0 (brute force 2^r)."""
    r = Nint.shape[0]
    triples = [(a, b, c) for a in range(r) for b in range(r) for c in range(r) if Nint[a, b, c] > 0]
    return [s for s in itertools.product([1, -1], repeat=r)
            if all(s[a] * s[b] == s[c] for a, b, c in triples)], len(triples)


def additive_charge_space(Nint):
    """EXACT solution space of { q(a)+q(b)-q(c) = 0 : N_ab^c > 0 } over Q (Fraction rref).
    Returns (n_constraints, rank, nullity, basis).  Nullity 0 <=> the ONLY additive Z-valued
    charge is q = 0 (a rational nullspace of dim 0 has no nonzero integer points either)."""
    r = Nint.shape[0]
    rows = []
    for a in range(r):
        for b in range(r):
            for c in range(r):
                if Nint[a, b, c] > 0:
                    v = [Fraction(0)] * r
                    v[a] += 1
                    v[b] += 1
                    v[c] -= 1
                    rows.append(v)
    n_con = len(rows)
    M = [row[:] for row in rows]
    pivots, ri = [], 0
    for col in range(r):
        piv = next((i for i in range(ri, len(M)) if M[i][col] != 0), None)
        if piv is None:
            continue
        M[ri], M[piv] = M[piv], M[ri]
        pv = M[ri][col]
        M[ri] = [x / pv for x in M[ri]]
        for i in range(len(M)):
            if i != ri and M[i][col] != 0:
                f = M[i][col]
                M[i] = [x - f * y for x, y in zip(M[i], M[ri])]
        pivots.append(col)
        ri += 1
        if ri == len(M):
            break
    rank = len(pivots)
    basis = []
    for fc in [c for c in range(r) if c not in pivots]:
        v = [Fraction(0)] * r
        v[fc] = Fraction(1)
        for i, pc in enumerate(pivots):
            v[pc] = -M[i][fc]
        basis.append(v)
    return n_con, rank, r - rank, basis


# ===========================================================================
# PART A -- the two candidate gauge groups, character tables COMPUTED + VERIFIED  (check (a))
# ===========================================================================
banner("MS-1a  PART A -- A4 and 2T built from scratch; character tables COMPUTED (Dixon)")

A4_e, A4_mul, A4_inv, A4_id = group_A4()
T2_e, T2_mul, T2_inv, T2_id = group_2T()
check("A4 has 12 elements (even permutations of 4)", len(A4_e) == 12)
check("2T closure gives exactly 24 unit quaternions", len(T2_e) == 24)
minus1 = (-2, 0, 0, 0)
check("2T contains the central -1 (order 2)", minus1 in T2_e
      and elt_order(minus1, T2_mul, T2_id) == 2)
Z2T = [g for g in T2_e if all(T2_mul(g, h) == T2_mul(h, g) for h in T2_e)]
check("Z(2T) = {+1, -1}  (the Z2 center that will carry the fermion parity)",
      sorted(Z2T) == sorted([T2_id, minus1]), detail=f"center = {Z2T}")
ZA4 = [g for g in A4_e if all(A4_mul(g, h) == A4_mul(h, g) for h in A4_e)]
check("Z(A4) trivial (no center => no center grading at the A4 level)", ZA4 == [A4_id])

A4_cls, A4_sz, A4_cof, A4_X, A4_d = character_table(A4_e, A4_mul, A4_inv, A4_id, "A4")
T2_cls, T2_sz, T2_cof, T2_X, T2_d = character_table(T2_e, T2_mul, T2_inv, T2_id, "2T")
check("A4: 4 conjugacy classes (sizes 1,3,4,4)", sorted(A4_sz) == [1, 3, 4, 4],
      detail=f"sizes = {A4_sz}")
check("2T: 7 conjugacy classes (sizes 1,1,4,4,4,4,6)", sorted(T2_sz) == [1, 1, 4, 4, 4, 4, 6],
      detail=f"sizes = {T2_sz}")

NA4 = fusion_table(A4_X, A4_sz, 12)
N2T = fusion_table(T2_X, T2_sz, 24)
A4_N, A4_dims = verify_ring("A4", A4_X, A4_sz, 12, NA4, name_irreps([int(round(x[0].real)) for x in A4_X]))
T2_N, T2_dims = verify_ring("2T", T2_X, T2_sz, 24, N2T, name_irreps([int(round(x[0].real)) for x in T2_X]))
A4_names = name_irreps(A4_dims)
T2_names = name_irreps(T2_dims)
check("A4 irrep dims = [1,1,1,3]", sorted(A4_dims) == [1, 1, 1, 3], detail=f"{A4_dims}")
check("2T irrep dims = [1,1,1,2,2,2,3] (three genuinely spinorial 2-dims)",
      sorted(T2_dims) == [1, 1, 1, 2, 2, 2, 3], detail=f"{T2_dims}")

print("\n  A4 fusion table (full):")
for a in range(4):
    for b in range(a, 4):
        print(f"    {A4_names[a]} x {A4_names[b]} = {fusion_str(A4_N, A4_names, a, b)}")
print("\n  2T fusion table (full):")
for a in range(7):
    for b in range(a, 7):
        print(f"    {T2_names[a]} x {T2_names[b]} = {fusion_str(T2_N, T2_names, a, b)}")

# the center grading of 2T (chi_a(-1)/chi_a(1)): the spinoriality sign
j_m1 = T2_cof[minus1]
T2_parity = tuple(int(round((T2_X[a][j_m1] / T2_X[a][0]).real)) for a in range(7))
check("2T center element -1 acts as +-1 in every irrep (chi(-1)/chi(1) in {+1,-1})",
      all(abs(T2_X[a][j_m1] / T2_X[a][0] - T2_parity[a]) < 1e-9 for a in range(7)),
      detail=f"center grading = {dict(zip(T2_names, T2_parity))}")
check("center grading is -1 EXACTLY on the three 2-dim irreps (spinorial = fermionic)",
      all((T2_parity[a] == -1) == (T2_dims[a] == 2) for a in range(7)))

# the even (center-trivial) sub-ring of R(2T) is R(A4)  (the A4 pullback, computed)
even = [a for a in range(7) if T2_parity[a] == +1]
closed = all(T2_N[a, b, c] == 0 for a in even for b in even for c in range(7) if c not in even)
check("even sub-ring of R(2T) is fusion-closed (dims {1,1,1,3})",
      closed and sorted(T2_dims[a] for a in even) == [1, 1, 1, 3])
iso_found = 0
for perm in itertools.permutations(range(4)):
    f = {even[i]: perm[i] for i in range(4)}
    if any(T2_dims[a] != A4_dims[f[a]] for a in even):
        continue
    if all(T2_N[a, b, c] == A4_N[f[a], f[b], f[c]] for a in even for b in even for c in even):
        iso_found += 1
check("even sub-ring of R(2T) ~= R(A4) as fusion rings (explicit isomorphism found)",
      iso_found >= 1, detail=f"{iso_found} dim-preserving fusion isomorphisms")


# ===========================================================================
# PART B -- the sectors AS BUILT are G-irreps of this ring (computed from the master object)
# ===========================================================================
banner("MS-1a  PART B -- sector <-> irrep map COMPUTED from the net's own J6 / edge rep")

J6 = net.complex_structure_J6()
wJ, VJ = np.linalg.eig(J6)
modes, _ = np.linalg.qr(VJ[:, np.where(np.abs(wJ - 1j) < 1e-9)[0]])
R6 = {p: net._edge_rep(dict(enumerate(p))) for p in A4_e}
covJ = max(np.max(np.abs(R6[p] @ J6 - J6 @ R6[p])) for p in A4_e)
check("every A4 edge rep commutes with J6 (A4-covariance of the complex structure)",
      covJ < 1e-9, detail=f"max ||[R6,J6]|| = {covJ:.2e}")
V = {p: modes.conj().T @ R6[p] @ modes for p in A4_e}      # the 3-mode rep of A4
uni = max(np.max(np.abs(V[p].conj().T @ V[p] - np.eye(3))) for p in A4_e)
hom = max(np.max(np.abs(V[A4_mul(p, q)] - V[p] @ V[q])) for p in A4_e for q in A4_e)
check("mode rep V_g is unitary and a GENUINE homomorphism V_gh = V_g V_h (no cocycle)",
      uni < 1e-9 and hom < 1e-9, detail=f"unitarity {uni:.2e}, hom defect {hom:.2e}")

# sector characters: sector n = Lambda^n(mode rep); chi_n(g) = e_n(spec V_g)
def sector_char(p):
    lam = np.linalg.eigvals(V[p])
    e1 = lam.sum()
    e2 = (e1 ** 2 - (lam ** 2).sum()) / 2
    e3 = np.linalg.det(V[p])
    return [1.0 + 0j, e1, e2, e3]

mult = np.zeros((4, 4))
for p in A4_e:
    sc = sector_char(p)
    for n in range(4):
        for a in range(4):
            mult[n, a] += (sc[n] * np.conj(A4_X[a][A4_cof[p]])).real / 12.0
mult_int = np.round(mult).astype(int)
check("sector characters decompose with INTEGER multiplicities",
      float(np.max(np.abs(mult - mult_int))) < 1e-7,
      detail=f"max dev {float(np.max(np.abs(mult - mult_int))):.2e}")
sector_irrep = {}
ok_irr = True
for n in range(4):
    nz = [a for a in range(4) if mult_int[n, a] > 0]
    ok_irr &= (len(nz) == 1 and mult_int[n, nz[0]] == 1)
    sector_irrep[n] = nz[0]
check("each sector n=0..3 is a SINGLE G-irrep (multiplicity 1)", ok_irr,
      detail=f"sector -> irrep: { {n: A4_names[sector_irrep[n]] for n in range(4)} }")
check("sector irreps = {nu: 1, d: 3, u: 3, e: 1} (dims match HK-6a {0:1,1:3,2:3,3:1})",
      [A4_dims[sector_irrep[n]] for n in range(4)] == [1, 3, 3, 1])
check("n=0 and n=3 both carry the TRIVIAL irrep => n (hence N-hat, hence 3Q) is NOT a "
      "function on the fusion ring; only torsion data (parity) can descend",
      sector_irrep[0] == 0 and sector_irrep[3] == 0)

# regression: the master object's own booked HK-6 fields
sc_ = net.gauge_sector_category()
check("REGRESSION HK-6a-d: net.gauge_sector_category() booked fields reproduce",
      sc_["species_sector_dims"] == {0: 1, 1: 3, 2: 3, 3: 1} and sc_["double_cover_2T"] is True
      and sc_["sectors_are_species"] is True
      and sc_["fermion_parity"] == {0: 1, 1: -1, 2: 1, 3: -1},
      detail=f"{sc_}")

# the fusion CLOSURE of the sectors as built = the full ring R(A4)
S = {sector_irrep[n] for n in range(4)}
while True:
    S2 = set(S) | {c for a in S for b in S for c in range(4) if A4_N[a, b, c] > 0}
    if S2 == S:
        break
    S = S2
check("fusion closure of the sectors as built = ALL of Irr(A4) (3 x 3 produces 1', 1'')",
      S == {0, 1, 2, 3},
      detail=f"closure = {sorted(A4_names[a] for a in S)}")


# ===========================================================================
# PART C -- ALL Z2 gradings, enumerated  (check (a) continued)
# ===========================================================================
banner("MS-1a  PART C -- ALL Z2 gradings of the fusion rings (brute-force enumeration)")

gr_A4, tri_A4 = z2_gradings(A4_N)
gr_2T, tri_2T = z2_gradings(T2_N)
check(f"R(A4): Z2 gradings found = 1 (the trivial one ONLY; {tri_A4} fusion constraints)",
      len(gr_A4) == 1 and gr_A4[0] == (1, 1, 1, 1),
      detail=f"gradings = {gr_A4}")
check(f"R(2T): Z2 gradings found = 2 (trivial + exactly ONE nontrivial; {tri_2T} constraints)",
      len(gr_2T) == 2, detail=f"gradings = {gr_2T}")
nontriv = [s for s in gr_2T if s != tuple([1] * 7)]
check("the unique nontrivial Z2 grading of R(2T) == the center grading chi(-1)/chi(1) "
      "== the spinoriality/FERMION-PARITY grading of the DR frame (F,2T)",
      len(nontriv) == 1 and nontriv[0] == T2_parity,
      detail=f"grading = {dict(zip(T2_names, nontriv[0])) if nontriv else '--'}")
print("""  READING: the booked HK-6d sector parity {0:+1,1:-1,2:+1,3:-1} = (-1)^n is the Klein
  twist of the Fock REALIZATION (statistics grading; N-hat mod 2).  At the fusion-ring level
  the ONLY Z2 grading available anywhere in the frame is the 2T center grading (spinoriality)
  -- computed above to be the unique nontrivial one.  Fermion parity is therefore the ONE AND
  ONLY Z2 sector grading the category supports; there is no room for a second, R-parity-like
  Z2 at sector level (consistent with the lattice-side Step 2 of
  docs/_scratch/theorem_matter_stability_attempt.md: no geometric Z2 R-parity either).""")


# ===========================================================================
# PART D -- THE THEOREM: all additive Z-valued charges, solved exactly  (check (b))
# ===========================================================================
banner("MS-1a  PART D -- THE THEOREM: additive Z-charge solution spaces (exact, over Q)")

results = {}
for tag, Nint, names in (("R(A4)  [= fusion closure of the sectors as built]", A4_N, A4_names),
                         ("R(2T)  [the DR-frame (F,2T) reading]", T2_N, T2_names)):
    n_con, rank, nullity, basis = additive_charge_space(Nint)
    results[tag] = (n_con, rank, nullity, basis)
    check(f"{tag}: additive-charge solution space = {{0}}  "
          f"({n_con} constraints, rank {rank}, nullity {nullity})",
          nullity == 0,
          detail="q == 0 is the ONLY additive Z-valued charge" if nullity == 0
          else f"NONZERO SOLUTION BASIS: {[[str(x) for x in v] for v in basis]}")

# the human-readable forcing chain for R(A4) (each fact read off the computed table)
i3 = A4_dims.index(3)
check("forcing chain, R(A4): 3 x 3 contains 3 itself  =>  q(3)+q(3)=q(3)  =>  q(3)=0",
      A4_N[i3, i3, i3] >= 1)
check("forcing chain, R(A4): 3 x 3 contains 1  =>  q(1) = 2 q(3) = 0",
      A4_N[i3, i3, 0] >= 1)
ones = [a for a in range(4) if A4_dims[a] == 1 and a != 0]
check("forcing chain, R(A4): 1' x 1' = 1'' and 1' x 1'' = 1  =>  3 q(1') = 0  =>  "
      "q(1')=q(1'')=0 (torsion)",
      A4_N[ones[0], ones[0], ones[1]] == 1 and A4_N[ones[0], ones[1], 0] == 1)
print("""  WHY (structural): a baryon-number-like law needs an INFINITE pointed direction in the
  sector lattice (a Z-graded ladder, as in a compact-group U(1) factor).  A FINITE gauge
  group's fusion ring has none: every object sits in a fusion loop (self-fusion 3 x 3 ∋ 3,
  torsion 1'^3 = 1), and every loop forces q = 0.  The general statement is
  Gelaki-Nikshych universal-grading: U(Rep(G)) = the character group of Z(G), FINITE for
  finite G, and Hom(finite group, Z) = 0.  Here it is COMPUTED, not cited.""")

# belt-and-suspenders: a Z3 group-ring (the winding, IF it were adjoined) carries none either
NZ3 = np.zeros((3, 3, 3), dtype=int)
for a in range(3):
    for b in range(3):
        NZ3[a, b, (a + b) % 3] = 1
n_con3, rank3, null3, _ = additive_charge_space(NZ3)
check("LEMMA (winding belt-and-suspenders): the Z3 group ring's additive-charge space = {0} "
      "(torsion kills Z-additivity even IF the deck winding were adjoined as a charge)",
      null3 == 0, detail=f"{n_con3} constraints, rank {rank3}")


# ===========================================================================
# PART E -- the winding-Z3 gauge-charge test  (check (c); ML-2b reproduced, cited)
# ===========================================================================
banner("MS-1a  PART E -- the winding screw FAILS the gauge-charge test (the_net.py:535-538)")

# identical construction to the_net.dr_frame_audit() (verbatim path; imported inputs)
g6 = [np.array(g) for g in AlgebraicUtility.cl6_generators()]
I8 = np.eye(8)
EIDX = {(min(i, j), max(i, j)): e for e, (i, j, v) in enumerate(net.EDGES)}
gam = lambda x: sum(x[a] * g6[a] for a in range(net.NE))


def spin_lift(R):
    rowsU = [np.kron(gam(R[:, a]), I8) - np.kron(I8, g6[a].T) for a in range(net.NE)]
    _, s, Vh = np.linalg.svd(np.vstack(rowsU))
    M = Vh[np.sum(s > 1e-9):].conj()[0].reshape(8, 8)
    return M / np.sqrt(np.abs(np.linalg.det(M @ M.conj().T)) ** (1 / 8))


A_ops = [gam(np.conj(modes[:, m])) / math.sqrt(2) for m in range(3)]
NHAT = sum(a.conj().T @ a for a in A_ops)
wN, VN = np.linalg.eigh(NHAT)
vac = VN[:, [int(np.argmin(wN))]]
check("N-hat spectrum = Hamming weights {0,1,2,3} with dims {1,3,3,1} (the sectors)",
      sorted(np.round(wN).astype(int).tolist()) == [0, 1, 1, 1, 2, 2, 2, 3])
sig3 = {0: 0, 1: 2, 2: 3, 3: 1}
Rpi = np.zeros((net.NE, net.NE))
for e_, (i, j, v) in enumerate(net.EDGES):
    a, b = sig3[i], sig3[j]
    Rpi[EIDX[(min(a, b), max(a, b))], e_] = 1.0
Upi2 = spin_lift(Rpi) @ spin_lift(Rpi)
z = complex((vac.conj().T @ Upi2 @ vac).item())
check("winding fails vacuum-fixing: |<0|U_pi^2|0>| = 1/2 != 1 (a gauge charge would give 1)",
      abs(abs(z) - 0.5) < 1e-6, detail=f"<0|U_pi^2|0> = {z:.6f}")
check("booked complex value reproduces: <0|U_pi^2|0> = i/2 (ML-2b / W2 chiral seed)",
      abs(z - 0.5j) < 1e-6, detail=f"|z - i/2| = {abs(z - 0.5j):.2e}")
comm = float(np.max(np.abs(Upi2 @ NHAT - NHAT @ Upi2)))
check("winding ALSO fails the second gauge-test component: [U_pi^2, N-hat] != 0 "
      "(dr_frame_audit demands < 1e-6; a gauge action must preserve the sectors)",
      comm > 1e-3, detail=f"max |[U_pi^2, N-hat]| = {comm:.3f} -- BOTH components fail")
vac_fix_A4 = max(abs(1.0 - abs(complex((vac.conj().T @ spin_lift(R6[p]) @ vac).item())))
                 for p in A4_e)
check("contrast: ALL 12 A4 lifts FIX the vacuum ray, |<0|U_g|0>| = 1 (phase-free test)",
      vac_fix_A4 < 1e-6, detail=f"max |1-|<0|U_g|0>|| = {vac_fix_A4:.2e}")
dr = net.dr_frame_audit()
check("REGRESSION ML-2b: net.dr_frame_audit() => winding_is_gauge=False, frame_forced=True",
      dr["winding_is_gauge"] is False and dr["frame_forced"] is True,
      detail=f"weld_bits = {dr['weld_bits']:.4f} (the unpaid H(w|t)=1.63 weld, UNTOUCHED here)")
print("""  => the winding/deck Z3 adds NO sectors (ML-2b verdict, conditional on TD-limit duality,
  premise P1) -- so the fusion rings of PART A/D are the WHOLE sector story in this frame, and
  no baryon-like charge can ride in on the generation screw.  This actively retires the old
  predictions.md 'Z3 triality (generation)' protection story (already flagged OVERSTATED in
  docs/_scratch/theorem_matter_stability_attempt.md:159-166).""")


# ===========================================================================
# PART F -- VERDICT + physics reading + scope guards
# ===========================================================================
banner("MS-1a  VERDICT")

nullities = [results[k][2] for k in results] + [null3]
surprise = any(n > 0 for n in nullities)
if surprise:
    print("  VERDICT: MS1a-SURPRISE -- a NONZERO additive grading exists (reported exactly")
    print("  above, not suppressed).  This would be a major finding: book it.")
elif ok_all:
    print("""  VERDICT: MS1a-THEOREM.
    Computed, on the sector category as built (premises P1-P3):
      * Z2 gradings: R(A4) has ONLY the trivial grading; R(2T) has EXACTLY ONE nontrivial
        grading = the center/spinoriality grading = the fermion parity.  No R-parity-like
        second Z2 exists at sector level.
      * Additive Z-valued charges: the solution space is {0} for R(A4) (= the fusion closure
        of the sectors as built), for R(2T) (the DR-frame reading), and for the Z3 group ring
        (the winding, even if adjoined).  q == 0 identically.
    PHYSICS RESULT: no baryon-number-like unbounded additive conservation law exists at the
    sector level of the category as built => exact-conservation protection of the proton is
    STRUCTURALLY IMPOSSIBLE here.  Matter stability, if it holds, must be a SUPPRESSION
    statement about the interaction layer (MS-1b, gated on I-0), exactly as in the Standard
    Model (B accidental, 't Hooft-violated, dim-6-suppressed).
    IMPLICATION: the framework's eta_B closure REQUIRES B-violation (Sakharov condition 1);
    this theorem shows the sector layer puts up no exact obstruction -- the booked
    eta_B = 6.112e-10 (-0.20 sigma) row and matter stability are CONSISTENT, not in tension.
    PARAMETER IMPACT: none (structural theorem; no scoreboard value moves).  Ledger row
    'Matter stability' stays PARTIALLY-FORCED-GAP-NAMED, with the scoping doc's sec 1.2
    point 3(iv) inference UPGRADED from assessment-grade to machine-checked THEOREM at the
    stated scope.""")
else:
    print("  VERDICT: MS1a-INCOMPLETE -- at least one machine check FAILED (see above; a")
    print("  finding, not a bug to massage).")

print("""
  SCOPE GUARDS (restated): proves NO EXACT sector-level conservation only; does NOT prove
  the proton decays; NO rate/lifetime/suppression magnitude computed (BLOCKED-2/3 stand;
  exp(-girth/2) NOT used); single-cell HK-7 scope; finite-G premise CONDITIONAL on TD-limit
  duality (P1); winding NOT promoted to a charge; the lattice C3 <-> triality identification
  (BLOCKED-1) NOT adjudicated; the_net.py accretion deferred to integration (standalone).""")
print("RESULT:", ("MS1a-SURPRISE (nonzero additive grading -- see basis above)" if surprise
                  else ("MS1a-THEOREM (all solution spaces {0}; all checks pass)" if ok_all
                        else "MS1a-INCOMPLETE (a machine check failed)")))
sys.exit(0 if (ok_all and not surprise) else 1)
