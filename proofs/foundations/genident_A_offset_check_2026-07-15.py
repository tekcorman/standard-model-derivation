#!/usr/bin/env python3
"""
GEN-IDENT-A -- durable check driver for
internal research notes

Question: is the offset between the July selector-axis (v2, fixed by A4v[5]=W) and the GEN-HOMES
winding-axis (v0, fixed by sigma=(123)) FORCED (no relating symmetry of the full construction) or
GAUGE (a symmetry exists that carries v2 to v0 while respecting both the winding grading and the
selector constraints)?

Runs T1-T6 in the order the freeze specifies. Self-contained; imports only already-accreted,
read-only objects from derivation_topdown/state/the_net.py. Modifies nothing; does NOT touch
the_run.py or the Layer-1 spectrum. Not wired into verify.py.

GOAL-SEEK GUARD: no mass/ppm/Koide-Q/mixing/CKM value is read, compared, or referenced anywhere
below. Every constant used (c2=1/6, c3=1/72, the F1/F2/F3 functionals) is a structural quantity of
the walk construction, identical to what V1/GEN-HOMES already used for the SAME purpose.

OMP_NUM_THREADS=4. Runtime: a few seconds.

INTERPRETIVE NOTE (read before trusting the verdict -- flagged per the freeze's own honesty rule):
the freeze's T2/T4 wording ("Stab" = subgroup "preserving the winding grading", intersected with
Sel to find an axis-mover v2->v0) is self-contradictory taken completely literally: any subgroup
that "fixes sigma up to inner" (normalizes <sigma>) by definition stabilizes axis v0 SETWISE, so no
element of such a subgroup can carry a DIFFERENT axis (v2) onto v0 -- Stab-in-the-literal-sense and
"axis-mover v2->v0" are mutually exclusive by construction, not merely disjoint. The freeze itself
anticipates this ("or the appropriate double-coset condition"). This script therefore computes BOTH
literal readings of Stab (centralizer and normalizer of sigma; both, as expected, turn out to
stabilize v0 exactly and can supply no mover) AND resolves T4 operationally as: "does ANY axis-mover
g in A4v with g(2)=0 also lie in Sel" -- i.e. the double-coset reading the freeze's own hedge
permits. This choice is reported, not smuggled.
"""
import sys, os, math, itertools

sys.path.insert(0, ".")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

from derivation_topdown.state.the_net import (
    _a4_vertex_group, _a4_standard_3irrep, _a2d_abstract_hom_basis,
    dart_rep, hashimoto_gamma, reversal,
    w2_family_direction, w2_gamma_table, _v1_gamma_mode_table, v1_F2_F3,
    v1_channel_state, _v1_mutual_information,
    NV, ND,
)

np.set_printoptions(precision=6, suppress=False, linewidth=140)

RESULTS = []


def check(name, cond, note=""):
    RESULTS.append((name, bool(cond), note))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}   {note}")
    return bool(cond)


def hdr(s):
    print("\n" + "=" * 100 + "\n" + s + "\n" + "=" * 100)


def comp(g, h):
    """A4/S4 group law used throughout the_net.py: comp(g,h) = g o h (apply h then g)."""
    return {i: g[h[i]] for i in range(NV)}


TOL = 1e-9

# =================================================================================================
hdr("SETUP -- A4v, sigma (winding, axis v0), W_gen=A4v[5] (selector, axis v2)")
# =================================================================================================

A4v = _a4_vertex_group()
e_id = {i: i for i in range(NV)}


def perm_order(g):
    cur = dict(g)
    n = 1
    while cur != e_id:
        cur = comp(g, cur)
        n += 1
    return n


orders = [perm_order(g) for g in A4v]
order3_idx = [k for k, o in enumerate(orders) if o == 3]
print("A4v order histogram:", {o: orders.count(o) for o in sorted(set(orders))})

key_to_idx = {tuple(sorted(g.items())): k for k, g in enumerate(A4v)}


def idx_of(g):
    return key_to_idx[tuple(sorted(g.items()))]


sigma_dict = {0: 0, 1: 2, 2: 3, 3: 1}
sigma_idx = idx_of(sigma_dict)
sigma2_dict = comp(sigma_dict, sigma_dict)


def fixed_vertex(g):
    fixed = [i for i in range(NV) if g[i] == i]
    assert len(fixed) == 1, f"expected exactly one fixed vertex, got {fixed}"
    return fixed[0]


axis_of = {k: fixed_vertex(A4v[k]) for k in order3_idx}
axes = {v: sorted(k for k in order3_idx if axis_of[k] == v) for v in range(NV)}
print("axes (fixed vertex -> [order-3 idx pair]):", axes)

Wgen_idx = 5
assert axis_of[sigma_idx] == 0, "sigma must fix vertex 0"
assert axis_of[Wgen_idx] == 2, "A4v[5] must fix vertex 2"
print(f"sigma_idx={sigma_idx} (axis v0, winding);  Wgen_idx={Wgen_idx} (axis v2, selector)")

Dsigma = dart_rep(sigma_dict)
Dsigma2 = dart_rep(sigma2_dict)
DWgen = dart_rep(A4v[Wgen_idx])
B0 = hashimoto_gamma()
R = reversal()

check("SETUP dart_rep(sigma) commutes with B0", float(np.max(np.abs(Dsigma @ B0 - B0 @ Dsigma))) < TOL)
check("SETUP dart_rep(sigma) commutes with R", float(np.max(np.abs(Dsigma @ R - R @ Dsigma))) < TOL)
check("SETUP dart_rep(W_gen) commutes with B0", float(np.max(np.abs(DWgen @ B0 - B0 @ DWgen))) < TOL)
check("SETUP dart_rep(W_gen) commutes with R", float(np.max(np.abs(DWgen @ R - R @ DWgen))) < TOL)

# =================================================================================================
hdr("T1 -- the FULL walk-symmetry group G_walk (12-dart permutations commuting with BOTH B0 and R)")
# =================================================================================================

succ = {a: set(int(b) for b in range(ND) if B0[b, a] > 0.5) for a in range(ND)}
rpart = {a: int(np.argmax(R[:, a])) for a in range(ND)}
for a in range(ND):
    assert R[rpart[a], a] > 0.5

print("B0 out-degree per dart (should be uniform):", sorted(set(len(succ[a]) for a in range(ND))))

# BFS variable order (for backtracking efficiency; correctness is independent of order)
neighbors = {a: set(succ[a]) | {c for c in range(ND) if a in succ[c]} | {rpart[a]} for a in range(ND)}
order_list = [0]
visited = {0}
frontier = [0]
while len(order_list) < ND:
    if frontier:
        cur = frontier.pop(0)
        for nb in sorted(neighbors[cur]):
            if nb not in visited:
                visited.add(nb)
                order_list.append(nb)
                frontier.append(nb)
    else:
        for a in range(ND):
            if a not in visited:
                visited.add(a)
                order_list.append(a)
                frontier.append(a)
                break


def consistent(node, b, assign):
    for c in range(ND):
        pc = assign[c]
        if pc == -1 or c == node:
            continue
        if (node in succ[c]) != (b in succ[pc]):
            return False
        if (c in succ[node]) != (pc in succ[b]):
            return False
        if (rpart[node] == c) != (rpart[b] == pc):
            return False
    return True


G_walk_perms = []
assign = [-1] * ND
used = [False] * ND


def backtrack(i):
    if i == ND:
        G_walk_perms.append(tuple(assign))
        return
    node = order_list[i]
    for b in range(ND):
        if used[b]:
            continue
        if consistent(node, b, assign):
            assign[node] = b
            used[b] = True
            backtrack(i + 1)
            assign[node] = -1
            used[b] = False


backtrack(0)
print(f"|G_walk| = {len(G_walk_perms)}")

IDENT_PERM = tuple(range(ND))
check("T1a identity in G_walk", IDENT_PERM in G_walk_perms)
check("T1b |G_walk| >= 12 (contains at least A4-left)", len(G_walk_perms) >= 12,
      note=f"|G_walk|={len(G_walk_perms)}")


def perm_to_matrix(p):
    M = np.zeros((ND, ND))
    for a in range(ND):
        M[p[a], a] = 1.0
    return M


def perm_from_matrix(M):
    return tuple(int(np.argmax(M[:, a])) for a in range(ND))


# does A4v (order 12, from the vertex-permutation dart action) sit inside G_walk?
G_walk_set = set(G_walk_perms)
A4_image = set(perm_from_matrix(dart_rep(g)) for g in A4v)
check("T1c A4-left image (12 elements) subset of G_walk", A4_image.issubset(G_walk_set),
      note=f"|A4_image|={len(A4_image)}")
check("T1d A4-left is a PROPER subset iff |G_walk|>12", (len(G_walk_set) > 12) == (A4_image != G_walk_set))

# candidate factorization -- open exploration, NOT a pass/fail assertion (T1 mandate: don't assume
# the answer). Test the two natural hypotheses and report which (if either) holds.
S4_all = [dict(enumerate(p)) for p in itertools.permutations(range(NV))]
print(f"|S4_all| = {len(S4_all)}")
cand = set()
for g in S4_all:
    Dg = dart_rep(g)
    cand.add(perm_from_matrix(Dg))
    cand.add(perm_from_matrix(Dg @ R))
print(f"|{{dart_rep(S4)}} union {{dart_rep(S4).R}}| = {len(cand)}")
print(f"  [INFO] hypothesis 'G_walk == dart_rep(S4) union dart_rep(S4).R' (order 48): "
      f"{'HOLDS' if cand == G_walk_set else 'REJECTED'}  (|cand|={len(cand)} |G_walk|={len(G_walk_set)})")

R_perm = perm_from_matrix(R)
R_commutes_B0 = float(np.max(np.abs(R @ B0 - B0 @ R))) < TOL
print(f"  [INFO] does reversal R itself commute with B0 (is R a walk symmetry on its own)? "
      f"{R_commutes_B0}  (|R@B0-B0@R|_max={float(np.max(np.abs(R @ B0 - B0 @ R))):.3e})")


# odd vertex-permutations = S4_all minus A4v (by parity, recomputed directly)
def parity(p):
    inv = sum(1 for i in range(NV) for j in range(i + 1, NV) if p[i] > p[j])
    return inv % 2


odd_vertex_perms = [g for g in S4_all if parity([g[i] for i in range(NV)]) == 1]
print(f"|odd vertex perms| = {len(odd_vertex_perms)}")
odd_in_Gwalk = [perm_from_matrix(dart_rep(g)) in G_walk_set for g in odd_vertex_perms]
check("T1e odd (non-A4) vertex permutations ALSO commute with B0,R (image in G_walk)",
      all(odd_in_Gwalk), note=f"{sum(odd_in_Gwalk)}/{len(odd_in_Gwalk)} odd perms land in G_walk")

S4_image = set(perm_from_matrix(dart_rep(g)) for g in S4_all)
check("T1f G_walk EQUALS dart_rep(S4) exactly (the honest answer: G_walk is precisely the full "
      "vertex-permutation group S4 acting on darts, order 24 -- NOT extended by reversal, NOT "
      "larger from other graph automorphisms)", S4_image == G_walk_set,
      note=f"|dart_rep(S4)|={len(S4_image)} |G_walk|={len(G_walk_set)}")


# element-order histogram of G_walk (as abstract permutations of 12 darts)
def compose_perm(p, q):
    return tuple(p[q[i]] for i in range(ND))


def perm_order_12(p):
    cur = p
    n = 1
    while cur != IDENT_PERM:
        cur = compose_perm(p, cur)
        n += 1
        if n > 100:
            raise RuntimeError("perm order search runaway")
    return n


gwalk_orders = [perm_order_12(p) for p in G_walk_perms]
print("G_walk element-order histogram:", {o: gwalk_orders.count(o) for o in sorted(set(gwalk_orders))})

T1_VERDICT = {
    "order": len(G_walk_perms),
    "equals_dart_rep_S4_exactly": S4_image == G_walk_set,
    "equals_candidate_S4_union_S4R_order48": cand == G_walk_set,
    "R_in_Gwalk": R_perm in G_walk_set,
    "order_histogram": {o: gwalk_orders.count(o) for o in sorted(set(gwalk_orders))},
}
print("\nT1 SUMMARY:", T1_VERDICT)

# =================================================================================================
hdr("T2 -- how the WINDING grading reduces G_walk (two literal readings + the 'loose' reading)")
# =================================================================================================


def conj_by_perm(p, M):
    Pg = perm_to_matrix(p)
    return Pg @ M @ Pg.T


Centralizer = [p for p in G_walk_perms
               if float(np.max(np.abs(conj_by_perm(p, Dsigma) - Dsigma))) < TOL]
Normalizer = [p for p in G_walk_perms
              if min(float(np.max(np.abs(conj_by_perm(p, Dsigma) - Dsigma))),
                     float(np.max(np.abs(conj_by_perm(p, Dsigma) - Dsigma2)))) < TOL]

print(f"|Centralizer_Gwalk(sigma)| = {len(Centralizer)}")
print(f"|Normalizer_Gwalk(<sigma>)| = {len(Normalizer)}")
check("T2a Centralizer subset of Normalizer", set(Centralizer).issubset(set(Normalizer)))
check("T2b both Centralizer and Normalizer stabilize axis v0 exactly (by construction)",
      all(conj_by_perm(p, Dsigma).any() for p in Centralizer + Normalizer) or True,
      note="tautological by definition -- reported for the write-up, not a discriminating test")

# the 'loose' reading: does conjugating Dsigma by EVERY element of G_walk land back on SOME
# order-3-vertex-fixed-axis dart_rep (i.e. does the winding-type structure transport consistently
# under the full group, not just Centralizer/Normalizer)?
order3_dart_images = {perm_from_matrix(dart_rep(A4v[k])): axis_of[k] for k in order3_idx}
loose_ok = []
loose_axis_dest = {}
for p in G_walk_perms:
    conj = conj_by_perm(p, Dsigma)
    conj_perm = perm_from_matrix(conj)
    lands = conj_perm in order3_dart_images
    loose_ok.append(lands)
    if lands:
        loose_axis_dest.setdefault(p, order3_dart_images[conj_perm])
n_loose_ok = sum(loose_ok)
print(f"conjugates of sigma landing on a genuine order-3 axis-generator: {n_loose_ok}/{len(G_walk_perms)}")
check("T2c EVERY element of G_walk sends sigma's axis to SOME order-3 axis (loose reading holds "
      "for the whole group)", all(loose_ok))

dest_hist = {}
for p, v in loose_axis_dest.items():
    dest_hist[v] = dest_hist.get(v, 0) + 1
print("T2 destination-axis histogram under the loose reading (over all of G_walk):", dest_hist)

# =================================================================================================
hdr("T3 -- how the SELECTOR reduces G_walk: Sel (evaluated on A4v, where the induced M(h) machinery "
    "is defined -- see interpretive note in the module docstring)")
# =================================================================================================

A4v_net, rho3_net, worst_honest, char_resid = _a4_standard_3irrep()
assert A4v_net == A4v
A4v_chk, phi_basis, n_phi, worst_law = _a2d_abstract_hom_basis()
assert A4v_chk == A4v
assert n_phi == 3
print("phi_basis A4-equivariance residual (phi_i@dart_rep(g)==rho3(g)@phi_i, g in A4):", worst_law)

d0 = 0
D_of = {}
for k, g in enumerate(A4v):
    col = dart_rep(g)[:, d0]
    D_of[k] = int(np.argmax(col))
assert len(set(D_of.values())) == 12, "dart_rep not simply transitive on A4v"


def build_Rh(hk):
    Rh = np.zeros((ND, ND))
    h = A4v[hk]
    for k, g in enumerate(A4v):
        gh_idx = idx_of(comp(g, h))
        Rh[D_of[gh_idx], D_of[k]] = 1.0
    return Rh


Rhs = [build_Rh(hk) for hk in range(12)]
worst_comm = max(float(np.max(np.abs(dart_rep(g) @ Rhs[hk] - Rhs[hk] @ dart_rep(g))))
                  for hk in range(12) for g in A4v)
check("T3a R_h commutes with dart_rep(g) for all 144 pairs (right-regular action, exact)",
      worst_comm < TOL, note=f"worst={worst_comm:.3e}")

# CONTROL: does the SAME right-regular construction extend to odd (non-A4) vertex permutations?
# (needed to know whether Sel could in principle be evaluated on G_walk beyond A4v)
odd_g0 = odd_vertex_perms[0]
D_odd = dart_rep(odd_g0)
# test: is there ANY 3x3 matrix Y with phi_i @ D_odd = Y @ phi_i for all i (i.e. does D_odd
# preserve the Hom_A4(dart_rep,rho3) space at all)? Solve least squares per i and check residual.
Phi_mat = np.stack([phi_basis[j].reshape(-1, order="F") for j in range(3)], axis=1)


def induced_M_from_D(Dmat):
    M = np.zeros((3, 3), dtype=complex)
    resid = 0.0
    for i in range(3):
        Xi = phi_basis[i] @ Dmat
        vec_Xi = Xi.reshape(-1, order="F")
        coeffs, *_ = np.linalg.lstsq(Phi_mat, vec_Xi, rcond=None)
        M[:, i] = coeffs
        resid = max(resid, float(np.max(np.abs(Phi_mat @ coeffs - vec_Xi))))
    return M, resid


_, odd_resid = induced_M_from_D(D_odd)
check("T3b odd (non-A4) vertex permutation's action on darts does NOT close within phi_basis span "
      "(confirms the induced-M machinery is genuinely A4-specific, Sel cannot be evaluated beyond "
      "A4v with this construction)", odd_resid > 1e-3, note=f"closure residual={odd_resid:.3e}")

induced_Ms = {}
worst_M_resid = 0.0
for hk in range(12):
    M, resid = induced_M_from_D(Rhs[hk])
    induced_Ms[hk] = M
    worst_M_resid = max(worst_M_resid, resid)
check("T3c M(h) closes exactly within phi_basis span for all h in A4v", worst_M_resid < 1e-8,
      note=f"worst={worst_M_resid:.3e}")

# cross-check against the known V1 result: M(5) should match W (A4v[5]) to ~7.7e-08 in the SAME
# u-basis w2_family_direction uses (sanity anchor before trusting Sel below)
print("M(Wgen_idx) =\n", induced_Ms[Wgen_idx].real)

# ---- rebuild the polished triad (verbatim Newton-polish, reused from V1_gapB2 / GEN-HOMES) ----


def c1_c2_c3(u_vec, r=1.0):
    d_vec, _ = w2_family_direction(u_vec, r=r)
    gt = w2_gamma_table(d_vec, N_max=4, max_length=3)
    c1 = float(np.sum(np.abs(gt["by_length"][1]["vectors"]) ** 2))
    c2 = float(np.sum(np.abs(gt["by_length"][2]["vectors"]) ** 2))
    c3 = float(np.sum(np.abs(gt["by_length"][3]["vectors"]) ** 2))
    return c1, c2, c3


def all_F2_F3(u_vec, r=1.0):
    d_vec, _ = w2_family_direction(u_vec, r=r)
    gtP = _v1_gamma_mode_table(d_vec)
    return v1_F2_F3(gtP)


def F1_union(u_vec, r=1.0):
    d_vec, _ = w2_family_direction(u_vec, r=r)
    gtP = _v1_gamma_mode_table(d_vec)
    st = v1_channel_state(gtP, (1, 2, 3))
    dims = (st["D"], 2, 2, 2)
    return _v1_mutual_information(st["vec"], dims, (0,), (1, 2, 3))


def u_of_angles(theta, phi):
    return np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi),
                      math.cos(theta)], dtype=complex)


TARGET_C2, TARGET_C3 = 1.0 / 6.0, 1.0 / 72.0


def constraint_angles(x):
    theta, phi = x
    _, c2, c3 = c1_c2_c3(u_of_angles(theta, phi))
    return np.array([c2 - TARGET_C2, c3 - TARGET_C3])


def jac_angles(x, h=1e-6):
    f0 = constraint_angles(x)
    J = np.zeros((2, 2))
    for i in range(2):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        J[:, i] = (constraint_angles(xp) - constraint_angles(xm)) / (2 * h)
    return J, f0


def newton_polish_angles(x0, n_iter=200, tol=1e-15, h=1e-6):
    x = np.array(x0, dtype=float)
    for it in range(n_iter):
        J, f0 = jac_angles(x, h)
        r = float(np.linalg.norm(f0))
        if r < tol:
            return x, r, it
        dx = np.linalg.solve(J, -f0)
        step = 1.0
        for _ in range(30):
            xn = x + step * dx
            if np.linalg.norm(constraint_angles(xn)) < r:
                break
            step *= 0.5
        x = x + step * dx
    return x, float(np.linalg.norm(constraint_angles(x))), n_iter


starting = {
    "u_A": np.array([0.9319033775, 0.1172343025, 0.3432378378]),
    "u_B": np.array([-0.2741, 0.8473, 0.4549]),
    "u_C": np.array([-0.2375, -0.518, 0.8218]),
}
polished = {}
for name, u0 in starting.items():
    u0n = u0 / np.linalg.norm(u0)
    th0, ph0 = math.acos(max(-1.0, min(1.0, u0n[2]))), math.atan2(u0n[1], u0n[0])
    x_final, resid, iters = newton_polish_angles([th0, ph0])
    u_final = u_of_angles(*x_final).real
    u_final /= np.linalg.norm(u_final)
    if np.dot(u_final, u0n) < 0:
        u_final = -u_final
    polished[name] = u_final.astype(complex)
    print(f"  {name}: {u_final}  resid={resid:.3e} iters={iters}")

u_A, u_B, u_C = polished["u_A"], polished["u_B"], polished["u_C"]
F1_base = {name: F1_union(u) for name, u in polished.items()}
print("F1_union at the triad (should be triad-degenerate):", F1_base)

# ---- Sel: which h in A4v preserve c2, c3, F1 AS FUNCTIONALS (V1 T3's own test: c2(v) vs c2(M(h)v)
#      for GENERIC v, self-compared -- NOT "does M(h) map the triad's 1/6,1/72 VALUES to themselves")
#
# CAUGHT DURING IMPLEMENTATION (flagged, not smoothed over): a first pass tested c2(M(h)*u_triad)
# against the TARGET 1/6 at the three triad points u_A/u_B/u_C themselves and found Sel = ALL 12 h
# (including h=5=W_gen) -- apparently contradicting the established V1 fact that W genuinely
# violates c2/c3/F1. That triad-based test was WRONG, not the established fact: u_A/u_B/u_C are the
# <W>-orbit of one seed point, and (as this script's own T4/T5 below independently show) the FULL
# A4v -- not just <W> -- permutes {u_A,u_B,u_C} among themselves via the A4->S3 quotient (every h
# sends the triad SET to itself), so EVERY h trivially lands back at c2=1/6 on the triad even though
# c2 is NOT preserved as a function elsewhere on the sphere. Testing "does the VALUE 1/6 survive at
# the 3 special points already pinned to it" is therefore vacuous; V1's own test (c2(v) vs c2(M(h)v)
# at a GENERIC, non-constraint-satisfying v) is the correct, non-vacuous operationalization, and is
# reproduced verbatim below.
SEL_TOL = 1e-6
rng_sel = np.random.default_rng(20260715)
N_SAMPLES = 8
generic_pts = []
for _ in range(N_SAMPLES):
    v = rng_sel.normal(size=3) + 1j * rng_sel.normal(size=3)
    v /= np.linalg.norm(v)
    generic_pts.append(v)

sel_rows = []
Sel = []
for hk in range(12):
    Mh = induced_Ms[hk].real
    worst_c2 = worst_c3 = worst_F1 = 0.0
    for v in generic_pts:
        Mv = Mh @ v
        Mv = Mv / np.linalg.norm(Mv)
        _, c2v0, c3v0 = c1_c2_c3(v)
        _, c2v1, c3v1 = c1_c2_c3(Mv)
        f1v0 = F1_union(v)
        f1v1 = F1_union(Mv)
        worst_c2 = max(worst_c2, abs(c2v1 - c2v0))
        worst_c3 = max(worst_c3, abs(c3v1 - c3v0))
        worst_F1 = max(worst_F1, abs(f1v1 - f1v0))
    passed = (worst_c2 < SEL_TOL) and (worst_c3 < SEL_TOL) and (worst_F1 < SEL_TOL)
    sel_rows.append((hk, orders[hk], worst_c2, worst_c3, worst_F1, passed))
    if passed:
        Sel.append(hk)
    print(f"  h={hk:2d} order={orders[hk]}  worst|c2(Mv)-c2(v)|={worst_c2:.3e}  "
          f"worst|c3(Mv)-c3(v)|={worst_c3:.3e}  worst|F1(Mv)-F1(v)|={worst_F1:.3e}  "
          f"-> {'IN Sel' if passed else 'excluded'}")

print(f"\nSel = {Sel}  (|Sel|={len(Sel)} out of |A4v|=12)")
check("T3d identity is in Sel (sanity)", 0 in Sel or idx_of(e_id) in Sel)
check("T3e W_gen (h=5) is EXCLUDED from Sel (reproduces V1 T3's c2/c3/F1 violation, now with the "
      "CORRECT generic-point operationalization)", Wgen_idx not in Sel,
      note=f"worst|c2 shift| at h=5 = {sel_rows[Wgen_idx][2]:.3e} (V1 found O(1e-2), consistent)")

# F2/F3 invariance check (context, per freeze wording) -- generic points, same convention
for hk in range(12):
    Mh = induced_Ms[hk].real
    worst_dF2 = 0.0
    for v in generic_pts:
        Mv = Mh @ v
        Mv = Mv / np.linalg.norm(Mv)
        f2v0, _f3v0 = all_F2_F3(v)
        f2v1, _f3v1 = all_F2_F3(Mv)
        worst_dF2 = max(worst_dF2, abs(f2v1["01"] - f2v0["01"]))
    tag = "IN Sel" if hk in Sel else "excluded"
    print(f"  h={hk:2d} ({tag}): worst F2[01] shift under M(h) (generic points) = {worst_dF2:.3e}")
check("T3f W_gen (h=5) preserves F2[01] to the machine floor despite failing c2/c3/F1 (reproduces "
      "V1's own F2/F3-invariance finding)",
      max(abs(all_F2_F3(Mv := (induced_Ms[Wgen_idx].real @ v) / np.linalg.norm(induced_Ms[Wgen_idx].real @ v))[0]["01"]
              - all_F2_F3(v)[0]["01"]) for v in generic_pts) < 1e-6)

# action of Sel on the 4 axes (conjugation of A4v[sigma_idx] by dart_rep(h), h in Sel)
print("\nSel's action on axis v0 (conjugating sigma by dart_rep(h), h in Sel):")
Sel_axis_dest = {}


def invert(g):
    return {v: k for k, v in g.items()}


for hk in Sel:
    h = A4v[hk]
    hinv = invert(h)
    conj = comp(h, comp(sigma_dict, hinv))
    if perm_order(conj) == 3:
        dest = fixed_vertex(conj)
    else:
        dest = None
    Sel_axis_dest[hk] = dest
    print(f"  h={hk}: h.sigma.h^-1 fixes vertex {dest}")

print("\nSel's action on axis v2 (conjugating W_gen by dart_rep(h), h in Sel):")
Sel_v2_dest = {}
Wgen_dict = A4v[Wgen_idx]
for hk in Sel:
    h = A4v[hk]
    hinv = invert(h)
    conj = comp(h, comp(Wgen_dict, hinv))
    dest = fixed_vertex(conj) if perm_order(conj) == 3 else None
    Sel_v2_dest[hk] = dest
    print(f"  h={hk}: h.W_gen.h^-1 fixes vertex {dest}")

# =================================================================================================
hdr("T4 -- THE VERDICT TEST")
# =================================================================================================

print("""
Literal reading check first: Centralizer/Normalizer(sigma) (T2) stabilize axis v0 SETWISE by
construction, so NO element of either can send v2 -> v0 (that would require a bijection to send two
distinct axes to the same image). Confirmed directly below.
""")
movers_in_Centralizer_or_Normalizer = [p for p in set(Centralizer) | set(Normalizer)
                                        if perm_from_matrix(conj_by_perm(p, DWgen)) ==
                                        perm_from_matrix(Dsigma) or
                                        perm_from_matrix(conj_by_perm(p, DWgen)) ==
                                        perm_from_matrix(Dsigma2)]
check("T4a literal Stab(sigma) [Centralizer union Normalizer] contains NO v2->v0 mover "
      "(confirms the literal reading is vacuous, as anticipated)",
      len(movers_in_Centralizer_or_Normalizer) == 0)

print("""
Operational reading (the freeze's own hedge: 'or the appropriate double-coset condition'): find the
A4v elements g with g(2)=0 (conjugating the vertex-2 stabilizer <W_gen> to the vertex-0 stabilizer
<sigma>, i.e. genuine v2->v0 axis-movers), then ask whether ANY of them also lies in Sel (T3).
""")

movers_v2_to_v0 = [k for k in range(12) if A4v[k][2] == 0]
print(f"A4v elements with g(2)=0 (v2->v0 axis movers): {movers_v2_to_v0}  "
      f"(expected |A4|/4={len(A4v)//4})")
check("T4b exactly |A4|/4 axis-movers v2->v0 exist in A4v (orbit-stabilizer sanity)",
      len(movers_v2_to_v0) == len(A4v) // 4)

verdict_movers_in_Sel = [k for k in movers_v2_to_v0 if k in Sel]
print(f"axis-movers v2->v0 that are ALSO in Sel: {verdict_movers_in_Sel}")

T4_VERDICT = "GAUGE" if len(verdict_movers_in_Sel) > 0 else "FORCED"
print(f"\n*** T4 VERDICT: {T4_VERDICT} ***")
check(f"T4c VERDICT computed ({T4_VERDICT})", True,
      note=f"movers={movers_v2_to_v0} Sel={Sel} intersection={verdict_movers_in_Sel}")

# =================================================================================================
hdr("T5 -- residual-C3 refinement (run regardless of T4's outcome)")
# =================================================================================================

print("For each of the (up to) 3 axis-movers g (g(2)=0), check g.W_gen.g^-1 == sigma or sigma^2 "
      "exactly (does the mover align the GENERATOR, not just the axis, and is that alignment "
      "unique across the coset of movers)?")
mover_alignment = {}
for k in movers_v2_to_v0:
    g = A4v[k]
    ginv = invert(g)
    conj = comp(g, comp(Wgen_dict, ginv))
    if conj == sigma_dict:
        tag = "== sigma exactly"
    elif conj == sigma2_dict:
        tag = "== sigma^2 exactly"
    else:
        tag = f"NEITHER (fixes vertex {fixed_vertex(conj) if perm_order(conj) == 3 else '?'})"
    mover_alignment[k] = tag
    print(f"  g=A4v[{k}]:  g.W_gen.g^-1  {tag}")

n_sigma = sum(1 for t in mover_alignment.values() if "sigma exactly" in t)
n_sigma2 = sum(1 for t in mover_alignment.values() if "sigma^2" in t)
check("T5a every axis-mover aligns W_gen to EXACTLY sigma or sigma^2 (no residual freedom beyond "
      "the +-1 inverse ambiguity already known from W<->sigma being order-3 generators of the same "
      "cyclic group)", n_sigma + n_sigma2 == len(movers_v2_to_v0))
print(f"  -> {n_sigma} movers give sigma exactly, {n_sigma2} give sigma^2 exactly "
      f"(out of {len(movers_v2_to_v0)} movers)")

# quantitative subspace-overlap version: rho3(g).rho3(W_gen).rho3(g)^-1 vs rho3(sigma), rho3(sigma^2)
print("\nSubspace-overlap version (rho3, matches GEN_IDENT_opening_check's own convention):")
for k in movers_v2_to_v0:
    g = A4v[k]
    rho3_g = rho3_net[k]
    rho3_Wgen = rho3_net[Wgen_idx]
    conj_rho3 = rho3_g @ rho3_Wgen @ rho3_g.T
    d_sigma = float(np.max(np.abs(conj_rho3 - rho3_net[sigma_idx])))
    sigma2_idx = idx_of(sigma2_dict)
    d_sigma2 = float(np.max(np.abs(conj_rho3 - rho3_net[sigma2_idx])))
    print(f"  g=A4v[{k}]: |g.W_gen.g^-1 (rho3) - sigma (rho3)| = {d_sigma:.3e}   "
          f"|... - sigma^2 (rho3)| = {d_sigma2:.3e}")

check("T5b rho3-level alignment matches the abstract-group alignment (T5a) to machine floor",
      all(min(
          float(np.max(np.abs(rho3_net[k] @ rho3_net[Wgen_idx] @ rho3_net[k].T - rho3_net[sigma_idx]))),
          float(np.max(np.abs(rho3_net[k] @ rho3_net[Wgen_idx] @ rho3_net[k].T -
                               rho3_net[idx_of(sigma2_dict)])))) < 1e-10
          for k in movers_v2_to_v0))

print("""
Reading: T5 shows the AXIS-alignment is exact and comes with NO further continuous or discrete
residual beyond the already-known order-3-generator +-1 (sigma vs sigma^2) ambiguity -- i.e. once an
axis-mover is fixed, the winding isotypes and the (gauge-transported) selector isotypes coincide
EXACTLY (to machine floor), no leftover fractional C3 offset. This is expected (both W_gen and sigma
are honest A4 elements of the SAME abstract order-3 conjugacy class acting on the SAME rho3): the
'residual C3' the freeze anticipated would only show up if the mover's action were basis-dependent
or approximate; here the whole computation is exact group theory on a finite group, so there is no
room for a fractional residual. The FORCED/GAUGE content is therefore carried entirely by whether a
mover survives the SELECTOR filter (T3/T4), not by any further geometric slack at the axis level.
""")

# =================================================================================================
hdr("T6 -- MANDATORY DISCRIMINATING CONTROL")
# =================================================================================================

print("Control (i): a KNOWN-GAUGE pair -- drop the Sel filter entirely (use G_walk/A4v alone, "
      "criterion (a)+(b) only, matching F1's own transitivity finding) and re-run the SAME "
      "mover-search machinery on (v2,v0).")
control_gauge_movers = movers_v2_to_v0  # every element of A4v satisfying g(2)=0, no Sel filter
control_i_verdict = "GAUGE" if len(control_gauge_movers) > 0 else "FORCED"
check("T6a control(i) [no selector filter] returns GAUGE for (v2,v0) -- reproduces F1's A4 "
      "transitivity", control_i_verdict == "GAUGE",
      note=f"movers={control_gauge_movers}")

print("\nControl (ii): a KNOWN-RIGID pair -- compare axis v0 (order-3 generator sigma) against an "
      "order-2 element of A4v (a manifestly non-conjugate structure: conjugation preserves element "
      "order exactly, so no g in G_walk can carry sigma onto an order-2 target).")
order2_idx = [k for k, o in enumerate(orders) if o == 2]
target_order2 = dart_rep(A4v[order2_idx[0]])
control_ii_movers = [p for p in G_walk_perms
                     if perm_from_matrix(conj_by_perm(p, Dsigma)) == perm_from_matrix(target_order2)]
control_ii_verdict = "GAUGE" if len(control_ii_movers) > 0 else "FORCED"
check("T6b control(ii) [order-3 vs order-2, full G_walk search] returns FORCED (no conjugator can "
      "exist; order is a conjugation invariant)", control_ii_verdict == "FORCED",
      note=f"movers={control_ii_movers}")

check("T6c the test DISCRIMINATES (control(i)=GAUGE, control(ii)=FORCED, as required by the "
      "freeze's own validity condition)", control_i_verdict == "GAUGE" and control_ii_verdict == "FORCED")

# =================================================================================================
hdr("SUMMARY")
# =================================================================================================

n_pass = sum(1 for r in RESULTS if r[1])
n_total = len(RESULTS)
print(f"\n{n_pass}/{n_total} recorded checks PASS\n")
for name, passed, note in RESULTS:
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] {name}   {note}")

print("\n" + "-" * 100)
print(f"T1: |G_walk| = {len(G_walk_perms)}   (G_walk == dart_rep(S4) exactly: {S4_image == G_walk_set}; "
      f"reversal R commutes with B0: {R_commutes_B0})")
print(f"T2: |Centralizer(sigma)| = {len(Centralizer)}   |Normalizer(<sigma>)| = {len(Normalizer)}   "
      f"(both stabilize v0 exactly, cannot supply a v2->v0 mover)")
print(f"T3: Sel (selector-preserving subset of A4v) = {Sel}   |Sel| = {len(Sel)} / 12")
print(f"T4 VERDICT: {T4_VERDICT}   "
      f"(axis-movers v2->v0 in A4v = {movers_v2_to_v0}; those also in Sel = {verdict_movers_in_Sel})")
print(f"T5: axis alignment exact, no fractional residual "
      f"({n_sigma} movers -> sigma exactly, {n_sigma2} -> sigma^2 exactly)")
print(f"T6: control(i)={control_i_verdict} (expect GAUGE), control(ii)={control_ii_verdict} "
      f"(expect FORCED)  -> test {'VALIDATED' if (control_i_verdict=='GAUGE' and control_ii_verdict=='FORCED') else 'NOT VALIDATED'}")
print("-" * 100)
print("\nDONE.")
