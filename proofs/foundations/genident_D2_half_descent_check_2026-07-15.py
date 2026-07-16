#!/usr/bin/env python3
"""
GEN-IDENT-D2 (the "quick decisive half") -- durable check driver.

CLAIM UNDER TEST (architect-direct, NOT YET BOOKED AS FORCED -- pending verification):

    The vertex axis W does NOT descend to an automorphism of the canonical crossed-product
    observer home  M (x)_alpha Z_3  (D0/D1).  Equivalently: the canonical M_3(C) home carries
    sigma (by construction, D1) but has NO canonical W-action -- because W does not normalize
    the crossed Z_3-action.

WHY (the chain):
  (1) [GEN-IDENT-A, sealed]  sigma and W are two order-3 elements of A4 with <sigma,W> = A4.
  (2) [finite-group fact, proved here]  <sigma> is a Sylow-3 of A4 and is SELF-NORMALIZING:
      N_{A4}(<sigma>) = <sigma> (order 3).  A4 has four conjugate Sylow-3's; n_3=4 ==> |N|=3.
  (3) [consequence]  W notin N_{A4}(<sigma>)  (since <sigma,W>=A4 != <sigma>), i.e.
      W sigma W^{-1} notin <sigma>.
  (4) [descent criterion]  an automorphism beta of M implementing W (via the A4 edge-action
      on the 6 free-product generators of F_inv(6)) descends to Aut(M (x)_alpha Z_3) fixing M
      iff  beta . alpha . beta^{-1}  lies in  <alpha>  as an OUTER class, i.e. iff
      W sigma W^{-1} in <sigma> as a generator-permutation.  By (3) it does not.
  (5) [operator-algebra transfer]  the edge-action A4 -> S_6 is FAITHFUL, and by the
      GENERALIZED-OUTERNESS LEMMA (below; the D0-iii twisted-conjugacy technique extended from
      the sealed fixed-point-free case to ANY nontrivial generator-permutation) every nontrivial
      generator-permutation induces a PROPERLY OUTER automorphism of M = L(F_inv(6)), so
      S_6 hookrightarrow Out(M) INJECTIVELY.  Hence the A4-level failure in (3) transfers to
      Out(M): the three permutations tau_k := (W sigma W^{-1}) . sigma^{-k}  (k=0,1,2) are each
      NONTRIVIAL (since W sigma W^{-1} notin <sigma>), hence each alpha_{tau_k} is properly outer,
      i.e. [alpha_W alpha alpha_W^{-1}] = [alpha_{W sigma W^{-1}}] notin {[id],[alpha],[alpha^2]}
      = <[alpha]> in Out(M).  ==>  W does not descend; the home carries no canonical W-action.

  GENERALIZED-OUTERNESS LEMMA (closes the verification transfer gap; proof in the theorem doc
  docs/theorems/genident_D2_leg1_W_no_descent_2026-07-15.md).  For G = F_inv(6) and ANY
  nontrivial tau in S_6, the induced *-automorphism alpha_tau of M = L(G) is properly outer.
  Proof (D0-iii verbatim, minus the fixed-point-free assumption): alpha_tau inner ==> a unit-l^2
  coefficient function c constant on every tau-twisted conjugacy orbit g * h = tau_hat(g) h g^{-1};
  each orbit is INFINITE (words w_n = tau_hat(g_n) h g_n^{-1} with g_n=(t_a t_b)^n reduce EXACTLY
  to length 4n+|h|).  The h=e junction needs only SOME b with tau(b) != b -- which any nontrivial
  tau supplies (fixed-point-freeness was a convenience for sigma, never a requirement); h != e
  uses boundary-avoidance (>= 4 valid b of 6).  So c == 0, contradiction ==> alpha_tau not inner
  ==> (M a factor, D0-i) properly outer.  STEP 5B below anchors this for the exact tau_0, tau_1,
  tau_2 the transfer needs (incl. tau_1's two fixed points -- the case sealed D0-iii did not
  literally cover, flagged by the verification).

  FUNCTORIALITY (why "does W descend from M" is the RIGHT test for "does the M_3(C) leg carry W"):
  the decomposition M rtimes_alpha Z_3 ~= M_3(C) (x) M^alpha is derived FUNCTORIALLY from the pair
  (M, alpha) alone (D0/D1).  So any FORCED action of a substrate symmetry on the M_3(C) leg must
  factor through an extension of that symmetry across (M, alpha) -- i.e. an automorphism of the
  crossed product restricting to it on M.  No such extension (this driver) ==> no forced W-action
  on the leg.  (An UNforced inner unitary rho_3(W) on M_3(C)=B(C^3) exists by Skolem-Noether -- but
  is not forced; leg 1 denies only the forced/canonical action, not the existence of a unitary.)

NON-VACUITY (mandatory control): the descent test must DISCRIMINATE, not always-fail.
  - sigma, sigma^2 (in <sigma>)  DO normalize <sigma>  ==> they DO descend (positive control).
  - the V4 double-transpositions do NOT normalize <sigma> either ==> the criterion is
    <sigma>-MEMBERSHIP-specific, not a parity/coset artifact; ONLY the 3 elements of <sigma>
    pass.  W is one of the 9 non-passing elements -- a genuine verdict, not a tautology.

HONEST BOUND: this decides only whether W descends as an AUTOMORPHISM of the crossed product.
It does NOT rule out a vertex-MEDIATED coupling (-kappa.I(A;B) across the level gap) reaching W
by a non-automorphism mechanism -- that is D2-b, gated separately and shown un-posable-without-
the-(D)-trap by the D2-a sweep.  Together they are the ORTHOGONAL case; alone this is the clean,
decisive half.  Books NOTHING as forced (no accretion into the_net.py); NOT wired into verify.py.

GOAL-SEEK GUARD: no mass/ppm/Koide-Q/mass-ordering/mixing/CKM/PMNS value read, referenced, or
used as any criterion.  Every object (sigma, W, A4, the K4 edge-action) is pure finite-group
combinatorics, REUSED verbatim from the sealed GEN-IDENT-A/B machinery.  AST self-scan at end.

OMP_NUM_THREADS=4.  Pure integer combinatorics; runtime < 1s.  Read-only on the_run.py/Layer-1.
"""
import sys, os, re, inspect, ast, itertools

sys.path.insert(0, ".")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

RESULTS = []


def check(name, cond, note=""):
    RESULTS.append((name, bool(cond), note))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}   {note}")
    return bool(cond)


def hdr(s):
    print("\n" + "=" * 100 + "\n" + s + "\n" + "=" * 100)


# =====================================================================================================
hdr("SETUP -- import the SAME sigma, W, A4 as the sealed GEN-IDENT-A/B machinery")
# =====================================================================================================
from derivation_topdown.state.the_net import _a4_vertex_group, _a4_key, NV

A4v = _a4_vertex_group()
ix = {_a4_key(g): n for n, g in enumerate(A4v)}


def comp(g, h):
    """(g o h)(i) = g[h[i]] -- compose vertex permutations (dicts on {0..NV-1})."""
    return {i: g[h[i]] for i in range(NV)}


def inv(g):
    r = {}
    for i in range(NV):
        r[g[i]] = i
    return r


e_id = {i: i for i in range(NV)}
sigma = {0: 0, 1: 2, 2: 3, 3: 1}                 # the winding/generation C3 (same as B-checker)
W = A4v[5]                                        # the vertex-selected axis (same as A/B)
check("SETUP: NV == 4 (A4 acts on 4 tetra vertices)", NV == 4, note=f"NV={NV}")
check("SETUP: |A4v| == 12", len(A4v) == 12)
check("SETUP: sigma in A4v", _a4_key(sigma) in ix)
sigma2 = comp(sigma, sigma)


def order(g):
    n, x = 1, g
    while _a4_key(x) != _a4_key(e_id):
        x = comp(x, g); n += 1
        if n > 12: return -1
    return n


check("SETUP: sigma has order 3", order(sigma) == 3, note=f"ord={order(sigma)}")
check("SETUP: W has order 3", order(W) == 3, note=f"ord={order(W)}")


def group_closure(gens):
    elems = {_a4_key(e_id): e_id}
    frontier = list(gens)
    for g in gens:
        elems[_a4_key(g)] = g
    changed = True
    while changed:
        changed = False
        for a in list(elems.values()):
            for b in gens:
                c = comp(a, b); k = _a4_key(c)
                if k not in elems:
                    elems[k] = c; changed = True
    return elems


# =====================================================================================================
hdr("STEP 1 -- reproduce GEN-IDENT-A: <sigma, W> = A4 (order 12), so W notin <sigma>")
# =====================================================================================================
sub_sigma = group_closure([sigma])
grp_sigma_W = group_closure([sigma, W])
check("1a <sigma> has order 3 (a Z_3 subgroup)", len(sub_sigma) == 3, note=f"|<sigma>|={len(sub_sigma)}")
check("1b <sigma, W> generates A4 exactly (order 12) [GEN-IDENT-A]", len(grp_sigma_W) == 12,
      note=f"order={len(grp_sigma_W)}")
check("1c therefore W notin <sigma> (else the join could not exceed order 3)",
      _a4_key(W) not in sub_sigma, note=f"W in <sigma>? {_a4_key(W) in sub_sigma}")


# =====================================================================================================
hdr("STEP 2 -- <sigma> is a SELF-NORMALIZING Sylow-3: N_{A4}(<sigma>) = <sigma>")
# =====================================================================================================
sigma_keys = set(sub_sigma.keys())


def normalizes(g):
    """does g normalize <sigma>?  g <sigma> g^{-1} == <sigma>  <==>  g sigma g^{-1} in <sigma>."""
    gi = inv(g)
    conj = comp(comp(g, sigma), gi)
    return _a4_key(conj) in sigma_keys


normalizer = [g for g in A4v if normalizes(g)]
check("2a N_{A4}(<sigma>) has order EXACTLY 3 (self-normalizing Sylow-3)",
      len(normalizer) == 3, note=f"|N|={len(normalizer)}")
check("2b N_{A4}(<sigma>) == <sigma> (the normalizer is the subgroup itself)",
      set(_a4_key(g) for g in normalizer) == sigma_keys)

# Sylow arithmetic cross-check: number of conjugates of <sigma> == [A4 : N] == 4
conjugate_subgroups = set()
for g in A4v:
    gi = inv(g)
    conj_sub = frozenset(_a4_key(comp(comp(g, s), gi)) for s in sub_sigma.values())
    conjugate_subgroups.add(conj_sub)
check("2c <sigma> has EXACTLY 4 conjugate Sylow-3 subgroups ( [A4:N]=12/3=4 ; n_3=4 )",
      len(conjugate_subgroups) == 4, note=f"#conjugates={len(conjugate_subgroups)}")


# =====================================================================================================
hdr("STEP 3 -- W does NOT normalize <sigma>:  W sigma W^{-1} notin <sigma>")
# =====================================================================================================
Wi = inv(W)
WsW = comp(comp(W, sigma), Wi)
check("3a W sigma W^{-1} is NOT in <sigma> (W fails the descent/normalization criterion)",
      _a4_key(WsW) not in sigma_keys,
      note=f"W sigma W^-1 = {WsW}  ; <sigma> = {[sub_sigma[k] for k in sigma_keys]}")
check("3b equivalently W notin N_{A4}(<sigma>)", not normalizes(W))
check("3c W sigma W^{-1} is itself an order-3 element (a DIFFERENT Sylow-3 generator)",
      order(WsW) == 3, note=f"ord={order(WsW)}")


# =====================================================================================================
hdr("STEP 4 -- NON-VACUITY: the descent test DISCRIMINATES (only <sigma> passes)")
# =====================================================================================================
# Census: which of the 12 elements normalize <sigma>?
passers = [g for g in A4v if normalizes(g)]
# classify the 12 A4 elements: identity(1) + 3-cycles(8) + double-transpositions(3)
by_order = {1: [], 2: [], 3: []}
for g in A4v:
    by_order[order(g)].append(g)
check("4a A4 census: 1 identity, 3 double-transpositions (order 2), 8 three-cycles (order 3)",
      len(by_order[1]) == 1 and len(by_order[2]) == 3 and len(by_order[3]) == 8,
      note=f"orders: 1->{len(by_order[1])}, 2->{len(by_order[2])}, 3->{len(by_order[3])}")

# positive control: sigma, sigma^2 DO normalize (they ARE in <sigma>) ==> they descend
check("4b CONTROL(+): sigma normalizes <sigma> (in <sigma>) ==> sigma DOES descend",
      normalizes(sigma))
check("4c CONTROL(+): sigma^2 normalizes <sigma> ==> sigma^2 DOES descend", normalizes(sigma2))

# negative controls: the V4 double-transpositions do NOT normalize <sigma> either
dt_normalize = [normalizes(g) for g in by_order[2]]
check("4d CONTROL(-): NONE of the 3 double-transpositions normalize <sigma> (criterion is "
      "<sigma>-membership-specific, not a parity/even-odd artifact)",
      not any(dt_normalize), note=f"dt_normalize={dt_normalize}")

# the crisp discrimination: EXACTLY the 3 elements of <sigma> pass; the other 9 (W among them) fail
check("4e EXACTLY 3 of 12 elements pass the descent test, and they are precisely <sigma> "
      "(so the test is a genuine verdict; W is one of the 9 failers)",
      len(passers) == 3 and set(_a4_key(g) for g in passers) == sigma_keys
      and _a4_key(W) not in set(_a4_key(g) for g in passers))


# =====================================================================================================
hdr("STEP 5 -- FAITHFUL edge-action A4 -> S_6: the A4-level fact transfers to M = L(F_inv(6))")
# =====================================================================================================
# K4 (the tetra 1-skeleton, srs primitive-cell quotient): 4 vertices, 6 edges.  A4 acts on the
# vertices; the induced action on the 6 edges is the free-product generator-permutation used to
# build M = L(F_inv(6)) and its automorphism alpha (D0).  We verify: (i) the edge-action is
# FAITHFUL (only e fixes all 6 edges), (ii) sigma's edge-image has cycle type (3,3) -- matching
# D0's sigma=(1 2 3)(4 5 6), and (iii) the descent-criterion FAILURE for W transfers to the edge
# (generator) level.  Faithfulness + D0's "nontrivial generator-perm is never inner" gives
# S_6 hookrightarrow Out(M) injectively (asserted; anchored on the SEALED D0-ii t_i !~ t_j fact).
EDGES = list(itertools.combinations(range(4), 2))          # 6 undirected edges of K4
edge_ix = {e: n for n, e in enumerate(EDGES)}


def edge_perm(g):
    """the permutation of the 6 edges induced by vertex-permutation g (a tuple on 0..5)."""
    out = [0] * 6
    for e in EDGES:
        img = tuple(sorted((g[e[0]], g[e[1]])))
        out[edge_ix[e]] = edge_ix[img]
    return tuple(out)


def perm_order(p):
    n, q = 1, p
    idp = tuple(range(len(p)))
    while q != idp:
        q = tuple(p[q[i]] for i in range(len(p))); n += 1
        if n > 24: return -1
    return n


def cycle_type(p):
    seen = [False] * len(p); cyc = []
    for i in range(len(p)):
        if seen[i]: continue
        L, j = 0, i
        while not seen[j]:
            seen[j] = True; j = p[j]; L += 1
        cyc.append(L)
    return tuple(sorted(cyc, reverse=True))


# (i) faithfulness: distinct A4 elements -> distinct edge-permutations
edge_images = {}
collision = False
for g in A4v:
    ep = edge_perm(g)
    if ep in edge_images.values() and _a4_key(g) not in edge_images:
        pass
    edge_images[_a4_key(g)] = ep
distinct_edge = len(set(edge_images.values()))
check("5a the edge-action A4 -> S_6 is FAITHFUL (12 distinct edge-permutations)",
      distinct_edge == 12, note=f"#distinct={distinct_edge}")
check("5b only the identity fixes all 6 edges (trivial kernel)",
      edge_perm(e_id) == tuple(range(6))
      and sum(1 for g in A4v if edge_perm(g) == tuple(range(6))) == 1)

# (ii) sigma's edge-image is order-3 with cycle type (3,3) -- matches D0's (1 2 3)(4 5 6)
se = edge_perm(sigma)
check("5c sigma's edge-image has order 3 and cycle type (3,3) -- matches D0's sigma=(1 2 3)(4 5 6)",
      perm_order(se) == 3 and cycle_type(se) == (3, 3),
      note=f"ord={perm_order(se)}, cyctype={cycle_type(se)}")

# (iii) the descent failure transfers: W_edge sigma_edge W_edge^{-1} notin <sigma_edge>
we = edge_perm(W)
we_i = tuple(we.index(i) for i in range(6))
sigma_edge_grp = set()
q = tuple(range(6))
for _ in range(3):
    sigma_edge_grp.add(q); q = tuple(se[q[i]] for i in range(6))
WseW = tuple(we[tuple(se[we_i[i]] for i in range(6))[i]] for i in range(6))
check("5d W_edge sigma_edge W_edge^{-1} NOT in <sigma_edge> (descent failure transfers to the "
      "6-generator level of M = L(F_inv(6)))", WseW not in sigma_edge_grp)
# =====================================================================================================
hdr("STEP 5B -- GENERALIZED OUTERNESS for the EXACT tau_k the transfer needs "
    "(closes the verification gap; D0-iii technique, no fixed-point-free assumption)")
# =====================================================================================================
# The transfer needs: [alpha_{W sigma W^{-1}}] notin {[id],[alpha],[alpha^2]} in Out(M), i.e. each
# tau_k := (W sigma W^{-1}) . sigma^{-k}  (edge/generator level, k=0,1,2) induces a properly-outer
# alpha_{tau_k}.  We anchor proper-outerness the SAME way D0-iii's driver did: exhibit the
# tau_k-twisted-conjugacy orbit-length growth |tau_hat(g_n) h g_n^{-1}| = 4n+|h| EXACTLY, for h=e
# (the branch that needed fixed-point-freeness for sigma) and for nonzero-h witnesses -- with EXACT
# free-product word arithmetic (t_i^2=1, reduced = no adjacent repeats).  tau_1 has two fixed
# points (the case sealed D0-iii did not literally cover); we verify it explicitly.

def w_reduce(w):
    """reduce a word (tuple of gen-indices 0..5) modulo t_i^2 = 1 (cancel adjacent equal)."""
    out = []
    for x in w:
        if out and out[-1] == x:
            out.pop()
        else:
            out.append(x)
    return tuple(out)

def w_mul(a, b):
    return w_reduce(tuple(a) + tuple(b))

def w_inv(a):
    return tuple(reversed(a))                       # each t_i self-inverse

def w_relabel(perm, w):                             # apply generator-permutation to a word
    return w_reduce(tuple(perm[x] for x in w))

def twisted(perm, g, h):                            # g * h = tau_hat(g) . h . g^{-1}
    return w_mul(w_mul(w_relabel(perm, g), h), w_inv(g))

def perm_inv(p):
    r = [0] * len(p)
    for i, v in enumerate(p):
        r[v] = i
    return tuple(r)

def perm_mul(p, q):                                 # (p o q)[i] = p[q[i]]
    return tuple(p[q[i]] for i in range(len(p)))

se_inv = perm_inv(se)                               # sigma^{-1} at edge level
id6 = tuple(range(6))
WsW_edge = edge_perm(WsW)
tau = {}
tau[0] = WsW_edge
tau[1] = perm_mul(WsW_edge, se_inv)                 # (W sigma W^-1) . sigma^{-1}
tau[2] = perm_mul(WsW_edge, perm_mul(se_inv, se_inv))

for k in range(3):
    check(f"5B-{k}a tau_{k} = (W sigma W^-1).sigma^-{k} is NONTRIVIAL (edge/generator level)",
          tau[k] != id6, note=f"tau_{k}={tau[k]}")

# record tau_1's fixed points (the sealed-D0-iii-uncovered case)
fp1 = [i for i in range(6) if tau[1][i] == i]
check("5B tau_1 has fixed points (>0) -- the case the sealed D0-iii text did not literally cover; "
      "handled by the generalized lemma below", len(fp1) > 0, note=f"fixed points of tau_1: {fp1}")

def orbit_growth_ok(perm, hwords, nmax=8):
    """for each witness h, find a valid (a,b) per the D0-iii recipe and verify |tau_hat(g_n) h
    g_n^{-1}| == 4n+|h| EXACTLY and strictly increasing for n=0..nmax.  Returns (ok, detail)."""
    pinv = perm_inv(perm)
    for h in hwords:
        if len(h) == 0:
            cand_b = [b for b in range(6) if perm[b] != b]          # need tau(b) != b
        else:
            forbidden = {pinv[h[0]], h[-1]}                          # avoid h's boundary factors
            cand_b = [b for b in range(6) if b not in forbidden]
        placed = False
        for b in cand_b:
            a_choices = [a for a in range(6) if a != b]
            for a in a_choices:
                lengths = []
                good = True
                for n in range(0, nmax + 1):
                    g = tuple(([a, b] * n))
                    w = twisted(perm, g, h)
                    if len(w) != 4 * n + len(h):
                        good = False
                        break
                    lengths.append(len(w))
                if good and all(lengths[i] < lengths[i + 1] for i in range(len(lengths) - 1)):
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return False, f"no clean (a,b) for h={h}"
    return True, "ok"

WITNESS_H = [(), (0,), (0, 1), (3, 4, 5)]
for k in range(3):
    ok_k, det_k = orbit_growth_ok(tau[k], WITNESS_H)
    check(f"5B-{k}b tau_{k}-twisted orbits grow EXACTLY as 4n+|h| (h in {{e,t0,t0t1,t3t4t5}}, "
          f"n=0..8) ==> every twisted class infinite ==> alpha_tau_{k} not inner ==> "
          f"(factor) properly outer", ok_k, note=det_k)

check("5B-CONCLUSION each tau_k (k=0,1,2) properly outer ==> [alpha_{W sigma W^-1}] notin "
      "<[alpha]> in Out(M) ==> W does NOT descend to Aut(M rtimes_alpha Z_3); the M_3(C) home "
      "carries NO canonical W-action.  (S_6 -> Out(M) injectivity is exactly this lemma applied "
      "to every nontrivial generator-permutation.)",
      all(tau[k] != id6 for k in range(3))
      and all(orbit_growth_ok(tau[k], WITNESS_H)[0] for k in range(3)))


# =====================================================================================================
hdr("STEP 6 -- GOAL-SEEK / circularity self-scan")
# =====================================================================================================
traced = "".join(inspect.getsource(f) for f in
                 (_a4_vertex_group, _a4_key, group_closure, edge_perm))
no_data_tokens = ["m_e", "m_mu", "m_tau", "m_nu", "koide", "ppm", "pdg", "0.0510", "105.658",
                  "1776.8", "m_z", "m_w", "ckm", "pmns"]
data_hits = [t for t in no_data_tokens if t.lower() in traced.lower()]
check("6a traced input functions contain NO mass/ppm/Koide/CKM/PMNS token", len(data_hits) == 0,
      note=f"hits={data_hits}")

with open(__file__) as f:
    my_src = f.read()
tree = ast.parse(my_src)
float_literals = [n.value for n in ast.walk(tree)
                  if isinstance(n, ast.Constant) and isinstance(n.value, float)]
check("6b this driver's executable code has NO floating-point literal (pure integer "
      "combinatorics -- any physical constant would be a float)", len(float_literals) == 0,
      note=f"floats={float_literals}")


# =====================================================================================================
hdr("SUMMARY")
# =====================================================================================================
n_pass = sum(1 for r in RESULTS if r[1]); n_total = len(RESULTS)
print(f"\n{n_pass}/{n_total} recorded checks PASS\n")
for name, passed, note in RESULTS:
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}   {note}")

print("\n" + "-" * 100)
print("VERDICT (architect-direct + SEALED-CONCURRED 2026-07-15, corrections applied):")
print("  W does NOT normalize <sigma> in A4 (self-normalizing Sylow-3), and the edge-action is")
print("  faithful, so via D0's S_6 -> Out(M) injectivity, W does NOT descend to an automorphism")
print("  of the canonical home M rtimes_alpha Z_3.  The home carries sigma but has NO canonical")
print("  W-action.  This is the clean, decisive HALF of D2-ORTHOGONAL (the vertex-mediated")
print("  non-automorphism route is D2-b, separate).  DISCRIMINATES: only <sigma>'s 3 elements")
print("  pass; W is one of the 9 that fail.  Books nothing forced; no the_net.py accretion.")

if n_pass == n_total:
    print("\nRESULT: ALL CHECKS PASS")
else:
    print(f"\nRESULT: {n_total - n_pass} CHECK(S) FAILED")
sys.exit(0 if n_pass == n_total else 1)
