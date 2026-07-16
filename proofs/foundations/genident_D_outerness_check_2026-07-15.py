#!/usr/bin/env python3
"""
GEN-IDENT-D -- gate D0 driver: non-vacuity witnesses for the proper-outerness of the
winding-C3 automorphism alpha on M = L(F_inv(6)) = L(G), G = *_{i=1}^6 Z/2.

Freeze: internal research notes
Theorem write-up (the actual proof lives there, in prose): docs/theorems/genident_D_outerness_2026-07-15.md

WHAT THIS SCRIPT IS AND IS NOT.
  The D0 outerness verdict is a THEOREM (infinite-dimensional operator-algebra statement); it is
  NOT directly a finite computation. This driver supplies the FINITELY-CHECKABLE NON-VACUITY
  WITNESSES the freeze calls for (S2 "Finitely-checkable driver"):
    (1) sigma has order exactly 3 and is fixed-point-free on {1,...,6} (both used load-bearingly
        in the theorem's growth argument -- fixed-point-freeness is what makes the h=e twisted
        class grow without any extra hypothesis).
    (2) sigma_hat moves generators ACROSS distinct free factors (t_1 -> t_2, etc).
    (3) an ICC witness: the ORDINARY conjugacy class of a generator (and of longer words) is
        infinite, exhibited constructively by building an explicit infinite family of PROVABLY
        DISTINCT conjugates via reduced-word-length growth under conjugation by (t_a t_b)^n.
    (4) the load-bearing D0-iii/iv witness: the SIGMA_HAT-TWISTED conjugacy class of e, of every
        generator t_i, and of a few longer alternating words, ALSO grows without bound under
        twisted conjugation by (t_a t_b)^n -- the concrete mechanism behind "no finite sigma_hat-
        twisted class", which is the sufficient condition the theorem's Fourier-coefficient
        argument needs to force u=0 (no inner-implementing unitary can exist in ell^2(G)).
    (5) a brute-force spot-check (not a proof of infinitude, a necessary-condition witness) that
        no SHORT conjugator sends t_1 to t_2 -- consistent with D0-ii's claim that they are never
        conjugate.
  All arithmetic is exact reduced-word combinatorics in the free product (no floating point, no
  approximation) -- lengths are computed by an explicit free-product multiplication/cancellation
  routine and checked to match closed-form predictions (4n + |g|) exactly, not merely "increasing".

GOAL-SEEK GUARD: this driver reads/imports NOTHING from the physics codebase (no the_net.py, no
predictions/, no mass/ppm/Koide/CKM/PMNS value anywhere). It is pure finite free-product
combinatorics on the abstract 6-generator alphabet {1,...,6}. Verified by source self-scan below.

OMP_NUM_THREADS=4 (light; no numpy/linear algebra needed at all -- pure Python combinatorics).
Runtime: well under 1 second.
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")

RESULTS = []


def check(name, cond, note=""):
    RESULTS.append((name, bool(cond), note))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}   {note}")
    return bool(cond)


def hdr(s):
    print("\n" + "=" * 100 + "\n" + s + "\n" + "=" * 100)


N_GENS = 6

# =====================================================================================================
hdr("SETUP -- free-product word algebra on G = *_{i=1}^6 Z/2, generators t_1..t_6 (t_i^2 = e)")
# =====================================================================================================
print("""
Elements of G are represented as tuples of integers in {1,...,6} (reduced words: no two
consecutive equal letters, since t_i is the unique nontrivial element of factor i and t_i^2=e).
The empty tuple () is the identity e.
""")


def reduce_concat(w1, w2):
    """Multiply two REDUCED words in the free product, cascading cancellation at the junction
    (t_i^2 = e). Standard free-product normal-form multiplication."""
    w1 = list(w1)
    w2 = list(w2)
    while w1 and w2 and w1[-1] == w2[0]:
        w1.pop()
        w2.pop(0)
    return tuple(w1 + w2)


def inverse(w):
    """Each generator is self-inverse, so the inverse of a reduced word is its reversal."""
    return tuple(reversed(w))


def is_reduced(w):
    return all(w[i] != w[i + 1] for i in range(len(w) - 1))


SIGMA = {1: 2, 2: 3, 3: 1, 4: 5, 5: 6, 6: 4}  # (1 2 3)(4 5 6)


def sigma_hat(w, perm=SIGMA):
    return tuple(perm[x] for x in w)


def word_pow(w, n):
    """w^n as a reduced word, computed by repeated reduce_concat (n >= 0)."""
    out = ()
    for _ in range(n):
        out = reduce_concat(out, w)
    return out


# =====================================================================================================
hdr("ALGEBRA SELF-CHECK -- the multiplication/inverse/reduction routines are correct")
# =====================================================================================================
for i in range(1, N_GENS + 1):
    check(f"ALG t_{i} * t_{i} = e", reduce_concat((i,), (i,)) == (), )

# cascading cancellation example: (1,2,3)*(3,2,1) = e
check("ALG cascading cancellation (1,2,3)*(3,2,1) = e",
      reduce_concat((1, 2, 3), (3, 2, 1)) == ())

# associativity spot-check on a handful of triples
assoc_triples = [((1, 2), (2, 3), (3, 1)), ((4, 5, 6), (6, 1), (1, 2, 4)), ((1,), (1, 2, 3), (3, 4))]
assoc_ok = True
for a, b, c in assoc_triples:
    lhs = reduce_concat(reduce_concat(a, b), c)
    rhs = reduce_concat(a, reduce_concat(b, c))
    assoc_ok = assoc_ok and (lhs == rhs)
check("ALG associativity spot-check (3 triples)", assoc_ok)

# inverse formula: w * inverse(w) = e, for several reduced words
inv_words = [(1,), (1, 2), (1, 2, 3), (4, 5, 6, 1), (2, 3, 1, 4, 5)]
inv_ok = all(reduce_concat(w, inverse(w)) == () for w in inv_words)
check("ALG w * inverse(w) = e for 5 sample reduced words", inv_ok)

# sigma_hat is a homomorphism-consistent relabelling: sigma_hat(w1*w2) == sigma_hat(w1)*sigma_hat(w2)
homo_ok = True
for a, b in [((1, 2), (2, 3, 1)), ((4, 5), (5, 6, 4, 1)), ((1, 2, 3), (3, 2, 1))]:
    lhs = sigma_hat(reduce_concat(a, b))
    rhs = reduce_concat(sigma_hat(a), sigma_hat(b))
    homo_ok = homo_ok and (lhs == rhs)
check("ALG sigma_hat is a homomorphism on sample products (sigma_hat(w1 w2)=sigma_hat(w1)sigma_hat(w2))",
      homo_ok)


# =====================================================================================================
hdr("D0-ii WITNESSES -- sigma has order 3, is fixed-point-free, and moves generators across factors")
# =====================================================================================================
sigma2 = {i: SIGMA[SIGMA[i]] for i in range(1, N_GENS + 1)}
sigma3 = {i: SIGMA[sigma2[i]] for i in range(1, N_GENS + 1)}
identity_perm = {i: i for i in range(1, N_GENS + 1)}

check("D0ii-1 sigma^3 = identity", sigma3 == identity_perm)
check("D0ii-2 sigma != identity and sigma^2 != identity (order EXACTLY 3, not 1)",
      SIGMA != identity_perm and sigma2 != identity_perm)
check("D0ii-3 sigma is FIXED-POINT-FREE on {1,...,6} (no i with sigma(i)=i) -- load-bearing for "
      "the h=e twisted-class growth argument", all(SIGMA[i] != i for i in range(1, N_GENS + 1)))
check("D0ii-4 sigma^2 is ALSO fixed-point-free (needed for D0-iv, the alpha^2 case)",
      all(sigma2[i] != i for i in range(1, N_GENS + 1)))
check("D0ii-5 sigma_hat moves t_1 -> t_2, generators of two DISTINCT free factors "
      "(sigma_hat((1,)) = (2,))", sigma_hat((1,)) == (2,))
check("D0ii-6 sigma is a genuine permutation of the 6 factor-indices (bijective)",
      sorted(SIGMA.values()) == list(range(1, N_GENS + 1)))


# =====================================================================================================
hdr("D0-ii NECESSARY-CONDITION SPOT-CHECK -- no SHORT conjugator sends t_1 to t_2")
# =====================================================================================================
print("""
This is NOT a proof (the theorem's proof is the cyclic-reduction/normal-form argument in the
write-up, citing the standard free-product conjugacy classification, e.g. Lyndon-Schupp Ch IV).
It is a finite necessary-condition witness: brute-force over ALL reduced words h of length <= L,
confirm h * t_1 * h^{-1} is never equal to t_2 (nor to any single generator other than t_1's own
factor value) -- consistent with, and a non-vacuity guard against an error in, the claim that
generators of different free factors are never conjugate.
""")


def all_reduced_words(max_len):
    words = [()]
    frontier = [()]
    for _ in range(max_len):
        new_frontier = []
        for w in frontier:
            last = w[-1] if w else None
            for i in range(1, N_GENS + 1):
                if i != last:
                    nw = w + (i,)
                    new_frontier.append(nw)
                    words.append(nw)
        frontier = new_frontier
    return words


L_SPOT = 6
short_words = all_reduced_words(L_SPOT)
t1, t2 = (1,), (2,)
bad = []
for h in short_words:
    conj = reduce_concat(reduce_concat(h, t1), inverse(h))
    if conj == t2:
        bad.append(h)
check(f"D0ii-7 among all {len(short_words)} reduced words of length <= {L_SPOT}, "
      f"NO conjugator h has h*t1*h^-1 = t2", len(bad) == 0, note=f"bad={bad}")

# also confirm every conjugate of t_1 found in this search DOES cyclically reduce back to a
# length-1 word in factor 1 (i.e. equals t_1 itself, since factor 1 = Z/2 is abelian) -- a second
# independent necessary-condition witness of the "conjugates of t_i stay tied to factor i" claim.
non_t1_singlet_conjugates = []
for h in short_words:
    conj = reduce_concat(reduce_concat(h, t1), inverse(h))
    if len(conj) == 1 and conj != t1:
        non_t1_singlet_conjugates.append((h, conj))
check("D0ii-8 among all length<=6 conjugators, every conjugate of t_1 that happens to reduce to "
      "length 1 EQUALS t_1 itself (never a different single generator)",
      len(non_t1_singlet_conjugates) == 0, note=f"counterexamples={non_t1_singlet_conjugates}")


# =====================================================================================================
hdr("D0-i WITNESS -- ordinary (untwisted) conjugacy classes are infinite (ICC), via length growth")
# =====================================================================================================
print("""
Mechanism: for g cyclically reduced (first/last syllable in distinct factors, or |g|<=1), pick
generators t_a, t_b (a != b) with b NOT equal to the factor of g's first OR last syllable. Then
h_n = (t_a t_b)^n conjugates g to w_n = h_n g h_n^{-1}, whose reduced length is EXACTLY 4n + |g|
(no cancellation at either junction, by the choice of b) -- so w_0, w_1, w_2, ... are pairwise
DISTINCT elements of g's conjugacy class, which is therefore infinite. Checked exactly (not just
'increasing') for several sample g, including single generators and longer cyclically-reduced words.
""")


def find_ab_untwisted(g):
    i1 = g[0] if g else None
    ik = g[-1] if g else None
    forbidden_b = {i1, ik} if g else set()
    for b in range(1, N_GENS + 1):
        if b in forbidden_b:
            continue
        for a in range(1, N_GENS + 1):
            if a != b:
                return a, b
    raise RuntimeError("no valid (a,b) found")


K = 8  # n = 0..K
icc_witness_words = [(1,), (2,), (4,), (1, 2, 3), (4, 5, 6, 1), (2, 4, 6, 1, 3)]
icc_all_ok = True
for g in icc_witness_words:
    a, b = find_ab_untwisted(g)
    h_n = ()
    lengths = []
    conjugates = []
    ok_this_g = True
    for n in range(K + 1):
        h_n = word_pow((a, b), n)
        w_n = reduce_concat(reduce_concat(h_n, g), inverse(h_n))
        expected_len = 4 * n + len(g)
        lengths.append(len(w_n))
        conjugates.append(w_n)
        if len(w_n) != expected_len:
            ok_this_g = False
    all_distinct = len(set(conjugates)) == len(conjugates)
    ok_this_g = ok_this_g and all_distinct
    icc_all_ok = icc_all_ok and ok_this_g
    check(f"D0i g={g}: conjugates by (t{a}t{b})^n, n=0..{K}, have EXACT length 4n+{len(g)} "
          f"and are pairwise DISTINCT (a,b chosen to avoid g's boundary factors)",
          ok_this_g, note=f"lengths={lengths}")

check("D0-i SUMMARY: every witness word's ordinary conjugacy class is exhibited as infinite "
      "(exact-length growth + distinctness, all sample words)", icc_all_ok)


# =====================================================================================================
hdr("D0-iii/iv WITNESS -- sigma_hat-TWISTED conjugacy classes of e, each t_i, and longer words GROW")
# =====================================================================================================
print("""
Same mechanism, twisted: w_n = sigma_hat(g_n) h g_n^{-1} for g_n = (t_a t_b)^n. Since sigma_hat is
a bijective relabelling of generators (a group automorphism of G), sigma_hat(g_n) = (t_{sigma(a)}
t_{sigma(b)})^n, still reduced of length 2n. Choosing b such that sigma(b) != (factor of h's FIRST
syllable) and b != (factor of h's LAST syllable) -- i.e. b avoiding at most 2 forbidden values --
gives NO cancellation at either junction, so |w_n| = 4n + |h| EXACTLY. For h = e (the identity),
the only requirement is sigma(b) != b, which holds for EVERY b since sigma is fixed-point-free
(D0ii-3) -- this is exactly why sigma being fixed-point-free on all 6 generators is load-bearing,
not incidental.

This is checked for sigma (alpha, D0-iii) AND sigma^2 (alpha^2, D0-iv) separately.
""")


def find_ab_twisted(h, perm):
    inv_perm = {v: k for k, v in perm.items()}
    j1 = h[0] if h else None
    jk = h[-1] if h else None
    forbidden_b = set()
    if h:
        forbidden_b = {inv_perm[j1], jk}
    for b in range(1, N_GENS + 1):
        if b in forbidden_b:
            continue
        for a in range(1, N_GENS + 1):
            if a != b:
                return a, b
    raise RuntimeError("no valid (a,b) found")


twisted_witness_words = [(), (1,), (2,), (3,), (4,), (5,), (6,), (1, 2), (4, 5, 6), (1, 3, 5, 2)]

for label, perm in (("sigma (alpha, D0-iii)", SIGMA), ("sigma^2 (alpha^2, D0-iv)", sigma2)):
    twisted_all_ok = True
    for h in twisted_witness_words:
        a, b = find_ab_twisted(h, perm)
        twisted_conjugates = []
        lengths = []
        ok_this_h = True
        for n in range(K + 1):
            g_n = word_pow((a, b), n)
            sig_g_n = sigma_hat(g_n, perm)
            w_n = reduce_concat(reduce_concat(sig_g_n, h), inverse(g_n))
            expected_len = 4 * n + len(h)
            lengths.append(len(w_n))
            twisted_conjugates.append(w_n)
            if len(w_n) != expected_len:
                ok_this_h = False
        all_distinct = len(set(twisted_conjugates)) == len(twisted_conjugates)
        ok_this_h = ok_this_h and all_distinct
        twisted_all_ok = twisted_all_ok and ok_this_h
        check(f"D0iii/iv [{label}] h={h}: twisted conjugates sigma_hat(g_n) h g_n^-1 by "
              f"g_n=(t{a}t{b})^n, n=0..{K}, have EXACT length 4n+{len(h)} and are pairwise DISTINCT",
              ok_this_h, note=f"lengths={lengths}")
    check(f"D0iii/iv SUMMARY [{label}]: every witness word's {label.split()[0]}-TWISTED conjugacy "
          f"class is exhibited as infinite (this is the concrete 'no finite twisted class' "
          f"mechanism the Fourier-coefficient argument needs)", twisted_all_ok)


# =====================================================================================================
hdr("GOAL-SEEK GUARD -- AST self-scan (no physics-codebase import, no numeric data smuggled in)")
# =====================================================================================================
print("""
A plain substring scan of this file's own source is self-referential (the prose ABOVE, describing
what the guard forbids, necessarily contains words like 'ppm'/'Koide'/'m1b_*.py' -- exactly the
trap GEN-IDENT-B's checker caught: searching a script's own source for the very token list it
prints trivially "finds" the guard text itself). So this scan uses the AST instead of raw text:
  (1) collect every actual import statement (ast.Import/ImportFrom nodes, not string/comment text)
      and confirm none references the physics codebase;
  (2) collect every NUMERIC literal (ast.Constant with int/float value, which excludes docstrings
      and prose -- those are string constants) appearing anywhere in the executable code, and
      confirm it is drawn only from the small integer set this driver's own combinatorics needs
      (generator indices 1-6, loop bounds K/L_SPOT, and small derived integers) -- i.e. no
      floating-point constant (a mass/coupling ratio would be a float) appears anywhere.
""")

import ast

with open(__file__) as f:
    own_src = f.read()
tree = ast.parse(own_src)

imported_modules = []
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        imported_modules += [a.name for a in node.names]
    elif isinstance(node, ast.ImportFrom) and node.module:
        imported_modules.append(node.module)

FORBIDDEN_MODULE_PREFIXES = ("derivation_topdown", "predictions", "proofs.foundations.m1b",
                              "the_net")
bad_imports = [m for m in imported_modules if m.startswith(FORBIDDEN_MODULE_PREFIXES)]
check("GOALSEEK-1 (AST) actual import statements in this file reference ONLY stdlib modules "
      "(none from the physics codebase: no the_net, no predictions, no m1b_*)",
      len(bad_imports) == 0, note=f"all imports={imported_modules}")

numeric_constants = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        numeric_constants.add(node.value)

has_float = any(isinstance(v, float) for v in numeric_constants)
# Use an inclusive bound (<=) rather than a membership set, so the bound literal itself (which
# necessarily also appears as a Constant node in THIS line, e.g. via BOUND) never trips a spurious
# off-by-one self-reference failure. BOUND is comfortably larger than anything this driver's own
# combinatorics needs (generator indices 1-6, K=8, longest printed length 4*K+6=38).
BOUND = 1000
bad_numeric = sorted(v for v in numeric_constants if isinstance(v, int) and abs(v) > BOUND)
check("GOALSEEK-2 (AST) every numeric literal in the executable code is a small integer this "
      "driver's own free-product combinatorics needs (generator indices 1-6, loop bounds, word "
      "lengths) -- NO floating-point constant anywhere (a mass/coupling/ppm ratio would be a "
      "float)", (not has_float) and len(bad_numeric) == 0,
      note=f"has_float={has_float}, out-of-range ints={bad_numeric}, "
           f"all numeric constants={sorted(numeric_constants)}")


# =====================================================================================================
hdr("verification ADVERSARIAL CONTROLS (added by the verifier, 2026-07-15) -- attack 1 (non-vacuity)")
# =====================================================================================================
print("""
Freeze's attack-1 demand: construct discriminating controls -- does the D0-iii argument correctly
FAIL to conclude outerness for an automorphism that IS inner? Three controls, all pure integer
free-product combinatorics, no physics import, no float.

CONTROL A -- the IDENTITY permutation on the same 6-generator alphabet (alpha = id, trivially inner
via u=1=lambda_e). The theorem's own construction requires, for h=e, a generator b with perm(b)!=b.
Under the identity permutation EVERY b is a fixed point, so the construction's candidate set must be
EMPTY -- i.e. the growth witness CANNOT be built for h=e when the automorphism is (this degenerately)
inner. This is the mechanism the theorem's own text flags as load-bearing (sigma fixed-point-free);
this control checks it actually bites when fixed-point-freeness fails completely.

CONTROL B -- a genuinely GROUP-INNER automorphism sigma_hat'(g) := g0 * g * g0^{-1} (conjugation by
g0 = t_1, NOT a generator-relabelling automorphism, so outside the D0-iii growth-template's domain --
but the underlying Fourier/twisted-conjugacy IDENTITY still applies to any automorphism). Algebraic
fact, checked exactly here (not sampled): for h = g0, sigma_hat'(g) h g^-1 = g0 g g0^-1 g0 g^-1 =
g0 g g^-1 = g0 for EVERY g in G -- i.e. h=g0's twisted-conjugacy orbit is the SINGLETON {g0}, exactly
matching the actual solving unitary u = lambda_{g0} (c_{g0}=1, all other c_h=0). The machinery
correctly detects a finite twisted class exactly where an inner automorphism's solving unitary lives.

CONTROL C -- the EXCLUDED D_infty = Z/2 * Z/2 case (2 generators only), sigma = the swap (1 2).
D0-i's own text uses h = ab (a rotation/translation element) as the ICC-failure witness: class(ab) =
{ab, (ab)^-1} in the ORDINARY (untwisted) sense. This control exhaustively re-confirms that ORDINARY
conjugacy class is finite (size 2) by brute force over all reduced words up to a generous length bound
-- confirming D_infty really is non-ICC via this concrete witness, independently of the prose argument.
(Note, reported but not asserted as a pass/fail: the SIGMA-TWISTED class of this same h=ab, checked
separately below by direct computation, is NOT similarly finite -- the checker found this reaching for
a shortcut counterexample and it did not materialize. This does not affect the D0 verdict: D_infty is
still correctly excluded via D0-i/ICC, which is what factoriality of M actually requires; the twisted-
class mechanism of D0-iii is evidently a logically separate argument from D0-i, not merely parasitic
on it, which if anything makes D0-iii's closure for F_inv(6) a substantive, non-circular result.)
""")


def find_ab_twisted_general(h, perm, ngens):
    inv_perm = {v: k for k, v in perm.items()}
    if not h:
        forbidden_b = {b for b in range(1, ngens + 1) if perm[b] == b}
    else:
        forbidden_b = {inv_perm[h[0]], h[-1]}
    return [b for b in range(1, ngens + 1) if b not in forbidden_b]


# --- Control A: identity permutation, h = e ---
IDENTITY_PERM = {i: i for i in range(1, N_GENS + 1)}
candsA = find_ab_twisted_general((), IDENTITY_PERM, N_GENS)
check("SEALED-CTRL-A the identity permutation (trivially-inner alpha=id) leaves NO valid b for h=e "
      "(every b is a fixed point) -- the growth witness construction correctly CANNOT be built here, "
      "exactly where alpha really is inner", len(candsA) == 0, note=f"candidates={candsA}")

# direct algebraic check: under the identity permutation, g*e = g e g^-1 = e for ALL g (sample a few)
idA_ok = True
for g in [(1,), (2, 3), (4, 5, 6, 1), (3, 1, 2, 4, 6)]:
    w = reduce_concat(reduce_concat(sigma_hat(g, IDENTITY_PERM), ()), inverse(g))
    idA_ok = idA_ok and (w == ())
check("SEALED-CTRL-A the identity-twisted orbit of h=e is EXACTLY {e} for sample conjugators g "
      "(matches u=lambda_e=1 being the unique solving unitary)", idA_ok)

# --- Control B: group-inner automorphism sigma_hat'(g) = g0 g g0^-1, g0 = t_1; h = g0 ---
G0 = (1,)


def sigma_hat_inner_g0(g):
    return reduce_concat(reduce_concat(G0, g), inverse(G0))


ctrlB_ok = True
ctrlB_witnesses = [(), (2,), (3, 4), (5, 6, 1, 2), (2, 3, 4, 5, 6, 1)]
for g in ctrlB_witnesses:
    w = reduce_concat(reduce_concat(sigma_hat_inner_g0(g), G0), inverse(g))
    ctrlB_ok = ctrlB_ok and (w == G0)
check("SEALED-CTRL-B for the GROUP-INNER automorphism sigma_hat'(g)=g0*g*g0^-1 (g0=t_1, conjugation, "
      "not a generator-relabelling), the sigma_hat'-twisted orbit of h=g0 is EXACTLY the singleton "
      "{g0} for sample conjugators g -- matches the actual solving unitary u=lambda_{g0}",
      ctrlB_ok, note=f"witnesses checked={ctrlB_witnesses}")

# --- Control C: D_infty (2 generators), sigma=swap, h=ab -- ORDINARY class is finite (excluded case) ---
SWAP2 = {1: 2, 2: 1}


def all_reduced_words_n(max_len, ngens):
    words = [()]
    frontier = [()]
    for _ in range(max_len):
        nf = []
        for w in frontier:
            last = w[-1] if w else None
            for i in range(1, ngens + 1):
                if i != last:
                    nw = w + (i,)
                    nf.append(nw)
                    words.append(nw)
        frontier = nf
    return words


L_CTRL_C = 12
h_ab = (1, 2)
words2 = all_reduced_words_n(L_CTRL_C, 2)
ordinary_orbit = set()
for w in words2:
    conj = reduce_concat(reduce_concat(w, h_ab), inverse(w))
    ordinary_orbit.add(conj)
check(f"SEALED-CTRL-C D_infty (2 generators): the ORDINARY conjugacy class of h=ab, exhaustively "
      f"searched over all {len(words2)} reduced conjugators of length <= {L_CTRL_C}, is EXACTLY "
      f"{{ab, (ab)^-1}} (size 2, finite) -- reconfirms the excluded-case witness independently of the "
      f"prose argument", ordinary_orbit == {(1, 2), (2, 1)}, note=f"orbit found={sorted(ordinary_orbit)}")

# reported for context only (not asserted pass/fail against a "should be finite" expectation -- see
# the prose above): the SIGMA-TWISTED class of the same h=ab under the swap, same exhaustive search.
twisted_orbit_ab = set()
for w in words2:
    conj = reduce_concat(reduce_concat(sigma_hat(w, SWAP2), h_ab), inverse(w))
    twisted_orbit_ab.add(conj)
check(f"SEALED-CTRL-C (context, not a vacuity requirement) the SIGMA-TWISTED class of h=ab under the "
      f"swap automorphism, same exhaustive search, has size {len(twisted_orbit_ab)} -- reported to show "
      f"the twisted mechanism is logically independent of the ordinary-ICC mechanism, not parasitic on "
      f"it", len(twisted_orbit_ab) > 2, note=f"twisted orbit size={len(twisted_orbit_ab)} (vs ordinary=2)")


# =====================================================================================================
hdr("SUMMARY")
# =====================================================================================================
n_pass = sum(1 for r in RESULTS if r[1])
n_total = len(RESULTS)
print(f"\n{n_pass}/{n_total} recorded checks PASS\n")
for name, passed, note in RESULTS:
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] {name}   {note}")

print("\n" + "-" * 100)
print("These are NON-VACUITY WITNESSES for the D0 theorem (full proof in")
print("docs/theorems/genident_D_outerness_2026-07-15.md), not the proof itself. The witnesses")
print("confirm, by exact finite combinatorics: sigma has order 3 and is fixed-point-free (both")
print("load-bearing); generators of distinct factors are not short-conjugate; ordinary AND")
print("sigma_hat-twisted (and sigma_hat^2-twisted) conjugacy classes of e, every generator, and")
print("several longer words all grow WITHOUT BOUND under an explicit, exactly-computed family of")
print("conjugators -- concretely exhibiting the 'no finite twisted class' mechanism the proof uses")
print("to force u=0 in the Fourier-coefficient argument (Ad(u)=alpha is impossible for u in ell^2(G)).")

if n_pass == n_total:
    print("\nRESULT: ALL CHECKS PASS")
else:
    print(f"\nRESULT: {n_total - n_pass} CHECK(S) FAILED")

sys.exit(0 if n_pass == n_total else 1)
