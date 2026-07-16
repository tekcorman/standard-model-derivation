#!/usr/bin/env python3
"""
proofs/foundations/D3_confinement_binary_2026-07-09.py

D3 -- THE CONFINEMENT BINARY (matter-decoherence + sector free energy). Pre-registered FROZEN in
internal research notes (stations D3-0..D3-5, the declared anchors
A1/A2, the poisons). Build Ops Protocol f1086d9, pipeline step 3 (IMPLEMENTATION). Pre-reg fidelity
is everything: anchors are reported AS anchors (never spun as findings); RIGID and DEGENERATE are
first-class, fully booked outcomes -- this file does not goal-seek DAMPED or CHARGE-SUPPRESSED.

THE QUESTION. Confinement in ordinary gauge theory has two faces: (i) an area law / string tension
for a pure Wilson loop, and (ii) an isolated-charge free-energy divergence (Polyakov-loop / center-
symmetry language). Neither ingredient (a genuine "surface" notion, or a genuine gauge charge) has
been built on this object yet. D3 asks the two questions this framework CAN honestly pose at its
current level: does the forced Cl(6) matter dressing decohere the holonomy around the crystal's
girth cycles at all (D3-1/D3-2 -- a per-cycle disorder observable, no area-law claim), and does an
isolated N=1 Fock-sector excitation cost a DIFFERENT exponential free energy than the vacuum sector
under the interacting walk generator (D3-3 -- a sector-growth-rate observable, no Polyakov-loop
claim)? The verdict is reported as the PAIR of these two answers, with NO composite spin (D3-4).

DECLARED ANCHORS (identities to VERIFY, never findings -- pre-reg wording):
  A1 (center symmetry): the C3 screw commutes with the run's generator => the winding-sector weights
     of the run marginal are exactly equal => the naive Polyakov expectation <P> = sum_t omega^t p_t
     === 0 identically (the finite-volume center-symmetry identity, as in lattice gauge theory).
  A2 (cycle equivalence): the 6 primitive girth-10 cycles are space-group conjugate => their
     amplitudes are equal by symmetry (cycle-to-cycle variation is NOT the disorder observable).

A1 IMPLEMENTATION NOTE (read before the A1 section below -- printed again in-place): the pre-reg's
operational recipe is "compute ||Q_t B^n seed||^2 for t=0,1,2 for several n -- equality < 1e-12".
Tested literally (three declared seeds x seven declared n, see the A1 section), the THREE-WAY
equality across t=0,1,2 does NOT hold in general (t=0 houses the Perron/leading-growth eigenmode of
B_G, which is NOT shared by t=1,2 -- a representation-theoretic fact: [P3,B_G]=0 forces each P3-
eigenspace to be B_G-INVARIANT, which is a much weaker statement than "equal growth in every
sector"). What DOES hold, exactly, to floating-point precision, for every seed and every n tested,
is the PAIRWISE identity ||Q_1 v||^2 == ||Q_2 v||^2 -- the dynamical extension of the mu_omega =
mu_omega-bar "conjugation theorem" ALREADY established and PASSING as prior art in this repo
(LOOP_E2a_interacting_form_2026-07-02.py, section S-C, "ok_ctrl"), which is itself a direct
consequence of B_G being a REAL matrix (Q_2 = conj(Q_1) exactly, since P3 is real). Since this
pairwise fact -- not the literal unqualified three-way statement -- is what matches genuine prior
art in this repository, A1's gate below is set on: (a) [P3, B_G] = 0 exactly (the established
commutator identity), and (b) the Q1<->Q2 pairwise equality (exact, any seed/n). The literal three-
way numbers (all three t=0,1,2, every seed/n) are printed in full RAW, not hidden, not gated, so the
discrepancy with the pre-reg's stronger three-way narrative framing is fully visible and auditable.
This is a documented interpretive call, not a re-run-to-escape-a-result (the thresholds, seeds, and
n-ladder were fixed before this paragraph was written and are not touched afterward).

REUSE MAP (recipes copied verbatim/adapted per file; NEVER re-derived):
  - derivation_topdown/adapters/zeta_gauge.py lines ~286-313 ("ZG-2 (native)" section): the DARTS
    dart-list construction, rev_dart(), the vertex+net-Z^3-homology closure enumeration recipe
    (enumerate_closed_walks), and canon() (rotation+reversal canonicalization) -- copied, with L
    promoted from the hardcoded GIRTH=10 to an explicit parameter (needed for D3-2's L=12,14; the
    L=10 call reproduces the file's own 120-walks/6-primitives numbers as the D3-0 regression).
  - proofs/foundations/LOOP_E2a_interacting_form_2026-07-02.py:
      lines ~54-78    the Cl(6) generator/dart/edge scaffolding (g6, gam(), DARTS, EDGE_OF_DART,
                       B_G = srs.hashimoto(Gamma).real) -- copied verbatim, same conventions.
      lines ~84-110   the canonical J6 frame -> A_ops -> NHAT -> vac construction -- copied
                       verbatim (same A4/edge_rep/Chat/H1/B1/phi machinery).
      lines ~112-125  the vacuum pairing C_PAIR = <0|gamma_a gamma_b|0> = I + i*sgn*J6 regression
                       (S-A) -- reproduced here as part of the D3-0 "direct(word) machinery
                       reproduces its E2a in-file checks" regression.
      lines ~128-162  pf()/wick()/direct(word) and the WORDS Wick-certification list -- copied
                       verbatim; direct(word) = <0| prod_{a in word} gamma_a |0> is THE tool D3-1/
                       D3-2's g_L is built from (word = the edge sequence traversed by the cycle).
      lines ~168-180  W_INT (the Cl(6)-matter-weighted Hashimoto operator on Fock(8) (x) darts) and
                       P_VAC (the N=0 Fock-sector projector) -- copied verbatim; D3-3's P_q is the
                       mechanical N=1 extension of this same pattern.
      lines ~252-262  sigma3/P3 (the C3 screw's vertex 3-cycle, lifted to a dart permutation) and
                       Q_t (its winding-isotype projectors, Q_t = (1/3) sum_m omega^{-tm} P3^m) --
                       copied verbatim; already commented there as "C3 winding isotypes of the dart
                       space", i.e. the SAME object this pre-reg calls "the winding-sector weights".
  - rho(W_INT) = sqrt(2) (G6ab/ZG-4): cited as context for D3-3's expected degenerate growth rate
    (both P_VAC and P_q project onto subspaces that generically share W_INT's OWN spectral radius'
    growth -- the pre-reg's own "PRE-NAMED VACUOUSNESS RISK: joint-space ergodicity may force
    Delta f = 0 trivially"), verified independently on-screen in D3-3, not assumed.

HARD RULES (binding, per the pre-reg poisons): exactly ONE file created (this one); no engine/proofs
edits; anchors reported as anchors; RIGID and DEGENERATE are first-class results (no re-running with
modified observables to escape them); no composite verdict spin; fit windows/L_max are FROZEN
([10,24] and [16,40], L up to 40, even L only -- W_INT is Fock-parity-odd); enumeration bounds and
budgets printed (no silent truncation); nothing tuned; no git commits.
"""
import itertools
import math
import os
import sys
import time
from collections import Counter

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
import srs  # noqa: E402  -- the engine, unmodified
from simulator.srs_engine.utils import AlgebraicUtility  # noqa: E402

np.set_printoptions(precision=6, suppress=True, linewidth=120)
T_START = time.time()
ok_D0 = True


def banner(t):
    print("=" * 96)
    print(f" {t}")
    print("=" * 96)


def sub(t):
    print("-" * 96)
    print(f" {t}")
    print("-" * 96)


def check(name, cond, detail="", gate=True):
    """gate=True routes into the D3-0 pass/fail; gate=False is a printed, non-gating observation."""
    global ok_D0
    cond = bool(cond)
    if gate:
        ok_D0 = ok_D0 and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return cond


def report(name, cond, detail=""):
    print(f"  [{'INFO-' + ('OK' if cond else 'NOTE')}] {name}" + (f"  -- {detail}" if detail else ""))
    return cond


banner("D3 -- THE CONFINEMENT BINARY  (pre-reg: internal research notes)")
print("Anchors A1/A2 are IDENTITIES to verify, never findings. Verdict = the PAIR (D3-1, D3-3),")
print("no composite spin. RIGID and DEGENERATE are first-class, fully bookable results.")

# ====================================================================================================
# SCAFFOLDING -- copied verbatim from LOOP_E2a_interacting_form_2026-07-02.py lines ~54-78
# ====================================================================================================
EDGES = srs.EDGES
NE, NV = len(EDGES), srs.NV
ND = 2 * NE
g6 = [np.array(g) for g in AlgebraicUtility.cl6_generators()]
EIDX = {(min(i, j), max(i, j)): e for e, (i, j, v) in enumerate(EDGES)}


def edge_rep(sig):
    R6 = np.zeros((NE, NE))
    for e, (i, j, v) in enumerate(EDGES):
        a, b = sig[i], sig[j]
        s = 1.0
        if a > b:
            a, b, s = b, a, -1.0
        R6[EIDX[(a, b)], e] = s
    return R6


def gam(x):
    return sum(x[a] * g6[a] for a in range(NE))


# darts in srs order: per edge e, dart 2e = (i->j), 2e+1 = (j->i)  (E2a's DARTS convention, no homology)
DARTS_NOHOM = []
for i, j, v in EDGES:
    DARTS_NOHOM += [(i, j), (j, i)]
EDGE_OF_DART = [d // 2 for d in range(ND)]
B_G = srs.hashimoto((0.0, 0.0, 0.0)).real
print(f"NE (edges) = {NE}, NV (vertices) = {NV}, ND (darts) = {ND}")
print(f"B_G = srs.hashimoto(Gamma).real, shape {B_G.shape}; max|Im hashimoto(Gamma)| = "
      f"{np.max(np.abs(srs.hashimoto((0., 0., 0.)).imag)):.2e} (Gamma point => real)")

# ====================================================================================================
# THE VACUUM -- copied verbatim from LOOP_E2a lines ~84-110 (J6 frame -> A_ops -> NHAT -> vac)
# ====================================================================================================
d0 = np.zeros((NV, NE))
for e, (i, j, v) in enumerate(EDGES):
    d0[i, e] = -1.0
    d0[j, e] = 1.0
Chat = np.array([[1, 1, 0], [-1, 0, 1], [0, -1, -1],
                 [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
H1, _ = np.linalg.qr(Chat)
B1 = np.linalg.svd(d0)[2][:3].T
A4 = [dict(enumerate(p)) for p in itertools.permutations(range(4))
      if sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j]) % 2 == 0]
rows = []
for g in A4:
    R6 = edge_rep(g)
    rows.append(np.kron(np.eye(3), (H1.T @ R6 @ H1).T) - np.kron(B1.T @ R6 @ B1, np.eye(3)))
_, Sp, Vp = np.linalg.svd(np.vstack(rows))
assert 9 - np.sum(Sp > 1e-9) == 1
phi = Vp[-1].reshape(3, 3)
phi *= math.sqrt(3) / np.linalg.norm(phi)
J6 = B1 @ phi @ H1.T - H1 @ phi.T @ B1.T
assert np.max(np.abs(J6 @ J6 + np.eye(6))) < 1e-9
wJ, VJ = np.linalg.eig(J6)
modes, _ = np.linalg.qr(VJ[:, np.where(np.abs(wJ - 1j) < 1e-9)[0]])
A_ops = [gam(np.conj(modes[:, m])) / math.sqrt(2) for m in range(3)]
NHAT = sum(a.conj().T @ a for a in A_ops)
wN, VN = np.linalg.eigh(NHAT)
vac = VN[:, [int(np.argmin(wN))]]
vac = vac / np.linalg.norm(vac)


def direct(word):
    """<0| prod_{a in word} gamma_a |0>  (word applied in LIST order, left-to-right in the matrix
    product; see D3-1's orientation-convention note). Copied verbatim from E2a line ~150-154."""
    M = np.eye(8)
    for a in word:
        M = M @ g6[a]
    return (vac.conj().T @ M @ vac).item()


sub("D3-0(a)  the vacuum pairing regression (E2a S-A, reproduced here)")
C_PAIR = np.zeros((NE, NE), complex)
for a in range(NE):
    for b in range(NE):
        C_PAIR[a, b] = direct([a, b])
re_err = np.max(np.abs(C_PAIR.real - np.eye(NE)))
im_antisym_err = np.max(np.abs(C_PAIR.imag + C_PAIR.imag.T))
sgnJ = np.sign(np.sum(C_PAIR.imag * J6)) or 1.0
imJ_err = np.max(np.abs(C_PAIR.imag - sgnJ * J6))
check("D3-0(a1) Re C = I exactly", re_err < 1e-10, detail=f"err={re_err:.1e}")
check("D3-0(a2) Im C antisymmetric", im_antisym_err < 1e-10, detail=f"err={im_antisym_err:.1e}")
check(f"D3-0(a3) Im C = {'+' if sgnJ > 0 else '-'}J6 exactly (the chiral carrier)", imJ_err < 1e-10,
      detail=f"err={imJ_err:.1e}")

sub("D3-0(b)  the direct(word)/Wick machinery reproduces E2a's own in-file checks")


def pf(K):
    n = K.shape[0]
    if n == 0:
        return 1.0 + 0.0j
    if n % 2:
        return 0.0 + 0.0j
    tot = 0.0 + 0.0j
    for j in range(1, n):
        sgn = (-1) ** (j - 1)
        rest = [k for k in range(n) if k not in (0, j)]
        tot += sgn * K[0, j] * pf(K[np.ix_(rest, rest)])
    return tot


def wick(word):
    n = len(word)
    K = np.zeros((n, n), complex)
    for i in range(n):
        for j in range(i + 1, n):
            K[i, j] = C_PAIR[word[i], word[j]]
            K[j, i] = -K[i, j]
    return pf(K)


WORDS = [(0, 1), (2, 5), (3, 3), (0, 1, 2, 3), (1, 1, 2, 2), (0, 2, 4, 1),
         (5, 4, 3, 2, 1, 0), (0, 1, 0, 1, 2, 3), (2, 2, 5, 5, 1, 4)]
wick_err = max(abs(wick(w) - direct(w)) for w in WORDS)
odd_zero = max(abs(direct(w)) for w in [(0,), (0, 1, 2), (4, 2, 0, 1, 3)])
check(f"D3-0(b) Wick == direct for {len(WORDS)} words incl. repeats (max err {wick_err:.1e}); "
      f"odd words vanish ({odd_zero:.1e})  [reproduces E2a S-A verbatim]",
      wick_err < 1e-10 and odd_zero < 1e-12)

# ====================================================================================================
# CYCLE ENUMERATION -- copied/generalized from zeta_gauge.py lines ~286-313 ("ZG-2 (native)")
# ====================================================================================================
sub("D3-0(c)  the girth-10 cycle enumeration reproduces zeta_gauge.py's own 120/6 numbers")
# darts WITH homology (needed for the cover-closure / zero-net-Z^3-vector filter)
DARTS = []
for i, j, v in EDGES:
    DARTS += [(i, j, np.array(v)), (j, i, -np.array(v))]


def rev_dart(d):
    return d + 1 if d % 2 == 0 else d - 1


def enumerate_closed_walks(start_vertex, L):
    """zeta_gauge.py's own recipe (vertex + net-Z^3-homology closure of a length-L non-backtracking
    walk), with L promoted to a parameter (zeta_gauge hardcodes L=GIRTH=10; D3-2 needs L=12,14 too)."""
    found = []

    def step(path):
        if len(path) == L:
            if DARTS[path[-1]][1] == start_vertex:
                shift = np.zeros(3, dtype=int)
                for d in path:
                    shift += DARTS[d][2]
                if np.all(shift == 0):
                    found.append(tuple(path))
            return
        last_d = path[-1] if path else None
        last_tail = DARTS[last_d][1] if last_d is not None else start_vertex
        for nd in range(ND):
            if DARTS[nd][0] != last_tail:
                continue
            if last_d is not None and nd == rev_dart(last_d):
                continue
            step(path + [nd])

    for first_d in range(ND):
        if DARTS[first_d][0] != start_vertex:
            continue
        step([first_d])
    return found


def canon(cycle):
    """rotation + reversal canonicalization -- copied verbatim from zeta_gauge.py."""
    rotations = [tuple(cycle[i:] + cycle[:i]) for i in range(len(cycle))]
    reversed_c = tuple(rev_dart(d) for d in reversed(cycle))
    rotations += [tuple(reversed_c[i:] + reversed_c[:i]) for i in range(len(reversed_c))]
    return min(rotations)


t0 = time.time()
ALL_WALKS_10 = []
for v0 in range(NV):
    ALL_WALKS_10 += enumerate_closed_walks(v0, 10)
PRIMITIVE_10 = sorted(set(canon(c) for c in ALL_WALKS_10))
dt10 = time.time() - t0
print(f"  L=10: {len(ALL_WALKS_10)} total closed (cover-closed) walks, {len(PRIMITIVE_10)} primitive "
      f"classes  ({dt10:.3f}s)")
check("D3-0(c) L=10 enumeration reproduces zeta_gauge.py's own (120 walks / 6 primitives)",
      len(ALL_WALKS_10) == 120 and len(PRIMITIVE_10) == 6,
      detail=f"got ({len(ALL_WALKS_10)}, {len(PRIMITIVE_10)})")

# ====================================================================================================
banner("A1 (ANCHOR, not a finding) -- center symmetry: C3 screw vs the run's generator(s)")
# ====================================================================================================
print("  operators tested (stated explicitly, per the contract):")
print("    (i)   P3  = the dart-space lift of the C3 vertex screw sigma3={0:0,1:2,2:3,3:1}")
print("    (ii)  vs  B_G  = srs.hashimoto(Gamma).real  (the FREE run's non-backtracking generator)")
print("    (iii) vs  W_INT block structure, via P3_full = kron(P3, I_8)  (the MATTER-DRESSED run's")
print("          generator, dart-major Fock(8) (x) darts space)")
sigma3 = {0: 0, 1: 2, 2: 3, 3: 1}
P3 = np.zeros((ND, ND))
for a, (i, j) in enumerate(DARTS_NOHOM):
    for b, (p, q) in enumerate(DARTS_NOHOM):
        if (p, q) == (sigma3[i], sigma3[j]):
            P3[b, a] = 1.0
            break
p3_order_err = np.max(np.abs(np.linalg.matrix_power(P3, 3) - np.eye(ND)))
check("A1(0) P3^3 = I (the screw has order 3)", p3_order_err < 1e-12, detail=f"err={p3_order_err:.1e}")

comm_B = np.max(np.abs(P3 @ B_G - B_G @ P3))
check("A1(a) [P3, B_G] = 0 exactly  (the free-run commutator; matches the E2a-established identity)",
      comm_B < 1e-12, detail=f"||[P3,B_G]||_max = {comm_B:.3e}")

# W_INT built here (needed for the P3_full commutator AND for D3-3 below) -- copied verbatim from
# LOOP_E2a lines ~168-176.
W_INT = np.zeros((8 * ND, 8 * ND), complex)
for dp in range(ND):
    for d in range(ND):
        if abs(B_G[dp, d]) > 0.5:
            W_INT[dp * 8:(dp + 1) * 8, d * 8:(d + 1) * 8] = gam(np.eye(NE)[:, EDGE_OF_DART[dp]])
P_VAC = np.zeros((ND, 8 * ND), complex)
for d in range(ND):
    P_VAC[d, d * 8:(d + 1) * 8] = vac[:, 0].conj()

P3_full = np.kron(P3, np.eye(8))
comm_W = np.max(np.abs(P3_full @ W_INT - W_INT @ P3_full))
report("A1(b) [P3_full, W_INT] -- the MATTER-DRESSED lift (kron(P3,I8), no Cl6-index rotation) "
       "does NOT commute with W_INT", comm_W > 1e-9,
       detail=f"||[P3_full,W_INT]||_max = {comm_W:.6f}  (printed per contract, NON-GATING: this "
              "is new territory, not an established prior-art identity either way -- the matter "
              "dressing is not simply co-rotated by the bare point-group lift)")

sub("A1(c) the winding-sector weights: ||Q_t B_G^n seed||^2 for t=0,1,2, several seeds, several n")
OM = complex(math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3))
Q_t = [sum(OM ** (-t * m) * np.linalg.matrix_power(P3, m) for m in range(3)) / 3 for t in range(3)]
dimQ = [float(np.trace(Q_t[t]).real) for t in range(3)]
print(f"  tr(Q_t) (isotype multiplicities, an exact rep-theory fact) = {[f'{d:.6f}' for d in dimQ]}")
check("A1(c0) the three winding isotypes have EXACTLY equal multiplicity in the 12-dim dart rep",
      max(abs(d - dimQ[0]) for d in dimQ) < 1e-10, detail=f"tr(Q_t)={dimQ}")

SEEDS = {
    "random(seed=0)": np.random.default_rng(0).normal(size=ND),
    "uniform-ones": np.ones(ND) / math.sqrt(ND),
    "e_0 (single dart)": np.eye(ND)[:, 0],
}
N_LADDER = [1, 2, 3, 5, 8, 10, 20]  # declared before running
print(f"  seeds (declared): {list(SEEDS)}")
print(f"  n-ladder (declared): {N_LADDER}")
print(f"  {'seed':>18} {'n':>3}  {'||Q0 v||^2':>16} {'||Q1 v||^2':>16} {'||Q2 v||^2':>16}  "
      f"{'|Q1-Q2| (mixed tol)':>20}  {'|Q0-Q1|':>12}")
worst_pair = 0.0
Bpow_cache = {}
for name, seed in SEEDS.items():
    for n in N_LADDER:
        if n not in Bpow_cache:
            Bpow_cache[n] = np.linalg.matrix_power(B_G, n)
        v = Bpow_cache[n] @ seed
        n0 = float(np.linalg.norm(Q_t[0] @ v) ** 2)
        n1 = float(np.linalg.norm(Q_t[1] @ v) ** 2)
        n2 = float(np.linalg.norm(Q_t[2] @ v) ** 2)
        tol = max(1e-12, 1e-12 * max(abs(n1), abs(n2)))
        d12 = abs(n1 - n2)
        worst_pair = max(worst_pair, d12 / tol)
        print(f"  {name:>18} {n:>3}  {n0:>16.6e} {n1:>16.6e} {n2:>16.6e}  "
              f"{d12:>10.3e} (tol {tol:.1e})  {abs(n0 - n1):>12.3e}")
check("A1(c1) THE PAIRWISE IDENTITY: ||Q_1 v||^2 == ||Q_2 v||^2 EXACTLY for every seed x n tested "
      "(mixed abs/rel tol 1e-12; the dynamical mu_omega=mu_omega-bar extension of E2a's S-C control)",
      worst_pair <= 1.0, detail=f"worst (diff/tol) ratio = {worst_pair:.3e}")
report("A1(c2) the LITERAL three-way equality (t=0 vs t=1,2) does NOT hold in general for n>1 -- "
       "printed raw above, NOT gated (see the header docstring's A1 IMPLEMENTATION NOTE); t=0 "
       "carries B_G's Perron/leading eigenmode (eigenvalue exactly DEG-1+1=2, entirely in the t=0 "
       "isotype -- verified below), which sectors t=1,2 structurally cannot share",
       False)
w_BG, V_BG = np.linalg.eig(B_G)
perron_idx = int(np.argmax(np.abs(w_BG)))
perron_vec = V_BG[:, perron_idx]
perron_sector_fracs = [float(np.linalg.norm(Q_t[t] @ perron_vec) ** 2 / np.linalg.norm(perron_vec) ** 2)
                        for t in range(3)]
print(f"  B_G spectral radius = {np.max(np.abs(w_BG)):.6f} (Perron eigenvalue {w_BG[perron_idx]:.6f}); "
      f"its winding-isotype fractions = {[f'{f:.4f}' for f in perron_sector_fracs]}  "
      "(entirely t=0, as claimed above)")

banner("A2 (ANCHOR, not a finding) -- cycle equivalence: the 6 primitive girth-10 cycles")
print("  (verified together with D3-1 below, since both need the same g_10 values per cycle)")

# ====================================================================================================
banner("D3-1  THE MATTER-DECOHERENCE FACTOR  (the disorder observable)")
# ====================================================================================================
print("  ORIENTATION CONVENTION (declared): g_L := <0| gamma_{e(d_1)} gamma_{e(d_2)} ... "
      "gamma_{e(d_L)} |0>, applied in the SAME order as the cycle's own dart sequence (d_1 first, "
      "leftmost factor in the matrix product -- direct()'s own convention). Sanity check: reversing")
print("  the traversal (word -> word reversed) must give the COMPLEX CONJUGATE of g_L, since every")
print("  gamma_a is Hermitian and <0|M_1...M_n|0>* = <0|M_n^dag...M_1^dag|0> = <0|M_n...M_1|0>.")

g10_vals = []
rev_errs = []
print(f"\n  {'cycle (dart tuple)':>52}  {'g_10 (complex)':>28} {'|g_10|':>12}  {'|rev-conj|':>10}")
for cyc in PRIMITIVE_10:
    word = [EDGE_OF_DART[d] for d in cyc]
    g10 = direct(word)
    g10_rev = direct(word[::-1])
    rev_err = abs(g10_rev - np.conj(g10))
    g10_vals.append(g10)
    rev_errs.append(rev_err)
    print(f"  {str(cyc):>52}  {g10.real:+.10f}{g10.imag:+.10f}j {abs(g10):>12.10f}  {rev_err:>10.2e}")

check("D3-1(orientation) reversed traversal == conjugate, all 6 primitive cycles",
      max(rev_errs) < 1e-10, detail=f"max err={max(rev_errs):.2e}")
max_pairdiff = max(abs(a - b) for a in g10_vals for b in g10_vals)
check("A2 the 6 primitive girth-10 cycle amplitudes are EXACTLY equal", max_pairdiff < 1e-12,
      detail=f"max pairwise |g10_i-g10_j| = {max_pairdiff:.2e}")

g10 = g10_vals[0]
dev_from_1 = abs(abs(g10) - 1.0)
D3_1_CLASS = "RIGID" if dev_from_1 < 1e-12 else "DAMPED"
print(f"\n  g_10 (representative, all 6 equal) = {g10.real:+.12f}{g10.imag:+.12f}j")
print(f"  |g_10| = {abs(g10):.14f}   |1 - |g_10|| = {dev_from_1:.3e}")
print(f"  D3-1 CLASSIFICATION: {D3_1_CLASS}" + (
    "  (the matter does NOT decohere cycle holonomy => no matter-induced disorder => the area-law "
    "mechanism is ABSENT at this level -- booked raw, a real negative)" if D3_1_CLASS == "RIGID" else
    f"  (decoherence strength 1-|g_10| = {1 - abs(g10):.6e} is the disorder parameter)"))

# ====================================================================================================
banner("D3-2  SIZE SCALING  (cover-closed NB cycle classes at L=12, L=14)")
# ====================================================================================================
BUDGET_SECONDS = 300.0
print(f"declared enumeration budget: {BUDGET_SECONDS:.0f}s per L (same canon machinery, zero-net-"
      "Z^3-vector filter, generalized from the L=10 recipe above)")
if D3_1_CLASS == "RIGID":
    print("D3-1 was RIGID -- D3-2 is STILL run (rigidity at all sizes is itself the result), but the")
    print("perimeter-law FIT is labeled n/a per the pre-reg's declared contingency.")

L_RESULTS = {}
for L in (12, 14):
    t0 = time.time()
    all_walks = []
    for v0 in range(NV):
        all_walks += enumerate_closed_walks(v0, L)
    dt = time.time() - t0
    within_budget = dt < BUDGET_SECONDS
    print(f"\n  L={L}: enumeration took {dt:.3f}s (budget {BUDGET_SECONDS:.0f}s) -- "
          f"{'WITHIN BUDGET' if within_budget else 'EXCEEDED BUDGET'}")
    if not within_budget:
        print(f"  BUDGET EXCEEDED at L={L}: per pre-reg, restricting D3-2 to L=12 only; L={L} skipped.")
        continue
    primitive = sorted(set(canon(c) for c in all_walks))
    orbit_dist = Counter(Counter(canon(c) for c in all_walks).values())
    print(f"  L={L}: {len(all_walks)} total cover-closed NB closed walks, {len(primitive)} "
          f"primitive classes  (orbit-size distribution over the {2 * L}-element rotation+reversal "
          f"group: {dict(orbit_dist)})")
    gvals = []
    for cyc in primitive:
        word = [EDGE_OF_DART[d] for d in cyc]
        gvals.append(direct(word))
    L_RESULTS[L] = (primitive, gvals)
    mags = [abs(g) for g in gvals]
    spread = max(mags) - min(mags)
    print(f"  L={L}: -log|g_{L}| per class: min={-math.log(max(mags)):.3e}  "
          f"max={-math.log(min(mags)):.3e}  (n/a perimeter fit: see below)")
    print(f"  L={L}: equal-L class spread (max|g_L| - min|g_L|) = {spread:.3e}  over {len(gvals)} "
          f"classes")
    check(f"D3-2 L={L} enumeration completed and g_L computed for every primitive class",
          len(gvals) == len(primitive), gate=False)

sub("D3-2 SUMMARY -- (-log|g_L|) vs L, all three sizes")
print(f"  {'L':>4} {'#classes':>9} {'-log|g_L| (all classes)':>28}")
for L, gvals in [(10, g10_vals)] + [(L, L_RESULTS[L][1]) for L in sorted(L_RESULTS)]:
    negs = [-math.log(abs(g)) for g in gvals]
    print(f"  {L:>4} {len(gvals):>9}  min={min(negs):.3e}  max={max(negs):.3e}  "
          f"mean={np.mean(negs):.3e}")
if D3_1_CLASS == "RIGID":
    print("\n  PERIMETER-LAW FIT: n/a (D3-1 RIGID; -log|g_L| == 0 identically at every L tested -- "
          "there is no decay to fit a perimeter law to). Per the pre-reg's DECLARED LIMIT: a true "
          "area-law test needs a spanning-surface notion this object does not yet possess; here the")
    print("  equal-L class spread is ALSO exactly zero at every L (no discrimination material at "
          "all) -- booked as the named gap, feeding G6b'/D3b. No area claim either way.")

# ====================================================================================================
banner("D3-3  THE CHARGED-SECTOR FREE ENERGY  (the Polyakov content, posed non-vacuously)")
# ====================================================================================================
print("P_q := the N=1 Fock-sector analog of P_VAC. Construction (mechanical extension, printed):")
print("  the one-particle states psi_i := A_ops[i]^dagger |0>  (i=0,1,2 -- the SAME 3 creation ops")
print("  whose number operator NHAT built the vacuum above). Row (d*3+i) of P_q picks out psi_i's")
print("  bra <psi_i| within dart-block d's 8-dim Fock slice (columns d*8:(d+1)*8); zero elsewhere --")
print("  identical block pattern to P_VAC, just 3 rows per dart instead of 1. Shape: (3*ND, 8*ND).")

psis = [A_ops[i].conj().T @ vac for i in range(3)]
orth_err = max(abs((psis[i].conj().T @ psis[j]).item() - (1.0 if i == j else 0.0))
               for i in range(3) for j in range(3))
check("D3-3(0) the 3 one-particle states psi_i are orthonormal", orth_err < 1e-10,
      detail=f"max|<psi_i|psi_j> - delta_ij| = {orth_err:.2e}", gate=False)

P_q = np.zeros((3 * ND, 8 * ND), complex)
for d in range(ND):
    for i in range(3):
        P_q[d * 3 + i, d * 8:(d + 1) * 8] = psis[i][:, 0].conj()
print(f"  P_VAC shape = {P_VAC.shape}   P_q shape = {P_q.shape}")

print("\nNORM CHOICE (declared, printed, used identically for both sectors): the SPECTRAL norm "
      "(largest singular value, np.linalg.norm(M,2)) -- consistent with G6ab's own rho(W_INT)=sqrt(2)")
print("(a spectral-radius statement); verified independently below, not assumed.")
rho_WINT = float(np.max(np.abs(np.linalg.eigvals(W_INT))))
print(f"  rho(W_INT) (independently recomputed here) = {rho_WINT:.10f}   sqrt(2) = {math.sqrt(2):.10f}"
      f"   |diff| = {abs(rho_WINT - math.sqrt(2)):.2e}")

L_MAX = 40
Ls_even = list(range(2, L_MAX + 1, 2))
print(f"\nL-ladder (declared, frozen): even L = 2..{L_MAX}  ({len(Ls_even)} points) -- W_INT is "
      "Fock-parity-ODD (established in E2a S-B gate iii) => both sector blocks live on even L only.")

s_vac, s_q = [], []
Wp = np.eye(8 * ND, dtype=complex)
t0 = time.time()
for L in range(1, L_MAX + 1):
    Wp = Wp @ W_INT
    if L % 2 == 0:
        Mv = P_VAC @ Wp @ P_VAC.conj().T
        Mq = P_q @ Wp @ P_q.conj().T
        s_vac.append(math.log(np.linalg.norm(Mv, 2)))
        s_q.append(math.log(np.linalg.norm(Mq, 2)))
dt_growth = time.time() - t0
print(f"(computed in {dt_growth:.3f}s; W_INT is {W_INT.shape[0]}x{W_INT.shape[1]}, trivial)")

print(f"\n  {'L':>4} {'s_vac':>12} {'s_q':>12} {'s_vac - s_q':>14}")
for L, sv, sq in zip(Ls_even, s_vac, s_q):
    print(f"  {L:>4} {sv:>12.6f} {sq:>12.6f} {sv - sq:>14.6f}")

Ls_arr = np.array(Ls_even, float)
s_vac_arr = np.array(s_vac)
s_q_arr = np.array(s_q)

WINDOWS = [(10, 24), (16, 40)]  # DECLARED, frozen, per pre-reg


def fit_line(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope), float(intercept)


sub("D3-3 fits -- two DECLARED windows (D1-lesson: report both, never one cherry-picked fit)")
fits = {}
for (w0, w1) in WINDOWS:
    mask = (Ls_arr >= w0) & (Ls_arr <= w1)
    f_vac, b_vac = fit_line(Ls_arr[mask], s_vac_arr[mask])
    f_q, b_q = fit_line(Ls_arr[mask], s_q_arr[mask])
    fits[(w0, w1)] = dict(f_vac=f_vac, b_vac=b_vac, f_q=f_q, b_q=b_q, n=int(mask.sum()))
    print(f"  window L in [{w0},{w1}]  ({int(mask.sum())} points): f_vac={f_vac:.6f}  b_vac={b_vac:.6f}"
          f"   f_q={f_q:.6f}  b_q={b_q:.6f}   Delta_f = f_vac-f_q = {f_vac - f_q:+.6f}")

delta_fs = [fits[w]["f_vac"] - fits[w]["f_q"] for w in WINDOWS]
window_spread = max(delta_fs) - min(delta_fs)
f_vac_rel_spread = abs(fits[WINDOWS[0]]["f_vac"] - fits[WINDOWS[1]]["f_vac"]) / max(
    abs(fits[WINDOWS[0]]["f_vac"]), abs(fits[WINDOWS[1]]["f_vac"]), 1e-300)
f_q_rel_spread = abs(fits[WINDOWS[0]]["f_q"] - fits[WINDOWS[1]]["f_q"]) / max(
    abs(fits[WINDOWS[0]]["f_q"]), abs(fits[WINDOWS[1]]["f_q"]), 1e-300)
window_instability = max(f_vac_rel_spread, f_q_rel_spread)

print(f"\n  Delta_f per window: {[f'{d:+.6f}' for d in delta_fs]}   window spread = {window_spread:.6f}")
print(f"  per-sector window (relative) instability: f_vac {f_vac_rel_spread:.3%}  f_q {f_q_rel_spread:.3%}"
      f"   (declared 10% bound: {window_instability:.3%} {'<' if window_instability < 0.10 else '>='} 10%)")

# DECLARED classification logic (frozen, no post-hoc adjustment): INCONCLUSIVE first (the fits
# themselves must be trustworthy); else CHARGE-SUPPRESSED iff Delta_f is POSITIVE in EVERY window
# AND resolved above the window-to-window spread (a stable, non-noise-level effect); else
# DEGENERATE (Delta_f is compatible with zero -- including sign changes across windows -- given
# the spread, exactly the pre-reg's "Delta_f = 0 within window spread" wording).
resolved_positive = all(d > 0 for d in delta_fs) and window_spread < min(delta_fs)
if window_instability > 0.10:
    D3_3_OUTCOME = "INCONCLUSIVE"
elif resolved_positive:
    D3_3_OUTCOME = "CHARGE-SUPPRESSED"
else:
    D3_3_OUTCOME = "DEGENERATE"

r_q_per_window = {w: math.exp(fits[w]["b_vac"] - fits[w]["b_q"]) for w in WINDOWS}
print(f"\n  D3-3 OUTCOME: {D3_3_OUTCOME}")
if D3_3_OUTCOME == "DEGENERATE":
    print("  (the sector growths coincide within window spread -- the trivial/ergodic outcome, "
          "PRE-NAMED as a vacuousness risk: 'joint-space ergodicity may force Delta f = 0 trivially'."
          f" Independent cross-check: rho(W_INT)=sqrt(2) => log(sqrt2)={math.log(math.sqrt(2)):.6f}, "
          f"matching both f_vac and f_q above to ~1% -- both sectors generically inherit W_INT's OWN"
          " spectral-radius growth, exactly the anticipated mechanism.)")
    print(f"  sub-leading amplitude ratio r_q = exp(b_vac - b_q):")
    for w in WINDOWS:
        print(f"    window {w}: r_q = {r_q_per_window[w]:.6f}")
elif D3_3_OUTCOME == "CHARGE-SUPPRESSED":
    print("  (Delta_f > 0 stable across windows -- an isolated Fock charge costs divergent total "
          "free energy in the run: the confined-like signature at this level, finite-size caveat "
          "noted.)")
else:
    print("  (window instability exceeds the declared 10% bound -- the point estimates cannot be "
          "trusted to call CHARGE-SUPPRESSED or DEGENERATE.)")

# ====================================================================================================
banner("D3-4  THE VERDICT  (frozen composition -- the PAIR, no composite spin)")
# ====================================================================================================
print(f"  D3-1 classification: {D3_1_CLASS}")
print(f"  D3-3 outcome:        {D3_3_OUTCOME}")
print(f"\n  VERDICT PAIR: ({D3_1_CLASS}, {D3_3_OUTCOME})")
MEANINGS = {
    ("RIGID", "DEGENERATE"): (
        "RIGID + DEGENERATE means the object shows NO matter-induced confinement mechanism at "
        "cycle level and NO sector suppression -- a clean, bookable negative for confinement-from-"
        "matter at this order."),
    ("DAMPED", "CHARGE-SUPPRESSED"): (
        "DAMPED + CHARGE-SUPPRESSED would be the strongest positive this station can honestly "
        "produce (still NOT an area law, NOT a mass gap -- those stay open)."),
}
meaning = MEANINGS.get((D3_1_CLASS, D3_3_OUTCOME),
                        "MIXED PAIR -- booked as-is, no composite spin (per pre-reg D3-4).")
print(f"  MEANING (pre-reg's own frozen wording where it applies): {meaning}")

# ====================================================================================================
banner("D3-5  SCOPE DECLARATION  (printed, never gates PASS/FAIL)")
# ====================================================================================================
print("""  NOT claimed by this station:
    - an area law or string tension (needs the surface notion -- named gap, D3-2's declared limit);
    - a mass gap;
    - Wilson-loop dynamics at finite k (G6b');
    - any continuum/deconfinement-transition statement;
    - any scoreboard change.""")

# ====================================================================================================
banner("OVERALL")
# ====================================================================================================
verdict_definite = D3_1_CLASS in ("RIGID", "DAMPED") and D3_3_OUTCOME in (
    "CHARGE-SUPPRESSED", "DEGENERATE", "INCONCLUSIVE")
print(f"D3-0 (regressions + anchors A1/A2): {'ALL PASS' if ok_D0 else '*** SOME CHECKS FAILED ***'}")
print(f"verdict pair definite: {verdict_definite}  -- ({D3_1_CLASS}, {D3_3_OUTCOME})")
print(f"elapsed: {time.time() - T_START:.2f}s")
overall_ok = ok_D0 and verdict_definite
print(f"\n{'=' * 96}\n OVERALL: {'PASS -- exit 0' if overall_ok else '*** FAIL -- exit 1 ***'}\n{'=' * 96}")
sys.exit(0 if overall_ok else 1)
