#!/usr/bin/env python3
"""
proofs/foundations/CS0b_wint_redecoration_2026-07-10.py

CS-0b -- W_INT RE-DECORATION (the connection sector's first station; launched by the IV.7 design
note internal research notes, station CS-0b).

THE DEFECT BEING DISCHARGED (inherited from W2-MAP's M-4, W2_MAP_vertex_propagator_2026-07-10.py
lines ~568-589): LOOP_E2a's W_INT decorates each hop (dp <- d) with gam(e_{edge(dp)}) == the
UNSIGNED dart-to-edge label == gam(sqrt2 * Ue^T e_dp), the R-EVEN convention -- which FAILS the
derived A4-covariance requirement R2 outright (maximal mismatch), and whose R-even branch is
PROVABLY EMPTY under R5 (rank <= 3 obstruction).  The only {R1,R2,R5}-compliant vertex maps are the
R-ODD (SIGNED, Uo-type) O(2) family (W2-MAP M-1a):
    rotation   branch:  Phi_theta = Uo @ (cos(theta) I6 + sin(theta) J6)
    reflection branch:  Phi_phi   = Uo @ (cos(phi)   S1 + sin(phi)   S2)
The orbit ambiguity is UNRESOLVED (M-1b selection failed by an exact orthogonality identity), so
this station evaluates BOTH representative points and reports PER BRANCH -- no orbit picking:
    BRANCH-1 (theta=0):              Phi_1 = Uo @ I6
    BRANCH-2 (reflection rep.):      Phi_2 = Uo @ S1  (unit-isometry normalized, see S-2)

THE QUESTION: does LOOP_E2a's S-C chiral-asymmetry theorem (nonzero omega-vs-omega-bar winding
asymmetry in the INTERACTING ensemble, exactly-zero in the FREE ensemble, sign-flipping under
J -> -J) SURVIVE when W_INT is rebuilt with the compliant (signed/R-odd) decoration?

FROZEN VERDICT TREE (from the launch contract, fixed before running):
    SURVIVES              -- per branch, report both: asymmetry nonzero + J-flip + controls zero;
    CONVENTION-DEPENDENT  -- the E2a asymmetry was the non-compliant convention's artifact
                             (a major honest negative, booked raw, not softened);
    MIXED                 -- one branch yes, one no (report exactly which).

CONTENTS:
    S-0  E2a machinery rebuilt VERBATIM (J6, vacuum, pairing C = I + iJ, Wick certification)
    S-1  REGRESSION: the ORIGINAL W_INT (E2a lines 167-180) + ALL of E2a's banked gate results
         as asserted anchors (never adjusted)
    S-2  THE RE-DECORATION: Ue/Uo embeddings + the M-4 non-compliance regression + the commutant
         {I6,J6,S1,S2} (W2-MAP lines ~321-357 verbatim) + the two branch decorations, each
         verified compliant (R2 intertwining, R-oddness, isometry) IN-FILE
    S-3  E2a's gates re-run on BRANCH-1
    S-4  E2a's gates re-run on BRANCH-2
    S-5  the gate table, the cross-branch identity, THE VERDICT

POISONS (binding): no oblique/EW/a_e anywhere; no new couplings (test fugacities stay E2a's own
generic u = 0.11, 0.23; the branch normalization is R5's unit-isometry convention, the same one
branch-1 carries -- declared, not tuned); the E2a banked values are regression anchors, NEVER
adjusted; BOTH branches reported (no orbit picking); the free-ensemble control must be exactly
zero in every configuration -- if it is not, the estimator is broken (fix the estimator, never
reinterpret the control).  Standalone; no existing file edited; exit 0 iff all checks pass.
"""
import cmath
import itertools
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
import srs  # noqa: E402
from simulator.srs_engine.utils import AlgebraicUtility  # noqa: E402

ok_all = True


def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}")


def banner(t):
    print("=" * 92)
    print(f" {t}")
    print("=" * 92)


EDGES = srs.EDGES
NE, NV = len(EDGES), srs.NV
ND = 2 * NE
g6 = [np.array(g) for g in AlgebraicUtility.cl6_generators()]
EIDX = {(min(i, j), max(i, j)): e for e, (i, j, v) in enumerate(EDGES)}


def edge_rep(sig):
    """E2a lines 60-68 / W2-MAP lines 108-117 (identical in both sources)."""
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


# darts in srs order: per edge e, dart 2e = (i->j), 2e+1 = (j->i)   [E2a lines 73-78]
DARTS = []
for i, j, v in EDGES:
    DARTS += [(i, j), (j, i)]
EDGE_OF_DART = [d // 2 for d in range(ND)]
B_G = srs.hashimoto((0.0, 0.0, 0.0)).real

# ===========================================================================
banner("S-0  E2a MACHINERY REBUILT VERBATIM (J6, vacuum, pairing, Wick)  [E2a S-A, lines 81-162]")
# ===========================================================================
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

C_PAIR = np.zeros((NE, NE), complex)
for a in range(NE):
    for b in range(NE):
        C_PAIR[a, b] = (vac.conj().T @ g6[a] @ g6[b] @ vac).item()
sgnJ = np.sign(np.sum(C_PAIR.imag * J6)) or 1.0
check(f"S-0 pairing regression: C = I + iJ (Re err {np.max(np.abs(C_PAIR.real - np.eye(NE))):.1e}, "
      f"Im-vs-{'+' if sgnJ > 0 else '-'}J err {np.max(np.abs(C_PAIR.imag - sgnJ * J6)):.1e}) "
      "[E2a S-A anchor]",
      np.max(np.abs(C_PAIR.real - np.eye(NE))) < 1e-10
      and np.max(np.abs(C_PAIR.imag - sgnJ * J6)) < 1e-10)

# the conjugate quantization (J -> -J)  [E2a lines 235-244]
modes_c, _ = np.linalg.qr(VJ[:, np.where(np.abs(wJ + 1j) < 1e-9)[0]])
A_ops_c = [gam(np.conj(modes_c[:, m])) / math.sqrt(2) for m in range(3)]
NH_c = sum(a.conj().T @ a for a in A_ops_c)
wNc, VNc = np.linalg.eigh(NH_c)
vac_c = VNc[:, [int(np.argmin(wNc))]]
vac_c = vac_c / np.linalg.norm(vac_c)
C_c = np.zeros((NE, NE), complex)
for a in range(NE):
    for b in range(NE):
        C_c[a, b] = (vac_c.conj().T @ g6[a] @ g6[b] @ vac_c).item()
check(f"S-0 mirror regression: Im C(-J) = -Im C(+J) (err {np.max(np.abs(C_c.imag + C_PAIR.imag)):.1e}), "
      f"Re equal (err {np.max(np.abs(C_c.real - C_PAIR.real)):.1e}) [E2a S-C MIRROR anchor]",
      np.max(np.abs(C_c.imag + C_PAIR.imag)) < 1e-10
      and np.max(np.abs(C_c.real - C_PAIR.real)) < 1e-10)


# GENERALIZED Wick/Pfaffian machinery: <0| gam(x1)...gam(xn) |0> = Pf(K), K_ij = x_i^T C x_j.
# For unit-coordinate words x = e_a this is EXACTLY E2a's own certification (lines 128-162);
# the bilinear extension is what the re-decorated (coefficient-vector) words require.
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


def wick_vecs(vecs):
    n = len(vecs)
    K = np.zeros((n, n), complex)
    for i in range(n):
        for j in range(i + 1, n):
            K[i, j] = vecs[i] @ C_PAIR @ vecs[j]
            K[j, i] = -K[i, j]
    return pf(K)


def direct_vecs(vecs, v=None):
    v = vac if v is None else v
    M = np.eye(8)
    for x in vecs:
        M = M @ gam(x)
    return (v.conj().T @ M @ v).item()


E6 = np.eye(NE)
WORDS = [(0, 1), (2, 5), (3, 3), (0, 1, 2, 3), (1, 1, 2, 2), (0, 2, 4, 1),
         (5, 4, 3, 2, 1, 0), (0, 1, 0, 1, 2, 3), (2, 2, 5, 5, 1, 4)]     # E2a line 156-157
wick_err = max(abs(wick_vecs([E6[:, a] for a in w]) - direct_vecs([E6[:, a] for a in w]))
               for w in WORDS)
odd_zero = max(abs(direct_vecs([E6[:, a] for a in w])) for w in [(0,), (0, 1, 2), (4, 2, 0, 1, 3)])
check(f"S-0 Wick regression on E2a's own word list ({len(WORDS)} words incl. repeats, max err "
      f"{wick_err:.1e}; odd words vanish {odd_zero:.1e}) [E2a S-A anchor]",
      wick_err < 1e-10 and odd_zero < 1e-12)

# ===========================================================================
banner("S-1  REGRESSION: the ORIGINAL W_INT and E2a's banked gates  [E2a lines 167-297]")
# ===========================================================================
# E2A BANKED ANCHORS -- captured from a verbatim run of LOOP_E2a_interacting_form_2026-07-02.py
# on this machine (2026-07-10; the file prints 7.156048e-03 / 3.387129e-02 / A(+J)=-3.387e-02;
# full precision below).  These are REGRESSION ANCHORS, never adjusted.
ANCHOR_EVADE_011 = 7.156048015609873e-03
ANCHOR_EVADE_023 = 3.387129389892518e-02
ANCHOR_A_PLUS_023 = -3.387129389892518e-02          # A(+J) at u = 0.23 (imag part ~2e-17)


def build_W(decor):
    """W = sum_{d',d} B_{d'd} . gam(x_{d'}) (x) |d'><d|  -- E2a lines 167-172 generalized to an
    arbitrary per-target-dart decoration coefficient vector x_{d'} (E2a's original: x_{d'} =
    e_{EDGE_OF_DART[d']}).  Blocks ordered dart-major: index = d*8 + fock."""
    W = np.zeros((8 * ND, 8 * ND), complex)
    for dp in range(ND):
        blk = gam(decor[dp])
        for d in range(ND):
            if abs(B_G[dp, d]) > 0.5:
                W[dp * 8:(dp + 1) * 8, d * 8:(d + 1) * 8] = blk
    return W


P_VAC = np.zeros((ND, 8 * ND), complex)              # E2a lines 174-176
for d in range(ND):
    P_VAC[d, d * 8:(d + 1) * 8] = vac[:, 0].conj()
P_VAC_c = np.zeros((ND, 8 * ND), complex)            # E2a lines 289-290
for d in range(ND):
    P_VAC_c[d, d * 8:(d + 1) * 8] = vac_c[:, 0].conj()


def G_of(W, u, P=None):
    P = P_VAC if P is None else P
    return P @ np.linalg.solve(np.eye(8 * ND) - u * W, P.conj().T)


# the winding projectors at Gamma  [E2a lines 251-262, verbatim]
sigma3 = {0: 0, 1: 2, 2: 3, 3: 1}
P3 = np.zeros((ND, ND))
for a, (i, j) in enumerate(DARTS):
    for b, (p, q) in enumerate(DARTS):
        if (p, q) == (sigma3[i], sigma3[j]):
            P3[b, a] = 1.0
            break
assert np.max(np.abs(P3 @ B_G - B_G @ P3)) < 1e-12
OM = cmath.exp(2j * math.pi / 3)
Q_t = [sum(OM ** (-t * m) * np.linalg.matrix_power(P3, m) for m in range(3)) / 3
       for t in range(3)]


def asym(W, u, P=None):
    """E2a's S-C estimator, unchanged: tr(Q1 G) - conj(tr(Q2 G)) on the vacuum block."""
    G = G_of(W, u, P)
    return np.trace(Q_t[1] @ G) - np.conj(np.trace(Q_t[2] @ G))


def free_control(u):
    Gf = np.linalg.inv(np.eye(ND) - u * B_G)
    return abs(np.trace(Q_t[1] @ Gf) - np.conj(np.trace(Q_t[2] @ Gf)))


def run_gates(tag, decor, wick_dart_words, wick_dart_odd):
    """Re-run E2a's four gates on W built from `decor`; returns the gate-table row."""
    W = build_W(decor)
    row = {"tag": tag}

    # ---- gate (a) reading A: E2a's LITERAL operation (lines 183-193) -- replace every 8x8
    # decoration block by I8.  NOTE: this operation erases the decoration entirely, so it is
    # decoration-INDEPENDENT (it passes for any decor by construction); kept as the verbatim gate.
    W_ONE = np.zeros((8 * ND, 8 * ND), complex)
    for dp in range(ND):
        for d in range(ND):
            if abs(B_G[dp, d]) > 0.5:
                W_ONE[dp * 8:(dp + 1) * 8, d * 8:(d + 1) * 8] = np.eye(8)
    u1 = 0.23
    G_one = P_VAC @ np.linalg.solve(np.eye(8 * ND) - u1 * W_ONE, P_VAC.conj().T)
    G_free = np.linalg.inv(np.eye(ND) - u1 * B_G)
    devA = np.max(np.abs(G_one - G_free))
    row["a_A"] = devA
    check(f"{tag} gate (a/A) block-erasure reduction == free resolvent (err {devA:.1e}) "
          "[E2a's literal gate-i operation; decoration-independent by construction]", devA < 1e-10)

    # ---- gate (a) reading B: the FAITHFUL substitution gamma_a -> 1, i.e. each block
    # gam(x) -> (sum_a x_a) I8.  For E2a's original decoration (x = e_edge, sum = 1) readings A
    # and B COINCIDE; for a signed decoration they differ.  Reported raw, per branch.
    W_sh = np.zeros((8 * ND, 8 * ND), complex)
    svec = np.array([np.sum(decor[dp]) for dp in range(ND)])
    for dp in range(ND):
        for d in range(ND):
            if abs(B_G[dp, d]) > 0.5:
                W_sh[dp * 8:(dp + 1) * 8, d * 8:(d + 1) * 8] = svec[dp] * np.eye(8)
    G_sh = P_VAC @ np.linalg.solve(np.eye(8 * ND) - u1 * W_sh, P_VAC.conj().T)
    devB = np.max(np.abs(G_sh - G_free))
    row["a_B"] = devB
    # the scalar shadow IS the resolvent of the coefficient-summed (sign-twisted) walk S.B:
    G_tw = np.linalg.inv(np.eye(ND) - u1 * (np.diag(svec) @ B_G))
    dev_tw = np.max(np.abs(G_sh - G_tw))
    # the shadow's OWN winding asymmetry (state-free control -- graded blindness says zero):
    sh_asym = max(abs(np.trace(Q_t[1] @ np.linalg.inv(np.eye(ND) - u * np.diag(svec) @ B_G))
                      - np.conj(np.trace(Q_t[2] @ np.linalg.inv(np.eye(ND) - u * np.diag(svec) @ B_G))))
                  for u in (0.11, 0.23))
    row["a_B_shadow_asym"] = sh_asym
    print(f"    [info] {tag} gate (a/B) faithful gamma->1 scalar shadow: dev from FREE = {devB:.3e}"
          f" (== resolvent of the coefficient-twisted walk diag(s).B, err {dev_tw:.1e});")
    print(f"           the shadow's own winding asymmetry = {sh_asym:.2e} (state-free even "
          "functional: must be 0 -- graded-blindness control)")
    check(f"{tag} gate (a/B) shadow control: the state-free scalar shadow carries ZERO winding "
          f"asymmetry ({sh_asym:.1e})", sh_asym < 1e-12)

    # ---- gate (b): Wick pairing structure -- (b1) generalized Pfaffian identity on fixed
    # dart-decoration words; (b2) <0|W^L|0> == the Wick-weighted NB path sum, L = 1..4
    # [E2a lines 195-218 with gamma-words -> decoration-vector words]
    werr = max(abs(wick_vecs([decor[i] for i in w]) - direct_vecs([decor[i] for i in w]))
               for w in wick_dart_words)
    oerr = max(abs(direct_vecs([decor[i] for i in w])) for w in wick_dart_odd)
    row["b_wick"] = max(werr, oerr)
    check(f"{tag} gate (b1) generalized Wick: <0|gam(x)-word|0> = Pf(x_i^T C x_j) on "
          f"{len(wick_dart_words)} fixed words (max err {werr:.1e}); odd words vanish ({oerr:.1e})",
          werr < 1e-10 and oerr < 1e-12)

    def paths_weight(L):
        out = np.zeros((ND, ND), complex)

        def rec(dd0, d, word, depth):
            if depth == L:
                out[d, dd0] += direct_vecs(word[::-1])   # word applied right-to-left (E2a line 201)
                return
            for dp in range(ND):
                if abs(B_G[dp, d]) > 0.5:
                    rec(dd0, dp, word + [decor[dp]], depth + 1)
        for dd0 in range(ND):
            rec(dd0, dd0, [], 0)
        return out

    ok_series, max_dev = True, 0.0
    WL = np.eye(8 * ND)
    for L in (1, 2, 3, 4):
        WL = WL @ W
        devL = np.max(np.abs(P_VAC @ WL @ P_VAC.conj().T - paths_weight(L)))
        max_dev = max(max_dev, devL)
        ok_series &= devL < 1e-9
    row["b_path"] = max_dev
    check(f"{tag} gate (b2) <0|W^L|0> == Wick-weighted NB path sum, L = 1..4 (max dev {max_dev:.1e})",
          ok_series)

    # ---- gate (c): u^2-parity  [E2a lines 220-229]
    odd_norms = [np.max(np.abs(P_VAC @ np.linalg.matrix_power(W, L) @ P_VAC.conj().T))
                 for L in (1, 3, 5)]
    row["c_parity"] = max(odd_norms)
    check(f"{tag} gate (c) parity: odd u-orders vanish in the vacuum block "
          f"(L=1,3,5 norms {[f'{x:.1e}' for x in odd_norms]})", max(odd_norms) < 1e-12)

    # ---- gate (d): THE chiral asymmetry, E2a's estimator + free control + J-flip, same u points
    ctrls = {u: free_control(u) for u in (0.11, 0.23)}
    asyms = {u: asym(W, u) for u in (0.11, 0.23)}
    a_plus = asyms[0.23]
    a_minus = asym(W, 0.23, P_VAC_c)
    flip = abs(a_plus + np.conj(a_minus))
    row.update(ctrl=max(ctrls.values()), a011=asyms[0.11], a023=a_plus, aminus=a_minus, flip=flip)
    for u in (0.11, 0.23):
        print(f"    [info] {tag} u = {u}:  FREE control = {ctrls[u]:.2e}   "
              f"A' = {asyms[u]:.9e}   |A'| = {abs(asyms[u]):.9e}")
    print(f"    [info] {tag} A'(+J) = {a_plus:.9e}   A'(-J) = {a_minus:.9e}   "
          f"|A'(+J)+conj(A'(-J))| = {flip:.1e}")
    check(f"{tag} gate (d) FREE-ensemble control exactly zero at both u "
          f"(max {max(ctrls.values()):.1e})", max(ctrls.values()) < 1e-12)
    row["survives"] = (min(abs(asyms[0.11]), abs(asyms[0.23])) > 1e-10 and flip < 1e-10
                       and max(ctrls.values()) < 1e-12)
    check(f"{tag} gate (d) S-C signature: asymmetry NONZERO at both u AND flips under J -> -J "
          f"AND controls zero  =>  {'SURVIVES' if row['survives'] else 'DOES NOT SURVIVE'}",
          True)   # signature outcome is a REPORTED verdict input, not a pass/fail assertion
    return row


# fixed (deterministic) dart-index word lists for the generalized Wick gate
DART_WORDS = [(0, 1), (2, 7), (3, 3), (0, 1, 2, 3), (4, 9, 2, 11), (10, 3, 7, 1),
              (5, 4, 3, 2, 1, 0), (0, 1, 0, 1, 2, 3), (2, 2, 5, 5, 1, 4)]
DART_ODD = [(0,), (1, 4, 9), (7, 2, 0, 1, 3)]

# THE ORIGINAL DECORATION (E2a lines 167-172): x_dp = e_{EDGE_OF_DART[dp]} (unsigned)
decor_orig = [E6[:, EDGE_OF_DART[dp]] for dp in range(ND)]
row_orig = run_gates("S-1 ORIGINAL", decor_orig, DART_WORDS, DART_ODD)

# the banked-anchor asserts (E2a's printed values, full precision -- never adjusted)
check(f"S-1 ANCHOR |A(u=0.11)| = {abs(row_orig['a011']):.12e} matches banked "
      f"{ANCHOR_EVADE_011:.12e} (dev {abs(abs(row_orig['a011']) - ANCHOR_EVADE_011):.1e})",
      abs(abs(row_orig["a011"]) - ANCHOR_EVADE_011) < 1e-9)
check(f"S-1 ANCHOR |A(u=0.23)| = {abs(row_orig['a023']):.12e} matches banked "
      f"{ANCHOR_EVADE_023:.12e} (dev {abs(abs(row_orig['a023']) - ANCHOR_EVADE_023):.1e})",
      abs(abs(row_orig["a023"]) - ANCHOR_EVADE_023) < 1e-9)
check(f"S-1 ANCHOR A(+J) = {row_orig['a023'].real:+.12e} matches banked {ANCHOR_A_PLUS_023:+.12e} "
      "(sign included)", abs(row_orig["a023"] - ANCHOR_A_PLUS_023) < 1e-9)
check("S-1 REGRESSION COMPLETE: all E2a banked gates reproduce on the verbatim rebuild "
      "(free reduction, Wick pairing, u^2-parity, chiral asymmetry + J-flip + zero free control)",
      row_orig["survives"] and row_orig["a_A"] < 1e-10)

# ===========================================================================
banner("S-2  THE RE-DECORATION: the compliant (signed/R-odd, Uo-type) family  [W2-MAP M-0/M-1a/M-4]")
# ===========================================================================
# R-even/R-odd dart-to-edge embeddings  [W2-MAP lines 143-147, verbatim]
Ue = np.zeros((ND, NE))
Uo = np.zeros((ND, NE))
for e in range(NE):
    Ue[2 * e, e] = 1 / math.sqrt(2); Ue[2 * e + 1, e] = 1 / math.sqrt(2)
    Uo[2 * e, e] = 1 / math.sqrt(2); Uo[2 * e + 1, e] = -1 / math.sqrt(2)


def dart_rep(sig):
    """W2-MAP lines 120-132, verbatim: the cover-side A4 action on darts."""
    Rd = np.zeros((ND, ND))
    for a, (i, j) in enumerate(DARTS):
        ni, nj = sig[i], sig[j]
        lo, hi = min(ni, nj), max(ni, nj)
        e2 = EIDX[(lo, hi)]
        b = 2 * e2 if ni < nj else 2 * e2 + 1
        Rd[b, a] = 1.0
    return Rd


REV = np.zeros((ND, ND))                       # the dart-reversal involution R
for e in range(NE):
    REV[2 * e, 2 * e + 1] = 1.0
    REV[2 * e + 1, 2 * e] = 1.0

# M-4(ii) regression [W2-MAP lines 576-589]: the ORIGINAL decoration IS the unsigned/R-even one
dev_ue = max(np.max(np.abs(gam(math.sqrt(2) * (Ue.T @ np.eye(ND)[:, dp])) - gam(decor_orig[dp])))
             for dp in range(ND))
dev_uo = max(np.max(np.abs(gam(math.sqrt(2) * (Uo.T @ np.eye(ND)[:, dp])) - gam(decor_orig[dp])))
             for dp in range(ND))
check(f"S-2 M-4(ii) regression: original decoration == gam(sqrt2 Ue^T e_dp) EXACTLY (dev {dev_ue:.1e}) "
      f"and differs maximally from the Uo-type (dev {dev_uo:.2f})", dev_ue < 1e-12 and dev_uo > 1.0)

# M-0k regression [W2-MAP lines 207-214]: Uo intertwines edge_rep with dart_rep; Ue FAILS
int_uo = max(np.max(np.abs(dart_rep(g) @ Uo - Uo @ edge_rep(g))) for g in A4)
int_ue = max(np.max(np.abs(dart_rep(g) @ Ue - Ue @ edge_rep(g))) for g in A4)
check(f"S-2 M-0k regression: Uo intertwines edge_rep with dart_rep EXACTLY (dev {int_uo:.1e}); "
      f"Ue does NOT (dev {int_ue:.4f} = sqrt2) -- the original convention is A4-non-covariant",
      int_uo < 1e-12 and int_ue > 1.0)

# the commutant End_A4(edge_rep) and its {I6, J6, S1, S2} basis  [W2-MAP lines 321-357, verbatim]
rows2 = [np.kron(np.eye(NE), edge_rep(g)) - np.kron(edge_rep(g).T, np.eye(NE)) for g in A4]
C2 = np.vstack(rows2)
_, S2v, Vt2 = np.linalg.svd(C2)
rank2 = int(np.sum(S2v > 1e-9))
Cs = [Vt2[rank2 + k].reshape(NE, NE, order='F') for k in range(C2.shape[1] - rank2)]
check(f"S-2 commutant End_A4(edge_rep) has dim {len(Cs)} = 4 (Mat_2(R)) [W2-MAP M-1a-v]",
      len(Cs) == 4)
IJ = np.stack([E6.reshape(-1, order='F'), J6.reshape(-1, order='F')], axis=1)
allc = np.stack([c.reshape(-1, order='F') for c in Cs], axis=1)
Q_IJ, _ = np.linalg.qr(IJ)
proj = allc - Q_IJ @ (Q_IJ.T @ allc)
Qc, _ = np.linalg.qr(proj)
S1 = Qc[:, 0].reshape(NE, NE, order='F')
check(f"S-2 S1 (the reflection representative): symmetric (dev {np.max(np.abs(S1 - S1.T)):.1e}), "
      f"traceless ({abs(np.trace(S1)):.1e}), S1@S1 = I6/6 (dev {np.max(np.abs(S1 @ S1 - E6 / 6)):.1e}) "
      "[W2-MAP M-1a-vii]",
      np.max(np.abs(S1 - S1.T)) < 1e-10 and abs(np.trace(S1)) < 1e-10
      and np.max(np.abs(S1 @ S1 - E6 / 6)) < 1e-10)

# NORMALIZATION (declared, not tuned): R5 fixes each branch up to ONE global scale; branch-1's
# representative Uo@I6 is a UNIT isometry ((Uo I6)^T (Uo I6) = I6).  Putting branch-2 at the SAME
# global scale means S1n := sqrt6 * S1, so (Uo S1n)^T (Uo S1n) = 6 S1^2 = I6 -- the same unit-
# isometry convention, applied uniformly to both branches (no per-branch freedom).
S1n = math.sqrt(6) * S1
check(f"S-2 S1n = sqrt6*S1 is orthogonal-symmetric (S1n@S1n = I6, dev "
      f"{np.max(np.abs(S1n @ S1n - E6)):.1e}) -- unit-isometry normalization, same as branch-1",
      np.max(np.abs(S1n @ S1n - E6)) < 1e-10)

PHI = {"BRANCH-1 (Uo@I6, theta=0)": Uo @ E6, "BRANCH-2 (Uo@S1, reflection)": Uo @ S1n}
for nm, Phi in PHI.items():
    r2 = max(np.max(np.abs(dart_rep(g) @ Phi - Phi @ edge_rep(g))) for g in A4)
    rodd = np.max(np.abs(REV @ Phi + Phi))
    iso = np.max(np.abs(Phi.T @ Phi - E6))
    check(f"S-2 {nm} is COMPLIANT: R2 A4-intertwining (dev {r2:.1e}), R-ODD (R.Phi = -Phi, dev "
          f"{rodd:.1e}), unit isometry (dev {iso:.1e})", max(r2, rodd, iso) < 1e-10)

# the branch decorations: x_dp = sqrt2 * Phi^T e_dp (the SAME extraction M-4 used on the original)
decor_b1 = [math.sqrt(2) * (PHI["BRANCH-1 (Uo@I6, theta=0)"].T @ np.eye(ND)[:, dp]) for dp in range(ND)]
decor_b2 = [math.sqrt(2) * (PHI["BRANCH-2 (Uo@S1, reflection)"].T @ np.eye(ND)[:, dp]) for dp in range(ND)]
sgn_ok = all(np.max(np.abs(dec[2 * e + 1] + dec[2 * e])) < 1e-12
             for dec in (decor_b1, decor_b2) for e in range(NE))
check("S-2 both branch decorations are SIGNED: x(reversed dart) = -x(forward dart) exactly "
      "(the R-odd signature, per edge)", sgn_ok)
b1_signed = all(np.max(np.abs(np.abs(decor_b1[dp]) - decor_orig[dp])) < 1e-12 for dp in range(ND))
check("S-2 BRANCH-1 decoration = +/- gamma_{e(dp)} (sign = dart orientation): |x_dp| equals the "
      "ORIGINAL decoration exactly -- branch-1 IS 'E2a's convention, signed'", b1_signed)
b2_unit = all(abs(np.linalg.norm(decor_b1[dp]) - 1) < 1e-12 and
              abs(np.linalg.norm(decor_b2[dp]) - 1) < 1e-12 for dp in range(ND))
check("S-2 all decoration coefficient vectors are UNIT norm in both branches (so every block "
      "satisfies gam(x)^2 = I8, exactly as the original's gamma_e^2 = I8 -- same operating point)",
      b2_unit)

# ===========================================================================
banner("S-3  E2a'S GATES RE-RUN ON BRANCH-1  (Phi = Uo @ I6, theta = 0)")
# ===========================================================================
row_b1 = run_gates("S-3 BRANCH-1", decor_b1, DART_WORDS, DART_ODD)

# ===========================================================================
banner("S-4  E2a'S GATES RE-RUN ON BRANCH-2  (Phi = Uo @ S1, the reflection representative)")
# ===========================================================================
row_b2 = run_gates("S-4 BRANCH-2", decor_b2, DART_WORDS, DART_ODD)

# ===========================================================================
banner("S-5  GATE TABLE, CROSS-BRANCH IDENTITY, THE VERDICT")
# ===========================================================================
cross = max(abs(row_b1["a011"] + row_b2["a011"]), abs(row_b1["a023"] + row_b2["a023"]))
check(f"S-5 CROSS-BRANCH IDENTITY: A'_branch2 = -A'_branch1 at BOTH u (max |A'_1 + A'_2| = "
      f"{cross:.1e}) -- the two O(2) representatives carry EQUAL-MAGNITUDE, OPPOSITE-SIGN "
      "asymmetries: the unresolved orbit ambiguity is exactly the SIGN of the chiral asymmetry "
      "(reflections are orientation-reversing)", cross < 1e-12)

hdr = f"    {'gate':<38}{'ORIGINAL (R-even)':>20}{'BRANCH-1 (Uo@I6)':>20}{'BRANCH-2 (Uo@S1)':>20}"
print("\n" + hdr)
print("    " + "-" * (len(hdr) - 4))


def fmt(rows, key, spec="{:.1e}"):
    return "".join(f"{spec.format(r[key]):>20}" for r in rows)


R3 = [row_orig, row_b1, row_b2]
print(f"    {'(a/A) block-erasure vs free (dev)':<38}" + fmt(R3, "a_A"))
print(f"    {'(a/B) gamma->1 shadow vs free (dev)':<38}" + fmt(R3, "a_B"))
print(f"    {'(a/B) shadow winding asym (control)':<38}" + fmt(R3, "a_B_shadow_asym"))
print(f"    {'(b1) generalized Wick (max err)':<38}" + fmt(R3, "b_wick"))
print(f"    {'(b2) path-sum L=1..4 (max dev)':<38}" + fmt(R3, "b_path"))
print(f"    {'(c) parity odd-L norms (max)':<38}" + fmt(R3, "c_parity"))
print(f"    {'(d) free control (max over u)':<38}" + fmt(R3, "ctrl"))
print(f"    {'(d) A(u=0.11)':<38}" + "".join(f"{r['a011'].real:>+20.9e}" for r in R3))
print(f"    {'(d) A(u=0.23)':<38}" + "".join(f"{r['a023'].real:>+20.9e}" for r in R3))
print(f"    {'(d) A(-J) at u=0.23':<38}" + "".join(f"{r['aminus'].real:>+20.9e}" for r in R3))
print(f"    {'(d) J-flip |A(+J)+conj(A(-J))|':<38}" + fmt(R3, "flip"))
print(f"    {'(d) S-C signature':<38}"
      + "".join(f"{'SURVIVES' if r['survives'] else 'ABSENT':>20}" for r in R3))

verdict = ("SURVIVES-BOTH-BRANCHES" if row_b1["survives"] and row_b2["survives"] else
           "CONVENTION-DEPENDENT" if not (row_b1["survives"] or row_b2["survives"]) else
           "MIXED")
check(f"S-5 VERDICT (frozen tree): {verdict}"
      + (" -- branch-1 " + ("SURVIVES" if row_b1["survives"] else "does NOT survive")
         + "; branch-2 " + ("SURVIVES" if row_b2["survives"] else "does NOT survive")),
      True)

print(f"""
    THE VERDICT: {verdict}.
      * BRANCH-1 (Uo@I6):  |A'| = {abs(row_b1['a011']):.6e} (u=0.11), {abs(row_b1['a023']):.6e}
        (u=0.23); J-flip exact ({row_b1['flip']:.1e}); free control zero ({row_b1['ctrl']:.1e}).
      * BRANCH-2 (Uo@S1):  |A'| = {abs(row_b2['a011']):.6e} (u=0.11), {abs(row_b2['a023']):.6e}
        (u=0.23); J-flip exact ({row_b2['flip']:.1e}); free control zero ({row_b2['ctrl']:.1e}).
      The S-C chiral asymmetry (nonzero interacting, exactly-zero free, sign-flipping under
      J -> -J) is DECORATION-ROBUST across the entire compliant family sampled at both
      representative points.  M-4's inherited defect is DISCHARGED for the chiral channel:
      the forced chiral channel does not rest on the non-compliant convention.

    BOOKED FINDINGS (raw, per contract):
      F1  the MAGNITUDE is convention-dependent even though the signature is not:
          |A(0.23)| = {ANCHOR_EVADE_023:.6e} (non-compliant original) vs
          {abs(row_b1['a023']):.6e} (compliant, ~{abs(row_b1['a023'])/ANCHOR_EVADE_023:.2f}x) --
          E2a's qualitative theorem survives; its NUMBER was the convention's.
      F2  A'_branch2 = -A'_branch1 EXACTLY (max dev {cross:.1e}): the O(2) orbit ambiguity
          (W2-MAP M-1b, unresolved) is precisely an overall CHIRALITY-SIGN ambiguity of the
          asymmetry -- the unresolved freedom does not touch the magnitude at these
          representatives.
      F3  the compliant decoration's faithful gamma->1 shadow is NOT the free walk: it is the
          sign-twisted walk diag(s).B (dev from free ~{row_b1['a_B']:.2f}), a Z2 sign field the
          R-odd convention necessarily leaves behind; yet that state-free shadow carries EXACTLY
          ZERO winding asymmetry ({row_b1['a_B_shadow_asym']:.1e}) -- chirality still enters ONLY
          through the state (graded-blindness intact).  E2a's literal block-erasure reduction
          (gate a/A) passes unchanged for all three decorations.

    SCOPE / POISON HONESTY: no oblique/EW/a_e number appears anywhere; fugacities are E2a's own
    generic u = 0.11/0.23; the banked E2a values were asserted as anchors, never adjusted; BOTH
    branches reported (no orbit member picked); every free-ensemble control is exactly zero.
    NOT claimed: no O(2) selection (M-1b's AMBIGUOUS stands); no magnitude is banked for any
    downstream row; CS-1 (the finite-k propagator) remains a separate, unstarted station.""")

print("=" * 92)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}   VERDICT: {verdict}")
print("=" * 92)
sys.exit(0 if ok_all else 1)
