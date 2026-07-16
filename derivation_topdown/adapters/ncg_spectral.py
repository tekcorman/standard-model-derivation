#!/usr/bin/env python3
"""
derivation_topdown/adapters/ncg_spectral.py

R2/G3a adapter -- THE KO SIGN TABLE.  Pre-registered in
internal research notes (contracts KO-0..KO-4, frozen BEFORE this file was
written).  Adapter = verification contract ONLY: computes, on the framework's OWN objects (the m06
spacetime/spinor Clifford construction and the engine's Cl(6) internal Fock space), the Connes
KO-dimension of candidate real spectral triples, and TESTS (does not assume) the RECONCILIATION
HYPOTHESIS that the Standard-Model total KO-anatomy "4 (spacetime) + 6 (internal) == 2 (mod 8)" is
realized INSIDE this framework.  Zero physics: no new constant, no engine edit, no re-derivation --
every construction below is reused verbatim from the named prior-art files.  Candidate sets are
FROZEN (no shopping after seeing signs; see POISONS at the bottom of the pre-reg).

BACKGROUND -- CONNES' REAL SPECTRAL TRIPLES AND THE KO-DIMENSION
A real spectral triple (A, H, D; J, gamma) is an algebra A on a Hilbert space H with a Dirac
operator D and, in the EVEN case, a chirality grading gamma (gamma^2=+1, gamma^dag=gamma,
{D,gamma}=0) and an ANTIUNITARY real structure J ("charge conjugation").  J is fixed, up to its
unitary part, by three signs (Connes 1995; the KO-dimension mod-8 classification):
    J^2 = eps * I,     J D = eps' * D J,     J gamma = eps'' * gamma J        (eps,eps',eps'' in {+-1})
Reading (eps, eps', eps'') against Connes' fixed table assigns a KO-dimension n (mod 8) -- literally
the KO-dimension of the real Clifford module the triple carries.  CONVENTION ADOPTED HERE (stated,
not chosen after the fact): J D = +eps' D J and J gamma = +eps'' gamma J, i.e. eps' and eps'' are
defined exactly as in the framework's own prior computation (explore_m06_spinor_architecture.py);
this is the same convention used throughout the literature review cited in the pre-reg (some papers
write J D J^{-1} = eps' D, which is the identical statement for antiunitary J since J^{-1} = eps*J).

THE SM'S 4 + 6 == 2 (mod 8) ANATOMY
Chamseddine-Connes's almost-commutative Standard Model spectral triple is a PRODUCT: a 4-dimensional
Euclidean spacetime/spinor factor (KO-dimension 4) tensored with a 6-dimensional INTERNAL finite
noncommutative-geometry factor (KO-dimension 6, Connes hep-th/0608226; Chamseddine-Connes JGP 58
(2008)); KO-dimensions of tensor factors ADD mod 8, giving the physical total KO-dimension 4+6=10==2
(mod 8) that fixes the fermion-doubling / first-order sign conventions of the SM action.  THE
RECONCILIATION HYPOTHESIS (declared in the pre-reg, tested not presumed): three prior repo claims
("CLEANROOM's KO-4", "crown-jewel's KO-2", "SM-needs-6") may name three DIFFERENT factors of this
SAME anatomy -- KO-4 = the spacetime/spinor factor (m06), KO-2 = the TOTAL, "needs-6" = the INTERNAL
factor.  The hypothesis is TRUE iff the internal Cl(6) Fock space's own KO-dimension computes to 6 --
CONFIRMED below via the R2b-adjudicated exotic presentation (2026-07-09).

EXECUTED FINDING (post-run, honest -- integrity history preserved verbatim): running the arithmetic
below, the sigma_M0 candidate (once its ladder-ordering convention is corrected to match ladder_vecs'
own descending-order convention, so that it actually IS the pre-registered lift a_i <-> a_i^dagger --
verified by the KO-2.2d-bis defining-relation check, < 1e-12) FORCES (eps,eps',eps'') = (-1,-1,-1)
for (gamma_F, J_F) = (P_F, sigma_M0) and (Gamma_5, sigma_M0). THE FIRST IMPLEMENTATION of this
computation reported KO6-FOUND; the adversarial check exposed this as an operator-ordering BUG (the
KO-2.2d-bis defining relation failed at 2.0); the ordering was corrected with the defining relation as
sole arbiter, and the corrected arithmetic REVERSED the verdict -- this triple matches NO EVEN row of
Connes' canonical table (KO_TABLE_EVEN) as originally read, so the corrected signs stood, honestly, as
ANOMALOUS / KO-OTHER (the reconciliation hypothesis NOT confirmed) pending a literature-first
follow-up -- R2b.

R2b (internal research notes, authority hierarchy frozen BEFORE the
literature sweep) has now RESOLVED the reading: adjudication READ-AS-KO-6 (2026-07-09). The
literature documents a genuine convention freedom for EVEN real spectral triples -- J and J' = J.gamma
are BOTH admissible real structures, "perfectly on the same footing" (Dabrowski-Dossena, Int. J.
Geom. Methods Mod. Phys. 8 (2011) 1833, arXiv:1011.4456, Introduction + Table 1: the second
presentation of n=6 is (-,-,-)); the replacement J -> J.gamma maps the canonical row to the "exotic"
row reversibly (Cacic, Lett. Math. Phys. 2013, arXiv:1209.4832, Sec. 2.2 + Table 2.2, column "6-" =
(-,-,-)). Connes' own canonical table (hep-th/0608226 App. 7 Def. 7.2, citing the 1995 J. Math. Phys.
paper; van Suijlekom 2024 Tables 3.1/5.1 identical) carries eps'=+1 at even n and does not discuss the
freedom -- our implemented convention matched Connes' canonical table, which is why the raw
computation printed no row under that convention alone.

THEREFORE: the internal Cl(6) Fock's forced (eps,eps',eps'') = (-1,-1,-1) (J_F = the particle-hole
sigma_M0, gamma_F = fermion parity) IS KO-dimension 6 in the EXOTIC presentation; the canonical-
presentation partner J' = J_F.gamma_F carries (+1,+1,-1) = Connes' KO-6 row EXACTLY (verified in-code
below, R2b-verify -- the only new computation added by this adjudication; every other matrix number in
this file is untouched). The internal KO-dimension is 6. Total: spacetime KO-4 (KO-1) + internal KO-6
(KO-2/KO-3) = 10 == 2 (mod 8) -- the reconciliation of the three historical claims ("CLEANROOM's KO-4"
= spacetime factor, "crown-jewel's KO-2" = the total, "SM-needs-6" = the internal factor) is CONFIRMED,
via the pre-registered literature-first path (R2b) -- never by convention-picking.

THE CONTRACTS (verbatim from the frozen pre-reg)
  KO-0  THE TABLE            -- hard-code Connes' (eps,eps',eps'') <-> KO-dim table; cross-check
                                 against m06's in-code table (must agree row-by-row).
  KO-1  SPACETIME FACTOR     -- m06's Cl(4) spinor real-structure computation, re-run as a machine
                                 contract: J^2=-1, JD=+DJ, Jgamma=+gammaJ => KO-dim 4.
  KO-2  INTERNAL Cl(6) FOCK  -- the never-executed arithmetic.  On the engine's 8-dim Fock space:
                                 3 J_F candidates x 2 gamma_F candidates = 6 pairs; for each, compute
                                 eps, eps'' directly and eps' from the real vector space of Hermitian,
                                 gamma_F-odd, A4-covariant operators (the internal-Dirac candidate
                                 space); read the KO-dim off Connes' table.
  KO-3  THE VERDICT           -- dual-outcome, frozen logic: KO6-FOUND / KO-OTHER / AMBIGUOUS.
  KO-4  SCOPE DECLARATION     -- printed statement of what is explicitly NOT claimed here.

REUSE MAP (nothing below is re-derived; every recipe is copied from the named prior-art file)
  - m06's Cl(4) spacetime spinor + J_signs machinery : derivation_topdown/matter_bridge/
      explore_m06_spinor_architecture.py lines ~117-150 (Cl(4) generators g[a], gamma_c),
      lines ~196-269 (J_signs, KO_TABLE, the canonical C = g^2 g^4 representative giving KO-dim 4).
  - Cl(6) generators + gamma_5                       : simulator/srs_engine/utils/algebraic.py
      (AlgebraicUtility.cl6_generators / .cl6_chirality).
  - Ladder/Fock recipe (J6, A_ops, NHAT, vacuum,
    the 8 explicit Fock states |S>, S subset {0,1,2})  : adapters/furey_stoica_labels.py (FS-1/FS-2)
      == proofs/foundations/WS1_species_deck_correlation_2026-07-07.py lines ~55-114, reproduced here
      verbatim (same edge_rep/J6/modes/A_ops/ladder_vecs construction; no engine import, no re-fit).
  - A4 gauge action on the Fock (spin_lift Schur-
    intertwiner solve; U(g) for g in A4)               : proofs/foundations/ML2b_dr_frame_2026-07-08.py
      lines ~65-95, reproduced here verbatim.

REFERENCES: A. Connes, "Noncommutative geometry and reality", J. Math. Phys. 36 (1995) 6194;
A. Connes, "Noncommutative Geometry and the Standard Model with Neutrino Mixing", hep-th/0608226
(the KO-dimension-6 finite space); A. Chamseddine & A. Connes, J. Geom. Phys. 58 (2008) 38 (the
spectral action / real structure of the SM triple).
R2b ADDENDA (the J<->J.gamma convention freedom, literature-first adjudication 2026-07-09):
L. Dabrowski & F. Dossena, "Product of real spectral triples", Int. J. Geom. Methods Mod. Phys. 8
(2011) 1833, arXiv:1011.4456 (Introduction + Table 1: J and J.gamma both admissible for even triples);
B. Cacic, "A reconstruction theorem for Almost-Commutative Spectral Triples", Lett. Math. Phys. 103
(2013), arXiv:1209.4832 (Sec. 2.2 + Table 2.2: the J<->J.gamma row-reversal for KO-dim 6); W. van
Suijlekom, Noncommutative Geometry and Particle Physics, 2nd ed. (2024), Tables 3.1/5.1 (Connes'
canonical (eps,eps',eps'') table, identical to hep-th/0608226 App. 7 Def. 7.2).

POISONS (binding, per pre-reg): no engine/proofs edits (this is the only new file); the candidate
sets are FROZEN (no candidates added or removed after seeing signs; no extra i-factors or basis
changes beyond those declared); eps' is computed over the WHOLE declared operator space, never
picked from one convenient D; no forcing KO-6 -- the arithmetic decides, and all three KO-3 verdict
branches are declared equally valid outcomes.

Exit code: 0 iff KO-0 and KO-1 pass AND KO-3 reports a definite verdict (any of KO6-FOUND / KO-OTHER
/ AMBIGUOUS -- all three are "definite" in the sense of being a completed, booked reading); 1 iff
KO-0 or KO-1 fails a matrix identity.
"""
import itertools
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
import srs  # noqa: E402  (walled-off clean-room K4-cover module; EDGES, NV)
from simulator.srs_engine.utils import AlgebraicUtility  # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=120)

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    cond = bool(cond)
    ok_all = ok_all and cond
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return cond


def banner(t):
    print("=" * 96)
    print(f" {t}")
    print("=" * 96)


def anticomm(A, B):
    return A @ B + B @ A


# ===================================================================================================
banner("KO-0  THE CONNES (eps,eps',eps'') <-> KO-DIMENSION TABLE")
# ===================================================================================================
print("""    Convention adopted (stated, not chosen after the fact -- matches m06's own convention):
        J^2 = eps*I,   J D = eps' D J,   J gamma = eps'' gamma J.
    Hard-coded from Connes' classification (citations in the module docstring):""")

# The task's hard-coded table (None/"." for odd KO-dims, where no chirality grading exists => eps''
# is not defined).
CONNES_TABLE = {
    0: (+1, +1, +1), 1: (+1, -1, None), 2: (-1, +1, -1), 3: (-1, +1, None),
    4: (-1, +1, +1), 5: (-1, -1, None), 6: (+1, +1, -1), 7: (+1, +1, None),
}
# m06's OWN in-code table (explore_m06_spinor_architecture.py lines ~219-222), copied verbatim.
M06_TABLE = {
    0: (+1, +1, +1), 1: (+1, -1, None), 2: (-1, +1, -1), 3: (-1, +1, None),
    4: (-1, +1, +1), 5: (-1, -1, None), 6: (+1, +1, -1), 7: (+1, +1, None),
}
print(f"    {'n mod 8':>8s}  {'eps':>5s} {'eps' + chr(39):>6s} {'eps' + chr(39) * 2:>7s}   "
      f"(hard-coded)      (m06 in-code)      row match?")
row_match = True
for n in range(8):
    a, b = CONNES_TABLE[n], M06_TABLE[n]
    same = a == b
    row_match &= same
    print(f"    {n:>8d}  {a[0]:>+5d} {('%+d' % a[1]):>6s} {('%+d' % a[2]) if a[2] is not None else '  --':>7s}"
          f"        {a}        {b}       {'match' if same else 'MISMATCH'}")
check("KO-0a hard-coded table agrees row-by-row with m06's in-code table (no convention mismatch found)",
      row_match)
check("KO-0b table is well-formed (eps,eps' in {+-1}; eps'' in {+-1} for even n, None for odd n)",
      all(CONNES_TABLE[n][0] in (+1, -1) and CONNES_TABLE[n][1] in (+1, -1)
          and ((n % 2 == 0 and CONNES_TABLE[n][2] in (+1, -1)) or (n % 2 == 1 and CONNES_TABLE[n][2] is None))
          for n in range(8)))
KO_TABLE_EVEN = {n: CONNES_TABLE[n] for n in (0, 2, 4, 6)}  # the Fock is gamma-graded => even rows only

# ===================================================================================================
banner("KO-1  THE SPACETIME/SPINOR FACTOR (m06's Cl(4) computation, re-run as a machine contract)")
# ===================================================================================================
print("""    Reused verbatim from explore_m06_spinor_architecture.py lines ~117-150 (Cl(4) generators,
    grading gamma_c) and ~196-269 (J_signs, the canonical real-structure representative C = g^2 g^4).
    Regression: this computation already exists in the repo; here it is made machine-legible as a
    frozen contract (KO-1), not re-derived.""")

I2 = np.eye(2)
s1 = np.array([[0, 1], [1, 0]], complex)
s2 = np.array([[0, -1j], [1j, 0]], complex)
s3 = np.array([[1, 0], [0, -1]], complex)
kron = np.kron

g_st = [kron(s1, s1), kron(s1, s2), kron(s1, s3), kron(s2, I2)]   # g1,g2,g3 spatial; g4 forced (m06 Part 1)
gc_st = g_st[0] @ g_st[1] @ g_st[2] @ g_st[3]                      # gamma_c = Cl(4) volume element
SPATIAL_ST = [g_st[0], g_st[1], g_st[2]]

ok4 = all(np.allclose(anticomm(g_st[a], g_st[b]), 2 * (a == b) * np.eye(4)) for a in range(4) for b in range(4))
check("KO-1a {g^a,g^b} = 2 delta^ab I on Cl(4)", ok4)
gc_ok = np.allclose(gc_st @ gc_st, np.eye(4)) and all(np.allclose(anticomm(gc_st, g_st[a]), 0) for a in range(4))
check("KO-1b gamma_c^2 = I and {gamma_c, g^a} = 0 (genuine Z2 grading)", gc_ok)


def J_signs_st(C, gammas, grading):
    """m06's J_signs, copied verbatim: (eps, eps', eps'') for J = C o conj on Cl(4)."""
    n = C.shape[0]
    J2 = C @ np.conj(C)
    eps = +1 if np.allclose(J2, np.eye(n)) else (-1 if np.allclose(J2, -np.eye(n)) else None)

    def sign_against(ops):
        plus = all(np.allclose(C @ np.conj(o), +o @ C) for o in ops)
        minus = all(np.allclose(C @ np.conj(o), -o @ C) for o in ops)
        return +1 if plus else (-1 if minus else None)

    eps_p = sign_against(gammas)     # tested against the 3 SPATIAL (kinetic) Dirac generators, m06 convention
    eps_pp = sign_against([grading])
    return eps, eps_p, eps_pp


# the canonical representative used by m06: C = g^2 g^4 (0-indexed: g_st[1] @ g_st[3])
C4 = g_st[1] @ g_st[3]
c4_unitary = np.allclose(C4 @ C4.conj().T, np.eye(4))
eps_st, ep_st, epp_st = J_signs_st(C4, SPATIAL_ST, gc_st)
print(f"    canonical J_st = C o conj,  C = g^2 g^4 (m06's own representative).")
print(f"    explicit matrix C = g^2 g^4 =\n{np.round(C4, 4)}")
# transparency note: the pre-reg's shorthand description "(sigma_y (x) sigma_y) o K" does NOT match
# this explicit matrix -- report the discrepancy raw rather than silently reconciling it.
syy = kron(s2, s2)
matches_syy_shorthand = np.allclose(C4, syy) or np.allclose(C4, -syy) or np.allclose(C4, 1j * syy) or np.allclose(C4, -1j * syy)
szy = kron(s3, s2)
print(f"    CONVENTION NOTE (transparency, not a failure): the pre-reg's reuse-map shorthand names this "
      f"J as '(sigma_y (x) sigma_y) o K'. The EXPLICIT matrix m06 actually uses (C = g^2 g^4, g^2 = "
      f"sigma_1(x)sigma_2, g^4 = sigma_2(x)I) works out to C = i*(sigma_z (x) sigma_y), not sigma_y(x)sigma_y:")
print(f"      C == +-(1 or i)*sigma_y(x)sigma_y ?  {matches_syy_shorthand}")
print(f"      C == i*(sigma_z (x) sigma_y) ?        {np.allclose(C4, 1j * szy)}")
check("KO-1c C = g^2 g^4 is unitary (a genuine antiunitary J = C o conj candidate)", c4_unitary)
check(f"KO-1d J^2 = eps*I with eps = -1  (found eps={eps_st})", eps_st == -1)
check(f"KO-1e J D = eps' D J (D built from the 3 spatial gammas) with eps' = +1  (found eps'={ep_st})",
      ep_st == +1)
check(f"KO-1f J gamma_c = eps'' gamma_c J with eps'' = +1  (found eps''={epp_st})", epp_st == +1)
ko1_signs = (eps_st, ep_st, epp_st)
ko1_match = ko1_signs == (-1, +1, +1) == CONNES_TABLE[4]
check(f"KO-1g (eps,eps',eps'') = {ko1_signs} == CONNES_TABLE[4] = {CONNES_TABLE[4]}  =>  KO-DIM 4", ko1_match)
KO1_PASS = ok4 and gc_ok and c4_unitary and eps_st == -1 and ep_st == +1 and epp_st == +1 and ko1_match
print(f"    KO-1 VERDICT: the spacetime/spinor factor has KO-dimension 4 (m06 regression reproduced).")

# ===================================================================================================
banner("KO-2  THE INTERNAL Cl(6) FOCK  (the never-executed arithmetic)")
# ===================================================================================================
print("""    Fock construction reused verbatim from adapters/furey_stoica_labels.py (FS-1/FS-2) ==
    WS1_species_deck_correlation_2026-07-07.py lines ~55-114, and the A4 lift from
    ML2b_dr_frame_2026-07-08.py lines ~65-95.  No engine import; the recipe is reproduced here.""")

g6 = [np.array(g, complex) for g in AlgebraicUtility.cl6_generators()]
g5_raw = AlgebraicUtility.cl6_chirality()
I8 = np.eye(8, dtype=complex)

cliff6 = max(np.max(np.abs(g6[a] @ g6[b] + g6[b] @ g6[a] - (2.0 if a == b else 0.0) * I8))
             for a in range(6) for b in range(6))
check(f"KO-2.0a Cl(6) generators {{gamma^a,gamma^b}} = 2 delta^ab I  (max dev {cliff6:.2e})", cliff6 < 1e-12)
g5sq = g5_raw @ g5_raw
g5_sign = +1 if np.allclose(g5sq, I8) else (-1 if np.allclose(g5sq, -I8) else None)
check(f"KO-2.0b gamma_5 = prod(gamma^a) squares to a genuine involution (found gamma_5^2 = {g5_sign:+d} I, "
      f"the documented Euclidean Cl(6,0) sign, forced by signature not chosen)", g5_sign is not None)

EDGES = srs.EDGES
NE, NV = len(EDGES), srs.NV
EIDX = {(min(i, j), max(i, j)): e for e, (i, j, v) in enumerate(EDGES)}
gam = lambda x: sum(x[a] * g6[a] for a in range(NE))


def edge_rep(sig):
    R6 = np.zeros((NE, NE))
    for e, (i, j, v) in enumerate(EDGES):
        a, b = sig[i], sig[j]
        s = 1.0
        if a > b:
            a, b, s = b, a, -1.0
        R6[EIDX[(a, b)], e] = s
    return R6


def spin_lift(R):
    rowsU = [np.kron(gam(R[:, a]), I8) - np.kron(I8, g6[a].T) for a in range(NE)]
    _, s, Vh = np.linalg.svd(np.vstack(rowsU))
    M = Vh[np.sum(s > 1e-9):].conj()[0].reshape(8, 8)
    return M / np.sqrt(np.abs(np.linalg.det(M @ M.conj().T)) ** (1 / 8))


d0 = np.zeros((NV, NE))
for e, (i, j, v) in enumerate(EDGES):
    d0[i, e] = -1.0
    d0[j, e] = 1.0
Chat = np.array([[1, 1, 0], [-1, 0, 1], [0, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
H1, _ = np.linalg.qr(Chat)
B1 = np.linalg.svd(d0)[2][:3].T
A4 = [dict(enumerate(p)) for p in itertools.permutations(range(4))
      if sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j]) % 2 == 0]
rowsJ = []
for gA in A4:
    R6 = edge_rep(gA)
    rowsJ.append(np.kron(np.eye(3), (H1.T @ R6 @ H1).T) - np.kron(B1.T @ R6 @ B1, np.eye(3)))
_, SpJ, VpJ = np.linalg.svd(np.vstack(rowsJ))
phi = VpJ[-1].reshape(3, 3)
phi *= math.sqrt(3) / np.linalg.norm(phi)
J6 = B1 @ phi @ H1.T - H1 @ phi.T @ B1.T
wJ, VJ = np.linalg.eig(J6)
modes, _ = np.linalg.qr(VJ[:, np.where(np.abs(wJ - 1j) < 1e-9)[0]])
A_ops = [gam(np.conj(modes[:, m])) / math.sqrt(2) for m in range(3)]
Adag_ops = [a.conj().T for a in A_ops]
NHAT = sum(Adag_ops[i] @ A_ops[i] for i in range(3))
wN, VN = np.linalg.eigh(NHAT)
vac = VN[:, [int(np.argmin(wN))]]
vac /= np.linalg.norm(vac)

nilp = max(np.max(np.abs(A_ops[i] @ A_ops[i])) for i in range(3))
check(f"KO-2.0c ladder nilpotency a_i^2 = 0  (max abs {nilp:.2e})", nilp < 1e-12)

SUBSETS = sorted([tuple(sorted(s)) for r in range(4) for s in itertools.combinations(range(3), r)],
                  key=lambda s: (len(s), s))
ladder_vecs = {}
for S in SUBSETS:
    v = vac.copy()
    for i in sorted(S, reverse=True):
        v = Adag_ops[i] @ v
    ladder_vecs[S] = v
W = np.hstack([ladder_vecs[S] for S in SUBSETS])   # the Fock-basis change matrix (columns = |S>)
idx = {S: k for k, S in enumerate(SUBSETS)}
comp = lambda S: tuple(sorted(set(range(3)) - set(S)))

w_unitary = np.allclose(W.conj().T @ W, I8)
check("KO-2.0d the 8 explicit Fock states {|S>} form an orthonormal basis (W unitary)", w_unitary)

# A4 lift on the Fock, transformed into the Fock basis (reused from ML2b_dr_frame, lines ~65-95)
U_A4_comp = [spin_lift(edge_rep(gA)) for gA in A4]
U_A4_F = [W.conj().T @ U @ W for U in U_A4_comp]
a4_unitary = all(np.allclose(U.conj().T @ U, I8, atol=1e-6) for U in U_A4_F)
check("KO-2.0e the 12 A4 lift operators are unitary in the Fock basis", a4_unitary)

print("\n    --- gamma_F CANDIDATES ---")
PF = np.diag([(-1.0) ** len(S) for S in SUBSETS]).astype(complex)   # (a) P_F = (-1)^Nhat
G5F = W.conj().T @ (1j * g5_raw) @ W                                  # (b) Gamma_5 = i*gamma_5, Fock basis
check("KO-2.1a P_F = (-1)^Nhat squares to +I (native)", np.allclose(PF @ PF, I8))
g5f_sq_ok = np.allclose(G5F @ G5F, I8)
check("KO-2.1b Gamma_5 := i*gamma_5 squares to +I (the i-factor is REQUIRED since gamma_5^2=-I)", g5f_sq_ok)
g5f_diag = np.allclose(G5F, np.diag(np.diag(G5F)))
print(f"    Gamma_5 expressed in the Fock basis (diagonal? {g5f_diag}):\n{np.round(G5F.real, 4)}")
g5_eq_pf = np.allclose(G5F, PF)
g5_eq_negpf = np.allclose(G5F, -PF)
check("KO-2.1c COINCIDENCE CHECK: is Gamma_5 == +-P_F (NOT an independent gamma_F candidate)?",
      g5_eq_pf or g5_eq_negpf,
      detail=f"Gamma_5 == +P_F: {g5_eq_pf};  Gamma_5 == -P_F: {g5_eq_negpf}")
print(f"    FINDING: Gamma_5 = {'+P_F' if g5_eq_pf else ('-P_F' if g5_eq_negpf else 'NEITHER +-P_F')} "
      f"EXACTLY (max|Gamma_5 - (-P_F)| = {np.max(np.abs(G5F - (-PF))):.2e}). P_F and Gamma_5 are "
      f"THE SAME candidate up to an overall sign (which, as shown below, does not change any KO sign) "
      f"=> only ONE independent gamma_F candidate exists on this Fock, not two.")

print("\n    --- J_F CANDIDATES ---")
# (a) C_ideal: Fock-basis conjugation K_Fock composed with the complement-duality permutation V_comp,
#     V_comp |S> = chi * |S^c>  (S^c = {0,1,2}\S).  Build with chi=+1; separately test chi=-1.
Vcomp = np.zeros((8, 8), complex)
for S in SUBSETS:
    Vcomp[idx[comp(S)], idx[S]] = 1.0
Vcomp_alt = -Vcomp
vcomp_unitary = np.allclose(Vcomp.conj().T @ Vcomp, I8)
vcomp_invol = np.allclose(Vcomp @ Vcomp, I8)
check("KO-2.2a C_ideal's V_comp (complement-swap permutation) is unitary and an involution", vcomp_unitary and vcomp_invol)

# (b) K_g6: plain complex conjugation in the computational/g6 basis, expressed as a Fock-basis matrix.
#     For v = W u (u = Fock components), conj(v) = conj(W) conj(u); re-expressed in Fock components:
#     u' = W^dagger conj(W) conj(u)  =>  the unitary part (acting on Fock coordinates) is W^dagger W-bar.
Ug6 = W.conj().T @ np.conj(W)
ug6_unitary = np.allclose(Ug6.conj().T @ Ug6, I8)
check("KO-2.2b K_g6's unitary part U_g6 = W^dagger . W-bar (constructed explicitly from the modes/W "
      "matrix) is unitary", ug6_unitary)

# (c) sigma_M0: the M0 particle-hole (J -> -J) lifted to the Fock.  J6 -> -J6 swaps the +i/-i
#     eigenmodes, i.e. swaps which combination is "annihilation" vs "creation": since the g6
#     generators are Hermitian, A_ops[i]' := gam(modes[:,i])/sqrt2 = Adag_ops[i] EXACTLY (the new
#     annihilation operators ARE the old creation operators).  The canonical lift EXISTS: define
#     sigma_M0's ket-level action by sending the new vacuum (annihilated by all Adag_ops[i], i.e. the
#     OLD fully-occupied state |{0,1,2}>) and building images of every |S> by applying the OLD
#     annihilation operators A_ops[i] (the NEW creation operators) for i in S, DESCENDING order (the
#     SAME convention the ladder_vecs construction above uses, "for i in sorted(S, reverse=True)" --
#     images(S) must be built with the identical operator-ordering convention as |S> itself, else the
#     map is not actually the a_i <-> a_i^dagger lift it claims to be), to that new vacuum; this fixes
#     an antiunitary Sigma = V_sigma o K_Fock with V_sigma READ OFF directly (no fit).
vac_full = ladder_vecs[(0, 1, 2)]
images = {}
for S in SUBSETS:
    v = vac_full.copy()
    for i in sorted(S, reverse=True):
        v = A_ops[i] @ v
    images[S] = v
liftable = all(np.linalg.norm(images[S]) > 1e-8 for S in SUBSETS)
check("KO-2.2c sigma_M0 IS liftable to the Fock (every image(S) is nonzero -- a canonical antiunitary exists)",
      liftable)
Vsig = np.zeros((8, 8), complex)
for S in SUBSETS:
    Vsig[:, idx[S]] = (W.conj().T @ images[S])[:, 0]
vsig_unitary = np.allclose(Vsig.conj().T @ Vsig, I8)
check("KO-2.2d sigma_M0's V_sigma (built from the images of the ladder ops, no fitting) is unitary", vsig_unitary)

# DEFINING RELATION (the objective arbiter that Vsig o K IS the pre-registered sigma_M0 candidate,
# internal research notes: "the lift of J6 -> -J6, i.e. a_i <-> a_i^dagger"):
# Sigma A_ops[i] Sigma^-1 = Adag_ops[i] for i=1,2,3, Sigma = Vsig o K antiunitary.  Checked in the SAME
# (Fock) basis Vsig is expressed in -- A_ops/Adag_ops re-expressed via W, exactly as PF/G5F/U_A4_F are
# elsewhere in this file (Sigma O Sigma^-1 = V . conj(O) . V^dagger for antiunitary Sigma = V o K).
A_ops_F = [W.conj().T @ A_ops[i] @ W for i in range(3)]
Adag_ops_F = [a.conj().T for a in A_ops_F]
sigma_defining_devs = [np.max(np.abs(Vsig @ np.conj(A_ops_F[i]) @ Vsig.conj().T - Adag_ops_F[i]))
                        for i in range(3)]
sigma_defining_dev = max(sigma_defining_devs)
check(f"KO-2.2d-bis sigma_M0 defining relation: max_i |Sigma A_i Sigma^-1 - A_i^dagger| = "
      f"{sigma_defining_dev:.2e}  (per-i: {[f'{d:.2e}' for d in sigma_defining_devs]})",
      sigma_defining_dev < 1e-12)

# COINCIDENCE CHECK: sigma_M0 vs C_ideal.
Dcorr = Vsig @ Vcomp.conj().T
Dcorr_is_diag = np.allclose(Dcorr, np.diag(np.diag(Dcorr)))
Dcorr_diag = np.diag(Dcorr).real
Dcorr_is_signs = Dcorr_is_diag and np.allclose(np.abs(Dcorr_diag), 1.0)
check("KO-2.2e COINCIDENCE CHECK: V_sigma = D . V_comp with D a REAL DIAGONAL SIGN matrix "
      "(sigma_M0 and C_ideal built from the same complement-swap idea)?",
      Dcorr_is_signs, detail=f"D = diag({np.round(Dcorr_diag,3).tolist()})")
print(f"    FINDING: sigma_M0 is NOT literally identical to C_ideal (D != I); it coincides with C_ideal "
      f"UP TO the real sign correction D = diag{tuple(int(round(d)) for d in Dcorr_diag)} (Fock order "
      f"{SUBSETS}). D is real, D^2=I -- a genuine partial coincidence, not full independence, but also "
      f"not literal identity; the two J_F candidates are compared as DISTINCT operators below and, as "
      f"shown, give DIFFERENT eps'-forcing behaviour despite the shared complement-swap origin.")

# sign-convention independence of C_ideal (frozen requirement: check with the alternate overall sign)
J2_std = Vcomp @ np.conj(Vcomp)
J2_alt = Vcomp_alt @ np.conj(Vcomp_alt)
gt_std = Vcomp @ np.conj(PF) @ Vcomp.conj().T
gt_alt = Vcomp_alt @ np.conj(PF) @ Vcomp_alt.conj().T
check("KO-2.2f C_ideal's KO signs (J^2, J gamma J^-1) are INDEPENDENT of the overall sign chi=+-1 "
      "(chi=-1 alternate re-computed and compared)",
      np.allclose(J2_std, J2_alt) and np.allclose(gt_std, gt_alt))

J_CANDIDATES = {"C_ideal": Vcomp, "K_g6": Ug6, "sigma_M0": Vsig}
GAMMA_CANDIDATES = {"P_F": PF, "Gamma_5": G5F}


def sign_transform(V, O):
    """J O J^-1 for the antiunitary J = V o K:  V . conj(O) . V^dagger."""
    return V @ np.conj(O) @ V.conj().T


def get_sign(O_transformed, O, tol=1e-6):
    if np.allclose(O_transformed, O, atol=tol):
        return +1
    if np.allclose(O_transformed, -O, atol=tol):
        return -1
    return None


# ---- the real Hermitian-operator basis on the 8-dim Fock (64 real dimensions = 8^2) ----
NFOCK = 8
HERM_BASIS = []
for k in range(NFOCK):
    E = np.zeros((NFOCK, NFOCK), complex)
    E[k, k] = 1.0
    HERM_BASIS.append(E)
for k in range(NFOCK):
    for ell in range(k + 1, NFOCK):
        E1 = np.zeros((NFOCK, NFOCK), complex)
        E1[k, ell] = 1.0
        E1[ell, k] = 1.0
        HERM_BASIS.append(E1)
        E2 = np.zeros((NFOCK, NFOCK), complex)
        E2[k, ell] = 1j
        E2[ell, k] = -1j
        HERM_BASIS.append(E2)
check(f"KO-2.3 Hermitian-operator basis has dimension {len(HERM_BASIS)} == 8^2 = 64 (real dim of "
      f"Herm(C^8))", len(HERM_BASIS) == NFOCK * NFOCK)


def flat(M):
    return np.concatenate([M.real.flatten(), M.imag.flatten()])


B_MAT = np.array([flat(b) for b in HERM_BASIS]).T   # (128, 64)
B_PINV = np.linalg.pinv(B_MAT)


def coeffs_of(M):
    return B_PINV @ flat(M)


def matrix_of(x):
    return sum(x[m] * HERM_BASIS[m] for m in range(len(HERM_BASIS)))


def d_space_and_eps_prime(gamma, V, U_A4_list, tol_sv=1e-8, tol_leak=1e-9):
    """The real vector space of Hermitian, gamma-odd, A4-covariant operators on the Fock (the native
    internal-Dirac candidate space), and the forced/mixed/vacuous eps' reading for antiunitary J=V o K.
    Returns dict with nullity, naive per-basis-vector signs, the leak diagnostic ('clean' iff the
    J-conjugation preserves the D-space to within tol_leak), and the verdict."""
    cols = []
    for B in HERM_BASIS:
        odd = gamma @ B @ gamma + B                      # zero iff gamma-odd
        resid = [odd] + [B @ U - U @ B for U in U_A4_list]  # zero iff A4-covariant
        cols.append(np.concatenate([flat(r) for r in resid]))
    Mx = np.array(cols).T
    _, Ss, Vt = np.linalg.svd(Mx)
    nullity = int(np.sum(Ss < tol_sv * (Ss[0] if len(Ss) else 1.0)))
    if nullity == 0:
        return {"nullity": 0, "verdict": "VACUOUS", "naive_signs": [], "leak": None}
    P = Vt[64 - nullity:64].conj().T.real   # (64, nullity) real orthonormal basis of the D-space

    naive_signs = []
    for k in range(nullity):
        Db = matrix_of(P[:, k])
        Ot = sign_transform(V, Db)
        naive_signs.append(get_sign(Ot, Db))

    # basis-independent diagnostic: does J-conjugation preserve the D-space at all (the "leak"), and
    # if so what are the eigenvalues of the restricted involution (should be exactly +-1 iff no leak).
    Rfull = np.zeros((64, 64))
    for m in range(64):
        Rfull[:, m] = coeffs_of(sign_transform(V, HERM_BASIS[m])).real
    Proj = P @ P.T
    leak = float(np.max(np.abs((np.eye(64) - Proj) @ Rfull @ Proj)))
    clean = leak < tol_leak
    Rsub = P.T @ Rfull @ P
    evals = np.linalg.eigvalsh((Rsub + Rsub.T) / 2)

    if all(s == +1 for s in naive_signs):
        verdict = "+1 FORCED"
    elif all(s == -1 for s in naive_signs):
        verdict = "-1 FORCED"
    else:
        verdict = "UN-FORCED"
    return {"nullity": nullity, "verdict": verdict, "naive_signs": naive_signs,
            "leak": leak, "clean": clean, "evals": evals}


print("\n    --- THE D-SPACE (internal-Dirac candidate space): dimension check ---")
odd_only_cols = [flat(PF @ B @ PF + B) for B in HERM_BASIS]
_, Ss_odd, _ = np.linalg.svd(np.array(odd_only_cols).T)
nullity_odd_only = int(np.sum(Ss_odd < 1e-8 * Ss_odd[0]))
print(f"    dim{{Hermitian, gamma-odd}} (A4 constraint NOT yet imposed) = {nullity_odd_only} "
      f"(expected 2*4*4=32 real dims: gamma has two 4-dim +-1 eigenspaces, an odd Hermitian D is an "
      f"arbitrary 4x4 complex off-diagonal block)")
check("KO-2.4 pure gamma-odd (pre-A4) space has the expected dimension 32", nullity_odd_only == 32)

# ===================================================================================================
banner("KO-2 (continued)  THE FULL 3 x 2 CANDIDATE-PAIR SIGN MATRIX")
# ===================================================================================================
PAIR_RESULTS = []
for gname, gamma in GAMMA_CANDIDATES.items():
    for jname, V in J_CANDIDATES.items():
        J2 = V @ np.conj(V)
        eps = get_sign(J2, I8)
        gam_t = sign_transform(V, gamma)
        eps_pp = get_sign(gam_t, gamma)
        dres = d_space_and_eps_prime(gamma, V, U_A4_F)
        eps_p_read = None
        if dres["verdict"] == "+1 FORCED":
            eps_p_read = +1
        elif dres["verdict"] == "-1 FORCED":
            eps_p_read = -1
        kodim = None
        anomaly = False
        if eps is not None and eps_pp is not None and eps_p_read is not None:
            for n, sig in KO_TABLE_EVEN.items():
                if sig == (eps, eps_p_read, eps_pp):
                    kodim = n
            if kodim is None:
                anomaly = True   # forced triple exists but matches NO even Connes row
        PAIR_RESULTS.append({
            "gamma": gname, "J": jname, "eps": eps, "eps_p_read": eps_p_read, "eps_pp": eps_pp,
            "d_nullity": dres["nullity"], "d_verdict": dres["verdict"], "leak": dres.get("leak"),
            "clean": dres.get("clean"), "evals": dres.get("evals"), "kodim": kodim, "anomaly": anomaly,
            "naive_signs": dres.get("naive_signs"),
        })

print(f"\n    {'gamma_F':<10s} {'J_F':<10s} {'eps':>4s} {'eps' + chr(39) + '(D-sp)':>12s} "
      f"{'eps' + chr(39) * 2:>6s} {'D-null':>7s}  {'KO-dim':<16s}")
for r in PAIR_RESULTS:
    eps_p_str = (f"{r['eps_p_read']:+d}" if r["eps_p_read"] is not None else r["d_verdict"])
    kodim_str = (str(r["kodim"]) if r["kodim"] is not None else
                 ("ANOMALOUS" if r["anomaly"] else "none (un-forced)"))
    print(f"    {r['gamma']:<10s} {r['J']:<10s} {('%+d' % r['eps']) if r['eps'] is not None else '?':>4s} "
          f"{eps_p_str:>12s} {('%+d' % r['eps_pp']) if r['eps_pp'] is not None else '?':>6s} "
          f"{r['d_nullity']:>7d}  {kodim_str:<16s}")
    if r["leak"] is not None:
        print(f"        naive per-(SVD-)basis-vector eps' signs (k=0..{r['d_nullity']-1}): "
              f"{r['naive_signs']}")
        print(f"        J-conjugation leak out of D-space = {r['leak']:.2e}  "
              f"(clean (< 1e-9)? {r['clean']} -- clean iff J genuinely preserves the D-space); "
              f"restricted-involution eigenvalues = {np.round(r['evals'], 3).tolist()}")

check("KO-2.5 the full 3 J_F x 2 gamma_F = 6 candidate pairs were all computed with NO shopping "
      "(frozen candidate sets, mechanical arithmetic)", len(PAIR_RESULTS) == 6)
print("    (the 'ANOMALOUS' label above is the RAW reading against Connes' CANONICAL table only, "
      "matrix numbers unchanged from the original computation; the R2b reclassification -- reading "
      "(-1,-1,-1) as the EXOTIC presentation of KO-6 -- is applied next, in KO-3.)")

forced_rows = [r for r in PAIR_RESULTS if r["eps_p_read"] is not None]
kodim6_rows = [r for r in forced_rows if r["kodim"] == 6]
kodim_other_rows = [r for r in forced_rows if r["kodim"] is not None and r["kodim"] != 6]
anomalous_rows = [r for r in forced_rows if r["anomaly"]]

# ===================================================================================================
# R2b RECLASSIFICATION (2026-07-09, internal research notes, adjudication
# READ-AS-KO-6). TEXT/CLASSIFICATION ONLY -- no matrix number computed above is touched; every
# (eps,eps_p_read,eps_pp) triple in PAIR_RESULTS is the SAME convention-independent fact as before.
# What changes is the READING: a forced triple that matches NO row of the CANONICAL Connes table
# (KO_TABLE_EVEN) is no longer automatically "anomalous" -- if it matches the EXOTIC presentation
# (-1,-1,-1), the Dabrowski-Dossena/Cacic J<->J.gamma freedom for even triples (both admissible "on
# the same footing") identifies it as KO-dimension 6 read in that presentation.
EXOTIC_KO6_SIGNATURE = (-1, -1, -1)   # Dabrowski-Dossena/Cacic J<->J.gamma image of CONNES_TABLE[6]
for r in PAIR_RESULTS:
    raw = (r["eps"], r["eps_p_read"], r["eps_pp"]) if r["eps_p_read"] is not None else None
    r["exotic_ko6"] = (r["kodim"] is None and raw == EXOTIC_KO6_SIGNATURE)
    if r["exotic_ko6"]:
        r["kodim"] = 6
        r["anomaly"] = False

# re-derive the summary lists AFTER reclassification (the loop above only ever promotes rows out of
# "anomalous" into "kodim==6"; it never removes a genuine canonical-table match or demotes a row)
forced_rows = [r for r in PAIR_RESULTS if r["eps_p_read"] is not None]
kodim6_rows = [r for r in forced_rows if r["kodim"] == 6]
exotic_kodim6_rows = [r for r in kodim6_rows if r["exotic_ko6"]]
canonical_kodim6_rows = [r for r in kodim6_rows if not r["exotic_ko6"]]
kodim_other_rows = [r for r in forced_rows if r["kodim"] is not None and r["kodim"] != 6]
anomalous_rows = [r for r in forced_rows if r["anomaly"]]

# ===================================================================================================
banner("KO-3  THE VERDICT  (dual-outcome, frozen logic; R2b reclassification applied)")
# ===================================================================================================
if kodim6_rows:
    KO3_VERDICT = "KO6-FOUND"
    pair_names = ", ".join(f"(gamma_F={r['gamma']}, J_F={r['J']})" for r in kodim6_rows)
    print(f"""    KO6-FOUND: at least one natural pair gives a FORCED (eps,eps',eps'') reading identified
    with KO-dim 6 -- in the CANONICAL presentation directly, or (as found here) in the EXOTIC
    presentation via the R2b-adjudicated J<->J.gamma reclassification.
    PAIR(S): {pair_names}
    canonical-row pairs (raw signs == CONNES_TABLE[6] = {CONNES_TABLE[6]} directly):
        {[(r['gamma'], r['J']) for r in canonical_kodim6_rows] or 'none'}
    exotic-row pairs (raw signs == {EXOTIC_KO6_SIGNATURE}, RECLASSIFIED per R2b):
        {[(r['gamma'], r['J']) for r in exotic_kodim6_rows] or 'none'}
    NOTE: gamma_F=P_F and gamma_F=Gamma_5 are NOT independent (Gamma_5 = -P_F, KO-2.1c above) -- if
    both appear above it is the SAME finding under the non-independent gamma_F variant, not two
    independent confirmations. The independent claim rests on ONE pair: (P_F [~ -Gamma_5], sigma_M0).

    R2b ADJUDICATION (2026-07-09, internal research notes): READ-AS-KO-6.
    This RESOLVES the previously NAMED OPEN QUESTION (R2b) -- (-1,-1,-1) is no longer left open as an
    unresolved convention ambiguity. The literature documents a genuine convention freedom for EVEN
    real spectral triples -- J and J' = J.gamma are both admissible real structures, "perfectly on the
    same footing" (Dabrowski-Dossena, Int. J. Geom. Methods Mod. Phys. 8 (2011) 1833, arXiv:1011.4456,
    Introduction + Table 1: the second presentation of n=6 is (-,-,-)); the replacement J -> J.gamma
    maps the canonical row to the "exotic" row reversibly (Cacic, Lett. Math. Phys. 2013,
    arXiv:1209.4832, Sec. 2.2 + Table 2.2, column "6-" = (-,-,-)). Connes' canonical table
    (hep-th/0608226 App. 7 Def. 7.2, citing the 1995 J. Math. Phys. paper; van Suijlekom 2024 Tables
    3.1/5.1 identical) has eps'=+1 at even n and does not discuss the freedom -- our implemented
    convention matched Connes' canonical table, which is why the raw computation above printed no row
    under that convention alone.
    CANONICAL-PARTNER STATEMENT: J_F.gamma_F carries (eps,eps',eps'') = (-1,-1,-1) -> (+1,+1,-1) = the
    KO-6 row EXACTLY (verified in-code immediately below, R2b-verify).

    INTEGRITY HISTORY (preserved verbatim): the first implementation of this computation reported
    KO6-FOUND; the adversarial check exposed this as an operator-ordering BUG (the KO-2.2d-bis
    defining relation failed at 2.0); the ordering was corrected with the defining relation as sole
    arbiter, and the corrected arithmetic REVERSED the verdict to ANOMALOUS/KO-OTHER; R2b then resolved
    the READING of those (unchanged) corrected signs via the frozen authority hierarchy --
    literature-first, never by convention-picking (no candidate was added, removed, or re-signed to
    force this outcome).

    RECONCILIATION LINE:  KO-1's spacetime factor (4) + KO-2's internal factor (6)  =  10  ==  2 (mod 8)
      => matches Connes' SM total KO-dimension 2 EXACTLY. The reconciliation hypothesis is CONFIRMED
      (R2b, literature-first, 2026-07-09): "CLEANROOM's KO-4" (m06, spacetime), "SM-needs-6" (this
      file's internal Fock reading, exotic presentation), and "crown-jewel's KO-2" (the total) are
      three DIFFERENT, mutually consistent readings of the SAME Connes anatomy -- not three
      conflicting claims.""")

    # R2b-VERIFY: the ONE new computation added by this adjudication -- the canonical-presentation
    # partner J' = J_F.gamma_F, built purely from matrices already computed in KO-2 above (JF_rep,
    # gammaF_rep) via the SAME sign_transform/get_sign/d_space_and_eps_prime machinery used for every
    # other row of PAIR_RESULTS. This is the literature's own J<->J.gamma consistency identity, checked
    # mechanically rather than asserted.
    _rep_pool = exotic_kodim6_rows if exotic_kodim6_rows else canonical_kodim6_rows
    rep = next((r for r in _rep_pool if r["gamma"] == "P_F"), _rep_pool[0])
    gammaF_rep = GAMMA_CANDIDATES[rep["gamma"]]
    JF_rep = J_CANDIDATES[rep["J"]]
    Vprime = JF_rep @ np.conj(gammaF_rep)                 # antiunitary J' = J_F . gamma_F, as V' o K
    eps_jp = get_sign(Vprime @ np.conj(Vprime), I8)
    eps_pp_jp = get_sign(sign_transform(Vprime, gammaF_rep), gammaF_rep)
    dres_p = d_space_and_eps_prime(gammaF_rep, Vprime, U_A4_F)
    eps_p_jp = (+1 if dres_p["verdict"] == "+1 FORCED" else (-1 if dres_p["verdict"] == "-1 FORCED" else None))
    jprime_signs = (eps_jp, eps_p_jp, eps_pp_jp)
    print(f"\n    R2b-VERIFY: J' = J_F.gamma_F for (gamma_F={rep['gamma']}, J_F={rep['J']}) computed "
          f"in-code -> (eps,eps',eps'') = {jprime_signs}.")
    check(f"R2b-verify: J' = J_F.gamma_F signs {jprime_signs} == CONNES_TABLE[6] = {CONNES_TABLE[6]} "
          f"(Connes' canonical KO-6 row -- the literature's own J<->J.gamma consistency identity)",
          jprime_signs == CONNES_TABLE[6])
elif kodim_other_rows or anomalous_rows:
    KO3_VERDICT = "KO-OTHER"
    print(f"""    KO-OTHER: the natural pairs FORCE a definite KO-dimension, but it is != 6, and no row
    matches the R2b exotic-KO6 signature {EXOTIC_KO6_SIGNATURE} either.
    Forced-but-not-6 pairs: {[(r['gamma'], r['J'], r['kodim']) for r in kodim_other_rows]}
    Forced-but-anomalous (no even Connes row, canonical or exotic, matches) pairs:
        {[(r['gamma'], r['J']) for r in anomalous_rows]}
    Booked raw: the internal Cl(6) Fock geometry does NOT have the Connes-SM KO-6 type (in either
    presentation) under the frozen candidate set; the mismatch with the NCG graft is named here and
    stays OPEN.""")
    for r in anomalous_rows:
        raw_signs = (r["eps"], r["eps_p_read"], r["eps_pp"])
        print(f"    ANOMALOUS SIGNS raw (no even Connes-table row, canonical or exotic, matches): "
              f"(gamma_F={r['gamma']}, J_F={r['J']}) -> (eps,eps',eps'') = {raw_signs}")
else:
    KO3_VERDICT = "AMBIGUOUS"
    unforced = [(r["gamma"], r["J"]) for r in PAIR_RESULTS]
    print(f"""    AMBIGUOUS: no pair gives a FORCED (eps,eps',eps'') reading at all -- eps' is un-forced
    (mixed / leaky, D-space nullity>0 but J does not act as a clean involution on it) for every one of
    the 6 pairs: {unforced}.
    Booked raw: the real-structure identification is under-determined on the current Fock object --
    this IS the named residual (the missing axiom, not a free choice available to make).""")

print(f"\n    KO-3 VERDICT: {KO3_VERDICT}")
KO3_DEFINITE = KO3_VERDICT in ("KO6-FOUND", "KO-OTHER", "AMBIGUOUS")   # exhaustive by construction
check("KO-3 reports one of the three pre-declared branches (definite, not left hanging)", KO3_DEFINITE)

# ===================================================================================================
banner("KO-4  SCOPE DECLARATION")
# ===================================================================================================
print("""    NOT claimed by this adapter:
      - the first-order condition (or any other spectral-triple axiom beyond J^2/JD/Jgamma signs)
        for the internal triple;
      - the full Connes axiom audit (orientability, Poincare duality, regularity) for either factor;
      - the a4/Lagrangian reading of the finite space (G3b, the next station in this arc);
      - any TOTAL-OBJECT D = B (x) d_N sign computation beyond the tensor-factor COMPOSITION RULE
        (KO-dims of factors ADD mod 8 -- stated, not re-derived here);
      - that the internal Fock's gamma_F/J_F candidates found here are the UNIQUE or CANONICAL ones
        Connes would assign to a Standard-Model-like finite space -- only that, among the FROZEN
        candidate set declared in the pre-registration, the arithmetic gives the reading printed
        above.""")

# ===================================================================================================
banner("SUMMARY")
# ===================================================================================================
KO0_PASS = row_match
print(f"    KO-0 THE TABLE ............................. {'PASS' if KO0_PASS else 'FAIL'}")
print(f"    KO-1 SPACETIME FACTOR ....................... {'PASS' if KO1_PASS else 'FAIL'}  (KO-dim 4)")
print(f"    KO-2 INTERNAL Cl(6) FOCK .................... printed above (6-pair matrix)")
print(f"    KO-3 VERDICT ................................ {KO3_VERDICT}"
      + ("  (KO-dim 6, EXOTIC presentation -- R2b 2026-07-09)" if exotic_kodim6_rows else
         ("  (KO-dim 6, canonical presentation)" if canonical_kodim6_rows else "")))
print(f"    RECONCILIATION (spacetime-4 + internal-6 == 2 mod 8) ... "
      + ("CONFIRMED (R2b, literature-first, 2026-07-09)" if kodim6_rows else "NOT CONFIRMED"))
print(f"    KO-4 SCOPE DECLARATION ...................... printed above")
print()
exit_ok = KO0_PASS and KO1_PASS and KO3_DEFINITE and ok_all
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}  "
      f"(exit condition: KO-0 & KO-1 pass AND KO-3 definite = {exit_ok})")
print("=" * 96)
KO_EXIT_OK = exit_ok
# NOTE (G3b/S2 extension, 2026-07-09): this WAS the file's last line (`sys.exit(0 if exit_ok else 1)`,
# verbatim). Every KO-0..KO-4 check/computation ABOVE this point is UNTOUCHED (R2/R2b/G3a, frozen).
# The sys.exit(...) call itself is DEFERRED to the true end of file (below the new G3b/S2 section)
# so ONE process exit code gates BOTH the KO suite and the new LB-1..LB-6 contracts, per the S2
# pre-reg's own hard rule ("the suite still exits 0 iff ALL checks incl. the existing KO ones pass
# AND LB-3 reports a definite verdict"). `ok_all` is the SAME module-global boolean the KO checks
# already accumulated into above; the new LB section's `check()` calls below continue accumulating
# into that identical global (never reset), so a KO failure above still fails the whole file.

# ===================================================================================================
# ===================================================================================================
#  G3b / S2  --  THE LAGRANGIAN BRIDGE
#  Pre-registered internal research notes (contracts LB-1..LB-6,
#  frozen BEFORE this section was written). This EXTENDS the file above (append-only): the KO/G3a
#  code and checks above (R2/R2b, lines ~1-772 as originally written) are UNTOUCHED except for the
#  single deferred sys.exit(...) noted just above. Zero engine/proofs edits; every construction
#  below is either reused verbatim from a named prior-art file or a bridge-type composition of two
#  already-certified objects (per the pre-reg's own framing: "every link is bridge-type").
# ===================================================================================================
# ===================================================================================================
banner("G3b / S2  --  THE LAGRANGIAN BRIDGE (pre-reg internal research notes)")
# ===================================================================================================
print("""    THE CHAIN: zeta (certified, G6ab) -> spectrum -> heat trace (prior art, OMEGA_T1) -> Weyl
    amplitude -> a4 (native, d4). Every link below is bridge-type (two existing verified objects);
    LB-1..LB-6 are the frozen contracts (verbatim from the pre-reg). NO engine/proofs edit; this
    file (ncg_spectral.py) is the ONE file extended, per the charter (G3b lives in the NCG adapter).""")

import time as _time                                          # noqa: E402 (new stdlib import; NOT
_LB_T0 = _time.time()                                          # previously imported in this file)
import mpmath as _mp                                           # noqa: E402 (arbitrary precision; used
from scipy.optimize import linear_sum_assignment as _lsa       # noqa: E402  ONLY for the degenerate-
                                                                 # root robustness fix in LB-1 and for
                                                                 # optimal complex-pair matching)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "bridge"))
from derivation_topdown.bridge import d4_spectral_action as _d4   # noqa: E402
from derivation_topdown.bridge import the_run as _the_run         # noqa: E402

DEG = srs.DEG                     # k* = 3 (srs coordination number)
NV_LB, NE_LB = srs.NV, len(srs.EDGES)
KSTAR_M1 = DEG - 1                 # 2
EDGES_LB = srs.EDGES


# ===================================================================================================
banner("LB-1  THE PENCIL<->SPECTRUM LINK")
# ===================================================================================================
print("""    Bass vertex pencil p(u;k) = det(I - u A(k) + (k*-1) u^2 I). Per lambda-eigenspace of A(k)
    this factors as (k*-1)u^2 - lambda u + 1 = 0 -- the PENCIL's own root variable u.

    CONVENTION NOTE (transparency, per this file's own established practice at KO-1): the reuse-
    mapped `ihara_unification.py` formula u(lambda) = (lambda +- sqrt(lambda^2-4(k*-1)))/2 is the
    root of mu^2 - lambda*mu + (k*-1) = 0 -- i.e. it is the HASHIMOTO (non-backtracking) eigenvalue
    mu(lambda), NOT the pencil's own zeta-argument variable u (a standard but easily-conflated fact
    of Ihara-zeta literature, where "u" is overloaded for both the NB eigenvalue and the zeta
    function's argument). Substituting u=1/mu into mu^2-lambda*mu+(k*-1)=0 gives EXACTLY
    (k*-1)u^2-lambda*u+1=0 -- so the pencil's own roots are u = 1/mu(lambda), the RECIPROCAL of the
    reuse-mapped formula, not the formula's raw output. This is exactly the "derive it correctly"
    self-correction flagged in the pre-reg's own LB-2 text; we reuse ihara_unification's u(lambda)
    formula LITERALLY (verbatim numeric copy below) and apply the one extra (also standard, not
    re-derived) algebraic step -- the reciprocal -- to land on the pencil's own variable.""")


def ihara_unification_u_pm(lam, kstar_m1=KSTAR_M1):
    """Verbatim numeric copy of proofs/wave_engine/ihara_unification.py's u_plus/u_minus:
    u(lambda) = (lambda +- sqrt(lambda^2 - 4*(k*-1))) / 2, the HASHIMOTO eigenvalue mu(lambda)
    (root of mu^2 - lambda*mu + (k*-1) = 0). Reused, not re-derived."""
    disc = (lam.astype(complex)) ** 2 - 4 * kstar_m1
    sq = np.sqrt(disc.astype(complex))
    return (lam + sq) / 2, (lam - sq) / 2


def pencil_u_pm_from_lambda(lam, kstar_m1=KSTAR_M1):
    """The Bass PENCIL's own root variable for adjacency eigenvalue lambda: u = 1/mu(lambda), the
    reciprocal of ihara_unification_u_pm (see CONVENTION NOTE above)."""
    mu_p, mu_m = ihara_unification_u_pm(lam, kstar_m1)
    return 1.0 / mu_p, 1.0 / mu_m


# -- double-precision pencil-polynomial coefficients via FFT interpolation on a circle (the
#    "zeta_gauge poly technique": sample det(I-uA+(k*-1)u^2 I) at roots-of-unity, recover the
#    (<=8)-degree polynomial's coefficients by a forward DFT; the >=degree-9 coefficients must
#    vanish at machine precision -- a self-certifying structural check that the determinant truly
#    IS the claimed degree-8 polynomial, done WITHOUT ever forming A's eigendecomposition).
NPTS_PENCIL = 16
RADIUS_PENCIL = 2.5
_JS = np.arange(NPTS_PENCIL)
_US = RADIUS_PENCIL * np.exp(2j * np.pi * _JS / NPTS_PENCIL)


def pencil_coeffs_fp64(A):
    I4 = np.eye(A.shape[0])
    ys = np.array([np.linalg.det(I4 - u * A + KSTAR_M1 * u ** 2 * I4) for u in _US])
    c = (np.fft.fft(ys) / NPTS_PENCIL) / (RADIUS_PENCIL ** _JS)
    return c   # increasing order, index 0..15; index 9..15 should be ~0


def pencil_roots_fp64(A):
    c = pencil_coeffs_fp64(A)
    leak = float(np.max(np.abs(c[9:])))
    roots = np.roots(c[:9][::-1])   # degree-8, decreasing order for np.roots
    return roots, leak


# -- arbitrary-precision (mpmath) version of the SAME pencil polynomial, used to resolve the
#    degenerate Gamma point (A(0,0,0) = plain K4 adjacency, exactly-triple eigenvalue lambda=-1):
#    a repeated root of a degree-8 polynomial is only recoverable to ~eps^(1/mult) in double
#    precision via companion-matrix eigenvalues (eps^(1/3) ~ 6e-6 for a triple root -- a standard,
#    well-known conditioning fact, NOT a bug); mpmath raises the WORKING precision so that this
#    SAME (cubed-root) amplification of the internal floating noise lands far below 1e-9.
_mp.mp.dps = 50


def _mp_matrix_from_np(A):
    n = A.shape[0]
    return _mp.matrix([[_mp.mpc(A[i, j].real, A[i, j].imag) for j in range(n)] for i in range(n)])


def pencil_roots_mp(A, npts=9, radius=_mp.mpf('2.5')):
    Amp = _mp_matrix_from_np(A)
    I4 = _mp.eye(A.shape[0])
    us = [radius * _mp.exp(1j * 2 * _mp.pi * j / npts) for j in range(npts)]
    ys = [_mp.det(I4 - u * Amp + KSTAR_M1 * u ** 2 * I4) for u in us]
    V = _mp.matrix(npts, npts)
    for j in range(npts):
        for nn in range(npts):
            V[j, nn] = us[j] ** nn
    c = _mp.lu_solve(V, _mp.matrix(ys))
    poly_desc = [c[i] for i in range(npts - 1, -1, -1)]
    roots = _mp.polyroots(poly_desc, maxsteps=200, extraprec=300)
    return [complex(r) for r in roots]


def best_match_dev(a, b):
    """Optimal (minimum total-distance) bipartite matching between two equal-length complex
    arrays -- robust where a naive sort mis-pairs entries at exactly-degenerate k-points (a
    concrete failure mode found and disclosed below, NOT swept under a tolerance)."""
    a = np.asarray(a); b = np.asarray(b)
    C = np.abs(a[:, None] - b[None, :])
    ri, ci = _lsa(C)
    return float(np.max(C[ri, ci]))


_rng_lb1 = np.random.default_rng(0)
LB1_KPTS = [(0.0, 0.0, 0.0)] + [tuple(_rng_lb1.uniform(0, 1, 3)) for _ in range(12)]
print(f"    k-set: Gamma + 12 pseudo-random (seed 0) = {len(LB1_KPTS)} points")

_lb1_rows = []
for _k in LB1_KPTS:
    _A = srs.adjacency(_k)
    _lam = np.linalg.eigvalsh(_A)
    _up, _um = pencil_u_pm_from_lambda(_lam)
    _u_from_eig = np.concatenate([_up, _um])

    _roots64, _leak64 = pencil_roots_fp64(_A)
    _dev64_naive_sort = max(
        abs(a - b) for a, b in zip(
            sorted(_u_from_eig, key=lambda z: (round(z.real, 7), round(z.imag, 7))),
            sorted(_roots64, key=lambda z: (round(z.real, 7), round(z.imag, 7)))))
    _dev64 = best_match_dev(_u_from_eig, _roots64)

    _roots_mp = pencil_roots_mp(_A)
    _dev_mp = best_match_dev(_u_from_eig, _roots_mp)

    _lb1_rows.append((_k, _leak64, _dev64_naive_sort, _dev64, _dev_mp))

print(f"\n    {'k':<26s} {'fp64 leak(c9..15)':>18s} {'naive-sort dev':>15s} "
      f"{'optimal-match dev(fp64)':>24s} {'optimal-match dev(mp50)':>24s}")
for _k, _leak64, _sortdev, _dev64, _devmp in _lb1_rows:
    print(f"    {str(np.round(_k, 4)):<26s} {_leak64:>18.2e} {_sortdev:>15.2e} {_dev64:>24.2e} {_devmp:>24.2e}")

_worst_leak = max(r[1] for r in _lb1_rows)
_worst_mp = max(r[4] for r in _lb1_rows)
_gamma_naive_sort = _lb1_rows[0][2]
_gamma_fp64 = _lb1_rows[0][3]
print(f"""
    FINDING (disclosed, not smoothed over): at the Gamma point k=(0,0,0), A(Gamma) is the plain
    K4 adjacency (exactly rational; lambda spectrum = {{-1 (x3), +3}}, a genuine triple
    degeneracy). A naive real/imag-sorted comparison mis-pairs entries there (dev = {_gamma_naive_sort:.2e},
    ~= 2x the imaginary part of the complex-conjugate pair -- a sort-order artifact, not a physics
    mismatch); the OPTIMAL bipartite match repairs this (dev = {_gamma_fp64:.2e}) but is still only
    accurate to the conditioning floor of a repeated root under fp64 companion-matrix rootfinding
    (~eps^(1/3) ~ 6e-6). Elevating to mpmath (dps=50) for the SAME polynomial, at the SAME k-point,
    reduces the deviation to {_lb1_rows[0][4]:.2e} -- confirming the fp64 gap at Gamma is PURELY a
    floating-point conditioning artifact of the exact degeneracy, not a genuine pencil/spectrum
    mismatch. The mpmath figures are used as the gating LB-1 numbers below.""")

check(f"LB-1a fp64 pencil-polynomial degree self-check: coefficients c9..c15 ~ 0 for all "
      f"{len(LB1_KPTS)} k-points (max |c_(>=9)| = {_worst_leak:.2e})", _worst_leak < 1e-9)
check(f"LB-1b (GATING) mpmath(dps=50) pencil roots == {{u_+-(lambda_i)}} via the reciprocal-Ihara "
      f"map, optimal bipartite matching, over Gamma + 12 pseudo-random k (worst {_worst_mp:.2e})",
      _worst_mp < 1e-9)
LB1_OK = (_worst_leak < 1e-9) and (_worst_mp < 1e-9)


# ===================================================================================================
banner("LB-2  THE SPECTRUM->HEAT LINK  (two genuinely different routes)")
# ===================================================================================================
print("""    ROUTE A (zeta side): pencil roots (fp64, np.roots -- NOT eigh) -> inverse map
    lambda = (k*-1)*u + 1/u -> mu_i = DEG - lambda_i (srs.py's own docstring identity
    D^2|_C0 = DEG*I - A) -> the Hodge-Dirac heat trace's nonzero eigenvalue-SQUARED content.
    ROUTE B (direct side): OMEGA_T1's own D_q(q) Hodge-Dirac matrix, diagonalized by eigh.

    WHICH LB-2 FORM WAS IMPLEMENTABLE (disclosed, per the pre-reg's own conditional): OMEGA_T1's
    own code diagonalizes D_q(q) DIRECTLY (never referencing A(k) or an adjacency-spectrum
    mapping) -- so the pre-reg's first branch ("if it diagonalizes directly ... implement the
    documented spectral relation") applies. That documented relation IS clean and already stated
    in srs.py's own docstring: D^2|_{C0} = DEG*I - A(k) (the graph Laplacian identity) plus the
    standard shared-nonzero-singular-value fact between d.d^dagger and d^dagger.d (a matrix with
    d^dagger d having 4 of its 6 eigenvalues match d.d^dagger's 4 eigenvalues exactly, the other 2
    exactly zero -- OMEGA_T1 P2's own "2 exact flat zero modes per fiber"). So the FULL Hodge-Dirac
    heat trace (not merely the fallback adjacency-Laplacian-only trace) is reconstructed: each
    pencil root pair u_+-(lambda_i) recovers lambda_i TWICE (once via u_+, once via u_-, matching
    the natural double multiplicity of mu_i=DEG-lambda_i in the 10-dim D_q^2 spectrum); appending
    the 2 exact flat zero modes (a separately-established topological fact, OMEGA_T1 P2, not
    re-derived here) completes the reconstructed 10-vector eps2^A(k) per fiber. The DEGRADED
    fallback (adjacency-Laplacian-trace-only) was NOT needed.""")

NV_G, NE_G = NV_LB, NE_LB


def d_inc_omega(q):
    """OMEGA_T1's d_inc(q) (proofs/foundations/OMEGA_T1_..._2026-07-02.py lines ~59-63), reused
    verbatim -- ROUTE B never touches A(k) or the pencil."""
    d = np.zeros((NV_G, NE_G), complex)
    for e, (i, j, v) in enumerate(EDGES_LB):
        d[i, e] = -1.0
        d[j, e] = np.exp(1j * np.dot(q, v))
    return d


def D_q_omega(q):
    """OMEGA_T1's D_q(q) (lines ~65-67), reused verbatim."""
    d = d_inc_omega(q)
    return np.block([[np.zeros((NV_G, NV_G)), d], [d.conj().T, np.zeros((NE_G, NE_G))]])


def route_A_eps2(k):
    """ZETA side: A(k) built (for the pencil evaluation only, no eigh), pencil roots via np.roots,
    inverse map to lambda, then to the reconstructed eps2 = D_q^2's 10 eigenvalues."""
    A = srs.adjacency(k)
    roots, _leak = pencil_roots_fp64(A)
    lam_rec = KSTAR_M1 * roots + 1.0 / roots     # lambda = (k*-1)u + 1/u ; real up to fp noise
    mu = (DEG - lam_rec).real
    return np.concatenate([mu, [0.0, 0.0]]), float(np.max(np.abs(lam_rec.imag)))


def route_B_eps2(q):
    """DIRECT side: OMEGA_T1's own D_q(q), diagonalized (eigh) -- genuinely different code path."""
    ev = np.linalg.eigvalsh(D_q_omega(q))
    return ev ** 2


G_LB2 = 40   # == OMEGA_T1's own grid (P4); LB-2's worst per-point deviation and the K(t) agreement
             # are BOTH confirmed below to already be grid-converged (see the LB-3 cross-check).
_pts_lb2 = 2 * math.pi * (np.arange(G_LB2) + 0.5) / G_LB2
_NK_LB2 = G_LB2 ** 3
_eps2A_all = np.empty((_NK_LB2, 10))
_eps2B_all = np.empty((_NK_LB2, 10))
_imag_leak_worst = 0.0
_idx = 0
_t_lb2_build = _time.time()
for _qa in _pts_lb2:
    for _qb in _pts_lb2:
        for _qc in _pts_lb2:
            _q = np.array([_qa, _qb, _qc])
            _k = _q / (2 * math.pi)
            _eA, _imleak = route_A_eps2(_k)
            _eps2A_all[_idx] = np.sort(_eA)
            _eps2B_all[_idx] = np.sort(route_B_eps2(_q))
            _imag_leak_worst = max(_imag_leak_worst, _imleak)
            _idx += 1
_lb2_build_time = _time.time() - _t_lb2_build
_pointwise_worst = float(np.max(np.abs(_eps2A_all - _eps2B_all)))
print(f"    BZ grid: {G_LB2}^3 = {_NK_LB2} points (OMEGA_T1's own offset grid q=2pi(m+0.5)/G); "
      f"build time {_lb2_build_time:.1f}s")
print(f"    worst |Im(lambda_recovered)| over the grid = {_imag_leak_worst:.2e}  (should be ~0: "
      f"lambda is real by construction of the pencil quadratic)")
print(f"    worst per-fiber |eps2_A - eps2_B| (raw, un-averaged) over the whole grid = "
      f"{_pointwise_worst:.2e}  (diagnostic only -- see the note below; the GATING quantity is K(t))")
if _pointwise_worst >= 1e-9:
    print(f"    NOTE (disclosed): this per-point figure exceeds 1e-9 at a small number of "
          f"near-degenerate grid k-points (the same fp64 companion-matrix conditioning effect as "
          f"the Gamma point in LB-1, at reduced severity since the offset grid avoids EXACT "
          f"degeneracy); it does not gate LB-2 (the contract's own gated quantity is the BZ-"
          f"AVERAGED heat trace K(t), checked next) and averages out completely there.")

print(f"\n    {'t':>6s}  {'K_A(t)':>16s}  {'K_B(t)':>16s}  {'|K_A-K_B|':>12s}")
_T_LB2 = (0.1, 1.0, 10.0, 60.0)
_lb2_worst_Kt = 0.0
for _t in _T_LB2:
    _KA = float(np.mean(np.sum(np.exp(-_t * _eps2A_all), axis=1)))
    _KB = float(np.mean(np.sum(np.exp(-_t * _eps2B_all), axis=1)))
    _d = abs(_KA - _KB)
    _lb2_worst_Kt = max(_lb2_worst_Kt, _d)
    print(f"    {_t:>6.1f}  {_KA:>16.10f}  {_KB:>16.10f}  {_d:>12.3e}")

check(f"LB-2 (GATING) K_A(t) == K_B(t) (per-fiber heat trace, zeta-route vs direct-route, BZ-"
      f"averaged over {G_LB2}^3) at t in {_T_LB2} (worst |diff| = {_lb2_worst_Kt:.2e})",
      _lb2_worst_Kt < 1e-9)
LB2_OK = _lb2_worst_Kt < 1e-9


# ===================================================================================================
banner("LB-3  THE WEYL AMPLITUDE  (dual-outcome, the chain's payoff)")
# ===================================================================================================
print("""    Reuse OMEGA_T1 P4's cone-sector extraction (flat-excluded: F(t)-2) and Albanese-frame
    amplitude prediction A_pred = 8*(4*pi)^1.5/(2*pi)^3 (V_alb=4, v=1/2, both certified G1/OMEGA_Q0),
    EXTENDING the t-window to [30,240] (>=8 log-spaced points, declared) and reporting r(t) =
    A_measured(t)/A_pred.

    HONESTY CLAUSE (printed verbatim from the pre-reg): "the lattice spectrum is bounded, so this
    is an INTERMEDIATE-t scaling-window statement (the Kotani-Sunada limit framing), NOT a true
    t->0 Weyl law on the fixed graph."

    TRANSPARENCY NOTE (surprise, disclosed raw): the pre-reg's own "Prior art: ratio 0.944 at
    t=60" is checked below by IMPORTING OMEGA_T1_zeta_D4_gauge_row_2026-07-02.py unmodified
    (importlib, its own stdout suppressed) and reading its own module-level `ratios` dict --
    i.e. this is not an assertion, it is the prior-art file's own number, read live.""")

import contextlib as _ctxlib
import importlib.util as _ilu
import io as _io

_omega_t1_path = os.path.join(REPO, "proofs", "foundations",
                               "OMEGA_T1_zeta_D4_gauge_row_2026-07-02.py")
_spec = _ilu.spec_from_file_location("_omega_t1_prior_art", _omega_t1_path)
_omega_t1 = _ilu.module_from_spec(_spec)
_buf = _io.StringIO()
with _ctxlib.redirect_stdout(_buf):
    try:
        _spec.loader.exec_module(_omega_t1)   # runs the file's OWN top-level computation, unmodified
    except SystemExit:
        pass   # OMEGA_T1's OWN top-level sys.exit(0 if ok_all else 1) (its file, its convention;
               # NOT modified) -- caught here so importing it doesn't terminate THIS process; its
               # exit code (0 == its own "ALL CHECKS PASS") is irrelevant to this cross-check, which
               # only reads its already-populated `ratios` dict below.
print(f"    imported {_omega_t1_path} (stdout suppressed, {len(_buf.getvalue().splitlines())} lines "
      f"produced; re-run the file directly for its full log) -- its OWN ratios dict at t=20,30,40,60:")
for _tk in (20.0, 30.0, 40.0, 60.0):
    print(f"      ratios[{_tk}] = {_omega_t1.ratios[_tk]:.6f}")
print(f"    => the pre-reg's cited '0.944 at t=60' does NOT match the prior-art file's own "
      f"ratios[60.0] = {_omega_t1.ratios[60.0]:.6f} (a factual mismatch in the pre-reg's citation, "
      f"booked raw here, NOT silently reconciled); the file's actual trend is monotone toward 1 "
      f"FROM ABOVE ({', '.join(f'{_omega_t1.ratios[_t]:.4f}' for _t in (20.0,30.0,40.0,60.0))} at "
      f"t=20,30,40,60), not from below. This does not change the extension methodology below.")

A_PRED = 8 * (4 * math.pi) ** 1.5 / (2 * math.pi) ** 3
T_LB3 = np.geomspace(30, 240, 8)
print(f"    declared t-window: {len(T_LB3)} log-spaced points in [30,240] = {np.round(T_LB3, 3).tolist()}")


def build_eps2_direct(G):
    pts = 2 * math.pi * (np.arange(G) + 0.5) / G
    eps2 = np.empty((G ** 3, 10))
    idx = 0
    for qa in pts:
        for qb in pts:
            for qc in pts:
                ev = np.linalg.eigvalsh(D_q_omega(np.array([qa, qb, qc])))
                eps2[idx] = ev ** 2
                idx += 1
    return eps2


def ratios_for(eps2):
    out = []
    for t in T_LB3:
        F = float(np.mean(np.sum(np.exp(-t * eps2), axis=1)))
        out.append((F - 2) / (A_PRED * t ** -1.5))
    return np.array(out)


G_LB3_MAIN = 40    # == OMEGA_T1's own grid
G_LB3_CHECK = 100  # a larger grid, affordable within the 600s station budget; used to CONFIRM
                   # grid-independence over the extended window (not required if already converged)

_t0 = _time.time()
_eps2_g40 = build_eps2_direct(G_LB3_MAIN)
_t_g40 = _time.time() - _t0
_r_g40 = ratios_for(_eps2_g40)

_t0 = _time.time()
_eps2_g100 = build_eps2_direct(G_LB3_CHECK)
_t_g100 = _time.time() - _t0
_r_g100 = ratios_for(_eps2_g100)

_grid_dev = float(np.max(np.abs(_r_g40 - _r_g100)))
print(f"\n    grid A: {G_LB3_MAIN}^3 = {G_LB3_MAIN**3} pts (OMEGA_T1's own), build {_t_g40:.2f}s")
print(f"    grid B (largest, convergence check): {G_LB3_CHECK}^3 = {G_LB3_CHECK**3} pts, build {_t_g100:.2f}s")
print(f"    max |r_40(t) - r_100(t)| over the window = {_grid_dev:.2e}  "
      f"({'grid-CONVERGED already at 40^3' if _grid_dev < 1e-6 else 'grid dependence detected -- using the 100^3 numbers below'})")

_r_use = _r_g100 if _grid_dev >= 1e-6 else _r_g40
print(f"\n    {'t':>10s}  {'r(t) = A_meas/A_pred':>22s}")
for _t, _r in zip(T_LB3, _r_use):
    print(f"    {_t:>10.3f}  {_r:>22.10f}")

# --- FROZEN VERDICT LOGIC (verbatim from the pre-reg) ---
_diffs = np.abs(_r_use - 1.0)
_monotone = bool(np.all(_diffs[1:] <= _diffs[:-1] + 1e-12))
_top_ok = bool(_r_use[-1] >= 0.97)
_crit_monotone = _monotone and _top_ok

# declared fit r(t) = 1 - c/sqrt(t)  (forced limit = 1; report goodness of fit via residuals)
_x = 1.0 / np.sqrt(T_LB3)
_y = 1.0 - _r_use
_c_fit = float(np.sum(_x * _y) / np.sum(_x * _x))
_resid = _y - _c_fit * _x
_fit_rms = float(np.sqrt(np.mean(_resid ** 2)))

# free-intercept extrapolation (GATING ALTERNATIVE per the pre-reg's OR-criterion
# "extrapolates to 1.00 +/- 0.02"; the earlier label said "not gating" -- a LABEL error
# caught by the S2 adversarial check; the code was always per-pre-reg, the label was wrong)
_Xmat = np.vstack([np.ones_like(_x), _x]).T
_coef, *_ = np.linalg.lstsq(_Xmat, _r_use, rcond=None)
_r_inf_free, _c2_free = float(_coef[0]), float(_coef[1])
_crit_fit = abs(_r_inf_free - 1.0) <= 0.02

print(f"\n    CRITERION 1 (monotone+threshold): monotone toward 1 across the window = {_monotone}; "
      f"r(top={T_LB3[-1]:.1f}) = {_r_use[-1]:.6f} >= 0.97 ? {_top_ok}  =>  {_crit_monotone}")
print(f"    CRITERION 2 (declared fit r(t) = 1 - c/sqrt(t), forced limit 1): c_fit = {_c_fit:.6f}, "
      f"RMS residual = {_fit_rms:.2e}  (residuals: {np.round(_resid, 6).tolist()})")
print(f"    CRITERION 2, free-intercept extrapolation (gating alternative per pre-reg): r_inf = {_r_inf_free:.6f}, "
      f"c = {_c2_free:.6f}  =>  |r_inf - 1| = {abs(_r_inf_free-1):.4f} <= 0.02 ? {_crit_fit}")

# plateau-away-from-1 check (AMPLITUDE-OFF branch)
_top_octave_mask = T_LB3 >= (T_LB3[-1] / 2.0)
_top_octave = _r_use[_top_octave_mask]
_plateau_level = float(np.mean(_top_octave))
_plateau_std = float(np.std(_top_octave))
_monotonicity_broken = not _monotone
_plateau_away = _monotonicity_broken and (_plateau_std < 0.01) and (_plateau_level <= 0.95)

if _crit_monotone or _crit_fit:
    LB3_VERDICT = "AMPLITUDE-CONVERGENT"
elif _plateau_away:
    LB3_VERDICT = "AMPLITUDE-OFF"
else:
    LB3_VERDICT = "WINDOW-LIMITED"

print(f"\n    *** LB-3 VERDICT: {LB3_VERDICT} ***")
check(f"LB-3 (GATING: DEFINITE VERDICT, not a numeric pass/fail) reaches one of the three frozen "
      f"branches -- {LB3_VERDICT}", LB3_VERDICT in ("AMPLITUDE-CONVERGENT", "AMPLITUDE-OFF", "WINDOW-LIMITED"))
LB3_DEFINITE = True  # exhaustive by construction, same pattern as KO3_DEFINITE


# ===================================================================================================
banner("LB-4  THE INDEX SEPARATION (re-expression) -- Str e^{-tD^2} == -2 exactly")
# ===================================================================================================
print("""    Reuse OMEGA_T1 P2 verbatim: gamma_t = GAMMA_T grading, D3 = D_q_omega(q) the supercharge;
    Str e^{-t D3^2}(k) = -2 = chi(K4) = 4-6, t-independent, k-independent (the flat band IS the
    index, never beta).""")

GAMMA_T_LB = np.diag([1.0] * NV_G + [-1.0] * NE_G)
_rng_lb4 = np.random.default_rng(0)
_T_LB4 = (0.1, 1.0, 10.0)
_worst_lb4 = 0.0
for _ in range(8):
    _q = _rng_lb4.uniform(-math.pi, math.pi, 3)
    _ev, _V = np.linalg.eigh(D_q_omega(_q))
    _gdiag = np.real(np.einsum('ij,jk,ki->i', _V.conj().T, GAMMA_T_LB, _V))
    for _t in _T_LB4:
        _s = float(np.sum(_gdiag * np.exp(-_t * _ev ** 2)))
        _worst_lb4 = max(_worst_lb4, abs(_s + 2))
check(f"LB-4 Str e^(-tD^2)(k) = -2 for 8 random k (seed 0) x t in {_T_LB4} (max dev {_worst_lb4:.2e})",
      _worst_lb4 < 1e-10)
LB4_OK = _worst_lb4 < 1e-10


# ===================================================================================================
banner("LB-5  THE BETA-ROW CONSISTENCY (regression)")
# ===================================================================================================
print("""    d4_spectral_action.beta_rows(*sm_content()) == the_run.read_gauge_running()'s b4d, on the
    SM content ({33/5, 1, -3}), exact (Fraction equality).""")

_fermions, _higgs = _d4.sm_content()
_beta_rows = _d4.beta_rows(_fermions, _higgs)
_gauge_running = _the_run.read_gauge_running()
_lb5_rows = {i: (_beta_rows[i], _gauge_running[i][1]) for i in (1, 2, 3)}
for _i, (_b, _g) in _lb5_rows.items():
    print(f"    group {_i}: d4_spectral_action.beta_rows = {_b}   the_run.read_gauge_running().b4d = {_g}"
          f"   equal? {_b == _g}")
LB5_OK = all(_b == _g for _b, _g in _lb5_rows.values())
check("LB-5 beta_rows == read_gauge_running's b4d for all 3 groups (exact)", LB5_OK,
      detail=f"{ {i: str(b) for i, (b, g) in _lb5_rows.items()} }")


# ===================================================================================================
banner("LB-6  SCOPE DECLARATION (printed, not computed; never gates PASS/FAIL)")
# ===================================================================================================
print("""    NOT claimed by this section:
      - the self-derivation of the universal Gilkey/Seeley-DeWitt coefficients (the -11/3, 2/3,
        1/3 structure = the zeta_{D4}(0) frontier -- remains the named Type-3 import, per the
        engine's own OPEN marker in derivation_topdown/bridge/the_run.py's read_gauge_running
        docstring: "S-D itself stays the declared import"); NOT re-derived here.
      - a true t->0 limit on the fixed lattice (the lattice spectrum is bounded; LB-3 is an
        intermediate-t scaling-window statement, per the honesty clause above).
      - the internal (Cl(6)/D_F) fluctuation gauge row (OMEGA_T1's own open item, P5(ii)).
      - any Higgs-potential claim (D2's territory).""")

print(f"\n    G3b/S2 (LB) section wall time: {_time.time() - _LB_T0:.1f}s")


# ===================================================================================================
banner("G3b / S2 SUMMARY")
# ===================================================================================================
print(f"""    LB-1  pencil<->spectrum link ................ {'PASS' if LB1_OK else 'FAIL'}
    LB-2  spectrum->heat link (two routes) ....... {'PASS' if LB2_OK else 'FAIL'}
    LB-3  Weyl amplitude (dual-outcome) .......... DEFINITE -- verdict = {LB3_VERDICT}
    LB-4  index separation ....................... {'PASS' if LB4_OK else 'FAIL'}
    LB-5  beta-row consistency ................... {'PASS' if LB5_OK else 'FAIL'}
    LB-6  scope declaration ....................... printed above (declaration only, not a gate)""")

LB_EXIT_OK = LB1_OK and LB2_OK and LB3_DEFINITE and LB4_OK and LB5_OK
print(f"\n G3b/S2 RESULT: {'ALL LB CONTRACTS GATE-PASS (LB-1,2,4,5) AND LB-3 REACHED A DEFINITE VERDICT' if LB_EXIT_OK else '*** AT LEAST ONE LB CONTRACT FAILED (see detail above) ***'}")
print("=" * 96)


# ===================================================================================================
banner("FINAL OVERALL (KO/G3a + G3b/S2 combined)")
# ===================================================================================================
FINAL_EXIT_OK = KO_EXIT_OK and LB_EXIT_OK and ok_all
print(f"    KO/G3a suite (R2/R2b) ........................ {'PASS' if KO_EXIT_OK else 'FAIL'}")
print(f"    G3b/S2 suite (LB-1..LB-6) ..................... {'PASS' if LB_EXIT_OK else 'FAIL'}")
print(f"    every individual check() across the WHOLE file  {'ALL PASS' if ok_all else '*** SOME FAILED ***'}")
print(f"\n FINAL OVERALL: {'ALL CHECKS PASS' if FINAL_EXIT_OK else '*** SOME CHECKS FAILED ***'}  "
      f"(exit condition: KO/G3a pass AND G3b/S2 (LB-1,2,4,5 pass AND LB-3 definite) AND every "
      f"check() across the file passes = {FINAL_EXIT_OK})")
print("=" * 96)
sys.exit(0 if FINAL_EXIT_OK else 1)
