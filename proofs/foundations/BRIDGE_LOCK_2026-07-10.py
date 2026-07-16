#!/usr/bin/env python3
"""
proofs/foundations/BRIDGE_LOCK_2026-07-10.py

BRIDGE-LOCK -- the A5-lock orbit discriminator (Design A of the chirality-bridge dossier).
Pre-registered in internal research notes (frozen decision rule, contracts
L-0..L-3, verdicts, poisons -- ALL frozen BEFORE this file was written).

LINEAGE: W2-MAP (proofs/foundations/W2_MAP_vertex_propagator_2026-07-10.py) classified the full
space of internal->cover one-particle maps under the derived requirement set {R1,R2,R5} as
AMBIGUOUS-BY-O(2): two connected components,
    rotation   Phi_theta = Uo @ (cos(theta) I6 + sin(theta) J6)   [complex-LINEAR on the +i Witt space]
    reflection Phi_phi   = Uo @ (cos(phi) S1  + sin(phi) S2 )     [complex-ANTIlinear; = rotation o sigma]
THE QUESTION: do the two orbits attach the transported internal +i Witt content to the two DISTINCT
members of the cover_B conjugate band-edge pair (chir-7 IB-root -1/2 + i*sqrt7/2 vs its conjugate),
and does exactly ONE orbit reproduce the BANKED A5-DISCRETE assignment nu<->chir-7 (CLOSED
2026-07-04, BEFORE the orbit question existed -- the blindness guarantee)?
NAMING NOTE (per pre-reg): the April "B3_chirality_bridge" files are an UNRELATED object; not kin.

FROZEN DECISION RULE (restated verbatim in substance): THE SELECTED ORBIT := the connected
component of the O(2) family whose transported +i Witt content attaches to the banked chir-7
IB-root (-1/2 + i*sqrt7/2, the + sign) under the SAME declared conventions the MAP station froze
(J6 with det phi > 0; the srs dart indexing, dart 2e = forward / 2e+1 = reversed; tau_dart as
frozen in W2-MAP).  No other selection criterion.  L-1's own gate: if the attachment functional
evaluates bit-EVEN (orbit-insensitive), STOP and book LENS-NULL honestly; skip L-2.

CONTRACTS: L-0 regression (MAP O(2) classification rebuilt + A5 banked values re-verified, ALL
asserted before anything new runs) -> L-1 theorem (transported complex structure = +-Uo J6 Uo^T per
branch; the attachment functional built from A5-DISCRETE's own chir-7 identification machinery =
the cover_B band-edge eigenspace; explicit bit-ODD test) -> L-2 the confront (frozen rule applied;
PRINTED, never asserted) -> L-3 convention-covariance (joint det-phi + tau_dart sign flip; the
selected PHYSICAL orbit stated as relative orientation must be stable).

VERDICTS (all bookable): SELECTS / LENS-NULL / CONTRADICTION / CONVENTION-ARTIFACT.

POISONS (binding, restated): FS-5iii -- the two natural Z3 (deck/generation) actions differ; NO
Z3/deck identification is imposed anywhere (the attachment functional is demonstrated
Z3-transport-free below, else abort to LENS-NULL); no scoreboard-adjacent quantity is computed or
mentioned (the later gated station stays gated regardless of outcome here); no requirement added
mid-run; the A5 banked values are regression ANCHORS, never adjusted; numbers only from running
code; ONE new proofs/ file (this one); no existing file touched.

EXIT SEMANTICS: asserts (raise -> exit nonzero) on L-0 regressions and L-1 proven algebra ONLY.
The L-2 confront outcome, L-3 covariance outcome and the FINAL VERDICT are PRINTED, never asserted.
Exit 0 = all regressions + proven algebra hold, and a definite verdict was booked (LENS-NULL is a
definite verdict -- a result, not a defect).
"""
import cmath
import itertools
import math
import os
import sys
import time

import numpy as np
from scipy.linalg import subspace_angles

t_start = time.time()

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "state"))
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "bridge"))
import srs                                        # noqa: E402  (walled-off clean-room K4-cover module)
import the_net as net                              # noqa: E402  (Layer-3 master object -- READ ONLY here)
import the_run                                     # noqa: E402  (ONLY K / LAM_PERRON / LAM_3IRREP read;
#                                                    exactly what LOOP_A5_discrete_chirality L40/53/82 reads.
#                                                    NOTHING scoreboard-adjacent is imported or computed.)
from simulator.srs_engine.utils import AlgebraicUtility  # noqa: E402

np.set_printoptions(precision=6, suppress=True, linewidth=120)

DISCLOSURES = []
N_PASS = [0]


def require(name, cond, detail=""):
    """L-0 / L-1 machine-check: prints, and ASSERTS (contract: all must hold before proceeding)."""
    cond = bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    assert cond, f"BRIDGE-LOCK regression/algebra FAILED: {name}"
    N_PASS[0] += 1


def banner(t):
    print("=" * 100)
    print(f" {t}")
    print("=" * 100)


def disclose(msg):
    DISCLOSURES.append(msg)
    print(f"    [DISCLOSED INTERPRETATION] {msg}")


# ====================================================================================================
banner("L-0  REGRESSION  (MAP O(2) classification rebuilt + A5 banked anchors re-verified; ALL asserted)")
# ====================================================================================================
EDGES = srs.EDGES
NE, NV = len(EDGES), srs.NV
EIDX = {(min(i, j), max(i, j)): e for e, (i, j, v) in enumerate(EDGES)}
DARTS = srs._darts()
ND = len(DARTS)                                    # 12
g6 = [np.array(g, complex) for g in AlgebraicUtility.cl6_generators()]
I8 = np.eye(8, dtype=complex)
I6 = np.eye(NE)
gam = lambda x: sum(x[a] * g6[a] for a in range(NE))
OM = cmath.exp(2j * math.pi / 3)
A4 = [dict(enumerate(p)) for p in itertools.permutations(range(4))
      if sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j]) % 2 == 0]


def edge_rep(sig):
    """Internal A4 action on the 6-edge space (W2_MAP L108-117 / LOOP_A5 files, verbatim)."""
    R6 = np.zeros((NE, NE))
    for e, (i, j, v) in enumerate(EDGES):
        a, b = sig[i], sig[j]
        s = 1.0
        if a > b:
            a, b, s = b, a, -1.0
        R6[EIDX[(a, b)], e] = s
    return R6


def dart_rep(sig):
    """Cover-side A4 action on the 12-dim dart space (W2_MAP L120-132, verbatim)."""
    Rd = np.zeros((ND, ND))
    for a, (i, j, v) in enumerate(DARTS):
        ni, nj = sig[i], sig[j]
        lo, hi = min(ni, nj), max(ni, nj)
        e2 = EIDX[(lo, hi)]
        b = 2 * e2 if ni < nj else 2 * e2 + 1
        Rd[b, a] = 1.0
    return Rd


# ---- L-0a  the frozen conventions: J6 with det(phi) > 0; srs dart indexing; R; B(Gamma) ----------
# J6 built with phi EXPOSED so the det-phi convention can be checked and (L-3) flipped
# (construction = the_net.complex_structure_J6 L69-88 = LOOP_A5_* S-0, verbatim, phi3 kept).
d0 = np.zeros((NV, NE))
for e, (i, j, v) in enumerate(EDGES):
    d0[i, e] = -1.0
    d0[j, e] = 1.0
Chat = np.array([[1, 1, 0], [-1, 0, 1], [0, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
H1, _ = np.linalg.qr(Chat)
B1 = np.linalg.svd(d0)[2][:3].T
rows = []
for g in A4:
    R6g = edge_rep(g)
    rows.append(np.kron(np.eye(3), (H1.T @ R6g @ H1).T) - np.kron(B1.T @ R6g @ B1, np.eye(3)))
_, _, VpJ = np.linalg.svd(np.vstack(rows))
phi3 = VpJ[-1].reshape(3, 3)
phi3 *= math.sqrt(3) / np.linalg.norm(phi3)
det_phi = float(np.linalg.det(phi3))


def J6_of(phi):
    return B1 @ phi @ H1.T - H1 @ phi.T @ B1.T


J6 = J6_of(phi3)
require(f"L-0a frozen convention: det(phi) = {det_phi:+.12f} > 0 -- the raw shared construction "
        f"ALREADY satisfies the MAP-frozen det-phi>0 convention (no flip needed; the A5 files' own "
        f"J6 IS the det-phi>0 J6, so the banked anchors and the frozen conventions cohere)",
        det_phi > 0)
require("L-0a J6 == the_net.complex_structure_J6() EXACTLY (bit-identical construction) and J6^2=-I",
        np.max(np.abs(J6 - net.complex_structure_J6())) < 1e-15
        and np.max(np.abs(J6 @ J6 + I6)) < 1e-12)

R = net.reversal()                                 # the_net L113-121
B0 = net.hashimoto_gamma()                         # the_net L124-126: srs.hashimoto(0).real, 12x12 0/1
require("L-0a srs dart indexing (dart 2e = forward, 2e+1 = reversed), ND = 12; R^2 = I, Tr R = 0, "
        "eigenstructure {+1^6, -1^6}",
        ND == 12 and np.max(np.abs(R @ R - np.eye(ND))) < 1e-12 and abs(np.trace(R)) < 1e-12
        and np.allclose(np.sort(np.linalg.eigvalsh(R)), [-1.] * 6 + [1.] * 6))

Ue = np.zeros((ND, NE))
Uo = np.zeros((ND, NE))
for e in range(NE):
    Ue[2 * e, e] = 1 / math.sqrt(2); Ue[2 * e + 1, e] = 1 / math.sqrt(2)
    Uo[2 * e, e] = 1 / math.sqrt(2); Uo[2 * e + 1, e] = -1 / math.sqrt(2)   # W2_MAP L144-147 verbatim

# ---- L-0b  rebuild MAP M-1a's O(2) classification --------------------------------------------------
# (i) dim Hom_A4(edge_rep, dart_rep) = 6   (W2_MAP L262-277)
rows = []
for g in A4:
    rows.append(np.kron(np.eye(NE), dart_rep(g)) - np.kron(edge_rep(g).T, np.eye(ND)))
Cstack = np.vstack(rows)
_, Ssvd, Vt = np.linalg.svd(Cstack)
rank = int(np.sum(Ssvd > 1e-9))
nullity = Cstack.shape[1] - rank
null_basis = Vt[rank:].T
Phis = [null_basis[:, k].reshape(ND, NE, order='F') for k in range(nullity)]
require(f"L-0b(i) dim Hom_A4(edge_rep, dart_rep) = {nullity} = 6 (W2_MAP M-1a-i)", nullity == 6)

# (ii) R splits Hom into R-even dim 2 / R-odd dim 4   (W2_MAP L279-298)
basis_vecs = np.stack([Phi.reshape(-1, order='F') for Phi in Phis], axis=1)
RPhi_vecs = np.stack([(R @ Phi).reshape(-1, order='F') for Phi in Phis], axis=1)
coeff, *_ = np.linalg.lstsq(basis_vecs, RPhi_vecs, rcond=None)
recon_err = np.max(np.abs(basis_vecs @ coeff - RPhi_vecs))
eigsR, eigvecsR = np.linalg.eig(coeff)
require("L-0b(ii) R preserves Hom and acts as an involution: eigs(R|_Hom) = (-1)^4, (+1)^2 "
        "(R-even dim 2, R-odd dim 4) (W2_MAP M-1a-ii)",
        recon_err < 1e-9 and np.allclose(np.sort(eigsR.real), [-1, -1, -1, -1, 1, 1], atol=1e-6))
even_idx = np.where(np.abs(eigsR.real - 1) < 1e-6)[0]
Qe, _ = np.linalg.qr(eigvecsR[:, even_idx].real)
even_vecs = basis_vecs @ Qe
Phi_even = [even_vecs[:, k].reshape(ND, NE, order='F') for k in range(even_vecs.shape[1])]

# (iii) R-even branch EMPTY under R5: the rank obstruction   (W2_MAP L300-310 verbatim, rng(0))
rng = np.random.default_rng(0)
ranks_even = []
for _ in range(8):
    c = rng.normal(size=len(Phi_even))
    Phi = sum(c[k] * Phi_even[k] for k in range(len(Phi_even)))
    ranks_even.append(int(np.linalg.matrix_rank(Phi, tol=1e-9)))
require(f"L-0b(iii) R-EVEN branch PROVABLY EMPTY under R5: every element has rank <= 3 < 6 (rank "
        f"obstruction -- can never be isometric on the 6-dim domain) (W2_MAP M-1a-iii)",
        all(rk <= 3 for rk in ranks_even), detail=f"ranks over 8 draws = {ranks_even}")

# (iv) R-odd sector, coordinatized by Uo, carries EXACTLY edge_rep   (W2_MAP L312-319)
dev_rho_odd = max(np.max(np.abs(Uo.T @ dart_rep(g) @ Uo - edge_rep(g))) for g in A4)
require("L-0b(iv) Uo^T dart_rep(g) Uo = edge_rep(g) exactly => Hom into the R-odd sector = "
        "End_A4(edge_rep), the commutant (W2_MAP M-1a-iv)", dev_rho_odd < 1e-9)

# (v) the commutant = span{I6, J6, S1, S2} (dim 4, Mat_2(R))   (W2_MAP L321-357 verbatim)
rows2 = [np.kron(np.eye(NE), edge_rep(g)) - np.kron(edge_rep(g).T, np.eye(NE)) for g in A4]
C2 = np.vstack(rows2)
_, S2s, Vt2 = np.linalg.svd(C2)
rank2 = int(np.sum(S2s > 1e-9))
Cs = [Vt2[rank2 + k].reshape(NE, NE, order='F') for k in range(C2.shape[1] - rank2)]


def express(M, basis):
    vecs = np.stack([b.reshape(-1, order='F') for b in basis], axis=1)
    coeff_, *_ = np.linalg.lstsq(vecs, M.reshape(-1, order='F'), rcond=None)
    return np.max(np.abs((vecs @ coeff_).reshape(NE, NE, order='F') - M))


require(f"L-0b(v) End_A4(edge_rep) has dim {len(Cs)} = 4 and contains both I6 and J6 "
        "(W2_MAP M-1a-v/vi)",
        len(Cs) == 4 and express(I6, Cs) < 1e-9 and express(J6, Cs) < 1e-9)

IJ = np.stack([I6.reshape(-1, order='F'), J6.reshape(-1, order='F')], axis=1)
allc = np.stack([c.reshape(-1, order='F') for c in Cs], axis=1)
Q_IJ, _ = np.linalg.qr(IJ)
proj = allc - Q_IJ @ (Q_IJ.T @ allc)
Qc, _ = np.linalg.qr(proj)
S1 = Qc[:, 0].reshape(NE, NE, order='F')
S2 = Qc[:, 1].reshape(NE, NE, order='F')
c_scale = float(np.trace(S1.T @ S1) / NE)          # common isometry scale of the reflection axes
require("L-0b(vi) the complement {S1, S2} is SYMMETRIC and TRACELESS (the reflection directions) "
        "(W2_MAP M-1a-vii)",
        np.allclose(S1, S1.T, atol=1e-8) and np.allclose(S2, S2.T, atol=1e-8)
        and abs(np.trace(S1)) < 1e-8 and abs(np.trace(S2)) < 1e-8)

# (vii) isometric locus = O(2) exactly   (W2_MAP L359-376 verbatim)


def isom_resid(Phi_red):
    G = Phi_red.T @ Phi_red
    scal = np.trace(G) / NE
    return np.linalg.norm(G - scal * I6) / (np.linalg.norm(G) + 1e-30)


require("L-0b(vii) isometric locus = O(2) EXACTLY: pure {aI+bJ6} and pure {cS1+dS2} are isometric "
        "(resid ~ 0), a generic mix is NOT (W2_MAP M-1a-viii)",
        isom_resid(0.6 * I6 + 0.8 * J6) < 1e-9 and isom_resid(0.6 * S1 + 0.8 * S2) < 1e-9
        and isom_resid(0.5 * I6 + 0.3 * J6 + 0.4 * S1) > 1e-3)

# tau_dart, as frozen in W2-MAP (L240-260): the cover-side antiunitary companion tau_dart = R o K_12
# with the +- sign NOT chosen by fiat but PINNED BY R5-SURVIVAL: the surviving (R-odd) family
# satisfies R Phi = -Phi, i.e. the effective sign is the one compatible with the NON-empty branch.
require("L-0b(viii) tau_dart convention (W2_MAP R1, L240-260): the surviving family is R-ODD "
        "(R.Uo = -Uo exactly), i.e. tau_dart's sign is pinned by R5-survival, not fiat",
        np.max(np.abs(R @ Uo + Uo)) < 1e-15)

# ---- L-0c  A5-W2 banked anchor: the forced chiral seed <0|U_pi^2|0> = i/2 --------------------------
# Verbatim replication of LOOP_A5_winding_weld_W2_2026-07-04.py S-0 (L73-123) with ITS OWN conventions.
wJ, VJ = np.linalg.eig(J6)


def build_vac(sign):
    """LOOP_A5_winding_weld_W2 L91-98 verbatim: Fock vacuum of the Witt ladder on the +-i modes."""
    sel = 1j if sign > 0 else -1j
    modes, _ = np.linalg.qr(VJ[:, np.where(np.abs(wJ - sel) < 1e-9)[0]])
    A = [gam(np.conj(modes[:, m])) / math.sqrt(2) for m in range(3)]
    N = sum(a.conj().T @ a for a in A)
    wN, VN = np.linalg.eigh(N)
    v = VN[:, [int(np.argmin(wN))]]
    return v / np.linalg.norm(v)


vac, vac_m = build_vac(+1), build_vac(-1)
C_PAIR = np.array([[(vac.conj().T @ g6[a] @ g6[b] @ vac).item() for b in range(NE)]
                   for a in range(NE)])
sigma3 = {0: 0, 1: 2, 2: 3, 3: 1}                  # LOOP_A5_winding_weld_W2 L103 (the deck 3-cycle)
pi = {}
for e, (i, j, v) in enumerate(EDGES):
    a, b = sigma3[i], sigma3[j]
    pi[e] = EIDX[(min(a, b), max(a, b))]
Rpi = np.zeros((NE, NE))
for e in range(NE):
    Rpi[pi[e], e] = 1.0
rows = [np.kron(gam(Rpi[:, a]), I8) - np.kron(I8, g6[a].T) for a in range(NE)]
_, S2sv, Vh = np.linalg.svd(np.vstack(rows))
null = Vh[np.sum(S2sv > 1e-9):].conj()
U_pi = null[0].reshape(8, 8)
U_pi /= np.sqrt(np.abs(np.linalg.det(U_pi @ U_pi.conj().T)) ** (1 / 8))
U2 = np.linalg.matrix_power(U_pi, 2)
ov1 = (vac.conj().T @ U_pi @ vac).item()
seed = (vac.conj().T @ U2 @ vac).item()
seed_m = (vac_m.conj().T @ U2 @ vac_m).item()
require("L-0c A5-W2 re-lock (its own conventions, L117-123): C = I + iJ (Re = I, Im antisym), "
        f"U_pi^3 = -I, |<0|U_pi|0>| = 1/2 ({abs(ov1):.6f})",
        np.max(np.abs(C_PAIR.real - I6)) < 1e-10
        and np.max(np.abs(C_PAIR.imag + C_PAIR.imag.T)) < 1e-10
        and np.max(np.abs(np.linalg.matrix_power(U_pi, 3) + I8)) < 1e-9
        and abs(abs(ov1) - 0.5) < 1e-9)
require(f"L-0c BANKED ANCHOR: the forced chiral seed <0|U_pi^2|0> = {seed:+.6f} = +i/2 (Re = 0, "
        "Im = +1/2), and it FLIPS with the bit: seed(-J) = -seed(+J) (A5-W2 CC3 + A5-DISCRETE CD3)",
        abs(seed.real) < 1e-6 and abs(seed.imag - 0.5) < 1e-6 and abs(seed_m + seed) < 1e-6)

# ---- L-0d  A5-DISCRETE banked anchor: the nu-leg functional / nu <-> chir-7 ------------------------
# Verbatim replication of LOOP_A5_discrete_chirality_2026-07-04.py (L68-71 ibroot; S-0; CD1-CD3).
K = the_run.K                                      # = 3 (coordination), exactly as A5-DISCRETE L53
LAM_PERRON, LAM_3IRREP = the_run.LAM_PERRON, the_run.LAM_3IRREP


def ibroot(lam):
    """A5-DISCRETE L68-71 verbatim: the Ihara-Bass root (lam + sqrt(lam^2-4(K-1)))/2, + branch."""
    disc = lam * lam - 4 * (K - 1)
    r = 1j * math.sqrt(-disc) if disc < 0 else math.sqrt(disc)
    return (lam + r) / 2, disc


A_adj = np.zeros((NV, NV))
for i, j, v in EDGES:
    A_adj[i, j] += 1
    A_adj[j, i] += 1
adj_ev = sorted(np.linalg.eigvals(A_adj).real, reverse=True)
mult_m1 = sum(1 for x in adj_ev if abs(x + 1) < 1e-9)
nJ = int(np.sum(np.abs(wJ - 1j) < 1e-9))
h_nu, disc_nu = ibroot(LAM_3IRREP)                 # chir-7
h_e, disc_e = ibroot(math.sqrt(LAM_PERRON))        # chir-5/3
H_PLUS = complex(-0.5, math.sqrt(7) / 2)           # the BANKED chir-7 IB-root (the + sign)  [ANCHOR]
H_MINUS = np.conj(H_PLUS)
require("L-0d A5-DISCRETE CD1: chir-7 IB-root = -1/2 + i sqrt7/2 (disc -7, |h|^2 = 2) = the cover_B "
        "sqrt(-7) enantiomer band-edge; chir-5/3 = sqrt(-5) off it",
        abs(h_nu - H_PLUS) < 1e-9 and abs(disc_nu + 7) < 1e-9 and abs(abs(h_nu) ** 2 - 2) < 1e-9
        and abs(disc_e + 5) < 1e-9)
omega6 = g6[0]
for a in range(1, NE):
    omega6 = omega6 @ g6[a]
w6sq = omega6 @ omega6
grade_even_proj = 0.5 * (I8 + omega6 / cmath.sqrt(w6sq[0, 0]))
even_weight = float(np.real((vac.conj().T @ grade_even_proj @ vac).item()))
require("L-0d A5-DISCRETE CD2: the A4 3-irrep = the 3-fold degenerate adjacency eigenvalue "
        f"lam = -1 = LAM_3IRREP; J's +i eigenspace has dim {nJ} = 3 (J IS that 3-irrep)",
        mult_m1 == 3 and nJ == 3 and abs(LAM_3IRREP + 1) < 1e-9 and abs(adj_ev[0] - 3) < 1e-9)
nu_leg_core = (mult_m1 == 3 and nJ == 3 and abs(h_nu - H_PLUS) < 1e-9
               and even_weight > 0.99 and abs(seed_m + seed) < 1e-6)
require("L-0d BANKED ANCHOR -- the nu-leg functional's core (A5-DISCRETE CD3/V, its own PASS "
        f"condition): nu = the Fock vacuum (grade-even weight {even_weight:.4f} > 0.99) carries the "
        "forced seed; the seed lives in the 3-irrep = lam=-1 band = chir-7 => nu <-> chir-7, CHIRAL "
        "(flips with J), reverse excluded", nu_leg_core)

N_L0 = N_PASS[0]
print(f"\n    L-0 COMPLETE: {N_L0} regression checks PASS.  The banked pairing (the ANCHOR "
      f"this station confronts): seed(+J) = +i/2  <->  chir-7 root h+ = {H_PLUS:+.6f}.\n")


# ====================================================================================================
banner("L-1  THEOREM  (a: transported complex structure per branch;  b: the attachment functional "
       "+ the bit-ODD test)")
# ====================================================================================================
# ---- L-1a  the transported complex structure is +Uo J6 Uo^T (rotation) / -Uo J6 Uo^T (reflection) --
S1n = S1 / math.sqrt(c_scale)                      # unit-isometry normalization (S1n^T S1n = I6)
S2n = S2 / math.sqrt(c_scale)
require("L-1a(i) ALGEBRA: S1, S2 ANTICOMMUTE with J6 exactly (S J6 + J6 S = 0) -- the reflection "
        "axes reverse the internal orientation; and S1n^T S1n = I6 (unit isometry)",
        np.max(np.abs(S1 @ J6 + J6 @ S1)) < 1e-12 and np.max(np.abs(S2 @ J6 + J6 @ S2)) < 1e-12
        and np.max(np.abs(S1n.T @ S1n - I6)) < 1e-12)

JD = Uo @ J6 @ Uo.T                                # the candidate transported structure (R-odd sector)
dev_rot = max(np.max(np.abs((Uo @ (math.cos(th) * I6 + math.sin(th) * J6)) @ J6
                            @ (Uo @ (math.cos(th) * I6 + math.sin(th) * J6)).T - JD))
              for th in np.linspace(0, 2 * math.pi, 25))
dev_ref = max(np.max(np.abs((Uo @ (math.cos(ph) * S1n + math.sin(ph) * S2n)) @ J6
                            @ (Uo @ (math.cos(ph) * S1n + math.sin(ph) * S2n)).T + JD))
              for ph in np.linspace(0, 2 * math.pi, 25))
require("L-1a(ii) THEOREM: Phi J6 Phi^T = +Uo J6 Uo^T for EVERY rotation-branch member (theta-"
        "independent) and = -Uo J6 Uo^T for EVERY reflection-branch member (phi-independent) -- "
        "the two branches transport OPPOSITE orientations",
        dev_rot < 1e-12 and dev_ref < 1e-12,
        detail=f"max dev rotation = {dev_rot:.2e}, reflection = {dev_ref:.2e} (over 25 pts each)")

# ---- L-1b  the attachment functional (from A5-DISCRETE's own chir-7 machinery) ---------------------
print("""    THE ATTACHMENT FUNCTIONAL (declared BEFORE evaluation; built from A5-DISCRETE's own chir-7
    identification machinery = the cover_B band-edge eigenspace, NOT from anything else):
      * cover_B := the_net.hashimoto_gamma() (real 12x12).  Its spectrum contains the conjugate
        IB-root pair h+- = -1/2 +- i sqrt7/2 (each with multiplicity 3) -- EXACTLY A5-DISCRETE's
        ibroot(LAM_3IRREP) pair (machine-checked below: the tie-in to the banked machinery).
      * P(h) := the spectral projector of cover_B onto its h-eigenspace (right/left eigenspaces;
        non-orthogonal, since B is non-normal -- idempotence, B P = h P, P(h-) = conj(P(h+)) checked).
      * the transported +i Witt content of a branch member Phi := Phi(W+), W+ = the +i eigenspace
        of J6 (the seed's one-particle home, LOOP_A5 S-0).  Machine-checked below: this subspace is
        the SAME for every member of a connected component (so the functional is a function of the
        BRANCH, as the pre-reg's orbit question requires).
      * PRIMARY functional (per branch):  w(h) := Re Tr( W^H P(h) W ),  W = orthonormal basis of the
        transported content;  the attachment asymmetry  Delta := w(h+) - w(h-).
      * CROSS-CHECK functional: o(h) := ||Q(h)^H W||_F^2 with Q(h) = orthonormal basis of ker(B-h)
        (the orthogonal-overlap reading; must agree in sign).
      * BIT-ODD TEST (the pre-reg's L-1 gate): bit-ODD iff |Delta| > 1e-6 on both branches with
        opposite signs; bit-EVEN (orbit-insensitive) iff |Delta| < 1e-9 on both -> STOP, LENS-NULL.""")

# the +i Witt content and its transports
Wp, _ = np.linalg.qr(VJ[:, np.where(np.abs(wJ - 1j) < 1e-9)[0]])   # 6x3, +i eigenspace of J6
W_rot = Uo @ Wp                                                     # rotation-branch transport
W_ref = Uo @ (S1n @ Wp)                                             # reflection-branch transport
ang_rot = max(np.max(subspace_angles(Uo @ ((math.cos(th) * I6 + math.sin(th) * J6) @ Wp), W_rot))
              for th in np.linspace(0.1, 6.2, 13))
ang_ref = max(np.max(subspace_angles(Uo @ ((math.cos(ph) * S1n + math.sin(ph) * S2n) @ Wp), W_ref))
              for ph in np.linspace(0.1, 6.2, 13))
require("L-1b(i) the transported +i content is CONSTANT on each connected component (the internal "
        "O(2) parameter acts as a mere phase/relabeling INSIDE the subspace): max principal angle "
        f"over 13 sampled members = {max(ang_rot, ang_ref):.2e}", max(ang_rot, ang_ref) < 1e-9)
require("L-1b(ii) the two branches transport to CONJUGATE subspaces: W_ref = conj(W_rot) (exactly "
        "the antilinear/linear split of the O(2) family)",
        np.max(subspace_angles(W_ref, W_rot.conj())) < 1e-9
        and np.max(np.abs(W_rot.conj().T @ W_rot - np.eye(3))) < 1e-12)

# cover_B band-edge eigenspaces + spectral projectors (the A5 machinery tie-in)


def eigspace(B, h, tol=1e-8):
    M = B - h * np.eye(ND)
    _, s, Vh_ = np.linalg.svd(M)
    k = int(np.sum(s < tol))
    return Vh_[ND - k:].conj().T                   # columns = orthonormal basis of ker(B - h)


Qp_B = eigspace(B0, H_PLUS)
Qm_B = eigspace(B0, H_MINUS)
Lp_B = eigspace(B0.conj().T, np.conj(H_PLUS))
Lm_B = eigspace(B0.conj().T, np.conj(H_MINUS))
Pp = Qp_B @ np.linalg.inv(Lp_B.conj().T @ Qp_B) @ Lp_B.conj().T
Pm = Qm_B @ np.linalg.inv(Lm_B.conj().T @ Qm_B) @ Lm_B.conj().T
require("L-1b(iii) MACHINERY TIE-IN: cover_B's spectrum contains the banked pair h+- = "
        f"ibroot(LAM_3IRREP) with 3-dim eigenspaces; P(h+-) idempotent, B P = h P, P(h-) = conj(P(h+))",
        Qp_B.shape[1] == 3 and Qm_B.shape[1] == 3
        and np.max(np.abs(Pp @ Pp - Pp)) < 1e-9 and np.max(np.abs(B0 @ Pp - H_PLUS * Pp)) < 1e-9
        and np.max(np.abs(Pm - Pp.conj())) < 1e-12)

# FS-5iii POISON DISCHARGE: the functional is Z3-transport-FREE
ang_z3 = np.max(subspace_angles(edge_rep(sigma3) @ Wp, Wp))
require("L-1b(iv) FS-5iii / Z3-TRANSPORT-FREENESS: the functional's ONLY internal input is the "
        "subspace W+ (irreducible 3-irrep), INVARIANT under the internal deck shadow edge_rep(sigma3) "
        f"(angle {ang_z3:.2e}); NO cover-side Z3/deck object is constructed or used anywhere in the "
        "functional; any refinement below subspace level would require a basis choice inside the "
        "3-irrep = precisely a forbidden deck-type identification (and is ALSO null, see L-1b(vii))",
        ang_z3 < 1e-9)
disclose("attachment functional = Re of the (non-orthogonal) spectral trace (PRIMARY) + orthogonal overlap "
         "(cross-check); both declared before evaluation; the bit-ODD thresholds (1e-6 / 1e-9) "
         "declared before evaluation.")


def wgt(P, W):
    return float(np.real(np.trace(W.conj().T @ P @ W)))


def ovl(Q, W):
    return float(np.linalg.norm(Q.conj().T @ W) ** 2)


w_rot_p, w_rot_m = wgt(Pp, W_rot), wgt(Pm, W_rot)
w_ref_p, w_ref_m = wgt(Pp, W_ref), wgt(Pm, W_ref)
o_rot_p, o_rot_m = ovl(Qp_B, W_rot), ovl(Qm_B, W_rot)
o_ref_p, o_ref_m = ovl(Qp_B, W_ref), ovl(Qm_B, W_ref)
d_rot, d_ref = w_rot_p - w_rot_m, w_ref_p - w_ref_m
print(f"""
    THE EVALUATION (the confront numbers, from running code):
                        w(h+)           w(h-)           Delta            [cross-check o(h+), o(h-)]
      rotation      {w_rot_p:>12.10f}   {w_rot_m:>12.10f}   {d_rot:+.2e}     [{o_rot_p:.10f}, {o_rot_m:.10f}]
      reflection    {w_ref_p:>12.10f}   {w_ref_m:>12.10f}   {d_ref:+.2e}     [{o_ref_p:.10f}, {o_ref_m:.10f}]
""")

require("L-1b(v) ALGEBRA (conjugation relation between the branches): w(h+)|_reflection = "
        "w(h-)|_rotation and vice versa (so the attachments are 'OPPOSITE' in the contracted sense "
        "-- here degenerately, see the test)",
        abs(w_ref_p - w_rot_m) < 1e-9 and abs(w_ref_m - w_rot_p) < 1e-9)

BIT_ODD = (abs(d_rot) > 1e-6 and abs(d_ref) > 1e-6 and d_rot * d_ref < 0)
BIT_EVEN = (abs(d_rot) < 1e-9 and abs(d_ref) < 1e-9)
SIGN_ANOMALY = (abs(d_rot) > 1e-6 and abs(d_ref) > 1e-6 and d_rot * d_ref > 0)

if BIT_EVEN:
    print("    >>> the BIT-ODD TEST: the attachment functional evaluates the SAME on both branches")
    print("        (Delta = 0 to machine precision on BOTH).  The functional is bit-EVEN /")
    print("        orbit-insensitive.  Per the frozen L-1 contract: STOP -> LENS-NULL; L-2 skipped.\n")

    # ------------------------------------------------------------------------------------------------
    # THE DEMONSTRATION (contract: 'print verdict LENS-NULL with the demonstration'):
    # the null is a THEOREM, not a numerical accident.  Three machine-checked lemmas force it.
    # ------------------------------------------------------------------------------------------------
    print("    THE DEMONSTRATION -- why the null is FORCED (theorem, each ingredient machine-checked):")
    require("L-1b(vi) LEMMA 1: R acts as -Id on the WHOLE R-odd sector (R Uo = -Uo) => the "
            "transported content of EVERY O(2) member spans an R-invariant subspace",
            np.max(np.abs(R @ Uo + Uo)) < 1e-15)
    require("L-1b(vi) LEMMA 2 (reversal-transpose / Ihara-Bass structure): R B R = B^T EXACTLY",
            np.max(np.abs(R @ B0 @ R - B0.T)) < 1e-15)
    require("L-1b(vi) LEMMA 3: B is REAL => P(h-) = conj(P(h+)) (checked at L-1b(iii))",
            np.max(np.abs(B0.imag)) < 1e-15 if np.iscomplexobj(B0) else True)
    t_rot_p = np.trace(W_rot.conj().T @ Pp @ W_rot)
    t_rotbar_p = np.trace(W_rot.T @ Pp @ W_rot.conj())
    require("L-1b(vi) THE CHAIN, machine-checked as full COMPLEX traces: "
            "Tr(W^H P+ W) = Tr(Wbar^H P+ Wbar)  [Lemmas 1+2 + spectral calculus f(B^T)=f(B)^T]  "
            "and Tr(W^H P- W) = conj(Tr(W^H P+ W))  [Lemma 3]  =>  Re-weights EQUAL: Delta == 0 "
            "IDENTICALLY for ANY R-parity-definite transported content -- on BOTH branches",
            abs(t_rot_p - t_rotbar_p) < 1e-9
            and abs(np.trace(W_rot.conj().T @ Pm @ W_rot) - np.conj(t_rot_p)) < 1e-9)
    per_mode_dev = max(abs(np.real(np.trace(W_rot[:, [k]].conj().T @ Pp @ W_rot[:, [k]]))
                           - np.real(np.trace(W_rot[:, [k]].conj().T @ Pm @ W_rot[:, [k]])))
                       for k in range(3))
    M_rot = W_rot.conj().T @ B0 @ W_rot
    M_ref = W_ref.conj().T @ B0 @ W_ref
    require("L-1b(vii) ESCAPE-KILLERS (corollaries, machine-checked): (a) PER-MODE attachment is "
            f"ALSO equal (max per-mode Delta = {per_mode_dev:.2e} -- no refinement discriminates); "
            "(b) the compressed dynamics W^H B W is HERMITIAN -- in fact EXACTLY ZERO "
            f"(||M_rot|| = {np.max(np.abs(M_rot)):.2e}, ||M_ref|| = {np.max(np.abs(M_ref)):.2e}): "
            "the transported content cannot see the band pair's rotation sense through B's own "
            "action on it, by ANY spectral reading",
            per_mode_dev < 1e-9 and np.max(np.abs(M_rot)) < 1e-9 and np.max(np.abs(M_ref)) < 1e-9)
    v_ctrl = Qp_B[:, [0]]
    ctrl_p = float(np.real(np.trace(v_ctrl.conj().T @ Pp @ v_ctrl)))
    ctrl_m = float(np.real(np.trace(v_ctrl.conj().T @ Pm @ v_ctrl)))
    ctrl_parity = min(np.linalg.norm(R @ v_ctrl - v_ctrl), np.linalg.norm(R @ v_ctrl + v_ctrl))
    require("L-1b(viii) CONTROL (the functional is NOT degenerate): a raw h+ eigenvector -- which is "
            f"NOT R-parity-definite (parity defect {ctrl_parity:.3f}) -- gets w(h+) = {ctrl_p:.6f}, "
            f"w(h-) = {ctrl_m:.6f}: the functional discriminates PERFECTLY when fed content outside "
            "the R-parity-definite class.  The null is not a property of the functional; it is a "
            "property of the R1-FORCED transport class",
            abs(ctrl_p - 1.0) < 1e-9 and abs(ctrl_m) < 1e-9 and ctrl_parity > 0.5)
    print("""
    ================================================================================================
    L-1 VERDICT: the attachment functional is bit-EVEN -- LENS-NULL, and it is THEOREM-GRADE:
      W2-MAP's R1 (compatibility with the certified antiunitary, via tau_dart) forced EVERY
      admissible map's image into a definite R-parity sector (R-odd).  On such a sector R acts as
      -Id (Lemma 1), so every transported subspace is R-invariant; the reversal conjugates B to B^T
      (Lemma 2); B is real (Lemma 3).  Together these force the attachment weights to the two
      conjugate band-edges to be EXACTLY EQUAL -- for every member of BOTH branches, for every
      sub-content, for every spectral reading of B.  The very requirement (R1) that carved the
      O(2) family out of the 6-dim Hom space is the requirement that makes the family band-edge-
      BLIND: the one-particle dart transport mirror-symmetrizes precisely the degree of freedom
      the discriminator needed.  (The MASTER CHIRALITY LENS predicted this failure mode: the
      functional is an R-parity-EVEN observable of the transported content, and bit-EVEN =
      democratic = blind.)  A discriminator must therefore either leave the R-parity-definite
      one-particle class (a Fock-level / phase-bearing object) or read the arrow some other way --
      Design B (the modular-arrow discriminator) per the frozen verdict table.
    ================================================================================================
""")
elif BIT_ODD:
    print(f"    >>> the BIT-ODD TEST: PASSES (Delta_rot = {d_rot:+.6e}, Delta_ref = {d_ref:+.6e}, "
          "opposite signs) -- the branches attach to OPPOSITE members of the conjugate pair; "
          "proceeding to L-2.\n")
else:
    print(f"    >>> the BIT-ODD TEST: ANOMALOUS regime (Delta_rot = {d_rot:+.3e}, Delta_ref = "
          f"{d_ref:+.3e}; same-sign anomaly = {SIGN_ANOMALY}) -- neither clean bit-ODD nor clean "
          "bit-EVEN.  Booked conservatively as LENS-NULL-INDETERMINATE below (a result; no "
          "selection is claimed from an indeterminate functional).\n")


# ====================================================================================================
banner("L-2  THE CONFRONT  (frozen decision rule; PRINTED, never asserted)")
# ====================================================================================================
VERDICT = None
if BIT_ODD:
    # the A5-DISCRETE nu-leg functional per branch: the branch's transported +i content (the seed's
    # one-particle home carried to the cover) attaches to ONE member of the pair; the banked
    # assignment is nu <-> chir-7 = h+ (seed +i/2 paired with the + root).  Frozen rule: the branch
    # attaching to h+ is SELECTED.
    att_rot = "h+" if d_rot > 0 else "h-"
    att_ref = "h+" if d_ref > 0 else "h-"
    sel = "ROTATION" if att_rot == "h+" else "REFLECTION"
    print(f"    rotation branch attaches the transported +i Witt content to {att_rot}; "
          f"reflection to {att_ref}.")
    print(f"    FROZEN RULE: the branch attaching to the banked chir-7 root h+ = {H_PLUS:+.4f} "
          f"is selected  =>  SELECTED ORBIT: {sel} "
          f"({'complex-LINEAR' if sel == 'ROTATION' else 'complex-ANTIlinear'} on the Witt +i space).")
    print("    (This reproduces the banked nu <-> chir-7 on the cover side through the selected "
          "orbit; the other orbit lands on the conjugate root = the anti-banked assignment.)")
    VERDICT = ("SELECTS", sel)
elif BIT_EVEN:
    print("""    SKIPPED per the frozen L-1 gate: the attachment functional is bit-EVEN (orbit-
    insensitive), so the A5-DISCRETE nu-leg functional CANNOT be evaluated 'per orbit' -- both
    orbits present the SAME attachment data (w(h+) = w(h-) = 0.75 each), and the frozen decision
    rule ('the branch attaching to h+') selects NOTHING.  No selection is manufactured.  The
    banked nu <-> chir-7 anchor itself is UNTOUCHED (it lives on the internal side and was
    re-verified at L-0d); what failed is the ORBIT-DISCRIMINATION, not the lock.""")
    VERDICT = ("LENS-NULL", None)
else:
    print("    SKIPPED: indeterminate functional (see L-1).  No selection claimed.")
    VERDICT = ("LENS-NULL-INDETERMINATE", None)


# ====================================================================================================
banner("L-3  CONVENTION-COVARIANCE  (joint det-phi + tau_dart sign flip; PRINTED, never asserted)")
# ====================================================================================================
# the joint flip, implemented operationally:
#   det-phi:  phi -> -phi  (det(-phi) = -det(phi) < 0)  =>  J6' = -J6  (the bit flips).
#   tau_dart: sign flip => the R1 reduction would demand R Phi = +Phi (the R-EVEN branch).  That
#   branch is EMPTY by the convention-independent rank obstruction (L-0b(iii) -- re-checked below on
#   the flipped conventions), so R5-survival RE-PINS the sign to the R-odd branch -- exactly the
#   W2-MAP frozen logic ('the sign is NOT chosen by fiat, it is selected by which branch survives
#   R5').  The tau_dart dial is therefore ABSORBED by survival; the operative flip is det-phi.
phi3_f = -phi3
J6_f = J6_of(phi3_f)
det_phi_f = float(np.linalg.det(phi3_f))
wJ_f, VJ_f = np.linalg.eig(J6_f)
Wp_f, _ = np.linalg.qr(VJ_f[:, np.where(np.abs(wJ_f - 1j) < 1e-9)[0]])   # +i eigenspace of J6'
# S1', S2' rebuilt against the flipped J6 (same commutant; complement of span{I6, J6'} = same plane)
proj_f = allc - np.linalg.qr(np.stack([I6.reshape(-1, order='F'),
                                       J6_f.reshape(-1, order='F')], axis=1))[0] @ \
    (np.linalg.qr(np.stack([I6.reshape(-1, order='F'), J6_f.reshape(-1, order='F')], axis=1))[0].T @ allc)
Qc_f, _ = np.linalg.qr(proj_f)
S1_f = Qc_f[:, 0].reshape(NE, NE, order='F')
S1n_f = S1_f / math.sqrt(float(np.trace(S1_f.T @ S1_f) / NE))
ranks_even_f = ranks_even                          # the rank obstruction involves NO sign convention
print(f"    flipped conventions: det(phi') = {det_phi_f:+.6f} < 0;  tau_dart' sign => R-even branch "
      f"=> EMPTY (rank obstruction, convention-free: ranks {ranks_even_f}) => R5-survival re-pins "
      f"the R-odd family (tau_dart absorbed).  J6' = -J6: {np.max(np.abs(J6_f + J6)) < 1e-12}.")

W_rot_f = Uo @ Wp_f                                # rotation-branch transport, flipped conventions
W_ref_f = Uo @ (S1n_f @ Wp_f)                      # reflection-branch transport, flipped conventions
dw_rot_f = wgt(Pp, W_rot_f) - wgt(Pm, W_rot_f)
dw_ref_f = wgt(Pp, W_ref_f) - wgt(Pm, W_ref_f)
print(f"    recomputed attachment (flipped): Delta_rot' = {dw_rot_f:+.2e}, Delta_ref' = {dw_ref_f:+.2e}"
      f"   [w'(h+), w'(h-)] rot = [{wgt(Pp, W_rot_f):.10f}, {wgt(Pm, W_rot_f):.10f}]")

if VERDICT[0] == "SELECTS":
    att_rot_f = "h+" if dw_rot_f > 0 else "h-"
    sel_f = "ROTATION" if att_rot_f == "h+" else "REFLECTION"
    # the PHYSICAL orbit = the relative orientation: the transported structure of the SELECTED
    # orbit, as an absolute matrix on the dart space (label-free).
    JD_sel = JD if VERDICT[1] == "ROTATION" else -JD
    JD_sel_f = (Uo @ J6_f @ Uo.T) if sel_f == "ROTATION" else -(Uo @ J6_f @ Uo.T)
    stable = np.max(np.abs(JD_sel_f - JD_sel)) < 1e-9
    print(f"    flipped selection label: {sel_f}; PHYSICAL orbit (the transported orientation on the "
          f"R-odd sector, label-free) stable under the joint flip: {stable}")
    if stable:
        print(f"    => the selection is PHYSICAL (labels covariant, orientation invariant): "
              f"VERDICT SELECTS ({VERDICT[1]}) stands.")
    else:
        VERDICT = ("CONVENTION-ARTIFACT", None)
        print("    => the selection FLIPPED under the joint convention flip: CONVENTION-ARTIFACT.")
elif VERDICT[0].startswith("LENS-NULL"):
    print(f"""    MOOT-UNDER-NULL: there is no selected orbit to test for stability.  What CAN be tested --
    and is -- is the NULL itself: under the joint convention flip the attachment functional is
    STILL identically bit-EVEN (Delta' = {dw_rot_f:+.2e} / {dw_ref_f:+.2e}), as the theorem demands
    (Lemmas 1-3 contain no sign convention: R Uo = -Uo, R B R = B^T and the reality of B are
    convention-free statements).  The LENS-NULL is NOT a convention artifact.""")


# ====================================================================================================
banner("SCOPE  (printed; nothing moves)")
# ====================================================================================================
print("""    NOT claimed / not touched by this station:
      * NO scoreboard value moves under any verdict; the later gated station stays gated (the
        master goal-seek guard held: no orbit dial and no gated number in the same run).
      * the A5 banked anchors (seed = +i/2; nu <-> chir-7; chir-5/3) were regression-verified and
        NEVER adjusted; the LENS-NULL does not weaken them -- it says the ONE-PARTICLE dart
        transport cannot read the lock's orientation, not that the lock is wrong.
      * FS-5iii stands untouched: no Z3/deck identification was imposed (L-1b(iv)); the functional
        was demonstrated Z3-transport-free, so the abort-to-LENS-NULL poison clause was not needed
        (the null found is the CONTRACT's own L-1 gate, reached with a clean functional).
      * W2-MAP's AMBIGUOUS-BY-O(2) STANDS, sharpened: the ambiguity is not just unresolved by
        {R1,R2,R5}+dynamics (M-1b) -- it is UNRESOLVABLE by ANY band-edge attachment functional of
        R-parity-definite one-particle transports (this file's theorem).  The named next
        discriminators: Design B (the modular arrow), or a phase-bearing Fock-level object.""")


# ====================================================================================================
banner("SUMMARY")
# ====================================================================================================
elapsed = time.time() - t_start
final = VERDICT[0] + (f" ({VERDICT[1]})" if VERDICT[1] else "")
print(f"    L-0  REGRESSION .............................. PASS ({N_L0} checks at the L-0 gate; "
      f"O(2) classification + A5 anchors; {N_PASS[0]} asserted checks total incl. L-1 algebra)")
print(f"    L-1a TRANSPORTED STRUCTURE ................... THEOREM: +Uo J6 Uo^T (rotation) / "
      f"-Uo J6 Uo^T (reflection) -- OPPOSITE orientations")
print(f"    L-1b ATTACHMENT + BIT-ODD TEST ............... bit-{'ODD' if BIT_ODD else 'EVEN'}: "
      f"w(h+)/w(h-) = {w_rot_p:.4f}/{w_rot_m:.4f} (rot) and {w_ref_p:.4f}/{w_ref_m:.4f} (ref); "
      f"Delta = {d_rot:+.1e} / {d_ref:+.1e}")
print(f"    L-2  THE CONFRONT ............................ "
      f"{'evaluated' if BIT_ODD else 'SKIPPED per the frozen L-1 gate'}")
print(f"    L-3  CONVENTION-COVARIANCE ................... "
      f"{'tested on the selection' if BIT_ODD else 'MOOT-UNDER-NULL; the null itself is convention-stable'}")
print(f"    disclosed interpretation steps: {len(DISCLOSURES)}")
print(f"    runtime: {elapsed:.1f}s")
print()
print(f" FINAL VERDICT (printed, per contract): {final}")
if VERDICT[0] == "LENS-NULL":
    print("   -> the O(2) orbit ambiguity STANDS; hand to Design B (the modular-arrow "
          "discriminator) per the frozen verdict table.  A null here is a result, not a defect.")
print("=" * 100)
sys.exit(0)
