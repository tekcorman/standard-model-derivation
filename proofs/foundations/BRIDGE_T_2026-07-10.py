#!/usr/bin/env python3
"""
proofs/foundations/BRIDGE_T_2026-07-10.py

BRIDGE-T -- the modular-arrow orbit discriminator (Design B of the chirality-bridge dossier;
milestone II.2).  Pre-registered in internal research notes (contracts T-0..T-4,
the frozen decision rule, the verdicts, the poisons -- ALL frozen BEFORE this file was written).

LINEAGE: W2-MAP (AMBIGUOUS-BY-O(2): rotation branch Phi_theta = Uo(cos th I6 + sin th J6),
complex-LINEAR; reflection branch Phi_phi = Uo(cos ph S1 + sin ph S2), complex-ANTIlinear =
rotation o sigma) -> BRIDGE-LOCK (route A: LENS-NULL, THEOREM-GRADE via the three lemmas:
(i) R = -Id on the R-odd dart sector; (ii) R.B.R = B^T; (iii) B real => conjugate band-edge
projectors).  This station is route B: sigma implements the ANTIUNITARY FLOW-REVERSING half of the
modular conjugation EXACTLY (M0-4b: C(-J) = I - C(J); K_A -> -K_A), so the two orbits pull the
cover structure back to modular flows of OPPOSITE direction; the framework possesses a DERIVED
arrow (M0-2R: sub-criticality u < u_c = 1/(k-1) = 1/2, operating point u = alpha_1 = (2/3)^8).
THE QUESTION: does a cover-side ARROW datum select the orbit?

====================================================================================================
THE FROZEN DATUM  (T-2's disclosed-interpretation declaration -- DECLARED HERE, BEFORE EVALUATION)
====================================================================================================
Pre-reg candidate (frozen): "the orientation sign carried by the sub-critical stationary run state
on the R-odd dart sector": build |G> = sum_{n=0}^{N} u^n B^n |seed> (u = alpha_1, converged in n),
form the stationary two-point object M_ij = <G| P_odd (|d_i><d_j|) P_odd |G> -- OR the equivalent
R-odd-sector restriction of the run's dart-dart correlation matrix -- and the datum is
    w(orbit) := < A , J_D(orbit) >_F ,   J_D(rotation) = +Uo J6 Uo^T,  J_D(reflection) = -Uo J6 Uo^T
with A the antisymmetrized correlation and <M,N>_F = sum_ij M_ij N_ij.

DISCLOSED INTERPRETATION (the pre-reg's own "or equivalent" clause, exercised and declared BEFORE
any number is computed):
  * READING (i), the LITERAL equal-time object: M1 = (P_odd G)(P_odd G)^T.  B and |seed> are REAL,
    so |G> is a REAL vector and M1 is a symmetric rank-1 outer product: antisym(M1) == 0 at MATRIX
    level, for the trivial reality reason -- an equal-time real two-point object carries no
    orientation AT ALL.  Reading (i) is therefore degenerately null BEFORE any orbit question and
    cannot be the operative form of the frozen candidate.  It is still computed and shown.
  * READING (ii), PRIMARY: the run's dart-dart CORRELATION matrix = the seed->run propagation
    (lag) correlation, the object that carries the run's time-order:
        M  :=  P_odd [ sum_{n>=0} u^n B^n |seed><seed| ] P_odd  =  (P_odd |G>)(P_odd |seed>)^T ,
        A  :=  (M - M^T)/2 .
    This is the unique member of the frozen family whose antisymmetric part is not already killed
    at matrix level by reality; its antisymmetric part is the run's stationary CURRENT-like object
    (nonzero as a matrix -- shown below).
  * READING (iii), cross-check: the seed-free restriction P_odd (I - uB)^{-1} P_odd (the path-gas
    dart-dart Green function on the R-odd sector), antisymmetrized.  (Not seed-anchored by
    construction; computed for the demonstration only, never the primary.)
  * SEED (declared): |seed> = dart 0 (the forward dart of edge 0 in the frozen srs indexing,
    dart 2e = forward / 2e+1 = reversed).  A4 acts simply transitively on the 12 darts (W2-MAP
    M-0l), so every dart seed is checked below and the choice is demonstrated irrelevant.
  * THRESHOLDS (declared): bit-ODD iff |w_rot| > 1e-6 AND |w_ref| > 1e-6 with opposite signs;
    bit-EVEN (machine-zero) iff |w_rot| < 1e-9 AND |w_ref| < 1e-9; state-level R-parity
    indefiniteness requires min(||RG - G||, ||RG + G||) > 0.1.
  * FROZEN SELECTION RULE (T-3): the selected orbit := the one whose transported flow orientation
    MATCHES the run's arrow (forward = the growth direction of u^n B^n, n increasing), read as
    w(orbit) > 0 (positive Frobenius pairing of the forward-run antisymmetrized correlation with
    that orbit's transported complex structure).  No other criterion.
  * FROZEN DECISION TREE: T-2 gate (a),(b),(c); if (c) evaluates bit-EVEN -> STOP, print ARROW-BLIND
    with the demonstration; T-3 selection only on a bit-ODD datum; T-4 joint convention flip must
    leave the PHYSICAL selection stable AND the reversed run (B -> B^T) must FLIP the selection,
    else CONVENTION-ARTIFACT.

VERDICTS (all bookable): ARROW-SELECTS / ARROW-BLIND / NULL-ESCAPE-FAIL / CONVENTION-ARTIFACT.

POISONS (binding, restated): no oblique/M_Z/EW quantity computed or mentioned anywhere; no
linear-in-B datum (the datum is the full resolvent series -- its order-by-order content is
additionally confronted below); the three BRIDGE-LOCK lemmas confronted BEFORE evaluation (T-0);
the A5 anchors and the arrow (u < u_c) are regression objects, never adjusted; FS-5iii binding (no
Z3/deck identification imposed anywhere -- the datum's only internal input is J6 itself, deck-free);
numbers only from running code; ONE new proofs/ file (this one); no existing file touched.

EXIT SEMANTICS: asserts (raise -> exit nonzero) on T-0 regressions, T-1 proven algebra, and the
machine-checked steps of whichever demonstration branch is reached.  The T-2 gate OUTCOME, the T-3
selection, the T-4 covariance outcome and the FINAL VERDICT are PRINTED, never asserted.  Exit 0 =
all regressions + algebra hold and a definite verdict was booked (ARROW-BLIND is a definite
verdict -- a result, not a defect).
"""
import itertools
import math
import os
import sys
import time

import numpy as np

t_start = time.time()

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "state"))
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "bridge"))
import srs                                        # noqa: E402  (walled-off clean-room K4-cover module)
import the_net as net                              # noqa: E402  (Layer-3 master object -- READ ONLY here)
import the_run                                     # noqa: E402  (ONLY K / GIRTH / U_RUN / LAM_3IRREP read --
#                                                    the arrow objects + the Ihara-Bass root input.
#                                                    NOTHING scoreboard-adjacent is imported or computed.)

np.set_printoptions(precision=6, suppress=True, linewidth=120)

DISCLOSURES = []
N_PASS = [0]

# frozen thresholds (docstring-declared; bound here once, before any evaluation)
TOL_ODD = 1e-6
TOL_ZERO = 1e-9
TOL_PARITY = 0.1


def require(name, cond, detail=""):
    """T-0 / T-1 / demonstration machine-check: prints, and ASSERTS."""
    cond = bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    assert cond, f"BRIDGE-T regression/algebra FAILED: {name}"
    N_PASS[0] += 1


def banner(t):
    print("=" * 100)
    print(f" {t}")
    print("=" * 100)


def disclose(msg):
    DISCLOSURES.append(msg)
    print(f"    [DISCLOSED INTERPRETATION] {msg}")


def frob(M, N):
    """The declared pairing <M,N>_F = sum_ij M_ij N_ij."""
    return float(np.sum(M * N))


# ====================================================================================================
banner("T-0  REGRESSION  (O(2) family + BRIDGE-LOCK's three lemmas + M0 C(-J)=I-C(J) + the arrow; "
       "ALL asserted BEFORE anything new runs)")
# ====================================================================================================
EDGES = srs.EDGES
NE, NV = len(EDGES), srs.NV
EIDX = {(min(i, j), max(i, j)): e for e, (i, j, v) in enumerate(EDGES)}
DARTS = srs._darts()
ND = len(DARTS)                                    # 12
I6 = np.eye(NE)
A4 = [dict(enumerate(p)) for p in itertools.permutations(range(4))
      if sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j]) % 2 == 0]


def edge_rep(sig):
    """Internal A4 action on the 6-edge space (W2_MAP L108-117, verbatim)."""
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


# ---- T-0a  frozen conventions: J6 (phi exposed for the T-4 flip), R, B, Ue/Uo, P_odd -------------
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
require(f"T-0a frozen convention det(phi) = {det_phi:+.12f} > 0; J6 == the_net.complex_structure_J6() "
        "EXACTLY; J6^2 = -I",
        det_phi > 0 and np.max(np.abs(J6 - net.complex_structure_J6())) < 1e-15
        and np.max(np.abs(J6 @ J6 + I6)) < 1e-12)

R = net.reversal()
B0 = net.hashimoto_gamma()                         # srs.hashimoto(0).real, 12x12 real 0/1
Ue = np.zeros((ND, NE))
Uo = np.zeros((ND, NE))
for e in range(NE):
    Ue[2 * e, e] = 1 / math.sqrt(2); Ue[2 * e + 1, e] = 1 / math.sqrt(2)
    Uo[2 * e, e] = 1 / math.sqrt(2); Uo[2 * e + 1, e] = -1 / math.sqrt(2)   # W2_MAP L144-147 verbatim
P_odd = (np.eye(ND) - R) / 2
require("T-0a srs dart indexing (2e fwd / 2e+1 rev); R^2 = I, Tr R = 0, eigs {+1^6,-1^6}; "
        "P_odd = (I-R)/2 = Uo Uo^T (the R-odd projector, Uo-coordinatized)",
        ND == 12 and np.max(np.abs(R @ R - np.eye(ND))) < 1e-12 and abs(np.trace(R)) < 1e-12
        and np.allclose(np.sort(np.linalg.eigvalsh(R)), [-1.] * 6 + [1.] * 6)
        and np.max(np.abs(Uo @ Uo.T - P_odd)) < 1e-12)

# ---- T-0b  the O(2) family classification rebuilt (W2-MAP M-1a / BRIDGE-LOCK L-0b, verbatim) -----
rows = []
for g in A4:
    rows.append(np.kron(np.eye(NE), dart_rep(g)) - np.kron(edge_rep(g).T, np.eye(ND)))
Cstack = np.vstack(rows)
_, Ssvd, Vt = np.linalg.svd(Cstack)
rank = int(np.sum(Ssvd > 1e-9))
nullity = Cstack.shape[1] - rank
null_basis = Vt[rank:].T
Phis = [null_basis[:, k].reshape(ND, NE, order='F') for k in range(nullity)]
require(f"T-0b(i) dim Hom_A4(edge_rep, dart_rep) = {nullity} = 6", nullity == 6)

basis_vecs = np.stack([Phi.reshape(-1, order='F') for Phi in Phis], axis=1)
RPhi_vecs = np.stack([(R @ Phi).reshape(-1, order='F') for Phi in Phis], axis=1)
coeff, *_ = np.linalg.lstsq(basis_vecs, RPhi_vecs, rcond=None)
recon_err = np.max(np.abs(basis_vecs @ coeff - RPhi_vecs))
eigsR, eigvecsR = np.linalg.eig(coeff)
require("T-0b(ii) R preserves Hom, involution: eigs (-1)^4,(+1)^2 (R-even dim 2 / R-odd dim 4)",
        recon_err < 1e-9 and np.allclose(np.sort(eigsR.real), [-1, -1, -1, -1, 1, 1], atol=1e-6))
even_idx = np.where(np.abs(eigsR.real - 1) < 1e-6)[0]
Qe, _ = np.linalg.qr(eigvecsR[:, even_idx].real)
even_vecs = basis_vecs @ Qe
Phi_even = [even_vecs[:, k].reshape(ND, NE, order='F') for k in range(even_vecs.shape[1])]
rng = np.random.default_rng(0)
ranks_even = []
for _ in range(8):
    c = rng.normal(size=len(Phi_even))
    Phi = sum(c[k] * Phi_even[k] for k in range(len(Phi_even)))
    ranks_even.append(int(np.linalg.matrix_rank(Phi, tol=1e-9)))
require("T-0b(iii) R-EVEN branch PROVABLY EMPTY under R5 (rank <= 3 obstruction)",
        all(rk <= 3 for rk in ranks_even), detail=f"ranks over 8 draws = {ranks_even}")
dev_rho_odd = max(np.max(np.abs(Uo.T @ dart_rep(g) @ Uo - edge_rep(g))) for g in A4)
require("T-0b(iv) Uo^T dart_rep(g) Uo = edge_rep(g) exactly (R-odd sector carries edge_rep)",
        dev_rho_odd < 1e-9)

rows2 = [np.kron(np.eye(NE), edge_rep(g)) - np.kron(edge_rep(g).T, np.eye(NE)) for g in A4]
C2 = np.vstack(rows2)
_, S2s, Vt2 = np.linalg.svd(C2)
rank2 = int(np.sum(S2s > 1e-9))
Cs = [Vt2[rank2 + k].reshape(NE, NE, order='F') for k in range(C2.shape[1] - rank2)]


def express(M, basis):
    vecs = np.stack([b.reshape(-1, order='F') for b in basis], axis=1)
    coeff_, *_ = np.linalg.lstsq(vecs, M.reshape(-1, order='F'), rcond=None)
    return np.max(np.abs((vecs @ coeff_).reshape(NE, NE, order='F') - M))


require(f"T-0b(v) End_A4(edge_rep) dim {len(Cs)} = 4 (Mat_2(R)), contains I6 and J6",
        len(Cs) == 4 and express(I6, Cs) < 1e-9 and express(J6, Cs) < 1e-9)
IJ = np.stack([I6.reshape(-1, order='F'), J6.reshape(-1, order='F')], axis=1)
allc = np.stack([c.reshape(-1, order='F') for c in Cs], axis=1)
Q_IJ, _ = np.linalg.qr(IJ)
proj = allc - Q_IJ @ (Q_IJ.T @ allc)
Qc, _ = np.linalg.qr(proj)
S1 = Qc[:, 0].reshape(NE, NE, order='F')
S2 = Qc[:, 1].reshape(NE, NE, order='F')
c_scale = float(np.trace(S1.T @ S1) / NE)
S1n = S1 / math.sqrt(c_scale)
S2n = S2 / math.sqrt(c_scale)
require("T-0b(vi) complement {S1,S2} SYMMETRIC + TRACELESS (reflection axes); both ANTICOMMUTE "
        "with J6; S1n isometric",
        np.allclose(S1, S1.T, atol=1e-8) and np.allclose(S2, S2.T, atol=1e-8)
        and abs(np.trace(S1)) < 1e-8 and abs(np.trace(S2)) < 1e-8
        and np.max(np.abs(S1 @ J6 + J6 @ S1)) < 1e-12 and np.max(np.abs(S2 @ J6 + J6 @ S2)) < 1e-12
        and np.max(np.abs(S1n.T @ S1n - I6)) < 1e-12)


def isom_resid(Phi_red):
    G = Phi_red.T @ Phi_red
    scal = np.trace(G) / NE
    return np.linalg.norm(G - scal * I6) / (np.linalg.norm(G) + 1e-30)


require("T-0b(vii) isometric locus = O(2) EXACTLY (pure rotation / pure reflection isometric; "
        "generic mix NOT)",
        isom_resid(0.6 * I6 + 0.8 * J6) < 1e-9 and isom_resid(0.6 * S1 + 0.8 * S2) < 1e-9
        and isom_resid(0.5 * I6 + 0.3 * J6 + 0.4 * S1) > 1e-3)

# ---- T-0c  BRIDGE-LOCK's THREE NULL LEMMAS (the baseline this station must escape) ---------------
require("T-0c LEMMA 1: R acts as -Id on the WHOLE R-odd sector (R Uo = -Uo; tau_dart sign pinned "
        "by R5-survival, not fiat)", np.max(np.abs(R @ Uo + Uo)) < 1e-15)
require("T-0c LEMMA 2 (reversal-transpose / Ihara-Bass): R B R = B^T EXACTLY",
        np.max(np.abs(R @ B0 @ R - B0.T)) < 1e-15)
K = the_run.K                                      # = 3, exactly as BRIDGE-LOCK L-0d reads it
LAM_3IRREP = the_run.LAM_3IRREP                    # = -1


def ibroot(lam):
    """A5-DISCRETE L68-71 verbatim: (lam + sqrt(lam^2 - 4(K-1)))/2, + branch."""
    disc = lam * lam - 4 * (K - 1)
    r = 1j * math.sqrt(-disc) if disc < 0 else math.sqrt(disc)
    return (lam + r) / 2, disc


H_PLUS, disc_nu = ibroot(LAM_3IRREP)               # the banked chir-7 IB-root -1/2 + i sqrt7/2
H_MINUS = np.conj(H_PLUS)


def eigspace(B, h, tol=1e-8):
    M = B - h * np.eye(ND)
    _, s, Vh_ = np.linalg.svd(M)
    k = int(np.sum(s < tol))
    return Vh_[ND - k:].conj().T


Qp_B = eigspace(B0, H_PLUS)
Qm_B = eigspace(B0, H_MINUS)
Lp_B = eigspace(B0.conj().T if np.iscomplexobj(B0) else B0.T, np.conj(H_PLUS))
Lm_B = eigspace(B0.conj().T if np.iscomplexobj(B0) else B0.T, np.conj(H_MINUS))
Pp = Qp_B @ np.linalg.inv(Lp_B.conj().T @ Qp_B) @ Lp_B.conj().T
Pm = Qm_B @ np.linalg.inv(Lm_B.conj().T @ Qm_B) @ Lm_B.conj().T
require("T-0c LEMMA 3: B is REAL => conjugate band-edge projectors P(h-) = conj(P(h+)) "
        f"(h+- = ibroot(LAM_3IRREP) = {H_PLUS:+.4f}/conj, 3-dim each; idempotent; B P = h P)",
        (not np.iscomplexobj(B0) or np.max(np.abs(B0.imag)) < 1e-15)
        and Qp_B.shape[1] == 3 and Qm_B.shape[1] == 3
        and np.max(np.abs(Pp @ Pp - Pp)) < 1e-9 and np.max(np.abs(B0 @ Pp - H_PLUS * Pp)) < 1e-9
        and np.max(np.abs(Pm - Pp.conj())) < 1e-12)

# the null BASELINE those lemmas force (BRIDGE-LOCK's booked LENS-NULL, rebuilt as the assert this
# station's datum must ESCAPE): every R-parity-definite one-particle attachment is bit-EVEN.
wJ, VJ = np.linalg.eig(J6)
Wp, _ = np.linalg.qr(VJ[:, np.where(np.abs(wJ - 1j) < 1e-9)[0]])   # +i eigenspace of J6 (6x3)
W_rot = Uo @ Wp
W_ref = Uo @ (S1n @ Wp)


def wgt(P, W):
    return float(np.real(np.trace(W.conj().T @ P @ W)))


d_rot_base = wgt(Pp, W_rot) - wgt(Pm, W_rot)
d_ref_base = wgt(Pp, W_ref) - wgt(Pm, W_ref)
require("T-0c BASELINE (BRIDGE-LOCK's LENS-NULL rebuilt): the one-particle attachment functional is "
        f"bit-EVEN on BOTH branches (Delta_rot = {d_rot_base:+.1e}, Delta_ref = {d_ref_base:+.1e}) "
        "-- the null this station's state-level datum must escape",
        abs(d_rot_base) < 1e-9 and abs(d_ref_base) < 1e-9)

# M-1b's EXACT perpendicularity (W2-MAP): the identity that kills ALL linear-in-B map-algebra data.
Beo = Ue.T @ B0 @ Uo
Call = [I6, J6, S1, S2]
bilinear = np.array([[frob(Beo @ Ci, Cj @ J6) for Cj in Call] for Ci in Call])
require("T-0c M-1b EXACT IDENTITY rebuilt: <Beo.C, C'.J6>_F = 0 for EVERY C,C' in the commutant "
        "(the kill of every LINEAR-in-B functional -- escape condition (b)'s baseline)",
        np.max(np.abs(bilinear)) < 1e-9, detail=f"max = {np.max(np.abs(bilinear)):.2e}")

# ---- T-0d  M0's cell objects: C = (I+iJ6)/2 rank-3 projector; C(-J) = I - C(J); eps-flip ---------
C_cell = (I6 + 1j * J6) / 2
C_cell_m = (I6 - 1j * J6) / 2
require("T-0d M0 cell projector: C = (I+iJ6)/2 is an EXACT rank-3 projector (C^2 = C, C = C^H, "
        "Tr C = 3)",
        np.max(np.abs(C_cell @ C_cell - C_cell)) < 1e-12
        and np.max(np.abs(C_cell - C_cell.conj().T)) < 1e-12
        and abs(np.trace(C_cell).real - 3) < 1e-12)
require("T-0d M0-4a EXACT: C(-J) = I - C(J) (the bit sigma is PARTICLE-HOLE on the covariance)",
        np.max(np.abs(C_cell_m - (I6 - C_cell))) < 1e-15)


def region_eps(C, A):
    C_A = C[np.ix_(A, A)]
    z = np.clip(np.linalg.eigvalsh(C_A).real, 1e-12, 1 - 1e-12)
    return np.sort(np.log((1 - z) / z))


maxd_eps = 0.0
for A_reg in itertools.combinations(range(NE), 3):
    ep = region_eps(C_cell, list(A_reg))
    em = region_eps(C_cell_m, list(A_reg))
    maxd_eps = max(maxd_eps, float(np.max(np.abs(ep + em[::-1]))))
require("T-0d M0-4b: the bit REVERSES the modular flow on EVERY 3-edge region (eps -> -eps, all "
        f"C(6,3)=20 regions; K_A -> -K_A)", maxd_eps < 1e-7, detail=f"max dev = {maxd_eps:.1e}")

# ---- T-0e  the ARROW objects (M0-2R, own conventions; regression, never adjusted) ----------------
GIRTH = the_run.GIRTH                              # = 10, read off B (renewal), not typed
u = float(the_run.U_RUN)                           # = alpha_1 = ((k-1)/k)^(g-2) = (2/3)^8
q = K - 1
u_c = 1.0 / q
b_edge = math.log2(q)
require(f"T-0e arrow objects: k = {K}, u_c = 1/(k-1) = {u_c} = 2^-b_edge (b_edge = {b_edge:.0f}); "
        f"u = alpha_1 = (2/3)^{GIRTH - 2} = {u:.12f}",
        K == 3 and abs(u_c - 0.5) < 1e-15 and abs(u_c - 2 ** (-b_edge)) < 1e-15
        and abs(u - (2.0 / 3.0) ** 8) < 1e-15)
lam_B_max = float(np.max(np.abs(np.linalg.eigvals(B0))))
require(f"T-0e SUB-CRITICALITY (the arrow): u < u_c, and the run series converges: "
        f"rho(uB) = u*(k-1) = {u * q:.6f} < 1 (Perron of B = {lam_B_max:.6f} = k-1)",
        u < u_c and u * q < 1 and abs(lam_B_max - q) < 1e-9)

N_RUN = 32
seed = np.zeros(ND)
seed[0] = 1.0                                      # the declared seed: dart 0 (edge 0, forward)
G_run = np.zeros(ND)
Bn_s = seed.copy()
for n in range(N_RUN + 1):
    G_run = G_run + (u ** n) * Bn_s
    if n < N_RUN:
        Bn_s = B0 @ Bn_s
tail = (u ** N_RUN) * float(np.linalg.norm(Bn_s))
G_exact = np.linalg.solve(np.eye(ND) - u * B0, seed)
require(f"T-0e the run vector |G> = sum_n u^n B^n |seed> CONVERGED (N = {N_RUN}: last term "
        f"{tail:.1e}; matches (I-uB)^-1|seed> to {np.max(np.abs(G_run - G_exact)):.1e})",
        tail < 1e-20 and np.max(np.abs(G_run - G_exact)) < 1e-14)

N_T0 = N_PASS[0]
print(f"\n    T-0 COMPLETE: {N_T0} regression checks PASS.  The three null lemmas + the LENS-NULL "
      "baseline + M0's C(-J) = I - C(J) + the arrow are all in place BEFORE the datum is touched.\n")


# ====================================================================================================
banner("T-1  THEOREM  (the two orbits transport the cell vacuum to C(+J_D) vs C(-J_D): opposite "
       "modular generators, K -> -K, on the R-odd sector)")
# ====================================================================================================
JD = Uo @ J6 @ Uo.T                                # the transported complex structure (R-odd sector)
require("T-1(i) J_D = Uo J6 Uo^T: antisymmetric, J_D^2 = -P_odd on the sector, R-odd-supported "
        "(P_odd J_D P_odd = J_D)",
        np.max(np.abs(JD + JD.T)) < 1e-12 and np.max(np.abs(JD @ JD + P_odd)) < 1e-12
        and np.max(np.abs(P_odd @ JD @ P_odd - JD)) < 1e-12)

dev_rot = 0.0
dev_ref = 0.0
for tt in np.linspace(0, 2 * math.pi, 25):
    Phi_r = Uo @ (math.cos(tt) * I6 + math.sin(tt) * J6)
    dev_rot = max(dev_rot, float(np.max(np.abs(Phi_r @ J6 @ Phi_r.T - JD))),
                  float(np.max(np.abs(Phi_r @ C_cell @ Phi_r.T - (P_odd + 1j * JD) / 2))))
    Phi_f = Uo @ (math.cos(tt) * S1n + math.sin(tt) * S2n)
    dev_ref = max(dev_ref, float(np.max(np.abs(Phi_f @ J6 @ Phi_f.T + JD))),
                  float(np.max(np.abs(Phi_f @ C_cell @ Phi_f.T - (P_odd - 1j * JD) / 2))))
require("T-1(ii) THEOREM (25 pts/branch): EVERY rotation member transports J6 -> +J_D and the cell "
        "vacuum C(J6) -> C(+J_D) = (P_odd + iJ_D)/2; EVERY reflection member -> -J_D and "
        "C(-J_D) = (P_odd - iJ_D)/2",
        dev_rot < 1e-12 and dev_ref < 1e-12,
        detail=f"max dev rotation = {dev_rot:.2e}, reflection = {dev_ref:.2e}")
require("T-1(iii) the transported pair satisfies the SECTOR particle-hole identity "
        "C(-J_D) = P_odd - C(+J_D) (M0-4a transported to the cover; the two orbit vacua are EXACT "
        "particle-hole conjugates)",
        np.max(np.abs((P_odd - 1j * JD) / 2 - (P_odd - (P_odd + 1j * JD) / 2))) < 1e-15)
dev_coord = max(np.max(np.abs(Uo.T @ ((P_odd + 1j * JD) / 2) @ Uo - C_cell)),
                np.max(np.abs(Uo.T @ ((P_odd - 1j * JD) / 2) @ Uo - C_cell_m)))
require("T-1(iv) in Uo coordinates the transported vacua are EXACTLY C(+-J6) => M0-4b applies "
        "VERBATIM: every region's modular spectrum flips eps -> -eps between the two orbit vacua "
        f"(re-checked at T-0d over all 20 regions, max dev {maxd_eps:.1e}) => the two orbits pull "
        "back modular generators of OPPOSITE SIGN, K -> -K, on the R-odd dart sector",
        dev_coord < 1e-12 and maxd_eps < 1e-7)
print("\n    T-1 THEOREM ESTABLISHED: rotation branch <-> C(+J_D), reflection branch <-> C(-J_D);")
print("    the orbit ambiguity IS a modular-flow-orientation ambiguity on the R-odd sector.  The")
print("    arrow question is therefore well-posed: does the run's derived arrow pick one sign?\n")


# ====================================================================================================
banner("T-2  THE ESCAPE GATE  (datum declared in the module docstring BEFORE evaluation; conditions "
       "(a),(b),(c) confronted in order)")
# ====================================================================================================
disclose("datum = reading (ii), the seed->run lag correlation M = (P_odd G)(P_odd seed)^T, "
         "antisymmetrized and Frobenius-paired with J_D(orbit) = +-Uo J6 Uo^T; readings (i)/(iii) "
         "computed as disclosed secondaries; seed = dart 0; thresholds 1e-6 (bit-ODD) / 1e-9 "
         "(machine-zero) / 0.1 (parity defect) -- ALL declared in the docstring before evaluation.")

# ---- the three readings, evaluated ----------------------------------------------------------------
M_ii = P_odd @ np.outer(G_run, seed) @ P_odd                     # reading (ii), PRIMARY
A_ii = (M_ii - M_ii.T) / 2
M_i = np.outer(P_odd @ G_run, P_odd @ G_run)                     # reading (i), literal equal-time
A_i = (M_i - M_i.T) / 2
Resv = np.linalg.inv(np.eye(ND) - u * B0)
M_iii = P_odd @ Resv @ P_odd                                     # reading (iii), seed-free
A_iii = (M_iii - M_iii.T) / 2

w_rot = frob(A_ii, JD)
w_ref = frob(A_ii, -JD)
print(f"""
    THE EVALUATION (the confront numbers, from running code):
      reading (ii) PRIMARY:  ||A||_F = {np.linalg.norm(A_ii):.6f}  (the current-like content is
                             genuinely NONZERO as a matrix)
                             w(rotation)   = <A, +J_D> = {w_rot:+.3e}
                             w(reflection) = <A, -J_D> = {w_ref:+.3e}
      reading (i)  literal:  ||antisym||_max = {np.max(np.abs(A_i)):.3e}   (reality lemma: a REAL
                             state's equal-time correlation is symmetric -- degenerately null at
                             matrix level, as disclosed in the docstring)
      reading (iii) seed-free: ||antisym||_max = {np.max(np.abs(A_iii)):.3e}  <A,J_D> = {frob(A_iii, JD):+.3e}
""")

# ---- (a) seed-anchoring / NOT R-parity-definite ----------------------------------------------------
dG_minus = float(np.linalg.norm(R @ G_run - G_run))
dG_plus = float(np.linalg.norm(R @ G_run + G_run))
a1_pass = min(dG_minus, dG_plus) > TOL_PARITY
print(f"    (a1) R|G> vs |G>: ||RG - G|| = {dG_minus:.6f}, ||RG + G|| = {dG_plus:.6f} "
      f"(both > {TOL_PARITY})  =>  the run state is NOT R-parity-definite: "
      f"{'PASS' if a1_pass else 'FAIL'}")
print("         (the state-level escape from the three lemmas' hypotheses is GENUINE: the seed")
print("          breaks R-parity and the run state carries that breaking)")

seed_even = (seed + R @ seed) / 2                                # the R-symmetrized (gauge-avg) seed
G_even = np.linalg.solve(np.eye(ND) - u * B0, seed_even)
M_even = P_odd @ np.outer(G_even, seed_even) @ P_odd
w_even = frob((M_even - M_even.T) / 2, JD)
M_gavg = P_odd @ Resv @ (np.eye(ND) / ND) @ P_odd                # seed density gauge-averaged -> I/12
w_gavg = frob((M_gavg - M_gavg.T) / 2, JD)
a2_changed = abs(w_rot - w_gavg) > TOL_ZERO
print(f"    (a2) the DATUM under seed removal / gauge-averaging: w(R-even seed) = {w_even:+.3e} "
      f"(P_odd kills the R-even seed identically); w(A4-gauge-averaged seed density) = {w_gavg:+.3e}")
print(f"         vs w(dart-0 seed) = {w_rot:+.3e}  =>  the datum "
      f"{'CHANGES' if a2_changed else 'DOES NOT CHANGE'} under gauge-averaging: "
      f"{'PASS' if a2_changed else 'FAIL -- the functional QUOTIENTS THE SEED-ANCHORING AWAY'}")

# ---- (b) NOT linear in B ---------------------------------------------------------------------------
print(f"""    (b)  STRUCTURAL: the datum is a STATE-LEVEL, SECOND-ORDER object (seed density inserted
         between the B's; ALL orders u^n B^n resummed) -- it is NOT a member of M-1b's killed class
         (linear-in-B, state-free Frobenius pairings on the map algebra <Beo.C, C'.J6>, re-proven
         ZERO at T-0c).  M-1b's exact perpendicularity therefore does NOT apply to it AS A CLASS.
         NUMERIC (the pre-reg's own check): the datum must be NONZERO where M-1b's pairing was
         exactly zero.  M-1b pairing (rebuilt) = {np.max(np.abs(bilinear)):.1e};  |w(datum)| = {abs(w_rot):.3e}""")
b_pass = abs(w_rot) > TOL_ODD
print(f"         =>  numeric escape check (b): {'PASS' if b_pass else 'FAIL -- the datum is ALSO zero'}")

# ---- (c) the LENS test: bit-ODD in evaluation ------------------------------------------------------
BIT_ODD = (abs(w_rot) > TOL_ODD and abs(w_ref) > TOL_ODD and w_rot * w_ref < 0)
BIT_EVEN = (abs(w_rot) < TOL_ZERO and abs(w_ref) < TOL_ZERO)
print(f"    (c)  bit-ODD test: w(rotation) = {w_rot:+.3e}, w(reflection) = {w_ref:+.3e}  ->  "
      f"{'bit-ODD' if BIT_ODD else ('bit-EVEN (machine-zero on BOTH orbits)' if BIT_EVEN else 'INDETERMINATE')}")

if BIT_EVEN:
    print("""
    >>> THE ESCAPE GATE FAILS AT (c): the datum evaluates bit-EVEN -- identically ZERO on both
        orbits.  Per the frozen decision tree: STOP -> ARROW-BLIND.  T-3 (selection) is SKIPPED --
        no selection is manufactured from a zero datum.  The demonstration follows (contract:
        'STOP, print ARROW-BLIND with the demonstration'); every step machine-checked.
""")
    # ------------------------------------------------------------------------------------------------
    # THE DEMONSTRATION -- the zero is a THEOREM, not a numerical accident.  Chain:
    #   D1 (A4-automorphy)      dart_rep(g) commutes with B; A4 simply transitive on the 12 darts.
    #   D2 (invariance)         J_D, P_odd are A4-invariant  =>  w(seed = any dart) IDENTICAL
    #                           =>  w = w(gauge-average) = (1/12)<antisym(P_odd (I-uB)^-1 P_odd), J_D>.
    #   D3 (SYMMETRIC-COMPRESSION LEMMA = Lemmas 1+2 promoted to state level):
    #                           F(u) := Uo^T (I-uB)^-1 Uo satisfies F = F^T EXACTLY
    #                           [F^T = Uo^T (I-uB^T)^-1 Uo = Uo^T R (I-uB)^-1 R Uo = F, using
    #                            R B R = B^T (Lemma 2) and R Uo = -Uo (Lemma 1)]
    #                           =>  antisym(P_odd (I-uB)^-1 P_odd) = 0  =>  w == 0 IDENTICALLY,
    #                           for every u, every dart seed, on BOTH orbits, at EVERY order in B.
    #   D4 (controls)           the pairing is nondegenerate; breaking the responsible symmetries
    #                           makes the SAME functional read nonzero orientations.
    #   D5 (corollary)          Uo^T f(B) Uo = Uo^T f(B^T) Uo for ANY f: the WHOLE functional class
    #                           is FORWARD/REVERSED-blind -- the arrow is invisible to it.
    # ------------------------------------------------------------------------------------------------
    require("T-2 D0 the current content is genuinely there: ||A||_F > 0.01 (what vanishes is the "
            "ORIENTATION PAIRING, not the correlation itself)", np.linalg.norm(A_ii) > 0.01,
            detail=f"||A||_F = {np.linalg.norm(A_ii):.6f}")
    dev_comm_B = max(np.max(np.abs(dart_rep(g) @ B0 - B0 @ dart_rep(g))) for g in A4)
    chi_dart = np.array([np.trace(dart_rep(g)) for g in A4])
    require("T-2 D1 dart_rep(g) commutes with B EXACTLY for every g (A4 = graph automorphisms of "
            "the cover); dart_rep = the REGULAR rep (trace 0 off identity: A4 SIMPLY TRANSITIVE on "
            "the 12 darts)",
            dev_comm_B < 1e-12 and np.allclose(chi_dart[1:], 0.0, atol=1e-9)
            and abs(chi_dart[0] - 12) < 1e-9, detail=f"max ||[g,B]|| = {dev_comm_B:.1e}")
    dev_JD_inv = max(np.max(np.abs(dart_rep(g) @ JD @ dart_rep(g).T - JD)) for g in A4)
    dev_Po_inv = max(np.max(np.abs(dart_rep(g) @ P_odd @ dart_rep(g).T - P_odd)) for g in A4)
    ws_all = []
    for d in range(ND):
        sd = np.zeros(ND)
        sd[d] = 1.0
        Gd = np.linalg.solve(np.eye(ND) - u * B0, sd)
        Md = P_odd @ np.outer(Gd, sd) @ P_odd
        ws_all.append(frob((Md - Md.T) / 2, JD))
    ws_all = np.array(ws_all)
    require("T-2 D2 J_D and P_odd are A4-invariant => SEED-TRANSITIVITY: w(seed = dart d) is "
            "IDENTICAL for ALL 12 darts (machine-checked; all equal, all machine-zero) => the "
            "seed-anchoring is PROVABLY quotiented out of the datum: w(seed) = w(gauge-average)",
            dev_JD_inv < 1e-12 and dev_Po_inv < 1e-12 and np.max(np.abs(ws_all)) < TOL_ZERO
            and np.max(np.abs(ws_all - ws_all[0])) < TOL_ZERO,
            detail=f"max |w(d)| = {np.max(np.abs(ws_all)):.2e}")
    F_u = Uo.T @ Resv @ Uo
    require("T-2 D3 THE SYMMETRIC-COMPRESSION LEMMA (Lemmas 1+2 at state level): "
            "F(u) = Uo^T (I-uB)^-1 Uo is EXACTLY SYMMETRIC => antisym(P_odd (I-uB)^-1 P_odd) = 0 "
            "=> the gauge-averaged datum -- and with D2 EVERY seeded datum -- is IDENTICALLY zero",
            np.max(np.abs(F_u - F_u.T)) < 1e-12,
            detail=f"||F - F^T|| = {np.max(np.abs(F_u - F_u.T)):.2e}")
    dev_order = 0.0
    Bn = np.eye(ND)
    for n in range(13):
        Fn = Uo.T @ Bn @ Uo
        dev_order = max(dev_order, float(np.max(np.abs(Fn - Fn.T))))
        Bn = Bn @ B0
    require("T-2 D3' ORDER-BY-ORDER: Uo^T B^n Uo is symmetric for every n = 0..12 (the zero is NOT "
            "a resummation accident; it holds at each order of the run series -- the poison's "
            "'no linear-in-B datum' worry is moot: even the n=1 content pairs to zero)",
            dev_order < 1e-12, detail=f"max asym over orders = {dev_order:.2e}")
    # D4 controls: the functional works; the SYMMETRY does the killing.
    rngc = np.random.default_rng(1)
    Xg = rngc.normal(size=(ND, ND))
    w_generic = frob((Xg - Xg.T) / 2, JD)
    rev = [2 * (d // 2) + 1 - d % 2 for d in range(ND)]
    hops = np.argwhere(B0 > 0.5)
    b_, a_ = int(hops[0][0]), int(hops[0][1])
    X1 = np.zeros((ND, ND))
    X1[b_, a_] += 1.0
    X1[rev[a_], rev[b_]] += 1.0                    # keeps R X R = X^T (Lemma 2 preserved)
    B_c1 = B0 + 0.3 * X1                           # control: A4 BROKEN, R-conjugation KEPT
    X2 = np.zeros((ND, ND))
    X2[b_, a_] += 1.0
    B_c2 = B0 + 0.3 * X2                           # control: BOTH broken


    def datum_of(Bmat, sd):
        Gd = np.linalg.solve(np.eye(ND) - u * Bmat, sd)
        Md = P_odd @ np.outer(Gd, sd) @ P_odd
        return frob((Md - Md.T) / 2, JD)


    ws_c1 = np.array([datum_of(B_c1, np.eye(ND)[:, d]) for d in range(ND)])
    F_c1 = Uo.T @ np.linalg.inv(np.eye(ND) - u * B_c1) @ Uo
    ws_c2 = np.array([datum_of(B_c2, np.eye(ND)[:, d]) for d in range(ND)])
    ws_c2T = np.array([datum_of(B_c2.T, np.eye(ND)[:, d]) for d in range(ND)])
    F_c2 = Uo.T @ np.linalg.inv(np.eye(ND) - u * B_c2) @ Uo
    require("T-2 D4 CONTROL 1 (nondegeneracy): a generic antisymmetric matrix pairs NONZERO with "
            f"J_D (<antisym(X), J_D> = {w_generic:+.4f}) -- the pairing itself discriminates fine",
            abs(w_generic) > 1e-3)
    require("T-2 D4 CONTROL 2 (break A4 only, KEEP R B R = B^T): the SAME functional on the "
            "perturbed dynamics reads NONZERO per-seed orientations "
            f"(max |w| = {np.max(np.abs(ws_c1)):.2e}) while the seed-AVERAGE stays zero "
            f"(F still symmetric, {np.max(np.abs(F_c1 - F_c1.T)):.1e}; avg w = {np.mean(ws_c1):+.1e}) "
            "-- A4-transitivity is what kills the SEED-anchored read",
            np.max(np.abs(R @ B_c1 @ R - B_c1.T)) < 1e-12 and np.max(np.abs(ws_c1)) > 1e-4
            and np.max(np.abs(F_c1 - F_c1.T)) < 1e-12 and abs(np.mean(ws_c1)) < 1e-12)
    require("T-2 D4 CONTROL 3 (break BOTH): F becomes ASYMMETRIC "
            f"({np.max(np.abs(F_c2 - F_c2.T)):.2e} > 1e-3), the averaged datum is NONZERO "
            f"({np.mean(ws_c2):+.2e}), AND the reversed run now gives DIFFERENT per-seed values "
            f"(max |w_fwd - w_rev| = {np.max(np.abs(ws_c2 - ws_c2T)):.2e}) -- the functional class "
            "genuinely CAN read an arrow when the symmetries are absent: the null on the TRUE B is "
            "100% a property of the srs walk's own symmetry, not of the functional",
            np.max(np.abs(F_c2 - F_c2.T)) > 1e-3 and abs(np.mean(ws_c2)) > 1e-5
            and np.max(np.abs(ws_c2 - ws_c2T)) > 1e-4)
    ResvT = np.linalg.inv(np.eye(ND) - u * B0.T)
    require("T-2 D5 COROLLARY (ARROW-INVISIBILITY): Uo^T (I-uB)^-1 Uo == Uo^T (I-uB^T)^-1 Uo "
            "EXACTLY [f(B^T) = R f(B) R + R Uo = -Uo] => the ENTIRE R-odd compression of the run is "
            "FORWARD/REVERSED-INVARIANT: no datum in this class can EVER see the arrow",
            np.max(np.abs(Uo.T @ ResvT @ Uo - F_u)) < 1e-12,
            detail=f"dev = {np.max(np.abs(Uo.T @ ResvT @ Uo - F_u)):.2e}")
    print("""
    ================================================================================================
    T-2 VERDICT: the datum is bit-EVEN -- ARROW-BLIND, and it is THEOREM-GRADE.  The escape from
    BRIDGE-LOCK's null was structurally genuine at the STATE level ((a1): the run state is NOT
    R-parity-definite, defect ~1.42) -- but the DATUM the state can feed the orbit question passes
    through two quotients that kill it EXACTLY:
      (1) A4 acts simply transitively on the darts and commutes with B, J_D, P_odd  =>  the seeded
          datum EQUALS its own gauge-average (the seed-anchoring is quotiented away; D1+D2);
      (2) the gauge-averaged object is the R-odd compression of a function of B, and Lemmas 1+2
          (R Uo = -Uo; R B R = B^T) force EVERY such compression to be SYMMETRIC (D3, order by
          order D3') -- zero antisymmetric part, zero pairing with ANY orientation form.
    This EXTENDS the BRIDGE-LOCK null theorem from R-parity-definite one-particle transports to
    ALL seed-anchored, state-level, all-orders-in-B two-point run data on the R-odd sector -- the
    theorem-strengthening the pre-reg's NULL-ESCAPE-FAIL clause anticipated, reached here through
    (c)'s own gate (the datum did sit outside the three lemmas' LITERAL hypotheses, so the booked
    verdict is ARROW-BLIND, with the strengthening as its demonstration).  The corollary (D5) is
    the sharpest form: the R-odd compression of the run CANNOT distinguish B from B^T at all --
    the derived arrow is structurally invisible to every functional of this class, forward or
    reversed.  (The MASTER CHIRALITY LENS reading: the compression is bit-EVEN = democratic =
    blind; the arrow lives in the run, but the R-odd window mirror-symmetrizes it away.)
    ================================================================================================
""")
elif BIT_ODD:
    print(f"    >>> ESCAPE GATE (c) PASSES: bit-ODD (w_rot = {w_rot:+.6e}, w_ref = {w_ref:+.6e}, "
          "opposite signs).  Proceeding to T-3.\n")
else:
    print(f"    >>> INDETERMINATE regime (w_rot = {w_rot:+.3e}, w_ref = {w_ref:+.3e}): neither "
          "clean bit-ODD nor clean bit-EVEN.  Booked conservatively as ARROW-BLIND-INDETERMINATE "
          "below; no selection claimed.\n")


# ====================================================================================================
banner("T-3  THE SELECTION READ  (frozen rule; PRINTED, never asserted)")
# ====================================================================================================
VERDICT = None
if BIT_ODD:
    sel = "ROTATION" if w_rot > 0 else "REFLECTION"
    print(f"    FROZEN RULE: MATCH := w(orbit) > 0 (transported flow orientation aligned with the "
          f"forward run).  w_rot = {w_rot:+.6e}, w_ref = {w_ref:+.6e}  =>  SELECTED ORBIT: {sel}")
    VERDICT = ("ARROW-SELECTS", sel)
elif BIT_EVEN:
    print("""    SKIPPED per the frozen T-2 gate: the datum is identically zero on both orbits -- the
    frozen rule ('the orbit whose transported flow orientation MATCHES the run's arrow') selects
    NOTHING, and no selection is manufactured.  The T-1 theorem STANDS (the orbits DO carry
    opposite modular orientations); what failed is the cover-side ARROW READ, not the orientation
    structure.""")
    VERDICT = ("ARROW-BLIND", None)
else:
    print("    SKIPPED: indeterminate datum.  No selection claimed.")
    VERDICT = ("ARROW-BLIND-INDETERMINATE", None)


# ====================================================================================================
banner("T-4  CONVENTION-COVARIANCE  (joint det-phi + tau_dart flip; the reversed-run flip test; "
       "PRINTED, never asserted)")
# ====================================================================================================
# the joint flip, operationally (BRIDGE-LOCK L-3 verbatim logic):
#   det-phi: phi -> -phi => J6' = -J6 (the bit flips; machine-checked below).
#   tau_dart: a sign flip would demand the R-EVEN branch, which is EMPTY by the convention-free
#   rank obstruction (T-0b(iii)) => R5-survival RE-PINS the R-odd family; tau_dart is ABSORBED.
phi3_f = -phi3
J6_f = J6_of(phi3_f)
JD_f = Uo @ J6_f @ Uo.T
w_rot_f = frob(A_ii, JD_f)                          # the datum against the flipped conventions
w_ref_f = frob(A_ii, -JD_f)
print(f"    joint flip: det(phi') = {float(np.linalg.det(phi3_f)):+.6f} < 0; J6' = -J6: "
      f"{np.max(np.abs(J6_f + J6)) < 1e-12}; tau_dart absorbed by R5-survival (rank obstruction "
      f"is convention-free).  Recomputed datum: w'(rot) = {w_rot_f:+.3e}, w'(ref) = {w_ref_f:+.3e}")

# the reversed run (the PHYSICAL arrow test): B -> B^T with everything else held fixed
G_T = np.linalg.solve(np.eye(ND) - u * B0.T, seed)
M_T = P_odd @ np.outer(G_T, seed) @ P_odd
A_T = (M_T - M_T.T) / 2
w_rot_T = frob(A_T, JD)
w_ref_T = frob(A_T, -JD)
print(f"    reversed run (B -> B^T): w_T(rot) = {w_rot_T:+.3e}, w_T(ref) = {w_ref_T:+.3e}   "
      f"[forward was {w_rot:+.3e} / {w_ref:+.3e}];  ||A_T - A||_max = {np.max(np.abs(A_T - A_ii)):.2e}")

if VERDICT[0] == "ARROW-SELECTS":
    sel_f = "ROTATION" if w_rot_f > 0 else "REFLECTION"
    JD_sel = JD if VERDICT[1] == "ROTATION" else -JD
    JD_sel_f = JD_f if sel_f == "ROTATION" else -JD_f
    stable = np.max(np.abs(JD_sel_f - JD_sel)) < 1e-9
    sel_T = "ROTATION" if w_rot_T > 0 else "REFLECTION"
    flipped = (sel_T != VERDICT[1]) and abs(w_rot_T) > TOL_ODD
    print(f"    flipped-convention label: {sel_f}; PHYSICAL orbit stable: {stable}")
    print(f"    reversed-run selection: {sel_T}; selection FLIPPED with the arrow: {flipped}")
    if stable and flipped:
        print(f"    => selection PHYSICAL and ARROW-TRACKING: VERDICT ARROW-SELECTS ({VERDICT[1]}) stands.")
    else:
        VERDICT = ("CONVENTION-ARTIFACT", None)
        print("    => the selection failed covariance (label-stability and/or the arrow flip): "
              "CONVENTION-ARTIFACT.")
elif VERDICT[0].startswith("ARROW-BLIND"):
    print(f"""    MOOT-UNDER-NULL: there is no selection to test.  What CAN be tested -- and is -- is the
    NULL itself:
      * under the joint convention flip the datum is STILL machine-zero on both orbits
        (w' = {w_rot_f:+.1e} / {w_ref_f:+.1e}) -- the demonstration's lemmas (R Uo = -Uo, R B R = B^T,
        A4-transitivity) contain NO sign convention; the null is NOT a convention artifact.
      * the reversed-run test cannot flip a null selection -- and the demonstration's corollary D5
        proves WHY at theorem grade: the R-odd compression of the run is IDENTICALLY the same for
        B and B^T (machine-checked, dev < 1e-12), so the datum class is arrow-invisible as such,
        not merely zero at the operating point.  The physical arrow (u < u_c) is untouched and was
        never adjusted; it is simply not readable through this window.""")


# ====================================================================================================
banner("SCOPE + POISON DISCHARGE  (printed; nothing moves)")
# ====================================================================================================
print("""    NOT claimed / not touched by this station:
      * NO oblique/M_Z/EW quantity was computed or mentioned; M-2/M-3 (transport/insertion) stay
        GATED regardless of this verdict; no scoreboard value moves.
      * the A5 anchors were not needed and not touched (BRIDGE-LOCK re-verified them; this
        station's baseline is the three lemmas + the LENS-NULL attachment, all re-proven at T-0).
      * the ARROW (u < u_c, u = alpha_1) was regression-verified (T-0e) and NEVER adjusted; the
        verdict says the R-odd dart window cannot READ it, not that it is absent.
      * FS-5iii DISCHARGED: no Z3/deck identification anywhere -- the datum's only internal input
        is J6 itself (via J_D = Uo J6 Uo^T, A4-invariant, deck-free); no cover-side Z3 object was
        constructed or used.
      * NO linear-in-B datum: the declared datum is the full resolvent series; its order-by-order
        content was additionally confronted (D3') and the M-1b killed class was re-proven zero
        BEFORE evaluation (T-0c), per the poison's ordering requirement.
      * the three BRIDGE-LOCK lemmas were confronted BEFORE the datum was evaluated (T-0c), per
        the poison's ordering requirement.
      * W2-MAP's AMBIGUOUS-BY-O(2) STANDS, sharpened again: after BRIDGE-LOCK (no R-parity-definite
        one-particle attachment functional discriminates) this station adds: no seed-anchored,
        state-level, all-orders two-point datum of the sub-critical run ON THE R-ODD DART SECTOR
        discriminates either -- and none can track the arrow (D5).  A discriminator must leave
        this class: a phase-bearing/Fock-level object, or a geometric (Design C / BRIDGE-GEOM)
        read, per the frozen verdict table.""")


# ====================================================================================================
banner("SUMMARY")
# ====================================================================================================
elapsed = time.time() - t_start
final = VERDICT[0] + (f" ({VERDICT[1]})" if VERDICT[1] else "")
print(f"    T-0  REGRESSION .............................. PASS ({N_T0} checks: O(2) family + three "
      f"lemmas + LENS-NULL baseline + M-1b identity + M0 C(-J)=I-C(J)/eps-flip + arrow objects)")
print(f"    T-1  THEOREM ................................. rotation -> C(+J_D), reflection -> "
      f"C(-J_D); sector particle-hole EXACT; eps -> -eps on all 20 regions (K -> -K)")
print(f"    T-2  ESCAPE GATE ............................. (a1) PASS (parity defect "
      f"{min(dG_minus, dG_plus):.3f} > {TOL_PARITY}); (a2) {'PASS' if abs(w_rot - w_gavg) > TOL_ZERO else 'FAIL (datum = its own gauge-average)'}; "
      f"(b) numeric {'PASS' if abs(w_rot) > TOL_ODD else 'FAIL (datum zero)'}; "
      f"(c) {'bit-ODD' if BIT_ODD else 'bit-EVEN'}: w = {w_rot:+.2e} / {w_ref:+.2e}")
print(f"    T-3  SELECTION ............................... "
      f"{'read: ' + str(VERDICT[1]) if VERDICT[0] == 'ARROW-SELECTS' else 'SKIPPED per the frozen gate'}")
print(f"    T-4  CONVENTION-COVARIANCE ................... "
      f"{'tested on the selection' if VERDICT[0] in ('ARROW-SELECTS', 'CONVENTION-ARTIFACT') else 'MOOT-UNDER-NULL; the null itself is convention-stable AND provably arrow-invisible (D5)'}")
print(f"    asserted checks total: {N_PASS[0]};  disclosed interpretation steps: {len(DISCLOSURES)}")
print(f"    runtime: {elapsed:.1f}s")
print()
print(f" FINAL VERDICT (printed, per contract): {final}")
if VERDICT[0] == "ARROW-BLIND":
    print("   -> the O(2) orbit ambiguity STANDS; hand to BRIDGE-GEOM (Design C) per the frozen")
    print("      verdict table.  The demonstration is itself a bookable theorem-strengthening of")
    print("      BRIDGE-LOCK's null (state-level extension via A4-transitivity + the symmetric-")
    print("      compression lemma).  A null here is a result, not a defect.")
print("=" * 100)
sys.exit(0)
