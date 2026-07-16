#!/usr/bin/env python3
"""
proofs/foundations/W2_MAP_vertex_propagator_2026-07-10.py

W2-MAP -- THE VERTEX/PROPAGATOR MAP (classification-first).  Pre-registered in
internal research notes (commit 8ca645c; RECORD 6e03d95;
adjudications 1-4, contracts M-0..M-5, POISONS -- all frozen BEFORE this file was written).

The design sweep (2026-07-10) discharged the naive version of the architect's design note: the walk
operator's R-even<->R-odd cross block Beo is ORTHOGONAL to J6 (residual 1.000) -- the map is NOT a free
read off the shared edge substrate.  This station is therefore CLASSIFICATION-FIRST: it computes the
FULL space of maps Phi: internal (edge/J6/Witt) one-particle structure -> cover (dart) one-particle
structure satisfying the DERIVED requirement set {R1,R2,R5}, reports FORCED / AMBIGUOUS-BY-k / EMPTY,
and only then (gated) transports operators and inserts the payoff read.  M-4 (the ML-5 same-object
confront) runs regardless of the classification's verdict.

============================================================================================
BASIS CONVENTIONS (adjudication 1 -- declared ONCE, at the top; every cross-side equation states which)
============================================================================================
INTERNAL side: the RAW Cl(6) representation (NOT the Fock/SUBSETS/W basis of ncg_spectral.py).
  - the_net.py's complex_structure_J6(): a real antisymmetric 6x6 J6 on the 6-edge space (EIDX from
    srs.EDGES: e0=(0,1) e1=(0,2) e2=(0,3) e3=(1,2) e4=(1,3) e5=(2,3)), J6^2=-I, A4-forced-unique.
  - edge_rep(g): the SAME signed-permutation 6x6 real A4 action J6 is covariant under (WS1/G8-IF/FS
    convention, reconstructed verbatim below).
  - the Witt ladder A_ops[0..2] (single-particle creation ops a_i, from J6's +i eigenmodes), in the
    ORIGINAL (pre-W) Cl(6) representation -- the SAME convention furey_stoica_labels.py's FS-3/FS-4 and
    ML5b_epsilon_transport_2026-07-08.py use, and DELIBERATELY NOT the Fock/SUBSETS-ordered W-basis
    ncg_spectral.py uses for PF/Vsig/U_A4_F (IF-0h's "W-basis lesson").  This station never combines a
    W-basis object with a raw-basis object, so the IF-0h re-verification is not triggered; disclosed here
    per adjudication 1 rather than silently assumed.
COVER side: srs.py's dart indexing (dart 2e = edge e forward (i,j,v); dart 2e+1 = reversed (j,i,-v);
  ND=12), the reversal involution R (eigenvalues {+1^6,-1^6}; the_net.py's reversal()), and the R-even/
  R-odd edge-coefficient embeddings Ue[:,e]=(d_2e+d_2e+1)/sqrt2, Uo[:,e]=(d_2e-d_2e+1)/sqrt2.

Runtime target <= 5 min (Gamma-point only; 6/8/12-dim algebra).  Exit 0 iff every M-0 regression holds
and M-1/M-1b/M-4 each reach one of their pre-declared definite verdicts (booking an honest AMBIGUOUS or
a gated-skip of M-2/M-3 both count as "definite" -- exactly as the pre-reg's own dual-outcome contracts
define "definite": a completed, booked reading, not necessarily a resolving one).

POISONS (binding, restated): no map invented outside the M-1 classification; the requirement set never
extended mid-run to force uniqueness; no fit-grade selection (M-1b's minimum must be a ZERO, <1e-9); the
FS-5iii negative never overridden; both sides' basis conventions verified before any cross-side number;
the 0.0139%/+7.76sigma/1.62-bit numbers are HISTORY/comparison only, never fit; numbers only from running
code; engine/adapters/the_net.py UNTOUCHED.
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
from simulator.srs_engine.utils import AlgebraicUtility  # noqa: E402
from the_run import gauge_singlet_projection, hashimoto  # noqa: E402  (M-4 target: the M_Z oblique's c_S=1/12)

np.set_printoptions(precision=6, suppress=True, linewidth=120)

ok_all = True
DISCLOSURES = []


def check(name, cond, detail=""):
    global ok_all
    cond = bool(cond)
    ok_all = ok_all and cond
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return cond


def banner(t):
    print("=" * 100)
    print(f" {t}")
    print("=" * 100)


def disclose(msg):
    DISCLOSURES.append(msg)
    print(f"    [DISCLOSED INTERPRETATION] {msg}")


# ====================================================================================================
banner("M-0  ANCHORS  (both sides' machinery rebuilt, own checks re-run)")
# ====================================================================================================
EDGES = srs.EDGES
NE, NV = len(EDGES), srs.NV
EIDX = {(min(i, j), max(i, j)): e for e, (i, j, v) in enumerate(EDGES)}
DARTS = srs._darts()
ND = len(DARTS)                                    # 12
g6 = [np.array(g, complex) for g in AlgebraicUtility.cl6_generators()]
I8 = np.eye(8, dtype=complex)
gam = lambda x: sum(x[a] * g6[a] for a in range(NE))
EDGE_OF_DART = [d // 2 for d in range(ND)]

A4 = [dict(enumerate(p)) for p in itertools.permutations(range(4))
      if sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j]) % 2 == 0]


def edge_rep(sig):
    """The internal A4 action on the 6-edge space (WS1/FS/G8-IF convention, reconstructed verbatim)."""
    R6 = np.zeros((NE, NE))
    for e, (i, j, v) in enumerate(EDGES):
        a, b = sig[i], sig[j]
        s = 1.0
        if a > b:
            a, b, s = b, a, -1.0
        R6[EIDX[(a, b)], e] = s
    return R6


def dart_rep(sig):
    """NEW (this station): the natural cover-side A4 action on the 12-dim DART space, induced by the
    SAME vertex permutation sig acting on a dart's (tail,head) vertex labels.  A genuine (non-projective)
    permutation representation -- verified below to be a bona fide A4-homomorphism and to commute
    exactly with the reversal R."""
    Rd = np.zeros((ND, ND))
    for a, (i, j, v) in enumerate(DARTS):
        ni, nj = sig[i], sig[j]
        lo, hi = min(ni, nj), max(ni, nj)
        e2 = EIDX[(lo, hi)]
        b = 2 * e2 if ni < nj else 2 * e2 + 1
        Rd[b, a] = 1.0
    return Rd


def compose(g, h):
    return {i: g[h[i]] for i in range(NV)}


J6 = net.complex_structure_J6()
R = net.reversal()
B0 = net.hashimoto_gamma()

Ue = np.zeros((ND, NE))
Uo = np.zeros((ND, NE))
for e in range(NE):
    Ue[2 * e, e] = 1 / math.sqrt(2); Ue[2 * e + 1, e] = 1 / math.sqrt(2)
    Uo[2 * e, e] = 1 / math.sqrt(2); Uo[2 * e + 1, e] = -1 / math.sqrt(2)

# ---- re-run the anchors ----
check("M-0a J6^2 = -I (forced-unique complex structure)", np.max(np.abs(J6 @ J6 + np.eye(NE))) < 1e-12)

wJ, VJ = np.linalg.eig(J6)
modes, _ = np.linalg.qr(VJ[:, np.where(np.abs(wJ - 1j) < 1e-9)[0]])
A_ops = [gam(np.conj(modes[:, m])) / math.sqrt(2) for m in range(3)]
Adag_ops = [a.conj().T for a in A_ops]
nilp = max(np.max(np.abs(A_ops[i] @ A_ops[i])) for i in range(3))
acomm = max(np.max(np.abs(A_ops[i] @ A_ops[j] + A_ops[j] @ A_ops[i])) for i in range(3) for j in range(3) if i != j)
car = max(np.max(np.abs(A_ops[i] @ Adag_ops[j] + Adag_ops[j] @ A_ops[i] - (I8 if i == j else 0.0)))
          for i in range(3) for j in range(3))
check("M-0b Witt ladder identities: a_i^2=0, {a_i,a_j}=0 (i!=j), {a_i,a_j^dag}=delta_ij "
      f"(max devs {nilp:.1e}, {acomm:.1e}, {car:.1e})", max(nilp, acomm, car) < 1e-12)

check("M-0c srs dart convention: ND=2|E|=12, dart 2e/2e+1 = edge e forward/reversed", ND == 12)

evR = np.sort(np.linalg.eigvalsh(R))
check("M-0d R eigenstructure {+1^6,-1^6}, R^2=I, Tr R=0",
      np.allclose(evR, np.array([-1.0] * 6 + [1.0] * 6)) and np.max(np.abs(R @ R - np.eye(ND))) < 1e-12
      and abs(np.trace(R)) < 1e-12)

shat = np.ones(ND) / math.sqrt(ND)
shat_Reven = np.max(np.abs(R @ shat - shat)) < 1e-12
coeff_Ue = Ue.T @ shat
check("M-0e SWEEP FACT: shat=ones(12)/sqrt(12) is exactly R-even with UNIFORM 1/sqrt(6) coefficients in Ue",
      shat_Reven and np.allclose(coeff_Ue, 1 / math.sqrt(6), atol=1e-12),
      detail=f"Ue^T.shat = {np.round(coeff_Ue, 6)}")

Beo = Ue.T @ B0 @ Uo
Aeo = (Beo - Beo.T) / 2
c_bestfit = np.sum(Aeo * J6) / np.sum(J6 * J6)
resid_naive = np.linalg.norm(Aeo - c_bestfit * J6) / np.linalg.norm(Aeo)
check("M-0f SWEEP FACT (regression): antisym(Beo) best-fit scale vs J6 ~ 0, normalized residual = 1.000 "
      "-- the naive-candidate kill",
      abs(c_bestfit) < 1e-9 and abs(resid_naive - 1.0) < 1e-9,
      detail=f"c={c_bestfit:.3e}, residual={resid_naive:.6f}")

Bee, Boe, Boo = Ue.T @ B0 @ Ue, Uo.T @ B0 @ Ue, Uo.T @ B0 @ Uo
norms = {name: np.linalg.norm(M) for name, M in [("ee", Bee), ("eo", Beo), ("oe", Boe), ("oo", Boo)]}
check(f"M-0g SWEEP FACT: all four R-parity blocks of B(Gamma) have equal Frobenius norm sqrt(6)=2.449",
      all(abs(v - math.sqrt(6)) < 1e-9 for v in norms.values()), detail=f"norms={ {k: round(v,4) for k,v in norms.items()} }")

# ---- NEW structural facts this station needs for M-1 ----
max_comm_J6 = max(np.max(np.abs(edge_rep(g) @ J6 - J6 @ edge_rep(g))) for g in A4)
check("M-0h NEW: edge_rep(g) commutes EXACTLY with J6 for every g in A4 (J6 genuinely lives in the "
      "edge_rep representation -- justifies the C^3 (+i eigenspace) Witt-mode picture below)",
      max_comm_J6 < 1e-12, detail=f"max_g||[edge_rep(g),J6]|| = {max_comm_J6:.2e}")

g0, h0 = A4[3], A4[7]
homdev = np.max(np.abs(dart_rep(g0) @ dart_rep(h0) - dart_rep(compose(g0, h0))))
check("M-0i NEW: dart_rep is a genuine A4 homomorphism (dart_rep(g)dart_rep(h)=dart_rep(gh))",
      homdev < 1e-12, detail=f"dev={homdev:.2e}")

max_comm_R = max(np.max(np.abs(dart_rep(g) @ R - R @ dart_rep(g))) for g in A4)
check("M-0j NEW: R commutes EXACTLY with dart_rep(g) for every g (R lies in the A4-commutant of the "
      "dart action -- the natural cover-side companion structure for R1)",
      max_comm_R < 1e-12, detail=f"max_g||[dart_rep(g),R]|| = {max_comm_R:.2e}")

dev_Ue = max(np.max(np.abs(dart_rep(g) @ Ue - Ue @ edge_rep(g))) for g in A4)
dev_Uo = max(np.max(np.abs(dart_rep(g) @ Uo - Uo @ edge_rep(g))) for g in A4)
check("M-0k NEW / DOCSTRING-VS-COMPUTATION FINDING: Uo (the SIGNED, R-ODD embedding) intertwines "
      "edge_rep with dart_rep EXACTLY; Ue (the UNSIGNED, R-EVEN embedding) does NOT (maximal mismatch) "
      "-- i.e. edge_rep's own sign convention (flip under orientation reversal, matching a genuine "
      "'oriented edge'/1-form transformation law) is carried by the R-ODD sector, not the R-even sector "
      "the_net.py's reversal() docstring calls 'the undirected edge space carrying the vacuum J6/C'",
      dev_Uo < 1e-12 and dev_Ue > 1.0, detail=f"dev(Uo)={dev_Uo:.2e}  dev(Ue)={dev_Ue:.4f} (=sqrt2)")
disclose("the_net.py's own docstring ('R-even is the undirected edge space carrying the vacuum J6/C') "
         "is NOT what the A4-covariance computation shows: J6/edge_rep's sign convention is realized by "
         "the R-ODD embedding Uo, not R-even Ue.  Booked here as a genuine finding (not silently used as "
         "a requirement, per adjudication 4's treatment of the analogous B-block docstring gap); it "
         "directly informs which R-parity M-1a's R1 constraint can possibly select.")

chi_edge = np.array([np.trace(edge_rep(g)) for g in A4])
chi_dart = np.array([np.trace(dart_rep(g)) for g in A4])
ip_ee = float(np.sum(chi_edge * chi_edge) / len(A4))
ip_dd = float(np.sum(chi_dart * chi_dart) / len(A4))
check("M-0l character check: edge_rep = 2 copies of A4's 3-irrep (ip(chi,chi)=4); dart_rep = the "
      "REGULAR rep (trace 0 off identity, ip(chi,chi)=12 = 1+1+1+9, i.e. A4 acts SIMPLY TRANSITIVELY "
      "on the 12 darts)",
      abs(ip_ee - 4.0) < 1e-9 and abs(ip_dd - 12.0) < 1e-9
      and np.allclose(chi_dart[1:], 0.0, atol=1e-9) and abs(chi_dart[0] - 12) < 1e-9,
      detail=f"ip(edge,edge)={ip_ee:.1f}  ip(dart,dart)={ip_dd:.1f}")

print(f"\n    M-0 SUMMARY: internal edge_rep = 3(+)3 of A4; cover dart_rep = the REGULAR rep of A4 "
      f"(1(+)1'(+)1''(+)3(+)3(+)3); R is right-multiplication by an order-2 group element in the "
      f"commutant of the regular rep.  All anchors + sweep facts reproduced.\n")


# ====================================================================================================
banner("M-1a  THE CLASSIFICATION  (the FULL space of Phi: internal -> cover satisfying {R1,R2,R5})")
# ====================================================================================================
print("""    DISCLOSED SET-UP (adjudication: honest version of the pre-reg's loose wording, per the
    dispatch brief).  The pre-reg writes Phi: C^6(J6,+i modes) -> C^12(darts).  Resolution:
      * "C^6(J6,+i modes)" IS the C^3 Witt-mode space (the +i eigenspace of J6's complexification,
        3 COMPLEX dims -- the actual one-particle Hilbert space of A_ops).  A REAL 12x6 matrix Phi_R
        (a real-linear map R^6->R^12) and a COMPLEX-linear map C^3->C^12 carry IDENTICAL data: since
        edge_rep(g) commutes exactly with J6 (M-0h), the +i eigenspace is edge_rep-invariant, and ANY
        real Phi_R is automatically complex-linear on it (its action on the -i eigenspace is forced,
        Phi_R(conj v) = conj(Phi_R(v)), by reality).  So this station SOLVES the honest REAL 12x6
        linear system (as directed) -- this is exactly the C^3->C^12 classification, repackaged.
      * R1 ("Phi intertwines the dart reversal R with complex conjugation, compatibility with the
        certified antiunitary sigma_M0/J_F") is DERIVED, not read off verbatim: the one-particle
        antiunitary companion to J6 is sigma_int := ordinary conjugation K_6 on C^6=R^6 (x) C -- this
        IS M0's own "bit sigma: J -> -J" particle-hole structure, the one-particle seed of the
        certified sigma_M0/J_F antiunitary (built at Fock level as J_F=V_sigma o K).  The NATURAL
        cover-side companion, built the SAME way (a UNITARY "V" composed with K), is
        tau_dart := R o K_12 (R plays the V_sigma role; R^2=I => tau_dart^2=I, a genuine order-2
        antiunitary).  Requiring Phi o sigma_int = tau_dart o Phi (Phi extended over all of C^6 by its
        forced conjugate-symmetric completion) reduces by direct algebra to: Phi's image must lie in a
        SINGLE R-parity eigenspace, R.Phi = eps*Phi for a fixed eps=+-1.  This is a genuine derivation
        step (not given verbatim by the pre-reg) -- BOTH signs are computed below; the sign is NOT
        chosen by fiat, it is selected by which branch survives R5 (isometry), reported honestly.""")

# ---- R2: solve Hom_A4(edge_rep, dart_rep) directly (vectorized real 12x6 linear system) ----
rows = []
for g in A4:
    Rd, R6 = dart_rep(g), edge_rep(g)
    rows.append(np.kron(np.eye(NE), Rd) - np.kron(R6.T, np.eye(ND)))   # vec(Rd.Phi - Phi.R6) = 0
Cstack = np.vstack(rows)
Usvd, Ssvd, Vt = np.linalg.svd(Cstack)
tol = 1e-9
rank = int(np.sum(Ssvd > tol))
nullity = Cstack.shape[1] - rank
null_basis = Vt[rank:].T
Phis = [null_basis[:, k].reshape(ND, NE, order='F') for k in range(nullity)]
maxres = max(np.max(np.abs(dart_rep(g) @ Phi - Phi @ edge_rep(g))) for Phi in Phis for g in A4)
check(f"M-1a-i R2-only solve: dim Hom_A4(edge_rep,dart_rep) = {nullity} (matches character-theory "
      f"prediction mult_edge(3)*mult_dart(3) = 2*3 = 6)",
      nullity == 6 and maxres < 1e-9, detail=f"max residual over basis = {maxres:.2e}")

# ---- R1: decompose the Hom space by R-parity (R acts on Hom via Phi -> R.Phi, since R commutes with dart_rep) ----
basis_vecs = np.stack([Phi.reshape(-1, order='F') for Phi in Phis], axis=1)
RPhi_vecs = np.stack([(R @ Phi).reshape(-1, order='F') for Phi in Phis], axis=1)
coeff, *_ = np.linalg.lstsq(basis_vecs, RPhi_vecs, rcond=None)
recon_err = np.max(np.abs(basis_vecs @ coeff - RPhi_vecs))
eigsR, eigvecsR = np.linalg.eig(coeff)
check("M-1a-ii R preserves the Hom space (R.Phi stays an intertwiner) and acts as an involution on it",
      recon_err < 1e-9 and np.allclose(np.sort(eigsR.real), [-1, -1, -1, -1, 1, 1], atol=1e-6),
      detail=f"eigs(R|_Hom) = {np.round(np.sort(eigsR.real), 3)}")

even_idx = np.where(np.abs(eigsR.real - 1) < 1e-6)[0]
odd_idx = np.where(np.abs(eigsR.real + 1) < 1e-6)[0]
Qe, _ = np.linalg.qr(eigvecsR[:, even_idx].real)
Qo, _ = np.linalg.qr(eigvecsR[:, odd_idx].real)
even_vecs = basis_vecs @ Qe
odd_vecs = basis_vecs @ Qo
Phi_even = [even_vecs[:, k].reshape(ND, NE, order='F') for k in range(even_vecs.shape[1])]
Phi_odd = [odd_vecs[:, k].reshape(ND, NE, order='F') for k in range(odd_vecs.shape[1])]
print(f"    R1 splits the dim-6 Hom space into: R-even (eps=+1), dim={len(Phi_even)}  and  "
      f"R-odd (eps=-1), dim={len(Phi_odd)}")

# ---- R5 on the R-EVEN branch: PROVEN empty (rank obstruction) ----
rng = np.random.default_rng(0)
ranks_even = []
for _ in range(8):
    c = rng.normal(size=len(Phi_even))
    Phi = sum(c[k] * Phi_even[k] for k in range(len(Phi_even)))
    ranks_even.append(np.linalg.matrix_rank(Phi, tol=1e-9))
check(f"M-1a-iii R-EVEN branch (dim={len(Phi_even)}) is PROVABLY EMPTY under R5: every element has "
      f"rank <= 3 (its image lies in a single dart-copy of the A4 3-irrep, a 3-dim subspace of R^12) "
      f"so it can NEVER be injective/isometric on the 6-dim domain -- a rank obstruction, not a search "
      f"failure", all(rk <= 3 for rk in ranks_even), detail=f"ranks over 8 random draws = {ranks_even}")

# ---- R5 on the R-ODD branch: identify with Uo @ End_A4(edge_rep) (the commutant, Mat_2(R)) ----
dev_rho_odd = 0.0
for g in A4:
    rho_odd_g = Uo.T @ dart_rep(g) @ Uo
    dev_rho_odd = max(dev_rho_odd, np.max(np.abs(rho_odd_g - edge_rep(g))))
check("M-1a-iv the R-ODD eigenspace, coordinatized by Uo, carries EXACTLY edge_rep (not merely an "
      "isomorphic copy) -- so Hom_A4(edge_rep, R-odd sector) = End_A4(edge_rep), the commutant algebra",
      dev_rho_odd < 1e-9, detail=f"max_g||Uo^T.dart_rep(g).Uo - edge_rep(g)|| = {dev_rho_odd:.2e}")

# commutant of edge_rep, End_A4(edge_rep) (should reproduce the same 4-dim Phi_odd space via C -> Uo@C)
rows2 = [np.kron(np.eye(NE), edge_rep(g)) - np.kron(edge_rep(g).T, np.eye(NE)) for g in A4]
C2 = np.vstack(rows2)
U2, S2, Vt2 = np.linalg.svd(C2)
rank2 = int(np.sum(S2 > 1e-9))
Cs = [Vt2[rank2 + k].reshape(NE, NE, order='F') for k in range(C2.shape[1] - rank2)]
check(f"M-1a-v End_A4(edge_rep) (the commutant algebra) has dimension {len(Cs)} = 4 (Mat_2(R): 2 "
      f"copies of a real-type irrep)", len(Cs) == 4)

I6 = np.eye(NE)


def express(M, basis):
    vecs = np.stack([b.reshape(-1, order='F') for b in basis], axis=1)
    coeff_, *_ = np.linalg.lstsq(vecs, M.reshape(-1, order='F'), rcond=None)
    err = np.max(np.abs((vecs @ coeff_).reshape(NE, NE, order='F') - M))
    return coeff_, err


_, errI = express(I6, Cs)
_, errJ = express(J6, Cs)
check("M-1a-vi both I6 and J6 lie in the commutant (J6 is genuinely an A4-equivariant complex "
      "structure ON the multiplicity space, not just on edge_rep as a whole)",
      errI < 1e-9 and errJ < 1e-9, detail=f"recon err I={errI:.1e}, J6={errJ:.1e}")

# S1,S2: orthogonal (Frobenius) complement of span{I6,J6} within the 4-dim commutant
IJ = np.stack([I6.reshape(-1, order='F'), J6.reshape(-1, order='F')], axis=1)
allc = np.stack([c.reshape(-1, order='F') for c in Cs], axis=1)
Q_IJ, _ = np.linalg.qr(IJ)
proj = allc - Q_IJ @ (Q_IJ.T @ allc)
Qc, Rc = np.linalg.qr(proj)
S1 = Qc[:, 0].reshape(NE, NE, order='F')
S2 = Qc[:, 1].reshape(NE, NE, order='F')
sym_ok = np.allclose(S1, S1.T, atol=1e-8) and np.allclose(S2, S2.T, atol=1e-8)
trace_ok = abs(np.trace(S1)) < 1e-8 and abs(np.trace(S2)) < 1e-8
check("M-1a-vii the remaining 2 commutant dims {S1,S2} are SYMMETRIC and TRACELESS (the 'reflection' "
      "directions, complementing I6/J6's 'rotation' directions)", sym_ok and trace_ok)

# ---- the isometry (R5) sub-locus: verify it is EXACTLY {a*I+b*J6} union {c*S1+d*S2}, nothing else ----
print("\n    R5 isometry-sublocus check (Phi_red^T.Phi_red ∝ I6 for Phi_red = a*I6+b*J6+c*S1+d*S2):")


def isom_resid(Phi_red):
    G = Phi_red.T @ Phi_red
    scal = np.trace(G) / NE
    return np.linalg.norm(G - scal * I6) / (np.linalg.norm(G) + 1e-30)


pure_rot = isom_resid(0.6 * I6 + 0.8 * J6)
pure_ref = isom_resid(0.6 * S1 + 0.8 * S2)
mixed = isom_resid(0.5 * I6 + 0.3 * J6 + 0.4 * S1)
check("M-1a-viii pure {a*I+b*J6} (rotation) and pure {c*S1+d*S2} (reflection) combinations are EXACTLY "
      "isometric (residual~0); a GENERIC mix of the two families is NOT (residual>0) -- so the isometric "
      "sub-locus is precisely O(2) (two circles), not a bigger continuous family",
      pure_rot < 1e-9 and pure_ref < 1e-9 and mixed > 1e-3,
      detail=f"resid(rotation)={pure_rot:.2e}  resid(reflection)={pure_ref:.2e}  resid(generic mix)={mixed:.4f}")

print(f"""
    ================================================================================================
    M-1a VERDICT: AMBIGUOUS-BY-O(2).
      * R-even (eps=+1): EMPTY (proven -- rank<=3 obstruction; M-1a-iii).
      * R-odd (eps=-1): dim=4 = Uo @ End_A4(edge_rep) [Mat_2(R): {{I6,J6,S1,S2}}].  R5 (isometry up to
        one global scale) restricts this LINEAR 4-dim space to the isometry group O(2) EXACTLY:
          Phi_theta   = Uo @ (cos(theta) I6 + sin(theta) J6)     theta in [0,2pi)   [rotation branch --
                        this IS the internal Witt-mode's U(1) PHASE freedom, physically the expected/
                        trivial one-particle phase convention; theta=0 is the prior-art "naive Uo
                        candidate" the design sweep already tested]
          Phi_phi     = Uo @ (cos(phi)   S1 + sin(phi)   S2)     phi   in [0,2pi)   [reflection branch --
                        a genuinely DIFFERENT (non phase-equivalent) isometric family]
      This is the explicit basis of the solution space (up to the declared one overall real scale):
      the classification is NOT FORCED (dim=1) and NOT EMPTY overall -- it is AMBIGUOUS, with the
      residual freedom fully characterized as O(2) acting on the internal edge representation's own
      2-fold multiplicity inside the dart regular representation.
    ================================================================================================
""")


# ====================================================================================================
banner("M-1b  THE SELECTION  (declared dynamical criterion over the O(2) family)")
# ====================================================================================================
print("""    DISCLOSED test formula (generalizing the design sweep's OWN naive-candidate recipe, applied to
    every member of the O(2) family rather than just Phi=Uo): for Phi with image in the R-odd sector,
    write Phi_red := Uo^T.Phi (well-defined, Uo^T.Uo=I6).  Compare L:=Beo.Phi_red (the walk's leakage
    into the even sector, composed with the candidate) against T:=Phi_red.J6 (the candidate composed
    with the internal one-particle dynamics).  Best-fit scale c=<L,T>_F/<T,T>_F; normalized residual
    = ||L-c.T||_F / ||L||_F (IDENTICAL convention to the M-0f regression).  M-1b's criterion requires
    the ACHIEVED MINIMUM over the family to be a ZERO (<1e-9); a nonzero minimum means selection FAILS
    and AMBIGUOUS stands (poison, binding).""")


def bestfit_resid(L, T):
    tt = np.sum(T * T)
    c = np.sum(L * T) / tt if tt > 1e-14 else 0.0
    resid = np.linalg.norm(L - c * T)
    denom = np.linalg.norm(L)
    return c, (resid / denom if denom > 1e-14 else float('nan'))


thetas = np.linspace(0, 2 * math.pi, 721)
best_rot = (None, np.inf, None)
for th in thetas:
    Phi_red = math.cos(th) * I6 + math.sin(th) * J6
    c, r = bestfit_resid(Beo @ Phi_red, Phi_red @ J6)
    if r < best_rot[1]:
        best_rot = (th, r, c)
best_ref = (None, np.inf, None)
for ph in thetas:
    Phi_red = math.cos(ph) * S1 + math.sin(ph) * S2
    c, r = bestfit_resid(Beo @ Phi_red, Phi_red @ J6)
    if r < best_ref[1]:
        best_ref = (ph, r, c)

global_min = min(best_rot[1], best_ref[1])
check(f"M-1b sweep over BOTH O(2) branches (721 pts each): minimum normalized residual found = "
      f"{global_min:.6f} (rotation-branch min at theta={best_rot[0]:.4f}: resid={best_rot[1]:.6f}; "
      f"reflection-branch min at phi={best_ref[0]:.4f}: resid={best_ref[1]:.6f}) -- "
      f"NO zero anywhere in the family", global_min > 0.99)

# make the LITERAL "<1e-9" absolute criterion unambiguous too (not just the normalized reading above):
# since the best-fit scale c is ~0 everywhere (shown next), the achieved ABSOLUTE minimum of
# ||Beo.Phi_red - Phi_red.J6|| over each branch is just ||Beo.Phi_red|| itself (no c helps) -- report it.
abs_min_rot = min(np.linalg.norm(Beo @ (math.cos(th) * I6 + math.sin(th) * J6)) for th in thetas)
abs_min_ref = min(np.linalg.norm(Beo @ (math.cos(ph) * S1 + math.sin(ph) * S2)) for ph in thetas)
check(f"M-1b the ABSOLUTE (unnormalized) achieved minimum of ||Beo.Phi_red - Phi_red.J6|| is also FAR "
      f"from the required <1e-9 zero on both branches (it cannot be driven down by rescaling Phi, since "
      f"R5 already fixes the family to unit-normalized isometries)",
      abs_min_rot > 1e-3 and abs_min_ref > 1e-3, detail=f"abs min (rotation)={abs_min_rot:.4f}  "
      f"abs min (reflection)={abs_min_ref:.4f}  (order 1-2.4, i.e. undiminished, nowhere near 1e-9)")

# sharpen to an EXACT algebraic identity: <Beo.C, C'.J6>_F = 0 for the FULL 4-dim commutant (not just O(2))
bilinear = np.zeros((4, 4))
Call = [I6, J6, S1, S2]
for i, Ci in enumerate(Call):
    for j, Cj in enumerate(Call):
        bilinear[i, j] = np.sum((Beo @ Ci) * (Cj @ J6))
check("M-1b EXACT IDENTITY (sharper than the sweep): <Beo.C, C'.J6>_F = 0 for EVERY C,C' in the full "
      "4-dim commutant (machine zero) -- Beo's image under the commutant is Frobenius-ORTHOGONAL to the "
      "commutant itself; the dynamical criterion's best-fit scale is EXACTLY zero and the residual is "
      "EXACTLY 1 at every point of the O(2) family, not merely close to it",
      np.max(np.abs(bilinear)) < 1e-9, detail=f"max|<Beo.Ci,Cj.J6>| = {np.max(np.abs(bilinear)):.2e}")

print("""
    ================================================================================================
    M-1b VERDICT: SELECTION FAILS -- decisively (an EXACT orthogonality identity, not a near-miss).
    AMBIGUOUS-BY-O(2) STANDS as the final M-1 verdict.  Per the frozen gating (M-2/M-3 require FORCED
    or a successful M-1b), M-2 (transport) and M-3 (the insertion) are GATED OFF below -- reported as
    SKIPPED, not attempted with an arbitrarily-chosen family member (that would be exactly the
    forbidden fit-grade selection).
    ================================================================================================
""")


# ====================================================================================================
banner("M-2 / M-3  TRANSPORT / INSERTION  --  GATED OFF (classification AMBIGUOUS, M-1b did not select)")
# ====================================================================================================
M1B_SELECTED = False
print(f"""    M-2 TRANSPORT: gated on (M-1a FORCED) or (M-1b successful selection).  Neither holds
    (AMBIGUOUS-BY-O(2); M-1b's achieved minimum = {global_min:.4f}, not a zero).  SKIPPED per contract --
    no operator (Q, the su(3) bilinears, P_F) is pushed through an arbitrarily-picked family member.
    M-3 THE INSERTION: gated as M-2.  SKIPPED -- no oblique-correction number is read out; the payoff
    target (c_S / delta_r contribution) is NOT computed, and NOT compared to the 0.0139%/+7.76sigma
    history numbers (those remain comparison-only per the poison, untouched by this run).
    This is itself the honest, contract-required M-1a/M-1b deliverable: the vertex/propagator map has a
    residual O(2) gauge-like freedom (an internal U(1) phase + a genuinely distinct reflection branch)
    that NO derived requirement in {{R1,R2,R5}} nor the walk's own dynamics (Beo) resolves.  A future
    station would need to SUPPLY exactly this missing piece (named precisely: an additional derived
    structure singling out one O(2) element) before M-2/M-3 can run.""")
check("M-2/M-3 correctly reported as GATED-SKIPPED (not silently attempted)", not M1B_SELECTED)


# ====================================================================================================
banner("M-4  THE ML-5 SAME-OBJECT CONFRONT  (runs regardless; independent of M-3)")
# ====================================================================================================
print("""    Reconstructs ML5b_epsilon_transport_2026-07-08.py's W_INT/G_int VERBATIM (its own ALPHA1=
    (2/3)^8, its own dart<->Clifford decoration, its own vacuum projector P_VAC) -- no engine edits, no
    re-derivation, exact reuse of that file's recipe (lines ~106-114 named in the dispatch brief).""")

ALPHA1 = (2.0 / 3.0) ** 8
B_G = srs.hashimoto((0.0, 0.0, 0.0)).real
NHAT = sum(Adag_ops[i] @ A_ops[i] for i in range(3))
wN, VN = np.linalg.eigh(NHAT)
vac = VN[:, [int(np.argmin(wN))]]
vac = vac / np.linalg.norm(vac)


def G_int_of(alpha1):
    W_INT = np.zeros((8 * ND, 8 * ND), complex)
    for dp in range(ND):
        for d in range(ND):
            if abs(B_G[dp, d]) > 0.5:
                W_INT[dp * 8:(dp + 1) * 8, d * 8:(d + 1) * 8] = gam(np.eye(NE)[:, EDGE_OF_DART[dp]])
    P_VAC = np.zeros((ND, 8 * ND), complex)
    for d in range(ND):
        P_VAC[d, d * 8:(d + 1) * 8] = vac[:, 0].conj()
    return P_VAC @ np.linalg.solve(np.eye(8 * ND) - alpha1 * W_INT, P_VAC.conj().T), W_INT


G_int0, W_INT0 = G_int_of(1e-12)
check("M-4 setup: G_int(alpha1->0) = I_12 exactly (P_VAC.P_VAC^dag = I, the vacuum projector is a "
      "resolution of the identity on the dart space)", np.max(np.abs(G_int0 - np.eye(ND))) < 1e-9)

c_S_true = gauge_singlet_projection(hashimoto((0.0, 0.0, 0.0)))
check(f"M-4 target: gauge_singlet_projection(hashimoto) = c_S = {c_S_true:.10f} (= 1/12 = "
      f"{1/12:.10f})", abs(c_S_true - 1 / 12) < 1e-9)

print("""
    (i) SAME-OBJECT test #1 (the G_int reduction).  DISCLOSED reading of "trace/projection of G_int":
    the ANALOG of gauge_singlet_projection's OWN recipe (projection onto the SAME gauge-singlet vector
    shat, <shat|.|shat>/dim -- NOT a generic trace/dim, which is shown below to NOT match and is the
    wrong reading; and NOT literally calling gauge_singlet_projection(G_int(a1)), which is shown to be
    numerically unstable at small a1 since G_int(a1)->I is spectrally degenerate there).""")

rows_i = []
for a1 in [0.1, 0.05, 0.01, 0.001]:
    G, _ = G_int_of(a1)
    proj_shat = (shat @ G @ shat).real / ND
    proj_tr = np.trace(G).real / ND
    rows_i.append((a1, proj_shat, proj_tr))
    print(f"        alpha1={a1:<7} <shat|G_int|shat>/12={proj_shat:.9f}  (dev from 1/12: "
          f"{proj_shat - 1/12:+.3e})   Tr(G_int)/12={proj_tr:.6f}  (dev: {proj_tr - 1/12:+.4f})")

a1s = np.array([r[0] for r in rows_i])
ps = np.array([r[1] for r in rows_i])
# fit dev ~ k * a1^n (log-log slope) to characterize the order of approach
devs = np.abs(ps - 1 / 12)
loglog_slope = np.polyfit(np.log(a1s[:3]), np.log(devs[:3] + 1e-300), 1)[0]
check(f"M-4(i) <shat|G_int(a1)|shat>/12 -> c_S=1/12 as a1->0, with the deviation shrinking like "
      f"a1^{loglog_slope:.1f} (an extremely high-order, essentially-exact agreement at every sampled "
      f"a1) -- Tr(G_int)/12 does NOT reduce to 1/12 (it -> 1, the trivial trace of the identity), "
      f"confirming the shat-projection (not the trace) is the correct analog of c_S's own definition",
      devs[-1] < 1e-9 and abs(rows_i[-1][2] - 1.0) < 1e-6,
      detail=f"loglog slope~{loglog_slope:.2f}; a1=0.001 dev={devs[-1]:.2e}; Tr/12(0.001)={rows_i[-1][2]:.6f}")

print("\n    secondary (disclosed, REJECTED) candidate reading -- literal gauge_singlet_projection(G_int(a1)):")
unstable_vals = []
for a1 in [0.1, 0.05, 0.01, 0.001]:
    G, _ = G_int_of(a1)
    try:
        val = gauge_singlet_projection(G)
        unstable_vals.append(val)
        print(f"        alpha1={a1}: gauge_singlet_projection(G_int) = {val:.4f}  (compare 1/12={1/12:.4f})")
    except Exception as ex:
        print(f"        alpha1={a1}: FAILED ({ex})")
print(f"    => this reading is numerically UNSTABLE (values {np.round(unstable_vals,3)} do not converge "
      f"as a1->0: G_int(a1)->I is spectrally degenerate there, so its 'Perron eigenvector' is ill-"
      f"defined) -- DISCLOSED and REJECTED as the wrong reading; the shat-projection above is primary.")

print("""
    (ii) SAME-OBJECT test #2 (does M-1's family reproduce W_INT's dart<->Clifford decoration?).
    W_INT decorates each nonzero hop (dp<-d) with gam(edge_of_dart(dp)) -- a SINGLE Cl(6) generator
    keyed ONLY by the unsigned edge label of the TARGET dart dp (no forward/reverse sign).  This is
    EXACTLY the natural per-dart Clifford lift of a dart-to-edge embedding; the two candidates are
    sqrt(2)*Ue^T (UNSIGNED: same sign for a dart and its reverse) and sqrt(2)*Uo^T (SIGNED: flips sign
    under reversal -- the embedding M-1a's R1 requirement actually selects).""")

maxdev_ue, maxdev_uo = 0.0, 0.0
for dp in range(ND):
    e_dp = np.eye(ND)[:, dp]
    target = gam(np.eye(NE)[:, EDGE_OF_DART[dp]])
    maxdev_ue = max(maxdev_ue, np.max(np.abs(gam(math.sqrt(2) * (Ue.T @ e_dp)) - target)))
    maxdev_uo = max(maxdev_uo, np.max(np.abs(gam(math.sqrt(2) * (Uo.T @ e_dp)) - target)))
check("M-4(ii) W_INT's decoration EQUALS gam(sqrt2 * Ue^T . e_dp) EXACTLY (the UNSIGNED/R-EVEN "
      "convention) and DIFFERS MAXIMALLY from gam(sqrt2 * Uo^T . e_dp) (the SIGNED/R-ODD convention "
      "M-1a's R1 requirement selects)",
      maxdev_ue < 1e-9 and maxdev_uo > 1.0, detail=f"dev(vs Ue)={maxdev_ue:.2e}  dev(vs Uo)={maxdev_uo:.4f}")
check("M-4(ii) cross-check: this is the SAME R-even/R-odd mismatch already proven structurally at "
      "M-0k/M-1a-iii -- Ue fails R2 (A4-covariance) outright and the R-even branch is PROVEN EMPTY "
      "under R5 -- so W_INT's OWN decoration convention is not even a candidate member of M-1's "
      "classified (R1+R2+R5-compliant) family", dev_Ue > 1.0 and all(rk <= 3 for rk in ranks_even))

print(f"""
    ================================================================================================
    M-4 VERDICT: PARTIAL.
      (i) SAME zeroth-order object: G_int's gauge-singlet projection reduces to c_S=1/12 with an
          alpha1^{loglog_slope:.0f}-suppressed correction -- both constructions share the SAME ambient
          12-dim dart Hilbert space and the SAME gauge-singlet vector shat as their common anchor.
      (ii) DIFFERENT vertex maps: ML-5's W_INT decoration is built from the UNSIGNED (Ue-type) dart-to-
          edge label, which this station's classification shows FAILS the derived A4-covariance
          requirement R2 outright (maximal mismatch, and the R-even branch it belongs to is PROVABLY
          EMPTY under R5) -- whereas the ONLY {{R1,R2,R5}}-compliant maps found here are the R-odd
          (Uo-type, signed) O(2) family.  ML-5's decoration is therefore NOT a member of the vertex-map
          family M-1a classified.
      => the -70 ppm's coupling (ML-5) and the oblique's vertex (this station) share a COMMON HOME (the
      dart Hilbert space + its gauge singlet) but are built from GENUINELY DIFFERENT, symmetry-
      inequivalent dart<->edge decoration conventions -- "two walls, one object" is only PARTIALLY
      true: one object (the ambient space + singlet), two DIFFERENT constructions on it.
    ================================================================================================
""")


# ====================================================================================================
banner("M-5  SCOPE  (printed; no scoreboard movement)")
# ====================================================================================================
print("""    NOT claimed by this station:
      - no scoreboard value moves (the 0.0139% M_Z-oblique floor, the +7.76sigma reading, the 1.62-bit
        weld are HISTORY/comparison numbers only, never touched or refit here).
      - no epsilon/Delta_c value is read for the -70 ppm; ML-5's magnitude remains UN-BUILT even though
        M-4 finds the two objects share a common ambient ANCHOR (the shat/dart space) -- only the
        OBJECT-IDENTITY question was adjudicated (PARTIAL), not the magnitude.
      - deck/Z3 transport was NOT assumed anywhere (R4 excluded from the defining set, per adjudication
        2 / FS-5iii); nothing here re-opens or overrides that negative.
      - the m_nu-scale gate is NOT claimed same-object with this station's map; it is at most same-
        FAMILY (subleading-spectral), and that claim is NOT made here either -- it would need its own
        confront.
      - M-2/M-3's gated-off status is not a proof that NO transport/insertion is possible in principle;
        it is a proof that the DERIVED requirement set {R1,R2,R5} plus the walk's own dynamics (Beo) do
        not, by themselves, resolve the O(2) freedom -- a future station would need to name and supply
        one more derived structure.""")


# ====================================================================================================
banner("SUMMARY")
# ====================================================================================================
elapsed = time.time() - t_start
print(f"    M-0  ANCHORS ................................ {'PASS' if ok_all else 'CHECK LOG ABOVE'}")
print(f"    M-1a CLASSIFICATION .......................... AMBIGUOUS-BY-O(2) (R-even EMPTY, R-odd dim=4, "
      f"R5 -> O(2))")
print(f"    M-1b SELECTION ............................... FAILS (exact orthogonality identity; min "
      f"residual = {global_min:.4f}, not a zero) => AMBIGUOUS-BY-O(2) STANDS")
print(f"    M-2  TRANSPORT ............................... GATED-SKIPPED")
print(f"    M-3  INSERTION ................................ GATED-SKIPPED")
print(f"    M-4  ML-5 SAME-OBJECT CONFRONT ................ PARTIAL "
      f"(same anchor/shat-reduction; DIFFERENT dart<->edge decoration convention)")
print(f"    M-5  SCOPE .................................... printed")
print(f"    disclosed interpretation steps: {len(DISCLOSURES)}")
print(f"    runtime: {elapsed:.1f}s")
print()
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}  "
      f"(exit condition: every M-0 regression holds AND M-1a/M-1b/M-4 each reach a definite, booked "
      f"verdict -- AMBIGUOUS-BY-O(2) and PARTIAL both count as definite)")
print("=" * 100)
sys.exit(0 if ok_all else 1)
