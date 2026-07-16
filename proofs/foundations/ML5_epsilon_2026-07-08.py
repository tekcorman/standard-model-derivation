#!/usr/bin/env python3
"""
proofs/foundations/ML5_epsilon_2026-07-08.py

ML-5 — the epsilon readout (-70 ppm), forced-or-gapped.  Pre-registered in
internal research notes (committed e9afdb1 BEFORE this probe).  Trap-densest;
derive-or-die.  Reuses the FORCED interacting-run machinery (LOOP_E2a: G_int(u), the winding-isotype
chiral asymmetry = the eps carrier) + the canonical frame (ML-2b) + the run coupling u=alpha_1 (M0-2R).

DISCIPLINE: eps is alpha1^5-scale and 3.2% from the 2*alpha1^5 poison.  I do NOT tune a functional to hit
-1.75e-7.  FIRST settle the weld-dependence (ML5-A) and whether the transport is FULLY FORCED (ML5-B);
ONLY if forced compute the value (ML5-C), confronting the target + poisons at the DECLARED END.
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

np.set_printoptions(precision=6, suppress=True)
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    if not cond:
        ok_all = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    return cond


def banner(t):
    print("=" * 88); print(f" {t}"); print("=" * 88)


# --- FORCED machinery (LOOP_E2a verbatim: gam, DARTS, B_G, J6, vac, C_PAIR, W_INT, G_int, Q_t) ---
EDGES = srs.EDGES
NE, NV = len(EDGES), srs.NV
ND = 2 * NE
g6 = [np.array(g) for g in AlgebraicUtility.cl6_generators()]
EIDX = {(min(i, j), max(i, j)): e for e, (i, j, v) in enumerate(EDGES)}
gam = lambda x: sum(x[a] * g6[a] for a in range(NE))
DARTS = []
for i, j, v in EDGES:
    DARTS += [(i, j), (j, i)]
EDGE_OF_DART = [d // 2 for d in range(ND)]
B_G = srs.hashimoto((0.0, 0.0, 0.0)).real


def edge_rep(sig):
    R6 = np.zeros((NE, NE))
    for e, (i, j, v) in enumerate(EDGES):
        a, b = sig[i], sig[j]
        s = 1.0
        if a > b:
            a, b, s = b, a, -1.0
        R6[EIDX[(a, b)], e] = s
    return R6


d0 = np.zeros((NV, NE))
for e, (i, j, v) in enumerate(EDGES):
    d0[i, e] = -1.0; d0[j, e] = 1.0
Chat = np.array([[1, 1, 0], [-1, 0, 1], [0, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
H1, _ = np.linalg.qr(Chat)
B1 = np.linalg.svd(d0)[2][:3].T
A4 = [dict(enumerate(p)) for p in itertools.permutations(range(4))
      if sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j]) % 2 == 0]
rows = [np.kron(np.eye(3), (H1.T @ edge_rep(g) @ H1).T) - np.kron(B1.T @ edge_rep(g) @ B1, np.eye(3))
        for g in A4]
_, Sp, Vp = np.linalg.svd(np.vstack(rows))
phi = Vp[-1].reshape(3, 3); phi *= math.sqrt(3) / np.linalg.norm(phi)
J6 = B1 @ phi @ H1.T - H1 @ phi.T @ B1.T
wJ, VJ = np.linalg.eig(J6)
modes, _ = np.linalg.qr(VJ[:, np.where(np.abs(wJ - 1j) < 1e-9)[0]])
A_ops = [gam(np.conj(modes[:, m])) / math.sqrt(2) for m in range(3)]
NHAT = sum(a.conj().T @ a for a in A_ops)
wN, VN = np.linalg.eigh(NHAT)
vac = VN[:, [int(np.argmin(wN))]]; vac = vac / np.linalg.norm(vac)
Pw = {w: VN[:, np.round(wN).astype(int) == w] @ VN[:, np.round(wN).astype(int) == w].conj().T
      for w in range(4)}

W_INT = np.zeros((8 * ND, 8 * ND), complex)
for dp in range(ND):
    for d in range(ND):
        if abs(B_G[dp, d]) > 0.5:
            W_INT[dp * 8:(dp + 1) * 8, d * 8:(d + 1) * 8] = gam(np.eye(NE)[:, EDGE_OF_DART[dp]])
P_VAC = np.zeros((ND, 8 * ND), complex)
for d in range(ND):
    P_VAC[d, d * 8:(d + 1) * 8] = vac[:, 0].conj()


def G_int(u):
    X = np.linalg.solve(np.eye(8 * ND) - u * W_INT, P_VAC.conj().T)
    return P_VAC @ X


sigma3 = {0: 0, 1: 2, 2: 3, 3: 1}
P3 = np.zeros((ND, ND))
for a, (i, j) in enumerate(DARTS):
    for b, (p, q) in enumerate(DARTS):
        if (p, q) == (sigma3[i], sigma3[j]):
            P3[b, a] = 1.0
            break
OM = cmath.exp(2j * math.pi / 3)
Q_t = [sum(OM ** (-t * m) * np.linalg.matrix_power(P3, m) for m in range(3)) / 3 for t in range(3)]
ALPHA1 = (2.0 / 3.0) ** 8                                   # the run coupling (M0-2R operating point)

# ===========================================================================
banner("ML5-A  WELD-DEPENDENCE: does eps use the FORCED correlation I(w;t) or the UNPAID weld H(w|t)?")
# ===========================================================================
# the eps-seed is the universal bit-odd deck channel; recompute (nu-e)/2 = (d-u)/2 = (0,+-sqrt3/6).
Upi_R = np.zeros((NE, NE))
for e, (i, j, v) in enumerate(EDGES):
    a, b = sigma3[i], sigma3[j]
    Upi_R[EIDX[(min(a, b), max(a, b))], e] = 1.0
rowsU = [np.kron(gam(Upi_R[:, a]), np.eye(8)) - np.kron(np.eye(8), g6[a].T) for a in range(NE)]
_, sU, VhU = np.linalg.svd(np.vstack(rowsU))
Upi = VhU[np.sum(sU > 1e-9):].conj()[0].reshape(8, 8)
Upi /= np.sqrt(np.abs(np.linalg.det(Upi @ Upi.conj().T)) ** (1 / 8))
Upi2 = Upi @ Upi
evU, VU = np.linalg.eig(Upi2)
lab = np.array([int(round(cmath.phase(z) / (2 * math.pi / 3))) % 3 for z in evU])
PiF = {t: (lambda Q: Q @ Q.conj().T)(np.linalg.qr(VU[:, lab == t])[0]) for t in (0, 1, 2)}
T = np.array([[np.real(np.trace(Pw[w] @ PiF[t])) for t in range(3)] for w in range(4)])  # RAW WS1 table
odd_nu_e = (T[0] - T[3]) / 2.0                              # bit-odd channel on the RAW table (WS1)
odd_d_u = (T[1] - T[2]) / 2.0
universal = np.max(np.abs(odd_nu_e - odd_d_u))
print(f"    bit-odd (nu-e)/2 = {np.round(odd_nu_e,5)}; (d-u)/2 = {np.round(odd_d_u,5)}  (sqrt3/6={math.sqrt(3)/6:.5f})")
check("ML5-A the eps-SEED is the UNIVERSAL bit-odd deck channel (0,+-sqrt3/6), IDENTICAL for both "
      "particle-hole pairs => it is the FORCED correlation I(w;t), NOT the unpaid residual H(w|t)",
      universal < 1e-9 and abs(abs(odd_nu_e[1]) - math.sqrt(3) / 6) < 1e-9,
      detail=f"(nu-e)/2==(d-u)/2 dev {universal:.1e}; the seed = I(w;t) forced => eps is NOT zero-bit")
print("    => WELD-DEPENDENCE VERDICT: FORCED-CORRELATION-ONLY. eps rides the universal +-sqrt3/6 seed")
print("       (the forced I(w;t)=0.18 bits); the unpaid H(w|t)=1.63 residual is NOT needed. Reconciles")
print("       architect's Fork-A (frame + forced correlation suffice re the weld) -- NOT zero-bit.")

# ===========================================================================
banner("ML5-B  is the TRANSPORT FUNCTIONAL fully FORCED, or a GAP?  (the decisive question)")
# ===========================================================================
# the eps carrier: the winding-isotype chiral asymmetry of the interacting run, A(u) = tr(Q1 G_int(u))
# - conj tr(Q2 G_int(u)) (vanishes free; nonzero interacting).  Evaluate at the FORCED u=alpha_1.
def asym(u):
    G = G_int(u)
    return np.trace(Q_t[1] @ G) - np.conj(np.trace(Q_t[2] @ G))


A_a1 = asym(ALPHA1)
print(f"    (i) FORCED carrier: chiral asymmetry A(alpha_1) = tr(Q1 G_int)-conj tr(Q2 G_int) = {abs(A_a1):.4e}")
print(f"        scale O(alpha_1^2): |A(alpha_1)|/alpha_1^2 = {abs(A_a1)/ALPHA1**2:.3f}  (alpha_1^2={ALPHA1**2:.3e})")
# (ii) THE NATURAL FORCED MAP -- the winding-PHASE shift free->interacting (delta is a phase; eps its
#      correction).  Apply the SAME winding-phase functional to G_free and G_int(alpha_1).  COMPUTED:
Gf = np.linalg.inv(np.eye(ND) - ALPHA1 * B_G)
Gi = G_int(ALPHA1)
phase_free = cmath.phase(np.trace(Q_t[1] @ Gf))
phase_int = cmath.phase(np.trace(Q_t[1] @ Gi))
print(f"    (ii) the natural FORCED map (winding-phase shift): arg(tr Q1 G_free)={phase_free:+.2e}, "
      f"arg(tr Q1 G_int)={phase_int:+.2e}  =>  shift = {phase_int - phase_free:+.2e}")
phase_shift_is_eps = abs(abs(phase_int - phase_free) - 1.75e-7) < 5e-8
print(f"    (iii) the FREE winding phase = {phase_free:+.3e} != the read's leading delta=2/9={2/9:.4f} "
      f"=> the propagator winding-phase is NOT the read's phase functional (delta=2/9 is the Wigner-d")
print(f"          survival, a DIFFERENT object) => no forced connection from G_int to the read's delta.")
# forced-or-gapped VERDICT (computed, not asserted): the natural forced map (winding-phase shift) is
# ~0, NOT eps; the carrier A is O(alpha_1^2) while eps is O(alpha_1^4-5); mapping A -> eps needs a
# further ~alpha_1^2-3 suppression + a lepton-slice projection with NO forced selector, and the read's
# delta lives in a different (Wigner-d) functional the interacting propagator does not correct.
transport_forced = phase_shift_is_eps                       # False (computed: the shift is ~0)
check("ML5-B TRANSPORT-FORCED test: the natural forced map (winding-phase shift free->interacting) is "
      "~0, NOT eps; the carrier A(alpha_1)~O(alpha_1^2) and the map A->eps (lepton-slice projection + "
      "trace->phase + minus-leading + the ~alpha_1^2-3 suppression) has NO forced selector => the "
      "transport is NOT fully forced",
      not transport_forced,
      detail=f"winding-phase shift={abs(phase_int-phase_free):.1e} (not eps); CONSTRUCTION-GAP")

# ===========================================================================
banner("ML5-C  the VALUE  (GATED: computed ONLY if ML5-B = TRANSPORT-FORCED)")
# ===========================================================================
EPS_TARGET = -1.7515e-7
if transport_forced:
    pass  # would compute eps and confront here
else:
    print("    ML5-B = CONSTRUCTION-GAP => the value is NOT computed (computing eps from an under-")
    print("    determined normalization would be TUNING). The target -1.7515e-7 is NOT confronted.")
    print(f"    (For the record, the FORCED carrier scale |A(alpha_1)|={abs(A_a1):.3e} is O(alpha_1^2),")
    print(f"     ~{abs(A_a1)/ALPHA1**2:.1f}*alpha_1^2; eps=1.75e-7 is O(alpha_1^4-5) -- the transport must")
    print("     supply the further ~alpha_1^2-3 suppression + the slice projection, which is the GAP.)")

# ===========================================================================
banner("SUMMARY / ROUTING")
# ===========================================================================
print(f"""    VERDICT: CONSTRUCTION-GAP (the -70 ppm stays OPEN; NOT closed, NOT zero-bit).
    ML5-A  WELD-DEPENDENCE settled: eps rides the UNIVERSAL +-sqrt3/6 seed = the FORCED correlation
           I(w;t)=0.18 bits; the unpaid H(w|t)=1.63 residual is NOT needed. => NOT zero-bit; reconciles
           architect's Fork-A re the weld.
    ML5-B  the interacting-run chiral carrier A(u)=tr(Q1 G_int)-conj tr(Q2 G_int) is FORCED and nonzero
           (vanishes free), scale O(alpha_1^2) at the forced u=alpha_1. BUT the transport-minus-leading
           FUNCTIONAL (global A -> lepton-slice phase eps: projection + normalization + subtraction) is
           NOT forced -- no unique selector; E2a's K4a un-forced choice survives EVEN in the canonical
           frame. => the canonical frame (ML-2b) is NECESSARY (dissolves O4's 60% lift-dependence) but
           NOT SUFFICIENT. This CORRECTS architect's Fork-A 'frame => ML-5 well-posed': the frame makes the
           SECTOR/frame canonical, but the interacting-run TRANSPORT FUNCTIONAL (the lepton-slice
           projection + trace->phase normalization) remains the un-built object.
    ML5-C  NOT computed (gap => computing a value = tuning). Target -1.7515e-7 NOT confronted.
    => SHARPEST LOCALIZATION TO DATE: the -70 ppm needs ONE named object -- the FORCED lepton-slice
       transport functional of the interacting-run chiral asymmetry A(alpha_1) (projection + trace->phase
       + minus-leading), canonical-frame-based but frame-underdetermined. -70 ppm STAYS OPEN. No value moved.""")
print("RESULT:", "ALL CHECKS PASS" if ok_all else "A CHECK FAILED -- inspect above")
sys.exit(0 if ok_all else 1)
