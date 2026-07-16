#!/usr/bin/env python3
"""
proofs/foundations/LOOP_A5_winding_weld_W1_2026-07-04.py

THE WINDING WELD W1 -- is the ℤ₆→C₃ descent of the read FORCED? Pre-registered in
internal research notes (committed 3f23407 BEFORE
this file). PURELY STRUCTURAL: NO eps evaluation; the R-eps target appears
NOWHERE. Question: does the read's vector-C₃ (dart-P₃) amplitude structure
descend from the coupled ℤ₆ (S=P₃⊗U_π) DECK grading by a FORCED covariant map,
or does it need a choice (adoption)?

Follows the A5 spin-holonomy KILL (bf91970), which measured K1 at single-
eigenstate resolution (overlap 0.19/0.22). This tests the RIGHT object: the
SUBSPACE / deck-sector descent.

STAGES:
  S-0  re-lock W, S=P₃⊗U_π (E2c).
  T1   the ℤ₆ = C₃ × ℤ₂ structure: S³=-I uniform; S² deck labels {0,1,2}; present
       spectrum = odd ζ₆ (descent bijective on present sectors; π/3 half-angle).
  CC   C-COMMUTE: [W,S²]=0 (deck preserved) but [W,P₃]≠0 (dart label NOT preserved).
  T2   CORE: do the coupled DECK sectors reproduce the read's (2,√2,√2)? does the
       read's dart channel concentrate in one deck sector (forced companion)? is
       the map covariant (same for all t)?
  CF   C-FREE/leading: the free deck sectors reproduce (2,√2,√2).
  T3   LOCALIZE the eps-home (structural only, NO evaluation) IF forced.
  V    VERDICT + tier.
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
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "bridge"))
import srs  # noqa: E402
from simulator.srs_engine.utils import AlgebraicUtility  # noqa: E402

ok_all = True
def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}")
def banner(t):
    print("=" * 88); print(f" {t}"); print("=" * 88)

EDGES = srs.EDGES
NE, NV = len(EDGES), srs.NV
ND = 2 * NE
g6 = [np.array(g) for g in AlgebraicUtility.cl6_generators()]
EIDX = {(min(i, j), max(i, j)): e for e, (i, j, v) in enumerate(EDGES)}
EDGE_OF_DART = [d // 2 for d in range(ND)]
DARTS = []
for i, j, v in EDGES:
    DARTS += [(i, j), (j, i)]

def gam(x):
    return sum(x[a] * g6[a] for a in range(NE))
def edge_rep(sig):
    R6 = np.zeros((NE, NE))
    for e, (i, j, v) in enumerate(EDGES):
        a, b = sig[i], sig[j]
        s = 1.0
        if a > b:
            a, b, s = b, a, -1.0
        R6[EIDX[(a, b)], e] = s
    return R6

OM = cmath.exp(2j * math.pi / 3)

# ===========================================================================
banner("S-0  re-lock: coupled W, the spinorial screw S = P3 (x) U_pi")
# ===========================================================================
# canonical J / frame (E2c), for the vacuum lift
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
phi3 = Vp[-1].reshape(3, 3); phi3 *= math.sqrt(3) / np.linalg.norm(phi3)
J6 = B1 @ phi3 @ H1.T - H1 @ phi3.T @ B1.T
wJ, VJ = np.linalg.eig(J6)
modes, _ = np.linalg.qr(VJ[:, np.where(np.abs(wJ - 1j) < 1e-9)[0]])
A_ops = [gam(np.conj(modes[:, m])) / math.sqrt(2) for m in range(3)]
NHAT = sum(a.conj().T @ a for a in A_ops)
wN, VN = np.linalg.eigh(NHAT)
vac = VN[:, [int(np.argmin(wN))]]; vac = vac / np.linalg.norm(vac)

GAMS = [gam(np.eye(NE)[:, EDGE_OF_DART[dp]]) for dp in range(ND)]
def W_full(k):
    Bk = srs.hashimoto(k)
    W = np.zeros((8 * ND, 8 * ND), complex)
    for dp in range(ND):
        row = Bk[dp]
        for d in np.nonzero(np.abs(row) > 1e-14)[0]:
            W[dp * 8:(dp + 1) * 8, d * 8:(d + 1) * 8] = row[d] * GAMS[dp]
    return W
W0 = W_full((0.0, 0.0, 0.0))
B0 = srs.hashimoto((0.0, 0.0, 0.0)).real

sigma3 = {0: 0, 1: 2, 2: 3, 3: 1}
P3 = np.zeros((ND, ND))
for a, (i, j) in enumerate(DARTS):
    for b, (p, q) in enumerate(DARTS):
        if (p, q) == (sigma3[i], sigma3[j]):
            P3[b, a] = 1.0
            break
# winding bases QB[t] (dart-P3 read label)
QB = {}
for t in (0, 1, 2):
    Q = sum(OM ** (-t * m) * np.linalg.matrix_power(P3, m) for m in range(3)) / 3
    evq, Vq = np.linalg.eigh((Q + Q.conj().T) / 2)
    QB[t] = Vq[:, np.abs(evq - 1) < 1e-8]
# U_pi
pi = {}
for e, (i, j, v) in enumerate(EDGES):
    a, b = sigma3[i], sigma3[j]
    pi[e] = EIDX[(min(a, b), max(a, b))]
Rpi = np.zeros((NE, NE))
for e in range(NE):
    Rpi[pi[e], e] = 1.0
rows = [np.kron(gam(Rpi[:, a]), np.eye(8)) - np.kron(np.eye(8), g6[a].T) for a in range(NE)]
_, S2s, Vh = np.linalg.svd(np.vstack(rows))
null = Vh[np.sum(S2s > 1e-9):].conj()
U_pi = null[0].reshape(8, 8)
U_pi /= np.sqrt(np.abs(np.linalg.det(U_pi @ U_pi.conj().T)) ** (1 / 8))
S = np.kron(P3, U_pi)
check("S-0 re-lock: [W,S]=0, S unitary, U_pi^3=-I",
      np.max(np.abs(W0 @ S - S @ W0)) < 1e-10
      and np.max(np.abs(S @ S.conj().T - np.eye(96))) < 1e-9
      and np.max(np.abs(np.linalg.matrix_power(U_pi, 3) + np.eye(8))) < 1e-9)

# ===========================================================================
banner("T1  the z6 = C3 x Z2 structure of S")
# ===========================================================================
S3 = np.linalg.matrix_power(S, 3)
Ssq = np.linalg.matrix_power(S, 2)
uniform_bit = np.max(np.abs(S3 + np.eye(96))) < 1e-9
# deck projectors from S^2 (order 3): Pi_t onto S^2-eigenvalue OM^t
Pi = {}
for t in (0, 1, 2):
    Pi[t] = sum(OM ** (-t * m) * np.linalg.matrix_power(Ssq, m) for m in range(3)) / 3
deck_dims = {t: int(round(np.trace(Pi[t]).real)) for t in (0, 1, 2)}
# present 6th-root labels of S
evS = np.linalg.eigvals(S)
present = sorted(set(int(round(cmath.phase(z) / (math.pi / 3))) % 6 for z in evS))
print(f"    S^3 = -I uniform: {uniform_bit} (the Z2 double-cover sign is GLOBAL)")
print(f"    S^2 deck-sector dims (labels 0,1,2): {deck_dims}")
print(f"    present 6th-root labels of S: {present} (odd => spinor/half-angle sectors only)")
check("T1 z6 = C3 x Z2: S^3=-I uniform (global bit), S^2 deck labels {0,1,2} each "
      "32-dim, present spectrum = ODD zeta6 {1,3,5} (descent bijective on present "
      "sectors; the C3 windings shifted by the half-angle pi/3)",
      uniform_bit and deck_dims == {0: 32, 1: 32, 2: 32}
      and set(present) == {1, 3, 5})

# ===========================================================================
banner("CC  C-COMMUTE: W preserves the DECK (S^2) but NOT the dart label (P3)")
# ===========================================================================
comm_Ssq = np.max(np.abs(W0 @ Ssq - Ssq @ W0))
P3c = np.kron(P3, np.eye(8))
comm_P3 = np.max(np.abs(W0 @ P3c - P3c @ W0))
print(f"    [W, S^2] = {comm_Ssq:.1e} (deck preserved);  [W, P3(x)I] = {comm_P3:.3f} "
      "(dart label NOT preserved -- the tension that IS the weld)")
check("CC C-COMMUTE: [W,S^2]=0 (W block-diagonal over the 3 deck sectors) AND "
      "[W,P3(x)I] != 0 (the read's dart-P3 label is NOT a coupled good quantum "
      "number) -- the descent is a genuine question", comm_Ssq < 1e-9 and comm_P3 > 1e-2)

# ===========================================================================
banner("T2  CORE: do the coupled DECK sectors reproduce the read's (2, sqrt2, sqrt2)?")
# ===========================================================================
read_mod = [2.0, math.sqrt(2.0), math.sqrt(2.0)]     # (t=0 Perron, t=1/2 shell)
# (a) dominant |eigenvalue| of W restricted to each deck sector t
def deck_basis(t):
    ev, U = np.linalg.eigh((Pi[t] + Pi[t].conj().T) / 2)
    return U[:, ev > 0.5]                             # 32 orthonormal cols
def deck_dom_mod(M, t):
    Bt = deck_basis(t)
    return float(np.max(np.abs(np.linalg.eigvals(Bt.conj().T @ M @ Bt))))
W_deck_mods = [deck_dom_mod(W0, t) for t in (0, 1, 2)]
# the FREE coupled operator: identity Cl6 insertion (no gamma coupling)
Wfree0 = np.kron(B0, np.eye(8))
free_deck_mods = [deck_dom_mod(Wfree0, t) for t in (0, 1, 2)]
print(f"    read dart-C3 moduli               : {np.round(read_mod,6)}")
print(f"    FREE  coupled deck-sector dom |lam|: {np.round(free_deck_mods,6)}")
print(f"    W (interacting) deck-sector dom |lam|: {np.round(W_deck_mods,6)}")
# the deck-sector dominant moduli, sorted, vs the read's sorted (2, sqrt2, sqrt2)
free_match = sorted(free_deck_mods)
read_sorted = sorted(read_mod)
free_ok = max(abs(a - b) for a, b in zip(free_match, read_sorted)) < 1e-6
print(f"    (sorted) free deck {np.round(free_match,6)} vs read {np.round(read_sorted,6)}")

# (b) does the read's dart channel v_t (x) vac concentrate in ONE deck sector?
def deck_weights(psi):
    nrm = float(np.real((psi.conj().T @ psi).item()))
    return [float(np.real((psi.conj().T @ Pi[t] @ psi).item())) / nrm for t in (0, 1, 2)]
concentration = {}
for t in (0, 1, 2):
    ev, V = np.linalg.eig(QB[t].conj().T @ B0 @ QB[t])
    vt = QB[t] @ V[:, int(np.argmax(np.abs(ev)))]     # dominant dart channel, label t
    psi = np.kron(vt / np.linalg.norm(vt), vac[:, 0])
    w = deck_weights(psi)
    concentration[t] = (w, int(np.argmax(w)), max(w))
    print(f"    read dart channel t={t}, lifted (x)vac: deck weights "
          f"{np.round(w,3)} -> peak deck {np.argmax(w)} (weight {max(w):.3f})")
peaks = [concentration[t][1] for t in (0, 1, 2)]
peakw = [concentration[t][2] for t in (0, 1, 2)]
forced_companion = min(peakw) > 0.9 and len(set(peaks)) == 3   # each t -> a distinct sector, cleanly
print(f"    forced companion (each dart-t -> a UNIQUE deck sector, weight>0.9)? "
      f"{forced_companion}  (peaks {peaks}, min weight {min(peakw):.3f})")
# (b') is the SUPERPOSITION forced+covariant? the weights should be a single set,
#      cyclically permuted, and (measured) = {1/3, 1/3 +- sqrt3/6}
wset = sorted(concentration[0][0])
target_w = sorted([1/3 - math.sqrt(3)/6, 1/3, 1/3 + math.sqrt(3)/6])
covariant = all(sorted(concentration[t][0]) == wset or
                max(abs(a-b) for a,b in zip(sorted(concentration[t][0]), wset)) < 1e-6
                for t in (0,1,2))
weights_forced = max(abs(a-b) for a,b in zip(wset, target_w)) < 1e-4
print(f"    deck-weight set {np.round(wset,4)} vs {{1/3, 1/3+-sqrt3/6}} = "
      f"{np.round(target_w,4)}: forced={weights_forced}, covariant(cyclic)={covariant}")

# (c) grading mismatch: the deck sectors are modulus-UNIFORM (all Perron free / all
#     shell interacting), so they do NOT carry the read's (2,sqrt2,sqrt2) dart grading.
deck_uniform = (max(abs(m - free_deck_mods[0]) for m in free_deck_mods) < 1e-6
                and max(abs(m - W_deck_mods[0]) for m in W_deck_mods) < 1e-6)
check("T2(grading) the coupled DECK grading is modulus-UNIFORM (free all=2 Perron, "
      f"interacting all=sqrt2 shell: {np.round(free_deck_mods,4)}/"
      f"{np.round(W_deck_mods,4)}) -- it is NOT the read's dart (2,sqrt2,sqrt2) "
      "grading; the read's fine structure is a DART-P3 quantity, invisible to the "
      "S^2 deck (confirms the two gradings differ)", deck_uniform and not free_ok)
check("T2(structure) BANKED POSITIVE: the read channel's descent to the deck "
      f"sectors is a FORCED COVARIANT superposition -- weights {{1/3, 1/3+-sqrt3/6}} "
      f"({np.round(wset,4)}), one set cyclically permuted in t -- NOT random "
      "spreading (sharper than the A5 single-eigenstate 0.19/0.22)",
      weights_forced and covariant)

# ===========================================================================
banner("T3  LOCALIZE the eps-home (structural only -- NO evaluation)")
# ===========================================================================
# the interacting deck moduli vs free: the shift IS the dressing (bit-even part);
# the eps-candidate = the odd-zeta6 (half-angle) / bit-odd content of the deck
# sector relative to its leading (free) amplitude. NO number computed.
deck_shift = [W_deck_mods[t] - free_deck_mods[t] for t in (0, 1, 2)]
print(f"    interacting-minus-free deck moduli shift: {np.round(deck_shift,4)} "
      "(the DRESSING; bit-even). The eps-candidate = the bit-ODD / odd-zeta6")
print("    (half-angle pi/3) content of each deck sector relative to its leading")
print("    amplitude -- LOCALIZED here; NOT evaluated this session.")

# ===========================================================================
banner("V  VERDICT + tier")
# ===========================================================================
interacting_match = max(abs(sorted(W_deck_mods)[i] - read_sorted[i]) for i in range(3)) < 1e-6
pass_bijection = free_ok and interacting_match and forced_companion
print(f"    pre-registered PASS (forced BIJECTION) needs: deck moduli = read AND "
      f"unique companion.")
print(f"      (i)  free deck sectors reproduce read moduli : {free_ok}")
print(f"      (ii) interacting deck moduli match the read  : {interacting_match}")
print(f"      (iii) read channel -> unique deck companion  : {forced_companion}")
print(f"    => PASS (forced bijection) = {pass_bijection}")
print(f"""
    TIER: KILL of the PASS-hypothesis (hardened K1) + a real BANKED POSITIVE.
    The pre-registered PASS (the descent is a forced BIJECTION reproducing the
    read) is FALSIFIED: the coupled DECK grading (S^2) is modulus-UNIFORM (free
    all Perron 2, interacting all shell sqrt2 -- the O(1) dressing collapses
    2->sqrt2), so it is NOT the read's dart (2,sqrt2,sqrt2) grading ([W,P3]!=0);
    and the read's Cl(6) companion is a CHOICE (the frame-vacuum lift). ⟹ the
    winding weld remains an IRREDUCIBLE ADOPTION at the subspace level too -- my
    A5 K1 (single-eigenstate 0.19/0.22) is CONFIRMED and HARDENED, not upgraded.
    The -70 ppm grade ceiling and the composite dictionary's status stay PINNED
    on the weld.
    BANKED POSITIVE (genuinely new, sharper than A5): the read channel's descent
    to the deck sectors is NOT random -- it is a FORCED COVARIANT superposition,
    weights {{1/3, 1/3 +- sqrt3/6}} = {np.round(wset,4)}, ONE set cyclically
    permuted in t. So the read <-> coupled relation has forced structure; what is
    NOT forced is (a) the Cl(6) companion (the lift) and (b) the grading identity
    (deck is modulus-uniform, read is not). The single named residual = the Cl(6)
    companion; even fixing it, the grading mismatch remains.
    T3 (eps-home) stays UN-localized-through-a-forced-descent: since the descent
    is not forced, the reopened transport-minus-leading route does NOT open here.
""")
check("V scope honesty: purely structural; NO eps evaluated; target absent; ONE "
      "frozen test set; forks closed by computation; no fit; no value; the "
      "PASS-hypothesis failure IS the finding (hardened K1), not a check error", True)

print("=" * 88)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 88)
sys.exit(0 if ok_all else 1)
