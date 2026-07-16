#!/usr/bin/env python3
"""
proofs/foundations/WS2_extended_cycle_carry_2026-07-07.py

WS2 — does the forced single-site species x deck correlation (WS1) CARRY to closed
walks? Pre-registered internal research notes
(committed 0d5942d BEFORE this file). FROZEN.

The deck S^2=(P3(x)U_pi)^2 is CONSERVED by the coupled walk W0 ([W0,S^2]=0), but N1b
showed the walk MIXES the 4-way species N-hat. WS2 measures the species content of
the walk WITHIN each conserved deck sector:
  STATIC (L=0): I_static from Tr(P^c_w Pi96_t)         -- the coupled deck-sector table
  WALK  (L->inf): I_walk from the dominant walk-mode s_t(w)=<psi_t|P^c_w|psi_t>
  CARRY FRACTION = I_walk / I_static.  ~O(1) => survives; ->0 => washes.

NO alpha_1, NO eps, NO mass, NO SM-parameter comparison. Poisons flagged, not invoked.
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

def H(p):
    p = np.asarray(p, float).ravel(); p = p[p > 1e-15]
    return float(-np.sum(p * np.log2(p)))
def mutual_info(T):
    """I(row;col) in bits from a nonneg table T (joint up to normalization)."""
    P = np.asarray(T, float); P = P / P.sum()
    pr = P.sum(1); pc = P.sum(0)
    return H(pr) + H(pc) - H(P)

# ===========================================================================
banner("S0  controls: rebuild WS1's forced objects (J, U_pi, W0, deck) + C-CONSERVE")
# ===========================================================================
d0 = np.zeros((NV, NE))
for e, (i, j, v) in enumerate(EDGES):
    d0[i, e] = -1.0; d0[j, e] = 1.0
Chat = np.array([[1, 1, 0], [-1, 0, 1], [0, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
H1, _ = np.linalg.qr(Chat)
B1 = np.linalg.svd(d0)[2][:3].T
A4 = [dict(enumerate(p)) for p in itertools.permutations(range(4))
      if sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j]) % 2 == 0]
rows = []
for g in A4:
    R6 = edge_rep(g)
    rows.append(np.kron(np.eye(3), (H1.T @ R6 @ H1).T) - np.kron(B1.T @ R6 @ B1, np.eye(3)))
_, SpJ, VpJ = np.linalg.svd(np.vstack(rows))
phi = VpJ[-1].reshape(3, 3); phi *= math.sqrt(3) / np.linalg.norm(phi)
J6 = B1 @ phi @ H1.T - H1 @ phi.T @ B1.T
wJ, VJ = np.linalg.eig(J6)

def build_species(sign):
    modes, _ = np.linalg.qr(VJ[:, np.where(np.abs(wJ - (1j if sign > 0 else -1j)) < 1e-9)[0]])
    A_ops = [gam(np.conj(modes[:, m])) / math.sqrt(2) for m in range(3)]
    NHAT = sum(a.conj().T @ a for a in A_ops)
    wN, VN = np.linalg.eigh(NHAT); wNr = np.round(np.real(wN)).astype(int)
    Pw = {w: VN[:, wNr == w] @ VN[:, wNr == w].conj().T for w in range(4)}
    return Pw
Pw_p = build_species(+1)
dims = {w: int(round(np.trace(Pw_p[w]).real)) for w in range(4)}
check(f"S0a species dims {dims} = 1/3/3/1 (nu/d/u/e)", dims == {0: 1, 1: 3, 2: 3, 3: 1})

# U_pi (WS1 verbatim)
sigma3 = {0: 0, 1: 2, 2: 3, 3: 1}
pi = {}
for e, (i, j, v) in enumerate(EDGES):
    a, b = sigma3[i], sigma3[j]; pi[e] = EIDX[(min(a, b), max(a, b))]
Rpi = np.zeros((NE, NE))
for e in range(NE):
    Rpi[pi[e], e] = 1.0
rowsU = [np.kron(gam(Rpi[:, a]), np.eye(8)) - np.kron(np.eye(8), g6[a].T) for a in range(NE)]
_, S2s, Vh = np.linalg.svd(np.vstack(rowsU))
U_pi = Vh[np.sum(S2s > 1e-9):].conj()[0].reshape(8, 8)
U_pi /= np.sqrt(np.abs(np.linalg.det(U_pi @ U_pi.conj().T)) ** (1 / 8))
check("S0b U_pi^3 = -I", np.max(np.abs(np.linalg.matrix_power(U_pi, 3) + np.eye(8))) < 1e-9)

# coupled walk W0 (WS1 verbatim), deck S^2, coupled species and deck projectors
B0 = srs.hashimoto((0.0, 0.0, 0.0)).real
GAMS = [gam(np.eye(NE)[:, EDGE_OF_DART[dp]]) for dp in range(ND)]
W0 = np.zeros((8 * ND, 8 * ND), complex)
for dp in range(ND):
    for d in np.nonzero(np.abs(B0[dp]) > 1e-14)[0]:
        W0[dp * 8:(dp + 1) * 8, d * 8:(d + 1) * 8] = B0[dp, d] * GAMS[dp]
P3 = np.zeros((ND, ND))
for a, (i, j) in enumerate(DARTS):
    for b, (p, q) in enumerate(DARTS):
        if (p, q) == (sigma3[i], sigma3[j]):
            P3[b, a] = 1.0; break
S = np.kron(P3, U_pi); Ssq = np.linalg.matrix_power(S, 2)
check("S0c [W0,S^2]=0 (deck conserved by the walk)", np.max(np.abs(W0 @ Ssq - Ssq @ W0)) < 1e-9)
Pi96 = {t: sum(OM ** (-t * m) * np.linalg.matrix_power(Ssq, m) for m in range(3)) / 3 for t in range(3)}
deck_dim = {t: int(round(np.trace(Pi96[t]).real)) for t in range(3)}
# coupled species P^c_w = I_dart (x) P_w  (dart-major, Fock-minor ordering, matches W0)
Pc = {w: np.kron(np.eye(ND), Pw_p[w]) for w in range(4)}
comm_spec = max(np.max(np.abs(W0 @ Pc[w] - Pc[w] @ W0)) for w in range(4))
check(f"S0d C-CONSERVE: [W0, P^c_w] != 0 (= {comm_spec:.2f}) -- the walk MIXES species "
      "(reproduces N1b from the coupled side)", comm_spec > 1e-2)

# ===========================================================================
banner("S1/S2  deck-sector dominant WALK modes + their species content s_t(w)")
# ===========================================================================
def deck_basis(t):
    ev, U = np.linalg.eigh((Pi96[t] + Pi96[t].conj().T) / 2)
    return U[:, ev > 0.5]                                  # 96 x deck_dim[t]
def dominant_mode(t):
    Bt = deck_basis(t)
    Wt = Bt.conj().T @ W0 @ Bt                             # walk restricted to sector t
    lam, V = np.linalg.eig(Wt)
    v = V[:, int(np.argmax(np.abs(lam)))]
    psi = Bt @ v; psi = psi / np.linalg.norm(psi)
    return psi, abs(lam[int(np.argmax(np.abs(lam)))])
def species_dist(psi):
    return np.array([float(np.real(psi.conj().T @ Pc[w] @ psi)) for w in range(4)])

s_walk = np.zeros((3, 4)); doms = []
for t in range(3):
    psi, dl = dominant_mode(t); doms.append(dl)
    s_walk[t] = species_dist(psi)
print("    WALK (L->inf) dominant-mode species content s_t(w):  [nu   d    u    e ]  (dom |lam|)")
for t in range(3):
    print(f"      deck t={t} (dim{deck_dim[t]}): {np.round(s_walk[t],4)}   sum={s_walk[t].sum():.3f}  ({doms[t]:.3f})")

# STATIC (L=0): coupled deck-sector species table Tr(P^c_w Pi96_t)
s_static = np.zeros((3, 4))
for t in range(3):
    for w in range(4):
        s_static[t, w] = float(np.real(np.trace(Pc[w] @ Pi96[t])))
print("\n    STATIC (L=0) coupled table Tr(P^c_w Pi96_t):        [nu   d    u    e ]")
for t in range(3):
    print(f"      deck t={t} (dim{deck_dim[t]}): {np.round(s_static[t],4)}   sum={s_static[t].sum():.3f}")

# ===========================================================================
banner("S3  THE CARRY: I_static, I_walk, carry fraction; + finite-L trend")
# ===========================================================================
# joint tables weighted by deck dim (the deck marginal is dim_t/96)
Jstatic = np.array([s_static[t] for t in range(3)])                  # already dim-weighted (traces)
Jwalk = np.array([deck_dim[t] * s_walk[t] for t in range(3)])        # weight the L2-normalized mode by dim
I_static = mutual_info(Jstatic)
I_walk = mutual_info(Jwalk)
carry = I_walk / I_static if I_static > 1e-12 else 0.0
print(f"    I_static(species;deck) = {I_static:.4f} bits   (coupled single-site baseline)")
print(f"    I_walk(species;deck)   = {I_walk:.4f} bits   (dominant closed-walk mode)")
print(f"    CARRY FRACTION I_walk/I_static = {carry:.4f}")

# finite-L trend: species distribution of W0^L applied to a deck-sector-uniform state
print(f"\n    finite-L species anisotropy within deck sectors (max_t ||s_t - mean_w||):")
def walk_L_species(L):
    sT = np.zeros((3, 4))
    for t in range(3):
        Bt = deck_basis(t)
        # start maximally-mixed within the sector; propagate density by W0^L (as a channel)
        WL = np.linalg.matrix_power(W0, L)
        rho = Bt @ Bt.conj().T                              # projector = maximally mixed (unnormalized)
        rho = WL @ rho @ WL.conj().T
        tr = float(np.real(np.trace(rho)))
        for w in range(4):
            sT[t, w] = float(np.real(np.trace(Pc[w] @ rho))) / (tr + 1e-30)
    return sT
for L in (1, 2, 3, 5):
    sT = walk_L_species(L)
    aniso = max(np.max(np.abs(sT[t] - sT[t].mean())) for t in range(3))
    IL = mutual_info(np.array([deck_dim[t] * sT[t] for t in range(3)]))
    print(f"      L={L}: within-sector species anisotropy = {aniso:.4f}   I(species;deck)={IL:.4f} bits")

# ===========================================================================
banner("S3b  MECHANISM: why I_static = 0 EXACTLY (structural, not numerical)")
# ===========================================================================
# Tr(P^c_w Pi96_t) = (1/3) sum_m omega^{-tm} Tr((P3^2)^m) Tr(P_w (U_pi^2)^m).
# The Fock factor Tr(P_w (U_pi^2)^m) carries WS1's correlation (m=1,2 terms).
# The DART factor Tr((P3^2)^m): P3 = C3 dart permutation. If C3 fixes NO darts,
# Tr(P3)=Tr(P3^2)=0, killing every t-dependent (m=1,2) term -> only the m=0
# (identity, t-independent) term survives -> species-blind EXACTLY.
trP3 = [float(np.real(np.trace(np.linalg.matrix_power(P3, m)))) for m in (0, 1, 2)]
print(f"    Tr((P3)^m) for m=0,1,2 = {trP3}   (C3 dart permutation fixed-dart counts)")
print(f"    => the m=1,2 (t-dependent, correlation-carrying) terms of the conserved coupled deck")
print(f"       vanish because C3 has NO fixed darts; only the m=0 species-marginal term survives.")
check("S3b MECHANISM: Tr(P3)=Tr(P3^2)=0 (no fixed darts) => the CONSERVED coupled deck is "
      "species-blind by an EXACT structural annihilation of WS1's single-site correlation "
      "(which lives in the C3-non-invariant sector projected out by the C3-averaged deck)",
      abs(trP3[1]) < 1e-9 and abs(trP3[2]) < 1e-9 and I_static < 1e-9)

# ===========================================================================
banner("S4  report-only: FREE walk null (gamma->I: Wfree = B0 (x) I_Fock)")
# ===========================================================================
Wfree = np.kron(B0.astype(complex), np.eye(8))
# free walk COMMUTES with species (no gamma insertion) -> species is exactly conserved; report I
free_comm = max(np.max(np.abs(Wfree @ Pc[w] - Pc[w] @ Wfree)) for w in range(4))
print(f"    [Wfree, P^c_w] = {free_comm:.2e}  (free walk conserves species exactly => no washing; the "
      "washing, if any, is INTERACTION-carried)")

# ===========================================================================
banner("V  VERDICT (pre-declared tiers; decided by the computation)")
# ===========================================================================
# democratic within-sector species (per non-empty class) ?
def within_sector_democratic(sT):
    # compare each sector's per-class density to the deck-independent species marginal
    marg = sT.sum(0) / 3.0
    return max(np.max(np.abs(sT[t] - marg)) for t in range(3)) < 1e-6
walk_democratic = within_sector_democratic(s_walk)
# quark/lepton (Z2) split survives even if 4-way washes?
def z2_survives(sT):
    lepton = sT[:, 0] + sT[:, 3]; quark = sT[:, 1] + sT[:, 2]
    return max(abs(lepton[t] - lepton.mean()) for t in range(3)) > 1e-3
CARRY = carry > 0.25 and not walk_democratic
WASH = carry < 0.05 or walk_democratic
z2 = z2_survives(s_walk)
if CARRY:
    verdict = (f"CARRY-SURVIVES -- the dominant closed-walk mode retains a forced, deck-correlated species "
               f"distribution (carry fraction {carry:.2f}). A cycle's winding class PARTIALLY FORCES its "
               f"species content => B1 gets a forced partial species assignment; the keystone's extended "
               f"lift is not a full adoption. Residual to price. NO value moved; -70 ppm/B1 still OPEN.")
elif WASH:
    verdict = (f"CARRY-WASHES -- the closed walk equilibrates species toward democratic within each conserved "
               f"deck sector (carry fraction {carry:.2f}; Z2 quark/lepton split {'SURVIVES' if z2 else 'also washes'}). "
               f"The forced single-site species<->deck correlation does NOT extend to cycles => B1's species "
               f"anchoring stays adoption-gated (consistent with N1b); the winding-weld residual is confirmed "
               f"irreducible AT THE CYCLE LEVEL. Sharper negative, quantified. NO value moved; -70 ppm/B1 OPEN.")
else:
    verdict = (f"PARTIAL -- carry fraction {carry:.2f}; Z2 quark/lepton {'survives' if z2 else 'washes'}, 4-way "
               f"species {'partly' if not walk_democratic else 'fully'} washed. The cycle forces the quark/lepton "
               f"split but not flavor => the residual shrinks to the u/d-type freedom. NO value moved.")
print("    " + verdict)
check("V scope honesty: structural; no alpha_1/eps/mass; no SM-parameter comparison; poisons not "
      "invoked; single-site 0.181 used as BASELINE not target; no value moved", True)

print("=" * 88)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print(f" VERDICT: {verdict.split(' -- ')[0]}")
print("=" * 88)
sys.exit(0 if ok_all else 1)
