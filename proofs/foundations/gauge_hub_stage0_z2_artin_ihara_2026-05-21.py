#!/usr/bin/env python3
"""
Gauge-hub merge — Stage 0: the Z2 Artin-Ihara factorization warm-up.

Scoping doc: an internal working note

CLAIM UNDER TEST (Stage 0, the "warm-up that must work").
  srs-z is the Z2 (bipartite double) cover of srs. If the gauge-hub-merge
  route is viable, the non-backtracking (Hashimoto) operator of the cover must
  factor, Bloch-point by Bloch-point, into the two irreps of the voltage group
  Z2 = {triv, sign}:

      B_NB(cover)(k)  ~=  B_NB(srs)(k)        [trivial rep]
                       (+)  ( - B_NB(srs)(k) )  [sign rep]

  i.e. spec B_cover(k) = spec B_NB(k)  U  spec(-B_NB(k)), at every k.
  Equivalently the Ihara zeta factors as an Artin-Ihara L-product:
      zeta_cover^-1(u,k) = L(u,k,triv)^-1 * L(u,k,sign)^-1
  with L(triv) = zeta_srs and L(sign) the (-1)-per-arc twisted zeta.

  IF THIS FAILS, the covering-tower route to the gauge hub is dead. Stage 0.

The Z2 voltage is the bipartite-crossing voltage: every srs edge carries the
non-trivial element of Z2; a closed walk's holonomy = parity of its length.
The sign rep then weights each Hashimoto arc by -1, so B_NB^(sign) = -B_NB.

NO observed input. Pure substrate linear algebra. 7 pre-declared gates.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from proofs.common import find_bonds, K_STAR

TOL = 1e-9
gates = []   # (name, passed, detail)


# ---------------------------------------------------------------------------
# srs primitive cell + Bloch-Hashimoto operator
# ---------------------------------------------------------------------------
bonds = find_bonds()                       # 12 directed bonds, (src,tgt,cell)
NB = len(bonds)                            # 12 directed arcs
assert NB == 12, f"expected 12 srs arcs, got {NB}"


def build_BNB(arc_list, k_frac):
    """Bloch non-backtracking (Hashimoto) operator on a directed-arc list.

    M[j,i] = (arc i then arc j) is an admissible non-backtracking step:
      head(i) == tail(j)  AND  arc j is not the immediate reverse of arc i.
    Phase exp(2*pi*i k.cell_i) is carried on the *incoming* arc i.
    """
    n = len(arc_list)
    M = np.zeros((n, n), dtype=complex)
    for j, (sj, tj, cj) in enumerate(arc_list):
        for i, (si, ti, ci) in enumerate(arc_list):
            if sj != ti:
                continue
            dc = tuple(int(ci[d]) + int(cj[d]) for d in range(3))
            if tj == si and dc == (0, 0, 0):
                continue                    # immediate reversal -> backtrack
            M[j, i] = np.exp(2j * np.pi * np.dot(k_frac, ci))
    return M


# ---------------------------------------------------------------------------
# The Z2 bipartite double cover of the srs primitive cell
# ---------------------------------------------------------------------------
# 8 atoms: a in 0..3 = sheet 0, a+4 = sheet 1. Voltage 1 on every edge:
# srs arc (src,tgt,cell) lifts to (src, tgt+4, cell) and (src+4, tgt, cell).
cover_bonds = []
for (src, tgt, cell) in bonds:
    cover_bonds.append((src,     tgt + 4, cell))
    cover_bonds.append((src + 4, tgt,     cell))
NC = len(cover_bonds)
assert NC == 24, f"expected 24 cover arcs, got {NC}"

# Deck transformation on arcs: sheet swap a <-> a+4.
def deck_atom(a):
    return (a + 4) % 8

deck_perm = []
for (s, t, c) in cover_bonds:
    img = (deck_atom(s), deck_atom(t), c)
    deck_perm.append(cover_bonds.index(img))
deck_perm = np.array(deck_perm)
# deck is an involution permutation of the 24 arcs
gates.append(("G1 deck is a fixed-point-free involution on the 24 cover arcs",
              np.array_equal(deck_perm[deck_perm], np.arange(NC))
              and not np.any(deck_perm == np.arange(NC)),
              f"deck^2 = id: {np.array_equal(deck_perm[deck_perm], np.arange(NC))}"))


# ---------------------------------------------------------------------------
# Gate 2-3: spectral factorization at named + random Bloch points
# ---------------------------------------------------------------------------
def spec(M):
    return np.linalg.eigvals(M)

def specs_match(a, b, tol=1e-6):
    """Tolerant multiset equality of two complex spectra (greedy matching).

    np.sort_complex is NOT safe here: degenerate conjugate pairs whose real
    parts differ at the 1e-10 float level get ordered inconsistently between
    the two arrays. Greedy nearest-neighbour matching is order-independent.
    """
    a, b = list(a), list(b)
    if len(a) != len(b):
        return False
    rem = list(b)
    for x in a:
        if not rem:
            return False
        dists = [abs(x - y) for y in rem]
        m = int(np.argmin(dists))
        if dists[m] > tol:
            return False
        rem.pop(m)
    return True

def spec_gap(a, b):
    """Worst residual of the greedy multiset match (0 if perfect)."""
    a, b = list(a), list(b)
    if len(a) != len(b):
        return float('inf')
    rem, worst = list(b), 0.0
    for x in a:
        dists = [abs(x - y) for y in rem]
        m = int(np.argmin(dists))
        worst = max(worst, dists[m])
        rem.pop(m)
    return worst

rng = np.random.default_rng(20260521)
named = {"Gamma": [0, 0, 0], "P": [0.25, 0.25, 0.25], "H": [0.5, 0.5, 0.5]}
k_random = [rng.random(3) for _ in range(12)]

named_ok, named_detail = True, []
for label, k in named.items():
    B = build_BNB(bonds, k)
    Bc = build_BNB(cover_bonds, k)
    predicted = np.concatenate([spec(B), spec(-B)])
    ok = specs_match(spec(Bc), predicted)
    named_ok &= ok
    named_detail.append(f"{label}:{'ok' if ok else 'FAIL'}")
gates.append(("G2 spec B_cover = spec(B_NB) U spec(-B_NB) at Gamma/P/H",
              named_ok, " ".join(named_detail)))

rand_worst = 0.0
for k in k_random:
    B = build_BNB(bonds, k)
    Bc = build_BNB(cover_bonds, k)
    predicted = np.concatenate([spec(B), spec(-B)])
    rand_worst = max(rand_worst, spec_gap(spec(Bc), predicted))
gates.append(("G3 same factorization holds at 12 random Bloch points",
              rand_worst < 1e-6, f"worst |dspec| = {rand_worst:.2e}"))


# ---------------------------------------------------------------------------
# Gate 4: the block-decomposition is the Z2 deck action (not just spectra)
# ---------------------------------------------------------------------------
# Build the deck operator D (permutation matrix) and the +/- projectors.
D = np.zeros((NC, NC))
for i, j in enumerate(deck_perm):
    D[j, i] = 1.0
k_test = k_random[0]
Bc = build_BNB(cover_bonds, k_test)
commutes = np.max(np.abs(Bc @ D - D @ Bc)) < 1e-9
# +1 / -1 eigenspaces of D
P_plus = (np.eye(NC) + D) / 2
P_minus = (np.eye(NC) - D) / 2
# B_cover restricted to each eigenspace, compared to +/- B_NB(srs)
def restrict(M, P):
    # orthonormal basis of range(P)
    w, V = np.linalg.eigh(P)
    basis = V[:, w > 0.5]
    return basis.conj().T @ M @ basis
B = build_BNB(bonds, k_test)
plus_block = restrict(Bc, P_plus)
minus_block = restrict(Bc, P_minus)
plus_ok = specs_match(spec(plus_block), spec(B))
minus_ok = specs_match(spec(minus_block), spec(-B))
gates.append(("G4 D commutes with B_cover; +1 block ~ B_NB, -1 block ~ -B_NB",
              commutes and plus_ok and minus_ok,
              f"[B_c,D]=0:{commutes}  +block~B:{plus_ok}  -block~-B:{minus_ok}"))


# ---------------------------------------------------------------------------
# Gate 5: the Artin-Ihara L-function factorization at the determinant level
#   zeta_cover^-1 = L(triv)^-1 * L(sign)^-1
#   det(I - u B_cover + (k*-1)u^2 I) = det(I - u B + ..) * det(I + u B + ..)
# ---------------------------------------------------------------------------
u = 2.0 / 3.0                              # the framework's q_NB working point
kk = K_STAR
worst_det = 0.0
for k in [named["Gamma"], named["P"]] + k_random[:5]:
    B = build_BNB(bonds, k)
    Bc = build_BNB(cover_bonds, k)
    I12, I24 = np.eye(NB), np.eye(NC)
    lhs = np.linalg.det(I24 - u * Bc + (kk - 1) * u**2 * I24)
    L_triv = np.linalg.det(I12 - u * B + (kk - 1) * u**2 * I12)
    L_sign = np.linalg.det(I12 + u * B + (kk - 1) * u**2 * I12)   # B -> -B
    worst_det = max(worst_det, abs(lhs - L_triv * L_sign))
gates.append(("G5 zeta_cover^-1 = L(triv)^-1 * L(sign)^-1 (Artin-Ihara, Z2)",
              worst_det < 1e-7, f"worst |det_cover - L_triv*L_sign| = {worst_det:.2e}"))


# ---------------------------------------------------------------------------
# Gate 6: reproduce the known srs-z fact -- h-multiplicity 4 vs 2 at P
# ---------------------------------------------------------------------------
# Framework fact (memory / r9 srs-z run): the Ramanujan h = (sqrt3 + i sqrt5)/2
# has multiplicity 2 on srs at the P-point and multiplicity 4 on srs-z.
# Under the factorization that requires -h also in spec(B_NB(srs),P) with mult 2,
# i.e. the P-point Hashimoto spectrum is symmetric under mu -> -mu.
h = (np.sqrt(3) + 1j * np.sqrt(5)) / 2
B_P = build_BNB(bonds, named["P"])
sp_P = spec(B_P)
mult_h_srs = int(np.sum(np.abs(sp_P - h) < 1e-6))
sp_cover_P = np.concatenate([sp_P, -sp_P])     # = spec B_cover(P) by G2
mult_h_cover = int(np.sum(np.abs(sp_cover_P - h) < 1e-6))
sym = specs_match(sp_P, -sp_P)
gates.append(("G6 h-multiplicity 2 (srs) -> 4 (srs-z) at P reproduced",
              mult_h_srs == 2 and mult_h_cover == 4,
              f"mult(h) srs={mult_h_srs} cover={mult_h_cover}; "
              f"P-spectrum +/- symmetric: {sym}"))


# ---------------------------------------------------------------------------
# Gate 7: the sign-rep twist is a NON-trivial cover (contrast: the archived
#   C3 probe found its twist cohomologically trivial, Z_omega = Z_0).
#   Here L(sign) must genuinely differ from L(triv).
# ---------------------------------------------------------------------------
B_g = build_BNB(bonds, named["Gamma"])
I12 = np.eye(NB)
L_triv_g = np.linalg.det(I12 - u * B_g + (kk - 1) * u**2 * I12)
L_sign_g = np.linalg.det(I12 + u * B_g + (kk - 1) * u**2 * I12)
nontrivial = abs(L_triv_g - L_sign_g) > 1e-6
gates.append(("G7 sign-rep twist is non-trivial (L(sign) != L(triv)) -- "
              "unlike the archived C3 twist",
              nontrivial,
              f"L(triv,Gamma)={L_triv_g:.5f}  L(sign,Gamma)={L_sign_g:.5f}  "
              f"|diff|={abs(L_triv_g - L_sign_g):.4f}"))


# ---------------------------------------------------------------------------
print("=" * 74)
print("GAUGE-HUB STAGE 0 -- Z2 ARTIN-IHARA FACTORIZATION WARM-UP")
print("=" * 74)
npass = 0
for name, ok, detail in gates:
    tag = "PASS" if ok else "FAIL"
    if ok:
        npass += 1
    print(f"  [{tag}] {name}")
    print(f"         {detail}")
print("-" * 74)
print(f"  {npass}/{len(gates)} gates")
if npass == len(gates):
    print("""
  STAGE 0 VERDICT: the Z2 covering case factorizes cleanly. srs-z's
  non-backtracking operator IS the trivial (+) sign Artin-Ihara
  decomposition of srs's -- the warm-up the scoping doc required.
  The covering-tower route to the gauge hub is NOT dead at Stage 0.
  Next: Stage 1 (the voltage-graph -> L-function dictionary) and the
  live unknown, Stage 2 (which group sits above Z2).""")
else:
    print("""
  STAGE 0 VERDICT: FAIL -- the Z2 case does not factorize as claimed.
  Per the scoping doc, the covering-tower route is dead. Stop and
  re-examine before any Stage 1+ work.""")
print("=" * 74)
sys.exit(0 if npass == len(gates) else 1)
