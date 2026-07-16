#!/usr/bin/env python3
"""Phase 4.1 -- the spectral triple on srs-z (spec frozen 2026-06-11).

Spec: internal research notes (hash row committed at
freeze, b4bb97b). Pre-declared Dirac class: D2 (DEFAULT) = the validated
32x32 Clifford-bundle D(k) = sum_e gamma^e (x) L_e(k) of
`bound_state_dirac_dispersion_2026-05-29.py`, folded to the body-centering
cover as D_z(k) = D(k) (+) D(k+Delta); D1 (de Rham d+d*) and D3
(Hashimoto/zeta-native) built as controls. Grading gamma = Gamma_7 (x) I.
Algebra A = C(cover atoms). Real-structure candidates J = (U K) with U a
product of gamma's, assembled intra-block (TRIM) or mirror-crossing (M-
composed); all signs COMPUTED, never assumed.

HARD GATES (construction validity; abort on failure):
  G1  Clifford {g^e,g^f} = 2 delta (1e-12).
  G2  L_e(k)^2 = I at all sample k.
  G3  D(k)^2 = 6I + R_sub(k) regression (the Lichnerowicz identity).
  G4  grading: Gamma_7 Hermitian, Gamma_7^2 = I, {Gamma_7, g^e} = 0
      => {gamma, D_z} = 0 on the cover.
  G5  MIRROR DECOMPOSITION OF THE DIRAC (new, load-bearing for 4.2):
      M^2 = I and  D(k) + D(k+Delta) = 2*Phi  with
      Phi = sum_e g^e (x) F_e (F_e = fixed-point projector of edge e),
      k-INDEPENDENT -- the Dirac's mirror-EVEN part is exactly the
      fixed-point term (the adjacency/Hashimoto antiperiod is exact; the
      Dirac's is exact-up-to-Phi). Controls: D3 FULLY mirror-odd
      (D3(k+Delta) = -D3(k)); D1 mirror-even part = its cell-internal
      incidence (same offset-parity mechanism).
  G6  reality on the Cl(6) factor: uniform-sign antiunitary classes
      J0 = U K exist (search all 64 gamma-subset products); each found
      class records (s, J0^2, Gamma_7-sign).
  G7  cover reality where the cover fiber is self-conjugate:
      at TRIM (Gamma, N): intra-block J; at P (-P = P+Delta): the
      mirror-CROSSING candidate J_x = M o (J0 (x) I) K. Gate: at least
      one candidate commutes with D_z at every self-conjugate sample
      (1e-9), and the commutation table is k-stable.
  G8  TRIM DICHOTOMY AS REAL STRUCTURE (the Phase-1.3 statement in NCG
      form): at P every intra-block candidate FAILS and a crossing
      candidate WORKS; at Gamma/N an intra-block candidate works.
  G9  order-zero exact for A = C(cover atoms) ([a, J b J^-1] = 0);
      first-order residuals COMPUTED + k-stable (scorecard row, no
      pass/fail prejudgment -- graph triples generically violate it).
  G10 heat anchors for 4.2: Tr D_z(k)^2 = 384 EXACTLY at every sampled k
      (a2-analog is k-independent: tr(g^e g^f) = 8 delta + traceless
      commutators); Tr exp(-t D_z^2) finite; Tr D_z^4 BZ-average recorded
      (regression anchor).

SCORECARD (recorded, k-stable; the 4.1 deliverable per the frozen spec):
KO signs (J^2, JD vs DJ, J gamma vs gamma J) -> KO-dimension candidates;
first-order residual; mirror-covariance; J-at-P crossing status; D1/D3
contrast rows.
"""
import os
import sys
from itertools import combinations

import numpy as np
from numpy import linalg as la

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from proofs.common import find_bonds, A_PRIM  # noqa: E402

FAILURES = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


# ----- Cl(6,0) gammas (8x8, Jordan-Wigner; the bound-state probe's build) -----
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron3(a, b, c):
    return np.kron(np.kron(a, b), c)


GAMMAS = [kron3(X, I2, I2), kron3(Y, I2, I2), kron3(Z, X, I2),
          kron3(Z, Y, I2), kron3(Z, Z, X), kron3(Z, Z, Y)]
GAMMA7 = ((-1j) ** 3) * np.linalg.multi_dot(GAMMAS)

N_ATOMS, N_EDGES = 4, 6


def undirected_edges():
    bonds = find_bonds()
    seen = {}
    for src, tgt, cell in bonds:
        cell = tuple(int(c) for c in cell)
        key = (src, tgt, cell) if src < tgt else (tgt, src, tuple(-c for c in cell))
        seen[key] = True
    edges = sorted(seen.keys())
    assert len(edges) == N_EDGES
    return edges


EDGES = undirected_edges()


def L_e(edge, k):
    a, b, n = edge
    L = np.zeros((N_ATOMS, N_ATOMS), dtype=complex)
    ph = np.exp(2j * np.pi * np.dot(k, n))
    L[b, a] = ph
    L[a, b] = np.conj(ph)
    for c in range(N_ATOMS):
        if c not in (a, b):
            L[c, c] = 1.0
    return L


def D_of_k(k):
    D = np.zeros((32, 32), dtype=complex)
    for i, edge in enumerate(EDGES):
        D += np.kron(GAMMAS[i], L_e(edge, k))
    return D


def R_sub_of_k(k):
    R = np.zeros((32, 32), dtype=complex)
    Ls = [L_e(e, k) for e in EDGES]
    for e in range(N_EDGES):
        for f in range(N_EDGES):
            if e != f:
                R += 0.5 * np.kron(GAMMAS[e] @ GAMMAS[f], Ls[e] @ Ls[f] - Ls[f] @ Ls[e])
    return R


DELTA = np.array([0.5, 0.5, -0.5])
SADDLES = {"Gamma": np.zeros(3), "H": np.array([0.5, 0.5, -0.5]),
           "P": np.array([0.25, 0.25, 0.25]), "N": A_PRIM @ np.array([0.0, 0.5, 0.5])}
RNG = np.random.default_rng(20260611)
SAMPLE_K = list(SADDLES.values()) + [RNG.uniform(-0.5, 0.5, 3) for _ in range(4)]


def D_cover(k):
    """64x64 folded Dirac: blocks (k, k+Delta)."""
    Dz = np.zeros((64, 64), dtype=complex)
    Dz[:32, :32] = D_of_k(k)
    Dz[32:, 32:] = D_of_k(np.asarray(k) + DELTA)
    return Dz


M_SWAP = np.kron(np.array([[0, 1], [1, 0]]), np.eye(32)).astype(complex)
GAMMA_COVER = np.kron(np.eye(2), np.kron(GAMMA7, np.eye(4))).astype(complex)

print("=" * 72)
print(" PHASE 4.1 -- the spectral triple on srs-z (D2 default + controls)")
print("=" * 72)

# G1 Clifford
dev = max(la.norm(GAMMAS[a] @ GAMMAS[b] + GAMMAS[b] @ GAMMAS[a]
                  - 2 * (a == b) * np.eye(8)) for a in range(6) for b in range(6))
gate("G1 Clifford {g^e,g^f} = 2 delta", dev < 1e-12, f"dev={dev:.1e}")

# G2 involutions
dev = max(la.norm(L_e(e, k) @ L_e(e, k) - np.eye(4)) for e in EDGES for k in SAMPLE_K)
gate("G2 L_e(k)^2 = I at all sample k", dev < 1e-12, f"dev={dev:.1e}")

# G3 Lichnerowicz regression
dev = max(la.norm(D_of_k(k) @ D_of_k(k) - 6 * np.eye(32) - R_sub_of_k(k))
          for k in SAMPLE_K)
gate("G3 D(k)^2 = 6I + R_sub(k) (validated-Dirac regression)", dev < 1e-10,
     f"dev={dev:.1e}")

# G4 grading
d1 = la.norm(GAMMA7 - GAMMA7.conj().T)
d2 = la.norm(GAMMA7 @ GAMMA7 - np.eye(8))
d3 = max(la.norm(GAMMA7 @ g + g @ GAMMA7) for g in GAMMAS)
d4 = max(la.norm(GAMMA_COVER @ D_cover(k) + D_cover(k) @ GAMMA_COVER)
         for k in SAMPLE_K)
gate("G4 grading: Gamma_7 Hermitian, ^2 = I, {gamma, D_z} = 0 on cover",
     max(d1, d2, d3, d4) < 1e-12, f"max dev={max(d1, d2, d3, d4):.1e}")

# G5 mirror decomposition of the Dirac
PHI = np.zeros((32, 32), dtype=complex)
for i, (a, b, n) in enumerate(EDGES):
    F = np.zeros((4, 4))
    for c in range(4):
        if c not in (a, b):
            F[c, c] = 1.0
    PHI += np.kron(GAMMAS[i], F)
dev_m = la.norm(M_SWAP @ M_SWAP - np.eye(64))
dev_phi = max(la.norm(D_of_k(k) + D_of_k(np.asarray(k) + DELTA) - 2 * PHI)
              for k in SAMPLE_K)
odd_offsets = all((sum(n) % 2) == 1 for (_, _, n) in EDGES)
gate("G5 M^2 = I and D(k) + D(k+Delta) = 2*Phi (fixed-point term), all k; "
     "every bond offset odd-sum",
     dev_m < 1e-12 and dev_phi < 1e-12 and odd_offsets,
     f"dev={max(dev_m, dev_phi):.1e}")
print("      -> the Dirac's mirror-EVEN component is EXACTLY the fixed-point "
      "term Phi (candidate Higgs direction);")
print("         per atom Phi acts as the sum of the 3 NON-incident gammas "
      f"(||Phi||={la.norm(PHI):.4f}, Phi Hermitian dev="
      f"{la.norm(PHI - PHI.conj().T):.1e})")

# panel correction C3 (2026-06-12): the identity stated GAUGE-COVARIANTLY.
# Re-gauge atom 1 by lattice vector t = (1,0,0): D'(k) = W(k) D(k) W(k)^dag
# with W(k) = I8 (x) diag-phase on atom 1.  The NAIVE sum drifts; the
# covariant mirror image V D'(k+Delta) V^dag with V = I8 (x) diag(1,-1,1,1)
# (= W(k)^dag W(k+Delta), Delta.t = 1/2) restores the identity exactly.
t_g = np.array([1.0, 0.0, 0.0])
k_g = np.array([0.17, -0.23, 0.31])


def W_gauge(k):
    ph = np.exp(2j * np.pi * np.dot(k, t_g))
    return np.kron(np.eye(8), np.diag([1.0, ph, 1.0, 1.0])).astype(complex)


V_mir = np.kron(np.eye(8), np.diag([1.0, -1.0, 1.0, 1.0])).astype(complex)
Dg_k = W_gauge(k_g) @ D_of_k(k_g) @ W_gauge(k_g).conj().T
Dg_kD = W_gauge(np.asarray(k_g) + DELTA) @ D_of_k(np.asarray(k_g) + DELTA) \
    @ W_gauge(np.asarray(k_g) + DELTA).conj().T
naive_drift = la.norm(Dg_k + Dg_kD - 2 * W_gauge(k_g) @ PHI
                      @ W_gauge(k_g).conj().T)
cov_dev = la.norm(Dg_k + V_mir @ Dg_kD @ V_mir.conj().T
                  - 2 * W_gauge(k_g) @ PHI @ W_gauge(k_g).conj().T)
gate("G5e Phi GAUGE-COVARIANT (panel-ordered): in a shifted-representative "
     "gauge the naive even part drifts O(1); the covariant mirror "
     "V = diag(1,-1,1,1) restores D'(k) + V D'(k+Delta) V^dag = 2 Phi' "
     "exactly",
     naive_drift > 1.0 and cov_dev < 1e-12,
     f"naive drift={naive_drift:.1f}, covariant dev={cov_dev:.1e}")

# G6 reality classes on the Cl(6) factor: J0 = U K, U = product of a gamma subset
classes = []
for r in range(7):
    for S in combinations(range(6), r):
        U = np.eye(8, dtype=complex)
        for i in S:
            U = U @ GAMMAS[i]
        signs = []
        for g in GAMMAS:
            t = U @ g.conj() @ U.conj().T
            if la.norm(t - g) < 1e-12:
                signs.append(+1)
            elif la.norm(t + g) < 1e-12:
                signs.append(-1)
            else:
                signs = None
                break
        if signs and len(set(signs)) == 1:
            s = signs[0]
            UU = U @ U.conj()
            j2 = +1 if la.norm(UU - np.eye(8)) < 1e-12 else \
                 (-1 if la.norm(UU + np.eye(8)) < 1e-12 else 0)
            tg = U @ GAMMA7.conj() @ U.conj().T
            g7s = +1 if la.norm(tg - GAMMA7) < 1e-12 else \
                  (-1 if la.norm(tg + GAMMA7) < 1e-12 else 0)
            classes.append((S, s, j2, g7s))
gate("G6 uniform-sign antiunitary classes on Cl(6) exist (J0 = U K)",
     len(classes) > 0 and any(c[1] == +1 for c in classes),
     f"{len(classes)} classes; s=+1 classes: "
     f"{[(c[0], c[2], c[3]) for c in classes if c[1] == +1]}")

# G7/G8 cover reality at self-conjugate points
def commut_dev(JU, anti_swap, k):
    """||J D_z J^-1 - D_z|| for J = (swap?) o (JU (x) I4 per block) K."""
    Dz = D_cover(k)
    JU_cov = np.kron(np.eye(2), np.kron(JU, np.eye(4)))
    if anti_swap:
        JU_cov = M_SWAP @ JU_cov
    # J X J^-1 = JU_cov X* JU_cov^dagger  (J = JU_cov K, K X K = X*)
    return la.norm(JU_cov @ Dz.conj() @ JU_cov.conj().T - Dz)


s_plus = [c for c in classes if c[1] == +1]
results = {}
for nm, k in (("Gamma", SADDLES["Gamma"]), ("N", SADDLES["N"]),
              ("P", SADDLES["P"])):
    intra = min(commut_dev(np.linalg.multi_dot([GAMMAS[i] for i in S])
                           if S else np.eye(8, dtype=complex), False, k)
                for (S, s, j2, g7s) in s_plus)
    cross = min(commut_dev(np.linalg.multi_dot([GAMMAS[i] for i in S])
                           if S else np.eye(8, dtype=complex), True, k)
                for (S, s, j2, g7s) in s_plus)
    results[nm] = (intra, cross)
    print(f"      J commutation dev at {nm:5s}: intra-block "
          f"{intra:.2e}, mirror-crossing {cross:.2e}")
gate("G7 a commuting J exists at every self-conjugate sample (1e-9)",
     all(min(v) < 1e-9 for v in results.values()))
gate("G8 TRIM dichotomy as real structure -- PANEL-SCOPED 2026-06-12 to the "
     "UNDRESSED (gamma-product (x) I_atom) charge-conjugation family: "
     "undressed intra-J works at Gamma/N, FAILS at P; crossing-J works at P",
     results["Gamma"][0] < 1e-9 and results["N"][0] < 1e-9
     and results["P"][0] > 1e-3 and results["P"][1] < 1e-9)

# --- panel correction C3 (2026-06-12): the dressed intra-block class + the
#     atom-trivial completeness statement.  The original J-search family
#     (gamma-subset products (x) I_atom) was an UNFROZEN construction-time
#     restriction; the panel exhibited an admissible intra-block antiunitary
#     at P OUTSIDE it (atom transposition + matching Fock dressing, KO-6
#     signs).  Computed here exactly via the linear intertwiner space:
#     solutions X of  (X (x) P_pi) D(P)* = D(P) (X (x) P_pi).
from itertools import permutations as _perms  # noqa: E402

D_P = D_of_k(SADDLES["P"])


def intertwiner_space(Dm, perm, per_atom=False):
    """Nullspace basis of U -> U Dm* - Dm U over the dressed class
    U = sum_c X_c (x) |perm(c)><c| (per_atom=True: independent 8x8 blocks
    X_c; False: X_c = X common). Returns (list of U solutions, P_perm)."""
    Pm = np.zeros((4, 4))
    for i4, j4 in enumerate(perm):
        Pm[j4, i4] = 1.0
    cols = []
    if per_atom:
        for c4 in range(4):
            Ea = np.zeros((4, 4))
            Ea[perm[c4], c4] = 1.0
            for a8 in range(8):
                for b8 in range(8):
                    E = np.zeros((8, 8), dtype=complex)
                    E[a8, b8] = 1.0
                    cols.append(np.kron(E, Ea))
    else:
        for a8 in range(8):
            for b8 in range(8):
                E = np.zeros((8, 8), dtype=complex)
                E[a8, b8] = 1.0
                cols.append(np.kron(E, Pm))
    M = np.zeros((32 * 32, len(cols)), dtype=complex)
    for j, T in enumerate(cols):
        M[:, j] = (T @ Dm.conj() - Dm @ T).reshape(-1)
    _, sv, Vh = la.svd(M)
    null = []
    for i in range(len(cols)):
        s_i = sv[i] if i < len(sv) else 0.0
        if s_i < 1e-9:
            U = sum(Vh[i, j].conj() * cols[j] for j in range(len(cols)))
            null.append(U)
    return null, Pm


def block_unitarize(U, perm):
    """Project U (supported on the perm pattern; kron(E8, E_atom) layout:
    index = a8*4 + atom) to the nearest per-atom-block unitary (polar)."""
    out = np.zeros_like(U)
    Ur = U.reshape(8, 4, 8, 4)
    Or = out.reshape(8, 4, 8, 4)
    for c4 in range(4):
        X = Ur[:, perm[c4], :, c4]
        Uu, _, Vt = la.svd(X)
        Or[:, perm[c4], :, c4] = Uu @ Vt
    return out


def find_unitary_intertwiner(Dm, perm, seed_idx=0, iters=400):
    """Alternating projection: nullspace of the dressed intertwiner
    equation <-> per-atom-block unitaries. Returns (U, eq_residual,
    unitarity_residual) or None if the nullspace is empty."""
    null, Pm = intertwiner_space(Dm, perm, per_atom=True)
    if not null:
        return None
    U = null[seed_idx % len(null)]
    for _ in range(iters):
        Ub = block_unitarize(U, perm)
        # project back onto the nullspace (Frobenius-orthonormal basis)
        U = sum(np.sum(np.conj(N) * Ub) * N for N in null)
    Ub = block_unitarize(U, perm)
    eq_res = la.norm(Ub @ Dm.conj() - Dm @ Ub)
    un_res = la.norm(Ub @ Ub.conj().T - np.eye(32))
    return Ub, eq_res, un_res


def perm_parity(perm):
    p, seen = 0, [False] * 4
    for i4 in range(4):
        if not seen[i4]:
            j4, ln = i4, 0
            while not seen[j4]:
                seen[j4] = True
                j4 = perm[j4]
                ln += 1
            p += ln - 1
    return (-1) ** p


# atom-trivial completeness: pi = id with ARBITRARY X in M_8 -> EMPTY
null_id, _ = intertwiner_space(D_P, (0, 1, 2, 3))
gate("G8b COMPLETENESS (panel-ordered): the atom-trivial intra-block "
     "intertwiner space at P is EXACTLY EMPTY for arbitrary X in M_8 -- "
     "the undressed forced-crossing statement is search-complete",
     len(null_id) == 0, f"nullspace dim = {len(null_id)}")

# dressed classes: odd atom permutations, per-atom Fock blocks
dressed = []
for perm in _perms(range(4)):
    if perm_parity(perm) != -1:
        continue
    res = find_unitary_intertwiner(D_P, perm)
    if res is None:
        continue
    Ju, eq_res, un_res = res
    if eq_res < 1e-9 and un_res < 1e-9:
        jj = Ju @ Ju.conj()
        j2d = +1 if la.norm(jj - np.eye(32)) < 1e-9 else \
              (-1 if la.norm(jj + np.eye(32)) < 1e-9 else 0)
        G7a = np.kron(GAMMA7, np.eye(4))
        tg = Ju @ G7a.conj() @ Ju.conj().T
        g7d = +1 if la.norm(tg - G7a) < 1e-9 else \
              (-1 if la.norm(tg + G7a) < 1e-9 else 0)
        dressed.append((perm, Ju, eq_res, un_res, j2d, g7d))
gate("G8c DRESSED intra-block class at P EXISTS (panel finding, scorecard "
     "row): odd-atom-permutation dressing (per-atom Fock blocks) admits a "
     "unitary intra-block J with KO-6 signs (J^2 = +1, eps'' = -1)",
     any(c[4] == +1 and c[5] == -1 for c in dressed),
     f"{len(dressed)} dressed odd perms admit unitary J; signs "
     f"{sorted(set((c[4], c[5]) for c in dressed))}")

# no constant-W global extension: the dressed U found at P does NOT
# intertwine at other momenta (and the same dressing has no unitary
# solution at N)
if dressed:
    perm0, U_d = dressed[0][0], dressed[0][1]
    devs_g = [la.norm(U_d @ D_of_k(-np.asarray(kk)).conj() @ U_d.conj().T
                      - D_of_k(kk))
              for kk in (np.array([0.13, 0.27, -0.31]),
                         np.array([0.41, -0.07, 0.19]))]
    res_N = find_unitary_intertwiner(D_of_k(SADDLES["N"]), perm0)
    unitary_at_N = (res_N is not None and res_N[1] < 1e-9
                    and res_N[2] < 1e-9)
    gate("G8d the dressed class has NO constant-W global extension "
         "(fails at generic k; the same dressing admits no unitary "
         "intertwiner at N)",
         min(devs_g) > 1.0 and not unitary_at_N,
         f"generic-k dev >= {min(devs_g):.2f}; unitary at N: {unitary_at_N}")
print("      PANEL HEADLINE REWORDING (binding): 'UNDRESSED charge "
      "conjugation is FORCED to cross the mirror at P (atom-trivial intra "
      "empty -- exact); point-group-DRESSED intra-block alternatives exist "
      "at P with KO-6 signs; none extends globally with constant W.'")

# KO signs for the best crossing J at P (scorecard)
bestS = min(s_plus, key=lambda c: commut_dev(
    np.linalg.multi_dot([GAMMAS[i] for i in c[0]]) if c[0] else np.eye(8, dtype=complex),
    True, SADDLES["P"]))
S, s, j2, g7s = bestS
U_best = np.linalg.multi_dot([GAMMAS[i] for i in S]) if S else np.eye(8, dtype=complex)
J_cov = M_SWAP @ np.kron(np.eye(2), np.kron(U_best, np.eye(4)))
j2_cov = +1 if la.norm(J_cov @ J_cov.conj() - np.eye(64)) < 1e-12 else \
         (-1 if la.norm(J_cov @ J_cov.conj() + np.eye(64)) < 1e-12 else 0)
g_dev_p = la.norm(J_cov @ GAMMA_COVER.conj() @ J_cov.conj().T - g7s * GAMMA_COVER)
print(f"      KO signs (crossing J at P, U = gammas{S}): J^2 = {j2_cov:+d}, "
      f"JD = +DJ (commuting), J gamma J^-1 = {g7s:+d} gamma (dev {g_dev_p:.1e})")
# standard even table (eps, eps', eps'') with eps' = +1 (J commutes with D):
# KO 0 = (+,+,+), KO 2 = (-,+,-), KO 4 = (-,+,+), KO 6 = (+,+,-)
ko = {(+1, +1): "0", (-1, -1): "2", (-1, +1): "4", (+1, -1): "6"}
print(f"      -> (eps, eps', eps'') = ({j2_cov:+d},+1,{g7s:+d}) => "
      f"KO-dimension {ko.get((j2_cov, g7s), '?')} mod 8")
print("      PANEL SCOPE (binding wording): KO-dim 2 holds FOR THE CANONICAL "
      "DECK CHARGE CONJUGATION J0 (the only class shown to extend globally); "
      "the structure is an EVEN FINITE TRIPLE WITH GLOBAL REAL STRUCTURE "
      "satisfying order-zero, VIOLATING order-one -- never an unqualified "
      "'KO-dim-2 real spectral triple'. The dressed intra class at P has "
      "KO-6 signs (G8c) and no global extension (G8d).")

# G9 algebra: order-zero exact; first-order residuals recorded
fo_res, oz_res = [], []
for k in (SADDLES["P"], SADDLES["Gamma"]):
    Dz = D_cover(k)
    for _ in range(4):
        a = np.kron(np.diag(RNG.normal(size=2)),
                    np.kron(np.eye(8), np.diag(RNG.normal(size=4))))
        b = np.kron(np.diag(RNG.normal(size=2)),
                    np.kron(np.eye(8), np.diag(RNG.normal(size=4))))
        b0 = J_cov @ b.conj() @ J_cov.conj().T
        oz_res.append(la.norm(a @ b0 - b0 @ a))
        Da = Dz @ a - a @ Dz
        fo_res.append(la.norm(Da @ b0 - b0 @ Da) / max(la.norm(Da), 1e-300))
gate("G9 order-zero EXACT for A = C(cover atoms); first-order residual "
     "recorded (scorecard)",
     max(oz_res) < 1e-12,
     f"order-zero {max(oz_res):.1e}; first-order rel residual "
     f"{min(fo_res):.3f}..{max(fo_res):.3f}")

# G10 heat anchors
tr2 = [np.trace(D_cover(k) @ D_cover(k)).real for k in SAMPLE_K]
gate("G10a Tr D_z(k)^2 = 384 exactly at every sampled k (a2-analog flat)",
     max(abs(t - 384.0) for t in tr2) < 1e-9,
     f"max dev={max(abs(t - 384.0) for t in tr2):.1e}")
grid = [np.array([i, j, l]) / 6.0 for i in range(6) for j in range(6) for l in range(6)]
tr4, tr6 = [], []
for k in grid:
    Dz2 = D_cover(k)
    Dz2 = Dz2 @ Dz2
    tr4.append(np.trace(Dz2 @ Dz2).real)
    tr6.append(np.trace(Dz2 @ Dz2 @ Dz2).real)
tr4, tr6 = np.array(tr4), np.array(tr6)
t_heat = 0.5
ht = float(np.mean([np.sum(np.exp(-t_heat * la.eigvalsh(D_cover(k) @ D_cover(k))))
                    for k in grid[::36]]))
gate("G10b Tr D_z^4 / D_z^6 BZ-recorded; heat trace finite",
     np.all(np.isfinite(tr4)) and np.all(np.isfinite(tr6)) and np.isfinite(ht),
     f"<Tr D^4> = {tr4.mean():.6f} (std {tr4.std():.2e}); "
     f"<Tr D^6> = {tr6.mean():.4f} (std {tr6.std():.4f}); "
     f"Tr exp(-{t_heat} D^2) ~ {ht:.4f}")

# D1 / D3 controls (mirror-parity contrast rows)
def D1_of(k):
    d = np.zeros((6, 4), dtype=complex)
    for i, (a, b, n) in enumerate(EDGES):
        d[i, a] = -1.0
        d[i, b] = np.exp(2j * np.pi * np.dot(k, n))
    D1 = np.zeros((10, 10), dtype=complex)
    D1[4:, :4] = d
    D1[:4, 4:] = d.conj().T
    return D1


def B_of(k):
    bonds = find_bonds()
    E = [(i, j, tuple(int(x) for x in c)) for (i, j, c) in bonds]
    idx = {e: a for a, e in enumerate(E)}
    rev = {a: idx[(j, i, tuple(-x for x in c))] for a, (i, j, c) in enumerate(E)}
    B = np.zeros((len(E), len(E)), dtype=complex)
    for a, (i, j, c) in enumerate(E):
        for b2, (i2, j2, c2) in enumerate(E):
            if i2 == j and b2 != rev[a]:
                B[b2, a] = np.exp(2j * np.pi * np.dot(k, np.asarray(c2, float)))
    return B


def D3_of(k):
    B = B_of(k)
    D3 = np.zeros((24, 24), dtype=complex)
    D3[12:, :12] = B
    D3[:12, 12:] = B.conj().T
    return D3


k_t = SAMPLE_K[5]
d3_anti = la.norm(D3_of(np.asarray(k_t) + DELTA) + D3_of(k_t))
d1_even = D1_of(np.asarray(k_t) + DELTA) + D1_of(k_t)
d1_pred = np.zeros((10, 10), dtype=complex)   # 2x the cell-internal (-1) incidence
for i, (a, b, n) in enumerate(EDGES):
    d1_pred[4 + i, a] = -2.0
    d1_pred[a, 4 + i] = -2.0
gate("D3 control: FULLY mirror-odd, D3(k+Delta) = -D3(k) (zeta-native)",
     d3_anti < 1e-12, f"dev={d3_anti:.1e}")
gate("D1 control: mirror-even part = 2x cell-internal incidence (offset-0 "
     "entries only)", la.norm(d1_even - d1_pred) < 1e-12,
     f"dev={la.norm(d1_even - d1_pred):.1e}")

print("\n--- 4.1 SCORECARD (k-stable; the deliverable) ---")
print("  D2 (default): grading EXACT; mirror-even part = Phi (fixed-point "
      "term) EXACT;")
print(f"  reality: s=+1 classes exist; commuting J at all self-conjugate "
      f"points; AT P THE REAL STRUCTURE IS FORCED MIRROR-CROSSING")
print(f"  (TRIM dichotomy = KO statement); KO signs (J^2, Jg) = "
      f"({j2_cov:+d},{g7s:+d}); order-zero EXACT; first-order VIOLATED "
      f"(rel residual ~{np.median(fo_res):.2f} -- recorded, standard for "
      f"graph Diracs);")
print("  a2-analog moment k-FLAT (384); D3 mirror-odd / D1 mixed / D2 "
      "carries Phi -- the Higgs-direction candidate is D2-specific.")

print("\n" + "=" * 72)
if FAILURES:
    print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
    sys.exit(1)
print(" RESULT: ALL GATES PASS -- triple defined; scorecard banked (4.2 next)")
print("=" * 72)
sys.exit(0)
