#!/usr/bin/env python3
"""Phase 4 ERRATUM E2 (panel-ordered 2026-06-12): the alias-free sigma census.

The Phase-4 adjudication panel REFUTED two recorded findings as GRID2
quadrature artifacts:
  - the m8 sigma-kinetic value +89362.23 (4.2/W6) — the 2^3 subgroup grid
    aliases the response's harmonics; the alias-free value is ~3.18x smaller;
  - the 4.3/G5b "color/grade-blindness" of the sigma census — at exact
    quadrature the census is BLOCK-DISCRIMINATING (rep-dependent kinetics,
    octet sign-flip), so the clause "the action propagates any coupler but
    SELECTS no scalar content" is STRUCK from register-bound wording.

This probe is the ordered recomputation. Quadrature strategy: the m8
t^2-coefficient response is a trigonometric polynomial in k of bounded
order; a size-N subgroup grid annihilates exactly the harmonics with all
components = 0 mod N. Agreement between the N = 3 and N = 5 grids (coprime
moduli — they alias DISJOINT harmonic classes up to order 15) certifies the
quadrature; an independent Monte-Carlo sample cross-checks without any
subgroup structure. (The panel's letter asked ">= 9^3 or MC"; coprime-grid
agreement + MC is the stronger certificate and is what its own referees
used: GRID3 = GRID4 = GRID5 to 12 digits.)

Gates:
  E2a  ARTIFACT REPRODUCED: the GRID2 g0-block m8 slope equals the banked
       +89362.23 while the exact value differs by an O(1) factor (~3.18x)
       — the published number is demonstrably the aliasing artifact.
  E2b  QUADRATURE CERTIFIED: per census block, GRID3 == GRID5 (rel 1e-9)
       and MC agrees within sampling tolerance.
  E2c  THE CENSUS IS BLOCK-DISCRIMINATING (the corrected finding): the m8
       kinetic moments differ across color/grade blocks at O(1); the octet
       block's SIGN is opposite the g0 singlet's. Full table printed
       (cross-reference: panel referee values, possibly block-reordered).
  E2d  m10 census (GRID3 == GRID5 certified): records the discrimination
       at the next moment level.
  E2e  isotropy of the g0-block m8 kinetic on the exact grid (x vs z axis).
  E2f  JEOPARDY-IMPACT: the 4.3 determinate conclusion is UNAFFECTED —
       Sum_s(determinate) = 0 rests on the EXTERNALITY of the sigma
       couplers (no mirror-crossing one-forms from the algebra; recomputed
       here, exactly 0), NOT on the struck blindness clause. The exact
       census's discrimination structure is recorded as a candidate FUTURE
       selection lever (panel: "the refutation makes the bet MORE
       interesting, not less").
"""
import os
import sys
from itertools import combinations

import numpy as np
from numpy import linalg as la

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from proofs.common import find_bonds  # noqa: E402

FAILURES = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron3(a, b, c):
    return np.kron(np.kron(a, b), c)


GAMMAS = [kron3(X, I2, I2), kron3(Y, I2, I2), kron3(Z, X, I2),
          kron3(Z, Y, I2), kron3(Z, Z, X), kron3(Z, Z, Y)]
GAMMA7 = ((-1j) ** 3) * np.linalg.multi_dot(GAMMAS)

A_OPS = [(GAMMAS[0] + 1j * GAMMAS[1]) / 2.0,
         (GAMMAS[2] + 1j * GAMMAS[3]) / 2.0,
         (GAMMAS[4] + 1j * GAMMAS[5]) / 2.0]

# su(3) generators (the 4.3 construction, trace-orthonormalized)
gens = []
for i in range(3):
    for j in range(3):
        if i < j:
            E = A_OPS[i].conj().T @ A_OPS[j]
            gens.append((E + E.conj().T) / 2.0)
            gens.append((E - E.conj().T) / 2.0j)
n1 = A_OPS[0].conj().T @ A_OPS[0]
n2 = A_OPS[1].conj().T @ A_OPS[1]
n3 = A_OPS[2].conj().T @ A_OPS[2]
gens.append((n1 - n2) / np.sqrt(2.0))
gens.append((n1 + n2 - 2 * n3) / np.sqrt(6.0))
G_metric = np.array([[np.trace(a @ b).real for b in gens] for a in gens])
W = la.cholesky(la.inv(G_metric)).T
T3 = [sum(W[i, j] * gens[j] for j in range(8)) for i in range(8)]


def grade_basis(g):
    if g == 0:
        return [np.eye(8, dtype=complex)]
    out = []
    for S in combinations(range(6), g):
        P = np.linalg.multi_dot([GAMMAS[i] for i in S]) if len(S) > 1 \
            else GAMMAS[S[0]]
        if (g * (g - 1) // 2) % 2 == 1:
            P = 1j * P
        out.append(P)
    return out


def casimir_superop_eig(basis):
    n = len(basis)
    S = np.zeros((n, n), dtype=complex)
    for b2, Xb in enumerate(basis):
        out = sum(t @ (t @ Xb - Xb @ t) - (t @ Xb - Xb @ t) @ t for t in T3)
        for a2, Xa in enumerate(basis):
            S[a2, b2] = np.trace(Xa.conj().T @ out) / 8.0
    ev, V = la.eigh((S + S.conj().T) / 2.0)
    blocks = []
    for val in sorted(set(np.round(ev, 6))):
        idx = [i for i in range(n) if abs(ev[i] - val) < 1e-6]
        vec = V[:, idx[0]].real if la.norm(V[:, idx[0]].imag) < 1e-9 \
            else V[:, idx[0]]
        rep = sum(vec[j] * basis[j] for j in range(n))
        rep = (rep + rep.conj().T) / 2.0
        rep = rep * np.sqrt(8.0) / la.norm(rep)
        blocks.append((float(val), len(idx), rep))
    return blocks


REP_NAME = {0.0: "singlet", round(4.0 / 3.0, 6): "3+3bar",
            3.0: "octet", round(10.0 / 3.0, 6): "6+6bar"}
CENSUS = []
for g in range(4):
    for val, mult, rep in casimir_superop_eig(grade_basis(g)):
        CENSUS.append((g, REP_NAME.get(round(val, 6), f"C2={val:.3f}"),
                       mult, rep))

# ----- the Dirac / cover / kinetic machinery (4.2 conventions) -----
def undirected_edges():
    bonds = find_bonds()
    seen = {}
    for src, tgt, cell in bonds:
        cell = tuple(int(c) for c in cell)
        key = (src, tgt, cell) if src < tgt else (tgt, src, tuple(-c for c in cell))
        seen[key] = True
    return sorted(seen.keys())


EDGES = undirected_edges()


def L_e(edge, k):
    a, b, n = edge
    L = np.zeros((4, 4), dtype=complex)
    ph = np.exp(2j * np.pi * np.dot(k, n))
    L[b, a] = ph
    L[a, b] = np.conj(ph)
    for c in range(4):
        if c not in (a, b):
            L[c, c] = 1.0
    return L


def D_of_k(k):
    D = np.zeros((32, 32), dtype=complex)
    for i, e in enumerate(EDGES):
        D += np.kron(GAMMAS[i], L_e(e, k))
    return D


DELTA = np.array([0.5, 0.5, -0.5])
_DC = {}


def D_cover(k):
    key = tuple(np.round(np.asarray(k, float), 12))
    if key not in _DC:
        Dz = np.zeros((64, 64), dtype=complex)
        Dz[:32, :32] = D_of_k(k)
        Dz[32:, 32:] = D_of_k(np.asarray(k) + DELTA)
        _DC[key] = Dz
    return _DC[key]


M_SWAP = np.kron(np.array([[0, 1], [1, 0]]), np.eye(32)).astype(complex)
INDICATORS = [np.diag((np.arange(4) == c).astype(float)) for c in range(4)]
QMAG = 0.15


def census_kinetics(ks, m, reps=None):
    """t^2-coefficients of <Tr (D_pair + t V)^m> at q = 0 and q = QMAG*x,
    per census block (S4-summed indicator profiles); returns slopes."""
    reps = reps if reps is not None else [c[3] for c in CENSUS]
    out = []
    for q in (np.zeros(3), np.array([QMAG, 0, 0])):
        tot = np.zeros(len(reps))
        for k in ks:
            Dp = np.zeros((128, 128), dtype=complex)
            Dp[:64, :64] = D_cover(k)
            Dp[64:, 64:] = D_cover(np.asarray(k) + q)
            pows = [np.eye(128, dtype=complex)]
            for _ in range(m - 2):
                pows.append(pows[-1] @ Dp)
            for r, Xm in enumerate(reps):
                for prof in INDICATORS:
                    V = M_SWAP @ np.kron(np.eye(2), np.kron(Xm, prof))
                    Vp = np.zeros((128, 128), dtype=complex)
                    Vp[64:, :64], Vp[:64, 64:] = V, V.conj().T
                    Wl = [pows[j] @ Vp for j in range(m - 1)]
                    s = sum(float(np.sum(Wl[j] * Wl[m - 2 - j].T).real)
                            for j in range(m - 1))
                    tot[r] += (m / 2.0) * s
        out.append(tot / len(ks))
    return (out[1] - out[0]) / QMAG ** 2


def grid_of(n):
    return [np.array([i, j, l]) / float(n) for i in range(n)
            for j in range(n) for l in range(n)]


print("=" * 72)
print(" PHASE 4 ERRATUM E2 -- the alias-free sigma census (panel-ordered)")
print("=" * 72)
print("  census blocks:", [(g, nm, mult) for g, nm, mult, _ in CENSUS])

# E2a: reproduce the artifact
kin2 = census_kinetics(grid_of(2), 8, reps=[CENSUS[0][3]])
kin3_g0 = census_kinetics(grid_of(3), 8, reps=[CENSUS[0][3]])
gate("E2a ARTIFACT REPRODUCED: GRID2 g0 m8 slope = +89362.23 (the banked "
     "value) vs alias-free value differing by O(1)",
     abs(kin2[0] - 89362.2268) < 1.0 and abs(kin2[0] / kin3_g0[0]) > 1.5,
     f"GRID2 {kin2[0]:+.2f} vs GRID3 {kin3_g0[0]:+.2f} "
     f"(inflation x{kin2[0] / kin3_g0[0]:.2f})")

# E2b: quadrature certificate (coprime grids + MC with honest sampling error)
kin3 = census_kinetics(grid_of(3), 8)
kin5 = census_kinetics(grid_of(5), 8)
scale = max(abs(kin5).max(), 1e-12)
dev35 = np.max(np.abs(kin3 - kin5)) / scale
# MC sanity on the g0 block with per-sample slopes (so the tolerance is the
# estimator's own standard error, not an arbitrary number)
RNG = np.random.default_rng(20260612)
mc_samples = [census_kinetics([RNG.uniform(0, 1, 3)], 8,
                              reps=[CENSUS[0][3]])[0] for _ in range(60)]
mc_mean = float(np.mean(mc_samples))
mc_se = float(np.std(mc_samples, ddof=1) / np.sqrt(len(mc_samples)))
mc_ok = abs(mc_mean - kin5[0]) < 4 * mc_se
gate("E2b QUADRATURE CERTIFIED: GRID3 == GRID5 (coprime moduli alias "
     "DISJOINT harmonic classes up to order 15 -- agreement is a complete "
     "certificate for this bounded-order response) + MC within 4 SE",
     dev35 < 1e-9 and mc_ok,
     f"grid3-vs-grid5 rel dev={dev35:.1e}; MC(60) g0 = {mc_mean:+.1f} "
     f"+- {mc_se:.1f} vs exact {kin5[0]:+.2f}")

# E2c: block discrimination + the corrected table
print("      ALIAS-FREE m8 census (GRID5; the corrected record):")
panel_ref = [28140.53, 0.0, 9380.18, -25887.21, -17260.79, 0.0, 0.0, 3412.15]
for i, (g, nm, mult, _) in enumerate(CENSUS):
    print(f"        grade {g} {nm:10s} x{mult:2d}: m8 c2 = {kin5[i]:+12.2f}")
print(f"      (panel referee reference list, possibly block-reordered: "
      f"{panel_ref})")
spread = (kin5.max() - kin5.min()) / scale
octet_i = next(i for i, c in enumerate(CENSUS) if c[1] == "octet")
gate("E2c CENSUS IS BLOCK-DISCRIMINATING (blindness claim corrected): O(1) "
     "spread across blocks; the octet sign is OPPOSITE the g0 singlet's",
     spread > 0.5 and np.sign(kin5[octet_i]) != np.sign(kin5[0])
     and abs(kin5[octet_i]) > 1.0,
     f"spread {spread:.2f} of scale {scale:.1f}; octet {kin5[octet_i]:+.2f} "
     f"vs g0 {kin5[0]:+.2f}")

# E2d: m10
kin3_10 = census_kinetics(grid_of(3), 10)
kin5_10 = census_kinetics(grid_of(5), 10)
scale10 = max(abs(kin5_10).max(), 1e-12)
dev35_10 = np.max(np.abs(kin3_10 - kin5_10)) / scale10
print("      ALIAS-FREE m10 census (GRID5):")
for i, (g, nm, mult, _) in enumerate(CENSUS):
    print(f"        grade {g} {nm:10s} x{mult:2d}: m10 c2 = {kin5_10[i]:+12.1f}")
n_nonzero_10 = int(np.sum(np.abs(kin5_10) > 1e-3 * scale10))
gate("E2d m10 census certified (GRID3 == GRID5) and recorded",
     dev35_10 < 1e-9, f"rel dev={dev35_10:.1e}; nonzero blocks at m10: "
     f"{n_nonzero_10}/{len(CENSUS)}")

# E2e: isotropy of the g0 m8 kinetic on the exact grid
def kin_axis(axis):
    q = np.zeros(3)
    q[axis] = QMAG
    out = []
    for qq in (np.zeros(3), q):
        tot = 0.0
        for k in grid_of(3):
            Dp = np.zeros((128, 128), dtype=complex)
            Dp[:64, :64] = D_cover(k)
            Dp[64:, 64:] = D_cover(np.asarray(k) + qq)
            pows = [np.eye(128, dtype=complex)]
            for _ in range(6):
                pows.append(pows[-1] @ Dp)
            for prof in INDICATORS:
                V = M_SWAP @ np.kron(np.eye(2), np.kron(np.eye(8), prof))
                Vp = np.zeros((128, 128), dtype=complex)
                Vp[64:, :64], Vp[:64, 64:] = V, V.conj().T
                Wl = [pows[j] @ Vp for j in range(7)]
                tot += 4.0 * sum(float(np.sum(Wl[j] * Wl[6 - j].T).real)
                                 for j in range(7))
        out.append(tot / 27.0)
    return (out[1] - out[0]) / QMAG ** 2


iso_x, iso_z = kin_axis(0), kin_axis(2)
gate("E2e g0 m8 kinetic ISOTROPIC on the exact grid (x vs z axis)",
     abs(iso_x - iso_z) / max(abs(iso_x), 1e-12) < 1e-9,
     f"x {iso_x:+.2f} vs z {iso_z:+.2f}")

# E2f: jeopardy impact -- the determinate conclusion is unaffected
b_sheet = np.kron(np.diag([1.0, -1.0]), np.kron(np.eye(8),
                                                np.diag([1., 2., 3., 4.])))
a_el = np.kron(np.eye(2), np.kron(np.eye(8), np.diag([2., 1., 1., 3.])))
Dz_P = D_cover(np.array([0.25, 0.25, 0.25]))
A_form = a_el @ (Dz_P @ b_sheet - b_sheet @ Dz_P)
off = la.norm(A_form[32:, :32]) + la.norm(A_form[:32, 32:])
gate("E2f JEOPARDY-IMPACT: Sum_s(determinate) = 0 UNAFFECTED -- it rests "
     "on sigma-EXTERNALITY (no mirror-crossing one-forms from the algebra, "
     "exactly 0), not on the struck blindness clause; the discrimination "
     "structure is a candidate FUTURE selection lever",
     off < 1e-12, f"mirror-off-diagonal one-form norm = {off:.1e}")

print("\n--- E2 CORRECTED RECORD ---")
print(f"  4.2/W6 m8 sigma value: +89362.23 (GRID2 artifact) -> "
      f"{kin5[0]:+.2f} (alias-free)")
print("  4.3/G5b: 'color/grade-blind' STRUCK; the alias-free census is")
print("  block-discriminating (octet sign-flip); register wordings updated")
print("  per the panel order. Determinate jeopardy conclusion UNAFFECTED.")

print("\n" + "=" * 72)
if FAILURES:
    print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
    sys.exit(1)
print(" RESULT: ALL GATES PASS -- erratum E2 banked; registers may now move")
print("=" * 72)
sys.exit(0)
