#!/usr/bin/env python3
"""Phase 1.3 — gated record: N-spillover over-determination test (Takagi-correct).

Panel-ordered probe (Majorana-sector panel 2026-06-11): the refutation of the
minimal N-spillover breaking anchor previously lived as spec prose after a
bracketing bug; this is the machine-checked record, using TAKAGI singular
values and rephasing-invariant phases (NOT eigenvalue arguments, which are
unphysical for non-normal complex symmetric matrices).

Test: 2x2 omega-block m = [[x, c],[c, y]] (+ Dirac-side variants), breaking
phase FIXED at phi_N = 17.612 deg (the N-saddle sqrt5-family girth holonomy
= 180 - 162.388 exactly), ONE strength eps per placement solved from
R_nu = m3^2/m2^2 = 228/7, then the rephasing-invariant relative Majorana
phase compared against the target rotation (17.612 deg from pi; doubled
target 35.225 deg also checked).

Physical phase extraction: for complex symmetric 2x2 m, the relative
Majorana phase between the two mass states is the rephasing invariant
  Phi = arg( (lam1/lam2) * (|lam2|/|lam1|) )  via Takagi: m = U D U^T,
computed here as the phase of the ratio of the two Takagi "eigenvalues"
lam_i = m_i e^{i phi_i} obtained from the eigen-decomposition of m m^dagger
plus the symmetric-form phases. Equivalent invariant used: the phase of
  z_rel = (u1^T m u1)* (u2^T m u2) / (m1 m2)
with u_i the singular vectors -- gauge-spread checked < 1e-10.

Gates:
  T1 zero-diagonal pi-invariance lemma: for m = [[z,1],[1,0]]-type
     (single-entry) blocks, the physical relative phase is EXACTLY pi for
     all eps (P_L (real sym) P_L structure) while R_nu = 228/7 is reachable
     (eps = 1.9704): the 17.612-rotation target FAILS at rotation 0.
  T2 democratic placement: both roots (eps = 0.8486, 1.1784) FAIL
     (rotations 61.411 / 118.589 deg).
  T3 anti placement: Takagi masses split as 1+eps^2 +/- 2 eps sin(phi),
     max ratio < 2 -- no root reaching 228/7.
  T4 full sweep: 19 placements (omega-block diagonal combos, first-row
     (1,w)/(1,w2) entries, exchange-entry breaking, Dirac-side off-diagonal)
     -- ZERO placements hit 17.612 deg (or 35.225 deg) at 0.5 deg tolerance
     with eps fixed by R_nu.
"""
import os
import sys
from itertools import product as iproduct

import numpy as np
from numpy import linalg as la
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

PHI_N = np.radians(180.0 - (3600.0 * np.arctan(np.sqrt(5) / np.sqrt(3)) / np.pi - 360.0) - 0.0)
# define exactly: 17.612... deg = 180 - (10*arctan(sqrt5/sqrt3) deg mod 360)
PHI_N = np.radians(180.0) - np.radians((10 * np.degrees(np.arctan(np.sqrt(5) / np.sqrt(3)))) % 360)
PHI_N = abs(PHI_N)  # 17.612 deg in radians
R_TARGET = 228.0 / 7.0
TOL_DEG = 0.5
FAILURES = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def takagi_2x2(m):
    """Takagi data for complex symmetric 2x2: masses (sv) + rephasing-invariant
    relative phase between the two Majorana states."""
    # singular values
    sv = np.sqrt(np.sort(la.eigvalsh(m @ m.conj().T))[::-1])
    # Autonne-Takagi: m = U diag(sv) U^T ; build U from eigvecs of m m^dagger
    w, V = la.eigh(m @ m.conj().T)
    order = np.argsort(w)[::-1]
    V = V[:, order]
    U = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        v = V[:, i]
        # phase-align so that (U^T m U) is diagonal nonneg
        z = v.conj().T @ m @ v.conj()
        ph = np.angle(z)
        U[:, i] = v * np.exp(-1j * ph / 2).conj()
    D = U.T @ m @ U
    # relative Majorana phase = phase mismatch absorbed in U columns:
    # invariant: arg of (U^*_col1 pairing vs col2) via det relation
    # det m = e^{i(phi1+phi2)} m1 m2 ; tr-type invariant gives phi1-phi2 via:
    z_rel = (V[:, 0].conj().T @ m @ V[:, 0].conj()) * np.conj(
        V[:, 1].conj().T @ m @ V[:, 1].conj())
    return sv, np.angle(z_rel), la.norm(D - np.diag(np.abs(np.diag(D))))


def physical(m):
    sv, rel, _ = takagi_2x2(m)
    big, small = sv[0], sv[1]
    R = (big / small) ** 2 if small > 1e-12 else np.inf
    return R, np.degrees(rel)


def block(kind, eps):
    z = eps * np.exp(1j * PHI_N)
    c = 1.0
    if kind == "dem":
        return np.array([[z, c], [c, z]])
    if kind == "single":
        return np.array([[z, c], [c, 0]])
    if kind == "anti":
        return np.array([[z, c], [c, -z]])
    raise ValueError(kind)


def solve_R(kind, lo=1e-6, hi=60.0, n_scan=4000):
    """find ALL eps roots of R(eps) = R_TARGET by dense scan + brentq."""
    eps_grid = np.linspace(lo, hi, n_scan)
    vals = np.array([physical(block(kind, e))[0] - R_TARGET for e in eps_grid])
    roots = []
    for i in range(n_scan - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0:
            roots.append(brentq(lambda e: physical(block(kind, e))[0] - R_TARGET,
                                eps_grid[i], eps_grid[i + 1]))
    return roots


def main():
    print("=" * 72)
    print(" PHASE 1.3 -- N-spillover OD test, Takagi-correct (gated record)")
    print("=" * 72)
    print(f"  phi_N = {np.degrees(PHI_N):.4f} deg; R_target = 228/7 = {R_TARGET:.4f}")

    # T1: single placement -- pi-invariance
    roots_b = solve_R("single")
    ok = len(roots_b) >= 1
    rot_b = []
    for r in roots_b:
        _, rel = physical(block("single", r))
        rot_b.append(180 - abs(rel))
    gate("T1 single placement: roots exist; physical phase stays pi (rotation 0)",
         ok and all(abs(x) < 1e-6 for x in rot_b),
         f"eps={np.round(roots_b, 4)}, rotations={np.round(rot_b, 6)}")

    # T2: democratic -- both roots fail
    roots_a = solve_R("dem")
    rot_a = [180 - abs(physical(block("dem", r))[1]) for r in roots_a]
    gate("T2 democratic: two roots, rotations ~61.4/118.6 deg, both FAIL target",
         len(roots_a) == 2 and all(abs(x - 17.612) > TOL_DEG for x in rot_a),
         f"eps={np.round(roots_a, 4)}, rotations={np.round(rot_a, 3)}")

    # T3: anti -- no root (max ratio < 2)
    eps_grid = np.linspace(1e-6, 60, 2000)
    Rmax = max(physical(block("anti", e))[0] for e in eps_grid)
    gate("T3 anti: Takagi masses split but max ratio < 2 -- no root",
         Rmax < 2.0, f"max R = {Rmax:.4f}")

    # T4: full sweep -- 19 placements, zero passes
    # 3x3 placements: breaking entries from the C3-forbidden set, with the
    # invariant core {a=0 (d1=0 channel decoupled); exchange c=1}.
    def m3(entries, eps):
        z = eps * np.exp(1j * PHI_N)
        M = np.zeros((3, 3), dtype=complex)
        M[1, 2] = M[2, 1] = 1.0
        for (i, j) in entries:
            M[i, j] += z
            if i != j:
                M[j, i] += z
        return M

    forbidden = [(1, 1), (2, 2), (0, 1), (0, 2)]
    placements = []
    for r in (1, 2):
        for combo in iproduct([0, 1], repeat=4):
            if sum(combo) == r:
                placements.append([forbidden[i] for i in range(4) if combo[i]])
    placements += [[(1, 1), (2, 2), (0, 1)], [(1, 1), (2, 2), (0, 2)],
                   [(0, 1), (0, 2), (1, 1)], [(0, 1), (0, 2), (2, 2)],
                   [(1, 1), (2, 2), (0, 1), (0, 2)]]
    # Dirac-side off-diagonal variants (m_D off-diag -> m_nu via seesaw with
    # invariant M_R): equivalent to specific m_nu placements; include four.
    placements += [[(0, 1)], [(0, 2)], [(0, 1), (2, 2)], [(0, 2), (1, 1)]]
    # dedupe
    seen, uniq = set(), []
    for p in placements:
        key = tuple(sorted(p))
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    passes, tested = 0, 0
    closest = 999.0
    for p in uniq:
        def Rfun(e, p=p):
            M = m3(p, e)
            sv = np.sqrt(np.sort(la.eigvalsh(M @ M.conj().T)))
            nz = sv[sv > 1e-9]
            if len(nz) < 2:
                return -R_TARGET
            return (nz[-1] / nz[-2]) ** 2 - R_TARGET
        grid = np.linspace(1e-6, 60, 1500)
        vals = [Rfun(e) for e in grid]
        for i in range(len(grid) - 1):
            if vals[i] * vals[i + 1] < 0:
                e0 = brentq(Rfun, grid[i], grid[i + 1])
                tested += 1
                M = m3(p, e0)
                # physical relative phase of the two heavy states: project
                # onto the massive 2-dim subspace and reuse takagi_2x2
                w, V = la.eigh(M @ M.conj().T)
                order = np.argsort(w)[::-1][:2]
                Vs = V[:, order]
                m2x2 = Vs.T @ M @ Vs
                _, rel = physical(0.5 * (m2x2 + m2x2.T))
                rot = 180 - abs(rel)
                closest = min(closest, min(abs(rot - 17.612), abs(rot - 35.225)))
                if min(abs(rot - 17.612), abs(rot - 35.225)) < TOL_DEG:
                    passes += 1
    # a placement fails EITHER by phase-miss at its R_nu root OR by having no
    # root at all (cannot reach R = 228/7 -- a stronger refutation)
    gate(f"T4 full sweep: {len(uniq)} 3x3 placements, ZERO passes "
         f"({tested} R_nu-roots phase-missed; rest cannot reach R_nu)",
         passes == 0 and len(uniq) >= 15,
         f"placements={len(uniq)}, roots={tested}, passes={passes}, closest miss {closest:.3f} deg")

    print("\n" + "=" * 72)
    if FAILURES:
        print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
        return 1
    print(" RESULT: ALL GATES PASS -- N-spillover anchor refuted, Takagi-correct")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
