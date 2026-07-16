#!/usr/bin/env python3
"""
proofs/foundations/NATIVE_a4_A5b_matter_cone_2026-07-05.py

A5(b)-locking arc, PROBE 3 (sub-question 1): WHICH cone is the physical zeta_{D4} matter
cone -- the ADJACENCY lambda=-1 triple (read_matter_row) or the HODGE-DIRAC D4 cone
(OMEGA_T1/T4)? The docs disagree, and the answer bears on the spin-1 -> spin-1/2 lock.
Pre-registration: internal research notes §7 (committed
BEFORE this probe: 25c8bab). CLASS: pure structure (class a). NO PDG.

P1: the adjacency triple = a spin-1 multifold (flat middle band; Chern +-2) -- UNLOCKED.
P2: the Hodge-Dirac D4 cone -- dispersion + Chern; LOCKED Dirac (paired +-v|k|, Chern 0) or not?
P3: verdict -- do the two candidate matter cones DIFFER in lock?
"""
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

NV, NE = 4, 6
EDGES = [(0, 1, (0, 0, 0)), (0, 2, (0, 0, 0)), (0, 3, (0, 0, 0)),
         (1, 2, (1, 0, 0)), (1, 3, (0, 1, 0)), (2, 3, (0, 0, 1))]
Cm = np.array([[0, 1, -1], [1, 0, 1], [-1, 1, 0]], float)
G12 = (5 * np.eye(3) + Cm) / 3

ok_all = True
def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}")

def A_q(q):
    A = np.zeros((NV, NV), complex)
    for i, j, v in EDGES:
        p = np.exp(1j * np.dot(q, v)); A[i, j] += p; A[j, i] += np.conj(p)
    return A

def d_inc(q):
    d = np.zeros((NV, NE), complex)
    for e, (i, j, v) in enumerate(EDGES):
        d[i, e] = -1.0; d[j, e] = np.exp(1j * np.dot(q, v))
    return d

def D_q(q):
    d = d_inc(q)
    return np.block([[np.zeros((NV, NV)), d], [d.conj().T, np.zeros((NE, NE))]])

def chern_sphere(Hf, band, r=0.15, Nth=36, Nph=72):
    thetas = np.linspace(1e-3, math.pi - 1e-3, Nth)
    phis = np.linspace(0, 2 * math.pi, Nph, endpoint=False)
    V = np.empty((Nth, Nph), object)
    for i, t in enumerate(thetas):
        for j, ph in enumerate(phis):
            p = r * np.array([math.sin(t) * math.cos(ph), math.sin(t) * math.sin(ph), math.cos(t)])
            _, W = np.linalg.eigh(Hf(p)); V[i, j] = W[:, band]
    F = 0.0
    for i in range(Nth - 1):
        for j in range(Nph):
            j2 = (j + 1) % Nph
            u1 = np.vdot(V[i, j], V[i, j2]); u2 = np.vdot(V[i, j2], V[i + 1, j2])
            u3 = np.vdot(V[i + 1, j2], V[i + 1, j]); u4 = np.vdot(V[i + 1, j], V[i, j])
            F += np.angle(u1 * u2 * u3 * u4)
    return F / (2 * math.pi)

rng = np.random.default_rng(7)

print("=" * 90)
print(" P1  the ADJACENCY lambda=-1 triple = a spin-1 multifold (UNLOCKED)")
print("=" * 90)
wG, UG = np.linalg.eigh(A_q((0, 0, 0)))
P3 = UG[:, np.abs(wG + 1) < 1e-6]
def dA(q, ax):
    A = np.zeros((NV, NV), complex)
    for i, j, v in EDGES:
        p = 1j * v[ax] * np.exp(1j * np.dot(q, v)); A[i, j] += p; A[j, i] += np.conj(p)
    return A
S = [P3.conj().T @ dA((0, 0, 0), ax) @ P3 for ax in range(3)]
flat_adj = True
for _ in range(20):
    kh = rng.normal(size=3); kh /= np.linalg.norm(kh)
    ev = np.sort(np.real(np.linalg.eigvalsh(sum(kh[a] * S[a] for a in range(3)))))
    flat_adj &= abs(ev[1]) < 1e-9 and abs(ev[0] + ev[2]) < 1e-9   # {-c,0,+c}
HA_p = lambda p: A_q(G12 @ np.asarray(p, float))
chA = [round(chern_sphere(HA_p, b)) for b in (0, 1, 2)]
check(f"adjacency triple: flat middle band {{-c,0,+c}} = {flat_adj}; Chern {chA} = (-2,0,+2) "
      "=> spin-1 multifold, UNLOCKED", flat_adj and chA == [-2, 0, 2])

print("=" * 90)
print(" P2  the HODGE-DIRAC D4 cone: dispersion + Chern")
print("=" * 90)
evD0 = np.round(np.real(np.linalg.eigvalsh(D_q((0, 0, 0)))), 6)
print(f"    D(Gamma) spectrum = {evD0.tolist()}")
# disperse the near-zero modes at small generic k: how many stay flat (index/H1) vs cone?
kmags = [0.02, 0.04, 0.08]
disp = {}
for _ in range(12):
    kh = rng.normal(size=3); kh /= np.linalg.norm(kh)
    for km in kmags:
        ev = np.sort(np.abs(np.real(np.linalg.eigvalsh(D_q(G12 @ (km * kh))))))
        disp.setdefault(km, []).append(ev[:6])   # the 6 lowest |E|
means = {km: np.mean(disp[km], axis=0) for km in kmags}
print("    lowest 6 |E| of D4 near Gamma (mean over directions):")
for km in kmags:
    print(f"      |k|={km:.2f}: {np.round(means[km], 5).tolist()}")
# classify: a mode is FLAT if |E| stays ~0; DISPERSING if |E| ~ v|k| (grows linearly).
n_flat = int(np.sum(means[0.08] < 1e-4))
# linear cone check: the dispersing |E| should scale ~ linearly with |k| (ratio ~ const)
dispersing_lowest = means[0.08][n_flat] if n_flat < 6 else float('nan')
lin = means[0.08][n_flat] / means[0.04][n_flat] if n_flat < 6 else float('nan')
print(f"    => {n_flat} FLAT zero modes (the H1 index/deck flats); the rest DISPERSE; "
      f"lowest dispersing |E| scales ~x{lin:.2f} for |k| x2 (=2.0 => linear cone)")
# Chern of the dispersing D4 bands around the zero sector (bands 3,4,5,6 straddle E=0)
HD_p = lambda p: D_q(G12 @ np.asarray(p, float))
chD = [round(chern_sphere(HD_p, b), 2) for b in (3, 4, 5, 6)]
print(f"    D4 near-zero band Chern (bands 3,4,5,6): {chD}")
check("the HODGE D4 cone: has the 2 H1 flats (index) + a LINEARLY DISPERSING sector, and "
      f"the near-zero bands are Chern-0/real ({chD}) -- NOT the adjacency's chiral +-2",
      n_flat == 2 and abs(lin - 2.0) < 0.3 and all(abs(c) < 0.15 for c in chD))

print("=" * 90)
print(" P3  VERDICT -- the two candidate matter cones DIFFER in lock")
print("=" * 90)
print(f"""    ADJACENCY lambda=-1 triple: spin-1 multifold -- flat middle band {{-c,0,+c}},
      Chern (-2,0,+2). CHIRAL, UNLOCKED (the 4/1/2-count object of OMEGA_T4 / probe 2).
      This is what read_matter_row uses (1 Weyl per cone).

    HODGE-DIRAC D4 cone: {n_flat} H1 flats (the index sector, topological) + a linearly
      dispersing sector; the near-zero bands are Chern-0 / REAL. This is what OMEGA_T1's
      a2/a4 dictionary and OMEGA_T4's ch_D=0 use.

    => The two candidate matter cones are GENUINELY DIFFERENT objects (chiral spin-1 vs
    real/Chern-0), and the docs use them inconsistently (read_matter_row: adjacency;
    OMEGA_T1/T4: Hodge). SUB-QUESTION 1 RESULT: the lock hinges on WHICH is physical.
    The Hodge D4 cone is closer to a locked/real structure (Chern 0), but its dispersing
    sector's full a4 counts (2/2/0 lock?) and the flat/index separation are the next
    sitting's construction (the Kahler-Dirac -> Dirac reduction via the internal Cl(6)).
    This probe does NOT close A5(b); it resolves the cone-ambiguity into a concrete fork
    and hands the next arc a specific object (the Hodge Chern-0 cone) to test for the lock.
    No value moved; no PDG.""")
print("=" * 90)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 90)
sys.exit(0 if ok_all else 1)
