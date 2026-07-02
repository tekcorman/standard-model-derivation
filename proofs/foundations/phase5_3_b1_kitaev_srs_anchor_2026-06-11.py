#!/usr/bin/env python3
"""Phase 5.3/B1 -- the Kitaev-on-srs anchor build (Majorana level).

Spec: docs/scoping/phase5_3_kitaev_spec_2026-06-11.md (FROZEN, SHA-256
a8d1d7ed..., register row in the freezing commit). srs = the hyperoctagon
lattice (10,3)-a; the Kitaev model on it is exactly solved in the
literature (Hermanns-Trebst 2014: gapless QSL with a Majorana FERMI
SURFACE + static Z2 gauge structure). The literature is a falsifiable
REFERENCE; everything here is computed from the repo's own cell
conventions (proofs/common.py).

B1 anchor gates:
  A1g 3-edge-coloring census: proper translation-periodic colorings of
      the 6 undirected bonds per primitive cell (each of the 4 nodes
      touches one x, one y, one z bond) EXIST; full census reported
      (it is a freedom inventory, not a choice -- at the isotropic point
      J_x = J_y = J_z the matter spectrum is coloring-independent).
  A2g uniform-gauge Bloch Hamiltonian H(k) = i A(k) (4x4) is Hermitian by
      construction (orientation convention from the bond list), and its
      real-space L=2 torus version reproduces the fiber spectra.
  A3g GAPLESSNESS + FERMI SURFACE: on BZ grids, min_k min_n |eps_n(k)|
      -> 0, and the measure of {k : min|eps| < delta} scales ~ delta^1
      (codimension-1 zero surface), NOT ~ delta^3 (point nodes) --
      the Hermanns-Trebst qualitative anchor.
  A4g Z2 gauge structure at the quadratic level (L=2 torus, 32 sites):
      flipping u on ALL bonds at one site (node gauge move) leaves the
      spectrum IDENTICAL (< 1e-12); flipping u on ONE bond (changes ring
      fluxes) CHANGES the spectrum -- physical content = flux, local
      flips = gauge.
"""
import os
import sys
from itertools import product

import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from proofs.common import find_bonds  # noqa: E402

FAILURES = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


bonds = find_bonds()
# undirected bonds: keep one orientation per pair (canonical: the bond list
# order; pair (i,j,c) with (j,i,-c))
seen = set()
UBONDS = []
for (i, j, c) in bonds:
    key = (j, i, tuple(-x for x in c))
    if key in seen:
        continue
    seen.add((i, j, tuple(c)))
    UBONDS.append((i, j, tuple(int(x) for x in c)))
NB = len(UBONDS)

print("=" * 72)
print(" PHASE 5.3/B1 -- Kitaev-on-srs anchor (Majorana level)")
print("=" * 72)
print(f"  undirected bonds per primitive cell: {NB}")

# A1g: proper 3-edge-colorings (translation-periodic)
colorings = []
for assign in product(range(3), repeat=NB):
    ok = True
    for v in range(4):
        cols = sorted(assign[b] for b, (i, j, c) in enumerate(UBONDS)
                      if i == v or j == v)
        if cols != [0, 1, 2]:
            ok = False
            break
    if ok:
        colorings.append(assign)
gate("A1g proper translation-periodic 3-edge-colorings EXIST",
     len(colorings) > 0, f"census = {len(colorings)} of 3^{NB} = {3**NB}")


# A2g: Bloch Hamiltonian H(k) = i A(k), orientation s=+1 along UBONDS
def H_of(k, u=None):
    if u is None:
        u = np.ones(NB)
    H = np.zeros((4, 4), dtype=complex)
    for b, (i, j, c) in enumerate(UBONDS):
        ph = np.exp(2j * np.pi * np.dot(k, np.asarray(c, float)))
        H[j, i] += 1j * u[b] * ph
        H[i, j] += -1j * u[b] * np.conj(ph)
    return H


rng = np.random.default_rng(3)
herm = 0.0
for _ in range(5):
    Hk = H_of(rng.uniform(-0.5, 0.5, 3))
    herm = max(herm, la.norm(Hk - Hk.conj().T))
gate("A2g H(k) Hermitian by construction (5 random k)", herm < 1e-12,
     f"max dev={herm:.1e}")

# real-space L=2 torus cross-check: spectra of iA match union of fibers
L = 2
cells = list(product(range(L), repeat=3))
cidx = {c: i for i, c in enumerate(cells)}
NS = len(cells) * 4


def torus_A(u_config):
    """Real-space antisymmetric A on the L^3 torus; u_config[b, cell]."""
    A = np.zeros((NS, NS))
    for d in cells:
        for b, (i, j, c) in enumerate(UBONDS):
            d2 = tuple((d[m] + c[m]) % L for m in range(3))
            u = u_config[b][cidx[d]]
            A[cidx[d2] * 4 + j, cidx[d] * 4 + i] += u
            A[cidx[d] * 4 + i, cidx[d2] * 4 + j] -= u
    return A


u_uniform = {b: np.ones(len(cells)) for b in range(NB)}
ev_torus = np.sort(la.eigvalsh(1j * torus_A(u_uniform)))
ev_fibers = np.sort(np.concatenate(
    [la.eigvalsh(H_of(np.asarray(d, float) / L)) for d in cells]))
gate("A2g' L=2 torus real-space spectrum = union of fiber spectra",
     np.max(np.abs(ev_torus - ev_fibers)) < 1e-9,
     f"max dev={np.max(np.abs(ev_torus - ev_fibers)):.1e}")

# A3g: gaplessness + codim-1 zero surface
def min_eps_grid(n):
    ks = (np.arange(n) + 0.5) / n - 0.5
    vals = np.empty((n, n, n))
    for a, ka in enumerate(ks):
        for bb, kb in enumerate(ks):
            for cc, kc in enumerate(ks):
                e = la.eigvalsh(H_of(np.array([ka, kb, kc])))
                vals[a, bb, cc] = np.min(np.abs(e))
    return vals


n_grid = 24
vals = min_eps_grid(n_grid)
gap_min = float(vals.min())
fracs = {d: float(np.mean(vals < d)) for d in (0.05, 0.1, 0.2)}
# codim-1: fraction ~ delta^1 => log-log slope ~ 1
import math
slope = (math.log(fracs[0.2]) - math.log(fracs[0.05])) / (math.log(0.2) - math.log(0.05))
gate("A3g GAPLESS with a codim-1 Majorana FERMI SURFACE "
     "(fraction(delta) ~ delta^1, not delta^3)",
     gap_min < 5e-2 and 0.6 < slope < 1.4,
     f"min|eps|={gap_min:.3f} (24^3 grid), scaling slope={slope:.2f}, "
     f"fracs={fracs}")

# A4g: Z2 gauge structure on the L=2 torus
site0 = 0  # (cell (0,0,0), vertex 0): flip u on all bonds incident to it
u_gauge = {b: np.ones(len(cells)) for b in range(NB)}
for b, (i, j, c) in enumerate(UBONDS):
    if i == 0:
        u_gauge[b][cidx[(0, 0, 0)]] *= -1
    if j == 0:
        # bond arriving at vertex 0 of cell (0,0,0) originates in cell -c
        d_src = tuple((-c[m]) % L for m in range(3))
        u_gauge[b][cidx[d_src]] *= -1
ev_gauge = np.sort(la.eigvalsh(1j * torus_A(u_gauge)))

u_flux = {b: np.ones(len(cells)) for b in range(NB)}
u_flux[0][cidx[(0, 0, 0)]] = -1.0
ev_flux = np.sort(la.eigvalsh(1j * torus_A(u_flux)))

gate("A4g Z2 gauge structure: node gauge move leaves spectrum IDENTICAL; "
     "single-bond flip (flux change) CHANGES it",
     np.max(np.abs(ev_gauge - ev_torus)) < 1e-12
     and np.max(np.abs(ev_flux - ev_torus)) > 1e-3,
     f"gauge dev={np.max(np.abs(ev_gauge - ev_torus)):.1e}, "
     f"flux dev={np.max(np.abs(ev_flux - ev_torus)):.3f}")

print("\n" + "=" * 72)
if FAILURES:
    print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
    sys.exit(1)
print(" RESULT: ALL GATES PASS -- anchor established (K4 not fired)")
print("=" * 72)
sys.exit(0)
