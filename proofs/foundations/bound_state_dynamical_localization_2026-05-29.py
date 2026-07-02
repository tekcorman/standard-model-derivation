#!/usr/bin/env python3
# ============================================================
# Bound-state, STAGE 2c: DYNAMICAL LOCALIZATION (real-time evolution)
# ============================================================
#
# Scoping: docs/scoping/bound_state_sector_scoping_2026-05-28.md (action F1).
# THE INDEPENDENT ARBITER. Stages 1.5/2/2b all routed through a loop integral
# Pi(E)/critical-coupling U_c, and that route had repeated NORMALIZATION bugs:
#   - Stage 1.5: finite-grid threshold spike faked a pole.
#   - Stage 2 / 2b: AVERAGING the loop over all Dirac band-pair channels (256x, and
#     16x for adjacency) diluted the threshold DOS -> inflated U_c. The whole
#     "adjacency doesn't bind / Dirac binds (flip)" narrative was that artifact:
#     single-band-normalized U_c is ~0.99 (adjacency) / ~0.26 (Dirac), so the
#     contact kernel U=3 binds for BOTH.
# This probe uses NONE of that machinery. It evolves two walkers in REAL TIME on
# the srs lattice and asks the bluntest possible question: with the MDL-entropic
# attraction switched on, do two initially-coincident walkers STAY together, or
# disperse like free particles? Localization that survives (and vanishes when the
# attraction is off) is the dynamical signature of a bound state.
#
# MODEL:
#   - srs primitive-cell supercell (L^3 cells x 4 atoms), adjacency hopping (t=1):
#     H_1 = A (the walker's kinetic operator).
#   - two distinguishable walkers: H_2 = H_1 (x) I + I (x) H_1 + V_int,
#     V_int = -U at coincidence (both walkers on the same vertex), U = dS*e_bit.
#     (Contact attraction; Stage 2b showed the srs edge-resolved kernel is
#     effectively contact, range 0.)
#   - initial state: both walkers on the same central vertex (maximal overlap).
#   - evolve psi(t) = exp(-i H_2 t) psi(0); measure
#       P_same(t)  = prob. both walkers on the same vertex,
#       <D>(t)     = mean graph distance between the two walkers.
#   - compare U=3 (MDL reward) vs U=0 (free): localization persisting only for U>0
#     is the bound state. Cross-checked by eigsh: lowest H_2 eigenvalue below the
#     2-particle continuum bottom 2*min(spec A) confirms a true bound state.
#
# This is independent of pole/loop normalization. Standing conditionals remain:
# U = dS*e_bit (e_bit=t=1), contact kernel, adjacency dispersion (the Dirac binds
# deeper but is too heavy for two-particle real-time on this lattice).

import os
import sys
import numpy as np
from itertools import product
from collections import defaultdict, deque
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply, eigsh

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from proofs.common import find_bonds  # noqa: E402

E_BIT = 1.0
DS = 3.0                 # Stage-0 max MDL reward (s=5 fused half-cycle)


def build_srs(L):
    """Adjacency (sparse) of an L^3 primitive-cell srs supercell (4 atoms/cell)."""
    bonds = find_bonds()
    Nv = L * L * L * 4

    def vid(n0, n1, n2, iv):
        return (((n0 % L) * L + (n1 % L)) * L + (n2 % L)) * 4 + iv

    rows, cols = [], []
    adj = defaultdict(set)
    for src, tgt, cell in bonds:
        c0, c1, c2 = int(cell[0]), int(cell[1]), int(cell[2])
        for n0, n1, n2 in product(range(L), repeat=3):
            a = vid(n0, n1, n2, src)
            b = vid(n0 + c0, n1 + c1, n2 + c2, tgt)
            for (x, y) in ((a, b), (b, a)):
                rows.append(x); cols.append(y)
                adj[x].add(y)
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(Nv, Nv))
    A = ((A + A.T) > 0).astype(float)        # symmetrize, 0/1
    deg = np.array(A.sum(1)).ravel()
    assert np.all(deg == 3), f"not 3-regular: degrees {set(deg)}"
    return A.tocsr(), adj, Nv


def graph_distances(adj, Nv):
    """All-pairs BFS distance matrix (Nv x Nv)."""
    D = np.full((Nv, Nv), -1, dtype=np.int16)
    for s in range(Nv):
        D[s, s] = 0
        q = deque([s])
        while q:
            v = q.popleft()
            for w in adj[v]:
                if D[s, w] < 0:
                    D[s, w] = D[s, v] + 1
                    q.append(w)
    return D


def two_particle_H(A, Nv, U):
    """H_2 = A (x) I + I (x) A - U * (coincidence projector)."""
    I = sp.identity(Nv, format="csr")
    H = sp.kron(A, I) + sp.kron(I, A)
    same = np.array([i * Nv + i for i in range(Nv)])
    Vdiag = np.zeros(Nv * Nv)
    Vdiag[same] = -U
    H = H + sp.diags(Vdiag)
    return H.tocsr(), same


def run(L=4, T=8.0, Nt=17):
    A, adj, Nv = build_srs(L)
    D = graph_distances(adj, Nv)
    spec_min = sp.linalg.eigsh(A, k=1, which="SA", return_eigenvectors=False)[0]
    cont_bottom = 2.0 * spec_min     # 2-particle continuum bottom (with H_1=A)
    print(f"srs supercell L={L}: {Nv} vertices (3-regular). "
          f"single-particle band min(A)={spec_min:.3f}; 2-particle continuum bottom={cont_bottom:.3f}")

    i0 = Nv // 2
    psi0 = np.zeros(Nv * Nv, dtype=complex)
    psi0[i0 * Nv + i0] = 1.0          # both walkers on the same central vertex
    Dflat = D.astype(float).ravel()

    results = {}
    for U in (0.0, 1.0, 3.0):
        H, same = two_particle_H(A, Nv, U)
        # bound-state check via lowest eigenvalue
        e_lo = eigsh(H, k=1, which="SA", return_eigenvectors=False)[0]
        bound = e_lo < cont_bottom - 1e-6
        # real-time evolution
        states = expm_multiply(-1j * H, psi0, start=0.0, stop=T, num=Nt)
        P_same = np.array([np.sum(np.abs(s[same]) ** 2) for s in states])
        Dmean = np.array([np.sum(Dflat * np.abs(s) ** 2) for s in states])
        results[U] = (P_same, Dmean, e_lo, bound)
        print(f"\nU={U}: lowest H_2 eigenvalue {e_lo:.3f}  "
              f"({'BELOW continuum -> BOUND STATE EXISTS' if bound else 'in continuum -> no bound state'})")
        print(f"   P_same(t): " + "  ".join(f"{p:.3f}" for p in P_same[::4]))
        print(f"   <D>(t):    " + "  ".join(f"{d:.2f}" for d in Dmean[::4]))

    ts = np.linspace(0, T, Nt)
    print("\n" + "=" * 72)
    print("VERDICT (dynamical, independent of loop/pole normalization)")
    print("=" * 72)
    Pfree = results[0.0][0]
    Pbound = results[3.0][0]
    Dfree = results[0.0][1]
    Dbound = results[3.0][1]
    tail = slice(Nt // 2, Nt)         # second half = long-time behaviour
    print(f"  long-time-avg P_same:  U=0 (free) {Pfree[tail].mean():.3f}   "
          f"U=3 (MDL)  {Pbound[tail].mean():.3f}")
    print(f"  long-time-avg <D>:     U=0 (free) {Dfree[tail].mean():.2f}    "
          f"U=3 (MDL)  {Dbound[tail].mean():.2f}")
    localizes = (Pbound[tail].mean() > 2 * max(Pfree[tail].mean(), 1e-9)) and \
                (Dbound[tail].mean() < Dfree[tail].mean())
    if results[3.0][3] and localizes:
        print(f"\n  CONFIRMED (independent method): with the MDL-entropic attraction the two")
        print(f"  walkers STAY together (P_same stays high, <D> stays small); with it off")
        print(f"  they disperse. A true bound state (lowest eigenvalue below the 2-particle")
        print(f"  continuum) exists. This reproduces the binding by REAL-TIME EVOLUTION,")
        print(f"  using none of the loop/Pi/U_c machinery that carried the earlier bugs.")
    elif results[3.0][3]:
        print(f"\n  PARTIAL: a bound state exists (eigenvalue below continuum) but the")
        print(f"  finite-lattice dynamics didn't cleanly show localization contrast.")
    else:
        print(f"\n  NOT BOUND dynamically: no localization contrast and no sub-continuum")
        print(f"  eigenvalue. Contradicts the loop-route binding -> the loop result was")
        print(f"  an artifact after all.")
    print(f"\n  Flagged: adjacency dispersion (Dirac binds deeper, too heavy here);")
    print(f"  contact kernel; U=dS*e_bit with e_bit=t; finite L={L} periodic lattice;")
    print(f"  distinguishable walkers. Independent of pole/loop normalization.")
    return results


if __name__ == "__main__":
    run()
