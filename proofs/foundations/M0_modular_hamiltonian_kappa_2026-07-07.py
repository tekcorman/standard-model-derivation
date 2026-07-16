#!/usr/bin/env python3
"""
proofs/foundations/M0_modular_hamiltonian_kappa_2026-07-07.py

M0 — THE MODULAR HAMILTONIAN OF THE ONE-BIT VACUUM (the substrate read). Pre-registered
in internal research notes (committed 3ba1633
BEFORE any probe code). Convention OWNED by the 2-mode control
proofs/foundations/M0_convention_control_2026-07-07.py (committed 20d794f, ALL LOCK).

WHAT THIS BUILDS (charting our own course; nothing imported):
  The framework's vacuum omega = the quasi-free (Gaussian) state of the A4-covariant
  complex structure J6 (WS1 S0, PROVEN unique up to the bit). Its covariance on the
  6-edge cell is C = (I + iJ6)/2 -- the master-lens 'C = I + iJ' object, exactly E2a's
  two-point function. This is a rank-3 PROJECTOR (pure state). Restricting C to an edge
  subregion A gives the reduced state's correlation C_A, hence the region's MODULAR
  (entanglement) Hamiltonian h_A = log((I - C_A)/C_A) in the OWNED convention.

STATIONS (pre-registered):
  S0     re-lock J6 (WS1 S0 verbatim), build C, verify pure/projector, Tr C = 3.
  M0-1   the modular spectrum + entanglement entropy of the vacuum on edge-regions;
         the cell's girth-cycle (triangle) region entropy = the first data point for
         the OEF-energy <-> modular-energy bridge.
  M0-2   the first-law / kappa BRIDGE at the level accessible in ONE cell; HONEST scope:
         the full kappa REGRESSION over the DeltaS ladder {1,2,3,4,6,13} needs the
         SUPERCELL Bloch covariance (booked, not faked).
  M0-4   H-BIT: does the bit sigma (J -> -J) act as the vacuum's MODULAR CONJUGATION?
         (allowed to KILL; not rescued.)
  M0-3   FLOW-ID (lightweight, structural): what natural single-particle operator does
         the modular generator h_A track?

POISONS (flagged, NEVER pattern-matched): 2a1^5, 2a1^3, 5/12, 0.197, and the NEW
2pi/ln2 family (A1 T_substrate cross-check ONLY, deferred to the supercell kappa read).
NO value moves off this probe. An open miss stays open.
"""
import cmath
import itertools
import math
import os
import sys

import numpy as np
from scipy.linalg import logm

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "bridge"))
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

# ===========================================================================
banner("S0  re-lock J6 (WS1 S0 verbatim) and build the vacuum covariance C = (I + iJ6)/2")
# ===========================================================================
EDGES = srs.EDGES
NE, NV = len(EDGES), srs.NV
EIDX = {(min(i, j), max(i, j)): e for e, (i, j, v) in enumerate(EDGES)}
print(f"    cell: NV={NV} vertices, NE={NE} edges")

def edge_rep(sig):
    R6 = np.zeros((NE, NE))
    for e, (i, j, v) in enumerate(EDGES):
        a, b = sig[i], sig[j]
        s = 1.0
        if a > b:
            a, b, s = b, a, -1.0
        R6[EIDX[(a, b)], e] = s
    return R6

# the canonical A4-covariant J (W1/E2c/WS1 verbatim)
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
jdim = int(np.sum(SpJ < 1e-9))
check(f"S0a C-UNIQUE-J re-lock: A4-covariance solution dim = {jdim} (J forced up to the bit)",
      jdim == 1 and SpJ[-2] > 1e-3)
phi = VpJ[-1].reshape(3, 3); phi *= math.sqrt(3) / np.linalg.norm(phi)
J6 = B1 @ phi @ H1.T - H1 @ phi.T @ B1.T

check("S0b J6 is a complex structure: J6^2 = -I (dev %.1e)" % np.max(np.abs(J6 @ J6 + np.eye(NE))),
      np.max(np.abs(J6 @ J6 + np.eye(NE))) < 1e-9)
check("S0c J6 real antisymmetric (Majorana covariance form)",
      np.max(np.abs(J6 + J6.T)) < 1e-9 and np.max(np.abs(J6.imag)) < 1e-12)

# the vacuum covariance (complex-fermion convention, OWNED): C = (I + iJ)/2
def cov(J, sign=+1):
    return (np.eye(NE) + 1j * sign * J) / 2.0
C = cov(J6, +1)
check("S0d C = (I+iJ)/2 Hermitian", np.max(np.abs(C - C.conj().T)) < 1e-12)
check("S0e C is a PROJECTOR (pure Gaussian state): C^2 = C (dev %.1e)"
      % np.max(np.abs(C @ C - C)), np.max(np.abs(C @ C - C)) < 1e-9)
check(f"S0f Tr C = {np.trace(C).real:.6f} = NE/2 = 3 (half-filling, 3 fermion modes)",
      abs(np.trace(C).real - NE / 2) < 1e-9)

# ===========================================================================
banner("M0-1  the modular Hamiltonian h_A and entanglement entropy on edge-regions")
# ===========================================================================
def H2(x):  # binary entropy in NATS
    x = np.clip(x, 1e-15, 1 - 1e-15)
    return -x * np.log(x) - (1 - x) * np.log(1 - x)

def region_data(C, A):
    """Modular spectrum {eps_k} and entanglement entropy S_ent for edge-region A."""
    idx = np.ix_(A, A)
    C_A = C[idx]
    zeta = np.linalg.eigvalsh(C_A).real
    zc = np.clip(zeta, 1e-12, 1 - 1e-12)
    eps = np.log((1 - zc) / zc)              # single-particle modular energies
    S = float(np.sum(H2(zeta)))              # entanglement entropy (nats)
    return zeta, eps, S, C_A

# entanglement entropy over ALL edge-subsets, by region size
print("    entanglement entropy S_ent(vacuum, A) by region size |A| (nats):")
S_by_size = {}
for r in range(1, NE):
    Ss = []
    for A in itertools.combinations(range(NE), r):
        _, _, S, _ = region_data(C, list(A))
        Ss.append(S)
    S_by_size[r] = Ss
    print(f"      |A|={r}: min={min(Ss):.6f} max={max(Ss):.6f} "
          f"mean={np.mean(Ss):.6f}  ({len(Ss)} regions)")
# purity self-check: S(A) == S(complement) for the pure global state
sym_ok = True
for A in itertools.combinations(range(NE), 3):
    B = [e for e in range(NE) if e not in A]
    _, _, SA, _ = region_data(C, list(A))
    _, _, SB, _ = region_data(C, B)
    if abs(SA - SB) > 1e-9:
        sym_ok = False
check("M0-1a purity check: S(A)=S(complement) for all 3|3 edge splits (global state pure)",
      sym_ok)

# bit-evenness of the entanglement entropy (master lens: what the read sees is bit-even)
Cm = cov(J6, -1)
bit_even = True
for A in itertools.combinations(range(NE), 3):
    _, _, SA, _ = region_data(C, list(A))
    _, _, SAm, _ = region_data(Cm, list(A))
    if abs(SA - SAm) > 1e-9:
        bit_even = False
check("M0-1b entanglement entropy is BIT-EVEN (S[J]=S[-J]): the read sees the bit-even part",
      bit_even)

# ---- the cell's girth cycles (girth = 3 => triangles in the K4 cell) ----
# a girth cycle's edge set is a genuine F1 'constituent'; its region entropy is the
# first data point of the OEF-energy <-> modular-energy bridge.
import networkx as nx  # noqa: E402  (only for cycle enumeration; not part of the physics)
Gnx = nx.Graph()
for e, (i, j, v) in enumerate(EDGES):
    Gnx.add_edge(i, j, eidx=e)
girth = min(len(c) for c in nx.minimum_cycle_basis(Gnx)) if Gnx.number_of_edges() else 0
triangles = []
for tri in itertools.combinations(range(NV), 3):
    es = []
    ok = True
    for a, b in itertools.combinations(tri, 2):
        key = (min(a, b), max(a, b))
        if key in EIDX:
            es.append(EIDX[key])
        else:
            ok = False
    if ok and len(es) == 3:
        triangles.append(sorted(es))
print(f"    girth = {girth};  girth cycles (triangles) in the cell: {len(triangles)}")
for t in triangles:
    _, eps, S, _ = region_data(C, t)
    print(f"      triangle edges {t}: S_ent = {S:.6f} nats  modular spectrum eps = {np.round(np.sort(eps),4)}")

# ===========================================================================
banner("M0-2  the first-law / kappa BRIDGE -- what is accessible in ONE cell (HONEST scope)")
# ===========================================================================
# The OEF says E = kappa * L (L = MDL description length, in bits). The state's own
# modular structure gives S_ent (nats). The first law d<K> = dS ties modular energy to
# entanglement entropy. The kappa BRIDGE = whether the vacuum's entanglement entropy of a
# region tracks its MDL description length L, universally. To FORCE kappa we need a SLOPE:
# regions with DIFFERENT DeltaS. Within ONE K4 cell every triangle PAIR shares exactly one
# edge => DeltaS is a single value (no slope). We therefore REPORT the single-cell numbers
# blind and BOOK the regression for the supercell.

# single-cell overlapping girth-cycle pairs and their DeltaS (MDL) + mutual information (state)
def deltaS_and_MI(C, A1, A2):
    union = sorted(set(A1) | set(A2))
    # DeltaS (MDL): sum(mult_e - 1) - sum_v max(deg_v-2,0) on the union edge multiset
    mult = {}
    for e in list(A1) + list(A2):
        mult[e] = mult.get(e, 0) + 1
    share = sum(m - 1 for m in mult.values())
    deg = {}
    for e in union:
        i, j, v = EDGES[e]
        deg[i] = deg.get(i, 0) + 1; deg[j] = deg.get(j, 0) + 1
    junction = sum(max(d - 2, 0) for d in deg.values())
    dS = share - junction
    # mutual information of the vacuum: I = S(A1)+S(A2)-S(A1 u A2)
    _, _, S1, _ = region_data(C, list(A1))
    _, _, S2, _ = region_data(C, list(A2))
    _, _, Su, _ = region_data(C, union)
    return dS, S1 + S2 - Su

print("    single-cell girth-cycle (triangle) pairs: DeltaS (MDL, bits) vs vacuum MI (nats)")
rows_pair = []
for a, b in itertools.combinations(range(len(triangles)), 2):
    dS, MI = deltaS_and_MI(C, triangles[a], triangles[b])
    rows_pair.append((dS, MI))
    print(f"      triangles {a},{b}: DeltaS = {dS}  I_vac = {MI:.6f}  "
          f"I_vac/DeltaS = {MI/dS if dS else float('nan'):.6f}")
dS_vals = sorted(set(r[0] for r in rows_pair))
print(f"    distinct DeltaS values in one cell: {dS_vals}  "
      f"(=> {'NO slope; regression needs the supercell' if len(dS_vals)==1 else 'slope available'})")
print()
print("    HONEST FINDING (not glossed): DeltaS = -1 here. Two K4 triangles sharing ONE edge")
print("    create TWO degree-3 junctions; the junction cost (2) exceeds the shared-edge saving")
print("    (1) => net DeltaS < 0 => these single-cell pairs are ANTI-binding (correct F1 behavior,")
print("    E_bind = -kappa*DeltaS = +kappa). Meanwhile the vacuum mutual information I_vac = %.4f > 0"
      % rows_pair[0][1])
print("    ALWAYS (MI is non-negative). => the naive bridge 'S_ent = ln2 * L' / 'I_vac = ln2 * DeltaS'")
print("    is TOO CRUDE: I_vac tracks total shared CORRELATION (unsigned); DeltaS tracks net")
print("    description ECONOMY incl. junction costs (signed). The kappa bridge must relate the")
print("    modular ENERGY (first law), not the entropy directly, to E_bind. Booked for the regression.")
print()
print("    HONEST SCOPE: the kappa first-law REGRESSION over the DeltaS ladder {1,2,3,4,6,13}")
print("    requires the SUPERCELL Bloch covariance C(edge_a,edge_b)=int_BZ [(I+iJ(k))/2] e^{ik.dr},")
print("    where the modular ENERGY change d<K> (not S_ent) is read against E_bind = -kappa*DeltaS.")
print("    BOOKED as the immediate M0-2 continuation; NOT faked here. No value moves.")

# ===========================================================================
banner("M0-4  H-BIT: does the bit sigma (J -> -J) act as the modular CONJUGATION?")
# ===========================================================================
# Modular conjugation J_mod (antiunitary) must satisfy: J_mod^2 = 1, and it REVERSES the
# modular flow: J_mod K J_mod = -K (equivalently Delta -> Delta^{-1}). The bit sigma is
# complex conjugation of the state (antiunitary, sigma^2 = 1). Its action on the covariance:
bit_ph = np.max(np.abs(cov(J6, -1) - (np.eye(NE) - C)))
check("M0-4a EXACT: C(-J) = I - C(J)  (the bit is PARTICLE-HOLE on the covariance; dev %.1e)"
      % bit_ph, bit_ph < 1e-12)
# particle-hole sends zeta_k -> 1 - zeta_k on EVERY region => eps_k -> -eps_k => K_A -> -K_A
flow_rev = True
maxdev = 0.0
for A in itertools.combinations(range(NE), 3):
    z, eps, _, _ = region_data(C, list(A))
    zm, epsm, _, _ = region_data(Cm, list(A))
    d = np.max(np.abs(np.sort(eps) + np.sort(epsm)[::-1]))
    maxdev = max(maxdev, d)
    if d > 1e-7:
        flow_rev = False
check("M0-4b the bit REVERSES the modular flow: K_A -> -K_A on every region (eps -> -eps; dev %.1e)"
      % maxdev, flow_rev)
print("    H-BIT verdict: the bit sigma implements the ANTIUNITARY, FLOW-REVERSING (K->-K) half of")
print("    the modular conjugation EXACTLY (via the exact particle-hole identity C(-J)=I-C(J)).")
print("    Full modular conjugation J_mod = sigma  o  (region A<->B geometric swap): the bit supplies")
print("    the state (particle-hole) half; the geometric half is the wedge reflection. => H-BIT PARTIAL")
print("    (NOT a naive pass): the master chirality lens's bit-odd = modular-odd gets its EXACT home in")
print("    the flow-reversal, but 'bit = J_mod' in full requires the geometric reflection too. Booked.")

# ===========================================================================
banner("M0-3  FLOW-ID (lightweight, structural): what does the modular generator track?")
# ===========================================================================
# Full operator FLOW-ID vs dN / G_int needs those objects on the 6-edge single-particle
# space; deferred with the supercell build. Here: structural reads on the FULL-CELL
# 'modular data' -- does h_A relate to J itself / the adjacency A / N-hat structure?
# The single-particle modular Hamiltonian for the FULL cell is trivial (pure state), so we
# examine the region generator's structure for the 3-edge regions.
A0 = triangles[0] if triangles else [0, 1, 2]
z0, eps0, S0v, C_A0 = region_data(C, A0)
h_A0 = logm((np.eye(len(A0)) - C_A0) @ np.linalg.inv(C_A0))
# does h_A0 commute with the restricted J (i.e. is the modular flow J-covariant)?
J_A0 = J6[np.ix_(A0, A0)]
comm_J = np.max(np.abs(h_A0 @ (1j * J_A0) - (1j * J_A0) @ h_A0))
print(f"    region {A0}: modular spectrum eps = {np.round(np.sort(eps0),4)}, S_ent = {S0v:.6f}")
print(f"    [h_A, iJ_A] max = {comm_J:.4f}  (0 => modular flow is J-covariant on the region)")
print("    Full FLOW-ID (h_A vs dN vs G_int at the operator level) BOOKED with the supercell build.")

# ===========================================================================
banner("SUMMARY")
# ===========================================================================
print("""    M0-1  BUILT: the modular (entanglement) Hamiltonian of the one-bit vacuum on the
          6-edge cell, from the OWNED convention. C=(I+iJ)/2 is an exact rank-3 projector;
          edge-region entropies computed; entanglement entropy is BIT-EVEN (master lens).
    M0-4  H-BIT PARTIAL (exact): the bit sigma = J->-J = EXACT particle-hole C(-J)=I-C(J)
          => reverses the modular flow K_A -> -K_A on every region. This is the antiunitary
          flow-reversing HALF of the modular conjugation, derived exactly. 'bit = J_mod' in
          FULL additionally needs the geometric A<->B reflection. bit-odd = modular-odd has
          its exact home. NOT a naive pass; NOT killed -- refined and banked.
    M0-2  kappa first-law read: single cell gives ONE DeltaS value (no slope). The forcing
          REGRESSION over the DeltaS ladder needs the SUPERCELL Bloch covariance. BOOKED,
          not faked. No scoreboard value moved.
    M0-3  FLOW-ID: structural reads only this pass; operator-level vs dN/G_int booked.
    POISONS: none invoked; 2pi/ln2 cross-check deferred to the supercell kappa read.""")
print("RESULT:", "ALL CHECKS PASS" if ok_all else "A CHECK FAILED -- inspect above")
sys.exit(0 if ok_all else 1)
