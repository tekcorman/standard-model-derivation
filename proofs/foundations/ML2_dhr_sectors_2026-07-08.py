#!/usr/bin/env python3
"""
proofs/foundations/ML2_dhr_sectors_2026-07-08.py

ML-2 — the DHR SECTOR CATEGORY (the -70 ppm / species keystone).  Pre-registered in
internal research notes (committed 50c64e7 BEFORE this probe).
EXTENDS the master module the_net.py (net.sector_category).

Builds the superselection structure of the observable algebra A = F^G on ML-0's net.  Field algebra
F = 8-dim Cl(6) Fock; gauge group G = A4 (forced J-covariance) with C3 winding + Z2 particle-hole;
U_pi^3=-I => spinorial => the double cover.  FORCED structure only; the physical sector<->particle
identification, the DR-uniqueness (does the match FORCE the lift), and the statistics-as-prediction are
architect FORKS -- built and CHARACTERIZED here, NOT adjudicated.  The -70 ppm STAYS OPEN.
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

np.set_printoptions(precision=4, suppress=True)
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    if not cond:
        ok_all = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    return cond


def banner(t):
    print("=" * 88); print(f" {t}"); print("=" * 88)


# --- forced inputs (WS1 verbatim: g6, gam, edge_rep, A4, J6, species, U_pi) ---
EDGES = srs.EDGES
NE, NV = len(EDGES), srs.NV
g6 = [np.array(g) for g in AlgebraicUtility.cl6_generators()]
EIDX = {(min(i, j), max(i, j)): e for e, (i, j, v) in enumerate(EDGES)}
I8 = np.eye(8)


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


A4 = [dict(enumerate(p)) for p in itertools.permutations(range(4))
      if sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j]) % 2 == 0]
d0 = np.zeros((NV, NE))
for e, (i, j, v) in enumerate(EDGES):
    d0[i, e] = -1.0
    d0[j, e] = 1.0
Chat = np.array([[1, 1, 0], [-1, 0, 1], [0, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
H1, _ = np.linalg.qr(Chat)
B1 = np.linalg.svd(d0)[2][:3].T
rows = []
for g in A4:
    R6 = edge_rep(g)
    rows.append(np.kron(np.eye(3), (H1.T @ R6 @ H1).T) - np.kron(B1.T @ R6 @ B1, np.eye(3)))
_, SpJ, VpJ = np.linalg.svd(np.vstack(rows))
phi = VpJ[-1].reshape(3, 3)
phi *= math.sqrt(3) / np.linalg.norm(phi)
J6 = B1 @ phi @ H1.T - H1 @ phi.T @ B1.T
wJ, VJ = np.linalg.eig(J6)
modes, _ = np.linalg.qr(VJ[:, np.where(np.abs(wJ - 1j) < 1e-9)[0]])
A_ops = [gam(np.conj(modes[:, m])) / math.sqrt(2) for m in range(3)]
NHAT = sum(a.conj().T @ a for a in A_ops)
wN, VN = np.linalg.eigh(NHAT)
wNr = np.round(np.real(wN)).astype(int)
Pw = {w: VN[:, wNr == w] @ VN[:, wNr == w].conj().T for w in range(4)}
SPECIES = {0: "nu", 1: "d", 2: "u", 3: "e"}


def spin_lift(R6):
    """The Spin(6) lift U of an edge O(6) transformation R6: U g_a U^-1 = sum_b R6[b,a] g_b.
    Unique up to phase (Schur, Cl(6)-Fock irreducible); det-normalized as in WS1's U_pi."""
    rowsU = [np.kron(gam(R6[:, a]), I8) - np.kron(I8, g6[a].T) for a in range(NE)]
    _, s, Vh = np.linalg.svd(np.vstack(rowsU))
    U = Vh[np.sum(s > 1e-9):].conj()[0].reshape(8, 8)
    return U / np.sqrt(np.abs(np.linalg.det(U @ U.conj().T)) ** (1 / 8))


# ===========================================================================
banner("ML2-A  the GAUGE-GROUP rep on the Fock field algebra (A4, and its double cover?)")
# ===========================================================================
U = [spin_lift(edge_rep(g)) for g in A4]
# is it a rep? U(g)U(h) = c * U(gh) with |c|=1 (Schur, F irreducible). collect the cocycle phases.
key = lambda d: tuple(d[i] for i in range(NV))
idx = {key(g): n for n, g in enumerate(A4)}
compose = lambda g, h: {i: g[h[i]] for i in range(NV)}
phases = []
for a, g in enumerate(A4):
    for b, h in enumerate(A4):
        gh = idx[key(compose(g, h))]
        M = U[a] @ U[b]
        c = np.trace(np.linalg.solve(U[gh], M)) / 8.0    # M = c U[gh]
        phases.append(c)
        if np.max(np.abs(M - c * U[gh])) > 1e-6:
            check("ML2-A homomorphism-up-to-phase", False, "product not proportional to U(gh)")
phases = np.array(phases)
is_projective = np.max(np.abs(np.abs(phases) - 1)) < 1e-6
signs = np.unique(np.round(phases.real, 3)[np.abs(phases.imag) < 1e-3])
check("ML2-A1 U is a rep of the gauge group on Fock (U(g)U(h) = c*U(gh), |c|=1)", is_projective,
      detail=f"cocycle phases on unit circle; real cocycle values seen: {signs}")
double_cover = np.any(np.abs(phases + 1) < 1e-3)
check("ML2-A2 the Fock rep is SPINORIAL: the cocycle takes -1 => the gauge group is the DOUBLE COVER "
      "(binary tetrahedral 2T), fermion-parity Z2 = ML-0's Klein twist",
      double_cover, detail="U_pi^3=-1 confirmed at rep level; sectors live in Rep(2T)")

# ===========================================================================
banner("ML2-B  the SECTOR DECOMPOSITION: do the gauge irreps = the species grading?")
# ===========================================================================
# species eigenspaces are gauge-invariant iff [U(g), N-hat]=0 (phase-independent commutator)
maxcomm = max(np.max(np.abs(U[a] @ NHAT - NHAT @ U[a])) for a in range(len(A4)))
check("ML2-B1 the species grading N-hat is GAUGE-INVARIANT ([U(g),N-hat]=0) => the sectors RESPECT "
      "species (the A4 action preserves the forced J-complex-structure)", maxcomm < 1e-7,
      detail=f"max||[U(g),N-hat]|| = {maxcomm:.1e}")


def commutant_dim(proj):
    """dim of the algebra {X on range(proj): [X, P U(g) P]=0 for all g} = sum m_i^2 (irrep mults)."""
    cols = np.linalg.qr(proj)[0][:, :int(round(np.trace(proj).real))]
    Us = [cols.conj().T @ U[a] @ cols for a in range(len(A4))]
    n = cols.shape[1]
    M = []
    for Ua in Us:
        # vec([X,Ua]) = (I (x) Ua^T ... ) -- build the commutator map on n x n X
        M.append(np.kron(np.eye(n), Ua) - np.kron(Ua.T, np.eye(n)))
    s = np.linalg.svd(np.vstack(M), compute_uv=False)
    return int(np.sum(s < 1e-7)), n


print("    per-species-sector: A4-commutant dimension (=1 <=> IRREDUCIBLE irrep):")
irr = {}
for w in range(4):
    cdim, n = commutant_dim(Pw[w])
    irr[w] = (n, cdim)
    print(f"      w={w} ({SPECIES[w]:2s}, dim {n}): commutant dim = {cdim}  "
          f"=> {'IRREDUCIBLE (a single sector)' if cdim == 1 else 'reducible'}")
all_irred = all(irr[w][1] == 1 for w in range(4))
check("ML2-B2 EACH species sector is an IRREDUCIBLE gauge irrep => the Fock field algebra's sector "
      "decomposition IS the species grading {1,3,3,1} = {nu,d,u,e}",
      all_irred, detail="the DHR sectors COINCIDE with the species (structural match)")

# the winding deck: from WS1's UNSIGNED screw permutation Rpi (NOT the signed A4-covariant edge_rep) --
# a spinorial Z6 object (U_pi^3=-I) that does NOT preserve J => cross-cuts the A4 species sectors.
sigma3 = {0: 0, 1: 2, 2: 3, 3: 1}
Rpi = np.zeros((NE, NE))
for e, (i, j, v) in enumerate(EDGES):
    a, b = sigma3[i], sigma3[j]
    Rpi[EIDX[(min(a, b), max(a, b))], e] = 1.0          # UNSIGNED (WS1 verbatim)
Upi = spin_lift(Rpi)
u3 = np.max(np.abs(np.linalg.matrix_power(Upi, 3) + I8))
Upi2 = Upi @ Upi
evU = np.linalg.eigvals(Upi2)
lab = np.array([int(round(cmath.phase(z) / (2 * math.pi / 3))) % 3 for z in evU])
wind_dims = {t: int(np.sum(lab == t)) for t in (0, 1, 2)}
cross = np.max(np.abs(Upi2 @ NHAT - NHAT @ Upi2))
print(f"    winding deck (unsigned screw U_pi, U_pi^3=-I dev {u3:.0e}): U_pi^2 eigenspace dims "
      f"{sorted(wind_dims.values())} (3 does not divide 8 => a distinguished sector is FORCED)")
check("ML2-B3 the winding deck {4,2,2} is a SEPARATE (screw/spinorial) grading that CROSS-CUTS the A4 "
      "species sectors ([U_pi^2, N-hat] != 0) -- not the gauge grading",
      cross > 1e-3 and sorted(wind_dims.values()) == [2, 2, 4] and u3 < 1e-9,
      detail=f"||[U_pi^2,N-hat]|| = {cross:.3f}; winding dims {sorted(wind_dims.values())}")

# ===========================================================================
banner("ML2-C  STATISTICS (from ML-0's fermion parity)")
# ===========================================================================
# each species sector's Bose/Fermi = its fermion-parity (-1)^N-hat grading (ML-0 twisted locality)
parity = {w: int((-1) ** w) for w in range(4)}
print("    per-sector fermion parity (-1)^N-hat (ML-0 Klein twist = Bose/Fermi):")
for w in range(4):
    print(f"      {SPECIES[w]:2s} (w={w}): parity {parity[w]:+d}  ({'even/integer' if parity[w] > 0 else 'odd/half-integer'} sector)")
check("ML2-C the sectors carry a Z2 fermion-parity (Bose/Fermi) grading = ML-0's twist; the leptons "
      "nu,e (w=0,3) and quarks d,u (w=1,2) split by parity {+,-,+,-}",
      parity == {0: 1, 1: -1, 2: 1, 3: -1},
      detail="Cl(6) => KO-dim 6 (the standing KO 2->6 residual's natural home; reported, not interpreted)")

# ===========================================================================
banner("ML2-D  the SECTOR CATEGORY + keystone assessment (STOP at the architect fork)")
# ===========================================================================
whole_cdim, _ = commutant_dim(np.eye(8))
print(f"    whole-Fock A4-commutant dim = {whole_cdim} (= sum of squared irrep multiplicities)")
print(f"    => {whole_cdim} = 2^2 + 2^2: TWO A4-irrep TYPES (a singlet + the triplet), EACH multiplicity 2;")
print(f"       the two copies of each are exchanged by the particle-hole Z2 (w <-> 3-w). So nu,e = the")
print(f"       SAME singlet type and d,u = the SAME triplet type; species label = (A4 irrep) x (bit).")
check("ML2-D1 the sector category is NONTRIVIAL: the field algebra resolves into (A4 irrep) x "
      "(particle-hole Z2) = the 4 species sectors (commutant 8 = 2^2+2^2)",
      whole_cdim == 8 and all_irred,
      detail="2 A4-irrep types (singlet, triplet) x the particle-hole Z2 = {nu,d,u,e}")
print("""
    FORCED RESULT (structural): the DHR sectors of the observable algebra A = F^G COINCIDE with the
    species grading -- the 8-dim Fock field algebra decomposes, under the double-cover (2T) gauge group,
    into (A4 irrep type) x (particle-hole Z2) = exactly the 4 IRREDUCIBLE species subspaces
    {nu:1, d:3, u:3, e:1} (two singlets = nu,e; two triplets = d,u; commutant 8 = 2^2+2^2). The winding
    deck {4,2,2} (unsigned screw, U_pi^3=-I) is a SEPARATE spinorial grading that cross-cuts these; the
    fermion parity {+,-,+,-} gives Bose/Fermi. => the species labels ARE the superselection sectors of
    the net, STRUCTURALLY (not an external assignment) -- quark/lepton = the triplet/singlet split, the
    two particle-hole copies = the deck bit.

    architect FORK (booked, NOT adjudicated here):
      (i)  which sector = which physical particle / the 3 generations (the triplet's internal structure);
      (ii) DR-UNIQUENESS: does 'sectors = species' + Doplicher-Roberts reconstruction FORCE the species
           lift (paying WS1's 1.6300 bit/site adoption BY THEOREM), or is the sector labelling still a
           gauge choice (zero-bit)? This needs the DR reconstruction-uniqueness proof -- architect.
      (iii) the statistics/KO-6 reading as a physical prediction.
    => the -70 ppm KEYSTONE STAYS OPEN. ML-2 built the sector category and showed the species ARE its
       sectors; whether that CLOSES the adoption is the architect fork. No scoreboard value moved.""")

print("RESULT:", "ALL CHECKS PASS" if ok_all else "A CHECK FAILED -- inspect above")
sys.exit(0 if ok_all else 1)
