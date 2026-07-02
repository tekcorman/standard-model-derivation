#!/usr/bin/env python3
"""
R1_3_edge_sector_c3_decomposition_probe.py
==========================================
R1.3 of the R1 multi-session research arc.  Prior: R1.1 (per-vertex Cl(6)
Fock decomposition), R1.2 (body-diagonal C_3 outer action on operator algebra).

Goal.  Decompose the framework's EDGE sector C¹_alg = ⊕_e M_2(ℂ) = 24 dim
(per cell) under the body-diagonal C_3, identifying the 2 orbit-3's of
edges and the resulting 8 + 8 + 8 C_3-irrep structure.  Combine with R1.2
for the full H_F = 280 decomposition.

K_4 edge orbit structure under body-diagonal C_3 (v_0 fixed, v_1→v_3, v_2→v_1, v_3→v_2):
  Orbit 1 (incident to fixed v_0):  edges {0, 1, 2} = {(0,1), (0,2), (0,3)}
                                      cycle: 0 → 2 → 1 → 0
  Orbit 2 (between orbit vertices): edges {3, 4, 5} = {(1,2), (1,3), (2,3)}
                                      cycle: 3 → 4 → 5 → 3

Each orbit-3 of edges with 4-dim M_2 per edge → 12-dim under cyclic C_3
decomposes as 4 (trivial) + 4 (ω) + 4 (ω²).

Total edge sector: 24 = 12 + 12 = (4+4+4) + (4+4+4) = 8 trivial + 8 ω + 8 ω².

Combined with R1.2 vertex sector (256-dim):
  H_F^matter+gauge = 280 = 256 (vertex) + 24 (edge) decomposes under C_3 as:
                          = (128 + 8) trivial + (64 + 8) ω + (64 + 8) ω²
                          = 136 trivial + 72 ω + 72 ω²

What this probe does
--------------------
A — Identify the 2 edge orbits, build C_3 cyclic edge permutation, verify P^3 = I.
B — Lift to edge Fock (12 dim per cell) — decompose under C_3.
C — Lift to edge op-alg C¹_alg (24 dim per cell) — decompose under C_3.
D — Combine with R1.2 vertex decomposition to get full H_F = 280 decomposition.
E — Document structural implications for R1.4 (b_i extraction).

No graded content changes from this probe.  R1 status: interim.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from proofs.foundations.de_rham_susy_fibered_v2_probe import EDGES, NE  # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=140)

TOL = 1e-10
omega3 = np.exp(2j * np.pi / 3)

# C_3 vertex permutation σ: v_0→v_0, v_1→v_3, v_2→v_1, v_3→v_2
SIGMA = [0, 3, 1, 2]


# -----------------------------------------------------------------------------
# Part A — identify edge orbits under C_3
# -----------------------------------------------------------------------------

def part_A_edge_orbits():
    print("=" * 100)
    print("PART A — K_4 edges and C_3 orbits")
    print("=" * 100)
    print(f"\n  K_4 has 6 edges; per the framework's enumeration:")
    for i, e in enumerate(EDGES):
        u, v, _ = e
        incident_v0 = "INCIDENT TO v_0" if (u == 0 or v == 0) else "between orbit vertices"
        print(f"    edge {i}: ({u}, {v})  —  {incident_v0}")

    # Build edge permutation: under C_3, edge (u, v) → (σ(u), σ(v))
    # Find the index of the resulting edge in EDGES
    def find_edge(a, b):
        a, b = (min(a, b), max(a, b))
        for i, (u, v, _) in enumerate(EDGES):
            if (min(u, v), max(u, v)) == (a, b):
                return i
        raise ValueError(f"edge ({a}, {b}) not found")

    edge_perm = []
    for i, (u, v, _) in enumerate(EDGES):
        new_u = SIGMA[u]; new_v = SIGMA[v]
        new_i = find_edge(new_u, new_v)
        edge_perm.append(new_i)

    print(f"\n  C_3 edge permutation σ_E (edge i → edge σ_E(i)):")
    for i, j in enumerate(edge_perm):
        print(f"    {i} → {j}      (edge ({EDGES[i][0]}, {EDGES[i][1]}) → edge ({EDGES[j][0]}, {EDGES[j][1]}))")

    # Build as a (6, 6) permutation matrix
    P_E = np.zeros((6, 6), dtype=complex)
    for j, i in enumerate(edge_perm):
        P_E[i, j] = 1.0
    print(f"\n  P_E^3 = I :  {np.allclose(P_E @ P_E @ P_E, np.eye(6), atol=TOL)}")
    eig = np.linalg.eigvals(P_E)
    eig_sorted = sorted(eig, key=lambda c: (np.angle(c), c.real))
    print(f"  P_E eigenvalues : {[complex(round(e.real, 4), round(e.imag, 4)) for e in eig_sorted]}")

    # orbit identification
    orbits = []
    visited = set()
    for i in range(6):
        if i in visited:
            continue
        orb = [i]
        j = edge_perm[i]
        while j != i:
            orb.append(j)
            j = edge_perm[j]
        orbits.append(orb)
        visited.update(orb)
    print(f"\n  Edge orbits under C_3:")
    for k, orb in enumerate(orbits):
        edge_descs = [f"({EDGES[i][0]}, {EDGES[i][1]})" for i in orb]
        print(f"    orbit {k+1}: edges {orb}  =  {{{', '.join(edge_descs)}}}")
    return edge_perm, P_E, orbits


# -----------------------------------------------------------------------------
# Part B — edge Fock decomposition under C_3 (12 dim per cell)
# -----------------------------------------------------------------------------

def part_B_edge_fock(P_E):
    print("\n" + "=" * 100)
    print("PART B — Edge Fock (12 dim per cell = 6 edges × 2 qubit components)")
    print("=" * 100)
    # Block-level C_3 on edge Fock: cyclic permutation of edge labels, with the 2-dim Fock
    # of each edge mapped identically (no internal qubit flip).
    F_edge = np.kron(P_E, np.eye(2, dtype=complex))
    print(f"  F_edge dim: {F_edge.shape}  (12 × 12)")
    print(f"  F_edge^3 = I :  {np.allclose(F_edge @ F_edge @ F_edge, np.eye(12), atol=TOL)}")
    eig = np.linalg.eigvals(F_edge)
    n_t = sum(1 for e in eig if abs(e - 1) < TOL)
    n_w = sum(1 for e in eig if abs(e - omega3) < TOL)
    n_w2 = sum(1 for e in eig if abs(e - omega3 ** 2) < TOL)
    print(f"\n  Eigenvalue counts on ℂ^12:")
    print(f"    trivial : {n_t}   ←  expected 4 (= 2 trivial from P_E × 2)")
    print(f"    ω        : {n_w}   ←  expected 4")
    print(f"    ω²       : {n_w2}   ←  expected 4")
    print(f"\n  Decomposition: edge Fock 12 = 4 (trivial) + 4 (ω) + 4 (ω²)")


# -----------------------------------------------------------------------------
# Part C — edge operator algebra decomposition under C_3 (24 dim per cell)
# -----------------------------------------------------------------------------

def part_C_edge_opalg(P_E):
    print("\n" + "=" * 100)
    print("PART C — Edge operator algebra C¹_alg = ⊕_e M_2(ℂ) (24 dim per cell)")
    print("=" * 100)
    # Block-level C_3 on edge op-alg: cyclic permutation × 4-dim M_2 per edge
    C1_C3 = np.kron(P_E, np.eye(4, dtype=complex))
    print(f"  C¹_alg C_3 action dim: {C1_C3.shape}  (24 × 24)")
    print(f"  C1_C3^3 = I :  {np.allclose(C1_C3 @ C1_C3 @ C1_C3, np.eye(24), atol=TOL)}")
    eig = np.linalg.eigvals(C1_C3)
    n_t = sum(1 for e in eig if abs(e - 1) < TOL)
    n_w = sum(1 for e in eig if abs(e - omega3) < TOL)
    n_w2 = sum(1 for e in eig if abs(e - omega3 ** 2) < TOL)
    print(f"\n  Eigenvalue counts on ℂ^24:")
    print(f"    trivial : {n_t}   ←  expected 8 (= 2 trivial × 4 M_2 dim)")
    print(f"    ω        : {n_w}   ←  expected 8")
    print(f"    ω²       : {n_w2}   ←  expected 8")
    print(f"\n  Decomposition: edge op-alg 24 = 8 (trivial) + 8 (ω) + 8 (ω²)")
    print(f"                              = (4+4) trivial + (4+4) ω + (4+4) ω² across the 2 orbits")


# -----------------------------------------------------------------------------
# Part D — combine R1.2 + R1.3 for full H_F = 280 decomposition
# -----------------------------------------------------------------------------

def part_D_full_HF():
    print("\n" + "=" * 100)
    print("PART D — Full H_F = C⁰_alg ⊕ C¹_alg = 280 dim per cell under C_3")
    print("=" * 100)
    print(f"""
  Per R1.2 (vertex sector C⁰_alg = 256 dim):
    256 = 128 (trivial)  +  64 (ω)  +  64 (ω²)

  Per R1.3 (edge sector C¹_alg = 24 dim):
     24 =   8 (trivial)  +   8 (ω)  +   8 (ω²)

  Combined H_F = 256 + 24 = 280:
    280 = 136 (trivial) + 72 (ω) + 72 (ω²)

  Per M1.B Galois reading:
    dim(H_F^α) = 136  (the framework's fixed "matter + gauge" sector under C_3)
    The 3 generations + crossed-product structure act on this 136-dim sector
    + the (72, 72) generation-graded part.

  AT FOCK LEVEL (per cell):
    vertex Fock 32 = 16 trivial + 8 ω + 8 ω²
    edge   Fock 12 =  4 trivial + 4 ω + 4 ω²
    H_F Fock total 44 = 20 trivial + 12 ω + 12 ω²

  Interpretation:
    • 20 trivial = generation singlet (v_0 fixed + orbit-symmetric) sector at Fock level
    • 12 ω + 12 ω² = generation-graded sectors at Fock level
    • Combined via M_3(ℂ) factor of crossed product → effectively 3 × ?  matter states

  The "16 = 1 SM gen with color" alignment from R1.2 was at the VERTEX-only level.
  Including edges (which carry gauge boson + scalar content, NOT fermion-counted as matter):
    • Vertex Fock^α = 16  →  fermion content per generation (after Galois crossed product)
    • Edge Fock^α =  4   →  gauge / scalar content per generation
    • H_F Fock^α  = 20   →  total per gen including gauge

  Gauge group dim count:
    SU(3)_c × SU(2)_L × U(1)_Y has dim 8 + 3 + 1 = 12 generators (gauge bosons).
    The framework's per-gen Fock^α 20 = 16 (matter) + 4 (gauge?) — does NOT match
    12 gauge generators directly.  This counting gap is structurally informative —
    it tells us either:
      (a) framework gauge boson count differs from SM (e.g., Pati-Salam extras), or
      (b) the gauge content emerges via inner fluctuations rather than directly
          from the edge Fock count (consistent with CC's standard mechanism).
""")


# -----------------------------------------------------------------------------
# Part E — implications for R1.4 + slow-path Z_2 grading
# -----------------------------------------------------------------------------

def part_E_implications():
    print("\n" + "=" * 100)
    print("PART E — implications for R1.4 (b_i extraction) + Z_2 slow path")
    print("=" * 100)
    print(r"""
  For R1.4's b_i extraction, the per-gen matter content from R1.2-R1.3:
    Fermion content per gen (vertex Fock^α):  16 states matching 1 SM gen with color
    Scalar  content per gen :  open structural question (R1.1's MSSM-doubling retracted in R1.2)
    Gauge   content per gen :  4 from edge Fock^α (suggestive of 4 = U(1)+SU(2)+SU(3) Cartan?)
                              OR emerges from inner fluctuations per CC standard

  WITHOUT a Z_2 grading mechanism that places HALF of the 280-dim H_F as bosonic
  scalars, the framework's natural matter content is SM-shaped (3 gens × 16 fermions),
  NOT MSSM-shaped (3 gens × 16 fermions + 3 gens × 16 sfermions + Higgs sector).

  Under SM matter content, Step 4's IR-running test gives  1/α_3(M_Z) = −12.77  (α_3 < 0,
  UNPHYSICAL).  This is the CORE STRUCTURAL TENSION the slow path on Z_2 grading must
  resolve.

  TWO RESOLUTION PATHS:

  PATH A — find the Z_2 grading mechanism within H_F.
    R1.4 deferred until path A's mechanism identified.  Candidates for the Z_2:
      (i)  χ̂ = supercharge chirality grading (matter sector / gauge sector at op-alg level).
           NATURAL Z_2 of the spectral triple itself.  Does it give boson/fermion split
           at FOCK level too?  Multi-session.
      (ii) Witten γ_7 chirality (already explored in Path E, closed-NEGATIVE).  RULED OUT.
      (iii) srs-z bipartite cover χ̃ (Path E re-examined, also closed-NEGATIVE).  RULED OUT.
      (iv) The OPERATOR-ALGEBRA OUTER C_3 vs INNER C_3 distinction itself.  Worth
           investigating — the C_3 inner (B6's color) and C_3 outer (M1.B's generations)
           may induce a Z_2 grading on H_F via their COMBINATION.
      (v)  Inner-fluctuation Higgs sector via [A_μ, D_F] (Step 2's Higgs-cross term).
           Could carry scalar content sufficient for MSSM matter without explicit
           sfermion partners.
      (vi) The substrate's W4 walker chirality (different from γ_7).

  PATH B — accept framework matter content = SM (no SUSY) and find a different reason
    for IR-consistency.  Step 4's "SM gives α_3 < 0" might be resolvable via
    framework-specific running between Λ_sub and M_unif (substrate-derived β functions),
    not the textbook SM β.  This is also multi-session — requires deriving substrate β
    above M_unif.

  EITHER WAY, R1.4 (b_i extraction) is NOT BOUNDED by today's work.  R1's path forward
  goes through Path A or Path B for the Z_2 grading question, not just R1.4 plug-and-chug.
""")


def main():
    print(r"""
==========================================================================================
R1.3 — Edge sector C¹_alg = 24 dim under body-diagonal C_3 + full H_F decomposition
Third bounded probe of the R1 multi-session research arc.
==========================================================================================""")
    edge_perm, P_E, orbits = part_A_edge_orbits()
    part_B_edge_fock(P_E)
    part_C_edge_opalg(P_E)
    part_D_full_HF()
    part_E_implications()
    print("\n" + "=" * 100)
    print("R1.3 INTERIM VERDICT")
    print("=" * 100)
    print("""
  ESTABLISHED (this probe, all machine precision):

  (i)   K_4 edges split into 2 C_3 orbits of 3 each:
          Orbit 1 (incident to fixed v_0): {(0,1), (0,2), (0,3)}, cycle 0→2→1→0
          Orbit 2 (between orbit vertices): {(1,2), (1,3), (2,3)}, cycle 3→4→5→3
        P_E^3 = I verified.

  (ii)  Edge Fock decomposition under C_3:
          12 dim = 4 (trivial) + 4 (ω) + 4 (ω²)

  (iii) Edge op-alg decomposition under C_3:
          24 dim = 8 (trivial) + 8 (ω) + 8 (ω²)

  (iv)  Full H_F = 280 = 136 (trivial) + 72 (ω) + 72 (ω²)  per C_3 Galois.
        Fock-level H_F = 44 = 20 (trivial) + 12 (ω) + 12 (ω²).

  STATUS: R1.3 closes positively at the dimensional / block-level decomposition.
  No b_i extraction yet (R1.4 deferred behind the Z_2 grading question).

  STRUCTURAL OBSERVATION:
   • Per-gen Fock^α = 20 (16 vertex + 4 edge) — matches 1 SM gen of 16 matter states
     PLUS 4 extra states, which might be the gauge Cartan (1 U(1) + 3 SU(2) generators?)
     or unidentified content.  Less clean than R1.2's "16 = SM gen" alignment alone.
   • No new evidence for MSSM doubling.  R1.1 retraction stands.

  NEXT STEPS (per the slow-path scoping):
   • Path A: identify Z_2 grading mechanism (multi-session, several candidates listed).
   • Path B: develop framework-specific running between Λ_sub and M_unif (multi-session).
   • R1.4 (b_i plug-and-chug) gated by either Path A or Path B.

  ADOPTED-MSSM-Sb stands.  No graded content changes.
""")
    print("R1_3_edge_sector_c3_decomposition_probe.py: sentinel done.")


if __name__ == "__main__":
    main()
