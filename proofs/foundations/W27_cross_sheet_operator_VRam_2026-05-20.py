#!/usr/bin/env python3
"""
W27 — Cross-sheet operator on V_Ram (item #1 sub-probe (i+))
=============================================================

Date: 2026-05-20
Predecessor: W26 closed-negative — W21 H_VEV = (1/√2)·χ̃ is χ̃-graded and
therefore SECTOR-DIAGONAL on V_Ram (cannot mix C_3 generations or χ̃
chiralities). The structural lesson: R-14 closure on V_Ram requires a
CROSS-SHEET operator, not a χ̃-graded scalar.

W27 tests the natural cross-sheet operator on srs-z's walker:
  P_swap : the sheet permutation (u, 0) ↔ (u, 1) on BD(K_4) vertices,
           lifted to the 24-arc walker as a permutation matrix.

Properties (all verified):
  - {P_swap, χ̃} = 0    (anticommutes — couples sheets across χ̃ sectors)
  - [P_swap, C_3] = 0   (commutes — C_3 is sheet-preserving)
  - [P_swap, B] = 0 at k=Γ (commutes — sheet swap is a graph automorphism
                            at k=Γ; per chi_tilde 2026-05-01 P_swap "only
                            commutes at k=Γ" — chi_tilde tested this on
                            Bloch B(k≠Γ), but k=Γ is sufficient for the
                            V_Ram structural test here)
  - P_swap² = I        (involution)
  - P_swap^T = P_swap  (Hermitian, real-symmetric)

The structural question for R-14:
  Does P_swap|_V_Ram have C_3-sector-DEPENDENT eigenvalues (= per-species
  labeling, would unblock R-14) or C_3-sector-UNIFORM eigenvalues (= no
  labeling, R-14 still blocked even with this cross-sheet input)?

PRE-DECLARED GATE CHECKS:
  M1. P_swap is well-defined on the walker (real symmetric, involution,
      P_swap² = I).
  M2. {P_swap, χ̃} = 0 (couples χ̃ = +1 ↔ χ̃ = -1).
  M3. [P_swap, C_3] = 0 (preserves C_3 sectors).
  M4. [P_swap, B] = 0 at k=Γ (preserves V_Ram).
  M5. P_swap|_V_Ram has eigenvalues +1 and -1 (each with multiplicity 6).
  M6. Decomposition by C_3 sectors: P_swap restricted to (C_3 = ω^k)
      subspace of V_Ram is independent of k OR sector-DEPENDENT —
      this is the decisive R-14 test.
  M7. χ̃-pair structure: within each C_3 sector, P_swap maps χ̃ = +1
      sub-sector ↔ χ̃ = -1 sub-sector (the cross-sheet action).

If M6 returns "sector-DEPENDENT": R-14 labeling foundation found.
If M6 returns "sector-UNIFORM": R-14 still blocked even with this input;
  the V_Ram structure is C_3-symmetric at the operator-spectrum level.

USAGE:
    python3 proofs/foundations/W27_cross_sheet_operator_VRam_2026-05-20.py
"""

from __future__ import annotations
import numpy as np

EXPECTED = {
    "M1_Pswap_involution":            True,
    "M2_Pswap_anticommutes_chi":      True,
    "M3_Pswap_commutes_C3":           True,
    "M4_Pswap_commutes_B_gamma":      True,
    "M5_Pswap_VRam_pm1_eigenvalues":  True,
    "M6_per_C3_sector_structure":     True,   # documented; verdict in §M6 below
    "M7_chi_pair_couples":            True,
}
RESULTS = {}

print("=" * 78)
print("W27 — cross-sheet operator on V_Ram (item #1 sub-probe (i+))")
print("=" * 78)


# ============================================================================
# Step A — Rebuild BD(K_4), B, V_Ram, χ̃, C_3 (same as W26)
# ============================================================================
N_V_K4 = 4
K4_edges = [(u, v) for u in range(N_V_K4) for v in range(u + 1, N_V_K4)]
N_V_BD = 8
def encode(u, sheet): return u + sheet * N_V_K4

bd_edges = []
for u, v in K4_edges:
    bd_edges.append((encode(u, 0), encode(v, 1)))
    bd_edges.append((encode(v, 0), encode(u, 1)))

def directed_arcs(edges):
    arcs = []
    for ei, (u, v) in enumerate(edges):
        arcs.append((u, v, ei))
        arcs.append((v, u, ei))
    return arcs

BD_arcs = directed_arcs(bd_edges)
N_ARCS = len(BD_arcs)
arc_lookup = {a: i for i, a in enumerate(BD_arcs)}

def hashimoto(arcs):
    n = len(arcs)
    B = np.zeros((n, n), dtype=complex)
    for i_p, (t_p, h_p, e_p) in enumerate(arcs):
        for i, (t, h, e) in enumerate(arcs):
            if h == t_p and e_p != e:
                B[i_p, i] = 1.0
    return B

B = hashimoto(BD_arcs)
eigvals_B, V_B = np.linalg.eig(B)
ram_mask = np.abs(np.abs(eigvals_B)**2 - (3 - 1)) < 1e-7   # |λ|² = k*-1 = 2
V_Ram_basis = V_B[:, ram_mask]
Q, _ = np.linalg.qr(V_Ram_basis)
V_Ram = Q

# χ̃
side_label = {idx: (+1 if idx < N_V_K4 else -1) for idx in range(N_V_BD)}
chi_diag = np.array([side_label[t] for (t, _, _) in BD_arcs], dtype=complex)
chi = np.diag(chi_diag)

# C_3 (sheet-preserving cycle 0→1→2, vertex 3 fixed)
def c3_vertex(v):
    base = v % N_V_K4
    sheet = v // N_V_K4
    return {0:1, 1:2, 2:0, 3:3}[base] + sheet * N_V_K4

bd_edge_lookup = {frozenset(e): i for i, e in enumerate(bd_edges)}
def c3_edge(ei):
    (u, v) = bd_edges[ei]
    return bd_edge_lookup[frozenset((c3_vertex(u), c3_vertex(v)))]

C3 = np.zeros((N_ARCS, N_ARCS), dtype=complex)
for i, (t, h, e) in enumerate(BD_arcs):
    new = (c3_vertex(t), c3_vertex(h), c3_edge(e))
    C3[arc_lookup[new], i] = 1.0

print(f"\nStep A — walker setup")
print(f"  N_arcs = {N_ARCS}, V_Ram dim = {V_Ram.shape[1]}")


# ============================================================================
# Step B — Build P_swap (sheet swap on walker)
# ============================================================================
def sheet_swap_vertex(v):
    return (v + N_V_K4) % (2 * N_V_K4)

def sheet_swap_edge(ei):
    (u, v) = bd_edges[ei]
    return bd_edge_lookup[frozenset((sheet_swap_vertex(u), sheet_swap_vertex(v)))]

P_swap = np.zeros((N_ARCS, N_ARCS), dtype=complex)
for i, (t, h, e) in enumerate(BD_arcs):
    new = (sheet_swap_vertex(t), sheet_swap_vertex(h), sheet_swap_edge(e))
    P_swap[arc_lookup[new], i] = 1.0

# M1: involution + Hermitian
P_sq = P_swap @ P_swap
hermitian_residual = np.linalg.norm(P_swap - P_swap.conj().T)
involution_residual = np.linalg.norm(P_sq - np.eye(N_ARCS))
print(f"\nStep B — P_swap construction")
print(f"  ||P_swap - P_swap^†||_F = {hermitian_residual:.2e}  (Hermitian if 0)")
print(f"  ||P_swap² - I||_F       = {involution_residual:.2e}  (involution if 0)")
M1 = hermitian_residual < 1e-12 and involution_residual < 1e-12
print(f"  M1 PASS: {M1}")
RESULTS["M1_Pswap_involution"] = bool(M1)


# ============================================================================
# Step C — Commutation/anticommutation properties
# ============================================================================
anticomm_chi = P_swap @ chi + chi @ P_swap
comm_C3 = P_swap @ C3 - C3 @ P_swap
comm_B = P_swap @ B - B @ P_swap
print(f"\nStep C — commutation properties")
print(f"  ||{{P_swap, χ̃}}||_F = {np.linalg.norm(anticomm_chi):.2e}  (anticommutes if 0)")
print(f"  ||[P_swap, C_3]||_F = {np.linalg.norm(comm_C3):.2e}  (commutes if 0)")
print(f"  ||[P_swap, B]||_F   = {np.linalg.norm(comm_B):.2e}  (commutes at k=Γ if 0)")
M2 = np.linalg.norm(anticomm_chi) < 1e-12
M3 = np.linalg.norm(comm_C3) < 1e-12
M4 = np.linalg.norm(comm_B) < 1e-12
print(f"  M2 PASS: {M2}")
print(f"  M3 PASS: {M3}")
print(f"  M4 PASS: {M4}")
RESULTS["M2_Pswap_anticommutes_chi"] = bool(M2)
RESULTS["M3_Pswap_commutes_C3"] = bool(M3)
RESULTS["M4_Pswap_commutes_B_gamma"] = bool(M4)


# ============================================================================
# Step D — P_swap restricted to V_Ram
# ============================================================================
P_swap_VRam = V_Ram.conj().T @ P_swap @ V_Ram   # 12x12

# Diagonalize: since P_swap² = I, eigenvalues are ±1.
eigs_PVRam = np.linalg.eigvalsh((P_swap_VRam + P_swap_VRam.conj().T) / 2)
n_plus = int((eigs_PVRam > 0.5).sum())
n_minus = int((eigs_PVRam < -0.5).sum())
print(f"\nStep D — P_swap restricted to V_Ram")
print(f"  P_swap|_V_Ram eigenvalues: {sorted(round(float(e),4) for e in eigs_PVRam)}")
print(f"  +1 count: {n_plus}, -1 count: {n_minus}")
M5 = (n_plus == 6) and (n_minus == 6)
print(f"  M5 PASS (±1 each multiplicity 6): {M5}")
RESULTS["M5_Pswap_VRam_pm1_eigenvalues"] = bool(M5)


# ============================================================================
# Step E — Decompose V_Ram by C_3 × χ̃ and project P_swap onto each block
# ============================================================================
omega = np.exp(2j * np.pi / 3)
P_C3 = [sum(omega ** (-k * m) * np.linalg.matrix_power(C3, m) for m in range(3)) / 3
        for k in range(3)]

# Sector bases (per C_3 × χ̃)
def sector_basis(k, s):
    P_chi_s = (np.eye(N_ARCS, dtype=complex) + s * chi) / 2
    P_joint = P_C3[k] @ P_chi_s
    P_joint_VRam_arc = P_joint @ V_Ram
    U, S, _ = np.linalg.svd(P_joint_VRam_arc)
    n_nonzero = int((S > 1e-7).sum())
    return U[:, :n_nonzero]

sectors = [(k, s) for k in range(3) for s in [+1, -1]]
sec_bases = {sec: sector_basis(*sec) for sec in sectors}

# Inter-sector blocks of P_swap
print(f"\nStep E — Inter-sector structure of P_swap on V_Ram")
print(f"  rows = target (k,s); cols = source (k,s)")
print(f"  {'':>14s} " + " ".join(f"({k},{s:+d})" for (k, s) in sectors))
block_norms = {}
for sec_i in sectors:
    Ui = sec_bases[sec_i]
    row_str = f"  ({sec_i[0]},{sec_i[1]:+d}) "
    for sec_j in sectors:
        Uj = sec_bases[sec_j]
        if Ui.shape[1] == 0 or Uj.shape[1] == 0:
            block = np.zeros((Ui.shape[1], Uj.shape[1]))
        else:
            block = Ui.conj().T @ P_swap @ Uj
        block_norms[(sec_i, sec_j)] = np.linalg.norm(block)
        row_str += f"  {np.linalg.norm(block):6.3f}"
    print(row_str)

# Identify structure
print()
print(f"  Block structure analysis:")
intra_chi_per_k = {k: block_norms[((k, +1), (k, -1))] for k in range(3)}  # (k,+) ↔ (k,-)
print(f"    χ̃-pair coupling within each k:")
for k in range(3):
    print(f"      C_3 = ω^{k}: ||P_swap[(k,+) → (k,-)]|| = {intra_chi_per_k[k]:.4f}")
inter_C3 = max(block_norms[((k1, s1), (k2, s2))] for (k1, s1) in sectors
               for (k2, s2) in sectors if k1 != k2)
print(f"    Max inter-C_3 coupling: {inter_C3:.4f}  (expect 0 — P_swap commutes with C_3)")
intra_chi_uniform = abs(intra_chi_per_k[0] - intra_chi_per_k[1]) < 1e-6 and \
                    abs(intra_chi_per_k[1] - intra_chi_per_k[2]) < 1e-6
print(f"    χ̃-pair coupling is C_3-UNIFORM across k: {intra_chi_uniform}")

M7 = all(intra_chi_per_k[k] > 0.1 for k in range(3))
RESULTS["M7_chi_pair_couples"] = bool(M7)


# ============================================================================
# Step F — The decisive R-14 test: spectrum of P_swap per C_3 sector
# ============================================================================
# Within each C_3 sector (which is 4-dim total: 2 + 2 from χ̃ = ±1), P_swap
# is a 4x4 matrix that couples the χ̃ = +1 sub-sector to the χ̃ = -1 sub-sector.
# Its eigenvalues are ±something.
#
# DECISIVE question: are these ±eigenvalues the SAME across C_3 sectors
# (= R-14 still blocked, sector-uniform) or DIFFERENT (= R-14 unblocked,
# per-species labeling found)?
print(f"\nStep F — DECISIVE R-14 TEST: spectrum of P_swap per C_3 sector")
print(f"  For each C_3 sector k, build P_swap restricted to (C_3 = ω^k subspace of V_Ram),")
print(f"  and compute eigenvalues.")
print()
sector_spectra = {}
for k in range(3):
    # Combine basis vectors from (k, +1) and (k, -1) sectors
    U_plus = sec_bases[(k, +1)]
    U_minus = sec_bases[(k, -1)]
    if U_plus.shape[1] == 0 and U_minus.shape[1] == 0:
        continue
    U_k = np.concatenate([U_plus, U_minus], axis=1) if U_minus.shape[1] > 0 else U_plus
    P_swap_k = U_k.conj().T @ P_swap @ U_k   # full P_swap action within k sector
    # Hermitize for clean eigenvalues
    P_swap_k_H = (P_swap_k + P_swap_k.conj().T) / 2
    eigs = np.linalg.eigvalsh(P_swap_k_H)
    sector_spectra[k] = sorted(round(float(e), 6) for e in eigs)
    print(f"  C_3 = ω^{k}: P_swap eigenvalues in 4-dim sector = {sector_spectra[k]}")

# Compare spectra across k
spectra_match = all(sector_spectra[k] == sector_spectra[0] for k in sector_spectra)
print()
print(f"  Are sector spectra IDENTICAL across all 3 C_3 generations? {spectra_match}")

if spectra_match:
    M6_verdict = "C_3-UNIFORM"
    print(f"  ⟹ DECISIVE R-14 TEST: NEGATIVE.")
    print(f"     P_swap on V_Ram has the SAME ±1 eigenvalue pair in every C_3 sector.")
    print(f"     The cross-sheet operator + W21 orientation IS still C_3-symmetric")
    print(f"     at the spectrum level. R-14 (per-species labeling) NOT unblocked")
    print(f"     by this sub-probe either.")
else:
    M6_verdict = "C_3-DEPENDENT"
    print(f"  ⟹ DECISIVE R-14 TEST: POSITIVE.")
    print(f"     P_swap on V_Ram has DIFFERENT eigenvalues in different C_3 sectors.")
    print(f"     The cross-sheet operator + W21 orientation introduces per-species")
    print(f"     labeling at the spectrum level. R-14 (per-species labeling)")
    print(f"     UNBLOCKED at the structural-existence level.")

M6 = True   # documents the structure
RESULTS["M6_per_C3_sector_structure"] = bool(M6)


# ============================================================================
# Step G — Verdict
# ============================================================================
print("\n" + "=" * 78)
print("W27 VERDICT — Gate Check")
print("=" * 78)
all_pass = True
for k, expected in EXPECTED.items():
    actual = RESULTS.get(k)
    status = "PASS" if actual == expected else "FAIL"
    if actual != expected: all_pass = False
    print(f"  {status}  {k:42s}  expected={expected}, got={actual}")

print()
print(f"DECISIVE R-14 TEST VERDICT: {M6_verdict}")
print()
if M6_verdict == "C_3-UNIFORM":
    print("  The cross-sheet operator P_swap successfully:")
    print("    - couples χ̃ sectors (the W26 limitation overcome)")
    print("    - has nontrivial V_Ram restriction (12-dim, ±1 eigenvalues)")
    print("    - preserves C_3 sectors (commutes with C_3)")
    print("  BUT its V_Ram spectrum is IDENTICAL across all 3 C_3 generations.")
    print("  This means even with W21 orientation AND cross-sheet operator, V_Ram is")
    print("  C_3-symmetric at the operator-spectrum level.")
    print()
    print("  STRUCTURAL CONCLUSION (sharpening of chi_tilde 2026-05-01 + W26):")
    print("  V_Ram on srs-z's bipartite cover at k=Γ is generation-symmetric under")
    print("  the natural sheet × chirality × C_3 structure. Per-species labeling")
    print("  needs additional input BEYOND the W21 orientation + cross-sheet operator.")
    print()
    print("  CANDIDATE NEXT-STEP PROBES (refined R-14 attack surface):")
    print("    (i++) C_3-asymmetric Higgs configuration: instead of uniform ⟨h⁰⟩")
    print("          on every edge, weight by C_3 character at each K_4 vertex —")
    print("          would break C_3 symmetry of V_Ram at the operator level.")
    print("    (ii) MDL waterline at gen-3 up-type: per W23 §5 (ii), the most")
    print("         direct attack on n_free = 0 derivation, independent of V_Ram.")
    print("    (iii) Bloch k ≠ Γ: P_swap fails to commute with B at k ≠ Γ (per")
    print("          chi_tilde memory). Compute V_Ram at k = k_R using RCSR data")
    print("          (if available) and re-test whether the k-dependence breaks")
    print("          the C_3 symmetry.")
elif M6_verdict == "C_3-DEPENDENT":
    print("  R-14 unblocked at the structural-existence level! Bounded next-step:")
    print("    - Map the per-C_3-sector eigenvalues to per-species (n, j) labels.")
    print("    - Test whether the eigenvalue magnitudes track the empirical")
    print("      Yukawa hierarchy.")

print()
print("=" * 78)
