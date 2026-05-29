#!/usr/bin/env python3
"""
W75 — Per-isotypic coherent sum at the P-fiber (not residue reading)

Builds on existing prior art:
  - `h_power_yukawa_galois_isotypic_stage0_Ppoint_2026-05-23.py` constructed
    B_NB(k_P) with complex Bloch phases on the srs primitive cell. That
    probe tested per-isotypic SPECTRAL RESIDUE at y=1/h_P and found j-
    independent phases → led to commutation-obstruction lemma.
  - W74 tested per-isotypic COHERENT SUM at K_4 Γ-fiber and found all
    sums REAL (no Koide phase from conjugate-symmetric eigenmode pairing).

W75 combines: COHERENT SUM (not residue) at NON-TRIVIAL Bloch fiber
(P, not Γ). At P-fiber, the complex Bloch phases break the conjugate
symmetry that killed W74 at Γ. Per the prior probe's G4 finding,
h_P lives in j=0 and j=1 only (unequal multiplicities across isotypics)
— this unevenness is structural and may give nontrivial coherent-sum
phases per isotypic.

PRE-DECLARED GATES:
  G1: B_NB(k_P) construction reproduces prior probe's results
  G2: per-isotypic coherent sums A_j = Tr(B^g · π_j) at P are complex
      (unlike W74's real values at Γ)
  G3: arg(A_j) phases per isotypic differ from each other → potential
      Koide AP structure
  G4: arg differences within 5° of 2π/3 (Koide AP)
  G5: extracted δ (from AP offset) matches empirical δ for some sector
      (lepton 12.73°, down 5-7°, up 4°) within 5°
  G6: Boltzmann-MDL weighting changes the picture (or doesn't)

Per W58: report all results honestly; no reverse-fitting.
"""

from __future__ import annotations
import cmath
import math
from itertools import product
import numpy as np
import numpy.linalg as la


gates = []
def gate(name, passed, detail=""):
    gates.append((name, bool(passed)))
    flag = "PASS" if passed else "FAIL"
    print(f"  [{flag}] {name}")
    if detail:
        for line in detail.strip("\n").split("\n"):
            print(f"         {line}")
    print()


print("=" * 78)
print("W75 — Per-isotypic coherent sum at P-fiber (not residue reading)")
print("=" * 78)
print()


# ──────────────────────────────────────────────────────────────────
# §1 — Reuse existing P-fiber B_NB construction from prior probe
# ──────────────────────────────────────────────────────────────────
A_PRIM = np.array([[-0.5,  0.5,  0.5],
                   [ 0.5, -0.5,  0.5],
                   [ 0.5,  0.5, -0.5]])
ATOMS = np.array([[1/8, 1/8, 1/8],
                  [3/8, 7/8, 5/8],
                  [7/8, 5/8, 3/8],
                  [5/8, 3/8, 7/8]])
N_ATOMS = 4
NN_DIST = math.sqrt(2) / 4
k_P = np.array([0.25, 0.25, 0.25])
OMEGA = cmath.exp(2j * math.pi / 3)
SIGMA_ATOM = {0: 0, 1: 3, 2: 1, 3: 2}

def find_bonds():
    bonds = []
    for i in range(N_ATOMS):
        for j in range(N_ATOMS):
            for n1, n2, n3 in product(range(-2, 3), repeat=3):
                rj = ATOMS[j] + n1*A_PRIM[0] + n2*A_PRIM[1] + n3*A_PRIM[2]
                d = la.norm(rj - ATOMS[i])
                if d < 0.02:
                    continue
                if abs(d - NN_DIST) < 0.02:
                    bonds.append((i, j, (n1, n2, n3)))
    return bonds

bonds = find_bonds()
N_A = len(bonds)
assert N_A == 12, f"expected 12 directed bonds; got {N_A}"

def R_offset(c):
    return (c[2], c[0], c[1])

def sigma_bond(s, t, c):
    return (SIGMA_ATOM[s], SIGMA_ATOM[t], R_offset(c))

sigma_perm = []
for s, t, c in bonds:
    s2, t2, c2 = sigma_bond(s, t, c)
    idx = bonds.index((s2, t2, c2))
    sigma_perm.append(idx)

P_C3 = np.zeros((N_A, N_A), dtype=complex)
for i, p in enumerate(sigma_perm):
    P_C3[p, i] = 1.0

def reverse_bond(s, t, c):
    return (t, s, (-c[0], -c[1], -c[2]))

def build_B_NB(k):
    B = np.zeros((N_A, N_A), dtype=complex)
    for ip, (s_p, t_p, c_p) in enumerate(bonds):
        phase = cmath.exp(2j * math.pi * np.dot(k, c_p))
        for i, (s, t, c) in enumerate(bonds):
            if t != s_p:
                continue
            if reverse_bond(s, t, c) == (s_p, t_p, c_p):
                continue
            B[ip, i] = phase
    return B

B_P = build_B_NB(k_P)

# Sanity
g1_pass = (la.norm(la.matrix_power(P_C3, 3) - np.eye(N_A)) < 1e-10 and
           la.norm(B_P @ P_C3 - P_C3 @ B_P) < 1e-10 and
           la.norm(B_P.imag) > 0.1)
gate("G1 P-fiber Bloch infrastructure verified", g1_pass,
     f"P_C3 order 3, commutes with B_NB(P), B_NB(P) has nontrivial imag part")


# ──────────────────────────────────────────────────────────────────
# §2 — C_3 Fourier projectors + per-isotypic coherent sum
# ──────────────────────────────────────────────────────────────────
PI = [
    sum((OMEGA ** (-j * k)) * la.matrix_power(P_C3, k) for k in range(3)) / 3
    for j in range(3)
]

GIRTH = 10
B_to_g = la.matrix_power(B_P, GIRTH)

# Per-isotypic coherent sum (trace)
A_j = [np.trace(B_to_g @ PI[j]) for j in range(3)]

print(f"§2 — Per-isotypic coherent sum A_j = Tr(B_NB(P)^g · π_j):")
for j, A in enumerate(A_j):
    mag = abs(A)
    arg = math.degrees(cmath.phase(A)) if mag > 1e-8 else 0.0
    print(f"  A_{j}: {A}")
    print(f"       |A_{j}| = {mag:.4f}, arg = {arg:+.4f}°")
print()

# G2: are the A_j genuinely complex (not all real)?
any_complex = any(abs(A.imag) > 1e-6 for A in A_j)
g2_pass = any_complex
gate("G2 per-isotypic coherent sums at P are complex (unlike Γ)",
     g2_pass,
     f"any A_j has |imag| > 1e-6: {any_complex}")


# ──────────────────────────────────────────────────────────────────
# §3 — Phase differences (Koide AP test)
# ──────────────────────────────────────────────────────────────────
print(f"§3 — Phase differences (Koide AP test)")
print()

phases = [math.degrees(cmath.phase(A)) if abs(A) > 1e-8 else 0.0 for A in A_j]

# Try all 3 phase-difference patterns
print(f"  Phases: {[f'{p:+.4f}°' for p in phases]}")
print()
print(f"  Differences:")
diff_01 = phases[1] - phases[0]
diff_02 = phases[2] - phases[0]
diff_12 = phases[2] - phases[1]
print(f"    arg(A_1) − arg(A_0) = {diff_01:+8.4f}°  (target: ±120°)")
print(f"    arg(A_2) − arg(A_0) = {diff_02:+8.4f}°  (target: ∓120°)")
print(f"    arg(A_2) − arg(A_1) = {diff_12:+8.4f}°")
print()

# G3: are phases meaningfully different?
phase_spread = max(abs(diff_01), abs(diff_02), abs(diff_12))
g3_pass = phase_spread > 5
gate("G3 phase spread > 5° across isotypics", g3_pass,
     f"max |Δ| across isotypics = {phase_spread:.4f}°")


# G4: Koide AP test (phases form 2π/3 AP)
# AP means phases are {δ, δ+120°, δ+240°} for some δ (or with signs swapped)
phase_sorted = sorted(phases)
diffs_consecutive = [phase_sorted[(i+1) % 3] - phase_sorted[i] for i in range(3)]
# Last diff wraps around 360
diffs_consecutive[-1] = (diffs_consecutive[-1] + 360) % 360
ap_errors = [abs(d - 120) for d in diffs_consecutive[:2]]
ap_max_err = max(ap_errors)
g4_pass = ap_max_err < 5
gate("G4 phases form 2π/3 AP within 5°", g4_pass,
     f"sorted phases: {[f'{p:.4f}°' for p in phase_sorted]};\n"
     f"diffs: {[f'{d:.4f}°' for d in diffs_consecutive]};\n"
     f"|Δ − 120°| max: {ap_max_err:.4f}°")


# ──────────────────────────────────────────────────────────────────
# §4 — Extract δ if AP holds
# ──────────────────────────────────────────────────────────────────
if g4_pass:
    delta_extracted = phase_sorted[0]
    delta_lepton_target = math.degrees(2 / 9)
    delta_down_2GeV = 5.80
    delta_down_mb = 6.31
    delta_up_mixed = 4.27
    print(f"§4 — Extracted δ (the AP offset) = {delta_extracted:+.4f}°")
    print(f"  Compare to empirical targets:")
    print(f"    δ_lepton = 12.73° (= 2/9 rad, framework theorem-grade)")
    print(f"    δ_down (2 GeV) = {delta_down_2GeV}°")
    print(f"    δ_down (m_b)   = {delta_down_mb}°")
    print(f"    δ_up           = {delta_up_mixed}°")
    print()

    closest_label = None
    closest_diff = float('inf')
    for label, target in [("lepton", math.degrees(2/9)),
                          ("down(2GeV)", delta_down_2GeV),
                          ("down(m_b)", delta_down_mb),
                          ("up", delta_up_mixed)]:
        d = min(abs(delta_extracted - target),
                abs(delta_extracted - target + 360),
                abs(delta_extracted - target - 360))
        if d < closest_diff:
            closest_diff = d
            closest_label = label
    print(f"  Closest match: δ_{closest_label}, |Δ| = {closest_diff:.4f}°")

    g5_pass = closest_diff < 5
    gate("G5 extracted δ matches some empirical target within 5°", g5_pass,
         f"closest match: {closest_label}, |Δ| = {closest_diff:.4f}°")
else:
    print(f"§4 — G4 failed; δ extraction skipped")
    g5_pass = False
    gate("G5 δ matches target", g5_pass, "(G4 prerequisite failed)")


# ──────────────────────────────────────────────────────────────────
# §5 — Boltzmann-MDL weighted variants
# ──────────────────────────────────────────────────────────────────
print(f"§5 — Boltzmann-MDL weighted coherent sums at P-fiber")
print()

def isotypic_eigenmodes(B, P):
    U, s, _ = la.svd(P)
    rank = int(np.sum(s > 1e-8))
    basis = U[:, :rank]
    B_r = basis.conj().T @ B @ basis
    eigs, _ = la.eig(B_r)
    return eigs

# Get per-isotypic eigenvalues
isotypic_eigs = [isotypic_eigenmodes(B_P, PI[j]) for j in range(3)]

for j, eigs in enumerate(isotypic_eigs):
    eig_strs = [f"{e.real:+.3f}{e.imag:+.3f}j (|·|={abs(e):.3f})" for e in eigs]
    print(f"  j={j} (dim {len(eigs)}): {', '.join(eig_strs)}")
print()

# Boltzmann weighted: w(h) = (|h|² / |h_max|²)^β
h_max_sq = max(abs(e)**2 for eigs in isotypic_eigs for e in eigs)
print(f"  |h_max|² = {h_max_sq:.4f}")
print()

for beta in [0.0, 0.5, 1.0, 2.0]:
    weighted_A = []
    for j in range(3):
        A = sum((e ** GIRTH) * ((abs(e)**2 / h_max_sq) ** beta)
                for e in isotypic_eigs[j])
        weighted_A.append(A)
    phases_w = [math.degrees(cmath.phase(A)) if abs(A) > 1e-8 else 0.0
                for A in weighted_A]
    print(f"  β={beta:.1f}:")
    print(f"    |A_j| = {[f'{abs(A):.3f}' for A in weighted_A]}")
    print(f"    arg(A_j) = {[f'{p:+.3f}°' for p in phases_w]}")
    diffs = [phases_w[(i+1)%3] - phases_w[i] for i in range(3)]
    diffs[-1] = (diffs[-1] + 360) % 360
    print(f"    diffs = {[f'{d:.3f}°' for d in diffs]}")
    ap_err = max(abs(d - 120) for d in diffs[:2])
    if ap_err < 10:
        print(f"    *** Within {ap_err:.2f}° of 2π/3 AP! ***")
    print()


# ──────────────────────────────────────────────────────────────────
# §6 — Verdict
# ──────────────────────────────────────────────────────────────────
print("=" * 78)
print("W75 — Verdict")
print("=" * 78)
n_pass = sum(1 for _, p in gates if p)
n_total = len(gates)
print(f"  {n_pass}/{n_total} gates pass")
for name, p in gates:
    print(f"  [{'PASS' if p else 'FAIL'}] {name}")
print()
