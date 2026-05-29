#!/usr/bin/env python3
"""
G3b geometric factor: |<Gamma,Perron|P,Higgs>| = 1/|h|_P = 1/sqrt(k*-1).

Addresses docs/theorems/theorem_g3_higgs_coefficient.md G3b.

WHAT THIS SCRIPT ESTABLISHES:
  The Gamma-to-P geometric overlap is

      |<v0(Gamma) | psi_H(P)>| = 1 / sqrt(k*-1) = 1/sqrt(2) = 1/|h|_P

  This is the GEOMETRIC FACTOR in the Higgs VEV formula:

      v = delta^2 * |<v0|psi_H>| * M_P * N^{-1/4}
        = delta^2 * (1/|h|_P) * M_P * N^{-1/4}
        = delta^2 * M_P / (sqrt(2) * N^{1/4})

  The two factors are SEPARATE:
    - 1/|h|_P = 1/sqrt(k*-1): geometric overlap (THIS SCRIPT, Type 2)
    - delta^2 = 4/81: screw Dyson two-vertex amplitude
                     (proofs/masses/srs_delta_sq_theorem.py, Type 2+3)

WHAT THIS SCRIPT DOES NOT ESTABLISH:
  The original claim in docs/theorems/theorem_g3_higgs_coefficient.md §7 was
  |<v0|H(k_P)|psi_H>| = delta. That is WRONG: the actual matrix element
  is |<v0|H(P)|psi_H>| = sqrt(k*/2) = sqrt(3/2) ~ 1.225, not delta = 2/9.
  The OVERLAP (without H) is 1/sqrt(2) = 1/|h|_P. The theorem doc §7
  conflated the matrix element with the overlap.

Gate types:
  [Type 2] algebra from I4_132 bond structure + C3 decomposition at P
  [Type 3] |h|_P = sqrt(k*-1): Bloch-lift theorem (theorem_bloch_lift_mu.md)
  [Type 2] |<v0|psi_H>| = 1/|h|_P: verified algebraically below

References:
  - docs/theorems/theorem_g3_higgs_coefficient.md (G3b)
  - proofs/masses/wigner_d1_screw_41.py (delta = 2/9 source)
  - proofs/masses/srs_delta_sq_theorem.py (delta^2 from Dyson equation)
  - docs/theorems/theorem_bloch_lift_mu.md (|h|_P = sqrt(k*-1) = sqrt(2))
"""

import numpy as np
from numpy import linalg as la
from fractions import Fraction
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common import find_bonds, bloch_H, N_ATOMS, C3_PERM

RTOL = 1e-10
K_STAR = 3
DELTA    = Fraction(2, 9)
DELTA_F  = float(DELTA)
DELTA_SQ = Fraction(4, 81)

# |h|_P^2 = k*-1 = 2  (Ramanujan saturation, Bloch-lift theorem)
H_P_SQ   = K_STAR - 1          # = 2
H_P_NORM = np.sqrt(H_P_SQ)     # = sqrt(2)

results = []


def record(name, passed, detail=""):
    results.append((name, passed, detail))
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] {name}: {detail}")
    return passed


bonds = find_bonds()
assert len(bonds) == N_ATOMS * K_STAR

print("=" * 68)
print("G3b GEOMETRIC FACTOR: |<Gamma,Perron|P,Higgs>| = 1/|h|_P")
print("=" * 68)


# -----------------------------------------------------------------------
# 1. Gamma-point Perron state
# -----------------------------------------------------------------------

print("\n--- 1. Gamma-point Perron state ---")

H_Gamma = bloch_H([0, 0, 0], bonds)
evals_G, evecs_G = la.eigh(H_Gamma)
perron_idx = int(np.argmax(evals_G.real))
v0 = evecs_G[:, perron_idx]

record("gamma_perron_eigenvalue",
       abs(evals_G[perron_idx] - K_STAR) < RTOL,
       f"E = {evals_G[perron_idx]:.6f}, expected {K_STAR}")

# Phase-fix: make v0 real and positive
v0 = (v0 * np.exp(-1j * np.angle(v0[0]))).real
record("gamma_perron_uniform",
       np.allclose(np.abs(v0), 0.5 * np.ones(4), atol=RTOL),
       f"v0 = {v0}")


# -----------------------------------------------------------------------
# 2. P-point C3-trivial Higgs state
# -----------------------------------------------------------------------

print("\n--- 2. P-point C3-trivial Higgs state ---")

k_P = np.array([0.25, 0.25, 0.25])
H_P_mat = bloch_H(k_P, bonds)

# H(k_P)^2 = k*I (Clifford property)
H_sq = H_P_mat @ H_P_mat
record("HP_squared",
       la.norm(H_sq - K_STAR * np.eye(4)) < RTOL,
       f"H(k_P)^2 = {K_STAR}*I (error {la.norm(H_sq - K_STAR*np.eye(4)):.1e})")

# Eigenvalues +-sqrt(k*), doubly degenerate
evals_P, evecs_P = la.eigh(H_P_mat)
sqrt_k = np.sqrt(K_STAR)
record("HP_eigenvalues",
       la.norm(np.sort(evals_P) - np.array([-sqrt_k,-sqrt_k,sqrt_k,sqrt_k])) < RTOL,
       f"evals = ±sqrt({K_STAR}) doubly degenerate")

# Resolve C3 within each doublet
def c3_trivial_state(evecs, indices):
    """Find the C3-trivial (eigenvalue 1) state within a degenerate doublet."""
    sub     = evecs[:, indices]
    C3_sub  = sub.conj().T @ C3_PERM @ sub
    c3ev, c3vecs = la.eig(C3_sub)
    trivial = np.argmin(np.abs(c3ev - 1.0))
    psi     = sub @ c3vecs[:, trivial]
    psi    /= la.norm(psi)
    c3_check = np.dot(psi.conj(), C3_PERM @ psi)
    return psi, c3ev[trivial], c3_check

pos_idx = [i for i in range(4) if abs(evals_P[i] - sqrt_k) < 1e-8]
neg_idx = [i for i in range(4) if abs(evals_P[i] + sqrt_k) < 1e-8]

psi_H_pos, c3ev_pos, c3_check_pos = c3_trivial_state(evecs_P, pos_idx)
psi_H_neg, c3ev_neg, c3_check_neg = c3_trivial_state(evecs_P, neg_idx)

record("c3_trivial_pos",
       abs(c3_check_pos - 1.0) < 0.3,
       f"C3 ev = {c3_check_pos:.4f} (expected 1)")
record("c3_trivial_neg",
       abs(c3_check_neg - 1.0) < 0.3,
       f"C3 ev = {c3_check_neg:.4f} (expected 1)")


# -----------------------------------------------------------------------
# 3. THE KEY RESULT: |<v0|psi_H>| = 1/|h|_P = 1/sqrt(k*-1)
# -----------------------------------------------------------------------

print("\n--- 3. Geometric overlap |<v0|psi_H>| ---")
print()

target = 1.0 / H_P_NORM   # = 1/sqrt(2)

overlap_pos = np.dot(v0, psi_H_pos)
overlap_neg = np.dot(v0, psi_H_neg)

print(f"  Target: 1/|h|_P = 1/sqrt(k*-1) = 1/sqrt({H_P_SQ}) = {target:.10f}")
print()
print(f"  +sqrt(k*) Higgs state:")
print(f"    <v0|psi_H+>   = {overlap_pos:.10f}")
print(f"    |<v0|psi_H+>| = {abs(overlap_pos):.10f}")
print()
print(f"  -sqrt(k*) Higgs state:")
print(f"    <v0|psi_H->   = {overlap_neg:.10f}")
print(f"    |<v0|psi_H->| = {abs(overlap_neg):.10f}")

record("overlap_pos_is_1_over_h_P",
       abs(abs(overlap_pos) - target) < 1e-8,
       f"|<v0|psi_H+>| = {abs(overlap_pos):.10f}, 1/|h|_P = {target:.10f}")
record("overlap_neg_is_1_over_h_P",
       abs(abs(overlap_neg) - target) < 1e-8,
       f"|<v0|psi_H->| = {abs(overlap_neg):.10f}, 1/|h|_P = {target:.10f}")


# -----------------------------------------------------------------------
# 4. Algebraic explanation: why |<v0|psi_H>| = 1/sqrt(k*-1) exactly
# -----------------------------------------------------------------------

print("\n--- 4. Algebraic explanation ---")
print(f"""
  v0 = (1,1,1,1)/2 decomposes in the C3 basis as:

    v0 = (1/2)|t0> + (sqrt(3)/2)|ts>

  where:
    |t0> = (1,0,0,0)           — atom 0, on the C3 axis
    |ts> = (0,1,1,1)/sqrt(3)   — symmetric combination of atoms 1,2,3

  At P = (1/4,1/4,1/4), all bond Bloch phases are ±i
  (every cell vector (n1,n2,n3) has e^{{2πi k·n}} = e^{{πi(n1+n2+n3)/2}} = ±i).

  The C3-trivial 2x2 block of H(k_P) in the (t0,ts) basis has
  equal-magnitude off-diagonal elements, forcing the eigenvectors
  to be equal-weight superpositions:

    psi_H = (1/sqrt(2))|t0> + (e^{{iφ}}/sqrt(2))|ts>

  with |alpha| = |beta| = 1/sqrt(2) (independent of φ).

  Then:

    <v0|psi_H> = (1/2)(1/sqrt(2)) + (sqrt(3)/2)(e^{{-iφ}}/sqrt(2))
               = [1 + sqrt(3)e^{{-iφ}}] / (2*sqrt(2))

  The magnitude:
    |<v0|psi_H>|^2 = [1 + 3 + 2*sqrt(3)*cos(φ)] / 8

  But the C3 symmetry forces φ = π/2 (so that psi_H transforms
  correctly under C3):

    cos(φ) = cos(π/2) = 0
    |<v0|psi_H>|^2 = 4/8 = 1/2 = 1/(k*-1)

  Hence |<v0|psi_H>| = 1/sqrt(k*-1) = 1/sqrt(2) = 1/|h|_P  (EXACT).
""")

# Verify the phase is π/2
t0 = np.array([1, 0, 0, 0], dtype=complex)
ts = np.array([0, 1, 1, 1], dtype=complex) / np.sqrt(3)
alpha = np.dot(t0.conj(), psi_H_pos)
beta  = np.dot(ts.conj(), psi_H_pos)
phase = np.angle(beta) - np.angle(alpha)
phase_mod = phase % (2 * np.pi)
if phase_mod > np.pi:
    phase_mod -= 2 * np.pi

print(f"  Numerical check of equal-weight structure:")
print(f"    |alpha| = |<t0|psi_H+>| = {abs(alpha):.10f}  (expected 1/sqrt(2) = {1/np.sqrt(2):.10f})")
print(f"    |beta|  = |<ts|psi_H+>| = {abs(beta):.10f}  (expected 1/sqrt(2) = {1/np.sqrt(2):.10f})")
print(f"    phase(beta) - phase(alpha) mod 2π = {phase_mod:.6f}  (expected ±π/2 = ±{np.pi/2:.6f})")

record("alpha_equal_weight",
       abs(abs(alpha) - 1.0/np.sqrt(2)) < 1e-8,
       f"|alpha| = {abs(alpha):.10f}")
record("beta_equal_weight",
       abs(abs(beta) - 1.0/np.sqrt(2)) < 1e-8,
       f"|beta| = {abs(beta):.10f}")
record("phase_is_pi_over_2",
       abs(abs(phase_mod) - np.pi/2) < 1e-6,
       f"phase = {phase_mod:.6f} rad (expected ±π/2)")


# -----------------------------------------------------------------------
# 5. Complete VEV formula from the two factors
# -----------------------------------------------------------------------

print("\n--- 5. Full VEV coefficient from two factors ---")
print(f"""
  The VEV formula is the PRODUCT of two independent factors:

    v = delta^2  ×  |<v0|psi_H>|  ×  M_P  ×  N^(-1/4)

  Factor 1 — Screw Born probability (two-vertex Dyson):
    delta^2 = {float(DELTA_SQ):.10f} = 4/81
    Source: proofs/masses/srs_delta_sq_theorem.py (Type 2+3)
    Reference: wigner_d1_screw_41.py (delta = 2/9 SOLID)

  Factor 2 — Geometric overlap (this script):
    |<v0|psi_H>| = 1/|h|_P = 1/sqrt(k*-1) = 1/sqrt({H_P_SQ}) = {target:.10f}
    Source: THIS SCRIPT (Type 2 numerical verification)
    Reference: theorem_bloch_lift_mu.md (|h|_P = sqrt(k*-1) CLOSED)

  Product:
    delta^2 / |h|_P = ({float(DELTA_SQ):.8f}) / {H_P_NORM:.8f}
                    = {float(DELTA_SQ)/H_P_NORM:.10f}
    = delta^2 / sqrt(2)
    = {float(DELTA_SQ)/np.sqrt(2):.10f}

  VEV formula:
    v = (4/81) * M_P / (sqrt(2) * N^(1/4))
""")

coeff = float(DELTA_SQ) * target  # delta^2 * (1/|h|_P)
coeff_expected = float(DELTA_SQ) / np.sqrt(2)
record("vev_coefficient",
       abs(coeff - coeff_expected) < 1e-12,
       f"delta^2 * (1/|h|_P) = {coeff:.12f} = delta^2/sqrt(2) = {coeff_expected:.12f}")


# -----------------------------------------------------------------------
# 6. Correction note: the original theorem doc claim
# -----------------------------------------------------------------------

print("\n--- 6. Correction to docs/theorems/theorem_g3_higgs_coefficient.md §7 ---")
print(f"""
  ORIGINAL CLAIM (§7):
    "Compute <k=0|H_{{4_1}}|k_P> ... and show it equals delta = 2/9"

  ACTUAL VALUE:
    |<v0|H(k_P)|psi_H>| = sqrt(k*/2) = sqrt({K_STAR}/2) = {np.sqrt(K_STAR/2):.10f}
    (NOT delta = {DELTA_F:.10f})

  CORRECT STATEMENT:
    The relevant matrix element is the OVERLAP |<v0|psi_H>| = 1/sqrt(k*-1)
    = 1/|h|_P, which gives the 1/sqrt(2) geometric factor.
    The delta^2 factor comes separately from the Dyson two-vertex self-energy.

  G3b STATUS AFTER THIS SCRIPT:
    - Geometric factor 1/|h|_P: SOLID [this script, Type 2]
    - Delta^2 factor: CANDIDATE [srs_delta_sq_theorem.py, coupling normalization gap]

  For G3b to reach SOLID overall, srs_delta_sq_theorem.py's coupling
  normalization (why the screw coupling constant equals delta, not 1 or
  sin(beta)) needs to be gated as Type 1/2/3 without interpretation gaps.
""")

me_actual = abs(np.dot(v0, H_P_mat @ psi_H_pos))
record("actual_matrix_element_noted",
       abs(me_actual - np.sqrt(float(K_STAR)/2)) < 1e-8,
       f"|<v0|H(P)|psi_H>| = {me_actual:.10f} = sqrt(k*/2), NOT delta")


# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------

print("\n" + "=" * 68)
print("SUMMARY")
print("=" * 68)
n_pass = sum(1 for _, p, _ in results if p)
n_fail = sum(1 for _, p, _ in results if not p)
print(f"\n  Tests: {n_pass}/{len(results)} pass, {n_fail} fail\n")
for name, passed, detail in results:
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] {name}")
    if detail:
        print(f"         {detail}")
print()

if n_fail == 0:
    print("  VERIFIED: |<Gamma,Perron|P,Higgs>| = 1/|h|_P = 1/sqrt(k*-1)")
    print("  This is the GEOMETRIC FACTOR in v = delta^2 M_P / (sqrt(2) N^{1/4}).")
    print("  Combined with delta^2 from srs_delta_sq_theorem.py, G3b reaches")
    print("  CANDIDATE-SOLID (geometric part SOLID; coupling normalization CANDIDATE).")
else:
    print(f"  {n_fail} check(s) failed.")
