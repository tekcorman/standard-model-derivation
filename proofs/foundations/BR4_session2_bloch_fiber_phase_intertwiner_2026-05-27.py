#!/usr/bin/env python3
"""
proofs/foundations/BR4_session2_bloch_fiber_phase_intertwiner_2026-05-27.py

BR4 Session 2 — Bloch-fiber-specific phase intertwiner candidate (direction ii).

PURPOSE
-------
Session 1 ruled out the naive circulant intertwiner via AB5+AB6. Session 2
attempts direction (ii) per entry-point §11.1: a candidate built from the
Bloch-fiber-specific phase `arg(h_P) = arctan(√(5/3))` at the P-point of
the Brillouin zone — the chir-5/3 walking angle.

STRUCTURAL HOOK
---------------
Per V_Ram_Cl6 theorem program T4 closure
:

  D_i = (√3/2)·γ_7 + i·(√5/2)·Q_i  for i ∈ {1, 2, 3}

where Q_i are the three Furey pair-complement operators on Cl(6) Fock:

  Q_1 = γ_3 γ_4 γ_5 γ_6   (omits Furey pair (γ_1, γ_2))
  Q_2 = γ_1 γ_2 γ_5 γ_6   (omits Furey pair (γ_3, γ_4))
  Q_3 = γ_1 γ_2 γ_3 γ_4   (omits Furey pair (γ_5, γ_6))

These satisfy Q_i Q_j = -Q_k (cyclic, quaternion-like), [Q_i, Q_j] = 0,
Q_i² = +I. The phase arg(h_P) = arctan(√(5/3)) is literally the phase
angle of the eigenvalue h = (√3 + i√5)/2 of the Hashimoto operator B(P)
restricted to V_Ram(P). So the chir-5/3 walking angle and the BR4
intertwiner are NOT independent — they originate from the same h.

WHAT THIS SESSION CHECKS
------------------------

Key structural question: does the diagonal Spin(3) lift σ_diag of the
substrate body-diagonal C_3 (which is the SAME C_3 as the observer's
σ_C3 per V_Ram_Cl6 T2 closure + M1.B) permute the Q_i cyclically?

  σ_diag Q_i σ_diag† = ?

If σ_diag(Q_i) = Q_{i+1 mod 3} (Q_i transform as a C_3 vector):
  → W built from Q_i IS C_3-equivariant on Cl(6) Fock
  → projection to C³_obs likely gives circulant W
  → AB6 FAILS (inherits W75 obstruction)

If σ_diag(Q_i) is NOT a cyclic permutation of {Q_1, Q_2, Q_3}:
  → Q_i are C_3-anomalous on Cl(6) Fock
  → W built from Q_i likely non-circulant on C³_obs
  → AB6 PASSES → viable Session 3 candidate

This is a STRUCTURAL pre-flight, not a closure. If AB6 passes, Session 3
must build the explicit C³_obs ↔ Cl(6) Fock projection and compute the
3×3 W matrix entries.

Run with:
    python3 proofs/foundations/BR4_session2_bloch_fiber_phase_intertwiner_2026-05-27.py
"""

import numpy as np
from scipy.linalg import expm
from collections import Counter

TOL = 1e-9


# ---------------------------------------------------------------------------
# Cl(6,0) Brauer-Weyl generators (matching V_Ram_Cl6_iso_T1/T2/T4 probes)
# ---------------------------------------------------------------------------

def kron(*mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

G = [None] * 7
G[1] = kron(sx, I2, I2)
G[2] = kron(sy, I2, I2)
G[3] = kron(sz, sx, I2)
G[4] = kron(sz, sy, I2)
G[5] = kron(sz, sz, sx)
G[6] = kron(sz, sz, sy)

G7 = -1j * G[1] @ G[2] @ G[3] @ G[4] @ G[5] @ G[6]


# ---------------------------------------------------------------------------
# The 3 Furey pair-complement operators (per V_Ram_Cl6 T4)
# ---------------------------------------------------------------------------

Q1 = G[3] @ G[4] @ G[5] @ G[6]   # omits Furey pair (γ_1, γ_2)
Q2 = G[1] @ G[2] @ G[5] @ G[6]   # omits Furey pair (γ_3, γ_4)
Q3 = G[1] @ G[2] @ G[3] @ G[4]   # omits Furey pair (γ_5, γ_6)

# Verify Q_i² = I, Hermitian, commute with γ_7
for label, Q in [("Q_1", Q1), ("Q_2", Q2), ("Q_3", Q3)]:
    assert np.allclose(Q @ Q, np.eye(8), atol=TOL), f"{label}² ≠ I"
    assert np.allclose(Q, Q.conj().T, atol=TOL), f"{label} not Hermitian"
    assert np.allclose(Q @ G7 - G7 @ Q, 0, atol=TOL), f"[{label}, γ_7] ≠ 0"

# Verify Q_i Q_j = -Q_k (cyclic)
assert np.allclose(Q1 @ Q2, -Q3, atol=TOL), "Q1 Q2 ≠ -Q3"
assert np.allclose(Q2 @ Q3, -Q1, atol=TOL), "Q2 Q3 ≠ -Q1"
assert np.allclose(Q3 @ Q1, -Q2, atol=TOL), "Q3 Q1 ≠ -Q2"


# ---------------------------------------------------------------------------
# Diagonal Spin(3) lift σ_diag (per V_Ram_Cl6 T2 closure)
# Acts cyclically on BOTH (γ_1, γ_2, γ_3) and (γ_4, γ_5, γ_6) simultaneously
# ---------------------------------------------------------------------------

# σ_(1,2,3): rotation by 120° about body-diagonal in (γ_1, γ_2, γ_3) 3-plane
S12 = -1j/2 * (G[1] @ G[2])
S13 = -1j/2 * (G[1] @ G[3])
S23 = -1j/2 * (G[2] @ G[3])
J_axis_123 = (1/np.sqrt(3)) * (S23 - S13 + S12)
sigma_123 = expm(-1j * (2 * np.pi / 3) * J_axis_123)

# σ_(4,5,6): same on the second triple
S45 = -1j/2 * (G[4] @ G[5])
S46 = -1j/2 * (G[4] @ G[6])
S56 = -1j/2 * (G[5] @ G[6])
J_axis_456 = (1/np.sqrt(3)) * (S56 - S46 + S45)
sigma_456 = expm(-1j * (2 * np.pi / 3) * J_axis_456)

# Diagonal lift = product (commute)
sigma_diag = sigma_123 @ sigma_456

# Verify σ_diag is unitary and σ_diag³ = ±I
assert np.allclose(sigma_diag @ sigma_diag.conj().T, np.eye(8), atol=TOL), "σ_diag not unitary"
sigma_diag_cubed = sigma_diag @ sigma_diag @ sigma_diag
order3_plus = np.allclose(sigma_diag_cubed, np.eye(8), atol=1e-8)
order3_minus = np.allclose(sigma_diag_cubed, -np.eye(8), atol=1e-8)
assert order3_plus or order3_minus, "σ_diag³ ≠ ±I"

# If σ_diag has order 6 (spin double cover), the C_3 element is σ_diag²
if order3_minus:
    sigma_eff = sigma_diag @ sigma_diag
    note_order = "σ_diag has order 6 (spin double cover); using σ_diag² as the C_3 element"
else:
    sigma_eff = sigma_diag
    note_order = "σ_diag has order 3"

assert np.allclose(sigma_eff @ sigma_eff @ sigma_eff, np.eye(8), atol=1e-8), \
    "σ_eff³ ≠ I"


# ---------------------------------------------------------------------------
# Main test: how does σ_eff transform each Q_i?
# ---------------------------------------------------------------------------

def conjugate_by(U, M):
    return U @ M @ U.conj().T


sigmaQ = {
    "Q_1": conjugate_by(sigma_eff, Q1),
    "Q_2": conjugate_by(sigma_eff, Q2),
    "Q_3": conjugate_by(sigma_eff, Q3),
}


# Check if conjugate matches any of {Q_1, Q_2, Q_3, -Q_1, -Q_2, -Q_3}
def classify_against_Qs(M, tol=1e-8):
    candidates = {
        "+Q_1": Q1, "-Q_1": -Q1,
        "+Q_2": Q2, "-Q_2": -Q2,
        "+Q_3": Q3, "-Q_3": -Q3,
    }
    for name, ref in candidates.items():
        if np.allclose(M, ref, atol=tol):
            return name
    return None


classifications = {label: classify_against_Qs(M) for label, M in sigmaQ.items()}


# ---------------------------------------------------------------------------
# Report Q_i transformation under σ_eff
# ---------------------------------------------------------------------------

print("=" * 76)
print("BR4 Session 2 — Bloch-fiber Q_i candidate AB5/AB6 pre-flight")
print("=" * 76)
print()
print(f"  Note: {note_order}")
print()
print("  How σ_eff (substrate C_3 lifted to Cl(6) Fock) transforms each Q_i:")
print()

for label in ["Q_1", "Q_2", "Q_3"]:
    cls = classifications[label]
    if cls is not None:
        print(f"    σ_eff · {label} · σ_eff† = {cls}")
    else:
        # Compute decomposition in the Q-algebra
        M = sigmaQ[label]
        # Components: try to write M = a I + b Q1 + c Q2 + d Q3 (if it's in that span)
        # Actually since the Q's anti-commute with γ_a γ_b for some pairs, σ_eff(Q_i)
        # might be a complex combination. Let's report the matrix norms.
        coeff_I = np.trace(M @ np.eye(8)) / 8
        coeff_Q1 = np.trace(M @ Q1) / 8
        coeff_Q2 = np.trace(M @ Q2) / 8
        coeff_Q3 = np.trace(M @ Q3) / 8
        print(f"    σ_eff · {label} · σ_eff† = NOT in {{±Q_1, ±Q_2, ±Q_3}}")
        print(f"      Projection onto {{I, Q_1, Q_2, Q_3}}: "
              f"I={coeff_I:.3f}, Q_1={coeff_Q1:.3f}, Q_2={coeff_Q2:.3f}, Q_3={coeff_Q3:.3f}")
        # Frobenius norm of remainder
        remainder = M - (coeff_I * np.eye(8) + coeff_Q1 * Q1 + coeff_Q2 * Q2 + coeff_Q3 * Q3)
        rem_norm = np.linalg.norm(remainder)
        print(f"      ‖remainder outside {{I,Q_1,Q_2,Q_3}} span‖_F = {rem_norm:.3e}")

print()

# Verify the cyclic-permutation hypothesis: σ_eff(Q_i) = Q_{i+1 mod 3}?
cyclic_pattern = (
    classifications["Q_1"] in ("+Q_2", "-Q_2")
    and classifications["Q_2"] in ("+Q_3", "-Q_3")
    and classifications["Q_3"] in ("+Q_1", "-Q_1")
)
inverse_cyclic = (
    classifications["Q_1"] in ("+Q_3", "-Q_3")
    and classifications["Q_2"] in ("+Q_1", "-Q_1")
    and classifications["Q_3"] in ("+Q_2", "-Q_2")
)

print(f"  Cyclic Q_i → Q_{{i+1}}? {cyclic_pattern}")
print(f"  Inverse cyclic Q_i → Q_{{i-1}}? {inverse_cyclic}")
print()


# ---------------------------------------------------------------------------
# Build candidate W on C³_obs via D_i restricted to a 3D subspace
# ---------------------------------------------------------------------------
# To check non-circularity at the C³_obs level, we need a 3-dim subspace
# of Cl(6) Fock representing the 3 mass-eigenstate generations.
#
# Natural choice: pick one eigenvector from each σ_eff isotype (+1, +ω, +ω̄).
# σ_eff has spectrum (4 trivial + 2 ω + 2 ω̄) per V_Ram_Cl6 T2.
# We pick |gen 1⟩ from +1 isotype, |gen 2⟩ from +ω, |gen 3⟩ from +ω̄ — this
# makes σ_eff in {|gen i⟩} basis exactly the diagonal C_3 = diag(1, ω, ω̄).
# But that's IN THE C_3 ISOTYPIC BASIS. The "mass basis" (where σ_C3 is
# the cyclic permutation) is the DFT_3 transform of this.

omega = np.exp(2j * np.pi / 3)
eigvals_sigma, eigvecs_sigma = np.linalg.eig(sigma_eff)

# Group eigenvectors by σ_eff eigenvalue
isotype = {1: [], "omega": [], "omegabar": []}
for k in range(8):
    e = eigvals_sigma[k]
    v = eigvecs_sigma[:, k]
    if abs(e - 1) < 1e-7:
        isotype[1].append(v)
    elif abs(e - omega) < 1e-7:
        isotype["omega"].append(v)
    elif abs(e - omega.conjugate()) < 1e-7:
        isotype["omegabar"].append(v)

print(f"  σ_eff isotypic decomposition: "
      f"trivial dim={len(isotype[1])}, "
      f"ω dim={len(isotype['omega'])}, "
      f"ω̄ dim={len(isotype['omegabar'])}")
print()

# Pick the natural mass-basis: DFT_3 of (one trivial + one ω + one ω̄ rep)
# This gives a 3D subspace where σ_eff acts as the cyclic permutation.
v_triv = isotype[1][0]
v_omega = isotype["omega"][0]
v_omegabar = isotype["omegabar"][0]

# DFT_3 transformation: |gen k⟩ = (1/√3) Σ_j ω^{jk} |c_j⟩
# where |c_0⟩ = trivial, |c_1⟩ = ω-isotype rep, |c_2⟩ = ω̄-isotype rep
DFT3 = (1 / np.sqrt(3)) * np.array([
    [1, 1, 1],
    [1, omega, omega.conjugate()],
    [1, omega.conjugate(), omega],
], dtype=complex)

# Build {|gen i⟩} in Cl(6) Fock as 8-dim vectors
# Use the isotypic basis matrix V_iso (8×3) then apply DFT3 inverse:
V_iso = np.column_stack([v_triv, v_omega, v_omegabar])
V_mass = V_iso @ DFT3.conj().T  # mass basis vectors in Cl(6) Fock

# Verify σ_eff in mass basis is the cyclic permutation
sigma_mass = V_mass.conj().T @ sigma_eff @ V_mass
sigma_C3_expected = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
], dtype=complex)
print(f"  σ_eff in mass basis vs expected cyclic-shift σ_C3:")
print(f"    ‖σ_eff_mass - σ_C3‖_F = {np.linalg.norm(sigma_mass - sigma_C3_expected):.3e}")
print()

# ---------------------------------------------------------------------------
# Build W on C³_obs from the D_i operators
# ---------------------------------------------------------------------------
# Per T4: D_i = (√3/2)γ_7 + i(√5/2) Q_i is the canonical D for generation i.
#
# Bloch-fiber-specific candidate intertwiner:
#   W_ji = ⟨gen j | D_i | gen i⟩  (using generation-i's canonical D_i)
#
# This is non-circulant by construction IF the Q_i don't permute cyclically
# under σ_eff (per the test above).

D1 = (np.sqrt(3) / 2) * G7 + 1j * (np.sqrt(5) / 2) * Q1
D2 = (np.sqrt(3) / 2) * G7 + 1j * (np.sqrt(5) / 2) * Q2
D3 = (np.sqrt(3) / 2) * G7 + 1j * (np.sqrt(5) / 2) * Q3

# W is 3×3 with W[j,i] = <gen j | D_i | gen i>
W = np.zeros((3, 3), dtype=complex)
for i, D_i in enumerate([D1, D2, D3]):
    gen_i = V_mass[:, i]
    for j in range(3):
        gen_j = V_mass[:, j]
        W[j, i] = gen_j.conj() @ D_i @ gen_i

print("  Candidate W on C³_obs (Bloch-fiber-specific via D_i = (√3/2)γ_7 + i(√5/2)Q_i):")
print()
print("  Magnitudes:")
print("    |W_ji| = ")
for j in range(3):
    print(f"      [ {abs(W[j,0]):.4f}  {abs(W[j,1]):.4f}  {abs(W[j,2]):.4f} ]")
print()
print("  Arguments (degrees):")
print("    arg(W_ji) = ")
for j in range(3):
    args = [np.degrees(np.angle(W[j, i])) if abs(W[j, i]) > 1e-12 else 0.0
            for i in range(3)]
    print(f"      [ {args[0]:>+8.2f}°  {args[1]:>+8.2f}°  {args[2]:>+8.2f}° ]")
print()

# ---------------------------------------------------------------------------
# AB5: is W in span of identity + cyclic shift σ_C3 + σ_C3²?
# (Circulant matrices on C³ form a 3-dim subspace spanned by I, σ, σ²)
# ---------------------------------------------------------------------------

# Components of W in the circulant basis: a + b σ_C3 + c σ_C3²
sigma_C3 = sigma_C3_expected
sigma_C3_sq = sigma_C3 @ sigma_C3

# Project W onto the circulant subspace
def trace_inner_3x3(A, B):
    return np.trace(A.conj().T @ B) / 3

a = trace_inner_3x3(np.eye(3), W)
b = trace_inner_3x3(sigma_C3, W)
c = trace_inner_3x3(sigma_C3_sq, W)
W_circ = a * np.eye(3) + b * sigma_C3 + c * sigma_C3_sq
W_residual = W - W_circ
residual_norm = np.linalg.norm(W_residual)

print(f"  Decomposition of W in the circulant basis (I, σ_C3, σ_C3²):")
print(f"    a (coefficient of I)    = {a:.6f}")
print(f"    b (coefficient of σ_C3) = {b:.6f}")
print(f"    c (coefficient of σ_C3²)= {c:.6f}")
print(f"    ‖W − W_circulant‖_F     = {residual_norm:.3e}")
print()


# ---------------------------------------------------------------------------
# AB6: commutation check [W, σ_C3] and [W, P_C3]
# ---------------------------------------------------------------------------

def isotypic_proj_3(omega_power):
    P = np.zeros((3, 3), dtype=complex)
    for k in range(3):
        P += (omega ** (-k * omega_power)) * np.linalg.matrix_power(sigma_C3, k)
    return P / 3

P_1 = isotypic_proj_3(0)
P_omega = isotypic_proj_3(1)
P_omegabar = isotypic_proj_3(2)

comm_sigma = W @ sigma_C3 - sigma_C3 @ W
comm_P1 = W @ P_1 - P_1 @ W
comm_Pom = W @ P_omega - P_omega @ W
comm_Pombar = W @ P_omegabar - P_omegabar @ W


print(f"  AB6 commutation checks:")
print(f"    ‖[W, σ_C3]‖_∞    = {np.abs(comm_sigma).max():.3e}")
print(f"    ‖[W, P_1]‖_∞     = {np.abs(comm_P1).max():.3e}")
print(f"    ‖[W, P_ω]‖_∞     = {np.abs(comm_Pom).max():.3e}")
print(f"    ‖[W, P_ω̄]‖_∞     = {np.abs(comm_Pombar).max():.3e}")
print()


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

is_circulant = residual_norm < 1e-9
commutes_with_sigma = np.abs(comm_sigma).max() < 1e-9

print("=" * 76)
print("VERDICT")
print("=" * 76)
print()

if cyclic_pattern or inverse_cyclic:
    print("  σ_eff DOES permute Q_i cyclically.")
    print("  This means the Q_i are C_3-equivariant on Cl(6) Fock,")
    print("  and any intertwiner built from them inherits AB5+AB6 obstruction.")
else:
    print("  σ_eff does NOT permute Q_i cyclically.")
    print("  This breaks the C_3-equivariance of {Q_1, Q_2, Q_3} as a set on")
    print("  Cl(6) Fock — promising structural hint for AB6 PASS.")
print()

if is_circulant:
    print(f"  However, W projected to C³_obs IS circulant (residual {residual_norm:.3e}).")
    print(f"  The natural mass-basis projection averages out the Q_i anomaly.")
    print(f"  AB6: candidate FAILS (W circulant ⟹ [W, P_C3] = 0).")
else:
    print(f"  W projected to C³_obs is NON-CIRCULANT (‖residual‖ = {residual_norm:.3e}).")
    print()
    if commutes_with_sigma:
        print(f"  But W still commutes with σ_C3 — unusual; investigate.")
    else:
        print(f"  W does NOT commute with σ_C3 → AB6 PASSES.")
        print()
        print(f"  This is a viable Session 3 candidate for the BR4 intertwiner.")
print()

print(f"  Companion structural data:")
print(f"    arg(h_P) = arctan(√(5/3)) = {np.degrees(np.arctan(np.sqrt(5/3))):.4f}°")
print(f"    h = (√3 + i√5)/2, |h|² = 2 = k* − 1")
print(f"    D_i = (√3/2)γ_7 + i(√5/2)Q_i is the canonical T4 form")
