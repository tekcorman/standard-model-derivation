#!/usr/bin/env python3
"""
T5 — τ Yukawa matrix element via the ISO: explicit attempt.

CONTEXT (post T4 closure):
  T4 identified D_Cl6 = (√3/2)γ_7 + i(√5/2)Q_3 for τ (generation 3).
  This is the iso's B(P)|_V_Ram-correspondent operator on Cl(6) Fock.

CRITICAL OBSERVATION (this probe's finding):
  D_τ commutes with γ_7 (chirality):
    - γ_7 commutes with itself
    - Q_3 = γ_1γ_2γ_3γ_4 is even-grade (4 gammas) → commutes with γ_7
  Therefore D_τ is BLOCK-DIAGONAL with respect to chirality.
  ⟨τ_L | D_τ | τ_R⟩ = 0 (Γ_7(τ_L) = +1 ≠ -1 = Γ_7(τ_R))

  So D_τ ≠ Yukawa operator. The Yukawa MUST flip chirality (anticommute
  with γ_7), so it's an ODD-grade combination of γ^a generators.

  IDENTIFICATION:
    Y_τ = (Yukawa coupling) × (γ^a · h⁰_a structure)
    where h⁰_a are Higgs edge-qubit components projected onto Cl(6).

PROBE STRATEGY:
  1. Verify D_τ is block-diagonal in chirality (matrix element computation
     gives zero for L↔R transitions)
  2. Construct candidate Y_τ operators: odd-grade combinations
  3. Identify which Y_τ corresponds to γ^a · h⁰_a per the framework's
     theorem_g2_edge_qubit_su2 (Higgs ↔ Cl(0,2) edge qubit)
  4. Compute ⟨τ_L | Y_τ | τ_R⟩ and compare to y_τ ≈ 0.007
"""

import sys, os
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

TOL = 1e-9

# Cl(6,0) Brauer-Weyl
def kron(*mats):
    out = mats[0]
    for m in mats[1:]: out = np.kron(out, m)
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

# Cartan operators (R1_1 convention): T_1 = M_12, T_2 = M_34, Y = M_56
def bivector(a, b):
    return (G[a] @ G[b] - G[b] @ G[a]) / (4j)

T_1 = bivector(1, 2)
T_2 = bivector(3, 4)
Y_op = bivector(5, 6)


# ============================================================
# STEP 1: Identify τ_L, τ_R Fock states
# ============================================================
# In Brauer-Weyl basis |n_1 n_2 n_3⟩ (n_i ∈ {0,1}), Cartan operators are:
#   T_1 = (1/2) sz ⊗ I2 ⊗ I2     → 2T_1 = (-1)^n_1
#   T_2 = (1/2) I2 ⊗ sz ⊗ I2     → 2T_2 = (-1)^n_2
#   Y   = (1/2) I2 ⊗ I2 ⊗ sz     → 2Y   = (-1)^n_3
# So state |n_1 n_2 n_3⟩ has weight (+1 if n_i=0 else -1) in each slot.
#
# Γ_7 = -i Γ_1 Γ_2 Γ_3 Γ_4 Γ_5 Γ_6 acts as... let's compute the relation
# to (n_1, n_2, n_3).
# Per Brauer-Weyl, Γ_7 = product of all generators (with sign). On |000⟩
# it gives some specific eigenvalue.

# Numerical check: get Γ_7 eigenvalues + Cartan eigenvalues per basis state
print("=" * 78)
print("  T5 — Yukawa matrix element on Cl(6) Fock")
print("=" * 78)

print("\n  Brauer-Weyl basis state weights:")
print(f"  {'state':>8} {'2T_1':>6} {'2T_2':>6} {'2Y':>6} {'Γ_7':>5}")
print(f"  {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*5}")
for k in range(8):
    e_k = np.zeros(8, dtype=complex)
    e_k[k] = 1
    t1 = int(round(2 * np.real(e_k.conj() @ T_1 @ e_k)))
    t2 = int(round(2 * np.real(e_k.conj() @ T_2 @ e_k)))
    y  = int(round(2 * np.real(e_k.conj() @ Y_op @ e_k)))
    g7 = int(round(np.real(e_k.conj() @ G7 @ e_k)))
    n1, n2, n3 = (k >> 2) & 1, (k >> 1) & 1, k & 1
    print(f"  |{n1}{n2}{n3}⟩ ({k:>2}) {t1:>+6} {t2:>+6} {y:>+6} {g7:>+5}")

# τ_L (electron, left-handed): Y = -1 (lepton), Γ_7 = +1 (left)
# τ_R (electron, right-handed): Y = -1 (lepton), Γ_7 = -1 (right)
# Within each chirality, "electron" vs "neutrino" distinguished by T_1 vs T_2:
#   e_L: T_3L = -1/2 (down-type isospin)
#   ν_L: T_3L = +1/2 (up-type isospin)
# In R1_1 reading: T_3L corresponds to one of the Cartan ops T_1 or T_2.

# To pick τ_L and τ_R as specific Fock states: states with same Y, opposite Γ_7,
# and "electron" labeling. Use convention: e_L = state with Y=-1, Γ_7=+1, T_1=-1/2
# i.e. 2T_1 = -1.

# Find τ_L
tau_L_candidates = []
tau_R_candidates = []
for k in range(8):
    e_k = np.zeros(8, dtype=complex)
    e_k[k] = 1
    t1 = int(round(2 * np.real(e_k.conj() @ T_1 @ e_k)))
    y  = int(round(2 * np.real(e_k.conj() @ Y_op @ e_k)))
    g7 = int(round(np.real(e_k.conj() @ G7 @ e_k)))
    # e_L = Y=-1 (lepton), Γ_7=+1 (left), 2T_1=-1 (down-isospin)
    if y == -1 and g7 == +1 and t1 == -1:
        tau_L_candidates.append(k)
    # e_R = Y=-1 (lepton), Γ_7=-1 (right)
    if y == -1 and g7 == -1:
        tau_R_candidates.append(k)

print(f"\n  τ_L candidates (Y=-1, Γ_7=+1, T_1=-1/2): {tau_L_candidates}")
print(f"  τ_R candidates (Y=-1, Γ_7=-1):           {tau_R_candidates}")

if not tau_L_candidates or not tau_R_candidates:
    print("\n  ERROR: τ_L or τ_R not found with expected weights.")
    print("  Falling back to first Y=-1 Γ_7=±1 state for each.")
    tau_L_idx = next(k for k in range(8) if
                     int(round(2 * np.real(np.eye(8, dtype=complex)[k].conj() @ Y_op @ np.eye(8, dtype=complex)[k]))) == -1
                     and int(round(np.real(np.eye(8, dtype=complex)[k].conj() @ G7 @ np.eye(8, dtype=complex)[k]))) == 1)
    tau_R_idx = next(k for k in range(8) if
                     int(round(2 * np.real(np.eye(8, dtype=complex)[k].conj() @ Y_op @ np.eye(8, dtype=complex)[k]))) == -1
                     and int(round(np.real(np.eye(8, dtype=complex)[k].conj() @ G7 @ np.eye(8, dtype=complex)[k]))) == -1)
else:
    tau_L_idx = tau_L_candidates[0]
    tau_R_idx = tau_R_candidates[0]

tau_L = np.zeros(8, dtype=complex)
tau_L[tau_L_idx] = 1
tau_R = np.zeros(8, dtype=complex)
tau_R[tau_R_idx] = 1

print(f"\n  Selected: τ_L = e_{tau_L_idx}, τ_R = e_{tau_R_idx}")


# ============================================================
# STEP 2: Verify D_τ is chirality-preserving (matrix element = 0)
# ============================================================
Q_3 = G[1] @ G[2] @ G[3] @ G[4]
D_tau = (np.sqrt(3)/2) * G7 + 1j * (np.sqrt(5)/2) * Q_3

# Check [D_tau, G7] = 0
comm_D_G7 = D_tau @ G7 - G7 @ D_tau
print(f"\n  [D_τ, Γ_7] = 0: {np.allclose(comm_D_G7, 0, atol=TOL)} (max |·| = {np.max(np.abs(comm_D_G7)):.2e})")

# Matrix element ⟨τ_L | D_τ | τ_R⟩
me_D = tau_L.conj() @ D_tau @ tau_R
print(f"  ⟨τ_L | D_τ | τ_R⟩ = {me_D:.4e}")
print(f"  → As expected, D_τ does NOT give Yukawa (chirality-preserving)")


# ============================================================
# STEP 3: The Yukawa operator must flip chirality (anticommute with γ_7)
# ============================================================
# Single Cl(6) generators γ^a anticommute with γ_7. So Y_τ is some
# combination of single generators (odd-grade combination).
#
# Per theorem_g2_edge_qubit_su2, h⁰ ↔ f_1 (one edge qubit basis vector).
# Per the framework's Yukawa derivation, Y_τ couples to specific γ^a.
#
# For PS leptonic Yukawa (e_L → e_R via Higgs): the coupling involves
# specific PS multiplet indices that map to specific Cl(6) generators.
#
# Without the explicit Higgs ↔ Cl(6) bridge from theorem_g2_edge_qubit_su2,
# we cannot uniquely identify Y_τ. But we can compute matrix elements for
# CANDIDATE Y_τ operators and see which gives y_τ ≈ 0.007.

print("\n" + "=" * 78)
print("  Step 3: Candidate Y_τ operators (must anticommute with γ_7)")
print("=" * 78)

y_tau_target = 1.776 / 246.22   # ≈ 0.00722

candidates = [
    ("γ_1",        G[1]),
    ("γ_2",        G[2]),
    ("γ_3",        G[3]),
    ("γ_4",        G[4]),
    ("γ_5",        G[5]),
    ("γ_6",        G[6]),
    ("γ_1γ_2γ_3",  G[1] @ G[2] @ G[3]),   # spatial trivector
    ("γ_4γ_5γ_6",  G[4] @ G[5] @ G[6]),   # internal trivector
    ("γ_1γ_2γ_3 + γ_4γ_5γ_6",  G[1] @ G[2] @ G[3] + G[4] @ G[5] @ G[6]),
    ("Σ γ^a (sum of all 6 generators)",  sum(G[a] for a in range(1, 7))),
]

print(f"\n  Target: y_τ = m_τ / v_Higgs = {y_tau_target:.5f}")
print(f"\n  {'Y_τ candidate':<40} {'⟨τ_L | Y_τ | τ_R⟩':>22} {'|·|':>10}")
print(f"  {'-'*40} {'-'*22} {'-'*10}")
for name, Y in candidates:
    # Check Y anticommutes with γ_7
    anticomm = np.allclose(Y @ G7 + G7 @ Y, 0, atol=TOL)
    if not anticomm:
        # Even grade — won't connect L↔R
        me = 0
        print(f"  {name:<40} (even-grade, no L↔R) {0:>10.4e}")
    else:
        me = complex(tau_L.conj() @ Y @ tau_R)
        print(f"  {name:<40} {me:>22.4e} {abs(me):>10.4e}")

# Walker-survival factor multiplied
alpha_1 = (2/3)**8   # ≈ 0.039
print(f"\n  α_1 walker survival factor (2/3)^8 = {alpha_1:.5f}")
print(f"  α_1 × matrix-element candidates:")
for name, Y in candidates:
    anticomm = np.allclose(Y @ G7 + G7 @ Y, 0, atol=TOL)
    if anticomm:
        me = complex(tau_L.conj() @ Y @ tau_R)
        product = alpha_1 * abs(me)
        ratio = product / y_tau_target if y_tau_target > 0 else 0
        print(f"    {name:<40} α_1 × |·| = {product:.5f}  ratio to y_τ = {ratio:.3f}")


# ============================================================
# STEP 4: HONEST T5 CONCLUSION
# ============================================================
print("\n" + "=" * 78)
print("  T5 — HONEST CONCLUSION")
print("=" * 78)
print(f"""
  KEY FINDING (Session 4):
    D_τ = (√3/2)γ_7 + i(√5/2)γ_1γ_2γ_3γ_4 (from T4 closure) is the
    iso's B(P)|_V_Ram-correspondent operator, but it COMMUTES with γ_7
    (chirality-preserving). Therefore D_τ ≠ Yukawa operator.

    The Yukawa operator Y_τ must ANTICOMMUTE with γ_7 (chirality-flipping)
    to mediate τ_R → τ_L transitions.

  ⟨τ_L | D_τ | τ_R⟩ = {me_D:.4e}  (zero, as expected)

  CANDIDATE Y_τ ANALYSIS:
    Single γ^a generators give nonzero matrix elements only when their
    index matches the specific (τ_L, τ_R) Fock-state coupling pattern.
    Computation above shows which γ^a candidates connect τ_L and τ_R.

  T5 STATUS: PARTIAL — structural form for Yukawa operator identified
    (must be odd-grade), but specific Y_τ = γ^a · h⁰_a requires:
    (i)   Edge-sector integration of Higgs coupling (theorem_g2_edge_qubit
          gives h⁰ ↔ f_1 edge qubit, but the explicit mapping to Cl(6)
          generators via the framework's Yukawa structure is needed)
    (ii)  Walker survival factor incorporation (α_1 = (2/3)^8 ≈ 0.039)
    (iii) Channel factor from PS structure

    Without (i)-(iii), the specific Y_τ matrix element cannot uniquely
    determine y_τ. The candidate matrix elements computed above give
    various values; the specific combination matching y_τ ≈ {y_tau_target:.5f}
    depends on the (currently open) edge-vertex coupling structure.

  CONTRIBUTION TO T5 PROGRAM:
    - Confirms D_τ is NOT the Yukawa operator (chirality-preserving)
    - Identifies that Y_τ must be ODD-grade (chirality-flipping)
    - Sets up the framework for explicit Y_τ identification via edge bridge
    - Identifies remaining open pieces (edge integration + walker factor)

  T5 GRADE: STRUCTURAL FOUNDATION COMPLETE.
    Full y_τ computation requires edge-sector integration (multi-session
    research). The ISO program's T1-T4 contribute the canonical setup;
    T5 needs ONE MORE structural ingredient (edge-sector bridge) to
    close completely.

  HONEST FINAL STATE OF ISO PROGRAM:
    T1: CLOSED (theorem-grade)
    T2: CLOSED (diagonal Spin(3) lift)
    T3: CLOSED-AS-NEGATIVE
    T4: CLOSED (generation-dependent D_i = (√3/2)γ_7 + i(√5/2)Q_i)
    T5: STRUCTURAL FOUNDATION COMPLETE; specific Y_τ requires edge bridge
""")
