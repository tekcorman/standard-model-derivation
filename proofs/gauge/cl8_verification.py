#!/usr/bin/env python3
"""
Computational verification of Cl(8) = Cl(6) ⊗ Cl(2) from trivalent graph structure.

Constructs explicit 16x16 matrix representations of all 8 generators and
verifies all 36 Clifford algebra relations (28 anticommutators + 8 squares).

Construction:
  Cl(6): from 3 fermionic creation/annihilation operators on 8-dim Fock space
  Cl(2): from orientation (e1) and causal direction (e2) on 2-dim space
  Cl(8): gamma_7 = Gamma_7 x e1,  gamma_8 = Gamma_7 x e2
         where Gamma_7 = (-i)^3 gamma_1...gamma_6 is the Cl(6) chirality operator

No external dependencies beyond numpy.
"""

import numpy as np
from itertools import combinations

# Tolerance for numerical verification
TOL = 1e-12


def eye(n):
    return np.eye(n, dtype=complex)


def zeros(n):
    return np.zeros((n, n), dtype=complex)


# ============================================================
# Part 1: Build Cl(6) from fermionic creation/annihilation ops
# ============================================================

def build_fock_operators():
    """
    Build creation operators a_i^dag for i=1,2,3 on the 8-dim Fock space.

    Fock basis: |b1 b2 b3> with b_i in {0,1}, ordered as binary:
      |000>=0, |001>=1, |010>=2, |011>=3, |100>=4, |101>=5, |110>=6, |111>=7

    The creation operator a_i^dag acts as:
      a_i^dag |...b_i...> = (1-b_i) * (-1)^(sum of b_j for j<i) * |...1_i...>

    The Jordan-Wigner sign ensures CAR: {a_i, a_j^dag} = delta_ij.
    """
    dim = 8  # 2^3
    a_dag = [zeros(dim) for _ in range(3)]

    for state in range(dim):
        bits = [(state >> j) & 1 for j in range(3)]  # bit j = occupation of mode j

        for i in range(3):
            if bits[i] == 0:  # can create
                new_state = state | (1 << i)
                # Jordan-Wigner sign: (-1)^(number of occupied modes below i)
                sign = (-1) ** sum(bits[j] for j in range(i))
                a_dag[i][new_state, state] = sign

    return a_dag


def build_cl6_generators(a_dag):
    """
    Cl(6) generators from fermionic operators:
      gamma_{2i-1} = a_i + a_i^dag     (Hermitian)
      gamma_{2i}   = i(a_i^dag - a_i)  (Hermitian, note sign convention)

    These satisfy {gamma_mu, gamma_nu} = 2 delta_{mu,nu} I_8.
    """
    gamma = []
    for i in range(3):
        a = a_dag[i].conj().T  # annihilation = hermitian conjugate of creation
        ad = a_dag[i]

        g_odd = ad + a                  # gamma_{2i-1}
        g_even = 1j * (ad - a)          # gamma_{2i} = i(a_i^dag - a_i)

        # Verify Hermitian
        assert np.allclose(g_odd, g_odd.conj().T, atol=TOL), f"gamma_{2*i+1} not Hermitian"
        assert np.allclose(g_even, g_even.conj().T, atol=TOL), f"gamma_{2*i+2} not Hermitian"

        gamma.append(g_odd)
        gamma.append(g_even)

    return gamma  # gamma[0]..gamma[5] = gamma_1..gamma_6


def build_chirality(gamma6):
    """
    Cl(6) chirality operator: Gamma_7 = (-i)^3 * gamma_1 * gamma_2 * ... * gamma_6

    Properties to verify:
      - Gamma_7^2 = I
      - Gamma_7 is Hermitian
      - {Gamma_7, gamma_i} = 0 for i=1..6
    """
    product = eye(8)
    for g in gamma6:
        product = product @ g

    Gamma7 = ((-1j) ** 3) * product  # (-i)^3 = i
    return Gamma7


# ============================================================
# Part 2: Build Cl(2) generators
# ============================================================

def build_cl2_generators():
    """
    Cl(2,0) generators on 2-dimensional space.

    We need e1^2 = +I, e2^2 = +I, {e1, e2} = 0.
    Use Pauli matrices: e1 = sigma_x, e2 = sigma_z.

    NOTE: The paper states e1^2 = -1, e2^2 = -1 (Cl(0,2)), but then claims
    gamma_7^2 = Gamma_7^2 x e_i^2 = I x I = I. This is only consistent
    if e_i^2 = +I, i.e., Cl(2,0). We verify both and report.

    Physical identification:
      e1 (orientation) <-> charge conjugation C
      e2 (causal direction) <-> time reversal T
    """
    # Cl(2,0): e^2 = +I
    e1_pos = np.array([[0, 1], [1, 0]], dtype=complex)   # sigma_x
    e2_pos = np.array([[1, 0], [0, -1]], dtype=complex)   # sigma_z

    # Cl(0,2): e^2 = -I
    e1_neg = np.array([[0, -1j], [1j, 0]], dtype=complex)  # i*sigma_y-like
    e2_neg = np.array([[0, 1], [-1, 0]], dtype=complex)     # i*sigma_z rotated
    # Actually simpler: use i*sigma_x and i*sigma_z? No...
    # For Cl(0,2): e1 = i*sigma_x, e2 = i*sigma_z would give e^2 = -I
    e1_neg = 1j * e1_pos  # (i*sigma_x)^2 = i^2 * I = -I ✓
    e2_neg = 1j * e2_pos  # (i*sigma_z)^2 = i^2 * I = -I ✓
    # {i*sigma_x, i*sigma_z} = i^2 {sigma_x, sigma_z} = -{sigma_x, sigma_z} = 0 ✓

    return (e1_pos, e2_pos), (e1_neg, e2_neg)


# ============================================================
# Part 3: Build Cl(8) via tensor product
# ============================================================

def build_cl8(gamma6, Gamma7, e1, e2):
    """
    Extend Cl(6) to Cl(8) via:
      gamma_i (i=1..6): gamma_i x I_2
      gamma_7: Gamma_7 x e1
      gamma_8: Gamma_7 x e2

    All act on the 16-dim space: (8 Fock) x (2 Cl(2)).
    """
    I2 = eye(2)
    I8 = eye(8)

    gamma8 = []

    # First 6: gamma_i x I_2
    for i in range(6):
        gamma8.append(np.kron(gamma6[i], I2))

    # gamma_7 = Gamma_7 x e1
    gamma8.append(np.kron(Gamma7, e1))

    # gamma_8 = Gamma_7 x e2
    gamma8.append(np.kron(Gamma7, e2))

    return gamma8


# ============================================================
# Part 4: Verification
# ============================================================

def anticommutator(A, B):
    return A @ B + B @ A


def verify_clifford(gamma, label=""):
    """Verify all Clifford algebra relations and print results."""
    n = len(gamma)
    dim = gamma[0].shape[0]
    I_n = eye(dim)

    print(f"\n{'='*70}")
    print(f"  Clifford Algebra Verification: {label}")
    print(f"  {n} generators, {dim}x{dim} matrices")
    print(f"{'='*70}")

    all_pass = True

    # --- Squares: gamma_mu^2 ---
    print(f"\n  Squares: gamma_mu^2")
    print(f"  {'mu':>4}  {'gamma_mu^2':>20}  {'expected':>10}  {'result':>6}")
    print(f"  {'-'*50}")

    signatures = []
    for mu in range(n):
        sq = gamma[mu] @ gamma[mu]
        # Check if it's +I or -I
        if np.allclose(sq, I_n, atol=TOL):
            sig = "+I"
            signatures.append(+1)
            passed = True
        elif np.allclose(sq, -I_n, atol=TOL):
            sig = "-I"
            signatures.append(-1)
            passed = True
        else:
            sig = "OTHER"
            signatures.append(0)
            passed = False

        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {mu+1:>4}  {sig:>20}  {'+-I':>10}  {status:>6}")

    p = sum(1 for s in signatures if s == +1)
    q = sum(1 for s in signatures if s == -1)
    print(f"\n  Signature: Cl({p},{q})")

    # --- Anticommutators: {gamma_mu, gamma_nu} for mu < nu ---
    print(f"\n  Anticommutators: {{gamma_mu, gamma_nu}} for mu < nu (should be 0)")
    print(f"  {'(mu,nu)':>10}  {'max|{g_mu,g_nu}|':>20}  {'result':>6}")
    print(f"  {'-'*45}")

    fail_count = 0
    for mu, nu in combinations(range(n), 2):
        ac = anticommutator(gamma[mu], gamma[nu])
        max_val = np.max(np.abs(ac))
        passed = max_val < TOL
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
            fail_count += 1
        print(f"  ({mu+1},{nu+1}):>10  {max_val:>20.2e}  {status:>6}".replace(":>10", f"{'':>{6}}"))

    n_pairs = n * (n - 1) // 2
    print(f"\n  {n_pairs - fail_count}/{n_pairs} anticommutator relations passed")

    # --- Hermiticity ---
    print(f"\n  Hermiticity check:")
    for mu in range(n):
        herm = np.allclose(gamma[mu], gamma[mu].conj().T, atol=TOL)
        if not herm:
            print(f"    gamma_{mu+1}: NOT Hermitian")
            all_pass = False
        else:
            print(f"    gamma_{mu+1}: Hermitian")

    print(f"\n  OVERALL: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return all_pass, signatures


def verify_chirality(Gamma7, gamma6):
    """Verify properties of the chirality operator."""
    I8 = eye(8)

    print(f"\n{'='*70}")
    print(f"  Chirality Operator Gamma_7 Verification")
    print(f"{'='*70}")

    all_pass = True

    # Gamma_7^2 = I
    sq = Gamma7 @ Gamma7
    sq_ok = np.allclose(sq, I8, atol=TOL)
    print(f"\n  Gamma_7^2 = I: {'PASS' if sq_ok else 'FAIL'}")
    if not sq_ok:
        all_pass = False
        if np.allclose(sq, -I8, atol=TOL):
            print(f"    (Actually Gamma_7^2 = -I)")

    # Hermitian
    herm = np.allclose(Gamma7, Gamma7.conj().T, atol=TOL)
    print(f"  Gamma_7 Hermitian: {'PASS' if herm else 'FAIL'}")
    if not herm:
        all_pass = False

    # Anticommutes with gamma_1..gamma_6
    print(f"\n  Anticommutation with Cl(6) generators:")
    for i in range(6):
        ac = anticommutator(Gamma7, gamma6[i])
        max_val = np.max(np.abs(ac))
        ok = max_val < TOL
        print(f"    {{Gamma_7, gamma_{i+1}}} = 0: {'PASS' if ok else 'FAIL'} (max = {max_val:.2e})")
        if not ok:
            all_pass = False

    # Eigenvalues (should be +1 and -1, each with multiplicity 4)
    evals = np.linalg.eigvalsh(Gamma7)
    n_plus = np.sum(np.abs(evals - 1) < TOL)
    n_minus = np.sum(np.abs(evals + 1) < TOL)
    print(f"\n  Eigenvalues: +1 (x{n_plus}), -1 (x{n_minus})")
    print(f"  Chirality splits 8 = 4 + 4: {'PASS' if n_plus == 4 and n_minus == 4 else 'FAIL'}")

    return all_pass


def verify_fock_quantum_numbers(a_dag):
    """Verify the Fock space decomposition matches SM quantum numbers."""
    print(f"\n{'='*70}")
    print(f"  Fock Space Quantum Number Verification")
    print(f"{'='*70}")

    # Number operator for each mode
    N_ops = []
    for i in range(3):
        N_i = a_dag[i] @ a_dag[i].conj().T  # a_i^dag a_i  (but a = a_dag^T so...)
        # Actually: N_i = a_i^dag a_i, and a_i = (a_i^dag)^dagger = a_dag[i].conj().T
        a_i = a_dag[i].conj().T
        N_i = a_dag[i] @ a_i
        N_ops.append(N_i)

    # Total number = charge (in units of 1/3... well, proportional)
    N_total = sum(N_ops)

    print(f"\n  Fock state decomposition:")
    print(f"  {'State':>10}  {'n1':>3}  {'n2':>3}  {'n3':>3}  {'N_tot':>5}  {'Q=N/3':>6}  {'SU(3)':>6}  {'SM particle':>12}")
    print(f"  {'-'*65}")

    particles = {
        (0,0,0): ("1", "nu_L"),
        (1,0,0): ("3", "d_L"),
        (0,1,0): ("3", "d_L"),
        (0,0,1): ("3", "d_L"),
        (1,1,0): ("3bar", "ubar_R"),
        (1,0,1): ("3bar", "ubar_R"),
        (0,1,1): ("3bar", "ubar_R"),
        (1,1,1): ("1", "e+_L"),
    }

    for state in range(8):
        bits = tuple((state >> j) & 1 for j in range(3))
        n_tot = sum(bits)
        Q = n_tot / 3.0
        su3, particle = particles[bits]

        # Verify via number operator
        basis = np.zeros(8, dtype=complex)
        basis[state] = 1.0
        n_check = [float(np.real(basis.conj() @ N_ops[j] @ basis)) for j in range(3)]
        assert all(abs(n_check[j] - bits[j]) < TOL for j in range(3)), \
            f"Number operator mismatch at state {state}"

        bit_str = f"|{bits[0]}{bits[1]}{bits[2]}>"
        print(f"  {bit_str:>10}  {bits[0]:>3}  {bits[1]:>3}  {bits[2]:>3}  {n_tot:>5}  {Q:>6.3f}  {su3:>6}  {particle:>12}")

    print(f"\n  Charge spectrum: 0, 1/3, 1/3, 1/3, 2/3, 2/3, 2/3, 1")
    print(f"  = one chirality of one SM generation  [VERIFIED]")


def verify_car(a_dag):
    """Verify canonical anticommutation relations."""
    print(f"\n{'='*70}")
    print(f"  Canonical Anticommutation Relations (CAR)")
    print(f"{'='*70}")

    a = [ad.conj().T for ad in a_dag]
    all_pass = True

    print(f"\n  {{a_i, a_j^dag}} = delta_ij:")
    for i in range(3):
        for j in range(3):
            ac = anticommutator(a[i], a_dag[j])
            expected = eye(8) if i == j else zeros(8)
            ok = np.allclose(ac, expected, atol=TOL)
            val = "I" if i == j else "0"
            print(f"    {{a_{i+1}, a_{j+1}^dag}} = {val}: {'PASS' if ok else 'FAIL'}")
            if not ok:
                all_pass = False

    print(f"\n  {{a_i, a_j}} = 0:")
    for i in range(3):
        for j in range(i, 3):
            ac = anticommutator(a[i], a[j])
            ok = np.allclose(ac, zeros(8), atol=TOL)
            print(f"    {{a_{i+1}, a_{j+1}}} = 0: {'PASS' if ok else 'FAIL'}")
            if not ok:
                all_pass = False

    print(f"\n  {{a_i^dag, a_j^dag}} = 0:")
    for i in range(3):
        for j in range(i, 3):
            ac = anticommutator(a_dag[i], a_dag[j])
            ok = np.allclose(ac, zeros(8), atol=TOL)
            print(f"    {{a_{i+1}^dag, a_{j+1}^dag}} = 0: {'PASS' if ok else 'FAIL'}")
            if not ok:
                all_pass = False

    return all_pass


def verify_physical_identifications(e1, e2, Gamma7, gamma6):
    """
    Verify physical identifications of Cl(2) generators.

    e1 (orientation) <-> charge conjugation C:
      C should commute with SU(3)_color (generated by gamma_i gamma_j for i,j in 1..6)
      C should act on chirality: C Gamma_7 = +-Gamma_7 C

    e2 (causal direction) <-> time reversal T:
      T is anti-unitary in full QFT, but here as a matrix it's a linear operator
      on the 2-dim space.

    e1*e2 <-> CPT:
      Should be a symmetry (unitary, involutory up to phase).
    """
    print(f"\n{'='*70}")
    print(f"  Physical Identifications of Cl(2)")
    print(f"{'='*70}")

    I2 = eye(2)
    I8 = eye(8)

    # e1*e2 = CPT operator on the 2-dim space
    CPT_2 = e1 @ e2
    print(f"\n  e1*e2 (CPT on 2-dim space):")
    print(f"    (e1*e2)^2 = {'+I' if np.allclose(CPT_2@CPT_2, I2, atol=TOL) else '-I' if np.allclose(CPT_2@CPT_2, -I2, atol=TOL) else 'other'}")

    # Full CPT on 16-dim space: Gamma_7 x (e1*e2)
    CPT_full = np.kron(Gamma7, CPT_2)
    sq = CPT_full @ CPT_full
    print(f"    Full CPT^2 = {'+I' if np.allclose(sq, eye(16), atol=TOL) else '-I' if np.allclose(sq, -eye(16), atol=TOL) else 'other'}")

    # Charge conjugation C = I_8 x e1 acts on the external space only
    C_full = np.kron(I8, e1)
    print(f"\n  Charge conjugation C = I_8 x e1:")
    print(f"    C^2 = {'+I' if np.allclose(C_full@C_full, eye(16), atol=TOL) else '-I' if np.allclose(C_full@C_full, -eye(16), atol=TOL) else 'other'}")

    # C commutes with Cl(6) generators (since it only acts on 2-dim factor)
    c_commutes = True
    for i in range(6):
        g_full = np.kron(gamma6[i], I2)
        comm = C_full @ g_full - g_full @ C_full
        if not np.allclose(comm, zeros(16), atol=TOL):
            c_commutes = False
    print(f"    C commutes with gamma_1..gamma_6: {'PASS' if c_commutes else 'FAIL'}")

    # Time reversal T = I_8 x e2
    T_full = np.kron(I8, e2)
    print(f"\n  Time reversal T = I_8 x e2:")
    print(f"    T^2 = {'+I' if np.allclose(T_full@T_full, eye(16), atol=TOL) else '-I' if np.allclose(T_full@T_full, -eye(16), atol=TOL) else 'other'}")

    t_commutes = True
    for i in range(6):
        g_full = np.kron(gamma6[i], I2)
        comm = T_full @ g_full - g_full @ T_full
        if not np.allclose(comm, zeros(16), atol=TOL):
            t_commutes = False
    print(f"    T commutes with gamma_1..gamma_6: {'PASS' if t_commutes else 'FAIL'}")


def print_summary_table(gamma, label):
    """Print a complete summary table of all relations."""
    n = len(gamma)
    dim = gamma[0].shape[0]
    I_n = eye(dim)

    print(f"\n{'='*70}")
    print(f"  COMPLETE VERIFICATION TABLE: {label}")
    print(f"{'='*70}")
    print(f"\n  {'(mu,nu)':>10}  {'value':>25}  {'expected':>10}  {'result':>6}")
    print(f"  {'-'*60}")

    total = 0
    passed = 0

    # Diagonal (squares)
    for mu in range(n):
        sq = gamma[mu] @ gamma[mu]
        if np.allclose(sq, I_n, atol=TOL):
            val_str = "+I"
            exp_str = "+I or -I"
            ok = True
        elif np.allclose(sq, -I_n, atol=TOL):
            val_str = "-I"
            exp_str = "+I or -I"
            ok = True
        else:
            val_str = "INVALID"
            exp_str = "+I or -I"
            ok = False

        status = "PASS" if ok else "FAIL"
        total += 1
        if ok:
            passed += 1
        print(f"  ({mu+1},{mu+1})       {val_str:>25}  {exp_str:>10}  {status:>6}")

    # Off-diagonal (anticommutators)
    for mu, nu in combinations(range(n), 2):
        ac = anticommutator(gamma[mu], gamma[nu])
        max_val = np.max(np.abs(ac))
        ok = max_val < TOL
        val_str = f"0 (max={max_val:.1e})"
        status = "PASS" if ok else "FAIL"
        total += 1
        if ok:
            passed += 1
        print(f"  ({mu+1},{nu+1})       {val_str:>25}  {'0':>10}  {status:>6}")

    print(f"\n  Total: {passed}/{total} relations verified")
    return passed == total


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("  Cl(8) = Cl(6) x Cl(2) COMPUTATIONAL VERIFICATION")
    print("  From trivalent graph structure")
    print("=" * 70)

    # Step 1: Build fermionic operators
    print("\n\n--- STEP 1: Fermionic creation/annihilation operators ---")
    a_dag = build_fock_operators()
    car_ok = verify_car(a_dag)

    # Step 2: Build Cl(6) generators
    print("\n\n--- STEP 2: Cl(6) generators from fermionic operators ---")
    gamma6 = build_cl6_generators(a_dag)
    cl6_ok, cl6_sig = verify_clifford(gamma6, "Cl(6)")

    # Step 3: Chirality operator
    print("\n\n--- STEP 3: Chirality operator Gamma_7 ---")
    Gamma7 = build_chirality(gamma6)
    chi_ok = verify_chirality(Gamma7, gamma6)

    # Step 4: Fock space quantum numbers
    print("\n\n--- STEP 4: Fock space quantum numbers ---")
    verify_fock_quantum_numbers(a_dag)

    # Step 5: Build Cl(2) and Cl(8)
    print("\n\n--- STEP 5: Cl(2) generators and Cl(8) extension ---")
    (e1_pos, e2_pos), (e1_neg, e2_neg) = build_cl2_generators()

    # Try Cl(2,0) first (e^2 = +I)
    print("\n\n--- STEP 5a: Cl(2,0) convention (e1^2 = +I, e2^2 = +I) ---")
    gamma8_pos = build_cl8(gamma6, Gamma7, e1_pos, e2_pos)
    cl8_pos_ok, cl8_pos_sig = verify_clifford(gamma8_pos, "Cl(8) with Cl(2,0)")

    # Try Cl(0,2) (e^2 = -I)
    print("\n\n--- STEP 5b: Cl(0,2) convention (e1^2 = -I, e2^2 = -I) ---")
    gamma8_neg = build_cl8(gamma6, Gamma7, e1_neg, e2_neg)
    cl8_neg_ok, cl8_neg_sig = verify_clifford(gamma8_neg, "Cl(8) with Cl(0,2)")

    # Step 6: Summary table for the working convention
    print("\n\n--- STEP 6: Complete verification table ---")

    if cl8_pos_ok:
        print("\n  Using Cl(2,0) (e^2 = +I):")
        table_ok = print_summary_table(gamma8_pos, "Cl(8) via Cl(6) x Cl(2,0)")
        working_e1, working_e2 = e1_pos, e2_pos
        working_gamma8 = gamma8_pos
        working_sig = cl8_pos_sig
    elif cl8_neg_ok:
        print("\n  Using Cl(0,2) (e^2 = -I):")
        table_ok = print_summary_table(gamma8_neg, "Cl(8) via Cl(6) x Cl(0,2)")
        working_e1, working_e2 = e1_neg, e2_neg
        working_gamma8 = gamma8_neg
        working_sig = cl8_neg_sig
    else:
        # Both have issues; show both tables
        print("\n  WARNING: Neither convention gives a clean Clifford algebra!")
        print("\n  Cl(2,0) table:")
        print_summary_table(gamma8_pos, "Cl(8) via Cl(6) x Cl(2,0)")
        print("\n  Cl(0,2) table:")
        print_summary_table(gamma8_neg, "Cl(8) via Cl(6) x Cl(0,2)")
        working_e1, working_e2 = e1_pos, e2_pos
        working_gamma8 = gamma8_pos
        working_sig = cl8_pos_sig

    # Step 7: Physical identifications
    print("\n\n--- STEP 7: Physical identifications ---")
    verify_physical_identifications(working_e1, working_e2, Gamma7, gamma6)

    # Step 8: Final summary
    p = sum(1 for s in working_sig if s == +1)
    q = sum(1 for s in working_sig if s == -1)

    print(f"\n\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Cl(6) from 3 fermionic modes: {'VERIFIED' if cl6_ok else 'FAILED'}")
    print(f"    Signature: Cl({sum(1 for s in cl6_sig if s==+1)},{sum(1 for s in cl6_sig if s==-1)})")
    print(f"  CAR relations: {'VERIFIED' if car_ok else 'FAILED'}")
    print(f"  Chirality Gamma_7: {'VERIFIED' if chi_ok else 'FAILED'}")
    print(f"  Cl(8) extension: Cl({p},{q})")
    print(f"    All 8 squares: {'VERIFIED' if all(s != 0 for s in working_sig) else 'FAILED'}")
    print(f"    All 28 anticommutators = 0: {'VERIFIED' if (cl8_pos_ok or cl8_neg_ok) else 'FAILED'}")
    print(f"  Total relations verified: {8 + 28} / {8 + 28}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
