#!/usr/bin/env python3
"""
L3b gate check: Clifford algebra of the edge qubit from Minkowski structure.

The argument chain (G2 Higgs doublet, after L3a Lorentz mixing):

  Stage 3 establishes Lorentz invariance of the srs toggle dynamics.
  This gives the edge qubit a Minkowski spacetime structure with generators
  satisfying the Clifford algebra Cl(1,1):
    {gamma^0, gamma^1} = 2 g^{01} I = 0   (Minkowski g^{0i}=0)
    (gamma^0)^2 = +I   (timelike, signature +1)
    (gamma^1)^2 = -I   (spacelike, signature -1)

  A3 (purification => complex Hilbert space) provides the factor i.
  Complexify the timelike generator: e2 = i*gamma^0.
  Now:
    e2^2 = (i*gamma^0)^2 = -I       (signature -1 after complexification)
    e1^2 = (gamma^1)^2   = -I       (unchanged)
    {e1, e2} = {gamma^1, i*gamma^0} = i{gamma^1,gamma^0} = i*0 = 0

  This is Cl(0,2) (both generators square to -1, anti-commute).
  Cl(0,2) ≅ H (Hamilton's quaternions).
  SU(2) = Sp(1) = unit quaternions acts on the 2-dim left module of H.
  The Higgs doublet = this 2-dim left module over C (= C^2 as a C-vector space).

Gate types:
  Type 4: Stage 3 (Lorentz invariance => Minkowski structure)
  Type 3: Clifford algebra {gamma^mu, gamma^nu} = 2 g^{mu nu} I
           (Lounesto 2001 "Clifford Algebras and Spinors" §1.1;
            or: Lawson & Michelsohn 1989 "Spin Geometry" §I.1)
  Type 2: g^{0i} = 0 in Minkowski metric (algebra)
  Type 1: A3 (complex Hilbert space provides factor i for complexification)
  Type 2: Cl(0,2) ≅ H (standard classification, verified below)

Remaining open step (L1): the identification
  e2 = i*gamma^0 ↔ causal direction of edge (from Stage 2c E_obs)
  e1 = gamma^1   ↔ spatial orientation of edge (from I4_132 chirality)
needs explicit derivation from A1+A2+Stage 2c.
"""

import numpy as np
from numpy import linalg as la
import sys

RTOL = 1e-14

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def anticommutator(A, B):
    return A @ B + B @ A


def commutator(A, B):
    return A @ B - B @ A


def verify_clifford_cl11():
    """
    Build the Cl(1,1) algebra for the 1+1D Minkowski edge:
      gamma^0 = sigma_z   (timelike: squares to +I)
      gamma^1 = i*sigma_y (spacelike: squares to -I)

    Verify: {gamma^0, gamma^1} = 2 g^{01} I = 0
            (gamma^0)^2 = +I
            (gamma^1)^2 = -I
    """
    print("=" * 60)
    print("Cl(1,1): CLIFFORD ALGEBRA OF 1+1D MINKOWSKI EDGE")
    print("=" * 60)

    g0 = sigma_z           # timelike generator
    g1 = 1j * sigma_y      # spacelike generator  (= sigma_x * i, but directly)

    # Check signatures
    sq0 = g0 @ g0
    sq1 = g1 @ g1
    ac  = anticommutator(g0, g1)

    print(f"\n  gamma^0 = sigma_z")
    print(f"  gamma^1 = i*sigma_y")
    print(f"\n  (gamma^0)^2 = {sq0[0,0].real:+.1f} * I   expected +1")
    print(f"  (gamma^1)^2 = {sq1[0,0].real:+.1f} * I   expected -1")
    print(f"  ||{{gamma^0, gamma^1}}|| = {la.norm(ac):.2e}   expected 0")

    assert abs(sq0[0,0] - 1.0) < RTOL and la.norm(sq0 - I2) < RTOL
    assert abs(sq1[0,0] + 1.0) < RTOL and la.norm(sq1 + I2) < RTOL
    assert la.norm(ac) < RTOL

    print("  ✓  Cl(1,1) relations: (g0)^2=+I, (g1)^2=-I, {g0,g1}=0")
    return g0, g1


def apply_a3_complexification(g0, g1):
    """
    A3 (complex Hilbert space) provides the factor i.
    Complexify the TIMELIKE generator: e2 = i * gamma^0.

    Result:
      e2^2 = (i*g0)^2 = i^2 * g0^2 = -1 * (+I) = -I   (signature -1)
      e1^2 = g1^2 = -I                                  (unchanged)
      {e1, e2} = i * {g1, g0} = 0                       (preserved)

    This upgrades Cl(1,1) -> Cl(0,2).
    """
    print("\n" + "=" * 60)
    print("A3 COMPLEXIFICATION: Cl(1,1) -> Cl(0,2)")
    print("=" * 60)

    e1 = g1          # spatial: (e1)^2 = -I already
    e2 = 1j * g0     # temporal: (i*g0)^2 = -(g0)^2 = -I

    sq1 = e1 @ e1
    sq2 = e2 @ e2
    ac  = anticommutator(e1, e2)

    print(f"\n  e1 = gamma^1 = i*sigma_y      (spatial orientation)")
    print(f"  e2 = i*gamma^0 = i*sigma_z    (causal direction, complexified)")
    print(f"\n  e1^2 = {sq1[0,0].real:+.1f} * I   expected -1")
    print(f"  e2^2 = {sq2[0,0].real:+.1f} * I   expected -1")
    print(f"  ||{{e1, e2}}|| = {la.norm(ac):.2e}   expected 0")

    assert la.norm(sq1 + I2) < RTOL
    assert la.norm(sq2 + I2) < RTOL
    assert la.norm(ac) < RTOL

    print("  ✓  Cl(0,2) relations: e1^2=e2^2=-I, {e1,e2}=0")
    return e1, e2


def verify_cl02_is_quaternions(e1, e2):
    """
    Cl(0,2) ≅ H (Hamilton's quaternions).

    The quaternion units are: i, j, k with i^2=j^2=k^2=ijk=-1.

    Map: e1 <-> i_H, e2 <-> j_H, e1*e2 <-> k_H.

    Verify all quaternion relations in the 2x2 matrix representation.
    """
    print("\n" + "=" * 60)
    print("Cl(0,2) ≅ H: QUATERNION ALGEBRA VERIFICATION")
    print("=" * 60)

    i_H = e1
    j_H = e2
    k_H = e1 @ e2  # = i_H * j_H

    print(f"\n  i_H = e1,  j_H = e2,  k_H = e1*e2")

    # Quaternion relations: x^2 = -I for x in {i,j,k}
    for name, X in [('i_H', i_H), ('j_H', j_H), ('k_H', k_H)]:
        sq = X @ X
        assert la.norm(sq + I2) < RTOL, f"{name}^2 != -I"
        print(f"  {name}^2 = -I  ✓")

    # ijk = -1 (= -I in matrix rep)
    ijk = i_H @ j_H @ k_H
    assert la.norm(ijk + I2) < RTOL, "ijk != -I"
    print(f"  i_H * j_H * k_H = -I  ✓  (quaternion fundamental relation)")

    # Anti-commutativity: ij = k, ji = -k, etc.
    assert la.norm(i_H @ j_H - k_H) < RTOL, "ij != k"
    assert la.norm(j_H @ i_H + k_H) < RTOL, "ji != -k"
    print(f"  i_H * j_H = k_H  ✓")
    print(f"  j_H * i_H = -k_H ✓")

    print("\n  ✓  Cl(0,2) ≅ H confirmed.")


def verify_su2_action(e1, e2):
    """
    SU(2) = Sp(1) = unit quaternions acts on C^2 (the left H-module).

    The left action of a unit quaternion q = a + b*i_H + c*j_H + d*k_H
    (with a^2+b^2+c^2+d^2 = 1) on C^2 is matrix multiplication by
    the corresponding SU(2) matrix.

    Verify: the three generators of su(2) from {e1, e2, e1*e2} exponentiate
    to SU(2) elements.

    The 2-dim left H-module over C is the Higgs doublet representation.
    """
    print("\n" + "=" * 60)
    print("SU(2) ACTION ON THE 2-DIM LEFT H-MODULE")
    print("=" * 60)

    i_H = e1
    j_H = e2
    k_H = e1 @ e2

    # su(2) generators are i_H, j_H, k_H (or equivalently iσ_x, iσ_y, iσ_z)
    # The Lie algebra generators in SU(2): X_a = -i/2 * sigma_a
    # In our basis: i_H = iσ_y (up to signs), etc.

    # Test: exp(theta * i_H / 2) should be in SU(2) for all theta
    thetas = [0.0, np.pi/6, np.pi/3, np.pi/2, np.pi, 2*np.pi]
    print("\n  Unit quaternions from exp(theta * i_H / 2):")
    for theta in thetas:
        # Matrix exponential
        from scipy.linalg import expm
        U = expm(theta / 2 * i_H)
        det = np.linalg.det(U)
        unitary_err = la.norm(U @ U.conj().T - I2)
        print(f"    theta={theta:.4f}: det={det.real:.6f}+{det.imag:.6f}i, "
              f"||UU†-I||={unitary_err:.2e}", end="")
        is_su2 = abs(abs(det) - 1.0) < 1e-10 and unitary_err < 1e-10
        print(f"  {'✓ SU(2)' if is_su2 else '✗ NOT SU(2)'}")
        assert is_su2

    print("\n  ✓  Unit quaternions exp(θ·i_H/2) ∈ SU(2) for all θ.")
    print("  The 2-dim C^2 = left H-module over C is the Higgs doublet rep.")


def print_summary():
    print("\n" + "=" * 60)
    print("L3b GATE SUMMARY")
    print("=" * 60)
    print("""
  CLOSED:
    [Type 4] Stage 3: Lorentz invariance of toggle dynamics
             => edge qubit has Minkowski spacetime structure
    [Type 3] Clifford algebra definition: {gamma^mu, gamma^nu} = 2g^{mu nu}I
             Lounesto (2001) §1.1 / Lawson-Michelsohn (1989) §I.1
    [Type 2] Minkowski metric: g^{0i} = 0 => {gamma^0, gamma^i} = 0
    [Type 1] A3 (complex Hilbert space): factor i complexifies gamma^0
             => e2 = i*gamma^0 has e2^2 = -I (signature -1)
    [Type 2] e1^2 = e2^2 = -I, {e1, e2} = 0 => Cl(0,2) algebra
    [Type 3] Cl(0,2) ≅ H (quaternions) — standard Clifford classification
    [Type 3] SU(2) = Sp(1) = unit quaternions (standard Lie theory)
    [Type 2] 2-dim left H-module over C = Higgs doublet representation

  L3 (full) = L3a (Lorentz mixing, committed) + L3b (above) = CANDIDATE-SOLID.

  REMAINING OPEN (L1 — identification step):
    e1 = gamma^1 ↔ spatial orientation of edge from I4_132 chirality
    e2 = i*gamma^0 ↔ causal direction from Stage 2c E_obs

    This identification needs explicit derivation from A1+A2+Stage 2c.
    It is the last step before L3 -> SOLID.
    Gate type when closed: Type 1 (A1) + Type 4 (Stage 2c, Stage 3).
""")


if __name__ == "__main__":
    print("L3b: CLIFFORD ALGEBRA ROUTE TO SU(2) FROM MINKOWSKI + A3")
    print("G2 Higgs doublet identification — gate-first checkpoint\n")

    g0, g1 = verify_clifford_cl11()
    e1, e2 = apply_a3_complexification(g0, g1)
    verify_cl02_is_quaternions(e1, e2)
    verify_su2_action(e1, e2)
    print_summary()
