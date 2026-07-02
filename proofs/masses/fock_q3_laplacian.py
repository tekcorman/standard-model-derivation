#!/usr/bin/env python3
"""
proofs/masses/fock_q3_laplacian.py

Analytic + CAS proof: |σ(0)|/|σ(1)| = 3 from the MDL compression
potential on the Fock hypercube Q_{k*}.

Gate type: Type 1+2 (A1+A2 axioms + explicit algebra; no external inputs).

DERIVATION
──────────
Step 1 [Type 1, A1]: k* = 3 toggle modes on each srs vertex.
  Fock states: {0,1}^3, 8 vertices connected by Hamming-distance-1 edges.

Step 2 [Type 1+2, A2]: MDL description length under a two-part code
  (sector index + position within sector, uniform prior within sector):
    DL(k) = log₂(k*+1) + log₂(C(k*,k))

  For k*=3:
    DL(0) = log₂4 + log₂1 = 2             (1 state in sector)
    DL(1) = log₂4 + log₂3 = 2 + log₂3    (3 states in sector)
    DL(2) = log₂4 + log₂3 = 2 + log₂3    (3 states — C(3,2)=C(3,1)=3)
    DL(3) = log₂4 + log₂1 = 2             (1 state in sector)

  Compression potential: φ(k) = -DL(k).

Step 3 [Type 2, algebra]: Sector Laplacian of φ on Q_{k*}:
  A state at occupation k has (k*-k) neighbors at level k+1 and k neighbors
  at level k-1, so:
    σ(k) = k*·φ(k) − (k*−k)·φ(k+1) − k·φ(k−1)

  For k=0: σ(0) = k*·(φ(0)−φ(1)) = k*·log₂(k*) = 3·log₂3
  For k=1: σ(1) = k*·φ(1) − (k*-1)·φ(2) − φ(0)
                = (k*·φ(1) − (k*-1)·φ(2)) − φ(0)
    Since φ(1)=φ(2) for k*=3 [C(3,1)=C(3,2)=3]:
                = φ(1) − φ(0) = −log₂3

  Ratio: |σ(0)|/|σ(1)| = 3·log₂3 / log₂3 = 3   (log₂3 cancels exactly)

The ratio = k* = 3 is algebraically exact. It follows from C(k*,1) = k*
and the symmetry C(k*,0) = C(k*,k*) = 1. The numerical value of log₂3
is irrelevant — it cancels.

SIGNIFICANCE: for k*=3 this equals the Georgi-Jarlskog ratio (ratio of
lepton to down-quark mass scales at the GUT scale, first generation).
The GJ ratio = k* follows from A1+A2 alone, with zero free parameters.

OPEN GAP: Connecting σ(k) to physical particle masses requires the T_mass
identification (Need-A of an internal working note). This
proof establishes the ratio from first principles; the absolute scale
requires A5(a).
"""

import math
import sys
from fractions import Fraction
from math import comb

# ────────────────────────────────────────────────────────────────────────
# Step 1 — Fock hypercube setup  [Type 1, A1]
# ────────────────────────────────────────────────────────────────────────

n = 3   # k* = 3 (from predictions/k_star.py; MDL-optimal degree in d=3)

C = [Fraction(comb(n, k)) for k in range(n + 1)]
assert C == [Fraction(1), Fraction(3), Fraction(3), Fraction(1)]

print("=" * 62)
print("  Fock Q₃ Laplacian — Georgi-Jarlskog ratio from A1+A2")
print("=" * 62)
print(f"\n  k* = n = {n},  Fock states: {{0,1}}^{n},  C(n,k) = {[int(c) for c in C]}")

# ────────────────────────────────────────────────────────────────────────
# Step 2 — MDL compression potential  [Type 1+2, A2]
# Represent each DL(k) as (a, b) meaning  a + b·log₂3  with a,b ∈ ℚ.
# log₂(n+1) = log₂4 = 2 exactly;  log₂(C(n,k)) is 0 or log₂3.
# ────────────────────────────────────────────────────────────────────────

assert abs(math.log2(n + 1) - 2.0) < 1e-15   # log₂4 = 2 exactly

DL_sym = []   # (integer_part, log2_3_coefficient) as Fraction pairs
for k in range(n + 1):
    int_part = Fraction(2)           # log₂(n+1) = log₂4 = 2
    log3_coeff = Fraction(0) if C[k] == 1 else Fraction(1)  # log₂(C(n,k)) = 0 or log₂3
    DL_sym.append((int_part, log3_coeff))

phi_sym = [(-a, -b) for (a, b) in DL_sym]   # φ(k) = -DL(k)

print(f"\n  DL(k) = integer_part + log₂3_coeff·log₂3:")
for k in range(n + 1):
    a, b = DL_sym[k]
    print(f"    DL({k}) = {a} + {b}·log₂3")

# ────────────────────────────────────────────────────────────────────────
# Step 3 — Sector Laplacian  [Type 2, algebra]
# σ(k) = n·φ(k) − (n−k)·φ(k+1) − k·φ(k−1)
# ────────────────────────────────────────────────────────────────────────

def laplacian(phi, k, n_modes):
    a, b = n_modes * phi[k][0], n_modes * phi[k][1]
    if k + 1 <= n_modes:
        a -= (n_modes - k) * phi[k + 1][0]
        b -= (n_modes - k) * phi[k + 1][1]
    if k - 1 >= 0:
        a -= k * phi[k - 1][0]
        b -= k * phi[k - 1][1]
    return (a, b)

sigma_sym = [laplacian(phi_sym, k, n) for k in range(n + 1)]

print(f"\n  σ(k) = integer_part + log₂3_coeff·log₂3:")
for k in range(n + 1):
    a, b = sigma_sym[k]
    print(f"    σ({k}) = {a} + {b}·log₂3")

# ────────────────────────────────────────────────────────────────────────
# Step 4 — Verify ratio  [Type 2, algebra]
# ────────────────────────────────────────────────────────────────────────

a0, b0 = sigma_sym[0]
a1, b1 = sigma_sym[1]

# Integer parts must vanish (log₂3 should not cancel with integer parts)
assert a0 == 0, f"σ(0) has integer part {a0}; expected 0"
assert a1 == 0, f"σ(1) has integer part {a1}; expected 0"

ratio = abs(b0) / abs(b1)

print(f"\n  log₂3 cancels in the ratio:")
print(f"    σ(0) = {b0}·log₂3")
print(f"    σ(1) = {b1}·log₂3")
print(f"\n  |σ(0)|/|σ(1)| = |{b0}|/|{b1}| = {ratio}")

assert ratio == Fraction(3), f"Expected ratio 3, got {ratio}"
print(f"\n  ✓ |σ(0)|/|σ(1)| = {ratio} = k* = {n}  (exact; log₂3 cancels)")

# ────────────────────────────────────────────────────────────────────────
# Step 5 — C-symmetry and full sector table  [Type 2]
# ────────────────────────────────────────────────────────────────────────

print(f"\n  Full sector table:")
log2_3 = math.log2(3)
for k in range(n + 1):
    a, b = sigma_sym[k]
    assert a == 0
    sym_k, sym_nk = sigma_sym[k], sigma_sym[n - k]
    assert sym_k == sym_nk, f"C-symmetry fails at k={k}"
    print(f"    σ({k}) = {b:+}·log₂3 = {float(b) * log2_3:+.6f}  [C-sym: σ({k})=σ({n-k}) ✓]")

# ────────────────────────────────────────────────────────────────────────
# Step 6 — Numerical cross-check  [Type 2]
# ────────────────────────────────────────────────────────────────────────

sigma_float = [float(a + b * log2_3) for (a, b) in sigma_sym]
ratio_num = abs(sigma_float[0]) / abs(sigma_float[1])
assert abs(ratio_num - 3.0) < 1e-13, f"Numerical ratio = {ratio_num}"
print(f"\n  Numerical cross-check: |σ(0)|/|σ(1)| = {ratio_num:.15f}  ✓")

# ────────────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────────────

print(f"\n" + "=" * 62)
print(f"  RESULT: |σ(0)|/|σ(1)| = {ratio}  (exact, log₂3 cancels)")
print(f"  = k* = {n} = Georgi-Jarlskog ratio")
print(f"")
print(f"  Proof depends only on C(3,1) = 3 and C(3,0) = C(3,3) = 1.")
print(f"  Gate: Type 1 (A1+A2) + Type 2 (algebra).  No free parameters.")
print(f"  Open gap: T_mass identification → Need-A (mass_operator_scoping.md).")
print("=" * 62)
