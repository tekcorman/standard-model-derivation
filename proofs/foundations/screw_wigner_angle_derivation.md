# Derivation: 4_1 Screw-C_3 Dihedral Angle and Wigner Structure

**Status:** theorem (geometric identity and Wigner computation are theorem-grade;
identification HM = delta_Koide is OPEN — see Section 5).
**Verification:** `predictions/screw_wigner_angle.py` (all assertions pass).
**Upstream:** `predictions/k_star.py` (k* = 3), `predictions/g_girth.py` (g = 10),
`docs/theorem_B5_3_core.md` (C_3 body-diagonal site symmetry of srs).

## Theorem Statement

**(a)** cos(beta) = u_C3 . R_4 u_C3 = 1/3 = 1/k*  [CAS-verified]

**(b)** cos(beta) = 1/k* holds at k* = 3 and only k* = 3.
For a k*-regular lattice: cos(beta) = (k*-2)/k*, which equals 1/k* iff k* = 3.

**(c)** Wigner d^1 diagonal elements at tilt angle beta = arccos(1/k*):

    d^1_{+/-1,+/-1} = (1 + cos beta)/2 = 2/3
    d^1_{00}         = cos beta         = 1/3

Survival probabilities: P_{+/-1} = 4/9, P_0 = 1/9.

**(d)** HM(4/9, 1/9, 4/9) = 2/9  (exact algebra).

## Proof of Part (a): Geometric Identity

R_4 = [[0,-1,0],[1,0,0],[0,0,1]] (90-degree rotation about [001]).
u_C3 = [1,1,1]/sqrt(3) (body-diagonal C_3 axis).

    R_4 . [1,1,1]/sqrt(3) = [-1,1,1]/sqrt(3)

    cos(beta) = [1,1,1].[-1,1,1] / (sqrt(3) * sqrt(3)) = 1/3

CAS-verified: numpy dot product gives 0.333...333 = 1/3. QED.

## Proof of Part (b): Uniqueness

For a k*-regular lattice with C_3 site symmetry along [111] and 4_1 screw along [001]:

    cos(beta) = (k* - 2) / k*

Setting cos(beta) = 1/k*: (k*-2)/k* = 1/k* => k* - 2 = 1 => k* = 3.
Verified for k* in {2,...,9}: only k* = 3 satisfies the identity. QED.

## Proof of Parts (c) and (d): Wigner Computation

Using exact Fraction arithmetic:

    cos_b = Fraction(1, 3)
    d1_pm1 = (1 + cos_b) / 2 = Fraction(2, 3)
    d1_00  = cos_b           = Fraction(1, 3)
    P_pm1  = d1_pm1^2        = Fraction(4, 9)
    P_0    = d1_00^2         = Fraction(1, 9)
    HM     = 3 / (1/P_pm1 + 1/P_0 + 1/P_pm1)
           = 3 / (9/4 + 9 + 9/4)
           = 3 / (27/2)
           = Fraction(2, 9)

Reference: Edmonds 1957, Angular Momentum in Quantum Mechanics, Ch. 4, Eq. 4.1.15.
QED Parts (c) and (d).

## What This Theorem Derives and What It Does Not

### Derived (theorem-grade)

- cos(beta) = 1/k* = 1/3: structural fact about the MDL-optimal lattice.
- Uniqueness: holds at k* = 3 only.
- Wigner structure {4/9, 1/9, 4/9}: forced by cos(beta) = 1/k* + standard d^1 formula.
- HM(4/9, 1/9, 4/9) = 2/9: pure algebra.

### NOT Derived (OPEN gap)

**The identification HM = delta_Koide is not derived.**

delta = 2/9 appears ONLY as the harmonic mean of the survival probabilities.
The identification requires a derivation connecting the 4_1 screw mixing to the
Koide propagator structure. Two candidate closure routes:

- **Route A (Dyson path):** d^1_{10}(beta)/k* = sin(beta)/(sqrt(2)*k*) = 2/9;
  two-vertex Dyson self-energy at P-point gives delta^2 as VEV coupling.
  Requires: per-edge screw coupling = Wigner off-diagonal element / k*.
- **Route B (A3 path):** MDL-optimal description of three survival rates under
  inverse-rate (propagator pole residue) observable forces HM via A2+A3.
  Requires: closing T_mass gap first.

## Rigor Audit

| Step | Claim | Source | Status |
|------|-------|--------|--------|
| (a) | R_4.[111] = [-111], cos(beta) = 1/3 | CAS: numpy dot product | PASS |
| (b) | cos(beta) = 1/k* unique to k* = 3 | Algebra: (k*-2)/k* = 1/k* iff k*=3 | PASS |
| (c) | Wigner d^1 elements at cos(beta) = 1/3 | Edmonds 1957 Eq. 4.1.15 | PASS |
| (c) | P_{+/-1} = 4/9, P_0 = 1/9 | CAS: Fraction arithmetic | PASS |
| (d) | HM(4/9, 1/9, 4/9) = 2/9 | CAS: Fraction arithmetic | PASS |
| Gap | HM = delta_Koide | NOT derived | OPEN |

## References

- Edmonds, A.R. (1957). Angular Momentum in Quantum Mechanics. Princeton University Press.
  Ch. 4, Eq. 4.1.15.
- International Tables for Crystallography (2016). Space group I4_132 (#214), 4_1 screw.
- predictions/k_star.py — k* = 3 (MDL-derived).
- docs/theorem_B5_3_core.md — C_3 body-diagonal site symmetry of srs.
- proofs/foundations/delta_dynamical.py — 10-approach numerical verification.
- proofs/masses/srs_delta_sq_theorem.py — Dyson two-vertex path; delta^2 derivation.
