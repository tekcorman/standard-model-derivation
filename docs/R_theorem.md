# The Neutrino Mass Splitting Ratio R = 228/7

## Result

The ratio of neutrino mass-squared splittings is

    R = Dm^2_31 / Dm^2_21 = 228/7 = 32.5714...

Observed (PDG 2024): R = 32.576. Match: 0.015%. Zero free parameters.

## The formula

    R = 2/sin^2(5*phi) - 4

where phi = arctan(sqrt(7)) is the Ihara phase of the K4 triplet sector.

Equivalently:

    R = (k*-1)^(g-2) / |D_Gamma| - (k*+1) = 256/7 - 4

where k*=3 is the coordination number, g=10 is the girth, and
|D_Gamma| = 4(k*-1)-1 = 7 is the Ihara discriminant.

## The derivation

**Step 1.** The srs crystal net has quotient graph K4, which is 3-regular
with adjacency eigenvalues {3, -1, -1, -1}. The triplet eigenvalue
lambda = -1 gives Ihara zeta poles at u = (-1 +/- i*sqrt(7))/4.
The Ihara phase is phi = arctan(sqrt(7)), with cos(phi) = 1/sqrt(8).

**Step 2.** The Chebyshev propagator at distance n is G_n = sin(n*phi)/sin(phi).
For q = k*-1 = 2, the cubic identity q^3 = 5q - 2 has q = 2 as its unique
positive integer root. This identity is equivalent to G_5 = -1/(k*+1) = -1/4.
The distance n = 5 is algebraically selected by the graph structure.

**Step 3.** From sin(5*phi) = G_5 * sin(phi) = (-1/4)*sqrt(7/8), we get
sin^2(5*phi) = 7/128 = |D_Gamma|/2^(g-3). This is exact algebra
(Chebyshev polynomial T_10(1/sqrt(8)) = 57/64).

**Step 4.** R = 2/sin^2(5*phi) - (k*+1) = 256/7 - 4 = 228/7.

Every step uses only k* = 3 (from MDL) and the resulting graph invariants.

## Why n = 5

The cubic identity q^3 - 5q + 2 = 0 factors as (q-2)(q^2+2q-1) = 0.
The unique positive integer root is q = 2, corresponding to k* = 3.
For any other coordination number, n = 5 does not produce a clean
propagator value. The number 5 is not g/2 by coincidence — it is
selected by the algebraic structure of K4.

## Physical interpretation

The neutrino is delocalized (the |000> Fock state). Its mass matrix
is determined by the spectral structure of K4, the quotient graph.

Each of the three generations propagates on K4 with a different
effective momentum, shifted by 2*pi/3 from the Z3 generation symmetry.
The Ihara phase phi is the momentum per step in the triplet sector.
After n = 5 steps, the propagator hits the special value |G_5| = 1/4,
where K4's spectral structure maximally distinguishes the generations.

R is the anisotropy of this propagator across the three Z3 channels:
- Total propagator intensity: 2/sin^2(5*phi) = 256/7
- Isotropic background: k*+1 = 4 (one per K4 vertex)
- Splitting: R = anisotropy - background = 228/7

## Why no dark correction

R is a topological invariant of K4. The Ihara phase arctan(sqrt(7)) is
exact (from the eigenvalue lambda = -1). The distance n = 5 is exact
(from the cubic identity). sin^2(5*phi) = 7/128 is exact (Chebyshev).

Quantities that need dark corrections (theta_23, v) involve dynamics
on the srs lattice, where the dark sector perturbs the evolution.
R involves only the topology of the quotient graph K4, which the dark
sector does not modify. This explains the 0.015% accuracy without
any correction.

## Verification

    python3 proofs/flavor/srs_r_physical_derivation.py
    python3 proofs/flavor/srs_r_theorem.py

## Key identity

    cos(10 * arctan(sqrt(7))) = 57/64

This is the Chebyshev polynomial T_10 evaluated at 1/sqrt(8).
The numerator 57 = 64 - 7 = 2^6 - |D_Gamma|.
The denominator 64 = 2^6.
R = 4 * 57 / 7 = 228/7.
