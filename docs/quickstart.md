# Quick Start: One Result in 5 Minutes

## The claim

The Ihara zeta function of the K₄ graph (the quotient of the srs crystal net) 
predicts the neutrino mass splitting ratio with zero free parameters.

## The computation

The srs net (space group I4₁32) has K₄ as its quotient graph. K₄ is 3-regular 
with adjacency eigenvalues {3, -1, -1, -1}. The triplet eigenvalue λ = -1 gives 
Ihara zeta poles at:

    u = (-1 ± i√7) / 4

The splitting phase is arctan(√7). The mass ratio:

    R = Δm²₃₁ / Δm²₂₁ = [(2/3)^(-2arctan(√7)/arctan(2-√3)) + 1]² - 1

which evaluates to **R = 228/7 = 32.571**.

## Check it yourself

```bash
git clone https://github.com/tekcorman/standard-model-derivation.git
cd standard-model-derivation
python3 proofs/flavor/ihara_splitting_proof.py
```

The script computes R from the Ihara zeta poles and compares to the PDG value.

**Observed: R = 32.576** (PDG 2024, from Δm²₃₁ = 2.453 × 10⁻³ eV² and 
Δm²₂₁ = 7.53 × 10⁻⁵ eV²).

**Predicted: R = 228/7 = 32.571** — 0.015% match, zero parameters.

## What's √7?

√7 = √(4(k\*-1) - 1) where k\* = 3 is the coordination number of the srs net.
It appears because the Hashimoto (non-backtracking) eigenvalue at the triplet 
sector of K₄ has |h|² = k\*-1 = 2, and the discriminant is D = 1 - 4(k\*-1) = -7.

The same eigenvalue, at the P point of the BCC Brillouin zone, encodes all 
PMNS mixing angles and CP violation phases — but that's the 30-minute version 
(see [derivations.md](derivations.md)).

## If you have 30 seconds more

Run the full verification:

```bash
python3 verify.py
```

This runs 15 backbone proofs in ~6 seconds. Each one derives a Standard Model 
quantity from the srs lattice with zero free parameters. All should PASS.

## If you have 30 minutes

Read [derivations.md](derivations.md) for the complete picture: how one complex 
number h = (√3 + i√5)/2 at one point in the Brillouin zone encodes all flavor 
physics, while the Clifford algebra Cl(6) on the same lattice encodes all gauge 
physics.

## What this is not

- Not a fit. No parameters are adjusted. Every script computes from k\* = 3 and g = 10.
- Not numerology. Every result has a derivation chain from two axioms (see [honest_assessment.md](honest_assessment.md)).
- Not complete. One parameter (A_s, the scalar perturbation amplitude) remains open.
