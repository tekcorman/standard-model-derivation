# Theorem G2: Cl(0,2) Boolean Edge Structure and n_channels = 2

**Date:** 2026-04-19
**Status:** STRICT-SOLID under A1 + A3-T + local CAR thm (theorem_car_local_jordan_wigner.md) + n_channels = 2 derivation. ADOPTED-B3 removed 2026-04-21: n_channels=2 is invariant under the (Z/2)^3 B3 chirality convention.
**Script:** `predictions/G2_cl2_channels.py`
**Detailed proof script:** `proofs/foundations/theorem_G2_cl2_channels.py`
**Closes:** BLOCK-2 sub-claims A, B, C of an internal working note.
**Supports:** `predictions/lambda_higgs.py` Step 6 (factor 2 = n_channels; F2-class adoption CLOSED).
**Supersedes:** `../predictions/G2_cl2_channels_derivation.md`.

## Abstract

Let {u,v} be any undirected edge of the srs K_4-quotient. The two directed versions e1=(u,v) and e2=(v,u) carry toggle operators satisfying A1+A4+A3. These generate a Clifford algebra isomorphic to Cl(0,2) over R (equivalently, M_2(C) over C). The minimal faithful complex representation of Cl(0,2)_C has dimension 2. Therefore:

n_channels = dim_C(min. faithful rep of Cl(0,2)_C) = 2  [STRICT-SOLID under A1 + A3-T + local CAR thm]

## Framework axioms invoked

- **(A1)** Binary self-inverse toggle (`docs/framework/framework_axioms.md` §2): T_{e1}^2 = T_{e2}^2 = I (involutions with eigenvalues ±1).
- **(A3-T)** Complex field F=C (derived theorem; see `docs/theorems/theorem_A3_complex_hilbert_from_multiway.md`): allows defining gamma_j = i*T_j so that gamma_j^2 = -I (Clifford generators with signature -1).
- **(local CAR thm)** CAR at k*-valent nodes (derived theorem; see `docs/theorems/theorem_car_local_jordan_wigner.md`): {T_{e1}, T_{e2}} = 0 for two distinct directed edges at the same vertex.

## Derivation

### Step 1 — Boolean involutions from A1

T_{e1}^2 = T_{e2}^2 = I. Each toggle operator is an involution with eigenvalues ±1. These are the "boolean DOF" asserted by BLOCK-2 sub-claim A.

### Step 2 — Anticommutation from A4

Both e1=(u,v) and e2=(v,u) are incident to vertex u. A4 gives {T_{e1}, T_{e2}} = 0 (off-diagonal case of the CAR {gamma_e, gamma_{e'}} = 2*delta_{ee'} * I). Closes sub-claim A and establishes the anticommutation for Clifford structure.

### Step 3 — Cl(0,2) generators from A3

Set gamma_j = i*T_{e_j} (A3 supplies F=C). Then:
- gamma_j^2 = (i*T_j)^2 = -T_j^2 = -I  (signature -1)
- {gamma_1, gamma_2} = i^2 * {T1, T2} = 0  (anticommute)

gamma_1, gamma_2 satisfy the defining relations of Cl(0,2): two generators each squaring to -I and anticommuting. The derived element gamma_12 = gamma_1*gamma_2 satisfies gamma_12^2=-I and anticommutes with both, completing the quaternion algebra i^2=j^2=k^2=-1.

Closes sub-claim B (Clifford signature (0,2) = (-1,-1) in Clifford convention).

### Step 4 — Algebra structure: Cl(0,2)_C isom to M_2(C)

Basis {I, gamma_1, gamma_2, gamma_12} is C-linearly independent (Gram matrix rank 4). Therefore:
- Over R: Cl(0,2) isom to H (quaternion algebra, dim 4 over R). Reference: Porteous 1995 §15.1.
- Over C: Cl(0,2)_C isom to M_2(C) (2x2 complex matrices, dim 4 over C). Reference: Lounesto 2001 §15.3.

### Step 5 — Minimal faithful representation has dimension 2

**Lower bound (dim >= 2):** On C^1, every operator is a scalar. The anticommutation {gamma_1, gamma_2}=0 then forces 2*gamma_1*gamma_2 = 0, so one generator acts as zero — not faithful. Therefore dim >= 2.

**Upper bound (2-dim faithful rep):** Explicit assignment:
- gamma_1 -> i*sigma_x = [[0,i],[i,0]]
- gamma_2 -> i*sigma_z = [[i,0],[0,-i]]

satisfies all Cl(0,2) relations on C^2. The basis {I_2, i*sx, i*sz, i*sx*(i*sz)} has Gram rank 4, confirming faithfulness.

**Conclusion:**

n_channels = dim_C(min. faithful rep of Cl(0,2)_C) = 2  [STRICT-SOLID]

Closes sub-claim C (SU(2) automorphism): Cl(0,2) isom to H over R, and unit quaternions Sp(1) isom SU(2) act on C^2 by left multiplication.

### Step 6 — Fock-space decomposition (supplementary)

The 4-dim fermionic Fock space (where T1, T2 live) is a reducible rep of Cl(0,2)_C. Its commutant has C-dimension 4 (verified: null-space of the commutator map has dim = 16 - rank(constraint system) = 4), confirming C^4 = C^2 + C^2 (two copies of the minimal 2-dim irrep).

## What IS proved (STRICT-SOLID)

- Every undirected edge of srs carries a Cl(0,2) algebra on its directed versions, from A1+A4+A3.
- The minimal faithful complex representation has dimension 2.
- The factor of 2 in lambda_higgs = 2 * (5/3) * (2/3)^8 is STRICT-SOLID, not adopted.
- SU(2) acts naturally on C^2 via Sp(1) isom SU(2) from the Cl(0,2) isom H isomorphism.

## What is NOT proved (requires ADOPTED-B3)

- Identification of C^2 with the SM SU(2)_L Higgs doublet (left-chiral vs right-chiral assignment).
- Hypercharge Y=+1/2 for the Higgs doublet.
- Identification of the abstract SU(2) (from Sp(1)) with the electroweak SU(2)_L specifically.

## Resolution of BLOCK-2 sub-claims

| Sub-claim | Description | Status |
|-----------|-------------|--------|
| A | Boolean DOF existence | CLOSED by Steps 1+2 |
| B | Clifford signature (-1,-1) | CLOSED by Step 3 |
| C | SU(2) automorphism | CLOSED for Cl(0,2): Sp(1) isom SU(2) acts on C^2 |
| D | Hypercharge Y=+1/2 | STILL REQUIRES ADOPTED-B3 |

## Impact on predictions/lambda_higgs.py

Step 6 of lambda_higgs.py was previously `ADOPTED (F2-class): factor 2 = Cl(2)/SU(2)_L dim`.

After G2 + 2026-04-21 closure, Step 6 is fully STRICT-SOLID:
- `STRICT-SOLID under A1 + A3-T + local CAR thm: n_channels = dim(min faithful C-rep of Cl(0,2)_C) = 2`

ADOPTED-B3 removed 2026-04-21: n_channels=2 is an intrinsic algebraic invariant of
Cl(0,2)_C, unchanged under the (Z/2)^3 L↔R convention choices of B3. λ uses
n_channels only as a count, so λ = 2560/19683 is convention-independent.

The F2-class adoption is CLOSED. ADOPTED-B3 is CLOSED. dark-map Class 2 assignment
CLOSED 2026-04-28 via `docs/theorems/theorem_dark_map_class2_closure.md` Theorem 5.1.
lambda_higgs.py overall verdict graduates to UNIQUE-THEOREM-GRADE (Row P41 of
`docs/parameters/parameter_uniqueness_ledger.md`).

## References

- A1, A3, A4: `docs/framework/framework_axioms.md`.
- BLOCK-2 definition: an internal working note §2.
- Cl(0,2) isom H over R: Porteous (1995) *Clifford Algebras and the Classical Groups*, §15.1.
- Cl(0,2)_C isom M_2(C): Lounesto (2001) *Clifford Algebras and Spinors*, §15.3.
- Sp(1) isom SU(2): Brocker-tom Dieck (1985) *Representations of Compact Lie Groups*, §I.2.
- Jordan-Wigner (1928): canonical anticommutation in Fock space.
- ADOPTED-B3: `docs/audits/registers/adoption_register.md`, an internal Sprint 9 kickoff doc.

## Files referenced

- `predictions/lambda_higgs.py` — downstream: n_channels=2 used in Step 6.
- `proofs/foundations/theorem_G2_cl2_channels.py` — detailed 6-part numerical proof.
