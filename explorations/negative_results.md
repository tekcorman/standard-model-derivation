# Negative Results and Killed Hypotheses

Science is as much about what does NOT work as what does.
These are approaches that were rigorously tested and definitively ruled out.

---

## Failed Mechanisms for alpha\_21 (Majorana CP phase)

The correct mechanism was eventually found (Hashimoto eigenvalue: alpha\_21 = arg(h^g) = 162.39 deg),
but only after systematically eliminating these alternatives:

### 1. Fock space (g-1)/g
Tried: S^g = -C3 on girth-10 cycle accumulates phase (g-1)/g pi.
Failed: The |000> state gets phase pi, same as localized states. Fock vacuum is a C3 eigenstate.
Lesson: Fock vacuum cannot distinguish girth cycles from trivial cycles.

### 2. Naive K4 seesaw
Tried: [P\_D, P\_R] != 0 on K4 quotient.
Failed: P\_D and P\_R COMMUTE because both are functions of J (generation permutation matrix).
Lesson: K4 is too symmetric. S3 ensures all generation-mixing structures commute.

### 3. Berry phase
Tried: Berry phase of triplet band on srs Bloch Hamiltonian.
Failed: Not topologically protected. Singlet band gives -161.64 deg (tantalizingly close), but path-dependent.
Lesson: srs bands are not topologically nontrivial in the relevant sector.

### 4. K4 sign pattern search
Tried: All 64 sign patterns of arccos(1/3) on K4 edges.
Failed: alpha\_31 = 0 across ALL 64 patterns. K4 mass matrices are real-orthogonally diagonalizable.
Lesson: K4 cannot produce nonzero Majorana phases regardless of phase assignment.

### 5. Mixed-base seesaw
Tried: Different bases (exp(-lambda\_1), 1/sqrt(2)) for Dirac vs Majorana.
Failed: All delocalized bases overshoot the mass ratio.
Lesson: (2/3) is the universal base. Spectral quantities enter as corrections, not replacements.

### 6. Seesaw CP from Ihara circulant
Tried: delta\_CP and alpha\_21 from seesaw using Ihara zeta on K4.
Failed: alpha\_21 = 145.8 deg (16 deg off). delta\_CP requires the charged lepton diagonalizer U\_l.
Lesson: CP phases arise from the INTERFACE between edge-local and delocalized regimes.

---

## Failed CKM Approaches

### 7. CKM from K4 Hashimoto
Tried: Hashimoto resolvent on K4 quotient for off-diagonal CKM elements.
Failed: S3 symmetry makes ALL off-diagonal elements equal. G(i,j) = G(i,k).
Lesson: K4 is too symmetric. Need full srs lattice to distinguish V\_us / V\_cb / V\_ub.
Script: `srs_ckm_from_c3.py`

### 8. CKM from NB amplitude matrix
Tried: Non-backtracking matrix elements as CKM amplitudes.
Failed: NB matrix is not unitary in the required sense.
Script: `srs_ckm_amplitude.py`

---

## Failed Cosmological Approaches

### 9. RPV washout for eta\_B
Tried: R-parity violating washout of baryon asymmetry.
Failed: Washout is too strong -- erases the asymmetry entirely.
Script: `srs_eta_b_rpv_washout.py`

### 10. Coherent sum diagram for eta\_B
Tried: sqrt(k\*) coherent sum of baryogenesis diagrams.
Failed: Coherent sum does not give the correct prefactor.
Script: `srs_eta_b_diagram.py`

### 11. Ramanujan upgrade for eta\_B
Tried: Ramanujan structure upgrades the baryogenesis calculation.
Result: Partial -- upgrades the gravitino calculation, not eta\_B directly.
Script: `srs_eta_b_ramanujan.py`

---

## Failed Mixing Angle Approaches

### 12. Wigner D1 as U\_l matrix
Tried: Use Wigner D1 rotation as the charged lepton diagonalizer.
Failed: D1 enters at the PARAMETER level (delta = 2/9), not the MATRIX level.
Destroys TBM structure when used as U\_l.
Lesson: Don't confuse parameter-level results with matrix-level constructions.

### 13. Band crossing for alpha\_21
Tried: omega band crossing on Gamma-P line.
Failed: No actual band crossing occurs; bands anticross.
Script: `srs_alpha21_crossing.py`

### 14. Koide U\_l for PMNS
Tried: Koide diagonalizer as U\_l in U\_PMNS = U\_l^dag U\_nu.
Failed: Koide U\_l is wrong (tested explicitly). Pati-Salam route works.
Script: `srs_ul_vckm.py`

### 15. Sum rules for theta\_12
Tried: Various quark-lepton complementarity sum rules.
Failed: Sum rules give worse results than spherical Pythagorean theorem.
Script: `srs_theta12_sumrule.py`

### 16. Dark correction for theta\_13
Tried: Apply delocalized dark correction (discriminant/k\*) to theta\_13.
Failed: Makes theta\_13 WORSE. Edge-local dark absorption (1-alpha\_1) is the correct mechanism.
Script: `srs_theta13_dark_consistent.py`

---

## Key Lessons

1. **K4 is too symmetric** for CKM, Majorana phases, and alpha\_21.
   The full srs Bloch Hamiltonian is needed whenever S3 breaking matters.

2. **(2/3) is universal.** Every attempt to replace the base with spectral
   quantities (exp(-lambda\_1), 1/sqrt(2)) failed. The base is (k-1)/k;
   spectral quantities enter as corrections to the exponent.

3. **Edge-local vs delocalized is fundamental.** The dark correction has
   opposite signs for the two regimes. Mixing them always fails.

4. **CP phases require the interface.** Neither the neutrino sector alone
   (delocalized) nor the charged lepton sector alone (edge-local) suffices.
   The CP phases emerge from the mismatch between the two.
