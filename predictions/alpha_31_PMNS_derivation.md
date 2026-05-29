# Derivation of α_31_PMNS (second Majorana phase)

**Date:** 2026-05-04 EOD+1; **re-graded 2026-05-12.**
**Status:** STRUCTURAL-DERIVATION-CONDITIONAL on ADOPTED-NU-MAJ-PHASE (+ C³_gen-L3 mass-ordering + ADOPTED-B3). (Was labelled UNIQUE-THEOREM-GRADE-CONDITIONAL 2026-05-04 EOD+1 — inflated.)

## Abstract

α_31 = 2g·arg(h) mod 360° ≈ 324.775° — equivalently arg((h_ω/h_ω²)^g) mod 360° — the second non-trivial C_3 generation channel of the same Pati-Salam seesaw + Takagi chain as α_21. See `alpha_21_PMNS_derivation.md` for the full structural derivation and the discussion of the load-bearing identification.

## Framework axioms / inputs invoked

Same as α_21 (see `alpha_21_PMNS_derivation.md` §"Framework axioms / inputs invoked"): h, g [theorem-grade]; 3 generations from C³_gen [L2 theorem-grade]; the M_R phase factor h_m^g [IDENTIFICATION — ADOPTED-NU-MAJ-PHASE, not derived]; the phase-free |M_R| [theorem-grade-conditional, not part of this conditional]; PS seesaw [Type 3]; Takagi [Type 2]; ADOPTED-B3.

## Derivation

Same Steps 1–5 as α_21 (`alpha_21_PMNS_derivation.md`), with the second non-trivial C_3 channel: h_ω² = (−√3+i√5)/2, arg(h_ω²^g) ≈ 197.61° (this is δ_CP, RETIRED on independent grounds), and the seesaw + NuFIT ordering gives
$$\alpha_{31} \;=\; \phi_3 - \phi_1 \;=\; \arg\!\big((h_\omega/h_{\omega^2})^g\big)\ \bmod 360^\circ \;=\; 2g\cdot\arg(h)\ \bmod 360^\circ.$$

## Result

$$\boxed{\;\alpha_{31} \;=\; 2g\cdot\arg(h)\ \bmod 360^\circ \;=\; 324.775^\circ\;}$$
under ADOPTED-NU-MAJ-PHASE.

## Comparison with experiment

Same as α_21 — unconstrained by current data; 0νββ would constrain the α_21 + α_31 combination as a future test. **Clause 8: vacuously PASS.** Not falsified; identification-conditional.

## Open questions

Same as α_21 (`alpha_21_PMNS_derivation.md` §6): the M_R phase factor h^g is the ADOPTED-NU-MAJ-PHASE identification, not derived — a discharge was attempted and failed (`proofs/foundations/majorana_M_R_waterfilling.py`): the A2-T loop-sum route diverges (Ramanujan saturation ⇒ no finite cutoff), and the Path-B "cardinality-k ↔ k girth rings" route is broken (K_4 cycle-space generators have nonzero Z³ voltage ⇒ don't lift to srs cycles). The m_ν₂/m_ν₃ magnitude rows ride on the phase-free |M_R| and are unaffected.

**Partial graduation 2026-05-14 (Probe B).** Per α_21 §6 partial-graduation note, the ω ↔ +Re(h), ω² ↔ −Re(h) sign-lock is now theorem-grade (Probe B verdict §5, an internal working note). This downgrades the Step-2 eigenvalue assignment from "chirality convention picking one of four ±h, ±h̄" to "residual ±Im pick within Re-sign-locked block" (one bit of convention, not two). The α_31 = 2g·arg(h) formula and the load-bearing ADOPTED-NU-MAJ-PHASE smuggle (h_m^g *exponent*) are NOT affected. Honest grade remains STRUCTURAL-DERIVATION-CONDITIONAL.

## References

See `alpha_21_PMNS_derivation.md` for the full chain and the discharge-attempt analysis (`proofs/foundations/majorana_M_R_waterfilling.py`). `docs/audits/registers/adoption_register.md` — ADOPTED-NU-MAJ-PHASE. Row P36 in `docs/parameters/parameter_uniqueness_ledger.md`.
