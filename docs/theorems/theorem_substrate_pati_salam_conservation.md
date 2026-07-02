# Substrate Pati-Salam Conservation — Spin(4) × Spin(2) corollary

**Date:** 2026-04-29.
**Status:** **THEOREM-GRADE for the abstract symmetry; APPLICATION-CONDITIONAL on ADOPTED-B3 / ADOPTED-PS-SCALE for specific physical labelings.** Corollary of Theorem 1 in `theorem_substrate_symmetry_to_martingale.md`. Phase 1c deliverable of the symmetry-shortcut program. Establishes the substrate's analog of conservation under the Pati-Salam embedding.

**Predecessors:**
- `theorem_substrate_symmetry_to_martingale.md` (engine theorem; this corollary's parent).
- `docs/theorem_B3_spinor_fermion.py` and paired derivation (Cl(6,0) → Spin(4) × Spin(2) decomposition; dimensionally forced).
- `../audits/registers/adoption_register.md` ADOPTED-B3 (Pati-Salam labeling) and ADOPTED-PS-SCALE (neutrino bare scale).

**Sibling deliverables:**
- `theorem_substrate_momentum_conservation.md` (Phase 1a).
- `theorem_substrate_generation_charge_conservation.md` (Phase 1b).

---

## Question

The Phase 0 engine reduces "prove conservation under symmetry G" to verifying (H1) prior G-invariance, (H2) filtration G-equivariance, (H3) functional G-invariance. This document executes the checklist for:

**G = Spin(4) × Spin(2) ⊂ Spin(6), the Pati-Salam embedding** acting on the substrate's Cl(6,0) spinor module (B3).

Two distinct claims are at stake:

1. **Abstract claim.** Does the abstract group action of Spin(4) × Spin(2) on the substrate's Cl(6,0) module support (H1)–(H3)? *This is independent of physical labeling.*
2. **Applied claim.** Once (H1)–(H3) hold, can specific physical observables (lepton/quark labels, hypercharge assignments, neutrino bare scale) be identified with G-invariant model functionals to give graduations? *This is adoption-conditional.*

---

## Result (preview)

**Corollary 4 (Substrate Pati-Salam Conservation, abstract form).** *For G = Spin(4) × Spin(2) acting via the dimensionally-forced Cl(6,0) → Spin(4) × Spin(2) decomposition (B3), every bounded G-invariant model functional f has E_{π_n}[f(Q)] a {𝒢_n}-martingale with value E_{π_0}[f(Q)] forced by the G-symmetric prior — provided (H1) holds.*

**Status of each hypothesis:**

| Hypothesis | Status |
|---|---|
| (H1) π_0 G-invariant | **CONDITIONAL** — plausible from Spin(6)-invariance of A2-T's Cl(6,0)-encoded description language, but requires explicit verification (sub-claim §2.1). Stronger than the discrete Galois-Z₃ case because G is continuous and has more ways to fail invariance implicitly. |
| (H2) {𝒢_n} G-equivariant | **HOLDS** for the substrate observation protocol that does not pre-tag spin-rep components (§2.2). |
| (H3) G-invariant functionals exist | **HOLDS** — Casimirs of Spin(4) and Spin(2), Spin(6) trace functionals restricted to the embedded subgroup, etc. (§2.3). |

**Net structural finding:** the abstract corollary is **theorem-grade conditional on (H1)** for the abstract group action, with (H2) and (H3) verified. The conditional status is honest: continuous-group invariance of the prior is a stronger requirement than discrete-group invariance, and the verification path is more delicate than Phase 1b's.

**Application-conditional caveat.** Connecting Corollary 4 to specific physical predictions (e.g., θ_12 PMNS via the SU(4)_PS perp argument, θ_13 PMNS embedding, neutrino seesaw scale) requires the ADOPTED-B3 labeling (which gauge factor corresponds to which physical force) and ADOPTED-PS-SCALE (M_R ~ 10¹⁵ GeV). Both adoptions remain active per `../audits/registers/adoption_register.md`. The corollary's *symmetry content* is structural; its *parameter applicability* is adoption-conditional.

---

## 1. Setup — Pati-Salam embedding on substrate spinors

### 1.1 The decomposition

Per `docs/theorem_B3_spinor_fermion.py` and B3 paired derivation, the substrate's spinor content factorizes:

$$\mathrm{Cl}(6,0) \;\to\; \mathrm{Spin}(4) \times \mathrm{Spin}(2)$$

at the dimensional level (8-dim Cl(6,0) spinors decompose as 4 ⊗ 2). This decomposition is **dimensionally forced** — it is not adopted; only the physical labeling (which Spin(4) factor is SU(2)_L vs SU(2)_R, what hypercharge assignments correspond to which embedding, etc.) is adopted under ADOPTED-B3.

**The abstract group G.** For the Phase 1c corollary, G is the abstract group Spin(4) × Spin(2) acting on the abstract Cl(6,0) module. No physical labeling is needed at this level.

### 1.2 Action on substrate states and models

The G-action lifts to:
- **Substrate states:** σ_g acts on Cl(6,0) spinor content by the standard Spin(4) × Spin(2) representation theory (left/right-multiplication by spinor generators).
- **Model class 𝒬:** push-forward; each Q ∈ 𝒬 carrying a Cl(6,0)-decomposed spinor sector maps to σ_g · Q with the spinor sector G-rotated.
- **Observation σ-algebras:** σ_g acts on observations of spinor-rep-sector outcomes by the corresponding G-orbit.

The time-shift τ : G → ℤ is identically zero — the PS embedding does not advance the observation clock.

---

## 2. Verification of (H1)–(H3)

### 2.1 (H1) — π_0 G-invariant *(CONDITIONAL — verification path identified)*

**Claim.** A2-T's plural-retention prior π_0 is invariant under G = Spin(4) × Spin(2): π_0(σ_g · Q) = π_0(Q) for all g ∈ G, Q ∈ 𝒬.

**Verification target.** σ(σ_g · M) = σ(M) for all g, M, where σ(M) = L_raw − L_model(M) − H(P) − D(P ‖ Q_M).

**Argument toward (H1).** The two-component check from Phase 1b §2.1 lifts to G:

**(a) The description language is G-invariant: L_model(σ_g · M) = L_model(M).**
The substrate's description language encodes Cl(6,0) spinors at a level that is intrinsically Spin(6)-symmetric (the full Cl(6,0) Clifford algebra carries Spin(6) ≅ SU(4) as its natural symmetry). Restricting Spin(6) to the dimensionally-forced subgroup Spin(4) × Spin(2) preserves G-invariance of the encoding. **Plausible but needs explicit check** that A2-T's MDL description-length encoding does not implicitly fix a Spin(6) → Spin(4) × Spin(2) embedding direction.

**(b) The substrate source distribution P is G-invariant: D(P ‖ σ_g · Q_M) = D(P ‖ Q_M).**
P is the substrate source — the toggle stationary distribution at the substrate level (below the spinor-rep decomposition). The Galois case argued P is invariant because Galois is OUTER on the substrate dynamics. The PS case is structurally similar: G = Spin(4) × Spin(2) acts on the spinor-rep content, which is built ABOVE the substrate's combinatorial dynamics. **Plausible but needs explicit check** that the substrate dynamics do not implicitly distinguish a particular Spin(4) × Spin(2) embedding via the choice of edge-labeling at trivalent vertices.

**Why (H1) is more delicate than Phase 1b's (H1).**
- G is continuous (dim 6 + 1 = 7), so there are continuously many ways to break invariance implicitly.
- The Cl(6,0) → Spin(4) × Spin(2) decomposition is dimensionally forced but the *embedding direction* — which choice of the embedding matters out of the continuously many possible Spin(4) × Spin(2) ⊂ Spin(6) embeddings — is structurally constrained but not yet verified to be unique at the prior level.
- Specifically: B3 fixes the dimensions; the physical labeling (ADOPTED-B3) fixes which embedding direction; (H1) asks whether the *prior is invariant under all G-rotations*, regardless of which embedding direction is chosen.

**Three tractable sub-cases for (H1).** Verification can proceed in stages:
- (H1)_orbit — π_0 is invariant under the G-orbit of any single embedding choice. *Plausible; ~1 session to verify.*
- (H1)_full — π_0 is invariant under continuous G-rotation of the embedding direction itself. *Stronger; may fail if A2-T description language picks out a preferred embedding.*
- (H1)_relative — π_0 is invariant relative to the ADOPTED-B3 labeling. *Adoption-conditional but most directly relevant for parameter applications.*

The natural Phase 1c+ deliverable is to verify (H1)_orbit unconditionally and (H1)_relative under ADOPTED-B3. (H1)_full is more demanding and may either follow from a deeper Spin(6)-invariance argument or fail outright.

**Status of (H1).** **CONDITIONAL.** Verification path identified; deferred to Phase 1c follow-up. Until executed, Corollary 4 ships conditional on (H1).

### 2.2 (H2) — {𝒢_n} G-equivariant *(HOLDS)*

**Claim.** σ_g(𝒢_n) = 𝒢_n for all g, n; τ ≡ 0.

**Argument.** 𝒢_n is generated by all substrate observations through time n, which include outcomes in spinor-rep sectors. Under σ_g, the spinor-rep outcomes are G-rotated, but the set of all possible outcomes through time n is preserved (G acts by permutation/rotation on the outcome space, not by adding or removing outcomes). Hence σ_g(𝒢_n) = 𝒢_n. ✓

(H2) holds independently of (H1) because the filtration is generated by the *outcome space*, while (H1) concerns the *prior weighting* on that outcome space. The two are decoupled.

### 2.3 (H3) — G-invariant model functionals *(HOLDS)*

**Claim.** A non-trivial family of natural model functionals are G-invariant; these are the conserved-expectation candidates.

**G-invariant functionals.** For G = Spin(4) × Spin(2), invariant functionals are scalars under both factors. Standard sources:

| Functional | G-invariant? | Provenance / use |
|---|---|---|
| Casimir C_2(Spin(4)) on a Spin(4)-irrep | ✓ | weights spin-irrep multiplicities |
| Casimir C_2(Spin(2)) on a Spin(2)-irrep | ✓ | analogous for the U(1)_PS factor |
| Trace Tr_Spin(6)(O) restricted to G-invariant O | ✓ | flavor-class observables |
| Sum of squared masses across a G-multiplet | ✓ | seesaw-related observables |
| Single-component spinor mass (e.g., m_e alone) | ✗ | NOT G-invariant; per-component observables are G-rotated |
| ADOPTED-B3-labeled "lepton mass" m_ν | depends on embedding | adoption-conditional G-invariance |

The pattern: **multiplet-traced observables are G-invariant; single-component observables are not.** This is the standard G-invariance picture for any continuous gauge group.

### 2.4 Combined conclusion

(H1) conditional + (H2) ✓ + (H3) ✓ ⇒ Theorem 1 applies *conditional on (H1)*.

**Corollary 4 statement (formal, conditional).** *Conditional on (H1) — that A2-T compression savings are Spin(4) × Spin(2)-invariant — every bounded G-invariant model functional f has*

$$M_n := \mathbb{E}_{\pi_n}[f(Q)] \text{ a martingale w.r.t.~} \{\mathcal{G}_n\}, \quad \mathbb{E}_{\pi_0}[M_n] = \mathbb{E}_{\pi_0}[f(Q)] \text{ for all } n \geq 0.$$

This is the substrate's PS-conservation analog (abstract).

---

## 3. What this conservation does and does not give

### 3.1 What PS conservation gives (under (H1))

- **Conservation law for multiplet-traced observables.** Sum-of-squared masses across a Spin(4) × Spin(2) multiplet, trace functionals on PS sectors, are conserved expectations.
- **Constraint on PS-sector mixing.** PS-class-function observables cannot drift in expectation; transitions across G-orbits are forbidden in the conserved sector.
- **Sanity check on derived PS-related predictions.** Any framework prediction violating G-invariance at the conserved level has a structural error.

### 3.2 What PS conservation does NOT give (even under (H1))

- **Per-component values are NOT conserved or forced.** Individual lepton or quark masses are not G-invariant individually; the cascade theorem deriving them is not displaced.
- **Numerical values still require model-class evaluation.** E_{π_0}[f] is a one-integral computation, not zero-effort.
- **No graduation without ADOPTED-B3 + (H1) closure.** Specific parameter applications (θ_12 PMNS, θ_13 PMNS, neutrino seesaw scale) require both.

### 3.3 Implication for currently-blocked PS-related parameters

| Parameter | Current status | Phase 1c reach |
|---|---|---|
| θ_12 PMNS (P32) | adoption-conditional via ADOPTED-A5b-Sub3 | Cross-check: SU(4)_PS perp argument is G-invariant ⇒ would be conserved under Corollary 4 if (H1) closes |
| θ_13 PMNS (P33) | sub-class part adoption-conditional; PS embedding step BLOCKED | Possible reach: PS-embedding step might unblock if Corollary 4 supplies the structural conservation argument |
| neutrino bare scale M_R | ADOPTED-PS-SCALE | Corollary 4 gives no scale information; (H1) closure does not produce a number |
| V_ub (P14), J_CKM (P45) | theorem-grade STRICT-SOLID via bridge | Cross-validates if PS-invariance and Galois-invariance are jointly applied |

The most meaningful reach is θ_13 PMNS embedding (if Corollary 4 with (H1) closure provides the missing structural step) and θ_12 PMNS cross-check (independent of adoption).

---

## 4. Honest scope

1. **Corollary 4 ships conditional on (H1).** Phase 1c lands the abstract symmetry framework but does not close (H1). Verification path identified in §2.1; estimated 1–2 sessions of focused work.

2. **No graduation in this document.** Phase 1c lands the structural verification but graduates no parameter rows. Parameter applications (§3.3) require both (H1) closure AND ADOPTED-B3 acceptance.

3. **(H1) is structurally harder than Phase 1b's (H1).** Continuous-group invariance is stronger than discrete-group invariance; the verification path involves three sub-cases ((H1)_orbit, (H1)_full, (H1)_relative) of varying tractability.

4. **The adoption-conditional reach is meaningful but bounded.** Even with full closure, Corollary 4 unlocks at most ~2 specific PS-related parameter rows (θ_13 embedding step, θ_12 PMNS cross-check). It is not the broad-spectrum tool Phase 1a+1b together provide.

5. **Cross-validation with Galois Z₃ corollary.** The abstract Spin(4) × Spin(2) and the Galois Z₃ act on different sectors of the substrate's representation content. Their conservation laws are independent; observables that are jointly invariant under both inherit the strongest constraint.

6. **No new prediction emerges from this corollary alone.** Like Phase 1b before its (H1) closure, Phase 1c lands the structural verification with the largest open hypothesis (H1) standing as a follow-up.

---

## 5. Status

**Corollary 4 (Pati-Salam, abstract form):** **THEOREM-GRADE conditional on (H1).** (H2) and (H3) verified. (H1) verification path identified in §2.1, with three tractable sub-cases.

**Phase 1c deliverable:** **CLOSED** at conditional status, with explicit verification path for full closure.

**Effect on framework:**
- PS-conservation analog established in abstract form.
- Phase 1 trio (1a momentum, 1b generation/color, 1c PS) now all delivered.
- Phase 2 priority remains: 2b G_sub > 2c running couplings > 2d arg(h) deferred.

**Effect on parameter ledger:** no graduations.

**Effect on QFT ontology meta-doc:** `../framework/framework_qft_ontology.md` should add a "Pati-Salam / Spin(6)-class conservation" entry pointing here.

---

## 6. Citations

**Type 3 (cited published) references:**

- **Williams, D.** (1991). *Probability with Martingales.* Cambridge University Press. §10.7 (inherited via Theorem 1).
- **Pati, J. C. & Salam, A.** (1974). Lepton number as the fourth color. *Phys. Rev. D* 10, 275. (Foundational PS embedding; ADOPTED-B3 references this for labeling.)
- **Lawson, H. B. & Michelsohn, M.-L.** (1989). *Spin Geometry.* Princeton University Press. Ch I (Cl(6,0) Clifford algebra; Spin(6) ≅ SU(4); standard subgroup decompositions). Used in §1.1 for the abstract group action.
- **Connes, A.** (1973). Une classification des facteurs de type III. *Annales scientifiques de l'É.N.S.* 4(6), 133–252. (Inherited via Phase 1b for the operator-algebraic framing of model class actions.)

All citations to peer-reviewed published work or standard textbooks.

---

## 7. Cross-references

- `theorem_substrate_symmetry_to_martingale.md` — engine theorem; this corollary's parent (§5.4 stub now executed).
- `theorem_substrate_momentum_conservation.md` — sibling Phase 1a corollary.
- `theorem_substrate_generation_charge_conservation.md` — sibling Phase 1b corollary.
- `../audits/registers/adoption_register.md` ADOPTED-B3 + ADOPTED-PS-SCALE — supplies the physical labeling and scale; required for parameter applications.
- `../framework/B3_B6_reconciliation.md` — Cl(6,0) decomposition structure.
- `../framework/framework_qft_ontology.md` — meta-doc; should be updated per §5.

---

## 8. Next forward-construction steps

Phase 1c complete with honest conditional status on Corollary 4. Natural next deliverables:

1. **(H1) verification for Corollary 4** — execute (H1)_orbit and (H1)_relative as outlined in §2.1. Estimated 1–2 sessions. Closes Corollary 4 at unconditional theorem-grade for the abstract case + adoption-conditional for parameter applications.

2. **Phase 2b G_sub scoping.** With Phase 1a+1b+1c trio delivered, the Phase 2b sharpened hypothesis (Phase 1a §3.3) can be re-examined. The candidate "additional symmetry G'" needed alongside translation may be identifiable as a combination of the corollaries already in hand.

3. **Phase 2c running couplings via RG-as-time-translation.** Independent of Phase 1c; can run in parallel with (H1) closure.
