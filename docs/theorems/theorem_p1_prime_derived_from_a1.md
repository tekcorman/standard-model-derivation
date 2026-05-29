# Theorem: P1' is derivable from A1 + framework meta-requirements

**Date:** 2026-05-02 EOD+9 (axiom elimination roadmap; derives P1' from A1 alone)
**Status:** STRUCTURAL-DERIVATION (axiom slate reduction)
**Depends on:** A1 (binary self-inverse toggle).
**Cross-references:**
- `docs/framework/framework_axioms.md` (post-2026-05-08 slate: P1' is derived theorem; this theorem's place in the chain remains unchanged)
- `docs/theorems/theorem_toggle_from_self_containment.md` (post-2026-05-08; supersedes the MR1/MR2/MR3 framing of this theorem at the top level)
- `docs/theorems/theorem_A2_mdl_from_finite_register.md` (A2 derived using P1')
- `docs/theorems/theorem_A3_complex_hilbert_from_multiway.md` (A3 derived using P1')

**Post-2026-05-08 status note.** Under the post-2026-05-08 axiom slate revision (`framework_axioms.md` §10), the meta-requirements MR1/MR2/MR3 used in this theorem's proof are subsumed by the new top-level commitments: MR1 (self-containment) → **(A)** self-containment of the universe (metaphysical commitment); MR2 (finite-resource physical realizability) → **(B)** finite observer (scoping commitment); MR3 (multi-observation predictivity) → consequence of (B) + standard finite-computation theory (Turing 1936, Cover-Thomas 2006). The theorem's content (P1' as derived) is preserved; only the framing of upstream commitments changes. See `theorem_toggle_from_self_containment.md` for the new top-level derivation that supplies A1 itself as a theorem of (A) + (B) + Shannon-Jaynes + active reading.

## Statement

Under axiom A1 (binary self-inverse toggle) plus three framework meta-requirements (self-containment, finite-resource physical realizability, multi-observation predictivity), the observer concept is UNIQUELY determined to be **a finite register built from binary toggles, persistent across observations**. This is exactly the content of P1'.

Therefore P1' transitions from a "definitional commitment about scope" to a **derived theorem of A1 + standard finite-computation theory**.

## Framework meta-requirements

The framework's stated purpose is to predict Standard Model observables from minimal axioms. This purpose imposes three meta-requirements that any consistent framework instantiation must satisfy:

**(MR1) Self-containment.** The observer concept must be definable in terms of the framework's own primitives. If the observer were specified externally (with primitives outside A1), the framework would be predictively dependent on un-axiomatized content — contradicting the minimal-axiom claim.

**(MR2) Finite-resource physical realizability.** The observer must correspond to a physically realizable system with finite resources. An observer with infinite memory or infinite computation has no physical instantiation; its predictions could not be implemented or verified.

**(MR3) Multi-observation predictivity.** The framework must support predictions across multiple observations (e.g., spectral statistics of B^L for multiple L, MDL compression of repeated data). A one-shot oracle that emits a single answer and resets does not constitute a meaningful framework "observer."

These meta-requirements are not physical postulates — they are constitutive of what it means for the framework to be a predictive theory at all. Any framework that fails MR1 is non-self-contained; any failing MR2 is unphysical; any failing MR3 is non-predictive.

## Proof

### Step 1: MR1 + A1 → observer built from binary toggles

A1 introduces the substrate primitives as binary toggles {T_e}_{e∈E} with T_e ∘ T_e = id. These are the framework's only primitives.

By MR1, any observer concept must be definable using these primitives. Therefore the observer is, at minimum, **built from binary toggles** — the same primitives that constitute the substrate.

Specifically: the observer's internal state space is a function of {T_e} compositions, and the observer's update operations are themselves toggle compositions. This is direct from A1's universality (no other primitives exist) plus MR1 (no external primitives admitted).

### Step 2: MR2 → observer is a finite register

By MR2, the observer corresponds to a physically realizable system with finite resources. By Turing 1936 + Cover & Thomas 2006 §1.6 (entropy of finite-alphabet sources), any finite-resource computational system has a finite state space. The observer's state space is therefore **finite**.

A finite state space accessible by binary toggle composition is, by definition, a **finite register**: a finite collection of binary cells whose state is updated by toggle operations. (Standard computer-science definition; see Sipser 2013 Ch. 1.)

Combining Step 1 + Step 2: the observer is a **finite register built from binary toggles**.

### Step 3: MR3 → observer persists across observations

By MR3, the framework supports multi-observation predictions. For an observer to make a multi-observation prediction, the observer must EXIST during all observations — i.e., the observer's state must persist between observations.

If the observer were re-instantiated for each observation (no persistence), each observation would be independent, and no MDL-compression-based prediction could be made (since MDL requires aggregating multiple observations into a single compressed model — Grünwald 2007 §5.1, §17.1). The framework's MDL waterline (theorem_A2_mdl_from_finite_register.md) DEPENDS on persistent observation accumulation; without it, A2 cannot be derived.

Therefore the observer **persists across observations**.

### Step 4: combining Steps 1-3 = P1'

The observer is:
- (Step 1) built from binary toggles (the substrate primitives, per A1)
- (Step 2) a finite register
- (Step 3) persistent across observations

This is **exactly the content of P1'** as stated in `framework_axioms.md` §10:

> "P1': The observer exists within the framework as a finite register, built from the same primitive (binary toggles) as the substrate, persisting across multiple observations."

∎

## Uniqueness

The proof shows P1' is FORCED by A1 + (MR1, MR2, MR3). Could a different observer concept also be consistent? No:

- Any observer with primitives other than binary toggles violates MR1 (not self-contained under A1).
- Any observer with infinite state space violates MR2 (not physically realizable).
- Any non-persistent observer violates MR3 (no multi-observation predictions).

Therefore P1' is the **unique** observer concept consistent with A1 + framework meta-requirements. There is no choice; P1' is determined.

## Consequence: axiom slate reduction

Prior framework axiom slate (per `framework_axioms.md` §10 as of 2026-04-27):

> {A1} structural axiom + A5-mass downstream labeling + P1' definitional commitment

Updated slate (post this theorem):

> **{A1} structural axiom + A5-mass downstream labeling**

P1' is now a derived theorem of A1 + standard finite-computation theory, not a separate framework commitment. Existing derivations that cited A1 + P1' (theorem_A2_mdl_from_finite_register.md, theorem_A3_complex_hilbert_from_multiway.md, etc.) continue to hold; they now cite "A1 + P1' (theorem)" instead of "A1 + P1' (definitional)."

The framework's irreducible structural commitments are now:
- **A1** (binary self-inverse toggle) — single physical axiom
- **A5-mass** (downstream labeling identifying which Bloch-Hashimoto eigenvalue corresponds to which SM mass) — empirical content, conceptually distinct from physical postulate

The endgame target ({A1} alone, eliminating A5-mass via structural derivation of generation labeling) remains open. P1' elimination is one step on that path.

## Remarks

**On the strength of the derivation.** This theorem is conceptually modest: it formalizes what the framework's existing language already states ("P1' adds no physical content beyond A1"). Its value is in TRANSFERRING P1' from the axiom slate to the theorem ledger, not in adding new mathematical content.

**On the meta-requirements.** MR1, MR2, MR3 are not themselves axioms — they are constitutive of the framework's purpose (being a self-contained, physically realizable, predictive theory of SM observables). Adopting them is not adopting a physical postulate; it is choosing to do science rather than philosophy.

**On comparison to prior treatment.** The prior framework treatment of P1' as "definitional commitment per no_free_bits §1.1" was equivalent in content to this theorem but did not formally derive uniqueness from MR1-MR3. This theorem's value is making the derivation explicit and the axiom-status reduction formal.

**On the next step in axiom elimination.** With P1' eliminated, the only remaining non-A1 commitment is A5-mass (the labeling clause). Eliminating A5-mass would require deriving the generation/mass labeling from substrate algebra alone — currently blocked by the color-vs-generation choke point (`memory/project_color_generation_choke_point_2026-04-25.md`). That is a deeper structural problem; this theorem makes no progress on it but also does not depend on it.

## References

- Turing, A. M. (1936). "On Computable Numbers, with an Application to the Entscheidungsproblem." *Proc. London Math. Soc.* 42(1), 230-265.
- Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley. §1.6.
- Sipser, M. (2013). *Introduction to the Theory of Computation* (3rd ed.). Cengage. Ch. 1.
- Grünwald, P. D. (2007). *The Minimum Description Length Principle*. MIT Press. §5.1, §17.1.
- `docs/framework/framework_axioms.md` (P1' historical statement)
- `docs/theorems/theorem_A2_mdl_from_finite_register.md` (consumer of P1')
