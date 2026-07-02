# Theorem: Binary self-inverse toggle and F_inv(E) derived from self-containment + finite observer

**Date:** 2026-05-07 (axiom slate revision; demotes A1 to derived theorem).
**Status:** STRUCTURAL-DERIVATION (axiom slate reduction). Demotes A1 from structural axiom to derived theorem. Establishes the framework's revised top-level: one metaphysical commitment (self-containment) + one scoping commitment (finite observer) + standard published mathematics + one explicit interpretive commitment (active reading of binary distinctions).
**Depends on:** (A) self-containment of the universe; (B) finite observer (scoping); standard published mathematics (Shannon 1948, Jaynes 1957, Cover-Thomas 2006, Serre 1980).
**Cross-references:**
- `docs/framework/framework_axioms.md` (to be revised after this theorem lands; A1 demoted to derived theorem)
- `docs/theorems/theorem_p1_prime_derived_from_a1.md` (currently cites A1 as input; after revision, cites this theorem and (B) directly)
- `docs/theorems/theorem_A2_mdl_from_finite_register.md` (cites A1; preamble to be updated)
- `docs/theorems/theorem_A3_complex_hilbert_from_multiway.md` (cites A1; preamble to be updated)
- `docs/theorems/theorem_substrate_agnosticism.md` (companion theorem on observational equivalence of substrates; not a dependency of this theorem)
- `predictions/p_toggle_derivation.md` (derives p = 2 given T² = id; this theorem derives T² = id from upstream)

---

## Statement

Under (A) self-containment of the universe + (B) finite observer, plus standard published mathematics + the active-reading interpretive commitment (Step 5), the observer's primitive update operation is uniquely determined to be a **binary self-inverse toggle** T_e satisfying T_e² = id, and the algebra of update compositions is the **free involutive monoid** F_inv(E) = *_{e ∈ E}(ℤ/2) on a finite alphabet E.

This is exactly the content of A1 as previously stated in `framework_axioms.md` §2. A1 therefore transitions from a structural axiom of the framework to a derived theorem of (A) + (B) + standard math + the active-reading commitment.

---

## Top-level commitments

The framework rests on two top-level commitments and one interpretive commitment, plus standard published mathematics.

**(A) Self-containment (metaphysical).** The universe is closed to itself; nothing comes from outside, because nothing is outside. This is metaphysical, not derivable. It cannot be proved from anything more fundamental, because it stipulates that nothing more fundamental is supplied.

Frameworks that smuggle in external information — boundary conditions, anthropic priors, fine-tuning, multiverse selection — make additional commitments. (A) refuses them.

**(B) Finite observer (scoping).** The framework describes observers with finite memory capacity. This is a scoping definition — a statement about the subject of the framework's predictions — not a physical postulate. It scopes the framework to the actual case (any real observer is finite; no science is conducted by an observer with unbounded memory).

(B) is conceptually distinct from (A). (A) does not entail finiteness of the universe (the substrate constructed in this theorem is countably infinite); (B) carries the finiteness needed for downstream content via the observer alone.

**(I) Active-reading interpretive commitment (Step 5 below).** The framework reads the observer's primitive binary distinction *actively* — as an operation that moves between two states — rather than passively as a static attribute. This is a structural choice motivated by the framework's relational stance. It is not derived from (A) and (B); it is adopted.

**Empirical labeling (A5-mass).** Out of scope of this theorem. Identifies which Bloch-Hashimoto eigenvalues correspond to which Standard Model masses. See `framework_axioms.md` §5b.

---

## Proof

The proof proceeds in eight steps. Steps 1–3 establish the substrate's structure (uniform measure, multiway, discreteness in observer's perceived substrate). Steps 4–5 establish the observer's primitive update (binary, involutive). Steps 6–7 establish the algebra (F_inv(E)). Step 8 identifies the multiway substrate with the Cayley graph of F_inv(E).

### Step 1 — (A) supplies no information; the unique zero-information distribution is uniform.

By (A), no information is supplied from outside. Information, by Shannon 1948, is imbalance from a baseline; quantitatively, the information of a configuration with probability p is −log₂ p. To carry zero information, a distribution must privilege no configuration over any other.

By Jaynes 1957's maximum-entropy theorem, under any symmetry over the space of alternatives — at minimum, the permutation symmetry over indistinguishable outcomes — the unique distribution that maximizes Shannon entropy under no informational constraints is uniform. Any non-uniform distribution privileges some outcome over others, and the privileging is itself a piece of information that (A) forbids supplying.

Therefore the substrate's measure is uniform over alternatives. This is forced, not chosen.

### Step 2 — Single-history substrates carry information; multiway is forced.

A single-history substrate corresponds to a delta-distribution: probability one on the realized history, zero on alternatives. This carries information — it privileges one outcome over all others — and so violates (A) by Step 1.

The unique distribution consistent with (A) is the uniform measure over all alternatives. Equivalently, the substrate is a multiway structure in which every alternative is realized with equal weight: μ(B) = |E|^{−L} for length-L compositions B over the (yet-to-be-constructed) alphabet E (cf. `theorem_multiway_branch_measure.md`, which derives this measure once E is in hand).

Multiway is therefore forced by (A) + Shannon-Jaynes uniqueness, not adopted as a separate ontological commitment.

### Step 3 — (B) requires the observer to compress; finite-memory compression forces discreteness in the observer's perceived substrate.

By (B), the observer's memory has finite capacity. Across multiple observations, the cumulative observed content may exceed that capacity. The observer must therefore compress: store a model M and residual data given M, with total description length L(M) + L(data | M) bounded by the memory capacity (Cover-Thomas 2006 §1.6, §5.4).

A compressing observer's model has finitely many distinguishable internal states (because the model is stored in finite memory). The observer's perceived substrate is structured by those states: distinct observations that the model assigns to the same internal state are operationally indistinguishable from the observer's perspective. The observer's perceived substrate is therefore at most as fine as the model's state count — that is, **discrete**.

This is a statement about the observer's *perceived* substrate. The substrate-in-itself (whether continuous, smooth, granular, or otherwise) is left undetermined — addressed separately in `theorem_substrate_agnosticism.md`. Discreteness is a property of observer access mode under (B), not of substrate metaphysics.

### Step 4 — Shannon's 1-bit minimum: the observer's primitive update is a binary distinction.

By Shannon 1948 §I, the smallest non-trivial information unit is one bit (= log₂ 2). A signal less than one bit (= log₂ 1) is no information at all; a signal more than one bit is decomposable into a sequence of bit-valued primitives (Cover-Thomas 2006 §5.4, source coding theorem).

Therefore the observer's primitive update — the smallest non-trivial change in the observer's cumulative state — is a **binary distinction**. Larger updates are not separate primitives; they are compositions of binary primitives. This is structural decomposition, not parsimony: no primitive smaller than one bit exists, and any larger primitive *is* a composition.

### Step 5 — Active reading of binary distinctions ⟹ involutive operator T_e² = id. (Interpretive commitment.)

Step 4 establishes the observer's primitive as a binary distinction. To incorporate the distinction into an *algebra* of updates requires reading the distinction *actively* — as an operation that moves between two values — rather than passively as a static attribute.

**Interpretive commitment (I).** The framework adopts the active reading. Under the active reading, a binary distinction labeled e is an operator T_e on configurations, mapping each configuration c to the configuration that differs from c precisely in the value of slot e.

This is a structural choice, not a derived consequence. Alternative readings exist:
- *Passive reading.* Treat the distinction as a static label attached to configurations (a property the configuration has, not an operation it undergoes). Under this reading, the framework's "dynamics" must be supplied externally (a second commitment), and the relational stance is forfeit.
- *Asymmetric reading.* Treat one direction of the distinction as primary and the other as derived. Under this reading, T_e ≠ T_e^{−1} in general, and a separate inverse operation must be specified.

The framework's adoption of the active reading is motivated by the *relational stance* that (A) suggests: with no exterior, there is no preferred frame supplying dynamics; what is observed as dynamics must therefore live in the observer's traversal of the substrate, not in the substrate's evolution. The active reading is what locates dynamics in the observer's accumulating contemplations rather than in any substrate ticking.

Under the active reading, T_e² = id follows from the binary symmetry of the distinction: applying T_e once moves between the two values of slot e; applying T_e twice returns to the starting value. There is no preferred direction of the distinction (no "more flipped" or "less flipped" state), so T_e and its inverse coincide:

$$T_e \circ T_e = \mathrm{id}, \quad T_e = T_e^{-1}.$$

This gives the involutive structure of the toggle.

### Step 6 — (B) ⟹ finite alphabet E of distinguishable involutive generators.

By (B), the observer has finite memory and operates with a finite repertoire of distinguishable update kinds. An observer that distinguishes infinitely many primitive update types violates the finite-resource requirement (storing infinitely many distinct labels exceeds finite memory).

Let E denote the finite set of primitive update labels. By construction, the observer can distinguish e ≠ e' in E (otherwise they would be identified as a single primitive). Therefore E is a finite alphabet of distinguishable involutive generators {T_e : e ∈ E}, with each T_e satisfying T_e² = id by Step 5.

### Step 7 — No a priori commutation among distinct generators ⟹ F_inv(E).

Compositions of primitive updates yield the observer's algebra. Four properties characterize this algebra:

(i) Generated by {T_e : e ∈ E} (Step 6).
(ii) T_e² = id for each e ∈ E (Step 5).
(iii) Composition is associative (automatic from operator composition).
(iv) For distinct e, e' ∈ E, no commutation relation T_e T_{e'} = T_{e'} T_e is imposed.

Property (iv) is the operational consequence of (A) at the structural level: imposing commutation among distinct generators would be additional structural commitment beyond what (A), (B), Shannon-Jaynes, and Steps 1–6 supply. Since (A) forbids supplying anything from outside, no such commutation is imposed. (This is the same parsimony principle that gives Step 1's uniform measure: no privilege without supplied input.)

Under properties (i)–(iv), the observer's algebra is the **free product** *_{e ∈ E}(ℤ/2) of |E| copies of the cyclic group of order 2. As a monoid presentation under the involutive relations e * e ~ ε for each e ∈ E, this is the **free involutive monoid** F_inv(E) on alphabet E. Reduced-word uniqueness in F_inv(E) is given by Serre 1980 §I.1 Proposition 4.

### Step 8 — Multiway substrate = Cayley graph of F_inv(E).

The multiway structure of Step 2 has vertices indexed by elements of F_inv(E) (each contemplated alternative is a reduced word) and edges given by single-generator applications (T_e takes one configuration to its e-flipped neighbor, by Step 5). This is precisely the **Cayley graph** of F_inv(E) with generator set {T_e : e ∈ E}.

The substrate (multiway, viewed externally) and the observer's algebra (F_inv(E), viewed internally) are therefore the same mathematical object viewed two ways. The substrate has no ontology independent of the observer's algebra; observer and substrate are dual aspects of one structure.

(Cf. `predictions/walker_dynamics_derivation.md` Step 1, which constructs F_inv(E) and its Cayley graph from A1; under this theorem, that construction is downstream of (A) + (B) rather than of A1 directly.)

∎

---

## Uniqueness

The proof shows the toggle and F_inv(E) are FORCED by (A) + (B) + standard math + the active-reading interpretive commitment. Could a different observer's primitive also be consistent? No:

- Any non-binary primitive violates Step 4 (Shannon's 1-bit minimum), or decomposes into binary primitives.
- Any non-involutive primitive violates Step 5 under the active reading.
- Any infinite alphabet of primitives violates Step 6 ((B) finite observer).
- Any algebra with a priori commutation among distinct generators violates Step 7's parsimony — adds structure not supplied by (A), (B), or standard math.

The active reading itself is interpretive, not derived. Under alternative readings (passive, asymmetric), different framework structures result, but each forfeits the relational stance that (A) suggests. Within the relational stance, the active reading is the natural and minimal choice.

---

## Consequence: axiom slate revision

Prior framework axiom slate (per `framework_axioms.md` §10 as of 2026-05-02):

> {A1} structural axiom + A5-mass downstream labeling

Updated slate (post this theorem):

> **(A)** self-containment (metaphysical commitment; not derivable)
> **(B)** finite observer (scoping commitment; defines the framework's subject)
> **(I)** active-reading interpretive commitment (Step 5 of this theorem)
> **A5-mass** (empirical labeling; identifies which Bloch-Hashimoto eigenvalues correspond to which SM masses)

A1 is now a derived theorem of (A) + (B) + standard math + (I).

P1' (derived in `theorem_p1_prime_derived_from_a1.md`) now cites this theorem rather than A1 directly; (B) supplies the finite-observer content that the prior MR2 supplied. The MR1, MR2, MR3 framing of `theorem_p1_prime_derived_from_a1.md` is subsumed:
- MR1 (self-containment) → (A).
- MR2 (finite-resource physical realizability) → (B) (scoping rather than physical postulate).
- MR3 (multi-observation predictivity) → property of (B) + standard finite-computation theory; not a separate commitment.

A2 (`theorem_A2_mdl_from_finite_register.md`), A3 (`theorem_A3_complex_hilbert_from_multiway.md`), A4 (`theorem_car_local_jordan_wigner.md`) remain derived theorems; their preambles will be updated to cite (A) + (B) via this theorem rather than A1 directly. Proofs unchanged.

The framework's irreducible structural commitments are now:
- (A) self-containment — single metaphysical premise.
- (B) finite observer — scoping definition.
- (I) active reading — interpretive commitment for binary distinctions.
- A5-mass — empirical labeling.

Plus standard published mathematics. No other physical postulates.

---

## Remarks

**On the strength of the derivation.** The structural content of A1 — binary, involutive, free involutive monoid algebra — is fully captured by Steps 4–7. Steps 1–3 supply the substrate-side context (uniform multiway measure + observer-perceived discreteness) within which the toggle algebra is the observer's natural primitive. Step 8 is the bridge identifying observer-algebra and substrate-graph as one object.

**On the active-reading commitment.** The active reading (I) is not on the same shelf as (A) and (B). It is an interpretive choice within the relational stance that (A) suggests. The framework adopts it explicitly rather than smuggling it in. Alternative readings yield different (and structurally weaker) frameworks; none has been pursued in this corpus.

**On the parsimony principle in Step 7.** "No a priori commutation" is the operational consequence of (A) at the structural level: nothing supplied from outside, no relations imposed beyond what is supplied. This is the same principle that gives uniform measure in Step 1 (nothing supplied → no privilege → uniform). It is not a separate commitment; it is (A) applied to algebraic structure.

**On (A) applied to spatial structure — the no-privilege principle once more (added 2026-05-12; closes R-9).** The same no-privilege principle — Step 1 applied it to *configurations* (uniform measure), Step 7 applied it to *algebra* (no commutation) — applies to *every* label the observer's model carries. In particular, once the spatial embedding of the substrate is derived downstream (`predictions/d_spatial_derivation.md`: MDL + Gleason ⟹ d = 3; `predictions/k_star_derivation.md`: k* = d = 3; the substrate is a 3-regular 3-periodic crystal net), (A) forbids privileging any *spatial direction* at a vertex or either *orientation* of an edge — that would be "which-way" information, exactly the kind (A) refuses to supply. The downstream walker's causal state is a *directed edge* (`predictions/walker_dynamics_derivation.md` Step 5 — Shalizi-Crutchfield 2001), so this no-privilege means the observer's model must treat all directed edges equivalently: its crystallographic automorphism group acts transitively on (vertex, directed-edge) pairs, i.e. the model is **strongly isotropic (arc-transitive)**. By `theorem_substrate_agnosticism.md` (the substrate *is* the observer's description-length-minimal canonical model), the substrate is strongly isotropic. This is "isotropic toggle dynamics" — already invoked, without this name, in `d_spatial_derivation.md`'s Shannon-chain-rule lemma. **It is derived from (A), not adopted** — on par with the uniform measure and the missing commutation. Combined with Sunada 2012 (the unique 3-regular 3-connected ℝ³ crystal net that is strongly isotropic is **srs**, the Laves / (10,3)-a net, up to handedness) and k* = 3, d = 3, this forces the substrate to be srs — the structural closure of R-9 (`docs/audits/registers/structural_residue_register.md`; full derivation chain in `walker_dynamics_derivation.md` Step 4b and `g_girth_derivation.md` Step 2). The 8 RCSR competitors (srs-z, srs-c4, srs-c8, srs-c27, lou, lov, okw, hcb-c4) are vertex- and edge-transitive but *not* arc-transitive (≥2 directed-edge orbits = "which-arc-type" structure (A) supplies nothing to justify; and, unlike srs, none can be specified by symmetry alone — they need explicit coordinate/index bits); srs-z, hcb-c4, lou, lov, okw are *additionally* non-minimal-cell (3-periodicity forces |V| ≥ 4 for a 3-regular net, so |V| = 4, |E| = 6 is the minimum — the |V| = 4 class).

**On the relationship to the prior {A1}-only slate.** The prior slate took A1 as the irreducible structural axiom. This theorem demotes A1 by deriving its content from a more parsimonious set of commitments — at the cost of making explicit one previously-implicit interpretive commitment (the active reading) and one previously-implicit scoping commitment (finite observer). The trade is honest: the prior {A1} framing did not name the active reading, even though A1's involutive content depends on it. This theorem makes the dependence explicit.

**On the framework's pitch.** The honest one-line version of the framework's foundational claim is:

> *One metaphysical commitment (self-containment of the universe) + one scoping commitment (finite observer) + one interpretive commitment (active reading of binary distinctions) + one empirical labeling (A5-mass) + standard published mathematics = the Standard Model.*

The "single axiom" framing is a special case in which the scoping and interpretive commitments are absorbed silently into the metaphysics. Naming them explicitly is structurally cleaner and defensible against careful reading.

Note that the substrate's *spatial form* — d = 3, k* = 3, and the specific net srs — is *not* an extra commitment: d and k* are derived from (A) + (B) + Gleason + MDL (`d_spatial_derivation.md`, `k_star_derivation.md`), and srs is then forced because (A)'s no-privilege applied to spatial directions/orientations makes the substrate model strongly isotropic (arc-transitive), and Sunada 2012 proves the strongly-isotropic 3-regular 3-connected ℝ³ crystal net is unique = srs (see the remark "On (A) applied to spatial structure" above). "Strong isotropy" is (A) wearing a spatial hat — the closure of R-9 — not an adopted lattice property.

---

## References

### Cited published theorems

- Shannon, C. E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal* 27(3), 379–423; 27(4), 623–656. §I (information as −log₂ probability); Theorem 9 (source coding theorem).
- Jaynes, E. T. (1957). Information theory and statistical mechanics. *Phys. Rev.* 106, 620–630. (Maximum-entropy theorem under symmetry.)
- Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley. §1.6 (entropy of finite-alphabet sources); §5.4 (source coding theorem).
- Serre, J.-P. (1980). *Trees.* Springer. §I.1 Proposition 4 (reduced-word uniqueness in free involutive monoids).

### Framework documents

- `docs/framework/framework_axioms.md` (prior axiom slate; to be revised after this theorem lands).
- `docs/theorems/theorem_p1_prime_derived_from_a1.md` (derives observer concept from prior A1; preamble to be updated).
- `docs/theorems/theorem_A2_mdl_from_finite_register.md` (derives MDL waterline from prior A1 + P1'; preamble to be updated).
- `docs/theorems/theorem_A3_complex_hilbert_from_multiway.md` (derives complex Hilbert structure; preamble to be updated).
- `docs/theorems/theorem_multiway_branch_measure.md` (derives the multiway branch measure μ; consistent with Step 2 of this theorem).
- `predictions/walker_dynamics_derivation.md` (constructs F_inv(E) and Cayley graph from prior A1; consistent with Step 8 of this theorem).
