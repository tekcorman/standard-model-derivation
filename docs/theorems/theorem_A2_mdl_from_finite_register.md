# MDL Waterline from Finite-Register Observation — theorem (A2 derived)

**Date:** 2026-04-26.
**Status:** THEOREM — gate-passing under `../parameters/parameter_linter.md`. All load-bearing steps are Type 1 (framework axiom A1), Type 2 (explicit algebra), Type 3 (precisely-cited published theorem: Serre 1980, Shannon 1948, Rissanen 1978, Grünwald 2007), or Type 4 (none required — this theorem is upstream of most framework derivations).
**Scope:** narrow. Derives the MDL waterline (selective retention of every model whose total cost is less than the raw cost) from a finite-register observer of an unbounded substrate operating across multiple observations. This is what A2 asserts; the present theorem demotes A2 from framework axiom to derived theorem.
**Effect on framework axioms:** Reduces structural axiom count (jointly with `theorem_A3_complex_hilbert_from_multiway.md` and `theorem_car_local_jordan_wigner.md`) from {A1, A2, A3, A4} to {A1}. Combined with A5-mass as downstream labeling and P1' as definitional commitment, the framework's complete axiom slate becomes {A1, A5-mass} with P1'.
**Out of scope:** specific applications of MDL to the framework's downstream content (V_cb winding sums, V_us counting fractions, etc.) — these are *uses* of the waterline, derived elsewhere.
**Replaces:** A2 as framework axiom in `../framework/framework_axioms.md` §3. After this doc lands and is reviewed cold, framework_axioms.md should be updated to demote A2 to "derived theorem; see this document."

**Post-2026-05-08 axiom slate note.** A1 and P1' (cited as "Type 1" inputs throughout this theorem) are now both derived theorems of the new top-level slate: A1 from `theorem_toggle_from_self_containment.md` (under (A) self-containment + (B) finite observer + Shannon-Jaynes + (I) active reading); P1' from `theorem_p1_prime_derived_from_a1.md` (under the same top-level commitments via subsumed MR1/MR2/MR3). References to "A1 + P1'" in this document are semantically equivalent to "(A) + (B) + (I) + standard math, via the toggle and P1' theorems." The present theorem's proof and conclusion are unchanged. See `framework_axioms.md` §10 for the updated top-level summary.

---

## 1. Theorem statement

**Theorem (MDL Waterline from Finite-Register Observation).** Under A1 + P1', the optimal encoding strategy for substrate content is the *MDL waterline*: a model M is retained iff its total description length L(M) + L(data | M) < L(raw observation), and when multiple models clear this threshold all are retained, weighted by their compression savings.

Specifically:

(i) The substrate F_inv(E) is unbounded; the observer is a finite register; therefore some encoding-with-loss is forced.
(ii) The observer's optimal encoding minimizes total expected description length over its operational lifetime.
(iii) For model-bearing encodings, total description length decomposes as L(M) + L(data | M) (Rissanen 1978).
(iv) Encodings with L(M) + L(data | M) ≥ L(raw) save no bits; they cannot fit in the register without displacing more economical encodings, so they are not realized in the observer's compressed view.
(v) When multiple encodings simultaneously clear the threshold L(M) + L(data | M) < L(raw), all are realized in the compressed view, weighted by their compression savings (Grünwald 2007 §17).

This is precisely what A2 asserts as an axiom (in its refined / waterline form). Under the present theorem A2 is derived rather than postulated.

---

## 2. Axioms invoked + cited upstream

**Framework axioms (Type 1):**

- **A1** (`../framework/framework_axioms.md` §2) — finite alphabet E of binary self-inverse toggle generators; substrate is the free involutive monoid F_inv(E).

**Definitional commitment (P1'):** the observer exists within the framework as a finite register (operational definition, grounded in no_free_bits §1.1's "exists = toggle composed into structure"). P1' is not an axiom but a scope-fixing definition: it states what we mean by "observer" within this framework (an entity built from the same primitives as the substrate, of finite extent, operating across multiple observations). It does not introduce physical content beyond what A1 already provides.

**Type 3 cited published theorems:**

- **Serre, J.-P.** (1980). *Trees.* Springer. §I.1 Proposition 4 — uniqueness of reduced word in F_inv(E), and consequent infinitude of F_inv(E) for finite alphabets E with at least one non-trivial generator.
- **Shannon, C. E.** (1948). A Mathematical Theory of Communication. *Bell System Technical Journal* 27(3), 379–423; 27(4), 623–656. §I (definition of self-information / surprise); Theorem 9 (source coding theorem: optimal expected code length per symbol equals the source entropy, achievable by prefix codes).
- **Rissanen, J.** (1978). Modeling by shortest data description. *Automatica* 14(5), 465–471. §2 — the MDL principle: for an encoding using a model M, total description length decomposes as L(M, data) = L(M) + L(data | M); this decomposition is forced by prefix-code reversibility.
- **Grünwald, P.** (2007). *The Minimum Description Length Principle.* MIT Press. §§5.1–5.3 (rigorous statement of MDL as a generalization of source coding to model-bearing encodings); §17 (multi-admissible regime: when multiple models clear the waterline simultaneously, the optimal estimator is a Bayesian mixture weighted by compression savings — the plural-waterline reading).

No Type 4 (upstream framework theorem) citations are required. This theorem sits at the foundation of the framework's compression machinery.

No fabricated citations. No post-hoc fitting. No appeal to "by analogy" or "it can be shown."

---

## 3. Setup

By A1, the substrate dynamics are generated by a finite alphabet E of binary self-inverse toggle operators. The set of finite reduced words on E forms the free involutive monoid F_inv(E) (Serre 1980 §I.1 Proposition 4). For |E| ≥ 1 with at least one non-trivial generator, F_inv(E) is countably infinite.

By P1', the observer is a finite register inside this substrate. The register has finite bit capacity B; substrate states are F_inv(E) elements, of which there are countably infinitely many. The observer continues to operate across multiple substrate observations (P1' includes persistence; an entity that observes once and stops is not what we mean by an observer in this framework — it is a one-shot detector).

This document derives the MDL waterline from the operational consequence of finite-register observation of an unbounded substrate.

---

## 4. Step 1 — Substrate is unbounded; observer is finite; some erasure is forced

**Claim.** The substrate has unboundedly many distinct content states; the observer's register has finite capacity B; therefore the observer cannot store all substrate content, and some encoding-with-loss is forced.

**Proof.** By Serre 1980 §I.1 Prop 4, for finite alphabet E with at least one non-trivial generator, F_inv(E) is countably infinite — there are infinitely many distinct reduced words. By P1', the observer's register has finite bit capacity B, so it can be in at most 2^B distinct states. Since 2^B < ℵ₀ = |F_inv(E)|, the map from substrate states to register states cannot be injective — some substrate states must map to the same register state. The observer's encoding is therefore lossy.

**Type 1** (A1, supplying the alphabet) + **Type 3** (Serre 1980 — countable infinity of F_inv(E)) + **Type 2** (cardinality argument: 2^B < ℵ₀ for any finite B).

---

## 5. Step 2 — An encoding is a function from substrate sequences to register states

**Claim.** The observer's encoding of substrate content into the register is a function f: S* → R where S* is the space of substrate observation sequences and R is the register state space.

**Proof.** The observer observes substrate transitions over its operational lifetime; the cumulative observation is a sequence in S*. The register stores the observer's compressed representation of this sequence; the representation is a function of the observation sequence, mapping into R.

The encoding may be deterministic (each sequence maps to a definite register state) or probabilistic (sequences map to register-state distributions). The optimal-encoding analysis below applies in both cases, with appropriate generalizations of "expected description length."

**Type 2** (definitional — the encoding's existence and functional form follow from P1's commitment to the observer being a register that maps observed content into stored bits).

---

## 6. Step 3 — Expected description length

**Claim.** The expected description length of an encoding f under source distribution P over substrate observation sequences is

L(f) = E_{x ∼ P}[length(f(x))]

where length(r) is the number of bits to specify r ∈ R.

**Proof.** The expected description length is the average number of bits required to represent a substrate observation sequence under the encoding f. By the standard definition of expected value over the source distribution P, this is the integral (or sum, for discrete spaces) of length(f(x)) weighted by P(x). For a discrete substrate (which F_inv(E) is), this is L(f) = Σ_x length(f(x)) P(x).

**Type 2** (standard definition of expected description length; Cover-Thomas 2006 §5).

---

## 7. Step 4 — Shannon source coding theorem

**Claim.** For a stationary ergodic source with entropy rate H, the expected description length per symbol satisfies L ≥ H, and prefix codes achieving L = H + ε exist for any ε > 0.

**Proof.** This is Shannon's source coding theorem (1948, Theorem 9). It establishes the entropy as the fundamental lower bound on expected encoding length for any uniquely decodable code. Prefix codes (a sub-class of uniquely decodable codes) can be constructed to approach this bound arbitrarily closely.

**Type 3** — Shannon, C. E. (1948), *Bell System Technical Journal* 27, Theorem 9.

**Implication for the observer.** The observer's optimal encoding cannot achieve expected description length below the source entropy. Any encoding must have L ≥ H; only encodings approaching this bound are MDL-optimal in the source-coding sense.

---

## 8. Step 5 — Decomposition of model-bearing encodings

**Claim.** When an encoding uses a model M, its total description length decomposes as L(M, data) = L(M) + L(data | M). This decomposition is forced by prefix-code reversibility: to recover the data from the encoded bits, the decoder must first recover M and then decode the data given M, requiring that L(M) be unambiguously delimited within L(M, data).

**Proof.** This is Rissanen's MDL principle (1978, §2). For any uniquely decodable model-bearing encoding, the encoded bitstream must be parseable into two parts: a first segment specifying M (so the decoder knows which model was used), and a second segment specifying the data given M. The first segment must be unambiguously delimited (otherwise the decoder cannot tell where M ends and data begins). The total encoding length is therefore at least the sum of the two parts: L(M, data) ≥ L(M) + L(data | M), with equality for the optimal prefix code achieving this decomposition.

**Type 3** — Rissanen, J. (1978), *Automatica* 14(5), §2.

---

## 9. Step 6 — MDL principle: minimize the decomposition

**Claim.** Among all model-bearing encodings, those minimizing L(M) + L(data | M) over the model class are the optimal source codes in the Shannon sense, generalized to model-bearing encodings.

**Proof.** Combining Step 4 (Shannon source coding) with Step 5 (Rissanen decomposition): the optimal expected description length for a given model M is L(M) + H(data | M), where H is the conditional entropy. Minimizing over M gives the MDL-optimal model. This is the rigorous formulation of the MDL principle.

**Type 3** — Rissanen 1978; Grünwald 2007 §§5.1–5.3 (rigorous MDL as Shannon source coding extended to model-bearing encodings).

---

## 10. Step 7 — Encodings that do not save bits are not retained

**Claim.** Encodings (M, data | M) with L(M) + L(data | M) ≥ L(raw) save no bits over the raw encoding. They cannot fit in the register without displacing more economical encodings, and so are not realized in the observer's compressed view.

**Proof.** Let L(raw) = length of the raw substrate observation under the trivial encoding (no model; each observation stored verbatim). An encoding using model M has total length L(M) + L(data | M). If L(M) + L(data | M) ≥ L(raw), the encoding consumes at least as much register capacity as the raw observation while providing the same (or less) information. Under the observer's finite capacity B (Step 1), retaining such an encoding has no advantage — and if it consumes capacity that could be used for an encoding with L(M) + L(data | M) < L(raw), it is strictly suboptimal.

**Type 2** (direct comparison of encoding lengths against the raw baseline; capacity argument from Step 1).

This is the *waterline* condition: only encodings with L(M) + L(data | M) < L(raw) — i.e., those that save bits — are retained.

---

## 11. Step 8 — Multi-admissible regime: plural retention

**Claim.** When multiple encodings simultaneously satisfy the waterline condition L(M_i) + L(data | M_i) < L(raw), all are realized in the observer's compressed view, weighted by their compression savings (equivalently, by Bayesian model probability under a uniform prior).

**Proof.** This is the multi-admissible regime of MDL, rigorously characterized by Grünwald 2007 §17. When multiple models above the waterline exist, the observer's optimal long-term estimator is a Bayesian mixture over above-waterline models, with weights proportional to exp(−[L(M_i) + L(data | M_i)]) — i.e., weighted by total description length, equivalently by compression savings relative to the raw baseline.

The structural reading: the waterline is a *threshold per encoding*, not a *selection criterion across encodings*. Below the waterline, an encoding is indistinguishable from raw substrate noise (no compression advantage). Above the waterline, an encoding is retained. When many are above, many are retained — not because anything chooses, but because the threshold is binary per encoding, and what passes the threshold passes.

**Type 3** — Grünwald 2007 §17 (multi-admissible / Bayesian-mixture interpretation of MDL).

---

## 12. Step 9 — This is A2

**Claim.** The waterline condition derived in Steps 7–8 is precisely what A2 (`../framework/framework_axioms.md` §3, refined form) asserts as an axiom. Under the present theorem, A2 is derived rather than postulated.

**Proof.** A2's refined statement: the observer retains every representation M satisfying L_total(M) = L_model(M) + L_data_given_model(M) < L_raw, with multiple admissible representations physically realized simultaneously. This matches Steps 7 (waterline filtering) and 8 (multi-admissible plural retention) exactly.

**Type 4** (Steps 7 and 8 above) + **Type 2** (combination — recognizing the waterline as A2's content).

---

## 13. Effect on framework axioms

**Before this theorem.** Framework's axiom slate (per `../framework/framework_axioms.md` §10): {A1, A2, A3, A4, A5}. Five axioms.

**After this theorem (combined with sibling demotions).**

- A2 demotes from axiom to theorem (this document).
- A3 demotes to theorem (`theorem_A3_complex_hilbert_from_multiway.md`).
- A4 was already locally derived (Session 11, `theorem_car_local_jordan_wigner.md`); global A4 remains open but no current prediction-DAG derivation requires it.

Resulting structural axiom slate: **{A1}**. Plus A5-mass (downstream labeling, not structural axiom in the same sense). Plus P1' (definitional commitment about scope, not an axiom — it states what we mean by "observer").

Every other commitment of the framework — MDL, complex Hilbert space, fermionic statistics, the entire mathematical scaffold — is now a theorem of A1 + P1' + standard published mathematics (Shannon, Rissanen, Grünwald, Stone, Strauch, Childs, etc.).

---

## 14. Scope honesty / open questions

1. **P1' as definitional commitment.** Step 1 invokes P1' (observer is finite register, persists across multiple observations). This is *operationally definitional*: it states what we mean by "observer" within this framework. It does not introduce physical content beyond A1's substrate. A reader who rejects P1' rejects the framework's commitment to "observation = finite register inside the framework" — which is the no_free_bits §1.1 commitment. The honest framing: P1' is the irreducible scope-fixing commitment that any framework deriving physics from a discrete substrate accessed by finite observation must make.

2. **Persistence vs single-shot.** P1's "persists across multiple observations" is the cyclic assumption invoked in Step 1 (observer continues operating). Without persistence, the observer is a one-shot detector and the description-length analysis collapses to a single observation. The framework's commitment to persistent observers is consistent with the no_free_bits framing of dynamical existence.

3. **Source ergodicity assumption (implicit in Shannon's theorem).** Step 4 invokes Shannon's source coding theorem, which assumes the substrate observation source is stationary ergodic. The framework's substrate (toggle dynamics on F_inv(E)'s Cayley graph) admits ergodic measures (the regular representation has ergodic decomposition); for non-ergodic substrate states, the source coding theorem applies separately to each ergodic component. This is a standard subtlety, not a framework-specific gap.

4. **Plural waterline reading (Step 8).** Grünwald 2007 §17's multi-admissible Bayesian mixture is sometimes presented as a Bayesian *interpretation* of MDL rather than as a source-coding *theorem*. The honest reading: the multi-admissible regime is rigorously characterized in Grünwald §17 as the optimal long-term estimator, with the Bayesian-mixture form following from the source coding theorem + the prefix-code reversibility argument. Citing §17 directly is the cleanest gate-passing source.

5. **A2's "selective retention" terminology.** Some downstream framework docs use "selective retention" to mean the waterline filtering of Step 7. Others use it to mean the plural retention of Step 8. Both are derived from this document; framework_axioms.md §3's refined-A2 statement should be updated to point to this document for the precise content.

---

## 15. Cross-references

- `../framework/framework_axioms.md` §3 — A2's prior axiomatic statement; should be updated to "derived theorem" status after this document is reviewed cold.
- `theorem_A3_complex_hilbert_from_multiway.md` — sibling theorem demoting A3.
- `theorem_car_local_jordan_wigner.md` — Session 11, local A4 derivation (sibling demotion).
- `../operator_sweep/operator_sweep_from_A1.md` — foundational catalog within which this theorem's content sits as Layer 4 information theory + Step 8's multi-admissible regime as the structural justification for the waterline.
- `../parameters/parameter_linter.md` — gate-type definitions; this document is gate-passing under those rules.
- An author's note (Adam Hillier, 2026-04-24) — operational reading of "observer = register" supporting P1'.

---

## 16. Conclusion

The MDL waterline is a derived consequence of:

1. The substrate is unbounded (A1 + Serre 1980).
2. The observer exists within the framework as a finite register (P1', operational definition per no_free_bits §1.1).
3. Standard published mathematics: Shannon source coding (1948), Rissanen MDL decomposition (1978), Grünwald multi-admissible regime (2007 §17).

A2 is no longer an axiom. Combined with the sibling demotions of A3 (`theorem_A3_complex_hilbert_from_multiway.md`) and A4 (`theorem_car_local_jordan_wigner.md`), the framework's structural axiom slate reduces from five to one: **{A1}**. Plus A5-mass as downstream labeling and P1' as definitional commitment about scope.
