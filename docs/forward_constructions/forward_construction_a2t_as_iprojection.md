# A2-T as I-Projection — forward-construction cross-validation

**Date:** 2026-04-26.
**Status:** Forward-construction result. **Cross-validation finding** (category-2 yield per the operator-sweep search-instrument rubric). First Tier 1 deliverable in the substrate quantum-information cluster (`../framework/framework_qft_ontology.md` §8).
**Source op:** §4.25 conditional expectation (in `../operator_sweep/operator_sweep_from_A1.md`); audited at `../operator_sweep/operator_sweep_audit_layer_4.md` §4.25.

---

## Question

The operator-sweep audit (Layer 4, §4.25) flagged conditional expectation as a candidate cross-validation route for A2-T's MDL canonicalization. The investigation question:

**Is A2-T's MDL canonicalization a conditional expectation in the appropriate σ-algebra?**

If yes: A2-T inherits the L²-orthogonality structure, tower property, and idempotence of conditional expectation — gaining a different mathematical apparatus and cross-validating the theorem.

If no: there is a different mathematical apparatus that fits A2-T, and the cross-validation surfaces it.

This document answers the question directly.

---

## Result (preview)

**A2-T's MDL canonicalization is NOT the standard L²-conditional expectation. It IS the *information projection* (I-projection) of Csiszár 1975.**

I-projection minimizes KL divergence (description-length excess) onto a family of distributions; conditional expectation minimizes L²-distance onto a sub-σ-algebra. These are different operations under different optimization criteria.

The cross-validation succeeds in a *generalized* sense:
- A2-T inherits **idempotence** ✓
- A2-T inherits a **Pythagorean theorem** (Csiszár 1975 Theorem 2.2) — the I-projection analog of L²-orthogonality
- A2-T inherits a **generalized tower property** (Csiszár-Matuš 2003) under nested exponential families
- A2-T does NOT inherit **L²-orthogonality** (which is unique to squared loss)
- A2-T does NOT inherit **linearity in input** (I-projection is non-linear)

**Net structural finding:** A2-T's MDL canonicalization is a well-studied mathematical object in information geometry — the I-projection — with rigorous theory developed by Csiszár, Matuš, and collaborators since 1975. The framework's compression apparatus inherits this entire body of mathematical structure.

---

## 1. Setup — A2-T canonicalization

Per `../theorems/theorem_A2_mdl_from_finite_register.md`:

- **Substrate state space.** Ω = F_inv(E) (substrate observation sequences); equipped with an underlying source distribution P.
- **Observer's finite register.** R, with capacity B, holding 2^B possible states.
- **Encoding.** A function f: Ω → R; equivalently, a partition of Ω into 2^B blocks.
- **Description length.** For a model-bearing encoding (M, data | M): L(M) + L(data | M); the optimal expected length under M's induced distribution Q_M is H(P) + D(P ‖ Q_M) where D is KL divergence (Cover-Thomas 2006 §5.3).
- **MDL canonicalization.** The encoding (model M*) that minimizes L(M) + L(data | M) over the model class 𝒞.
- **Waterline.** The threshold L(M) + L(data | M) < L(raw); only encodings clearing the threshold are retained.

The canonicalization is therefore:

$$M^* = \arg\min_{M \in \mathcal{C}} \big[L(M) + L(\text{data} \mid M)\big]$$

For the optimal prefix code on Q_M-distributed data, L(data | M) is approximately the cross-entropy −log Q_M(data); averaged over the source, the expected description length under M is H(P) + D(P ‖ Q_M).

So the expected-description-length minimization becomes:

$$Q^* = \arg\min_{Q \in \mathcal{Q}} \big[L_{\text{model}}(Q) + D(P \,\|\, Q)\big]$$

For a fixed model class 𝒬 with negligible model-cost variation L_model(Q) ≈ const (or a maximum-likelihood / minimum-KL approximation regime), this reduces to:

$$Q^* = \arg\min_{Q \in \mathcal{Q}} D(P \,\|\, Q)$$

This is exactly the **I-projection** of P onto 𝒬.

---

## 2. Classical conditional expectation — L²-best projection

The standard conditional expectation E[X | 𝒢] satisfies:

$$E[X \mid \mathcal{G}] = \arg\min_{Y \in L^2(\Omega, \mathcal{G}, P)} \mathbb{E}_P[(X - Y)^2]$$

It is the **L²-orthogonal projection** of X onto the closed subspace of 𝒢-measurable square-integrable functions. Its defining properties (standard probability theory; Folland 1999 §3, Cover-Thomas 2006 §B):

- **Linearity:** E[αX + βY | 𝒢] = α E[X | 𝒢] + β E[Y | 𝒢].
- **Tower:** E[ E[X | 𝒢] | ℋ] = E[X | ℋ] for ℋ ⊂ 𝒢.
- **Idempotence:** E[ E[X | 𝒢] | 𝒢] = E[X | 𝒢].
- **L²-Pythagorean:** ‖X‖² = ‖X − E[X | 𝒢]‖² + ‖E[X | 𝒢]‖².
- **Best-approximation:** the L²-distance from X to E[X | 𝒢] is the minimum L²-distance from X to any 𝒢-measurable function.

These properties characterize the L²-projection. The optimization criterion is squared error.

---

## 3. I-projection — KL-best projection

For a probability measure P and a closed convex family 𝒬 of probability measures on Ω, the **information projection** of P onto 𝒬 is:

$$P^* = \arg\min_{Q \in \mathcal{Q}} D(P \,\|\, Q)$$

where D(P ‖ Q) = E_P[log(P/Q)] is KL divergence.

I-projection was introduced by Csiszár 1975 (*Annals of Probability* 3, 146–158) and has been developed extensively since, particularly by Csiszár & Matuš (e.g., Csiszár-Matuš 2003, *IEEE Trans. Inf. Theory* 49(7), 1474–1490).

Key properties (Csiszár 1975 §2):

- **Existence + uniqueness:** if 𝒬 is closed and convex and the I-projection exists (D(P ‖ Q) < ∞ for some Q ∈ 𝒬), it is unique.
- **Pythagorean inequality (Csiszár 1975 Theorem 2.2):** for any Q ∈ 𝒬:

$$D(P \,\|\, Q) \geq D(P \,\|\, P^*) + D(P^* \,\|\, Q)$$

  Equality holds if and only if 𝒬 is an exponential family containing Q. This is the I-projection's analog of the L²-Pythagorean.

- **Idempotence:** I-projection of P* onto 𝒬 is P* itself (P* is already in 𝒬).
- **Tower under nested exponential families (Csiszár-Matuš 2003):** for ℋ ⊂ 𝒢 nested exponential families, sequential I-projection (project onto 𝒢, then onto ℋ) equals direct I-projection onto ℋ. **This is a generalized tower property.**
- **NOT L²-orthogonal.** I-projection minimizes KL, not L². The L²-Pythagorean does not hold.
- **NOT linear in P.** D(αP_1 + βP_2 ‖ Q) ≠ α D(P_1 ‖ Q) + β D(P_2 ‖ Q) (KL is convex in the first argument but not linear).

---

## 4. Identification — A2-T canonicalization IS I-projection

**Claim.** A2-T's MDL canonicalization Q* = arg min_{Q ∈ 𝒬} D(P ‖ Q) is the I-projection of the source distribution P onto the model family 𝒬.

**Proof.** From Section 1 of this document, A2-T's expected-description-length minimization is:

$$Q^* = \arg\min_{Q \in \mathcal{Q}} D(P \,\|\, Q)$$

(under the negligible-model-cost regime, or with model cost absorbed into the structure of 𝒬). This is the definition of the I-projection of P onto 𝒬 (Csiszár 1975, equation 2.1). ∎

**Note on model cost.** When L_model(Q) is non-negligible, A2-T's full minimization is L_model(Q) + D(P ‖ Q). This is the I-projection onto a *weighted* family — Csiszár 1975 §3 covers this case as the "minimum-divergence" generalization. The cleanest framing: A2-T is the I-projection onto 𝒬 with model-cost regularizer.

---

## 5. Properties inherited by A2-T from I-projection theory

By the identification of Section 4, A2-T's MDL canonicalization inherits the following structural properties from I-projection theory (Csiszár 1975, Csiszár-Matuš 2003):

### 5.1 Idempotence (already implicit in A2-T)

Applying A2-T's canonicalization to an already-canonicalized state gives the same result. Csiszár 1975 idempotence makes this explicit: I-projection of Q* onto 𝒬 is Q*, since Q* ∈ 𝒬 by construction.

**Framework consequence:** the framework's "MDL waterline is stable" intuition is now a theorem of I-projection theory, not a side claim.

### 5.2 Pythagorean inequality — A2-T's analog of L²-orthogonality

For any model Q ∈ 𝒬:

$$D(P \,\|\, Q) \geq D(P \,\|\, Q^*) + D(Q^* \,\|\, Q)$$

with equality for exponential families.

**Framework consequence:** the "compression cost" decomposes additively into "irreducible compression cost to canonicalization Q*" plus "extra cost of using a sub-optimal Q over Q*". This is structurally analogous to the L²-decomposition ‖X‖² = ‖X − E[X | 𝒢]‖² + ‖E[X | 𝒢]‖² but with KL-divergence in place of squared error.

### 5.3 Generalized tower property under nested exponential families

For nested exponential families 𝒬_ℋ ⊂ 𝒬_𝒢, sequential I-projection (onto 𝒬_𝒢, then onto 𝒬_ℋ) equals direct I-projection onto 𝒬_ℋ (Csiszár-Matuš 2003 Theorem 6).

**Framework consequence:** sequential MDL canonicalizations through a hierarchy of compression levels (e.g., substrate → coarser substrate → coarsest substrate) factor through the hierarchy. This is the framework's "tower of compression levels" intuition, now a theorem when the levels are exponential families.

**Caveat:** the nested-exponential-family condition is a real restriction. Generic model families are NOT exponential. The framework's specific compression families need to be checked: do they form an exponential family? Typically yes for Markov-type substrate dynamics (the toggle-rate Markov chain is exponential), but explicit verification is a follow-up open question.

### 5.4 Existence + uniqueness

For closed convex model families 𝒬, the I-projection Q* exists and is unique whenever D(P ‖ Q) < ∞ for some Q ∈ 𝒬.

**Framework consequence:** A2-T's canonicalization is well-defined and unique under standard regularity assumptions on the model family. The framework's implicit assumption of unique canonicalization gains an explicit existence theorem (Csiszár 1975).

---

## 6. Properties NOT inherited

The cross-validation does NOT extend the framework's apparatus to:

### 6.1 Linearity (in P)

L² conditional expectation is linear: E[αX + βY | 𝒢] = α E[X | 𝒢] + β E[Y | 𝒢].

I-projection is NOT linear: D(αP_1 + (1−α)P_2 ‖ Q) is convex in the first argument (Csiszár 1975 Lemma 1) but not linear. The I-projection of a mixture is not generally the mixture of I-projections.

**Framework consequence:** A2-T's canonicalization does not commute with linear superposition of source distributions. This is structural — when the framework considers superposed sources, the canonicalization must be applied to the full superposition, not to components separately.

### 6.2 L²-Pythagorean / orthogonality

The L²-Pythagorean ‖X‖² = ‖X − E[X | 𝒢]‖² + ‖E[X | 𝒢]‖² is unique to squared loss. The framework's compression cost decomposes via the *Csiszár-Pythagorean* (Section 5.2 above), not the L²-Pythagorean.

### 6.3 Universal tower property

L² tower (E[E[X | 𝒢] | ℋ] = E[X | ℋ] for ℋ ⊂ 𝒢) holds for ANY nested σ-algebras. I-projection tower requires the model families to form *nested exponential families*. The framework gains a *restricted* tower property, not the universal one.

---

## 7. Implications for framework

### 7.1 New mathematical apparatus available

A2-T can now be analyzed using I-projection theory. Specifically:

- **Csiszár 1975** (*Annals of Probability* 3, 146–158) — foundational paper; existence, uniqueness, Pythagorean.
- **Csiszár-Matuš 2003** (*IEEE Trans. Inf. Theory* 49, 1474–1490) — generalized tower; convergence; Sanov's theorem connections.
- **Amari & Nagaoka 2000** (*Methods of Information Geometry*, AMS) — information-geometric formulation; α-divergences; dually flat structures.

These provide a mathematically rigorous foundation for the framework's compression apparatus, with theorems on:
- **Sanov's theorem** (large deviations of empirical distributions; relates to the framework's "rare-event" predictions).
- **Information-geometric duality** (the framework's compression structure has a *dually flat* geometry in the sense of Amari-Nagaoka, which could organize multi-level canonicalization).
- **Maximum-entropy duality** (I-projection onto a maximum-entropy family equals maximum-entropy with constraints; underlies Stage 2a's use of Jaynes maximum-entropy).

### 7.2 Connection to Stage 2a

Stage 2a (`../theorems/theorem_edge_surprise_thresholds.md`) derives p_create = 1/2, p_destroy = 1/3 via Bayesian Beta posterior + Jaynes maximum-entropy. Maximum-entropy is mathematically dual to I-projection (Csiszár 1975 §3): maximum-entropy with constraints = I-projection from uniform onto the constraint family.

**Cross-validation:** Stage 2a's maximum-entropy derivation is the I-projection of the uniform substrate distribution onto the constraint family. A2-T's MDL canonicalization is the I-projection of the source distribution onto the model family. **Both are I-projections, with different sources and different constraint families.**

This is a structural unification: Stage 2a and A2-T are both I-projection operations, with the difference being in choice of source (uniform vs source distribution) and constraint family (edge-surprise vs MDL model class). The framework's information-geometric apparatus is more unified than its previous presentation suggested.

### 7.3 Connection to A.15 martingales (Tier 1 cluster, next)

I-projection has natural martingale formulation: the sequence of I-projections through nested filtrations is an information-theoretic martingale (Csiszár-Matuš 2003 §5). This connects Tier 1 cluster ops 4.25 (this doc) and A.15 (martingales on multiway filtration).

**Forward-construction follow-up:** the framework's MDL canonicalization through nested observation sequences is an I-projection martingale. This is the substrate analog of "Noether currents from time-translation invariance" — A.15's ontological grounding from the appendix audit. The next forward-construction op is to make this explicit.

### 7.4 Connection to A.5–A.6 operator algebras (Tier 1 cluster)

I-projection on commutative probability spaces extends to non-commutative probability spaces via Umegaki 1962 (*Kodai Math. Sem. Rep.* 14, 59–85): the *non-commutative I-projection* of a state on a von Neumann algebra onto a sub-algebra. For F_inv(E)'s group von Neumann algebra L(F_inv(E)) (Appendix A.6), the substrate-analog of I-projection lives there.

**Forward-construction follow-up:** A2-T's quantum analog is a non-commutative I-projection on L(F_inv(E)). This connects directly to the substrate quantum-information cluster (§5.34–§5.38).

---

## 8. Honest scope

1. **Negligible-model-cost regime.** Section 4's identification used the regime where L_model(Q) is approximately constant across Q ∈ 𝒬. The full A2-T includes model cost as a regularizer; the cross-validation extends to the full case via Csiszár 1975 §3, but this is a notational extension, not a separate mathematical content.

2. **Exponential-family condition for tower (Section 5.3).** The tower property requires nested exponential families. The framework's specific compression families (Markov-substrate, MDL-model-class) need explicit checking. Toggle-rate Markov is exponential (the canonical example); other families may not be. Open question: which framework compression hierarchies satisfy the exponential-family condition?

3. **Cross-validation scope.** This document establishes that A2-T is an I-projection. It does NOT re-derive A2-T from scratch via I-projection theory (that would require redoing the substrate-Shannon-Rissanen-Grünwald chain). The cross-validation is *interpretive* — A2-T's existing derivation is sound; the new finding is that the resulting canonicalization is an I-projection.

4. **L²-orthogonality is genuinely lost.** Section 6.2's negative finding is structural, not a defect of the cross-validation. A2-T does not have an L²-Pythagorean; the right structure is the Csiszár-Pythagorean. Anyone hoping that A2-T's compression cost would decompose as L²-distance squared is using the wrong tool.

5. **Tier 1 forward-construction sequence.** This is the first op in the Tier 1 quantum-information cluster. The result enables follow-ups for A.15 martingales (Section 7.3) and A.5–A.6 non-commutative I-projection (Section 7.4). Each of those is its own focused investigation; this document does not produce them.

---

## 9. Status

**Cross-validation succeeded** in the generalized sense: A2-T's MDL canonicalization is the I-projection (Csiszár 1975) onto the model family. Idempotence, Csiszár-Pythagorean, restricted tower property, existence + uniqueness all inherit. L²-orthogonality and full linearity do not.

**Category:** category-2 yield per the operator-sweep search-instrument rubric. Forward-construction direction §4.25 closed at first-pass.

**Effect on framework:** A2-T's compression apparatus now has rigorous information-geometric foundation (Csiszár, Matuš, Amari-Nagaoka). Stage 2a is unified with A2-T as different I-projections. Tier 1 cluster opens up for follow-up ops (A.15 martingales, A.5–A.6 non-commutative I-projection).

**No new SM-matching prediction.** The cross-validation produces structural understanding, not a new numerical prediction. This is consistent with the rubric: category-2 yields are valuable because they raise confidence in existing derivations and surface mathematical apparatus for downstream theorems.

---

## 10. Cross-references

- `../theorems/theorem_A2_mdl_from_finite_register.md` — A2-T's full derivation; this document cross-validates it.
- `../theorems/theorem_edge_surprise_thresholds.md` — Stage 2a; structurally unified with A2-T as I-projection (Section 7.2).
- `../operator_sweep/operator_sweep_audit_layer_4.md` §4.25 — original op flag.
- `../framework/framework_qft_ontology.md` §6 (information-theoretic), §8 (Tier 1 program).
- `../operator_sweep/operator_sweep_audit_appendix.md` §A.15 — martingales follow-up.

**Type 3 (cited published) references for I-projection theory:**

- **Csiszár, I.** (1975). I-divergence geometry of probability distributions and minimization problems. *Annals of Probability* 3(1), 146–158. §2 (Pythagorean), §3 (existence, uniqueness, weighted version).
- **Csiszár, I. & Matuš, F.** (2003). Information projections revisited. *IEEE Transactions on Information Theory* 49(7), 1474–1490. §5 (martingale convergence), Theorem 6 (generalized tower under nested exponential families).
- **Amari, S. & Nagaoka, H.** (2000). *Methods of Information Geometry.* American Mathematical Society / Oxford University Press. (Information-geometric formulation; dually flat structures.)
- **Umegaki, H.** (1962). Conditional expectation in an operator algebra IV. *Kodai Mathematical Seminar Reports* 14, 59–85. (Non-commutative I-projection on von Neumann algebras.)

All citations are to peer-reviewed published work.

---

## 11. Next forward-construction steps

The Tier 1 cluster's natural follow-up sequence after this result:

1. **A.15 martingales** — Csiszár-Matuš martingale formulation of nested I-projections on the substrate's multiway filtration. Estimated 1–2 sessions.
2. **A.5–A.6 non-commutative I-projection** — Umegaki's quantum extension, applied to L(F_inv(E)) (group von Neumann algebra). Bridges to the §5.34–§5.38 quantum thermal/information cluster. Estimated 2–3 sessions.
3. **§5.34–§5.38 substrate thermal apparatus** — quantum partition function, vN entropy, thermal density on substrate Hilbert space. Pairs with A.7 KMS states. Estimated 2–3 sessions.
4. **A.4 Atiyah-Singer index** — graph-Dirac index for substrate fermion-anomaly accounting. Estimated 2–3 sessions.

Total Tier 1 program: ~8–13 focused sessions. This document closes the foundational op (§4.25) in 1 session.
