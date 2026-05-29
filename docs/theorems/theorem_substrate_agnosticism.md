# Theorem: Substrate agnosticism — observer-access-determined substrate, canonical Cayley-graph representative

**Date:** 2026-05-08 (companion to `theorem_toggle_from_self_containment.md`).
**Status:** STRUCTURAL-DERIVATION (philosophical / methodological theorem). Establishes that the framework's predictions are invariant under choice of substrate within an observational equivalence class, and that the Cayley graph of F_inv(E) is the description-length-minimal canonical representative of that class. The framework is therefore provably indifferent to substrate metaphysics.
**Depends on:** `theorem_toggle_from_self_containment.md` (supplies F_inv(E) as the observer's question algebra). Standard published mathematics (Kolmogorov 1965 / Solomonoff 1964 / Li & Vitányi 2008 — algorithmic information theory).
**Cross-references:**
- `docs/theorems/theorem_toggle_from_self_containment.md` (Step A — supplies F_inv(E) and the Cayley-graph identification).
- `docs/framework/framework_axioms.md` (to be revised; substrate agnosticism added as derived theorem).
- `docs/framework/framework_architecture.md` (Layer-1/Layer-2 structure; this theorem clarifies the metaphysical status of Layer 1).
- `docs/theorems/theorem_F_inv_E_to_srs_compression.md` (downstream use of canonical-representative reasoning at the srs lattice level).

---

## Statement

Let O be a binary-access observer with question algebra F_inv(E) (per Step A) and let X be a substrate. Define the **response pattern** of X under O as the function

$$R_X : F_{inv}(E) \to \{0, 1\}^{\mathbb{N}}$$

that assigns to each F_inv(E)-element (each finite sequence of binary questions) the corresponding sequence of yes/no answers X provides.

**Theorem (Substrate Agnosticism).** Two substrates X₁, X₂ that yield identical response patterns ($R_{X_1} = R_{X_2}$) are operationally indistinguishable from O — no sequence of binary questions can tell them apart.

The space of substrates is therefore partitioned into **observational equivalence classes** $[X] = \{X' : R_{X'} = R_X\}$.

Within each non-empty equivalence class, the **Cayley graph of F_inv(E)** equipped with the response function read off the equivalence class's common $R$ is the **description-length-minimal representative** (canonical representative).

Any predictions derived on the canonical representative are invariant under choice of substrate within the equivalence class. The substrate-in-itself is therefore underdetermined by observation, and the framework cannot — and need not — distinguish substrate metaphysics within an equivalence class.

**Corollary (discreteness is observer access mode).** Discreteness is a property of the observer's binary access mode (per Step A's Shannon-1-bit-minimum + active-reading), not a property of substrate metaphysics. The framework cannot distinguish digital from analog at the substrate level; both can lie in the same equivalence class.

---

## Setup

By Step A, the observer O's question algebra is F_inv(E) on a finite alphabet E, with each generator T_e an involutive operator T_e² = id, and the observer's primitive update is a binary distinction.

Let $\mathfrak{X}$ denote the (very general) class of structures that can serve as a substrate for O — that is, any mathematical object X equipped with:

1. A *configuration space* $\mathrm{Conf}(X)$.
2. A *question response function* $\rho_X : \mathrm{Conf}(X) \times E \to \{0, 1\}$ giving the binary answer to applying generator T_e at configuration c.
3. A *transition function* $\tau_X : \mathrm{Conf}(X) \times E \to \mathrm{Conf}(X)$ giving the configuration after applying T_e.

These three pieces are the operational interface between X and O. The class $\mathfrak{X}$ includes discrete graphs, continuous manifolds with measurable response functions, smooth bundles, hypergraphs, and any other structure compatible with this interface.

The **response pattern** of X under O is the family of yes/no sequences X produces under all F_inv(E)-question sequences applied at any starting configuration c ∈ Conf(X):

$$R_X(c, w) = \rho_X(\tau_X(c, w_1), w_2) \cdot \rho_X(\tau_X(\tau_X(c, w_1), w_2), w_3) \cdots$$

for $w = w_1 w_2 \cdots \in F_{inv}(E)$ (notation collapsed; the response pattern records the bit sequence produced as the question sequence is executed).

We say two substrates X₁, X₂ are **observationally equivalent under O** iff there exists a configuration-space bijection $\phi : \mathrm{Conf}(X_1) \to \mathrm{Conf}(X_2)$ such that $R_{X_1}(c, w) = R_{X_2}(\phi(c), w)$ for every $c \in \mathrm{Conf}(X_1)$ and every $w \in F_{inv}(E)$.

This is an equivalence relation. Substrates fall into equivalence classes $[X]$.

---

## Proof

### Step 1 — Observer's cumulative record is determined by $R_X$.

By Step A, O's primitive update is a binary distinction (Shannon's 1-bit minimum) and O's algebra of compositions is F_inv(E). O's interaction with X is exhausted by sequences of F_inv(E)-applications and the corresponding bit answers; O has no other access channel to X.

Therefore O's cumulative observation record after any history is the prefix of $R_X(c_0, w)$ corresponding to the question sequence $w$ executed from initial configuration $c_0$. No information about X beyond $R_X$ is available to O.

### Step 2 — Two substrates yielding identical $R$ are operationally indistinguishable.

Suppose $R_{X_1}(c_1, \cdot) = R_{X_2}(c_2, \cdot)$ for some $c_1, c_2$ and all $w \in F_{inv}(E)$. By Step 1, O's cumulative record from initial configurations $c_1$ and $c_2$ respectively is identical for every question history. No sequence of questions can distinguish $X_1$ from $X_2$.

The relation $X_1 \sim X_2$ defined by existence of a configuration bijection $\phi$ such that $R_{X_1} = R_{X_2} \circ \phi$ is reflexive, symmetric, and transitive (standard verification). It is an equivalence relation; the substrate space partitions into equivalence classes $[X]$.

### Step 3 — Each equivalence class is non-empty and admits a canonical Cayley-graph representative.

Fix a response pattern $R : F_{inv}(E) \to \{0, 1\}^{\mathbb{N}}$ that is realizable (i.e., produced by at least one substrate). Construct a candidate substrate $X_R^{\text{Cay}}$ as follows:

- $\mathrm{Conf}(X_R^{\text{Cay}}) = F_{inv}(E)$ — configurations are reduced words.
- $\tau_{X_R^{\text{Cay}}}(w, e) = w \cdot T_e$ — the transition takes the reduced word $w$ to the reduced word $w \cdot T_e$, with reduction performed under $T_e^2 = \mathrm{id}$ if a final $T_e$ cancels.
- $\rho_{X_R^{\text{Cay}}}(w, e)$ is read off $R$ at position $w \cdot T_e$.

By construction, $X_R^{\text{Cay}}$ has the Cayley graph of F_inv(E) as its underlying transition graph (Step A, Step 8) and yields response pattern $R$ exactly. Therefore $X_R^{\text{Cay}} \in [X]$ for the equivalence class $[X]$ corresponding to $R$.

### Step 4 — $X_R^{\text{Cay}}$ is description-length-minimal in $[X]$.

We measure description length by Kolmogorov complexity (Kolmogorov 1965; Solomonoff 1964; Li & Vitányi 2008 §2): $K(X)$ is the length of the shortest program that, given any $(c, w)$, produces $R_X(c, w)$.

For any substrate $X' \in [X]$, the program "construct F_inv(E) on alphabet $E$; construct the Cayley graph; install response function $R$ at each vertex; execute the requested $w$ from $c$" reproduces $R_{X'}$ exactly (because $X' \sim X_R^{\text{Cay}}$). The length of this program is bounded by $K(F_{inv}(E)) + K(R)$ plus a fixed constant (the program scaffold).

For $X' \neq X_R^{\text{Cay}}$ in $[X]$, $X'$ contains additional structure beyond the response pattern (e.g., a smooth-manifold metric, a higher-dimensional embedding, or a coarse-graining map) that does not appear in $R_{X'}$ but must be specified to fix $X'$ as a mathematical object. Specifying this additional structure adds to the description length: $K(X') \geq K(X_R^{\text{Cay}}) + K(\text{extra structure} \mid X_R^{\text{Cay}})$ up to an additive constant.

Therefore $K(X_R^{\text{Cay}}) \leq K(X')$ for all $X' \in [X]$, up to an additive Kolmogorov constant (Li & Vitányi 2008 invariance theorem, Theorem 2.1.1). $X_R^{\text{Cay}}$ is description-length-minimal in $[X]$.

(The additive constant is the language-choice ambiguity inherent in Kolmogorov complexity; it is independent of $X'$ and bounded universally. The minimality holds in the standard up-to-constant sense.)

### Step 5 — Framework predictions on the canonical representative are invariant within $[X]$.

The framework's predictions (Bloch decomposition spectra, Hashimoto walker eigenvalues, MDL waterline content, etc.) are derived as functions of the response pattern $R$ on the Cayley graph of F_inv(E). For any $X' \in [X]$ with $R_{X'} = R$, the same predictions follow from $R$ regardless of which $X'$ supplies it.

Therefore predictions are invariant under choice of substrate within $[X]$. The substrate-in-itself contributes to predictions only through $R$.

### Step 6 — Substrate-in-itself is underdetermined by observation.

By Step 2, two substrates $X_1, X_2 \in [X]$ are operationally indistinguishable. By Step 5, they yield identical predictions. By Step 1, no observation O can perform discriminates them.

Therefore the substrate-in-itself is not determined by observation. The framework's commitment is to the equivalence class $[X]$, not to any particular member.

∎

---

## Corollary (discreteness as observer access mode)

The Cayley graph of F_inv(E) is a discrete combinatorial structure (countably many vertices, finitely many edges per vertex). However, the equivalence class $[X_R^{\text{Cay}}]$ contains substrates of every cardinality and topological type:

- *Discrete substrates*: any quotient of the Cayley graph by an automorphism subgroup that preserves $R$.
- *Continuous substrates*: a smooth manifold with a measurable coarse-graining map onto the Cayley graph that pulls back $R$.
- *Hybrid substrates*: hypergraphs, fibered structures, etc., yielding the same $R$.

All of these are in $[X_R^{\text{Cay}}]$ and are operationally indistinguishable from the Cayley graph by Step 2.

Therefore the apparent discreteness of the framework's substrate is a consequence of the **observer's binary access mode** (Step A, Shannon's 1-bit minimum + active reading), not a metaphysical commitment about the substrate. The framework cannot — and need not — distinguish digital from analog at the substrate level; both lie in the same equivalence class.

This is yet another instance of (A)'s parsimony: nothing about substrate metaphysics is supplied, so nothing about it is committed to. The framework commits only to what observation determines.

---

## Uniqueness and scope

**Uniqueness of the canonical representative.** $X_R^{\text{Cay}}$ is the description-length-minimal representative of $[X]$ up to the standard additive Kolmogorov constant. Within that constant, it is unique: any $X'$ achieving the same minimum complexity is operationally identical to $X_R^{\text{Cay}}$ (different up to a Turing-equivalent reformulation that does not add structural content).

**Scope.** This theorem applies to any binary-access observer whose question algebra is F_inv(E). It does not extend to observers with non-binary access (which would have different question algebras and hence different equivalence classes) — but Step A establishes that any observer satisfying (A) + (B) has F_inv(E) as their algebra, so the scope coincides with the framework's actual scope.

**What this theorem does NOT do.**

- It does not assert that the substrate IS the Cayley graph. It asserts that the framework cannot distinguish the Cayley graph from any other substrate in $[X_R^{\text{Cay}}]$, and that the Cayley graph is the canonical (minimal-description) representative.
- It does not derive predictions. The framework's quantitative predictions live elsewhere; this theorem clarifies the metaphysical status of the substrate on which those predictions are derived.
- It does not address whether the framework's actual response pattern $R$ is realized in our universe — that is the empirical content of the framework, addressed by the prediction files matching observation.

---

## Consequences

**1. Layer 1's metaphysical status clarified.** `framework_architecture.md` Layer 1 (the multiway substrate) is the framework's working representative of $[X]$ — namely, the Cayley graph of F_inv(E). The framework operates ON this representative; predictions are derived ON this representative; observation tests predictions ON this representative. The framework does not claim Layer 1 is the substrate-in-itself. The substrate-in-itself is whatever member of $[X]$ underlies our universe; the framework is provably indifferent to which member.

**2. Defense against discreteness objections.** Hostile readings of the framework that object "the universe is not discrete" are addressed: the framework does not commit to discreteness of the substrate; it commits to discreteness of the observer's record, which is a Shannon consequence of binary access. The substrate-in-itself may be continuous; the framework cannot tell, and its predictions do not depend on which it is.

**3. "It from bit" structurally operationalized.** Wheeler 1989's "it from bit" conjectured that information is fundamental. This theorem operationalizes the conjecture: the framework's substrate is determined by observation up to an equivalence class of which the canonical representative is a graph indexed by binary distinctions. Information (binary access) is fundamental in the framework not by metaphysical preference but by structural necessity — it is what is left when substrate metaphysics is refused.

**4. Recursive "no free bits" applied to substrate metaphysics.** Each layer of the framework refuses to commit beyond what observer access forces. Refusing to commit to substrate metaphysics is one such layer, and the present theorem makes the refusal formal: there is no commitment to make, because observation cannot distinguish.

---

## Remarks

**On the formality of the canonical representative.** Step 4 uses Kolmogorov complexity (Kolmogorov 1965 / Li & Vitányi 2008) to define description-length-minimality. This is a standard, language-independent (up to additive constant) measure. Alternatives — minimum description length on a fixed universal language (Rissanen 1978 / Grünwald 2007), Solomonoff prior (Solomonoff 1964) — yield equivalent canonical representatives up to language-additive-constants. The choice of complexity measure does not affect the theorem's content.

**On the relationship to (A) + (B).** This theorem does not require new commitments beyond (A) and (B). It is a consequence of: (a) (B)'s specification of the observer's access mode (binary, via Step A), (b) standard algorithmic information theory, and (c) (A)'s prohibition on supplying anything from outside (which forbids commitments to substrate metaphysics that observation cannot reach).

**On the framework's pitch.** The framework's claims about the substrate are claims about $[X]$, not about a specific X. This is honest underdetermination: the framework predicts what an observer accesses and refuses to commit to what observation cannot reach. The Cayley graph is the canonical representative used for derivation; it is not asserted as the substrate-in-itself.

**On substrate-*net* selection vs channel multiplexing (added 2026-05-12; bears on R-9).** Two distinct MDL operations should not be conflated. (i) **Channel multiplexing** — `channel_select` in the A2-T-waterline machinery (`theorem_dark_correction_mdl.md` Lemma 1; `feedback_waterline_not_minimum_canonical_distinction.md`): within a *fixed* structural model, the observer keeps *every* above-waterline channel — different operators/functionals coupling to *different* observables — each physically realized. (ii) **Substrate-net selection** — choosing the spatial crystal net the observer's compression resolves to (srs vs srs-z vs srs-c4 vs …; these are derived downstream of this theorem via `d_spatial_derivation.md`, `k_star_derivation.md`, `g_girth_derivation.md`). Candidate nets make *competing* predictions for the *same* observables (V_us, η_B, m_ν, …), so they are **competing whole-substrate hypotheses**, not distinct channels — `channel_select` does not apply. The right operation is exactly Step 4 above: pick the **Kolmogorov-minimal description of the observed data** ($K(X) = $ shortest program producing $R_X$), which is $\text{DL}_{\text{model}} + \text{DL}_{\text{data}\mid\text{model}}$ minimized — the MAP hypothesis, not a $\text{DL}_{\text{model}}$-weighted superposition of every above-waterline net. (Treating the substrate-net as a $\text{DL}_{\text{model}}$-only Boltzmann ensemble — as one historical R-9 audit did — is `channel_select` misapplied; it spuriously gives subdominant nets like srs-z weight ~0.2 and "breaks PDG", whereas the full-$K$ posterior is sharply peaked on srs.) **And the substrate-net selection does not even need the data term:** by the remark "On (A) applied to spatial structure" in `theorem_toggle_from_self_containment.md`, (A)'s no-privilege applied to spatial directions/orientations makes the observer's model strongly isotropic (arc-transitive); by Sunada 2012 the strongly-isotropic 3-regular 3-connected ℝ³ crystal net is unique = srs; so the substrate-net is srs *structurally*, with the data fit (only srs reproduces the SM) as supplementary confirmation. That is the closure of R-9 (`docs/audits/registers/structural_residue_register.md`).

**On comparison to philosophical realism.** This is structurally honest indifference, not philosophical anti-realism. The framework does not deny the substrate has metaphysical properties; it asserts that the framework's predictions are invariant under those properties within the equivalence class. A philosophical realist may consistently hold that one specific X ∈ [X_R^{Cay}] is "the real" substrate; the framework is silent on this question and its predictions are the same either way.

---

## References

### Cited published theorems

- Kolmogorov, A. N. (1965). Three approaches to the quantitative definition of information. *Problems of Information Transmission* 1(1), 1–7. (Definition of Kolmogorov complexity.)
- Solomonoff, R. J. (1964). A formal theory of inductive inference. *Information and Control* 7(1), 1–22; 7(2), 224–254. (Universal prior; algorithmic probability.)
- Li, M., & Vitányi, P. (2008). *An Introduction to Kolmogorov Complexity and Its Applications* (3rd ed.). Springer. §2.1 (invariance theorem; complexity is well-defined up to additive constant).
- Wheeler, J. A. (1989). Information, physics, quantum: the search for links. In *Proceedings of the 3rd International Symposium on the Foundations of Quantum Mechanics in the Light of New Technology*. Tokyo. (Cited as prior art for "it from bit" lineage; conceptual reference, not load-bearing for the proof.)

### Framework documents

- `docs/theorems/theorem_toggle_from_self_containment.md` (Step A — supplies F_inv(E), the Cayley-graph identification, and the binary-access setup).
- `docs/framework/framework_axioms.md` (to be revised after this theorem lands).
- `docs/framework/framework_architecture.md` (Layer 1 metaphysical status clarified by this theorem).
- `docs/theorems/theorem_F_inv_E_to_srs_compression.md` (downstream use of canonical-representative reasoning at the srs lattice level).
