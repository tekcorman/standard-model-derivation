# Operator Sweep Audit — Appendix

**Date:** 2026-04-26.
**Status:** Per-operation audit. Final layer of the catalog sweep. By construction every Appendix op is unused (the Appendix is the operator-sweep's curated list of permitted-but-not-invoked operations). Deliverables per op: search-instrument application sketch + ontological grounding + forward-construction priority.
**Source catalog:** `operator_sweep_from_A1.md` Appendix.
**Predecessors:** `operator_sweep_audit_layer_0_1.md`, `_2.md`, `_3.md`, `_4.md`, `_5.md`, `_6.md`.

## Methodology adjustment for Appendix

The audit lens collapses (every op is unused by construction). The two remaining lenses are:
- **Search-instrument application** — what would this op produce when applied to substrate? Concrete candidate output (with honesty about what's first-order vs requires real computation).
- **Ontological grounding** — what is this op IN THE SUBSTRATE? What QFT-postulated object does it inform?

Plus a fourth lens added for the Appendix:
- **Forward-construction priority** — high / medium / low based on (a) match to existing framework apparatus, (b) tractability for a focused 1–3 session investigation, (c) ontological reach.

The handoff flags **A.1 group cohomology**, **A.16 modular forms**, **A.4 Atiyah-Singer index** as the three highest-leverage candidates per the Appendix's own honest verdict. This audit confirms or revises those rankings.

---

## Topological / homological (4 ops)

### A.1 — Group cohomology H^n(F_inv(E); ℤ)

**Search-instrument.** F_inv(E) = *_{e ∈ E} ℤ/2 is the free product of |E| copies of ℤ/2. Mayer-Vietoris for free products gives:

H^n(F_inv(E); ℤ) ≅ ⊕_{e ∈ E} H^n(ℤ/2; ℤ) for n ≥ 1, with H^0 = ℤ.

H^*(ℤ/2; ℤ) is classical: H^0 = ℤ, H^{2k} = ℤ/2 for k ≥ 1, H^{odd} = 0. So H^n(F_inv(E); ℤ) for n ≥ 1 is (ℤ/2)^{|E|} for n even, 0 for n odd.

For the framework's |E| = 6 (the 6 undirected edges per srs primitive cell): H^{2k}(F_inv(E); ℤ) = (ℤ/2)^6 for k ≥ 1. Six copies of ℤ/2 is exactly the rank of the abelianization (Layer 1.10 ↔ 4.21) — recovering structure already at Layer 1.

**Concrete first-pass output.** Six independent ℤ/2-valued cohomology classes per even degree. These are *integer invariants* of the substrate.

**Compressibility check.** Six independent ℤ/2 classes → 6 bits per cohomology degree. Low MDL. Could correspond to substrate-level discrete charges.

**SM observable check.** Six is suggestive — it equals (color × generation) = 3 × 3 (off by half) or (left + right) × generations in some accountings. *Not a direct match*; would need more structure (the (ℤ/2)^6 doesn't naturally split into 3 × 2 the way SM charges do).

**Ontological grounding.** **Substrate:** the substrate's "topological data" — what survives compression to homotopy. **Why this form:** Mayer-Vietoris on free products is mechanical given the substrate's group structure. **QFT ground:** anomaly coefficients in QFT are integer invariants — group cohomology of the *gauge group* is the standard provenance (e.g., H^4(BG; ℤ) for Chern-Simons levels). The substrate's H^*(F_inv(E); ℤ) is a *different* cohomology — it lives on the substrate group itself, not the gauge group — and would inform a substrate-level anomaly accounting that doesn't have a direct standard QFT analog.

**Forward-construction priority.** **Medium.** Provides clean integer invariants (6-fold ℤ/2 structure) but the first-pass output doesn't obviously match SM charges. Worth a focused 1-2 session investigation to see whether higher cohomology degrees or twisted coefficients (H^*(F_inv(E); ℤ/2), H^*(F_inv(E); 𝕜) for various 𝕜) produce more structured invariants. Lower-priority than the handoff suggested because the first-pass output is too uniform.

### A.2 — Classifying space BF_inv(E)

**Search-instrument.** BF_inv(E) is the Eilenberg-MacLane space K(F_inv(E), 1). For free products of finite groups, the classifying space is the wedge of classifying spaces: BF_inv(E) ≃ ⋁_{e ∈ E} BℤZ/2 ≃ ⋁_{e ∈ E} ℝP^∞.

**Concrete first-pass output.** A wedge of |E| copies of ℝP^∞. For the framework's |E| = 6: a wedge of 6 ℝP^∞'s. Topologically simple but aspherical.

**Compressibility check.** Aspherical = π_n = 0 for n ≥ 2 → all topological information lives in π_1 = F_inv(E). The classifying space is determined by F_inv(E) itself; no new information beyond what's already in the group.

**Ontological grounding.** **Substrate:** the topological "shape" of the substrate's symmetry group. **QFT ground:** classifying spaces appear in characteristic-class theory (BG for principal G-bundles); in QFT they ground topological terms in the action (Chern-Simons, theta angles). The substrate's BF_inv(E) is a wedge of projective spaces — not a familiar QFT target.

**Forward-construction priority.** **Low.** Asphericity means BF_inv(E) is determined by F_inv(E) itself; no new information. Useful only as packaging for A.1 results.

### A.3 — K-theory K_*(C*_red(F_inv(E)))

**Search-instrument.** Pimsner-Voiculescu six-term exact sequence computes K-theory of free products of C*-algebras. For F_inv(E) = *_{e ∈ E} ℤ/2:
- K_0(C*_red(ℤ/2)) = ℤ² (two characters)
- K_1(C*_red(ℤ/2)) = 0
- Free product gives K_0(C*_red(F_inv(E))) = ℤ + (ℤ/2)^{|E|} (rough; exact computation requires care).

**Concrete first-pass output.** Integer invariants combining group cohomology data with operator-algebraic structure. Provides a different lens than A.1.

**Compressibility check.** Same scale as A.1 — order-|E| integer invariants.

**Ontological grounding.** **Substrate:** operator-algebraic invariants of the substrate. **QFT ground:** K-theory in QFT classifies D-brane charges and topological insulator phases. K_0(C*_red(F_inv(E))) would inform whether the substrate has any analog of these "integer-classified topological phases".

**Forward-construction priority.** **Medium.** Like A.1, gives integer invariants. Co-priority with A.1; might be investigated together.

### A.4 — Atiyah-Singer index theorem for Hashimoto-as-elliptic

**Search-instrument.** Atiyah-Singer requires:
1. An elliptic operator on a smooth manifold.
2. The continuum-limit smooth-manifold structure.

The framework has neither directly: the Hashimoto operator is on a *discrete* graph (not a smooth manifold), and §C smooth-manifold closure is partial. So Atiyah-Singer in its classical form does not apply.

A *substrate-side* analog: discrete index theory (Sunada-style for graphs), where the index of a graph Dirac operator counts spectral asymmetry. For srs, the graph Dirac operator (built from JW + Cl(6;ℂ) spinor at each node) has index counted by:
index(D) = dim ker D - dim ker D†

For a finite primitive cell, this is finite. For the Bloch-decomposed family D(k), the family-index may be non-trivial.

**Concrete first-pass output.** A graph-theoretic Dirac index for srs at the P-point. Computation-heavy but tractable; would yield an integer.

**Compressibility check.** A single integer fermion-anomaly-style coefficient. High compressibility.

**Ontological grounding.** **Substrate:** spectral asymmetry of the substrate Dirac operator. **Why this form:** discrete analog of Dirac index counting chiral excess. **QFT ground:** axial anomaly / fermion-number anomaly in QFT — the integer that distinguishes left-from-right-handed fermion counts. The substrate could ground the SM's chiral structure (left-handed doublets, right-handed singlets, ν_R-forced) in a substrate-Dirac-index computation.

**Forward-construction priority.** **High** (matching handoff's verdict, despite §C partial closure). The graph-theoretic version is tractable; doesn't require smooth-manifold limit; would directly address the SM's chirality structure. Estimated 2-3 sessions for a focused investigation.

---

## Operator algebra constructions (3 ops)

### A.5 — Reduced group C*-algebra C*_red(F_inv(E))

**Search-instrument.** Standard operator algebra. C*_red(F_inv(E)) is generated by the left regular representation acting on L²(F_inv(E)). Different perspective than spectral analysis: focuses on the operator algebra structure rather than spectra.

**Concrete first-pass output.** A non-amenable simple C*-algebra (since F_inv(E) is non-amenable for |E| ≥ 2). Has known KMS structure.

**Compressibility check.** Non-amenable simple C* has constrained representation theory (Powers' factor classification). Operator-algebraic content is rich but largely orthogonal to framework's spectral focus.

**Ontological grounding.** **Substrate:** operator algebra of the substrate's symmetry algebra. **QFT ground:** algebraic QFT (Haag-Kastler) uses C*-algebras as the primary ontology. The substrate's C*_red would ground the operator-algebraic formulation of QFT.

**Forward-construction priority.** **Medium-low.** Mathematical lens; would systematize Layers 1.7, 1.8, 2.14 (left/right action, conjugation, right regular rep). Bundle with A.6.

### A.6 — Group von Neumann algebra L(F_inv(E))

**Search-instrument.** Type II_1 factor for non-amenable F_inv(E) (which holds for |E| ≥ 2). Has a unique tracial state τ. Computable invariants include free entropy, ℓ²-Betti numbers.

**Concrete first-pass output.** A type II_1 factor with known free-probability structure (Voiculescu-Dykema-Nica). For free products of ℤ/2's, the L(F_inv(E)) is the *interpolated free group factor* L(𝔽_t) for some t ≥ 1.

**Compressibility check.** Free-probability invariants are highly structured (free entropy is a specific number).

**Ontological grounding.** **Substrate:** type II_1 factor of the substrate symmetry. **Why this form:** non-amenability is a structural property of free products of ℤ/2 with |E| ≥ 2. **QFT ground:** type II_1 factor structure underlies Jones-index / subfactor theory in QFT — appears in 2D conformal field theory, Connes' noncommutative geometry. The substrate's type II_1 structure could ground Jones-index-style invariants.

**Forward-construction priority.** **Medium.** Free-probability invariants are computable for L(𝔽_t); a focused investigation could produce a numerical free-entropy value for the substrate. Bundle with A.5, A.8, A.9.

### A.7 — KMS states on C*_red(F_inv(E))

**Search-instrument.** KMS states are equilibrium states under a one-parameter automorphism. For C*_red(F_inv(E)) with the time evolution induced by the continuum-limit Hamiltonian (Layer 3.13), KMS states would be the substrate's thermal states.

**Concrete first-pass output.** A 1-parameter family of states (parametrized by inverse temperature β) over the substrate C*-algebra. At β → ∞: ground state. At β → 0: tracial state.

**Compressibility check.** KMS states are determined by H + β; well-structured.

**Ontological grounding.** **Substrate:** thermal equilibrium over the substrate symmetry algebra. **Why this form:** standard quantum-statistical-mechanics formulation. **QFT ground:** KMS states are the QFT formulation of thermal equilibrium — appear in Tomita-Takesaki modular theory, Hawking radiation analyses, AdS-CFT thermal sectors. **Pairs directly with the §5.34–§5.38 quantum thermal/information cluster.**

**Forward-construction priority.** **High when paired with §5.34–§5.38.** Together they would ground the quantum-statistical-mechanics + KMS / area-law / holographic-entropy apparatus that QFT uses but doesn't structurally derive.

---

## Free probability (2 ops)

### A.8 — Free convolution of measures

**Search-instrument.** Voiculescu's free additive convolution ⊞ replaces classical convolution for free random variables. For F_inv(E)'s regular representation, two free generators T_1, T_2 satisfy: spectral distribution of T_1 + T_2 is the *free convolution* of distributions of T_1 and T_2 (each a Bernoulli ±1 measure).

**Concrete first-pass output.** Free convolution of |E| Bernoulli ±1 distributions (one per generator). For the framework's |E| = 6, this is a 6-fold free convolution. The result is the *free Bernoulli convolution* — known but not framework-named. Its spectral support is the spectrum of the adjacency operator A = Σ_e T_e on F_inv(E).

**Compressibility check.** Free Bernoulli convolution has explicit moments; spectral support is computed via R-transform. Highly structured.

**Ontological grounding.** **Substrate:** *the* natural probability theory on substrate (replaces classical convolution because generators are free). **Why this form:** F_inv(E) is the free product, so its random variables are free in Voiculescu's sense. **QFT ground:** free probability has emerged in QFT for matrix models, large-N gauge theories, random tensor networks. The substrate's natural free-probability structure could ground these connections.

**Forward-construction priority.** **Medium.** Computational; produces explicit substrate spectral data. Could cross-validate Layer 4.20 Alon-Boppana / Ramanujan saturation from a free-probability angle.

### A.9 — Free entropy / free Fisher information

**Search-instrument.** Voiculescu's free entropy χ replaces Shannon entropy for free random variables. For the substrate's regular generators (each Bernoulli ±1), χ has explicit form.

**Concrete first-pass output.** A specific real number per generator, additive across free factors: χ(F_inv(E)) = Σ_e χ(T_e) = |E| · χ(Bernoulli ±1) = |E| · (-∞) (since Bernoulli distributions have free entropy −∞). Hmm — that's a sign Voiculescu's free entropy is *not* the right invariant for the substrate's discrete generators.

A substitute: free Fisher information Φ. For Bernoulli ±1, Φ is well-defined and finite.

**Compressibility check.** Single number per substrate; structurally constrained.

**Ontological grounding.** **Substrate:** information-theoretic invariant of the substrate. **Why this form:** free probability replaces Shannon when variables are free. **QFT ground:** entropy in random matrix theory / random tensor networks; appears in holographic-entropy contexts. *Possibly* connects to the framework's MDL apparatus (Layer 4.5 Shannon entropy on substrate distributions) via a free-probability extension.

**Forward-construction priority.** **Medium.** Bundle with A.8; both are free-probability tools that may not yield surprises but cross-validate spectral structure.

---

## Categorical / monoidal (3 ops)

### A.10 — F_inv(E) as monoidal category

**Search-instrument.** F_inv(E) viewed as a discrete monoidal category (objects = group elements; morphisms = identities; tensor = group multiplication). Self-dual (every generator is its own dual via involutivity).

**Concrete first-pass output.** A *strict* discrete monoidal category with self-dual generators. Mathematically thin but the right starting point for higher-categorical generalizations.

**Compressibility check.** Trivial as stated; nontrivial when extended to enriched / 2-categorical structure.

**Ontological grounding.** **Substrate:** abstract structural shape of substrate symmetry. **QFT ground:** monoidal categories underlie tensor-network / fusion-category / TQFT formulations of QFT. The substrate's monoidal structure could ground these frameworks.

**Forward-construction priority.** **Low** as standalone op. **Medium** as preparatory for A.11 / A.12 / A.20.

### A.11 — ZX-calculus diagrammatic reasoning

**Search-instrument.** ZX-calculus is a diagrammatic language for ℂ²-tensor-network manipulations, particularly suited to multiway / quantum-walk substrates. Wolfram-Gorard work uses ZX-calculus for the Wolfram-multiway substrate.

**Concrete first-pass output.** Diagrammatic representation of the substrate's toggle operations + JW-derived CAR + Cl(6;ℂ) at trivalent nodes. Could systematize the framework's existing JW work.

**Compressibility check.** Visual; aids derivation but doesn't directly produce new numerical content.

**Ontological grounding.** **Substrate:** alternative encoding language. **QFT ground:** ZX-calculus is a primary tool in quantum-circuit synthesis and TQFT computations. The substrate's ZX-calculus encoding could connect to quantum-information formulations of QFT.

**Forward-construction priority.** **Medium.** Tooling rather than result-producing. Could systematize Layer 5.B work but not generate new predictions.

### A.12 — Monoidal functors between substrate categories

**Search-instrument.** Functors between substrate's monoidal category and other monoidal categories (e.g., Vec_ℂ for representations, Hilb for Hilbert-space reps). Each functor encodes a way of representing substrate structure.

**Concrete first-pass output.** Catalog of representation functors. The framework already uses several implicitly (regular rep, JW representation on Cl(6;ℂ)).

**Compressibility check.** Same as A.10 — structural rather than numerical.

**Ontological grounding.** **Substrate:** the substrate's representation theory organized categorically. **QFT ground:** functorial QFT (Atiyah-Segal axioms); TQFT classifications. Substrate's monoidal-functor structure could ground the categorical formulation of QFT.

**Forward-construction priority.** **Low** standalone. **Medium** in conjunction with A.10, A.11, A.20.

---

## Stochastic processes beyond Markov (3 ops)

### A.13 — Brownian motion as continuum limit of discrete walk

**Search-instrument.** Standard limit theorem: discrete random walk on a graph with proper scaling converges to Brownian motion (or diffusion with non-trivial drift) in the continuum limit. For F_inv(E)'s Cayley graph with toggle dynamics, the continuum limit is a *quantum* walk (unitary), not a classical Brownian motion. Wick rotation would convert: U(t) = exp(−iHt) → exp(−Hτ), and the Wick-rotated process IS a classical heat-kernel / Brownian-like process.

**Concrete first-pass output.** The Wick-rotated heat kernel exp(−Hτ) on substrate L²(F_inv(E)). Connection to A.13 Brownian motion via standard heat-kernel formalism.

**Compressibility check.** Heat kernel is determined by spectral data of H; well-structured.

**Ontological grounding.** **Substrate:** Wick-rotated continuum dynamics. **Why this form:** standard duality between unitary and stochastic processes. **QFT ground:** Brownian motion in path integrals (Wick-rotated propagator); diffusion equations from quantum dynamics. The substrate's Brownian-via-Wick-rotation grounds the Euclidean-path-integral formulation of QFT.

**Forward-construction priority.** **Medium.** Pairs with 5.33 Wick rotation (already invoked) and the §5.34–§5.38 cluster.

### A.14 — Stochastic differential equations on L²

**Search-instrument.** SDEs generalize Markov chains to continuous-state continuous-time. For substrate L²(F_inv(E)), an SDE-driven process would be a stochastic perturbation of the unitary evolution.

**Concrete first-pass output.** A specific SDE (e.g., stochastic Schrödinger) modeling substrate decoherence. The framework doesn't currently use stochastic perturbations of unitary evolution.

**Compressibility check.** Depends on the specific SDE.

**Ontological grounding.** **Substrate:** decoherence at the substrate level. **QFT ground:** stochastic-Lindblad evolution in open QFT; Schwinger-Keldysh formalism. Could ground these in substrate.

**Forward-construction priority.** **Low.** Substrate decoherence isn't a current concern; framework's compression theory (A2-T) is the substrate-decoherence analog already invoked.

### A.15 — Martingales adapted to multiway filtration

**Search-instrument.** A martingale is a process whose conditional expectation given past history equals the present value. The substrate's multiway / branching dynamics has a natural filtration (past toggle history).

**Concrete first-pass output.** Martingale-valued framework quantities (e.g., the running-MDL-savings is plausibly a martingale under the toggle process).

**Compressibility check.** Martingales have the nice property that their expectation is constant — could provide conserved-quantity analogs in substrate.

**Ontological grounding.** **Substrate:** information-theoretic-conservation laws. **Why this form:** standard martingale theory adapted to substrate filtration. **QFT ground:** Noether-currents from time-translation invariance; substrate analog is martingale conservation under toggle process. Pairs with 4.25 conditional expectation (already queued for cross-validation of A2-T).

**Forward-construction priority.** **Medium-high when paired with 4.25.** A martingale formulation could provide a clean conservation-law framing for the framework's MDL apparatus.

---

## Modular / automorphic (3 ops)

### A.16 — Modular forms attached to spectral content

**Search-instrument.** The framework's "Ramanujan eigenvalue" terminology (h_max = (√3 + i√5)/2 with |h|² = 2 saturating Alon-Boppana) is suggestive of modular structure. Ramanujan-Petersson conjectures bound Hecke-eigenvalues of modular forms; saturation at |h|² = k − 1 is the *Ramanujan property*.

For F_inv(E)'s Cayley graph with |E| = 6 (k = 3 trivalent), the Ramanujan property holds. This is the discrete-graph analog of modular-form Ramanujan-Petersson. Whether the substrate's spectral content sits in a richer modular family is an open question.

**Concrete first-pass output.** Identification of the substrate's Hecke-like operators and their eigenvalues. If the eigenvalue h fits in a known modular family (e.g., a Hecke eigenform of weight 2 on some modular curve), additional eigenvalues / spectral content would be predicted.

**Compressibility check.** Modular forms have rigid structure (level, weight, character). If substrate fits, the spectral content would be highly compressible.

**Ontological grounding.** **Substrate:** the substrate's spectral content viewed through arithmetic-geometry lens. **Why this form:** Ramanujan property hints at hidden modular structure. **QFT ground:** modular forms appear in QFT for partition functions of 2D CFTs (modular invariance), gauge theories on tori (S-duality), elliptic genera. The substrate's modular structure (if confirmed) would ground these.

**Forward-construction priority.** **High** (matching handoff). The Ramanujan saturation already in framework is the *strongest pre-existing hint* that the substrate sits in a modular family. Estimated 2-3 sessions for a focused investigation; mathematically heavier than A.1 / A.4 but with the highest "explained-mystery" potential.

**Status update 2026-04-26 (PM):** Tier 2 setup-and-scoping shipped (`../forward_constructions/forward_construction_substrate_modular_structure.md`) — adjacency eigenvalue λ = √3, Hecke eigenvalue a_2 = √3 at p = 2 computed; LPS framework match confirmed. **M1 LMFDB lookup performed:** spectral match across **candidate set of hundreds** of weight-2 dim-2 newforms with Hecke field Q(√3) and a_2 = √3. Smallest-level candidates: `63.2.a.b`, `65.2.a.c`, `81.2.a.a`, `85.2.a.c`, `117.2.a.b`, `165.2.a.b`, `169.2.a.a`. Unique identification deferred to Pizer-Brandt disambiguation (Tier 2 Path A, 2–3 sessions). Single a_p insufficient under Strong Multiplicity One. See an internal note.

### A.17 — Automorphic L-functions

**Search-instrument.** L-functions are Dirichlet series attached to modular forms / automorphic representations. If A.16 succeeds (substrate eigenvalue is an automorphic Hecke eigenvalue), the corresponding L-function would be a substrate invariant.

**Concrete first-pass output.** Pending A.16; downstream of modular identification.

**Ontological grounding.** **Substrate:** number-theoretic structure on substrate spectrum. **QFT ground:** L-functions appear in QFT for partition functions on finite-genus surfaces; in number-theoretic-physics correspondences (Langlands, Geometric-Langlands).

**Forward-construction priority.** **Pending A.16.** Cannot precede A.16.

### A.18 — Selberg zeta function for the Cayley graph

**Search-instrument.** Selberg zeta encodes spectral data of the Laplacian on hyperbolic surfaces; analogs exist for graphs (Ihara zeta). For F_inv(E)'s Cayley graph, the Ihara zeta function is well-defined and computable.

**Concrete first-pass output.** Ihara zeta function ζ_F_inv(E)(u) — already implicitly used by the framework via Ihara-Bass determinant identity in `predictions/h_walker_eigenvalue.py`. So A.18 in its Ihara-zeta form is *invoked-indirect*.

**Compressibility check.** Ihara zeta is highly structured; its functional equation and connections to graph spectrum are exploited.

**Ontological grounding.** **Substrate:** zeta-function packaging of substrate spectral data. **Why this form:** standard graph-theoretic construction. **QFT ground:** functional determinants in QFT (Faddeev-Popov); zeta-regularization of one-loop effective actions.

**Forward-construction priority.** **Already invoked-indirect via Ihara-Bass.** Worth a meta-finding correction: this Appendix entry is not *fully* unused; the framework uses the Ihara-zeta determinant identity. The Selberg-zeta-on-hyperbolic-surface form is unused (and probably inapplicable since substrate isn't hyperbolic continuum).

---

## Extended physics (3 ops)

### A.19 — Quantum gravity operations

**Search-instrument.** Beyond classical GR — would include things like holographic operators, Wheeler-DeWitt equations, loop-quantum-gravity holonomies, causal-set dynamics. Catalog flags this would need a *new layer* (Layer 7+).

**Concrete first-pass output.** None first-pass; would require defining the new-layer operations.

**Ontological grounding.** **Substrate:** quantum-gravity operations would extend the substrate's continuum-limit structure to fully quantum spacetime. **QFT ground:** quantum gravity is QFT's largest open problem; the framework's substrate-multiway-causal-set structure has natural overlap with causal-set quantum gravity.

**Forward-construction priority.** **Research-level (>3 sessions); pairs with §C smooth-manifold closure.** The framework already has natural causal-set structure (§6.24 invoked); extending to quantum-gravity-as-quantum-causal-set is a major open direction.

### A.20 — TQFT operations (categorical)

**Search-instrument.** TQFT functors between cobordism categories and Vec_ℂ (Atiyah-Segal axioms). The substrate's monoidal structure (A.10) could be extended to a TQFT functor.

**Concrete first-pass output.** Pending A.10–A.12 maturation.

**Ontological grounding.** **Substrate:** TQFT structure on substrate. **QFT ground:** topological gauge theories (Chern-Simons), 4-manifold invariants (Donaldson-Witten). The substrate's TQFT (if constructible) could ground topological sectors.

**Forward-construction priority.** **Low** standalone; bundled with A.10–A.12.

### A.21 — CFT operators (OPE, Virasoro)

**Search-instrument.** 2D CFT requires the substrate's continuum limit to produce a 2D conformal sector. The framework's substrate is 3-spatial + 1-temporal; no obvious 2D CFT sector, unless restricted to a 2D subspace.

**Concrete first-pass output.** Negative — no natural 2D CFT sector at substrate scale. *However*, holographic / boundary-CFT considerations could produce one.

**Ontological grounding.** **Substrate:** would require holographic-style restriction. **QFT ground:** 2D CFT for boundary excitations, AdS_3/CFT_2.

**Forward-construction priority.** **Low.** No first-pass match between substrate's natural dimensionality and 2D CFT; would require major restriction.

---

## Aggregate (Appendix)

| Status | Topo | Op-Alg | Free-Prob | Cat | Stoch | Modular | Phys | Total |
|---|---|---|---|---|---|---|---|---|
| invoked-indirect (Appendix-discovered) | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| forward-construction high | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 3 |
| forward-construction medium | 2 | 1 | 2 | 1 | 2 | 0 | 0 | 8 |
| forward-construction low | 1 | 1 | 0 | 2 | 1 | 0 | 2 | 7 |
| forward-construction pending-other | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| research-level open | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| **Appendix total** | **4** | **3** | **2** | **3** | **3** | **3** | **3** | **21** |

**Coverage.** 21/21 Appendix entries audited.

**Meta-finding:** A.18 Selberg-zeta (in its Ihara-zeta-on-graph form) is **already invoked-indirect** via Ihara-Bass determinant identity — the operator sweep's own classification was slightly conservative. One Appendix op moves from "unused" to "invoked-indirect" upon careful audit.

---

## Forward-construction ranking (consolidating with prior layers)

### Tier 1 — High priority, focused 1-3 session investigations

1. **§5.34–§5.38 quantum thermal/information cluster + A.7 KMS states** — bundle of 6 + 1 = 7 ops. Could ground QFT KMS / area law / holographic entropy in substrate. Substrate-grounded substrate-thermal Z, vN entropy, entanglement entropy.
2. **§4.25 conditional expectation as A2-T cross-validation** — single op; reformulate MDL canonicalization as conditional expectation; gain L²-orthogonality structure for free.
3. **A.4 Atiyah-Singer index for substrate Dirac (graph form)** — discrete fermion-anomaly index. Connects to SM chirality structure (left/right asymmetry, ν_R-forced).
4. **A.16 modular forms attached to substrate spectrum** — Ramanujan saturation at |h|² = k − 1 hints substrate sits in modular family. Highest "explained-mystery" potential; mathematically heaviest.

### Tier 2 — Medium priority

5. **§5.16 Schmidt rank of A3-T purification** — integer invariant tied to compression structure.
6. **A.1 group cohomology + A.3 K-theory** — bundle for integer invariants from substrate topology.
7. **A.5 + A.6 + A.8 + A.9 operator algebras + free probability** — bundle for systematic operator-algebraic / free-probability lens; cross-validates spectral structure.
8. **A.15 martingales on multiway filtration** — paired with 4.25 conditional expectation; substrate analog of conserved currents.
9. **§5.22 Heisenberg picture investigation** — substrate-natural picture; conceptual rather than predictive.
10. **A.11 ZX-calculus systematization** of Layer 5.B (CAR/JW/Cl) — tooling.

### Tier 3 — Research-level / pending other ops

11. **A.17 automorphic L-functions** — pending A.16.
12. **A.19 quantum gravity operations** — pairs with §C smooth-manifold closure.
13. **§6.A/B GR-internal smooth-manifold cluster** — 8 ops pending §C closure.
14. **A.20 TQFT operations** — pending A.10–A.12 maturation.

### Tier 4 — Likely null

15. **A.21 CFT operators** — no obvious match to substrate dimensionality.
16. **A.13 Brownian motion** — already covered by Wick rotation (5.33) + heat kernel use.

---

## Honest verdict on Appendix sweep

**Search-instrument value:** Appendix is the strongest layer for forward-construction candidates because every op is unused by construction. Tier 1 has 4 high-priority items; Tier 2 has 6 medium-priority items. Total: 10 actionable forward-construction directions across the catalog.

**Ontological yield:** The Appendix exposes the strongest potential ontology landings for QFT-postulated objects without current substrate grounding:
- A.7 + §5.34–§5.38: KMS / thermal / area law / holographic entropy
- A.4: chirality / fermion-number anomaly
- A.16: modular structure / spectral compactness
- A.5–A.9: operator-algebraic / free-probability formulation of QFT (Haag-Kastler / Voiculescu lens)
- A.19: causal-set quantum gravity

**Re-ranking versus handoff's prior verdict:**
- **A.4 confirmed high** (handoff was right).
- **A.16 confirmed high** (handoff was right).
- **A.1 demoted from high to medium** (first-pass cohomology output is too uniform — six independent ℤ/2 invariants don't naturally split into SM charge structure).
- **A.7 elevated to high** (handoff didn't separately call it out, but pairing with §5.34–§5.38 makes it Tier 1).

**Cross-cutting cluster discovered:** the Tier 1 forward-construction direction is dominated by a *single coherent program*: substrate-quantum-information ↔ KMS / thermal / entanglement / area law / holographic entropy. §5.34–§5.38 + A.7 + parts of A.5–A.9 + A.4 + 4.25 + A.15 all live in this program. ~14 ops cluster around substrate-quantum-information theory. This is the operator-sweep's strongest single search-instrument finding.

---

## Cumulative across the entire sweep

**174 layered ops + 21 Appendix ops = 195 catalog entries audited.**

| Status | Count |
|---|---|
| invoked-direct (Layers 0–6) | ~140 |
| invoked-indirect (Layers 0–6) | ~10 |
| invoked-indirect (Appendix-recovered: A.18) | 1 |
| invoked-negatively (1.10 abelianization in observer construction) | 1 |
| unused-applied-negative (pinned obstructions: 2.21, 3.9, 4.13, 5.23, 5.24) | 5 |
| unused-deferred (not yet applied) | ~38 |
| **Total** | **~195** |

**Forward-construction queue (Tier 1):** 4 high-priority items; ~14 ops in the substrate-quantum-information cluster.

**Pinned obstructions:** 5 (most informative: 2.21 compact operators *structurally excluded*; 5.23/5.24 interaction picture / TDPT *displaced by static spectral PT + waterline-MDL summation*).

**Ontology landings harvested:** ~25 QFT-postulated objects with substrate grounding sketched (CAR algebra, Dirac spinor, density matrix, Pauli matrices, Killing-form gauge, FLRW, causal structure, Berry phase, Wick rotation, Pati-Salam embedding, etc.).

**Ontology gaps still open:** vacuum |0⟩, field operator φ(x), path integrals (Wick-rotated form partially grounded), BRST / gauge fixing, renormalization derivation, Einstein equations, full smooth-manifold limit, quantum-thermal cluster (deferred to Tier 1 program).

---

## Cross-references

- `operator_sweep_from_A1.md` Appendix — source.
- All predecessor audits (`docs/operator_sweep_audit_layer_*.md`).
- `../theorems/theorem_bloch_lift_mu.md` — Ramanujan saturation context for A.16.
- `predictions/h_walker_eigenvalue.py` — Ihara-Bass determinant (A.18 invoked-indirect).

---

## Status

**Catalog sweep complete.** All 195 catalog entries audited (174 layered + 21 Appendix). Three deliverables ready for next phase:

1. **`../framework/framework_qft_ontology.md` meta-doc** — harvest QFT-postulated objects with substrate grounding from Layers 0–6 + Appendix audits, organized *by QFT object* (vacuum, field operator, CAR, density matrix, etc.) rather than *by catalog op*.
2. **Backfill ontology lens into Layers 0–4 audits** — predecessor audits used two-lens entries; recalibration to three-lens (with ontology) was added at Layer 5. One pass to add ontology-grounding lines for Layers 0–4 ops.
3. **Forward-construction Tier 1 program: substrate quantum-information** — focused investigation of the 14-op cluster (§5.34–§5.38 + A.7 + portions of A.5–A.9 + A.4 + 4.25 + A.15) that could ground QFT's KMS / thermal / entanglement / area-law / holographic apparatus.

Recommendation: execute (1) first — the meta-doc — since it both consolidates the ontology harvest and defines the gap-list against which Tier 1 forward-construction work would be measured.
