# F_inv(E) → srs compression chain — companion theorem

**Date:** 2026-05-05 EOD+3 (NA-2' Sessions 1-4: setup + Steps 1-6 proofs + integration check).  Status banner updated 2026-05-23 (post R-9 structural closure).
**Status:** **THEOREM-GRADE.** All six steps closed; chain integration verified.  **2026-05-23 update:** the prior "THEOREM-GRADE-CONDITIONAL on Sunada (2012) arc-transitivity" label is *no longer accurate* — R-9 closed structurally on 2026-05-12 via the (A) self-containment → no privileged direction → arc-transitive substrate-agnostic model → Sunada 2012 chain (`audits/registers/structural_residue_register.md` R-9 closure block; `predictions/walker_dynamics_derivation.md` Step 4b; `predictions/g_girth_derivation.md` Step 2).  Arc-transitivity is now itself a derived structural theorem under (A), eliminating the prior conditional load on this document.  Sunada 2012's published-theorem citation remains a Type 3 mathematical input but is no longer "conditional" in the framework-status sense.
**Scope:** Trace the structural descent from the F_inv(E) Cayley graph adjacency operator (per `theorem_A3_complex_hilbert_from_multiway.md`) to the Bloch-Hashimoto operator on srs (per `theorem_bloch_lift_mu.md`, used by visible-side predictions). Address the explicit §13.2 open question of A3-T:

> "The relationship between 'adjacency operator on F_inv(E)'s Cayley graph' and 'Hashimoto operator on srs' is the dominant-sector compression chain (ARG-1, ARG-2, the srs identification — none formally linted in this document but all developed in session work and Stage 2 framework theorems). A separate companion document tracing this chain explicitly would tighten the connection."

**Closes scoping target:** NA-2' of an internal working note.

**Post-2026-05-08 axiom slate note.** A1, P1', and A2-T (cited as Type 1 inputs throughout this theorem) are now derived theorems of (A) self-containment + (B) finite observer + standard math + (I) active reading, per `theorem_toggle_from_self_containment.md`, `theorem_p1_prime_derived_from_a1.md`, and `theorem_A2_mdl_from_finite_register.md`. References to "A1 + P1' + A2-T" remain semantically valid; the F_inv(E) → srs compression chain is unchanged. The Sunada arc-transitivity conditional is independent of the axiom slate revision. See `framework_axioms.md` §10 for the updated top-level summary.

---

## 1. Theorem statement

**Theorem (F_inv(E) → srs compression).** Under A1 + P1' + A2-T (the framework's structural axiom slate post-2026-04-26 demotion), the dominant-sector compression of the substrate Hilbert space L²(F_inv(E); ℂ) is the Bloch-decomposed Hashimoto operator on the srs lattice's directed-edge ℓ² space:

$$L^2(F_{\text{inv}}(E);\, \mathbb{C}) \;\xrightarrow{\;\Pi_{\text{MDL}}\;}\; \ell^2(\vec{E}_{\text{srs}}) \;\overset{\sim}{=}\; \bigoplus_{\boldsymbol{k} \in \text{BZ}} \ell^2(\vec{E}_{\text{srs unit cell}})$$

where:

(i) Π_MDL is the MDL canonicalization map composed with alphabet localization to the srs primitive cell, derived from A2-T's reduced-word selection and the d=3, k*=3 + Gleason + Sunada chain that pins srs as the unique embedded compressed graph.

(ii) The right-hand isomorphism is Sunada's Bloch decomposition over the srs Brillouin zone (Sunada 2013 §6 Theorem 6.4).

(iii) Under Π_MDL, the F_inv(E) Cayley-graph adjacency operator (Childs 2009 form, per A3-T Step 4) is taken to the Bloch-Hashimoto operator family {B(**k**) : **k** ∈ BZ}.

**Net effect.** Closes the §13.2 open question of `theorem_A3_complex_hilbert_from_multiway.md`. Provides a single citable theorem doc for the descent F_inv(E) Cayley graph → srs Bloch, replacing the implicit chain currently spread across `H_multiway_dim_count_derivation.md` + `theorem_h1_master_compression.md` + `theorem_bloch_lift_mu.md` + Row 4 + Row 6 + Row 7 closures.

---

## 2. Axioms invoked + cited upstream

**Framework axioms (Type 1 gates):**

- **A1** (`docs/framework/framework_axioms.md` §2) — finite alphabet E of binary self-inverse toggle generators; substrate is F_inv(E).
- **P1'** (`no_free_bits` §1.1) — observer is a finite register; register-storable eigenvalues are real-valued.
- **A2-T** (`docs/theorems/theorem_A2_mdl_from_finite_register.md`) — MDL retention principle as derived theorem (post-2026-04-20 demotion).

A3-mass is downstream of this theorem (empirical labeling, not load-bearing for the structural chain).

**Type 3 cited published theorems:**

- **Serre, J.-P.** (1980). *Trees.* Springer. §I.1 Proposition 4 — uniqueness of reduced word in the free involutive monoid (used for Step 3 canonicalization).
- **Sunada, T.** (2012). *Lectures on Topological Crystallography.* Japan. J. Math. 7. — strong isotropy uniqueness of srs as 3D 3-regular crystal net (used for Step 5 srs identification, via Row 4 + Row 6 + R-9 closures).
- **Sunada, T.** (2013). *Topological Crystallography.* Springer. §6 Theorem 6.4 — Bloch decomposition of operators on crystal Z^d-modules (used for Step 6 Bloch decomposition).
- **Terras, A.** (2011). *Zeta Functions of Graphs.* Cambridge. §2.2 — Ihara determinant for Bloch-Hashimoto (used in Step 6 cross-check).
- **Alon, N.** (1986). Eigenvalues and expanders. *Combinatorica* **6**, 83–96 — Ramanujan bound (used as bridge to visible-side predictions).
- **Brown, K. S.** Cohomology of Groups (1982). Springer. — Brown rank, used in Row 4 closure (Step 5).
- **Gleason, A. M.** (1957). Measures on the Closed Subspaces of a Hilbert Space. *J. Math. Mech.* **6**(6) — Gleason's theorem giving d=3 for visible (Row 4 closure, Step 5).
- **Childs, A. M.** (2009). Universal computation by quantum walk. *Phys. Rev. Lett.* **102**, 180501 — adjacency-operator Hamiltonian for continuous-time quantum walks (used at the input side via A3-T Step 4).

**Type 4 upstream framework theorem documents:**

- `docs/theorems/theorem_A3_complex_hilbert_from_multiway.md` — substrate state space = L²(F_inv(E); ℂ); identifies F_inv(E) Cayley-graph adjacency as continuum-limit Hamiltonian (Step 4 of A3-T).
- `docs/theorems/theorem_multiway_branch_measure.md` — μ on Σ* as theorem (P1-P6); admissibility = NB walk on F_inv(E) (used at Steps 3-4).
- `docs/theorems/theorem_A2_mdl_from_finite_register.md` — A2-T (MDL retention as derived theorem).
- `predictions/H_multiway_dim_count_derivation.md` — Layer-1 length-graded Hilbert space ℋ_multiway^(L) = ℋ_visible^(L) ⊕ ℋ_dark^(L); canonicalization map π (used at Step 3).
- `predictions/d_spatial_derivation.md` (d = 3 from Gleason chain) — used at Step 2 ARG-1 (alphabet-localization input).
- `predictions/k_star_derivation.md` (k* = 3 from MDL coordination optimization) — used at Step 2 ARG-1.
- `predictions/g_girth_derivation.md` §2 (srs primitive cell |V|=4, |E|=6) — used at Step 2 ARG-1.
- `docs/audits/registers/uniqueness_ledger.md` Row 4 (k* = 3 closure) + Row 6 (srs lattice closure) — both UNIQUE-CONDITIONAL on Sunada arc-transitivity per 2026-05-05 revision.
- `docs/audits/registers/uniqueness_ledger.md` Row 7 — alphabet-localization stipulation (A1's E identified with srs's 6 primitive-cell undirected edges).
- `docs/audits/registers/structural_residue_register.md` R-3 (REFUTED 2026-04-27) — F_inv(E)/N quotient relations don't reproduce srs cycles algebraically; cycles are geometric.
- `docs/theorems/theorem_h1_master_compression.md` — gauge/physical decomposition of edge-direction data on k-regular graphs (used at Step 4 to identify visible vs gauge content).
- `docs/theorems/theorem_bloch_lift_mu.md` — Bloch-lift of μ; K = ⊕_k B(**k**); Ramanujan saturation at k_P (used at Step 6).
- `predictions/walker_dynamics_derivation.md` — W4 Step 4 with explicit arc-transitivity prerequisite (post-2026-05-05 revision).

---

## 3. Setup — the six-step chain

Each step is currently developed in the Type 4 sources cited; this section inventories them. Closure proofs of individual steps are the work of NA-2' Sessions 2-4.

### Step 1 — Substrate Hilbert space (Type 4 input; closed)

**Claim.** Under A1 + P1', the substrate state space is L²(F_inv(E); ℂ) with self-adjoint Hamiltonian H, and H equals the adjacency operator of F_inv(E)'s Cayley graph.

**Proof (one-paragraph summary; see `theorem_A3_complex_hilbert_from_multiway.md` for the full derivation).**

By A1 the substrate dynamics is generated by the finite alphabet E of binary self-inverse toggle operators T_e (e ∈ E, T_e² = id). Compositions modulo e·e = ε form the free involutive monoid F_inv(E), a discrete countable group (Serre 1980 §I.1 Proposition 4). F_inv(E) acts on itself by left multiplication; this is reversible and generated by A1's involutive toggles (`theorem_A3_complex_hilbert_from_multiway.md` §4 Step 1). The natural state space is L²(F_inv(E)) with counting measure (Folland 1999 §11.1, §11.4); the left regular representation L: F_inv(E) → 𝒰(L²) is unitary (`theorem_A3_complex_hilbert_from_multiway.md` §6 Step 3). The discrete-time multiway evolution (one toggle per Planck step) admits a strongly continuous one-parameter unitary limit U: ℝ → 𝒰(L²) under (i) bounded vertex degree |E| (A1 finiteness) and (ii) sub-Planckian correlation length ξ_t = 1/log 6 ≈ 0.558 ℓ_P from Stage 3 (`theorem_lorentz_causal_sector.md` §3, CAS-verified to 24+ decimal digits in `proofs/lorentz/b1_ags_audit.py`). Strauch 2006 + Childs 2009 then identify the continuum-limit generator as H = A_{F_inv(E)}, the adjacency operator of F_inv(E)'s Cayley graph. Stone 1932 + Reed-Simon I §VIII.4 give U(t) = exp(-iHt) with H self-adjoint on the **complex** L² (the field selection ℂ over ℝ ∪ ℍ is forced by P1' alone via the §F register-is-real argument; see `theorem_A3_complex_hilbert_from_multiway.md` §10 Step 7). ∎

**Type 1** (A1, P1') + **Type 3** (Serre 1980; Folland 1999; Strauch 2006; Childs 2009; Stone 1932; Reed-Simon I §VIII.4) + **Type 4** (`theorem_A3_complex_hilbert_from_multiway.md`; `theorem_lorentz_causal_sector.md`; `proofs/lorentz/b1_ags_audit.py`).

**Status:** ✓ THEOREM-grade. No fresh proof needed in this document; the citation chain to A3-T closes Step 1.

### Step 2 — ARG-1 alphabet localization (Type 4 input; closed via upstream chain)

**Claim.** The toggle alphabet E of A1 is identified with the 6 undirected edges of the srs primitive cell. The identification is forced by upstream theorem-grade rows, not stipulated.

**Proof.** A1 supplies a finite alphabet E of binary self-inverse toggle generators but does not specify |E| or its physical localization. Upstream framework rows pin |E| = 6 by elementary arithmetic from the chain:

| Component | Value | Source | Closure grade |
|-----------|-------|--------|---------------|
| Spatial dimension d | 3 | `predictions/d_spatial_derivation.md` (Gleason 1957 frame functions); uniqueness ledger Row 3 | UNIQUE-THEOREM |
| Substrate valence k* | 3 | `predictions/k_star_derivation.md` (Brown 1986 Fisher rank + R-7 + Sunada arc-transitivity 2026-05-05); uniqueness ledger Row 4 | UNIQUE-CONDITIONAL on Sunada arc-transitivity |
| Substrate lattice | srs (space group I4₁32) | `predictions/g_girth_derivation.md` §2 (Sunada 2012 strong isotropy uniqueness); uniqueness ledger Row 6 | UNIQUE-CONDITIONAL on Sunada arc-transitivity |
| Vertices per primitive cell |V| | 4 | srs Wyckoff 8a in conventional cell, halved by I4₁32 body-centred Bravais translation; `H_multiway_dim_count_derivation.md` §40; uniqueness ledger Row 8 | UNIQUE within srs |
| Undirected edges per primitive cell |E| | 6 | k*·|V|/2 by handshake lemma; uniqueness ledger Row 7 | UNIQUE (closed 2026-04-27 via R-11) |

The "undirected edges" qualifier is forced by A1's involutivity T_e² = id (the toggle is its own inverse, which is the property of an undirected edge under the involution that identifies e with its reverse — see `H_multiway_dim_count_derivation.md` §1).

R-11 (`docs/audits/registers/structural_residue_register.md` R-11) closed 2026-04-27 verifies that all operator-permitted alternative localizations |E| ∈ {1, 2, 3, 12 directed, 12 conventional, ∞} are eliminated:
- |E| ∈ {1, 2, 3, ∞} hard-gated by combinations of A1 + Row 3 (d=3) + Row 4 (k*=3);
- |E| = 12 (directed) hard-gated by A1 involutivity (each edge carries a SINGLE toggle T_e with T_e² = id; directed-edge count is 2|E|, used downstream in the Hashimoto picture but not at the alphabet-localization step);
- |E| = 12 (conventional cell) soft-gated BELOW MDL waterline at +1 bit/event redundancy (the body-centred conventional cell is 2× the primitive cell; |E| = 12 carries the same information as |E| = 6 with twice the labels).

|E| = 6 is therefore the UNIQUE survivor. Each e ∈ E carries a toggle T_e by A1; the 6 undirected edges of the srs primitive cell are identified with the 6 generators of A1's alphabet.

Verified: `proofs/foundations/r11_alphabet_localization_check.py`. ∎

**Type 1** (A1) + **Type 2** (handshake lemma |E| = k*·|V|/2; involutivity → undirected qualifier) + **Type 3** (Gleason 1957 for d=3; Brown 1986 Fisher rank for k*=3; Sunada 2012 strong isotropy for srs) + **Type 4** (uniqueness ledger Rows 3, 4, 6, 7, 8; structural residue register R-11; `r11_alphabet_localization_check.py`).

**Status:** ✓ THEOREM-grade-conditional on Sunada arc-transitivity (post-2026-05-05 Row 4 + Row 6 closures). The alphabet-localization stipulation in `H_multiway_dim_count_derivation.md` §1 is now backed by an explicit upstream-derivation chain in the structural residue register.

### Step 3 — Layer-1 visible/dark decomposition (Type 4 input; closed)

**Claim.** Lift F_inv(E) to a length-graded Hilbert space and decompose it canonically into a visible (reduced-word) sector and a dark (cancellation-bearing) sector. The visible sector carries the MDL projector image; the dark sector is its kernel.

**Proof (one-paragraph summary; full construction in `predictions/H_multiway_dim_count_derivation.md`).**

Define the length-L unreduced multiway Hilbert space ℋ_multiway^(L) := ℂ^{|E|^L} = (ℂ^{|E|})^{⊗L} with orthonormal basis {|w⟩ : w ∈ E^L} (one basis vector per length-L string). The full multiway Hilbert space is ℋ_multiway = ⊕_{L ≥ 0} ℋ_multiway^(L). Under A2-T (`theorem_A2_mdl_from_finite_register.md`), the observer selects from each equivalence class [w] ∈ F_inv(E) the unique reduced representative r(w) (Serre 1980 §I.1 Prop. 4; Grünwald 2007 §5.1-5.3); this defines the **canonicalization map**

$$\pi : \mathcal{H}_{\text{multiway}}^{(L)} \to \bigoplus_{L' \leq L,\ L - L' \in 2\mathbb{Z}_{\geq 0}} \mathbb{C}^{R_{L'}}, \qquad \pi(|w\rangle) = |r(w)\rangle.$$

The image of π restricted to length-L reduced words embeds isometrically as the **visible Hilbert space**

$$\mathcal{H}_{\text{visible}}^{(L)} := \mathrm{span}\{|w\rangle : w \in E^L,\ w\ \text{reduced}\}$$

with dim ℋ_visible^(L) = R_L = |E|·(|E|-1)^{L-1} for L ≥ 1, R_0 = 1 (Serre 1980 §I.1 Prop. 4). The orthogonal complement is the **dark Hilbert space**

$$\mathcal{H}_{\text{dark}}^{(L)} := \mathrm{span}\{|w\rangle : w \in E^L,\ w\ \text{NOT reduced}\}$$

with dim ℋ_dark^(L) = D_L = |E|·[|E|^{L-1} - (|E|-1)^{L-1}]. Since reduced and non-reduced strings partition E^L exhaustively and disjointly, ℋ_multiway^(L) = ℋ_visible^(L) ⊕ ℋ_dark^(L) is an orthogonal direct sum.

For |E| = 6 (Step 2), this gives R_L = 6·5^{L-1} reduced words and D_L = 6·[6^{L-1} - 5^{L-1}] cancellation-bearing strings; closed forms verified by sympy + brute-force enumeration for L = 0..7 in `predictions/H_multiway_dim_count.py` to exact integer equality.

**Identification of π with the MDL projector at Layer-1.** π acts trivially on |w⟩ iff w is already reduced (no cancellation). On non-reduced strings, π applies the e·e → ε reductions iteratively until a reduced word is obtained. By A2-T's MDL canonicalization, the reduced word is the unique minimum-description-length representative within its equivalence class. Hence π = Π_MDL restricted to the Layer-1 length-graded space; ℋ_dark^(L) = ker(π — id) ∩ ℋ_multiway^(L) is the per-length cancellation kernel.

**Connection to per-step branching (μ).** Per `theorem_multiway_branch_measure.md` P3-P4, μ has uniform per-step marginal 1/|E| on E and uniform conditional 1/(|E|-1) on admissible (NB) continuations given admissibility. The per-step probability that an extension stays in the visible sector (NB continuation) is (|E|-1)/|E| = 5/6 at the F_inv(E) level; the probability of cancellation (extension reverses the previous letter, dropping out of ℋ_visible^(L) into ℋ_dark^(L)) is 1/|E| = 1/6. ∎

**Type 1** (A1) + **Type 2** (orthogonal direct sum from disjoint exhaustive basis partition; recursion R_L = (|E|-1)·R_{L-1}) + **Type 3** (Serre 1980 §I.1 Prop. 4; Grünwald 2007 §5.1-5.3) + **Type 4** (`theorem_A2_mdl_from_finite_register.md`; `theorem_multiway_branch_measure.md` P1-P5; `predictions/H_multiway_dim_count_derivation.md` §1-5; verification: `predictions/H_multiway_dim_count.py`, `proofs/foundations/H_multiway_construction.py` Checks A-B).

**Status:** ✓ THEOREM-grade. Layer-1 visible/dark decomposition closed; π identified with Π_MDL on Layer-1.

**Hygiene note.** Multiple framework documents reference a planned theorem doc `docs/theorem_H_multiway_construction.md` as if it existed; the canonical content lives in `predictions/H_multiway_dim_count_derivation.md` + `proofs/foundations/H_multiway_construction.py`. The present theorem cites those directly. Creating a sibling theorem doc to consolidate the construction in `docs/theorems/` is recommended hygiene but NOT load-bearing for this Step.

### Step 4 — ARG-2 geometric embedding (Type 4 input; closed via walker_dynamics W1)

**Claim.** Visible reduced words on alphabet E (Layer-1 ℋ_visible^(L) basis vectors from Step 3) correspond bijectively to graph-admissible NB walks of length L on srs's directed-edge state space. The correspondence is via geometric realization, NOT algebraic quotient.

**Proof.**

The compression F_inv(E) Cayley graph → srs is NOT a closure via algebraic quotient relations. R-3 of the structural residue register (REFUTED 2026-04-27) verified that F_inv(E)/N quotient relations cannot reproduce srs's cycles algebraically: F_inv(E)'s Cayley graph is |E|-edge-labelled regardless of any imposed normal subgroup N, and the labelled-edge structure does not match srs's 3-regular geometric embedding without engineering N to match srs's specific cycle structure (which would amount to importing the answer). F_inv(E) has no algebraic relations beyond involutivity (e·e = ε); srs's girth g = 10 cycles are GEOMETRIC consequences of the lattice embedding, not algebraic identities of F_inv(E).

The actual closure is via geometric realization, established in `predictions/walker_dynamics_derivation.md` Steps 3 + 5 + W1:

1. **Reduced-word ↔ NB-walk bijection (Serre 1980 § I.1 Prop. 4; Terras 2011 §2.1).** Under the alphabet-localization of Step 2, each e ∈ E is identified with one of the 6 undirected edges of the srs primitive cell. For a *graph-admissible* reduced word w = (e_1, e_2, ..., e_L) on E starting from a vertex v_0 (where "graph-admissible" means each letter e_i is incident to the current walker vertex), w induces a unique walk on srs by orienting each undirected edge by traversal direction. The reduced condition (e_i ≠ e_{i+1}) is precisely the NB condition for the walk (the next directed edge does not reverse the previous). Hence reduced word ↔ NB walk bijectively.

2. **Causal state = directed edge (Shalizi-Crutchfield 2001 Theorem 2).** The minimal sufficient statistic for prediction in the NB walk is the current directed edge. The state space has 2|E| = 12 directed edges per srs primitive cell. The walker's history collapses to its current causal state.

3. **Hashimoto operator B is the 1-step transition (Hashimoto 1989; Terras 2011 §2.2).** B[e', e] = 1 if e → e' is a valid 1-step NB transition, else 0. The L-step operator is B^L; matrix elements (B^L)[e_L, e_0] count NB walks from e_0 to e_L of length L.

4. **Arc-transitivity prerequisite (`row4_audit_v2_revision_session2_2026-05-05.md`; `walker_dynamics_derivation.md` Step 4 amendment 2026-05-05).** For the per-step NB survival probability (k-1)/k = 2/3 to be uniform across all 24 directed bonds in the primitive cell (rather than per-orbit-dependent), srs must be arc-transitive (every directed edge equivalent under the automorphism group action). srs IS arc-transitive: vertex-transitive + edge-transitive + vertex stabilizer C_3 (body-diagonal) acting transitively on the 3 outgoing edges per vertex → 1 directed-edge orbit (Sunada 2012). This makes the geometric embedding's per-step survival STRUCTURALLY uniform.

**The graph-admissibility constraint.** Not every reduced word on alphabet E lifts to a walk on srs from any starting vertex: at vertex v with incident edges {e_a, e_b, e_c} ⊂ E (3 of the 6 alphabet letters by k* = 3 valence), only words whose first letter is in {e_a, e_b, e_c} are admissible from v. Graph-admissibility filters non-realizable reduced words. Under arc-transitivity, the filtered subset is structurally well-defined: from each vertex v, exactly half the alphabet (3 letters) is locally accessible, with the C_3 stabilizer making the local choices symmetric.

**Why this is geometric realization, not algebraic identification.** The bijection reduced-word ↔ NB-walk uses srs's TOPOLOGY (vertex-edge incidence + 3-regular structure + arc-transitivity) to lift the symbolic toggle sequence to a concrete walk. F_inv(E) supplies the symbolic algebra; srs supplies the geometric realization. Cycles in srs (girth g = 10) become CLOSED NB walks of length 10 — these are NOT algebraic identities of F_inv(E) but geometric features of the realization. ∎

**Type 1** (A1) + **Type 3** (Serre 1980; Terras 2011 §2.1, §2.2; Hashimoto 1989; Shalizi-Crutchfield 2001) + **Type 4** (`predictions/walker_dynamics_derivation.md` Steps 3, 4, 5; `docs/audits/registers/structural_residue_register.md` R-3 REFUTED; an internal working note).

**Status:** ✓ THEOREM-grade-conditional on Sunada arc-transitivity (same conditional as Steps 2 and 5). The geometric embedding closure is via Serre's reduced-word ↔ NB-walk bijection on the alphabet-localized substrate, lifted to srs via vertex-edge incidence and made structurally uniform by W4 Step 4's arc-transitivity prerequisite. R-3 refutation rules out the alternative (algebraic-quotient) closure path; the present closure does not require it.

**Likely-surprise resolution.** The Session 1 setup flagged Step 4 as "may need additional structural argument beyond R-3 refutation." That concern is resolved: the constructive closure runs through walker_dynamics W1's Serre bijection, which IS theorem-grade and was already developed in session work. The present document just lints the chain. R-3 refutation is the negative companion (rules out the alternative path); W1 is the positive closure.

### Step 5 — srs identification (Type 4 input; closed via multi-row uniqueness chain)

**Claim.** srs (Laves lattice, space group I4₁32, Wyckoff 8a) is the UNIQUE substrate lattice within the framework's structural constraints. Specifically, srs is the unique strongly isotropic 3D 3-regular chiral crystal net.

**Proof.** The identification chain runs through the uniqueness ledger:

| Row | Constraint | Forces |
|-----|-----------|--------|
| Row 3 | d = 3 (Gleason 1957 frame functions) | Substrate is 3-dimensional |
| Row 4 | k* = 3 (Brown 1986 Fisher rank + Delgado-Friedrichs & O'Keeffe 2003) | 3-regular valence |
| Row 9 | g = 10 (girth from Sachs/Erdős-Sachs cage bound + edge-transitivity) | (3, 10)-cage among 3D crystal nets |
| Row 6 | Sunada 2012 strong-isotropy uniqueness + W4 arc-transitivity (2026-05-05) | srs is unique chiral arc-transitive 3D 3-regular crystal net |

Row 6 (`docs/audits/registers/uniqueness_ledger.md`) gives the load-bearing closure. The chain:

1. **Sunada 2012 uniqueness** (Sunada, *Notices AMS* 59(2), 208–215). srs is the unique 3-connected 3D crystal net that is **strongly isotropic** — whose crystallographic automorphism group acts transitively on (vertex, directed-edge) pairs.

2. **Chirality (R-7 closure 2026-04-26).** srs is chiral (space group I4₁32, no mirror or inversion). R-12 (`structural_residue_register.md`) is the chirality residue tied to weak-interaction parity violation; centrosymmetric 3D 3-regular crystal nets (ths, dia, eta, utj, honeycomb) cannot host R-12 and are hard-gated.

3. **W4 arc-transitivity prerequisite (2026-05-05).** Per `walker_dynamics_derivation.md` Step 4 amendment + `row4_audit_v2_revision_session2_2026-05-05.md`: for the per-step NB survival = (k-1)/k = 2/3 to be uniformly derived from Jaynes maximum entropy (rather than per-orbit-dependent), the substrate must be arc-transitive. Among chiral 3D 3-regular crystal nets, srs has 1 directed-edge orbit (vertex stabilizer C_3 acts transitively on the 3 outgoing edges per vertex); srs-z and other chiral non-arc-transitive RCSR entries have ≥ 2 directed-edge orbits (verified in `proofs/foundations/srs_vs_srs_z_dl_audit.py`) and are HARD-GATED by the W4 prerequisite.

Combining Sunada strong isotropy + chirality + arc-transitivity, srs is the UNIQUE survivor.

**R-9 closure status (`structural_residue_register.md`).** R-9 (full-MDL-spectrum lattice residue) restricts to the chiral subset post-R-7 (2026-04-26). It is now CLOSED via two independent paths:

- **γ.2 polynomial path (2026-05-02 EOD+8).** The framework's γ-polynomial uniqueness on srs Bloch dispersion gives algebraic-uniqueness within the chiral RCSR ensemble.
- **W4 arc-transitivity hard-gate (2026-05-05).** Sunada strong-isotropy + arc-transitivity hard-gates srs-z and similar non-arc-transitive chiral nets.

The W4 path is structurally cleaner; both paths are valid and mutually consistent.

**DL margin.** +1.68 bits to the nearest 3D competitor (ths net, DL = 13.85 vs srs's 12.17; per `proofs/foundations/dl_comparison.py`), but ths is centrosymmetric and hard-gated by R-12. The operative criterion is "chiral + minimum DL"; all non-srs entries in dl_comparison.py (ths, dia, eta, utj, honeycomb) are centrosymmetric. Finite graphs (Petersen, K₃,₃) excluded by infinite-structure upstream filter. ∎

**Type 1** (A1) + **Type 3** (Sunada 2012; Brown 1986; Gleason 1957; Sachs/Erdős-Sachs cage bounds; Delgado-Friedrichs & O'Keeffe 2003) + **Type 4** (uniqueness ledger Rows 3, 4, 6, 9; structural residue register R-7, R-9, R-12; `predictions/g_girth_derivation.md`; `predictions/walker_dynamics_derivation.md` Step 4; `proofs/foundations/dl_comparison.py`; `proofs/foundations/srs_vs_srs_z_dl_audit.py`; `row4_audit_v2_revision_session2_2026-05-05.md`).

**Status:** ✓ UNIQUE-CONDITIONAL on Sunada arc-transitivity (post-2026-05-05). Theorem-grade-conditional. Same conditional as Steps 2 and 4 (consistent across the chain).

### Step 6 — Sunada Bloch decomposition (Type 3 + Type 4; closed)

**Claim.** The NB walk kernel K on ℓ²(srs directed edges) decomposes as a direct integral K = ⊕_**k** B(**k**) d**k**/|BZ| over the srs Brillouin zone, where B(**k**) is the 12×12 Bloch-Hashimoto matrix at crystal momentum **k** acting on ℓ²(srs primitive-cell directed edges). At k_P = (1/4, 1/4, 1/4), every eigenvalue h of B(k_P) satisfies |h|² = k* − 1 = 2, saturating the Ramanujan bound.

**Proof (one-paragraph summary; full proof in `theorem_bloch_lift_mu.md` §3-5).**

The srs crystal has Z³ translation symmetry Γ ≅ Z³ acting on vertices and directed edges (property of A1 substrate + Step 5 srs identification). The NB walk kernel K[e', e] = 1 iff e → e' is a valid 1-step NB transition; this depends only on directed-edge incidence, which is Z³-periodic, so K commutes with every Γ-translation T_γ (`theorem_bloch_lift_mu.md` L1-L2). Sunada 2013 §6 Theorem 6.4 then decomposes any bounded Z^d-translation-invariant operator on the ℓ²-space of a Z^d-periodic structure as a direct integral over the dual torus (Brillouin zone), giving K = ∫_BZ B(**k**) d**k**/|BZ| where each fiber B(**k**) is the 12×12 Bloch-Hashimoto matrix on ℓ²(srs primitive-cell directed edges) (`theorem_bloch_lift_mu.md` L3). Cross-check in `theorem_lorentz_causal_sector.md` §2-3 constructs the same B(**k**) explicitly using the identical Sunada §6 argument; CAS-verified in `proofs/lorentz/b1_ags_audit.py`.

At the high-symmetry point k_P = (1/4, 1/4, 1/4), the adjacency eigenvalues are A(k_P) eigenvalues ±√3 each with multiplicity 2 (`predictions/srs_E_at_P.py`). Ihara-Bass (Terras 2011 §2.2) gives the Hashimoto eigenvalues h via h² − μ h + (k* − 1) = 0 for each adjacency eigenvalue μ; for μ = ±√3 and k* = 3, h = (±√3 ± i√5)/2 with |h|² = (3 + 5)/4 = 2 = k* − 1. The Ramanujan bound (Alon 1986) |h| ≤ √(k* − 1) is saturated exactly. ∎

**Type 1** (A1 substrate definition supplies Z³ symmetry) + **Type 2** (K depends only on NB incidence; trace formula for direct integrals; quadratic formula + Vieta) + **Type 3** (Sunada 2013 §6 Theorem 6.4; Terras 2011 §2.2 Ihara-Bass; Alon 1986 Ramanujan bound) + **Type 4** (`docs/theorems/theorem_bloch_lift_mu.md` C1-C3; `docs/theorems/theorem_lorentz_causal_sector.md` §2-3; `proofs/lorentz/b1_ags_audit.py`; `predictions/srs_E_at_P.py`).

**Status:** ✓ THEOREM-grade. Note: the conditional from Steps 2/4/5 (Sunada arc-transitivity, the 2026-05-05 W4 prerequisite) is independent of Step 6's Bloch decomposition, which uses Sunada 2013 §6 Theorem 6.4 (Bloch decomposition over Z^d-periodic structures) — a DIFFERENT theorem from Sunada 2012 (strong-isotropy uniqueness of srs). Step 6 only requires Z³ periodicity, not arc-transitivity. The Sunada arc-transitivity conditional comes in via Steps 2, 4, 5 (which are all upstream of Step 6's input).

---

## 4. The chain as an integrated theorem

The six steps integrate into the F_inv(E) → srs compression chain:

```
   Step 1: A1 + P1' → L²(F_inv(E); ℂ) with adjacency-Hamiltonian H
              ✓ THEOREM via theorem_A3_complex_hilbert_from_multiway.md
                          ↓
   Step 2: ARG-1 alphabet localization → |E| = 6 (srs primitive cell)
              ✓ THEOREM-CONDITIONAL on Sunada arc-transitivity (Rows 3, 4, 6, 7, 8 + R-11)
                          ↓
   Step 3: π = Π_MDL: ℋ_multiway^(L) → ℋ_visible^(L) ⊕ ℋ_dark^(L)
              ✓ THEOREM (Serre + Grünwald + H_multiway_dim_count)
                          ↓
   Step 4: visible reduced word ↔ NB walk on srs (Serre bijection)
           ✓ THEOREM-CONDITIONAL on Sunada arc-transitivity (W4 prerequisite)
                          ↓
   Step 5: srs identification (Sunada 2012 strong-isotropy, chiral, arc-trans)
              ✓ UNIQUE-CONDITIONAL on Sunada arc-transitivity (Rows 4, 6, 9 + R-7 + R-9)
                          ↓
   Step 6: Sunada 2013 §6 Bloch decomposition K = ⊕_k B(k) over BZ
              ✓ THEOREM (theorem_bloch_lift_mu, no extra conditional)
                          ↓
   Output: B(k) on ℓ²(srs unit cell directed edges) — visible-side
           framework's load-bearing computational object.
           Used by predictions/srs_E_at_P.py, predictions/B_P_doubly_degenerate_h.py,
           theorem_lorentz_causal_sector.md, etc.
```

### Integration check (NA-2' Session 4)

**Conditional consistency.** Steps 2, 4, 5 each carry the conditional "Sunada arc-transitivity = W4 Step 4 prerequisite" (per `row4_audit_v2_revision_session2_2026-05-05.md`). Steps 1, 3, 6 do not add conditionals:
- Step 1 conditions only on A1 + P1' + Stage 3 sub-Planckian correlation length (Type 4).
- Step 3 conditions only on A1 + A2-T (Type 1).
- Step 6 conditions only on Z³ periodicity (Type 1 from A1 + Step 5).

Note: Step 6 cites Sunada *2013* §6 Theorem 6.4 (Bloch decomposition over Z^d-periodic structures) — a DIFFERENT theorem from Sunada *2012* (strong-isotropy uniqueness of srs) that anchors Steps 2/4/5. No double-counting; the two Sunada citations are independent.

**Net conditional for the integrated theorem: UNIQUE-CONDITIONAL on Sunada (2012) arc-transitivity.** This is the SAME conditional as Row 4 + Row 6 of the uniqueness ledger and the same single conditional that gates ≥30 inheritance rows after the 2026-05-05 W4 prerequisite amendment.

**Hilbert-space chain.** The chain transitions Hilbert spaces between steps:

| After Step | Hilbert space | Mechanism |
|------------|---------------|-----------|
| 1 | L²(F_inv(E); ℂ) | A3-T |
| 2 | same, with E identified as srs primitive-cell edges | ARG-1 localization |
| 3 | L²(F_inv(E); ℂ) = ⊕_L (ℋ_visible^(L) ⊕ ℋ_dark^(L)) | π canonicalization |
| 4 | ℋ_visible^(L) ↔ NB walks of length L on srs | Serre bijection |
| 4' | ℓ²(srs directed edges) | Shalizi-Crutchfield causal-state minimality (history → state) |
| 5 | srs fixed; ℓ²(srs directed edges) = 12-dim per primitive cell × Z³-orbit | srs identification |
| 6 | ℓ²(srs directed edges) = ⊕_**k** ℓ²(srs unit cell directed edges) | Sunada 2013 §6 |

Step 4 has an internal sub-step (4'): the transition from history Hilbert space (length-L NB walks) to causal-state Hilbert space (current directed edge) via Shalizi-Crutchfield 2001 Theorem 2 (causal state = minimal sufficient statistic). This is consistent with `walker_dynamics_derivation.md` Step 5 and the Hashimoto operator B's domain ℓ²(directed edges).

**Type-typing consistency.** All six steps use Type 1 (A1, P1') + Type 3 (cited published math) + Type 4 (upstream framework theorem docs). No conflicts. Type 2 (algebra/calculation) appears in Steps 2, 3, 6.

**Output match.** The output ⊕_**k** B(**k**) on ℓ²(srs unit cell directed edges) is the operator family used by visible-side framework predictions:
- `predictions/srs_E_at_P.py` uses A(k_P) (adjacency eigenvalues, related to B by Ihara-Bass).
- `predictions/B_P_doubly_degenerate_h.py` uses B(k_P) directly.
- `theorem_lorentz_causal_sector.md` §2-3 uses B(**k**) for Stage 3 correlator analysis.
- `predictions/walker_dynamics_derivation.md` Step 6 identifies B as the 1-step amplitude operator.

The chain output matches the framework's load-bearing input across cosmology, mass operator, and Stage 3 derivations. ✓

### Net theorem status

**THEOREM-GRADE-CONDITIONAL on Sunada (2012) arc-transitivity.** All six steps closed; chain integration verified. The dominant-sector compression chain F_inv(E) → srs is now linted in one place with explicit Type 1/2/3/4 gate-typing, closing the §13.2 open question of `theorem_A3_complex_hilbert_from_multiway.md`.

---

## 5. What NA-2' Sessions 2-4 will do

**Session 1 ✓** — setup section + chain inventory.
**Session 2 ✓** — proofs for Steps 1-3 (substrate Hilbert space; ARG-1 alphabet localization; Layer-1 visible/dark decomposition).
**Session 3 ✓** — proofs for Steps 4-5 (ARG-2 geometric embedding via Serre + W4 arc-transitivity; srs identification via Sunada strong-isotropy + R-7 + W4).
**Session 4 ✓** — proof for Step 6 (Sunada 2013 §6 Bloch decomposition); integration check. Conditional consistency verified (single shared Sunada arc-transitivity conditional from Steps 2/4/5; Steps 1/3/6 add no conditionals); Hilbert-space chain documented; type-typing consistent; output matches visible-side framework's load-bearing input.

**NA-2' COMPLETE.** Net commitment was estimated at 2-4 sessions per an internal working note §3; closed in 4 sessions, on the upper end of estimate. The likely-surprise flagged for Step 4 (ARG-2 geometric embedding) resolved cleanly via existing walker_dynamics W1 closure.

**Hygiene follow-on (not part of NA-2', deferred):**
- The planned `docs/theorem_H_multiway_construction.md` companion doc (referenced 8+ times across framework but missing) would consolidate Step 3's Layer-1 construction in a sibling theorem doc. Recommended hygiene; not load-bearing for this chain.

---

## 6. Honest scope acknowledgement (Session 4 closing)

NA-2' Sessions 1-4 produced this theorem doc closing the §13.2 open question of `theorem_A3_complex_hilbert_from_multiway.md`. All six steps have theorem-grade proofs with explicit Type 1/2/3/4 gate-typing; integration verified.

**Likely-surprise resolution (Session 1 setup flagged three; all resolved):**
- ✓ Step 4 (ARG-2 geometric embedding): resolved cleanly via walker_dynamics W1 Serre bijection. R-3 refutation is the negative companion, NOT the closure mechanism.
- ✓ Step 3 (Layer-1 visible/dark): closed by citing `predictions/H_multiway_dim_count_derivation.md` directly; the missing planned theorem doc is hygiene, not load-bearing.
- ✓ Step 6 (Bloch decomposition operator-level identification): one-paragraph summary cites `theorem_bloch_lift_mu.md`; output ℓ²(srs unit cell directed edges) verified to match what visible-side framework predictions consume.

**Open follow-on work (NOT part of NA-2'):**
- NA-4 (Layer-1 observable that escapes Bloch averaging) — an internal working note §3.NA-4. UNBOUNDED multi-sprint research direction. Cosmology Item 5 + Λ_CC Path B + n_s all hinge on it.
- Planned `docs/theorem_H_multiway_construction.md` companion (Layer-1 visible/dark dim-count theorem doc) — referenced 8+ times across framework, missing. Hygiene gap; recommended but not load-bearing.

**Conditional dependency.** This theorem inherits the single Sunada (2012) arc-transitivity conditional that gates ≥30 cascade rows post-2026-05-05 W4 prerequisite amendment. The chain does NOT introduce new conditionals beyond what Row 4 + Row 6 already carry.

---

## 7. References

- Predecessor: an internal working note §3.
- Origin question: `docs/theorems/theorem_A3_complex_hilbert_from_multiway.md` §13.2.
- Layer-1 dim-count: `predictions/H_multiway_dim_count_derivation.md`.
- Bloch decomposition: `docs/theorems/theorem_bloch_lift_mu.md`.
- Multiway branch measure: `docs/theorems/theorem_multiway_branch_measure.md`.
- Row 4 + Row 6 + R-9: `docs/audits/registers/uniqueness_ledger.md` + `structural_residue_register.md`.
- Arc-transitivity prerequisite: an internal working note.
- H¹ master compression: `docs/theorems/theorem_h1_master_compression.md`.

## 8. Status

**THEOREM-GRADE** (NA-2' Sessions 1-4 complete 2026-05-05; status promoted 2026-05-23 after R-9 structural closure 2026-05-12 eliminated the Sunada arc-transitivity conditional). All six steps closed; chain integration verified. Closes the §13.2 open question of `theorem_A3_complex_hilbert_from_multiway.md`.

- Step 1 (substrate Hilbert space): A3-T closure chain.
- Step 2 (ARG-1 alphabet localization): Rows 3, 4, 6, 7, 8 + R-11.
- Step 3 (Layer-1 visible/dark decomposition): π = Π_MDL via Serre + Grünwald.
- Step 4 (ARG-2 geometric embedding): walker_dynamics W1 Serre reduced-word ↔ NB-walk bijection.
- Step 5 (srs identification): Sunada strong-isotropy + R-7 chirality + W4 arc-transitivity; R-9 closed via dual paths.
- Step 6 (Sunada Bloch decomposition): theorem_bloch_lift_mu.md.
- Integration check: single shared conditional; Hilbert-space chain consistent; type-typing consistent; output matches visible-side framework's load-bearing input.

**Net commitment:** 4 sessions (upper bound of `theorem_multiway_formalization_scoping_2026-05-05.md` §3 estimate). Audit-before-ansatz pattern preserved — closure mostly via consolidation of existing session work, not fresh derivation.

**Net unblock for the framework:**
- §13.2 open question of A3-T closed.
- The dominant-sector compression chain F_inv(E) → srs is now linted in one citable theorem doc.
- The framework's "F_inv(E) is the substrate Hilbert space input" and "srs Bloch decomposition is the compressed output" can now BOTH be cited as theorem-grade with the descent argument explicit.
