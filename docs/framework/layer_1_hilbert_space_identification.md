# Identification of the Layer 1 Hilbert space under A3

**Date:** 2026-04-17
**Status:** Organizational / structural doc. Not a standalone theorem. Pins down the specific Hilbert space that the multiway substrate (Layer 1 of `framework_architecture.md`) IS under axiom A3 (`framework_axioms.md` Section 4). Compares three candidates, evaluates each against the framework's existing content, and recommends one.
**Scope:** This doc answers the question A3 leaves open: A3 says Layer 1 is a pure-state Hilbert space whose partial trace over Layer 6 (dark) gives Layer 2 (visible). A3 does not say WHICH Hilbert space. This doc argues for a specific choice.
**Companion:** `framework_axioms.md` (A1 + A2 + A3), `docs/theorem_H_multiway_construction.md` (explicit length-graded construction), `predictions/observer_hilbert_space_derivation.md` (G.1 + G.5 closure under A3).
**Template:** Organizational doc, follows the house style of `framework_architecture.md`.

## 0. Clarification note (2026-04-17) -- abstract-auxiliary revision of A3

an internal working note Section 4.1 identified a tension between (a) the prior A3 statement ("pure state on Layer 1 tensor Layer 6") and (b) the Candidate 1 identification in this doc (H_multiway = H_visible (+) H_dark as a DIRECT-SUM, not a tensor product). The tension: A3 requires a tensor product per CDP 2011, but the direct-sum decomposition of H_multiway at each length L has dim H_multiway^(L) = 6^L while a tensor-product H_visible^(L) (x) H_dark^(L) would have dimension 6 * 5^(L-1) * 6 * [6^(L-1) - 5^(L-1)], which disagrees starting at L = 2.

**Resolution (adopted).** `framework_axioms.md` Section 4.1 (revised 2026-04-17) restates A3 with an ABSTRACT purifying auxiliary H_aux. CDP 2011's composite "system AB" is the operational tensor-product composition in their OPT framework; the purifying system B need not be a concrete subspace of any pre-existing Hilbert space. Under the revised A3:

- **Layer 1 identification (this doc's Candidate 1) unchanged.** H_multiway = +_L C^(|E|^L) remains the length-graded Hilbert-space lift of F_inv(E); its reduced-word subspace H_visible is the visible sector's Hilbert space; its cancellable-string complement H_dark remains a direct-sum summand inside H_multiway. The dim-count lemma of `predictions/H_multiway_dim_count.py` remains the CANONICAL KINEMATICAL statement about the MDL kernel inside H_multiway.
- **A3 purification is now on a SEPARATE tensor product H_visible tensor H_aux.** The abstract H_aux is not identified with H_dark as a subspace. H_aux's operational interpretation coincides with the physical dark sector (Layer 6 of `framework_architecture.md`) -- the observationally-inaccessible "uncompressed multiway residue" developed in an external research note on dark-matter compression and in `predictions/Omega_DM_over_Omega_m.py` -- but its mathematical identity is determined only up to the CDP essential-uniqueness clause, not by the F_inv(E) combinatorics.

**Consequence for this doc's Section 4.1, 4.3 evaluations.** The "MDL canonicalization = partial trace" row for Candidate 1 is REFINED: the projection H_multiway^(L) -> H_visible^(L) is the combinatorial / kinematical canonicalization map (projection onto reduced words); the A3 PURIFICATION partial trace is a separate operator-algebraic operation on H_visible tensor H_aux that asserts existence of H_aux producing the visible mixed state by partial trace. The two operations are conceptually related but mathematically distinct.

Candidate 1 is still the recommended choice, because:

- H_visible (reduced-word subspace of H_multiway) is directly identified as the visible-sector Hilbert space; this is exactly the "A" of A3's H_visible tensor H_aux tensor product. Candidate 1 supplies H_visible natively.
- The dim-count lemma, Schur-complement analysis, and Bloch fibre embedding all remain valid as statements about H_visible and its relation to Hashimoto B and gamma_phys = 1/16; none of these rely on identifying H_aux with H_dark.
- The abstract-auxiliary revision REMOVES the direct-sum-vs-tensor tension that was the main structural concern about Candidate 1.

**Consequence for Candidate 2 and Candidate 3.** Candidate 2 (Szegedy C^144) has a native tensor structure C^12 (x) C^12 but was scored against the OLD A3 wording that required Layer 6 to be a subspace. Under the revised A3, Candidate 2's failure modes (no faithful toggle algebra, no native MDL canonicalization) remain; it is still not the recommended choice. Candidate 3 (l^2(F_inv(E)) GNS) gains nothing from the revision.

**Consequence for Open Question 1 (this doc).** The open question "Layer 6 = H_dark consistent up to unitary?" becomes "does H_aux admit a canonical concrete realization from framework-native data (GNS on a dynamical C*-algebra, Feshbach reduction over H_dark, Davies-limit environment construction, etc.)?". This is a harder question than the original, because it asks for CONSTRUCTION of H_aux rather than IDENTIFICATION of H_aux with H_dark. Tracked separately.

**Downstream effect on an internal working note.** The Lindblad-from-A3 derivation remains BLOCKED, but for a slightly different reason: the Davies-limit derivation needs a concrete tensor-product structure H_S (x) H_E with a specific H_E and an H_int coupling. Under the revised A3, H_aux is abstract and H_int is not supplied; the stall is now "H_aux not concretely realized" rather than "H_dark is direct-sum not tensor." The operational consequence is unchanged.

**Scope of this note.** This is a clarification, not a new theorem. The Candidate 1 recommendation of this doc is unchanged; only the framing of what H_multiway "is" under A3 has been refined. See `framework_axioms.md` Section 4 (revised) for the authoritative A3 wording.

## 1. Status header

Under A1 + A2 + A3 (`framework_axioms.md`), Layer 1 is a pure-state Hilbert space. The Chiribella-D'Ariano-Perinotti 2011 (CDP) chain (Theorem 25 of their Section VIII) then fixes the visible-sector model class as density operators on a finite-dim complex Hilbert space, with the visible sector being the partial-trace image of a pure state on Layer 1 tensor Layer 6.

What CDP does NOT fix: the specific Hilbert space that Layer 1 IS. Three candidates survive as mathematically consistent with the framework's existing structure:

- **Candidate 1** (recommended) -- the length-graded free-involutive-monoid Hilbert space H_multiway = +_{L >= 0} C^(|E|^L), with H_visible = span of reduced words and H_dark = span of cancellable strings, per `docs/theorem_H_multiway_construction.md` and `predictions/H_multiway_dim_count.py`. Infinite-dim overall; each length-L slice is finite-dim n^L = 6^L.
- **Candidate 2** -- a Szegedy 2004 quantum-walk unitary lift on C^(2|E|) tensor C^(2|E|) = C^(144) for srs. Finite-dim. Faithfully unitarizes the Hashimoto B.
- **Candidate 3** -- the CDP-canonical GNS construction on the toggle algebra. The CDP derivation does not explicitly produce a Hilbert space; one takes the GNS construction with the multiway vacuum state and gets (as Dixmier 1977 Section 2.4.4 + Section 13.9 show) precisely the left-regular representation of F_inv(E) on l^2(F_inv(E)). Infinite-dim.

The recommendation below is Candidate 1. It is also the choice the parallel A3 formalization agent has made in `framework_axioms.md` Section 6 and `predictions/observer_hilbert_space_derivation.md`: both docs identify the length-graded H_multiway with H_visible + H_dark as the explicit Layer 1 Hilbert space whose partial trace gives Layer 2. This doc records the convergence and the rationale.

## 2. Context: A3 establishes Layer 1 is pure-state Hilbert; question is WHICH

A3 (`framework_axioms.md` Section 4) asserts:

> For every observer-accessible mixed state rho on Layer 2, there exists a pure state |psi> on Layer 1 tensor Layer 6 such that rho = Tr_{Layer 6}( |psi><psi| ). Moreover, this purification is essentially unique up to reversible transformations on the Layer 6 sector.

This commits the framework to an operator-algebraic picture: Layer 1 tensor Layer 6 is a Hilbert space, pure states on it reduce to density operators on Layer 2 by partial trace. CDP 2011 Theorem 25 then forces the state space on Layer 2 to be density operators on a finite-dim complex Hilbert space.

What A3 does NOT commit to:

1. Which specific Hilbert space structures the substrate Layer 1. The CDP axioms are operational; they fix the structure of the state space but do not uniquely name the Hilbert space.
2. How the toggle operators T_e of A1 act on this Hilbert space.
3. Whether Layer 1 is finite-dim, infinite-dim, length-graded, etc.

The framework has already built out substantial Layer 1 content in existing docs:

- `../../predictions/walker_dynamics_derivation.md` Step 1: A1 + monoid congruence gives F_inv(E) = free product of |E| = 6 copies of Z/2 (Serre 1980 Section I.1 Prop. 4).
- `../../predictions/walker_dynamics_derivation.md` Step 2: A2 (MDL) canonicalizes to reduced words (Grunwald 2007 Section 5.1-5.3).
- `docs/theorem_H_multiway_construction.md`: explicit length-graded Hilbert-space lift H_multiway^(L) = C^(|E|^L), with visible / dark decomposition by reduced vs cancellable.
- `predictions/H_multiway_dim_count.py`: closed-form dimensions dim H_visible^(L) = n*(n-1)^(L-1), dim H_dark^(L) = n*[n^(L-1) - (n-1)^(L-1)] verified by brute force for L up to 7.

Any candidate Hilbert space must at minimum (a) contain the F_inv(E) algebra faithfully (A1 content), (b) admit MDL canonicalization as a structural operation (A2 content), (c) support partial trace to Layer 2 (A3 content), and (d) reproduce the framework's existing derived quantities (Hashimoto B spectrum at the P-point, Bloch dispersion gamma = 1/16, R = 228/7, alpha_1 Feshbach coupling, the (4, 2, 2) C_3-isotypic decomposition).

## 3. Three candidates with precise mathematical descriptions

### 3.1 Candidate 1 -- length-graded free-involutive-monoid Hilbert space

**Hilbert space.**

    H_multiway := +_{L >= 0} H_multiway^(L),   H_multiway^(L) := C^(|E|^L) = tensor_{i=1}^L C^(|E|),

with |E| = 6 for the srs primitive cell.

**Basis.** For each L, the orthonormal basis is {|w> : w in E^L}, i.e. all length-L tuples of toggle labels without cancellation applied. Total basis is countable, parametrized by the free monoid E*.

**Visible / dark split.** On each length-L slice:

    H_visible^(L) := span { |w> : w in E^L reduced (no adjacent equal letters) },
    H_dark^(L)    := span { |w> : w in E^L, r(w) strictly shorter than w }.

Then H_multiway^(L) = H_visible^(L) + H_dark^(L) orthogonally, and H_visible = +_L H_visible^(L) is the framework's visible sector at Layer 2.

**Partial trace.** The canonicalization map pi : H_multiway -> H_visible is the A3 partial trace: for pure states |psi> in H_multiway, rho_visible = Tr_{H_dark}(|psi><psi|) is a density operator on H_visible. Per A3, every mixed rho on H_visible arises this way, essentially uniquely up to unitary on H_dark.

**Dimension.** Countably infinite. At each length L, finite dim n^L = 6^L, with dim H_visible^(L) = n*(n-1)^(L-1) = 6 * 5^(L-1) and dim H_dark^(L) = n^L - n*(n-1)^(L-1) = 6 * [6^(L-1) - 5^(L-1)]. Verified: `predictions/H_multiway_dim_count.py`.

**Structural properties.** Length-graded, carries a natural causal order (length = number of toggle events so far), explicitly decomposes into visible + dark per MDL canonicalization. Toggle operators T_e do NOT act as ordinary operators: on the unreduced space, any natural action fails T_e^2 = I (see an internal working note option (b) stalls). The T_e's act naturally on the quotient H_visible = l^2(F_inv(E)), not on H_multiway itself.

Asymptotic dark fraction dim H_dark^(L) / n^L = 1 - ((n-1)/n)^(L-1) -> 1 as L -> inf. Almost all sufficiently long strings are dark.

### 3.2 Candidate 2 -- Szegedy quantum-walk Hilbert space

**Hilbert space.** H_Szeg := C^(2|E|) tensor C^(2|E|) = C^((2|E|)^2) = C^144 for srs (2|E| = 12).

**Basis.** The tensor product basis {|a> tensor |b> : a, b directed edges}, dim = (2|E|)^2 = 144.

**Definition.** Szegedy 2004 Theorem 1 constructs a unitary operator U_Szeg on H_Szeg such that spec(U_Szeg^2) is in 2-to-1 correspondence with spec(K), where K = B / (k - 1) is the normalized Markov kernel on directed edges (k = 3 for srs, so K = B/2). The map lambda |-> exp(+- i arccos(lambda)) pairs each classical eigenvalue with a unit-complex pair.

**Visible / dark.** Not canonical. Szegedy's construction does not inherently distinguish a visible vs dark split. One could try to identify dark with the orthogonal complement of the "diagonal" subspace {|a> tensor |a>}, or with one of the two eigenspaces corresponding to complex-conjugate pairs. Neither is structurally forced by A3's partial-trace requirement.

**Partial trace.** Tr_2 (second tensor factor) gives a density operator on C^(2|E|) (the directed-edge space). But the directed-edge space is NOT Layer 2 per the framework; Layer 2 is the graph-theoretic Hashimoto spectrum and its Bloch decomposition, not the naked directed-edge space. The Szegedy partial trace does not naturally land on the 12-dim directed-edge Bloch fibre at P.

**Dimension.** Finite, 144. Per-Bloch-momentum decomposition exists but requires extra structure (the srs translation group action must be embedded into the Szegedy space by hand).

**Structural properties.** Finite-dim, canonical faithful unitarization of B. Carries the B spectral data (including the Ihara-Bass identity h^2 - h * Tr(B) + det(B) = 0) by Szegedy 2004 Theorem 1. Does not carry the MDL-canonicalization / cancellation structure -- there is no "dark absorbing class" on the Szegedy side; everything is already unitary.

**Relation to F_inv(E).** Szegedy's unitarization is defined on the doubled directed-edge space, not on F_inv(E). The link from the Szegedy space back to the F_inv(E) reduced-word structure is not canonical -- reduced words correspond to non-backtracking walks (Serre 1980; Terras 2011), and NB walks sit inside the Szegedy evolution as a specific subset, but there is no canonical isometric embedding F_inv(E) injective-hom-into H_Szeg.

### 3.3 Candidate 3 -- CDP / GNS-canonical Hilbert space

**Hilbert space.** The CDP 2011 derivation produces a finite-dim complex Hilbert space H as the state space of a given system, but the dimension of H is determined by the specific system (the number of perfectly-distinguishable states per CDP axiom 2). CDP does not fix which Hilbert space H IS; it says the state space of every system IS density operators on some finite-dim complex Hilbert space.

**GNS construction.** The canonical Hilbert space from a non-commutative C*-algebra via GNS is: given a C*-algebra A and a state omega on A, H_omega = completion of A / (null ideal of omega) with inner product <a | b>_omega = omega(a^* b). For the toggle C*-algebra C*_r(F_inv(E)) with the vacuum state omega = delta_epsilon (Dirac at the identity element), the GNS Hilbert space is

    H_GNS = l^2(F_inv(E))

with orthonormal basis {|w> : w in F_inv(E)} indexed by reduced words (Dixmier 1977 Theorem 2.4.4 + Section 13.9; Kesten 1959 for the spectrum of the regular representation).

**Dimension.** Countably infinite (one basis vector per reduced word).

**Basis.** {|w> : w in F_inv(E)}, i.e. the reduced words (NOT the unreduced strings as in Candidate 1).

**Visible / dark.** The CDP / GNS construction does NOT produce a visible / dark split. On l^2(F_inv(E)), every basis vector is already a reduced word; there is no "dark" subspace. Any dark content must be grafted on externally -- e.g. by tensoring with a second Hilbert space as "environment" -- but CDP / GNS does not say WHICH second Hilbert space.

**Structural properties.** On l^2(F_inv(E)), toggle operators T_e act faithfully as unitary self-inverse basis-permutation operators (an internal working note option (a)). Hashimoto B acts on the "length-2 reduced-word" sector, plus finite-rank corrections (Cartwright-Soardi 1986; Kesten 1959 spectrum -2 sqrt(n-1) to +2 sqrt(n-1) for n = |E|). Non-amenable (contains free group F_{|E|-1} as a subgroup, an internal working note Step A).

**Partial trace.** Not naturally defined. l^2(F_inv(E)) is not canonically a tensor product, so partial trace requires choosing a tensor-factorization externally. A3 would then require that choice to be the visible / dark split, but the construction provides no such split internally.

## 4. Evaluation against framework structure

We evaluate each candidate against five framework requirements:

(i) Toggle dynamics (Hashimoto B) act naturally on the Hilbert space.
(ii) MDL canonicalization pi : Layer 1 -> Layer 2 corresponds to a natural partial-trace structure.
(iii) srs lattice structure emerges naturally.
(iv) Visible Bloch fibre at P (12-dim directed-edge space) embeds naturally.
(v) Derived quantities (h, gamma_phys, (4, 2, 2), alpha_1) stay consistent.

### 4.1 Candidate 1 evaluation (length-graded H_multiway)

(i) **Toggle dynamics.** Per an internal working note, T_e cannot be simultaneously unitary AND involutive as an operator on the unreduced H_multiway -- the cancellation action is not a single-valued involution. BUT: the Markov extension operator B_full of `docs/theorem_H_multiway_construction.md` Section C IS well-defined on H_multiway; it is lower-triangular in the visible/dark split (B_VD = 0 because dark is absorbing in F_inv(E)). The L-step "visible to visible" block B_VV's spectrum, per `../../predictions/walker_dynamics_derivation.md` Step 8, reproduces the Hashimoto B spectrum on the directed-edge state space after restriction to graph-compatible reduced words. So Candidate 1 carries Hashimoto B on its visible sub-Hilbert space, in the form required by Layer 2 of `framework_architecture.md`. PASSES with a caveat: toggle operators are natural on the quotient F_inv(E), not on the unreduced H_multiway.

(ii) **MDL canonicalization = partial trace.** YES, canonically. The canonicalization map pi : H_multiway^(L) -> H_visible^(L) is (on the basis) the projection onto the reduced-word subspace. Under A3, this projection IS the partial trace over H_dark^(L) of pure states. `framework_axioms.md` Section 6 (the parallel A3 doc) makes this identification explicit: "the canonicalization map H_unred^(L) -> H_visible^(L) IS the partial trace over H_dark^(L)." PASSES (load-bearing match).

(iii) **srs lattice structure.** The alphabet E has |E| = 6 edges per primitive cell, a structural fact that is derived from A1 + A2 + `predictions/k_star.py` (k* = 3) and `predictions/d_spatial.py` (d = 3) and the srs Wyckoff structure (`predictions/g_girth.py`, RCSR srs entry). The free-involutive-monoid alphabet is exactly this E. Bloch decomposition (Sunada 2012 Section 5-6) factorizes H_multiway by lattice-translation momentum, identifying H_multiway^(L)(q) as the length-L Bloch fibre at momentum q. PASSES.

(iv) **Visible Bloch fibre at P.** The 12-dim directed-edge space at P (carrying h with multiplicity 2, (4, 2, 2) C_3 decomposition per `../../predictions/B_P_doubly_degenerate_h_derivation.md` and `docs/theorem_B5_3_core.md`) is a 12-dim sub-Hilbert space of H_visible at finite length. Specifically: the Hashimoto B operator lives on the 2|E| = 12 directed-edge states per primitive cell (`../../predictions/walker_dynamics_derivation.md` Step 6); this 12-dim space is the length-1 visible Bloch fibre at P of the directed-edge quotient of H_visible. PASSES.

(v) **Derived quantities.** Per Step F of `docs/theorem_H_multiway_construction.md`, the Schur complement reduction of B_full gives T_eff = B_VV identically (because B_VD = 0), so the visible Bloch dispersion gamma_phys = 1/16 of `predictions/srs_bloch_dispersion_gamma.py` is preserved. The Hashimoto B spectrum at P (h = (sqrt(3) + i sqrt(5))/2 with multiplicity 2, `predictions/h_walker_eigenvalue.py`, `predictions/B_P_doubly_degenerate_h.py`) is preserved. The (4, 2, 2) C_3-isotypic decomposition on the 8-dim Ramanujan subspace is preserved (`docs/theorem_B5_3_core.md`). alpha_1 Feshbach coupling: this is a separate Feshbach computation at the tree level and not affected by the Schur complement being trivial. PASSES.

**Net for Candidate 1: all five requirements PASS.** Candidate 1 is structurally exact.

### 4.2 Candidate 2 evaluation (Szegedy C^144)

(i) **Toggle dynamics.** Szegedy U_Szeg on C^144 is unitary by construction and carries spec(B^2) faithfully. But U_Szeg does NOT arise from T_e's naturally; its Markov data input is the non-backtracking walk kernel K = B / (k - 1), not the toggle generators. T_e's do not act on H_Szeg in any canonical way. PARTIAL MATCH (unitarization of B is natural; toggle-algebra faithfulness is NOT natural).

(ii) **MDL canonicalization = partial trace.** No canonical match. Partial trace over the second tensor factor of C^(2|E|) tensor C^(2|E|) gives a density operator on C^(2|E|), not on Layer 2 in the framework's sense. The Szegedy construction has no intrinsic visible / dark split; one must be imposed by hand. FAILS structurally.

(iii) **srs lattice structure.** The Szegedy construction uses directed edges as its index set. srs lattice structure enters only through the specific B matrix (its non-zero pattern follows srs incidence). Bloch decomposition is natural on the directed-edge index -- the 12-dim Bloch fibre at each q is a natural structure on C^(2|E|), and its tensor square is a 144-dim Bloch fibre at each q. PASSES.

(iv) **Visible Bloch fibre at P.** The 12-dim directed-edge Bloch fibre at P sits inside C^(2|E|) = C^12 naturally. The 144-dim tensor square contains it as a rank-1 tensor-product subspace. PASSES (but by projection, not by natural embedding with a partial-trace structure).

(v) **Derived quantities.** By Szegedy 2004 Theorem 1, spec(U_Szeg^2) is in 2-to-1 correspondence with spec(K), so B's eigenvalues (including h) are recoverable. gamma_phys = 1/16 is a property of B's low-q expansion, hence recoverable. But the A3 partial-trace structure is NOT natural here, so the CDP chain to finite-dim complex Hilbert space is not automatic. PARTIAL.

**Net for Candidate 2: three out of five PASS, two FAIL.** Specifically, MDL canonicalization is not natural on H_Szeg (no visible / dark split intrinsic to the Szegedy construction), and toggle algebra is not natural (T_e's don't act canonically). Szegedy is a good unitarization OF LAYER 2's spectral data; it is not a good candidate for LAYER 1.

### 4.3 Candidate 3 evaluation (l^2(F_inv(E)) GNS)

(i) **Toggle dynamics.** T_e acts naturally as unitary self-inverse basis-permutation (an internal working note option (a)). Toggle algebra is faithfully represented (C*_r(F_inv(E)) by construction; Dixmier 1977 Section 13.9). PASSES.

(ii) **MDL canonicalization = partial trace.** l^2(F_inv(E)) IS the image of the MDL canonicalization -- every basis vector is already reduced. There is no "unreduced" superstructure to trace out. The partial-trace structure that A3 requires (Tr_{Layer 6} of a pure state in Layer 1 tensor Layer 6) does NOT correspond to any canonical operation on l^2(F_inv(E)) alone. To get A3's partial trace, one must tensor l^2(F_inv(E)) with another Hilbert space as "the dark environment" -- but GNS / CDP does not specify what that environment Hilbert space is. FAILS.

(iii) **srs lattice structure.** The alphabet E has |E| = 6; the free product structure depends on |E| and nothing else about the srs lattice's geometry (translation, point group). Bloch decomposition on l^2(F_inv(E)) requires the non-amenable group F_inv(E) to act by translations, which it does not. PARTIAL (alphabet size enters; lattice geometry does not).

(iv) **Visible Bloch fibre at P.** l^2(F_inv(E)) has no natural length-grading compatible with srs translations, so the 12-dim directed-edge Bloch fibre at P is not naturally embedded. One can project the "length 2 reduced" subspace onto directed-edge states, but this is not a natural isometric embedding. PARTIAL.

(v) **Derived quantities.** The Cartwright-Soardi 1986 spectrum [-2 sqrt(n-1), 2 sqrt(n-1)] on l^2(F_inv(E)) under sum_e T_e is the tree-Hashimoto spectrum, not the srs-Hashimoto spectrum. h = (sqrt(3) + i sqrt(5))/2 is an srs-specific eigenvalue coming from the finite-graph Ihara-Bass identity h^2 - h*Tr(B(P)) + det(B(P)) = 0, NOT from the infinite-tree spectrum. So the srs-specific derived quantities are NOT on l^2(F_inv(E)) directly. FAILS.

**Net for Candidate 3: two out of five PASS, two FAIL, one PARTIAL.** l^2(F_inv(E)) is structurally correct for the TOGGLE ALGEBRA but NOT for the srs LATTICE-SPECIFIC content that Layer 2 carries. It is also not naturally decomposed into visible + dark for A3.

## 5. Recommended choice + rationale

**Recommendation: Candidate 1** (the length-graded H_multiway = +_L C^(|E|^L) with its canonical visible / dark decomposition).

**Rationale.**

- Unique candidate passing all five framework requirements (Section 4). Candidates 2 and 3 each fail the MDL-canonicalization-equals-partial-trace requirement, which is the defining structural content of A3.
- Matches the parallel A3 formalization: `framework_axioms.md` Section 6 and `predictions/observer_hilbert_space_derivation.md` both identify this space as the explicit Layer 1 Hilbert space. CONVERGENCE.
- Already fully constructed in `docs/theorem_H_multiway_construction.md` with brute-force-verified dim-count lemma in `predictions/H_multiway_dim_count.py`. No new construction needed.
- Carries the derived srs-lattice-specific content natively: the visible sub-space H_visible contains the directed-edge Bloch fibre at P (via the reduced-word / NB-walk bijection of `../../predictions/walker_dynamics_derivation.md` Step 3), and the visible Schur-reduced dynamics preserve gamma_phys = 1/16 (Step F of `docs/theorem_H_multiway_construction.md`).
- Respects the framework's structural commitments: axiom A1 (toggle acts on the index F_inv(E)), axiom A2 (MDL canonicalizes to reduced words; dark is the kernel of canonicalization), axiom A3 (dark is the partial-trace target).
- The alphabet |E| = 6 is forced by upstream srs derivations (`predictions/k_star.py`, `predictions/d_spatial.py`, `predictions/g_girth.py`); it is not a free parameter.

**The minor caveat** from an internal working note option (b) -- that T_e does not act naturally on the unreduced H_multiway as a well-defined unitary involution -- is resolved in Candidate 1 by the framework's reading. Under A3, the T_e's are NOT required to act on all of Layer 1. They naturally act on the quotient F_inv(E), and lift to a Markov (not unitary) extension B_full on H_multiway per `docs/theorem_H_multiway_construction.md` Section C. The unitarity of Layer 1 is NOT that T_e is unitary on H_multiway; it is that some pure state |psi> on H_multiway tensor Layer 6 purifies Layer 2's mixed states per A3. The dynamics on Layer 1 are generated by the toggle algebra acting on its CANONICAL QUOTIENT (F_inv(E)), not by single-step T_e action on the unreduced space.

This reading also matches the Reading Conventions (3) Open System of `../../predictions/walker_dynamics_derivation.md`: the visible sector is the unitary Hilbert factor (what Layer 2 is after the CDP chain fixes its structure), and the dark sector appears Markov from the visible side because it is the partial-trace target of A3.

## 6. Consistency with B7.1 C^3 observer Hilbert space

`../../predictions/observer_dim_three_derivation.md` (B7.1) derives that the observer's OBSERVATIONAL Hilbert space is C^3, via MDL + non-contextuality + Gleason 1957. This is a DIFFERENT Hilbert space from Layer 1 of Candidate 1:

- Layer 1's H_multiway is the full substrate's pure-state Hilbert space (infinite-dim, length-graded).
- B7.1's C^3 is the observer's INTERNAL probability-assignment Hilbert space (finite-dim, abstract three-basis Hilbert space, identified with three SM generations per an internal working note).

**Does B7.1 become a theorem about Layer 1 -> C^3 partial trace?** Partially, with caveats.

Under A3 + CDP 2011 Theorem 25, the observer's model class on Layer 2 is density operators on a finite-dim complex Hilbert space. The specific finite dim is fixed by CDP axiom 2 (perfect distinguishability) applied to the observer's measurement frames. For the observer's OBSERVATIONAL Hilbert space -- the space of pure orthogonal measurement outcomes -- B7.1 applies MDL + non-contextuality + Gleason to fix this dim at n = 3.

So the chain is:

1. A1 + A2 + A3 -> (via CDP 2011 Theorem 25) -> observer model class is density operators on some finite-dim complex Hilbert H_obs.
2. A1 + A2 + Gleason (B7.1, given H_obs exists) -> dim H_obs = 3.
3. Therefore H_obs = C^3.

Under this chain, the Layer 1 Hilbert space (Candidate 1) is the FULL substrate Hilbert space, and the observer's C^3 is a SPECIFIC FINITE-DIM SUBSPACE of H_visible that corresponds to the probability-assignment / measurement-frame structure of an orthogonal triple.

**Is H_obs = C^3 the partial-trace image of H_multiway?** Not directly in the naive sense. The partial-trace image of H_multiway tensor Layer 6 is H_visible (entire visible sector), which is infinite-dim. H_obs = C^3 sits inside H_visible as the specific measurement-frame subspace. The two-step reduction is:

    H_multiway tensor Layer 6 (pure) --(partial trace)--> H_visible (mixed, infinite-dim) --(projection onto observer's measurement frame)--> C^3.

The second projection is not a partial trace in the standard operator-algebraic sense; it is the projection that B7.1's MDL + Gleason argument selects (the observer's minimum-parameter frame function), and it is where B7.1 does work that A3 does not do.

**So B7.1 does NOT become "just" a theorem about Layer 1 -> C^3 partial trace.** B7.1 does load-bearing work that A3 does not: selecting the observer's MEASUREMENT FRAME within the visible Hilbert space. A3 fixes the structural shape (density operators on a complex Hilbert space); B7.1 fixes which three-dim subspace the observer's measurements operate on.

The two theorems are complementary, not redundant. A3 + CDP closes G.1 (Hilbert space exists) and G.5 (complex field); B7.1 closes G.4 (dim = 3) within the now-derived Hilbert space structure. Together with G.2 (non-contextuality) and G.6 (positivity + trace 1), the full seven-assumption audit of an internal audit of the seven Gleason sub-assumptions is closed under A1 + A2 + A3 except for the separately-load-bearing Need-A2 (canonical generation-Z_3 on C^3_gen) and Need-RR (reading rule from substrate amplitude to mass parameter).

## 7. Open questions

1. **Explicit construction of the Layer 6 Hilbert space.** A3 asserts purification on Layer 1 tensor Layer 6, with Layer 6 essentially unique up to unitary. The framework's existing Layer 6 content (`framework_architecture.md`, an internal working note, `predictions/Omega_DM_over_Omega_m.py`) describes dark as uncompressed multiway residue. Identifying Layer 6 with H_dark = +_L H_dark^(L) is natural under Candidate 1, but the essential-uniqueness clause of A3 requires checking that this identification is consistent (up to unitary on H_dark). Open.

2. **Dynamics on Layer 1.** A3 says Layer 1's state is pure but does not supply a Hamiltonian. The framework's natural candidate is some self-adjoint operator H_full on H_multiway whose restriction to H_visible reproduces the Hashimoto B's eigendynamics and whose off-diagonal V <-> D blocks provide the dark-visible coupling needed for mass-as-flux. Per an internal working note, no natural T_e-based H_full exists on the unreduced H_multiway; the dynamics must come from some other operator (candidates: a Feshbach-projected effective Hamiltonian; a Lindblad generator as in Reading 3 of `../../predictions/walker_dynamics_derivation.md`; a Szegedy-style unitarization lifted to H_multiway). Open.

3. **Finite-dim truncation of H_multiway.** The framework's downstream work -- Bloch fibres at the P-point, Hashimoto B at 12 dim, (4, 2, 2) C_3 decomposition on 8-dim Ramanujan subspace, Koide Q = 2/3 at 3 dim -- operates in finite-dim contexts. The mapping from infinite-dim H_multiway to these finite-dim working spaces is the "CDP axiom 3 (ideal compressions)" step, per Section 7 of `framework_axioms.md`. Explicit construction of the specific compressions that produce the Bloch fibre at P (from H_multiway length-L content) is open.

4. **Compatibility of Candidate 1 with the Szegedy unitarization (Candidate 2).** A3 + CDP fix the structural shape of Layer 2 (finite-dim complex Hilbert space with density operators). The Szegedy 2004 unitarization of Hashimoto B on C^(2|E|) tensor C^(2|E|) is a concrete finite-dim complex Hilbert-space representation of Layer 2's spectral data. Whether the CDP-derived Layer 2 Hilbert space IS (canonically isomorphic to) the Szegedy space, or is a different finite-dim complex Hilbert space whose spectral data coincides with Szegedy's, is open. Resolving this is part of the Need-RR reading-rule closure: if Layer 2 IS the Szegedy space, then the reading rule from substrate amplitude to observable is forced to the Szegedy rule; if not, alternative rules remain on the table.

5. **The GNS connection.** Candidate 3 (l^2(F_inv(E)) via GNS on the toggle C*-algebra) is structurally valid for the toggle algebra but not directly useful as the Layer 1 Hilbert space. One conjecture: H_visible at fixed length L (the reduced-word sub-Hilbert space of Candidate 1) embeds isometrically into l^2(F_inv(E)) via the length-L-truncation map |w> |-> |w> in F_inv(E). Under this embedding, Candidate 1's visible sector sits inside Candidate 3's Hilbert space as the length-graded subspace. This would unify Candidates 1 and 3 via Candidate 1 = length-graded LIFT of Candidate 3 + dark complement. Open (explicit isometry not constructed).

6. **Relationship between H_multiway partial trace and the C^3 observer Hilbert space of B7.1.** Per Section 6, the reduction is two-step: H_multiway tensor Layer 6 -> H_visible (partial trace) -> C^3 (observer's measurement frame). The second step is Gleason + MDL selection of the minimum-parameter non-contextual frame. Whether this second step can be recast as a further partial trace (over some operator-algebraic structure "above C^3") is open.

7. **A3-uniqueness across the three candidates.** A3 says the purification is essentially unique up to unitary on Layer 6. If Candidates 1 and 3 are related (per Open Question 5) and both produce the same Layer 2 visible sector, the essential-uniqueness clause reconciles them. If Candidate 2 (Szegedy) also produces the same Layer 2 spectral data (per Szegedy 2004 Theorem 1), then Candidate 2 is a UNITARILY-EQUIVALENT purification of Layer 2, with Layer 6 being the complementary Szegedy factor -- but this requires an explicit unitary isomorphism between the three candidates as A3-purifications of the same Layer 2 mixed state. Open.

## 8. Per-candidate gate-clear status

Against the parameter-linter hard gate (1 axiom / 2 explicit algebra / 3 cited theorem / 4 upstream closed file):

| Candidate | Gate-clear summary |
|---|---|
| 1 (length-graded H_multiway) | Passes: A1 (toggle alphabet), A2 (MDL canonicalization), A3 (partial trace); Serre 1980 Section I.1 (F_inv(E)); Sunada 2012 Section 5-6 (Bloch decomp); `predictions/H_multiway_dim_count.py` (dim count); `docs/theorem_H_multiway_construction.md` (six-step construction); `../../predictions/walker_dynamics_derivation.md` (W1-W3). All six load-bearing references are closed at theorem grade. |
| 2 (Szegedy C^144) | Passes at cited-theorem level: Szegedy 2004 Theorem 1. But fails framework-requirement-(ii) (MDL canonicalization is not naturally a partial trace on H_Szeg) and framework-requirement-(i) (toggle algebra is not naturally faithful). |
| 3 (l^2(F_inv(E)) GNS) | Passes at cited-theorem level: Dixmier 1977 Theorem 2.4.4, Section 13.9; Serre 1980 Section I.1; Kesten 1959. But fails framework-requirement-(ii) (no visible / dark split) and framework-requirement-(v) (srs-specific spectral data is tree-Hashimoto on l^2(F_inv(E)), not srs-Hashimoto; Cartwright-Soardi 1986 vs Ihara-Bass). |

Only Candidate 1 is both cited-theorem-grade AND passes all framework-requirement checks. **Recommendation confirmed: Candidate 1.**

## 9. Summary table

| Requirement | Candidate 1 | Candidate 2 | Candidate 3 |
|---|---|---|---|
| Hilbert space precisely identified | Yes (length-graded H_multiway) | Yes (C^144 Szegedy) | Yes (l^2(F_inv(E))) |
| Toggle algebra A1 faithful | Via quotient to F_inv(E) | No (T_e not natural) | Yes |
| MDL canonicalization A2 = partial trace | Yes (native) | No (no V/D split) | No (no V/D split) |
| A3 purification structure | Yes | Partial (tensor square not a V/D purification) | No (not a tensor product naturally) |
| srs Bloch structure | Yes (Sunada) | Yes (by projection) | No (tree spectrum) |
| Hashimoto B at P (h, mult 2) | Yes (on H_visible) | Yes (by Szegedy) | No (tree) |
| gamma_phys = 1/16 preserved | Yes (Schur trivial) | Yes | No (tree) |
| C_3 (4, 2, 2) at P preserved | Yes | Yes | N/A |
| Parallel A3 agent agreement | Yes | No | No |
| Dim | Infinite, length-graded | Finite 144 | Countably infinite |
| Recommended? | YES | No | No |

## 10. Scope honesty

This doc is not a theorem proving that Candidate 1 is the UNIQUE Hilbert space consistent with A3. The uniqueness-up-to-reversible-transformations clause of A3 implies that if two candidates produce the same Layer 2 visible mixed state, they are unitarily equivalent as purifications. It is possible that Candidates 1, 2, 3 (or suitable sub-versions) are all unitarily equivalent as A3-purifications -- in which case the choice between them is a matter of presentation, not structure. The present recommendation of Candidate 1 is based on (a) framework-native fit, (b) convergence with the parallel A3 agent's identification, and (c) existing framework infrastructure (dim count, V/D split, Schur structure) already built on Candidate 1.

If downstream work (Feshbach coupling at P, Lindblad dark dynamics, B7.3 mass operator, Need-RR reading rule closure) reveals structural features that single out Candidate 2 or Candidate 3, this recommendation can be revisited. For now, Candidate 1 is the natural working assumption.

No new prediction file is produced by this doc. It is a structural / organizational doc only. No commits performed; no remote push.

## 11. References

### Memory / standards

- `../parameters/parameter_linter.md` -- hard quality gate.

### Upstream framework theorems (closed or adopted)

- `framework_axioms.md` -- A1 + A2 + A3 canonical statement (parallel A3 formalization).
- `framework_architecture.md` -- multi-layer view; Layer 1 multiway substrate, Layer 2 visible srs, Layer 6 dark sector.
- `../../predictions/walker_dynamics_derivation.md` -- W1 + W2 + W3 + Reading Conventions.
- `../../predictions/B_P_doubly_degenerate_h_derivation.md` -- B(P) spectrum, h-eigenvalue multiplicity 2, C_3-protected.
- `docs/theorem_H_multiway_construction.md` -- explicit length-graded H_multiway = H_visible + H_dark construction, Schur analysis.
- `../../predictions/observer_dim_three_derivation.md` (B7.1) -- observer Hilbert space dim = 3 via MDL + Gleason.
- `docs/theorem_B5_3_core.md` -- (4, 2, 2) C_3-isotypic decomposition on 8-dim Ramanujan subspace.
- `predictions/H_multiway_dim_count.py` + `predictions/H_multiway_dim_count_derivation.md` -- dim-count lemma.
- `predictions/observer_hilbert_space.py` + `predictions/observer_hilbert_space_derivation.md` -- G.1 + G.5 closure under A3 (parallel A3 agent).
- `predictions/h_walker_eigenvalue.py` -- h = (sqrt(3) + i sqrt(5))/2.
- `predictions/B_P_doubly_degenerate_h.py` -- 12-dim directed-edge Bloch fibre at P.
- `predictions/srs_bloch_dispersion_gamma.py` -- gamma_phys = 1/16.
- `predictions/k_star.py`, `predictions/d_spatial.py`, `predictions/g_girth.py` -- srs lattice derivations (|E| = 6, |V| = 4, k* = 3, d = 3, g = 10).
- `predictions/Omega_DM_over_Omega_m.py` -- cosmological dark/luminous ratio.

### Cited mathematical theorems

- **Chiribella, G., D'Ariano, G.M., Perinotti, P.** (2011). Informational derivation of quantum theory. *Phys. Rev. A* **84**, 012311. Theorem 25 (Section VIII): five-axiom derivation of finite-dim complex Hilbert-space QM.
- **Szegedy, M.** (2004). Quantum speed-up of Markov chain based algorithms. *FOCS 2004*, 32-41. Theorem 1: unitary U_Szeg on doubled directed-edge space with spec(U_Szeg^2) = spec(Markov kernel) up to 2-to-1 correspondence.
- **Dixmier, J.** (1977). *C*-algebras.* North-Holland. Section 2.4.4 (GNS construction), Section 13.9 (reduced group C*-algebra on l^2(G)).
- **Serre, J.-P.** (1980). *Trees.* Springer. Section I.1 Proposition 4 (free involutive monoid; reduced-word canonical form).
- **Kesten, H.** (1959). Symmetric random walks on groups. *Trans. Amer. Math. Soc.* **92**, 336-354. Spectrum of regular representation of free product of cyclic groups of order 2.
- **Cartwright, D.I., Soardi, P.M.** (1986). Random walks on free products, quotients and amalgams. *Math. Z.* **191**, 1-12. Infinite-tree Hashimoto spectrum.
- **Gleason, A.M.** (1957). Measures on the Closed Subspaces of a Hilbert Space. *J. Math. Mech.* **6**, 885-893. Used in B7.1 for dim = 3 closure.
- **Sunada, T.** (2012). *Topological Crystallography.* Springer Surveys and Tutorials in the Applied Mathematical Sciences, Vol. 6. Sections 5-6 (Bloch decomposition of periodic graphs).
- **Grunwald, P.** (2007). *The Minimum Description Length Principle.* MIT Press. Sections 5.1-5.3 (model equivalence and canonicalization).
- **Jaynes, E.T.** (1957). Information theory and statistical mechanics. *Phys. Rev.* **106**, 620-630. Max-entropy uniform-on-alphabet measure.
- **Terras, A.** (2011). *Zeta Functions of Graphs.* Cambridge University Press. Section 2.1-2.2 (NB walks, Hashimoto matrix).
- **Reed, M., Simon, B.** (1972). *Methods of Modern Mathematical Physics, I.* Academic Press. Section VIII.3-4 (self-adjoint operators; spectral theorem).

### Files referenced (read-only) but NOT modified

- `results/parameters.csv`
- `../parameters/derivations.md`
- All B3 / B5 / B6 docs
- All existing `predictions/` files
- All sibling scoping / attempt docs

The only new file produced by this task is the present doc. No edits to any other file.
