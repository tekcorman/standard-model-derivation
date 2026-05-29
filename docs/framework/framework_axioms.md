# Framework Axioms — canonical statement (POST 2026-05-08 A1-elimination: top-level is (A) self-containment + (B) finite observer + A5-mass)

**Date of major revision:** 2026-05-08. Prior major revisions: 2026-05-02 (P1'-elimination); 2026-04-26 (A2/A3/A4 demotion to theorems); 2026-04-18 (original A3 promotion); 2026-04-17 (auxiliary-is-abstract); 2026-04-19 (A4 + A5 additions).

**Status (post-2026-05-08 A1-elimination):** A1 itself is demoted from structural axiom to derived theorem (`../theorems/theorem_toggle_from_self_containment.md`, 2026-05-07). The framework's top-level commitments are now: **(A)** self-containment of the universe (metaphysical, not derivable), **(B)** finite observer (scoping definition, not a physical postulate), **(I)** active reading of binary distinctions (interpretive commitment, motivated by relational stance), plus **A5-mass** as downstream empirical labeling. A1, P1', A2, A3, A4 are all derived theorems of (A) + (B) + (I) + standard published mathematics. Substrate agnosticism (`../theorems/theorem_substrate_agnosticism.md`, 2026-05-08) establishes that the framework's predictions are invariant under choice of substrate within an observational equivalence class; the Cayley graph of F_inv(E) is the description-length-minimal canonical representative. See §10 Summary table for the updated slate. The five-axiom presentation in §§2-5b below is preserved as the historical record of what the framework asserted prior to the 2026-04-26 / 2026-05-02 / 2026-05-08 revisions; the *content* of A1, A2, A3, A4 is preserved (now as theorems) — only their *axiomatic status* changes.

**Theorem documents demoting prior axioms to derived theorems:**
- `../theorems/theorem_toggle_from_self_containment.md` (A1 → theorem of (A) + (B) + Shannon-Jaynes + active reading)
- `../theorems/theorem_substrate_agnosticism.md` (companion theorem on observational equivalence class; canonical Cayley-graph representative)
- `../theorems/theorem_p1_prime_derived_from_a1.md` (P1' → theorem; under the new slate, MR1/MR2/MR3 framing is subsumed: MR1 → (A), MR2 → (B), MR3 → consequence of (B) + standard finite-computation theory)
- `../theorems/theorem_A2_mdl_from_finite_register.md` (A2 → theorem of finite-register source coding)
- `../theorems/theorem_A3_complex_hilbert_from_multiway.md` (A3 → theorem via multiway-level Stone)
- `../theorems/theorem_car_local_jordan_wigner.md` (A4 → local theorem via Jordan-Wigner; global remains open but not load-bearing)

**Foundational catalog:** `../operator_sweep/operator_sweep_from_A1.md` enumerates ~180 mathematical operations the framework permits, layer by layer, with field selection (A1+P1'→ℂ) and continuum-limit closure as interleaved derivations.

**Uniqueness audit + residue register (2026-04-26):** `../audits/registers/uniqueness_ledger.md` audits each load-bearing structural claim by characterising its operator-permitted alternative set and the selection criterion that picks the framework's claim out of that set. `../audits/registers/structural_residue_register.md` is the canonical living register of all soft-gated discarded alternatives (R-N entries) — alternatives the framework eliminates by a finite MDL margin rather than by an exact algebraic zero, which under A2-T waterline reading carry non-zero downstream weight and may produce artifacts the framework has not yet computed. Every new prediction or theorem must check the register for residues affecting its target observable.
**Scope:** This document is the authoritative statement of which axioms the framework presupposes. Every theorem, prediction, and audit in the repo must trace its derivation chain back to (a) one or more of A1, A2, A3, A4, A5, (b) explicit algebra, (c) a cited mathematical theorem, or (d) an upstream framework file that itself passes this gate recursively.
**Supersedes:** the implicit two-axiom framing in earlier docs that referred to "MDL + binary self-inverse toggle as the ONLY axioms" (the an internal note formulation, written prior to A3 promotion). Also supersedes the prior concrete-subspace reading of A3 ("pure state on Layer 1 tensor Layer 6" with Layer 6 identified as H_dark subspace of H_multiway); the revised reading is the abstract-auxiliary form of Section 4.1.

## 1. Historical note

The framework was founded on a two-axiom set: A1 (binary self-inverse toggle) and A2 (MDL canonicalization). Under that two-axiom set, the Layer 0-2 stratum (toggle combinatorics, MDL canonical reduced words, srs lattice geometry, Hashimoto walker spectral data, B(P) multiplicities, the Ihara-Bass quadratic identity h^2 - h*Tr(B) + det(B) = 0, R = 228/7, gamma = 1/16, c1 photon bundle, R_nu_splitting Chebyshev identity, theta_QCD discrete-Z3 holonomy) is rigorously derivable. See an internal strict-gating audit for the full STRICT-SOLID inventory of 13 prediction files at Layer 0-2 under the two-axiom setup.

The Layer 4+ stratum (observer Hilbert space, Born rule, density operators, frame functions, Lindblad master equations, mass operator, Q_Koide, all CKM/PMNS angles and phases, Higgs, dark corrections) was found to depend on at least one of seven sub-assumptions (G.1 - G.7 of an internal audit of the seven Gleason sub-assumptions) that are NOT derivable from A1 + A2 alone. Specifically:

- **G.1 ASSUMED:** the observer's MDL-optimal model class is a Hilbert space (vector space + inner product).
- **G.5 GAP:** the field of that Hilbert space is C (not R, not H).

Both stalls were attacked head-on. an internal working note worked the parameter-count / Szegedy / Cencov-L^2 route and found that MDL strictly DISPREFERS any Hilbert-space lift over the bare probability simplex on Markov data (more parameters, no data-fit benefit). an internal working note worked the Gelfand-Naimark / non-commutative C*-algebra route and found that the observer's MDL-optimal model is the directed-edge Markov causal-state quotient (Shalizi-Crutchfield 2001), which is unfaithful by construction; faithfulness is not MDL-forced. Both attempts converge on the conclusion: A1 + A2 alone are insufficient to derive Hilbert-space structure on the observer's model class. The framework needs a third axiom or it has to accept that the entire Layer 4+ stack rests on undischarged structural inputs.

The user authorized addition of A3 on 2026-04-18 after extensive trade-space exploration. A3 codifies an existing framework commitment (the multiway substrate is a pure state, the MDL canonicalization map is a partial trace over the dark sector, the dark sector is definite-but-inaccessible) rather than adding new physical content. Under A1 + A2 + A3, G.1 and G.5 are DERIVED via the Chiribella-D'Ariano-Perinotti 2011 chain, and the 26 BLOCKED predictions of an internal strict-gating audit become STRICT-SOLID-CONDITIONAL on A3 (modulo the separately load-bearing Need-A2, Need-RR, Pati-Salam labeling, and Feshbach Exponent Principle gaps documented elsewhere).

**2026-05-07/08 revision: A1 itself derived.** Following the 2026-05-02 P1' elimination, the only remaining structural axiom was A1. Session work on 2026-05-07/08 (`../theorems/theorem_toggle_from_self_containment.md`, `../theorems/theorem_substrate_agnosticism.md`) derived A1 itself from two top-level commitments: **(A)** self-containment of the universe (metaphysical) + **(B)** finite observer (scoping), plus standard Shannon-Jaynes-Serre mathematics + one explicit interpretive commitment **(I)** (active reading of binary distinctions). The substrate-agnosticism theorem additionally establishes that the framework's predictions are invariant under choice of substrate within an observational equivalence class, with the Cayley graph of F_inv(E) as the description-length-minimal canonical representative. The framework's top-level slate is therefore now (A) + (B) + (I) + A5-mass, with A1, A2, A3, A4, P1' all derived theorems. The prior P1' theorem's MR1/MR2/MR3 framing is subsumed: MR1 → (A); MR2 → (B); MR3 → consequence of (B) + standard finite-computation theory.

## 2. A1 -- Binary self-inverse toggle

**Status (post-2026-05-08):** A1 is now a **derived theorem** of (A) + (B) + standard math + (I). See `../theorems/theorem_toggle_from_self_containment.md`. The statement below is preserved as the historical content of A1; downstream derivations that cite "A1" can now be read as citing "the theorem of (A) + (B) + (I) yielding F_inv(E)."

**Statement.** The substrate dynamics are generated by a single binary self-inverse operation. Concretely: there exists a finite alphabet E of toggle symbols, and to each e in E is associated a toggle operator T_e satisfying

    T_e o T_e = identity   for every e in E,

where o denotes operator composition. Equivalently, T_e is involutive: T_e = T_e^{-1}. The substrate's raw event stream is a finite sequence of toggle applications (e_1, e_2, ..., e_L) interpreted as the composition T_{e_L} o ... o T_{e_2} o T_{e_1}.

**Algebraic content.** Under monoid congruence by the relations e * e ~ epsilon for each e in E, raw streams quotient to the **free involutive monoid** F_inv(E) = E* / (e * e ~ epsilon for each e in E), which is isomorphic as a group to the free product *_{e in E} (Z/2) of |E| copies of the cyclic group of order 2 (Serre 1980, Trees, Section I.1 Proposition 4).

**Cross-references.**

- `../theorems/theorem_toggle_from_self_containment.md` derives the involutive structure and F_inv(E) from (A) + (B) + (I) + Shannon-Jaynes-Serre mathematics. A1 is the conclusion of this theorem, no longer a structural axiom.
- `predictions/p_toggle.py` derives p = 2 (binarity of the toggle) from MDL on involution arity. Under the new slate, this can be read as a downstream consistency check: Shannon's 1-bit minimum (Step 4 of the toggle theorem) gives binarity directly.
- `../../predictions/walker_dynamics_derivation.md` Step 1 builds F_inv(E); under the new slate this construction is downstream of the toggle theorem rather than of A1 directly.

## 3. A2 -- MDL canonicalization (selective retention)

**Status:** refined 2026-04-20 from strict-minimum to selective-retention form per an internal A2 selective-retention downstream audit §6. The refinement is a strengthening of rigor, not a change of physical content: it aligns A2 with its published foundation (Shannon's Rate-Distortion Theorem, A-IT5 in `information_theoretic_stability_axioms.md`) and correctly admits structural content (e.g., g_girth's chirality residual) that the framework was already committed to under strict-minimum tie-breaking.

**Statement.** The observer retains EVERY representation that saves more bits than it costs to encode, relative to the uncompressed observation. Concretely: a representation M is retained if and only if

    L_total(M) < L_raw,

where L_total(M) = L_model(M) + L_data_given_model(M) is the Rissanen 1978/1983 total description length, and L_raw is the description length of the raw uncompressed observation data. When multiple representations satisfy this condition, all are physically realized simultaneously.

**MDL as waterline, not optimum.** MDL does not select the globally optimal compression. It acts as a threshold — a waterline. Every representation above the waterline (positive compression savings: L_total < L_raw) is retained; every representation below it (no savings: L_total ≥ L_raw) is discarded. Representations that just barely save bits are retained alongside highly efficient compressions.

This is analogous to the ΔΦ > 0 waterline in integrated information theory: mechanisms above the waterline contribute; those below do not. Here, compressions above the waterline (any positive Δ = L_raw − L_total > 0) are retained regardless of whether better compressions exist.

**Why not strict minimization.** Strict MDL (keep only the global optimum) is the special case where a UNIQUE representation achieves the maximum savings. For the framework's structural predictions (d=3, k*=3, g=10, srs geometry), the compression landscape has a unique peak — strict minimum and waterline agree. For physical couplings and multi-realizable representations, there are many above-waterline representations simultaneously, and the waterline reading is essential:

- The chirality of srs (mirror-image degeneracy) is above the waterline in both hands simultaneously.
- All three generations under C₃ triality are above the waterline simultaneously.
- All winding-number classes of a girth-cycle NB walk are above the waterline simultaneously (see §5b).

**Shannon rate-distortion interpretation.** The waterline condition L_total < L_raw is equivalent to achieving a strictly positive compression rate I(M; M̂) > 0 in Shannon's rate-distortion framework (Shannon 1959, A-IT5). Rissanen 1978/1983 and Grunwald 2007 §§5.1-5.3 establish MDL as an operational implementation of this criterion. Shannon's theorem is a bound on achievable rates — it does not single out a unique encoder, and the waterline retains all encoders achieving any positive rate.

**Operational statement.** MDL is a licensing criterion (which representations save bits) rather than a selection criterion (which single representation saves the most). Any configuration clearing the waterline is physically realized in superposition or coexistence. The threshold is the uncompressed observation cost, not the global optimum.

**Algebraic content.** Among elements of F_inv(E) that represent the same equivalence class under the involutive-monoid congruence, MDL retains every minimum-symbol-count representative. Per Serre 1980 Proposition I.1.4 the reduced word is unique in F_inv(E), so selective retention here coincides with strict minimization. The non-trivial selective-retention admissions occur downstream — at the lattice selection (srs chirality), the C_3 irrep labeling, and the Cl(2) pseudoscalar orientation.

**Regime where waterline gives a unique representation.** When the compression landscape has a single dominant peak (all other representations save far fewer bits), the waterline effectively selects a unique model. This is the case for structural predictions: d=3, k*=3, g=10, srs geometry. The optimal compression is unique AND well above the waterline; all alternatives are well below. The waterline gives the same answer as strict minimization here.

**Regime where the waterline admits multiple representations simultaneously.** When multiple representations have positive savings and similar savings magnitudes, all are retained:
- Chirality of srs: both handed srs copies save the same bits → both retained.
- C₃ triality: all three generation labels are above the waterline → all retained simultaneously.
- Girth-cycle NB walk windings: the n-th winding of a girth-cycle walk (cost O(log n) bits, savings O(8n) bits) is above the waterline for ALL n ≥ 1 → all retained. This is the coupling-sum mechanism for V_cb and other topological-route CKM elements (see §5b).

**Key consequence for coupling sums.** The coupling V_cb = Σ_{n=1}^∞ (2/3)^{8n} = (2/3)^8/(1−(2/3)^8) is the sum over ALL above-waterline girth-cycle windings. The single-term (2/3)^8 (strict minimum) is wrong under the waterline reading — the second and all higher windings also clear the waterline, so all must be included. This resolves the −0.99σ discrepancy: the waterline coupling gives +0.07σ from PDG. See §5b and `proofs/flavor/a5b_coupling_prescription.py`.

**Cross-references.**

- `../../predictions/walker_dynamics_derivation.md` Step 2 (MDL canonicalization to reduced words; unique-optimum regime).
- `predictions/d_spatial.py` (MDL + Cencov 1982 chain to d = 3 Fisher rank; unique optimum).
- `predictions/k_star.py` (MDL + reticular chemistry to k* = 3; unique optimum).
- `predictions/g_girth.py` (Sunada 2012 uniqueness up to chirality; selective-retention regime).
- `information_theoretic_stability_axioms.md` §I (A-IT5 Rate-Distortion Theorem; published foundation).
- an internal A2 selective-retention downstream audit (the downstream audit establishing refinement safety).

## 4. A3 -- MDL canonicalization is a partial trace over an abstract purifying auxiliary

**Statement (canonical, post-2026-04-17 revision).** The observer's model class is the set of states of the visible sector (Layer 2 of `framework_architecture.md`) obtained by MDL canonicalization of some underlying mixed-state content on the substrate. A3 asserts: the MDL canonicalization map

    pi_MDL : states_visible(mixed) <- states(abstract purifying system)

is equivalent to a **partial trace** over an **abstract purifying auxiliary Hilbert space** H_aux of a **pure state** on the tensor-product Hilbert space H_visible tensor H_aux. Formally:

> For every observer-accessible mixed state rho on Layer 2 (H_visible), there exists an abstract auxiliary Hilbert space H_aux and a pure state |psi> on H_visible tensor H_aux such that
>
>     rho = Tr_{H_aux} ( |psi><psi| ).
>
> Moreover, this purification is essentially unique up to reversible transformations on the H_aux factor.

**Framework-native framing.** The tensor product H_visible tensor H_aux is an ABSTRACT tensor product in the sense of Chiribella-D'Ariano-Perinotti 2011: the composite "system AB" of their purification axiom is defined operationally by the tensor-product composition rule of their operational-probabilistic-theory framework, and the purifying system B is an abstract auxiliary that need not be identified with any specific physical subsystem. CDP 2011 does NOT require B to be a concrete subspace of a pre-existing Hilbert space; it only requires that the composite state space AB be the operational tensor product of A's and B's state spaces.

A3 codifies in framework-native terms the CDP 2011 operational statement: "every mixed state has a pure-state purification on a larger system (via abstract tensor-product composition), essentially unique up to reversible transformations on the purifying system."

**Algebraic content.** A3 supplies four coupled commitments that the framework was already implicitly making:

1. **Purity at the purification level.** For every visible-sector mixed state rho, there is a definite (pure) state |psi> on H_visible tensor H_aux whose partial trace over H_aux is rho. There is no "fundamental randomness" at the purification level; randomness at the visible level is an emergent feature of the trace-out.
2. **MDL canonicalization is partial trace.** The information-theoretic operation of compressing substrate content to the visible sector is precisely the operator-algebraic operation of tracing out an abstract auxiliary tensor factor.
3. **Auxiliary is the trace-out target.** The abstract H_aux carries the definite-but-inaccessible content: it has a state, but the observer cannot access it. Its OPERATIONAL interpretation -- not its mathematical identity -- is the framework's dark sector (Layer 6).
4. **Essential uniqueness of purification.** Two pure states on H_visible tensor H_aux that yield the same visible mixed state under partial trace are related by a reversible (unitary) transformation on H_aux alone. This is the operational content of the CDP 2011 purification axiom.

### 4.1 Why the auxiliary is abstract (not a subspace of H_multiway)

The prior statement of A3 (pre-2026-04-17) asserted the purification lives on "Layer 1 tensor Layer 6" with Layer 1 identified with the length-graded H_multiway of `docs/theorem_H_multiway_construction.md` and Layer 6 identified with H_dark as a SUBSPACE of H_multiway. That identification creates a structural tension documented in an internal working note Section 4.1: at each length L,

    H_multiway^(L) = H_visible^(L) (+) H_dark^(L)

is a DIRECT-SUM orthogonal-complement split, not a tensor product. A tensor product would require dim H_multiway^(L) = dim H_visible^(L) * dim H_dark^(L), which fails already at L = 2 (LHS 36, RHS 180). The CDP 2011 purification axiom requires a tensor product, not a direct sum; identifying Layer 6 with H_dark as a subspace of H_multiway is therefore inconsistent with the literal tensor-product structure CDP requires.

The revision resolves the tension by treating H_aux as an ABSTRACT purifying auxiliary, not a concrete subspace of H_multiway. CDP 2011 explicitly supports this reading: in their operational-probabilistic-theory framework (their Section II), the composite "system AB" is a primitive notion defined by the tensor-product composition rule for state spaces, and the purifying system B is characterized only by the operational role it plays (producing a pure state on AB whose A-marginal is the given mixed rho_A). B is not required to be realized as a subspace of any pre-existing Hilbert space; it is identified only up to the equivalence imposed by CDP's purification uniqueness clause (essential uniqueness up to reversible transformations on B).

Under the revision, H_multiway of `docs/theorem_H_multiway_construction.md` plays a DIFFERENT role: it is not the ambient H_visible tensor H_aux space, but rather supplies the length-graded mathematical scaffolding whose REDUCED-WORD subspace is H_visible. The direct-sum lemma dim H_dark^(L) = n[n^(L-1) - (n-1)^(L-1)] of `predictions/H_multiway_dim_count.py` remains valid and useful as a statement about MDL canonicalization's kernel on length-L strings, but it does NOT describe an A3 tensor factor. The A3 purifying auxiliary H_aux is a separate abstract Hilbert space whose dimension and basis are determined by the essential-uniqueness clause of A3 applied to the specific visible mixed state being purified.

### 4.2 Relationship to Layer 1 and Layer 6 in framework_architecture.md

The multi-layer view of `framework_architecture.md` places Layer 1 as the "multiway substrate" and Layer 6 as the "dark sector." Under the revised A3:

- **Layer 1 (multiway substrate).** Unchanged as a framework layer. H_multiway = +_L C^(|E|^L) of `docs/theorem_H_multiway_construction.md` remains the length-graded Hilbert-space lift of F_inv(E). Its reduced-word subspace H_visible = +_L H_visible^(L) is identified with the visible sector's Hilbert space (the "A" of A3's H_visible tensor H_aux). Its cancellable-string complement H_dark sits inside H_multiway as a direct-sum summand and carries the dim-count content of `predictions/H_multiway_dim_count.py`.
- **Layer 6 (dark sector).** Unchanged as a framework layer in its OPERATIONAL / physical interpretation (the uncompressed multiway residue, dark matter candidate per an external research note on dark-matter compression and `predictions/Omega_DM_over_Omega_m.py`). But Layer 6 is NOT mathematically identified with H_dark as a subspace of H_multiway; it is identified with the abstract purifying auxiliary H_aux of A3, whose operational role coincides with the physical dark sector but whose mathematical structure is determined by the CDP 2011 essential-uniqueness clause, not by the combinatorial structure of F_inv(E).

The consequence is that the dark sector has TWO roles that need not coincide mathematically:

- **Dark-as-H_dark (combinatorial).** The subspace of H_multiway spanned by non-reduced strings. Carries the dim-count content of `predictions/H_multiway_dim_count.py`. Direct-sum complement of H_visible inside H_multiway.
- **Dark-as-H_aux (operator-algebraic purification).** The abstract auxiliary Hilbert space whose tensor product with H_visible supports a purification of the visible mixed state. Tensor-product complement, not subspace complement.

These two "darks" are RELATED but not IDENTIFIED: both code the MDL kernel ("what the observer cannot see"), but the first is a combinatorial / kinematical object and the second is an operator-algebraic / purification-theoretic object. Whether there is a canonical isomorphism between them -- for example, whether H_aux can be built explicitly from H_dark by a GNS-style construction or a Feshbach reduction -- is an open structural question, tracked in Section 11 and in `docs/theorem_layer_1_hilbert_space_identification.md` Open Question 1.

The observational interpretation of dark matter as "uncompressed multiway residue" (the Layer 6 physical content developed in an external research note on dark-matter compression) is compatible with both readings; it does not force mathematical identification with either H_dark or H_aux specifically. A3 only requires the abstract H_aux to exist; the physical identification with multiway residue is a SEPARATE operational claim consistent with but not forced by A3.

## 5. A4 -- Node grading / fermionic statistics (CAR)

**Statement.** The edge modes at each k*-valent node of the srs lattice satisfy **canonical anticommutation relations (CAR)**; the node state space is graded by fermionic parity. Concretely: to each directed edge $e$ incident to a node $v$ is associated an operator $\gamma_e$ acting on the node's local Fock space, satisfying

$$\{\gamma_e,\, \gamma_{e'}\} \;=\; \gamma_e \gamma_{e'} + \gamma_{e'} \gamma_e \;=\; 2\,\delta_{ee'} \cdot I,$$

where $\{\cdot,\cdot\}$ is the anticommutator, and the Fock space is $\mathbb{Z}/2$-graded by the total mode-occupation number mod 2 (fermionic parity).

**What A4 does.**

1. **Clifford structure.** With $k^* = 3$ edges per node, the three generators $\gamma_{e_1}, \gamma_{e_2}, \gamma_{e_3}$ satisfying CAR generate the Clifford algebra $\text{Cl}(3) \cong \mathbb{H} \oplus \mathbb{H}$ locally. On the K_4 quotient with 6 directed edge modes, A4 promotes the local mode algebra to $\text{Cl}(6)$. Without A4 these modes generate the Weyl (bosonic oscillator) algebra $\mathfrak{W}(3)$, which has no $\mathbb{Z}/2$-grading and does not furnish spinor representations.

2. **Cl(8) factorization.** Together with the Cl(2) factor from the complex directed-edge structure of K_4 (one complex Z_2 channel per directed edge, per the F2-class identification documented in `predictions/lambda_higgs.py` §6), A4 grounds $\text{Cl}(8) = \text{Cl}(6) \otimes \text{Cl}(2)$. The spinor decomposition $S = \text{Cl}(8)^{\text{even}}$ then carries the 64-dimensional even subalgebra from which the Standard Model fermion representations are assembled in the B3 workstream (`../../predictions/theorem_B3_spinor_fermion_derivation.md`).

3. **Unlocks B3 and downstream.** B3 (spinor–fermion identification), B4 (color structure), F2-class (factor 2 = Cl(2) = SU(2)_L doublet dimension), and the full CKM-sector derivation chain all require $\text{Cl}(6)$ to be a Clifford algebra (not a Weyl algebra). A4 is the structural commitment that makes this valid.

**Why A4 is NOT derivable from A1+A2+A3.**

- **A1 (toggle)** generates involutions $T_e^2 = \text{id}$, which is the Clifford relation $\gamma_e^2 = I$ in disguise. But A1 is symmetric under exchange of any two toggles — there is no sign: $T_e T_{e'} = T_{e'} T_e$ (both are just reduced-word swaps). No anticommutator $\gamma_e \gamma_{e'} + \gamma_{e'} \gamma_e = 0$ (for $e \ne e'$) emerges.
- **A2 (MDL)** selects the unique MDL-minimal representation among equivalent ones. Swapping the order of two edge modes at a node produces a description of the same reduced word with the same length — MDL-equidistant. No sign differential arises from description-length minimization.
- **A3 (partial trace / purification)** supplies Hilbert-space structure and the Born rule. It provides $\mathbb{Z}/2$-grading only if the state space is already graded; A3 does not ground the grading itself.

**Best available justification for A4.**

The Spin-Statistics theorem (Pauli 1940; Streater and Wightman 1964, *PCT, Spin and Statistics, and All That*, Theorem 4-5; Fierz 1939) states: in any Lorentz-covariant local quantum field theory satisfying the standard Wightman axioms, fields of half-integer spin satisfy CAR and fields of integer spin satisfy CCR. The derivation chain is:

1. A4 (assumed) → Cl(6) is a Clifford algebra → B3 derives a Lorentz-covariant spin-½ field from the Cl(6) spinor.
2. Streater-Wightman Theorem 4-5 applied to that spin-½ field: Lorentz covariance + locality + spectral condition force CAR.
3. Step 2 validates A4 as a derived consequence of Lorentz covariance applied to the B3 output.

This is not a circular derivation of A4 from itself. It is a **consistency check**: A4 is presupposed to derive B3, and B3's Lorentz covariance then makes A4 a theorem (not a postulate) of the completed theory. The same structure holds for A3: A3 is presupposed to derive G.1+G.5, and the full CDP 2011 chain then makes A3 a theorem of quantum mechanics (its purification axiom). A4's status is formally parallel.

**Alternative justification (MDL-Fock grading, exploratory).**

MDL on Fock-space descriptions: the Fock space over $k^*$ edge modes has two representations — bosonic (symmetric tensor powers, exponentially growing Hilbert space dimension) and fermionic ($\mathbb{Z}/2$-graded, finite-dimensional at each grade: $2^{k^*} = 8$ states for $k^*=3$). The fermionic Fock space has strictly smaller description length at every grade level. This would make A4 an instance of A2 (MDL prefers the shorter description). However, this argument is not written up at journal grade: it requires a formal comparison of the Kolmogorov complexity of bosonic vs fermionic Fock spaces applied to the toggle-generated word algebra, which has not been carried out. Flagged as a potential A2-grounded closure route for A4 (would reduce the axiom count back to three if it closes).

**Status.** A4 is adopted by user decision on 2026-04-19, with Spin-Statistics + Streater-Wightman 1964 as the primary justification and MDL-Fock as a speculative secondary route. It is explicitly listed as a fourth axiom, not a derived theorem. The MDL-Fock route, if carried out, would reduce A4 to a consequence of A2 and remove it from the axiom list.

**Cross-references.**

- `../../predictions/theorem_B3_spinor_fermion_derivation.md` — B3 workstream (spinor–fermion identification); presupposes A4 via Cl(6).
- `predictions/lambda_higgs.py` §6 — F2-class factor-2 identification; depends on Cl(2) from A4.
- an internal Sprint 9 kickoff doc §2 — Sprint 9 Cl(6)/Path B plan; the entire B-workstream presupposes A4.

**References.**

- **Pauli, W.** (1940). The connection between spin and statistics. *Phys. Rev.* **58**, 716–722. Original spin-statistics theorem.
- **Fierz, M.** (1939). Über die relativistische Theorie kräftefreier Teilchen mit beliebigem Spin. *Helv. Phys. Acta* **12**, 3–37.
- **Streater, R.F., Wightman, A.S.** (1964). *PCT, Spin and Statistics, and All That.* Princeton University Press. Theorem 4-5 (the standard proof of the spin-statistics connection from Lorentz covariance + locality + spectrum condition). The load-bearing theorem for the consistency justification of A4.
- **Jordan, P., Wigner, E.** (1928). Über das Paulische Äquivalenzverbot. *Zeitschrift für Physik* **47**, 631–651. Jordan-Wigner transformation: the explicit realization of CAR from the toggle / involution structure via a graded string. Directly relevant to the A1→A4 upgrade: the Jordan-Wigner string converts A1's involutions into A4's anticommutators on a 1D chain; the srs lattice generalization requires a node-by-node Jordan-Wigner ordering, which is the B1 ordering workstream (`../../predictions/theorem_B1_ordering_derivation.md`).

---

## 5b. A5 -- Physical identification (reading rule)

**Statement.** The framework is a theory of Standard Model particle physics. Specifically:

- **A5(a) — Mass clause.** The Ramanujan eigenvalues of the srs Bloch-Hashimoto operator are identified with the physical mass spectrum of the Standard Model visible sector.
- **A5(b) — Coupling clause.** The total branch-measure probability of all above-waterline NB walk representations of the process is identified with the physical coupling strength. Under A2's waterline, the retained walk class for a process with n_fixed pinned endpoint causal states consists of all girth-cycle winding classes {n × (g−n_fixed) steps : n = 1, 2, 3, …}, each of which saves Ω(n) bits over uncompressed description. The coupling is the sum over all retained windings:

      coupling = Σ_{n=1}^∞ ((k*−1)/k*)^{n(g−n_fixed)} = u^L / (1 − u^L)

  where u = (k*−1)/k* and L = g − n_fixed. In particular for the b→c vertex (n_fixed=2, L=8):

      V_cb = (2/3)^8 / (1 − (2/3)^8) = 256/6305 ≈ 40.60 × 10⁻³  (+0.07σ from PDG).

  α₁_bare = (2/3)^8 is the FIRST WINDING probability; the physical coupling α₁ = (2/3)^8/(1−(2/3)^8) is the full above-waterline sum.

  **Note on MDL probability distribution forms (session 24).** The geometric series u^L/(1−u^L) is the specific form for EXPONENTIAL (branch-measure) weighting, appropriate when pathways are girth-cycle windings and each step contributes a multiplicative factor u = (k*−1)/k* from the branch measure. A5(b) also covers a second form: COUNTING (uniform) weighting, appropriate when all pathway positions are structurally equivalent. When the Moore bound identity g = k*²+1 forces exactly one coupling event per girth-cycle slot (floor(g/k*²) = 1), no slot is preferred over any other, so the MDL distribution over coupling events per girth step per unit cell is uniform. The MDL probability takes the form:

        coupling = k*² / (g × N_ATOMS)   [counting fraction — V_us = 9/40, THEOREM-GRADE]

  Uniformity is justified by Type 1 (A2: girth cycles are retained as indivisible units by A2's waterline) + Type 2 (Moore bound symmetry: floor(g/k*²) = 1 makes all g steps structurally equivalent, so no step can be MDL-preferred). Both the geometric series and the counting fraction are valid MDL probability forms under A5(b)'s principle — "MDL probability = coupling strength" — differing only in whether pathway weights are exponential (branch-measure, V_cb mechanism) or uniform (Moore-equivalent slots, V_us mechanism). The concern "L ≈ 4.18 is not an integer" is a red herring from forcing V_us into the wrong formula structure: V_us is a counting probability, not an exponential one, and A5(b) covers both.

  **Note on level-specific prescription (session 25).** A5(b) identifies MDL probabilities with physical couplings, but the SPECIFIC FORMULA depends on what level of the framework hierarchy the coupling lives at:

  - **Case (A) — direct-moment form (Level 2, srs-intrinsic).** When α₁ enters a coupling's formula as a closed-form numerical coefficient representing ONE graph-theoretic event probability (a single NB walk survival, a specific cycle amplitude, a perturbation magnitude), A5(b) identifies the coupling with that direct moment. No winding sum.
    - Examples: α₁_bare = (2/3)^8, α₁_full = (5/3)(2/3)^8, λ_Higgs = 2α₁_full, y_τ = α₁_full/k*², θ_23_PMNS.

  - **Case (B) — walk-representation sum (Level 3, Hashimoto walk-sums).** When the coupling is identified with a sum over Hashimoto walk representations between pinned causal states (different windings contribute as distinct walks to the SAME coupling), A5(b) identifies the coupling with the full above-waterline geometric series u^L/(1−u^L).
    - Examples: V_cb = α₁/(1−α₁), v_Higgs dark correction (5/12)α₁/(1−α₁).

  - **Case (B') — counting form (Level 3, Moore-equivalent slots).** Special sub-case of (B) where slots are structurally equivalent under Moore bound, giving uniform MDL probability k*²/(g·N_ATOMS).
    - Example: V_us = 9/40.

  **Criterion for Case selection (derivation-structure, not observation).** If the coupling's derivation expresses it as a closed-form function of α₁ and other graph invariants, Case (A). If the derivation is a sum Σ_n α₁^{f(n)} with n indexing distinct walk representations, Case (B). If Moore-equivalent uniform, Case (B'). The classification is determined by the STRUCTURAL ROLE of α₁ in the formula — the question of which level it appears at — independent of observational match. See `../theorems/theorem_A5b_level_prescription.md` for the full theorem and classification audit.

Predictions derived from these identifications are compared to experimental values.

That is all A5 says. It is the declaration that this mathematical structure is *about* particle physics, not merely formally analogous to it. A5 is one axiom with two clauses — the mass clause and the coupling clause are the same kind of identification (math object → physical observable), differing only in which math object is on the left-hand side.

**What A5 codifies.**

A5 makes explicit an identification that was implicit in the framework from its first prediction. Every time a derivation chain terminates at a V_Ram eigenvalue and that value is compared to an experimental particle mass, the comparison presupposes A5. Without A5, the framework produces a mathematical structure — the srs Bloch-Hashimoto spectrum — with no claim about physical content. With A5, every such derivation chain terminates at a Standard Model prediction.

A5 is the framework's empirical anchor. A1 through A4 are statements about mathematical structure (toggle involutions, MDL selection, purification, fermionic statistics). A5 is the statement "this structure describes reality." That claim cannot be proved from mathematical first principles; it is validated by prediction accuracy.

**What A5 enables.**

Under A5, the adoption labels ADOPTED-P1, ADOPTED-CS, ADOPTED-Y, and **I-Feshbach** all collapse to a single item: A5. They were never independent structural postulates — they were downstream restatements of the same identification. Specifically:

- **ADOPTED-P1** (mass content supported on V_Ram, not V_tree): direct consequence of A5(a). If masses are V_Ram eigenvalues, they are supported on V_Ram by definition.
- **ADOPTED-CS** (mass operator is C₃-scalar): consequence of A5(a) + the scalar pairing theorem, which shows that the unique A1-A4-consistent form of the A5(a) identification is a C₃-scalar operator supported on V_Ram.
- **ADOPTED-Y** (substrate amplitudes = Yukawa couplings): consequence of A5(a) applied to the Yukawa sector. The framework's Bloch-fiber amplitudes at each vertex are identified with Yukawa couplings under A5; there is no additional content.
- **I-Feshbach** (MDL-minimal constrained NB walk probability = dark-sector coupling): consequence of A5(b). The Feshbach Exponent Principle gives ((k−1)/k)^(g−n_fixed) as the EXACT MDL-minimal probability (not a truncated series). This is identified with the physical coupling α₁_bare under A5(b). The exhaustive enumeration of bridges (six P/Q decompositions, Wigner-Weisskopf, holonomy) all hit the same wall — the bridge is irreducible to A1+A2+A3 mathematics and is the framework's empirical content. See `../theorems/theorem_ifeshbach_percycle_resolution.md` and an internal working note for the closure attempts that motivated the A5(b) extension.

Downstream predictions that previously cited multiple adoption labels (Q_Koide, epsilon_Koide, delta_Koide, CKM elements, PMNS angles, srs_cubic_moment, θ_23, m_τ, m_ν, M_R, V_us, V_cb, V_ub, η_B, anything depending on α₁) should now cite A5 as the single structural input, with the scalar pairing theorem providing the uniqueness justification for the mass clause and the Exponent Principle (`predictions/feshbach_exponent_principle.py`) providing the rigorous combinatorial half for the coupling clause.

**What A5 does NOT do.**

A5 does NOT claim:

- That the srs lattice was chosen to match SM data. The lattice is derived independently via A1+A2 (MDL selects it uniquely). A5 identifies the mathematical output with physical observables after the fact.
- That all SM parameters follow from A5+A1-A4 alone. Several adoptions remain genuinely independent of A5: the hypercharge assignment Y = +1/2 (requires gauge coupling measurement), the SU(2)_L chirality assignment (left vs. right, requires Lorentzian structure or observation), the dark-map classification (tan²(arg h) = 5/3 factor in λ_Higgs), and the Pati-Salam neutrino bare scale.
- That A5 is derivable from A1-A4. It is not. The scalar pairing theorem shows A5 is *consistent with* and *uniquely constrained by* A1-A4 (T-symmetry forces the condensate to be C₃-scalar, which is the only form the identification can take), but consistency is not derivation. The empirical fit is the ultimate justification.

**Why A5 is not derivable from A1-A4.**

The gap is the same gap that exists between mathematics and physics in every foundational theory. A1 (toggle) derives the free involutive monoid. A2 (MDL) selects srs. A3 (purification) gives complex Hilbert-space structure. A4 (CAR) gives fermionic statistics and Clifford algebra. Together they produce a specific mathematical object: the srs Bloch-Hashimoto spectrum with Ramanujan eigenvalues h = (√3+i√5)/2 plus a specific MDL probability distribution over multiway processes. What they do not produce, and cannot produce from first principles, is the sentence "these eigenvalues are particle masses and these probabilities are coupling strengths." That sentence is A5.

The scalar pairing theorem narrows the gap for the mass clause: given A1-A4, if A5(a) holds, then it must hold in the specific form "M = gap operator of the C₃-scalar I-Feshbach condensate." For the coupling clause A5(b), the analogous narrowing is the **Feshbach Exponent Principle** (`predictions/feshbach_exponent_principle.py`, STRICT-SOLID under A1+A2+Jaynes 1957+Serre 1980+Terras 2011): given A1-A4, the only natural candidate for the dark-sector coupling at n_fixed = 2 is ((k−1)/k)^(g−2). This is uniqueness up to A1-A4; it does not eliminate the need for A5(b) itself.

NOTE (2026-04-21, revised): The correct prescription is the WATERLINE SUM over all girth-cycle winding classes, giving the geometric series u^L/(1−u^L). Earlier session-13 analysis incorrectly blocked this by applying strict-minimum MDL (retain only the globally optimal compression). Under the correct waterline reading of A2, all n-th windings of the girth cycle are above the waterline and all contribute. The Green's function (sum over ALL NB walks, including random non-compressible ones) overestimates by orders of magnitude — that computation confirmed the geometric series must restrict to structurally simple (compressible) walks only. The Feshbach Exponent Principle ((k−1)/k)^{g−n_fixed} gives the FIRST WINDING probability; the physical coupling is the geometric series sum over all windings.

By analogy: in standard physics, the value of Newton's constant G is not derived from anything more fundamental — it is measured. The framework's situation with α₁_bare is similar: the *form* ((k−1)/k)^(g−2) is forced by A1+A2+Jaynes (the Exponent Principle), but the *identification* of that probability with a physical coupling strength is the framework's empirical claim. Validation is via prediction accuracy (θ_23 within 0.4σ, etc.).

**Status.** A5 is adopted by user decision on 2026-04-19. It is explicitly listed as the fifth axiom, not a derived theorem. Unlike A1-A4, A5 has no plausible closure route: no mathematical argument will ever derive "this is about particle physics" without at some point importing particle physics data. A5 is irreducible.

**Cross-references.**

- `docs/theorem_P1_ramanujan_support.md` — MDL discriminability argument showing V_tree cannot be the A2-selected sector; ADOPTED-P1 collapses to A5 under this result.
- `predictions/Q_Koide.py`, `predictions/epsilon_Koide.py`, `predictions/delta_Koide.py` — flavor predictions whose ADOPTED-P1 and ADOPTED-Y labels should be updated to cite A5 directly.
- `../audits/registers/adoption_register.md` — full register of remaining independent adoptions (Y, chirality, dark-map, Pati-Salam) not subsumed by A5.

---

**Joint-Feshbach reformulation of A5(b) — note added 2026-05-09 EOD+1.**

The feedback-loop synthesis (an internal working note and the P2-P5 follow-on docs in the same directory) provides an alternative *presentational* form of A5(b)'s coupling clause:

> A5(b)' (joint-Feshbach reformulation): The observable coupling strength is the closure rate ν of the joint observer-substrate feedback loop, i.e., ν = |Im(Σ(h))|/α_1, where Σ(h) is the substrate Feshbach self-energy at the Ramanujan-circle saddle h evaluated under the canonical formula Σ(h) = α_1/h proven in `theorem_analytical_feshbach_ramanujan_boundary.md`, combined with the Shannon-Jaynes principle of indifference for combinatorial factors over MDL-equidistinct alternatives.

This reformulation expresses A5(b) in language closer to standard QFT optical-theorem apparatus (Cutkosky 1960, Peskin-Schroeder §7) and connects substrate Feshbach machinery to physical couplings via the same identity used in standard scattering-amplitude calculations. Under this reading, A5(b)'s identification "MDL probability = coupling strength" is a substrate-side instance of the optical-theorem identity |Im(Σ)| = observable rate.

**The reformulation is a presentational alternative, NOT a structural reduction.** The empirical content of A5(b) — that substrate Feshbach quantities ARE the right operators to apply the optical theorem to — remains a load-bearing identification. The synthesis does not eliminate this identification; it reformulates it in operator-theoretic language. A5(b)'s irreducibility per §5b paragraph 279 ("no mathematical argument will ever derive 'this is about particle physics' without at some point importing particle physics data") still holds in spirit: the synthesis just makes the import look closer to standard physics.

The synthesis docs further establish:
- **Vertex form derivation** (`P3_vertex_form_derivation_2026-05-09.md`): SM Yukawa, gauge, and Higgs quartic vertex forms are forced by Cl(6)⊗Cl(0,2) algebra at the trivalent srs node + chirality + symmetry constraints — no Peskin-Schroeder ansatz needed for vertex *form* (only for low-energy EFT translation).
- **Backward compat on y_τ, λ_H, V_cb** at structural-factor level (P2) and matrix-element consistency check on y_τ (P4).
- **SU(2)_L doublet partner prediction**: h⁺·ν̄_L·τ_R coupling forced to magnitude y_τ by SU(2)_L gauge invariance, dropped out of the joint-Feshbach formalism (P4).
- **Need-D-3 reframe** (`P5_need_D3_joint_feshbach_reattack_2026-05-09.md`): the substrate has multiple per-pattern-pair Σ_AB mechanisms for different CKM elements, not a single unified Y matrix; the framework's existing heterogeneous CKM derivation (Class A + C + E + unitarity) is structurally correct under this reading.

The synthesis docs explicitly acknowledge their honest scope:
- The matrix-element computations in P4 use the existing structural decomposition, not independent Bloch-fiber computation (consistency check, not from-scratch derivation).
- P3 vertex enumeration is structural rather than rigorous (no full Wedderburn decomposition).
- The Shannon-Jaynes principle of indifference is invoked as a separate well-founded principle, not eliminated.

The framework slate as listed in §10 is unchanged in form but the synthesis suggests A5(b) can be presented as a corollary of {A2-T waterline + Feshbach formula + Shannon-Jaynes principle of indifference + optical-theorem identity}, with the empirical content concentrated in "substrate quantities are the right operators to feed into standard scattering apparatus" — which remains an irreducible identification in the spirit of A5.

Cross-references for the synthesis:
- `proofs/foundations/joint_feshbach_y_tau_verification.py` (probe, 6/6 PASS)

---

## 6. Prior art for A3

A3 is the Chiribella-D'Ariano-Perinotti 2011 purification axiom in framework-native form. The CDP 2011 derivation of finite-dimensional complex Hilbert-space quantum mechanics from purification + four other operationally-motivated axioms (causality, perfect distinguishability, ideal compression, local distinguishability) is the canonical reference.

- **Chiribella, D'Ariano, Perinotti 2011** "Informational derivation of quantum theory" Phys. Rev. A 84, 012311. Five axioms: causality, perfect distinguishability, ideal compressions, local distinguishability, PURIFICATION. Their Theorem 25 (Section VIII) states that purification (axiom 5) added to causality + perfect distinguishability + ideal compressions + local distinguishability (axioms 1-4) uniquely forces the state space to be the density operators on a finite-dimensional **complex** Hilbert space, with reversible transformations being unitary conjugation. Their Section VIII explicitly carries through the C-vs-R-vs-H field selection: only F = C is consistent with all five axioms.
- **Dakic, Brukner 2011** "Quantum theory and beyond: is entanglement special?" arXiv:0911.0695. Related purification-based derivation; uses purification (their axiom 4) + tomographic locality + continuity to arrive at finite-dim complex QM.
- **Hardy 2001** "Quantum theory from five reasonable axioms" arXiv:quant-ph/0101012. Earlier axiomatic derivation; uses simplicity (a Cantor-like axiom that is operationally close to purification) + composition + continuous reversibility to derive QM.
- **Masanes, Mueller 2011** "A derivation of quantum theory from physical requirements" New J. Phys. 13, 063001. Axiomatic derivation using a no-restriction hypothesis + composition + continuous reversibility.
- **Fuchs 2010** "QBism, the perimeter of quantum Bayesianism" arXiv:1003.5209. Alternative subjectivist framing of purification; not used in this framework.

The CDP 2011 derivation is the closest match to the framework's intended use because (a) it produces finite-dimensional complex Hilbert spaces (matching the framework's downstream needs), (b) its purification axiom is operationally clean (every mixed state purifies), and (c) the four supporting axioms (causality, perfect distinguishability, ideal compressions, local distinguishability) all have direct framework-native readings under A1 + A2 (see Section 7 below).

## 6. What A3 codifies

A3 codifies the multiway substrate's pure-state structure that the framework was already implicitly assuming. Specifically:

- `framework_architecture.md` Layer 1 ("Multiway substrate") describes Wolfram-Gorard 2020 multiway dynamics as the substrate; multiway dynamics is naturally pure-state at the foundational level (the wave function over the branching state space is well-defined; mixedness arises from compression / coarse-graining).
- An external research note on dark-matter compression treats dark matter as multiway branches that fail MDL compression (i.e., the trace-out target). The "dark is definite-but-inaccessible" reading of A3 is the operational content of that proposal.
- `../../predictions/walker_dynamics_derivation.md` Reading Conventions Section ("Open System Reading") explicitly identifies the canonicalization step (W2) as "a measurement event: the visible-sector amplitude is what survives the canonicalization, and the dark-sector measure is what is shed." This is exactly A3's partial-trace structure.
- `docs/theorem_H_multiway_construction.md` builds the explicit length-graded Hilbert-space lift H_unred^(L) = H_visible^(L) (+) H_dark^(L) of the F_inv(E) substrate, with H_visible the span of MDL-canonical reduced words and H_dark the span of strings containing cancellable adjacent pairs. Under the revised A3 (Section 4.1), the canonicalization map H_unred^(L) -> H_visible^(L) is the KINEMATICAL projection onto the reduced-word subspace; the A3 purification partial trace lives on the SEPARATE tensor-product space H_visible tensor H_aux, with H_aux an abstract purifying auxiliary whose operational content coincides with but is not mathematically identified as H_dark.
- `predictions/H_multiway_dim_count.py` computes the dim counts dim H_visible^(L) = n*(n-1)^(L-1), dim H_dark^(L) = n * [n^(L-1) - (n-1)^(L-1)] for the srs alphabet n = |E| = 6. Under the revised A3, these dim counts remain valid statements about the COMBINATORIAL / KINEMATICAL dark subspace (the MDL kernel inside H_multiway); they do NOT directly describe the dim of the abstract H_aux, which is determined by the CDP essential-uniqueness clause applied to the specific visible mixed state being purified.

In every case the framework was already operating as if A3 were in force. A3 makes the commitment explicit and adds it to the audit trail.

## 7. What A3 enables -- derivation of G.1 and G.5

Per an internal audit of the seven Gleason sub-assumptions, B7.1 (`../../predictions/observer_dim_three_derivation.md`) deploys Gleason 1957 to derive observer Hilbert space dim n = 3, but Gleason's theorem implicitly bundles seven sub-assumptions, of which G.1 (Hilbert-space structure on the model class) was ASSUMED and G.5 (complex field) was a GAP under A1 + A2 alone.

**Under A1 + A2 + A3, G.1 and G.5 are DERIVED via CDP 2011.** The chain:

1. A3 (revised, Section 4) says the observer's mixed states on Layer 2 (H_visible) are partial traces of pure states on the ABSTRACT tensor product H_visible tensor H_aux, with H_aux a purifying auxiliary Hilbert space in the CDP 2011 operational sense. Partial trace is an operator-algebraic operation defined on a tensor product of Hilbert spaces. So the observer's model class is naturally embedded in the operator algebra of a Hilbert space.
2. A1 + A2 supply the four CDP supporting axioms in framework-native form:
   - **Causality** (CDP axiom 1): the W3 directed-edge Markov dynamics on srs is causal -- one direction of propagation, no signaling backward in walk-time. Derivable from A1 (toggle generates the involutive monoid; canonical reduced words have a canonical past-to-future order).
   - **Perfect distinguishability** (CDP axiom 2): the substrate spectra (`../../predictions/B_P_doubly_degenerate_h_derivation.md` (4, 2, 2) C_3 multiplicities at P) supply distinct eigenvalues that are distinguishable by spectral measurement. Derivable from A1 + A2 + the srs structure derived in `predictions/d_spatial.py` and `predictions/k_star.py`.
   - **Ideal compressions** (CDP axiom 3): A2 (MDL) IS the ideal compression principle. Grunwald 2007 Sections 5.1-5.3 are the canonical statement.
   - **Local distinguishability** (CDP axiom 4): the srs lattice's local structure (3-regular vertices, girth 10, primitive cell of 4 vertices and 6 edges) supplies local degrees of freedom whose joint state is determined by local marginals on the constituent vertices. Derivable from `predictions/d_spatial.py` + `predictions/k_star.py` + `predictions/g_girth.py`.
3. Adding A3 (purification, CDP axiom 5) completes the CDP five-axiom chain. By CDP 2011 Theorem 25, the observer's state space is forced to be the density operators on a finite-dimensional complex Hilbert space.

**G.1 (Hilbert-space structure exists) is derived under A1 + A2 + A3.** Specifically: the observer's model class is the set of density operators on a finite-dim Hilbert space. This Hilbert space is the visible-sector tensor factor H_visible of the abstract H_visible tensor H_aux ambient pure-state space of A3.

**G.5 (complex field) is also derived under A1 + A2 + A3.** CDP 2011 Section VIII shows that the field selection is forced to F = C by the combination of local distinguishability + purification. Real Hilbert space fails because the local-distinguishability axiom requires the joint state space of two systems to be the tensor product of the constituent state spaces, and for real Hilbert space this constraint is incompatible with purification (the partial trace of a real-Hilbert-space pure state can fail to have a real-Hilbert-space purification on a real-Hilbert-space environment of the right dimension; CDP 2011 Section VIII Lemma 11). Quaternionic Hilbert space fails because the tensor-product structure is non-associative for quaternions (CDP 2011 Section VIII Theorem 24).

**Conclusion of Section 7.** Under A1 + A2 + A3, the four ASSUMED items (G.1, G.3, G.7) and the GAP item (G.5) of an internal audit of the seven Gleason sub-assumptions move to DERIVED status via the CDP 2011 chain. G.2 (non-contextuality), G.4 (dim >= 3), and G.6 (positivity + trace 1) were already DERIVED under A1 + A2 (B7.1 Step 1, Step 2, and Gleason output respectively); they remain DERIVED.

## 8. What A3 does NOT do

A3 is a structural axiom about the relationship between Layer 1 (substrate) and Layer 2 (visible sector); it does NOT supply additional physical content beyond that. Specifically:

- **A3 does not specify which specific pure state on H_visible tensor H_aux.** The pure state |psi> is essentially unique up to unitary on H_aux (per A3's uniqueness clause), but the equivalence class itself is not selected. The dynamics on the equivalence class are still A1-driven (the toggle operators T_e generate the substrate dynamics).
- **A3 does not supply a concrete identification of H_aux with H_dark or any subspace of H_multiway.** A3 asserts only the existence of an abstract purifying auxiliary; any concrete realization (length-graded H_dark, GNS construction on a dynamical C*-algebra, Szegedy tensor factor, etc.) is a SEPARATE structural commitment beyond A3.
- **The dynamics are still A1-driven.** A3 says nothing about how the pure state evolves; that is determined by A1 (the toggle operators and their composition into walks on srs).
- **MDL is still the observer's rule.** A3 says the canonicalization map IS a partial trace; it does not say the observer chooses partial-trace operators by some new criterion. The observer still selects models by A2 (MDL); the partial-trace structure is the operator-algebraic shape that the MDL-optimal canonicalization takes.
- **A3 does not by itself unblock Q_Koide = 2/3.** Per an internal audit of the seven Gleason sub-assumptions Section 6 hidden-gap discussion: closing G.1 + G.5 does not automatically close Need-A2 (canonical generation-Z_3 on C^3_gen) or Need-RR (canonical reading rule from substrate amplitude to mass parameter). Those remain separately load-bearing structural inputs, requiring their own derivations.
- **A3 does not derive the Pati-Salam labeling.** B3's "adopted-postulate at Layer 3" (the Pati-Salam labeling of Spin(4) x Spin(2) factors in the Cl(6,0) spinor decomposition) remains an OTHER-SMUGGLE under A1 + A2 + A3.
- **A3 does not derive the Feshbach Exponent Principle.** That structural identification (`../../predictions/Feshbach_coupling_strength_derivation.md` Section 3) remains separately load-bearing.

In short: A3 closes the Hilbert-space-structure and complex-field gaps at the foundational level (Layer 4), but the framework's downstream stalls (Need-A2, Need-RR, Pati-Salam labeling, Exponent Principle, Lindblad reading legitimacy beyond the visible/dark partial-trace structure) are independent of A3 and remain open.

## 9. Cited published foundations (A-IT1..7)

**Status:** the "axioms" labeled A-IT1 through A-IT7 in `information_theoretic_stability_axioms.md` are STANDARD PUBLISHED RESULTS from thermodynamics and information theory, not framework-native axioms. They are cited by the framework in the same way Shannon 1948, Sunada 2012, Jaynes 1957, or CDP 2011 are cited: as upstream mathematical / physical results whose content the framework uses but does not re-derive.

**Why they are not axioms of this framework:** every result in A-IT1..7 is a theorem of an established discipline (classical thermodynamics for A-IT1, A-IT2; information theory for A-IT3..A-IT7). The framework does not introduce novel physical content with any of them. They are cited because specific framework derivations (especially A2's refined form and the IT-1..5 theorems that unlock N_hub, M ∝ N^(2/3), and the bulk/boundary dark-correction taxonomy) rest on them.

By contrast, A1 through A5 are framework-native POSTULATES that cannot be sourced from any prior discipline without importing the framework's interpretation: A1 (toggle) chooses the substrate generator; A2 (MDL) chooses the observer's selection rule; A3 (purification) chooses the substrate-to-visible map; A4 (CAR) chooses fermionic statistics for edge modes; A5 (physical identification) declares the substrate is the substrate of particle physics.

**Indexed list of cited A-IT results:**

| Label | Content | Primary citation | Where used in this framework |
|---|---|---|---|
| A-IT1 | First Law (open system) | Clausius / Carnot | IT-1 observer stability (via maintenance budget) |
| A-IT2 | Second Law | Clausius 1865 | IT-1 observer stability; A-IT7 |
| A-IT3 | Landauer's Principle | Landauer 1961; Bennett 1973 | IT-2 maintenance scaling; A4's MDL-Fock closure route |
| A-IT4 | Data Processing Inequality | Shannon 1948; Cover-Thomas §2.8 | IT-3 / IT-4 dark-correction taxonomy (DPI is the load-bearing step); A3's partial-trace structure |
| A-IT5 | Rate-Distortion Theorem | Shannon 1959 | **Published foundation for A2-refined** (selective retention). Cited in A2's canonical statement above. |
| A-IT6 | KL Divergence non-negativity | Kullback-Leibler 1951 | Consistency check for "equivalent representations" claims in A2 |
| A-IT7 | Sagawa-Ueda Generalized Second Law | Sagawa-Ueda 2010, 2012 | Observer extraction bounds (not yet load-bearing in any specific prediction) |

**Cited theorems downstream of A-IT1..7 (also treated as published results):**

| Label | Content | Source doc |
|---|---|---|
| IT-1 | Observer stable iff Re > 1/η_max | `information_theoretic_stability_axioms.md` §II |
| IT-2 | M(N) ∝ N^((d-1)/d) in d dimensions | `information_theoretic_stability_axioms.md` §III |
| IT-3 | Bulk observables: ×(1 + |D|/k*) | `information_theoretic_stability_axioms.md` §V |
| IT-4 | Boundary observables: ×(1 − α₁) | `information_theoretic_stability_axioms.md` §V |
| IT-5 | z* = 17/6 unique Ihara fixed point | `information_theoretic_stability_axioms.md` §VI |

**Axiom count remains FIVE.** A-IT1..7 and IT-1..5 are supporting published material. This framework's commitment to A1-A5 is the same as before session 3; we have added supporting literature, not new postulates.

**When to cite A-IT axioms directly in a derivation:**

- A-IT5 (Rate-Distortion) is the published foundation for A2. Cite alongside Rissanen 1978/1983 when A2's canonicalization is invoked.
- A-IT4 (DPI) is load-bearing for the IT-3/IT-4 bulk/boundary taxonomy. Cite when classifying an observable as delocalized bulk vs edge-local boundary.
- A-IT3 (Landauer) is load-bearing for any erasure-cost argument. Cite in maintenance-cost derivations (IT-2 downstream).
- Others: cite only when the specific result is invoked.

**Cross-references.**

- `information_theoretic_stability_axioms.md` -- the canonical statement of A-IT1..7 and IT-1..5.
- an internal A2 selective-retention downstream audit §3.1 -- the 5/3 vs 7/3 analysis showing IT-3 applies to a DIFFERENT correction than our ADOPTED-DARK-MAP (not a drop-in replacement).

---

## 10. Summary table (UPDATED 2026-05-08 — A1 demoted to derived theorem; top-level is (A) + (B) + (I) + A5-mass)

**Major update 2026-05-08 (A1-elimination):** Session work on 2026-05-07/08 derived A1 itself from two top-level commitments + standard published mathematics + one interpretive commitment. Combined with the prior demotions (A2/A3/A4 in 2026-04-26; P1' in 2026-05-02), the framework's foundation is now structurally complete at the top level: every prior structural axiom is a derived theorem.

Theorem documents:

- `../theorems/theorem_toggle_from_self_containment.md` — A1 derived from (A) self-containment + (B) finite observer + Shannon 1948 + Jaynes 1957 + Cover-Thomas 2006 + Serre 1980 + (I) active reading of binary distinctions (2026-05-07). Demotes A1 from structural axiom to derived theorem; supplies F_inv(E) and the Cayley-graph-of-F_inv(E) identification.
- `../theorems/theorem_substrate_agnosticism.md` — Substrate-in-itself underdetermined; observational equivalence class admits Cayley graph of F_inv(E) as description-length-minimal canonical representative; framework predictions invariant within class (2026-05-08). Establishes substrate agnosticism via Kolmogorov complexity (Kolmogorov 1965 / Solomonoff 1964 / Li-Vitányi 2008).
- `../theorems/theorem_p1_prime_derived_from_a1.md` — P1' derived (2026-05-02). Under the new slate, the prior MR1/MR2/MR3 framing of this theorem is subsumed: MR1 → (A); MR2 → (B); MR3 → consequence of (B) + standard finite-computation theory. The theorem's content is preserved; its axiom-invocation reads from (A) + (B) rather than from A1 + MR1/MR2/MR3.
- `../theorems/theorem_A2_mdl_from_finite_register.md` — A2 derived as theorem of finite-register source coding (Shannon 1948 + Rissanen 1978 + Grünwald 2007 §17). Preamble to be updated to cite (A) + (B) via the toggle theorem.
- `../theorems/theorem_A3_complex_hilbert_from_multiway.md` — A3 derived via multiway-level Stone (Stage 2a + Stage 3 rapid-decay continuum closure + register-is-real field selection). Preamble to be updated.
- `../theorems/theorem_car_local_jordan_wigner.md` — A4 locally derived (Session 11) via Jordan-Wigner. Global A4 remains open (B1 ordering); no current prediction-DAG file requires global CAR (per W1 audit, 2026-04-26). Preamble to be updated.

| Item | One-line content | Status |
|---|---|---|
| **(A)** | Self-containment of the universe — nothing comes from outside, because there is no outside | **Metaphysical commitment** (irreducible; not derivable) |
| **(B)** | Finite observer — the framework describes observers with finite memory capacity | **Scoping commitment** (definitional; describes the framework's subject, not a physical postulate) |
| **(I)** | Active reading of binary distinctions — a binary distinction is read as an operation that moves between two values, not as a static attribute | **Interpretive commitment** (motivated by the relational stance that (A) suggests; alternative readings exist but forfeit relational physics) |
| **A5-mass** | The Ramanujan eigenvalues of the substrate's Bloch-Hashimoto operator are identified with the SM mass spectrum (and analogously for couplings under A5b's MDL-probability identification) | **Empirical labeling** (irreducible empirical content; identifies which math object corresponds to which physical observable) |
| A1 | Binary self-inverse toggle T_e² = id; algebra is F_inv(E) | **Derived theorem** — see `../theorems/theorem_toggle_from_self_containment.md` |
| A2 | MDL waterline: every encoding with L(M) + L(data\|M) < L(raw) is retained, plurally weighted by compression savings | **Derived theorem** — see `../theorems/theorem_A2_mdl_from_finite_register.md` |
| A3 | Substrate's natural state space is complex L²(F_inv(E)); mixed states are partial traces of pure substrate states | **Derived theorem** — see `../theorems/theorem_A3_complex_hilbert_from_multiway.md` |
| A4 | Local CAR at each k*-valent node, generating Cl(2k*; ℂ) | **Derived theorem (local)** — see `../theorems/theorem_car_local_jordan_wigner.md`. Global CAR remains open but not load-bearing for any current prediction. |
| P1' | The observer exists within the framework as a finite register, built from the same primitive (binary toggles) as the substrate, persisting across multiple observations | **Derived theorem** — see `../theorems/theorem_p1_prime_derived_from_a1.md` |
| Substrate agnosticism | Observer-substrate response patterns partition substrate space into equivalence classes; Cayley graph of F_inv(E) is description-length-minimal canonical representative; framework predictions invariant within class | **Derived theorem** — see `../theorems/theorem_substrate_agnosticism.md` |

**Field selection** (ℂ over ℝ for the substrate's Hilbert space) is *structural*, derived via the register-is-real argument: under (B) the observer's register stores binary/real values, so register-extractable spectral content must be real; on ℝ-L² the relevant Stone generator has imaginary spectrum (incompatible); on ℂ-L² it has real spectrum (compatible); on ℍ-L² Adler 1995's quaternionic Stone gives anti-self-adjoint generator with σ ⊂ Im(ℍ) (3-real-dim pure-imaginary quaternions, also incompatible — see uniqueness ledger Row 5 + R-6 closure 2026-04-27, `proofs/foundations/r6_quaternionic_su2L_check.py`). A5-mass is NOT load-bearing for field selection. The structural derivation is in `../theorems/theorem_A3_complex_hilbert_from_multiway.md` (updated 2026-04-27 to drop A5-mass dependency from chain (iv)) and refined in the operator sweep §F (`../operator_sweep/operator_sweep_from_A1.md`).

**Further tightening (2026-04-27 update via R-6 closure of uniqueness ledger Row 5):** The ℝ-vs-ℂ field-selection chain originally cited A5-mass to force real eigenvalues. Following the 2026-04-27 closure of R-6 (quaternionic ℍ residue, REFUTED), the field-selection chain is now closable from (B) (via the prior P1' theorem) alone. `theorem_A3_complex_hilbert_from_multiway.md` Step 7 has been rewritten accordingly. Complex Hilbert space structure is now derivable without invoking the empirical mass spectrum.

**Continuum-limit closure** for the unitary-evolution piece (used in the A3 derivation) is rigorous via Stage 2a + Stage 3 rapid-decay (CAS-verified ξ_t ≈ 0.558 ℓ_P sub-Planckian) + Strauch 2006 + Childs 2009. Continuum-limit for the smooth-manifold portion (used in cosmological / GR-style predictions) remains partial — same gap as Stage 3 Lorentz.

**Honest summary (POST 2026-05-08):** the framework's foundation is:

> *(A) self-containment + (B) finite observer + (I) active reading of binary distinctions + A5-mass empirical labeling + standard published mathematics = the Standard Model.*

(A) is metaphysical, not derivable. (B) is a scoping definition — it says what kind of subject the framework's predictions describe. (I) is an interpretive commitment, motivated by the relational stance and named explicitly to avoid smuggling. A5-mass is the empirical anchor (which math = which physics). Standard mathematics is Shannon, Jaynes, Rissanen, Grünwald, Serre, Kolmogorov, Stone, CDP, Sunada, Childs-Strauch, Jordan-Wigner, etc. — all published, none new to the framework.

Everything else — A1 (binary self-inverse toggle), F_inv(E), the Cayley-graph substrate, MDL waterline (A2), complex Hilbert space structure (A3), local fermionic statistics (A4), observer concept (P1'), substrate agnosticism, and the entire downstream scaffold — is derived theorem.

Operator sweep at `../operator_sweep/operator_sweep_from_A1.md` enumerates ~180 mathematical operations the framework's structural content permits, layer by layer, with field selection and continuum-limit closure as interleaved derivations. (Sweep filename retained from prior slate; under the new slate, "from A1" reads as "from the toggle theorem," semantically equivalent.)

**Bridge convention for comparison to Standard Model observables.** Framework-native tree-level couplings (V_us, V_cb, v_Higgs, m_τ, etc.) are NOT MS̄-at-some-scale objects; they are combinatorial outputs augmented by Feshbach self-energy corrections from substrate the MDL projection threw away. Comparisons to SM observables go through "bare + Feshbach = SM pole-mass-equivalent," not through MS̄ scheme machinery. See [`framework_scheme_convention.md`](framework_scheme_convention.md) for the canonical statement, with worked examples (Higgs v 5/12 correction, V_us Class-2 stripping, m_ν Class-1 Feshbach, λ_Higgs and y_τ as open residuals). The convention is load-bearing for how Clause-8 σ_PDG comparisons are evaluated; readers comparing framework predictions to PDG values should consult it.

### Historical record of the prior axiom slates

For audit and historical interest, the prior axiom statements are preserved in §§2–5b above. These should be read as *what the framework asserted* at successive points in its development:

- Pre-2026-04-26: A1 + A2 + A3 + A4 + A5 (five-axiom slate; A2/A3/A4 as structural axioms).
- 2026-04-26 to 2026-05-02: A1 + P1' (definitional) + A5-mass (A2/A3/A4 demoted to derived theorems; P1' a definitional commitment).
- 2026-05-02 to 2026-05-08: A1 + A5-mass (P1' demoted to derived theorem via MR1/MR2/MR3).
- Post-2026-05-08: (A) + (B) + (I) + A5-mass (A1 demoted to derived theorem; substrate agnosticism added as derived theorem).

The *content* of A1, A2, A3, A4 is preserved (now as theorems) — only their *axiomatic status* changes at each step. A5's content is preserved as A5-mass (the labeling clause) with the previously-bundled "framework describes external reality" commitment subsumed into the (A) + (B) framing.

## 10. References

### Cited mathematical theorems

- **Chiribella, G., D'Ariano, G.M., Perinotti, P.** (2011). Informational derivation of quantum theory. *Phys. Rev. A* **84**, 012311. The five-axiom derivation of finite-dim complex Hilbert-space QM. Theorem 25 (Section VIII) is the load-bearing result for A3's role.
- **Dakic, B., Brukner, C.** (2011). Quantum theory and beyond: is entanglement special? arXiv:0911.0695.
- **Hardy, L.** (2001). Quantum theory from five reasonable axioms. arXiv:quant-ph/0101012.
- **Masanes, L., Mueller, M.P.** (2011). A derivation of quantum theory from physical requirements. *New J. Phys.* **13**, 063001.
- **Gleason, A.M.** (1957). Measures on the Closed Subspaces of a Hilbert Space. *J. Math. Mech.* **6**, 885-893.
- **Rissanen, J.** (1978). Modeling by shortest data description. *Automatica* **14**, 465-471.
- **Rissanen, J.** (1983). A universal prior for integers and estimation by minimum description length. *Annals of Statistics* **11**, 416-431.
- **Grunwald, P.** (2007). *The Minimum Description Length Principle.* MIT Press. Sections 5.1-5.3 (model equivalence and canonicalization).
- **Serre, J.-P.** (1980). *Trees.* Springer. Section I.1 Proposition 4 (free product of Z/2's; reduced word uniqueness).

### Framework documents

- `framework_architecture.md` -- multi-layer view; Layer 1 multiway substrate, Layer 2 visible srs, Layer 6 dark sector.
- `../../predictions/walker_dynamics_derivation.md` -- W1, W2, W3 + Reading Conventions; the Open System Reading is the natural reading under A3.
- `../../predictions/observer_dim_three_derivation.md` (B7.1) -- Gleason chain to n = 3; under A1 + A2 + A3 the Hilbert-space premise is now derived.
- an internal audit of the seven Gleason sub-assumptions -- the seven-assumption audit that identified G.1 and G.5 as the gaps.
- an internal strict-gating audit -- the 26 BLOCKED predictions that A3 closure unblocks.
- an internal Markov-vs-unitary classification audit -- per-file Markov-vs-unitary classification; A3 is the structural input that resolves the Markov-vs-unitary tension at the framework-architecture level (the visible sector is the partial-trace image of a pure state, hence a unitary system; the dark sector is the trace-out target, hence appears classical / Markov from the visible side).
- `docs/theorem_H_multiway_construction.md` -- the explicit length-graded H_unred^(L) = H_visible^(L) (+) H_dark^(L) decomposition; under A3 the canonicalization H_unred^(L) -> H_visible^(L) is the partial trace over H_dark^(L).
- `predictions/H_multiway_dim_count.py` -- explicit dim counts of the visible and dark sectors at length L.

### Memory / standards (flagged for update)

- `../parameters/parameter_linter.md` -- hard quality gate; the four-item list (axiom / explicit algebra / cited theorem / upstream closed file) is unchanged; the only change is that "axiom" now includes A3 in addition to A1 and A2.

## 11. Scope honesty

Under the post-2026-05-08 slate, the framework's commitments divide into four kinds: one metaphysical, one scoping, one interpretive, one empirical.

**(A) self-containment is metaphysical.** It cannot be proved from anything more fundamental, because it stipulates that nothing more fundamental is supplied. (A) is the framework's only foundational article-of-faith. Frameworks that smuggle in external information — boundary conditions, anthropic priors, fine-tuning, multiverse selection — make additional metaphysical commitments that this framework refuses.

**(B) finite observer is scoping.** It says what kind of subject the framework's predictions describe: an observer with finite memory. This is not a physical postulate (the framework does not claim the universe is finite — the substrate constructed under the toggle theorem is countably infinite). It is a definition of the case the framework addresses, which is the actual case for any real observer.

**(I) active reading of binary distinctions is interpretive.** It is the structural choice to read the observer's primitive distinction as an *operation* rather than as a static *attribute*. Adopting (I) is what locates dynamics in the observer's traversal of the substrate (the relational stance) rather than in any substrate evolution. Alternative readings (passive, asymmetric) yield different framework structures and forfeit the relational stance; the framework adopts (I) explicitly rather than smuggling it in.

**A5-mass is empirical.** It is not a structural claim about the substrate — it is the declaration that the substrate's mathematical structure corresponds to the substrate *of particle physics*. Every physical theory has an equivalent commitment, usually implicit: Newton's mechanics assumes it applies to real objects; BCS theory assumes its gap equation describes real superconductors; the Standard Model assumes its Lagrangian is the Lagrangian of nature. A5-mass plays that role here. It cannot be derived; it is validated by prediction accuracy.

A1, A2, A3, A4, and P1' are now derived theorems of (A) + (B) + (I) + standard published mathematics. They make no independent commitments beyond what (A), (B), and (I) supply.

The framework's honest claim is therefore: the Standard Model visible-sector mass spectrum, mixing angles, and coupling constants are derivable from (A) + (B) + (I), given the empirical labeling A5-mass and standard published mathematics. The derivations are either STRICT-SOLID (requiring no further adoption beyond the top-level slate) or carry one of the remaining residue items tracked in `../audits/registers/structural_residue_register.md`.

A5-mass also cleans up a layer of confusion that accumulated in earlier documentation. Labels like ADOPTED-P1, ADOPTED-CS, and ADOPTED-Y were written as if they were independent structural postulates. They are not — they are all downstream restatements of A5. The scalar pairing theorem proves that under the framework's structural content, A5's identification has a unique consistent form: the mass operator must be the C₃-scalar gap operator on V_Ram. This is not a new axiom; it is a theorem showing A5 is internally constrained, not free-floating.

## 12. Bridge convention to Standard Model observables

The five axioms specify how the framework's couplings are DERIVED. They do not by themselves specify how those derived numbers are COMPARED to Standard Model observables. The framework's tree-level couplings are not MS̄-at-some-scale objects — they are framework-native combinatorial outputs, augmented by structural Feshbach self-energy corrections from substrate the MDL projection threw away. Comparison to SM observables therefore goes through a bridge convention rather than through conventional renormalization-scheme machinery.

This bridge convention is set out in `framework_scheme_convention.md`. The short statement: a framework-native coupling C equals C_bare (computed on the visible srs from MDL-licensed structures per A5(b)) plus a Feshbach self-energy Σ_C (computed from the substrate complement of the MDL projection); the total C_bare + Σ_C is intended to equal the SM pole-mass-equivalent coupling at the observable's natural physical scale, with no further scheme/scale machinery applied. The (5/12) Higgs VEV correction (`predictions/v_higgs.py`) and the (Im(h)/|h|²) amplitude correction on V_us / m_ν (the author's separate private derivation §4a) are the canonical examples of derived Feshbach corrections. Analogs on λ and y_τ are open research items (Priority 4.4 step 2.1 of `docs/master_plan.md`).

This convention applies to framework-native α₁-dependent tree-level couplings (λ, y_τ, V_us, V_cb, m_ν, θ_23, ...). It does NOT apply to couplings whose comparison to data explicitly requires SM RG running (g_1, g_2, g_3, α_em, α_s, sin²θ_W at M_Z) — those use standard SM/MSSM RG with M_Z as input and are outside the bridge convention's scope. See `framework_scheme_convention.md` §7 for the full scope statement.
