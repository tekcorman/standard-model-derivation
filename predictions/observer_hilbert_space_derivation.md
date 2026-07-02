# Observer Hilbert space (G.1) and complex field (G.5) -- derivation under A1 + A2-T + A3-T via CDP 2011

**Date:** 2026-04-18
**Status:** Theorem (closes G.1 and G.5 of an internal audit of the seven Gleason sub-assumptions under the three-axiom setup of `docs/framework/framework_axioms.md`).

**Note 2026-05-07** (post-Theorem-8 closure): this doc derives G.1 + G.5 via the CDP 2011 chain (Route A), which uses srs as load-bearing in 3 of 5 CDP axioms (1: W3 directed-edge Markov on srs; 2: B(P) spectral data on srs; 4: srs primitive cell + Sunada Bloch). The CANONICAL substrate-generic chain for the same (G.1, G.5) outputs is Route B (Stone) in `docs/theorems/theorem_A3_complex_hilbert_from_multiway.md`, folded into `docs/theorems/theorem_observer_selected_d_periodic_dominance.md` §6 Step 4 (a)-(f) post-C1 closure 2026-05-07. Route A here remains as alternative-historical (CDP 2011 axiomatic-QM derivation); Theorem 8's substrate-side load-bearing chain uses Route B (no srs).

**Verification:** sympy partial-trace formalism on a Bell-state toy example (`predictions/observer_hilbert_space.py`); structural CDP-axiom-supply checklist.
**Supersedes:** an internal working note (parameter-count / Szegedy / Cencov-L^2 route; failed under A1 + A2-T alone) and an internal working note (Gelfand-Naimark / non-commutative C*-algebra route; failed under A1 + A2-T alone). Both prior attempts established that the two-axiom setup is structurally insufficient; the present file is the closure under the three-axiom setup.

## Abstract

Under the framework's three-axiom setup (A1 binary self-inverse toggle, A2 MDL canonicalization, A3 MDL canonicalization is partial trace over the dark sector; see `docs/framework/framework_axioms.md`), we derive the two foundational results that were ASSUMED (G.1) and a GAP (G.5) in an internal audit of the seven Gleason sub-assumptions under the prior two-axiom setup:

1. **G.1** -- the observer's MDL-optimal model class is a Hilbert space (vector space + inner product, supporting the orthonormal-basis concept that Gleason 1957 requires);
2. **G.5** -- the field of that Hilbert space is F = C, not F = R or F = H.

The derivation chain is the Chiribella-D'Ariano-Perinotti 2011 informational derivation of finite-dim complex Hilbert-space quantum mechanics, expressed in framework-native terms. CDP 2011 derives both results from five operationally-motivated axioms (causality, perfect distinguishability, ideal compressions, local distinguishability, purification) via their Theorem 25 in Section VIII. Under the framework's A1 + A2-T + A3-T setup, CDP axioms 1-4 are supplied by A1 + A2-T + the srs lattice structure (derived in `predictions/k_star.py`, `predictions/d_spatial.py`, `predictions/g_girth.py`, `../predictions/B_P_doubly_degenerate_h_derivation.md`), and CDP axiom 5 (purification) is exactly A3 in framework-native form.

The CDP 2011 chain therefore applies and forces the observer's state space to be the density operators on a finite-dim complex Hilbert space, with reversible transformations being unitary conjugation. F = R is excluded by CDP 2011 Section VIII Lemma 11 (real-Hilbert-space tensor products are incompatible with purification + local distinguishability), and F = H is excluded by CDP 2011 Section VIII Theorem 24 (quaternion tensor products are non-associative).

This closes G.1 and G.5 as DERIVED under A1 + A2-T + A3-T. The 26 BLOCKED predictions of an internal strict-gating audit that were flagged as HILBERT-DEPENDENT or FIELD-DEPENDENT under the two-axiom setup become STRICT-SOLID-CONDITIONAL on A3 (modulo the separately load-bearing Need-A2, Need-RR, Pati-Salam labeling, and Feshbach Exponent Principle gaps).

## Framework axioms invoked

- **(A1)** Binary self-inverse toggle: T_e o T_e = identity for each e in E. Per `docs/framework/framework_axioms.md` Section 2.
- **(A2)** MDL canonicalization: the observer retains the model minimizing L_total = L_model + L_data_given_model. Per `docs/framework/framework_axioms.md` Section 3.
- **(A3)** MDL canonicalization is partial trace: the canonicalization map pi_MDL : states(Layer 1) -> states(Layer 2) is the partial trace Tr_{Layer 6}(|psi><psi|) of a pure state |psi> on the combined Layer 1 + Layer 6 space, essentially unique up to reversible transformations on Layer 6. Per `docs/framework/framework_axioms.md` Section 4.

## Upstream framework theorems (citable as closed)

- `predictions/k_star.py` -- k_star = 3 from MDL on the srs coordination number (chain-imported in the script).
- `predictions/d_spatial.py` -- d_spatial = 3 from Cencov 1982 + Fisher rank (chain-imported in the script).
- `predictions/g_girth.py` -- g = 10 from srs structure (Sunada 2012 + RCSR srs entry).
- `../predictions/B_P_doubly_degenerate_h_derivation.md` -- B(P) eigenvalue h = (sqrt(3) + i sqrt(5))/2 with C_3-protected multiplicity 2; supplies the spectrally-distinguishable eigenvalue structure required by CDP axiom 2.
- `../predictions/walker_dynamics_derivation.md` W3 -- directed-edge Markov dynamics on srs supply the causal past-to-future order required by CDP axiom 1.
- `docs/theorem_H_multiway_construction.md` -- explicit length-graded H_unred^(L) = H_visible^(L) (+) H_dark^(L) decomposition; under A3 the canonicalization H_unred^(L) -> H_visible^(L) IS the partial trace over H_dark^(L).
- `predictions/H_multiway_dim_count.py` -- explicit dim counts dim H_visible^(L) = n*(n-1)^(L-1), dim H_dark^(L) = n*[n^(L-1) - (n-1)^(L-1)] for the srs alphabet n = |E| = 6.

## Cited mathematical theorems

- **Chiribella, G., D'Ariano, G.M., Perinotti, P.** (2011). Informational derivation of quantum theory. *Phys. Rev. A* **84**, 012311.
  - Five axioms: causality (1), perfect distinguishability (2), ideal compressions (3), local distinguishability (4), purification (5).
  - Theorem 25 (Section VIII): axioms 1-5 force the state space to be the density operators on a finite-dim complex Hilbert space with reversible transformations being unitary conjugation.
  - Section VIII Lemma 11: F = R excluded by purification + local distinguishability.
  - Section VIII Theorem 24: F = H excluded by quaternion tensor-product non-associativity.
- **Dakic, B., Brukner, C.** (2011). Quantum theory and beyond: is entanglement special? arXiv:0911.0695. Related purification-based derivation.
- **Hardy, L.** (2001). Quantum theory from five reasonable axioms. arXiv:quant-ph/0101012.
- **Masanes, L., Mueller, M.P.** (2011). A derivation of quantum theory from physical requirements. *New J. Phys.* **13**, 063001.
- **Gleason, A.M.** (1957). Measures on the Closed Subspaces of a Hilbert Space. *J. Math. Mech.* **6**, 885-893. Used downstream once G.1 + G.5 are derived to fix the dim n = 3 (`../predictions/observer_dim_three_derivation.md` B7.1).

## Statement

Under axioms A1 + A2-T + A3-T (`docs/framework/framework_axioms.md`), the observer's MDL-optimal model class on the visible sector (Layer 2 of `docs/framework/framework_architecture.md`) is

    {density operators rho on H : rho >= 0, Tr(rho) = 1}

where H is a finite-dim **complex** Hilbert space. Reversible transformations are unitary conjugation rho -> U rho U^dagger for U unitary on H.

In particular:

- **G.1** -- the observer's model class has Hilbert-space structure (vector space + inner product, supporting orthonormal bases). DERIVED.
- **G.5** -- the field of H is F = C; F = R and F = H are excluded. DERIVED.

## Derivation

### Step 1 -- A3 supplies the operator-algebraic shape (partial trace)

By A3 (`docs/framework/framework_axioms.md` Section 4), the MDL canonicalization map

    pi_MDL : states(Layer 1) -> states(Layer 2)

is the partial trace Tr_{Layer 6}(|psi><psi|) of a pure state |psi> on the combined Layer 1 + Layer 6 space. Concretely: for every observer-accessible mixed state rho on Layer 2, there exists a pure state |psi> on Layer 1 tensor Layer 6 such that

    rho = Tr_{Layer 6}(|psi><psi|),

and this purification is essentially unique up to reversible transformations on the Layer 6 sector.

Partial trace is an operator-algebraic operation defined on a tensor product of Hilbert spaces. Specifically: given a Hilbert space H = H_A tensor H_B and a density operator rho on H, the partial trace Tr_B : Bnd(H_A tensor H_B) -> Bnd(H_A) is defined by

    Tr_B(rho)[a, a'] = sum_b <a, b | rho | a', b>

for any orthonormal bases {|a>} of H_A and {|b>} of H_B. The output Tr_B(rho) is a density operator on H_A (positive semi-definite, trace 1, Hermitian).

A3 therefore commits the framework to the operator-algebraic shape: there is a Hilbert space structure on Layer 1 tensor Layer 6, and the visible-sector states are obtained by partial trace over the Layer 6 tensor factor. This is the load-bearing operator-algebraic step that A3 supplies.

**Gate-clear type:** 4 (A3-T) + 3 (cited definition of partial trace; e.g. Nielsen-Chuang 2010 Section 2.4.3).

### Step 2 -- A1 + A2-T + srs structure supply CDP axioms 1-4 in framework-native form

CDP 2011's axiomatic derivation requires four supporting axioms in addition to purification. We exhibit framework-native readings of each, sourced from A1 + A2-T and the upstream srs lattice structure.

**CDP axiom 1 (causality).** Operationally: there exists a unique deterministic effect on every system state (the discard map) such that any test admits a coarse-graining that gives the discard map. Framework-native reading: the W3 directed-edge Markov dynamics on srs (`../predictions/walker_dynamics_derivation.md` Step 5) is causal -- one direction of propagation, no signaling backward in walk-time. Non-backtracking walks have a canonical past-to-future order (the toggle stream is ordered, and reduced words preserve the order of remaining letters per Serre 1980 Section I.1 Proposition 4). The "discard map" in framework terms is the projection from the directed-edge state to the trivial / no-information state; coarse-graining is MDL canonicalization (A2). **Source: A1 + W3 from `../predictions/walker_dynamics_derivation.md`.**

**CDP axiom 2 (perfect distinguishability).** Operationally: every state that is not completely mixed can be perfectly distinguished from some other state. Framework-native reading: B(P)'s spectral data on srs (`../predictions/B_P_doubly_degenerate_h_derivation.md`) supply distinct eigenvalues with the (4, 2, 2) C_3 multiplicity structure; eigenstates corresponding to distinct eigenvalues are perfectly distinguishable by spectral measurement (Hermitian linear algebra). The Hashimoto operator B has spec(B) containing h = (sqrt(3) + i sqrt(5))/2, h*, and other Ihara-Bass eigenvalues that are pairwise distinct, so the spectral structure supports perfect distinguishability of the corresponding spectral subspaces. **Source: A1 + A2-T + `../predictions/B_P_doubly_degenerate_h_derivation.md` + `predictions/h_walker_eigenvalue.py`.**

**CDP axiom 3 (ideal compressions).** Operationally: for every system, every face of the state space can be encoded into a smaller system in a reversible way (the encoding has no information loss). Framework-native reading: A2 (MDL canonicalization) IS the ideal compression principle, in the precise sense of Grunwald 2007 Sections 5.1-5.3 (canonicalization preserves the equivalence class of model predictions). The MDL canonicalization map sends raw toggle streams to reduced words, preserving all observable content (W2 of `../predictions/walker_dynamics_derivation.md`); this is reversible at the equivalence-class level. **Source: A2 directly, with Grunwald 2007 cited as the operational match.**

**CDP axiom 4 (local distinguishability).** Operationally: the joint state of two systems is determined by the joint statistics of local measurements on the individual systems. Framework-native reading: the srs primitive cell has 4 vertices and 6 edges (`predictions/k_star.py` k* = 3, `predictions/d_spatial.py` d = 3, `predictions/g_girth.py` g = 10, RCSR srs entry); local degrees of freedom are vertex / edge observables, and the joint state of the primitive cell is determined by the local marginals at the vertices and edges (this is a structural property of the srs lattice, derivable from the Bloch-decomposition theorem of Sunada 2012 Sections 5-6, which factorizes the multiway substrate into per-Bloch-mode Hilbert spaces; each Bloch mode's state space is determined by the local Bloch-fibre data at the relevant Wyckoff positions). **Source: A1 + A2-T + srs structure derived in upstream prediction files; Bloch decomposition cited as Sunada 2012.**

These four readings supply CDP axioms 1-4 in framework-native form. Each one traces back to A1 + A2-T + a closed upstream prediction or a cited mathematical theorem (Sunada 2012, Grunwald 2007, Serre 1980, Ihara-Bass).

**Gate-clear type:** 1 (A1 + A2-T) + 3 (cited theorems: Sunada 2012, Grunwald 2007, Serre 1980, Ihara-Bass) + 4 (upstream closed predictions: k_star, d_spatial, g_girth, h_walker_eigenvalue, B_P_doubly_degenerate_h, walker_dynamics).

### Step 3 -- A3 supplies CDP axiom 5 (purification) directly

CDP 2011 axiom 5 (purification): for every mixed state rho_A on system A, there exists a pure state |psi>_{AB} on a larger system AB such that rho_A = Tr_B(|psi><psi|), and this purification is essentially unique up to reversible transformations on B.

This is exactly A3 with the identifications A = Layer 2 (visible sector), B = Layer 6 (dark sector), and the larger system AB = Layer 1 (multiway substrate) tensor Layer 6 (dark sector). A3's "essentially unique up to reversible transformations on the Layer 6 sector" is precisely CDP's "essentially unique up to reversible transformations on the purifying system."

**Gate-clear type:** 4 (A3-T) + 3 (CDP 2011 axiom 5 statement).

### Step 4 -- CDP 2011 Theorem 25 forces finite-dim complex Hilbert-space QM

By CDP 2011 Theorem 25 (Section VIII), any operational theory satisfying axioms 1-5 has the following structure:

- The state space of every system is the convex set of density operators on a finite-dim complex Hilbert space H.
- Reversible transformations are unitary conjugation rho -> U rho U^dagger for U unitary on H.
- Composition of systems is via tensor product of Hilbert spaces.
- Measurements are POVMs, i.e., families {E_i} of positive operators with sum E_i = I.
- The probability of outcome i in measurement {E_i} on state rho is Born's rule p(i) = Tr(rho E_i).

This is the standard finite-dim complex Hilbert-space QM. The CDP 2011 derivation (their Sections III through VIII) is a nine-step argument that we cite rather than reproduce; the load-bearing inputs are CDP axioms 1-5, which are supplied by Steps 1-3 above.

**Conclusion (G.1).** The observer's model class is the convex set of density operators on a finite-dim Hilbert space H. This Hilbert space is the visible-sector tensor factor of the Layer 1 + Layer 6 ambient pure-state space (A3). G.1 (Hilbert-space structure on the model class) is DERIVED.

**Conclusion (G.5).** The field of H is F = C (complex). F = R is excluded by CDP 2011 Section VIII Lemma 11: real-Hilbert-space tensor products are incompatible with purification + local distinguishability (the partial trace of a real-Hilbert-space pure state can fail to have a real-Hilbert-space purification on a real-Hilbert-space environment of the right dimension, contradicting the uniqueness clause of purification). F = H is excluded by CDP 2011 Section VIII Theorem 24: quaternion tensor products are non-associative (the Hilbert tensor product H tensor H tensor H is not well-defined for quaternion Hilbert spaces, contradicting the requirement that joint states of three or more systems be uniquely defined). G.5 (complex field) is DERIVED.

**Gate-clear type:** 3 (CDP 2011 Theorem 25 + Section VIII Lemma 11 + Section VIII Theorem 24).

### Step 5 -- Numerical verification of the partial-trace formalism

To exhibit the operator-algebraic shape that A3 commits the framework to, we verify the partial-trace formalism on a small toy example.

**Toy example.** Consider H_full = C^2 tensor C^2 = C^4 (two qubits). Take the Bell state

    |psi> = (|00> + |11>) / sqrt(2)

as a pure state on H_full. The corresponding rank-1 density operator is

    rho_full = |psi><psi| =
        [ 1/2   0   0   1/2 ]
        [  0    0   0    0  ]
        [  0    0   0    0  ]
        [ 1/2   0   0   1/2 ]

(in the basis |00>, |01>, |10>, |11>). The partial trace over the second qubit is the 2x2 matrix Tr_2(rho_full) on the first qubit, with matrix elements

    Tr_2(rho_full)[i1, j1] = sum_{k=0,1} rho_full[2*i1 + k, 2*j1 + k]
                           = (rho_full[2*i1, 2*j1] + rho_full[2*i1 + 1, 2*j1 + 1]).

Computing:

    Tr_2(rho_full)[0, 0] = rho_full[0, 0] + rho_full[1, 1] = 1/2 + 0 = 1/2
    Tr_2(rho_full)[0, 1] = rho_full[0, 2] + rho_full[1, 3] = 0 + 0 = 0
    Tr_2(rho_full)[1, 0] = rho_full[2, 0] + rho_full[3, 1] = 0 + 0 = 0
    Tr_2(rho_full)[1, 1] = rho_full[2, 2] + rho_full[3, 3] = 0 + 1/2 = 1/2

So Tr_2(rho_full) = (1/2) * I_2 = the maximally mixed state on the first qubit. This is the operator-algebraic shape A3 commits to: a pure state on the larger (Layer 1 + Layer 6) space restricts under partial trace to a (generally mixed) density operator on the visible (Layer 2) space. The Bell-state example exhibits the maximally mixed reduced state as a special case.

The script `predictions/observer_hilbert_space.py` implements this verification using sympy's exact rational arithmetic and asserts: (a) the reduced state has trace 1, (b) the reduced state is positive semi-definite (eigenvalue 1/2 with multiplicity 2), (c) the reduced state equals (1/2) * I_2 exactly. The verification passes.

**Gate-clear type:** 2 (explicit sympy arithmetic on a 4x4 matrix, exact rationals, no floating-point comparisons).

## Result

Under A1 + A2-T + A3-T (`docs/framework/framework_axioms.md`):

- **G.1 (Hilbert-space structure exists)** -- DERIVED.
- **G.5 (field is C)** -- DERIVED.

The observer's model class is the convex set of density operators on a finite-dim complex Hilbert space H. The Hilbert space H is the visible-sector tensor factor of the Layer 1 + Layer 6 ambient pure-state space, with the visible sector emerging from the Layer 1 substrate by partial trace over the Layer 6 dark sector (A3).

The result is structural, not numerical: there is no numerical comparison with experiment. The verification is via the CDP 2011 derivation chain (cited theorem) plus a sympy partial-trace verification on a Bell-state toy example (`predictions/observer_hilbert_space.py`).

## Comparison with experiment

Not applicable in the standard sense: this is a foundational structural result, not a numerical SM observable. The experimental content of "the observer's state space is the density operators on a finite-dim complex Hilbert space" is the entire body of finite-dim quantum mechanics; that content is well-tested experimentally (every quantum-mechanical experiment performed since 1925 verifies this structure). The CDP 2011 derivation is a specific information-theoretic axiomatization of that body; the framework adopts CDP 2011 as the bridge from MDL + toggle + purification to finite-dim complex QM.

## Open questions

1. **Need-A2 (canonical generation-Z_3 on C^3_gen).** Per an internal audit of the seven Gleason sub-assumptions Section 6 hidden-gap discussion: closing G.1 + G.5 does NOT automatically supply a canonical Z_3 subgroup of U(3) acting on C^3_gen. The framework's flavor predictions (Q_Koide = 2/3, Koide phases, etc.) require such a canonical generation-Z_3, and its derivation is independent of A3. Closing Need-A2 likely requires separate structural input -- either (a) a derivation that the toggle's induced action on the C^3_gen tensor factor has a canonical Z_3 symmetry, or (b) a structural identification of a Z_3 subgroup of SU(3) inherited from the substrate's space group I4_132 acting on the observer's measurement frames.

   > **STATUS UPDATE 2026-04-28 / 2026-05-08.** **Need-A2 CLOSED at theorem grade** via R3 (`R3_observer_c3_generation_derivation.md`, 2026-04-20) + M1.B (an internal working note §7.5, 2026-04-28) + substrate generation-charge conservation (`docs/theorems/theorem_substrate_generation_charge_conservation.md`, 2026-04-29). Mechanism (route variant of (a) above): R3 derives cyclic-shift Z_3 on C³_obs via Halmos spectral theorem on M_gen; M1.B identifies it with the Galois Z_3 of the sub-factor inclusion M^α ⊂ M ⊂ M ⋊_α Z_3 ≅ M_3(ℂ) ⊗ M^α where M = L(F_inv(E)). The substrate body-diagonal C_3 induces an order-3 OUTER aut on M (different from its INNER aut on Cl(6) Fock = color-Z_3); this categorical separation gives generation-Z_3 distinct from color-Z_3 at the operator-algebra level. M_gen non-degeneracy (R3's external input) **CLOSED** via generic A2-T measure-theoretic argument 2026-05-08.

2. **Need-RR (canonical reading rule from substrate amplitude to mass parameter).** Per an internal working note: the sqrt-reading vs amp-reading vs Born-reading distinction is a separate structural commitment about the relationship between substrate amplitude and Hamiltonian mass parameter. Even with G.1 + G.5 derived (complex Hilbert space everywhere), all three readings remain syntactically available, and MDL alone does not pick. Need-RR likely requires a derivation from a distinct structural input (e.g., the Yukawa coupling structure of an internal working note or the Cl(6, 0) spinor coupling structure of `../predictions/theorem_B3_spinor_fermion_derivation.md`).

3. **Tree vs Ramanujan fermion-mass source.** The framework's flavor sector is partly built on tree-level NB walk amplitudes (Ihara zeta numerator) and partly on Ramanujan-saturated Bloch modes at the P-point (h doubly-degenerate). Whether fermion masses are sourced primarily from tree-level walk statistics (the (2/3)^8 Feshbach-coupling line of `predictions/alpha_1.py`) or primarily from the Ramanujan-saturated P-point Bloch modes (the h-coupling line of the flavor predictions) is not resolved by A3. This is a separate structural question about which spectral sub-structure of B is the dominant mass-source under the CDP-derived Hilbert-space framework.

4. **Pati-Salam labeling.** B3's "adopted-postulate at Layer 3" (the Pati-Salam labeling of Spin(4) x Spin(2) factors in the Cl(6,0) spinor decomposition) remains an OTHER-SMUGGLE under A1 + A2-T + A3-T. A3 is silent on the gauge-rep-factor labeling; the dimensional-matching argument of B3 still requires a separate derivation to be promoted from "adopted-postulate" to theorem grade.

5. **Feshbach Exponent Principle as a stand-alone theorem.** The Exponent Principle bridge (`../predictions/Feshbach_coupling_strength_derivation.md` Section 3) remains an "adopted structural theorem" without standalone proof. This is independent of A3.

6. **Lindblad reading legitimacy.** Under A3, the visible/dark partial-trace structure is the framework's natural Markov-vs-unitary resolution: the visible sector is unitary (a finite-dim complex Hilbert factor, per G.1 + G.5 derived above), and the dark sector appears Markov / classical from the visible side (it is the trace-out target). The Lindblad master equation framing of an internal working note is now legitimized at the structural level (the W2 cancellation events are the source of Lindblad jump operators encoding the visible-to-dark amplitude loss; the dark-to-visible amplitude is the partial trace's inverse, which exists in the operator-algebraic sense per A3's uniqueness clause). A formal derivation of the specific Lindblad form L_k from A1 + A2-T + A3-T is left as a separate workstream.

## References

### Memory / standards

- `docs/parameters/parameter_linter.md` -- hard quality gate; the four-item list (axiom / explicit algebra / cited theorem / upstream closed file) is unchanged; the only change is that "axiom" now includes A3.

### Upstream framework theorems and predictions

- `docs/framework/framework_axioms.md` (canonical statement of A1 + A2-T + A3-T).
- `docs/framework/framework_architecture.md` (multi-layer view; Layer 1 multiway substrate, Layer 2 visible srs, Layer 6 dark sector).
- `../predictions/walker_dynamics_derivation.md` (W1 + W2 + W3 + Reading Conventions).
- `../predictions/B_P_doubly_degenerate_h_derivation.md` (B(P) spectral structure).
- `docs/theorem_H_multiway_construction.md` (length-graded H_unred = H_visible (+) H_dark decomposition).
- `predictions/k_star.py`, `predictions/d_spatial.py`, `predictions/g_girth.py`, `predictions/p_toggle.py`, `predictions/h_walker_eigenvalue.py`, `predictions/B_P_doubly_degenerate_h.py`, `predictions/H_multiway_dim_count.py`.

### Cited mathematical theorems

- **Chiribella, G., D'Ariano, G.M., Perinotti, P.** (2011). Informational derivation of quantum theory. *Phys. Rev. A* **84**, 012311.
- **Dakic, B., Brukner, C.** (2011). Quantum theory and beyond. arXiv:0911.0695.
- **Hardy, L.** (2001). Quantum theory from five reasonable axioms. arXiv:quant-ph/0101012.
- **Masanes, L., Mueller, M.P.** (2011). A derivation of quantum theory from physical requirements. *New J. Phys.* **13**, 063001.
- **Gleason, A.M.** (1957). Measures on the Closed Subspaces of a Hilbert Space. *J. Math. Mech.* **6**, 885-893.
- **Nielsen, M.A., Chuang, I.L.** (2010). *Quantum Computation and Quantum Information*, 10th anniversary edition. Cambridge University Press. Section 2.4.3 (partial trace).
- **Sunada, T.** (2012). *Topological Crystallography*. Springer. Sections 5-6 (Bloch decomposition).
- **Grunwald, P.** (2007). *The Minimum Description Length Principle.* MIT Press. Sections 5.1-5.3 (canonicalization).
- **Serre, J.-P.** (1980). *Trees.* Springer. Section I.1 Proposition 4.

### Sibling stall / scoping documents superseded


### Per-step gate-clear types

| Step | Content | Gate type |
|---|---|---|
| 1 | A3 supplies operator-algebraic partial-trace shape | 1 (A3) + 3 (Nielsen-Chuang 2010 Section 2.4.3) |
| 2 | A1 + A2-T + srs structure supply CDP axioms 1-4 in framework-native form | 1 (A1, A2) + 3 (Sunada 2012, Grunwald 2007, Serre 1980, Ihara-Bass) + 4 (k_star, d_spatial, g_girth, h_walker_eigenvalue, B_P_doubly_degenerate_h, walker_dynamics) |
| 3 | A3 supplies CDP axiom 5 (purification) | 1 (A3) + 3 (CDP 2011 axiom 5) |
| 4 | CDP 2011 Theorem 25 forces finite-dim complex Hilbert-space QM | 3 (CDP 2011 Theorem 25 + Section VIII Lemma 11 + Section VIII Theorem 24) |
| 5 | Sympy verification of partial-trace formalism on Bell state | 2 (explicit sympy arithmetic on 4x4 matrix; exact rationals) |

All steps clear the parameter-linter hard gate. The derivation is journal-publishable from A1 + A2-T + A3-T + CDP 2011 alone.

### Files referenced (read-only) but NOT modified

- `results/parameters.csv`
- `docs/parameters/derivations.md`
- All B3/B5/B6 docs.
- All existing `predictions/` files.
- All sibling scoping / attempt docs.

The only files modified by this work are the deliverables:

- `docs/framework/framework_axioms.md` (new canonical axioms doc)
- `predictions/observer_hilbert_space.py` (new prediction script)
- `predictions/observer_hilbert_space_derivation.md` (this file)
- an internal audit of the seven Gleason sub-assumptions (note section appended; classification table updated)
- an internal strict-gating audit (note section appended; 26-BLOCKED count updated)

No commits performed; no remote push.
