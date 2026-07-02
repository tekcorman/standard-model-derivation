# Framework QFT Ontology

**Date:** 2026-04-26.
**Status:** Meta-document. Organizes the operator-sweep audit's ontology lens *by QFT-postulated object* rather than *by catalog operation*. Companion to `../operator_sweep/operator_sweep_from_A1.md` (the catalog) and the audit docs `docs/operator_sweep_audit_layer_*.md` + `../operator_sweep/operator_sweep_audit_appendix.md`.

## Purpose

Standard QFT has many operators that are *postulated* — creation/annihilation, density matrices, Pauli matrices, vacuum |0⟩, propagators, Wick rotation, BRST, renormalization. Their algebraic forms are taken as starting points; "why this form?" is rarely asked, and standard answers ("operational convenience", "historical contingency") don't satisfy.

This document collects what the framework derives from A1 + P1' + downstream theorems for each QFT-postulated object: which are *grounded* in substrate primitives, which are *partially* grounded, and which remain *open gaps*.

The goal is to convert "QFT operators without explanation" into "QFT operators with substrate grounding" wherever the framework's apparatus permits.

## Structure of this document

Each entry has:
- **Standard QFT statement** — what the object is in canonical QFT.
- **Framework grounding** — the substrate origin.
- **Why this form** — the structural reason (forced by what?).
- **Provenance** — operator-sweep catalog ops + theorem docs.
- **Status** — Grounded / Partially-grounded / Open gap.

Entries are organized by QFT subject area:
1. Algebraic structure (CAR, spinors, Pauli, parity)
2. State-space structure (Hilbert space, density matrices, purification)
3. Dynamics (Schrödinger, Wick rotation)
4. Symmetries (T, Berry, gauge)
5. Gauge / GUT structure (Pati-Salam, Lie, Killing)
6. Statistical mechanics / information (entropy, MDL, BZJ, RG)
7. Cosmology / GR (FLRW, Lorentzian, causal structure)
8. **Open gaps** — what the framework does NOT yet ground

---

## 1 — Algebraic structure of QFT

### CAR algebra of fermions: {c_i, c_j†} = δ_ij, {c_i, c_j} = 0

- **Standard QFT.** Postulated as the defining anti-commutation relations of fermion creation and annihilation operators. The reason fermions anti-commute is taken as part of the spin-statistics theorem (which itself rests on Lorentz invariance + locality + positive-energy axioms).
- **Framework grounding.** The Jordan-Wigner transform of involutive substrate toggles produces CAR algebra. Each toggle T_e satisfies T_e² = id (A1 involutivity); JW maps the σ^x-like generators (per edge) into anti-commuting fermion operators c_j = (σ^z_1 ⊗ … ⊗ σ^z_{j−1}) ⊗ σ^−_j on the multi-edge tensor space.
- **Why this form.** Forced. JW is the unique transform converting commuting-on-different-sites involutions into anti-commuting fermions. The σ^z-string handles the parity bookkeeping that makes anti-commutation work. The substrate's involutivity is the input; CAR is the output.
- **Provenance.** Layer 5.6 (JW), 5.7 (CAR); A1 involutivity. Theorem doc: `../theorems/theorem_car_local_jordan_wigner.md`.
- **Status.** **Grounded.** The framework explains *why* fermions anti-commute: because their substrate primitives are involutive.

### Dirac spinor (8-component complex)

- **Standard QFT.** Postulated as the irreducible representation of Cl(1,3) = Cl(4) — gives the 4-component complex Dirac spinor of QFT. (The 8-component version arises in higher-dimensional Cl algebras, e.g., Spin(6) reps.)
- **Framework grounding.** The 8-dim complex spinor is the irreducible representation of Cl(6;ℂ) at each substrate node, where the 6 generators come from 3 undirected edges × 2 directions (k = 3 trivalent srs).
- **Why this form.** Forced by node-locality of the substrate (k = 3 incident edges per node) + complexification (§F field selection) + Lawson-Michelsohn classification of Cl(n;ℂ) irreducible reps (dim 2³ = 8 for Cl(6;ℂ)).
- **Provenance.** Layer 5.9 (spinor reps), 5.30 (Pati-Salam embedding); `predictions/theorem_B3_spinor_fermion.py`.
- **Status.** **Grounded.** The dimensionality (8) and complex structure of the Dirac spinor are derived, not postulated.

### Pauli matrices σ^x, σ^y, σ^z

- **Standard QFT.** Postulated as the generators of SU(2) acting on a 2-component complex spinor; appear in spin operators, qubit gates, electroweak structure.
- **Framework grounding.** Edge-qubit operators on undirected edges of the K_4-quotient. The two directed versions of an undirected edge {u, v} carry T_1, T_2 with T_1² = T_2² = id and {T_1, T_2} = 0 — generators of Cl(0,2;ℝ), which on ℂ² is realized by σ^x and σ^z. σ^y = iσ^x σ^z exists only after §F field-selection.
- **Why this form.** Forced. Each undirected substrate edge has exactly two directed versions; these are involutive and anti-commute as edge operators. The Cl(0,2) → ℂ² action gives σ^x, σ^z. σ^y requires i (post-§F).
- **Provenance.** Layer 5.2; `predictions/G2_cl2_channels.py`.
- **Status.** **Grounded.** σ^x and σ^z are forced by substrate edge structure; σ^y appears post-field-selection.

### Fermion-number conservation: (−1)^F

- **Standard QFT.** Postulated as a global ℤ/2 symmetry of the action; conservation of fermion number mod 2.
- **Framework grounding.** ℤ/2-grading by toggle-count parity. The σ^z-string product in JW (5.6) is exactly the parity operator (−1)^F.
- **Why this form.** Forced. JW maps σ^z-products to fermion-number parity; this is the JW-image of substrate's natural toggle-count parity.
- **Provenance.** Layer 5.10.
- **Status.** **Grounded.** Fermion-number conservation is the substrate's toggle-count parity, viewed through JW.

### Majorana fermions

- **Standard QFT.** A real-fermion structure with γ_i = γ_i† and γ_i² = 1; appears in BSM theories and topological superconductors.
- **Framework grounding.** Hermitian combinations of CAR ops: γ_{2j−1} = c_j + c_j†, γ_{2j} = i(c_j† − c_j). These are *directly* the substrate's directed-edge generators T_1, T_2 (up to JW dressing). The substrate's primitive operators are involutive (Hermitian, T_e² = 1), so Majorana is the *natural* fermionic basis at substrate scale.
- **Why this form.** Forced by substrate involutivity. CAR ops c, c† are *complex combinations* of Majorana — the Dirac fermion is downstream of Majorana in this framework, *inverting* the standard QFT priority where Dirac is primary.
- **Provenance.** Layer 5.11; `predictions/G2_cl2_channels.py` ("T_1 = σ^x ⊗ I (Majorana mode 1); T_2 = σ^z ⊗ σ^x (JW string for mode 2)").
- **Status.** **Grounded** — and provides a structural explanation for why Majorana might be more fundamental than Dirac (substrate primitives are involutive, complex combinations are downstream).

### Hermitian observables

- **Standard QFT.** Postulated: physical observables correspond to self-adjoint operators on Hilbert space, with real spectra.
- **Framework grounding.** §F field-selection forces observables to be self-adjoint on ℂ-L²: register-storable observable values must be real, Stone's complex-form requires self-adjoint generator, Stone's real-form would give imaginary spectrum (incompatible with register).
- **Why this form.** Forced by P1' (observer is finite register) + Layer 3 Stone + register-storability requirement. Hermiticity is *not* postulated; it is derived.
- **Provenance.** Layer 5.3, §F derivation in `../theorems/theorem_A3_complex_hilbert_from_multiway.md`.
- **Status.** **Grounded.** The "Hermitian observable postulate" of QM is derived from finite-register storability.

### Anti-Hermitian generators (gauge field generators)

- **Standard QFT.** Generators of unitary gauge transformations are anti-Hermitian: A* = −A. The factor i in iA_μ converts physical (Hermitian) gauge potentials to anti-Hermitian generators.
- **Framework grounding.** Derivative of unitary at identity is anti-Hermitian (standard Lie theory); for substrate's complex Lie groups Spin(6;ℂ), SU(n) etc., the anti-Hermitian generators close on Lie brackets via structure constants.
- **Why this form.** Standard mathematical structure post-§F.
- **Provenance.** Layer 5.4, 4.40 (Lie algebra), 4.42 (structure constants).
- **Status.** **Grounded** as Lie-algebraic consequence of unitary group + complexification.

---

## 2 — State-space structure

### Complex Hilbert space

- **Standard QFT.** Postulated as the state space of a quantum system; complex-valued amplitudes underlie superposition and interference.
- **Framework grounding.** L²(F_inv(E); ℂ) on the substrate's Cayley graph. The complex field is *forced* by the §F register-is-real argument: a finite register must store real eigenvalues, which require a complex generator (real-skew-symmetric generators give imaginary spectra incompatible with register storability).
- **Why this form.** Forced by A1 + P1' + Stone's theorem. The substrate is field-agnostic through Layer 4; ℂ enters at §F as the unique field consistent with finite-register reading.
- **Provenance.** Layer 2.5 (L²), 5.1 (i), 3.4 (Stone complex form), §F derivation; `../theorems/theorem_A3_complex_hilbert_from_multiway.md`.
- **Status.** **Grounded.** The complex Hilbert space is derived, not postulated.

### Density matrix ρ (mixed states)

- **Standard QFT.** Postulated for systems in incomplete information / open quantum systems / thermal ensembles. Mixedness is taken as a basic state-space possibility alongside pure states.
- **Framework grounding.** Mixed states emerge from compression. A3-T's purification axiom (CDP 2011) gives every observer mixed state ρ_A as the partial trace of a pure substrate-auxiliary state |ψ⟩_AB. Mixedness is *derived* from observer integrating out auxiliary degrees of freedom (compression).
- **Why this form.** Forced. CDP 2011 purification axiom is a theorem-grade structural property; compression (A2-T's MDL canonicalization) is the operation that produces mixedness.
- **Provenance.** Layer 5.12 (density matrix), 5.14 (partial trace), 5.15 (purification); `../theorems/theorem_A3_complex_hilbert_from_multiway.md`.
- **Status.** **Grounded.** Mixed states arise from substrate-observer compression, not as a separate postulate.

### Pure vs mixed state distinction

- **Standard QFT.** Pure = state of the full system; mixed = state of a subsystem after tracing out the rest.
- **Framework grounding.** Pure = pre-compression substrate state; mixed = post-compression observer state. The distinction maps onto the observer's compression operation.
- **Why this form.** Direct corollary of the purification grounding.
- **Provenance.** Layer 5.13.
- **Status.** **Grounded.**

### Partial trace

- **Standard QFT.** Operationally defined: trace over a tensor-factor subsystem.
- **Framework grounding.** *The* compression operation. A3-T identifies MDL canonicalization with quantum partial trace over the abstract auxiliary purifying space.
- **Why this form.** A3-T derived structure: partial trace is the quantum realization of compression.
- **Provenance.** Layer 5.14, 4.24; A3-T derivations across `predictions/m_tau_derivation.md`, `predictions/y_tau_derivation.md`, `predictions/R3_observer_c3_generation_derivation.md`.
- **Status.** **Grounded.**

### Purification: every mixed ρ has a pure |ψ⟩_AB

- **Standard QFT.** Theorem in finite-dimensional quantum information; lifted to infinite-dimensional contexts via Stinespring dilation.
- **Framework grounding.** CDP 2011 axiom — a structural property of compression.
- **Why this form.** CDP 2011 theorem (axiom-grade in their formulation).
- **Provenance.** Layer 5.15; `predictions/observer_hilbert_space_derivation.md`.
- **Status.** **Grounded** via CDP 2011.

### Multi-particle Fock space (tensor products)

- **Standard QFT.** Constructed by tensor products of single-particle Hilbert spaces, with appropriate (anti-)symmetrization for bosons (fermions).
- **Framework grounding.** Multi-edge / multi-node states as tensor products of local Cl(2;ℂ) factors per edge. Anti-symmetrization is automatic via JW (CAR is anti-commutation = anti-symmetrization).
- **Why this form.** Substrate locality (each edge has its own ℂ²) + JW.
- **Provenance.** Layer 5.17, 2.36 (Hilbert tensor product).
- **Status.** **Grounded.**

### Vacuum |0⟩

- **Standard QFT.** Postulated lowest-energy state. Vacuum expectation values ⟨0|T(φφ)|0⟩ ground propagator computations. *No first-principles structural account in canonical QFT.*
- **Framework grounding.** **Substrate's maximally-symmetric Bloch-trivial eigenstate of adjacency operator A at λ_max = k = 3.** Equivalently: the constant function on F_inv(E) (uniform superposition over all positions). Equivalently: the β → ∞ limit of the substrate KMS state ρ_β with H = 3I − A.
- **Why this form.** Forced. The β → ∞ thermal limit projects onto the lowest-eigenvalue eigenspace of H = 3I − A, which is the highest-eigenvalue eigenspace of A. For srs (k = 3), this is the trivial Bloch eigenstate at all k.
- **Observer interpretation.** Substrate vacuum = "zero-information state" from observer's perspective: the observer has no positional information about toggle locations. Consistent with QFT's "no particles" reading.
- **Provenance.** Layer 5.34 (quantum partition function); `../forward_constructions/forward_construction_substrate_thermal_apparatus.md` §3.1.
- **Status.** **Grounded.** *One of the framework's most consequential ontology landings* — QFT has historically struggled to give a structural account of "what the vacuum is"; the substrate gives it as the unique zero-information eigenstate of the adjacency operator.

### Thermal density matrix ρ_β

- **Standard QFT.** Postulated for thermal field theory; ρ_β = e^{−βH}/Z(β); satisfies KMS condition.
- **Framework grounding.** **I-projection-with-energy-constraint on substrate von Neumann algebra L(F_inv(E)).** Specifically, ρ_β is the unique state on L(F_inv(E)) minimizing relative entropy S(ρ ‖ τ) under the energy constraint τ(ρ H) = const. Equivalently, the maximum-entropy state on L(F_inv(E)) at the given energy expectation. Per-Bloch-fiber: ρ_β = e^{−β(3I − A)}/Z(β), normalized 4×4 density matrix per srs primitive cell.
- **Why this form.** Maximum-entropy ↔ I-projection duality (Csiszár 1975 §3) applied to substrate vN algebra (Petz 2008 §11) with energy constraint via H_continuum.
- **Provenance.** Layers 5.34–5.35; `../forward_constructions/forward_construction_substrate_thermal_apparatus.md` §3.2; non-commutative I-projection via `../forward_constructions/forward_construction_noncommutative_iprojection.md`.
- **Status.** **Grounded.**

### von Neumann entropy / area-law entanglement entropy

- **Standard QFT.** S(ρ) = −Tr(ρ log ρ); central to entanglement entropy, holographic bounds, area-law for ground states. Postulated; standard formulas like Bekenstein-Hawking + Ryu-Takayanagi *use* but don't structurally derive these.
- **Framework grounding.** Substrate vN entropy on type II_1 trace: S(ρ_β) = β⟨H⟩_{ρ_β} + log Z(β) (standard thermodynamic identity). For substrate ground state (β → ∞): S(ρ_∞) = 0 (pure state). For Cayley-graph bipartition with reduced density ρ_A = Tr_B(ρ_β), the entanglement entropy S(ρ_A) is bounded by boundary degree-count (substrate area-law first-pass; rigorous version pending).
- **Why this form.** Gapped Hashimoto spectrum (substrate has Ramanujan gap) → finite correlation length → boundary-bound on entanglement entropy → area-law structure.
- **Provenance.** Layers 5.36, 5.38; `../forward_constructions/forward_construction_substrate_thermal_apparatus.md` §4.
- **Status.** **Grounded** at first-pass (rigorous area-law theorem is Tier 1 follow-up).

---

## 3 — Dynamics

### Schrödinger evolution: |ψ(t)⟩ = e^{−iHt}|ψ(0)⟩

- **Standard QFT.** Postulated as the time-evolution rule. The factor i in the exponent is conventional; H is taken to be Hermitian.
- **Framework grounding.** Stone's complex-form on substrate L²(F_inv(E); ℂ) gives U(t) = e^{−iHt} for unique self-adjoint H. The continuum-limit Hamiltonian H_continuum is derived from the substrate's discrete-time quantum walk via Strauch 2006 + Stage 3 rapid-decay closure.
- **Why this form.** Forced by §F + Stone (3.4) + §C continuum-limit closure (unitary part).
- **Provenance.** Layer 5.21, 3.4, 3.13.
- **Status.** **Grounded.** Schrödinger equation is derived from Stone's theorem applied to substrate continuum dynamics.

### One-parameter unitary group

- **Standard QFT.** Underlying mathematical structure of time evolution.
- **Framework grounding.** Strongly continuous one-parameter unitary group on substrate continuum L².
- **Why this form.** Stone (3.4); §C closure for the unitary part.
- **Provenance.** Layer 3.1, 3.2.
- **Status.** **Grounded.**

### Wick rotation: t → −iτ

- **Standard QFT.** Analytic continuation between Lorentzian and Euclidean signatures; underlies path-integral formulations and QFT on Euclidean lattices.
- **Framework grounding.** Substrate admits both Lorentzian (real-time unitary, U(t) = e^{−iHt}) and Euclidean (imaginary-time, e^{−Hτ}) continuum dynamics via Layer 3 Stone (real and complex forms) on appropriate sectors. The Wick rotation interpolates between them.

  **Refined finding.** The Euclidean evolution e^{−Hτ} *is* the substrate's heat kernel. Its supertrace gives the Atiyah-Singer index of the substrate Dirac operator (McKean-Singer formula). So substrate Wick-rotated dynamics + chirality grading = topological invariant via the index theorem (`../forward_constructions/forward_construction_substrate_atiyah_singer.md` §2.3).
- **Why this form.** Standard analytic-continuation duality between unitary and stochastic / heat-kernel evolution.
- **Provenance.** Layer 5.33; `proofs/foundations/theorem_feshbach_scalar_pairing.py`; refined via `../forward_constructions/forward_construction_substrate_atiyah_singer.md`.
- **Status.** **Grounded** for the unitary-evolution sector; partial for the path-integral interpretation pending the smooth-manifold continuum closure.

### Quantum partition function Z(β) = Tr(e^{−βH})

- **Standard QFT.** Postulated thermal-field-theory partition function. Underlies thermal QFT, KMS states, finite-temperature corrections.
- **Framework grounding.** Type II_1 trace τ on substrate vN algebra L(F_inv(E)) applied to e^{−βH_continuum}: Z(β) = τ(e^{−β(3I − A)}). Bloch-decomposed: Z(β) = ∫_BZ Z_k(β) d³k where Z_k(β) = Tr_{4×4}(e^{−β(3I − A(k))}) is the per-fiber partition function on srs primitive cell. At the P-point: Z_P(β) = 1 + 2e^{(√3 − 3)β} + e^{−4β} (first-pass).
- **Why this form.** Standard partition function definition adapted to substrate's type II_1 trace + Bloch decomposition.
- **Provenance.** Layer 5.34; `../forward_constructions/forward_construction_substrate_thermal_apparatus.md` §1–2.
- **Status.** **Grounded** at formalism level; full BZ-integration is Tier 1 follow-up.

### KMS condition / Tomita-Takesaki modular flow

- **Standard QFT.** KMS states characterize thermal equilibrium. Tomita-Takesaki modular flow σ_t^ρ(x) = ρ^{it} x ρ^{-it} is foundational for algebraic QFT (Haag-Kastler) and connects to Lorentz boosts via Bisognano-Wichmann theorem.
- **Framework grounding.** Substrate KMS state at β: ρ_β(x) = τ(e^{−βH} x)/τ(e^{−βH}). Modular flow at thermal state ρ_β coincides with substrate Hamiltonian flow generated by H_continuum (up to inverse-temperature scaling).
- **Why this form.** Standard KMS construction on substrate type-II_1 vN algebra; modular flow is determined by the state and the algebra.
- **Provenance.** Layer 5.34, A.7 (Appendix); `../forward_constructions/forward_construction_substrate_thermal_apparatus.md` §5.
- **Status.** **Grounded.** *Substrate Bisognano-Wichmann conjecture (modular flow ↔ Lorentz boost on substrate sub-region) is research-level Tier 3 follow-up.*

### Heisenberg picture

- **Standard QFT.** Operators evolve, states static. Mathematically equivalent to Schrödinger picture; preferred in canonical QFT.
- **Framework grounding.** Unused but more substrate-natural than Schrödinger: the substrate's primitive objects are toggles (operators), not states. The Heisenberg picture treats operators as primary, matching substrate ontology.
- **Why this form (if invoked).** Mathematical duality with Schrödinger.
- **Provenance.** Layer 5.22 (unused-deferred).
- **Status.** **Conceptually substrate-aligned but operationally unexplored.** No new predictions from this picture; framework's predictions are spectral and picture-independent.

---

## 4 — Symmetries

### Time-reversal symmetry T

- **Standard QFT.** Anti-unitary operator; for fermions T² = −1, for bosons T² = +1. T-violation is one of the discrete symmetry breakings.
- **Framework grounding.** The framework distinguishes *two* T-symmetries:
  1. **Graph-theoretic T-symmetry:** Bloch Hamiltonian satisfies d(−k) = d(k)*. Exact at substrate level (real-symmetric A on srs). Forces Berry curvature → 0 for photon Hodge bundle (c₁ = 0 in `predictions/c1_photon_bundle.py`).
  2. **Toggle-process T-symmetry:** broken because p_create = 1/2 ≠ p_destroy = 1/3.
  These are *independent*. Standard QFT does not separately track them.
- **Why this form.** Substrate has *two* time-reversal-related structures; only the toggle-process is broken. This is one of the framework's structural insights.
- **Provenance.** Layer 5.20, 5.18 (K), 5.19 (anti-unitary V); `predictions/c1_photon_bundle.py`, `predictions/eta_5_lorentz_dim5_derivation.md`.
- **Status.** **Grounded** — and provides a structurally cleaner T-symmetry account than standard QFT.

### Berry / geometric phases

- **Standard QFT.** Geometric phases on parameter loops; ground anomalies, Wilson loops, gauge-field topology. Postulated as a feature of parameter-dependent quantum systems.
- **Framework grounding.** Bloch-eigenvector holonomy on substrate Brillouin zone. CKM phases and δ_CP arise from substrate Berry phases on BZ closed paths.
- **Why this form.** Substrate is a periodic graph (srs); Bloch decomposition gives a vector bundle over the BZ; closed paths in BZ accumulate holonomy.
- **Provenance.** Layer 5.27; `proofs/flavor/srs_bloch_ckm.py`.
- **Status.** **Grounded.** Berry phases are substrate Bloch holonomy.

### Anti-unitary operations (general)

- **Standard QFT.** Wigner's theorem: QM symmetries are unitary or anti-unitary.
- **Framework grounding.** §F-induced complexification of real-Hilbert structure has canonical complex conjugation K (anti-linear); composition with unitary gives anti-unitary V.
- **Why this form.** Standard structural property of complexification.
- **Provenance.** Layer 5.18, 5.19.
- **Status.** **Grounded.**

---

## 5 — Gauge / GUT structure

### Killing-form gauge unification

- **Standard QFT.** Common normalization of gauge couplings at unification scale; sin²θ_W = 3/8 at M_GUT in canonical SU(5)-style unification.
- **Framework grounding.** Cl(6,0) bivectors are the so(6) Lie algebra; T_a all carry common Killing-form normalization inherited from the Clifford structure. sin²θ_W = 3/8 at M_unif derives via Path γ + B6 color-Z₃ multiplicity (theorem-grade in `../theorems/theorem_sin2_theta_W_unification.md`).
- **Why this form.** Cl(6;ℂ) bivectors share normalization; Killing-form unification at substrate Lie algebra.
- **Provenance.** Layer 4.43 (Killing form), 5.30 (Pati-Salam); `predictions/sin2_theta_W.py`.
- **Status.** **Grounded.**

### Pati-Salam embedding Spin(4) × Spin(2) ⊂ Spin(6)

- **Standard QFT.** GUT-scale embedding of SU(2)_L × SU(2)_R × U(1)_{B−L} into Pati-Salam SU(4) × SU(2) × SU(2). Standard in Pati-Salam GUT.
- **Framework grounding.** Subgroup embedding is determined by the substrate's spatial point-group symmetry: the srs primitive cell has S_4 ⊂ S_6 symmetry from the cubic 432 = O ≅ S_4 point group acting on Wyckoff 8a positions. This forces a matching-partition Cartan subalgebra in Cl(6,0), realizing the Pati-Salam embedding.
- **Why this form.** Substrate's spatial cubic point-group structure, inherited onto Cl(6) bivectors.
- **Provenance.** Layer 5.30; `proofs/gauge/k4_pati_salam_cl8.py`, `proofs/foundations/B3_chirality_bridge_derivation.md`.
- **Status.** **Grounded.** Pati-Salam embedding is derived from substrate spatial symmetry.

### SM gauge groups (SU(3)_c × SU(2)_L × U(1)_Y)

- **Standard QFT.** Postulated.
- **Framework grounding.** Subgroups of Spin(6) acting on the 8-dim Cl(6;ℂ) Dirac spinor at substrate node. SU(3)_c emerges from C₃ color-cyclic-shift symmetry; SU(2)_L × U(1)_Y from Pati-Salam restriction.
- **Why this form.** Spatial-symmetry-induced subgroup chain Spin(6) → Spin(4) × Spin(2) → SU(2)_L × SU(2)_R × U(1)_{B−L} → SU(2)_L × U(1)_Y; color from C₃.
- **Provenance.** Layer 5.28, 5.29, 5.30; multiple framework theorems.
- **Status.** **Grounded** (chain is theorem-grade across multiple workstream products).

### Lie algebra structure constants

- **Standard QFT.** Defining property [T_a, T_b] = if^c_{ab} T_c of gauge-algebra generators.
- **Framework grounding.** Standard Lie-algebra structure of Cl(6,0) bivectors and their Clifford-bracket commutators.
- **Why this form.** Cl(6;ℂ) → so(6) is the Clifford algebra's natural Lie algebra structure.
- **Provenance.** Layer 4.42; `proofs/foundations/theorem_B3_B6_reconciliation.py`, `proofs/foundations/K4_matchings_C3_check.py`.
- **Status.** **Grounded.**

### Complex characters / charge labels

- **Standard QFT.** Complex characters of compact gauge groups label charges (e.g., color charges, hypercharge).
- **Framework grounding.** C₃ characters {1, ω, ω²} label color components within one Pati-Salam family; complex weights of su(n) reps generalize.
- **Why this form.** C₃ on srs primitive cell at P-point has complex characters; 1-dim reps over ℂ.
- **Provenance.** Layer 5.31, 4.31 (character).
- **Status.** **Grounded.**

### Dirac operator D and the Dirac equation

- **Standard QFT.** Postulated first-order differential operator D = γ^μ ∂_μ (or with covariant derivative); fundamental for fermion dynamics; spectrum gives mass.
- **Framework grounding.** Substrate Dirac operator D_substrate = Σ_e γ^e ⊗ L_e on srs Cl(6;ℂ) spinor bundle: γ^e are the Cl(6;ℂ) generators, L_e is the substrate left-regular representation (Layer 1.6 / 2.13). D_substrate is self-adjoint, anti-commutes with chirality operator γ_5 = (−1)^F.
- **Why this form.** Forced by the substrate's spinor structure (Cl(6;ℂ) at each node, Layer 5.9) + Cayley-graph hopping. The Dirac form D = γ ⊗ ∂ emerges naturally as the *only* anti-chirality-commuting first-order operator built from substrate primitives.
- **Provenance.** `../forward_constructions/forward_construction_substrate_atiyah_singer.md` §1; uses Layers 5.6 (JW), 5.9 (spinor), 1.6 (left action).
- **Status.** **Grounded.**

### Atiyah-Singer index / chiral anomaly accounting

- **Standard QFT.** Atiyah-Singer index theorem connects spectral data of Dirac operator to topology. In QFT, the Dirac operator's index is the *axial anomaly coefficient* — the failure of classical chiral symmetry to be a quantum symmetry. Postulated structure with rich consequences (instanton effects, 't Hooft anomaly matching, anomaly cancellation conditions).
- **Framework grounding.** McKean-Singer heat-kernel formula on substrate: ind(D_substrate) = Tr_s(e^{−tD²}), independent of t. Per-Bloch-fiber: 32×32 matrix on srs primitive cell, finite-dim and computable. Per primitive cell: 16_L + 16_R = one Pati-Salam family (consistent with framework's existing chirality predictions).

  **Heat-kernel ↔ thermal connection.** ind(D) = Z_{S_+}(t) − Z_{S_-}(t) is the chirality-asymmetric difference of substrate thermal partition functions. McKean-Singer's t-independence is the substrate's analog of topological invariance of the index.
- **Why this form.** Standard McKean-Singer + supertrace adapted to substrate's finite-dim Bloch fibers.
- **Provenance.** `../forward_constructions/forward_construction_substrate_atiyah_singer.md` §2; uses A.4 (Appendix).
- **Status.** **Grounded** at formalism level; concrete numerical ind(D(P)) computation is Tier 1 follow-up.

---

## 6 — Statistical mechanics / information

### Shannon entropy / KL divergence / mutual information

- **Standard QFT.** Information-theoretic quantities used in subsystem analysis, thermal field theory, holographic entropy bounds.
- **Framework grounding.** Shannon entropy on substrate toggle distributions (`predictions/Q_Koide.py`); KL divergence on adjacent C₃ sectors (`proofs/foundations/srs_foundation_closure.py`); mutual information in A-IT axioms (`information_theoretic_stability_axioms.md`). All standard information-theoretic apparatus applied to substrate distributions.
- **Why this form.** Layer 4.B information-theory ops are standard; framework uses them as primitives.
- **Provenance.** Layer 4.5, 4.6, 4.7.
- **Status.** **Grounded** at the classical level. The *quantum* analogs (5.36 vN entropy, 5.38 entanglement entropy) are unused — see open gaps below.

### MDL apparatus / description length

- **Standard QFT.** Not standard in QFT; appears in information-theoretic formulations of physics.
- **Framework grounding.** Central. A2-T derives MDL canonicalization from finite-register Shannon source coding (`../theorems/theorem_A2_mdl_from_finite_register.md`). Description length is the framework's *primary* compression metric, replacing energy minimization in many contexts.

  **Refined finding (2026-04-26 forward construction):** A2-T's MDL canonicalization is the **information projection (I-projection)** of Csiszár 1975 — minimizing KL divergence onto a model family — not the standard L²-conditional expectation. Inherits Pythagorean theorem (Csiszár 1975 Thm 2.2), idempotence, generalized tower property under nested exponential families. **Stage 2a and A2-T are both I-projections** (different sources / constraint families) — structural unification of the framework's compression apparatus. See `../forward_constructions/forward_construction_a2t_as_iprojection.md`.
- **Why this form.** Forced by P1' + Shannon source coding.
- **Provenance.** Layer 4.8 (description length), 4.9 (source coding); A2-T; Csiszár 1975.
- **Status.** **Grounded** — with rigorous information-geometric foundation (Csiszár, Matuš, Amari-Nagaoka).

### Noether's theorem / time-translation conservation

- **Standard QFT.** Continuous time-translation invariance of action → conserved current J^μ with ∂_μ J^μ = 0 → conserved energy charge Q = ∫ J^0 d³x.
- **Framework grounding.** **Bayesian-posterior martingale** on substrate observation filtration. Under the no-coarse-graining filtration {𝒢_n} (more observations, no compression), the running posterior π_n(Q) over above-waterline models is a martingale (Williams 1991 §10.7); equivalently, posterior expectations of model functionals are conserved.
- **Why this form.** Statistical time-translation invariance of toggle Markov chain (constant p_create=1/2, p_destroy=1/3) → Doob's martingale theorem applied to A2-T's plural-retention Bayesian mixture.
- **Provenance.** Appendix A.15; `../forward_constructions/forward_construction_substrate_martingales.md`.
- **Status.** **Grounded** at the information-theoretic level. **Open gap:** identifying *specific* Noether currents (e.g., T^{μν}) requires explicit identification of model functionals with stress-energy components — pairs with Layer 6.23 ontology.

### H-theorem / second law

- **Standard physics.** Boltzmann's H is non-increasing under collision dynamics; entropy increase under coarse-graining; arrow of time.
- **Framework grounding.** **I-divergence sub-martingale** under coarse-graining. As the substrate's model family contracts (coarser σ-algebras), the irreducible compression cost D(P ‖ Q_n*) is non-decreasing (Csiszár-Matuš 2003 Cor 4 + data-processing inequality). This is structurally identical to H-theorem entropy increase.
- **Why this form.** Coarse-graining is information loss; I-projection theory makes this monotonicity a theorem.
- **Provenance.** Appendix A.15; `../forward_constructions/forward_construction_substrate_martingales.md`.
- **Status.** **Grounded** at the information-theoretic level. **Structural unification with Noether:** the framework grounds *both* Noether conservation AND the second law in the same I-projection / martingale apparatus, applied to opposite filtration directions (growing vs contracting). Standard physics treats these as distinct principles; the substrate makes them corollaries of one mathematical structure.

### Markov chain / toggle dynamics

- **Standard QFT.** Stochastic dynamics; not part of standard QFT but appears in stochastic quantization and Schwinger-Keldysh formalism.
- **Framework grounding.** Discrete-time Markov chain at substrate toggle level: each toggle event has p_create = 1/2, p_destroy = 1/3 from edge-surprise thresholds (Stage 2a, `../theorems/theorem_edge_surprise_thresholds.md`).
- **Why this form.** Forced by Bayesian Beta posterior + Jaynes maximum entropy on edge-occupancy priors.
- **Provenance.** Layer 4.11, Stage 2a.
- **Status.** **Grounded.**

### Modular-form / Hecke structure attached to substrate spectrum

- **Standard math.** Lubotzky-Phillips-Sarnak (1988) identified spectra of (p+1)-regular Ramanujan graphs with Hecke eigenvalues a_p of weight-2 cuspidal newforms on Γ_0(N) (Bruhat-Tits / Pizer-Brandt picture). Ramanujan-Petersson conjecture (Deligne 1974) bounds |a_p| ≤ 2√p with saturation characterizing certain Galois representations.
- **Framework grounding.** Substrate's k = 3 trivalent srs corresponds to p = 2. Ramanujan saturation \|h\|² = p = 2 holds (`../theorems/theorem_bloch_lift_mu.md`). Adjacency eigenvalue λ = √3 at the P-point gives Hecke eigenvalue **a_2 = √3** at p = 2. **M1 LMFDB lookup (2026-04-26 PM):** spectral match across a candidate set of weight-2 dim-2 newforms with Hecke field Q(√3) and a_2 = √3; smallest-level candidates `63.2.a.b`, `65.2.a.c`, `81.2.a.a`, `85.2.a.c`, `117.2.a.b`, `165.2.a.b`, `169.2.a.a` (extends to ~hundreds at higher levels).
- **Why this form.** Substrate's k = 3 + Ramanujan saturation forces a_p match at p = 2 under LPS framework.
- **Provenance.** Appendix A.16; `../forward_constructions/forward_construction_substrate_modular_structure.md`; an internal note.
- **Status.** **Partially grounded.** Spectral match confirmed at theorem-grade; **unique newform NOT identified** because Strong Multiplicity One requires a_p for almost all p, and substrate at fixed k = 3 only directly provides Hecke data at p = 2. Disambiguation deferred to Tier 2 Pizer-Brandt construction (substrate's primitive-cell structure → quaternion-algebra ramification set → unique level). 2–3 sessions estimated for unique closure. If closure positive, L(s, f) and Galois representation ρ_f become substrate invariants — chain to Langlands opens.

### BZJ scaling / Higgs vacuum value

- **Standard QFT.** Brézin–Zinn-Justin 1985: v ∝ N^{−1/4} for quartic O(n) potential at criticality. Used in lattice-field-theory derivations of Higgs vacuum value.
- **Framework grounding.** Substrate's N(t) toggle density connects to v_Higgs via BZJ scaling at criticality. Load-bearing for v_Higgs prediction.
- **Why this form.** Standard critical-phenomena scaling for quartic potentials; substrate's toggle dynamics exhibit this scaling.
- **Provenance.** Layer 4.51; `predictions/v_higgs.py`, `predictions/H_0.py`, `predictions/G_F.py`.
- **Status.** **Grounded.** BZJ scaling is invoked from the literature; substrate provides the N. The prediction of v_Higgs from substrate is the resulting structure.

### Mean-field theory / Curie-Weiss

- **Standard QFT.** Classical mean-field for spin systems / Higgs sector.
- **Framework grounding.** Mean-field treatment of Higgs sector via Cl(0,2) channels and Curie-Weiss-style fermion-chain structure (`predictions/v_higgs.py`, `predictions/G_F.py`, `../theorems/theorem_mdl_mean_field_higgs.md`).
- **Why this form.** Substrate's edge-local fermion structure admits mean-field reduction.
- **Provenance.** Layer 4.50, 4.53.
- **Status.** **Grounded.**

### Renormalization-group (RG) flow

- **Standard QFT.** Flow of couplings under scale change; central to QFT's predictive structure.
- **Framework grounding.** RG flow is *invoked* via standard MSSM RGE running for gauge couplings (`proofs/masses/mssm_rg_running.py`, `proofs/gauge/_mssm_rge.py`), but the framework does *not* derive RG flow from substrate. The substrate's "scale" structure (continuum-limit, BZJ scaling) implies a hidden RG structure, but it's not yet articulated as the explicit RG flow QFT uses.
- **Why this form.** Standard RGE machinery imported.
- **Provenance.** Layer 4.52.
- **Status.** **Partially grounded.** RG is invoked operationally but not derived from substrate. **Open gap:** derive QFT RG flow from substrate scale structure.

---

## 7 — Cosmology / GR

### FLRW universe and scale factor a(t)

- **Standard QFT/GR.** Postulated cosmological-principle-based metric ds² = −dt² + a(t)² dΣ_k². Imposed for homogeneous-isotropic cosmology.
- **Framework grounding.** FLRW form is imposed phenomenologically given substrate homogeneity (uniform N(t) toggle density across substrate) + isotropy (srs symmetry). The scale factor a(t) is a derived quantity from substrate's N(t) via BZJ scaling and continuum-limit bridge.
- **Why this form.** Cosmological principle is grounded in substrate's spatial uniformity; scale-factor evolution is grounded via N(t).
- **Provenance.** Layer 6.18, 6.22; `predictions/H_0.py`, `predictions/N_hub.py`, `proofs/cosmology/coasting_sn1a_comparison.py`.
- **Status.** **Partially grounded.** FLRW form is grounded in substrate symmetry; smooth-manifold tensor structure is partial (§C closure pending).

### Hubble parameter H_0

- **Standard QFT/GR.** Observable derived from a(t).
- **Framework grounding.** H_0 = 68.18 km/s/Mpc derived from G_F-anchored N_hub + Friedmann equation (theorem-grade in framework). One of the framework's strongest cosmology predictions.
- **Provenance.** Layer 6.21; `predictions/H_0.py`.
- **Status.** **Grounded.**

### Lorentzian signature (−,+,+,+)

- **Standard QFT.** Postulated as the spacetime signature; underlies causality.
- **Framework grounding.** Derived via Stage 3 (toggle 4-density correlations decay rapidly) + asymmetry between toggle-process T-broken (time direction, p_create ≠ p_destroy) and graph-T-symmetric (space directions, isotropic on srs).
- **Why this form.** Combination of Stage 3 rapid-decay + toggle-process T-asymmetry. Time picks up the broken-T direction; space picks up the symmetric directions.
- **Provenance.** Layer 6.10; `../theorems/theorem_lorentz_causal_sector.md`.
- **Status.** **Partially grounded.** Lorentz invariance at leading order is theorem-grade; full lattice-to-Lorentzian-manifold limit is research-level (§C).

### Causal structure / lightcones

- **Standard QFT/GR.** Postulated structure of spacetime that defines past/future and information flow.
- **Framework grounding.** Substrate is intrinsically causal — the multiway / non-backtracking-walk structure has a built-in causal partition (past / future via toggle history). No postulated speed limit; emergent causal structure from substrate topology.
- **Why this form.** A1 + reduced-word ordering (Serre 1980) give substrate a primitive causal structure.
- **Provenance.** Layer 6.24; `framework_architecture.md`.
- **Status.** **Grounded** — one of the framework's cleanest cosmology landings.

### Riemannian (spatial) metric

- **Standard QFT/GR.** Postulated metric for spatial slice.
- **Framework grounding.** Fisher information metric on substrate distributions (Čencov 1982 unique-up-to-scale theorem); Bloch-tangent metric γ_ab from Hashimoto-eigenvalue Hessian at high-symmetry point.
- **Why this form.** Information geometry on substrate distributions + spectral geometry on Bloch tangent space.
- **Provenance.** Layer 6.9; `predictions/d_spatial_derivation.md`, `predictions/srs_bloch_dispersion_gamma_derivation.md`.
- **Status.** **Grounded** at the spatial-metric level; full Lorentzian-spacetime metric tensor pending §C.

### Substrate Lichnerowicz formula + discrete curvature stack

- **Standard QFT/GR.** Riemann tensor R^a_{bcd}, Ricci R_{ab}, scalar curvature R; Lichnerowicz formula D² = ∇*∇ + R/4 for Dirac operator on Riemannian spin manifold. Postulated.
- **Framework grounding.** **Substrate Lichnerowicz formula** $D_{\text{sub}}^2 = n \cdot I + R_{\text{sub}}$ rigorous (theorem-grade 2026-04-26 PM, `../forward_constructions/forward_construction_substrate_lichnerowicz.md`), where $R_{\text{sub}} = \tfrac{1}{2}\sum_{e \neq e'} \gamma^e \gamma^{e'} \otimes [L_e, L_{e'}]$ is self-adjoint, mean-zero ($\tau(R_{\text{sub}}) = 0$), and has $\|R_{\text{sub}}\|_\tau^2 = n(n-1) = 30$ for srs ($n = 6$). Vanishes iff F_inv(E) is replaced by its abelianization. **Substrate Riemann tensor analog** $R^{ee'f}(g) := \delta_{g, ee'f} - \delta_{g, e'ef}$ defined as 3-index tensor on F_inv(E); Ricci analog $\text{Ric}^{ef}$ obtained by contraction.
- **Why this form.** $D_{\text{sub}}^2 = \sum_{e, e'} \gamma^e \gamma^{e'} \otimes L_e L_{e'}$ expansion + Cl anti-commutation + substrate non-commutativity. Lichnerowicz form emerges naturally; commutator structure of F_inv(E) plays the role of spacetime curvature.
- **Why operator-valued, not scalar.** Substrate is a *non-commutative geometry* (Connes 1994), not a Riemannian manifold. "Scalar curvature" is the operator $R_{\text{sub}}$ with non-trivial moments under the type II_1 trace; reading it as a single scalar $R(x)$ requires §C smooth-manifold closure (still open).
- **Provenance.** `../forward_constructions/forward_construction_substrate_atiyah_singer.md` §1.4, §4 (sketch); `../forward_constructions/forward_construction_substrate_lichnerowicz.md` (theorem-grade closure).
- **Status.** **Discrete-curvature stack grounded** at theorem grade — Lichnerowicz formula, scalar-curvature moments, Riemann tensor, Ricci tensor — *without* requiring §C smooth-manifold closure. Smooth-manifold continuum-limit identification of $R_{\text{sub}} \to R(x) \cdot I$ remains §C-open (Tier 3); the discrete substrate version is complete.

### Stress-energy tensor T_{ab}

- **Standard QFT/GR.** Source of gravitational field in Einstein equations; encodes matter content.
- **Framework grounding.** Components arise from substrate matter content (toggle-density energy density, p_destroy/p_create-asymmetry pressure, dark-correction). For Λ-domination: T_μν = −Λ g_μν giving w = −1.
- **Why this form.** Standard stress-energy decomposition; substrate provides the matter inputs.
- **Provenance.** Layer 6.23; `predictions/w_DE.py`.
- **Status.** **Grounded** at the cosmology level.

---

## 8 — Open gaps (no substrate grounding yet)

This section is the operator-sweep's most useful diagnostic: QFT-postulated objects that the framework does *not* yet ground in substrate primitives. Each is a forward-construction direction.

### Vacuum |0⟩ — ✅ GROUNDED 2026-04-26 (moved to §2)

Now grounded as substrate's maximally-symmetric Bloch-trivial eigenstate of A. See §2 entry above.

### Field operator ψ(g) — substrate fermionic field

- **Standard QFT.** Operator-valued distribution at spacetime point x; underlies all of QFT.
- **Framework grounding (fermionic, this work).** $\psi(g) = (1/\sqrt{V}) \sum_{\alpha, k} u_\alpha(k, r) e^{ik \cdot R} c_{\alpha, k}$ at substrate vertex $g = (R, r)$, with $c_{\alpha, k}$ the Bloch-mode CAR operators (hybrid B+C synthesis: Bloch decomposition + JW/CAR creation/annihilation). Per `../forward_constructions/forward_construction_substrate_propagator.md` §1.3, §2.1.
- **Why this form.** Bloch decomposition of substrate Dirac D_sub is canonical; JW/CAR provide the discrete-substrate creation/annihilation; combining gives the continuum-QFT-like mode expansion at substrate-discrete level.
- **Provenance.** `../forward_constructions/forward_construction_field_operator_phi_x.md` (setup); `../forward_constructions/forward_construction_substrate_propagator.md` (theorem-grade for the fermionic case + propagator).
- **Status.** **Fermionic case grounded.** Bosonic field operator (Candidate A smeared toggle-density) status open — bridge to Boson Field Grounding workstream. Continuum-limit identification with continuum φ(x) requires §C smooth-manifold closure (still open).

### Wightman 2-point function / Feynman propagator (fermionic)

- **Standard QFT.** $W(x, y) = \langle 0 | \psi(x) \bar\psi(y) | 0\rangle$ Wightman, $G_F(x-y) = \langle 0 | T \psi(x) \bar\psi(y) | 0 \rangle$ Feynman; central computational tool of perturbative QFT.
- **Framework grounding.** $\tilde G_F^{\text{sub}}(k, \omega) = i(\omega + D(k))/(\omega^2 - D(k)^2 + i\varepsilon)$ in closed form at substrate level; using G2 Lichnerowicz $D(k)^2 = n \cdot I + R_{\text{sub}}(k)$, $\tilde G_F^{\text{sub}} = i(\omega + D(k))/(\omega^2 - n - R_{\text{sub}}(k) + i\varepsilon)$. **Substrate intrinsic mass scale $n = |E| = 6$** is Planckian; SM-fermion masses are A5-mass-labeled separately on top.
- **Why this form.** Standard Bloch + Feynman pole prescription on substrate-rigorous fermion field; closed-form follows from chiral structure of D_sub.
- **Provenance.** `../forward_constructions/forward_construction_substrate_propagator.md` Theorems 2.1, 3.2.
- **Status.** **Grounded** at substrate-discrete level. Continuum-QFT limit requires §C; substrate version is rigorous on its own. F2–F7 cascade (Wick, LSZ, S-matrix, Feynman rules, RG) unblocked as concrete follow-up.

### Wick's theorem / n-point functions / Feynman diagrams (fermionic)

- **Standard QFT.** Wick's theorem (Wick 1950) reduces $\langle 0 | T(\psi_1 \cdots \psi_n) | 0 \rangle$ to a signed sum over pair-contractions of the propagator. Generates Feynman-diagram graphical perturbation expansion.
- **Framework grounding.** **Substrate Wick theorem** theorem-grade given F1 propagator + CAR + bilinear free $H$ + Dirac-sea vacuum. Standard Wick induction transposes verbatim. n-point functions $= \sum_\sigma \mathrm{sgn}(\sigma) \prod_k G_F^{\text{sub}}(g_k, g'_{\sigma(k)};\, t_k - t'_{\sigma(k)})$ in Bloch-vertex basis. Per `../forward_constructions/forward_construction_substrate_wick.md` Theorem 3.1 + Corollary 3.2.
- **Why this form.** CAR + bilinear free Hamiltonian + Dirac-sea vacuum together imply the standard Wick proof transposes; no novel machinery needed.
- **Provenance.** `../forward_constructions/forward_construction_substrate_wick.md` Theorem 3.1.
- **Status.** **Grounded (free theory).** Interaction vertices via Dyson expansion; specific substrate vertex enumeration is F5/F6 follow-up.

### Path integrals

- **Standard QFT.** Sum over field histories weighted by e^{iS[φ]}; equivalent to canonical quantization for many purposes.
- **Framework status.** Partially grounded via Wick rotation (5.33) — Wick-rotated substrate dynamics is heat-kernel-like, suggesting Euclidean path-integral grounding. Lorentzian path-integral form derivable from substrate Feynman propagator + Wick (now grounded); concrete Euclidean closure is a 1–2 session bounded follow-up.
- **Forward construction.** Most ingredients now present (F1 propagator + F3 Wick); explicit path-integral construction tractable.
- **Tier:** 2 — most concrete remaining gap is the Euclidean-form construction.

### BRST / gauge fixing

- **Standard QFT.** BRST cohomology underlies gauge-invariant quantization. Postulated framework for handling gauge redundancy.
- **Framework status.** Entirely absent. Framework's gauge-invariance arrives via Killing-form unification + Pati-Salam embedding, but the *cohomological* machinery of BRST has no substrate analog yet.
- **Forward construction.** Could potentially connect to A.1 group cohomology of F_inv(E) — but the first-pass cohomology output (uniform 6-fold ℤ/2) didn't naturally split into BRST structure.
- **Tier:** 3 (research-level).

### Renormalization derivation from substrate

- **Standard QFT.** RG flow of couplings; Wilsonian effective field theory.
- **Framework status.** RG is *invoked* operationally (MSSM RGE) but not derived from substrate. The substrate's continuum-limit + BZJ scaling implies a hidden RG, but the connection is not articulated.
- **Forward construction.** A.4 Atiyah-Singer index could connect; A.16 modular forms could connect via L-function functional equations. Substantial open direction.
- **Tier:** 2 — high-value, mathematically heavy.

### Einstein equations / GR dynamics

- **Standard QFT/GR.** G_{ab} + Λ g_{ab} = 8πG T_{ab}; the dynamical equation of gravity.
- **Framework status.** Imposed at Friedmann level; not derived from substrate at framework rigor. The framework's scoping doc an internal working note discusses Gorard-emergent-Einstein direction (Route S3.A) as research-level.
- **Forward construction.** Pairs with §C smooth-manifold closure. Most prominent ontology gap in the GR sector.
- **Tier:** 3 (research-level; multi-session).

### Full smooth-manifold continuum limit

- **Standard QFT/GR.** Spacetime is a smooth Lorentzian manifold; basis of all classical-fields formulations.
- **Framework status.** §C closure is partial. Unitary-evolution continuum is closed at journal grade; smooth-manifold continuum is not.
- **Forward construction.** Causal-set-theory / Gorard 2020 direction. Research-level.
- **Tier:** 3.

### Quantum thermal / KMS / area law / holographic entropy — ✅ GROUNDED 2026-04-26 (moved to §2, §3, §6)

Tier 1 program completed. Z(β), ρ_β, vN entropy, KMS condition, modular flow, area-law (first-pass) all grounded. Holographic-entropy bound (Bekenstein-Hawking, Ryu-Takayanagi) remains research-level; substrate area-law is consistent.

### Specific SM mass / mixing labels (A5-mass downstream)

- **Standard QFT.** Yukawa couplings, masses, mixing angles are inputs.
- **Framework status.** Partial. Some specific values are derived (Q_Koide = 2/3, sin²θ_W = 3/8 at M_unif, V_cb = 256/6305, V_us = 9/40, y_τ, dark correction 5/12, etc.). Many remain at A5-mass labeling stage (post-substrate identification of which spectral component is which SM observable).
- **Forward construction.** Active ongoing workstream; not specifically a "gap" in the QFT-ontology sense — these are predictions, not postulates.

---

## Tier 1 forward-construction program: substrate quantum-information

The operator-sweep's strongest single search-instrument finding: **a 14-op cluster** that could ground QFT's full quantum-information apparatus (KMS, vN entropy, entanglement entropy, area law, holographic bounds).

**Ops in the cluster:**
- §5.34 Quantum partition function Z(β) = Tr(e^{−βH})
- §5.35 Thermal density matrix ρ(β) = Z⁻¹ e^{−βH}
- §5.36 von Neumann entropy S(ρ) = −Tr(ρ log ρ)
- §5.37 Schmidt rank of bipartite pure state
- §5.38 Entanglement entropy
- A.7 KMS states on C*_red(F_inv(E))
- §4.25 Conditional expectation (cross-validation of A2-T)
- A.15 Martingales (substrate analog of conserved currents)
- §5.16 Schmidt decomposition (paired with 5.37)
- A.5 Reduced group C*-algebra (operator-algebraic lens)
- A.6 Group von Neumann algebra L(F_inv(E)) (type II_1 factor)
- A.8 Free convolution (free-probability lens)
- A.9 Free entropy / Fisher information (free analog of Shannon)
- A.4 Atiyah-Singer index (chirality / fermion-anomaly grounding)

**Estimated:** ~3–5 focused sessions for first-pass results across the cluster. High potential payoff: would simultaneously address vacuum, thermal, entanglement, area-law, and KMS gaps.

---

## Honest assessment

### Successfully grounded by the framework

QFT-postulated objects with substrate explanation:
- Algebraic: CAR, Pauli, Dirac spinor, fermion-number, Majorana, Hermiticity, anti-Hermitian gauge generators, **Dirac operator D**
- State-space: complex Hilbert, density matrix, partial trace, purification, multi-particle Fock, **vacuum |0⟩**, **thermal density matrix ρ_β**, **vN entropy + area-law (first-pass)**
- Dynamics: Schrödinger evolution, one-parameter unitary group, Wick rotation (refined via heat-kernel/index theorem), **quantum partition function Z(β)**, **KMS condition / Tomita-Takesaki modular flow**
- Symmetries: T-symmetry (cleaner two-symmetry account), Berry phases, anti-unitary
- Gauge: Killing-form unification, Pati-Salam embedding, SM gauge groups, structure constants, complex characters, **Atiyah-Singer index / chiral anomaly accounting**
- Information: Shannon, KL, MDL (refined: I-projection of Csiszár 1975), Markov dynamics, BZJ scaling, mean-field, Curie-Weiss, **Noether/time-translation conservation (Bayesian-posterior martingale)**, **H-theorem/second law (I-divergence sub-martingale)**
- Cosmology / GR: FLRW (partial), Hubble H_0, Lorentzian (partial), causal structure, Riemannian spatial metric, T_μν, **substrate Lichnerowicz formula + Riemann tensor + Ricci tensor + scalar-curvature moments (theorem-grade discrete-curvature stack 2026-04-26 PM)**
- Field theory (substrate-discrete level): **substrate fermionic field operator ψ(g)**, **Wightman 2-point function**, **substrate Feynman propagator $G_F^{\text{sub}}(k, \omega)$ in closed form** (F1 2026-04-26 PM); **Wick's theorem / n-point functions / Feynman-diagram structure** (F3 2026-04-26 PM)

**Total grounded:** ~46 distinct QFT objects (+10 Tier 1 program 2026-04-26 AM/PM, +3 G2 closure 2026-04-26 PM, +3 F1 closure 2026-04-26 PM, +3 F3 closure 2026-04-26 PM: Wick's theorem, n-point functions, Feynman-diagram structure).

### Still open

- ~~Vacuum |0⟩~~ (grounded 2026-04-26), ~~field operator (fermionic)~~ + ~~Wightman 2-point~~ + ~~Feynman propagator~~ (F1 grounded), ~~Wick / n-point / time-ordered products~~ (F3 grounded); **Bosonic field operator** still open (separate workstream)
- Path integrals (Euclidean form follow-up tractable; Lorentzian needs §C) — partial
- BRST / gauge fixing
- Renormalization derivation from substrate (most ingredients present after F1 + F3; F7 in cascade)
- Einstein equations, full smooth-manifold limit
- ~~Quantum thermal / KMS / vN~~ (Tier 1 grounded); area law (rigorous) still partial

**Total open:** ~5 distinct major QFT objects (down from ~10 before 2026-04-26 forward-construction sessions).

### Headline

The framework grounds *most of QFT's algebraic structure* (fermion algebra, gauge structure, state-space) and *most of its cosmology landings* (FLRW, Hubble, causal structure). It does *not* yet ground the *quantum-information apparatus* (KMS / thermal / entanglement) or the *deep dynamical structure* (Einstein equations, full RG, BRST).

The operator-sweep's diagnostic value is high: it identifies which gaps are tractable (Tier 1: quantum thermal cluster), which are research-level (Tier 3: smooth-manifold limit, Einstein equations), and which require structural extension beyond the current catalog (BRST cohomology).

---

## Cross-references

- Source catalog: `../operator_sweep/operator_sweep_from_A1.md`
- Per-layer audits: `../operator_sweep/operator_sweep_audit_layer_0_1.md` through `_6.md`, plus `_appendix.md`
- Methodology: an internal note
- Foundational theorems: `../theorems/theorem_A2_mdl_from_finite_register.md`, `../theorems/theorem_A3_complex_hilbert_from_multiway.md`, `../theorems/theorem_car_local_jordan_wigner.md`, `../theorems/theorem_lorentz_causal_sector.md`, `../theorems/theorem_bloch_lift_mu.md`, `../theorems/theorem_sin2_theta_W_unification.md`
- Open-problems scoping: an internal working note

---

## Status

Meta-doc complete. Three remaining workstream items:
1. **Backfill ontology lens into Layers 0–4 audits** — predecessor audits used two-lens entries; ontology grounding lines need to be added retroactively.
2. **Tier 1 forward-construction program** — substrate quantum-information cluster (14 ops; ~3–5 sessions).
3. **Tier 2/3 follow-ups** as time and priority allow.
