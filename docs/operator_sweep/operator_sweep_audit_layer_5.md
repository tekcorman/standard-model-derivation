# Operator Sweep Audit — Layer 5

**Date:** 2026-04-26.
**Status:** Per-operation audit with **three-lens** entries (audit + search-instrument + ontological grounding). Layer-by-layer execution of the operation-constructor workstream; recalibrated this session to add the ontology lens after the user identified ontological landing as the deeper goal.
**Source catalog:** `operator_sweep_from_A1.md` §Layer 5.
**Predecessors:** `operator_sweep_audit_layer_0_1.md`, `_2.md`, `_3.md`, `_4.md` (audit-only; ontological grounding to be backfilled).

## Three-lens methodology

For each op:
- **Audit** — invoked? where? (supports uniqueness of existing derivations; coverage proof)
- **Search-instrument** — for unused ops, what does first-order application produce? (identifies forward-construction candidates)
- **Ontological grounding** — what is the op IN THE SUBSTRATE? Why does it take this specific form (forced by what)? Which QFT-postulated operator does it inform? (fills standard QFT's "operator without explanation" gap)

The third lens is the deepest deliverable: it converts the catalog from a coverage proof into a structural account of why each operation looks the way it does, given A1 + P1' + downstream theorems.

---

## 5.A — Complex algebraic structures requiring i (5 ops)

| # | Op | Verdict | Citation | Ontological grounding |
|---|---|---|---|---|
| 5.1 | Imaginary unit i in operator algebra | invoked-direct | Pervasive post-§F; load-bearing in `../theorems/theorem_A3_complex_hilbert_from_multiway.md`. | **Substrate:** i enters the operator algebra only after §F field-selection. **Why this form:** P1' + register-storable spectrum forces ℂ (real-Hilbert generator B is skew-symmetric with imaginary spectrum, incompatible with finite-register reading; ℂ-Hilbert generator H is self-adjoint with real spectrum). i is the structurally-forced complexification, not a postulated convenience. **QFT ground:** i in Schrödinger eq, complex amplitudes, gauge phases — all derived from §F, not postulated. |
| 5.2 | Pauli σ^x, σ^y, σ^z (σ^y requires i) | invoked-direct | `predictions/G2_cl2_channels.py`; `predictions/theta_23_PMNS.py`; `predictions/theorem_B3_spinor_fermion.py`. | **Substrate:** edge-qubit operators on undirected edge in K_4-quotient. σ^x flips edge state (toggle action); σ^z gives signed parity. **Why this form:** the two directed versions of an undirected edge {u,v} carry T_1, T_2 with T_1² = T_2² = id and {T_1, T_2} = 0 — Cl(0,2) algebra → Pauli matrices on ℂ². σ^y = iσ^xσ^z exists only after §F. **QFT ground:** Pauli matrices in spin/qubit/Dirac operators are the substrate's edge-direction algebra after field selection. |
| 5.3 | Hermitian operators with complex matrix entries | invoked-direct | Self-adjoint H on ℂ-L²; pervasive. | **Substrate:** observables on observer's complex Hilbert space. **Why this form:** Stone complex-form (3.4) requires self-adjoint H to generate unitary U(t) = exp(−iHt). **QFT ground:** the postulated "Hermitian observables" of QM — derived as the §F-forced spectral content. |
| 5.4 | Anti-Hermitian operators (A* = −A) | invoked-direct | Generators of unitary subgroups; Lie-algebra elements (Layer 4.40). | **Substrate:** elements of so(n;ℂ), su(n) — derivatives at identity of unitary group elements. **Why this form:** d/dt[U(t)]_{t=0} of a unitary is anti-Hermitian; structurally tied to one-parameter subgroups. **QFT ground:** gauge generators iA_μ in QED/QCD; their algebra closes via structure constants (4.42). |
| 5.5 | Spectral decomposition with real eigenvalues, complex eigenvectors | invoked-direct | `predictions/h_walker_eigenvalue.py`; Bloch-mode analysis throughout. | **Substrate:** eigenstates of self-adjoint A or H on ℂ-L²(F_inv(E)). Eigenvalues real (A is Hermitian on srs); eigenvectors complex (Bloch phases at non-zero k). **Why this form:** spectral theorem on ℂ-L². **QFT ground:** energy eigenstates of QM with momentum-space phase structure. |

**5.A totals:** 5/5 invoked-direct.

---

## 5.B — Clifford algebras and Jordan-Wigner (6 ops)

This sub-section is **the densest ontological territory** in the catalog: the algebraic structure of QFT fermions is grounded here, not postulated.

| # | Op | Verdict | Citation | Ontological grounding |
|---|---|---|---|---|
| 5.6 | Jordan-Wigner construction c_j = (σ^z_1 ⊗ … ⊗ σ^z_{j-1}) ⊗ σ^-_j | invoked-direct | `../theorems/theorem_car_local_jordan_wigner.md` (foundational); `predictions/koide_quark_ratio.py`; `predictions/G2_cl2_channels.py`. | **Substrate:** transforms involutive toggle/spin operators on a chain into anti-commuting fermion operators via the σ^z-string. **Why this form:** A1's involutivity (T_e² = id) gives σ^x-like generators per edge; JW is the unique transform converting commuting-on-different-sites involutions into anti-commuting fermions. **QFT ground:** the CAR algebra of fermion creation/annihilation operators is *forced*, not postulated. The σ^z-string is the substrate's parity bookkeeping. |
| 5.7 | CAR: {c_i, c_j†} = δ_{ij}, {c_i, c_j} = 0 | invoked-direct | Direct corollary of 5.6 in `../theorems/theorem_car_local_jordan_wigner.md`. | **Substrate:** anti-commutation relations follow from JW on involutive substrate generators. **Why this form:** combinatorics of σ^z-string overlaps; algebraic identity, not a separate postulate. **QFT ground:** the defining algebraic relation of QFT fermion operators. The framework explains *why* fermions anti-commute: their substrate primitives are involutive. |
| 5.8 | Complex Clifford algebra Cl(n; ℂ) | invoked-direct | `predictions/k_star.py`; `predictions/delta_CP_CKM_geometry.py`; `proofs/gauge/k4_pati_salam_cl8.py`. | **Substrate:** local Clifford algebra at trivalent node from involutive edge generators. **Why this form:** k = 3 incident edges at each substrate node; products of involutive generators define Cl(6;ℂ) on the K_4-quotient (each undirected edge contributes 2 directions × 3 undirected edges = 6 generators). **QFT ground:** the Cl(6;ℂ) → Spin(6) → SU(4) → Pati-Salam chain that gives SM gauge structure is grounded in substrate node-locality. |
| 5.9 | Spinor representations of Cl(n; ℂ) | invoked-direct | `predictions/theorem_B3_spinor_fermion.py` (8-dim Cl(6;ℂ) Dirac spinor); `proofs/foundations/B3_chirality_bridge.py`. | **Substrate:** 8-dim complex Dirac spinor at each substrate node. **Why this form:** irreducible Cl(6;ℂ)-rep has dim 2³ = 8 (Lawson-Michelsohn). **QFT ground:** Dirac spinor of QFT — its 8-component complex structure derives from substrate node having 6 generators (3 undirected × 2 directions). |
| 5.10 | ℤ/2-grading by fermionic parity (−1)^F | invoked-direct | Used in JW construction (5.6); cited in CAR derivation. | **Substrate:** parity of toggle-count, equivalently the σ^z-string product. **Why this form:** JW maps σ^z-products to fermion-number-parity operator. **QFT ground:** fermion-number conservation in QFT — derived as parity of substrate toggle count, not postulated. |
| 5.11 | Majorana operators γ_{2j-1} = c_j + c_j†, γ_{2j} = i(c_j† − c_j) | invoked-direct | `predictions/G2_cl2_channels.py` ("T_1 = σ^x ⊗ I (Majorana mode 1); T_2 = σ^z ⊗ σ^x (JW string for mode 2)"); `proofs/flavor/srs_dcp_exponent.py`. | **Substrate:** Hermitian combinations of c, c† — directly the directed-edge generators T_1, T_2 of an undirected edge (before JW dressing). **Why this form:** the substrate's primary objects are involutive (Hermitian, satisfying T² = id), so Majorana is the *natural* fermionic basis; CAR ops c, c† are *derived* via complex combinations. **QFT ground:** Majorana fermions are the substrate's primary fermion ontology; Dirac fermions c, c† are downstream complexifications. The framework *inverts* the standard QFT priority where Dirac is primary. |

**5.B totals:** 6/6 invoked-direct. **Highest-density ontological grounding in the catalog so far.**

---

## 5.C — Density matrices and quantum states (6 ops)

| # | Op | Verdict | Citation | Ontological grounding |
|---|---|---|---|---|
| 5.12 | Density matrix ρ (positive, self-adjoint, trace 1) | invoked-direct | `predictions/observer_hilbert_space.py`; A3-T derivations. | **Substrate:** observer's compressed state derived from purification + partial trace over auxiliary. Mixed states are not postulated; they emerge from compression. **Why this form:** A3-T's CDP-2011 purification axiom guarantees every ρ_A = Tr_B(\|ψ⟩⟨ψ\|_{AB}). **QFT ground:** density matrices in QM/QFT are derived from substrate-observer compression, filling the standard "where do mixed states come from?" gap. |
| 5.13 | Pure vs mixed state distinction | invoked-direct | A3-T context; `predictions/observer_hilbert_space.py`. | **Substrate:** pure = uncompressed pre-canonicalization state; mixed = post-canonicalization observer state. **Why this form:** A2-T's MDL canonicalization is the compression operation that produces mixedness. **QFT ground:** the QM distinction "pure state of full system, mixed state of subsystem" is grounded in substrate compression structure. |
| 5.14 | Partial trace ρ_A = Tr_B(ρ_AB) | invoked-direct | `predictions/m_tau_derivation.md`, `predictions/y_tau_derivation.md`, `predictions/R3_observer_c3_generation_derivation.md` (A3-T realization of MDL canonicalization as quantum partial trace). | **Substrate:** *the* compression operation — observer integrates out auxiliary degrees of freedom. **Why this form:** A3-T identifies MDL canonicalization with partial trace over the abstract auxiliary purifying space. **QFT ground:** open-quantum-system reduced density operator; here derived from compression principle, not postulated. |
| 5.15 | Purification: every mixed ρ_A is partial trace of pure \|ψ⟩_AB | invoked-direct | `predictions/observer_hilbert_space_derivation.md` (CDP 2011 axiom 5). | **Substrate:** the abstract auxiliary B is the "rest of the substrate not currently in the observer's compression". **Why this form:** CDP 2011 quantum-info axiom; structural property of compression. **QFT ground:** Stinespring dilation in quantum info; entanglement-with-environment in QFT. |
| 5.16 | Schmidt decomposition \|ψ⟩_AB = Σ √λ_i \|i⟩_A ⊗ \|i⟩_B | unused-deferred | See application §5.16 below. | **Substrate:** canonical bipartite-pure form for the observer-auxiliary purified state. **Why this form (if invoked):** SVD applied to bipartite pure state. **QFT ground:** the canonical entanglement structure between subsystems; substrate's compressibility imposes a specific Schmidt structure that isn't yet exploited. |
| 5.17 | Quantum tensor products with entanglement | invoked-direct | `predictions/observer_hilbert_space.py`; `../theorems/theorem_car_local_jordan_wigner.md` (JW tensor structure). | **Substrate:** multi-edge / multi-node states as tensor products of local Hilbert spaces. **Why this form:** local Cl(2) factors per edge × node count via JW. **QFT ground:** multi-particle Fock space; the "tensor structure of identical particles" — derived from substrate locality. |

**5.C totals:** 5/6 invoked-direct, 1/6 unused-deferred.

### §5.16 — Schmidt decomposition (application sketch)

**Audit.** The framework uses purification (5.15) and partial trace (5.14) but does not explicitly use Schmidt decomposition.

**Search-instrument.** Schmidt rank r of the observer-auxiliary purified state |ψ⟩_AB measures the correlation dimensionality. For mass-content predictions cited via A3-T (m_τ, y_τ, etc.), the Schmidt rank of the purification is implicit but never extracted as a quantity. Computing it could:
- Identify the *minimum auxiliary dimension* needed for the framework's predictions (a low-MDL constraint).
- Produce an integer invariant tied to the framework's compression structure (potentially comparable to mass-content multiplicities).

**Ontological grounding.** **Substrate:** Schmidt rank = number of independent observer-auxiliary correlation channels. **Why this form:** SVD on bipartite pure state, basis-independent. **QFT ground:** entanglement structure; appears in entanglement entropy, holographic bounds, area law. The framework's substrate could ground a specific Schmidt-rank value for the observer-substrate correlation, filling the gap "what determines entanglement entropy of substrate sectors?"

**Verdict.** unused-deferred. Worth focused investigation: compute Schmidt rank of the A3-T purification for representative predictions; check whether the rank is a structurally-determined integer.

---

## 5.D — Anti-unitary operations (3 ops)

| # | Op | Verdict | Citation | Ontological grounding |
|---|---|---|---|---|
| 5.18 | Complex conjugation operator K (anti-linear) | invoked-direct | `predictions/c1_photon_bundle.py` (time-reversal symmetry d(−k) = d(k)*); `predictions/eta_5_lorentz_dim5.py`. | **Substrate:** the anti-linear "flip" of the §F-induced complexification. **Why this form:** complexification of real-Hilbert structure has a canonical complex conjugation; anti-linear by construction. **QFT ground:** complex-conjugation operator underlying T-symmetry in QM. |
| 5.19 | Anti-unitary V (anti-linear, V*V = I) | invoked-direct | Cited in T-symmetry analyses (c1_photon_bundle, eta_5). | **Substrate:** composition of K with a unitary; symmetries of substrate Hamiltonian that flip ℂ. **Why this form:** Wigner's theorem on QM symmetries — symmetries are unitary OR anti-unitary. **QFT ground:** anti-unitary representations of T, CT in particle physics. |
| 5.20 | Time-reversal symmetry | invoked-direct | `predictions/c1_photon_bundle.py` (d(−k) = d(k)* forces Berry curvature to vanish, c₁ = 0); `predictions/eta_5_lorentz_dim5_derivation.md`. | **Substrate:** graph-theoretic symmetry of real adjacency matrix → Bloch Hamiltonian satisfies d(−k) = d(k)*. **Why this form:** real-symmetric A on srs (Hermitian, real entries) → Bloch operator satisfies the complex-conjugation symmetry. **QFT ground:** T-symmetry in QFT; its breaking (toggle-process p_create ≠ p_destroy) is independent of graph T-symmetry. |

**5.D totals:** 3/3 invoked-direct.

**Ontology meta-finding.** The framework distinguishes *graph-theoretic* T-symmetry (5.20, exact) from *toggle-process* T-symmetry (broken because p_create = 1/2 ≠ p_destroy = 1/3). This distinction is not standard in QFT and is one of the framework's structural insights — the substrate has *two* time-reversal-related symmetries, only one of which is broken.

---

## 5.E — Schrödinger-picture quantum dynamics (4 ops)

| # | Op | Verdict | Citation | Ontological grounding |
|---|---|---|---|---|
| 5.21 | Schrödinger evolution \|ψ(t)⟩ = e^{−iHt}\|ψ(0)⟩ | invoked-direct | Stone-derived continuum dynamics (Layer 3); applied to ℂ-Hilbert observer states. | **Substrate:** observer's state evolves under the §F-forced complex Hamiltonian. **Why this form:** Stone complex-form (3.4) on ℂ-L² gives U(t) = exp(−iHt) for the substrate's continuum-limit Hamiltonian. **QFT ground:** Schrödinger equation — derived, not postulated. |
| 5.22 | Heisenberg picture O(t) = e^{iHt} O e^{−iHt} | unused-deferred | See application §5.22 below. | **Substrate:** if invoked, would have toggles T_e (the substrate's primary objects, which ARE operators) evolving in time. **Why this form (if invoked):** dual picture to Schrödinger; mathematically equivalent. **QFT ground:** Heisenberg picture of QFT operator dynamics; especially natural in canonical QFT. |
| 5.23 | Interaction picture | unused-applied-negative | See application §5.23 below. | **Substrate:** would require an H_0/V split of the substrate Hamiltonian. **Why this form (would require):** decomposition into "free" + "interacting" parts. **QFT ground:** Dyson-series perturbation expansion; substrate has no natural H_0/V decomposition. |
| 5.24 | Time-dependent perturbation theory | unused-applied-negative | See application §5.24 below. | **Substrate:** corrections to evolution from a small V over finite time. **Why this form (would require):** small-parameter expansion in time. **QFT ground:** transition amplitudes via Dyson series; substrate uses *static* perturbation theory (Sakurai §5.2 degenerate PT) at the spectral level instead — the framework's "perturbation" lives in compressibility-space (waterline corrections), not in time-evolution. |

### §5.22 — Heisenberg picture (application sketch)

**Audit.** Not directly invoked; framework's predictions come from spectral content (eigenvalues of fixed operators), not from operator time-evolution.

**Search-instrument.** First-order: Heisenberg-picture evolution of the toggle operators T_e(t) = e^{iHt} T_e e^{−iHt} would expose how toggles propagate in continuum time. For srs at the P-point, the relevant generator is the adjacency operator A = Σ L_e; the Heisenberg evolution gives T_e(t) as a linear combination of generators with k-dependent phases. This is dual to the Bloch decomposition the framework already uses (4.17).

**Ontological grounding.** **Substrate:** in the substrate, toggles ARE the primary operators — the Heisenberg picture is *more natural* than Schrödinger because operators are the substrate's primitives, not states. **Why this form:** mathematical duality with Schrödinger; equivalent observable predictions. **QFT ground:** Heisenberg picture is preferred in canonical QFT; the framework provides ontological backing for that preference (substrate primitives are operators, not states).

**Verdict.** unused-deferred. The Heisenberg picture is conceptually more substrate-aligned than Schrödinger but produces no new predictions — the spectral content the framework extracts is picture-independent.

### §5.23 — Interaction picture (application sketch)

**Audit.** Not invoked.

**Search-instrument.** Requires H = H_0 + V split. The substrate's continuum-limit Hamiltonian is the Hashimoto-graph operator on F_inv(E) — there's no natural "free + interacting" split. The framework's "perturbation" lives in compressibility-space (waterline) and in spectral degeneracy-space (Sakurai §5.2 static degenerate PT), not in a Hamiltonian-decomposition sense.

**Ontological grounding.** **Substrate:** structurally absent. The substrate Hamiltonian is monolithic (single Hashimoto operator at given Bloch fiber). **Why this form (if invoked):** would require additional structure not present in framework. **QFT ground:** interaction picture in QFT presupposes a free-Lagrangian + interaction-Lagrangian split; the framework's substrate is more like a single fully-interacting field, no "free" sector to peel off.

**Verdict.** unused-applied-negative. The interaction picture is structurally inapplicable. This is mildly informative for QFT ontology: the standard "free + interaction" decomposition is *not* fundamental — it's a calculational convenience that the substrate's monolithic structure avoids.

### §5.24 — Time-dependent perturbation theory (application sketch)

**Audit.** Not invoked. Framework uses static degenerate perturbation theory (Sakurai §5.2) at the spectral level (e.g., `predictions/dark_extraction_map_derivation.md`, retracted θ_23) — different operation.

**Search-instrument.** Same obstruction as 5.23 — no natural H_0/V split.

**Ontological grounding.** **Substrate:** the framework's perturbation is in *compressibility-space* (above-waterline survival) and *spectral-degeneracy-space* (lifting symmetry-protected degeneracies via static PT), not in time-evolution-space. **QFT ground:** Dyson series for transition amplitudes. The framework's substitution: instead of summing time-ordered V-insertions, sum waterline-survival contributions per cycle.

**Verdict.** unused-applied-negative. TDPT is displaced by static spectral PT + waterline-MDL summation; the substrate's "perturbation theory" is structurally different and does not invoke TDPT.

---

## 5.F — Specific complex-valued spectral content (3 ops)

| # | Op | Verdict | Citation | Ontological grounding |
|---|---|---|---|---|
| 5.25 | Eigenvalues with non-real algebraic structure (e.g., h = (√3+i√5)/2) | invoked-direct | `predictions/h_walker_eigenvalue.py`; `../theorems/theorem_bloch_lift_mu.md` (Ramanujan saturation \|h\|² = k − 1 = 2 at the P-point). | **Substrate:** Hashimoto eigenvalue at the high-symmetry P-point on srs. **Why this form:** specific algebraic form of B's characteristic polynomial at k = P; (√3 + i√5)/2 has \|h\|² = 2 = k − 1 (Ramanujan saturation, Alon-Boppana bound). **QFT ground:** no direct standard QFT analog — this is a substrate-specific spectral feature. The fact that QFT lacks an analog is informative: standard QFT doesn't expose the algebraic-eigenvalue structure of a discrete substrate. |
| 5.26 | Eigenvectors with complex phases | invoked-direct | Bloch states with k-dependent phases throughout `proofs/flavor/srs_bloch_*.py`. | **Substrate:** Bloch eigenvectors carry complex phases on Brillouin loops. **Why this form:** Layer 4.17 Bloch decomposition. **QFT ground:** momentum-space wavefunctions in QM — derived from substrate translation invariance. |
| 5.27 | Berry / geometric phases on parameter spaces | invoked-direct | `proofs/flavor/srs_bloch_ckm.py` (Wilson loops → CP phases δ_CP). | **Substrate:** phases accumulated by Bloch eigenvectors along closed BZ paths. **Why this form:** holonomy of a Bloch bundle over the BZ. **QFT ground:** anomaly coefficients, Wilson loops, gauge-field topology, θ-vacua. The framework's CKM phases trace to substrate Berry phases. |

**5.F totals:** 3/3 invoked-direct.

---

## 5.G — Complex Lie groups and spin representations (5 ops)

| # | Op | Verdict | Citation | Ontological grounding |
|---|---|---|---|---|
| 5.28 | Complex Lie groups: Spin(n;ℂ), SU(n), U(n), GL(n;ℂ) | invoked-direct | `predictions/sin2_theta_W.py`; `proofs/gauge/k4_pati_salam_cl8.py`. | **Substrate:** automorphism groups of Cl(n;ℂ) at substrate node; act on the 8-dim Dirac spinor. **Why this form:** complexification of real Lie groups (4.39) post-§F. **QFT ground:** SM gauge groups — derived as automorphism structure of substrate Clifford algebra. |
| 5.29 | Spin representations of Spin(n) on Cl(n;ℂ) spinors | invoked-direct | `predictions/theorem_B3_spinor_fermion.py`; `proofs/foundations/B3_chirality_bridge.py`. | **Substrate:** 8-dim Cl(6;ℂ) spinor rep at each substrate node. **Why this form:** Spin(6;ℂ) ≅ SU(4) acts on the 8-dim Dirac spinor as the unique irreducible rep. **QFT ground:** Dirac spinor → Pati-Salam fermion content. |
| 5.30 | Pati-Salam embedding Spin(4) × Spin(2) ⊂ Spin(6) on Cl(6;ℂ) spinor | invoked-direct | `proofs/gauge/k4_pati_salam_cl8.py`; `predictions/Q_Koide.py`; `predictions/sin2_theta_W_derivation.md`. | **Substrate:** subgroup embedding determined by srs primitive cell's symmetry (S_4 → 432 cubic point group). **Why this form:** structural property of substrate's spatial symmetry group inheriting onto Cl(6) bivector decomposition. **QFT ground:** Pati-Salam GUT structure of SM gauge — derived from substrate spatial symmetry, not postulated. |
| 5.31 | Complex characters χ_ρ ∈ ℂ for complex representations | invoked-direct | `predictions/ADOPTED_P1_ramanujan_support.py` (C₃ characters χ_ω, χ_{ω²}); `predictions/Q_Koide.py`. | **Substrate:** characters of C₃ acting on srs primitive cell at P; (1, ω, ω²) values. **Why this form:** C₃ irreps over ℂ have complex characters. **QFT ground:** complex reps of compact gauge groups; charge labels. |
| 5.32 | Complex Clebsch-Gordan for SU(n) | invoked-direct | `proofs/masses/srs_nu_mass_ps.py` Pati-Salam CG factor; `proofs/masses/greens_mass_predictions.py` SU(5) Clebsches in Yukawa relations. | **Substrate:** decomposition of tensor products of Spin(6;ℂ)/SU(n) reps. **Why this form:** standard complex rep theory; CG coefficients are explicit. **QFT ground:** mass relations and gauge-coupling unification factors at GUT scale. |

**5.G totals:** 5/5 invoked-direct.

---

## 5.H — Wick rotation and quantum statistical mechanics (6 ops)

| # | Op | Verdict | Citation | Ontological grounding |
|---|---|---|---|---|
| 5.33 | Wick rotation t → −iτ | invoked-direct | `proofs/foundations/theorem_feshbach_scalar_pairing.py` (Wick-rotation selection in B3.4 spinor analysis). | **Substrate:** rotation between Lorentzian and Euclidean continuum dynamics; the substrate admits both via Layer 3 Stone (real and complex forms) on appropriate sectors. **Why this form:** unitary continuum dynamics e^{−iHt} → e^{−Hτ} under analytic continuation. **QFT ground:** Wick rotation in path integrals — the substrate's Euclidean-Lorentzian duality grounds it. |
| 5.34 | Quantum partition function Z(β) = Tr(e^{−βH}) | unused-deferred | See application §5.34 below. | **Substrate:** thermal trace over substrate Hilbert space. **Why this form (if invoked):** standard QFT thermal partition function. **QFT ground:** thermal field theory; KMS condition. |
| 5.35 | Thermal density matrix ρ(β) = Z⁻¹ e^{−βH} | unused-deferred | See application §5.35 below. | **Substrate:** thermal state on substrate Hilbert space. **Why this form (if invoked):** Boltzmann weighting. **QFT ground:** KMS state; gauge thermal field theory. |
| 5.36 | von Neumann entropy S(ρ) = −Tr(ρ log ρ) | unused-deferred | See application §5.36 below. | **Substrate:** quantum analog of Shannon entropy on substrate density matrices. **Why this form (if invoked):** quantum information measure for mixed states. **QFT ground:** entanglement entropy, black-hole entropy, area law. |
| 5.37 | Schmidt rank of bipartite pure state | unused-deferred | Pairs with 5.16. | **Substrate:** correlation dimensionality observer-auxiliary. (See §5.16 for fuller treatment.) |
| 5.38 | Entanglement entropy S(Tr_B\|ψ⟩⟨ψ\|) | unused-deferred | See application §5.38 below. | **Substrate:** quantitative measure of observer-substrate entanglement. **Why this form (if invoked):** vN entropy of partial trace. **QFT ground:** entanglement entropy in QFT; area law for ground states. |

**5.H totals:** 1/6 invoked-direct, 5/6 unused-deferred.

### §5.34–§5.38 — Quantum thermal/information ops (consolidated sketch)

These five ops form a tightly-coupled cluster: thermal trace Z (5.34), thermal density ρ(β) (5.35), vN entropy S(ρ) (5.36), Schmidt rank (5.37), entanglement entropy (5.38). All five are post-field-selection quantum-information machinery. None is currently invoked.

**Audit.** Framework uses Shannon entropy (4.5) at the *classical* substrate level (toggle distributions) and partial trace (5.14) for *operational* compression, but not the *quantum statistical-information* apparatus.

**Search-instrument.** Major candidate forward-construction direction: derive a substrate-thermal structure with H = continuum-limit Hamiltonian, β = inverse-Planck-temperature, and compute:
- Z(β) = Tr(e^{−βH}) over substrate Hilbert space.
- vN entropy S of an observer reduced density.
- Entanglement entropy of substrate-observer split.

If these quantities have closed-form expressions tied to substrate spectral content (Hashimoto eigenvalues, Ramanujan saturation), they could provide:
- Cosmological-thermodynamics-type predictions (entropy density, partition function asymptotics).
- A substrate-grounded derivation of QFT's KMS state.
- A substrate value for entanglement entropy of vacuum sectors → potential connection to area law / holographic bounds.

**Ontological grounding.** **Substrate:** quantum thermal/information ops generalize classical Shannon (4.5) + classical statmech (4.45–4.47) + partial trace (5.14) to the *quantum* register at substrate-Planck scale. **Why this form:** standard quantum-information machinery applies once §F selects ℂ. **QFT ground:** the entire quantum-statistical-mechanics + entanglement-entropy + holographic-entropy apparatus that QFT *uses* but doesn't structurally derive. The framework's substrate could ground this apparatus by giving specific values to Z, S, S_ent for substrate sectors.

**Verdict.** unused-deferred (cluster). High-priority forward-construction candidate. Worth a focused investigation of substrate thermal/information structure once the conditional-expectation route (4.25) is explored — both could ladder up to a substrate-based grounding of QFT's full quantum-information apparatus.

---

## Aggregate (Layer 5)

| Status | 5.A | 5.B | 5.C | 5.D | 5.E | 5.F | 5.G | 5.H | Total |
|---|---|---|---|---|---|---|---|---|---|
| invoked-direct | 5 | 6 | 5 | 3 | 1 | 3 | 5 | 1 | 29 |
| unused-applied-negative | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 2 |
| unused-deferred | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 5 | 7 |
| **Layer total** | **5** | **6** | **6** | **3** | **4** | **3** | **5** | **6** | **38** |

**Coverage.** 38/38 catalog entries audited.

**Forward-construction docs queued.** §5.16 Schmidt decomposition; §5.22 Heisenberg picture; §5.34–§5.38 quantum thermal/information cluster. None spawned this pass; consolidated as priority queue.

---

## Honest verdict on Layer 5 sweep (with ontology lens)

**Three-lens yield categories:**
1. New low-MDL invariant matching SM observable: **none** this layer.
2. Cross-validation of existing prediction via distinct route: **none** this layer.
3. Pinned obstruction: **two** (5.23 interaction picture, 5.24 TDPT — both structurally inapplicable because substrate has no H_0/V split and no time-evolution-perturbation series).
4. **Forward-construction candidates with high ontological yield:** **§5.16 Schmidt rank**, **§5.34–§5.38 quantum thermal/information cluster** (six unused ops in a tightly-coupled bundle).
5. **Ontological grounding density:** Layer 5 is the catalog's densest ontological territory. §5.B (CAR / JW / Cl / spinor / parity / Majorana) grounds the algebraic structure of QFT fermions in substrate involutivity. §5.C grounds the density-matrix postulate of QM in compression. §5.D distinguishes graph T-symmetry from process T-symmetry — a substrate-derived insight not standard in QFT. §5.G grounds the Pati-Salam GUT embedding in substrate spatial symmetry.

**Key ontological findings to harvest into the meta-doc:**

| QFT-postulated object | Layer-5 grounding (this audit) |
|---|---|
| **CAR algebra of fermions** | JW transform of substrate involutive toggles (5.6 → 5.7). |
| **Dirac spinor (8-component complex)** | Irreducible Cl(6;ℂ) rep at trivalent substrate node (5.9). |
| **Fermion-number conservation** | ℤ/2-grading by toggle-count parity (5.10). |
| **Density matrix (mixed states)** | A3-T purification + partial trace; mixedness from compression (5.12, 5.14). |
| **Hermitian observables postulate** | §F register-storable spectrum forces self-adjoint H on ℂ (5.3). |
| **Schrödinger evolution** | Stone complex-form on substrate continuum H (5.21). |
| **Pati-Salam GUT structure** | Substrate spatial symmetry (cubic 432 → S_4) inherited onto Cl(6) (5.30). |
| **Berry phases / Wilson loops** | Bloch-eigenvector holonomy on substrate BZ (5.27). |
| **Wick rotation** | Substrate's Euclidean-Lorentzian duality at continuum limit (5.33). |
| **Majorana fermions are primary, Dirac derivative** | Substrate's primitive operators are involutive (Hermitian); Dirac c, c† are downstream complexifications (5.11). |

**Ontological gaps still open after Layer 5:**
- **Vacuum |0⟩** — which substrate state is this? Empty Cayley graph? Thermal β → ∞? Not yet grounded.
- **Field operator φ(x)** — averaged toggle density at a substrate point? Bloch-mode coordinates?
- **Path integrals** — Wick-rotated substrate partition function? Sum over toggle histories?
- **Interaction-picture / Dyson series** — structurally absent (substrate is monolithic).
- **BRST / gauge-fixing** — entirely absent from the framework.
- **Renormalization** — RG flow (4.52) is invoked via MSSM running but not derived from substrate.
- **Quantum thermal/information** — cluster identified but not grounded yet.

---

## Cumulative through Layer 5 (with three-lens columns)

| Layer | Ops | invoked | unused-applied-negative | unused-deferred | Notable ontology landings |
|---|---|---|---|---|---|
| 0 | 4 | 4 | 0 | 0 | substrate primitives (toggles, composition) |
| 1 | 13 | 11 | 0 | 2 | Cayley graph, word length, distance |
| 2 | 33 | 31 | 1 | 1 | L²(F_inv(E)) Hilbert space; adjacency op A |
| 3 | 13 | 12 | 1 | 0 | Stone → Schrödinger; field-selection chain |
| 4 | 49 | 47 | 1 | 1 | MDL apparatus; Bloch decomp; Killing-form gauge |
| 5 | 38 | 29 | 2 | 7 | **CAR/JW grounding QFT fermion algebra; ρ from compression; Pati-Salam from spatial symmetry** |
| **Cumulative** | **150** | **134** | **5** | **11** | — |

**Headline:** 150 ops audited; 134 invoked; 16 unused (5 applied-negative pinning obstructions, 11 deferred); 0 SM-matching positive yields, 1 cross-validation candidate queued (4.25), 1 high-priority cluster queued (5.34-5.38 quantum thermal/information).

**Ontological harvest is now substantial** — Layer 5 contributed ~10 QFT-postulated objects with substrate grounding. The meta-doc `../framework/framework_qft_ontology.md` (when harvested) will have material for multiple sections.

---

## Cross-references

- `operator_sweep_from_A1.md` §Layer 5 — source catalog.
- `../theorems/theorem_car_local_jordan_wigner.md` — central citation hub for 5.B (CAR / JW grounding).
- `predictions/theorem_B3_spinor_fermion.py` — Cl(6;ℂ) spinor (5.9, 5.30).
- `../theorems/theorem_A3_complex_hilbert_from_multiway.md` — purification + partial trace (5.14, 5.15).
- `proofs/flavor/srs_bloch_ckm.py` — Berry phase / Wilson loops (5.27).
- Predecessor audits: `operator_sweep_audit_layer_0_1.md`, `_2.md`, `_3.md`, `_4.md`.

---

## Status

Layer 5 audit complete with three-lens entries. Major ontological grounding harvested for QFT fermion algebra, density matrix, Pati-Salam GUT, Berry phases, Wick rotation. Forward-construction queue now has three priority items:
1. **§4.25 conditional expectation** (cross-validation candidate for A2-T).
2. **§5.16 Schmidt rank** of the A3-T purification (integer invariant tied to compression structure).
3. **§5.34–§5.38 quantum thermal/information cluster** (six coupled ops; could ground QFT KMS, area law, holographic entropy in substrate).

Next: Layer 6 (~24 ops on smooth manifold / Lorentzian / GR / cosmology); Appendix (21 explicitly-unused ops). After both, harvest into `../framework/framework_qft_ontology.md` meta-doc and backfill ontology lens into Layer 0–4 audits.
