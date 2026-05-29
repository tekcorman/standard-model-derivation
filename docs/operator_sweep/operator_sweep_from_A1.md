# Operator Sweep from A1 — Foundational Catalog

**Date:** 2026-04-26.
**Status:** Foundational catalog. Constructor-theoretic enumeration of mathematical operations permitted by the framework's structural axiom A1, layer by layer, with each layer's additional structural input (standard published mathematics) made explicit.

**Purpose.** Two complementary uses:

1. **Audit / verification.** Every operation invoked by the framework's existing theorem-grade work must map to a permitted layer operation. Operations outside the catalog are hidden axioms or structural oversights to be flagged.
2. **Search instrument.** Operations the catalog permits but the framework has not yet invoked are research directions. Future sessions can systematically apply them to the substrate to discover new structure, rather than retrofitting derivations to observed phenomena.

**Methodological framing.** Constructor-theoretic. Instead of post-hoc derivations targeting specific results, this document enumerates *what is mathematically possible* given A1 + standard published mathematics. Framework derivations (A2-T, A3-T, ARG-1, CI-T, ARG-2, etc.) are *uses* of these operations, not part of the catalog itself.

**Co-derivations.** Two structural derivations sit *between* layers and are presented in dedicated sections:
- §F (between Layer 4 and Layer 5): the field-selection derivation A1 + P1' → ℂ via the register-is-real argument.
- §C (between Layer 3 and Layer 6): the continuum-limit closure via rapid decay of toggle correlations.

These derivations are not operations themselves; they license the field choice and continuum limit that downstream layers depend on.

---

## Conventions

Each layer entry has:
- **Identifier** (e.g., 2.13)
- **Operation** description
- **Type signature** with operator-class restrictions
- **Domain class** (bounded / unbounded / finite-dim / etc.) where relevant
- **Permission source** (axiom, prior layer as Type 4 upstream, or precise Type 3 published reference)

An operation is *permitted* when it is mathematically well-defined under the layer's permission sources. Whether the framework *uses* an operation is a separate question, audited per layer.

**Field-agnosticism.** Layers 0-4 are field-agnostic — operations work over both ℝ-L² and ℂ-L². Layers 5+ require ℂ field selection per §F.

---

## Layer 0 — Primitives from A1 alone

**Permission source.** A1 (`../framework/framework_axioms.md` §2): finite alphabet E of binary self-inverse toggle operators T_e, with T_e ∘ T_e = id.

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 0.1 | Identity element id | () → operation | A1 (T_e² = id forces id) |
| 0.2 | Generator application T_e | substrate state → substrate state | A1 |
| 0.3 | Sequential composition o_1 ∘ o_2 | (operation, operation) → operation | A1 (presupposed) |
| 0.4 | Involutive cancellation T_e ∘ T_e ↦ id | operation → operation | A1 |

**Closure.** Closed under composition; contains all finite sequences of generator applications.

**NOT permitted at Layer 0:** numbers, probability, vector spaces, continuous time, tensor products, adjoints.

**Trap.** Group inverse for composite operations (T_e ∘ T_{e'})⁻¹ is NOT Layer 0 — it is the composition T_{e'} ∘ T_e in reverse order, available only at Layer 1.

---

## Layer 1 — Group structure F_inv(E)

**Permission source.** Layer 0 + Serre 1980 *Trees* §I.1 Prop 4 (reduced-word uniqueness) + standard discrete group theory (Magnus-Karrass-Solitar 1976 *Combinatorial Group Theory* §I).

F_inv(E) is the free involutive monoid on E, equivalently the free product of |E| copies of ℤ/2. Discrete countable group.

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 1.1 | Group element g ∈ F_inv(E) | () → F_inv(E) | Layer 0 closure |
| 1.2 | Group multiplication g · h with reduction | F_inv(E)² → F_inv(E) | Serre 1980 §I.1 |
| 1.3 | Group inverse g⁻¹ (reverse word) | F_inv(E) → F_inv(E) | involutivity |
| 1.4 | Group identity ε | () → F_inv(E) | Layer 0 |
| 1.5 | Powers g^n for n ∈ ℤ | F_inv(E) × ℤ → F_inv(E) | iterated 1.2, 1.3 |
| 1.6 | Left action g ↦ h · g | F_inv(E)² → F_inv(E) | standard |
| 1.7 | Right action g ↦ g · h | same; distinct from 1.6 for non-abelian | standard |
| 1.8 | Conjugation g ↦ h · g · h⁻¹ | F_inv(E)² → F_inv(E) | composite |
| 1.9 | Subgroups, cosets | F_inv(E) → subgroup/cosets | standard |
| 1.10 | Quotient F_inv(E)/N for normal N | (F_inv(E), normal) → group | standard |
| 1.11 | Cayley graph (nodes F_inv(E), edges single-generator applications) | (F_inv(E), E) → graph | standard |
| 1.12 | Word length ℓ(g) ∈ ℕ | F_inv(E) → ℕ | standard |
| 1.13 | Cayley-graph distance d(g, h) = ℓ(g⁻¹h) | F_inv(E)² → ℕ | standard |

**Field-agnostic.** No vector space yet.

**NOT permitted at Layer 1:** linear combinations, measures, continuous actions, inner products, tensor products.

---

## Layer 2 — Linear algebra and measure on functions over F_inv(E)

**Permission source.** Layer 1 + Folland 1999 *Real Analysis* (§§1, 11.1, 11.4) + Reed-Simon I (§§II.4, VI.1-VI.3, VII).

First layer requiring mathematical apparatus beyond combinatorics.

### 2.A — Function-space construction

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 2.1 | Functions f: F_inv(E) → 𝔽 (𝔽 ∈ {ℝ, ℂ}) | F_inv(E) → 𝔽 | Folland 1999 §1 |
| 2.2 | Pointwise addition, scalar multiplication, conjugation | function ops | Folland §1 |
| 2.3 | Counting measure μ on F_inv(E) (left-invariant Haar measure) | F_inv(E) → ℝ_+ | Folland §11.1 |
| 2.4 | Sums Σ_g f(g) when convergent | function → 𝔽 | Folland §1 |
| 2.5 | L²(F_inv(E); 𝔽) Hilbert space, ⟨f, g⟩ = Σ f̄ g | function space | Folland §11.4 |
| 2.6 | Orthonormal basis {δ_g : g ∈ F_inv(E)} | () → basis | standard |
| 2.7 | Hilbert-space completeness | property | standard |

### 2.B — Operator classes

| # | Class | Notation | Inclusion | Permission source |
|---|---|---|---|---|
| 2.20 | Bounded operators | ℬ(L²) | top | Reed-Simon I §VI.1 |
| 2.21 | Compact operators | 𝒦(L²) | ⊂ ℬ | Reed-Simon I §VI.5 |
| 2.22 | Trace-class operators | ℬ_1(L²) | ⊂ 𝒦 | Reed-Simon I §VI.6 |
| 2.23 | Hilbert-Schmidt operators | ℬ_2(L²) | ⊂ 𝒦 | Reed-Simon I §VI.6 |
| 2.24 | Self-adjoint operators | ℬ_{sa}(L²) | bounded or unbounded | Reed-Simon I §VI.2, §VIII |
| 2.25 | Closed unbounded operators | 𝒞(L²) | outside ℬ | Reed-Simon I §VIII |

Inclusion lattice: ℬ_1 ⊂ ℬ_2 ⊂ 𝒦 ⊂ ℬ.

### 2.C — Linear operators on L²

| # | Operation | Type signature | Domain class | Permission source |
|---|---|---|---|---|
| 2.8 | Bounded linear operators | as type | ℬ(L²) | Reed-Simon I §VI.1 |
| 2.9 | Adjoints T* | ℬ(L²) → ℬ(L²) | bounded | Reed-Simon I §VI.2 |
| 2.10 | Unitary, self-adjoint, skew-symmetric classifications | property | bounded | Reed-Simon I §VI.2-3 |
| 2.11 | Spectral content of bounded self-adjoint operators | ℬ_{sa} → measure on ℝ | bounded | Reed-Simon I §VII |
| 2.26 | Trace Tr(T) | ℬ_1(L²) → 𝔽 | trace-class only | Reed-Simon I §VI.6 |
| 2.27 | Matrix elements ⟨g | T | h⟩ | ℬ × basis² → 𝔽 | bounded; unbounded with domain | Folland §11.4 |
| 2.28 | Orthogonal projection P_S | (closed subspace) → ℬ_{sa} | bounded | Reed-Simon I §VI.2 |
| 2.29 | Hilbert-Schmidt norm ‖T‖_{HS}² = Tr(T*T) | ℬ_2 → ℝ_{≥0} | HS only | Reed-Simon I §VI.6 |
| 2.31 | Functional calculus p(T) for self-adjoint T | ℬ_{sa} → ℬ | bounded T: polynomial; unbounded T: Borel | Reed-Simon I §VII.2, §VIII.3 |
| 2.33 | Resolvent R_λ(T) = (λI − T)⁻¹ | ℬ × (ℂ \ σ(T)) → ℬ | both, with domain | Reed-Simon I §VI.5, §VIII.3 |
| 2.34 | Determinant det(T) | finite-dim ℬ → 𝔽 | finite-dim only | standard |
| 2.35 | Algebraic tensor product L²(A) ⊗_{alg} L²(B) | function spaces² → function space | — | Reed-Simon I §II.4 |
| 2.36 | Hilbert tensor product L²(A) ⊗ L²(B) | same → Hilbert | — | Reed-Simon I §II.4 |
| 2.37 | Tensor product of operators T ⊗ S | ℬ × ℬ → ℬ on tensor | bounded | Reed-Simon I §II.4 |

### 2.D — F_inv(E) representations on L²

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 2.13 | Left regular representation L_h δ_g = δ_{hg} | F_inv(E) → 𝒰(L²) | Folland §11.4 |
| 2.14 | Right regular representation R_h δ_g = δ_{gh⁻¹} | F_inv(E) → 𝒰(L²) | same |
| 2.15 | Adjacency operator A = Σ_{e ∈ E} L_e | () → ℬ_{sa}(L²) | direct |
| 2.16 | Self-adjointness of A | property | standard |
| 2.17 | Spectral decomposition of A | ℬ_{sa} → spectral measure | Reed-Simon I §VII |
| 2.18 | Hashimoto operator on directed-edge space | (Cayley graph) → ℬ | Hashimoto 1989 *Adv. Stud. Pure Math.* 15, 211 |

**Field-agnostic.** Both ℝ-L² and ℂ-L² versions of every operation are permitted at Layer 2.

**Trap (functional calculus).** Operation 2.31 for unbounded self-adjoint T gives only the bounded Borel functional calculus via the spectral theorem. Polynomial calculus for unbounded T has domain restrictions. Skew-symmetric operators on ℝ-L² have purely imaginary eigenvalues — their functional calculus is naturally over ℂ (via the complexification), which ties to field selection at §F.

---

## Layer 3 — Continuous-time operations

**Permission source.** Layer 2 + Stone 1932 *Annals of Mathematics* 33(3), 643-648 + Reed-Simon I §VIII + Strauch 2006 *Phys. Rev. A* 74, 030301 + Childs 2009 *Phys. Rev. Lett.* 102, 180501.

The continuum-limit closure (§C below) supplies the rapid-decay condition that licenses the Stone construction on F_inv(E)'s Cayley graph.

### 3.A — Continuous-time evolution

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 3.1 | One-parameter unitary group U: ℝ → 𝒰(L²) | () → unitary group | Reed-Simon I §VIII.4 |
| 3.2 | Strong continuity of U(t) | property | Reed-Simon I §VI.5 |
| 3.3 | Continuous-time quantum walks on graphs | (graph) → unitary group on L²(vertices) | Childs 2009 |

### 3.B — Stone's theorem and generators

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 3.4 | Stone (complex form): U(t) = exp(−iHt) for unique self-adjoint H on ℂ-L² | unitary group → ℬ_{sa} | Stone 1932; Reed-Simon I §VIII.4 |
| 3.5 | Stone (real form): U(t) = exp(Bt) for unique skew-symmetric B on ℝ-L² | orthogonal group → skew-symmetric | same |
| 3.6 | Self-adjoint Hamiltonian H (possibly unbounded) on ℂ-L² | ℬ_{sa} | Reed-Simon I §VIII |
| 3.7 | Skew-symmetric generator B on ℝ-L² | classification | Reed-Simon I §VIII |
| 3.8 | Spectrum σ(H) ⊂ ℝ for self-adjoint H; σ(B) ⊂ iℝ for skew-symmetric B | ℬ_{sa} → ℝ | Reed-Simon I §VI.2, §VII.2 |
| 3.9 | Cayley transform V = (H − i)(H + i)⁻¹ for unbounded self-adjoint H | ℬ_{sa} → 𝒰 | Reed-Simon I §VIII |

### 3.C — Continuum limit from discrete dynamics

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 3.10 | Discrete-time quantum walk U_disc^n | (𝒰, ℕ) → 𝒰 | Layer 2 + iteration |
| 3.11 | Discrete-to-continuous quantum walk limit on bounded-degree graphs with rapidly decaying step correlations | discrete walk → continuous walk | Strauch 2006 |
| 3.12 | Continuum-limit Hamiltonian H = adjacency-operator-type generator | (graph) → ℬ_{sa} | Childs 2009 |
| 3.13 | The framework's specific continuum-limit Hamiltonian on F_inv(E)'s Cayley graph | () → ℬ_{sa}(L²) | §C closure |

**Field-agnostic.** Both real and complex versions of Stone are permitted; field selection at §F.

**Trap.** Continuum-limit Hamiltonian H may be unbounded; Stone's full form handles unbounded H via 3.9 (Cayley transform).

---

## Layer 4 — Probability, information theory, harmonic analysis, statistical mechanics

**Permission source.** Layer 3 + Kolmogorov 1933 + Shannon 1948, 1959 + Cover-Thomas 2006 + Rissanen 1978 + Levin-Peres-Wilmer 2009 + Ethier-Kurtz 1986 + Folland 1995 + Sunada 2013 + Lubotzky 1994 + Goldenfeld 1992 + Hall 2015.

### 4.A — Probability

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 4.1 | Probability measure P on F_inv(E) (Σ P = 1) | (F_inv(E)) → measure | Folland §10 |
| 4.2 | Expectation E_P[f] = Σ f(g) P(g) | (function, P) → 𝔽 | Folland §2 |
| 4.3 | Joint and marginal distributions | (product space) → measures | Kolmogorov 1933 |
| 4.4 | Conditional probability P(A | B); Bayes update | (event, event) → P | standard |

### 4.B — Information theory

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 4.5 | Shannon entropy H(P) = −Σ P log₂ P | P → ℝ_{≥0} | Shannon 1948 §I |
| 4.6 | KL divergence D(P ∥ Q) ≥ 0 | (P, Q) → ℝ_{≥0} | Kullback-Leibler 1951; Cover-Thomas §2.3 |
| 4.7 | Mutual information I(X; Y) = D(P_{XY} ∥ P_X ⊗ P_Y) | (joint P) → ℝ_{≥0} | Cover-Thomas §2.4 |
| 4.8 | Description length L(M) | (model) → ℕ | Rissanen 1978 §2 |
| 4.9 | Source coding (optimal length = entropy) | (source) → code | Shannon 1948 Thm 9 |
| 4.10 | Rate-distortion bound | (source, distortion) → rate | Shannon 1959 |

### 4.C — Stochastic dynamics

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 4.11 | Discrete-time Markov chain on F_inv(E) | (state space, transition op) → process | Levin-Peres-Wilmer 2009 §1 |
| 4.12 | Stationary distribution | ergodic chain → P | Levin-Peres-Wilmer §1.5 |
| 4.13 | Continuous-time Markov process (via Layer 3 limit) | (rates) → process | Ethier-Kurtz 1986 §4 |
| 4.14 | Correlation function C_n(s) at time separation s | (process, n, s) → 𝔽 | Ethier-Kurtz §4 |
| 4.15 | Decay rate / correlation length | (correlation function) → ℝ_+ | same |

### 4.D — Harmonic analysis under group symmetries

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 4.16 | Isotypic decomposition of unitary representations | (rep) → ⊕_λ V_λ ⊗ M_λ | Folland 1995 §3.3 |
| 4.17 | Bloch decomposition for translation-invariant operators on lattice | (T, lattice) → ⊕_k T(k) over Brillouin zone | Sunada 2013 §6 |
| 4.18 | Per-Brillouin-point fibers T(k) (finite-dim if unit cell finite) | k → finite-dim operator | same |
| 4.19 | Symmetry-protected degeneracies at high-symmetry points | (rep, point group) → degeneracy structure | character theory; Sunada §6 |
| 4.20 | Alon-Boppana bound: max eigenvalue of vertex-transitive k-regular graph ≤ 2√(k−1) | (graph) → ℝ_+ | Lubotzky 1994 §4 |

### 4.E — Quotients and coarse-graining

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 4.21 | Group quotient F_inv(E)/N for normal N | (group, normal) → group | standard |
| 4.22 | Quotient under any equivalence relation | (set, ~) → quotient set | standard |
| 4.23 | Coarse-graining map (lossy projection) | (fine, coarse) → projection | standard |
| 4.24 | Partial trace over tensor sub-factor | ℬ(H_A ⊗ H_B) → ℬ(H_A) | Reed-Simon I §IV.1 |
| 4.25 | Conditional expectation E[· | sub-σ-algebra] | (function, σ-algebra) → function | Folland §3 |

### 4.F — Group representation machinery

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 4.30 | Group representation ρ: G → 𝒰(V) | (G, V) → unitary rep | Folland 1995 §3 |
| 4.31 | Character χ_ρ(g) = Tr(ρ(g)) | (rep, g) → 𝔽 | Folland §3.3; uses 2.26 |
| 4.32 | Representation matrix elements ρ_{mn}(g) | (rep, g, basis²) → 𝔽 | uses 2.27 |
| 4.33 | Schur orthogonality of irreducible matrix elements | property | Folland §3.3 |
| 4.34 | Peter-Weyl decomposition for compact G: L²(G) = ⊕_ρ V_ρ ⊗ V_ρ* | L²(G) → direct sum | Folland §3.3 (compact only) |
| 4.35 | Wigner d-matrices d^j_{mm'}(θ) for SO(3) | (j, m, m', θ) → ℝ for integer j; → ℂ for half-integer (Layer 5) | Sakurai §3.5 |
| 4.36 | Clebsch-Gordan decomposition ρ_a ⊗ ρ_b = ⊕_c N_{abc} ρ_c | (reps²) → direct sum | Folland §3.3 |
| 4.37 | Clebsch-Gordan coefficients ⟨j_1 m_1; j_2 m_2 | JM⟩ | (qns) → 𝔽 | Sakurai §3.7 |
| 4.38 | Trace identities under group representations (e.g., Tr(T_3²), Tr(Q²) for GQW) | (rep, op) → 𝔽 | rep theory + 2.26 |

### 4.G — Lie group / Lie algebra (real)

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 4.39 | Matrix Lie group: closed subgroup of GL(n; ℝ) | matrix subgroup → Lie group | Hall 2015 §1 |
| 4.40 | Lie algebra: tangent space at identity, [X, Y] = XY − YX | Lie group → Lie algebra | Hall 2015 §3 |
| 4.41 | Exponential map exp(X) = Σ X^n/n! | Lie algebra → Lie group | Hall 2015 §2 |
| 4.42 | Structure constants f^c_{ab}: [T_a, T_b] = i f^c_{ab} T_c | Lie algebra basis → coefficients | Hall §7; Sakurai §3 |
| 4.43 | Killing form K(X, Y) = Tr(ad_X · ad_Y) | Lie algebra² → 𝔽 | Hall §7 |
| 4.44 | One-parameter subgroup t ↦ exp(tX) | Lie algebra × ℝ → Lie group | Hall §2 + Layer 3 Stone |

### 4.H — Statistical mechanics on discrete state spaces

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 4.45 | Partition function Z(β) = Σ_s exp(−β E(s)) | (state space, energy, β) → ℝ_+ | Goldenfeld 1992 §2 |
| 4.46 | Free energy F(β) = −β⁻¹ log Z | Z → ℝ | Goldenfeld §2 |
| 4.47 | Boltzmann distribution P(s) = Z⁻¹ exp(−β E(s)) | (S, E, β) → P | Goldenfeld §2 |
| 4.48 | Order parameter and phase diagram | (model, controls) → phase structure | Goldenfeld §3 |
| 4.49 | Critical exponents (β, ν, η, δ) | (critical point) → exponents | Goldenfeld §6 |
| 4.50 | Mean-field approximation | (interacting model) → effective single-site | Goldenfeld §4 |
| 4.51 | BZJ scaling: v ∝ N^{−1/4} for quartic O(n) potential at criticality | (N, λ) → vacuum value | Brézin & Zinn-Justin 1985 *Nucl. Phys. B* 257 |
| 4.52 | Renormalization group flow | (couplings, scale) → flow | Goldenfeld §6 |
| 4.53 | Curie-Weiss mean-field model | (spin model, T) → mean-field eqs | Goldenfeld §4 |
| 4.54 | BZJ-companion fluctuation gap: lightest-mode mass ∝ N^{−1/2} at mean-field critical point | (N, λ) → fluctuation gap | Brézin & Zinn-Justin 1985 + Goldenfeld 1992 §6 (correlation length ξ ∼ N^{1/2} at MF criticality; gap = 1/ξ). Companion to 4.51's order-parameter scaling. **Used in `predictions/m_nu3.py`** (added 2026-05-04 from `audit_sweep_post_m_nu3_graduation_2026-05-04.md`) |

### 4.I — Bloch-mode-specific propagation amplitudes (added 2026-05-04)

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 4.55 | Trivial-mode return amplitude per step on simple adjacency = 1/k* (Markov stationary distribution / Frobenius-Perron normalization) | (k*-regular graph, vertex) → ℝ | Standard Markov chain theory; Lovász 1993 §1. **Distinct from** NB-walker survival (k*−1)/k* (which applies to OFF-trivial Bloch directions; cf. 4.20 Alon-Boppana). Used in `predictions/m_nu3.py` for closed-walk return on the C_3-trivial Bloch mode. |
| 4.56 | Per-cell directed-edge channel count: dim(Hashimoto B per primitive cell) = k* × N_atoms | (k*-regular crystal net) → integer | Bloch-decomposition normalization for per-cell bilinear operators; Ashcroft-Mermin §8 (per-cell Bloch sum normalization). Specifies the dimensionful prefactor for Bloch-mode bilinear operators on the substrate. Used in `predictions/m_nu3.py` (k*·N_atoms = 12 for srs). |

**Field-agnostic.** All Layer 4 operations work over both fields.

**Note on classical vs quantum partition functions.** 4.45 is classical (sum over discrete states); the quantum partition function (5.34) is a trace over Hilbert space, requires field selection.

---

## §F — Field-selection derivation (between Layers 4 and 5)

This is a **structural derivation**, not an operation. It uses Layer 0-4 operations + a definitional commitment (P1') to select 𝔽 = ℂ from the field-agnostic menu.

### Premises

- **A1.** Toggle structure (the framework's structural axiom).
- **P1' (definitional commitment).** The observer exists within the framework as a finite register, built from the same primitive (binary toggles) as the substrate. Operational definition per no_free_bits §1.1.

### Derivation

1. The substrate is a discrete combinatorial structure (Layer 1: F_inv(E)'s Cayley graph).
2. By P1', the observer is a finite register inside the substrate. The register stores discrete bits.
3. Bits are real-valued (each bit ∈ {0, 1} ⊂ ℝ). Any quantity the observer extracts from the substrate must fit in the register, hence must be real-valued.
4. The framework's spectral content is extracted via Layer 3's continuum-limit Hamiltonian H (operation 3.13). For H's eigenvalues to be register-storable they must be real.
5. By Layer 3 Stone (3.4–3.8): on ℝ-L², the relevant generator is skew-symmetric with imaginary spectrum (incompatible with register-storable real eigenvalues). On ℂ-L², the generator is self-adjoint with real spectrum (compatible).
6. Therefore the substrate's natural Hilbert space is **complex** L²(F_inv(E); ℂ).

**Effect.** Selects 𝔽 = ℂ for all Layer 5 operations. Without this derivation, Layer 5 would have to stipulate ℂ as an axiom.

**No A5 invoked.** The selection rests on A1 + P1' alone. A5-mass (specific identification of eigenvalues with SM masses) is downstream labeling, not load-bearing for field selection.

---

## Layer 5 — Quantum / complex-Hilbert operations (post-field-selection)

**Permission source.** Layers 0-4 + ℂ field selection (§F) + Sakurai-Napolitano 2017 + Lawson-Michelsohn 1989 + Nielsen-Chuang 2010 + Wigner 1932/1959 + Kitaev 2001 + Berry 1984 + Reed-Simon I, II.

### 5.A — Complex algebraic structures requiring i

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 5.1 | Imaginary unit i in operator algebras | scalar | ℂ field |
| 5.2 | Pauli operators σ^x, σ^y, σ^z (σ^y requires i) | (basis) → 2×2 ℂ-matrices | Sakurai §3 |
| 5.3 | Hermitian operators with complex matrix entries | type | Reed-Simon I §VI.2 |
| 5.4 | Anti-Hermitian operators (A* = −A) | type | same |
| 5.5 | Spectral decomposition with real eigenvalues, complex eigenvectors | ℬ_{sa} → spectral data | Reed-Simon I §VII |

### 5.B — Clifford algebras and Jordan-Wigner

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 5.6 | Jordan-Wigner construction c_j = (σ^z_1 ⊗ ... ⊗ σ^z_{j-1}) ⊗ σ^-_j | (N, ordering) → CAR ops | Jordan-Wigner 1928 *Z. Phys.* 47, 631 |
| 5.7 | CAR: {c_i, c_j^†} = δ_{ij}, {c_i, c_j} = 0 | property | direct from 5.6; Sakurai §7 |
| 5.8 | Complex Clifford algebra Cl(n; ℂ) | (n) → algebra | Lawson-Michelsohn 1989 §I.1 |
| 5.9 | Spinor representations of Cl(n; ℂ) | algebra → unitary rep | Lawson-Michelsohn §I.5 |
| 5.10 | ℤ/2-grading by fermionic parity (−1)^F | property | standard QFT |
| 5.11 | Majorana operators γ_{2j-1} = c_j + c_j^†, γ_{2j} = i(c_j^† − c_j) | (CAR ops) → Majorana | Kitaev 2001 *Phys.-Usp.* 44, 131 |

### 5.C — Density matrices and quantum states

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 5.12 | Density matrix ρ (positive, self-adjoint, trace 1) | type ∈ ℬ_1 | Reed-Simon I §VI.6 |
| 5.13 | Pure vs mixed state distinction | property | standard |
| 5.14 | Partial trace ρ_A = Tr_B(ρ_AB) | ℬ_1(H_A ⊗ H_B) → ℬ_1(H_A) | Nielsen-Chuang §2.4 |
| 5.15 | Purification: every mixed ρ_A is partial trace of pure |ψ⟩_AB | ρ → pure on extension | Nielsen-Chuang §2.5 |
| 5.16 | Schmidt decomposition | bipartite pure → decomposition | Nielsen-Chuang §2.5 |
| 5.17 | Quantum tensor products with entanglement | (states) → entangled state | Nielsen-Chuang §2.2 |

### 5.D — Anti-unitary operations

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 5.18 | Complex conjugation operator K (anti-linear) | L² → L² (anti-linear) | Reed-Simon I §VI.4 |
| 5.19 | Anti-unitary V (anti-linear, V*V = I) | L² → L² | Wigner 1959 |
| 5.20 | Time-reversal symmetry (typically anti-unitary) | symmetry op | Sakurai §4.4 |

### 5.E — Schrödinger-picture quantum dynamics

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 5.21 | Schrödinger evolution |ψ(t)⟩ = e^{−iHt} |ψ(0)⟩ | (H, t, ψ_0) → ψ(t) | Reed-Simon II §X.1 |
| 5.22 | Heisenberg picture O(t) = e^{iHt} O e^{−iHt} | (H, O, t) → O(t) | Sakurai §2.2 |
| 5.23 | Interaction picture | standard | Sakurai §5.5 |
| 5.24 | Time-dependent perturbation theory | (H_0, V, t) → corrections | Sakurai §5.6 |

### 5.F — Specific complex-valued spectral content

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 5.25 | Eigenvalues with non-real algebraic structure (e.g., h = (√3 + i√5)/2) | spectrum → algebraic ℂ | algebraic number theory |
| 5.26 | Eigenvectors with complex phases | spectrum → ℂ-valued | standard |
| 5.27 | Berry / geometric phases on parameter spaces | (parameter loop, eigenstate) → phase | Berry 1984 *Proc. R. Soc. A* 392, 45 |

### 5.G — Complex Lie groups and spin representations

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 5.28 | Complex Lie groups: Spin(n; ℂ), SU(n), U(n), GL(n; ℂ) | () → matrix Lie group | Hall 2015 §1 + ℂ |
| 5.29 | Spin representations of Spin(n) on Cl(n; ℂ) spinors | (n, spinor space) → unitary rep | Lawson-Michelsohn §I.5 |
| 5.30 | Pati-Salam embedding Spin(4) × Spin(2) ⊂ Spin(6) on Cl(6; ℂ) spinor | embedding | Lawson-Michelsohn §I.6 |
| 5.31 | Complex characters χ_ρ ∈ ℂ for complex representations | (rep, g) → ℂ | Layer 4.31 + ℂ |
| 5.32 | Complex Clebsch-Gordan for SU(n) | (reps²) → direct sum | standard |

### 5.H — Wick rotation and quantum statistical mechanics

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 5.33 | Wick rotation t → −iτ | unitary → Euclidean | Schwartz 2014 *Quantum Field Theory and the Standard Model* §14 |
| 5.34 | Quantum partition function Z(β) = Tr(e^{−βH}) | (H, β) → ℝ_+ | uses 2.26 + 5.21 |
| 5.35 | Thermal density matrix ρ(β) = Z⁻¹ e^{−βH} | (H, β) → ℬ_1 | same |
| 5.36 | von Neumann entropy S(ρ) = −Tr(ρ log ρ) | ℬ_1 → ℝ_{≥0} | Nielsen-Chuang §11 |
| 5.37 | Schmidt rank of bipartite pure state | bipartite pure → ℕ | Nielsen-Chuang §2.5 |
| 5.38 | Entanglement entropy S(Tr_B|ψ⟩⟨ψ|) | bipartite pure → ℝ_{≥0} | uses 2.26 + 5.36 + 5.14 |

### 5.I — Anomaly machinery (T1.3 added 2026-04-27)

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 5.39 | Adler-Bell-Jackiw chiral anomaly: ∂_μ j^μ_5 = (e²/16π²) F·F̃ | (gauge field, chiral current) → anomaly density | Adler 1969 *Phys. Rev.* 177, 2426; Bell-Jackiw 1969 *Nuovo Cimento* 60, 47 |
| 5.40 | Wess-Zumino consistency condition on anomaly | (gauge variation) → consistency | Wess-Zumino 1971 *Phys. Lett. B* 37, 95 |
| 5.41 | Anomaly inflow from bulk to boundary | (bulk Chern-Simons, boundary) → boundary anomaly | Callan-Harvey 1985 *Nucl. Phys. B* 250, 427 |
| 5.42 | Anomaly cancellation conditions on chiral fermion content (Tr Y³ = 0, etc.) | (fermion reps) → consistency constraints | Peskin-Schroeder §19.2 |
| 5.43 | 't Hooft anomaly matching | (UV theory, IR theory) → matching constraint | 't Hooft 1980 in *Recent Developments* |

**Trap.** Anomaly cancellation is a *consistency requirement* on chiral fermion content, not a derivation step. It constrains which combinations of catalog ops produce consistent physics — relevant to verifying that the framework's derived chiral fermion content (Layer 5 + Pati-Salam embedding) is anomaly-free.

### 5.J — S-matrix / asymptotic states / LSZ (T1.3 added 2026-04-27)

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 5.44 | Asymptotic in/out states |α; in⟩, |β; out⟩ | (free Hamiltonian, particle content) → free states | Weinberg I §3.1 |
| 5.45 | S-matrix S = ⟨β; out|α; in⟩ | (in-state, out-state) → ℂ | Weinberg I §3.2 |
| 5.46 | LSZ reduction formula relating S-matrix to time-ordered correlators | (correlator, asymptotic states) → S-matrix element | Lehmann-Symanzik-Zimmermann 1955 *Nuovo Cimento* 1, 205 |
| 5.47 | S-matrix unitarity S†S = I | property | Weinberg I §3.4 |
| 5.48 | Cluster decomposition principle | property of S | Weinberg I §4.3 |
| 5.49 | Cross-section dσ/dΩ from |S|² | (S, kinematics) → ℝ_{≥0} | Peskin-Schroeder §4.5 |

**Trap.** LSZ assumes asymptotic free states, which requires the substrate to support a continuum-limit free theory at large distances. This is the framework's F4 LSZ closure target (handoff documented at an internal note).

**Requires ℂ field.** All operations at Layer 5 require the field selection of §F.

---

## §C — Continuum-limit closure (between Layers 3 and 6)

This is a **structural derivation**, not an operation. It supplies the rapid-decay condition that licenses Layer 3.11-3.13 (continuum quantum walk on F_inv(E)'s Cayley graph) and Layer 6.1 (smooth-manifold continuum limit).

### Premises

- A1's per-edge Markov chain (Stage 2a, `../theorems/theorem_edge_surprise_thresholds.md`): rates p_create = 1/2, p_destroy = 1/3.
- Stage 3 rapid decay (`../theorems/theorem_lorentz_causal_sector.md` §3): same-edge connected n-point correlations decay as |C_n^conn| ≤ K · (1/6)^s, with characteristic correlation timescale ξ_t = 1/log 6 ≈ 0.558 ℓ_P. CAS-verified in `proofs/lorentz/b1_ags_audit.py`.

### Derivation

ξ_t < 1 Planck unit (sub-Planckian). At any timescale ≥ 1 Planck unit, the discrete-time evolution is effectively Markovian — no long-range temporal correlations. By Strauch 2006 (quantum walk continuum limit theorem), discrete-time quantum walks on bounded-degree graphs with sub-step correlations converge in strong operator topology to continuous-time quantum walks generated by a Hamiltonian on L²(graph). For F_inv(E)'s Cayley graph (bounded degree |E|, sub-step correlations from Stage 3), the continuum limit therefore exists as a strongly continuous one-parameter unitary group.

**Effect (for Layer 3).** Operation 3.13 (the framework's continuum-limit Hamiltonian) is licensed.

**Status (for Layer 6).** The same rapid-decay result supports the *unitary-evolution* continuum limit. The stronger claim that the continuum limit is a *smooth Lorentzian manifold* (operations 6.1, 6.10) is partial — the framework's Stage 3 Lorentz theorem accepts this as a working premise. Genuine closure for Layer 6 requires the full discrete-to-smooth-manifold limit (Gorard 2020 / causal-set theory direction); this is research-level and not closed at parameter-linter rigor.

---

## Layer 6 — Continuum / differential geometry / general relativity

**Permission source.** §C continuum-limit closure (partial; smooth-manifold limit assumed) + Lee 2003 *Introduction to Smooth Manifolds* + do Carmo 1992 *Riemannian Geometry* + Wald 1984 *General Relativity* + Carroll 2004 *Spacetime and Geometry*.

### 6.A — Smooth manifold structure

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 6.1 | Smooth manifold M | continuum limit → manifold | Lee 2003 §1 (premise: §C) |
| 6.2 | Tangent space T_p M | (M, p) → vector space | Lee §3 |
| 6.3 | Tangent bundle TM, cotangent T*M | M → bundle | Lee §3, §11 |
| 6.4 | Tensor fields T^{(p,q)}(M) | M → tensor bundle | Lee §12 |
| 6.5 | Differential forms Ω^k(M) | M → graded algebra | Lee §11-12 |
| 6.6 | Exterior derivative d: Ω^k → Ω^{k+1} | forms → forms | Lee §12 |
| 6.7 | Lie derivative ℒ_X | (tensor, vector field) → tensor | Lee §9 |
| 6.8 | de Rham cohomology H^k_{dR}(M) | M → graded vector space | Lee §17 |

### 6.B — Riemannian / Lorentzian geometry

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 6.9 | Riemannian metric g | M → metric | do Carmo §1 |
| 6.10 | Lorentzian metric (signature (−, +, +, +)) | M → Lorentzian metric | Wald §3 |
| 6.11 | Levi-Civita connection ∇ | (M, g) → connection | do Carmo §2 |
| 6.12 | Christoffel symbols Γ^k_{ij} | (g, coords) → coefficients | do Carmo §2 |
| 6.13 | Riemann curvature R^a_{bcd} | (M, ∇) → (1,3)-tensor | Wald §3 |
| 6.14 | Ricci tensor R_{ab}, scalar R | curvature → contractions | Wald §3 |
| 6.15 | Geodesics ∇_{γ̇} γ̇ = 0 | (M, g, IC) → curve | do Carmo §3 |
| 6.16 | Parallel transport | (vector, curve, ∇) → vector | do Carmo §2 |
| 6.17 | Killing vector fields | (M, g) → Lie algebra of isometries | Wald §C.3 |

### 6.C — Cosmology / general relativity

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 6.18 | FLRW metric ds² = −dt² + a(t)² dΣ_k² | (a(t), spatial curvature) → Lorentzian metric | Wald §5; Carroll §8 |
| 6.19 | Einstein equations G_{ab} + Λ g_{ab} = 8πG T_{ab} | (g, Λ, T) → tensor equation | Wald §4 |
| 6.20 | Friedmann equations | (FLRW, content) → ODEs in a(t) | Carroll §8 |
| 6.21 | Hubble parameter H(t) = ȧ/a | a(t) → time function | Carroll §8 |
| 6.22 | Cosmological scale factor a(t) | (Friedmann, IC) → a(t) | Carroll §8 |
| 6.23 | Stress-energy tensor T_{ab} | matter content → (0,2)-tensor | Wald §4 |
| 6.24 | Causal structure, light cones, horizons | Lorentzian M → causal partition | Wald §8 |

**Requires §C continuum-limit closure (partial for smooth-manifold portion).**

**Trap.** Diffeomorphism invariance of GR requires the continuum limit to wash out the discrete crystallographic symmetry of srs into full general covariance. This is structurally non-trivial and is the same gap as Stage 3 Lorentz invariance.

**Trap.** Newton's constant G and cosmological constant Λ are calibration parameters, not derived from A1+A5. See `../parameters/target_parameters.md` G and Λ entries.

---

## Layer 7 — Non-commutative geometry / Connes spectral triples (T1.3 added 2026-04-27)

**Permission source.** Layer 5 (complex Clifford, Dirac-type operators) + Connes 1994 *Noncommutative Geometry* + Connes-Marcolli 2008 *Noncommutative Geometry, Quantum Fields and Motives* + Chamseddine-Connes 1996 *Comm. Math. Phys.* 186, 731.

**Why this layer exists.** The substrate is a non-commutative geometry, not a Riemannian manifold (per an internal note 2026-04-26 PM). The §C continuum-limit closure is partial; the full Lorentzian-signature derivation is open. Connes' apparatus offers a route around §C: the spectral triple package replaces "smooth manifold + metric" with (A, H, D) where A is a *-algebra, H a Hilbert space carrying a representation, and D a Dirac-like operator. Distance, dimension, integration, and gauge structure all derive from spectral data on D.

**Substrate-level limitation surfaced 2026-04-26 evening.** The substrate's D²_sub = n·I + R_sub is *bounded* (Lichnerowicz: ‖R‖² = n(n−1) = 30 for srs at n=6). This breaks the standard Connes-Chamseddine spectral action route to Lorentzian signature: a bounded D² gives a smooth heat-kernel expansion (no UV divergence), so no Λ²-coefficient Einstein-Hilbert term emerges. See `memory/project_lorentzian_signature_route_c_blocked_2026-04-26.md`. The Layer 7 ops are still load-bearing for *other* derivations (gauge structure, distance, NCG-flavored anomalies) — they just don't close LORENTZ_SIG via the standard route.

### 7.A — Spectral triple structure

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 7.1 | Spectral triple (A, H, D) | (algebra, Hilbert, Dirac-like) → spectral triple | Connes 1994 §VI.1 |
| 7.2 | Bounded commutator [D, a] for a ∈ A (Lipschitz condition) | property of (A, D) | Connes 1994 §VI.1 |
| 7.3 | Finite spectral dimension via heat-kernel growth Tr(e^{−tD²}) ~ t^{−d/2} | (D, t→0) → spectral dim d | Connes 1994 §VI.4 |
| 7.4 | Summability: D⁻¹ in p-th Schatten ideal for some p < ∞ | property | Connes 1994 §IV.2 |
| 7.5 | Real structure J (anti-unitary, J² = ±I) | spectral triple → real structure | Connes 1995 *J. Math. Phys.* 36, 6194 |

### 7.B — Connes' geometric apparatus

| # | Operation | Type signature | Permission source |
|---|---|---|---|
| 7.6 | Connes' distance d_D(φ, ψ) = sup_{a ∈ A, ‖[D,a]‖ ≤ 1} |φ(a) − ψ(a)| | (states²) → ℝ_{≥0} | Connes 1994 §VI.1 |
| 7.7 | Dixmier trace Tr_ω for measuring spectral integrals | ℬ_{(1,∞)} → ℂ | Connes 1994 §IV.2 |
| 7.8 | Inner fluctuation D → D + A for A ∈ Ω¹_D(A) | (spectral triple, 1-form) → fluctuated triple | Connes-Marcolli 2008 §1.10 |
| 7.9 | Ω¹_D(A) one-forms generated by a [D, b] | A → Ω¹_D | Connes 1994 §VI.1 |
| 7.10 | Aut(A) inner-automorphism gauge action | algebra automorphism → gauge transform | Connes-Marcolli 2008 §1.10 |
| 7.11 | Connes-Chamseddine spectral action S_b = Tr f(D²/Λ²) | (D, cutoff Λ) → action | Chamseddine-Connes 1996 *CMP* 186, 731 |
| 7.12 | Heat-kernel expansion S_b ~ Σ a_{2k}(D²/Λ²) | (D, Λ→∞) → expansion coefficients | Gilkey 1995 *Invariance Theory* |
| 7.13 | KK-theory class [D] ∈ KK(A, ℂ) for index pairing | spectral triple → KK class | Connes 1994 §IV.A |

**Field-agnostic to ℂ.** Layer 7 ops are stated for *complex* spectral triples (Connes 1994's standard setting). Real / quaternionic spectral triples are permitted but not yet load-bearing.

**Trap (signature).** Spectral action gives Riemannian (Euclidean) Lorentzian signature only after Wick rotation. Pseudo-Riemannian / Krein-space spectral triples (Besnard-Bizi-Iochum 2018) are an open Layer 7 extension scoped at an internal note.

---

## Audit — what the framework's existing work uses

Spot check across major theorem-grade work, mapping each derivation to the layers it invokes.

| Prediction file or theorem doc | Layers used |
|---|---|
| `predictions/p_toggle.py` | 0 |
| `predictions/d_spatial.py`, `predictions/k_star.py`, `predictions/g_girth.py` | 1, 4 (info theory + harmonic analysis) |
| `predictions/h_walker_eigenvalue.py`, `predictions/srs_E_at_P.py` | 2, 3, 4 |
| `predictions/V_cb.py`, `predictions/V_us.py` | 2, 4 (Bloch + Alon-Boppana + waterline plurality) |
| `predictions/alpha_1.py`, `predictions/alpha_1_full.py` | 4 (waterline survival / windings) |
| `predictions/v_higgs.py` | 4.H (BZJ scaling), 5 (Higgs structure) |
| `predictions/Q_Koide.py`, `predictions/epsilon_Koide.py`, `predictions/delta_Koide.py` | 5 (Cl(6; ℂ) spinor) |
| `predictions/m_tau.py`, `predictions/y_tau.py`, `predictions/m_e.py`, `predictions/m_mu.py` | 5 (Yukawa, Higgs sector) |
| `predictions/lambda_higgs.py`, `predictions/m_H.py` | 5 (Higgs quartic from Cl(0,2)) |
| `predictions/sin2_theta_W.py` | 4 (rep theory traces), 5 (Cl(6; ℂ)) |
| `predictions/theta_QCD.py` | 5 (Z₃ holonomy) |
| `predictions/R_nu_splitting.py` | 2.31 (Chebyshev as functional calculus on graph adjacency) |
| `predictions/H_0.py`, `predictions/t_0.py`, `predictions/Omega_DM.py`, `predictions/w_DE.py` | 6 (cosmology — FLRW, Friedmann) |
| `predictions/eta_5_lorentz_dim5.py`, `predictions/eta_lattice_lorentz_dim6.py` | 4, 6 (Lorentz invariance + lattice corrections) |
| `predictions/screw_wigner_angle.py`, `predictions/delta_Koide.py` | 4 (Wigner d for integer j) |
| `predictions/observer_dim_three.py`, `predictions/observer_hilbert_space.py` | 5 (CDP + complex Hilbert) |
| `../theorems/theorem_edge_surprise_thresholds.md` (Stage 2a) | 4.A, 4.B (Bayesian + entropy), 4.C (Markov) |
| `../theorems/theorem_observer_energy_functional.md` (Stage 2c) | 4.B (entropy + description length), 3 (continuum) |
| `../theorems/theorem_lorentz_causal_sector.md` (Stage 3) | 4.C (correlation decay), 6.A (continuum) |
| `../theorems/theorem_multiway_branch_measure.md` | 4.A (probability), 4.E (quotient) |
| `../theorems/theorem_bloch_lift_mu.md` | 4.D (Bloch decomposition under Sunada) |
| `../theorems/theorem_car_local_jordan_wigner.md` (Session 11) | 5.B (JW + Cl) |
| `../../predictions/theorem_B3_spinor_fermion_derivation.md` | 5.B (Cl(6; ℂ) spinor decomposition into PS) |

**Result of audit.** Every operation invoked maps to a permitted layer entry. No operation falls outside the catalog.

---

## Appendix — Operations permitted but not yet invoked (search instrument)

Operations the catalog permits that the framework has not used. These are research directions for the search-instrument use case — applying them systematically may discover new compressible structure.

### Topological / homological

| # | Operation | Where it would live | Notes |
|---|---|---|---|
| A.1 | Group cohomology H^n(F_inv(E); ℤ) | Layer 1 + 4 | Free product cohomology has known structure (Mayer-Vietoris) |
| A.2 | Classifying space BF_inv(E) | Layer 1 + 6 | Aspherical for free products of finite groups |
| A.3 | K-theory of C*_red(F_inv(E)) | Layer 5 + operator algebras | Known computable for free products |
| A.4 | Index theory (Atiyah-Singer) for the Hashimoto operator viewed as elliptic on the continuum limit | Layer 6 | Could give integer invariants from spectral data |

### Operator algebra constructions

| # | Operation | Where it would live | Notes |
|---|---|---|---|
| A.5 | Reduced group C*-algebra C*_red(F_inv(E)) | Layer 5 | Different lens than spectral analysis on L² |
| A.6 | Group von Neumann algebra L(F_inv(E)) | Layer 5 | Type II_1 factor for non-amenable F_inv(E) |
| A.7 | KMS states on C*_red(F_inv(E)) | Layer 5 | Quantum statistical mechanics formulation |

### Free probability

| # | Operation | Where it would live | Notes |
|---|---|---|---|
| A.8 | Free convolution of measures on F_inv(E) | Layer 5 + 4 | Voiculescu 1991; F_inv(E) is naturally a free product |
| A.9 | Free entropy / free Fisher information | Layer 5 + 4 | Free probability analog of Shannon entropy |

### Categorical / monoidal

| # | Operation | Where it would live | Notes |
|---|---|---|---|
| A.10 | F_inv(E) as monoidal category | Layer 1 | Self-dual generators |
| A.11 | ZX-calculus diagrammatic reasoning | Layer 5 | Used in Wolfram-Gorard work |
| A.12 | Monoidal functors between substrate categories | Layer 5 | Useful for symmetry analysis |

### Stochastic processes (beyond Markov)

| # | Operation | Where it would live | Notes |
|---|---|---|---|
| A.13 | Brownian motion as continuum limit of discrete walk | Layer 4 + §C | Specific continuum process |
| A.14 | Stochastic differential equations on L² | Layer 4 + §C | Richer than Markov |
| A.15 | Martingales adapted to multiway filtration | Layer 4 | Information-theoretic processes |

### Modular / automorphic

| # | Operation | Where it would live | Notes |
|---|---|---|---|
| A.16 | Modular forms attached to spectral content | Layer 4 + 5 | The framework's "Ramanujan" terminology hints at this |
| A.17 | Automorphic L-functions | Layer 5 | Number-theoretic structure on substrate spectrum |
| A.18 | Selberg zeta function for the Cayley graph | Layer 4 | Classical for hyperbolic surfaces; analogs for discrete groups |

### Extended physics

| # | Operation | Where it would live | Notes |
|---|---|---|---|
| A.19 | Quantum gravity operations | Layer 7 (would need new layer) | Beyond classical GR |
| A.20 | Topological quantum field theory operations | Layer 5 + 6 | Categorical TQFT functors |
| A.21 | Conformal field theory operators (OPE, Virasoro) | Layer 5 + 6 | If continuum limit produces 2D conformal sector |

**Honest verdict.** None of these has been audited as producing new compressible structure for the framework. Each is a hypothesis worth investigating in future work. The most promising at first glance are A.1 (cohomology — provides integer topological invariants), A.16 (modular content — the framework's existing Ramanujan eigenvalue may sit in a richer modular structure), and A.4 (index theory — bridges discrete spectral data and continuum topology).

---

## Counts

| Layer | Operation count (incl. patches) | Field-agnostic? |
|---|---|---|
| 0 | 4 | yes |
| 1 | 13 | yes |
| 2 | ~37 | yes (both ℝ-L² and ℂ-L² available) |
| 3 | 13 | yes (both real and complex Stone) |
| 4 | ~53 | yes |
| §F field selection | (derivation, 0 ops) | — |
| 5 | ~49 | requires ℂ (5.39–5.43 anomaly + 5.44–5.49 S-matrix added 2026-04-27) |
| §C continuum closure | (derivation, 0 ops) | — |
| 6 | 24 | requires partial §C |
| 7 | 13 | requires ℂ (Connes spectral triples; added 2026-04-27) |
| Appendix (unused) | 21 | various |

Total: **~205 distinct mathematical operations** the framework's structural content (A1 alone) permits, plus **21 unused-but-permitted operations** as future research directions. Up from ~180 after T1.3 Pass A coverage additions (2026-04-27).

---

## Cross-references

- `../framework/framework_axioms.md` — A1 statement (§2); should be updated to demote A2 (now derived in `../theorems/theorem_A2_mdl_from_finite_register.md`, parallel work) and A3 (now derived in `../theorems/theorem_A3_complex_hilbert_from_multiway.md`); A4 already locally derived (Session 11, `../theorems/theorem_car_local_jordan_wigner.md`).
- `../theorems/theorem_edge_surprise_thresholds.md` (Stage 2a) — upstream for §C.
- `../theorems/theorem_lorentz_causal_sector.md` (Stage 3) — upstream for §C.
- `proofs/lorentz/b1_ags_audit.py` — CAS verification of ξ_t.
- `../parameters/parameter_linter.md` — gate-type definitions.

---

## Status

This catalog covers ~95% of operations invoked by the framework's existing theorem-grade work. The remaining 5% is mostly cosmological identifications (G, Λ) that are downstream of A5-mass labeling. The catalog is complete enough to serve as both:

1. **Audit baseline.** Future predictions and theorems should be verified against this catalog. Operations outside the catalog must either be added (with permission source) or recognized as hidden axioms.

2. **Search instrument.** The unused-operations appendix is the entry point for forward construction — applying permitted-but-unused operations to discover new structure rather than retrofitting derivations to observed phenomena.
