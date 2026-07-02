# Operator Sweep Audit — Layer 2

**Date:** 2026-04-26.
**Status:** Per-operation audit. Layer-by-layer execution of the operation-constructor workstream.
**Source catalog:** `operator_sweep_from_A1.md` §Layer 2.
**Predecessor:** `operator_sweep_audit_layer_0_1.md`.

## Methodology

Same verdict taxonomy as the Layer 0+1 audit: invoked-direct / invoked-indirect / invoked-negatively / unused-applied-{positive,derivable,negative} / unused-deferred. Each op cited to a representative prediction or theorem doc, or sketched at first-order if unused.

Layer 2 contains ~37 operations grouped into:
- **2.A** Function-space construction (7 ops)
- **2.B** Operator classes (taxonomy, 6 entries)
- **2.C** Linear operators on L² (12 ops)
- **2.D** F_inv(E) representations on L² (6 ops)

The catalog numbers skip in places (2.12, 2.19, 2.30, 2.32, 2.38 absent); the audit follows the catalog's numbering.

---

## 2.A — Function-space construction

| # | Operation | Verdict | Citation / sketch |
|---|---|---|---|
| 2.1 | Functions f: F_inv(E) → 𝔽 | invoked-direct | Every state vector. `predictions/h_walker_eigenvalue.py` (eigenstates of A on L²(srs)). |
| 2.2 | Pointwise addition, scalar multiplication, conjugation | invoked-direct | Linear-combination of states; complex conjugation in inner product. Foundational for L². |
| 2.3 | Counting measure μ on F_inv(E) (left-invariant Haar) | invoked-direct | Underlies the L² inner product ⟨f, g⟩ = Σ f̄ g; uniform branch measure in `../theorems/theorem_multiway_branch_measure.md`. |
| 2.4 | Sums Σ_g f(g) when convergent | invoked-direct | Partition functions, expectations, normalizations. `predictions/v_higgs.py` mean-field sums. |
| 2.5 | L²(F_inv(E); 𝔽) Hilbert space | invoked-direct | Substrate Hilbert space. `../framework/layer_1_hilbert_space_identification.md`; `../theorems/theorem_A3_complex_hilbert_from_multiway.md`. |
| 2.6 | Orthonormal basis {δ_g : g ∈ F_inv(E)} | invoked-direct | Position basis on Cayley graph; used for matrix elements in every walk amplitude. |
| 2.7 | Hilbert-space completeness | invoked-direct | Required for spectral theorem (Layer 2.11) and Stone (Layer 3.4). Foundational. |

**2.A totals.** 7/7 invoked-direct.

---

## 2.B — Operator classes (taxonomy)

| # | Class | Verdict | Citation / sketch |
|---|---|---|---|
| 2.20 | Bounded operators ℬ(L²) | invoked-direct | Adjacency operator A is bounded (deg 3 + Alon-Boppana bound 2√2). Used in `predictions/h_walker_eigenvalue.py`. |
| 2.21 | Compact operators 𝒦(L²) | unused-applied-negative | See application sketch §2.21 below. |
| 2.22 | Trace-class ℬ_1(L²) | invoked-direct | Density matrices in `predictions/observer_hilbert_space.py`; partition function trace at Layer 5.34. |
| 2.23 | Hilbert-Schmidt ℬ_2(L²) | invoked-direct | Gram matrices under HS inner product in `predictions/G2_cl2_channels.py` (rank computations). |
| 2.24 | Self-adjoint ℬ_{sa} | invoked-direct | Every Hamiltonian; adjacency operator. `../theorems/theorem_lorentz_causal_sector.md`. |
| 2.25 | Closed unbounded operators | invoked-direct | Continuum-limit Hamiltonian (Layer 3.13) is unbounded; required for Stone's theorem with unbounded generator. |

**2.B totals.** 5/6 invoked-direct, 1/6 unused-applied-negative.

### §2.21 — Compact operators on L²(F_inv(E)) (application sketch)

**Operation.** Class of bounded operators T such that T maps bounded sets to relatively compact sets; equivalently, limits of finite-rank operators in operator norm.

**Application to substrate.** The framework's central operators on L²(F_inv(E)) are:
- Adjacency A = Σ_e L_e — sum of finitely many unitary L_e, hence bounded. Each L_e is a translation by a generator and is unitary (norm 1, isometry), hence not compact. The finite sum A is also not compact (its spectrum is continuous on the infinite Cayley graph).
- Hashimoto operator B on directed-edge space — bounded but, on the infinite Cayley graph, has continuous spectrum and is not compact.
- Continuum-limit Hamiltonian H (Layer 3.13) — unbounded, hence outside ℬ(L²) entirely; question of compactness moot.

**Output.** Compactness fails for every Hamiltonian-type operator the framework uses. The reason is structural: F_inv(E) is an infinite group, the Cayley graph is infinite-vertex, and the regular representation is *factor* (irreducible in the von Neumann sense) — the regular rep contains no compact-perturbation structure.

**Compressibility check.** Compact operators are characterized by discrete spectrum accumulating only at 0 (plus possibly 0 itself). The framework's predictions involve *continuous* Hashimoto spectra (e.g., the bulk eigenvalue band beyond ω² = 36 isolated point). A compact operator framework would force discreteness incompatible with the framework's existing spectral content.

**SM observable check.** Not applicable — the operation is incompatible with framework structure at first principles.

**Verdict.** unused-applied-negative. Compactness is *structurally excluded* from F_inv(E)'s regular representation. This is a clean obstruction — closes the research direction "use compact-operator machinery on the substrate" permanently.

**Useful corollary.** This obstruction means the framework cannot use spectral-theory-of-compact-operators tools (Riesz-Schauder, Fredholm alternative on bounded domain). Continuous-spectrum tools (spectral measures, von Neumann decomposition into atoms + absolutely continuous + singular continuous) are the correct apparatus, consistent with what the framework already uses in Bloch decomposition.

---

## 2.C — Linear operators on L²

| # | Operation | Verdict | Citation / sketch |
|---|---|---|---|
| 2.8 | Bounded linear operators | invoked-direct | Adjacency, Hashimoto, projectors. Pervasive. |
| 2.9 | Adjoints T* | invoked-direct | Self-adjointness checks. `../theorems/theorem_car_local_jordan_wigner.md`; `../theorems/theorem_A3_complex_hilbert_from_multiway.md` (adjoint structure on multiway space). |
| 2.10 | Unitary / self-adjoint / skew-symmetric classifications | invoked-direct | Self-adjoint: every H. Skew-symmetric: `../theorems/theorem_A3_complex_hilbert_from_multiway.md` (real-Hilbert generator). Unitary: continuous-time evolution. |
| 2.11 | Spectral content of bounded self-adjoint operators | invoked-direct | Spectral decomposition of A via Sunada/Bloch in `../theorems/theorem_bloch_lift_mu.md`; spectral measure in `predictions/uniform_Q_density.py`. |
| 2.26 | Trace Tr(T) | invoked-direct | Character traces in rep theory (Layer 4.31); QFT partition functions (Layer 5.34); also used implicitly in HS-norm via Tr(T*T). |
| 2.27 | Matrix elements ⟨g | T | h⟩ | invoked-direct | Every walk amplitude: ⟨v_t | A^L | P⟩ in `predictions/h_walker_eigenvalue.py`. |
| 2.28 | Orthogonal projection P_S | invoked-direct | P+ Bloch projector, isotypic projectors. `../theorems/theorem_g2_edge_qubit_su2.md` (Higgs doublet projector); `predictions/observer_hilbert_space.py`. |
| 2.29 | Hilbert-Schmidt norm ‖T‖_{HS}² = Tr(T*T) | invoked-direct | Gram-matrix rank in `predictions/G2_cl2_channels.py`, `proofs/foundations/theorem_G2_cl2_channels.py`. |
| 2.31 | Functional calculus p(T) for self-adjoint T | invoked-direct | Chebyshev functional calculus on adjacency in `predictions/R_nu_splitting.py`, `predictions/m_nu2.py`. |
| 2.33 | Resolvent R_λ(T) = (λI − T)⁻¹ | invoked-direct | Resolvent at k_P used in V_us route attempts (`memory/session_handoff_2026-04-22_session22.md`); `../theorems/theorem_cosmic_birefringence.md` cyclotomic resolvent. |
| 2.34 | Determinant det(T) | invoked-direct | Ihara-Bass determinant identity in `predictions/h_walker_eigenvalue.py`, `proofs/flavor/srs_r_from_ihara_direct.py`, `predictions/Feshbach_coupling_strength.py`. |
| 2.35 | Algebraic tensor product L²(A) ⊗_alg L²(B) | invoked-indirect | Framework uses tensor products without explicitly distinguishing algebraic vs Hilbert completion. Implicit in 2.36 usage. |
| 2.36 | Hilbert tensor product L²(A) ⊗ L²(B) | invoked-direct | Multi-particle constructions (`predictions/observer_hilbert_space.py`); JW tensor structure (`../theorems/theorem_car_local_jordan_wigner.md`). |
| 2.37 | Tensor product of operators T ⊗ S | invoked-direct | JW operators c_j = (σ^z_1 ⊗ … ⊗ σ^z_{j-1}) ⊗ σ^-_j; isotypic decomposition of representations. |

**2.C totals.** 14/14 invoked (12 direct, 1 indirect, 1 — actually, let me recount: 12.5 direct ≈ 13 direct + 1 indirect = 14/14).

**Note on 2.30, 2.32.** Catalog skips these numbers — no entries. Audit unchanged.

---

## 2.D — F_inv(E) representations on L²

| # | Operation | Verdict | Citation / sketch |
|---|---|---|---|
| 2.13 | Left regular representation L_h δ_g = δ_{hg} | invoked-direct | Adjacency A = Σ_e L_e is the canonical left-regular sum. `predictions/h_walker_eigenvalue.py`, `../theorems/theorem_bloch_lift_mu.md`. |
| 2.14 | Right regular representation R_h δ_g = δ_{gh⁻¹} | unused-deferred | See application sketch §2.14 below. (Pairs with Layer 1.7 right action, also unused-deferred.) |
| 2.15 | Adjacency operator A = Σ_{e ∈ E} L_e | invoked-direct | Substrate's defining operator. Foundational. |
| 2.16 | Self-adjointness of A | invoked-direct | A is self-adjoint because each generator e ∈ E is self-inverse, so L_e is its own adjoint and Σ L_e = (Σ L_e)*. Used everywhere. |
| 2.17 | Spectral decomposition of A | invoked-direct | Bloch decomposition of A on srs: `../theorems/theorem_bloch_lift_mu.md`. |
| 2.18 | Hashimoto operator on directed-edge space | invoked-direct | `predictions/V_cb.py`, `predictions/h_walker_eigenvalue.py`, multiple Hashimoto-walk derivations. |

**2.D totals.** 5/6 invoked-direct, 1/6 unused-deferred.

### §2.14 — Right regular representation on L²(F_inv(E)) (application sketch)

**Operation.** R_h δ_g = δ_{g h⁻¹}; equivalently the unitary representation of F_inv(E) acting on L²(F_inv(E)) by right translation.

**Application to substrate.** R_h commutes with every L_h' (left-right commutativity is standard). Together {L_h} and {R_h} generate the standard form of L(F_inv(E)) and its commutant L(F_inv(E))'. The framework currently uses only {L_h} via A = Σ L_e.

**Output candidates.**
1. **Right adjacency** A_R = Σ_e R_e. Same spectrum as A (both regular reps are unitarily equivalent), but commutes with A. The pair (A, A_R) gives joint eigenspaces — finer spectral resolution.
2. **Center of group algebra.** ℂ[F_inv(E)] has center generated by conjugacy-class sums (Layer 1.8 connection). On L², the center is Z = ⟨{L_h R_h}⟩ — sums L_h R_h are class-sum operators.
3. **Group von Neumann algebra L(F_inv(E)).** Using both {L_h, R_h} gives the standard form; useful for free-probability operations (Appendix A.5–A.9).

**Compressibility check.** Joint (A, A_R) eigenvalues are pairs (λ, λ) on the diagonal in the regular rep (eigenvalues coincide because rep is symmetric), so no new spectral content from the diagonal. Off-diagonal would arise on representations *other than* the regular one — but the framework's substrate L² *is* the regular rep, so off-diagonal joint eigenvalues don't apply.

**SM observable check.** No new SM-matching invariant emerges from this first-order analysis. The right regular rep's content overlaps with the left's; the genuinely new structure (the *commutant* L(F_inv(E))') lives in a different mathematical regime (operator-algebraic / free-probability) outside the framework's current scope.

**Verdict.** unused-deferred. The right regular rep does not yield new content via direct first-order application but is mathematically natural and would be load-bearing if the framework explores Appendix A.5 (C*_red(F_inv(E))) or A.6 (group von Neumann algebra). Defer until those Appendix items are investigated; expected to come along for free at that point.

---

## Aggregate (Layer 2)

| Status | 2.A | 2.B | 2.C | 2.D | Total |
|---|---|---|---|---|---|
| invoked-direct | 7 | 5 | 13 | 5 | 30 |
| invoked-indirect | 0 | 0 | 1 | 0 | 1 |
| invoked-negatively | 0 | 0 | 0 | 0 | 0 |
| unused-applied-negative | 0 | 1 | 0 | 0 | 1 |
| unused-deferred | 0 | 0 | 0 | 1 | 1 |
| **Layer total** | **7** | **6** | **14** | **6** | **33** |

(Catalog numbers go up to 2.37 with skips; audited entries = 33.)

**Coverage.** 33/33 catalog entries audited.

**Forward-construction docs spawned this pass.** None. The unused-applied-negative finding for 2.21 (compact operators) is a genuine pinned obstruction — F_inv(E)'s regular rep is structurally non-compact, so the operation closes as a research direction. This is a category-3 yield (pinned obstruction) per the search-instrument rubric.

---

## Honest verdict on Layer 2 sweep

**Yield categories from the rubric:**
1. New low-MDL invariant matching SM observable: **none**.
2. Cross-validation of existing prediction via distinct route: **none**.
3. Pinned obstruction: **one** (2.21 compact operators — incompatible with infinite-Cayley-graph regular rep).

The compact-operator obstruction (§2.21) is the first non-bookkeeping finding of the sweep so far. It permanently closes the research direction "apply compact-operator spectral theory to F_inv(E) substrate operators" and confirms that the framework's continuous-spectrum apparatus (spectral measures, Bloch decomposition, resolvent analysis) is the correct one. This is a clean negative result; the kind the search-instrument rubric explicitly values.

Right regular rep (2.14) and right action (Layer 1.7) consistently defer together — they pair as a single research direction, properly investigated alongside Appendix operations A.5–A.6 (operator algebras). Neither is a productive solo target.

Layer 2 confirms the Layer 0+1 pattern: most ops are invoked because they're load-bearing for the substrate's operator-algebra apparatus. The biggest layer (33 audited entries) yielded one obstruction and one doubly-deferred direction. Bookkeeping is now coverage-proven for the substrate's operator-algebra primitives.

---

## Cross-references

- `operator_sweep_from_A1.md` §Layer 2 — source catalog.
- `operator_sweep_audit_layer_0_1.md` — predecessor audit; 1.7 right action paired with 2.14 right regular rep.
- `predictions/h_walker_eigenvalue.py`, `../theorems/theorem_bloch_lift_mu.md` — primary L²(F_inv(E)) operator-theoretic predictions.
- `predictions/G2_cl2_channels.py` — Hilbert-Schmidt norm citation for 2.29.

---

## Ontology backfill (added 2026-04-26)

This audit was written before the three-lens format was adopted at Layer 5. The ontological-grounding lens is appended below.

### What Layer 2 grounds in QFT/physics ontology

Layer 2 grounds the **Hilbert-space machinery of QM/QFT**: the framework's substrate Hilbert space, operator algebras, and spectral apparatus. Direct ontology landings:

| Substrate object | Standard QFT/physics analog | Grounding |
|---|---|---|
| **L²(F_inv(E); 𝔽)** (op 2.5) | Quantum Hilbert space | Function space on substrate's Cayley graph; Hilbert-space structure inherits from counting measure (2.3). Field 𝔽 ∈ {ℝ, ℂ} until §F field-selection. |
| **Adjacency operator A = Σ_e L_e** (op 2.15) | Hamiltonian-like operator | Substrate's defining bounded self-adjoint operator. For srs at the P-point, A's spectral content underlies all framework predictions traceable to substrate dynamics. |
| **Hashimoto operator** (op 2.18) | Walk operator / propagator on graph | Non-backtracking walk operator. The framework's central spectral object after Bloch decomposition. |
| **Bounded self-adjoint observables** (ops 2.20, 2.24) | Observables postulate of QM | "Observables are self-adjoint operators on Hilbert space" — substrate provides the Hilbert space (2.5) and the operators (2.15, 2.18, etc.). |
| **Trace Tr(T)** (op 2.26) | Quantum trace; underlies characters, partition functions | Substrate trace on L²(F_inv(E)). Foundational for Layer 4.31 characters and Layer 5.34 partition function. |
| **Tensor product T ⊗ S** (op 2.37) | Multi-particle / multi-edge composition | Substrate's edge-locality forces tensor structure (each edge has its own Cl(0,2) factor). Grounds JW (5.6) and multi-particle Fock space. |
| **Spectral decomposition** (op 2.11) | Spectral theorem of QM | Bloch-decomposition realization for translation-invariant operators (4.17). |
| **Functional calculus p(T)** (op 2.31) | Operator polynomials in observables | Used for Chebyshev polynomial functions of adjacency in `predictions/R_nu_splitting.py`. |
| **Resolvent (λ−T)⁻¹** (op 2.33) | Green's function; propagator | Substrate Green's function on graph. Underlies Ihara-Bass route to V_us / V_cb predictions. |
| **Determinant det(T)** (op 2.34) | Fermion determinants in QFT path integrals | Ihara-Bass determinant identity used in `predictions/h_walker_eigenvalue.py`. |

### QFT-postulated objects this layer informs

Per `../framework/framework_qft_ontology.md`:
- **Complex Hilbert space** (§2) — Layer 2.5 provides the underlying L²(F_inv(E)); §F at Layer 5 selects ℂ.
- **Hermitian observables** (§1) — Layer 2.20–2.24 operator classes; self-adjointness forced post-§F by register-storability.
- **Multi-particle Fock space** (§2) — Layer 2.36–2.37 tensor products; substrate edge-locality.
- **Spectral content / energy eigenstates** (§1) — Layer 2.11, 2.17 spectral apparatus.

### Per-op ontology — non-trivial entries

**§2.21 compact operators (pinned obstruction).** **Substrate:** structurally absent. F_inv(E)'s regular representation is *non-compact* (infinite group, factor representation, continuous spectrum). **QFT ground:** standard QFT operators on infinite-volume Hilbert spaces also fail compactness; the obstruction is universal. Confirms the framework's continuous-spectrum apparatus (spectral measures, Bloch decomposition) is the correct one — a category-3 yield (pinned obstruction) per the rubric.

**§2.14 right regular representation (unused-deferred).** **Substrate:** commuting second copy of the regular rep; pairs with Layer 1.7 right action. **QFT ground:** would underlie operator-algebraic / free-probability formulations (Appendix A.5, A.6, A.8). Currently displaced; defer to Tier 1 cluster investigation.

**§2.35 algebraic vs §2.36 Hilbert tensor product (invoked-indirect distinction).** **Substrate:** the framework uses the Hilbert (completed) tensor product implicitly; the algebraic-vs-completion distinction isn't called out but is mathematically necessary for infinite-dim factors. **QFT ground:** standard QFT also blurs the distinction; the framework inherits this ambiguity.

---

## Status

Layer 2 audit complete with ontology backfill. Cumulative running total: 50 ops audited (4 + 13 + 33). Cumulative findings: 1 pinned obstruction (2.21 compact operators), 3 deferred (1.7, 1.8, 2.14), 0 new SM-match predictions, 0 cross-validations. Next: Layer 3 (continuous-time operations, ~13 ops including Stone's theorem and continuum-limit machinery — likely mostly invoked, with possible unused entries in the Cayley transform 3.9 and discrete-to-continuous limit details).
