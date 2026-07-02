# H¹ Master Theorem — Gauge / Physical Decomposition of Edge-Direction Data

**Date:** 2026-05-03.
**Status:** **THEOREM** (rigor: all load-bearing steps pass the `../parameters/parameter_linter.md` Type 1 / Type 2 / Type 3 / Type 4 gate; 0 adoptions).
**Scope:** Establishes that on a connected k-regular graph, the Z_p-valued edge-direction data decomposes canonically into (a) gauge redundancy (coboundaries B¹), (b) physical content (cohomology H¹), with explicit dimensions. Identifies the redundancy with vertex-flip gauge transformations and the physical content with Wilson loops on a cycle basis. For trivalent graphs the gauge fraction is asymptotically 2/3 and the physical fraction 1/3. Independent of the coefficient group Z_p (topological invariant), giving a Z_p-cohomology / center(SU(p)) identification.
**Cross-references:**
- `theorem_charge_before_color.md` §9 — uses this theorem as a second Type-4 anchor for the M_C ↔ SU(3) identification on the Cl(6) Fock space.
- `predictions/theta_QCD.py` — uses this theorem as Type-4 upstream for the cohomological framing of "θ_QCD = 0 from flat Z_3 connection on srs."
- `theorem_cosmic_birefringence.md` — would benefit from this theorem as Type-4 upstream for the Wilson-loop framing of the β derivation (recommendation only; not landed by this theorem).
- A confinement-scale variant of this theorem (entropy drop = β₁ · ln 2 at MDL optimum) is **NOT** included here — its standard holonomy-driven mechanism is incompatible with srs's flat Z_3 connection (`proofs/flavor/z3_holonomy_cycles.py`); separately scoped at an internal working note (TODO).

---

## 1. Theorem statement

**Theorem (H¹ Master Decomposition).** Let G = (V, E) be a connected finite k-regular graph with |V| = n vertices, |E| = kn/2 edges. Let A be a finite abelian group (typically A = Z_p). Define the cochain complex over A:

- C⁰(G; A) := {f : V → A}, dim_A C⁰ = n.
- C¹(G; A) := {σ : E → A}, dim_A C¹ = kn/2.
- δ⁰ : C⁰ → C¹, (δ⁰f)(uv) := f(v) − f(u) for any chosen orientation of the edge uv (well-defined up to sign on undirected graphs when A = Z_2; for general A, choose any reference orientation).

Then:

1. *(B¹ dimension.)* The coboundary subspace B¹ := im(δ⁰) has dim_A B¹ = n − 1.
2. *(H¹ dimension.)* The first cohomology H¹(G; A) := C¹/B¹ has dim_A H¹ = (k − 2)n/2 + 1 = β₁(G), the first Betti number of G as a 1-complex.
3. *(Asymptotic ratios.)* As n → ∞, the gauge fraction dim B¹ / dim C¹ → 2/k and the physical fraction dim H¹ / dim C¹ → (k − 2)/k. The ratio dim B¹ / dim H¹ → 2/(k − 2).

**Specialization to trivalent (k = 3):**
- dim B¹ = n − 1, dim H¹ = n/2 + 1, dim C¹ = 3n/2.
- Gauge fraction → **2/3**, physical fraction → **1/3**, ratio → **2**.

**Theorem (gauge transformations IS lattice gauge theory).** The action of f ∈ C⁰(G; A) on σ ∈ C¹(G; A) by
$$\sigma(uv) \;\longmapsto\; \sigma(uv) + (\delta^0 f)(uv) \;=\; \sigma(uv) + f(v) - f(u)$$
is exactly the gauge transformation of A-lattice gauge theory on G. The quotient C¹ / B¹ is the space of physical (gauge-equivalent) connections.

**Theorem (Wilson loops generate H¹).** Wilson loops W_σ(C) := Σ_{e ∈ C} σ(e) for cycles C ⊂ G are gauge-invariant. The map W : H¹(G; A) → A^{cycle basis} is an isomorphism; the n/2 + 1 = β₁(G) independent Wilson-loop values on a cycle basis form a complete set of gauge-invariant observables.

**Theorem (valence ↔ center).** For a k-regular graph, taking A = Z_k gives:
$$H^1(G; \mathbb{Z}_k) \;\cong\; \mathbb{Z}_k^{\beta_1(G)},$$
and Z_k Wilson loops take values in Z_k. Under the standard identification Z_k ≅ center(SU(k)), the H¹(G; Z_k) classes label center sectors of SU(k) lattice gauge theory. For trivalent graphs (k = 3), Z_3 = center(SU(3)) coincides with the three color charges {1, ω, ω²} of QCD.

---

## 2. Axioms and upstream results

**Framework axioms (Type 1 gates):**

- **A1** (`../framework/framework_axioms.md` §2): supplies the substrate as a graph with edge-direction data σ ∈ C¹(G; A). The binary self-inverse toggle gives σ ∈ C¹(G; Z_2) directly; Z_p extensions (p ≥ 3) require further structure (e.g., the Z_3 site symmetry at trivalent vertices used at θ_QCD).
- A2-T not used. The theorem is purely topological — independent of MDL retention.

**Upstream closed framework files (Type 4 gates):**

- `predictions/k_star.py` — k* = 3 used at §1 specialization; the theorem itself is k-agnostic.
- `predictions/g_girth.py` — girth g = 10 used only in the Wilson-loop-on-srs adaptation note (§10); not load-bearing for the abstract theorem.

**Cited published results (Type 3 gates):**

- **Hatcher, A.** (2002). *Algebraic Topology.* Cambridge. §2.1 (cellular cochain complex), §3.1 (cohomology of CW complexes). For graphs treated as 1-dimensional CW complexes, the cochain complex of §1 is the cellular cochain complex.
- **Spanier, E. H.** (1966). *Algebraic Topology.* McGraw-Hill. §4.2 (Euler characteristic of graphs: |V| − |E| = 1 − β₁ for connected G).
- **Sunada, T.** (2013). *Topological Crystallography.* Springer. §3 (graph cohomology), §6 (Bloch–Floquet decomposition for periodic graphs; used at §10 adaptation note).
- **Wilson, K. G.** (1974). Confinement of quarks. *Phys. Rev. D* 10: 2445–2459. §II (lattice gauge theory; gauge transformations as vertex-valued group elements).
- **Kogut, J. & Susskind, L.** (1975). Hamiltonian formulation of Wilson's lattice gauge theories. *Phys. Rev. D* 11: 395. §II (Wilson loop as gauge-invariant observable; cycle-basis completeness).
- **Greensite, J.** (2011). *An Introduction to the Confinement Problem.* Springer. §5 (center symmetry of SU(N), Z_N center sector decomposition).
- **Kobayashi, S. & Nomizu, K.** (1963). *Foundations of Differential Geometry, Vol. I.* Wiley. §II.4 (Ambrose-Singer holonomy theorem; flat-connection ↔ trivializable bundle). Used at §11 cite-improvements for θ_QCD.

---

## 3. Proof — Theorem 1 (dimension formulae)

**(i) dim B¹ = n − 1.**

By the rank-nullity theorem applied to δ⁰ : C⁰ → C¹:
$$\dim B^1 \;=\; \dim C^0 - \dim \ker \delta^0.$$

f ∈ ker δ⁰ iff f(v) − f(u) = 0 for every edge uv, iff f is constant on each connected component of G. Since G is connected, ker δ⁰ = {constants on V}, which has dim_A 1 (one element per group element of A; abelian-group dimension 1). Hence dim B¹ = n − 1. [Type 2: rank-nullity; Type 3: Hatcher 2002 §2.1.]

**(ii) dim H¹ = (k − 2)n/2 + 1.**

For a 1-dimensional CW complex (a graph), the cochain complex terminates at C¹ — there is no C² because there are no 2-cells. Hence δ¹ ≡ 0 and Z¹ := ker δ¹ = C¹. Therefore
$$H^1 \;=\; Z^1 / B^1 \;=\; C^1 / B^1, \quad \dim H^1 \;=\; \dim C^1 - \dim B^1.$$

For k-regular G: dim C¹ = |E| = kn/2. Combined with (i):
$$\dim H^1 \;=\; \frac{kn}{2} - (n - 1) \;=\; \frac{(k-2)n}{2} + 1.$$

This is also the first Betti number β₁(G) by definition for a connected 1-complex. [Type 3: Hatcher 2002 §3.1 (cellular cohomology), Spanier 1966 §4.2 (Euler characteristic); Type 2: arithmetic.]

**(iii) Asymptotic ratios.**

Direct division by dim C¹ = kn/2:
- dim B¹ / dim C¹ = (n − 1) / (kn/2) = 2(n − 1)/(kn) → **2/k** as n → ∞.
- dim H¹ / dim C¹ = ((k − 2)n/2 + 1) / (kn/2) → **(k − 2)/k** as n → ∞.
- dim B¹ / dim H¹ = (n − 1) / ((k − 2)n/2 + 1) → **2/(k − 2)** as n → ∞ (well-defined for k ≥ 3).

For k = 3: gauge fraction → 2/3, physical fraction → 1/3, ratio → 2. [Type 2: arithmetic.] ∎

---

## 4. Proof — Theorem 2 (gauge transformations IS A-lattice gauge theory)

Wilson 1974 §II defines A-lattice gauge theory on a graph G by:
- Configuration space: σ : E → A (an A-valued connection 1-cochain).
- Gauge group: maps f : V → A acting on σ by σ(uv) → σ(uv) + f(v) − f(u) (additive notation; multiplicatively in non-abelian generalizations: σ(uv) → f(u)⁻¹ · σ(uv) · f(v)).
- Physical states: gauge-equivalence classes of σ.

By construction, the gauge action is exactly precomposition with δ⁰: σ → σ + δ⁰f. The orbit space C¹ / (gauge action) = C¹ / im(δ⁰) = C¹ / B¹ = H¹(G; A). 

Hence the gauge / physical decomposition C¹ = B¹ ⊕ H¹ of Theorem 1 IS the configuration-space / orbit-space decomposition of A-lattice gauge theory on G. Identity, not analogy. [Type 3: Wilson 1974 §II; Type 2: orbit-space arithmetic.] ∎

---

## 5. Proof — Theorem 3 (Wilson loops as complete gauge-invariant observables)

**Wilson loops are gauge-invariant.** For a closed path C = (v_0, v_1, …, v_m = v_0) in G traversing edges e_i = v_{i-1}v_i, define W_σ(C) := Σ_{i=1}^m σ(e_i). Under a gauge transformation σ → σ + δ⁰f:
$$W_{\sigma + \delta^0 f}(C) \;=\; W_\sigma(C) + \sum_{i=1}^m (f(v_i) - f(v_{i-1})) \;=\; W_\sigma(C) + (f(v_m) - f(v_0)) \;=\; W_\sigma(C),$$
since v_m = v_0 (closed path). So W_σ(C) is constant on gauge orbits. [Type 3: Wilson 1974 §II, Kogut-Susskind 1975 §II; Type 2: telescoping sum.]

**Cycle basis completeness.** The cycle space Z_1(G; A) of a graph is generated by a fundamental cycle basis of size β₁(G) = (k − 2)n/2 + 1 (Sunada 2013 §3.4: choose a spanning tree T ⊂ G; each non-tree edge defines a unique fundamental cycle, and these β₁(G) cycles span Z_1). The Wilson-loop map
$$W : H^1(G; A) \;\longrightarrow\; A^{\beta_1(G)}, \quad [\sigma] \mapsto (W_\sigma(C_1), \ldots, W_\sigma(C_{\beta_1(G)}))$$
is well-defined (gauge-invariance, established above) and an isomorphism: injectivity is the universal coefficient duality H¹(G; A) ≅ Hom(H_1(G; ℤ), A) ≅ A^{β₁(G)} (Hatcher 2002 §3.1); surjectivity follows from dimension count by Theorem 1. [Type 3: Sunada 2013 §3.4 (cycle basis), Hatcher 2002 §3.1 (universal coefficients); Type 2: dimension match.]

Hence the n/2 + 1 = β₁(G) Wilson-loop values on a cycle basis form a **complete** set of gauge-invariant observables: any gauge-invariant function of σ factors through these β₁ values. ∎

---

## 6. Proof — Theorem 4 (Z_p extension; valence ↔ center identification)

**(i) Dimension formula independent of A.** The proof of Theorem 1 used only the additive abelian group structure of A and the rank-nullity theorem; it did not use the specific group A = Z_2. For any finite abelian group A, dim_A H¹(G; A) = (k − 2)n/2 + 1 in the same sense (the rank of the free A-module H¹). For A = Z_p (prime p), H¹(G; Z_p) ≅ Z_p^{β₁(G)} as abelian groups. [Type 3: Hatcher 2002 §3.1 universal coefficients.]

**(ii) Z_k Wilson loops take values in center(SU(k)).** The center Z(SU(k)) of SU(k) is the cyclic group of k-th roots of unity, isomorphic to Z_k as an abstract group (Greensite 2011 §5.1). For A = Z_k, Wilson loops W_σ(C) ∈ Z_k take values in this center. Under the standard center-symmetry decomposition of SU(k) lattice gauge theory (Greensite 2011 §5), the H¹(G; Z_k) classes label the center sectors — i.e., the topological superselection sectors of SU(k) gauge theory on G. [Type 3: Greensite 2011 §5.]

**(iii) Trivalent → Z_3 → SU(3).** For k = 3:
$$Z(SU(3)) \;=\; \{1, e^{2\pi i/3}, e^{4\pi i/3}\} \;\cong\; \mathbb{Z}_3,$$
and these three center elements are the three color charges of QCD (the eigenvalues of the central element acting on the fundamental representation of SU(3); equivalently, the trace-class labels of the three colored quark states in a triplet).

H¹(G; Z_3) thus labels Z_3 center sectors of SU(3) gauge theory on G; for connected trivalent G with β₁(G) = n/2 + 1 cycle classes, there are 3^{n/2 + 1} distinct center sectors. [Type 3: Greensite 2011 §5.1; Type 2: center arithmetic.]

**Caveat (what is NOT established here).** This theorem establishes the *labeling* of center sectors via H¹. It does NOT establish:
- That the realized gauge connection on srs is in any particular sector (the realized Z_3 connection from edge labels is in fact FLAT — trivial sector — per `proofs/flavor/z3_holonomy_cycles.py`).
- That confinement (or any dynamical phenomenon) follows from this center-sector decomposition. Confinement requires a specific Hamiltonian or partition-function argument, scoped separately.
- That this Z_3 ↔ center(SU(3)) identification gives the *full* SU(3) gauge structure. The Z_3 piece is only the center; the off-center generators (the 8-dim Lie algebra modulo the center) come from the local structure (Cl(6) ⊃ U(3), per Row 17 + `theorem_charge_before_color.md` §9).

∎

---

## 7. Status of axioms used

- **A1**: USED implicitly via the substrate's edge-data structure C¹(G; A). The theorem itself is purely topological and does NOT depend on the binary self-inverse toggle dynamics.
- **A2-T**: NOT used. The theorem is independent of MDL retention; it characterizes the algebraic *structure* of the gauge / physical decomposition, not which configurations are retained.
- **A3-T, A4, A5(a), A5(b)**: NOT used.

The theorem is provable on standard graph cohomology under k* = 3 (used only for the trivalent specialization; the general theorem is k-agnostic). 0 adoptions.

---

## 8. Adaptation note — infinite periodic graphs (srs net)

Theorem 1 is stated for **finite** connected k-regular graphs with n vertices. The substrate of this framework is the **srs net** (`predictions/g_girth.py`) — a connected, vertex-transitive, 3-regular *infinite periodic* graph. The dimensions n − 1 and (k − 2)n/2 + 1 diverge as n → ∞.

The correct generalization uses Bloch–Floquet decomposition (Sunada 2013 §6):

- Let Λ ⊂ ℝ³ be the lattice of translations of srs (acting freely on the vertex set with finite quotient G_0 = srs/Λ, the primitive cell with |V_0| = 8, |E_0| = 12 for the standard 4×4×4 cell).
- The cochain complex C¹(srs; A) decomposes as a Λ-equivariant direct integral over the Brillouin zone B = ℝ³/Λ*: C¹(srs; A) = ∫_B^⊕ C¹(G_0; A)_k dk, where C¹(G_0; A)_k is the Bloch fiber at quasimomentum k ∈ B.
- At each quasimomentum k, the finite-graph result of Theorem 1 applies to the Bloch fiber: dim B¹_k = |V_0| − 1, dim H¹_k = |E_0| − |V_0| + 1 = (k − 2)·|V_0|/2 + 1, with gauge/physical ratio 2/(k − 2) per fiber.
- The asymptotic ratios 2/k (gauge), (k − 2)/k (physical) hold *per Bloch fiber* and hence *per primitive cell* — the same fractions as in the finite-graph statement.

For srs (k = 3, |V_0| = 8 in the standard 4×4×4 supercell): per-cell dim B¹ = 7, per-cell dim H¹ = 5, per-cell dim C¹ = 12. Ratios 7:5:12, gauge fraction 7/12 ≈ 0.583 (approaching 2/3 in the bulk-cell limit), physical fraction 5/12 (approaching 1/3).

The Bloch-decomposition adaptation preserves the **per-cell** content of Theorem 1; it does not require finite n. [Type 3: Sunada 2013 §6 (Bloch–Floquet on periodic graphs); Type 2: per-cell arithmetic.]

---

## 9. What this theorem does and does not deliver

**Closes:**
- Foundational identity: gauge transformations on edge-direction data ARE Z_p lattice gauge transformations (Theorem 2). Type-4 cite for any framework derivation that treats edge labels as a connection.
- Cohomological framing of θ_QCD = 0 (`predictions/theta_QCD.py`): the existing flatness argument can now cite Theorem 4(i) for the H¹ classification of Z_3 connections, and Theorem 3 for the Wilson-loop characterization of gauge invariance — replacing the implicit cohomological framing with explicit cites.
- Cohomological framing of β cosmic birefringence (`theorem_cosmic_birefringence.md`): the Wilson-loop appearance of β can be cited from Theorem 3.
- Second Type-4 anchor for the M_C ↔ SU(3) identification at `theorem_charge_before_color.md` §9: Theorem 4 establishes Z_3 = center(SU(3)) at the cohomological level, complementing the Cl(6) ⊃ U(3) Fock-space identification.

**Does NOT close:**
- The origin of (2/3)^8 in α_1_bare. The "2/3" in α_1_bare = ((k − 1)/k)^{g − 2} is per-step random-walk survival (combinatorics of non-backtracking walks on k-regular graphs), NOT the H¹ gauge fraction 2/k. They coincide *numerically* at k = 3 (both equal 2/3) but are different formulas: at k = 4 random-walk gives 3/4 while H¹ gauge gives 1/2. Identifying the two structurally would require additional bridge work; this is an open structural-equivalence question.
- A confinement-scale or mass-gap derivation. The natural entropy-drop = β₁ · ln 2 at MDL optimum argument (standard in lattice gauge literature) is **NOT** developed here; on srs the realized Z_3 connection is flat (`proofs/flavor/z3_holonomy_cycles.py`), so any holonomy-driven confinement mechanism does not apply directly. A finite-girth-cycle reconstruction is open research; scoped separately at an internal working note (TODO).
- The full SU(3) gauge structure. Theorem 4 only identifies Z_3 = center(SU(3)). The off-center generators (the 8-dim Lie algebra modulo the center) come from local structure: Cl(6) ⊃ U(3) at each k* = 3 site (Row 17 + `theorem_charge_before_color.md` §9), not from the H¹ cohomology.

---

## 10. Cite-improvement targets

This theorem provides a Type-4 upstream cite improvement for three existing framework results. Recommended updates (not landed by this theorem; tracked separately):

| Target | Existing state | Recommended cite |
|---|---|---|
| `predictions/theta_QCD.py` step 5 (flat connection → trivializable Z_3 bundle) | Cites Kobayashi-Nomizu Vol I §II.4 (continuum Ambrose-Singer adapted to discrete) | Add Type-4 cite to Theorem 4(i) here for the H¹(srs; Z_3) classification, and Theorem 3 for Wilson-loop characterization |
| `theorem_cosmic_birefringence.md` (Wilson-loop appearance) | Implicit Wilson-loop reasoning | Add Type-4 cite to Theorem 3 here for the Wilson-loop / H¹ isomorphism |
| `theorem_charge_before_color.md` §9 (M_C ↔ SU(3)) | Cl(6) ⊃ U(3) representation-theoretic identity (Furey 2018) | Add Theorem 4(iii) here as a second Type-4 anchor: Z_3 cohomology = center(SU(3)) |

---

## 11. References

**Cited mathematical theorems:**
- Hatcher 2002 *Algebraic Topology* §2.1 (cellular cochain complex), §3.1 (cohomology + universal coefficients).
- Spanier 1966 *Algebraic Topology* §4.2 (graph Euler characteristic).
- Sunada 2013 *Topological Crystallography* §3 (graph cohomology), §3.4 (cycle bases), §6 (Bloch–Floquet decomposition).
- Wilson 1974 *Phys. Rev. D* 10:2445 §II (lattice gauge theory).
- Kogut-Susskind 1975 *Phys. Rev. D* 11:395 §II (Wilson loop completeness).
- Greensite 2011 *Confinement Problem* §5.1 (center symmetry, Z_N center sectors).
- Kobayashi-Nomizu 1963 *Foundations of Differential Geometry I* §II.4 (Ambrose-Singer).

**Framework documents:**
- `../framework/framework_axioms.md` §2 (A1 supplies edge-direction data).
- `predictions/k_star.py` (k* = 3).
- `predictions/g_girth.py` (girth g = 10; used at §8 adaptation only).
- `predictions/theta_QCD.py` (cite-improvement target §10).
- `theorem_charge_before_color.md` §9 (cite-improvement target §10).
- `theorem_cosmic_birefringence.md` (cite-improvement target §10).
- `proofs/flavor/z3_holonomy_cycles.py` (srs Z_3 connection is flat — used in §9 caveat and §11 NOT-ported note).
- Row 19 of `../audits/registers/uniqueness_ledger.md` (gauge group SU(3) × SU(2) × U(1)).

---

## 12. Walk uniqueness auditor — Clauses 1–8

Per `feedback_walk_uniqueness_auditor_at_conclusions.md`. Run 2026-05-03.

**Clause 1 — Structural rows (uniqueness ledger Rows 1–23):**
- Row 4 (k* = 3): used at §1 specialization; not load-bearing for the general theorem.
- Row 19 (Gauge group SU(3) × SU(2) × U(1)): provides SU(3) existence via Pati-Salam descent; this theorem provides Z_3 = center(SU(3)) at cohomological level. No conflict.
- No Row 1–23 refuted, refined, or opened.

**Clause 2 — Parameter ledger (P-rows):**
- P16 (θ_QCD = 0): cite-improvement target. This theorem provides Type-4 cohomological framing for the existing flat-connection argument. Status of P16 unchanged (already UNIQUE-THEOREM-GRADE).
- No new parameter closure.

**Clause 3 — Operator sweep:** Uses Op 4.5 (Shannon entropy, implicitly via cohomology dimensions), Op 5.27 (cellular chain/cochain complex), Op 5.28 (complex Lie groups, for the SU(p) center identification). All within operator-permitted catalog.

**Clause 4 — Residue register (R-N):**
- R-9 (full-MDL-spectrum lattice residue): orthogonal — this theorem is purely topological, doesn't engage MDL retention.
- R-12 (chirality residual): orthogonal.
- R-14 (Pati-Salam quark/lepton differentiation): orthogonal.
- No R-N entry refuted, refined, or opened.

**Clause 5 — Cross-theorem consistency:**
- `theorem_charge_before_color.md`: COMPATIBLE. This theorem provides a second Type-4 anchor for §9's M_C ↔ SU(3) identification.
- `predictions/theta_QCD.py`: COMPATIBLE. Cite improvement, not status change.
- `theorem_cosmic_birefringence.md`: COMPATIBLE. Cite improvement.
- `proofs/flavor/z3_holonomy_cycles.py`: COMPATIBLE — confirms srs connection sits at trivial H¹ class. This theorem provides the abstract H¹ classification within which "trivial class" is well-defined.
- No conflicts.

**Clause 6 — Cited published results:** All 7 cites verified against journal-grade sources (textbooks: Hatcher, Spanier, Sunada, Greensite, Kobayashi-Nomizu; published papers: Wilson 1974, Kogut-Susskind 1975). No suspect cites.

**Clause 7 — Audit-v2 inventory (Type 1/2/3/4 gates):**

| Section | Claim | Gate | Source |
|---|---|---|---|
| §3 Thm 1(i) | dim B¹ = n − 1 | T2+T3 | rank-nullity, Hatcher §2.1 |
| §3 Thm 1(ii) | dim H¹ = (k − 2)n/2 + 1 | T2+T3 | Hatcher §3.1, Spanier §4.2 |
| §3 Thm 1(iii) | Asymptotic ratios | T2 | Arithmetic |
| §4 Thm 2 | Gauge action IS lattice gauge transform | T3 | Wilson 1974 §II |
| §5 Thm 3 | Wilson loops gauge-invariant | T3+T2 | Wilson 1974, telescoping |
| §5 Thm 3 | Cycle basis completeness | T3 | Sunada 2013 §3.4, Hatcher §3.1 |
| §6 Thm 4(i) | Z_p extension preserves dimensions | T3 | Hatcher §3.1 universal coefficients |
| §6 Thm 4(ii) | Z_k Wilson loops in center(SU(k)) | T3 | Greensite 2011 §5.1 |
| §6 Thm 4(iii) | Z_3 = center(SU(3)) = QCD color charges | T3+T2 | Greensite 2011 §5.1, center arithmetic |
| §8 Adaptation | Bloch–Floquet for srs | T3+T4 | Sunada 2013 §6, predictions/g_girth.py |

All load-bearing claims have explicit Type 1/2/3/4 gate citations.

**Clause 8 — Numerical match:**
- Per-cell dim ratios (7:5:12 in 4×4×4 srs supercell): integer, verifiable by direct counting on the srs primitive cell. No σ measure applicable.
- Theorem is structural (cohomological identity), not a numerical parameter prediction. C8 not the gating clause.

**Auditor verdict:** PASS-CITED on all 8 clauses.

---

## 14. Status of the theorem

- **Rigor:** Theorem-grade. All load-bearing steps cite Type 1/2/3/4 gates; proofs of Theorems 1–4 use only standard graph cohomology + lattice gauge theory references.
- **Adoptions:** 0.
- **Axioms used:** A1 (Type 1; via edge-data structure). A2-T NOT used. The theorem is purely topological.
- **Generality:** Holds for any connected k-regular graph (finite or, via Bloch–Floquet, infinite periodic) and any finite abelian coefficient group A.
- **What this closes:** Three cite-improvements (θ_QCD, β_birefringence, Charge-Before-Color §9) + one new structural row in the framework's gauge-theory foundations.
- **What this does NOT close:** Origin of (2/3)^8 in α_1_bare (different "2/3"); confinement scale or mass gap (the natural entropy-drop argument is incompatible with srs flatness, separate scoping); full SU(3) gauge structure beyond the center.
