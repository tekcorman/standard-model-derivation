# Operator Sweep Audit — Layers 0 and 1

**Date:** 2026-04-26.
**Status:** Per-operation audit. First layer-by-layer execution of the operation-constructor workstream.
**Source catalog:** `operator_sweep_from_A1.md`.

## Methodology

For each operation #X.Y in the layer:

- **Verdict** is one of:
  - **invoked-direct** — framework uses this operation explicitly in a prediction or theorem
  - **invoked-indirect** — operation is implicitly required by an invoked higher-layer construct
  - **invoked-negatively** — operation appears in a structural-elimination argument (proving non-existence)
  - **unused-applied-positive** — applied per the handoff template, produces a new SM-matching prediction
  - **unused-applied-derivable** — applied, produces compressible internal structure but no immediate SM match
  - **unused-applied-negative** — applied, produces no compressible new content
  - **unused-deferred** — application non-trivial, deferred to a focused future investigation
- **Citation** is a representative prediction/theorem doc using the op (for invoked) or a brief application sketch (for unused).

The user's objective is exhaustive level coverage: every op gets a verdict, including trivially invoked ones, so the audit doubles as a coverage proof.

---

## Layer 0 — Primitives from A1 alone

All four ops are required by every framework derivation that touches the substrate at all. Coverage is trivially complete; entries below cite a representative use rather than enumerating the full set.

| # | Operation | Verdict | Citation / sketch |
|---|---|---|---|
| 0.1 | Identity element id | invoked-direct | Empty word ε ∈ F_inv(E); base point in `predictions/h_walker_eigenvalue.py`, `predictions/srs_E_at_P.py`. Used as the identity in every operator algebra (Layer 2.20–2.25 require it). |
| 0.2 | Generator application T_e | invoked-direct | Defining act of A1. `predictions/p_toggle.py` derives p_create = 1/2, p_destroy = 1/3 directly from generator-application rates on a fresh edge. |
| 0.3 | Sequential composition o₁ ∘ o₂ | invoked-direct | Walker dynamics in `proofs/masses/lindblad_steady_state_at_P.py`; reduced-word formation in `predictions/V_cb.py`. |
| 0.4 | Involutive cancellation T_e ∘ T_e ↦ id | invoked-direct | A1 statement (`../framework/framework_axioms.md` §2). Drives reduced-word uniqueness via Serre 1980 (cited Layer 1 permission source). |

**Layer 0 totals.** 4/4 invoked-direct. No unused ops.

---

## Layer 1 — Group structure F_inv(E)

| # | Operation | Verdict | Citation / sketch |
|---|---|---|---|
| 1.1 | Group element g ∈ F_inv(E) | invoked-direct | Walker state as reduced word in `../theorems/theorem_multiway_branch_measure.md`; loop-element labeling in `predictions/V_cb.py`. |
| 1.2 | Group multiplication g · h with reduction | invoked-direct | Winding-series concatenation in `predictions/V_cb.py` (cycle composed with itself L times); BFS step composition in `proofs/flavor/vcb_hashimoto_bfs.py`. |
| 1.3 | Group inverse g⁻¹ | invoked-direct | Reverse-word argument in `../theorems/theorem_car_local_jordan_wigner.md` (A4 from JW + ordering); also load-bearing for distance op (1.13). |
| 1.4 | Group identity ε | invoked-direct | Cayley-graph base point P in `predictions/h_walker_eigenvalue.py`, `predictions/srs_E_at_P.py`. Appears as k = 0 mode in Bloch decomposition (Layer 4.17). |
| 1.5 | Powers g^n for n ∈ ℤ | invoked-direct | Girth-cycle iterations: V_cb winding series Σ_{n≥1} (2/3)^{8n} = (2/3)^8 / (1 − (2/3)^8) is the n-th power of the cycle-of-length-8 generator. `predictions/V_cb.py`; an internal note. |
| 1.6 | Left action g ↦ h · g | invoked-direct | Adjacency operator A = Σ_{e ∈ E} L_e is the canonical sum of left-regular generators. `predictions/h_walker_eigenvalue.py`, `predictions/srs_E_at_P.py`, `../theorems/theorem_bloch_lift_mu.md`. |
| 1.7 | Right action g ↦ g · h | unused-deferred | See application sketch §1.7 below. |
| 1.8 | Conjugation g ↦ h · g · h⁻¹ | unused-deferred | See application sketch §1.8 below. |
| 1.9 | Subgroups, cosets | invoked-direct | Pati-Salam SU(2)_L × SU(2)_R subgroup structure in `proofs/gauge/k4_pati_salam_cl8.py`; color/generation coset analysis in `memory/project_color_generation_choke_point_2026-04-25.md`. |
| 1.10 | Quotient F_inv(E)/N for normal N | invoked-negatively | `proofs/foundations/observer_hilbert_space_construction.py` uses the abelianization F_inv(E) → (ℤ/2)^{|E|} to show no 1-dim representation is faithful (commutators are non-trivial in F_inv but identity in the abelianization). Constructive use of F_inv(E)/N is not currently invoked. |
| 1.11 | Cayley graph (nodes F_inv(E), edges single-generator applications) | invoked-direct | Substrate is the Cayley graph of F_inv(E) with |E| = 6 (the 6 undirected edges of the srs primitive cell); k* = 3 is the regularity of the srs quotient, not |E|. Every framework prediction tracing to substrate structure uses this. |
| 1.12 | Word length ℓ(g) ∈ ℕ | invoked-direct | Girth length L = 8 in V_cb winding series (`predictions/V_cb.py`); waterline truncation by description length. |
| 1.13 | Cayley-graph distance d(g, h) = ℓ(g⁻¹ h) | invoked-direct | BFS from P in `proofs/flavor/vcb_hashimoto_bfs.py`; girth = shortest cycle distance in `predictions/g_girth.py`. |

**Layer 1 totals.** 11/13 invoked (10 direct + 1 negatively); 2/13 unused-deferred (1.7, 1.8).

### §1.7 — Right action g ↦ g · h on F_inv(E) (application sketch)

**Operation.** Define R_h: f(g) ↦ f(g · h⁻¹) on functions over F_inv(E). For self-inverse generators (T_e = T_e⁻¹), each R_e is involutive.

**Application to substrate.** L_h and R_h commute as operators on L²(F_inv(E)) (standard for any group's regular representations). The pair generates the standard form of the group von Neumann algebra: {L_h} generate L(F_inv(E)); {R_h} generate the commutant L(F_inv(E))'.

**Output.** A commuting second copy of the regular representation. On each isotypic component the spectrum of A_R := Σ_e R_e equals σ(A_L) (regular rep is invariant under left/right swap). The new content is the *commuting-pair structure*, not new spectral content.

**Compressibility check.** A commuting symmetry refines spectral decomposition into joint eigenspaces; it organizes degeneracies but does not by itself produce new low-MDL invariants.

**SM observable check.** Framework's existing Hashimoto operator on directed edges does not factor as L_e or R_e but as a non-backtracking variant. Right action could in principle support a "right-Hashimoto" formulation — interesting only if it produces new invariants distinct from the existing left version.

**Verdict.** unused-deferred. Worth a focused investigation in a later session; not an immediate yield. No `docs/forward_construction_*.md` spawned at this audit pass.

### §1.8 — Conjugation g ↦ h · g · h⁻¹ on F_inv(E) (application sketch)

**Operation.** Inner automorphism c_h(g) = h · g · h⁻¹ partitions F_inv(E) into conjugacy classes.

**Application to substrate.** F_inv(E) is the free product *_{e ∈ E} ℤ/2. Conjugacy classes of free products of finite groups are well-studied: a non-trivial reduced word's conjugacy class is the set of all its cyclic rotations; the trivial element forms a class by itself. Each conjugacy class of length-L reduced words has at most L elements (cyclic rotations), often fewer if the word has internal periodicity.

**Output.** A combinatorial structure on reduced words: cyclically-reduced words form representatives, and conjugation organizes the closed-loop space (cycles based at any vertex) by base-point-shift equivalence. Class sums Σ_{g ∈ C} L_g lie in the center of the group algebra ℂ[F_inv(E)].

**Compressibility check.** Conjugacy-class structure is intrinsic — closed loops are tagged by their cyclically-reduced representative without needing a base point. This *could* compress the framework's loop-counting (currently parametrized by base-point + reduced-word direction).

**SM observable check.** The framework's winding-series predictions (V_cb, V_us, dark correction 5/12) count cycles through a base point. Conjugation invariance would re-tag these as base-point-free. Whether this yields new structural input requires detailed work — the existing winding-series predictions already match observation, so the question is whether conjugation produces *additional* compressible content beyond the cycle-counting view.

**Verdict.** unused-deferred. Conjugation is mathematically natural and likely to organize Layer 4 representation theory (4.30+) and group-cohomology applications (Appendix A.1) more cleanly. Defer to a focused investigation; possibly bundled with A.1 group cohomology since H^1(G; ℤ) and H^2(G; ℤ) for free products use class structure directly.

---

## Aggregate

| Status | Layer 0 | Layer 1 | Total |
|---|---|---|---|
| invoked-direct | 4 | 10 | 14 |
| invoked-negatively | 0 | 1 | 1 |
| unused-deferred | 0 | 2 | 2 |
| unused-applied-positive | 0 | 0 | 0 |
| unused-applied-derivable | 0 | 0 | 0 |
| unused-applied-negative | 0 | 0 | 0 |
| **Layer total** | **4** | **13** | **17** |

**Coverage.** 17/17 catalog entries audited. 15/17 invoked by existing framework work; 2/17 deferred for focused future investigation.

**Forward-construction docs spawned this pass.** None — both Layer 1 unused ops (1.7, 1.8) deferred at sketch level; their compressibility evaluation is non-trivial and warrants a dedicated session each before producing standalone derivation docs.

---

## Honest verdict on Layer 0+1 sweep

Layer 0 is trivially complete (any framework derivation invokes all four primitives). Layer 1 is mostly invoked-direct; the two unused operations (right action, group conjugation) are mathematically natural and likely productive but not low-hanging fruit — first-pass application produced no immediate SM-matching prediction nor obviously new internal structure. Both deferred.

Layer 0+1 has limited yield as a search instrument because the framework's substrate IS the Cayley graph of F_inv(E), so essentially every Layer 0/1 op is structurally needed by the substrate definition itself. Higher layers (2 — operator algebra; 4 — probability/info-theory/harmonic analysis; 5 — quantum/complex-Hilbert structure) are where unused-but-permitted operations are more numerous and the search-instrument framing is more likely to produce new content.

---

## Cross-references

- `operator_sweep_from_A1.md` — source catalog, particularly §Layer 0, §Layer 1.
- `../theorems/theorem_multiway_branch_measure.md` — branch measure on F_inv(E) words.
- `../theorems/theorem_bloch_lift_mu.md` — left-regular Bloch decomposition.
- `../theorems/theorem_car_local_jordan_wigner.md` — group-inverse argument in A4 derivation.
- `proofs/foundations/observer_hilbert_space_construction.py` — abelianization (1.10) used in negative form.

---

## Ontology backfill (added 2026-04-26)

This audit was written before the three-lens format was adopted at Layer 5. The ontological-grounding lens is appended below.

### What Layer 0+1 grounds in QFT/physics ontology

Layer 0+1 ops are the substrate's *load-bearing primitives*. Every QFT-postulated object eventually traces back to substrate primitives via downstream layers, but the direct ontology landings at this layer are:

| Substrate object | Standard QFT/physics analog | Grounding |
|---|---|---|
| **Toggle T_e** (op 0.2) | Generator-of-symmetry / spin-flip / qubit operation | Substrate primitive (A1). Postulate-free; the framework's structural axiom *is* "things exist as toggles". |
| **Sequential composition** (op 0.3) | Operator product in QFT | Substrate-primitive associativity. |
| **Involutive cancellation T_e ∘ T_e = id** (op 0.4) | Discrete spin-1/2 operator algebra; σ² = I | The framework's input that grounds the entire CAR algebra of fermions (via JW at Layer 5.6). The reason fermions anti-commute traces all the way back to *this* op. |
| **Group element g ∈ F_inv(E)** (op 1.1) | Substrate state / configuration | Reduced word in F_inv(E); the substrate's intrinsic state-label. |
| **Word length ℓ(g)** (op 1.12) | Process complexity / event count | Number of toggles; underlies waterline truncation, V_cb winding-series counting. |
| **Cayley graph** (op 1.11) | Spacetime / configuration manifold | Substrate's Cayley graph IS the discrete pre-image of spacetime. The framework's continuum-limit closure (§C) maps this to a smooth Lorentzian manifold (partial). |
| **Cayley-graph distance d(g,h)** (op 1.13) | Geodesic distance | Discrete metric; for srs (k=3) realizes graph geodesics that ground Layer 6.15 graph-geodesics. |

### QFT-postulated objects this layer informs

Layer 0+1 doesn't directly ground specific QFT operators (those land at Layers 5–6). But it provides the structural pre-image:

- **Fermion anti-commutation** ← involutivity 0.4 (via JW at 5.6/5.7)
- **Spacetime manifold** ← Cayley graph 1.11 (via continuum limit at §C)
- **Geodesic structure of GR** ← graph distance 1.13 (via 6.15)
- **Process counting in QFT amplitudes** ← word length 1.12 (via waterline-MDL framework)

### Layer 0+1 ontology contribution to the meta-doc

Per `../framework/framework_qft_ontology.md`:
- Foundational input to **CAR algebra** entry (§1)
- Foundational input to **Spacetime manifold / Lorentzian signature / causal structure** entries (§7)
- Foundational input to **Geodesics** entry (§7)

### Per-op ontology — unused-deferred entries

**§1.7 right action g↦g·h.** **Substrate:** a commuting copy of the regular representation; substrate symmetry algebra has L_h and R_h commuting. **QFT ground:** would underlie group von Neumann algebra structure (Appendix A.6); appears in Tomita-Takesaki modular theory. Not directly grounded yet.

**§1.8 conjugation g↦h·g·h⁻¹.** **Substrate:** inner automorphism of F_inv(E); organizes closed loops by base-point-shift equivalence. **QFT ground:** class-function machinery underlying characters and rep theory; loop-counting in winding-series predictions. Indirectly used (winding series counts cycles), but conjugation as a standalone op isn't isolated.

---

## Status

Layer 0+1 audit complete with ontology backfill. Next layer in the sweep: Layer 2 (~37 ops, biggest single layer; mix of operator-algebra primitives and F_inv(E)-representation-on-L² content).
