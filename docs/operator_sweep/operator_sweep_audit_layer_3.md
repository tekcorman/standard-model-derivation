# Operator Sweep Audit — Layer 3

**Date:** 2026-04-26.
**Status:** Per-operation audit. Layer-by-layer execution of the operation-constructor workstream.
**Source catalog:** `operator_sweep_from_A1.md` §Layer 3.
**Predecessors:** `operator_sweep_audit_layer_0_1.md`, `operator_sweep_audit_layer_2.md`.

## Layer 3 — Continuous-time operations

13 operations grouped into:
- **3.A** Continuous-time evolution (3 ops)
- **3.B** Stone's theorem and generators (6 ops)
- **3.C** Continuum limit from discrete dynamics (4 ops)

### 3.A — Continuous-time evolution

| # | Operation | Verdict | Citation / sketch |
|---|---|---|---|
| 3.1 | One-parameter unitary group U: ℝ → 𝒰(L²) | invoked-direct | `../theorems/theorem_A3_complex_hilbert_from_multiway.md` §8 (Step 5: Stone's theorem applied to substrate's continuous-time evolution). |
| 3.2 | Strong continuity of U(t) | invoked-direct | Same — strongly-continuous one-parameter group is the precondition for invoking Stone in theorem_A3. |
| 3.3 | Continuous-time quantum walks on graphs | invoked-direct | Childs 2009-style continuum walks underlie the framework's substrate dynamics; cited in operator_sweep §3. |

### 3.B — Stone's theorem and generators

| # | Operation | Verdict | Citation / sketch |
|---|---|---|---|
| 3.4 | Stone (complex form): U(t) = exp(−iHt) on ℂ-L² | invoked-direct | `../theorems/theorem_A3_complex_hilbert_from_multiway.md` §8 — load-bearing for the field-selection chain. |
| 3.5 | Stone (real form): U(t) = exp(Bt) on ℝ-L² | invoked-direct | Same theorem — both forms invoked side-by-side as the contrast that drives field selection (§F of operator_sweep). |
| 3.6 | Self-adjoint Hamiltonian H on ℂ-L² | invoked-direct | Result of Step 5 in theorem_A3; the H whose real spectrum is required by A5-mass identification. |
| 3.7 | Skew-symmetric generator B on ℝ-L² | invoked-direct | Result of Step 5 real form; the B whose imaginary spectrum is incompatible with A5-mass — drives ℂ selection. |
| 3.8 | Spectrum σ(H) ⊂ ℝ for self-adjoint H; σ(B) ⊂ iℝ for skew-symmetric B | invoked-direct | `../theorems/theorem_A3_complex_hilbert_from_multiway.md` §8-9 — the spectral asymmetry between ℝ and ℂ generators is the engine of field selection. |
| 3.9 | Cayley transform V = (H − i)(H + i)⁻¹ for unbounded self-adjoint H | unused-applied-negative | See application sketch §3.9 below. |

### 3.C — Continuum limit from discrete dynamics

| # | Operation | Verdict | Citation / sketch |
|---|---|---|---|
| 3.10 | Discrete-time quantum walk U_disc^n | invoked-direct | `predictions/H_multiway_dim_count.py`, `predictions/alpha_1.py`, `predictions/Feshbach_coupling_strength.py` (multi-step walks for waterline counting). |
| 3.11 | Discrete-to-continuous quantum walk limit on bounded-degree graphs with rapidly decaying step correlations | invoked-direct | Operator sweep §C closure — Strauch 2006 + Stage 3 rapid decay. Load-bearing for continuum-limit Hamiltonian (3.13). |
| 3.12 | Continuum-limit Hamiltonian H = adjacency-operator-type generator | invoked-direct | `../theorems/theorem_A3_complex_hilbert_from_multiway.md` §8 (continuum-limit H from discrete substrate). |
| 3.13 | Framework's specific continuum-limit Hamiltonian on F_inv(E)'s Cayley graph | invoked-direct | Substrate dynamics throughout the framework; spectral content underlies all Hashimoto-based predictions. |

---

## §3.9 — Cayley transform on substrate Hamiltonian (application sketch)

**Operation.** For self-adjoint H (possibly unbounded), V = (H − i)(H + i)⁻¹ is unitary. Spectrally: σ(V) = {(λ − i)/(λ + i) : λ ∈ σ(H)}. This maps ℝ to the unit circle minus {−1}, providing a bounded-unitary handle on unbounded H.

**Application to substrate.** Two candidate inputs:

1. **Adjacency operator A on srs** (bounded, σ(A) ⊂ [−3, 3]). Cayley transform V_A = (A − i)(A + i)⁻¹ is unitary; σ(V_A) is the image of [−3, 3] on the unit circle: an arc from (−3 − i)/(−3 + i) ≈ 0.6 + 0.8i through 1 (the image of ∞, not in σ(A)) and back. Concretely, eigenvalue λ = 0 maps to V_A = −1 (interesting boundary point).

2. **Continuum-limit Hamiltonian H_continuum** (3.13). For continuous-time quantum walk on srs with bounded degree 3, H_continuum is bounded (norm ≤ 2 deg = 6 by Childs 2009 normalization). Hence not the unbounded case the Cayley transform was designed for; apply per construction 1 above.

**Output.** V is a bounded unitary on L²(srs) whose spectrum is a coordinate-rescaling of σ(A) onto the unit circle. The spectral measure transports along the Möbius map λ ↔ (λ − i)/(λ + i); the framework's existing spectral content (Bloch fibers, Ramanujan-saturated h_max for Hashimoto) is unchanged in informational content, just re-parametrized.

**Compressibility check.** Cayley transform is a coordinate change in spectral parameter space — bijective, smooth, information-preserving. It produces no new compressible structure beyond what σ(H) already encodes.

**SM observable check.** No new SM-matching invariant emerges. The operation produces a unitary V on the substrate Hilbert space whose eigenvalue moduli are all 1 (definitionally) and whose phases are functions of σ(H). Nothing the framework hasn't already extracted from σ(H) directly.

**Why 3.9 might *seem* useful.** The Cayley transform is the natural tool when H is unbounded — but the framework's H_continuum is bounded (degree-3 substrate), so the unbounded-handling motivation does not apply. Cayley transform is sometimes useful in number-theoretic / Selberg-zeta contexts where the unit circle is more natural than ℝ; this overlaps Appendix A.18 (Selberg zeta function), so 3.9 may revive as an instrument when A.18 is investigated.

**Verdict.** unused-applied-negative. The operation is permitted but produces no new structure for the framework's bounded continuum H. Not a *structural* obstruction (Cayley transform is well-defined; it just does nothing new); soft negative finding. May revive when Selberg-zeta / modular forms (Appendix A.16, A.18) are investigated.

---

## Aggregate (Layer 3)

| Status | 3.A | 3.B | 3.C | Total |
|---|---|---|---|---|
| invoked-direct | 3 | 5 | 4 | 12 |
| unused-applied-negative | 0 | 1 | 0 | 1 |
| **Layer total** | **3** | **6** | **4** | **13** |

**Coverage.** 13/13 catalog entries audited.

**Forward-construction docs spawned this pass.** None. The unused-applied-negative for 3.9 is a soft negative (no structural obstruction; just no new content). It may revive when modular/Selberg ops are investigated.

---

## Honest verdict on Layer 3 sweep

**Yield categories from the rubric:**
1. New low-MDL invariant matching SM observable: **none**.
2. Cross-validation of existing prediction via distinct route: **none**.
3. Pinned obstruction: **none**. (Unlike Layer 2's compact-operator obstruction, the Cayley transform soft-fails — it works but produces no new content for the framework's specific bounded H.)

Layer 3 is the most concentrated layer encountered so far: 12/13 ops cite a single load-bearing theorem (`../theorems/theorem_A3_complex_hilbert_from_multiway.md`). This makes sense — Layer 3 *is* the continuous-time apparatus, and the field-selection chain at §F is precisely the place the framework cashes in this apparatus. The Cayley transform is the only Layer 3 op the framework genuinely doesn't need; it's permitted but redundant for bounded H.

---

## Cumulative through Layer 3

| Layer | Ops | invoked-direct | invoked-indirect | invoked-negatively | unused-applied-negative | unused-deferred |
|---|---|---|---|---|---|---|
| 0 | 4 | 4 | 0 | 0 | 0 | 0 |
| 1 | 13 | 10 | 0 | 1 | 0 | 2 |
| 2 | 33 | 30 | 1 | 0 | 1 | 1 |
| 3 | 13 | 12 | 0 | 0 | 1 | 0 |
| **Cumulative** | **63** | **56** | **1** | **1** | **2** | **3** |

**Headline:** 63 ops audited; 58 invoked (any flavor); 5 unused (2 applied-negative, 3 deferred); 0 SM-matching positive yields.

---

## Cross-references

- `operator_sweep_from_A1.md` §Layer 3, §C.
- `../theorems/theorem_A3_complex_hilbert_from_multiway.md` — primary citation hub for Layer 3.
- `operator_sweep_audit_layer_0_1.md`, `operator_sweep_audit_layer_2.md` — predecessor audits.

---

## Ontology backfill (added 2026-04-26)

This audit was written before the three-lens format was adopted at Layer 5. The ontological-grounding lens is appended below.

### What Layer 3 grounds in QFT/physics ontology

Layer 3 grounds the **continuous-time evolution apparatus of QM**: Schrödinger evolution, the unitary group, and the field-selection chain. **Single most concentrated layer of the catalog**: 12 of 13 ops cite a single load-bearing theorem (`../theorems/theorem_A3_complex_hilbert_from_multiway.md`).

| Substrate object | Standard QFT/physics analog | Grounding |
|---|---|---|
| **One-parameter unitary group U: ℝ → 𝒰(L²)** (3.1) | Time evolution in QM | Substrate's continuum-limit dynamics under §C closure (unitary part). |
| **Stone (complex form) U(t) = exp(−iHt)** (3.4) | Schrödinger equation | *Forced* on ℂ-L² for the substrate's continuum-limit Hamiltonian. The Schrödinger equation is derived, not postulated. |
| **Stone (real form) U(t) = exp(Bt)** (3.5) | (Hypothetical real-Hilbert dynamics) | The framework's *contrast operation*: forced to be ruled out because B's spectrum is imaginary, incompatible with finite-register storability of mass eigenvalues. Drives the ℂ field-selection at §F. |
| **Self-adjoint H on ℂ-L²** (3.6) | Hamiltonian observable of QM | Forced by Stone complex-form + register storability. |
| **Skew-symmetric B on ℝ-L²** (3.7) | (Hypothetical real-Hilbert generator) | Forced by Stone real-form; ruled out via spectrum check. |
| **Spectrum σ(H) ⊂ ℝ vs σ(B) ⊂ iℝ** (3.8) | Reality of energy eigenvalues | The asymmetry that *engines* field selection. Real spectrum requires complex generator. |
| **Continuum-limit Hamiltonian H = adjacency-type** (3.12, 3.13) | Effective Hamiltonian of substrate | Strauch 2006 + Stage 3 rapid-decay close the unitary-evolution part of §C. |
| **Cayley transform V = (H−i)(H+i)⁻¹** (3.9) | Bounded-unitary form of unbounded H | Soft-unused: framework's H_continuum is bounded (degree-3 substrate), so Cayley transform applies trivially. May revive at Appendix A.18 Selberg-zeta investigation. |

### QFT-postulated objects this layer informs

Per `../framework/framework_qft_ontology.md`:
- **Schrödinger evolution** (§3) — Layer 3.4 + ℂ field selection (§F) gives U(t) = exp(−iHt) as derived structure.
- **One-parameter unitary group** (§3) — Layer 3.1, 3.2.
- **Hermitian observables postulate** (§1) — Layer 3.6 + register-storability argument.
- **Wick rotation** (§3) — Layer 3 admits both real and complex forms; Wick rotation is the analytic continuation between them. (Direct Wick op is at 5.33.)

### Field-selection chain (§F) — central ontology landing

Layer 3's structural significance is that it *engines* the field-selection chain via the contrast between complex-form (3.4) self-adjoint H with real spectrum and real-form (3.5) skew-symmetric B with imaginary spectrum. P1' (observer is finite register) + register-storability of mass eigenvalues forces ℂ. This is one of the framework's foundational ontology landings: **the complex Hilbert space of QM is derived, not postulated.**

### Per-op ontology — unused entries

**§3.9 Cayley transform (unused-applied-negative).** **Substrate:** Möbius transformation of σ(H) onto unit circle; coordinate change with no new spectral information. **Why it doesn't apply:** the framework's H is bounded (degree-3 srs), so the unbounded-handling motivation is moot. **QFT ground:** Cayley transform is sometimes useful in number-theoretic contexts (Selberg zeta, modular structure); may revive there.

---

## Status

Layer 3 audit complete with ontology backfill. Next: Layer 4 (probability + information theory + harmonic analysis + statistical mechanics + Lie-algebra primitives, ~53 ops — the largest layer in the catalog and the one most likely to surface unused-but-permitted operations productively, since it spans ~6 distinct mathematical subfields).
