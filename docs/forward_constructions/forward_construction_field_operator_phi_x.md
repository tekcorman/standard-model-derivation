# Field Operator φ(x) — forward-construction setup (Tier 3)

**Date:** 2026-04-26.
**Status:** Forward-construction result. **First Tier 3 deliverable.** **Setup-and-scoping scope** — identifies the most consequential remaining QFT ontology gap (the field operator φ(x) of QFT) and proposes substrate-grounded candidates with their structural implications. Concrete construction + theorem-grade derivation requires substantial follow-up work (research-level; multi-session).
**Source:** `../framework/framework_qft_ontology.md` §8 — flagged as the single most consequential ontology gap.
**Predecessors:** Tier 1 program (5 deliverables) + Tier 2 modular structure setup.

---

## Why this is the most consequential ontology gap

QFT's central object is the **field operator** φ(x): an operator-valued distribution at each spacetime point x. Almost all of QFT's structural apparatus is downstream of field operators:

- **Lagrangians**: written in terms of φ(x) and ∂_μ φ(x).
- **Path integrals**: ∫ Dφ exp(iS[φ]) integrate over field configurations.
- **Vacuum expectation values**: ⟨0|T(φ(x_1) ... φ(x_n))|0⟩ are the central observables (Wightman functions).
- **Propagators**: G_F(x − y) = ⟨0|T(φ(x)φ(y))|0⟩.
- **Vertices**: interaction terms = polynomials in φ.
- **Renormalization**: counterterms are φ-polynomials.
- **Currents and conservation**: J^μ(x) = q ψ̄(x)γ^μ ψ(x) etc., all built from field operators.
- **BRST**: ghosts and gauge-fixing involve auxiliary field operators.

**If the substrate grounds φ(x), all of these become substrate-derivable in principle.** This is the gateway to grounding the rest of QFT's ontology in substrate primitives.

The gap status (per `../framework/framework_qft_ontology.md` §8 prior to Tier 1):
> **Field operator φ(x).** Operator-valued distribution at spacetime point x; underlies all of QFT. Not grounded as a substrate object. Candidates: averaged toggle density at substrate point; Bloch-mode-coordinate operator; limit of edge-creation operators at fine resolution. Tier 2 — most consequential single ontology gap.

This document re-classifies as **Tier 3** (research-level) given the substantive setup required and proposes three concrete candidate constructions with their pros/cons.

---

## 1. The structural difficulty

The substrate is **discrete**: F_inv(E)'s Cayley graph has discrete vertices and edges. QFT's field operator φ(x) is defined at *continuum* spacetime points x. The continuum-limit closure (§C in `../operator_sweep/operator_sweep_from_A1.md`) is **partial**: the unitary-evolution continuum is closed at theorem grade, but the smooth-manifold continuum is research-level open.

So φ(x) cannot be defined directly on the substrate's discrete structure. It requires either:
1. **Smearing**: define a smeared field operator φ(f) = ∫ f(x) φ(x) d³x for test function f, treated as the primitive; this is the Wightman-axioms approach.
2. **Lattice approach**: define φ at substrate vertices (or edges) and take a continuum limit only after computing observables.
3. **Bloch-mode approach**: define φ in momentum space (k ∈ BZ) and Fourier-transform to position space inside the substrate's discrete structure.

The framework's existing apparatus favors the **third approach**: most framework predictions are computed in Bloch space (per-Brillouin-point fibers) and only translated to position space via Fourier transform when needed.

---

## 2. Three candidate constructions

### 2.1 Candidate A: Smeared toggle-density operator

**Definition.** For test function f: F_inv(E) → ℝ with finite support, define:

$$\phi_A(f) = \sum_{e \in E} \sum_{g \in F_{inv}(E)} f(g) \cdot T_e(g)$$

where T_e(g) is the toggle-event indicator at substrate position g for edge e. Equivalently: φ_A(f) is the *expected number of toggles weighted by f*.

**Pros.**
- Directly built from substrate primitives (toggles).
- Linear in test function; well-defined on bounded f.
- Natural Hermitian operator on substrate Hilbert space.
- Local: φ_A(f) commutes with φ_A(g) when supp(f) ∩ supp(g) = ∅.

**Cons.**
- *Bosonic* by construction. The framework's substrate fundamental fermions (CAR/JW) are not directly captured.
- Continuum limit: requires smooth-manifold closure (§C partial) to interpret as φ(x) at continuum spacetime point.
- The "toggle density" is an *observable* (real-valued); QFT's φ(x) is an *operator* with non-trivial commutators. Need to check whether the substrate's φ_A satisfies QFT-like equal-time CCR.

**Status.** Plausible bosonic-field candidate; needs CCR check and continuum-limit work.

### 2.2 Candidate B: Bloch-mode coordinate operator

**Definition.** For each Bloch mode k ∈ BZ and band index n, define creation/annihilation operators a_{n,k}^†, a_{n,k} via the spectral decomposition of the substrate adjacency operator A. Then:

$$\phi_B(x) = \int_{BZ} \frac{d^3k}{(2\pi)^3} \sum_n \frac{1}{\sqrt{2\omega_{n,k}}} \big( a_{n,k} u_{n,k}(x) e^{ik \cdot x} + a_{n,k}^\dagger u_{n,k}^*(x) e^{-ik \cdot x} \big)$$

where u_{n,k}(x) is the Bloch periodic part and ω_{n,k} = √(λ_{n,k} − λ_min) (frequency from energy).

**Pros.**
- Standard QFT mode-expansion form; immediately recognizable.
- Bloch decomposition is an existing framework apparatus (Layer 4.17).
- ω_{n,k} is computable from substrate adjacency spectrum.
- The "x" in φ_B(x) is the substrate's Cayley-graph position label; *no smooth-manifold continuum required* for the mode-expansion to make sense at substrate-graph level.

**Cons.**
- The "spacetime" the field lives on is the substrate's discrete Cayley graph, not smooth Minkowski. Standard QFT quantum field theory in flat spacetime is approximated only in the continuum limit.
- Requires identification of which substrate band corresponds to which physical field. The framework's existing identification (Layer 5.9 spinor at each node, B3 chirality bridge) gives this for fermions; for bosons, less clear.

**Status.** Strong candidate; matches QFT's mode-expansion form most closely; needs continuum limit only for matching to standard-QFT spacetime field.

### 2.3 Candidate C: Edge-creation operator

**Definition.** For each substrate edge e ∈ E and substrate position g ∈ F_inv(E), the framework already has a creation/annihilation operator pair via JW (Layer 5.6):

$$c_{e,g}^\dagger = \text{JW}(\sigma^+_{e,g}), \quad c_{e,g} = \text{JW}(\sigma^-_{e,g})$$

Define a position-space "field" operator at substrate vertex v ∈ F_inv(E):

$$\phi_C(v) = \sum_{e \in E} (c_{e, v}^\dagger + c_{e, v})$$

i.e., the sum of creation + annihilation operators on all edges incident to v.

**Pros.**
- Built directly from JW / CAR (Layer 5.6, 5.7) — uses the framework's already-grounded fermion algebra.
- Each substrate vertex carries its own field; locality is exact at the discrete level.
- Hermitian (real combination of c, c†).
- *Fermionic* by construction; matches SM fermion content.

**Cons.**
- *Discrete index v ∈ F_inv(E)* rather than continuum spacetime point x. To match QFT's φ(x), need continuum-limit interpolation.
- Sum over edges e is *aggregate*; doesn't distinguish edge-direction-type structure (which the framework's predictions often need).
- Standard QFT field operators are *not* simple c + c† combinations; they include momentum factors and Bloch coefficients. Candidate B's mode-expansion form is closer to standard.

**Status.** Conceptually clean; useful for fermionic field-theory analog. Probably best treated as a *sub-case* of Candidate B, where each substrate vertex's field decomposes into edge-creation modes.

### 2.4 Recommended synthesis: hybrid B + C

For substrate fermions: use Candidate B's mode-expansion form (matches QFT structure) with Candidate C's JW-creation operators for the c, c† pieces. This gives:

$$\phi(x) = \int_{BZ} \frac{d^3k}{(2\pi)^3} \sum_n \frac{1}{\sqrt{2\omega_{n,k}}} \big( c_{n,k} u_{n,k}(x) e^{ik \cdot x} + c_{n,k}^\dagger u_{n,k}^*(x) e^{-ik \cdot x} \big)$$

where c_{n,k}, c_{n,k}^† are JW-derived CAR operators in the (n, k) Bloch-mode basis. The "x" is a substrate vertex (discrete) at the substrate-rigorous level; in the continuum limit (pending §C), x becomes a smooth-manifold point.

For substrate bosons: Candidate A (smeared toggle-density) is the natural starting point; commutator structure (CCR vs CAR) needs to be checked against substrate's underlying involutivity.

**Honest verdict:** the substrate has natural fermionic field operators (Candidate B + C synthesis); bosonic field operators are less clear since the substrate's primitives are involutive (Hermitian, T_e² = id), which is intrinsically fermionic via JW. **Boson grounding may require a separate construction (Higgs as composite of fermion bilinears? Bosonization?) outside the scope of this scoping document.**

---

## 3. Implications if Candidate B+C closes

### 3.1 Direct ontology landings

**Field operator φ(x)**: grounded as Bloch-mode-expanded substrate operator with JW-derived c, c†. Major QFT ontology gap closed.

**Time-ordered products / Wightman functions**: built from φ via standard QFT formulas. Substrate Wightman functions become substrate-computable n-point functions.

**Feynman propagator G_F(x − y)**: ⟨0|T(φ(x)φ(y))|0⟩ on substrate vacuum |0⟩ (already grounded; Tier 1 op 4). Concretely: integrate substrate's Bloch-mode propagator over BZ.

**Path integrals**: Wick-rotated form (Euclidean) is grounded via heat-kernel apparatus (Tier 1 op 4 + 5). Lorentzian form requires continuum-limit closure for proper definition.

**Vertex operators / interaction terms**: polynomial in φ; computable from substrate Bloch-mode structure.

**Currents J^μ**: ψ̄γ^μ ψ etc.; built from substrate Dirac operator (Tier 1 op 5) + field operators.

### 3.2 Cascade of follow-up grounding

Once φ(x) is grounded, the following QFT objects become substrate-derivable:
- **Wick's theorem**: substrate analog via JW Wick contractions.
- **LSZ reduction**: substrate analog via Bloch-mode asymptotics.
- **S-matrix**: scattering amplitudes from substrate Bloch-mode structure.
- **Renormalization**: substrate-level UV cutoff is the lattice scale (Planck length); RG flow becomes the *coarse-graining* operation in I-projection language (Tier 1 op 1+2).
- **Operator product expansion (OPE)**: substrate operators have natural OPE structure via Bloch-mode coefficient functions.

### 3.3 What still doesn't close even if φ(x) closes

- **BRST / gauge fixing**: requires gauge-redundancy analysis at substrate level. Open.
- **Anomalies (full)**: chirality grounded via Atiyah-Singer (Tier 1 op 5); but full 't Hooft anomaly matching across mass scales requires renormalization derivation.
- **Non-perturbative effects (instantons)**: substrate analog of instanton number is the Atiyah-Singer index of substrate Dirac in non-trivial Bloch backgrounds.
- **Confinement / asymptotic freedom**: substrate-level QCD analog requires color-charge structure derivation beyond what the framework currently has.

### 3.4 The single biggest remaining gap after φ(x)

**Smooth-manifold continuum closure (§C).** Even with φ(x) grounded at substrate-graph level, matching QFT's flat-Minkowski formulation requires the substrate-to-smooth-manifold limit. This is research-level (Gorard 2020 emergent-Einstein direction). **The framework's most prominent open structural problem.**

---

## 4. Honest scope

1. **No theorem-grade derivation in this document.** All three candidates are *proposals*, not derivations. Each requires:
   - Verification of QFT-like commutator/anti-commutator structure on substrate.
   - Continuum-limit analysis (pending §C closure for full-rigor).
   - Cross-validation against framework's existing predictions.

2. **Bosonic field grounding is harder than fermionic.** The substrate's intrinsic structure (involutive toggles → JW → CAR fermions) doesn't directly produce bosonic CCR operators. Bosons in the framework appear via composite fermion bilinears (e.g., Higgs as fermion condensate) or via classical-field limits of bosonic mean-field theory (BZJ scaling at Layer 4.51). A clean substrate-level boson field operator is research-level.

3. **The most concrete first-pass deliverable would be**: substrate fermionic two-point function ⟨0|T(ψ(x)ψ̄(y))|0⟩ computed via Bloch-mode expansion + substrate adjacency spectrum. This is computable and would directly check whether the substrate gives QFT-like propagator structure. ~2–3 sessions for a focused investigation.

4. **Tier 3 status genuinely needed.** This is not a 1–2 session investigation; the cascade of follow-ups (Wick, LSZ, S-matrix, RG, ...) is multi-session each. The setup-and-scoping in this document is the entry point; the full program is 10+ sessions.

---

## 5. Status

**Field operator φ(x) grounding: setup-and-scoping complete.** Three candidates identified; recommended synthesis (Candidate B + C, hybrid Bloch-mode + JW-CAR) is the strongest fermionic-field candidate. Bosonic field grounding flagged as separate research direction.

**Category:** **scoping document** (Tier 3 setup; not a closed theorem). Provides the framework for future work.

**Effect on framework:**
- Identifies the most consequential remaining ontology gap with concrete candidates.
- Roadmap for cascade-of-follow-ups (propagator, S-matrix, RG, ...).
- Distinguishes substrate-rigor structure (Candidate B + C at discrete level) from continuum-limit structure (pending §C).

**Effect on QFT ontology meta-doc:** §8 entry should be updated:
- "Field operator φ(x): scoping done; Candidate B + C synthesis recommended; Tier 3 follow-up."

---

## 6. Cross-references

- `../framework/framework_qft_ontology.md` §8 — original gap flag.
- `../operator_sweep/operator_sweep_from_A1.md` Layer 5.6 (JW), 4.17 (Bloch), 5.21 (Schrödinger evolution).
- `forward_construction_substrate_thermal_apparatus.md` — vacuum |0⟩ identification (input for Wightman / propagator).
- `forward_construction_substrate_atiyah_singer.md` — Dirac operator (input for ψ field).
- `forward_construction_a2t_as_iprojection.md` — RG / coarse-graining grounding via I-projection (informs renormalization gap).
- `../theorems/theorem_car_local_jordan_wigner.md` — JW / CAR substrate apparatus.

**Type 3 (cited published) references:**

- **Wightman, A. S.** (1956). Quantum field theory in terms of vacuum expectation values. *Phys. Rev.* 101, 860–866. (Wightman-axioms approach to field operators.)
- **Streater, R. F. & Wightman, A. S.** (1964). *PCT, Spin and Statistics, and All That.* Benjamin. (Foundational treatment of QFT field operators.)
- **Haag, R.** (1996). *Local Quantum Physics: Fields, Particles, Algebras.* Springer. (Algebraic-QFT formulation; field operators as elements of local algebras.)
- **Peskin, M. E. & Schroeder, D. V.** (1995). *An Introduction to Quantum Field Theory.* Westview. §4 (canonical quantization; field-operator mode expansions).

---

## 7. Next forward-construction steps

**Tier 3 follow-ups for φ(x) grounding** (research-level; multi-session each):

1. ~~**Substrate fermionic two-point function**~~ ✅ CLOSED 2026-04-26 (PM): `forward_construction_substrate_propagator.md`. Substrate Feynman propagator $\tilde G_F^{\text{sub}}(k, \omega) = i(\omega + D(k))/(\omega^2 - n - R_{\text{sub}}(k) + i\varepsilon)$ in closed form via Bloch + JW/CAR + Lichnerowicz substitution. Substrate intrinsic mass scale $= n = |E|$ identified.
2. **Substrate CCR commutator check for Candidate A** (~1–2 sessions). Verify whether smeared toggle-density operators have QFT-like commutators.
3. **Bosonic field grounding** (~3+ sessions). Substrate Higgs-as-composite-fermion-bilinear; or classical-field-limit via BZJ scaling.
4. ~~**Wick-contraction / OPE structure**~~ ✅ Wick CLOSED 2026-04-26 (PM): `forward_construction_substrate_wick.md`. Substrate Wick theorem theorem-grade; n-point functions = signed sum of $G_F^{\text{sub}}$ pair-contractions; Feynman diagrams inherited via Dyson expansion. **OPE coefficients still pending** (~1 session bounded).
5. **Renormalization derivation as substrate coarse-graining** (~3+ sessions). Connects to A2-T's I-projection apparatus (Tier 1 op 1).

**Tier 3 / research-level macro-program**: full smooth-manifold continuum closure (§C), Bisognano-Wichmann substrate analog, BRST grounding, full anomaly matching across scales. Multi-session each; multi-month per major item.
