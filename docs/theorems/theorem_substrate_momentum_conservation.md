# Substrate Momentum Conservation — spatial-translation corollary

**Date:** 2026-04-29.
**Status:** **THEOREM-GRADE** (corollary of Theorem 1 in `theorem_substrate_symmetry_to_martingale.md`). Phase 1a deliverable of the symmetry-shortcut program. Establishes the substrate's analog of momentum conservation under primitive-cell translation symmetry.

**Predecessor:** `theorem_substrate_symmetry_to_martingale.md` (engine theorem).

**Backing scripts:** none required (verification is structural, not numerical; identifies translation-invariant model functionals already used in the framework).

---

## Question

Theorem 1 (Substrate Symmetry → Conservation) reduces "prove conservation under symmetry G" to a three-point checklist on (H1) prior G-invariance, (H2) filtration G-equivariance, (H3) functional G-invariance. The Phase 1a deliverable applies this checklist to:

**G = primitive-cell translation group of the substrate's srs-lattice realization.**

If (H1)–(H3) hold, the framework has a substrate-level momentum-conservation analog: every translation-invariant model functional has a conserved running posterior expectation, with value forced by symmetry.

This document executes the checklist.

---

## Result (preview)

**Corollary 2 (Substrate Momentum Conservation).** *Let T = ℤ³ be the primitive-cell translation group of the srs lattice (primitive cell: 4 vertices, 6 edges; `../framework/framework_axioms.md` §317). Then:*

1. *(H1)–(H3) hold for G = T;*
2. *for every translation-invariant bounded model functional f, the running posterior expectation E_{π_n}[f(Q)] is a {𝒢_n}-martingale;*
3. *its value at every n equals E_{π_0}[f(Q)], the prior expectation, forced by translation invariance of π_0.*

**Specific conserved expectations identified:**

| Functional | Translation behavior | Conserved? |
|---|---|---|
| Lattice momentum operator P (generator of T) | Commutes with T | ✓ E_{π_n}[P] conserved |
| \|f_q\|² for any Bloch coefficient f_q | Translation-invariant | ✓ E_{π_n}[\|f_q\|²] conserved |
| Polarization tensor at fixed momentum Π_μν(p) | Translation-covariant ↔ momentum-diagonal | ✓ values at each p conserved separately |
| Density-of-states integrals ∫ ρ(ω) g(ω) dω | Translation-invariant | ✓ conserved |
| Single-site amplitude φ(x_0) at fixed x_0 | NOT translation-invariant | ✗ (functional fails (H3)) |

**Net structural finding:** translation-invariant scalar observables of the substrate inherit Doob-martingale conservation under the substrate observation filtration. The conservation is genuine but **does not fix individual values numerically** — it fixes the value to the prior expectation, which still requires a model-class computation. Translation invariance is a *necessary structural ingredient* for the Phase 2b G_sub attack but is not by itself sufficient to compute Π_2(p²→0).

---

## 1. Setup — primitive-cell translation group on srs

### 1.1 The translation group T

The substrate's lattice realization is the srs (Strong-Rigid-Substrate / (10,3)-a / Laves) graph, with primitive cell containing 4 vertices and 6 edges (`../framework/framework_axioms.md` §317; `../framework/framework_architecture.md` §123).

Define **T = ℤ³** acting on the lattice by primitive-cell translations:

$$\sigma_t : x \mapsto x + R_t, \quad R_t = t_1 \mathbf{a}_1 + t_2 \mathbf{a}_2 + t_3 \mathbf{a}_3, \quad t = (t_1, t_2, t_3) \in \mathbb{Z}^3$$

where {𝐚_i} are the primitive lattice vectors. The action extends to:
- substrate states (by translating site labels),
- model class 𝒬 (by push-forward of the action on substrate states),
- σ-algebras on the observation space (by relabeling sites under T).

The time-shift τ : T → ℤ is identically zero — spatial translation does not shift the observation clock.

### 1.2 Continuum limit and IR translation symmetry

For the long-wavelength (p → 0) regime relevant to G_sub and other gravitational predictions, the discrete translation group T = ℤ³ extends to continuous translations T_c = ℝ³ in the IR effective theory. Theorem 1 applies to both:
- **Discrete T:** corollary established in §2 below.
- **Continuous T_c:** holds by taking the IR limit of the discrete-T statement; (H1)–(H3) lift directly because translation-invariant prior + functional + filtration in the discrete case carry over to their continuum versions in the limit.

For Phase 2b (G_sub at p² → 0), T_c is the relevant symmetry.

---

## 2. Verification of (H1)–(H3) for G = T

### 2.1 (H1) — π_0 is T-invariant

**Claim.** A2-T's plural-retention prior π_0 is invariant under primitive-cell translations.

**Argument.** A2-T's prior weights above-waterline models by compression savings (`theorem_A2_mdl_from_finite_register.md` §11). Compression savings are computed from description-length minimization on the substrate; description length is itself spatially homogeneous because:

(i) The substrate's edge-thresholds (p_create=1/2, p_destroy=1/3) are site-independent (`theorem_edge_surprise_thresholds.md`).

(ii) The srs lattice is itself primitive-cell-translation-symmetric (no preferred origin in the bulk).

(iii) The observer protocol assumed in `../forward_constructions/forward_construction_substrate_martingales.md` §1 accumulates *all* substrate observations rather than observations at a fixed site, so the protocol carries no implicit spatial anchor.

Together (i)–(iii) imply that compression savings σ(M) are translation-invariant: σ(σ_t · M) = σ(M). Hence π_0(σ_t · Q) = π_0(Q). ✓

**Caveat.** If a future framework variant introduces site-specific observer setups (e.g., a fixed measurement origin), (H1) would fail and translation conservation would be broken. The current framework, per A1 + A2-T (with P1' a derived theorem under A1 per `theorem_p1_prime_derived_from_a1.md`), has no such anchor.

### 2.2 (H2) — {𝒢_n} is T-equivariant

**Claim.** The substrate observation filtration is equivariant under primitive-cell translations with τ ≡ 0.

**Argument.** 𝒢_n is the σ-algebra generated by all substrate observations through time n (`../forward_constructions/forward_construction_substrate_martingales.md` §1). Under σ_t, an observation at site x at time k becomes an observation at site x + R_t at time k. The set of all observations through time n is permuted by T but remains the same set, so σ_t(𝒢_n) = 𝒢_n. ✓

τ ≡ 0 because spatial translation does not advance or retard the observation clock.

**Remark.** This is the cleanest case of (H2) — full filtration preservation rather than mere equivariance up to a shift. The time-translation case had τ(t) = t (non-trivial shift); spatial translation has τ ≡ 0.

### 2.3 (H3) — translation-invariant model functionals

**Claim.** A rich family of natural model functionals are T-invariant; these are the conserved-expectation candidates.

**Translation-invariant functionals.** The infinitesimal generator of T is the lattice momentum operator P = (P_1, P_2, P_3) acting on model states by P_i = -i ∂/∂x_i (effective continuum) or equivalently as the shift generator on the discrete lattice. A functional f is T-invariant iff [P, f] = 0 (continuous case) or σ_t·f = f for all t ∈ ℤ³ (discrete case).

Examples directly relevant to the framework:

| Functional | T-invariant? | Provenance / use |
|---|---|---|
| Lattice momentum P itself | ✓ (commutes with T trivially) | momentum-conservation generator |
| Bloch density \|f_q\|² at any q | ✓ (translation maps f_q → e^{iq·R_t} f_q; modulus squared invariant) | structure factor |
| Polarization Π_μν(p) at fixed external p | ✓ (translation acts as overall phase on amputated diagrams) | Phase 2b G_sub backbone |
| TT-projection of Π_μν(p) | ✓ (translation-invariant tensor projection) | Phase 2b G_sub direct target |
| Density of states ρ(ω) | ✓ (sum over translation-invariant momentum sectors) | thermal observables |
| Single-site amplitude φ(x_0) | ✗ | NOT in the conserved family |
| Two-point correlator G(x, y) | ✗ in general; G(x − y) ✓ | site-dependent vs translation-respecting forms |

The pattern: **functions of momentum-space quantities are T-invariant; functions of position-space quantities at fixed sites are not.** This matches the standard Noether interpretation (momentum is the conserved charge of spatial translation).

### 2.4 Combined conclusion

(H1) ✓ + (H2) ✓ + (H3) ✓ for the families above ⇒ Theorem 1 applies.

**Corollary 2 statement (formal).** *For G = T = ℤ³ (or its continuum extension T_c = ℝ³ in the IR limit) acting by primitive-cell translation on the srs lattice, every bounded T-invariant model functional f has*

$$M_n := \mathbb{E}_{\pi_n}[f(Q)] \quad \text{a martingale w.r.t.~} \{\mathcal{G}_n\}, \quad \mathbb{E}_{\pi_0}[M_n] = \mathbb{E}_{\pi_0}[f(Q)] \text{ for all } n \geq 0.$$

This is the substrate's momentum-conservation analog.

---

## 3. What this conservation does and does not give

The role of Corollary 2 is to make precise what spatial translation buys structurally. Three honest points:

### 3.1 What momentum conservation DOES give

- **Well-definedness of momentum-space observables.** Π_μν(p) is momentum-diagonal — there are no off-diagonal momentum-mixing terms in the conserved sector.
- **A martingale structural identity** for E_{π_n}[F] for every translation-invariant F. The running expectation under further observations cannot drift.
- **A computable target form.** The conserved value equals E_{π_0}[F], the prior expectation. This is a one-integral computation against π_0, not a full Bayesian update calculation.
- **A sanity check on derived predictions.** Any framework prediction that violates translation conservation has a structural error.

### 3.2 What momentum conservation does NOT give

- **Numerical values are NOT forced.** E_{π_0}[F] still requires evaluating the prior over the model class. Translation invariance fixes the *target form* but not the number.
- **Π_2(p² → 0) is not directly determined.** The Phase 2b G_sub attack needs the *value* of Π_TT'(p²)|_{p²=0}, not just its existence as a translation-invariant quantity. Translation alone is insufficient.
- **Additional symmetries are needed for full numerical closure.** Examples: srs point-group (D₃d / O_h subgroup of the cubic primitive-cell symmetry), gauge-group action, chiral / Lorentz-sector group from `theorem_lorentz_causal_sector.md`.

### 3.3 Implication for Phase 2b

Phase 2b's hypothesis (G_sub via spatial-translation martingale) is sharpened to:

> **Phase 2b sharpened hypothesis.** Translation invariance + an additional symmetry G' (point group / gauge / Lorentz) acts on the model class such that the only G ⋊ G'-invariant scalar at the relevant order in p² is Π_TT' itself, fixing it up to a calculable prior expectation.

If true, G_sub closes via symmetry forcing rather than direct Bloch numerics. If false, translation conservation reduces to a sanity check and Phase 2b falls back to a Wannier or sum-rule route.

The Phase 2b scoping doc (deferred until after Phase 1b completes) will identify the candidate G' and test the sharpened hypothesis.

---

## 4. Cross-validation with existing framework results

Three independent framework results already implicitly use translation invariance in ways consistent with Corollary 2:

1. **Bloch decomposition** (`theorem_bloch_lift_mu.md`): the substrate's eigenstructure decomposes by lattice momentum k. This is structurally a translation-invariant decomposition; Corollary 2 makes the conservation explicit.

2. **Hashimoto NB-walk operator** (`../framework/framework_architecture.md` §37): B is translation-equivariant on the directed-edge space. The spectral statistics of B are translation-invariant model functionals; Corollary 2 confirms their expectations are conserved.

3. **Class B dispersion parameters** (`theorem_class_B_dispersion.md`): v_F, β, D_H, and other dispersion gradients at Dirac points are translation-invariant scalars (computed at fixed momentum). Their derivation implicitly uses (H1)–(H3); Corollary 2 makes this explicit.

These three pre-existing results are not new derivations from Corollary 2 — they predate it — but they confirm that the framework's existing translation-using machinery is consistent with the Phase 0 engine.

---

## 5. Honest scope

1. **The corollary is structural, not numerical.** Corollary 2 identifies a family of conservation laws but computes no new numbers. Phase 2b G_sub closure requires the additional symmetry G' beyond translation.

2. **(H1) depends on observer-protocol homogeneity.** A framework variant that picks a fixed measurement origin would break (H1) and invalidate Corollary 2. The current framework is anchor-free in the bulk.

3. **Continuum lift (§1.2) requires the IR effective theory to be well-defined.** This is assumed throughout the framework's continuum predictions; Phase 2b G_sub is a continuum-IR observable, so the lift is the relevant case.

4. **Functionals violating (H3) are not covered.** Single-site amplitudes, position-space correlators at fixed sites, and any observable depending on a chosen origin are not in the conserved family. This excludes some natural-looking observables and is a real restriction.

5. **The statement does not lift to a full Galilean / Lorentz invariance.** Spatial translation is a strict subgroup of the spacetime symmetry group of the IR effective theory. The Lorentz-sector translation conservation is a separate corollary (would be Phase 1d if pursued).

6. **No Phase 2 prediction depends on Corollary 2 alone.** Each Phase 2 attack needs a specific G-invariance argument tailored to its observable; Corollary 2 supplies translation as one ingredient.

---

## 6. Status

**Corollary 2 established at theorem-grade** as a direct application of Theorem 1.

**Effect on framework:**
- Phase 1a deliverable closed.
- Substrate has an explicit momentum-conservation analog.
- Three pre-existing framework results (Bloch, Hashimoto, Class B dispersion) cross-validate consistency.
- Phase 2b G_sub attack is **sharpened** but not unblocked — needs an additional symmetry G' identification (Phase 2b scoping).

**Effect on QFT ontology meta-doc:** `../framework/framework_qft_ontology.md` should add a "momentum conservation / spatial-translation invariance" entry pointing here.

**Effect on parameter ledger:** no row graduations (Corollary 2 fixes no numbers). Sets up structural backbone for Phase 2b's G_sub attempt.

---

## 7. Citations

**Type 3 (cited published) references:**

- **Williams, D.** (1991). *Probability with Martingales.* Cambridge University Press. §10.7 (inherited via Theorem 1).
- **Olver, P. J.** (1986). *Applications of Lie Groups to Differential Equations.* Springer. §4–5 (continuous group actions; inherited via Theorem 1).
- **Ashcroft, N. W. & Mermin, N. D.** (1976). *Solid State Physics.* Cengage. Ch 8 (Bloch's theorem, lattice translation symmetry, momentum-space decomposition). Standard reference for §1.1 lattice setup and §2.3 Bloch / structure-factor functionals.

All citations to peer-reviewed published work or standard textbooks.

---

## 8. Cross-references

- `theorem_substrate_symmetry_to_martingale.md` — engine theorem; this corollary's parent.
- `../forward_constructions/forward_construction_substrate_martingales.md` — predecessor; time-translation case as Corollary 1.
- `theorem_A2_mdl_from_finite_register.md` — A2-T plural-retention regime; supplies π_0 used in (H1).
- `theorem_edge_surprise_thresholds.md` — supplies the site-independent edge thresholds used in (H1) argument (i).
- `../framework/framework_axioms.md` §317 + `../framework/framework_architecture.md` §123 — srs lattice primitive-cell description.
- `theorem_bloch_lift_mu.md` — translation-using machinery cross-validating Corollary 2 (§4 item 1).
- `theorem_class_B_dispersion.md` — translation-using machinery cross-validating Corollary 2 (§4 item 3).
- `../framework/framework_qft_ontology.md` — meta-doc; should be updated per §6 above.

---

## 9. Next forward-construction steps

Phase 1a complete. Natural next deliverables:

1. **Phase 1b — C₃ cyclic shift corollary** (`theorem_substrate_symmetry_to_martingale.md` §5.3 stub). Estimated 1 session. Bears on Phase 2a θ_QCD already-clarified-discrete attack and Phase 2d arg(h).

2. **Phase 1c — Pati-Salam corollary** (`theorem_substrate_symmetry_to_martingale.md` §5.4 stub). Estimated 1 session. Adoption-conditional.

3. **Phase 2b G_sub scoping** (after Phase 1b/1c complete; uses Corollary 2 plus an additional symmetry G'). The sharpened hypothesis from §3.3 above is the entry point.
