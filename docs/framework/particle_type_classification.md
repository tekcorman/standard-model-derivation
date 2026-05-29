# Particle-type classification in the framework

**Date:** 2026-04-17
**Status:** Classification document, not a theorem. Inventories SM and BSM particle content and assigns each to a framework-layer type (local graph / global Bloch / Dirac / scalar / dark). Closes Sprint 11 B7.6 Thread D and supports Theorem B7.2 Step 5 (gauge bosons lack C³_gen factor).
**Sprint:** 11, Workstream B7.6 Thread D.

## Purpose

`framework_architecture.md` §Layer 7 invokes three particle-type classes — local graph features, global Bloch modes, Dirac spinors — and claims the classification has physical consequences (e.g., gauge bosons lack generation multiplicity because they are global Bloch modes without a C³_gen factor). The classification was described as "implicit in the repo" with Sprint 11 B7.6 as the formalization target. This document makes the classification explicit.

## Definitions

### Class 1 — Local graph features

**Definition.** A local graph feature is an observable that depends only on a localized region of the srs lattice (a single vertex, a single edge, or a bounded neighborhood). It compresses into a **single scalar or tensor value per region**, not a Bloch mode or spinor.

**Mathematical home.** $\mathcal{H}_{\text{local}} \cong L^2(\text{srs vertices}) \otimes \mathbb{C}^{d_{\text{internal}}}$ where $d_{\text{internal}}$ is the internal-tensor dimension (typically 1 for scalars).

**Generation multiplicity.** None. A local feature has one value per region.

**Examples (framework):** scalar condensate density (Higgs VEV order parameter before symmetry breaking is picked), vertex-level measurement operators, local charge density.

### Class 2 — Global Bloch modes

**Definition.** A global Bloch mode is an eigenmode of the walker Bloch operator $B(k)$ or its derived objects (gauge curvature), extending across the entire lattice with a specific $(k, \alpha)$ labeling (k-vector × band-index). It compresses into a **(k-space, band, polarization) tuple**, not a localized feature.

**Mathematical home.** $\mathcal{H}_{\text{Bloch}} \cong \int^\oplus_{\text{BZ}} \mathcal{H}_{\text{k}} \, dk$ with fiber $\mathcal{H}_{\text{k}}$ carrying internal band/polarization labels.

**Generation multiplicity.** None. A Bloch mode is labeled by $(k, \alpha)$; no generation factor.

**Examples:** photon $\gamma$ (Bloch mode of the U(1)_em gauge curvature on srs), W±, Z⁰ (SU(2)_L × U(1)_Y gauge bosons, same structure), gluons $g$ (SU(3)_c curvature; one per color via the color-Z_3 center derived in B6), graviton (if a framework object — see open questions).

### Class 3 — Dirac spinors

**Definition.** A Dirac spinor particle is a state in the 8-dim Cl(6,0) spinor space (Theorem B3, `../../predictions/theorem_B3_spinor_fermion_derivation.md`) at the P-point of the srs Brillouin zone, carrying chirality + species labels + gauge-rep content + generation label. It compresses into a **full tensor-product state** $|j\rangle_{\text{gen}} \otimes \psi_{\text{gauge}} \otimes \psi_{\text{spinor}}$.

**Mathematical home.** $\mathcal{H}_{\text{Dirac}} \cong \mathbb{C}^3_{\text{gen}} \otimes \mathcal{H}_{\text{gauge}} \otimes \mathbb{C}^8_{\text{spinor}}$ per B7.2. Also has a k-space factor for propagation: full per-particle Hilbert space is $\mathcal{H}_{\text{Dirac}} \otimes L^2(\text{BZ})$.

**Generation multiplicity.** Three (from the C³_gen factor per B7.2).

**Examples:** all SM fermions: charged leptons (e, μ, τ), neutrinos (ν_e, ν_μ, ν_τ), up-type quarks (u, c, t), down-type quarks (d, s, b). 12 fermion types × 3 colors (quarks only) × 2 chiralities = 48 states at one k-point, matching `results/parameters.csv` row `fermion_content`.

### Class 4 — Global scalar fields

**Definition.** A global scalar is a Bloch mode that is also a scalar under the spinor structure (no Cl(6,0) content). It has Bloch labeling but no spinor degrees of freedom, no generation multiplicity.

**Mathematical home.** $\mathcal{H}_{\text{scalar}} \cong L^2(\text{BZ})$ with a single internal dimension.

**Generation multiplicity.** None.

**Examples:** Higgs field H (SU(2)_L × U(1)_Y doublet scalar; global field content, single copy).

### Class 5 — Dark (uncompressed multiway residue)

**Definition.** A dark-sector particle is multiway-substrate content that fails MDL "pays for itself" (per an internal working note). It has structural content (gravitates) but does not appear in the compressed visible sector with gauge interactions.

**Mathematical home.** The multiway substrate beyond the observer's compression capacity N*. Not inside any of $\mathcal{H}_{\text{local}}, \mathcal{H}_{\text{Bloch}}, \mathcal{H}_{\text{Dirac}}, \mathcal{H}_{\text{scalar}}$.

**Generation multiplicity.** Unspecified — the framework does not derive a dark-particle spectrum.

**Examples:** dark matter as uncompressed multiway structure (per an external research note on dark-matter compression).

## SM particle catalog

| Particle | SM role | Class | Generations? | Layer(s) active | Notes |
|---|---|---|---|---|---|
| Quarks (u, c, t) | fermion, up-type, color-triplet | **Dirac** | **3** | 3 (Cl(6,0) + color-Z_3), 4 (C³_gen) | 3 copies × 3 colors × 2 chiralities = 18 states per k |
| Quarks (d, s, b) | fermion, down-type, color-triplet | **Dirac** | **3** | 3, 4 | Same as above |
| Charged leptons (e, μ, τ) | fermion, color-singlet | **Dirac** | **3** | 3, 4 | 3 copies × 1 color × 2 chiralities = 6 states per k |
| Neutrinos (ν_e, ν_μ, ν_τ) | fermion, color-singlet, EM-neutral | **Dirac** | **3** | 3, 4 | ν_R required by Cl(6,0) signature (no Majorana-Weyl); seesaw still possible |
| Photon γ | gauge boson, U(1)_em | **Global Bloch mode** | 1 | 2, 3 | Single photon species, Bloch decomposition over BZ |
| W± | gauge boson, SU(2)_L charged | **Global Bloch mode** | 1 | 2, 3 | Pair, massive (EW-broken) |
| Z⁰ | gauge boson, neutral | **Global Bloch mode** | 1 | 2, 3 | Single species, massive |
| Gluons g | gauge boson, SU(3)_c | **Global Bloch mode** | 1 per color | 2, 3 | 8 gluons = 3² − 1 SU(3)_c adjoint; color is Layer-3 gauge rep factor |
| Higgs H | scalar field | **Global scalar** | 1 | 2, 3 (partial) | Doublet of SU(2)_L; single copy; VEV not generation-labeled |
| Graviton G (if in framework) | metric perturbation | **(Open)** | (Open) | 1 (causal graph curvature) | Not in current framework; gravitation is causal-graph Ricci per Gorard |

## BSM particle catalog (scoping)

| Particle | BSM role | Class | Generations? | Status in framework |
|---|---|---|---|---|
| Sfermions, Gauginos, Higgsino, Gravitino (SUSY partners) | β-coefficient label only | (none in framework substrate) | (n/a) | **NOT substrate-derived.** The framework's substrate-derived matter content is 3 PS generations + 2 Higgs doublets (all-fermionic Cl(6) Fock per Path-E recheck 2026-05-12). The MSSM β-coefficient values (33/5, 1, −3) that the framework predicts are derived by algebraic inversion ([`theorem_beta_coefficients_derived.md`](../theorems/theorem_beta_coefficients_derived.md), mathematically complete); whether literal sparticles physically realize these values is an open theoretical question separate from the framework's derivation chain. See [R-19](../audits/registers/structural_residue_register.md) for the precise Δb_2 = +4 gap characterization and SUSY-load-bearing audit 2026-05-27 for the verification that no framework prediction depends on literal SUSY particles. Sparticle-spectrum values in `../parameters/predictions.md` §SUSY Spectrum are honest-conditional on the literal-particle interpretation, not substrate-derived. |
| Dark matter | unknown | **Dark (Class 5)** | unspecified | Structurally derived; no particle spectrum |
| Dark energy | uniform vacuum | **Branching-rate phenomenon** | N/A | Structurally derived via ε ~ 10⁻⁶¹ branching rate |
| Axion (not in framework) | QCD CP problem | (none) | — | Not needed: θ_QCD = 0 from edge-locality per `predictions/theta_QCD_derivation.md` |

## Structural consequences

### Why only Dirac fermions have generations

From B7.2 Step 5 + this classification: the C³_gen Hilbert space is a tensor factor specifically in the Dirac-spinor construction. Global Bloch modes (gauge bosons), global scalars (Higgs), and dark matter do NOT have a C³_gen factor, so they have no generation multiplicity.

This matches observation:
- 12 fermion species × 3 generations = 36 base states per color.
- 4+ gauge bosons (γ, W±, Z, 8 gluons) — 12 distinct species, each one copy.
- 1 Higgs doublet.
- Dark matter: structural content, no distinguishable "generations" in the observer's catalog.

The non-generation-multiplicity of gauge bosons is a structural consequence, not a coincidence.

### Why there are exactly 12 fermion species

From B3 (Cl(6,0) spinor = 8 states) + color (SU(3)_c gauge rep branching): 8 × 3 (colors in quark sector; 1 in lepton sector) = 8 + 16 ... let me count correctly.

Per generation, per chirality: 2 quarks (u, d) × 3 colors + 2 leptons (ν, e) = 6 + 2 = 8 states. Matches Cl(6,0) spinor dim.

With both chiralities: 16 states per generation per color structure. With three generations: 48 states. Matches `results/parameters.csv` row `fermion_content`.

### Gauge boson count

SU(3)_c × SU(2)_L × U(1)_Y adjoint: 8 + 3 + 1 = 12. Plus graviton (if included) = 13.

Observed: 8 gluons + W± + Z⁰ + γ = 12 gauge bosons. Matches.

## Open questions

1. **Graviton in the framework.** Gravity is causal-graph Ricci curvature (Gorard). Is a "graviton" particle in the observer's compressed catalog, or only a derived-gravitational-field phenomenon? The framework lacks a clear answer. Likely it's a Layer-1 multiway-substrate object, not a Layer-2 srs-compressed particle.

2. **Dark-matter particle vs structural phenomenon.** This classification puts dark matter in Class 5 (uncompressed multiway). Whether it has distinguishable "particle species" within Class 5 is an open framework question — current work treats it as structural content, not a particle zoo.

3. **SUSY partner Hilbert spaces (RESOLVED 2026-05-27 — open question retired).** The framework does NOT include literal SUSY partners in its substrate-derived particle catalog. The substrate's matter content is 3 PS generations + 2 Higgs doublets, all-fermionic via Cl(6) Fock per Path-E recheck. The MSSM β-coefficient values are derived by algebraic inversion (not by literal-particle counting). Sparticle-spectrum entries in `../parameters/predictions.md` §SUSY Spectrum are honest-conditional only. See [R-19](../audits/registers/structural_residue_register.md) + SUSY-load-bearing audit.

4. **Higgs generation-independence.** The Higgs couples to all three generations differently (Yukawa matrix); but the Higgs itself is Class 4 (global scalar) with no C³_gen factor. The generation-dependence of Yukawa couplings lives in the mass operator $M_{\text{gen}}$ (Sprint 11 B7.3 target), not in the Higgs-field Hilbert space.

5. **Class assignments for exotic candidates.** If the framework produces a "Z' ", a "heavy W' ", or similar BSM gauge boson, it would fit Class 2 (global Bloch mode) with gauge-rep-factor labeling. Current framework does not derive such particles.

## Consequences for Sprint 11

### For B7.3 (mass operator derivation)

The mass operator $M_{\text{gen}}$ acts on C³_gen and is species-dependent. Under this classification:
- Quark $M_{\text{gen}}$ acts on the quark species' generation factor (up-type 3 eigenvalues, down-type 3 eigenvalues, species-distinguished by the Cl(6,0) spinor label).
- Lepton $M_{\text{gen}}$ similarly (charged 3 eigenvalues, neutrino 3 eigenvalues).
- Gauge bosons have no $M_{\text{gen}}$ (no C³_gen factor); their mass structure (non-zero for W/Z, zero for γ/gluons) comes from electroweak symmetry breaking on the gauge rep factor, not from generation-indexed operators.

### For B7.6 Thread D (this thread)

**Closed.** The classification above is explicit. Every SM particle is mapped to a framework class. BSM extensions (SUSY, dark sector) are tabulated with clear conditional-status flags.

## References

- `framework_architecture.md` — Layer 7 framing.
- `../../predictions/theorem_B3_spinor_fermion_derivation.md` — 8-dim Cl(6,0) spinor as one PS family.
- `../../predictions/observer_dim_three_derivation.md` — observer Hilbert space dim = 3.
- `docs/theorem_B6_bridge.md` — color-Z_3.
- External research note on dark-matter compression — dark matter structural argument.
- Sprint plan: `docs/master_plan.md` §Sprint 11 B7.6.
- `results/parameters.csv` rows: `fermion_content`, `gauge_group`, `higgs_rep`, `generations`, `charge_quantization`.

## Scope honesty

This document is a classification / catalog, not a theorem. The classes (Class 1–5) are defined structurally from the framework's existing Hilbert-space factors; the assignments of specific SM particles to classes follow standard physics (observed multiplicity and interactions). The document is honest about open questions: graviton placement, dark-matter spectrum, SUSY partner details.

The most important structural output is the **Class 3 = Dirac spinor = only class with generation multiplicity** statement. This makes rigorous the informal claim in B7.2 Step 5 and should be cited there.
