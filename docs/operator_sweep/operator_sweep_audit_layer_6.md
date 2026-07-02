# Operator Sweep Audit — Layer 6

**Date:** 2026-04-26.
**Status:** Per-operation audit with three-lens entries (audit + search-instrument + ontological grounding).
**Source catalog:** `operator_sweep_from_A1.md` §Layer 6.
**Predecessors:** `operator_sweep_audit_layer_0_1.md`, `_2.md`, `_3.md`, `_4.md`, `_5.md`.

## Layer 6 — Continuum / differential geometry / general relativity

24 operations grouped into:
- **6.A** Smooth manifold structure (8 ops)
- **6.B** Riemannian / Lorentzian geometry (9 ops)
- **6.C** Cosmology / general relativity (7 ops)

**Critical context.** Per §C of the operator sweep, the smooth-manifold portion of the continuum-limit closure is **partial** at framework rigor — the unitary-evolution continuum is closed at journal grade, but the smooth-Lorentzian-manifold continuum is research-level open. Layer 6 ops requiring smooth-manifold structure inherit that partial status. The framework's GR/cosmology predictions are made *despite* this gap, by using FLRW + Friedmann phenomenologically rather than deriving them from substrate.

---

## 6.A — Smooth manifold structure (8 ops)

| # | Op | Verdict | Citation | Ontological grounding |
|---|---|---|---|---|
| 6.1 | Smooth manifold M | invoked-indirect | FLRW spacetime as smooth manifold; phenomenological in `predictions/H_0.py`, `predictions/t_0.py`. Partial under §C. | **Substrate:** the smooth manifold is the *target* of a continuum-limit map from the discrete substrate; not a primitive object. **Why this form (partial):** Strauch 2006 + Stage 3 close the unitary-evolution part; the smooth-Lorentzian-manifold part is open (Gorard 2020 / causal-set direction). **QFT ground:** the spacetime manifold of GR — partially grounded; full derivation is the framework's most prominent open structural problem. |
| 6.2 | Tangent space T_p M | unused-deferred | Catalog only. | **Substrate:** would emerge as the limit of "infinitesimal toggle directions" at a substrate point under the smooth-manifold closure. **Why this form (would require):** Lee 2003 §3 standard differential geometry. **QFT ground:** vector-field machinery on spacetime; absent at framework rigor pending §C closure. |
| 6.3 | Tangent bundle TM, cotangent T*M | unused-deferred | Catalog only. | **Substrate:** would arise as the discrete-to-continuum limit of the substrate's edge-based vector structure. **QFT ground:** field-bundle structure; absent. |
| 6.4 | Tensor fields T^(p,q)(M) | invoked-direct | γ_ab symmetric tensor in `predictions/srs_bloch_dispersion_gamma.py`; T_μν stress-energy in `predictions/w_DE.py`. | **Substrate:** symmetric tensors arise as quadratic-form coefficients in Bloch-dispersion expansions on srs; the framework uses tensor fields *on the substrate's BZ*, not on the smooth-manifold continuum. **Why this form:** quadratic Taylor expansion of Hashimoto eigenvalues at high-symmetry points. **QFT ground:** stress-energy tensor and metric-perturbation tensors of GR — partially grounded via Bloch expansions. |
| 6.5 | Differential forms Ω^k(M) | invoked-indirect | `predictions/c1_photon_bundle.py` Hodge bundle on srs (graph forms, not smooth-manifold forms); cosmological line-element ds² uses 1-forms phenomenologically. | **Substrate:** the framework uses *discrete* differential forms (chain complexes on the srs graph) rather than smooth forms. **Why this form:** Hashimoto / NB-walk chain complex provides a discrete-de-Rham analog on the substrate. **QFT ground:** form fields in gauge theory; partially grounded via discrete substrate forms. |
| 6.6 | Exterior derivative d: Ω^k → Ω^{k+1} | invoked-indirect | `predictions/c1_photon_bundle.py`: kernel of d† for photon Hodge subspace; chain-map work in `proofs/cosmology/srs_photon_c3_chainmap.py`. | **Substrate:** the discrete d operator on the chain complex C^0 ← C^1 ← C^2 of srs primitive cell. **Why this form:** boundary operator of the substrate's chain complex; standard discrete-de-Rham construction. **QFT ground:** exterior derivative in gauge theory and Hodge-decomposition of fields. |
| 6.7 | Lie derivative ℒ_X | unused-deferred | Catalog only. | **Substrate:** would compute the rate of change of a tensor field along a vector-field flow in the smooth limit. **QFT ground:** energy-momentum conservation via diffeomorphism invariance; absent at framework rigor pending §C. |
| 6.8 | de Rham cohomology H^k_dR(M) | unused-deferred | Catalog only. (Note: 6.8 the *smooth* de Rham; the substrate's *discrete* de Rham via 6.6 is invoked-indirect.) | **Substrate:** topological invariants of the smooth-manifold continuum-limit. **QFT ground:** topological gauge sectors, instantons, anomalies; partially substituted at framework rigor by discrete cohomology of substrate chain complex (cf. Appendix A.1). |

**6.A totals:** 2/8 invoked-direct, 3/8 invoked-indirect, 3/8 unused-deferred.

**Ontology meta-finding for 6.A.** The framework operates at Layer 6 with *discrete* differential structure (chain complexes on srs) substituting for *smooth* differential structure (forms on Lorentzian manifold). The substitution is partial — discrete chain complex captures the substrate's information content but does not yet license the smooth-tensor-bundle apparatus the standard formulation of GR requires.

---

## 6.B — Riemannian / Lorentzian geometry (9 ops)

| # | Op | Verdict | Citation | Ontological grounding |
|---|---|---|---|---|
| 6.9 | Riemannian metric g | invoked-direct | `predictions/d_spatial_derivation.md` Fisher metric on statistical manifold (Čencov 1982); `predictions/srs_bloch_dispersion_gamma.py` Riemannian metric γ_ab on Bloch tangent space at P. | **Substrate:** Fisher information metric on the space of substrate distributions; Bloch-tangent metric γ_ab from Hashimoto-eigenvalue Hessian. **Why this form:** quadratic-form structure on tangent space of statistical / spectral manifold. **QFT ground:** spatial metric of GR — grounded in statistical-manifold geometry of substrate distributions. |
| 6.10 | Lorentzian metric (signature −,+,+,+) | invoked-indirect | Lorentz invariance of toggle 4-density at leading order in `../theorems/theorem_lorentz_causal_sector.md`; FLRW Lorentzian form used phenomenologically in cosmology predictions. | **Substrate:** Lorentzian signature is *derived* via Stage 3 (toggle 4-density correlations decay rapidly + emergent causal partition between past/future toggle events). The signature (−,+,+,+) emerges from the asymmetry between time-direction (toggle process, broken T-symmetry, p_create ≠ p_destroy) and space-directions (graph T-symmetry intact, isotropic on srs). **Why this form:** combination of Stage 3 rapid-decay + toggle-process-T-asymmetry. **QFT ground:** Lorentzian signature of spacetime — partially derived; full lattice-to-Lorentzian-manifold limit is research-level. |
| 6.11 | Levi-Civita connection ∇ | unused-deferred | Catalog only. | **Substrate:** would emerge as the unique torsion-free metric-compatible connection on smooth-manifold continuum-limit. **Why this form (would require):** standard differential geometry post §C closure. **QFT ground:** the connection underlying covariant derivatives in GR. |
| 6.12 | Christoffel symbols Γ^k_{ij} | unused-deferred | Catalog only. | **Substrate:** components of Levi-Civita connection in coordinates. **QFT ground:** gravitational interaction terms in matter Lagrangians. |
| 6.13 | Riemann curvature R^a_{bcd} | unused-deferred | Catalog only. | **Substrate:** would emerge as obstruction to flatness in smooth-manifold limit. **Why this form (would require):** §C smooth closure + holonomy of Levi-Civita connection. **QFT ground:** gravitational tidal forces; Einstein equations; absent at framework rigor. |
| 6.14 | Ricci tensor R_{ab}, scalar R | unused-deferred | Catalog only. | **Substrate:** contractions of Riemann curvature. **QFT ground:** Einstein-Hilbert action, sourced by stress-energy. |
| 6.15 | Geodesics ∇_{γ̇} γ̇ = 0 | invoked-indirect | Graph geodesics in `proofs/flavor/ckm_holonomy.py`, `proofs/foundations/delta_dynamical.py`; non-backtracking-walk geodesics. | **Substrate:** graph geodesics (shortest non-backtracking paths) on the substrate Cayley graph; not smooth-manifold geodesics. **Why this form:** discrete analog where minimum word-length paths replace zero-acceleration smooth paths. **QFT ground:** light rays / particle trajectories of GR — partially grounded via substrate NB-walk geodesics. |
| 6.16 | Parallel transport | invoked-indirect | `proofs/foundations/delta_dynamical.py` parallel transport of generation labeling; Wilson loops in `proofs/flavor/srs_bloch_ckm.py`. | **Substrate:** discrete parallel transport along the substrate's chain complex / Bloch bundle. **Why this form:** Wilson-line construction on graph holonomy. **QFT ground:** gauge connection holonomy; here grounded in substrate Bloch-bundle parallel transport. |
| 6.17 | Killing vector fields | unused-deferred | Catalog only. | **Substrate:** would emerge as Lie algebra of isometries of smooth-manifold continuum-limit. **QFT ground:** symmetries of GR backgrounds (e.g., FLRW homogeneity/isotropy Killing vectors). Absent at framework rigor; framework recovers FLRW symmetries phenomenologically, not from a Killing-vector analysis. |

**6.B totals:** 1/9 invoked-direct, 3/9 invoked-indirect, 5/9 unused-deferred.

---

## 6.C — Cosmology / general relativity (7 ops)

| # | Op | Verdict | Citation | Ontological grounding |
|---|---|---|---|---|
| 6.18 | FLRW metric ds² = −dt² + a(t)² dΣ_k² | invoked-direct | `predictions/H_0.py`, `predictions/t_0.py`, `predictions/Omega_DM.py`, `predictions/w_DE.py`; `proofs/cosmology/coasting_sn1a_comparison.py`. | **Substrate:** FLRW form is *imposed phenomenologically* given the substrate's homogeneity (uniform N(t) toggle density) and isotropy (srs symmetry). **Why this form:** the simplest Lorentzian metric consistent with cosmological-principle homogeneity + isotropy; framework adopts it given §C partial closure. **QFT ground:** FLRW universe of cosmology — grounded in substrate homogeneity + isotropy, but the manifold-tensor structure remains partial under §C. |
| 6.19 | Einstein equations G_{ab} + Λ g_{ab} = 8πG T_{ab} | invoked-indirect | `predictions/w_DE.py` (T_μν = −Λ g_μν, w = −1); referenced in scoping an internal working note. | **Substrate:** Einstein equations are not derived from substrate at framework rigor — they are imposed at the Friedmann-equation level. The Einstein-Hilbert action would need to emerge from substrate via §C smooth closure (Gorard 2020 emergent-Einstein direction). **Why this form (partial):** standard GR. **QFT ground:** dynamics of spacetime — the framework's most prominent ontological gap. |
| 6.20 | Friedmann equations | invoked-direct | `predictions/H_0.py`, `predictions/N_hub.py`, `predictions/N_fit.py`. | **Substrate:** Friedmann ODEs as constraints on N(t) (substrate toggle density) → a(t) (cosmological scale factor). **Why this form:** energy/momentum constraints on FLRW with a perfect-fluid stress-energy; framework's Λ ∝ 1/t² coasting variant explored in cosmology workstream. **QFT ground:** cosmological dynamics — partially grounded via N(t) bridge. |
| 6.21 | Hubble parameter H(t) = ȧ/a | invoked-direct | `predictions/H_0.py` (H_0 = 68.18 km/s/Mpc derived from G_F anchor + Friedmann). | **Substrate:** H_0 derived from substrate's BZJ-scaling v ∝ N^{−1/4} (Layer 4.51), inverting to extract N_hub, then Friedmann gives H_0. **Why this form:** observable derived from a(t). **QFT ground:** cosmological expansion rate. The framework's H_0 prediction is one of its strongest cosmology landings. |
| 6.22 | Cosmological scale factor a(t) | invoked-direct | `predictions/N_hub.py`, `proofs/cosmology/coasting_sn1a_comparison.py`. | **Substrate:** a(t) is a derived quantity from substrate's N(t); the "scale factor" emerges as the post-canonicalization observer-side spatial-extent variable. **Why this form:** standard cosmology + framework's BZJ-N(t) bridge. **QFT ground:** cosmological scale-factor evolution. |
| 6.23 | Stress-energy tensor T_{ab} | invoked-direct | `predictions/w_DE.py` (T_μν for Λ-domination); cosmology predictions across the cosmology workstream. | **Substrate:** stress-energy components arise from substrate matter content (toggle-density energy density, p_destroy/p_create-asymmetry pressure). **Why this form:** standard cosmological matter content + dark-correction. **QFT ground:** matter content's stress-energy in GR. |
| 6.24 | Causal structure, light cones, horizons | invoked-direct | `predictions/N_hub.py` ("Each toggle modifies 1/(k*N) of the universe's causal structure"); `../framework/framework_architecture.md` (multiway substrate causal structure); `proofs/cosmology/As_promotion.py` (Hubble horizon). | **Substrate:** *intrinsically* causal — multiway / NB-walk substrate has a built-in causal partition (past/future cone via toggle history). **Why this form:** A1 + Layer 1 reduced-word ordering give substrate a primitive causal structure; no postulated speed limit. **QFT ground:** lightcone / causal structure of GR — *grounded* in substrate causal partition (one of the framework's cleanest cosmology ontology landings). |

**6.C totals:** 6/7 invoked-direct, 1/7 invoked-indirect.

**Ontology meta-finding for 6.C.** The framework's cosmology landings are stronger than its GR-internal landings. FLRW + Friedmann + Hubble + scale factor + stress-energy + causal structure are all grounded in substrate, with phenomenological + bridge-quantity inputs (N(t), BZJ scaling). But the *Einstein equations themselves* — the dynamics of the metric — are not derived; they are imposed.

---

## Aggregate (Layer 6)

| Status | 6.A | 6.B | 6.C | Total |
|---|---|---|---|---|
| invoked-direct | 2 | 1 | 6 | 9 |
| invoked-indirect | 3 | 3 | 1 | 7 |
| unused-deferred | 3 | 5 | 0 | 8 |
| **Layer total** | **8** | **9** | **7** | **24** |

**Coverage.** 24/24 catalog entries audited.

**Cluster finding.** All 8 unused-deferred ops at Layer 6 are smooth-manifold-internal apparatus (tangent space, tangent bundle, Lie derivative, smooth de Rham, Levi-Civita, Christoffel, Riemann, Ricci, Killing vectors). They cluster as the **GR-internal apparatus the framework lacks pending §C smooth-manifold closure**. The cluster is parallel to:
- §5.34–§5.38 quantum thermal/information cluster (6 ops, also unused-deferred, also research-level).
- §1.7 + §1.8 + §2.14 left/right-action cluster (3 ops, deferred to Appendix investigation).

**Forward-construction docs queued.** None spawned this pass; cluster deferred until §C smooth-manifold closure is achievable (research-level direction outside this audit's scope).

---

## Honest verdict on Layer 6 sweep (with ontology lens)

**Three-lens yield categories:**
1. New low-MDL invariant matching SM observable: **none**.
2. Cross-validation of existing prediction via distinct route: **none**.
3. Pinned obstruction: **none direct**, but the §C partial-closure constraint is reaffirmed — the smooth-manifold continuum requires research-level work outside the catalog (Gorard 2020 direction).
4. **Forward-construction cluster:** GR-internal smooth-manifold apparatus (8 ops). Deferred pending §C.
5. **Ontological grounding density:** intermediate. 6.C cosmology lands well; 6.A/B differential geometry lands partially.

### Key Layer 6 ontology landings

| QFT/GR-postulated object | Layer-6 grounding (this audit) |
|---|---|
| **FLRW universe / scale factor a(t)** | Substrate's N(t) (toggle density) + BZJ-scaling bridge; FLRW form imposed phenomenologically given homogeneity + isotropy. |
| **Hubble expansion H_0** | Derived from G_F-anchored N_hub + Friedmann ODE; one of framework's strongest cosmology landings. |
| **Causal structure / lightcones** | Substrate's multiway / NB-walk causal partition; primitive (no postulated speed limit). |
| **Lorentzian signature** | Stage 3 toggle correlations + toggle-process T-asymmetry vs graph T-symmetry. |
| **Riemannian metric (spatial)** | Fisher information metric on substrate distributions (Čencov 1982); Bloch-tangent metric on substrate momentum space. |
| **Stress-energy tensor T_μν** | Substrate matter content + dark-correction; Λ-domination case w_DE = −1 forces T_μν = −Λg_μν. |

### Ontological gaps reaffirmed at Layer 6

- **Smooth manifold limit (§C partial)** — unitary continuum closed; smooth-Lorentzian manifold not.
- **Tangent / cotangent bundle apparatus** — pending §C.
- **Levi-Civita connection / Christoffel / Riemann / Ricci / Killing** — pending §C.
- **Einstein equations** — imposed at Friedmann level; not derived from substrate at framework rigor (the framework's most prominent ontology gap).
- **Newton's constant G** — calibration, not derived (per `../parameters/target_parameters.md`).
- **Cosmological constant Λ** — calibration / scoping doc an internal working note.

---

## Cumulative through Layer 6

| Layer | Ops | invoked | unused-applied-negative | unused-deferred | Notable ontology landings |
|---|---|---|---|---|---|
| 0 | 4 | 4 | 0 | 0 | substrate primitives |
| 1 | 13 | 11 | 0 | 2 | Cayley graph, word length |
| 2 | 33 | 31 | 1 | 1 | L²(F_inv(E)); adjacency op A |
| 3 | 13 | 12 | 1 | 0 | Stone → Schrödinger |
| 4 | 49 | 47 | 1 | 1 | MDL apparatus; Killing-form gauge |
| 5 | 38 | 29 | 2 | 7 | CAR/JW grounding QFT fermions; ρ from compression; Pati-Salam from spatial symmetry |
| 6 | 24 | 16 | 0 | 8 | FLRW from N(t); causal structure primitive; Lorentzian from T-asymmetries |
| **Cumulative** | **174** | **150** | **5** | **19** | — |

**Headline:** 174 ops audited; 150 invoked; 24 unused (5 applied-negative, 19 deferred); 0 SM-matching positive yields, 1 cross-validation candidate (4.25), 3 forward-construction clusters (5.34–5.38 quantum thermal/info, 6.A/B GR-internal, smaller §1.7/1.8/2.14 right-action).

**Ontological harvest now spans:** substrate primitives (Layers 0–1) → Hilbert apparatus (Layer 2) → continuous-time evolution (Layer 3) → MDL + harmonic + Lie + statmech (Layer 4) → QFT fermions + density matrices + GUT structure (Layer 5) → cosmology + causal structure (Layer 6). The meta-doc harvest will have substantial material across all sections.

---

## Cross-references

- `operator_sweep_from_A1.md` §Layer 6 + §C — source catalog and partial-closure context.
- `../theorems/theorem_lorentz_causal_sector.md` — Stage 3 / Lorentzian-signature derivation.
- `predictions/H_0.py`, `predictions/t_0.py`, `predictions/N_hub.py` — cosmology landings.
- `predictions/c1_photon_bundle.py` — discrete-de-Rham / Hodge bundle.
- Predecessor audits.

---

## Status

Layer 6 audit complete. Catalog layer-sweep coverage now: 174/174 catalog ops in Layers 0–6 audited (only the Appendix's 21 explicitly-unused ops remain).

**Major Layer 6 finding:** the framework's cosmology landings (6.C) are notably stronger than its GR-internal landings (6.A/B). This is consistent with §C partial-closure status. The 8-op GR-internal cluster is the framework's most prominent ontological gap and is research-level (Gorard 2020 emergent-Einstein direction).

Next: Appendix sweep (21 explicitly-unused operations including A.1 group cohomology, A.16 modular forms, A.4 Atiyah-Singer index — flagged in the operator sweep's own honest verdict as the three highest-leverage candidates). Then meta-doc harvest into `../framework/framework_qft_ontology.md`.
