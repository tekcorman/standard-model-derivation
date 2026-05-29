# Master Plan — current frontier and what would unblock it

**Last sync:** 2026-05-26. If you're reading this after that date, cross-check against [`parameters/target_parameters.md`](parameters/target_parameters.md) and the recent `state_of_*.md` files in `docs/`.

**North star.** Before using this worklist, read [`north_star.md`](north_star.md) — the standing definition of "done": a complete derived CSCO; the mass sector forced into the same over-determined resolvent that governs the oblique/CKM sector. Priority calls should be judged against it: work that produces another isolated theorem-grade number but doesn't move the selection map toward generativity or toward over-determination is enumeration, not progress.

---

## Where the framework stands (2026-05-26)

Per [`parameters/target_parameters.md`](parameters/target_parameters.md):

| Status | Count |
|---|---|
| ✅ closed | 91 |
| 🟡 in progress | 9 |
| ❌ not started / retired | 13 |
| ⚙️ structural | 10 |
| **Total tracked** | **123** |

Three major fronts have substantially closed in the last six weeks:

- **Lorentz arc** — 2026-04-27 (LORENTZ_SIG/CCLOSE → NC_GEOM joint closure)
- **Gauge coupling chain** including sector-specific c_color = 1/4 — 2026-05-04 → 2026-05-26 EOD+1
- **M_persistence 12-mass fermion operator** including Type-II saturation for m_t — 2026-05-26

And four deep structural results:

- **R-9 closure** — srs is forced by (A)+(B)+(I) + Sunada 2012, not adopted (2026-05-12)
- **(A)+(B)+(I) axiom demotion** — A1 itself derived from three irreducible commitments (2026-05-07/08)
- **12-observable §8 over-determination** — same B_NB resolvent yields 12 readings (2026-05-16)
- **Multi-axial dark-sector waterfilling theorem** — promoted theorem-grade-structural (2026-05-24)

And one flagship gate closed **negative** (an honest reach-limit, 2026-05-28):

- **Gravity coupling factor-of-2 (κ / Newton's G) — CLOSED-NEGATIVE.** Exhaustive blind investigation across every route (R1 work-density; R2 entropy-normalization; blind trilemma→dilemma, which eliminated the Shannon-surprise c_S≈2.585 horn; extent-vs-flux; 2-sphere boundary) shows the factor of 2 is the horizon-entropy count c_S (κ cancels), the framework's own accounting gives **c_S=1 → G_eff=2G**, and **c_S=2 is not forced** by any framework structure. The parameter-free-Newton's-G flagship does **not** close; gravity stays **FORM-LEVEL** (emergent Lorentzian metric + emergent standard Friedmann + coasting from the native information-Clausius relation — all c_S-independent and robust). κ is not promotable; `predictions/G_N.py` is unaffected (independent G_sub-Drude route). One foundational question is parked (must be settled *blind*): is the gravitating horizon entropy the boundary mutual information (2×) or the entanglement entropy (1×)? — an inherited emergent-gravity question, not a framework calculation. Probes: `proofs/cosmology/cS_horizon_entropy_blind`, `cS_extent_vs_flux`, `cS_2sphere_boundary_reopener`.

---

## Active frontier — what's open

### L6 cluster (cosmology precision)
- **Targets:** n_s, σ_8, r_s, θ_*, recombination quantities
- **Status:** Sprint A+B (2026-05-15) ruled out the obvious closure paths. 2026-05-26 propagation cascade reframe gave partial traction (first F-fiber L_r=3 theorem-grade) but the wider cluster remains genuinely open.
- **Block:** No clear mechanism for the rest of the cluster.
- **Unblocks:** L6 cosmology rows; A_s grade upgrade from DOMINANT-THEOREM-GRADE-CONDITIONAL to UNIQUE.

### Need-B δ_quark closure
- **Targets:** the 4 light quark masses (m_u, m_d, m_s, m_c) precision; V_ub graduation
- **Status:** Substrate-side surface exhausted across 10 categorically-distinct mechanisms. δ(n) = 2/(9(n+1)) is theorem-grade-structural via W3 PS sector-connectivity (2026-05-26); residual precision is per-species.
- **Block:** Surviving route is the observer-side BR4 C³_obs ↔ substrate intertwiner; multi-session research with ~15-25% closure probability.
- **Unblocks:** Row P39 precision; potentially V_ub graduation; possibly the 12-observable §8 family for the light-quark sector.

### N_hub first-principles derivation
- **Targets:** ~6 G1-cluster rows (v_Higgs, m_τ family, H_0, t_0, Λ_CC, m_H)
- **Status:** Adopted-N_hub value pinned to ppm by consistency with the measured G_F (calibration). The G1b R2 path (2026-04-28) closed the structural relation H · N · t_P = 1; the *value* of N remains externally anchored.
- **Block:** No closure path identified; research-level.
- **Unblocks:** All "(the value of the adopted N_hub is pinned via the measured G_F)" rows graduate to fully unconditional UNIQUE-THEOREM-GRADE.

### Two-loop MSSM threshold corrections (m_t, m_b precision)
- **Targets:** m_t residual (+4.71σ_PDG; +0.82% relative); m_b borderline (+2.99σ_PDG; +2.15% relative)
- **Status:** Current chain is M_persistence + Type-II saturation y_t(GUT) = 1 + MSSM RGE. Residual is MSSM-threshold + two-loop class.
- **Block:** Standard QFT machinery, not a framework defect. W23 (2026-05-22) tested 2-loop MSSM RG and found it WORSENS all 6 gauge residuals — loop order alone is not the lever.
- **Unblocks:** m_t to within σ_PDG; m_b similarly.

### Literal-particle β-coefficient gap (R-19)
- **Status:** Substrate-derived matter content (3 PS gens + 2HDM, no superpartners) gives 1-loop b_2 = −3 at SU(2)_L (literature 2HDM value); observation requires MSSM b_2 = +1 for unification at α_GUT⁻¹ = 24. Δb_2 = +4 gap precisely characterized ([R-19](audits/registers/structural_residue_register.md), A1 Session 1 closure 2026-05-27).
- **Block:** No substrate-side derivation route for Δb_i closure identified. Branch A (A1 heat-kernel, A3 V_Ram-iso re-exam, A4 unused saddles) comprehensively exhausted 2026-05-27. Branch A2 (M_unif threshold matching) undeveloped with no concrete starting point.
- **Honest position:** Literal-SUSY interpretation of the MSSM β-coefficient values is one realization; alternative non-SUSY interpretations (e.g., non-perturbative substrate contributions, two-loop substrate effects, threshold matching at M_unif) remain candidate mechanisms; none yet structurally derived. Per the SUSY-load-bearing audit, no framework prediction depends on literal sparticle existence.
- **Open item:** tan β documented-vs-live disagreement (Row P46, surfaced 2026-05-26 by literal-fallback audit). Live `predictions/tan_beta.py` computes ≈60.07; `proofs/masses/srs_tan_beta.py` documented 44.73. Reconciliation pending.

### M_Z + m_W precision (δ_r oblique frontier)
- **Targets:** M_Z residual (+7.76σ_PDG; ~2.3 ppm intrinsic floor); m_W (+2.39σ_PDG)
- **Status:** Per the 2026-05-26 gauge-cluster triage lint, these sit on the δ_r oblique frontier separate from c_EW. Unified-oblique theorem (2026-05-16) closes the structural form; precision residual is a deeper-layer object.
- **Block:** Continuum/dispersive Fano-type self-energy on the McKay cut (the §2 deep-layer object).
- **Unblocks:** M_Z, m_W σ_PDG-class matches.

### Citation-graph repair (hygiene)
- **Targets:** ~52 orphan probes in `proofs/` that compute results the framework relies on but no doc cites
- **Status:** Identified by 2026-05-21 Spring-cleaning audit (HANDOFF.md item #1). Cleanup pass moved 42 superseded probes to `proofs/_archive/`; the residual ~52 are evidence the framework actually uses but with broken citation.
- **Block:** Per-probe; each needs the right theorem/derivation to cite it; not mechanical.
- **Unblocks:** Closes the evidence-layer-vs-claims gap (credibility item, not a framework defect).
- **Pointer:** [`../proofs/README.md`](../proofs/README.md) self-check recipe.

---

## Recently closed — no further work needed

| Item | Date | Theorem doc |
|---|---|---|
| R-9 srs uniqueness | 2026-05-12 | `theorems/theorem_substrate_agnosticism.md` + `theorem_toggle_from_self_containment.md` remark "On (A) applied to spatial structure" |
| (A)+(B)+(I) axiom demotion | 2026-05-07/08 | `theorems/theorem_toggle_from_self_containment.md` |
| Unified-oblique theorem (§8 family) | 2026-05-16 | `theorems/theorem_unified_oblique.md` |
| A_s closure | 2026-05-05 | `theorems/theorem_lattice_coupling_algebraicity.md` chain |
| Multi-axial dark-sector waterfilling theorem | 2026-05-24 | `theorems/theorem_dark_sector_multiaxial_waterfilling_candidate.md` |
| Selection-map theorem | 2026-05-21 | `theorems/theorem_selection_map_2026-05-21.md` |
| d/u split via conjugate Higgs | 2026-05-21 | `theorems/theorem_updown_split_conjugate_higgs_2026-05-21.md` |
| Sector-specific c_color = 1/4 (BS-T × J) | 2026-05-26 | `theorems/theorem_alpha_GUT_sector_specific_c_BST_J_2026-05-26.md` |
| Sector-specific c_EW = 1/3 (Z_k*-saturation) | 2026-05-26 | `theorems/theorem_Z_k_star_saturation_c_EW_2026-05-26.md` |
| M_persistence 12-mass fermion operator | 2026-05-26 | `predictions/M_persistence.py` + `_derivation.md` |
| Quark Koide ε²(n) Need-B closure (W4) | 2026-05-26 | `theorems/theorem_quark_koide_eps_n_2026-05-26.md` |
| **V_Ram ≅ Cl(6) Fock ISO — unified SM flavor framework (T1-T5)** | **2026-05-26** | **`theorems/theorem_V_Ram_Cl6_Fock_iso_2026-05-26.md`** |
| Cosmic birefringence β uniqueness | 2026-04-29 | `theorems/theorem_beta_uniqueness_closure.md` |
| η_B Sakharov-Hashimoto chain | 2026-04-30 | (per Row P28 of parameter_uniqueness_ledger) |
| δ_CP^PMNS revival via V_{−1}–T_{B-L} | 2026-05-05 | `predictions/delta_CP_PMNS_derivation.md` §6.1 |
| m_ν₃ global spectral gap | 2026-05-04 | `predictions/m_nu3_derivation.md` |
| V_us Level-2 density | 2026-04-22 | `predictions/V_us_derivation.md` |
| V_cb Hashimoto BFS + A2 resummation | 2026-04-21 | `predictions/V_cb_derivation.md` |
| V_ub M1 twisted walker | 2026-04-30 | `predictions/V_ub_derivation.md` |
| Lorentz arc joint closure | 2026-04-27 | `theorems/lorentz_sig_ccclose_joint_closure.md` |
| Gauge cluster 5-stage closure | 2026-05-04 EOD+1 | `theorems/theorem_alpha_GUT.md` chain |

---

## Out of scope for this plan

- Detailed content of individual theorems — see `theorems/theorem_*.md` files
- Per-prediction numerical breakdowns — see `predictions/*_derivation.md` files and `parameters/parameter_uniqueness_ledger.md`
- Physics identification beyond what A5-mass declares
- Non-SM / non-BSM-observable content
- The dark-correction Class 1-4 taxonomy from the 2026-04-15 era — most of its examples (V_us, m_ν, V_cb) have been superseded by incompatible live derivations; the taxonomy itself is no longer load-bearing.
