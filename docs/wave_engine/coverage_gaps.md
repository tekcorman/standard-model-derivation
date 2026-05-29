# Coverage Gaps (Pass A) — Catalog Completeness Audit

**Status:** High-priority items closed 2026-04-27. Medium- and lower-priority items still scoping.

## 2026-04-27 closure pass (T1.3 focused)

The three highest-priority gaps below are now in the catalog (`../operator_sweep/operator_sweep_from_A1.md`):

- **Layer 5.I — Anomaly machinery** (5 ops: ABJ, Wess-Zumino, anomaly inflow, anomaly cancellation, 't Hooft matching). Section "5.I — Anomaly machinery" in the catalog. All `STRUCT`-only (consistency constraints, no compression). Fires under standard tag preconditions.
- **Layer 5.J — S-matrix / asymptotic states / LSZ** (6 ops: in/out states, S-matrix, LSZ, unitarity, cluster decomposition, cross-section). Section "5.J — S-matrix / asymptotic states / LSZ". All `STRUCT`-only. Fires under STRAUCH + FF.
- **Layer 7 — Connes spectral triples / NCG** (13 ops: 7.A spectral triple structure + 7.B Connes' apparatus). Now load-bearing for the NCG-flavored route to gauge structure / distance / spectral action. Section "Layer 7 — Non-commutative geometry". All but 7.13 (`K_THEORY` template) are `STRUCT`. The bounded-D² obstruction (per `memory/project_lorentzian_signature_route_c_blocked_2026-04-26.md`) is documented as a Layer 7 trap rather than a missing op.

Catalog size: 195 → 219 ops (+24 from T1.3). Wave simulator fires all 24 new ops (preconditions met under existing tag set). No new tags introduced.



The 195-op catalog (`../operator_sweep/operator_sweep_from_A1.md`) was extracted from the framework's existing audits — it covers what the framework has *already used*, not necessarily what it eventually needs. For the wave engine to be exhaustive (closed under physics-structure reachability), the catalog must contain every operation that load-bears for any current or near-term derivation target.

## Method (proposed)

Three structured cross-checks:

1. **Against `target_parameters.md`.** For each of the 75–85 SM+cosmo parameters, identify the load-bearing catalog ops. If a parameter has no covering ops, that's a gap.
2. **Against the 6 active workstreams' forward queues.** For each pending step in `workstream_*.md`, identify the catalog ops it needs. Gaps surface as ops the queue references but the catalog doesn't list.
3. **Against standard QFT structure.** Walk through Weinberg / Peskin-Schroeder topic by topic; flag any structural primitive without a catalog entry.

## Known gaps (initial pass)

These were surfaced during the wave-simulator session 2026-04-26. Not yet addressed in the catalog.

### Highest priority — CLOSED 2026-04-27 (T1.3 pass)

- ✅ **Connes spectral-triple ops** — Layer 7 added (13 ops: 7.1–7.13).
- ✅ **Anomaly machinery** — Layer 5.I added (5 ops: 5.39–5.43).
- ✅ **S-matrix / asymptotic states / LSZ** — Layer 5.J added (6 ops: 5.44–5.49).

### Medium priority (relevant to upcoming work)

- **Wilsonian RG / explicit renormalization.** 4.52 has "RG flow" as a single op; Wilsonian step-by-step coarse-graining and matching aren't represented.
  - `Wilsonian effective action at scale Λ`
  - `Polchinski equation`
  - `matching condition between EFTs`
  - Framework's renormalization workstream (F7 in field-operator cascade) will need these.

- **Operator product expansion (general).** A.21 is CFT-specific OPE/Virasoro; no general OPE op for QFT.
  - `OPE coefficients C_{ABC}(x-y)`
  - `OPE associativity (crossing)`

- **Generalized global / higher-form symmetries** (Gaiotto-Kapustin-Seiberg-Willett 2014+). Recent QFT structure with no catalog presence. Relevant to anomaly understanding.
  - `1-form symmetries acting on Wilson lines`
  - `2-form symmetries`
  - `higher-group symmetries`
  - `'t Hooft anomaly matching`

- **Lattice gauge theory.** The framework IS lattice in some sense (srs); explicit lattice gauge-theory ops would systematize.
  - `link variables U_μ ∈ G`
  - `plaquette action`
  - `Wilson loops on lattice`
  - `transfer matrix`

### Lower priority (speculative or framework may not need)

- **Causal-set theory specifics.** Sorkin-style causal-set ops for the Gorard 2020 direction. Even if framework chooses Connes route, having causal-set ops catalogued enables comparison.
  - `causal poset (S, ≺)`
  - `discrete-to-continuum sprinkling`
  - `Benincasa-Dowker action`

- **Higher categorical structure.**
  - `∞-categories`
  - `derived algebras`
  - `homotopy types as types`

- **BV-BRST quantization.**
  - `BRST charge Q²=0`
  - `ghost number grading`
  - `BV anti-bracket`

- **Bootstrap methods.**
  - `conformal bootstrap`
  - `S-matrix bootstrap`
  - `crossing equations`

- **Quantum groups / Hopf algebras.**
- **Yangian / quantum integrability.**
- **Operads / cooperads.**
- **Geometric quantization** (Souriau, Kostant).

## Coverage-gap addition workflow

For each gap identified:

1. **Define the op** in the catalog format: `(id, layer, name, template, L, extras, refinement)`.
2. **Identify dependencies.** Which existing tags/ops does it require? Add to `extras`.
3. **Identify what it establishes.** Does it open new tags? Add to `ESTABLISHES`.
4. **Identify its Φ template.** Is it `STRUCT`, a new template, or one of the existing templates?
5. **Hand-rate L.** Pending Pass B replacement with formal encoding.
6. **Re-run the wave simulator.** See if the new op fires and what it unlocks downstream.
7. **Document in the catalog** (`../operator_sweep/operator_sweep_from_A1.md`) and update the strict-table (`compressibility_table_strict.md`).

## Estimated effort

- **High-priority gaps (Connes ops, anomaly, S-matrix):** ~1 session each.
- **Medium-priority gaps:** ~0.5 session each, depending on how mathematically novel.
- **Lower-priority:** can be batched; not load-bearing for current workstreams.

**Total Pass A first-pass:** ~3–5 sessions to close high+medium priority. Lower priority can be added incrementally as workstreams progress.

## Status

- **Initial gap list compiled** (this doc).
- **Cross-check against `target_parameters.md`:** pending.
- **Cross-check against workstreams:** partial (Field Operator Cascade flagged S-matrix / LSZ — closed 2026-04-27).
- **Cross-check against standard QFT topics:** pending.
- **Op specifications written:** high-priority done (Layer 5.I, 5.J, 7); medium and lower priority pending.
- **Catalog updated:** Layer 5.I, 5.J, 7 added 2026-04-27 (195 → 219 ops).

## Open questions

- Should "Connes spectral-triple ops" be a new Layer 7, or a new Appendix sub-section?
- For `STRUCT`-only ops (no compression contribution), is it worth expanding the catalog at all? The wave engine ignores them unless they establish a tag or refinement. A coverage-completeness argument says yes; an A2-strict argument says only the ones that compress matter.
- How should anomaly cancellation be modeled? It's a *consistency requirement* on chiral fermion content, not an op per se. Maybe it lives outside the catalog as a constraint that the catalog ops together must satisfy.
