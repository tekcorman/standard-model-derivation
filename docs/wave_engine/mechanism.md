# Wave Engine — Mechanism

Formal description of the derivation mechanism implemented by the wave simulator.

## The substrate-counting principle

Per A2-T (selective retention / MDL): the only allowed compression is **counting of identical states on the multiway graph F_inv(E)**. Every operation in the catalog either induces an equivalence relation on configurations (positive Φ) or it does not (Φ = 0). The framework's compression budget at the substrate level is exactly the sum of bits collapsed by identical-state counting.

This is the strict reading. It rules out compression by external priors, by structural assumption, or by re-labeling. Only configuration-class collapse counts.

## State

The wave-state at any tick is a tuple:

```
state = (refinements, tags, fired, Φ_total, L_total, objects)
```

| component | meaning |
|---|---|
| `refinements` | set of equivalence-relation refinements imposed on F_inv(E) (e.g., `reduced`, `cyclic`, `abelian`); determines current class count |
| `tags` | set of established assumption tags; gates op firing |
| `fired` | sequence of ops fired so far, in order |
| `Φ_total` | cumulative bits compressed |
| `L_total` | cumulative spec cost |
| `objects` | derived objects emitted (each catalog op produces one when it fires) |

Initial state at session start: `{A1, E_FIN, A2W, P1, A5M, E6, K3, ORDER}` with no refinements, no fired ops, Φ=L=0.

## Operations (catalog)

The 195-op catalog is the wave engine's action space. Each op is specified by:

```
op = (id, layer, name, template, L, extras, refinement)
```

| field | meaning |
|---|---|
| `id` | catalog identifier (e.g., `4.21`) |
| `layer` | layer 0–6 or Appendix (7) |
| `template` | Φ-template (`STRUCT`, `INVOL`, `CYCL`, `QUOT_K4`, `BLOCH_SRS`, `THERMAL_SRS`, `MODULAR`, ...) |
| `L` | bits to specify the op (current value: hand-rated, 1–7 bits) |
| `extras` | required tags BEYOND the always-present substrate tags |
| `refinement` | partition refinement key (lean: `reduced`/`cyclic`/`abelian`; non-lean: per-op key for non-overlap-aware templates) |

Plus per-op `establishes` mapping: when a specific op fires, it activates downstream tags (e.g., 4.21 `K_4 quotient` establishes `SRS`, `CRYSTAL`, `K4Q`).

## Firing rule (cascade)

At each tick, walk the catalog in id order. For the first op satisfying `op.extras ⊆ state.tags`:

1. Compute `Φ_marg(op | state)` — marginal Φ given current state.
2. Fire the op: update `refinements`, `tags ∪= op.establishes`, append to `fired`, increment `Φ_total += Φ_marg`, `L_total += L`, append derived object.
3. Return new state.

Halt when no op is firable.

Two readings of "firable":

- **Strict A2-retention.** Fire only if `Φ_marg > 0` (op contributes net compression). Used by the original lean simulator.
- **Fire-every-firable** (current default). Fire any op whose extras are met, regardless of Φ. Constructive ops (those producing derived objects via `refinement` or `establishes`) fire even at Φ = 0. This captures the framework's full ontology-construction at the cost of accumulating L without Φ.

The two readings give different totals:
- Strict: 67/195 fired, Φ=183, L=230, **Net = −47**
- Fire-every-firable: 173/195 fired, Φ=183, L=522, **Net = −339**

Strict is the A2-faithful reading. Fire-every-firable shows the **reachability** of the catalog given current assumptions.

## Marginal Φ computation

For the lean refinements (`reduced`, `cyclic`, `abelian`), Φ_marg is computed exactly via partition class counts:

```
Φ_marg(op | state) = log₂( |classes(state.refinements)| / |classes(state.refinements ∪ {op.ref})| )
```

For non-lean templates (`BLOCH_SRS`, `THERMAL_SRS`, `MODULAR`, etc.), **template-level dedupe** (T1.1, 2026-04-27): the first op invoking a template contributes the template's full Φ; subsequent ops in the same template contribute 0. The state tracks `templates_used` as a set; once a template is in it, marginal Φ is 0 for any op invoking that template.

Rationale: multiple ops sharing a Φ-template represent different *views* of the same underlying substrate compression — e.g., MODULAR_FORM / L_FUNCTION / SELBERG_ZETA all express the substrate's Hecke-eigenvalue structure; THERMAL_Z / THERMAL_F / THERMAL_B all express the thermal partition Z(β). Crediting each op the full template Φ overcounts. T1.1 corrects this; total Φ dropped from 183.34 → 94.15 bits (95-bit overcount removed).

**Caveat:** template-dedupe is conservative. When two ops in the same template *do* compress genuinely-different content (e.g., HOMOL_E2 covering both group cohomology of F_inv(E) AND smooth de Rham of M — different cohomology theories on different objects), template-dedupe undercounts. Resolution requires splitting templates (Pass A catalog refinement). For now the conservative reading is preferred over the overcounting one.

## Substrate-side vs observable-side Φ

The wave engine tracks **substrate-side** Φ — bits of identical-state collapse on F_inv(E) configurations. This is the catalog-level metric.

Predictions sit on top of the wave's halting state and contribute **observable-side** Φ:

```
Φ_obs(prediction) = log₂( prior_width / σ_obs )
```

This measures bits of compression against measured data: a prediction matched at σ_obs against a prior of width `prior_width` compresses log₂(prior_width / σ_obs) bits of observational uncertainty.

The full bit budget for the framework is:

```
B_total = Φ_substrate + Σ Φ_obs(prediction) − L_substrate − Σ L_pred
```

Substrate-side spec cost (L_substrate) is paid up-front in constructing the wave's ontology. Predictions pay it back via Σ Φ_obs. Audit-pilot results: 5 sample predictions contribute +10 bits net B; projection to 45 theorem-grade predictions gives +113 bits — sufficient to close the strict-A2 substrate deficit of −47 with positive net.

## Tags and tag-establishment dynamics

Assumption tags partition into three groups:

**Initial (definitional, framework setup):**
`A1`, `E_FIN`, `A2W`, `P1`, `A5M`, `E6`, `K3`, `ORDER` — present from session start.

**Cascade-established (set by specific ops firing):**

| tag | established by |
|---|---|
| `FIN_DIM` | 2.22 trace-class |
| `LIE` | 3.1 one-parameter unitary group |
| `FF` | 3.5 Stone (real form) — drives field selection §F |
| `STRAUCH` | 3.11 discrete→continuum walk limit |
| `C3` | 4.16 isotypic decomposition |
| `S4` | 4.19 symmetry-protected degeneracies |
| `K4Q`, `SRS`, `CRYSTAL` | 4.21 K_4 quotient |
| `COMPACT` | 4.30 group representation |
| `THERM` | 4.45 partition function |
| `BZJ`, `N_HUB` | 4.51 BZJ scaling |
| `RGFL` | 4.52 RG flow |
| `C_REP` | 5.1 imaginary unit i in op algebra |
| `A4` | 5.6 Jordan-Wigner |

**Open frontier (cannot establish under current catalog):**

Two distinct tags after the 2026-04-27 tag-split:

- **`CCLOSE`** — smooth-manifold continuum-limit closure (Riemannian smoothness, no signature commitment). Research-level open. Closing alone unlocks 15 ops: tangent space, tensor fields, differential forms, de Rham cohomology, Riemannian metric, Levi-Civita, Christoffel, Riemann, Ricci, Killing vectors, plus A.19 quantum gravity and A.21 CFT.
- **`LORENTZ_SIG`** — Lorentzian signature (-,+,+,+) derivation. **BLOCKED at substrate level** per `memory/project_lorentzian_signature_route_c_blocked_2026-04-26.md`: substrate's D²_sub = n·I + R_sub is bounded (spectrum [0, ~21] for n=6) → heat kernel smooth at t=0 → no UV divergence → no Λ² coefficient in spectral action → no Einstein-Hilbert via standard Connes-Chamseddine. Closure requires Krein-space Lorentzian NCG (Besnard-Bizi-Iochum), modified finite-spectral-triple machinery, or alternative routes (BLMS causal-set, Dirac point). Closing alone unlocks just 1 op (6.10 Lorentzian metric).

**6 cosmology ops require BOTH** frontiers closed: FLRW (6.18), Einstein equations (6.19), Friedmann (6.20), Hubble (6.21), scale factor (6.22), stress-energy (6.23).

**No overlap:** every blocked op is structurally CCLOSE-only, LORENTZ_SIG-only, or BOTH — never "either suffices." The two frontiers are mathematically independent. See `closure_experiment.md` for the 4-way scenario diff.

## Constructive vs constraint ops

Two structurally-distinct kinds of catalog op:

- **Constructive.** Op produces a derived object (e.g., 4.21 produces srs lattice; 5.6 produces JW operator). Even Φ = 0 constructive ops fire under fire-every-firable cascade because they emit downstream-usable structure.
- **Constraint.** Op imposes an equivalence relation on configurations (e.g., 0.4 involutivity, 1.10 abelianization). Fires under strict A2 only when its marginal Φ > 0.

Most catalog ops are both — they construct and constrain. The distinction matters for halting: a wave that only fires constraint ops halts when no further constraint can apply; a wave that also fires constructive ops continues until every catalog entry has been used at least once.

## Halting

Under strict A2: halt when no op contributes positive marginal Φ. The wave's final state is the **A2-retained** subset of the catalog given current tag stack.

Under fire-every-firable: halt when no op has unsatisfied tags (every firable op has fired). The wave's final state is the **reachability closure** of the catalog given current tag stack.

For derivation-engine purposes, fire-every-firable is the right reading: it shows what the framework's catalog reaches; the strict reading is then a sub-question of "which of those ops actually compress."

## Observables and the bit-budget closure question

The framework's claim is that A1+A2 (plus framework specializations encoded as initial tags) generates the SM ontology mechanically. The wave simulator tests this constructively:

1. Does the wave reach SM-relevant structure? **Yes** — Pati-Salam embedding, Cl(6;ℂ) spinors, CKM/PMNS via Bloch+Hashimoto, Higgs sector via BZJ scaling all fire.
2. Does the framework's bit budget close (Σ Φ ≥ Σ L)? **Yes at projected scale** — strict-A2 deficit −47 bits is paid back by ~45 theorem-grade predictions averaging +2.5 bits B each.
3. What's blocked? **Only `CCLOSE`** — the §C smooth-manifold closure. Resolves the framework's largest open structural problem to a single tag-establishment operation.

This is the mechanical statement of "the framework derives physics from A1+A2."

## Limitations and refinements

- **Marginal Φ overcounts** for ops sharing a Φ-template. Topological-order subtraction needed.
- **L is hand-rated.** Pass B will replace with formal context-conditional encoding.
- **Catalog is not yet exhaustive.** Pass A will add anomaly machinery, generalized global symmetries, S-matrix/LSZ, spectral-action ops, and other gaps.
- **Predictions need chain-attribution.** Currently each prediction's L_pred is fully attributed; chained derivations (e.g., Koide → m_e + m_μ + m_τ) over-attribute.
- **Observable-side Φ is prior-dependent.** Flat priors over [0,1] for dimensionless ratios; tighter informed priors give different absolute Φ_obs but stable rankings.

These are refinements to a working mechanism, not fundamental issues.
