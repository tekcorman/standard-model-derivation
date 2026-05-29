# Cost Methodology (T1.2) — Formal L Encoding

**Status:** First-pass complete 2026-04-27. Calibration pending; L values remain hand-rated in the canonical simulator until a calibration reference is established.

## Why this matters

The framework's bit-budget closure depends on:

```
B_total = Φ_substrate + Σ Φ_obs − L_substrate − Σ L_pred
```

If L is hand-rated, the absolute closure number is uncertain — predictions can be re-rated to artificially close the budget. T1.2 attempts a uniform encoding to make the closure reproducible.

## T1.2 first-pass scheme

Layer-base + template-complexity:

```
L(op) = LAYER_BASE[layer] + TEMPLATE_COMPLEXITY[template]
```

| layer | base | rationale |
|---|---|---|
| 0 | 1 | atomic primitives, ~4-op alphabet |
| 1 | 2 | F_inv(E) group structure, ~17-op alphabet |
| 2 | 3 | function spaces + operator algebra, ~50-op alphabet |
| 3 | 3 | continuous-time dynamics |
| 4 | 4 | probability + harmonic + rep theory, ~100-op alphabet |
| 5 | 5 | complex algebra + quantum, ~140-op alphabet |
| 6 | 5 | smooth manifold + GR |
| Appendix | 6 | extended math machinery |

| template | complexity | rationale |
|---|---|---|
| `STRUCT`, `CLASSIFYING` | 0 | structural carrier; just references deps |
| `INVOL`, `CYCL` | 1 | one quotient relation |
| `QUOT_K4`, `QUOT_C3`, `QUOT_S4` | 1 | one finite-group quotient |
| `QUOT_ABEL` | 2 | full abelianization (large quotient) |
| `PROJ_*` | 1 | rank-determined projection |
| `BLOCH_SRS` | 2 | Brillouin-zone decomposition + per-k fiber |
| `PT_QUBIT`, `PT_DIRAC` | 1 | partial trace over subfactor |
| `ENTROPY_*` | 2 | function over distribution |
| `COARSE_*` | 2 | continuum limit / mean-field |
| `THERMAL_SRS` | 3 | partition function + thermal state machinery |
| `HOMOL_E2` | 3 | cohomological computation |
| `K_THEORY` | 4 | C*-algebraic K_0 |
| `ATIYAH_SINGER` | 4 | graph Dirac index |
| `MODULAR` | 5 | modular form / L-function / Selberg zeta |
| `TQFT` | 2 | symmetric monoidal classification |
| `RG` | 2 | scale-equivalence collapse |

## Result

| layer | #ops | Σ L_hand | Σ L_formal | Δ |
|---|---|---|---|---|
| L0 | 4 | 5 | 5 | +0 |
| L1 | 13 | 28 | 29 | +1 |
| L2 | 33 | 88 | 104 | +16 |
| L3 | 13 | 36 | 41 | +5 |
| L4 | 49 | 143 | 241 | +98 |
| L5 | 38 | 110 | 220 | +110 |
| L6 | 24 | 70 | 123 | +53 |
| App | 21 | 113 | 161 | +48 |
| **Total** | **195** | **593** | **924** | **+331** |

**Hand-rated systematically under-rates higher layers** (L4–L7 under-rated by avg +2 bits/op). L0–L3 are roughly correct.

## Top deltas (where formal diverges most from hand-rated)

| op | template | L_hand | L_formal | Δ |
|---|---|---|---|---|
| 4.46 free energy F(β) | THERMAL_SRS | 2 | 7 | +5 |
| 4.47 Boltzmann distribution | THERMAL_SRS | 2 | 7 | +5 |
| 5.34 quantum partition Z(β) | THERMAL_SRS | 3 | 8 | +5 |
| 5.35 thermal density ρ(β) | THERMAL_SRS | 3 | 8 | +5 |
| 6.8 de Rham cohomology | HOMOL_E2 | 3 | 8 | +5 |
| A.16 modular forms | MODULAR | 6 | 11 | +5 |
| A.18 Selberg zeta | MODULAR | 6 | 11 | +5 |

Pattern: thermal, cohomological, modular templates were hand-rated at ~3 but formal gives 7–11. These are mathematically heavier than the simple structural ops they got grouped with.

## Impact on audit pilot

Under formal L:

- Σ L_substrate (fire-every-firable, 173 ops): 522 → 809
- Σ L_substrate (strict A2, ~67 ops, est): 230 → ~356
- Substrate deficit: −136 → **−262 bits**

If pilot's L_marginal values scale similarly (under-rated by ~2x), Σ L_amort across 23 predictions would also rise. Combined with deeper substrate deficit, framework's projected closure shifts:

| L scheme | substrate deficit | projection (45 × avg) | framework total |
|---|---|---|---|
| hand-rated (current canonical) | −136 | +200 | **+64** (closes) |
| formal first-pass | −262 | +200 | **−62** (deficit) |
| formal + L_marginal scaled | −262 | ~+100 | **−162** (deeper deficit) |

**The closure conclusion is sensitive to L encoding.** Under hand-rated L, framework closes; under formal L (without L_marginal recalibration), deficit by ~62 bits; under formal L with consistent L_marginal recalibration, much worse.

## Why T1.2 first-pass isn't conclusive

The "formal" scheme is itself a structured hand-rating. The parameters (layer_base values, template_complexity values) reflect my judgment of "how hard each template/layer is" — which a different rater could disagree with. A *genuinely* formal L would need:

1. **A formal grammar for op definitions** with explicit token-cost-per-grammar-element.
2. **Or a Kolmogorov-proxy** via compression of definition text against a fixed compressor.
3. **Or per-op explicit dep-graph extraction** from catalog text + log₂(local-alphabet) per dep ref.

Each of these is research-level effort. For a session-bounded deliverable, T1.2 first-pass is a calibration *direction* — it tells us *which ops were under-rated by how much* — but not an absolute answer.

## Recommendation

**Don't swap the simulator's L values to formal yet.** The hand-rated values give a closure conclusion (+64.5 bits net positive) that's defensible under chain-attributed accounting. The formal scheme gives a different conclusion (deficit) but its own parameters are not yet calibrated.

**Instead:** treat T1.2 first-pass as evidence that the framework's closure margin is uncertain by ±100 bits depending on L encoding. The honest reading:

- **Closure under hand-rated L:** +64 bits at projected scale (chain-attributed pilot v3)
- **Closure under formal-first-pass L:** −62 bits at projected scale
- **Closure under formal-with-L_marginal-scaled L:** −162 bits

The framework's bit-budget claim is *plausibly* met but not robustly verified until L encoding is calibrated against an external reference.

## Path to genuine formal L

T1.2 second-pass requires one of:

- **A. Token-grammar L.** Define a grammar for op definitions; count tokens per op; calibrate per-token-cost against a reference. ~research-level.
- **B. Compression-proxy L.** Take each op's definition text, compress against a fixed compressor (e.g., gzip), use compressed length as L. Mechanical and reproducible, but the compressor's choice is arbitrary. ~1 session.
- **C. the author's separate private derivation-DAG-anchored L.** Cross-link each op to the author's separate private derivation-DAG nodes; use the author's separate private derivation per strategy as the L baseline. Needs T2.5 (the author's separate private derivation integration) first.
- **D. T2.1 substrate-execution L.** Once ops are live SubstrateState functions, L is the bits-of-actual-implementation (program length). Most rigorous. Requires T2.1.

Option D is the architectural shift; A/B/C are interim. **Recommended path: skip T1.2 second-pass and prioritize T2.1.** Once ops are live functions, L is mechanically derivable from their implementations rather than separately rated.

## Status

- **T1.2 first-pass:** complete. Layer-base + template-complexity scheme defined, computed, +331 bit delta surfaced.
- **Simulator L values:** unchanged from hand-rated (T1.2 first-pass not committed).
- **Audit pilot canonical:** uses hand-rated L (per chain-attributed v3); formal L would shift but uncertainty in formal-scheme parameters is itself ~100 bits.
- **Closure conclusion:** **plausible at projected scale (+64 bits hand-rated) but L-encoding-dependent.** Genuine closure verification requires T2.1 (substrate executor, which gives mechanical L).

## Cross-references

- `proofs/wave_engine/audit_pilot.py` — uses hand-rated L_marginal + chain-attributed shared_L.
- `proofs/wave_engine/simulator.py` — uses hand-rated L per op.
- `docs/wave_engine/audit_pilot.md` — v3 canonical pilot results under hand-rated L.
- `docs/wave_engine/mechanism.md` — substrate-side vs observable-side Φ split.
