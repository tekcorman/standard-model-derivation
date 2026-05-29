# Wave Engine

A derivation engine for the framework: **wave propagation through a DAG of operations, gated by a compression metric**, that generates the framework's physics ontology from A1+A2 mechanically.

The DAG is the 195-op operator catalog from `../operator_sweep/operator_sweep_from_A1.md`. The compression metric is identical-state counting on the multiway graph F_inv(E). The wave starts at {A1, E_FIN} and propagates by firing operations whose preconditions are met, accumulating substrate-side compression Φ and emitting derived objects. Predictions sit on top of the wave's halting state and contribute observable-side compression Φ_obs against measured data. The full bit budget is `B = Φ_substrate + Φ_observable − L_substrate − L_predictions`.

## Why this exists

The 2026-04-26 evening session formalized what the framework had been doing implicitly: every derivation is a sequence of catalog ops gated by A2 (retain ops with positive compression savings). Making this mechanical converts an audit-style framework into a generative one — predictions become wave-reached nodes you didn't write down, open structural problems become wave-frontier nodes the cascade halts at, and falsifiability gets sharp (mismatch at a wave-reached observable falsifies either the substrate or the A2 reading).

The wave-simulator session also pinned `CCLOSE` as the framework's single open-frontier tag, identified Connes' spectral-action route as the tractable closure path, and validated the mechanism via a 5-prediction bit-budget pilot.

## Files in this folder

| file | what |
|------|------|
| `README.md` | this orientation |
| `mechanism.md` | formal description: state, gating, cascade modes, substrate-vs-observable Φ split |
| `simulator.md` | wave-simulator usage + lean and full-cascade results |
| `audit_pilot.md` | per-prediction bit-budget audit methodology + pilot results on 5 predictions |
| `closure_experiment.md` | §C closure experiment results + interpretation |
| `compressibility_table.md` | first-pass per-op (Φ, L, Net) table over 195 ops |
| `compressibility_table_strict.md` | strict-assumption edition with explicit assumption-tag column |
| `coverage_gaps.md` | Pass A — catalog completeness audit (in progress) |
| `cost_methodology.md` | Pass B — formal L-encoding scheme (first-pass; calibration deferred) |
| `substrate_state.md` | T2.1 — live executor (lean ops verified, expansion pending) |

## Code (`proofs/wave_engine/`)

| file | what |
|------|------|
| `simulator.py` | full 195-op wave simulator with tag-establishment dynamics |
| `simulator_lean.py` | minimal lean-catalog simulator (Layer 0+1 only) |
| `audit_pilot.py` | v3 canonical bit-budget pilot (23 predictions, chain-attributed) |
| `closure_experiment.py` | §C closure experiment script |
| `substrate_state.py` | T2.1 live executor (word-layer + graph-layer ops verified) |

## Quick start

```bash
# Run the lean simulator (verifies the mechanism on Layer 0+1)
python3 proofs/wave_engine/simulator_lean.py

# Run the full 195-op simulator (wave propagates through the framework's ontology)
python3 proofs/wave_engine/simulator.py

# Run the §C closure experiment
python3 proofs/wave_engine/closure_experiment.py

# Run the v3 canonical audit pilot (23 predictions, chain-attributed)
python3 proofs/wave_engine/audit_pilot.py

# Run the T2.1 live executor (lean cascade verification)
python3 proofs/wave_engine/substrate_state.py
```

## Status (2026-04-27, post T1.1 template-dedupe)

- Lean simulator: validated, mechanism halts coherently at substrate ontology
- Full simulator: 173/195 ops fire under fire-every-firable cascade; CCLOSE + LORENTZ_SIG are the open frontier tags
- Tag-split (2026-04-27): two distinct frontiers — CCLOSE (smooth-manifold, 15 ops) + LORENTZ_SIG (signature, 1 op + 6 with CCLOSE). LORENTZ_SIG BLOCKED at substrate level by bounded-D² obstruction (`memory/project_lorentzian_signature_route_c_blocked_2026-04-26.md`).
- T1.1 marginal-Φ template dedupe: total Φ dropped 183.34 → 94.15 (95-bit overcount removed). Framework substrate-deficit deepened from −47 → −136 bits.
- T2.1g live-Φ override (2026-04-27): `--live` flag in `simulator.py` substitutes live executor values for ops 4.21, 2.18, 4.17, 5.9. Net: +1.0 bit substrate Φ (94.15 → 95.15). Live override BYPASSES template dedupe — surfaces ~1 bit of compression that 2.18/4.17 contribute independently of 2.17 even though they share BLOCH_SRS template.
- Closure experiment under T1.1: §C closure adds ZERO substrate Φ — all unlocked ops are STRUCT carriers; compression payoff is at prediction layer.
- Audit pilot v3 (T1.4 chain attribution + 23 predictions across 8 chains): pilot avg **+4.45 bits/pred (flat) / +1.22 bits/pred (T1.5 Bayes)**. Projections to 45 theorem-grade: flat +200 bits (+64.5 net positive), Bayes +55 bits (−80.7 deficit). T1.5 surfaces an Occam factor (½log₂(2π) ≈ 1.33 bits/pred) and χ² fit penalty plus σ_eff=√(σ_obs²+theoretical-uncertainty band [RETRACTED]²) compression — the deficit is REAL once framework's intrinsic prediction uncertainty is honestly accounted for.
- Spectral-action route handoff: an internal note — Steps 2/3/5 partially blocked by same bounded-D² finding.
- Lorentzian signature alternative routes handoff: an internal note — three research-level routes (Krein-NCG, BLMS causal-set, Dirac-point); recommended start: P-point Dirac check (~1 session).

## Open work

Tier-1 (bounded improvements):
- **T1.1 — marginal-Φ template dedupe.** ✅ DONE 2026-04-27. 95-bit Φ overcount removed.
- **T1.2 — formal L encoding.** First-pass complete 2026-04-27. Layer-base + template-complexity scheme: +331 bit delta vs hand-rated; substrate deficit shifts from −136 to −262. **Closure conclusion sensitive to L encoding** — hand-rated L gives +64 net positive; formal L gives −62 deficit. Calibration requires either external reference (token grammar / compression proxy) or T2.1 substrate executor. Recommendation: skip T1.2 second-pass, prioritize T2.1. Scope: `cost_methodology.md`.
- **T1.3 — Pass A coverage audit (focused).** Pending. Add Connes spectral-triple ops, S-matrix/LSZ, anomaly machinery. Scope: `coverage_gaps.md`.
- **T1.4 — chain-attribution for predictions.** ✅ DONE 2026-04-27. v3 audit pilot canonicalized with 23 predictions across 8 chains, chain-amortized L. Result: +64.5 bits net positive at projected scale.
- **T1.5 — Bayesian observable-side Φ.** ✅ DONE 2026-04-27. Bayes-factor formulation: Φ_obs = log₂(W/σ_eff) − ½log₂(2π) − χ²·log₂(e)/2 with σ_eff² = σ_obs² + theoretical-uncertainty band [RETRACTED]². Adds Occam factor (1.33 bits/pred) + fit penalty. Headline impact: per-prediction avg drops from +4.45 (flat) to +1.22 (Bayes); 45-prediction projection drops from +200 → +55 bits. Framework moves from +64.5 (flat) to **−80.7 (Bayes deficit)** at projected scale. The flat-prior reading was systematically over-optimistic.

Tier-2 (architectural shift):
- **T2.1 — substrate state representation.** First deliverable + T2.1e complete 2026-04-27. SubstrateState class + 3 lean ops (involutivity, conjugation, abelianization) + 3 graph-layer ops (srs quotient, adjacency, Hashimoto) live; all verified against analytic counts. Word-layer surfaces ~+1 bit Φ correction (25/24 asymptotic ratio); graph-layer T2.1e disentangles op 2.18 Φ = log₂(3/2) ≈ 0.585 bits from BLOCH_SRS template's 3.0 bits (~2.4-bit overcount per Hashimoto firing). h_max(Γ) = 2 verified; Ihara factorization u² − λu + 2 = 0 verified live. Expansion pending: Bloch decomposition (T2.1d), JW (T2.1f). Scope: `substrate_state.md`.
- **T2.2 — op execution functions.** Pending. Implement each op as live function on SubstrateState (incremental T2.1 expansion).
- **T2.3 — extras derivation.** Pending. Auto-derive op preconditions from execution graph.
- **T2.4 — Φ-template values computed live.** Pending. Replace lookup with actual partition arithmetic.
- **T2.5 — the author's separate private derivation DAG integration.** Pending. Cross-link catalog ops to the author's separate private derivation-DAG nodes.
- **T2.6 — wave-reached prediction generation.** Pending. Auto-enumerate candidate predictions from wave halting state.

Tier-3 (research-level): formal verification (Lean/Coq), self-modifying catalog, bidirectional engine, live observable likelihood machinery.
