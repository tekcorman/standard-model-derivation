# Falsification criteria

**Any one would refute the framework.** The framework makes specific numerical predictions, several at the precision frontier. Any of these going against measurement is fatal:

| Prediction | What kills it | Experiment | Timeline |
|---|---|---|---|
| **δ_CP^PMNS = 180°** | Clearly maximal-CP-violating ~270° (outside ~180° ± 30°) | DUNE, Hyper-K | 2028+ |
| θ_23 = 48.72° (non-maximal) | Exactly maximal: 45.00 ± 0.3° | DUNE | 2028 |
| **m_ν₁ = 0 exactly** | 0νββ never observed AND m_ν₁ > 0 established | KATRIN, Project 8, nEXO | 2027+ |
| **N_eff = 3 exactly** (3 ν_L + 3 super-heavy ν_R; no entropy transfer at Phase IIb separation) | N_eff confirmed > 3.04 (i.e., the standard 3.046 ΛCDM value with statistical significance) | CMB-S4 | ~2030+ |
| No WIMP dark matter (DM = gauge-decoupled uncompressed multiway) | WIMP found at LZ / XENONnT | LZ, XENONnT | ongoing |
| SUSY required for consistency (R-parity violated) | SUSY ruled out below 10 TeV | LHC RPV searches, FCC-hh | ongoing → 2040+ |
| \|β cosmic birefringence\| ≤ α_EM ≈ 0.418° (hard cap from c_1 = 0) | β measured > 0.418° | LiteBIRD (~0.05° precision) | ~2032 |
| η_5 = 0 exactly (dim-5 LIV) | dim-5 LIV detected | LHAASO, HESS | ongoing |
| α₂₁ = 162.39° (Majorana phase) | α₂₁ outside ~162° ± 30° | nEXO, LEGEND-1000 | 2030+ |
| α₃₁ = 324.78° (Majorana phase) | α₃₁ outside ~325° ± 30° | future 0νββ | — |
| m_ββ ≈ 2.55 meV (0νββ amplitude from m_ν₂ + α_21 chain) | m_ββ measured outside ~1–5 meV | nEXO, LEGEND-1000 | 2030+ |

## One historical falsification has fired as designed

The original Hashimoto-phase route for $\delta_{CP}^{PMNS}$ predicted **249.85° (±30°)**. It failed at **+3.83σ vs NuFIT 6.0 IC19 on 2026-05-02** and was retired. The current value **180°** came from an *independent parameter-free identity* — $V_{−1}$–$T_{B-L}$ = arccos(−1) — three days later, on 2026-05-05.

The retracted derivation is preserved at [`predictions/retracted/delta_CP_PMNS.py`](https://github.com/tekcorman/standard-model-derivation/tree/main/predictions/retracted) as honest record. The same identity simultaneously fixes $\delta_{CP}^{CKM} = \arccos(1/3) = 70.53°$ at +0.68σ on a *different* observable — independent corroboration.

This is the falsification-and-revival pattern the framework is built around: when a mechanism is wrong, it is retired in the open and any replacement must come from an independent structural argument, not from patching the failed mechanism.

## What this is NOT a list of

Two important distinctions:

- **Not "predictions we'd like to see confirmed."** The above are predictions where *failure is fatal* — not where confirmation is incremental support. The framework's existence-claim depends on these surviving measurement.
- **Not "all open predictions."** The framework has many predictions where the framework's accuracy is already at the precision frontier (e.g., $V_{us} = 9/40$ at $-0.01\sigma$ vs PDG). Those are not on this list because they're already confirmed within experimental uncertainty.

## The rigorous version

- [`docs/honest_assessment.md`](https://github.com/tekcorman/standard-model-derivation/blob/main/docs/honest_assessment.md) §"What would falsify the framework"
- [`docs/parameters/target_parameters.md`](https://github.com/tekcorman/standard-model-derivation/blob/main/docs/parameters/target_parameters.md) — every tracked parameter with current grade
- [`predictions/retracted/`](https://github.com/tekcorman/standard-model-derivation/tree/main/predictions/retracted) — honest archive of derivations that failed re-audit
