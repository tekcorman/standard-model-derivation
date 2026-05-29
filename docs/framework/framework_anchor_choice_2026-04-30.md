# Framework anchor choice — alternatives enabled by G_sub closure

> **SUPERSEDED FRAMING (2026-05-12 — N_hub-pivot).** The framework's one *physical* adopted input is **N_hub ≈ 8.394881e60** (the universe's worldline length / hub count). Everything dimensional is derived from it — including G_F (G_F = 1/(√2 v²), v ← N_hub via BZJ: G_F is a DOWNSTREAM PREDICTION, not an anchor). The *value* of N_hub is currently pinned to ppm by consistency with the measured G_F (a calibration; `predictions/N_hub.py:n_hub_from_g_f_consistency`). A unit-setting constant (M_Pl ≡ G_N ≡ t_Pl, mutually derivable) is the conventional *unit* choice — not a "physics anchor" (and M_substrate = 1 makes M_Pl nearly derived via M_substrate/M_Pl = √π/8). So the "two external dimensional observations / (G_F, M_P) anchor pair" framing below is RETRACTED and re-read as: (one *physical* adopted input: N_hub) + (one unit-convention: any of {M_Pl, G_N, t_Pl}); G_F is downstream. See `simulator.axioms.n_hub_pivot()`. The text below is preserved for the relation-counting / precision discussion only.

**Date:** 2026-04-30 EOD final.
**Status:** SUPERSEDED FRAMING — see the banner above. (Historical: documented the unit-conversion inputs; pre-2026-05-12 this was framed as a "(G_F, M_P) anchor pair". Post-pivot: the one *physical* adopted input is N_hub; the unit-setting constant (M_Pl ≡ G_N ≡ t_Pl) is the conventional unit; G_F is downstream — the observable that calibrates N_hub's value.)

## Background

The framework predicts dimensionless physics from substrate first principles. The dimensional content needs: (1) one *physical* adopted input — **N_hub ≈ 8.4e60** (the universe's worldline length); (2) one unit-setting constant — the conventional unit (any of {M_Pl, G_N, t_Pl}, mutually derivable; M_substrate = 1 makes it nearly derived via M_substrate/M_Pl = √π/8). G_F is NOT an input — it is a downstream prediction (G_F = 1/(√2 v²), v ← N_hub via BZJ) whose measured value is currently used to *calibrate* N_hub's value to ppm precision. [Historical: pre-2026-05-12 this was framed as a "(G_F, M_P) anchor pair" — RETRACTED.]

## Why two anchors are needed

The framework's relation set:
1. **BZJ** (theorem-grade): $v_{\rm GF} = \delta^2 \cdot M_{\rm Pl} \cdot {\rm dark} / (\sqrt{2} \cdot N^{1/4})$
2. **Cascade** (theorem-grade): $t_{\rm now} = N \cdot t_{\rm P}$, $H_0 \cdot t_0 = 1$
3. **Drude** (theorem-grade post G_sub closure): $G_{\rm UV} \cdot M_{\rm substrate}^2 = \pi/64$
4. **Path (b)** (theorem-grade): $M_{\rm Pl}/M_{\rm substrate} = 8/\sqrt{\pi}$ (equivalently $M_{\rm substrate}/M_{\rm Pl} = \sqrt{\pi}/8$)

Combined: 4 framework relations with 5 dimensional unknowns (M_Pl, M_substrate, N, t_now, v_GF). Need ONE more constraint per anchor → 2 anchors total (3+ relations with 5 unknowns → 2 free parameters → 2 anchors).

(Pre-closure: 2 framework relations [BZJ, cascade] + Row 25 commitment with 5 unknowns → also 2 anchors needed. Same count, but G_sub closure makes the relations cleaner and the choice more flexible.)

## Anchor pair options

### Current default: (G_F, M_P)
- G_F = 1.1663787 × 10⁻⁵ GeV⁻² (PDG/MuLan, 0.51 ppm).
- M_P = 1.22089 × 10¹⁹ GeV (CODATA 2018, ~50 ppm; derived from G_N via M_P = √(ℏc/G_N)).
- Pros: G_F at unbeatable precision; M_P is conventional.
- Cons: M_P is derived from G_N — uses gravitational measurement (50 ppm precision) to set Planck scale; conceptually mixes electroweak with derived gravity.

### Cleaner alternative: (G_F, G_N)
- G_F = 1.1663787 × 10⁻⁵ GeV⁻² (0.51 ppm).
- G_N = 6.67430 × 10⁻¹¹ m³/(kg·s²) (CODATA 2018, ~22 ppm; direct gravitational measurement).
- Pros: Two pure fundamental constants. G_N is the directly-measured gravitational coupling, not derived. Cleaner conceptual structure.
- Cons: Same total precision as (G_F, M_P) since M_P = √(ℏc/G_N) is informationally equivalent to G_N.

### Cosmological alternative: (G_F, t_0)
- G_F = 1.1663787 × 10⁻⁵ GeV⁻² (0.51 ppm).
- t_0 = 14.38 ± 0.80 Gyr (Methuselah star, model-independent; or 13.797 ± 0.023 Gyr Planck CMB ΛCDM, model-dependent).
- Pros: t_0 is a model-independent cosmological observation (avoids Hubble tension entirely).
- Cons: Less precise than G_N; the framework's BZJ chain still needs M_Pl, derived from t_0 via cascade theorem.

### Particle-physics alternative: (m_e, G_N)
- m_e = 0.51099895 MeV (CODATA, ~10⁻⁸ precision).
- G_N as above.
- Pros: m_e is the most precisely-measured mass scale. With theorem-grade m_τ/m_e ratio (chain through Koide), this gives v_Higgs (= m_τ/y_τ) at high precision.
- Cons: m_e's chain to v_Higgs goes through more steps (Koide ratios + y_τ), accumulating framework-internal precision losses.

### Atomic alternative: (Rydberg, G_N)
- Rydberg constant R_∞ = 1.0973731568160 × 10⁷ m⁻¹ (CODATA, ~10⁻¹² precision — most precisely measured in physics).
- G_N as above.
- Pros: Rydberg is THE most-precise dimensional constant in physics. Sets atomic/electron scale.
- Cons: Rydberg involves α_EM and m_e via R_∞ = α²m_e c/(2h); the chain to framework primitives isn't direct.

## Recommended choice (revised 2026-05-01 final): **(m_e, G_N)**

After user feedback caught the hidden α_EM dependency in the (R∞, G_N) chain
(R∞ → m_e via Rydberg formula needs α), the cleanest pair is:

  **m_e (electron mass) + G_N (Newton's gravitational constant)**

### Why (m_e, G_N) over (R∞, G_N)

- m_e is measured directly via Penning trap (~3×10⁻¹⁰ relative precision,
  CODATA 2018) — NO α_EM input needed.
- (R∞, G_N) hides α_EM as a third input via the Rydberg formula
  m_e = 2hR∞/(α²c). Calling it a "2-anchor" pair is misleading.
- (m_e, G_N) is a clean 2-anchor pair: both pure dimensional measurements,
  no dimensionless coupling needed for chain closure.
- Same precision in practice as (R∞, G_N) since CODATA m_e value is at
  ~3×10⁻¹⁰, equivalent to Rydberg-derived m_e.

α_EM and R∞ are tracked as PREDICTIONS in the parameter ledger (via
target_parameters.md and run_predictions.py manifest entries), pending
RG-running closure from α_GUT = 1/24 down to atomic scale (research-level).

(Earlier (R∞, G_N) recommendation in this doc revision is superseded —
see below for the full enumeration of options including (R∞, G_N) for
historical reference.)

## Earlier recommendation (superseded 2026-05-01 final): **(R∞, G_N)**

After full enumeration of options, the framework's recommended anchor pair is:

  **R∞ (Rydberg constant) + G_N (Newton's gravitational constant)**

### Why this pair

1. **R∞ is the most precisely measured dimensional constant in all of physics** (~10⁻¹² relative precision; CODATA 2018: 10973731.568160(21) m⁻¹). No other physical measurement is comparably precise.
2. **G_N is the unique direct gravitational measurement** — every other gravitational quantity (M_Pl, t_Pl, ℓ_Pl) is derived from G_N. Anchoring through G_N is therefore at the source of gravitational physics rather than through derived quantities.
3. **Both are universally recognized** as fundamental physical constants (every undergraduate physics text covers both).
4. **Both are model-independent**: R∞ is hydrogen spectroscopy + QED; G_N is direct mechanical measurement (Cavendish-type torsion balance, modern variants). Neither depends on cosmological models (avoiding ΛCDM/Hubble tension).
5. **c and ℏ are exact-by-definition in SI** (post-1983 and post-2019 respectively), so they're not anchors but unit conversion factors. The framework's anchor count remains 2.

### Practical conversion chain via (R∞, G_N)

To convert R∞ to a mass scale: use the Rydberg formula $R_\infty = \alpha^2 m_e c / (2h)$, equivalently $m_e = 2 h R_\infty / (\alpha^2 c)$, which requires α_EM (independently measured at ~10⁻¹⁰ precision via electron g-2). With c and h exact in SI, $(R_\infty, \alpha_{EM})$ jointly give $m_e$ at ~10⁻¹⁰ precision.

Then the framework chain:
- $m_e$ (from Rydberg + α_EM) → $m_\tau$ via theorem-grade ratio (Koide + chiral substrate dynamics).
- $m_\tau$ → $v_{\rm Higgs}$ via Yukawa coupling $y_\tau$ (theorem-grade per Row P11).
- $v_{\rm Higgs}$ → $G_F$ via SM tree-level relation (now PREDICTED rather than anchored).
- $G_N$ → $M_{\rm Pl}$ directly via $M_{\rm Pl} = \sqrt{\hbar c / G_N}$.
- $M_{\rm Pl}$ + $v_{\rm Higgs}$ → $N$ via BZJ inversion.
- $N$ + $M_{\rm Pl}$ → $H_0$, $t_0$ via cascade theorem (PREDICTED, testable against cosmology).

### What this changes from the (historical, RETRACTED) "(G_F, M_P) anchor pair" framing

| aspect | (G_F, M_P) old | (R∞, G_N) new |
|---|---|---|
| Mass anchor precision | 0.51 ppm | ~10⁻¹⁰ (×10⁴ better) |
| Gravity anchor | M_P (50 ppm, derived from G_N) | G_N (22 ppm, direct) |
| α_EM dependence | None (G_F is α-independent) | Required (R∞ → m_e via α²) |
| Cosmology dependence | None (M_P from CODATA) | None (G_N direct) |
| G_F status | Anchor | **Now a framework prediction** (testable!) |
| H_0, t_0 status | Predicted | Predicted (same, with cleaner chain) |
| Fundamental-constant feel | Particle physics + derived gravity | Two pure fundamental constants |

### Note on α_EM

Because R∞ → m_e requires α_EM, the (R∞, G_N) pair effectively uses (R∞, α_EM, G_N) when converting to absolute mass scales. α_EM is measured to ~10⁻¹⁰ precision and isn't currently a framework prediction at high precision (the framework's α_EM connection involves QED + cascade running, not yet theorem-grade). So in practice, the framework relies on three external inputs:
- R∞ (Rydberg, dimensional, ~10⁻¹² precision)
- α_EM (dimensionless, ~10⁻¹⁰ precision)
- G_N (gravitational, ~22 ppm precision)

The first two combine to give m_e (mass anchor); the third gives gravitational scale. Conceptually still 2 dimensional anchors; α_EM is a dimensionless coupling whose framework-internal status is separate (see Row TBD for α_EM).

For comparison, the (historical, RETRACTED) "(G_F, M_P) anchor pair" framing similarly involved 2 numbers, with no α_EM dependence in the immediate chain (though α_EM is implicit in many electroweak observables). [Post-pivot: N_hub + a unit constant; G_F downstream.]

## What G_sub closure changes

Pre-G_sub-closure:
- Row 25 was a **structural commitment** ("substrate ≈ Planck") — couldn't be checked.
- M_P was needed externally because the framework had no derived path to it.
- [Historical framing, RETRACTED:] the "(G_F, M_P) anchor pair" — both numbers were needed. [Post-pivot: N_hub (physical input) + a unit constant; G_F downstream.]

Post-G_sub-closure:
- Row 25 is sharpened to **derived ratio** $M_{\rm substrate}/M_{\rm Pl} = \sqrt{\pi}/8$ (theorem-grade).
- M_P moves from "necessary external input" to "derivable from M_substrate via theorem-grade ratio."
- The framework can be re-anchored through ANY pair of dimensional observations spanning electroweak + gravitational sectors.
- Cross-checks become possible (predict the unit constant — or G_F — from N_hub + Λ_CC + the framework, compare to CODATA / PDG).

The anchor count (2) doesn't reduce, but the choice becomes flexible.

## Going to 1 anchor (research-level)

Reducing to a single dimensional anchor would require a **5th independent theorem-grade dimensionless relation** in the framework. Currently the relations (BZJ, cascade, Drude, path-b) give 4 constraints; with 5 unknowns, 2 inputs are needed.

Candidates for a 5th relation (research-level):
- **Anthropic / observer-existence argument** that pins cosmic age N from initial conditions. Multi-session conceptual work.
- **Cosmological initial-conditions theorem** that determines N from substrate primordial dynamics. Multi-session.
- **Bekenstein-Hawking entropy bound saturation** that ties N to substrate information capacity. Speculative.

These are deep research questions, not session-end label upgrades.

## Going to 0 anchors

Mathematically impossible without external observations: dimensional content (lengths, masses, times in SI) cannot emerge from pure mathematics. Any framework predicting SI-unit values needs at least one calibration measurement.

The framework's irreducible minimum is **one observed dimensional quantity** (paired with the cosmological dimensional content above).

## Summary

| Aspect | Pre G_sub closure | Post G_sub closure |
|---|---|---|
| Anchor count | 2 (G_F, M_P) | 2 (any equivalent pair) |
| Row 25 | Structural commitment | Derived ratio (theorem-grade) |
| M_P status | External input | Derivable from G_F + cosmological |
| Cross-anchor checks | Not possible | Possible (consistency tests) |
| Conceptual cleanness | Mixed (particle + derived gravity) | Cleanest pair: (G_F, G_N) — two fundamental constants |

## Cross-references

- `../theorems/theorem_g_sub_drude_closure_2026-04-30.md` — Drude form theorem-grade Step 1+2 + path (b).
- `../audits/registers/uniqueness_ledger.md` Row 25 — substrate-Planck identification (sharpened to derived ratio).
- `../parameters/parameter_uniqueness_ledger.md` Row P60 (G_N · M_Pl² = 1 derived) + Row P61 (M_substrate/M_Pl = √π/8).
- `../theorems/theorem_dimensionless_ratio_principle_2026-04-30.md` — meta-principle for dimensional observables.
- `predictions/N_hub.py` (THE adopted dimensional input), `predictions/G_F.py` (a downstream prediction; the observable that calibrates N_hub's value), `predictions/G_N.py` (the unit-setting constant). See `simulator.axioms.n_hub_pivot()`.
