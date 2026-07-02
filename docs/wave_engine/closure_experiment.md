# §C Closure Experiment

A 5-second test that quantifies what § smooth-manifold closure would unlock mechanically: re-run the wave with `CCLOSE` (and/or `LORENTZ_SIG`) added to the initial tag set.

**2026-04-27 update:** the catalog's `CCLOSE` tag was split into two distinct frontiers per the bounded-D² finding (`memory/project_lorentzian_signature_route_c_blocked_2026-04-26.md`):

- **`CCLOSE`** — smooth-manifold continuum-limit closure (Riemannian smoothness; no signature commitment).
- **`LORENTZ_SIG`** — Lorentzian signature (-,+,+,+) derivation. **BLOCKED at substrate level**: D²_sub bounded → heat kernel smooth at t=0 → no Λ² coefficient → no Einstein-Hilbert via standard Connes-Chamseddine. Closure requires Krein-space NCG, BLMS causal-set, or Dirac-point routes (research-level).

## Method

Two runs of `proofs/wave_engine/simulator.py`:

1. **Baseline** — initial tags `{A1, E_FIN, A2W, P1, A5M, E6, K3, ORDER}` (CCLOSE not present).
2. **Closure** — initial tags above ∪ `{CCLOSE}` (closure hypothetically achieved).

Diff the firing sets and the bit budget.

Implementation: `proofs/wave_engine/closure_experiment.py`.

## Result — 4-way scenario (2026-04-27, post T1.1 template-dedupe)

| scenario | ops fired | Φ_total | L_total | Net |
|---|---|---|---|---|
| baseline (open) | 173/195 | 94.15 | 522 | −427.85 |
| + CCLOSE only | 188/195 | 94.15 | 573 | −478.85 |
| + LORENTZ_SIG only | 174/195 | 94.15 | 525 | −430.85 |
| + both closed | 195/195 | 94.15 | 593 | −498.85 |

**Striking T1.1 finding: closure adds ZERO substrate-counting Φ.** Pre-T1.1 the experiment showed Δ Φ = +6 (op 6.8 de Rham cohomology contributing HOMOL_E2 template at +6 bits). Under T1.1, op A.1 (group cohomology of F_inv(E), extras = empty) fires from baseline at +6 HOMOL_E2 bits — by the time CCLOSE unlocks 6.8, the HOMOL_E2 template is already used and 6.8 contributes 0. The substrate's homological compression is paid by A.1; smooth de Rham doesn't add to it at the catalog level.

**This sharpens the closure-experiment lesson: §C closure unlocks structural carriers (Riemann, Christoffel, Einstein, Friedmann, ...) that have ZERO substrate-counting Φ. ALL their compression value is at the prediction layer (FLRW vs CMB, etc.), not at the catalog layer.**

**Two frontiers, no overlap:**

- **CCLOSE alone unlocks 15 ops** — the Riemannian smooth-manifold side.
- **LORENTZ_SIG alone unlocks just 1 op** (6.10 Lorentzian metric).
- **6 cosmology ops require BOTH** — FLRW, Einstein, Friedmann, Hubble, scale factor, stress-energy.
- **0 ops are unlocked by "either" frontier alone** — every Layer-6+Appendix gap is structurally one frontier or the other or both.

The empty overlap shows the split is mathematically real: Riemannian smoothness and signature derivation are independent structural problems.

## What unlocks per frontier

**CCLOSE alone (15 ops — smooth-manifold side, no signature):**
```
6.1   smooth manifold M               6.11  Levi-Civita connection
6.2   tangent space T_p M             6.12  Christoffel symbols
6.3   tangent / cotangent bundle      6.13  Riemann curvature R^a_{bcd}
6.4   tensor fields T^(p,q)           6.14  Ricci R_ab, scalar R
6.5   differential forms              6.17  Killing vector fields
6.7   Lie derivative                  A.19  quantum gravity operations
6.8   de Rham cohomology  ★ +6.00     A.21  CFT operators (OPE, Virasoro)
6.9   Riemannian metric
```

**LORENTZ_SIG alone (1 op — signature only):**
```
6.10  Lorentzian metric (-,+,+,+)
```

**Require BOTH frontiers (6 ops — cosmology cluster):**
```
6.18  FLRW metric              6.21  Hubble parameter H(t)
6.19  Einstein equations       6.22  cosmological scale factor a(t)
6.20  Friedmann equations      6.23  stress-energy tensor T_ab
```

Note that **6.6 exterior d, 6.15 geodesics, 6.16 parallel transport, and 6.24 causal structure already fire pre-closure** — these have substrate-grounded discrete analogs (chain-complex d on srs primitive cell, NB-walk geodesics, Bloch-bundle parallel transport, multiway causal partition) that don't need smooth-manifold structure.

## The striking result: closure makes the substrate-counting bit budget WORSE, not better

Of the 22 unlocked ops, **only one (6.8 de Rham cohomology) contributes substrate-counting Φ**. The other 21 are STRUCT carriers — Riemann tensor, Christoffel symbols, Levi-Civita connection, FLRW metric, Einstein equations, Friedmann equations, Hubble parameter, etc. — which cost L without compressing on the substrate-counting metric.

Net Δ from closure: **+6 Φ, +71 L, −65 Net.**

This isn't a bug. It's the mechanism revealing a real distinction:

- **Substrate-counting Φ** (what the catalog tracks): bits of identical-state collapse on F_inv(E) configurations. The catalog-level metric.
- **Observable-counting Φ** (not directly tracked at catalog level): bits of compression against observed data when a prediction lands. The prediction-level metric.

The unlocked GR/cosmology ops contribute their compression at the **prediction layer**: H_0 vs. CMB, Friedmann vs. observed expansion, FLRW vs. supernova distances, etc. None of that shows up in substrate-counting Φ. **Closure is a permission gate that lets prediction-layer compression happen** — at substrate-counting cost, with observable-counting payoff.

## Three readings worth pinning

1. **§C closure is necessary, not compressing-on-its-own.** The wave's reach extends to GR/cosmology only after closure. Without it, predictions H_0/t_0/Ω_Λ/etc. can't ground at the framework's level of rigor.
2. **The framework's compression payoff is asymmetric** — substrate-side spec cost paid up front; observable-side compression collected at prediction time. This matches the strict-table finding (Net = −47 baseline; framework operates at substrate-counting deficit by design).
3. **The simulator's current Φ definition misses the prediction-layer compression entirely.** Extending the metric to include observable-counting Φ is the missing half. Without it, §C closure looks like a 65-bit spec-cost expense, which under-reads its actual structural value. The audit pilot (`audit_pilot.md`) is the framework that pulls in observable-side Φ.

## Implication for the §C closure route

The Connes spectral-action route to §C closure (handoff: an internal note) bypasses the smooth-manifold limit entirely. Once it lands:

- Replace `CCLOSE` tag with `NC_GEOM` (established by spectral-action computation).
- Re-tag the 22 currently-blocked ops to need `NC_GEOM` instead of `CCLOSE`.
- Re-run wave simulator: those ops fire as NC analogs (NC tangent bundle, NC metric, NC Riemann, NC Einstein equations).
- Re-run audit pilot: cosmology predictions (H_0, t_0, Λ_CC) now have computable B's; framework's bit budget includes their observable-side Φ.

The closure-experiment numbers shift accordingly: 22 newly-firing ops contribute ~6 substrate-counting bits + whatever observable-side Φ their downstream predictions add.

## Running the experiment

```bash
python3 proofs/wave_engine/closure_experiment.py
```

Output prints baseline vs closure numbers + per-op Δ for the 22 unlocked ops.
