# Inverse-Noether Scanner — empirical symmetry-detection probe

**Date:** 2026-04-29.
**Status:** **Phase 4 deliverable** of the symmetry-shortcut program. Working tool with validated behavior on a canonical test case.

**Predecessors:**
- `../theorems/theorem_substrate_symmetry_to_martingale.md` §6 (inverse-Noether parsimony argument).
- `../theorems/theorem_substrate_generation_charge_conservation.md` §6 (cross-reference).

**Backing script:** `proofs/foundations/inverse_noether_scanner.py`.

---

## Question

Phase 0 §6 introduced the inverse-Noether reading as a parsimony heuristic:

> If a substrate observable is found numerically to satisfy a symmetry-conservation signature, the most parsimonious explanation is that there exists a symmetry G of the (prior, filtration) under which the observable is invariant.

The Phase 4 question: **build an operational tool that scans a list of substrate observables and flags which ones witness — or fail to witness — a candidate symmetry of the dynamics.**

This document describes the tool, validates its behavior on a known-symmetric and a deliberately-broken test case, and records its honest limits.

---

## The empirical test

The scanner uses the simplest meaningful empirical signature of distributional symmetry:

> **Test (G-distributional symmetry of stationary).** *For a candidate symmetry G acting on the substrate state space, the stationary distribution is G-invariant iff E[f(state)] = E[f(σ·state)] for every observable f and every σ ∈ G.*

Under a G-symmetric dynamics:
- Observables that are *pointwise* G-invariant (f(s) = f(σ·s) for all s) trivially satisfy the test.
- Observables that are *not* pointwise G-invariant satisfy the test in expectation only because the dynamics is G-symmetric.

If G is broken, the second class (non-G-invariant observables) becomes the **witness set**: the spread of E[f(σ·state)] across σ is non-zero, and quantifies the breaking at that observable.

This test is:
- **Direct.** It empirically probes the dynamics, not a posited prior.
- **Theory-light.** It needs only a simulator and a list of test observables.
- **Diagnostic.** Witnesses pinpoint *which* observable detects the break and the *magnitude* of the break.

The test is *not* a full inverse Noether (Theorem 1's converse) because it probes distributional invariance of the stationary, not invariance of A2-T's plural-retention prior. The two are related but not identical; the scanner produces parsimony evidence only.

---

## Validated test case — site-C₃ on the trivalent vertex

### Setup

- **Substrate:** trivalent vertex toggle chain. State (s_0, s_1, s_2) ∈ {0,1}³. Each edge toggles independently per A1 + Stage 2a (`../theorems/theorem_edge_surprise_thresholds.md`): p_create = 1/2, p_destroy = 1/3.
- **Candidate symmetry G:** site-C₃ acting by cyclic edge permutation σ : (s_0, s_1, s_2) → (s_2, s_0, s_1).
- **Observables (9 total):** `edge0`, `edge2`, `avg`, `edge_variance`, `pair_diff_01`, `pair_diff_02`, `cyclic_weighted`, `total`, `triple_product`. Four are pointwise C₃-invariant (`avg`, `edge_variance`, `total`, `triple_product`); five are not.

### Two runs

1. **Symmetric run** (canonical toggle, all edges p_create=1/2). C₃-symmetric by construction.
2. **Asymmetric run** (edge 2 has p_create=0.20 instead of 0.50). C₃ explicitly broken.

Each run accumulates ≈ 6.4 × 10⁵ post-warmup state samples (4000 realizations × 160 post-warmup steps).

### Result table

**Symmetric run — 0 witnesses of broken C₃** (expected: 0).

| Observable | <f>_id | <f>_σ¹ | <f>_σ² | spread | z-score | flag |
|---|---|---|---|---|---|---|
| edge0 | 0.5998 | 0.5990 | 0.5999 | 0.0009 | 0.85 | + consistent with C₃ |
| edge2 | 0.5990 | 0.5999 | 0.5998 | 0.0009 | 0.85 | + consistent with C₃ |
| pair_diff_01 | −0.0001 | −0.0008 | 0.0009 | 0.0017 | 1.11 | + consistent with C₃ |
| pair_diff_02 | 0.0008 | −0.0009 | 0.0001 | 0.0017 | 1.11 | + consistent with C₃ |
| cyclic_weighted | 3.5965 | 3.5982 | 3.5972 | 0.0017 | 0.42 | + consistent with C₃ |

(Pointwise-invariant observables omitted; trivial.)

**Asymmetric run — 5 witnesses of broken C₃** (z-scores 114–303).

| Observable | <f>_id | <f>_σ¹ | <f>_σ² | spread | z-score | flag |
|---|---|---|---|---|---|---|
| edge0 | 0.5998 | 0.3736 | 0.5999 | 0.2263 | 214.21 | − WITNESSES C₃ BREAK |
| edge2 | 0.3736 | 0.5999 | 0.5998 | 0.2263 | 214.21 | − WITNESSES C₃ BREAK |
| pair_diff_01 | −0.0001 | −0.2261 | 0.2263 | 0.4524 | 302.84 | − WITNESSES C₃ BREAK |
| pair_diff_02 | 0.2261 | −0.2263 | 0.0001 | 0.4524 | 302.84 | − WITNESSES C₃ BREAK |
| cyclic_weighted | 2.9204 | 3.3728 | 3.1464 | 0.4524 | 114.47 | − WITNESSES C₃ BREAK |

The witnesses correctly localise the broken symmetry: `edge0` and `edge2` cleanly diagnose that one specific edge has different rate (its expected value disagrees with the others under cyclic permutation). `pair_diff_*` and `cyclic_weighted` are amplified diagnostics with even higher z-scores.

### Validation status

- Symmetric run: **0 false positives** (expected 0 by Theorem 1 + dynamics being C₃-symmetric).
- Asymmetric run: **5 true positives** (every non-pointwise-invariant observable witnessed the break).
- Z-scores in the asymmetric run are 114–303, far above the 3σ threshold — break detection is robust against statistical noise.

The scanner's behaviour matches the theoretical prediction in both directions.

---

## How to use the scanner on other substrates

The scanner's structure generalises to any:
- finite substrate state space,
- step function evolving the state stochastically,
- candidate symmetry G acting on the state space,
- list of observable functions.

Adaptation steps:

1. Replace `toggle_step` with the substrate-specific step function.
2. Replace `c3_apply` with the action of the candidate G on states.
3. Provide a list of observables relevant to the framework target.
4. (Optional) Provide a deliberately-broken comparison dynamics to validate the scanner detects breaking on the new substrate.

For framework-internal use, candidate symmetries to scan include:
- Discrete cubic O_h on srs primitive cell (point group of the lattice).
- Galois Z₃ on the M ⋊_α Z_3 tower (Phase 1b).
- Site-C₃ at trivalent vertices (validated above).
- Spin(4) × Spin(2) on Cl(6,0) spinor sectors (Phase 1c, requires representation-theoretic action).

---

## Honest scope

1. **Distributional ≠ Bayesian.** The scanner detects G-invariance of the *stationary distribution* of the dynamics, not G-invariance of A2-T's plural-retention prior. The two are related (a G-symmetric prior under G-equivariant dynamics produces a G-symmetric stationary), but not identical. A G-symmetric stationary is consistent with — but does not prove — Theorem 1 (H1)–(H3).

2. **Observable-list dependence.** The witness set is determined by which observables are scanned. Adding more non-invariant observables strengthens the scan; restricting to invariant observables makes the scanner trivially produce zero witnesses (since pointwise-invariant observables can't witness anything).

3. **Detection threshold.** The current implementation uses z = 3 as the consistent-with-symmetry threshold. For high-precision applications, the threshold should be tuned to control the false-positive rate at the desired level given the sample-size and the expected effect size.

4. **Discrete G only (current implementation).** The scanner handles discrete groups by enumerating G's elements and applying them to each state. For continuous G, the test extends to infinitesimal-generator commutator tests but requires more machinery (one-parameter-subgroup integration, Lie-derivative discretisation). Not implemented here.

5. **No new prediction emerges from the scanner.** The scanner is a *diagnostic*, not a *derivation*. Its operational value is detecting hidden or broken symmetries that the user has not written down explicitly. To turn a detected symmetry into a parameter prediction, the forward direction (Theorem 1) must be invoked separately.

6. **Confirmation requires forward verification.** A passed empirical test is parsimony evidence for symmetry; it is not a proof. Confirmation requires verifying (H1) prior G-invariance + (H2) filtration G-equivariance + (H3) functional G-invariance separately — i.e., applying Theorem 1 in the forward direction.

---

## Status

**Phase 4 deliverable closed.** The scanner is working, validated on a canonical test case (site-C₃ on the trivalent vertex), and ready for use on other substrate models. Two regimes confirmed:

- **Sanity check:** known-symmetric dynamics produces zero false-positive witnesses.
- **Discovery mode:** deliberately-broken dynamics produces high-z-score witnesses that pinpoint the break.

**Effect on framework:**
- The framework now has an operational tool for the inverse-Noether direction unique to its observer-layer formulation. Conventional QFT has no analog.
- The tool is bounded — does not graduate parameters by itself — but is the only Phase-2+ tool the symmetry-shortcut program produced that is genuinely novel relative to the existing framework apparatus.

**Effect on the program:** completes the structural inventory of the symmetry-shortcut program. Combined with Phases 0–1, the program ships as: clean engine theorem (Phase 0) + three corollary docs (Phase 1) + diagnostic tool (Phase 4). Phase 2 attacks are deferred or earmarked.

---

## Citations

**Type 3 (cited published) references:**

- **Williams, D.** (1991). *Probability with Martingales.* Cambridge University Press. §10.7 (inherited via Theorem 1).
- **Cover, T. & Thomas, J.** (2006). *Elements of Information Theory*, 2nd ed. Wiley. §2.8 (data processing; relevant to spread metric construction).
- **Wasserman, L.** (2004). *All of Statistics.* Springer. §10 (z-test threshold construction; standard reference for the detection threshold in §3).

All citations to peer-reviewed published work or standard textbooks.

---

## Cross-references

- `../theorems/theorem_substrate_symmetry_to_martingale.md` §6 — predecessor heuristic statement.
- `../theorems/theorem_substrate_generation_charge_conservation.md` §6 — cross-reference.
- `proofs/foundations/inverse_noether_scanner.py` — backing script.

---

## Next forward-construction steps

The scanner is ready for use. Concrete future applications:

1. **Apply to Hashimoto NB-walk dynamics on srs.** Scan for unwritten symmetries beyond the known site-C₃ + lattice translation. May surface emergent IR symmetries the framework hasn't yet identified.

2. **Apply to deliberately-broken framework variants.** As a validation tool when the framework adopts new structural assumptions: scan to confirm the new assumption preserves the expected symmetries.

3. **Continuous-G extension.** Implement infinitesimal-generator-based detection for Lorentz, gauge, and Pati-Salam symmetries. ~1–2 sessions.

4. **Pair with Phase 2b execution.** When Phase 2b execution proper happens, the scanner can validate that route 2 (Bloch sum rules) preserves the expected Lorentz/rotation symmetries empirically — providing an independent cross-check on the closure.
