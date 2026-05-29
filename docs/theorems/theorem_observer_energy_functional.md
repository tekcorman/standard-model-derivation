# Observer energy functional — theorem

**Date:** 2026-04-20 (Session 9 continuation).
**Status:** THEOREM — gate-passing under parameter_linter.md. All load-bearing steps are Type 1 (axiom), Type 2 (explicit algebra), Type 3 (precisely-cited published theorem: Landauer 1961, Bennett 1973, both already load-bearing in framework's A-IT3), or Type 4 (upstream closed theorem: `theorem_edge_surprise_thresholds.md`). Per user permission (session 9), Type 4 upstream from theorem-grade docs/ files is acceptable when those files themselves pass the gate.
**Scope:** narrow. Defines an observer energy functional E_obs(t) as the Landauer-scaled accumulated surprise, and proves it is non-negative, monotonically non-decreasing, and extensive. Derives arrow of time as a corollary.
**Out of scope:** claiming E_obs equals physical dissipation exactly; calibrating k_B T to a specific value; invoking Sagawa-Ueda (A-IT7); connecting E_obs to cosmological observables (Λ_CC, Ω_DM).
**Upstream scoping:** an internal working note and Stage 2c sketched therein.
**Replaces Stage 2b (arrow of time separate theorem):** not needed — the arrow falls out as a corollary of this theorem's monotonicity.

**Post-2026-05-08 axiom slate note.** A1 and A2-T (cited throughout this theorem) are now derived theorems of (A) self-containment + (B) finite observer + standard math + (I) active reading, per `theorem_toggle_from_self_containment.md` and `theorem_A2_mdl_from_finite_register.md`. References to "A1 + A2-T" remain semantically valid; the energy functional construction and arrow-of-time corollary are unchanged. See `framework_axioms.md` §10 for the updated top-level summary.

---

## 1. Theorem statement

**Theorem (Observer Energy Functional).** Let an observer under A1 + A2-T (waterline thm; refined A2) accumulate observations (o_1, o_2, …) from an iid toggle stream on the substrate, maintaining a per-pair Bernoulli-Beta posterior model per `theorem_edge_surprise_thresholds.md` (hereafter: Stage 2a). Define the observer's accumulated surprise after t observations:

$$S_{\text{total}}(t) := \sum_{i=1}^{t} S(o_i \mid \text{model}_{i-1})$$

where $S(x; P) = -\log_2 P(x)$ is Shannon self-information and model_{i-1} is the observer's state before observation i.

Let $\kappa := k_B T \ln 2$ be the Landauer conversion constant (Landauer 1961, A-IT3 in `../framework/framework_axioms.md` §9), with T a reference temperature characterizing the observer's physical realization. Define:

$$E_{\text{obs}}(t) := \kappa \cdot S_{\text{total}}(t).$$

Then:

**(Non-negativity)** $E_{\text{obs}}(t) \geq 0$ for all t.

**(Monotonicity)** $E_{\text{obs}}(t+1) \geq E_{\text{obs}}(t)$ for all t.

**(Extensivity)** $E_{\text{obs}}(t+s) = E_{\text{obs}}(t) + \kappa \cdot \Delta S(\text{observations } t+1..t+s \mid \text{model}_t)$ for all t, s ≥ 0.

**(Arrow of time corollary)** Under iid observations with non-trivial branch measure $\mu$ (session-9 theorem), $E_{\text{obs}}(t)$ is strictly increasing in t almost surely. For t₁ < t₂, the observer's state at t₂ is distinguishable from its state at t₁ by $E_{\text{obs}}$ alone — an internal arrow of time that requires no external reference.

---

## 2. Axioms and cited upstream

**Framework axioms:**
- **A1** (`../framework/framework_axioms.md` §2) — toggle alphabet.
- **A2-T** (derived theorem; `theorem_A2_mdl_from_finite_register.md`) — MDL observer with selective retention.

**Type 3 citations (all already load-bearing in the framework):**
- **Landauer, R.** (1961). *Irreversibility and heat generation in the computing process.* IBM J. Res. Dev. 5, 183-191. §2 establishes that any logically irreversible information-processing operation corresponds to at least $k_B T \ln 2$ of free energy dissipation per erased bit.
- **Bennett, C. H.** (1973). *Logical reversibility of computation.* IBM J. Res. Dev. 17, 525-532. Refines Landauer's argument and establishes the $k_B T \ln 2$ per-bit factor in its canonical form.
- Both cited as A-IT3 in `../framework/information_theoretic_stability_axioms.md` and `../framework/framework_axioms.md` §9.

**Type 4 upstream (per user permission, session 9):**
- `theorem_edge_surprise_thresholds.md` (Stage 2a) — theorem-grade. Provides: the observer's per-pair Bernoulli-Beta model, the S_fresh = 1 bit and S_disconfirm = log₂(3) bit surprise values, and Shannon 1948 as the surprise definition's ultimate upstream.

No fabricated citations. No post-hoc fitting. No acceptance/rejection rule imposed.

---

## 3. Setup

By Stage 2a (§3), the observer maintains a Bayesian probability model on the existence state $q_{XY} \in [0,1]$ of each pair (X, Y), with uniform Beta(1, 1) prior and conjugate updating under observations.

At each observation step, the observer observes a toggle outcome at some pair (X, Y) and updates the posterior. The Shannon surprise $S(o_i \mid \text{model}_{i-1})$ of each observation is a non-negative real number in bits, as defined in Stage 2a §8.

Over t observations, the observer accumulates:

$$S_{\text{total}}(t) := \sum_{i=1}^{t} S(o_i \mid \text{model}_{i-1}).$$

This is a well-defined real-valued function of t and the observation sequence. The theorem establishes its properties and the associated energy functional.

---

## 4. Step E1 — non-negativity of per-observation surprise (Type 2)

**Claim.** For any observation o_i and any model (probability distribution) P with $P(o_i) \in (0, 1]$:

$$S(o_i; P) = -\log_2 P(o_i) \geq 0.$$

**Proof.** Since $P(o_i) \leq 1$ (probability), $\log_2 P(o_i) \leq 0$, hence $-\log_2 P(o_i) \geq 0$. Equality iff $P(o_i) = 1$ (no uncertainty; certain prediction). $\square$

---

## 5. Step E2 — non-negativity of S_total (Type 2)

**Claim.** $S_{\text{total}}(t) \geq 0$ for all t ≥ 0.

**Proof.** By definition, $S_{\text{total}}(t) = \sum_{i=1}^{t} S(o_i \mid \text{model}_{i-1})$. By step E1 (§4), each summand is ≥ 0. Sum of non-negatives is non-negative: $S_{\text{total}}(t) \geq 0$. Convention: $S_{\text{total}}(0) = 0$ (empty sum). $\square$

---

## 6. Step E3 — monotonicity of S_total (Type 2)

**Claim.** $S_{\text{total}}(t+1) \geq S_{\text{total}}(t)$ for all t ≥ 0.

**Proof.** By definition:

$$S_{\text{total}}(t+1) = S_{\text{total}}(t) + S(o_{t+1} \mid \text{model}_t).$$

By step E1, $S(o_{t+1} \mid \text{model}_t) \geq 0$. Hence $S_{\text{total}}(t+1) \geq S_{\text{total}}(t)$. $\square$

**Remark.** Strict inequality holds whenever the predicted probability $P(o_{t+1} \mid \text{model}_t) < 1$, which is the case under Stage 2a whenever the posterior is not degenerate. For Beta(α, β) with α, β both finite, no outcome has predictive probability 1, so strict positivity holds.

---

## 7. Step E4 — extensivity of S_total (Type 2)

**Claim.** For all t, s ≥ 0:

$$S_{\text{total}}(t+s) = S_{\text{total}}(t) + \sum_{i=t+1}^{t+s} S(o_i \mid \text{model}_{i-1}).$$

**Proof.** By definition:

$$S_{\text{total}}(t+s) = \sum_{i=1}^{t+s} S(o_i \mid \text{model}_{i-1}) = \sum_{i=1}^{t} S(o_i \mid \text{model}_{i-1}) + \sum_{i=t+1}^{t+s} S(o_i \mid \text{model}_{i-1}) = S_{\text{total}}(t) + \Delta S. \square$$

---

## 8. Step E5 — Stage 2a-specific surprise values (Type 4)

The following values are established in Stage 2a and serve as numerical anchors:

- **Fresh pair** (Beta(1,1) prior): surprise of either "exists" or "absent" = 1 bit. (Stage 2a §9.)
- **Once-confirmed pair, confirming observation** (Beta(2,1) posterior, observe "exists"): surprise = $-\log_2(2/3) = \log_2(3/2) \approx 0.585$ bits. (Stage 2a §6-7 arithmetic.)
- **Once-confirmed pair, disconfirming observation** (Beta(2,1) posterior, observe "absent"): surprise = $-\log_2(1/3) = \log_2 3 \approx 1.585$ bits. (Stage 2a §10.)

These Type 4 values show that S_total's per-observation increments lie in a range with non-trivial structure: confirming observations are cheaper (0.585 bits) than fresh ones (1 bit) which are cheaper than disconfirming ones (log₂ 3 ≈ 1.585 bits).

---

## 9. Step E6 — Landauer conversion constant (Type 3)

**Claim.** By Landauer 1961 + Bennett 1973 (A-IT3), the minimum free-energy cost of erasing one bit of information in an environment at temperature T is $k_B T \ln 2$.

**Citation.** Landauer 1961 §2 derives this as a consequence of the Second Law applied to logically irreversible operations: erasing one bit reduces information entropy by 1 bit $= k_B \ln 2$ of thermodynamic entropy (converting between base-2 information and natural-log thermodynamic units); the Second Law requires this entropy decrease to be compensated by entropy increase elsewhere, minimally $k_B \ln 2$ released as heat at temperature T, giving $k_B T \ln 2$ of free energy dissipation. Bennett 1973 gives the canonical modern formulation.

Define the Landauer conversion constant:

$$\kappa := k_B T \ln 2.$$

This is a positive real number with units of energy per bit. T is a reference temperature characterizing the observer's physical realization; this theorem does not calibrate T to a specific value — κ serves as an information-to-energy conversion constant.

---

## 10. Step E7 — definition of E_obs (Type 2)

**Definition.** The observer energy functional is:

$$E_{\text{obs}}(t) := \kappa \cdot S_{\text{total}}(t).$$

---

## 11. Step E8 — properties of E_obs (Type 2)

**Claim.** E_obs satisfies non-negativity, monotonicity, and extensivity.

**Proof.** Since κ > 0:

- **Non-negativity.** By E2 (§5), $S_{\text{total}}(t) \geq 0$, hence $E_{\text{obs}}(t) = \kappa \cdot S_{\text{total}}(t) \geq 0$. $\square$

- **Monotonicity.** By E3 (§6), $S_{\text{total}}(t+1) \geq S_{\text{total}}(t)$, hence $E_{\text{obs}}(t+1) = \kappa \cdot S_{\text{total}}(t+1) \geq \kappa \cdot S_{\text{total}}(t) = E_{\text{obs}}(t)$. $\square$

- **Extensivity.** By E4 (§7):
$$E_{\text{obs}}(t+s) = \kappa \cdot S_{\text{total}}(t+s) = \kappa \cdot S_{\text{total}}(t) + \kappa \cdot \Delta S = E_{\text{obs}}(t) + \kappa \cdot \Delta S. \square$$

---

## 12. Step E9 — arrow of time corollary (Type 2)

**Claim.** Under iid observations from the session-9 branch measure μ with non-trivial per-pair posteriors, $E_{\text{obs}}(t)$ is strictly increasing in t almost surely. Distinct times t₁ < t₂ give distinct $E_{\text{obs}}$ values, so t is distinguishable from its own internal state.

**Proof.** By E3 remark (§6), strict inequality $S_{\text{total}}(t+1) > S_{\text{total}}(t)$ holds whenever $P(o_{t+1} \mid \text{model}_t) < 1$, i.e., whenever the observer does not predict the next observation with certainty.

For iid observations at pairs drawn uniformly from all pairs (branch measure μ's structure, session-9 theorem), the observation's predictive probability is the Beta posterior mean $\alpha/(\alpha+\beta) \in (0, 1)$ for any non-degenerate Beta state. Since α, β remain finite at every t, the predictive probability is strictly in (0, 1), hence the surprise is strictly positive.

Therefore $S_{\text{total}}$ is strictly increasing almost surely, and so is $E_{\text{obs}}$.

Hence for t₁ < t₂: $E_{\text{obs}}(t_1) < E_{\text{obs}}(t_2)$, and the observer's internal energy distinguishes past from future without external reference. This is the arrow of time as an observer-internal quantity. $\square$

---

## 13. Parameter_linter gate summary

| Step | Claim | Gate type | Source |
|---|---|---|---|
| E1 | S ≥ 0 | Type 2 | -log₂ P ≥ 0 for P ∈ (0, 1] (§4) |
| E2 | S_total ≥ 0 | Type 2 | Sum of non-negatives (§5) |
| E3 | S_total monotonic | Type 2 | Addition of non-negative (§6) |
| E4 | S_total extensive | Type 2 | Sum decomposition (§7) |
| E5 | Stage 2a values | Type 4 | `theorem_edge_surprise_thresholds.md` §§9-10 (per session-9 user permission for Type 4 from docs/) |
| E6 | Landauer conversion | Type 3 | **Landauer 1961** §2, **Bennett 1973**, both A-IT3 (§9) |
| E7 | E_obs definition | Type 2 | Definition using E6's constant (§10) |
| E8 | E_obs properties | Type 2 | Inherit from E2, E3, E4 (§11) |
| E9 | Arrow of time | Type 2 | Strict positivity of increments under non-degenerate predictive (§12) |

**All steps gate-passing.** Two Type 3 citations (Landauer 1961, Bennett 1973), both already load-bearing in the framework's A-IT3. One Type 4 upstream (Stage 2a), permitted per session-9 user ruling.

---

## 14. What this theorem closes

- **Energy functional defined.** E_obs(t) = κ · S_total(t) is a well-defined function of observation count with units of energy.
- **Monotonicity.** E_obs is non-decreasing in observation count; strictly increasing under non-trivial observations.
- **Arrow of time.** The observer's state at time t₂ is distinguishable from its state at time t₁ < t₂ by E_obs alone. Arrow of time is observer-internal and requires no external reference.
- **A-IT3 (Landauer) load-bearing for the first time.** Invoked as Type 3 citation for the κ conversion constant.
- **Stage 2b (arrow of time as separate theorem) obviated.** Arrow falls out as Corollary §12.

---

## 15. What this theorem does NOT close

- **Does NOT claim E_obs equals physical dissipation.** Landauer gives a LOWER BOUND on dissipation for irreversible computation; E_obs is κ-scaled surprise, a different quantity. The two coincide only in idealized limits.
- **Does NOT calibrate T.** The reference temperature in κ = $k_B T \ln 2$ is observer-dependent and not fixed by this theorem. Connecting the observer to a specific T is a physical-realization question.
- **Does NOT invoke A-IT7 (Sagawa-Ueda).** That strengthens observer-measurement bounds and is out of scope. Stage 2d material.
- **Does NOT connect to cosmology (Λ_CC, Ω_DM).** Those connections require additional structural content.
- **Does NOT establish that E_obs grows linearly.** The rate depends on observer-state distribution across pairs; at steady state under μ it grows at a specific rate, but computing that rate is additional work.
- **Does NOT derive a Hamiltonian or equations of motion.** E_obs is an accumulated functional, not a generator.

---

## 16. Honesty

**Relationship of observer surprise to physical dissipation.** Landauer's principle gives a LOWER BOUND on free-energy dissipation per bit of irreversible information-processing. The observer's Bayesian update Beta(α, β) → Beta(α+1, β) is irreversible in that multiple (prior, observation) pairs yield the same posterior, producing information loss. Per-update information loss is at least 1 bit (two possible predecessors in most cases), so total Landauer dissipation is at least $t \cdot k_B T \ln 2$ after t observations.

The observer's accumulated surprise $S_{\text{total}}(t)$ is a DIFFERENT quantity: it measures how much the observer's model was "wrong" cumulatively, not how much information was lost in the updates. Under iid observations on fresh pairs, expected surprise per observation = 1 bit, matching the Landauer lower bound. Under observations on confirmed pairs, surprise can range from log₂(3/2) ≈ 0.585 bits (confirming) to log₂(3) ≈ 1.585 bits (disconfirming).

**So E_obs(t) = κ · S_total(t) is neither a strict lower nor upper bound on physical dissipation.** It is a Landauer-scaled information-theoretic quantity that:
- Has units of energy via κ;
- Is non-negative, monotonic, extensive (properties of energy functionals);
- Dominates the Landauer lower bound in expectation under iid observations on fresh pairs (where E[surprise per observation] = 1 bit).

This is honest: E_obs is AN energy functional, not THE unique physical energy. Its physical interpretation requires identifying the observer's physical realization, which is outside this theorem's scope.

**The arrow of time corollary (§12) doesn't depend on physical interpretation.** It requires only that S_total is strictly increasing, which follows from non-degenerate predictive probabilities under iid observations. Under the session-9 branch measure μ, the observer's Beta posteriors never become degenerate in finite t, so strict increase holds almost surely.

**What I did NOT use from the sister project.** I did not cite an external sister-project draft on toggle dynamics as upstream. That draft's Landauer-related content is not in the framework's cite list. Landauer 1961 and Bennett 1973 are cited directly.

**What I did NOT do.** I did not invoke:
- Acceptance/rejection rules for observations (not in refined A2 directly).
- Self-consistency for threshold θ* (toggle paper concept, not yet formalized).
- Sagawa-Ueda (A-IT7) for entropy-reduction bounds.
- Any parametric-complexity formula (the fabricated-citation failure from my prior attempt).
- Post-hoc matching to any numerical prediction.

---

## 17. Downstream consequences

- **Arrow of time (Stage 2b)** — closed as Corollary §12. No separate theorem needed.
- **Stage 2d candidate:** linking E_obs rate to Hubble expansion or Λ_CC. Requires substrate-size content (N) and cosmological connection. Not in this theorem.
- **Stage 3 (Lorentzian signature) unblocked.** Prerequisite (time + energy) now available at theorem grade.
- **A-IT3 load-bearing.** Load-chain-updated in framework.

---

## 18. References

### Framework
- `../framework/framework_axioms.md` §2 (A1), §9 (A-IT3 Landauer / Bennett); `theorem_A2_mdl_from_finite_register.md` (A2-T derived theorem).
- `theorem_edge_surprise_thresholds.md` (Stage 2a; Type 4 upstream here).
- `../framework/information_theoretic_stability_axioms.md` §III (A-IT3 canonical statement).

### Cited published theorems (Type 3)
- **Landauer, R.** (1961). *Irreversibility and heat generation in the computing process.* IBM J. Res. Dev. 5, 183-191.
- **Bennett, C. H.** (1973). *Logical reversibility of computation.* IBM J. Res. Dev. 17, 525-532.
- **Shannon, C. E.** (1948). *A Mathematical Theory of Communication.* Bell Syst. Tech. J. 27, 379-423. (Via Stage 2a.)

### Framework memory

### NOT cited (available but not used)
- External sister-project draft on toggle dynamics — not treated as upstream per discipline "derive from ground up."

---

## 19. Status

**THEOREM (rigor: closed under parameter_linter.md hard gate).** Every load-bearing step annotated and derived. No fabricated citations; no post-hoc fitting; no scope creep. Stage 2c of axiom-elimination roadmap complete; Stage 2b subsumed as Corollary §12.

**Next targets:**
- Stage 3 (Lorentzian signature via frame-invariance of equilibrium toggle distribution). Prerequisites: time and energy now available.
- Stage 2d (optional): link E_obs to cosmological scales. Requires substrate-size N.

**Recommended next session work:** Checkpoint A+B for Stage 3 (Lorentzian signature) before any theorem composition, continuing gate-first methodology.
