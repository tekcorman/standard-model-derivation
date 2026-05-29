# Edge-surprise thresholds — theorem

**Date:** 2026-04-20 (Session 9 continuation).
**Status:** THEOREM — gate-passing under parameter_linter.md. All load-bearing steps are Type 1 (axiom), Type 2 (explicit algebra), or Type 3 (precisely-cited published theorem, canonically Jaynes 1957 and Shannon 1948 — both already load-bearing in the framework). No fabricated citations; no post-hoc fitting; no steady-state claims; no observer-decision-rule claims (acceptance/rejection is out of scope).
**Scope:** narrow. Computes two specific surprise values: (i) surprise of any first observation of an edge-pair's state is 1 bit; (ii) surprise of observing an edge absent after one prior confirmation is log₂(3) bits. The asymmetry log₂(3/2) between them is a direct corollary.
**Out of scope for this theorem:** arrow of time, energy functional, threshold-based acceptance rules, steady-state distributions. These are Stage 2b corollaries contingent on this theorem.
**Upstream scoping:** an internal working note; superseded by this narrower target.
**Prior attempt:** an internal working note — failed gate, preserved as methodological record.

**Post-2026-05-08 axiom slate note.** A1 and A2-T (cited as Framework axioms below) are now derived theorems of (A) self-containment + (B) finite observer + standard math + (I) active reading, per `theorem_toggle_from_self_containment.md` and `theorem_A2_mdl_from_finite_register.md`. References to "A1 + A2-T" remain semantically valid; the surprise-threshold derivation is unchanged. See `framework_axioms.md` §10 for the updated top-level summary.

---

## 1. Theorem statement

**Theorem (Edge-Surprise Thresholds).** Let an observer under A1 + A2-T (waterline thm; refined A2) maintain a Bayesian probability model of the existence-state of each pair (X, Y) in the substrate, with uniform prior derived from Jaynes 1957 MaxEnt and surprise function from Shannon 1948. Assume further that toggle events are iid at the pair level (a hypothesis discussed in §16; holds in particular under the session-9 branch measure μ of `theorem_multiway_branch_measure.md`, used to justify per-pair Bernoulli as the MDL-minimal model class in §4). Then:

- **(S_fresh)** The surprise of any first observation of a pair's state (edge present or absent) is exactly **1 bit**.
- **(S_disconfirm)** The surprise of observing an edge absent on a pair after exactly one prior observation of it as present (or equivalently, the mirror case) is exactly **log₂(3) ≈ 1.585 bits**.
- **(Asymmetry)** S_disconfirm − S_fresh = log₂(3/2) ≈ 0.585 bits > 0.

---

## 2. Axioms and cited upstream

**Framework axioms:**

- **A1** (`../framework/framework_axioms.md` §2) — binary self-inverse toggle alphabet E; toggles act on specific pairs (X, Y).
- **A2-T** (derived theorem; `theorem_A2_mdl_from_finite_register.md`) — observer selects model minimizing total description length L_total = L(model) + L(data | model); selective retention at rate-distortion optimum. (Demoted from axiom A2 to derived theorem 2026-04-26.)

**Cited published theorems (Type 3 gate):**

- **Jaynes 1957**, *Information Theory and Statistical Mechanics*, Phys. Rev. 106, 620-630. §II establishes the MaxEnt principle: among probability distributions consistent with stated constraints, the entropy-maximizing one is selected. Applied here to the "no prior information" case on a continuous parameter space.
- **Shannon 1948**, *A Mathematical Theory of Communication*, Bell Syst. Tech. J. 27, 379-423. §I defines self-information (surprise) of an event x under probability distribution P as S(x) = −log₂ P(x).

No Type 4 (upstream predictions/ file) citations required.

No fabricated citations. Steps not covered by the above are Type 2 (explicit algebra) and derived inline.

---

## 3. Setup

Under A1, each toggle event operates on a specific pair (X, Y) in the substrate. A pair's observable state is binary: an edge between X and Y either exists or doesn't. Let $q_{XY} \in [0, 1]$ be the observer's probabilistic model of this state:

$$q_{XY} := P(\text{edge exists between } X \text{ and } Y).$$

Under A2 refined, the observer minimizes L_total = L(model) + L(data | model). Computing L(data | model) requires a probability distribution; under standard information theory (Shannon 1948), the minimal expected code length achieving the data cost is $-\sum_i \log_2 P(\text{observation}_i)$, so the observer's model IS a probability distribution (or a posterior over candidate distributions).

The observer's model of pair (X, Y) is a probability distribution on $q_{XY} \in [0,1]$ — equivalently, a posterior π(q_{XY}).

---

## 4. Step S3 — per-pair Bernoulli is the MDL-minimal model class (Type 2)

**Claim.** For iid toggle events at pair level, the MDL-minimal observer model class is per-pair independent (not joint).

**Proof.** Under A1, each toggle event operates on a specific pair. Under A2 refined, the observer considers candidate model classes and selects by L_total minimization.

Consider two candidate model classes:

- (a) **Per-pair Bernoulli:** one independent Beta posterior π_{XY}(q_{XY}) per pair. Total model size: 2 parameters per pair × N_pairs = 2 N_pairs parameters.
- (b) **Joint:** a single distribution over $\{0,1\}^{N_{\text{pairs}}}$. Total model size: $2^{N_{\text{pairs}}} - 1$ parameters.

Assuming A1 gives iid observations at the pair level (toggle events occur independently across pairs), the true joint distribution factorizes:

$$P(\{e_{XY}\}) = \prod_{XY} P(e_{XY}).$$

Under this factorization, L(data | joint-model) = L(data | factorized-model). The data-fit cost is the same.

However, L(joint-model) > L(factorized-model) because the joint model requires $2^{N_{\text{pairs}}} - 1$ parameters vs $2 N_{\text{pairs}}$ for factorized.

Therefore under MDL (refined A2):

$$L_{\text{total}}(\text{factorized}) = 2 N_{\text{pairs}} \log_2(\text{precision}) + L(\text{data}|\text{factorized}) < L_{\text{total}}(\text{joint})$$

and the observer selects per-pair Bernoulli. $\square$

**Remark.** This is a Type 2 (explicit algebra) step. No citation required. The argument uses: A1 (iid toggle events at pair level), A2 refined (MDL minimization), and the standard information-theoretic fact that independent events factorize joint probabilities (Shannon 1948 §I chain rule, though this is also directly definitional).

---

## 5. Step S4 — uniform prior from Jaynes MaxEnt (Type 3 + Type 2)

**Claim.** The observer's MDL-optimal prior on each $q_{XY} \in [0,1]$, given no prior information, is uniform: π(q) = 1 for q ∈ [0,1].

**Proof.** Apply Jaynes 1957 MaxEnt principle: among probability distributions π(q) on [0,1] satisfying the normalization constraint $\int_0^1 \pi(q) dq = 1$ and no other constraints, the observer selects the entropy-maximizing distribution:

$$\pi^* = \arg\max_{\pi} \left[ -\int_0^1 \pi(q) \log \pi(q) dq \right] \text{ subject to } \int_0^1 \pi(q) dq = 1.$$

**Explicit derivation (Type 2):** use Lagrange multipliers. Form the functional:

$$\mathcal{L}[\pi, \lambda] = -\int_0^1 \pi(q) \log \pi(q) dq - \lambda \left( \int_0^1 \pi(q) dq - 1 \right).$$

Variational derivative with respect to π(q):

$$\frac{\delta \mathcal{L}}{\delta \pi(q)} = -\log \pi(q) - 1 - \lambda = 0.$$

Solving: π(q) = exp(−1 − λ). The constant (independent of q), normalized by $\int_0^1 \pi(q) dq = 1$, gives π(q) = 1 for q ∈ [0,1].

Equivalently: **π(q) = 1 = Beta(1, 1)(q)**, since Beta(1,1) has density $\frac{\Gamma(2)}{\Gamma(1)\Gamma(1)} q^0 (1-q)^0 = 1$. $\square$

**Citation:** Jaynes 1957 §I-II for the MaxEnt principle. The specific application to Bernoulli parameters on [0,1] is standard; derived explicitly above.

---

## 6. Step S5 — Bayesian conjugate update for Beta-Bernoulli (Type 2)

**Claim.** Given a prior π(q) = Beta(α, β)(q) and one observation of "edge exists," the posterior is Beta(α + 1, β).

**Proof (Type 2 — explicit algebra).** The Beta(α, β) density is:

$$\pi(q; \alpha, \beta) = \frac{1}{B(\alpha, \beta)} q^{\alpha-1} (1-q)^{\beta-1}$$

where $B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$ is the normalizing beta function.

The likelihood of observing "edge exists" (Bernoulli success) given parameter q is:

$$P(\text{exists} | q) = q.$$

By Bayes' rule (standard probability axiom — Kolmogorov 1933 or derived from conditional probability definition):

$$\pi(q | \text{exists}) = \frac{P(\text{exists} | q) \pi(q)}{\int_0^1 P(\text{exists} | q') \pi(q') dq'}.$$

Substituting:

$$\pi(q | \text{exists}) = \frac{q \cdot \frac{1}{B(\alpha,\beta)} q^{\alpha-1}(1-q)^{\beta-1}}{\int_0^1 q' \cdot \frac{1}{B(\alpha,\beta)} q'^{\alpha-1}(1-q')^{\beta-1} dq'}$$

$$= \frac{q^{\alpha}(1-q)^{\beta-1}}{\int_0^1 q'^{\alpha}(1-q')^{\beta-1} dq'}$$

$$= \frac{q^{\alpha}(1-q)^{\beta-1}}{B(\alpha+1, \beta)}$$

$$= \text{Beta}(\alpha+1, \beta)(q). \quad \square$$

Symmetrically, observing "edge absent" gives posterior Beta(α, β+1).

---

## 7. Step S6 — predictive probability under Beta posterior (Type 2)

**Claim.** Under posterior π = Beta(α, β), the predictive probability of observing "edge exists" on the next observation is $\alpha/(\alpha+\beta)$.

**Proof (Type 2).** The predictive probability marginalizes q:

$$P(\text{exists} | \text{posterior}) = \int_0^1 P(\text{exists} | q) \pi(q; \alpha, \beta) dq = \int_0^1 q \cdot \frac{1}{B(\alpha,\beta)} q^{\alpha-1}(1-q)^{\beta-1} dq.$$

This is the first moment of the Beta distribution:

$$= \frac{B(\alpha+1, \beta)}{B(\alpha, \beta)} = \frac{\Gamma(\alpha+1)\Gamma(\beta)/\Gamma(\alpha+\beta+1)}{\Gamma(\alpha)\Gamma(\beta)/\Gamma(\alpha+\beta)}$$

Using $\Gamma(\alpha+1) = \alpha \Gamma(\alpha)$ and $\Gamma(\alpha+\beta+1) = (\alpha+\beta)\Gamma(\alpha+\beta)$:

$$= \frac{\alpha \Gamma(\alpha)\Gamma(\beta)/[(\alpha+\beta)\Gamma(\alpha+\beta)]}{\Gamma(\alpha)\Gamma(\beta)/\Gamma(\alpha+\beta)} = \frac{\alpha}{\alpha+\beta}. \quad \square$$

---

## 8. Step S7 — Shannon surprise (Type 3)

**Definition.** The surprise (self-information) of observation x under probability distribution P is:

$$S(x; P) := -\log_2 P(x).$$

Units: bits.

Citation: **Shannon 1948**, *A Mathematical Theory of Communication*, Bell Syst. Tech. J. 27, 379-423. Shannon's entropy $H(X) = -\sum_x P(x) \log_2 P(x)$ is the expectation of $S(x; P)$ over the distribution; Shannon's paper introduces and motivates this functional form (additivity for independent events, non-negativity, unit of bits for base-2 log). Modern textbook treatment making the per-event self-information definition explicit: **Cover-Thomas 2006**, *Elements of Information Theory* 2nd ed., §2.1. Both are canonically cited in the framework.

---

## 9. S_fresh = 1 bit (Type 2)

**Claim (S_fresh).** For any pair (X, Y) with no prior observations, the surprise of observing either "edge exists" or "edge absent" is exactly 1 bit.

**Proof.** By step S4 (§5), the observer's prior on $q_{XY}$ is Beta(1,1). By step S6 (§7), the predictive probability:

$$P(\text{exists}) = \frac{\alpha}{\alpha+\beta} = \frac{1}{1+1} = \frac{1}{2}.$$

Symmetrically, $P(\text{absent}) = 1/2$.

By step S7 (§8, Shannon 1948):

$$S(\text{exists}; \text{Beta}(1,1)) = -\log_2 \frac{1}{2} = 1 \text{ bit}.$$

$$S(\text{absent}; \text{Beta}(1,1)) = -\log_2 \frac{1}{2} = 1 \text{ bit}.$$

Both outcomes carry exactly 1 bit of surprise. **S_fresh = 1 bit.** $\square$

---

## 10. S_disconfirm = log₂(3) bits (Type 2)

**Claim (S_disconfirm).** For a pair (X, Y) after exactly one prior observation of "edge exists" (and no prior observations of "absent"), the surprise of observing "edge absent" is exactly log₂(3) bits.

**Proof.** By step S4 (§5), the initial prior is Beta(1,1). By step S5 (§6), after one observation of "exists," the posterior is Beta(2, 1).

By step S6 (§7), the predictive probability under Beta(2, 1):

$$P(\text{exists}) = \frac{2}{2+1} = \frac{2}{3}.$$

$$P(\text{absent}) = 1 - \frac{2}{3} = \frac{1}{3}.$$

By step S7 (§8, Shannon 1948):

$$S(\text{absent}; \text{Beta}(2,1)) = -\log_2 \frac{1}{3} = \log_2 3 \approx 1.585 \text{ bits}.$$

**S_disconfirm = log₂(3) bits.** $\square$

---

## 11. Asymmetry (Type 2)

**Claim.** S_disconfirm − S_fresh = log₂(3/2) ≈ 0.585 bits > 0.

**Proof.** Arithmetic:

$$S_{\text{disconfirm}} - S_{\text{fresh}} = \log_2 3 - 1 = \log_2 3 - \log_2 2 = \log_2 \frac{3}{2}. \quad \square$$

---

## 12. Parameter_linter gate summary

| Step | Claim | Gate type | Source |
|---|---|---|---|
| S1 | Toggle events are per-pair binary self-inverse | Type 1 | A1 |
| S2 | Observer maintains probability model under A2 | Type 2 | L_total requires probability; derivation in §3 |
| S3 | Per-pair Bernoulli is MDL-minimal | Type 2 | Derivation in §4 (MDL minimization comparing model classes) |
| S4 | Prior is uniform Beta(1,1) | Type 3 + Type 2 | Jaynes 1957 §I-II for principle; derivation in §5 for specific application |
| S5 | Beta conjugate update | Type 2 | Derivation in §6 from Bayes' rule |
| S6 | Beta predictive mean = α/(α+β) | Type 2 | Derivation in §7 from Beta integral |
| S7 | Surprise = −log₂ P | Type 3 | Shannon 1948 §I (definitional) |
| S8 | S_fresh = 1 bit | Type 2 | §9 arithmetic |
| S9 | S_disconfirm = log₂(3) bits | Type 2 | §10 arithmetic |
| S10 | Asymmetry > 0 | Type 2 | §11 arithmetic |

**Verdict: all steps gate-passing.** Two external citations (Jaynes 1957, Shannon 1948), both canonical and already load-bearing in the framework.

---

## 13. What this theorem closes

- **Specific surprise values.** S_fresh = 1 bit; S_disconfirm = log₂(3) bits. Numerical values are exact under the derivation.
- **Asymmetry.** Confirmation-then-disconfirmation costs strictly more surprise than a fresh observation. This is a per-observation structural fact about the observer's Bayesian model.
- **Stage 2 of axiom-elimination roadmap, partial.** The numerical content that my prior attempt ( an internal working note ) tried to establish via graph-MDL is now established via Bayesian surprise with a clean derivation.

---

## 14. What this theorem does NOT close

- **Arrow of time.** The asymmetry is per-observation; whether it integrates to a monotonically growing observer model is a SEPARATE claim requiring analysis of the observer's trajectory under iid observations.
- **Energy functional.** Landauer + Sagawa-Ueda linkage requires the arrow-of-time result first.
- **Steady-state existence.** Not claimed.
- **Acceptance/rejection threshold interpretation.** The toggle_paper.md treats θ_create and θ_persist as decision thresholds for accepting/rejecting toggle observations. This theorem makes no such decision claim — it computes surprise values only. Whether the observer uses these values as acceptance thresholds is a separate modeling step.
- **Connection to srs structure or branch measure μ.** This theorem is at the level of individual pair-state observations, not substrate-embedded structure. The link to `theorem_multiway_branch_measure.md` (session-9 theorem) requires additional content.

---

## 15. Relationship to axiom-elimination roadmap

- **Stage 2a (this theorem):** edge-surprise thresholds. **CLOSED at theorem grade.**
- **Stage 2b (next):** arrow of time from iid observation integration. Requires proving the observer's graph-state trajectory under iid Bernoulli events is monotonic in expectation. New theorem target.
- **Stage 2c:** energy functional. Requires Stage 2b + Landauer (A-IT3) + Sagawa-Ueda (A-IT7) as load-bearing.
- **Stage 3+ (Lorentz, A4 elimination, etc.):** unchanged, prerequisite is Stage 2c.

---

## 16. Honesty section

**What was discovered mid-derivation that changed the scope:**
- The original retraction-cost scoping aimed at a specific per-edge cost number (2.77 bits per edge, with a 1.65-bit parametric-complexity content). That derivation was unsound.
- The correct narrow result is two surprise VALUES (1 bit, log₂(3) bits) with an ASYMMETRY (log₂(3/2) bits).
- The arrow-of-time and energy-functional corollaries previously claimed as immediate are actually additional theorem targets requiring more work.

**What was NOT done:**
- NOT treated toggle_paper.md as upstream. The derivation here is independent, using only framework axioms + canonical Jaynes 1957 and Shannon 1948 citations.
- NOT fabricated citations. Where specific textbook references couldn't be verified, the result was derived inline as Type 2 algebra instead.
- NOT fit to numerical predictions. The values 1 bit and log₂(3) emerge from the derivation; no target value was retrofit.

**Assumption I am explicit about:**
- §4's "iid toggle events at pair level" is a commitment that deserves its own scrutiny. Under A1, toggles occur; under A2 refined, the observer compresses them. Whether these events are iid at the pair level depends on whether the substrate has cross-pair correlations. For the session-9 branch measure μ (uniform product measure), cross-pair independence holds at the toggle-alphabet level, but once the observer's model assigns probabilities to COMPOSITE STATES of multiple pairs, the observations at different pairs become marginally correlated through shared model-parameter uncertainty.

  For this theorem, I use the simplification "each pair's Bernoulli is treated independently" — valid under the MDL-minimal model class derivation in §4. A more refined treatment would allow joint-posterior models at additional L(model) cost; MDL rejects them for iid data.

---

## 17. References

### Framework axioms
- `../framework/framework_axioms.md` §2 (A1); `theorem_A2_mdl_from_finite_register.md` (A2-T derived theorem).

### Framework docs
- `theorem_multiway_branch_measure.md` — session-9 theorem on branch measure μ (related but independent of this theorem).

### Cited published theorems (Type 3)
- **Jaynes, E.T.** (1957). *Information theory and statistical mechanics.* Phys. Rev. 106, 620-630. §I-II (MaxEnt principle).
- **Shannon, C.E.** (1948). *A mathematical theory of communication.* Bell Syst. Tech. J. 27, 379-423. §I (self-information / surprise definition).

### Framework memory

### NOT cited (available but not used)
- External sister-project draft on toggle dynamics — read for orientation but NOT used as upstream, per discipline "derive from ground up." Independent rederivation in this doc matches its numerical values (S_fresh = 1, S_disconfirm = log₂(3)) as external validation.

---

## 18. Status

**THEOREM (rigor: closed under parameter_linter.md hard gate).** Every load-bearing step annotated with its gate type and either explicit algebra or canonical citation. No fabricated citations; no post-hoc fitting; no scope creep. Stage 2a of axiom-elimination roadmap complete.

**Next targets:**
- Stage 2b (arrow of time from this asymmetry). Separate theorem.
- Stage 2c (energy functional). Separate theorem, prerequisite Stage 2b.

**Recommended next session work:** write Checkpoint A+B for Stage 2b before any theorem content, continuing gate-first methodology.
