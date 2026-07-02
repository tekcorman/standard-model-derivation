# Fresh-observation surprise (S_fresh)

## Abstract

The Shannon self-information of the first observation of an srs edge pair's existence state, under the observer's MDL-optimal Jaynes-MaxEnt Beta(1,1) prior, equals 1 bit exactly. This is the minimum cost of novelty: the observer's uniform prior is maximally uncertain, so any first observation carries exactly 1 bit of information. Framework-internal information quantity; cited as upstream by the 2-state Markov chain that sets p_create = 1/2 = 2^(−S_fresh).

**Result:** S_fresh = 1 bit exactly.
**Grade:** THEOREM under A1 + A2-T + Jaynes 1957 + Shannon 1948.

## Framework axioms invoked

- **A1**: toggle alphabet.
- **A2-T (waterline thm)**: MDL observer (derived theorem; see `docs/theorems/theorem_A2_mdl_from_finite_register.md`).

## Derivation

### Step 1 — Jaynes MaxEnt prior (Type 3)

Jaynes 1957 §II: the maximum-entropy prior on a Bernoulli parameter q ∈ [0, 1] with no constraint other than normalization is uniform on [0, 1], equivalent to Beta(α = 1, β = 1). See `docs/theorems/theorem_edge_surprise_thresholds.md` §5 for the explicit Lagrangian derivation.

### Step 2 — Predictive probability (Type 2)

Under prior Beta(α, β), the predictive probability of observing "exists" on the next observation is the Beta mean:

$$P(\text{exists} \mid \text{Beta}(\alpha, \beta)) = \frac{\alpha}{\alpha + \beta}.$$

For α = β = 1: P(exists) = 1/2. Symmetrically, P(absent) = 1/2.

### Step 3 — Shannon surprise (Type 3)

Shannon 1948: self-information of an event with probability P is S = −log₂ P. Applied to either outcome:

$$S_{\text{fresh}} = -\log_2 \frac{1}{2} = 1 \text{ bit.}$$

### Result

$$\boxed{S_{\text{fresh}} = 1 \text{ bit exactly.}}$$

## Comparison with experiment

Not directly observed — this is an information-theoretic quantity on the observer's internal state. Its downstream observational consequences propagate through the 2-state Markov chain (which has p_create = 2^(−S_fresh) = 1/2) to the toggle rate λ = 2/5 and temporal correlation length ξ_t = 1/log 6, and from there through Stage 3 to the dimension-6 Lorentz violation coefficient η_lattice = 1/12 and scale energy ~147 PeV.

## Open questions

- The IDENTIFICATION of the Shannon surprise with a Markov-chain transition probability (p_create = 2^(−S_fresh)) is standard under Bayesian-Markov duality but not derived here. Stage 2a + Stage 3 treat it as a modeling commitment consistent with the observer's MDL minimization.

## References

### Framework
- `docs/theorems/theorem_edge_surprise_thresholds.md` (Stage 2a, full derivation).

### Published
- **Jaynes, E.T.** (1957). *Information theory and statistical mechanics.* Phys. Rev. 106, 620-630. §II (MaxEnt).
- **Shannon, C.E.** (1948). *A Mathematical Theory of Communication.* Bell Syst. Tech. J. 27, 379-423. §I.
