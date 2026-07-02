# Disconfirming-observation surprise (S_disconfirm)

## Abstract

The Shannon self-information of a disconfirming observation ("edge absent") on a pair with a single prior confirmation equals log₂(3) ≈ 1.585 bits exactly. This follows from the Beta(2, 1) Bayesian conjugate posterior, under which the predictive probability of "absent" is 1/3. The asymmetry S_disconfirm − S_fresh = log₂(3/2) > 0 is the microscopic source of the arrow of time (creation is cheaper than disconfirmation).

**Result:** S_disconfirm = log₂(3) ≈ 1.585 bits exactly.
**Grade:** THEOREM under A1 + A2-T + Jaynes 1957 + Shannon 1948.

## Framework axioms invoked

- **A1**: toggle alphabet on edges.
- **A2-T (waterline thm)**: MDL observer (derived theorem; see `docs/theorems/theorem_A2_mdl_from_finite_register.md`).

## Derivation

### Step 1 — Uniform Beta(1, 1) prior (Type 3)

Per S_fresh's derivation: Jaynes 1957 MaxEnt gives uniform prior Beta(1, 1).

### Step 2 — Conjugate Bayesian update (Type 2)

After observing "exists" once, the posterior is Beta(2, 1) by conjugacy:

$$\pi(q \mid \text{exists}) = \frac{P(\text{exists} \mid q) \pi(q)}{\int P(\text{exists} \mid q') \pi(q') dq'} = \frac{q \cdot 1}{1/2} = 2q = \text{Beta}(2, 1)(q).$$

Full derivation in `docs/theorems/theorem_edge_surprise_thresholds.md` §6.

### Step 3 — Predictive probability (Type 2)

Under Beta(α, β), the Beta mean gives the predictive probability. For Beta(2, 1):

$$P(\text{exists} \mid \text{Beta}(2, 1)) = \frac{\alpha}{\alpha + \beta} = \frac{2}{3}, \quad P(\text{absent}) = \frac{1}{3}.$$

### Step 4 — Shannon surprise of "absent" (Type 3)

Shannon 1948 §I:

$$S_{\text{disconfirm}} = -\log_2 P(\text{absent}) = -\log_2 \frac{1}{3} = \log_2 3.$$

### Step 5 — Asymmetry (Type 2)

$$S_{\text{disconfirm}} - S_{\text{fresh}} = \log_2 3 - 1 = \log_2 \frac{3}{2} \approx 0.585 \text{ bits} > 0.$$

This asymmetry is the microscopic source of the arrow of time: once a pair is confirmed, removing the observation costs more surprise than the original confirmation required.

### Result

$$\boxed{S_{\text{disconfirm}} = \log_2 3 \approx 1.585 \text{ bits exactly.}}$$

## Comparison with experiment

Not directly observed. Sets p_destroy = 2^(−S_disconfirm) = 1/3 in the 2-state Markov chain, which propagates through λ = 2/5, ξ_t = 1/log 6, η_lattice = 1/12, scale energy ~147 PeV.

## Open questions

- Identification of the Shannon-surprise asymmetry with the specific physical "arrow of time" in Stage 2c and Stage 3 is load-bearing but not explicitly connected here; Stage 2c makes the connection via Landauer scaling.

## References

### Framework
- `docs/theorems/theorem_edge_surprise_thresholds.md` (Stage 2a, full derivation).
- `docs/theorems/theorem_observer_energy_functional.md` (Stage 2c, arrow-of-time connection).
- `predictions/S_fresh.py` (sibling quantity).

### Published
- **Jaynes, E.T.** (1957). *Phys. Rev.* 106, 620-630.
- **Shannon, C.E.** (1948). *Bell Syst. Tech. J.* 27, 379-423.
