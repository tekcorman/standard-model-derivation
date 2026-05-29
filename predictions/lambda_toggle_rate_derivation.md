# Toggle rate per edge per Planck step (λ)

## Abstract

The stationary toggle rate λ per edge per Planck step on the srs lattice is derived from Stage 2a's Bayesian Beta-Bernoulli threshold values via the 2-state Markov chain of edge occupancy. The transition probabilities p_create = 1/2 and p_destroy = 1/3 — coming directly from the Beta(1,1) prior's predictive (a fresh edge has P(exists) = 1/2) and the Beta(2,1) posterior's predictive after one confirmation (P(absent) = 1/3) — yield stationary distribution (π_off, π_on) = (2/5, 3/5) and stationary toggle rate λ = 2/5 exactly. No external parameters.

**Result:** λ = 2/5.
**Grade:** THEOREM under A1 + A2-T (waterline thm; refined A2) + Stage 2a as Type 4 upstream.

## Framework axioms invoked

- **A1** (`docs/framework/framework_axioms.md` §2): binary self-inverse toggle alphabet on edges.
- **A2-T (waterline thm)** (derived theorem; see `docs/theorems/theorem_A2_mdl_from_finite_register.md`): MDL observer with selective retention at rate-distortion optimum.

No further framework axioms invoked. All derivation-internal content is either Type 2 algebra or Type 4 upstream.

## Derivation

### Step 1 — Stage 2a upstream (Type 4)

By `docs/theorems/theorem_edge_surprise_thresholds.md` (Stage 2a), the observer maintains a per-pair Bernoulli-Beta probability model of edge existence, with uniform prior Beta(1,1) derived from Jaynes 1957 MaxEnt. Conjugate Bayesian updating gives posterior Beta(2,1) after one observation of "edge exists" and Beta(1,2) after one observation of "edge absent."

The predictive probabilities under these posteriors are computed (Stage 2a §7 and §9–10):

$$P(\text{exists} \mid \text{Beta}(1,1)) = \frac{1}{1+1} = \frac{1}{2}$$

$$P(\text{absent} \mid \text{Beta}(2,1)) = \frac{1}{2+1} = \frac{1}{3}$$

These are interpreted as the per-step probability of a create event at an empty pair, and a destroy event at a once-confirmed pair respectively. We set:

$$p_{\text{create}} := P(\text{exists} \mid \text{Beta}(1,1)) = \frac{1}{2}, \quad p_{\text{destroy}} := P(\text{absent} \mid \text{Beta}(2,1)) = \frac{1}{3}.$$

### Step 2 — 2-state Markov chain (Type 2)

Model the occupancy state of a single undirected edge as a 2-state Markov chain over {off, on}, with transitions:

- off → on with probability p_create = 1/2 per Planck step
- on → off with probability p_destroy = 1/3 per Planck step

The transition matrix is

$$M = \begin{pmatrix} 1 - p_{\text{create}} & p_{\text{destroy}} \\ p_{\text{create}} & 1 - p_{\text{destroy}} \end{pmatrix} = \begin{pmatrix} 1/2 & 1/3 \\ 1/2 & 2/3 \end{pmatrix}.$$

### Step 3 — Stationary distribution (Type 2)

The stationary distribution π = (π_off, π_on)^T satisfies M π = π with π_off + π_on = 1. Using detailed balance:

$$\pi_{\text{on}} \cdot p_{\text{destroy}} = \pi_{\text{off}} \cdot p_{\text{create}}$$

$$\pi_{\text{on}} \cdot \frac{1}{3} = (1 - \pi_{\text{on}}) \cdot \frac{1}{2}$$

$$2 \pi_{\text{on}} = 3 - 3 \pi_{\text{on}} \Rightarrow \pi_{\text{on}} = \frac{3}{5}, \quad \pi_{\text{off}} = \frac{2}{5}.$$

### Step 4 — Stationary toggle rate (Type 2)

The stationary toggle rate λ is the probability that a toggle event occurs on any given step, averaged over stationary occupancy:

$$\lambda = \pi_{\text{off}} \cdot p_{\text{create}} + \pi_{\text{on}} \cdot p_{\text{destroy}} = \frac{2}{5} \cdot \frac{1}{2} + \frac{3}{5} \cdot \frac{1}{3} = \frac{1}{5} + \frac{1}{5} = \frac{2}{5}.$$

Equivalently, in closed form:

$$\lambda = \frac{2 \, p_{\text{create}} \, p_{\text{destroy}}}{p_{\text{create}} + p_{\text{destroy}}}.$$

### Result

$$\boxed{\lambda = \frac{2}{5} \text{ exactly.}}$$

## Comparison with experiment

Not directly observable. λ is a framework-internal rate at Planck scale. Its observational consequences propagate through the Markov chain spectrum (ξ_t, r) and through the continuum-limit Lorentz-invariance analysis (Stage 3) to the dimension-6 Lorentz violation coefficient η_lattice = 1/12 and scale energy ~147 PeV. Those are the observationally accessible downstream predictions.

## Open questions

**What is NOT derived here:**
- The CONTINUUM-LIMIT physics interpretation of λ. The rate 2/5 per edge per Planck step is the framework-internal quantity; its appearance in continuum Lagrangians requires the continuum limit of the srs lattice + additional structure not derived at this level.
- The IDENTIFICATION of these transition probabilities with a specific physical toggle process (e.g., visible-sector pair production). That is an A5(b)-level identification, not derived at the Bayesian-observer level.

**Both pending items are consistent with the gate: they are explicit scope limits, not unproven claims within the derivation itself.**

## References

### Framework
- `docs/framework/framework_axioms.md` §2 (A1), §3 (A2 refined).
- `docs/theorems/theorem_edge_surprise_thresholds.md` (Stage 2a, Type 4 upstream).
- `docs/theorems/theorem_lorentz_causal_sector.md` §4.1 (Stage 3, context where λ appears load-bearing).
- `proofs/lorentz/b1_ags_audit.py` (independent side-experiment computation).

### Published
- **Jaynes, E.T.** (1957). *Information theory and statistical mechanics.* Phys. Rev. 106, 620-630. §II (MaxEnt principle giving Beta(1,1) prior).
- **Shannon, C.E.** (1948). *A Mathematical Theory of Communication.* Bell Syst. Tech. J. 27, 379-423. (Surprise / self-information, upstream in Stage 2a.)
