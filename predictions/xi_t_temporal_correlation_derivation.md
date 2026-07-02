# Temporal correlation length per srs edge (ξ_t)

## Abstract

The per-edge toggle Markov chain's connected autocorrelation function decays exponentially with characteristic length ξ_t = 1/log 6 ≈ 0.558 Planck units. This follows from the second eigenvalue r = 1/6 of the 2-state transition matrix whose entries (p_create = 1/2, p_destroy = 1/3) come from Stage 2a's Bayesian Beta-Bernoulli predictive probabilities. Framework-internal quantity; not directly observable, but load-bearing in Stage 3 as the scale below which same-edge connected correlations are exponentially suppressed.

**Result:** ξ_t = 1/log 6 ≈ 0.558 ℓ_P.
**Grade:** THEOREM under A1 + A2-T + Stage 2a Type 4 upstream.

## Framework axioms invoked

- **A1** (`docs/framework/framework_axioms.md` §2): toggle alphabet on edges.
- **A2-T (waterline thm)** (derived theorem; see `docs/theorems/theorem_A2_mdl_from_finite_register.md`): MDL observer.

## Derivation

### Step 1 — Stage 2a upstream (Type 4)

From `docs/theorems/theorem_edge_surprise_thresholds.md`: p_create = 1/2 (Beta(1,1) predictive), p_destroy = 1/3 (Beta(2,1) predictive).

### Step 2 — Transition matrix (Type 2)

The per-edge 2-state Markov chain has:

$$M = \begin{pmatrix} 1 - p_{\text{create}} & p_{\text{destroy}} \\ p_{\text{create}} & 1 - p_{\text{destroy}} \end{pmatrix} = \begin{pmatrix} 1/2 & 1/3 \\ 1/2 & 2/3 \end{pmatrix}.$$

### Step 3 — Second eigenvalue (Type 2)

For a 2×2 stochastic matrix, eigenvalues are 1 and tr(M) − 1:

$$\text{tr}(M) = \frac{1}{2} + \frac{2}{3} = \frac{7}{6}, \quad r = \text{tr}(M) - 1 = \frac{1}{6}.$$

Equivalently: r = 1 − p_create − p_destroy = 1 − 1/2 − 1/3 = 1/6.

### Step 4 — Autocorrelation decay (Type 2)

Standard Markov chain spectral theory: the connected autocorrelation of edge state at time separation s decays as r^s. The correlation length is:

$$\xi_t = \frac{1}{\log(1/r)} = \frac{1}{\log 6}.$$

### Result

$$\boxed{\xi_t = \frac{1}{\log 6} \approx 0.5581 \text{ Planck units}.}$$

## Comparison with experiment

Not directly observable — Planck-scale per-edge correlations are below any current experimental resolution by ~61 orders of magnitude in length. The downstream consequence in Stage 3 is that same-edge connected correlations at time separation s decay as (1/6)^s, making the toggle process effectively Markov at scales L ≫ ξ_t.

## Open questions

- The specific identification "Markov 2nd eigenvalue = decay rate of a physical observable correlation" is standard spectral theory; the specific observable that inherits this decay in the continuum limit is not derived here (would be part of a continuum-limit theorem).

## References

### Framework
- `docs/theorems/theorem_edge_surprise_thresholds.md` (Stage 2a upstream).
- `docs/theorems/theorem_lorentz_causal_sector.md` §4.2 (Stage 3 context).
- `proofs/lorentz/b1_ags_audit.py` (side-experiment computation).

### Published
- Levin, Peres, Wilmer (2009). *Markov Chains and Mixing Times.* AMS. Theorem 1.14 and Ch. 4 for general spectral theory of correlation decay.
