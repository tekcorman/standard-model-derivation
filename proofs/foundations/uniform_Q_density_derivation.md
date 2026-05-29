# Derivation: Uniform Q-space spectral density at MDL optimum (Part A)

**Status:** Theorem (Part A, density shape).  Part B (Feshbach coupling magnitude)
is ADOPTED/open -- NOT included here; see `predictions/Feshbach_coupling_strength.py`.
**Verification:** `predictions/uniform_Q_density.py` (all assertions pass).
**Upstream:** `predictions/k_star.py` (k* = 3), `../predictions/walker_dynamics_derivation.md`.

## Theorem Statement (Part A only)

**Theorem A.** At MDL optimum, the Q-space (ruliad complement) spectral density
rho_Q(phi) is uniform on the Ramanujan circle |lambda|^2 = k-1:

    rho_Q(phi) = 1/(2*pi) + O(sqrt(log(N)/N))

in total-variation distance, where N is the observer's sample size.
For a cosmological observer with N ~ 10^60, the remainder is below 10^{-29}.

## Proof of Theorem A

### Setup

Let B_hat be an MDL-optimal compressor.  Parameterize the Q-space spectrum by angle
phi on the Ramanujan circle |lambda|^2 = k-1.  Consider two competing models:

- **Model B**: current MDL-optimal observer; Q-space is an unstructured bulk.
- **Model B'**: B augmented with one parameter (angle phi_0) absorbing a putative
  Q-space peak of amplitude eps and support Delta_phi.

### MDL Code-Length Comparison

By Rissanen's two-part MDL (Rissanen 1978; Grunwald 2007 §5.3):

    L_{B'}(N) = -log P_{B'}(data | theta_hat) + (d_B + 1)/2 * log(N) + O(1)

Subtracting L_B:

    Delta_L = -(likelihood benefit) + (1/2)*log(N) + O(1)

The likelihood benefit for a peak of amplitude eps on support Delta_phi is
(by Pinsker/chi^2 expansion, Cover & Thomas 2006 Lemma 17.3.2):

    log P_{B'} - log P_B = N * D_KL(rho_hat || rho_{Q,B}) = N * eps^2 * Delta_phi / 2 + O(eps^3)

Combining:

    Delta_L = -N/2 * eps^2 * Delta_phi + (1/2)*log(N) + O(1)

### MDL Optimality Condition

At MDL optimum, Delta_L >= 0 for every (phi_0, Delta_phi):

    eps(phi_0, Delta_phi)^2 * Delta_phi <= log(N)/N

Taking Delta_phi = Delta_phi_min = O(1) (positive constant, angular resolution):

    |rho_Q(phi) - 1/(2*pi)| <= sqrt(log(N) / (N * Delta_phi_min)) = O(sqrt(log(N)/N))

Integrating gives the total-variation bound.  **QED.**

## Remarks

1. **Sharpening from previous argument:** The prior in-session closure (§4c.3 of
   dark_correction_theorem) used "cost O(1), benefit O(eps^2 * width)."
   This is sharpened: cost = (1/2)*log(N) exactly (Rissanen 1978), benefit uses
   Pinsker/chi^2 expansion (Cover & Thomas 2006 Lemma 17.3.2), and the threshold
   scales as sqrt(log(N)/N) (not 1/sqrt(N); Grunwald 2007 §14.3 Theorem 14.3).

2. **Why not Kesten-McKay:** The Kesten-McKay density rho_KM(lambda) describes the
   adjacency spectrum of the universal covering tree -- a different object from the
   Q-space MDL residual.  They do not compete for this role.

3. **Cosmological N:** For N ~ 10^60, sqrt(log(N)/N) ~ sqrt(138/10^60) ~ 10^{-29},
   far below any physical-observable precision.

## Part B (NOT in predictions/)

**Claim B (ADOPTED):** The P*H*Q Feshbach coupling strength equals alpha_1 = (2/3)^(g-2).

Part B is separately addressed in `predictions/Feshbach_coupling_strength.py` with the
following split:
- Lemma 1 (tree NB walk survival): theorem-grade.
- I-Feshbach identification: ADOPTED under the Exponent Principle.
- Precise gap: eigenspace projectors commute with B, so finite K_4 matrix computation
  cannot close I-Feshbach (algebraic impossibility, confirmed numerically).

## Consequences for P5 Postulate

The P5 postulate (W4_identification_catalog.md §3) bundles two sub-parts:
- "Q-space has uniform spectral density on Ramanujan circle" -- **now Theorem A**.
- "P*H*Q Feshbach coupling has strength alpha_1" -- **Claim B (ADOPTED)**.

This is a partial reduction: the density-shape part of P5 is promoted to theorem.


## Audit v2 (Clause 7) status

This prediction inherits Row 4 (k* = 3) audit v2 closure. The §3 multi-mechanism
table (M1-M6 vs qtz at k=4) is consolidated in
an internal working note §2.1; closure doc:
an internal working note.

- **Status (post-audit-v2):** UNIQUE-THEOREM-GRADE-CONDITIONAL on Row 4 audit v2 closure (DOMINANT-with-named-margins; UNIQUE-on-η_B).
- **Named margin** (from index §2.1): combined M2 (~0.45 Boltzmann) × observable-specific M3·M4·M5 product.
- **Inherits structurally:** Row 4 v2 closure protects against qtz at k=4 alternative via combined mechanism product. M6 sign-flip is irrelevant for this observable (uses Im(h)/|h|² or magnitude, not Re(h) sign) unless the observable uses Re(h_P) directly (η_B specifically; see `predictions/eta_B_derivation.md` for that case).
- **Conditionals deferred:** RCSR-vetted qtz bond list verification; selection-rule audit; data-conditional MDL.

## References

- Cover, T.M. & Thomas, J.A. (2006). Elements of Information Theory, 2nd ed. Wiley.
  Lemma 17.3.2 (Pinsker/chi^2 expansion around uniform distribution).
- Grunwald, P. (2007). The Minimum Description Length Principle. MIT Press.
  §5.3 (two-part MDL code length), §7.1 (stochastic complexity),
  §14.3 Theorem 14.3 (MDL model selection consistency).
- Rissanen, J. (1978). Modeling by shortest data description. Automatica 14, 465-471.
- Rissanen, J. (1983). A universal prior for integers and estimation by MDL.
  Ann. Statist. 11, 416-431.
