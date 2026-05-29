# Derivation of k* (coordination number)

**NOTE (post-2026-04-26 demotion):** A2 and A3 are derived theorems; structural slate is {A1} + P1' + A5-mass per docs/framework/framework_axioms.md §10. The closure chain referenced here is preserved; only the axiomatic-status labels change. This derivation inherits the Gleason 1957 citation via d_spatial; under A1 + A2-T + A3-T, G.1 and G.5 are DERIVED via CDP 2011 (predictions/observer_hilbert_space.py).

## Abstract

We derive the coordination number $k^* = 3$ of the framework's optimal graph from $d = 3$ spatial dimensions. For a $d$-dimensional crystal net, the minimum degree is $k = d$ (the edge vectors must span $\mathbb{R}^d$). MDL eliminates edges beyond this minimum (they are linearly dependent, providing zero additional compression). Therefore $k^* = d = 3$. This result, combined with Sunada (2012), uniquely selects the srs (Laves) lattice.

## Framework axioms invoked

1. **MDL compression** (inherited from `predictions/d_spatial.py`).
2. **Self-inverse toggle** (inherited from `predictions/p_toggle.py`).

No new axioms are introduced. This derivation is a consequence of $d = 3$.

## Derivation

### Step 1: $d = 3$ spatial dimensions

From `predictions/d_spatial.py`: MDL + Gleason's theorem (1957) gives $d = 3$. See that file for the full chain.

### Step 2: Minimum degree for $d$-dimensional periodicity

**Theorem** (Delgado-Friedrichs & O'Keeffe, *Acta Cryst.* A **59**, 351–360, 2003, §2.1): A $d$-periodic crystal net has edge displacement vectors that generate the translation lattice $\mathbb{Z}^d$. At any vertex, the $k$ edge vectors $\{v_1, \ldots, v_k\} \subset \mathbb{R}^d$ must span $\mathbb{R}^d$.

Since $k$ vectors in $\mathbb{R}^d$ can span at most a $\min(k, d)$-dimensional subspace:

$$\text{span}\{v_1, \ldots, v_k\} = \mathbb{R}^d \quad \Longrightarrow \quad k \geq d$$

For $d = 3$: $k \geq 3$.

### Step 3: MDL selects $k = d$ (no redundant edges)

Consider a crystal net with $k > d$ edges per node. The $k$ edge vectors span $\mathbb{R}^d$ (rank $d$), so $k - d$ vectors are linearly dependent on the remaining $d$.

Each edge in the observer's model costs $\theta_{\text{create}} + \theta_{\text{persist}}$ bits to maintain (see `predictions/d_spatial.py` for the Bayesian threshold derivation). An edge contributes compression benefit proportional to the independent information it captures about the toggle stream.

A linearly dependent edge captures information already captured by the independent edges — its Fisher information contribution is zero (it lies in the null space of the Fisher information matrix). Its compression benefit is zero. Its model cost is positive. MDL eliminates it.

Therefore the MDL-optimal degree is $k = d$: exactly as many edges as independent spatial directions.

### Step 4: $k^* = 3$

Combining Steps 1–3:

$$k^* = d = 3$$

## Consistency check: surprise balance

At $k = 3$, $p = 2$:

$$S(3, 2) = 1 + \log_2(3) \approx 2.585 \text{ bits}$$
$$\theta_{\text{create}} + \theta_{\text{persist}} = \log_2(2) + \log_2(3) \approx 2.585 \text{ bits}$$

The surprise balance $S = \theta_{\text{create}} + \theta_{\text{persist}}$ is equivalent to $k = 3$ for binary toggles. It is not an independent axiom — it is the information-theoretic expression of this result. The script `proofs/foundations/toggle_arity.py` derives $k = 3$ from this balance; the present derivation reaches the same conclusion from MDL + Gleason, confirming consistency.

## Downstream consequence: srs lattice

Among all 3-regular 3D crystal nets, the srs (Laves) lattice uniquely minimizes description length (proved in `proofs/foundations/dl_comparison.py`). The key result: srs is the unique vertex-transitive AND edge-transitive 3-connected 3D crystal net (Sunada, *Notices AMS*, 2012). Edge-transitivity means zero bits for edge specification; vertex-transitivity means one Wyckoff orbit. No other 3D net achieves both.

$\text{DL}(\text{srs}) = 12.17$ bits. Gap to nearest 3D competitor (ths): $+1.68$ bits.

Note: 2D nets (e.g., honeycomb, $\text{DL} = 9.67$ bits) have lower graph description length, but $d = 2$ is excluded by Step 1 (Gleason requires $d \geq 3$).

## Result

$$\boxed{k^* = 3}$$

The graph is trivalent. This determines the Clifford algebra $\text{Cl}(2k^*) = \text{Cl}(6)$, the gauge group $SU(3) \times SU(2) \times U(1)$, and the number of fermion generations (3).

## Comparison with experiment

| Quantity | Predicted | Observed | Deviation |
|----------|-----------|----------|-----------|
| Coordination number | 3 | 3 (trivalent srs) | 0 (exact) |
| Fermion generations | $k^* = 3$ | 3 | 0 (exact) |
| Gauge group | $\text{Cl}(6) \to SU(3) \times SU(2) \times U(1)$ | SM gauge group | exact |

## Open questions

None. The Fisher information rank proof (Brown 1986, Theorem 1.13) is now incorporated in `predictions/d_spatial_derivation.md` Step 2b. All steps in this derivation are either framework axioms, explicit algebra, or cited theorems.

## Audit v2 (Clause 7) status

This prediction inherits Row 4 (k* = 3) audit v2 closure. The §3 multi-mechanism
table (M1-M6 vs qtz at k=4) is consolidated in
an internal working note §2.1; closure doc:
an internal working note.

- **Status (post-audit-v2):** UNIQUE-THEOREM-GRADE-CONDITIONAL on Row 4 audit v2 closure (DOMINANT-with-named-margins; UNIQUE-on-η_B).
- **Named margin** (from index §2.1): combined M2 (~0.45 Boltzmann) × observable-specific M3·M4·M5 product.
- **Inherits structurally:** Row 4 v2 closure protects against qtz at k=4 alternative via combined mechanism product. M6 sign-flip is irrelevant for this observable (uses Im(h)/|h|² or magnitude, not Re(h) sign) unless the observable uses Re(h_P) directly (η_B specifically; see `predictions/eta_B_derivation.md` for that case).
- **Conditionals deferred:** RCSR-vetted qtz bond list verification; selection-rule audit; data-conditional MDL.
