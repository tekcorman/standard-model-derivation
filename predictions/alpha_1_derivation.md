# Derivation of alpha_1 (bare NB walk survival probability)

**Audit anchor:** Row P1 of `docs/parameters/parameter_uniqueness_ledger.md`. UNIQUE within "NB walks on k\*-regular graphs with branch measure μ", conditional on Row 4 (k\* = 3), Row 9 (g = 10), Row 12 (branch measure μ uniform per-step) of `docs/audits/registers/uniqueness_ledger.md`. The dressed coupling α_1_full = α_1/(1−α_1) (Row P2) follows by A2-T waterline geometric resummation.

## Abstract

We derive the bare coupling constant $\alpha_1 = (2/3)^8 = 256/6561 \approx 0.03902$ as the survival probability of a non-backtracking walk of length $g - 2 = 8$ on the srs lattice. At each step, the walker has $(k^* - 1)/k^* = 2/3$ probability of continuing forward. After $g - 2$ steps, the accumulated survival is $(2/3)^8$. The derivation is combinatorics on $k$-regular graphs — no assumptions beyond the upstream values of $k^*$ and $g$.

## Framework axioms invoked

None beyond those inherited from upstream:
- $k^* = 3$ (from `predictions/k_star.py`)
- $g = 10$ (from `predictions/g_girth.py`)

## Derivation

### Step 1: Per-step survival probability

On a $k^*$-regular graph, a non-backtracking walker at vertex $v$ arrived via one specific edge. Of the $k^*$ edges incident to $v$:
- 1 edge leads back to the previous vertex (forbidden by the NB constraint)
- $k^* - 1$ edges are forward choices

The per-step survival probability (probability of continuing the walk rather than being forced to backtrack) is:

$$p_{\text{step}} = \frac{k^* - 1}{k^*} \tag{1}$$

For $k^* = 3$: $p_{\text{step}} = 2/3$.

This is a combinatorial fact about $k$-regular graphs. The NB walker at a vertex of degree $k$ has exactly $k - 1$ forward choices out of $k$ total edges. No citation needed — it follows from the definition of non-backtracking walk.

### Step 2: Walk length

The girth $g$ is the length of the shortest cycle on the graph. A non-backtracking cycle of length $g$ visits $g$ vertices and traverses $g$ edges.

The NB constraint applies at the $g - 2$ **intermediate** vertices (vertices 2 through $g - 1$ in the cycle):
- At vertex 1 (start): no arrival edge, so no NB constraint. The walker chooses freely.
- At vertices 2 through $g - 1$: the walker arrived from the previous vertex and must choose a forward edge. The NB constraint applies.
- At vertex $g$ (return to start): the walker must close the cycle by returning to vertex 1. This is a forced step, not a free choice.

Therefore the effective walk length (number of NB-constrained steps) is:

$$\ell = g - 2 \tag{2}$$

For $g = 10$: $\ell = 8$.

This counting convention follows from the definition of NB walks on graphs (see Terras, *Zeta Functions of Graphs*, Cambridge University Press, 2011, Chapter 1 for NB walk definitions and conventions on regular graphs).

### Step 3: Total survival probability

The accumulated survival probability over $\ell$ independent NB steps:

$$\alpha_1 = p_{\text{step}}^{\,\ell} = \left(\frac{k^* - 1}{k^*}\right)^{g - 2} \tag{3}$$

Substituting $k^* = 3$, $g = 10$:

$$\alpha_1 = \left(\frac{2}{3}\right)^8 = \frac{256}{6561} \approx 0.039018442310623 \tag{4}$$

This is an exact rational number. Each step is arithmetic.

## Result

$$\boxed{\alpha_1 = \left(\frac{k^* - 1}{k^*}\right)^{g-2} = \left(\frac{2}{3}\right)^8 = \frac{256}{6561}}$$

## Comparison with experiment

$\alpha_1$ is not directly measured. It is the bare walk amplitude that enters downstream observables:

| Downstream parameter | How $\alpha_1$ enters |
|---------------------|----------------------|
| $V_{cb}$ | Bare amplitude: $V_{cb} = \alpha_1(1 + \alpha_1)$ |
| $\theta_{23}$ (PMNS) | Dark correction: $(1 + \alpha_1^{\text{full}})/(1 - \alpha_1^{\text{full}})$ where $\alpha_1^{\text{full}} = (5/3)\alpha_1$ |
| $m_\tau$ | Yukawa coupling: $y_\tau = \alpha_1^{\text{full}}/k^{*2}$ |
| Higgs quartic $\lambda$ | $\lambda = 2\alpha_1^{\text{full}}$ |

Verification is indirect, through the accuracy of these predictions.

## Notation: bare vs full

This file defines $\alpha_1 = \alpha_1^{\text{bare}} = (2/3)^8$. Some framework scripts use $\alpha_1$ to mean:

$$\alpha_1^{\text{full}} = \frac{5}{3} \cdot \alpha_1^{\text{bare}} = \tan^2(\arg h) \cdot (2/3)^8 \approx 0.06504$$

where $h = (\sqrt{3} + i\sqrt{5})/2$ is the Hashimoto eigenvalue. The factor $5/3 = \text{Im}(h)^2/\text{Re}(h)^2$ is the mass²-class chirality enhancement. Both notations coexist in the codebase; this file always refers to the **bare** value.

## Open questions

None. Every step is combinatorics or arithmetic on values derived upstream.

## Audit v2 (Clause 7) status

This prediction inherits Row 4 (k* = 3) audit v2 closure. The §3 multi-mechanism
table (M1-M6 vs qtz at k=4) is consolidated in
an internal working note §2.1; closure doc:
an internal working note.

- **Status (post-audit-v2):** UNIQUE-THEOREM-GRADE-CONDITIONAL on Row 4 audit v2 closure (DOMINANT-with-named-margins; UNIQUE-on-η_B).
- **Named margin** (from index §2.1): combined M2 (~0.45 Boltzmann) × observable-specific M3·M4·M5 product.
- **Inherits structurally:** Row 4 v2 closure protects against qtz at k=4 alternative via combined mechanism product. M6 sign-flip is irrelevant for this observable (uses Im(h)/|h|² or magnitude, not Re(h) sign) unless the observable uses Re(h_P) directly (η_B specifically; see `predictions/eta_B_derivation.md` for that case).
- **Conditionals deferred:** RCSR-vetted qtz bond list verification; selection-rule audit; data-conditional MDL.
