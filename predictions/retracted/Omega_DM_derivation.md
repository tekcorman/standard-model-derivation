# Derivation of the dark matter density fraction Omega_DM

## Abstract

We derive the cold dark matter density fraction $\Omega_{\mathrm{DM}}$ today as a single step of elementary algebra combining:

1. the derived ratio $r := \Omega_{\mathrm{DM}}/\Omega_{\mathrm{m}} = 1 - P(k \leq k^{*} \mid \mathrm{Poisson}(2k^{*}))$, closed in `predictions/Omega_DM_over_Omega_m.py` from the MDL compression argument with $k^{*} = 3$; and

2. the cosmologically-measured baryon density $\Omega_{\mathrm{b}} = 0.0493 \pm 0.0006$, taken from Planck 2018 (arXiv:1807.06209) as recorded in the PDG snapshot. This is an **external input**: the framework does not currently derive $\Omega_{\mathrm{b}}$ from its two foundational axioms.

The closing identity

$$\Omega_{\mathrm{m}} = \Omega_{\mathrm{b}} + \Omega_{\mathrm{DM}} \quad\Longrightarrow\quad \Omega_{\mathrm{DM}} = \Omega_{\mathrm{b}}\,\frac{r}{1 - r}$$

is elementary algebra on the definitions of $\Omega_{\mathrm{b}}$, $\Omega_{\mathrm{DM}}$, and $\Omega_{\mathrm{m}}$ as density fractions with $\Omega_{\mathrm{m}} = \Omega_{\mathrm{b}} + \Omega_{\mathrm{DM}}$ by definition (matter = baryonic + dark matter in the flat-$\Lambda$CDM partition used by PDG).

Because one input is $[\mathrm{external}]$, the grade of this derivation is **mathematically complete**, not theorem-grade. The CSV line `Omega_DM,0.265,0.265,<1,theorem,...` in `results/parameters.csv` is aspirational on this point; see the Open Questions section.

## Framework axioms invoked

Inherited from upstream predictions files -- no new axioms introduced here:

- **(A1)** Binary self-inverse toggle (`predictions/p_toggle.py`).
- **(A2)** MDL compression (`predictions/d_spatial.py`, `predictions/k_star.py`).

## Derivation

### Step 1. Upstream: $k^{*} = 3$

From `predictions/k_star.py` and `predictions/d_spatial.py`, the MDL-optimal observer has coordination number $k^{*} = 3$. Closed upstream under (A1) + (A2).

*Gate clearance*: upstream `predictions/*.py` file.

### Step 2. Upstream: the ratio $r = \Omega_{\mathrm{DM}}/\Omega_{\mathrm{m}}$

From `predictions/Omega_DM_over_Omega_m.py`, the dark matter fraction of total matter is

$$r \;=\; 1 - P(k \leq k^{*} \mid \mathrm{Poisson}(2k^{*})) \;=\; 1 - e^{-2k^{*}}\sum_{j=0}^{k^{*}}\frac{(2k^{*})^{j}}{j!}.$$

At $k^{*} = 3$,

$$r \;=\; 1 - e^{-6}\bigl(1 + 6 + 18 + 36\bigr) \;=\; 1 - 61\,e^{-6} \;=\; 0.848796\ldots$$

This ratio is derived from the Poisson max-entropy argument for the raw toggle degree distribution (Jaynes 1957) together with the MDL cutoff at $k = k^{*}$. The derivation retains one interpretive step (the identification "mean raw degree = $2k^{*}$", from Clifford creation + annihilation symmetry); see the discussion in `proofs/cosmology/dm_hierarchy_derivation.py` Step 2 and the Remark in `predictions/Omega_DM_over_Omega_m.py`.

*Gate clearance*: upstream `predictions/*.py` file.

### Step 3. External: $\Omega_{\mathrm{b}}$ from BBN / PDG

The PDG 2024 snapshot (reproduced from Planck 2018 VI, arXiv:1807.06209) gives

$$\Omega_{\mathrm{b}} \;=\; 0.0493 \pm 0.0006 \qquad \bigl(\Omega_{\mathrm{b}} h^{2} = 0.02237 \pm 0.00015,\ h = 0.674 \pm 0.005\bigr).$$

This value is BBN-consistent: the primordial deuterium abundance constrains $\Omega_{\mathrm{b}} h^{2} = 0.0224 \pm 0.0003$ (independent of CMB; see Cooke, Pettini and Steidel 2018, ApJ 855:102, arXiv:1710.11129), in agreement with the Planck value.

**This is an external, measured input.** The framework does not currently contain a proof of $\Omega_{\mathrm{b}}$ from (A1) + (A2). Flagged explicitly in Open Questions below.

*Gate clearance*: **$[\mathrm{external}]$**; no upstream derivation in this repository.

### Step 4. Algebra: $\Omega_{\mathrm{DM}} = \Omega_{\mathrm{b}}\,r/(1-r)$

By the PDG / Planck definition of cosmological density parameters, matter splits into baryonic and cold-dark components with

$$\Omega_{\mathrm{m}} \;=\; \Omega_{\mathrm{b}} + \Omega_{\mathrm{DM}}.$$

Writing $r = \Omega_{\mathrm{DM}}/\Omega_{\mathrm{m}}$ and substituting,

$$\Omega_{\mathrm{DM}} \;=\; r\,\Omega_{\mathrm{m}} \;=\; r\,(\Omega_{\mathrm{b}} + \Omega_{\mathrm{DM}}) \quad\Longrightarrow\quad \Omega_{\mathrm{DM}}(1 - r) \;=\; r\,\Omega_{\mathrm{b}}$$

and therefore

$$\boxed{\;\Omega_{\mathrm{DM}} \;=\; \Omega_{\mathrm{b}}\,\frac{r}{1-r}\;}.$$

*Gate clearance*: explicit algebra.

## Result

Substituting $r = 1 - 61 e^{-6} = 0.848796$ and $\Omega_{\mathrm{b}} = 0.0493$ (external):

$$\Omega_{\mathrm{DM}} \;=\; 0.0493 \cdot \frac{0.848796}{0.151204} \;=\; 0.0493 \cdot 5.6136 \;=\; 0.27675.$$

Consistency check:

$$\Omega_{\mathrm{m}} \;=\; \Omega_{\mathrm{b}} + \Omega_{\mathrm{DM}} \;=\; 0.0493 + 0.27675 \;=\; 0.32605,$$

compared with the Planck 2018 direct fit $\Omega_{\mathrm{m}} = 0.315 \pm 0.007$ -- a $1.5\,\sigma$ tension driven by the ratio prediction $r = 0.8488$ being mildly above the observed $\Omega_{\mathrm{DM,obs}}/\Omega_{\mathrm{m,obs}} = 0.2645/0.315 = 0.840$.

## Comparison with experiment

| Quantity | Predicted | Observed (PDG 2024 / Planck 2018) | Deviation |
|---|---|---|---|
| $\Omega_{\mathrm{DM}}$ | $0.27675$ | $0.2645 \pm 0.0050$ | $+0.01225$ ($+2.4\,\sigma$) |
| $\Omega_{\mathrm{DM}}/\Omega_{\mathrm{m}}$ (upstream) | $0.84880$ | $0.8398 \pm 0.016$ | $+0.0090$ ($+0.6\,\sigma$) |

The $+2.4\,\sigma$ tension on $\Omega_{\mathrm{DM}}$ is smaller in relative terms than its appearance in $\sigma$ units suggests: the absolute deviation is $4.6\%$, and the $\sigma$-count is driven by the small Planck uncertainty $\sigma_{\mathrm{obs}} = 0.0050$. The tension propagates entirely from the $+0.0090$ absolute offset on the ratio $r$; the algebra of Step 4 introduces no additional error.

If instead the observed ratio $r_{\mathrm{obs}} = 0.8398$ is used in Step 4 together with $\Omega_{\mathrm{b}} = 0.0493$, the formula reproduces $\Omega_{\mathrm{DM}} = 0.0493 \cdot 0.8398 / 0.1602 = 0.2585$, within $1.2\,\sigma$ of the Planck $\Omega_{\mathrm{DM}}$ -- confirming the deviation is located in the ratio prediction, not the algebra.

## Open questions

1. **$\Omega_{\mathrm{b}}$ is external.** The framework has no derivation of the baryon density fraction from its current axiom slate (A1 + A2-T + A3-T, per `docs/framework/framework_axioms.md` §10). Closing this gap requires a full baryogenesis derivation: an argument that reproduces $\eta_{B} = n_{B}/n_{\gamma} \approx 6.1 \times 10^{-10}$ and the matter-radiation equality epoch, together with an equation of state that propagates $\eta_{B}$ to $\Omega_{\mathrm{b}}$ at $z = 0$. Partial work on $\eta_{B}$ is in `proofs/cosmology/eta_B_derivation.py`, but that file itself contains unresolved open steps. Until $\Omega_{\mathrm{b}}$ is closed, the grade of the present derivation is **mathematically complete**, not theorem.

2. **Grade marking in `results/parameters.csv`.** The row

   ```
   Omega_DM,0.265,0.265,<1,theorem,"From Omega_DM/Omega_m = 0.842 + Omega_b from BBN",dm_hierarchy_derivation.py,P3
   ```

   marks this parameter as `theorem`-grade. That is **aspirational** under the rigor bar: since $\Omega_{\mathrm{b}}$ is $[\mathrm{external}]$, the actual grade is `mathematically complete`. Propagate this correction at the next CSV rebuild.

3. **Residual soft step in the ratio.** The upstream ratio derivation (`predictions/Omega_DM_over_Omega_m.py`) contains one interpretive step -- the identification of $2k^{*} = 6$ Clifford generators with the mean raw degree of the toggle graph. This step is motivated (Cl$(2k^{*})$ is derived; the creation/annihilation symmetric raw dynamics is a natural hypothesis) but not a theorem. Closing it needs a proof that the raw toggle graph (prior to MDL compression) has mean degree exactly $2k^{*}$. See `proofs/cosmology/dm_hierarchy_derivation.py` Step 5 for the current honest assessment.

4. **Nothing else.** Steps 1, 2, and 4 of the present derivation pass the gate under their stated citations / upstream files / explicit algebra. The only admission is the one marked $[\mathrm{external}]$ in Step 3.

## References

- Cooke, R.J., Pettini, M. and Steidel, C.C. (2018). One percent determination of the primordial deuterium abundance. *ApJ* **855**, 102. arXiv:1710.11129.
- Jaynes, E.T. (1957). Information theory and statistical mechanics. *Phys. Rev.* **106**, 620.
- Planck Collaboration (2020). Planck 2018 results. VI. Cosmological parameters. *A and A* **641**, A6. arXiv:1807.06209.
- `predictions/Omega_DM_over_Omega_m.py` -- upstream ratio.
- `predictions/k_star.py`, `predictions/d_spatial.py` -- upstream axiomatic derivations of $k^{*} = 3$.
- `proofs/cosmology/dm_hierarchy_derivation.py` -- honest-assessment analysis of the Poisson$(2k^{*})$ argument.
